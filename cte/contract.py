"""
CTE contract — the shared-infrastructure agreement between altyapı oturumu,
HB uzman oturumu ve FC uzman oturumu.

Amaç: HB ve FC araştırması ayrı oturumlarda yürürken, ortak dataset /
feature / split / trigger / target tanımları sessizce kaymasın. Altyapı
oturumu contract'ı yazar; uzman oturumları her training çalışmasında
doğrular.

Contract içeriği:
  - dataset_path          — production dataset artifact path
  - dataset_schema_hash   — kolon adı + dtype listesinin sha256'sı
  - dataset_row_count     — sanity (schema eşit ama satır dramatik değişmişse şüphe)
  - dataset_mtime_iso     — file modification time (insan-okunur)
  - feature_list_version  — "FEATURES_V1" gibi etiket
  - feature_list_hash     — tuple contents sha256
  - split_version_hash    — SplitParams alanları sha256
  - target_primary        — CONFIG.label.primary_target
  - trigger_config_hash   — bar + firstness + compression + hb + fc param hash
  - contract_version      — altyapı oturumunun explicit bumped version (int)
  - written_at            — ISO-8601 timestamp
  - written_by            — session tag (altyapı / HB uzman / FC uzman)

CLI:
  python -m cte.contract write                # altyapı oturumu
  python -m cte.contract verify                # uzman oturumu
  python -m cte.contract show                  # diagnostic dump
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve()
_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_ROOT))


CONTRACT_PATH = Path("output/cte_contract.json")
CURRENT_CONTRACT_VERSION = 1  # altyapı oturumu bump eder


def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _dataset_schema_fingerprint(path: Path) -> dict[str, Any]:
    """Read dataset parquet schema without loading full data."""
    import pyarrow.parquet as pq

    md = pq.read_metadata(str(path))
    schema = pq.read_schema(str(path))
    cols = [(f.name, str(f.type)) for f in schema]
    schema_str = ";".join(f"{n}:{t}" for n, t in cols)
    return {
        "dataset_path": str(path),
        "dataset_row_count": int(md.num_rows),
        "dataset_column_count": int(md.num_columns),
        "dataset_schema_hash": _sha256(schema_str),
        "dataset_mtime_iso": datetime.fromtimestamp(
            path.stat().st_mtime, tz=timezone.utc
        ).isoformat(),
    }


def _feature_list_fingerprint() -> dict[str, Any]:
    from cte.features import FEATURES_V1

    lst = list(FEATURES_V1)
    return {
        "feature_list_version": "FEATURES_V1",
        "feature_list_count": len(lst),
        "feature_list_hash": _sha256(";".join(lst)),
    }


def _split_fingerprint() -> dict[str, Any]:
    from cte.config import CONFIG

    split = CONFIG.split
    parts = [f"train_start={split.train_start}", f"horizon={split.label_horizon_bars}"]
    for fs in split.folds:
        parts.append(
            f"{fs.name}:tr_end={fs.train_end};va={fs.val_start}>{fs.val_end};"
            f"qt={fs.test_start}>{fs.test_end}"
        )
    return {
        "split_version_hash": _sha256("|".join(parts)),
        "split_fold_count": len(split.folds),
    }


def _trigger_fingerprint() -> dict[str, Any]:
    """Emit three trigger hashes:

    - `trigger_config_hash`: full picture (legacy/top-level sanity)
    - `trigger_config_hb_hash`: HB line's semantic deps
      (compression + hb + firstness + bar + dryup). FC-geom changes don't touch it.
    - `trigger_config_fc_hash`: FC line's semantic deps
      (compression + fc + firstness + bar + dryup). HB-geom changes don't touch it.

    `verify_contract(line="hb")` only checks the HB hash + shared fields, so a
    bar-filter change scoped to FC (via line-specific params once introduced)
    doesn't break the HB uzman oturumu's verify, and vice versa.
    """
    from cte.config import CONFIG

    def _d(obj) -> str:
        return ";".join(f"{k}={v}" for k, v in sorted(asdict(obj).items()))

    shared = [_d(CONFIG.compression), _d(CONFIG.firstness),
              _d(CONFIG.bar), _d(CONFIG.dryup)]
    hb_parts = shared + [_d(CONFIG.hb)]
    fc_parts = shared + [_d(CONFIG.fc)]
    full_parts = shared + [_d(CONFIG.hb), _d(CONFIG.fc)]
    return {
        "trigger_config_hash":    _sha256("|".join(full_parts)),
        "trigger_config_hb_hash": _sha256("|".join(hb_parts)),
        "trigger_config_fc_hash": _sha256("|".join(fc_parts)),
    }


def _target_fingerprint() -> dict[str, Any]:
    from cte.config import CONFIG

    lp = CONFIG.label
    return {
        "target_primary": lp.primary_target,
        "target_early_horizons": list(lp.early_horizons),
        "target_runner_horizons": list(lp.runner_horizons),
        "target_runner_mfe_atr": lp.runner_mfe_atr,
        "target_runner_max_mae_atr": lp.runner_max_mae_atr,
        "target_runner_min_hold": lp.runner_min_hold_h,
        "target_spike_reject_ratio": lp.spike_reject_close_peak_ratio,
    }


def build_current_contract(dataset_path: str | Path) -> dict[str, Any]:
    """Compute the contract fingerprint from *current* repo state."""
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"dataset not found at {dataset_path} — build it first "
            f"(python -m cte.dataset)"
        )

    c: dict[str, Any] = {"contract_version": CURRENT_CONTRACT_VERSION}
    c.update(_dataset_schema_fingerprint(dataset_path))
    c.update(_feature_list_fingerprint())
    c.update(_split_fingerprint())
    c.update(_trigger_fingerprint())
    c.update(_target_fingerprint())
    return c


def write_contract(
    dataset_path: str | Path,
    written_by: str,
    out_path: str | Path = CONTRACT_PATH,
    bump_version: bool = False,
) -> dict[str, Any]:
    """Altyapı oturumu kullanır: mevcut state'i contract olarak kaydeder.

    bump_version=True: CURRENT_CONTRACT_VERSION'ı disk'teki eski'den +1
    olarak yazar. Bu uzman oturumlardaki verify() çağrılarını kırar —
    uzman oturumunun retrain'i bilinçli bir aksiyon haline gelir.
    """
    os.chdir(str(_ROOT))
    c = build_current_contract(dataset_path)

    out_path = Path(out_path)
    # Version bump logic
    if bump_version and out_path.exists():
        try:
            with open(out_path) as f:
                old = json.load(f)
            old_v = int(old.get("contract_version", 0))
            c["contract_version"] = old_v + 1
        except Exception:
            pass

    c["written_at"] = datetime.now(timezone.utc).isoformat()
    c["written_by"] = written_by

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(c, f, indent=2, sort_keys=True)
    print(f"[WRITE] {out_path}  contract_version={c['contract_version']}")
    return c


def load_contract(path: str | Path = CONTRACT_PATH) -> dict[str, Any] | None:
    path = Path(path)
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def diff_contracts(
    expected: dict[str, Any],
    actual: dict[str, Any],
    *,
    line: str | None = None,
) -> list[str]:
    """Return list of field diffs. Skips write-time metadata.

    If `line` is "hb" or "fc", the *other* line's trigger hash is excluded —
    so a HB uzman oturumu's verify doesn't break when altyapı yalnızca FC'nin
    trigger param'larında değişiklik yapar, ve vice versa.
    `trigger_config_hash` (full) is always checked for shared-state sanity
    UNLESS `line` is set, in which case it's skipped (because either
    line hash covers the line-relevant subset, and the full hash would fire
    on any out-of-line change by design).
    """
    skip = {"written_at", "written_by"}
    if line == "hb":
        skip |= {"trigger_config_fc_hash", "trigger_config_hash"}
    elif line == "fc":
        skip |= {"trigger_config_hb_hash", "trigger_config_hash"}

    keys = set(expected) | set(actual)
    diffs = []
    for k in sorted(keys):
        if k in skip:
            continue
        ev, av = expected.get(k), actual.get(k)
        if ev != av:
            diffs.append(f"  {k}: expected={ev!r}  actual={av!r}")
    return diffs


def verify_contract(
    dataset_path: str | Path,
    contract_path: str | Path = CONTRACT_PATH,
    raise_on_mismatch: bool = True,
    verbose: bool = True,
    *,
    line: str | None = None,
) -> bool:
    """Uzman oturumları her training run başında bunu çağırır.

    `line` ∈ {"hb", "fc", None}. Uzman oturumunda kendi line'ını geç — sadece
    ilgili trigger hash + tüm shared alanlar doğrulanır; diğer line'ın
    trigger hash'indeki değişiklik verify'ı bozmaz. `line=None` tüm alanları
    kontrol eder (altyapı oturumu için tam sanity).

    Returns True if OK. If mismatch and raise_on_mismatch=True → raises
    RuntimeError with a clear diff. If False → returns False and logs.
    """
    stored = load_contract(contract_path)
    if stored is None:
        msg = (
            f"❌ no contract at {contract_path}. Run:\n"
            f"     python -m cte.contract write --written-by 'altyapı oturumu'"
        )
        if raise_on_mismatch:
            raise RuntimeError(msg)
        if verbose:
            print(msg)
        return False

    current = build_current_contract(dataset_path)
    current["contract_version"] = stored.get("contract_version", CURRENT_CONTRACT_VERSION)
    diffs = diff_contracts(stored, current, line=line)

    if not diffs:
        if verbose:
            print(
                f"✔ contract OK  version={stored.get('contract_version')}  "
                f"dataset={stored.get('dataset_path')}  "
                f"features={stored.get('feature_list_version')}"
            )
        return True

    msg_lines = [
        "❌ CTE contract mismatch — uzman oturumu ortak altyapıdan kaymış:",
        f"   contract file: {contract_path}",
        f"   stored written_at: {stored.get('written_at')}  "
        f"by: {stored.get('written_by')}",
        "",
        "Diff (expected = contract on disk, actual = current repo state):",
        *diffs,
        "",
        "Muhtemel neden:",
        "  - uzman oturumunda dataset yeniden build edildi",
        "  - config.py'de split/trigger/label parametreleri değişti",
        "  - FEATURES_V1 listesi değişti",
        "Çözüm:",
        "  - eğer değişiklik kasıtlı + ortak altyapı işi → altyapı oturumunda:",
        "      python -m cte.contract write --bump-version "
        "--written-by 'altyapı oturumu vN'",
        "  - değilse: uzman oturumu değişikliği geri al",
        "  - geçici bypass: --ignore-contract (tavsiye edilmez)",
    ]
    msg = "\n".join(msg_lines)
    if raise_on_mismatch:
        raise RuntimeError(msg)
    if verbose:
        print(msg)
    return False


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> int:
    ap = argparse.ArgumentParser(description="CTE shared-infra contract tool")
    sub = ap.add_subparsers(dest="cmd", required=True)

    w = sub.add_parser("write", help="Altyapı oturumu yazar")
    w.add_argument("--dataset", default="output/cte_dataset_v1.parquet")
    w.add_argument("--out", default=str(CONTRACT_PATH))
    w.add_argument("--written-by", default="altyapı oturumu")
    w.add_argument("--bump-version", action="store_true",
                   help="Explicit version bump (kırar uzman oturumlarını)")

    v = sub.add_parser("verify", help="Uzman oturumu doğrular")
    v.add_argument("--dataset", default="output/cte_dataset_v1.parquet")
    v.add_argument("--contract", default=str(CONTRACT_PATH))
    v.add_argument("--no-raise", action="store_true",
                   help="Mismatch'te exit 0 dönsün (diff sadece rapor)")
    v.add_argument("--line", choices=["hb", "fc"], default=None,
                   help="Sadece bu line'ın trigger hash'ini kontrol et "
                   "(HB/FC uzman oturumu için). Default None = hepsi.")

    s = sub.add_parser("show", help="Mevcut contract'ı yazdır")
    s.add_argument("--contract", default=str(CONTRACT_PATH))

    args = ap.parse_args()
    os.chdir(str(_ROOT))

    if args.cmd == "write":
        c = write_contract(
            dataset_path=args.dataset,
            written_by=args.written_by,
            out_path=args.out,
            bump_version=args.bump_version,
        )
        print(json.dumps(c, indent=2, sort_keys=True))
        return 0

    if args.cmd == "verify":
        ok = verify_contract(
            dataset_path=args.dataset,
            contract_path=args.contract,
            raise_on_mismatch=not args.no_raise,
            verbose=True,
            line=args.line,
        )
        return 0 if ok else 2

    if args.cmd == "show":
        stored = load_contract(args.contract)
        if stored is None:
            print(f"❌ no contract at {args.contract}")
            return 2
        print(json.dumps(stored, indent=2, sort_keys=True))
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
