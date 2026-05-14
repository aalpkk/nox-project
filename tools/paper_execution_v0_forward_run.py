"""paper_execution_v0_forward_run — single-fire orchestrator for the LOCKED
paper_execution_v0 Forward Run (LOCKED 2026-05-06).

Spec: memory/paper_execution_v0_forward_run_spec.md (LOCKED v1 2026-05-06).

Cascade:
  Pre-flight (§5) → Phase A pre-state → Phase B archive → Phase C producer
  cascade (Line E then Line TR with --forward) → Phase D validate (§8)
  → Phase E manifest write.

Single-fire LOCK discipline. PASS or FAIL halts the LOCK. No patch/retry.
Live trade gate CLOSED. No DE Tier 2/Tier 3, no ranking/portfolio/live, no
backtest/PF/WR/meanR.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent

# Make sibling `tools._decision_target_date` importable when this script is
# invoked directly via `python tools/paper_execution_v0_forward_run.py`.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools._decision_target_date import derive_operational_target  # noqa: E402

# ---------------------------------------------------------------------------
# constants

UTC_NOW_ISO = _dt.datetime.now(_dt.timezone.utc).isoformat()
RUN_DATE_UTC = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d")

# Operational target date — dynamic per LOCKED Q-A (signal-vs-outcome
# semantics fix 2026-05-06). Replaces hardcoded `PAPER_FORWARD_FLOOR`.
_OP_CTX = derive_operational_target()
OPERATIONAL_TARGET_DATE: _dt.date = _OP_CTX.operational_target_date

# Producer's exit-window horizon (BIST trading days). Mirrored from
# `tools/paper_execution_v0.py:HORIZON_TRADING_DAYS`. Used by Phase D
# `closed_outcome_lag_excessive` validator (Q-B).
HORIZON_TRADING_DAYS = 10
# Calendar tolerance for closed_outcome_lag (Q-B): +2 trading days.
CLOSED_OUTCOME_LAG_TOLERANCE = 2

# Forward EMA artifact sha256 pins (from decision_v1_ema_forward_alignment_manifest.json post-state).
FORWARD_EMA_SHA_PINS = {
    "output/ema_context_pilot4_earliness_per_event_forward.parquet":
        "6837ee2057ac11218535688dc140cd431f315cdcbd94633b3c95377531912117",
    "output/ema_context_pilot5_panel_forward.parquet":
        "81ab43be15615073bf848447911654b97f4e3e8360e87bd48f7cf888e5d05bdd",
    "output/ema_context_pilot6_panel_forward.parquet":
        "5497b8dbf5ea58af9a2d5207e0d4318833f928f7a7060b498709e7caa5c3880f",
    "output/ema_context_pilot7_panel_forward.parquet":
        "1c80db58c5afb6acea3168a6605fd51f55fe992441cb50dcaf291dd2e795fea3",
}
EMA_FORWARD_MANIFEST = ROOT / "output/decision_v1_ema_forward_alignment_manifest.json"

LOCKED_HB_ARCHIVE = ROOT / "output/_archive/horizontal_base_event_v1__pre_refresh__asof_2026-04-29__sha256_2eb8a9a5.parquet"
LOCKED_HB_ARCHIVE_SHA = "2eb8a9a5d68e7e4831158f6a3e97c8b74521591af55e29a9682aa3ae7107b818"

HB_PARQUET = ROOT / "output/horizontal_base_event_v1.parquet"
EMA_CONTEXT_DAILY = ROOT / "output/ema_context_daily.parquet"
OHLCV_MASTER = ROOT / "output/extfeed_intraday_1h_3y_master.parquet"

LINE_E_PRODUCER = ROOT / "tools/paper_execution_v0.py"
LINE_TR_PRODUCER = ROOT / "tools/paper_execution_v0_trigger_retest.py"

LINE_E_TARGETS = [
    ROOT / "output/paper_execution_v0_trades.parquet",
    ROOT / "output/paper_execution_v0_trades.csv",
    ROOT / "output/paper_execution_v0_daily_summary.csv",
    ROOT / "output/paper_execution_v0_preview_audit.csv",
    ROOT / "output/paper_execution_v0_summary.md",
    ROOT / "output/paper_execution_v0_manifest.json",
]
LINE_TR_TARGETS = [
    ROOT / "output/paper_execution_v0_trigger_retest_trades.parquet",
    ROOT / "output/paper_execution_v0_trigger_retest_trades.csv",
    ROOT / "output/paper_execution_v0_trigger_retest_daily_summary.csv",
    ROOT / "output/paper_execution_v0_trigger_retest_preview_audit.csv",
    ROOT / "output/paper_execution_v0_trigger_retest_summary.md",
    ROOT / "output/paper_execution_v0_trigger_retest_manifest.json",
]
ALL_TARGETS = LINE_E_TARGETS + LINE_TR_TARGETS

LINE_E_TRADES = LINE_E_TARGETS[0]
LINE_TR_TRADES = LINE_TR_TARGETS[0]

OUT_MANIFEST = ROOT / "output/paper_execution_v0_forward_run_manifest.json"

# Corrective run attempt marker — disambiguates this run's archive paths from
# the prior FAILed run's partial leftovers in output/_archive/. Per ONAY
# 2026-05-06 (fix-archive-helper-and-rerun): "If new archive path collides
# with existing partial archive files, create a new unique archive suffix
# for this corrective run, e.g. __attempt2."
ATTEMPT_MARKER = "__attempt4"

# Validity-revision wiring substrings expected in producers.
VALIDITY_WIRING = ["compute_ema_signature_id", "attach_validity_fields", "apply_r1_r2_filter"]
LINE_E_COLLISION_GUARD = "_assert_no_line_e_collision"

# Forbidden imports inside this orchestrator (anti-rescue).
FORBIDDEN_IMPORT_TOKENS = [
    "decision_engine_v1",
    "nyxmomentum",
    "nyxexpansion",
    "portfolio_merge",
    "ranking",
]

# ---------------------------------------------------------------------------
# helpers

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def _file_meta(path: Path) -> dict:
    if not path.exists():
        return {"path": str(path.relative_to(ROOT)), "exists": False}
    sz = path.stat().st_size
    sha = _sha256(path)
    rows = None
    max_asof = None
    if path.suffix == ".parquet":
        try:
            md = pq.read_metadata(path)
            rows = md.num_rows
        except Exception:
            rows = None
        try:
            t = pq.read_table(path, columns=[c for c in pq.read_schema(path).names if c == "asof_date"])
            if t.num_columns > 0:
                col = t["asof_date"].to_pandas()
                col_clean = col.dropna()
                if len(col_clean) > 0:
                    try:
                        max_asof = str(pd.to_datetime(col_clean).max().date())
                    except Exception:
                        max_asof = str(col_clean.max())
        except Exception:
            max_asof = None
    return {
        "path": str(path.relative_to(ROOT)),
        "exists": True,
        "size": sz,
        "sha256": sha,
        "row_count": rows,
        "max_asof_date": max_asof,
    }


def _max_date_col(path: Path, col: str) -> str | None:
    try:
        t = pq.read_table(path, columns=[col])
        s = t[col].to_pandas().dropna()
        if len(s) == 0:
            return None
        return str(pd.to_datetime(s).max().date())
    except Exception:
        return None


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _archive_target(path: Path, run_date: str) -> dict:
    """Archive a pre-existing target file at mode 0444 + JSON sidecar.

    Refuses to clobber a colliding archive whose sha256 differs. Sidecar is
    always emitted with `.sidecar.json` suffix so its path can never equal
    the archive path even when the source target's extension is `.json`.
    """
    if not path.exists():
        return {"archived": False, "reason": "source_missing", "path": str(path.relative_to(ROOT))}
    archive_dir = ROOT / "output" / "_archive"
    _ensure_dir(archive_dir)
    pre = _file_meta(path)
    short_sha = pre["sha256"][:8]
    stem = path.stem
    ext = path.suffix
    base = f"{stem}__pre_forward_run__asof_{run_date}__sha256_{short_sha}{ATTEMPT_MARKER}"
    archive_path = archive_dir / f"{base}{ext}"
    sidecar_path = archive_dir / f"{base}.sidecar.json"
    assert sidecar_path != archive_path, (
        f"sidecar_path_equals_archive_path: {sidecar_path} == {archive_path} for source {path}"
    )
    if archive_path.exists():
        existing_sha = _sha256(archive_path)
        if existing_sha != pre["sha256"]:
            raise RuntimeError(
                f"archive_collision: {archive_path} exists with sha {existing_sha[:12]} "
                f"!= current pre-state sha {pre['sha256'][:12]} for {path}"
            )
        return {
            "archived": False,
            "reason": "archive_already_exists_content_matches",
            "archive_path": str(archive_path.relative_to(ROOT)),
            "sidecar_path": str(sidecar_path.relative_to(ROOT)),
        }
    archive_path.write_bytes(path.read_bytes())
    os.chmod(archive_path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    sidecar_data = {
        "path": str(path.relative_to(ROOT)),
        "size": pre["size"],
        "sha256": pre["sha256"],
        "rows": pre.get("row_count"),
        "max_asof_date": pre.get("max_asof_date"),
        "archived_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "archive_path": str(archive_path.relative_to(ROOT)),
    }
    sidecar_path.write_text(json.dumps(sidecar_data, indent=2))
    os.chmod(sidecar_path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    return {
        "archived": True,
        "archive_path": str(archive_path.relative_to(ROOT)),
        "sidecar_path": str(sidecar_path.relative_to(ROOT)),
    }


def _check_self_imports() -> list[str]:
    src = Path(__file__).read_text()
    bad: list[str] = []
    for tok in FORBIDDEN_IMPORT_TOKENS:
        if re.search(rf"^\s*(import|from)\s+\S*{re.escape(tok)}", src, re.MULTILINE):
            bad.append(tok)
    return bad


def _assert_validity_wiring(producer_path: Path) -> list[str]:
    src = producer_path.read_text()
    missing = [t for t in VALIDITY_WIRING if f"def {t}" not in src]
    return missing


def _assert_collision_guard(producer_path: Path) -> bool:
    src = producer_path.read_text()
    return f"def {LINE_E_COLLISION_GUARD}" in src


# ---------------------------------------------------------------------------
# pre-flight (§5)

def preflight() -> tuple[dict, list[str]]:
    info: dict = {"phase": "preflight", "checks": {}}
    fails: list[str] = []

    # 1. orchestrator self-import scan
    bad_imports = _check_self_imports()
    info["checks"]["forbidden_imports"] = {"forbidden_tokens_present": bad_imports}
    if bad_imports:
        fails.append(f"forbidden_runner_invoked: orchestrator imports {bad_imports}")

    # 2. EMA forward manifest pin verification
    if not EMA_FORWARD_MANIFEST.exists():
        fails.append("pre_flight_sha_mismatch: EMA forward manifest missing")
        return info, fails
    forward_manifest = json.loads(EMA_FORWARD_MANIFEST.read_text())
    info["checks"]["ema_forward_manifest_verdict"] = forward_manifest.get("verdict")
    if forward_manifest.get("verdict") != "PASS":
        fails.append(f"pre_flight_sha_mismatch: EMA forward manifest verdict={forward_manifest.get('verdict')}, expected PASS")

    # 3. forward EMA sha pins
    sha_check: dict = {}
    for path_str, expected in FORWARD_EMA_SHA_PINS.items():
        p = ROOT / path_str
        if not p.exists():
            sha_check[path_str] = {"exists": False}
            fails.append(f"pre_flight_freshness: {path_str} missing")
            continue
        actual = _sha256(p)
        sha_check[path_str] = {"exists": True, "sha256": actual, "matches_pin": actual == expected}
        if actual != expected:
            fails.append(f"pre_flight_sha_mismatch: {path_str} sha {actual[:12]} != pinned {expected[:12]}")
    info["checks"]["forward_ema_sha_pins"] = sha_check

    # 4. input row counts
    row_check: dict = {}
    for path_str in [HB_PARQUET.relative_to(ROOT).as_posix(), *FORWARD_EMA_SHA_PINS.keys()]:
        p = ROOT / path_str
        try:
            n = pq.read_metadata(p).num_rows
        except Exception as e:
            n = f"ERR:{e}"
        row_check[path_str] = n
    info["checks"]["row_counts"] = row_check
    expected_rows = row_check.get(HB_PARQUET.relative_to(ROOT).as_posix())
    if expected_rows != 10726:
        fails.append(f"pre_flight_row_count: HB rows={expected_rows}, expected 10726")
    for path_str in FORWARD_EMA_SHA_PINS:
        n = row_check.get(path_str)
        if n != 10726:
            fails.append(f"pre_flight_row_count: {path_str} rows={n}, expected 10726")

    # 5. freshness floors (max date >= 2026-05-05)
    freshness: dict = {}
    for path, col in [
        (HB_PARQUET, "bar_date"),
        (ROOT / "output/ema_context_pilot4_earliness_per_event_forward.parquet", "bar_date"),
        (ROOT / "output/ema_context_pilot5_panel_forward.parquet", "bar_date"),
        (ROOT / "output/ema_context_pilot6_panel_forward.parquet", "bar_date"),
        (ROOT / "output/ema_context_pilot7_panel_forward.parquet", "bar_date"),
        (EMA_CONTEXT_DAILY, "date"),
    ]:
        if not path.exists():
            freshness[str(path.relative_to(ROOT))] = "missing"
            continue
        m = _max_date_col(path, col)
        freshness[str(path.relative_to(ROOT))] = m
        if m is None:
            fails.append(f"pre_flight_freshness: {path} max({col}) unresolvable")
        else:
            d = pd.to_datetime(m).date()
            if d < OPERATIONAL_TARGET_DATE:
                fails.append(f"pre_flight_freshness: {path} max({col})={m} < target {OPERATIONAL_TARGET_DATE}")
    info["checks"]["freshness"] = freshness

    # 6. HB ↔ Earliness key parity (load full key tuple)
    key_parity: dict = {}
    try:
        hb_keys = pq.read_table(
            HB_PARQUET,
            columns=["ticker", "bar_date", "setup_family", "signal_type", "breakout_bar_date"],
        ).to_pandas()
        earl_path = ROOT / "output/ema_context_pilot4_earliness_per_event_forward.parquet"
        earl_keys = pq.read_table(
            earl_path,
            columns=["ticker", "bar_date", "setup_family", "signal_type", "breakout_bar_date"],
        ).to_pandas()
        for df in (hb_keys, earl_keys):
            df["_bar_date_d"] = pd.to_datetime(df["bar_date"]).dt.date
            df["_breakout_d"] = pd.to_datetime(df["breakout_bar_date"]).dt.date
        keys = ["ticker", "_bar_date_d", "setup_family", "signal_type", "_breakout_d"]
        hb_dups = int(hb_keys.duplicated(subset=keys).sum())
        earl_dups = int(earl_keys.duplicated(subset=keys).sum())
        hb_set = set(map(tuple, hb_keys[keys].itertuples(index=False, name=None)))
        earl_set = set(map(tuple, earl_keys[keys].itertuples(index=False, name=None)))
        only_hb = hb_set - earl_set
        only_earl = earl_set - hb_set
        key_parity = {
            "hb_dups": hb_dups,
            "earl_dups": earl_dups,
            "only_in_hb": len(only_hb),
            "only_in_earl": len(only_earl),
            "set_equal": (len(only_hb) == 0 and len(only_earl) == 0),
        }
        if hb_dups > 0:
            fails.append(f"pre_flight_key_parity: HB has {hb_dups} duplicate stable_event_key tuples")
        if earl_dups > 0:
            fails.append(f"pre_flight_key_parity: earliness has {earl_dups} duplicate stable_event_key tuples")
        if not key_parity["set_equal"]:
            fails.append(
                f"pre_flight_key_parity: HB↔earliness key set mismatch only_in_hb={len(only_hb)} only_in_earl={len(only_earl)}"
            )
    except Exception as e:
        key_parity["error"] = repr(e)
        fails.append(f"pre_flight_key_parity: exception {e!r}")
    info["checks"]["key_parity"] = key_parity

    # 6b. HB ↔ Earliness merge diagnostic (Remediation C — ONAY 2026-05-06).
    # Diagnostic-only: reports merge indicator counts + earliness_score_pct null
    # distribution by signal_state / track to surface the producer's halt
    # condition before the producer is even invoked. Halts only when
    # `unmatched_count > 0` (true key-mismatch); null earliness is reported but
    # not a halt condition since cohort filters drop those rows naturally.
    merge_diag: dict = {}
    try:
        hb_full = pd.read_parquet(
            HB_PARQUET,
            columns=[
                "ticker",
                "bar_date",
                "setup_family",
                "signal_type",
                "breakout_bar_date",
                "signal_state",
            ],
        )
        earl_path = ROOT / "output/ema_context_pilot4_earliness_per_event_forward.parquet"
        earl_full = pd.read_parquet(earl_path)
        for df in (hb_full, earl_full):
            df["_bar_date_d"] = pd.to_datetime(df["bar_date"]).dt.date
            df["_breakout_d"] = pd.to_datetime(df["breakout_bar_date"]).dt.date
        keys = ["ticker", "_bar_date_d", "setup_family", "signal_type", "_breakout_d"]
        merge_diag["hb_rows"] = int(len(hb_full))
        merge_diag["earliness_rows"] = int(len(earl_full))
        merge_diag["hb_key_dups"] = int(hb_full.duplicated(subset=keys).sum())
        merge_diag["earliness_key_dups"] = int(earl_full.duplicated(subset=keys).sum())
        earl_cols = [c for c in ["earliness_score_pct", "track"] if c in earl_full.columns]
        m = hb_full.merge(
            earl_full[keys + earl_cols],
            on=keys,
            how="left",
            indicator=True,
        )
        merge_diag["merge_indicator"] = {
            str(k): int(v) for k, v in m["_merge"].astype(str).value_counts().to_dict().items()
        }
        unmatched_count = int((m["_merge"] != "both").sum())
        merge_diag["unmatched_count"] = unmatched_count
        if "earliness_score_pct" in m.columns:
            merge_diag["earliness_null_count"] = int(m["earliness_score_pct"].isna().sum())
            null_mask = m["earliness_score_pct"].isna()
            if null_mask.any():
                if "signal_state" in m.columns:
                    merge_diag["null_distribution_by_signal_state"] = {
                        str(k): int(v)
                        for k, v in m.loc[null_mask, "signal_state"]
                        .value_counts(dropna=False)
                        .to_dict()
                        .items()
                    }
                if "track" in m.columns:
                    merge_diag["null_distribution_by_track"] = {
                        str(k): int(v)
                        for k, v in m.loc[null_mask, "track"]
                        .value_counts(dropna=False)
                        .to_dict()
                        .items()
                    }
        if unmatched_count > 0:
            fails.append(
                f"pre_flight_merge_diag: HB↔earliness key-merge unmatched_count={unmatched_count}"
            )
    except Exception as e:
        merge_diag["error"] = repr(e)
        fails.append(f"pre_flight_merge_diag: exception {e!r}")
    info["checks"]["merge_diag"] = merge_diag

    # 7. locked HB archive sanity
    if not LOCKED_HB_ARCHIVE.exists():
        fails.append(f"pre_flight_locked_archive_missing: {LOCKED_HB_ARCHIVE}")
        info["checks"]["locked_hb_archive"] = {"exists": False}
    else:
        actual = _sha256(LOCKED_HB_ARCHIVE)
        mode = oct(LOCKED_HB_ARCHIVE.stat().st_mode & 0o777)
        info["checks"]["locked_hb_archive"] = {
            "exists": True, "sha256": actual, "mode": mode,
            "matches_pin": actual == LOCKED_HB_ARCHIVE_SHA,
        }
        if actual != LOCKED_HB_ARCHIVE_SHA:
            fails.append(f"pre_flight_locked_archive_missing: sha mismatch {actual[:12]} != pinned")

    # 8. validity wiring static check on both producers
    e_missing = _assert_validity_wiring(LINE_E_PRODUCER)
    tr_missing = _assert_validity_wiring(LINE_TR_PRODUCER)
    info["checks"]["validity_wiring"] = {
        "line_e_missing": e_missing,
        "line_tr_missing": tr_missing,
    }
    if e_missing:
        fails.append(f"pre_flight_validity_wiring_missing: Line E missing {e_missing}")
    if tr_missing:
        fails.append(f"pre_flight_validity_wiring_missing: Line TR missing {tr_missing}")

    # 9. Line E collision guard in TR producer
    has_guard = _assert_collision_guard(LINE_TR_PRODUCER)
    info["checks"]["line_e_collision_guard_present"] = has_guard
    if not has_guard:
        fails.append("pre_flight_collision_guard_missing: Line TR producer missing _assert_no_line_e_collision")

    return info, fails


# ---------------------------------------------------------------------------
# Phase A: pre-state for §4.1 + §4.2 targets + protected sets

def phase_a_prestate() -> dict:
    pre: dict = {"targets": {}, "protected_sets": {}}
    for p in ALL_TARGETS:
        pre["targets"][str(p.relative_to(ROOT))] = _file_meta(p)
    forward_manifest = json.loads(EMA_FORWARD_MANIFEST.read_text())
    locked_research = list(
        forward_manifest["phases"]["A_pre_state"]["locked_research_pre"].keys()
    )
    protected_non_target_full = list(
        forward_manifest["phases"]["A_pre_state"]["protected_non_target_pre"].keys()
    )
    paper_target_paths = {str(p.relative_to(ROOT)) for p in ALL_TARGETS}
    protected_non_target = [
        p for p in protected_non_target_full if p not in paper_target_paths
    ]
    pre["protected_sets"]["locked_research_paths"] = locked_research
    pre["protected_sets"]["protected_non_target_paths"] = protected_non_target
    pre["protected_sets"]["paper_targets_excluded_from_protection"] = sorted(
        list(paper_target_paths & set(protected_non_target_full))
    )

    locked_research_pre: dict = {}
    for path_str in locked_research:
        p = ROOT / path_str
        if not p.exists():
            locked_research_pre[path_str] = {"exists": False}
        else:
            locked_research_pre[path_str] = {"exists": True, "size": p.stat().st_size, "sha256": _sha256(p)}
    pre["locked_research_pre"] = locked_research_pre

    protected_non_target_pre: dict = {}
    for path_str in protected_non_target:
        p = ROOT / path_str
        if not p.exists():
            protected_non_target_pre[path_str] = {"exists": False}
        else:
            protected_non_target_pre[path_str] = {"exists": True, "size": p.stat().st_size, "sha256": _sha256(p)}
    pre["protected_non_target_pre"] = protected_non_target_pre

    # ONAY 2026-05-06: record prior failed archive leftovers (do NOT delete).
    archive_dir = ROOT / "output" / "_archive"
    leftovers: list[dict] = []
    if archive_dir.exists():
        for f in sorted(archive_dir.glob("*__pre_forward_run__asof_*")):
            if ATTEMPT_MARKER in f.name:
                continue  # files written by THIS corrective run
            leftovers.append({
                "path": str(f.relative_to(ROOT)),
                "size": f.stat().st_size,
                "sha256_short": _sha256(f)[:12],
                "mode": oct(stat.S_IMODE(os.stat(f).st_mode)),
            })
    pre["prior_failed_archive_leftovers"] = leftovers
    pre["prior_failed_archive_leftovers_count"] = len(leftovers)
    return pre


# ---------------------------------------------------------------------------
# Phase B: archive

def phase_b_archive() -> dict:
    archived: dict = {}
    for p in ALL_TARGETS:
        archived[str(p.relative_to(ROOT))] = _archive_target(p, RUN_DATE_UTC)
    return archived


# ---------------------------------------------------------------------------
# Phase C: producer cascade

def _run_producer(producer_path: Path) -> dict:
    cmd = [sys.executable, str(producer_path), "--mode", "close_confirmed", "--forward"]
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
    dur = time.time() - t0
    return {
        "command": " ".join(cmd),
        "returncode": proc.returncode,
        "duration_sec": round(dur, 2),
        "stdout_tail": proc.stdout[-4000:] if proc.stdout else "",
        "stderr_tail": proc.stderr[-4000:] if proc.stderr else "",
    }


def phase_c_cascade() -> tuple[dict, list[str]]:
    fails: list[str] = []
    line_e = _run_producer(LINE_E_PRODUCER)
    if line_e["returncode"] != 0:
        fails.append(f"producer_rc_nonzero: Line E rc={line_e['returncode']}")
    if fails:
        return {"line_e": line_e, "line_tr": None}, fails
    line_tr = _run_producer(LINE_TR_PRODUCER)
    if line_tr["returncode"] != 0:
        fails.append(f"producer_rc_nonzero: Line TR rc={line_tr['returncode']}")
    return {"line_e": line_e, "line_tr": line_tr}, fails


# ---------------------------------------------------------------------------
# Phase D: validate (§8 PASS criteria)

VALIDITY_COLS = [
    "paper_valid_from", "paper_valid_until", "paper_signal_age",
    "paper_expired_flag", "ema_signature_id",
]


def _xist_trading_days_between(d_from: _dt.date, d_to: _dt.date) -> int:
    """Number of XIST trading sessions strictly between d_from and d_to (inclusive
    on the later end, exclusive on the earlier end). Returns 0 if d_from >= d_to.
    Uses the XIST calendar to be consistent with the producer's horizon math.
    """
    if d_from >= d_to:
        return 0
    import exchange_calendars as _xcals
    xist = _xcals.get_calendar("XIST")
    sessions = xist.sessions_in_range(pd.Timestamp(d_from) + pd.Timedelta(days=1), pd.Timestamp(d_to))
    return int(len(sessions))


def _validate_paper_parquet(path: Path, cohort_col: str, cohort_value: str) -> dict:
    out: dict = {"path": str(path.relative_to(ROOT))}
    if not path.exists():
        out["error"] = "missing"
        return out
    t = pq.read_table(path)
    df = t.to_pandas()
    out["row_count"] = len(df)

    # ── Q-D counters (LOCKED 2026-05-06 signal-vs-outcome semantics fix) ──
    # asof_date kept as backward-compat alias of signal_asof_date (Q-C).
    if "signal_asof_date" in df.columns:
        s = pd.to_datetime(df["signal_asof_date"]).dropna()
        out["signal_asof_max"] = str(s.max().date()) if len(s) else None
    elif "asof_date" in df.columns:  # Line TR not yet patched: alias still works
        s = pd.to_datetime(df["asof_date"]).dropna()
        out["signal_asof_max"] = str(s.max().date()) if len(s) else None
    else:
        out["signal_asof_max"] = None
    # Backward-compat field name; same value as signal_asof_max.
    out["max_asof_date"] = out["signal_asof_max"]

    # Status distribution (Q-D).
    if "status" in df.columns:
        sc = df["status"].value_counts(dropna=False).to_dict()
        out["open_count"] = int(sc.get("OPEN", 0))
        out["closed_count"] = int(sc.get("CLOSED", 0))
        out["skipped_count"] = int(sc.get("SKIPPED", 0))
    else:
        out["open_count"] = out["closed_count"] = out["skipped_count"] = None

    # Outcome side: closed_outcome_max from outcome_asof_date over CLOSED rows
    # (fall back to exit_date when outcome_asof_date is absent — Line TR not yet patched).
    closed = df[df.get("status") == "CLOSED"] if "status" in df.columns else df.iloc[0:0]
    if len(closed):
        if "outcome_asof_date" in closed.columns:
            oc = pd.to_datetime(closed["outcome_asof_date"]).dropna()
        elif "exit_date" in closed.columns:
            oc = pd.to_datetime(closed["exit_date"]).dropna()
        else:
            oc = pd.Series([], dtype="datetime64[ns]")
        out["closed_outcome_max"] = str(oc.max().date()) if len(oc) else None
    else:
        out["closed_outcome_max"] = None

    # OPEN realized_R must be null (Q-E invariant).
    open_rows = df[df.get("status") == "OPEN"] if "status" in df.columns else df.iloc[0:0]
    if "realized_R_paper" in df.columns and len(open_rows):
        out["open_realized_R_not_null"] = int(open_rows["realized_R_paper"].notna().sum())
    else:
        out["open_realized_R_not_null"] = 0

    # CLOSED with observable outcome must have non-null realized_R_paper, EXCEPT
    # rows where (entry_reference_price - invalidation_level) <= 0 (producer
    # edge-case allowed at tools/paper_execution_v0.py risk_per_share guard).
    if len(closed) and "realized_R_paper" in closed.columns:
        outcome_col = "outcome_asof_date" if "outcome_asof_date" in closed.columns else "exit_date"
        observable = closed[closed[outcome_col].notna()] if outcome_col in closed.columns else closed
        null_R = observable[observable["realized_R_paper"].isna()]
        # Exclude risk_per_share<=0 edge case if both columns present.
        if "entry_reference_price" in null_R.columns and "invalidation_level" in null_R.columns:
            risk = null_R["entry_reference_price"] - null_R["invalidation_level"]
            null_R = null_R[risk > 0]
        out["closed_realized_R_null_for_observable_outcome"] = int(len(null_R))
    else:
        out["closed_realized_R_null_for_observable_outcome"] = 0

    out["columns_present"] = {c: (c in df.columns) for c in VALIDITY_COLS}

    if cohort_col in df.columns:
        cohort_rows = df[df[cohort_col] == cohort_value]
        out["cohort_count"] = int(len(cohort_rows))
        # Restrict to status OPEN/CLOSED — SKIPPED rows carry NULL validity per producer contract.
        emitted = cohort_rows[cohort_rows["status"].isin(["OPEN", "CLOSED"])]
        out["cohort_emitted_count"] = int(len(emitted))
        nulls = {}
        for col in VALIDITY_COLS:
            if col in emitted.columns:
                if col in {"paper_signal_age"}:
                    n_null = int(emitted[col].isna().sum())
                else:
                    n_null = int(emitted[col].isna().sum())
                nulls[col] = n_null
            else:
                nulls[col] = "MISSING_COLUMN"
        out["cohort_emitted_validity_null_counts"] = nulls
        if "paper_signal_age" in emitted.columns:
            sage = emitted["paper_signal_age"].dropna()
            out["paper_signal_age_unique"] = sorted([int(v) for v in sage.unique().tolist()])
        if "paper_expired_flag" in emitted.columns:
            out["paper_expired_flag_unique"] = sorted(emitted["paper_expired_flag"].dropna().unique().astype(bool).tolist())
        if "ema_signature_id" in emitted.columns:
            sigs = emitted["ema_signature_id"].dropna().astype(str)
            out["ema_signature_id_lengths_unique"] = sorted(sigs.str.len().unique().tolist())
            out["ema_signature_id_hex_only"] = bool(sigs.str.match(r"^[0-9a-f]+$").all())
    return out


def phase_d_validate() -> tuple[dict, list[str]]:
    """Phase D — LOCKED 2026-05-06 signal-vs-outcome semantics fix §5.3.

    Fail classes:
      - signal_asof_floor              HARD FAIL (Q-A/Q-E)
      - closed_outcome_lag_excessive   HARD FAIL (Q-B; HORIZON+tolerance trading days)
      - open_realized_R_not_null       HARD FAIL (Q-E invariant)
      - closed_realized_R_null_for_observable_outcome  HARD FAIL (Q-D)
      - signal_asof_max_less_than_outcome_max  HARD FAIL (sanity)
    """
    fails: list[str] = []
    out: dict = {
        "operational_target_date": OPERATIONAL_TARGET_DATE.isoformat(),
        "horizon_trading_days": HORIZON_TRADING_DAYS,
        "closed_outcome_lag_tolerance": CLOSED_OUTCOME_LAG_TOLERANCE,
    }

    line_e = _validate_paper_parquet(LINE_E_TRADES, "ema_tier", "TIER_A_PAPER")
    line_tr = _validate_paper_parquet(LINE_TR_TRADES, "ema_tr_tier", "EVENT_NEAR_TIER")
    out["line_e"] = line_e
    out["line_tr"] = line_tr

    for label, vd in (("Line E", line_e), ("Line TR", line_tr)):
        if "error" in vd:
            fails.append(f"target_missing: {label} target missing post-run")
            continue

        # ── signal_asof_floor (Q-A) ──
        sm = vd.get("signal_asof_max")
        if sm is None:
            fails.append(f"signal_asof_floor: {label} signal_asof_max unresolvable")
        else:
            sm_d = pd.to_datetime(sm).date()
            if sm_d < OPERATIONAL_TARGET_DATE:
                fails.append(
                    f"signal_asof_floor: {label} signal_asof_max={sm} < target {OPERATIONAL_TARGET_DATE}"
                )

            # ── closed_outcome_lag_excessive (Q-B) ──
            om = vd.get("closed_outcome_max")
            if om is not None:
                om_d = pd.to_datetime(om).date()
                lag = _xist_trading_days_between(om_d, sm_d)
                vd["closed_outcome_lag_trading_days"] = lag
                if lag > HORIZON_TRADING_DAYS + CLOSED_OUTCOME_LAG_TOLERANCE:
                    fails.append(
                        f"closed_outcome_lag_excessive: {label} lag={lag} trading days "
                        f"(signal_asof_max={sm}, closed_outcome_max={om}) > "
                        f"HORIZON({HORIZON_TRADING_DAYS}) + tolerance({CLOSED_OUTCOME_LAG_TOLERANCE})"
                    )

                # ── signal_asof_max_less_than_outcome_max (sanity) ──
                if sm_d < om_d:
                    fails.append(
                        f"signal_asof_max_less_than_outcome_max: {label} "
                        f"signal_asof_max={sm} < closed_outcome_max={om} (impossible)"
                    )

        # ── open_realized_R_not_null (Q-E) ──
        n_open_R = vd.get("open_realized_R_not_null", 0)
        if isinstance(n_open_R, int) and n_open_R > 0:
            fails.append(
                f"open_realized_R_not_null: {label} {n_open_R} OPEN rows have non-null realized_R_paper"
            )

        # ── closed_realized_R_null_for_observable_outcome (Q-D) ──
        n_closed_null = vd.get("closed_realized_R_null_for_observable_outcome", 0)
        if isinstance(n_closed_null, int) and n_closed_null > 0:
            fails.append(
                f"closed_realized_R_null_for_observable_outcome: {label} {n_closed_null} "
                f"CLOSED rows with observable outcome have null realized_R_paper "
                f"(excluding risk_per_share<=0 edge case)"
            )

        # ── existing validity-column hygiene checks (unchanged) ──
        if any(v is False for v in vd.get("columns_present", {}).values()):
            fails.append(f"validity_columns_missing_or_null: {label} missing columns {[k for k,v in vd['columns_present'].items() if not v]}")
        nulls = vd.get("cohort_emitted_validity_null_counts", {})
        for col, n in nulls.items():
            if n == "MISSING_COLUMN" or (isinstance(n, int) and n > 0):
                fails.append(f"validity_columns_missing_or_null: {label} {col}={n}")
        if "paper_signal_age_unique" in vd and vd["paper_signal_age_unique"] not in ([0], []):
            fails.append(f"validity_columns_missing_or_null: {label} paper_signal_age_unique={vd['paper_signal_age_unique']} expected [0]")
        if "paper_expired_flag_unique" in vd and vd["paper_expired_flag_unique"] not in ([False], []):
            fails.append(f"validity_columns_missing_or_null: {label} paper_expired_flag_unique={vd['paper_expired_flag_unique']} expected [False]")
        if "ema_signature_id_lengths_unique" in vd and vd["ema_signature_id_lengths_unique"] not in ([16], []):
            fails.append(f"validity_columns_missing_or_null: {label} ema_signature_id length set={vd['ema_signature_id_lengths_unique']}")
        if vd.get("ema_signature_id_hex_only") is False:
            fails.append(f"validity_columns_missing_or_null: {label} ema_signature_id non-hex")

    return out, fails


# ---------------------------------------------------------------------------
# Phase E: post-state + protected drift

def phase_e_poststate(prestate: dict) -> tuple[dict, list[str]]:
    fails: list[str] = []
    post: dict = {"targets": {}, "protected_sets": {}}
    for p in ALL_TARGETS:
        post["targets"][str(p.relative_to(ROOT))] = _file_meta(p)

    # locked_research drift
    lr_pre = prestate["locked_research_pre"]
    lr_post: dict = {}
    lr_drift: list[str] = []
    for path_str, pre_meta in lr_pre.items():
        p = ROOT / path_str
        if not p.exists():
            lr_post[path_str] = {"exists": False}
            if pre_meta.get("exists"):
                lr_drift.append(f"{path_str}: present_pre absent_post")
            continue
        cur = {"exists": True, "size": p.stat().st_size, "sha256": _sha256(p)}
        lr_post[path_str] = cur
        if pre_meta.get("sha256") != cur["sha256"]:
            lr_drift.append(f"{path_str}: sha256 {pre_meta.get('sha256','MISS')[:8]}→{cur['sha256'][:8]}")
    post["locked_research_post"] = lr_post
    post["locked_research_drift"] = lr_drift
    post["locked_research_byte_equal"] = (len(lr_drift) == 0)
    if lr_drift:
        fails.append(f"locked_research_drift: {len(lr_drift)} files drifted")

    # protected_non_target drift
    pn_pre = prestate["protected_non_target_pre"]
    pn_post: dict = {}
    pn_drift: list[str] = []
    for path_str, pre_meta in pn_pre.items():
        p = ROOT / path_str
        if not p.exists():
            pn_post[path_str] = {"exists": False}
            if pre_meta.get("exists"):
                pn_drift.append(f"{path_str}: present_pre absent_post")
            continue
        cur = {"exists": True, "size": p.stat().st_size, "sha256": _sha256(p)}
        pn_post[path_str] = cur
        if pre_meta.get("sha256") != cur["sha256"]:
            pn_drift.append(f"{path_str}: sha256 {pre_meta.get('sha256','MISS')[:8]}→{cur['sha256'][:8]}")
    post["protected_non_target_post"] = pn_post
    post["protected_non_target_drift"] = pn_drift
    post["protected_non_target_byte_equal"] = (len(pn_drift) == 0)
    if pn_drift:
        fails.append(f"protected_non_target_drift: {len(pn_drift)} files drifted")

    # locked HB archive drift
    if LOCKED_HB_ARCHIVE.exists():
        actual = _sha256(LOCKED_HB_ARCHIVE)
        post["locked_hb_archive_post"] = {
            "sha256": actual,
            "matches_pin": actual == LOCKED_HB_ARCHIVE_SHA,
        }
        if actual != LOCKED_HB_ARCHIVE_SHA:
            fails.append(f"locked_hb_archive_drift: sha {actual[:12]} != pinned")
    else:
        post["locked_hb_archive_post"] = {"exists": False}
        fails.append("locked_hb_archive_drift: archive missing post-run")

    return post, fails


# ---------------------------------------------------------------------------
# producer-manifest extraction (R1/R2/validity counts)

def _extract_producer_manifest(path: Path) -> dict:
    if not path.exists():
        return {"exists": False}
    try:
        d = json.loads(path.read_text())
    except Exception as e:
        return {"exists": True, "error": repr(e)}
    keys_of_interest = [
        "trade_log_rows",
        "tier_a_count",
        "tier_b_count",
        "cohort_counts",
        "trigger_count",
        "retest_bounce_count",
        "status_distribution",
        "r1_stale_signature_blocks_count",
        "r2_open_duplicate_blocks_count",
        "signal_date_unavailable_halts",
        "validity_revision_applied",
        "validity_window_default",
        "boot_integrity",
    ]
    return {k: d.get(k) for k in keys_of_interest if k in d}


# ---------------------------------------------------------------------------
# main

def main() -> int:
    print(f"[paper_execution_v0_forward_run] start {UTC_NOW_ISO}")
    overall_fails: list[str] = []

    print("[paper_execution_v0_forward_run] preflight (§5)...")
    pre_info, pre_fails = preflight()
    overall_fails.extend(pre_fails)

    if pre_fails:
        manifest = {
            "spec_path": "memory/paper_execution_v0_forward_run_spec.md",
            "spec_status": "LOCKED v1 2026-05-06",
            "run_started_utc": UTC_NOW_ISO,
            "verdict": "FAIL",
            "fail_classes": [_classify(f) for f in pre_fails],
            "fail_reasons": pre_fails,
            "phases": {"preflight": pre_info},
        }
        OUT_MANIFEST.write_text(json.dumps(manifest, indent=2, default=str))
        print(f"[paper_execution_v0_forward_run] FAIL pre-flight; manifest at {OUT_MANIFEST.relative_to(ROOT)}")
        return 1

    print("[paper_execution_v0_forward_run] phase A pre-state...")
    pre_state = phase_a_prestate()

    print("[paper_execution_v0_forward_run] phase B archive (mode 0444)...")
    archived = phase_b_archive()

    print("[paper_execution_v0_forward_run] phase C producer cascade (--forward)...")
    cascade, cascade_fails = phase_c_cascade()
    overall_fails.extend(cascade_fails)
    if cascade_fails:
        manifest = {
            "spec_path": "memory/paper_execution_v0_forward_run_spec.md",
            "spec_status": "LOCKED v1 2026-05-06",
            "run_started_utc": UTC_NOW_ISO,
            "verdict": "FAIL",
            "fail_classes": [_classify(f) for f in cascade_fails],
            "fail_reasons": cascade_fails,
            "phases": {
                "preflight": pre_info,
                "phase_a_prestate": pre_state,
                "phase_b_archive": archived,
                "phase_c_cascade": cascade,
            },
        }
        OUT_MANIFEST.write_text(json.dumps(manifest, indent=2, default=str))
        print(f"[paper_execution_v0_forward_run] FAIL producer cascade; manifest at {OUT_MANIFEST.relative_to(ROOT)}")
        return 1

    print("[paper_execution_v0_forward_run] phase D validate (§8)...")
    validate, validate_fails = phase_d_validate()
    overall_fails.extend(validate_fails)

    print("[paper_execution_v0_forward_run] phase E post-state + drift...")
    post_state, post_fails = phase_e_poststate(pre_state)
    overall_fails.extend(post_fails)

    line_e_manifest_path = ROOT / "output/paper_execution_v0_manifest.json"
    line_tr_manifest_path = ROOT / "output/paper_execution_v0_trigger_retest_manifest.json"
    producer_manifests = {
        "line_e_manifest": _extract_producer_manifest(line_e_manifest_path),
        "line_tr_manifest": _extract_producer_manifest(line_tr_manifest_path),
    }

    verdict = "PASS" if not overall_fails else "FAIL"
    fail_classes = [_classify(f) for f in overall_fails]
    manifest = {
        "spec_path": "memory/paper_execution_v0_forward_run_spec.md",
        "spec_status": "LOCKED v1 2026-05-06",
        "run_started_utc": UTC_NOW_ISO,
        "run_date_utc": RUN_DATE_UTC,
        "operation": "Line E + Line TR paper_execution forward emission (single-fire)",
        "verdict": verdict,
        "fail_classes": fail_classes,
        "fail_reasons": overall_fails,
        "phases": {
            "preflight": pre_info,
            "phase_a_prestate": pre_state,
            "phase_b_archive": archived,
            "phase_c_cascade": cascade,
            "phase_d_validate": validate,
            "phase_e_poststate": post_state,
        },
        "producer_manifests_summary": producer_manifests,
        "live_trade_gate": "CLOSED",
        "next_surface": "Tier 2 rerun under separate ONAY (PASS only)",
    }
    OUT_MANIFEST.write_text(json.dumps(manifest, indent=2, default=str))
    print(f"[paper_execution_v0_forward_run] verdict={verdict}; manifest at {OUT_MANIFEST.relative_to(ROOT)}")
    if verdict == "FAIL":
        for f in overall_fails:
            print(f"  FAIL: {f}", file=sys.stderr)
        return 1
    return 0


def _classify(reason: str) -> str:
    head = reason.split(":", 1)[0].strip()
    return head


if __name__ == "__main__":
    sys.exit(main())
