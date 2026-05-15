"""Decision Engine v1 — Upstream Scanner Refresh orchestrator.

LOCKED 2026-05-05 by `memory/decision_engine_v1_upstream_scanner_refresh_spec.md`.
Single authorized refresh sequence. Per Q1: invokes the existing producers
verbatim via subprocess; this file is a thin orchestrator, NOT a new producer.

Producers (UNTOUCHED):
  - tools/build_horizontal_base_event_v1.py
  - nyxexpansion/tools/rebuild_dataset_delta_intraday.py

Run order (Q2): HB first, then nyxexp; if HB FAILs do NOT run nyxexp.
Single-fire per producer (Q8); pre-flight dep check (Q4); archive-before-run
mode 0444 (Q5); JSON manifest (Q9); protected-files byte-equal (Q10).

Verdict semantics:
  - PASS: both producers PASS, both refreshed max_date >= LCTD, archives
    written, manifest written, protected files byte-equal, no downstream.
  - PARTIAL_FAIL: HB PASS but nyxexp FAIL — keep HB refreshed output, no
    auto-rollback, no downstream, await ONAY (Q7).
  - FAIL: HB FAIL (nyxexp SKIPPED), or any pre-flight/dep/protected-file
    violation, or unauthorized write detected.

Invocation:
  PYTHONPATH=. python tools/decision_engine_v1_upstream_scanner_refresh.py
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

# ──────────────────────────────────────────────────────────────────────────
# P3 LOCK 2026-05-05 (memory/decision_engine_v1_backbone_refresh_lctd_spec.md §5):
# replaced hardcoded `LCTD = "2026-04-30"` with calendar-driven derivation.
# Variable name `LCTD` retained as a legacy alias bound to the dynamic value
# so existing manifest field names + downstream string compares stay stable
# (minimal-rename scope per Q2 ONAY).
#
# PR-DE-3.11: added `--lctd-required YYYY-MM-DD` CLI override. When invoked
# from .github/workflows/decision-engine-v1.yml the workflow's `target_date`
# input is propagated so Stage 3 LCTD binds to the same date as the
# selected master-data-pull artifact (close-mode contract). Local-shell
# invocations omit the flag and retain the runtime-derived behavior.
# No fallback weakening: extfeed-vs-LCTD and HB-post-vs-LCTD gates unchanged.
from tools._decision_target_date import (  # noqa: E402
    derive_operational_target,
    assert_freshness_contract,
)

_OP_CTX = derive_operational_target()
LCTD = _OP_CTX.operational_target_date.isoformat()  # legacy alias, dynamic value
LCTD_RUNTIME_DERIVED = LCTD          # snapshot of runtime-derived value (audit)
LCTD_SOURCE = "runtime_derived"      # mutated by main() if --lctd-required given
ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)

# Refresh targets (Q2 order)
HB_PARQUET = "output/horizontal_base_event_v1.parquet"
NYXEXP_PARQUET = "output/nyxexp_dataset_v4.parquet"

# HB sidecar files written by its producer (not refresh targets but emitted)
HB_SIDECARS = [
    "output/horizontal_base_event_v1_manifest.json",
    "output/horizontal_base_event_v1_split.json",
    "output/horizontal_base_event_v1_audit.log",
]

# HB upstream dependencies (Q4 pre-flight)
HB_DEPS = {
    "scanner_v1_event_quality_events": "output/scanner_v1_event_quality_events.csv",
    "regime_labels_daily_rdp_v1": "output/regime_labels_daily_rdp_v1.csv",
    "xu100_extfeed_daily": "output/xu100_extfeed_daily.parquet",
    "extfeed_intraday_1h_3y_master": "output/extfeed_intraday_1h_3y_master.parquet",
}

# nyxexp upstream dependencies (Q4 pre-flight)
NYXEXP_DEPS = {
    "ohlcv_10y_fintables_master": "output/ohlcv_10y_fintables_master.parquet",
    "extfeed_intraday_1h_3y_master": "output/extfeed_intraday_1h_3y_master.parquet",
    # nyxexp_intraday_master is a CACHE the producer can rebuild — not a hard dep.
}

# Protected non-target files (Q10)
PROTECTED_FILES = [
    "output/paper_execution_v0_trades.parquet",
    "output/paper_execution_v0_trigger_retest_trades.parquet",
    "output/decision_engine_v1_events.parquet",
    "output/decision_engine_v1_paper_link_integrity.csv",
    "output/decision_engine_v1_tier2_label_distribution.csv",
    "output/decision_engine_v1_tier2_label_table_summary.md",
    "output/decision_engine_v1_tier3_review_report.md",
    "output/decision_v0_classification_panel.parquet",
    "output/extfeed_intraday_1h_3y_master.parquet",
    # mb_scanner event-history (8 files)
    "output/mb_scanner_events_mb_5h.parquet",
    "output/mb_scanner_events_mb_1d.parquet",
    "output/mb_scanner_events_mb_1w.parquet",
    "output/mb_scanner_events_mb_1M.parquet",
    "output/mb_scanner_events_bb_5h.parquet",
    "output/mb_scanner_events_bb_1d.parquet",
    "output/mb_scanner_events_bb_1w.parquet",
    "output/mb_scanner_events_bb_1M.parquet",
]
# portfolio_merge_paper outputs — discover at runtime by glob
import glob as _glob
PROTECTED_FILES.extend(sorted(_glob.glob("output/portfolio_merge_paper_*")))

ARCHIVE_DIR = Path("output/_archive")
MANIFEST_PATH = "output/decision_v1_upstream_scanner_refresh_manifest.json"

# ──────────────────────────────────────────────────────────────────────────


def _sha256_size(path: str) -> tuple[str, int]:
    p = Path(path)
    if not p.exists():
        return ("", 0)
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return (h.hexdigest(), p.stat().st_size)


def _capture_protected() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for f in PROTECTED_FILES:
        sha, sz = _sha256_size(f)
        out[f] = {"sha256": sha, "size": sz}
    return out


def _verify_protected(pre: dict[str, dict]) -> tuple[bool, list[str]]:
    drift: list[str] = []
    for f, meta in pre.items():
        sha, sz = _sha256_size(f)
        if sha != meta["sha256"] or sz != meta["size"]:
            drift.append(
                f"{f}: pre=(sz={meta['size']},sha={meta['sha256'][:12]}) "
                f"post=(sz={sz},sha={sha[:12]})"
            )
    return (len(drift) == 0, drift)


def _max_date_parquet(path: str, candidates: tuple[str, ...]) -> str | None:
    if not Path(path).exists():
        return None
    df = pd.read_parquet(path)
    for c in candidates:
        if c in df.columns:
            try:
                m = pd.to_datetime(df[c]).max()
                if pd.isna(m):
                    continue
                return str(m.date())
            except Exception:
                continue
    return None


def _max_date_csv(path: str, candidates: tuple[str, ...]) -> str | None:
    if not Path(path).exists():
        return None
    df = pd.read_csv(path)
    for c in candidates:
        if c in df.columns:
            try:
                m = pd.to_datetime(df[c]).max()
                if pd.isna(m):
                    continue
                return str(m.date())
            except Exception:
                continue
    return None


def _row_count(path: str) -> int | None:
    if not Path(path).exists():
        return None
    if path.endswith(".parquet"):
        return len(pd.read_parquet(path))
    if path.endswith(".csv"):
        return len(pd.read_csv(path))
    return None


def _archive(path: str, pre_max_date: str | None, pre_sha: str) -> str:
    """Q5: archive parquet BEFORE producer run, mode 0444."""
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    base = Path(path).name.replace(".parquet", "")
    md = pre_max_date or "unknown"
    short = pre_sha[:8] if pre_sha else "00000000"
    dest = ARCHIVE_DIR / f"{base}__pre_refresh__asof_{md}__sha256_{short}.parquet"
    if dest.exists():
        # Already archived for this exact pre-state — keep it.
        os.chmod(dest, 0o444)
        return str(dest)
    shutil.copy2(path, dest)
    os.chmod(dest, 0o444)
    return str(dest)


def _write_manifest(payload: dict) -> None:
    fd, tmp_path = tempfile.mkstemp(prefix=".manifest__", dir="output", suffix=".json")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        os.replace(tmp_path, MANIFEST_PATH)
    except Exception:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        raise


def _run_subprocess(cmd: list[str]) -> tuple[int, str, str]:
    print(f"[refresh] $ {' '.join(cmd)}", flush=True)
    p = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, "PYTHONPATH": "."})
    return p.returncode, p.stdout, p.stderr


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Decision Engine v1 — Stage 3 upstream-scanner-refresh orchestrator. "
            "Operational target (LCTD) resolves from --lctd-required (explicit "
            "wins) else from tools/_decision_target_date.derive_operational_target(). "
            "When invoked from .github/workflows/decision-engine-v1.yml the "
            "workflow's target_date input is propagated as --lctd-required so "
            "Stage 3 binds to the same date as the selected master-data-pull "
            "artifact (close-mode contract). No fallback weakening: "
            "extfeed-vs-LCTD and HB-post-vs-LCTD freshness gates unchanged."
        )
    )
    p.add_argument(
        "--lctd-required",
        type=lambda s: _dt.date.fromisoformat(s),
        default=None,
        metavar="YYYY-MM-DD",
        help=(
            "Operational target date. Explicit value wins over "
            "derive_operational_target(). When omitted Stage 3 uses the "
            "runtime-derived value (legacy shell behavior preserved)."
        ),
    )
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────


def main() -> int:
    global LCTD, LCTD_SOURCE
    args = _parse_args()
    if args.lctd_required is not None:
        cli_lctd = args.lctd_required.isoformat()
        if cli_lctd != LCTD_RUNTIME_DERIVED:
            print(
                f"[refresh] LCTD override: cli={cli_lctd} "
                f"runtime_derived={LCTD_RUNTIME_DERIVED} "
                f"(asof_mode={_OP_CTX.asof_mode}); "
                f"CLI wins (close-mode workflow contract)",
                flush=True,
            )
        LCTD = cli_lctd
        LCTD_SOURCE = "cli_override"
    print(f"[refresh] LCTD={LCTD} (source={LCTD_SOURCE})", flush=True)
    print(f"[refresh] LCTD_runtime_derived={LCTD_RUNTIME_DERIVED}", flush=True)
    print(f"[refresh] cwd={os.getcwd()}", flush=True)

    # ── pre-state capture ─────────────────────────────────────────────────
    print("[refresh] capturing pre-state on protected files …", flush=True)
    protected_pre = _capture_protected()
    print(f"  protected files captured: {len(protected_pre)}", flush=True)

    hb_pre_sha, hb_pre_sz = _sha256_size(HB_PARQUET)
    hb_pre_rows = _row_count(HB_PARQUET)
    hb_pre_max = _max_date_parquet(HB_PARQUET, ("bar_date", "breakout_bar_date", "date"))
    print(f"  HB pre: rows={hb_pre_rows} max_date={hb_pre_max} sha={hb_pre_sha[:12]}", flush=True)

    nyx_pre_sha, nyx_pre_sz = _sha256_size(NYXEXP_PARQUET)
    nyx_pre_rows = _row_count(NYXEXP_PARQUET)
    nyx_pre_max = _max_date_parquet(NYXEXP_PARQUET, ("date",))
    print(f"  nyxexp pre: rows={nyx_pre_rows} max_date={nyx_pre_max} sha={nyx_pre_sha[:12]}", flush=True)

    # ── Q4 pre-flight: extfeed master must reach LCTD ─────────────────────
    sys.path.insert(0, str(ROOT))
    from tools.decision_engine_v0_classification_backtest import build_daily_panel
    daily = build_daily_panel()
    extfeed_max = str(pd.to_datetime(daily["date"]).max().date())
    print(f"  extfeed daily max: {extfeed_max}", flush=True)
    if extfeed_max < LCTD:
        return _fail_run(
            "FAIL",
            f"extfeed_below_lctd: extfeed_max={extfeed_max} lctd={LCTD}",
            protected_pre,
            hb_status="SKIPPED",
            nyx_status="SKIPPED",
            hb_pre={"sha256": hb_pre_sha, "size": hb_pre_sz, "row_count": hb_pre_rows, "max_date": hb_pre_max},
            nyx_pre={"sha256": nyx_pre_sha, "size": nyx_pre_sz, "row_count": nyx_pre_rows, "max_date": nyx_pre_max},
            extfeed_max=extfeed_max,
        )

    # ── Q4 pre-flight: HB dependency staleness ────────────────────────────
    hb_dep_state: dict[str, dict] = {}
    hb_dep_fail: str | None = None
    for name, path in HB_DEPS.items():
        p = Path(path)
        if not p.exists():
            hb_dep_state[name] = {"path": path, "exists": False, "max_date": None}
            hb_dep_fail = f"hb_dep_missing: {name}={path}"
            continue
        # find max date if possible
        if path.endswith(".parquet"):
            mx = _max_date_parquet(path, ("date", "bar_date", "asof_date"))
        else:
            mx = _max_date_csv(path, ("date", "bar_date", "asof_date"))
        hb_dep_state[name] = {"path": path, "exists": True, "max_date": mx}
        # if dep has a date column and is below LCTD → fail-fast
        if mx is not None and mx < LCTD:
            hb_dep_fail = (
                f"producer_dependency_stale: horizontal_base "
                f"dep={name} dep_path={path} dep_max={mx} lctd={LCTD}"
            )
            # don't break — record all dep states for manifest

    if hb_dep_fail is not None:
        return _fail_run(
            "FAIL",
            hb_dep_fail,
            protected_pre,
            hb_status="SKIPPED_DEPENDENCY_STALE",
            nyx_status="SKIPPED",
            hb_pre={"sha256": hb_pre_sha, "size": hb_pre_sz, "row_count": hb_pre_rows, "max_date": hb_pre_max},
            nyx_pre={"sha256": nyx_pre_sha, "size": nyx_pre_sz, "row_count": nyx_pre_rows, "max_date": nyx_pre_max},
            extfeed_max=extfeed_max,
            hb_dep_state=hb_dep_state,
        )

    # ── HB archive + run ──────────────────────────────────────────────────
    print(f"[refresh] archiving HB target …", flush=True)
    hb_archive_path = _archive(HB_PARQUET, hb_pre_max, hb_pre_sha)
    hb_archive_sha, hb_archive_sz = _sha256_size(hb_archive_path)
    print(f"  HB archive: {hb_archive_path} (sha={hb_archive_sha[:12]} sz={hb_archive_sz})", flush=True)

    print(f"[refresh] running HB producer …", flush=True)
    hb_cmd = ["python", "tools/build_horizontal_base_event_v1.py"]
    hb_rc, hb_stdout, hb_stderr = _run_subprocess(hb_cmd)
    print(f"  HB exit={hb_rc}", flush=True)
    if hb_stderr:
        print(f"  HB stderr (last 500 chars): {hb_stderr[-500:]}", flush=True)

    hb_post_sha, hb_post_sz = _sha256_size(HB_PARQUET)
    hb_post_rows = _row_count(HB_PARQUET)
    hb_post_max = _max_date_parquet(HB_PARQUET, ("bar_date", "breakout_bar_date", "date"))
    print(f"  HB post: rows={hb_post_rows} max_date={hb_post_max} sha={hb_post_sha[:12]}", flush=True)

    hb_status = "PASS"
    hb_fail_reason = None
    if hb_rc != 0:
        hb_status = "FAIL"
        hb_fail_reason = f"producer_failed: horizontal_base exit={hb_rc}"
    elif hb_post_max is None:
        hb_status = "FAIL"
        hb_fail_reason = "producer_failed: horizontal_base post_max_date_unreadable"
    elif hb_post_max < LCTD:
        hb_status = "FAIL"
        hb_fail_reason = (
            f"producer_did_not_reach_lctd: horizontal_base "
            f"post_max={hb_post_max} lctd={LCTD}"
        )

    hb_record = {
        "pre_path": HB_PARQUET,
        "post_path": HB_PARQUET,
        "pre_sha256": hb_pre_sha,
        "post_sha256": hb_post_sha,
        "pre_size": hb_pre_sz,
        "post_size": hb_post_sz,
        "pre_row_count": hb_pre_rows,
        "post_row_count": hb_post_rows,
        "pre_max_date": hb_pre_max,
        "post_max_date": hb_post_max,
        "archive_path": hb_archive_path,
        "archive_sha256": hb_archive_sha,
        "archive_size": hb_archive_sz,
        "command": " ".join(hb_cmd),
        "exit_code": hb_rc,
        "status": hb_status,
        "fail_reason": hb_fail_reason,
        "ahead_of_extfeed": (hb_post_max is not None and hb_post_max > extfeed_max),
    }

    # ── Q2: HB FAIL → SKIP nyxexp ─────────────────────────────────────────
    if hb_status == "FAIL":
        return _fail_run(
            "FAIL",
            hb_fail_reason,
            protected_pre,
            hb_record=hb_record,
            nyx_status="SKIPPED",
            nyx_pre={"sha256": nyx_pre_sha, "size": nyx_pre_sz, "row_count": nyx_pre_rows, "max_date": nyx_pre_max},
            extfeed_max=extfeed_max,
            hb_dep_state=hb_dep_state,
        )

    # ── nyxexp dependency pre-flight ──────────────────────────────────────
    nyx_dep_state: dict[str, dict] = {}
    nyx_dep_fail: str | None = None
    for name, path in NYXEXP_DEPS.items():
        p = Path(path)
        if not p.exists():
            nyx_dep_state[name] = {"path": path, "exists": False, "max_date": None}
            nyx_dep_fail = f"nyxexp_dep_missing: {name}={path}"
            continue
        mx = None
        if path.endswith(".parquet"):
            mx = _max_date_parquet(path, ("date", "bar_date"))
        nyx_dep_state[name] = {"path": path, "exists": True, "max_date": mx}

    if nyx_dep_fail is not None:
        return _partial_fail_run(
            nyx_dep_fail,
            protected_pre,
            hb_record,
            nyx_status="SKIPPED_DEPENDENCY_MISSING",
            nyx_pre={"sha256": nyx_pre_sha, "size": nyx_pre_sz, "row_count": nyx_pre_rows, "max_date": nyx_pre_max},
            extfeed_max=extfeed_max,
            hb_dep_state=hb_dep_state,
            nyx_dep_state=nyx_dep_state,
        )

    # ── nyxexp archive + run ──────────────────────────────────────────────
    print(f"[refresh] archiving nyxexp target …", flush=True)
    nyx_archive_path = _archive(NYXEXP_PARQUET, nyx_pre_max, nyx_pre_sha)
    nyx_archive_sha, nyx_archive_sz = _sha256_size(nyx_archive_path)
    print(f"  nyxexp archive: {nyx_archive_path} (sha={nyx_archive_sha[:12]} sz={nyx_archive_sz})", flush=True)

    print(f"[refresh] running nyxexp producer …", flush=True)
    nyx_cmd = [
        "python", "-m", "nyxexpansion.tools.rebuild_dataset_delta_intraday",
        "--date", LCTD,
    ]
    nyx_rc, nyx_stdout, nyx_stderr = _run_subprocess(nyx_cmd)
    print(f"  nyxexp exit={nyx_rc}", flush=True)
    if nyx_stderr:
        print(f"  nyxexp stderr (last 500 chars): {nyx_stderr[-500:]}", flush=True)

    nyx_post_sha, nyx_post_sz = _sha256_size(NYXEXP_PARQUET)
    nyx_post_rows = _row_count(NYXEXP_PARQUET)
    nyx_post_max = _max_date_parquet(NYXEXP_PARQUET, ("date",))
    print(f"  nyxexp post: rows={nyx_post_rows} max_date={nyx_post_max} sha={nyx_post_sha[:12]}", flush=True)

    nyx_status = "PASS"
    nyx_fail_reason = None
    if nyx_rc != 0:
        nyx_status = "FAIL"
        nyx_fail_reason = f"producer_failed: nyxexp exit={nyx_rc}"
    elif nyx_post_max is None:
        nyx_status = "FAIL"
        nyx_fail_reason = "producer_failed: nyxexp post_max_date_unreadable"
    elif nyx_post_max < LCTD:
        nyx_status = "FAIL"
        nyx_fail_reason = (
            f"producer_did_not_reach_lctd: nyxexp "
            f"post_max={nyx_post_max} lctd={LCTD}"
        )

    nyx_record = {
        "pre_path": NYXEXP_PARQUET,
        "post_path": NYXEXP_PARQUET,
        "pre_sha256": nyx_pre_sha,
        "post_sha256": nyx_post_sha,
        "pre_size": nyx_pre_sz,
        "post_size": nyx_post_sz,
        "pre_row_count": nyx_pre_rows,
        "post_row_count": nyx_post_rows,
        "pre_max_date": nyx_pre_max,
        "post_max_date": nyx_post_max,
        "archive_path": nyx_archive_path,
        "archive_sha256": nyx_archive_sha,
        "archive_size": nyx_archive_sz,
        "command": " ".join(nyx_cmd),
        "exit_code": nyx_rc,
        "status": nyx_status,
        "fail_reason": nyx_fail_reason,
        "ahead_of_extfeed": (nyx_post_max is not None and nyx_post_max > extfeed_max),
    }

    # ── post-state protected check ────────────────────────────────────────
    print("[refresh] verifying protected files byte-equal …", flush=True)
    ok, drift = _verify_protected(protected_pre)
    print(f"  protected byte-equal: {ok}", flush=True)
    if not ok:
        for d in drift:
            print(f"    DRIFT: {d}", flush=True)

    # ── final verdict ─────────────────────────────────────────────────────
    if not ok:
        verdict = "FAIL"
        verdict_reason = f"protected_file_modified: {len(drift)} files drifted"
    elif hb_status == "PASS" and nyx_status == "PASS":
        verdict = "PASS"
        verdict_reason = None
    elif hb_status == "PASS" and nyx_status == "FAIL":
        verdict = "PARTIAL_FAIL"
        verdict_reason = nyx_fail_reason
    else:
        verdict = "FAIL"
        verdict_reason = nyx_fail_reason or hb_fail_reason

    payload = {
        "refresh_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "lctd": LCTD,
        "lctd_source": LCTD_SOURCE,
        "lctd_runtime_derived": LCTD_RUNTIME_DERIVED,
        "extfeed_max_date": extfeed_max,
        "verdict": verdict,
        "verdict_reason": verdict_reason,
        "horizontal_base": hb_record,
        "nyxexpansion": nyx_record,
        "hb_dep_state": hb_dep_state,
        "nyx_dep_state": nyx_dep_state,
        "protected_files_byte_equal": ok,
        "protected_drift": drift,
        "spec_path": "memory/decision_engine_v1_upstream_scanner_refresh_spec.md",
    }
    _write_manifest(payload)
    print(f"[refresh] manifest written: {MANIFEST_PATH}", flush=True)
    print(f"[refresh] VERDICT: {verdict}", flush=True)
    if verdict_reason:
        print(f"  reason: {verdict_reason}", flush=True)

    return 0 if verdict == "PASS" else 1


def _fail_run(
    verdict: str,
    reason: str,
    protected_pre: dict,
    hb_status: str = "FAIL",
    nyx_status: str = "SKIPPED",
    hb_record: dict | None = None,
    hb_pre: dict | None = None,
    nyx_pre: dict | None = None,
    extfeed_max: str | None = None,
    hb_dep_state: dict | None = None,
    nyx_dep_state: dict | None = None,
) -> int:
    print(f"[refresh] FAIL: {reason}", flush=True)

    # verify protected files unchanged on FAIL path
    ok, drift = _verify_protected(protected_pre)
    print(f"[refresh] protected byte-equal post-FAIL: {ok}", flush=True)
    if not ok:
        for d in drift:
            print(f"    DRIFT: {d}", flush=True)

    payload = {
        "refresh_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "lctd": LCTD,
        "lctd_source": LCTD_SOURCE,
        "lctd_runtime_derived": LCTD_RUNTIME_DERIVED,
        "extfeed_max_date": extfeed_max,
        "verdict": verdict,
        "verdict_reason": reason,
        "horizontal_base": hb_record if hb_record is not None else {
            "status": hb_status, "pre": hb_pre,
        },
        "nyxexpansion": {"status": nyx_status, "pre": nyx_pre},
        "hb_dep_state": hb_dep_state,
        "nyx_dep_state": nyx_dep_state,
        "protected_files_byte_equal": ok,
        "protected_drift": drift,
        "spec_path": "memory/decision_engine_v1_upstream_scanner_refresh_spec.md",
    }
    _write_manifest(payload)
    print(f"[refresh] manifest written: {MANIFEST_PATH}", flush=True)
    print(f"[refresh] VERDICT: {verdict}", flush=True)
    return 1


def _partial_fail_run(
    reason: str,
    protected_pre: dict,
    hb_record: dict,
    nyx_status: str,
    nyx_pre: dict,
    extfeed_max: str,
    hb_dep_state: dict,
    nyx_dep_state: dict,
) -> int:
    print(f"[refresh] PARTIAL_FAIL: {reason}", flush=True)
    ok, drift = _verify_protected(protected_pre)
    payload = {
        "refresh_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "lctd": LCTD,
        "lctd_source": LCTD_SOURCE,
        "lctd_runtime_derived": LCTD_RUNTIME_DERIVED,
        "extfeed_max_date": extfeed_max,
        "verdict": "PARTIAL_FAIL",
        "verdict_reason": reason,
        "horizontal_base": hb_record,
        "nyxexpansion": {"status": nyx_status, "pre": nyx_pre, "fail_reason": reason},
        "hb_dep_state": hb_dep_state,
        "nyx_dep_state": nyx_dep_state,
        "protected_files_byte_equal": ok,
        "protected_drift": drift,
        "spec_path": "memory/decision_engine_v1_upstream_scanner_refresh_spec.md",
    }
    _write_manifest(payload)
    print(f"[refresh] manifest written: {MANIFEST_PATH}", flush=True)
    print(f"[refresh] VERDICT: PARTIAL_FAIL", flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
