"""mb_scanner Phase 1 Event-Quality Diagnostic — runner.

Reads the per-family event-log parquets produced by
`tools/mb_scanner_events_run.py`, joins forward outcomes / B0 / B2 baselines,
aggregates per (family, event_type, horizon) cohort, and applies the locked
acceptance gate (PASS ≥3/5 horizons clear: raw_ret_mean > B0 AND PF > 1.57).

Pre-registration: the locked spec at
`memory/mb_scanner_phase1_event_quality.md` allows ONE authorized run.
By default this script is `--dry-run` (writes nothing, prints scaffold
status only); pass `--go` to fire the single authorized run.

Usage:
    # scaffold validation (default — no parquets written, brief sanity print)
    python tools/mb_scanner_phase1_run.py [--families mb_5h ...]

    # the single authorized run (writes outputs, prints full report)
    python tools/mb_scanner_phase1_run.py --go [--families mb_5h ...]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from mb_scanner.events import EVENT_TYPES
from mb_scanner.phase1 import (
    HORIZONS,
    PHASE1_SPEC_ID,
    Q60_REBOUND_PF,
    run_phase1,
)
from mb_scanner.schema import FAMILIES


def _print_verdict_summary(verdicts: pd.DataFrame, metrics: pd.DataFrame) -> None:
    if verdicts.empty:
        print("[verdicts] empty.")
        return
    print()
    print("=== Phase 1 Verdicts (per cohort) ===")
    counts = verdicts["verdict"].value_counts().to_dict()
    print(f"  totals: PASS={counts.get('PASS', 0)}  WEAK={counts.get('WEAK', 0)}  "
          f"FAIL={counts.get('FAIL', 0)}  thin={counts.get('thin', 0)}")
    print()
    fmt = "{:<6} {:<22} {:>7}  {:>7}/{:<2}  {:<14}  {:<6}"
    print(fmt.format("family", "event_type", "N", "clear", "5", "horizons", "verdict"))
    print(fmt.format("-" * 6, "-" * 22, "-" * 7, "-" * 7, "--", "-" * 14, "-" * 6))
    for _, r in verdicts.sort_values(["family", "event_type"]).iterrows():
        print(fmt.format(
            r["family"], r["event_type"], int(r["N_total"]),
            int(r["horizons_clear"]), int(r["horizons_eval"]),
            r["cleared_horizons"] or "—",
            r["verdict"],
        ))


def _print_cohort_metrics(metrics: pd.DataFrame) -> None:
    if metrics.empty:
        print("[metrics] empty.")
        return
    print()
    print("=== Cohort metrics (raw_ret_mean / PF / B0_mean / N_eval) ===")
    fmt = "{:<6} {:<22} {:>3}  {:>6}  {:>+8.4f}  {:>6.2f}  {:>+8.4f}  {:>+8.4f}  {:>+6.2f}"
    head = "{:<6} {:<22} {:>3}  {:>6}  {:>8}  {:>6}  {:>8}  {:>8}  {:>6}".format(
        "family", "event_type", "h", "N", "raw_mu", "PF", "B0_mu", "B2_mu", "MFE_R"
    )
    print(head)
    print("-" * len(head))
    for _, r in metrics.sort_values(["family", "event_type", "horizon"]).iterrows():
        if r.get("N_eval", 0) <= 0:
            print(f"{r['family']:<6} {r['event_type']:<22} {int(r['horizon']):>3}  {int(r['N_total']):>6}  (no eval)")
            continue
        pf = r["PF"]
        pf_str = "inf" if pf == float("inf") else f"{pf:6.2f}"
        print("{:<6} {:<22} {:>3}  {:>6}  {:>+8.4f}  {:>6}  {:>+8.4f}  {:>+8.4f}  {:>+6.2f}".format(
            r["family"], r["event_type"], int(r["horizon"]), int(r["N_eval"]),
            r["raw_ret_mean"], pf_str, r["B0_mean"], r["B2_mean"], r["MFE_R_mean"],
        ))


def _scaffold_check(args) -> int:
    """Dry-run sanity: confirm all event-log parquets exist + load shape."""
    print(f"[dry-run] spec_id={PHASE1_SPEC_ID}")
    print(f"[dry-run] horizons={list(HORIZONS)}  B1_PF={Q60_REBOUND_PF}")
    print(f"[dry-run] families={args.families}")
    print(f"[dry-run] event_types={list(EVENT_TYPES)}  (extended is reference-only, not a Phase 1 cohort)")
    out_dir = Path("output")
    missing = []
    counts = []
    for fam in args.families:
        p = out_dir / f"mb_scanner_events_{fam}.parquet"
        if not p.exists():
            missing.append(fam)
        else:
            df = pd.read_parquet(p, columns=["ticker", "event_type"])
            counts.append((fam, len(df), df["event_type"].value_counts().to_dict()))
    if missing:
        print(f"[dry-run][error] missing event-log parquets: {missing}")
        print("[dry-run]         run: python tools/mb_scanner_events_run.py")
        return 2
    print()
    print("=== Phase 0 event-log inventory ===")
    print("{:<6} {:>9}  birth/touch/retest".format("family", "N"))
    for fam, n, by_et in counts:
        b = by_et.get("above_mb_birth", 0)
        m = by_et.get("mit_touch_first", 0)
        r = by_et.get("retest_bounce_first", 0)
        print(f"{fam:<6} {n:>9,}  {b:>5,} / {m:>5,} / {r:>5,}")
    print()
    print("[dry-run] scaffold OK. To fire the single authorized run, pass --go.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", nargs="*", default=list(FAMILIES))
    ap.add_argument("--tickers", nargs="*", default=None)
    ap.add_argument("--min-coverage", type=float, default=0.0)
    ap.add_argument("--go", action="store_true",
                    help="Fire the single authorized Phase 1 run (writes outputs).")
    ap.add_argument("--no-write", action="store_true",
                    help="Even with --go, suppress parquet/csv writes (reports only).")
    args = ap.parse_args()

    bad = [f for f in args.families if f not in FAMILIES]
    if bad:
        print(f"[error] unknown families: {bad}; valid={list(FAMILIES)}", file=sys.stderr)
        return 2

    if not args.go:
        return _scaffold_check(args)

    print(f"[phase1] spec_id={PHASE1_SPEC_ID}  ::  AUTHORIZED RUN")
    print(f"[phase1] families={args.families}  horizons={list(HORIZONS)}  B1_PF={Q60_REBOUND_PF}")
    t0 = time.time()
    result = run_phase1(
        families=args.families,
        tickers=args.tickers,
        min_coverage=args.min_coverage,
        write_outputs=not args.no_write,
    )
    elapsed = time.time() - t0

    metrics = result["cohort_metrics"]
    verdicts = result["verdicts"]

    _print_verdict_summary(verdicts, metrics)
    _print_cohort_metrics(metrics)

    print()
    print(f"[phase1] done. cohorts={len(verdicts)}  metric-rows={len(metrics)}  elapsed={elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
