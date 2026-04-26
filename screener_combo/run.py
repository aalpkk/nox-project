"""End-to-end screener combo backtest on nox_intraday_v1 daily bars.

Phases:
  1. Load panel (TRAIN+VAL only) + XU100 benchmark.
  2. Scan 3 signals across the panel → trigger table.
  3. Compute forward returns (T+1 open → T+H close) at H ∈ {5, 10, 20}.
  4. Build per-signal & combination cohort metrics.
  5. Build daily top-K aggregation for each gate.
  6. Write CSVs under output/.

Usage:
    python -m screener_combo.run --tag v1
    python -m screener_combo.run --tag v1 --tickers 30   # smoke
"""

from __future__ import annotations
import argparse
import time
from pathlib import Path

import pandas as pd

from screener_combo.data_prep import load_panel_daily, load_xu100_close, trainval_window
from screener_combo.scan import scan_panel
from screener_combo.backtest import (
    add_forward_returns,
    attach_returns,
    add_combinations,
    all_cohorts,
    daily_topk_metrics,
    HORIZONS,
)


OUTPUT_DIR = Path("output")
NAMESPACE = "screener_combo_v1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="default")
    ap.add_argument("--tickers", type=int, default=None,
                    help="Smoke-mode: only scan first N tickers.")
    ap.add_argument("--min-coverage", type=float, default=0.50)
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] Loading panel...")
    panel = load_panel_daily(min_coverage=args.min_coverage)
    bench = load_xu100_close()
    start, end = trainval_window()
    print(f"  panel: {len(panel):,} rows, {panel.ticker.nunique()} tickers, "
          f"{panel.date.min().date()} → {panel.date.max().date()}")
    print(f"  bench: {len(bench)} bars")
    print(f"  TRAIN+VAL window: {start.date()} → {end.date()}")

    if args.tickers:
        keep = sorted(panel.ticker.unique())[: args.tickers]
        panel = panel[panel.ticker.isin(keep)]
        print(f"  smoke mode: limiting to {len(keep)} tickers")

    print(f"[{time.strftime('%H:%M:%S')}] Scanning signals...")
    triggers = scan_panel(panel, bench, start=start, end=end)
    print(f"  trigger rows: {len(triggers):,}")
    print(f"  per-signal totals: "
          f"RT={int(triggers.regime_trig.sum())}  "
          f"NWeekly={int(triggers.weekly_trig.sum())}  "
          f"AS={int(triggers.alsat_trig.sum())}")

    print(f"[{time.strftime('%H:%M:%S')}] Computing forward returns...")
    panel_in_window = panel[(panel.date >= start) & (panel.date <= end + pd.Timedelta(days=30))]
    fwd = add_forward_returns(panel_in_window, HORIZONS)
    print(f"  fwd rows: {len(fwd):,}")

    table = attach_returns(triggers, fwd)
    table = add_combinations(table)

    # Persist trigger+fwd table
    out_long = OUTPUT_DIR / f"{NAMESPACE}_triggers_{args.tag}.parquet"
    table.to_parquet(out_long, index=False)
    print(f"  → {out_long}")

    print(f"[{time.strftime('%H:%M:%S')}] Building cohort metrics...")
    cohorts = all_cohorts(table)
    cohorts_path = OUTPUT_DIR / f"{NAMESPACE}_cohorts_{args.tag}.csv"
    cohorts.to_csv(cohorts_path, index=False, float_format="%.3f")
    print(f"  → {cohorts_path}")

    # Daily top-K
    print(f"[{time.strftime('%H:%M:%S')}] Daily top-{args.top_k} aggregation...")
    rows = []
    gates = ["regime_trig", "weekly_trig", "alsat_trig",
             "vote2", "vote3", "cascade_RT_AS", "cascade_RT_NW", "cascade_NW_AS"]
    for gate in gates:
        for h in HORIZONS:
            rows.append(daily_topk_metrics(table, gate, h, args.top_k))
    daily_df = pd.DataFrame(rows)
    daily_path = OUTPUT_DIR / f"{NAMESPACE}_dailytop{args.top_k}_{args.tag}.csv"
    daily_df.to_csv(daily_path, index=False, float_format="%.3f")
    print(f"  → {daily_path}")

    elapsed = time.time() - t0
    print(f"[{time.strftime('%H:%M:%S')}] DONE in {elapsed:.1f}s")
    print()
    print("=== Cohort metrics (compact view) ===")
    print(cohorts.to_string(index=False))
    print()
    print(f"=== Daily top-{args.top_k} ===")
    print(daily_df.to_string(index=False))


if __name__ == "__main__":
    main()
