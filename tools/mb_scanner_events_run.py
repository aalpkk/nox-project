"""mb_scanner event-log extractor — emits historical state-transition events.

Phase 0 prereq for Phase 1 backtest. Walks each ticker's resampled panel
once per family, finds all quartets, emits 1-3 events per quartet:
above_mb_birth (HH/HL+pivot_n), mit_touch_first, retest_bounce_first.

Output: output/mb_scanner_events_<family>.parquet (8 files when all
families requested).

Usage:
    python tools/mb_scanner_events_run.py [--families mb_5h mb_1d ...]
                                          [--tickers MGROS HDFGS]
                                          [--min-coverage 0.0]
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

from mb_scanner.events import EVENT_TYPES, extract_events
from mb_scanner.schema import FAMILIES


def _summarize(df: pd.DataFrame, family: str) -> None:
    if df.empty:
        print(f"[{family}] 0 events.")
        return
    print()
    print(f"=== {family} | N={len(df)} ===")
    by_type = df["event_type"].value_counts().reindex(EVENT_TYPES, fill_value=0)
    for et in EVENT_TYPES:
        print(f"  {et:<25s} {int(by_type.get(et, 0)):>6d}")
    by_ticker = df.groupby("ticker").size()
    print(f"  unique tickers           {by_ticker.shape[0]:>6d}")
    if "event_bar_date" in df.columns and not df["event_bar_date"].isna().all():
        first = pd.Timestamp(df["event_bar_date"].min()).strftime("%Y-%m-%d")
        last = pd.Timestamp(df["event_bar_date"].max()).strftime("%Y-%m-%d")
        print(f"  event window             {first} → {last}")
    if "concurrent_quartets" in df.columns:
        cq = df["concurrent_quartets"]
        print(f"  concurrent_quartets      mean={cq.mean():.2f}  max={int(cq.max())}")
    if "pivot_confirm_lag_bars" in df.columns:
        lag = df["pivot_confirm_lag_bars"]
        share = float((lag > 0).mean())
        print(f"  HH→event lag>0           {share*100:.1f}% of events  (max={int(lag.max())})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", nargs="*", default=list(FAMILIES),
                    help="Subset of families to extract (default: all 8).")
    ap.add_argument("--tickers", nargs="*", default=None,
                    help="Subset tickers (default: full universe).")
    ap.add_argument("--min-coverage", type=float, default=0.0,
                    help="Minimum bar-coverage filter (default 0.0 = all).")
    args = ap.parse_args()

    bad = [f for f in args.families if f not in FAMILIES]
    if bad:
        print(f"[error] unknown families: {bad}; valid={list(FAMILIES)}", file=sys.stderr)
        return 2

    print(f"[events] families={args.families}")
    print(f"[events] tickers={'all' if args.tickers is None else len(args.tickers)}")
    t0 = time.time()
    out_map = extract_events(
        families=args.families,
        tickers=args.tickers,
        min_coverage=args.min_coverage,
    )
    elapsed = time.time() - t0

    grand_total = 0
    for fam in args.families:
        df = out_map.get(fam, pd.DataFrame())
        _summarize(df, fam)
        grand_total += 0 if df is None else len(df)

    print()
    print(f"[events] done. families={len(args.families)}  events={grand_total:,}  elapsed={elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
