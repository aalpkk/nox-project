"""channel_break runner — descriptive parallel-channel scan over 5h/1d/1w/1M.

Writes per-family channel_break_<fam>.parquet (accepted) +
pending_triangle_<fam>.parquet (parallelism-fail fits forwarded to
triangle workstream).

Usage:
    python tools/channel_break_scan_live.py [--asof "2026-04-29"]
                                            [--families ch_5h ch_1d ...]
                                            [--tickers ASELS KRDMD]
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

from channel_break.engine import scan
from channel_break.schema import FAMILIES


def _summarize(family: str, ch_df: pd.DataFrame, pd_df: pd.DataFrame) -> None:
    print()
    print(f"=== {family} | channels={len(ch_df)}  pending_triangle={len(pd_df)} ===")
    if ch_df.empty:
        print("  (no channels)")
    else:
        states = ch_df["signal_state"].value_counts().to_dict()
        slopes = ch_df["slope_class"].value_counts().to_dict()
        n_tier_a = int(ch_df["tier_a"].sum())
        print(f"  states: {states}")
        print(f"  slope_class: {slopes}")
        print(f"  tier_a: {n_tier_a}/{len(ch_df)}")
        cols = [
            "ticker", "signal_state", "slope_class", "tier_a",
            "channel_width_pct", "n_swing_touches",
            "upper_slope_pct_per_bar", "lower_slope_pct_per_bar",
            "fit_quality", "asof_close", "breakout_age_bars",
        ]
        avail = [c for c in cols if c in ch_df.columns]
        print()
        print(ch_df[avail].head(15).to_string(index=False))

    if not pd_df.empty:
        kinds = pd_df["triangle_kind_hint"].value_counts().to_dict()
        print(f"  pending_triangle kinds: {kinds}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asof", default=None,
                    help="ISO timestamp (Europe/Istanbul). Default = last bar in each panel.")
    ap.add_argument("--families", nargs="*", default=list(FAMILIES),
                    help=f"Families to run; default = all {list(FAMILIES)}")
    ap.add_argument("--tickers", nargs="*", default=None,
                    help="Subset tickers (default: full universe).")
    ap.add_argument("--min-coverage", type=float, default=0.0)
    args = ap.parse_args()

    print(f"[run] asof={args.asof or 'latest'}  families={args.families}  "
          f"tickers={'all' if args.tickers is None else args.tickers}")

    t0 = time.time()
    out = scan(
        families=args.families,
        tickers=args.tickers,
        asof=args.asof,
        min_coverage=args.min_coverage,
        write_parquet=True,
    )
    elapsed = time.time() - t0
    total_ch = sum(len(v["channels"]) for v in out.values())
    total_pd = sum(len(v["pending"]) for v in out.values())
    print(f"\n[run] done in {elapsed:.1f}s  channels={total_ch}  "
          f"pending_triangle={total_pd}")

    for f in args.families:
        v = out.get(f, {"channels": pd.DataFrame(), "pending": pd.DataFrame()})
        _summarize(f, v["channels"], v["pending"])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
