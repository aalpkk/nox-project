"""V1.3.1+ horizontal_base — historical trigger count distribution.

Reads live production gates from `scanner.triggers.horizontal_base`
(currently SCANNER 1.4.0 / FEATURE 1.3.0, body floor 0.35, slope cap
0.0050/day, both-dim).

For each ticker and each daily bar (asof) in the bundle window, count how often
the trigger fires. Reuses _compute_indicators (heavy) once per ticker; only the
state classification iterates per asof.

Output:
  - per-day trigger counts
  - per-day pre_breakout counts
  - per-day extended counts
  - histogram + summary stats
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data import intraday_1h
from scanner.triggers.horizontal_base import (
    BB_LENGTH,
    MAX_SQUEEZE_AGE_BARS,
    SLOPE_REJECT_PER_DAY,
    WIDTH_PCT_REJECT,
    _box_from_squeeze,
    _classify_state,
    _compute_indicators,
    _find_squeeze_runs,
    _line_fit,
)


def historical_states(daily_df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Return per-bar state for one ticker: date, state ∈ {none, pre, trig, ext}."""
    if len(daily_df) < BB_LENGTH * 3:
        return pd.DataFrame()
    df = _compute_indicators(daily_df)
    runs = _find_squeeze_runs(df["squeeze"])
    if not runs:
        return pd.DataFrame()

    out_rows = []
    n = len(df)
    closes = df["close"].values
    dates = df.index

    # For each asof_idx, find the latest squeeze run ending ≤ asof_idx
    # Pre-sort runs by end-bar
    runs_sorted = sorted(runs, key=lambda r: r[1])
    run_ends = np.array([r[1] for r in runs_sorted])

    for asof_idx in range(BB_LENGTH * 2, n):
        # latest run with sq_e <= asof_idx
        pos = int(np.searchsorted(run_ends, asof_idx, side="right") - 1)
        if pos < 0:
            continue
        sq_s, sq_e, _ = runs_sorted[pos]
        if asof_idx - sq_e > MAX_SQUEEZE_AGE_BARS:
            continue

        base_window = df.iloc[sq_s: sq_e + 1]
        base_slope, _ = _line_fit(base_window["close"].values)
        if abs(base_slope) > SLOPE_REJECT_PER_DAY:
            continue
        res_slope, _ = _line_fit(base_window["high"].values)
        if abs(res_slope) > SLOPE_REJECT_PER_DAY:
            continue

        box_top, hard_resistance, box_bot = _box_from_squeeze(df, sq_s, sq_e)
        if not (box_top > box_bot > 0):
            continue

        asof_close = float(closes[asof_idx])
        width_pct = (box_top - box_bot) / asof_close if asof_close > 0 else float("inf")
        if width_pct > WIDTH_PCT_REJECT:
            continue

        state, _ = _classify_state(df, asof_idx, sq_s, sq_e, box_top, hard_resistance)
        if state == "none":
            continue
        out_rows.append({"ticker": ticker, "date": dates[asof_idx], "state": state})

    return pd.DataFrame(out_rows)


def main():
    t0 = time.time()
    print("[1/3] loading master parquet …", flush=True)
    bars = intraday_1h.load_intraday(
        tickers=None, start=None, end=None, min_coverage=0.0,
    )
    print(f"  loaded {len(bars):,} bars / {bars['ticker'].nunique()} tickers in {time.time()-t0:.1f}s", flush=True)

    print("[2/3] resampling to daily panel …", flush=True)
    t1 = time.time()
    daily = intraday_1h.daily_resample(bars)
    print(f"  daily panel: {len(daily):,} rows in {time.time()-t1:.1f}s", flush=True)

    print("[3/3] historical state-scan per ticker …", flush=True)
    t2 = time.time()
    all_rows = []
    tickers = daily["ticker"].unique()
    for i, t in enumerate(tickers, 1):
        g = daily[daily["ticker"] == t].sort_values("date")
        idx = pd.DatetimeIndex(pd.to_datetime(g["date"]))
        df = g[["open", "high", "low", "close", "volume"]].set_index(idx)
        sub = historical_states(df, str(t))
        if not sub.empty:
            all_rows.append(sub)
        if i % 100 == 0:
            print(f"  [{i}/{len(tickers)}] elapsed={time.time()-t2:.1f}s", flush=True)
    print(f"  per-ticker scan: {time.time()-t2:.1f}s", flush=True)

    if not all_rows:
        print("!! no historical states found", flush=True)
        return 1

    states = pd.concat(all_rows, ignore_index=True)
    states["date"] = pd.to_datetime(states["date"]).dt.normalize()

    # ---- per-day counts -----------------------------------------------------
    daily_counts = (
        states.groupby(["date", "state"]).size().unstack(fill_value=0)
        .reindex(columns=["trigger", "pre_breakout", "extended"], fill_value=0)
    )
    daily_counts.to_csv("output/scanner_v1_trigger_history_per_day.csv")

    print()
    print("=" * 70)
    print("HISTORICAL TRIGGER DISTRIBUTION (V1.2.3 horizontal_base)")
    print("=" * 70)
    print(f"window: {daily_counts.index.min().date()} → {daily_counts.index.max().date()}  "
          f"({len(daily_counts)} trading days)")
    print(f"total emissions: trigger={daily_counts['trigger'].sum():,}  "
          f"pre={daily_counts['pre_breakout'].sum():,}  "
          f"ext={daily_counts['extended'].sum():,}")
    print()
    print("per-day counts (mean / median / p25 / p75 / p95 / max):")
    for col in ["trigger", "pre_breakout", "extended"]:
        s = daily_counts[col]
        print(f"  {col:<14s}  mean={s.mean():.2f}  med={int(s.median())}  "
              f"p25={int(s.quantile(0.25))}  p75={int(s.quantile(0.75))}  "
              f"p95={int(s.quantile(0.95))}  max={int(s.max())}")

    print()
    print("trigger-count histogram (per day):")
    bins = [0, 1, 2, 3, 5, 10, 20, 50, 200]
    hist = pd.cut(daily_counts["trigger"], bins=bins, right=False).value_counts().sort_index()
    for interval, cnt in hist.items():
        pct = cnt / len(daily_counts) * 100
        print(f"  {str(interval):<14s} {cnt:>4d} days ({pct:5.1f}%)")

    print()
    print("monthly trigger totals (last 24 months):")
    monthly = daily_counts["trigger"].resample("ME").sum().tail(24)
    for d, v in monthly.items():
        bar = "#" * min(int(v / 5), 60)
        print(f"  {d.strftime('%Y-%m')}  {int(v):>4d}  {bar}")

    print()
    print(f"wrote: output/scanner_v1_trigger_history_per_day.csv")
    print(f"total elapsed: {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
