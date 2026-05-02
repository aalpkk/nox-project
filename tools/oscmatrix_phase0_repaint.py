"""Phase 0 — repaint diagnostic.

For each TV signal (arrows + HWO dots) at bar B, find the local price
extremum in window [B-W, B+W] and compute lag = B - argmin/argmax_bar.

Interpretation:
  lag distribution mode = 0  AND saturated at ±0  → repaint or pivot-perfect
                                                    (TV used future bars OR
                                                     signal exactly at pivot)
  lag mode = +1..+3, negative tail rare           → non-repaint, realistic latency
  lag negative (signal BEFORE pivot) common       → predictive (rare for technical
                                                    indicators; would be impressive)

Also report saturation at window edge (lag = ±W). If extremum is at edge,
the true pivot may lie outside our window — interpret with caution.

For AL-side signals (Reversal Up, HWO Up, Oversold HWO Up): look for LOW.
For SAT-side signals (Reversal Down, HWO Down, Overbought HWO Down): look for HIGH.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from oscmatrix.validate import load_tv_csv  # noqa: E402

CSV_PATHS = {
    "THYAO": "/Users/alpkarakasli/Library/Mobile Documents/com~apple~CloudDocs/Downloads1/BIST_THYAO, 1D.csv",
    "GARAN": "/Users/alpkarakasli/Library/Mobile Documents/com~apple~CloudDocs/Downloads1/New Folder With Items 3/BIST_GARAN, 1D.csv",
    "EREGL": "/Users/alpkarakasli/Library/Mobile Documents/com~apple~CloudDocs/Downloads1/New Folder With Items 3/BIST_EREGL, 1D.csv",
    "ASELS": "/Users/alpkarakasli/Library/Mobile Documents/com~apple~CloudDocs/Downloads1/New Folder With Items 3/BIST_ASELS, 1D.csv",
}

WINDOW = 5  # bars on each side
WIDE = 10   # second window for robustness check

AL_SIDE = {
    "Reversal Up +":         "low",
    "Reversal Up -":         "low",
    "HWO Up":                "low",
    "Oversold HWO Up":       "low",
}
SAT_SIDE = {
    "Reversal Down +":       "high",
    "Reversal Down -":       "high",
    "HWO Down":              "high",
    "Overbought HWO Down":   "high",
}
ALL_KINDS = {**AL_SIDE, **SAT_SIDE}


def lag_for_signal(bar_loc: int, prices: pd.Series, window: int, side: str) -> tuple[int, bool]:
    """Return (lag, saturated) where lag = bar_loc - extremum_bar.

    side='low'  → extremum = argmin
    side='high' → extremum = argmax
    saturated = True if extremum is at the edge of the window.
    """
    n = len(prices)
    lo = max(0, bar_loc - window)
    hi = min(n, bar_loc + window + 1)
    sub = prices.iloc[lo:hi]
    if side == "low":
        ext_offset = int(sub.values.argmin())
    else:
        ext_offset = int(sub.values.argmax())
    ext_bar = lo + ext_offset
    lag = bar_loc - ext_bar
    saturated = (ext_bar == lo) or (ext_bar == hi - 1)
    return lag, saturated


def diagnose_ticker(ticker: str, csv_path: str) -> pd.DataFrame:
    tv = load_tv_csv(csv_path).set_index("ts")
    rows = []
    for kind, side in ALL_KINDS.items():
        if kind not in tv.columns:
            continue
        prices = tv["low"] if side == "low" else tv["high"]
        fire_bars = np.where(tv[kind].notna().values)[0]
        for bar_loc in fire_bars:
            lag5, sat5 = lag_for_signal(bar_loc, prices, WINDOW, side)
            lag10, sat10 = lag_for_signal(bar_loc, prices, WIDE, side)
            close_at_signal = tv["close"].iloc[bar_loc]
            extremum_price = prices.iloc[bar_loc - lag5] if 0 <= bar_loc - lag5 < len(prices) else np.nan
            atr = (tv["high"] - tv["low"]).rolling(14, min_periods=14).mean().iloc[bar_loc]
            missed_atr = abs(close_at_signal - extremum_price) / atr if atr and not np.isnan(atr) else np.nan
            rows.append({
                "ticker": ticker,
                "kind": kind,
                "side": side,
                "bar": bar_loc,
                "lag_w5": lag5,
                "sat_w5": sat5,
                "lag_w10": lag10,
                "sat_w10": sat10,
                "missed_move_atr": missed_atr,
            })
    return pd.DataFrame(rows)


def summarize(all_df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for kind, g in all_df.groupby("kind", sort=False):
        n = len(g)
        if n == 0:
            continue
        lag_w5 = g["lag_w5"]
        lag_w10 = g["lag_w10"]
        out.append({
            "kind": kind,
            "n": n,
            "lag_w5_mode": int(lag_w5.mode().iloc[0]) if not lag_w5.empty else None,
            "lag_w5_median": float(lag_w5.median()),
            "lag_w5_p10": float(lag_w5.quantile(0.10)),
            "lag_w5_p90": float(lag_w5.quantile(0.90)),
            "lag_w5_eq0_pct": float((lag_w5 == 0).mean() * 100),
            "lag_w5_neg_pct": float((lag_w5 < 0).mean() * 100),
            "lag_w5_sat_pct": float(g["sat_w5"].mean() * 100),
            "lag_w10_median": float(lag_w10.median()),
            "lag_w10_eq0_pct": float((lag_w10 == 0).mean() * 100),
            "lag_w10_neg_pct": float((lag_w10 < 0).mean() * 100),
            "lag_w10_sat_pct": float(g["sat_w10"].mean() * 100),
            "missed_move_atr_med": float(g["missed_move_atr"].median()),
        })
    return pd.DataFrame(out)


def per_ticker_breakdown(all_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (ticker, kind), g in all_df.groupby(["ticker", "kind"], sort=False):
        n = len(g)
        if n == 0:
            continue
        rows.append({
            "ticker": ticker,
            "kind": kind,
            "n": n,
            "lag_w5_med": float(g["lag_w5"].median()),
            "lag_w5_eq0_pct": float((g["lag_w5"] == 0).mean() * 100),
            "lag_w5_neg_pct": float((g["lag_w5"] < 0).mean() * 100),
        })
    return pd.DataFrame(rows)


def lag_histogram(all_df: pd.DataFrame, kind: str) -> str:
    g = all_df[all_df["kind"] == kind]["lag_w5"]
    if g.empty:
        return "  (no fires)"
    counts = g.value_counts().sort_index()
    lines = []
    max_count = counts.max()
    for lag_val, c in counts.items():
        bar = "#" * int(20 * c / max_count) if max_count else ""
        lines.append(f"  lag={int(lag_val):+d}  n={int(c):3d}  {bar}")
    return "\n".join(lines)


def main() -> None:
    frames = []
    for ticker, path in CSV_PATHS.items():
        frames.append(diagnose_ticker(ticker, path))
    all_df = pd.concat(frames, ignore_index=True)

    pd.set_option("display.float_format", lambda x: f"{x:,.2f}")

    print("=== Phase 0 repaint diagnostic — lag = signal_bar − local_extremum_bar ===")
    print(f"  AL-side: extremum = lowest LOW in window;  SAT-side: extremum = highest HIGH")
    print(f"  Windows: w5 = ±5 bars  |  w10 = ±10 bars\n")

    summary = summarize(all_df)
    print("--- Summary across 4 tickers ---")
    print(summary.to_string(index=False))
    print()

    print("\n--- Per-ticker breakdown (w5) ---")
    pt = per_ticker_breakdown(all_df)
    print(pt.to_string(index=False))
    print()

    print("\n--- Lag histograms (w5) ---")
    for kind in ALL_KINDS:
        print(f"\n{kind}:")
        print(lag_histogram(all_df, kind))


if __name__ == "__main__":
    main()
