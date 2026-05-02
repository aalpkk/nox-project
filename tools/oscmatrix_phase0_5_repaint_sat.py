"""Phase 0.5 — refined repaint diagnostic for HW dots, AL & SAT side-by-side.

Phase 0 conflated two failure modes:
  (a) repaint / pivot-right confirmation: signal painted onto past bar after
      future bars reveal the pivot. lag negative regardless of window;
      edge-saturation low.
  (b) trend-saturation: signal is honestly late, but local price extremum is
      simply outside ±W. edge-saturation high at w5; drops as W widens.

For each HW-dot kind (AL: HWO Up, Oversold HWO Up; SAT: HWO Down, Overbought
HWO Down), at windows ±5/±10/±20:

  p_lag_med            : signal_bar − local price_extremum_bar
  edge_sat_pct         : % of signals where price extremum is at window edge
  hw_lag_med           : signal_bar − local HW_extremum_bar
  hw_edge_sat_pct      : same on HW
  missed_atr_med       : |close@signal − price_extremum| / ATR14
  sig_after_hw_turn_%  : % of signals with hw_lag > 0 (HW already turned)

NOTE: Phase 0's negative-lag finding is "consistent with repaint OR
pivot-right confirmation". Without TV bar-replay we cannot prove repaint.
Decision is operational: drop reversal arrows as a live-clone target.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from oscmatrix import DEFAULT_PARAMS, compute_all  # noqa: E402
from oscmatrix.validate import load_tv_csv, to_ohlcv  # noqa: E402

CSV_PATHS = {
    "THYAO": "/Users/alpkarakasli/Library/Mobile Documents/com~apple~CloudDocs/Downloads1/BIST_THYAO, 1D.csv",
    "GARAN": "/Users/alpkarakasli/Library/Mobile Documents/com~apple~CloudDocs/Downloads1/New Folder With Items 3/BIST_GARAN, 1D.csv",
    "EREGL": "/Users/alpkarakasli/Library/Mobile Documents/com~apple~CloudDocs/Downloads1/New Folder With Items 3/BIST_EREGL, 1D.csv",
    "ASELS": "/Users/alpkarakasli/Library/Mobile Documents/com~apple~CloudDocs/Downloads1/New Folder With Items 3/BIST_ASELS, 1D.csv",
}

WINDOWS = [5, 10, 20]

AL_DOTS = ["HWO Up", "Oversold HWO Up"]
SAT_DOTS = ["HWO Down", "Overbought HWO Down"]


def lag_in_window(bar_loc: int, series: pd.Series, window: int, side: str) -> tuple[int, bool]:
    """Return (lag, saturated). lag = bar_loc − extremum_bar; saturated = extremum at window edge."""
    n = len(series)
    lo = max(0, bar_loc - window)
    hi = min(n, bar_loc + window + 1)
    sub = series.iloc[lo:hi]
    if side == "low":
        ext_offset = int(sub.values.argmin())
    else:
        ext_offset = int(sub.values.argmax())
    ext_bar = lo + ext_offset
    saturated = (ext_bar == lo) or (ext_bar == hi - 1)
    return bar_loc - ext_bar, saturated


def diagnose(ticker: str, csv_path: str) -> pd.DataFrame:
    tv = load_tv_csv(csv_path).set_index("ts")
    ohlcv = to_ohlcv(tv.reset_index())
    ours = compute_all(ohlcv, DEFAULT_PARAMS, zone_rule="above_50")
    hw = ours["hyperwave"].reindex(tv.index)
    atr = (tv["high"] - tv["low"]).rolling(14, min_periods=14).mean()

    rows = []
    for kind in AL_DOTS + SAT_DOTS:
        if kind not in tv.columns:
            continue
        side = "low" if kind in AL_DOTS else "high"
        price_series = tv["low"] if side == "low" else tv["high"]
        fire_bars = np.where(tv[kind].notna().values)[0]
        for bar_loc in fire_bars:
            for W in WINDOWS:
                p_lag, p_sat = lag_in_window(bar_loc, price_series, W, side)
                hw_lag, hw_sat = lag_in_window(bar_loc, hw, W, side)
                ext_offset = bar_loc - p_lag
                if 0 <= ext_offset < len(price_series):
                    ext_price = price_series.iloc[ext_offset]
                    close = tv["close"].iloc[bar_loc]
                    a = atr.iloc[bar_loc]
                    missed = abs(close - ext_price) / a if a and not np.isnan(a) else np.nan
                else:
                    missed = np.nan
                rows.append({
                    "ticker": ticker,
                    "kind": kind,
                    "side": side,
                    "window": W,
                    "bar": bar_loc,
                    "p_lag": p_lag,
                    "p_sat": p_sat,
                    "hw_lag": hw_lag,
                    "hw_sat": hw_sat,
                    "missed_atr": missed,
                    "sig_after_hw_turn": hw_lag > 0,
                })
    return pd.DataFrame(rows)


def summarize(all_df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (kind, W), g in all_df.groupby(["kind", "window"], sort=False):
        out.append({
            "kind": kind,
            "n": len(g),
            "win": W,
            "p_lag_med": float(g["p_lag"].median()),
            "p_lag_neg_%": float((g["p_lag"] < 0).mean() * 100),
            "edge_sat_%": float(g["p_sat"].mean() * 100),
            "hw_lag_med": float(g["hw_lag"].median()),
            "hw_lag_neg_%": float((g["hw_lag"] < 0).mean() * 100),
            "hw_edge_sat_%": float(g["hw_sat"].mean() * 100),
            "missed_atr_med": float(g["missed_atr"].median()),
            "sig_after_hw_turn_%": float(g["sig_after_hw_turn"].mean() * 100),
        })
    return pd.DataFrame(out)


def main() -> None:
    frames = []
    for ticker, path in CSV_PATHS.items():
        frames.append(diagnose(ticker, path))
    all_df = pd.concat(frames, ignore_index=True)
    pd.set_option("display.float_format", lambda x: f"{x:,.2f}")

    print("=== Phase 0.5 — refined repaint diagnostic (HW dots, AL & SAT) ===")
    print("Lag = signal_bar − local extremum bar; positive = signal AFTER extremum (live-feasible).")
    print("Three windows: ±5 / ±10 / ±20.\n")

    s = summarize(all_df)
    order_kind = AL_DOTS + SAT_DOTS
    s["kind_ord"] = s["kind"].map({k: i for i, k in enumerate(order_kind)})
    s = s.sort_values(["kind_ord", "win"]).drop(columns=["kind_ord"])

    cols = ["kind", "n", "win", "p_lag_med", "p_lag_neg_%", "edge_sat_%",
            "hw_lag_med", "hw_lag_neg_%", "hw_edge_sat_%",
            "missed_atr_med", "sig_after_hw_turn_%"]
    print(s[cols].to_string(index=False))
    print()

    print("\n--- Acceptance gate per HW-dot kind ---")
    print("  (G1) hw_lag_med ≥ +1 in ALL three windows         — oscillator-pivot lag is honest")
    print("  (G2) sig_after_hw_turn_% ≥ 70% (max across W)     — HW already turned at signal time")
    print("  (G3) edge_sat_% drops widening w5 → w20           — high w5 saturation = trend, not repaint")
    print()
    for kind in order_kind:
        rows = s[s["kind"] == kind].sort_values("win")
        if rows.empty:
            print(f"  {kind:<22}: no fires")
            continue
        hw_lag_min = rows["hw_lag_med"].min()
        after_max = rows["sig_after_hw_turn_%"].max()
        sat_w5 = float(rows[rows["win"] == 5]["edge_sat_%"].iloc[0])
        sat_w20 = float(rows[rows["win"] == 20]["edge_sat_%"].iloc[0])
        g1 = "PASS" if hw_lag_min >= 1 else "FAIL"
        g2 = "PASS" if after_max >= 70 else "FAIL"
        g3 = "PASS" if sat_w20 <= sat_w5 * 0.5 + 1e-9 else "FAIL"
        if g1 == "PASS" and g2 == "PASS":
            verdict = "CLEAN (oscillator-pivot signal)"
        elif g1 == "PASS":
            verdict = "OSC-CLEAN, PRICE-FUZZY (exit signal, not price-top detector)"
        else:
            verdict = "AMBIGUOUS"
        print(f"  {kind:<22}  hw_lag_med_min={hw_lag_min:+.1f} ({g1})  "
              f"after_hw_turn_max={after_max:.0f}% ({g2})  "
              f"edge_sat w5→w20: {sat_w5:.0f}%→{sat_w20:.0f}% ({g3})")
        print(f"  {'':<22}  → {verdict}")


if __name__ == "__main__":
    main()
