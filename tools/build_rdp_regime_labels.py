"""
RDP-based offline regime labeler for XU100 daily.

Pipeline:
  OHLC -> ATR(14) -> close/ATR vol-space -> RDP(mult) -> anchor pivots
  -> per-segment classify (bull / bear / sideways by net %) -> per-day labels.

Output: output/regime_labels_daily_rdp_v1.csv (does NOT overwrite the SMA-cross
file). Schema matches the existing regime_labels_daily.csv consumers.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OHLC_PATH = ROOT / "output" / "xu100_cache.parquet"
OUT_PATH  = ROOT / "output" / "regime_labels_daily_rdp_v1.csv"

# ---- params (locked per user choice) ----
RDP_MULT          = 1.0    # tolerance multiplier in vol-space units
ATR_LEN           = 14
SIDEWAYS_PCT      = 0.12   # |net move| < 12% -> sideways
LABEL_SOURCE      = "rdp_v1"
WINDOW_ID         = f"rdp_v1_mult{RDP_MULT:g}_atr{ATR_LEN}_sw{int(SIDEWAYS_PCT*100)}"


def true_range(h: pd.Series, l: pd.Series, c: pd.Series) -> pd.Series:
    pc = c.shift(1)
    tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr


def atr(h: pd.Series, l: pd.Series, c: pd.Series, n: int = 14) -> pd.Series:
    return true_range(h, l, c).rolling(n, min_periods=n).mean()


def perpendicular_distance(p1: tuple[float, float],
                           p2: tuple[float, float],
                           p0: tuple[float, float]) -> float:
    """Perpendicular distance from p0 to the line p1-p2."""
    x1, y1 = p1
    x2, y2 = p2
    x0, y0 = p0
    dx, dy = x2 - x1, y2 - y1
    denom = math.sqrt(dx * dx + dy * dy)
    if denom == 0.0:
        return 0.0
    num = abs(dy * x0 - dx * y0 + x2 * y1 - y2 * x1)
    return num / denom


def rdp_iter(points: list[tuple[float, float]], eps: float) -> list[int]:
    """
    Iterative Ramer-Douglas-Peucker. Returns sorted indices of kept anchors.
    Same algorithm as the LuxAlgo Pine version.
    """
    n = len(points)
    if n < 3:
        return list(range(n))
    keep = [False] * n
    keep[0] = True
    keep[-1] = True
    stack: list[tuple[int, int]] = [(0, n - 1)]
    while stack:
        first, last = stack.pop()
        max_d = 0.0
        idx = first
        p1, p2 = points[first], points[last]
        for i in range(first + 1, last):
            d = perpendicular_distance(p1, p2, points[i])
            if d > max_d:
                max_d = d
                idx = i
        if max_d > eps:
            keep[idx] = True
            stack.append((idx, last))
            stack.append((first, idx))
    return [i for i, k in enumerate(keep) if k]


def classify_segment(close_a: float, close_b: float, sideways_pct: float) -> str:
    """3-state regime label. Schema follows existing CSV: long / short / neutral."""
    net = (close_b - close_a) / close_a
    if abs(net) < sideways_pct:
        return "neutral"          # sideways
    return "long" if net > 0 else "short"


def main() -> None:
    df = pd.read_parquet(OHLC_PATH).copy()
    df.columns = [c.lower() for c in df.columns]
    df = df[["open", "high", "low", "close"]].dropna().sort_index()
    df["atr"] = atr(df["high"], df["low"], df["close"], ATR_LEN)
    df = df.dropna(subset=["atr"])
    df = df[df["atr"] > 0].copy()

    # vol-space points: x = bar index, y = close / ATR
    bars = np.arange(len(df), dtype=float)
    y    = (df["close"].values / df["atr"].values).astype(float)
    points = list(zip(bars.tolist(), y.tolist()))

    anchor_idx = rdp_iter(points, eps=RDP_MULT)
    anchors = df.iloc[anchor_idx]
    print(f"[rdp] {len(df)} bars -> {len(anchor_idx)} anchors "
          f"(mult={RDP_MULT}, eps in vol-space units)")

    # Per-segment classify, forward-fill labels per day
    regimes = pd.Series(index=df.index, dtype=object)
    for i in range(len(anchor_idx) - 1):
        s = anchor_idx[i]
        e = anchor_idx[i + 1]
        c0 = df["close"].iat[s]
        c1 = df["close"].iat[e]
        label = classify_segment(c0, c1, SIDEWAYS_PCT)
        # inclusive on the left, exclusive on the right; final segment closes on the last anchor
        regimes.iloc[s:e] = label
    # tail bar
    regimes.iloc[anchor_idx[-1]] = regimes.iloc[anchor_idx[-1] - 1]

    out = pd.DataFrame({
        "date":         df.index.strftime("%Y-%m-%d"),
        "close":        df["close"].values,
        "regime":       regimes.values,
        "sub_regime":   "",
        "window_id":    WINDOW_ID,
        "label_source": LABEL_SOURCE,
    })
    out.to_csv(OUT_PATH, index=False)
    print(f"[write] {OUT_PATH.relative_to(ROOT)}  rows={len(out)}")

    # Diagnostics
    print()
    print("regime distribution:")
    print(out["regime"].value_counts().to_string())
    trans = (out["regime"] != out["regime"].shift(1)).sum() - 1
    print(f"\ntransitions: {trans}")

    # Anchor list (compact)
    print("\nanchor pivots (date, close):")
    pivots = df.iloc[anchor_idx][["close"]].copy()
    pivots["date"] = pivots.index.strftime("%Y-%m-%d")
    for _, row in pivots.iterrows():
        print(f"  {row['date']}  {row['close']:>10.2f}")


if __name__ == "__main__":
    main()
