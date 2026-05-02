"""EMA context features — Phase 0 LOCKED.

Periods fixed 8/21/50; no parameter search.
ATR + non-ATR paralel feature kuralı (G7 trap audit baseline).
"""
from __future__ import annotations

import pandas as pd

EMA_PERIODS: tuple[int, int, int] = (8, 21, 50)
ATR_PERIOD: int = 14
SLOPE_LOOKBACK_21: int = 5
SLOPE_LOOKBACK_50: int = 10


def _atr_rolling_mean(df: pd.DataFrame, n: int = ATR_PERIOD) -> pd.Series:
    """Rolling-mean True Range (matches scanner.triggers.horizontal_base._atr)."""
    h, l, c = df["high"], df["low"], df["close"]
    pc = c.shift(1)
    tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def _stack_state(ema8: pd.Series, ema21: pd.Series, ema50: pd.Series) -> pd.Series:
    bull = (ema8 > ema21) & (ema21 > ema50)
    bear = (ema8 < ema21) & (ema21 < ema50)
    out = pd.Series("mixed", index=ema8.index, dtype=object)
    out[bull] = "bull"
    out[bear] = "bear"
    return out


def _compute_one(df: pd.DataFrame) -> pd.DataFrame:
    """Per-ticker feature compute. Caller must pre-sort by date."""
    p8, p21, p50 = EMA_PERIODS
    close = df["close"]

    ema8 = close.ewm(span=p8, adjust=False).mean()
    ema21 = close.ewm(span=p21, adjust=False).mean()
    ema50 = close.ewm(span=p50, adjust=False).mean()
    atr14 = _atr_rolling_mean(df, ATR_PERIOD)

    stack_max = pd.concat([ema8, ema21, ema50], axis=1).max(axis=1)
    stack_min = pd.concat([ema8, ema21, ema50], axis=1).min(axis=1)
    stack_range = stack_max - stack_min

    ema21_prev = ema21.shift(SLOPE_LOOKBACK_21)
    ema50_prev = ema50.shift(SLOPE_LOOKBACK_50)
    close_prev = close.shift(1)
    ema21_prev_bar = ema21.shift(1)

    out = pd.DataFrame({
        "ticker": df["ticker"].values,
        "date": df["date"].values,
        "ema_stack_width_atr": stack_range / atr14,
        "ema_stack_width_pct": stack_range / close,
        "ema_distance_21_atr": (close - ema21) / atr14,
        "ema_distance_21_pct": (close - ema21) / close,
        "ema21_slope_5": ema21 / ema21_prev - 1.0,
        "ema50_slope_10": ema50 / ema50_prev - 1.0,
        "ema21_reclaim": (close > ema21) & (close_prev <= ema21_prev_bar),
        "ema_stack_state": _stack_state(ema8, ema21, ema50).values,
    })
    return out


def compute_ema_features(daily: pd.DataFrame) -> pd.DataFrame:
    """Bulk per-ticker EMA context features.

    Input: daily DataFrame with columns [ticker, date, open, high, low, close, volume].
    Output: [ticker, date, 8 features].
    """
    required = {"ticker", "date", "high", "low", "close"}
    missing = required - set(daily.columns)
    if missing:
        raise ValueError(f"compute_ema_features missing cols: {sorted(missing)}")

    parts = []
    for ticker, g in daily.groupby("ticker", sort=False):
        g_sorted = g.sort_values("date").reset_index(drop=True)
        parts.append(_compute_one(g_sorted))
    return pd.concat(parts, ignore_index=True)
