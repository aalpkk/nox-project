"""Patterns A (failed breakdown) and B (spring / reclaim).

C and D are disabled in v1.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from prsr import config as C


def add_pattern_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per-ticker rolling features used by both patterns.

    min20         : rolling 20-bar min of low, EXCLUDING bar t (shift(1) on rolling)
    loc           : (close - low) / (high - low), 0.5 if high == low
    vol_med60     : rolling 60-bar median of volume, EXCLUDING bar t
    """
    out = df.copy()
    g = out.groupby("ticker", sort=False, group_keys=False)
    out["min20"] = g["low"].transform(lambda s: s.shift(1).rolling(C.PATTERN_N).min())
    rng = out["high"] - out["low"]
    out["loc"] = np.where(rng > 0, (out["close"] - out["low"]) / rng.replace(0, np.nan), 0.5)
    out["vol_med60"] = g["volume"].transform(
        lambda s: s.shift(1).rolling(C.PATTERN_VOL_LOOKBACK).median()
    )
    return out


def detect_pattern_a(df: pd.DataFrame) -> pd.Series:
    """Failed breakdown: intraday penetration of support, close reclaims, strong location, vol >= median."""
    return (
        df["min20"].notna()
        & df["vol_med60"].notna()
        & (df["low"] < df["min20"])
        & (df["close"] >= df["min20"])
        & (df["loc"] >= C.PATTERN_LOC_FLOOR)
        & (df["volume"] >= df["vol_med60"])
    )


def detect_pattern_b(df: pd.DataFrame) -> pd.Series:
    """Spring/reclaim: prior 5 bars had at least one close below ITS support, today reclaims, up vs prior close.

    Per-bar 'below_support[t] = close[t] < min20[t]' is a state; we then require ANY True
    in the lookback window [t-K, t-1].
    """
    out = df.copy()
    out["_below_support"] = (out["close"] < out["min20"]).fillna(False)
    g = out.groupby("ticker", sort=False, group_keys=False)
    # rolling().max() on bool is "any True in window" — apply on shifted to exclude bar t
    out["_below_support_recent"] = g["_below_support"].transform(
        lambda s: s.shift(1).rolling(C.PATTERN_B_LOOKBACK).max() > 0
    )
    out["_close_up"] = out.groupby("ticker", sort=False)["close"].transform(
        lambda s: s >= s.shift(1)
    )
    cond = (
        out["min20"].notna()
        & out["_below_support_recent"].fillna(False)
        & (out["close"] >= out["min20"])
        & (out["loc"] >= C.PATTERN_LOC_FLOOR)
        & out["_close_up"].fillna(False)
    )
    return cond


def attach_pattern_low(df: pd.DataFrame, a_mask: pd.Series, b_mask: pd.Series) -> pd.DataFrame:
    """Compute pattern_low per row.

    For A: pattern_low = low[t]
    For B: pattern_low = min(low[t-5..t])  (inclusive window)
    For both: min of the two
    """
    out = df.copy()
    g = out.groupby("ticker", sort=False, group_keys=False)
    low_window = g["low"].transform(lambda s: s.rolling(C.PATTERN_B_LOOKBACK + 1).min())
    out["pattern_low_a"] = np.where(a_mask, out["low"], np.nan)
    out["pattern_low_b"] = np.where(b_mask, low_window, np.nan)
    pl = pd.concat([out["pattern_low_a"], out["pattern_low_b"]], axis=1).min(axis=1)
    out["pattern_low"] = pl
    kind = pd.Series("none", index=out.index, dtype=object)
    kind.loc[a_mask & ~b_mask] = "a"
    kind.loc[b_mask & ~a_mask] = "b"
    kind.loc[a_mask & b_mask] = "both"
    out["pattern_kind"] = kind
    return out
