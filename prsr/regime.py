"""ER60 + ATR%20 cross-sectional regime gate.

Both quantiles are recomputed every date over the Tradeable Core cohort
on that date.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from prsr import config as C


def _wilder_atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / n, adjust=False).mean()


def _efficiency_ratio(close: pd.Series, n: int) -> pd.Series:
    direction = (close - close.shift(n)).abs()
    volatility = close.diff().abs().rolling(n).sum()
    return direction / volatility.replace(0, np.nan)


def add_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per-ticker ER60, atr20, ATR%20."""
    out = df.copy()
    g = out.groupby("ticker", sort=False, group_keys=False)
    out["atr20"] = g.apply(lambda x: _wilder_atr(x["high"], x["low"], x["close"], C.ATR_LOOKBACK))
    out["atr_pct20"] = out["atr20"] / out["close"]
    out["er60"] = g.apply(lambda x: _efficiency_ratio(x["close"], C.ER_LOOKBACK))
    return out


def attach_cross_sectional_thresholds(df: pd.DataFrame) -> pd.DataFrame:
    """Per-date q40 of ER60 (floor) and q90 of ATR%20 (ceiling) over Core cohort."""
    out = df.copy()
    core = out["tier"].eq("core")

    er_floor = (
        out.loc[core & out["er60"].notna()]
        .groupby("date")["er60"]
        .quantile(C.ER_QUANTILE)
        .rename("er60_q40")
    )
    out = out.merge(er_floor, left_on="date", right_index=True, how="left")

    atr_ceil = (
        out.loc[core & out["atr_pct20"].notna()]
        .groupby("date")["atr_pct20"]
        .quantile(C.ATR_PCT_QUANTILE)
        .rename("atr_pct20_q90")
    )
    out = out.merge(atr_ceil, left_on="date", right_index=True, how="left")

    return out


def regime_pass(df: pd.DataFrame) -> pd.Series:
    """ER60 >= q40 AND ATR%20 <= q90, evaluated for each (ticker, date)."""
    return (
        df["er60"].notna()
        & df["atr_pct20"].notna()
        & df["er60_q40"].notna()
        & df["atr_pct20_q90"].notna()
        & (df["er60"] >= df["er60_q40"])
        & (df["atr_pct20"] <= df["atr_pct20_q90"])
    )
