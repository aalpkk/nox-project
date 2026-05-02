"""Daily panel build + 3-tier universe classification (Core / Watchable / IPO).

Local to prsr/. The IPO carve-out and q60 liquidity gate do NOT propagate
to other modules.
"""
from __future__ import annotations

import pandas as pd

from prsr import config as C


def load_daily_panel(parquet_path: str = C.MASTER_PARQUET) -> pd.DataFrame:
    """Aggregate the 1h master to a daily OHLCV panel keyed by (ticker, date)."""
    raw = pd.read_parquet(parquet_path)
    raw = raw.copy()
    raw["date"] = raw["ts_istanbul"].dt.tz_localize(None).dt.normalize()
    grp = raw.groupby(["ticker", "date"], sort=True)
    daily = grp.agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    ).reset_index()
    daily = daily.sort_values(["ticker", "date"]).reset_index(drop=True)
    daily = daily[
        (daily["date"] >= pd.Timestamp(C.START_DATE))
        & (daily["date"] <= pd.Timestamp(C.END_DATE))
    ].reset_index(drop=True)
    return daily


def add_universe_features(daily: pd.DataFrame) -> pd.DataFrame:
    """Per-ticker rolling features for tier gating.

    history_bars       — number of bars seen up to and including t
    median_turnover_60 — trailing 60-bar median of close*volume,
                         excluding bar t (shift(1) on rolling)
    """
    df = daily.copy()
    df["turnover"] = df["close"] * df["volume"]
    g = df.groupby("ticker", sort=False)
    df["history_bars"] = g.cumcount() + 1
    df["median_turnover_60"] = (
        g["turnover"]
        .transform(lambda s: s.shift(1).rolling(C.LIQUIDITY_LOOKBACK).median())
    )
    return df


def attach_q60_liquidity(df: pd.DataFrame) -> pd.DataFrame:
    """Per-date cross-sectional q60 of median_turnover_60 over Core-eligible cohort.

    Core-eligibility for q60 reference set: history_bars >= 500 AND median_turnover_60 not null.
    """
    out = df.copy()
    core_mask = (out["history_bars"] >= C.HISTORY_BARS_CORE) & out["median_turnover_60"].notna()
    q = (
        out.loc[core_mask]
        .groupby("date")["median_turnover_60"]
        .quantile(C.LIQUIDITY_QUANTILE)
        .rename("q60_turnover")
    )
    out = out.merge(q, left_on="date", right_index=True, how="left")
    return out


def classify_tier(df: pd.DataFrame) -> pd.DataFrame:
    """Assign tier ∈ {core, watchable, ipo, ineligible}.

    core      : history_bars >= 500 AND median_turnover_60 >= q60_turnover[date]
    watchable : 250 <= history_bars < 500 (no liquidity gate; watchlist only)
    ipo       : history_bars < 250 (excluded entirely)
    ineligible: above 500 bars but below liquidity floor
    """
    out = df.copy()
    tier = pd.Series("ineligible", index=out.index, dtype=object)
    ipo = out["history_bars"] < C.HISTORY_BARS_WATCHABLE_MIN
    watchable = (~ipo) & (out["history_bars"] < C.HISTORY_BARS_CORE)
    core_eligible = (
        (out["history_bars"] >= C.HISTORY_BARS_CORE)
        & out["median_turnover_60"].notna()
        & out["q60_turnover"].notna()
        & (out["median_turnover_60"] >= out["q60_turnover"])
    )
    tier.loc[ipo] = "ipo"
    tier.loc[watchable] = "watchable"
    tier.loc[core_eligible] = "core"
    out["tier"] = tier
    return out


def build_universe(parquet_path: str = C.MASTER_PARQUET) -> pd.DataFrame:
    """End-to-end: daily panel + tier classification."""
    daily = load_daily_panel(parquet_path)
    daily = add_universe_features(daily)
    daily = attach_q60_liquidity(daily)
    daily = classify_tier(daily)
    return daily
