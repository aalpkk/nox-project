"""Aggregate 15m Matriks bars into 17:00-truncated daily bars.

Cutoff: include bars whose CLOSE timestamp (bar_ts) is ≤ 16:45 TR.
The 16:45-17:00 bar is excluded — it may not be finalized when the live
cron fires at 17:00 TR. Expected bar count per (ticker, signal_date) is
27 (10:15 .. 16:45 closes).
"""

from __future__ import annotations

import pandas as pd

from sbt1700.config import (
    BAR_CUTOFF_HH,
    BAR_CUTOFF_MM,
    EXPECTED_BARS_PER_PAIR,
)


def aggregate_truncated_bars(intraday_df: pd.DataFrame) -> pd.DataFrame:
    """Collapse 15m bars to one truncated daily bar per (ticker, signal_date).

    Input columns: ticker, signal_date, bar_ts (ms epoch UTC), open, high,
    low, close, volume[, quantity].

    Output columns: ticker, signal_date (datetime), Open, High, Low, Close,
    Volume, n_bars, last_bar_ts_tr, intraday_coverage.
    """
    if intraday_df.empty:
        return pd.DataFrame(columns=[
            "ticker", "signal_date", "Open", "High", "Low", "Close", "Volume",
            "n_bars", "last_bar_ts_tr", "intraday_coverage",
        ])

    df = intraday_df.copy()
    df["ts_utc"] = pd.to_datetime(df["bar_ts"], unit="ms", utc=True)
    df["ts_tr"] = df["ts_utc"].dt.tz_convert("Europe/Istanbul")
    df["hh"] = df["ts_tr"].dt.hour
    df["mm"] = df["ts_tr"].dt.minute

    # Strict 16:45 cutoff: all bars before the cutoff hour, plus the bar
    # whose close lands exactly at 16:45 (mm == 45 when hh == 16).
    mask = (df["hh"] < BAR_CUTOFF_HH) | (
        (df["hh"] == BAR_CUTOFF_HH) & (df["mm"] <= BAR_CUTOFF_MM)
    )
    trunc = df[mask].copy()
    if trunc.empty:
        return pd.DataFrame(columns=[
            "ticker", "signal_date", "Open", "High", "Low", "Close", "Volume",
            "n_bars", "last_bar_ts_tr", "intraday_coverage",
        ])

    g = (
        trunc.sort_values("ts_tr")
        .groupby(["ticker", "signal_date"], as_index=False)
    )
    agg = g.agg(
        Open=("open", "first"),
        High=("high", "max"),
        Low=("low", "min"),
        Close=("close", "last"),
        Volume=("volume", "sum"),
        n_bars=("open", "size"),
        last_bar_ts_tr=("ts_tr", "last"),
    )

    agg["signal_date"] = pd.to_datetime(agg["signal_date"])
    agg["intraday_coverage"] = agg["n_bars"] / EXPECTED_BARS_PER_PAIR
    return agg
