"""Aggregate nox_intraday_v1 1h bars into 17:00-truncated daily bars.

Cutoff: include bars whose ts_istanbul (= bar OPEN timestamp, TV/extfeed
convention) has hour ≤ 16. The 16:00–17:00 bar is the last bar whose
window closes at 17:00 TR — the SBT decision moment. The 17:00–18:00
bar is excluded.

BIST hourly grid (post-2017 continuous trading):
    09:00, 10:00, 11:00, 12:00, 13:00, 14:00, 15:00, 16:00, 17:00, 18:00
17:00-truncation includes 09:00..16:00 = 8 bars. The 09:00 BIST opening-
auction bar carries real OHLC + volume and is included.

Output schema is identical to ``aggregator.aggregate_truncated_bars`` so
``signals.detect_candidates`` and ``features.build_features`` consume it
unchanged (with ``expected_bars=8`` instead of the 15m default of 27).
"""

from __future__ import annotations

import pandas as pd

from sbt1700.config import BAR_CUTOFF_HOUR_1H, EXPECTED_BARS_1H


_OUTPUT_COLS = [
    "ticker", "signal_date", "Open", "High", "Low", "Close", "Volume",
    "n_bars", "last_bar_ts_tr", "intraday_coverage",
]


def aggregate_truncated_bars_1h(intraday_df: pd.DataFrame) -> pd.DataFrame:
    """Collapse 1h bars to one truncated daily bar per (ticker, signal_date).

    Args:
        intraday_df: nox_intraday_v1 master rows with columns ``ticker``,
            ``ts_istanbul`` (tz-aware, Europe/Istanbul, bar OPEN), ``open``,
            ``high``, ``low``, ``close``, ``volume``.

    Returns:
        DataFrame with columns ticker, signal_date (datetime, normalized),
        Open, High, Low, Close, Volume, n_bars, last_bar_ts_tr,
        intraday_coverage (= n_bars / 8).
    """
    if intraday_df.empty:
        return pd.DataFrame(columns=_OUTPUT_COLS)

    df = intraday_df.copy()
    ts = pd.to_datetime(df["ts_istanbul"])
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize("Europe/Istanbul")
    df["_ts_tr"] = ts
    df["_hh"] = df["_ts_tr"].dt.hour

    # ts_istanbul is bar OPEN; include bars whose window ends ≤ 17:00
    # (i.e., open hour ≤ 16). Filters out 17:00-18:00 bar and any
    # off-grid late-evening bars.
    mask = df["_hh"] <= BAR_CUTOFF_HOUR_1H
    trunc = df[mask].copy()
    if trunc.empty:
        return pd.DataFrame(columns=_OUTPUT_COLS)

    trunc["signal_date"] = trunc["_ts_tr"].dt.normalize().dt.tz_localize(None)

    g = (
        trunc.sort_values(["ticker", "_ts_tr"])
        .groupby(["ticker", "signal_date"], as_index=False)
    )
    agg = g.agg(
        Open=("open", "first"),
        High=("high", "max"),
        Low=("low", "min"),
        Close=("close", "last"),
        Volume=("volume", "sum"),
        n_bars=("open", "size"),
        last_bar_ts_tr=("_ts_tr", "last"),
    )
    agg["signal_date"] = pd.to_datetime(agg["signal_date"])
    agg["intraday_coverage"] = agg["n_bars"] / EXPECTED_BARS_1H
    return agg[_OUTPUT_COLS]


def daily_resample_full(intraday_df: pd.DataFrame) -> pd.DataFrame:
    """Full-day OHLCV per (ticker, date) from the same 1h panel.

    Used to build the daily indicator panel (EMAs, ATR, vol SMA) and
    the forward window for 5d labels. Includes ALL bars on each session
    (09:00..18:00, no cutoff). Returns a long table indexed by Date with
    ``ticker`` column and standard ``Open/High/Low/Close/Volume`` casing
    so downstream signals/features modules consume it unchanged.
    """
    if intraday_df.empty:
        return pd.DataFrame(columns=["ticker", "Open", "High", "Low", "Close", "Volume"])

    df = intraday_df.copy()
    ts = pd.to_datetime(df["ts_istanbul"])
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize("Europe/Istanbul")
    df["_ts_tr"] = ts
    df["Date"] = df["_ts_tr"].dt.normalize().dt.tz_localize(None)

    g = (
        df.sort_values(["ticker", "_ts_tr"])
        .groupby(["ticker", "Date"], as_index=False)
    )
    daily = g.agg(
        Open=("open", "first"),
        High=("high", "max"),
        Low=("low", "min"),
        Close=("close", "last"),
        Volume=("volume", "sum"),
    )
    return daily.set_index("Date").sort_index()
