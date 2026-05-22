"""nox_intraday_us_v1 — US adapter, mirrors data/intraday_1h.py for NASDAQ.

Reads output/extfeed_intraday_1h_3y_master_us.parquet (produced by
tools/extfeed_pull_us.py) and exposes the same public surface as the BIST
adapter so mb_scanner / downstream consumers can plug in.

Schema in storage:
    ticker, ts_utc, ts_newyork, open, high, low, close, volume

For downstream compatibility (mb_scanner.resample uses `ts_istanbul` column
name throughout), load_intraday() returns the timestamp column named
`ts_istanbul` — the values are America/New_York-localized, not Istanbul.
This is a deliberate legacy-name carryover to avoid refactoring the
resample pipeline.

POC scope: mb_1d / mb_1w / mb_1M families work directly. mb_5h family is
NOT supported here because mb_scanner.resample.to_5h is hardcoded to the
BIST 09:00–18:00 TR session split; NYSE 09:30–16:00 ET needs a different
session map. Run mb_scanner with families=['mb_1d','mb_1w','mb_1M'] for US.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


DATASET_VERSION = "nox_intraday_us_v1"

MASTER = Path("output/extfeed_intraday_1h_3y_master_us.parquet")

# 3y design window — matches BIST adapter convention. Adjust if pull window
# differs.
MIN_DATE = "2023-01-02"


def load_intraday(
    tickers: Iterable[str] | None = None,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    min_coverage: float = 0.0,
) -> pd.DataFrame:
    """Load US intraday 1h bars with optional filters.

    Returns DataFrame:
        ticker, ts_utc, ts_istanbul (legacy name — values are America/New_York),
        open, high, low, close, volume
    sorted by (ticker, ts_istanbul).
    """
    if not MASTER.exists():
        raise FileNotFoundError(
            f"US master parquet missing: {MASTER} — run tools/extfeed_pull_us.py "
            "with --mode bootstrap first."
        )

    bars = pd.read_parquet(MASTER)
    bars["ts_newyork"] = pd.to_datetime(bars["ts_newyork"])

    # Legacy name carryover for resample.py compat — values stay NY-tz-aware.
    bars = bars.rename(columns={"ts_newyork": "ts_istanbul"})

    effective_start = start if start is not None else MIN_DATE

    if tickers is not None:
        wanted = set(tickers)
        bars = bars[bars["ticker"].isin(wanted)]

    start_ts = pd.Timestamp(effective_start)
    if start_ts.tz is None and bars["ts_istanbul"].dt.tz is not None:
        start_ts = start_ts.tz_localize(bars["ts_istanbul"].dt.tz)
    bars = bars[bars["ts_istanbul"] >= start_ts]

    if end is not None:
        end_ts = pd.Timestamp(end)
        if end_ts.tz is None and bars["ts_istanbul"].dt.tz is not None:
            end_ts = end_ts.tz_localize(bars["ts_istanbul"].dt.tz)
        if pd.Timestamp(end).normalize() == pd.Timestamp(end):
            end_ts = end_ts + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        bars = bars[bars["ts_istanbul"] <= end_ts]

    bars = bars.sort_values(["ticker", "ts_istanbul"]).reset_index(drop=True)
    return bars


def daily_resample(
    df: pd.DataFrame,
    *,
    tz: str = "America/New_York",
) -> pd.DataFrame:
    """1h → daily OHLCV per ticker (NY trading date)."""
    if df.empty:
        return df
    df = df.copy()
    if df["ts_istanbul"].dt.tz is None:
        df["ts_istanbul"] = df["ts_istanbul"].dt.tz_localize(tz)
    df["date"] = df["ts_istanbul"].dt.date
    agg = (
        df.groupby(["ticker", "date"], observed=True)
        .agg(open=("open", "first"),
             high=("high", "max"),
             low=("low", "min"),
             close=("close", "last"),
             volume=("volume", "sum"),
             n_bars=("close", "count"))
        .reset_index()
    )
    return agg


__all__ = [
    "DATASET_VERSION",
    "MIN_DATE",
    "MASTER",
    "load_intraday",
    "daily_resample",
]
