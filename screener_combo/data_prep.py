"""Data preparation for screener combo backtest.

Loads nox_intraday_v1 master parquet, resamples to session-aware daily bars,
loads XU100 benchmark, and exposes TRAIN+VAL split bounds.
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd

from data.intraday_1h import (
    load_intraday,
    daily_resample,
    load_splits,
    eligible_tickers,
)


XU100_CACHE = Path("output/xu100_cache.parquet")


def load_xu100_close() -> pd.Series:
    """Load XU100 daily close as a tz-naive DatetimeIndex Series."""
    if not XU100_CACHE.exists():
        raise FileNotFoundError(f"XU100 cache missing: {XU100_CACHE}")
    xu = pd.read_parquet(XU100_CACHE)
    s = xu["Close"].copy()
    s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
    return s.sort_index()


def load_panel_daily(min_coverage: float = 0.50) -> pd.DataFrame:
    """Load full panel of daily bars (session-aware via intraday_1h adapter).

    Returns a long-form DataFrame: ticker, date (datetime64), open/high/low/close/volume.
    """
    bars = load_intraday(min_coverage=min_coverage)
    daily = daily_resample(bars)
    daily["date"] = pd.to_datetime(daily["date"]).dt.tz_localize(None).dt.normalize()
    return daily.sort_values(["ticker", "date"]).reset_index(drop=True)


def split_bounds() -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    """Return train/val/test (start, end) timestamps."""
    s = load_splits()
    out = {}
    for k in ("train", "val", "test"):
        out[k] = (
            pd.Timestamp(s[k]["start"]).normalize(),
            pd.Timestamp(s[k]["end"]).normalize(),
        )
    return out


def trainval_window() -> tuple[pd.Timestamp, pd.Timestamp]:
    b = split_bounds()
    return b["train"][0], b["val"][1]
