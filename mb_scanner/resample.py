"""1h → {5h, 1d, 1w, 1M} resamplers for the nox_intraday_v1 master.

Master convention: bar timestamped HH:00 TR covers prices in [HH:00, HH+1:00).
BIST 09:55 opening auction prints into the 09:00 hourly bar; closing auction
(18:05) and closing-price trades (18:08–18:10) print into the 18:00 hourly
bar. So the master typically holds 10 hourly bars per BIST session for hours
{9..18}, with mid-session hours sometimes thin for low-liquidity tickers.

5h grouping mirrors TV's "5 saatlik" timeframe for BIST equities — two clean
session-aligned bars per day, no partial bars:
  morning   = hours [9, 10, 11, 12, 13]   → 5h bar labeled <date> 09:00 TR
  afternoon = hours [14, 15, 16, 17, 18]  → 5h bar labeled <date> 14:00 TR

Daily = standard date groupby (already provided by data.intraday_1h, mirrored
here for self-containment).

Weekly = daily resampled with W-FRI anchor (BIST closes Friday). Weekly bar
is labeled with the Friday date (right-anchored).

Monthly = daily resampled with BME (Business Month End) anchor — labeled
with the last business day of the month. Right-anchored.
"""
from __future__ import annotations

import pandas as pd


def to_5h(bars_1h: pd.DataFrame, *, tz: str = "Europe/Istanbul") -> pd.DataFrame:
    """Resample 1h bars to 2-bars-per-day 5h panel (TV-aligned).

    Morning bin (label 09:00) covers hourly bars hh ∈ [9..13]: opening auction
    print at 09:00 + continuous session 10:00–14:00.
    Afternoon bin (label 14:00) covers hourly bars hh ∈ [14..18]: continuous
    14:00–18:00 + closing auction (18:05) + closing-price trades (18:08–18:10).

    Returns columns: ticker, ts_istanbul (label = bin start), open, high,
    low, close, volume, n_bars. Bins with 0 source bars are dropped.
    """
    if bars_1h.empty:
        return bars_1h.iloc[0:0].copy()
    df = bars_1h.copy()
    if df["ts_istanbul"].dt.tz is None:
        df["ts_istanbul"] = df["ts_istanbul"].dt.tz_localize(tz)
    df["date"] = df["ts_istanbul"].dt.date
    df["hh"] = df["ts_istanbul"].dt.hour
    df["half"] = (df["hh"] >= 14).astype(int)  # 0 = morning, 1 = afternoon
    df["label_hh"] = df["half"].map({0: 9, 1: 14})
    df["bin_ts"] = pd.to_datetime(
        df["date"].astype(str) + " " + df["label_hh"].astype(str).str.zfill(2) + ":00"
    ).dt.tz_localize(tz)

    agg = (
        df.groupby(["ticker", "bin_ts"], observed=True)
        .agg(open=("open", "first"),
             high=("high", "max"),
             low=("low", "min"),
             close=("close", "last"),
             volume=("volume", "sum"),
             n_bars=("close", "count"))
        .reset_index()
        .rename(columns={"bin_ts": "ts_istanbul"})
        .sort_values(["ticker", "ts_istanbul"])
        .reset_index(drop=True)
    )
    return agg


def to_daily(bars_1h: pd.DataFrame, *, tz: str = "Europe/Istanbul") -> pd.DataFrame:
    """Self-contained 1h→1d resample (mirrors data.intraday_1h.daily_resample).

    Returns columns: ticker, date (datetime64[ns], naive), open, high, low,
    close, volume, n_bars.
    """
    if bars_1h.empty:
        return bars_1h.iloc[0:0].copy()
    df = bars_1h.copy()
    if df["ts_istanbul"].dt.tz is None:
        df["ts_istanbul"] = df["ts_istanbul"].dt.tz_localize(tz)
    df["date"] = pd.to_datetime(df["ts_istanbul"].dt.date)
    agg = (
        df.groupby(["ticker", "date"], observed=True)
        .agg(open=("open", "first"),
             high=("high", "max"),
             low=("low", "min"),
             close=("close", "last"),
             volume=("volume", "sum"),
             n_bars=("close", "count"))
        .reset_index()
        .sort_values(["ticker", "date"])
        .reset_index(drop=True)
    )
    return agg


def to_weekly(bars_1h: pd.DataFrame, *, tz: str = "Europe/Istanbul") -> pd.DataFrame:
    """1h → weekly via daily, anchored to Friday (BIST week ends Friday).

    Weekly bar label = Friday of that week (datetime64[ns], naive).
    Returns columns: ticker, week_end, open, high, low, close, volume, n_bars.
    """
    daily = to_daily(bars_1h, tz=tz)
    if daily.empty:
        return daily.assign(week_end=pd.NaT).iloc[0:0]

    out_frames: list[pd.DataFrame] = []
    for ticker, g in daily.groupby("ticker", observed=True):
        g = g.set_index("date").sort_index()
        wk = g.resample("W-FRI", label="right", closed="right").agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
            n_bars=("close", "count"),
        )
        wk = wk.dropna(subset=["close"]).reset_index().rename(columns={"date": "week_end"})
        wk["ticker"] = ticker
        out_frames.append(wk)
    if not out_frames:
        return daily.iloc[0:0].assign(week_end=pd.NaT)
    out = pd.concat(out_frames, ignore_index=True)
    return out[["ticker", "week_end", "open", "high", "low", "close", "volume", "n_bars"]]


def to_monthly(bars_1h: pd.DataFrame, *, tz: str = "Europe/Istanbul") -> pd.DataFrame:
    """1h → monthly via daily, anchored to last business day (BME).

    Monthly bar label = last business day of that month (datetime64[ns],
    naive). Returns columns: ticker, month_end, open, high, low, close,
    volume, n_bars.
    """
    daily = to_daily(bars_1h, tz=tz)
    if daily.empty:
        return daily.assign(month_end=pd.NaT).iloc[0:0]

    out_frames: list[pd.DataFrame] = []
    for ticker, g in daily.groupby("ticker", observed=True):
        g = g.set_index("date").sort_index()
        mo = g.resample("BME", label="right", closed="right").agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
            n_bars=("close", "count"),
        )
        mo = mo.dropna(subset=["close"]).reset_index().rename(columns={"date": "month_end"})
        mo["ticker"] = ticker
        out_frames.append(mo)
    if not out_frames:
        return daily.iloc[0:0].assign(month_end=pd.NaT)
    out = pd.concat(out_frames, ignore_index=True)
    return out[["ticker", "month_end", "open", "high", "low", "close", "volume", "n_bars"]]


def per_ticker_panel(
    resampled: pd.DataFrame,
    ts_col: str,
) -> dict[str, pd.DataFrame]:
    """Split resampled long DF into per-ticker DataFrames indexed by ts_col.

    Each per-ticker DF has columns [open, high, low, close, volume] indexed
    by a tz-aware (or naive for daily/weekly) DatetimeIndex.
    """
    out: dict[str, pd.DataFrame] = {}
    for ticker, g in resampled.groupby("ticker", observed=True):
        g = g.sort_values(ts_col)
        idx = pd.DatetimeIndex(pd.to_datetime(g[ts_col]))
        df = g[["open", "high", "low", "close", "volume"]].set_index(idx)
        df.attrs["ticker"] = str(ticker)
        out[str(ticker)] = df
    return out
