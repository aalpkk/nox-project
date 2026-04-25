"""yfinance 1-hour fallback fetcher.

Last-resort source when both Fintables and Matriks fail. Granularity drops
from 15m → 1h, so the 17:00 TR cutoff keeps 7 hourly bars (10:00..17:00)
instead of 28. The aggregated daily OHLCV is still well-defined; only
sub-hourly features lose resolution. Volume (yfinance share count) is
multiplied by close to produce a TL-equivalent figure consistent with
Matriks/Fintables conventions.

bar_ts convention: yfinance hourly timestamps are bar-OPEN; we shift by
+1h to match the project's CLOSE-time convention.
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Iterable

import pandas as pd

from nyxexpansion.intraday.fetchers.base import (
    EmptyResult,
    FetchResult,
    FetcherError,
)


def _suffix(ticker: str) -> str:
    return ticker if ticker.endswith(".IS") else f"{ticker}.IS"


def _strip(ticker: str) -> str:
    return ticker[:-3] if ticker.endswith(".IS") else ticker


def _to_close_ms(ts: pd.Timestamp) -> int:
    """yfinance ts (bar OPEN) → close time UTC ms."""
    if ts.tz is None:
        ts = ts.tz_localize("Europe/Istanbul")
    close_ts = ts + pd.Timedelta(hours=1)
    return int(close_ts.tz_convert("UTC").value // 1_000_000)


def _fetch_one(ticker: str, target_date) -> list[dict]:
    import yfinance as yf

    target = pd.Timestamp(target_date).normalize()
    target_date_obj = target.date()
    sym = _suffix(ticker)
    start = target.strftime("%Y-%m-%d")
    end = (target + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    try:
        df = yf.download(
            sym, start=start, end=end, interval="1h",
            progress=False, auto_adjust=False, threads=False,
        )
    except Exception as exc:
        raise FetcherError(f"yfinance download error: {exc}")
    if df is None or df.empty:
        return []
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    out = []
    for ts, row in df.iterrows():
        try:
            bar_ts_ms = _to_close_ms(pd.Timestamp(ts))
            close_val = float(row["Close"])
            shares = float(row["Volume"]) if pd.notna(row["Volume"]) else 0.0
            out.append({
                "ticker": _strip(ticker),
                "signal_date": target_date_obj,
                "bar_ts": bar_ts_ms,
                "date": target.strftime("%Y-%m-%d"),
                "open": float(row["Open"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "close": close_val,
                "volume": shares * close_val,
                "quantity": shares,
                "bars_source": "yfinance_1h",
            })
        except (KeyError, ValueError, TypeError):
            continue
    return out


def fetch_per_ticker(
    tickers: Iterable[str],
    target_date: date | datetime | pd.Timestamp,
) -> dict[str, FetchResult]:
    """Per-ticker yfinance 1h fetch — last resort."""
    target = pd.Timestamp(target_date).normalize()
    target_date_obj = target.date()

    out: dict[str, FetchResult] = {}
    for tk in tickers:
        try:
            rows = _fetch_one(tk, target)
        except FetcherError as exc:
            out[tk] = FetchResult(
                ticker=tk, signal_date=target_date_obj,
                bars_source=None, rows=[], note="fetch_error",
                detail=str(exc)[:200],
            )
            continue
        if not rows:
            out[tk] = FetchResult(
                ticker=tk, signal_date=target_date_obj,
                bars_source=None, rows=[], note="empty",
                detail="yfinance returned no 1h bars",
            )
            continue
        out[tk] = FetchResult(
            ticker=tk, signal_date=target_date_obj,
            bars_source="yfinance_1h", rows=rows, note="ok",
        )
    return out
