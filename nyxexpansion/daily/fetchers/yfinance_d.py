"""yfinance daily fallback (tier-3, last resort).

One batch download per call (multi-ticker, ``interval="1d"``); ``.IS``
suffix added automatically. Volume is shares; we also surface ``volume``
as TL (shares × close) for parity with the upstream tiers.
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Iterable

import pandas as pd

from nyxexpansion.daily.fetchers.base import (
    EmptyResult,
    FetchResult,
    FetcherError,
)


def _suffix(ticker: str) -> str:
    return ticker if ticker.endswith(".IS") else f"{ticker}.IS"


def _strip(ticker: str) -> str:
    return ticker[:-3] if ticker.endswith(".IS") else ticker


def _yf_batch(tickers: list[str], start_date: date, end_date: date) -> dict[str, list[dict]]:
    import yfinance as yf

    if not tickers:
        return {}
    yf_syms = [_suffix(t) for t in tickers]
    end_excl = (pd.Timestamp(end_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    raw = yf.download(
        yf_syms,
        start=pd.Timestamp(start_date).strftime("%Y-%m-%d"),
        end=end_excl,
        interval="1d",
        progress=False,
        auto_adjust=False,
        group_by="ticker",
        threads=True,
    )
    out: dict[str, list[dict]] = {t: [] for t in tickers}
    if raw is None or raw.empty:
        return out

    def _emit(sub: pd.DataFrame, ticker_clean: str) -> None:
        sub = sub.dropna(how="all")
        if sub.empty:
            return
        rows: list[dict] = []
        for ts, r in sub.iterrows():
            try:
                close_v = float(r["Close"])
                qty = float(r["Volume"]) if pd.notna(r["Volume"]) else 0.0
                rows.append({
                    "ticker": ticker_clean,
                    "date": pd.Timestamp(ts).date(),
                    "open": float(r["Open"]),
                    "high": float(r["High"]),
                    "low":  float(r["Low"]),
                    "close": close_v,
                    "volume": qty * close_v,  # TL-equivalent
                    "quantity": qty,
                    "bars_source": "yfinance_d",
                })
            except (KeyError, ValueError, TypeError):
                continue
        out[ticker_clean] = rows

    if isinstance(raw.columns, pd.MultiIndex):
        for tk_yf in yf_syms:
            tk_clean = _strip(tk_yf)
            if tk_yf not in raw.columns.get_level_values(0):
                continue
            sub = raw[tk_yf][["Open", "High", "Low", "Close", "Volume"]]
            _emit(sub, tk_clean)
    else:
        sub = raw[["Open", "High", "Low", "Close", "Volume"]]
        _emit(sub, _strip(yf_syms[0]))
    return out


def fetch_per_ticker(
    tickers: Iterable[str],
    start_date: date | datetime | pd.Timestamp,
    end_date: date | datetime | pd.Timestamp,
) -> dict[str, FetchResult]:
    s = pd.Timestamp(start_date).date()
    e = pd.Timestamp(end_date).date()
    tk_list = list(tickers)
    out: dict[str, FetchResult] = {}
    if not tk_list:
        return out
    try:
        rows_by_tk = _yf_batch(tk_list, s, e)
    except Exception as exc:
        for tk in tk_list:
            out[tk] = FetchResult(
                ticker=tk, start_date=s, end_date=e,
                bars_source=None, rows=[], note="fetch_error",
                detail=f"{type(exc).__name__}: {str(exc)[:160]}",
            )
        return out
    for tk in tk_list:
        rows = rows_by_tk.get(tk, [])
        if rows:
            out[tk] = FetchResult(
                ticker=tk, start_date=s, end_date=e,
                bars_source="yfinance_d", rows=rows, note="ok",
            )
        else:
            out[tk] = FetchResult(
                ticker=tk, start_date=s, end_date=e,
                bars_source=None, rows=[], note="empty",
                detail="yfinance returned no rows",
            )
    return out
