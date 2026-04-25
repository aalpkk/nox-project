"""External daily bar fetcher (markets.extfeed) for the layered orchestrator.

Tier-2 source between Fintables (delayed batch SQL) and Matriks (live, but
60-bar daily cap). Cookie-authenticated TradingView WS feed; one call per
ticker covers the requested window (TV limit ~5000 daily bars per stream).

Bars are emitted in the canonical daily schema and filtered to the
[start_date, end_date] Europe/Istanbul calendar window.
"""
from __future__ import annotations

import os
from datetime import date, datetime
from typing import Iterable

import pandas as pd

from nyxexpansion.daily.fetchers.base import (
    AuthError,
    FetchResult,
)


_TIMEFRAME = "D"
_BAR_BUFFER = 50  # request a bit more than business-day estimate, in case of holidays


def _ensure_env() -> None:
    if not os.environ.get("INTRADAY_SID") or not os.environ.get("INTRADAY_SIGN"):
        raise AuthError("INTRADAY_SID / INTRADAY_SIGN env vars not set")


def _expected_bdays(s: date, e: date) -> int:
    if e < s:
        return 0
    return max(1, len(pd.bdate_range(s, e)))


def _fetch_one(auth, ticker: str, start_d: date, end_d: date, n_bars: int) -> list[dict]:
    from markets.extfeed import fetch_bars

    symbol = f"BIST:{ticker}"
    df = fetch_bars(symbol, _TIMEFRAME, n_bars, auth=auth, timeout_s=20)
    if df.empty:
        return []

    df = df.copy()
    # `time` for daily bars is bar-open tz-aware Europe/Istanbul; the calendar
    # date of that timestamp IS the trading day.
    df["bar_date"] = df["time"].dt.date
    mask = (df["bar_date"] >= start_d) & (df["bar_date"] <= end_d)
    df = df[mask]
    if df.empty:
        return []

    out: list[dict] = []
    for _, row in df.iterrows():
        try:
            close_v = float(row["close"])
            qty = float(row["volume"]) if pd.notna(row["volume"]) else 0.0
        except (KeyError, ValueError, TypeError):
            continue
        out.append({
            "ticker": ticker,
            "date": row["bar_date"],
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low":  float(row["low"]),
            "close": close_v,
            "volume": qty * close_v,  # TL-equivalent for parity with upstream tiers
            "quantity": qty,
            "bars_source": "extfeed_d",
        })
    return out


def fetch_per_ticker(
    tickers: Iterable[str],
    start_date: date | datetime | pd.Timestamp,
    end_date: date | datetime | pd.Timestamp,
) -> dict[str, FetchResult]:
    """Per-ticker extfeed daily fetch — one WS call per ticker, shared JWT cache."""
    s = pd.Timestamp(start_date).date()
    e = pd.Timestamp(end_date).date()
    n_bars = _expected_bdays(s, e) + _BAR_BUFFER

    try:
        _ensure_env()
        from markets.extfeed import auth_from_env
        auth = auth_from_env()
        auth.token()  # warm JWT cache
    except AuthError as exc:
        return {
            tk: FetchResult(
                ticker=tk, start_date=s, end_date=e,
                bars_source=None, rows=[], note="auth_error",
                detail=str(exc)[:200],
            )
            for tk in tickers
        }
    except Exception as exc:
        return {
            tk: FetchResult(
                ticker=tk, start_date=s, end_date=e,
                bars_source=None, rows=[], note="auth_error",
                detail=f"{type(exc).__name__}: {str(exc)[:160]}",
            )
            for tk in tickers
        }

    expected = _expected_bdays(s, e)
    out: dict[str, FetchResult] = {}
    for tk in tickers:
        try:
            rows = _fetch_one(auth, tk, s, e, n_bars)
        except Exception as exc:
            out[tk] = FetchResult(
                ticker=tk, start_date=s, end_date=e,
                bars_source=None, rows=[], note="fetch_error",
                detail=f"{type(exc).__name__}: {str(exc)[:160]}",
            )
            continue
        if not rows:
            out[tk] = FetchResult(
                ticker=tk, start_date=s, end_date=e,
                bars_source=None, rows=[], note="empty",
                detail="no bars in requested window",
            )
            continue
        coverage = len(rows) / expected if expected else 1.0
        if coverage < 0.5:
            out[tk] = FetchResult(
                ticker=tk, start_date=s, end_date=e,
                bars_source=None, rows=[], note="partial",
                detail=f"got {len(rows)}/{expected} bdays ({coverage:.0%})",
            )
            continue
        out[tk] = FetchResult(
            ticker=tk, start_date=s, end_date=e,
            bars_source="extfeed_d", rows=rows, note="ok",
        )
    return out
