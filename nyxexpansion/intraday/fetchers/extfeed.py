"""External 15m bar fetcher (markets.extfeed) for the layered orchestrator.

Tier-2 source between Fintables (delayed batch) and Matriks (live, rate-limited).
Provides cookie-authenticated 15m bars with no per-ticker rate limit. Bars are
filtered to the target trading day in Europe/Istanbul.

bar_ts convention: extfeed `time` is bar-OPEN unix seconds; we shift by the
timeframe duration (+15 min for 15m) to match the project's bar-CLOSE
``bar_ts`` (UTC ms) convention used by matriks/yfinance fetchers.
"""
from __future__ import annotations

import os
from datetime import date, datetime
from typing import Iterable

import pandas as pd

from nyxexpansion.intraday.fetchers.base import (
    AuthError,
    FetchResult,
    FetcherError,
)


_TIMEFRAME = "15"
_BAR_DURATION_MIN = 15
_N_BARS_REQUEST = 60  # one trading day = 32 × 15m bars; buffer for tz alignment


def _ensure_env() -> None:
    if not os.environ.get("INTRADAY_SID") or not os.environ.get("INTRADAY_SIGN"):
        raise AuthError("INTRADAY_SID / INTRADAY_SIGN env vars not set")


def _fetch_one(auth, ticker: str, target_date) -> list[dict]:
    from markets.extfeed import fetch_bars

    target = pd.Timestamp(target_date).normalize()
    target_date_obj = target.date()
    symbol = f"BIST:{ticker}"

    df = fetch_bars(symbol, _TIMEFRAME, _N_BARS_REQUEST, auth=auth, timeout_s=20)
    if df.empty:
        return []

    # df["time"] is bar-OPEN tz-aware Europe/Istanbul. Shift to bar-CLOSE.
    df = df.copy()
    df["close_ts_tr"] = df["time"] + pd.Timedelta(minutes=_BAR_DURATION_MIN)
    df["close_ts_utc"] = df["close_ts_tr"].dt.tz_convert("UTC")

    # Keep bars whose CLOSE falls on target_date (Europe/Istanbul calendar day).
    same_day = df["close_ts_tr"].dt.date == target_date_obj
    df = df[same_day]
    if df.empty:
        return []

    out = []
    for _, row in df.iterrows():
        bar_ts_ms = int(row["close_ts_utc"].value // 1_000_000)
        out.append({
            "ticker": ticker,
            "signal_date": target_date_obj,
            "bar_ts": bar_ts_ms,
            "date": target.strftime("%Y-%m-%d"),
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row["volume"]),
            "quantity": None,
            "bars_source": "extfeed_15m",
        })
    return out


def fetch_per_ticker(
    tickers: Iterable[str],
    target_date: date | datetime | pd.Timestamp,
) -> dict[str, FetchResult]:
    """Per-ticker extfeed fetch — one WS call per ticker, shared JWT cache."""
    target = pd.Timestamp(target_date).normalize()
    target_date_obj = target.date()

    try:
        _ensure_env()
        from markets.extfeed import auth_from_env
        auth = auth_from_env()
        # Warm the JWT once so per-ticker WS calls reuse the cached token.
        auth.token()
    except AuthError as exc:
        return {
            tk: FetchResult(
                ticker=tk, signal_date=target_date_obj,
                bars_source=None, rows=[], note="auth_error",
                detail=str(exc)[:200],
            )
            for tk in tickers
        }
    except Exception as exc:
        return {
            tk: FetchResult(
                ticker=tk, signal_date=target_date_obj,
                bars_source=None, rows=[], note="auth_error",
                detail=f"{type(exc).__name__}: {str(exc)[:160]}",
            )
            for tk in tickers
        }

    out: dict[str, FetchResult] = {}
    for tk in tickers:
        try:
            rows = _fetch_one(auth, tk, target)
        except Exception as exc:
            out[tk] = FetchResult(
                ticker=tk, signal_date=target_date_obj,
                bars_source=None, rows=[], note="fetch_error",
                detail=f"{type(exc).__name__}: {str(exc)[:160]}",
            )
            continue
        if not rows:
            out[tk] = FetchResult(
                ticker=tk, signal_date=target_date_obj,
                bars_source=None, rows=[], note="empty",
                detail="no bars on target trading day",
            )
            continue
        out[tk] = FetchResult(
            ticker=tk, signal_date=target_date_obj,
            bars_source="extfeed_15m", rows=rows, note="ok",
        )
    return out
