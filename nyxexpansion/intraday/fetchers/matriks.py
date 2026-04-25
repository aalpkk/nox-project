"""Matriks 15m fetcher wrapper for the layered orchestrator.

Reuses the existing ``agent.matriks_client.MatriksClient`` (rate-limit safe,
429 circuit-breaker), and emits rows in the canonical ``BAR_COLUMNS`` schema
tagged ``bars_source='matriks_15m'``. One historicalData call per
``(ticker, signal_date)``.
"""
from __future__ import annotations

import os
from datetime import date, datetime
from typing import Iterable

import pandas as pd

from nyxexpansion.intraday.fetchers.base import (
    AuthError,
    EmptyResult,
    FetchResult,
    FetcherError,
)


def _make_client():
    if not os.environ.get("MATRIKS_API_KEY"):
        raise AuthError("MATRIKS_API_KEY env var is empty")
    from agent.matriks_client import MatriksClient
    return MatriksClient()


def _fetch_one(client, ticker: str, target_date) -> list[dict]:
    s = pd.Timestamp(target_date).strftime("%Y-%m-%d")
    target_date_obj = pd.Timestamp(target_date).date()
    resp = client.call_tool("historicalData", {
        "symbol": ticker,
        "startDate": s,
        "endDate": s,
        "interval": "15min",
        "rawBars": True,
    })
    if not isinstance(resp, dict):
        return []
    bars = resp.get("allBars") or []
    out = []
    for b in bars:
        if not isinstance(b, dict):
            continue
        out.append({
            "ticker": ticker,
            "signal_date": target_date_obj,
            "bar_ts": b.get("timestamp"),
            "date": b.get("date"),
            "open": b.get("open"),
            "high": b.get("high"),
            "low": b.get("low"),
            "close": b.get("close"),
            "volume": b.get("volume"),
            "quantity": b.get("quantity"),
            "bars_source": "matriks_15m",
        })
    return out


def fetch_per_ticker(
    tickers: Iterable[str],
    target_date: date | datetime | pd.Timestamp,
    *,
    client=None,
) -> dict[str, FetchResult]:
    """Per-ticker Matriks fetch — one historicalData call per ticker."""
    target = pd.Timestamp(target_date).normalize()
    target_date_obj = target.date()

    try:
        cli = client or _make_client()
    except AuthError as exc:
        return {
            tk: FetchResult(
                ticker=tk, signal_date=target_date_obj,
                bars_source=None, rows=[], note="auth_error",
                detail=str(exc)[:200],
            )
            for tk in tickers
        }

    out: dict[str, FetchResult] = {}
    for tk in tickers:
        try:
            rows = _fetch_one(cli, tk, target)
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
                detail="no bars in allBars",
            )
            continue
        out[tk] = FetchResult(
            ticker=tk, signal_date=target_date_obj,
            bars_source="matriks_15m", rows=rows, note="ok",
        )
    return out
