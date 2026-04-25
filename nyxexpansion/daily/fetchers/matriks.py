"""Matriks daily fetcher — historicalData with rawBars=True, interval=daily.

Reuses ``agent.matriks_client.MatriksClient``. Without an explicit
``interval``, Matriks silently returns a small (~3 month) recent
window; the documented enum is ``1min|5min|15min|1hour|daily|
weekly|monthly`` so we always pin ``daily``. One call per ticker
covers the full date range; rate limit handled by the client.
"""
from __future__ import annotations

import os
from datetime import date, datetime
from typing import Iterable

import pandas as pd

from nyxexpansion.daily.fetchers.base import (
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


def _bar_to_canonical(b: dict, ticker: str) -> dict | None:
    if not isinstance(b, dict):
        return None
    raw_date = b.get("date") or b.get("d") or b.get("timestamp")
    if raw_date is None:
        return None
    try:
        if isinstance(raw_date, (int, float)):
            d = pd.Timestamp(raw_date, unit="ms", tz="UTC").tz_convert("Europe/Istanbul").date()
        else:
            d = pd.Timestamp(raw_date).date()
    except Exception:
        return None
    try:
        close_v = float(b.get("close"))
        qty = float(b.get("quantity") or b.get("volume") or 0.0)
    except (TypeError, ValueError):
        return None
    return {
        "ticker": ticker,
        "date": d,
        "open": float(b.get("open")),
        "high": float(b.get("high")),
        "low":  float(b.get("low")),
        "close": close_v,
        "volume": float(b.get("volume", qty * close_v)) if b.get("volume") is not None else qty * close_v,
        "quantity": qty,
        "bars_source": "matriks_d",
    }


def _fetch_one(client, ticker: str, start_date: date, end_date: date) -> list[dict]:
    s = pd.Timestamp(start_date).strftime("%Y-%m-%d")
    e = pd.Timestamp(end_date).strftime("%Y-%m-%d")
    resp = client.call_tool("historicalData", {
        "symbol": ticker,
        "startDate": s,
        "endDate": e,
        "interval": "daily",
        "rawBars": True,
    })
    if not isinstance(resp, dict):
        return []
    bars = resp.get("allBars") or resp.get("bars") or resp.get("rawBars") or []
    out: list[dict] = []
    for b in bars:
        row = _bar_to_canonical(b, ticker)
        if row is not None:
            out.append(row)
    return out


def fetch_per_ticker(
    tickers: Iterable[str],
    start_date: date | datetime | pd.Timestamp,
    end_date: date | datetime | pd.Timestamp,
    *,
    client=None,
) -> dict[str, FetchResult]:
    s = pd.Timestamp(start_date).date()
    e = pd.Timestamp(end_date).date()
    try:
        cli = client or _make_client()
    except AuthError as exc:
        return {
            tk: FetchResult(
                ticker=tk, start_date=s, end_date=e,
                bars_source=None, rows=[], note="auth_error",
                detail=str(exc)[:200],
            )
            for tk in tickers
        }

    out: dict[str, FetchResult] = {}
    for tk in tickers:
        try:
            rows = _fetch_one(cli, tk, s, e)
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
                detail="no daily bars",
            )
            continue
        out[tk] = FetchResult(
            ticker=tk, start_date=s, end_date=e,
            bars_source="matriks_d", rows=rows, note="ok",
        )
    return out
