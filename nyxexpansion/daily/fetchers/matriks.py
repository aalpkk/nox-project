"""Matriks daily fetcher — historicalData with rawBars=True, interval=daily.

Reuses ``agent.matriks_client.MatriksClient``. The ``allBars`` response
is hard-capped server-side at 60 daily rows; ``startDate``/``endDate``/
``period``/``barCount`` overrides have no effect (verified 2026-04-25
via ``debug_matriks_daily`` workflow). So matriks_d only serves the
last ~3 months reliably — long bootstraps must fall through to a
batch tier (yfinance). We surface partial coverage as
``note='partial'`` so the layered orchestrator skips the ticker.
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


MIN_COVERAGE = 0.5  # matriks_d ok only if rows ≥ 50% of expected business days


def _expected_bdays(s: date, e: date) -> int:
    if e < s:
        return 0
    return max(1, len(pd.bdate_range(s, e)))


def fetch_per_ticker(
    tickers: Iterable[str],
    start_date: date | datetime | pd.Timestamp,
    end_date: date | datetime | pd.Timestamp,
    *,
    client=None,
) -> dict[str, FetchResult]:
    s = pd.Timestamp(start_date).date()
    e = pd.Timestamp(end_date).date()
    expected = _expected_bdays(s, e)
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
        coverage = len(rows) / expected if expected else 1.0
        if coverage < MIN_COVERAGE:
            out[tk] = FetchResult(
                ticker=tk, start_date=s, end_date=e,
                bars_source=None, rows=[], note="partial",
                detail=f"got {len(rows)}/{expected} bdays "
                       f"({coverage:.0%}); matriks daily caps at ~60",
            )
            continue
        out[tk] = FetchResult(
            ticker=tk, start_date=s, end_date=e,
            bars_source="matriks_d", rows=rows, note="ok",
        )
    return out
