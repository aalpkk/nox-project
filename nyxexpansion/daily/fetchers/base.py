"""Shared types for the layered daily fetcher.

Daily counterpart of ``nyxexpansion.intraday.fetchers.base``. The schema is
flat per-bar dicts; the orchestrator merges per-ticker FetchResults into a
single pandas frame.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Literal

DailyBarsSource = Literal["fintables_d", "extfeed_d", "matriks_d", "yfinance_d"]

DAILY_BAR_COLUMNS: tuple[str, ...] = (
    "ticker",
    "date",        # Istanbul calendar date (datetime.date)
    "open",
    "high",
    "low",
    "close",
    "volume",      # TL volume when available; else close * quantity
    "quantity",    # share count when available
    "bars_source",
)


class FetcherError(Exception):
    """Generic failure raised by a tier; orchestrator falls through."""


class AuthError(FetcherError):
    """Authentication/credential problem (401/403/missing env)."""


class EmptyResult(FetcherError):
    """Endpoint responded but returned no daily bars."""


@dataclass
class FetchResult:
    """Outcome for a single ticker over [start_date, end_date]."""
    ticker: str
    start_date: date
    end_date: date
    bars_source: DailyBarsSource | None
    rows: list[dict] = field(default_factory=list)
    note: str = "ok"
    detail: str = ""
