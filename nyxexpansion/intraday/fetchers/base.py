"""Shared types and constants for the layered intraday fetchers."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

BarsSource = Literal["fintables_15m", "extfeed_15m", "matriks_15m", "yfinance_1h"]

BAR_COLUMNS: tuple[str, ...] = (
    "ticker",
    "signal_date",
    "bar_ts",
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "bars_source",
)


class FetcherError(Exception):
    """Generic failure raised by a fetcher; orchestrator falls through."""


class AuthError(FetcherError):
    """Authentication problem (expired token, missing key, 401/403)."""


class EmptyResult(FetcherError):
    """Endpoint responded but returned no bars for the requested key."""


@dataclass
class FetchResult:
    """Outcome of a single (ticker, date) fetch attempt.

    ``rows`` are dicts with the ``BAR_COLUMNS`` schema (already tagged with
    ``bars_source``). ``note`` is a short tag describing what happened
    (``ok``, ``empty``, ``auth_error``, ``http_error``, ``timeout``...).
    """
    ticker: str
    signal_date: object
    bars_source: BarsSource | None
    rows: list[dict] = field(default_factory=list)
    note: str = "ok"
    detail: str = ""
