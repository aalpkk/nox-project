"""Layered intraday bar fetchers (Fintables → Matriks → yfinance).

Each fetcher returns a list of bar rows tagged with ``bars_source`` so that
downstream stages (retention, HTML banner, log writer) can surface which
data tier served each candidate.
"""
from nyxexpansion.intraday.fetchers.base import (
    BAR_COLUMNS,
    BarsSource,
    FetchResult,
    FetcherError,
    AuthError,
    EmptyResult,
)

__all__ = [
    "BAR_COLUMNS",
    "BarsSource",
    "FetchResult",
    "FetcherError",
    "AuthError",
    "EmptyResult",
]
