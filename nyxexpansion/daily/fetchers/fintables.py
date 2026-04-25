"""Fintables MCP daily fetcher — ``mumlar_gunluk_gh`` table.

Reuses the streamable-http MCP client from the intraday tier. Daily bars
come from ``mumlar_gunluk_gh`` with this column convention (verified
2026-04-25):

- ``zaman_utc``  D 21:00 UTC = D+1 00:00 Istanbul → represents Istanbul day D+1
- ``kod``        ticker code
- ``acilis``     daily open
- ``yuksek``     daily high
- ``dusuk``      daily low
- ``kapanis``    daily close
- ``islem_adedi_15_dk_gecikmeli`` share count (15-min delayed)
- ``hacim_15_dk_gecikmeli``       TL volume (15-min delayed)

The ``veri_sorgula`` tool has a hard 300-row cap per call. We auto-page by
splitting the ticker batch and the date range.
"""
from __future__ import annotations

import os
from datetime import date, datetime, timedelta
from typing import Iterable

import pandas as pd

from nyxexpansion.daily.fetchers.base import (
    AuthError,
    EmptyResult,
    FetchResult,
    FetcherError,
)
from nyxexpansion.intraday.fetchers.fintables import (
    FintablesMCPClient,
    _coerce_float,
    _parse_markdown_table,
    _validate_tickers,
)

DAILY_TABLE = "mumlar_gunluk_gh"
ROW_CAP = 300                # MCP veri_sorgula hard cap
SAFE_BATCH_ROWS = 280        # leave headroom for boundary noise
MAX_CALLS_PER_PULL = 80      # circuit-break: skip fintables tier if we'd need more


def _zaman_utc_to_istanbul_date(zaman_utc_str: str) -> date:
    """`2026-04-23T21:00:00.000Z` → Istanbul date (Apr 24)."""
    s = zaman_utc_str.replace("Z", "+00:00")
    return (pd.Timestamp(s).tz_convert("Europe/Istanbul") + pd.Timedelta(hours=0)).date()


def _utc_window(start_date: date, end_date: date) -> tuple[str, str]:
    """Istanbul [start, end] (inclusive both) → UTC half-open [start_lo, end_hi)."""
    start_ts = pd.Timestamp(start_date).tz_localize("Europe/Istanbul")
    end_ts_excl = (pd.Timestamp(end_date) + pd.Timedelta(days=1)).tz_localize("Europe/Istanbul")
    start_utc = start_ts.tz_convert("UTC").strftime("%Y-%m-%d %H:%M:%S")
    end_utc = end_ts_excl.tz_convert("UTC").strftime("%Y-%m-%d %H:%M:%S")
    return start_utc, end_utc


def _build_sql(tickers: list[str], start_date: date, end_date: date,
               *, table: str = DAILY_TABLE) -> str:
    in_clause = ", ".join(f"'{t}'" for t in tickers)
    start_utc, end_utc = _utc_window(start_date, end_date)
    return (
        "SELECT zaman_utc, kod, acilis, yuksek, dusuk, kapanis, "
        "islem_adedi_15_dk_gecikmeli, hacim_15_dk_gecikmeli "
        f"FROM {table} "
        f"WHERE kod IN ({in_clause}) "
        f"AND zaman_utc >= TIMESTAMP '{start_utc}' "
        f"AND zaman_utc <  TIMESTAMP '{end_utc}' "
        "ORDER BY kod, zaman_utc"
    )


def _parse_response_rows(payload: dict) -> list[dict]:
    table_md = payload.get("table") or ""
    if not table_md:
        return []
    raw = _parse_markdown_table(table_md)
    out: list[dict] = []
    for r in raw:
        try:
            d = _zaman_utc_to_istanbul_date(r["zaman_utc"])
            close_val = _coerce_float(r["kapanis"])
            qty = _coerce_float(r["islem_adedi_15_dk_gecikmeli"])
            tl_vol = _coerce_float(r.get("hacim_15_dk_gecikmeli", "")) if "hacim_15_dk_gecikmeli" in r else float("nan")
            volume = tl_vol if tl_vol == tl_vol else (qty * close_val)  # NaN check
            out.append({
                "ticker": r["kod"],
                "date": d,
                "open": _coerce_float(r["acilis"]),
                "high": _coerce_float(r["yuksek"]),
                "low":  _coerce_float(r["dusuk"]),
                "close": close_val,
                "volume": volume,
                "quantity": qty,
                "bars_source": "fintables_d",
            })
        except (KeyError, ValueError):
            continue
    return out


def _fetch_window(
    cli: FintablesMCPClient,
    tickers: list[str],
    start_date: date,
    end_date: date,
) -> list[dict]:
    """Single MCP call. Caller must size the request under ROW_CAP."""
    sql = _build_sql(tickers, start_date, end_date)
    payload = cli.call_tool("veri_sorgula", {
        "sql": sql,
        "purpose": "nox layered daily fetch",
    })
    if not isinstance(payload, dict):
        raise FetcherError(f"Unexpected veri_sorgula payload type: {type(payload).__name__}")
    return _parse_response_rows(payload)


def _date_chunks(start_date: date, end_date: date, days_per_chunk: int) -> list[tuple[date, date]]:
    out = []
    cur = start_date
    while cur <= end_date:
        nxt = min(cur + timedelta(days=days_per_chunk - 1), end_date)
        out.append((cur, nxt))
        cur = nxt + timedelta(days=1)
    return out


def _ticker_chunks(tickers: list[str], n: int) -> list[list[str]]:
    return [tickers[i:i + n] for i in range(0, len(tickers), n)]


def fetch_range(
    tickers: Iterable[str],
    start_date: date | datetime | pd.Timestamp,
    end_date: date | datetime | pd.Timestamp,
    *,
    client: FintablesMCPClient | None = None,
    expected_bars_per_day: float = 1.0,
    bd_per_week: float = 5.0,
) -> list[dict]:
    """Pull daily bars for tickers in [start_date, end_date] (inclusive).

    Auto-pages so each MCP call stays under the 300-row cap. The page-size
    heuristic uses an estimated bar count = tickers_in_chunk × business_days
    × expected_bars_per_day. We pick chunk sizes so the estimate ≤ 280.
    """
    safe_tickers = _validate_tickers(tickers)
    s = pd.Timestamp(start_date).date()
    e = pd.Timestamp(end_date).date()
    if e < s:
        raise FetcherError(f"end_date {e} < start_date {s}")

    total_days = (e - s).days + 1
    bd_estimate = max(1.0, total_days * bd_per_week / 7.0)
    bars_per_ticker_window = bd_estimate * expected_bars_per_day

    # Pre-flight cost estimate. Fintables MCP per-call latency dominates;
    # if the batch would exceed MAX_CALLS_PER_PULL we bail out so the
    # orchestrator can fall through to a faster tier (matriks/yfinance).
    if bars_per_ticker_window <= 1.5:
        max_tk_per_call = max(1, int(SAFE_BATCH_ROWS // max(bars_per_ticker_window, 1.0)))
        n_calls_estimate = -(-len(safe_tickers) // max_tk_per_call)
    else:
        target_rows = 200
        days_per_chunk = max(1, int(target_rows / expected_bars_per_day))
        max_tk_per_call = max(1, int(target_rows // max(days_per_chunk * expected_bars_per_day, 1.0)))
        n_date_chunks = -(-total_days // days_per_chunk)
        n_calls_estimate = (-(-len(safe_tickers) // max_tk_per_call)) * n_date_chunks
    if n_calls_estimate > MAX_CALLS_PER_PULL:
        raise FetcherError(
            f"pull would need {n_calls_estimate} MCP calls "
            f"(> MAX_CALLS_PER_PULL={MAX_CALLS_PER_PULL}); "
            f"falling through to next tier"
        )

    cli = client or FintablesMCPClient()
    out_rows: list[dict] = []
    if bars_per_ticker_window <= 1.5:
        for tk_chunk in _ticker_chunks(safe_tickers, max_tk_per_call):
            try:
                rows = _fetch_window(cli, tk_chunk, s, e)
            except (AuthError, EmptyResult):
                raise
            out_rows.extend(rows)
    else:
        for tk_chunk in _ticker_chunks(safe_tickers, max_tk_per_call):
            for ds, de in _date_chunks(s, e, days_per_chunk):
                try:
                    rows = _fetch_window(cli, tk_chunk, ds, de)
                except EmptyResult:
                    continue
                out_rows.extend(rows)
    return out_rows


def fetch_per_ticker(
    tickers: Iterable[str],
    start_date: date | datetime | pd.Timestamp,
    end_date: date | datetime | pd.Timestamp,
    *,
    client: FintablesMCPClient | None = None,
) -> dict[str, FetchResult]:
    """Per-ticker FetchResult mapping; orchestrator falls through on failure."""
    s = pd.Timestamp(start_date).date()
    e = pd.Timestamp(end_date).date()
    safe_tickers = _validate_tickers(tickers)

    out: dict[str, FetchResult] = {}
    try:
        rows = fetch_range(safe_tickers, s, e, client=client)
    except AuthError as exc:
        for tk in safe_tickers:
            out[tk] = FetchResult(
                ticker=tk, start_date=s, end_date=e,
                bars_source=None, rows=[], note="auth_error",
                detail=str(exc)[:200],
            )
        return out
    except FetcherError as exc:
        for tk in safe_tickers:
            out[tk] = FetchResult(
                ticker=tk, start_date=s, end_date=e,
                bars_source=None, rows=[], note="fetch_error",
                detail=str(exc)[:200],
            )
        return out

    by_ticker: dict[str, list[dict]] = {tk: [] for tk in safe_tickers}
    for r in rows:
        by_ticker.setdefault(r["ticker"], []).append(r)
    for tk in safe_tickers:
        tk_rows = by_ticker.get(tk, [])
        if tk_rows:
            out[tk] = FetchResult(
                ticker=tk, start_date=s, end_date=e,
                bars_source="fintables_d", rows=tk_rows, note="ok",
            )
        else:
            out[tk] = FetchResult(
                ticker=tk, start_date=s, end_date=e,
                bars_source=None, rows=[], note="empty",
                detail="no rows for ticker in batch result",
            )
    return out
