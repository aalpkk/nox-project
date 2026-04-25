"""Fintables MCP HTTP fetcher — batch SQL for 15m bars.

Speaks the streamable-http MCP protocol against ``https://evo.fintables.com/mcp``
using a Bearer token (set ``FINTABLES_MCP_TOKEN`` env var). One ``veri_sorgula``
call pulls all today's candidates' 15m bars in a single SQL — no per-ticker
rate limiting, unlike the Matriks fallback.

Schema mapping (output → existing cache convention):
- ``zaman_utc``                       → ``bar_ts`` (ms since epoch, UTC; close time)
- ``kod``                             → ``ticker``
- ``acilis_15_dk_gecikmeli``          → ``open``
- ``yuksek_15_dk_gecikmeli``          → ``high``
- ``dusuk_15_dk_gecikmeli``           → ``low``
- ``kapanis_15_dk_gecikmeli``         → ``close``
- ``hacim_15_dk_gecikmeli``           → ``volume`` (TL — matches Matriks)
- ``islem_adedi_15_dk_gecikmeli``     → ``quantity`` (shares)

The Fintables `hacim_*` field is TL, and the canonical Matriks ``volume`` field
is also TL (verified: ``close × quantity ≈ volume`` on the existing cache),
so the two sources are unit-compatible at the truncated-bar layer.
"""
from __future__ import annotations

import json
import os
import re
from datetime import date, datetime
from typing import Iterable

import pandas as pd
import requests

from nyxexpansion.intraday.fetchers.base import (
    AuthError,
    EmptyResult,
    FetchResult,
    FetcherError,
)

MCP_URL = "https://evo.fintables.com/mcp"
PROTOCOL_VERSION = "2024-11-05"
CLIENT_NAME = "nox-nyxexp-retention"
CLIENT_VERSION = "1.0"

DEFAULT_TIMEOUT_S = 60
TICKER_RE = re.compile(r"^[A-Z][A-Z0-9]{1,9}$")


class FintablesMCPClient:
    """Minimal streamable-http MCP client (initialize → tools/call)."""

    def __init__(
        self,
        token: str | None = None,
        url: str = MCP_URL,
        timeout: int = DEFAULT_TIMEOUT_S,
    ):
        self.token = token or os.environ.get("FINTABLES_MCP_TOKEN")
        if not self.token:
            raise AuthError("FINTABLES_MCP_TOKEN env var is empty")
        self.url = url
        self.timeout = timeout
        self.session_id: str | None = None
        self._req_id = 0
        self._initialized = False

    def _next_id(self) -> int:
        self._req_id += 1
        return self._req_id

    def _headers(self) -> dict:
        h = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if self.session_id:
            h["Mcp-Session-Id"] = self.session_id
        return h

    @staticmethod
    def _parse_sse(text: str) -> dict:
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("data:"):
                payload = line[5:].strip()
                if payload:
                    return json.loads(payload)
        raise FetcherError("Empty SSE response from Fintables MCP")

    def _post(self, payload: dict) -> dict:
        r = requests.post(
            self.url, headers=self._headers(), json=payload, timeout=self.timeout,
        )
        if r.status_code in (401, 403):
            raise AuthError(
                f"Fintables MCP {r.status_code}: token expired or unauthorized"
            )
        if r.status_code >= 400:
            raise FetcherError(
                f"Fintables MCP HTTP {r.status_code}: {r.text[:200]}"
            )
        sid = r.headers.get("Mcp-Session-Id") or r.headers.get("mcp-session-id")
        if sid and not self.session_id:
            self.session_id = sid
        ct = r.headers.get("Content-Type", "")
        if "text/event-stream" in ct:
            return self._parse_sse(r.text)
        return r.json()

    def initialize(self) -> None:
        if self._initialized:
            return
        resp = self._post({
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "initialize",
            "params": {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": CLIENT_NAME, "version": CLIENT_VERSION},
            },
        })
        if "error" in resp:
            err = resp["error"]
            raise FetcherError(f"Fintables MCP initialize: {err.get('message', err)}")
        try:
            requests.post(
                self.url, headers=self._headers(),
                json={"jsonrpc": "2.0", "method": "notifications/initialized"},
                timeout=self.timeout,
            )
        except Exception:
            pass
        self._initialized = True

    def call_tool(self, name: str, arguments: dict) -> dict | str | None:
        self.initialize()
        resp = self._post({
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
        })
        if "error" in resp:
            err = resp["error"]
            raise FetcherError(f"Fintables MCP {name}: {err.get('message', err)}")
        result = resp.get("result", {})
        content = result.get("content") or []
        if not content:
            return None
        first = content[0]
        text = first.get("text", "") if isinstance(first, dict) else str(first)
        try:
            return json.loads(text)
        except Exception:
            return text


def _parse_markdown_table(table_md: str) -> list[dict]:
    """Parse the markdown table body returned by ``veri_sorgula``."""
    lines = [ln for ln in table_md.splitlines() if ln.strip().startswith("|")]
    if len(lines) < 3:
        return []
    header = [c.strip() for c in lines[0].strip().strip("|").split("|")]
    rows = []
    for ln in lines[2:]:
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        if len(cells) != len(header):
            continue
        rows.append(dict(zip(header, cells)))
    return rows


def _coerce_float(s: str) -> float:
    if s == "" or s.lower() in {"null", "none", "nan"}:
        return float("nan")
    return float(s)


def _coerce_bar_ts_ms(zaman_utc_str: str) -> int:
    """Convert ISO timestamp (e.g. ``2026-04-22T15:00:00.000Z``) to ms epoch."""
    s = zaman_utc_str.replace("Z", "+00:00")
    return int(pd.Timestamp(s).value // 1_000_000)


def _validate_tickers(tickers: Iterable[str]) -> list[str]:
    out = []
    for t in tickers:
        t = str(t).strip().upper()
        if not TICKER_RE.match(t):
            raise FetcherError(f"Refusing to query unsafe ticker code: {t!r}")
        out.append(t)
    if not out:
        raise FetcherError("Empty ticker list")
    return out


def fetch_batch(
    tickers: Iterable[str],
    target_date: date | datetime | pd.Timestamp,
    *,
    client: FintablesMCPClient | None = None,
    table: str = "mumlar_15dk_gf",
) -> list[dict]:
    """Pull 15m bars for ``tickers`` on ``target_date`` in one SQL call.

    Returns rows with the canonical schema (``BAR_COLUMNS``) tagged
    ``bars_source='fintables_15m'``. Caller handles per-ticker missing fills.
    """
    safe_tickers = _validate_tickers(tickers)
    cli = client or FintablesMCPClient()
    target = pd.Timestamp(target_date).normalize()
    next_day = (target + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    target_str = target.strftime("%Y-%m-%d")
    in_clause = ", ".join(f"'{t}'" for t in safe_tickers)
    sql = (
        "SELECT zaman_utc, kod, "
        "acilis_15_dk_gecikmeli, yuksek_15_dk_gecikmeli, "
        "dusuk_15_dk_gecikmeli, kapanis_15_dk_gecikmeli, "
        "hacim_15_dk_gecikmeli, islem_adedi_15_dk_gecikmeli "
        f"FROM {table} "
        f"WHERE kod IN ({in_clause}) "
        f"AND zaman_utc >= TIMESTAMP '{target_str} 00:00:00' "
        f"AND zaman_utc <  TIMESTAMP '{next_day} 00:00:00' "
        "ORDER BY kod, zaman_utc"
    )
    payload = cli.call_tool("veri_sorgula", {
        "sql": sql,
        "purpose": "nyxexp retention 15m bars batch fetch",
    })
    if not isinstance(payload, dict):
        raise FetcherError(f"Unexpected veri_sorgula payload type: {type(payload)}")
    table_md = payload.get("table") or ""
    raw_rows = _parse_markdown_table(table_md) if table_md else []
    if not raw_rows:
        raise EmptyResult(f"Fintables returned 0 rows for {len(safe_tickers)} ticker(s) on {target_str}")

    target_date_obj = target.date() if hasattr(target, "date") else target
    out = []
    for r in raw_rows:
        try:
            ts_ms = _coerce_bar_ts_ms(r["zaman_utc"])
            out.append({
                "ticker": r["kod"],
                "signal_date": target_date_obj,
                "bar_ts": ts_ms,
                "date": pd.Timestamp(ts_ms, unit="ms", tz="UTC")
                          .strftime("%Y-%m-%d"),
                "open": _coerce_float(r["acilis_15_dk_gecikmeli"]),
                "high": _coerce_float(r["yuksek_15_dk_gecikmeli"]),
                "low": _coerce_float(r["dusuk_15_dk_gecikmeli"]),
                "close": _coerce_float(r["kapanis_15_dk_gecikmeli"]),
                "volume": _coerce_float(r["hacim_15_dk_gecikmeli"]),
                "quantity": _coerce_float(r["islem_adedi_15_dk_gecikmeli"]),
                "bars_source": "fintables_15m",
            })
        except (KeyError, ValueError) as exc:
            raise FetcherError(f"Row parse failed: {exc} on row {r!r}")
    return out


def fetch_per_ticker(
    tickers: Iterable[str],
    target_date: date | datetime | pd.Timestamp,
    *,
    client: FintablesMCPClient | None = None,
) -> dict[str, FetchResult]:
    """Try a single batch fetch; return a per-ticker FetchResult mapping.

    On batch-level auth/HTTP failure, every ticker FetchResult is marked failed
    so the orchestrator can fall through to Matriks for the entire batch.
    """
    target = pd.Timestamp(target_date).normalize()
    target_date_obj = target.date()
    safe_tickers = _validate_tickers(tickers)

    out: dict[str, FetchResult] = {}
    try:
        rows = fetch_batch(safe_tickers, target, client=client)
    except AuthError as exc:
        for tk in safe_tickers:
            out[tk] = FetchResult(
                ticker=tk, signal_date=target_date_obj,
                bars_source=None, rows=[], note="auth_error",
                detail=str(exc)[:200],
            )
        return out
    except EmptyResult as exc:
        for tk in safe_tickers:
            out[tk] = FetchResult(
                ticker=tk, signal_date=target_date_obj,
                bars_source=None, rows=[], note="empty",
                detail=str(exc)[:200],
            )
        return out
    except FetcherError as exc:
        for tk in safe_tickers:
            out[tk] = FetchResult(
                ticker=tk, signal_date=target_date_obj,
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
                ticker=tk, signal_date=target_date_obj,
                bars_source="fintables_15m", rows=tk_rows, note="ok",
            )
        else:
            out[tk] = FetchResult(
                ticker=tk, signal_date=target_date_obj,
                bars_source=None, rows=[], note="empty",
                detail="ticker missing from batch result",
            )
    return out
