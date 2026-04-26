"""WebSocket client for the external intraday bar feed.

Public API:
    fetch_bars(symbol, timeframe, n_bars) -> pd.DataFrame
    fetch_bars_until(symbol, timeframe, until_date, ...) -> pd.DataFrame

Wire protocol (frame-encoded, length-prefixed JSON):
    1. set_auth_token(<jwt>)
    2. chart_create_session(<session_id>, "")
    3. resolve_symbol(<session_id>, <symbol_alias>, '={"symbol":"...","adjustment":"splits"}')
    4. create_series(<session_id>, <series_id>, <sub_id>, <symbol_alias>, <tf>, <n>, "")
    Server replies with timescale_update frames (bars) and series_completed (done).
    For backward pagination beyond the initial chunk:
    5. request_more_data(<session_id>, <series_id>, <n_more>)
    Server pushes additional timescale_update frames then series_completed again.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import secrets
import string
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
import websockets

from .auth import AuthCache, auth_from_env

_DEFAULT_WS_URL = "wss://prodata.tradingview.com/socket.io/websocket?type=chart"
_DEFAULT_HOST = "www.tradingview.com"
_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.5 Safari/605.1.15"
)

_FRAME_HEADER_RE = re.compile(r"~m~(\d+)~m~")


def _encode_frame(payload: str) -> str:
    return f"~m~{len(payload)}~m~{payload}"


def _encode_msg(method: str, *params) -> str:
    return _encode_frame(json.dumps({"m": method, "p": list(params)}))


def _rand_id(prefix: str, n: int = 12) -> str:
    alphabet = string.ascii_lowercase + string.digits
    return prefix + "".join(secrets.choice(alphabet) for _ in range(n))


@dataclass
class _Session:
    chart_session: str = field(default_factory=lambda: _rand_id("cs_"))
    symbol_alias: str = "symbol_1"
    series_id: str = "sds_1"
    sub_id: str = "s1"


async def _fetch_bars_async(
    auth: AuthCache,
    symbol: str,
    timeframe: str,
    n_bars: int,
    timeout_s: float,
) -> pd.DataFrame:
    ws_url = os.getenv("INTRADAY_WS_URL", _DEFAULT_WS_URL)
    host = os.getenv("INTRADAY_HOST", _DEFAULT_HOST)
    headers = {
        "Origin": f"https://{host}",
        "User-Agent": _USER_AGENT,
    }

    bars: list[list[float]] = []
    completed = asyncio.Event()
    error: list[str] = []

    async with websockets.connect(
        ws_url,
        additional_headers=headers,
        max_size=2 ** 24,
        ping_interval=None,
    ) as ws:
        sess = _Session()
        token = auth.token()

        # outbound message sequence
        outbound = [
            _encode_msg("set_auth_token", token),
            _encode_msg("chart_create_session", sess.chart_session, ""),
            _encode_msg(
                "resolve_symbol",
                sess.chart_session,
                sess.symbol_alias,
                f'={{"symbol":"{symbol}","adjustment":"splits"}}',
            ),
            _encode_msg(
                "create_series",
                sess.chart_session,
                sess.series_id,
                sess.sub_id,
                sess.symbol_alias,
                timeframe,
                n_bars,
                "",
            ),
        ]
        for frame in outbound:
            await ws.send(frame)

        async def _read_loop():
            buf = ""
            while not completed.is_set():
                chunk = await ws.recv()
                if isinstance(chunk, bytes):
                    chunk = chunk.decode("utf-8", errors="ignore")
                buf += chunk

                while True:
                    m = _FRAME_HEADER_RE.match(buf)
                    if not m:
                        break
                    length = int(m.group(1))
                    start = m.end()
                    end = start + length
                    if end > len(buf):
                        break
                    payload = buf[start:end]
                    buf = buf[end:]

                    # Heartbeat — echo back
                    if payload.startswith("~h~"):
                        await ws.send(_encode_frame(payload))
                        continue

                    try:
                        msg = json.loads(payload)
                    except json.JSONDecodeError:
                        continue

                    method = msg.get("m")
                    params = msg.get("p", [])
                    if method == "timescale_update":
                        if len(params) >= 2 and isinstance(params[1], dict):
                            for ser in params[1].values():
                                if isinstance(ser, dict):
                                    for pt in ser.get("s", []):
                                        bars.append(pt["v"])
                    elif method == "series_completed":
                        completed.set()
                        return
                    elif method in ("protocol_error", "critical_error"):
                        error.append(json.dumps(params))
                        completed.set()
                        return

        try:
            await asyncio.wait_for(_read_loop(), timeout=timeout_s)
        except asyncio.TimeoutError:
            pass

    if error:
        raise RuntimeError(f"server error for {symbol}: {error[0]}")
    if not bars:
        raise RuntimeError(
            f"no bars received for {symbol} (tf={timeframe}, n={n_bars}) — "
            "check symbol format (e.g. 'BIST:THYAO') and JWT permissions"
        )

    df = pd.DataFrame(bars, columns=["time", "open", "high", "low", "close", "volume"])
    df["time"] = (
        pd.to_datetime(df["time"], unit="s", utc=True)
        .dt.tz_convert("Europe/Istanbul")
    )
    df = df.drop_duplicates(subset=["time"]).sort_values("time").reset_index(drop=True)
    return df


def fetch_bars(
    symbol: str,
    timeframe: str = "15",
    n_bars: int = 200,
    auth: Optional[AuthCache] = None,
    timeout_s: float = 30.0,
) -> pd.DataFrame:
    """Sync wrapper.

    Args:
        symbol: e.g. 'BIST:THYAO'
        timeframe: '1', '5', '15', '60', '240', 'D', 'W'
        n_bars: number of historical bars to request
        auth: optional pre-built AuthCache; defaults to env-based
        timeout_s: total wall-clock timeout

    Returns:
        DataFrame with columns [time, open, high, low, close, volume],
        time in Europe/Istanbul.
    """
    if auth is None:
        auth = auth_from_env()
    return asyncio.run(_fetch_bars_async(auth, symbol, timeframe, n_bars, timeout_s))


async def _fetch_bars_until_async(
    auth: AuthCache,
    symbol: str,
    timeframe: str,
    until_ts: pd.Timestamp,
    chunk_n: int,
    max_chunks: int,
    chunk_timeout_s: float,
    inter_chunk_delay_s: float,
) -> tuple[pd.DataFrame, dict]:
    """Open a single WS session, seed with create_series, then loop
    request_more_data until the oldest bar falls on/before `until_ts` (UTC),
    `max_chunks` is reached, or the server stops returning new bars.

    Returns (DataFrame, stats_dict).
    """
    ws_url = os.getenv("INTRADAY_WS_URL", _DEFAULT_WS_URL)
    host = os.getenv("INTRADAY_HOST", _DEFAULT_HOST)
    headers = {
        "Origin": f"https://{host}",
        "User-Agent": _USER_AGENT,
    }

    bars: list[list[float]] = []
    chunk_completed = asyncio.Event()
    error: list[str] = []
    stats = {
        "chunks_sent": 0,
        "chunks_received": 0,
        "no_progress_break": False,
        "reached_target": False,
        "max_chunks_break": False,
    }

    until_unix = int(pd.Timestamp(until_ts).timestamp())

    async with websockets.connect(
        ws_url,
        additional_headers=headers,
        max_size=2 ** 24,
        ping_interval=None,
    ) as ws:
        sess = _Session()
        token = auth.token()

        outbound = [
            _encode_msg("set_auth_token", token),
            _encode_msg("chart_create_session", sess.chart_session, ""),
            _encode_msg(
                "resolve_symbol",
                sess.chart_session,
                sess.symbol_alias,
                f'={{"symbol":"{symbol}","adjustment":"splits"}}',
            ),
            _encode_msg(
                "create_series",
                sess.chart_session,
                sess.series_id,
                sess.sub_id,
                sess.symbol_alias,
                timeframe,
                chunk_n,
                "",
            ),
        ]
        for frame in outbound:
            await ws.send(frame)
        stats["chunks_sent"] = 1

        async def reader():
            buf = ""
            while True:
                chunk = await ws.recv()
                if isinstance(chunk, bytes):
                    chunk = chunk.decode("utf-8", errors="ignore")
                buf += chunk
                while True:
                    m = _FRAME_HEADER_RE.match(buf)
                    if not m:
                        break
                    length = int(m.group(1))
                    start = m.end()
                    end = start + length
                    if end > len(buf):
                        break
                    payload = buf[start:end]
                    buf = buf[end:]
                    if payload.startswith("~h~"):
                        await ws.send(_encode_frame(payload))
                        continue
                    try:
                        msg = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    method = msg.get("m")
                    params = msg.get("p", [])
                    if method == "timescale_update":
                        if len(params) >= 2 and isinstance(params[1], dict):
                            for ser in params[1].values():
                                if isinstance(ser, dict):
                                    for pt in ser.get("s", []):
                                        bars.append(pt["v"])
                    elif method == "series_completed":
                        chunk_completed.set()
                    elif method in ("protocol_error", "critical_error"):
                        error.append(json.dumps(params))
                        chunk_completed.set()
                        return

        reader_task = asyncio.create_task(reader())

        try:
            await asyncio.wait_for(chunk_completed.wait(), timeout=chunk_timeout_s)
            chunk_completed.clear()
            stats["chunks_received"] = 1

            for _ in range(max_chunks - 1):
                if error:
                    break
                if not bars:
                    break
                oldest = min(b[0] for b in bars)
                if oldest <= until_unix:
                    stats["reached_target"] = True
                    break

                if inter_chunk_delay_s > 0:
                    await asyncio.sleep(inter_chunk_delay_s)

                await ws.send(_encode_msg(
                    "request_more_data",
                    sess.chart_session,
                    sess.series_id,
                    chunk_n,
                ))
                stats["chunks_sent"] += 1

                prev_count = len(bars)
                try:
                    await asyncio.wait_for(
                        chunk_completed.wait(), timeout=chunk_timeout_s
                    )
                except asyncio.TimeoutError:
                    break
                chunk_completed.clear()
                stats["chunks_received"] += 1
                if len(bars) == prev_count:
                    stats["no_progress_break"] = True
                    break
            else:
                stats["max_chunks_break"] = True
        finally:
            reader_task.cancel()
            try:
                await reader_task
            except (asyncio.CancelledError, Exception):
                pass

    if error:
        raise RuntimeError(f"server error for {symbol}: {error[0]}")
    if not bars:
        raise RuntimeError(
            f"no bars received for {symbol} (tf={timeframe}) — "
            "check symbol format and JWT permissions"
        )

    df = pd.DataFrame(bars, columns=["time", "open", "high", "low", "close", "volume"])
    df["time"] = (
        pd.to_datetime(df["time"], unit="s", utc=True)
        .dt.tz_convert("Europe/Istanbul")
    )
    df = df.drop_duplicates(subset=["time"]).sort_values("time").reset_index(drop=True)
    return df, stats


def fetch_bars_until(
    symbol: str,
    timeframe: str,
    until_date,
    chunk_n: int = 5000,
    max_chunks: int = 50,
    chunk_timeout_s: float = 30.0,
    inter_chunk_delay_s: float = 0.5,
    auth: Optional[AuthCache] = None,
) -> tuple[pd.DataFrame, dict]:
    """Pull bars from the latest available point backward until the oldest bar
    is on or before ``until_date`` (interpreted as UTC midnight if naive, or
    converted to UTC if tz-aware).

    Single WebSocket session — initial ``create_series`` then ``request_more_data``
    loop. ``stats`` dict reports chunks sent/received and termination reason
    (``reached_target`` / ``no_progress_break`` / ``max_chunks_break``).

    Args:
        symbol: e.g. 'BIST:THYAO'
        timeframe: '1', '5', '15', '60', '240', 'D'
        until_date: target start date — string, datetime, or Timestamp.
        chunk_n: bars per request (TV server typically caps at ~5000).
        max_chunks: safety cap on total request_more_data iterations.
        chunk_timeout_s: per-chunk wait.
        inter_chunk_delay_s: throttle between chunks (politeness).
        auth: optional pre-built AuthCache; defaults to env-based.
    """
    if auth is None:
        auth = auth_from_env()
    until_ts = pd.Timestamp(until_date)
    if until_ts.tzinfo is None:
        until_ts = until_ts.tz_localize("UTC")
    else:
        until_ts = until_ts.tz_convert("UTC")
    return asyncio.run(_fetch_bars_until_async(
        auth=auth,
        symbol=symbol,
        timeframe=timeframe,
        until_ts=until_ts,
        chunk_n=chunk_n,
        max_chunks=max_chunks,
        chunk_timeout_s=chunk_timeout_s,
        inter_chunk_delay_s=inter_chunk_delay_s,
    ))
