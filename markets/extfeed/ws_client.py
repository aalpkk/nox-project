"""WebSocket client for the external intraday bar feed.

Public API:
    fetch_bars(symbol, timeframe, n_bars) -> pd.DataFrame

Wire protocol (frame-encoded, length-prefixed JSON):
    1. set_auth_token(<jwt>)
    2. chart_create_session(<session_id>, "")
    3. resolve_symbol(<session_id>, <symbol_alias>, '={"symbol":"...","adjustment":"splits"}')
    4. create_series(<session_id>, <series_id>, <sub_id>, <symbol_alias>, <tf>, <n>, "")
    Server replies with timescale_update frames (bars) and series_completed (done).
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
