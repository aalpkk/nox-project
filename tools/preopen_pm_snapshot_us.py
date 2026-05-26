"""Pre-market snapshot fetcher for US equities (TV WS, extended session).

Fetches 1-minute bars from TV WS with `session=extended` resolve payload,
filters to the pre-market window (default 04:00-09:30 ET) for today (US/ET),
and aggregates per-ticker into the schema consumed by preopen_validator_v0.

Output columns: ticker, snapshot_datetime_et, pm_last, pm_bid, pm_ask,
pm_volume, pm_vwap, pm_high, pm_low, pm_open, pm_n_bars.
pm_bid/pm_ask are always NaN (TV WS does not surface L1 quotes here);
validator will mark these rows with `missing_spread_data` WATCH, which is
expected and acceptable.

Universe is read from the T-close candidates file's `ticker` column so we
only pull what we need (typically 20-40 tickers).

Usage:
    python tools/preopen_pm_snapshot_us.py \\
        --candidates output/tclose_candidates_us_mb_1d_2026_05_22.csv \\
        --output     output/premarket_snapshot_us_2026_05_26.csv \\
        [--exchange-map output/us_exchange_map.csv] \\
        [--pm-start 04:00] [--pm-end 09:30] \\
        [--n-bars 500] [--inter-symbol-delay 0.3]

Exchange routing: if --exchange-map provided (CSV with ticker,exchange),
the script honors it (NASDAQ/NYSE/AMEX); otherwise it derives from
markets.us.data._fetch_us_exchange_map at runtime (cached NASDAQ Trader
listed-files). Symbol form is `{exchange}:{ticker}`.

Auth: INTRADAY_SID / INTRADAY_SIGN env vars (same TV cookies used elsewhere).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import secrets
import string
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, time as dtime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import websockets

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from markets.extfeed.auth import AuthCache, auth_from_env  # noqa: E402

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


async def _fetch_1m_extended_async(
    auth: AuthCache,
    symbol: str,
    n_bars: int,
    timeout_s: float,
) -> pd.DataFrame:
    """Fetch last n_bars 1-minute bars including extended hours via TV WS.

    Returns DataFrame with [time, open, high, low, close, volume],
    time tz-aware in America/New_York.
    """
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

        # resolve_symbol payload carries session=extended → TV returns
        # pre/post bars on supported symbols.
        resolve_payload = (
            '={"symbol":"' + symbol + '","adjustment":"splits","session":"extended"}'
        )

        outbound = [
            _encode_msg("set_auth_token", token),
            _encode_msg("chart_create_session", sess.chart_session, ""),
            _encode_msg(
                "resolve_symbol",
                sess.chart_session,
                sess.symbol_alias,
                resolve_payload,
            ),
            _encode_msg(
                "create_series",
                sess.chart_session,
                sess.series_id,
                sess.sub_id,
                sess.symbol_alias,
                "1",
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
        # Not necessarily fatal — symbol may have no recent bars.
        return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])

    df = pd.DataFrame(bars, columns=["time", "open", "high", "low", "close", "volume"])
    df["time"] = (
        pd.to_datetime(df["time"], unit="s", utc=True)
        .dt.tz_convert("America/New_York")
    )
    df = df.drop_duplicates(subset=["time"]).sort_values("time").reset_index(drop=True)
    return df


def fetch_1m_extended(
    symbol: str,
    n_bars: int,
    auth: Optional[AuthCache] = None,
    timeout_s: float = 30.0,
) -> pd.DataFrame:
    if auth is None:
        auth = auth_from_env()
    return asyncio.run(_fetch_1m_extended_async(auth, symbol, n_bars, timeout_s))


def _parse_hhmm(s: str) -> dtime:
    h, m = s.split(":")
    return dtime(int(h), int(m))


def _aggregate_premarket(
    df1m: pd.DataFrame,
    *,
    today_et: pd.Timestamp,
    pm_start: dtime,
    pm_end: dtime,
) -> dict:
    """Filter 1m bars to today's pre-market window and aggregate."""
    if df1m.empty:
        return {
            "pm_last": np.nan, "pm_volume": 0.0, "pm_vwap": np.nan,
            "pm_high": np.nan, "pm_low": np.nan, "pm_open": np.nan,
            "pm_n_bars": 0,
        }
    df = df1m.copy()
    df["date_et"] = df["time"].dt.date
    df["time_only"] = df["time"].dt.time
    target_date = today_et.date()
    mask = (
        (df["date_et"] == target_date)
        & (df["time_only"] >= pm_start)
        & (df["time_only"] < pm_end)
    )
    pm = df.loc[mask].copy()
    if pm.empty:
        return {
            "pm_last": np.nan, "pm_volume": 0.0, "pm_vwap": np.nan,
            "pm_high": np.nan, "pm_low": np.nan, "pm_open": np.nan,
            "pm_n_bars": 0,
        }
    vol = pm["volume"].fillna(0.0).astype(float)
    typ = ((pm["high"] + pm["low"] + pm["close"]) / 3.0).astype(float)
    vwap = float((typ * vol).sum() / vol.sum()) if vol.sum() > 0 else float("nan")
    return {
        "pm_last": float(pm["close"].iloc[-1]),
        "pm_volume": float(vol.sum()),
        "pm_vwap": vwap,
        "pm_high": float(pm["high"].max()),
        "pm_low": float(pm["low"].min()),
        "pm_open": float(pm["open"].iloc[0]),
        "pm_n_bars": int(len(pm)),
    }


def _resolve_exchange_map(
    tickers: list[str], explicit_map_path: Optional[str]
) -> dict[str, str]:
    if explicit_map_path:
        em = pd.read_csv(explicit_map_path)
        em.columns = [c.lower() for c in em.columns]
        if not {"ticker", "exchange"} <= set(em.columns):
            raise ValueError(
                f"exchange map file must have ticker,exchange columns: "
                f"got {list(em.columns)}"
            )
        return {str(r["ticker"]).strip(): str(r["exchange"]).strip().upper()
                for _, r in em.iterrows()}
    # Fall back to nasdaqtrader fetch.
    from markets.us.data import _fetch_us_exchange_map
    full = _fetch_us_exchange_map()
    return {t: full.get(t, "NASDAQ") for t in tickers}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True,
                    help="T-close candidates CSV/parquet (uses 'ticker' col).")
    ap.add_argument("--output", required=True,
                    help="Pre-market snapshot output CSV/parquet.")
    ap.add_argument("--exchange-map", default=None,
                    help="Optional CSV with ticker,exchange columns.")
    ap.add_argument("--pm-start", default="04:00",
                    help="Pre-market window start (HH:MM ET). Default 04:00.")
    ap.add_argument("--pm-end", default="09:30",
                    help="Pre-market window end (HH:MM ET, exclusive). "
                         "Default 09:30.")
    ap.add_argument("--n-bars", type=int, default=500,
                    help="1m bars to request per symbol. Default 500 "
                         "(covers 04:00→09:30 + buffer).")
    ap.add_argument("--inter-symbol-delay", type=float, default=0.3,
                    help="Sleep between symbols (s). Default 0.3.")
    ap.add_argument("--timeout-s", type=float, default=30.0,
                    help="Per-symbol WS timeout. Default 30.")
    ap.add_argument("--snapshot-date", default=None,
                    help="ISO date for snapshot day (default = today in ET).")
    args = ap.parse_args()

    cand_path = Path(args.candidates)
    if cand_path.suffix.lower() == ".parquet":
        cands = pd.read_parquet(cand_path)
    else:
        cands = pd.read_csv(cand_path)
    if "ticker" not in cands.columns:
        print(f"!! candidates file missing 'ticker' col: {cand_path}",
              file=sys.stderr)
        return 1
    tickers = sorted({str(t).strip() for t in cands["ticker"].dropna()})
    if not tickers:
        print("!! candidates file has no tickers.", file=sys.stderr)
        return 1
    print(f"[load] {len(tickers)} tickers from {cand_path.name}", flush=True)

    pm_start = _parse_hhmm(args.pm_start)
    pm_end = _parse_hhmm(args.pm_end)
    today_et = (
        pd.Timestamp(args.snapshot_date, tz="America/New_York")
        if args.snapshot_date
        else pd.Timestamp.now(tz="America/New_York")
    )
    print(f"[window] {today_et.date()} {pm_start.strftime('%H:%M')}→"
          f"{pm_end.strftime('%H:%M')} ET", flush=True)

    exch_map = _resolve_exchange_map(tickers, args.exchange_map)

    auth = auth_from_env()
    _ = auth.token()
    print(f"[auth] JWT ok, expires_in={auth.expires_at - int(time.time())}s",
          flush=True)

    snapshot_dt_et = pd.Timestamp.now(tz="America/New_York")
    rows: list[dict] = []
    fail: list[str] = []
    t0 = time.time()
    for i, tkr in enumerate(tickers, 1):
        exch = exch_map.get(tkr, "NASDAQ")
        symbol = f"{exch}:{tkr}"
        try:
            df1m = fetch_1m_extended(
                symbol=symbol, n_bars=args.n_bars,
                auth=auth, timeout_s=args.timeout_s,
            )
            agg = _aggregate_premarket(
                df1m, today_et=today_et,
                pm_start=pm_start, pm_end=pm_end,
            )
            rows.append({
                "ticker": tkr,
                "snapshot_datetime_et": snapshot_dt_et.isoformat(),
                "pm_last": agg["pm_last"],
                "pm_bid": np.nan,
                "pm_ask": np.nan,
                "pm_volume": agg["pm_volume"],
                "pm_vwap": agg["pm_vwap"],
                "pm_high": agg["pm_high"],
                "pm_low": agg["pm_low"],
                "pm_open": agg["pm_open"],
                "pm_n_bars": agg["pm_n_bars"],
            })
        except Exception as e:
            fail.append(f"{symbol}: {e}")
            rows.append({
                "ticker": tkr,
                "snapshot_datetime_et": snapshot_dt_et.isoformat(),
                "pm_last": np.nan, "pm_bid": np.nan, "pm_ask": np.nan,
                "pm_volume": 0.0, "pm_vwap": np.nan,
                "pm_high": np.nan, "pm_low": np.nan, "pm_open": np.nan,
                "pm_n_bars": 0,
            })
        if args.inter_symbol_delay > 0 and i < len(tickers):
            time.sleep(args.inter_symbol_delay)
        if i % 25 == 0 or i == len(tickers):
            elapsed = time.time() - t0
            eta = elapsed / i * (len(tickers) - i) if i < len(tickers) else 0.0
            print(f"  [{i}/{len(tickers)}] elapsed={elapsed:.0f}s "
                  f"eta={eta:.0f}s fails={len(fail)}", flush=True)

    out = pd.DataFrame(rows)
    n_with_data = int((out["pm_n_bars"] > 0).sum())
    print(f"[done] {n_with_data}/{len(out)} with pm bars "
          f"({len(fail)} hard fails)", flush=True)
    if fail:
        head = "; ".join(fail[:5])
        more = f" (+{len(fail) - 5} more)" if len(fail) > 5 else ""
        print(f"[fail] {head}{more}", flush=True)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() == ".parquet":
        out.to_parquet(out_path, index=False)
    else:
        out.to_csv(out_path, index=False)
    print(f"[write] {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
