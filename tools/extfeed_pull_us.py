"""Pull NASDAQ 1h bars from TV WS feed → US master parquet.

Mirrors tools/extfeed_delta_pull.py for US equities. Two modes:

  bootstrap   First-time pull. Master parquet missing. Universe seeded from
              markets.us.data._NDX100_STATIC (100 tickers). Pulls ~3y of 1h
              bars per ticker.
  delta       Master parquet exists. Universe read from existing tickers.
              Pulls last 4 days, deduplicates, merges.

Auto-detect: if --mode omitted, choose bootstrap when master missing else delta.

Output: output/extfeed_intraday_1h_3y_master_us.parquet
Schema: ticker, ts_utc, ts_newyork, open, high, low, close, volume

Symbols sent to TV are formed as "NASDAQ:{code}". NDX100 is fully NASDAQ-listed
so no NYSE prefix handling needed in POC. For mixed universes (S&P 500 etc.)
extend the symbol_for() mapping.

Auth: INTRADAY_SID / INTRADAY_SIGN env vars (same TV session cookies used by
the BIST extfeed pull).
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from markets.extfeed import auth_from_env, fetch_bars_until
from markets.us.data import _NDX100_STATIC

OUT_BARS = REPO / "output" / "extfeed_intraday_1h_3y_master_us.parquet"

# ~252 trading days/yr * 6.5h/day * 3y ≈ 4900 1h bars per ticker.
# 5000 chunk + 2 chunks ceiling gives generous margin.
BOOTSTRAP_CHUNK_N = 5000
BOOTSTRAP_MAX_CHUNKS = 2
BOOTSTRAP_YEARS_BACK = 3

DELTA_CHUNK_N = 200
DELTA_MAX_CHUNKS = 2
DELTA_DAYS_BACK = 4


def symbol_for(code: str) -> str:
    """Map ticker code → TV symbol. NDX100 universe is all NASDAQ-listed."""
    return f"NASDAQ:{code}"


def _normalize_bars(df: pd.DataFrame, code: str) -> pd.DataFrame:
    """ws_client returns df with `time` tz=Europe/Istanbul. Rewrite to US schema."""
    df = df.copy()
    df["ts_utc"] = df["time"].dt.tz_convert("UTC")
    df["ts_newyork"] = df["time"].dt.tz_convert("America/New_York")
    df["ticker"] = code
    return df[["ticker", "ts_utc", "ts_newyork", "open", "high", "low", "close", "volume"]]


def pull_universe(
    *,
    universe: list[str],
    until_date: pd.Timestamp,
    chunk_n: int,
    max_chunks: int,
) -> tuple[pd.DataFrame, int, int]:
    """Loop universe, pull bars until until_date, return concat + counts."""
    auth = auth_from_env()
    _ = auth.token()
    print(f"[auth] JWT acquired, expires_in={auth.expires_at - int(time.time())}s", flush=True)

    new_chunks: list[pd.DataFrame] = []
    fail = 0
    ok = 0
    t_start = time.time()
    for i, code in enumerate(universe, 1):
        symbol = symbol_for(code)
        try:
            df, stats = fetch_bars_until(
                symbol=symbol,
                timeframe="60",
                until_date=until_date,
                chunk_n=chunk_n,
                max_chunks=max_chunks,
                chunk_timeout_s=30.0,
                inter_chunk_delay_s=0.3,
                auth=auth,
            )
            if df is None or df.empty:
                fail += 1
                continue
            new_chunks.append(_normalize_bars(df, code))
            ok += 1
        except Exception as e:
            fail += 1
            if i <= 5 or i % 25 == 0:
                print(f"  fail {symbol}: {e}", flush=True)
        if i % 25 == 0:
            elapsed = time.time() - t_start
            eta = elapsed / i * (len(universe) - i)
            print(f"  [{i}/{len(universe)}] elapsed={elapsed:.0f}s eta={eta:.0f}s "
                  f"ok={ok} fails={fail}", flush=True)

    if not new_chunks:
        return pd.DataFrame(columns=["ticker", "ts_utc", "ts_newyork",
                                     "open", "high", "low", "close", "volume"]), ok, fail
    return pd.concat(new_chunks, ignore_index=True), ok, fail


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["bootstrap", "delta", "auto"], default="auto")
    ap.add_argument("--universe", nargs="*", default=None,
                    help="Override ticker list (otherwise NDX100 for bootstrap, "
                         "existing master for delta).")
    args = ap.parse_args()

    mode = args.mode
    if mode == "auto":
        mode = "delta" if OUT_BARS.exists() else "bootstrap"
    print(f"[mode] {mode}", flush=True)

    if mode == "bootstrap":
        universe = args.universe or sorted(set(_NDX100_STATIC))
        until_date = pd.Timestamp(
            datetime.utcnow() - timedelta(days=BOOTSTRAP_YEARS_BACK * 366)
        ).tz_localize("UTC")
        print(f"[bootstrap] universe={len(universe)} tickers, until={until_date.date()} "
              f"(~{BOOTSTRAP_YEARS_BACK}y back)", flush=True)
        existing = pd.DataFrame(columns=["ticker", "ts_utc", "ts_newyork",
                                         "open", "high", "low", "close", "volume"])
        delta, ok, fail = pull_universe(
            universe=universe,
            until_date=until_date,
            chunk_n=BOOTSTRAP_CHUNK_N,
            max_chunks=BOOTSTRAP_MAX_CHUNKS,
        )
    else:
        if not OUT_BARS.exists():
            print(f"!! master parquet missing for delta mode: {OUT_BARS}\n"
                  f"   run with --mode bootstrap first.", flush=True)
            return 1
        existing = pd.read_parquet(OUT_BARS)
        print(f"[delta] loaded master: {existing['ticker'].nunique()} tickers, "
              f"{len(existing):,} rows, max ts_utc={existing['ts_utc'].max()}", flush=True)
        universe = args.universe or sorted(existing["ticker"].dropna().unique().tolist())
        until_date = pd.Timestamp(
            datetime.utcnow() - timedelta(days=DELTA_DAYS_BACK)
        ).tz_localize("UTC")
        print(f"[delta] universe={len(universe)} tickers, until={until_date} "
              f"(~{DELTA_DAYS_BACK} days back)", flush=True)
        delta, ok, fail = pull_universe(
            universe=universe,
            until_date=until_date,
            chunk_n=DELTA_CHUNK_N,
            max_chunks=DELTA_MAX_CHUNKS,
        )

    if delta.empty:
        print("!! no rows pulled", flush=True)
        return 2

    print(f"[pull] {len(delta):,} rows, "
          f"date range {delta['ts_utc'].min()} → {delta['ts_utc'].max()} "
          f"({ok} ok / {fail} fail)", flush=True)

    combined = pd.concat([existing, delta], ignore_index=True)
    before = len(combined)
    combined = combined.drop_duplicates(subset=["ticker", "ts_utc"], keep="last")
    after = len(combined)
    combined = combined.sort_values(["ticker", "ts_utc"]).reset_index(drop=True)
    print(f"[merge] {before:,} → {after:,} rows ({before - after:,} dups dropped)", flush=True)
    print(f"[master] tickers={combined['ticker'].nunique()} "
          f"rows={len(combined):,} max ts_utc={combined['ts_utc'].max()}", flush=True)

    OUT_BARS.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(OUT_BARS, index=False)
    print(f"[write] {OUT_BARS}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
