"""Smoke probe — does request_more_data actually paginate backwards?

Single ticker, two scenarios:

  1) Baseline: fetch_bars latest 200 bars → record min(time)
  2) Paginated: fetch_bars_until pulled back ~10 chunks of 200 bars each
     with target until_date 5 years ago → check that min(time) << baseline

If pagination works, paginated min(time) is materially older than baseline.
If TV server rejects request_more_data, we'd see protocol_error or no progress.

Usage:
    python tools/extfeed_paginate_probe.py [SYMBOL] [TIMEFRAME] [CHUNK_N]
Defaults: BIST:THYAO 60 200
"""
from __future__ import annotations

import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from markets.extfeed import auth_from_env, fetch_bars, fetch_bars_until


def main() -> int:
    symbol = sys.argv[1] if len(sys.argv) > 1 else "BIST:THYAO"
    timeframe = sys.argv[2] if len(sys.argv) > 2 else "60"
    chunk_n = int(sys.argv[3]) if len(sys.argv) > 3 else 200

    print(f"  symbol     : {symbol}")
    print(f"  timeframe  : {timeframe}")
    print(f"  chunk_n    : {chunk_n}")
    print()

    auth = auth_from_env()
    _ = auth.token()
    print("  ✓ JWT cached")

    print("\n[1/2] baseline — fetch_bars(latest 200)")
    t0 = time.time()
    df_a = fetch_bars(symbol, timeframe, 200, auth=auth)
    dt_a = time.time() - t0
    print(f"  bars : {len(df_a)}")
    print(f"  range: {df_a['time'].min()}  →  {df_a['time'].max()}")
    print(f"  time : {dt_a:.2f}s")

    print("\n[2/2] paginated — fetch_bars_until(3y ago)")
    target = pd.Timestamp(datetime.utcnow() - timedelta(days=365 * 3)).tz_localize("UTC")
    print(f"  target_until : {target}")
    t0 = time.time()
    df_b, stats = fetch_bars_until(
        symbol=symbol,
        timeframe=timeframe,
        until_date=target,
        chunk_n=chunk_n,
        max_chunks=20,
        chunk_timeout_s=30.0,
        inter_chunk_delay_s=0.3,
        auth=auth,
    )
    dt_b = time.time() - t0
    print(f"  bars : {len(df_b)}")
    print(f"  range: {df_b['time'].min()}  →  {df_b['time'].max()}")
    print(f"  time : {dt_b:.2f}s  ({len(df_b) / max(dt_b, 1e-6):.0f} bars/s)")
    print(f"  stats: {stats}")

    print()
    print("=" * 60)
    span_a = df_a["time"].max() - df_a["time"].min()
    span_b = df_b["time"].max() - df_b["time"].min()
    print(f"  baseline span    : {span_a}  ({len(df_a)} bars)")
    print(f"  paginated span   : {span_b}  ({len(df_b)} bars)")
    if span_b > span_a * 2:
        print("  ✓ pagination WORKS — paginated reaches further back")
        return 0
    else:
        print("  ✗ pagination DID NOT work — paginated span ≈ baseline span")
        return 2


if __name__ == "__main__":
    sys.exit(main())
