"""Multi-symbol stress test for markets.extfeed.

Fetches a configurable timeframe + bar count for a list of BIST tickers
sequentially, measures throughput and per-ticker error rate.

Usage:
    python tools/extfeed_stress.py [TIMEFRAME] [N_BARS] [N_TICKERS] [DELAY_S]
Defaults: 15  200  30  0.3
"""
from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from markets.extfeed import auth_from_env, fetch_bars

# Liquid BIST blue-chips + mid-caps for stress sample
DEFAULT_UNIVERSE = [
    "AKBNK", "GARAN", "ISCTR", "VAKBN", "YKBNK", "HALKB", "SKBNK", "ALBRK",
    "THYAO", "PGSUS", "TUPRS", "PETKM", "EREGL", "KRDMD", "KCHOL", "SAHOL",
    "ARCLK", "ASELS", "BIMAS", "MGROS", "FROTO", "TOASO", "TCELL", "TTKOM",
    "SISE", "AEFES", "CCOLA", "ULKER", "SASA", "AKSA",
]


def main() -> int:
    timeframe = sys.argv[1] if len(sys.argv) > 1 else "15"
    n_bars = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    n_tickers = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    delay_s = float(sys.argv[4]) if len(sys.argv) > 4 else 0.3

    universe = DEFAULT_UNIVERSE[:n_tickers]
    print(f"  timeframe : {timeframe}")
    print(f"  n_bars    : {n_bars}")
    print(f"  n_tickers : {len(universe)}")
    print(f"  delay     : {delay_s}s between fetches")

    auth = auth_from_env()
    _ = auth.token()
    print(f"  ✓ JWT cached")

    results = []
    t0 = time.time()
    fail_count = 0

    for i, code in enumerate(universe, 1):
        symbol = f"BIST:{code}"
        t_req = time.time()
        try:
            df = fetch_bars(symbol, timeframe, n_bars, auth=auth, timeout_s=20)
            dt = time.time() - t_req
            results.append({
                "symbol": code,
                "bars": len(df),
                "first": df["time"].min(),
                "last": df["time"].max(),
                "latency_s": round(dt, 2),
                "status": "ok",
                "error": "",
            })
            print(f"  [{i:2d}/{len(universe)}] {code:6s}  {len(df):3d} bars  "
                  f"{dt:5.2f}s  last={df['time'].max()}")
        except Exception as e:
            dt = time.time() - t_req
            fail_count += 1
            results.append({
                "symbol": code,
                "bars": 0,
                "first": None,
                "last": None,
                "latency_s": round(dt, 2),
                "status": "fail",
                "error": str(e)[:200],
            })
            print(f"  [{i:2d}/{len(universe)}] {code:6s}  FAIL  {dt:.2f}s  {e}")
        time.sleep(delay_s)

    elapsed = time.time() - t0
    df = pd.DataFrame(results)

    print()
    print("=" * 60)
    print(f"  total elapsed : {elapsed:.1f}s")
    print(f"  ok            : {(df['status']=='ok').sum()}/{len(df)}")
    print(f"  fail          : {fail_count}")
    if (df["status"] == "ok").any():
        ok = df[df["status"] == "ok"]
        print(f"  avg latency   : {ok['latency_s'].mean():.2f}s")
        print(f"  p95 latency   : {ok['latency_s'].quantile(0.95):.2f}s")
        print(f"  total bars    : {ok['bars'].sum():,}")
        print(f"  throughput    : {ok['bars'].sum() / elapsed:.0f} bars/s")
    if fail_count:
        print()
        print("  failures:")
        for _, row in df[df["status"] == "fail"].iterrows():
            print(f"    {row['symbol']}: {row['error']}")

    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
