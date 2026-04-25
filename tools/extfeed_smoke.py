"""Smoke test for markets.extfeed — fetch bars for a symbol and print summary.

Usage:
    python tools/extfeed_smoke.py [SYMBOL] [TIMEFRAME] [N_BARS]

Defaults: BIST:THYAO 15 50
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from markets.extfeed import auth_from_env, fetch_bars


def main() -> int:
    symbol = sys.argv[1] if len(sys.argv) > 1 else "BIST:THYAO"
    timeframe = sys.argv[2] if len(sys.argv) > 2 else "15"
    n_bars = int(sys.argv[3]) if len(sys.argv) > 3 else 50

    print(f"  symbol     : {symbol}")
    print(f"  timeframe  : {timeframe}")
    print(f"  n_bars     : {n_bars}")

    print("\n[1/2] auth — exchange cookies for JWT")
    auth = auth_from_env()
    token = auth.token()
    remain = auth.expires_at - int(time.time())
    print(f"  ✓ JWT acquired  len={len(token)}  expires_in={remain}s")

    print("\n[2/2] WS — fetch bars")
    t0 = time.time()
    df = fetch_bars(symbol, timeframe, n_bars, auth=auth)
    dt = time.time() - t0
    print(f"  ✓ {len(df)} bars in {dt:.2f}s")
    print(f"  range : {df['time'].min()}  →  {df['time'].max()}")
    print()
    print(df.tail(10).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
