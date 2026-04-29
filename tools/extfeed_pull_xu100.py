"""Pull XU100 daily OHLC via extfeed (TV WebSocket) and save parquet.

Used as a clean replacement for the yfinance-sourced output/xu100_cache.parquet
when high-fidelity OHLC is required (e.g. RDP regime labeling).

Symbol: BIST:XU100   Timeframe: D   Range: 2023-01-01 -> latest

Output (separate file, never overwrites yfinance cache):
    output/xu100_extfeed_daily.parquet
        index = Date (Europe/Istanbul calendar day, tz-naive)
        columns = open, high, low, close, volume

Run locally if INTRADAY_SID/SIGN/HOST/WS_URL are set, otherwise dispatch via
the `xu100_pull` mode in .github/workflows/extfeed-smoke.yml.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from markets.extfeed import auth_from_env, fetch_bars_until

SYMBOL    = "BIST:XU100"
TIMEFRAME = "D"
START     = "2023-01-01"
OUT_PATH  = Path("output/xu100_extfeed_daily.parquet")


def main() -> int:
    print(f"[xu100] symbol={SYMBOL}  tf={TIMEFRAME}  start={START}")

    print("\n[1/3] auth")
    auth = auth_from_env()
    token = auth.token()
    print(f"  ok  jwt_len={len(token)}  expires_in={auth.expires_at - int(time.time())}s")

    print("\n[2/3] fetch")
    t0 = time.time()
    df, stats = fetch_bars_until(SYMBOL, TIMEFRAME, START, auth=auth)
    dt = time.time() - t0
    print(f"  ok  bars={len(df)}  in {dt:.2f}s  stats={stats}")
    if df.empty:
        print("  [!] empty DataFrame — abort")
        return 2

    # Normalize: convert UTC timestamps to Europe/Istanbul calendar dates,
    # drop any rows before START (extfeed may return slightly older), set Date index.
    if "time" not in df.columns:
        print(f"  [!] expected 'time' column, got {df.columns.tolist()}")
        return 2

    ts = pd.to_datetime(df["time"], utc=True).dt.tz_convert("Europe/Istanbul")
    out = pd.DataFrame({
        "open":   df["open"].astype(float),
        "high":   df["high"].astype(float),
        "low":    df["low"].astype(float),
        "close":  df["close"].astype(float),
        "volume": df.get("volume", pd.Series(0, index=df.index)).astype(float),
    })
    out.index = ts.dt.normalize().dt.tz_localize(None)
    out.index.name = "Date"
    out = out[~out.index.duplicated(keep="last")].sort_index()
    out = out[out.index >= pd.Timestamp(START)]

    print(f"\n[3/3] write")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PATH)
    print(f"  ok  {OUT_PATH}  rows={len(out)}  range={out.index.min().date()} -> {out.index.max().date()}")
    print()
    print(out.tail(5).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
