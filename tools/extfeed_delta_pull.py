"""Delta-pull recent 1h bars from extfeed and merge into master parquet.

For each ticker in the existing master, pulls the last few days worth of
bars (small chunk_n) and de-duplicates against existing rows.
"""
from __future__ import annotations

import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from markets.extfeed import auth_from_env, fetch_bars_until

OUT_BARS = Path("output/extfeed_intraday_1h_3y_master.parquet")


def main() -> int:
    if not OUT_BARS.exists():
        print(f"!! master parquet missing: {OUT_BARS}", flush=True)
        return 1

    existing = pd.read_parquet(OUT_BARS)
    print(f"loaded master: {existing['ticker'].nunique()} tickers, "
          f"{len(existing):,} rows, "
          f"max ts={existing['ts_istanbul'].max()}", flush=True)

    universe = sorted(existing["ticker"].dropna().unique().tolist())
    print(f"universe: {len(universe)} tickers", flush=True)

    until_date = pd.Timestamp(
        datetime.utcnow() - timedelta(days=4)
    ).tz_localize("UTC")
    print(f"delta until_date: {until_date} (≈ last 4 days)", flush=True)

    auth = auth_from_env()
    _ = auth.token()
    print(f"JWT acquired, expires_in={auth.expires_at - int(time.time())}s", flush=True)

    new_chunks: list[pd.DataFrame] = []
    fail = 0
    t_start = time.time()
    for i, code in enumerate(universe, 1):
        symbol = f"BIST:{code}"
        try:
            df, stats = fetch_bars_until(
                symbol=symbol, timeframe="60", until_date=until_date,
                chunk_n=200, max_chunks=2, chunk_timeout_s=15.0,
                inter_chunk_delay_s=0.2, auth=auth,
            )
            if df is None or df.empty:
                fail += 1
                continue
            df["ticker"] = code
            df["ts_utc"] = df["time"].dt.tz_convert("UTC")
            df["ts_istanbul"] = df["time"]
            df = df[["ticker", "ts_utc", "ts_istanbul",
                     "open", "high", "low", "close", "volume"]]
            new_chunks.append(df)
        except Exception as e:
            fail += 1
            if i <= 5 or i % 100 == 0:
                print(f"  fail {code}: {e}", flush=True)
        if i % 50 == 0:
            elapsed = time.time() - t_start
            eta = elapsed / i * (len(universe) - i)
            print(f"  [{i}/{len(universe)}] elapsed={elapsed:.0f}s eta={eta:.0f}s "
                  f"fails={fail}", flush=True)

    if not new_chunks:
        print("!! no delta rows pulled", flush=True)
        return 2

    delta = pd.concat(new_chunks, ignore_index=True)
    print(f"pulled delta: {len(delta):,} rows, "
          f"date range {delta['ts_istanbul'].min()} → {delta['ts_istanbul'].max()}", flush=True)

    combined = pd.concat([existing, delta], ignore_index=True)
    before = len(combined)
    combined = combined.drop_duplicates(subset=["ticker", "ts_utc"], keep="last")
    after = len(combined)
    print(f"merge: {before:,} → {after:,} rows ({before - after:,} dups dropped)", flush=True)
    print(f"new master: max ts={combined['ts_istanbul'].max()}", flush=True)

    combined.to_parquet(OUT_BARS, index=False)
    print(f"wrote {OUT_BARS}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
