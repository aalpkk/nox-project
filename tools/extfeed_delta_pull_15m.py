"""Delta-pull recent 15m bars from extfeed and merge into the 15m master parquet.

Mirror of extfeed_delta_pull.py but timeframe=15. Refills the last LOOKBACK_DAYS
(default 35, > the 20-day trailing feature window) so the scan's rolling features
are intact even if the committed master seed is stale. Schema matches the bulk
puller: [time, open, high, low, close, volume, ticker]; dedup on (ticker, time).

Usage: python tools/extfeed_delta_pull_15m.py [--days 35]
"""
from __future__ import annotations
import argparse, sys, time
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from markets.extfeed import auth_from_env, fetch_bars_until  # noqa: E402

OUT = Path("output/extfeed_intraday_15m_master.parquet")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=35)
    args = ap.parse_args()
    if not OUT.exists():
        print(f"!! 15m master missing: {OUT} (run tavan_pull_15m_bulk_v0.py first)", flush=True)
        return 1
    existing = pd.read_parquet(OUT)
    print(f"loaded 15m master: {existing['ticker'].nunique()} tickers, {len(existing):,} rows, "
          f"max time={pd.to_datetime(existing['time']).max()}", flush=True)
    universe = sorted(existing["ticker"].dropna().unique().tolist())
    until_date = pd.Timestamp(datetime.utcnow() - timedelta(days=args.days)).tz_localize("UTC")
    print(f"delta until_date: {until_date} (last {args.days}d), {len(universe)} tickers", flush=True)
    auth = auth_from_env(); _ = auth.token()

    new, fail, t0 = [], 0, time.time()
    for i, code in enumerate(universe, 1):
        try:
            df, _st = fetch_bars_until(symbol=f"BIST:{code}", timeframe="15", until_date=until_date,
                                       chunk_n=500, max_chunks=3, chunk_timeout_s=20.0,
                                       inter_chunk_delay_s=0.2, auth=auth)
            if df is None or df.empty:
                fail += 1; continue
            df["ticker"] = code
            new.append(df[["time", "open", "high", "low", "close", "volume", "ticker"]])
        except Exception as e:
            fail += 1
            if i <= 5 or i % 100 == 0:
                print(f"  fail {code}: {e}", flush=True)
        if i % 100 == 0:
            el = time.time() - t0
            print(f"  [{i}/{len(universe)}] elapsed={el:.0f}s eta={el/i*(len(universe)-i):.0f}s fails={fail}", flush=True)
    if not new:
        print("!! no delta rows pulled", flush=True); return 2
    delta = pd.concat(new, ignore_index=True)
    combined = pd.concat([existing, delta], ignore_index=True)
    before = len(combined)
    combined = combined.drop_duplicates(subset=["ticker", "time"], keep="last")
    print(f"merge: {before:,} → {len(combined):,} rows; new max time={pd.to_datetime(combined['time']).max()}", flush=True)
    combined.to_parquet(OUT, index=False)
    print(f"wrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
