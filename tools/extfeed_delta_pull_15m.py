"""Delta-pull recent 15m bars from extfeed and merge into the 15m master parquet.

Each ticker's fetch_bars_until opens its own independent WS session (asyncio.run +
random chart_session), so we fan out across a small thread pool + retry the fails.

NOTE on speed: the extfeed server THROTTLES total throughput to ~1.7 successful
pulls/sec — measured: 5 workers = 607 tickers in ~350s, 0 fails; 12 workers = 186s
but 43% FAIL (server rejects excess concurrent connections). So fan-out beyond ~5
buys failures, not speed; a full 607-ticker reliable pull is ~6min floor (server-
bound, not our code). Real speedups would need WS symbol-multiplexing or pulling
only the day's +3-8% candidates (needs a cheap price snapshot first). Default
workers=5 (0 fails) + retry passes at half concurrency to recover transient fails.

Refills the last --days (default 7; the workflow caches the master so 7d covers the
daily gap). Schema: [time, open, high, low, close, volume, ticker]; dedup (ticker, time).

Usage: python tools/extfeed_delta_pull_15m.py [--days 7] [--workers 5] [--retries 3]
"""
from __future__ import annotations
import argparse, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    from dotenv import load_dotenv
    load_dotenv()                      # local .env; harmless in CI where env is set
except Exception:
    pass
from markets.extfeed import auth_from_env, fetch_bars_until  # noqa: E402

OUT = Path("output/extfeed_intraday_15m_master.parquet")


def _pull_one(code, until_date, auth):
    """Fetch one ticker. Returns (code, DataFrame) or (code, None) on empty/fail.
    Each call opens its own WS session, so this is safe to run concurrently."""
    try:
        df, _st = fetch_bars_until(symbol=f"BIST:{code}", timeframe="15", until_date=until_date,
                                   chunk_n=500, max_chunks=3, chunk_timeout_s=20.0,
                                   inter_chunk_delay_s=0.2, auth=auth)
        if df is None or df.empty:
            return code, None
        df["ticker"] = code
        return code, df[["time", "open", "high", "low", "close", "volume", "ticker"]]
    except Exception:
        return code, None


def _pass(tickers, until_date, auth, workers):
    """One parallel pass. Returns (list_of_dfs, list_of_failed_codes)."""
    rows, fails = [], []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_pull_one, c, until_date, auth): c for c in tickers}
        for fut in as_completed(futs):
            code, res = fut.result()
            (rows if res is not None else fails).append(res if res is not None else code)
    return rows, fails


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--retries", type=int, default=3)
    args = ap.parse_args()
    if not OUT.exists():
        print(f"!! 15m master missing: {OUT} (run tavan_pull_15m_bulk_v0.py first)", flush=True)
        return 1
    existing = pd.read_parquet(OUT)
    print(f"loaded 15m master: {existing['ticker'].nunique()} tickers, {len(existing):,} rows, "
          f"max time={pd.to_datetime(existing['time']).max()}", flush=True)
    universe = sorted(existing["ticker"].dropna().unique().tolist())
    until_date = pd.Timestamp(datetime.utcnow() - timedelta(days=args.days)).tz_localize("UTC")
    print(f"delta until_date: {until_date} (last {args.days}d), {len(universe)} tickers, "
          f"{args.workers} workers", flush=True)
    auth = auth_from_env(); _ = auth.token()   # warm JWT once so worker threads read cached token

    # First pass at full concurrency, then retry only the fails at HALF concurrency
    # (server throttles concurrent WS — retries at lower fan-out recover most).
    new, t0 = [], time.time()
    remaining = universe
    for rnd in range(args.retries + 1):
        w = args.workers if rnd == 0 else max(2, args.workers // 2)
        rows, fails = _pass(remaining, until_date, auth, w)
        new.extend(rows)
        print(f"  round {rnd}: workers={w} ok={len(rows)} fail={len(fails)} "
              f"(elapsed={time.time()-t0:.0f}s)", flush=True)
        remaining = fails
        if not remaining:
            break
        time.sleep(2)
    if remaining:
        print(f"  WARN: {len(remaining)} tickers still failed after retries: {remaining[:10]}...", flush=True)
    if not new:
        print("!! no delta rows pulled", flush=True); return 2
    delta = pd.concat(new, ignore_index=True)
    combined = pd.concat([existing, delta], ignore_index=True)
    before = len(combined)
    combined = combined.drop_duplicates(subset=["ticker", "time"], keep="last")
    print(f"merge: {before:,} → {len(combined):,} rows; new max time={pd.to_datetime(combined['time']).max()}", flush=True)
    combined.to_parquet(OUT, index=False)
    print(f"wrote {OUT} in {time.time()-t0:.0f}s total", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
