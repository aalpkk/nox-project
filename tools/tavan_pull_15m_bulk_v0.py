"""Bulk-pull reachable 15m history (fetch_bars_until checkpoint-paging reaches
back to ~2022, ~4y) for the full BIST universe. Writes a 15m master parquet,
incrementally checkpointed every CHECKPOINT tickers so a crash keeps progress.
RESUMABLE: on restart, skips tickers already present in the output parquet.

Usage: python tools/tavan_pull_15m_bulk_v0.py [--until 2022-01-01] [--limit N]
Output: output/extfeed_intraday_15m_master.parquet
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
load_dotenv()
from markets.extfeed import auth_from_env                       # noqa: E402
from markets.extfeed.ws_client import fetch_bars_until          # noqa: E402

COVERAGE = REPO / "output/extfeed_intraday_coverage.csv"
OUT = REPO / "output/extfeed_intraday_15m_master.parquet"
CHECKPOINT = 25


def universe(limit=None):
    cov = pd.read_csv(COVERAGE)
    tk = sorted(cov["ticker"].dropna().unique().tolist())
    return tk[:limit] if limit else tk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--until", default="2022-01-01")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    tickers = universe(args.limit)
    auth = auth_from_env()
    auth.token()  # warm JWT

    # RESUME: load existing master, skip tickers already pulled
    frames, done, fail = [], 0, 0
    have = set()
    if OUT.exists():
        prev = pd.read_parquet(OUT)
        have = set(prev["ticker"].unique())
        frames.append(prev)
        tickers = [t for t in tickers if t not in have]
        print(f"RESUME: {len(have)} tickers already in master, "
              f"{len(tickers)} remaining", flush=True)
    print(f"15m bulk pull: {len(tickers)} tickers, until={args.until}", flush=True)

    t0 = time.time()
    for i, tk in enumerate(tickers, 1):
        try:
            df, st = fetch_bars_until(f"BIST:{tk}", "15", args.until,
                                      chunk_n=5000, max_chunks=8,
                                      chunk_timeout_s=45, inter_chunk_delay_s=0.5,
                                      auth=auth)
            df["ticker"] = tk
            frames.append(df)
            done += 1
        except Exception as e:
            fail += 1
            print(f"  [{i}/{len(tickers)}] {tk}: ERR {type(e).__name__}: {str(e)[:80]}", flush=True)
        if i % CHECKPOINT == 0 or i == len(tickers):
            if frames:
                pd.concat(frames, ignore_index=True).to_parquet(OUT, index=False)
            el = (time.time() - t0) / 60
            print(f"  [{i}/{len(tickers)}] ok={done} fail={fail} "
                  f"rows={sum(len(f) for f in frames):,} elapsed={el:.1f}m "
                  f"-> checkpoint", flush=True)
        time.sleep(0.2)

    if frames:
        out = pd.concat(frames, ignore_index=True)
        out.to_parquet(OUT, index=False)
        t = pd.to_datetime(out["time"])
        print(f"\nDONE: {len(out):,} rows, {out.ticker.nunique()} tickers, "
              f"{t.min()} -> {t.max()} -> {OUT.name}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
