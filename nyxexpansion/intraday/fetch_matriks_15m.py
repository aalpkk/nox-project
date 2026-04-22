"""
Fetch 15-minute intraday bars from Matriks for v4C top-D signal days.

Matriks historicalData tool constraints (probed 2026-04-22):
  - interval enum: 1min | 5min | 15min | 1hour | daily | weekly | monthly
  - Response bar list under 'allBars' key
  - Per-request cap ≈ 1 trading day for 15min (~32 bars)
  - Deep history works when startDate==endDate==target date (tested back to 2021)

Pipeline:
  1. Read v4C top-D signals (ticker, date) — 339 rows
  2. For each (ticker, date): one historicalData call with interval=15min,
     startDate=endDate=signal_date
  3. Cache: output/nyxexp_intraday_15m_matriks.parquet
     - Resume across runs, skip already-fetched
  4. Canary mode: --canary 5 → only first 5 signals (429 sanity check)

Env:
  MATRIKS_API_KEY (required)
  MATRIKS_CLIENT_ID (optional, default 33667)

Output parquet columns:
  ticker, signal_date, bar_ts, date, open, high, low, close, volume, quantity
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent.matriks_client import MatriksClient  # noqa: E402


SIGNALS_PATH = Path("output/nyxexp_backtest_v4C.parquet")
CACHE_PATH = Path("output/nyxexp_intraday_15m_matriks.parquet")


def _load_signals() -> pd.DataFrame:
    if not SIGNALS_PATH.exists():
        raise FileNotFoundError(f"Signals missing: {SIGNALS_PATH}")
    df = pd.read_parquet(SIGNALS_PATH)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    # (ticker, date) unique çiftlere indirge
    keys = df[["ticker", "date"]].drop_duplicates().reset_index(drop=True)
    keys = keys.sort_values(["date", "ticker"]).reset_index(drop=True)
    return keys


def _load_cache() -> pd.DataFrame:
    if CACHE_PATH.exists():
        return pd.read_parquet(CACHE_PATH)
    return pd.DataFrame(columns=[
        "ticker", "signal_date", "bar_ts", "date",
        "open", "high", "low", "close", "volume", "quantity",
    ])


def _cached_keys(cache_df: pd.DataFrame) -> set:
    if cache_df.empty:
        return set()
    return set(
        (r.ticker, r.signal_date) for r in cache_df[["ticker", "signal_date"]]
        .drop_duplicates().itertuples(index=False)
    )


def _fetch_one(client: MatriksClient, ticker: str, date) -> list[dict]:
    """Single day 15min bars via Matriks historicalData."""
    s = date.isoformat()
    resp = client.call_tool("historicalData", {
        "symbol": ticker,
        "startDate": s,
        "endDate": s,
        "interval": "15min",
        "rawBars": True,
    })
    if not isinstance(resp, dict):
        return []
    bars = resp.get("allBars") or []
    out = []
    for b in bars:
        if not isinstance(b, dict):
            continue
        out.append({
            "ticker": ticker,
            "signal_date": date,
            "bar_ts": b.get("timestamp"),
            "date": b.get("date"),
            "open": b.get("open"),
            "high": b.get("high"),
            "low": b.get("low"),
            "close": b.get("close"),
            "volume": b.get("volume"),
            "quantity": b.get("quantity"),
        })
    return out


def _persist(cache_df: pd.DataFrame, new_rows: list[dict]) -> pd.DataFrame:
    if not new_rows:
        return cache_df
    new_df = pd.DataFrame(new_rows)
    combined = pd.concat([cache_df, new_df], ignore_index=True)
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(CACHE_PATH, index=False)
    return combined


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--canary", type=int, default=0,
                    help="Only fetch first N signals (sanity/429 check)")
    ap.add_argument("--flush-every", type=int, default=20,
                    help="Write cache to disk every N successful fetches")
    args = ap.parse_args()

    if not os.environ.get("MATRIKS_API_KEY"):
        print("❌ MATRIKS_API_KEY env değişkeni set değil")
        return 1

    keys = _load_signals()
    cache = _load_cache()
    done = _cached_keys(cache)

    todo = [
        (r.ticker, r.date) for r in keys.itertuples(index=False)
        if (r.ticker, r.date) not in done
    ]
    if args.canary > 0:
        todo = todo[: args.canary]

    print(f"Total signals: {len(keys)}")
    print(f"Already cached: {len(done)}")
    print(f"To fetch: {len(todo)}")
    if args.canary > 0:
        print(f"⚠ CANARY mode: limited to {args.canary}")
    print(f"Cache: {CACHE_PATH}")
    print()

    client = MatriksClient()
    fetched = 0
    failed = []
    pending_rows: list[dict] = []
    t_start = time.time()

    for i, (ticker, date) in enumerate(todo, 1):
        t0 = time.time()
        try:
            rows = _fetch_one(client, ticker, date)
        except Exception as e:
            failed.append((ticker, date, type(e).__name__, str(e)[:120]))
            print(f"  [{i:3d}/{len(todo)}] {ticker} {date}  ❌ {type(e).__name__}: {str(e)[:100]}")
            continue
        dt = time.time() - t0
        if not rows:
            failed.append((ticker, date, "empty", "no bars returned"))
            print(f"  [{i:3d}/{len(todo)}] {ticker} {date}  ⚠ empty ({dt:.1f}s)")
            continue
        pending_rows.extend(rows)
        fetched += 1
        if i % 10 == 0 or i == len(todo):
            elapsed = time.time() - t_start
            rate = i / elapsed if elapsed > 0 else 0
            eta = (len(todo) - i) / rate if rate > 0 else 0
            print(f"  [{i:3d}/{len(todo)}] {ticker} {date}: {len(rows)} bars "
                  f"({dt:.1f}s) | elapsed={elapsed:.0f}s ETA={eta:.0f}s")
        if fetched % args.flush_every == 0:
            cache = _persist(cache, pending_rows)
            pending_rows = []

    # Final flush
    cache = _persist(cache, pending_rows)

    print()
    print(f"✅ Fetched: {fetched}/{len(todo)}")
    print(f"   Cache rows: {len(cache):,}")
    print(f"   Failed: {len(failed)}")
    if failed[:10]:
        print("   First failures:")
        for t, d, typ, msg in failed[:10]:
            print(f"     {t} {d}: {typ} — {msg}")
    return 0 if not failed else 2


if __name__ == "__main__":
    raise SystemExit(main())
