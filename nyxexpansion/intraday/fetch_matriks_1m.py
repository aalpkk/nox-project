"""
Fetch 1-minute intraday bars from Matriks for v4C top-D signal days.

C-step of 17:30 proxy study:
  1min bars reveal where the edge concentrates between 17:30–18:00:
  17:30, 17:40, 17:45, 17:50, 17:55, 17:58 candidate entries.

Matriks historicalData notes:
  - interval enum: 1min | 5min | 15min | 1hour | daily | weekly | monthly
  - Response bar list under 'allBars'
  - startDate==endDate=signal_date → full day (~480 bars for 1min)
  - Rate: same pattern as 15m fetch (~1 call/sec)

Cache:
  output/nyxexp_intraday_1m_matriks.parquet — full day 1min bars
  (post-processing filters 17:00–18:00 window for the proxy study)
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


SIGNALS_CSV = Path("data/nyxexp_topd_signals.csv")
SIGNALS_PARQUET = Path("output/nyxexp_backtest_v4C.parquet")
CACHE_PATH = Path("output/nyxexp_intraday_1m_matriks.parquet")


def _load_signals() -> pd.DataFrame:
    if SIGNALS_CSV.exists():
        df = pd.read_csv(SIGNALS_CSV)
        df["date"] = pd.to_datetime(df["date"]).dt.date
        return df[["ticker", "date"]].sort_values(["date", "ticker"]).reset_index(drop=True)
    if SIGNALS_PARQUET.exists():
        df = pd.read_parquet(SIGNALS_PARQUET)
        df["date"] = pd.to_datetime(df["date"]).dt.date
        keys = df[["ticker", "date"]].drop_duplicates().reset_index(drop=True)
        return keys.sort_values(["date", "ticker"]).reset_index(drop=True)
    raise FileNotFoundError(f"Signals missing at {SIGNALS_CSV} or {SIGNALS_PARQUET}")


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
    """Single day 1min bars via Matriks historicalData.

    Post-filter: only keep TR 17:00–18:00 window (60 bars/day) to reduce
    parquet size by ~8x; full-day fetch is still one request.
    """
    s = date.isoformat()
    resp = client.call_tool("historicalData", {
        "symbol": ticker,
        "startDate": s,
        "endDate": s,
        "interval": "1min",
        "rawBars": True,
    })
    if not isinstance(resp, dict):
        return []
    bars = resp.get("allBars") or []
    out = []
    for b in bars:
        if not isinstance(b, dict):
            continue
        ts = b.get("timestamp")
        if ts is None:
            continue
        # TR saat 17:00–18:00 inclusive filter (UTC ms → TR)
        # Hour 17 → 17:00–17:59; hour 18 min 0 → closing auction tick @ 18:00
        tr = pd.Timestamp(ts, unit="ms", tz="UTC").tz_convert("Europe/Istanbul")
        if not (tr.hour == 17 or (tr.hour == 18 and tr.minute == 0)):
            continue
        out.append({
            "ticker": ticker,
            "signal_date": date,
            "bar_ts": ts,
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
    print(f"Filter: TR 17:00–18:00 window only (60 bars/signal)")
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
            failed.append((ticker, date, "empty", "no bars in 17:00-18:00"))
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
