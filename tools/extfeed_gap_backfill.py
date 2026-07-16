"""Backfill historical gaps in the extfeed 1h master parquet.

Why: delta_pull only merges the last ~4 days (chunk_n=200, until=now-4d).
If the pull chain is down for a stretch (e.g. 2026-04-25 → 2026-06-16,
universe-wide), the hole never heals — resampled weekly/monthly panels then
bridge the gap and mb_scanner emits ghost pivots/quartets (PEKGY bb_1w
2026-07-10 "birth" post-mortem).

What it does, per ticker in the master:
  1. find the largest bar-date gap inside [--scan-start, now]
  2. if gap >= --min-gap-days, pull 1h bars back to (gap start - buffer)
     via fetch_bars_until and merge (dedup on ticker+ts_utc, keep new)
  3. checkpoint the merged master every --checkpoint tickers (rerun-safe:
     healed tickers no longer exceed the gap threshold and are skipped)
  4. report tickers whose gap persists after the pull (suspended/delisted
     names legitimately have holes — reported, not fatal)

Usage:
  set -a; source .env; set +a   # INTRADAY_SID / INTRADAY_SIGN
  python tools/extfeed_gap_backfill.py [--dry-run] [--tickers PEKGY ...]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from markets.extfeed import auth_from_env, fetch_bars_until

MASTER = Path("output/extfeed_intraday_1h_3y_master.parquet")


def _bar_days(ts: pd.Series) -> pd.Series:
    """tz-aware ts_istanbul → naive normalized Istanbul dates."""
    s = pd.to_datetime(ts, utc=True).dt.tz_convert("Europe/Istanbul")
    return s.dt.normalize().dt.tz_localize(None)


def find_max_gap(days: list[pd.Timestamp]) -> tuple[int, pd.Timestamp | None, pd.Timestamp | None]:
    """(gap_days, last_bar_before, first_bar_after) over sorted unique days."""
    if len(days) < 2:
        return 0, None, None
    arr = pd.DatetimeIndex(days)
    deltas = (arr[1:] - arr[:-1]).days
    i = int(np.argmax(deltas))
    return int(deltas[i]), arr[i], arr[i + 1]


def main() -> int:
    ap = argparse.ArgumentParser()
    default_scan_start = (pd.Timestamp.now() - pd.Timedelta(days=90)).strftime("%Y-%m-%d")
    ap.add_argument("--scan-start", default=default_scan_start,
                    help="start of the window scanned for gaps (default: today-90d, "
                         "rolling — old legit suspensions age out)")
    ap.add_argument("--min-gap-days", type=int, default=14,
                    help="calendar-day gap that triggers a backfill (default 14)")
    ap.add_argument("--buffer-days", type=int, default=7,
                    help="pull this many days before the gap start (default 7)")
    ap.add_argument("--chunk-n", type=int, default=1500,
                    help="bars per WS chunk (default 1500 ≈ 6 months of 1h)")
    ap.add_argument("--max-chunks", type=int, default=4)
    ap.add_argument("--checkpoint", type=int, default=100,
                    help="write merged master every N healed tickers (default 100)")
    ap.add_argument("--tickers", nargs="*", default=None,
                    help="subset (default: all master tickers with a gap)")
    ap.add_argument("--dry-run", action="store_true",
                    help="detect + report gaps only, no pull/write")
    args = ap.parse_args()

    if not MASTER.exists():
        print(f"!! master parquet missing: {MASTER}", flush=True)
        return 1

    master = pd.read_parquet(MASTER)
    print(f"loaded master: {master['ticker'].nunique()} tickers, {len(master):,} rows, "
          f"max ts={master['ts_istanbul'].max()}", flush=True)

    scan_start = pd.Timestamp(args.scan_start)
    day_col = _bar_days(master["ts_utc"])

    # -------- gap detection
    todo: list[tuple[str, int, pd.Timestamp, pd.Timestamp]] = []
    universe = args.tickers or sorted(master["ticker"].dropna().unique().tolist())
    for code in universe:
        days = sorted(set(day_col[master["ticker"] == code]))
        days = [d for d in days if d >= scan_start]
        gap, before, after = find_max_gap(days)
        if gap >= args.min_gap_days:
            todo.append((code, gap, before, after))

    if not todo:
        print("no gaps ≥ threshold — nothing to do", flush=True)
        return 0

    starts = pd.Series([b for _, _, b, _ in todo])
    print(f"gap tickers: {len(todo)}/{len(universe)}  "
          f"(gap-start mode: {starts.mode().iat[0].date()}, "
          f"max gap {max(g for _, g, _, _ in todo)}d)", flush=True)
    if args.dry_run:
        for code, gap, b, a in todo[:20]:
            print(f"  {code:8s} {gap:3d}d  {b.date()} → {a.date()}")
        if len(todo) > 20:
            print(f"  ... +{len(todo) - 20} more")
        return 0

    # -------- pull + merge
    auth = auth_from_env()
    _ = auth.token()
    print("JWT acquired", flush=True)

    healed: list[pd.DataFrame] = []
    fail: list[str] = []
    t0 = time.time()
    n_since_ckpt = 0

    def _flush(final: bool = False) -> None:
        nonlocal master, healed, n_since_ckpt
        if not healed:
            return
        delta = pd.concat(healed, ignore_index=True)
        combined = pd.concat([master, delta], ignore_index=True)
        combined = combined.drop_duplicates(subset=["ticker", "ts_utc"], keep="last")
        combined = combined.sort_values(["ticker", "ts_utc"], ignore_index=True)
        tmp = MASTER.with_suffix(".parquet.tmp")
        combined.to_parquet(tmp, index=False)
        tmp.replace(MASTER)
        tag = "final" if final else "checkpoint"
        print(f"  [{tag}] wrote master: {len(master):,} → {len(combined):,} rows", flush=True)
        master = combined
        healed = []
        n_since_ckpt = 0

    for i, (code, gap, before, after) in enumerate(todo, 1):
        until = (before - pd.Timedelta(days=args.buffer_days)).tz_localize(
            "Europe/Istanbul").tz_convert("UTC")
        try:
            df, stats = fetch_bars_until(
                symbol=f"BIST:{code}", timeframe="60", until_date=until,
                chunk_n=args.chunk_n, max_chunks=args.max_chunks,
                chunk_timeout_s=20.0, inter_chunk_delay_s=0.2, auth=auth,
            )
            if df is None or df.empty:
                fail.append(code)
                continue
            df["ticker"] = code
            df["ts_utc"] = df["time"].dt.tz_convert("UTC")
            df["ts_istanbul"] = df["time"]
            df = df[["ticker", "ts_utc", "ts_istanbul",
                     "open", "high", "low", "close", "volume"]]
            healed.append(df)
            n_since_ckpt += 1
        except Exception as e:
            fail.append(code)
            print(f"  fail {code}: {e}", flush=True)
        if i % 25 == 0:
            el = time.time() - t0
            print(f"  [{i}/{len(todo)}] elapsed={el:.0f}s "
                  f"eta={el / i * (len(todo) - i):.0f}s fails={len(fail)}", flush=True)
        if n_since_ckpt >= args.checkpoint:
            _flush()
    _flush(final=True)

    # -------- post-verify
    day_col = _bar_days(master["ts_utc"])
    remaining: list[tuple[str, int, pd.Timestamp, pd.Timestamp]] = []
    for code, *_ in todo:
        days = sorted(set(day_col[master["ticker"] == code]))
        days = [d for d in days if d >= scan_start]
        gap, before, after = find_max_gap(days)
        if gap >= args.min_gap_days:
            remaining.append((code, gap, before, after))

    print(f"\npost-verify: {len(todo) - len(remaining)}/{len(todo)} healed; "
          f"{len(remaining)} still gapped; pull-fails={len(fail)}", flush=True)
    for code, gap, b, a in remaining[:15]:
        print(f"  STILL {code:8s} {gap:3d}d  {b.date()} → {a.date()}  "
              f"(suspended/delisted?)", flush=True)
    if fail:
        print(f"  pull-fails: {fail[:15]}{' ...' if len(fail) > 15 else ''}", flush=True)
    print(f"master final: {len(master):,} rows, max ts={master['ts_istanbul'].max()}",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
