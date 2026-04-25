"""Gap-only Matriks 15m fetcher for SBT-1700 signal seeds.

Reads ``output/sbt1700_signal_seed.csv`` (built by sbt1700.signal_seed),
identifies (ticker, date) pairs missing from BOTH the nyxexp 15m cache
and the SBT-1700 cache, and fetches the gap via the existing Matriks
client. Writes new rows ONLY to ``output/sbt1700_intraday_15m.parquet``
to avoid polluting the nyxexp cache.

Resumable: re-running picks up where the last run left off.

Env: MATRIKS_API_KEY (required).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd

# Re-use the existing low-level helpers from nyxexpansion.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.matriks_client import MatriksClient  # noqa: E402
from nyxexpansion.intraday.fetch_matriks_15m import _fetch_one  # noqa: E402


SEED_PATH = Path("output/sbt1700_signal_seed.csv")
NYXEXP_CACHE = Path("output/nyxexp_intraday_15m_matriks.parquet")
SBT_CACHE = Path("output/sbt1700_intraday_15m.parquet")


def _load_seed(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df[["ticker", "date"]].drop_duplicates().reset_index(drop=True)


def _existing_keys(paths: list[Path]) -> set[tuple[str, object]]:
    keys: set[tuple[str, object]] = set()
    for p in paths:
        if not p.exists():
            continue
        df = pd.read_parquet(p, columns=["ticker", "signal_date"])
        df["signal_date"] = pd.to_datetime(df["signal_date"]).dt.date
        for r in df.drop_duplicates().itertuples(index=False):
            keys.add((r.ticker, r.signal_date))
    return keys


def _persist(rows: list[dict]) -> int:
    if not rows:
        return 0
    new_df = pd.DataFrame(rows)
    if SBT_CACHE.exists():
        existing = pd.read_parquet(SBT_CACHE)
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df
    SBT_CACHE.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(SBT_CACHE, index=False)
    return len(new_df)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=Path, default=SEED_PATH)
    ap.add_argument("--canary", type=int, default=0,
                    help="Only fetch first N missing pairs.")
    ap.add_argument("--flush-every", type=int, default=20)
    args = ap.parse_args()

    if not os.environ.get("MATRIKS_API_KEY"):
        print("ERROR: MATRIKS_API_KEY not set")
        return 1
    if not args.seed.exists():
        print(f"ERROR: seed file missing: {args.seed}")
        return 1

    seed = _load_seed(args.seed)
    have = _existing_keys([NYXEXP_CACHE, SBT_CACHE])
    todo = [
        (r.ticker, r.date) for r in seed.itertuples(index=False)
        if (r.ticker, r.date) not in have
    ]
    if args.canary > 0:
        todo = todo[: args.canary]

    print(f"Seed pairs: {len(seed):,}")
    print(f"Already cached: {len(have):,} (nyxexp + sbt1700)")
    print(f"Gap to fetch: {len(todo):,}")
    if args.canary:
        print(f"  [canary] limited to {args.canary}")

    client = MatriksClient()
    fetched = 0
    failed: list[tuple[str, object, str, str]] = []
    pending: list[dict] = []
    t0 = time.time()

    for i, (tk, dte) in enumerate(todo, 1):
        try:
            rows = _fetch_one(client, tk, dte)
        except Exception as e:
            failed.append((tk, dte, type(e).__name__, str(e)[:120]))
            print(f"  [{i:4d}/{len(todo)}] {tk} {dte}  ERR {type(e).__name__}: {str(e)[:80]}")
            continue
        if not rows:
            failed.append((tk, dte, "empty", "no bars"))
            continue
        pending.extend(rows)
        fetched += 1
        if i % 25 == 0 or i == len(todo):
            elapsed = time.time() - t0
            rate = i / elapsed if elapsed > 0 else 0
            eta = (len(todo) - i) / rate if rate > 0 else 0
            print(f"  [{i:4d}/{len(todo)}] fetched={fetched} failed={len(failed)} "
                  f"elapsed={elapsed:.0f}s ETA={eta:.0f}s")
        if fetched % args.flush_every == 0:
            written = _persist(pending)
            pending = []
            print(f"    flushed {written} rows → {SBT_CACHE}")

    if pending:
        written = _persist(pending)
        print(f"  final flush {written} rows → {SBT_CACHE}")

    print()
    print(f"Fetched OK: {fetched}/{len(todo)}")
    print(f"Failed: {len(failed)}")
    return 0 if fetched > 0 or not todo else 2


if __name__ == "__main__":
    raise SystemExit(main())
