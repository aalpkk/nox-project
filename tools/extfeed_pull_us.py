"""Pull US 1h bars from TV WS feed → US master parquet.

Mirrors tools/extfeed_delta_pull.py for US equities. Two modes:

  bootstrap   Pulls ~3y of 1h bars per ticker. If master parquet already
              exists, only NEW tickers (not already in master) are pulled —
              i.e. additive bootstrap. Existing tickers are left to delta.
              If master missing, full universe pulled.
  delta       Master parquet must exist. Universe read from existing tickers
              (delta does not re-derive exchange — uses cached exchange map).
              Pulls last 4 days, deduplicates, merges.

Auto-detect: if --mode omitted, choose bootstrap when master missing else delta.

Output: output/extfeed_intraday_1h_3y_master_us.parquet
Schema: ticker, ts_utc, ts_newyork, open, high, low, close, volume

Universe selection (priority):
  --universe TICKER1 TICKER2 ...      explicit list (all routed to NASDAQ)
  --use-universe {ndx100, sp500}      named universe with proper exchange routing
  (none) → bootstrap defaults to NDX100; delta uses master's existing tickers.

TV symbols are formed as "{exchange}:{ticker}". NDX100 is all NASDAQ-listed;
SP500 spans NASDAQ + NYSE + AMEX so the exchange routing lookup (NASDAQ Trader
listed files) is required.

Auth: INTRADAY_SID / INTRADAY_SIGN env vars (same TV session cookies used by
the BIST extfeed pull).
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from markets.extfeed import auth_from_env, fetch_bars_until
from markets.us.data import (
    _NDX100_STATIC,
    _fetch_us_exchange_map,
    get_ndx100_universe,
    get_sp500_universe,
)

OUT_BARS = REPO / "output" / "extfeed_intraday_1h_3y_master_us.parquet"

# ~252 trading days/yr * 6.5h/day * 3y ≈ 4900 1h bars per ticker.
# 5000 chunk + 2 chunks ceiling gives generous margin.
BOOTSTRAP_CHUNK_N = 5000
BOOTSTRAP_MAX_CHUNKS = 2
BOOTSTRAP_YEARS_BACK = 3

DELTA_CHUNK_N = 200
DELTA_MAX_CHUNKS = 2
DELTA_DAYS_BACK = 4


def symbol_for(code: str, exchange: str = "NASDAQ") -> str:
    """Map (ticker, exchange) → TV symbol."""
    return f"{exchange}:{code}"


def _normalize_bars(df: pd.DataFrame, code: str) -> pd.DataFrame:
    """ws_client returns df with `time` tz=Europe/Istanbul. Rewrite to US schema."""
    df = df.copy()
    df["ts_utc"] = df["time"].dt.tz_convert("UTC")
    df["ts_newyork"] = df["time"].dt.tz_convert("America/New_York")
    df["ticker"] = code
    return df[["ticker", "ts_utc", "ts_newyork", "open", "high", "low", "close", "volume"]]


def _normalize_universe(
    universe: list[str] | list[tuple[str, str]],
) -> list[tuple[str, str]]:
    """Accept either bare tickers (default NASDAQ) or (ticker, exchange) tuples."""
    norm: list[tuple[str, str]] = []
    for item in universe:
        if isinstance(item, str):
            norm.append((item, "NASDAQ"))
        else:
            norm.append((item[0], item[1]))
    return norm


def pull_universe(
    *,
    universe: list[str] | list[tuple[str, str]],
    until_date: pd.Timestamp,
    chunk_n: int,
    max_chunks: int,
) -> tuple[pd.DataFrame, int, int, list[str]]:
    """Loop universe, pull bars until until_date, return (df, ok, fail, fail_list)."""
    auth = auth_from_env()
    _ = auth.token()
    print(f"[auth] JWT acquired, expires_in={auth.expires_at - int(time.time())}s", flush=True)

    pairs = _normalize_universe(universe)
    new_chunks: list[pd.DataFrame] = []
    fail_list: list[str] = []
    ok = 0
    t_start = time.time()
    for i, (code, exch) in enumerate(pairs, 1):
        symbol = symbol_for(code, exch)
        try:
            df, stats = fetch_bars_until(
                symbol=symbol,
                timeframe="60",
                until_date=until_date,
                chunk_n=chunk_n,
                max_chunks=max_chunks,
                chunk_timeout_s=30.0,
                inter_chunk_delay_s=0.3,
                auth=auth,
            )
            if df is None or df.empty:
                fail_list.append(symbol)
                continue
            new_chunks.append(_normalize_bars(df, code))
            ok += 1
        except Exception as e:
            fail_list.append(symbol)
            if len(fail_list) <= 5 or i % 25 == 0:
                print(f"  fail {symbol}: {e}", flush=True)
        if i % 25 == 0:
            elapsed = time.time() - t_start
            eta = elapsed / i * (len(pairs) - i)
            print(f"  [{i}/{len(pairs)}] elapsed={elapsed:.0f}s eta={eta:.0f}s "
                  f"ok={ok} fails={len(fail_list)}", flush=True)

    if not new_chunks:
        return (pd.DataFrame(columns=["ticker", "ts_utc", "ts_newyork",
                                      "open", "high", "low", "close", "volume"]),
                ok, len(fail_list), fail_list)
    return pd.concat(new_chunks, ignore_index=True), ok, len(fail_list), fail_list


def _resolve_named_universe(name: str) -> list[tuple[str, str]]:
    name = name.lower()
    if name == "ndx100":
        return get_ndx100_universe()
    if name == "sp500":
        return get_sp500_universe()
    raise ValueError(f"Unknown --use-universe: {name!r} (expected ndx100|sp500)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["bootstrap", "delta", "auto"], default="auto")
    ap.add_argument("--universe", nargs="*", default=None,
                    help="Explicit ticker list (all routed to NASDAQ). For "
                         "mixed-exchange universes use --use-universe.")
    ap.add_argument("--use-universe", choices=["ndx100", "sp500"], default=None,
                    help="Named universe with exchange routing (overrides --universe).")
    args = ap.parse_args()

    mode = args.mode
    if mode == "auto":
        mode = "delta" if OUT_BARS.exists() else "bootstrap"
    print(f"[mode] {mode}", flush=True)

    if mode == "bootstrap":
        if args.use_universe:
            universe: list[str] | list[tuple[str, str]] = _resolve_named_universe(args.use_universe)
            uni_label = args.use_universe.upper()
        elif args.universe:
            universe = sorted(set(args.universe))
            uni_label = "custom"
        else:
            universe = sorted(set(_NDX100_STATIC))
            uni_label = "NDX100-static"
        until_date = pd.Timestamp(
            datetime.utcnow() - timedelta(days=BOOTSTRAP_YEARS_BACK * 366)
        ).tz_localize("UTC")

        # Additive bootstrap: if master exists, skip tickers already present.
        if OUT_BARS.exists():
            existing = pd.read_parquet(OUT_BARS)
            existing_tickers = set(existing["ticker"].dropna().unique().tolist())
            pairs = _normalize_universe(universe)
            new_pairs = [(t, e) for t, e in pairs if t not in existing_tickers]
            skipped = len(pairs) - len(new_pairs)
            print(f"[bootstrap] universe={uni_label} {len(pairs)} tickers — "
                  f"master has {len(existing_tickers)} existing, "
                  f"pulling {len(new_pairs)} new (skipped {skipped})", flush=True)
            universe = new_pairs
            if not new_pairs:
                print("[bootstrap] nothing new to pull — master already covers universe.",
                      flush=True)
                return 0
        else:
            existing = pd.DataFrame(columns=["ticker", "ts_utc", "ts_newyork",
                                             "open", "high", "low", "close", "volume"])
            print(f"[bootstrap] universe={uni_label} {len(_normalize_universe(universe))} "
                  f"tickers, until={until_date.date()} (~{BOOTSTRAP_YEARS_BACK}y back, "
                  f"fresh master)", flush=True)

        delta, ok, fail, fail_list = pull_universe(
            universe=universe,
            until_date=until_date,
            chunk_n=BOOTSTRAP_CHUNK_N,
            max_chunks=BOOTSTRAP_MAX_CHUNKS,
        )
    else:
        if not OUT_BARS.exists():
            print(f"!! master parquet missing for delta mode: {OUT_BARS}\n"
                  f"   run with --mode bootstrap first.", flush=True)
            return 1
        existing = pd.read_parquet(OUT_BARS)
        print(f"[delta] loaded master: {existing['ticker'].nunique()} tickers, "
              f"{len(existing):,} rows, max ts_utc={existing['ts_utc'].max()}", flush=True)
        # Re-derive exchange routing for master tickers via NASDAQ Trader map.
        exch_map = _fetch_us_exchange_map()
        master_tickers = sorted(existing["ticker"].dropna().unique().tolist())
        if args.universe:
            master_tickers = sorted(set(args.universe))
        universe = [(t, exch_map.get(t, "NASDAQ")) for t in master_tickers]
        until_date = pd.Timestamp(
            datetime.utcnow() - timedelta(days=DELTA_DAYS_BACK)
        ).tz_localize("UTC")
        print(f"[delta] universe={len(universe)} tickers, until={until_date} "
              f"(~{DELTA_DAYS_BACK} days back)", flush=True)
        delta, ok, fail, fail_list = pull_universe(
            universe=universe,
            until_date=until_date,
            chunk_n=DELTA_CHUNK_N,
            max_chunks=DELTA_MAX_CHUNKS,
        )

    if delta.empty:
        print("!! no rows pulled", flush=True)
        return 2

    print(f"[pull] {len(delta):,} rows, "
          f"date range {delta['ts_utc'].min()} → {delta['ts_utc'].max()} "
          f"({ok} ok / {fail} fail)", flush=True)
    if fail_list:
        head = ", ".join(fail_list[:20])
        more = f" (+{len(fail_list) - 20} more)" if len(fail_list) > 20 else ""
        print(f"[fail-list] {head}{more}", flush=True)

    combined = pd.concat([existing, delta], ignore_index=True)
    before = len(combined)
    combined = combined.drop_duplicates(subset=["ticker", "ts_utc"], keep="last")
    after = len(combined)
    combined = combined.sort_values(["ticker", "ts_utc"]).reset_index(drop=True)
    print(f"[merge] {before:,} → {after:,} rows ({before - after:,} dups dropped)", flush=True)
    print(f"[master] tickers={combined['ticker'].nunique()} "
          f"rows={len(combined):,} max ts_utc={combined['ts_utc'].max()}", flush=True)

    OUT_BARS.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(OUT_BARS, index=False)
    print(f"[write] {OUT_BARS}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
