"""Phase 0 — multi-ticker × 3y × 1h smoke for shared intraday backtest dataset.

Pulls 1h bars 3 years back for a representative 10-ticker BIST sample,
measuring per-ticker throughput, JWT survival, and pagination depth.
Writes the bar union to parquet so a follow-up step can run volume
reconciliation against the Fintables daily master.

Usage:
    python tools/extfeed_phase0_smoke.py [N_TICKERS] [TIMEFRAME] [YEARS]
Defaults: 10 60 3
"""
from __future__ import annotations

import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from markets.extfeed import auth_from_env, fetch_bars_until


# 10-ticker mix: 4 A-tier liquid + 4 mid-cap + 2 small/recent
DEFAULT_UNIVERSE = [
    # A-tier liquid (banks + heavyweights)
    "GARAN", "AKBNK", "THYAO", "BIMAS",
    # mid-cap broad
    "ASELS", "FROTO", "EREGL", "TUPRS",
    # small / less-liquid
    "MGROS", "VESBE",
]


def main() -> int:
    n_tickers = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    timeframe = sys.argv[2] if len(sys.argv) > 2 else "60"
    years = int(sys.argv[3]) if len(sys.argv) > 3 else 3

    universe = DEFAULT_UNIVERSE[:n_tickers]
    target_until = pd.Timestamp(
        datetime.utcnow() - timedelta(days=365 * years)
    ).tz_localize("UTC")

    print(f"  universe   : {len(universe)} tickers — {universe}")
    print(f"  timeframe  : {timeframe}")
    print(f"  target     : {years}y back  →  until_ts={target_until.date()}")
    print()

    auth = auth_from_env()
    t_jwt0 = time.time()
    token = auth.token()
    jwt_age0 = auth.expires_at - int(time.time())
    print(f"  ✓ JWT acquired  expires_in={jwt_age0}s  ({(time.time() - t_jwt0):.2f}s)")
    print()

    rows = []
    all_bars = []
    t_start = time.time()

    for i, code in enumerate(universe, 1):
        symbol = f"BIST:{code}"
        t0 = time.time()
        try:
            df, stats = fetch_bars_until(
                symbol=symbol,
                timeframe=timeframe,
                until_date=target_until,
                chunk_n=2000,
                max_chunks=10,
                chunk_timeout_s=30.0,
                inter_chunk_delay_s=0.5,
                auth=auth,
            )
            dt = time.time() - t0
            df["ticker"] = code
            all_bars.append(df)
            term = (
                "target" if stats["reached_target"]
                else "no_progress" if stats["no_progress_break"]
                else "max_chunks" if stats["max_chunks_break"]
                else "unknown"
            )
            rows.append({
                "ticker": code,
                "n_bars": len(df),
                "first_ts": df["time"].min(),
                "last_ts": df["time"].max(),
                "span_days": (df["time"].max() - df["time"].min()).days,
                "chunks_sent": stats["chunks_sent"],
                "chunks_received": stats["chunks_received"],
                "termination": term,
                "time_s": round(dt, 2),
                "bars_per_s": round(len(df) / max(dt, 1e-6), 1),
                "status": "ok",
                "error": "",
            })
            print(f"  [{i:2d}/{len(universe)}] {code:6s} "
                  f"{len(df):5d} bars  {(df['time'].max() - df['time'].min()).days:4d}d  "
                  f"chunks={stats['chunks_sent']:2d}  {dt:5.2f}s  term={term}")
        except Exception as e:
            dt = time.time() - t0
            rows.append({
                "ticker": code,
                "n_bars": 0, "first_ts": None, "last_ts": None,
                "span_days": 0, "chunks_sent": 0, "chunks_received": 0,
                "termination": "error",
                "time_s": round(dt, 2), "bars_per_s": 0.0,
                "status": "fail",
                "error": f"{type(e).__name__}: {str(e)[:160]}",
            })
            print(f"  [{i:2d}/{len(universe)}] {code:6s}  FAIL  {dt:.2f}s  {e}")
        # politeness pause between tickers
        if i < len(universe):
            time.sleep(1.0)

    elapsed = time.time() - t_start
    jwt_age1 = auth.expires_at - int(time.time())

    print()
    print("=" * 70)
    summary = pd.DataFrame(rows)
    ok = summary[summary["status"] == "ok"]
    print(f"  total elapsed   : {elapsed:.1f}s")
    print(f"  ok/fail         : {len(ok)}/{len(summary)}")
    if not ok.empty:
        total_bars = int(ok["n_bars"].sum())
        print(f"  total bars      : {total_bars:,}")
        print(f"  avg bars/tk     : {total_bars / len(ok):,.0f}")
        print(f"  span avg/min    : {ok['span_days'].mean():.0f}d / {ok['span_days'].min()}d")
        print(f"  throughput agg  : {total_bars / elapsed:.0f} bars/s")
        print(f"  per-ticker time : avg {ok['time_s'].mean():.2f}s  "
              f"p95 {ok['time_s'].quantile(0.95):.2f}s")
        term_counts = ok["termination"].value_counts().to_dict()
        print(f"  terminations    : {term_counts}")
    print(f"  JWT age before  : {jwt_age0}s  (~ttl)")
    print(f"  JWT age after   : {jwt_age1}s  (refresh needed if ≤ 0)")
    if (summary["status"] == "fail").any():
        print()
        print("  failures:")
        for _, row in summary[summary["status"] == "fail"].iterrows():
            print(f"    {row['ticker']}: {row['error']}")

    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)
    summary_csv = out_dir / "extfeed_phase0_smoke_summary.csv"
    summary.to_csv(summary_csv, index=False)
    print()
    print(f"  ✓ summary  → {summary_csv}")

    if all_bars:
        bars_df = pd.concat(all_bars, ignore_index=True)
        bars_pq = out_dir / "extfeed_phase0_smoke_bars.parquet"
        # Convert tz-aware time to UTC for parquet compat across pipelines
        bars_df["ts_utc"] = bars_df["time"].dt.tz_convert("UTC")
        bars_df["ts_istanbul"] = bars_df["time"]
        bars_df = bars_df[["ticker", "ts_utc", "ts_istanbul",
                           "open", "high", "low", "close", "volume"]]
        bars_df.to_parquet(bars_pq, index=False)
        print(f"  ✓ bars     → {bars_pq}  ({len(bars_df):,} rows)")

    n_fail = int((summary["status"] == "fail").sum())
    return 0 if n_fail == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
