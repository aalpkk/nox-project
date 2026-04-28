"""Phase 1 — full 607 × 3y × 1h bulk pull via extfeed (TV WebSocket).

Reads the ticker universe from output/ohlcv_10y_fintables_master.parquet,
pulls 1h bars 3 years back, writes a canonical master parquet, a coverage
report, and a volume reconciliation table against the daily master.

Resume support: if the master parquet already exists, tickers already
present (with bars covering the target span) are skipped. Force a rebuild
with --force.

Usage:
    python tools/extfeed_phase1_bulk.py [--force] [--limit N] [--years Y]
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from markets.extfeed import auth_from_env, fetch_bars_until


DAILY_MASTER = Path("output/ohlcv_10y_fintables_master.parquet")
OUT_BARS = Path("output/extfeed_intraday_1h_3y_master.parquet")
OUT_COVERAGE = Path("output/extfeed_intraday_coverage.csv")
OUT_VOLUME = Path("output/extfeed_intraday_volume_recon.csv")
OUT_LOG = Path("output/_extfeed_phase1.log")


def load_universe() -> list[str]:
    df = pd.read_parquet(DAILY_MASTER, columns=["ticker"])
    tickers = sorted(df["ticker"].dropna().unique().tolist())
    return tickers


def coverage_pct(span_days: int, target_days: int) -> float:
    if target_days <= 0:
        return 0.0
    return min(span_days / target_days, 1.0)


def write_progress(bars_accum: list[pd.DataFrame], rows: list[dict]) -> None:
    if bars_accum:
        df = pd.concat(bars_accum, ignore_index=True)
        df.to_parquet(OUT_BARS, index=False)
    if rows:
        pd.DataFrame(rows).to_csv(OUT_COVERAGE, index=False)


def volume_reconciliation(bars: pd.DataFrame) -> pd.DataFrame:
    """sum(1h volume per Istanbul date) vs Fintables daily volume."""
    if bars.empty:
        return pd.DataFrame()
    bars = bars.copy()
    bars["date"] = bars["ts_istanbul"].dt.date
    intraday_daily = (
        bars.groupby(["ticker", "date"], observed=True)["volume"]
        .sum().reset_index().rename(columns={"volume": "intraday_vol"})
    )
    daily = pd.read_parquet(DAILY_MASTER, columns=["ticker", "Volume"])
    daily.index.name = "Date"
    daily = daily.reset_index()
    daily["date"] = pd.to_datetime(daily["Date"]).dt.date
    daily = daily.rename(columns={"Volume": "daily_vol"})[["ticker", "date", "daily_vol"]]
    merged = intraday_daily.merge(daily, on=["ticker", "date"], how="inner")
    if merged.empty:
        return pd.DataFrame()
    merged["ratio"] = merged["intraday_vol"] / merged["daily_vol"].replace(0, pd.NA)

    def per_ticker(g):
        return pd.Series({
            "n_dates": int(len(g)),
            "median_ratio": float(g["ratio"].median()),
            "mean_ratio": float(g["ratio"].mean()),
            "p10_ratio": float(g["ratio"].quantile(0.10)),
            "p90_ratio": float(g["ratio"].quantile(0.90)),
            "n_within_10pct": int(g["ratio"].between(0.9, 1.1).sum()),
        })

    return merged.groupby("ticker", observed=True).apply(per_ticker).reset_index()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="ignore existing master parquet, rebuild from scratch")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap universe size (debug)")
    ap.add_argument("--years", type=int, default=3,
                    help="target lookback years (default 3)")
    args = ap.parse_args()

    universe = load_universe()
    if args.limit:
        universe = universe[: args.limit]
    target_until = pd.Timestamp(
        datetime.utcnow() - timedelta(days=365 * args.years)
    ).tz_localize("UTC")
    target_days = args.years * 365

    print(f"  universe   : {len(universe)} tickers")
    print(f"  timeframe  : 60 (1h)")
    print(f"  target     : {args.years}y back  →  until_ts={target_until.date()}")

    skip = set()
    bars_accum: list[pd.DataFrame] = []
    rows: list[dict] = []

    if OUT_BARS.exists() and not args.force:
        existing = pd.read_parquet(OUT_BARS)
        print(f"  resuming   : found existing master with "
              f"{existing['ticker'].nunique()} tickers, {len(existing):,} rows")
        for tk, g in existing.groupby("ticker", observed=True):
            span = (g["ts_istanbul"].max() - g["ts_istanbul"].min()).days
            if span >= target_days * 0.95:
                skip.add(tk)
                bars_accum.append(g)
        print(f"  skipping   : {len(skip)} tickers already covered")
        # Re-load existing coverage rows so we don't lose them
        if OUT_COVERAGE.exists():
            existing_cov = pd.read_csv(OUT_COVERAGE)
            rows = existing_cov[existing_cov["ticker"].isin(skip)].to_dict("records")

    auth = auth_from_env()
    _ = auth.token()
    print(f"  ✓ JWT acquired  expires_in={auth.expires_at - int(time.time())}s")
    print()

    work = [tk for tk in universe if tk not in skip]
    print(f"  to pull    : {len(work)} tickers")
    print()

    t_start = time.time()
    progress_every = 50
    log_lines: list[str] = []

    for i, code in enumerate(work, 1):
        symbol = f"BIST:{code}"
        t0 = time.time()
        try:
            df, stats = fetch_bars_until(
                symbol=symbol, timeframe="60", until_date=target_until,
                chunk_n=2000, max_chunks=10, chunk_timeout_s=30.0,
                inter_chunk_delay_s=0.5, auth=auth,
            )
            dt = time.time() - t0
            df["ticker"] = code
            df["ts_utc"] = df["time"].dt.tz_convert("UTC")
            df["ts_istanbul"] = df["time"]
            df = df[["ticker", "ts_utc", "ts_istanbul",
                     "open", "high", "low", "close", "volume"]]
            bars_accum.append(df)
            span = (df["ts_istanbul"].max() - df["ts_istanbul"].min()).days
            term = (
                "target" if stats["reached_target"]
                else "no_progress" if stats["no_progress_break"]
                else "max_chunks" if stats["max_chunks_break"]
                else "unknown"
            )
            rows.append({
                "ticker": code,
                "n_bars": len(df),
                "first_ts": df["ts_istanbul"].min(),
                "last_ts": df["ts_istanbul"].max(),
                "span_days": span,
                "coverage_pct": round(coverage_pct(span, target_days), 4),
                "chunks_sent": stats["chunks_sent"],
                "termination": term,
                "time_s": round(dt, 2),
                "status": "ok",
                "error": "",
            })
            if i % 25 == 0 or i == len(work):
                msg = (f"  [{i:4d}/{len(work)}] {code:6s} {len(df):5d} bars "
                       f"{span:4d}d  {dt:5.2f}s  term={term}")
                print(msg)
                log_lines.append(msg)
        except Exception as e:
            dt = time.time() - t0
            rows.append({
                "ticker": code,
                "n_bars": 0, "first_ts": None, "last_ts": None,
                "span_days": 0, "coverage_pct": 0.0,
                "chunks_sent": 0, "termination": "error",
                "time_s": round(dt, 2),
                "status": "fail",
                "error": f"{type(e).__name__}: {str(e)[:160]}",
            })
            msg = f"  [{i:4d}/{len(work)}] {code:6s}  FAIL  {dt:.2f}s  {e}"
            print(msg)
            log_lines.append(msg)

        if i % progress_every == 0:
            write_progress(bars_accum, rows)

        if i < len(work):
            time.sleep(1.0)

    write_progress(bars_accum, rows)
    elapsed = time.time() - t_start

    print()
    print("=" * 70)
    summary = pd.DataFrame(rows)
    ok = summary[summary["status"] == "ok"]
    print(f"  total elapsed   : {elapsed/60:.1f} min  ({elapsed:.0f}s)")
    print(f"  ok/fail         : {len(ok)}/{len(summary)}")
    if not ok.empty:
        total_bars = int(ok["n_bars"].sum())
        print(f"  total bars      : {total_bars:,}")
        print(f"  coverage avg    : {ok['coverage_pct'].mean():.3f}")
        print(f"  coverage<0.95   : {(ok['coverage_pct'] < 0.95).sum()} tickers")
        print(f"  coverage<0.50   : {(ok['coverage_pct'] < 0.50).sum()} tickers")
        term_counts = ok["termination"].value_counts().to_dict()
        print(f"  terminations    : {term_counts}")
    if (summary["status"] == "fail").any():
        n_fail = int((summary["status"] == "fail").sum())
        print(f"  failures        : {n_fail}")
        print()
        for _, row in summary[summary["status"] == "fail"].head(20).iterrows():
            print(f"    {row['ticker']:6s}  {row['error']}")

    print()
    print(f"  ✓ master   → {OUT_BARS}")
    print(f"  ✓ coverage → {OUT_COVERAGE}")

    if bars_accum:
        all_bars = pd.read_parquet(OUT_BARS)
        recon = volume_reconciliation(all_bars)
        if not recon.empty:
            recon.to_csv(OUT_VOLUME, index=False)
            print(f"  ✓ volume   → {OUT_VOLUME}")
            print()
            print(f"  volume reconciliation:")
            print(f"    median ratio (per-ticker median) : "
                  f"{recon['median_ratio'].median():.3f}")
            print(f"    n_tickers with median∈[0.9,1.1]  : "
                  f"{recon['median_ratio'].between(0.9, 1.1).sum()}/{len(recon)}")

    OUT_LOG.write_text("\n".join(log_lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
