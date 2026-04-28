"""Reconciliation: extfeed daily-D bars vs 1h-pulled-and-resampled-daily.

For 10 liquid BIST tickers, pull both:
  - timeframe="D",  ~80 bars
  - timeframe="60", ~80*7 bars (cover same span)
Resample 1h → daily (Europe/Istanbul calendar), inner-join on (ticker, date),
report close/volume diff stats.

Gating decision for screener_combo daily live system:
  - if median |close diff|/close < 0.1% AND volume ratio ∈ [0.95, 1.05]
    → daily-D is safe, proceed with A
  - else                                                       → fall back to B (1h pull + resample)

Output:
  output/extfeed_daily_recon.csv          — per (ticker,date) diff
  output/extfeed_daily_recon_summary.csv  — per-ticker aggregates

Usage (locally if creds set, else dispatch via .github/workflows/extfeed-smoke.yml):
  python tools/extfeed_daily_recon.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from markets.extfeed import auth_from_env, fetch_bars


TICKERS = ["THYAO", "GARAN", "AKBNK", "ASELS", "EREGL",
           "TUPRS", "BIMAS", "KCHOL", "SAHOL", "FROTO"]
N_BARS_D = 80  # ~80 trading days back
OUT_DETAIL = Path("output/extfeed_daily_recon.csv")
OUT_SUMMARY = Path("output/extfeed_daily_recon_summary.csv")


def pull(auth, ticker: str, tf: str, n: int) -> pd.DataFrame:
    df = fetch_bars(f"BIST:{ticker}", tf, n, auth=auth, timeout_s=30)
    if df.empty:
        return df
    df = df.copy()
    df["ticker"] = ticker
    return df


def to_daily_d(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["date"] = out["time"].dt.date
    return out[["ticker", "date", "open", "high", "low", "close", "volume"]]


def resample_1h_to_daily(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    d = df.copy()
    d["date"] = d["time"].dt.date
    agg = (
        d.groupby(["ticker", "date"])
        .agg(open=("open", "first"),
             high=("high", "max"),
             low=("low", "min"),
             close=("close", "last"),
             volume=("volume", "sum"),
             n_bars=("close", "count"))
        .reset_index()
    )
    return agg


def main() -> int:
    print(f"  tickers    : {TICKERS}")
    print(f"  n_bars (D) : {N_BARS_D}")
    print()

    auth = auth_from_env()
    auth.token()
    print(f"  ✓ JWT acquired")

    n_1h = N_BARS_D * 8  # 7-bar TR session + cushion
    daily_d_frames = []
    bars_1h_frames = []
    t0 = time.time()
    for tk in TICKERS:
        try:
            d_df = pull(auth, tk, "D", N_BARS_D)
            time.sleep(0.4)
            h_df = pull(auth, tk, "60", n_1h)
            if d_df.empty or h_df.empty:
                print(f"  {tk:6s}  EMPTY  (D={len(d_df)} 1h={len(h_df)})")
                continue
            daily_d_frames.append(to_daily_d(d_df))
            bars_1h_frames.append(h_df)
            print(f"  {tk:6s}  D={len(d_df):3d}  1h={len(h_df):4d}")
        except Exception as e:
            print(f"  {tk:6s}  FAIL  {type(e).__name__}: {str(e)[:80]}")
        time.sleep(0.4)
    print(f"  pulled in {time.time()-t0:.1f}s")
    print()

    if not daily_d_frames:
        print("  no daily-D bars retrieved — abort")
        return 1
    daily_d = pd.concat(daily_d_frames, ignore_index=True)
    bars_1h = pd.concat(bars_1h_frames, ignore_index=True)
    daily_1h = resample_1h_to_daily(bars_1h)
    print(f"  daily_d rows : {len(daily_d):,}")
    print(f"  daily_1h rows: {len(daily_1h):,}")
    print()

    # join
    j = daily_d.merge(
        daily_1h.rename(columns={
            "open": "open_1h", "high": "high_1h", "low": "low_1h",
            "close": "close_1h", "volume": "volume_1h", "n_bars": "n_bars_1h",
        }),
        on=["ticker", "date"], how="inner",
    )
    j["close_diff_pct"] = (j["close"] - j["close_1h"]) / j["close_1h"] * 100
    j["vol_ratio"] = j["volume"] / j["volume_1h"].replace(0, pd.NA)
    j = j.sort_values(["ticker", "date"]).reset_index(drop=True)
    OUT_DETAIL.parent.mkdir(parents=True, exist_ok=True)
    j.to_csv(OUT_DETAIL, index=False, float_format="%.6f")
    print(f"  ✓ detail   → {OUT_DETAIL}  ({len(j):,} matched rows)")

    summary_rows = []
    for tk, g in j.groupby("ticker"):
        d_only = set(daily_d[daily_d.ticker == tk].date)
        h_only = set(daily_1h[daily_1h.ticker == tk].date)
        summary_rows.append({
            "ticker": tk,
            "n_d": len(d_only),
            "n_1h": len(h_only),
            "n_match": len(g),
            "missing_in_1h": len(d_only - h_only),
            "missing_in_d": len(h_only - d_only),
            "median_close_diff_pct": float(g.close_diff_pct.abs().median()),
            "p95_close_diff_pct":    float(g.close_diff_pct.abs().quantile(0.95)),
            "median_vol_ratio":      float(g.vol_ratio.median()),
            "p10_vol_ratio":         float(g.vol_ratio.quantile(0.10)),
            "p90_vol_ratio":         float(g.vol_ratio.quantile(0.90)),
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT_SUMMARY, index=False, float_format="%.4f")
    print(f"  ✓ summary  → {OUT_SUMMARY}")
    print()
    print(summary.to_string(index=False))
    print()

    # Verdict
    med_diff = summary["median_close_diff_pct"].median()
    med_vol = summary["median_vol_ratio"].median()
    p95_diff = summary["p95_close_diff_pct"].max()
    print(f"=== VERDICT ===")
    print(f"  cross-ticker median |close diff|: {med_diff:.4f}%")
    print(f"  worst-ticker p95   |close diff|: {p95_diff:.4f}%")
    print(f"  cross-ticker median vol ratio  : {med_vol:.4f}")
    pass_close = med_diff < 0.10 and p95_diff < 0.50
    pass_vol = 0.95 <= med_vol <= 1.05
    if pass_close and pass_vol:
        print("  VERDICT: PASS — extfeed-D and 1h-resampled match within tolerance.")
        print("           Safe to proceed with daily-D in screener_combo live system.")
        return 0
    print("  VERDICT: REVIEW — diff exceeds tolerance.")
    if not pass_close:
        print("           close diff too large.")
    if not pass_vol:
        print("           volume ratio diverged.")
    print("           Consider B (1h pull + local resample) for live system.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
