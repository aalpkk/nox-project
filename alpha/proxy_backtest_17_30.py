"""
nyxalpha 17:30 entry proxy backtest.

Takes existing trade log (close-entry) and re-prices every entry at T-day 17:30
bar close from Matriks 15m intraday cache. Exit logic (TP_50/TRAIL/STOP/MAX_HOLD)
and exit prices are held constant — first-order rescale.

Inputs:
  output/alpha_trades.csv                     — close-entry trades (317 rows, 207 unique entries)
  output/nyxexp_intraday_15m_matriks.parquet  — 15m bars cache (populated for nyxalpha dates via fetch_matriks_15m.py)

Output:
  output/nyxalpha_proxy_17_30_trades.csv
  output/nyxalpha_proxy_17_30_summary.csv

17:30 bar convention: TR 17:15 bar closes at 17:30 (bar_ts hh=17, mm=15).
                     TR 17:45 bar closes at 18:00 (hh=17, mm=45) — sanity check.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


TRADES_PATH = Path("output/alpha_trades.csv")
INTRADAY_PATH = Path("output/nyxexp_intraday_15m_matriks.parquet")
OUT_TRADES = Path("output/nyxalpha_proxy_17_30_trades.csv")
OUT_SUMMARY = Path("output/nyxalpha_proxy_17_30_summary.csv")


def _equity_maxdd(returns: pd.Series) -> float:
    if returns.empty:
        return 0.0
    eq = (1.0 + returns.fillna(0.0)).cumprod()
    peak = eq.cummax()
    dd = (eq - peak) / peak
    return float(dd.min() * 100)


def _agg(name: str, r: pd.Series) -> dict:
    r = r.dropna()
    if r.empty:
        return {"scenario": name, "N": 0, "WR%": 0, "avg%": 0, "median%": 0, "PF": 0, "total%": 0, "MaxDD%": 0}
    wins = (r > 0).sum()
    wr = wins / len(r) * 100
    wins_sum = r[r > 0].sum()
    loss_sum = -r[r < 0].sum()
    pf = (wins_sum / loss_sum) if loss_sum > 0 else float("inf")
    total_pct = (np.prod(1 + r) - 1) * 100
    return {
        "scenario": name,
        "N": len(r),
        "WR%": round(wr, 1),
        "avg%": round(r.mean() * 100, 2),
        "median%": round(r.median() * 100, 2),
        "PF": round(pf, 2) if np.isfinite(pf) else "inf",
        "total%": round(total_pct, 1),
        "MaxDD%": round(_equity_maxdd(r), 1),
    }


def main() -> int:
    trades = pd.read_csv(TRADES_PATH)
    trades["entry_date"] = pd.to_datetime(trades["entry_date"]).dt.date
    trades["exit_date"] = pd.to_datetime(trades["exit_date"]).dt.date
    print(f"nyxalpha trades: {len(trades)} rows ({trades[['ticker','entry_date']].drop_duplicates().shape[0]} unique entries)")

    intraday = pd.read_parquet(INTRADAY_PATH)
    intraday["ts_utc"] = pd.to_datetime(intraday["bar_ts"], unit="ms", utc=True)
    intraday["ts_tr"] = intraday["ts_utc"].dt.tz_convert("Europe/Istanbul")
    intraday["hh"] = intraday["ts_tr"].dt.hour
    intraday["mm"] = intraday["ts_tr"].dt.minute
    intraday["signal_date"] = pd.to_datetime(intraday["signal_date"]).dt.date
    print(f"Intraday cache: {len(intraday)} bars, {intraday[['ticker','signal_date']].drop_duplicates().shape[0]} signal-days")

    # 17:15 bar closes AT 17:30
    bar_17_30 = (intraday[(intraday.hh == 17) & (intraday.mm == 15)]
                 .rename(columns={"close": "price_17_30", "signal_date": "entry_date"})
                 [["ticker", "entry_date", "price_17_30"]])
    # 17:45 bar closes AT 18:00 (sanity cross-check vs entry_price)
    bar_18_00 = (intraday[(intraday.hh == 17) & (intraday.mm == 45)]
                 .rename(columns={"close": "price_18_00", "signal_date": "entry_date"})
                 [["ticker", "entry_date", "price_18_00"]])

    df = trades.merge(bar_17_30, on=["ticker", "entry_date"], how="left")
    df = df.merge(bar_18_00, on=["ticker", "entry_date"], how="left")

    # Coverage
    n_total = len(df)
    n_uniq = df[["ticker", "entry_date"]].drop_duplicates().shape[0]
    u_df = df.drop_duplicates(["ticker", "entry_date"])
    n_17_30 = u_df["price_17_30"].notna().sum()
    n_18_00 = u_df["price_18_00"].notna().sum()
    n_both = (u_df["price_17_30"].notna() & u_df["price_18_00"].notna()).sum()
    print(f"\nCoverage (unique entries = {n_uniq}):")
    print(f"  with 17:30 bar: {n_17_30} ({n_17_30/n_uniq*100:.1f}%)")
    print(f"  with 18:00 bar: {n_18_00} ({n_18_00/n_uniq*100:.1f}%)")
    print(f"  with both (proxy usable): {n_both} ({n_both/n_uniq*100:.1f}%)")

    # Sanity drift (should be small most of the time — 18:00 close vs 17:30 close)
    u_df = u_df.copy()
    u_df["drift_30m_pct"] = (u_df["price_18_00"] - u_df["price_17_30"]) / u_df["price_17_30"] * 100
    dr = u_df["drift_30m_pct"].dropna()
    if not dr.empty:
        print(f"\n18:00 vs 17:30 drift: mean={dr.mean():+.2f}% median={dr.median():+.2f}% p10={dr.quantile(.1):+.2f}% p90={dr.quantile(.9):+.2f}%")

    # Rescale each trade PnL
    # orig entry_price ≈ signal_close × (1+slippage) ; exit_price already net of slippage
    # pnl_pct_orig = (exit_price / entry_price - 1) * 100
    # new pnl: use price_17_30 as the "target" entry. Add same slippage factor.
    SLIPPAGE = 0.001
    df["new_entry_price"] = df["price_17_30"] * (1 + SLIPPAGE)
    df["new_pnl_pct"] = (df["exit_price"] / df["new_entry_price"] - 1) * 100
    df.loc[df["price_17_30"].isna(), "new_pnl_pct"] = np.nan

    # Orig pnl as %
    df["orig_pnl_pct"] = df["pnl_pct"].astype(float)
    # Weighted return (same weight applies to both entries since sizing is the same)
    df["orig_contrib_pct"] = df["orig_pnl_pct"] * df["weight"]
    df["new_contrib_pct"] = df["new_pnl_pct"] * df["weight"]

    # Aggregate at trade level (raw pnl_pct, unweighted)
    all_orig = df["orig_pnl_pct"] / 100
    all_new = df["new_pnl_pct"] / 100
    same_set_orig = df.loc[df["new_pnl_pct"].notna(), "orig_pnl_pct"] / 100
    same_set_new = df.loc[df["new_pnl_pct"].notna(), "new_pnl_pct"] / 100

    summ_rows = [
        _agg("close-entry (orig, all 317 trade rows)", all_orig),
        _agg("close-entry (orig, only-covered subset)", same_set_orig),
        _agg("17:30 proxy (only-covered subset)", same_set_new),
    ]
    summary = pd.DataFrame(summ_rows)
    print("\n" + "=" * 80)
    print("SUMMARY (trade-row level, unweighted)")
    print("=" * 80)
    print(summary.to_string(index=False))

    # Per-ticker aggregate position level (fold half-exits back into single position)
    # One position = one (ticker, entry_date); its realized pnl is weight-summed across dilim rows.
    pos_orig = df.groupby(["ticker", "entry_date"], as_index=False).agg(
        orig_contrib_pct=("orig_contrib_pct", "sum"),
        new_contrib_pct=("new_contrib_pct", "sum"),
        weight_sum=("weight", "sum"),
        any_nan=("new_pnl_pct", lambda s: s.isna().any()),
    )
    # Position-level pnl_pct = contrib / total_weight (since weight-summed contrib already = pnl-weighted)
    pos_orig["orig_pos_pnl_pct"] = pos_orig["orig_contrib_pct"] / pos_orig["weight_sum"]
    pos_orig["new_pos_pnl_pct"] = pos_orig["new_contrib_pct"] / pos_orig["weight_sum"]
    pos_orig.loc[pos_orig["any_nan"], "new_pos_pnl_pct"] = np.nan

    po_orig = pos_orig["orig_pos_pnl_pct"] / 100
    po_new = pos_orig["new_pos_pnl_pct"] / 100
    po_same_orig = pos_orig.loc[pos_orig["new_pos_pnl_pct"].notna(), "orig_pos_pnl_pct"] / 100
    po_same_new = pos_orig.loc[pos_orig["new_pos_pnl_pct"].notna(), "new_pos_pnl_pct"] / 100

    pos_rows = [
        _agg("close-entry (position-level, all)", po_orig),
        _agg("close-entry (position-level, only-covered)", po_same_orig),
        _agg("17:30 proxy (position-level, only-covered)", po_same_new),
    ]
    pos_summ = pd.DataFrame(pos_rows)
    print("\n" + "=" * 80)
    print("POSITION-LEVEL (half-exits rolled up)")
    print("=" * 80)
    print(pos_summ.to_string(index=False))

    # Save
    OUT_TRADES.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_TRADES, index=False)
    combined = pd.concat([summary, pos_summ], ignore_index=True)
    combined.to_csv(OUT_SUMMARY, index=False)
    print(f"\nWrote {OUT_TRADES}")
    print(f"Wrote {OUT_SUMMARY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
