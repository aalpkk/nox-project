"""Threshold formula sweep — percentile/conditional/extrema-based candidates."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from oscmatrix import DEFAULT_PARAMS, compute_all  # noqa: E402
from oscmatrix.validate import load_tv_csv, to_ohlcv  # noqa: E402

CSV_DIR = "/Users/alpkarakasli/Library/Mobile Documents/com~apple~CloudDocs/Downloads1"
CSVS = {
    "THYAO": f"{CSV_DIR}/BIST_THYAO, 1D.csv",
    "GARAN": f"{CSV_DIR}/New Folder With Items 3/BIST_GARAN, 1D.csv",
    "EREGL": f"{CSV_DIR}/New Folder With Items 3/BIST_EREGL, 1D.csv",
    "ASELS": f"{CSV_DIR}/New Folder With Items 3/BIST_ASELS, 1D.csv",
}

LEN = 35
SMOOTH = 6


def _ema(s, n): return s.ewm(span=n, adjust=False).mean()
def _sma(s, n): return s.rolling(n, min_periods=n).mean()


def thr_percentile(mf, p_up=0.80, p_lo=0.20, win=LEN):
    upper = mf.rolling(win, min_periods=win).quantile(p_up)
    lower = mf.rolling(win, min_periods=win).quantile(p_lo)
    return upper, lower


def thr_minmax(mf, win=LEN):
    return mf.rolling(win, min_periods=win).max(), mf.rolling(win, min_periods=win).min()


def thr_sma_stdev(mf, win=LEN, k=1.0):
    sma = _sma(mf, win)
    std = mf.rolling(win, min_periods=win).std()
    return sma + k * std, sma - k * std


def thr_conditional_ema(mf, win=LEN):
    """EMA of MF only when MF>50 (upper) and MF<50 (lower)."""
    above = mf.where(mf > 50)
    below = mf.where(mf < 50)
    upper = above.ewm(span=win, ignore_na=True, adjust=False).mean()
    lower = below.ewm(span=win, ignore_na=True, adjust=False).mean()
    upper = upper.fillna(50.0)
    lower = lower.fillna(50.0)
    return upper, lower


def thr_centerline_offset(mf, win=LEN):
    """50 + EMA(max(MF-50, 0), win), 50 - EMA(max(50-MF, 0), win)."""
    bull_excess = (mf - 50).clip(lower=0)
    bear_excess = (50 - mf).clip(lower=0)
    upper = 50 + _ema(bull_excess, win)
    lower = 50 - _ema(bear_excess, win)
    return upper, lower


def thr_centerline_offset_sma(mf, win=LEN):
    bull_excess = (mf - 50).clip(lower=0)
    bear_excess = (50 - mf).clip(lower=0)
    upper = 50 + _sma(bull_excess, win)
    lower = 50 - _sma(bear_excess, win)
    return upper, lower


def thr_long_ema(mf, win=LEN * 4):
    """Long-period mean ± a constant."""
    base = _ema(mf, win)
    return base + 8, base - 8


def thr_rolling_stdev_long(mf, win=LEN * 2):
    sma = _sma(mf, win)
    std = mf.rolling(win, min_periods=win).std()
    return sma + std, sma - std


CANDIDATES = {
    "percentile_p70_p30": lambda mf: thr_percentile(mf, 0.70, 0.30, LEN),
    "percentile_p80_p20": lambda mf: thr_percentile(mf, 0.80, 0.20, LEN),
    "percentile_p75_p25_long": lambda mf: thr_percentile(mf, 0.75, 0.25, LEN * 2),
    "percentile_p65_p35": lambda mf: thr_percentile(mf, 0.65, 0.35, LEN),
    "minmax_LEN": lambda mf: thr_minmax(mf, LEN),
    "minmax_LEN2": lambda mf: thr_minmax(mf, LEN // 2),
    "sma_stdev_k1": lambda mf: thr_sma_stdev(mf, LEN, 1.0),
    "sma_stdev_k0.5": lambda mf: thr_sma_stdev(mf, LEN, 0.5),
    "conditional_ema": lambda mf: thr_conditional_ema(mf, LEN),
    "centerline_offset_ema": lambda mf: thr_centerline_offset(mf, LEN),
    "centerline_offset_sma": lambda mf: thr_centerline_offset_sma(mf, LEN),
    "long_ema_const8": lambda mf: thr_long_ema(mf, LEN * 4),
    "stdev_long2x": lambda mf: thr_rolling_stdev_long(mf, LEN * 2),
}


def evaluate(name, fn):
    rows = []
    for ticker, path in CSVS.items():
        tv = load_tv_csv(path).set_index("ts")
        ohlcv = to_ohlcv(tv.reset_index())
        ours = compute_all(ohlcv, DEFAULT_PARAMS)
        mf = ours["money_flow"]
        try:
            upper, lower = fn(mf)
        except Exception as e:
            rows.append({"name": name, "ticker": ticker, "err": str(e)[:40]})
            continue

        for tag, our_series, tv_col in [
            ("upper", upper, "Upper Money Flow Threshold"),
            ("lower", lower, "Lower Money Flow Threshold"),
        ]:
            merged = pd.DataFrame({"a": tv[tv_col], "b": our_series}).dropna()
            if merged.empty:
                continue
            d = merged["a"] - merged["b"]
            rows.append(
                {
                    "name": name,
                    "tag": tag,
                    "ticker": ticker,
                    "n": len(merged),
                    "rmse": float((d**2).mean() ** 0.5),
                    "bias": float(d.mean()),
                    "corr": float(merged["a"].corr(merged["b"])),
                }
            )
    return rows


def main() -> None:
    all_rows = []
    for name, fn in CANDIDATES.items():
        all_rows.extend(evaluate(name, fn))
    df = pd.DataFrame(all_rows)
    pd.set_option("display.float_format", lambda x: f"{x:,.4f}")

    agg = df.groupby(["name", "tag"]).agg(
        avg_rmse=("rmse", "mean"),
        avg_corr=("corr", "mean"),
        max_rmse=("rmse", "max"),
        avg_bias=("bias", "mean"),
    ).reset_index()
    pivot_rmse = agg.pivot(index="name", columns="tag", values="avg_rmse")
    pivot_corr = agg.pivot(index="name", columns="tag", values="avg_corr")
    out = pd.concat([pivot_rmse.add_suffix("_rmse"), pivot_corr.add_suffix("_corr")], axis=1)
    out["combined_rmse"] = (pivot_rmse["upper"] + pivot_rmse["lower"]) / 2
    out["combined_corr"] = (pivot_corr["upper"] + pivot_corr["lower"]) / 2
    out = out.sort_values("combined_rmse")
    print("=== Threshold candidates (avg across 4 tickers) ===")
    print(out.to_string())


if __name__ == "__main__":
    main()
