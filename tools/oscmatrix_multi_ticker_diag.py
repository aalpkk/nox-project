"""Cross-ticker validation: is the hybrid stable, and is residual portable?"""
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

TARGET_COLS = {
    "Money Flow": "money_flow",
    "Upper Money Flow Threshold": "upper_threshold",
    "Lower Money Flow Threshold": "lower_threshold",
    "Bullish Overflow": "bullish_overflow",
    "Bearish Overflow": "bearish_overflow",
    "HyperWave": "hyperwave",
}


def load_pair(path: str):
    tv = load_tv_csv(path).set_index("ts")
    ohlcv = to_ohlcv(tv.reset_index())
    ours = compute_all(ohlcv, DEFAULT_PARAMS)
    return tv, ours


def diff_summary(tv, ours, ticker):
    rows = []
    for tv_col, our_col in TARGET_COLS.items():
        if tv_col not in tv.columns or our_col not in ours.columns:
            continue
        merged = pd.DataFrame({"a": tv[tv_col], "b": ours[our_col]}).dropna()
        if merged.empty:
            continue
        d = merged["a"] - merged["b"]
        rows.append(
            {
                "ticker": ticker,
                "component": tv_col,
                "n": len(merged),
                "rmse": float((d**2).mean() ** 0.5),
                "max_abs": float(d.abs().max()),
                "bias": float(d.mean()),
                "corr": float(merged["a"].corr(merged["b"])),
            }
        )
    return rows


def main() -> None:
    all_rows = []
    residuals = {}
    for ticker, path in CSVS.items():
        tv, ours = load_pair(path)
        all_rows.extend(diff_summary(tv, ours, ticker))
        # store MF residual
        merged = pd.DataFrame({"truth": tv["Money Flow"], "ours": ours["money_flow"]}).dropna()
        residuals[ticker] = merged["truth"] - merged["ours"]

    out = pd.DataFrame(all_rows)
    pd.set_option("display.float_format", lambda x: f"{x:,.4f}")

    print("=== Per-ticker × per-component diff ===")
    print(out.to_string(index=False))

    print("\n=== Money Flow residual stats per ticker ===")
    for t, r in residuals.items():
        print(f"  {t:<6} n={len(r):3d}  mean_bias={r.mean():>+7.3f}  std={r.std():>6.3f}  range=[{r.min():>+7.2f},{r.max():>+7.2f}]  autocorr_lag1={r.autocorr(1):.3f}")

    print("\n=== Pivot view: Money Flow ===")
    pivot = out[out["component"] == "Money Flow"].set_index("ticker")[["n", "rmse", "max_abs", "bias", "corr"]]
    print(pivot.to_string())

    print("\n=== Pivot view: HyperWave ===")
    pivot = out[out["component"] == "HyperWave"].set_index("ticker")[["n", "rmse", "max_abs", "bias", "corr"]]
    print(pivot.to_string())

    # Check if residual is portable: pairwise corr of residuals on overlapping dates
    print("\n=== Pairwise residual cross-ticker correlation (dates aligned) ===")
    resid_df = pd.DataFrame(residuals).dropna(how="all")
    print(resid_df.corr().to_string())

    # Mean residual time series — if shape similar across tickers, residual is partially portable
    print("\n=== Mean MF residual by date (across tickers, shows shared signal) ===")
    mean_resid = resid_df.mean(axis=1)
    print(f"  shared mean shape: n={len(mean_resid.dropna())}  std={mean_resid.std():.3f}  range=[{mean_resid.min():+.2f},{mean_resid.max():+.2f}]")
    print(f"  per-ticker corr to shared mean:")
    for t in residuals:
        c = resid_df[t].corr(mean_resid)
        print(f"    {t}: {c:.3f}")


if __name__ == "__main__":
    main()
