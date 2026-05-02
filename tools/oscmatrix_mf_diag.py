"""Linear-regression diagnostic: can we reach TV Money Flow with a linear combo of features?"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from oscmatrix.validate import load_tv_csv, to_ohlcv  # noqa: E402

CSV_PATH = (
    "/Users/alpkarakasli/Library/Mobile Documents/com~apple~CloudDocs/"
    "Downloads1/BIST_THYAO, 1D.csv"
)
LEN = 35
SMOOTH = 6


def _sma(s, n): return s.rolling(n, min_periods=n).mean()
def _ema(s, n): return s.ewm(span=n, adjust=False).mean()
def _rma(s, n): return s.ewm(alpha=1.0 / n, adjust=False).mean()


def _bp(df):
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    return (2 * df["close"] - df["high"] - df["low"]) / rng


def _cmf_raw(df):
    bp = _bp(df)
    return (bp * df["volume"]).rolling(LEN, min_periods=LEN).sum() / df["volume"].rolling(LEN, min_periods=LEN).sum()


def _mfi_raw(df):
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    raw = tp * df["volume"]
    diff = tp.diff()
    pos = raw.where(diff > 0, 0.0)
    neg = raw.where(diff < 0, 0.0)
    pos_sum = pos.rolling(LEN, min_periods=LEN).sum()
    neg_sum = neg.rolling(LEN, min_periods=LEN).sum()
    mfr = pos_sum / neg_sum.replace(0, np.nan)
    return 100.0 - 100.0 / (1.0 + mfr)


def _rsi(close, n):
    diff = close.diff()
    up = diff.clip(lower=0)
    dn = (-diff).clip(lower=0)
    return 100 - 100 / (1 + _rma(up, n) / _rma(dn, n).replace(0, np.nan))


def main() -> None:
    tv = load_tv_csv(CSV_PATH).set_index("ts")
    truth = tv["Money Flow"]
    ohlcv = to_ohlcv(tv.reset_index())

    mfi = _sma(_mfi_raw(ohlcv), SMOOTH)
    cmf = _sma(50 + 50 * _cmf_raw(ohlcv), SMOOTH)
    rsi35 = _rsi(ohlcv["close"], LEN)
    rsi14 = _rsi(ohlcv["close"], 14)
    ema35_dist = 100 * (ohlcv["close"] / _ema(ohlcv["close"], LEN) - 1)
    sma35_dist = 100 * (ohlcv["close"] / _sma(ohlcv["close"], LEN) - 1)
    rvol = ohlcv["volume"] / _ema(ohlcv["volume"], LEN)

    # roc-based
    roc14 = 100 * ohlcv["close"].pct_change(14)
    # raw (unsmoothed) cmf scaled
    cmf_raw = 50 + 50 * _cmf_raw(ohlcv)
    # raw mfi unsmoothed
    mfi_raw_v = _mfi_raw(ohlcv)

    feats = pd.DataFrame(
        {
            "mfi_smoothed": mfi,
            "cmf_smoothed": cmf,
            "mfi_raw": mfi_raw_v,
            "cmf_raw": cmf_raw,
            "rsi14": rsi14,
            "rsi35": rsi35,
            "ema35_dist_pct": ema35_dist,
            "sma35_dist_pct": sma35_dist,
            "rvol": rvol,
            "roc14": roc14,
        }
    )

    df = feats.copy()
    df["truth"] = truth
    df = df.dropna()
    print(f"N={len(df)} after dropna")

    X = df[[c for c in df.columns if c != "truth"]].values
    y = df["truth"].values
    X_ = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(X_, y, rcond=None)
    yhat = X_ @ beta
    resid = y - yhat
    ss_res = (resid**2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot
    rmse = (resid**2).mean() ** 0.5

    print(f"\nFull-feature OLS:  R²={r2:.6f}  RMSE={rmse:.4f}")
    print("Coefficients (intercept first):")
    names = ["intercept"] + [c for c in df.columns if c != "truth"]
    for n, b in zip(names, beta):
        print(f"  {n:<20} {b:>10.4f}")

    # Subset 1: just mfi + cmf (smoothed)
    cols = ["mfi_smoothed", "cmf_smoothed"]
    Xs = df[cols].values
    Xs_ = np.column_stack([np.ones(len(Xs)), Xs])
    bs, *_ = np.linalg.lstsq(Xs_, y, rcond=None)
    yh = Xs_ @ bs
    r2s = 1 - ((y - yh) ** 2).sum() / ss_tot
    rmses = ((y - yh) ** 2).mean() ** 0.5
    print(f"\n[mfi+cmf only] R²={r2s:.6f}  RMSE={rmses:.4f}")
    print(f"  intercept={bs[0]:.4f}  mfi_w={bs[1]:.4f}  cmf_w={bs[2]:.4f}  (sum_w={bs[1]+bs[2]:.4f})")

    # Subset 2: + RSI
    cols2 = ["mfi_smoothed", "cmf_smoothed", "rsi14"]
    Xs = df[cols2].values
    Xs_ = np.column_stack([np.ones(len(Xs)), Xs])
    bs, *_ = np.linalg.lstsq(Xs_, y, rcond=None)
    yh = Xs_ @ bs
    r2s = 1 - ((y - yh) ** 2).sum() / ss_tot
    rmses = ((y - yh) ** 2).mean() ** 0.5
    print(f"\n[mfi+cmf+rsi14] R²={r2s:.6f}  RMSE={rmses:.4f}")
    print(f"  coefs={dict(zip(['intercept']+cols2, [round(float(x),4) for x in bs]))}")

    # Subset 3: + RSI + EMA dist
    cols3 = ["mfi_smoothed", "cmf_smoothed", "rsi14", "ema35_dist_pct"]
    Xs = df[cols3].values
    Xs_ = np.column_stack([np.ones(len(Xs)), Xs])
    bs, *_ = np.linalg.lstsq(Xs_, y, rcond=None)
    yh = Xs_ @ bs
    r2s = 1 - ((y - yh) ** 2).sum() / ss_tot
    rmses = ((y - yh) ** 2).mean() ** 0.5
    print(f"\n[mfi+cmf+rsi14+ema35_dist] R²={r2s:.6f}  RMSE={rmses:.4f}")
    print(f"  coefs={dict(zip(['intercept']+cols3, [round(float(x),4) for x in bs]))}")


if __name__ == "__main__":
    main()
