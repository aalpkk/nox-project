"""HyperWave formula sweep across 4 tickers."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from oscmatrix.validate import load_tv_csv, to_ohlcv  # noqa: E402

CSV_DIR = "/Users/alpkarakasli/Library/Mobile Documents/com~apple~CloudDocs/Downloads1"
CSVS = {
    "THYAO": f"{CSV_DIR}/BIST_THYAO, 1D.csv",
    "GARAN": f"{CSV_DIR}/New Folder With Items 3/BIST_GARAN, 1D.csv",
    "EREGL": f"{CSV_DIR}/New Folder With Items 3/BIST_EREGL, 1D.csv",
    "ASELS": f"{CSV_DIR}/New Folder With Items 3/BIST_ASELS, 1D.csv",
}

HW_LEN = 7
SIG_LEN = 3
SMOOTH_3 = 3


def _sma(s, n): return s.rolling(n, min_periods=n).mean()
def _ema(s, n): return s.ewm(span=n, adjust=False).mean()
def _rma(s, n): return s.ewm(alpha=1.0 / n, adjust=False).mean()


def _rsi(close, n):
    diff = close.diff()
    up = diff.clip(lower=0)
    dn = (-diff).clip(lower=0)
    rs = _rma(up, n) / _rma(dn, n).replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def cand_stoch_raw(df):
    ll = df["low"].rolling(HW_LEN, min_periods=HW_LEN).min()
    hh = df["high"].rolling(HW_LEN, min_periods=HW_LEN).max()
    return 100 * (df["close"] - ll) / (hh - ll).replace(0, np.nan)


def cand_stoch_sma3(df):
    return _sma(cand_stoch_raw(df), SIG_LEN)


def cand_rsi(df):
    return _rsi(df["close"], HW_LEN)


def cand_rsi_sma(df):
    return _sma(_rsi(df["close"], HW_LEN), SIG_LEN)


def cand_stoch_rsi_k(df):
    rsi = _rsi(df["close"], HW_LEN)
    ll = rsi.rolling(HW_LEN, min_periods=HW_LEN).min()
    hh = rsi.rolling(HW_LEN, min_periods=HW_LEN).max()
    return 100 * (rsi - ll) / (hh - ll).replace(0, np.nan)


def cand_stoch_rsi_k_smoothed(df):
    return _sma(cand_stoch_rsi_k(df), SIG_LEN)


def cand_wt(df):
    """WaveTrend rescaled to 0-100."""
    ap = (df["high"] + df["low"] + df["close"]) / 3.0
    esa = _ema(ap, HW_LEN)
    d = _ema((ap - esa).abs(), HW_LEN)
    ci = (ap - esa) / (0.015 * d.replace(0, np.nan))
    wt = _ema(ci, SIG_LEN)
    # rescale via tanh squash
    return 50 + 25 * np.tanh(wt / 60.0)  # map ±60 → ±25


def cand_smoothed_stoch(df):
    """Stoch with extra ema smoothing."""
    ll = df["low"].rolling(HW_LEN, min_periods=HW_LEN).min()
    hh = df["high"].rolling(HW_LEN, min_periods=HW_LEN).max()
    raw = 100 * (df["close"] - ll) / (hh - ll).replace(0, np.nan)
    return _ema(raw, SIG_LEN)


def cand_stoch_close_to_close(df):
    """Stoch on close only (TV's ta.stoch with src=close)."""
    ll = df["close"].rolling(HW_LEN, min_periods=HW_LEN).min()
    hh = df["close"].rolling(HW_LEN, min_periods=HW_LEN).max()
    return 100 * (df["close"] - ll) / (hh - ll).replace(0, np.nan)


def cand_stoch_close_then_sma(df):
    return _sma(cand_stoch_close_to_close(df), SIG_LEN)


def cand_stoch_hlc3(df):
    """Stoch on hlc3."""
    src = (df["high"] + df["low"] + df["close"]) / 3.0
    ll = src.rolling(HW_LEN, min_periods=HW_LEN).min()
    hh = src.rolling(HW_LEN, min_periods=HW_LEN).max()
    return 100 * (src - ll) / (hh - ll).replace(0, np.nan)


def cand_smi_ergodic(df):
    """Stochastic Momentum Index (Ergodic): double-smoothed."""
    hh = df["high"].rolling(HW_LEN, min_periods=HW_LEN).max()
    ll = df["low"].rolling(HW_LEN, min_periods=HW_LEN).min()
    mid = (hh + ll) / 2.0
    rng = (hh - ll).replace(0, np.nan)
    diff = df["close"] - mid
    smi = 200 * _ema(_ema(diff, SIG_LEN), SIG_LEN) / _ema(_ema(rng, SIG_LEN), SIG_LEN)
    return 50 + smi / 2  # map ±100 → 0-100


CANDIDATES = {
    "stoch_raw": cand_stoch_raw,
    "stoch_sma3": cand_stoch_sma3,
    "rsi": cand_rsi,
    "rsi_sma": cand_rsi_sma,
    "stoch_rsi_k": cand_stoch_rsi_k,
    "stoch_rsi_k_smoothed": cand_stoch_rsi_k_smoothed,
    "wavetrend_tanh": cand_wt,
    "smoothed_stoch_ema": cand_smoothed_stoch,
    "stoch_close": cand_stoch_close_to_close,
    "stoch_close_sma": cand_stoch_close_then_sma,
    "stoch_hlc3": cand_stoch_hlc3,
    "smi_ergodic": cand_smi_ergodic,
}


def evaluate(name, fn):
    rows = []
    for ticker, path in CSVS.items():
        tv = load_tv_csv(path).set_index("ts")
        ohlcv = to_ohlcv(tv.reset_index())
        try:
            ours = fn(ohlcv)
        except Exception as e:
            rows.append({"name": name, "ticker": ticker, "err": str(e)[:40]})
            continue
        merged = pd.DataFrame({"a": tv["HyperWave"], "b": ours}).dropna()
        if merged.empty:
            continue
        d = merged["a"] - merged["b"]
        rows.append(
            {
                "name": name,
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

    # aggregate per candidate (avg across tickers)
    agg = df.groupby("name").agg(
        avg_rmse=("rmse", "mean"),
        avg_corr=("corr", "mean"),
        max_rmse=("rmse", "max"),
        min_rmse=("rmse", "min"),
        avg_bias=("bias", "mean"),
    ).sort_values("avg_rmse")
    print("=== HyperWave candidates (averaged across 4 tickers) ===")
    print(agg.to_string())

    # show best in detail
    best = agg.index[0]
    print(f"\n=== Detail for best: {best} ===")
    print(df[df["name"] == best].to_string(index=False))


if __name__ == "__main__":
    main()
