"""Sweep candidate Money Flow formulas vs TV ground truth."""
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

LENGTH = 35
SMOOTH = 6


def cand_mfi(df: pd.DataFrame) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    raw = tp * df["volume"]
    diff = tp.diff()
    pos = raw.where(diff > 0, 0.0)
    neg = raw.where(diff < 0, 0.0)
    pos_sum = pos.rolling(LENGTH, min_periods=LENGTH).sum()
    neg_sum = neg.rolling(LENGTH, min_periods=LENGTH).sum()
    mfr = pos_sum / neg_sum.replace(0, np.nan)
    mfi = 100.0 - 100.0 / (1.0 + mfr)
    return mfi.rolling(SMOOTH, min_periods=SMOOTH).mean()


def cand_cmf_rescaled(df: pd.DataFrame) -> pd.Series:
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    mfm = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / rng
    mfv = mfm * df["volume"]
    cmf = mfv.rolling(LENGTH, min_periods=LENGTH).sum() / df["volume"].rolling(LENGTH, min_periods=LENGTH).sum()
    mf = 50.0 + 50.0 * cmf
    return mf.rolling(SMOOTH, min_periods=SMOOTH).mean()


def cand_range_pos_vw(df: pd.DataFrame) -> pd.Series:
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    pos = (df["close"] - df["low"]) / rng
    vw = (pos * df["volume"]).rolling(LENGTH, min_periods=LENGTH).sum() / df["volume"].rolling(LENGTH, min_periods=LENGTH).sum()
    return (100.0 * vw).rolling(SMOOTH, min_periods=SMOOTH).mean()


def cand_klinger_like(df: pd.DataFrame) -> pd.Series:
    """Sign of typical price change × volume, EMA-smoothed, rescaled."""
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    sign = np.sign(tp.diff()).fillna(0)
    sv = sign * df["volume"]
    fast = sv.ewm(span=LENGTH, adjust=False).mean()
    vol_ema = df["volume"].ewm(span=LENGTH, adjust=False).mean()
    ratio = fast / vol_ema.replace(0, np.nan)
    mf = 50.0 + 50.0 * ratio.clip(-1, 1)
    return mf.rolling(SMOOTH, min_periods=SMOOTH).mean()


def cand_buy_press_vw(df: pd.DataFrame) -> pd.Series:
    """(2*close - high - low) / (high-low) range, vol-weighted, rescaled."""
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    bp = (2 * df["close"] - df["high"] - df["low"]) / rng  # -1..1
    vw = (bp * df["volume"]).rolling(LENGTH, min_periods=LENGTH).sum() / df["volume"].rolling(LENGTH, min_periods=LENGTH).sum()
    mf = 50.0 + 50.0 * vw
    return mf.rolling(SMOOTH, min_periods=SMOOTH).mean()


CANDIDATES = {
    "mfi": cand_mfi,
    "cmf_rescaled": cand_cmf_rescaled,
    "range_pos_vw": cand_range_pos_vw,
    "klinger_like": cand_klinger_like,
    "buy_press_vw": cand_buy_press_vw,
}


def main() -> None:
    tv = load_tv_csv(CSV_PATH).set_index("ts")
    truth = tv["Money Flow"]
    ohlcv = to_ohlcv(tv.reset_index())

    rows = []
    for name, fn in CANDIDATES.items():
        ours = fn(ohlcv)
        merged = pd.DataFrame({"a": truth, "b": ours}).dropna()
        if merged.empty:
            rows.append({"candidate": name, "n": 0})
            continue
        d = merged["a"] - merged["b"]
        rows.append(
            {
                "candidate": name,
                "n": len(merged),
                "rmse": float((d**2).mean() ** 0.5),
                "max_abs": float(d.abs().max()),
                "mean_bias": float(d.mean()),
                "corr": float(merged["a"].corr(merged["b"])),
            }
        )

    out = pd.DataFrame(rows).sort_values("rmse")
    pd.set_option("display.float_format", lambda x: f"{x:,.4f}")
    print("=== Money Flow candidate sweep (TV − ours) ===")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
