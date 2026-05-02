"""Round 3: lock the hybrid mix — ratio sweep, directional CMF, ema smoothing."""
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


def _sma(s, n):
    return s.rolling(n, min_periods=n).mean()


def _ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def _rma(s, n):
    return s.ewm(alpha=1.0 / n, adjust=False).mean()


def _bp(df):
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    return (2 * df["close"] - df["high"] - df["low"]) / rng


def _cmf_scaled(df):
    bp = _bp(df)
    cmf = (bp * df["volume"]).rolling(LEN, min_periods=LEN).sum() / df["volume"].rolling(LEN, min_periods=LEN).sum()
    return 50 + 50 * cmf


def _mfi(df):
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    raw = tp * df["volume"]
    diff = tp.diff()
    pos = raw.where(diff > 0, 0.0)
    neg = raw.where(diff < 0, 0.0)
    pos_sum = pos.rolling(LEN, min_periods=LEN).sum()
    neg_sum = neg.rolling(LEN, min_periods=LEN).sum()
    mfr = pos_sum / neg_sum.replace(0, np.nan)
    return 100.0 - 100.0 / (1.0 + mfr)


def _directional_cmf(df):
    """CMF gated by typical price direction (up bars only contribute pos, etc)."""
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    diff = tp.diff()
    bp = _bp(df)
    weighted = bp * df["volume"]
    pos = weighted.where(diff > 0, 0.0)
    neg = weighted.where(diff < 0, 0.0)
    s_pos = pos.rolling(LEN, min_periods=LEN).sum()
    s_neg = neg.rolling(LEN, min_periods=LEN).sum()
    s_all = df["volume"].rolling(LEN, min_periods=LEN).sum()
    cmf = (s_pos + s_neg) / s_all.replace(0, np.nan)
    return 50 + 50 * cmf


def _mix(a, b, w):
    return w * a + (1 - w) * b


def _evaluate(name, ours, truth, rows):
    merged = pd.DataFrame({"a": truth, "b": ours}).dropna()
    if merged.empty:
        rows.append({"candidate": name, "n": 0})
        return
    d = merged["a"] - merged["b"]
    rows.append(
        {
            "candidate": name,
            "n": len(merged),
            "rmse": float((d**2).mean() ** 0.5),
            "max_abs": float(d.abs().max()),
            "bias": float(d.mean()),
            "corr": float(merged["a"].corr(merged["b"])),
        }
    )


def main() -> None:
    tv = load_tv_csv(CSV_PATH).set_index("ts")
    truth = tv["Money Flow"]
    ohlcv = to_ohlcv(tv.reset_index())

    mfi_raw = _mfi(ohlcv)
    cmf_raw = _cmf_scaled(ohlcv)
    dir_cmf = _directional_cmf(ohlcv)

    rows: list[dict] = []

    # mix ratio sweep on raw MFI + raw CMF, then SMA(smooth)
    for w in [0.3, 0.4, 0.5, 0.6, 0.7]:
        _evaluate(f"mix(mfi×{w:.1f}+cmf)_smaSMOOTH", _sma(_mix(mfi_raw, cmf_raw, w), SMOOTH), truth, rows)

    # mix with EMA outer smoothing
    for w in [0.4, 0.5, 0.6]:
        _evaluate(f"mix(mfi×{w:.1f}+cmf)_emaSMOOTH", _ema(_mix(mfi_raw, cmf_raw, w), SMOOTH), truth, rows)

    # directional CMF alone, sma
    _evaluate("directional_cmf_sma", _sma(dir_cmf, SMOOTH), truth, rows)
    _evaluate("directional_cmf_ema", _ema(dir_cmf, SMOOTH), truth, rows)

    # mix with directional cmf
    for w in [0.4, 0.5, 0.6]:
        _evaluate(f"mix(mfi×{w:.1f}+dir_cmf)_sma", _sma(_mix(mfi_raw, dir_cmf, w), SMOOTH), truth, rows)

    # MFI × CMF average rebalanced
    _evaluate("mix3(mfi+cmf+dir_cmf)/3_sma", _sma((mfi_raw + cmf_raw + dir_cmf) / 3, SMOOTH), truth, rows)

    out = pd.DataFrame(rows).sort_values("rmse")
    pd.set_option("display.float_format", lambda x: f"{x:,.4f}")
    print("=== MF round-3 sweep ===")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
