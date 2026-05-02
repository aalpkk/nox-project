"""Round 2: refine MF formula — EMA/RMA smoothing, hybrids, double-smooth, etc."""
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


def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def _rma(s: pd.Series, n: int) -> pd.Series:
    """Wilder's RMA (TradingView ta.rma)."""
    return s.ewm(alpha=1.0 / n, adjust=False).mean()


def _sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def _bp(df: pd.DataFrame) -> pd.Series:
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    return (2 * df["close"] - df["high"] - df["low"]) / rng


def cmf_sma_sma(df: pd.DataFrame) -> pd.Series:
    bp = _bp(df)
    cmf = (bp * df["volume"]).rolling(LEN, min_periods=LEN).sum() / df["volume"].rolling(LEN, min_periods=LEN).sum()
    return _sma(50 + 50 * cmf, SMOOTH)


def cmf_ema_outer(df: pd.DataFrame) -> pd.Series:
    bp = _bp(df)
    cmf = (bp * df["volume"]).rolling(LEN, min_periods=LEN).sum() / df["volume"].rolling(LEN, min_periods=LEN).sum()
    return _ema(50 + 50 * cmf, SMOOTH)


def cmf_rma_outer(df: pd.DataFrame) -> pd.Series:
    bp = _bp(df)
    cmf = (bp * df["volume"]).rolling(LEN, min_periods=LEN).sum() / df["volume"].rolling(LEN, min_periods=LEN).sum()
    return _rma(50 + 50 * cmf, SMOOTH)


def cmf_ema_inner(df: pd.DataFrame) -> pd.Series:
    """EMA-based CMF instead of rolling sum."""
    bp = _bp(df)
    num = _ema(bp * df["volume"], LEN)
    den = _ema(df["volume"], LEN)
    cmf = num / den.replace(0, np.nan)
    return _sma(50 + 50 * cmf, SMOOTH)


def cmf_rma_inner(df: pd.DataFrame) -> pd.Series:
    bp = _bp(df)
    num = _rma(bp * df["volume"], LEN)
    den = _rma(df["volume"], LEN)
    cmf = num / den.replace(0, np.nan)
    return _sma(50 + 50 * cmf, SMOOTH)


def mfi_sma(df: pd.DataFrame) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    raw = tp * df["volume"]
    diff = tp.diff()
    pos = raw.where(diff > 0, 0.0)
    neg = raw.where(diff < 0, 0.0)
    pos_sum = pos.rolling(LEN, min_periods=LEN).sum()
    neg_sum = neg.rolling(LEN, min_periods=LEN).sum()
    mfr = pos_sum / neg_sum.replace(0, np.nan)
    mfi = 100.0 - 100.0 / (1.0 + mfr)
    return _sma(mfi, SMOOTH)


def mfi_rma(df: pd.DataFrame) -> pd.Series:
    """RMA-based MFI (Wilder style)."""
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    raw = tp * df["volume"]
    diff = tp.diff()
    pos = raw.where(diff > 0, 0.0)
    neg = raw.where(diff < 0, 0.0)
    pos_rma = _rma(pos, LEN)
    neg_rma = _rma(neg, LEN)
    mfr = pos_rma / neg_rma.replace(0, np.nan)
    mfi = 100.0 - 100.0 / (1.0 + mfr)
    return _sma(mfi, SMOOTH)


def hybrid_avg(df: pd.DataFrame) -> pd.Series:
    return 0.5 * (mfi_sma(df) + cmf_sma_sma(df))


def hybrid_rma(df: pd.DataFrame) -> pd.Series:
    return 0.5 * (mfi_rma(df) + cmf_sma_sma(df))


def cmf_double_smooth(df: pd.DataFrame) -> pd.Series:
    """Pre-smooth bp before vol-weight."""
    bp = _bp(df).rolling(SMOOTH, min_periods=SMOOTH).mean()
    cmf = (bp * df["volume"]).rolling(LEN, min_periods=LEN).sum() / df["volume"].rolling(LEN, min_periods=LEN).sum()
    return 50 + 50 * cmf


def rsi_volume(df: pd.DataFrame) -> pd.Series:
    """RSI on (close * volume) instead of close — sometimes used as money-flow proxy."""
    src = df["close"] * df["volume"]
    diff = src.diff()
    up = diff.clip(lower=0)
    dn = (-diff).clip(lower=0)
    up_r = _rma(up, LEN)
    dn_r = _rma(dn, LEN)
    rs = up_r / dn_r.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    return _sma(rsi, SMOOTH)


CANDIDATES = {
    "cmf_sma_sma_(prev best)": cmf_sma_sma,
    "cmf_ema_outer": cmf_ema_outer,
    "cmf_rma_outer": cmf_rma_outer,
    "cmf_ema_inner": cmf_ema_inner,
    "cmf_rma_inner": cmf_rma_inner,
    "mfi_sma_(prev)": mfi_sma,
    "mfi_rma": mfi_rma,
    "hybrid_avg(mfi+cmf)": hybrid_avg,
    "hybrid_rma": hybrid_rma,
    "cmf_double_smooth": cmf_double_smooth,
    "rsi_volume": rsi_volume,
}


def main() -> None:
    tv = load_tv_csv(CSV_PATH).set_index("ts")
    truth = tv["Money Flow"]
    ohlcv = to_ohlcv(tv.reset_index())

    rows = []
    for name, fn in CANDIDATES.items():
        try:
            ours = fn(ohlcv)
        except Exception as e:
            rows.append({"candidate": name, "n": 0, "note": f"err: {e}"})
            continue
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
                "bias": float(d.mean()),
                "corr": float(merged["a"].corr(merged["b"])),
            }
        )

    out = pd.DataFrame(rows).sort_values("rmse")
    pd.set_option("display.float_format", lambda x: f"{x:,.4f}")
    print("=== MF round-2 sweep ===")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
