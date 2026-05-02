"""Round 6 (final): WaveTrend variants, vol-weighted typical, MACD-style flow."""
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


def _bp(df):
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    return (2 * df["close"] - df["high"] - df["low"]) / rng


def cand_wt_unscaled_div2(df):
    ap = (df["high"] + df["low"] + df["close"]) / 3.0
    esa = _ema(ap, LEN)
    d = _ema((ap - esa).abs(), LEN)
    ci = (ap - esa) / (0.015 * d.replace(0, np.nan))
    wt1 = _ema(ci, SMOOTH)
    return 50 + wt1 / 2.0


def cand_wt_volwgt_ap(df):
    """WaveTrend on volume-weighted AP (VWAP-typical)."""
    ap = (df["high"] + df["low"] + df["close"]) / 3.0
    vwap = (ap * df["volume"]).rolling(LEN).sum() / df["volume"].rolling(LEN).sum()
    esa = _ema(vwap, LEN)
    d = _ema((vwap - esa).abs(), LEN)
    ci = (vwap - esa) / (0.015 * d.replace(0, np.nan))
    wt1 = _ema(ci, SMOOTH)
    return 50 + wt1 / 2.0


def cand_macd_flow(df):
    """MACD-style on bp*volume normalized."""
    bp = _bp(df)
    flow = bp * df["volume"]
    fast = _ema(flow, SMOOTH)
    slow = _ema(flow, LEN)
    macd = fast - slow
    norm = _ema(df["volume"], LEN).replace(0, np.nan)
    sig = (macd / norm).clip(-1, 1)
    return 50 + 50 * sig


def cand_hybrid_with_long_detrend(df):
    """Best hybrid + ema(LEN*4) detrend."""
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    bp = (2 * df["close"] - df["high"] - df["low"]) / rng
    cmf_raw = (bp * df["volume"]).rolling(LEN).sum() / df["volume"].rolling(LEN).sum()
    cmf_s = _sma(50 + 50 * cmf_raw, SMOOTH)

    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    raw = tp * df["volume"]
    diff = tp.diff()
    pos_sum = raw.where(diff > 0, 0.0).rolling(LEN).sum()
    neg_sum = raw.where(diff < 0, 0.0).rolling(LEN).sum()
    mfr = pos_sum / neg_sum.replace(0, np.nan)
    mfi_s = _sma(100 - 100 / (1 + mfr), SMOOTH)

    h = 0.4 * mfi_s + 0.6 * cmf_s
    detrend = h - _ema(h, LEN * 4) + 50
    return detrend


def cand_macd_then_smooth(df):
    """Buying-pressure as ema(close, fast) - ema(close, slow), volume-weighted."""
    fast_close = _ema(df["close"], SMOOTH)
    slow_close = _ema(df["close"], LEN)
    macd_close = (fast_close / slow_close - 1)
    fast_vol = _ema(df["volume"], SMOOTH)
    slow_vol = _ema(df["volume"], LEN)
    rvol_macd = (fast_vol / slow_vol - 1)
    sig = (macd_close * 5 + rvol_macd * 2).clip(-1, 1)
    return 50 + 50 * sig


def cand_hlc3_macd(df):
    """ema-flow on hlc3 weighted by volume."""
    hlc3 = (df["high"] + df["low"] + df["close"]) / 3.0
    bp = _bp(df)
    flow = bp * df["volume"] * (hlc3 / hlc3.shift(1))
    fast = _ema(flow, SMOOTH)
    slow = _ema(flow, LEN)
    norm = _ema(df["volume"] * hlc3, LEN).replace(0, np.nan)
    return 50 + 50 * ((fast - slow) / norm).clip(-1, 1)


CANDIDATES = {
    "wt_unscaled_div2": cand_wt_unscaled_div2,
    "wt_volwgt_ap": cand_wt_volwgt_ap,
    "macd_flow": cand_macd_flow,
    "hybrid_long_detrend": cand_hybrid_with_long_detrend,
    "macd_close_then_smooth": cand_macd_then_smooth,
    "hlc3_macd": cand_hlc3_macd,
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
            rows.append({"candidate": name, "n": 0, "err": str(e)[:60]})
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
    print("=== MF round-6 (final structural) ===")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
