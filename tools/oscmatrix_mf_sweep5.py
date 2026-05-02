"""Round 5: structural candidates — WaveTrend, KAMA, TSI, KVO, T3-smoothed."""
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


def cand_wavetrend(df):
    """WaveTrend money-flow variant: vol-weighted typical price oscillation."""
    ap = (df["high"] + df["low"] + df["close"]) / 3.0
    # vol weight inside the WT
    esa = _ema(ap, LEN)
    d = _ema((ap - esa).abs(), LEN)
    ci = (ap - esa) / (0.015 * d.replace(0, np.nan))
    wt1 = _ema(ci, SMOOTH)
    # rescale to 0-100 around 50
    return 50 + 50 * np.tanh(wt1 / 100.0)


def cand_kama_cmf(df):
    """KAMA-smoothed CMF."""
    bp = _bp(df)
    cmf_raw = (bp * df["volume"]).rolling(LEN, min_periods=LEN).sum() / df["volume"].rolling(LEN, min_periods=LEN).sum()
    mf = 50 + 50 * cmf_raw

    # KAMA smoothing
    n = SMOOTH
    er_window = LEN
    change = mf.diff(er_window).abs()
    volatility = mf.diff().abs().rolling(er_window).sum()
    er = change / volatility.replace(0, np.nan)
    fast = 2 / (2 + 1)
    slow = 2 / (30 + 1)
    sc = (er * (fast - slow) + slow) ** 2

    out = mf.copy() * np.nan
    last = None
    for i in range(len(mf)):
        x = mf.iloc[i]
        s = sc.iloc[i]
        if pd.isna(x):
            continue
        if last is None or pd.isna(last):
            out.iloc[i] = x
            last = x
            continue
        if pd.isna(s):
            out.iloc[i] = last
            continue
        last = last + s * (x - last)
        out.iloc[i] = last
    return out


def cand_tsi_volume(df):
    """TSI applied to (close * sign change)*volume."""
    src = df["close"] * df["volume"]
    diff = src.diff()
    dsm = _ema(_ema(diff, LEN), SMOOTH)
    dsam = _ema(_ema(diff.abs(), LEN), SMOOTH)
    tsi = 100 * dsm / dsam.replace(0, np.nan)
    return 50 + 0.5 * tsi


def cand_double_ema_cmf(df):
    """CMF, then EMA(LEN) detrend, recentered at 50."""
    bp = _bp(df)
    cmf_raw = (bp * df["volume"]).rolling(LEN, min_periods=LEN).sum() / df["volume"].rolling(LEN, min_periods=LEN).sum()
    mf = 50 + 50 * cmf_raw
    detrend = mf - _ema(mf, LEN * 4) + 50
    return _sma(detrend, SMOOTH)


def cand_chaikin_osc(df):
    """Chaikin Oscillator: ema(ad, fast) - ema(ad, slow), rescaled."""
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    mfm = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / rng
    ad = (mfm * df["volume"]).cumsum()
    osc = _ema(ad, SMOOTH) - _ema(ad, LEN)
    # normalize by rolling vol
    norm = osc / df["volume"].rolling(LEN).mean().replace(0, np.nan)
    return 50 + 50 * np.tanh(norm)


def cand_bp_ema_only(df):
    """No rolling sum: just EMA(bp*v)/EMA(v)."""
    bp = _bp(df)
    num = _ema(bp * df["volume"], LEN)
    den = _ema(df["volume"], LEN)
    cmf = num / den.replace(0, np.nan)
    mf = 50 + 50 * cmf
    return _ema(mf, SMOOTH)


def cand_smoothed_bp_alt(df):
    """SMOOTH-period bp, then LEN-period vol-weight."""
    bp = _bp(df)
    bp_s = _sma(bp, SMOOTH)
    cmf = (bp_s * df["volume"]).rolling(LEN, min_periods=LEN).sum() / df["volume"].rolling(LEN, min_periods=LEN).sum()
    return 50 + 50 * cmf


def cand_klinger_proper(df):
    """Klinger Volume Oscillator (KVO)."""
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    trend = (tp.diff() > 0).astype(int) * 2 - 1
    dm = df["high"] - df["low"]
    cm = dm.copy()  # simplified — full KVO has cumulative tracking
    vf = df["volume"] * trend * np.abs(2 * dm / cm.replace(0, np.nan) - 1) * 100
    kvo = _ema(vf, 34) - _ema(vf, 55)
    norm = kvo / df["volume"].rolling(LEN).mean().replace(0, np.nan)
    return 50 + 50 * np.tanh(norm / 5)


def cand_relative_vigor(df):
    """RVI variant — close-open vs high-low, vol-weighted."""
    co = (df["close"] - df["open"])
    hl = (df["high"] - df["low"]).replace(0, np.nan)
    rvi_num = _ema(co * df["volume"], LEN)
    rvi_den = _ema(hl * df["volume"], LEN)
    rvi = rvi_num / rvi_den.replace(0, np.nan)
    return _sma(50 + 50 * rvi, SMOOTH)


CANDIDATES = {
    "wavetrend_tanh": cand_wavetrend,
    "kama_cmf": cand_kama_cmf,
    "tsi_volume": cand_tsi_volume,
    "double_ema_detrend_cmf": cand_double_ema_cmf,
    "chaikin_osc": cand_chaikin_osc,
    "bp_ema_only": cand_bp_ema_only,
    "smoothed_bp_alt": cand_smoothed_bp_alt,
    "klinger_proper": cand_klinger_proper,
    "relative_vigor": cand_relative_vigor,
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
    print("=== MF round-5 (structural) ===")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
