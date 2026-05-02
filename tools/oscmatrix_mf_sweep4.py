"""Round 4: residual analysis + stoch-normalized + rvol-weighted candidates."""
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


def _stoch_norm(s, n):
    lo = s.rolling(n, min_periods=n).min()
    hi = s.rolling(n, min_periods=n).max()
    return 100 * (s - lo) / (hi - lo).replace(0, np.nan)


def cand_stoch_cmf(df):
    cmf = _cmf_raw(df)  # -1..1
    return _sma(_stoch_norm(cmf, LEN), SMOOTH)


def cand_stoch_mfi(df):
    return _sma(_stoch_norm(_mfi_raw(df), LEN), SMOOTH)


def cand_stoch_hybrid(df):
    h = 0.5 * (_mfi_raw(df) + (50 + 50 * _cmf_raw(df)))
    return _sma(_stoch_norm(h, LEN), SMOOTH)


def cand_rvol_weighted(df):
    """CMF where volume is replaced by relative volume."""
    rvol = df["volume"] / _ema(df["volume"], LEN).replace(0, np.nan)
    bp = _bp(df)
    cmf = (bp * rvol).rolling(LEN, min_periods=LEN).sum() / rvol.rolling(LEN, min_periods=LEN).sum()
    return _sma(50 + 50 * cmf, SMOOTH)


def cand_dual_smooth(df):
    """Pre-smooth bp before vol-weighting + post-smooth."""
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    raw_bp = (2 * df["close"] - df["high"] - df["low"]) / rng
    bp = _ema(raw_bp, SMOOTH)
    cmf = (bp * df["volume"]).rolling(LEN, min_periods=LEN).sum() / df["volume"].rolling(LEN, min_periods=LEN).sum()
    return _ema(50 + 50 * cmf, SMOOTH)


def cand_alma(df):
    """ALMA smoothing on hybrid."""
    h = 0.5 * (_mfi_raw(df) + (50 + 50 * _cmf_raw(df)))
    return _alma(h, SMOOTH * 3, 0.85, 6.0)


def _alma(s, n, offset=0.85, sigma=6.0):
    m = offset * (n - 1)
    s_ = sigma  # noqa
    weights = np.exp(-((np.arange(n) - m) ** 2) / (2 * (n / sigma) ** 2))
    weights /= weights.sum()
    return s.rolling(n, min_periods=n).apply(lambda x: float(np.dot(x, weights)), raw=True)


def cand_combined_two_lengths(df):
    """Average of CMF at two lengths (35 and 14)."""
    bp = _bp(df)
    def _c(n):
        return (bp * df["volume"]).rolling(n, min_periods=n).sum() / df["volume"].rolling(n, min_periods=n).sum()
    return _sma(50 + 50 * 0.5 * (_c(LEN) + _c(14)), SMOOTH)


def cand_clip_squash(df):
    """Hybrid then clip to [10,90] then expand back."""
    h = 0.5 * (_mfi_raw(df) + (50 + 50 * _cmf_raw(df)))
    h = h.clip(10, 90)
    return _sma(h, SMOOTH)


CANDIDATES = {
    "stoch_cmf": cand_stoch_cmf,
    "stoch_mfi": cand_stoch_mfi,
    "stoch_hybrid": cand_stoch_hybrid,
    "rvol_weighted_cmf": cand_rvol_weighted,
    "dual_smooth_cmf_ema": cand_dual_smooth,
    "alma_hybrid": cand_alma,
    "two_lengths_cmf": cand_combined_two_lengths,
    "clip_squash_hybrid": cand_clip_squash,
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
            rows.append({"candidate": name, "n": 0, "note": str(e)})
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
    print("=== MF round-4 sweep ===")
    print(out.to_string(index=False))

    # Residual diagnostic: regress (TV - best_hybrid) on various features
    print("\n=== Residual structure (best hybrid mfi×0.4+cmf, sma6) ===")
    h = 0.5 * (_mfi_raw(ohlcv) + (50 + 50 * _cmf_raw(ohlcv)))
    best = _sma(0.4 * _mfi_raw(ohlcv) + 0.6 * (50 + 50 * _cmf_raw(ohlcv)), SMOOTH)
    resid = (truth - best).dropna()
    print(f"  resid mean={resid.mean():.3f}  std={resid.std():.3f}  range=[{resid.min():.2f}, {resid.max():.2f}]")
    # autocorrelation
    print(f"  resid autocorr lag1={resid.autocorr(1):.3f}  lag5={resid.autocorr(5):.3f}")
    # correlation with raw price/volume features
    rng = (ohlcv["high"] - ohlcv["low"])
    body = (ohlcv["close"] - ohlcv["open"]).abs()
    feats = pd.DataFrame(
        {
            "log_vol": np.log(ohlcv["volume"]),
            "rng_pct": rng / ohlcv["close"],
            "body_pct": body / ohlcv["close"],
            "ret_1d": ohlcv["close"].pct_change(),
            "rvol": ohlcv["volume"] / ohlcv["volume"].rolling(LEN).mean(),
            "rsi_close": tv.get("RSI", pd.Series(index=tv.index, dtype=float)),
        }
    )
    feats = feats.loc[resid.index]
    print(feats.corrwith(resid).sort_values(key=abs, ascending=False).to_string())


if __name__ == "__main__":
    main()
