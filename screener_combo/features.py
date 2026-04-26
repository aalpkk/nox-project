"""Per-ticker per-day feature panel for ranker discovery.

For each daily bar, produce ~25 numeric features spanning price/volume/trend/
structure/quality/RS/regime context. Features are computed series-wise.

Features (all computed on the daily DataFrame indexed by date):
  price_vol:
    ret_1d, ret_5d, ret_20d
    atr, atr_pct
    rvol_20
    range_atr
  trend:
    dist_ema21_pct, dist_ema55_pct
    ema21_above_ema55 (bool→int)
    supertrend_long
    adx, adx_slope
  structure:
    dist_20d_high_pct, dist_20d_low_pct
    drawdown_20d_pct
  quality (alsat semantics):
    quality_score, clv, wick_ratio
  RS:
    rs_score
  regime/breadth:
    alsat_regime           (0-3)
    nox_br_score, nox_rg_score, nox_gate_open
    rt_trend_score, rt_part_score, rt_exp_score, rt_regime
  context:
    cmf_20, obv_slope_5
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from screener_combo.signals import (
    _ema, _sma, _rma, _atr, _true_range, _adx_with_slope,
    _supertrend_dir, _quality_score, _rs_score, _alsat_signal_components,
)
from markets.bist.regime_transition import (
    compute_trend_score, compute_participation_score,
    compute_expansion_score, determine_regime,
)
from markets.bist.nox_v3_signals import compute_nox_v3, _pine_rsi


def compute_feature_panel(
    daily: pd.DataFrame,
    weekly: pd.DataFrame,
    bench: pd.Series,
) -> pd.DataFrame:
    """Return a feature DataFrame indexed by daily.index."""
    c, h, l, o, v = daily["close"], daily["high"], daily["low"], daily["open"], daily["volume"]

    feats = {}

    # ===== price/vol =====
    feats["ret_1d"] = c.pct_change(1)
    feats["ret_5d"] = c.pct_change(5)
    feats["ret_20d"] = c.pct_change(20)
    atr = _atr(daily, 14)
    feats["atr"] = atr
    feats["atr_pct"] = (atr / c) * 100.0
    vol_sma20 = _sma(v, 20)
    feats["rvol_20"] = v / vol_sma20.replace(0, np.nan)
    candle_range = (h - l).replace(0, np.nan)
    feats["range_atr"] = candle_range / atr.replace(0, np.nan)

    # ===== trend =====
    ema21 = _ema(c, 21)
    ema55 = _ema(c, 55)
    feats["dist_ema21_pct"] = (c - ema21) / ema21 * 100
    feats["dist_ema55_pct"] = (c - ema55) / ema55 * 100
    feats["ema21_above_ema55"] = (ema21 > ema55).astype(int)
    feats["supertrend_long"] = (_supertrend_dir(daily, 10, 3.0) == 1).astype(int)
    adx, adx_slope, _, _ = _adx_with_slope(daily, 14, 3)
    feats["adx"] = adx
    feats["adx_slope"] = adx_slope

    # ===== structure =====
    high_20 = h.rolling(20).max()
    low_20 = l.rolling(20).min()
    feats["dist_20d_high_pct"] = (c - high_20) / high_20 * 100  # negative = below high
    feats["dist_20d_low_pct"] = (c - low_20) / low_20 * 100
    feats["drawdown_20d_pct"] = (c - high_20) / high_20.replace(0, np.nan) * 100

    # ===== quality (alsat) =====
    q = _quality_score(daily)
    feats["quality_score"] = q
    feats["clv"] = (c - l) / candle_range
    upper_wick = h - pd.concat([c, o], axis=1).max(axis=1)
    feats["wick_ratio"] = upper_wick / candle_range

    # ===== RS =====
    feats["rs_score"] = _rs_score(c, bench)

    # ===== regime — alsat semantics =====
    comp = _alsat_signal_components(daily, weekly, bench)
    feats["alsat_regime"] = comp["regime"]
    feats["alsat_macd_pos"] = comp["macd_pos"].astype(int)

    # ===== regime — nox v3 (daily-level breadth/regime score) =====
    nox_d = compute_nox_v3(daily)
    feats["nox_br_score"] = nox_d["br_score"]
    feats["nox_rg_score"] = nox_d["rg_score"]
    feats["nox_gate_open"] = nox_d["gate_open"].astype(int)

    # ===== regime — regime_transition components =====
    try:
        trend_data = compute_trend_score(daily, weekly)
        part_data = compute_participation_score(daily)
        exp_data = compute_expansion_score(daily)
        feats["rt_trend_score"] = trend_data["trend_score"]
        feats["rt_part_score"] = part_data["participation_score"]
        feats["rt_exp_score"] = exp_data["expansion_score"]
        feats["rt_regime"] = determine_regime(
            trend_data["trend_score"],
            part_data["participation_score"],
            exp_data["expansion_score"],
        )
        feats["rt_cmf"] = part_data["cmf"]
        feats["rt_di_spread"] = exp_data["di_spread"]
    except Exception as e:
        # Fill with NaN if RT components fail (rare; e.g. very short series)
        for k in ("rt_trend_score", "rt_part_score", "rt_exp_score",
                  "rt_regime", "rt_cmf", "rt_di_spread"):
            feats[k] = pd.Series(np.nan, index=daily.index)

    # ===== context: OBV slope =====
    obv = (np.sign(c.diff()).fillna(0) * v).cumsum()
    obv_ema = obv.ewm(span=13, adjust=False).mean()
    feats["obv_slope_5"] = obv_ema.diff(5)

    df = pd.DataFrame(feats, index=daily.index)
    return df


FEATURE_COLS = [
    "ret_1d", "ret_5d", "ret_20d", "atr", "atr_pct", "rvol_20", "range_atr",
    "dist_ema21_pct", "dist_ema55_pct", "ema21_above_ema55", "supertrend_long",
    "adx", "adx_slope",
    "dist_20d_high_pct", "dist_20d_low_pct", "drawdown_20d_pct",
    "quality_score", "clv", "wick_ratio",
    "rs_score",
    "alsat_regime", "alsat_macd_pos",
    "nox_br_score", "nox_rg_score", "nox_gate_open",
    "rt_trend_score", "rt_part_score", "rt_exp_score", "rt_regime",
    "rt_cmf", "rt_di_spread",
    "obv_slope_5",
]
