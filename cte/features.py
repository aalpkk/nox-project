"""
CTE feature blocks.

Block A — structure geometry (from cte.structure)
Block B — volume dry-up + breakout expansion (from cte.volume)
Block C — firstness (from cte.firstness)
Block D — breakout bar quality (from cte.breakout_bar)
Block E — regime / relative-strength (XU100 benchmark, computed here)

Bloklar A-D zaten `cte.trigger.compute_trigger` içinde üretildi; bu modül
trigger frame'ini alıp Block E'yi ekliyor, aynı zamanda "model'e gidecek"
feature seti için ortak isim alanını sabitliyor.

Leakage
-------
- Blok A-D: `cte.*` modüllerinin `[t-W, t-1]` guard'ı geçerli (structure,
  vol dry-up). Bar-t'ye ait: bar_* (kasıtlı, trigger'ın kendisi).
- Blok E:
    * XU100 RS, regime: XU100 daily close [t-W, t-1] pencerelerinden.
    * US-close (VIX/DXY/SPY) eklenmedi — ayrı modül olarak features_macro.py
      tarafında olacak; leakage guardsı için `.shift(1)` ZORUNLU kullanılmalı
      (memory/macro_timing_leakage.md).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from cte.structure import _atr_prior


@dataclass(frozen=True)
class FeatureParams:
    xu100_fast: int = 21
    xu100_slow: int = 55
    xu100_slope_window: int = 10
    rs_windows: tuple[int, ...] = (5, 10, 20)


# ═══════════════════════════════════════════════════════════════════════════════
# Block E — Regime / relative strength (XU100 benchmark)
# ═══════════════════════════════════════════════════════════════════════════════

def _xu100_trend_score(xu100_close: pd.Series, p: FeatureParams) -> pd.Series:
    """0..1 regime score: ema_fast>ema_slow + close>ema_fast + slope>0 → /3.

    Computed on XU100 daily close; `.shift(1)` at the end so today's XU100
    close isn't used at ticker row t (same-session but still avoid
    contemporaneous bleed).
    """
    if xu100_close is None or xu100_close.empty:
        return pd.Series(dtype=float)
    c = xu100_close.astype(float)
    ema_f = c.ewm(span=p.xu100_fast, adjust=False).mean()
    ema_s = c.ewm(span=p.xu100_slow, adjust=False).mean()
    slope = ema_f.pct_change(p.xu100_slope_window, fill_method=None)
    score = ((ema_f > ema_s).astype(int)
             + (c > ema_f).astype(int)
             + (slope > 0).astype(int)) / 3.0
    return score.shift(1)


def _block_e_regime(
    df: pd.DataFrame,
    xu100_close: pd.Series | None,
    params: FeatureParams,
) -> pd.DataFrame:
    """Per-ticker-bar regime + relative strength columns.

    Columns:
        xu100_trend_score_today
        rs_{w} for w in rs_windows   = stock_ret_w - xu100_ret_w  (both on [t-w, t-1])
        above_ma20, above_ma50        = close vs rolling MA20/50 at t-1
    """
    close = df["Close"].astype(float)

    out: dict[str, pd.Series] = {}

    # XU100 regime (aligned + shift(1))
    if xu100_close is None or xu100_close.empty:
        out["xu100_trend_score_today"] = pd.Series(np.nan, index=df.index)
    else:
        xu_score = _xu100_trend_score(xu100_close, params).reindex(df.index, method="ffill")
        out["xu100_trend_score_today"] = xu_score

    # RS: stock vs XU100 over [t-w, t-1]
    xu_close_aligned = (
        xu100_close.reindex(df.index, method="ffill").astype(float)
        if (xu100_close is not None and not xu100_close.empty)
        else pd.Series(np.nan, index=df.index)
    )
    for w in params.rs_windows:
        stock_ret = close.pct_change(w, fill_method=None).shift(1)
        xu_ret = xu_close_aligned.pct_change(w, fill_method=None).shift(1)
        out[f"rs_{w}"] = stock_ret - xu_ret

    # Moving-average regime (based on closes up to t-1)
    for w, label in [(20, "above_ma20"), (50, "above_ma50")]:
        ma = close.rolling(w, min_periods=w).mean().shift(1)
        prev_close = close.shift(1)
        out[label] = (prev_close > ma).astype(float).where(~ma.isna())

    return pd.DataFrame(out, index=df.index)


# ═══════════════════════════════════════════════════════════════════════════════
# Model-facing feature assembly
# ═══════════════════════════════════════════════════════════════════════════════

# Canonical feature column list for single-head LGBM (ilk run için).
# Bloklar A-D'den seçilmiş + Block E. Listeyi genişletmek için ablation tarafı
# kullanılır; burası "ilk pass, noise-free" set.
FEATURES_V1: tuple[str, ...] = (
    # Block A — structure geometry
    "hb_width_atr",
    "hb_slope_atr",
    "hb_touches_upper",
    "hb_touches_lower",
    "hb_close_density_atr",
    "fc_upper_slope_atr",
    "fc_lower_slope_atr",
    "fc_width_atr",
    "fc_width_cv",
    "fc_convergence",
    "fc_lower_high_count",
    "bb_width_pctile",
    "compression_score",
    # Block B — volume dry-up / breakout expansion
    "dryup_ratio_3_20",
    "dryup_ratio_5_30",
    "quiet_bar_count",
    "breakout_vol_ratio",
    "breakout_vol_pctile",
    # Block C — firstness
    "hb_prior_break_count",
    "fc_prior_break_count",
    "hb_failed_break_count",
    "fc_failed_break_count",
    "hb_last_break_age",
    "fc_last_break_age",
    # Block D — breakout bar quality
    "bar_return_1d",
    "bar_rvol",
    "bar_close_loc",
    "bar_body_pct_range",
    # Block E — regime / RS
    "xu100_trend_score_today",
    "rs_5",
    "rs_10",
    "rs_20",
    "above_ma20",
    "above_ma50",
    # Categorical — setup type ("hb"=0, "fc"=1, "both"=2)
    "setup_type_code",
)


def enrich_with_block_e(
    df: pd.DataFrame,
    trigger_frame: pd.DataFrame,
    xu100_close: pd.Series | None = None,
    params: FeatureParams | None = None,
) -> pd.DataFrame:
    """Trigger frame + Block E → per-bar feature frame with all columns needed
    by FEATURES_V1.

    Parameters
    ----------
    df : OHLCV.
    trigger_frame : cte.trigger.compute_trigger output.
    xu100_close : XU100 daily close series.
    params : FeatureParams.

    Returns
    -------
    DataFrame indexed like df with trigger_frame columns + Block E + setup_type_code.
    """
    if params is None:
        params = FeatureParams()

    block_e = _block_e_regime(df, xu100_close, params)

    # Encode setup_type
    if "setup_type" in trigger_frame.columns:
        setup_map = {"hb": 0, "fc": 1, "both": 2}
        setup_code = trigger_frame["setup_type"].map(setup_map)
    else:
        setup_code = pd.Series(np.nan, index=df.index)

    out = pd.concat(
        [trigger_frame, block_e, setup_code.rename("setup_type_code")],
        axis=1,
    )
    return out
