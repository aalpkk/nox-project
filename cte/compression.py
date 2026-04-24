"""
Compression score — structure-agnostic tightness metric.

Horizontal-base ve falling-channel detektörleri yapısal filtre sağlar; bu modül
"yapıdan bağımsız" tightness ölçer. compression_score hem scan-time filtre,
hem feature hem de debug amacıyla kullanılır.

Üç ham bileşen:
  1. bb_width_atr       : bollinger width / ATR — kutunun ne kadar dar olduğu
  2. bb_width_pctile    : son `bb_pctile_lookback` bardaki percentile rank
                           (0=en dar, 1=en geniş) — rejim-göreli
  3. close_density_atr  : std(close, close_density_window) / ATR — mumlar ne
                           kadar sıkışık

Composite `compression_score` ∈ [0, 1] — 1 = maksimum sıkışma.

Leakage
-------
Tüm bileşenler `[t-W, t-1]` penceresinde hesaplanır. ATR `cte/structure.py`
'daki `_atr_prior` ile aynı convention'da — t barı dışarıda.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from cte.config import CompressionParams
from cte.structure import _atr_prior


def _rolling_pctile_prior(s: pd.Series, lookback: int) -> pd.Series:
    """Percentile rank of the last value in [t-L, t-1] — returns 0..1.

    0 = en küçük (en dar band), 1 = en büyük.
    """
    def _rank(window: np.ndarray) -> float:
        if np.isnan(window).any():
            return np.nan
        last = window[-1]
        return float((window <= last).sum() - 1) / (len(window) - 1)

    return (
        s.rolling(lookback, min_periods=lookback)
        .apply(_rank, raw=True)
        .shift(1)
    )


def compute_compression(
    df: pd.DataFrame,
    comp: CompressionParams | None = None,
) -> pd.DataFrame:
    """Per-bar compression metrics + composite score.

    Returns DataFrame indexed like `df` with columns:
        bb_upper, bb_lower, bb_width, bb_width_atr, bb_width_pctile,
        close_density_atr, compression_score
    """
    if comp is None:
        comp = CompressionParams()

    if df.empty:
        return pd.DataFrame(index=df.index)

    close = df["Close"].astype(float)
    atr = _atr_prior(df, comp.atr_window)

    # Bollinger over [t-W, t-1]
    ma = close.rolling(comp.bb_window, min_periods=comp.bb_window).mean().shift(1)
    sd = close.rolling(comp.bb_window, min_periods=comp.bb_window).std().shift(1)
    bb_upper = ma + 2.0 * sd
    bb_lower = ma - 2.0 * sd
    bb_width = bb_upper - bb_lower
    bb_width_atr = bb_width / atr.replace(0.0, np.nan)

    # Percentile rank of width over longer lookback — rejim-göreli sıkışma
    bb_width_pctile = _rolling_pctile_prior(bb_width, comp.bb_pctile_lookback)

    # Close dispersion / ATR on [t-W, t-1]
    close_std = (
        close.rolling(comp.close_density_window, min_periods=comp.close_density_window)
        .std()
        .shift(1)
    )
    close_density_atr = close_std / atr.replace(0.0, np.nan)

    # Composite: hepsi 0..1'e map'lenir (düşük = sıkı), ortalama alınır.
    # bb_width_atr: 0..6 → 0..1 clamp (6 ATR üstü "geniş" kabul)
    comp_width = 1.0 - (bb_width_atr / 6.0).clip(lower=0.0, upper=1.0)
    # close_density_atr: 0..2 → 0..1 clamp
    comp_density = 1.0 - (close_density_atr / 2.0).clip(lower=0.0, upper=1.0)
    # bb_width_pctile already 0..1 where 0=en dar → 1-p
    comp_pctile = 1.0 - bb_width_pctile

    # Eşit ağırlık. NaN varsa NaN döner.
    compression_score = (comp_width + comp_density + comp_pctile) / 3.0

    out = pd.DataFrame(
        {
            "bb_upper": bb_upper,
            "bb_lower": bb_lower,
            "bb_width": bb_width,
            "bb_width_atr": bb_width_atr,
            "bb_width_pctile": bb_width_pctile,
            "close_density_atr": close_density_atr,
            "compression_score": compression_score,
        },
        index=df.index,
    )
    return out
