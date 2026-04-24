"""
Volume dry-up ve breakout expansion metrikleri.

Compression dönemlerinin karakteristik imzası: volatilite ve hacim birlikte
kurur. Dry-up metrikleri `[t-W, t-1]` penceresinde; breakout bar metrikleri
bar t'de ölçülür (rvol, vol_pctile).

Leakage
-------
- dryup_ratio_*, quiet_bar_count: tamamen prior bars.
- breakout_vol_ratio, breakout_vol_pctile: t barı nominatör; denominator yine
  prior window (leakage-safe, çünkü "bugün ne kadar patladı" sorusu).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from cte.config import CompressionParams, DryupParams


def _mean_prior(s: pd.Series, window: int) -> pd.Series:
    """rolling mean shifted by 1 → [t-W, t-1]."""
    return s.rolling(window, min_periods=window).mean().shift(1)


def compute_volume(
    df: pd.DataFrame,
    dry: DryupParams | None = None,
    comp: CompressionParams | None = None,
) -> pd.DataFrame:
    """Per-bar volume dry-up + breakout volume columns.

    Returns DataFrame indexed like `df` with columns:
        vol_mean_prior_{w}           (prior-window means for each w in short+long)
        dryup_ratio_{s}_{l}          (short/long cross product)
        quiet_bar_count              (# of low-vol bars in last `quiet_bar_window`)
        structure_vol_ref            (mean vol over prior structure_vol_ref_window)
        breakout_vol_ratio           (vol_t / structure_vol_ref)
        breakout_vol_pctile          (percentile rank of vol_t over prior
                                       `bb_pctile_lookback` bars)
    """
    if dry is None:
        dry = DryupParams()
    if comp is None:
        comp = CompressionParams()

    if df.empty:
        return pd.DataFrame(index=df.index)

    vol = df["Volume"].astype(float)

    cols: dict[str, pd.Series] = {}

    # Rolling means for each window we need
    all_windows = set(dry.dryup_windows_short) | set(dry.dryup_windows_long) | {
        dry.structure_vol_ref_window,
    }
    for w in sorted(all_windows):
        cols[f"vol_mean_prior_{w}"] = _mean_prior(vol, w)

    # Dry-up ratios: short/long
    for s in dry.dryup_windows_short:
        for l in dry.dryup_windows_long:
            num = cols[f"vol_mean_prior_{s}"]
            den = cols[f"vol_mean_prior_{l}"].replace(0.0, np.nan)
            cols[f"dryup_ratio_{s}_{l}"] = num / den

    # Quiet-bar count: son `quiet_bar_window` barda vol < thresh * structure_avg
    struct_ref = cols[f"vol_mean_prior_{dry.structure_vol_ref_window}"]
    # Compare each prior bar's vol vs structure_ref AT THAT PRIOR BAR;
    # conservative simplification: use current-row struct_ref for all prior W bars
    # (struct_ref zaten [t-20, t-1] ortalaması → yaklaşık istikrarlı).
    thresh = dry.quiet_bar_rel_vol_thresh * struct_ref
    quiet_flag = (vol < thresh).astype(float)
    cols["quiet_bar_count"] = (
        quiet_flag.shift(1).rolling(dry.quiet_bar_window, min_periods=dry.quiet_bar_window).sum()
    )

    # Breakout volume — bar t nominator
    cols["structure_vol_ref"] = struct_ref
    cols["breakout_vol_ratio"] = vol / struct_ref.replace(0.0, np.nan)

    # Percentile of today's vol over prior lookback
    def _rank(window: np.ndarray) -> float:
        if np.isnan(window).any():
            return np.nan
        last = window[-1]
        return float((window <= last).sum() - 1) / (len(window) - 1)

    # Build window that ENDS at today, then compute rank; no shift(1) since we
    # want t included in the percentile computation.
    cols["breakout_vol_pctile"] = (
        vol.rolling(comp.bb_pctile_lookback, min_periods=comp.bb_pctile_lookback)
        .apply(_rank, raw=True)
    )

    return pd.DataFrame(cols, index=df.index)
