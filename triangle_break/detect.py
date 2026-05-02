"""Triangle detection: converging-line geometry + apex/contraction gates.

Reuses fit_geometry + scan_breakout_state from channel_break.detect.
Branching:
  - parallelism ≤ params.parallelism_max → channel territory, skip
  - subtype ∉ accepted_subtypes → skip (expanding/ambiguous out)
  - bars_to_apex < bars_to_apex_min → expired triangle, skip
  - width_contraction_ratio ≥ width_contraction_max → not converging, skip
  - else compute breakout state via shared helper, emit row
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from channel_break.detect import (
    _bar_date,
    _line_at,
    _triangle_kind,
    fit_geometry,
    scan_breakout_state,
)

from .schema import TRIANGLE_BREAK_VERSION, TriangleParams, empty_row


def _apex(s_u: float, b_u: float, s_l: float, b_l: float) -> Optional[float]:
    """x where upper(x) = lower(x); None if lines parallel."""
    denom = s_u - s_l
    if denom == 0.0:
        return None
    return (b_l - b_u) / denom


def detect(
    df: pd.DataFrame,
    asof_idx: int,
    ticker: str,
    fam_name: str,
    params: TriangleParams,
) -> Optional[dict]:
    """Detect triangle at asof_idx. Returns row dict or None."""
    geom = fit_geometry(df, asof_idx, params)
    if geom is None:
        return None

    if geom["parallelism"] <= params.parallelism_max:
        return None  # parallel — channel territory

    subtype = _triangle_kind(geom["s_u_pct"], geom["s_l_pct"])
    if subtype not in params.accepted_subtypes:
        return None

    # apex and bars_to_apex
    apex_x = _apex(geom["s_u"], geom["b_u"], geom["s_l"], geom["b_l"])
    if apex_x is None:
        return None
    bars_to_apex = int(round(apex_x - asof_idx))
    if bars_to_apex < params.bars_to_apex_min:
        return None  # expired

    # width contraction: width at first pivot vs width at asof
    first_idx = geom["first_pivot_idx"]
    upper_at_first = _line_at(geom["s_u"], geom["b_u"], first_idx)
    lower_at_first = _line_at(geom["s_l"], geom["b_l"], first_idx)
    asof_close = geom["asof_close"]
    if upper_at_first <= lower_at_first or asof_close <= 0:
        return None
    initial_width_pct = (upper_at_first - lower_at_first) / asof_close
    contraction = (
        geom["channel_width_pct"] / initial_width_pct
        if initial_width_pct > 0 else float("inf")
    )
    if contraction >= params.width_contraction_max:
        return None  # not actually converging

    if (geom["channel_width_pct"] < params.width_min_pct
            or geom["channel_width_pct"] > params.width_max_pct):
        return None

    signal_state, breakout_idx, breakout_age, brk_over_upper = scan_breakout_state(
        df, asof_idx, geom, params,
    )
    if signal_state is None:
        return None

    breakout_bar_date = None
    if breakout_idx is not None:
        breakout_bar_date = _bar_date(pd.Timestamp(df.index[breakout_idx]))

    apex_progress = (
        geom["channel_age_bars"]
        / (geom["channel_age_bars"] + bars_to_apex)
        if (geom["channel_age_bars"] + bars_to_apex) > 0 else 0.0
    )

    # Apex date: nearest valid index in df
    n_bars = len(df)
    apex_idx_int = int(round(apex_x))
    apex_bar_date = None
    if 0 <= apex_idx_int < n_bars:
        apex_bar_date = _bar_date(pd.Timestamp(df.index[apex_idx_int]))

    direction_map = {
        "ascending": "asc_up_break",
        "symmetric": "sym_up_break",
        "descending": "desc_up_break",
    }

    row = empty_row()
    row.update({
        "ticker": ticker,
        "setup_family": fam_name,
        "data_frequency": params.frequency,
        "as_of_ts": geom["asof_ts"],
        "bar_date": geom["bar_date"],

        "signal_state": signal_state,
        "triangle_subtype": subtype,
        "direction_tag": direction_map[subtype],
        "breakout_idx": int(breakout_idx) if breakout_idx is not None else None,
        "breakout_bar_date": breakout_bar_date,
        "breakout_age_bars": breakout_age,
        "breakout_close_over_upper_pct": brk_over_upper,

        "lookback_bars": params.lookback_bars,
        "n_pivots_upper": geom["n_pivots_upper"],
        "n_pivots_lower": geom["n_pivots_lower"],
        "n_swing_touches": geom["n_swing_touches"],
        "tier_a": geom["tier_a"],
        "upper_slope_pct_per_bar": geom["s_u_pct"],
        "lower_slope_pct_per_bar": geom["s_l_pct"],
        "mean_slope_pct_per_bar": geom["mean_slope_pct"],
        "upper_at_asof": geom["upper_at_asof"],
        "lower_at_asof": geom["lower_at_asof"],
        "channel_width_pct": geom["channel_width_pct"],
        "initial_width_pct": initial_width_pct,
        "width_contraction_ratio": contraction,
        "parallelism": geom["parallelism"],
        "fit_max_residual_pct": geom["fit_max_residual_pct"],
        "fit_quality": geom["fit_quality"],
        "channel_age_bars": geom["channel_age_bars"],
        "first_pivot_idx": int(first_idx),
        "first_pivot_bar_date": _bar_date(pd.Timestamp(df.index[first_idx])),
        "last_pivot_idx": int(geom["last_pivot_idx"]),
        "last_pivot_bar_date": _bar_date(pd.Timestamp(df.index[geom["last_pivot_idx"]])),

        "apex_idx": apex_idx_int,
        "apex_bar_date": apex_bar_date,
        "bars_to_apex": bars_to_apex,
        "apex_progress": apex_progress,

        "asof_close": geom["asof_close"],
        "asof_open": geom["asof_open"],
        "asof_high": geom["asof_high"],
        "asof_low": geom["asof_low"],
        "asof_volume": geom["asof_volume"],
        "atr_14": geom["atr_14"],
        "atr_pct": geom["atr_pct"],
        "vol_ratio_20": geom["vol_ratio_20"],
        "range_pos": geom["range_pos"],
        "body_atr": geom["body_atr"],
        "close_vs_sma20": geom["close_vs_sma20"],
        "schema_version": TRIANGLE_BREAK_VERSION,
    })
    return row
