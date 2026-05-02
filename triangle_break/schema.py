"""triangle_break output schema (v0.3 — ATR-relative tolerance).

One parquet per family — triangle_break_<fam>.parquet.

Inherits ChannelParams field semantics (lookback, pivot_n, gates) and
adds three triangle-specific fields plus a subtype filter.

v0.3: containment tolerance is ATR-relative (k × ATR / close, clipped
to [min, max]). Replaces fixed v0.2 percent. Low-vol stocks get tighter
bands, high-vol get wider, both bounded.

v0.2: line fit upgraded from OLS-through-pivots to containment fit. Upper
line must satisfy close[k] ≤ upper(k) × (1+tol) for every bar in window
(at most `max_line_violations` exceptions); lower line symmetric. Drops
"trendline broken multiple times before trigger" artifacts.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import FrozenSet, Literal

TRIANGLE_BREAK_VERSION = "0.3"


@dataclass(frozen=True)
class TriangleParams:
    family: str
    frequency: Literal["5h", "1d", "1w", "1M"]
    lookback_bars: int
    pivot_n: int = 2
    atr_n: int = 14
    vol_sma_n: int = 20
    sma_n: int = 20

    # parallelism — opposite sign from channel: triangle REQUIRES non-parallel
    parallelism_max: float = 0.25  # used by ChannelParams API; here represents
                                   # the lower bound: triangle gate is parallelism > this
    width_min_pct: float = 0.03
    width_max_pct: float = 0.40    # triangles can be wider than channels at first pivot
    flat_slope_pct_per_bar: float = 0.15
    breakout_close_pct: float = 0.005
    pre_breakout_close_pct: float = 0.005

    # containment fit — ATR-relative tolerance (v0.3, shared with ChannelParams API)
    # tol = clip(k × ATR_n / close_at_asof, lo, hi)
    # k=0.30 = plateau midpoint (k∈[0.25,0.32] stable; 0.35 = phase transition).
    containment_tol_atr_k: float = 0.30
    containment_tol_min: float = 0.004
    containment_tol_max: float = 0.020
    max_line_violations: int = 0

    vol_ratio_min: float = 1.3
    range_pos_min: float = 0.60
    body_atr_min: float = 0.35
    extended_lookback_bars: int = 5

    # Triangle-specific
    accepted_subtypes: FrozenSet[str] = field(
        default_factory=lambda: frozenset({"ascending", "symmetric", "descending"}),
    )
    bars_to_apex_min: int = 1
    width_contraction_max: float = 0.85


FAMILIES: dict[str, TriangleParams] = {
    "tr_5h": TriangleParams("tr_5h", "5h", lookback_bars=80),
    "tr_1d": TriangleParams("tr_1d", "1d", lookback_bars=60),
    "tr_1w": TriangleParams("tr_1w", "1w", lookback_bars=40),
    "tr_1M": TriangleParams("tr_1M", "1M", lookback_bars=18),
}


_BASE_COLS = [
    "ticker",
    "setup_family",
    "data_frequency",
    "as_of_ts",
    "bar_date",

    "signal_state",          # trigger / extended / pre_breakout
    "triangle_subtype",      # ascending / symmetric / descending
    "direction_tag",         # asc_up_break / sym_up_break / desc_up_break
    "breakout_idx",
    "breakout_bar_date",
    "breakout_age_bars",
    "breakout_close_over_upper_pct",

    # geometry
    "lookback_bars",
    "n_pivots_upper",
    "n_pivots_lower",
    "n_swing_touches",
    "tier_a",
    "upper_slope_pct_per_bar",
    "lower_slope_pct_per_bar",
    "mean_slope_pct_per_bar",
    "upper_at_asof",
    "lower_at_asof",
    "channel_width_pct",       # width at asof
    "initial_width_pct",        # width at first_pivot_idx
    "width_contraction_ratio",  # width_at_asof / width_at_first_pivot
    "parallelism",
    "fit_max_residual_pct",
    "fit_quality",
    "channel_age_bars",
    "first_pivot_idx",
    "first_pivot_bar_date",
    "last_pivot_idx",
    "last_pivot_bar_date",

    # triangle-specific
    "apex_idx",
    "apex_bar_date",
    "bars_to_apex",
    "apex_progress",            # channel_age / (channel_age + bars_to_apex)

    # market
    "asof_close",
    "asof_open",
    "asof_high",
    "asof_low",
    "asof_volume",
    "atr_14",
    "atr_pct",
    "vol_ratio_20",
    "range_pos",
    "body_atr",
    "close_vs_sma20",

    "schema_version",
]

OUTPUT_COLUMNS: tuple[str, ...] = tuple(_BASE_COLS)


def empty_row() -> dict:
    return {k: None for k in OUTPUT_COLUMNS}
