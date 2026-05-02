"""channel_break output schema (v0.3).

v0.3 — containment tolerance is now ATR-relative (Suh et al. 2004 spirit;
Lo/Mamaysky/Wang 2000 volatility scaling). Per-setup tolerance:

    tol = clip(containment_tol_atr_k × ATR_n / close_at_asof,
               lo=containment_tol_min, hi=containment_tol_max)

Replaces fixed `containment_tol_pct`. Low-vol tickers get tighter bands,
high-vol tickers get wider, both bounded by floor/cap.

v0.2 — switched from OLS-through-pivots to containment-line fit:
upper line must satisfy close[k] ≤ upper(k) × (1 + tol) for every bar k in
[first_pivot..asof] (with at most max_line_violations exceptions);
lower line symmetric. n_pivots_upper / n_pivots_lower now mean *touches*
on the line, not raw pivot count in window.

Two parquet schemas:
  channel_break_<fam>.parquet   — accepted parallel channels with break states
  pending_triangle_<fam>.parquet — non-parallel fits forwarded to triangle workstream
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

CHANNEL_BREAK_VERSION = "0.3"


@dataclass(frozen=True)
class ChannelParams:
    family: str
    frequency: Literal["5h", "1d", "1w", "1M"]
    lookback_bars: int
    pivot_n: int = 2
    atr_n: int = 14
    vol_sma_n: int = 20
    sma_n: int = 20

    # detection thresholds
    parallelism_max: float = 0.25
    width_min_pct: float = 0.03
    width_max_pct: float = 0.25
    flat_slope_pct_per_bar: float = 0.15
    breakout_close_pct: float = 0.005       # close > upper * 1.005
    pre_breakout_close_pct: float = 0.005   # close >= upper * 0.995

    # containment fit — ATR-relative tolerance (v0.3)
    # tol = clip(k × ATR_n / close_at_asof, lo, hi)
    # k=0.30 chosen on tier-A plateau analysis (k∈[0.25,0.32] stable; jump
    # at 0.35 = phase transition, leaves plateau).
    containment_tol_atr_k: float = 0.30     # fraction of ATR
    containment_tol_min: float = 0.004      # floor (low-vol stocks)
    containment_tol_max: float = 0.020      # cap   (high-vol stocks)
    max_line_violations: int = 0            # bars allowed to break line

    # gates
    vol_ratio_min: float = 1.3
    range_pos_min: float = 0.60
    body_atr_min: float = 0.35

    # extended lookback (bars to scan back from asof for prior trigger)
    extended_lookback_bars: int = 5


FAMILIES: dict[str, ChannelParams] = {
    "ch_5h": ChannelParams("ch_5h", "5h", lookback_bars=80),
    "ch_1d": ChannelParams("ch_1d", "1d", lookback_bars=60),
    "ch_1w": ChannelParams("ch_1w", "1w", lookback_bars=40),
    "ch_1M": ChannelParams("ch_1M", "1M", lookback_bars=18),
}


_BASE_COLS = [
    "ticker",
    "setup_family",
    "data_frequency",
    "as_of_ts",
    "bar_date",

    # geometry
    "lookback_bars",
    "n_pivots_upper",
    "n_pivots_lower",
    "n_swing_touches",       # merged-zigzag length contributing to channel
    "tier_a",                # bool: ≥3 H AND ≥3 L touches
    "upper_slope_pct_per_bar",
    "lower_slope_pct_per_bar",
    "mean_slope_pct_per_bar",
    "upper_at_asof",
    "lower_at_asof",
    "channel_width_pct",
    "parallelism",           # |Δslope|/mean(|slope|), ∞-replaced for flat-flat
    "fit_max_residual_pct",  # max(pivot_close − line) / channel_width
    "fit_quality",           # tight / loose / rough
    "channel_age_bars",      # asof_idx − first_pivot_idx
    "first_pivot_idx",
    "first_pivot_bar_date",
    "last_pivot_idx",
    "last_pivot_bar_date",

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

    # schema marker
    "schema_version",
]

OUTPUT_COLUMNS: tuple[str, ...] = tuple(_BASE_COLS[:5] + [
    "signal_state",          # trigger / extended / pre_breakout
    "slope_class",           # asc / desc / flat
    "direction_tag",         # asc_up_break / desc_up_break / flat_up_break
    "breakout_idx",
    "breakout_bar_date",
    "breakout_age_bars",
    "breakout_close_over_upper_pct",
] + _BASE_COLS[5:])


PENDING_TRIANGLE_COLUMNS: tuple[str, ...] = tuple(_BASE_COLS[:5] + [
    "triangle_kind_hint",    # ascending / descending / symmetric / expanding
] + _BASE_COLS[5:])


def empty_row() -> dict:
    return {k: None for k in OUTPUT_COLUMNS}


def empty_pending_row() -> dict:
    return {k: None for k in PENDING_TRIANGLE_COLUMNS}
