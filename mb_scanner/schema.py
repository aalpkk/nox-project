"""Output schema for mb_scanner pipeline (V0.3).

Family universes are split by market — BIST and US do not mix:

  BIST_FAMILIES = mb_{5h,1d,1w,1M}, bb_{5h,1d,1w,1M}            (8)
  US_FAMILIES   = mb_{1h,3h,1d,1w,1M}, bb_{1h,3h,1d,1w,1M}      (10)
  ALL_FAMILIES  = union of both                                  (12)

`FAMILIES` is an alias for `BIST_FAMILIES` (the historical contract — all
BIST tooling defaults to it). US tooling imports `US_FAMILIES` explicitly.

Each scan emits 0..N rows per (ticker, family) at as-of — one row per
simultaneously-active quartet (wide macro setups can coexist with
narrower nested ones; all are reported).

1h frequency reads the raw nox_intraday master directly (no resample) — used
for US (NYSE 09:30–16:00 ET → 7 hourly bars/day incl. last 30min stub).
3h frequency is the US NYSE session-split mirror of BIST mb_5h: AM bar
labeled 09:30 (09:30–13:00, 3.5h incl. opening stub) + PM bar labeled
13:00 (13:00–16:00, 3h).
5h frequency mirrors TV's "5 saatlik" for BIST: 09:00 morning bar
(opening auction + continuous 10–14) and 14:00 afternoon bar (continuous
14–18 + closing auction at 18:05 + closing-price trades 18:08–18:10).

`quartet_rank` is 0-indexed by hh_idx ascending (0 = oldest HH = widest);
`n_active_quartets` = total rows for that (ticker, family) at as-of.
"""
from __future__ import annotations

VERSION = "0.3.0"

BIST_FAMILIES = (
    "mb_5h", "mb_1d", "mb_1w", "mb_1M",
    "bb_5h", "bb_1d", "bb_1w", "bb_1M",
)

US_FAMILIES = (
    "mb_1h", "mb_3h", "mb_1d", "mb_1w", "mb_1M",
    "bb_1h", "bb_3h", "bb_1d", "bb_1w", "bb_1M",
)

ALL_FAMILIES = tuple(dict.fromkeys(BIST_FAMILIES + US_FAMILIES))

FAMILIES = BIST_FAMILIES

STATES = ("above_mb", "mitigation_touch", "retest_bounce", "extended")

# Output column order (keep stable for downstream consumers).
OUTPUT_COLUMNS = (
    # identity
    "ticker", "setup_family", "data_frequency", "signal_state",
    "as_of_ts", "bar_date",
    "quartet_rank", "n_active_quartets",
    # quartet structure
    "ll_idx", "ll_bar_date", "ll_price",
    "lh_idx", "lh_bar_date", "lh_price",
    "hl_idx", "hl_bar_date", "hl_price",
    "hh_idx", "hh_bar_date", "hh_close",
    # zone
    "zone_origin_idx", "zone_origin_bar_date",
    "zone_high", "zone_low", "zone_width_pct", "zone_width_atr",
    "zone_age_bars",
    # state details
    "touches_into_zone", "deepest_low_after_hh",
    "retest_depth_atr", "retest_kind", "retest_idx", "retest_bar_date",
    # context at asof
    "asof_close", "asof_volume",
    "bos_distance_pct", "bos_distance_atr",  # asof vs LH price
    "atr_14", "atr_pct", "vol_ratio_20",
    "close_vs_sma20",
    # risk
    "structural_invalidation_low", "initial_risk_pct",
    # provenance
    "schema_version",
)


def empty_row() -> dict:
    return {col: None for col in OUTPUT_COLUMNS}
