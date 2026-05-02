"""Scanner V1 frozen schema.

Column registry, dtypes, prefix rules, leakage classes, and rule-score weight
priors for the multi-family breakout scanner. Versioned spec — bump
SCHEMA_VERSION / FEATURE_VERSION on any structural change; never mutate values
in place once a build has emitted with them.

Frozen V1 decisions (see project memory):
    1. Rule-score weights are fixed priors. No tuning, no grid search.
    2. Breakout mechanics live in `common__`. Family columns describe geometry only.
    3. Pivot-derived features require confirmed-lag semantics by definition,
       not just naming. Tag with leakage="pivot_confirmed_lag".
    4. `signal_tags` is a JSON-encoded list[str]; free-text only in
       optional `signal_reason_text`.
    5. Output rows MUST NOT carry `label__` or `model__` prefixes — those
       belong to labeled datasets and post-prediction artifacts respectively.
    6. Replace RS flags with numeric percentiles / distances.
    7. `market_regime_score` is decomposed into a vector — never collapsed.
    8. Every family populates the contract columns (trigger_level,
       entry_reference_price, invalidation_level, initial_risk_pct).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal


SCHEMA_VERSION = "1.2.0"   # +mitigation_block, +breaker_block families (1h ICT/SMC)
FEATURE_VERSION = "1.3.0"  # +family__zone_* / family__bos_* / family__impulse_* on
                           # mitigation_block & breaker_block
SCANNER_VERSION = "1.4.0"  # MB/BB ICT/SMC engines on 1h bars: bullish mitigation
                           # block + bullish breaker block, both with retest_bounce
                           # + extended state vocabulary aligned with V1.2.9.

PREFIX_COMMON = "common__"
PREFIX_FAMILY = "family__"
PREFIX_LABEL = "label__"
PREFIX_MODEL = "model__"

ALLOWED_OUTPUT_PREFIXES: tuple[str, ...] = (PREFIX_COMMON, PREFIX_FAMILY)
FORBIDDEN_OUTPUT_PREFIXES: tuple[str, ...] = (PREFIX_LABEL, PREFIX_MODEL)

FAMILIES: tuple[str, ...] = (
    "horizontal_base",
    "ascending_triangle",
    "falling_channel",
    "volatility_squeeze",
    "failed_breakdown_reclaim",
    "breakout_pullback",
    "squeeze_breakout_loose_v0_diagnostic",  # NOT production — diagnostic-only.
                                              # Validation outcome decides whether
                                              # it graduates to a real family.
    "mitigation_block",   # ICT/SMC bullish MB on 1h bars (V1.4.0)
    "breaker_block",      # ICT/SMC bullish BB on 1h bars (V1.4.0)
)


LeakageClass = Literal[
    "current_bar_safe",      # uses only data <= as_of_ts
    "pivot_confirmed_lag",   # swing at t becomes usable at t+k (post-confirmation)
    "rolling_excl_current",  # rolling window must exclude the current bar
    "next_open_only",        # may only be filled at next-bar open
]


@dataclass(frozen=True)
class Column:
    name: str
    dtype: str
    group: str
    leakage: LeakageClass
    nullable: bool = True
    notes: str = ""


# -- Identity & audit ---------------------------------------------------------

IDENTITY_COLUMNS: list[Column] = [
    Column("ticker", "string", "identity", "current_bar_safe", nullable=False),
    Column("bar_date", "datetime64[ns]", "identity", "current_bar_safe", nullable=False,
           notes="snapshot date; for trigger == breakout bar; for pre/extended == as-of bar"),
    Column("setup_family", "string", "identity", "current_bar_safe", nullable=False),
    Column("signal_type", "string", "identity", "current_bar_safe", nullable=False),
    Column("signal_state", "string", "identity", "current_bar_safe", nullable=False,
           notes="family-specific state vocabulary. horizontal_base: "
                 "pre_breakout / trigger / retest_bounce / extended. "
                 "mitigation_block & breaker_block: zone_armed / "
                 "mitigation_touch / retest_bounce / extended."),
    Column("breakout_bar_date", "datetime64[ns]", "identity", "current_bar_safe", nullable=True,
           notes="actual breakout / BoS bar (NaT for pre_breakout). For "
                 "mitigation_block & breaker_block this is the bar where the "
                 "Break-of-Structure confirmed and the zone armed."),
]

AUDIT_COLUMNS: list[Column] = [
    Column("as_of_ts", "datetime64[ns, Europe/Istanbul]", "audit", "current_bar_safe", nullable=False,
           notes="bar-close timestamp the features were computed against"),
    Column("data_frequency", "string", "audit", "current_bar_safe", nullable=False,
           notes="e.g. '1d', '1h'"),
    Column("schema_version", "string", "audit", "current_bar_safe", nullable=False),
    Column("feature_version", "string", "audit", "current_bar_safe", nullable=False),
    Column("scanner_version", "string", "audit", "current_bar_safe", nullable=False),
]


# -- Trigger / invalidation contract — every family must populate ------------

CONTRACT_COLUMNS: list[Column] = [
    Column("family__trigger_level", "Float32", "contract", "current_bar_safe", nullable=False,
           notes="setup-defined breakout level (channel high, resistance, reclaimed support, ...)"),
    Column("entry_reference_price", "Float32", "contract", "current_bar_safe", nullable=False,
           notes="close used for the trigger decision; not the realized fill"),
    Column("invalidation_level", "Float32", "contract", "current_bar_safe", nullable=False,
           notes="price below/above which the setup is structurally void"),
    Column("initial_risk_pct", "Float32", "contract", "current_bar_safe", nullable=False,
           notes="(entry - invalidation) / entry, signed by setup direction"),
]


# -- Common features (shared across families) --------------------------------

COMMON_COLUMNS: list[Column] = [
    # Bar mechanics on the trigger bar
    Column("common__breakout_pct", "Float32", "mechanics", "current_bar_safe"),
    Column("common__breakout_atr", "Float32", "mechanics", "current_bar_safe"),
    Column("common__range_position", "Float32", "mechanics", "current_bar_safe"),
    Column("common__body_pct", "Float32", "mechanics", "current_bar_safe"),
    Column("common__volume_ratio_20", "Float32", "mechanics", "current_bar_safe"),
    Column("common__extension_from_trigger", "Float32", "mechanics", "current_bar_safe"),
    Column("common__entry_distance_to_trigger", "Float32", "mechanics", "current_bar_safe"),
    Column("common__gap_pct", "Float32", "mechanics", "current_bar_safe",
           notes="same-day open gap; never use next-day"),
    Column("common__gap_atr", "Float32", "mechanics", "current_bar_safe",
           notes="same-day open gap; never use next-day"),
    Column("common__day_return", "Float32", "mechanics", "current_bar_safe",
           notes="close / prev_close - 1; day-over-day momentum (orthogonal to body)"),
    Column("common__day_return_atr", "Float32", "mechanics", "current_bar_safe",
           notes="(close - prev_close) / atr_sq; day-return normalized by base volatility"),

    # Volatility / liquidity
    Column("common__atr_14", "Float32", "vol_liq", "current_bar_safe"),
    Column("common__atr_pct", "Float32", "vol_liq", "current_bar_safe"),
    Column("common__realized_vol_20", "Float32", "vol_liq", "current_bar_safe"),
    Column("common__realized_vol_pctile_120", "Float32", "vol_liq", "current_bar_safe"),
    Column("common__volume", "Float64", "vol_liq", "current_bar_safe"),
    Column("common__turnover", "Float64", "vol_liq", "current_bar_safe"),
    Column("common__liquidity_score", "Float32", "vol_liq", "current_bar_safe"),
    Column("common__risk_pct_score", "Float32", "vol_liq", "current_bar_safe"),

    # Trend / VWAP context
    Column("common__close_vs_sma20", "Float32", "trend", "current_bar_safe"),
    Column("common__sma20_slope", "Float32", "trend", "current_bar_safe"),
    Column("common__close_vs_vwap10", "Float32", "trend", "current_bar_safe"),
    Column("common__close_vs_vwap52", "Float32", "trend", "current_bar_safe"),
    Column("common__close_vs_vwap_ytd", "Float32", "trend", "current_bar_safe",
           notes="YTD-anchored VWAP — resets on Jan 1 of each year"),
    Column("common__close_vs_vwap_base", "Float32", "trend", "current_bar_safe",
           notes="base-anchored VWAP — anchored to squeeze-start bar"),
    Column("common__vwap10_vs_vwap52", "Float32", "trend", "current_bar_safe"),
    Column("common__extension_from_sma20", "Float32", "trend", "current_bar_safe"),
    Column("common__extension_from_vwap52", "Float32", "trend", "current_bar_safe"),

    # Momentum / RS — numeric only, no boolean flags
    Column("common__ret_1w", "Float32", "momentum", "current_bar_safe"),
    Column("common__ret_4w", "Float32", "momentum", "current_bar_safe"),
    Column("common__ret_12w", "Float32", "momentum", "current_bar_safe"),
    Column("common__rs_20d", "Float32", "rs", "current_bar_safe"),
    Column("common__rs_60d", "Float32", "rs", "current_bar_safe"),
    Column("common__rs_pctile_120", "Float32", "rs", "current_bar_safe"),
    Column("common__rs_pctile_252", "Float32", "rs", "current_bar_safe"),
    Column("common__rs_dist_to_252_high", "Float32", "rs", "current_bar_safe"),
    Column("common__sector_rs_20d", "Float32", "rs", "current_bar_safe"),

    # Market regime — vector, never collapsed to one score
    Column("common__market_trend_score", "Float32", "regime", "current_bar_safe"),
    Column("common__market_breadth_pct_above_sma20", "Float32", "regime", "current_bar_safe"),
    Column("common__market_vol_regime", "Float32", "regime", "current_bar_safe"),
    Column("common__index_ret_5d", "Float32", "regime", "current_bar_safe"),
    Column("common__index_ret_20d", "Float32", "regime", "current_bar_safe"),
]


# -- Family-specific features (geometry / setup context only) ----------------

_HORIZONTAL_BASE: list[Column] = [
    Column("family__base_lookback_weeks", "Int16", "geometry", "current_bar_safe"),
    Column("family__channel_high", "Float32", "geometry", "rolling_excl_current"),
    Column("family__channel_low", "Float32", "geometry", "rolling_excl_current"),
    Column("family__channel_mid", "Float32", "geometry", "rolling_excl_current"),
    Column("family__channel_width_pct", "Float32", "geometry", "rolling_excl_current"),
    Column("family__channel_width_atr", "Float32", "geometry", "rolling_excl_current"),
    Column("family__channel_width_pctile_252", "Float32", "geometry", "rolling_excl_current"),
    Column("family__base_duration_weeks", "Float32", "geometry", "rolling_excl_current"),
    Column("family__upper_touch_count", "Int16", "geometry", "pivot_confirmed_lag"),
    Column("family__lower_touch_count", "Int16", "geometry", "pivot_confirmed_lag"),
    Column("family__touch_balance", "Float32", "geometry", "pivot_confirmed_lag"),
    Column("family__close_position_in_base", "Float32", "geometry", "current_bar_safe"),
    Column("family__prebreakout_distance_to_high", "Float32", "geometry", "current_bar_safe"),
    Column("family__bars_since_base_start", "Int16", "geometry", "rolling_excl_current"),
    Column("family__volume_dryup_ratio", "Float32", "geometry", "rolling_excl_current"),
    Column("family__volume_dryup_pctile", "Float32", "geometry", "rolling_excl_current"),
    Column("family__false_break_below_count", "Int16", "geometry", "pivot_confirmed_lag"),
    Column("family__inside_base_close_ratio", "Float32", "geometry", "rolling_excl_current"),
    Column("family__base_slope", "Float32", "geometry", "rolling_excl_current"),
    Column("family__base_r2", "Float32", "geometry", "rolling_excl_current"),
    Column("family__resistance_slope", "Float32", "geometry", "rolling_excl_current"),
    Column("family__hard_resistance", "Float32", "geometry", "rolling_excl_current"),
    Column("family__atr_decline_5_pct", "Float32", "geometry", "rolling_excl_current"),
    Column("family__atr_slope_10", "Float32", "geometry", "rolling_excl_current"),
    Column("family__atr_ratio_mean_base", "Float32", "geometry", "rolling_excl_current"),
    Column("family__days_since_last_pivot", "Int16", "geometry", "pivot_confirmed_lag"),
    # Retest_bounce (T2) features — populated when signal_state == retest_bounce.
    # breakout_age is also meaningful for extended (= asof - T1).
    Column("family__breakout_age", "Int16", "geometry", "current_bar_safe",
           notes="bars from T1 to asof; 0 for trigger; >0 for retest_bounce/extended"),
    Column("family__retest_depth_atr", "Float32", "geometry", "current_bar_safe",
           notes="(box_top - deepest_low_in_(T1,T2]) / atr_sq; positive = pulled below box_top, "
                 "negative = held above. NaN unless state == retest_bounce."),
    Column("family__retest_close_position", "Float32", "geometry", "current_bar_safe",
           notes="(close - box_top) / atr_sq on T2; how decisively the bounce reclaimed. "
                 "NaN unless state == retest_bounce."),
    Column("family__retest_vol_pattern", "Float32", "geometry", "current_bar_safe",
           notes="vol_T2 / mean(vol in (T1, T2)); >1 means the bounce had a relative "
                 "spike vs the pullback. NaN if asof == T1+1 or state != retest_bounce."),
    Column("family__retest_kind", "string", "geometry", "current_bar_safe",
           notes="taxonomy of T2 quality based on retest_depth_atr: deep_touch "
                 "(≥0.30 ATR below box_top), shallow_touch (0..0.30), no_touch "
                 "(low stayed above box_top — drift-up continuation, weakest "
                 "forward 20d). Empty string when state != retest_bounce. "
                 "Not a gate — kept for ML / downstream slicing."),
    Column("family__body_atr", "Float32", "geometry", "current_bar_safe",
           notes="trigger-bar body / atr_sq (= |close-open|/atr_sq at T1). "
                 "NaN when state == pre_breakout (no T1 yet). Always ≥ 0.35 "
                 "after V1.3.1 since that is the trigger gate floor."),
    Column("family__body_class", "string", "geometry", "current_bar_safe",
           notes="taxonomy of T1 body strength (V1.3.1): mid_body (0.35..0.65), "
                 "strict_body (0.65..1.05), large_body (≥1.05). Empty string "
                 "when state == pre_breakout. Not a gate — kept for ML / "
                 "downstream slicing. The 0.65 boundary is the legacy V1.2.9 "
                 "trigger floor preserved for audit."),
]

_ASCENDING_TRIANGLE: list[Column] = [
    Column("family__resistance_level", "Float32", "geometry", "rolling_excl_current"),
    Column("family__support_slope", "Float32", "geometry", "pivot_confirmed_lag"),
    Column("family__support_r2", "Float32", "geometry", "pivot_confirmed_lag"),
    Column("family__resistance_touch_count", "Int16", "geometry", "pivot_confirmed_lag"),
    Column("family__higher_low_count", "Int16", "geometry", "pivot_confirmed_lag"),
    Column("family__low_trendline_distance", "Float32", "geometry", "pivot_confirmed_lag"),
    Column("family__triangle_height_pct", "Float32", "geometry", "rolling_excl_current"),
    Column("family__triangle_height_atr", "Float32", "geometry", "rolling_excl_current"),
    Column("family__triangle_compression_ratio", "Float32", "geometry", "rolling_excl_current"),
    Column("family__apex_distance_bars", "Int16", "geometry", "current_bar_safe"),
    Column("family__last_pullback_depth_pct", "Float32", "geometry", "pivot_confirmed_lag"),
    Column("family__last_pullback_depth_atr", "Float32", "geometry", "pivot_confirmed_lag"),
    Column("family__higher_low_quality_score", "Float32", "geometry", "pivot_confirmed_lag"),
    Column("family__failed_resistance_attempt_count", "Int16", "geometry", "pivot_confirmed_lag"),
    Column("family__days_since_last_pivot", "Int16", "geometry", "pivot_confirmed_lag"),
]

_FALLING_CHANNEL: list[Column] = [
    Column("family__channel_upper_slope", "Float32", "geometry", "rolling_excl_current"),
    Column("family__channel_lower_slope", "Float32", "geometry", "rolling_excl_current"),
    Column("family__channel_parallelism_error", "Float32", "geometry", "rolling_excl_current"),
    Column("family__channel_r2", "Float32", "geometry", "rolling_excl_current"),
    Column("family__channel_width_pct", "Float32", "geometry", "rolling_excl_current"),
    Column("family__channel_width_atr", "Float32", "geometry", "rolling_excl_current"),
    Column("family__downtrend_duration_bars", "Int16", "geometry", "rolling_excl_current"),
    Column("family__decline_from_prior_high_pct", "Float32", "geometry", "pivot_confirmed_lag"),
    Column("family__lower_high_count", "Int16", "geometry", "pivot_confirmed_lag"),
    Column("family__lower_low_count", "Int16", "geometry", "pivot_confirmed_lag"),
    Column("family__selling_pressure_decay", "Float32", "geometry", "rolling_excl_current"),
    Column("family__volume_decay_downtrend", "Float32", "geometry", "rolling_excl_current"),
    Column("family__entry_distance_to_upper_band", "Float32", "geometry", "current_bar_safe"),
    Column("family__distance_to_channel_mid", "Float32", "geometry", "current_bar_safe"),
    Column("family__reclaim_sma20_flag", "boolean", "geometry", "current_bar_safe"),
    Column("family__reclaim_vwap52_flag", "boolean", "geometry", "current_bar_safe"),
    Column("family__last_low_undercut_flag", "boolean", "geometry", "pivot_confirmed_lag"),
    Column("family__last_low_reclaim_strength", "Float32", "geometry", "pivot_confirmed_lag"),
    Column("family__oversold_reversal_score", "Float32", "geometry", "current_bar_safe"),
    Column("family__days_since_last_pivot", "Int16", "geometry", "pivot_confirmed_lag"),
]

_VOLATILITY_SQUEEZE: list[Column] = [
    Column("family__bb_width", "Float32", "geometry", "current_bar_safe"),
    Column("family__bb_width_pctile_252", "Float32", "geometry", "current_bar_safe"),
    Column("family__atr_pctile_120", "Float32", "geometry", "current_bar_safe"),
    Column("family__donchian_width_20", "Float32", "geometry", "current_bar_safe"),
    Column("family__donchian_width_pctile_120", "Float32", "geometry", "current_bar_safe"),
    Column("family__squeeze_duration_bars", "Int16", "geometry", "rolling_excl_current"),
    Column("family__squeeze_depth_score", "Float32", "geometry", "rolling_excl_current"),
    Column("family__range_contraction_ratio", "Float32", "geometry", "rolling_excl_current"),
    Column("family__volume_contraction_ratio", "Float32", "geometry", "rolling_excl_current"),
    Column("family__expansion_bar_range_ratio", "Float32", "geometry", "current_bar_safe"),
    Column("family__expansion_bar_volume_ratio", "Float32", "geometry", "current_bar_safe"),
    Column("family__breakout_direction", "Int8", "geometry", "current_bar_safe",
           notes="+1 long / -1 short / 0 none"),
    Column("family__close_above_donchian_high_flag", "boolean", "geometry", "current_bar_safe"),
    Column("family__close_below_donchian_low_flag", "boolean", "geometry", "current_bar_safe"),
    Column("family__post_squeeze_range_position", "Float32", "geometry", "current_bar_safe"),
    Column("family__trend_bias_score", "Float32", "geometry", "current_bar_safe"),
]

_FAILED_BREAKDOWN_RECLAIM: list[Column] = [
    Column("family__support_level", "Float32", "geometry", "pivot_confirmed_lag"),
    Column("family__support_touch_count", "Int16", "geometry", "pivot_confirmed_lag"),
    Column("family__breakdown_depth_pct", "Float32", "geometry", "current_bar_safe"),
    Column("family__breakdown_depth_atr", "Float32", "geometry", "current_bar_safe"),
    Column("family__breakdown_duration_bars", "Int16", "geometry", "current_bar_safe"),
    Column("family__reclaim_strength_pct", "Float32", "geometry", "current_bar_safe"),
    Column("family__reclaim_range_position", "Float32", "geometry", "current_bar_safe"),
    Column("family__reclaim_volume_ratio", "Float32", "geometry", "current_bar_safe"),
    Column("family__close_back_above_support_pct", "Float32", "geometry", "current_bar_safe"),
    Column("family__bear_trap_score", "Float32", "geometry", "current_bar_safe"),
    Column("family__stop_sweep_volume_ratio", "Float32", "geometry", "current_bar_safe"),
    Column("family__days_since_breakdown", "Int16", "geometry", "current_bar_safe"),
    Column("family__lowest_close_below_support", "Float32", "geometry", "current_bar_safe"),
    Column("family__failed_breakdown_count_120", "Int16", "geometry", "pivot_confirmed_lag"),
    Column("family__support_reclaim_persistence_2bar", "Int8", "geometry", "current_bar_safe"),
]

_BREAKOUT_PULLBACK: list[Column] = [
    Column("family__original_breakout_date", "datetime64[ns]", "geometry", "current_bar_safe"),
    Column("family__original_breakout_level", "Float32", "geometry", "current_bar_safe"),
    Column("family__days_since_breakout", "Int16", "geometry", "current_bar_safe"),
    Column("family__breakout_strength_pct", "Float32", "geometry", "current_bar_safe"),
    Column("family__breakout_volume_ratio", "Float32", "geometry", "current_bar_safe"),
    Column("family__pullback_depth_pct", "Float32", "geometry", "current_bar_safe"),
    Column("family__pullback_depth_atr", "Float32", "geometry", "current_bar_safe"),
    Column("family__pullback_to_level_distance_pct", "Float32", "geometry", "current_bar_safe"),
    Column("family__pullback_volume_ratio", "Float32", "geometry", "current_bar_safe"),
    Column("family__pullback_volume_dryup", "Float32", "geometry", "current_bar_safe"),
    Column("family__retest_hold_flag", "boolean", "geometry", "current_bar_safe"),
    Column("family__retest_low_vs_breakout_level", "Float32", "geometry", "current_bar_safe"),
    Column("family__bounce_from_retest_pct", "Float32", "geometry", "current_bar_safe"),
    Column("family__bounce_range_position", "Float32", "geometry", "current_bar_safe"),
    Column("family__higher_low_after_breakout", "Int8", "geometry", "pivot_confirmed_lag"),
    Column("family__failed_retest_count", "Int16", "geometry", "current_bar_safe"),
]

_SQUEEZE_BREAKOUT_LOOSE_V0_DIAGNOSTIC: list[Column] = [
    # Trigger geometry
    Column("family__channel_high", "Float32", "geometry", "rolling_excl_current",
           notes="max(high) of squeeze run — trigger level (no robust quantile)"),
    Column("family__channel_low", "Float32", "geometry", "rolling_excl_current",
           notes="min(low) of squeeze run — for invalidation level"),
    Column("family__channel_width_pct", "Float32", "geometry", "rolling_excl_current"),
    Column("family__channel_width_atr", "Float32", "geometry", "rolling_excl_current"),
    Column("family__base_duration_weeks", "Float32", "geometry", "rolling_excl_current"),
    Column("family__base_slope", "Float32", "geometry", "rolling_excl_current",
           notes="close-line slope (per-day, fraction of price); audit only — no cap"),
    Column("family__resistance_slope", "Float32", "geometry", "rolling_excl_current",
           notes="high-line slope (per-day, fraction of price); audit only — no cap"),
    # Audit columns — required by user before family can graduate to production.
    Column("family__width_pct", "Float32", "audit", "rolling_excl_current",
           notes="width / asof_close — duplicate of channel_width_pct exposed under "
                 "this name for explicit audit / bucket reporting"),
    Column("family__slope_pct_per_day", "Float32", "audit", "rolling_excl_current",
           notes="max(|base_slope|, |resistance_slope|) — binding slope, audit-only, "
                 "no cap in this family"),
    Column("family__strict_overlap_flag", "boolean", "audit", "current_bar_safe",
           notes="True if strict V1.2.9 horizontal_base would also emit a trigger "
                 "on the same (ticker, asof). Used to separate loose-only from "
                 "strict-overlap candidates in validation slicing."),
    Column("family__strict_reject_reason_tags", "string", "audit", "current_bar_safe",
           notes="comma-joined list of strict-gate names that would reject this "
                 "loose candidate (e.g. 'body_low,vol_low,sma20_below'). Empty "
                 "string when strict_overlap_flag is True."),
]


# -- ICT/SMC families (V1.4.0, 1h bars) --------------------------------------
# Both families share a near-identical column shape: a price zone (high, low,
# width), a Break-of-Structure event that armed the zone, and a retest_bounce
# tail that mirrors horizontal_base's V1.2.9 retest taxonomy. Despite the
# shape similarity they are NOT collapsed into a shared block — zone semantics
# differ (MB = bearish-body inside an impulse-down leg; BB = bullish-body of a
# failed bearish OB after a sweep). Keeping them split lets each family carry
# its own provenance fields without smuggling MB-specific names into BB rows.

_MITIGATION_BLOCK: list[Column] = [
    # Zone (price-frozen at zone formation; never repaints)
    Column("family__zone_high", "Float32", "geometry", "current_bar_safe", nullable=False,
           notes="upper edge of the bullish MB zone. zone_kind=='body' uses "
                 "max(open, close) of the origin bearish bar; "
                 "zone_kind=='body_plus_wick' uses high of the origin bar."),
    Column("family__zone_low", "Float32", "geometry", "current_bar_safe", nullable=False,
           notes="lower edge of the bullish MB zone. zone_kind=='body' uses "
                 "min(open, close) of the origin bearish bar; "
                 "zone_kind=='body_plus_wick' uses low of the origin bar."),
    Column("family__zone_width_pct", "Float32", "geometry", "current_bar_safe"),
    Column("family__zone_width_atr", "Float32", "geometry", "current_bar_safe"),
    Column("family__zone_kind", "string", "geometry", "current_bar_safe",
           notes="taxonomy: 'body' (open/close span only — canonical ICT MB) or "
                 "'body_plus_wick' (full bar high/low — wider, more lenient). "
                 "Sub-classification, not a gate."),
    Column("family__zone_age_bars", "Int16", "geometry", "current_bar_safe",
           notes="bars from zone formation (= origin bearish bar) to asof."),
    # BoS (impulse leg that armed the zone)
    Column("family__bos_level", "Float32", "geometry", "pivot_confirmed_lag",
           notes="prior swing high broken by the impulse-up leg confirming the MB."),
    Column("family__bos_distance_atr", "Float32", "geometry", "current_bar_safe",
           notes="(close_at_bos - bos_level) / atr at BoS bar; impulse strength."),
    Column("family__impulse_leg_bars", "Int16", "geometry", "pivot_confirmed_lag",
           notes="length of the impulse-up leg from zone bar to BoS bar (inclusive)."),
    Column("family__impulse_leg_atr", "Float32", "geometry", "pivot_confirmed_lag",
           notes="(leg_high - leg_low) / atr at BoS bar; leg magnitude."),
    Column("family__leg_swing_low", "Float32", "geometry", "pivot_confirmed_lag",
           notes="lowest low of the impulse-down leg whose last bearish bar "
                 "defined this zone — invalidation reference."),
    # Touches & retest (retest_bounce / mitigation_touch states)
    Column("family__touches_into_zone", "Int16", "geometry", "current_bar_safe",
           notes="count of bars whose low ≤ zone_high (and high ≥ zone_low) "
                 "between BoS+1 and asof. ≥1 means zone has been mitigated."),
    Column("family__retest_depth_atr", "Float32", "geometry", "current_bar_safe",
           notes="(zone_high - deepest_low_in_(BoS, asof]) / atr_sq; positive = "
                 "wicked into zone, negative = stayed above. NaN when no touch."),
    Column("family__retest_close_position", "Float32", "geometry", "current_bar_safe",
           notes="(close - zone_high) / atr_sq on the retest_bounce bar; how "
                 "decisively the bounce reclaimed. NaN unless state == retest_bounce."),
    Column("family__retest_vol_pattern", "Float32", "geometry", "current_bar_safe",
           notes="vol_T2 / mean(vol in (BoS, T2)); >1 means the bounce had a "
                 "relative volume spike vs the pullback. NaN when N/A."),
    Column("family__retest_kind", "string", "geometry", "current_bar_safe",
           notes="taxonomy: deep_touch (≥0.30 ATR into zone), shallow_touch "
                 "(0..0.30), no_touch (low stayed above zone — drift-up). "
                 "Empty string when state ∈ {zone_armed}."),
]

_BREAKER_BLOCK: list[Column] = [
    # Zone (price-frozen at zone formation; never repaints)
    Column("family__zone_high", "Float32", "geometry", "current_bar_safe", nullable=False,
           notes="upper edge of the bullish BB zone — the high of the failed "
                 "bearish OB (last up-candle before the sweep+drop). After "
                 "the BoS this level flips from resistance to support."),
    Column("family__zone_low", "Float32", "geometry", "current_bar_safe", nullable=False,
           notes="lower edge of the bullish BB zone. zone_kind=='body' uses "
                 "min(open, close) of the origin up-bar; 'body_plus_wick' "
                 "uses low of the origin up-bar."),
    Column("family__zone_width_pct", "Float32", "geometry", "current_bar_safe"),
    Column("family__zone_width_atr", "Float32", "geometry", "current_bar_safe"),
    Column("family__zone_kind", "string", "geometry", "current_bar_safe",
           notes="'body' (canonical ICT breaker — body span only) or "
                 "'body_plus_wick' (wider). Sub-classification, not a gate."),
    Column("family__zone_age_bars", "Int16", "geometry", "current_bar_safe",
           notes="bars from zone formation (= origin up-bar) to asof."),
    # Sweep + BoS provenance
    Column("family__sweep_low", "Float32", "geometry", "pivot_confirmed_lag",
           notes="the new lower-low printed AFTER the bearish OB but BEFORE "
                 "the bullish BoS. This is the liquidity sweep that flips the "
                 "OB into a breaker."),
    Column("family__sweep_depth_atr", "Float32", "geometry", "pivot_confirmed_lag",
           notes="(prior_swing_low - sweep_low) / atr at sweep bar; depth of "
                 "the LL below the prior structural low."),
    Column("family__bos_distance_atr", "Float32", "geometry", "current_bar_safe",
           notes="(close_at_bos - zone_high) / atr at BoS bar; impulse strength "
                 "of the rally that broke the OB high and armed the breaker."),
    Column("family__impulse_leg_bars", "Int16", "geometry", "pivot_confirmed_lag",
           notes="bars from sweep_low bar to BoS bar (inclusive)."),
    Column("family__impulse_leg_atr", "Float32", "geometry", "pivot_confirmed_lag",
           notes="(close_at_bos - sweep_low) / atr at BoS bar; rally magnitude."),
    Column("family__prior_swing_low", "Float32", "geometry", "pivot_confirmed_lag",
           notes="the structural low that was swept — invalidation context."),
    # Touches & retest
    Column("family__touches_into_zone", "Int16", "geometry", "current_bar_safe",
           notes="count of bars whose low ≤ zone_high (and high ≥ zone_low) "
                 "between BoS+1 and asof."),
    Column("family__retest_depth_atr", "Float32", "geometry", "current_bar_safe",
           notes="(zone_high - deepest_low_in_(BoS, asof]) / atr_sq; positive = "
                 "wicked into zone, negative = stayed above. NaN when no touch."),
    Column("family__retest_close_position", "Float32", "geometry", "current_bar_safe",
           notes="(close - zone_high) / atr_sq on the retest_bounce bar."),
    Column("family__retest_vol_pattern", "Float32", "geometry", "current_bar_safe",
           notes="vol_T2 / mean(vol in (BoS, T2))."),
    Column("family__retest_kind", "string", "geometry", "current_bar_safe",
           notes="deep_touch / shallow_touch / no_touch — same convention as MB."),
]


FAMILY_COLUMNS: dict[str, list[Column]] = {
    "horizontal_base": _HORIZONTAL_BASE,
    "ascending_triangle": _ASCENDING_TRIANGLE,
    "falling_channel": _FALLING_CHANNEL,
    "volatility_squeeze": _VOLATILITY_SQUEEZE,
    "failed_breakdown_reclaim": _FAILED_BREAKDOWN_RECLAIM,
    "breakout_pullback": _BREAKOUT_PULLBACK,
    "squeeze_breakout_loose_v0_diagnostic": _SQUEEZE_BREAKOUT_LOOSE_V0_DIAGNOSTIC,
    "mitigation_block": _MITIGATION_BLOCK,
    "breaker_block": _BREAKER_BLOCK,
}

assert set(FAMILY_COLUMNS) == set(FAMILIES), "FAMILY_COLUMNS keys must equal FAMILIES"


# -- Score outputs ------------------------------------------------------------

SCORE_COLUMNS: list[Column] = [
    Column("rule_score", "Float32", "score", "current_bar_safe", nullable=False,
           notes="weighted sum of common_score and family_score; sort key only, not edge claim"),
    Column("common_score", "Float32", "score", "current_bar_safe"),
    Column("family_score", "Float32", "score", "current_bar_safe"),
    Column("signal_tags", "string", "score", "current_bar_safe",
           notes="JSON-encoded list[str] of fired subscore / quality tags"),
    Column("signal_reason_text", "string", "score", "current_bar_safe",
           notes="optional human-readable summary; never the primary filter column"),
]


# -- Rule-score weights — FROZEN PRIORS, do not tune to backtest -------------

RULE_WEIGHTS: dict[str, dict[str, float]] = {
    "horizontal_base": {
        "base_tightness_score": 20.0,            # was 25; 5 pt routed to atr quality
        "volatility_contraction_score": 5.0,     # V1.2.2: ATR ratio + decline (no longer a gate)
        "touch_quality_score": 15.0,
        "breakout_quality_score": 20.0,
        "volume_confirmation_score": 15.0,
        "trend_reclaim_score": 10.0,
        "relative_strength_score": 10.0,
        "liquidity_risk_score": 5.0,
        "extension_penalty": -10.0,
    },
    "ascending_triangle": {
        "resistance_quality_score": 20.0,
        "rising_lows_score": 20.0,
        "compression_score": 15.0,
        "breakout_quality_score": 20.0,
        "volume_confirmation_score": 10.0,
        "relative_strength_score": 10.0,
        "liquidity_risk_score": 5.0,
        "apex_late_penalty": -10.0,
        "extension_penalty": -10.0,
    },
    "falling_channel": {
        "channel_fit_score": 20.0,
        "seller_exhaustion_score": 15.0,
        "upper_breakout_score": 20.0,
        "reclaim_trend_score": 15.0,
        "entry_distance_score": 10.0,
        "relative_strength_recovery_score": 10.0,
        "liquidity_risk_score": 10.0,
        "dead_cat_penalty": -15.0,
        "overextension_penalty": -10.0,
    },
    "volatility_squeeze": {
        "squeeze_depth_score": 25.0,
        "squeeze_duration_score": 15.0,
        "expansion_quality_score": 20.0,
        "volume_expansion_score": 15.0,
        "trend_bias_score": 10.0,
        "relative_strength_score": 10.0,
        "liquidity_risk_score": 5.0,
        "whipsaw_penalty": -10.0,
        "extension_penalty": -10.0,
    },
    "failed_breakdown_reclaim": {
        "support_quality_score": 20.0,
        "breakdown_failure_score": 20.0,
        "reclaim_strength_score": 20.0,
        "volume_reversal_score": 15.0,
        "trap_quality_score": 10.0,
        "relative_strength_recovery_score": 10.0,
        "liquidity_risk_score": 5.0,
        "prolonged_breakdown_penalty": -10.0,
        "weak_reclaim_penalty": -10.0,
    },
    "breakout_pullback": {
        "original_breakout_quality_score": 15.0,
        "pullback_quality_score": 20.0,
        "retest_hold_score": 20.0,
        "bounce_confirmation_score": 15.0,
        "low_volume_pullback_score": 10.0,
        "trend_persistence_score": 10.0,
        "liquidity_risk_score": 10.0,
        "deep_pullback_penalty": -10.0,
        "stale_breakout_penalty": -10.0,
    },
    # Diagnostic family — no rule scoring yet; weights all 0 until the family
    # graduates and gets its own gate-validated weight set.
    "squeeze_breakout_loose_v0_diagnostic": {
        "diagnostic_placeholder": 0.0,
    },
    # ICT/SMC families — V1.4.0 priors. Same shape; weights are placeholders
    # until 3y-validation produces calibrated priors. Do NOT tune to backtest.
    "mitigation_block": {
        "zone_quality_score": 15.0,
        "bos_quality_score": 20.0,
        "retest_quality_score": 20.0,
        "volume_confirmation_score": 10.0,
        "trend_alignment_score": 10.0,
        "relative_strength_score": 10.0,
        "liquidity_risk_score": 5.0,
        "stale_zone_penalty": -10.0,
        "extension_penalty": -10.0,
    },
    "breaker_block": {
        "zone_quality_score": 15.0,
        "sweep_quality_score": 15.0,
        "bos_quality_score": 20.0,
        "retest_quality_score": 20.0,
        "volume_confirmation_score": 5.0,
        "trend_alignment_score": 10.0,
        "relative_strength_score": 10.0,
        "liquidity_risk_score": 5.0,
        "stale_zone_penalty": -10.0,
        "extension_penalty": -10.0,
    },
}

assert set(RULE_WEIGHTS) == set(FAMILIES), "RULE_WEIGHTS keys must equal FAMILIES"


# -- Helpers ------------------------------------------------------------------

def output_columns_for(family: str) -> list[Column]:
    """Full ordered column set a scanner row for `family` must carry."""
    if family not in FAMILY_COLUMNS:
        raise ValueError(f"unknown family: {family!r}")
    return [
        *IDENTITY_COLUMNS,
        *AUDIT_COLUMNS,
        *CONTRACT_COLUMNS,
        *COMMON_COLUMNS,
        *FAMILY_COLUMNS[family],
        *SCORE_COLUMNS,
    ]


def required_output_columns(family: str) -> list[str]:
    """Non-nullable columns for `family` — must be populated on every row."""
    return [c.name for c in output_columns_for(family) if not c.nullable]


def validate_output_columns(columns: Iterable[str]) -> None:
    """Reject scanner outputs that smuggle in label__ / model__ columns."""
    bad = [c for c in columns if c.startswith(FORBIDDEN_OUTPUT_PREFIXES)]
    if bad:
        raise ValueError(
            f"forbidden prefix in scanner output (label__/model__ reserved): {bad}"
        )
