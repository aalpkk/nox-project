"""
Feature availability manifest for ml/features.py output.

This module classifies every per-ticker feature emitted by
`compute_all_features` so downstream model code can answer:

  * Which group does this feature belong to (A..AK)?
  * When is it known — at T-open, T-close, or derived?
  * What inputs does it require — XU100 hydration?
  * Is it safe to consume in a T-close model (signal_asof = T close)?
  * Is it safe to consume in a T+1-open model (executable at next open)?
  * Was an automatic _lag1 variant generated for it?
  * Is it a direct alias of another feature?
  * Should the model be allowed to consume it directly (vs. only via lag)?

The classification is rule-based on the feature name (prefix/suffix/explicit
sets) — there is no runtime introspection of feature semantics. If a feature
name changes in `features.py`, update the rules here in lock-step.

The cross-sectional aggregator (`compute_cross_sectional_features`) is not yet
shipped — `cs_rank_*` / `cs_z_*` / `cs_pctile_*` features are still detected by
prefix here so the framework is ready to absorb them in a follow-up commit
without a schema break.

Whitelists for tiered model training are also defined here:
  * F0_CORE_EXISTING:    A..O groups that existed before the P-AK expansion
  * F1_CONTEXT_ADDED:    F0 + per-ticker P-AK groups (S/T/U/R/V/W/X/Y/Z/AA/AB/AF/AI/AK)
  * F2_FULL_EXPERIMENTAL: F1 + every auto-_lag1 sibling (cross-sectional groups
                          land in a follow-up commit)

  * T_CLOSE_FEATURES:    features known at end of day T (the default model surface)
  * T1_OPEN_FEATURES:    subset usable when scoring at next-day open
  * EXCLUDE_FEATURES:    leakage-risky or raw-price / unscaled features the
                         model should never see directly

Usage:
    from ml.feature_registry import build_feature_manifest, T_CLOSE_FEATURES
    manifest = build_feature_manifest(list(panel.columns))
    safe = [c for c in panel.columns if c in T_CLOSE_FEATURES]
"""

from __future__ import annotations

import re
from typing import Iterable

import pandas as pd


# ────────────────────────────────────────────────────────────────────────────
# Group membership — explicit per-feature dictionaries.
# Keys are bare (non-_lag1) feature names. The _lag1 variants are detected at
# classification time and inherit the group of their parent.
# ────────────────────────────────────────────────────────────────────────────

_GROUP_A = {  # A. Fiyat yapısı
    'close', 'returns_1d', 'returns_5d', 'returns_10d',
    'close_position', 'gap_pct', 'ema21_dist_pct', 'ema55_dist_pct',
}

_GROUP_B = {  # B. Trend
    'adx_14', 'adx_slope_5', 'plus_di', 'minus_di', 'di_spread',
    'ema_trend_up', 'supertrend_dir', 'pmax_dir', 'phase_above_ema21',
    'htf_adx', 'htf_adx_slope', 'htf_trend_up',
}

_GROUP_C = {  # C. Momentum
    'rsi_14', 'rsi_2',
    'macd_line', 'macd_signal', 'macd_hist',
    'wt1', 'wt2', 'wt_bullish',
    'squeeze_on', 'squeeze_mom',
}

_GROUP_D = {  # D. Volatilite
    'atr_14', 'atr_pct', 'bb_pctb', 'bb_width', 'drawdown_20', 'daily_move_atr',
}

_GROUP_E = {  # E. Hacim
    'vol_ratio_20', 'vol_ratio_30',
    'cmf_20', 'obv_trend', 'obv_slope_5', 'mfi_14', 'consecutive_green',
}

_GROUP_F = {  # F. SMC
    'swing_bias', 'bos_age', 'choch_age',
    'structure_break', 'higher_low', 'near_40high',
}

_GROUP_G = {  # G. Q score
    'q_rvol_s', 'q_clv_s', 'q_wick_s', 'q_range_s', 'q_total',
}

_GROUP_H = {  # H. NW breadth primitives
    'br_rsi_thrust', 'br_rsi_gradual', 'br_ad_proxy', 'br_ema_reclaim', 'br_score',
}

_GROUP_I = {  # I. NW regime primitives
    'rg_slope_score', 'rg_di_score', 'rg_ema_above', 'rg_adx_rebound',
    'rg_was_trending', 'rg_score', 'gate_open',
}

_GROUP_J = {  # J. Sell severity primitives
    'red_count', 'drawdown_20_pct', 'decline_5d_atr',
    'sell_severity', 'pivot_delta_pct',
}

_GROUP_K = {  # K. RT TPE primitives
    'rt_ema_bull', 'rt_st_bull', 'rt_wk_trend_up',
    'rt_cmf_pos', 'rt_rvol_high', 'rt_obv_slope_pos',
    'rt_adx_slope_pos', 'rt_atr_expanding', 'rt_di_bull',
    'trend_score', 'participation_score', 'expansion_score', 'tpe_total',
}

_GROUP_L = {  # L. RT regime + entry + OE + exit
    'regime_score', 'entry_score', 'oe_score',
    'pb_ema_dist', 'pb_rsi_low',
    'exit_stage', 'days_in_trade',
}

_GROUP_M = {  # M. Tavan primitives
    'is_tavan', 'tavan_streak', 'close_to_high', 'tavan_locked',
    'hit_tavan_intraday', 'recent_tavan_10d',
}

_GROUP_N = {  # N. Relatif güç (XU100-bağımlı)
    'rs_10', 'rs_60', 'rs_composite',
}

_GROUP_O = {  # O. Pre-breakout (incl. O7 lag-1 in features.py)
    'range_contraction_5_20', 'range_contraction_5_40',
    'atr_contraction_5_20', 'bb_width_pctile_20',
    'vol_dryup_ratio', 'vol_surge_today', 'vol_acceleration',
    'tl_volume_20d_avg', 'vol_pattern_score',
    'rsi_momentum_5d', 'macd_hist_accel',
    'consecutive_higher_close', 'close_vs_20d_high_pct',
    'dist_to_52w_high_pct', 'dist_to_20d_high_pct',
    'near_52w_high', 'price_range_position_20',
    'near_tavan_miss', 'recent_near_tavan_5d', 'max_daily_ret_5d',
    'rs_acceleration', 'market_tavan_count_10d',
    # O7 — hand-written lag-1 variants live here, not in the auto-lag-1 set
    'returns_1d_lag1', 'close_position_lag1', 'close_to_high_lag1',
    'max_daily_ret_5d_lag1', 'recent_near_tavan_5d_lag1',
    'vol_surge_yesterday', 'consecutive_green_lag1', 'recent_tavan_10d_lag1',
}

_GROUP_S = {  # S. Volatility regime transitions
    'atr_pct_rank_60', 'atr_pct_rank_252',
    'realized_vol_5d', 'realized_vol_10d', 'realized_vol_20d',
    'vol_of_vol_20d',
    'atr_expansion_ratio_5_20', 'atr_expansion_ratio_10_40',
    'bb_width_expansion_5d',
    'range_expansion_today', 'range_expansion_5d',
    'volatility_compression_score', 'volatility_release_score',
    'volatility_regime_change',
}

_GROUP_T = {  # T. Return distribution
    'ret_skew_20d', 'ret_skew_60d', 'ret_kurt_20d', 'ret_kurt_60d',
    'max_up_day_20d', 'max_down_day_20d',
    'up_down_vol_ratio_20d', 'positive_day_ratio_20d',
    'large_green_count_20d', 'large_red_count_20d',
    'tail_risk_score', 'lottery_profile_score',
}

_GROUP_U = {  # U. Liquidity / tradability
    'tl_volume_5d_avg', 'tl_volume_60d_avg', 'tl_volume_z_20',
    'volume_consistency_20d', 'zero_volume_days_60d',
    'amihud_illiq_20d', 'amihud_illiq_60d',
    'spread_proxy', 'impact_cost_proxy',
    'min_tl_volume_20d', 'liquidity_drop_flag', 'tradability_score',
}

_GROUP_R_TICKER = {  # R. XU100-derived per-ticker (market-breadth ayrı aggregator'da)
    'xu100_return_1d', 'xu100_return_5d',
    'xu100_above_ema21', 'xu100_above_ema55',
    'xu100_adx_14', 'xu100_bb_width', 'xu100_atr_pct',
    'xu100_drawdown_20', 'xu100_regime_score',
}

_GROUP_V = {  # V. Gap behaviour
    'gap_atr', 'gap_direction',
    'gap_after_squeeze', 'gap_after_volume_dryup', 'gap_after_tavan',
    'avg_gap_5d', 'gap_volatility_20d',
    'overnight_return_1d', 'intraday_return_1d',
    'overnight_vs_intraday_20d', 'gap_continuation_rate_20d',
}

_GROUP_W = {  # W. Candle anatomy
    'body_pct_range', 'upper_wick_pct_range', 'lower_wick_pct_range',
    'body_atr', 'range_atr',
    'close_location_value', 'open_location_value',
    'bullish_engulfing_flag', 'bearish_engulfing_flag',
    'hammer_flag', 'shooting_star_flag',
    'inside_bar_flag', 'outside_bar_flag',
    'nr7_flag', 'nr4_flag', 'wide_range_bar_flag',
    'close_above_prev_high', 'close_below_prev_low',
    'reversal_from_low_atr', 'rejection_from_high_atr',
}

_GROUP_X = {  # X. Support/resistance/pivot geometry
    'dist_to_pivot_high_pct', 'dist_to_pivot_low_pct',
    'dist_to_resistance_20d', 'dist_to_support_20d',
    'resistance_touch_count_20d', 'support_touch_count_20d',
    'failed_breakout_count_60d', 'successful_breakout_count_60d',
    'pivot_high_age', 'pivot_low_age',
    'range_mid_dist_pct', 'range_upper_dist_pct', 'range_lower_dist_pct',
    'base_duration_bars', 'base_width_pct', 'base_width_atr',
    'base_slope', 'base_quality_score',
}

_GROUP_Y = {  # Y. Breakout quality
    'breakout_distance_pct', 'breakout_distance_atr',
    'breakout_volume_ratio', 'breakout_close_strength',
    'breakout_body_atr', 'breakout_upper_wick_pct',
    'prior_breakout_failure_rate',
    'breakout_from_squeeze_flag', 'breakout_from_accumulation_flag',
    'breakout_near_52w_high',
    'breakout_above_20d_high', 'breakout_above_60d_high', 'breakout_above_252d_high',
    'false_breakout_risk_score',
}

_GROUP_Z = {  # Z. Pullback / retest
    'pullback_depth_atr', 'pullback_depth_pct',
    'pullback_duration', 'pullback_volume_dryup',
    'pullback_lower_wick_score',
    'pullback_to_ema21_dist', 'pullback_to_ema55_dist',
    'pullback_to_breakout_level_dist',
    'retest_hold_flag', 'retest_break_flag',
    'retest_age', 'retest_quality_score',
    'higher_low_after_breakout', 'shakeout_flag',
}

_GROUP_AA = {  # AA. Mean reversion / exhaustion
    'rsi_2_percentile_252', 'rsi_14_percentile_252',
    'distance_from_vwap_proxy',
    'ema21_overextension_atr', 'ema55_overextension_atr',
    'bb_upper_extension_pct',
    'consecutive_up_days', 'consecutive_down_days',
    'climax_volume_flag', 'climax_return_flag',
    'exhaustion_gap_flag', 'blowoff_top_risk',
    'capitulation_flag',
    'mean_reversion_long_score', 'mean_reversion_short_risk',
}

_GROUP_AB = {  # AB. Multi-timeframe (weekly + monthly proxy)
    'weekly_return_1w', 'weekly_return_4w',
    'weekly_rsi_14', 'weekly_macd_hist',
    'weekly_close_position', 'weekly_bb_width', 'weekly_atr_pct',
    'weekly_near_52w_high', 'weekly_breakout_flag', 'weekly_regime_score',
    'daily_weekly_alignment',
    'monthly_return_1m', 'monthly_trend_up', 'monthly_close_position',
}

_GROUP_AF = {  # AF. Calendar / seasonality
    'day_of_week', 'month_of_year', 'week_of_month',
    'month_start_flag', 'month_end_flag', 'quarter_end_flag',
    'days_from_month_start', 'days_to_month_end',
    'pre_holiday_flag', 'post_holiday_flag', 'short_week_flag',
    'earnings_season_proxy', 'index_rebalance_window', 'tax_loss_window_proxy',
}

_GROUP_AI = {  # AI. Risk / stop / target geometry
    'stop_distance_pct', 'stop_distance_atr',
    'target_distance_pct', 'target_distance_atr',
    'reward_risk_ratio',
    'nearest_support_stop_dist', 'nearest_resistance_target_dist',
    'atr_stop_viability', 'gap_stop_risk',
    'position_size_score', 'trail_stop_k_suggested',
}

_GROUP_AK = {  # AK. Interaction
    'rvol_x_close_position', 'rvol_x_breakout_distance',
    'squeeze_x_volume_surge',
    'market_regime_x_signal',
    'atr_x_gap', 'trend_x_pullback_depth',
    'overextension_x_volume_climax',
    'near_high_x_rvol', 'tavan_x_recent_tavan',
    'liquidity_x_volatility',
}


_ALL_BASE_GROUPS = {
    'A': _GROUP_A, 'B': _GROUP_B, 'C': _GROUP_C, 'D': _GROUP_D, 'E': _GROUP_E,
    'F': _GROUP_F, 'G': _GROUP_G, 'H': _GROUP_H, 'I': _GROUP_I, 'J': _GROUP_J,
    'K': _GROUP_K, 'L': _GROUP_L, 'M': _GROUP_M, 'N': _GROUP_N, 'O': _GROUP_O,
    'S': _GROUP_S, 'T': _GROUP_T, 'U': _GROUP_U, 'R-ticker': _GROUP_R_TICKER,
    'V': _GROUP_V, 'W': _GROUP_W, 'X': _GROUP_X, 'Y': _GROUP_Y, 'Z': _GROUP_Z,
    'AA': _GROUP_AA, 'AB': _GROUP_AB, 'AF': _GROUP_AF, 'AI': _GROUP_AI, 'AK': _GROUP_AK,
}

# Groups that get auto-_lag1 variants by features.py (the P-AK marker range).
# F0 (A..O) does NOT auto-lag; O7 lag-1 features are hand-rolled.
_AUTO_LAG_GROUPS = {
    'S', 'T', 'U', 'R-ticker', 'V', 'W', 'X', 'Y', 'Z',
    'AA', 'AB', 'AF', 'AI', 'AK',
}

# Groups that depend on XU100 hydration.
_REQUIRES_XU100_GROUPS = {'N', 'R-ticker'}
# AK depends on XU100 indirectly through market_regime_x_signal — flag that one only.
_REQUIRES_XU100_EXTRA = {'market_regime_x_signal'}

# Cross-sectional / sector group registration is reserved for a follow-up
# commit that wires `compute_cross_sectional_features`. The `cs_*` prefix
# detection below still classifies any panel features that ship before then.
_REQUIRES_PANEL_GROUPS: set[str] = set()
_REQUIRES_SECTOR_MAP_GROUPS: set[str] = set()

# Features known at T-open (computable when the day-T bar is still forming or
# at the open print). Mostly calendar / overnight / gap-direction.
_T_OPEN_FEATURES = {
    'gap_pct', 'gap_atr', 'gap_direction',
    'overnight_return_1d',
    'gap_after_squeeze', 'gap_after_volume_dryup', 'gap_after_tavan',
    # Calendar — pure date functions
    'day_of_week', 'month_of_year', 'week_of_month',
    'month_start_flag', 'month_end_flag', 'quarter_end_flag',
    'days_from_month_start', 'days_to_month_end',
    'pre_holiday_flag', 'post_holiday_flag', 'short_week_flag',
    'earnings_season_proxy', 'index_rebalance_window', 'tax_loss_window_proxy',
}

# Features that are derived constants (no time dependence) — flag for caution
# even though they are technically "known".
_DERIVED_CONSTANT_FEATURES = {
    'stop_distance_atr', 'target_distance_atr', 'reward_risk_ratio',
}

# Aliases — features that are SOURCE-LEVEL identical formulas (not just
# data-driven high correlation). Verified against ml/features.py by reading the
# assignment expressions; correlation evidence is necessary but not sufficient.
# All aliased features are added to EXCLUDE_FEATURES automatically.
_ALIASES = {
    # (c - l) / (h - l) — three names for the same candle-location ratio.
    'close_location_value': 'close_position',           # ml/features.py L901
    'breakout_close_strength': 'close_position',        # ml/features.py L982
    # (o - c_prev) / c_prev * 100 == (o / c_prev - 1) * 100 algebraically.
    'overnight_return_1d': 'gap_pct',                   # ml/features.py L875
    # (c - rolling20_high) / rolling20_high * 100 — same formula, same source series.
    'drawdown_20_pct': 'drawdown_20',                   # ml/features.py L534
    'dist_to_20d_high_pct': 'close_vs_20d_high_pct',    # ml/features.py L710 (literal copy)
    'dist_to_resistance_20d': 'close_vs_20d_high_pct',  # ml/features.py L939
    'range_upper_dist_pct': 'close_vs_20d_high_pct',    # ml/features.py L962
    # Note: drawdown_20 and close_vs_20d_high_pct are mathematically identical
    # but historically distinct names (one in group D, one in O); kept as two
    # separate canonicals by deliberate choice.
}

# Features the model should NOT consume directly (use rank/z/normalized form
# or lag-1 instead). Raw price / target-derived / leakage-risky / static
# configuration constants that carry no information.
_DO_NOT_USE_DIRECT = {
    'close',                # raw price — not cross-stock comparable
    'stop_distance_pct',    # depends on close (raw)
    'target_distance_pct',  # depends on close (raw); also same as target proxy
    'tl_volume_20d_avg',    # raw TL volume — scale heterogeneous; use cs_rank
    'tl_volume_5d_avg', 'tl_volume_60d_avg', 'min_tl_volume_20d',
    # Static configuration constants — _STOP_K / _TGT_K / TGT_K/STOP_K /
    # adaptive-trail-K. Carry no information until dynamic stop/target is wired
    # in; until then they are zero-variance noise for the model.
    'stop_distance_atr', 'target_distance_atr', 'reward_risk_ratio',
    'trail_stop_k_suggested',
    # Externally set (set to NaN inside compute_all_features and populated
    # downstream by a panel-level step). Never use as a model input directly.
    'market_tavan_count_10d',
}


_LAG1_SUFFIX_RE = re.compile(r'_lag1$')
_CS_PREFIX_RE = re.compile(r'^cs_(rank|z|pctile)_')


def _classify_one(name: str) -> dict:
    """Classify a single feature name. Returns a dict of registry columns."""
    is_lag = bool(_LAG1_SUFFIX_RE.search(name))
    parent = _LAG1_SUFFIX_RE.sub('', name) if is_lag else name

    # Cross-sectional features have a deterministic prefix. Even though the
    # aggregator is not yet shipped, classify by prefix so a forward-ship
    # works without touching this file.
    cs_match = _CS_PREFIX_RE.match(parent)
    if cs_match:
        return {
            'feature_name': name,
            'group': 'P-CS',
            'known_at': 'T_close_panel',
            'requires_market_panel': True,
            'requires_sector_map': False,
            'requires_xu100': False,
            'safe_for_t_close': True,
            'safe_for_t1_open': is_lag,
            'auto_lagged': False,
            'allow_model_direct': True,
            'is_lag1': is_lag,
            'parent_feature': parent if is_lag else None,
            'alias_of': None,
            'notes': 'Cross-sectional per-date aggregate; requires full panel slice.',
        }

    group_label = None
    for label, members in _ALL_BASE_GROUPS.items():
        if parent in members:
            group_label = label
            break

    if group_label is None:
        return {
            'feature_name': name,
            'group': 'UNKNOWN',
            'known_at': 'unknown',
            'requires_market_panel': False,
            'requires_sector_map': False,
            'requires_xu100': False,
            'safe_for_t_close': False,
            'safe_for_t1_open': False,
            'auto_lagged': False,
            'allow_model_direct': False,
            'is_lag1': is_lag,
            'parent_feature': parent if is_lag else None,
            'alias_of': _ALIASES.get(parent),
            'notes': 'Not present in feature registry — update ml/feature_registry.py.',
        }

    if parent in _T_OPEN_FEATURES:
        known_at = 'T_open'
    elif group_label in _REQUIRES_PANEL_GROUPS:
        known_at = 'T_close_panel'
    elif parent in _DERIVED_CONSTANT_FEATURES:
        known_at = 'derived'
    else:
        known_at = 'T_close'

    requires_panel = group_label in _REQUIRES_PANEL_GROUPS
    requires_sector_map = group_label in _REQUIRES_SECTOR_MAP_GROUPS
    requires_xu100 = (
        group_label in _REQUIRES_XU100_GROUPS
        or parent in _REQUIRES_XU100_EXTRA
    )

    auto_lagged = group_label in _AUTO_LAG_GROUPS and not is_lag

    safe_t_close = True
    if is_lag:
        safe_t1_open = True
    elif known_at == 'T_open':
        safe_t1_open = True
    elif known_at == 'derived':
        safe_t1_open = True
    else:
        safe_t1_open = False

    allow_direct = parent not in _DO_NOT_USE_DIRECT
    alias = _ALIASES.get(parent)

    notes = []
    if parent in _DO_NOT_USE_DIRECT:
        notes.append('Raw / unscaled — prefer cross-sectional rank or _lag1.')
    if alias is not None:
        notes.append(f'Alias of {alias}.')
    if requires_xu100:
        notes.append('Depends on XU100 hydration.')
    if requires_sector_map:
        notes.append('Depends on tools/sector_map.json.')
    if requires_panel:
        notes.append('Computed via compute_cross_sectional_features (full panel).')
    if known_at == 'T_open':
        notes.append('Computable at T+1 open.')

    return {
        'feature_name': name,
        'group': group_label,
        'known_at': known_at,
        'requires_market_panel': requires_panel,
        'requires_sector_map': requires_sector_map,
        'requires_xu100': requires_xu100,
        'safe_for_t_close': safe_t_close,
        'safe_for_t1_open': safe_t1_open,
        'auto_lagged': auto_lagged,
        'allow_model_direct': allow_direct,
        'is_lag1': is_lag,
        'parent_feature': parent if is_lag else None,
        'alias_of': alias,
        'notes': '; '.join(notes) if notes else '',
    }


def build_feature_manifest(feature_names: Iterable[str]) -> pd.DataFrame:
    """
    Build a manifest DataFrame describing every feature.

    Parameters
    ----------
    feature_names : iterable of str
        Names of all feature columns (per-ticker output of
        compute_all_features; cross-sectional / sector / macro additions
        land in a follow-up commit).

    Returns
    -------
    pd.DataFrame
        Columns: feature_name, group, known_at, requires_market_panel,
        requires_sector_map, requires_xu100, safe_for_t_close,
        safe_for_t1_open, auto_lagged, allow_model_direct, is_lag1,
        parent_feature, alias_of, notes.
    """
    rows = [_classify_one(name) for name in feature_names]
    df = pd.DataFrame(rows)
    cols = [
        'feature_name', 'group', 'known_at',
        'requires_market_panel', 'requires_sector_map', 'requires_xu100',
        'safe_for_t_close', 'safe_for_t1_open',
        'auto_lagged', 'allow_model_direct',
        'is_lag1', 'parent_feature', 'alias_of', 'notes',
    ]
    return df[cols]


# ────────────────────────────────────────────────────────────────────────────
# Whitelists for tiered model training.
# These are bare feature-name sets — intersect with actual panel columns at
# the call site.
# ────────────────────────────────────────────────────────────────────────────

F0_CORE_EXISTING: frozenset[str] = frozenset(
    _GROUP_A | _GROUP_B | _GROUP_C | _GROUP_D | _GROUP_E
    | _GROUP_F | _GROUP_G | _GROUP_H | _GROUP_I | _GROUP_J
    | _GROUP_K | _GROUP_L | _GROUP_M | _GROUP_N | _GROUP_O
)

F1_CONTEXT_ADDED: frozenset[str] = frozenset(
    F0_CORE_EXISTING
    | _GROUP_S | _GROUP_T | _GROUP_U | _GROUP_R_TICKER
    | _GROUP_V | _GROUP_W | _GROUP_X | _GROUP_Y | _GROUP_Z
    | _GROUP_AA | _GROUP_AB | _GROUP_AF | _GROUP_AI | _GROUP_AK
)

# F2 adds every auto-_lag1 sibling. Cross-sectional / sector / macro features
# join F2 in a follow-up commit when the aggregator ships.
_F2_AUTO_LAG_NAMES: frozenset[str] = frozenset(
    f'{name}_lag1'
    for group in _AUTO_LAG_GROUPS
    for name in _ALL_BASE_GROUPS[group]
)

F2_FULL_EXPERIMENTAL: frozenset[str] = frozenset(
    F1_CONTEXT_ADDED
    | _F2_AUTO_LAG_NAMES
)

T_CLOSE_FEATURES: frozenset[str] = frozenset(
    name for name in (F2_FULL_EXPERIMENTAL)
    if name not in _DO_NOT_USE_DIRECT
    and name not in _ALIASES
)

T1_OPEN_FEATURES: frozenset[str] = frozenset(
    name for name in (F2_FULL_EXPERIMENTAL)
    if (
        name.endswith('_lag1')
        or name in _T_OPEN_FEATURES
        or name in _DERIVED_CONSTANT_FEATURES
    )
    and name not in _DO_NOT_USE_DIRECT
)

EXCLUDE_FEATURES: frozenset[str] = frozenset(
    _DO_NOT_USE_DIRECT | set(_ALIASES.keys())
)


__all__ = [
    'build_feature_manifest',
    'F0_CORE_EXISTING',
    'F1_CONTEXT_ADDED',
    'F2_FULL_EXPERIMENTAL',
    'T_CLOSE_FEATURES',
    'T1_OPEN_FEATURES',
    'EXCLUDE_FEATURES',
]
