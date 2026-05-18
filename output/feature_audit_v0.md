# Feature audit — v0

- panel rows: **1,800**
- panel columns: **562**
- tickers: **9** (GARAN, AKBNK, TUPRS, EREGL, BIMAS, SAHOL, SISE, YKBNK, FROTO)
- skipped (insufficient bars): KOZAL(0)
- date range: **2025-07-21 → 2026-04-30**

## Registry coverage

- features classified: **560** / 562
- UNKNOWN (not in registry, not date/ticker): **0**

### Group distribution

| group | count |
|---|---:|
| W | 40 |
| X | 36 |
| P-CS | 31 |
| AA | 30 |
| S | 28 |
| AF | 28 |
| AB | 28 |
| Z | 28 |
| Y | 28 |
| O | 25 |
| T | 24 |
| U | 24 |
| AI | 22 |
| V | 22 |
| AK | 20 |
| R-ticker | 18 |
| Q-sector | 13 |
| K | 13 |
| B | 12 |
| A | 10 |
| C | 10 |
| R-breadth | 10 |
| M | 8 |
| E | 8 |
| L | 7 |
| I | 7 |
| F | 6 |
| D | 6 |
| J | 5 |
| H | 5 |
| G | 5 |
| N | 3 |
| UNKNOWN | 2 |

### `known_at` distribution

| known_at | count |
|---|---:|
| T_close | 459 |
| T_close_panel | 54 |
| T_open | 41 |
| derived | 6 |
| unknown | 2 |

### Whitelist coverage (manifest vs. panel)

| whitelist | defined | present | missing |
|---|---:|---:|---:|
| F0_CORE_EXISTING | 130 | 130 | 0 |
| F1_CONTEXT_ADDED | 318 | 318 | 0 |
| F2_FULL_EXPERIMENTAL | 536 | 529 | 7 |
| T_CLOSE_FEATURES | 516 | 509 | 7 |
| T1_OPEN_FEATURES | 216 | 216 | 0 |
| EXCLUDE_FEATURES | 20 | 20 | 0 |

## NaN / inf

- features with NaN ratio > 0.50: **1**
- features with NaN ratio > 0.90: **1**

### Top-30 NaN ratio

| feature | nan_ratio |
|---|---:|
| market_tavan_count_10d | 1.000 |
| gap_continuation_rate_20d_lag1 | 0.208 |
| gap_continuation_rate_20d | 0.208 |
| prior_breakout_failure_rate | 0.057 |
| prior_breakout_failure_rate_lag1 | 0.057 |
| pullback_lower_wick_score_lag1 | 0.038 |
| pullback_lower_wick_score | 0.038 |
| pullback_volume_dryup_lag1 | 0.038 |
| pullback_volume_dryup | 0.038 |
| sector_momentum_accel | 0.025 |
| up_down_vol_ratio_20d | 0.017 |
| up_down_vol_ratio_20d_lag1 | 0.017 |
| gap_direction_lag1 | 0.000 |
| gap_stop_risk_lag1 | 0.000 |
| gap_atr_lag1 | 0.000 |
| gap_after_volume_dryup_lag1 | 0.000 |
| gap_volatility_20d_lag1 | 0.000 |
| gap_after_tavan_lag1 | 0.000 |
| hammer_flag_lag1 | 0.000 |
| gap_after_squeeze_lag1 | 0.000 |
| higher_low_after_breakout_lag1 | 0.000 |
| false_breakout_risk_score_lag1 | 0.000 |
| failed_breakout_count_60d_lag1 | 0.000 |
| date | 0.000 |
| exhaustion_gap_flag_lag1 | 0.000 |
| ema55_overextension_atr_lag1 | 0.000 |
| ema21_overextension_atr_lag1 | 0.000 |
| distance_from_vwap_proxy_lag1 | 0.000 |
| dist_to_support_20d_lag1 | 0.000 |
| dist_to_resistance_20d_lag1 | 0.000 |

### Infinities

- none.

## Constant / near-constant

- exactly constant (nunique<=1): **13**
  - market_tavan_count_10d, zero_volume_days_60d, capitulation_flag, stop_distance_atr, target_distance_atr, reward_risk_ratio, capitulation_flag_lag1, reward_risk_ratio_lag1, stop_distance_atr_lag1, target_distance_atr_lag1, zero_volume_days_60d_lag1, market_taban_count, sector_laggard_flag
- near-constant (nunique==2): **132**
  - ema_trend_up, supertrend_dir, pmax_dir, phase_above_ema21, htf_trend_up, wt_bullish, squeeze_on, obv_trend, swing_bias, structure_break, higher_low, near_40high, br_rsi_thrust, br_rsi_gradual, br_ad_proxy, br_ema_reclaim, rg_ema_above, rg_adx_rebound, rg_was_trending, gate_open, rt_ema_bull, rt_st_bull, rt_wk_trend_up, rt_cmf_pos, rt_rvol_high, rt_obv_slope_pos, rt_adx_slope_pos, rt_atr_expanding, rt_di_bull, pb_rsi_low
- zero-std numeric: **12**
  - zero_volume_days_60d, capitulation_flag, stop_distance_atr, target_distance_atr, reward_risk_ratio, capitulation_flag_lag1, reward_risk_ratio_lag1, stop_distance_atr_lag1, target_distance_atr_lag1, zero_volume_days_60d_lag1, market_taban_count, sector_laggard_flag

## High-correlation pairs (|corr| ≥ 0.95)

- total pairs above threshold: **255**

| feature_a | feature_b | corr |
|---|---|---:|
| rg_ema_above | rt_ema_bull | +1.000 |
| phase_above_ema21 | rg_ema_above | +1.000 |
| gap_pct | overnight_return_1d | +1.000 |
| ema55_dist_pct | pullback_to_ema55_dist | +1.000 |
| supertrend_dir | rt_st_bull | +1.000 |
| atr_pct | stop_distance_pct | +1.000 |
| atr_pct | target_distance_pct | +1.000 |
| atr_pct | trail_stop_k_suggested | +1.000 |
| market_pct_new_20d_high | market_pct_new_20d_low | -1.000 |
| close_position | close_location_value | +1.000 |
| close_position | breakout_close_strength | +1.000 |
| ema21_dist_pct | pullback_to_ema21_dist | +1.000 |
| phase_above_ema21 | rt_ema_bull | +1.000 |
| drawdown_20 | drawdown_20_pct | +1.000 |
| drawdown_20 | close_vs_20d_high_pct | +1.000 |
| drawdown_20 | dist_to_20d_high_pct | +1.000 |
| drawdown_20 | dist_to_resistance_20d | +1.000 |
| drawdown_20 | range_upper_dist_pct | +1.000 |
| drawdown_20 | pullback_depth_pct | +1.000 |
| vol_ratio_20 | breakout_volume_ratio | +1.000 |
| drawdown_20_pct | close_vs_20d_high_pct | +1.000 |
| drawdown_20_pct | dist_to_20d_high_pct | +1.000 |
| drawdown_20_pct | dist_to_resistance_20d | +1.000 |
| drawdown_20_pct | range_upper_dist_pct | +1.000 |
| drawdown_20_pct | pullback_depth_pct | +1.000 |
| pivot_delta_pct | dist_to_pivot_low_pct | +1.000 |
| is_tavan | tavan_streak | +1.000 |
| is_tavan | tavan_locked | +1.000 |
| is_tavan | tavan_x_recent_tavan | +1.000 |
| tavan_streak | tavan_locked | +1.000 |
