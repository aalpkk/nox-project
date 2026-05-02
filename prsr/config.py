"""Locked numerical constants for PRSR v1.

DO NOT MUTATE AFTER RUN-1. Any change here is a new spec version.
"""
from __future__ import annotations

VERSION = "1.0.0"

# -- Window
START_DATE = "2023-01-02"
END_DATE = "2026-04-29"

# -- Universe tiers (this module only)
HISTORY_BARS_CORE = 500
HISTORY_BARS_WATCHABLE_MIN = 250
LIQUIDITY_QUANTILE = 0.60   # cross-sectional q60 of median_turnover_60
LIQUIDITY_LOOKBACK = 60

# -- Pattern parameters
PATTERN_N = 20              # rolling support window
PATTERN_LOC_FLOOR = 0.65    # close location threshold
PATTERN_VOL_LOOKBACK = 60   # vol_med60 window
PATTERN_B_LOOKBACK = 5      # spring/reclaim history window

# -- Regime
ER_LOOKBACK = 60
ER_QUANTILE = 0.40          # cross-sectional q40 floor
ATR_LOOKBACK = 20
ATR_PCT_QUANTILE = 0.90     # cross-sectional q90 ceiling

# -- Oscillator (close-source HW; diagnostic only, NOT gating)
HW_RSI_LENGTH = 7
HW_SIG_LEN = 3
HW_LOWER_ZONE_THRESHOLD = 30.0
HW_LT_50 = 50.0
HWO_UP_LOOKBACK = 3         # HWO Up fired in [t-2, t]
HW_SLOPE_BARS = 3

# -- Stops / exits
INITIAL_STOP_ATR_MULT = 0.25
TIME_STOP_PRIMARY = 10
TIME_STOP_SIDE = (5, 20)
TP_R_MULTIPLE = 2.0         # TP appendix only

# -- Random baseline
BASELINE_SEED = 17

# -- Acceptance thresholds (LOCKED before run-1)
ACCEPT_PF_MIN = 1.30
ACCEPT_PF_OVER_RANDOM = 0.20
ACCEPT_REALIZED_MED_OVER_RANDOM_PP = 0.005   # 0.5 percentage points
ACCEPT_YEARLY_BEAT_RANDOM_MIN = 3            # of 4 yearly slices
ACCEPT_YEARLY_BEAT_RANDOM_OF = 4
ACCEPT_TOP5_DATE_SHARE_MAX = 0.30
ACCEPT_N_MIN = 200

# -- Files
MASTER_PARQUET = "output/extfeed_intraday_1h_3y_master.parquet"
OUT_PER_TRADE = "output/prsr_v1_per_trade.csv"
OUT_RANDOM = "output/prsr_v1_random_baseline.csv"
OUT_AGGREGATE = "output/prsr_v1_aggregate.csv"
OUT_CONCENTRATION = "output/prsr_v1_concentration.csv"
OUT_DRAWDOWN = "output/prsr_v1_drawdown.csv"
OUT_OSC_SUBSET = "output/prsr_v1_osc_subset.csv"
OUT_WATCHABLE = "output/prsr_v1_watchable_list.csv"
OUT_REPORT = "output/prsr_v1_report.md"
