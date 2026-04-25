"""Locked constants for SBT-1700.

Any change here invalidates the dataset, the trained ranker, and the
acceptance gate. Do NOT tune these silently — bump the schema version
and rebuild from scratch.
"""

from __future__ import annotations

from dataclasses import dataclass

SCHEMA_VERSION = "v1"

# 17:00 cutoff = last completed 15m bar with close ≤ 16:45 TR.
# Rationale: the 16:45-17:00 bar may not be finalized when the live cron
# fires at 17:00 TR. Use the last bar whose close is strictly < 17:00.
# Expected count: bars closing at 10:15, 10:30, ..., 16:30, 16:45 = 27 bars.
# (nyxexp uses 17:00 inclusive = 28 bars; SBT-1700 takes the safer cut.)
BAR_CUTOFF_HH = 16
BAR_CUTOFF_MM = 45
EXPECTED_BARS_PER_PAIR = 27

# Coverage policy
COVERAGE_DROP_THRESHOLD = 0.80   # drop rows with intraday_coverage < 80%
COVERAGE_PARTIAL_LOWER = 0.80    # 0.80 ≤ coverage < 0.95 → keep but flag


# ── E3 execution rule (frozen for v1) ─────────────────────────────────
@dataclass(frozen=True)
class E3Params:
    side: str = "long"            # LONG only in phase 1
    atr_window: int = 14          # ATR lookback (daily, prior days only)
    sl_atr_mult: float = 0.30     # stop = entry - 0.3 * atr_1700
    tp_R_mult: float = 1.0        # tp = entry + 1.0R
    timeout_bars: int = 5         # exit after 5 daily bars if neither hit
    failed_breakout_exit: bool = False
    use_trail: bool = False
    use_partial: bool = False
    # Worst-case-fill on a same-bar tag: if both High>=tp and Low<=stop hit
    # on the same daily bar, assume SL filled first. Conservative.
    worst_case_same_bar: bool = True
    # Slippage + commission, applied symmetrically per side (entry + exit).
    # 20 bps round-trip ≈ 10 bps per side. Net R subtracts this from gross.
    slippage_bps_per_side: float = 5.0
    commission_bps_per_side: float = 5.0


E3 = E3Params()


# ── Feature engineering windows ───────────────────────────────────────
# All windows use prior days only (T-1 and earlier are EOD-complete);
# T is patched with the 17:00-truncated bar.
PRIOR_EMA_SPANS = (20, 50, 100, 200)
PRIOR_ATR_WINDOW = 14
PRIOR_BB_WINDOW = 20
PRIOR_RETURNS_WINDOWS = (1, 5, 10, 20)
PRIOR_VOL_WINDOW = 20
PRIOR_52W_WINDOW = 252


# ── SBT signal detection (daily, identical to run_smart_breakout.py
# optimized params, but evaluated on 17:00-truncated T) ────────────────
SBT_BB_LENGTH = 20
SBT_BB_MULT = 2.0
SBT_BB_WIDTH_THRESH = 0.80
SBT_ATR_LENGTH = 10
SBT_ATR_SMA_LENGTH = 20
SBT_ATR_SQUEEZE_RATIO = 1.00
SBT_MIN_SQUEEZE_BARS = 5
SBT_MAX_SQUEEZE_BARS = 40
SBT_IMPULSE_ATR_MULT = 0.35
SBT_VOL_SMA_LENGTH = 20
SBT_VOL_MULT = 1.5
SBT_HTF_EMA_LENGTH = 50
