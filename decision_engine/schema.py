"""Decision Engine v0 — schema, dtypes, taxonomy, closed reason-code list.

Locked spec: memory/decision_engine_v0_spec.md
"""

from __future__ import annotations

from dataclasses import dataclass

# ─── final_action taxonomy ───────────────────────────────────────────────
FINAL_ACTIONS = (
    "TRADEABLE",
    "WAIT_TRIGGER",
    "WAIT_RETEST",
    "WATCHLIST",
    "AVOID",
    "EXIT_WARNING",
)

# ─── phase taxonomy ──────────────────────────────────────────────────────
PHASES = (
    "early_setup",
    "trigger",
    "retest_pending",
    "retest",
    "continuation",
    "accepted_continuation",
    "reversal",
    "extended",
    "exit_warning",
    "strength_context",
)

# ─── reason codes (closed list) ──────────────────────────────────────────
REASON_CODES_POSITIVE = (
    "regime_ok",
    "risk_ok",
    "retest_confirmed",
    "clean_trigger",
    "clean_breakout",
    "reversal_confirmed",
    "weekly_support",
    "accepted_horizon_h1_20d",
    "liquidity_ok",
    "higher_tf_aligned",
)

REASON_CODES_NEGATIVE = (
    "no_stop_ref",
    "no_entry_ref",
    "risk_too_wide",
    "late_chase",
    "avoid_late",
    "extended",
    "extension_high",
    "regime_mismatch",
    "nox_sat_conflict",
    "oe_high",
    "low_liquidity_below_floor",
    "liquidity_low",
    "gap_too_large",
    "pre_breakout",
    "zone_armed",
    "mit_touch_context_only",
    "nox_rt_daily_weak_standalone",
    "h2_failed_no_15d_rule",
    "continuation_no_prior",
    "strength_context_only",
    "unmatched_default",
    "fill_realism_unresolved",
    "horizon_review_due",
    "horizon_status_stale",
)

REASON_CODES_ALL = REASON_CODES_POSITIVE + REASON_CODES_NEGATIVE
_REASON_SET = set(REASON_CODES_ALL)


def assert_reason(code: str) -> str:
    if code not in _REASON_SET:
        raise ValueError(
            f"reason code {code!r} not in v0 closed list — adding requires reopening spec"
        )
    return code


# ─── normalized event schema (decision_signal_events.parquet) ────────────
EVENT_COLUMNS = [
    "date",
    "ticker",
    "source",
    "family",
    "state",
    "phase",
    "timeframe",
    "direction",
    "raw_signal_present",
    "entry_ref",
    "stop_ref",
    "risk_pct",
    "risk_atr",
    "extension_atr",
    "liquidity_score",
    "regime",
    "regime_stale_days",
    "higher_tf_context",
    "lower_tf_context",
    "expected_horizon",
    "horizon_source",
    "horizon_status",
    "horizon_accepted_date",
    "horizon_review_due",
    "reason_candidates",
    "raw_score",
    # fill-realism reserved (v0 leaves null/unresolved)
    "fill_assumption",
    "bar_timestamp",
    "auction_window_flag",
    "volume_at_trigger",
    "next_open_gap",
]

# ─── action schema (decision_actions.parquet) ────────────────────────────
ACTION_COLUMNS = [
    "date",
    "ticker",
    "source",
    "family",
    "phase",
    "timeframe",
    "final_action",
    "reason_codes",
    "entry_ref",
    "stop_ref",
    "risk_pct",
    "risk_atr",
    "extension_atr",
    "expected_horizon",
    "horizon_source",
    "horizon_status",
    "horizon_review_due",
    "regime",
    "regime_stale_days",
    "atr",
    "fill_assumption",
]

# ─── thresholds (configurable but locked at v0) ──────────────────────────
MAX_RISK_ATR = 2.5
EXTENSION_CAP_ATR = 1.5
LIQUIDITY_FLOOR_PCTILE = 0.05  # bottom-5%-ile = configurable floor

# ─── accepted priors table (H1 only — H2 FAIL, no entry) ─────────────────
@dataclass(frozen=True)
class AcceptedPrior:
    family: str
    expected_horizon: int
    horizon_source: str
    horizon_status: str
    horizon_accepted_date: str
    horizon_review_due: str
    review_event_count: int  # OOS event count threshold for re-eval


ACCEPTED_PRIORS: tuple[AcceptedPrior, ...] = (
    AcceptedPrior(
        family="mb_5h__above_mb_birth",
        expected_horizon=20,
        horizon_source="exit_framework_v1_h1_pass",
        horizon_status="accepted_prior",
        horizon_accepted_date="2026-05-03",
        horizon_review_due="2026-11-03",
        review_event_count=1000,
    ),
    AcceptedPrior(
        family="mb_1d__above_mb_birth",
        expected_horizon=20,
        horizon_source="exit_framework_v1_h1_pass",
        horizon_status="accepted_prior",
        horizon_accepted_date="2026-05-03",
        horizon_review_due="2026-11-03",
        review_event_count=1000,
    ),
)

# Family aliases — live mb_scanner uses bare names; H1 prior anchors used
# the historical *_birth event-tag form. These resolve to the same cohort.
FAMILY_ALIASES: dict[str, str] = {
    "mb_5h__above_mb": "mb_5h__above_mb_birth",
    "mb_1d__above_mb": "mb_1d__above_mb_birth",
    "mb_1w__above_mb": "mb_1w__above_mb_birth",
    "mb_1M__above_mb": "mb_1M__above_mb_birth",
    "bb_5h__above_mb": "bb_5h__above_mb_birth",
    "bb_1d__above_mb": "bb_1d__above_mb_birth",
    "bb_1w__above_mb": "bb_1w__above_mb_birth",
    "bb_1M__above_mb": "bb_1M__above_mb_birth",
    "mb_5h__mitigation_touch": "mb_5h__mit_touch_first",
    "mb_1d__mitigation_touch": "mb_1d__mit_touch_first",
    "mb_1w__mitigation_touch": "mb_1w__mit_touch_first",
    "mb_1M__mitigation_touch": "mb_1M__mit_touch_first",
    "bb_5h__mitigation_touch": "bb_5h__mit_touch_first",
    "bb_1d__mitigation_touch": "bb_1d__mit_touch_first",
    "bb_1w__mitigation_touch": "bb_1w__mit_touch_first",
    "bb_1M__mitigation_touch": "bb_1M__mit_touch_first",
    "mb_5h__retest_bounce": "mb_5h__retest_bounce_first",
    "mb_1d__retest_bounce": "mb_1d__retest_bounce_first",
    "mb_1w__retest_bounce": "mb_1w__retest_bounce_first",
    "mb_1M__retest_bounce": "mb_1M__retest_bounce_first",
    "bb_5h__retest_bounce": "bb_5h__retest_bounce_first",
    "bb_1d__retest_bounce": "bb_1d__retest_bounce_first",
    "bb_1w__retest_bounce": "bb_1w__retest_bounce_first",
    "bb_1M__retest_bounce": "bb_1M__retest_bounce_first",
}


def canonical_family(family: str) -> str:
    """Map source-native family key to canonical (matches accepted-prior table)."""
    return FAMILY_ALIASES.get(family, family)


# Mit-touch / retest-bounce state aliases — for hard-rule matching.
MIT_TOUCH_STATES = ("mit_touch_first", "mitigation_touch")
RETEST_BOUNCE_STATES = ("retest_bounce_first", "retest_bounce")


__all__ = [
    "FINAL_ACTIONS",
    "PHASES",
    "REASON_CODES_POSITIVE",
    "REASON_CODES_NEGATIVE",
    "REASON_CODES_ALL",
    "assert_reason",
    "EVENT_COLUMNS",
    "ACTION_COLUMNS",
    "MAX_RISK_ATR",
    "EXTENSION_CAP_ATR",
    "LIQUIDITY_FLOOR_PCTILE",
    "ACCEPTED_PRIORS",
    "AcceptedPrior",
    "canonical_family",
    "FAMILY_ALIASES",
    "MIT_TOUCH_STATES",
    "RETEST_BOUNCE_STATES",
]
