"""Decision Engine v1 — LOCKED mapping table (review §4, 32 rows).

The mapping table is **append-only** within v1's lifecycle. Any divergence
between this dict and review §4 markdown is a non-compliance condition
(spec §4). Adding/removing rows requires fresh pre-reg.

The 32 rows here are panel cells (mb_scanner ×24, nox_rt_daily, nox_weekly,
nyxexpansion, horizontal_base ×3, nyxmomentum ×2). Paper-line attachment
is NOT a separate row — it is a column-attach pass over rows #28–#30
performed by `paper_stream_link.py` (review §4 closing note).

Risk-conditional cells emit `execution_label_default = "RISK_BRANCH"` here;
`risk.py` resolves it to one of {EXECUTABLE, SIZE_REDUCED, WAIT_BETTER_ENTRY,
NOT_EXECUTABLE} per LOCKED four-way branch (spec §3 Step 3 + §6).
"""

from __future__ import annotations

from typing import Final


RISK_BRANCH: Final[str] = "RISK_BRANCH"  # sentinel resolved by risk.py


# ─── Mapping table key + row schema ──────────────────────────────────────
#
# Key: (source, family, state, timeframe). `state` is the panel `phase`
# field. The (source, family, timeframe) trio alone is not unique — for
# example mb_scanner mit_touch_first families share (source, family, tf)
# with their retest_bounce_first counterparts on different rows; the
# `phase` discriminator is required.

MappingKey = tuple[str, str, str, str]


def _row(
    setup_label: str,
    execution_label_default: str,
    expected_horizon: int | None,
    horizon_source: str,
    live_execution_allowed: bool = True,
    allowed_setup_reason_codes: tuple[str, ...] = (),
    allowed_execution_reason_codes: tuple[str, ...] = (),
) -> dict:
    return {
        "setup_label": setup_label,
        "execution_label_default": execution_label_default,
        "expected_horizon": expected_horizon,
        "horizon_source": horizon_source,
        "live_execution_allowed": live_execution_allowed,
        "allowed_setup_reason_codes": allowed_setup_reason_codes,
        "allowed_execution_reason_codes": allowed_execution_reason_codes,
    }


# Reason-code allow-lists per cell are reproduced from review §2.1–§2.9.
# Duplicate execution-side allow-lists are factored as constants.

_RISK_BRANCH_EXEC_REASONS: Final[tuple[str, ...]] = (
    "entry_ref_present",
    "stop_ref_present",
    "execution_risk_ok",
    "execution_risk_too_wide",
    "execution_risk_above_normal_below_reject",
    "wait_better_entry",
    "size_reduced",
    "no_entry_ref",
    "no_stop_ref",
    "fill_realism_unresolved",
)

_RETEST_PENDING_EXEC_REASONS: Final[tuple[str, ...]] = (
    "wait_better_entry",
    "no_entry_ref",
    "no_stop_ref",
)


MAPPING_TABLE: Final[dict[MappingKey, dict]] = {
    # ── mb_scanner H1 cohort (rows #1–#2) ────────────────────────────────
    ("mb_scanner", "mb_5h__above_mb_birth", "accepted_continuation", "5h"): _row(
        setup_label="ACCEPTED_CONTINUATION_SETUP",
        execution_label_default=RISK_BRANCH,
        expected_horizon=20,
        horizon_source="exit_framework_v1_h1_pass",
        allowed_setup_reason_codes=(
            "accepted_horizon_h1_20d",
            "clean_breakout_context",
        ),
        allowed_execution_reason_codes=_RISK_BRANCH_EXEC_REASONS,
    ),
    ("mb_scanner", "mb_1d__above_mb_birth", "accepted_continuation", "1d"): _row(
        setup_label="ACCEPTED_CONTINUATION_SETUP",
        execution_label_default=RISK_BRANCH,
        expected_horizon=20,
        horizon_source="exit_framework_v1_h1_pass",
        allowed_setup_reason_codes=(
            "accepted_horizon_h1_20d",
            "clean_breakout_context",
        ),
        allowed_execution_reason_codes=_RISK_BRANCH_EXEC_REASONS,
    ),
    # ── mb_scanner non-H1 above_mb_birth (rows #3–#8) ────────────────────
    ("mb_scanner", "mb_1w__above_mb_birth", "continuation", "1w"): _row(
        setup_label="CONTINUATION_SETUP",
        execution_label_default=RISK_BRANCH,
        expected_horizon=10,
        horizon_source="default_10d",
        allowed_setup_reason_codes=("clean_breakout_context",),
        allowed_execution_reason_codes=_RISK_BRANCH_EXEC_REASONS,
    ),
    ("mb_scanner", "mb_1M__above_mb_birth", "continuation", "1M"): _row(
        setup_label="CONTINUATION_SETUP",
        execution_label_default=RISK_BRANCH,
        expected_horizon=10,
        horizon_source="exit_framework_v1_h2_fail",
        allowed_setup_reason_codes=(
            "clean_breakout_context",
            "h2_failed_no_15d_rule",
        ),
        allowed_execution_reason_codes=_RISK_BRANCH_EXEC_REASONS,
    ),
    ("mb_scanner", "bb_5h__above_mb_birth", "continuation", "5h"): _row(
        setup_label="CONTINUATION_SETUP",
        execution_label_default=RISK_BRANCH,
        expected_horizon=10,
        horizon_source="default_10d",
        allowed_setup_reason_codes=("clean_breakout_context",),
        allowed_execution_reason_codes=_RISK_BRANCH_EXEC_REASONS,
    ),
    ("mb_scanner", "bb_1d__above_mb_birth", "continuation", "1d"): _row(
        setup_label="CONTINUATION_SETUP",
        execution_label_default=RISK_BRANCH,
        expected_horizon=10,
        horizon_source="default_10d",
        allowed_setup_reason_codes=("clean_breakout_context",),
        allowed_execution_reason_codes=_RISK_BRANCH_EXEC_REASONS,
    ),
    ("mb_scanner", "bb_1w__above_mb_birth", "continuation", "1w"): _row(
        setup_label="CONTINUATION_SETUP",
        execution_label_default=RISK_BRANCH,
        expected_horizon=10,
        horizon_source="default_10d",
        allowed_setup_reason_codes=("clean_breakout_context",),
        allowed_execution_reason_codes=_RISK_BRANCH_EXEC_REASONS,
    ),
    ("mb_scanner", "bb_1M__above_mb_birth", "continuation", "1M"): _row(
        setup_label="CONTINUATION_SETUP",
        execution_label_default=RISK_BRANCH,
        expected_horizon=10,
        horizon_source="default_10d",
        allowed_setup_reason_codes=("clean_breakout_context",),
        allowed_execution_reason_codes=_RISK_BRANCH_EXEC_REASONS,
    ),
    # ── mb_scanner mit_touch_first (rows #9–#16) ────────────────────────
    ("mb_scanner", "mb_5h__mit_touch_first", "retest_pending", "5h"): _row(
        setup_label="RETEST_SETUP",
        execution_label_default="WAIT_RETEST",
        expected_horizon=10,
        horizon_source="default_10d",
        allowed_setup_reason_codes=("retest_pending_context",),
        allowed_execution_reason_codes=_RETEST_PENDING_EXEC_REASONS,
    ),
    ("mb_scanner", "mb_1d__mit_touch_first", "retest_pending", "1d"): _row(
        setup_label="RETEST_SETUP",
        execution_label_default="WAIT_RETEST",
        expected_horizon=10,
        horizon_source="default_10d",
        allowed_setup_reason_codes=("retest_pending_context",),
        allowed_execution_reason_codes=_RETEST_PENDING_EXEC_REASONS,
    ),
    ("mb_scanner", "mb_1w__mit_touch_first", "retest_pending", "1w"): _row(
        setup_label="RETEST_SETUP",
        execution_label_default="WAIT_RETEST",
        expected_horizon=10,
        horizon_source="default_10d",
        allowed_setup_reason_codes=("retest_pending_context",),
        allowed_execution_reason_codes=_RETEST_PENDING_EXEC_REASONS,
    ),
    ("mb_scanner", "mb_1M__mit_touch_first", "retest_pending", "1M"): _row(
        setup_label="RETEST_SETUP",
        execution_label_default="WAIT_RETEST",
        expected_horizon=10,
        horizon_source="default_10d",
        allowed_setup_reason_codes=("retest_pending_context",),
        allowed_execution_reason_codes=_RETEST_PENDING_EXEC_REASONS,
    ),
    ("mb_scanner", "bb_5h__mit_touch_first", "retest_pending", "5h"): _row(
        setup_label="RETEST_SETUP",
        execution_label_default="WAIT_RETEST",
        expected_horizon=10,
        horizon_source="default_10d",
        allowed_setup_reason_codes=("retest_pending_context",),
        allowed_execution_reason_codes=_RETEST_PENDING_EXEC_REASONS,
    ),
    ("mb_scanner", "bb_1d__mit_touch_first", "retest_pending", "1d"): _row(
        setup_label="RETEST_SETUP",
        execution_label_default="WAIT_RETEST",
        expected_horizon=10,
        horizon_source="default_10d",
        allowed_setup_reason_codes=("retest_pending_context",),
        allowed_execution_reason_codes=_RETEST_PENDING_EXEC_REASONS,
    ),
    ("mb_scanner", "bb_1w__mit_touch_first", "retest_pending", "1w"): _row(
        setup_label="RETEST_SETUP",
        execution_label_default="WAIT_RETEST",
        expected_horizon=10,
        horizon_source="default_10d",
        allowed_setup_reason_codes=("retest_pending_context",),
        allowed_execution_reason_codes=_RETEST_PENDING_EXEC_REASONS,
    ),
    ("mb_scanner", "bb_1M__mit_touch_first", "retest_pending", "1M"): _row(
        setup_label="RETEST_SETUP",
        execution_label_default="WAIT_RETEST",
        expected_horizon=10,
        horizon_source="default_10d",
        allowed_setup_reason_codes=("retest_pending_context",),
        allowed_execution_reason_codes=_RETEST_PENDING_EXEC_REASONS,
    ),
    # ── mb_scanner retest_bounce_first (rows #17–#24) ───────────────────
    ("mb_scanner", "mb_5h__retest_bounce_first", "retest", "5h"): _row(
        setup_label="RETEST_SETUP",
        execution_label_default=RISK_BRANCH,
        expected_horizon=10,
        horizon_source="default_10d",
        allowed_setup_reason_codes=("retest_confirmed",),
        allowed_execution_reason_codes=_RISK_BRANCH_EXEC_REASONS,
    ),
    ("mb_scanner", "mb_1d__retest_bounce_first", "retest", "1d"): _row(
        setup_label="RETEST_SETUP",
        execution_label_default=RISK_BRANCH,
        expected_horizon=10,
        horizon_source="default_10d",
        allowed_setup_reason_codes=("retest_confirmed",),
        allowed_execution_reason_codes=_RISK_BRANCH_EXEC_REASONS,
    ),
    ("mb_scanner", "mb_1w__retest_bounce_first", "retest", "1w"): _row(
        setup_label="RETEST_SETUP",
        execution_label_default=RISK_BRANCH,
        expected_horizon=10,
        horizon_source="default_10d",
        allowed_setup_reason_codes=("retest_confirmed",),
        allowed_execution_reason_codes=_RISK_BRANCH_EXEC_REASONS,
    ),
    ("mb_scanner", "mb_1M__retest_bounce_first", "retest", "1M"): _row(
        setup_label="RETEST_SETUP",
        execution_label_default=RISK_BRANCH,
        expected_horizon=10,
        horizon_source="default_10d",
        allowed_setup_reason_codes=("retest_confirmed",),
        allowed_execution_reason_codes=_RISK_BRANCH_EXEC_REASONS,
    ),
    ("mb_scanner", "bb_5h__retest_bounce_first", "retest", "5h"): _row(
        setup_label="RETEST_SETUP",
        execution_label_default=RISK_BRANCH,
        expected_horizon=10,
        horizon_source="default_10d",
        allowed_setup_reason_codes=("retest_confirmed",),
        allowed_execution_reason_codes=_RISK_BRANCH_EXEC_REASONS,
    ),
    ("mb_scanner", "bb_1d__retest_bounce_first", "retest", "1d"): _row(
        setup_label="RETEST_SETUP",
        execution_label_default=RISK_BRANCH,
        expected_horizon=10,
        horizon_source="default_10d",
        allowed_setup_reason_codes=("retest_confirmed",),
        allowed_execution_reason_codes=_RISK_BRANCH_EXEC_REASONS,
    ),
    ("mb_scanner", "bb_1w__retest_bounce_first", "retest", "1w"): _row(
        setup_label="RETEST_SETUP",
        execution_label_default=RISK_BRANCH,
        expected_horizon=10,
        horizon_source="default_10d",
        allowed_setup_reason_codes=("retest_confirmed",),
        allowed_execution_reason_codes=_RISK_BRANCH_EXEC_REASONS,
    ),
    ("mb_scanner", "bb_1M__retest_bounce_first", "retest", "1M"): _row(
        setup_label="RETEST_SETUP",
        execution_label_default=RISK_BRANCH,
        expected_horizon=10,
        horizon_source="default_10d",
        allowed_setup_reason_codes=("retest_confirmed",),
        allowed_execution_reason_codes=_RISK_BRANCH_EXEC_REASONS,
    ),
    # ── nox_rt_daily / nox_weekly (rows #25–#26) ─────────────────────────
    ("nox_rt_daily", "nox_rt_daily__pivot_al", "trigger", "1d"): _row(
        setup_label="WEAK_CONTEXT",
        execution_label_default="CONTEXT_ONLY",
        expected_horizon=10,
        horizon_source="default_10d",
        allowed_setup_reason_codes=("weak_standalone_context",),
        allowed_execution_reason_codes=("no_entry_ref",),
    ),
    ("nox_weekly", "nox_weekly__weekly_pivot_al", "trigger", "1w"): _row(
        setup_label="REVERSAL_SETUP",
        execution_label_default="WAIT_CONFIRMATION",
        expected_horizon=10,
        horizon_source="default_10d",
        allowed_setup_reason_codes=("reversal_context",),
        allowed_execution_reason_codes=(
            "wait_better_entry",
            "no_entry_ref",
            "no_stop_ref",
        ),
    ),
    # ── nyxexpansion (row #27) ──────────────────────────────────────────
    ("nyxexpansion", "nyxexpansion__triggerA", "trigger", "1d"): _row(
        setup_label="TRIGGERED_SETUP",
        execution_label_default=RISK_BRANCH,
        expected_horizon=10,
        horizon_source="default_10d",
        allowed_setup_reason_codes=("clean_breakout_context",),
        allowed_execution_reason_codes=_RISK_BRANCH_EXEC_REASONS,
    ),
    # ── horizontal_base (rows #28–#30; paper-stream linkage attaches here) ─
    ("horizontal_base", "horizontal_base__trigger", "trigger", "1d"): _row(
        setup_label="TRIGGERED_SETUP",
        execution_label_default=RISK_BRANCH,
        expected_horizon=10,
        horizon_source="default_10d",
        allowed_setup_reason_codes=("clean_breakout_context",),
        allowed_execution_reason_codes=_RISK_BRANCH_EXEC_REASONS,
    ),
    ("horizontal_base", "horizontal_base__retest_bounce", "retest", "1d"): _row(
        setup_label="RETEST_SETUP",
        execution_label_default=RISK_BRANCH,
        expected_horizon=10,
        horizon_source="default_10d",
        allowed_setup_reason_codes=("retest_confirmed",),
        allowed_execution_reason_codes=_RISK_BRANCH_EXEC_REASONS,
    ),
    ("horizontal_base", "horizontal_base__extended", "extended", "1d"): _row(
        setup_label="EXTENDED_CONTEXT",
        execution_label_default="NOT_EXECUTABLE",
        expected_horizon=10,
        horizon_source="default_10d",
        allowed_setup_reason_codes=("extended_context",),
        allowed_execution_reason_codes=("no_entry_ref",),
    ),
    # ── nyxmomentum (rows #31–#32) ──────────────────────────────────────
    ("nyxmomentum", "nyxmomentum__m0", "strength_context", "1d"): _row(
        setup_label="STRENGTH_CONTEXT",
        execution_label_default="CONTEXT_ONLY",
        expected_horizon=None,
        horizon_source="unresolved",
        allowed_setup_reason_codes=("strength_context",),
        allowed_execution_reason_codes=("no_entry_ref",),
    ),
    ("nyxmomentum", "nyxmomentum__v5", "strength_context", "1d"): _row(
        setup_label="STRENGTH_CONTEXT",
        execution_label_default="CONTEXT_ONLY",
        expected_horizon=None,
        horizon_source="unresolved",
        allowed_setup_reason_codes=("strength_context",),
        allowed_execution_reason_codes=("no_entry_ref",),
    ),
}


# ─── Source whitelist (Step 0 live-scope guard) ──────────────────────────

V1_SOURCE_WHITELIST: Final[frozenset[str]] = frozenset(
    {
        "mb_scanner",
        "nox_rt_daily",
        "nox_weekly",
        "nyxexpansion",
        "horizontal_base",
        "nyxmomentum",
    }
)


# ─── HB family → paper-line token (Step 5.0 derivation) ──────────────────
#
# Used only by `paper_stream_link.py` to resolve the `line` component of
# the paper-stream match key. Maps the v1 event's `family` to the
# upstream parquet's `line` column value (case-sensitive).

HB_FAMILY_TO_PAPER_LINE: Final[dict[str, str]] = {
    "horizontal_base__extended": "EXTENDED",
    "horizontal_base__trigger": "TRIGGER_RETEST",
    "horizontal_base__retest_bounce": "TRIGGER_RETEST",
}

PAPER_LINE_TO_STREAM_REF: Final[dict[str, str]] = {
    "EXTENDED": "paper_execution_v0",
    "TRIGGER_RETEST": "paper_execution_v0_trigger_retest",
}


def get_mapping_row(
    source: str, family: str, state: str, timeframe: str
) -> dict | None:
    """Return the LOCKED mapping row for a (source, family, state, tf) cell.

    Returns None when the cell is not in the mapping table; the caller
    (live_scope.Step 0) is responsible for fail-fast on None.
    """
    return MAPPING_TABLE.get((source, family, state, timeframe))


def all_mapped_cells() -> tuple[MappingKey, ...]:
    """Iterate all 32 LOCKED cells."""
    return tuple(MAPPING_TABLE.keys())


def assert_table_row_count() -> None:
    """Hard-assert the row count against the LOCKED 32 (review §4)."""
    n = len(MAPPING_TABLE)
    if n != 32:
        raise AssertionError(
            f"MAPPING_TABLE has {n} rows; LOCKED expectation is 32 "
            "(mapping review §4). Any divergence is non-compliance."
        )


__all__ = [
    "RISK_BRANCH",
    "MappingKey",
    "MAPPING_TABLE",
    "V1_SOURCE_WHITELIST",
    "HB_FAMILY_TO_PAPER_LINE",
    "PAPER_LINE_TO_STREAM_REF",
    "get_mapping_row",
    "all_mapped_cells",
    "assert_table_row_count",
]
