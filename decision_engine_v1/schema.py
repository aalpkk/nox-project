"""Decision Engine v1 — schema, enums, dtype contracts, forbidden-field guard.

Spec sections: §3 (schema fields), §3.3.1 (paper validity carry, REVISION
2026-05-04), §3.6 (forbidden fields), §3.7 (Paper Signal Validity Policy),
§10.1–10.3 + additive (reason codes).

Three independent label fields per design §4: setup_label / execution_label /
market_context. AVOID string is banished: weakness -> setup_label=WEAK_CONTEXT;
non-executability -> execution_label=NOT_EXECUTABLE.
"""

from __future__ import annotations

from dataclasses import dataclass


# ─── Label enums (LOCKED) ────────────────────────────────────────────────

SETUP_LABELS: tuple[str, ...] = (
    "EARLY_SETUP",
    "TRIGGERED_SETUP",
    "RETEST_SETUP",
    "CONTINUATION_SETUP",
    "ACCEPTED_CONTINUATION_SETUP",
    "REVERSAL_SETUP",
    "STRENGTH_CONTEXT",
    "WEAK_CONTEXT",
    "EXTENDED_CONTEXT",
    "WARNING_CONTEXT",
)

EXECUTION_LABELS: tuple[str, ...] = (
    "EXECUTABLE",
    "WAIT_TRIGGER",
    "WAIT_RETEST",
    "WAIT_BETTER_ENTRY",
    "SIZE_REDUCED",
    "NOT_EXECUTABLE",
    "CONTEXT_ONLY",
    "WAIT_CONFIRMATION",  # additive A1.1 (nox_weekly conservative landing)
    "PAPER_ONLY",         # additive A1.2 (external-stream linkage only)
)

MARKET_CONTEXTS: tuple[str, ...] = (
    "REGIME_LONG",
    "REGIME_NEUTRAL",
    "REGIME_SHORT",
    "REGIME_UNKNOWN",
    "REGIME_STALE",
)

HORIZON_SOURCES: tuple[str, ...] = (
    "exit_framework_v1_h1_pass",
    "exit_framework_v1_h2_fail",
    "default_10d",
    "unresolved",
)

HORIZON_STATUSES: tuple[str, ...] = (
    "accepted_prior",
    "default_unchanged",
    "horizon_review_due",
    "horizon_status_stale",
    "unresolved",
)

EXECUTION_RISK_STATUSES: tuple[str, ...] = (
    "ok",
    "above_normal_below_reject",  # SIZE_REDUCED band (2.0, 3.0]
    "too_wide",                    # > 3.0 with valid inputs
    "missing_inputs",              # entry/stop/atr None
    "not_applicable",              # mapping default not risk-conditional
)

FILL_ASSUMPTIONS: tuple[str, ...] = (
    "next_open",
    "close_confirmed",
    "unresolved",
)

REGIME_STALE_FLAGS: tuple[str, ...] = (
    "fresh",
    "stale_1d",
    "stale_2d_plus",
    "missing",
)

# ─── Paper-stream reference enums (additive, post-revision) ──────────────

PAPER_ORIGINS: tuple[str | None, ...] = (
    None,
    "external_reference",  # only allowed non-None value
)

PAPER_STREAM_REFS: tuple[str | None, ...] = (
    None,
    "paper_execution_v0",                # Line E
    "paper_execution_v0_trigger_retest", # Line TR
)

# ─── Reason codes (LOCKED initial sets per design §10) ───────────────────

SETUP_REASON_CODES_INITIAL: tuple[str, ...] = (
    "accepted_horizon_h1_20d",
    "clean_breakout_context",
    "retest_confirmed",
    "retest_pending_context",  # additive A1.3 (mit_touch_first)
    "reversal_context",
    "strength_context",
    "weak_standalone_context",
    "extended_context",
    "failed_continuation_context",
    "h2_failed_no_15d_rule",
)

EXECUTION_REASON_CODES_INITIAL: tuple[str, ...] = (
    "entry_ref_present",
    "stop_ref_present",
    "execution_risk_ok",
    "execution_risk_too_wide",
    "execution_risk_above_normal_below_reject",  # SIZE_REDUCED band
    "no_stop_ref",
    "no_entry_ref",
    "liquidity_limited",
    "fill_uncertain",
    "wait_better_entry",
    "size_reduced",
    "fill_realism_unresolved",
)

CONTEXT_REASON_CODES_INITIAL: tuple[str, ...] = (
    "regime_long",
    "regime_neutral",
    "regime_short",
    "regime_unknown",
    "regime_stale",
    "nox_sat_conflict",
    "oe_high",
    "weekly_support",
    "momentum_support",
    "hw_exit_rejected_descriptive_only",
)

# ─── Forbidden fields (must not appear in v1 events parquet) ─────────────
#
# Hygiene patch 2026-05-04: replaced wildcard `*_score / *_rank` substring
# logic with exact-name enumeration plus an explicit allow-list. The previous
# wildcard erroneously flagged the LOCKED legitimate execution columns
# `liquidity_score` and `capacity_score`. The forbidden surface continues to
# capture the spec §3.6 ranker/portfolio artefacts; the allow-list pins the
# three score-named fields that are LOCKED legitimate (descriptive carry +
# liquidity / capacity execution diagnostics).

FORBIDDEN_FIELDS: frozenset[str] = frozenset(
    {
        "final_action",         # v0 collapsed-action artifact
        "selection_score",
        "edge_score",
        "prior_edge_score",
        "setup_score",          # additive 2026-05-04 hygiene patch
        "rank",
        "portfolio_pick",
        "portfolio_weight",
    }
)

# Score/rank-named fields that are LOCKED legitimate (NOT ranker artefacts):
#   * earliness_score_pct  — descriptive passive carry (§3.6 explicit exception)
#   * liquidity_score      — execution diagnostic (LOCKED EXECUTION_COLUMNS)
#   * capacity_score       — execution diagnostic (LOCKED EXECUTION_COLUMNS)
_ALLOWED_SCORE_NAMED_FIELDS: frozenset[str] = frozenset(
    {
        "earliness_score_pct",
        "liquidity_score",
        "capacity_score",
    }
)


def is_forbidden_score_field(field_name: str) -> bool:
    """Exact-name forbidden-field check (hygiene patch 2026-05-04).

    Returns True iff the column name matches one of the LOCKED forbidden
    names in `FORBIDDEN_FIELDS`. The pre-patch wildcard `*_score / *_rank`
    substring rule was removed: it incorrectly classified
    `liquidity_score` / `capacity_score` (LOCKED EXECUTION_COLUMNS) as
    forbidden. The allow-list `_ALLOWED_SCORE_NAMED_FIELDS` is kept for
    callers that want to whitelist-style verify a name before consulting
    the forbidden set.
    """
    if field_name in _ALLOWED_SCORE_NAMED_FIELDS:
        return False
    return field_name in FORBIDDEN_FIELDS


# ─── Schema columns (consolidated; spec §3.1–§3.5 + §3.3.1 additive) ─────

IDENTITY_COLUMNS: tuple[str, ...] = (
    "date",
    "ticker",
    "source",
    "family",
    "state",
    "timeframe",
    "direction",
)

STRUCTURE_COLUMNS: tuple[str, ...] = (
    "setup_label",
    "execution_label",
    "market_context",
    "phase",
    "expected_horizon",
    "horizon_source",
    "horizon_status",
    "horizon_review_due",
)

EXECUTION_COLUMNS: tuple[str, ...] = (
    "entry_ref",
    "stop_ref",
    "atr",
    "risk_pct",
    "risk_atr",
    "execution_risk_status",
    "fill_assumption",
    "next_open_gap_if_available",
    "liquidity_score",
    "capacity_score",
    "live_execution_allowed",
)

PAPER_REFERENCE_COLUMNS: tuple[str, ...] = (
    "paper_origin",
    "paper_stream_ref",
    "paper_trade_id",
    "paper_match_key",
)

PAPER_VALIDITY_COLUMNS: tuple[str, ...] = (  # §3.3.1 additive REVISION 2026-05-04
    "paper_valid_from",
    "paper_valid_until",
    "paper_signal_age",
    "paper_expired_flag",
    "paper_validity_metadata_missing",
)

CONTEXT_COLUMNS: tuple[str, ...] = (
    "regime",
    "regime_stale_flag",
    "higher_tf_context",
    "lower_tf_context",
    "supporting_signals",
    "conflict_flags",
    "hw_context_tags",
    "ema_context_tags",
    "ema_context_tag",
    "earliness_score_pct",
)

REASON_COLUMNS: tuple[str, ...] = (
    "setup_reason_codes",
    "execution_reason_codes",
    "context_reason_codes",
)

ALL_EVENT_COLUMNS: tuple[str, ...] = (
    IDENTITY_COLUMNS
    + STRUCTURE_COLUMNS
    + EXECUTION_COLUMNS
    + PAPER_REFERENCE_COLUMNS
    + PAPER_VALIDITY_COLUMNS
    + CONTEXT_COLUMNS
    + REASON_COLUMNS
)


# ─── Validators (Step 1 — schema validation) ─────────────────────────────


def assert_setup_label(value: str) -> str:
    if value not in SETUP_LABELS:
        raise ValueError(f"setup_label {value!r} not in {SETUP_LABELS}")
    return value


def assert_execution_label(value: str) -> str:
    if value not in EXECUTION_LABELS:
        raise ValueError(f"execution_label {value!r} not in {EXECUTION_LABELS}")
    return value


def assert_market_context(value: str) -> str:
    if value not in MARKET_CONTEXTS:
        raise ValueError(f"market_context {value!r} not in {MARKET_CONTEXTS}")
    return value


def assert_no_forbidden_fields(columns) -> None:
    """Hard-fail if any forbidden field is present (Step 1.4)."""
    for col in columns:
        if col in FORBIDDEN_FIELDS:
            raise ValueError(
                f"forbidden field {col!r} present in v1 event schema "
                "(see implementation spec §3.6)"
            )
        if is_forbidden_score_field(col):
            raise ValueError(
                f"forbidden score/rank field {col!r} present in v1 event schema "
                "(only earliness_score_pct is allowed; see §3.6)"
            )


# ─── Paper-reference invariants (§3.3 + §3.3.1) ──────────────────────────


@dataclass(frozen=True)
class PaperReferenceState:
    """Snapshot of the paper-reference fields on a single v1 event row.
    Used by purity_check and the schema validator to enforce invariants
    without importing pandas at validation time.
    """

    execution_label: str
    paper_origin: str | None
    paper_stream_ref: str | None
    paper_trade_id: str | None
    paper_match_key: str | None
    live_execution_allowed: bool
    paper_valid_from: object | None       # date or None
    paper_valid_until: object | None      # date or None
    paper_signal_age: int | None
    paper_expired_flag: bool | None
    paper_validity_metadata_missing: bool | None


def assert_paper_reference_invariants(state: PaperReferenceState) -> None:
    """§3.3 + §3.3.1 invariants. Raises on first violation; never patches."""
    is_paper = state.execution_label == "PAPER_ONLY"
    if is_paper:
        if state.live_execution_allowed is not False:
            raise ValueError(
                "PAPER_ONLY row must have live_execution_allowed=False"
            )
        if state.paper_origin != "external_reference":
            raise ValueError(
                "PAPER_ONLY row must have paper_origin='external_reference'"
            )
        if state.paper_stream_ref not in (
            "paper_execution_v0",
            "paper_execution_v0_trigger_retest",
        ):
            raise ValueError(
                "PAPER_ONLY row must have paper_stream_ref in "
                "{paper_execution_v0, paper_execution_v0_trigger_retest}"
            )
        has_id = state.paper_trade_id is not None
        has_key = state.paper_match_key is not None
        if has_id == has_key:
            raise ValueError(
                "PAPER_ONLY row must have exactly one of "
                "paper_trade_id / paper_match_key set"
            )
        if state.paper_validity_metadata_missing is None:
            raise ValueError(
                "PAPER_ONLY row must have paper_validity_metadata_missing "
                "set to a non-null bool (§3.3.1)"
            )
    else:
        if state.paper_origin is not None:
            raise ValueError(
                "non-PAPER_ONLY row must have paper_origin=None"
            )
        if state.paper_stream_ref is not None:
            raise ValueError(
                "non-PAPER_ONLY row must have paper_stream_ref=None"
            )
        if state.paper_trade_id is not None:
            raise ValueError(
                "non-PAPER_ONLY row must have paper_trade_id=None"
            )
        if state.paper_match_key is not None:
            raise ValueError(
                "non-PAPER_ONLY row must have paper_match_key=None"
            )
        # All five validity fields must be None on non-PAPER_ONLY rows
        for field_name in (
            "paper_valid_from",
            "paper_valid_until",
            "paper_signal_age",
            "paper_expired_flag",
            "paper_validity_metadata_missing",
        ):
            if getattr(state, field_name) is not None:
                raise ValueError(
                    f"non-PAPER_ONLY row must have {field_name}=None (§3.3.1)"
                )


__all__ = [
    "SETUP_LABELS",
    "EXECUTION_LABELS",
    "MARKET_CONTEXTS",
    "HORIZON_SOURCES",
    "HORIZON_STATUSES",
    "EXECUTION_RISK_STATUSES",
    "FILL_ASSUMPTIONS",
    "REGIME_STALE_FLAGS",
    "PAPER_ORIGINS",
    "PAPER_STREAM_REFS",
    "SETUP_REASON_CODES_INITIAL",
    "EXECUTION_REASON_CODES_INITIAL",
    "CONTEXT_REASON_CODES_INITIAL",
    "FORBIDDEN_FIELDS",
    "is_forbidden_score_field",
    "IDENTITY_COLUMNS",
    "STRUCTURE_COLUMNS",
    "EXECUTION_COLUMNS",
    "PAPER_REFERENCE_COLUMNS",
    "PAPER_VALIDITY_COLUMNS",
    "CONTEXT_COLUMNS",
    "REASON_COLUMNS",
    "ALL_EVENT_COLUMNS",
    "assert_setup_label",
    "assert_execution_label",
    "assert_market_context",
    "assert_no_forbidden_fields",
    "PaperReferenceState",
    "assert_paper_reference_invariants",
]
