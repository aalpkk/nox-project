"""Decision Engine v1 — label assignment via deterministic mapping lookup.

Spec §2 Step 2 + Step 3 + Step 4 composition: this module is the only
place that assigns `setup_label`, `execution_label`, and `market_context`
on a v1 event. Mapping-only — no scoring, no inference, no decision
logic beyond the LOCKED §4 mapping table and the LOCKED four-way risk
branch (delegated to `risk.py`).

Hard rules:
  - `setup_label` is taken straight from the mapping table; risk
    arithmetic does NOT modify it (spec §3 Step 3.5).
  - `execution_label_default = RISK_BRANCH` is resolved by `risk.py`.
    All other defaults are passed through verbatim.
  - This module does NOT read `paper_origin` / `paper_stream_ref` /
    paper-validity columns (cross-layer purity static check enforces).
  - This module does NOT read `earliness_score_pct` (cross-layer purity).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from decision_engine_v1.ingest import ScannerEvent
from decision_engine_v1.mapping_table import (
    MAPPING_TABLE,
    RISK_BRANCH,
)
from decision_engine_v1.risk import RiskResolution, resolve_risk_branch


# Reason-code wiring tables (subset of design §10 — only those that the
# label layer actively wires; the rest are added by the source-side
# enrichment code or by paper_stream_link.py).

_SETUP_REASON_BY_LABEL: Final[dict[str, tuple[str, ...]]] = {
    "ACCEPTED_CONTINUATION_SETUP": ("accepted_horizon_h1_20d", "clean_breakout_context"),
    "CONTINUATION_SETUP": ("clean_breakout_context",),
    "RETEST_SETUP": (),  # discriminated below by state (mit_touch vs retest)
    "TRIGGERED_SETUP": ("clean_breakout_context",),
    "REVERSAL_SETUP": ("reversal_context",),
    "WEAK_CONTEXT": ("weak_standalone_context",),
    "EXTENDED_CONTEXT": ("extended_context",),
    "STRENGTH_CONTEXT": ("strength_context",),
}


@dataclass(frozen=True)
class LabelDecision:
    """All structure/execution fields produced by the label layer.

    `paper_*` fields are NOT touched here — `paper_stream_link.py`
    overrides `execution_label` to PAPER_ONLY and attaches reference
    fields when an upstream paper-stream record matches.
    """

    setup_label: str
    execution_label: str
    expected_horizon: int | None
    horizon_source: str
    live_execution_allowed: bool
    risk_pct: float | None
    risk_atr: float | None
    execution_risk_status: str
    setup_reason_codes: tuple[str, ...]
    execution_reason_codes: tuple[str, ...]


def _setup_reasons_for(event: ScannerEvent, setup_label: str) -> tuple[str, ...]:
    base = _SETUP_REASON_BY_LABEL.get(setup_label, ())
    # RETEST_SETUP discrimination: mit_touch_first → retest_pending_context;
    # *_retest_bounce_first / horizontal_base__retest_bounce → retest_confirmed.
    if setup_label == "RETEST_SETUP":
        if event.state == "retest_pending":
            return ("retest_pending_context",)
        if event.state == "retest":
            return ("retest_confirmed",)
        return ()
    # CONTINUATION_SETUP on mb_1M__above_mb_birth carries h2_failed_no_15d_rule.
    if setup_label == "CONTINUATION_SETUP" and event.family == "mb_1M__above_mb_birth":
        return base + ("h2_failed_no_15d_rule",)
    return base


def assign_label(event: ScannerEvent) -> LabelDecision:
    """Look up the LOCKED mapping row for `event` and produce the
    `LabelDecision`. Risk-conditional cells delegate to `risk.py`.

    Caller must have already passed `event` through Step 0 live-scope
    validation; this function does not re-validate (it would be a
    redundant lookup).

    Raises:
        KeyError: if the (source, family, state, timeframe) cell is
            not in the mapping table. This should never happen post-
            Step 0; if it does, it is a non-compliance signal.
    """
    key = (event.source, event.family, event.state, event.timeframe)
    row = MAPPING_TABLE[key]  # raises KeyError if not found

    setup_label = row["setup_label"]
    default_execution = row["execution_label_default"]
    setup_reasons = _setup_reasons_for(event, setup_label)

    risk_resolution: RiskResolution | None = None
    if default_execution == RISK_BRANCH:
        risk_resolution = resolve_risk_branch(
            entry_ref=event.entry_ref,
            stop_ref=event.stop_ref,
            atr=event.atr,
        )
        execution_label = risk_resolution.execution_label
        execution_risk_status = risk_resolution.execution_risk_status
        risk_pct = risk_resolution.risk_pct
        risk_atr = risk_resolution.risk_atr
        execution_reasons = risk_resolution.reason_codes
    else:
        execution_label = default_execution
        execution_risk_status = "not_applicable"
        risk_pct = None
        risk_atr = None
        # Non-risk-conditional cells get a minimal reason-code prefix.
        # Specific reasons (e.g., `no_entry_ref` for nox_rt_daily) are
        # encoded in the mapping table's `allowed_execution_reason_codes`
        # but the label layer emits the canonical ones tied to the
        # default execution_label.
        if execution_label == "WAIT_RETEST":
            execution_reasons = ("no_entry_ref",)
        elif execution_label == "CONTEXT_ONLY":
            execution_reasons = ("no_entry_ref",)
        elif execution_label == "NOT_EXECUTABLE":
            execution_reasons = ("no_entry_ref",)
        elif execution_label == "WAIT_CONFIRMATION":
            # nox_weekly: reasons are entry/stop dependent — leave empty
            # at this layer; the future entry/stop enrichment can add
            # `wait_better_entry` / `no_entry_ref` accurately.
            execution_reasons = ()
        else:
            execution_reasons = ()

    return LabelDecision(
        setup_label=setup_label,
        execution_label=execution_label,
        expected_horizon=row["expected_horizon"],
        horizon_source=row["horizon_source"],
        live_execution_allowed=row["live_execution_allowed"],
        risk_pct=risk_pct,
        risk_atr=risk_atr,
        execution_risk_status=execution_risk_status,
        setup_reason_codes=setup_reasons,
        execution_reason_codes=execution_reasons,
    )


__all__ = [
    "LabelDecision",
    "assign_label",
]
