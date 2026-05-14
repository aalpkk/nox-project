"""Decision Engine v1 — risk arithmetic + execution_label four-way branch.

LOCKED 2026-05-04 thresholds (spec §6):

    ATR_WINDOW = 14                      # Wilder's True Range smoothing
    MAX_RISK_ATR_EXECUTABLE = 2.0
    MAX_RISK_ATR_SIZE_REDUCED = 3.0

LOCKED four-way branch (spec §3 Step 3):

    entry_ref is None        → NOT_EXECUTABLE   (no_entry_ref)
    stop_ref  is None        → NOT_EXECUTABLE   (no_stop_ref)
    atr       is None        → NOT_EXECUTABLE   (fill_realism_unresolved)
    risk_atr  ≤ 2.0          → EXECUTABLE       (execution_risk_ok)
    2.0 < risk_atr ≤ 3.0     → SIZE_REDUCED     (execution_risk_above_normal_below_reject)
    risk_atr  > 3.0          → WAIT_BETTER_ENTRY(execution_risk_too_wide)

Hard rules:
  - `setup_label` is NEVER modified by this module (spec §3 Step 3.5).
  - `SIZE_REDUCED` is a label-only paper/manual-review tag; this module
    NEVER attaches a notional, multiplier, or sizing arithmetic.
  - The four branches are STRUCTURALLY DISTINCT and must not be collapsed.
  - Risk arithmetic does not read `earliness_score_pct`, `paper_origin`,
    or any paper-stream reference field — those live in their own
    layers (paper_stream_link.py / write.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


ATR_WINDOW: Final[int] = 14
MAX_RISK_ATR_EXECUTABLE: Final[float] = 2.0
MAX_RISK_ATR_SIZE_REDUCED: Final[float] = 3.0


@dataclass(frozen=True)
class RiskResolution:
    """Outcome of risk-arithmetic branching for a single risk-conditional
    event. Carries enough context to set the row's
    `execution_label` / `execution_risk_status` / `risk_pct` / `risk_atr`
    / `execution_reason_codes` (additive).
    """

    execution_label: str
    execution_risk_status: str
    risk_pct: float | None
    risk_atr: float | None
    reason_codes: tuple[str, ...]


def _compute_risk_pct(entry_ref: float, stop_ref: float) -> float:
    if entry_ref == 0.0:
        # Defensive: a zero entry_ref would divide by zero. We treat it
        # as a structural input failure (caller should have caught it).
        raise ValueError("entry_ref is zero; cannot compute risk_pct")
    return (entry_ref - stop_ref) / entry_ref


def _compute_risk_atr(entry_ref: float, stop_ref: float, atr: float) -> float:
    if atr == 0.0:
        raise ValueError("atr is zero; cannot compute risk_atr")
    return (entry_ref - stop_ref) / atr


def resolve_risk_branch(
    entry_ref: float | None,
    stop_ref: float | None,
    atr: float | None,
) -> RiskResolution:
    """LOCKED four-way branch on (entry_ref, stop_ref, atr).

    Returns a `RiskResolution`. Never raises on missing inputs — those
    route to NOT_EXECUTABLE per spec. Raises only if a numeric branch
    would divide by zero (treated as upstream data integrity error).
    """
    # Missing-input branches first (structurally distinct from wide-risk)
    if entry_ref is None:
        return RiskResolution(
            execution_label="NOT_EXECUTABLE",
            execution_risk_status="missing_inputs",
            risk_pct=None,
            risk_atr=None,
            reason_codes=("no_entry_ref",),
        )
    if stop_ref is None:
        return RiskResolution(
            execution_label="NOT_EXECUTABLE",
            execution_risk_status="missing_inputs",
            risk_pct=None,
            risk_atr=None,
            reason_codes=("no_stop_ref",),
        )
    if atr is None:
        return RiskResolution(
            execution_label="NOT_EXECUTABLE",
            execution_risk_status="missing_inputs",
            risk_pct=None,
            risk_atr=None,
            reason_codes=("fill_realism_unresolved",),
        )

    risk_pct = _compute_risk_pct(entry_ref, stop_ref)
    risk_atr = _compute_risk_atr(entry_ref, stop_ref, atr)

    if risk_atr <= MAX_RISK_ATR_EXECUTABLE:
        return RiskResolution(
            execution_label="EXECUTABLE",
            execution_risk_status="ok",
            risk_pct=risk_pct,
            risk_atr=risk_atr,
            reason_codes=(
                "entry_ref_present",
                "stop_ref_present",
                "execution_risk_ok",
            ),
        )
    if risk_atr <= MAX_RISK_ATR_SIZE_REDUCED:
        # SIZE_REDUCED — label only. No sizing arithmetic.
        return RiskResolution(
            execution_label="SIZE_REDUCED",
            execution_risk_status="above_normal_below_reject",
            risk_pct=risk_pct,
            risk_atr=risk_atr,
            reason_codes=(
                "entry_ref_present",
                "stop_ref_present",
                "execution_risk_above_normal_below_reject",
                "size_reduced",
            ),
        )
    # risk_atr > 3.0 with all of entry/stop/atr valid
    return RiskResolution(
        execution_label="WAIT_BETTER_ENTRY",
        execution_risk_status="too_wide",
        risk_pct=risk_pct,
        risk_atr=risk_atr,
        reason_codes=(
            "entry_ref_present",
            "stop_ref_present",
            "execution_risk_too_wide",
            "wait_better_entry",
        ),
    )


def compute_atr_wilder(true_ranges: list[float]) -> float | None:
    """Wilder's True Range smoothing for ATR_WINDOW=14.

    Standard formula: ATR_i = (ATR_{i-1} * (n-1) + TR_i) / n where
    n = ATR_WINDOW. Initial seed is the simple mean of the first n
    true ranges. Returns None if fewer than `ATR_WINDOW` true ranges
    are available.
    """
    n = ATR_WINDOW
    if len(true_ranges) < n:
        return None
    atr = sum(true_ranges[:n]) / n
    for tr in true_ranges[n:]:
        atr = (atr * (n - 1) + tr) / n
    return atr


__all__ = [
    "ATR_WINDOW",
    "MAX_RISK_ATR_EXECUTABLE",
    "MAX_RISK_ATR_SIZE_REDUCED",
    "RiskResolution",
    "resolve_risk_branch",
    "compute_atr_wilder",
]
