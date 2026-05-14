"""Decision Engine v1 — cross-layer reason-code purity static check
+ forbidden-field absence check + module-level static-import sanity.

Spec §2 Step 6 + §3.6 + §13 (architectural separation).

Three classes of check:

1. **Reason-code partition purity.** A code lives in exactly one of
   {setup_reason_codes, execution_reason_codes, context_reason_codes};
   any code that appears in two partitions is non-compliance.

2. **Allow-list membership.** Every observed reason code must belong
   to its layer's LOCKED initial set. No carve-outs without a fresh
   pre-reg.

3. **Forbidden-field absence + forbidden-namespace absence.** A v1
   events parquet must not contain `final_action`, `selection_score`,
   etc. (spec §3.6). A v1 module must not import from `decision_engine`
   (v0 source untouched) or write to forbidden output namespaces
   (`*paper_execution*`, `*tier_a_paper*`, `*portfolio_merge_paper*`).

This module is import-cheap: it does not depend on pandas, parquet, or
any I/O. The runner invokes these checks before write.

Static-restriction enforcement (§3.3.1 + §3.4):
  - The fields `paper_expired_flag`, `paper_signal_age`,
    `paper_validity_metadata_missing`, and `earliness_score_pct` may
    only be referenced by `write.py`. Referencing them in `label.py`,
    `risk.py`, `paper_stream_link.py`, or `purity_check.py` itself is
    non-compliance. We expose `STATIC_REFERENCE_RESTRICTED_FIELDS` so
    a future Tier 1+ static analyzer (or CI grep job) can enforce.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Iterable

from decision_engine_v1.schema import (
    CONTEXT_REASON_CODES_INITIAL,
    EXECUTION_REASON_CODES_INITIAL,
    FORBIDDEN_FIELDS,
    SETUP_REASON_CODES_INITIAL,
    is_forbidden_score_field,
)


class PurityViolation(RuntimeError):
    """Raised when reason codes leak across layers or violate allow-list."""


class ForbiddenFieldError(RuntimeError):
    """Raised when a forbidden v1 events column appears."""


class ForbiddenNamespaceError(RuntimeError):
    """Raised when a forbidden output namespace (`*paper_execution*` etc.)
    is targeted by `write.py`."""


# ─── Static-restriction lists (enforced by external linter / CI) ─────────

# Files allowed to reference these passive-carry fields. Any other file
# under decision_engine_v1/ referencing them is non-compliance.
WRITE_ONLY_REFERENCED_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "paper_expired_flag",
        "paper_signal_age",
        "paper_validity_metadata_missing",
        "earliness_score_pct",
    }
)

# Output-namespace patterns (substrings) that v1 must NEVER write to.
FORBIDDEN_OUTPUT_NAMESPACE_SUBSTRINGS: Final[tuple[str, ...]] = (
    "paper_execution",
    "tier_a_paper",
    "portfolio_merge_paper",
)

# Forbidden module filenames under decision_engine_v1/ (spec §1).
FORBIDDEN_MODULE_NAMES: Final[frozenset[str]] = frozenset(
    {
        "score.py",
        "priors.py",
        "portfolio.py",
        "rank.py",
        "selection.py",
        "weights.py",
        "bucket_priors.py",
        "expectancy.py",
        "paper_lines.py",
        "paper_eligibility.py",
        "paper_emission.py",
        "ema_compute.py",
        "ema_pilot4_replay.py",
        "ema_pilot7_replay.py",
        "tier_a_paper.py",
    }
)


@dataclass(frozen=True)
class PurityReport:
    setup_codes_seen: frozenset[str]
    execution_codes_seen: frozenset[str]
    context_codes_seen: frozenset[str]
    cross_layer_violations: tuple[str, ...]
    unknown_codes: tuple[tuple[str, str], ...]  # (layer, code)

    @property
    def ok(self) -> bool:
        return not (self.cross_layer_violations or self.unknown_codes)


def check_reason_code_purity(
    setup_codes: Iterable[Iterable[str]],
    execution_codes: Iterable[Iterable[str]],
    context_codes: Iterable[Iterable[str]],
) -> PurityReport:
    """Aggregate reason codes across all events and verify partition
    purity + allow-list membership. Returns a `PurityReport`; the
    caller is responsible for `enforce_purity_or_raise()`.

    Inputs are iterables of per-event reason-code lists (one inner
    iterable per event).
    """
    setup_seen: set[str] = set()
    execution_seen: set[str] = set()
    context_seen: set[str] = set()

    for codes in setup_codes:
        setup_seen.update(codes)
    for codes in execution_codes:
        execution_seen.update(codes)
    for codes in context_codes:
        context_seen.update(codes)

    # Cross-layer leak detection
    overlaps = (
        (setup_seen & execution_seen)
        | (setup_seen & context_seen)
        | (execution_seen & context_seen)
    )

    # Allow-list violations
    unknown: list[tuple[str, str]] = []
    for code in setup_seen - set(SETUP_REASON_CODES_INITIAL):
        unknown.append(("setup", code))
    for code in execution_seen - set(EXECUTION_REASON_CODES_INITIAL):
        unknown.append(("execution", code))
    for code in context_seen - set(CONTEXT_REASON_CODES_INITIAL):
        unknown.append(("context", code))

    return PurityReport(
        setup_codes_seen=frozenset(setup_seen),
        execution_codes_seen=frozenset(execution_seen),
        context_codes_seen=frozenset(context_seen),
        cross_layer_violations=tuple(sorted(overlaps)),
        unknown_codes=tuple(sorted(unknown)),
    )


def enforce_purity_or_raise(report: PurityReport) -> None:
    """Convert non-OK report into a fail-fast exception."""
    if report.cross_layer_violations:
        joined = ", ".join(report.cross_layer_violations)
        raise PurityViolation(
            f"reason-code cross-layer leak: {joined} appear in more than "
            "one of {setup, execution, context}; design pre-reg §10 + "
            "implementation spec §2 Step 6 require strict partitioning"
        )
    if report.unknown_codes:
        joined = ", ".join(f"{layer}:{code}" for layer, code in report.unknown_codes)
        raise PurityViolation(
            f"reason codes outside LOCKED initial sets: {joined}; "
            "additive amendment to design §10.1–10.3 required before "
            "any new reason code may be emitted"
        )


def check_forbidden_fields(columns: Iterable[str]) -> None:
    """Raise on any forbidden field. (Spec §3.6 hard list + wildcard.)"""
    for col in columns:
        if col in FORBIDDEN_FIELDS:
            raise ForbiddenFieldError(
                f"forbidden field {col!r} present in v1 events schema "
                "(implementation spec §3.6)"
            )
        if is_forbidden_score_field(col):
            raise ForbiddenFieldError(
                f"forbidden score/rank field {col!r} present in v1 events "
                "schema (only `earliness_score_pct` is allowed; spec §3.6)"
            )


def check_forbidden_output_namespace(path: str) -> None:
    """Raise on any output path that targets a forbidden namespace.

    `path` is the destination filename / suffix / parquet path being
    asked of `write.py`. We look for substring matches against the
    LOCKED forbidden-namespace list. Decision Engine v1 must never
    round-trip into the upstream paper streams.
    """
    lowered = path.lower()
    for pattern in FORBIDDEN_OUTPUT_NAMESPACE_SUBSTRINGS:
        if pattern in lowered:
            raise ForbiddenNamespaceError(
                f"output path {path!r} targets forbidden namespace "
                f"matching {pattern!r}; v1 must write only to "
                "`output/decision_engine_v1_*` (spec §3.6 + §13)"
            )


__all__ = [
    "PurityViolation",
    "ForbiddenFieldError",
    "ForbiddenNamespaceError",
    "WRITE_ONLY_REFERENCED_FIELDS",
    "FORBIDDEN_OUTPUT_NAMESPACE_SUBSTRINGS",
    "FORBIDDEN_MODULE_NAMES",
    "PurityReport",
    "check_reason_code_purity",
    "enforce_purity_or_raise",
    "check_forbidden_fields",
    "check_forbidden_output_namespace",
]
