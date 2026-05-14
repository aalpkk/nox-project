"""Decision Engine v1 — Step 0 live-scope validation (MANDATORY first step).

Spec §2 Step 0. The runner MUST execute this before any other v1 logic.

Behaviors:
  - Enumerate every (source, family, state, timeframe) cell emitted by
    the live scanner outputs for the run's `asof` date.
  - Hard-fail on any cell not in `mapping_table.MAPPING_TABLE`. No
    `--force` override. No silent default.
  - Hard-fail on any source not in `V1_SOURCE_WHITELIST` (alpha legacy
    Q5 resolution).
  - Log informationally when a known cell is absent from the run.

This module raises typed exceptions; the runner catches them and exits
nonzero. Tier 0: code-only — runner is not authorized to execute, but
this validator is still importable / testable independently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from decision_engine_v1.mapping_table import (
    MAPPING_TABLE,
    V1_SOURCE_WHITELIST,
    MappingKey,
)


class UnmappedCellError(RuntimeError):
    """Raised when an emitted cell is not in the LOCKED mapping table."""


class UnmappedSourceError(RuntimeError):
    """Raised when an emitted source is not in the v1 whitelist."""


@dataclass(frozen=True)
class LiveScopeReport:
    """Result of Step 0 validation.

    A non-empty `unmapped_cells` or `unmapped_sources` is a fail-fast
    condition; the runner must halt. `unseen_cells` is informational.
    """

    asof_date: str
    emitted_cells: tuple[MappingKey, ...]
    unseen_cells: tuple[MappingKey, ...]
    unmapped_cells: tuple[MappingKey, ...] = field(default_factory=tuple)
    unmapped_sources: tuple[str, ...] = field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        return not (self.unmapped_cells or self.unmapped_sources)


def validate_live_scope(
    asof_date: str,
    emitted_cells: Iterable[MappingKey],
) -> LiveScopeReport:
    """Run Step 0 validation against an iterable of emitted cells.

    Each cell is a (source, family, state, timeframe) tuple. The function
    returns a `LiveScopeReport` summarizing the outcome; it does NOT
    raise on its own. Callers (the runner) are responsible for calling
    `enforce_live_scope_or_raise(report)` to convert a non-OK report
    into a halt.
    """
    emitted = tuple(emitted_cells)
    unmapped: list[MappingKey] = []
    unmapped_sources_set: set[str] = set()
    mapped_set = set(MAPPING_TABLE.keys())

    for cell in emitted:
        source = cell[0]
        if source not in V1_SOURCE_WHITELIST:
            unmapped_sources_set.add(source)
            continue
        if cell not in mapped_set:
            unmapped.append(cell)

    emitted_set = set(emitted)
    unseen = tuple(sorted(mapped_set - emitted_set))

    return LiveScopeReport(
        asof_date=asof_date,
        emitted_cells=emitted,
        unseen_cells=unseen,
        unmapped_cells=tuple(unmapped),
        unmapped_sources=tuple(sorted(unmapped_sources_set)),
    )


def enforce_live_scope_or_raise(report: LiveScopeReport) -> None:
    """Convert a non-OK report into a fail-fast exception.

    Raises `UnmappedSourceError` first if any source is non-whitelisted
    (since cell-level mapping is undefined for unknown sources), then
    `UnmappedCellError` for any otherwise-unmapped cell.
    """
    if report.unmapped_sources:
        joined = ", ".join(report.unmapped_sources)
        raise UnmappedSourceError(
            f"unmapped source(s): {joined}; mapping review §2.10 "
            "(alpha legacy resolution Q5) requires explicit additive "
            "extension or `not_in_v1_scope` exclusion before this "
            "source can be ingested"
        )
    if report.unmapped_cells:
        first = report.unmapped_cells[0]
        source, family, state, tf = first
        n_more = len(report.unmapped_cells) - 1
        suffix = f" (+{n_more} more)" if n_more else ""
        raise UnmappedCellError(
            f"unmapped cell: source={source}, family={family}, "
            f"state={state}, tf={tf}{suffix}; mapping review §4 must "
            "be extended via additive amendment before this run can "
            "proceed"
        )


__all__ = [
    "UnmappedCellError",
    "UnmappedSourceError",
    "LiveScopeReport",
    "validate_live_scope",
    "enforce_live_scope_or_raise",
]
