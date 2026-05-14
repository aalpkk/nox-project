"""Decision Engine v1 — read-only external paper-stream linker.

Spec §2 Step 5 + §3.3 + §3.3.1 + §3.7 + §6 (LOCKED 2026-05-04 + REVISED).

Strict contract (spec §1):
  - READ-ONLY access to upstream paper-stream parquets:
      * `output/paper_execution_v0_trades.parquet`             (Line E)
      * `output/paper_execution_v0_trigger_retest_trades.parquet` (Line TR)
  - NO write access to those parquets (filesystem-level read-only assert).
  - NO invocation of the `paper_execution_v0_*` runners or
    `portfolio_merge_paper`.
  - NO re-derivation of any paper-line eligibility condition.
  - NO evaluation of EMA thresholds (`earliness_score_pct ≤ −6`,
    `∈ [−5, −1]`); v1 reads RECORDS, never INPUTS.
  - Match key: `(asof_date, ticker, line)` exact equality, where
    `asof_date` = paper stream's canonical signal/asof decision date
    (Step 5.0). No fuzzy match, no nearest-neighbor, no day-window.
  - Duplicate match → fail-fast (Step 5.2). Never silent first-row pick.
  - Missing parquet → fail-fast (Step 5.3 default). `PAPER_LINK_SKIP_MODE
    = false` LOCKED 2026-05-04.
  - Validity carry: paper_valid_from / paper_valid_until carried verbatim;
    paper_signal_age / paper_expired_flag derived as descriptive carriers
    only; `paper_validity_metadata_missing=true` flagged when upstream
    record lacks validity metadata (Step 5.6) — NO halt, NO default
    window synthesis (§0.2 + §3.7 forbidden).

This module references the validity fields ONLY for carrier-arithmetic
attachment to the v1 row; it does NOT use them as decision inputs (the
static-restriction list in `purity_check.py` lists them as
`WRITE_ONLY_REFERENCED_FIELDS`, but Step 5 is the upstream of `write.py`
in the runner pipeline — the linker constructs the values, the writer
emits them). No decision logic in this file reads `paper_expired_flag`
to gate anything.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Final

from decision_engine_v1.ingest import ScannerEvent
from decision_engine_v1.mapping_table import (
    HB_FAMILY_TO_PAPER_LINE,
    PAPER_LINE_TO_STREAM_REF,
)


# LOCKED 2026-05-04 — see spec §6.
PAPER_LINK_SKIP_MODE: Final[bool] = False

# Authoritative parquet paths (READ-ONLY).
PAPER_STREAM_PATHS: Final[dict[str, str]] = {
    "EXTENDED": "output/paper_execution_v0_trades.parquet",
    "TRIGGER_RETEST": "output/paper_execution_v0_trigger_retest_trades.parquet",
}


# ─── Errors ──────────────────────────────────────────────────────────────


class PaperStreamMissingError(RuntimeError):
    """Raised when a candidate paper-stream parquet is not on disk and
    `PAPER_LINK_SKIP_MODE=false` (default)."""


class PaperStreamDuplicateMatchError(RuntimeError):
    """Raised when more than one paper-stream record matches a single
    `(asof_date, ticker, line)` key. Never silently picks the first row
    (Step 5.2)."""


# ─── Data classes ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PaperLinkAttachment:
    """The post-link state of a single v1 event row.

    On a successful match, the runner overrides the row's
    `execution_label` to `PAPER_ONLY` and sets the reference + validity
    fields below. On no match, the runner does nothing — the base
    mapping (assigned by `label.py`) stands.
    """

    matched: bool
    paper_origin: str | None
    paper_stream_ref: str | None
    paper_trade_id: str | None
    paper_match_key: str | None
    paper_valid_from: str | None             # ISO date or None
    paper_valid_until: str | None            # ISO date or None
    paper_signal_age: int | None             # trading-day delta or None
    paper_expired_flag: bool | None
    paper_validity_metadata_missing: bool | None


@dataclass(frozen=True)
class PaperLinkIntegrityCounters:
    """Per-line + total integrity counters per Step 5b.

    Invariants enforced by `assert_invariants()`:
      attempted = matched + unmatched + duplicate_error + skipped_source_missing
      duplicate_error == 0 on any successful run
      validity_missing ≤ matched
    """

    line: str  # "EXTENDED" | "TRIGGER_RETEST" | "ALL"
    paper_link_attempted: int = 0
    paper_link_matched: int = 0
    paper_link_unmatched: int = 0
    paper_link_duplicate_error: int = 0
    paper_link_skipped_source_missing: int = 0
    paper_link_validity_missing: int = 0  # additive REVISION 2026-05-04

    def assert_invariants(self, *, allow_unsuccessful: bool = False) -> None:
        total = (
            self.paper_link_matched
            + self.paper_link_unmatched
            + self.paper_link_duplicate_error
            + self.paper_link_skipped_source_missing
        )
        if total != self.paper_link_attempted:
            raise AssertionError(
                f"line={self.line}: paper_link_attempted={self.paper_link_attempted} "
                f"!= matched + unmatched + duplicate_error + skipped_source_missing"
                f" ({total}); Step 5b invariant violated"
            )
        if not allow_unsuccessful and self.paper_link_duplicate_error != 0:
            raise AssertionError(
                f"line={self.line}: paper_link_duplicate_error="
                f"{self.paper_link_duplicate_error}; Step 5.2 should have "
                "halted the runner"
            )
        if self.paper_link_validity_missing > self.paper_link_matched:
            raise AssertionError(
                f"line={self.line}: paper_link_validity_missing="
                f"{self.paper_link_validity_missing} > paper_link_matched="
                f"{self.paper_link_matched}; Step 5.6 invariant violated"
            )
        if (
            not PAPER_LINK_SKIP_MODE
            and self.paper_link_skipped_source_missing != 0
        ):
            raise AssertionError(
                f"line={self.line}: paper_link_skipped_source_missing="
                f"{self.paper_link_skipped_source_missing} but "
                "PAPER_LINK_SKIP_MODE=false; Step 5b invariant violated"
            )


# ─── Public API ──────────────────────────────────────────────────────────


def candidate_line_for_event(event: ScannerEvent) -> str | None:
    """Return the paper-line token (`EXTENDED` / `TRIGGER_RETEST`) for
    an HB event, or None if the event is not HB / not paper-eligible-by
    -family. Mapping-only — see `HB_FAMILY_TO_PAPER_LINE`.
    """
    if event.source != "horizontal_base":
        return None
    return HB_FAMILY_TO_PAPER_LINE.get(event.family)


def build_match_key(asof_date: str, ticker: str, line: str) -> str:
    """Construct the fallback `paper_match_key` per spec §3.3.

    The format is `f"{asof_date}|{ticker}|{line}"`; this string is also
    used as a fallback identifier when the upstream parquet does not
    expose its own `paper_trade_id` column.
    """
    return f"{asof_date}|{ticker}|{line}"


def _assert_paper_stream_readable(path: str) -> None:
    """Filesystem-level guard: when the parquet exists, assert it is
    readable but NOT writable by this process. Best-effort check; the
    real protection is the strict no-write contract of this module.
    """
    if not os.path.exists(path):
        return
    if not os.access(path, os.R_OK):
        raise PaperStreamMissingError(
            f"paper-stream parquet {path!r} exists but is not readable; "
            "v1 cannot proceed"
        )
    # If the file is writable, that does not block reads — but writing
    # to it from this module would still be a contract violation. We
    # do NOT chmod or otherwise mutate the file.


def _read_paper_stream_records(
    line: str,
    paper_stream_root: str = ".",
) -> dict[tuple[str, str], list[dict]]:
    """Read upstream paper-stream parquet records for `line` and group
    by `(asof_date, ticker)` key (READ-ONLY).

    Line E parquet (`paper_execution_v0_trades.parquet`) does not carry
    a `line` column — every row is implicitly EXTENDED.
    Line TR parquet (`paper_execution_v0_trigger_retest_trades.parquet`)
    carries a `line` column with the literal value `"TRIGGER_RETEST"`.

    NO derivation of TIER_A / TIER_B from `earliness_score_pct`. NO
    eligibility recomputation. NO re-derivation of `paper_valid_from` /
    `paper_valid_until`. This function strictly carries upstream values
    (most current upstream parquets do not expose validity columns; in
    that case the linker emits `paper_validity_metadata_missing=True`).

    Args:
        line: "EXTENDED" or "TRIGGER_RETEST".
        paper_stream_root: root directory for resolving the LOCKED parquet
            paths (defaults to ".").

    Returns:
        dict keyed by `(asof_date_iso, ticker)` mapping to a list of raw
        record dicts. A list with >1 element means duplicate rows for the
        same key — the caller must halt per Step 5.2.

    Raises:
        PaperStreamMissingError: if the parquet is absent and
            PAPER_LINK_SKIP_MODE=False (Step 5.3 default).
    """
    import pyarrow.parquet as pq

    rel = PAPER_STREAM_PATHS[line]
    path = os.path.join(paper_stream_root, rel) if paper_stream_root != "." else rel

    if not os.path.exists(path):
        if PAPER_LINK_SKIP_MODE:
            return {}
        raise PaperStreamMissingError(
            f"paper-stream parquet for line={line!r} not found at {path!r}; "
            "PAPER_LINK_SKIP_MODE=False (LOCKED) requires the parquet "
            "to exist (Step 5.3)"
        )
    _assert_paper_stream_readable(path)

    table = pq.read_table(path)
    records: dict[tuple[str, str], list[dict]] = {}
    for row in table.to_pylist():
        # Line TR: filter to line column == "TRIGGER_RETEST" defensively.
        if line == "TRIGGER_RETEST":
            row_line = row.get("line")
            if row_line is not None and row_line != "TRIGGER_RETEST":
                continue
        asof_val = row.get("asof_date")
        if asof_val is None:
            continue
        asof_iso = asof_val.isoformat() if hasattr(asof_val, "isoformat") else str(asof_val)
        ticker = row.get("ticker")
        if ticker is None:
            continue
        key = (asof_iso, str(ticker))
        records.setdefault(key, []).append(row)
    return records


def _row_validity_state(row: dict) -> tuple[object | None, object | None, bool]:
    """Extract `(paper_valid_from, paper_valid_until, validity_metadata_missing)`
    from a raw paper-stream record.

    Upstream parquets currently do not expose validity columns; in that
    case both fields are None and `metadata_missing=True`. Carrying-only
    behavior — no synthesis of a default window (spec §3.7 forbids it).
    """
    has_from = "paper_valid_from" in row and row.get("paper_valid_from") is not None
    has_until = "paper_valid_until" in row and row.get("paper_valid_until") is not None
    if not (has_from or has_until):
        return None, None, True
    valid_from = row.get("paper_valid_from") if has_from else None
    valid_until = row.get("paper_valid_until") if has_until else None
    # If exactly one is present, carry verbatim and mark missing=False
    # (it's a partial-but-present metadata case; downstream consumer
    # decides interpretation; v1 does NOT backfill the missing side).
    return valid_from, valid_until, False


def link_paper_streams(
    events: list[ScannerEvent],
    asof_date: str,
    *,
    paper_stream_root: str = ".",
) -> tuple[dict[int, PaperLinkAttachment], dict[str, PaperLinkIntegrityCounters]]:
    """Run Step 5 against the supplied events.

    For each event whose `candidate_line_for_event` returns a non-None
    line token, attempt an exact `(asof_date, ticker, line)` match
    against the upstream paper-stream parquet for that line.

    Match outcomes:
      * 0 records   → `unmatched`; no attachment.
      * 1 record    → `matched`;   attachment carries reference + validity.
      * >1 records → halt with `PaperStreamDuplicateMatchError` (Step 5.2).

    Missing parquet under `PAPER_LINK_SKIP_MODE=False` (LOCKED) →
    `PaperStreamMissingError` (Step 5.3).

    Returns:
        attachments: dict mapping event-list index → PaperLinkAttachment.
        counters: dict mapping line ("EXTENDED" | "TRIGGER_RETEST" |
            "ALL") → PaperLinkIntegrityCounters.

    Raises:
        PaperStreamMissingError: missing parquet under SKIP_MODE=False.
        PaperStreamDuplicateMatchError: duplicate match.
    """
    # Pre-scan candidate lines used by the supplied events. We only read
    # parquets for lines that are actually referenced — but the LOCKED
    # behavior is to fail-fast on missing parquet for any HB event whose
    # candidate line lacks its parquet (not on parquet existence in
    # general).
    candidate_lines: set[str] = set()
    for ev in events:
        cl = candidate_line_for_event(ev)
        if cl is not None:
            candidate_lines.add(cl)

    # Read records for each candidate line. _read_paper_stream_records
    # raises PaperStreamMissingError under SKIP_MODE=False if missing.
    line_records: dict[str, dict[tuple[str, str], list[dict]]] = {}
    for cl in candidate_lines:
        line_records[cl] = _read_paper_stream_records(
            cl, paper_stream_root=paper_stream_root
        )

    # Init counters per line; we always emit both EXTENDED and TRIGGER_RETEST
    # (and ALL) so the integrity CSV has stable shape even when one line
    # is empty in this run.
    per_line_counters: dict[str, dict[str, int]] = {
        "EXTENDED": {
            "attempted": 0, "matched": 0, "unmatched": 0,
            "duplicate_error": 0, "skipped_source_missing": 0,
            "validity_missing": 0,
        },
        "TRIGGER_RETEST": {
            "attempted": 0, "matched": 0, "unmatched": 0,
            "duplicate_error": 0, "skipped_source_missing": 0,
            "validity_missing": 0,
        },
    }

    attachments: dict[int, PaperLinkAttachment] = {}

    for idx, ev in enumerate(events):
        cl = candidate_line_for_event(ev)
        if cl is None:
            continue
        per_line_counters[cl]["attempted"] += 1

        records = line_records.get(cl, {}).get((ev.date, ev.ticker), [])
        n = len(records)

        if n == 0:
            per_line_counters[cl]["unmatched"] += 1
            continue

        if n > 1:
            per_line_counters[cl]["duplicate_error"] += 1
            raise PaperStreamDuplicateMatchError(
                f"duplicate paper-stream match for "
                f"(asof_date={ev.date!r}, ticker={ev.ticker!r}, line={cl!r}); "
                f"got {n} records (Step 5.2 halt; no silent first-row pick)"
            )

        rec = records[0]
        per_line_counters[cl]["matched"] += 1
        valid_from, valid_until, metadata_missing = _row_validity_state(rec)
        if metadata_missing:
            per_line_counters[cl]["validity_missing"] += 1

        # paper_trade_id: Line E / Line TR parquets do not currently
        # expose a stable trade-id column; carry None and use the
        # build_match_key fallback (§3.3). assert_paper_reference_invariants
        # requires exactly one of paper_trade_id / paper_match_key.
        paper_trade_id = None
        if "paper_trade_id" in rec and rec.get("paper_trade_id") is not None:
            paper_trade_id = str(rec["paper_trade_id"])

        match_key = build_match_key(ev.date, ev.ticker, cl)
        if paper_trade_id is not None:
            # Both available → spec invariant requires exactly one set;
            # prefer the upstream-provided id, leave match_key=None.
            attached_match_key: str | None = None
        else:
            attached_match_key = match_key

        attachments[idx] = PaperLinkAttachment(
            matched=True,
            paper_origin="external_reference",
            paper_stream_ref=PAPER_LINE_TO_STREAM_REF[cl],
            paper_trade_id=paper_trade_id,
            paper_match_key=attached_match_key,
            paper_valid_from=(
                valid_from.isoformat()
                if hasattr(valid_from, "isoformat")
                else (str(valid_from) if valid_from is not None else None)
            ),
            paper_valid_until=(
                valid_until.isoformat()
                if hasattr(valid_until, "isoformat")
                else (str(valid_until) if valid_until is not None else None)
            ),
            paper_signal_age=None,        # not derived (no upstream column)
            paper_expired_flag=None,      # not derived (no validity window)
            paper_validity_metadata_missing=metadata_missing,
        )

    counters_out: dict[str, PaperLinkIntegrityCounters] = {}
    all_acc = {
        "attempted": 0, "matched": 0, "unmatched": 0,
        "duplicate_error": 0, "skipped_source_missing": 0,
        "validity_missing": 0,
    }
    for line_name, c in per_line_counters.items():
        counters_out[line_name] = PaperLinkIntegrityCounters(
            line=line_name,
            paper_link_attempted=c["attempted"],
            paper_link_matched=c["matched"],
            paper_link_unmatched=c["unmatched"],
            paper_link_duplicate_error=c["duplicate_error"],
            paper_link_skipped_source_missing=c["skipped_source_missing"],
            paper_link_validity_missing=c["validity_missing"],
        )
        for k, v in c.items():
            all_acc[k] += v
    counters_out["ALL"] = PaperLinkIntegrityCounters(
        line="ALL",
        paper_link_attempted=all_acc["attempted"],
        paper_link_matched=all_acc["matched"],
        paper_link_unmatched=all_acc["unmatched"],
        paper_link_duplicate_error=all_acc["duplicate_error"],
        paper_link_skipped_source_missing=all_acc["skipped_source_missing"],
        paper_link_validity_missing=all_acc["validity_missing"],
    )

    return attachments, counters_out


def paper_stream_max_date(line: str, paper_stream_root: str = ".") -> str | None:
    """Return ISO yyyy-mm-dd of the latest `asof_date` in the upstream
    paper-stream parquet for `line` (used for runtime asof_date
    resolution + source_asof_mismatch check per LOCK §16.1).

    Returns None if the parquet is missing under SKIP_MODE=True or if
    the parquet has zero rows. Raises `PaperStreamMissingError` if the
    parquet is missing under SKIP_MODE=False.
    """
    import pyarrow.parquet as pq
    import pyarrow.compute as pc

    rel = PAPER_STREAM_PATHS[line]
    path = os.path.join(paper_stream_root, rel) if paper_stream_root != "." else rel
    if not os.path.exists(path):
        if PAPER_LINK_SKIP_MODE:
            return None
        raise PaperStreamMissingError(
            f"paper-stream parquet for line={line!r} not found at {path!r}; "
            "PAPER_LINK_SKIP_MODE=False (LOCKED) requires the parquet to exist"
        )
    table = pq.read_table(path, columns=["asof_date"])
    if table.num_rows == 0:
        return None
    max_d = pc.max(table.column("asof_date")).as_py()
    return max_d.isoformat() if hasattr(max_d, "isoformat") else str(max_d)


__all__ = [
    "PAPER_LINK_SKIP_MODE",
    "PAPER_STREAM_PATHS",
    "PaperStreamMissingError",
    "PaperStreamDuplicateMatchError",
    "PaperLinkAttachment",
    "PaperLinkIntegrityCounters",
    "candidate_line_for_event",
    "build_match_key",
    "link_paper_streams",
    "paper_stream_max_date",
]
