"""Decision Engine v1 — parquet writer (gated; Tier 0 fail-closed).

Spec §2 Step 7 + §7 (output authorization tiers).

Tier 0 contract: the writer is **not authorized to execute**. Calling
`write_events_parquet()` raises `WriterNotAuthorizedError` immediately.
The function exists in this scaffold so that callers (the runner) can
import it cleanly; converting it from "scaffolded but disabled" to
"executable" requires a separate future Tier 1+ LOCK that explicitly
re-affirms the §6 thresholds and PAPER_LINK_SKIP_MODE setting.

Per spec, this is the SINGLE module under `decision_engine_v1/` that is
permitted to reference the passive-carry / descriptive fields:

    - `paper_expired_flag`
    - `paper_signal_age`
    - `paper_validity_metadata_missing`
    - `earliness_score_pct`
    - `ema_context_tag`

Other modules MUST NOT reference these fields (cross-layer purity static
restriction). The list is mirrored in `purity_check.WRITE_ONLY_REFERENCED
_FIELDS`.

Forbidden output namespaces (`*paper_execution*`, `*tier_a_paper*`,
`*portfolio_merge_paper*`) are guarded by `purity_check
.check_forbidden_output_namespace()`.
"""

from __future__ import annotations

from typing import Final

from decision_engine_v1.purity_check import (
    check_forbidden_fields,
    check_forbidden_output_namespace,
)
from decision_engine_v1.schema import ALL_EVENT_COLUMNS


# Output paths (LOCKED prefix). All v1 outputs MUST start with this.
V1_OUTPUT_PREFIX: Final[str] = "output/decision_engine_v1_"

EVENTS_PARQUET_PATH: Final[str] = f"{V1_OUTPUT_PREFIX}events.parquet"
EVENTS_CSV_PATH: Final[str] = f"{V1_OUTPUT_PREFIX}events.csv"
PAPER_LINK_INTEGRITY_CSV_PATH: Final[str] = (
    f"{V1_OUTPUT_PREFIX}paper_link_integrity.csv"
)
LABEL_TABLE_MD_PATH: Final[str] = f"{V1_OUTPUT_PREFIX}label_table.md"

# Tier 2 LOCKED paths (`memory/decision_engine_v1_tier2_label_table_spec.md`
# §12.3). The label-distribution CSV and run-summary markdown are NEW paths
# (NOT the Tier 1 dry-run mirror filenames) per LOCK §12.6.
TIER2_LABEL_DISTRIBUTION_CSV_PATH: Final[str] = (
    f"{V1_OUTPUT_PREFIX}tier2_label_distribution.csv"
)
TIER2_LABEL_TABLE_SUMMARY_MD_PATH: Final[str] = (
    f"{V1_OUTPUT_PREFIX}tier2_label_table_summary.md"
)

# Tier 3 LOCKED path (`memory/decision_engine_v1_tier3_review_report_spec.md`
# §12.2 Q1). Single authorized output for the human-readable review report.
TIER3_REVIEW_REPORT_MD_PATH: Final[str] = (
    f"{V1_OUTPUT_PREFIX}tier3_review_report.md"
)


class WriterNotAuthorizedError(RuntimeError):
    """Raised under Tier 0: the writer refuses to execute."""


def _enforce_v1_output_prefix(path: str) -> None:
    """A v1 output path MUST start with `output/decision_engine_v1_`."""
    if not path.startswith(V1_OUTPUT_PREFIX):
        raise WriterNotAuthorizedError(
            f"v1 output path {path!r} does not start with the locked "
            f"prefix {V1_OUTPUT_PREFIX!r}; only outputs under that "
            "namespace are permitted (spec §7 + §13)"
        )
    check_forbidden_output_namespace(path)


def assert_v1_writer_compliance(
    columns: list[str],
    output_path: str,
) -> None:
    """Run all pre-write static guards.

    Raises on any of:
      - forbidden field present in `columns` (spec §3.6)
      - any `*_score` field other than `earliness_score_pct`
      - output path outside `output/decision_engine_v1_*`
      - output path matching forbidden namespace substrings
      - column missing from the LOCKED schema (spec §3)
    """
    check_forbidden_fields(columns)
    _enforce_v1_output_prefix(output_path)

    # Schema completeness: every emitted column must be a known v1 column.
    known = set(ALL_EVENT_COLUMNS)
    unknown = [c for c in columns if c not in known]
    if unknown:
        raise WriterNotAuthorizedError(
            f"unknown columns in v1 events frame: {unknown}; only the "
            "LOCKED schema (§3) may be written"
        )


def write_events_parquet(
    events_rows,
    output_path: str = EVENTS_PARQUET_PATH,
    *,
    integrity_counters=None,
    paper_link_integrity_csv_path: str = PAPER_LINK_INTEGRITY_CSV_PATH,
    tier2_authorized: bool = False,
) -> None:
    """Write the v1 events parquet + mandatory paper-link integrity CSV.

    Tier 0/1: this function is **disabled** unless `tier2_authorized=True`
    is passed by the runner under the LOCKED Tier 2 label-table spec
    (`memory/decision_engine_v1_tier2_label_table_spec.md` §12). The
    integrity CSV is mandatory at every parquet-writing tier (Tier 2+);
    its emission is co-gated with the parquet by this same authorization
    check (impl spec §5b + §7).

    Args:
        events_rows: list[dict] — one dict per v1 event, keys must be a
            subset of `ALL_EVENT_COLUMNS`, and `assert_v1_writer_compliance`
            confirms there are no forbidden fields.
        output_path: parquet output path; must start with the LOCKED
            `output/decision_engine_v1_` prefix.
        integrity_counters: dict[str, PaperLinkIntegrityCounters] keyed
            by line ∈ {"EXTENDED", "TRIGGER_RETEST", "ALL"}. Mandatory.
        paper_link_integrity_csv_path: integrity CSV path (defaults to
            the Tier 2+ canonical `output/decision_engine_v1_paper_link_
            integrity.csv`).
        tier2_authorized: must be True under Tier 2 LOCK.

    Raises:
        WriterNotAuthorizedError: when `tier2_authorized=False`, when
            `events_rows` is empty, when `integrity_counters` is None,
            or on any compliance-guard failure.
    """
    if not tier2_authorized:
        raise WriterNotAuthorizedError(
            "Decision Engine v1 events parquet emission requires explicit "
            "Tier 2 authorization (LOCKED in `memory/decision_engine_v1_"
            "tier2_label_table_spec.md` §12). Pass tier2_authorized=True "
            "from the runner under the LOCKED Tier 2 invocation only."
        )
    if not events_rows:
        raise WriterNotAuthorizedError(
            "events_rows is empty; refusing to write a zero-row parquet"
        )
    if integrity_counters is None:
        raise WriterNotAuthorizedError(
            "integrity_counters is mandatory under Tier 2 (impl spec §5b)"
        )

    import csv
    import pyarrow as pa
    import pyarrow.parquet as pq

    # Compliance pre-check on the column set we are about to emit.
    columns = list(events_rows[0].keys())
    assert_v1_writer_compliance(columns, output_path)

    # Materialize parquet via pyarrow.
    table = pa.Table.from_pylist(events_rows)
    pq.write_table(table, output_path)

    # Mandatory paper-link integrity CSV co-emission (impl spec §5b + §7).
    _enforce_v1_output_prefix(paper_link_integrity_csv_path)
    with open(paper_link_integrity_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(_DRY_RUN_INTEGRITY_COLUMNS)
        for line_name in ("EXTENDED", "TRIGGER_RETEST", "ALL"):
            c = integrity_counters.get(line_name)
            if c is None:
                writer.writerow([line_name, 0, 0, 0, 0, 0, 0])
                continue
            writer.writerow([
                c.line,
                c.paper_link_attempted,
                c.paper_link_matched,
                c.paper_link_unmatched,
                c.paper_link_duplicate_error,
                c.paper_link_skipped_source_missing,
                c.paper_link_validity_missing,
            ])


def write_label_table_markdown(events_frame, output_path: str = LABEL_TABLE_MD_PATH) -> None:
    """Optional markdown rendering of (setup_label, execution_label,
    market_context) cell counts. Tier 0 NOT authorized.
    """
    raise WriterNotAuthorizedError(
        "Decision Engine v1 markdown label-table emission is at Tier 0 "
        "(code-only); Tier 3+ LOCK required (spec §11)."
    )


def write_events_html(events_frame, output_path: str) -> None:
    """Optional briefing HTML. Tier 0 NOT authorized.

    Default authorization scope (parquet + integrity CSV) does NOT
    include HTML. HTML requires explicit Tier 4 opt-in.
    """
    raise WriterNotAuthorizedError(
        "Decision Engine v1 HTML emission is at Tier 0 (code-only); "
        "Tier 4 LOCK required (spec §11)."
    )


# ─── Tier 1 dry-run emitters (separate authorization surface) ────────────
#
# These four functions are gated by per-table opt-in flags from the runner
# under the LOCKED Tier 1 dry-run spec (`memory/decision_engine_v1_tier1
# _dry_run_spec.md` §16). They share `_enforce_v1_output_prefix()` for
# path discipline. They are deliberately distinct from the Tier 2+
# events-parquet / events-CSV / label-table / HTML emit paths above
# (which remain WriterNotAuthorizedError-gated until a Tier 2+ LOCK).
#
# Hard rules (LOCKED Tier 1 spec §0.2 + §16):
#   - NO forward returns / WR / PF / DD / R / Sortino / Sharpe / horizon.
#   - NO ranking / portfolio / score column.
#   - NO `final_action` column.
#   - NO parquet, NO HTML.
#   - NO promotion / live-eligible language in summary markdown.

DRY_RUN_INTEGRITY_CSV_PATH: Final[str] = (
    f"{V1_OUTPUT_PREFIX}tier1_paper_link_integrity.csv"
)
DRY_RUN_LABEL_DISTRIBUTION_CSV_PATH: Final[str] = (
    f"{V1_OUTPUT_PREFIX}tier1_label_distribution.csv"
)
DRY_RUN_UNMAPPED_CELLS_CSV_PATH: Final[str] = (
    f"{V1_OUTPUT_PREFIX}tier1_unmapped_cells.csv"
)
DRY_RUN_SUMMARY_MD_PATH: Final[str] = (
    f"{V1_OUTPUT_PREFIX}tier1_dry_run_summary.md"
)

_DRY_RUN_INTEGRITY_COLUMNS: Final[tuple[str, ...]] = (
    "line",
    "paper_link_attempted",
    "paper_link_matched",
    "paper_link_unmatched",
    "paper_link_duplicate_error",
    "paper_link_skipped_source_missing",
    "paper_link_validity_missing",
)

_DRY_RUN_LABEL_DISTRIBUTION_COLUMNS: Final[tuple[str, ...]] = (
    "setup_label",
    "execution_label",
    "market_context",
    "count",
)

_DRY_RUN_UNMAPPED_CELL_COLUMNS: Final[tuple[str, ...]] = (
    "source",
    "family",
    "state",
    "timeframe",
    "row_count",
)


def write_dry_run_integrity_csv(
    counters_per_line,
    output_path: str = DRY_RUN_INTEGRITY_CSV_PATH,
) -> None:
    """Emit the Tier 1 paper-link integrity CSV.

    Args:
        counters_per_line: dict[str, PaperLinkIntegrityCounters] keyed by
            line ∈ {"EXTENDED", "TRIGGER_RETEST", "ALL"}.
        output_path: defaults to LOCKED Tier 1 path.

    No metrics. Just counter values per Step 5b.
    """
    import csv

    _enforce_v1_output_prefix(output_path)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(_DRY_RUN_INTEGRITY_COLUMNS)
        for line_name in ("EXTENDED", "TRIGGER_RETEST", "ALL"):
            c = counters_per_line.get(line_name)
            if c is None:
                writer.writerow([line_name, 0, 0, 0, 0, 0, 0])
                continue
            writer.writerow([
                c.line,
                c.paper_link_attempted,
                c.paper_link_matched,
                c.paper_link_unmatched,
                c.paper_link_duplicate_error,
                c.paper_link_skipped_source_missing,
                c.paper_link_validity_missing,
            ])


def write_dry_run_label_distribution_csv(
    label_counts,
    output_path: str = DRY_RUN_LABEL_DISTRIBUTION_CSV_PATH,
) -> None:
    """Emit the Tier 1 label-distribution CSV (descriptive census).

    Args:
        label_counts: iterable of ((setup_label, execution_label,
            market_context), count) tuples.
        output_path: defaults to LOCKED Tier 1 path.

    NO forward returns. NO win-rate. NO PF. NO horizon-conditional
    metric. Pure descriptive count.
    """
    import csv

    _enforce_v1_output_prefix(output_path)
    rows = sorted(label_counts, key=lambda kv: (kv[0][0], kv[0][1], kv[0][2]))
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(_DRY_RUN_LABEL_DISTRIBUTION_COLUMNS)
        for (setup, execution, ctx), count in rows:
            writer.writerow([setup, execution, ctx, count])


def write_dry_run_unmapped_cells_csv(
    unmapped_cells,
    output_path: str = DRY_RUN_UNMAPPED_CELLS_CSV_PATH,
) -> None:
    """Emit the Tier 1 unmapped-cell CSV (forensic, FAIL-only per LOCK).

    Args:
        unmapped_cells: iterable of ((source, family, state, timeframe),
            row_count) tuples for cells that appeared in the live scanner
            output but were NOT in the LOCKED MAPPING_TABLE.
        output_path: defaults to LOCKED Tier 1 path.
    """
    import csv

    _enforce_v1_output_prefix(output_path)
    rows = sorted(unmapped_cells, key=lambda kv: kv[0])
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(_DRY_RUN_UNMAPPED_CELL_COLUMNS)
        for (source, family, state, tf), count in rows:
            writer.writerow([source, family, state, tf, count])


def write_dry_run_summary_markdown(
    summary,
    output_path: str = DRY_RUN_SUMMARY_MD_PATH,
) -> None:
    """Emit the Tier 1 dry-run summary markdown.

    Args:
        summary: dict with keys:
            run_timestamp_utc (str), asof_date (str),
            panel_max_date (str), line_e_max_date (str|None),
            line_tr_max_date (str|None), inputs (list[str]),
            step_1_0_status (str), step_5b_status (str),
            verdict (str — "PASS" or "FAIL"),
            fail_category (str|None — only when verdict=="FAIL"),
            fail_reason (str|None),
            event_count (int),
            execution_label_distribution (dict[str, int]),
            paper_link_counters (dict[str, PaperLinkIntegrityCounters]),
            paper_validity_missing_total (int),
            unmapped_cell_count (int),
        output_path: defaults to LOCKED Tier 1 path.

    NO metrics. NO forward returns. NO promotion language.
    """
    _enforce_v1_output_prefix(output_path)
    lines: list[str] = []
    lines.append("# Decision Engine v1 — Tier 1 dry-run summary")
    lines.append("")
    lines.append(f"- Run timestamp (UTC): {summary['run_timestamp_utc']}")
    lines.append(f"- asof_date (resolved): {summary['asof_date']}")
    lines.append(f"- panel max date: {summary['panel_max_date']}")
    lines.append(f"- Line E max asof_date: {summary.get('line_e_max_date') or 'N/A'}")
    lines.append(f"- Line TR max asof_date: {summary.get('line_tr_max_date') or 'N/A'}")
    lines.append("")
    lines.append("## Inputs (read-only)")
    for path in summary["inputs"]:
        lines.append(f"- `{path}`")
    lines.append("")
    lines.append("## Verdict")
    lines.append(f"- step 1.0 live-scope: {summary['step_1_0_status']}")
    lines.append(f"- step 5b paper-link integrity: {summary['step_5b_status']}")
    lines.append(f"- **verdict: {summary['verdict']}**")
    if summary["verdict"] != "PASS":
        if summary.get("fail_category"):
            lines.append(f"- fail_category: {summary['fail_category']}")
        if summary.get("fail_reason"):
            lines.append(f"- fail_reason: {summary['fail_reason']}")
    lines.append("")
    lines.append("## Event count")
    lines.append(f"- total events processed: {summary['event_count']}")
    lines.append(f"- unmapped cells observed: {summary['unmapped_cell_count']}")
    lines.append("")
    lines.append("## Execution label distribution (descriptive)")
    dist = summary["execution_label_distribution"]
    if dist:
        for label, count in sorted(dist.items()):
            lines.append(f"- {label}: {count}")
    else:
        lines.append("- (no events)")
    lines.append("")
    lines.append("## Paper-link integrity (Step 5b)")
    counters = summary["paper_link_counters"]
    for line_name in ("EXTENDED", "TRIGGER_RETEST", "ALL"):
        c = counters.get(line_name)
        if c is None:
            continue
        lines.append(
            f"- {line_name}: attempted={c.paper_link_attempted}, "
            f"matched={c.paper_link_matched}, "
            f"unmatched={c.paper_link_unmatched}, "
            f"duplicate_error={c.paper_link_duplicate_error}, "
            f"skipped_source_missing={c.paper_link_skipped_source_missing}, "
            f"validity_missing={c.paper_link_validity_missing}"
        )
    lines.append("")
    lines.append(
        f"- paper_validity_metadata_missing total: "
        f"{summary['paper_validity_missing_total']}"
    )
    lines.append("")
    lines.append(
        "_Tier 1 dry-run; no parquet, no HTML, no metrics, no ranking, "
        "no portfolio, no live integration._"
    )
    lines.append("")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ─── Tier 2 label-table emitters (LOCKED 2026-05-04) ─────────────────────
#
# Spec: `memory/decision_engine_v1_tier2_label_table_spec.md` §12.
# Two artefacts under this surface are authorized as opt-in companions
# to the Tier 2 events parquet:
#   - `output/decision_engine_v1_tier2_label_distribution.csv`
#   - `output/decision_engine_v1_tier2_label_table_summary.md`
# Both share `_enforce_v1_output_prefix()` for path discipline. Forbidden
# under Tier 2 LOCK:
#   - events CSV mirror
#   - label-table markdown count rollup (Tier 3)
#   - HTML (Tier 4)
#   - any forward-return / metric / ranking / portfolio output


def write_tier2_label_distribution_csv(
    label_counts,
    output_path: str = TIER2_LABEL_DISTRIBUTION_CSV_PATH,
) -> None:
    """Emit Tier 2 label-distribution CSV per LOCK §12.3.

    Args:
        label_counts: iterable of ((setup_label, execution_label,
            market_context), count) tuples.
        output_path: defaults to LOCKED Tier 2 path.

    NO forward returns. NO win-rate. NO PF. Pure descriptive count
    (same schema as Tier 1 dry-run distribution; new path).
    """
    import csv

    _enforce_v1_output_prefix(output_path)
    rows = sorted(label_counts, key=lambda kv: (kv[0][0], kv[0][1], kv[0][2]))
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(_DRY_RUN_LABEL_DISTRIBUTION_COLUMNS)
        for (setup, execution, ctx), count in rows:
            writer.writerow([setup, execution, ctx, count])


def write_tier2_label_table_summary_markdown(
    summary,
    output_path: str = TIER2_LABEL_TABLE_SUMMARY_MD_PATH,
) -> None:
    """Emit Tier 2 label-table run summary markdown per LOCK §12.3.

    Args:
        summary: dict with keys (mirroring the Tier 1 dry-run summary plus
            §12.4 sha256 pre/post + §12.3 output paths):
                run_timestamp_utc, asof_date, panel_max_date,
                line_e_max_date, line_tr_max_date, inputs,
                step_1_0_status, step_5b_status, verdict,
                fail_category, fail_reason,
                event_count, execution_label_distribution,
                paper_link_counters, paper_validity_missing_total,
                paper_pre, paper_post, output_paths.
        output_path: defaults to LOCKED Tier 2 path.

    NO metrics. NO forward returns. NO promotion language.
    """
    _enforce_v1_output_prefix(output_path)
    lines: list[str] = []
    lines.append("# Decision Engine v1 — Tier 2 label table run summary")
    lines.append("")
    lines.append(f"- Run timestamp (UTC): {summary['run_timestamp_utc']}")
    lines.append(f"- asof_date (resolved): {summary['asof_date']}")
    lines.append(f"- panel max date: {summary['panel_max_date']}")
    lines.append(f"- Line E max asof_date: {summary.get('line_e_max_date') or 'N/A'}")
    lines.append(f"- Line TR max asof_date: {summary.get('line_tr_max_date') or 'N/A'}")
    lines.append("")
    lines.append("## Inputs (read-only)")
    for path in summary["inputs"]:
        lines.append(f"- `{path}`")
    lines.append("")
    lines.append("## Verdict")
    lines.append(f"- step 1.0 live-scope: {summary['step_1_0_status']}")
    lines.append(f"- step 5b paper-link integrity: {summary['step_5b_status']}")
    lines.append(f"- **verdict: {summary['verdict']}**")
    if summary["verdict"] != "PASS":
        if summary.get("fail_category"):
            lines.append(f"- fail_category: {summary['fail_category']}")
        if summary.get("fail_reason"):
            lines.append(f"- fail_reason: {summary['fail_reason']}")
    lines.append("")
    lines.append("## Event count")
    lines.append(f"- total events processed: {summary['event_count']}")
    lines.append("")
    lines.append("## Execution label distribution (descriptive)")
    dist = summary["execution_label_distribution"]
    if dist:
        for label, count in sorted(dist.items()):
            lines.append(f"- {label}: {count}")
    else:
        lines.append("- (no events)")
    lines.append("")
    lines.append("## Paper-link integrity (Step 5b)")
    counters = summary["paper_link_counters"]
    for line_name in ("EXTENDED", "TRIGGER_RETEST", "ALL"):
        c = counters.get(line_name)
        if c is None:
            continue
        lines.append(
            f"- {line_name}: attempted={c.paper_link_attempted}, "
            f"matched={c.paper_link_matched}, "
            f"unmatched={c.paper_link_unmatched}, "
            f"duplicate_error={c.paper_link_duplicate_error}, "
            f"skipped_source_missing={c.paper_link_skipped_source_missing}, "
            f"validity_missing={c.paper_link_validity_missing}"
        )
    lines.append("")
    lines.append(
        f"- paper_validity_metadata_missing total: "
        f"{summary['paper_validity_missing_total']}"
    )
    lines.append("")
    lines.append("## Paper-stream parquet sha256 pre/post (LOCK §12.4)")
    pre = summary.get("paper_pre", {})
    post = summary.get("paper_post", {})
    for line_key, label in (("line_e", "Line E"), ("line_tr", "Line TR")):
        pre_state = pre.get(line_key, {})
        post_state = post.get(line_key, {})
        lines.append(
            f"- {label} pre:  size={pre_state.get('size')}, "
            f"sha256={pre_state.get('sha256')}"
        )
        lines.append(
            f"- {label} post: size={post_state.get('size')}, "
            f"sha256={post_state.get('sha256')}"
        )
    lines.append("")
    lines.append("## Authorized outputs (LOCK §12.3)")
    for path in summary.get("output_paths", []):
        lines.append(f"- `{path}`")
    lines.append("")
    lines.append(
        "_Tier 2 label-table run; no forward returns, no PF/WR/meanR, "
        "no ranking, no portfolio, no live integration, no HTML, "
        "no events CSV mirror, no markdown rollup._"
    )
    lines.append("")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ─── Tier 3 review report emitter (LOCKED 2026-05-04) ────────────────────


def write_tier3_review_report_markdown(
    report_text: str,
    output_path: str = TIER3_REVIEW_REPORT_MD_PATH,
) -> None:
    """Emit Tier 3 review report markdown atomically (LOCK §12.2 Q7).

    The caller is responsible for full markdown composition. This writer is
    a pure commit step: enforce v1 output prefix, write to a sibling temp
    file, then `os.replace()` to the final path. If any failure occurs
    before the rename, the final path is untouched (per Q7 atomic-write).

    Args:
        report_text: fully-rendered markdown body (UTF-8 string).
        output_path: defaults to LOCKED Tier 3 path.
    """
    import os
    import tempfile

    _enforce_v1_output_prefix(output_path)
    target_dir = os.path.dirname(output_path) or "."
    fd, tmp_path = tempfile.mkstemp(
        prefix=".tier3_review_report.",
        suffix=".md.tmp",
        dir=target_dir,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(report_text)
        os.replace(tmp_path, output_path)
    except Exception:
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        raise


__all__ = [
    "V1_OUTPUT_PREFIX",
    "EVENTS_PARQUET_PATH",
    "EVENTS_CSV_PATH",
    "PAPER_LINK_INTEGRITY_CSV_PATH",
    "LABEL_TABLE_MD_PATH",
    "TIER2_LABEL_DISTRIBUTION_CSV_PATH",
    "TIER2_LABEL_TABLE_SUMMARY_MD_PATH",
    "TIER3_REVIEW_REPORT_MD_PATH",
    "DRY_RUN_INTEGRITY_CSV_PATH",
    "DRY_RUN_LABEL_DISTRIBUTION_CSV_PATH",
    "DRY_RUN_UNMAPPED_CELLS_CSV_PATH",
    "DRY_RUN_SUMMARY_MD_PATH",
    "WriterNotAuthorizedError",
    "assert_v1_writer_compliance",
    "write_events_parquet",
    "write_label_table_markdown",
    "write_events_html",
    "write_dry_run_integrity_csv",
    "write_dry_run_label_distribution_csv",
    "write_dry_run_unmapped_cells_csv",
    "write_dry_run_summary_markdown",
    "write_tier2_label_distribution_csv",
    "write_tier2_label_table_summary_markdown",
    "write_tier3_review_report_markdown",
]
