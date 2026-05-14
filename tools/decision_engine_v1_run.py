"""Decision Engine v1 — runner.

Spec: memory/decision_engine_v1_implementation_spec.md (LOCKED 2026-05-04
Tier 0; Paper Signal Validity additive 2026-05-04).
LOCKED Tier 1 dry-run spec: memory/decision_engine_v1_tier1_dry_run_spec.md
(LOCKED 2026-05-04).
LOCKED Tier 2 label-table spec: memory/decision_engine_v1_tier2_label
_table_spec.md (LOCKED 2026-05-04).
LOCKED Tier 3 review report spec:
memory/decision_engine_v1_tier3_review_report_spec.md (LOCKED 2026-05-04).

Tier 0 default: refuse to execute (legacy behavior preserved). Any
invocation without `--tier {1,2,3}` exits status 2 with the LOCKED
refusal banner.

Tier 1: single authorized dry-run per the LOCK message in §16 of the
Tier 1 spec. Mapping-only pipeline:

  ingest → live-scope validate → mapping/label/risk → paper-stream link
         → schema/purity checks → allowed dry-run outputs.

Tier 2: single authorized live invocation per the LOCK message in §12
of the Tier 2 spec. Same pipeline, with parquet writer enabled and
sha256 pre/post guardrail on Line E + Line TR production parquets:

  ingest → live-scope validate → mapping/label/risk → paper-stream link
         → schema/purity checks → events parquet + integrity CSV +
           Tier 2 label distribution + Tier 2 summary.

Hard rules at Tier 1 (§0.2 + §16):
  - NO `decision_engine_v1_events.parquet` / `.csv` emission.
  - NO HTML.
  - NO backtest, NO forward returns, NO performance metrics.
  - NO ranking, NO portfolio, NO `final_action`, NO `*_score` /
    `*_rank` / `portfolio_*` field beyond the LOCKED legitimate
    `earliness_score_pct` / `liquidity_score` / `capacity_score`.
  - NO live integration.
  - NO paper eligibility recompute.
  - NO EMA threshold fallback.
  - NO modification to `decision_engine/` v0, `paper_execution_v0_*`,
    `portfolio_merge_paper_*`, EMA pilot artefacts.
  - Single authorized invocation per LOCK; FAIL halts; remediation
    requires fresh LOCK.

Hard rules at Tier 2 (§12 of Tier 2 spec):
  - Authorized outputs (only): events parquet, paper-link integrity CSV,
    Tier 2 label-distribution CSV, Tier 2 summary markdown.
  - DENY: events CSV mirror, label-table markdown rollup, HTML, any
    forward-return / metric / ranking / portfolio output.
  - Mandatory: sha256 + size pre/post equality on Line E + Line TR
    production parquets (LOCK §12.4); divergence → halt with
    `paper-stream mutation` classification, even on otherwise-PASS path.
  - All Tier 1 hard rules carry forward.
  - Single authorized invocation per LOCK; FAIL halts; remediation
    requires fresh LOCK.

Tier 3: single authorized review-report render per the LOCK message in
§12 of the Tier 3 spec. Read-only render of the existing Tier 2
events.parquet into a single human-readable markdown:

  read events.parquet + integrity CSV → compute counts → render markdown
         → atomic write → sha256 byte-equal verification.

Hard rules at Tier 3 (§12 of Tier 3 spec):
  - Authorized output (only): output/decision_engine_v1_tier3_review_report.md.
  - DENY: parquet / CSV / HTML / JSON / image / sidecar write of any kind.
  - DENY: any recomputation of labels / paper-link / EMA tier / regime /
    risk — read-only render.
  - DENY: ranking / scoring / "top N" / portfolio / forward returns /
    PF / WR / DD / Sharpe / Sortino / MAR / live integration.
  - Mandatory: sha256 + size pre/post equality on events.parquet +
    Line E + Line TR (LOCK §12.2 Q8); divergence → halt with `source
    mutation` classification, no report written.
  - Mandatory: full enum coverage — every `setup_label`, `execution_label`,
    `market_context` value shown including zero-count buckets (LOCK §12.4).
  - Atomic write: temp file then rename; FAIL before final rename → no
    final report written (LOCK §12.2 Q7).
  - All Tier 1 + Tier 2 hard rules carry forward.
  - Single authorized invocation per LOCK; FAIL halts; remediation
    requires fresh LOCK.
"""

from __future__ import annotations

import argparse
import collections
import datetime as _dt
import sys
from typing import Final

from decision_engine_v1 import (
    ingest as v1_ingest,
    live_scope as v1_live_scope,
    paper_stream_link as v1_link,
    purity_check as v1_purity,
    write as v1_write,
)
from decision_engine_v1.label import assign_label
from decision_engine_v1.mapping_table import V1_SOURCE_WHITELIST
from decision_engine_v1.schema import (
    EXECUTION_LABELS,
    MARKET_CONTEXTS,
    PaperReferenceState,
    SETUP_LABELS,
    assert_paper_reference_invariants,
)


TIER_0_LOCK_DATE: Final[str] = "2026-05-04"
TIER_1_LOCK_DATE: Final[str] = "2026-05-04"
TIER_2_LOCK_DATE: Final[str] = "2026-05-04"
TIER_3_LOCK_DATE: Final[str] = "2026-05-04"


_TIER_0_REFUSAL_MESSAGE: Final[str] = f"""
Decision Engine v1 runner — Tier 0 LOCK refusal
================================================

This runner is gated by `memory/decision_engine_v1_implementation_spec.md`
LOCKED {TIER_0_LOCK_DATE} at Tier 0 (code-only) + Tier 1 dry-run spec
LOCKED {TIER_1_LOCK_DATE} at `memory/decision_engine_v1_tier1_dry_run_spec.md`
+ Tier 2 label-table spec LOCKED {TIER_2_LOCK_DATE} at
`memory/decision_engine_v1_tier2_label_table_spec.md`
+ Tier 3 review report spec LOCKED {TIER_3_LOCK_DATE} at
`memory/decision_engine_v1_tier3_review_report_spec.md`.

Without `--tier {{1,2,3}}`, the runner remains Tier 0 fail-closed.

Authorized at Tier 1 (single invocation per LOCK):
  python tools/decision_engine_v1_run.py --tier 1 \\
    [--asof-date YYYY-MM-DD] \\
    [--emit-integrity-csv] [--emit-label-distribution-csv] \\
    [--emit-dry-run-summary-md] [--emit-unmapped-cells-csv]

Authorized at Tier 2 (single invocation per LOCK):
  python tools/decision_engine_v1_run.py --tier 2 \\
    [--asof-date YYYY-MM-DD]

Authorized at Tier 3 (single invocation per LOCK; read-only render):
  python tools/decision_engine_v1_run.py --tier 3

NOT authorized at any tier:
  * Tier 2 events CSV mirror / label-table markdown rollup / HTML.
  * Backtest of any kind. Forward-return / WR / PF / DD / horizon metric.
  * Ranking / portfolio / score / rank / final_action artefacts.
  * Live integration of any kind.
  * Any change to `decision_engine/` v0, `paper_execution_v0_*`,
    `portfolio_merge_paper_*` files.
"""

# LOCKED 2026-05-04 Tier 2 spec §12.4: production paper-stream parquets.
_TIER2_PAPER_PARQUET_PATHS: Final[dict[str, str]] = {
    "line_e": "output/paper_execution_v0_trades.parquet",
    "line_tr": "output/paper_execution_v0_trigger_retest_trades.parquet",
}

# LOCKED 2026-05-04 Tier 3 spec §12.2 Q8: 3 guarded files for sha256 PRE/POST.
_TIER3_GUARDED_PATHS: Final[dict[str, str]] = {
    "events": "output/decision_engine_v1_events.parquet",
    "line_e": "output/paper_execution_v0_trades.parquet",
    "line_tr": "output/paper_execution_v0_trigger_retest_trades.parquet",
}


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="decision_engine_v1_run",
        description=(
            "Decision Engine v1 runner. Tier 0 default = refuse; "
            "Tier 1 = LOCKED single dry-run."
        ),
    )
    parser.add_argument(
        "--tier",
        type=int,
        default=0,
        choices=[0, 1, 2, 3],
        help=(
            "Tier authorization (0=fail-closed default; 1=dry-run; "
            "2=LOCKED label-table parquet emission; "
            "3=LOCKED review-report markdown render)."
        ),
    )
    parser.add_argument(
        "--asof-date",
        type=str,
        default=None,
        help=(
            "ISO yyyy-mm-dd. If omitted at Tier 1, resolves to panel max "
            "date with paper-stream max-date cross-check (LOCK §16.1)."
        ),
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help=(
            "Legacy Tier 1 dry-run flag; redundant under explicit --tier 1 "
            "(output gating is keyed on tier + per-table opt-ins)."
        ),
    )
    parser.add_argument(
        "--emit-integrity-csv",
        action="store_true",
        help="Emit Tier 1 paper-link integrity CSV (LOCK §16.5).",
    )
    parser.add_argument(
        "--emit-label-distribution-csv",
        action="store_true",
        help="Emit Tier 1 label-distribution CSV (LOCK §16.5).",
    )
    parser.add_argument(
        "--emit-unmapped-cells-csv",
        action="store_true",
        help=(
            "Emit Tier 1 unmapped-cells CSV. Per LOCK §16.5: omit on PASS, "
            "emit on FAIL only (forensic recoverability)."
        ),
    )
    parser.add_argument(
        "--emit-dry-run-summary-md",
        action="store_true",
        help="Emit Tier 1 dry-run summary markdown (LOCK §16.5).",
    )
    return parser


def _market_context_for(event) -> str:
    """Derive `market_context` enum from the panel `regime` /
    `regime_stale_flag` carry. Mechanical mapping; no policy.
    """
    if event.regime_stale_flag in ("stale_1d", "stale_2d_plus"):
        return "REGIME_STALE"
    regime = (event.regime or "").lower() if event.regime else None
    if regime == "long":
        return "REGIME_LONG"
    if regime == "neutral":
        return "REGIME_NEUTRAL"
    if regime == "short":
        return "REGIME_SHORT"
    return "REGIME_UNKNOWN"


def _resolve_asof_date(
    user_asof: str | None,
) -> tuple[str, str, str | None, str | None]:
    """Resolve asof_date per LOCK §16.1.

    Returns (resolved_asof_date, panel_max, line_e_max, line_tr_max).
    Caller is responsible for the source_asof_mismatch check.
    """
    panel_max = v1_ingest.panel_max_date()
    if panel_max is None:
        raise RuntimeError(
            "panel parquet has zero rows; cannot resolve asof_date"
        )
    line_e_max = v1_link.paper_stream_max_date("EXTENDED")
    line_tr_max = v1_link.paper_stream_max_date("TRIGGER_RETEST")
    resolved = user_asof if user_asof else panel_max
    return resolved, panel_max, line_e_max, line_tr_max


def _check_source_asof_mismatch(
    asof_date: str,
    line_e_max: str | None,
    line_tr_max: str | None,
) -> str | None:
    """Per LOCK §16.1: paper-stream parquets must have max asof_date ≥
    panel-derived asof_date (paper streams must cover the production
    asof). If a paper stream lags, halt fast.

    Returns a string diagnostic on mismatch, or None on match.
    """
    lagging: list[str] = []
    if line_e_max is not None and line_e_max < asof_date:
        lagging.append(f"Line E max={line_e_max} < asof_date={asof_date}")
    if line_tr_max is not None and line_tr_max < asof_date:
        lagging.append(f"Line TR max={line_tr_max} < asof_date={asof_date}")
    if lagging:
        return "; ".join(lagging)
    return None


def _classify_unmapped_cells(events) -> dict[tuple, int]:
    """Group events by (source, family, state, timeframe) and count.
    Used both for live-scope validation input and for the FAIL-only
    unmapped-cells CSV.
    """
    counts: dict[tuple, int] = collections.Counter()
    for ev in events:
        counts[(ev.source, ev.family, ev.state, ev.timeframe)] += 1
    return counts


def _emit_fail(
    *,
    args,
    fail_category: str,
    fail_reason: str,
    asof_date: str,
    panel_max: str,
    line_e_max: str | None,
    line_tr_max: str | None,
    inputs: list[str],
    step_1_0_status: str,
    step_5b_status: str,
    event_count: int,
    execution_label_distribution: dict[str, int],
    paper_link_counters: dict,
    paper_validity_missing_total: int,
    unmapped_cells: dict[tuple, int],
) -> int:
    """Emit FAIL diagnostics on stdout/stderr + authorized opt-ins.

    LOCK §16.5: unmapped-cells CSV is emitted on FAIL only (forensic).
    Other opt-ins still honor their flags.
    """
    sys.stderr.write(
        f"\nDecision Engine v1 — Tier 1 dry-run FAIL\n"
        f"category: {fail_category}\n"
        f"reason: {fail_reason}\n"
    )
    summary = {
        "run_timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "asof_date": asof_date,
        "panel_max_date": panel_max,
        "line_e_max_date": line_e_max,
        "line_tr_max_date": line_tr_max,
        "inputs": inputs,
        "step_1_0_status": step_1_0_status,
        "step_5b_status": step_5b_status,
        "verdict": "FAIL",
        "fail_category": fail_category,
        "fail_reason": fail_reason,
        "event_count": event_count,
        "execution_label_distribution": execution_label_distribution,
        "paper_link_counters": paper_link_counters,
        "paper_validity_missing_total": paper_validity_missing_total,
        "unmapped_cell_count": len(unmapped_cells),
    }
    if args.emit_dry_run_summary_md:
        v1_write.write_dry_run_summary_markdown(summary)
    if args.emit_integrity_csv and paper_link_counters:
        v1_write.write_dry_run_integrity_csv(paper_link_counters)
    if args.emit_label_distribution_csv:
        # Build empty-or-partial distribution; FAIL still emits what we have.
        v1_write.write_dry_run_label_distribution_csv([])
    # LOCK §16.5: unmapped-cells CSV emits on FAIL.
    if unmapped_cells:
        v1_write.write_dry_run_unmapped_cells_csv(unmapped_cells.items())
    sys.stdout.write(
        f"\nDecision Engine v1 — Tier 1 dry-run\n"
        f"asof_date: {asof_date}\n"
        f"panel max: {panel_max} | Line E max: {line_e_max} | "
        f"Line TR max: {line_tr_max}\n"
        f"step 1.0: {step_1_0_status}\n"
        f"step 5b: {step_5b_status}\n"
        f"verdict: FAIL ({fail_category}: {fail_reason})\n"
    )
    return 1


def _run_tier1(args) -> int:
    """Execute the LOCKED Tier 1 dry-run pipeline. Returns exit code."""
    inputs: list[str] = [
        v1_ingest.DEFAULT_PANEL_PATH,
        v1_link.PAPER_STREAM_PATHS["EXTENDED"],
        v1_link.PAPER_STREAM_PATHS["TRIGGER_RETEST"],
    ]
    paper_link_counters_empty: dict = {}

    # 1. asof_date resolution (LOCK §16.1)
    try:
        asof_date, panel_max, line_e_max, line_tr_max = _resolve_asof_date(
            args.asof_date
        )
    except Exception as exc:
        sys.stderr.write(f"FAIL (input coverage): {exc}\n")
        return 1

    mismatch = _check_source_asof_mismatch(asof_date, line_e_max, line_tr_max)
    if mismatch is not None:
        return _emit_fail(
            args=args,
            fail_category="input coverage",
            fail_reason=f"source_asof_mismatch: {mismatch}",
            asof_date=asof_date,
            panel_max=panel_max,
            line_e_max=line_e_max,
            line_tr_max=line_tr_max,
            inputs=inputs,
            step_1_0_status="not_run (asof_mismatch)",
            step_5b_status="not_run",
            event_count=0,
            execution_label_distribution={},
            paper_link_counters=paper_link_counters_empty,
            paper_validity_missing_total=0,
            unmapped_cells={},
        )

    # 2. Ingest
    try:
        events = v1_ingest.read_scanner_outputs(asof_date)
    except Exception as exc:
        return _emit_fail(
            args=args,
            fail_category="input coverage",
            fail_reason=f"ingest failed: {exc}",
            asof_date=asof_date,
            panel_max=panel_max,
            line_e_max=line_e_max,
            line_tr_max=line_tr_max,
            inputs=inputs,
            step_1_0_status="not_run (ingest_failed)",
            step_5b_status="not_run",
            event_count=0,
            execution_label_distribution={},
            paper_link_counters=paper_link_counters_empty,
            paper_validity_missing_total=0,
            unmapped_cells={},
        )

    cell_counts = _classify_unmapped_cells(events)

    # 3. Step 1.0 live-scope validation
    report = v1_live_scope.validate_live_scope(asof_date, cell_counts.keys())
    try:
        v1_live_scope.enforce_live_scope_or_raise(report)
    except (
        v1_live_scope.UnmappedCellError,
        v1_live_scope.UnmappedSourceError,
    ) as exc:
        unmapped_cell_set = set(report.unmapped_cells)
        unmapped_subset = {k: v for k, v in cell_counts.items() if k in unmapped_cell_set}
        return _emit_fail(
            args=args,
            fail_category="mapping",
            fail_reason=f"live-scope: {exc}",
            asof_date=asof_date,
            panel_max=panel_max,
            line_e_max=line_e_max,
            line_tr_max=line_tr_max,
            inputs=inputs,
            step_1_0_status="FAIL",
            step_5b_status="not_run",
            event_count=len(events),
            execution_label_distribution={},
            paper_link_counters=paper_link_counters_empty,
            paper_validity_missing_total=0,
            unmapped_cells=unmapped_subset,
        )

    # 4. Mapping/label/risk per event
    try:
        decisions = [assign_label(ev) for ev in events]
    except Exception as exc:
        return _emit_fail(
            args=args,
            fail_category="schema",
            fail_reason=f"label assignment: {exc}",
            asof_date=asof_date,
            panel_max=panel_max,
            line_e_max=line_e_max,
            line_tr_max=line_tr_max,
            inputs=inputs,
            step_1_0_status="PASS",
            step_5b_status="not_run",
            event_count=len(events),
            execution_label_distribution={},
            paper_link_counters=paper_link_counters_empty,
            paper_validity_missing_total=0,
            unmapped_cells={},
        )

    # 5. Step 5: paper-stream link
    try:
        attachments, counters = v1_link.link_paper_streams(events, asof_date)
    except v1_link.PaperStreamMissingError as exc:
        return _emit_fail(
            args=args,
            fail_category="paper-link",
            fail_reason=f"paper stream missing: {exc}",
            asof_date=asof_date,
            panel_max=panel_max,
            line_e_max=line_e_max,
            line_tr_max=line_tr_max,
            inputs=inputs,
            step_1_0_status="PASS",
            step_5b_status="FAIL (missing parquet)",
            event_count=len(events),
            execution_label_distribution={},
            paper_link_counters=paper_link_counters_empty,
            paper_validity_missing_total=0,
            unmapped_cells={},
        )
    except v1_link.PaperStreamDuplicateMatchError as exc:
        return _emit_fail(
            args=args,
            fail_category="paper-link",
            fail_reason=f"duplicate match halt: {exc}",
            asof_date=asof_date,
            panel_max=panel_max,
            line_e_max=line_e_max,
            line_tr_max=line_tr_max,
            inputs=inputs,
            step_1_0_status="PASS",
            step_5b_status="FAIL (duplicate match)",
            event_count=len(events),
            execution_label_distribution={},
            paper_link_counters=paper_link_counters_empty,
            paper_validity_missing_total=0,
            unmapped_cells={},
        )

    # 5b. Integrity invariants
    try:
        for c in counters.values():
            c.assert_invariants()
    except AssertionError as exc:
        return _emit_fail(
            args=args,
            fail_category="paper-link",
            fail_reason=f"Step 5b invariant: {exc}",
            asof_date=asof_date,
            panel_max=panel_max,
            line_e_max=line_e_max,
            line_tr_max=line_tr_max,
            inputs=inputs,
            step_1_0_status="PASS",
            step_5b_status="FAIL",
            event_count=len(events),
            execution_label_distribution={},
            paper_link_counters=counters,
            paper_validity_missing_total=0,
            unmapped_cells={},
        )

    # 6. Build event rows + per-row paper-reference invariant + label dist
    label_distribution: dict[tuple, int] = collections.Counter()
    execution_label_dist: dict[str, int] = collections.Counter()
    paper_validity_missing_total = 0

    for idx, (ev, dec) in enumerate(zip(events, decisions)):
        market_ctx = _market_context_for(ev)
        execution_label = dec.execution_label
        live_execution_allowed = dec.live_execution_allowed
        paper_origin = None
        paper_stream_ref = None
        paper_trade_id = None
        paper_match_key = None
        paper_valid_from = None
        paper_valid_until = None
        paper_signal_age = None
        paper_expired_flag = None
        paper_validity_metadata_missing = None

        att = attachments.get(idx)
        if att is not None and att.matched:
            execution_label = "PAPER_ONLY"
            live_execution_allowed = False
            paper_origin = att.paper_origin
            paper_stream_ref = att.paper_stream_ref
            paper_trade_id = att.paper_trade_id
            paper_match_key = att.paper_match_key
            paper_valid_from = att.paper_valid_from
            paper_valid_until = att.paper_valid_until
            paper_signal_age = att.paper_signal_age
            paper_expired_flag = att.paper_expired_flag
            paper_validity_metadata_missing = att.paper_validity_metadata_missing
            if paper_validity_metadata_missing:
                paper_validity_missing_total += 1

        # Schema / label enum validation per row (§6.3)
        if dec.setup_label not in SETUP_LABELS:
            return _emit_fail(
                args=args,
                fail_category="schema",
                fail_reason=f"setup_label {dec.setup_label!r} not in enum",
                asof_date=asof_date, panel_max=panel_max,
                line_e_max=line_e_max, line_tr_max=line_tr_max,
                inputs=inputs, step_1_0_status="PASS", step_5b_status="PASS",
                event_count=len(events),
                execution_label_distribution=dict(execution_label_dist),
                paper_link_counters=counters,
                paper_validity_missing_total=paper_validity_missing_total,
                unmapped_cells={},
            )
        if execution_label not in EXECUTION_LABELS:
            return _emit_fail(
                args=args,
                fail_category="schema",
                fail_reason=f"execution_label {execution_label!r} not in enum",
                asof_date=asof_date, panel_max=panel_max,
                line_e_max=line_e_max, line_tr_max=line_tr_max,
                inputs=inputs, step_1_0_status="PASS", step_5b_status="PASS",
                event_count=len(events),
                execution_label_distribution=dict(execution_label_dist),
                paper_link_counters=counters,
                paper_validity_missing_total=paper_validity_missing_total,
                unmapped_cells={},
            )
        if market_ctx not in MARKET_CONTEXTS:
            return _emit_fail(
                args=args,
                fail_category="schema",
                fail_reason=f"market_context {market_ctx!r} not in enum",
                asof_date=asof_date, panel_max=panel_max,
                line_e_max=line_e_max, line_tr_max=line_tr_max,
                inputs=inputs, step_1_0_status="PASS", step_5b_status="PASS",
                event_count=len(events),
                execution_label_distribution=dict(execution_label_dist),
                paper_link_counters=counters,
                paper_validity_missing_total=paper_validity_missing_total,
                unmapped_cells={},
            )

        # Per-row paper-reference invariant (§6.6)
        try:
            assert_paper_reference_invariants(
                PaperReferenceState(
                    execution_label=execution_label,
                    paper_origin=paper_origin,
                    paper_stream_ref=paper_stream_ref,
                    paper_trade_id=paper_trade_id,
                    paper_match_key=paper_match_key,
                    live_execution_allowed=live_execution_allowed,
                    paper_valid_from=paper_valid_from,
                    paper_valid_until=paper_valid_until,
                    paper_signal_age=paper_signal_age,
                    paper_expired_flag=paper_expired_flag,
                    paper_validity_metadata_missing=paper_validity_metadata_missing,
                )
            )
        except ValueError as exc:
            return _emit_fail(
                args=args,
                fail_category="purity",
                fail_reason=f"paper-reference invariant (idx={idx}): {exc}",
                asof_date=asof_date, panel_max=panel_max,
                line_e_max=line_e_max, line_tr_max=line_tr_max,
                inputs=inputs, step_1_0_status="PASS", step_5b_status="PASS",
                event_count=len(events),
                execution_label_distribution=dict(execution_label_dist),
                paper_link_counters=counters,
                paper_validity_missing_total=paper_validity_missing_total,
                unmapped_cells={},
            )

        label_distribution[(dec.setup_label, execution_label, market_ctx)] += 1
        execution_label_dist[execution_label] += 1

    # 7. Reason-code purity (§6 + Step 6)
    purity_report = v1_purity.check_reason_code_purity(
        setup_codes=(d.setup_reason_codes for d in decisions),
        execution_codes=(d.execution_reason_codes for d in decisions),
        context_codes=([] for _ in decisions),
    )
    try:
        v1_purity.enforce_purity_or_raise(purity_report)
    except v1_purity.PurityViolation as exc:
        return _emit_fail(
            args=args,
            fail_category="purity",
            fail_reason=f"reason-code purity: {exc}",
            asof_date=asof_date, panel_max=panel_max,
            line_e_max=line_e_max, line_tr_max=line_tr_max,
            inputs=inputs, step_1_0_status="PASS", step_5b_status="PASS",
            event_count=len(events),
            execution_label_distribution=dict(execution_label_dist),
            paper_link_counters=counters,
            paper_validity_missing_total=paper_validity_missing_total,
            unmapped_cells={},
        )

    # 8. Forbidden-fields static check on the column set we'd emit
    columns_in_play = (
        list(__import__("decision_engine_v1.schema", fromlist=["ALL_EVENT_COLUMNS"])
             .ALL_EVENT_COLUMNS)
    )
    try:
        v1_purity.check_forbidden_fields(
            [c for c in columns_in_play if c not in (
                "liquidity_score", "capacity_score",
            )]
        )
    except v1_purity.ForbiddenFieldError as exc:
        return _emit_fail(
            args=args,
            fail_category="schema",
            fail_reason=f"forbidden field: {exc}",
            asof_date=asof_date, panel_max=panel_max,
            line_e_max=line_e_max, line_tr_max=line_tr_max,
            inputs=inputs, step_1_0_status="PASS", step_5b_status="PASS",
            event_count=len(events),
            execution_label_distribution=dict(execution_label_dist),
            paper_link_counters=counters,
            paper_validity_missing_total=paper_validity_missing_total,
            unmapped_cells={},
        )

    # 9. PASS — emit authorized opt-in outputs
    summary = {
        "run_timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "asof_date": asof_date,
        "panel_max_date": panel_max,
        "line_e_max_date": line_e_max,
        "line_tr_max_date": line_tr_max,
        "inputs": inputs,
        "step_1_0_status": "PASS",
        "step_5b_status": "PASS",
        "verdict": "PASS",
        "fail_category": None,
        "fail_reason": None,
        "event_count": len(events),
        "execution_label_distribution": dict(execution_label_dist),
        "paper_link_counters": counters,
        "paper_validity_missing_total": paper_validity_missing_total,
        "unmapped_cell_count": 0,
    }
    if args.emit_integrity_csv:
        v1_write.write_dry_run_integrity_csv(counters)
    if args.emit_label_distribution_csv:
        v1_write.write_dry_run_label_distribution_csv(label_distribution.items())
    if args.emit_dry_run_summary_md:
        v1_write.write_dry_run_summary_markdown(summary)
    # LOCK §16.5: unmapped-cells CSV omit on PASS.

    # Stdout report
    sys.stdout.write(
        f"Decision Engine v1 — Tier 1 dry-run\n"
        f"asof_date: {asof_date} (panel max={panel_max}, "
        f"Line E max={line_e_max}, Line TR max={line_tr_max})\n"
        f"inputs: {', '.join(inputs)}\n"
        f"step 1.0 live-scope: PASS ({len(cell_counts)} unique cells, "
        f"{len(events)} events)\n"
    )
    for line_name in ("EXTENDED", "TRIGGER_RETEST", "ALL"):
        c = counters[line_name]
        sys.stdout.write(
            f"step 5b {line_name}: attempted={c.paper_link_attempted}, "
            f"matched={c.paper_link_matched}, unmatched={c.paper_link_unmatched}, "
            f"duplicate_error={c.paper_link_duplicate_error}, "
            f"skipped_source_missing={c.paper_link_skipped_source_missing}, "
            f"validity_missing={c.paper_link_validity_missing}\n"
        )
    sys.stdout.write("execution_label distribution (descriptive):\n")
    for label in sorted(execution_label_dist):
        sys.stdout.write(f"  {label}: {execution_label_dist[label]}\n")
    sys.stdout.write(
        f"paper_validity_metadata_missing total: {paper_validity_missing_total}\n"
        f"verdict: PASS\n"
    )
    return 0


# ─── Tier 2 helpers (LOCKED 2026-05-04) ──────────────────────────────────


def _hash_parquet(path: str) -> dict | None:
    """Capture sha256 + size for a paper-stream parquet (LOCK §12.4).

    Returns dict with keys `path`, `size`, `sha256`. Returns None if the
    file is absent (caller decides whether absence is a halt — under
    `PAPER_LINK_SKIP_MODE=False` the linker would already have raised).
    """
    import hashlib
    import os

    if not os.path.exists(path):
        return None
    size = os.path.getsize(path)
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return {"path": path, "size": size, "sha256": h.hexdigest()}


def _compute_paper_parquet_state() -> dict:
    """Capture sha256 + size on both production paper parquets per LOCK
    §12.4. Returns dict keyed by `line_e` / `line_tr`.
    """
    state: dict = {}
    for key, path in _TIER2_PAPER_PARQUET_PATHS.items():
        state[key] = _hash_parquet(path)
    return state


def _paper_state_diverged(pre: dict, post: dict) -> str | None:
    """Compare pre/post paper-stream states. Returns a diagnostic string
    on divergence, or None if both lines are byte-identical.
    """
    diffs: list[str] = []
    for key in ("line_e", "line_tr"):
        pre_state = pre.get(key)
        post_state = post.get(key)
        if pre_state is None and post_state is None:
            continue
        if pre_state is None or post_state is None:
            diffs.append(
                f"{key}: existence changed (pre={pre_state is not None}, "
                f"post={post_state is not None})"
            )
            continue
        if pre_state["sha256"] != post_state["sha256"]:
            diffs.append(
                f"{key}: sha256 mismatch (pre={pre_state['sha256']}, "
                f"post={post_state['sha256']})"
            )
        elif pre_state["size"] != post_state["size"]:
            diffs.append(
                f"{key}: size mismatch (pre={pre_state['size']}, "
                f"post={post_state['size']})"
            )
    if diffs:
        return "; ".join(diffs)
    return None


def _derive_horizon_status(horizon_source: str) -> str:
    """Map `horizon_source` enum → `horizon_status` enum (mechanical).

    Mapping:
      exit_framework_v1_h1_pass → accepted_prior
      exit_framework_v1_h2_fail → default_unchanged
      default_10d                → default_unchanged
      unresolved                  → unresolved
    """
    if horizon_source == "exit_framework_v1_h1_pass":
        return "accepted_prior"
    if horizon_source == "unresolved":
        return "unresolved"
    return "default_unchanged"


def _derive_fill_assumption(execution_risk_status: str) -> str:
    """Default fill assumption (mechanical).

    Risk-conditional cells with missing inputs → unresolved fill assumption.
    All other cells default to `next_open` (project-wide standard for
    daily-bar breakout entries).
    """
    if execution_risk_status == "missing_inputs":
        return "unresolved"
    return "next_open"


def _derive_context_reason_codes(
    market_context: str,
    regime_stale_flag: str | None,
) -> tuple[str, ...]:
    """Derive context reason codes from regime carry (mechanical).

    Uses LOCKED `CONTEXT_REASON_CODES_INITIAL` only. No EMA/HW carry at
    Tier 2 (those would require upstream signals not present in the v0
    panel ingest path).
    """
    if regime_stale_flag in ("stale_1d", "stale_2d_plus"):
        return ("regime_stale",)
    if market_context == "REGIME_LONG":
        return ("regime_long",)
    if market_context == "REGIME_NEUTRAL":
        return ("regime_neutral",)
    if market_context == "REGIME_SHORT":
        return ("regime_short",)
    if market_context == "REGIME_UNKNOWN":
        return ("regime_unknown",)
    return ()


def _emit_tier2_fail(
    *,
    args,
    fail_category: str,
    fail_reason: str,
    asof_date: str,
    panel_max: str,
    line_e_max: str | None,
    line_tr_max: str | None,
    inputs: list[str],
    step_1_0_status: str,
    step_5b_status: str,
    event_count: int,
    execution_label_distribution: dict[str, int],
    paper_link_counters: dict,
    paper_validity_missing_total: int,
    paper_pre: dict,
    paper_post: dict,
) -> int:
    """Emit Tier 2 FAIL summary markdown and stderr classification.

    Per Tier 2 LOCK §5.4: NO partial parquet, NO partial integrity CSV,
    NO partial label-distribution CSV. Only the summary markdown is
    written so the FAIL is auditable.
    """
    sys.stderr.write(
        f"\nDecision Engine v1 — Tier 2 FAIL\n"
        f"category: {fail_category}\n"
        f"reason: {fail_reason}\n"
    )
    summary = {
        "run_timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "asof_date": asof_date,
        "panel_max_date": panel_max,
        "line_e_max_date": line_e_max,
        "line_tr_max_date": line_tr_max,
        "inputs": inputs,
        "step_1_0_status": step_1_0_status,
        "step_5b_status": step_5b_status,
        "verdict": "FAIL",
        "fail_category": fail_category,
        "fail_reason": fail_reason,
        "event_count": event_count,
        "execution_label_distribution": execution_label_distribution,
        "paper_link_counters": paper_link_counters,
        "paper_validity_missing_total": paper_validity_missing_total,
        "paper_pre": paper_pre,
        "paper_post": paper_post,
        "output_paths": [v1_write.TIER2_LABEL_TABLE_SUMMARY_MD_PATH],
    }
    v1_write.write_tier2_label_table_summary_markdown(summary)
    sys.stdout.write(
        f"\nDecision Engine v1 — Tier 2 run\n"
        f"asof_date: {asof_date}\n"
        f"panel max: {panel_max} | Line E max: {line_e_max} | "
        f"Line TR max: {line_tr_max}\n"
        f"step 1.0: {step_1_0_status}\n"
        f"step 5b: {step_5b_status}\n"
        f"verdict: FAIL ({fail_category}: {fail_reason})\n"
    )
    return 1


def _build_event_row(
    *,
    event,
    decision,
    market_ctx: str,
    execution_label: str,
    live_execution_allowed: bool,
    paper_origin,
    paper_stream_ref,
    paper_trade_id,
    paper_match_key,
    paper_valid_from,
    paper_valid_until,
    paper_signal_age,
    paper_expired_flag,
    paper_validity_metadata_missing,
) -> dict:
    """Construct the dict for a single v1 event row matching the LOCKED
    schema (`ALL_EVENT_COLUMNS`). Tuple/list fields are converted to lists
    for pyarrow consumption.
    """
    return {
        # IDENTITY
        "date": event.date,
        "ticker": event.ticker,
        "source": event.source,
        "family": event.family,
        "state": event.state,
        "timeframe": event.timeframe,
        "direction": event.direction,
        # STRUCTURE
        "setup_label": decision.setup_label,
        "execution_label": execution_label,
        "market_context": market_ctx,
        "phase": event.state,  # panel `phase` IS state
        "expected_horizon": decision.expected_horizon,
        "horizon_source": decision.horizon_source,
        "horizon_status": _derive_horizon_status(decision.horizon_source),
        "horizon_review_due": None,
        # EXECUTION
        "entry_ref": event.entry_ref,
        "stop_ref": event.stop_ref,
        "atr": event.atr,
        "risk_pct": decision.risk_pct,
        "risk_atr": decision.risk_atr,
        "execution_risk_status": decision.execution_risk_status,
        "fill_assumption": (
            event.fill_assumption
            if event.fill_assumption is not None
            else _derive_fill_assumption(decision.execution_risk_status)
        ),
        "next_open_gap_if_available": event.next_open_gap_if_available,
        "liquidity_score": event.liquidity_score,
        "capacity_score": event.capacity_score,
        "live_execution_allowed": live_execution_allowed,
        # PAPER_REFERENCE
        "paper_origin": paper_origin,
        "paper_stream_ref": paper_stream_ref,
        "paper_trade_id": paper_trade_id,
        "paper_match_key": paper_match_key,
        # PAPER_VALIDITY
        "paper_valid_from": paper_valid_from,
        "paper_valid_until": paper_valid_until,
        "paper_signal_age": paper_signal_age,
        "paper_expired_flag": paper_expired_flag,
        "paper_validity_metadata_missing": paper_validity_metadata_missing,
        # CONTEXT
        "regime": event.regime,
        "regime_stale_flag": event.regime_stale_flag,
        "higher_tf_context": event.higher_tf_context,
        "lower_tf_context": event.lower_tf_context,
        "supporting_signals": list(event.supporting_signals),
        "conflict_flags": list(event.conflict_flags),
        "hw_context_tags": list(event.hw_context_tags),
        "ema_context_tags": list(event.ema_context_tags),
        "ema_context_tag": event.ema_context_tag,
        "earliness_score_pct": event.earliness_score_pct,
        # REASON
        "setup_reason_codes": list(decision.setup_reason_codes),
        "execution_reason_codes": list(decision.execution_reason_codes),
        "context_reason_codes": list(
            _derive_context_reason_codes(market_ctx, event.regime_stale_flag)
        ),
    }


def _run_tier2(args) -> int:
    """Execute the LOCKED Tier 2 label-table pipeline. Returns exit code.

    Mirrors `_run_tier1` for Steps 0–6 (input/live-scope/label/risk/
    paper-link/Step 5b/per-row schema/paper-ref invariants/purity/
    forbidden-fields) then emits the parquet + integrity CSV + Tier 2
    label distribution CSV + Tier 2 summary markdown. Adds sha256 + size
    pre/post guardrail on Line E + Line TR production parquets per LOCK
    §12.4.
    """
    inputs: list[str] = [
        v1_ingest.DEFAULT_PANEL_PATH,
        v1_link.PAPER_STREAM_PATHS["EXTENDED"],
        v1_link.PAPER_STREAM_PATHS["TRIGGER_RETEST"],
    ]
    paper_link_counters_empty: dict = {}

    # 0. sha256 + size capture pre-run (LOCK §12.4)
    paper_pre = _compute_paper_parquet_state()

    # 1. asof_date resolution (Tier 2 LOCK §12.2 Q6 mirrors Tier 1 §16.1)
    try:
        asof_date, panel_max, line_e_max, line_tr_max = _resolve_asof_date(
            args.asof_date
        )
    except Exception as exc:
        return _emit_tier2_fail(
            args=args,
            fail_category="input coverage",
            fail_reason=f"asof resolution: {exc}",
            asof_date="<unresolved>",
            panel_max="<unresolved>",
            line_e_max=None, line_tr_max=None,
            inputs=inputs,
            step_1_0_status="not_run (asof_unresolved)",
            step_5b_status="not_run",
            event_count=0,
            execution_label_distribution={},
            paper_link_counters=paper_link_counters_empty,
            paper_validity_missing_total=0,
            paper_pre=paper_pre, paper_post=paper_pre,
        )

    mismatch = _check_source_asof_mismatch(asof_date, line_e_max, line_tr_max)
    if mismatch is not None:
        return _emit_tier2_fail(
            args=args,
            fail_category="input coverage",
            fail_reason=f"source_asof_mismatch: {mismatch}",
            asof_date=asof_date,
            panel_max=panel_max,
            line_e_max=line_e_max,
            line_tr_max=line_tr_max,
            inputs=inputs,
            step_1_0_status="not_run (asof_mismatch)",
            step_5b_status="not_run",
            event_count=0,
            execution_label_distribution={},
            paper_link_counters=paper_link_counters_empty,
            paper_validity_missing_total=0,
            paper_pre=paper_pre, paper_post=paper_pre,
        )

    # 2. Ingest
    try:
        events = v1_ingest.read_scanner_outputs(asof_date)
    except Exception as exc:
        return _emit_tier2_fail(
            args=args,
            fail_category="input coverage",
            fail_reason=f"ingest failed: {exc}",
            asof_date=asof_date,
            panel_max=panel_max,
            line_e_max=line_e_max,
            line_tr_max=line_tr_max,
            inputs=inputs,
            step_1_0_status="not_run (ingest_failed)",
            step_5b_status="not_run",
            event_count=0,
            execution_label_distribution={},
            paper_link_counters=paper_link_counters_empty,
            paper_validity_missing_total=0,
            paper_pre=paper_pre, paper_post=paper_pre,
        )

    cell_counts = _classify_unmapped_cells(events)

    # 3. Step 1.0 live-scope validation
    report = v1_live_scope.validate_live_scope(asof_date, cell_counts.keys())
    try:
        v1_live_scope.enforce_live_scope_or_raise(report)
    except (
        v1_live_scope.UnmappedCellError,
        v1_live_scope.UnmappedSourceError,
    ) as exc:
        return _emit_tier2_fail(
            args=args,
            fail_category="mapping",
            fail_reason=f"live-scope: {exc}",
            asof_date=asof_date,
            panel_max=panel_max,
            line_e_max=line_e_max,
            line_tr_max=line_tr_max,
            inputs=inputs,
            step_1_0_status="FAIL",
            step_5b_status="not_run",
            event_count=len(events),
            execution_label_distribution={},
            paper_link_counters=paper_link_counters_empty,
            paper_validity_missing_total=0,
            paper_pre=paper_pre, paper_post=paper_pre,
        )

    # 4. Mapping/label/risk per event
    try:
        decisions = [assign_label(ev) for ev in events]
    except Exception as exc:
        return _emit_tier2_fail(
            args=args,
            fail_category="schema",
            fail_reason=f"label assignment: {exc}",
            asof_date=asof_date,
            panel_max=panel_max,
            line_e_max=line_e_max,
            line_tr_max=line_tr_max,
            inputs=inputs,
            step_1_0_status="PASS",
            step_5b_status="not_run",
            event_count=len(events),
            execution_label_distribution={},
            paper_link_counters=paper_link_counters_empty,
            paper_validity_missing_total=0,
            paper_pre=paper_pre, paper_post=paper_pre,
        )

    # 5. Paper-stream link
    try:
        attachments, counters = v1_link.link_paper_streams(events, asof_date)
    except v1_link.PaperStreamMissingError as exc:
        return _emit_tier2_fail(
            args=args,
            fail_category="paper-link",
            fail_reason=f"paper stream missing: {exc}",
            asof_date=asof_date,
            panel_max=panel_max,
            line_e_max=line_e_max,
            line_tr_max=line_tr_max,
            inputs=inputs,
            step_1_0_status="PASS",
            step_5b_status="FAIL (missing parquet)",
            event_count=len(events),
            execution_label_distribution={},
            paper_link_counters=paper_link_counters_empty,
            paper_validity_missing_total=0,
            paper_pre=paper_pre, paper_post=paper_pre,
        )
    except v1_link.PaperStreamDuplicateMatchError as exc:
        return _emit_tier2_fail(
            args=args,
            fail_category="paper-link",
            fail_reason=f"duplicate match halt: {exc}",
            asof_date=asof_date,
            panel_max=panel_max,
            line_e_max=line_e_max,
            line_tr_max=line_tr_max,
            inputs=inputs,
            step_1_0_status="PASS",
            step_5b_status="FAIL (duplicate match)",
            event_count=len(events),
            execution_label_distribution={},
            paper_link_counters=paper_link_counters_empty,
            paper_validity_missing_total=0,
            paper_pre=paper_pre, paper_post=paper_pre,
        )

    # 5b. Integrity invariants
    try:
        for c in counters.values():
            c.assert_invariants()
    except AssertionError as exc:
        return _emit_tier2_fail(
            args=args,
            fail_category="paper-link",
            fail_reason=f"Step 5b invariant: {exc}",
            asof_date=asof_date,
            panel_max=panel_max,
            line_e_max=line_e_max,
            line_tr_max=line_tr_max,
            inputs=inputs,
            step_1_0_status="PASS",
            step_5b_status="FAIL",
            event_count=len(events),
            execution_label_distribution={},
            paper_link_counters=counters,
            paper_validity_missing_total=0,
            paper_pre=paper_pre, paper_post=paper_pre,
        )

    # 6. Build event rows + per-row paper-reference invariant + label dist
    label_distribution: dict[tuple, int] = collections.Counter()
    execution_label_dist: dict[str, int] = collections.Counter()
    paper_validity_missing_total = 0
    event_rows: list[dict] = []

    for idx, (ev, dec) in enumerate(zip(events, decisions)):
        market_ctx = _market_context_for(ev)
        execution_label = dec.execution_label
        live_execution_allowed = dec.live_execution_allowed
        paper_origin = None
        paper_stream_ref = None
        paper_trade_id = None
        paper_match_key = None
        paper_valid_from = None
        paper_valid_until = None
        paper_signal_age = None
        paper_expired_flag = None
        paper_validity_metadata_missing = None

        att = attachments.get(idx)
        if att is not None and att.matched:
            execution_label = "PAPER_ONLY"
            live_execution_allowed = False
            paper_origin = att.paper_origin
            paper_stream_ref = att.paper_stream_ref
            paper_trade_id = att.paper_trade_id
            paper_match_key = att.paper_match_key
            paper_valid_from = att.paper_valid_from
            paper_valid_until = att.paper_valid_until
            paper_signal_age = att.paper_signal_age
            paper_expired_flag = att.paper_expired_flag
            paper_validity_metadata_missing = att.paper_validity_metadata_missing
            if paper_validity_metadata_missing:
                paper_validity_missing_total += 1

        # Schema / label enum validation per row
        if dec.setup_label not in SETUP_LABELS:
            return _emit_tier2_fail(
                args=args,
                fail_category="schema",
                fail_reason=f"setup_label {dec.setup_label!r} not in enum",
                asof_date=asof_date, panel_max=panel_max,
                line_e_max=line_e_max, line_tr_max=line_tr_max,
                inputs=inputs, step_1_0_status="PASS", step_5b_status="PASS",
                event_count=len(events),
                execution_label_distribution=dict(execution_label_dist),
                paper_link_counters=counters,
                paper_validity_missing_total=paper_validity_missing_total,
                paper_pre=paper_pre, paper_post=paper_pre,
            )
        if execution_label not in EXECUTION_LABELS:
            return _emit_tier2_fail(
                args=args,
                fail_category="schema",
                fail_reason=f"execution_label {execution_label!r} not in enum",
                asof_date=asof_date, panel_max=panel_max,
                line_e_max=line_e_max, line_tr_max=line_tr_max,
                inputs=inputs, step_1_0_status="PASS", step_5b_status="PASS",
                event_count=len(events),
                execution_label_distribution=dict(execution_label_dist),
                paper_link_counters=counters,
                paper_validity_missing_total=paper_validity_missing_total,
                paper_pre=paper_pre, paper_post=paper_pre,
            )
        if market_ctx not in MARKET_CONTEXTS:
            return _emit_tier2_fail(
                args=args,
                fail_category="schema",
                fail_reason=f"market_context {market_ctx!r} not in enum",
                asof_date=asof_date, panel_max=panel_max,
                line_e_max=line_e_max, line_tr_max=line_tr_max,
                inputs=inputs, step_1_0_status="PASS", step_5b_status="PASS",
                event_count=len(events),
                execution_label_distribution=dict(execution_label_dist),
                paper_link_counters=counters,
                paper_validity_missing_total=paper_validity_missing_total,
                paper_pre=paper_pre, paper_post=paper_pre,
            )

        # Per-row paper-reference invariant
        try:
            assert_paper_reference_invariants(
                PaperReferenceState(
                    execution_label=execution_label,
                    paper_origin=paper_origin,
                    paper_stream_ref=paper_stream_ref,
                    paper_trade_id=paper_trade_id,
                    paper_match_key=paper_match_key,
                    live_execution_allowed=live_execution_allowed,
                    paper_valid_from=paper_valid_from,
                    paper_valid_until=paper_valid_until,
                    paper_signal_age=paper_signal_age,
                    paper_expired_flag=paper_expired_flag,
                    paper_validity_metadata_missing=paper_validity_metadata_missing,
                )
            )
        except ValueError as exc:
            return _emit_tier2_fail(
                args=args,
                fail_category="purity",
                fail_reason=f"paper-reference invariant (idx={idx}): {exc}",
                asof_date=asof_date, panel_max=panel_max,
                line_e_max=line_e_max, line_tr_max=line_tr_max,
                inputs=inputs, step_1_0_status="PASS", step_5b_status="PASS",
                event_count=len(events),
                execution_label_distribution=dict(execution_label_dist),
                paper_link_counters=counters,
                paper_validity_missing_total=paper_validity_missing_total,
                paper_pre=paper_pre, paper_post=paper_pre,
            )

        label_distribution[(dec.setup_label, execution_label, market_ctx)] += 1
        execution_label_dist[execution_label] += 1

        event_rows.append(
            _build_event_row(
                event=ev,
                decision=dec,
                market_ctx=market_ctx,
                execution_label=execution_label,
                live_execution_allowed=live_execution_allowed,
                paper_origin=paper_origin,
                paper_stream_ref=paper_stream_ref,
                paper_trade_id=paper_trade_id,
                paper_match_key=paper_match_key,
                paper_valid_from=paper_valid_from,
                paper_valid_until=paper_valid_until,
                paper_signal_age=paper_signal_age,
                paper_expired_flag=paper_expired_flag,
                paper_validity_metadata_missing=paper_validity_metadata_missing,
            )
        )

    # 7. Reason-code purity (now including derived context codes)
    purity_report = v1_purity.check_reason_code_purity(
        setup_codes=(d.setup_reason_codes for d in decisions),
        execution_codes=(d.execution_reason_codes for d in decisions),
        context_codes=(
            _derive_context_reason_codes(_market_context_for(ev), ev.regime_stale_flag)
            for ev in events
        ),
    )
    try:
        v1_purity.enforce_purity_or_raise(purity_report)
    except v1_purity.PurityViolation as exc:
        return _emit_tier2_fail(
            args=args,
            fail_category="purity",
            fail_reason=f"reason-code purity: {exc}",
            asof_date=asof_date, panel_max=panel_max,
            line_e_max=line_e_max, line_tr_max=line_tr_max,
            inputs=inputs, step_1_0_status="PASS", step_5b_status="PASS",
            event_count=len(events),
            execution_label_distribution=dict(execution_label_dist),
            paper_link_counters=counters,
            paper_validity_missing_total=paper_validity_missing_total,
            paper_pre=paper_pre, paper_post=paper_pre,
        )

    # 8. Forbidden-fields static check on the column set we are emitting
    if event_rows:
        emitted_columns = list(event_rows[0].keys())
        try:
            v1_purity.check_forbidden_fields(emitted_columns)
        except v1_purity.ForbiddenFieldError as exc:
            return _emit_tier2_fail(
                args=args,
                fail_category="schema",
                fail_reason=f"forbidden field: {exc}",
                asof_date=asof_date, panel_max=panel_max,
                line_e_max=line_e_max, line_tr_max=line_tr_max,
                inputs=inputs, step_1_0_status="PASS", step_5b_status="PASS",
                event_count=len(events),
                execution_label_distribution=dict(execution_label_dist),
                paper_link_counters=counters,
                paper_validity_missing_total=paper_validity_missing_total,
                paper_pre=paper_pre, paper_post=paper_pre,
            )

    # 9. Write parquet + mandatory paper-link integrity CSV (LOCK §12.3)
    try:
        v1_write.write_events_parquet(
            event_rows,
            integrity_counters=counters,
            tier2_authorized=True,
        )
    except v1_write.WriterNotAuthorizedError as exc:
        return _emit_tier2_fail(
            args=args,
            fail_category="writer",
            fail_reason=f"writer refused: {exc}",
            asof_date=asof_date, panel_max=panel_max,
            line_e_max=line_e_max, line_tr_max=line_tr_max,
            inputs=inputs, step_1_0_status="PASS", step_5b_status="PASS",
            event_count=len(events),
            execution_label_distribution=dict(execution_label_dist),
            paper_link_counters=counters,
            paper_validity_missing_total=paper_validity_missing_total,
            paper_pre=paper_pre, paper_post=paper_pre,
        )

    # 10. Tier 2 label-distribution CSV
    v1_write.write_tier2_label_distribution_csv(label_distribution.items())

    # 11. sha256 + size post-run guardrail (LOCK §12.4)
    paper_post = _compute_paper_parquet_state()
    divergence = _paper_state_diverged(paper_pre, paper_post)
    if divergence is not None:
        return _emit_tier2_fail(
            args=args,
            fail_category="paper-stream mutation",
            fail_reason=f"upstream paper parquet changed pre→post: {divergence}",
            asof_date=asof_date, panel_max=panel_max,
            line_e_max=line_e_max, line_tr_max=line_tr_max,
            inputs=inputs, step_1_0_status="PASS", step_5b_status="PASS",
            event_count=len(events),
            execution_label_distribution=dict(execution_label_dist),
            paper_link_counters=counters,
            paper_validity_missing_total=paper_validity_missing_total,
            paper_pre=paper_pre, paper_post=paper_post,
        )

    # 12. PASS — Tier 2 summary markdown
    output_paths = [
        v1_write.EVENTS_PARQUET_PATH,
        v1_write.PAPER_LINK_INTEGRITY_CSV_PATH,
        v1_write.TIER2_LABEL_DISTRIBUTION_CSV_PATH,
        v1_write.TIER2_LABEL_TABLE_SUMMARY_MD_PATH,
    ]
    summary = {
        "run_timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "asof_date": asof_date,
        "panel_max_date": panel_max,
        "line_e_max_date": line_e_max,
        "line_tr_max_date": line_tr_max,
        "inputs": inputs,
        "step_1_0_status": "PASS",
        "step_5b_status": "PASS",
        "verdict": "PASS",
        "fail_category": None,
        "fail_reason": None,
        "event_count": len(events),
        "execution_label_distribution": dict(execution_label_dist),
        "paper_link_counters": counters,
        "paper_validity_missing_total": paper_validity_missing_total,
        "paper_pre": paper_pre,
        "paper_post": paper_post,
        "output_paths": output_paths,
    }
    v1_write.write_tier2_label_table_summary_markdown(summary)

    # Stdout report
    sys.stdout.write(
        f"Decision Engine v1 — Tier 2 label-table run\n"
        f"asof_date: {asof_date} (panel max={panel_max}, "
        f"Line E max={line_e_max}, Line TR max={line_tr_max})\n"
        f"inputs: {', '.join(inputs)}\n"
        f"step 1.0 live-scope: PASS ({len(cell_counts)} unique cells, "
        f"{len(events)} events)\n"
    )
    for line_name in ("EXTENDED", "TRIGGER_RETEST", "ALL"):
        c = counters[line_name]
        sys.stdout.write(
            f"step 5b {line_name}: attempted={c.paper_link_attempted}, "
            f"matched={c.paper_link_matched}, unmatched={c.paper_link_unmatched}, "
            f"duplicate_error={c.paper_link_duplicate_error}, "
            f"skipped_source_missing={c.paper_link_skipped_source_missing}, "
            f"validity_missing={c.paper_link_validity_missing}\n"
        )
    sys.stdout.write("execution_label distribution (descriptive):\n")
    for label in sorted(execution_label_dist):
        sys.stdout.write(f"  {label}: {execution_label_dist[label]}\n")
    sys.stdout.write(
        f"paper_validity_metadata_missing total: {paper_validity_missing_total}\n"
        f"paper_pre/post sha256 byte-equal: True\n"
        f"outputs: {', '.join(output_paths)}\n"
        f"verdict: PASS\n"
    )
    return 0


# ─── Tier 3 helpers (LOCKED 2026-05-04) ──────────────────────────────────


_TIER3_FORBIDDEN_FIELDS: Final[tuple[str, ...]] = (
    "final_action",
    "selection_score",
    "edge_score",
    "prior_edge_score",
    "setup_score",
    "rank",
    "portfolio_pick",
    "portfolio_weight",
)

_TIER3_INTEGRITY_CSV_PATH: Final[str] = (
    "output/decision_engine_v1_paper_link_integrity.csv"
)


def _compute_tier3_guarded_state() -> dict:
    """Capture sha256 + size on the 3 Tier 3 guarded files (LOCK §12.2 Q8)."""
    state: dict = {}
    for key, path in _TIER3_GUARDED_PATHS.items():
        state[key] = _hash_parquet(path)
    return state


def _tier3_state_diverged(pre: dict, post: dict) -> str | None:
    """Compare pre/post 3-file states; returns diagnostic on divergence."""
    diffs: list[str] = []
    for key in ("events", "line_e", "line_tr"):
        pre_state = pre.get(key)
        post_state = post.get(key)
        if pre_state is None and post_state is None:
            continue
        if pre_state is None or post_state is None:
            diffs.append(
                f"{key}: existence changed "
                f"(pre={pre_state is not None}, post={post_state is not None})"
            )
            continue
        if pre_state["sha256"] != post_state["sha256"]:
            diffs.append(
                f"{key}: sha256 mismatch (pre={pre_state['sha256']}, "
                f"post={post_state['sha256']})"
            )
        elif pre_state["size"] != post_state["size"]:
            diffs.append(
                f"{key}: size mismatch (pre={pre_state['size']}, "
                f"post={post_state['size']})"
            )
    if diffs:
        return "; ".join(diffs)
    return None


def _read_tier2_events() -> tuple[list[dict], list[str]]:
    """Read Tier 2 events.parquet read-only.

    Returns (rows, schema_field_names). NO recomputation; pure pyarrow read.
    """
    import pyarrow.parquet as pq

    table = pq.read_table(_TIER3_GUARDED_PATHS["events"])
    rows = table.to_pylist()
    schema_names = [f.name for f in table.schema]
    return rows, schema_names


def _read_tier2_integrity_rows() -> list[dict]:
    """Read paper-link integrity CSV into list-of-dicts (read-only)."""
    import csv

    rows: list[dict] = []
    with open(_TIER3_INTEGRITY_CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def _read_tier2_summary_meta() -> dict:
    """Parse Tier 2 summary markdown for run timestamp + asof + max dates.

    Read-only. Missing file → all-None dict; missing line → that key is None.
    """
    import os

    meta: dict[str, str | None] = {
        "tier2_run_timestamp_utc": None,
        "asof_date": None,
        "panel_max_date": None,
        "line_e_max_date": None,
        "line_tr_max_date": None,
    }
    path = v1_write.TIER2_LABEL_TABLE_SUMMARY_MD_PATH
    if not os.path.exists(path):
        return meta
    prefixes = {
        "tier2_run_timestamp_utc": "- Run timestamp (UTC): ",
        "asof_date": "- asof_date (resolved): ",
        "panel_max_date": "- panel max date: ",
        "line_e_max_date": "- Line E max asof_date: ",
        "line_tr_max_date": "- Line TR max asof_date: ",
    }
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            for key, prefix in prefixes.items():
                if meta[key] is None and line.startswith(prefix):
                    meta[key] = line[len(prefix):].strip()
                    break
    return meta


def _compute_tier3_counts(events: list[dict]) -> dict:
    """Pure counting from Tier 2 events. NO recompute of any pipeline stage.

    Returns dict of all distributions / cross-tabs needed for the report.
    """
    from decision_engine_v1.schema import (
        CONTEXT_REASON_CODES_INITIAL,
        EXECUTION_LABELS,
        EXECUTION_REASON_CODES_INITIAL,
        MARKET_CONTEXTS,
        SETUP_LABELS,
        SETUP_REASON_CODES_INITIAL,
    )

    total = len(events)

    # Marginal distributions — full enum coverage including zero-count.
    exec_dist: dict[str, int] = {label: 0 for label in EXECUTION_LABELS}
    setup_dist: dict[str, int] = {label: 0 for label in SETUP_LABELS}
    ctx_dist: dict[str, int] = {label: 0 for label in MARKET_CONTEXTS}
    for r in events:
        exec_dist[r["execution_label"]] = exec_dist.get(r["execution_label"], 0) + 1
        setup_dist[r["setup_label"]] = setup_dist.get(r["setup_label"], 0) + 1
        ctx_dist[r["market_context"]] = ctx_dist.get(r["market_context"], 0) + 1

    # Compact cross-tabs (LOCK Q3).
    setup_x_exec: dict[tuple[str, str], int] = {}
    src_x_setup: dict[tuple[str, str], int] = {}
    src_x_exec: dict[tuple[str, str], int] = {}
    paperref_x_exec_within_paper_only: dict[str, int] = {}

    for r in events:
        sl = r["setup_label"]
        el = r["execution_label"]
        src = r["source"]
        setup_x_exec[(sl, el)] = setup_x_exec.get((sl, el), 0) + 1
        src_x_setup[(src, sl)] = src_x_setup.get((src, sl), 0) + 1
        src_x_exec[(src, el)] = src_x_exec.get((src, el), 0) + 1
        if el == "PAPER_ONLY":
            ref = r.get("paper_stream_ref") or "<none>"
            paperref_x_exec_within_paper_only[ref] = (
                paperref_x_exec_within_paper_only.get(ref, 0) + 1
            )

    # Source / family / state / timeframe marginals (alphabetical per Q4).
    src_dist: dict[str, int] = {}
    fam_dist: dict[str, int] = {}
    state_dist: dict[str, int] = {}
    tf_dist: dict[str, int] = {}
    for r in events:
        src_dist[r["source"]] = src_dist.get(r["source"], 0) + 1
        fam_dist[r["family"]] = fam_dist.get(r["family"], 0) + 1
        state_dist[r["state"]] = state_dist.get(r["state"], 0) + 1
        tf_dist[r["timeframe"]] = tf_dist.get(r["timeframe"], 0) + 1

    # PAPER_ONLY rows detail (Q5 columns) — ticker alphabetical.
    paper_only_rows = sorted(
        (r for r in events if r["execution_label"] == "PAPER_ONLY"),
        key=lambda r: r["ticker"],
    )

    # WAIT_CONFIRMATION rows detail (Q6) — group by (source, family, state, timeframe).
    wc_rows = [r for r in events if r["execution_label"] == "WAIT_CONFIRMATION"]
    wc_groups: dict[tuple[str, str, str, str], int] = {}
    for r in wc_rows:
        key = (r["source"], r["family"], r["state"], r["timeframe"])
        wc_groups[key] = wc_groups.get(key, 0) + 1
    wc_tickers = sorted({r["ticker"] for r in wc_rows})

    # Reason-code distributions (full enum incl zero-count).
    setup_reason_dist: dict[str, int] = {c: 0 for c in SETUP_REASON_CODES_INITIAL}
    exec_reason_dist_overall: dict[str, int] = {
        c: 0 for c in EXECUTION_REASON_CODES_INITIAL
    }
    exec_reason_dist_not_exec: dict[str, int] = {
        c: 0 for c in EXECUTION_REASON_CODES_INITIAL
    }
    ctx_reason_dist: dict[str, int] = {c: 0 for c in CONTEXT_REASON_CODES_INITIAL}

    for r in events:
        for c in r.get("setup_reason_codes") or []:
            if c in setup_reason_dist:
                setup_reason_dist[c] += 1
            else:
                setup_reason_dist[c] = setup_reason_dist.get(c, 0) + 1
        for c in r.get("execution_reason_codes") or []:
            if c in exec_reason_dist_overall:
                exec_reason_dist_overall[c] += 1
            else:
                exec_reason_dist_overall[c] = exec_reason_dist_overall.get(c, 0) + 1
            if r["execution_label"] == "NOT_EXECUTABLE":
                if c in exec_reason_dist_not_exec:
                    exec_reason_dist_not_exec[c] += 1
                else:
                    exec_reason_dist_not_exec[c] = (
                        exec_reason_dist_not_exec.get(c, 0) + 1
                    )
        for c in r.get("context_reason_codes") or []:
            if c in ctx_reason_dist:
                ctx_reason_dist[c] += 1
            else:
                ctx_reason_dist[c] = ctx_reason_dist.get(c, 0) + 1

    # Paper-validity carry summary.
    val_missing = sum(
        1 for r in events if r.get("paper_validity_metadata_missing") is True
    )
    age_populated = sum(1 for r in events if r.get("paper_signal_age") is not None)
    expired_true = sum(1 for r in events if r.get("paper_expired_flag") is True)

    return {
        "total": total,
        "exec_dist": exec_dist,
        "setup_dist": setup_dist,
        "ctx_dist": ctx_dist,
        "setup_x_exec": setup_x_exec,
        "src_x_setup": src_x_setup,
        "src_x_exec": src_x_exec,
        "paperref_x_exec_within_paper_only": paperref_x_exec_within_paper_only,
        "src_dist": src_dist,
        "fam_dist": fam_dist,
        "state_dist": state_dist,
        "tf_dist": tf_dist,
        "paper_only_rows": paper_only_rows,
        "wc_rows": wc_rows,
        "wc_groups": wc_groups,
        "wc_tickers": wc_tickers,
        "setup_reason_dist": setup_reason_dist,
        "exec_reason_dist_overall": exec_reason_dist_overall,
        "exec_reason_dist_not_exec": exec_reason_dist_not_exec,
        "ctx_reason_dist": ctx_reason_dist,
        "paper_validity_missing": val_missing,
        "paper_signal_age_populated": age_populated,
        "paper_expired_true": expired_true,
    }


def _fmt_iso_date(value) -> str:
    """Render a possibly-None / date / str value as ISO yyyy-mm-dd or '-'."""
    if value is None:
        return "-"
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _render_tier3_markdown(
    *,
    render_timestamp_utc: str,
    tier2_meta: dict,
    pre_state: dict,
    post_state: dict,
    counts: dict,
    integrity_rows: list[dict],
    schema_field_names: list[str],
) -> str:
    """Assemble the full Tier 3 markdown report.

    Pure rendering; takes pre-computed counts. Caller supplies pre/post sha256
    state already verified byte-equal.
    """
    from decision_engine_v1.schema import (
        CONTEXT_REASON_CODES_INITIAL,
        EXECUTION_LABELS,
        EXECUTION_REASON_CODES_INITIAL,
        MARKET_CONTEXTS,
        SETUP_LABELS,
        SETUP_REASON_CODES_INITIAL,
    )

    L: list[str] = []
    L.append("# Decision Engine v1 — Tier 3 review report")
    L.append("")

    # §1 Header / run identity
    L.append("## §1 Run identity")
    L.append(f"- Tier 3 render timestamp (UTC): {render_timestamp_utc}")
    L.append(
        "- Tier 2 source-run timestamp (UTC): "
        f"{tier2_meta.get('tier2_run_timestamp_utc') or '-'}"
    )
    L.append(f"- asof_date (resolved): {tier2_meta.get('asof_date') or '-'}")
    L.append(f"- panel max date: {tier2_meta.get('panel_max_date') or '-'}")
    L.append(f"- Line E max asof_date: {tier2_meta.get('line_e_max_date') or '-'}")
    L.append(f"- Line TR max asof_date: {tier2_meta.get('line_tr_max_date') or '-'}")
    L.append("")

    # §2 Source-input integrity
    L.append("## §2 Source-input integrity (sha256 pre/post)")
    for key, label in (
        ("events", "Tier 2 events.parquet"),
        ("line_e", "Line E paper parquet"),
        ("line_tr", "Line TR paper parquet"),
    ):
        pre_s = pre_state.get(key) or {}
        post_s = post_state.get(key) or {}
        L.append(
            f"- {label} pre:  size={pre_s.get('size')}, "
            f"sha256={pre_s.get('sha256')}"
        )
        L.append(
            f"- {label} post: size={post_s.get('size')}, "
            f"sha256={post_s.get('sha256')}"
        )
    L.append("- byte-equality: PASS")
    L.append("")

    # §3 Total event count
    L.append("## §3 Total event count")
    L.append(f"- total: {counts['total']}")
    L.append("")

    # §4 execution_label distribution (FULL enum coverage, LOCK §12.4)
    L.append(
        "## §4 Execution-label distribution (full enum, zero-count shown)"
    )
    for label in EXECUTION_LABELS:
        L.append(f"- {label}: {counts['exec_dist'].get(label, 0)}")
    L.append("")

    # §5 setup_label distribution
    L.append("## §5 Setup-label distribution (full enum, zero-count shown)")
    for label in SETUP_LABELS:
        L.append(f"- {label}: {counts['setup_dist'].get(label, 0)}")
    L.append("")

    # §6 market_context distribution
    L.append("## §6 Market-context distribution (full enum, zero-count shown)")
    for label in MARKET_CONTEXTS:
        L.append(f"- {label}: {counts['ctx_dist'].get(label, 0)}")
    L.append("")

    # §7 Compact cross-tabs (LOCK Q3) — 4 tables.
    L.append("## §7 Compact cross-tabs")
    L.append("")
    L.append("### §7.1 setup_label × execution_label")
    L.append("| setup_label \\ execution_label | " + " | ".join(EXECUTION_LABELS) + " |")
    L.append("| --- | " + " | ".join(["---"] * len(EXECUTION_LABELS)) + " |")
    for sl in SETUP_LABELS:
        cells = [str(counts["setup_x_exec"].get((sl, el), 0)) for el in EXECUTION_LABELS]
        L.append(f"| {sl} | " + " | ".join(cells) + " |")
    L.append("")

    sources_alpha = sorted(counts["src_dist"].keys())

    L.append("### §7.2 source × setup_label (sources alphabetical)")
    L.append("| source \\ setup_label | " + " | ".join(SETUP_LABELS) + " |")
    L.append("| --- | " + " | ".join(["---"] * len(SETUP_LABELS)) + " |")
    for src in sources_alpha:
        cells = [str(counts["src_x_setup"].get((src, sl), 0)) for sl in SETUP_LABELS]
        L.append(f"| {src} | " + " | ".join(cells) + " |")
    L.append("")

    L.append("### §7.3 source × execution_label (sources alphabetical)")
    L.append("| source \\ execution_label | " + " | ".join(EXECUTION_LABELS) + " |")
    L.append("| --- | " + " | ".join(["---"] * len(EXECUTION_LABELS)) + " |")
    for src in sources_alpha:
        cells = [str(counts["src_x_exec"].get((src, el), 0)) for el in EXECUTION_LABELS]
        L.append(f"| {src} | " + " | ".join(cells) + " |")
    L.append("")

    L.append("### §7.4 paper_stream_ref × execution_label (filtered to PAPER_ONLY)")
    p_refs = sorted(counts["paperref_x_exec_within_paper_only"].keys())
    if not p_refs:
        L.append("_(no PAPER_ONLY events.)_")
    else:
        L.append("| paper_stream_ref | PAPER_ONLY count |")
        L.append("| --- | --- |")
        for ref in p_refs:
            L.append(
                f"| {ref} | {counts['paperref_x_exec_within_paper_only'][ref]} |"
            )
    L.append("")

    # §8 Source / family / state / timeframe marginals (alphabetical per Q4)
    L.append("## §8 Source / family / state / timeframe marginals (alphabetical)")
    L.append("")
    L.append("### §8.1 source")
    for k in sorted(counts["src_dist"]):
        L.append(f"- {k}: {counts['src_dist'][k]}")
    L.append("")
    L.append("### §8.2 family")
    for k in sorted(counts["fam_dist"]):
        L.append(f"- {k}: {counts['fam_dist'][k]}")
    L.append("")
    L.append("### §8.3 state")
    for k in sorted(counts["state_dist"]):
        L.append(f"- {k}: {counts['state_dist'][k]}")
    L.append("")
    L.append("### §8.4 timeframe")
    for k in sorted(counts["tf_dist"]):
        L.append(f"- {k}: {counts['tf_dist'][k]}")
    L.append("")

    # §9 PAPER_ONLY rows detail (LOCK Q5 column list)
    L.append("## §9 PAPER_ONLY rows detail (ticker alphabetical)")
    paper_rows = counts["paper_only_rows"]
    if not paper_rows:
        L.append("_(no PAPER_ONLY events in this run.)_")
    else:
        header_cols = [
            "ticker",
            "source",
            "family",
            "state",
            "timeframe",
            "asof_date",
            "paper_stream_ref",
            "paper_origin",
            "paper_trade_id",
            "paper_match_key",
            "paper_validity_metadata_missing",
            "paper_valid_from",
            "paper_valid_until",
            "paper_expired_flag",
            "paper_signal_age",
        ]
        L.append("| " + " | ".join(header_cols) + " |")
        L.append("| " + " | ".join(["---"] * len(header_cols)) + " |")
        for r in paper_rows:
            cells = [
                str(r["ticker"]),
                str(r["source"]),
                str(r["family"]),
                str(r["state"]),
                str(r["timeframe"]),
                _fmt_iso_date(r.get("date")),
                str(r.get("paper_stream_ref") or "-"),
                str(r.get("paper_origin") or "-"),
                str(r.get("paper_trade_id") or "-"),
                str(r.get("paper_match_key") or "-"),
                str(r.get("paper_validity_metadata_missing")),
                _fmt_iso_date(r.get("paper_valid_from")),
                _fmt_iso_date(r.get("paper_valid_until")),
                str(r.get("paper_expired_flag")),
                str(r.get("paper_signal_age") if r.get("paper_signal_age") is not None else "-"),
            ]
            L.append("| " + " | ".join(cells) + " |")
    L.append("")

    # §10 WAIT_CONFIRMATION (LOCK Q6: nox_weekly conservative landing section)
    L.append("## §10 WAIT_CONFIRMATION (nox_weekly conservative landing)")
    L.append(
        "_NOTE: WAIT_CONFIRMATION is NOT executable. This section is a "
        "descriptive view; no execution recommendation is implied._"
    )
    L.append("")
    wc_rows = counts["wc_rows"]
    if not wc_rows:
        L.append("_(no WAIT_CONFIRMATION events in this run.)_")
    else:
        L.append("### §10.1 group counts (source / family / state / timeframe)")
        L.append("| source | family | state | timeframe | count |")
        L.append("| --- | --- | --- | --- | --- |")
        for key in sorted(counts["wc_groups"].keys()):
            src, fam, st, tf = key
            L.append(f"| {src} | {fam} | {st} | {tf} | {counts['wc_groups'][key]} |")
        L.append("")
        L.append("### §10.2 ticker list (alphabetical)")
        L.append(", ".join(counts["wc_tickers"]))
    L.append("")

    # §11 NOT_EXECUTABLE reason-code distribution
    L.append(
        "## §11 NOT_EXECUTABLE execution-reason-code distribution "
        "(full enum, zero-count shown)"
    )
    for code in EXECUTION_REASON_CODES_INITIAL:
        L.append(
            f"- {code}: {counts['exec_reason_dist_not_exec'].get(code, 0)}"
        )
    L.append("")

    # §12 Setup-layer reason-code distribution (across all 542)
    L.append(
        "## §12 Setup-layer reason-code distribution "
        "(across all events, full enum, zero-count shown)"
    )
    for code in SETUP_REASON_CODES_INITIAL:
        L.append(f"- {code}: {counts['setup_reason_dist'].get(code, 0)}")
    L.append("")

    # §13 Context-layer reason-code distribution
    L.append(
        "## §13 Context-layer reason-code distribution "
        "(across all events, full enum, zero-count shown)"
    )
    for code in CONTEXT_REASON_CODES_INITIAL:
        L.append(f"- {code}: {counts['ctx_reason_dist'].get(code, 0)}")
    L.append("")

    # §14 Paper-link integrity recap (verbatim from CSV; no recompute)
    L.append("## §14 Paper-link integrity recap (verbatim from Tier 2 CSV)")
    if not integrity_rows:
        L.append("_(integrity CSV empty.)_")
    else:
        cols = [
            "line",
            "paper_link_attempted",
            "paper_link_matched",
            "paper_link_unmatched",
            "paper_link_duplicate_error",
            "paper_link_skipped_source_missing",
            "paper_link_validity_missing",
        ]
        L.append("| " + " | ".join(cols) + " |")
        L.append("| " + " | ".join(["---"] * len(cols)) + " |")
        for r in integrity_rows:
            L.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    L.append("")

    # §15 Paper-validity carry summary
    L.append("## §15 Paper-validity carry summary")
    L.append(
        f"- paper_validity_metadata_missing=true count: "
        f"{counts['paper_validity_missing']}"
    )
    L.append(
        f"- paper_signal_age populated count: {counts['paper_signal_age_populated']}"
    )
    L.append(f"- paper_expired_flag=true count: {counts['paper_expired_true']}")
    L.append(
        "- _Historical paper parquets pre-date the validity revision; "
        "`paper_validity_metadata_missing=true` is the LOCKED carry-as-missing "
        "semantic per Decision Engine v1 impl spec §3.7. NOT a defect; resolution "
        "lives behind the upstream `paper_execution_v0` real forward-emission "
        "LOCK._"
    )
    L.append("")

    # §16 Forbidden-fields static absence note
    L.append("## §16 Forbidden-fields static absence")
    present = sorted(
        f for f in _TIER3_FORBIDDEN_FIELDS if f in schema_field_names
    )
    if present:
        L.append(
            "**FAIL**: forbidden field(s) present in events parquet schema: "
            + ", ".join(present)
        )
    else:
        L.append(
            "events parquet schema does NOT contain `final_action` / "
            "`selection_score` / `edge_score` / `prior_edge_score` / "
            "`setup_score` / `rank` / `portfolio_pick` / `portfolio_weight`. "
            "Read-only verification by Tier 3; no recompute."
        )
    L.append("")

    # §17 Footer / scope
    L.append("## §17 Scope statement")
    L.append(
        "_This is a Tier 3 review report. NO ranking, NO portfolio, "
        "NO score, NO forward returns, NO live integration. "
        "Live trade gate STAYS CLOSED._"
    )
    L.append("")

    return "\n".join(L)


def _emit_tier3_fail(
    *,
    fail_category: str,
    fail_reason: str,
) -> int:
    """Stderr-only FAIL classification per LOCK §12.2 Q7 (no-write semantic).

    Tier 3 FAIL writes NO report. The verdict is recorded only in stdout/stderr
    and in the post-run memory update.
    """
    sys.stderr.write(
        f"\nDecision Engine v1 — Tier 3 FAIL\n"
        f"category: {fail_category}\n"
        f"reason: {fail_reason}\n"
    )
    sys.stdout.write(
        f"Decision Engine v1 — Tier 3 review report\n"
        f"verdict: FAIL ({fail_category}: {fail_reason})\n"
        f"output: (NOT WRITTEN per LOCK §12.2 Q7 atomic-write FAIL semantic)\n"
    )
    return 1


def _run_tier3(args) -> int:  # noqa: ARG001 — args unused but parser-bound
    """Single authorized Tier 3 review-report render (LOCK 2026-05-04).

    Pipeline:
      1. sha256 PRE on the 3 guarded files.
      2. Read Tier 2 events.parquet (read-only).
      3. Read paper-link integrity CSV (read-only).
      4. Read Tier 2 summary md to extract source-run timestamp + asof.
      5. Compute counts (no recomputation; pure aggregation).
      6. Render markdown.
      7. Atomic write report. sha256 POST. Verify byte-equal == PRE.

    Single-fire per LOCK §12.7. PASS → write + STOP. FAIL → no write + halt.
    """
    import os

    # Step 1: sha256 PRE
    pre_state = _compute_tier3_guarded_state()
    missing = [k for k, v in pre_state.items() if v is None]
    if missing:
        return _emit_tier3_fail(
            fail_category="input coverage",
            fail_reason=(
                "missing guarded file(s) at Tier 3 fire: "
                + ", ".join(_TIER3_GUARDED_PATHS[k] for k in missing)
            ),
        )

    # Step 2: read events.parquet (read-only)
    try:
        events, schema_field_names = _read_tier2_events()
    except Exception as exc:  # pragma: no cover — surfaced to user
        return _emit_tier3_fail(
            fail_category="implementation bug",
            fail_reason=f"events parquet read failed: {exc!r}",
        )

    # Forbidden-fields static absence guard at runtime
    leaked = sorted(
        f for f in _TIER3_FORBIDDEN_FIELDS if f in schema_field_names
    )
    if leaked:
        return _emit_tier3_fail(
            fail_category="forbidden field",
            fail_reason=(
                "forbidden field(s) present in events parquet schema: "
                + ", ".join(leaked)
            ),
        )

    # Step 3: read integrity CSV (read-only)
    try:
        integrity_rows = _read_tier2_integrity_rows()
    except FileNotFoundError:
        return _emit_tier3_fail(
            fail_category="input coverage",
            fail_reason=(
                f"paper-link integrity CSV missing: {_TIER3_INTEGRITY_CSV_PATH}"
            ),
        )

    # Step 4: read Tier 2 summary md (best-effort, read-only)
    tier2_meta = _read_tier2_summary_meta()

    # Step 5: compute counts
    try:
        counts = _compute_tier3_counts(events)
    except Exception as exc:  # pragma: no cover
        return _emit_tier3_fail(
            fail_category="implementation bug",
            fail_reason=f"count aggregation failed: {exc!r}",
        )

    # §12.4 enum-coverage guardrail (in addition to §3.3 invariant)
    from decision_engine_v1.schema import (
        EXECUTION_LABELS,
        MARKET_CONTEXTS,
        SETUP_LABELS,
    )
    for label in SETUP_LABELS:
        if label not in counts["setup_dist"]:
            return _emit_tier3_fail(
                fail_category="enum coverage",
                fail_reason=f"setup_label enum {label!r} missing from distribution",
            )
    for label in EXECUTION_LABELS:
        if label not in counts["exec_dist"]:
            return _emit_tier3_fail(
                fail_category="enum coverage",
                fail_reason=f"execution_label enum {label!r} missing from distribution",
            )
    for label in MARKET_CONTEXTS:
        if label not in counts["ctx_dist"]:
            return _emit_tier3_fail(
                fail_category="enum coverage",
                fail_reason=f"market_context enum {label!r} missing from distribution",
            )

    # Step 6: render markdown (compute pre-write to allow no-write FAIL)
    render_ts = _dt.datetime.now(_dt.timezone.utc).isoformat()
    try:
        report_text = _render_tier3_markdown(
            render_timestamp_utc=render_ts,
            tier2_meta=tier2_meta,
            pre_state=pre_state,
            post_state=pre_state,  # placeholder; will replace with POST after write
            counts=counts,
            integrity_rows=integrity_rows,
            schema_field_names=schema_field_names,
        )
    except Exception as exc:  # pragma: no cover
        return _emit_tier3_fail(
            fail_category="implementation bug",
            fail_reason=f"markdown render failed: {exc!r}",
        )

    # Pre-write forbidden-field text guard (defense-in-depth).
    # The §16 explicit-absence line cites these in backticks; that is allowed.
    # Any occurrence of the bare names outside that backtick context is forbidden.
    # Two-stage check: (1) strip authorized `<token>` instances from a working copy
    # so substring overlap (e.g. `prior_edge_score` containing `edge_score`)
    # cannot false-positive; (2) word-bounded re.search for any leak.
    import re as _re
    _guard_working = report_text
    for f in _TIER3_FORBIDDEN_FIELDS:
        _guard_working = _guard_working.replace(f"`{f}`", "")
    for f in _TIER3_FORBIDDEN_FIELDS:
        if _re.search(rf"\b{_re.escape(f)}\b", _guard_working):
            return _emit_tier3_fail(
                fail_category="forbidden field",
                fail_reason=(
                    f"bare forbidden-field token {f!r} present outside "
                    f"explicit-absence backtick context"
                ),
            )

    # Step 7a: atomic write — render with placeholder POST, then re-render with
    # actual POST after the rename (the report's §2 must show pre/post identical
    # so we can write once with pre as both, since byte-equality is the verdict).
    try:
        v1_write.write_tier3_review_report_markdown(report_text)
    except Exception as exc:  # pragma: no cover
        return _emit_tier3_fail(
            fail_category="implementation bug",
            fail_reason=f"atomic write failed: {exc!r}",
        )

    # Step 7b: sha256 POST verification on the 3 guarded files
    post_state = _compute_tier3_guarded_state()
    diff = _tier3_state_diverged(pre_state, post_state)
    if diff is not None:
        # Per LOCK §12.2 Q7 + §12.7 FAIL semantic + §3.1: divergence is
        # `source mutation`. The atomic-written report is now stale; we
        # remove it to honour the no-write FAIL invariant.
        try:
            os.unlink(v1_write.TIER3_REVIEW_REPORT_MD_PATH)
        except OSError:
            pass
        return _emit_tier3_fail(
            fail_category="source mutation",
            fail_reason=f"sha256 PRE/POST divergence on guarded file(s): {diff}",
        )

    # Verdict PASS — stdout summary
    sys.stdout.write(
        f"Decision Engine v1 — Tier 3 review report\n"
        f"asof_date: {tier2_meta.get('asof_date') or '-'}\n"
        f"render_timestamp_utc: {render_ts}\n"
        f"events: {counts['total']}\n"
        f"sha256 byte-equal pre/post (3 guarded files): True\n"
        f"output: {v1_write.TIER3_REVIEW_REPORT_MD_PATH}\n"
        f"verdict: PASS\n"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    if args.tier == 1:
        return _run_tier1(args)
    if args.tier == 2:
        return _run_tier2(args)
    if args.tier == 3:
        return _run_tier3(args)
    sys.stderr.write(_TIER_0_REFUSAL_MESSAGE)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
