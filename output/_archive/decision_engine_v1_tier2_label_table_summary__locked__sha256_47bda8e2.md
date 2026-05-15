# Decision Engine v1 — Tier 2 label table run summary

- Run timestamp (UTC): 2026-05-14T23:18:20.253504+00:00
- asof_date (resolved): 2026-05-14
- panel max date: 2026-05-14
- Line E max asof_date: 2026-05-14
- Line TR max asof_date: 2026-05-14

## Inputs (read-only)
- `output/decision_v0_classification_panel.parquet`
- `output/paper_execution_v0_trades.parquet`
- `output/paper_execution_v0_trigger_retest_trades.parquet`

## Verdict
- step 1.0 live-scope: PASS
- step 5b paper-link integrity: PASS
- **verdict: PASS**

## Event count
- total events processed: 338

## Execution label distribution (descriptive)
- EXECUTABLE: 75
- PAPER_ONLY: 23
- SIZE_REDUCED: 64
- WAIT_BETTER_ENTRY: 118
- WAIT_RETEST: 58

## Paper-link integrity (Step 5b)
- EXTENDED: attempted=20, matched=20, unmatched=0, duplicate_error=0, skipped_source_missing=0, validity_missing=20
- TRIGGER_RETEST: attempted=3, matched=3, unmatched=0, duplicate_error=0, skipped_source_missing=0, validity_missing=3
- ALL: attempted=23, matched=23, unmatched=0, duplicate_error=0, skipped_source_missing=0, validity_missing=23

- paper_validity_metadata_missing total: 23

## Paper-stream parquet sha256 pre/post (LOCK §12.4)
- Line E pre:  size=408908, sha256=789be7a5ea2e8c63de037cb08e7a07964f047b1c6234b426a22d9e67704331e4
- Line E post: size=408908, sha256=789be7a5ea2e8c63de037cb08e7a07964f047b1c6234b426a22d9e67704331e4
- Line TR pre:  size=125716, sha256=1273aaee5a08e452b40a25bfccb591e71f313bd513c81be890b8ddf1afbc788d
- Line TR post: size=125716, sha256=1273aaee5a08e452b40a25bfccb591e71f313bd513c81be890b8ddf1afbc788d

## Authorized outputs (LOCK §12.3)
- `output/decision_engine_v1_events.parquet`
- `output/decision_engine_v1_paper_link_integrity.csv`
- `output/decision_engine_v1_tier2_label_distribution.csv`
- `output/decision_engine_v1_tier2_label_table_summary.md`

_Tier 2 label-table run; no forward returns, no PF/WR/meanR, no ranking, no portfolio, no live integration, no HTML, no events CSV mirror, no markdown rollup._
