# Decision Engine v1 — Tier 2 label table run summary

- Run timestamp (UTC): 2026-06-05T09:40:19.603540+00:00
- asof_date (resolved): 2026-06-04
- panel max date: 2026-06-04
- Line E max asof_date: 2026-06-04
- Line TR max asof_date: 2026-06-02

## Inputs (read-only)
- `output/decision_v0_classification_panel.parquet`
- `output/paper_execution_v0_trades.parquet`
- `output/paper_execution_v0_trigger_retest_trades.parquet`

## Verdict
- step 1.0 live-scope: PASS
- step 5b paper-link integrity: PASS
- **verdict: PASS**

## Event count
- total events processed: 230

## Execution label distribution (descriptive)
- EXECUTABLE: 29
- PAPER_ONLY: 3
- SIZE_REDUCED: 24
- WAIT_BETTER_ENTRY: 50
- WAIT_RETEST: 124

## Paper-link integrity (Step 5b)
- EXTENDED: attempted=3, matched=3, unmatched=0, duplicate_error=0, skipped_source_missing=0, validity_missing=3
- TRIGGER_RETEST: attempted=0, matched=0, unmatched=0, duplicate_error=0, skipped_source_missing=0, validity_missing=0
- ALL: attempted=3, matched=3, unmatched=0, duplicate_error=0, skipped_source_missing=0, validity_missing=3

- paper_validity_metadata_missing total: 3

## Paper-stream parquet sha256 pre/post (LOCK §12.4)
- Line E pre:  size=423426, sha256=a755c74f2a084dbcf0e34bfc1016c11bac25d211ceee3cff633f34c70a89924d
- Line E post: size=423426, sha256=a755c74f2a084dbcf0e34bfc1016c11bac25d211ceee3cff633f34c70a89924d
- Line TR pre:  size=126276, sha256=e89ac85c2516dd1509266dad78bcbcc97df0439b11bfda6082092784da9a91aa
- Line TR post: size=126276, sha256=e89ac85c2516dd1509266dad78bcbcc97df0439b11bfda6082092784da9a91aa

## Authorized outputs (LOCK §12.3)
- `output/decision_engine_v1_events.parquet`
- `output/decision_engine_v1_paper_link_integrity.csv`
- `output/decision_engine_v1_tier2_label_distribution.csv`
- `output/decision_engine_v1_tier2_label_table_summary.md`

_Tier 2 label-table run; no forward returns, no PF/WR/meanR, no ranking, no portfolio, no live integration, no HTML, no events CSV mirror, no markdown rollup._
