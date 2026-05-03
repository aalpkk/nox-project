# portfolio_merge_paper — Run Summary

- Spec: `memory/portfolio_merge_paper_spec.md` (LOCKED 2026-05-04, ONAY 2026-05-04)
- Run timestamp: 2026-05-03T23:07:22.274053+00:00
- asof: 2026-05-04
- Mode: reporting-only merge of two LOCKED paper/shadow lines

## Source files (READ-ONLY)
- Line E trades: `output/paper_execution_v0_trades.parquet` (sha256 `b5e427d4fc4c…`)
- Line E manifest: `output/paper_execution_v0_manifest.json`
- Line TR trades: `output/paper_execution_v0_trigger_retest_trades.parquet` (sha256 `ccaaca6a6c0a…`)
- Line TR manifest: `output/paper_execution_v0_trigger_retest_manifest.json`

## Per-line manifest references (verbatim)
- Line E spec_status: LOCKED 2026-05-03, ONAY 2026-05-04
- Line TR spec_status: LOCKED 2026-05-04, ONAY 2026-05-04
- Line E HB sha256: `2eb8a9a5d68e…`
- Line TR HB sha256: `2eb8a9a5d68e…`
  - HB sha256 matches across both lines.

## Locked merge constants
- max_new_positions_per_day (capped view): 5
- dedup priority (operational, NOT alpha): TRIGGER_RETEST > EXTENDED
- capped selection: line priority then ascending alphabetical ticker
- raw view: union of both lines' raw_all_candidates, no cap, no dedup beyond exact dupes
- R drawdown vs % drawdown: reported under separate field names

## §4.1 Primary HEADLINE — `merged_raw_all_candidates`

- n_closed: **4610**
- n_open: 110
- avg_R_paper: **0.3171**
- median_R_paper: 0.1400
- avg_gross_return_pct: **0.0381**
- avg_net_return_pct: 0.0361
- win_rate: **0.5742**
- profit_factor: **2.2048**
- max_drawdown_R: -73.8673
- max_drawdown_pct_gross: -9.6075
- max_drawdown_pct_net: -10.4595
- pct_from_extended: 0.8696
- pct_from_trigger_retest: 0.1304

## All 5 slices (descriptive)

### line_e_only
- n_closed: 4009  |  n_open: 95  |  n_skipped: 4270
- avg_R_paper: 0.3162  |  win_rate: 0.5702  |  PF: 2.1612
- avg_gross_return_pct: 0.0373  |  avg_net_return_pct: 0.0353
- max_drawdown_R: -74.5908  |  max_drawdown_pct_gross: -10.0899  |  max_drawdown_pct_net: -10.8559
- forward_closed: 0  |  forward_open: 0  |  backfill_closed: 4009  |  backfill_open: 95

### line_tr_only
- n_closed: 601  |  n_open: 15  |  n_skipped: 1340
- avg_R_paper: 0.3229  |  win_rate: 0.6007  |  PF: 2.5347
- avg_gross_return_pct: 0.0435  |  avg_net_return_pct: 0.0415
- max_drawdown_R: -11.0518  |  max_drawdown_pct_gross: -1.5419  |  max_drawdown_pct_net: -1.5779
- forward_closed: 0  |  forward_open: 0  |  backfill_closed: 601  |  backfill_open: 15

### merged_raw_all_candidates
- n_closed: 4610  |  n_open: 110  |  n_skipped: 0
- avg_R_paper: 0.3171  |  win_rate: 0.5742  |  PF: 2.2048
- avg_gross_return_pct: 0.0381  |  avg_net_return_pct: 0.0361
- max_drawdown_R: -73.8673  |  max_drawdown_pct_gross: -9.6075  |  max_drawdown_pct_net: -10.4595
- forward_closed: 0  |  forward_open: 0  |  backfill_closed: 4610  |  backfill_open: 110

### merged_dedup_same_ticker_day
- n_closed: 4610  |  n_open: 110  |  n_skipped: 0
- avg_R_paper: 0.3171  |  win_rate: 0.5742  |  PF: 2.2048
- avg_gross_return_pct: 0.0381  |  avg_net_return_pct: 0.0361
- max_drawdown_R: -73.8673  |  max_drawdown_pct_gross: -9.6075  |  max_drawdown_pct_net: -10.4595
- forward_closed: 0  |  forward_open: 0  |  backfill_closed: 4610  |  backfill_open: 110

### merged_capped_portfolio
- n_closed: 2627  |  n_open: 35  |  n_skipped: 0
- avg_R_paper: 0.2969  |  win_rate: 0.5744  |  PF: 2.1160
- avg_gross_return_pct: 0.0377  |  avg_net_return_pct: 0.0357
- max_drawdown_R: -44.8111  |  max_drawdown_pct_gross: -4.8555  |  max_drawdown_pct_net: -4.9755
- forward_closed: 0  |  forward_open: 0  |  backfill_closed: 2627  |  backfill_open: 35

## Same-ticker / same-asof_date conflicts (raw view)

- conflict count: **0**

## View discipline (spec §4.4)

- **Primary headline = `merged_raw_all_candidates`.**
- `merged_dedup_same_ticker_day` and `merged_capped_portfolio` are **operational views only**.
- If capped or dedup beats raw: label as selection artifact, NOT alpha (spec §4.3 / §4.4).

## Promotion-floor status (spec §10) — descriptive only

- EXTENDED: forward_closed=0 (need ≥50), forward_months_elapsed=0.00 (need ≥6) → **NOT_MET**
- TRIGGER_RETEST: forward_closed=0 (need ≥50), forward_months_elapsed=0.00 (need ≥6) → **NOT_MET**
- MERGED: forward_closed=0 (need ≥50), forward_months_elapsed=0.00 (need ≥6) → **NOT_MET**

**This merged paper report does NOT unlock live trading.**
**Live trade gate STAYS CLOSED.** No promotion claim is made by this run.

## Forbidden interpretations (spec §12)

- ❌ Merged view is a live portfolio
- ❌ TRIGGER_RETEST should be preferred because it performed better
- ❌ EXTENDED should be dropped
- ❌ Capped portfolio performance proves alpha
- ❌ Same-ticker priority is alpha ranking
- ❌ This enables live trading
- ❌ This changes either line's eligibility rule
- ❌ Cross-line comparison ranks one line above the other
- ❌ Capped view is the better strategy

