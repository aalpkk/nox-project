# SBT-1700 — Smart Breakout Targets at 17:00 TR

Independent of the legacy SBT ML / bucket system. Trains a fresh ranker
on a 17:00-truncated feature set with realized-R labels from a frozen
E3 execution rule.

## Locked decisions (do not silently change — bump SCHEMA_VERSION instead)

### Entry semantic
- BIST session 17:00 TR snapshot.
- Cutoff = last 15m bar whose **close timestamp ≤ 16:45 TR**. The 16:45–17:00
  bar is excluded because it may not be finalized when the live cron fires.
- Live cron target: 14:01 UTC = 17:01 TR (Mon-Fri) — bar finalization
  buffer ensures the 16:45 close is already in cache.
- Expected bars per (ticker, signal_date) = **27** (closes 10:15..16:45).
- The daily OHLCV row of T is patched: `Open=first 15m open`,
  `High=max`, `Low=min`, `Close=last 15m close ≤ 16:45`, `Volume=sum`.
- T's EOD daily close is **never** consumed — neither for features nor labels.

### Coverage policy
- `intraday_coverage = n_bars / 27`.
- Drop train/eval rows with coverage < 0.80.
- Keep 0.80 ≤ coverage < 0.95 with diagnostics
  (`intraday_coverage`, `missing_bar_count`, `n_bars_1700`,
  `last_bar_ts_tr`).
- Validation report includes drop counts.

### Exit rule (E3, frozen)
- LONG only.
- Entry: `close_1700`.
- Stop: `entry - 0.30 × atr_1700`, where `atr_1700 = ATR(14)` on
  EOD-complete prior daily bars.
- Take-profit: `entry + 1.0 × R` (`R = entry - stop`).
- Timeout: 5 daily bars (exit at the close of the 5th bar).
- Failed-breakout, trailing, partial-exit: **all OFF** in v1.
- Same-bar TP+SL tag: assume worst (SL fills first).
- Slippage + commission: 5 bps each, per side. Reported as `cost_R`,
  with `realized_R_gross` and `realized_R_net` both kept.
- Forward window starts at T+1 (T's daily close is forbidden).

### Labels emitted
`realized_R_gross`, `realized_R_net`, `win_label = R_net > 0`,
`tp_hit`, `sl_hit`, `timeout_hit`, `exit_reason`, `bars_held`,
`entry_px`, `stop_px`, `tp_px`, `atr_1700`, `initial_R_price`,
`exit_px`, `exit_date`, `cost_R`.

### Model
- LightGBM regression, target = `realized_R_net`.
- Optional comparison: LightGBM classifier, target = `win_label`.
- Production rank score (PR-2) = regression score.
- **No ensemble with legacy SBT ML.** **No bucket inputs.**

### RESET methodology (PR-2 superseded — see Reset section)

> **Previous PR-2 E7 result was contaminated by same-dataset exit selection
> and is excluded from this reset.**

The original PR-2 acceptance gate has been retired. Exit selection,
model selection, and ranker evaluation are now strictly separated by
phase, with the test slice locked behind an explicit unlock flag.
See the **Reset pipeline** section below.

## Module layout

| File | Role |
|---|---|
| `config.py` | All locked constants + `E3` params |
| `aggregator.py` | 15m bars → 17:00-truncated daily bars (16:45 cutoff) |
| `signal_seed.py` | Daily-only seed of (ticker, date) candidates for Matriks gap fetch |
| `signals.py` | 17:00 SBT signal detection (with prior squeeze prefix + impulse + vol pace + HTF gates) |
| `features.py` | 17:00-aware features (no lookahead; `*_prior` lookups index T-1) |
| `execution.py` | `simulate_e3()` (LONG, 1R TP, 0.3 ATR SL, 5-bar timeout, gross+net) — used by PR-1 dataset builder |
| `exits.py` | Multi-exit simulator (E3..E7 family) — used by reset pipeline |
| `labels.py` | Run E3 per row, emit label columns (PR-1 dataset) |
| `build_dataset.py` | End-to-end orchestration → `output/sbt_1700_dataset.parquet` |
| `validate_dataset.py` | Coverage / label / concentration markdown report |
| `splits.py` | RESET: locked train/val/test bounds + test-period lock guard |
| `train_ranker.py` | RESET: locked LightGBM regression, blacklist enforcement, walk-forward |
| `eval_ranker.py` | RESET: phase-aware cohort + ranker metrics |
| `reset_pipeline.py` | RESET: phase orchestrator (manifest / discovery / validation / final_test / report) |
| `run_scan_1700.py` | (PR-3 stub) live cron — only after a final_test pass + explicit go-decision |

External tools:
- `tools/sbt1700_fetch_intraday.py` — gap-only Matriks 15m fetcher;
  appends to `output/sbt1700_intraday_15m.parquet`.

## Build flow (PR-1)

```bash
# 1. Seed (daily-only): identify candidate (ticker, date) pairs to fetch.
python -m sbt1700.signal_seed \
    --master output/ohlcv_10y_fintables_master.parquet \
    --start 2023-11-15 \
    --out output/sbt1700_signal_seed.csv

# 2. Gap-only Matriks fetch (skips pairs already in nyxexp / sbt1700 caches).
MATRIKS_API_KEY=… python tools/sbt1700_fetch_intraday.py

# 3. Build dataset + validation report.
python -m sbt1700.build_dataset \
    --master output/ohlcv_10y_fintables_master.parquet \
    --intraday output/nyxexp_intraday_15m_matriks.parquet \
               output/sbt1700_intraday_15m.parquet \
    --start 2023-11-15 \
    --out output/sbt_1700_dataset.parquet \
    --validate
```

The GitHub Actions workflow (`.github/workflows/sbt1700-build-dataset.yml`)
runs all three steps on workflow_dispatch.

## Reset pipeline (PR-3)

> **Previous PR-2 E7 result was contaminated by same-dataset exit
> selection and is excluded from this reset.**

The original PR-2 ran exit selection and ranker evaluation on the same
2024–2026 cohort, so the "E7 unconditional + rankable" verdict cannot
be treated as out-of-sample evidence. The reset rebuilds the
methodology from scratch with an explicit train / validation / test
split and a lock that prevents the test slice from being read until a
final, one-shot readout.

### Locked split (do not silently change)

| Phase | Date range | Use |
|---|---|---|
| train | 2024-01-16 → 2025-06-30 | All discovery: exit family scan, feature exploration, ranker walk-forward |
| validation | 2025-07-01 → 2025-12-31 | At most 2 carried exits + locked model; freezes one final exit |
| test | 2026-01-01 → 2026-04-24 | One-shot readout. Locked behind `--unlock-test` |

`sbt1700/splits.py::load_split(phase="test")` raises `TestLockError`
unless `allow_test=True` is passed. The CI guard
(`tests/sbt1700/test_splits_lock.py`) verifies this so the lock cannot
regress silently. The reset workflow refuses `phase=final_test`
without `unlock_test=true`.

### Phase semantics

- **Discovery pass** ≠ validated. Discovery may surface candidates only.
- **Validation pass** ≠ production. Validation freezes one exit + one model.
- **Test pass** = paper candidate only. Live deployment requires a
  separate go-decision and a forward paper-traded ledger built after
  this reset.

### Hard rules

- The test slice is never accessed during discovery or validation.
- Hyperparameters are locked in `train_ranker.LGBM_REGRESSION_PARAMS`.
  No sweep, no random search, no per-fold tuning.
- Feature blacklist (`train_ranker.FEATURE_BLACKLIST`) drops all
  label-derived columns, identifiers, raw date strings, and per-trade
  parameters (entry_px / stop_px / tp_px / atr_1700) before fitting.
- Re-running validation does *not* re-search exits. If discovery
  promotes E5 and E7, validation runs only those two and freezes one.

### Run flow

```bash
# 1. Build dataset under the reset filename (re-run PR-1 builder, new path).
python -m sbt1700.build_dataset \
    --master output/ohlcv_10y_fintables_master.parquet \
    --intraday output/nyxexp_intraday_15m_matriks.parquet \
               output/sbt1700_intraday_15m.parquet \
    --start 2023-11-15 \
    --out output/sbt_1700_dataset_reset.parquet

# 2. Manifest — record split bounds + dataset fingerprint.
python -m sbt1700.reset_pipeline manifest

# 3. Discovery — exit matrix + per-exit ranker WF on TRAIN only.
python -m sbt1700.reset_pipeline discovery

# 4. Validation — pick ≤2 carried exits, freeze one.
python -m sbt1700.reset_pipeline validation --carry E5_symmetric,E7_partial

# 5. Final test — ONE-SHOT readout, requires --unlock-test.
python -m sbt1700.reset_pipeline final_test --exit E7_partial --unlock-test

# 6. Report — synthesize sbt_1700_reset_report.md.
python -m sbt1700.reset_pipeline report
```

The same phases are wired into
`.github/workflows/sbt1700-reset.yml` (workflow_dispatch). The workflow
runs the lock-guard pytest before every phase and refuses
`phase=final_test` unless `unlock_test=true` is set on the dispatch.

### Reset outputs

- `output/sbt_1700_reset_split_manifest.json` — split bounds + dataset SHA-256 prefix
- `output/sbt_1700_exit_discovery_train.csv` — raw cohort × 5 exits on TRAIN
- `output/sbt_1700_ranker_discovery_train.csv` — per-exit walk-forward (TRAIN internal)
- `output/sbt_1700_exit_validation.csv` — raw cohort × carried exits on VAL
- `output/sbt_1700_ranker_validation.csv` — TRAIN-fit, VAL-applied ranker metrics
- `output/sbt_1700_final_test_LOCKED.csv` / `.json` / `_trades.csv` — one-shot readout
- `output/sbt_1700_reset_report.md` — synthesized report

## What this module does NOT do

- Does not consume `breakout_master`, `tavan_*`, `rally_*`, `ml_breakout_*`,
  or any bucket score field from the legacy pipeline.
- Does not patch the legacy `output/ml_dataset.parquet`.
- Does not modify `run_smart_breakout.py` or its workflow.
- Does not enable any live cron from the reset PR. Promotion to
  `run_scan_1700.py` requires an explicit go-decision after the
  final-test readout, never automatically.
