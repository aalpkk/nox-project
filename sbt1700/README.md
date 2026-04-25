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

### Acceptance gate (PR-2)
- mean Spearman rho ≥ 0.10
- fold_rho_min ≥ 0
- top-decile PF_net ≥ 1.5
- top-decile avg_R_net > 0
- permutation p(avg_R) ≤ 0.05
- top-5 trades' total R ≤ 50% of cohort total R (small-N concentration check)
- Gate fail ⇒ no live cron.

## Module layout

| File | Role |
|---|---|
| `config.py` | All locked constants + `E3` params |
| `aggregator.py` | 15m bars → 17:00-truncated daily bars (16:45 cutoff) |
| `signal_seed.py` | Daily-only seed of (ticker, date) candidates for Matriks gap fetch |
| `signals.py` | 17:00 SBT signal detection (with prior squeeze prefix + impulse + vol pace + HTF gates) |
| `features.py` | 17:00-aware features (no lookahead; `*_prior` lookups index T-1) |
| `execution.py` | `simulate_e3()` (LONG, 1R TP, 0.3 ATR SL, 5-bar timeout, gross+net) |
| `labels.py` | Run E3 per row, emit label columns |
| `build_dataset.py` | End-to-end orchestration → `output/sbt_1700_dataset.parquet` |
| `validate_dataset.py` | Coverage / label / concentration markdown report |
| `train_ranker.py` | (PR-2 stub) LightGBM train |
| `eval_ranker.py` | (PR-2 stub) acceptance gate |
| `run_scan_1700.py` | (PR-3 stub) live cron |

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

## What this module does NOT do

- Does not consume `breakout_master`, `tavan_*`, `rally_*`, `ml_breakout_*`,
  or any bucket score field from the legacy pipeline.
- Does not patch the legacy `output/ml_dataset.parquet`.
- Does not modify `run_smart_breakout.py` or its workflow.
- Does not push to Telegram / GH Pages until the PR-2 gate is green.
