# Core q60 Random Rebound Baseline — Decomposition Analysis

Pre-registration spec.

Status: **PRE-REGISTERED, NOT STARTED**
Date: 2026-04-30
Module: `baseline_decomp/`
Predecessors:
- OSCMTRX thread CLOSED_REJECTED (`memory/oscmatrix_thread_closed_rejected.md`)
- PRSR v1 CLOSED_REJECTED (`memory/prsr_v1_closed_rejected.md`)
- BIST 607 q60 random floor (`memory/bist607_q60_random_floor.md`)

## Purpose discipline

This is a **diagnostic**, not a strategy. The goal is to understand why
the same-date same-N random baseline on the q60 Core universe sits at
PF ≈ 1.57 and why two independent reversal engines (AL_F6, PRSR A+B)
failed to clear it. No new entry rule, scoring layer, or universe is
introduced inside this thread.

The output is a report that answers six pre-registered questions. The
report's conclusion drives the choice of the *next* pre-registered
thread (see Decision Tree). It does NOT propose, tune, or rescue a
signal inside this thread.

## Status discipline

All inputs, methods, and decision rules below are FROZEN before any
analysis is run. If a number changes after the run, that is a new spec
version (v1.1, v2, …) and the original v1 verdict stands.

## Inputs (locked)

### Universe and window

- Universe: BIST 607-panel q60 Core
  (`history_bars ≥ 500 ∧ median_turnover_60 ≥ cross-sectional q60`),
  recomputed per date.
- Window: 2023-01-02 → 2026-04-29.
- Random seed: 17.
- Daily panel source: `output/extfeed_intraday_1h_3y_master.parquet`
  aggregated to daily via `prsr.universe.load_daily_panel`.

### Signal artifacts (read-only)

| signal | per-trade file | random baseline file |
|---|---|---|
| AL_F6 (OSCMTRX Phase 4A) | `output/oscmatrix_phase4a_per_trade.csv` | `output/oscmatrix_phase4a_random_baseline.csv` |
| PRSR v1 | `output/prsr_v1_per_trade.csv` (filter `source=='PRSR'`) | `output/prsr_v1_random_baseline.csv` (or `output/prsr_v1_per_trade.csv` filter `source=='RANDOM'`) |

Headline numbers being decomposed (LOCKED reference, do not recompute
or re-derive inside this thread):

| thread | random PF (open_T1, K=10 / 10-bar time-stop) | signal PF | N signal | N random |
|---|---|---|---|---|
| OSCMTRX Phase 4A | 1.57 | 0.93 | 2,260 fires | 2,205 baseline |
| PRSR v1          | 1.568 | 1.065 | 1,549 | 1,541 |

Both decompositions run side-by-side; conclusions must hold on both
to be called a finding.

### Reference series

- XU100 daily: `output/xu100_extfeed_daily.parquet`.
- Equal-weight Core daily return: per-date arithmetic mean of
  `close.pct_change()` across the q60 Core universe membership *as of
  that date*. Computed once, cached to
  `output/baseline_decomp_ew_core_daily.parquet`.
- Cross-sectional same-date return distribution: per fire date, the
  full vector of next-K-bar returns for every q60 Core ticker
  eligible that date. Used for rank-percentile attribution.

K is fixed at the value already used by each thread (K=10 for AL_F6,
10-bar time stop for PRSR primary). Entry is T+1 open. No alternative
horizons or entries are evaluated in this thread.

## Six pre-registered questions

Each question is answered by a deterministic computation defined in
the next section. The report must state, per question, the numeric
answer, the artifact used, and the verdict bucket from the Decision
Tree.

1. Random PF 1.57 hangi tarihlerden geliyor?
   → date-attribution Lorenz curve of random PnL.
2. Bu tarihler piyasa-wide rebound günleri mi?
   → XU100 and equal-weight Core return on/around those dates.
3. AL_F6 / PRSR sinyalleri bu rebound günlerinde cluster mı?
   → fire-date overlap and lift vs random fire dates.
4. Sinyaller random'dan kötü çünkü yanlış hisse mi seçiyor, yoksa
   entry timing mi kötü?
   → same-date cross-sectional rank of selected tickers.
5. Equal-weight core rebound stratejisi zaten sinyal stratejilerinden
   iyi mi?
   → mechanical EW-core "rebound" comparison (definition below).
6. Alpha hisse seçiminde mi, market-regime/timing tarafında mı?
   → additive decomposition: per-trade return = same-date EW-core
   return + selection residual.

## Methods (exact)

All computations are run twice — once on AL_F6, once on PRSR v1 — and
reported side-by-side. "Random" refers to each thread's own same-date
same-N random baseline file.

### M1. Date attribution (Q1)

For random baseline:
- Group by `date` (or `fire_date` for PRSR).
- Compute per-date sum of `realized_pct` (OSCMTRX) or `ret_primary`
  (PRSR), filtered to `entry_mode ∈ {open_T1}` and the K matching the
  headline (K=10 / time-stop primary).
- Sort dates by per-date PnL descending. Report:
  - top 5, 10, 20 dates and their share of total PnL,
  - cumulative PnL share at the top decile, top quintile of dates,
  - Gini coefficient of per-date PnL.

Same computation repeated for AL_F6 / PRSR signal trades.

### M2. Rebound classification (Q2)

For each calendar date in the window:
- Compute `xu100_ret_d` = XU100 close-to-close return that date.
- Compute `ew_core_ret_d` = equal-weight Core close-to-close return.
- Compute `prior_dd_5d` = `(close_d − max(close_{d-5..d-1})) /
  max(close_{d-5..d-1})` for both XU100 and EW Core.

Define **rebound day** (frozen, applied identically to all dates):
- `ew_core_ret_d ≥ +1.0%` AND `prior_dd_5d ≤ −2.0%` on EW Core.

The threshold pair is locked. No tuning.

For Q2: take the top-20 random-PnL dates from M1; report what fraction
are rebound days, the median `ew_core_ret_d` on those dates, and the
median `xu100_ret_d`. Also report the unconditional base rate of
rebound days in the window for reference.

### M3. Signal clustering on rebound days (Q3)

- Total rebound-day count R, total non-rebound-day count NR.
- Signal fires per-date for AL_F6 and PRSR.
- Random fires per-date for both.
- Compute lift:
  `lift_signal_rebound = (signal_fires_on_rebound / R) /
                          (signal_fires_total / (R + NR))`
  and the equivalent for random.
- Report 2×2 contingency table (rebound × signal-fired) with χ²
  p-value. Verdict bucket: cluster-positive if lift ≥ 1.5 with χ² p
  < 0.05; cluster-negative otherwise.

### M4. Cross-sectional rank of selected tickers (Q4)

For each signal fire `(ticker, date)`:
- Build the same-date eligible-Core return vector for the same K-bar
  forward horizon: `R_d = {ret_K(t, d) : t ∈ Core(d)}`.
- Compute `rank_pct(t, d)` = percentile of selected ticker's
  `ret_K(t, d)` within `R_d` (1.0 = best).

Report:
- distribution of `rank_pct` for AL_F6 fires, PRSR fires, random
  fires (sanity check: random should be uniform [0,1] with mean 0.5),
- mean and median rank_pct, KS test against uniform,
- breakdown by rebound vs non-rebound day,
- breakdown by year.

Verdict buckets:
- **selection-negative**: signal mean rank_pct < 0.50 with KS p < 0.05.
- **selection-flat**: signal mean rank_pct ∈ [0.48, 0.52], KS p ≥ 0.05.
- **selection-positive**: signal mean rank_pct ≥ 0.55 with KS p
  < 0.05. (Bar set above random's expected 0.5 and the −1pp PRSR shortfall.)

### M5. Mechanical EW-core rebound strategy (Q5)

A mechanical, parameter-free comparison. Trigger on date `d`:
- `ew_core_ret_d ≥ +1.0%` AND `prior_dd_5d ≤ −2.0%` (same rebound
  definition as M2).
- Buy equal-weight basket of all Core tickers at T+1 open, hold K
  bars (matches the signal horizon), exit at K-bar close.

Compute PF, win-rate, MaxDD, mean trade, total return, N. Report
side-by-side with AL_F6 and PRSR primaries. No tuning of the rebound
threshold inside this thread.

### M6. Additive return decomposition (Q6)

For each signal trade, decompose the realized return:

```
ret_signal(t, d) = ret_ewcore(d, K)  +  selection_residual(t, d)
                   ^^^^^^^^^^^^^^^^^    ^^^^^^^^^^^^^^^^^^^^^^^^^
                   market/timing leg    selection leg
```

where `ret_ewcore(d, K)` = equal-weight Core return over the same
[T+1 open, T+K close] window the trade actually used.

Report:
- mean and sum of each leg, separately for AL_F6, PRSR, RANDOM,
- PF computed on each leg alone (selection-only PF, market-only PF),
- correlation of `ret_signal` with `ret_ewcore` per signal,
- per-rebound-day vs non-rebound-day split.

Verdict bucket:
- **timing-dominated**: |market_leg sum| ≥ 2 × |selection_leg sum|
  on both signals, and selection_leg PF < 1.10 on both.
- **selection-positive on subset**: selection_leg PF ≥ 1.30 on at
  least one bucket (year, rebound-day, regime tag).
- **null**: neither leg shows a structured edge.

## Implementation conditions (locked)

1. **Random baseline disclosure.** The report must state explicitly,
   for each thread, the random-baseline seed (17) and the draw count
   per fire (1-per-fire same-date sample) used to produce the input
   `*_random_baseline.csv` file. These numbers are taken from the
   producing thread's documentation, not re-derived here.

2. **XU100 graceful degradation.** If `output/xu100_extfeed_daily.parquet`
   is missing, has rows outside the [2023-01-02, 2026-04-29] window,
   or has more than 5 missing trading days inside the window, XU100
   becomes an auxiliary diagnostic only. The rebound classification
   (M2) and every primary verdict in the Decision Tree are then
   driven by EW Core alone. The report must call out this fallback in
   a single line at the top of the M2 section.

3. **Cross-signal conflict handling.** A result is called a "finding"
   only when AL_F6 and PRSR agree on the verdict bucket for that
   question. If they disagree (e.g. AL_F6 selection-flat, PRSR
   selection-negative), the result is labelled **signal-specific**
   and is reported but does NOT drive the Decision Tree. The Decision
   Tree fires only on findings that hold on both signals. Partial-fit
   default to **early-reversal scanner family CLOSED** is preserved.

## Outputs

Single report `output/baseline_decomp_v1_report.md` with sections
mapped 1:1 to the six questions, plus appendix tables:

- `output/baseline_decomp_v1_date_attribution.csv` — per-date PnL,
  rebound flag, signal fire counts (AL_F6, PRSR, random).
- `output/baseline_decomp_v1_rank_pct.csv` — per-trade `rank_pct`
  for all three pools (AL_F6, PRSR, RANDOM).
- `output/baseline_decomp_v1_decomposition.csv` — per-trade market
  leg, selection leg.
- `output/baseline_decomp_v1_ew_core_strategy.csv` — mechanical EW
  rebound trade list.
- `output/baseline_decomp_ew_core_daily.parquet` — cached EW-core
  daily-return series.

Tool: `tools/baseline_decomp_v1_run.py`, single entry point. No
config knobs — every threshold above is hard-coded from this spec.

## Decision tree (locked)

After the report is produced, the verdict on the six questions selects
the *next* pre-registered thread. This is the only decision this
thread makes. No retroactive editing of acceptance.

| pattern in report | next thread (separate pre-reg) |
|---|---|
| Q2 rebound-day fraction in top-20 ≥ 0.60 AND Q6 = timing-dominated | **market-regime/timing model** thread |
| Q4 verdict varies by year/regime (selection-positive on a subset) AND Q3 cluster-positive | **date/regime conditional filter** thread |
| Q4 verdict differs by ticker subgroup (e.g. mega-cap, low-vol) reproducibly across both signals | **cross-sectional ranker** thread |
| no bucket distinguishes signal from random (Q4 = selection-flat AND Q6 = null) | **early-reversal scanner family CLOSED** |

If the report's signature does not match any row exactly, default to
"reversal scanner family CLOSED" — partial fits are not promoted into
new threads inside this spec.

## Acceptance (for the diagnostic itself)

The thread is "complete" when:

- All six questions have a numeric answer in the report.
- Both signals (AL_F6, PRSR) are reported side-by-side for every
  question.
- Random rank_pct distribution is verified ≈ uniform on [0, 1] (KS p
  ≥ 0.05) — sanity check the random baseline.
- The decision-tree row chosen is named explicitly.

The thread does NOT have a PF acceptance threshold; it is not a
strategy.

## Do-not list (locked)

- Do not propose a new entry rule, exit rule, or universe inside this
  thread.
- Do not relax the rebound definition (`+1.0%` / `−2.0% / 5d`) or
  the rank_pct verdict thresholds after seeing the data.
- Do not add new horizons (K ≠ 10) or entries (close_T, open_T2).
- Do not introduce a new signal source. Only AL_F6 and PRSR per-trade
  files are read.
- Do not rerun OSCMTRX or PRSR backtests. Their per-trade CSVs are
  read-only inputs.
- Do not call any of the "next thread" buckets a recommendation
  before the report exists.
