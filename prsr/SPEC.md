# PRSR v1 — Price-Structure Reversal Engine

Pre-registration spec.

Status: **PRE-REGISTERED, NOT STARTED**
Date: 2026-04-30
Module: `prsr/`
Predecessor: OSCMTRX thread, CLOSED_REJECTED 2026-04-30
(`output/oscmatrix_final_closure_report.md`).

## Status discipline

This document is the locked pre-registration for PRSR v1. All gates,
parameters, and acceptance numbers below are FROZEN before any
backtest is run. Acceptance must not be relaxed post-hoc to make a
failing run pass; that pattern was the explicit failure mode of the
closed OSCMTRX thread.

If a number changes after run-1, that is a new spec version
(v1.1, v2, ...) and the original v1 verdict stands.

## Design principle

Phase 4A (OSCMTRX thread, REJECTED) put the oscillator at the centre
and let price be the consequence; that lost to a same-date random
baseline on the objective 607 panel. PRSR inverts:

> Price structure leads. Oscillator is a quality tag, never a gate.

Concretely: candidate = price-pattern-pass ∧ regime-pass ∧
universe-pass. HWO/HW confirmation is computed and reported as a
diagnostic subset only. The acceptance verdict is judged on the
all-candidates set; the oscillator-confirmed subset cannot rescue or
modify the primary verdict.

## Scope (v1)

In scope:
- Patterns A (failed breakdown) and B (spring / reclaim).
- Tradeable Core universe with IPO carve-out (this module only).
- Single-bar candidate definition + T+1 open entry.
- Initial stop + 10-bar time stop primary.

Out of scope (v1):
- Patterns C (higher-low) and D (reversal candle + location).
- ML scoring.
- TP-based primary (TP 2R is a side analysis only).
- HWO-Down primary exit (optional risk-reduction appendix only).
- close_T optimistic entry as primary (cannot rescue an open_T1 fail).
- BIST100-only universe (would be a separate pre-registered thread).

## Universe (this module only)

Three tiers, computed daily from rolling history:

| tier | rule | use |
|---|---|---|
| Tradeable Core | `history_bars ≥ 500` ∧ `median_turnover_60 ≥ cross-sectional q60` | primary scan |
| Watchable | `250 ≤ history_bars < 500` | watchlist CSV only, no signal / no trade |
| New IPO | `history_bars < 250` | excluded entirely |

Definitions:
- `history_bars` = number of daily bars available up to the evaluation
  date for the ticker.
- `median_turnover_60` = trailing 60 daily-bar median of
  `close × volume`, evaluated at T using bars t ∈ [T−60, T−1].
- `q60` = cross-sectional 60th percentile of `median_turnover_60`
  computed across all tickers with `history_bars ≥ 500` on that date.

This universe rule is local to PRSR. It does NOT propagate to
Scanner V1, CTE, nyxexp, nyxmomentum, or any other pipeline.

## Window

2023-01-02 → 2026-04-29. Same window as Phase 4A for direct
comparability.

## Pattern definitions (v1: A + B only)

Helpers:

- `min20[t]` = `min(low[t-20..t-1])` — rolling 20-bar minimum of `low`,
  excluding bar t to keep the support level free of the candidate
  bar's own data.
- `loc(t)` = `(close[t] − low[t]) / (high[t] − low[t])`. Defined as
  0.5 if `high[t] == low[t]`.
- `atr20[t]` = Wilder ATR over the trailing 20 bars.
- `vol_med60[t]` = `rolling_60_median(volume)` evaluated on bars
  t ∈ [t−60, t−1].

### Pattern A — Failed breakdown

All of:
- `low[t] < min20[t]` — intraday penetration of support.
- `close[t] ≥ min20[t]` — close reclaims support.
- `loc(t) ≥ 0.65` — close in upper third of bar range.
- `volume[t] ≥ vol_med60[t]` — non-trivial seller-rejection flow.

Pattern stop reference: `pattern_low_A = low[t]`.

### Pattern B — Spring / reclaim

All of:
- There exists `t' ∈ [t−5, t−1]` with `close[t'] < min20[t']`
  — the close went below support inside the last 5 bars.
- `close[t] ≥ min20[t]` — today reclaims support.
- `loc(t) ≥ 0.65`.
- `close[t] ≥ close[t−1]` — today is up-or-flat versus prior close.

Pattern stop reference: `pattern_low_B = min(low[t−5..t])`.

### Joint firing

Patterns A and B are not mutually exclusive. If both fire on the same
(ticker, date), the row is logged once with `pattern_kind = "both"`
and `pattern_low = min(pattern_low_A, pattern_low_B)`.

## Regime gate (cross-sectional, per evaluation date)

Both required at the candidate bar t:

- `ER60(t) ≥ q40` — `ER60` = 60-bar efficiency ratio
  `|close[t] − close[t−60]| / Σ |close[i] − close[i−1]|` over
  `i ∈ [t−59, t]`. q40 = 40th cross-sectional percentile across the
  Tradeable Core cohort on date t.
- `ATR%20(t) ≤ q90` — `ATR%20 = atr20 / close`. q90 = 90th
  cross-sectional percentile across the Tradeable Core cohort on
  date t.

Both quantiles are recomputed every date from that date's Tradeable
Core membership only.

## Candidate gate

```
candidate(t) = universe_pass(t)
             ∧ regime_pass(t)
             ∧ (pattern_A(t) ∨ pattern_B(t))
```

Oscillator is **not** in this gate.

## Oscillator confirmation tag (diagnostic only)

Computed on every candidate row but never gating:

```
osc_confirmed(t) =
    (HWO Up fired in [t-2, t]   ∨  HW(t) < 50)
  ∧ (slope(HW, last 3 bars) > 0  ∨  lower_zone(t))
```

`HWO Up`, `HW`, and `lower_zone` are computed locally inside
`prsr/oscillator.py` from the same daily panel — NOT imported from
the closed `oscmatrix/` module. The implementation may reference the
Phase 0.6.3 close-source HW formula but is owned by PRSR.

The `osc_confirmed` subset appears as a side report only. It cannot
modify, rescue, or replace the primary acceptance verdict.

## Trigger / entry

Primary:
- Entry timestamp: open of T+1.
- Entry price: `open[T+1]`.

Secondary (side analysis only, cannot pass acceptance alone):
- close_T optimistic: `close[T]`.

If T+1 is a non-trading day or the ticker is halted at T+1, the
candidate is dropped (no roll forward to T+2).

## Stop / exit

Primary mechanics:

- Initial stop: `pattern_low − 0.25 × atr20[T]`.
- Stop is checked intraday using `low[s]` for each subsequent bar s.
  If `low[s] ≤ stop`, the trade exits at `stop` price (assume stop is
  filled at the stop level; do not model slippage in v1).
- Time stop: 10 trading bars. If the stop is not hit by the close of
  T+10, exit at `close[T+10]`.
- R-multiple per trade:
  `R = (exit_price − entry_price) / (entry_price − initial_stop)`.

Side analyses (reported, not primary):
- 5-bar and 20-bar time-stop variants.
- TP 2R take-profit variant: exit at `entry_price + 2R` if reached
  before stop or time-stop.

Optional appendix (NOT in the primary backtest, separate report
section):
- HWO Down partial exit: 50% off when `HWO Down` triggers
  intra-trade. Reported as a risk-reduction diagnostic only.

## Same-date random baseline

For each PRSR primary candidate fire date, randomly sample one
(ticker) from the same Tradeable Core cohort that did NOT fire any
PRSR candidate that day. Apply identical entry / stop / time-stop
machinery. Use seed = 17.

Acceptance compares PRSR primary against this baseline on identical
metrics.

## Acceptance (LOCKED before run-1)

Judged on the open_T1 primary trade pool, all-candidates set
(oscillator-confirmed subset is diagnostic only).

All of the following must hold simultaneously:

| metric | threshold |
|---|---|
| PF_proxy | ≥ 1.30 AND ≥ random + 0.20 |
| realized_med | ≥ random + 0.5pp |
| yearly slices PF beating random | ≥ 3 of 4 |
| top-5 fire-date PnL share | ≤ 30% |
| N (primary trades) | ≥ 200 |
| MaxDD proxy | ≤ random's MaxDD proxy (i.e. no worse than random) |
| open_T1 primary verdict | must pass — close_T optimistic alone does not qualify |

Tie-breaking: thresholds are inclusive (`≥`, `≤` as written). Any
single failed criterion ⇒ verdict REJECTED. There is no partial pass.

## Reporting

Required outputs:

```
output/prsr_v1_per_trade.csv         per-(fire, K, entry_mode) row
output/prsr_v1_random_baseline.csv   same-date random trades
output/prsr_v1_aggregate.csv         overall + per-year aggregates
output/prsr_v1_concentration.csv     per-ticker, per-date fire counts
output/prsr_v1_drawdown.csv          chronological cumulative equity
output/prsr_v1_osc_subset.csv        diagnostic oscillator-confirmed subset
output/prsr_v1_watchable_list.csv    Watchable tier (250–499 bar) candidates, no trades
output/prsr_v1_report.md             pass/fail verdict against locked acceptance
```

The report MUST cite each acceptance row's value vs threshold and
declare PASS or REJECTED in a header line. No softening language.

## File layout (when implementation starts)

```
prsr/
  __init__.py
  SPEC.md           (this file)
  config.py         all numerical constants — read-only at runtime
  universe.py       tier classification + q60 liquidity
  patterns.py       A, B detectors
  regime.py         ER60 + ATR% cross-sectional cuts
  oscillator.py     HWO/HW feature wrapper for diagnostic tag (NO gating logic)
  trigger.py        T+1 open entry mechanics
  exits.py          initial stop, time stop, side variants
  backtest.py       orchestration, per-trade rows, drawdown
  random_baseline.py
tools/
  prsr_run_v1.py
```

`prsr/oscillator.py` MUST NOT expose any function that returns a
gating boolean for the primary candidate. It may only return the
`osc_confirmed` tag column and underlying HW/HWO feature columns
labelled as "diagnostic".

## Do-not list (locked at pre-registration)

- Do not add the oscillator to the candidate gate after seeing run-1
  results. If the diagnostic subset looks great, that is a v3
  hypothesis, not a v1 patch.
- Do not relax acceptance thresholds to make run-1 pass.
- Do not enable patterns C / D after run-1 to rescue verdict.
- Do not promote close_T optimistic to primary if open_T1 fails.
- Do not propagate the IPO universe rule outside `prsr/`.
- Do not call this engine a LuxAlgo / OSCMTRX clone in any artifact.
- Do not couple `prsr/` to `oscmatrix/`. Oscillator features are
  computed locally; OSCMTRX module stays archived.

## Allowed if v1 PASSES

- v1.1: enable patterns C and D, re-run with the same acceptance
  numbers (same window, same baseline).
- v2: optional ML scorer on top of the locked v1 candidate set.
- v3: oscillator-confirmed subset becomes its own pre-registered
  thread if the diagnostic subset is dramatically better than primary.

## Allowed if v1 FAILS

- v1 is REJECTED. Document and close, mirroring the OSCMTRX closure
  pattern.
- A new thread (PRSR v2 or a different module) may try a different
  hypothesis but must define its own pre-registered acceptance.
- Do not re-run v1 with adjusted parameters and call it v1 again.
