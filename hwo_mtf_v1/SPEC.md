# HWO Multi-Timeframe Turning Point — Pre-Registered Research Thread v1

**Status:** DRAFT — pending user lock
**Date drafted:** 2026-05-01
**Thread name:** `hwo_mtf_v1`
**NOT a successor to OSCMTRX engine line** (that line is CLOSED_REJECTED on Phase 4A
single-TF evidence; see `memory/oscmatrix_thread_closed_rejected.md`). This thread
tests a genuinely new hypothesis — **multi-timeframe agreement** on big-dot
turning points — which was not evaluated in Phase 4A.

---

## 1. Awareness of prior closures (must be read before editing)

This thread MUST NOT silently rescue:

- AL_F6 single-TF daily engine — CLOSED_REJECTED (Phase 4A: random PF 1.57
  vs AL_F6 PF 0.93 on q60 Core, 6/6 PF comparisons lost).
- Reversal scanner family — CLOSED (`memory/baseline_decomp_v1_finding.md`:
  AL_F6 + PRSR v1 both selection-negative, rank_pct ≈ 0.48, KS p < 0.001).
- SAT short-entry — CLOSED, do not re-evaluate.

**Why this thread is NOT a rescue:**
Phase 4A tested AL_F6 = `HWO_up ∧ armed_recent10 ∧ refractory10` on **daily-only**
data. It did not test cross-timeframe confluence. The new hypothesis below
is structural (multi-TF agreement), not parametric (different threshold on the
same single-TF signal). If multi-TF confluence fails the same acceptance gate
that AL_F6 failed, this thread closes — no rescue, no retune.

---

## 2. Hypothesis

**H1**: HWO big-dot turning points (Oversold/Overbought, per LuxAlgo definition)
agreeing across two timeframes within a tight window provide a tradeable
long-side edge over the same-date random baseline on the BIST 607 q60 Core
universe at K=10 horizon, where the single-TF version does not.

**H0**: HWO multi-TF confluence has no edge over random at K=10 (same null
that AL_F6 failed under).

Falsifiable. Single primary trigger. Acceptance gates locked numerically (§ 7).

---

## 3. Signal mechanics (LOCKED)

Reference implementation: `oscmatrix/components/hyperwave.py` (already audited,
Phase 0.6 F1±1 = 0.82 vs TV).

```
HW = SMA( RSI(close, 7), 3 )
signal = SMA( HW, 3 )
hwo_up   = HW crosses ABOVE signal     (small + big bullish dots)
hwo_down = HW crosses BELOW signal     (small + big bearish dots)
os_hwo_up   = hwo_up   AND HW < 20     (BIG bullish dot — Oversold turning)
ob_hwo_down = hwo_down AND HW > 80     (BIG bearish dot — Overbought turning)
```

This thread uses `os_hwo_up` (big bullish only) for long entries.
`ob_hwo_down` is computed but **not traded** in this thread (short-entry
freeze, see §1).

Source = `close` (close-source variant; production HWO source per Phase 0.6.3).
DO NOT switch to `ohlc4` mid-thread.

---

## 4. Multi-timeframe data sources (LOCKED)

| TF | Source | Build rule |
|---|---|---|
| 1d | `prsr.universe.load_daily_panel()` (resamples 1h master → daily OHLCV by `ts_istanbul.normalize()`) | existing |
| 1w | resample 1d to W-FRI close (Mon-Fri group; ISO week) | new helper |
| 5h | `mb_scanner.resample.to_5h()` on `output/extfeed_intraday_1h_3y_master.parquet` (TV-canonical Istanbul session bars) | existing helper |

**5h session bar definition (LOCKED — TV-aligned):**
BIST trading session 09:00–18:59 TR. 5h aggregation produces **two bars per
trading day**:
- Bar A (morning, label 09:00): hourly closes 9, 10, 11, 12, 13 → aggregated.
  Includes opening-auction print at 09:55.
- Bar B (afternoon, label 14:00): hourly closes 14, 15, 16, 17, 18 → aggregated.
  Includes closing-auction (18:05) and closing-price trades (18:08–18:10).

Per-bin completeness rule: each 5h bin must have `n_bars == 5` (all five
underlying hourly bars present). Partial bins (`n_bars < 5`) are **dropped**;
no carry-forward, no synthetic fill. A trading day may yield 0, 1, or 2 clean
5h bars depending on hourly coverage. Universe membership (§5) is unaffected
by per-day completeness — exclusion is per-bin only.

`hyperwave.compute_hyperwave()` is applied independently on each TF panel.

---

## 5. Universe (LOCKED)

BIST 607 q60 Core, defined by `prsr.universe.build_universe()`:
- `history_bars >= 500`
- `median_turnover_60 >= q60_turnover[date]` (cross-sectional q60 within
  Core-eligible cohort, computed per date)
- IPO carve-out (< 250 bars) excluded entirely

Same universe Phase 4A and baseline_decomp used. **No alternate universe
is permitted** — no BIST100 substitute, no alpha-filtered subset, no
regime-conditional cohort. (See "Allowed future threads B" in
`memory/oscmatrix_thread_closed_rejected.md` if BIST100 is desired — it
must be its own pre-reg thread.)

---

## 6. Window + entry/horizon (LOCKED)

| param | value |
|---|---|
| `START_DATE` | 2023-01-02 |
| `END_DATE` | 2026-04-29 |
| Entry | next 1d open after fire (open_T1) |
| Horizons | K ∈ {5, 10, 20} trading days |
| Primary K | **K = 10** (matches baseline_decomp + Phase 4A primary) |
| Side | LONG only |

For TF ∈ {1w, 5h}, "fire" is mapped to the daily entry calendar by:
- 1w fire on week W → entry at first 1d open of week W+1
- 5h fire on day D bar B (B ∈ {AM=09:00, PM=14:00}) → entry at next 1d open
  after D (i.e. open of D+1). If both AM and PM bars on D fire, that counts
  as a single fire-day for the daily-entry calendar (deduplicated by date);
  the originating bar is recorded in `hwo_mtf_v1_per_trade.csv` for audit.

This eliminates intraday execution complexity in this thread; that is a
separate paper-trade thread if accepted.

---

## 7. Triggers (LOCKED — primary marked)

Five triggers; one primary; four secondary (reported, not gated).

| ID | Definition | Role |
|---|---|---|
| T_1d | `os_hwo_up_1d` fires on day D | secondary (sanity replicates AL_F6-like single-TF; expected to fail) |
| T_1w | `os_hwo_up_1w` fires on week W | secondary |
| T_5h | `os_hwo_up_5h` fires on day D bar B (AM or PM) | secondary |
| **T_1d∧5h** | T_1d fires on day D **AND** T_5h fires on D, D-1, or D-2 (≤ 3 trading-day window, both directions; either AM or PM bar qualifies) | **PRIMARY** |
| T_1d∧1w | T_1d fires on day D **AND** T_1w fires within current or prior week (≤ 5 trading-day window) | secondary |

Confluence directionality: both signals must be bullish big-dot (`os_hwo_up`).
A 5h `os_hwo_up` paired with a 1d `ob_hwo_down` does NOT constitute confluence.

Primary trigger (T_1d∧5h) is the verdict-bearing trigger. Secondary triggers
are diagnostic only — they CANNOT be promoted to primary if primary fails.

---

## 8. Random baseline (LOCKED)

Same-date same-N from q60 Core, matching `baseline_decomp_v1_run.py` /
Phase 4A construction:

- For each fire date D in primary trigger T_1d∧5h, sample N random tickers
  from `tier=='core'` membership on D (same calendar date)
- Same horizon K=10
- Same entry mechanics (open_T1)
- 100 random draws; report mean PF + 5th/95th percentile

Random seed: `0x4857_4F4D_5446_3031` (i.e. ASCII "HWO MTF01" packed) —
disclosed up front, single seed, no seed sweep.

Reproducibility: seed locked, draw count locked, no stratification beyond
same-date same-tier.

---

## 9. Acceptance gates (LOCKED, primary trigger only)

All four gates must pass simultaneously for ACCEPTED verdict:

| gate | metric | threshold | rationale |
|---|---|---|---|
| **G1 — PF lift** | `PF_T_1d∧5h_K10` | ≥ 1.87 (= random PF 1.57 + 0.30 absolute lift) | must beat the baseline that beat AL_F6 by a non-trivial margin |
| **G2 — Selection-residual** | mean `rank_pct_K10` within same-date Core cohort | ≥ 0.52 | prior selection-negative finding was 0.48; 0.52 = symmetric flip across 0.50 |
| **G3 — KS vs uniform** | KS test p-value of rank_pct distribution | < 0.05 (rejects uniform) | distribution-shift evidence beyond mean |
| **G4 — Reach** | event count | ≥ 30 fires across window | underpowered below this |

All four gates → **CLOSED_ACCEPTED** (this thread; not production).
Any gate fails → **CLOSED_REJECTED**, full stop.

PF computed identically to baseline_decomp:
`pf_proxy = sum(positive returns) / |sum(negative returns)|`.

---

## 10. Reporting (locked output set)

A single execution writes:

- `output/hwo_mtf_v1_report.md` — verdict, all 5 trigger results, gates evaluation
- `output/hwo_mtf_v1_per_trade.csv` — one row per (trigger, fire_date, ticker)
  with `entry_open, exit_close_K, ret_open_K, rank_pct_K, ew_core_K`
- `output/hwo_mtf_v1_random_baseline.csv` — random PF distribution per K
- `output/hwo_mtf_v1_aggregate.csv` — PF + KS + rank_pct mean per trigger × K

No HTML, no live cron, no ML — pure backtest report.

---

## 11. Single authorized run (LOCKED)

- One execution of `tools/hwo_mtf_v1_run.py` with locked params.
- If implementation bug (numerical assertion / coverage failure) is found,
  ONE rerun is allowed after explicit fix; the fix must be documented in
  the SPEC's "Implementation log" section (added post-hoc only for bugs,
  not tuning).
- After successful run, **no parameter sweep** (no threshold tweak, no
  trigger-window adjustment, no horizon swap, no universe carve-out).
- Acceptance verdict is final.

---

## 12. Closure rules

**CLOSED_ACCEPTED outcome** (all four gates pass):
- Verdict written to `output/hwo_mtf_v1_verdict.json`
- This thread closes; production / paper-trade is a SEPARATE thread
- A successor thread for paper-trade execution viability (intraday timing,
  borrow / reachability for shorts, slippage, etc) would have to be
  pre-registered separately

**CLOSED_REJECTED outcome** (any gate fails):
- Verdict written to `output/hwo_mtf_v1_verdict.json`
- This thread closes; **no rescue allowed**
  - No "what if K=20 instead of 10"
  - No "what if 5h confluence window is 5 days instead of 3"
  - No "what if we add regime filter"
  - No "what if we drop early dates from window"
- Memory updated to reflect closure (joins existing reversal-family closures).
- A wholly different hypothesis (e.g., HWO + structure feature, HWO as
  exit-only signal, etc.) is permitted in a NEW pre-reg thread.

---

## 13. Implementation conditions

- **Random seed disclosure** — seed printed in report header, traceable.
- **Coverage check** — for each TF, panel coverage % per ticker per year
  reported; if < 80% on any year, thread cannot run (blocks underpowered runs).
- **Cross-TF timing audit** — primary trigger T_1d∧5h pairs must be
  reproducible from raw data; manifest writes the exact (1d_fire_date,
  5h_fire_date, 5h_session_bar ∈ {AM, PM}) tuples.
- **No look-ahead** — `os_hwo_up` is computed bar-by-bar; HW value at fire
  bar uses only data through fire bar close.
- **5h session-bin exclusion** — bins with `n_bars != 5` dropped per
  ticker per bin (LOCKED rule §4).
- **No regime filter** — RDP / SMA / liquidity quantile gates are NOT
  applied beyond q60 Core membership. Regime gating is a separate thread.

---

## 14. What is OUT OF SCOPE (locked)

- Short entries (`ob_hwo_down`) — computed for diagnostic but not traded
- Other timeframes (5m, 15m, 30m, 2h, 4h, monthly)
- Other big-dot thresholds (HW < 25 or HW < 15 instead of 20)
- Other K horizons beyond {5, 10, 20}
- Custom universes (BIST100, sector subsets, watchable tier)
- ML scorer on top of HWO events
- Live execution / paper trade
- Re-evaluation of AL_F6 / PRSR v1 / OSCMTRX engine

Each of these is its own pre-reg thread if pursued.

---

## 15. Sign-off block

```
Approved by:        Alp Karakasli (akarakasli.289@gmail.com)
Approval ack:       "tamam onayladım" — session 2026-05-01
Date approved:      2026-05-01
Locked params hash: b8f42e1642652ec5b7efde400268b274d5ee75272330fd8934a6882873d0e9cc
                    (sha256 of SPEC.md at moment of approval, before this
                    sign-off block was filled in)
Run command:        python -m tools.hwo_mtf_v1_run
```

**LOCKED.** Post-approval edits to §1–§14 are forbidden; if a numerical bug
or implementation gap is discovered during the single authorized run, fix
it under §11's documented one-rerun allowance and log the change in an
"Implementation log" appendix below — never by silently editing the locked
sections.

---

## 16. Implementation log (§11 rerun allowance)

| # | When | Issue | Fix | Impact on verdict |
|---|---|---|---|---|
| 1 | run-1, post-aggregation | `json.dumps(asdict(verdict))` raised `TypeError: Object of type bool_ is not JSON serializable`. Numerical aggregation completed normally; only the verdict JSON write failed. | Cast each gate flag explicitly to Python `bool` in `evaluate_gates()` (4 lines, semantics unchanged). | None. All numerical metrics — PF, rank_pct, KS p, N, random baseline — were computed and printed pre-crash. The fix only enables JSON serialization. |

