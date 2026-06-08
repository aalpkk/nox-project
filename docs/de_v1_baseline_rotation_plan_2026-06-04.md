# Decision Engine v1 — Baseline-Lock Rotation Plan (04-29 → 06-04)

**Status:** ✅ EXECUTED 2026-06-05 with ONAY. All gates PASS. Validation receipt in §8. NOT committed/pushed (awaiting commit-target decision §7-Q3).
**Author run:** 2026-06-05 (TR), branch `chore/de-baseline-rotation-2026-06-04`.
**Trigger:** DE v1 forward chain frozen since the 2026-04-29 research baseline; Stage 5
`ema_context_pilot3_forward` HALTs at SPEC §6.4 (`unmatched_old=136 > sub-ceiling=11`).

---

## 0. Why a rotation (diagnosis, confirmed)

- The 136 `unmatched_old` events are **100% concentrated in the last 3 trading days of the
  04-29 archive** (04-27: 63, 04-28: 41, 04-29: 32), all `horizontal_base / long_breakout`.
- For all 136 the ticker still exists in current HB but that base bar_date is gone — i.e. these
  were **provisional trailing-edge breakouts that failed forward confirmation** over the
  subsequent ~5 weeks. Benign right-edge maturation, **not** a `stable_event_key` bug, **not**
  random historical churn.
- `new_current=482` (04: 64, 05: 404, 06: 14) — the genuinely new post-cutoff events.
- The §6.4 sub-ceiling (=11) assumes the forward run happens ~1 trading day after the lock.
  A 5-week gap legitimately blows it. The fix is a **deliberate baseline rotation**, not a guard
  override.

**Continuity impact (checked):** downstream products
(`paper_execution_v0_trades.parquet` n=8693, `..._trigger_retest_trades.parquet` n=2003,
`decision_engine_v1_events.parquet` n=338) carry **no `event_id` / `stable_event_key` column** —
they key on (date, ticker, source, family, …). Rotating the cascade's internal `event_id`
namespace does **not** orphan open paper trades. Contained.

---

## 1. Governance (repo's own rules — must be honored)

- `tools/pin_baselines.toml` header: *"Edits to this file MUST go through a tracked PR with
  explicit ONAY (any change is a baseline-lock rotation). … Any future rotation requires a
  dedicated PR with new archive parquet(s), updated full sha256 entries, and validation evidence."*
- Each pilot generator (`ema_context_pilot{3,5,6,7}.py`) is a **"single authorized run; bug →
  void+restart, no amend"** artifact governed by its own LOCKED spec
  (`memory/ema_context_pilot{3,5,6}_spec.md` LOCKED 2026-05-03;
  `memory/ema_context_pilot7_trigger_retest_spec.md` LOCKED 2026-05-04). Re-running them on new
  data is a **re-issue of those specs**, requiring explicit ONAY per spec.
- There is **no rotation runbook** in `memory/` or `docs/`. This document is the proposed one.

---

## 2. Anchors to rotate — `tools/pin_baselines.toml` (16 entries)

`asof_date`/`path`/`sha256` fields. New `path` encodes new asof + new sha8. New `sha256` is the
full digest of the regenerated artifact (TBD = produced during execution, not knowable until the
generating step runs).

| # | Anchor | Current sha8 / asof | New value | Source of new value |
|---|--------|---------------------|-----------|---------------------|
| 1 | `locked_hb_archive` | `2eb8a9a5` / 2026-04-29 | sha `d6ae9f2a` / **2026-06-04** | **computable now** — current `horizontal_base_event_v1.parquet` |
| 2 | `research_frozen_metadata_archive` | `5aa1936b` / 2026-05-05 | new `ema_context_daily_metadata.json` snapshot | freeze current `ema_context_daily_metadata.json` |
| 3 | `locked_pilot3_panel` | `5809fef4` | TBD | regen `ema_context_pilot3.py` (after §3 const bump) |
| 4 | `locked_pilot5_panel` | `683366dc` | TBD | regen `ema_context_pilot5.py` |
| 5 | `locked_pilot6_panel` | `4a362677` | TBD | regen `ema_context_pilot6.py` |
| 6 | `locked_pilot7_panel` | `688513e0` | TBD | regen `ema_context_pilot7.py` |
| 7 | `locked_pilot7_gates_main` | `49d0fdcc` | TBD | pilot7 regen sidecar |
| 8 | `locked_pilot7_drop_diagnostic_census` | `ff802f6c` | TBD | pilot7 regen sidecar |
| 9 | `locked_de_v1_events` | `c1c5ba39` | TBD | Stage 8 Tier-2 output @06-04 |
| 10 | `locked_de_v1_tier2_label_distribution` | `5183da12` | TBD | Stage 8 output |
| 11 | `locked_de_v1_paper_link_integrity` | `892d496e` | TBD | Stage 8 output |
| 12 | `locked_de_v1_tier2_label_summary` | `47bda8e2` | TBD | Stage 8 output |
| 13 | `locked_de_v1_tier3_review_report` | `54030072` | TBD | Stage 8/Tier-3 render |
| 14 | `locked_de_v0_classification_panel` | `0bae7104` | TBD | Stage 7 panel @06-04 |

(Items 9–14 are *consumed* by Stages 6–8 guards; they can only be re-locked **after** the chain
runs clean once on the new baseline — chicken/egg resolved by the two-pass ordering in §5.)

---

## 3. Hardcoded constants to bump (source code — each tied to a LOCKED spec)

| File:line | Const | Current | New | Computable now? |
|-----------|-------|---------|-----|------|
| `ema_context_pilot3.py:101` | `EXPECTED_HB_ROWS` | 10470 | **10816** | ✅ |
| `ema_context_pilot3.py:103` | `EXPECTED_EC_ROWS` | 451475 | **461898** | ✅ |
| `ema_context_pilot5.py:90` | `EXPECTED_HB_ROWS` | 10470 | 10816 | ✅ |
| `ema_context_pilot5.py:91` | `EXPECTED_EARLINESS_ROWS` | 10470 | TBD (new HB event count) | after regen |
| `ema_context_pilot6.py:97` | `EXPECTED_HB_ROWS` | 10470 | 10816 | ✅ |
| `ema_context_pilot6.py:98` | `EXPECTED_PILOT5_PANEL_ROWS` | 10470 | TBD | after pilot5 regen |
| `ema_context_pilot7.py:111` | `EXPECTED_HB_ROWS` | 10470 | 10816 | ✅ |
| `ema_context_pilot7.py:112` | `EXPECTED_PILOT5_PANEL_ROWS` | 10470 | TBD | after pilot5 regen |
| `ema_context_pilot7.py:113` | `EXPECTED_HB_SHA256` | `2eb8a9a5…` | `d6ae9f2a4f71…` (full) | ✅ |
| `ema_context_pilot7.py:114` | `EXPECTED_EARLINESS_SHA256` | `a6926de0…` | TBD | after pilot5 regen |
| `ema_context_pilot3_forward.py:57` | `RESEARCH_BASELINE_HB_ROWS` | 10470 | 10816 | ✅ |
| `ema_context_pilot3_forward.py:58` | `EXPECTED_LOCKED_HB_ARCHIVE_SHA` | `2eb8a9a5…` | `d6ae9f2a4f71…` | ✅ |
| `ema_context_pilot3_forward.py:61` | `EXPECTED_LOCKED_PILOT3_PANEL_ROWS` | 322011 | TBD | after pilot3 regen |
| `ema_context_pilot3_forward.py:62` | `EXPECTED_LOCKED_PILOT3_UNIQUE_EID` | 10470 | 10816 | ✅ (= new HB event count) |
| `ema_context_pilot3_forward.py:68` | hardcoded archive path | `…asof_2026-04-29…2eb8a9a5…` | `…asof_2026-06-04…d6ae9f2a…` | ✅ |
| `decision_engine_v1_ema_forward_alignment.py:69` | `RESEARCH_BASELINE_HB_ROWS` | 10470 | 10816 | ✅ |

Note: `UNMAPPED_LOCKED_COUNT_SUBCEILING = max(10, ceil(0.001·N))` auto-derives → stays **11** at
N=10816 (coincidence; no edit needed but re-verify).

**Full new HB sha256:** `d6ae9f2a4f71bb206a4ca8f07e349cd5ce8ea85afb2114697714b2f95f84a98f`
(rows 10816, size 3,598,685 bytes).

---

## 4. Spec files requiring ONAY (re-issue / amend note)

- `memory/ema_context_pilot3_spec.md` (LOCKED 2026-05-03)
- `memory/ema_context_pilot5_spec.md` (LOCKED 2026-05-03)
- `memory/ema_context_pilot6_spec.md` (LOCKED 2026-05-03)
- `memory/ema_context_pilot7_trigger_retest_spec.md` (LOCKED 2026-05-04)
- `memory/decision_engine_v1_tier{1,2,3}_*_spec.md` + implementation spec — append a
  "baseline rotation 2026-06-04" addendum recording the new anchor set + validation evidence.

Each is "single authorized run / no amend" → the rotation is a **new authorized run**, logged as
such (void-old-baseline + restart-at-06-04), with this plan as the evidence trail.

---

## 5. Execution order (two-pass; resolves the consumer/producer chicken-egg)

**Pass A — regenerate frozen inputs (Steps that the cascade reads FROM):**
1. Freeze current HB → `output/_archive/horizontal_base_event_v1__pre_refresh__asof_2026-06-04__sha256_d6ae9f2a.parquet`.
2. Freeze `ema_context_daily_metadata.json` (research_frozen_metadata).
3. Bump computable-now constants (§3 ✅ rows) in pilot3/5/6/7 + pilot3_forward + ema_forward_alignment.
4. Regen pilot3 → record panel rows + unique_eid → write back `EXPECTED_LOCKED_PILOT3_PANEL_ROWS`, snapshot to `__locked__` archive, sha.
5. Regen pilot5 → record earliness rows/sha → write back pilot6/7 `EXPECTED_PILOT5_PANEL_ROWS` + pilot7 `EXPECTED_EARLINESS_SHA256`; snapshot+sha.
6. Regen pilot6, pilot7 → snapshot panels + pilot7 sidecars (gates_main, drop_census); sha.
7. Update `pin_baselines.toml` anchors 1–8.

**Validation gate A:** re-run Stage 5 `ema_forward_alignment` with baseline=06-04, current HB=06-04
→ expect `unmatched_old=0, new_current=0, delta_rows=0` → PASS. If not 0, STOP (regen diverged).

**Pass B — produce + re-lock downstream baselines:**
8. Stage 6 `paper_execution_v0_forward_run --lctd-required 2026-06-04`.
9. Stage 7 `decision_engine_v1_panel_refresh` → re-lock `locked_de_v0_classification_panel` (anchor 14).
10. Stage 7.5 `trident_probe_mb_3y`.
11. Stage 8 `decision_engine_v1_run --tier 2 --asof-date 2026-06-04` → re-lock anchors 9–13 from its outputs.
12. Stage 9 `decision_engine_v1_watchlist_generator --asof-date 2026-06-04`.

**Validation gate B:** events parquet date max = 2026-06-04; watchlist emitted; pin_baselines all
16 anchors point at existing files whose actual sha256 == pinned sha256 (anti-tamper self-check).

---

## 6. Rollback

- All work on branch `chore/de-baseline-rotation-2026-06-04`; nothing committed/pushed without ONAY.
- Operational pilot panels backed up at `/tmp/de_rot_backup/` (pilot3/5/6/7, all 04-29-era).
- Old `__locked__` archives are NOT deleted (new ones get new sha8 filenames) → revert = restore
  `pin_baselines.toml` + constants from git, `git restore` the panels.

---

## 7. Open questions for ONAY

1. Approve regenerating the four pilot panels (re-issuing their LOCKED single-authorized-run specs)?
2. Approve `pin_baselines.toml` rotation (all 16 anchors) as the governed PR?
3. Commit target: this branch → PR into `main`, or fold into another branch?
4. Should the data-layer refresh (HB/nyxexp/ema_context/scanners/regime @06-04) be committed
   together with the rotation, or as a separate prior commit?

---

## 8. EXECUTION RECEIPT (2026-06-05, ONAY granted)

**Branch:** `chore/de-baseline-rotation-2026-06-04` (not committed/pushed). Backups `/tmp/de_rot_backup/`.

### New baseline values (actual)
| Artifact | Old | New rows | New sha256 |
|---|---|---|---|
| HB archive | 10470 / 2eb8a9a5 | 10816 | d6ae9f2a4f71bb206a4ca8f07e349cd5ce8ea85afb2114697714b2f95f84a98f |
| ema_context (n_rows) | 451475 | 461898 | — |
| pilot3 panel | 322011 / 5809fef4 | 335053 (eid 10816) | ff0c9b2227395ae27a3015d4dd972ccfce21488a6923396a4a2be22f9ca92307 |
| pilot4 earliness | 10470 / a6926de0 | 10816 | cd98847bacbaa1d8d1681c8d093178ebe6b9dccd2404cb18104baf9fe9a86fe0 |
| pilot5 panel | 10470 / 683366dc | 10816 | 89877c1db343fd0cc233385ee84852e2bebc9e8119d03798a5a7eb87cb2deefb |
| pilot6 panel | 4a362677 | — | 310b17c2c87bf61f0e3dcf7ec5e009d2d591e5cce2f35996472e160230a93316 |
| pilot7 panel | 688513e0 | — | a832f6bffa1622707603b764624b5287d35075b63f3a6246952edaf6eb0b4048 |
| pilot7 gates_main | 49d0fdcc | — | 9add1212a1415dd74022223f55cad377a22c17f896dd181b48a87cdd75959634 |
| pilot7 drop_census | ff802f6c | — | ac607356174998b7156f2edffd8d802d177c05fa8512b60f17aabd22f1eefa35 |
| de_v1 events | c1c5ba39 (338) | 230 @06-04 | d239769cbeae4e287e1530c3fc469684acbb2f910e31c4ef7392c47d95c99e7b |
| de_v1 label_distribution | 5183da12 | — | 4b87f0d7ff0176d661904ee85a53935348cca4ddff2969b9e2895b092a4cce53 |
| de_v1 paper_link_integrity | 892d496e | — | efc742028bc4906d22de171ef9065711696f00ab85a3d781424054549637b52c |
| de_v1 tier2_label_summary | 47bda8e2 | — | bf173dcb9a949ca237cbceac9059af151c3d8cf9a3f564d90e94194d91731dd7 |
| de_v0 classification_panel | 0bae7104 (454666) | 456576 @06-04 | a76bf5b3d3795d0efad2b89ce01464f35836de7d0bf1f8bbb97661e6f0af01ed |

### Correction during execution
- **`research_frozen_metadata_archive` NOT rotated** — kept at 2026-05-05. The research_frozen
  breakpoints are permanently frozen (forward consumers tag with the original breakpoints for
  cross-chain label consistency); they do NOT track the HB baseline. Confirmed: 05-05 bp ≠ 06-04 bp,
  existing sidecar matches 05-05. Initial rotation of this anchor was reverted; the erroneous
  06-04 metadata archive was removed.
- **`locked_de_v1_tier3_review_report` NOT rotated** — stale placeholder (internal asof 2026-04-01),
  byte-equality sentinel only, not produced by the close chain. Left as-is per pin note.

### Code constants bumped (10470→10816 unless noted)
pilot3.py (HB_ROWS, EC_ROWS 451475→461898); pilot4.py (HB_ROWS, PILOT3_PANEL_ROWS 322011→335053);
pilot5.py (HB_ROWS, EARLINESS_ROWS); pilot6.py (HB_ROWS, PILOT5_PANEL_ROWS);
pilot7.py (HB_ROWS, PILOT5_PANEL_ROWS, HB_SHA256, EARLINESS_SHA256);
pilot3_forward.py (RESEARCH_BASELINE_HB_ROWS, EXPECTED_LOCKED_HB_ARCHIVE_SHA, PILOT3_PANEL_ROWS
322011→335053, UNIQUE_EID, archive path); ema_forward_alignment.py (RESEARCH_BASELINE_HB_ROWS).

### Gate results
- **Gate A** (Stage 5 forward cascade): PASS — matched_old=10816, unmatched_old=0, new_current=0,
  delta_rows=0; pilot3/4/5/6/7 forward all PASS.
- **Stage 6** paper_execution: PASS_DEGRADED (line_e PASS; line_tr STALE → TRIGGER_RETEST excluded).
- **Stage 7** panel_refresh: PASS (456576 rows @06-04). **Stage 7.5** trident_probe: PASS (130716).
- **Stage 8** DE Tier-2: PASS — 230 events @06-04 (EXECUTABLE 29 / SIZE_REDUCED 24 /
  WAIT_BETTER_ENTRY 50 / WAIT_RETEST 124 / PAPER_ONLY 3). **Stage 9** watchlist: emitted @06-04.
- **Gate B** pin anti-tamper self-check: 16/16 anchors exist + sha MATCH.
