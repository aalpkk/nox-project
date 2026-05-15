# Decision Engine v1 — Tier 3 review report

## §1 Run identity
- Tier 3 render timestamp (UTC): 2026-05-04T21:07:59.084401+00:00
- Tier 2 source-run timestamp (UTC): 2026-05-04T19:02:14.076314+00:00
- asof_date (resolved): 2026-04-01
- panel max date: 2026-04-01
- Line E max asof_date: 2026-04-24
- Line TR max asof_date: 2026-04-29

## §2 Source-input integrity (sha256 pre/post)
- Tier 2 events.parquet pre:  size=28608, sha256=b191ffee30d25de031963779d29e29156292064fe0ae371f454f21737c41d7b5
- Tier 2 events.parquet post: size=28608, sha256=b191ffee30d25de031963779d29e29156292064fe0ae371f454f21737c41d7b5
- Line E paper parquet pre:  size=285019, sha256=b5e427d4fc4c3647cb128bf80fa11faf0b31b396a523aa316f8cf52fe3051873
- Line E paper parquet post: size=285019, sha256=b5e427d4fc4c3647cb128bf80fa11faf0b31b396a523aa316f8cf52fe3051873
- Line TR paper parquet pre:  size=102905, sha256=ccaaca6a6c0afc649de984c3994f8a2ff39c3dda580059b0333551622d0e0a7a
- Line TR paper parquet post: size=102905, sha256=ccaaca6a6c0afc649de984c3994f8a2ff39c3dda580059b0333551622d0e0a7a
- byte-equality: PASS

## §3 Total event count
- total: 542

## §4 Execution-label distribution (full enum, zero-count shown)
- EXECUTABLE: 0
- WAIT_TRIGGER: 0
- WAIT_RETEST: 76
- WAIT_BETTER_ENTRY: 0
- SIZE_REDUCED: 0
- NOT_EXECUTABLE: 289
- CONTEXT_ONLY: 53
- WAIT_CONFIRMATION: 118
- PAPER_ONLY: 6

## §5 Setup-label distribution (full enum, zero-count shown)
- EARLY_SETUP: 0
- TRIGGERED_SETUP: 14
- RETEST_SETUP: 128
- CONTINUATION_SETUP: 118
- ACCEPTED_CONTINUATION_SETUP: 107
- REVERSAL_SETUP: 118
- STRENGTH_CONTEXT: 0
- WEAK_CONTEXT: 53
- EXTENDED_CONTEXT: 4
- WARNING_CONTEXT: 0

## §6 Market-context distribution (full enum, zero-count shown)
- REGIME_LONG: 542
- REGIME_NEUTRAL: 0
- REGIME_SHORT: 0
- REGIME_UNKNOWN: 0
- REGIME_STALE: 0

## §7 Compact cross-tabs

### §7.1 setup_label × execution_label
| setup_label \ execution_label | EXECUTABLE | WAIT_TRIGGER | WAIT_RETEST | WAIT_BETTER_ENTRY | SIZE_REDUCED | NOT_EXECUTABLE | CONTEXT_ONLY | WAIT_CONFIRMATION | PAPER_ONLY |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EARLY_SETUP | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| TRIGGERED_SETUP | 0 | 0 | 0 | 0 | 0 | 12 | 0 | 0 | 2 |
| RETEST_SETUP | 0 | 0 | 76 | 0 | 0 | 52 | 0 | 0 | 0 |
| CONTINUATION_SETUP | 0 | 0 | 0 | 0 | 0 | 118 | 0 | 0 | 0 |
| ACCEPTED_CONTINUATION_SETUP | 0 | 0 | 0 | 0 | 0 | 107 | 0 | 0 | 0 |
| REVERSAL_SETUP | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 118 | 0 |
| STRENGTH_CONTEXT | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| WEAK_CONTEXT | 0 | 0 | 0 | 0 | 0 | 0 | 53 | 0 | 0 |
| EXTENDED_CONTEXT | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 4 |
| WARNING_CONTEXT | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

### §7.2 source × setup_label (sources alphabetical)
| source \ setup_label | EARLY_SETUP | TRIGGERED_SETUP | RETEST_SETUP | CONTINUATION_SETUP | ACCEPTED_CONTINUATION_SETUP | REVERSAL_SETUP | STRENGTH_CONTEXT | WEAK_CONTEXT | EXTENDED_CONTEXT | WARNING_CONTEXT |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| horizontal_base | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 4 | 0 |
| mb_scanner | 0 | 0 | 128 | 118 | 107 | 0 | 0 | 0 | 0 | 0 |
| nox_rt_daily | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 53 | 0 | 0 |
| nox_weekly | 0 | 0 | 0 | 0 | 0 | 118 | 0 | 0 | 0 | 0 |
| nyxexpansion | 0 | 12 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

### §7.3 source × execution_label (sources alphabetical)
| source \ execution_label | EXECUTABLE | WAIT_TRIGGER | WAIT_RETEST | WAIT_BETTER_ENTRY | SIZE_REDUCED | NOT_EXECUTABLE | CONTEXT_ONLY | WAIT_CONFIRMATION | PAPER_ONLY |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| horizontal_base | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 6 |
| mb_scanner | 0 | 0 | 76 | 0 | 0 | 277 | 0 | 0 | 0 |
| nox_rt_daily | 0 | 0 | 0 | 0 | 0 | 0 | 53 | 0 | 0 |
| nox_weekly | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 118 | 0 |
| nyxexpansion | 0 | 0 | 0 | 0 | 0 | 12 | 0 | 0 | 0 |

### §7.4 paper_stream_ref × execution_label (filtered to PAPER_ONLY)
| paper_stream_ref | PAPER_ONLY count |
| --- | --- |
| paper_execution_v0 | 4 |
| paper_execution_v0_trigger_retest | 2 |

## §8 Source / family / state / timeframe marginals (alphabetical)

### §8.1 source
- horizontal_base: 6
- mb_scanner: 353
- nox_rt_daily: 53
- nox_weekly: 118
- nyxexpansion: 12

### §8.2 family
- bb_1d__above_mb_birth: 42
- bb_1d__mit_touch_first: 10
- bb_1d__retest_bounce_first: 10
- bb_5h__above_mb_birth: 76
- bb_5h__mit_touch_first: 23
- bb_5h__retest_bounce_first: 21
- horizontal_base__extended: 4
- horizontal_base__trigger: 2
- mb_1d__above_mb_birth: 28
- mb_1d__mit_touch_first: 7
- mb_1d__retest_bounce_first: 9
- mb_5h__above_mb_birth: 79
- mb_5h__mit_touch_first: 36
- mb_5h__retest_bounce_first: 12
- nox_rt_daily__pivot_al: 53
- nox_weekly__weekly_pivot_al: 118
- nyxexpansion__triggerA: 12

### §8.3 state
- accepted_continuation: 107
- continuation: 118
- extended: 4
- retest: 52
- retest_pending: 76
- trigger: 185

### §8.4 timeframe
- 1d: 177
- 1w: 118
- 5h: 247

## §9 PAPER_ONLY rows detail (ticker alphabetical)
| ticker | source | family | state | timeframe | asof_date | paper_stream_ref | paper_origin | paper_trade_id | paper_match_key | paper_validity_metadata_missing | paper_valid_from | paper_valid_until | paper_expired_flag | paper_signal_age |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ADEL | horizontal_base | horizontal_base__extended | extended | 1d | 2026-04-01 | paper_execution_v0 | external_reference | - | 2026-04-01|ADEL|EXTENDED | True | - | - | None | - |
| AKMGY | horizontal_base | horizontal_base__extended | extended | 1d | 2026-04-01 | paper_execution_v0 | external_reference | - | 2026-04-01|AKMGY|EXTENDED | True | - | - | None | - |
| ALGYO | horizontal_base | horizontal_base__extended | extended | 1d | 2026-04-01 | paper_execution_v0 | external_reference | - | 2026-04-01|ALGYO|EXTENDED | True | - | - | None | - |
| BORSK | horizontal_base | horizontal_base__trigger | trigger | 1d | 2026-04-01 | paper_execution_v0_trigger_retest | external_reference | - | 2026-04-01|BORSK|TRIGGER_RETEST | True | - | - | None | - |
| ECZYT | horizontal_base | horizontal_base__trigger | trigger | 1d | 2026-04-01 | paper_execution_v0_trigger_retest | external_reference | - | 2026-04-01|ECZYT|TRIGGER_RETEST | True | - | - | None | - |
| FRIGO | horizontal_base | horizontal_base__extended | extended | 1d | 2026-04-01 | paper_execution_v0 | external_reference | - | 2026-04-01|FRIGO|EXTENDED | True | - | - | None | - |

## §10 WAIT_CONFIRMATION (nox_weekly conservative landing)
_NOTE: WAIT_CONFIRMATION is NOT executable. This section is a descriptive view; no execution recommendation is implied._

### §10.1 group counts (source / family / state / timeframe)
| source | family | state | timeframe | count |
| --- | --- | --- | --- | --- |
| nox_weekly | nox_weekly__weekly_pivot_al | trigger | 1w | 118 |

### §10.2 ticker list (alphabetical)
AEFES, ALBRK, ARSAN, ATAKP, BESLR, BORLS, BRISA, CEMAS, DERIM, DESPC, DNISI, DOAS, EDIP, EKSUN, EREGL, EUHOL, GRSEL, GSDHO, HEKTS, ICBCT, IHYAY, ISYAT, KARSN, KBORU, LUKSK, MEGAP, MERCN, MERIT, MGROS, NUGYO, OSTIM, PENTA, PGSUS, RAYSG, SAHOL, SAMAT, SANKO, SARKY, SASA, SELVA, TCELL, TLMAN, TRMET, USAK, YKBNK, YKSLN, YUNSA

## §11 NOT_EXECUTABLE execution-reason-code distribution (full enum, zero-count shown)
- entry_ref_present: 0
- stop_ref_present: 0
- execution_risk_ok: 0
- execution_risk_too_wide: 0
- execution_risk_above_normal_below_reject: 0
- no_stop_ref: 0
- no_entry_ref: 0
- liquidity_limited: 0
- fill_uncertain: 0
- wait_better_entry: 0
- size_reduced: 0
- fill_realism_unresolved: 289

## §12 Setup-layer reason-code distribution (across all events, full enum, zero-count shown)
- accepted_horizon_h1_20d: 107
- clean_breakout_context: 239
- retest_confirmed: 52
- retest_pending_context: 76
- reversal_context: 118
- strength_context: 0
- weak_standalone_context: 53
- extended_context: 4
- failed_continuation_context: 0
- h2_failed_no_15d_rule: 0

## §13 Context-layer reason-code distribution (across all events, full enum, zero-count shown)
- regime_long: 542
- regime_neutral: 0
- regime_short: 0
- regime_unknown: 0
- regime_stale: 0
- nox_sat_conflict: 0
- oe_high: 0
- weekly_support: 0
- momentum_support: 0
- hw_exit_rejected_descriptive_only: 0

## §14 Paper-link integrity recap (verbatim from Tier 2 CSV)
| line | paper_link_attempted | paper_link_matched | paper_link_unmatched | paper_link_duplicate_error | paper_link_skipped_source_missing | paper_link_validity_missing |
| --- | --- | --- | --- | --- | --- | --- |
| EXTENDED | 4 | 4 | 0 | 0 | 0 | 4 |
| TRIGGER_RETEST | 2 | 2 | 0 | 0 | 0 | 2 |
| ALL | 6 | 6 | 0 | 0 | 0 | 6 |

## §15 Paper-validity carry summary
- paper_validity_metadata_missing=true count: 6
- paper_signal_age populated count: 0
- paper_expired_flag=true count: 0
- _Historical paper parquets pre-date the validity revision; `paper_validity_metadata_missing=true` is the LOCKED carry-as-missing semantic per Decision Engine v1 impl spec §3.7. NOT a defect; resolution lives behind the upstream `paper_execution_v0` real forward-emission LOCK._

## §16 Forbidden-fields static absence
events parquet schema does NOT contain `final_action` / `selection_score` / `edge_score` / `prior_edge_score` / `setup_score` / `rank` / `portfolio_pick` / `portfolio_weight`. Read-only verification by Tier 3; no recompute.

## §17 Scope statement
_This is a Tier 3 review report. NO ranking, NO portfolio, NO score, NO forward returns, NO live integration. Live trade gate STAYS CLOSED._
