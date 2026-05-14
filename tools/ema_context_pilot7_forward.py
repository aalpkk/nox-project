"""ema_context Pilot 7 — Trigger/Retest FORWARD sibling (stable_event_key panel, grow-aware).

Spec: memory/ema_context_forward_alignment_spec.md (LOCK candidate v2 → LOCKED 2026-05-05)

Forward output contract:
  - Skip locked `_validate_inputs` (sha256 + row-count guards anchor frozen-era HB +
    earliness fingerprints; incompatible with forward state by design).
  - Read forward Pilot 5 panel `output/ema_context_pilot5_panel_forward.parquet`
    (already carries trigger / retest_bounce / extended rows + earliness scores).
  - Tag `ema_tr_tier` + `ema_tr_tier_atr` per locked Pilot 7 `_build_panel`
    semantics (EVENT_NEAR_TIER / NON_EVENT_NEAR / DROP_EARLY_DIAGNOSTIC / OUT_OF_SCOPE).
  - Reuse all locked Pilot 7 statistical kernels:
        _gate_pair, _eligible_universe, _g5_state_stability,
        _supplementary_table, _drop_diagnostic_census, _final_verdict,
        _emit_gate_block, _write_gates_main_csv
  - Locked Pilot 7 outputs (6 files) remain BYTE-EQUAL pre/post.

Forward siblings emitted:
  output/ema_context_pilot7_panel_forward.parquet
  output/ema_context_pilot7_gates_main_forward.csv
  output/ema_context_pilot7_state_stability_forward.csv
  output/ema_context_pilot7_supplementary_forward.csv
  output/ema_context_pilot7_drop_diagnostic_census_forward.csv
  output/ema_context_pilot7_summary_forward.md
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.ema_context_pilot7 import (  # type: ignore
    _gate_pair,
    _eligible_universe,
    _g5_state_stability,
    _supplementary_table,
    _drop_diagnostic_census,
    _final_verdict,
    _emit_gate_block,
    _write_gates_main_csv,
    EVENT_NEAR_LO,
    EVENT_NEAR_HI,
    DROP_BOUNDARY,
    EVENT_NEAR_TIER,
    NON_EVENT_NEAR,
    DROP_EARLY_DIAGNOSTIC,
    TIER_OOS,
    TR_STATES,
    MIN_N_TIER,
    MIN_N_BASELINE,
    UPLIFT_FLOOR_R,
    PRIMARY_OUTCOME,
    P_THRESHOLD,
    BOOTSTRAP_N,
    BOOTSTRAP_SEED,
)

# =============================================================================
# LOCKED CONSTANTS (forward namespace)
# =============================================================================

CANDIDATE_KEY = ["ticker", "bar_date", "setup_family", "signal_type", "breakout_bar_date"]

PILOT5_PANEL_FORWARD = PROJECT_ROOT / "output" / "ema_context_pilot5_panel_forward.parquet"

OUT_PANEL = PROJECT_ROOT / "output" / "ema_context_pilot7_panel_forward.parquet"
OUT_GATES_MAIN = PROJECT_ROOT / "output" / "ema_context_pilot7_gates_main_forward.csv"
OUT_STATE_STABILITY = PROJECT_ROOT / "output" / "ema_context_pilot7_state_stability_forward.csv"
OUT_SUPPLEMENTARY = PROJECT_ROOT / "output" / "ema_context_pilot7_supplementary_forward.csv"
OUT_DROP_CENSUS = PROJECT_ROOT / "output" / "ema_context_pilot7_drop_diagnostic_census_forward.csv"
OUT_SUMMARY = PROJECT_ROOT / "output" / "ema_context_pilot7_summary_forward.md"


def _atomic_write_parquet(df: pd.DataFrame, dest: Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(dest)


def _atomic_write_csv(df: pd.DataFrame, dest: Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(dest)


def _atomic_write_text(text: str, dest: Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(dest)


# =============================================================================
# Forward panel build — replicates locked Pilot 7 _build_panel tier semantics
# =============================================================================


def _build_panel_forward() -> pd.DataFrame:
    if not PILOT5_PANEL_FORWARD.exists():
        raise FileNotFoundError(
            f"Forward Pilot 5 panel missing: {PILOT5_PANEL_FORWARD}. "
            f"Pilot 5 forward must run before Pilot 7 forward."
        )
    panel = pd.read_parquet(PILOT5_PANEL_FORWARD).reset_index(drop=True)
    required = ["signal_state", "earliness_score_pct", "earliness_score_atr", PRIMARY_OUTCOME]
    missing = [c for c in required if c not in panel.columns]
    if missing:
        raise RuntimeError(
            f"Forward Pilot 5 panel missing required columns: {missing}. SPEC violation — STOP."
        )

    def _tier(state, score):
        if state not in TR_STATES:
            return TIER_OOS
        if pd.isna(score):
            return TIER_OOS
        if score < DROP_BOUNDARY:
            return DROP_EARLY_DIAGNOSTIC
        if EVENT_NEAR_LO <= score <= EVENT_NEAR_HI:
            return EVENT_NEAR_TIER
        if score > EVENT_NEAR_HI:
            return NON_EVENT_NEAR
        return TIER_OOS

    panel["ema_tr_tier"] = [
        _tier(s, e) for s, e in zip(panel["signal_state"], panel["earliness_score_pct"])
    ]
    panel["ema_tr_tier_atr"] = [
        _tier(s, e) for s, e in zip(panel["signal_state"], panel["earliness_score_atr"])
    ]
    return panel


# =============================================================================
# Forward summary writer (Markdown)
# =============================================================================


def _format_summary_forward(
    final_verdict: str,
    notes: list[str],
    g1: dict,
    g2: dict,
    g6_g1: dict,
    g5_summary: dict,
    state_stability_df: pd.DataFrame,
    supp_df: pd.DataFrame,
    drop_census: pd.DataFrame,
    panel: pd.DataFrame,
    runtime_s: float,
    track_counts: dict,
) -> str:
    lines: list[str] = []
    lines.append("# ema_context Pilot 7 FORWARD — Trigger/Retest Outcome Separation Run Summary")
    lines.append("")
    lines.append("- Spec: `memory/ema_context_forward_alignment_spec.md` LOCK candidate v2 (LOCKED 2026-05-05)")
    lines.append(f"- Run date (UTC): {pd.Timestamp.utcnow().isoformat()}")
    lines.append(f"- Runtime: {runtime_s:.1f}s")
    lines.append("- Source: forward Pilot 5 panel `output/ema_context_pilot5_panel_forward.parquet`")
    lines.append(f"- Forward panel rows: {len(panel):,}")
    lines.append(f"- Track distribution: {track_counts}")
    lines.append(
        f"- Tier rule (LOCKED): EVENT_NEAR_TIER = signal_state ∈ {{trigger, retest_bounce}} ∧ "
        f"{EVENT_NEAR_LO} ≤ earliness_score_pct ≤ {EVENT_NEAR_HI}"
    )
    lines.append(
        f"- NON_EVENT_NEAR = signal_state ∈ {{trigger, retest_bounce}} ∧ earliness_score_pct > {EVENT_NEAR_HI}"
    )
    lines.append(
        f"- DROP_EARLY_DIAGNOSTIC = signal_state ∈ {{trigger, retest_bounce}} ∧ "
        f"earliness_score_pct < {DROP_BOUNDARY} (DROPPED from BASELINE; descriptive census only)"
    )
    lines.append(
        f"- OUT_OF_SCOPE = signal_state == 'extended' OR NaN earliness OR non-{{trigger,retest_bounce,extended}} state"
    )
    lines.append(f"- Primary outcome (LOCKED): `{PRIMARY_OUTCOME}`")
    lines.append("")

    n_event_near = int((panel["ema_tr_tier"] == EVENT_NEAR_TIER).sum())
    n_non_event = int((panel["ema_tr_tier"] == NON_EVENT_NEAR).sum())
    n_drop = int((panel["ema_tr_tier"] == DROP_EARLY_DIAGNOSTIC).sum())
    n_oos = int((panel["ema_tr_tier"] == TIER_OOS).sum())
    n_baseline = n_event_near + n_non_event

    lines.append("## Forward cohort census (pct primary)")
    lines.append("")
    lines.append(f"- EVENT_NEAR_TIER: {n_event_near:,}")
    lines.append(f"- NON_EVENT_NEAR: {n_non_event:,}")
    lines.append(f"- BASELINE (EVENT_NEAR ∪ NON_EVENT_NEAR): {n_baseline:,}")
    lines.append(f"- DROP_EARLY_DIAGNOSTIC: {n_drop:,} (DROPPED, descriptive census only)")
    lines.append(f"- OUT_OF_SCOPE (extended / NaN earliness / non-tr/retest states): {n_oos:,}")
    lines.append("")

    n_trig = int(((panel["signal_state"] == "trigger") & (panel["ema_tr_tier"] != TIER_OOS)).sum())
    n_retest = int(((panel["signal_state"] == "retest_bounce") & (panel["ema_tr_tier"] != TIER_OOS)).sum())
    n_trig_event_near = int(((panel["signal_state"] == "trigger") & (panel["ema_tr_tier"] == EVENT_NEAR_TIER)).sum())
    n_retest_event_near = int(
        ((panel["signal_state"] == "retest_bounce") & (panel["ema_tr_tier"] == EVENT_NEAR_TIER)).sum()
    )
    lines.append(f"- trigger total in scope: {n_trig:,} (EVENT_NEAR: {n_trig_event_near:,})")
    lines.append(f"- retest_bounce total in scope: {n_retest:,} (EVENT_NEAR: {n_retest_event_near:,})")
    lines.append("")

    lines.append(f"## Overall Pilot 7 FORWARD verdict: **{final_verdict}**")
    lines.append("")
    for n in notes:
        lines.append(f"- {n}")
    lines.append("")

    lines.append("## G1 — EVENT_NEAR_TIER vs BASELINE (uplift; primary pct)")
    lines.append("")
    _emit_gate_block(lines, g1, magnitude_floor_label=f"≥ {UPLIFT_FLOOR_R} R uplift")
    lines.append("")

    lines.append("## G2 — NON_EVENT_NEAR vs BASELINE (LABEL SANITY CHECK)")
    lines.append("")
    _emit_gate_block(lines, g2, magnitude_floor_label=None, sanity=True)
    lines.append("")

    lines.append("## G3 — Min-N (forward corpus)")
    lines.append("")
    lines.append(
        f"- EVENT_NEAR_TIER N={g1['n_a']:,} (≥ {MIN_N_TIER}: "
        f"{'PASS' if g1['n_a'] >= MIN_N_TIER else 'FAIL'}) | "
        f"BASELINE N={g1['n_b']:,} (≥ {MIN_N_BASELINE}: "
        f"{'PASS' if g1['n_b'] >= MIN_N_BASELINE else 'FAIL'}) | "
        f"NON_EVENT_NEAR N={g2['n_a']:,} (≥ {MIN_N_TIER}: "
        f"{'PASS' if g2['n_a'] >= MIN_N_TIER else 'FAIL'})"
    )
    lines.append("")

    lines.append("## G5 — State stability (trigger-only + retest-only secondary)")
    lines.append("")
    lines.append(
        f"- trigger-only: n_event_near={g5_summary['trigger_n_a']} / "
        f"n_baseline={g5_summary['trigger_n_b']} | "
        f"Δ={(g5_summary['trigger_delta'] if g5_summary['trigger_delta'] is not None else float('nan')):.4f} R "
        f"| classification={g5_summary['trigger_classification']}"
    )
    lines.append(
        f"- retest_bounce-only: n_event_near={g5_summary['retest_n_a']} / "
        f"n_baseline={g5_summary['retest_n_b']} | "
        f"Δ={(g5_summary['retest_delta'] if g5_summary['retest_delta'] is not None else float('nan')):.4f} R "
        f"| classification={g5_summary['retest_classification']}"
    )
    lines.append(
        f"- G5 status: **{g5_summary['g5_status']}** "
        f"(trigger-only must be NON_CONTRADICTING; retest-only INSUFFICIENT does not fail)"
    )
    lines.append("")

    lines.append("## G6 — ATR mirror audit (audit-only)")
    lines.append("")
    _emit_gate_block(lines, g6_g1, magnitude_floor_label=f"≥ {UPLIFT_FLOOR_R} R uplift")
    lines.append("")

    lines.append("## Supplementary descriptive — EVENT_NEAR_TIER vs NON_EVENT_NEAR (NOT gates)")
    lines.append("")
    lines.append("| outcome | n_event_near | n_non_event_near | mean_event_near | mean_non_event_near | Δmean | CI | status |")
    lines.append("|---|---|---|---|---|---|---|---|")

    def _f(x, fmt=".4f"):
        if x is None:
            return "—"
        if isinstance(x, float) and np.isnan(x):
            return "—"
        return format(x, fmt)

    for _, r in supp_df.iterrows():
        ci_disp = (
            f"[{_f(r['ci_lo'])}, {_f(r['ci_hi'])}]"
            if r["ci_lo"] is not None else "—"
        )
        lines.append(
            f"| {r['outcome']} | {r['n_event_near']} | {r['n_non_event_near']} | "
            f"{_f(r['mean_event_near'])} | {_f(r['mean_non_event_near'])} | "
            f"{_f(r['delta_mean'])} | {ci_disp} | {r['status']} |"
        )
    lines.append("")

    lines.append("## DROP_EARLY_DIAGNOSTIC census (descriptive only — NO gate, NO interpretation)")
    lines.append("")
    lines.append("| scope | n | n_with_outcome | mean_R | median_R | win_rate | profit_factor |")
    lines.append("|---|---|---|---|---|---|---|")
    for _, r in drop_census.iterrows():
        lines.append(
            f"| {r['scope']} | {r['n']} | {r['n_with_outcome']} | "
            f"{_f(r['mean_realized_R_10d'])} | {_f(r['median_realized_R_10d'])} | "
            f"{_f(r['win_rate'])} | "
            f"{_f(r['profit_factor'], '.4f') if r['profit_factor'] is not None else '—'} |"
        )
    lines.append("")

    lines.append("## PASS interpretation ceiling (verbatim from locked Pilot 7 spec)")
    lines.append("")
    lines.append("- Forward verdict does NOT change locked-research Pilot 7 verdict. Locked outputs remain byte-equal.")
    lines.append("- Layer B paper-execution shadow remains OUT OF SCOPE; live trade gate STAYS CLOSED.")
    lines.append("- Forbidden interpretations: live entry filter / scanner ranking / position sizing / ML feature /")
    lines.append("  pooling with extended cohort / cross-pilot cascade without their own pre-reg /")
    lines.append("  threshold sweep / neighbor-bucket promotion / 'short NON_EVENT_NEAR' shortcut /")
    lines.append("  DROP_EARLY_DIAGNOSTIC reading translated into trigger/retest trade rule.")
    lines.append("")
    return "\n".join(lines) + "\n"


# =============================================================================
# Main
# =============================================================================


def main() -> int:
    print("=== ema_context Pilot 7 FORWARD — single authorized run ===", flush=True)
    print("Spec: memory/ema_context_forward_alignment_spec.md (LOCKED 2026-05-05)", flush=True)
    t0 = time.time()

    print("[1/8] Building forward panel via Pilot 5 forward panel + tier tagging...", flush=True)
    panel = _build_panel_forward()
    track_counts = (
        panel.drop_duplicates(subset=CANDIDATE_KEY)["track"].value_counts().to_dict()
        if "track" in panel.columns else {}
    )
    n_event_near = int((panel["ema_tr_tier"] == EVENT_NEAR_TIER).sum())
    n_non_event = int((panel["ema_tr_tier"] == NON_EVENT_NEAR).sum())
    n_drop = int((panel["ema_tr_tier"] == DROP_EARLY_DIAGNOSTIC).sum())
    n_oos = int((panel["ema_tr_tier"] == TIER_OOS).sum())
    n_baseline = n_event_near + n_non_event
    print(
        f"  EVENT_NEAR_TIER={n_event_near:,} | NON_EVENT_NEAR={n_non_event:,} | "
        f"BASELINE={n_baseline:,} | DROP_EARLY_DIAGNOSTIC={n_drop:,} | OUT_OF_SCOPE={n_oos:,}",
        flush=True,
    )
    print(f"  track distribution: {track_counts}", flush=True)

    g3_baseline_pass = n_baseline >= MIN_N_BASELINE

    # G1 — EVENT_NEAR_TIER vs BASELINE (uplift, primary pct)
    print("[2/8] G1 (EVENT_NEAR_TIER vs BASELINE, pct primary)...", flush=True)
    eligible = _eligible_universe(panel, "ema_tr_tier")
    a_arr = eligible.loc[eligible["ema_tr_tier"] == EVENT_NEAR_TIER, PRIMARY_OUTCOME].astype(float).to_numpy()
    baseline_arr = eligible[PRIMARY_OUTCOME].astype(float).to_numpy()
    g1 = _gate_pair(
        a_arr, baseline_arr,
        magnitude_floor=UPLIFT_FLOOR_R,
        require_positive=True,
        n_a_min=MIN_N_TIER,
        n_b_min=MIN_N_BASELINE,
    )
    if g1["delta"] is not None:
        print(
            f"  G1 verdict={g1['verdict']} | Δuplift={g1['delta']:.4f} R | p={g1['p_value']:.2e}",
            flush=True,
        )
    else:
        print(f"  G1 verdict={g1['verdict']} (INSUFFICIENT)", flush=True)

    # G2 — NON_EVENT_NEAR vs BASELINE (label sanity)
    print("[3/8] G2 (NON_EVENT_NEAR vs BASELINE, label sanity)...", flush=True)
    b_arr = eligible.loc[eligible["ema_tr_tier"] == NON_EVENT_NEAR, PRIMARY_OUTCOME].astype(float).to_numpy()
    g2 = _gate_pair(
        b_arr, baseline_arr,
        magnitude_floor=0.0,
        require_positive=False,
        n_a_min=MIN_N_TIER,
        n_b_min=MIN_N_BASELINE,
    )
    if g2["delta"] is not None:
        print(
            f"  G2 sanity verdict={g2['verdict']} | Δ(NON_EVENT_NEAR − BASELINE)={g2['delta']:.4f} R",
            flush=True,
        )
    else:
        print(f"  G2 verdict={g2['verdict']} (INSUFFICIENT)", flush=True)

    # G6 — atr mirror audit
    print("[4/8] G6 (ATR mirror audit)...", flush=True)
    eligible_atr = panel[panel["ema_tr_tier_atr"].isin([EVENT_NEAR_TIER, NON_EVENT_NEAR])].copy()
    eligible_atr = eligible_atr.dropna(subset=[PRIMARY_OUTCOME, "ema_tr_tier_atr"])
    a_atr = eligible_atr.loc[eligible_atr["ema_tr_tier_atr"] == EVENT_NEAR_TIER, PRIMARY_OUTCOME].astype(float).to_numpy()
    baseline_atr = eligible_atr[PRIMARY_OUTCOME].astype(float).to_numpy()
    g6_g1 = _gate_pair(
        a_atr, baseline_atr,
        magnitude_floor=UPLIFT_FLOOR_R,
        require_positive=True,
        n_a_min=MIN_N_TIER,
        n_b_min=MIN_N_BASELINE,
    )
    if g6_g1["delta"] is not None:
        print(f"  G6 verdict={g6_g1['verdict']} | Δuplift_atr={g6_g1['delta']:.4f} R", flush=True)
    else:
        print(f"  G6 verdict={g6_g1['verdict']} (INSUFFICIENT)", flush=True)

    # G5 — state stability
    print("[5/8] G5 (state stability: trigger-only + retest-only)...", flush=True)
    state_stability_df, g5_summary = _g5_state_stability(panel, "ema_tr_tier")
    print(
        f"  trigger classification={g5_summary['trigger_classification']} | "
        f"retest classification={g5_summary['retest_classification']} | "
        f"G5 {g5_summary['g5_status']}",
        flush=True,
    )

    # Supplementary
    print("[6/8] Supplementary descriptives...", flush=True)
    supp_df = _supplementary_table(panel, "ema_tr_tier")

    # DROP_EARLY_DIAGNOSTIC census
    print("[7/8] DROP_EARLY_DIAGNOSTIC census...", flush=True)
    drop_census = _drop_diagnostic_census(panel)

    # Final verdict
    final_verdict, notes = _final_verdict(g1, g2, g6_g1, g3_baseline_pass, g5_summary["g5_status"])
    print(f"  FINAL VERDICT: {final_verdict}", flush=True)

    runtime_s = time.time() - t0

    # Write outputs
    print("[8/8] Writing forward outputs (atomic)...", flush=True)
    _atomic_write_parquet(panel, OUT_PANEL)
    _write_gates_main_csv(g1, g2, g6_g1, OUT_GATES_MAIN)
    _atomic_write_csv(state_stability_df, OUT_STATE_STABILITY)
    _atomic_write_csv(supp_df, OUT_SUPPLEMENTARY)
    _atomic_write_csv(drop_census, OUT_DROP_CENSUS)
    summary_text = _format_summary_forward(
        final_verdict, notes, g1, g2, g6_g1,
        g5_summary, state_stability_df, supp_df, drop_census, panel,
        runtime_s, track_counts,
    )
    _atomic_write_text(summary_text, OUT_SUMMARY)
    for p in [OUT_PANEL, OUT_GATES_MAIN, OUT_STATE_STABILITY, OUT_SUPPLEMENTARY, OUT_DROP_CENSUS, OUT_SUMMARY]:
        print(f"  Wrote: {p.name} ({p.stat().st_size:,} B)", flush=True)
    print(f"[Pilot 7 forward] Done in {runtime_s:.1f}s — verdict {final_verdict}", flush=True)

    summary_json = {
        "tool": "ema_context_pilot7_forward",
        "runtime_s": round(runtime_s, 3),
        "panel_rows": int(len(panel)),
        "track_counts": track_counts,
        "event_near_count": n_event_near,
        "non_event_near_count": n_non_event,
        "baseline_count": n_baseline,
        "drop_early_diagnostic_count": n_drop,
        "out_of_scope_count": n_oos,
        "g1_verdict": g1["verdict"],
        "g2_verdict": g2["verdict"],
        "g6_verdict": g6_g1["verdict"],
        "g5_status": g5_summary["g5_status"],
        "g3_baseline_pass": g3_baseline_pass,
        "final_verdict": final_verdict,
        "outputs": {
            "panel": str(OUT_PANEL.relative_to(PROJECT_ROOT)),
            "gates_main": str(OUT_GATES_MAIN.relative_to(PROJECT_ROOT)),
            "state_stability": str(OUT_STATE_STABILITY.relative_to(PROJECT_ROOT)),
            "supplementary": str(OUT_SUPPLEMENTARY.relative_to(PROJECT_ROOT)),
            "drop_diagnostic_census": str(OUT_DROP_CENSUS.relative_to(PROJECT_ROOT)),
            "summary": str(OUT_SUMMARY.relative_to(PROJECT_ROOT)),
        },
    }
    print("PILOT7_FORWARD_SUMMARY_JSON_BEGIN")
    print(json.dumps(summary_json, default=str))
    print("PILOT7_FORWARD_SUMMARY_JSON_END")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
