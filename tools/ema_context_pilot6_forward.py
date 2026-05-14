"""ema_context Pilot 6 — Layer A FORWARD sibling (stable_event_key panel, grow-aware).

Spec: memory/ema_context_forward_alignment_spec.md (LOCK candidate v2 → LOCKED 2026-05-05)

Forward output contract:
  - Read forward Pilot 5 panel `output/ema_context_pilot5_panel_forward.parquet`.
  - Tag tier_pct + tier_atr per locked Pilot 6 `_build_panel` semantics.
  - Reuse all locked Pilot 6 statistical kernels:
        _gate_pair, _g5_stability, _supplementary_table, _final_verdict,
        _write_gates_main_csv, _emit_gate_block
  - Locked Pilot 6 outputs (5 files) remain BYTE-EQUAL pre/post.

Forward siblings emitted:
  output/ema_context_pilot6_panel_forward.parquet
  output/ema_context_pilot6_gates_main_forward.csv
  output/ema_context_pilot6_stability_forward.csv
  output/ema_context_pilot6_supplementary_forward.csv
  output/ema_context_pilot6_summary_forward.md
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

from tools.ema_context_pilot6 import (  # type: ignore
    _gate_pair,
    _g5_stability,
    _supplementary_table,
    _final_verdict,
    _write_gates_main_csv,
    _emit_gate_block,
    EARLY_THRESHOLD,
    PRIMARY_OUTCOME,
    SUPPLEMENTARY_OUTCOMES,
    P_THRESHOLD,
    BOOTSTRAP_N,
    BOOTSTRAP_SEED,
    UPLIFT_FLOOR_R,
    SUBSLICE_UPLIFT_FLOOR_R,
    PILOT5_SANITY_FLOOR_R,
    MIN_N_TIER,
    MIN_N_BASELINE,
    TIER_A,
    TIER_B,
    TIER_OOS,
)

# =============================================================================
# LOCKED CONSTANTS (forward namespace)
# =============================================================================

CANDIDATE_KEY = ["ticker", "bar_date", "setup_family", "signal_type", "breakout_bar_date"]

PILOT5_PANEL_FORWARD = PROJECT_ROOT / "output" / "ema_context_pilot5_panel_forward.parquet"

OUT_PANEL = PROJECT_ROOT / "output" / "ema_context_pilot6_panel_forward.parquet"
OUT_GATES_MAIN = PROJECT_ROOT / "output" / "ema_context_pilot6_gates_main_forward.csv"
OUT_STABILITY = PROJECT_ROOT / "output" / "ema_context_pilot6_stability_forward.csv"
OUT_SUPPLEMENTARY = PROJECT_ROOT / "output" / "ema_context_pilot6_supplementary_forward.csv"
OUT_SUMMARY = PROJECT_ROOT / "output" / "ema_context_pilot6_summary_forward.md"


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


def _build_panel_forward() -> pd.DataFrame:
    if not PILOT5_PANEL_FORWARD.exists():
        raise FileNotFoundError(
            f"Forward Pilot 5 panel missing: {PILOT5_PANEL_FORWARD}. "
            f"Pilot 5 forward must run before Pilot 6 forward."
        )
    panel = pd.read_parquet(PILOT5_PANEL_FORWARD).reset_index(drop=True)
    required = ["signal_state", "is_early_pct", "is_early_atr", PRIMARY_OUTCOME]
    missing = [c for c in required if c not in panel.columns]
    if missing:
        raise RuntimeError(
            f"Forward Pilot 5 panel missing columns: {missing}. SPEC violation."
        )

    def _tier(state, is_early):
        if state != "extended":
            return TIER_OOS
        if pd.isna(is_early):
            return TIER_OOS
        return TIER_A if bool(is_early) else TIER_B

    panel["tier_pct"] = [
        _tier(s, e) for s, e in zip(panel["signal_state"], panel["is_early_pct"])
    ]
    panel["tier_atr"] = [
        _tier(s, e) for s, e in zip(panel["signal_state"], panel["is_early_atr"])
    ]
    return panel


def _format_summary_forward(
    final_verdict: str,
    notes: list[str],
    g1: dict,
    g2: dict,
    g7: dict,
    g6_g1: dict,
    g5_summary: dict,
    stability_df: pd.DataFrame,
    supp_df: pd.DataFrame,
    panel: pd.DataFrame,
    runtime_s: float,
    track_counts: dict,
) -> str:
    lines: list[str] = []
    lines.append("# ema_context Pilot 6 Layer A FORWARD — Run Summary")
    lines.append("")
    lines.append("- Spec: `memory/ema_context_forward_alignment_spec.md` LOCK candidate v2 (LOCKED 2026-05-05)")
    lines.append(f"- Run date (UTC): {pd.Timestamp.utcnow().isoformat()}")
    lines.append(f"- Runtime: {runtime_s:.1f}s")
    lines.append("- Source: forward Pilot 5 panel `output/ema_context_pilot5_panel_forward.parquet`")
    lines.append(f"- Forward panel rows: {len(panel):,}")
    lines.append(f"- Track distribution: {track_counts}")
    lines.append(f"- Tier rule (LOCKED): TIER_A_PAPER = extended ∧ earliness_score_pct ≤ {EARLY_THRESHOLD}")
    lines.append(f"- Primary outcome (LOCKED): `{PRIMARY_OUTCOME}`")
    lines.append("")

    ext = panel[panel["signal_state"] == "extended"]
    n_tier_a = int((ext["tier_pct"] == TIER_A).sum())
    n_tier_b = int((ext["tier_pct"] == TIER_B).sum())
    n_oos = int((panel["tier_pct"] == TIER_OOS).sum())
    lines.append("## Forward cohort census (pct primary)")
    lines.append("")
    lines.append(f"- TIER_A_PAPER: {n_tier_a:,}")
    lines.append(f"- TIER_B_WARNING: {n_tier_b:,}")
    lines.append(f"- BASELINE (TIER_A ∪ TIER_B): {n_tier_a + n_tier_b:,}")
    lines.append(f"- OUT_OF_SCOPE: {n_oos:,}")
    lines.append("")

    lines.append(f"## Overall Pilot 6 Layer A FORWARD verdict: **{final_verdict}**")
    lines.append("")
    for n in notes:
        lines.append(f"- {n}")
    lines.append("")

    lines.append("## G1 — TIER_A_PAPER vs BASELINE (uplift; primary pct)")
    lines.append("")
    _emit_gate_block(lines, g1, magnitude_floor_label=f"≥ {UPLIFT_FLOOR_R} R uplift")
    lines.append("")

    lines.append("## G2 — TIER_B_WARNING vs BASELINE (LABEL SANITY CHECK)")
    lines.append("")
    _emit_gate_block(lines, g2, magnitude_floor_label=None, sanity=True)
    lines.append("")

    lines.append("## G3 — Min-N (forward corpus)")
    lines.append("")
    lines.append(
        f"- TIER_A N={g1['n_a']:,} (≥ {MIN_N_TIER}: {'PASS' if g1['n_a'] >= MIN_N_TIER else 'FAIL'}) | "
        f"BASELINE N={g1['n_b']:,} (≥ {MIN_N_BASELINE}: {'PASS' if g1['n_b'] >= MIN_N_BASELINE else 'FAIL'}) | "
        f"TIER_B N={g2['n_a']:,} (≥ {MIN_N_TIER}: {'PASS' if g2['n_a'] >= MIN_N_TIER else 'FAIL'})"
    )
    lines.append("")

    lines.append("## G5 — Stability (TIER_A vs BASELINE within slope_tier × width_tier)")
    lines.append("")
    lines.append(
        f"- slope_tier slices (3): PASS={g5_summary['slope_pass']} / "
        f"INSUFFICIENT={g5_summary['slope_inconclusive']} / "
        f"FAIL_OR_INVERTED={g5_summary['slope_fail_or_inverted']}"
    )
    lines.append(
        f"- width_tier slices (3): PASS={g5_summary['width_pass']} / "
        f"INSUFFICIENT={g5_summary['width_inconclusive']} / "
        f"FAIL_OR_INVERTED={g5_summary['width_fail_or_inverted']}"
    )
    lines.append(f"- G5 status: **{g5_summary['g5_status']}** (sub-slice uplift floor {SUBSLICE_UPLIFT_FLOOR_R} R)")
    lines.append("")

    lines.append("## G6 — ATR mirror audit")
    lines.append("")
    _emit_gate_block(lines, g6_g1, magnitude_floor_label=f"≥ {UPLIFT_FLOOR_R} R uplift")
    lines.append("")

    lines.append("## G7 — Pilot 5 sanity replication (TIER_A vs TIER_B on this forward panel)")
    lines.append("")
    _emit_gate_block(lines, g7, magnitude_floor_label=f"≥ {PILOT5_SANITY_FLOOR_R} R separation")
    lines.append("")

    lines.append("## Supplementary outcomes (TIER_A vs TIER_B, extended cohort)")
    lines.append("")
    lines.append("| outcome | n_tier_a | n_tier_b | mean_tier_a | mean_tier_b | Δmean | CI | status |")
    lines.append("|---|---|---|---|---|---|---|---|")

    def _f(x, fmt=".4f"):
        return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else format(x, fmt)

    for _, r in supp_df.iterrows():
        ci_disp = (
            f"[{_f(r['ci_lo'])}, {_f(r['ci_hi'])}]"
            if r["ci_lo"] is not None else "—"
        )
        lines.append(
            f"| {r['outcome']} | {r['n_tier_a']} | {r['n_tier_b']} | "
            f"{_f(r['mean_tier_a'])} | {_f(r['mean_tier_b'])} | "
            f"{_f(r['delta_mean'])} | {ci_disp} | {r['status']} |"
        )
    lines.append("")

    lines.append("## PASS interpretation ceiling (verbatim from locked Pilot 6 spec)")
    lines.append("")
    lines.append("- Forward verdict does NOT change locked-research Pilot 6 verdict. Locked outputs remain byte-equal.")
    lines.append("- Layer B (paper execution shadow) remains OUT OF SCOPE; live trade gate STAYS CLOSED.")
    lines.append("- Forbidden interpretations: live entry filter / scanner ranking / position sizing / ML feature /")
    lines.append("  generalization beyond extended state / cross-pilot cascade without their own pre-reg.")
    lines.append("")
    return "\n".join(lines) + "\n"


# =============================================================================
# Main
# =============================================================================


def main() -> int:
    print("=== ema_context Pilot 6 Layer A FORWARD — single authorized run ===", flush=True)
    print("Spec: memory/ema_context_forward_alignment_spec.md (LOCKED 2026-05-05)", flush=True)
    t0 = time.time()

    print("[1/7] Building forward panel via Pilot 5 forward panel + tier tagging...", flush=True)
    panel = _build_panel_forward()
    track_counts = (
        panel.drop_duplicates(subset=CANDIDATE_KEY)["track"].value_counts().to_dict()
        if "track" in panel.columns else {}
    )
    ext = panel[panel["signal_state"] == "extended"]
    n_tier_a = int((ext["tier_pct"] == TIER_A).sum())
    n_tier_b = int((ext["tier_pct"] == TIER_B).sum())
    print(
        f"  TIER_A_PAPER={n_tier_a:,} | TIER_B_WARNING={n_tier_b:,} | "
        f"BASELINE={n_tier_a + n_tier_b:,}",
        flush=True,
    )
    print(f"  track distribution: {track_counts}", flush=True)

    # G1 — TIER_A vs BASELINE (uplift)
    print("[2/7] G1 (TIER_A vs BASELINE, pct primary)...", flush=True)
    ext_clean = panel[
        (panel["signal_state"] == "extended")
        & (panel["tier_pct"] != TIER_OOS)
    ].dropna(subset=[PRIMARY_OUTCOME, "tier_pct"])
    a_arr = ext_clean.loc[ext_clean["tier_pct"] == TIER_A, PRIMARY_OUTCOME].astype(float).to_numpy()
    baseline_arr = ext_clean[PRIMARY_OUTCOME].astype(float).to_numpy()
    g1 = _gate_pair(
        a_arr, baseline_arr,
        magnitude_floor=UPLIFT_FLOOR_R,
        require_positive=True,
        n_a_min=MIN_N_TIER,
        n_b_min=MIN_N_BASELINE,
    )
    print(f"  G1 verdict={g1['verdict']} Δuplift={g1.get('delta')}", flush=True)

    # G2 — TIER_B vs BASELINE label sanity
    print("[3/7] G2 (TIER_B vs BASELINE, label sanity)...", flush=True)
    b_arr = ext_clean.loc[ext_clean["tier_pct"] == TIER_B, PRIMARY_OUTCOME].astype(float).to_numpy()
    g2 = _gate_pair(
        b_arr, baseline_arr,
        magnitude_floor=0.0,
        require_positive=False,
        n_a_min=MIN_N_TIER,
        n_b_min=MIN_N_BASELINE,
    )
    print(f"  G2 sanity verdict={g2['verdict']}", flush=True)

    # G6 — atr mirror
    print("[4/7] G6 (atr mirror audit)...", flush=True)
    ext_atr = panel[
        (panel["signal_state"] == "extended")
        & (panel["tier_atr"] != TIER_OOS)
    ].dropna(subset=[PRIMARY_OUTCOME, "tier_atr"])
    a_atr = ext_atr.loc[ext_atr["tier_atr"] == TIER_A, PRIMARY_OUTCOME].astype(float).to_numpy()
    baseline_atr = ext_atr[PRIMARY_OUTCOME].astype(float).to_numpy()
    g6_g1 = _gate_pair(
        a_atr, baseline_atr,
        magnitude_floor=UPLIFT_FLOOR_R,
        require_positive=True,
        n_a_min=MIN_N_TIER,
        n_b_min=MIN_N_BASELINE,
    )
    print(f"  G6 verdict={g6_g1['verdict']}", flush=True)

    # G7 — Pilot 5 sanity replication
    print("[5/7] G7 (Pilot 5 sanity replication, TIER_A vs TIER_B)...", flush=True)
    g7 = _gate_pair(
        a_arr, b_arr,
        magnitude_floor=PILOT5_SANITY_FLOOR_R,
        require_positive=True,
        n_a_min=MIN_N_TIER,
        n_b_min=MIN_N_TIER,
    )
    print(f"  G7 verdict={g7['verdict']}", flush=True)

    # G5 — stability
    print("[6/7] G5 stability (slope_tier × width_tier)...", flush=True)
    stability_df, g5_summary = _g5_stability(panel, "tier_pct")
    print(
        f"  slope_tier PASS={g5_summary['slope_pass']}/3 | "
        f"width_tier PASS={g5_summary['width_pass']}/3 | G5 {g5_summary['g5_status']}",
        flush=True,
    )

    supp_df = _supplementary_table(panel, "tier_pct")
    final_verdict, notes = _final_verdict(g1, g2, g7, g6_g1, g5_summary["g5_status"])
    print(f"  FINAL VERDICT: {final_verdict}", flush=True)

    runtime_s = time.time() - t0

    print("[7/7] Writing forward outputs (atomic)...", flush=True)
    _atomic_write_parquet(panel, OUT_PANEL)
    _write_gates_main_csv(g1, g2, g6_g1, g7, OUT_GATES_MAIN)
    _atomic_write_csv(stability_df, OUT_STABILITY)
    _atomic_write_csv(supp_df, OUT_SUPPLEMENTARY)
    summary_text = _format_summary_forward(
        final_verdict, notes, g1, g2, g7, g6_g1,
        g5_summary, stability_df, supp_df, panel,
        runtime_s, track_counts,
    )
    _atomic_write_text(summary_text, OUT_SUMMARY)
    for p in [OUT_PANEL, OUT_GATES_MAIN, OUT_STABILITY, OUT_SUPPLEMENTARY, OUT_SUMMARY]:
        print(f"  Wrote: {p.name} ({p.stat().st_size:,} B)", flush=True)
    print(f"[Pilot 6 forward] Done in {runtime_s:.1f}s — verdict {final_verdict}", flush=True)

    summary_json = {
        "tool": "ema_context_pilot6_forward",
        "runtime_s": round(runtime_s, 3),
        "panel_rows": int(len(panel)),
        "track_counts": track_counts,
        "tier_a_count": n_tier_a,
        "tier_b_count": n_tier_b,
        "g1_verdict": g1["verdict"],
        "g2_verdict": g2["verdict"],
        "g6_verdict": g6_g1["verdict"],
        "g7_verdict": g7["verdict"],
        "g5_status": g5_summary["g5_status"],
        "final_verdict": final_verdict,
        "outputs": {
            "panel": str(OUT_PANEL.relative_to(PROJECT_ROOT)),
            "gates_main": str(OUT_GATES_MAIN.relative_to(PROJECT_ROOT)),
            "stability": str(OUT_STABILITY.relative_to(PROJECT_ROOT)),
            "supplementary": str(OUT_SUPPLEMENTARY.relative_to(PROJECT_ROOT)),
            "summary": str(OUT_SUMMARY.relative_to(PROJECT_ROOT)),
        },
    }
    print("PILOT6_FORWARD_SUMMARY_JSON_BEGIN")
    print(json.dumps(summary_json, default=str))
    print("PILOT6_FORWARD_SUMMARY_JSON_END")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
