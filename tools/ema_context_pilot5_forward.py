"""ema_context Pilot 5 — FORWARD sibling (stable_event_key join, grow-aware).

Spec: memory/ema_context_forward_alignment_spec.md (LOCK candidate v2 → LOCKED 2026-05-05)

Forward output contract:
  - Replace locked Pilot 5 `_validate_inputs` row-count guard (10,470).
  - Replace locked Pilot 5 `_build_panel` (which positionally aligns HB↔earliness)
    with a stable_event_key join:
        forward_earliness  (P4 forward) ⋈ current_HB  on stable_event_key
  - Reuse all locked statistical primitives + stability + supplementary + final-verdict:
        _evaluate_pair, _split_cohort_by_state_and_early, _stability_check,
        _supplementary_table, _final_verdict,
        _write_gates_primary_csv, _write_gates_secondary_csv
  - Locked Pilot 5 outputs (6 files) remain BYTE-EQUAL pre/post.

Forward siblings emitted:
  output/ema_context_pilot5_panel_forward.parquet
  output/ema_context_pilot5_gates_primary_forward.csv
  output/ema_context_pilot5_gates_secondary_forward.csv
  output/ema_context_pilot5_stability_forward.csv
  output/ema_context_pilot5_supplementary_forward.csv
  output/ema_context_pilot5_summary_forward.md

Hard rules:
  - No positional fallback (key-only join).
  - No synthetic event injection.
  - No locked Pilot 5 script / output modification.
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

# Locked Pilot 5 kernels (NOT modified, library-import only)
from tools.ema_context_pilot5 import (  # type: ignore
    _evaluate_pair,
    _split_cohort_by_state_and_early,
    _stability_check,
    _supplementary_table,
    _final_verdict,
    _write_gates_primary_csv,
    _write_gates_secondary_csv,
    EARLY_THRESHOLD,
    PRIMARY_OUTCOME,
    SUPPLEMENTARY_OUTCOMES,
    P_THRESHOLD,
    BOOTSTRAP_N,
    BOOTSTRAP_SEED,
    MAGNITUDE_FLOOR_R,
    MIN_N_PER_GROUP,
    WIDTH_LO_Q,
    WIDTH_HI_Q,
    SLOPE_TIER_COL,
    WIDTH_PCTILE_COL,
)

# =============================================================================
# LOCKED CONSTANTS
# =============================================================================

CANDIDATE_KEY = ["ticker", "bar_date", "setup_family", "signal_type", "breakout_bar_date"]

HB_EVENT_PARQUET = PROJECT_ROOT / "output" / "horizontal_base_event_v1.parquet"
PILOT4_EARLINESS_FORWARD = (
    PROJECT_ROOT / "output" / "ema_context_pilot4_earliness_per_event_forward.parquet"
)

OUT_PANEL = PROJECT_ROOT / "output" / "ema_context_pilot5_panel_forward.parquet"
OUT_GATES_PRIMARY = PROJECT_ROOT / "output" / "ema_context_pilot5_gates_primary_forward.csv"
OUT_GATES_SECONDARY = PROJECT_ROOT / "output" / "ema_context_pilot5_gates_secondary_forward.csv"
OUT_STABILITY = PROJECT_ROOT / "output" / "ema_context_pilot5_stability_forward.csv"
OUT_SUPPLEMENTARY = PROJECT_ROOT / "output" / "ema_context_pilot5_supplementary_forward.csv"
OUT_SUMMARY = PROJECT_ROOT / "output" / "ema_context_pilot5_summary_forward.md"


# =============================================================================
# Helpers
# =============================================================================


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


def _normalize_key_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["bar_date"] = pd.to_datetime(out["bar_date"]).dt.normalize()
    out["breakout_bar_date"] = pd.to_datetime(out["breakout_bar_date"]).dt.normalize()
    for c in ("ticker", "setup_family", "signal_type"):
        out[c] = out[c].astype(str)
    return out


def _build_panel_forward() -> pd.DataFrame:
    """Stable_event_key join HB ⋈ Pilot 4 forward earliness; tag cohorts + width tiers."""
    if not HB_EVENT_PARQUET.exists():
        raise FileNotFoundError(f"Current HB missing: {HB_EVENT_PARQUET}")
    if not PILOT4_EARLINESS_FORWARD.exists():
        raise FileNotFoundError(
            f"Forward Pilot 4 earliness missing: {PILOT4_EARLINESS_FORWARD}. "
            f"Pilot 4 forward must run before Pilot 5 forward."
        )

    hb_cols = [
        *CANDIDATE_KEY,
        "signal_state",
        SLOPE_TIER_COL,
        WIDTH_PCTILE_COL,
        PRIMARY_OUTCOME,
        *SUPPLEMENTARY_OUTCOMES,
    ]
    hb = pd.read_parquet(HB_EVENT_PARQUET, columns=hb_cols)
    hb = _normalize_key_cols(hb)

    ep = pd.read_parquet(PILOT4_EARLINESS_FORWARD)
    # Forward earliness has stable_event_key cols attached (per pilot4_forward emission)
    missing = [c for c in CANDIDATE_KEY if c not in ep.columns]
    if missing:
        raise RuntimeError(
            f"Forward Pilot 4 earliness missing key columns: {missing}. "
            f"Pilot 4 forward must emit stable_event_key carry."
        )
    ep = _normalize_key_cols(ep)

    # Stable_event_key uniqueness
    n_hb_dup = int(hb.duplicated(subset=CANDIDATE_KEY).sum())
    n_ep_dup = int(ep.duplicated(subset=CANDIDATE_KEY).sum())
    if n_hb_dup > 0 or n_ep_dup > 0:
        raise RuntimeError(
            f"stable_key_duplicate_detected: hb_dup={n_hb_dup} ep_dup={n_ep_dup}. "
            f"SPEC §3.5 violated."
        )

    # Inner join on stable_event_key (matched_old + new_current both present
    # in current HB by construction of pilot 3/4 forward).
    panel = ep.merge(
        hb,
        on=CANDIDATE_KEY,
        how="inner",
        suffixes=("", "_hb"),
        validate="one_to_one",
    )
    if "signal_state_hb" in panel.columns:
        # signal_state must agree (locked HB row carries the canonical state).
        mismatch = int((panel["signal_state"] != panel["signal_state_hb"]).sum())
        if mismatch > 0:
            raise RuntimeError(
                f"signal_state mismatch between forward earliness and HB: {mismatch} rows. "
                f"Stable_event_key bridge inconsistent."
            )
        panel = panel.drop(columns=["signal_state_hb"])

    # Post-join row-count contract
    n_ep = len(ep)
    n_panel = len(panel)
    if n_panel != n_ep:
        raise RuntimeError(
            f"Forward panel join coverage mismatch: panel={n_panel} != forward_earliness={n_ep}. "
            f"Some forward earliness events have no current-HB row — Pilot 3/4 forward upstream "
            f"contract violated."
        )

    # Cohort flags (early_pct / early_atr) — same semantics as locked Pilot 5
    panel["is_early_pct"] = panel["earliness_score_pct"].apply(
        lambda v: True if pd.notna(v) and v <= EARLY_THRESHOLD
        else (False if pd.notna(v) else None)
    )
    panel["is_early_atr"] = panel["earliness_score_atr"].apply(
        lambda v: True if pd.notna(v) and v <= EARLY_THRESHOLD
        else (False if pd.notna(v) else None)
    )

    # Width terciles on extended cohort with non-null pct earliness — identical to locked
    ext_mask = (panel["signal_state"] == "extended") & panel["earliness_score_pct"].notna()
    width_vals = panel.loc[ext_mask, WIDTH_PCTILE_COL].dropna()
    if len(width_vals) == 0:
        raise RuntimeError("No extended events with width_pctile — cannot compute terciles.")
    w_lo = float(width_vals.quantile(WIDTH_LO_Q))
    w_hi = float(width_vals.quantile(WIDTH_HI_Q))

    def _wt(v):
        if pd.isna(v):
            return None
        if v <= w_lo:
            return "low"
        if v <= w_hi:
            return "mid"
        return "high"

    panel["width_tier_label"] = panel[WIDTH_PCTILE_COL].apply(_wt)
    panel.attrs["width_tier_breaks"] = {
        "lo_q": WIDTH_LO_Q, "hi_q": WIDTH_HI_Q, "lo_v": w_lo, "hi_v": w_hi,
    }
    return panel


# =============================================================================
# Summary writer (forward — descriptive of forward run)
# =============================================================================


def _format_summary_forward(
    final_verdict: str,
    notes: list[str],
    primary_pct: dict,
    primary_atr: dict,
    g5_summary: dict,
    stability_df: pd.DataFrame,
    secondaries: dict,
    supp_df: pd.DataFrame,
    panel: pd.DataFrame,
    runtime_s: float,
    track_counts: dict,
) -> str:
    lines: list[str] = []
    lines.append("# ema_context Pilot 5 FORWARD — Run Summary")
    lines.append("")
    lines.append("- Spec: `memory/ema_context_forward_alignment_spec.md` LOCK candidate v2 (LOCKED 2026-05-05)")
    lines.append(f"- Run date (UTC): {pd.Timestamp.utcnow().isoformat()}")
    lines.append(f"- Runtime: {runtime_s:.1f}s")
    lines.append(f"- HB event source: `output/horizontal_base_event_v1.parquet` (current; forward join on stable_event_key)")
    lines.append("- Pilot 4 forward earliness reused: `output/ema_context_pilot4_earliness_per_event_forward.parquet`")
    lines.append(f"- Forward panel rows: {len(panel):,} / unique stable_event_key: "
                 f"{panel.drop_duplicates(subset=CANDIDATE_KEY).shape[0]:,}")
    lines.append(f"- Track distribution: {track_counts}")
    lines.append(f"- Early threshold (LOCKED): earliness_score_pct ≤ {EARLY_THRESHOLD}")
    lines.append(f"- Primary outcome (LOCKED): `{PRIMARY_OUTCOME}`")
    lines.append("")

    lines.append(f"## Overall Pilot 5 FORWARD verdict: **{final_verdict}**")
    lines.append("")
    for n in notes:
        lines.append(f"- {n}")
    lines.append("")

    lines.append("## Primary (extended early vs extended non-early) — pct primary")
    lines.append("")
    lines.append(
        f"- N_early={primary_pct['n_a']:,} / N_non_early={primary_pct['n_b']:,} "
        f"(min-N {MIN_N_PER_GROUP}: {primary_pct['g3_status']})"
    )
    if primary_pct["mean_a"] is not None:
        lines.append(
            f"- mean(early)={primary_pct['mean_a']:.4f} R / "
            f"mean(non_early)={primary_pct['mean_b']:.4f} R"
        )
    if primary_pct["g1_p_value"] is not None:
        lines.append(
            f"- G1 Mann-Whitney p={primary_pct['g1_p_value']:.6e} → {primary_pct['g1_status']} "
            f"(threshold p < {P_THRESHOLD})"
        )
        lines.append(
            f"- G2 Bootstrap (n_boot={BOOTSTRAP_N}, seed={BOOTSTRAP_SEED}): "
            f"point={primary_pct['g2_point_estimate']:.4f} R "
            f"CI=[{primary_pct['g2_ci_lo']:.4f}, {primary_pct['g2_ci_hi']:.4f}] → "
            f"{primary_pct['g2_status']}"
        )
    lines.append(
        f"- G4 hypothesis_aligned={primary_pct['g4_hypothesis_aligned']} → {primary_pct['g4_status']}"
    )
    lines.append("")

    lines.append("## Audit (atr mirror — G6 G7-trap defense)")
    lines.append("")
    lines.append(
        f"- N_early={primary_atr['n_a']:,} / N_non_early={primary_atr['n_b']:,} (min-N: {primary_atr['g3_status']})"
    )
    if primary_atr["g1_p_value"] is not None:
        lines.append(
            f"- G1 p={primary_atr['g1_p_value']:.6e} → {primary_atr['g1_status']}; "
            f"G2 point={primary_atr['g2_point_estimate']:.4f} R "
            f"CI=[{primary_atr['g2_ci_lo']:.4f}, {primary_atr['g2_ci_hi']:.4f}] → {primary_atr['g2_status']}; "
            f"G4 aligned={primary_atr['g4_hypothesis_aligned']} → {primary_atr['g4_status']}"
        )
    lines.append("- ATR-only PASS does NOT count as Pilot 5 PASS (G7-trap defense).")
    lines.append("")

    lines.append("## G5 Stability — slope/width slices (extended cohort)")
    lines.append("")
    lines.append(
        f"- slope_tier slices (3): PASS={g5_summary['slope_pass']} / "
        f"INSUFFICIENT={g5_summary['slope_inconclusive']} / "
        f"FAIL_OR_INVERTED={g5_summary['slope_fail_or_inverted']} → "
        f"{'≥2/3 PASS' if g5_summary['slope_g5'] else '<2/3 PASS'}"
    )
    lines.append(
        f"- width_tier slices (3): PASS={g5_summary['width_pass']} / "
        f"INSUFFICIENT={g5_summary['width_inconclusive']} / "
        f"FAIL_OR_INVERTED={g5_summary['width_fail_or_inverted']} → "
        f"{'≥2/3 PASS' if g5_summary['width_g5'] else '<2/3 PASS'}"
    )
    lines.append(f"- G5 status: **{g5_summary['g5_status']}**")
    lines.append("")

    lines.append("## Secondary cohorts (descriptive support only)")
    lines.append("")
    lines.append("| cohort | n_early | n_non_early | Δmean | CI | p | verdict |")
    lines.append("|---|---|---|---|---|---|---|")

    def _f(x, fmt=".4f"):
        return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else format(x, fmt)

    for cohort_name, gates in secondaries.items():
        ci_disp = (
            f"[{_f(gates['g2_ci_lo'])}, {_f(gates['g2_ci_hi'])}]"
            if gates["g2_ci_lo"] is not None else "—"
        )
        p_disp = _f(gates["g1_p_value"], ".2e") if gates["g1_p_value"] is not None else "—"
        lines.append(
            f"| {cohort_name} | {gates['n_a']} | {gates['n_b']} | "
            f"{_f(gates['g2_point_estimate'])} | {ci_disp} | {p_disp} | {gates['verdict']} |"
        )
    lines.append("")

    lines.append("## Supplementary outcomes (descriptive — extended cohort)")
    lines.append("")
    lines.append("| outcome | n_early | n_non_early | mean_early | mean_non_early | Δmean | CI | status |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for _, r in supp_df.iterrows():
        ci_disp = (
            f"[{_f(r['ci_lo'])}, {_f(r['ci_hi'])}]"
            if r["ci_lo"] is not None else "—"
        )
        lines.append(
            f"| {r['outcome']} | {r['n_early']} | {r['n_non_early']} | "
            f"{_f(r['mean_early'])} | {_f(r['mean_non_early'])} | "
            f"{_f(r['delta_mean'])} | {ci_disp} | {r['status']} |"
        )
    lines.append("")

    lines.append("## PASS interpretation ceiling (verbatim from locked Pilot 5 spec)")
    lines.append("")
    lines.append("- Forbidden interpretations regardless of forward verdict: live trading / hard or soft EMA gate /")
    lines.append("  ML feature / ranking-scoring / live entry timing / position sizing / forward-looking signature.")
    lines.append("- Allowed PASS claim is descriptive only.")
    lines.append("- This forward run does NOT change locked-research Pilot 5 verdict; locked outputs remain byte-equal.")
    lines.append("")

    return "\n".join(lines) + "\n"


# =============================================================================
# Main
# =============================================================================


def main() -> int:
    print("=== ema_context Pilot 5 FORWARD — single authorized run ===", flush=True)
    print("Spec: memory/ema_context_forward_alignment_spec.md (LOCKED 2026-05-05)", flush=True)
    t0 = time.time()

    print("[1/6] Building forward panel via stable_event_key join (HB ⋈ Pilot 4 forward earliness)...", flush=True)
    panel = _build_panel_forward()
    track_counts = (
        panel.groupby(CANDIDATE_KEY)["track"].first().value_counts().to_dict()
        if "track" in panel.columns else {}
    )
    wt = panel.attrs["width_tier_breaks"]
    print(f"  Panel rows: {len(panel):,}", flush=True)
    print(
        f"  width_tier breaks (extended cohort, q={wt['lo_q']}/{wt['hi_q']}): "
        f"lo={wt['lo_v']:.4f} hi={wt['hi_v']:.4f}",
        flush=True,
    )
    print(f"  track distribution: {track_counts}", flush=True)

    print("[2/6] Primary cohort (extended early vs non-early) on pct primary...", flush=True)
    primary_a, primary_b = _split_cohort_by_state_and_early(
        panel,
        state_filter=lambda df: df["signal_state"] == "extended",
        early_col="is_early_pct",
    )
    primary_pct = _evaluate_pair(primary_a, primary_b)
    print(
        f"  N_early={primary_pct['n_a']} / N_non_early={primary_pct['n_b']} | "
        f"verdict={primary_pct['verdict']}",
        flush=True,
    )

    print("[3/6] G6 audit on atr mirror...", flush=True)
    atr_a, atr_b = _split_cohort_by_state_and_early(
        panel,
        state_filter=lambda df: df["signal_state"] == "extended",
        early_col="is_early_atr",
    )
    primary_atr = _evaluate_pair(atr_a, atr_b)
    print(
        f"  ATR audit verdict={primary_atr['verdict']} "
        f"(N_early={primary_atr['n_a']}, N_non_early={primary_atr['n_b']})",
        flush=True,
    )

    print("[4/6] G5 stability (slope_tier × width_tier)...", flush=True)
    stability_df, g5_summary = _stability_check(panel, "is_early_pct")
    print(
        f"  slope_tier PASS={g5_summary['slope_pass']}/3 | "
        f"width_tier PASS={g5_summary['width_pass']}/3 | G5 {g5_summary['g5_status']}",
        flush=True,
    )

    print("[5/6] Secondary cohorts (trigger / retest_bounce / pooled)...", flush=True)
    sec_trigger_a, sec_trigger_b = _split_cohort_by_state_and_early(
        panel, lambda df: df["signal_state"] == "trigger", "is_early_pct"
    )
    sec_retest_a, sec_retest_b = _split_cohort_by_state_and_early(
        panel, lambda df: df["signal_state"] == "retest_bounce", "is_early_pct"
    )
    sec_pooled_a, sec_pooled_b = _split_cohort_by_state_and_early(
        panel,
        lambda df: df["signal_state"].isin(["trigger", "retest_bounce"]),
        "is_early_pct",
    )
    sec_trigger = _evaluate_pair(sec_trigger_a, sec_trigger_b)
    sec_retest = _evaluate_pair(sec_retest_a, sec_retest_b)
    sec_pooled = _evaluate_pair(sec_pooled_a, sec_pooled_b)
    print(
        f"  trigger={sec_trigger['verdict']} retest={sec_retest['verdict']} pooled={sec_pooled['verdict']}",
        flush=True,
    )

    secondaries = {
        "trigger": sec_trigger,
        "retest_bounce": sec_retest,
        "pooled_trigger_or_retest": sec_pooled,
    }

    supp_df = _supplementary_table(panel, "is_early_pct")
    final_verdict, notes = _final_verdict(
        primary_pct, primary_atr, g5_summary["g5_status"], secondaries
    )
    print(f"  FINAL VERDICT: {final_verdict}", flush=True)

    runtime_s = time.time() - t0

    print("[6/6] Writing forward outputs (atomic)...", flush=True)
    _atomic_write_parquet(panel, OUT_PANEL)
    _write_gates_primary_csv(primary_pct, primary_atr, OUT_GATES_PRIMARY)
    _write_gates_secondary_csv(sec_trigger, sec_retest, sec_pooled, OUT_GATES_SECONDARY)
    _atomic_write_csv(stability_df, OUT_STABILITY)
    _atomic_write_csv(supp_df, OUT_SUPPLEMENTARY)
    summary_text = _format_summary_forward(
        final_verdict, notes, primary_pct, primary_atr,
        g5_summary, stability_df, secondaries, supp_df, panel,
        runtime_s, track_counts,
    )
    _atomic_write_text(summary_text, OUT_SUMMARY)

    for p in [
        OUT_PANEL, OUT_GATES_PRIMARY, OUT_GATES_SECONDARY,
        OUT_STABILITY, OUT_SUPPLEMENTARY, OUT_SUMMARY,
    ]:
        print(f"  Wrote: {p.name} ({p.stat().st_size:,} B)", flush=True)
    print(f"[Pilot 5 forward] Done in {runtime_s:.1f}s — verdict {final_verdict}", flush=True)

    summary_json = {
        "tool": "ema_context_pilot5_forward",
        "runtime_s": round(runtime_s, 3),
        "panel_rows": int(len(panel)),
        "track_counts": track_counts,
        "primary_pct_verdict": primary_pct["verdict"],
        "primary_atr_verdict": primary_atr["verdict"],
        "g5_status": g5_summary["g5_status"],
        "secondary_verdicts": {k: v["verdict"] for k, v in secondaries.items()},
        "final_verdict": final_verdict,
        "outputs": {
            "panel": str(OUT_PANEL.relative_to(PROJECT_ROOT)),
            "gates_primary": str(OUT_GATES_PRIMARY.relative_to(PROJECT_ROOT)),
            "gates_secondary": str(OUT_GATES_SECONDARY.relative_to(PROJECT_ROOT)),
            "stability": str(OUT_STABILITY.relative_to(PROJECT_ROOT)),
            "supplementary": str(OUT_SUPPLEMENTARY.relative_to(PROJECT_ROOT)),
            "summary": str(OUT_SUMMARY.relative_to(PROJECT_ROOT)),
        },
    }
    print("PILOT5_FORWARD_SUMMARY_JSON_BEGIN")
    print(json.dumps(summary_json, default=str))
    print("PILOT5_FORWARD_SUMMARY_JSON_END")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
