"""ema_context Pilot 6 — Layer A single authorized run.

Spec: memory/ema_context_pilot6_spec.md (LOCKED 2026-05-03, onay 2026-05-03)

Layer A: paper-tier labelling validation.
Layer B (paper execution shadow) = OUT OF SCOPE for this run; requires
separate `paper_execution_v0_spec.md` drafting cycle.

Question: Within extended HB events, can the early-compression signal
(earliness_score_pct ≤ -6) be operationalized as a paper-only risk-tier
label that produces meaningful uplift over an undifferentiated baseline,
with stability across geometry slices, while preserving forbidden-action
ceilings?

Tier rule (LOCKED upfront):
  TIER_A_PAPER     = extended ∧ earliness_score_pct ≤ -6   (paper-eligible)
  TIER_B_WARNING   = extended ∧ earliness_score_pct >  -6   (paper-warning, NOT excluded)
  OUT_OF_SCOPE     = trigger / retest_bounce / NaN earliness

Anti-tweak (locked):
  - Pilot 5 panel reuse, no recompute earliness/outcomes/geometry
  - HB event parquet manifest hash check
  - realized_R_10d primary outcome LOCKED, no horizon sweep
  - uplift magnitude floor 0.05 R LOCKED
  - G7 Pilot 5 sanity floor 0.10 R LOCKED
  - p-threshold 0.001 LOCKED, no fallback
  - n_boot=2000, seed=7
  - min-N 200 (per tier) / 1000 (baseline) LOCKED
  - G2 = LABEL SANITY CHECK (TIER_B ⊂ BASELINE → not an independent test;
    used only to detect "warning label is broken" pathology)
  - G1 framing also notes that TIER_A ⊂ BASELINE; bootstrap CI still valid
    via independent resampling per group, but uplift is semi-overlap construct
  - PASS ceiling: drafting `paper_execution_v0_spec.md` becomes a candidate
    user-approval ask; NO automatic Layer B start, NO live trade, NO ML feature,
    NO scanner ranking, NO position sizing, NO production HTML emission
  - INSUFFICIENT ≠ FAIL
  - Single authorized run; bug → void+restart, no amend
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# =============================================================================
# LOCKED CONSTANTS — DO NOT MODIFY POST-RUN
# =============================================================================

HB_EVENT_PARQUET = PROJECT_ROOT / "output" / "horizontal_base_event_v1.parquet"
HB_EVENT_MANIFEST = PROJECT_ROOT / "output" / "horizontal_base_event_v1_manifest.json"
PILOT5_PANEL = PROJECT_ROOT / "output" / "ema_context_pilot5_panel.parquet"

OUT_PANEL = PROJECT_ROOT / "output" / "ema_context_pilot6_panel.parquet"
OUT_GATES_MAIN = PROJECT_ROOT / "output" / "ema_context_pilot6_gates_main.csv"
OUT_STABILITY = PROJECT_ROOT / "output" / "ema_context_pilot6_stability.csv"
OUT_SUPPLEMENTARY = PROJECT_ROOT / "output" / "ema_context_pilot6_supplementary.csv"
OUT_SUMMARY = PROJECT_ROOT / "output" / "ema_context_pilot6_summary.md"

# Tier rule thresholds (LOCKED)
EARLY_THRESHOLD = -6  # earliness_score ≤ -6 ⇒ TIER_A

# Outcome (LOCKED)
PRIMARY_OUTCOME = "realized_R_10d"
SUPPLEMENTARY_OUTCOMES = (
    "mfe_R_10d",
    "mae_R_10d",
    "failed_breakout_10d",
    "quality_20d",
    "time_to_MFE_10d",
)

# Statistical gates (LOCKED)
P_THRESHOLD = 0.001
BOOTSTRAP_N = 2000
BOOTSTRAP_SEED = 7
BOOTSTRAP_CI_LO_PCT = 2.5
BOOTSTRAP_CI_HI_PCT = 97.5

UPLIFT_FLOOR_R = 0.05  # G1 magnitude floor (TIER_A vs BASELINE)
SUBSLICE_UPLIFT_FLOOR_R = 0.025  # G5 sub-slice floor (half of main)
PILOT5_SANITY_FLOOR_R = 0.10  # G7 Pilot 5 (TIER_A vs TIER_B) replication floor
MIN_N_TIER = 200
MIN_N_BASELINE = 1000

# Manifest expectations
EXPECTED_SCANNER_VERSION = "1.4.0"
EXPECTED_HB_ROWS = 10470
EXPECTED_PILOT5_PANEL_ROWS = 10470

# Tier labels
TIER_A = "TIER_A_PAPER"
TIER_B = "TIER_B_WARNING"
TIER_OOS = "OUT_OF_SCOPE"

SLOPE_TIER_COL = "family__slope_tier"
WIDTH_TIER_COL = "width_tier_label"


# =============================================================================
# BOOT — fail-fast manifest check
# =============================================================================


def _validate_inputs() -> tuple[dict, int]:
    if not HB_EVENT_PARQUET.exists():
        raise FileNotFoundError(f"HB event parquet missing: {HB_EVENT_PARQUET}")
    if not HB_EVENT_MANIFEST.exists():
        raise FileNotFoundError(f"HB event manifest missing: {HB_EVENT_MANIFEST}")
    if not PILOT5_PANEL.exists():
        raise FileNotFoundError(
            f"Pilot 5 panel missing: {PILOT5_PANEL}. "
            f"SPEC violation — Pilot 6 reuses Pilot 5 panel, NO recompute. STOP."
        )

    manifest = json.loads(HB_EVENT_MANIFEST.read_text())
    sv = manifest.get("scanner_version")
    if sv != EXPECTED_SCANNER_VERSION:
        raise RuntimeError(
            f"HB manifest scanner_version mismatch: got {sv!r} != "
            f"{EXPECTED_SCANNER_VERSION!r}. SPEC violation."
        )
    rows = manifest.get("rows")
    if rows != EXPECTED_HB_ROWS:
        raise RuntimeError(
            f"HB manifest rows mismatch: got {rows!r} != {EXPECTED_HB_ROWS}. "
            f"SPEC violation."
        )

    p5_rows = pd.read_parquet(PILOT5_PANEL, columns=["event_id"]).shape[0]
    if p5_rows != EXPECTED_PILOT5_PANEL_ROWS:
        raise RuntimeError(
            f"Pilot 5 panel rows mismatch: got {p5_rows} != {EXPECTED_PILOT5_PANEL_ROWS}. "
            f"SPEC violation — panel may have been rebuilt."
        )

    return manifest, p5_rows


# =============================================================================
# Panel build (tier tagging on top of Pilot 5 panel)
# =============================================================================


def _build_panel() -> pd.DataFrame:
    panel = pd.read_parquet(PILOT5_PANEL).reset_index(drop=True)

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


# =============================================================================
# Statistical primitives
# =============================================================================


def _mannwhitney(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    res = stats.mannwhitneyu(a, b, alternative="two-sided")
    return float(res.statistic), float(res.pvalue)


def _bootstrap_mean_diff(
    a: np.ndarray, b: np.ndarray, n_boot: int, seed: int
) -> dict:
    """Bootstrap mean(a) - mean(b). Sign convention: positive ⇒ A higher."""
    rng = np.random.default_rng(seed)
    n_a = len(a)
    n_b = len(b)
    diffs = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sa = rng.choice(a, size=n_a, replace=True)
        sb = rng.choice(b, size=n_b, replace=True)
        diffs[i] = sa.mean() - sb.mean()
    point_estimate = float(a.mean() - b.mean())
    ci_lo = float(np.percentile(diffs, BOOTSTRAP_CI_LO_PCT))
    ci_hi = float(np.percentile(diffs, BOOTSTRAP_CI_HI_PCT))
    return {
        "n_boot": n_boot,
        "seed": seed,
        "point_estimate": point_estimate,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "ci_excludes_zero": (ci_lo > 0) or (ci_hi < 0),
    }


def _gate_pair(
    a: np.ndarray,
    b: np.ndarray,
    *,
    magnitude_floor: float,
    require_positive: bool,
    n_a_min: int = MIN_N_TIER,
    n_b_min: int = MIN_N_TIER,
) -> dict:
    """Generic gate evaluation: G1-style (Mann-Whitney + bootstrap + sign).

    require_positive=True: hypothesis-aligned positive Δ; significant negative → CLOSED inverted.
    require_positive=False (G2 sanity check): hypothesis-aligned Δ ≤ 0; significant positive → CLOSED contradiction.
    """
    n_a = int(len(a))
    n_b = int(len(b))

    if n_a < n_a_min or n_b < n_b_min:
        return {
            "n_a": n_a,
            "n_b": n_b,
            "mean_a": float(a.mean()) if n_a > 0 else None,
            "mean_b": float(b.mean()) if n_b > 0 else None,
            "u_stat": None,
            "p_value": None,
            "p_status": "INSUFFICIENT",
            "delta": None,
            "ci_lo": None,
            "ci_hi": None,
            "ci_excludes_zero": None,
            "magnitude_pass": None,
            "g_status": "INSUFFICIENT",
            "hypothesis_aligned": None,
            "sign_status": "INSUFFICIENT",
            "verdict": "INSUFFICIENT",
        }

    u_stat, p_value = _mannwhitney(a, b)
    p_pass = p_value < P_THRESHOLD

    boot = _bootstrap_mean_diff(a, b, BOOTSTRAP_N, BOOTSTRAP_SEED)
    delta = boot["point_estimate"]
    ci_excludes_zero = boot["ci_excludes_zero"]
    magnitude_pass = abs(delta) >= magnitude_floor

    if require_positive:
        hypothesis_aligned = delta > 0
        sig = p_pass and ci_excludes_zero and magnitude_pass
        if sig and hypothesis_aligned:
            verdict = "PASS"
            sign_status = "PASS"
        elif sig and not hypothesis_aligned:
            verdict = "CLOSED_HYPOTHESIS_INVERTED"
            sign_status = "INVERTED"
        else:
            verdict = "FAIL"
            sign_status = "FAIL" if not hypothesis_aligned else "OK_BUT_INSIGNIFICANT"
    else:
        # G2 label-sanity: hypothesis-aligned Δ ≤ 0 (TIER_B at or below baseline)
        # PASS = CI includes 0 OR significant Δ ≤ 0 (CI fully below zero)
        # CONTRADICTION = significant Δ > 0 (CI fully above zero)
        hypothesis_aligned = delta <= 0
        if ci_excludes_zero and delta > 0:
            verdict = "CLOSED_CONTRADICTION"
            sign_status = "CONTRADICTION_TIER_B_ABOVE_BASELINE"
        else:
            verdict = "PASS"
            sign_status = "PASS"

    return {
        "n_a": n_a,
        "n_b": n_b,
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "u_stat": u_stat,
        "p_value": p_value,
        "p_status": "PASS" if p_pass else "FAIL",
        "delta": delta,
        "ci_lo": boot["ci_lo"],
        "ci_hi": boot["ci_hi"],
        "ci_excludes_zero": ci_excludes_zero,
        "magnitude_pass": magnitude_pass,
        "g_status": "PASS" if (p_pass and ci_excludes_zero and magnitude_pass) else "FAIL",
        "hypothesis_aligned": hypothesis_aligned,
        "sign_status": sign_status,
        "verdict": verdict,
    }


# =============================================================================
# Cohort builders
# =============================================================================


def _arr(panel: pd.DataFrame, mask: pd.Series, col: str) -> np.ndarray:
    return panel.loc[mask].dropna(subset=[col])[col].astype(float).to_numpy()


# =============================================================================
# Stability check (G5)
# =============================================================================


def _g5_stability(panel: pd.DataFrame, tier_col: str) -> tuple[pd.DataFrame, dict]:
    """Re-run G1 (TIER_A vs BASELINE) within slope_tier and width_tier slices.

    Sub-slice uplift floor 0.025 R (half of main 0.05). Sub-min-N → INCONCLUSIVE.
    G5 PASS = ≥2/3 slope_tier OR ≥2/3 width_tier slices PASS hypothesis-aligned.
    """
    rows = []
    ext = panel[panel["signal_state"] == "extended"].dropna(subset=[PRIMARY_OUTCOME, tier_col]).copy()
    ext = ext[ext[tier_col] != TIER_OOS]

    def _slice_eval(label, slice_mask):
        sub = ext[slice_mask]
        a = sub.loc[sub[tier_col] == TIER_A, PRIMARY_OUTCOME].astype(float).to_numpy()
        baseline = sub[PRIMARY_OUTCOME].astype(float).to_numpy()
        gates = _gate_pair(
            a, baseline,
            magnitude_floor=SUBSLICE_UPLIFT_FLOOR_R,
            require_positive=True,
            n_a_min=MIN_N_TIER,
            n_b_min=MIN_N_BASELINE,
        )
        return {"slice_label": label, **gates}

    slope_results = []
    for st in ["flat", "mild", "loose"]:
        result = _slice_eval(f"slope_tier={st}", ext[SLOPE_TIER_COL] == st)
        slope_results.append(result)
        rows.append({"dimension": "slope_tier", **result})

    width_results = []
    for wt in ["low", "mid", "high"]:
        result = _slice_eval(f"width_tier={wt}", ext[WIDTH_TIER_COL] == wt)
        width_results.append(result)
        rows.append({"dimension": "width_tier", **result})

    df = pd.DataFrame(rows)

    slope_pass = sum(1 for r in slope_results if r["verdict"] == "PASS")
    slope_inc = sum(1 for r in slope_results if r["verdict"] == "INSUFFICIENT")
    slope_fail = sum(1 for r in slope_results if r["verdict"] in ("FAIL", "CLOSED_HYPOTHESIS_INVERTED"))
    width_pass = sum(1 for r in width_results if r["verdict"] == "PASS")
    width_inc = sum(1 for r in width_results if r["verdict"] == "INSUFFICIENT")
    width_fail = sum(1 for r in width_results if r["verdict"] in ("FAIL", "CLOSED_HYPOTHESIS_INVERTED"))

    slope_g5 = slope_pass >= 2
    width_g5 = width_pass >= 2
    g5_overall = slope_g5 or width_g5

    summary = {
        "slope_pass": slope_pass,
        "slope_inconclusive": slope_inc,
        "slope_fail_or_inverted": slope_fail,
        "width_pass": width_pass,
        "width_inconclusive": width_inc,
        "width_fail_or_inverted": width_fail,
        "slope_g5": slope_g5,
        "width_g5": width_g5,
        "g5_status": "PASS" if g5_overall else "FAIL",
    }
    return df, summary


# =============================================================================
# Supplementary descriptives
# =============================================================================


def _supplementary_table(panel: pd.DataFrame, tier_col: str) -> pd.DataFrame:
    rows = []
    ext = panel[panel["signal_state"] == "extended"].copy()
    for outcome in SUPPLEMENTARY_OUTCOMES:
        if outcome not in ext.columns:
            continue
        sub = ext.dropna(subset=[outcome, tier_col]).copy()
        a = sub.loc[sub[tier_col] == TIER_A, outcome].astype(float).to_numpy()
        b = sub.loc[sub[tier_col] == TIER_B, outcome].astype(float).to_numpy()
        n_a = len(a)
        n_b = len(b)
        if n_a < MIN_N_TIER or n_b < MIN_N_TIER:
            row = {
                "outcome": outcome,
                "n_tier_a": n_a,
                "n_tier_b": n_b,
                "mean_tier_a": float(a.mean()) if n_a > 0 else None,
                "mean_tier_b": float(b.mean()) if n_b > 0 else None,
                "delta_mean": (
                    float(a.mean() - b.mean()) if (n_a > 0 and n_b > 0) else None
                ),
                "ci_lo": None,
                "ci_hi": None,
                "ci_excludes_zero": None,
                "status": "INSUFFICIENT",
            }
        else:
            boot = _bootstrap_mean_diff(a, b, BOOTSTRAP_N, BOOTSTRAP_SEED)
            row = {
                "outcome": outcome,
                "n_tier_a": n_a,
                "n_tier_b": n_b,
                "mean_tier_a": float(a.mean()),
                "mean_tier_b": float(b.mean()),
                "delta_mean": boot["point_estimate"],
                "ci_lo": boot["ci_lo"],
                "ci_hi": boot["ci_hi"],
                "ci_excludes_zero": boot["ci_excludes_zero"],
                "status": "DESCRIPTIVE",
            }
        rows.append(row)
    return pd.DataFrame(rows)


# =============================================================================
# Decision tree
# =============================================================================


def _final_verdict(
    g1: dict, g2: dict, g7: dict, g6_g1: dict, g5_status: str
) -> tuple[str, list[str]]:
    notes: list[str] = []

    # G7 sanity must pass — panel corruption check, abort if broken
    if g7["verdict"] not in ("PASS", "INSUFFICIENT"):
        notes.append(
            f"G7 Pilot 5 sanity broken (verdict={g7['verdict']}, "
            f"Δmean(TIER_A−TIER_B)={g7['delta']}). Panel corruption suspected. ABORT — no verdict."
        )
        return "ABORT_PANEL_CORRUPTION", notes
    if g7["verdict"] == "INSUFFICIENT":
        notes.append("G7 sanity INSUFFICIENT — small N. Treat with caution.")

    # G1 hypothesis-inverted → CLOSED outright
    if g1["verdict"] == "CLOSED_HYPOTHESIS_INVERTED":
        notes.append(
            "G1 violation: TIER_A significantly BELOW baseline (hypothesis-inverted). "
            "→ CLOSED. No tier label rule. No 'sell-TIER_A' shortcut."
        )
        return "CLOSED", notes

    # G2 contradiction → CLOSED
    if g2["verdict"] == "CLOSED_CONTRADICTION":
        notes.append(
            "G2 sanity contradiction: TIER_B significantly ABOVE baseline. "
            "Warning label semantics broken. → CLOSED."
        )
        return "CLOSED", notes

    g1_pass = g1["verdict"] == "PASS"
    g2_pass = g2["verdict"] == "PASS"
    g5_pass = g5_status == "PASS"

    # G6 atr mirror not contradicting:
    atr_contradicts = g6_g1["verdict"] == "CLOSED_HYPOTHESIS_INVERTED"

    if g1_pass and g2_pass and g5_pass and not atr_contradicts:
        notes.append(
            "G1 PASS (TIER_A uplift over baseline) AND G2 PASS (TIER_B label sanity OK) "
            "AND G5 PASS (stability) AND G6 atr mirror non-contradicting → PASS."
        )
        if g6_g1["verdict"] == "PASS":
            notes.append("G6 NOTE: atr mirror replicates uplift — descriptive cross-normalization signal.")
        elif g6_g1["verdict"] == "INSUFFICIENT":
            notes.append("G6 NOTE: atr mirror INSUFFICIENT (sample-size only) — not a contradiction.")
        else:
            notes.append("G6 NOTE: atr mirror does not replicate (no inverted sign) — primary stands.")
        return "PASS", notes

    # PARTIAL path
    if g1_pass and not (g2_pass and g5_pass and not atr_contradicts):
        weak_parts = []
        if not g2_pass:
            weak_parts.append("G2 sanity not clean")
        if not g5_pass:
            weak_parts.append("G5 stability fail")
        if atr_contradicts:
            weak_parts.append("G6 atr mirror contradicts (HYPOTHESIS_INVERTED)")
        notes.append(
            f"G1 PASS but {'; '.join(weak_parts)} → PARTIAL. "
            f"paper_execution_v0_spec drafting NOT unlocked."
        )
        return "PARTIAL", notes

    notes.append(
        f"G1 verdict={g1['verdict']} — primary uplift gate not passed → CLOSED. "
        f"Pilot 5 PASS stands as descriptive timing claim only. No paper risk modifier."
    )
    return "CLOSED", notes


# =============================================================================
# Output writers
# =============================================================================


def _write_gates_main_csv(
    g1: dict, g2: dict, g6_g1: dict, g7: dict, path: Path
) -> None:
    rows = [
        {**g1, "gate_label": "G1_TIER_A_vs_BASELINE_pct"},
        {**g2, "gate_label": "G2_TIER_B_vs_BASELINE_pct_label_sanity"},
        {**g6_g1, "gate_label": "G6_TIER_A_vs_BASELINE_atr_audit"},
        {**g7, "gate_label": "G7_pilot5_sanity_TIER_A_vs_TIER_B_pct"},
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


# =============================================================================
# Summary writer
# =============================================================================


def _format_summary(
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
    hb_manifest: dict,
) -> str:
    lines: list[str] = []
    lines.append("# ema_context Pilot 6 — Layer A Run Summary")
    lines.append("")
    lines.append("- Spec: `memory/ema_context_pilot6_spec.md` LOCKED 2026-05-03")
    lines.append(f"- Run date: {pd.Timestamp.utcnow().isoformat()} (single authorized Layer A run)")
    lines.append(f"- Runtime: {runtime_s:.1f}s")
    lines.append(
        f"- HB event source: `output/horizontal_base_event_v1.parquet` "
        f"(scanner_version {hb_manifest.get('scanner_version')}, "
        f"frozen_at {hb_manifest.get('frozen_at')}, rows {hb_manifest.get('rows')})"
    )
    lines.append("- Pilot 5 panel reused: `output/ema_context_pilot5_panel.parquet`")
    lines.append(f"- Tier rule (LOCKED): TIER_A_PAPER = extended ∧ earliness_score_pct ≤ {EARLY_THRESHOLD}")
    lines.append(f"- TIER_B_WARNING = extended ∧ earliness_score_pct > {EARLY_THRESHOLD} (warning, NOT excluded)")
    lines.append(f"- OUT_OF_SCOPE = trigger / retest_bounce / NaN earliness")
    lines.append(f"- Primary outcome (LOCKED): `{PRIMARY_OUTCOME}`")
    lines.append("")

    lines.append("## Question (descriptive only)")
    lines.append("")
    lines.append(
        "Within extended HB events, can the early-compression signal be operationalized "
        "as a paper-only risk-tier label (paper-tier *labelling* validation only — "
        "Layer B paper execution shadow OUT OF SCOPE for this run)?"
    )
    lines.append("")

    # Tier counts
    ext = panel[panel["signal_state"] == "extended"]
    n_tier_a = int((ext["tier_pct"] == TIER_A).sum())
    n_tier_b = int((ext["tier_pct"] == TIER_B).sum())
    n_tier_oos_pct = int((panel["tier_pct"] == TIER_OOS).sum())
    lines.append("## Cohort census (pct primary)")
    lines.append("")
    lines.append(f"- TIER_A_PAPER: {n_tier_a:,}")
    lines.append(f"- TIER_B_WARNING: {n_tier_b:,}")
    lines.append(f"- BASELINE (TIER_A ∪ TIER_B): {n_tier_a + n_tier_b:,}")
    lines.append(f"- OUT_OF_SCOPE (trigger/retest_bounce/NaN earliness): {n_tier_oos_pct:,}")
    lines.append("")

    lines.append(f"## Overall Pilot 6 Layer A Verdict: **{final_verdict}**")
    lines.append("")
    for n in notes:
        lines.append(f"- {n}")
    lines.append("")

    lines.append("## G1 — TIER_A_PAPER vs BASELINE (uplift; primary pct)")
    lines.append("")
    lines.append(
        "**Note on construction**: TIER_A is a strict subset of BASELINE (all-extended). "
        "Bootstrap CI is computed via independent resampling per group and remains valid; "
        "the practical interpretation is *uplift if you only paper-traded TIER_A vs an undifferentiated extended cohort*."
    )
    lines.append("")
    _emit_gate_block(lines, g1, magnitude_floor_label=f"≥ {UPLIFT_FLOOR_R} R uplift")
    lines.append("")

    lines.append("## G2 — TIER_B_WARNING vs BASELINE (LABEL SANITY CHECK, NOT independence test)")
    lines.append("")
    lines.append(
        "**Construction caveat (per spec note)**: TIER_B is a strict subset of BASELINE — this is NOT "
        "a mathematically independent test. It is reported as a **label sanity check**: if the warning "
        "tier is significantly *above* baseline, the warning semantics are broken (CLOSED). "
        "Hypothesis-aligned Δmean(TIER_B − BASELINE) ≤ 0."
    )
    lines.append("")
    _emit_gate_block(lines, g2, magnitude_floor_label=None, sanity=True)
    lines.append("")

    lines.append("## G3 — Min-N")
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
        f"FAIL_OR_INVERTED={g5_summary['slope_fail_or_inverted']} → "
        f"{'≥2/3 PASS' if g5_summary['slope_g5'] else '<2/3 PASS'}"
    )
    lines.append(
        f"- width_tier slices (3): PASS={g5_summary['width_pass']} / "
        f"INSUFFICIENT={g5_summary['width_inconclusive']} / "
        f"FAIL_OR_INVERTED={g5_summary['width_fail_or_inverted']} → "
        f"{'≥2/3 PASS' if g5_summary['width_g5'] else '<2/3 PASS'}"
    )
    lines.append(f"- G5 status: **{g5_summary['g5_status']}** (sub-slice uplift floor {SUBSLICE_UPLIFT_FLOOR_R} R)")
    lines.append("")
    lines.append("### Stability table (per slice)")
    lines.append("")
    lines.append("| dimension | slice | n_tier_a | n_baseline | mean_tier_a | mean_baseline | Δuplift | CI_lo | CI_hi | p | verdict |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for _, r in stability_df.iterrows():
        def _f(x, fmt=".4f"):
            return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else format(x, fmt)
        lines.append(
            f"| {r['dimension']} | {r['slice_label']} | {r['n_a']} | {r['n_b']} | "
            f"{_f(r['mean_a'])} | {_f(r['mean_b'])} | {_f(r['delta'])} | "
            f"{_f(r['ci_lo'])} | {_f(r['ci_hi'])} | "
            f"{_f(r['p_value'], '.2e') if r['p_value'] is not None else '—'} | "
            f"{r['verdict']} |"
        )
    lines.append("")

    lines.append("## G6 — ATR mirror audit (G7-trap defense, audit-only)")
    lines.append("")
    lines.append(
        "Re-runs G1 with TIER_A defined via earliness_score_atr ≤ -6. "
        "ATR-only PASS does NOT count. ATR-contradiction (significant negative) → demote PASS to PARTIAL."
    )
    lines.append("")
    _emit_gate_block(lines, g6_g1, magnitude_floor_label=f"≥ {UPLIFT_FLOOR_R} R uplift")
    lines.append("")

    lines.append("## G7 — Pilot 5 sanity replication (TIER_A vs TIER_B on this panel)")
    lines.append("")
    lines.append(
        f"Sanity check: panel must reproduce Pilot 5 separation Δmean(TIER_A − TIER_B) ≥ {PILOT5_SANITY_FLOOR_R} R "
        "with CI excluding 0. Failure ⇒ panel corruption ⇒ ABORT no verdict."
    )
    lines.append("")
    _emit_gate_block(lines, g7, magnitude_floor_label=f"≥ {PILOT5_SANITY_FLOOR_R} R separation")
    lines.append("")

    lines.append("## Supplementary outcomes — descriptive context (TIER_A vs TIER_B, extended cohort)")
    lines.append("")
    lines.append(
        "Supplementary tests are NOT independent gates. Sign-mismatch with primary direction is logged below."
    )
    lines.append("")
    lines.append("| outcome | n_tier_a | n_tier_b | mean_tier_a | mean_tier_b | Δmean | CI | status |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for _, r in supp_df.iterrows():
        def _f(x, fmt=".4f"):
            return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else format(x, fmt)
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

    # CONTRADICTION detection on supplementary
    primary_sign = (
        np.sign(g1["delta"]) if g1["delta"] is not None else 0
    )
    contradictions = []
    for _, r in supp_df.iterrows():
        if r["status"] != "DESCRIPTIVE" or r["ci_excludes_zero"] is not True:
            continue
        if r["delta_mean"] is None:
            continue
        outcome = r["outcome"]
        delta_sign = np.sign(r["delta_mean"])
        positive_align = {
            "mfe_R_10d": True,
            "mae_R_10d": True,
            "failed_breakout_10d": False,
            "quality_20d": True,
            "time_to_MFE_10d": None,
        }.get(outcome, None)
        if positive_align is None:
            continue
        expected_sign = primary_sign if positive_align else -primary_sign
        if delta_sign != 0 and expected_sign != 0 and delta_sign != expected_sign:
            contradictions.append(outcome)

    if contradictions:
        lines.append("### CONTRADICTION flag (supplementary vs primary direction)")
        lines.append("")
        for c in contradictions:
            lines.append(f"- ⚠️ {c}: significant Δ in opposite direction from primary `{PRIMARY_OUTCOME}`")
        lines.append("")
    else:
        lines.append(
            "### CONTRADICTION flag: none — supplementary outcomes do not contradict primary direction."
        )
        lines.append("")

    # PASS interpretation ceiling
    lines.append("## PASS Interpretation — LOCKED CEILING")
    lines.append("")
    if final_verdict == "PASS":
        lines.append("**Allowed Layer A PASS claim (the entire allowed claim, descriptive only)**:")
        lines.append("")
        lines.append(
            f"> Per Pilot 6 Layer A, the TIER_A_PAPER label "
            f"(extended HB events with earliness_score_pct ≤ {EARLY_THRESHOLD}) produces "
            f"statistically and practically meaningful `{PRIMARY_OUTCOME}` uplift over an "
            f"all-extended baseline (descriptive Δmean ≥ {UPLIFT_FLOOR_R} R, p < {P_THRESHOLD}, "
            f"slope/width stability, atr mirror non-contradicting, Pilot 5 sanity replicates). "
            f"Tier label is a paper-only descriptive risk modifier candidate; it is NOT a live "
            f"entry filter, NOT a position sizing input, NOT a ranking score component, NOT "
            f"generalizable beyond extended state."
        )
        lines.append("")
        lines.append("**Allowed downstream actions on Layer A PASS**:")
        lines.append("- Drafting `memory/paper_execution_v0_spec.md` becomes a CANDIDATE user-approval ask.")
        lines.append("  - Drafting itself requires explicit user onay; PASS does NOT auto-start drafting.")
        lines.append("- Updating HB HTML note text (separate user-approval cycle).")
        lines.append("")
        lines.append("**Layer B (paper execution shadow) is NOT auto-unlocked.**")
        lines.append("**Live trade gate STAYS CLOSED.**")
        lines.append("")
    elif final_verdict == "PARTIAL":
        lines.append("**PARTIAL** — tier labelling validation incomplete; paper_execution_v0_spec drafting NOT unlocked.")
        lines.append("")
    elif final_verdict == "CLOSED":
        lines.append("**CLOSED** — tier label rule does not operationalize. Pilot 5 PASS stands as descriptive timing claim only. No paper risk modifier.")
        lines.append("")
    elif final_verdict == "ABORT_PANEL_CORRUPTION":
        lines.append("**ABORT** — G7 sanity broken; panel corruption suspected. Run is invalid. NO verdict.")
        lines.append("")

    lines.append("**Forbidden interpretations (locked ceiling, all verdicts)**:")
    forbidden = [
        "Live trade priority / scanner ranking elevation",
        "Live entry timing recommendation",
        "Live entry filter ('only buy TIER_A signals')",
        "Hard or soft EMA gate in any scanner / decision_engine",
        "EMA ML feature / earliness as ranking-scoring sort key / tier as classifier label",
        "Position sizing change (paper or live) — uniform sizing required in any future paper spec",
        "Generalization to trigger / retest_bounce / non-extended states",
        "Forward-looking signature claim",
        "Cross-pilot cascade (Pilot 7/8 by analogy without their own pre-reg)",
        "Re-running Pilot 6 with different threshold / outcome / tier definition / horizon",
        "Reopening Pilot 1/2/3/4/5 verdicts based on Pilot 6",
        "Removing Pilot 4 HTML note on Pilot 6 CLOSED",
        "Treating 'warning tier' as a directive to exclude TIER_B from any live workflow",
        "Auto-emission of TIER_A flag in production HTML / decision_engine report / NOX briefing without separate per-target user approval",
        "Skipping Layer B (paper execution shadow) and going straight to live trade",
    ]
    for f in forbidden:
        lines.append(f"- ❌ {f}")
    lines.append("")

    lines.append("## Anti-tweak Confirmation")
    lines.append("")
    confs = [
        f"Pilot 5 panel reused (rows {EXPECTED_PILOT5_PANEL_ROWS:,}) — no recompute earliness/outcomes/geometry",
        f"HB event parquet reused (frozen scanner_version 1.4.0) — no rebuild",
        f"Tier definitions LOCKED at earliness_score_pct {{≤ {EARLY_THRESHOLD}, > {EARLY_THRESHOLD}}}",
        f"Primary outcome `{PRIMARY_OUTCOME}` LOCKED — no horizon sweep",
        f"Uplift magnitude floor {UPLIFT_FLOOR_R} R LOCKED",
        f"Sub-slice uplift floor {SUBSLICE_UPLIFT_FLOOR_R} R LOCKED",
        f"G7 Pilot 5 sanity floor {PILOT5_SANITY_FLOOR_R} R LOCKED",
        f"p-threshold {P_THRESHOLD} LOCKED (no fallback)",
        f"Bootstrap n_boot={BOOTSTRAP_N}, seed={BOOTSTRAP_SEED}, group-independent resample",
        f"Min-N TIER {MIN_N_TIER} / BASELINE {MIN_N_BASELINE} LOCKED",
        f"G2 = LABEL SANITY CHECK (TIER_B ⊂ BASELINE → not independent test, used to detect warning-label pathology)",
        f"G6 ATR mirror = audit only; ATR-only PASS does NOT count",
        f"No new tier (no TIER_C, no quartile carving inside TIER_A, no top-decile)",
        f"No regime-conditional stratification, no multi-TF confirmation",
        f"No new feature in tier definition",
        f"Single authorized Layer A run — no post-hoc parameter change",
        f"Layer B (paper execution shadow) IS OUT OF SCOPE — NOT auto-unlocked by Layer A PASS",
        f"Live trade gate STAYS CLOSED regardless of Layer A or Layer B verdict",
        f"INSUFFICIENT ≠ FAIL",
        f"Pilot 5 ↔ Pilot 6 boundary kanonik — independent closures",
        f"Pilot 1/2/3/4/5 closure UNCHANGED regardless of Pilot 6 result",
        f"PASS ceiling: descriptive risk-tier label + paper_execution_v0_spec drafting candidate user ask only",
    ]
    for c in confs:
        lines.append(f"- {c}")
    lines.append("")

    return "\n".join(lines)


def _emit_gate_block(lines: list[str], gate: dict, magnitude_floor_label: str | None, sanity: bool = False) -> None:
    if gate.get("p_value") is None:
        lines.append(f"- N_a={gate.get('n_a')} / N_b={gate.get('n_b')} | verdict={gate.get('verdict')}")
        return
    lines.append(
        f"- N_a={gate['n_a']:,} / N_b={gate['n_b']:,}"
    )
    lines.append(
        f"- mean_a={gate['mean_a']:.4f} R / mean_b={gate['mean_b']:.4f} R | "
        f"Δ={gate['delta']:.4f} R"
    )
    lines.append(
        f"- Mann-Whitney U: stat={gate['u_stat']:.0f} p={gate['p_value']:.6e} → {gate['p_status']}"
    )
    lines.append(
        f"- Bootstrap CI=[{gate['ci_lo']:.4f}, {gate['ci_hi']:.4f}] | "
        f"excludes 0: {gate['ci_excludes_zero']}"
    )
    if magnitude_floor_label is not None:
        lines.append(f"- Magnitude check ({magnitude_floor_label}): {gate['magnitude_pass']}")
    if sanity:
        lines.append(
            f"- Sanity sign status: {gate['sign_status']} (hypothesis-aligned Δ ≤ 0; "
            f"significant positive ⇒ contradiction)"
        )
    else:
        lines.append(f"- Sign status: {gate['sign_status']} (hypothesis-aligned Δ > 0)")
    lines.append(f"- **Gate verdict: {gate['verdict']}**")


# =============================================================================
# Main
# =============================================================================


def main() -> int:
    t0 = time.time()
    print("[Pilot 6 Layer A] Validating inputs...", flush=True)
    hb_manifest, p5_rows = _validate_inputs()
    print(
        f"  HB manifest OK (scanner_version={hb_manifest.get('scanner_version')}, rows={hb_manifest.get('rows')})",
        flush=True,
    )
    print(f"  Pilot 5 panel rows={p5_rows}", flush=True)

    print("[Pilot 6 Layer A] Tagging tiers on Pilot 5 panel...", flush=True)
    panel = _build_panel()
    ext = panel[panel["signal_state"] == "extended"]
    n_tier_a = int((ext["tier_pct"] == TIER_A).sum())
    n_tier_b = int((ext["tier_pct"] == TIER_B).sum())
    print(f"  TIER_A_PAPER={n_tier_a:,} | TIER_B_WARNING={n_tier_b:,} | BASELINE={n_tier_a + n_tier_b:,}", flush=True)

    # ------------------------------------------------------------------
    # G1 — TIER_A vs BASELINE (pct primary, uplift)
    # ------------------------------------------------------------------
    print("[Pilot 6 Layer A] G1 (TIER_A vs BASELINE, pct primary)...", flush=True)
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
    print(f"  G1 verdict={g1['verdict']} | Δuplift={g1['delta']:.4f} R | p={g1['p_value']:.2e}", flush=True)

    # ------------------------------------------------------------------
    # G2 — TIER_B vs BASELINE (label sanity check, NOT independent test)
    # ------------------------------------------------------------------
    print("[Pilot 6 Layer A] G2 (TIER_B vs BASELINE, label sanity check)...", flush=True)
    b_arr = ext_clean.loc[ext_clean["tier_pct"] == TIER_B, PRIMARY_OUTCOME].astype(float).to_numpy()
    g2 = _gate_pair(
        b_arr, baseline_arr,
        magnitude_floor=0.0,  # no magnitude requirement on sanity gate
        require_positive=False,
        n_a_min=MIN_N_TIER,
        n_b_min=MIN_N_BASELINE,
    )
    print(f"  G2 sanity verdict={g2['verdict']} | Δ(TIER_B − BASELINE)={g2['delta']:.4f} R", flush=True)

    # ------------------------------------------------------------------
    # G6 — atr mirror audit (TIER_A_atr vs BASELINE_atr)
    # ------------------------------------------------------------------
    print("[Pilot 6 Layer A] G6 (atr mirror audit)...", flush=True)
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
    print(f"  G6 verdict={g6_g1['verdict']} | Δuplift_atr={g6_g1['delta']:.4f} R", flush=True)

    # ------------------------------------------------------------------
    # G7 — Pilot 5 sanity (TIER_A vs TIER_B replication)
    # ------------------------------------------------------------------
    print("[Pilot 6 Layer A] G7 (Pilot 5 sanity replication)...", flush=True)
    g7 = _gate_pair(
        a_arr, b_arr,
        magnitude_floor=PILOT5_SANITY_FLOOR_R,
        require_positive=True,
        n_a_min=MIN_N_TIER,
        n_b_min=MIN_N_TIER,
    )
    print(f"  G7 verdict={g7['verdict']} | Δ(TIER_A − TIER_B)={g7['delta']:.4f} R", flush=True)

    # ------------------------------------------------------------------
    # G5 — stability across slope/width slices
    # ------------------------------------------------------------------
    print("[Pilot 6 Layer A] G5 (stability)...", flush=True)
    stability_df, g5_summary = _g5_stability(panel, "tier_pct")
    print(
        f"  slope_tier PASS={g5_summary['slope_pass']}/3 | "
        f"width_tier PASS={g5_summary['width_pass']}/3 | G5 {g5_summary['g5_status']}",
        flush=True,
    )

    # ------------------------------------------------------------------
    # Supplementary
    # ------------------------------------------------------------------
    print("[Pilot 6 Layer A] Supplementary descriptives (TIER_A vs TIER_B)...", flush=True)
    supp_df = _supplementary_table(panel, "tier_pct")

    # ------------------------------------------------------------------
    # Final verdict
    # ------------------------------------------------------------------
    print("[Pilot 6 Layer A] Final verdict (decision tree)...", flush=True)
    final_verdict, notes = _final_verdict(g1, g2, g7, g6_g1, g5_summary["g5_status"])
    print(f"  FINAL VERDICT: {final_verdict}", flush=True)

    runtime_s = time.time() - t0

    # ------------------------------------------------------------------
    # Write outputs
    # ------------------------------------------------------------------
    print("[Pilot 6 Layer A] Writing outputs...", flush=True)
    panel.to_parquet(OUT_PANEL, index=False)
    _write_gates_main_csv(g1, g2, g6_g1, g7, OUT_GATES_MAIN)
    stability_df.to_csv(OUT_STABILITY, index=False)
    supp_df.to_csv(OUT_SUPPLEMENTARY, index=False)
    summary_text = _format_summary(
        final_verdict,
        notes,
        g1, g2, g7, g6_g1,
        g5_summary,
        stability_df,
        supp_df,
        panel,
        runtime_s,
        hb_manifest,
    )
    OUT_SUMMARY.write_text(summary_text)
    print(
        f"  Wrote: {OUT_PANEL.name} / {OUT_GATES_MAIN.name} / "
        f"{OUT_STABILITY.name} / {OUT_SUPPLEMENTARY.name} / {OUT_SUMMARY.name}",
        flush=True,
    )
    print(f"[Pilot 6 Layer A] Done in {runtime_s:.1f}s — verdict {final_verdict}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
