"""ema_context Pilot 5 — single authorized run.

Spec: memory/ema_context_pilot5_spec.md (LOCKED 2026-05-03, onay 2026-05-03)

EMA-HB outcome separation / descriptive-to-risk bridge.
Fresh, ayrı pre-reg — Pilot 4 rescue DEĞİL.

Question: Within extended HB events, do early-compression events
(earliness_score_pct ≤ -6) show frozen-outcome separation
(realized_R_10d) against non-early extended events?

Anti-tweak (locked):
  - Pilot 4 earliness panel reuse (NO recompute earliness)
  - HB event parquet reuse (NO recompute outcomes)
  - Early threshold ≤ -6 LOCKED (Pilot 3 boundary, not Pilot 4 mode)
  - Primary outcome realized_R_10d LOCKED (single horizon, no sweep)
  - Magnitude floor |Δmean| ≥ 0.10 R LOCKED
  - Mann-Whitney p < 0.001 LOCKED (no fallback)
  - Bootstrap n_boot=2000 seed=7
  - Min-N 200 per cohort LOCKED
  - Primary cohort within extended only LOCKED
  - Secondary trigger/retest/pooled descriptive only — cannot rescue primary into PASS
  - G4 sign violation (significant negative) → CLOSED, not PASS (no inverted-trade-rule shortcut)
  - G5 stability: ≥2/3 slope_tier OR ≥2/3 width_tier slices hypothesis-aligned
  - G6 atr mirror = audit only; atr-only PASS does NOT count
  - PASS ceiling: descriptive HTML note + Pilot 6 paper-only spec drafting
    (NO live trading / gate / ML / ranking / scoring / entry timing / position sizing)
  - INSUFFICIENT ≠ FAIL
  - Bug → void run + restart, NO amend
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
PILOT4_EARLINESS = PROJECT_ROOT / "output" / "ema_context_pilot4_earliness_per_event.parquet"

OUT_PANEL = PROJECT_ROOT / "output" / "ema_context_pilot5_panel.parquet"
OUT_GATES_PRIMARY = PROJECT_ROOT / "output" / "ema_context_pilot5_gates_primary.csv"
OUT_GATES_SECONDARY = PROJECT_ROOT / "output" / "ema_context_pilot5_gates_secondary.csv"
OUT_STABILITY = PROJECT_ROOT / "output" / "ema_context_pilot5_stability.csv"
OUT_SUPPLEMENTARY = PROJECT_ROOT / "output" / "ema_context_pilot5_supplementary.csv"
OUT_SUMMARY = PROJECT_ROOT / "output" / "ema_context_pilot5_summary.md"

# Cohort threshold (LOCKED — Pilot 3 boundary, not Pilot 4 mode)
EARLY_THRESHOLD = -6  # earliness_score ≤ -6 ⇒ early

# Outcome (LOCKED — single horizon, no sweep)
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
MAGNITUDE_FLOOR_R = 0.10
MIN_N_PER_GROUP = 200

# Stability — slice fractions
WIDTH_LO_Q = 0.333
WIDTH_HI_Q = 0.666

# Manifest expectations
EXPECTED_SCANNER_VERSION = "1.4.0"
EXPECTED_HB_ROWS = 10470
EXPECTED_EARLINESS_ROWS = 10470

# Pilot 4 surviving sample filter (mirror Pilot 4 ≥25/31 non-null)
NON_NULL_MIN_PER_EVENT = 25  # only meaningful via earliness null check (NaN ⇒ filtered Pilot 4)

# Geometry slice columns
SLOPE_TIER_COL = "family__slope_tier"
WIDTH_PCTILE_COL = "family__channel_width_pctile_252"

# Cohort labels
COHORT_PRIMARY_A = "extended_early"
COHORT_PRIMARY_B = "extended_non_early"


# =============================================================================
# BOOT — fail-fast manifest check
# =============================================================================


def _validate_inputs() -> tuple[dict, int, int]:
    if not HB_EVENT_PARQUET.exists():
        raise FileNotFoundError(f"HB event parquet missing: {HB_EVENT_PARQUET}")
    if not HB_EVENT_MANIFEST.exists():
        raise FileNotFoundError(f"HB event manifest missing: {HB_EVENT_MANIFEST}")
    if not PILOT4_EARLINESS.exists():
        raise FileNotFoundError(
            f"Pilot 4 earliness panel missing: {PILOT4_EARLINESS}. "
            f"SPEC violation — Pilot 5 reuses Pilot 4 panel, NO recompute. STOP."
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

    ep_rows = pd.read_parquet(PILOT4_EARLINESS, columns=["event_id"]).shape[0]
    if ep_rows != EXPECTED_EARLINESS_ROWS:
        raise RuntimeError(
            f"Pilot 4 earliness panel rows mismatch: got {ep_rows} != "
            f"{EXPECTED_EARLINESS_ROWS}. SPEC violation — panel may have been rebuilt."
        )

    return manifest, rows, ep_rows


# =============================================================================
# Panel build
# =============================================================================


def _build_panel() -> pd.DataFrame:
    """Join HB event outcomes + Pilot 4 earliness positionally."""
    hb_cols = [
        "ticker",
        "bar_date",
        "signal_state",
        SLOPE_TIER_COL,
        WIDTH_PCTILE_COL,
        PRIMARY_OUTCOME,
        *SUPPLEMENTARY_OUTCOMES,
    ]
    hb = pd.read_parquet(HB_EVENT_PARQUET, columns=hb_cols).reset_index(drop=True)
    ep = pd.read_parquet(PILOT4_EARLINESS).reset_index(drop=True)

    # Positional alignment audit (mandatory abort on mismatch)
    if len(hb) != len(ep):
        raise RuntimeError(
            f"Positional length mismatch: HB={len(hb)} ep={len(ep)}. SPEC violation."
        )
    t_mism = int((hb["ticker"].values != ep["ticker"].values).sum())
    s_mism = int((hb["signal_state"].values != ep["signal_state"].values).sum())
    if t_mism > 0 or s_mism > 0:
        raise RuntimeError(
            f"Positional alignment fail: ticker_mismatch={t_mism} state_mismatch={s_mism}. "
            f"SPEC violation."
        )

    panel = pd.concat(
        [
            hb,
            ep[[
                "event_id",
                "n_offsets_nonnull_pct",
                "n_offsets_nonnull_atr",
                "earliness_score_pct",
                "earliness_score_atr",
            ]].reset_index(drop=True),
        ],
        axis=1,
    )

    # Tag cohort_primary, cohort_secondary_*, early flags
    panel["is_early_pct"] = panel["earliness_score_pct"].apply(
        lambda v: True if pd.notna(v) and v <= EARLY_THRESHOLD else (False if pd.notna(v) else None)
    )
    panel["is_early_atr"] = panel["earliness_score_atr"].apply(
        lambda v: True if pd.notna(v) and v <= EARLY_THRESHOLD else (False if pd.notna(v) else None)
    )

    # Width tier — terciles computed on extended cohort with non-null pct earliness
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
    panel.attrs["width_tier_breaks"] = {"lo_q": WIDTH_LO_Q, "hi_q": WIDTH_HI_Q, "lo_v": w_lo, "hi_v": w_hi}
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
    """Bootstrap mean(a) - mean(b). Sign convention: positive ⇒ A higher.

    For Pilot 5 primary: a=early, b=non_early. Positive ⇒ early outcome higher (hypothesis-aligned).
    """
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


def _evaluate_pair(
    a: np.ndarray, b: np.ndarray, *, magnitude_floor: float = MAGNITUDE_FLOOR_R
) -> dict:
    """Run G1 (Mann-Whitney) + G2 (bootstrap) + G3 (min-N) + G4 (sign).

    Returns a dict with all gate readings. Sign convention: positive Δ = a higher than b.
    """
    n_a = int(len(a))
    n_b = int(len(b))

    g3_pass = (n_a >= MIN_N_PER_GROUP) and (n_b >= MIN_N_PER_GROUP)
    if not g3_pass:
        return {
            "n_a": n_a,
            "n_b": n_b,
            "mean_a": float(a.mean()) if n_a > 0 else None,
            "mean_b": float(b.mean()) if n_b > 0 else None,
            "median_a": float(np.median(a)) if n_a > 0 else None,
            "median_b": float(np.median(b)) if n_b > 0 else None,
            "g1_u_stat": None,
            "g1_p_value": None,
            "g1_status": "INSUFFICIENT",
            "g2_point_estimate": None,
            "g2_ci_lo": None,
            "g2_ci_hi": None,
            "g2_ci_excludes_zero": None,
            "g2_magnitude_pass": None,
            "g2_status": "INSUFFICIENT",
            "g3_status": "INSUFFICIENT",
            "g4_hypothesis_aligned": None,
            "g4_status": "INSUFFICIENT",
            "verdict": "INSUFFICIENT",
        }

    # G1
    u_stat, p_value = _mannwhitney(a, b)
    g1_pass = p_value < P_THRESHOLD

    # G2
    boot = _bootstrap_mean_diff(a, b, BOOTSTRAP_N, BOOTSTRAP_SEED)
    g2_magnitude_pass = abs(boot["point_estimate"]) >= magnitude_floor
    g2_pass = boot["ci_excludes_zero"] and g2_magnitude_pass

    # G4 — sign
    hypothesis_aligned = boot["point_estimate"] > 0
    # Significant negative direction = HYPOTHESIS-INVERTED PASS-LIKE
    g4_status = (
        "PASS" if hypothesis_aligned
        else ("HYPOTHESIS_INVERTED" if (g1_pass and g2_pass) else "FAIL")
    )

    if g1_pass and g2_pass and hypothesis_aligned:
        verdict = "PASS"
    elif g1_pass and g2_pass and not hypothesis_aligned:
        verdict = "CLOSED_HYPOTHESIS_INVERTED"
    else:
        verdict = "FAIL"

    return {
        "n_a": n_a,
        "n_b": n_b,
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "median_a": float(np.median(a)),
        "median_b": float(np.median(b)),
        "g1_u_stat": u_stat,
        "g1_p_value": p_value,
        "g1_status": "PASS" if g1_pass else "FAIL",
        "g2_point_estimate": boot["point_estimate"],
        "g2_ci_lo": boot["ci_lo"],
        "g2_ci_hi": boot["ci_hi"],
        "g2_ci_excludes_zero": boot["ci_excludes_zero"],
        "g2_magnitude_pass": g2_magnitude_pass,
        "g2_status": "PASS" if g2_pass else "FAIL",
        "g3_status": "PASS",
        "g4_hypothesis_aligned": hypothesis_aligned,
        "g4_status": g4_status,
        "verdict": verdict,
    }


# =============================================================================
# Cohort builders
# =============================================================================


def _split_cohort_by_state_and_early(
    panel: pd.DataFrame,
    state_filter: callable,
    early_col: str,
    outcome: str = PRIMARY_OUTCOME,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (early_outcome_array, non_early_outcome_array)."""
    sub = panel[state_filter(panel)].copy()
    sub = sub.dropna(subset=[outcome, early_col])
    early = sub.loc[sub[early_col] == True, outcome].astype(float).to_numpy()
    non_early = sub.loc[sub[early_col] == False, outcome].astype(float).to_numpy()
    return early, non_early


# =============================================================================
# Stability
# =============================================================================


def _stability_check(panel: pd.DataFrame, early_col: str) -> tuple[pd.DataFrame, dict]:
    """Run primary G1+G2+G3+G4 within each slope_tier and width_tier slice.

    Returns (stability_df, stability_summary).
    Stability PASS: ≥2/3 slope_tier slices AND/OR ≥2/3 width_tier slices hypothesis-aligned PASS
    (whichever the data supports per spec). Sub-min-N → INCONCLUSIVE for that slice only.
    """
    rows = []
    ext_mask = panel["signal_state"] == "extended"
    sub = panel[ext_mask].dropna(subset=[PRIMARY_OUTCOME, early_col]).copy()

    def _eval_slice(label, slice_mask):
        sub_s = sub[slice_mask]
        early = sub_s.loc[sub_s[early_col] == True, PRIMARY_OUTCOME].astype(float).to_numpy()
        non_early = sub_s.loc[sub_s[early_col] == False, PRIMARY_OUTCOME].astype(float).to_numpy()
        gates = _evaluate_pair(early, non_early)
        return {
            "slice_label": label,
            **gates,
        }

    slope_tiers = ["flat", "mild", "loose"]
    slope_results = []
    for st in slope_tiers:
        result = _eval_slice(f"slope_tier={st}", sub[SLOPE_TIER_COL] == st)
        slope_results.append(result)
        rows.append({"dimension": "slope_tier", **result})

    width_tiers = ["low", "mid", "high"]
    width_results = []
    for wt in width_tiers:
        result = _eval_slice(f"width_tier={wt}", sub["width_tier_label"] == wt)
        width_results.append(result)
        rows.append({"dimension": "width_tier", **result})

    df = pd.DataFrame(rows)

    # Stability counters: count slices where verdict=='PASS' (G1+G2+G4 aligned)
    slope_pass = sum(1 for r in slope_results if r["verdict"] == "PASS")
    slope_inconclusive = sum(1 for r in slope_results if r["verdict"] == "INSUFFICIENT")
    slope_fail_or_inverted = sum(
        1 for r in slope_results if r["verdict"] in ("FAIL", "CLOSED_HYPOTHESIS_INVERTED")
    )
    width_pass = sum(1 for r in width_results if r["verdict"] == "PASS")
    width_inconclusive = sum(1 for r in width_results if r["verdict"] == "INSUFFICIENT")
    width_fail_or_inverted = sum(
        1 for r in width_results if r["verdict"] in ("FAIL", "CLOSED_HYPOTHESIS_INVERTED")
    )

    # Spec: G5 PASS = ≥2/3 slope_tier OR ≥2/3 width_tier slices hypothesis-aligned PASS
    slope_g5 = slope_pass >= 2
    width_g5 = width_pass >= 2
    g5_overall = slope_g5 or width_g5

    summary = {
        "slope_pass": slope_pass,
        "slope_inconclusive": slope_inconclusive,
        "slope_fail_or_inverted": slope_fail_or_inverted,
        "width_pass": width_pass,
        "width_inconclusive": width_inconclusive,
        "width_fail_or_inverted": width_fail_or_inverted,
        "slope_g5": slope_g5,
        "width_g5": width_g5,
        "g5_status": "PASS" if g5_overall else "FAIL",
    }
    return df, summary


# =============================================================================
# Supplementary outcomes
# =============================================================================


def _supplementary_table(panel: pd.DataFrame, early_col: str) -> pd.DataFrame:
    """Descriptive supplementary outcome readouts on extended cohort.

    No verdict — sign mismatch with primary triggers CONTRADICTION flag in summary.
    """
    rows = []
    ext = panel[panel["signal_state"] == "extended"].copy()
    for outcome in SUPPLEMENTARY_OUTCOMES:
        if outcome not in ext.columns:
            continue
        sub = ext.dropna(subset=[outcome, early_col]).copy()
        early = sub.loc[sub[early_col] == True, outcome].astype(float).to_numpy()
        non_early = sub.loc[sub[early_col] == False, outcome].astype(float).to_numpy()
        n_a = len(early)
        n_b = len(non_early)
        if n_a < MIN_N_PER_GROUP or n_b < MIN_N_PER_GROUP:
            row = {
                "outcome": outcome,
                "n_early": n_a,
                "n_non_early": n_b,
                "mean_early": float(early.mean()) if n_a > 0 else None,
                "mean_non_early": float(non_early.mean()) if n_b > 0 else None,
                "delta_mean": (
                    float(early.mean() - non_early.mean()) if (n_a > 0 and n_b > 0) else None
                ),
                "ci_lo": None,
                "ci_hi": None,
                "ci_excludes_zero": None,
                "status": "INSUFFICIENT",
            }
        else:
            boot = _bootstrap_mean_diff(early, non_early, BOOTSTRAP_N, BOOTSTRAP_SEED)
            row = {
                "outcome": outcome,
                "n_early": n_a,
                "n_non_early": n_b,
                "mean_early": float(early.mean()),
                "mean_non_early": float(non_early.mean()),
                "delta_mean": boot["point_estimate"],
                "ci_lo": boot["ci_lo"],
                "ci_hi": boot["ci_hi"],
                "ci_excludes_zero": boot["ci_excludes_zero"],
                "status": "DESCRIPTIVE",
            }
        rows.append(row)
    return pd.DataFrame(rows)


# =============================================================================
# Output writers
# =============================================================================


def _write_gates_primary_csv(primary_pct: dict, primary_atr: dict, path: Path) -> None:
    rows = [
        {**primary_pct, "audit_label": "PRIMARY_pct"},
        {**primary_atr, "audit_label": "AUDIT_atr"},
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_gates_secondary_csv(
    sec_trigger: dict, sec_retest: dict, sec_pooled: dict, path: Path
) -> None:
    rows = [
        {**sec_trigger, "cohort": "trigger"},
        {**sec_retest, "cohort": "retest_bounce"},
        {**sec_pooled, "cohort": "pooled_trigger_or_retest"},
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


# =============================================================================
# Decision tree
# =============================================================================


def _final_verdict(
    primary_pct: dict,
    primary_atr: dict,
    g5_status: str,
    secondaries: dict,
) -> tuple[str, list[str]]:
    """Apply spec §7 decision tree.

    PASS: pct primary G1+G2+G3+G4+G5 PASS AND atr not contradicting (replicates OR atr fail by N only).
    PARTIAL: primary FAIL/inverted but ≥1 secondary all-gates PASS hypothesis-aligned.
    CLOSED: nothing passes; OR G4 violated regardless.
    """
    notes: list[str] = []
    pct_v = primary_pct["verdict"]
    atr_v = primary_atr["verdict"]

    # Hypothesis-inverted on pct primary → CLOSED outright
    if pct_v == "CLOSED_HYPOTHESIS_INVERTED":
        notes.append(
            "G4 violation on pct primary (significant negative Δmean): "
            "HYPOTHESIS_INVERTED → CLOSED. No 'sell-early' rule shortcut."
        )
        return "CLOSED", notes

    pct_primary_full_pass = (
        pct_v == "PASS" and g5_status == "PASS"
    )

    # Atr mirror G6 not contradicting:
    # - replicates (atr verdict PASS) ⇒ ok, descriptive cross-norm signal
    # - atr verdict INSUFFICIENT ⇒ N-only, not contradiction
    # - atr verdict CLOSED_HYPOTHESIS_INVERTED ⇒ CONTRADICTION → demote PASS to PARTIAL
    # - atr FAIL via p / CI failure (no inverted sign) ⇒ not contradiction (still ok per spec wording)
    atr_contradicts = atr_v == "CLOSED_HYPOTHESIS_INVERTED"

    if pct_primary_full_pass and not atr_contradicts:
        notes.append("Pct primary all gates pass AND atr mirror does not contradict → PASS.")
        if atr_v == "PASS":
            notes.append("G6 NOTE: atr replicates — descriptive cross-normalization signal.")
        elif atr_v == "INSUFFICIENT":
            notes.append("G6 NOTE: atr INSUFFICIENT (sample-size only) — not a contradiction.")
        else:
            notes.append("G6 NOTE: atr does not replicate (no inverted sign) — primary stands.")
        return "PASS", notes

    # PARTIAL path: primary not full pass; check secondaries
    secondary_passes = [
        name for name, gates in secondaries.items() if gates["verdict"] == "PASS"
    ]
    if secondary_passes:
        notes.append(
            f"Pct primary failed full-PASS criteria (verdict={pct_v}, g5={g5_status}); "
            f"secondary cohort(s) PASSed hypothesis-aligned all-gates: {', '.join(secondary_passes)} "
            f"→ PARTIAL. Pilot 6 spec drafting NOT unlocked."
        )
        return "PARTIAL", notes

    notes.append(
        f"Primary fail (verdict={pct_v}, g5={g5_status}) AND no secondary all-gates PASS "
        f"→ CLOSED. Pilot 4 HTML note UNCHANGED."
    )
    return "CLOSED", notes


# =============================================================================
# Summary writer
# =============================================================================


def _format_summary(
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
    hb_manifest: dict,
) -> str:
    lines: list[str] = []
    lines.append("# ema_context Pilot 5 — Run Summary")
    lines.append("")
    lines.append("- Spec: `memory/ema_context_pilot5_spec.md` LOCKED 2026-05-03")
    lines.append(f"- Run date: {pd.Timestamp.utcnow().isoformat()} (single authorized run)")
    lines.append(f"- Runtime: {runtime_s:.1f}s")
    lines.append(
        f"- HB event source: `output/horizontal_base_event_v1.parquet` "
        f"(scanner_version {hb_manifest.get('scanner_version')}, "
        f"frozen_at {hb_manifest.get('frozen_at')}, rows {hb_manifest.get('rows')})"
    )
    lines.append("- Pilot 4 earliness panel reused: `output/ema_context_pilot4_earliness_per_event.parquet`")
    lines.append(f"- Early threshold (LOCKED): earliness_score_pct ≤ {EARLY_THRESHOLD}")
    lines.append(f"- Primary outcome (LOCKED): `{PRIMARY_OUTCOME}`")
    lines.append("")

    lines.append("## Question (descriptive only)")
    lines.append("")
    lines.append(
        "Within extended HB events, do early-compression events "
        f"(earliness_score_pct ≤ {EARLY_THRESHOLD}) show frozen-outcome separation "
        f"(`{PRIMARY_OUTCOME}`) against non-early extended events?"
    )
    lines.append("")

    lines.append(f"## Overall Pilot 5 Verdict: **{final_verdict}**")
    lines.append("")
    for n in notes:
        lines.append(f"- {n}")
    lines.append("")

    lines.append("## Primary (extended early vs extended non-early)")
    lines.append("")
    lines.append(f"### Primary metric: ema_stack_width_pct earliness")
    lines.append(
        f"- N_early={primary_pct['n_a']:,} / N_non_early={primary_pct['n_b']:,} "
        f"(min-N {MIN_N_PER_GROUP}: {primary_pct['g3_status']})"
    )
    lines.append(
        f"- mean(early)={primary_pct['mean_a']:.4f} R / "
        f"mean(non_early)={primary_pct['mean_b']:.4f} R"
    )
    lines.append(
        f"- median(early)={primary_pct['median_a']:.4f} R / "
        f"median(non_early)={primary_pct['median_b']:.4f} R"
    )
    if primary_pct["g1_p_value"] is not None:
        lines.append(
            f"- G1 Mann-Whitney U: stat={primary_pct['g1_u_stat']:.0f} "
            f"p={primary_pct['g1_p_value']:.6e} → {primary_pct['g1_status']} "
            f"(threshold p < {P_THRESHOLD})"
        )
        lines.append(
            f"- G2 Bootstrap (n_boot={BOOTSTRAP_N}, seed={BOOTSTRAP_SEED}): "
            f"point={primary_pct['g2_point_estimate']:.4f} R "
            f"CI=[{primary_pct['g2_ci_lo']:.4f}, {primary_pct['g2_ci_hi']:.4f}]"
        )
        lines.append(
            f"  - CI excludes 0: {primary_pct['g2_ci_excludes_zero']}; "
            f"|Δ| ≥ {MAGNITUDE_FLOOR_R} R: {primary_pct['g2_magnitude_pass']} "
            f"→ {primary_pct['g2_status']}"
        )
    lines.append(
        f"- G4 sign convention: hypothesis_aligned={primary_pct['g4_hypothesis_aligned']} "
        f"→ {primary_pct['g4_status']}"
    )
    lines.append("")

    lines.append(f"### Audit metric: ema_stack_width_atr earliness (G6 G7-trap mirror)")
    lines.append(
        f"- N_early={primary_atr['n_a']:,} / N_non_early={primary_atr['n_b']:,} "
        f"(min-N {MIN_N_PER_GROUP}: {primary_atr['g3_status']})"
    )
    if primary_atr["g1_p_value"] is not None:
        lines.append(
            f"- mean(early)={primary_atr['mean_a']:.4f} R / "
            f"mean(non_early)={primary_atr['mean_b']:.4f} R"
        )
        lines.append(
            f"- G1 Mann-Whitney U: p={primary_atr['g1_p_value']:.6e} → {primary_atr['g1_status']}"
        )
        lines.append(
            f"- G2 Bootstrap: point={primary_atr['g2_point_estimate']:.4f} R "
            f"CI=[{primary_atr['g2_ci_lo']:.4f}, {primary_atr['g2_ci_hi']:.4f}] "
            f"→ {primary_atr['g2_status']}"
        )
        lines.append(
            f"- G4 hypothesis_aligned={primary_atr['g4_hypothesis_aligned']} "
            f"→ {primary_atr['g4_status']}"
        )
    lines.append(
        "- ATR-only PASS does NOT count as Pilot 5 PASS (G7-trap defense)."
    )
    lines.append("")

    lines.append(f"## G5 Stability — primary slope/width slices")
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
    lines.append("### Stability table (per slice)")
    lines.append("")
    lines.append("| dimension | slice | n_early | n_non_early | mean_early | mean_non_early | Δmean | CI_lo | CI_hi | p | verdict |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for _, r in stability_df.iterrows():
        def _f(x, fmt=".4f"):
            return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else format(x, fmt)
        lines.append(
            f"| {r['dimension']} | {r['slice_label']} | {r['n_a']} | {r['n_b']} | "
            f"{_f(r['mean_a'])} | {_f(r['mean_b'])} | {_f(r['g2_point_estimate'])} | "
            f"{_f(r['g2_ci_lo'])} | {_f(r['g2_ci_hi'])} | "
            f"{_f(r['g1_p_value'], '.2e') if r['g1_p_value'] is not None else '—'} | "
            f"{r['verdict']} |"
        )
    lines.append("")

    lines.append("## Secondary descriptive (within state, early vs non-early)")
    lines.append("")
    lines.append(
        "Pre-registered as **descriptive support only**. Cannot rescue primary into PASS. "
        "Can turn primary fail into PARTIAL only if at least one cohort all-gates PASSes hypothesis-aligned."
    )
    lines.append("")
    lines.append("| cohort | n_early | n_non_early | Δmean | CI | p | verdict |")
    lines.append("|---|---|---|---|---|---|---|")
    for cohort_name, gates in secondaries.items():
        def _f(x, fmt=".4f"):
            return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else format(x, fmt)
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

    lines.append("## Supplementary outcomes (descriptive context — extended cohort only)")
    lines.append("")
    lines.append(
        "Sign-mismatch with primary direction is logged as CONTRADICTION below. "
        "Supplementary tests are NOT independent gates."
    )
    lines.append("")
    lines.append("| outcome | n_early | n_non_early | mean_early | mean_non_early | Δmean | CI | status |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for _, r in supp_df.iterrows():
        def _f(x, fmt=".4f"):
            return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else format(x, fmt)
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

    # CONTRADICTION detection on supplementary
    primary_sign = (
        np.sign(primary_pct["g2_point_estimate"]) if primary_pct["g2_point_estimate"] is not None else 0
    )
    contradictions = []
    for _, r in supp_df.iterrows():
        if r["status"] != "DESCRIPTIVE" or r["ci_excludes_zero"] is not True:
            continue
        if r["delta_mean"] is None:
            continue
        # mfe higher = better; mae higher = worse (more adverse); failed_breakout higher = worse
        # quality_20d higher = better; time_to_MFE higher = slower (could be worse, descriptive)
        outcome = r["outcome"]
        delta_sign = np.sign(r["delta_mean"])
        # Outcome semantic direction (positive Δearly is hypothesis-aligned for these):
        positive_align = {
            "mfe_R_10d": True,
            "mae_R_10d": True,  # less negative = higher mean = better, hypothesis-align positive
            "failed_breakout_10d": False,  # higher rate of failures = worse, opposite sign
            "quality_20d": True,
            "time_to_MFE_10d": None,  # ambiguous descriptive
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
        lines.append("### CONTRADICTION flag: none — supplementary outcomes do not contradict primary direction.")
        lines.append("")

    # PASS interpretation ceiling
    lines.append("## PASS Interpretation — LOCKED CEILING")
    lines.append("")
    if final_verdict == "PASS":
        lines.append("**Allowed PASS claim (the entire allowed claim, descriptive only)**:")
        lines.append("")
        lines.append(
            f"> Per Pilot 5, **within the extended HB cohort, early-compression events "
            f"(earliness_score_pct ≤ {EARLY_THRESHOLD}) show `{PRIMARY_OUTCOME}` separation "
            f"against non-early events** (descriptive, Mann-Whitney p<{P_THRESHOLD} + bootstrap "
            f"Δmean CI excluding 0 with magnitude ≥ {MAGNITUDE_FLOOR_R} R, "
            f"stability across geometry slices, atr mirror non-contradicting)."
        )
        lines.append("")
        lines.append("**Allowed downstream actions on PASS**:")
        lines.append("- Update HB HTML note text in `tools/horizontal_base_html.py` (descriptive only).")
        lines.append("- Draft `memory/ema_context_pilot6_spec.md` for paper-only risk modifier (separate pre-reg + closure).")
        lines.append("")
    elif final_verdict == "PARTIAL":
        lines.append("**PARTIAL** — secondary descriptive support only; Pilot 6 spec drafting NOT unlocked.")
        lines.append("")
    elif final_verdict == "CLOSED":
        lines.append("**CLOSED** — workstream closes. Pilot 4 HTML note UNCHANGED. HB v1 / Pilot 1-4 verdicts UNCHANGED.")
        lines.append("")

    lines.append("**Forbidden interpretations (locked ceiling, all verdicts)**:")
    forbidden = [
        "Live trade priority / scanner ranking elevation",
        "Live entry timing recommendation ('buy early-extended sooner')",
        "Hard or soft EMA gate in any scanner / decision_engine",
        "EMA ML feature / earliness as ranking-scoring sort key",
        "Live position sizing change",
        "Generalization to non-extended states ('this proves trigger early-compression also wins')",
        "Forward-looking signature claim",
        "Cascade: Pilot 5 PASS triggering Pilot 7/8 by analogy without their own pre-reg",
        "Re-running Pilot 5 with different threshold / outcome / horizon to find PASS",
        "Reopening Pilot 1/2/3/4 verdicts based on Pilot 5",
        "Removing Pilot 4 HTML note on Pilot 5 CLOSED (timing claim and outcome claim are independent)",
    ]
    for f in forbidden:
        lines.append(f"- ❌ {f}")
    lines.append("")

    lines.append("## Anti-tweak Confirmation")
    lines.append("")
    confs = [
        f"Pilot 4 earliness panel reused (rows 10,470) — no recompute",
        f"HB event parquet reused (frozen scanner_version 1.4.0) — no rebuild",
        f"Early threshold ≤ {EARLY_THRESHOLD} LOCKED (Pilot 3 boundary, not Pilot 4 mode)",
        f"Primary outcome `{PRIMARY_OUTCOME}` LOCKED — no horizon sweep",
        f"Magnitude floor {MAGNITUDE_FLOOR_R} R LOCKED",
        f"p-threshold {P_THRESHOLD} LOCKED (no fallback)",
        f"Bootstrap n_boot={BOOTSTRAP_N}, seed={BOOTSTRAP_SEED}, group-independent resample",
        f"Min-N {MIN_N_PER_GROUP} per group LOCKED",
        f"Primary cohort within extended only — cross-state comparison forbidden",
        f"Secondary trigger/retest descriptive only — cannot rescue primary into PASS",
        f"G4 hypothesis-inverted (significant negative) → CLOSED, not PASS (no inverted-trade-rule shortcut)",
        f"G5 stability ≥2/3 slope_tier OR ≥2/3 width_tier slices",
        f"G6 ATR mirror = audit only; ATR-only PASS does NOT count",
        f"Single authorized run — no post-hoc parameter change",
        f"INSUFFICIENT ≠ FAIL",
        f"Pilot 4 ↔ Pilot 5 boundary kanonik — independent closures",
        f"Pilot 1/2/3/4 closure UNCHANGED regardless of Pilot 5 result",
        f"PASS ceiling: descriptive HTML note + Pilot 6 paper-only spec drafting only",
        f"NO live trading / gate / ML / ranking / scoring / entry timing / position sizing",
    ]
    for c in confs:
        lines.append(f"- {c}")
    lines.append("")

    return "\n".join(lines)


# =============================================================================
# Main
# =============================================================================


def main() -> int:
    t0 = time.time()
    print("[Pilot 5] Validating inputs...", flush=True)
    hb_manifest, hb_rows, ep_rows = _validate_inputs()
    print(
        f"  HB manifest OK (scanner_version={hb_manifest.get('scanner_version')}, rows={hb_rows})",
        flush=True,
    )
    print(f"  Earliness panel rows={ep_rows}", flush=True)

    print("[Pilot 5] Building joined panel (positional, hb ↔ earliness)...", flush=True)
    panel = _build_panel()
    print(f"  Panel rows: {len(panel):,}", flush=True)
    wt = panel.attrs["width_tier_breaks"]
    print(
        f"  width_tier breaks (extended cohort, q={wt['lo_q']}/{wt['hi_q']}): "
        f"lo={wt['lo_v']:.4f} hi={wt['hi_v']:.4f}",
        flush=True,
    )

    print("[Pilot 5] Primary cohort (extended early vs non-early) on pct primary...", flush=True)
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

    print("[Pilot 5] G6 audit on atr mirror...", flush=True)
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

    print("[Pilot 5] G5 stability (slope_tier × width_tier)...", flush=True)
    stability_df, g5_summary = _stability_check(panel, "is_early_pct")
    print(
        f"  slope_tier PASS={g5_summary['slope_pass']}/3 | "
        f"width_tier PASS={g5_summary['width_pass']}/3 | G5 {g5_summary['g5_status']}",
        flush=True,
    )

    print("[Pilot 5] Secondary cohorts (trigger / retest_bounce / pooled)...", flush=True)
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
        f"  trigger N=({sec_trigger['n_a']},{sec_trigger['n_b']}) verdict={sec_trigger['verdict']} | "
        f"retest_bounce N=({sec_retest['n_a']},{sec_retest['n_b']}) verdict={sec_retest['verdict']} | "
        f"pooled N=({sec_pooled['n_a']},{sec_pooled['n_b']}) verdict={sec_pooled['verdict']}",
        flush=True,
    )

    print("[Pilot 5] Supplementary outcomes (descriptive)...", flush=True)
    supp_df = _supplementary_table(panel, "is_early_pct")

    secondaries = {
        "trigger": sec_trigger,
        "retest_bounce": sec_retest,
        "pooled_trigger_or_retest": sec_pooled,
    }

    print("[Pilot 5] Final verdict (decision tree)...", flush=True)
    final_verdict, notes = _final_verdict(
        primary_pct, primary_atr, g5_summary["g5_status"], secondaries
    )
    print(f"  FINAL VERDICT: {final_verdict}", flush=True)

    runtime_s = time.time() - t0

    print("[Pilot 5] Writing outputs...", flush=True)
    panel.to_parquet(OUT_PANEL, index=False)
    _write_gates_primary_csv(primary_pct, primary_atr, OUT_GATES_PRIMARY)
    _write_gates_secondary_csv(sec_trigger, sec_retest, sec_pooled, OUT_GATES_SECONDARY)
    stability_df.to_csv(OUT_STABILITY, index=False)
    supp_df.to_csv(OUT_SUPPLEMENTARY, index=False)
    summary_text = _format_summary(
        final_verdict,
        notes,
        primary_pct,
        primary_atr,
        g5_summary,
        stability_df,
        secondaries,
        supp_df,
        panel,
        runtime_s,
        hb_manifest,
    )
    OUT_SUMMARY.write_text(summary_text)
    print(f"  Wrote: {OUT_PANEL.name} / {OUT_GATES_PRIMARY.name} / {OUT_GATES_SECONDARY.name}", flush=True)
    print(f"  Wrote: {OUT_STABILITY.name} / {OUT_SUPPLEMENTARY.name} / {OUT_SUMMARY.name}", flush=True)
    print(f"[Pilot 5] Done in {runtime_s:.1f}s — verdict {final_verdict}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
