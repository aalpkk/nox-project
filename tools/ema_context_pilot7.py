"""ema_context Pilot 7 — Trigger/Retest outcome separation, single authorized run.

Spec: memory/ema_context_pilot7_trigger_retest_spec.md (LOCKED 2026-05-04, ONAY 2026-05-04)

Distinct biology from extended TIER_A_PAPER. NOT an extension, NOT a pooling
exercise, NOT a Pilot 5/6 rescue.

Question: Within HB `trigger ∪ retest_bounce` cohort, does **event-near EMA
compression** (-5 ≤ earliness_score_pct ≤ -1) descriptively separate
`realized_R_10d` against an undifferentiated trigger/retest baseline?

Cohort col `ema_tr_tier` (LOCKED enumeration):
  EVENT_NEAR_TIER       = signal_state ∈ {trigger, retest_bounce} ∧ -5 ≤ earliness_score_pct ≤ -1
  NON_EVENT_NEAR        = signal_state ∈ {trigger, retest_bounce} ∧ earliness_score_pct > -1
  DROP_EARLY_DIAGNOSTIC = signal_state ∈ {trigger, retest_bounce} ∧ earliness_score_pct < -5
  OUT_OF_SCOPE          = signal_state == "extended" OR signal_state ∉ {trigger, retest_bounce, extended}
                          OR earliness_score_pct IS NULL

BASELINE = eligible universe = EVENT_NEAR_TIER ∪ NON_EVENT_NEAR.
DROP_EARLY_DIAGNOSTIC is NOT in BASELINE (extended-cohort biology, Pilot 5/6).
DROP rows reported descriptively in `drop_diagnostic_census.csv` — no gate, no claim.

Threshold provenance (LOCKED text):
  The event-near window [-5,-1] is derived from Pilot 3's pre-registered state
  decomposition (trigger and retest_bounce both showed argmin compression in
  [-5,-1]). Pilot 7 does NOT sweep thresholds, NOT promote neighbor buckets,
  will NOT test variants [-4,0] / [-6,-1] / [-5,0] post-hoc.

Anti-rescue (locked, end-to-end):
  - Pilot 5 panel reuse, no recompute earliness/outcomes/geometry
  - HB event parquet + earliness parquet sha256 hashes asserted at boot
  - realized_R_10d primary outcome LOCKED, no horizon sweep
  - uplift magnitude floor 0.05 R LOCKED
  - p-threshold 0.001 LOCKED, no fallback
  - n_boot=2000, seed=7, group-independent resample
  - min-N 200 (A) / 200 (B) / 500 (BASELINE) LOCKED
  - G2 = LABEL SANITY CHECK (NON_EVENT_NEAR ⊂ BASELINE → not independent test)
  - G5 = STATE STABILITY (trigger-only + retest-only secondary; slope/width
    sub-stratification dropped, supplementary descriptive only)
  - G6 ATR mirror = audit only; ATR-only PASS does NOT count
  - DROP_EARLY_DIAGNOSTIC reported descriptively, no gate
  - No new tier (no top-decile, no quartile, no atr+pct agreement carve-out)
  - No regime-conditional sub-test, no multi-TF confirmation
  - No new feature (no ema21 reclaim, no stack mixed→bull — Pilot 8/9 if ever)
  - No pooling with extended cohort, even on PASS
  - Single authorized run; bug → void+rerun, no amend
  - INSUFFICIENT ≠ FAIL
  - FAIL/CLOSED → NO reverse-direction "short" claim
"""

from __future__ import annotations

import hashlib
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
EARLINESS_PARQUET = PROJECT_ROOT / "output" / "ema_context_pilot4_earliness_per_event.parquet"
PILOT5_PANEL = PROJECT_ROOT / "output" / "ema_context_pilot5_panel.parquet"

OUT_PANEL = PROJECT_ROOT / "output" / "ema_context_pilot7_panel.parquet"
OUT_GATES_MAIN = PROJECT_ROOT / "output" / "ema_context_pilot7_gates_main.csv"
OUT_STATE_STABILITY = PROJECT_ROOT / "output" / "ema_context_pilot7_state_stability.csv"
OUT_SUPPLEMENTARY = PROJECT_ROOT / "output" / "ema_context_pilot7_supplementary.csv"
OUT_DROP_CENSUS = PROJECT_ROOT / "output" / "ema_context_pilot7_drop_diagnostic_census.csv"
OUT_SUMMARY = PROJECT_ROOT / "output" / "ema_context_pilot7_summary.md"

# Tier rule thresholds (LOCKED — Pilot 3 decomposition origin, no sweep)
EVENT_NEAR_LO = -5
EVENT_NEAR_HI = -1
DROP_BOUNDARY = -5  # earliness_score_pct < -5 → DROP_EARLY_DIAGNOSTIC

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

UPLIFT_FLOOR_R = 0.05  # G1 magnitude floor
MIN_N_TIER = 200       # A and B
MIN_N_BASELINE = 500   # BASELINE (lower than Pilot 6 1000 because trigger+retest cohort smaller)

# Manifest expectations (LOCKED hashes from paper_execution_v0 manifest 2026-05-04)
EXPECTED_SCANNER_VERSION = "1.4.0"
EXPECTED_HB_ROWS = 10470
EXPECTED_PILOT5_PANEL_ROWS = 10470
EXPECTED_HB_SHA256 = "2eb8a9a5d68e7e4831158f6a3e97c8b74521591af55e29a9682aa3ae7107b818"
EXPECTED_EARLINESS_SHA256 = "a6926de05d98c7957a4d18680ba9a6825d9a35f5b69bc3fd841b35bef7cf1641"

# Tier labels
EVENT_NEAR_TIER = "EVENT_NEAR_TIER"
NON_EVENT_NEAR = "NON_EVENT_NEAR"
DROP_EARLY_DIAGNOSTIC = "DROP_EARLY_DIAGNOSTIC"
TIER_OOS = "OUT_OF_SCOPE"

TR_STATES = {"trigger", "retest_bounce"}

THRESHOLD_PROVENANCE_EN = (
    "The event-near window [-5,-1] is derived from Pilot 3's pre-registered state "
    "decomposition, where both `trigger` and `retest_bounce` showed argmin compression "
    "in [-5,-1]. Pilot 7 does NOT sweep thresholds, does NOT promote neighbor buckets, "
    "and will NOT test variants such as [-4,0], [-6,-1], or [-5,0] after seeing results. "
    "The drop boundary at -6 is symmetric: ≤ -6 is extended-cohort biology already "
    "covered by Pilot 5/6, and is excluded from trigger/retest BASELINE rather than pooled."
)

THRESHOLD_PROVENANCE_TR = (
    "[-5,-1] penceresi Pilot 3 state decomposition önceden kilitli sonucundan türetilmiştir; "
    "trigger ve retest_bounce için argmin bucket [-5,-1] idi. Pilot 7 threshold sweep yapmaz, "
    "komşu bucket terfisi yapmaz, sonuçtan sonra [-4,0] / [-6,-1] / [-5,0] varyantları "
    "denenmez. ≤ -6 bucket'ı extended biyolojidir (Pilot 5/6 kapsamı), trigger/retest "
    "BASELINE'ına dahil edilmez, drop edilir."
)


# =============================================================================
# BOOT — fail-fast integrity check (16-item authorization checklist embodiment)
# =============================================================================


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _validate_inputs() -> tuple[dict, int, dict]:
    if not HB_EVENT_PARQUET.exists():
        raise FileNotFoundError(f"HB event parquet missing: {HB_EVENT_PARQUET}")
    if not HB_EVENT_MANIFEST.exists():
        raise FileNotFoundError(f"HB event manifest missing: {HB_EVENT_MANIFEST}")
    if not EARLINESS_PARQUET.exists():
        raise FileNotFoundError(f"Pilot 4 earliness parquet missing: {EARLINESS_PARQUET}")
    if not PILOT5_PANEL.exists():
        raise FileNotFoundError(
            f"Pilot 5 panel missing: {PILOT5_PANEL}. SPEC violation — Pilot 7 reuses "
            f"Pilot 5 panel, NO recompute. STOP."
        )

    manifest = json.loads(HB_EVENT_MANIFEST.read_text())
    sv = manifest.get("scanner_version")
    if sv != EXPECTED_SCANNER_VERSION:
        raise RuntimeError(
            f"HB manifest scanner_version mismatch: got {sv!r} != "
            f"{EXPECTED_SCANNER_VERSION!r}. SPEC violation — STOP."
        )
    rows = manifest.get("rows")
    if rows != EXPECTED_HB_ROWS:
        raise RuntimeError(
            f"HB manifest rows mismatch: got {rows!r} != {EXPECTED_HB_ROWS}. "
            f"SPEC violation — STOP."
        )

    hb_sha = _sha256(HB_EVENT_PARQUET)
    if hb_sha != EXPECTED_HB_SHA256:
        raise RuntimeError(
            f"HB event parquet sha256 mismatch:\n  got      {hb_sha}\n  expected {EXPECTED_HB_SHA256}\n"
            f"SPEC §7 violation — frozen artifact changed. STOP."
        )
    earliness_sha = _sha256(EARLINESS_PARQUET)
    if earliness_sha != EXPECTED_EARLINESS_SHA256:
        raise RuntimeError(
            f"Earliness parquet sha256 mismatch:\n  got      {earliness_sha}\n  "
            f"expected {EXPECTED_EARLINESS_SHA256}\n"
            f"SPEC §7 violation — frozen artifact changed. STOP."
        )

    p5_rows = pd.read_parquet(PILOT5_PANEL, columns=["event_id"]).shape[0]
    if p5_rows != EXPECTED_PILOT5_PANEL_ROWS:
        raise RuntimeError(
            f"Pilot 5 panel rows mismatch: got {p5_rows} != {EXPECTED_PILOT5_PANEL_ROWS}. "
            f"SPEC violation — panel may have been rebuilt."
        )

    boot_integrity = {
        "hb_event_parquet_sha256": hb_sha,
        "earliness_parquet_sha256": earliness_sha,
        "scanner_version": sv,
        "hb_event_rows": rows,
        "pilot5_panel_rows": p5_rows,
    }
    return manifest, p5_rows, boot_integrity


# =============================================================================
# Cohort tagging
# =============================================================================


def _build_panel() -> pd.DataFrame:
    panel = pd.read_parquet(PILOT5_PANEL).reset_index(drop=True)

    def _tier_pct(state, score):
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
        # score in (DROP_BOUNDARY, EVENT_NEAR_LO) is impossible since EVENT_NEAR_LO == DROP_BOUNDARY
        # (i.e., -5). Score == -5 → EVENT_NEAR; score < -5 → DROP. Defensive fallthrough:
        return TIER_OOS

    def _tier_atr(state, score):
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
        _tier_pct(s, e) for s, e in zip(panel["signal_state"], panel["earliness_score_pct"])
    ]
    panel["ema_tr_tier_atr"] = [
        _tier_atr(s, e) for s, e in zip(panel["signal_state"], panel["earliness_score_atr"])
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
    n_a_min: int,
    n_b_min: int,
) -> dict:
    """Generic gate evaluation.

    require_positive=True : hypothesis-aligned positive Δ; significant negative → CLOSED inverted.
    require_positive=False: G2-style sanity; hypothesis-aligned Δ ≤ 0;
                            significant positive → CLOSED contradiction.
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
        # G2 label-sanity: hypothesis-aligned Δ ≤ 0 (NON_EVENT_NEAR at or below BASELINE)
        hypothesis_aligned = delta <= 0
        if ci_excludes_zero and delta > 0:
            verdict = "CLOSED_CONTRADICTION"
            sign_status = "CONTRADICTION_NON_EVENT_NEAR_ABOVE_BASELINE"
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


def _eligible_universe(panel: pd.DataFrame, tier_col: str) -> pd.DataFrame:
    """Eligible universe = EVENT_NEAR_TIER ∪ NON_EVENT_NEAR (BASELINE)."""
    eligible = panel[panel[tier_col].isin([EVENT_NEAR_TIER, NON_EVENT_NEAR])].copy()
    eligible = eligible.dropna(subset=[PRIMARY_OUTCOME, tier_col])
    return eligible


# =============================================================================
# G5 — state stability (trigger-only, retest-only secondary)
# =============================================================================


def _g5_state_stability(panel: pd.DataFrame, tier_col: str) -> tuple[pd.DataFrame, dict]:
    """G5: trigger-only and retest-only secondary tests.

    Pooled primary G1 PASS AND trigger-only sufficient-N non-contradicting
    AND retest-only either non-contradicting or INSUFFICIENT → G5 PASS.
    Trigger-only contradicting → G5 FAIL.
    """
    rows = []
    state_results: dict[str, dict] = {}

    for state in ("trigger", "retest_bounce"):
        sub = _eligible_universe(panel[panel["signal_state"] == state], tier_col)
        a = sub.loc[sub[tier_col] == EVENT_NEAR_TIER, PRIMARY_OUTCOME].astype(float).to_numpy()
        baseline = sub[PRIMARY_OUTCOME].astype(float).to_numpy()
        # Use TIER min-N for sub-cohort; relax baseline floor since per-state baseline
        # is naturally smaller. Spec: "if EVENT_NEAR sub-N ≥ 200, evaluate".
        gate = _gate_pair(
            a, baseline,
            magnitude_floor=0.0,  # no magnitude floor on sub-state secondary; descriptive
            require_positive=True,
            n_a_min=MIN_N_TIER,
            n_b_min=MIN_N_TIER,  # match TIER min, not baseline floor
        )
        rows.append({"state": state, **gate})
        state_results[state] = gate

    df = pd.DataFrame(rows)

    trigger = state_results["trigger"]
    retest = state_results["retest_bounce"]

    def _classify(g: dict) -> str:
        """Classify a sub-state gate as CONTRADICTING / NON_CONTRADICTING / INSUFFICIENT."""
        if g["verdict"] == "INSUFFICIENT":
            return "INSUFFICIENT"
        # Contradicting = significant negative delta (ci excludes zero AND delta < 0)
        if g["ci_excludes_zero"] and (g["delta"] is not None) and g["delta"] < 0:
            return "CONTRADICTING"
        return "NON_CONTRADICTING"

    trigger_class = _classify(trigger)
    retest_class = _classify(retest)

    # G5 logic per spec §5.5:
    # - trigger-only sufficient-N must be non-contradicting (CONTRADICTING → G5 FAIL)
    # - retest-only must be non-contradicting OR INSUFFICIENT (INSUFFICIENT does not fail)
    g5_fail = (trigger_class == "CONTRADICTING") or (retest_class == "CONTRADICTING")
    g5_status = "FAIL" if g5_fail else "PASS"

    summary = {
        "trigger_n_a": trigger["n_a"],
        "trigger_n_b": trigger["n_b"],
        "trigger_delta": trigger["delta"],
        "trigger_ci_lo": trigger["ci_lo"],
        "trigger_ci_hi": trigger["ci_hi"],
        "trigger_classification": trigger_class,
        "retest_n_a": retest["n_a"],
        "retest_n_b": retest["n_b"],
        "retest_delta": retest["delta"],
        "retest_ci_lo": retest["ci_lo"],
        "retest_ci_hi": retest["ci_hi"],
        "retest_classification": retest_class,
        "g5_status": g5_status,
    }
    return df, summary


# =============================================================================
# Supplementary descriptive (NOT gates)
# =============================================================================


def _supplementary_table(panel: pd.DataFrame, tier_col: str) -> pd.DataFrame:
    """Supplementary outcomes (mfe/mae/failed_breakout/quality/time_to_MFE)
    EVENT_NEAR_TIER vs NON_EVENT_NEAR within trigger∪retest cohort.
    Descriptive only.
    """
    rows = []
    eligible = panel[panel[tier_col].isin([EVENT_NEAR_TIER, NON_EVENT_NEAR])].copy()
    for outcome in SUPPLEMENTARY_OUTCOMES:
        if outcome not in eligible.columns:
            continue
        sub = eligible.dropna(subset=[outcome, tier_col]).copy()
        a = sub.loc[sub[tier_col] == EVENT_NEAR_TIER, outcome].astype(float).to_numpy()
        b = sub.loc[sub[tier_col] == NON_EVENT_NEAR, outcome].astype(float).to_numpy()
        n_a = len(a)
        n_b = len(b)
        if n_a < MIN_N_TIER or n_b < MIN_N_TIER:
            row = {
                "outcome": outcome,
                "n_event_near": n_a,
                "n_non_event_near": n_b,
                "mean_event_near": float(a.mean()) if n_a > 0 else None,
                "mean_non_event_near": float(b.mean()) if n_b > 0 else None,
                "delta_mean": (float(a.mean() - b.mean()) if (n_a > 0 and n_b > 0) else None),
                "ci_lo": None,
                "ci_hi": None,
                "ci_excludes_zero": None,
                "status": "INSUFFICIENT",
            }
        else:
            boot = _bootstrap_mean_diff(a, b, BOOTSTRAP_N, BOOTSTRAP_SEED)
            row = {
                "outcome": outcome,
                "n_event_near": n_a,
                "n_non_event_near": n_b,
                "mean_event_near": float(a.mean()),
                "mean_non_event_near": float(b.mean()),
                "delta_mean": boot["point_estimate"],
                "ci_lo": boot["ci_lo"],
                "ci_hi": boot["ci_hi"],
                "ci_excludes_zero": boot["ci_excludes_zero"],
                "status": "DESCRIPTIVE",
            }
        rows.append(row)

    # Add slope/width descriptive sub-stratification (supplementary, NOT gates)
    eligible_clean = eligible.dropna(subset=[PRIMARY_OUTCOME, tier_col])
    for dim_col, dim_name, slices in [
        ("family__slope_tier", "slope_tier", ["flat", "mild", "loose"]),
        ("width_tier_label", "width_tier", ["low", "mid", "high"]),
    ]:
        for sl in slices:
            sub = eligible_clean[eligible_clean[dim_col] == sl]
            a = sub.loc[sub[tier_col] == EVENT_NEAR_TIER, PRIMARY_OUTCOME].astype(float).to_numpy()
            b = sub.loc[sub[tier_col] == NON_EVENT_NEAR, PRIMARY_OUTCOME].astype(float).to_numpy()
            n_a = len(a)
            n_b = len(b)
            if n_a < MIN_N_TIER or n_b < MIN_N_TIER:
                rows.append({
                    "outcome": f"{dim_name}={sl}__{PRIMARY_OUTCOME}",
                    "n_event_near": n_a,
                    "n_non_event_near": n_b,
                    "mean_event_near": float(a.mean()) if n_a > 0 else None,
                    "mean_non_event_near": float(b.mean()) if n_b > 0 else None,
                    "delta_mean": (float(a.mean() - b.mean()) if (n_a > 0 and n_b > 0) else None),
                    "ci_lo": None, "ci_hi": None, "ci_excludes_zero": None,
                    "status": "INSUFFICIENT_DESCRIPTIVE",
                })
            else:
                boot = _bootstrap_mean_diff(a, b, BOOTSTRAP_N, BOOTSTRAP_SEED)
                rows.append({
                    "outcome": f"{dim_name}={sl}__{PRIMARY_OUTCOME}",
                    "n_event_near": n_a,
                    "n_non_event_near": n_b,
                    "mean_event_near": float(a.mean()),
                    "mean_non_event_near": float(b.mean()),
                    "delta_mean": boot["point_estimate"],
                    "ci_lo": boot["ci_lo"],
                    "ci_hi": boot["ci_hi"],
                    "ci_excludes_zero": boot["ci_excludes_zero"],
                    "status": "DESCRIPTIVE",
                })

    return pd.DataFrame(rows)


# =============================================================================
# DROP_EARLY_DIAGNOSTIC census (descriptive only — NO gate, NO interpretation)
# =============================================================================


def _drop_diagnostic_census(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for state in ("trigger", "retest_bounce", "all"):
        if state == "all":
            sub = panel[panel["ema_tr_tier"] == DROP_EARLY_DIAGNOSTIC]
        else:
            sub = panel[
                (panel["signal_state"] == state)
                & (panel["ema_tr_tier"] == DROP_EARLY_DIAGNOSTIC)
            ]
        n = len(sub)
        outcomes = sub[PRIMARY_OUTCOME].astype(float).dropna()
        n_with_outcome = len(outcomes)
        if n_with_outcome == 0:
            row = {
                "scope": state,
                "n": n,
                "n_with_outcome": 0,
                "mean_realized_R_10d": None,
                "median_realized_R_10d": None,
                "std_realized_R_10d": None,
                "win_rate": None,
                "profit_factor": None,
            }
        else:
            wins = outcomes[outcomes > 0]
            losses = outcomes[outcomes < 0]
            wins_sum = float(wins.sum()) if len(wins) > 0 else 0.0
            losses_sum_abs = float(-losses.sum()) if len(losses) > 0 else 0.0
            pf = (wins_sum / losses_sum_abs) if losses_sum_abs > 0 else None
            row = {
                "scope": state,
                "n": n,
                "n_with_outcome": n_with_outcome,
                "mean_realized_R_10d": float(outcomes.mean()),
                "median_realized_R_10d": float(outcomes.median()),
                "std_realized_R_10d": float(outcomes.std(ddof=0)),
                "win_rate": float((outcomes > 0).mean()),
                "profit_factor": pf,
            }
        rows.append(row)
    return pd.DataFrame(rows)


# =============================================================================
# Decision tree
# =============================================================================


def _final_verdict(
    g1: dict,
    g2: dict,
    g6_g1: dict,
    g3_baseline_pass: bool,
    g5_status: str,
) -> tuple[str, list[str]]:
    notes: list[str] = []

    # G3 baseline floor not met → INSUFFICIENT (no gate verdict)
    if not g3_baseline_pass:
        notes.append(
            f"G3 BASELINE floor not met (n_baseline < {MIN_N_BASELINE}). "
            f"Pre-reg verdict INSUFFICIENT — panel frozen, INSUFFICIENT terminal. "
            f"INSUFFICIENT ≠ FAIL."
        )
        return "INSUFFICIENT", notes

    # G1 hypothesis-inverted → CLOSED (no contra-signal claim)
    if g1["verdict"] == "CLOSED_HYPOTHESIS_INVERTED":
        notes.append(
            "G1 violation: EVENT_NEAR_TIER significantly BELOW BASELINE (hypothesis-inverted). "
            "→ CLOSED. NO 'short EVENT_NEAR' / 'trade NON_EVENT_NEAR' shortcut."
        )
        return "CLOSED", notes

    # G2 contradiction → CLOSED
    if g2["verdict"] == "CLOSED_CONTRADICTION":
        notes.append(
            "G2 sanity contradiction: NON_EVENT_NEAR significantly ABOVE BASELINE. "
            "Warning-tier semantics broken. → CLOSED."
        )
        return "CLOSED", notes

    # G1 INSUFFICIENT (e.g. EVENT_NEAR_TIER < 200) → INSUFFICIENT
    if g1["verdict"] == "INSUFFICIENT":
        notes.append(
            f"G1 INSUFFICIENT (n_event_near < {MIN_N_TIER}). Pre-reg INSUFFICIENT — "
            f"panel frozen, terminal. NO claim, NO threshold sweep, NO gate substitution."
        )
        return "INSUFFICIENT", notes

    g1_pass = g1["verdict"] == "PASS"
    g2_pass = g2["verdict"] == "PASS"
    g5_pass = g5_status == "PASS"
    atr_contradicts = g6_g1["verdict"] == "CLOSED_HYPOTHESIS_INVERTED"

    if g1_pass and g2_pass and g5_pass and not atr_contradicts:
        notes.append(
            "G1 PASS (EVENT_NEAR_TIER uplift over BASELINE) AND G2 PASS (NON_EVENT_NEAR "
            "label sanity OK) AND G5 PASS (state stability) AND G6 atr mirror "
            "non-contradicting → PASS."
        )
        if g6_g1["verdict"] == "PASS":
            notes.append("G6 NOTE: atr mirror replicates uplift — descriptive cross-normalization signal.")
        elif g6_g1["verdict"] == "INSUFFICIENT":
            notes.append("G6 NOTE: atr mirror INSUFFICIENT (sample-size only) — not a contradiction.")
        elif g6_g1["verdict"] == "FAIL":
            notes.append("G6 NOTE: atr mirror does not replicate uplift, but does not contradict — primary stands.")
        return "PASS", notes

    if g1_pass and not (g2_pass and g5_pass and not atr_contradicts):
        weak = []
        if not g2_pass:
            weak.append("G2 sanity not clean")
        if not g5_pass:
            weak.append("G5 state stability fail (trigger-only contradicting)")
        if atr_contradicts:
            weak.append("G6 atr mirror contradicts")
        notes.append(
            f"G1 PASS but {'; '.join(weak)} → PARTIAL. paper_execution_v0_trigger_retest_spec "
            f"drafting NOT unlocked."
        )
        return "PARTIAL", notes

    notes.append(
        f"G1 verdict={g1['verdict']} — primary uplift gate not passed → FAIL. "
        f"NO threshold sweep, NO neighbor-bucket promotion, NO Pilot 7b. "
        f"Extended TIER_A_PAPER and Pilot 1-6 closures UNCHANGED."
    )
    return "FAIL", notes


# =============================================================================
# Output writers
# =============================================================================


def _write_gates_main_csv(g1: dict, g2: dict, g6_g1: dict, path: Path) -> None:
    rows = [
        {**g1, "gate_label": "G1_EVENT_NEAR_vs_BASELINE_pct"},
        {**g2, "gate_label": "G2_NON_EVENT_NEAR_vs_BASELINE_pct_label_sanity"},
        {**g6_g1, "gate_label": "G6_EVENT_NEAR_vs_BASELINE_atr_audit"},
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


def _emit_gate_block(
    lines: list[str],
    gate: dict,
    magnitude_floor_label: str | None,
    sanity: bool = False,
) -> None:
    if gate.get("p_value") is None:
        lines.append(f"- N_a={gate.get('n_a')} / N_b={gate.get('n_b')} | verdict={gate.get('verdict')}")
        return
    lines.append(f"- N_a={gate['n_a']:,} / N_b={gate['n_b']:,}")
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


def _format_summary(
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
    hb_manifest: dict,
    boot_integrity: dict,
) -> str:
    lines: list[str] = []
    lines.append("# ema_context Pilot 7 — Trigger/Retest Outcome Separation Run Summary")
    lines.append("")
    lines.append("- Spec: `memory/ema_context_pilot7_trigger_retest_spec.md` LOCKED 2026-05-04, ONAY 2026-05-04")
    lines.append(f"- Run date: {pd.Timestamp.utcnow().isoformat()} (single authorized run)")
    lines.append(f"- Runtime: {runtime_s:.1f}s")
    lines.append(
        f"- HB event source: `output/horizontal_base_event_v1.parquet` "
        f"(scanner_version {hb_manifest.get('scanner_version')}, "
        f"frozen_at {hb_manifest.get('frozen_at')}, rows {hb_manifest.get('rows')})"
    )
    lines.append(f"- HB sha256: `{boot_integrity['hb_event_parquet_sha256']}`")
    lines.append(f"- Earliness sha256: `{boot_integrity['earliness_parquet_sha256']}`")
    lines.append("- Pilot 5 panel reused: `output/ema_context_pilot5_panel.parquet`")
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

    lines.append("## Threshold provenance (LOCKED, verbatim)")
    lines.append("")
    lines.append(f"> {THRESHOLD_PROVENANCE_EN}")
    lines.append("")
    lines.append(f"> _Türkçe_: {THRESHOLD_PROVENANCE_TR}")
    lines.append("")

    lines.append("## Question (descriptive only)")
    lines.append("")
    lines.append(
        "Within HB `trigger ∪ retest_bounce` cohort, does **event-near EMA stack compression** "
        "(window `[-5, -1]` derived from Pilot 3 state decomposition) descriptively separate "
        "`realized_R_10d` outcomes against an undifferentiated trigger/retest baseline? "
        "Paper-tier *labelling* candidacy validation only — Layer B paper execution shadow "
        "OUT OF SCOPE for this run."
    )
    lines.append("")

    # Cohort census
    n_event_near = int((panel["ema_tr_tier"] == EVENT_NEAR_TIER).sum())
    n_non_event = int((panel["ema_tr_tier"] == NON_EVENT_NEAR).sum())
    n_drop = int((panel["ema_tr_tier"] == DROP_EARLY_DIAGNOSTIC).sum())
    n_oos = int((panel["ema_tr_tier"] == TIER_OOS).sum())
    n_baseline = n_event_near + n_non_event

    lines.append("## Cohort census (pct primary)")
    lines.append("")
    lines.append(f"- EVENT_NEAR_TIER: {n_event_near:,}")
    lines.append(f"- NON_EVENT_NEAR: {n_non_event:,}")
    lines.append(f"- BASELINE (EVENT_NEAR ∪ NON_EVENT_NEAR): {n_baseline:,}")
    lines.append(f"- DROP_EARLY_DIAGNOSTIC: {n_drop:,} (DROPPED, descriptive census only)")
    lines.append(f"- OUT_OF_SCOPE (extended / NaN earliness / non-tr/retest states): {n_oos:,}")
    lines.append("")

    # Trigger / retest breakdown
    n_trig = int(((panel["signal_state"] == "trigger") & (panel["ema_tr_tier"] != TIER_OOS)).sum())
    n_retest = int(((panel["signal_state"] == "retest_bounce") & (panel["ema_tr_tier"] != TIER_OOS)).sum())
    n_trig_event_near = int(((panel["signal_state"] == "trigger") & (panel["ema_tr_tier"] == EVENT_NEAR_TIER)).sum())
    n_retest_event_near = int(
        ((panel["signal_state"] == "retest_bounce") & (panel["ema_tr_tier"] == EVENT_NEAR_TIER)).sum()
    )
    lines.append(f"- trigger total in scope: {n_trig:,} (EVENT_NEAR: {n_trig_event_near:,})")
    lines.append(f"- retest_bounce total in scope: {n_retest:,} (EVENT_NEAR: {n_retest_event_near:,})")
    lines.append("")

    lines.append(f"## Overall Pilot 7 Verdict: **{final_verdict}**")
    lines.append("")
    for n in notes:
        lines.append(f"- {n}")
    lines.append("")

    # G1
    lines.append("## G1 — EVENT_NEAR_TIER vs BASELINE (uplift; primary pct)")
    lines.append("")
    lines.append(
        "**Note on construction**: EVENT_NEAR_TIER is a strict subset of BASELINE "
        "(eligible universe = EVENT_NEAR_TIER ∪ NON_EVENT_NEAR). Bootstrap CI is computed via "
        "independent resampling per group and remains valid; the practical interpretation is "
        "*uplift if you only paper-traded EVENT_NEAR_TIER vs an undifferentiated trigger/retest cohort*."
    )
    lines.append("")
    _emit_gate_block(lines, g1, magnitude_floor_label=f"≥ {UPLIFT_FLOOR_R} R uplift")
    lines.append("")

    # G2
    lines.append("## G2 — NON_EVENT_NEAR vs BASELINE (LABEL SANITY CHECK, NOT independence test)")
    lines.append("")
    lines.append(
        "**Construction caveat**: NON_EVENT_NEAR ⊂ BASELINE — this is **NOT** a mathematically "
        "independent test of B vs (A∪B). It is reported as a **label sanity check**: if "
        "NON_EVENT_NEAR is significantly *above* BASELINE, the warning-tier semantics are broken (CLOSED). "
        "Hypothesis-aligned Δmean(NON_EVENT_NEAR − BASELINE) ≤ 0."
    )
    lines.append("")
    _emit_gate_block(lines, g2, magnitude_floor_label=None, sanity=True)
    lines.append("")

    # G3
    lines.append("## G3 — Min-N")
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

    # G4 (sign convention is enforced inline in G1; just narrate)
    lines.append("## G4 — Sign convention")
    lines.append("")
    lines.append(
        f"- G1 sign_status: {g1.get('sign_status')} (hypothesis-aligned Δ > 0 required for PASS; "
        f"significant negative Δ → CLOSED hypothesis-inverted, NOT translated into "
        f"'short / avoid' claim)."
    )
    lines.append("")

    # G5
    lines.append("## G5 — State stability (trigger-only + retest-only secondary)")
    lines.append("")
    lines.append(
        "Slope/width sub-stratification dropped from G5 (supplementary descriptive only) to "
        "preserve N. State-stability tests:"
    )
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

    # G6
    lines.append("## G6 — ATR mirror audit (G7-trap defense, audit-only)")
    lines.append("")
    lines.append(
        "Re-runs G1 with EVENT_NEAR_TIER defined via earliness_score_atr ∈ [-5, -1]. "
        "ATR-only PASS does NOT count standalone. ATR-contradiction (significant negative) "
        "→ demote PASS to PARTIAL."
    )
    lines.append("")
    _emit_gate_block(lines, g6_g1, magnitude_floor_label=f"≥ {UPLIFT_FLOOR_R} R uplift")
    lines.append("")

    # Supplementary
    lines.append("## Supplementary descriptive — EVENT_NEAR_TIER vs NON_EVENT_NEAR (NOT gates)")
    lines.append("")
    lines.append(
        "Supplementary outcomes + slope/width breakdowns are **descriptive only**, NEVER gate-passing. "
        "Sign-mismatch with primary direction is logged below."
    )
    lines.append("")
    lines.append("| outcome | n_event_near | n_non_event_near | mean_event_near | mean_non_event_near | Δmean | CI | status |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for _, r in supp_df.iterrows():
        def _f(x, fmt=".4f"):
            if x is None:
                return "—"
            if isinstance(x, float) and np.isnan(x):
                return "—"
            return format(x, fmt)
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

    # CONTRADICTION detection on supplementary outcomes (named outcomes only)
    primary_sign = np.sign(g1["delta"]) if g1["delta"] is not None else 0
    contradictions = []
    sign_alignment = {
        "mfe_R_10d": True,
        "mae_R_10d": True,
        "failed_breakout_10d": False,
        "quality_20d": True,
        "time_to_MFE_10d": None,
    }
    for _, r in supp_df.iterrows():
        if r["status"] != "DESCRIPTIVE" or r["ci_excludes_zero"] is not True:
            continue
        if r["delta_mean"] is None:
            continue
        outcome = r["outcome"]
        positive_align = sign_alignment.get(outcome)
        if positive_align is None:
            continue
        delta_sign = np.sign(r["delta_mean"])
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

    # DROP_EARLY_DIAGNOSTIC census (descriptive only)
    lines.append("## DROP_EARLY_DIAGNOSTIC census (descriptive only — NO gate, NO interpretation)")
    lines.append("")
    lines.append(
        "Per spec §2: trigger/retest events with `earliness_score_pct < -5` are extended-cohort "
        "biology (Pilot 5/6 covered) and excluded from BASELINE. Reported here for transparency; "
        "**no gate, no claim, no trade rule derivable from these numbers**."
    )
    lines.append("")
    lines.append("| scope | n | n_with_outcome | mean_R | median_R | win_rate | profit_factor |")
    lines.append("|---|---|---|---|---|---|---|")
    for _, r in drop_census.iterrows():
        def _f(x, fmt=".4f"):
            if x is None:
                return "—"
            if isinstance(x, float) and np.isnan(x):
                return "—"
            return format(x, fmt)
        lines.append(
            f"| {r['scope']} | {r['n']} | {r['n_with_outcome']} | "
            f"{_f(r['mean_realized_R_10d'])} | {_f(r['median_realized_R_10d'])} | "
            f"{_f(r['win_rate'])} | {_f(r['profit_factor'], '.4f') if r['profit_factor'] is not None else '—'} |"
        )
    lines.append("")

    # PASS interpretation ceiling
    lines.append("## PASS Interpretation — LOCKED CEILING")
    lines.append("")
    if final_verdict == "PASS":
        lines.append("**Allowed PASS claim (the entire allowed claim, descriptive only)**:")
        lines.append("")
        lines.append(
            f"> Within HB `trigger ∪ retest_bounce` cohort, the EVENT_NEAR_TIER label "
            f"({EVENT_NEAR_LO} ≤ earliness_score_pct ≤ {EVENT_NEAR_HI}) produces a "
            f"statistically and practically meaningful descriptive `{PRIMARY_OUTCOME}` uplift over "
            f"an undifferentiated trigger/retest baseline (descriptive `Δmean ≥ {UPLIFT_FLOOR_R} R`, "
            f"p < {P_THRESHOLD}, state-stability non-contradicting, ATR mirror non-contradicting). "
            f"Tier label is a paper-only descriptive risk modifier candidate **for trigger/retest "
            f"only**, distinct from extended TIER_A_PAPER."
        )
        lines.append("")
        lines.append("**Allowed downstream actions on PASS**:")
        lines.append("- Drafting `memory/paper_execution_v0_trigger_retest_spec.md` becomes a CANDIDATE user-approval ask.")
        lines.append("  - Drafting itself requires explicit user ONAY; PASS does NOT auto-start drafting.")
        lines.append("- HB HTML note text update for trigger/retest rows (separate user-approval cycle).")
        lines.append("")
        lines.append("**PASS does NOT permit**:")
        lines.append("- Modifying extended TIER_A_PAPER definition or paper_execution_v0 spec.")
        lines.append("- Pooling trigger/retest with extended in any later spec.")
        lines.append("- Live trade gate opening (STAYS CLOSED).")
        lines.append("- EMA gate / scanner ranking / ML feature / classifier label / position sizing input.")
        lines.append("")
    elif final_verdict == "PARTIAL":
        lines.append(
            "**PARTIAL** — primary G1 PASS but G2 sanity / G5 stability / G6 atr mirror condition unmet; "
            "paper_execution_v0_trigger_retest_spec drafting NOT unlocked."
        )
        lines.append("")
    elif final_verdict == "CLOSED":
        lines.append(
            "**CLOSED** — hypothesis inverted or sanity-tier contradiction. NO 'short EVENT_NEAR' / "
            "'trade NON_EVENT_NEAR' shortcut. Pilot 1-6 closures UNCHANGED."
        )
        lines.append("")
    elif final_verdict == "FAIL":
        lines.append(
            "**FAIL** — primary uplift gate not passed. NO threshold sweep, NO neighbor-bucket "
            "promotion, NO Pilot 7b. Extended TIER_A_PAPER and Pilot 1-6 closures UNCHANGED."
        )
        lines.append("")
    elif final_verdict == "INSUFFICIENT":
        lines.append(
            "**INSUFFICIENT** — min-N floor not met. Panel frozen, INSUFFICIENT terminal. "
            "NOT FAIL. NO 'collect more data and rerun' on this artifact."
        )
        lines.append("")

    lines.append("**Forbidden interpretations (locked ceiling, all verdicts)**:")
    forbidden = [
        "Live trade priority / scanner ranking elevation",
        "Live entry timing recommendation",
        "Live entry filter ('only buy EVENT_NEAR_TIER signals')",
        "Hard or soft EMA gate in any scanner / decision_engine",
        "EMA ML feature / earliness as ranking-scoring sort key / tier as classifier label",
        "Position sizing change (paper or live)",
        "Pooling with extended TIER_A_PAPER ('they're both EMA tiers, just merge')",
        "Treating NON_EVENT_NEAR as a directive to short / avoid / exclude",
        "DROP_EARLY_DIAGNOSTIC reading translated into trigger/retest trade rule",
        "Generalizing to signal_state == 'extended' (different biology, different pilot)",
        "Forward-looking signature claim",
        "Cross-pilot cascade (Pilot 8/9 by analogy without own pre-reg)",
        "Re-running Pilot 7 with different threshold / outcome / tier definition / horizon",
        "Reopening Pilot 1/2/3/4/5/6 verdicts based on Pilot 7",
        "Removing extended-only HTML notes on Pilot 7 PASS",
        "Auto-emission of EVENT_NEAR_TIER flag in production HTML / decision_engine report / NOX briefing without separate per-target user approval",
        "Skipping paper-execution-shadow phase and going straight to live trade",
    ]
    for f in forbidden:
        lines.append(f"- ❌ {f}")
    lines.append("")

    lines.append("## Anti-tweak Confirmation")
    lines.append("")
    confs = [
        f"Pilot 5 panel reused (rows {EXPECTED_PILOT5_PANEL_ROWS:,}) — no recompute earliness/outcomes/geometry",
        f"HB event parquet reused (scanner_version {EXPECTED_SCANNER_VERSION}) — no rebuild",
        f"Threshold window LOCKED at [{EVENT_NEAR_LO},{EVENT_NEAR_HI}]; drop boundary LOCKED at < {DROP_BOUNDARY}; both derived from Pilot 3 state decomposition (no sweep, no variant, no neighbor-bucket promotion)",
        f"Primary outcome `{PRIMARY_OUTCOME}` LOCKED — no horizon sweep",
        f"Uplift magnitude floor {UPLIFT_FLOOR_R} R LOCKED",
        f"Min-N EVENT_NEAR_TIER {MIN_N_TIER} / NON_EVENT_NEAR {MIN_N_TIER} / BASELINE {MIN_N_BASELINE} LOCKED",
        f"p-threshold {P_THRESHOLD} LOCKED (no fallback)",
        f"Bootstrap n_boot={BOOTSTRAP_N}, seed={BOOTSTRAP_SEED}, group-independent resample",
        f"G2 = LABEL SANITY CHECK (NON_EVENT_NEAR ⊂ BASELINE → not independent test, used to detect warning-label pathology)",
        f"G5 = STATE STABILITY (trigger-only + retest-only secondary; slope/width sub-stratification dropped, supplementary descriptive only)",
        f"G6 ATR mirror = audit only; ATR-only PASS does NOT count",
        f"DROP_EARLY_DIAGNOSTIC reported descriptively, no gate, no interpretation, no claim",
        f"No new tier (no top-decile carving, no quartile inside EVENT_NEAR_TIER, no atr+pct agreement carve-out)",
        f"No regime-conditional stratification, no multi-TF confirmation",
        f"No new feature in tier definition (no ema21 reclaim, no stack mixed→bull, no slope-magnitude — those are Pilot 8/9 if ever)",
        f"Single authorized run — no post-hoc parameter change",
        f"TIER_A_PAPER (extended cohort) UNCHANGED on Pilot 7 PASS or FAIL",
        f"No pooling with extended cohort regardless of Pilot 7 verdict",
        f"Live trade gate STAYS CLOSED regardless of Pilot 7 verdict",
        f"INSUFFICIENT ≠ FAIL",
        f"Pilot 1-6 closure UNCHANGED regardless of Pilot 7 result",
        f"PASS ceiling: descriptive paper-tier label + paper_execution_v0_trigger_retest_spec drafting candidate user ask only",
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
    print("[Pilot 7] Validating inputs (16-item authorization checklist)...", flush=True)
    hb_manifest, p5_rows, boot_integrity = _validate_inputs()
    print(
        f"  HB manifest OK (scanner_version={hb_manifest.get('scanner_version')}, rows={hb_manifest.get('rows')})",
        flush=True,
    )
    print(f"  HB sha256 OK ({boot_integrity['hb_event_parquet_sha256'][:16]}…)", flush=True)
    print(
        f"  Earliness sha256 OK ({boot_integrity['earliness_parquet_sha256'][:16]}…)",
        flush=True,
    )
    print(f"  Pilot 5 panel rows={p5_rows}", flush=True)

    print("[Pilot 7] Tagging tiers on Pilot 5 panel...", flush=True)
    panel = _build_panel()
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

    g3_baseline_pass = n_baseline >= MIN_N_BASELINE

    # --------------------------------------------------------------
    # G1 — EVENT_NEAR_TIER vs BASELINE (uplift, primary)
    # --------------------------------------------------------------
    print("[Pilot 7] G1 (EVENT_NEAR_TIER vs BASELINE, pct primary)...", flush=True)
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

    # --------------------------------------------------------------
    # G2 — NON_EVENT_NEAR vs BASELINE (label sanity, NOT independence)
    # --------------------------------------------------------------
    print("[Pilot 7] G2 (NON_EVENT_NEAR vs BASELINE, label sanity)...", flush=True)
    b_arr = eligible.loc[eligible["ema_tr_tier"] == NON_EVENT_NEAR, PRIMARY_OUTCOME].astype(float).to_numpy()
    g2 = _gate_pair(
        b_arr, baseline_arr,
        magnitude_floor=0.0,  # no magnitude requirement on sanity gate
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

    # --------------------------------------------------------------
    # G6 — ATR mirror audit (EVENT_NEAR_TIER_atr vs BASELINE_atr)
    # --------------------------------------------------------------
    print("[Pilot 7] G6 (ATR mirror audit)...", flush=True)
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
        print(
            f"  G6 verdict={g6_g1['verdict']} | Δuplift_atr={g6_g1['delta']:.4f} R",
            flush=True,
        )
    else:
        print(f"  G6 verdict={g6_g1['verdict']} (INSUFFICIENT)", flush=True)

    # --------------------------------------------------------------
    # G5 — state stability (trigger-only, retest-only secondary)
    # --------------------------------------------------------------
    print("[Pilot 7] G5 (state stability: trigger-only + retest-only)...", flush=True)
    state_stability_df, g5_summary = _g5_state_stability(panel, "ema_tr_tier")
    print(
        f"  trigger classification={g5_summary['trigger_classification']} | "
        f"retest classification={g5_summary['retest_classification']} | "
        f"G5 {g5_summary['g5_status']}",
        flush=True,
    )

    # --------------------------------------------------------------
    # Supplementary descriptives
    # --------------------------------------------------------------
    print("[Pilot 7] Supplementary descriptives (EVENT_NEAR vs NON_EVENT_NEAR + slope/width)...", flush=True)
    supp_df = _supplementary_table(panel, "ema_tr_tier")

    # --------------------------------------------------------------
    # DROP_EARLY_DIAGNOSTIC census (descriptive only)
    # --------------------------------------------------------------
    print("[Pilot 7] DROP_EARLY_DIAGNOSTIC census (descriptive only)...", flush=True)
    drop_census = _drop_diagnostic_census(panel)

    # --------------------------------------------------------------
    # Final verdict
    # --------------------------------------------------------------
    print("[Pilot 7] Final verdict (decision tree)...", flush=True)
    final_verdict, notes = _final_verdict(g1, g2, g6_g1, g3_baseline_pass, g5_summary["g5_status"])
    print(f"  FINAL VERDICT: {final_verdict}", flush=True)

    runtime_s = time.time() - t0

    # --------------------------------------------------------------
    # Write outputs
    # --------------------------------------------------------------
    print("[Pilot 7] Writing outputs...", flush=True)
    panel.to_parquet(OUT_PANEL, index=False)
    _write_gates_main_csv(g1, g2, g6_g1, OUT_GATES_MAIN)
    state_stability_df.to_csv(OUT_STATE_STABILITY, index=False)
    supp_df.to_csv(OUT_SUPPLEMENTARY, index=False)
    drop_census.to_csv(OUT_DROP_CENSUS, index=False)
    summary_text = _format_summary(
        final_verdict,
        notes,
        g1, g2, g6_g1,
        g5_summary,
        state_stability_df,
        supp_df,
        drop_census,
        panel,
        runtime_s,
        hb_manifest,
        boot_integrity,
    )
    OUT_SUMMARY.write_text(summary_text)
    print(
        f"  Wrote: {OUT_PANEL.name} / {OUT_GATES_MAIN.name} / "
        f"{OUT_STATE_STABILITY.name} / {OUT_SUPPLEMENTARY.name} / "
        f"{OUT_DROP_CENSUS.name} / {OUT_SUMMARY.name}",
        flush=True,
    )
    print(f"[Pilot 7] Done in {runtime_s:.1f}s — verdict {final_verdict}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
