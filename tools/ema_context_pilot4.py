"""ema_context Pilot 4 — single authorized run.

Spec: memory/ema_context_pilot4_spec.md (LOCKED 2026-05-03)

Extended-state EMA earliness signature.
Fresh, ayrı pre-reg — Pilot 3 rescue DEĞİL.
Path A — outcome-free, descriptive event-level discrimination test.

Question: extended state events show event-level earliness_score distribution
that locates EMA compression earlier than pooled (trigger ∪ retest_bounce)?

Anti-tweak (locked):
  - Pilot 3 panel reuse only (no new data pull, STOP if missing/stale)
  - Same 7-bucket window family [-20, +10] (no new bucket boundary)
  - earliness_score = argmin offset within [-20, +10]; tie → earliest (null-favoring)
  - ≥25/31 non-null required per event else dropped (descriptive count)
  - KS p-threshold 0.001 (no fallback)
  - Bootstrap n_boot=2000, seed=7, group-independent resample
  - Magnitude floor 3 bars
  - Primary metric ema_stack_width_pct LOCKED; ATR mirror audit-only
  - Comparison A=extended vs B=pooled(trigger ∪ retest_bounce); trigger ↔ retest separate FORBIDDEN
  - ATR-only PASS does NOT count as Pilot 4 PASS
  - PASS interpretation ceiling: descriptive HTML note only — no trade logic / gate / ML / ranking / scoring / entry timing / forward outcome
  - INSUFFICIENT ≠ FAIL
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
EMA_CONTEXT_METADATA = PROJECT_ROOT / "output" / "ema_context_daily_metadata.json"
PILOT3_PANEL = PROJECT_ROOT / "output" / "ema_context_pilot3_panel.parquet"

OUT_EARLINESS = PROJECT_ROOT / "output" / "ema_context_pilot4_earliness_per_event.parquet"
OUT_DIST_PCT = PROJECT_ROOT / "output" / "ema_context_pilot4_distribution_pct.csv"
OUT_DIST_ATR = PROJECT_ROOT / "output" / "ema_context_pilot4_distribution_atr.csv"
OUT_GATES = PROJECT_ROOT / "output" / "ema_context_pilot4_gates.csv"
OUT_SUMMARY = PROJECT_ROOT / "output" / "ema_context_pilot4_summary.md"

# Window — same family as Pilot 3
OFFSET_MIN = -20
OFFSET_MAX = 10
N_OFFSETS_TOTAL = OFFSET_MAX - OFFSET_MIN + 1  # 31
NON_NULL_MIN_PER_EVENT = 25

# Statistical gates
KS_P_THRESHOLD = 0.001
BOOTSTRAP_N = 2000
BOOTSTRAP_SEED = 7
BOOTSTRAP_CI_LO_PCT = 2.5
BOOTSTRAP_CI_HI_PCT = 97.5
MAGNITUDE_FLOOR_BARS = 3.0
MIN_N_PER_GROUP = 200

# Group definitions (LOCKED)
GROUP_A = "extended"
GROUP_B_STATES = ("trigger", "retest_bounce")
GROUP_B_LABEL = "trigger_or_retest_bounce"

# Source manifest expectations
EXPECTED_SCANNER_VERSION = "1.4.0"
EXPECTED_HB_ROWS = 10470
EXPECTED_BREAKPOINTS_VERSION = "v0.0"
EXPECTED_PILOT3_PANEL_ROWS = 322011


# =============================================================================
# BOOT — fail-fast manifest check
# =============================================================================


def _validate_inputs() -> tuple[dict, dict, int]:
    if not HB_EVENT_PARQUET.exists():
        raise FileNotFoundError(f"HB event parquet missing: {HB_EVENT_PARQUET}")
    if not HB_EVENT_MANIFEST.exists():
        raise FileNotFoundError(f"HB event manifest missing: {HB_EVENT_MANIFEST}")
    if not EMA_CONTEXT_METADATA.exists():
        raise FileNotFoundError(f"ema_context metadata missing: {EMA_CONTEXT_METADATA}")
    if not PILOT3_PANEL.exists():
        raise FileNotFoundError(
            f"Pilot 3 panel missing: {PILOT3_PANEL}. "
            f"SPEC violation — Pilot 4 reuses Pilot 3 panel, no rebuild allowed. STOP."
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

    em_meta = json.loads(EMA_CONTEXT_METADATA.read_text())
    bp_version = em_meta.get("tag_breakpoints", {}).get("version")
    if bp_version != EXPECTED_BREAKPOINTS_VERSION:
        raise RuntimeError(
            f"ema_context breakpoints version mismatch: got {bp_version!r} != "
            f"{EXPECTED_BREAKPOINTS_VERSION!r}. SPEC violation."
        )

    panel_rows = pd.read_parquet(PILOT3_PANEL, columns=["event_id"]).shape[0]
    if panel_rows != EXPECTED_PILOT3_PANEL_ROWS:
        raise RuntimeError(
            f"Pilot 3 panel rows mismatch: got {panel_rows} != {EXPECTED_PILOT3_PANEL_ROWS}. "
            f"SPEC violation — panel may have been rebuilt; no rebuild allowed in Pilot 4."
        )

    return manifest, em_meta, panel_rows


# =============================================================================
# Earliness score computation
# =============================================================================


def _compute_earliness_per_event(panel: pd.DataFrame) -> pd.DataFrame:
    """Compute earliness_score per event for both pct and atr metrics.

    earliness_score = argmin offset within [-20, +10]; ties broken by earliest offset.
    """
    sub = panel[(panel["offset"] >= OFFSET_MIN) & (panel["offset"] <= OFFSET_MAX)].copy()

    # Sort to make tie-breaking deterministic: ascending by offset means idxmin returns
    # earliest offset on a plateau (pandas idxmin returns first occurrence of min).
    sub = sub.sort_values(["event_id", "offset"]).reset_index(drop=True)

    rows = []
    for eid, grp in sub.groupby("event_id", sort=False):
        signal_state = grp["signal_state"].iloc[0]
        ticker = grp["ticker"].iloc[0]

        # Non-null counts within window
        n_nonnull_pct = int(grp["pct"].notna().sum())
        n_nonnull_atr = int(grp["atr"].notna().sum())

        # earliness_score: argmin offset; pandas idxmin returns first occurrence,
        # which is the earliest offset because grp is offset-sorted ascending.
        if n_nonnull_pct >= NON_NULL_MIN_PER_EVENT:
            pct_min_idx = grp["pct"].idxmin()
            earliness_pct = int(grp.loc[pct_min_idx, "offset"])
        else:
            earliness_pct = None

        if n_nonnull_atr >= NON_NULL_MIN_PER_EVENT:
            atr_min_idx = grp["atr"].idxmin()
            earliness_atr = int(grp.loc[atr_min_idx, "offset"])
        else:
            earliness_atr = None

        rows.append({
            "event_id": int(eid),
            "ticker": ticker,
            "signal_state": signal_state,
            "n_offsets_nonnull_pct": n_nonnull_pct,
            "n_offsets_nonnull_atr": n_nonnull_atr,
            "earliness_score_pct": earliness_pct,
            "earliness_score_atr": earliness_atr,
        })

    return pd.DataFrame(rows)


# =============================================================================
# Statistical gates
# =============================================================================


def _ks_test(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Two-sample KS test. Returns (ks_stat, p_value)."""
    res = stats.ks_2samp(a, b, alternative="two-sided", mode="auto")
    return float(res.statistic), float(res.pvalue)


def _bootstrap_mean_diff(
    a: np.ndarray, b: np.ndarray, n_boot: int, seed: int
) -> dict:
    """Bootstrap mean(B) - mean(A) with group-independent resample.

    Sign convention: positive ⇒ extended (A) earlier than pooled (B).
    """
    rng = np.random.default_rng(seed)
    n_a = len(a)
    n_b = len(b)
    diffs = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sa = rng.choice(a, size=n_a, replace=True)
        sb = rng.choice(b, size=n_b, replace=True)
        diffs[i] = sb.mean() - sa.mean()
    point_estimate = float(b.mean() - a.mean())
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


def _evaluate_gates(
    earliness_df: pd.DataFrame, metric_col: str, label: str
) -> dict:
    """Evaluate G1 (KS), G2 (bootstrap), G3 (min-N) for a metric column."""
    valid = earliness_df.dropna(subset=[metric_col]).copy()

    a_mask = valid["signal_state"] == GROUP_A
    b_mask = valid["signal_state"].isin(GROUP_B_STATES)

    a_vals = valid.loc[a_mask, metric_col].astype(float).to_numpy()
    b_vals = valid.loc[b_mask, metric_col].astype(float).to_numpy()

    n_a = int(len(a_vals))
    n_b = int(len(b_vals))

    # G3 — min-N
    g3_pass = (n_a >= MIN_N_PER_GROUP) and (n_b >= MIN_N_PER_GROUP)
    g3_status = "PASS" if g3_pass else "INSUFFICIENT"

    # If insufficient, skip G1/G2 evaluation (return INSUFFICIENT shells)
    if not g3_pass:
        return {
            "metric": metric_col,
            "label": label,
            "n_a": n_a,
            "n_b": n_b,
            "mean_a": float(a_vals.mean()) if n_a > 0 else None,
            "mean_b": float(b_vals.mean()) if n_b > 0 else None,
            "g1_ks_stat": None,
            "g1_p_value": None,
            "g1_status": "INSUFFICIENT",
            "g2_point_estimate": None,
            "g2_ci_lo": None,
            "g2_ci_hi": None,
            "g2_ci_excludes_zero": None,
            "g2_magnitude_pass": None,
            "g2_status": "INSUFFICIENT",
            "g3_status": g3_status,
            "verdict": "INSUFFICIENT",
        }

    # G1 — KS
    ks_stat, p_value = _ks_test(a_vals, b_vals)
    g1_pass = p_value < KS_P_THRESHOLD

    # G2 — bootstrap mean diff
    boot = _bootstrap_mean_diff(a_vals, b_vals, BOOTSTRAP_N, BOOTSTRAP_SEED)
    g2_magnitude_pass = abs(boot["point_estimate"]) >= MAGNITUDE_FLOOR_BARS
    g2_pass = boot["ci_excludes_zero"] and g2_magnitude_pass

    # Sign-aware: spec says positive (mean_b - mean_a > 0) is hypothesis-aligned (A=extended earlier).
    # G2 PASS requires |Δ| ≥ 3 AND CI excludes 0 — both directions allowed numerically,
    # but sign reported transparently. Hypothesis-aligned PASS is point_estimate > 0.
    hypothesis_aligned = boot["point_estimate"] > 0

    # Verdict per spec:
    # PASS only if pct primary G1+G2+G3 PASS (and hypothesis-aligned for interpretation).
    # If any of G1/G2 fails on pct primary → CLOSED.
    if g1_pass and g2_pass:
        verdict = "PASS" if hypothesis_aligned else "CLOSED_WRONG_DIRECTION"
    else:
        verdict = "CLOSED"

    return {
        "metric": metric_col,
        "label": label,
        "n_a": n_a,
        "n_b": n_b,
        "mean_a": float(a_vals.mean()),
        "mean_b": float(b_vals.mean()),
        "g1_ks_stat": ks_stat,
        "g1_p_value": p_value,
        "g1_status": "PASS" if g1_pass else "FAIL",
        "g2_point_estimate": boot["point_estimate"],
        "g2_ci_lo": boot["ci_lo"],
        "g2_ci_hi": boot["ci_hi"],
        "g2_ci_excludes_zero": boot["ci_excludes_zero"],
        "g2_magnitude_pass": g2_magnitude_pass,
        "g2_status": "PASS" if g2_pass else "FAIL",
        "g3_status": g3_status,
        "hypothesis_aligned": hypothesis_aligned,
        "verdict": verdict,
    }


# =============================================================================
# Distribution histogram
# =============================================================================


def _build_distribution(earliness_df: pd.DataFrame, metric_col: str) -> pd.DataFrame:
    """Per-group histogram of earliness_score by offset (counts per group)."""
    valid = earliness_df.dropna(subset=[metric_col]).copy()
    valid["group"] = valid["signal_state"].apply(
        lambda s: "extended" if s == GROUP_A else (
            GROUP_B_LABEL if s in GROUP_B_STATES else "other"
        )
    )
    valid = valid[valid["group"] != "other"]
    offsets = list(range(OFFSET_MIN, OFFSET_MAX + 1))
    rows = []
    for off in offsets:
        ext_n = int(((valid["group"] == "extended") & (valid[metric_col] == off)).sum())
        pool_n = int(((valid["group"] == GROUP_B_LABEL) & (valid[metric_col] == off)).sum())
        rows.append({"offset": off, "n_extended": ext_n, "n_trigger_or_retest": pool_n})
    out = pd.DataFrame(rows)
    n_a = int((valid["group"] == "extended").sum())
    n_b = int((valid["group"] == GROUP_B_LABEL).sum())
    out["share_extended"] = out["n_extended"] / n_a if n_a > 0 else np.nan
    out["share_trigger_or_retest"] = out["n_trigger_or_retest"] / n_b if n_b > 0 else np.nan
    return out


# =============================================================================
# Output writers
# =============================================================================


def _write_gates_csv(gate_pct: dict, gate_atr: dict, path: Path) -> None:
    rows = [
        {**gate_pct, "primary_or_audit": "PRIMARY_pct"},
        {**gate_atr, "primary_or_audit": "AUDIT_atr"},
    ]
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)


def _format_summary(
    gate_pct: dict,
    gate_atr: dict,
    earliness_df: pd.DataFrame,
    dist_pct: pd.DataFrame,
    dist_atr: pd.DataFrame,
    runtime_s: float,
    em_meta: dict,
    hb_manifest: dict,
    panel_rows: int,
    n_events_dropped: int,
) -> str:
    lines: list[str] = []
    lines.append("# ema_context Pilot 4 — Run Summary")
    lines.append("")
    lines.append(f"- Spec: `memory/ema_context_pilot4_spec.md` LOCKED 2026-05-03")
    lines.append(f"- Run date: {pd.Timestamp.utcnow().isoformat()} (single authorized run)")
    lines.append(f"- Runtime: {runtime_s:.1f}s")
    lines.append(f"- HB event source: `output/horizontal_base_event_v1.parquet` "
                 f"(scanner_version {hb_manifest.get('scanner_version')}, frozen_at {hb_manifest.get('frozen_at')}, rows {hb_manifest.get('rows')})")
    lines.append(f"- ema_context source breakpoints: {em_meta.get('tag_breakpoints', {}).get('version')}")
    lines.append(f"- Pilot 3 panel reused: `output/ema_context_pilot3_panel.parquet` (rows {panel_rows:,})")
    lines.append(f"- Events dropped (non-null < {NON_NULL_MIN_PER_EVENT}/{N_OFFSETS_TOTAL}): {n_events_dropped:,} (descriptive — not FAIL)")
    lines.append("")
    lines.append("## Question (descriptive only)")
    lines.append("")
    lines.append(f"Does **extended** state events show event-level `earliness_score` distribution that locates EMA compression earlier than pooled (**trigger ∪ retest_bounce**)?")
    lines.append("")
    lines.append(f"- Group A (test): `extended` events")
    lines.append(f"- Group B (control): pooled (`trigger` ∪ `retest_bounce`) events")
    lines.append(f"- Comparison structure LOCKED — trigger ↔ retest_bounce ayrı ayrı karşılaştırma YASAK")
    lines.append("")

    # Verdict summary
    lines.append("## Overall Pilot 4 Verdict")
    lines.append("")
    pct_verdict = gate_pct["verdict"]
    atr_verdict = gate_atr["verdict"]
    pct_pass = pct_verdict == "PASS"
    atr_pass = atr_verdict == "PASS"

    lines.append(f"**Primary (ema_stack_width_pct): {pct_verdict}**")
    lines.append("")
    lines.append(f"- G1 KS: stat={gate_pct['g1_ks_stat']} p={gate_pct['g1_p_value']} → {gate_pct['g1_status']} (threshold p < {KS_P_THRESHOLD})")
    g2_disp = (
        f"point={gate_pct['g2_point_estimate']:.4f} bars  CI=[{gate_pct['g2_ci_lo']:.4f}, {gate_pct['g2_ci_hi']:.4f}]"
        if gate_pct['g2_point_estimate'] is not None else "N/A"
    )
    lines.append(f"- G2 Bootstrap (n_boot={BOOTSTRAP_N}, seed={BOOTSTRAP_SEED}): {g2_disp}")
    lines.append(
        f"  - CI excludes 0: {gate_pct['g2_ci_excludes_zero']}; |Δ| ≥ {MAGNITUDE_FLOOR_BARS} bars: {gate_pct['g2_magnitude_pass']}"
        f" → {gate_pct['g2_status']}"
    )
    lines.append(f"- G3 Min-N: N_extended={gate_pct['n_a']:,} / N_pooled={gate_pct['n_b']:,} (each ≥ {MIN_N_PER_GROUP}) → {gate_pct['g3_status']}")
    lines.append(f"- Mean earliness_score: extended={gate_pct['mean_a']:.4f}  pooled(trigger∪retest)={gate_pct['mean_b']:.4f}")
    if gate_pct.get("hypothesis_aligned") is not None:
        lines.append(f"- Hypothesis-aligned (extended earlier ⇒ point_estimate > 0): {gate_pct['hypothesis_aligned']}")
    lines.append("")

    lines.append(f"**Audit (ema_stack_width_atr): {atr_verdict}** — audit-only, ATR-only PASS does NOT count")
    lines.append("")
    lines.append(f"- G1 KS: stat={gate_atr['g1_ks_stat']} p={gate_atr['g1_p_value']} → {gate_atr['g1_status']}")
    g2_atr_disp = (
        f"point={gate_atr['g2_point_estimate']:.4f} bars  CI=[{gate_atr['g2_ci_lo']:.4f}, {gate_atr['g2_ci_hi']:.4f}]"
        if gate_atr['g2_point_estimate'] is not None else "N/A"
    )
    lines.append(f"- G2 Bootstrap: {g2_atr_disp}")
    if gate_atr['g2_status'] != "INSUFFICIENT":
        lines.append(
            f"  - CI excludes 0: {gate_atr['g2_ci_excludes_zero']}; |Δ| ≥ {MAGNITUDE_FLOOR_BARS} bars: {gate_atr['g2_magnitude_pass']}"
            f" → {gate_atr['g2_status']}"
        )
    lines.append(f"- G3 Min-N: N_extended={gate_atr['n_a']:,} / N_pooled={gate_atr['n_b']:,} → {gate_atr['g3_status']}")
    lines.append("")

    # G7 note
    lines.append("## G7 Audit (ATR-mirror — audit-only, NOT a gate)")
    lines.append("")
    if pct_pass and atr_pass:
        lines.append("**G7 NOTE**: pct primary PASS replicates on atr — descriptive cross-normalization signal.")
    elif pct_pass and not atr_pass:
        lines.append("**G7 NOTE**: pct primary PASS does NOT replicate on atr — pct-specific earliness signature (descriptive).")
    elif not pct_pass and atr_pass:
        lines.append("**G7 NOTE**: ATR-mirror PASSes but pct primary does NOT — per spec, ATR-only PASS does NOT count as Pilot 4 PASS. Treated as 'atr-only artifact'; verdict follows pct primary (CLOSED).")
    else:
        lines.append("**G7 NOTE**: neither pct nor atr passes — extended-state earliness signature not confirmed at either normalization.")
    lines.append("")

    # PASS interpretation ceiling
    lines.append("## PASS Interpretation — LOCKED CEILING")
    lines.append("")
    if pct_pass:
        lines.append("**Allowed PASS claim (the entire allowed claim, descriptive only)**:")
        lines.append("")
        lines.append("> Per Pilot 4, **extended HB events show event-level compression timing locus that distributes earlier than trigger/retest_bounce events** (descriptive, KS p<0.001 + bootstrap mean offset diff ≥ 3 bars CI excluding 0).")
        lines.append("")
        lines.append("**Forbidden interpretations even on PASS** (locked ceiling):")
    else:
        lines.append("Pilot 4 not PASS — extended-state earliness signature *not* statistically discriminated. Allowed PASS claim and ceiling listed for transparency:")
        lines.append("")
        lines.append("**Forbidden interpretations regardless of verdict** (locked ceiling):")
    lines.append("- ❌ \"Extended cohort daha iyi trade edilebilir\" / better trade outcome")
    lines.append("- ❌ \"Extended cohort'u önceliklendir\" / scanner ranking")
    lines.append("- ❌ \"Extended state'e EMA gate ekle\" / hard or soft gate")
    lines.append("- ❌ \"EMA ML feature\"")
    lines.append("- ❌ \"Earliness_score'u ranking-scoring'e ekle\"")
    lines.append("- ❌ \"Entry timing önerisi\"")
    lines.append("- ❌ Forward outcome / MFE / runner / horizon claim")
    lines.append("- ❌ Forward-looking signature: \"yeni events'te extended olmaya yatkın olanları erken yakala\"")
    lines.append("")
    lines.append("**Ceiling = descriptive HTML note only.**")
    lines.append("")

    # Distribution snapshots
    lines.append("## Distribution snapshots")
    lines.append("")
    lines.append(f"### earliness_score (pct primary) per offset")
    lines.append("")
    lines.append("| offset | n_extended | share_extended | n_trigger_or_retest | share_pooled |")
    lines.append("|---|---|---|---|---|")
    for _, r in dist_pct.iterrows():
        sh_e = f"{r['share_extended']:.4f}" if pd.notna(r['share_extended']) else "—"
        sh_p = f"{r['share_trigger_or_retest']:.4f}" if pd.notna(r['share_trigger_or_retest']) else "—"
        lines.append(f"| {int(r['offset']):+d} | {int(r['n_extended']):,} | {sh_e} | {int(r['n_trigger_or_retest']):,} | {sh_p} |")
    lines.append("")

    lines.append(f"### earliness_score (atr audit) per offset")
    lines.append("")
    lines.append("| offset | n_extended | share_extended | n_trigger_or_retest | share_pooled |")
    lines.append("|---|---|---|---|---|")
    for _, r in dist_atr.iterrows():
        sh_e = f"{r['share_extended']:.4f}" if pd.notna(r['share_extended']) else "—"
        sh_p = f"{r['share_trigger_or_retest']:.4f}" if pd.notna(r['share_trigger_or_retest']) else "—"
        lines.append(f"| {int(r['offset']):+d} | {int(r['n_extended']):,} | {sh_e} | {int(r['n_trigger_or_retest']):,} | {sh_p} |")
    lines.append("")

    # Anti-tweak confirmation
    lines.append("## Anti-tweak Confirmation")
    lines.append("")
    lines.append(f"- Pilot 3 panel reused (rows {panel_rows:,}) — no new data pull, no rebuild")
    lines.append(f"- Window family `[-20, +10]` (same as Pilot 3) — no boundary tweak")
    lines.append(f"- earliness_score = argmin offset; ties → earliest (null-favoring conservative)")
    lines.append(f"- Non-null filter ≥{NON_NULL_MIN_PER_EVENT}/{N_OFFSETS_TOTAL} per event")
    lines.append(f"- KS p-threshold {KS_P_THRESHOLD} LOCKED (no fallback)")
    lines.append(f"- Bootstrap n_boot={BOOTSTRAP_N}, seed={BOOTSTRAP_SEED}, group-independent resample")
    lines.append(f"- Magnitude floor {MAGNITUDE_FLOOR_BARS} bars LOCKED")
    lines.append(f"- Min-N per group {MIN_N_PER_GROUP} LOCKED")
    lines.append(f"- Primary metric ema_stack_width_pct LOCKED; ATR mirror audit-only")
    lines.append(f"- Comparison A=extended vs B=pooled(trigger ∪ retest_bounce); trigger ↔ retest ayrı YASAK")
    lines.append(f"- ATR-only PASS does NOT count as Pilot 4 PASS (G7 trap defense)")
    lines.append(f"- Single authorized run; no post-hoc parameter change")
    lines.append(f"- INSUFFICIENT ≠ FAIL")
    lines.append(f"- Pilot 3 closure NOT relitigated — fresh question, fresh methodology")
    lines.append(f"- Pilot 4 result CASCADE prohibited — no feature/gate/ML/ranking/scoring/entry timing/outcome claim regardless of verdict")
    lines.append(f"- No outcome metrics anywhere (PF/WR/lift/AUC vocabulary forbidden)")
    lines.append(f"- No Bonferroni / FDR (single-pair test by design)")
    lines.append(f"- Path A (outcome-free)")

    return "\n".join(lines) + "\n"


# =============================================================================
# Main orchestrator
# =============================================================================


def main() -> int:
    print("=== ema_context Pilot 4 — single authorized run ===")
    print("Spec: memory/ema_context_pilot4_spec.md (LOCKED 2026-05-03)")
    print()
    t_start = time.time()

    # [1/5] Boot
    print("[1/5] Validating inputs (manifest + breakpoints + Pilot 3 panel)...")
    hb_manifest, em_meta, panel_rows = _validate_inputs()
    print(f"  HB scanner_version={hb_manifest['scanner_version']} rows={hb_manifest['rows']}")
    print(f"  ema_context breakpoints={em_meta['tag_breakpoints']['version']}")
    print(f"  Pilot 3 panel rows={panel_rows:,}")

    # [2/5] Load panel
    print("[2/5] Loading Pilot 3 panel...")
    t0 = time.time()
    panel = pd.read_parquet(PILOT3_PANEL)
    print(f"  panel: {len(panel):,} rows / {panel['event_id'].nunique():,} events / {panel['ticker'].nunique()} tickers in {time.time()-t0:.2f}s")
    print(f"  signal_state dist: {panel.groupby('event_id')['signal_state'].first().value_counts().to_dict()}")

    # [3/5] Compute earliness per event
    print("[3/5] Computing earliness_score per event (argmin offset, tie → earliest)...")
    t0 = time.time()
    earliness_df = _compute_earliness_per_event(panel)
    n_events_total = len(earliness_df)
    n_events_with_pct = int(earliness_df["earliness_score_pct"].notna().sum())
    n_events_with_atr = int(earliness_df["earliness_score_atr"].notna().sum())
    n_dropped_pct = n_events_total - n_events_with_pct
    print(f"  earliness rows: {n_events_total:,} events  (pct usable: {n_events_with_pct:,} / atr usable: {n_events_with_atr:,})  dropped(pct<{NON_NULL_MIN_PER_EVENT}): {n_dropped_pct:,}  in {time.time()-t0:.2f}s")
    print(f"  state×pct earliness mean: {earliness_df.dropna(subset=['earliness_score_pct']).groupby('signal_state')['earliness_score_pct'].agg(['mean','count']).to_dict()}")

    # [4/5] Evaluate gates (pct primary + atr audit)
    print("[4/5] Evaluating gates G1+G2+G3 on pct primary and atr audit...")
    gate_pct = _evaluate_gates(earliness_df, "earliness_score_pct", "PRIMARY_pct")
    gate_atr = _evaluate_gates(earliness_df, "earliness_score_atr", "AUDIT_atr")
    print(f"  PRIMARY pct: G1={gate_pct['g1_status']} (KS p={gate_pct['g1_p_value']}) / G2={gate_pct['g2_status']} (Δ={gate_pct['g2_point_estimate']} CI=[{gate_pct['g2_ci_lo']}, {gate_pct['g2_ci_hi']}]) / G3={gate_pct['g3_status']} → verdict={gate_pct['verdict']}")
    print(f"  AUDIT  atr: G1={gate_atr['g1_status']} (KS p={gate_atr['g1_p_value']}) / G2={gate_atr['g2_status']} (Δ={gate_atr['g2_point_estimate']} CI=[{gate_atr['g2_ci_lo']}, {gate_atr['g2_ci_hi']}]) / G3={gate_atr['g3_status']} → verdict={gate_atr['verdict']}")

    # [5/5] Distribution + outputs
    print("[5/5] Building distributions + writing outputs...")
    dist_pct = _build_distribution(earliness_df, "earliness_score_pct")
    dist_atr = _build_distribution(earliness_df, "earliness_score_atr")

    earliness_df.to_parquet(OUT_EARLINESS, index=False)
    dist_pct.to_csv(OUT_DIST_PCT, index=False)
    dist_atr.to_csv(OUT_DIST_ATR, index=False)
    _write_gates_csv(gate_pct, gate_atr, OUT_GATES)

    runtime_s = time.time() - t_start
    summary = _format_summary(
        gate_pct, gate_atr,
        earliness_df, dist_pct, dist_atr,
        runtime_s, em_meta, hb_manifest,
        panel_rows, n_dropped_pct,
    )
    OUT_SUMMARY.write_text(summary)
    print(f"  earliness: {OUT_EARLINESS}")
    print(f"  distributions: {OUT_DIST_PCT} + {OUT_DIST_ATR}")
    print(f"  gates: {OUT_GATES}")
    print(f"  summary: {OUT_SUMMARY}")

    print()
    print(f"=== Overall Pilot 4 Verdict (pct primary): {gate_pct['verdict']} ===")
    print(f"Runtime: {runtime_s:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
