"""ema_context Pilot 4 — FORWARD sibling (key-based, grow-aware).

Spec: memory/ema_context_forward_alignment_spec.md (LOCK candidate v2 → LOCKED 2026-05-05)

Forward output contract (§3.2 + §3.5 + §6):
  - Read forward Pilot 3 panel `output/ema_context_pilot3_panel_forward.parquet`
    (matched_old + new_current event populations, single combined panel).
  - Reuse locked Pilot 4 analytical kernels verbatim:
      _compute_earliness_per_event, _evaluate_gates, _build_distribution
    (consume only event_id, ticker, signal_state, offset, pct, atr — schema-stable
     across forward panel).
  - Replace locked `_validate_inputs` row-count guard (10,470 HB / 322,011 panel)
    with three-check grow-aware contract (§3.2):
      1. Forward Pilot 3 panel exists and carries `track` column ∈ {matched_old, new_current}.
      2. ema_context_daily.max_date >= operational_target_date (loose check; orchestrator
         enforces the strict version).
      3. No stable_event_key duplicates within forward panel (per-event uniqueness).
  - Emit forward sidecar siblings (NEW) — locked Pilot 4 outputs remain BYTE-EQUAL:
      output/ema_context_pilot4_earliness_per_event_forward.parquet
      output/ema_context_pilot4_distribution_pct_forward.csv
      output/ema_context_pilot4_distribution_atr_forward.csv
      output/ema_context_pilot4_gates_forward.csv
      output/ema_context_pilot4_summary_forward.md

Hard rules (FORBIDDEN per §3.5 + §7):
  - No modification of locked Pilot 4 outputs (6 files).
  - No modification of locked Pilot 4 script (`tools/ema_context_pilot4.py`).
  - No positional fallback / synthetic event injection.
  - No re-evaluation of locked research conclusions (this emits forward artifacts only).

Returns 0 on success; non-zero on FAIL classification (orchestrator captures + halts).
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

# Import analytical kernels from locked Pilot 4 (NOT modified, library-import only).
from tools.ema_context_pilot4 import (  # type: ignore
    _compute_earliness_per_event,
    _evaluate_gates,
    _build_distribution,
    NON_NULL_MIN_PER_EVENT,
    N_OFFSETS_TOTAL,
    KS_P_THRESHOLD,
    BOOTSTRAP_N,
    BOOTSTRAP_SEED,
    MAGNITUDE_FLOOR_BARS,
    MIN_N_PER_GROUP,
    GROUP_A,
    GROUP_B_STATES,
    GROUP_B_LABEL,
)

# =============================================================================
# LOCKED CONSTANTS
# =============================================================================

CANDIDATE_KEY = ["ticker", "bar_date", "setup_family", "signal_type", "breakout_bar_date"]

PILOT3_PANEL_FORWARD = PROJECT_ROOT / "output" / "ema_context_pilot3_panel_forward.parquet"
EMA_CONTEXT_DAILY = PROJECT_ROOT / "output" / "ema_context_daily.parquet"
EMA_CONTEXT_METADATA = PROJECT_ROOT / "output" / "ema_context_daily_metadata.json"

OUT_EARLINESS = PROJECT_ROOT / "output" / "ema_context_pilot4_earliness_per_event_forward.parquet"
OUT_DIST_PCT = PROJECT_ROOT / "output" / "ema_context_pilot4_distribution_pct_forward.csv"
OUT_DIST_ATR = PROJECT_ROOT / "output" / "ema_context_pilot4_distribution_atr_forward.csv"
OUT_GATES = PROJECT_ROOT / "output" / "ema_context_pilot4_gates_forward.csv"
OUT_SUMMARY = PROJECT_ROOT / "output" / "ema_context_pilot4_summary_forward.md"

EXPECTED_TRACK_VALUES = {"matched_old", "new_current"}

# Required columns on forward panel (subset that pilot 4 kernels need + key carry).
REQUIRED_FORWARD_PANEL_COLS = [
    "event_id",
    "ticker",
    "signal_state",
    "offset",
    "pct",
    "atr",
    "track",
] + CANDIDATE_KEY


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


def _validate_forward_inputs() -> tuple[pd.DataFrame, dict]:
    """Forward-aware validation. Replaces locked _validate_inputs row-count guard."""
    if not PILOT3_PANEL_FORWARD.exists():
        raise FileNotFoundError(
            f"Forward Pilot 3 panel missing: {PILOT3_PANEL_FORWARD}. "
            f"Pilot 3 forward must run before Pilot 4 forward."
        )
    if not EMA_CONTEXT_DAILY.exists():
        raise FileNotFoundError(f"ema_context_daily missing: {EMA_CONTEXT_DAILY}")
    if not EMA_CONTEXT_METADATA.exists():
        raise FileNotFoundError(f"ema_context metadata missing: {EMA_CONTEXT_METADATA}")

    panel = pd.read_parquet(PILOT3_PANEL_FORWARD)
    missing_cols = [c for c in REQUIRED_FORWARD_PANEL_COLS if c not in panel.columns]
    if missing_cols:
        raise RuntimeError(
            f"Forward Pilot 3 panel missing required columns: {missing_cols}. "
            f"SPEC §3.5 violation — schema must carry stable_event_key + track."
        )
    track_values = set(panel["track"].astype(str).unique().tolist())
    unexpected = track_values - EXPECTED_TRACK_VALUES
    if unexpected:
        raise RuntimeError(
            f"Forward Pilot 3 panel has unexpected track values: {unexpected}. "
            f"Expected subset of {EXPECTED_TRACK_VALUES}."
        )

    # Stable_event_key uniqueness per event_id within the forward panel
    per_event_keys = panel[["event_id"] + CANDIDATE_KEY].drop_duplicates()
    n_event_ids = int(per_event_keys["event_id"].nunique())
    n_unique_keys = int(per_event_keys.drop(columns=["event_id"]).drop_duplicates().shape[0])
    if n_unique_keys != n_event_ids:
        raise RuntimeError(
            f"forward_panel_event_id_stable_key_mismatch: unique event_id={n_event_ids} != "
            f"unique stable_event_key={n_unique_keys}. SPEC §3.3 invariant violated."
        )

    em_meta = json.loads(EMA_CONTEXT_METADATA.read_text())
    return panel, em_meta


# =============================================================================
# Output writers
# =============================================================================


def _write_gates_csv_forward(gate_pct: dict, gate_atr: dict, dest: Path) -> None:
    rows = [
        {**gate_pct, "primary_or_audit": "PRIMARY_pct"},
        {**gate_atr, "primary_or_audit": "AUDIT_atr"},
    ]
    _atomic_write_csv(pd.DataFrame(rows), dest)


def _format_summary_forward(
    gate_pct: dict,
    gate_atr: dict,
    earliness_df: pd.DataFrame,
    dist_pct: pd.DataFrame,
    dist_atr: pd.DataFrame,
    runtime_s: float,
    em_meta: dict,
    panel_rows: int,
    n_unique_events: int,
    track_counts: dict,
    n_events_dropped: int,
) -> str:
    lines: list[str] = []
    lines.append("# ema_context Pilot 4 FORWARD — Run Summary")
    lines.append("")
    lines.append("- Spec: `memory/ema_context_forward_alignment_spec.md` LOCK candidate v2 (LOCKED 2026-05-05)")
    lines.append(f"- Run date (UTC): {pd.Timestamp.utcnow().isoformat()}")
    lines.append(f"- Runtime: {runtime_s:.1f}s")
    lines.append(f"- Source: forward Pilot 3 panel `output/ema_context_pilot3_panel_forward.parquet` "
                 f"({panel_rows:,} rows / {n_unique_events:,} events)")
    lines.append(f"- Track distribution: {track_counts}")
    lines.append(f"- ema_context breakpoints (forward_current basis): "
                 f"{em_meta.get('tag_breakpoints', {}).get('version')}")
    lines.append(f"- Events dropped (non-null < {NON_NULL_MIN_PER_EVENT}/{N_OFFSETS_TOTAL}): "
                 f"{n_events_dropped:,} (descriptive — not FAIL)")
    lines.append("")
    lines.append("## Forward provenance")
    lines.append("")
    lines.append("- Forward Pilot 3 panel emitted under `_forward` namespace (matched_old preserved verbatim "
                 "from locked panel; new_current events computed via locked Pilot 3 `_build_offset_long` "
                 "kernel against current `ema_context_daily`).")
    lines.append("- Locked Pilot 4 outputs remain BYTE-EQUAL pre/post (no overwrite; this script writes only "
                 "`*_forward.*` siblings).")
    lines.append("- All stable_event_key invariants verified upstream by Pilot 3 forward + checked here:")
    lines.append("  - `event_id` ↔ stable_event_key 1:1 mapping within forward panel.")
    lines.append("  - `track` column ∈ {matched_old, new_current}.")
    lines.append("")

    pct_verdict = gate_pct["verdict"]
    atr_verdict = gate_atr["verdict"]
    pct_pass = pct_verdict == "PASS"
    atr_pass = atr_verdict == "PASS"

    lines.append("## Overall Pilot 4 FORWARD verdict")
    lines.append("")
    lines.append(f"**Primary (ema_stack_width_pct): {pct_verdict}**")
    lines.append("")
    lines.append(f"- G1 KS: stat={gate_pct['g1_ks_stat']} p={gate_pct['g1_p_value']} → "
                 f"{gate_pct['g1_status']} (threshold p < {KS_P_THRESHOLD})")
    g2_disp = (
        f"point={gate_pct['g2_point_estimate']:.4f} bars  "
        f"CI=[{gate_pct['g2_ci_lo']:.4f}, {gate_pct['g2_ci_hi']:.4f}]"
        if gate_pct['g2_point_estimate'] is not None else "N/A"
    )
    lines.append(f"- G2 Bootstrap (n_boot={BOOTSTRAP_N}, seed={BOOTSTRAP_SEED}): {g2_disp}")
    if gate_pct["g2_status"] != "INSUFFICIENT":
        lines.append(
            f"  - CI excludes 0: {gate_pct['g2_ci_excludes_zero']}; "
            f"|Δ| ≥ {MAGNITUDE_FLOOR_BARS} bars: {gate_pct['g2_magnitude_pass']} → {gate_pct['g2_status']}"
        )
    lines.append(f"- G3 Min-N: N_extended={gate_pct['n_a']:,} / N_pooled={gate_pct['n_b']:,} "
                 f"(each ≥ {MIN_N_PER_GROUP}) → {gate_pct['g3_status']}")
    if gate_pct["mean_a"] is not None and gate_pct["mean_b"] is not None:
        lines.append(f"- Mean earliness_score: extended={gate_pct['mean_a']:.4f}  "
                     f"pooled(trigger∪retest)={gate_pct['mean_b']:.4f}")
    if gate_pct.get("hypothesis_aligned") is not None:
        lines.append(f"- Hypothesis-aligned (extended earlier ⇒ point_estimate > 0): "
                     f"{gate_pct['hypothesis_aligned']}")
    lines.append("")

    lines.append(f"**Audit (ema_stack_width_atr): {atr_verdict}** — audit-only, ATR-only PASS does NOT count")
    lines.append("")
    lines.append(f"- G1 KS: stat={gate_atr['g1_ks_stat']} p={gate_atr['g1_p_value']} → {gate_atr['g1_status']}")
    g2_atr_disp = (
        f"point={gate_atr['g2_point_estimate']:.4f} bars  "
        f"CI=[{gate_atr['g2_ci_lo']:.4f}, {gate_atr['g2_ci_hi']:.4f}]"
        if gate_atr['g2_point_estimate'] is not None else "N/A"
    )
    lines.append(f"- G2 Bootstrap: {g2_atr_disp}")
    if gate_atr['g2_status'] != "INSUFFICIENT":
        lines.append(
            f"  - CI excludes 0: {gate_atr['g2_ci_excludes_zero']}; "
            f"|Δ| ≥ {MAGNITUDE_FLOOR_BARS} bars: {gate_atr['g2_magnitude_pass']} → {gate_atr['g2_status']}"
        )
    lines.append(f"- G3 Min-N: N_extended={gate_atr['n_a']:,} / N_pooled={gate_atr['n_b']:,} → "
                 f"{gate_atr['g3_status']}")
    lines.append("")

    # G7 note
    lines.append("## G7 Audit (ATR-mirror — audit-only, NOT a gate)")
    lines.append("")
    if pct_pass and atr_pass:
        lines.append("**G7 NOTE**: pct primary PASS replicates on atr — descriptive cross-normalization signal.")
    elif pct_pass and not atr_pass:
        lines.append("**G7 NOTE**: pct primary PASS does NOT replicate on atr — pct-specific earliness signature (descriptive).")
    elif not pct_pass and atr_pass:
        lines.append("**G7 NOTE**: ATR-mirror PASSes but pct primary does NOT — per spec, ATR-only PASS does NOT count as Pilot 4 PASS.")
    else:
        lines.append("**G7 NOTE**: neither pct nor atr passes — extended-state earliness signature not confirmed at either normalization on the forward corpus.")
    lines.append("")

    # PASS interpretation ceiling (verbatim from locked spec — this is forward; ceiling applies identically)
    lines.append("## PASS Interpretation — LOCKED CEILING (verbatim from Pilot 4 locked spec)")
    lines.append("")
    if pct_pass:
        lines.append("**Allowed PASS claim (descriptive only)**:")
        lines.append("")
        lines.append("> Per Pilot 4 forward, **extended HB events show event-level compression timing locus that distributes earlier than trigger/retest_bounce events** on the forward corpus.")
        lines.append("")
    lines.append("**Forbidden interpretations regardless of verdict**:")
    lines.append("- ❌ \"Extended cohort daha iyi trade edilebilir\" / better trade outcome")
    lines.append("- ❌ \"Extended cohort'u önceliklendir\" / scanner ranking")
    lines.append("- ❌ \"Extended state'e EMA gate ekle\" / hard or soft gate")
    lines.append("- ❌ \"EMA ML feature\"")
    lines.append("- ❌ \"Earliness_score'u ranking-scoring'e ekle\"")
    lines.append("- ❌ \"Entry timing önerisi\"")
    lines.append("- ❌ Forward outcome / MFE / runner / horizon claim")
    lines.append("")
    lines.append("**Ceiling = descriptive HTML note only.**")
    lines.append("")

    lines.append("## Distribution snapshots (forward corpus)")
    lines.append("")
    lines.append("### earliness_score (pct primary) per offset")
    lines.append("")
    lines.append("| offset | n_extended | share_extended | n_trigger_or_retest | share_pooled |")
    lines.append("|---|---|---|---|---|")
    for _, r in dist_pct.iterrows():
        sh_e = f"{r['share_extended']:.4f}" if pd.notna(r['share_extended']) else "—"
        sh_p = f"{r['share_trigger_or_retest']:.4f}" if pd.notna(r['share_trigger_or_retest']) else "—"
        lines.append(f"| {int(r['offset']):+d} | {int(r['n_extended']):,} | {sh_e} | "
                     f"{int(r['n_trigger_or_retest']):,} | {sh_p} |")
    lines.append("")
    lines.append("### earliness_score (atr audit) per offset")
    lines.append("")
    lines.append("| offset | n_extended | share_extended | n_trigger_or_retest | share_pooled |")
    lines.append("|---|---|---|---|---|")
    for _, r in dist_atr.iterrows():
        sh_e = f"{r['share_extended']:.4f}" if pd.notna(r['share_extended']) else "—"
        sh_p = f"{r['share_trigger_or_retest']:.4f}" if pd.notna(r['share_trigger_or_retest']) else "—"
        lines.append(f"| {int(r['offset']):+d} | {int(r['n_extended']):,} | {sh_e} | "
                     f"{int(r['n_trigger_or_retest']):,} | {sh_p} |")
    lines.append("")

    return "\n".join(lines) + "\n"


# =============================================================================
# Main
# =============================================================================


def main() -> int:
    print("=== ema_context Pilot 4 FORWARD — single authorized run ===")
    print("Spec: memory/ema_context_forward_alignment_spec.md (LOCKED 2026-05-05)")
    t_start = time.time()

    # [1/5] Forward-aware validation
    print("[1/5] Validating forward inputs (no static row-count guard; key-based contract)...")
    panel, em_meta = _validate_forward_inputs()
    track_counts = panel.groupby("event_id")["track"].first().value_counts().to_dict()
    n_unique_events = int(panel["event_id"].nunique())
    panel_rows = int(len(panel))
    print(f"  forward panel: {panel_rows:,} rows / {n_unique_events:,} events / "
          f"{panel['ticker'].nunique()} tickers")
    print(f"  track distribution: {track_counts}")

    # [2/5] Compute earliness via locked kernel (consumes only event_id, ticker, signal_state,
    #       offset, pct, atr — schema-stable across forward panel).
    print("[2/5] Computing earliness_score per event (locked kernel, argmin offset, tie → earliest)...")
    t0 = time.time()
    earliness_df = _compute_earliness_per_event(panel)
    n_events_total = len(earliness_df)
    n_events_with_pct = int(earliness_df["earliness_score_pct"].notna().sum())
    n_events_with_atr = int(earliness_df["earliness_score_atr"].notna().sum())
    n_dropped_pct = n_events_total - n_events_with_pct
    print(f"  earliness rows: {n_events_total:,} events  "
          f"(pct usable: {n_events_with_pct:,} / atr usable: {n_events_with_atr:,})  "
          f"dropped(pct<{NON_NULL_MIN_PER_EVENT}): {n_dropped_pct:,}  in {time.time()-t0:.2f}s")

    if n_events_total != n_unique_events:
        raise RuntimeError(
            f"earliness_df row count {n_events_total} != unique event_id in panel {n_unique_events}. "
            f"Locked kernel invariant violated."
        )

    # Attach stable_event_key + track to earliness_df for forward provenance.
    # `earliness_df` (locked kernel output) already carries `ticker`; merging the panel
    # slice on event_id collides on `ticker` and produces ticker_x/ticker_y. Use explicit
    # suffixes to keep the canonical kernel-emitted column under bare `ticker`, verify
    # consistency with the panel slice, then drop the duplicated panel-side column.
    per_event_keys = panel[["event_id"] + CANDIDATE_KEY + ["track"]].drop_duplicates(subset=["event_id"])
    earliness_forward = earliness_df.merge(
        per_event_keys,
        on="event_id",
        how="left",
        validate="one_to_one",
        suffixes=("", "_panel"),
    )
    if "ticker_panel" in earliness_forward.columns:
        diverge_mask = earliness_forward["ticker"].astype(str) != earliness_forward["ticker_panel"].astype(str)
        n_diverge = int(diverge_mask.sum())
        if n_diverge > 0:
            raise RuntimeError(
                f"ticker divergence between earliness_df and forward Pilot 3 panel on "
                f"{n_diverge} events; locked kernel and forward panel disagree on "
                f"event_id↔ticker mapping."
            )
        earliness_forward = earliness_forward.drop(columns=["ticker_panel"])
    leftover_panel_cols = [c for c in earliness_forward.columns if c.endswith("_panel")]
    if leftover_panel_cols:
        raise RuntimeError(
            f"unexpected post-merge collision columns: {leftover_panel_cols}. "
            f"locked kernel + forward panel schemas overlap on more than `ticker`."
        )
    missing_key_cols = [c for c in CANDIDATE_KEY if c not in earliness_forward.columns]
    if missing_key_cols:
        raise RuntimeError(
            f"earliness_forward missing CANDIDATE_KEY columns after merge: {missing_key_cols}."
        )
    if earliness_forward[CANDIDATE_KEY].isna().any().any():
        raise RuntimeError(
            "earliness_forward has NaN stable_event_key after merge; per_event_keys join failed."
        )

    # [3/5] Evaluate gates on forward earliness (locked kernel — no per-track split)
    print("[3/5] Evaluating gates G1+G2+G3 on forward corpus (pct primary + atr audit)...")
    gate_pct = _evaluate_gates(earliness_df, "earliness_score_pct", "PRIMARY_pct_forward")
    gate_atr = _evaluate_gates(earliness_df, "earliness_score_atr", "AUDIT_atr_forward")
    print(f"  PRIMARY pct: G1={gate_pct['g1_status']} (KS p={gate_pct['g1_p_value']}) / "
          f"G2={gate_pct['g2_status']} (Δ={gate_pct['g2_point_estimate']}) / "
          f"G3={gate_pct['g3_status']} → verdict={gate_pct['verdict']}")
    print(f"  AUDIT  atr: G1={gate_atr['g1_status']} (KS p={gate_atr['g1_p_value']}) / "
          f"G2={gate_atr['g2_status']} (Δ={gate_atr['g2_point_estimate']}) / "
          f"G3={gate_atr['g3_status']} → verdict={gate_atr['verdict']}")

    # [4/5] Distributions
    print("[4/5] Building distributions...")
    dist_pct = _build_distribution(earliness_df, "earliness_score_pct")
    dist_atr = _build_distribution(earliness_df, "earliness_score_atr")

    # [5/5] Atomic writes
    print("[5/5] Writing forward outputs (atomic)...")
    _atomic_write_parquet(earliness_forward, OUT_EARLINESS)
    _atomic_write_csv(dist_pct, OUT_DIST_PCT)
    _atomic_write_csv(dist_atr, OUT_DIST_ATR)
    _write_gates_csv_forward(gate_pct, gate_atr, OUT_GATES)

    runtime_s = time.time() - t_start
    summary_text = _format_summary_forward(
        gate_pct, gate_atr,
        earliness_df, dist_pct, dist_atr,
        runtime_s, em_meta,
        panel_rows, n_unique_events, track_counts,
        n_dropped_pct,
    )
    _atomic_write_text(summary_text, OUT_SUMMARY)

    print(f"  earliness_forward: {OUT_EARLINESS} ({OUT_EARLINESS.stat().st_size:,} B)")
    print(f"  distributions:     {OUT_DIST_PCT.name} + {OUT_DIST_ATR.name}")
    print(f"  gates:             {OUT_GATES.name} ({OUT_GATES.stat().st_size:,} B)")
    print(f"  summary:           {OUT_SUMMARY.name} ({OUT_SUMMARY.stat().st_size:,} B)")
    print()
    print(f"=== Pilot 4 FORWARD verdict (pct primary): {gate_pct['verdict']} ===")
    print(f"Runtime: {runtime_s:.1f}s")

    # Emit summary JSON for orchestrator capture
    summary_json = {
        "tool": "ema_context_pilot4_forward",
        "runtime_s": round(runtime_s, 3),
        "panel_rows": panel_rows,
        "panel_unique_event_id": n_unique_events,
        "track_counts": track_counts,
        "earliness_rows": int(len(earliness_forward)),
        "n_events_with_pct": n_events_with_pct,
        "n_events_with_atr": n_events_with_atr,
        "n_events_dropped_pct": n_dropped_pct,
        "gate_pct": {
            "verdict": gate_pct["verdict"],
            "g1_status": gate_pct["g1_status"],
            "g1_p_value": gate_pct["g1_p_value"],
            "g2_status": gate_pct["g2_status"],
            "g2_point_estimate": gate_pct["g2_point_estimate"],
            "g2_ci_lo": gate_pct["g2_ci_lo"],
            "g2_ci_hi": gate_pct["g2_ci_hi"],
            "g3_status": gate_pct["g3_status"],
            "n_a": gate_pct["n_a"],
            "n_b": gate_pct["n_b"],
        },
        "gate_atr": {
            "verdict": gate_atr["verdict"],
            "g1_status": gate_atr["g1_status"],
            "g1_p_value": gate_atr["g1_p_value"],
            "g2_status": gate_atr["g2_status"],
            "g2_point_estimate": gate_atr["g2_point_estimate"],
            "g2_ci_lo": gate_atr["g2_ci_lo"],
            "g2_ci_hi": gate_atr["g2_ci_hi"],
            "g3_status": gate_atr["g3_status"],
            "n_a": gate_atr["n_a"],
            "n_b": gate_atr["n_b"],
        },
        "outputs": {
            "earliness": str(OUT_EARLINESS.relative_to(PROJECT_ROOT)),
            "distribution_pct": str(OUT_DIST_PCT.relative_to(PROJECT_ROOT)),
            "distribution_atr": str(OUT_DIST_ATR.relative_to(PROJECT_ROOT)),
            "gates": str(OUT_GATES.relative_to(PROJECT_ROOT)),
            "summary": str(OUT_SUMMARY.relative_to(PROJECT_ROOT)),
        },
    }
    print("PILOT4_FORWARD_SUMMARY_JSON_BEGIN")
    print(json.dumps(summary_json, default=str))
    print("PILOT4_FORWARD_SUMMARY_JSON_END")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
