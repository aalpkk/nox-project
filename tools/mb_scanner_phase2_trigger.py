"""Phase 2 salvage-trigger probe for mb_scanner Phase 1 FAIL cohorts.

Per locked spec `mb_scanner_phase1_event_quality_v1` §"Salvage trigger":
  "If a cohort FAILs but top-decile-by-score lift CI excludes 1.0× →
   trigger Phase 2 (score-bucket diagnostic)."

This script implements that conditional check on the Phase 1 outputs.
For each FAIL cohort and 5 locked candidate scores, we compute top-decile
mean forward return at each horizon and bootstrap a 95% CI on the
difference (top_decile_mean - cohort_mean). A cell "triggers" if its CI
lower bound > 0 at the per-cell alpha (Bonferroni-adjusted across the
25 cells per cohort: alpha = 0.05 / 25 = 0.002 → 99.8% CI).

A cohort SALVAGE_TRIGGER fires iff at least one (score, horizon) cell
triggers. Cohorts whose trigger does not fire flow toward formal closure
per Phase 1 closure rule (no rescue without explicit reframe).

This is a CHEAP probe. It does NOT train a model. It only signals
whether Phase 2 (the actual ML salvage diagnostic) should be opened
under its own pre-reg.

Locked candidate scores (chosen 2026-05-01, no post-hoc swap):
  above_mb_birth:
    +vol_ratio_20_at_event, +bos_distance_atr_at_event,
    +concurrent_quartets, -zone_width_atr, -pivot_confirm_lag_bars
  mit_touch_first / retest_bounce_first:
    +vol_ratio_20_at_event, +bos_distance_atr_at_event,
    +concurrent_quartets, -zone_width_atr, +retest_depth_atr

Outputs:
  output/mb_scanner_phase2_trigger_per_cell.csv
  output/mb_scanner_phase2_trigger_per_cohort.csv
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from mb_scanner.events import EVENT_TYPES
from mb_scanner.phase1 import HORIZONS

OUT_DIR = Path("output")
N_BOOT = 1000
SEED = 42
TOP_DECILE = 0.10
PER_COHORT_FAMILY_ALPHA = 0.05
N_CELLS_PER_COHORT = 25
ALPHA_PER_CELL = PER_COHORT_FAMILY_ALPHA / N_CELLS_PER_COHORT  # 0.002 → 99.8% CI

# (score_col, sign, label) per event_type. Sign +1 means "higher = better",
# -1 means we negate the column so top decile = lowest values.
SCORES_BIRTH = (
    ("vol_ratio_20_at_event",        +1, "vol_ratio_20"),
    ("bos_distance_atr_at_event",    +1, "bos_dist_atr"),
    ("concurrent_quartets",          +1, "concur_quartets"),
    ("zone_width_atr",               -1, "zone_width_inv"),
    ("pivot_confirm_lag_bars",       -1, "pivot_lag_inv"),
)
SCORES_TOUCH = (
    ("vol_ratio_20_at_event",        +1, "vol_ratio_20"),
    ("bos_distance_atr_at_event",    +1, "bos_dist_atr"),
    ("concurrent_quartets",          +1, "concur_quartets"),
    ("zone_width_atr",               -1, "zone_width_inv"),
    ("retest_depth_atr",             +1, "retest_depth"),
)
SCORES_BY_ET = {
    "above_mb_birth":      SCORES_BIRTH,
    "mit_touch_first":     SCORES_TOUCH,
    "retest_bounce_first": SCORES_TOUCH,
}


def _bootstrap_diff_ci(
    rng: np.random.Generator,
    top: np.ndarray,
    full: np.ndarray,
    *,
    n_boot: int,
    alpha: float,
) -> tuple[float, float, float]:
    """Bootstrap CI on (mean(top) - mean(full)). Returns (diff, ci_lo, ci_hi)."""
    diff = top.mean() - full.mean()
    n_top = len(top)
    n_full = len(full)
    # vectorized bootstrap: draw n_boot × n_top resample indices for top,
    # n_boot × n_full for full
    idx_top = rng.integers(0, n_top, size=(n_boot, n_top))
    idx_full = rng.integers(0, n_full, size=(n_boot, n_full))
    bs_top = top[idx_top].mean(axis=1)
    bs_full = full[idx_full].mean(axis=1)
    boots = bs_top - bs_full
    lo = float(np.percentile(boots, 100 * (alpha / 2)))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return float(diff), lo, hi


def _compute_cohort(
    sub: pd.DataFrame,
    family: str,
    event_type: str,
    horizons,
    rng: np.random.Generator,
) -> list[dict]:
    rows: list[dict] = []
    scores = SCORES_BY_ET[event_type]
    for score_col, sign, score_label in scores:
        if score_col not in sub.columns:
            continue
        s = sub[score_col].to_numpy(dtype=float) * sign
        valid_score = ~np.isnan(s)
        if valid_score.sum() < 50:
            continue
        # top decile cutoff on valid values
        cutoff = np.nanpercentile(s, 100 * (1 - TOP_DECILE))
        top_mask = valid_score & (s >= cutoff)

        for h in horizons:
            r_col = f"r_{h}"
            if r_col not in sub.columns:
                continue
            r_arr = sub[r_col].to_numpy(dtype=float)
            full_valid = r_arr[~np.isnan(r_arr)]
            top_valid = r_arr[top_mask & ~np.isnan(r_arr)]
            if top_valid.size < 5 or full_valid.size < 30:
                continue

            mean_full = float(np.mean(full_valid))
            mean_top = float(np.mean(top_valid))
            diff, ci_lo, ci_hi = _bootstrap_diff_ci(
                rng, top_valid, full_valid,
                n_boot=N_BOOT, alpha=ALPHA_PER_CELL,
            )
            ratio = float("nan")
            if abs(mean_full) > 1e-9:
                ratio = mean_top / mean_full
            triggered = bool(ci_lo > 0)
            rows.append({
                "family": family,
                "event_type": event_type,
                "score": score_label,
                "score_col": score_col,
                "score_sign": sign,
                "horizon": int(h),
                "n_cohort": int(full_valid.size),
                "n_topdecile": int(top_valid.size),
                "mean_cohort": mean_full,
                "mean_topdecile": mean_top,
                "diff": diff,
                "lift_ratio": ratio,
                "ci_lo_diff": ci_lo,
                "ci_hi_diff": ci_hi,
                "alpha_per_cell": ALPHA_PER_CELL,
                "triggered": triggered,
            })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args()

    verdicts = pd.read_csv(OUT_DIR / "mb_scanner_phase1_verdicts.csv")
    fails = verdicts[verdicts["verdict"] == "FAIL"][["family", "event_type"]]
    if fails.empty:
        print("[phase2-trigger] no FAIL cohorts; nothing to probe.")
        return 0

    print(f"[phase2-trigger] FAIL cohorts to probe: {len(fails)}")
    print(f"[phase2-trigger] alpha_per_cell={ALPHA_PER_CELL:.4f} (Bonferroni 0.05/25)")
    print(f"[phase2-trigger] n_boot={N_BOOT}  seed={SEED}")
    print()

    rng = np.random.default_rng(SEED)
    t0 = time.time()
    all_cells: list[dict] = []
    for fam, et in fails.itertuples(index=False, name=None):
        path = OUT_DIR / f"mb_scanner_phase1_events_{fam}.parquet"
        df = pd.read_parquet(path)
        sub = df[df["event_type"] == et]
        if sub.empty:
            continue
        rows = _compute_cohort(sub, fam, et, HORIZONS, rng)
        all_cells.extend(rows)
    cells_df = pd.DataFrame(all_cells)
    elapsed = time.time() - t0

    # cohort-level rollup
    cohort_rows = []
    for (fam, et), grp in cells_df.groupby(["family", "event_type"]):
        n_triggered = int(grp["triggered"].sum())
        best = grp.loc[grp["ci_lo_diff"].idxmax()] if not grp.empty else None
        cohort_rows.append({
            "family": fam,
            "event_type": et,
            "cells_evaluated": int(len(grp)),
            "cells_triggered": n_triggered,
            "salvage_trigger": "FIRED" if n_triggered > 0 else "no",
            "best_score": best["score"] if best is not None else "",
            "best_horizon": int(best["horizon"]) if best is not None else 0,
            "best_diff": float(best["diff"]) if best is not None else float("nan"),
            "best_ci_lo": float(best["ci_lo_diff"]) if best is not None else float("nan"),
            "best_lift_ratio": float(best["lift_ratio"]) if best is not None else float("nan"),
            "best_n_top": int(best["n_topdecile"]) if best is not None else 0,
        })
    cohort_df = pd.DataFrame(cohort_rows).sort_values(["salvage_trigger", "family", "event_type"])

    # report
    fired = cohort_df[cohort_df["salvage_trigger"] == "FIRED"]
    closed = cohort_df[cohort_df["salvage_trigger"] == "no"]
    print(f"=== Phase 2 salvage-trigger summary ({len(cohort_df)} FAIL cohorts) ===")
    print(f"  FIRED → Phase 2 candidate: {len(fired)}")
    print(f"  no    → flow to closure  : {len(closed)}")
    print()
    print("{:<6} {:<22} {:>5} {:>5}  {:<12} {:>3}  {:>8} {:>8}  {:>5}  {:<6}".format(
        "family", "event_type", "cells", "fired", "best_score", "h",
        "diff", "ci_lo", "lift", "trig"
    ))
    print("-" * 100)
    for _, r in cohort_df.sort_values(["family", "event_type"]).iterrows():
        print("{:<6} {:<22} {:>5} {:>5}  {:<12} {:>3}  {:>+8.4f} {:>+8.4f}  {:>5.2f}  {:<6}".format(
            r["family"], r["event_type"],
            r["cells_evaluated"], r["cells_triggered"],
            r["best_score"], r["best_horizon"],
            r["best_diff"], r["best_ci_lo"],
            r["best_lift_ratio"] if not np.isnan(r["best_lift_ratio"]) else 0.0,
            r["salvage_trigger"],
        ))
    print()
    print(f"[phase2-trigger] cells={len(cells_df)}  elapsed={elapsed:.1f}s")

    if not args.no_write:
        cells_df.to_csv(OUT_DIR / "mb_scanner_phase2_trigger_per_cell.csv", index=False)
        cohort_df.to_csv(OUT_DIR / "mb_scanner_phase2_trigger_per_cohort.csv", index=False)
        print(f"[phase2-trigger] wrote per_cell ({len(cells_df)}) and per_cohort ({len(cohort_df)}) csv")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
