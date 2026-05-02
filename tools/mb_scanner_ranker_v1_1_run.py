"""Driver for mb_scanner PASS-cohort multi-horizon ranker (LOCKED v1.1).

Spec: `memory/mb_scanner_pass_cohort_ranker_v1_1.md` (LOCKED 2026-05-02).

Successor to v1 (PASS-DEGENERATE). v1.1 differences vs v1:
  - per-cohort tradeable risk-band universe filter
    (mb_1d [2%, 10%], mb_1w [8%, 25%]; mb_1M / bb_1M dropped)
  - dual T_h schedule (aggressive + moderate) trained in parallel
  - 5 horizons × 2 schedules = 10 horizon-models
  - target = absolute mfe_pct_h ≥ T_h (NOT mfe_r — denominator removed)
  - 4 risk-proxy features dropped (r_distance_pct, atr_pct_at_event,
    zone_width_atr, zone_width_pct)

Default = scaffold dry-run (validates band filter, feature build, target
coverage, base rates per (schedule, horizon) cell, but does NOT train any
model). Use `--go` to fire the single authorized run.

Anti-rescue: any 0/10 outcome → close, no v1.1 tweak. v1.2 (Path C) opens
ONLY by explicit user authorization as a separate pre-reg.
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

from mb_scanner.ranker_v1_1 import (  # noqa: E402
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    DROPPED_RISK_PROXY_FEATURES,
    HORIZONS,
    NUMERIC_FEATURES,
    PASS_COHORTS,
    RANKER_SPEC_ID,
    T_SCHEDULES,
    TRADEABLE_BANDS,
    _attach_cross_tf_flags,
    _attach_rdp_regime,
    _build_targets_absolute,
    _derive_features,
    _load_pool,
    build_pool,
    run_ranker,
)

OUT_DIR = Path("output")
SUMMARY_CSV = OUT_DIR / "mb_scanner_ranker_v1_1_summary.csv"
QUARTER_CSV = OUT_DIR / "mb_scanner_ranker_v1_1_quarter_stability.csv"
CELL_CSV = OUT_DIR / "mb_scanner_ranker_v1_1_cell_floor.csv"
PERM_CSV = OUT_DIR / "mb_scanner_ranker_v1_1_permutation.csv"
TEST_PRED_PARQUET = OUT_DIR / "mb_scanner_ranker_v1_1_test_preds.parquet"


def _scaffold_check() -> int:
    print(f"[{RANKER_SPEC_ID}] DRY-RUN scaffold check (no training)\n")
    t0 = time.time()

    # --- pre-band pool (only feature build, no band filter) ---
    pre = _load_pool()
    pre = _derive_features(pre)
    pre = _attach_rdp_regime(pre)
    pre = _attach_cross_tf_flags(pre)
    pre_counts = pre["family"].value_counts().to_dict()
    print("  PRE-BAND pool")
    print(f"    total N:           {len(pre)}")
    for fam, _ in PASS_COHORTS:
        n = pre_counts.get(fam, 0)
        if "r_distance_pct" in pre.columns:
            rd = pre.loc[pre["family"] == fam, "r_distance_pct"]
            if len(rd):
                p25, p50, p75 = np.nanpercentile(rd, [25, 50, 75])
                print(f"    {fam:<8} N={n}  r_dist_pct p25={p25:.3f} p50={p50:.3f} p75={p75:.3f}")
            else:
                print(f"    {fam:<8} N={n}")
        else:
            print(f"    {fam:<8} N={n}")

    # --- band filter retention ---
    print("\n  TRADEABLE BAND filter (LOCKED v1.1)")
    for fam, (lo, hi) in TRADEABLE_BANDS.items():
        sub = pre[pre["family"] == fam]
        n_pre = len(sub)
        if n_pre == 0:
            print(f"    {fam:<8} band [{lo:.2%}, {hi:.2%}] : pool empty")
            continue
        m = (sub["r_distance_pct"] >= lo) & (sub["r_distance_pct"] <= hi)
        n_post = int(m.sum())
        too_tight = int((sub["r_distance_pct"] < lo).sum())
        too_wide = int((sub["r_distance_pct"] > hi).sum())
        print(f"    {fam:<8} band [{lo:.2%}, {hi:.2%}] : "
              f"{n_pre} → {n_post}  ({n_post / n_pre * 100:.1f}% kept) "
              f"[tight<lo: {too_tight}, wide>hi: {too_wide}]")

    # --- post-band pool (full pipeline) ---
    pool = build_pool()
    fams = pool["family"].value_counts().to_dict()
    print("\n  POST-BAND pool")
    print(f"    total N:           {len(pool)}")
    print(f"    family counts:     {fams}")

    splits = pool["split"].value_counts().to_dict()
    print(f"    split distribution: {splits}")

    feat_complete = pool[list(NUMERIC_FEATURES)].notna().all(axis=1).sum()
    print(f"    feature-complete:  {feat_complete}/{len(pool)} "
          f"({100 * feat_complete / max(1, len(pool)):.1f}%)")

    # --- feature dropout audit ---
    print("\n  FEATURE AUDIT")
    print(f"    in-model features ({len(ALL_FEATURES)}): {list(ALL_FEATURES)}")
    print(f"    DROPPED risk-proxies ({len(DROPPED_RISK_PROXY_FEATURES)}): "
          f"{list(DROPPED_RISK_PROXY_FEATURES)}")
    leak = [c for c in DROPPED_RISK_PROXY_FEATURES if c in ALL_FEATURES]
    if leak:
        print(f"    !! leak: dropped feature in active set: {leak}")
        return 2
    print("    drop check OK — none of the 4 risk-proxies are in-model")

    # --- targets (absolute mfe_pct_h ≥ T_h) per (schedule, horizon) cell ---
    print("\n  TARGET COVERAGE per (schedule, horizon) cell")
    print(f"    {'schedule':<11} {'h':>3} {'T_h':>6} "
          f"{'valid':>7} {'hits':>6} {'base_rate':>10}")
    for sched, T in T_SCHEDULES.items():
        for h in HORIZONS:
            col = f"quality_{sched}_{h}"
            if col not in pool.columns:
                print(f"    {sched:<11} {h:>3}   ??   missing target column")
                continue
            valid = int(pool[col].notna().sum())
            hits = int((pool[col] == 1).sum())
            rate = hits / valid if valid else 0.0
            print(f"    {sched:<11} {h:>3} {T[h]:>6.2%} "
                  f"{valid:>7} {hits:>6} {rate * 100:>9.2f}%")

    # --- per-cohort base-rate sanity (mb_1d vs mb_1w) ---
    print("\n  BASE RATE per cohort (h=5, moderate as reference)")
    for fam in [f for f, _ in PASS_COHORTS]:
        sub = pool[pool["family"] == fam]
        col = f"quality_moderate_5"
        if col in sub.columns and len(sub):
            valid = int(sub[col].notna().sum())
            hits = int((sub[col] == 1).sum())
            rate = hits / valid if valid else 0.0
            print(f"    {fam:<8} N={len(sub):>5} valid={valid:>5} "
                  f"hits={hits:>4} base={rate * 100:5.2f}%")

    # --- regime / cross-TF coverage (for cell-floor gate) ---
    if "regime" in pool.columns:
        rcounts = pool["regime"].value_counts(dropna=False).to_dict()
        print(f"\n  RDP regime coverage: {rcounts}")
    cb1w = pool["concurrent_birth_1w"].astype(int)
    cb1m = pool["concurrent_birth_1M"].astype(int)
    print(f"  concurrent_birth_1w fired: {int(cb1w.sum())} "
          f"({100 * cb1w.mean():.1f}%)")
    print(f"  concurrent_birth_1M fired: {int(cb1m.sum())} "
          f"({100 * cb1m.mean():.1f}%)")

    # sample weight stats
    w = 1.0 / (1.0 + pool["concurrent_quartets"].clip(lower=0))
    print(f"  sample weight mean : {w.mean():.3f}  "
          f"min={w.min():.3f} max={w.max():.3f}")

    print(f"\n[scaffold] elapsed {time.time() - t0:.1f}s — "
          f"pass `--go` to fire single authorized run.")
    return 0


def _print_summary(summary: pd.DataFrame) -> None:
    print()
    print("=" * 110)
    print("CELL SUMMARY (LOCKED 6 gates: ALL must pass for cell to PASS)  "
          "[10 cells = 2 schedules × 5 horizons]")
    print("=" * 110)
    cols_short = [
        "schedule", "horizon", "n_train", "n_test", "base_rate",
        "fold_auc_min", "fold_auc_max", "pooled_oos_auc",
        "perm_auc_mean", "test_auc",
        "top_decile_lift", "top_decile_lift_ci_lo",
        "cell_floor_min_lift",
        "gate_G1", "gate_G2", "gate_G3", "gate_G4", "gate_G5", "gate_G6",
        "passed",
    ]
    fmt = {c: lambda x: f"{x:.4f}" for c in cols_short
           if c not in ("schedule", "horizon", "n_train", "n_test",
                        "gate_G1", "gate_G2", "gate_G3", "gate_G4",
                        "gate_G5", "gate_G6", "passed")}
    print(summary[cols_short].to_string(index=False,
                                        float_format=lambda x: f"{x:.4f}"))
    n_pass = int(summary["passed"].sum())
    print()
    print(f"Cells PASSing all 6 gates: {n_pass} / {len(summary)}")
    if n_pass:
        passing = summary[summary["passed"]].copy()
        print("\nPASS cells:")
        for _, r in passing.iterrows():
            print(f"  schedule={r['schedule']:<11} h={int(r['horizon']):>2}  "
                  f"pooled_AUC={r['pooled_oos_auc']:.3f}  "
                  f"top10_lift={r['top_decile_lift']:.2f}× "
                  f"CI[{r['top_decile_lift_ci_lo']:.2f}, "
                  f"{r['top_decile_lift_ci_hi']:.2f}]")
        print("\nPromotion rule: thread PASSes iff at least one cell PASS AND "
              "not a single-cell anomaly (coherent pattern check).")
    else:
        print("THREAD FAIL — anti-rescue clause active. v1.2 (Path C) requires "
              "explicit user authorization as separate new pre-reg.")


def _serialize_artifacts(artifacts_by_cell: dict) -> None:
    # quarter stability
    qrows = []
    for (sched, h), art in artifacts_by_cell.items():
        for yq, (n, auc) in art["quarter_aucs"].items():
            qrows.append({"schedule": sched, "horizon": h,
                          "year_quarter": yq, "n": n, "auc": auc})
    if qrows:
        pd.DataFrame(qrows).to_csv(QUARTER_CSV, index=False)
        print(f"  wrote {QUARTER_CSV} ({len(qrows)} rows)")

    # cell floor (regime × concurrency cells)
    crows = []
    for (sched, h), art in artifacts_by_cell.items():
        cdf = art["cell_df"]
        if isinstance(cdf, pd.DataFrame) and not cdf.empty:
            cdf = cdf.copy()
            cdf["schedule"] = sched
            cdf["horizon"] = h
            crows.append(cdf)
    if crows:
        cell_full = pd.concat(crows, ignore_index=True)
        cell_full.to_csv(CELL_CSV, index=False)
        print(f"  wrote {CELL_CSV} ({len(cell_full)} rows)")

    # permutation seeds
    prows = []
    for (sched, h), art in artifacts_by_cell.items():
        for s, a in enumerate(art["perm_aucs"]):
            prows.append({"schedule": sched, "horizon": h,
                          "seed": s, "perm_auc": a})
    if prows:
        pd.DataFrame(prows).to_csv(PERM_CSV, index=False)
        print(f"  wrote {PERM_CSV} ({len(prows)} rows)")

    # test predictions
    test_rows = []
    for (sched, h), art in artifacts_by_cell.items():
        idx = art["test_idx"]
        preds = art["test_preds"]
        for i, p in zip(idx, preds):
            test_rows.append({"schedule": sched, "horizon": h,
                              "row_idx": int(i), "pred": float(p)})
    if test_rows:
        pd.DataFrame(test_rows).to_parquet(TEST_PRED_PARQUET, index=False)
        print(f"  wrote {TEST_PRED_PARQUET} ({len(test_rows)} rows)")


def _go() -> int:
    print(f"[{RANKER_SPEC_ID}] FIRING single authorized run\n")
    t0 = time.time()
    summary, artifacts = run_ranker(verbose=True)
    summary.to_csv(SUMMARY_CSV, index=False)
    _print_summary(summary)
    print(f"\n  wrote {SUMMARY_CSV}")
    _serialize_artifacts(artifacts)
    print(f"\n[ranker --go] total elapsed {time.time() - t0:.1f}s")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--go", action="store_true",
                    help="Fire single authorized run "
                         "(default = dry-run scaffold check)")
    args = ap.parse_args()
    if args.go:
        return _go()
    return _scaffold_check()


if __name__ == "__main__":
    raise SystemExit(main())
