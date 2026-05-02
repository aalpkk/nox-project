"""Driver for mb_scanner PASS-cohort multi-horizon ranker (LOCKED v1).

Spec: `memory/mb_scanner_pass_cohort_ranker_v1.md` (LOCKED 2026-05-01).

Default = scaffold dry-run (validates inputs / feature build / split coverage,
but does NOT train any model). Use `--go` to fire the single authorized run.

Per-reg: only one `--go` run is authorized. After this fires, the result is
the verdict; gate failure → close per anti-rescue clause.
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

from mb_scanner.ranker import (  # noqa: E402
    ALL_FEATURES,
    HORIZONS,
    NUMERIC_FEATURES,
    PASS_COHORTS,
    RANKER_SPEC_ID,
    build_pool,
    primary_horizon,
    run_ranker,
)

OUT_DIR = Path("output")
SUMMARY_CSV = OUT_DIR / "mb_scanner_ranker_v1_summary.csv"
QUARTER_CSV = OUT_DIR / "mb_scanner_ranker_v1_quarter_stability.csv"
CELL_CSV = OUT_DIR / "mb_scanner_ranker_v1_cell_floor.csv"
PERM_CSV = OUT_DIR / "mb_scanner_ranker_v1_permutation.csv"
TEST_PRED_PARQUET = OUT_DIR / "mb_scanner_ranker_v1_test_preds.parquet"


def _scaffold_check() -> int:
    print(f"[{RANKER_SPEC_ID}] DRY-RUN scaffold check (no training)\n")
    t0 = time.time()
    pool = build_pool()
    fams = pool["family"].value_counts().to_dict()
    print(f"  pool size:           N={len(pool)}")
    print(f"  family counts:       {fams}")
    print(f"  expected PASS cohorts: {[f for f, _ in PASS_COHORTS]}")

    # split coverage
    splits = pool["split"].value_counts().to_dict()
    print(f"  split distribution:  {splits}")

    # NaN feature coverage
    feat_complete = pool[list(NUMERIC_FEATURES)].notna().all(axis=1).sum()
    print(f"  feature-complete:    {feat_complete}/{len(pool)} "
          f"({100*feat_complete/len(pool):.1f}%)")

    # target coverage per horizon
    print("  target hit-rate per horizon (mfe_r_h ≥ 2.0):")
    for h in HORIZONS:
        col = f"quality_{h}"
        valid = pool[col].notna().sum()
        hit = (pool[col] == 1).sum()
        rate = hit / valid if valid else 0.0
        print(f"    h={h:>2}: valid={valid:>6}  hits={hit:>5}  base={rate*100:5.2f}%")

    # RDP regime coverage
    if "regime" in pool.columns:
        rcounts = pool["regime"].value_counts(dropna=False).to_dict()
        print(f"  RDP regime coverage: {rcounts}")

    # cross-TF flags (cast back to int for summing — kept as cat in the model)
    cb1w = pool["concurrent_birth_1w"].astype(int)
    cb1m = pool["concurrent_birth_1M"].astype(int)
    print(f"  concurrent_birth_1w fired: {int(cb1w.sum())} "
          f"({100*cb1w.mean():.1f}%)")
    print(f"  concurrent_birth_1M fired: {int(cb1m.sum())} "
          f"({100*cb1m.mean():.1f}%)")

    # sample weight stats
    w = 1.0 / (1.0 + pool["concurrent_quartets"].clip(lower=0))
    print(f"  sample weight mean   : {w.mean():.3f}  min={w.min():.3f}  max={w.max():.3f}")

    print(f"\n  features locked ({len(ALL_FEATURES)}): {list(ALL_FEATURES)}")
    print(f"\n[scaffold] elapsed {time.time()-t0:.1f}s — pass `--go` to fire single authorized run.")
    return 0


def _print_summary(summary: pd.DataFrame) -> None:
    print()
    print("=" * 100)
    print("HORIZON-MODEL SUMMARY (LOCKED gates: ALL 6 must pass for horizon to PASS)")
    print("=" * 100)
    cols_short = ["horizon", "n_train", "n_test", "base_rate",
                  "fold_auc_min", "fold_auc_max", "pooled_oos_auc",
                  "perm_auc_mean", "test_auc",
                  "top_decile_lift", "top_decile_lift_ci_lo",
                  "cell_floor_min_lift",
                  "gate_G1", "gate_G2", "gate_G3", "gate_G4", "gate_G5", "gate_G6",
                  "passed"]
    print(summary[cols_short].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    n_pass = int(summary["passed"].sum())
    print()
    print(f"Horizons PASSing all 6 gates: {n_pass} / {len(summary)}")
    if n_pass:
        prim = primary_horizon(summary)
        print(f"Primary horizon election (highest min-fold AUC, tie-break shorter): h={prim}")
    else:
        print("THREAD FAIL — anti-rescue clause active. Close per locked spec.")


def _serialize_artifacts(artifacts_by_h: dict) -> None:
    # quarter stability
    qrows = []
    for h, art in artifacts_by_h.items():
        for yq, (n, auc) in art["quarter_aucs"].items():
            qrows.append({"horizon": h, "year_quarter": yq, "n": n, "auc": auc})
    if qrows:
        pd.DataFrame(qrows).to_csv(QUARTER_CSV, index=False)
        print(f"  wrote {QUARTER_CSV} ({len(qrows)} rows)")

    # cell floor
    crows = []
    for h, art in artifacts_by_h.items():
        cdf = art["cell_df"]
        if isinstance(cdf, pd.DataFrame) and not cdf.empty:
            cdf = cdf.copy()
            cdf["horizon"] = h
            crows.append(cdf)
    if crows:
        cell_full = pd.concat(crows, ignore_index=True)
        cell_full.to_csv(CELL_CSV, index=False)
        print(f"  wrote {CELL_CSV} ({len(cell_full)} rows)")

    # permutation seeds
    prows = []
    for h, art in artifacts_by_h.items():
        for s, a in enumerate(art["perm_aucs"]):
            prows.append({"horizon": h, "seed": s, "perm_auc": a})
    if prows:
        pd.DataFrame(prows).to_csv(PERM_CSV, index=False)
        print(f"  wrote {PERM_CSV} ({len(prows)} rows)")

    # test predictions
    test_rows = []
    for h, art in artifacts_by_h.items():
        idx = art["test_idx"]
        preds = art["test_preds"]
        for i, p in zip(idx, preds):
            test_rows.append({"horizon": h, "row_idx": int(i), "pred": float(p)})
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
    print(f"\n[ranker --go] total elapsed {time.time()-t0:.1f}s")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--go", action="store_true",
                    help="Fire single authorized run (default = dry-run scaffold check)")
    args = ap.parse_args()
    if args.go:
        return _go()
    return _scaffold_check()


if __name__ == "__main__":
    raise SystemExit(main())
