"""Salvage error-analysis for v1.1 — NO retraining, NO refit.

Runs diagnostic slicing on v1.1's frozen TEST predictions to identify
whether the failure has a coherent pattern (regime / horizon / cohort /
single-feature dominance) before considering a v1.2 pre-reg.

Per `feedback_salvage_before_close.md`: triggered because lift CI > 1.0×
in 9/10 cells means modest pooled edge exists but gates fail.

Strict invariants (anti-rescue compliant):
  - NO model retraining
  - NO feature/target swap
  - NO threshold tuning post-hoc
  - Pure analysis on existing predictions + features

Diagnostics:
  Q1: per-cohort AUC (does mb_1d alone clear where pooled fails?)
  Q2: per-regime AUC on TEST (regime-specific signal?)
  Q3: single-feature dominance defense — for each cell, compute
      univariate AUC for every numeric feature on TEST; compare to
      model AUC. If model_AUC < best_univariate + 0.03 → univariate
      reproducible (mb_scanner v1's failure mode resurfaces).
  Q4: top-decile concentration audit (tickers / dates / regime mix)
  Q5: quarter stability robustness — exclude N<100 boundary quarters,
      re-check whether mid-period quarters cleanly clear AUC≥0.50.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from mb_scanner.ranker_v1_1 import (  # noqa: E402
    ALL_FEATURES,
    HORIZONS,
    NUMERIC_FEATURES,
    RANKER_SPEC_ID,
    T_SCHEDULES,
    build_pool,
)

OUT_DIR = Path("output")
TEST_PRED = OUT_DIR / "mb_scanner_ranker_v1_1_test_preds.parquet"
SUMMARY = OUT_DIR / "mb_scanner_ranker_v1_1_summary.csv"
QUARTER = OUT_DIR / "mb_scanner_ranker_v1_1_quarter_stability.csv"

REPORT_MD = OUT_DIR / "mb_scanner_ranker_v1_1_salvage_report.md"


def _safe_auc(y, p):
    y = np.asarray(y); p = np.asarray(p)
    m = ~np.isnan(y) & ~np.isnan(p)
    y = y[m]; p = p[m]
    if len(y) < 50 or len(set(y)) < 2:
        return float("nan"), len(y)
    try:
        return float(roc_auc_score(y.astype(int), p)), len(y)
    except ValueError:
        return float("nan"), len(y)


def _q1_per_cohort(pool, preds, schedules, horizons):
    rows = []
    for sched in schedules:
        for h in horizons:
            tcol = f"quality_{sched}_{h}"
            sub = preds[(preds["schedule"] == sched)
                        & (preds["horizon"] == h)]
            if sub.empty:
                continue
            j = sub.merge(
                pool[["family", tcol]].rename(columns={tcol: "y"}),
                left_on="row_idx", right_index=True, how="inner",
            )
            j = j[j["y"].notna()]
            for fam, g in j.groupby("family", observed=True):
                auc, n = _safe_auc(g["y"], g["pred"])
                rows.append({
                    "schedule": sched, "horizon": h, "family": fam,
                    "n": n, "auc": auc,
                    "base": float(g["y"].mean()) if len(g) else float("nan"),
                })
    return pd.DataFrame(rows)


def _q2_per_regime(pool, preds, schedules, horizons):
    rows = []
    for sched in schedules:
        for h in horizons:
            tcol = f"quality_{sched}_{h}"
            sub = preds[(preds["schedule"] == sched)
                        & (preds["horizon"] == h)]
            if sub.empty:
                continue
            j = sub.merge(
                pool[["regime", tcol]].rename(columns={tcol: "y"}),
                left_on="row_idx", right_index=True, how="inner",
            )
            j = j[j["y"].notna()]
            for reg, g in j.groupby("regime", observed=True):
                auc, n = _safe_auc(g["y"], g["pred"])
                rows.append({
                    "schedule": sched, "horizon": h, "regime": reg,
                    "n": n, "auc": auc,
                    "base": float(g["y"].mean()) if len(g) else float("nan"),
                })
    return pd.DataFrame(rows)


def _q3_single_feat_dominance(pool, preds, schedules, horizons,
                              numeric_feats):
    """For each cell, compute univariate AUC of every feature on TEST.

    A feature 'dominates' if abs(univariate_AUC - 0.5) > model edge.
    If model AUC < (best feature AUC + 0.03), model adds nothing (this
    is mb_scanner v1's PASS-DEGENERATE pattern + scanner v1_2's G7).
    """
    rows = []
    summary_rows = []
    for sched in schedules:
        for h in horizons:
            tcol = f"quality_{sched}_{h}"
            sub = preds[(preds["schedule"] == sched)
                        & (preds["horizon"] == h)]
            if sub.empty:
                continue
            cols = ["row_idx", "pred"]
            j = sub[cols].merge(
                pool[list(numeric_feats) + [tcol]],
                left_on="row_idx", right_index=True, how="inner",
            )
            j = j[j[tcol].notna()]
            y = j[tcol].astype(int).to_numpy()
            model_auc, n = _safe_auc(y, j["pred"].to_numpy())
            best_feat = ""
            best_abs_edge = -1.0
            best_signed_auc = float("nan")
            for f in numeric_feats:
                v = j[f].to_numpy()
                if np.isnan(v).all():
                    continue
                m = ~np.isnan(v)
                if m.sum() < 50 or len(set(y[m])) < 2:
                    continue
                try:
                    a = roc_auc_score(y[m], v[m])
                except ValueError:
                    continue
                # try both polarities
                a_pos = a; a_neg = 1 - a
                a_eff = max(a_pos, a_neg)
                edge = abs(a_eff - 0.5)
                rows.append({
                    "schedule": sched, "horizon": h, "feature": f,
                    "univariate_auc_pos": a_pos,
                    "univariate_auc_eff": a_eff,
                })
                if edge > best_abs_edge:
                    best_abs_edge = edge
                    best_feat = f
                    best_signed_auc = a_eff
            margin = (model_auc - best_signed_auc
                      if not (np.isnan(model_auc) or np.isnan(best_signed_auc))
                      else float("nan"))
            summary_rows.append({
                "schedule": sched, "horizon": h, "n": n,
                "model_test_auc": model_auc,
                "best_feature": best_feat,
                "best_feat_auc_eff": best_signed_auc,
                "margin": margin,
                "G7_pass": (margin > 0.03) if not np.isnan(margin) else False,
            })
    return pd.DataFrame(rows), pd.DataFrame(summary_rows)


def _q4_top_decile_concentration(pool, preds, schedules, horizons,
                                 top_q=0.10):
    rows = []
    for sched in schedules:
        for h in horizons:
            tcol = f"quality_{sched}_{h}"
            sub = preds[(preds["schedule"] == sched)
                        & (preds["horizon"] == h)]
            if sub.empty:
                continue
            j = sub.merge(
                pool[["ticker", "event_bar_date", "regime", "family", tcol]
                     ].rename(columns={tcol: "y"}),
                left_on="row_idx", right_index=True, how="inner",
            )
            j = j[j["y"].notna()]
            if j.empty:
                continue
            cutoff = j["pred"].quantile(1 - top_q)
            top = j[j["pred"] >= cutoff]
            if top.empty:
                continue
            n_top = len(top)
            n_tickers = top["ticker"].nunique()
            n_dates = top["event_bar_date"].nunique()
            base_hr = j["y"].mean()
            top_hr = top["y"].mean()
            lift = top_hr / base_hr if base_hr > 0 else float("nan")
            top_reg = top["regime"].value_counts(normalize=True).to_dict()
            top_fam = top["family"].value_counts(normalize=True).to_dict()
            rows.append({
                "schedule": sched, "horizon": h, "n_top": n_top,
                "n_unique_tickers": n_tickers,
                "n_unique_dates": n_dates,
                "ticker_per_event": round(n_top / max(1, n_tickers), 2),
                "base_hit": base_hr, "top_hit": top_hr, "lift": lift,
                "top_regime_long_pct": round(
                    100 * top_reg.get("long", 0.0), 1
                ),
                "top_regime_neutral_pct": round(
                    100 * top_reg.get("neutral", 0.0), 1
                ),
                "top_regime_short_pct": round(
                    100 * top_reg.get("short", 0.0), 1
                ),
                "top_mb_1d_pct": round(100 * top_fam.get("mb_1d", 0.0), 1),
                "top_mb_1w_pct": round(100 * top_fam.get("mb_1w", 0.0), 1),
            })
    return pd.DataFrame(rows)


def _q5_quarter_stability_robust(quarter_df, min_n=100):
    """Re-check Q-stab gate excluding N<100 boundary quarters."""
    sub = quarter_df[quarter_df["n"] >= min_n].copy()
    rows = []
    for (sched, h), g in sub.groupby(["schedule", "horizon"]):
        rows.append({
            "schedule": sched, "horizon": h,
            "n_quarters_kept": len(g),
            "min_auc": g["auc"].min(),
            "max_auc": g["auc"].max(),
            "mean_auc": g["auc"].mean(),
            "G2_pass_robust": bool(g["auc"].min() >= 0.50),
        })
    return pd.DataFrame(rows)


def main():
    print(f"[{RANKER_SPEC_ID}] SALVAGE error-analysis (anti-rescue compliant)")
    print(f"  load TEST preds: {TEST_PRED}")
    preds = pd.read_parquet(TEST_PRED)
    print(f"  test_preds rows: {len(preds)}")

    print("  rebuild pool ... (band + features + targets)")
    pool = build_pool()
    print(f"  pool size: {len(pool)}")

    schedules = list(T_SCHEDULES.keys())
    horizons = list(HORIZONS)

    print("\n  Q1 — per-cohort AUC on TEST")
    q1 = _q1_per_cohort(pool, preds, schedules, horizons)
    print(q1.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n  Q2 — per-regime AUC on TEST")
    q2 = _q2_per_regime(pool, preds, schedules, horizons)
    print(q2.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n  Q3 — single-feature dominance defense (mb_scanner v1 trap repeat?)")
    _, q3_sum = _q3_single_feat_dominance(
        pool, preds, schedules, horizons, NUMERIC_FEATURES,
    )
    print(q3_sum.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    n_g7_fail = int((q3_sum["G7_pass"] == False).sum())
    print(f"  G7 (model > best_feat + 0.03) FAILS in {n_g7_fail}/{len(q3_sum)} cells")

    print("\n  Q4 — top-decile concentration audit on TEST")
    q4 = _q4_top_decile_concentration(pool, preds, schedules, horizons)
    print(q4.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n  Q5 — quarter stability robust to small N (drop N<100)")
    qstab = pd.read_csv(QUARTER)
    q5 = _q5_quarter_stability_robust(qstab, min_n=100)
    print(q5.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    n_g2_robust = int(q5["G2_pass_robust"].sum())
    print(f"  G2 robust (min N=100, AUC≥0.50) PASSES in {n_g2_robust}/{len(q5)} cells")

    # write report
    with open(REPORT_MD, "w") as f:
        f.write("# mb_scanner ranker v1.1 — Salvage error-analysis\n\n")
        f.write("Anti-rescue compliant: NO retraining, pure analysis on frozen artifacts.\n\n")
        f.write("## Q1 — per-cohort AUC on TEST\n\n")
        f.write(q1.to_markdown(index=False, floatfmt=".4f"))
        f.write("\n\n## Q2 — per-regime AUC on TEST\n\n")
        f.write(q2.to_markdown(index=False, floatfmt=".4f"))
        f.write("\n\n## Q3 — single-feature dominance defense\n\n")
        f.write("(`G7_pass = model_AUC > best_univariate_AUC + 0.03`)\n\n")
        f.write(q3_sum.to_markdown(index=False, floatfmt=".4f"))
        f.write(f"\n\n**G7 FAILS in {n_g7_fail}/{len(q3_sum)} cells.**\n\n")
        f.write("## Q4 — top-decile concentration on TEST\n\n")
        f.write(q4.to_markdown(index=False, floatfmt=".4f"))
        f.write("\n\n## Q5 — quarter stability robust to small N\n\n")
        f.write(q5.to_markdown(index=False, floatfmt=".4f"))
        f.write(f"\n\n**G2 robust PASSES in {n_g2_robust}/{len(q5)} cells.**\n")
    print(f"\n  wrote {REPORT_MD}")


if __name__ == "__main__":
    raise SystemExit(main())
