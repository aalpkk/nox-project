"""
Step 4 runner — cross-sectional ranking models framed as "does this fix the
D10 reversal and shallow ordering that showed up in Step 3?"

NOT a top-20 CAGR race. Primary objective (locked by user, 2026-04-21):

  1. OOS decile monotonicity vs THREE realized targets:
       • l2_excess_vs_universe_median   (train target)
       • l1_forward_return              (raw return)
       • forward_max_dd_intraperiod     (downside — should DECREASE with rank)
  2. D10-D1 long-short Sharpe as ranking-quality scalar
  3. Net-of-cost at 0/20/40/60/100 bps round-trip
  4. Turnover + concentration only as tiebreakers

Models (uniform walk-forward interface):
  M0   handcrafted composite (no fit, recent_extreme_21d dropped)
  M1   ridge on per-date z-scored features
  M2   LightGBM regressor on L2, val-based early stopping

Ablations (M2 only):
  trend_quality_only          trend_r2_126d, trend_above_ma200_pct_126d,
                              px_over_ma50
  momentum_rs_only            mom_* + rs_xu100_*
  all_core                    FEATURE_COLUMNS (full)
  all_core_minus_liquidity    all_core minus log_tl_turnover/log_amihud

Feature governance manifest emitted as CSV — dual_use=True is flagged loudly.

Success bar (from user directive #9):
  OOS D10-D1 Sharpe (on L2) > 1.22  AND  monotonic decile stacking
    AND net-of-cost 60bps CAGR >= 87.7% (handcrafted_no_overlay Step 3 baseline)
  OR similar result with materially lower turnover / drawdown.

Artifacts (output/nyxmomentum/reports/):
  step4_predictions_<model>.parquet
  step4_deciles_<model>_<target>.csv
  step4_d9_d10_<model>.csv
  step4_returns_<model>.csv
  step4_cost_<model>.csv
  step4_ablation_summary.csv
  step4_feature_governance.csv
  step4_folds.csv
  step4_run_meta.json
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

from nyxmomentum.config import CONFIG, EXECUTION_COLUMN_ROLES, LABEL_COLUMN_ROLES
from nyxmomentum.features import FEATURE_COLUMNS
from nyxmomentum.walk_forward import usable_folds, summarize_folds
from nyxmomentum.models import (
    run_m0, run_m1, run_m2,
    feature_governance_manifest,
)
from nyxmomentum.evaluation import (
    model_summary,
    decile_table,
    monotonicity_score,
    d9_vs_d10_feature_delta,
    top_n_portfolio_returns,
    portfolio_cost_table,
)
from nyxmomentum.utils import ensure_dir, save_json


DEFAULT_FEATURES = "output/nyxmomentum/reports/step2_features.parquet"
DEFAULT_LABELS   = "output/nyxmomentum/reports/step1_labels.parquet"
DEFAULT_PANEL    = "output/nyxmomentum/reports/step0_universe_panel_extended.parquet"

TARGET_COLS = (
    "l2_excess_vs_universe_median",   # train target
    "l1_forward_return",              # raw return
    "forward_max_dd_intraperiod",     # downside — should DECREASE with rank
)

# Step 3 audit handcrafted_no_overlay after dropping recent_extreme_21d:
#   CAGR 87.7%, D10-D1 Sharpe 1.22
SUCCESS_BAR_CAGR = 0.877
SUCCESS_BAR_D10D1_SHARPE = 1.22

ABLATIONS: dict[str, tuple[str, ...]] = {
    "trend_quality_only": (
        "trend_r2_126d", "trend_above_ma200_pct_126d", "px_over_ma50",
    ),
    "momentum_rs_only": (
        "mom_21d", "mom_63d", "mom_126d",
        "mom_252d_skip_21d", "mom_63d_skip_21d",
        "rs_xu100_63d", "rs_xu100_126d", "rs_xu100_252d_skip_21d",
    ),
    "all_core": tuple(FEATURE_COLUMNS),
    "all_core_minus_liquidity": tuple(
        c for c in FEATURE_COLUMNS
        if c not in {"log_tl_turnover_20d", "log_tl_turnover_60d", "log_amihud_20d"}
    ),
}


# ── Panel assembly ───────────────────────────────────────────────────────────

def build_model_panel(features: pd.DataFrame,
                      labels: pd.DataFrame) -> pd.DataFrame:
    """Join features ← labels[l2_excess_vs_universe_median] inner; keep
    eligibility column from features. Rows without label are dropped so
    models only train where we have a target."""
    keep_label = ["ticker", "rebalance_date", "l2_excess_vs_universe_median"]
    return features.merge(labels[keep_label], on=["ticker", "rebalance_date"], how="inner")


# ── Single-model driver ──────────────────────────────────────────────────────

def _emit_model_reports(model_name: str,
                        preds: pd.DataFrame,
                        labels: pd.DataFrame,
                        features: pd.DataFrame,
                        reports_dir: str,
                        top_n: int,
                        n_buckets: int,
                        file_tag: str | None = None) -> dict:
    """Call evaluation.model_summary and write all artifacts to disk.
    Returns the summary dict plus headline metrics for console."""
    tag = file_tag or model_name
    preds_path = os.path.join(reports_dir, f"step4_predictions_{tag}.parquet")
    # DataFrame.attrs (holds LightGBM importances) is JSON-serialized into
    # parquet metadata — drop it to avoid TypeError on DataFrame attr values.
    preds_to_write = preds.copy()
    preds_to_write.attrs.clear()
    preds_to_write.to_parquet(preds_path, index=False)

    feat_cols = [c for c in FEATURE_COLUMNS if c in features.columns]
    summary = model_summary(
        model_name=model_name,
        preds=preds,
        labels=labels,
        features=features[["ticker", "rebalance_date", *feat_cols]],
        target_cols=TARGET_COLS,
        top_n=top_n,
        n_buckets=n_buckets,
    )

    for tgt, d in summary["deciles"].items():
        if d is None or d.empty:
            continue
        d.to_csv(
            os.path.join(reports_dir, f"step4_deciles_{tag}_{tgt}.csv"),
            index=False,
        )
    if not summary["d9_vs_d10"].empty:
        summary["d9_vs_d10"].to_csv(
            os.path.join(reports_dir, f"step4_d9_d10_{tag}.csv"), index=False,
        )
    summary["returns"].to_csv(
        os.path.join(reports_dir, f"step4_returns_{tag}.csv"), index=False,
    )
    summary["cost_table"].to_csv(
        os.path.join(reports_dir, f"step4_cost_{tag}.csv"), index=False,
    )

    # Headline for console + ablation summary
    l2_d = summary["deciles"].get(TARGET_COLS[0], pd.DataFrame())
    l2_ls = l2_d.loc[l2_d["_decile"] == -1] if not l2_d.empty else pd.DataFrame()
    d10_d1_sharpe = float(l2_ls["sharpe_annualized"].iloc[0]) if len(l2_ls) else np.nan

    raw_d = summary["deciles"].get(TARGET_COLS[1], pd.DataFrame())
    raw_ls = raw_d.loc[raw_d["_decile"] == -1] if not raw_d.empty else pd.DataFrame()
    d10_d1_raw_sharpe = float(raw_ls["sharpe_annualized"].iloc[0]) if len(raw_ls) else np.nan

    mono = summary["monotonicity"].get(TARGET_COLS[0], {}) or {}
    mono_raw = summary["monotonicity"].get(TARGET_COLS[1], {}) or {}
    mono_dd  = summary["monotonicity"].get(TARGET_COLS[2], {}) or {}

    cost = summary["cost_table"]
    cagr_net_60 = float(
        cost.loc[cost["bps_round_trip"] == 60, "cagr_net"].iloc[0]
    ) if (cost is not None and not cost.empty and
          (cost["bps_round_trip"] == 60).any()) else np.nan
    cagr_net_0 = float(
        cost.loc[cost["bps_round_trip"] == 0, "cagr_net"].iloc[0]
    ) if (cost is not None and not cost.empty and
          (cost["bps_round_trip"] == 0).any()) else np.nan
    sharpe_net_60 = float(
        cost.loc[cost["bps_round_trip"] == 60, "sharpe_annualized_net"].iloc[0]
    ) if (cost is not None and not cost.empty and
          (cost["bps_round_trip"] == 60).any()) else np.nan
    dd_net_60 = float(
        cost.loc[cost["bps_round_trip"] == 60, "max_drawdown_net"].iloc[0]
    ) if (cost is not None and not cost.empty and
          (cost["bps_round_trip"] == 60).any()) else np.nan
    avg_turnover = float(summary["returns"]["turnover_fraction"].dropna().mean()) \
        if not summary["returns"].empty else np.nan

    headline = {
        "model": model_name,
        "tag": tag,
        "d10_d1_sharpe_L2": d10_d1_sharpe,
        "d10_d1_sharpe_raw": d10_d1_raw_sharpe,
        "spearman_monotonicity_L2":   mono.get("spearman_monotonicity"),
        "spearman_monotonicity_raw":  mono_raw.get("spearman_monotonicity"),
        "spearman_monotonicity_DD":   mono_dd.get("spearman_monotonicity"),
        "n_inversions_L2":  mono.get("n_inversions"),
        "d10_minus_d9_L2":  mono.get("d10_minus_d9"),
        "cagr_net_0bps":    cagr_net_0,
        "cagr_net_60bps":   cagr_net_60,
        "sharpe_net_60bps": sharpe_net_60,
        "max_drawdown_net_60bps": dd_net_60,
        "avg_turnover":     avg_turnover,
        "n_rebalances": int(summary["returns"].shape[0]) if not summary["returns"].empty else 0,
    }
    return {"summary": summary, "headline": headline}


def _pass_fail(headline: dict) -> dict:
    sharpe = headline.get("d10_d1_sharpe_L2")
    cagr = headline.get("cagr_net_60bps")
    mono = headline.get("spearman_monotonicity_L2")
    pass_sharpe = bool(sharpe is not None and np.isfinite(sharpe) and sharpe > SUCCESS_BAR_D10D1_SHARPE)
    pass_cagr = bool(cagr is not None and np.isfinite(cagr) and cagr >= SUCCESS_BAR_CAGR)
    pass_mono = bool(mono is not None and np.isfinite(mono) and mono > 0.7)
    return {
        "pass_d10_d1_sharpe_bar": pass_sharpe,
        "pass_cagr_60bps_bar":    pass_cagr,
        "pass_monotonicity_L2":   pass_mono,
        "pass_all": pass_sharpe and pass_cagr and pass_mono,
    }


# ── Run ──────────────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    t0 = time.time()
    reports_dir = ensure_dir(CONFIG.paths.reports)

    print("[1/7] Loading features / labels / extended panel …")
    features = pd.read_parquet(args.features)
    labels   = pd.read_parquet(args.labels)
    panel_ex = pd.read_parquet(args.panel)
    print(f"  features={len(features):,}  labels={len(labels):,}  "
          f"panel_ex={len(panel_ex):,}")

    print("[2/7] Assembling model panel (features ⋈ labels) …")
    model_panel = build_model_panel(features, labels)
    print(f"  model_panel rows={len(model_panel):,} "
          f"(eligible={int(model_panel['eligible'].astype(bool).sum()):,})")

    print("[3/7] Building walk-forward folds …")
    folds = usable_folds(model_panel["rebalance_date"])
    if not folds:
        raise RuntimeError("No usable folds — check CONFIG.split.folds vs data coverage.")
    fold_summary = summarize_folds(folds, model_panel["rebalance_date"])
    fold_summary.to_csv(os.path.join(reports_dir, "step4_folds.csv"), index=False)
    print(f"  {len(folds)} usable fold(s): " +
          ", ".join(f"{f.fold_id}({f.test_start.date()}→{f.test_end.date()})" for f in folds))

    # ── Models ────────────────────────────────────────────────────────────
    print("[4/7] Fitting models + generating OOS predictions …")

    print("  M0 handcrafted composite …")
    preds_m0 = run_m0(model_panel, folds)

    print("  M1 ridge on z-scored features …")
    preds_m1 = run_m1(model_panel, folds, features=FEATURE_COLUMNS,
                      target_col="l2_excess_vs_universe_median")

    print("  M2 LightGBM on L2 (all_core) …")
    preds_m2 = run_m2(model_panel, folds, features=FEATURE_COLUMNS,
                      target_col="l2_excess_vs_universe_median")

    # ── Model reports ─────────────────────────────────────────────────────
    print("[5/7] Emitting per-model reports …")
    per_model_headlines: dict[str, dict] = {}
    for name, preds in [("M0", preds_m0), ("M1", preds_m1), ("M2", preds_m2)]:
        if preds is None or preds.empty:
            print(f"  {name}: NO predictions — skipping reports")
            continue
        out = _emit_model_reports(
            model_name=name, preds=preds, labels=labels, features=features,
            reports_dir=reports_dir, top_n=args.top_n, n_buckets=args.n_buckets,
            file_tag=name,
        )
        per_model_headlines[name] = out["headline"]

    # Save LightGBM importances if available
    imp = preds_m2.attrs.get("importances", pd.DataFrame()) if isinstance(preds_m2, pd.DataFrame) else pd.DataFrame()
    if isinstance(imp, pd.DataFrame) and not imp.empty:
        imp.to_csv(os.path.join(reports_dir, "step4_importances_M2.csv"), index=False)

    # ── Ablations (M2 only) ───────────────────────────────────────────────
    print("[6/7] Ablations (M2) …")
    ablation_rows: list[dict] = []
    for ab_name, feat_tuple in ABLATIONS.items():
        feat_list = [c for c in feat_tuple if c in FEATURE_COLUMNS]
        if not feat_list:
            print(f"  {ab_name}: empty feature list — skipping")
            continue
        tag = f"M2__{ab_name}"
        print(f"  {ab_name} ({len(feat_list)} features) …")
        preds_ab = run_m2(model_panel, folds, features=feat_list,
                          target_col="l2_excess_vs_universe_median")
        if preds_ab is None or preds_ab.empty:
            continue
        out = _emit_model_reports(
            model_name=tag, preds=preds_ab, labels=labels, features=features,
            reports_dir=reports_dir, top_n=args.top_n, n_buckets=args.n_buckets,
            file_tag=tag,
        )
        row = dict(out["headline"])
        row["ablation"] = ab_name
        row["n_features"] = len(feat_list)
        ablation_rows.append(row)

    if ablation_rows:
        pd.DataFrame(ablation_rows).to_csv(
            os.path.join(reports_dir, "step4_ablation_summary.csv"), index=False,
        )

    # ── Feature governance manifest ───────────────────────────────────────
    print("[7/7] Feature governance manifest …")
    overlay_inputs = [
        c for c, role in EXECUTION_COLUMN_ROLES.items() if role == "ex_ante"
    ]
    diagnostic_cols = [
        c for c, role in EXECUTION_COLUMN_ROLES.items() if role == "diagnostic_only"
    ] + [
        c for c, role in LABEL_COLUMN_ROLES.items() if role == "diagnostic"
    ]
    gov = feature_governance_manifest(
        model_features={
            "M0": tuple(),                              # composite, reaches features directly
            "M1": tuple(FEATURE_COLUMNS),
            "M2": tuple(FEATURE_COLUMNS),
        },
        overlay_inputs=overlay_inputs,
        diagnostic_columns=diagnostic_cols,
    )
    gov.to_csv(os.path.join(reports_dir, "step4_feature_governance.csv"), index=False)
    dual_use = gov.loc[gov["dual_use_model_and_overlay"]]
    if not dual_use.empty:
        print(f"  WARNING: {len(dual_use)} dual-use columns flagged:")
        for _, r in dual_use.iterrows():
            print(f"    {r['column']}")

    # ── Meta ──────────────────────────────────────────────────────────────
    meta = {
        "produced_at":        pd.Timestamp.utcnow().isoformat(),
        "n_folds_total":      len(CONFIG.split.folds),
        "n_folds_usable":     len(folds),
        "usable_fold_ids":    [f.fold_id for f in folds],
        "target_cols":        list(TARGET_COLS),
        "train_target":       "l2_excess_vs_universe_median",
        "top_n":              args.top_n,
        "n_buckets":          args.n_buckets,
        "success_bar": {
            "d10_d1_sharpe_min": SUCCESS_BAR_D10D1_SHARPE,
            "cagr_net_60bps_min": SUCCESS_BAR_CAGR,
        },
        "models":            list(per_model_headlines.keys()),
        "ablations":         list(ABLATIONS.keys()),
        "headlines":         per_model_headlines,
        "elapsed_sec":       time.time() - t0,
    }
    save_json(os.path.join(reports_dir, "step4_run_meta.json"), meta)

    # ── Console summary ───────────────────────────────────────────────────
    _print_summary(per_model_headlines, ablation_rows, reports_dir, t0)


def _fmt_pct(v) -> str:
    return f"{v:+.1%}" if (v is not None and pd.notna(v) and np.isfinite(v)) else "   —  "


def _fmt_num(v, w: int = 6) -> str:
    if v is None or pd.isna(v) or not np.isfinite(v):
        return "  —  ".rjust(w)
    return f"{v:+.2f}".rjust(w)


def _print_summary(per_model: dict, ablations: list[dict],
                   reports_dir: str, t0: float) -> None:
    print()
    print("══ nyxmomentum Step 4 ══")
    print()
    print("  SUCCESS BAR:  D10-D1 Sharpe (L2) > "
          f"{SUCCESS_BAR_D10D1_SHARPE:.2f}  AND  "
          f"CAGR_net@60bps ≥ {SUCCESS_BAR_CAGR:.1%}  AND  monotonicity > 0.70")
    print()
    print("  PER-MODEL HEADLINES:")
    print(f"  {'model':<8}{'D10-D1 L2':>10}{'D10-D1 Raw':>12}"
          f"{'ρ_mono L2':>11}{'ρ_mono DD':>11}{'inv':>5}"
          f"{'CAGR@0':>9}{'CAGR@60':>10}{'Shp@60':>8}{'DD@60':>8}{'Turn':>6}  verdict")
    for name, h in per_model.items():
        pf = _pass_fail(h)
        verdict = "PASS" if pf["pass_all"] else \
                  ("PART" if (pf["pass_d10_d1_sharpe_bar"] or pf["pass_cagr_60bps_bar"]) else "FAIL")
        print(f"  {name:<8}"
              f"{_fmt_num(h['d10_d1_sharpe_L2'], 10)}"
              f"{_fmt_num(h['d10_d1_sharpe_raw'], 12)}"
              f"{_fmt_num(h['spearman_monotonicity_L2'], 11)}"
              f"{_fmt_num(h['spearman_monotonicity_DD'], 11)}"
              f"{(str(h.get('n_inversions_L2')) if h.get('n_inversions_L2') is not None else '—'):>5}"
              f"{_fmt_pct(h['cagr_net_0bps']):>9}"
              f"{_fmt_pct(h['cagr_net_60bps']):>10}"
              f"{_fmt_num(h['sharpe_net_60bps'], 8)}"
              f"{_fmt_pct(h['max_drawdown_net_60bps']):>8}"
              f"{_fmt_pct(h['avg_turnover']):>6}"
              f"  {verdict}")

    if ablations:
        print()
        print("  M2 ABLATIONS:")
        print(f"  {'ablation':<30}{'nFeat':>6}{'D10-D1':>9}{'ρ_mono':>9}"
              f"{'CAGR@60':>10}{'Shp@60':>8}{'DD@60':>8}{'Turn':>6}")
        for r in ablations:
            print(f"  {r['ablation']:<30}{r['n_features']:>6}"
                  f"{_fmt_num(r['d10_d1_sharpe_L2'], 9)}"
                  f"{_fmt_num(r['spearman_monotonicity_L2'], 9)}"
                  f"{_fmt_pct(r['cagr_net_60bps']):>10}"
                  f"{_fmt_num(r['sharpe_net_60bps'], 8)}"
                  f"{_fmt_pct(r['max_drawdown_net_60bps']):>8}"
                  f"{_fmt_pct(r['avg_turnover']):>6}")

    # Headline D9-D10 overextension readback — read saved CSVs for M0/M1/M2
    print()
    print("  D9 vs D10 OVEREXTENSION (top-5 |Δ| for each model):")
    for name in per_model:
        p = os.path.join(reports_dir, f"step4_d9_d10_{name}.csv")
        if not os.path.exists(p):
            continue
        try:
            d = pd.read_csv(p).head(5)
        except Exception:
            continue
        print(f"    [{name}]")
        for _, r in d.iterrows():
            print(f"      {r['feature']:<32}  "
                  f"D9={r['mean_d9']:+.3f}  D10={r['mean_d10']:+.3f}  "
                  f"Δ={r['delta_d10_minus_d9']:+.3f}")

    print()
    print(f"  reports written to: {reports_dir}/  (elapsed {time.time() - t0:.1f}s)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--features", default=DEFAULT_FEATURES)
    p.add_argument("--labels",   default=DEFAULT_LABELS)
    p.add_argument("--panel",    default=DEFAULT_PANEL)
    p.add_argument("--top-n",    type=int, default=20)
    p.add_argument("--n-buckets", type=int, default=10)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        run(args)
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        raise
