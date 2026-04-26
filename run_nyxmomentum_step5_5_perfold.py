"""
Step 5.5 — Gate A remediation: per-fold robustness readout.

The original Gate A split the 3-fold OOS window at 2024-01-01 (pre/post).
fold1 was skipped (train_start=2015 pre-dates data coverage), so the OOS
window is 2023-11 → 2026-04 — that split gave N=2 pre vs N=15 post. Too
thin to draw regime conclusions.

This runner replaces that split with a per-fold readout. fold2 / fold3 /
fold4 are disjoint OOS windows (~5-6 months each), so each fold is its
own regime sample.

LOCKED PARAMETERS (no tuning this round — robustness readout only):
  V4: rank_blend(M0, M1, 0.5, 0.5) → strict top-20
  V5: same ensemble → dampener(n_enter=20, n_exit=30, alpha=0.5)
  M0 reference: strict top-20 on raw M0 prediction

Per-fold metrics:
  nCAGR@60, nShp@60, DD@60, turnover, ρ_mono L2, D10-D1 Sharpe (L2)

Cross-fold stability summary: mean/std/min/max on nShp@60, D10-D1 Sharpe,
ρ_mono L2 (the three stability-critical scalars).

OUTPUT (output/nyxmomentum/reports/):
  step5_5_perfold.csv                  (one row per variant×fold)
  step5_5_perfold_stability.csv        (one row per variant, agg stats)
  step5_5_perfold_run_meta.json
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

from nyxmomentum.config import CONFIG
from nyxmomentum.dampener import DampenerConfig, apply_dampener
from nyxmomentum.ensemble import rank_blend
from nyxmomentum.evaluation import (
    decile_table,
    monotonicity_score,
    top_n_portfolio_returns,
    portfolio_cost_table,
)
from nyxmomentum.utils import ensure_dir, save_json


REPORTS_DIR = CONFIG.paths.reports
PREDS_M0    = f"{REPORTS_DIR}/step4_predictions_M0.parquet"
PREDS_M1    = f"{REPORTS_DIR}/step4_predictions_M1.parquet"
LABELS      = f"{REPORTS_DIR}/step1_labels.parquet"

V5_DAMP = DampenerConfig(n_enter=20, n_exit=30, smoothing_alpha=0.5)


# ── Variant builders (identical to Step 5.5) ────────────────────────────────

def _v4_preds(preds_m0: pd.DataFrame, preds_m1: pd.DataFrame,
               w0: float = 0.5, w1: float = 0.5) -> pd.DataFrame:
    return rank_blend(preds_m0, preds_m1, weight_a=w0, weight_b=w1,
                       name_a="M0", name_b="M1")


def _strict_topn_portfolio(preds: pd.DataFrame,
                            labels: pd.DataFrame,
                            top_n: int) -> pd.DataFrame:
    if "fold_id" not in preds.columns:
        preds = preds.assign(fold_id="unknown")
    return top_n_portfolio_returns(preds, labels, top_n=top_n, eligible_only=True)


# ── Per-fold metrics ────────────────────────────────────────────────────────

def _per_fold_metrics(variant: str,
                       preds: pd.DataFrame,
                       portfolio: pd.DataFrame,
                       labels: pd.DataFrame,
                       folds: list[str]) -> pd.DataFrame:
    """
    Slice portfolio and preds by fold_id; compute metrics independently.

    Turnover note: the first rebalance of each fold slice inherits a
    cross-fold turnover value (because the portfolio chain is built
    chronologically across all folds). We zero-out the first turnover of
    each fold so the per-fold 'avg_turnover' reflects within-fold churn
    only. CAGR annualization is 12/n on the fold's own rebalance count.
    """
    rows: list[dict] = []
    for f in folds:
        # Portfolio rows for this fold
        p_fold = portfolio.loc[portfolio["fold_id"] == f].copy()
        if p_fold.empty:
            rows.append({"variant": variant, "fold_id": f, "n_rebalances": 0})
            continue
        p_fold = p_fold.sort_values("rebalance_date").reset_index(drop=True)
        # Neutralize first-rebalance turnover (cross-fold inheritance)
        if len(p_fold) > 0:
            p_fold.loc[p_fold.index[0], "turnover_fraction"] = np.nan

        cost = portfolio_cost_table(p_fold, bps_list=(0, 60, 100, 150))

        def _at(bps: int, col: str) -> float:
            if cost.empty:
                return np.nan
            row = cost.loc[cost["bps_round_trip"] == bps]
            return float(row[col].iloc[0]) if (len(row) and col in row.columns
                                               and pd.notna(row[col].iloc[0])) else np.nan

        # Predictions for this fold — ordering metrics on L2 only
        preds_fold = preds.loc[preds["fold_id"] == f] \
            if "fold_id" in preds.columns else preds
        if preds_fold.empty:
            mono_l2, d10_d1 = {}, np.nan
        else:
            dec_l2 = decile_table(preds_fold, labels,
                                   target_col="l2_excess_vs_universe_median")
            mono_l2 = monotonicity_score(dec_l2) if not dec_l2.empty else {}
            ls = dec_l2.loc[dec_l2["_decile"] == -1] if not dec_l2.empty else pd.DataFrame()
            d10_d1 = float(ls["sharpe_annualized"].iloc[0]) if len(ls) else np.nan

        rows.append({
            "variant":              variant,
            "fold_id":              f,
            "n_rebalances":         int(len(p_fold)),
            "date_start":           p_fold["rebalance_date"].min(),
            "date_end":             p_fold["rebalance_date"].max(),
            "net_cagr_60bps":       _at(60, "cagr_net"),
            "sharpe_net_60bps":     _at(60, "sharpe_annualized_net"),
            "max_drawdown_net_60bps": _at(60, "max_drawdown_net"),
            "avg_turnover":         float(p_fold["turnover_fraction"].dropna().mean())
                                     if p_fold["turnover_fraction"].notna().any() else np.nan,
            "spearman_monotonicity_L2": mono_l2.get("spearman_monotonicity"),
            "d10_d1_sharpe_L2":     d10_d1,
        })
    return pd.DataFrame(rows)


def _stability_summary(per_fold: pd.DataFrame,
                        stat_cols: tuple[str, ...] = (
                            "sharpe_net_60bps",
                            "d10_d1_sharpe_L2",
                            "spearman_monotonicity_L2",
                            "net_cagr_60bps",
                            "max_drawdown_net_60bps",
                            "avg_turnover",
                        )) -> pd.DataFrame:
    """For each variant, aggregate per-fold mean/std/min/max on stat_cols."""
    rows: list[dict] = []
    for variant, g in per_fold.groupby("variant", sort=False):
        row: dict = {"variant": variant}
        for c in stat_cols:
            vals = g[c].astype(float).dropna()
            if len(vals):
                row[f"{c}_mean"] = float(vals.mean())
                row[f"{c}_std"]  = float(vals.std(ddof=0))
                row[f"{c}_min"]  = float(vals.min())
                row[f"{c}_max"]  = float(vals.max())
            else:
                row[f"{c}_mean"] = row[f"{c}_std"] = row[f"{c}_min"] = row[f"{c}_max"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


# ── Printing helpers ────────────────────────────────────────────────────────

def _fmt_pct(v, w: int = 8) -> str:
    return (f"{v:+.1%}" if (v is not None and pd.notna(v) and np.isfinite(v)) else "   —  ").rjust(w)


def _fmt_num(v, w: int = 6) -> str:
    if v is None or pd.isna(v) or not np.isfinite(v):
        return "  —  ".rjust(w)
    return f"{v:+.2f}".rjust(w)


def _print_perfold(per_fold: pd.DataFrame, stability: pd.DataFrame,
                    folds: list[str], t0: float, reports_dir: str) -> None:
    print()
    print("══ nyxmomentum Step 5.5 — per-fold robustness readout ══")
    print()
    print("  PER FOLD (net@60bps):")
    print(f"  {'variant':<22}{'fold':<8}"
          f"{'nCAGR60':>10}{'nShp60':>8}{'DD60':>8}"
          f"{'turn':>7}{'ρ_mono L2':>11}{'D10-D1 Shp':>12}{'N':>4}")
    for _, r in per_fold.iterrows():
        print(f"  {r['variant']:<22}{r['fold_id']:<8}"
              f"{_fmt_pct(r.get('net_cagr_60bps'), 10)}"
              f"{_fmt_num(r.get('sharpe_net_60bps'), 8)}"
              f"{_fmt_pct(r.get('max_drawdown_net_60bps'), 8)}"
              f"{_fmt_pct(r.get('avg_turnover'), 7)}"
              f"{_fmt_num(r.get('spearman_monotonicity_L2'), 11)}"
              f"{_fmt_num(r.get('d10_d1_sharpe_L2'), 12)}"
              f"{int(r['n_rebalances']):>4}")

    print()
    print("  CROSS-FOLD STABILITY (mean ± std [min, max] over folds):")
    print(f"  {'variant':<22}"
          f"{'nShp60':>22}{'D10-D1 L2':>22}{'ρ_mono L2':>22}")
    for _, r in stability.iterrows():
        def _agg(col: str) -> str:
            mu = r.get(f"{col}_mean"); sd = r.get(f"{col}_std")
            lo = r.get(f"{col}_min"); hi = r.get(f"{col}_max")
            if pd.isna(mu):
                return "  —  ".rjust(22)
            return f"{mu:+.2f}±{sd:.2f} [{lo:+.2f},{hi:+.2f}]".rjust(22)
        print(f"  {r['variant']:<22}"
              f"{_agg('sharpe_net_60bps')}"
              f"{_agg('d10_d1_sharpe_L2')}"
              f"{_agg('spearman_monotonicity_L2')}")

    print()
    print(f"  reports: {reports_dir}/step5_5_perfold.csv + _stability.csv"
          f"  (elapsed {time.time() - t0:.1f}s)")


# ── Commentary + verdict ────────────────────────────────────────────────────

def _commentary_and_verdict(per_fold: pd.DataFrame,
                             stability: pd.DataFrame) -> tuple[str, str]:
    """
    Generate Turkish commentary + promotion verdict. Pure function of the
    tables — no human judgment baked in here beyond the pre-registered
    thresholds below.

    Verdict thresholds (pre-registered):
      - promotion-ready shortlist:
          • V4 nShp@60 > 1.5 in ALL folds
          • V4 D10-D1 Sharpe > 1.5 in ALL folds
          • V4 ρ_mono L2 > 0.50 in ALL folds
          • V4 nCAGR@60 beats M0_reference in ≥ 2/3 folds
      - candidate but regime-sensitive:
          • V4 meets above in ≥ 2/3 folds but has at least one weak fold
      - needs more data / inconclusive:
          • otherwise
    """
    def _row(df, variant, fold=None):
        d = df.loc[df["variant"] == variant]
        if fold is not None:
            d = d.loc[d["fold_id"] == fold]
        return d.iloc[0] if len(d) else None

    folds = sorted(per_fold["fold_id"].unique().tolist())

    def _pass_in_fold(variant: str, fold: str) -> dict:
        r = _row(per_fold, variant, fold)
        if r is None:
            return {"n_shp": False, "d10_d1": False, "mono": False, "beats_m0": False}
        m0r = _row(per_fold, "M0_reference", fold)
        return {
            "n_shp":   bool(r.get("sharpe_net_60bps", np.nan) > 1.5),
            "d10_d1":  bool(r.get("d10_d1_sharpe_L2", np.nan) > 1.5),
            "mono":    bool(r.get("spearman_monotonicity_L2", np.nan) > 0.50),
            "beats_m0": bool(
                pd.notna(r.get("net_cagr_60bps")) and
                pd.notna(m0r.get("net_cagr_60bps") if m0r is not None else np.nan) and
                r["net_cagr_60bps"] > m0r["net_cagr_60bps"]
            ) if m0r is not None else False,
        }

    v4_flags = {f: _pass_in_fold("V4_ensemble_strict", f) for f in folds}
    v5_flags = {f: _pass_in_fold("V5_ensemble_damp",   f) for f in folds}

    v4_all = {k: sum(1 for f in folds if v4_flags[f][k]) for k in ["n_shp", "d10_d1", "mono", "beats_m0"]}
    v5_all = {k: sum(1 for f in folds if v5_flags[f][k]) for k in ["n_shp", "d10_d1", "mono", "beats_m0"]}
    n = len(folds)

    # Pick the fold where V4 is clearly best (highest D10-D1 × mono composite)
    def _best_fold(variant: str) -> str | None:
        d = per_fold.loc[per_fold["variant"] == variant].copy()
        if d.empty:
            return None
        d["_score"] = d["d10_d1_sharpe_L2"].fillna(-9) + d["spearman_monotonicity_L2"].fillna(-9)
        return str(d.sort_values("_score", ascending=False).iloc[0]["fold_id"])

    # Pick the fold where V5 is most robust (lowest DD and turnover; highest Sharpe)
    def _most_robust_fold(variant: str) -> str | None:
        d = per_fold.loc[per_fold["variant"] == variant].copy()
        if d.empty:
            return None
        # Composite = sharpe + |DD|^-1 (smaller DD = more robust) + inverse turnover
        d["_robust"] = (d["sharpe_net_60bps"].fillna(0)
                        - d["max_drawdown_net_60bps"].fillna(-1) * 5
                        - d["avg_turnover"].fillna(1))
        return str(d.sort_values("_robust", ascending=False).iloc[0]["fold_id"])

    v4_best_fold = _best_fold("V4_ensemble_strict")
    v5_robust_fold = _most_robust_fold("V5_ensemble_damp")

    # M0-beating consistency
    m0_beats_v4 = sum(1 for f in folds if v4_flags[f]["beats_m0"])
    m0_beats_v5 = sum(1 for f in folds if v5_flags[f]["beats_m0"])

    # Promotion verdict
    all_pass_v4 = (v4_all["n_shp"] == n and v4_all["d10_d1"] == n
                   and v4_all["mono"] == n and v4_all["beats_m0"] >= 2)
    mostly_pass_v4 = (v4_all["n_shp"] >= n - 1 and v4_all["d10_d1"] >= n - 1
                       and v4_all["mono"] >= n - 1 and v4_all["beats_m0"] >= 1)

    if all_pass_v4:
        verdict = "promotion-ready shortlist"
    elif mostly_pass_v4:
        verdict = "candidate but regime-sensitive"
    else:
        verdict = "needs more data / inconclusive"

    # Narrative
    lines: list[str] = []
    lines.append(f"V4 en güçlü fold: **{v4_best_fold}** "
                  f"(D10-D1 Sharpe + ρ_mono L2 kompoziti en yüksek).")
    lines.append(f"V5 en robust fold: **{v5_robust_fold}** "
                  f"(sharpe + düşük DD + düşük turnover kompoziti).")
    lines.append(f"V4 M0'ı net CAGR@60bps'de geçiyor: **{m0_beats_v4}/{n}** fold.")
    lines.append(f"V5 M0'ı net CAGR@60bps'de geçiyor: **{m0_beats_v5}/{n}** fold.")

    # Spot weak folds
    weak_folds_v4 = [f for f in folds if not all(v4_flags[f].values())]
    if weak_folds_v4:
        lines.append(f"V4 zayıf fold(lar): {', '.join(weak_folds_v4)} "
                      f"— threshold'lardan en az biri tutmadı.")
    else:
        lines.append("V4 tüm foldlarda tüm threshold'ları geçti.")

    # M0-beating sensitivity
    if m0_beats_v4 == n:
        lines.append("M0-beating pencere-bağımsız: V4 tüm OOS foldlarda kazanıyor.")
    elif m0_beats_v4 >= 2:
        lines.append(f"M0-beating kısmen pencere-bağımlı: {m0_beats_v4}/{n} foldda kazanıyor, "
                      f"tek fold kaybediyor → pencere hassasiyeti var.")
    else:
        lines.append(f"M0-beating pencere-bağımlı: sadece {m0_beats_v4}/{n} foldda kazanıyor.")

    commentary = "\n".join(f"  • {s}" for s in lines)
    return commentary, verdict


# ── Run ─────────────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    t0 = time.time()
    reports_dir = ensure_dir(CONFIG.paths.reports)

    print("[1/4] Loading …")
    preds_m0 = pd.read_parquet(args.preds_m0)
    preds_m1 = pd.read_parquet(args.preds_m1)
    labels   = pd.read_parquet(args.labels)
    folds = sorted(preds_m0["fold_id"].astype(str).unique().tolist())
    print(f"  preds_M0={len(preds_m0):,}  preds_M1={len(preds_m1):,}  "
          f"labels={len(labels):,}")
    print(f"  folds: {folds}")

    print("[2/4] Building variants (V4 strict, V5 damp, M0 reference)…")
    # V4 ensemble strict
    v4_preds = _v4_preds(preds_m0, preds_m1, w0=0.5, w1=0.5)
    v4_port  = _strict_topn_portfolio(v4_preds, labels, top_n=args.top_n)

    # V5 ensemble + dampener
    _, v5_port = apply_dampener(v4_preds, labels, V5_DAMP)
    # Dampener portfolio frame doesn't carry fold_id — backfill from predictions
    fold_map = v4_preds.groupby("rebalance_date")["fold_id"].first()
    v5_port = v5_port.assign(
        fold_id=v5_port["rebalance_date"].map(fold_map)
    )
    v5_preds = v4_preds  # same predictions; selection differs

    # M0 reference
    m0_port  = _strict_topn_portfolio(preds_m0, labels, top_n=args.top_n)

    print("[3/4] Per-fold metrics …")
    per_fold = pd.concat([
        _per_fold_metrics("V4_ensemble_strict", v4_preds, v4_port, labels, folds),
        _per_fold_metrics("V5_ensemble_damp",   v5_preds, v5_port, labels, folds),
        _per_fold_metrics("M0_reference",       preds_m0, m0_port, labels, folds),
    ], ignore_index=True)
    per_fold.to_csv(os.path.join(reports_dir, "step5_5_perfold.csv"), index=False)

    print("[4/4] Cross-fold stability summary …")
    stability = _stability_summary(per_fold)
    stability.to_csv(os.path.join(reports_dir, "step5_5_perfold_stability.csv"),
                     index=False)

    commentary, verdict = _commentary_and_verdict(per_fold, stability)

    meta = {
        "produced_at":    pd.Timestamp.utcnow().isoformat(),
        "top_n":          args.top_n,
        "folds":          folds,
        "v5_config":      {"n_enter": V5_DAMP.n_enter, "n_exit": V5_DAMP.n_exit,
                           "smoothing_alpha": V5_DAMP.smoothing_alpha},
        "weights_locked": {"w_M0": 0.5, "w_M1": 0.5},
        "verdict":        verdict,
        "commentary":     commentary,
        "elapsed_sec":    time.time() - t0,
    }
    save_json(os.path.join(reports_dir, "step5_5_perfold_run_meta.json"), meta)

    _print_perfold(per_fold, stability, folds, t0, reports_dir)
    print()
    print("  COMMENTARY:")
    print(commentary)
    print()
    print(f"  PROMOTION VERDICT: **{verdict}**")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--preds-m0", default=PREDS_M0)
    p.add_argument("--preds-m1", default=PREDS_M1)
    p.add_argument("--labels",   default=LABELS)
    p.add_argument("--top-n",    type=int, default=20)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        run(args)
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        raise
