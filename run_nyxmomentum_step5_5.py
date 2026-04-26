"""
Step 5.5 — promotion gate before V4/V5 are locked as production candidates.

Step 5 produced one encouraging OOS result; before treating it as the winner,
four independent stress tests must pass. This runner executes all four and
emits a single pass/fail table so nothing sneaks through.

GATES:

  (A) SUBPERIOD ROBUSTNESS
      V4 and V5 evaluated separately on pre-2024-01-01 and post-2024-01-01
      test rows. Reports per-half: net CAGR@60, net Sharpe@60, max DD@60,
      D10-D1 Sharpe on L2, ρ_mono L2. One-leg alpha → candidate status
      weakens.

  (B) ENSEMBLE WEIGHT SENSITIVITY
      Rebuild V4 (strict top-N) with (w_M0, w_M1) in {30/70, 50/50, 70/30}.
      Same for V5 (ensemble + dampener). Not an optimum search — a
      fragility check. If only 50/50 works, that is a bad sign.

  (C) COST STRESS
      V4 and V5 at 60 / 100 / 150 bps round-trip. Expectation: V5's lower
      turnover should let it hold up better at 150bps. If V4 collapses
      while V5 stays above M0 baseline, that changes the primary candidate.

  (D) SIMPLICITY COMPARATOR
      Is the ensemble earning its complexity, or is V4 ≈ "M0 plus a single
      anti-stretch correction"? Build:
        M0                                   (reference, already have V0)
        M0 + anti-stretch penalty            (one feature: px_over_ma50_z)
        V4 ensemble                          (reference, already have V4)
      If the penalty variant closes most of the gap, ensemble is over-
      engineered and the simpler thing wins on Occam's razor.

OUTPUT (output/nyxmomentum/reports/):
  step5_5_subperiod.csv
  step5_5_weight_sensitivity.csv
  step5_5_cost_stress.csv
  step5_5_simplicity.csv
  step5_5_run_meta.json
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
    holding_persistence,
)
from nyxmomentum.utils import ensure_dir, save_json


REPORTS_DIR = CONFIG.paths.reports
PREDS_M0   = f"{REPORTS_DIR}/step4_predictions_M0.parquet"
PREDS_M1   = f"{REPORTS_DIR}/step4_predictions_M1.parquet"
LABELS     = f"{REPORTS_DIR}/step1_labels.parquet"
FEATURES   = f"{REPORTS_DIR}/step2_features.parquet"

M0_REFERENCE_CAGR = 0.877   # Step 3 handcrafted_no_overlay

# V5 dampener convention — same as Step 5
V5_DAMP = DampenerConfig(n_enter=20, n_exit=30, smoothing_alpha=0.5)


# ── Common metrics helper ───────────────────────────────────────────────────

def _metrics_from_portfolio(portfolio: pd.DataFrame,
                             preds: pd.DataFrame,
                             labels: pd.DataFrame) -> dict:
    """Compute gross/net CAGR, Sharpe, DD, turnover, ordering metrics."""
    cost = portfolio_cost_table(portfolio, bps_list=(0, 60, 100, 150)) \
        if not portfolio.empty else pd.DataFrame()

    def _at(bps: int, col: str) -> float:
        if cost.empty:
            return np.nan
        row = cost.loc[cost["bps_round_trip"] == bps]
        return float(row[col].iloc[0]) if (len(row) and col in row.columns and pd.notna(row[col].iloc[0])) else np.nan

    gross = portfolio["portfolio_return"].astype(float).dropna()
    n = len(gross)
    g_mu = float(gross.mean()) if n else np.nan
    g_sd = float(gross.std(ddof=0)) if n else np.nan
    gross_sharpe = float(g_mu / g_sd * np.sqrt(12)) if (g_sd and g_sd > 0) else np.nan
    gross_cagr = float((1.0 + gross).prod() ** (12.0 / n) - 1.0) if n > 0 else np.nan

    dec_l2 = decile_table(preds, labels, target_col="l2_excess_vs_universe_median")
    mono_l2 = monotonicity_score(dec_l2) if not dec_l2.empty else {}
    ls_row = dec_l2.loc[dec_l2["_decile"] == -1] if not dec_l2.empty else pd.DataFrame()
    d10_d1 = float(ls_row["sharpe_annualized"].iloc[0]) if len(ls_row) else np.nan

    return {
        "n_rebalances":            n,
        "gross_cagr":              gross_cagr,
        "sharpe_gross":            gross_sharpe,
        "net_cagr_60bps":          _at(60, "cagr_net"),
        "net_cagr_100bps":         _at(100, "cagr_net"),
        "net_cagr_150bps":         _at(150, "cagr_net"),
        "sharpe_net_60bps":        _at(60, "sharpe_annualized_net"),
        "sharpe_net_100bps":       _at(100, "sharpe_annualized_net"),
        "sharpe_net_150bps":       _at(150, "sharpe_annualized_net"),
        "max_drawdown_net_60bps":  _at(60, "max_drawdown_net"),
        "max_drawdown_net_100bps": _at(100, "max_drawdown_net"),
        "max_drawdown_net_150bps": _at(150, "max_drawdown_net"),
        "avg_turnover":            float(portfolio["turnover_fraction"].dropna().mean())
                                     if "turnover_fraction" in portfolio.columns else np.nan,
        "spearman_monotonicity_L2": mono_l2.get("spearman_monotonicity"),
        "d10_d1_sharpe_L2":         d10_d1,
    }


def _strict_topn_portfolio(preds: pd.DataFrame,
                            labels: pd.DataFrame,
                            top_n: int) -> pd.DataFrame:
    if "fold_id" not in preds.columns:
        preds = preds.assign(fold_id="unknown")
    return top_n_portfolio_returns(preds, labels, top_n=top_n, eligible_only=True)


def _v4_preds(preds_m0: pd.DataFrame, preds_m1: pd.DataFrame,
              w0: float = 0.5, w1: float = 0.5) -> pd.DataFrame:
    return rank_blend(preds_m0, preds_m1, weight_a=w0, weight_b=w1,
                       name_a="M0", name_b="M1")


# ── Gate A: subperiod robustness ────────────────────────────────────────────

def gate_a_subperiod(preds_m0, preds_m1, labels, split_date: str, top_n: int) -> pd.DataFrame:
    """Split V4/V5 rebalance dates pre/post split_date; evaluate each half."""
    cutoff = pd.Timestamp(split_date)
    rows: list[dict] = []

    def _eval(variant_name: str, preds: pd.DataFrame, portfolio: pd.DataFrame,
              period: str, period_mask_preds, period_mask_port):
        pp = preds.loc[period_mask_preds]
        po = portfolio.loc[period_mask_port]
        if pp.empty or po.empty:
            return None
        m = _metrics_from_portfolio(po, pp, labels)
        m["variant"] = variant_name
        m["period"] = period
        return m

    v4_preds = _v4_preds(preds_m0, preds_m1)
    v4_port = _strict_topn_portfolio(v4_preds, labels, top_n=top_n)
    _, v5_port = apply_dampener(_v4_preds(preds_m0, preds_m1), labels, V5_DAMP)
    v5_preds = v4_preds  # same predictions, different selection

    for (variant_name, preds_frame, port_frame) in [
        ("V4_ensemble_strict",    v4_preds, v4_port),
        ("V5_ensemble_damp",      v5_preds, v5_port),
    ]:
        for period, mask_preds, mask_port in [
            ("pre_2024",
                preds_frame["rebalance_date"] < cutoff,
                port_frame["rebalance_date"] < cutoff),
            ("post_2024",
                preds_frame["rebalance_date"] >= cutoff,
                port_frame["rebalance_date"] >= cutoff),
        ]:
            r = _eval(variant_name, preds_frame, port_frame, period, mask_preds, mask_port)
            if r is not None:
                rows.append(r)
    return pd.DataFrame(rows)


# ── Gate B: ensemble weight sensitivity ─────────────────────────────────────

def gate_b_weight_sensitivity(preds_m0, preds_m1, labels, top_n: int) -> pd.DataFrame:
    weight_pairs = [(0.3, 0.7), (0.5, 0.5), (0.7, 0.3)]
    rows: list[dict] = []
    for w0, w1 in weight_pairs:
        tag = f"{int(w0*100)}_{int(w1*100)}"
        preds = _v4_preds(preds_m0, preds_m1, w0=w0, w1=w1)

        port_strict = _strict_topn_portfolio(preds, labels, top_n=top_n)
        m = _metrics_from_portfolio(port_strict, preds, labels)
        m.update({"variant": "V4_strict", "weights": tag,
                  "w_M0": w0, "w_M1": w1})
        rows.append(m)

        _, port_damp = apply_dampener(preds, labels, V5_DAMP)
        m = _metrics_from_portfolio(port_damp, preds, labels)
        m.update({"variant": "V5_damp", "weights": tag,
                  "w_M0": w0, "w_M1": w1})
        rows.append(m)
    return pd.DataFrame(rows)


# ── Gate C: cost stress ─────────────────────────────────────────────────────

def gate_c_cost_stress(preds_m0, preds_m1, labels, top_n: int) -> pd.DataFrame:
    """Same V4/V5 built at 50/50 as in Step 5, report cost at 60/100/150bps."""
    v4_preds = _v4_preds(preds_m0, preds_m1)
    v4_port = _strict_topn_portfolio(v4_preds, labels, top_n=top_n)
    _, v5_port = apply_dampener(v4_preds, labels, V5_DAMP)

    rows: list[dict] = []
    for name, port in [("V4_ensemble_strict", v4_port),
                       ("V5_ensemble_damp",   v5_port),
                       ("V0_M0_baseline_raw", _strict_topn_portfolio(preds_m0, labels, top_n))]:
        cost = portfolio_cost_table(port, bps_list=(0, 60, 100, 150))
        for _, c in cost.iterrows():
            rows.append({
                "variant":                name,
                "bps_round_trip":         int(c["bps_round_trip"]),
                "cagr_net":               float(c.get("cagr_net", np.nan)) if pd.notna(c.get("cagr_net")) else np.nan,
                "sharpe_annualized_net":  float(c.get("sharpe_annualized_net", np.nan)) if pd.notna(c.get("sharpe_annualized_net")) else np.nan,
                "max_drawdown_net":       float(c.get("max_drawdown_net", np.nan)) if pd.notna(c.get("max_drawdown_net")) else np.nan,
                "avg_turnover":           float(c.get("avg_turnover", np.nan)) if pd.notna(c.get("avg_turnover")) else np.nan,
            })
    return pd.DataFrame(rows)


# ── Gate D: simplicity comparator (M0 + anti-stretch penalty) ───────────────

def _zscore_per_date(df: pd.DataFrame, col: str) -> pd.Series:
    """Per-rebalance-date z-score. NaN → 0 (median of the cross-section)."""
    def _z(x: pd.Series) -> pd.Series:
        sd = x.std(ddof=0)
        return pd.Series(0.0, index=x.index) if (not np.isfinite(sd) or sd == 0) \
            else (x - x.mean()) / sd
    return df.groupby("rebalance_date", sort=False)[col].transform(_z)


def _m0_anti_stretch_preds(preds_m0: pd.DataFrame,
                            features: pd.DataFrame,
                            stretch_col: str = "px_over_ma50_zscore_20d",
                            lam: float = 0.5) -> pd.DataFrame:
    """
    One-feature anti-stretch correction, nothing more:

       score_corrected = z_date(M0_score)  −  λ · z_date(stretch_feature)

    Same per-date cross-sectional z-score transform on both sides, then
    weighted subtraction. λ fixed at 0.5 — chosen as the 'reasonable but
    not tuned' default. This is deliberately less expressive than M1 (all
    20 features) or V4 (rank blend) to test whether simpler wins.
    """
    m = preds_m0[["ticker", "rebalance_date", "eligible", "prediction", "fold_id"]].merge(
        features[["ticker", "rebalance_date", stretch_col]],
        on=["ticker", "rebalance_date"], how="inner",
    )
    # Fill NaN stretch with its median so z-score is defined for every row.
    m[stretch_col] = m[stretch_col].fillna(m[stretch_col].median())
    m["_z_m0"]      = _zscore_per_date(m, "prediction")
    m["_z_stretch"] = _zscore_per_date(m, stretch_col)
    m["prediction_original"] = m["prediction"]
    m["prediction"] = m["_z_m0"] - lam * m["_z_stretch"]
    return m[["ticker", "rebalance_date", "eligible", "prediction", "fold_id",
              "prediction_original", "_z_stretch"]]


def gate_d_simplicity(preds_m0, preds_m1, labels, features, top_n: int) -> pd.DataFrame:
    rows: list[dict] = []

    # M0 (reference)
    port_m0 = _strict_topn_portfolio(preds_m0, labels, top_n=top_n)
    m = _metrics_from_portfolio(port_m0, preds_m0, labels)
    m["variant"] = "M0_reference"
    rows.append(m)

    # M0 + anti-stretch penalty (λ=0.5 on px_over_ma50_zscore_20d)
    preds_as = _m0_anti_stretch_preds(preds_m0, features,
                                        stretch_col="px_over_ma50_zscore_20d",
                                        lam=0.5)
    port_as = _strict_topn_portfolio(preds_as, labels, top_n=top_n)
    m = _metrics_from_portfolio(port_as, preds_as, labels)
    m["variant"] = "M0_plus_anti_stretch"
    rows.append(m)

    # V4 ensemble (reference)
    preds_v4 = _v4_preds(preds_m0, preds_m1)
    port_v4 = _strict_topn_portfolio(preds_v4, labels, top_n=top_n)
    m = _metrics_from_portfolio(port_v4, preds_v4, labels)
    m["variant"] = "V4_ensemble"
    rows.append(m)

    return pd.DataFrame(rows)


# ── Printing helpers ────────────────────────────────────────────────────────

def _fmt_pct(v, w: int = 8) -> str:
    return (f"{v:+.1%}" if (v is not None and pd.notna(v) and np.isfinite(v)) else "   —  ").rjust(w)


def _fmt_num(v, w: int = 6) -> str:
    if v is None or pd.isna(v) or not np.isfinite(v):
        return "  —  ".rjust(w)
    return f"{v:+.2f}".rjust(w)


def _print_summary(sub, wts, cost, simp, reports_dir: str, t0: float) -> None:
    print()
    print("══ nyxmomentum Step 5.5 — promotion gate ══")

    # A
    print()
    print("  (A) SUBPERIOD ROBUSTNESS (split @ 2024-01-01):")
    print(f"  {'variant':<20}{'period':<12}"
          f"{'nCAGR60':>10}{'nShp60':>8}{'DD60':>8}"
          f"{'ρ_mono L2':>11}{'D10-D1':>9}{'turn':>7}{'N':>4}")
    for _, r in sub.iterrows():
        print(f"  {r['variant']:<20}{r['period']:<12}"
              f"{_fmt_pct(r['net_cagr_60bps'], 10)}"
              f"{_fmt_num(r['sharpe_net_60bps'], 8)}"
              f"{_fmt_pct(r['max_drawdown_net_60bps'], 8)}"
              f"{_fmt_num(r['spearman_monotonicity_L2'], 11)}"
              f"{_fmt_num(r['d10_d1_sharpe_L2'], 9)}"
              f"{_fmt_pct(r['avg_turnover'], 7)}"
              f"{int(r['n_rebalances']):>4}")

    # B
    print()
    print("  (B) ENSEMBLE WEIGHT SENSITIVITY (w_M0 / w_M1):")
    print(f"  {'variant':<12}{'weights':>10}"
          f"{'nCAGR60':>10}{'nShp60':>8}{'DD60':>8}"
          f"{'ρ_mono':>9}{'D10-D1':>9}{'turn':>7}")
    for _, r in wts.iterrows():
        print(f"  {r['variant']:<12}{r['weights']:>10}"
              f"{_fmt_pct(r['net_cagr_60bps'], 10)}"
              f"{_fmt_num(r['sharpe_net_60bps'], 8)}"
              f"{_fmt_pct(r['max_drawdown_net_60bps'], 8)}"
              f"{_fmt_num(r['spearman_monotonicity_L2'], 9)}"
              f"{_fmt_num(r['d10_d1_sharpe_L2'], 9)}"
              f"{_fmt_pct(r['avg_turnover'], 7)}")

    # C
    print()
    print("  (C) COST STRESS (60 / 100 / 150 bps round-trip):")
    piv_cagr = cost.pivot_table(index="variant", columns="bps_round_trip",
                                 values="cagr_net").reindex(columns=[0, 60, 100, 150])
    piv_shp  = cost.pivot_table(index="variant", columns="bps_round_trip",
                                 values="sharpe_annualized_net").reindex(columns=[0, 60, 100, 150])
    piv_dd   = cost.pivot_table(index="variant", columns="bps_round_trip",
                                 values="max_drawdown_net").reindex(columns=[0, 60, 100, 150])
    print("    CAGR_net:")
    print(piv_cagr.map(lambda v: f"{v:+.1%}" if pd.notna(v) else "   —  ").to_string())
    print()
    print("    Sharpe_net:")
    print(piv_shp.map(lambda v: f"{v:+.2f}" if pd.notna(v) else "   — ").to_string())
    print()
    print("    MaxDD_net:")
    print(piv_dd.map(lambda v: f"{v:+.1%}" if pd.notna(v) else "   —  ").to_string())

    # D
    print()
    print("  (D) SIMPLICITY COMPARATOR:")
    print(f"  {'variant':<24}{'nCAGR60':>10}{'nShp60':>8}{'DD60':>8}"
          f"{'ρ_mono':>9}{'D10-D1':>9}{'turn':>7}")
    for _, r in simp.iterrows():
        print(f"  {r['variant']:<24}"
              f"{_fmt_pct(r['net_cagr_60bps'], 10)}"
              f"{_fmt_num(r['sharpe_net_60bps'], 8)}"
              f"{_fmt_pct(r['max_drawdown_net_60bps'], 8)}"
              f"{_fmt_num(r['spearman_monotonicity_L2'], 9)}"
              f"{_fmt_num(r['d10_d1_sharpe_L2'], 9)}"
              f"{_fmt_pct(r['avg_turnover'], 7)}")

    # Gate verdict
    print()
    print("  GATE VERDICTS:")
    # A: consistent across halves?
    def _both_above(var: str, col: str, thresh: float) -> bool:
        d = sub.loc[sub["variant"] == var]
        if len(d) < 2:
            return False
        return bool((d[col] > thresh).all())
    v4_both_sharpe = _both_above("V4_ensemble_strict", "sharpe_net_60bps", 1.5)
    v5_both_sharpe = _both_above("V5_ensemble_damp",   "sharpe_net_60bps", 1.5)
    v4_both_mono = _both_above("V4_ensemble_strict", "spearman_monotonicity_L2", 0.50)
    v5_both_mono = _both_above("V5_ensemble_damp",   "spearman_monotonicity_L2", 0.50)
    print(f"    (A) V4 both-halves Shp>1.5 & mono>0.5: "
          f"{'✓' if (v4_both_sharpe and v4_both_mono) else '✗'}")
    print(f"    (A) V5 both-halves Shp>1.5 & mono>0.5: "
          f"{'✓' if (v5_both_sharpe and v5_both_mono) else '✗'}")

    # B: all three weight pairs give Sharpe > 2.0?
    def _all_above(df, var: str, col: str, thresh: float) -> bool:
        d = df.loc[df["variant"] == var]
        return bool(len(d) == 3 and (d[col] > thresh).all())
    v4_b = _all_above(wts, "V4_strict", "sharpe_net_60bps", 2.0)
    v5_b = _all_above(wts, "V5_damp",   "sharpe_net_60bps", 2.0)
    print(f"    (B) V4 all weight pairs Shp>2.0: {'✓' if v4_b else '✗'}")
    print(f"    (B) V5 all weight pairs Shp>2.0: {'✓' if v5_b else '✗'}")

    # C: at 150bps, does V4/V5 beat M0_reference_at_150bps?
    def _cost_at(variant: str, bps: int, col: str) -> float:
        r = cost.loc[(cost["variant"] == variant) & (cost["bps_round_trip"] == bps)]
        return float(r[col].iloc[0]) if len(r) else np.nan
    m0_150 = _cost_at("V0_M0_baseline_raw", 150, "cagr_net")
    v4_150 = _cost_at("V4_ensemble_strict", 150, "cagr_net")
    v5_150 = _cost_at("V5_ensemble_damp",   150, "cagr_net")
    print(f"    (C) V4 @150bps beats M0 @150bps ({m0_150:+.1%}): "
          f"{'✓' if pd.notna(v4_150) and v4_150 > m0_150 else '✗'} ({v4_150:+.1%})")
    print(f"    (C) V5 @150bps beats M0 @150bps ({m0_150:+.1%}): "
          f"{'✓' if pd.notna(v5_150) and v5_150 > m0_150 else '✗'} ({v5_150:+.1%})")

    # D: does V4 meaningfully beat M0+anti_stretch?
    v4_cagr = float(simp.loc[simp["variant"] == "V4_ensemble",
                              "net_cagr_60bps"].iloc[0])
    as_cagr = float(simp.loc[simp["variant"] == "M0_plus_anti_stretch",
                              "net_cagr_60bps"].iloc[0])
    m0_cagr = float(simp.loc[simp["variant"] == "M0_reference",
                              "net_cagr_60bps"].iloc[0])
    # "Meaningful" = closes ≥50% of the V4−M0 gap with the simple correction.
    gap_closed = (as_cagr - m0_cagr) / max(v4_cagr - m0_cagr, 1e-9)
    verdict_d = "ensemble earns complexity" if gap_closed < 0.5 else "simpler variant competitive"
    print(f"    (D) M0+anti_stretch closes {gap_closed:+.0%} of V4−M0 gap → {verdict_d}")

    print()
    print(f"  reports written to: {reports_dir}/  (elapsed {time.time() - t0:.1f}s)")


# ── Run ─────────────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    t0 = time.time()
    reports_dir = ensure_dir(CONFIG.paths.reports)

    print("[1/5] Loading …")
    preds_m0 = pd.read_parquet(args.preds_m0)
    preds_m1 = pd.read_parquet(args.preds_m1)
    labels   = pd.read_parquet(args.labels)
    features = pd.read_parquet(args.features)
    print(f"  preds_M0={len(preds_m0):,}  preds_M1={len(preds_m1):,}  "
          f"labels={len(labels):,}  features={len(features):,}")

    print("[2/5] (A) Subperiod robustness …")
    sub = gate_a_subperiod(preds_m0, preds_m1, labels,
                            split_date=args.split_date, top_n=args.top_n)
    sub.to_csv(os.path.join(reports_dir, "step5_5_subperiod.csv"), index=False)

    print("[3/5] (B) Ensemble weight sensitivity …")
    wts = gate_b_weight_sensitivity(preds_m0, preds_m1, labels, top_n=args.top_n)
    wts.to_csv(os.path.join(reports_dir, "step5_5_weight_sensitivity.csv"), index=False)

    print("[4/5] (C) Cost stress …")
    cost = gate_c_cost_stress(preds_m0, preds_m1, labels, top_n=args.top_n)
    cost.to_csv(os.path.join(reports_dir, "step5_5_cost_stress.csv"), index=False)

    print("[5/5] (D) Simplicity comparator (anti-stretch λ=0.5 on px_over_ma50_zscore_20d) …")
    simp = gate_d_simplicity(preds_m0, preds_m1, labels, features, top_n=args.top_n)
    simp.to_csv(os.path.join(reports_dir, "step5_5_simplicity.csv"), index=False)

    meta = {
        "produced_at": pd.Timestamp.utcnow().isoformat(),
        "split_date":  args.split_date,
        "top_n":       args.top_n,
        "weight_pairs": [(0.3, 0.7), (0.5, 0.5), (0.7, 0.3)],
        "bps_stress":   [60, 100, 150],
        "anti_stretch_feature": "px_over_ma50_zscore_20d",
        "anti_stretch_lambda":  0.5,
        "m0_reference_cagr":   M0_REFERENCE_CAGR,
        "elapsed_sec": time.time() - t0,
    }
    save_json(os.path.join(reports_dir, "step5_5_run_meta.json"), meta)

    _print_summary(sub, wts, cost, simp, reports_dir, t0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--preds-m0", default=PREDS_M0)
    p.add_argument("--preds-m1", default=PREDS_M1)
    p.add_argument("--labels",   default=LABELS)
    p.add_argument("--features", default=FEATURES)
    p.add_argument("--top-n",    type=int, default=20)
    p.add_argument("--split-date", default="2024-01-01")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        run(args)
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        raise
