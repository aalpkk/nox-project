"""
Step 5 runner — information-to-turnover efficiency comparison.

The Step 4 diagnostic localized the binding constraint: M1 has ordering
information, M0 has economic efficiency. Step 5 tests whether rule-based,
ex-ante dampeners can carry M1's ordering into a net-of-cost portfolio,
and whether a rank ensemble of M0+M1 splits the difference.

NOT a new-model sweep. All dampening is explicit, rule-based, same-across-
folds. No per-fold tuning.

Variants (all on the same fold2+3+4 OOS preds from Step 4):
  V0. M0 baseline                                 (reference — no change)
  V1. M1 raw                                      (strict top-N, no dampener)
  V2. M1 + smoothing(α=0.5) + hysteresis(20/30)   (both dampeners on)
  V3. M1 + persistence(20/35)                     (stronger hysteresis, no smooth)
  V4. M0+M1 rank ensemble 50/50, strict top-N
  V5. M0+M1 ensemble + dampener(α=0.5, 20/30)

Unified report per variant:
  gross_cagr, net_cagr_60bps, sharpe_gross, sharpe_net_60bps,
  max_drawdown_net_60bps, avg_turnover, avg_names_changed,
  median_hold_months, re_entry_rate,
  spearman_monotonicity_L2, d10_d1_sharpe_L2,
  d10_minus_d9_on_mom_252_skip (D10 reversal indicator)

Success targets (from user directive):
  - ρ_mono L2 > 0.80  AND  ρ_mono DD > 0.50
  - turnover materially below M1 raw (<60% target)
  - net CAGR@60bps approaches or beats M0's 86.8%
  - DD not worse than M1 raw (-6%), preferably preserved

Artifacts (output/nyxmomentum/reports/):
  step5_variant_comparison.csv
  step5_selection_<variant>.parquet
  step5_portfolio_<variant>.csv
  step5_cost_<variant>.csv
  step5_deciles_<variant>.csv       (L2 target only)
  step5_persistence_<variant>.json
  step5_run_meta.json
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
    d9_vs_d10_feature_delta,
    holding_persistence,
    portfolio_cost_table,
    top_n_portfolio_returns,
)
from nyxmomentum.features import FEATURE_COLUMNS
from nyxmomentum.utils import ensure_dir, save_json


REPORTS_DIR = CONFIG.paths.reports
PREDS_M0 = f"{REPORTS_DIR}/step4_predictions_M0.parquet"
PREDS_M1 = f"{REPORTS_DIR}/step4_predictions_M1.parquet"
LABELS   = f"{REPORTS_DIR}/step1_labels.parquet"
FEATURES = f"{REPORTS_DIR}/step2_features.parquet"

# M0 step3-handcrafted-no-overlay reference (from memory / Step 3 audit after
# recent_extreme_21d was dropped)
M0_REFERENCE_CAGR = 0.877


VARIANTS: list[dict] = [
    {"name": "V0_M0_baseline",          "source": "M0",       "dampener": None},
    {"name": "V1_M1_raw",               "source": "M1",       "dampener": None},
    {"name": "V2_M1_smooth_hyst",       "source": "M1",
     "dampener": DampenerConfig(n_enter=20, n_exit=30, smoothing_alpha=0.5)},
    {"name": "V3_M1_persistence",       "source": "M1",
     "dampener": DampenerConfig(n_enter=20, n_exit=35, smoothing_alpha=1.0)},
    {"name": "V4_M0M1_ensemble",        "source": "ensemble", "dampener": None},
    {"name": "V5_M0M1_ensemble_damp",   "source": "ensemble",
     "dampener": DampenerConfig(n_enter=20, n_exit=30, smoothing_alpha=0.5)},
]


# ── Helpers ─────────────────────────────────────────────────────────────────

def _build_strict_topn_selection(preds: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """Strict top-N per rebalance (no hysteresis, no smoothing). Emits the
    same selection schema as hysteresis_selection so the rest of the
    pipeline is uniform."""
    m = preds.copy()
    if "eligible" in m.columns:
        m = m.loc[m["eligible"].astype(bool)].copy()
    m = m.dropna(subset=["prediction"])
    m["rank"] = m.groupby("rebalance_date", sort=False)["prediction"].rank(
        method="first", ascending=False
    )
    m["selected"] = m["rank"] <= top_n
    m["prev_selected"] = False  # unused in strict mode
    return m[["rebalance_date", "ticker", "selected", "prev_selected",
              "rank", "prediction"]].copy()


def _portfolio_from_strict(preds: pd.DataFrame,
                            labels: pd.DataFrame,
                            top_n: int) -> pd.DataFrame:
    """Wrapper that preserves the fold_id column for provenance, then calls
    top_n_portfolio_returns (the Step 4 implementation)."""
    if "fold_id" not in preds.columns:
        preds = preds.assign(fold_id="unknown")
    return top_n_portfolio_returns(preds, labels, top_n=top_n, eligible_only=True)


def _cost_table_from_returns(returns: pd.DataFrame) -> pd.DataFrame:
    return portfolio_cost_table(returns, bps_list=(0, 20, 40, 60, 100))


def _variant_headline(name: str,
                      preds: pd.DataFrame,
                      selection: pd.DataFrame | None,
                      portfolio: pd.DataFrame,
                      cost: pd.DataFrame,
                      labels: pd.DataFrame,
                      features: pd.DataFrame) -> dict:
    """Collect the per-variant report row (user directive §7)."""
    # Decile stats on L2 target
    dec_l2 = decile_table(preds, labels, target_col="l2_excess_vs_universe_median")
    mono_l2 = monotonicity_score(dec_l2) if not dec_l2.empty else {}
    dec_dd = decile_table(preds, labels, target_col="forward_max_dd_intraperiod")
    mono_dd = monotonicity_score(dec_dd) if not dec_dd.empty else {}

    # D10 reversal marker on mom_252_skip
    feat_cols = [c for c in FEATURE_COLUMNS if c in features.columns]
    d9d10 = d9_vs_d10_feature_delta(
        preds, features[["ticker", "rebalance_date", *feat_cols]],
        feat_cols=feat_cols,
    ) if feat_cols else pd.DataFrame()
    delta_mom252 = np.nan
    if not d9d10.empty:
        row = d9d10.loc[d9d10["feature"] == "mom_252d_skip_21d"]
        if len(row):
            delta_mom252 = float(row["delta_d10_minus_d9"].iloc[0])

    # Long-short Sharpe on L2
    ls_row = dec_l2.loc[dec_l2["_decile"] == -1] if not dec_l2.empty else pd.DataFrame()
    d10_d1_sharpe_l2 = float(ls_row["sharpe_annualized"].iloc[0]) if len(ls_row) else np.nan

    # Gross / net
    gross = portfolio["portfolio_return"].astype(float).dropna()
    g_mu = float(gross.mean()) if len(gross) else np.nan
    g_sd = float(gross.std(ddof=0)) if len(gross) else np.nan
    gross_sharpe = float(g_mu / g_sd * np.sqrt(12)) if (g_sd and g_sd > 0) else np.nan
    n = len(gross)
    gross_cagr = float((1.0 + gross).prod() ** (12.0 / n) - 1.0) if n > 0 else np.nan

    def _cost_at(bps: int, col: str) -> float:
        if cost is None or cost.empty:
            return np.nan
        row = cost.loc[cost["bps_round_trip"] == bps]
        if row.empty or col not in row.columns:
            return np.nan
        v = row[col].iloc[0]
        return float(v) if pd.notna(v) else np.nan

    avg_turnover = float(portfolio["turnover_fraction"].dropna().mean()) \
        if not portfolio.empty else np.nan
    avg_names_changed = float(portfolio["names_changed"].dropna().astype(float).mean()) \
        if "names_changed" in portfolio.columns else np.nan

    persist = holding_persistence(
        selection if selection is not None else portfolio,
        rebalances_per_year=12,
    )

    return {
        "variant": name,
        "n_rebalances": n,
        "gross_cagr": gross_cagr,
        "sharpe_gross": gross_sharpe,
        "net_cagr_0bps":    _cost_at(0, "cagr_net"),
        "net_cagr_60bps":   _cost_at(60, "cagr_net"),
        "net_cagr_100bps":  _cost_at(100, "cagr_net"),
        "sharpe_net_60bps":  _cost_at(60, "sharpe_annualized_net"),
        "max_drawdown_net_60bps": _cost_at(60, "max_drawdown_net"),
        "avg_turnover":     avg_turnover,
        "avg_names_changed": avg_names_changed,
        "median_hold_months": persist.get("median_span_months", np.nan),
        "mean_hold_rebalances": persist.get("mean_span_rebalances", np.nan),
        "re_entry_rate":    persist.get("re_entry_rate", np.nan),
        "n_unique_tickers": persist.get("n_unique_tickers", np.nan),
        "spearman_monotonicity_L2":  mono_l2.get("spearman_monotonicity"),
        "spearman_monotonicity_DD":  mono_dd.get("spearman_monotonicity"),
        "n_inversions_L2":           mono_l2.get("n_inversions"),
        "d10_d1_sharpe_L2":          d10_d1_sharpe_l2,
        "d9_d10_delta_mom252":       delta_mom252,
    }


def _run_variant(v: dict,
                 preds_m0: pd.DataFrame,
                 preds_m1: pd.DataFrame,
                 labels: pd.DataFrame,
                 features: pd.DataFrame,
                 top_n: int,
                 reports_dir: str) -> dict:
    name = v["name"]
    source = v["source"]
    damp = v["dampener"]

    if source == "M0":
        preds = preds_m0.copy()
    elif source == "M1":
        preds = preds_m1.copy()
    elif source == "ensemble":
        preds = rank_blend(preds_m0, preds_m1, weight_a=0.5, weight_b=0.5,
                           name_a="M0", name_b="M1")
    else:
        raise ValueError(f"unknown source: {source}")

    if damp is None:
        selection = _build_strict_topn_selection(preds, top_n=top_n)
        portfolio = _portfolio_from_strict(preds, labels, top_n=top_n)
    else:
        selection, portfolio = apply_dampener(preds, labels, damp, score_col="prediction")

    cost = _cost_table_from_returns(portfolio)

    # Persist artifacts
    selection.to_parquet(
        os.path.join(reports_dir, f"step5_selection_{name}.parquet"), index=False,
    )
    portfolio.to_csv(
        os.path.join(reports_dir, f"step5_portfolio_{name}.csv"), index=False,
    )
    cost.to_csv(os.path.join(reports_dir, f"step5_cost_{name}.csv"), index=False)
    dec_l2 = decile_table(preds, labels, target_col="l2_excess_vs_universe_median")
    if not dec_l2.empty:
        dec_l2.to_csv(
            os.path.join(reports_dir, f"step5_deciles_{name}.csv"), index=False,
        )
    persist = holding_persistence(selection, rebalances_per_year=12)
    save_json(os.path.join(reports_dir, f"step5_persistence_{name}.json"), persist)

    return _variant_headline(
        name=name, preds=preds, selection=selection,
        portfolio=portfolio, cost=cost, labels=labels, features=features,
    )


def run(args: argparse.Namespace) -> None:
    t0 = time.time()
    reports_dir = ensure_dir(CONFIG.paths.reports)

    print("[1/4] Loading Step 4 predictions + labels + features …")
    preds_m0 = pd.read_parquet(args.preds_m0)
    preds_m1 = pd.read_parquet(args.preds_m1)
    labels   = pd.read_parquet(args.labels)
    features = pd.read_parquet(args.features)
    print(f"  preds_M0={len(preds_m0):,}  preds_M1={len(preds_m1):,}  "
          f"labels={len(labels):,}  features={len(features):,}")

    print(f"[2/4] Running {len(VARIANTS)} variants …")
    rows: list[dict] = []
    for v in VARIANTS:
        print(f"  · {v['name']}  ({v['source']}, damp={v['dampener']})")
        row = _run_variant(
            v, preds_m0=preds_m0, preds_m1=preds_m1,
            labels=labels, features=features,
            top_n=args.top_n, reports_dir=reports_dir,
        )
        rows.append(row)

    print("[3/4] Writing comparison table + run_meta …")
    comp = pd.DataFrame(rows)
    comp.to_csv(os.path.join(reports_dir, "step5_variant_comparison.csv"), index=False)

    meta = {
        "produced_at": pd.Timestamp.utcnow().isoformat(),
        "top_n":       args.top_n,
        "variants":    [v["name"] for v in VARIANTS],
        "m0_reference_cagr": M0_REFERENCE_CAGR,
        "elapsed_sec": time.time() - t0,
    }
    save_json(os.path.join(reports_dir, "step5_run_meta.json"), meta)

    print("[4/4] Console summary …")
    _print_summary(comp, reports_dir, t0)


def _fmt_pct(v, w: int = 8) -> str:
    return (f"{v:+.1%}" if (v is not None and pd.notna(v) and np.isfinite(v)) else "   —  ").rjust(w)


def _fmt_num(v, w: int = 6) -> str:
    if v is None or pd.isna(v) or not np.isfinite(v):
        return "  —  ".rjust(w)
    return f"{v:+.2f}".rjust(w)


def _print_summary(comp: pd.DataFrame, reports_dir: str, t0: float) -> None:
    print()
    print("══ nyxmomentum Step 5 ══")
    print("  (information-to-turnover efficiency, not raw CAGR)")
    print()
    print("  ECONOMICS TABLE:")
    print(f"  {'variant':<26}"
          f"{'gCAGR':>8}{'nCAGR60':>10}{'gShp':>6}{'nShp60':>8}{'DD60':>8}"
          f"{'turn':>7}{'chg/rb':>8}{'holdM':>7}{'reEnt':>7}")
    for _, r in comp.iterrows():
        print(f"  {r['variant']:<26}"
              f"{_fmt_pct(r['gross_cagr'], 8)}"
              f"{_fmt_pct(r['net_cagr_60bps'], 10)}"
              f"{_fmt_num(r['sharpe_gross'], 6)}"
              f"{_fmt_num(r['sharpe_net_60bps'], 8)}"
              f"{_fmt_pct(r['max_drawdown_net_60bps'], 8)}"
              f"{_fmt_pct(r['avg_turnover'], 7)}"
              f"{_fmt_num(r.get('avg_names_changed'), 8)}"
              f"{_fmt_num(r.get('median_hold_months'), 7)}"
              f"{_fmt_pct(r.get('re_entry_rate'), 7)}")

    print()
    print("  ORDERING TABLE (on L2):")
    print(f"  {'variant':<26}"
          f"{'ρ_mono L2':>11}{'ρ_mono DD':>11}{'D10-D1':>9}"
          f"{'inv':>5}{'Δ(mom252)':>11}")
    for _, r in comp.iterrows():
        inv = r.get("n_inversions_L2")
        inv_s = str(int(inv)) if (inv is not None and pd.notna(inv)) else "—"
        print(f"  {r['variant']:<26}"
              f"{_fmt_num(r['spearman_monotonicity_L2'], 11)}"
              f"{_fmt_num(r['spearman_monotonicity_DD'], 11)}"
              f"{_fmt_num(r['d10_d1_sharpe_L2'], 9)}"
              f"{inv_s:>5}"
              f"{_fmt_num(r['d9_d10_delta_mom252'], 11)}")

    print()
    # Anchor verdicts vs M0 reference (from Step 3 handcrafted_no_overlay)
    print(f"  REFERENCE: M0 step3 handcrafted_no_overlay CAGR = {M0_REFERENCE_CAGR:+.1%}")
    print("  KEY CHECKS:")
    base_m1 = comp.loc[comp["variant"] == "V1_M1_raw"].iloc[0] if (comp["variant"] == "V1_M1_raw").any() else None
    if base_m1 is not None:
        m1_turn = base_m1["avg_turnover"]
        m1_cagr60 = base_m1["net_cagr_60bps"]
        m1_mono = base_m1["spearman_monotonicity_L2"]
        for _, r in comp.iterrows():
            if r["variant"] == "V1_M1_raw":
                continue
            if r["variant"] == "V0_M0_baseline":
                continue
            cut_turn = (m1_turn - r["avg_turnover"]) if pd.notna(r["avg_turnover"]) and pd.notna(m1_turn) else np.nan
            kept_mono = pd.notna(r["spearman_monotonicity_L2"]) and r["spearman_monotonicity_L2"] >= 0.80
            beat_m0 = pd.notna(r["net_cagr_60bps"]) and r["net_cagr_60bps"] >= M0_REFERENCE_CAGR
            print(f"    {r['variant']:<26}  "
                  f"turn_vs_M1_raw={_fmt_pct(cut_turn, 7)}  "
                  f"ρ_mono≥0.80={'✓' if kept_mono else '✗'}  "
                  f"net@60≥M0={'✓' if beat_m0 else '✗'}")
    print()
    print(f"  reports written to: {reports_dir}/  (elapsed {time.time() - t0:.1f}s)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--preds-m0", default=PREDS_M0)
    p.add_argument("--preds-m1", default=PREDS_M1)
    p.add_argument("--labels",   default=LABELS)
    p.add_argument("--features", default=FEATURES)
    p.add_argument("--top-n",    type=int, default=20)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        run(args)
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        raise
