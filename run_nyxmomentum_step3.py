"""
Step 3 runner — pre-ML baselines.

Per the locked spec (rule #6): before any ML model is trained, separate
"does the overlay help?" from "does the feature set help?" by running four
variants end-to-end with identical portfolio construction:

  A. classic_no_overlay       — 12-1 momentum,       overlay OFF
  B. classic_with_overlay     — 12-1 momentum,       overlay ON
  C. handcrafted_no_overlay   — handcrafted blend,   overlay OFF
  D. handcrafted_with_overlay — handcrafted blend,   overlay ON

Comparison:
  B − A  →  overlay contribution on the classic ranker
  C − A  →  feature-set contribution (more blocks than just momentum)
  D − C  →  overlay contribution on the richer ranker
  D − A  →  combined lift

Produces (output/nyxmomentum/reports/):
  step3_portfolio_holdings.parquet   — per (variant, date, ticker) holdings
  step3_portfolio_returns.csv        — per (variant, date) portfolio P&L + context
  step3_summary.csv                  — one row per variant (Sharpe, CAGR, DD, …)
  step3_equity_curves.csv            — cumulative equity per variant
  step3_run_meta.json

Inputs:
  output/nyxmomentum/reports/step0_universe_panel_extended.parquet
  output/nyxmomentum/reports/step1_labels.parquet
  output/nyxmomentum/reports/step2_features.parquet

Usage:
  python run_nyxmomentum_step3.py
  python run_nyxmomentum_step3.py --top-n 30
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import pandas as pd

from nyxmomentum.config import CONFIG, EXECUTION_COLUMN_ROLES
from nyxmomentum.baselines import evaluate_all, default_variants
from nyxmomentum.utils import ensure_dir, save_json


DEFAULT_EXTENDED = "output/nyxmomentum/reports/step0_universe_panel_extended.parquet"
DEFAULT_LABELS   = "output/nyxmomentum/reports/step1_labels.parquet"
DEFAULT_FEATURES = "output/nyxmomentum/reports/step2_features.parquet"


def run(args: argparse.Namespace) -> None:
    t0 = time.time()
    reports_dir = ensure_dir(CONFIG.paths.reports)

    print("[1/4] Loading inputs …")
    extended = pd.read_parquet(args.extended)
    labels = pd.read_parquet(args.labels)
    features = pd.read_parquet(args.features)
    print(f"  extended={len(extended):,}  labels={len(labels):,}  features={len(features):,}")

    # Sanity: feature panel must carry eligible flag + (ticker, rebalance_date)
    for required in (["ticker", "rebalance_date", "eligible"],):
        missing = set(required) - set(features.columns)
        if missing:
            raise ValueError(f"features missing columns: {missing}")

    # Build proxies frame: ex-ante columns only (hard leakage guard upstream).
    ex_ante_cols = [c for c, r in EXECUTION_COLUMN_ROLES.items() if r == "ex_ante"]
    keep = ["ticker", "rebalance_date", *ex_ante_cols]
    missing_proxy = [c for c in keep if c not in extended.columns]
    if missing_proxy:
        raise ValueError(f"extended panel missing proxy columns: {missing_proxy}")
    proxies = extended[keep].copy()

    print("[2/4] Evaluating variants …")
    t_e = time.time()
    out = evaluate_all(
        features=features,
        labels=labels,
        proxies=proxies,
        portfolio_cfg=CONFIG.portfolio,
        top_n=args.top_n,
        variants=default_variants(),
    )
    print(f"  {len(out['summary'])} variants in {time.time() - t_e:.1f}s")

    print("[3/4] Writing artifacts …")
    out["portfolios"].to_parquet(
        os.path.join(reports_dir, "step3_portfolio_holdings.parquet")
    )
    out["returns"].to_csv(
        os.path.join(reports_dir, "step3_portfolio_returns.csv"), index=False
    )
    out["summary"].to_csv(
        os.path.join(reports_dir, "step3_summary.csv"), index=False
    )
    out["equity"].to_csv(
        os.path.join(reports_dir, "step3_equity_curves.csv"), index=False
    )

    meta = {
        "produced_at": pd.Timestamp.utcnow().isoformat(),
        "top_n": args.top_n,
        "n_variants": int(len(out["summary"])),
        "n_holdings_rows": int(len(out["portfolios"])),
        "overlay_weights": dict(CONFIG.portfolio.risk_overlay_weights),
        "overlay_strength": float(CONFIG.portfolio.risk_overlay_strength),
        "elapsed_sec": time.time() - t0,
    }
    save_json(os.path.join(reports_dir, "step3_run_meta.json"), meta)

    # ── Console summary ───────────────────────────────────────────────────
    print("[4/4] Summary …")
    print()
    print("══ nyxmomentum Step 3 — pre-ML baselines ══")
    print(f"  top_n per rebalance:      {args.top_n}")
    print(f"  overlay strength:         {CONFIG.portfolio.risk_overlay_strength}")
    print(f"  overlay weights:          "
          f"{', '.join(f'{k}={v:.2f}' for k, v in CONFIG.portfolio.risk_overlay_weights.items())}")
    print()

    s = out["summary"].set_index("variant")
    cols_order = [
        "n_rebalances", "cagr", "sharpe_annualized", "max_drawdown",
        "hit_rate", "mean_monthly_return", "std_monthly_return",
        "mean_excess_xu100", "mean_excess_median", "avg_turnover",
    ]
    cols_in = [c for c in cols_order if c in s.columns]
    print("  per-variant metrics:")
    # Header
    print(f"    {'variant':<28}  {'CAGR':>7} {'Sharpe':>7} {'MaxDD':>7} "
          f"{'Hit':>5} {'μMo':>7} {'σMo':>7} "
          f"{'exXU':>7} {'exMed':>7} {'Trn':>6}")
    for var_name, row in s.iterrows():
        if row.get("n_rebalances", 0) == 0:
            print(f"    {var_name:<28}  (no valid rebalances)")
            continue
        def fmt(k, width, pct=False):
            v = row.get(k)
            if v is None or pd.isna(v):
                return " " * width
            if pct:
                return f"{v:+{width - 1}.1%}"
            return f"{v:+{width - 1}.2f}"
        print(f"    {var_name:<28}  "
              f"{fmt('cagr', 7, pct=True)} "
              f"{fmt('sharpe_annualized', 7)} "
              f"{fmt('max_drawdown', 7, pct=True)} "
              f"{fmt('hit_rate', 5, pct=True)} "
              f"{fmt('mean_monthly_return', 7, pct=True)} "
              f"{fmt('std_monthly_return', 7, pct=True)} "
              f"{fmt('mean_excess_xu100', 7, pct=True)} "
              f"{fmt('mean_excess_median', 7, pct=True)} "
              f"{fmt('avg_turnover', 6, pct=True)}")

    print()
    print("  attribution (variant − baseline, CAGR):")
    try:
        base_cagr = float(s.loc["classic_no_overlay", "cagr"])
        for var in ["classic_with_overlay", "handcrafted_no_overlay", "handcrafted_with_overlay"]:
            if var in s.index and "cagr" in s.columns:
                diff = float(s.loc[var, "cagr"]) - base_cagr
                print(f"    {var:<28}  ΔCAGR = {diff:+.1%}")
    except (KeyError, TypeError):
        print("    (attribution skipped — missing baseline)")

    print()
    print(f"  reports written to: {reports_dir}/")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--extended", default=DEFAULT_EXTENDED)
    p.add_argument("--labels",   default=DEFAULT_LABELS)
    p.add_argument("--features", default=DEFAULT_FEATURES)
    p.add_argument("--top-n",    type=int, default=20,
                   help="Top-N eligible names per rebalance (default: 20)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        run(args)
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        raise
