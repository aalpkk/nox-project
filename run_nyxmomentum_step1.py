"""
Step 1 runner — labels.

Reads the extended universe panel from Step 0 + the BIST + XU100 data, computes
the full label family, and writes the label panel + distribution reports.

Produces (output/nyxmomentum/reports/):
  step1_labels.parquet                   — long-format label panel
  step1_label_distribution.csv           — per-rebalance numeric summary
  step1_label_role_manifest.csv          — train_target vs diagnostic vs context
  step1_label_global_stats.json          — global stats for quick sanity
  step1_run_meta.json

Inputs:
  output/ohlcv_6y.parquet
  output/xu100_cache.parquet
  output/nyxmomentum/reports/step0_universe_panel_extended.parquet
  output/nyxmomentum/reports/step0_rebalance_calendar.csv

Usage:
  python run_nyxmomentum_step1.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict

import numpy as np
import pandas as pd

from nyxmomentum.config import CONFIG, LABEL_COLUMN_ROLES
from nyxmomentum.labels import (
    compute_labels,
    label_distribution_report,
    label_role_manifest,
    l2_decomposition_report,
    LABEL_COLUMNS,
)
from nyxmomentum.utils import ensure_dir, save_json


DEFAULT_OHLCV = "output/ohlcv_6y.parquet"
DEFAULT_XU100 = "output/xu100_cache.parquet"
DEFAULT_UNIVERSE = "output/nyxmomentum/reports/step0_universe_panel_extended.parquet"
DEFAULT_REBALANCE = "output/nyxmomentum/reports/step0_rebalance_calendar.csv"


def load_panel(path: str) -> dict[str, pd.DataFrame]:
    raw = pd.read_parquet(path)
    if raw.index.tz is not None:
        raw = raw.tz_localize(None)
    out: dict[str, pd.DataFrame] = {}
    for ticker, g in raw.groupby("ticker", sort=False):
        df = g.drop(columns="ticker").sort_index()
        df = df[~df.index.duplicated(keep="last")]
        out[ticker] = df
    return out


def load_xu100(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if df.index.tz is not None:
        df = df.tz_localize(None)
    return df.sort_index()


def load_rebalance_calendar(path: str) -> pd.DatetimeIndex:
    cal = pd.read_csv(path, parse_dates=["rebalance_date"])
    return pd.DatetimeIndex(cal["rebalance_date"].sort_values().values)


def run(args: argparse.Namespace) -> None:
    t0 = time.time()
    reports_dir = ensure_dir(CONFIG.paths.reports)

    print("[1/4] Loading inputs …")
    panel = load_panel(args.ohlcv)
    xu100 = load_xu100(args.xu100)
    rebalance_dates = load_rebalance_calendar(args.rebalance)
    uni_ext = pd.read_parquet(args.universe)
    # eligible-subset for label computation
    eligible = uni_ext[["ticker", "rebalance_date", "eligible"]].copy()
    print(f"  tickers={len(panel)}  rebalances={len(rebalance_dates)}  "
          f"universe_rows={len(uni_ext):,}  eligible_rows={int(eligible['eligible'].sum()):,}")

    print("[2/4] Computing labels …")
    t_lab = time.time()
    labels = compute_labels(
        panel=panel,
        rebalance_dates=rebalance_dates,
        xu100=xu100,
        eligible_panel=eligible,
        config=CONFIG.label,
    )
    print(f"  labels: {len(labels):,} rows in {time.time() - t_lab:.1f}s")

    if labels.empty:
        print("  WARN: label frame is empty. Aborting reporting.", file=sys.stderr)
        return

    # Persist
    labels.to_parquet(os.path.join(reports_dir, "step1_labels.parquet"))

    # Per-date distribution
    dist = label_distribution_report(labels)
    dist.to_csv(os.path.join(reports_dir, "step1_label_distribution.csv"), index=False)

    # Role manifest
    roles = label_role_manifest()
    roles.to_csv(os.path.join(reports_dir, "step1_label_role_manifest.csv"), index=False)

    # Global stats
    global_stats = {}
    for col in ["l1_forward_return", "l2_excess_vs_universe_median",
                "l3_outperform_binary", "l4_quality_adjusted_return",
                "l5_drawdown_aware_binary", "forward_max_dd",
                "forward_max_dd_intraperiod",
                "xu100_excess_return"]:
        if col in labels.columns:
            s = pd.to_numeric(labels[col], errors="coerce").dropna()
            if len(s):
                global_stats[col] = {
                    "n": int(len(s)),
                    "mean": float(s.mean()),
                    "median": float(s.median()),
                    "std": float(s.std()),
                    "q05": float(s.quantile(0.05)),
                    "q25": float(s.quantile(0.25)),
                    "q75": float(s.quantile(0.75)),
                    "q95": float(s.quantile(0.95)),
                    "min": float(s.min()),
                    "max": float(s.max()),
                    "role": LABEL_COLUMN_ROLES.get(col, "unknown"),
                }

    save_json(os.path.join(reports_dir, "step1_label_global_stats.json"), global_stats)

    # L2 decomposition — is the global mean real alpha or a date/size artifact?
    l2_decomp = l2_decomposition_report(labels)
    save_json(os.path.join(reports_dir, "step1_l2_decomposition.json"), l2_decomp)

    print("[3/4] Sanity checks …")
    # 1. Universe median should be ~0 when averaged across rows on a given date
    med_check = labels.groupby("rebalance_date")["l2_excess_vs_universe_median"].median()
    print(f"  median-of-L2 per rebalance_date: median={med_check.median():.6f}  "
          f"max_abs={med_check.abs().max():.6f}   (≈0 expected by construction)")

    # 2. L1 and universe_median consistency
    check = (labels["l1_forward_return"] - labels["universe_median_return"]
             - labels["l2_excess_vs_universe_median"]).abs().max()
    print(f"  L1 − median − L2 residual max_abs: {check:.2e}   (≈0 expected)")

    # 3. Forward max_dd non-positive (both variants)
    pos_dd = (labels["forward_max_dd"] > 1e-9).sum()
    print(f"  forward_max_dd positive count: {pos_dd}   (0 expected)")
    if "forward_max_dd_intraperiod" in labels.columns:
        pos_dd_i = (labels["forward_max_dd_intraperiod"] > 1e-9).sum()
        print(f"  forward_max_dd_intraperiod positive count: {pos_dd_i}   (0 expected)")
        # 3b. Intraperiod dd must be ≤ close-only dd (more or equally negative)
        gap = (labels["forward_max_dd_intraperiod"] - labels["forward_max_dd"]).dropna()
        bad = int((gap > 1e-9).sum())
        print(f"  intraperiod ≤ close-only violations: {bad}   (0 expected)")
        print(f"  dd gap (close-only − intraperiod): "
              f"mean={(labels['forward_max_dd'] - labels['forward_max_dd_intraperiod']).mean():+.4f}  "
              f"median={(labels['forward_max_dd'] - labels['forward_max_dd_intraperiod']).median():+.4f}  "
              f"max={(labels['forward_max_dd'] - labels['forward_max_dd_intraperiod']).max():+.4f}")

    # 4. Partial holding share
    partial = labels["partial_holding"].mean() if "partial_holding" in labels.columns else 0.0
    print(f"  partial_holding share: {partial:.1%}")

    # 5. Last rebalance excluded?
    max_reb_with_labels = labels["rebalance_date"].max()
    true_last = rebalance_dates[-1]
    print(f"  last rebalance in calendar: {pd.Timestamp(true_last).date()}")
    print(f"  last rebalance in labels:   {pd.Timestamp(max_reb_with_labels).date()}  "
          f"(must be < calendar last)")
    assert max_reb_with_labels < true_last, "Last rebalance leaked into labels!"

    print("[4/4] Writing meta + console summary …")
    run_meta = {
        "produced_at": pd.Timestamp.utcnow().isoformat(),
        "label_config": asdict(CONFIG.label),
        "n_tickers": len(panel),
        "n_rebalances": int(len(rebalance_dates)),
        "n_label_rows": int(len(labels)),
        "train_target": CONFIG.label.train_target,
        "elapsed_sec": time.time() - t0,
    }
    save_json(os.path.join(reports_dir, "step1_run_meta.json"), run_meta)

    print()
    print("══ nyxmomentum Step 1 ══")
    print(f"  label rows:            {len(labels):,}")
    print(f"  train target:          {CONFIG.label.train_target}")
    print(f"  entry / exit mode:     {CONFIG.label.entry_mode} → {CONFIG.label.exit_mode}")
    print()
    print("  ⚠ L5 (l5_drawdown_aware_binary) is a DIAGNOSTIC ONLY — never feed to")
    print("    a learner and never use to filter selection. It reads future DD.")
    print()
    print("  global label stats (train target first):")
    order = ["l2_excess_vs_universe_median", "l1_forward_return",
             "forward_max_dd", "forward_max_dd_intraperiod",
             "xu100_excess_return",
             "l3_outperform_binary", "l4_quality_adjusted_return",
             "l5_drawdown_aware_binary"]
    for col in order:
        if col not in global_stats:
            continue
        st = global_stats[col]
        tag = "◆ TRAIN" if st["role"] == "train_target" else "  diag "
        print(f"  {tag} {col:<34} "
              f"mean={st['mean']:+.4f}  med={st['median']:+.4f}  "
              f"std={st['std']:.4f}  n={st['n']:,}")
    print()
    if l2_decomp:
        print("  L2 mean decomposition (train target sanity):")
        print(f"    row-EW          : {l2_decomp['row_ew']:+.4f}   (naive global mean)")
        print(f"    date-EW         : {l2_decomp['date_ew']:+.4f}   (equal weight per rebalance_date)")
        print(f"    ticker-EW       : {l2_decomp['ticker_ew']:+.4f}   (equal weight per ticker)")
        sc = l2_decomp.get("size_spearman_corr_mean_vs_n")
        if sc is not None:
            print(f"    size corr (N × L2 mean): {sc:+.3f}   "
                  f"(|r|<0.3 ≈ size-neutral, |r|>0.5 = mean driven by universe size)")
        print("    by year:")
        for y, st in sorted(l2_decomp.get("by_year", {}).items()):
            print(f"      {y}   mean={st['mean']:+.4f}  std={st['std']:.4f}  n={st['count']:,}")
        dd = l2_decomp.get("date_mean_distribution", {})
        if dd:
            print(f"    per-date mean distribution: "
                  f"q05={dd['q05']:+.4f}  q50={dd['q50']:+.4f}  q95={dd['q95']:+.4f}")
    print()
    print(f"  reports written to: {reports_dir}/")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ohlcv", default=DEFAULT_OHLCV)
    p.add_argument("--xu100", default=DEFAULT_XU100)
    p.add_argument("--universe", default=DEFAULT_UNIVERSE)
    p.add_argument("--rebalance", default=DEFAULT_REBALANCE)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        run(args)
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        raise
