"""
Step 3 audit runner — skeptical checks on the Step 3 baselines before Step 4.

Motivation: CAGR 76-86% / Sharpe 1.76-2.12 / MaxDD -24 to -29% on 2022-2026
BIST is suspicious-strong. Before training ML, confirm each channel that
could be inflating the number:

  A. Cost-adjusted: do gross numbers survive 20/40/60/100bps RT?
  B. Subperiod: is it one bull leg (2022-2023) or durable across (2024-2026)?
  C. Concentration: are we really holding 20 unique names, or rotating 30 total?
  D. Decile monotonicity: does the full cross-section stack, or is top-tail doing all the work?
  E. Block contribution: which block of the handcrafted composite is the engine?
  F. Suspicious ticker-month contributions vs DQ flags: did the biggest wins have a bad Low print?

Produces (output/nyxmomentum/reports/):
  step3_audit_costed_summary.csv
  step3_audit_subperiod.csv
  step3_audit_rolling_12m.csv
  step3_audit_concentration.csv
  step3_audit_decile_classic.csv
  step3_audit_decile_handcrafted.csv
  step3_audit_block_contribution.csv
  step3_audit_top_contrib_<variant>.csv   (one per variant)
  step3_audit_run_meta.json

Inputs:
  output/nyxmomentum/reports/step3_portfolio_holdings.parquet
  output/nyxmomentum/reports/step3_portfolio_returns.csv
  output/nyxmomentum/reports/step2_features.parquet
  output/nyxmomentum/reports/step1_labels.parquet
  output/nyxmomentum/reports/step0_dq_suspicious_lows.csv

Usage:
  python run_nyxmomentum_step3_audit.py
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import pandas as pd

from nyxmomentum.config import CONFIG
from nyxmomentum.baselines import (
    classic_momentum_score,
    handcrafted_composite_score,
)
from nyxmomentum.baselines_audit import (
    cost_adjusted_summary,
    subperiod_summary,
    rolling_12m_return,
    concentration_report,
    decile_performance,
    block_contribution_report,
    top_contribution_audit,
)
from nyxmomentum.utils import ensure_dir, save_json


DEFAULT_HOLDINGS = "output/nyxmomentum/reports/step3_portfolio_holdings.parquet"
DEFAULT_RETURNS  = "output/nyxmomentum/reports/step3_portfolio_returns.csv"
DEFAULT_FEATURES = "output/nyxmomentum/reports/step2_features.parquet"
DEFAULT_LABELS   = "output/nyxmomentum/reports/step1_labels.parquet"
DEFAULT_DQ       = "output/nyxmomentum/reports/step0_dq_suspicious_lows.csv"


def run(args: argparse.Namespace) -> None:
    t0 = time.time()
    reports_dir = ensure_dir(CONFIG.paths.reports)

    print("[1/7] Loading …")
    holdings = pd.read_parquet(args.holdings)
    returns = pd.read_csv(args.returns, parse_dates=["rebalance_date"])
    features = pd.read_parquet(args.features)
    labels = pd.read_parquet(args.labels)
    try:
        dq = pd.read_csv(args.dq, parse_dates=["date"])
    except FileNotFoundError:
        dq = pd.DataFrame()
    print(f"  holdings={len(holdings):,}  returns={len(returns):,}  "
          f"features={len(features):,}  labels={len(labels):,}  dq={len(dq):,}")

    # ── A. Cost-adjusted ──────────────────────────────────────────────────
    print("[2/7] (A) Cost-adjusted summary …")
    costed = cost_adjusted_summary(returns, bps_round_trip_list=(0, 20, 40, 60, 100))
    costed.to_csv(os.path.join(reports_dir, "step3_audit_costed_summary.csv"), index=False)

    # ── B. Subperiod + rolling ────────────────────────────────────────────
    print("[3/7] (B) Subperiod split + rolling 12M …")
    sub = subperiod_summary(returns, split_date=args.split_date)
    sub.to_csv(os.path.join(reports_dir, "step3_audit_subperiod.csv"), index=False)
    roll = rolling_12m_return(returns)
    roll.to_csv(os.path.join(reports_dir, "step3_audit_rolling_12m.csv"), index=False)

    # ── C. Concentration ──────────────────────────────────────────────────
    print("[4/7] (C) Breadth / concentration …")
    conc = concentration_report(holdings)
    conc.to_csv(os.path.join(reports_dir, "step3_audit_concentration.csv"), index=False)

    # ── D. Decile monotonicity ────────────────────────────────────────────
    print("[5/7] (D) Decile monotonicity …")
    decile_classic = decile_performance(
        features=features, labels=labels,
        score_fn=classic_momentum_score, score_name="classic_mom_252_skip_21",
        n_buckets=10,
    )
    decile_classic.to_csv(
        os.path.join(reports_dir, "step3_audit_decile_classic.csv"), index=False
    )
    decile_handcrafted = decile_performance(
        features=features, labels=labels,
        score_fn=handcrafted_composite_score, score_name="handcrafted_composite",
        n_buckets=10,
    )
    decile_handcrafted.to_csv(
        os.path.join(reports_dir, "step3_audit_decile_handcrafted.csv"), index=False
    )

    # ── E. Block contribution ─────────────────────────────────────────────
    print("[6/7] (E) Block contribution decomposition …")
    blocks = block_contribution_report(features=features, labels=labels, n_buckets=10)
    blocks.to_csv(
        os.path.join(reports_dir, "step3_audit_block_contribution.csv"), index=False
    )

    # ── F. Top contribution audit ─────────────────────────────────────────
    print("[7/7] (F) Top-contribution + DQ cross-ref …")
    top_tables: dict[str, pd.DataFrame] = {}
    for var_name, gf in holdings.groupby("variant", sort=False):
        top = top_contribution_audit(
            portfolios_for_variant=gf,
            dq_flagged=dq if len(dq) else None,
            n_top=30,
        )
        top.insert(0, "variant", var_name)
        safe = var_name.replace("/", "_")
        top.to_csv(
            os.path.join(reports_dir, f"step3_audit_top_contrib_{safe}.csv"),
            index=False,
        )
        top_tables[var_name] = top

    meta = {
        "produced_at": pd.Timestamp.utcnow().isoformat(),
        "split_date": args.split_date,
        "bps_round_trip_list": [0, 20, 40, 60, 100],
        "n_variants": int(returns["variant"].nunique()),
        "elapsed_sec": time.time() - t0,
    }
    save_json(os.path.join(reports_dir, "step3_audit_run_meta.json"), meta)

    # ── Console summary ───────────────────────────────────────────────────
    print()
    print("══ nyxmomentum Step 3 audit ══")

    # A
    print()
    print("  (A) COST-ADJUSTED SUMMARY — CAGR_net by bps round-trip:")
    piv = costed.pivot_table(
        index="variant", columns="bps_round_trip", values="cagr_net"
    ).reindex(columns=[0, 20, 40, 60, 100])
    print(piv.map(lambda v: f"{v:+.1%}" if pd.notna(v) else "  —  ").to_string())
    print()
    print("  (A) SHARPE_net by bps round-trip:")
    piv_s = costed.pivot_table(
        index="variant", columns="bps_round_trip", values="sharpe_annualized_net"
    ).reindex(columns=[0, 20, 40, 60, 100])
    print(piv_s.map(lambda v: f"{v:+.2f}" if pd.notna(v) else "  —  ").to_string())

    # B
    print()
    print(f"  (B) SUBPERIOD SPLIT at {args.split_date} (pre / post):")
    for var_name, g in sub.groupby("variant", sort=False):
        g = g.set_index("period")
        def fmt(row, k, pct=False):
            v = row.get(k)
            if v is None or pd.isna(v):
                return "    —  "
            return f"{v:+.1%}" if pct else f"{v:+.2f}"
        for period, r in g.iterrows():
            print(f"    {var_name:<28} {period:<14} "
                  f"CAGR={fmt(r,'cagr',True)}  "
                  f"Shp={fmt(r,'sharpe_annualized')}  "
                  f"DD={fmt(r,'max_drawdown',True)}  "
                  f"Hit={fmt(r,'hit_rate',True)}  "
                  f"N={int(r.get('n_rebalances', 0))}")

    # C
    print()
    print("  (C) CONCENTRATION:")
    for _, r in conc.iterrows():
        print(f"    {r['variant']:<28}  "
              f"unique={r['unique_tickers']:>3}  "
              f"mean_occ={r['mean_occurrences_per_ticker']:.1f}  "
              f"top10_share={r['top10_share_of_slots']:.1%}  "
              f"HHI={r['herfindahl_index']:.3f}")
        print(f"      top10: {r['top10_names']}")

    # D
    print()
    print("  (D) DECILE MONOTONICITY — mean per-date decile return:")
    for name, tbl in [("classic", decile_classic), ("handcrafted", decile_handcrafted)]:
        print(f"    [{name}]")
        d = tbl.loc[tbl["_decile"] > 0].sort_values("_decile")
        for _, r in d.iterrows():
            print(f"       D{int(r['_decile']):>2}  "
                  f"mean={r['mean_return']:+.2%}  "
                  f"σ={r['std_return']:.2%}  "
                  f"hit={r['hit_rate']:.0%}  "
                  f"Shp={r['sharpe_annualized']:+.2f}  "
                  f"N={int(r['n_dates'])}")
        tmb = tbl.loc[tbl["_decile"] == -1]
        if len(tmb):
            r = tmb.iloc[0]
            print(f"       D10-D1  mean={r['mean_return']:+.2%}  "
                  f"σ={r['std_return']:.2%}  "
                  f"hit={r['hit_rate']:.0%}  "
                  f"Shp={r['sharpe_annualized']:+.2f}  "
                  f"N={int(r['n_dates'])}")

    # E
    print()
    print("  (E) HANDCRAFTED BLOCK CONTRIBUTION:")
    print(f"    {'feature':<32} {'block':<18} {'w':>6} {'share':>6} {'corr':>6} "
          f"{'D10-D1':>8} {'Shp':>6} {'aligned':>7}")
    for _, r in blocks.iterrows():
        aligned = "yes" if r["aligned_with_weight_sign"] else "NO"
        d1 = r["single_feature_D10_minus_D1_return"]
        shp = r["single_feature_D10_minus_D1_sharpe"]
        d1_fmt = f"{d1:+.2%}" if pd.notna(d1) else "   —  "
        shp_fmt = f"{shp:+.2f}" if pd.notna(shp) else "   — "
        print(f"    {r['feature']:<32} {r['block']:<18} "
              f"{r['weight']:+.2f} {r['weight_share']:.2f} "
              f"{r['corr_with_full_composite']:+.2f} "
              f"{d1_fmt:>8} {shp_fmt:>6} {aligned:>7}")

    # F
    print()
    print("  (F) TOP-30 CONTRIBUTION AUDIT — DQ-flagged ticker-months:")
    total_flagged = 0
    for var_name, top in top_tables.items():
        n_flag = int(top["dq_flagged"].sum())
        total_flagged += n_flag
        print(f"    {var_name:<28}  "
              f"top30_flagged={n_flag}/30  "
              f"(best single={top['contribution'].max():+.3%}, "
              f"top3 mean={top.head(3)['contribution'].mean():+.3%})")
        if n_flag:
            flagged = top.loc[top["dq_flagged"]]
            for _, r in flagged.head(5).iterrows():
                print(f"      ⚠ {r['ticker']:<6} "
                      f"rd={pd.Timestamp(r['rebalance_date']).date()}  "
                      f"ret={r['l1_forward_return']:+.2%}  "
                      f"dq_dates={r['dq_hit_dates']}")

    print()
    print(f"  reports written to: {reports_dir}/  (elapsed {time.time() - t0:.1f}s)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--holdings", default=DEFAULT_HOLDINGS)
    p.add_argument("--returns",  default=DEFAULT_RETURNS)
    p.add_argument("--features", default=DEFAULT_FEATURES)
    p.add_argument("--labels",   default=DEFAULT_LABELS)
    p.add_argument("--dq",       default=DEFAULT_DQ)
    p.add_argument("--split-date", default="2024-01-01",
                   help="Subperiod boundary (default: 2024-01-01)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        run(args)
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        raise
