"""
Step 2 runner — features.

Reads the universe panel (Step 0) + label panel (Step 1) + OHLCV panel +
XU100, computes the v1 feature family (6 blocks, 20 features), writes the
feature panel + manifest + redundancy + IC-sanity reports.

Produces (output/nyxmomentum/reports/):
  step2_features.parquet                 — long-format (ticker, rd, eligible, *features)
  step2_feature_manifest.csv             — machine-readable time contract
  step2_feature_coverage.csv             — per-feature missingness + stats
  step2_feature_correlation.csv          — full Spearman matrix
  step2_feature_correlation_flags.csv    — |r| ≥ 0.85 pairs
  step2_feature_ic_summary.csv           — SANITY ONLY (do not select on)
  step2_run_meta.json

Inputs:
  output/ohlcv_6y.parquet
  output/xu100_cache.parquet
  output/nyxmomentum/reports/step0_universe_panel.parquet
  output/nyxmomentum/reports/step1_labels.parquet

Usage:
  python run_nyxmomentum_step2.py
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import pandas as pd

from nyxmomentum.config import CONFIG
from nyxmomentum.features import (
    FEATURE_COLUMNS,
    FEATURE_BLOCKS,
    FEATURE_COLUMN_ROLES,
    build_feature_panel,
    feature_manifest,
    feature_coverage_report,
    feature_correlation_matrix,
    feature_high_corr_pairs,
    feature_ic_summary,
)
from nyxmomentum.utils import ensure_dir, save_json


DEFAULT_OHLCV    = "output/ohlcv_6y.parquet"
DEFAULT_XU100    = "output/xu100_cache.parquet"
DEFAULT_UNIVERSE = "output/nyxmomentum/reports/step0_universe_panel.parquet"
DEFAULT_LABELS   = "output/nyxmomentum/reports/step1_labels.parquet"


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


def run(args: argparse.Namespace) -> None:
    t0 = time.time()
    reports_dir = ensure_dir(CONFIG.paths.reports)

    print("[1/5] Loading inputs …")
    panel = load_panel(args.ohlcv)
    xu100 = load_xu100(args.xu100)
    uni = pd.read_parquet(args.universe)
    labels = pd.read_parquet(args.labels)
    print(f"  tickers={len(panel)}  universe_rows={len(uni):,}  label_rows={len(labels):,}")

    print("[2/5] Computing features …")
    t_f = time.time()
    features = build_feature_panel(uni, panel, xu100, CONFIG.feature)
    elapsed_f = time.time() - t_f
    print(f"  features: {len(features):,} rows × {len(FEATURE_COLUMNS)} cols "
          f"in {elapsed_f:.1f}s")
    features.to_parquet(os.path.join(reports_dir, "step2_features.parquet"))

    # Manifest (time contract table) — rule #4
    manifest = feature_manifest()
    manifest.to_csv(os.path.join(reports_dir, "step2_feature_manifest.csv"), index=False)

    # Redundancy — compute over eligible rows only (matches overlay convention)
    eligible = (
        features.loc[features["eligible"].astype(bool)]
        if "eligible" in features.columns else features
    )
    print(f"  eligible rows for reports: {len(eligible):,}")

    print("[3/5] Coverage + variance …")
    coverage = feature_coverage_report(eligible)
    coverage.to_csv(
        os.path.join(reports_dir, "step2_feature_coverage.csv"), index=False
    )

    print("[4/5] Pairwise correlation …")
    corr = feature_correlation_matrix(eligible, method="spearman")
    corr.round(4).to_csv(os.path.join(reports_dir, "step2_feature_correlation.csv"))
    high_pairs = feature_high_corr_pairs(corr, threshold=0.85)
    high_pairs.to_csv(
        os.path.join(reports_dir, "step2_feature_correlation_flags.csv"),
        index=False,
    )

    print("[5/5] IC summary (SANITY ONLY — not for feature selection) …")
    ic = feature_ic_summary(eligible, labels)
    ic.to_csv(os.path.join(reports_dir, "step2_feature_ic_summary.csv"), index=False)

    # Meta
    run_meta = {
        "produced_at": pd.Timestamp.utcnow().isoformat(),
        "n_tickers": len(panel),
        "n_features": len(FEATURE_COLUMNS),
        "blocks": list(FEATURE_BLOCKS),
        "n_rows": int(len(features)),
        "n_eligible_rows": int(len(eligible)),
        "high_corr_pairs_n": int(len(high_pairs)),
        "features_with_low_coverage": coverage.loc[
            coverage["coverage"] < 0.95, "feature"
        ].tolist(),
        "features_near_zero_variance": coverage.loc[
            coverage["near_zero_variance"], "feature"
        ].tolist(),
        "role_breakdown": {
            role: sum(1 for r in FEATURE_COLUMN_ROLES.values() if r == role)
            for role in set(FEATURE_COLUMN_ROLES.values())
        },
        "elapsed_sec": time.time() - t0,
    }
    save_json(os.path.join(reports_dir, "step2_run_meta.json"), run_meta)

    # Console summary
    print()
    print("══ nyxmomentum Step 2 ══")
    print(f"  features:              {len(FEATURE_COLUMNS)} across "
          f"{len(FEATURE_BLOCKS)} blocks ({', '.join(FEATURE_BLOCKS)})")
    print(f"  role breakdown:        {run_meta['role_breakdown']}")
    print()
    print("  block → features:")
    for blk in FEATURE_BLOCKS:
        bfeats = manifest.loc[manifest["block"] == blk, "feature"].tolist()
        print(f"    {blk:<18} {', '.join(bfeats)}")
    print()
    print("  coverage (eligible rows):")
    for _, r in coverage.iterrows():
        flag = "⚠" if (r["coverage"] < 0.95 or bool(r["near_zero_variance"])) else " "
        mean_txt = f"{r['mean']:+.4f}" if r["mean"] is not None else "   NA"
        std_txt = f"{r['std']:.4f}"  if r["std"]  is not None else "  NA"
        print(f"   {flag} {r['feature']:<34} cov={r['coverage']:.0%}  "
              f"mean={mean_txt}  std={std_txt}")
    print()
    print(f"  redundancy (|Spearman r| ≥ 0.85 pairs): {len(high_pairs)}")
    for _, p in high_pairs.head(10).iterrows():
        print(f"    {p['feature_a']:<30} ↔ {p['feature_b']:<30}  "
              f"r={p['corr']:+.2f}")
    if high_pairs.empty:
        print("    (none — feature set is not mutually collinear at 0.85)")
    print()
    print("  ⚠ IC vs L2 (SANITY ONLY — DO NOT select features on this):")
    ic_valid = ic.dropna(subset=["mean_ic"]).copy()
    if not ic_valid.empty:
        ic_valid["abs_ic"] = ic_valid["mean_ic"].abs()
        ic_valid = ic_valid.sort_values("abs_ic", ascending=False)
        for _, r in ic_valid.iterrows():
            print(f"    {r['feature']:<32} "
                  f"mean_IC={r['mean_ic']:+.4f}  "
                  f"t={r['ic_t']:+.2f}  "
                  f"hit={r['hit_rate']:.0%}  n_dates={int(r['n_dates'])}")
    print()
    print(f"  reports written to: {reports_dir}/")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ohlcv",    default=DEFAULT_OHLCV)
    p.add_argument("--xu100",    default=DEFAULT_XU100)
    p.add_argument("--universe", default=DEFAULT_UNIVERSE)
    p.add_argument("--labels",   default=DEFAULT_LABELS)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        run(args)
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        raise
