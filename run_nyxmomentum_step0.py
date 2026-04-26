"""
Step 0 + 0.5 runner — universe, rebalance calendar, ex-ante execution proxies,
and t+1 realized fillability diagnostics.

Produces (output/nyxmomentum/reports/):
  step0_rebalance_calendar.csv
  step0_universe_panel.parquet           — admittance panel (base)
  step0_universe_panel_extended.parquet  — + ex-ante proxies + realized diagnostics
  step0_universe_summary.csv
  step0_rejection_histogram.csv
  step0_proxy_coverage.csv               — per-proxy NaN/distribution summary
  step0_proxy_role_manifest.csv          — which columns may enter overlay
  step0_run_meta.json

Inputs:
  output/ohlcv_6y.parquet  — long-format BIST OHLCV panel (Date index + ticker col)
  output/xu100_cache.parquet — canonical trading calendar source

Usage:
  python run_nyxmomentum_step0.py
  python run_nyxmomentum_step0.py --start 2022-01-01 --frequency M
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, replace

import pandas as pd

from nyxmomentum.config import CONFIG, RebalanceConfig, UniverseConfig
from nyxmomentum.rebalance import build_rebalance_calendar, rebalance_summary
from nyxmomentum.universe import (
    build_universe_panel,
    universe_summary,
    reason_histogram,
)
from nyxmomentum.execution import (
    extend_universe_panel,
    proxy_coverage_report,
    column_role_manifest,
    ex_ante_correlation_matrix,
    ex_ante_high_corr_pairs,
)
from nyxmomentum.data_quality import scan_suspicious_lows, dq_summary
from nyxmomentum.utils import ensure_dir, save_json


DEFAULT_OHLCV = "output/ohlcv_6y.parquet"
DEFAULT_XU100 = "output/xu100_cache.parquet"


def load_panel(path: str, min_bars: int = 252) -> dict[str, pd.DataFrame]:
    """Convert long-format parquet to dict[ticker → per-ticker OHLCV frame]."""
    raw = pd.read_parquet(path)
    if "ticker" not in raw.columns:
        raise ValueError(f"{path} missing 'ticker' column; got {raw.columns.tolist()}")
    if not isinstance(raw.index, pd.DatetimeIndex):
        raise ValueError(f"{path} index must be DatetimeIndex; got {type(raw.index).__name__}")

    # Strip tz if present, normalize
    idx = raw.index
    if idx.tz is not None:
        raw = raw.tz_localize(None)

    out: dict[str, pd.DataFrame] = {}
    dropped = 0
    for ticker, g in raw.groupby("ticker", sort=False):
        df = g.drop(columns="ticker").sort_index()
        # Deduplicate any repeat timestamps, keep last
        df = df[~df.index.duplicated(keep="last")]
        if len(df) < min_bars:
            dropped += 1
            continue
        out[ticker] = df
    print(f"  loaded {len(out)} tickers from {path} ({dropped} dropped < {min_bars} bars)")
    return out


def load_trading_calendar(xu100_path: str) -> pd.DatetimeIndex:
    df = pd.read_parquet(xu100_path)
    if df.index.tz is not None:
        df = df.tz_localize(None)
    cal = df.index.sort_values().unique()
    print(f"  trading calendar: {len(cal)} days, {cal.min().date()} → {cal.max().date()}")
    return cal


def run(args: argparse.Namespace) -> None:
    t0 = time.time()

    reb_cfg = replace(
        CONFIG.rebalance,
        frequency=args.frequency,
        start_date=args.start,
        end_date=args.end,
    )
    uni_cfg = CONFIG.universe

    reports_dir = ensure_dir(CONFIG.paths.reports)

    print("[1/4] Loading data …")
    calendar = load_trading_calendar(args.xu100)
    panel = load_panel(args.ohlcv, min_bars=uni_cfg.min_history_days)

    print("[2/4] Building rebalance calendar …")
    anchors = build_rebalance_calendar(calendar, reb_cfg)
    anchors_summary = rebalance_summary(anchors)
    print(f"  anchors: {anchors_summary['count']} "
          f"({anchors_summary['first']} → {anchors_summary['last']}, "
          f"median gap {anchors_summary['median_gap_days']:.0f}d)")

    # Persist rebalance calendar
    pd.DataFrame({"rebalance_date": anchors}).to_csv(
        os.path.join(reports_dir, "step0_rebalance_calendar.csv"), index=False
    )

    print("[3/4] Running universe filter …")
    # Optional: trim very old anchors that pre-date the panel (faster)
    panel_first_dates = [df.index[0] for df in panel.values()]
    earliest_panel_date = min(panel_first_dates) if panel_first_dates else None
    if earliest_panel_date is not None:
        keep = anchors[anchors >= earliest_panel_date + pd.Timedelta(days=uni_cfg.min_history_days)]
        if len(keep) < len(anchors):
            print(f"  trimmed {len(anchors) - len(keep)} anchors before any ticker had min_history")
            anchors = keep

    t_uni = time.time()
    uni_panel = build_universe_panel(panel, anchors, uni_cfg)
    print(f"  built panel: {len(uni_panel):,} rows in {time.time() - t_uni:.1f}s")

    # Persist panel + summaries
    uni_panel.to_parquet(os.path.join(reports_dir, "step0_universe_panel.parquet"))

    summary = universe_summary(uni_panel)
    # Flatten top_rejections dict to JSON-encodable string for CSV
    summary_csv = summary.copy()
    summary_csv["top_rejections"] = summary_csv["top_rejections"].apply(
        lambda d: json.dumps(d, ensure_ascii=False)
    )
    summary_csv.to_csv(os.path.join(reports_dir, "step0_universe_summary.csv"), index=False)

    hist = reason_histogram(uni_panel)
    hist.to_csv(os.path.join(reports_dir, "step0_rejection_histogram.csv"), index=False)

    print("[4/5] Attaching ex-ante execution proxies + realized diagnostics …")
    t_px = time.time()
    extended = extend_universe_panel(uni_panel, panel, CONFIG.execution_proxy)
    extended.to_parquet(os.path.join(reports_dir, "step0_universe_panel_extended.parquet"))
    coverage = proxy_coverage_report(extended)
    coverage.to_csv(os.path.join(reports_dir, "step0_proxy_coverage.csv"), index=False)
    roles = column_role_manifest()
    roles.to_csv(os.path.join(reports_dir, "step0_proxy_role_manifest.csv"), index=False)

    # Pairwise correlation across ex-ante proxies (eligible rows only).
    # Flag |r| ≥ 0.7 pairs — would double-count in the overlay weighted sum.
    corr = ex_ante_correlation_matrix(extended, method="spearman")
    if not corr.empty:
        corr.round(4).to_csv(
            os.path.join(reports_dir, "step0_proxy_correlation.csv"),
            index=True,
        )
        high_pairs = ex_ante_high_corr_pairs(corr, threshold=0.7)
        high_pairs.to_csv(
            os.path.join(reports_dir, "step0_proxy_correlation_flags.csv"),
            index=False,
        )
    else:
        high_pairs = pd.DataFrame()

    print(f"  extended panel: +{len(extended.columns) - len(uni_panel.columns)} cols "
          f"in {time.time() - t_px:.1f}s")

    # Data-quality audit: suspicious Low prints that would corrupt intraperiod
    # DD and Parkinson vol. Surface, do not silently mask.
    print("  DQ: scanning suspicious Low prints …")
    dq_flagged = scan_suspicious_lows(panel)
    dq_flagged.to_csv(
        os.path.join(reports_dir, "step0_dq_suspicious_lows.csv"), index=False
    )
    dq_sum = dq_summary(dq_flagged)
    print(f"  DQ flagged: {dq_sum.get('n_flagged_rows', 0):,} rows "
          f"across {dq_sum.get('n_flagged_tickers', 0)} tickers")

    print("[5/5] Writing meta + console summary …")
    run_meta = {
        "produced_at": pd.Timestamp.utcnow().isoformat(),
        "inputs": {"ohlcv": args.ohlcv, "xu100": args.xu100},
        "rebalance_config": asdict(reb_cfg),
        "universe_config": asdict(uni_cfg),
        "tickers_loaded": len(panel),
        "rebalance_count": int(len(anchors)),
        "rebalance_summary": anchors_summary,
        "panel_rows": int(len(uni_panel)),
        "eligibility_rate_global": (
            float(uni_panel["eligible"].mean()) if len(uni_panel) else None
        ),
        "dq_summary": dq_sum,
        "elapsed_sec": time.time() - t0,
    }
    save_json(os.path.join(reports_dir, "step0_run_meta.json"), run_meta)

    # Console summary
    print()
    print("══ nyxmomentum Step 0 ══")
    print(f"  tickers:              {run_meta['tickers_loaded']}")
    print(f"  rebalance dates:      {run_meta['rebalance_count']}")
    print(f"  panel rows:           {run_meta['panel_rows']:,}")
    print(f"  global eligibility:   {run_meta['eligibility_rate_global']:.1%}")
    print()
    print("  per-date eligibility head/tail:")
    for _, r in summary.head(3).iterrows():
        print(f"    {r['rebalance_date'].date()}  "
              f"eligible={r['eligible']:>3}/{r['total_considered']:<4} "
              f"({r['eligibility_rate']:.0%})")
    print("    …")
    for _, r in summary.tail(3).iterrows():
        print(f"    {r['rebalance_date'].date()}  "
              f"eligible={r['eligible']:>3}/{r['total_considered']:<4} "
              f"({r['eligibility_rate']:.0%})")
    print()
    print("  top rejection reasons (global):")
    for _, r in hist.head(8).iterrows():
        print(f"    {r['reason']:<24} {r['count']:>6}")
    print()
    print("  execution proxy coverage (ex-ante → overlay-eligible):")
    for _, r in coverage[coverage["role"] == "ex_ante"].iterrows():
        print(f"    {r['column']:<32} coverage={r['coverage']:.0%}  "
              f"median={r['median']:.4f}" if r['median'] is not None
              else f"    {r['column']:<32} coverage={r['coverage']:.0%}  median=None")
    print("  realized diagnostics (NOT in overlay, diagnostic only):")
    for _, r in coverage[coverage["role"] == "diagnostic_only"].iterrows():
        print(f"    {r['column']:<32} coverage={r['coverage']:.0%}")
    print()
    print("  ex-ante proxy pairwise |Spearman r| (eligible rows, Spearman):")
    if not corr.empty:
        for _, rr in high_pairs.head(10).iterrows():
            print(f"    {rr['col_a']:<30} ↔ {rr['col_b']:<30} "
                  f"r={rr['corr']:+.2f}   ⚠ redundant")
        if high_pairs.empty:
            print("    none at |r|≥0.7 — proxies are not mutually collinear")
        elif len(high_pairs) > 10:
            print(f"    … (+{len(high_pairs) - 10} more)")
    print()
    print(f"  reports written to: {reports_dir}/")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ohlcv", default=DEFAULT_OHLCV, help=f"Panel parquet (default: {DEFAULT_OHLCV})")
    p.add_argument("--xu100", default=DEFAULT_XU100, help=f"XU100 parquet (default: {DEFAULT_XU100})")
    p.add_argument("--frequency", default="M", choices=["M", "W"], help="Rebalance frequency")
    p.add_argument("--start", default="2022-01-01", help="Rebalance window start (YYYY-MM-DD)")
    p.add_argument("--end", default=None, help="Rebalance window end (YYYY-MM-DD); default: latest")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        run(args)
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        raise
