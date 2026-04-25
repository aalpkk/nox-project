"""
nyxexpansion daily scan orchestrator — runs scan_latest, builds 4-stock Markowitz,
renders HTML, writes outputs.

Usage:
  python -m nyxexpansion.scan.run [--date YYYY-MM-DD] [--dataset PATH] [--top 25]
                                  [--out-html PATH] [--push]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).resolve()
_ROOT = _HERE.parent.parent.parent
sys.path.insert(0, str(_ROOT))

from nyxexpansion.scan.markowitz import combinatorial_max_sharpe
from nyxexpansion.scan.html_report import render_html
from nyxexpansion.retention.stage import (
    run_retention_stage,
    mark_stage_disabled,
)
from nyxexpansion.retention.log import append_row as append_retention_log
from nyxexpansion import retention as ret_pkg


DEFAULT_DATASET = Path("output/nyxexp_dataset_v4.parquet")
OUT_DIR = Path("output")
DEFAULT_INTRADAY_15M = Path("output/nyxexp_intraday_master.parquet")
DEFAULT_MASTER_OHLCV = Path("output/ohlcv_10y_fintables_master.parquet")
DEFAULT_SURROGATE = Path("output/nyxexp_retention_surrogate_v1.pkl")


def _run_scan_subprocess(dataset: Path, date: str | None, top: int) -> int:
    cmd = [
        sys.executable, "-m", "nyxexpansion.tools.scan_latest",
        "--dataset", str(dataset),
        "--top", str(top),
        "--also-last-n-days", "1",
    ]
    if date:
        cmd += ["--date", date]
    print(f"  → subprocess: {' '.join(cmd)}")
    res = subprocess.run(cmd, cwd=str(_ROOT))
    return res.returncode


def _resolve_target_date(dataset: Path, date_arg: str | None) -> pd.Timestamp:
    if date_arg:
        return pd.Timestamp(date_arg)
    df = pd.read_parquet(dataset, columns=["date"])
    return pd.Timestamp(pd.to_datetime(df["date"]).max())


def _load_scan_output(target: pd.Timestamp) -> pd.DataFrame:
    path = OUT_DIR / f"nyxexp_scan_{target.strftime('%Y%m%d')}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"scan output missing: {path}")
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] == target].copy()
    df = df.sort_values("winner_R_pred", ascending=False).reset_index(drop=True)
    return df


def _universe_size(dataset: Path) -> int:
    try:
        df = pd.read_parquet(dataset, columns=["ticker"])
        return int(df["ticker"].nunique())
    except Exception:
        return -1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=None, help="target date YYYY-MM-DD (default: dataset max)")
    ap.add_argument("--dataset", default=str(DEFAULT_DATASET))
    ap.add_argument("--top", type=int, default=25,
                    help="top-N adayı scan.tools'a verilecek (HTML'de hepsi gösterilir)")
    ap.add_argument("--mk-universe-top", type=int, default=15,
                    help="Markowitz evreni için winR-top-N (severe hariç)")
    ap.add_argument("--out-html", default=None,
                    help="HTML output path (default: output/nyxexp_scan_<date>.html)")
    ap.add_argument("--push", action="store_true",
                    help="GitHub Pages push via core.reports.push_html_to_github")
    ap.add_argument("--retention-intraday", default=str(DEFAULT_INTRADAY_15M),
                    help="15m bars cache path used by the timing-clean retention stage")
    ap.add_argument("--retention-master-ohlcv", default=str(DEFAULT_MASTER_OHLCV),
                    help="Daily master OHLCV parquet for truncated feature rebuild")
    ap.add_argument("--retention-artifact", default=str(DEFAULT_SURROGATE),
                    help="Persisted surrogate artifact (.pkl) — fail-fast if missing")
    ap.add_argument("--retention-rank", type=int, default=10,
                    help="Pessimistic rank threshold for retention_pass (default 10)")
    ap.add_argument("--skip-retention", action="store_true",
                    help="Skip the timing-clean retention stage; HTML will flag this explicitly")
    ap.add_argument("--skip-scan", action="store_true",
                    help="Skip scan_latest subprocess; assume nyxexp_scan_<date>.parquet already exists")
    ap.add_argument("--out-suffix", default="",
                    help="Suffix for output names (e.g. 'live' → nyxexp_scan_live_<date>.html, "
                         "Pages target nyxexp_scan_live.html). Default empty = canonical EOD outputs.")
    args = ap.parse_args()

    suffix = args.out_suffix.strip().strip("_")
    suffix_part = f"_{suffix}" if suffix else ""

    dataset = Path(args.dataset)
    if not dataset.exists():
        print(f"❌ dataset not found: {dataset}")
        return 2

    target = _resolve_target_date(dataset, args.date)
    print(f"═══ nyxexpansion DAILY SCAN ORCHESTRATOR ═══")
    print(f"  dataset: {dataset}")
    print(f"  target : {target.date()}")

    # 1) Run scan
    if args.skip_scan:
        print("\n[1/3] scan_latest SKIPPED (--skip-scan); using existing parquet")
    else:
        print("\n[1/3] scan_latest …")
        rc = _run_scan_subprocess(dataset, args.date or target.strftime("%Y-%m-%d"), args.top)
        if rc != 0:
            print(f"❌ scan_latest failed with rc={rc}")
            return rc

    # 2) Load scan output, filter to target
    scan_df = _load_scan_output(target)
    n_total = len(scan_df)
    if n_total == 0:
        print(f"⚠ scan returned 0 signals for {target.date()}")
    print(f"\n[2/3] scan output: {n_total} candidate(s) at {target.date()}")

    # 2b) Timing-clean retention filter (locked 2026-04-25)
    retention_meta = {"enabled": not args.skip_retention}
    if args.skip_retention:
        print("\n[2b/3] timing-clean retention stage SKIPPED (--skip-retention)")
        scan_df = mark_stage_disabled(scan_df)
        retention_meta.update({
            "n_pass": 0, "n_drop": 0, "n_unscored": len(scan_df),
            "notes": {"stage_disabled": len(scan_df)},
            "rank_threshold": args.retention_rank,
            "artifact": str(args.retention_artifact),
        })
    else:
        print("\n[2b/3] timing-clean retention stage …")
        from nyxexpansion.tools.presmoke import load_xu100
        xu = load_xu100(refresh=False, period="6y")
        xu_close = xu["Close"] if xu is not None and not xu.empty else None
        outcome = run_retention_stage(
            scan_df, target,
            intraday_15m_path=Path(args.retention_intraday),
            master_ohlcv_path=Path(args.retention_master_ohlcv),
            surrogate_artifact_path=Path(args.retention_artifact),
            rank_threshold=args.retention_rank,
            xu100_close=xu_close,
        )
        scan_df = outcome.enriched
        retention_meta.update({
            "n_pass": outcome.n_pass,
            "n_drop": outcome.n_drop,
            "n_unscored": outcome.n_unscored,
            "notes": outcome.notes,
            "source_breakdown": outcome.source_breakdown,
            "rank_threshold": args.retention_rank,
            "artifact": str(args.retention_artifact),
        })
        print(f"  PASS (rank ≤ {args.retention_rank}): {outcome.n_pass}  "
              f"DROP: {outcome.n_drop}  unscored: {outcome.n_unscored}")
        for note, count in outcome.notes.items():
            print(f"    note={note}: {count}")
        if outcome.source_breakdown:
            srcs = ", ".join(f"{k}={v}" for k, v in sorted(outcome.source_breakdown.items()))
            print(f"  bars source breakdown: {srcs}")

    # 3) Build 4-stock Markowitz from top-15 non-severe AND retention_pass
    print(f"\n[3/3] Markowitz 4-stock combinatorial …")
    non_severe = scan_df[scan_df["risk_bucket"] != "severe"].copy()
    if not args.skip_retention:
        before = len(non_severe)
        non_severe = non_severe[non_severe["retention_pass"]].copy()
        print(f"  retention filter: {before} → {len(non_severe)} (pass-only)")
    universe = non_severe.head(args.mk_universe_top)["ticker"].tolist()
    print(f"  Markowitz universe (top {args.mk_universe_top} non-severe by winR): {universe}")
    portfolio = combinatorial_max_sharpe(universe, target)
    if portfolio.get("error"):
        print(f"  ⚠ portfolio error: {portfolio['error']}")
    else:
        print(f"  ✓ sharpe={portfolio['sharpe']:.3f}  "
              f"ret={portfolio['expected_return']:+.2f}% risk={portfolio['expected_risk']:.2f}%  "
              f"{portfolio['combos_evaluated']:,} combos")
        for t, w in sorted(portfolio["weights"].items(), key=lambda x: -x[1]):
            print(f"    {t:<8}  {w*100:5.1f}%")

    # 4) Render HTML
    regime_dist = scan_df["xu_regime"].value_counts().to_dict() if "xu_regime" in scan_df.columns else {}
    meta = {
        "dataset_path": str(dataset),
        "universe_size": _universe_size(dataset),
        "regime_dist": regime_dist,
        "retention": retention_meta,
    }
    html_str = render_html(scan_df, portfolio, target, meta)

    enriched_path = OUT_DIR / f"nyxexp_scan_retention{suffix_part}_{target.strftime('%Y%m%d')}.parquet"
    scan_df.to_parquet(enriched_path, index=False)
    print(f"\n✅ enriched parquet: {enriched_path}")

    if not args.skip_retention:
        try:
            log_path = append_retention_log(
                target, retention_meta=retention_meta,
                surrogate_schema=ret_pkg.TRUNCATED_FEATURE_SCHEMA_VERSION,
            )
            print(f"✅ retention log: {log_path}")
        except Exception as exc:
            print(f"⚠ retention log write failed: {exc}")

    out_path = Path(args.out_html) if args.out_html else \
        OUT_DIR / f"nyxexp_scan{suffix_part}_{target.strftime('%Y%m%d')}.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_str, encoding="utf-8")
    print(f"✅ HTML: {out_path}")

    latest_name = f"nyxexp_scan{suffix_part}_latest.html"
    latest = OUT_DIR / latest_name
    latest.write_text(html_str, encoding="utf-8")
    print(f"✅ HTML (latest alias): {latest}")

    # 5) Optional GH Pages push
    if args.push:
        try:
            from core.reports import push_html_to_github
            pages_target = f"nyxexp_scan{suffix_part}.html"
            url = push_html_to_github(
                html_str, pages_target, target.strftime("%Y-%m-%d")
            )
            print(f"✅ published: {url}")
        except Exception as e:
            print(f"⚠ pages push failed: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
