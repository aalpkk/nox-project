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


DEFAULT_DATASET = Path("output/nyxexp_dataset_v4.parquet")
OUT_DIR = Path("output")


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
    args = ap.parse_args()

    dataset = Path(args.dataset)
    if not dataset.exists():
        print(f"❌ dataset not found: {dataset}")
        return 2

    target = _resolve_target_date(dataset, args.date)
    print(f"═══ nyxexpansion DAILY SCAN ORCHESTRATOR ═══")
    print(f"  dataset: {dataset}")
    print(f"  target : {target.date()}")

    # 1) Run scan
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

    # 3) Build 4-stock Markowitz from top-15 non-severe
    print(f"\n[3/3] Markowitz 4-stock combinatorial …")
    non_severe = scan_df[scan_df["risk_bucket"] != "severe"].copy()
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
    }
    html_str = render_html(scan_df, portfolio, target, meta)

    out_path = Path(args.out_html) if args.out_html else OUT_DIR / f"nyxexp_scan_{target.strftime('%Y%m%d')}.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_str, encoding="utf-8")
    print(f"\n✅ HTML: {out_path}")

    latest = OUT_DIR / "nyxexp_scan_latest.html"
    latest.write_text(html_str, encoding="utf-8")
    print(f"✅ HTML (latest alias): {latest}")

    # 5) Optional GH Pages push
    if args.push:
        try:
            from core.reports import push_html_to_github
            url = push_html_to_github(
                html_str, "nyxexp_scan.html", target.strftime("%Y-%m-%d")
            )
            print(f"✅ published: {url}")
        except Exception as e:
            print(f"⚠ pages push failed: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
