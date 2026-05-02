"""mb_scanner HTML report — briefing aesthetic, per-TF separate pages.

Reads `output/mb_scanner_<family>.parquet` × 8 (run `tools/mb_scanner_run.py`
first) and emits 5 files per as-of:
    output/mb_scanner_5h_<YYYY-MM-DD>.html      + _latest.html
    output/mb_scanner_1d_<YYYY-MM-DD>.html      + _latest.html
    output/mb_scanner_1w_<YYYY-MM-DD>.html      + _latest.html
    output/mb_scanner_1mo_<YYYY-MM-DD>.html     + _latest.html
    output/mb_scanner_scan_<YYYY-MM-DD>.html    + _latest.html  (index)

Each TF page has MB and BB tables stacked full-width plus the cross-TF
stack panel; the index page lists all 4 TFs as tiles. Ticker cells link
to TV charts; hover shows pivot dates.

Usage:
    python tools/mb_scanner_html.py [--asof 2026-04-30] [--top 25]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from mb_scanner.html_report import TF_ORDER, _file_for, build_all
from mb_scanner.rank import OUT_DIR


def _resolve_asof(arg: str | None) -> pd.Timestamp:
    if arg:
        return pd.Timestamp(arg)
    # peek any family parquet to grab as_of_ts
    for fam in ("mb_1d", "mb_5h", "mb_1w", "mb_1M"):
        p = OUT_DIR / f"mb_scanner_{fam}.parquet"
        if p.exists():
            df = pd.read_parquet(p, columns=["as_of_ts"])
            if not df.empty:
                return pd.Timestamp(df["as_of_ts"].max())
    return pd.Timestamp("today").normalize()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asof", default=None,
                    help="As-of date YYYY-MM-DD (default: max as_of_ts in parquets).")
    ap.add_argument("--top", type=int, default=25,
                    help="Top-N rows per family table (default 25).")
    args = ap.parse_args()

    asof = _resolve_asof(args.asof)
    asof_str = asof.strftime("%Y-%m-%d")

    pages = build_all(asof=asof, out_dir=OUT_DIR, top=args.top)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    total_chars = 0
    for tf in TF_ORDER:
        dated = OUT_DIR / _file_for(tf, asof_str)
        latest = OUT_DIR / _file_for(tf)
        html_str = pages[tf]
        dated.write_text(html_str, encoding="utf-8")
        latest.write_text(html_str, encoding="utf-8")
        total_chars += len(html_str)
        print(f"[html] wrote {dated.name}  ({len(html_str):,} chars)")
        print(f"[html] wrote {latest.name}")

    # index
    index_html = pages["index"]
    idx_dated = OUT_DIR / f"mb_scanner_scan_{asof_str}.html"
    idx_latest = OUT_DIR / "mb_scanner_scan_latest.html"
    idx_dated.write_text(index_html, encoding="utf-8")
    idx_latest.write_text(index_html, encoding="utf-8")
    total_chars += len(index_html)
    print(f"[html] wrote {idx_dated.name}  ({len(index_html):,} chars)")
    print(f"[html] wrote {idx_latest.name}")

    print(f"[html] asof={asof_str}  top={args.top}  total_size={total_chars:,} chars")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
