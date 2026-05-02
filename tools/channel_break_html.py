"""channel_break HTML report — briefing aesthetic, per-TF separate pages.

Reads `output/channel_break_ch_<tf>.parquet` × 4 +
`output/pending_triangle_ch_<tf>.parquet` × 4 (run
`tools/channel_break_scan_live.py` first) and emits 5 files per as-of:
    output/channel_break_5h_<YYYY-MM-DD>.html      + _latest.html
    output/channel_break_1d_<YYYY-MM-DD>.html      + _latest.html
    output/channel_break_1w_<YYYY-MM-DD>.html      + _latest.html
    output/channel_break_1mo_<YYYY-MM-DD>.html     + _latest.html
    output/channel_break_scan_<YYYY-MM-DD>.html    + _latest.html  (index)

Usage:
    python tools/channel_break_html.py [--asof 2026-04-29] [--top 30]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from channel_break.html_report import (
    OUT_DIR,
    TF_ORDER,
    _file_for,
    build_all,
)


def _resolve_asof(arg: str | None) -> pd.Timestamp:
    if arg:
        return pd.Timestamp(arg)
    for fam in ("ch_1d", "ch_5h", "ch_1w", "ch_1M"):
        p = OUT_DIR / f"channel_break_{fam}.parquet"
        if p.exists():
            df = pd.read_parquet(p, columns=["as_of_ts"])
            if not df.empty:
                return pd.Timestamp(df["as_of_ts"].max())
    return pd.Timestamp("today").normalize()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asof", default=None,
                    help="As-of date YYYY-MM-DD (default: max as_of_ts in parquets).")
    ap.add_argument("--top", type=int, default=30,
                    help="Top-N rows per TF table (default 30).")
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

    index_html = pages["index"]
    idx_dated = OUT_DIR / f"channel_break_scan_{asof_str}.html"
    idx_latest = OUT_DIR / "channel_break_scan_latest.html"
    idx_dated.write_text(index_html, encoding="utf-8")
    idx_latest.write_text(index_html, encoding="utf-8")
    total_chars += len(index_html)
    print(f"[html] wrote {idx_dated.name}  ({len(index_html):,} chars)")
    print(f"[html] wrote {idx_latest.name}")

    print(f"[html] asof={asof_str}  top={args.top}  total_size={total_chars:,} chars")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
