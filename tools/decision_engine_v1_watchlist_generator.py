#!/usr/bin/env python3
"""Decision Engine v1 — post-Class-B tomorrow-watchlist generator.

Reads:  output/decision_engine_v1_events.parquet
Writes: output/decision_engine_v1_tomorrow_watchlist_post_class_b_<asof>.{md,csv}

asof resolution (PR-DE-2b):
  - `--asof-date YYYY-MM-DD` wins. Must match the events parquet's asof set
    (otherwise hard FAIL — refusing to write a filename that misrepresents
    the underlying source).
  - else auto-detect from max(events.date).

Sections (CSV column 0 and MD H2 ordering):
  EXECUTABLE -> SIZE_REDUCED -> WAIT_BETTER_ENTRY -> WAIT_RETEST -> PAPER_ONLY

EXECUTABLE row order: multi-cell tickers first (overlap-count desc, ticker
                      alpha tie-break), then single-cell rows ticker alpha.
Other sections: ticker alpha, then family / timeframe tie-break.

Manual-review watchlist only. Live gate CLOSED — no live_execution path
touched, no portfolio/rank/score emitted.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "output"
EVENTS_PATH = OUT_DIR / "decision_engine_v1_events.parquet"

SECTIONS = ["EXECUTABLE", "SIZE_REDUCED", "WAIT_BETTER_ENTRY", "WAIT_RETEST", "PAPER_ONLY"]

CSV_COLS = [
    "section", "ticker", "source", "family", "state", "timeframe",
    "setup_label", "execution_label", "market_context", "reason_codes",
    "entry_ref", "stop_ref", "atr", "fill_assumption", "risk_atr",
    "paper_stream_ref", "notes",
]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Decision Engine v1 — post-Class-B tomorrow-watchlist generator. "
            "Reads decision_engine_v1_events.parquet; writes "
            "decision_engine_v1_tomorrow_watchlist_post_class_b_<asof>.{md,csv}. "
            "asof resolves from --asof-date (must match events parquet) or "
            "max(events.date)."
        )
    )
    p.add_argument(
        "--asof-date",
        type=lambda s: dt.date.fromisoformat(s).isoformat(),
        default=None,
        metavar="YYYY-MM-DD",
        help=(
            "Operational asof date. If omitted, auto-detect from "
            "max(events.date). Mismatch with events parquet is a hard FAIL."
        ),
    )
    return p.parse_args()


def _resolve_asof(arg_asof: str | None, events_dates: set[str]) -> str:
    if not events_dates:
        print(
            f"[watchlist_generator] FAIL: events parquet has zero `date` values",
            flush=True,
        )
        sys.exit(1)
    if arg_asof is not None:
        if arg_asof not in events_dates:
            print(
                f"[watchlist_generator] FAIL: --asof-date {arg_asof} not "
                f"present in events.parquet (have: {sorted(events_dates)})",
                flush=True,
            )
            sys.exit(1)
        return arg_asof
    return sorted(events_dates)[-1]


def _fmt_num(v, digits=4):
    if v is None:
        return ""
    try:
        f = float(v)
    except Exception:
        return ""
    if f != f:
        return ""
    return f"{f:.{digits}f}"


def _fmt_risk(v):
    if v is None:
        return ""
    try:
        f = float(v)
    except Exception:
        return ""
    if f != f:
        return ""
    return f"{f:.2f}"


def _join_codes(*lists):
    out = []
    seen = set()
    for lst in lists:
        if not lst:
            continue
        for c in lst:
            if not c or c in seen:
                continue
            seen.add(c)
            out.append(c)
    return ";".join(out)


def main():
    args = _parse_args()

    if not EVENTS_PATH.exists():
        print(
            f"[watchlist_generator] FAIL: events parquet missing at "
            f"{EVENTS_PATH.relative_to(ROOT)}",
            flush=True,
        )
        sys.exit(1)

    t = pq.read_table(EVENTS_PATH)
    rows = t.to_pylist()
    events_dates = {str(r["date"]) for r in rows}
    asof = _resolve_asof(args.asof_date, events_dates)
    total = len(rows)

    out_md = OUT_DIR / f"decision_engine_v1_tomorrow_watchlist_post_class_b_{asof}.md"
    out_csv = OUT_DIR / f"decision_engine_v1_tomorrow_watchlist_post_class_b_{asof}.csv"

    records = []
    for r in rows:
        rec = {
            "ticker": r["ticker"],
            "source": r["source"],
            "family": r["family"],
            "state": r["state"],
            "timeframe": r["timeframe"],
            "setup_label": r["setup_label"],
            "execution_label": r["execution_label"],
            "market_context": r["regime"] or "",
            "reason_codes": _join_codes(
                r.get("setup_reason_codes"),
                r.get("execution_reason_codes"),
                r.get("context_reason_codes"),
            ),
            "entry_ref": r["entry_ref"],
            "stop_ref": r["stop_ref"],
            "atr": r["atr"],
            "fill_assumption": r["fill_assumption"] or "",
            "risk_atr": r["risk_atr"],
            "paper_stream_ref": r["paper_stream_ref"] or "",
            "notes": "",
        }
        records.append(rec)

    by_section: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        sec = rec["execution_label"]
        if sec in SECTIONS:
            by_section[sec].append(rec)

    exec_rows = by_section.get("EXECUTABLE", [])
    overlap_counts = Counter(r["ticker"] for r in exec_rows)
    multi_tickers = {tk for tk, c in overlap_counts.items() if c > 1}

    def exec_sort_key(r):
        tk = r["ticker"]
        cnt = overlap_counts[tk]
        primary = 0 if cnt > 1 else 1
        secondary = -cnt
        return (primary, secondary, tk, r["family"], r["timeframe"])

    sorted_exec = sorted(exec_rows, key=exec_sort_key)

    sorted_sections: dict[str, list[dict]] = {"EXECUTABLE": sorted_exec}
    for sec in SECTIONS[1:]:
        sorted_sections[sec] = sorted(
            by_section.get(sec, []),
            key=lambda r: (r["ticker"], r["family"], r["timeframe"]),
        )

    source_dist: dict[str, Counter] = {sec: Counter() for sec in SECTIONS}
    for sec in SECTIONS:
        for r in sorted_sections[sec]:
            source_dist[sec][r["source"]] += 1

    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_COLS)
        w.writeheader()
        for sec in SECTIONS:
            for r in sorted_sections[sec]:
                row = {
                    "section": sec,
                    "ticker": r["ticker"],
                    "source": r["source"],
                    "family": r["family"],
                    "state": r["state"],
                    "timeframe": r["timeframe"],
                    "setup_label": r["setup_label"],
                    "execution_label": r["execution_label"],
                    "market_context": r["market_context"],
                    "reason_codes": r["reason_codes"],
                    "entry_ref": r["entry_ref"] if r["entry_ref"] is not None else "",
                    "stop_ref": r["stop_ref"] if r["stop_ref"] is not None else "",
                    "atr": r["atr"] if r["atr"] is not None else "",
                    "fill_assumption": r["fill_assumption"],
                    "risk_atr": r["risk_atr"] if r["risk_atr"] is not None else "",
                    "paper_stream_ref": r["paper_stream_ref"],
                    "notes": r["notes"],
                }
                w.writerow(row)

    now_utc = dt.datetime.now(dt.timezone.utc).isoformat(timespec="microseconds")
    lines: list[str] = []
    A = lines.append
    A(f"# Decision Engine v1 — Tomorrow Watchlist (post-Class-B {asof})")
    A("")
    A(f"- Generated (UTC): {now_utc}")
    A(f"- Source events: `output/decision_engine_v1_events.parquet` (Tier 2 PASS post-Class-B {asof})")
    A(f"- asof_date in events: {asof}")
    A(f"- Total events processed: {total}")
    A("")
    A("## Hard warnings")
    A("")
    A("- **NOT live auto-trade.** Manual-review watchlist only.")
    A("- **Live gate CLOSED.** No `live_execution_allowed=True` row reaches an executor.")
    A("- **Manual review required** before any paper or live action.")
    A("- **EXECUTABLE = LOCKED v1 risk branch passed thresholds (risk_atr ≤ 2.0).** This is NOT portfolio approval; gap, liquidity, regime context still need review.")
    A("- **No score, no rank, no portfolio_weight emitted.** No PF/WR/meanR. No backtest.")
    A("")
    A("## Distribution")
    A("")
    A("| section | count |")
    A("|---|---:|")
    sec_total = 0
    for sec in SECTIONS:
        n = len(sorted_sections[sec])
        sec_total += n
        A(f"| {sec} | {n} |")
    A(f"| **total** | **{sec_total}** |")
    A("")

    A("## Source distribution per section")
    A("")
    A("| section | source | count |")
    A("|---|---|---:|")
    for sec in SECTIONS:
        for src, n in sorted(source_dist[sec].items(), key=lambda x: (-x[1], x[0])):
            A(f"| {sec} | {src} | {n} |")
    A("")

    A("## EXECUTABLE overlap summary (descriptive)")
    A("")
    multi_sorted = sorted(
        [(tk, overlap_counts[tk]) for tk in multi_tickers],
        key=lambda x: (-x[1], x[0]),
    )
    overlap_rows = sum(overlap_counts[tk] for tk in multi_tickers)
    if multi_sorted:
        A(f"_{len(multi_sorted)} tickers carry >1 EXECUTABLE row across (family, timeframe) cells. {overlap_rows} of {len(exec_rows)} EXECUTABLE rows participate. Listed in descending overlap count._")
        A("")
        A("| ticker | exec rows | family × timeframe cells |")
        A("|---|---:|---|")
        ticker_cells: dict[str, list[str]] = defaultdict(list)
        for r in sorted_exec:
            ticker_cells[r["ticker"]].append(f"{r['family']}@{r['timeframe']}")
        for tk, cnt in multi_sorted:
            cells = ", ".join(ticker_cells[tk])
            A(f"| {tk} | {cnt} | {cells} |")
    else:
        A("_No tickers carry >1 EXECUTABLE row._")
    A("")

    def write_section_table(section: str, rows: list[dict], extra_note: str = ""):
        A(f"## {section} ({len(rows)} rows)")
        A("")
        if extra_note:
            A(f"_{extra_note}_")
            A("")
        A("| ticker | source | family | state | tf | setup | exec | regime | risk_atr | entry | stop | atr | fill | paper_ref | reasons |")
        A("|---|---|---|---|---|---|---|---|---:|---:|---:|---:|---|---|---|")
        for r in rows:
            A(
                "| "
                + " | ".join([
                    r["ticker"],
                    r["source"],
                    r["family"],
                    r["state"],
                    r["timeframe"],
                    r["setup_label"],
                    r["execution_label"],
                    r["market_context"],
                    _fmt_risk(r["risk_atr"]),
                    _fmt_num(r["entry_ref"], 4),
                    _fmt_num(r["stop_ref"], 4),
                    _fmt_num(r["atr"], 4),
                    r["fill_assumption"],
                    r["paper_stream_ref"],
                    r["reason_codes"],
                ])
                + " |"
            )
        A("")

    exec_note = ""
    if multi_sorted:
        exec_note = f"Overlap candidates: {len(multi_sorted)} tickers carry >1 EXECUTABLE row ({overlap_rows} rows). Listed first."
    write_section_table("EXECUTABLE", sorted_sections["EXECUTABLE"], exec_note)
    write_section_table("SIZE_REDUCED", sorted_sections["SIZE_REDUCED"])
    write_section_table("WAIT_BETTER_ENTRY", sorted_sections["WAIT_BETTER_ENTRY"])
    write_section_table("WAIT_RETEST", sorted_sections["WAIT_RETEST"])
    write_section_table("PAPER_ONLY", sorted_sections["PAPER_ONLY"])

    out_md.write_text("\n".join(lines), encoding="utf-8")

    md_bytes = out_md.read_bytes()
    csv_bytes = out_csv.read_bytes()
    md_sha = hashlib.sha256(md_bytes).hexdigest()
    csv_sha = hashlib.sha256(csv_bytes).hexdigest()

    print(f"=== watchlist post-class-b {asof} ===")
    print(f"asof_date: {asof}")
    print(f"total events processed: {total}")
    for sec in SECTIONS:
        print(f"  {sec}: {len(sorted_sections[sec])}")
    print(f"  total sectioned: {sec_total}")
    print(f"EXECUTABLE multi-cell tickers: {len(multi_sorted)} ({overlap_rows} rows)")
    print(f"OUT_MD: {out_md} bytes={len(md_bytes)} sha256={md_sha}")
    print(f"OUT_CSV: {out_csv} bytes={len(csv_bytes)} sha256={csv_sha}")


if __name__ == "__main__":
    main()
