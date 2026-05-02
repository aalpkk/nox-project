"""Render hwo_full multi-TF scan CSVs to a single HTML — briefing aesthetic.

Pairs with tools/hwo_full_scan.py. Reads up to three CSVs per as-of:
    output/hwo_full_5h_scan_<asof>.csv
    output/hwo_full_1d_scan_<asof>.csv
    output/hwo_full_1w_scan_<asof>.csv

Emits one HTML index file + sticky TF tabs:
    output/hwo_full_scan_<asof>.html  (+ _latest.html with --also-latest)

Output is descriptive: full HW × signal crossover event log per TF
(small + big dots). NO trade-edge claim — pre-registered backtest
hwo_mtf_v1 CLOSED_REJECTED.
"""
from __future__ import annotations

import argparse
import html
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from core.reports import _NOX_CSS

KIND_BADGE = {
    "AL_OS":  ("AL · OS",  "var(--nox-green)",  "big"),
    "AL":     ("AL",       "var(--nox-cyan)",   "small"),
    "SAT_OB": ("SAT · OB", "var(--nox-red)",    "big"),
    "SAT":    ("SAT",      "var(--nox-orange)", "small"),
}
KIND_ORDER = ["AL_OS", "AL", "SAT_OB", "SAT"]

TF_ORDER = ["5h", "1d", "1w"]
TF_LABELS = {"5h": "5-Saatlik", "1d": "Günlük", "1w": "Haftalık"}
TF_DOT = {
    "5h": "var(--nox-gold)",
    "1d": "var(--nox-green)",
    "1w": "var(--nox-blue)",
}
TF_TS_FMT = {"5h": "%Y-%m-%d %H:%M", "1d": "%Y-%m-%d", "1w": "%Y-%m-%d"}

_LOCAL_CSS = """
.briefing-container {
  position: relative; z-index: 1;
  max-width: 1180px; margin: 0 auto;
  padding: 0 1.4rem 2rem;
}
.hw-tab-nav {
  display: flex; gap: 0.6rem; flex-wrap: wrap;
  margin: 0.8rem 0 1.2rem 0;
  position: sticky; top: 0; z-index: 5;
  background: var(--bg, #0c0d10); padding: 0.5rem 0;
  border-bottom: 1px solid var(--border-dim);
}
.hw-tab {
  display: inline-flex; align-items: center; gap: 0.45rem;
  padding: 0.4rem 0.8rem; border: 1px solid var(--border-dim);
  border-radius: 999px; font-size: 0.78rem; cursor: pointer;
  background: var(--bg-elevated); color: var(--text-secondary);
  text-decoration: none;
}
.hw-tab:hover { border-color: var(--nox-cyan); color: var(--nox-cyan); }
.hw-tab .cnt {
  font-family: var(--font-mono); font-weight: 700; font-size: 0.78rem;
  color: var(--text-primary);
}
.hw-tab .dot { width: 8px; height: 8px; border-radius: 50%; }
.hw-card {
  background: var(--bg-elevated); border: 1px solid var(--border-dim);
  border-radius: 10px; padding: 1rem 1.1rem; margin: 1rem 0;
  scroll-margin-top: 5rem;
}
.hw-card h2 {
  font-size: 0.95rem; font-weight: 600; letter-spacing: 0.04em;
  text-transform: uppercase; color: var(--text-secondary);
  margin: 0 0 0.7rem 0;
  display: flex; align-items: center; gap: 0.6rem; flex-wrap: wrap;
}
.hw-card h2 .tag {
  font-size: 0.7rem; font-weight: 600; padding: 1px 8px; border-radius: 999px;
  border: 1px solid var(--border-dim); color: var(--text-muted);
  letter-spacing: 0.05em;
}
.hw-card h3 {
  font-size: 0.78rem; font-weight: 600; letter-spacing: 0.05em;
  text-transform: uppercase; color: var(--text-muted);
  margin: 1rem 0 0.5rem 0;
}
.hw-note {
  font-size: 0.78rem; color: var(--text-muted);
  line-height: 1.5; margin: 0 0 0.8rem 0;
}
table.nox-table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
table.nox-table th, table.nox-table td {
  padding: 0.45rem 0.6rem; border-bottom: 1px solid var(--border-dim);
}
table.nox-table th {
  text-align: left; font-weight: 600; letter-spacing: 0.03em;
  color: var(--text-secondary); text-transform: uppercase; font-size: 0.72rem;
}
table.nox-table tbody tr:hover { background: var(--bg-hover); }
.kind-pill {
  display: inline-block; padding: 1px 7px; border-radius: 999px;
  font-size: 0.72rem; font-weight: 700; letter-spacing: 0.03em;
}
.dot-mark {
  display: inline-block; width: 8px; height: 8px;
  border-radius: 50%; margin-right: 5px; vertical-align: middle;
}
.dot-mark.big   { width: 10px; height: 10px; }
.dot-mark.small { width: 6px;  height: 6px;  opacity: 0.85; }
.empty-row td {
  text-align: center; opacity: 0.6; padding: 1.1rem 0.6rem;
}
.hw-footer {
  margin-top: 1.4rem; font-size: 0.72rem; color: var(--text-muted);
  line-height: 1.5;
}
.hw-footer code { color: var(--text-secondary); }
details.hw-collapse > summary {
  cursor: pointer; font-size: 0.78rem; color: var(--text-muted);
  margin: 0.5rem 0; user-select: none;
}
details.hw-collapse[open] > summary { color: var(--nox-cyan); }
"""


def _fmt_ts(ts, tf: str) -> str:
    t = pd.Timestamp(ts)
    if tf == "5h":
        if t.tz is None:
            t = t.tz_localize("Europe/Istanbul")
        else:
            t = t.tz_convert("Europe/Istanbul")
    fmt = TF_TS_FMT.get(tf, "%Y-%m-%d")
    return t.strftime(fmt)


def _row(ev: pd.Series, tf: str) -> str:
    label, color, size = KIND_BADGE[ev["kind"]]
    ts = _fmt_ts(ev["ts"], tf)
    dot_class = "big" if size == "big" else "small"
    return (
        "<tr>"
        f"<td>{ts}</td>"
        f"<td><b>{html.escape(str(ev['ticker']))}</b></td>"
        f"<td><span class='dot-mark {dot_class}' style='background:{color}'></span>"
        f"<span class='kind-pill' style='color:{color};border:1px solid {color}'>{label}</span></td>"
        f"<td style='text-align:right'>{ev['hyperwave']:.2f}</td>"
        f"<td style='text-align:right'>{ev['signal']:.2f}</td>"
        f"<td style='text-align:right'>{ev['close']:.2f}</td>"
        f"<td style='text-align:right'>{int(ev['volume']):,}</td>"
        "</tr>"
    )


def _table(rows_html: str, header: str, empty_msg: str, colspan: int = 7) -> str:
    body = rows_html if rows_html else f"<tr class='empty-row'><td colspan='{colspan}'>{empty_msg}</td></tr>"
    return f"<table class='nox-table'>{header}<tbody>{body}</tbody></table>"


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=["ts"])
    if df.empty:
        return df
    if isinstance(df["ts"].dtype, pd.DatetimeTZDtype):
        df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_convert("Europe/Istanbul")
    else:
        df["ts"] = pd.to_datetime(df["ts"])
    return df


def _kind_counts(df: pd.DataFrame) -> dict:
    if df.empty:
        return {k: 0 for k in KIND_ORDER}
    c = df["kind"].value_counts().to_dict()
    return {k: int(c.get(k, 0)) for k in KIND_ORDER}


def _tf_section(tf: str, df: pd.DataFrame) -> tuple[str, dict]:
    table_header = (
        "<thead><tr>"
        "<th>Bar</th><th>Sembol</th><th>Sinyal</th>"
        "<th style='text-align:right'>HW</th>"
        "<th style='text-align:right'>Signal</th>"
        "<th style='text-align:right'>Close</th>"
        "<th style='text-align:right'>Volume</th>"
        "</tr></thead>"
    )

    sorted_df = (
        df.sort_values(["ts", "kind", "ticker"], ascending=[False, True, True]).reset_index(drop=True)
        if not df.empty else df
    )

    if sorted_df.empty:
        latest_bar = None
        latest = sorted_df
    else:
        latest_bar = sorted_df["ts"].max()
        latest = sorted_df[sorted_df["ts"] == latest_bar].sort_values(["kind", "ticker"])

    counts_total = _kind_counts(sorted_df)
    counts_latest = _kind_counts(latest)
    n_total = sum(counts_total.values())
    n_latest = sum(counts_latest.values())

    latest_bar_str = _fmt_ts(latest_bar, tf) if latest_bar is not None else "—"

    latest_rows = "".join(_row(r, tf) for _, r in latest.iterrows())
    log_rows = "".join(_row(r, tf) for _, r in sorted_df.iterrows())

    counts_summary = (
        f"AL_OS {counts_total['AL_OS']} / AL {counts_total['AL']} / "
        f"SAT_OB {counts_total['SAT_OB']} / SAT {counts_total['SAT']}"
    )

    summary = {
        "n_total": n_total,
        "n_latest": n_latest,
        "counts_total": counts_total,
        "counts_latest": counts_latest,
        "latest_bar": latest_bar_str,
        "latest_tickers": {
            kind: latest[latest["kind"] == kind]["ticker"].tolist() if not latest.empty else []
            for kind in KIND_ORDER
        },
    }

    counts_pill_html = " ".join(
        f'<span class="tag" style="color:{KIND_BADGE[k][1]}">{k} {counts_latest[k]}</span>'
        for k in KIND_ORDER
    )

    body = f"""
<section class="hw-card" id="tf-{tf}">
  <h2>
    <span style="color:{TF_DOT[tf]};">●</span>
    {TF_LABELS[tf]} <span style="color:var(--text-muted);font-weight:400;font-size:0.85rem;">· {tf}</span>
    <span class="tag">son fire {html.escape(latest_bar_str)}</span>
    {counts_pill_html}
    <span class="tag">window {n_total}</span>
  </h2>

  <h3>Son Fire Barı Sinyalleri ({html.escape(latest_bar_str)})</h3>
  {_table(latest_rows, table_header, "Pencerede fire yok")}

  <details class="hw-collapse">
    <summary>Pencere event log ({n_total} event · {counts_summary}) — aç/kapat</summary>
    {_table(log_rows, table_header, "Pencerede event yok")}
  </details>
</section>
"""
    return body, summary


def render(asof: str, in_dir: Path, tfs: list[str]) -> tuple[str, dict]:
    sections: list[str] = []
    summaries: dict[str, dict] = {}
    asof_bars: list[str] = []

    for tf in tfs:
        path = in_dir / f"hwo_full_{tf}_scan_{asof}.csv"
        df = _load_csv(path)
        body, summary = _tf_section(tf, df)
        sections.append(body)
        summaries[tf] = summary
        if summary["latest_bar"] != "—":
            asof_bars.append(f"{tf}: {summary['latest_bar']}")

    grand_total = sum(s["n_total"] for s in summaries.values())
    grand_latest = sum(s["n_latest"] for s in summaries.values())

    nav_html = '<div class="hw-tab-nav">' + "".join(
        f'<a class="hw-tab" href="#tf-{tf}">'
        f'<span class="dot" style="background:{TF_DOT[tf]}"></span>'
        f'{TF_LABELS[tf]} <span class="cnt">{summaries[tf]["n_latest"]}</span>'
        f'</a>'
        for tf in tfs
    ) + '</div>'

    asof_meta_str = " · ".join(asof_bars) if asof_bars else "—"
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    picks_json = {
        "schema_version": 1,
        "system": "hwo_full",
        "asof": asof,
        "tfs": tfs,
        "tf_summaries": summaries,
        "grand_total": grand_total,
        "grand_latest": grand_latest,
    }
    picks_marker = (
        '<script id="hwo-full-data" type="application/json">'
        + json.dumps(picks_json, ensure_ascii=False)
        + '</script>'
    )

    body = f"""
<div class="aurora-bg">
  <div class="aurora-layer aurora-layer-1"></div>
  <div class="aurora-layer aurora-layer-2"></div>
  <div class="aurora-layer aurora-layer-3"></div>
</div>
<div class="mesh-overlay"></div>

<div class="briefing-container">

  <div class="nox-header" style="margin-bottom:1.2rem;">
    <div class="nox-logo">
      HWO FULL<span class="proj"> · MULTI-TF TARAMA</span>
      <span class="mode">hyperwave 7 · sma 3 · all crossings (small + big) · 5h/1d/1w</span>
    </div>
    <div class="nox-meta">
      As-of: <b>{html.escape(asof)}</b><br>
      Son bar fire: <b>{grand_latest}</b> · pencere toplam: <b>{grand_total}</b><br>
      generated {html.escape(now_str)}
    </div>
  </div>

  <div class="nox-stats">
""" + "".join(
        f'<div class="nox-stat"><span class="dot" style="background:{TF_DOT[tf]};"></span>'
        f'<span>{TF_LABELS[tf]}</span><span class="cnt">{summaries[tf]["n_latest"]}</span></div>'
        for tf in tfs
    ) + f"""
  </div>

  {nav_html}

  {"".join(sections)}

  <div class="hw-footer">
    Pipeline: <code>tools/hwo_full_scan.py</code> · render: <code>tools/hwo_full_html.py</code> ·
    HW = <code>oscmatrix.components.hyperwave</code> (length 7, sig_len 3, source close).<br>
    Veri: <code>output/extfeed_intraday_1h_3y_master.parquet</code> (BIST 607 evren, 3y).
    Resample: <code>mb_scanner.resample</code> (5h TV-aligned, 1w W-FRI).<br>
    Son bar timestamps: {html.escape(asof_meta_str)}<br>
    <b>Bu bir tarama çıktısıdır</b> — descriptive event log, trade-edge iddiası YOK.
    Pre-registered multi-TF backtest <code>hwo_mtf_v1</code> CLOSED_REJECTED
    (PF 1.028 vs random p05 1.319 · rank_pct 0.431).<br>
    Sister tool: <code>tools/hw_obos_scan.py</code> sadece big-dot (AL_OS / SAT_OB) yazar; bu tool ise small + big tüm crossings.
  </div>

</div>
"""

    page = f"""<!DOCTYPE html>
<html lang="tr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NOX HWO FULL Multi-TF · {html.escape(asof)}</title>
<style>
{_NOX_CSS}
{_LOCAL_CSS}
</style>
</head>
<body>
{picks_marker}
{body}
</body>
</html>
"""
    return page, picks_json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asof", required=True, help="YYYY-MM-DD; matches CSV filenames")
    ap.add_argument("--in-dir", default="output")
    ap.add_argument("--out", default=None,
                    help="default: output/hwo_full_scan_<asof>.html")
    ap.add_argument("--also-latest", action="store_true",
                    help="also write output/hwo_full_scan_latest.html")
    ap.add_argument("--tfs", nargs="+", default=TF_ORDER,
                    help="Subset of {5h,1d,1w}; default = all three")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_path = Path(args.out) if args.out else in_dir / f"hwo_full_scan_{args.asof}.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    page, _ = render(args.asof, in_dir, args.tfs)
    out_path.write_text(page, encoding="utf-8")
    print(f"[write] {out_path}")

    if args.also_latest:
        latest = out_path.parent / "hwo_full_scan_latest.html"
        latest.write_text(page, encoding="utf-8")
        print(f"[write] {latest}")


if __name__ == "__main__":
    main()
