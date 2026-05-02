"""HTML report for mb_scanner — briefing aesthetic, per-TF separate pages.

Emits one standalone HTML per timeframe (5h / 1d / 1w / 1M). Each page
has MB and BB tables stacked full-width (no squeeze) plus the cross-TF
stack panel and a TF-nav strip linking to sibling pages.

Public entry points:
    build_tf_html(tf, asof, ...) -> str   # one TF
    build_all(asof, ...) -> dict[str, str]  # {tf: html}
    build_index_html(asof, ...) -> str    # landing page

Driver: tools/mb_scanner_html.py
"""
from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from core.reports import _NOX_CSS

from .rank import OUT_DIR, build_cross_tf_index, rank_family

TF_LABELS = {
    "5h": "5 saatlik",
    "1d": "Günlük",
    "1w": "Haftalık",
    "1M": "Aylık",
}
TF_NOTES = {
    "5h": "TV-aligned 5h — 09:00 + 14:00 binleri (BIST seans-içi).",
    "1d": "Günlük bar (TR günü, 18:10 closing-price kapanışı).",
    "1w": "Haftalık bar — TV gibi Pazartesi (hafta-başı) etiketi gösterilir; data anchor Cuma kapanışı.",
    "1M": "Aylık bar — TV gibi ayın 1'i etiketi gösterilir; data anchor ayın son iş günü.",
}
TF_DOT = {
    "5h": "var(--nox-gold)",
    "1d": "var(--nox-green)",
    "1w": "var(--nox-blue)",
    "1M": "var(--nox-orange)",
}
TF_ORDER = ("5h", "1d", "1w", "1M")

STATE_LABEL = {
    "retest_bounce": "RETEST",
    "above_mb":      "ABOVE-MB",
    "mitigation_touch": "MIT-TOUCH",
}
STATE_CLASS = {
    "retest_bounce": "st-retest",
    "above_mb": "st-above",
    "mitigation_touch": "st-mit",
}

TV_BASE = "https://www.tradingview.com/chart/?symbol=BIST:"


def _file_for(tf: str, dated_label: str | None = None) -> str:
    """Path-safe filename for a TF page (1M → 1m on disk; URL-safe)."""
    slug = "1mo" if tf == "1M" else tf
    if dated_label:
        return f"mb_scanner_{slug}_{dated_label}.html"
    return f"mb_scanner_{slug}_latest.html"


_LOCAL_CSS = """
.mb-stack { display: grid; grid-template-columns: 1fr; gap: 1rem; margin-bottom: 1.5rem; }

.mb-card {
  background: rgba(199,189,190,0.07);
  backdrop-filter: blur(16px);
  -webkit-backdrop-filter: blur(16px);
  border: none;
  border-radius: 16px;
  padding: 0.9rem 1rem 1.1rem;
  position: relative;
  overflow-x: auto;
}
.mb-card .card-title {
  font-family: var(--font-display);
  font-size: 0.72rem; font-weight: 600;
  color: var(--text-muted);
  text-transform: uppercase; letter-spacing: 0.08em;
  margin-bottom: 0.55rem;
}
.mb-card .card-title b { color: var(--nox-gold); font-weight: 700; }
.mb-card .card-title .freq { color: var(--text-secondary); font-weight:500; }
.mb-card .card-title .n { color: var(--text-secondary); font-weight: 500; margin-left: 0.4rem; }

.mb-banner {
  background: rgba(122,143,165,0.08);
  border: 1px solid rgba(122,143,165,0.30);
  border-left: 3px solid var(--nox-blue);
  border-radius: 10px;
  padding: 0.7rem 1rem;
  margin-bottom: 1.0rem;
  font-size: 0.82rem;
  color: var(--text-primary);
  line-height: 1.5;
}
.mb-banner b { color: var(--nox-blue); }
.mb-banner.gold {
  background: rgba(201,169,110,0.07);
  border-color: rgba(201,169,110,0.30);
  border-left-color: var(--nox-gold);
}
.mb-banner.gold b { color: var(--nox-gold); }

.mb-tfnav {
  display: flex; flex-wrap: wrap; gap: 6px;
  margin-bottom: 1rem;
}
.mb-tfnav a {
  background: var(--bg-card);
  border: 1px solid var(--border-subtle);
  border-radius: 999px;
  padding: 7px 16px;
  font-family: var(--font-display);
  font-size: 0.78rem; font-weight: 500;
  color: var(--text-secondary);
  text-decoration: none;
  cursor: pointer; user-select: none;
  transition: all 0.15s;
  display: inline-flex; align-items: center; gap: 0.5rem;
}
.mb-tfnav a:hover { color: var(--text-primary); border-color: var(--border-dim); }
.mb-tfnav a.active {
  border-color: var(--nox-gold);
  background: var(--nox-gold-dim);
  color: var(--nox-gold);
}
.mb-tfnav a .tab-count {
  font-family: var(--font-mono);
  font-size: 0.68rem; font-weight: 700;
  color: var(--text-muted);
}
.mb-tfnav a.active .tab-count { color: var(--nox-gold); }

.mb-tf-note {
  font-family: var(--font-mono);
  font-size: 0.72rem;
  color: var(--text-muted);
  margin-bottom: 0.6rem;
  letter-spacing: 0.02em;
}

table.mb {
  width: 100%; border-collapse: collapse;
  font-size: 0.76rem;
  table-layout: auto;
}
table.mb th, table.mb td {
  padding: 6px 8px;
  border-bottom: 1px solid rgba(39,39,42,0.5);
  text-align: left;
  white-space: nowrap;
}
table.mb th {
  background: var(--bg-elevated);
  color: var(--text-muted);
  font-family: var(--font-display);
  font-size: 0.64rem; font-weight: 600;
  text-transform: uppercase; letter-spacing: 0.05em;
  position: sticky; top: 0;
}
table.mb td {
  font-family: var(--font-mono);
  color: var(--text-primary);
}
table.mb tr:hover { background: var(--bg-hover); }
table.mb td.tk {
  font-family: 'Inter', sans-serif;
  font-weight: 800; font-size: 0.86rem;
  color: #fff;
  letter-spacing: -0.01em;
}
table.mb td.tk a {
  color: #fff;
  text-decoration: none;
  border-bottom: 1px dashed rgba(255,255,255,0.25);
  transition: border-color 0.15s, color 0.15s;
}
table.mb td.tk a:hover {
  color: var(--nox-gold);
  border-bottom-color: var(--nox-gold);
}
table.mb td.score {
  color: var(--nox-gold); font-weight: 700;
}
table.mb td.num.pos { color: var(--nox-green); }
table.mb td.num.neg { color: var(--nox-red); }
table.mb td.muted { color: var(--text-muted); }

.st-pill {
  display: inline-block; padding: 2px 8px;
  border-radius: 999px;
  font-family: var(--font-display);
  font-size: 0.62rem; font-weight: 700;
  letter-spacing: 0.04em; text-transform: uppercase;
  border: 1px solid transparent;
}
.st-retest { background: rgba(201,169,110,0.18); color: var(--nox-gold); border-color: rgba(201,169,110,0.45); }
.st-above  { background: rgba(122,158,122,0.13); color: var(--nox-green); border-color: rgba(122,158,122,0.35); }
.st-mit    { background: rgba(122,143,165,0.12); color: var(--nox-blue); border-color: rgba(122,143,165,0.30); }

.rk-pill {
  display: inline-block; padding: 1px 6px;
  border-radius: 4px;
  font-family: var(--font-mono);
  font-size: 0.62rem; font-weight: 600;
  letter-spacing: 0.02em;
  border: 1px solid var(--border-subtle);
  color: var(--text-secondary);
  background: var(--bg-elevated);
}
.rk-deep { color: var(--nox-gold); border-color: rgba(201,169,110,0.45); background: rgba(201,169,110,0.10); }
.rk-shallow { color: var(--nox-blue); border-color: rgba(122,143,165,0.35); background: rgba(122,143,165,0.10); }
.rk-no { color: var(--text-muted); }

.tf-chip {
  display: inline-block; padding: 1px 5px;
  border-radius: 4px;
  font-family: var(--font-mono);
  font-size: 0.6rem; font-weight: 700;
  letter-spacing: 0.02em;
  border: 1px solid var(--border-subtle);
  color: var(--text-secondary);
  background: var(--bg-card);
  margin-right: 2px;
}
.tf-chip.mb { color: var(--nox-gold); border-color: rgba(201,169,110,0.30); }
.tf-chip.bb { color: var(--nox-orange); border-color: rgba(168,135,106,0.30); }

.stack-grid {
  display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
  gap: 0.5rem;
}
.stack-item {
  background: rgba(6,7,9,0.55);
  border: 1px solid var(--border-subtle);
  border-radius: 10px;
  padding: 0.5rem 0.7rem;
}
.stack-item .stk-tk {
  font-family: 'Inter', sans-serif;
  font-weight: 800; font-size: 1rem;
  color: #fff;
  margin-right: 0.45rem;
}
.stack-item .stk-tk a {
  color: #fff;
  text-decoration: none;
}
.stack-item .stk-tk a:hover { color: var(--nox-gold); }
.stack-item .stk-n {
  font-family: var(--font-mono);
  font-size: 0.7rem; font-weight: 700;
  color: var(--nox-gold);
  margin-right: 0.4rem;
}
.stack-item .stk-fams { font-family: var(--font-mono); font-size: 0.62rem; color: var(--text-muted); margin-top: 4px; line-height: 1.5; }

.mb-empty {
  text-align: center;
  font-family: var(--font-mono);
  color: var(--text-muted);
  padding: 1rem 0;
  font-size: 0.84rem;
}

.mb-footer {
  margin-top: 1.5rem;
  padding-top: 1rem;
  border-top: 1px solid var(--border-subtle);
  font-family: var(--font-mono);
  font-size: 0.7rem;
  color: var(--text-muted);
  line-height: 1.6;
}
.mb-footer code {
  background: var(--bg-elevated);
  padding: 1px 6px; border-radius: 3px;
  color: var(--text-secondary);
}

.mb-index-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 1rem; margin-bottom: 1.5rem;
}
.mb-index-tile {
  background: rgba(199,189,190,0.07);
  border: 1px solid var(--border-subtle);
  border-radius: 14px;
  padding: 1rem 1.1rem;
  text-decoration: none;
  color: inherit;
  transition: border-color 0.15s, transform 0.15s;
  display: block;
}
.mb-index-tile:hover { border-color: var(--nox-gold); transform: translateY(-2px); }
.mb-index-tile .tile-tf {
  font-family: var(--font-display);
  font-size: 0.66rem; font-weight: 600;
  color: var(--text-muted);
  text-transform: uppercase; letter-spacing: 0.08em;
}
.mb-index-tile .tile-name {
  font-family: 'Inter', sans-serif;
  font-size: 1.4rem; font-weight: 800;
  color: #fff;
  letter-spacing: -0.01em;
  margin: 0.25rem 0;
}
.mb-index-tile .tile-n {
  font-family: var(--font-mono);
  font-size: 0.84rem; color: var(--nox-gold); font-weight: 700;
}
.mb-index-tile .tile-note {
  font-family: var(--font-mono);
  font-size: 0.66rem; color: var(--text-secondary);
  margin-top: 0.4rem; letter-spacing: 0.02em;
}
"""


# ── helpers ─────────────────────────────────────────────────────────

def _fmt_num(v, prec=2):
    if v is None or pd.isna(v):
        return "—"
    return f"{v:.{prec}f}"


def _fmt_date(v):
    if v is None or pd.isna(v):
        return "—"
    ts = pd.Timestamp(v)
    return ts.strftime("%Y-%m-%d")


def _fmt_pivot_date(v, tf: str | None = None) -> str:
    """TV-aligned date label.

    1w: our parquet stores W-FRI (week-end Friday). TV labels weekly bars
    with week-start (Monday, or first trading day). We approximate by
    subtracting 4 days. Bayram/holiday weeks where Monday is closed will
    display Monday anyway — minor cosmetic miss, not a data error.

    1M: our parquet stores BME (business month-end). TV labels monthly
    bars with month-start (1st of month). We snap to day=1.
    """
    if v is None or pd.isna(v):
        return "—"
    ts = pd.Timestamp(v)
    if tf == "1w":
        ts = ts - pd.Timedelta(days=4)
    elif tf == "1M":
        ts = ts.replace(day=1)
    return ts.strftime("%Y-%m-%d")


def _state_pill(state: str) -> str:
    label = STATE_LABEL.get(state, state.upper())
    klass = STATE_CLASS.get(state, "st-mit")
    return f'<span class="st-pill {klass}">{html.escape(label)}</span>'


def _retest_pill(rk: str) -> str:
    if not rk:
        return ""
    klass = {"deep_touch": "rk-deep", "shallow_touch": "rk-shallow", "no_touch": "rk-no"}.get(rk, "rk-no")
    label = rk.replace("_", " ")
    return f'<span class="rk-pill {klass}">{html.escape(label)}</span>'


def _tf_chips(also_fires_in: str) -> str:
    if not also_fires_in:
        return '<span class="muted">—</span>'
    chips = []
    for fam in sorted(also_fires_in.split(",")):
        if not fam:
            continue
        klass = "mb" if fam.startswith("mb_") else "bb"
        chips.append(f'<span class="tf-chip {klass}">{html.escape(fam)}</span>')
    return "".join(chips) or '<span class="muted">—</span>'


def _signal_date(row: pd.Series, tf: str | None = None) -> str:
    """AL signal day. retest_bounce/extended → retest_bar_date; else hh_bar_date."""
    state = row["signal_state"]
    if state in ("retest_bounce", "extended") and pd.notna(row.get("retest_bar_date")):
        return _fmt_pivot_date(row["retest_bar_date"], tf)
    return _fmt_pivot_date(row.get("hh_bar_date"), tf)


def _bos_class(v) -> str:
    if v is None or pd.isna(v):
        return ""
    return "pos" if float(v) >= 0 else "neg"


def _pivot_tooltip(row: pd.Series, tf: str | None = None) -> str:
    parts = [
        f"LL: {_fmt_pivot_date(row.get('ll_bar_date'), tf)}",
        f"LH: {_fmt_pivot_date(row.get('lh_bar_date'), tf)}",
        f"HL: {_fmt_pivot_date(row.get('hl_bar_date'), tf)}",
        f"HH: {_fmt_pivot_date(row.get('hh_bar_date'), tf)}",
    ]
    rt = row.get("retest_bar_date")
    if rt is not None and pd.notna(rt):
        parts.append(f"Retest: {_fmt_pivot_date(rt, tf)}")
    zh = row.get("zone_high")
    zl = row.get("zone_low")
    if pd.notna(zh) and pd.notna(zl):
        parts.append(f"Zone: {_fmt_num(zl, 2)}–{_fmt_num(zh, 2)}")
    return " | ".join(parts)


def _ticker_link(ticker: str, tooltip: str) -> str:
    safe_t = html.escape(ticker)
    safe_tip = html.escape(tooltip, quote=True)
    return (
        f'<a href="{TV_BASE}{safe_t}" target="_blank" rel="noopener" '
        f'title="{safe_tip}">{safe_t}</a>'
    )


def _ranked_table(df: pd.DataFrame, *, top: int = 25, tf: str | None = None) -> str:
    if df is None or df.empty:
        return '<div class="mb-empty">aktif sinyal yok</div>'

    df = df.head(top).reset_index(drop=True)
    rows_html: list[str] = []
    for i in range(len(df)):
        row = df.iloc[i]
        ticker = str(row["ticker"])
        score = float(row["score"])
        bos_atr = row.get("bos_distance_atr")
        zone_age = int(row["zone_age_bars"]) if pd.notna(row.get("zone_age_bars")) else 0
        n_q = int(row["n_active_quartets"]) if pd.notna(row.get("n_active_quartets")) else 1
        tooltip = _pivot_tooltip(row, tf)
        rows_html.append(
            "<tr>"
            f"<td class='muted'>{i+1}</td>"
            f"<td class='tk'>{_ticker_link(ticker, tooltip)}</td>"
            f"<td class='score'>{score:.2f}</td>"
            f"<td>{_state_pill(row['signal_state'])}</td>"
            f"<td class='num {_bos_class(bos_atr)}'>{_fmt_num(bos_atr, 2)}</td>"
            f"<td>{zone_age}</td>"
            f"<td>{_retest_pill(row.get('retest_kind') or '')}</td>"
            f"<td>{_signal_date(row, tf)}</td>"
            f"<td>{n_q}</td>"
            f"<td class='num'>{_fmt_num(row.get('asof_close'), 2)}</td>"
            f"<td>{_tf_chips(row.get('also_fires_in') or '')}</td>"
            "</tr>"
        )
    body = "\n".join(rows_html)
    return (
        '<table class="mb">'
        '<thead><tr>'
        '<th>#</th><th>Ticker</th><th>Score</th><th>State</th>'
        '<th>BoS·ATR</th><th>Age</th><th>Retest</th><th>AL günü</th>'
        '<th>nQ</th><th>Close</th><th>Cross-TF</th>'
        '</tr></thead>'
        f'<tbody>{body}</tbody></table>'
    )


def _stack_card(cross_tf: dict[str, set[str]], min_count: int = 3, top_n: int = 18) -> str:
    items = sorted(cross_tf.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    items = [(t, fams) for t, fams in items if len(fams) >= min_count]
    if not items:
        return ""
    blocks: list[str] = []
    for ticker, fams in items[:top_n]:
        chips = "".join(
            f'<span class="tf-chip {"mb" if f.startswith("mb_") else "bb"}">{html.escape(f)}</span>'
            for f in sorted(fams)
        )
        link = (
            f'<a href="{TV_BASE}{html.escape(ticker)}" target="_blank" rel="noopener">'
            f'{html.escape(ticker)}</a>'
        )
        blocks.append(
            '<div class="stack-item">'
            f'<span class="stk-tk">{link}</span>'
            f'<span class="stk-n">×{len(fams)}</span>'
            f'<div class="stk-fams">{chips}</div>'
            '</div>'
        )
    return (
        '<div class="mb-card">'
        f'<div class="card-title">Cross-TF Stack <b>· ≥{min_count} aile</b>'
        f'<span class="n">— top {min(top_n, len(items))} / {len(items)} ticker</span></div>'
        f'<div class="stack-grid">{"".join(blocks)}</div>'
        '</div>'
    )


def _tf_nav(active_tf: str | None, tf_totals: dict[str, int]) -> str:
    items: list[str] = []
    items.append(
        f'<a href="mb_scanner_scan_latest.html" class="{"" if active_tf else "active"}">'
        f'<span>Index</span></a>'
    )
    for tf in TF_ORDER:
        klass = "active" if tf == active_tf else ""
        n = tf_totals.get(tf, 0)
        href = _file_for(tf)
        items.append(
            f'<a href="{href}" class="{klass}">'
            f'<span>{html.escape(TF_LABELS[tf])}</span>'
            f'<span class="tab-count">N={n}</span></a>'
        )
    return f'<div class="mb-tfnav">{"".join(items)}</div>'


# ── public ──────────────────────────────────────────────────────────

def _gather_ranked(out_dir: Path) -> tuple[dict[str, pd.DataFrame], dict[str, int], dict[str, set[str]]]:
    cross_tf = build_cross_tf_index(out_dir=out_dir)
    ranked: dict[str, pd.DataFrame] = {}
    fam_n: dict[str, int] = {}
    for fam in (
        "mb_5h", "bb_5h", "mb_1d", "bb_1d",
        "mb_1w", "bb_1w", "mb_1M", "bb_1M",
    ):
        df = rank_family(fam, cross_tf, out_dir=out_dir)
        ranked[fam] = df
        fam_n[fam] = 0 if df is None or df.empty else len(df)
    return ranked, fam_n, cross_tf


def _shell(
    *,
    page_title: str,
    asof_label: str,
    grand_total: int,
    tf_totals: dict[str, int],
    nav_html: str,
    body_html: str,
    picks_marker: str,
    now_str: str,
) -> str:
    return f"""<!DOCTYPE html>
<html lang="tr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{html.escape(page_title)}</title>
<style>
{_NOX_CSS}

.briefing-container {{
  position: relative; z-index: 1;
  max-width: 1280px; margin: 0 auto;
  padding: 0 1.5rem 2rem;
}}

{_LOCAL_CSS}
</style>
</head>
<body>
{picks_marker}

<div class="aurora-bg">
  <div class="aurora-layer aurora-layer-1"></div>
  <div class="aurora-layer aurora-layer-2"></div>
  <div class="aurora-layer aurora-layer-3"></div>
</div>
<div class="mesh-overlay"></div>

<div class="briefing-container">

  <div class="nox-header" style="margin-bottom:1.2rem;">
    <div class="nox-logo">
      MB/BB SCANNER<span class="proj"> · ICT/SMC</span>
      <span class="mode">strict MSS · close-only · multi-quartet</span>
    </div>
    <div class="nox-meta">
      As-of: <b>{asof_label}</b><br>
      Aktif setup: <b>{grand_total}</b><br>
      generated {now_str}
    </div>
  </div>

  <div class="nox-stats">
    <div class="nox-stat"><span class="dot" style="background:var(--nox-gold);"></span>
      <span>5-saatlik</span><span class="cnt">{tf_totals.get('5h', 0)}</span></div>
    <div class="nox-stat"><span class="dot" style="background:var(--nox-green);"></span>
      <span>Günlük</span><span class="cnt">{tf_totals.get('1d', 0)}</span></div>
    <div class="nox-stat"><span class="dot" style="background:var(--nox-blue);"></span>
      <span>Haftalık</span><span class="cnt">{tf_totals.get('1w', 0)}</span></div>
    <div class="nox-stat"><span class="dot" style="background:var(--nox-orange);"></span>
      <span>Aylık</span><span class="cnt">{tf_totals.get('1M', 0)}</span></div>
  </div>

  {nav_html}

  {body_html}

  <div class="mb-footer">
    Pipeline: <code>mb_scanner/</code> · close-only pivots · multi-quartet schema v0.3 ·
    8 aile (mb/bb × 5h/1d/1w/1M).<br>
    Driver: <code>tools/mb_scanner_html.py</code> · scoring: <code>mb_scanner/rank.py</code>.<br>
    Veri: <code>output/extfeed_intraday_1h_3y_master.parquet</code> (3y, BIST 607 evren).
  </div>

</div>

</body>
</html>
"""


def build_tf_html(
    tf: str,
    asof: pd.Timestamp | str | None = None,
    out_dir: Path = OUT_DIR,
    top: int = 25,
    *,
    ranked: dict[str, pd.DataFrame] | None = None,
    fam_n: dict[str, int] | None = None,
    cross_tf: dict[str, set[str]] | None = None,
) -> str:
    """Build standalone HTML for a single timeframe."""
    if tf not in TF_ORDER:
        raise ValueError(f"Unknown tf {tf!r}; expected one of {TF_ORDER}")

    if ranked is None or fam_n is None or cross_tf is None:
        ranked, fam_n, cross_tf = _gather_ranked(out_dir)

    tf_totals = {t: fam_n.get(f"mb_{t}", 0) + fam_n.get(f"bb_{t}", 0) for t in TF_ORDER}
    grand_total = sum(tf_totals.values())

    asof_label = "—"
    if asof is not None:
        try:
            asof_label = pd.Timestamp(asof).strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            asof_label = str(asof)

    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    mb_df = ranked.get(f"mb_{tf}")
    bb_df = ranked.get(f"bb_{tf}")
    n_mb = 0 if mb_df is None or mb_df.empty else len(mb_df)
    n_bb = 0 if bb_df is None or bb_df.empty else len(bb_df)

    note = TF_NOTES.get(tf, "")
    body_html = f"""
  <div class="mb-banner">
    <b>RESEARCH WATCHLIST · NOT LIVE.</b>
    Strict MSS-validated LL→LH→HL→HH quartet'i + ICT/SMC mitigation block (MB)
    veya breaker block (BB) zone'una retest. Close-only fractal pivots
    (n=2). State'ler: <b>RETEST</b> (taze reclaim), <b>ABOVE-MB</b>
    (zone'a değmeden devam), <b>MIT-TOUCH</b> (zone'a değmiş, henüz
    bullish reclaim yok). Score = freshness × state weight × BoS penalty
    × retest kind × nesting bonus. AL günü = retest_bar_date (varsa) veya
    hh_bar_date (BoS günü). Ticker üstüne tıkla → TV grafik; hover et → pivot tarihleri.
  </div>

  {_stack_card(cross_tf, min_count=3, top_n=18)}

  <div class="mb-tf-note">{html.escape(TF_LABELS[tf])} — {html.escape(note)}</div>
  <div class="mb-stack">
    <div class="mb-card">
      <div class="card-title">MB · <b>mb_{html.escape(tf)}</b><span class="n">N={n_mb}</span></div>
      {_ranked_table(mb_df, top=top, tf=tf)}
    </div>
    <div class="mb-card">
      <div class="card-title">BB · <b>bb_{html.escape(tf)}</b><span class="n">N={n_bb}</span></div>
      {_ranked_table(bb_df, top=top, tf=tf)}
    </div>
  </div>
"""

    picks_json = {
        "schema_version": 2,
        "system": "mb_scanner",
        "page": tf,
        "asof": asof_label,
        "fam_n": fam_n,
        "tf_totals": tf_totals,
        "stack_count": len([1 for fams in cross_tf.values() if len(fams) >= 3]),
    }
    picks_marker = (
        '<script id="mb-scanner-data" type="application/json">'
        + json.dumps(picks_json, ensure_ascii=False)
        + '</script>'
    )

    return _shell(
        page_title=f"NOX MB/BB Scanner · {TF_LABELS[tf]} — {asof_label}",
        asof_label=asof_label,
        grand_total=grand_total,
        tf_totals=tf_totals,
        nav_html=_tf_nav(tf, tf_totals),
        body_html=body_html,
        picks_marker=picks_marker,
        now_str=now_str,
    )


def build_index_html(
    asof: pd.Timestamp | str | None = None,
    out_dir: Path = OUT_DIR,
    *,
    ranked: dict[str, pd.DataFrame] | None = None,
    fam_n: dict[str, int] | None = None,
    cross_tf: dict[str, set[str]] | None = None,
) -> str:
    """Landing page linking to all 4 TF pages + showing cross-TF stack."""
    if ranked is None or fam_n is None or cross_tf is None:
        ranked, fam_n, cross_tf = _gather_ranked(out_dir)

    tf_totals = {t: fam_n.get(f"mb_{t}", 0) + fam_n.get(f"bb_{t}", 0) for t in TF_ORDER}
    grand_total = sum(tf_totals.values())

    asof_label = "—"
    if asof is not None:
        try:
            asof_label = pd.Timestamp(asof).strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            asof_label = str(asof)

    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    tiles: list[str] = []
    for tf in TF_ORDER:
        href = _file_for(tf)
        n = tf_totals.get(tf, 0)
        n_mb = fam_n.get(f"mb_{tf}", 0)
        n_bb = fam_n.get(f"bb_{tf}", 0)
        note = TF_NOTES.get(tf, "")
        tiles.append(
            f'<a class="mb-index-tile" href="{href}">'
            f'<div class="tile-tf" style="color:{TF_DOT[tf]};">{html.escape(tf)}</div>'
            f'<div class="tile-name">{html.escape(TF_LABELS[tf])}</div>'
            f'<div class="tile-n">N={n} · MB {n_mb} · BB {n_bb}</div>'
            f'<div class="tile-note">{html.escape(note)}</div>'
            f'</a>'
        )

    body_html = f"""
  <div class="mb-banner gold">
    <b>NOX MB/BB Scanner</b> — TF başına ayrı sayfalar. Cross-TF stack panel
    ≥3 aile fire eden ticker'ları topluyor. Ticker üstüne tıkla → TV grafik;
    hover et → pivot tarihleri. Score = freshness × state × BoS · retest · nesting.
  </div>

  <div class="mb-index-grid">
    {"".join(tiles)}
  </div>

  {_stack_card(cross_tf, min_count=3, top_n=24)}
"""

    picks_json = {
        "schema_version": 2,
        "system": "mb_scanner",
        "page": "index",
        "asof": asof_label,
        "fam_n": fam_n,
        "tf_totals": tf_totals,
        "stack_count": len([1 for fams in cross_tf.values() if len(fams) >= 3]),
    }
    picks_marker = (
        '<script id="mb-scanner-data" type="application/json">'
        + json.dumps(picks_json, ensure_ascii=False)
        + '</script>'
    )

    return _shell(
        page_title=f"NOX MB/BB Scanner · Index — {asof_label}",
        asof_label=asof_label,
        grand_total=grand_total,
        tf_totals=tf_totals,
        nav_html=_tf_nav(None, tf_totals),
        body_html=body_html,
        picks_marker=picks_marker,
        now_str=now_str,
    )


def build_all(
    asof: pd.Timestamp | str | None = None,
    out_dir: Path = OUT_DIR,
    top: int = 25,
) -> dict[str, str]:
    """Build all 4 TF HTMLs + index. Returns {key: html} where key ∈ TF_ORDER ∪ {'index'}."""
    ranked, fam_n, cross_tf = _gather_ranked(out_dir)
    pages: dict[str, str] = {}
    for tf in TF_ORDER:
        pages[tf] = build_tf_html(
            tf, asof=asof, out_dir=out_dir, top=top,
            ranked=ranked, fam_n=fam_n, cross_tf=cross_tf,
        )
    pages["index"] = build_index_html(
        asof=asof, out_dir=out_dir,
        ranked=ranked, fam_n=fam_n, cross_tf=cross_tf,
    )
    return pages
