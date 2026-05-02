"""HTML report for channel_break — briefing aesthetic, per-TF separate pages.

Mirrors mb_scanner/html_report layout. Reads channel_break_<fam>.parquet
+ pending_triangle_<fam>.parquet × 4 from out_dir.

Public:
    build_tf_html(tf, asof, ...) -> str
    build_index_html(asof, ...) -> str
    build_all(asof, ...) -> dict[str, str]   # {tf|'index': html}
"""
from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from core.reports import _NOX_CSS

OUT_DIR = Path("output")

TF_ORDER = ("5h", "1d", "1w", "1M")
TF_TO_FAM = {tf: f"ch_{tf}" for tf in TF_ORDER}

TF_LABELS = {
    "5h": "5 saatlik",
    "1d": "Günlük",
    "1w": "Haftalık",
    "1M": "Aylık",
}
TF_NOTES = {
    "5h": "TV-aligned 5h — 09:00 + 14:00 binleri (BIST seans-içi).",
    "1d": "Günlük bar (TR günü, 18:10 closing-price kapanışı).",
    "1w": "Haftalık bar — TV gibi Pazartesi etiketi; data anchor Cuma kapanışı.",
    "1M": "Aylık bar — TV gibi ayın 1'i etiketi; data anchor ayın son iş günü.",
}
TF_DOT = {
    "5h": "var(--nox-gold)",
    "1d": "var(--nox-green)",
    "1w": "var(--nox-blue)",
    "1M": "var(--nox-orange)",
}

STATE_LABEL = {
    "trigger":      "TRIGGER",
    "pre_breakout": "PRE-BREAK",
    "extended":     "EXTENDED",
}
STATE_CLASS = {
    "trigger":      "st-trig",
    "pre_breakout": "st-pre",
    "extended":     "st-ext",
}
SLOPE_LABEL = {"asc": "↗ ASC", "desc": "↘ DESC", "flat": "→ FLAT"}
SLOPE_CLASS = {"asc": "sl-asc", "desc": "sl-desc", "flat": "sl-flat"}

FIT_CLASS = {"tight": "fit-tight", "loose": "fit-loose", "rough": "fit-rough"}

TV_BASE = "https://www.tradingview.com/chart/?symbol=BIST:"


def _file_for(tf: str, dated_label: str | None = None) -> str:
    slug = "1mo" if tf == "1M" else tf
    if dated_label:
        return f"channel_break_{slug}_{dated_label}.html"
    return f"channel_break_{slug}_latest.html"


_LOCAL_CSS = """
.cb-stack { display: grid; grid-template-columns: 1fr; gap: 1rem; margin-bottom: 1.5rem; }

.cb-card {
  background: rgba(199,189,190,0.07);
  backdrop-filter: blur(16px);
  -webkit-backdrop-filter: blur(16px);
  border: none;
  border-radius: 16px;
  padding: 0.9rem 1rem 1.1rem;
  position: relative;
  overflow-x: auto;
}
.cb-card .card-title {
  font-family: var(--font-display);
  font-size: 0.72rem; font-weight: 600;
  color: var(--text-muted);
  text-transform: uppercase; letter-spacing: 0.08em;
  margin-bottom: 0.55rem;
}
.cb-card .card-title b { color: var(--nox-gold); font-weight: 700; }
.cb-card .card-title .n { color: var(--text-secondary); font-weight: 500; margin-left: 0.4rem; }

.cb-banner {
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
.cb-banner b { color: var(--nox-blue); }
.cb-banner.gold {
  background: rgba(201,169,110,0.07);
  border-color: rgba(201,169,110,0.30);
  border-left-color: var(--nox-gold);
}
.cb-banner.gold b { color: var(--nox-gold); }

.cb-tfnav {
  display: flex; flex-wrap: wrap; gap: 6px;
  margin-bottom: 1rem;
}
.cb-tfnav a {
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
.cb-tfnav a:hover { color: var(--text-primary); border-color: var(--border-dim); }
.cb-tfnav a.active {
  border-color: var(--nox-gold);
  background: var(--nox-gold-dim);
  color: var(--nox-gold);
}
.cb-tfnav a .tab-count {
  font-family: var(--font-mono);
  font-size: 0.68rem; font-weight: 700;
  color: var(--text-muted);
}
.cb-tfnav a.active .tab-count { color: var(--nox-gold); }

.cb-tf-note {
  font-family: var(--font-mono);
  font-size: 0.72rem;
  color: var(--text-muted);
  margin-bottom: 0.6rem;
  letter-spacing: 0.02em;
}

table.cb {
  width: 100%; border-collapse: collapse;
  font-size: 0.76rem;
  table-layout: auto;
}
table.cb th, table.cb td {
  padding: 6px 8px;
  border-bottom: 1px solid rgba(39,39,42,0.5);
  text-align: left;
  white-space: nowrap;
}
table.cb th {
  background: var(--bg-elevated);
  color: var(--text-muted);
  font-family: var(--font-display);
  font-size: 0.64rem; font-weight: 600;
  text-transform: uppercase; letter-spacing: 0.05em;
  position: sticky; top: 0;
}
table.cb td {
  font-family: var(--font-mono);
  color: var(--text-primary);
}
table.cb tr:hover { background: var(--bg-hover); }
table.cb td.tk {
  font-family: 'Inter', sans-serif;
  font-weight: 800; font-size: 0.86rem;
  color: #fff;
  letter-spacing: -0.01em;
}
table.cb td.tk a {
  color: #fff;
  text-decoration: none;
  border-bottom: 1px dashed rgba(255,255,255,0.25);
  transition: border-color 0.15s, color 0.15s;
}
table.cb td.tk a:hover {
  color: var(--nox-gold);
  border-bottom-color: var(--nox-gold);
}
table.cb td.num.pos { color: var(--nox-green); }
table.cb td.num.neg { color: var(--nox-red); }
table.cb td.muted { color: var(--text-muted); }

.st-pill, .sl-pill, .fit-pill {
  display: inline-block; padding: 2px 8px;
  border-radius: 999px;
  font-family: var(--font-display);
  font-size: 0.62rem; font-weight: 700;
  letter-spacing: 0.04em; text-transform: uppercase;
  border: 1px solid transparent;
}
.st-trig { background: rgba(122,158,122,0.14); color: var(--nox-green); border-color: rgba(122,158,122,0.40); }
.st-pre  { background: rgba(201,169,110,0.16); color: var(--nox-gold);  border-color: rgba(201,169,110,0.42); }
.st-ext  { background: rgba(122,143,165,0.12); color: var(--nox-blue);  border-color: rgba(122,143,165,0.30); }

.sl-asc  { background: rgba(122,158,122,0.10); color: var(--nox-green); border-color: rgba(122,158,122,0.30); }
.sl-desc { background: rgba(168,106,106,0.12); color: var(--nox-red);   border-color: rgba(168,106,106,0.32); }
.sl-flat { background: rgba(199,189,190,0.08); color: var(--text-secondary); border-color: var(--border-subtle); }

.fit-pill {
  font-size: 0.58rem; padding: 1px 6px; border-radius: 4px;
}
.fit-tight { background: rgba(122,158,122,0.10); color: var(--nox-green); border-color: rgba(122,158,122,0.30); }
.fit-loose { background: rgba(122,143,165,0.10); color: var(--nox-blue); border-color: rgba(122,143,165,0.30); }
.fit-rough { background: rgba(168,106,106,0.10); color: var(--nox-red); border-color: rgba(168,106,106,0.30); }

.tier-pill {
  display: inline-block; padding: 1px 7px; border-radius: 4px;
  font-family: var(--font-display);
  font-size: 0.62rem; font-weight: 800;
  border: 1px solid var(--border-subtle);
}
.tier-A { color: var(--nox-gold); border-color: rgba(201,169,110,0.45); background: rgba(201,169,110,0.10); }
.tier-B { color: var(--text-secondary); }

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
.tf-chip.ch { color: var(--nox-gold); border-color: rgba(201,169,110,0.30); }

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
.stack-item .stk-tk a { color: #fff; text-decoration: none; }
.stack-item .stk-tk a:hover { color: var(--nox-gold); }
.stack-item .stk-n {
  font-family: var(--font-mono);
  font-size: 0.7rem; font-weight: 700;
  color: var(--nox-gold);
  margin-right: 0.4rem;
}
.stack-item .stk-fams { font-family: var(--font-mono); font-size: 0.62rem; color: var(--text-muted); margin-top: 4px; line-height: 1.5; }

.cb-empty {
  text-align: center;
  font-family: var(--font-mono);
  color: var(--text-muted);
  padding: 1rem 0;
  font-size: 0.84rem;
}

.cb-footer {
  margin-top: 1.5rem;
  padding-top: 1rem;
  border-top: 1px solid var(--border-subtle);
  font-family: var(--font-mono);
  font-size: 0.7rem;
  color: var(--text-muted);
  line-height: 1.6;
}
.cb-footer code {
  background: var(--bg-elevated);
  padding: 1px 6px; border-radius: 3px;
  color: var(--text-secondary);
}

.cb-index-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 1rem; margin-bottom: 1.5rem;
}
.cb-index-tile {
  background: rgba(199,189,190,0.07);
  border: 1px solid var(--border-subtle);
  border-radius: 14px;
  padding: 1rem 1.1rem;
  text-decoration: none;
  color: inherit;
  transition: border-color 0.15s, transform 0.15s;
  display: block;
}
.cb-index-tile:hover { border-color: var(--nox-gold); transform: translateY(-2px); }
.cb-index-tile .tile-tf {
  font-family: var(--font-display);
  font-size: 0.66rem; font-weight: 600;
  color: var(--text-muted);
  text-transform: uppercase; letter-spacing: 0.08em;
}
.cb-index-tile .tile-name {
  font-family: 'Inter', sans-serif;
  font-size: 1.4rem; font-weight: 800;
  color: #fff;
  letter-spacing: -0.01em;
  margin: 0.25rem 0;
}
.cb-index-tile .tile-n {
  font-family: var(--font-mono);
  font-size: 0.84rem; color: var(--nox-gold); font-weight: 700;
}
.cb-index-tile .tile-note {
  font-family: var(--font-mono);
  font-size: 0.66rem; color: var(--text-secondary);
  margin-top: 0.4rem; letter-spacing: 0.02em;
}

.tri-summary {
  background: rgba(168,135,106,0.06);
  border: 1px solid rgba(168,135,106,0.25);
  border-left: 3px solid var(--nox-orange);
  border-radius: 10px;
  padding: 0.7rem 1rem;
  font-family: var(--font-mono);
  font-size: 0.74rem;
  color: var(--text-secondary);
  margin-bottom: 1.0rem;
}
.tri-summary b { color: var(--nox-orange); font-family: var(--font-display); }
.tri-summary .kind {
  display: inline-block;
  background: var(--bg-elevated);
  border: 1px solid var(--border-subtle);
  border-radius: 4px;
  padding: 1px 6px;
  margin: 0 4px;
  font-size: 0.68rem;
  color: var(--text-primary);
}
"""


# ---------------------------------------------------------------- helpers

def _fmt_num(v, prec=2):
    if v is None or pd.isna(v):
        return "—"
    return f"{v:.{prec}f}"


def _fmt_pct(v, prec=1):
    if v is None or pd.isna(v):
        return "—"
    return f"{v*100:.{prec}f}%"


def _fmt_date(v):
    if v is None or pd.isna(v):
        return "—"
    return pd.Timestamp(v).strftime("%Y-%m-%d")


def _state_pill(state: str) -> str:
    label = STATE_LABEL.get(state, state.upper())
    klass = STATE_CLASS.get(state, "st-ext")
    return f'<span class="st-pill {klass}">{html.escape(label)}</span>'


def _slope_pill(slope: str) -> str:
    label = SLOPE_LABEL.get(slope, slope.upper())
    klass = SLOPE_CLASS.get(slope, "sl-flat")
    return f'<span class="sl-pill {klass}">{html.escape(label)}</span>'


def _fit_pill(fq: str) -> str:
    klass = FIT_CLASS.get(fq, "fit-loose")
    return f'<span class="fit-pill {klass}">{html.escape(fq or "—")}</span>'


def _tier_pill(is_a: bool) -> str:
    if is_a:
        return '<span class="tier-pill tier-A">A</span>'
    return '<span class="tier-pill tier-B">B</span>'


def _ticker_link(ticker: str, tooltip: str = "") -> str:
    safe_t = html.escape(ticker)
    if tooltip:
        safe_tip = html.escape(tooltip, quote=True)
        return (
            f'<a href="{TV_BASE}{safe_t}" target="_blank" rel="noopener" '
            f'title="{safe_tip}">{safe_t}</a>'
        )
    return f'<a href="{TV_BASE}{safe_t}" target="_blank" rel="noopener">{safe_t}</a>'


def _row_tooltip(row: pd.Series) -> str:
    parts = [
        f"First pivot: {_fmt_date(row.get('first_pivot_bar_date'))}",
        f"Last pivot: {_fmt_date(row.get('last_pivot_bar_date'))}",
        f"H/L pivots: {int(row.get('n_pivots_upper') or 0)}/{int(row.get('n_pivots_lower') or 0)}",
        f"Upper@asof: {_fmt_num(row.get('upper_at_asof'), 2)}",
        f"Lower@asof: {_fmt_num(row.get('lower_at_asof'), 2)}",
        f"Parallelism: {_fmt_num(row.get('parallelism'), 2)}",
    ]
    if pd.notna(row.get("breakout_bar_date")):
        parts.append(f"Break: {_fmt_date(row.get('breakout_bar_date'))}")
    return " | ".join(parts)


# ---------------------------------------------------------------- rank/sort

_STATE_ORD = {"trigger": 0, "pre_breakout": 1, "extended": 2}
_FIT_ORD = {"tight": 0, "loose": 1, "rough": 2}


def _rank_channels(df: pd.DataFrame) -> pd.DataFrame:
    """Sort accepted channels by tier_a desc, state, fit, channel age asc."""
    if df is None or df.empty:
        return df
    df = df.copy()
    df["_tier_ord"] = (~df["tier_a"].fillna(False).astype(bool)).astype(int)
    df["_state_ord"] = df["signal_state"].map(_STATE_ORD).fillna(9).astype(int)
    df["_fit_ord"] = df["fit_quality"].map(_FIT_ORD).fillna(9).astype(int)
    df["_age"] = df["channel_age_bars"].fillna(0).astype(int)
    df = df.sort_values(
        ["_tier_ord", "_state_ord", "_fit_ord", "_age"],
        ascending=[True, True, True, True],
    ).reset_index(drop=True)
    return df.drop(columns=["_tier_ord", "_state_ord", "_fit_ord", "_age"])


# ---------------------------------------------------------------- table

def _channel_table(df: pd.DataFrame, *, top: int = 30) -> str:
    if df is None or df.empty:
        return '<div class="cb-empty">aktif kanal yok</div>'
    df = df.head(top).reset_index(drop=True)
    rows: list[str] = []
    for i in range(len(df)):
        r = df.iloc[i]
        ticker = str(r["ticker"])
        tip = _row_tooltip(r)
        rows.append(
            "<tr>"
            f"<td class='muted'>{i+1}</td>"
            f"<td class='tk'>{_ticker_link(ticker, tip)}</td>"
            f"<td>{_state_pill(r['signal_state'])}</td>"
            f"<td>{_slope_pill(r.get('slope_class') or 'flat')}</td>"
            f"<td>{_tier_pill(bool(r.get('tier_a')))}</td>"
            f"<td class='num'>{_fmt_pct(r.get('channel_width_pct'))}</td>"
            f"<td class='num'>{int(r.get('n_pivots_upper') or 0)}/{int(r.get('n_pivots_lower') or 0)}</td>"
            f"<td class='num {_slope_sign(r.get('upper_slope_pct_per_bar'))}'>"
            f"{_fmt_num(r.get('upper_slope_pct_per_bar'), 2)}</td>"
            f"<td class='num {_slope_sign(r.get('lower_slope_pct_per_bar'))}'>"
            f"{_fmt_num(r.get('lower_slope_pct_per_bar'), 2)}</td>"
            f"<td>{_fit_pill(r.get('fit_quality') or '')}</td>"
            f"<td class='num'>{int(r.get('channel_age_bars') or 0)}</td>"
            f"<td class='num'>{_fmt_age(r.get('breakout_age_bars'))}</td>"
            f"<td class='num'>{_fmt_num(r.get('asof_close'), 2)}</td>"
            f"<td class='num'>{_fmt_pct(r.get('atr_pct'))}</td>"
            "</tr>"
        )
    body = "\n".join(rows)
    return (
        '<table class="cb">'
        '<thead><tr>'
        '<th>#</th><th>Ticker</th><th>State</th><th>Slope</th><th>Tier</th>'
        '<th>Width</th><th>H/L</th><th>S↑·%/bar</th><th>S↓·%/bar</th>'
        '<th>Fit</th><th>Age</th><th>Brk-Age</th><th>Close</th><th>ATR%</th>'
        '</tr></thead>'
        f'<tbody>{body}</tbody></table>'
    )


def _slope_sign(v) -> str:
    if v is None or pd.isna(v):
        return ""
    return "pos" if v >= 0 else "neg"


def _fmt_age(v) -> str:
    if v is None or pd.isna(v):
        return "—"
    try:
        return str(int(v))
    except (TypeError, ValueError):
        return "—"


# ---------------------------------------------------------------- pending triangle summary

def _triangle_summary(out_dir: Path) -> tuple[dict[str, int], int]:
    """Total pending_triangle counts per kind, across all 4 TFs."""
    kinds: dict[str, int] = {}
    total = 0
    for tf in TF_ORDER:
        fam = TF_TO_FAM[tf]
        p = out_dir / f"pending_triangle_{fam}.parquet"
        if not p.exists():
            continue
        try:
            df = pd.read_parquet(p, columns=["triangle_kind_hint"])
        except (FileNotFoundError, ValueError):
            continue
        if df.empty:
            continue
        total += len(df)
        for k, n in df["triangle_kind_hint"].value_counts().items():
            kinds[k] = kinds.get(k, 0) + int(n)
    return kinds, total


def _triangle_summary_html(kinds: dict[str, int], total: int) -> str:
    if total == 0:
        return ""
    pieces = []
    for k in sorted(kinds, key=lambda x: -kinds[x]):
        pieces.append(
            f'<span class="kind">{html.escape(k)} <b>{kinds[k]}</b></span>'
        )
    return (
        '<div class="tri-summary">'
        f'<b>PENDING TRIANGLES</b> · {total} fit (parallelism > 0.25) → triangle workstream\'e devir. '
        f'Kind dağılımı: {"".join(pieces)}'
        '</div>'
    )


# ---------------------------------------------------------------- cross-TF stack

def _cross_tf_stack(per_tf_dfs: dict[str, pd.DataFrame]) -> dict[str, set[str]]:
    """{ticker: set(tf)} for tickers firing in ≥1 family."""
    out: dict[str, set[str]] = {}
    for tf, df in per_tf_dfs.items():
        if df is None or df.empty:
            continue
        for t in df["ticker"].unique():
            out.setdefault(str(t), set()).add(tf)
    return out


def _stack_card(cross_tf: dict[str, set[str]], min_count: int = 2, top_n: int = 18) -> str:
    items = sorted(cross_tf.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    items = [(t, tfs) for t, tfs in items if len(tfs) >= min_count]
    if not items:
        return ""
    blocks: list[str] = []
    for ticker, tfs in items[:top_n]:
        chips = "".join(
            f'<span class="tf-chip ch">ch_{html.escape(t)}</span>'
            for t in sorted(tfs, key=lambda x: TF_ORDER.index(x))
        )
        link = (
            f'<a href="{TV_BASE}{html.escape(ticker)}" target="_blank" rel="noopener">'
            f'{html.escape(ticker)}</a>'
        )
        blocks.append(
            '<div class="stack-item">'
            f'<span class="stk-tk">{link}</span>'
            f'<span class="stk-n">×{len(tfs)}</span>'
            f'<div class="stk-fams">{chips}</div>'
            '</div>'
        )
    return (
        '<div class="cb-card">'
        f'<div class="card-title">Cross-TF Stack <b>· ≥{min_count} TF</b>'
        f'<span class="n">— top {min(top_n, len(items))} / {len(items)} ticker</span></div>'
        f'<div class="stack-grid">{"".join(blocks)}</div>'
        '</div>'
    )


# ---------------------------------------------------------------- nav

def _tf_nav(active_tf: str | None, tf_totals: dict[str, int]) -> str:
    items: list[str] = []
    items.append(
        f'<a href="channel_break_scan_latest.html" class="{"" if active_tf else "active"}">'
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
    return f'<div class="cb-tfnav">{"".join(items)}</div>'


# ---------------------------------------------------------------- gather

def _gather(out_dir: Path) -> dict[str, pd.DataFrame]:
    """{tf: ranked DataFrame}."""
    per_tf: dict[str, pd.DataFrame] = {}
    for tf in TF_ORDER:
        fam = TF_TO_FAM[tf]
        p = out_dir / f"channel_break_{fam}.parquet"
        if not p.exists():
            per_tf[tf] = pd.DataFrame()
            continue
        try:
            df = pd.read_parquet(p)
        except (FileNotFoundError, ValueError):
            per_tf[tf] = pd.DataFrame()
            continue
        per_tf[tf] = _rank_channels(df)
    return per_tf


# ---------------------------------------------------------------- shell

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
      CHANNEL BREAK<span class="proj"> · v0.1</span>
      <span class="mode">parallel-channel · close-only · long-only</span>
    </div>
    <div class="nox-meta">
      As-of: <b>{asof_label}</b><br>
      Aktif kanal: <b>{grand_total}</b><br>
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

  <div class="cb-footer">
    Pipeline: <code>channel_break/</code> · close-only pivots · OLS line fits · 4 aile (5h/1d/1w/1M).<br>
    Driver: <code>tools/channel_break_html.py</code>. Tier A = ≥3 H + ≥3 L pivot touch.<br>
    Veri: <code>output/extfeed_intraday_1h_3y_master.parquet</code> (3y, BIST 607 evren).<br>
    <b>Descriptive only</b> — backtest/ML çalışması yok. Pending triangles ayrı parquet'e devrediliyor.
  </div>

</div>

</body>
</html>
"""


# ---------------------------------------------------------------- public

def build_tf_html(
    tf: str,
    asof: pd.Timestamp | str | None = None,
    out_dir: Path = OUT_DIR,
    *,
    top: int = 30,
    per_tf: dict[str, pd.DataFrame] | None = None,
) -> str:
    if tf not in TF_ORDER:
        raise ValueError(f"Unknown tf {tf!r}; expected one of {TF_ORDER}")

    if per_tf is None:
        per_tf = _gather(out_dir)

    tf_totals = {t: 0 if per_tf.get(t) is None or per_tf[t].empty else len(per_tf[t]) for t in TF_ORDER}
    grand_total = sum(tf_totals.values())

    asof_label = "—"
    if asof is not None:
        try:
            asof_label = pd.Timestamp(asof).strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            asof_label = str(asof)

    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    df = per_tf.get(tf, pd.DataFrame())
    note = TF_NOTES.get(tf, "")
    n = 0 if df is None or df.empty else len(df)

    cross_tf = _cross_tf_stack(per_tf)
    tri_kinds, tri_total = _triangle_summary(out_dir)

    body_html = f"""
  <div class="cb-banner">
    <b>RESEARCH WATCHLIST · NOT LIVE.</b>
    Strict alternating-touch ≥2/2 paralel kanal (asc / desc / flat) +
    long-only up-break. State'ler: <b>TRIGGER</b> (asof'ta kırılım,
    gates pass), <b>EXTENDED</b> (1-5 bar önce gates-pass kırılım),
    <b>PRE-BREAK</b> (close ±%0.5 üst trendline, henüz kırılmadı veya
    gates fail). Tier A = ≥3 H + ≥3 L pivot. Fit pill close-pivot'ların
    OLS line'a uzaklığını gösterir (tight &lt; loose &lt; rough).
    Ticker → TV grafik. Hover → pivot tarihleri.
  </div>

  {_triangle_summary_html(tri_kinds, tri_total)}

  {_stack_card(cross_tf, min_count=2, top_n=18)}

  <div class="cb-tf-note">{html.escape(TF_LABELS[tf])} — {html.escape(note)}</div>
  <div class="cb-stack">
    <div class="cb-card">
      <div class="card-title">CHANNEL · <b>ch_{html.escape(tf)}</b><span class="n">N={n}</span></div>
      {_channel_table(df, top=top)}
    </div>
  </div>
"""

    picks_json = {
        "schema_version": 1,
        "system": "channel_break",
        "page": tf,
        "asof": asof_label,
        "tf_totals": tf_totals,
        "grand_total": grand_total,
        "pending_triangle_total": tri_total,
        "pending_triangle_kinds": tri_kinds,
        "stack_count": len([1 for tfs in cross_tf.values() if len(tfs) >= 2]),
    }
    picks_marker = (
        '<script id="channel-break-data" type="application/json">'
        + json.dumps(picks_json, ensure_ascii=False)
        + '</script>'
    )

    return _shell(
        page_title=f"NOX Channel Break · {TF_LABELS[tf]} — {asof_label}",
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
    per_tf: dict[str, pd.DataFrame] | None = None,
) -> str:
    if per_tf is None:
        per_tf = _gather(out_dir)

    tf_totals = {t: 0 if per_tf.get(t) is None or per_tf[t].empty else len(per_tf[t]) for t in TF_ORDER}
    grand_total = sum(tf_totals.values())

    asof_label = "—"
    if asof is not None:
        try:
            asof_label = pd.Timestamp(asof).strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            asof_label = str(asof)

    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    cross_tf = _cross_tf_stack(per_tf)
    tri_kinds, tri_total = _triangle_summary(out_dir)

    tiles: list[str] = []
    for tf in TF_ORDER:
        href = _file_for(tf)
        n = tf_totals.get(tf, 0)
        df = per_tf.get(tf, pd.DataFrame())
        n_a = 0 if df is None or df.empty else int(df["tier_a"].fillna(False).sum())
        note = TF_NOTES.get(tf, "")
        tiles.append(
            f'<a class="cb-index-tile" href="{href}">'
            f'<div class="tile-tf" style="color:{TF_DOT[tf]};">{html.escape(tf)}</div>'
            f'<div class="tile-name">{html.escape(TF_LABELS[tf])}</div>'
            f'<div class="tile-n">N={n} · Tier-A {n_a}</div>'
            f'<div class="tile-note">{html.escape(note)}</div>'
            f'</a>'
        )

    body_html = f"""
  <div class="cb-banner gold">
    <b>NOX Channel Break v0.1</b> — TF başına ayrı sayfalar. Strict
    alternating-touch paralel kanal + long-only up-break. Cross-TF stack
    panel ≥2 TF'de fire eden ticker'ları topluyor. Pending triangles ayrı
    parquet'lere düşüyor (triangle workstream'inde işlenecek).
  </div>

  {_triangle_summary_html(tri_kinds, tri_total)}

  <div class="cb-index-grid">
    {"".join(tiles)}
  </div>

  {_stack_card(cross_tf, min_count=2, top_n=24)}
"""

    picks_json = {
        "schema_version": 1,
        "system": "channel_break",
        "page": "index",
        "asof": asof_label,
        "tf_totals": tf_totals,
        "grand_total": grand_total,
        "pending_triangle_total": tri_total,
        "pending_triangle_kinds": tri_kinds,
        "stack_count": len([1 for tfs in cross_tf.values() if len(tfs) >= 2]),
    }
    picks_marker = (
        '<script id="channel-break-data" type="application/json">'
        + json.dumps(picks_json, ensure_ascii=False)
        + '</script>'
    )

    return _shell(
        page_title=f"NOX Channel Break · Index — {asof_label}",
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
    top: int = 30,
) -> dict[str, str]:
    per_tf = _gather(out_dir)
    pages: dict[str, str] = {}
    for tf in TF_ORDER:
        pages[tf] = build_tf_html(
            tf, asof=asof, out_dir=out_dir, top=top, per_tf=per_tf,
        )
    pages["index"] = build_index_html(
        asof=asof, out_dir=out_dir, per_tf=per_tf,
    )
    return pages
