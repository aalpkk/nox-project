"""Scanner V1.1 HTML renderer — briefing aesthetic.

Copies the NOX briefing visual language (_NOX_CSS palette + aurora background
+ glassmorphism cards + KATMAN layer titles). Three sections mapped to
`signal_state`:

  KATMAN 1 — PRE-BREAKOUT WATCH   pre_breakout cards (proximity-ranked)
  KATMAN 2 — CONFIRMED TODAY      trigger cards (today's breakouts)
  KATMAN 3 — EXTENDED             extended cards + collapsible detail table
"""
from __future__ import annotations

import html
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from core.reports import _NOX_CSS
from .schema import SCANNER_VERSION, SCHEMA_VERSION


# -------------------------------------------------------------- formatting
def _fmt_pct(v, places=2) -> str:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "—"
    return "—" if not np.isfinite(f) else f"{f * 100:+.{places}f}%"


def _fmt_num(v, places=2) -> str:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "—"
    return "—" if not np.isfinite(f) else f"{f:.{places}f}"


def _fmt_score(v) -> str:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "—"
    return "—" if not np.isfinite(f) else f"{f:.1f}"


def _fmt_int(v) -> str:
    try:
        return str(int(v))
    except (TypeError, ValueError):
        return "—"


def _safe_float(v, default=float("nan")) -> float:
    try:
        f = float(v)
        return f if np.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def _age_days(bd, anchor) -> int:
    if pd.isna(bd):
        return -1
    a = pd.Timestamp(anchor)
    b = pd.Timestamp(bd)
    if a.tzinfo is not None:
        a = a.tz_localize(None)
    if b.tzinfo is not None:
        b = b.tz_localize(None)
    return int((a.normalize() - b.normalize()).days)


def _tags_to_list(s) -> list[str]:
    if isinstance(s, list):
        return [str(t) for t in s]
    if isinstance(s, str) and s:
        try:
            v = json.loads(s)
            return [str(t) for t in v] if isinstance(v, list) else []
        except Exception:
            return []
    return []


# -------------------------------------------------------------- chips
_TAG_PALETTE = {
    "state:pre_breakout":  ("#7a8fa5", "rgba(122,143,165,0.14)"),  # blue
    "state:trigger":       ("#c9a96e", "rgba(201,169,110,0.18)"),  # gold
    "state:extended":      ("#a8876a", "rgba(168,135,106,0.14)"),  # copper
    "base_tight":          ("#e8e4dc", "rgba(232,228,220,0.08)"),
    "flat_base":           ("#e8e4dc", "rgba(232,228,220,0.08)"),
    "touch_ok":            ("#7a8fa5", "rgba(122,143,165,0.10)"),
    "breakout_confirmed":  ("#7a9e7a", "rgba(122,158,122,0.14)"),
    "volume_expansion":    ("#c9a96e", "rgba(201,169,110,0.14)"),
    "volume_dryup_pre":    ("#8a7a9e", "rgba(138,122,158,0.12)"),
    "trend_reclaim":       ("#7a9e7a", "rgba(122,158,122,0.10)"),
    "rs_top20":            ("#c9a96e", "rgba(201,169,110,0.12)"),
    "at_resistance":       ("#c9a96e", "rgba(201,169,110,0.18)"),
    "late_chase":          ("#9e5a5a", "rgba(158,90,90,0.14)"),
}


def _chip(tag: str) -> str:
    fg, bg = _TAG_PALETTE.get(tag, ("#8a8580", "rgba(138,133,128,0.08)"))
    label = tag.split(":", 1)[-1] if tag.startswith("state:") else tag
    return (
        f'<span class="sc-chip" style="color:{fg};background:{bg}">'
        f'{html.escape(label)}</span>'
    )


def _tv_link(ticker: str) -> str:
    t = html.escape(str(ticker))
    return f'<a class="sc-ticker" href="https://www.tradingview.com/chart/?symbol=BIST:{t}" target="_blank">{t}</a>'


# -------------------------------------------------------------- top strip
def _market_pills(row: dict) -> str:
    pills: list[str] = []
    breadth = _safe_float(row.get("common__market_breadth_pct_above_sma20"))
    if np.isfinite(breadth):
        pills.append(f'<span class="mpill">BREADTH <b>{breadth*100:.0f}%</b></span>')
    trend = _safe_float(row.get("common__market_trend_score"))
    if np.isfinite(trend):
        cls = "pos" if trend > 0 else ("neg" if trend < 0 else "neu")
        pills.append(f'<span class="mpill"><span class="dot {cls}"></span>TREND <b>{trend:+.1f}</b></span>')
    vol = _safe_float(row.get("common__market_vol_regime"))
    if np.isfinite(vol):
        pills.append(f'<span class="mpill">VOL p<b>{vol*100:.0f}</b></span>')
    r5 = _safe_float(row.get("common__index_ret_5d"))
    if np.isfinite(r5):
        pills.append(f'<span class="mpill">XU100 5d <b>{r5*100:+.2f}%</b></span>')
    r20 = _safe_float(row.get("common__index_ret_20d"))
    if np.isfinite(r20):
        pills.append(f'<span class="mpill">XU100 20d <b>{r20*100:+.2f}%</b></span>')
    return ''.join(pills)


def _status_bar(df: pd.DataFrame, asof_label: str) -> str:
    cnt_pre = int((df["signal_state"] == "pre_breakout").sum())
    cnt_trg = int((df["signal_state"] == "trigger").sum())
    cnt_ext = int((df["signal_state"] == "extended").sum())
    market_row = df.iloc[0].to_dict() if not df.empty else {}
    pills_html = _market_pills(market_row)
    return f"""
<div class="sc-status">
  <div class="sc-logo">
    <span class="nox-text">NOX</span><span class="brief-text">scanner</span>
  </div>
  <div class="sc-counts">
    <span class="cpill cpill-pre">PRE <b>{cnt_pre}</b></span>
    <span class="cpill cpill-trg">TRIG <b>{cnt_trg}</b></span>
    <span class="cpill cpill-ext">EXT <b>{cnt_ext}</b></span>
  </div>
  <div class="sc-macros">{pills_html}</div>
  <div class="sc-meta">{html.escape(asof_label)}</div>
</div>
"""


# -------------------------------------------------------------- card
def _entry_levels(r: dict, state: str) -> str:
    entry = _safe_float(r.get("entry_reference_price"))
    trig = _safe_float(r.get("family__trigger_level"))
    inv = _safe_float(r.get("invalidation_level"))
    risk = _safe_float(r.get("initial_risk_pct"))
    parts: list[str] = []
    if np.isfinite(trig):
        parts.append(f'<span class="e-label">TRIG</span><span class="e-trig">{trig:.2f}</span>')
    if np.isfinite(entry):
        parts.append(f'<span class="e-label">ENT</span><span class="e-entry">{entry:.2f}</span>')
    if np.isfinite(inv):
        parts.append(f'<span class="e-label">SL</span><span class="e-sl">{inv:.2f}</span>')
    if np.isfinite(risk):
        parts.append(f'<span class="e-label">R%</span><span class="e-r">{risk*100:.1f}%</span>')
    return ''.join(parts)


def _card(r: dict, state: str, anchor_date) -> str:
    ticker = str(r.get("ticker", "?"))
    score = _safe_float(r.get("rule_score"))
    sector = str(r.get("sector", "")) if r.get("sector") else ""
    tags = [t for t in _tags_to_list(r.get("signal_tags")) if not t.startswith("state:")]
    chips_html = ''.join(_chip(t) for t in tags[:6])

    # state-specific top-line metric
    if state == "pre_breakout":
        dist_top = _safe_float(r.get("family__prebreakout_distance_to_high"))
        sub = (f'distance to top <b>{dist_top*100:.2f}%</b>'
               if np.isfinite(dist_top) else 'pre-breakout')
    elif state == "trigger":
        bo_atr = _safe_float(r.get("common__breakout_atr"))
        vr = _safe_float(r.get("common__volume_ratio_20"))
        bo_s = f'{bo_atr:.2f}×ATR' if np.isfinite(bo_atr) else '—'
        vr_s = f'{vr:.1f}×vol' if np.isfinite(vr) else '—'
        sub = f'breakout <b>{bo_s}</b> · <b>{vr_s}</b>'
    else:  # extended
        ext = _safe_float(r.get("common__extension_from_trigger"))
        bd = r.get("breakout_bar_date")
        age = _age_days(bd, anchor_date) if bd is not None else -1
        ext_s = f'+{ext*100:.1f}%' if np.isfinite(ext) else '—'
        age_s = f'{age}d ago' if age >= 0 else 'older'
        sub = f'ext <b>{ext_s}</b> · trig <b>{age_s}</b>'

    rs = _safe_float(r.get("common__rs_pctile_252"))
    rs_html = f'<span class="rs-pill">RS p{rs*100:.0f}</span>' if np.isfinite(rs) else ''
    sec_html = f'<span class="sec-pill">{html.escape(sector)}</span>' if sector and sector != "Unknown" else ''
    score_html = f'<span class="score-pill">{score:.1f}</span>' if np.isfinite(score) else ''

    levels_html = _entry_levels(r, state)

    return f"""
<div class="sc-card sc-card-{state}" data-sector="{html.escape(sector)}" data-tags="{html.escape(' '.join(tags))}">
  <div class="card-head">
    {_tv_link(ticker)}
    {sec_html}
    {rs_html}
    {score_html}
  </div>
  <div class="card-sub">{sub}</div>
  <div class="card-chips">{chips_html}</div>
  <div class="card-levels">{levels_html}</div>
</div>
"""


def _section(state: str, title: str, label_no: int, df_state: pd.DataFrame, anchor_date) -> str:
    if df_state.empty:
        return f"""
<div class="layer-title"><span class="layer-no">KATMAN {label_no}</span> {title}
  <span class="cnt">0</span></div>
<div class="empty-state">— bu state'te aday yok —</div>
"""
    cards = ''.join(_card(r, state, anchor_date) for r in df_state.to_dict("records"))
    return f"""
<div class="layer-title"><span class="layer-no">KATMAN {label_no}</span> {title}
  <span class="cnt">{len(df_state)}</span></div>
<div class="sc-grid">{cards}</div>
"""


# -------------------------------------------------------------- extended detail table
_DETAIL_COLS = [
    ("ticker", "Ticker"),
    ("sector", "Sector"),
    ("breakout_bar_date", "Trig date"),
    ("common__extension_from_trigger", "Ext"),
    ("family__channel_high", "Box ↑"),
    ("family__channel_low", "Box ↓"),
    ("family__base_duration_weeks", "Base wk"),
    ("family__base_slope", "Slope/bar"),
    ("common__breakout_atr", "BO ATR"),
    ("common__volume_ratio_20", "Vol×"),
    ("common__rs_pctile_252", "RS252"),
    ("common__close_vs_vwap_ytd", "vs VWAP_YTD"),
    ("rule_score", "Score"),
]


def _td(value, col_name: str) -> str:
    if col_name == "ticker":
        return f'<td>{_tv_link(str(value))}</td>'
    if col_name == "sector":
        return f'<td>{html.escape(str(value)) if value else "—"}</td>'
    if col_name == "breakout_bar_date":
        if pd.isna(value):
            return '<td>—</td>'
        return f'<td>{pd.Timestamp(value).strftime("%Y-%m-%d")}</td>'
    if col_name in ("common__extension_from_trigger", "family__base_slope",
                    "common__close_vs_vwap_ytd"):
        return f'<td class="num">{_fmt_pct(value)}</td>'
    if col_name == "common__rs_pctile_252":
        f = _safe_float(value)
        return f'<td class="num">{"—" if not np.isfinite(f) else f"p{f*100:.0f}"}</td>'
    if col_name == "rule_score":
        return f'<td class="num"><b>{_fmt_score(value)}</b></td>'
    if col_name == "common__breakout_atr":
        f = _safe_float(value)
        return f'<td class="num">{"—" if not np.isfinite(f) else f"{f:.2f}×"}</td>'
    if col_name == "common__volume_ratio_20":
        f = _safe_float(value)
        return f'<td class="num">{"—" if not np.isfinite(f) else f"{f:.1f}×"}</td>'
    if col_name in ("family__channel_high", "family__channel_low"):
        return f'<td class="num">{_fmt_num(value, 2)}</td>'
    if col_name == "family__base_duration_weeks":
        f = _safe_float(value)
        return f'<td class="num">{"—" if not np.isfinite(f) else f"{f:.1f}w"}</td>'
    return f'<td class="num">{_fmt_num(value)}</td>'


def _detail_table(df_state: pd.DataFrame) -> str:
    if df_state.empty:
        return ""
    header = ''.join(f'<th>{html.escape(label)}</th>' for _, label in _DETAIL_COLS)
    rows: list[str] = []
    df_sorted = df_state.sort_values("rule_score", ascending=False)
    for r in df_sorted.to_dict("records"):
        cells = ''.join(_td(r.get(c), c) for c, _ in _DETAIL_COLS)
        rows.append(f'<tr>{cells}</tr>')
    return f"""
<details class="detail-block">
  <summary>full table <span class="det-count">{len(df_state)} rows</span></summary>
  <div class="detail-body">
    <div class="sc-table-wrap">
      <table class="sc-table">
        <thead><tr>{header}</tr></thead>
        <tbody>{''.join(rows)}</tbody>
      </table>
    </div>
  </div>
</details>
"""


# -------------------------------------------------------------- filter bar
def _filter_bar(df: pd.DataFrame) -> str:
    sectors = sorted({s for s in df["sector"].dropna().tolist() if s and s != "Unknown"})
    sec_opts = ''.join(f'<option value="{html.escape(s)}">{html.escape(s)}</option>' for s in sectors)
    all_tags = sorted({
        t for tags in df["signal_tags"].apply(_tags_to_list)
        for t in tags if not t.startswith("state:")
    })
    chip_html = ''.join(
        f'<span class="filter-chip" data-tag="{html.escape(t)}">{html.escape(t)}</span>'
        for t in all_tags
    )
    return f"""
<div class="sc-filters">
  <label>SECTOR</label>
  <select id="f-sector"><option value="">all</option>{sec_opts}</select>
  <label>MIN SCORE</label>
  <input type="number" id="f-score" value="0" step="1" min="0">
  <button class="sc-btn" id="f-clear">RESET</button>
  <div class="filter-chips">{chip_html}</div>
</div>
"""


# -------------------------------------------------------------- scanner-specific CSS
_SCANNER_CSS = """
.sc-container { position: relative; z-index: 1; max-width: 1200px; margin: 0 auto; padding: 0 1.25rem 2rem; }

/* Status bar (sticky) */
.sc-status {
  position: sticky; top: 0; z-index: 100;
  background: rgba(6,6,8,0.55);
  backdrop-filter: blur(24px) saturate(1.3);
  -webkit-backdrop-filter: blur(24px) saturate(1.3);
  border-bottom: 1px solid rgba(201,169,110,0.10);
  padding: 0.65rem 1.25rem;
  margin: 0 -1.25rem 1.25rem;
  display: flex; align-items: center; gap: 0.85rem; flex-wrap: wrap;
}
.sc-logo { display: inline-flex; align-items: baseline; gap: 0.15rem; white-space: nowrap; }
.sc-logo .nox-text { font-family: var(--font-brand); font-size: 2.1rem; color: #fff; letter-spacing: 0.05em; line-height: 0.85; }
.sc-logo .brief-text { font-family: var(--font-handwrite); font-size: 1.05rem; color: #fff; margin-left: 0.2rem; position: relative; top: -0.08rem; }
.sc-counts { display: flex; gap: 0.35rem; }
.cpill {
  font-family: var(--font-mono); font-size: 0.7rem;
  padding: 0.18rem 0.55rem; border-radius: 0.75rem;
  background: var(--bg-card); border: 1px solid var(--border-subtle);
  color: var(--text-secondary); white-space: nowrap;
}
.cpill b { font-weight: 700; color: var(--text-primary); margin-left: 0.3rem; }
.cpill-pre { border-color: rgba(122,143,165,0.40); }
.cpill-pre b { color: #9eb3c9; }
.cpill-trg { border-color: rgba(201,169,110,0.50); background: rgba(201,169,110,0.08); }
.cpill-trg b { color: var(--nox-gold); }
.cpill-ext { border-color: rgba(168,135,106,0.30); }
.cpill-ext b { color: #c8b095; }
.sc-macros { display: flex; gap: 0.4rem; flex-wrap: wrap; flex: 1; }
.sc-macros .mpill {
  font-family: var(--font-mono); font-size: 0.68rem;
  padding: 0.15rem 0.5rem; border-radius: 0.75rem;
  background: var(--bg-card); border: 1px solid var(--border-subtle);
  color: var(--text-secondary); white-space: nowrap;
}
.sc-macros .mpill b { font-weight: 600; color: var(--text-primary); }
.sc-macros .dot { display:inline-block; width:6px; height:6px; border-radius:50%; margin-right:0.3rem; vertical-align: middle; }
.sc-macros .dot.pos { background: var(--nox-green); }
.sc-macros .dot.neg { background: var(--nox-red); }
.sc-macros .dot.neu { background: var(--text-muted); }
.sc-meta { font-size: 0.7rem; color: var(--text-muted); font-family: var(--font-mono); white-space: nowrap; margin-left: auto; }

/* Filter bar */
.sc-filters {
  display: flex; gap: 0.6rem; flex-wrap: wrap; align-items: center;
  padding: 0.55rem 0.85rem; margin-bottom: 1rem;
  background: var(--bg-card); border: 1px solid var(--border-subtle);
  border-radius: var(--radius);
}
.sc-filters label { font-size: 0.65rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.08em; }
.sc-filters select, .sc-filters input {
  background: var(--bg-primary); color: var(--text-primary);
  border: 1px solid var(--border-subtle); border-radius: var(--radius-sm);
  padding: 4px 9px; font-size: 0.74rem; font-family: var(--font-mono); outline: none;
}
.sc-filters select:focus, .sc-filters input:focus { border-color: var(--nox-gold); }
.sc-filters input[type=number] { width: 60px; }
.sc-btn {
  background: var(--bg-elevated); color: var(--text-secondary);
  border: 1px solid var(--border-subtle); border-radius: var(--radius-sm);
  padding: 4px 10px; cursor: pointer; font-size: 0.66rem; letter-spacing: 0.08em;
  font-family: var(--font-display); font-weight: 600; transition: all 0.15s;
}
.sc-btn:hover { color: var(--text-primary); border-color: var(--nox-gold); }
.filter-chips { display: flex; gap: 0.3rem; flex-wrap: wrap; margin-left: auto; }
.filter-chip {
  font-family: var(--font-mono); font-size: 0.62rem;
  padding: 0.12rem 0.45rem; border-radius: 0.65rem;
  background: var(--bg-elevated); color: var(--text-muted);
  border: 1px solid var(--border-subtle); cursor: pointer; user-select: none;
  transition: all 0.15s;
}
.filter-chip:hover { color: var(--text-primary); border-color: var(--border-dim); }
.filter-chip.active { color: var(--nox-gold); border-color: var(--nox-gold); background: var(--nox-gold-dim); }

/* Layer titles */
.layer-title {
  font-family: var(--font-display); color: var(--text-secondary);
  font-size: 0.78rem; font-weight: 600;
  text-transform: uppercase; letter-spacing: 0.14em;
  margin: 1.6rem 0 0.7rem;
  padding-bottom: 0.4rem;
  border-bottom: 1px solid transparent;
  border-image: linear-gradient(90deg, rgba(201,169,110,0.28), transparent 60%) 1;
  display: flex; align-items: center; gap: 0.6rem;
}
.layer-title:first-of-type { margin-top: 0.4rem; }
.layer-title .layer-no {
  font-family: var(--font-mono); font-size: 0.62rem;
  color: var(--nox-gold); letter-spacing: 0.10em;
  border: 1px solid rgba(201,169,110,0.35); padding: 0.10rem 0.45rem;
  border-radius: 0.5rem;
}
.layer-title .cnt { font-family: var(--font-mono); font-size: 0.7rem; color: var(--text-muted); margin-left: auto; }
.empty-state { color: var(--text-muted); font-size: 0.78rem; padding: 0.8rem 0.4rem; }

/* Card grid */
.sc-grid {
  display: grid; gap: 0.7rem;
  grid-template-columns: repeat(auto-fill, minmax(290px, 1fr));
  margin-bottom: 1rem;
}
.sc-card {
  background: rgba(199,189,190,0.07);
  backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px);
  border-radius: 16px; padding: 0.8rem 0.95rem;
  border-left: 3px solid transparent;
  transition: all 0.22s ease;
  animation: scFadeIn 0.3s ease-out both;
}
.sc-card:hover {
  background: rgba(199,189,190,0.12);
  box-shadow: 0 8px 32px rgba(0,0,0,0.4), 0 0 16px rgba(201,169,110,0.04);
  transform: translateY(-1px);
}
.sc-card-pre_breakout { border-left-color: rgba(122,143,165,0.45); }
.sc-card-trigger      { border-left-color: rgba(201,169,110,0.55); }
.sc-card-extended     { border-left-color: rgba(168,135,106,0.35); }
@keyframes scFadeIn {
  from { opacity: 0; transform: translateY(6px); }
  to { opacity: 1; transform: translateY(0); }
}
.card-head { display: flex; align-items: center; gap: 0.45rem; margin-bottom: 0.35rem; flex-wrap: wrap; }
.sc-ticker { font-family: 'Inter', sans-serif; font-weight: 900; font-size: 1.05rem; color: #fff; text-decoration: none; }
.sc-ticker:hover { color: var(--nox-gold); }
.sec-pill {
  font-family: var(--font-mono); font-size: 0.6rem; padding: 0.10rem 0.4rem;
  border-radius: 0.4rem; background: var(--bg-elevated); color: var(--text-secondary);
  letter-spacing: 0.04em;
}
.rs-pill {
  font-family: var(--font-mono); font-size: 0.6rem; padding: 0.10rem 0.4rem;
  border-radius: 0.4rem; background: rgba(201,169,110,0.10); color: var(--nox-gold);
}
.score-pill {
  font-family: var(--font-mono); font-weight: 700; font-size: 0.72rem;
  padding: 0.12rem 0.5rem; border-radius: 0.4rem;
  background: var(--nox-gold-dim); color: var(--nox-gold); margin-left: auto;
}
.card-sub { font-size: 0.74rem; color: var(--text-secondary); margin-bottom: 0.5rem; font-family: var(--font-mono); }
.card-sub b { color: var(--text-primary); font-weight: 600; }
.card-chips { display: flex; flex-wrap: wrap; gap: 0.25rem; margin-bottom: 0.5rem; }
.sc-chip {
  display: inline-block; font-family: var(--font-mono); font-size: 0.6rem;
  padding: 0.10rem 0.4rem; border-radius: 0.4rem; letter-spacing: 0.02em;
}
.card-levels {
  display: flex; gap: 0.5rem; flex-wrap: wrap;
  padding-top: 0.45rem; border-top: 1px solid var(--border-subtle);
  font-family: var(--font-mono); font-size: 0.68rem; align-items: center;
}
.card-levels .e-label { color: var(--text-muted); margin-right: 0.15rem; }
.card-levels .e-trig { color: var(--nox-gold); font-weight: 600; margin-right: 0.6rem; }
.card-levels .e-entry { color: var(--text-primary); font-weight: 600; margin-right: 0.6rem; }
.card-levels .e-sl { color: var(--nox-red); font-weight: 600; margin-right: 0.6rem; }
.card-levels .e-r { color: var(--text-secondary); font-weight: 600; }

/* Detail block */
.detail-block { margin-bottom: 1rem; }
.detail-block > summary {
  font-family: var(--font-display); color: var(--nox-gold);
  font-size: 0.82rem; font-weight: 600; cursor: pointer; user-select: none;
  padding: 0.55rem 0.8rem; background: rgba(199,189,190,0.07);
  backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px);
  border-radius: 14px; display: flex; align-items: center; gap: 0.4rem; list-style: none;
  transition: all 0.2s;
}
.detail-block > summary::-webkit-details-marker { display: none; }
.detail-block > summary::before { content: '▸'; font-size: 0.75rem; color: var(--text-muted); transition: transform 0.2s; }
.detail-block[open] > summary::before { transform: rotate(90deg); }
.detail-block > summary:hover { background: rgba(199,189,190,0.10); }
.detail-block > summary .det-count { font-family: var(--font-mono); font-size: 0.66rem; color: var(--text-muted); margin-left: auto; }
.detail-block .detail-body { padding: 0.7rem 0.4rem 0; }

/* Detail table */
.sc-table-wrap { overflow-x: auto; border-radius: 12px; border: 1px solid var(--border-subtle); background: var(--bg-card); }
.sc-table { width: 100%; border-collapse: collapse; font-size: 0.74rem; }
.sc-table thead { position: sticky; top: 0; z-index: 5; }
.sc-table th {
  background: var(--bg-elevated); color: var(--text-muted);
  font-weight: 600; font-size: 0.62rem; text-transform: uppercase; letter-spacing: 0.06em;
  padding: 8px 10px; text-align: left; border-bottom: 1px solid var(--border-subtle); white-space: nowrap;
  font-family: var(--font-display);
}
.sc-table td { padding: 7px 10px; border-bottom: 1px solid rgba(39,39,42,0.5); white-space: nowrap; font-family: var(--font-mono); font-size: 0.7rem; color: var(--text-secondary); }
.sc-table td.num { text-align: right; color: var(--text-primary); }
.sc-table tbody tr:hover { background: var(--bg-hover); }

/* Footer */
.sc-footer {
  text-align: center; padding: 1.6rem 0 1rem;
  font-size: 0.62rem; color: rgba(255,255,255,0.32);
  font-family: var(--font-mono); letter-spacing: 0.10em;
}
"""

_FILTER_JS = """
(function() {
  var fSec = document.getElementById('f-sector');
  var fSco = document.getElementById('f-score');
  var fCl  = document.getElementById('f-clear');
  var chips = Array.from(document.querySelectorAll('.filter-chip'));
  var activeTags = new Set();
  function apply() {
    var sec = fSec.value;
    var minS = parseFloat(fSco.value) || 0;
    document.querySelectorAll('.sc-card').forEach(function(c) {
      var ok = true;
      if (sec && c.dataset.sector !== sec) ok = false;
      var pill = c.querySelector('.score-pill');
      if (pill) {
        var sv = parseFloat(pill.textContent) || 0;
        if (sv < minS) ok = false;
      }
      if (activeTags.size > 0) {
        var ts = (c.dataset.tags || '').split(/\\s+/);
        var hit = false;
        activeTags.forEach(function(t) { if (ts.indexOf(t) >= 0) hit = true; });
        if (!hit) ok = false;
      }
      c.style.display = ok ? '' : 'none';
    });
  }
  fSec.addEventListener('change', apply);
  fSco.addEventListener('input', apply);
  fCl.addEventListener('click', function() {
    fSec.value = ''; fSco.value = '0';
    activeTags.clear();
    chips.forEach(function(ch) { ch.classList.remove('active'); });
    apply();
  });
  chips.forEach(function(ch) {
    ch.addEventListener('click', function() {
      var t = ch.dataset.tag;
      if (activeTags.has(t)) { activeTags.delete(t); ch.classList.remove('active'); }
      else { activeTags.add(t); ch.classList.add('active'); }
      apply();
    });
  });
})();
"""


# -------------------------------------------------------------- entry point
def render_html(
    df: pd.DataFrame,
    *,
    out_path: str | Path,
    title: str = "Scanner V1.1 — Horizontal Base",
    sector_map: dict[str, str] | None = None,
    asof: Optional[pd.Timestamp] = None,
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if df.empty:
        out_path.write_text(
            f"""<!DOCTYPE html><html lang="tr"><head><meta charset="utf-8">
<title>{html.escape(title)}</title><style>{_NOX_CSS}{_SCANNER_CSS}</style></head>
<body>
<div class="aurora-bg"><div class="aurora-layer aurora-layer-1"></div>
<div class="aurora-layer aurora-layer-2"></div>
<div class="aurora-layer aurora-layer-3"></div></div>
<div class="mesh-overlay"></div>
<div class="sc-container"><div class="empty-state">Bugün için aday yok.</div></div>
</body></html>""",
            encoding="utf-8",
        )
        return out_path

    df = df.copy()
    if sector_map and "sector" not in df.columns:
        df["sector"] = df["ticker"].map(sector_map)
    elif "sector" not in df.columns:
        df["sector"] = "Unknown"
    df["sector"] = df["sector"].fillna("Unknown")

    if asof is None:
        asof_ts = pd.to_datetime(df["bar_date"]).max().normalize()
    else:
        asof_ts = pd.Timestamp(asof).normalize()

    # split by state, sort each
    pre  = df[df["signal_state"] == "pre_breakout"].sort_values("rule_score", ascending=False)
    trg  = df[df["signal_state"] == "trigger"].sort_values("rule_score", ascending=False)
    ext  = df[df["signal_state"] == "extended"].sort_values("rule_score", ascending=False)

    asof_label = asof_ts.strftime("%a · %d %b %Y").upper()

    sec_pre  = _section("pre_breakout", "PRE-BREAKOUT WATCH", 1, pre, asof_ts)
    sec_trg  = _section("trigger",      "CONFIRMED TODAY",    2, trg, asof_ts)
    sec_ext  = _section("extended",     "EXTENDED",           3, ext, asof_ts)
    detail_ext = _detail_table(ext)

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    page = f"""<!DOCTYPE html>
<html lang="tr"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>{_NOX_CSS}{_SCANNER_CSS}</style>
</head><body>
<div class="aurora-bg">
  <div class="aurora-layer aurora-layer-1"></div>
  <div class="aurora-layer aurora-layer-2"></div>
  <div class="aurora-layer aurora-layer-3"></div>
</div>
<div class="mesh-overlay"></div>
<div class="sc-container">
  {_status_bar(df, asof_label)}
  {_filter_bar(df)}
  {sec_pre}
  {sec_trg}
  {sec_ext}
  {detail_ext}
  <div class="sc-footer">
    NOX SCANNER · SCHEMA v{SCHEMA_VERSION} · SCANNER v{SCANNER_VERSION} ·
    GENERATED {now}
  </div>
</div>
<script>{_FILTER_JS}</script>
</body></html>
"""
    out_path.write_text(page, encoding="utf-8")
    return out_path
