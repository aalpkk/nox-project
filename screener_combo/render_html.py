"""NOX-themed HTML renderer for screener_combo daily 9-list.

Imports `_NOX_CSS` from core.reports for visual parity with briefing/regime/dip
reports. Output: single self-contained HTML with combined union list (top
section) + per-gate top-3 detail (bottom section).

Functions:
  render_html(basket_df, pergate_df, asof, n_universe, n_today_signal) -> str
"""
from __future__ import annotations

import datetime
import json
from typing import Iterable

import numpy as np
import pandas as pd

from core.reports import _NOX_CSS, _sanitize


GATE_LABELS = {"regime_trig": "RT", "weekly_trig": "NW", "alsat_trig": "AS"}
GATE_COLORS = {
    "regime_trig": "#7a8fa5",   # blue — regime transition
    "weekly_trig": "#c9a96e",   # gold — weekly pivot
    "alsat_trig":  "#b8956e",   # copper — al/sat
}
TIER_COLORS = {
    "ALTIN": "#c9a96e", "GUMUS": "#b8956e", "BRONZ": "#a8876a",
    "NORMAL": "#7a8fa5", "ELE": "#9e5a5a",
}
DEC_COLORS = {"AL": "#7a9e7a", "IZLE": "#c9a96e", "ATLA": "#9e5a5a"}


def _basket_to_records(basket_df: pd.DataFrame) -> list[dict]:
    if basket_df.empty:
        return []
    out = []
    for _, r in basket_df.iterrows():
        out.append({
            "ticker": r.get("ticker", ""),
            "gate": r.get("gate", ""),
            "gate_label": GATE_LABELS.get(r.get("gate", ""), r.get("gate", "")),
            "rank_in_gate": int(r["rank_in_gate"]) if pd.notna(r.get("rank_in_gate")) else None,
            "rank_score": float(r["rank_score"]) if pd.notna(r.get("rank_score")) else None,
            "rt_subtype": r.get("rt_subtype", "") or "",
            "rt_tier": r.get("rt_tier", "") or "",
            "rt_gir": int(r["rt_gir"]) if pd.notna(r.get("rt_gir")) else None,
            "rt_ok": int(r["rt_ok"]) if pd.notna(r.get("rt_ok")) else None,
            "as_subtype": r.get("as_subtype", "") or "",
            "as_decision": r.get("as_decision", "") or "",
            "nox_dw_type": r.get("nox_dw_type", "") or "",
            "close": float(r["close"]) if pd.notna(r.get("close")) else None,
            "atr_pct": float(r["atr_pct"]) if pd.notna(r.get("atr_pct")) else None,
            "rs_score": float(r["rs_score"]) if pd.notna(r.get("rs_score")) else None,
        })
    return out


def _pergate_to_records(pergate_df: pd.DataFrame) -> dict[str, list[dict]]:
    """{gate: [rank1, rank2, rank3]}"""
    if pergate_df is None or pergate_df.empty:
        return {}
    out: dict[str, list[dict]] = {}
    for gate, sub in pergate_df.groupby("gate"):
        s = sub.sort_values("rank_in_gate")
        recs = []
        for _, r in s.iterrows():
            recs.append({
                "rank": int(r["rank_in_gate"]),
                "ticker": r.get("ticker", ""),
                "rank_score": float(r["rank_score"]) if pd.notna(r.get("rank_score")) else None,
                "rt_subtype": r.get("rt_subtype", "") or "",
                "rt_tier": r.get("rt_tier", "") or "",
                "rt_gir": int(r["rt_entry_score"]) if pd.notna(r.get("rt_entry_score")) else None,
                "rt_ok": int(r["rt_oe_score"]) if pd.notna(r.get("rt_oe_score")) else None,
                "as_subtype": r.get("as_subtype", "") or "",
                "as_decision": r.get("as_decision", "") or "",
                "nox_dw_type": r.get("nox_dw_type", "") or "",
            })
        out[gate] = recs
    return out


def render_html(
    basket_df: pd.DataFrame,
    pergate_df: pd.DataFrame,
    *,
    asof: pd.Timestamp,
    n_universe: int,
    n_today_signal: int,
) -> str:
    now_str = datetime.datetime.now().strftime("%d.%m.%Y %H:%M")
    asof_str = pd.Timestamp(asof).strftime("%d.%m.%Y")
    rows = _basket_to_records(basket_df)
    pergate = _pergate_to_records(pergate_df)
    rows_json = json.dumps(_sanitize(rows), ensure_ascii=False)
    pergate_json = json.dumps(_sanitize(pergate), ensure_ascii=False)
    gate_colors_json = json.dumps(GATE_COLORS)
    tier_colors_json = json.dumps(TIER_COLORS)
    dec_colors_json = json.dumps(DEC_COLORS)
    gate_labels_json = json.dumps(GATE_LABELS)
    n_unique = len(rows)
    n_picks = sum(len(v) for v in pergate.values())

    extra_css = """
.gate-section { margin: 32px 0 16px; }
.gate-section h2 {
  font-family: var(--font-display); font-size: 0.95rem; font-weight: 700;
  letter-spacing: 0.04em; text-transform: uppercase; color: var(--text-secondary);
  padding: 8px 0; border-bottom: 1px solid var(--border-subtle); margin-bottom: 12px;
}
.gate-pill { display: inline-block; padding: 2px 8px; border-radius: 999px;
  font-family: var(--font-mono); font-size: 0.65rem; font-weight: 600;
  letter-spacing: 0.05em; color: #060709; }
.tier-badge { display: inline-block; padding: 1px 6px; border-radius: var(--radius-sm);
  font-family: var(--font-mono); font-size: 0.6rem; font-weight: 700;
  letter-spacing: 0.04em; color: #060709; }
.dec-badge { display: inline-block; padding: 1px 6px; border-radius: var(--radius-sm);
  font-family: var(--font-mono); font-size: 0.6rem; font-weight: 700; color: #fff; }
.score-pill { font-family: var(--font-mono); font-size: 0.7rem;
  color: var(--nox-gold); font-weight: 600; }
.giok-wrap { display: inline-flex; gap: 4px; align-items: center; font-family: var(--font-mono); font-size: 0.65rem; }
.giok-tag { padding: 0 4px; border-radius: 3px; font-weight: 600; }
.giok-gir { background: rgba(122,143,165,0.15); color: var(--nox-blue); }
.giok-ok-clean { background: rgba(122,158,122,0.15); color: var(--nox-green); }
.giok-ok-warn { background: rgba(158,90,90,0.15); color: var(--nox-red); }
.empty-row { padding: 14px; text-align: center; color: var(--text-muted);
  font-family: var(--font-mono); font-size: 0.75rem; }
.basket-foot { margin-top: 24px; padding: 14px 16px; border: 1px dashed var(--border-dim);
  border-radius: var(--radius-sm); font-family: var(--font-mono); font-size: 0.7rem;
  color: var(--text-muted); line-height: 1.5; }
.basket-foot b { color: var(--text-secondary); }
"""

    html = f"""<!DOCTYPE html>
<html lang="tr"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NOX — Screener Combo · BIST · {asof_str}</title>
<style>{_NOX_CSS}
{extra_css}
</style>
</head><body>
<div class="aurora-bg">
  <div class="aurora-layer aurora-layer-1"></div>
  <div class="aurora-layer aurora-layer-2"></div>
  <div class="aurora-layer aurora-layer-3"></div>
</div>
<div class="mesh-overlay"></div>
<div class="nox-container">
<div class="nox-header">
  <div class="nox-logo">NOX<span class="proj">project</span><span class="mode">screener combo · BIST</span></div>
  <div class="nox-meta"><b>{n_unique}</b> hisse / {n_picks} kapı seçimi<br>
       {n_today_signal} sinyal / {n_universe} taranan<br>
       asof {asof_str} &nbsp;·&nbsp; {now_str}</div>
</div>

<div class="gate-section">
  <h2>9'lu liste — kapıların kesişimi</h2>
  <div class="nox-table-wrap">
  <table><thead><tr>
    <th>Hisse</th><th>Kapı</th><th>Sıra</th><th>Skor</th>
    <th>RT subtype</th><th>RT tier</th><th>GIR/OK</th>
    <th>AS subtype</th><th>AS</th>
    <th>NOX</th><th>Fiyat</th><th>ATR%</th><th>RS</th>
  </tr></thead><tbody id="bk"></tbody></table>
  </div>
</div>

<div id="pergate-sections"></div>

<div class="basket-foot">
  <b>VAL backtest baseline (locked, h=20):</b>
  Combined 9-list PF_daily 3.51 vs random 2.71; weekly_trig top-3 PF 5.26 (random 3.05);
  regime_trig top-3 PF 2.20 (random 1.97); alsat_trig top-3 PF 2.77 (random 2.37).
  Bu rakamlar VAL penceresi geriye-bakış sonucudur — canlı beklenti olarak okunmamalıdır.
  TEST kohortu sealed.
  <br><br>
  Kategori-only edge: RT BRONZ tier PF 2.88 (TRAIN), AS subtype ZAYIF PF 2.10, NOX W PF 2.27,
  GIR≥3 × OK=0 cohort PF 2.23 (VAL). Detaylar: <code>memory/screener_combo_v1_categories.md</code>.
</div>

</div>
<script>
const D = {rows_json};
const PG = {pergate_json};
const GC = {gate_colors_json};
const TC = {tier_colors_json};
const DC = {dec_colors_json};
const GL = {gate_labels_json};
const TV = "BIST:";

function tierBadge(t) {{
  if (!t) return '<span style="color:var(--text-muted)">—</span>';
  const c = TC[t] || 'var(--text-muted)';
  return '<span class="tier-badge" style="background:'+c+'">'+t+'</span>';
}}
function decBadge(d) {{
  if (!d || d==='-') return '<span style="color:var(--text-muted)">—</span>';
  const c = DC[d] || 'var(--text-muted)';
  return '<span class="dec-badge" style="background:'+c+'">'+d+'</span>';
}}
function gatePill(g) {{
  const c = GC[g] || '#71717a';
  return '<span class="gate-pill" style="background:'+c+'">'+(GL[g]||g)+'</span>';
}}
function giokBadge(gir, ok) {{
  if (gir==null && ok==null) return '<span style="color:var(--text-muted)">—</span>';
  let out = '<span class="giok-wrap">';
  if (gir!=null) out += '<span class="giok-tag giok-gir">GIR '+gir+'/4</span>';
  if (ok!=null) {{
    const cls = ok===0 ? 'giok-ok-clean' : (ok>=2 ? 'giok-ok-warn' : 'giok-tag');
    out += '<span class="giok-tag '+cls+'">OK '+ok+'/4</span>';
  }}
  return out + '</span>';
}}
function fmtScore(v) {{
  return v==null ? '—' : '<span class="score-pill">'+v.toFixed(2)+'</span>';
}}
function fmtNum(v, d) {{
  return v==null ? '—' : v.toFixed(d||2);
}}

function renderBasket() {{
  const tb = document.getElementById('bk');
  tb.innerHTML = '';
  if (!D.length) {{
    tb.innerHTML = '<tr><td colspan="13" class="empty-row">Bugün hiçbir kapıda sinyal yok.</td></tr>';
    return;
  }}
  D.forEach(r => {{
    const tr = document.createElement('tr');
    tr.innerHTML = ''
      + '<td><a class="tv-link" href="https://www.tradingview.com/chart/?symbol=BIST:'+r.ticker+'" target="_blank">'+r.ticker+'</a></td>'
      + '<td>'+gatePill(r.gate)+'</td>'
      + '<td style="color:var(--nox-gold);font-weight:600">#'+(r.rank_in_gate||'—')+'</td>'
      + '<td>'+fmtScore(r.rank_score)+'</td>'
      + '<td style="color:var(--text-secondary);font-size:.7rem">'+(r.rt_subtype||'—')+'</td>'
      + '<td>'+tierBadge(r.rt_tier)+'</td>'
      + '<td>'+giokBadge(r.rt_gir, r.rt_ok)+'</td>'
      + '<td style="color:var(--text-secondary);font-size:.7rem">'+(r.as_subtype||'—')+'</td>'
      + '<td>'+decBadge(r.as_decision)+'</td>'
      + '<td style="color:var(--text-secondary);font-size:.7rem">'+(r.nox_dw_type||'—')+'</td>'
      + '<td>'+fmtNum(r.close, 2)+'</td>'
      + '<td style="color:var(--text-muted)">'+fmtNum(r.atr_pct, 2)+'</td>'
      + '<td style="color:'+(r.rs_score>0?'var(--nox-green)':'var(--nox-red)')+'">'+fmtNum(r.rs_score, 1)+'</td>';
    tb.appendChild(tr);
  }});
}}

function renderPerGate() {{
  const root = document.getElementById('pergate-sections');
  root.innerHTML = '';
  ['regime_trig', 'weekly_trig', 'alsat_trig'].forEach(g => {{
    const items = PG[g] || [];
    const sec = document.createElement('div');
    sec.className = 'gate-section';
    let body = '';
    if (!items.length) {{
      body = '<div class="empty-row">— sıfır sinyal —</div>';
    }} else {{
      body = '<div class="nox-table-wrap"><table><thead><tr>'
        + '<th>#</th><th>Hisse</th><th>Skor</th>'
        + '<th>RT subtype</th><th>RT tier</th><th>GIR/OK</th>'
        + '<th>AS subtype</th><th>AS</th><th>NOX</th>'
        + '</tr></thead><tbody>';
      items.forEach(r => {{
        body += '<tr>'
          + '<td style="color:var(--nox-gold);font-weight:600">'+r.rank+'</td>'
          + '<td><a class="tv-link" href="https://www.tradingview.com/chart/?symbol=BIST:'+r.ticker+'" target="_blank">'+r.ticker+'</a></td>'
          + '<td>'+fmtScore(r.rank_score)+'</td>'
          + '<td style="color:var(--text-secondary);font-size:.7rem">'+(r.rt_subtype||'—')+'</td>'
          + '<td>'+tierBadge(r.rt_tier)+'</td>'
          + '<td>'+giokBadge(r.rt_gir, r.rt_ok)+'</td>'
          + '<td style="color:var(--text-secondary);font-size:.7rem">'+(r.as_subtype||'—')+'</td>'
          + '<td>'+decBadge(r.as_decision)+'</td>'
          + '<td style="color:var(--text-secondary);font-size:.7rem">'+(r.nox_dw_type||'—')+'</td>'
          + '</tr>';
      }});
      body += '</tbody></table></div>';
    }}
    sec.innerHTML = '<h2>'+gatePill(g)+'  &nbsp; '+(GL[g])+' top-3</h2>' + body;
    root.appendChild(sec);
  }});
}}

renderBasket();
renderPerGate();
</script></body></html>"""
    return html


__all__ = ["render_html"]
