#!/usr/bin/env python3
"""
NOX Regime Transition Screener — Runner
========================================
BIST hisselerinde rejim gecis taramasi.
Trend + Participation + Expansion modeli ile erken giris, 3 asamali cikis.

Kullanim:
    python run_regime_transition.py                        # tum hisseler
    python run_regime_transition.py --tickers THYAO ASELS  # spesifik
    python run_regime_transition.py --html                 # HTML rapor
    python run_regime_transition.py --csv                  # CSV kaydet
    python run_regime_transition.py --transitions          # sadece gecisler
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

from markets.bist import data as data_mod
from markets.bist.regime_transition import (
    scan_regime_transition, find_last_transition,
    RegimeTransitionSignal, RT_CFG, REGIME_NAMES,
)
from core.reports import _NOX_CSS, _sanitize


# =============================================================================
# YARDIMCI
# =============================================================================

def _to_lower_cols(df):
    return df.rename(columns={
        'Open': 'open', 'High': 'high', 'Low': 'low',
        'Close': 'close', 'Volume': 'volume',
    })


def _is_halted(df, n=3):
    """Son n bar H==L ise halt/suspend."""
    if len(df) < n:
        return False
    tail = df.tail(n)
    return (tail['high'] == tail['low']).all()


def _to_weekly(df):
    weekly = df.resample('W-FRI').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
    }).dropna(subset=['close'])
    if len(weekly) > 0:
        last_friday = weekly.index[-1]
        last_data_date = df.index[-1]
        if last_data_date < last_friday:
            weekly = weekly.iloc[:-1]
    return weekly


# =============================================================================
# TARAMA
# =============================================================================

def _scan_all(stock_dfs, weekly_dfs=None, scan_bars=60):
    """Tum hisselerde regime transition taramasi.
    Son scan_bars bar icerisindeki en son anlamli gecisi bulur.
    Gecis tarihi + gecisten bugune getiri raporlar."""
    results = []
    n_scanned = 0
    last_date = None
    regime_dist = {0: 0, 1: 0, 2: 0, 3: 0}

    for ticker, df in stock_dfs.items():
        if len(df) < 100:
            continue
        if _is_halted(df):
            continue

        try:
            wk_df = weekly_dfs.get(ticker) if weekly_dfs else None
            out = scan_regime_transition(df, weekly_df=wk_df)
            n_scanned += 1

            if last_date is None and len(df) > 0:
                last_date = df.index[-1]

            last = len(df) - 1
            regime_val = int(out['regime'].iloc[last])
            regime_name = REGIME_NAMES.get(regime_val, '?')
            current_close = float(df['close'].iloc[last])

            regime_dist[regime_val] = regime_dist.get(regime_val, 0) + 1

            # Son scan_bars icerisindeki en son anlamli gecisi bul
            trans = find_last_transition(
                out['regime'], out['close'], df.index, scan_bars=scan_bars
            )

            if trans:
                direction = trans['direction']
                transition = trans['transition']
                transition_date = trans['date']
                transition_close = trans['close_at_transition']
                gain_pct = (current_close - transition_close) / transition_close * 100
                days_since = (df.index[last] - transition_date).days
                prev_regime = trans['from_regime']
            else:
                direction = 'TUT'
                transition = 'TUT'
                transition_date = None
                transition_close = 0.0
                gain_pct = 0.0
                days_since = 0
                prev_regime = regime_val

            sig = RegimeTransitionSignal(
                ticker=ticker,
                date=df.index[last],
                regime=regime_val,
                regime_name=regime_name,
                trend_score=int(out['trend_score'].iloc[last]),
                participation_score=int(out['participation_score'].iloc[last]),
                expansion_score=int(out['expansion_score'].iloc[last]),
                exit_stage=int(out['exit_stage'].iloc[last]),
                transition=transition,
                direction=direction,
                close=current_close,
                transition_date=transition_date,
                transition_close=transition_close,
                gain_since_pct=round(gain_pct, 1),
                days_since=days_since,
                prev_regime=prev_regime,
                atr=float(out['atr'].iloc[last]) if pd.notna(out['atr'].iloc[last]) else 0.0,
                adx=float(out['adx'].iloc[last]) if pd.notna(out['adx'].iloc[last]) else 0.0,
                cmf=float(out['cmf'].iloc[last]) if pd.notna(out['cmf'].iloc[last]) else 0.0,
                rvol=float(out['rvol'].iloc[last]) if pd.notna(out['rvol'].iloc[last]) else 0.0,
                di_spread=float(out['di_spread'].iloc[last]) if pd.notna(out['di_spread'].iloc[last]) else 0.0,
                adx_slope=float(out['adx_slope'].iloc[last]) if pd.notna(out['adx_slope'].iloc[last]) else 0.0,
            )
            results.append(sig)

        except Exception as e:
            print(f"  ! {ticker}: {e}")
            continue

    date_str = last_date.strftime('%Y-%m-%d') if last_date else datetime.now().strftime('%Y-%m-%d')

    # Siralama: once AL gecisler (getiriye gore), sonra SAT, sonra TUT
    results.sort(key=lambda s: (
        0 if s.direction == 'AL' else 1 if s.direction == 'SAT' else 2,
        -s.regime,
        -s.gain_since_pct,
        s.ticker,
    ))

    return results, n_scanned, date_str, regime_dist


# =============================================================================
# KONSOL RAPOR
# =============================================================================

def _regime_icon(regime):
    icons = {0: '·', 1: '◇', 2: '◆', 3: '★'}
    return icons.get(regime, '?')


def _dir_icon(direction):
    icons = {'AL': '▲', 'SAT': '▼', 'TUT': '─'}
    return icons.get(direction, '?')


def _fmt_date(dt):
    if dt is None:
        return '-'
    if hasattr(dt, 'strftime'):
        return dt.strftime('%m-%d')
    return str(dt)[-5:]


def _print_results(results, n_scanned, date_str, regime_dist, transitions_only=False):
    w = 130
    print(f"\n{'═' * w}")
    print(f"  NOX REGIME TRANSITION SCREENER — {date_str} — {n_scanned} hisse tarandi")
    print(f"{'═' * w}")

    # Regime dagilimi
    print(f"\n  Regime Dagilimi:")
    for r in range(4):
        name = REGIME_NAMES.get(r, '?')
        cnt = regime_dist.get(r, 0)
        bar = '█' * (cnt // 5) + '░' * max(0, 20 - cnt // 5)
        print(f"    {_regime_icon(r)} {name:<12} {cnt:>4}  {bar}")

    # Gecisler
    transitions_al = [s for s in results if s.direction == 'AL']
    transitions_sat = [s for s in results if s.direction == 'SAT']

    hdr = (f"  {'Hisse':<8} {'Gecis':<22} {'Tarih':>6} {'Gun':>4} {'Fiyat':>8} "
           f"{'Getiri':>7} {'T':>2} {'P':>2} {'E':>2} "
           f"{'Exit':>4} {'ADX':>6} {'Slope':>7} {'CMF':>6} {'RVOL':>5} {'DI±':>6}")

    if transitions_al:
        print(f"\n  ▲ AL GECISLERI ({len(transitions_al)})")
        print(f"  {'─' * w}")
        print(hdr)
        print(f"  {'─' * w}")
        for s in transitions_al:
            gain_str = f"{s.gain_since_pct:>+6.1f}%" if s.gain_since_pct != 0 else '   NEW'
            print(f"  {s.ticker:<8} {s.transition:<22} {_fmt_date(s.transition_date):>6} "
                  f"{s.days_since:>4} {s.close:>8.2f} "
                  f"{gain_str:>7} {s.trend_score:>2} {s.participation_score:>2} {s.expansion_score:>2} "
                  f"{s.exit_stage:>4} {s.adx:>6.1f} {s.adx_slope:>+7.2f} "
                  f"{s.cmf:>6.3f} {s.rvol:>5.2f} {s.di_spread:>+6.1f}")
        print(f"  {'─' * w}")

    if transitions_sat:
        print(f"\n  ▼ SAT GECISLERI ({len(transitions_sat)})")
        print(f"  {'─' * w}")
        print(hdr)
        print(f"  {'─' * w}")
        for s in transitions_sat:
            gain_str = f"{s.gain_since_pct:>+6.1f}%"
            print(f"  {s.ticker:<8} {s.transition:<22} {_fmt_date(s.transition_date):>6} "
                  f"{s.days_since:>4} {s.close:>8.2f} "
                  f"{gain_str:>7} {s.trend_score:>2} {s.participation_score:>2} {s.expansion_score:>2} "
                  f"{s.exit_stage:>4} {s.adx:>6.1f} {s.adx_slope:>+7.2f} "
                  f"{s.cmf:>6.3f} {s.rvol:>5.2f} {s.di_spread:>+6.1f}")
        print(f"  {'─' * w}")

    if not transitions_only:
        # Regime bazinda gruplama (sadece TUT olanlar)
        for r in [3, 2, 1]:
            group = [s for s in results if s.regime == r and s.direction == 'TUT']
            if not group:
                continue
            name = REGIME_NAMES.get(r, '?')
            print(f"\n  {_regime_icon(r)} {name} ({len(group)} hisse)")
            print(f"  {'─' * w}")
            for s in group[:15]:
                print(f"  {s.ticker:<8} {s.close:>8.2f} "
                      f"T:{s.trend_score} P:{s.participation_score} E:{s.expansion_score} "
                      f"Exit:{s.exit_stage} ADX:{s.adx:.1f} CMF:{s.cmf:.3f}")
            if len(group) > 15:
                print(f"  ...ve {len(group) - 15} hisse daha")
            print(f"  {'─' * w}")

    # Ozet
    print(f"\n{'═' * w}")
    print(f"  OZET: {len(transitions_al)} AL gecis + {len(transitions_sat)} SAT gecis "
          f"| {regime_dist.get(3,0)} FULL + {regime_dist.get(2,0)} TREND "
          f"+ {regime_dist.get(1,0)} GRI + {regime_dist.get(0,0)} CHOPPY")
    print(f"{'═' * w}")


# =============================================================================
# CSV
# =============================================================================

def _save_csv(results, date_str, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    rows = []
    for s in results:
        rows.append({
            'ticker': s.ticker,
            'date': s.date.strftime('%Y-%m-%d') if hasattr(s.date, 'strftime') else str(s.date),
            'regime': s.regime,
            'regime_name': s.regime_name,
            'direction': s.direction,
            'transition': s.transition,
            'transition_date': s.transition_date.strftime('%Y-%m-%d') if hasattr(s.transition_date, 'strftime') else '',
            'days_since': s.days_since,
            'gain_since_pct': s.gain_since_pct,
            'trend_score': s.trend_score,
            'participation_score': s.participation_score,
            'expansion_score': s.expansion_score,
            'exit_stage': s.exit_stage,
            'close': round(s.close, 2),
            'adx': round(s.adx, 2),
            'adx_slope': round(s.adx_slope, 3),
            'cmf': round(s.cmf, 4),
            'rvol': round(s.rvol, 2),
            'di_spread': round(s.di_spread, 2),
            'atr': round(s.atr, 4),
        })
    if rows:
        csv_df = pd.DataFrame(rows)
        fname = f"regime_transition_{date_str.replace('-', '')}.csv"
        path = os.path.join(output_dir, fname)
        csv_df.to_csv(path, index=False)
        print(f"\n  CSV: {path}")
    else:
        print(f"\n  Sinyal yok, CSV olusturulmadi.")


# =============================================================================
# HTML RAPOR
# =============================================================================

def _generate_html(results, n_scanned, date_str, regime_dist):
    now = datetime.now().strftime('%d.%m.%Y %H:%M')

    rows_data = []
    for s in results:
        rows_data.append({
            'ticker': s.ticker,
            'regime': s.regime,
            'regime_name': s.regime_name,
            'direction': s.direction,
            'transition': s.transition,
            'transition_date': s.transition_date.strftime('%Y-%m-%d') if hasattr(s.transition_date, 'strftime') else '',
            'days_since': s.days_since,
            'gain_since_pct': s.gain_since_pct,
            'trend_score': s.trend_score,
            'participation_score': s.participation_score,
            'expansion_score': s.expansion_score,
            'exit_stage': s.exit_stage,
            'close': round(s.close, 2),
            'adx': round(s.adx, 1),
            'adx_slope': round(s.adx_slope, 2),
            'cmf': round(s.cmf, 3),
            'rvol': round(s.rvol, 2),
            'di_spread': round(s.di_spread, 1),
        })

    data = {
        'rows': _sanitize(rows_data),
        'n_scanned': n_scanned,
        'date': date_str,
        'regime_dist': regime_dist,
    }
    data_json = json.dumps(data, ensure_ascii=False)

    html = f"""<!DOCTYPE html>
<html lang="tr"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NOX — Regime Transition · {now}</title>
<style>{_NOX_CSS}

/* REGIME BADGES */
.regime-badge {{
  display: inline-block; padding: 3px 10px; border-radius: var(--radius-sm);
  font-size: 0.72rem; font-weight: 700; font-family: var(--font-mono);
  letter-spacing: 0.02em;
}}
.regime-0 {{ background: rgba(113,113,122,0.15); color: var(--text-muted); }}
.regime-1 {{ background: rgba(250,204,21,0.12); color: var(--nox-yellow); }}
.regime-2 {{ background: rgba(34,211,238,0.12); color: var(--nox-cyan); }}
.regime-3 {{ background: rgba(74,222,128,0.15); color: var(--nox-green); }}

/* TRANSITION BADGES */
.trans-badge {{
  display: inline-flex; align-items: center; gap: 4px;
  padding: 3px 10px; border-radius: var(--radius-sm);
  font-size: 0.72rem; font-weight: 700; font-family: var(--font-mono);
}}
.trans-al {{ background: rgba(74,222,128,0.15); color: var(--nox-green); }}
.trans-sat {{ background: rgba(248,113,113,0.15); color: var(--nox-red); }}
.trans-tut {{ background: rgba(113,113,122,0.08); color: var(--text-muted); }}

/* SCORE CELL */
.score-cell {{
  display: inline-block; width: 22px; height: 22px; line-height: 22px;
  text-align: center; border-radius: 4px; font-size: 0.72rem;
  font-weight: 700; font-family: var(--font-mono);
}}
.score-0 {{ background: rgba(113,113,122,0.1); color: var(--text-muted); }}
.score-1 {{ background: rgba(250,204,21,0.12); color: var(--nox-yellow); }}
.score-2 {{ background: rgba(34,211,238,0.12); color: var(--nox-cyan); }}
.score-3 {{ background: rgba(74,222,128,0.15); color: var(--nox-green); }}

/* EXIT STAGE */
.exit-0 {{ color: var(--nox-green); }}
.exit-1 {{ color: var(--nox-yellow); }}
.exit-2 {{ color: var(--nox-orange); }}
.exit-3 {{ color: var(--nox-red); }}

/* DIST BAR */
.dist-bar {{
  display: flex; gap: 4px; margin-bottom: 20px;
  font-family: var(--font-mono); font-size: 0.75rem;
}}
.dist-item {{
  display: flex; align-items: center; gap: 6px;
  background: var(--bg-card); border: 1px solid var(--border-subtle);
  border-radius: 20px; padding: 6px 14px;
  font-weight: 500; flex-wrap: nowrap;
}}
.dist-cnt {{
  font-weight: 700; font-size: 0.85rem;
}}

/* SECTION */
.section-title {{
  font-size: 0.9rem; font-weight: 700; padding: 14px 0 8px;
  color: var(--text-secondary); font-family: var(--font-display);
  display: flex; align-items: center; gap: 8px;
}}
.section-count {{
  font-family: var(--font-mono); font-size: 0.72rem;
  padding: 2px 8px; border-radius: 10px;
}}
.cnt-al {{ background: rgba(74,222,128,0.12); color: var(--nox-green); }}
.cnt-sat {{ background: rgba(248,113,113,0.12); color: var(--nox-red); }}
</style>
</head><body>
<div class="nox-container">
<div class="nox-header">
  <div class="nox-logo">NOX<span class="proj">project</span><span class="mode">regime transition screener</span></div>
  <div class="nox-meta">{now} · <b>{n_scanned}</b> hisse</div>
</div>

<div class="nox-filters" style="margin-bottom:12px">
  <div><label>Hisse</label><input type="text" id="fS" placeholder="ARA" oninput="render()"></div>
  <div><label>Regime</label>
  <select id="fRegime" onchange="render()"><option value="">Tumu</option>
  <option value="3">FULL_TREND</option><option value="2">TREND</option>
  <option value="1">GRI_BOLGE</option><option value="0">CHOPPY</option></select></div>
  <div><label>Gecis</label>
  <select id="fDir" onchange="render()"><option value="">Tumu</option>
  <option value="AL">AL</option><option value="SAT">SAT</option>
  <option value="TUT">TUT</option></select></div>
  <div><label>Exit≤</label><input type="number" id="fExit" value="" step="1" min="0" max="3" placeholder="max" oninput="render()"></div>
  <div><label>ADX≥</label><input type="number" id="fADX" value="0" step="5" min="0" oninput="render()"></div>
  <div><button class="nox-btn" onclick="resetF()">Sifirla</button></div>
</div>

<div id="dist-bar"></div>
<div id="content"></div>
<div class="nox-status" id="st"></div>
</div>

<script>
const D={data_json};

function resetF(){{
  document.getElementById('fS').value='';
  document.getElementById('fRegime').value='';
  document.getElementById('fDir').value='';
  document.getElementById('fExit').value='';
  document.getElementById('fADX').value='0';
  render();
}}

function sortRows(rows, col, asc){{
  return [...rows].sort((a,b)=>{{
    let va=a[col],vb=b[col];
    if(typeof va==='string') return asc?va.localeCompare(vb):vb.localeCompare(va);
    if(typeof va==='boolean') return asc?(va?-1:1)-(vb?-1:1):(vb?-1:1)-(va?-1:1);
    return asc?((va||0)-(vb||0)):((vb||0)-(va||0));
  }});
}}

let sortState={{}};
function doSort(tbl, col){{
  const key=tbl;
  if(!sortState[key]||sortState[key].col!==col)
    sortState[key]={{col:col,asc:col==='ticker'}};
  else sortState[key].asc=!sortState[key].asc;
  render();
}}

function applyFilters(rows){{
  const sr=document.getElementById('fS').value.toUpperCase();
  const fRegime=document.getElementById('fRegime').value;
  const fDir=document.getElementById('fDir').value;
  const fExit=parseInt(document.getElementById('fExit').value);
  const fADX=parseFloat(document.getElementById('fADX').value)||0;
  return rows.filter(r=>{{
    if(sr&&!r.ticker.includes(sr)) return false;
    if(fRegime!==''&&r.regime!==parseInt(fRegime)) return false;
    if(fDir&&r.direction!==fDir) return false;
    if(!isNaN(fExit)&&r.exit_stage>fExit) return false;
    if(fADX>0&&r.adx<fADX) return false;
    return true;
  }});
}}

function mkRegimeBadge(regime, name){{
  return `<span class="regime-badge regime-${{regime}}">${{name}}</span>`;
}}

function mkTransBadge(dir, label){{
  if(dir==='AL') return `<span class="trans-badge trans-al">▲ ${{label}}</span>`;
  if(dir==='SAT') return `<span class="trans-badge trans-sat">▼ ${{label}}</span>`;
  return `<span class="trans-badge trans-tut">─</span>`;
}}

function mkScoreCell(val){{
  return `<span class="score-cell score-${{val}}">${{val}}</span>`;
}}

function mkTable(rows, label, cssClass){{
  const sk=sortState[label];
  if(sk) rows=sortRows(rows, sk.col, sk.asc);
  rows=applyFilters(rows);
  if(!rows.length) return '';
  const srt=(c)=>`onclick="doSort('${{label}}','${{c}}')" style="cursor:pointer"`;
  let h=`<div class="section-title">${{label}}<span class="section-count ${{cssClass}}">${{rows.length}}</span></div>`;
  h+=`<div class="nox-table-wrap" style="margin-bottom:16px"><table><thead><tr>
  <th ${{srt('ticker')}}>Hisse</th>
  <th ${{srt('regime')}}>Regime</th>
  <th ${{srt('direction')}}>Gecis</th>
  <th ${{srt('transition_date')}}>Tarih</th>
  <th ${{srt('days_since')}}>Gun</th>
  <th ${{srt('gain_since_pct')}}>Getiri</th>
  <th ${{srt('trend_score')}}>T</th>
  <th ${{srt('participation_score')}}>P</th>
  <th ${{srt('expansion_score')}}>E</th>
  <th ${{srt('exit_stage')}}>Exit</th>
  <th ${{srt('close')}}>Fiyat</th>
  <th ${{srt('adx')}}>ADX</th>
  <th ${{srt('adx_slope')}}>Slope</th>
  <th ${{srt('cmf')}}>CMF</th>
  <th ${{srt('rvol')}}>RVOL</th>
  <th ${{srt('di_spread')}}>DI±</th>
  </tr></thead><tbody>`;
  rows.forEach(r=>{{
    const slopeC=r.adx_slope>0?'var(--nox-green)':r.adx_slope<-0.3?'var(--nox-red)':'var(--text-muted)';
    const cmfC=r.cmf>0?'var(--nox-green)':r.cmf<0?'var(--nox-red)':'var(--text-muted)';
    const diC=r.di_spread>5?'var(--nox-green)':r.di_spread<-5?'var(--nox-red)':'var(--text-muted)';
    const rvolC=r.rvol>=1.5?'var(--nox-green)':r.rvol>=1.0?'var(--nox-cyan)':'var(--text-muted)';
    const gainC=r.gain_since_pct>0?'var(--nox-green)':r.gain_since_pct<0?'var(--nox-red)':'var(--text-muted)';
    const gainStr=r.direction==='TUT'?'—':((r.gain_since_pct>0?'+':'')+r.gain_since_pct+'%');
    const dateStr=r.transition_date||'—';
    const daysStr=r.direction==='TUT'?'—':r.days_since+'g';
    h+=`<tr>
    <td><a class="tv-link" href="https://www.tradingview.com/chart/?symbol=BIST:${{r.ticker}}" target="_blank">${{r.ticker}}</a></td>
    <td>${{mkRegimeBadge(r.regime, r.regime_name)}}</td>
    <td>${{mkTransBadge(r.direction, r.transition)}}</td>
    <td style="font-family:var(--font-mono);font-size:0.72rem;color:var(--text-muted)">${{dateStr}}</td>
    <td style="font-family:var(--font-mono);font-size:0.72rem;color:var(--text-muted)">${{daysStr}}</td>
    <td style="color:${{gainC}};font-weight:700;font-family:var(--font-mono)">${{gainStr}}</td>
    <td>${{mkScoreCell(r.trend_score)}}</td>
    <td>${{mkScoreCell(r.participation_score)}}</td>
    <td>${{mkScoreCell(r.expansion_score)}}</td>
    <td class="exit-${{r.exit_stage}}" style="font-weight:700;font-family:var(--font-mono)">${{r.exit_stage}}</td>
    <td>${{r.close}}</td>
    <td>${{r.adx}}</td>
    <td style="color:${{slopeC}}">${{r.adx_slope>0?'+':''}}${{r.adx_slope}}</td>
    <td style="color:${{cmfC}}">${{r.cmf}}</td>
    <td style="color:${{rvolC}}">${{r.rvol}}</td>
    <td style="color:${{diC}}">${{r.di_spread>0?'+':''}}${{r.di_spread}}</td>
    </tr>`;
  }});
  h+=`</tbody></table></div>`;
  return h;
}}

function render(){{
  // Distribution bar
  const rd=D.regime_dist;
  const names={{0:'CHOPPY',1:'GRI_BOLGE',2:'TREND',3:'FULL_TREND'}};
  const colors={{0:'var(--text-muted)',1:'var(--nox-yellow)',2:'var(--nox-cyan)',3:'var(--nox-green)'}};
  let db='<div class="dist-bar">';
  [3,2,1,0].forEach(r=>{{
    db+=`<div class="dist-item" style="border-color:${{colors[r]}}30">
      <span style="color:${{colors[r]}}">${{names[r]}}</span>
      <span class="dist-cnt" style="color:${{colors[r]}}">${{rd[r]||0}}</span>
    </div>`;
  }});
  db+='</div>';
  document.getElementById('dist-bar').innerHTML=db;

  // Tables
  const all=D.rows;
  const al=all.filter(r=>r.direction==='AL');
  const sat=all.filter(r=>r.direction==='SAT');
  const tut=all.filter(r=>r.direction==='TUT');

  let html='';
  html+=mkTable(al, '▲ AL Gecisleri', 'cnt-al');
  html+=mkTable(sat, '▼ SAT Gecisleri', 'cnt-sat');
  html+=mkTable(tut, '─ Regime Degismedi (TUT)', '');

  document.getElementById('content').innerHTML=html;

  // Status
  document.getElementById('st').innerHTML=
    `<b>${{al.length}}</b> AL gecis · <b>${{sat.length}}</b> SAT gecis · <b>${{tut.length}}</b> TUT · ${{D.date}}`;
}}

render();
</script></body></html>"""
    return html


def _save_html(html_content, date_str, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    fname = f"regime_transition_{date_str.replace('-', '')}.html"
    path = os.path.join(output_dir, fname)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"\n  HTML: {path}")
    return path


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="NOX Regime Transition Screener")
    parser.add_argument('--tickers', nargs='*', help='Spesifik ticker(lar)')
    parser.add_argument('--period', default='2y', help='Veri periyodu (default: 2y)')
    parser.add_argument('--csv', action='store_true', help='CSV kaydet')
    parser.add_argument('--html', action='store_true', help='HTML rapor kaydet ve ac')
    parser.add_argument('--output', default='output', help='Cikti dizini')
    parser.add_argument('--transitions', action='store_true',
                        help='Sadece gecis olan hisseleri goster')
    args = parser.parse_args()

    w = 80
    print(f"\n{'═' * w}")
    print(f"  NOX REGIME TRANSITION SCREENER")
    print(f"{'═' * w}")

    # ── 1. Ticker listesi ────────────────────────────────────────────────────
    if args.tickers:
        tickers = [t.upper() for t in args.tickers]
        print(f"\n  {len(tickers)} ticker belirtildi: {', '.join(tickers)}")
    else:
        print(f"\n  Ticker listesi aliniyor...")
        tickers = data_mod.get_all_bist_tickers()
        print(f"  {len(tickers)} ticker bulundu.")

    # ── 2. Veri yukleme ──────────────────────────────────────────────────────
    print(f"\n  Veri yukleniyor (period={args.period})...")
    t0 = time.time()
    all_data = data_mod.fetch_data(tickers, period=args.period)
    print(f"  {len(all_data)} hisse yuklendi ({time.time() - t0:.1f}s)")

    if not all_data:
        print("  HATA: Hicbir hisse verisi yuklenemedi!")
        sys.exit(1)

    # ── 3. Lowercase donusum ─────────────────────────────────────────────────
    stock_dfs = {ticker: _to_lower_cols(df) for ticker, df in all_data.items()}

    # ── 4. Haftalik resample (trend score icin) ──────────────────────────────
    print(f"\n  Haftalik resample...")
    weekly_dfs = {t: _to_weekly(df) for t, df in stock_dfs.items()}
    weekly_dfs = {t: df for t, df in weekly_dfs.items() if len(df) >= 25}

    # ── 5. Tarama ────────────────────────────────────────────────────────────
    print(f"\n  Regime transition taramasi...")
    t1 = time.time()
    results, n_scanned, date_str, regime_dist = _scan_all(stock_dfs, weekly_dfs)
    print(f"  {n_scanned} hisse tarandi ({time.time() - t1:.1f}s)")

    # ── 6. Rapor ─────────────────────────────────────────────────────────────
    _print_results(results, n_scanned, date_str, regime_dist,
                   transitions_only=args.transitions)

    # ── 7. CSV ───────────────────────────────────────────────────────────────
    if args.csv:
        _save_csv(results, date_str, args.output)

    # ── 8. HTML ──────────────────────────────────────────────────────────────
    if args.html:
        html = _generate_html(results, n_scanned, date_str, regime_dist)
        html_path = _save_html(html, date_str, args.output)
        subprocess.Popen(['open', html_path])

    print(f"\n  Toplam sure: {time.time() - t0:.1f}s")
    print(f"  NOX Regime Transition tamamlandi.\n")


if __name__ == '__main__':
    main()
