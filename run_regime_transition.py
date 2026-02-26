#!/usr/bin/env python3
"""
NOX Regime Transition Screener — Runner
========================================
BIST hisselerinde rejim gecis taramasi.
Sticky AL: rejim yukselince trade baslar, close < EMA21 olunca biter.
OE (Overextended) skoru ile risk degerlendirmesi.

Kullanim:
    python run_regime_transition.py                        # tum hisseler (AL aktif)
    python run_regime_transition.py --tickers THYAO ASELS  # spesifik
    python run_regime_transition.py --html                 # HTML rapor
    python run_regime_transition.py --csv                  # CSV kaydet
    python run_regime_transition.py --backtest 60           # son 60 bar backtest
    python run_regime_transition.py --backtest 60 --csv     # backtest + CSV
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
    compute_trailing_stop, _find_pivot_lows,
    compute_trade_state, calc_oe_score,
    RegimeTransitionSignal, RT_CFG, REGIME_NAMES,
)
from collections import Counter
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


def _entry_window(days_since):
    """Gecisten bu yana gecen gune gore giris penceresi etiketi.
    Backtest verisi: 0-1g WR %63 (Score4), 3-7g WR %52-54, 10-20g WR %58-60, 30+g dusuyor.
    """
    if days_since <= 1:
        return 'TAZE'       # En iyi giris penceresi
    elif days_since <= 7:
        return 'BEKLE'      # Konsolidasyon/pullback bolge
    elif days_since <= 20:
        return '2.DALGA'    # Ikinci firsat penceresi
    else:
        return 'GEC'        # Sinyal eskimis


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
    Sticky AL: trade aktif hisseleri listeler, OE skoru hesaplar."""
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

            # Sticky AL trade state
            trade_state = compute_trade_state(out['regime'], out['close'], out['ema21'])

            # Sadece trade aktif hisseleri listele
            if not trade_state['in_trade'].iloc[last]:
                continue

            # Trade baslangic bilgisi
            trade_start_idx = int(trade_state['trade_start_idx'].iloc[last])
            transition_date = df.index[trade_start_idx]
            transition_close = float(df['close'].iloc[trade_start_idx])
            gain_pct = (current_close - transition_close) / transition_close * 100
            days_since = (df.index[last] - transition_date).days

            # Gecis label: trade basladigi bardaki rejim degisimi
            prev_regime = int(out['regime'].iloc[trade_start_idx - 1]) if trade_start_idx > 0 else 0
            curr_regime_at_start = int(out['regime'].iloc[trade_start_idx])
            transition = f"{REGIME_NAMES.get(prev_regime, '?')}→{REGIME_NAMES.get(curr_regime_at_start, '?')}"

            # Stop degerleri
            initial_stop = out.get('initial_stop', 0.0)
            trailing_stop_val = out.get('trailing_stop', 0.0)
            eff_stop = max(initial_stop, trailing_stop_val) if initial_stop > 0 else trailing_stop_val

            # Meta degerler
            atr_val = float(out['atr'].iloc[last]) if pd.notna(out['atr'].iloc[last]) else 0.0
            cmf_val = float(out['cmf'].iloc[last]) if pd.notna(out['cmf'].iloc[last]) else 0.0
            adx_slope_val = float(out['adx_slope'].iloc[last]) if pd.notna(out['adx_slope'].iloc[last]) else 0.0
            exit_stage_val = int(out['exit_stage'].iloc[last])
            rvol_val = float(out['rvol'].iloc[last]) if pd.notna(out['rvol'].iloc[last]) else 0.0

            # Entry score
            entry_score = 0
            entry_parts = []
            atr_pct = (atr_val / current_close * 100) if current_close > 0 else 0
            # 1. Dusuk volatilite (ATR% < 3)
            if atr_pct < 3.0:
                entry_score += 1
                entry_parts.append('Vol dusuk: \u2713')
            else:
                entry_parts.append('Vol dusuk: \u2717')
            # 2. Erken giris (ADX slope < 0)
            if adx_slope_val < 0:
                entry_score += 1
                entry_parts.append('Erken giris: \u2713')
            else:
                entry_parts.append('Erken giris: \u2717')
            # 3. Buyume odasi (regime <= 2)
            if regime_val <= 2:
                entry_score += 1
                entry_parts.append('Buyume odasi: \u2713')
            else:
                entry_parts.append('Buyume odasi: \u2717')
            # 4. Pump filtresi (RVOL < 2)
            if rvol_val < 2.0:
                entry_score += 1
                entry_parts.append('Pump yok: \u2713')
            else:
                entry_parts.append('Pump yok: \u2717')
            entry_detail = ' | '.join(entry_parts)

            # OE skoru
            oe = calc_oe_score(df, out['ema21'])

            sig = RegimeTransitionSignal(
                ticker=ticker,
                date=df.index[last],
                regime=regime_val,
                regime_name=regime_name,
                trend_score=int(out['trend_score'].iloc[last]),
                participation_score=int(out['participation_score'].iloc[last]),
                expansion_score=int(out['expansion_score'].iloc[last]),
                exit_stage=exit_stage_val,
                transition=transition,
                close=current_close,
                transition_date=transition_date,
                transition_close=transition_close,
                gain_since_pct=round(gain_pct, 1),
                days_since=days_since,
                prev_regime=prev_regime,
                stop=round(eff_stop, 2),
                trailing_stop=round(trailing_stop_val, 2),
                entry_score=entry_score,
                entry_detail=entry_detail,
                oe_score=oe['oe_score'],
                oe_tags=oe['oe_tags'],
                oe_warning=oe['oe_warning'],
                atr=atr_val,
                adx=float(out['adx'].iloc[last]) if pd.notna(out['adx'].iloc[last]) else 0.0,
                cmf=cmf_val,
                rvol=rvol_val,
                di_spread=float(out['di_spread'].iloc[last]) if pd.notna(out['di_spread'].iloc[last]) else 0.0,
                adx_slope=adx_slope_val,
            )
            results.append(sig)

        except Exception as e:
            print(f"  ! {ticker}: {e}")
            continue

    date_str = last_date.strftime('%Y-%m-%d') if last_date else datetime.now().strftime('%Y-%m-%d')

    # Siralama: entry_score desc → regime desc → days_since asc
    results.sort(key=lambda s: (
        -s.entry_score,
        -s.regime,
        s.days_since,
        s.ticker,
    ))

    return results, n_scanned, date_str, regime_dist


# =============================================================================
# KONSOL RAPOR
# =============================================================================

def _regime_icon(regime):
    icons = {0: '·', 1: '◇', 2: '◆', 3: '★'}
    return icons.get(regime, '?')


def _oe_icon(score):
    if score <= 1:
        return '·'
    elif score == 2:
        return '!'
    else:
        return '!!'


def _fmt_date(dt):
    if dt is None:
        return '-'
    if hasattr(dt, 'strftime'):
        return dt.strftime('%m-%d')
    return str(dt)[-5:]


def _print_results(results, n_scanned, date_str, regime_dist, transitions_only=False):
    w = 140
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

    hdr = (f"  {'Hisse':<8} {'Gecis':<22} {'Tarih':>6} {'Gun':>4} {'Pencere':>8} {'Giris':>5} {'OE':>3} "
           f"{'Fiyat':>8} {'Getiri':>7} {'T':>2} {'P':>2} {'E':>2} "
           f"{'Exit':>4} {'ADX':>6} {'Slope':>7} {'CMF':>6} {'RVOL':>5} {'DI±':>6}")

    if results:
        print(f"\n  ▲ AL AKTIF ({len(results)} hisse)")
        print(f"  {'─' * w}")
        print(hdr)
        print(f"  {'─' * w}")
        for s in results:
            gain_str = f"{s.gain_since_pct:>+6.1f}%" if s.gain_since_pct != 0 else '   NEW'
            window = _entry_window(s.days_since)
            oe_str = f"{s.oe_score}" if s.oe_score < 3 else f"{s.oe_score}!"
            print(f"  {s.ticker:<8} {s.transition:<22} {_fmt_date(s.transition_date):>6} "
                  f"{s.days_since:>4} {window:>8} {s.entry_score:>4}/4 {oe_str:>3} "
                  f"{s.close:>8.2f} {gain_str:>7} "
                  f"{s.trend_score:>2} {s.participation_score:>2} {s.expansion_score:>2} "
                  f"{s.exit_stage:>4} {s.adx:>6.1f} {s.adx_slope:>+7.2f} "
                  f"{s.cmf:>6.3f} {s.rvol:>5.2f} {s.di_spread:>+6.1f}")
        print(f"  {'─' * w}")
    else:
        print(f"\n  Trade aktif hisse yok.")

    # Ozet
    print(f"\n{'═' * w}")
    oe_warn_cnt = sum(1 for s in results if s.oe_warning)
    print(f"  OZET: {len(results)} AL aktif hisse"
          f" | OE uyari: {oe_warn_cnt}"
          f" | {regime_dist.get(3,0)} FULL + {regime_dist.get(2,0)} TREND "
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
            'transition': s.transition,
            'transition_date': s.transition_date.strftime('%Y-%m-%d') if hasattr(s.transition_date, 'strftime') else '',
            'days_since': s.days_since,
            'entry_window': _entry_window(s.days_since),
            'gain_since_pct': s.gain_since_pct,
            'trend_score': s.trend_score,
            'participation_score': s.participation_score,
            'expansion_score': s.expansion_score,
            'exit_stage': s.exit_stage,
            'close': round(s.close, 2),
            'stop': round(s.stop, 2),
            'entry_score': s.entry_score,
            'oe_score': s.oe_score,
            'oe_tags': '|'.join(s.oe_tags) if s.oe_tags else '',
            'oe_warning': s.oe_warning,
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
        # Stop-fiyat uzakligi %
        stop_dist_pct = round((s.close - s.stop) / s.close * 100, 1) if s.stop > 0 and s.close > 0 else 0
        rows_data.append({
            'ticker': s.ticker,
            'regime': s.regime,
            'regime_name': s.regime_name,
            'transition': s.transition,
            'transition_date': s.transition_date.strftime('%Y-%m-%d') if hasattr(s.transition_date, 'strftime') else '',
            'transition_date_iso': s.transition_date.strftime('%Y-%m-%d') if hasattr(s.transition_date, 'strftime') else '',
            'days_since': s.days_since,
            'entry_window': _entry_window(s.days_since),
            'gain_since_pct': s.gain_since_pct,
            'trend_score': s.trend_score,
            'participation_score': s.participation_score,
            'expansion_score': s.expansion_score,
            'exit_stage': s.exit_stage,
            'close': round(s.close, 2),
            'stop': s.stop,
            'stop_dist_pct': stop_dist_pct,
            'entry_score': s.entry_score,
            'entry_detail': s.entry_detail,
            'oe_score': s.oe_score,
            'oe_tags': ', '.join(s.oe_tags) if s.oe_tags else '',
            'oe_warning': s.oe_warning,
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

/* ENTRY BADGE */
.entry-badge {{
  display: inline-block; padding: 3px 10px; border-radius: var(--radius-sm);
  font-size: 0.72rem; font-weight: 700; font-family: var(--font-mono);
  letter-spacing: 0.02em; cursor: help;
}}
.entry-4 {{ background: rgba(74,222,128,0.18); color: var(--nox-green); }}
.entry-3 {{ background: rgba(34,211,238,0.15); color: var(--nox-cyan); }}
.entry-2 {{ background: rgba(250,204,21,0.15); color: var(--nox-yellow); }}
.entry-1 {{ background: rgba(248,113,113,0.12); color: var(--nox-red); }}
.entry-0 {{ background: rgba(248,113,113,0.12); color: var(--nox-red); }}

/* OE BADGE */
.oe-badge {{
  display: inline-block; padding: 3px 10px; border-radius: var(--radius-sm);
  font-size: 0.72rem; font-weight: 700; font-family: var(--font-mono);
  letter-spacing: 0.02em; cursor: help;
}}
.oe-0 {{ background: rgba(74,222,128,0.12); color: var(--nox-green); }}
.oe-1 {{ background: rgba(74,222,128,0.12); color: var(--nox-green); }}
.oe-2 {{ background: rgba(250,204,21,0.15); color: var(--nox-yellow); }}
.oe-3 {{ background: rgba(248,113,113,0.18); color: var(--nox-red); }}
.oe-4 {{ background: rgba(248,113,113,0.22); color: var(--nox-red); }}

/* STOP CELL */
.stop-close {{ color: var(--nox-orange); font-weight: 700; }}
.stop-ok {{ color: var(--nox-red); }}

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

/* WINDOW BADGES */
.window-badge {{
  display: inline-block; padding: 3px 10px; border-radius: var(--radius-sm);
  font-size: 0.72rem; font-weight: 700; font-family: var(--font-mono);
  letter-spacing: 0.02em;
}}
.window-TAZE {{ background: rgba(74,222,128,0.18); color: var(--nox-green); }}
.window-BEKLE {{ background: rgba(250,204,21,0.15); color: var(--nox-yellow); }}
.window-2DALGA {{ background: rgba(34,211,238,0.12); color: var(--nox-cyan); }}
.window-GEC {{ background: rgba(113,113,122,0.12); color: var(--text-muted); }}
</style>
</head><body>
<div class="nox-container">
<div class="nox-header">
  <div class="nox-logo">NOX<span class="proj">project</span><span class="mode">regime transition screener</span></div>
  <div class="nox-meta">{now} · <b>{n_scanned}</b> hisse · <b>{len(results)}</b> AL aktif</div>
</div>

<div class="nox-filters" style="margin-bottom:12px">
  <div><label>Hisse</label><input type="text" id="fS" placeholder="ARA" oninput="render()"></div>
  <div><label>Regime</label>
  <select id="fRegime" onchange="render()"><option value="">Tumu</option>
  <option value="3">FULL_TREND</option><option value="2">TREND</option>
  <option value="1">GRI_BOLGE</option><option value="0">CHOPPY</option></select></div>
  <div><label>Tarih</label>
  <select id="fDate" onchange="render()"><option value="">Tumu</option>
  <option value="0">Bugun</option><option value="3">Son 3 gun</option>
  <option value="7">Son 1 hafta</option><option value="14">Son 2 hafta</option>
  <option value="30">Son 1 ay</option></select></div>
  <div><label>Pencere</label>
  <select id="fWindow" onchange="render()"><option value="">Tumu</option>
  <option value="TAZE">TAZE (0-1g)</option><option value="BEKLE">BEKLE (3-7g)</option>
  <option value="2.DALGA">2.DALGA (10-20g)</option><option value="GEC">GEC (30+g)</option></select></div>
  <div><label>OE≤</label><input type="number" id="fOE" value="" step="1" min="0" max="4" placeholder="max" oninput="render()"></div>
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
  document.getElementById('fDate').value='';
  document.getElementById('fWindow').value='';
  document.getElementById('fOE').value='';
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
  const fDateVal=document.getElementById('fDate').value;
  const fWindow=document.getElementById('fWindow').value;
  const fOE=parseInt(document.getElementById('fOE').value);
  const fExit=parseInt(document.getElementById('fExit').value);
  const fADX=parseFloat(document.getElementById('fADX').value)||0;

  let dateCutoff=null;
  if(fDateVal!==''){{
    const days=parseInt(fDateVal);
    dateCutoff=new Date();
    dateCutoff.setDate(dateCutoff.getDate()-days);
    dateCutoff.setHours(0,0,0,0);
  }}

  return rows.filter(r=>{{
    if(sr&&!r.ticker.includes(sr)) return false;
    if(fRegime!==''&&r.regime!==parseInt(fRegime)) return false;
    if(fWindow&&r.entry_window!==fWindow) return false;
    if(dateCutoff&&r.transition_date_iso){{
      const td=new Date(r.transition_date_iso+'T00:00:00');
      if(td<dateCutoff) return false;
    }}
    if(dateCutoff&&!r.transition_date_iso) return false;
    if(!isNaN(fOE)&&r.oe_score>fOE) return false;
    if(!isNaN(fExit)&&r.exit_stage>fExit) return false;
    if(fADX>0&&r.adx<fADX) return false;
    return true;
  }});
}}

function mkRegimeBadge(regime, name){{
  return `<span class="regime-badge regime-${{regime}}">${{name}}</span>`;
}}

function mkTransBadge(label){{
  return `<span class="trans-badge trans-al">▲ ${{label}}</span>`;
}}

function mkScoreCell(val){{
  return `<span class="score-cell score-${{val}}">${{val}}</span>`;
}}

function mkEntryBadge(score, detail){{
  const labels={{4:'GIR',3:'FIRSAT',2:'RISKLI',1:'GEC',0:'GEC'}};
  return `<span class="entry-badge entry-${{score}}" title="${{detail}}">${{labels[score]||'GEC'}} ${{score}}/4</span>`;
}}

function mkOeBadge(score, tags){{
  const labels={{0:'OK',1:'OK',2:'DIKKAT',3:'RISKLI',4:'RISKLI'}};
  const cls=Math.min(score, 4);
  return `<span class="oe-badge oe-${{cls}}" title="${{tags||''}}">${{labels[score]||'?'}} ${{score}}/4</span>`;
}}

function mkWindowBadge(win){{
  if(!win) return '<span style="color:var(--text-muted);font-size:0.72rem">—</span>';
  const cls=win==='2.DALGA'?'2DALGA':win;
  const tips={{'TAZE':'0-1 gun: en iyi giris penceresi','BEKLE':'3-7 gun: konsolidasyon/pullback','2.DALGA':'10-20 gun: ikinci firsat penceresi','GEC':'30+ gun: sinyal eskimis'}};
  return `<span class="window-badge window-${{cls}}" title="${{tips[win]||''}}">${{win}}</span>`;
}}

function mkStopCell(stop, dist, close){{
  if(!stop||stop<=0) return '<span style="color:var(--text-muted)">—</span>';
  const cls=dist<2?'stop-close':'stop-ok';
  return `<span class="${{cls}}" title="Fiyattan uzaklik: %${{dist}}">${{stop.toFixed(1)}}</span>`;
}}

function mkTable(rows, label, cssClass){{
  const sk=sortState[label];
  if(sk) rows=sortRows(rows, sk.col, sk.asc);
  rows=applyFilters(rows);
  if(!rows.length) return '<div style="color:var(--text-muted);padding:20px">Filtre kriterlerine uyan hisse yok.</div>';
  const srt=(c)=>`onclick="doSort('${{label}}','${{c}}')" style="cursor:pointer"`;
  let h=`<div class="section-title">${{label}}<span class="section-count ${{cssClass}}">${{rows.length}}</span></div>`;
  h+=`<div class="nox-table-wrap" style="margin-bottom:16px"><table><thead><tr>
  <th ${{srt('ticker')}}>Hisse</th>
  <th ${{srt('regime')}}>Regime</th>
  <th ${{srt('transition')}}>Gecis</th>
  <th ${{srt('transition_date')}}>Tarih</th>
  <th ${{srt('days_since')}}>Gun</th>
  <th ${{srt('entry_window')}}>Pencere</th>
  <th ${{srt('gain_since_pct')}}>Getiri</th>
  <th ${{srt('entry_score')}}>Giris</th>
  <th ${{srt('oe_score')}}>OE</th>
  <th ${{srt('trend_score')}}>T</th>
  <th ${{srt('participation_score')}}>P</th>
  <th ${{srt('expansion_score')}}>E</th>
  <th ${{srt('exit_stage')}}>Exit</th>
  <th ${{srt('close')}}>Fiyat</th>
  <th ${{srt('stop')}}>Stop</th>
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
    const gainStr=(r.gain_since_pct>0?'+':'')+r.gain_since_pct+'%';
    const dateStr=r.transition_date||'—';
    const daysStr=r.days_since+'g';
    h+=`<tr>
    <td><a class="tv-link" href="https://www.tradingview.com/chart/?symbol=BIST:${{r.ticker}}" target="_blank">${{r.ticker}}</a></td>
    <td>${{mkRegimeBadge(r.regime, r.regime_name)}}</td>
    <td>${{mkTransBadge(r.transition)}}</td>
    <td style="font-family:var(--font-mono);font-size:0.72rem;color:var(--text-muted)">${{dateStr}}</td>
    <td style="font-family:var(--font-mono);font-size:0.72rem;color:var(--text-muted)">${{daysStr}}</td>
    <td>${{mkWindowBadge(r.entry_window)}}</td>
    <td style="color:${{gainC}};font-weight:700;font-family:var(--font-mono)">${{gainStr}}</td>
    <td>${{mkEntryBadge(r.entry_score, r.entry_detail||'')}}</td>
    <td>${{mkOeBadge(r.oe_score, r.oe_tags||'')}}</td>
    <td>${{mkScoreCell(r.trend_score)}}</td>
    <td>${{mkScoreCell(r.participation_score)}}</td>
    <td>${{mkScoreCell(r.expansion_score)}}</td>
    <td class="exit-${{r.exit_stage}}" style="font-weight:700;font-family:var(--font-mono)">${{r.exit_stage}}</td>
    <td>${{r.close}}</td>
    <td>${{mkStopCell(r.stop, r.stop_dist_pct, r.close)}}</td>
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

  // Tek tablo: tum AL aktif hisseler
  let html=mkTable(D.rows, '▲ AL Aktif', 'cnt-al');
  document.getElementById('content').innerHTML=html;

  // Status
  const oeWarn=D.rows.filter(r=>r.oe_warning).length;
  document.getElementById('st').innerHTML=
    `<b>${{D.rows.length}}</b> AL aktif · OE uyari: <b>${{oeWarn}}</b> · ${{D.date}}`;
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
# BACKTEST
# =============================================================================

def _run_backtest(stock_dfs, weekly_dfs, N):
    """Son N bar'daki AL/SAT sinyallerini cikar, forward return hesapla.
    Her ticker 1 kez scan edilir (O(ticker) karmasiklik).

    Returns: (signals_list, summary_dict)
    """
    signals = []
    n_scanned = 0
    last_date = None
    lb = RT_CFG['stop_swing_lb']

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

            # Pivot lows (stop referansi icin)
            pivot_lows = _find_pivot_lows(df['low'], lb)

            n_bars = len(df)
            start_idx = max(0, n_bars - N)

            direction_s = out['direction']
            transition_s = out['transition']
            regime_s = out['regime']
            prev_regime_s = out['prev_regime']
            close_s = df['close']
            low_s = df['low']
            atr_s = out['atr']
            adx_s = out['adx']
            adx_slope_s = out['adx_slope']
            cmf_s = out['cmf']
            rvol_s = out['rvol']
            exit_stage_s = out['exit_stage']
            trend_score_s = out['trend_score']
            part_score_s = out['participation_score']
            exp_score_s = out['expansion_score']

            for i in range(start_idx, n_bars):
                d = str(direction_s.iloc[i])
                if d == 'TUT':
                    continue

                entry_price = float(close_s.iloc[i])
                if entry_price <= 0:
                    continue

                sig_date = df.index[i]
                trans = str(transition_s.iloc[i])
                regime_val = int(regime_s.iloc[i])
                prev_regime_val = int(prev_regime_s.iloc[i])
                atr_val = float(atr_s.iloc[i]) if pd.notna(atr_s.iloc[i]) else 0.0
                adx_val = float(adx_s.iloc[i]) if pd.notna(adx_s.iloc[i]) else 0.0
                adx_slope_val = float(adx_slope_s.iloc[i]) if pd.notna(adx_slope_s.iloc[i]) else 0.0
                cmf_val = float(cmf_s.iloc[i]) if pd.notna(cmf_s.iloc[i]) else 0.0
                rvol_val = float(rvol_s.iloc[i]) if pd.notna(rvol_s.iloc[i]) else 0.0
                exit_stage_val = int(exit_stage_s.iloc[i])
                t_score = int(trend_score_s.iloc[i])
                p_score = int(part_score_s.iloc[i])
                e_score = int(exp_score_s.iloc[i])
                atr_pct = (atr_val / entry_price * 100) if entry_price > 0 else 0.0

                # Stop: son pivot_low (bar i'den once) - 0.5 * ATR
                valid_pivots = pivot_lows.iloc[:i].dropna()
                if len(valid_pivots) > 0:
                    swing_low = float(valid_pivots.iloc[-1])
                else:
                    swing_low = float(low_s.iloc[max(0, i - 20):i].min())
                stop = swing_low - RT_CFG['stop_atr_initial'] * atr_val
                stop_dist_pct = (entry_price - stop) / entry_price * 100 if entry_price > 0 else 0.0

                # Entry Score (veri-odakli kriterler)
                entry_score = 0
                # 1. Dusuk volatilite (ATR% < 3 — en guclu tekil filtre)
                if atr_pct < 3.0:
                    entry_score += 1
                # 2. Erken giris (ADX slope < 0 — yapisal gecis, buyume odasi)
                if adx_slope_val < 0:
                    entry_score += 1
                # 3. Buyume odasi (regime <= 2 — henuz FULL degil)
                if regime_val <= 2:
                    entry_score += 1
                # 4. Pump filtresi (RVOL < 2 — asiri hacim = pump tuzagi)
                if rvol_val < 2.0:
                    entry_score += 1

                # Forward returns
                ret_5d = None
                ret_10d = None
                ret_20d = None
                for window, key in [(5, 'ret_5d'), (10, 'ret_10d'), (20, 'ret_20d')]:
                    end_idx = i + window
                    if end_idx < n_bars:
                        r = (float(close_s.iloc[end_idx]) - entry_price) / entry_price * 100
                        if d == 'SAT':
                            r = -r
                        if key == 'ret_5d':
                            ret_5d = round(r, 2)
                        elif key == 'ret_10d':
                            ret_10d = round(r, 2)
                        else:
                            ret_20d = round(r, 2)

                # Stop hit (sadece AL icin, 5G icerisinde)
                stop_hit = False
                stop_hit_bar = None
                if d == 'AL' and stop > 0:
                    for j in range(i + 1, min(i + 6, n_bars)):
                        if float(low_s.iloc[j]) < stop:
                            stop_hit = True
                            stop_hit_bar = j - i
                            break

                signals.append({
                    'ticker': ticker,
                    'signal_date': sig_date.strftime('%Y-%m-%d') if hasattr(sig_date, 'strftime') else str(sig_date),
                    'direction': d,
                    'transition': trans,
                    'regime': regime_val,
                    'prev_regime': prev_regime_val,
                    'entry_price': round(entry_price, 2),
                    'stop': round(stop, 2),
                    'stop_dist_pct': round(stop_dist_pct, 2),
                    'entry_score': entry_score,
                    'trend_score': t_score,
                    'participation_score': p_score,
                    'expansion_score': e_score,
                    'exit_stage': exit_stage_val,
                    'adx': round(adx_val, 2),
                    'adx_slope': round(adx_slope_val, 3),
                    'cmf': round(cmf_val, 4),
                    'atr_pct': round(atr_pct, 2),
                    'rvol': round(rvol_val, 2),
                    'ret_5d': ret_5d,
                    'ret_10d': ret_10d,
                    'ret_20d': ret_20d,
                    'stop_hit': stop_hit,
                    'stop_hit_bar': stop_hit_bar,
                })

        except Exception as e:
            print(f"  ! {ticker}: {e}")
            continue

    date_str = last_date.strftime('%Y-%m-%d') if last_date else datetime.now().strftime('%Y-%m-%d')
    summary = _compute_backtest_summary(signals)
    summary['n_scanned'] = n_scanned
    summary['date_str'] = date_str
    return signals, summary


def _compute_backtest_summary(signals):
    """Backtest sinyallerinden istatistik ozet cikar."""
    if not signals:
        return {'total': 0}

    df = pd.DataFrame(signals)
    summary = {'total': len(df)}

    al = df[df['direction'] == 'AL']
    sat = df[df['direction'] == 'SAT']
    summary['n_al'] = len(al)
    summary['n_sat'] = len(sat)

    # Genel pencere istatistikleri
    for window in ['5d', '10d', '20d']:
        col = f'ret_{window}'
        valid = df[df[col].notna()][col]
        n = len(valid)
        if n > 0:
            wr = (valid > 0).sum() / n * 100
            summary[f'all_{window}'] = {
                'n': n, 'wr': round(wr, 1),
                'avg': round(valid.mean(), 2),
                'med': round(valid.median(), 2),
            }
        else:
            summary[f'all_{window}'] = {'n': 0, 'wr': 0, 'avg': 0, 'med': 0}

    # Stop hit orani (AL, 5G icinde)
    al_with_5d = al[al['ret_5d'].notna()]
    if len(al_with_5d) > 0:
        summary['stop_pct_5d'] = round(al_with_5d['stop_hit'].sum() / len(al_with_5d) * 100, 1)
    else:
        summary['stop_pct_5d'] = 0.0

    # Yon bazinda
    summary['by_dir'] = {}
    for d_name, d_df in [('AL', al), ('SAT', sat)]:
        d_stats = {}
        for window in ['5d', '10d', '20d']:
            col = f'ret_{window}'
            valid = d_df[d_df[col].notna()][col]
            n = len(valid)
            if n > 0:
                wr = (valid > 0).sum() / n * 100
                d_stats[window] = {'n': n, 'wr': round(wr, 1), 'avg': round(valid.mean(), 2)}
            else:
                d_stats[window] = {'n': 0, 'wr': 0, 'avg': 0}
        d_stats['total'] = len(d_df)
        summary['by_dir'][d_name] = d_stats

    # Gecis tipi bazinda (AL, top 5)
    summary['by_transition_al'] = {}
    if len(al) > 0:
        trans_counts = al['transition'].value_counts()
        for trans_name in trans_counts.index[:8]:
            t_df = al[al['transition'] == trans_name]
            valid_5 = t_df[t_df['ret_5d'].notna()]['ret_5d']
            n = len(valid_5)
            if n >= 3:
                summary['by_transition_al'][trans_name] = {
                    'n': n,
                    'wr': round((valid_5 > 0).sum() / n * 100, 1),
                    'avg': round(valid_5.mean(), 2),
                }

    # Giris skoru bazinda (AL)
    summary['by_score_al'] = {}
    if len(al) > 0:
        for score in [4, 3, 2]:
            s_df = al[al['entry_score'] == score]
            valid_5 = s_df[s_df['ret_5d'].notna()]
            n = len(valid_5)
            if n > 0:
                wr = (valid_5['ret_5d'] > 0).sum() / n * 100
                stop_r = valid_5['stop_hit'].sum() / n * 100
                summary['by_score_al'][score] = {
                    'n': n, 'wr': round(wr, 1),
                    'avg': round(valid_5['ret_5d'].mean(), 2),
                    'stop_pct': round(stop_r, 1),
                }
        # <=1 grubu
        s_df = al[al['entry_score'] <= 1]
        valid_5 = s_df[s_df['ret_5d'].notna()]
        n = len(valid_5)
        if n > 0:
            wr = (valid_5['ret_5d'] > 0).sum() / n * 100
            stop_r = valid_5['stop_hit'].sum() / n * 100
            summary['by_score_al']['<=1'] = {
                'n': n, 'wr': round(wr, 1),
                'avg': round(valid_5['ret_5d'].mean(), 2),
                'stop_pct': round(stop_r, 1),
            }

    # Exit stage bazinda (AL)
    summary['by_exit_al'] = {}
    if len(al) > 0:
        for exit_val in [0, 1]:
            e_df = al[al['exit_stage'] == exit_val]
            valid_5 = e_df[e_df['ret_5d'].notna()]
            n = len(valid_5)
            if n > 0:
                wr = (valid_5['ret_5d'] > 0).sum() / n * 100
                stop_r = valid_5['stop_hit'].sum() / n * 100
                summary['by_exit_al'][exit_val] = {
                    'n': n, 'wr': round(wr, 1),
                    'avg': round(valid_5['ret_5d'].mean(), 2),
                    'stop_pct': round(stop_r, 1),
                }
        # >=2 grubu
        e_df = al[al['exit_stage'] >= 2]
        valid_5 = e_df[e_df['ret_5d'].notna()]
        n = len(valid_5)
        if n > 0:
            wr = (valid_5['ret_5d'] > 0).sum() / n * 100
            stop_r = valid_5['stop_hit'].sum() / n * 100
            summary['by_exit_al']['>=2'] = {
                'n': n, 'wr': round(wr, 1),
                'avg': round(valid_5['ret_5d'].mean(), 2),
                'stop_pct': round(stop_r, 1),
            }

    # Filtreli (AL, Score>=3, Exit<=1)
    if len(al) > 0:
        filt = al[(al['entry_score'] >= 3) & (al['exit_stage'] <= 1)]
        summary['filtered_al'] = {}
        for window in ['5d', '10d', '20d']:
            col = f'ret_{window}'
            valid = filt[filt[col].notna()][col]
            n = len(valid)
            if n > 0:
                wr = (valid > 0).sum() / n * 100
                summary['filtered_al'][window] = {
                    'n': n, 'wr': round(wr, 1), 'avg': round(valid.mean(), 2),
                }
            else:
                summary['filtered_al'][window] = {'n': 0, 'wr': 0, 'avg': 0}
        summary['filtered_al']['total'] = len(filt)
    else:
        summary['filtered_al'] = {'total': 0}

    return summary


def _print_backtest_summary(summary, N, n_scanned, date_str):
    """Backtest ozet ciktisi."""
    w = 70
    print(f"\n{'═' * w}")
    print(f"  REGIME TRANSITION BACKTEST — Son {N} bar")
    print(f"{'═' * w}")
    print(f"  {n_scanned} hisse, {summary['total']} sinyal "
          f"(AL:{summary.get('n_al', 0)}, SAT:{summary.get('n_sat', 0)})")

    # Genel
    print(f"\n  GENEL")
    print(f"  {'Pencere':<10} {'N':>6} {'WR%':>7} {'Ort%':>8} {'Med%':>8}")
    print(f"  {'─' * 45}")
    for window in ['5d', '10d', '20d']:
        s = summary.get(f'all_{window}', {})
        n = s.get('n', 0)
        if n > 0:
            print(f"  {window.upper():<10} {n:>6} {s['wr']:>6.1f}% {s['avg']:>+7.2f} {s['med']:>+7.2f}")
        else:
            print(f"  {window.upper():<10} {n:>6}    —       —       —")

    # Yon bazinda
    by_dir = summary.get('by_dir', {})
    if by_dir:
        print(f"\n  YON BAZINDA")
        print(f"  {'Yon':<6} {'N':>5} {'5G WR%':>8} {'5G Ort%':>9} {'10G WR%':>9} {'10G Ort%':>10}")
        print(f"  {'─' * 55}")
        for d_name in ['AL', 'SAT']:
            d = by_dir.get(d_name, {})
            total = d.get('total', 0)
            s5 = d.get('5d', {})
            s10 = d.get('10d', {})
            if total > 0:
                warn = '  ** ZAYIF' if d_name == 'SAT' and s5.get('wr', 0) < 50 else ''
                print(f"  {d_name:<6} {total:>5} "
                      f"{s5.get('wr', 0):>7.1f}% {s5.get('avg', 0):>+8.2f} "
                      f"{s10.get('wr', 0):>8.1f}% {s10.get('avg', 0):>+9.2f}{warn}")

    # Gecis tipi (AL, top entries)
    by_trans = summary.get('by_transition_al', {})
    if by_trans:
        print(f"\n  GECIS TIPI (AL, Top {min(5, len(by_trans))})")
        print(f"  {'Gecis':<28} {'N':>5} {'5G WR%':>8} {'5G Ort%':>9}")
        print(f"  {'─' * 55}")
        sorted_trans = sorted(by_trans.items(), key=lambda x: x[1]['n'], reverse=True)
        for trans_name, ts in sorted_trans[:5]:
            print(f"  {trans_name:<28} {ts['n']:>5} {ts['wr']:>7.1f}% {ts['avg']:>+8.2f}")

    # Giris skoru (AL)
    by_score = summary.get('by_score_al', {})
    if by_score:
        print(f"\n  GIRIS SKORU (AL)")
        print(f"  {'Skor':<6} {'N':>5} {'5G WR%':>8} {'5G Ort%':>9} {'Stop%':>7}")
        print(f"  {'─' * 45}")
        for score_key in [4, 3, 2, '<=1']:
            s = by_score.get(score_key)
            if s:
                label = str(score_key)
                print(f"  {label:<6} {s['n']:>5} {s['wr']:>7.1f}% {s['avg']:>+8.2f} {s['stop_pct']:>6.1f}%")

    # Exit stage (AL)
    by_exit = summary.get('by_exit_al', {})
    if by_exit:
        print(f"\n  EXIT STAGE (AL)")
        print(f"  {'Exit':<6} {'N':>5} {'5G WR%':>8} {'5G Ort%':>9} {'Stop%':>7}")
        print(f"  {'─' * 45}")
        for exit_key in [0, 1, '>=2']:
            s = by_exit.get(exit_key)
            if s:
                label = str(exit_key)
                print(f"  {label:<6} {s['n']:>5} {s['wr']:>7.1f}% {s['avg']:>+8.2f} {s['stop_pct']:>6.1f}%")

    # Filtreli
    filt = summary.get('filtered_al', {})
    filt_total = filt.get('total', 0)
    if filt_total > 0:
        f5 = filt.get('5d', {})
        f10 = filt.get('10d', {})
        print(f"\n  FILTRELI (AL, Score>=3, Exit<=1)")
        print(f"  N: {filt_total}  "
              f"5G: WR {f5.get('wr', 0):.1f}% Ort {f5.get('avg', 0):+.2f}%  "
              f"10G: WR {f10.get('wr', 0):.1f}% Ort {f10.get('avg', 0):+.2f}%")

    print(f"{'═' * w}")


def _save_backtest_csv(signals, date_str, output_dir):
    """Backtest sinyallerini CSV'ye kaydet."""
    if not signals:
        print(f"\n  Sinyal yok, CSV olusturulmadi.")
        return
    os.makedirs(output_dir, exist_ok=True)
    csv_df = pd.DataFrame(signals)
    fname = f"regime_transition_backtest_{date_str.replace('-', '')}.csv"
    path = os.path.join(output_dir, fname)
    csv_df.to_csv(path, index=False)
    print(f"\n  Backtest CSV: {path} ({len(signals)} sinyal)")


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
    parser.add_argument('--backtest', type=int, metavar='N',
                        help='Backtest: son N bar icin forward return hesapla')
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

    # ── 5. Backtest veya normal tarama ──────────────────────────────────────
    if args.backtest:
        print(f"\n  Backtest modu: son {args.backtest} bar...")
        t1 = time.time()
        signals, summary = _run_backtest(stock_dfs, weekly_dfs, args.backtest)
        n_scanned = summary.get('n_scanned', 0)
        date_str = summary.get('date_str', datetime.now().strftime('%Y-%m-%d'))
        print(f"  {n_scanned} hisse tarandi, {len(signals)} sinyal ({time.time() - t1:.1f}s)")
        _print_backtest_summary(summary, args.backtest, n_scanned, date_str)
        if args.csv:
            _save_backtest_csv(signals, date_str, args.output)
    else:
        print(f"\n  Regime transition taramasi...")
        t1 = time.time()
        results, n_scanned, date_str, regime_dist = _scan_all(stock_dfs, weekly_dfs)
        print(f"  {n_scanned} hisse tarandi ({time.time() - t1:.1f}s)")

        # ── 6. Rapor ─────────────────────────────────────────────────────────
        _print_results(results, n_scanned, date_str, regime_dist)

        # ── 7. CSV ───────────────────────────────────────────────────────────
        if args.csv:
            _save_csv(results, date_str, args.output)

        # ── 8. HTML ──────────────────────────────────────────────────────────
        if args.html:
            html = _generate_html(results, n_scanned, date_str, regime_dist)
            html_path = _save_html(html, date_str, args.output)
            subprocess.Popen(['open', html_path])

    print(f"\n  Toplam sure: {time.time() - t0:.1f}s")
    print(f"  NOX Regime Transition tamamlandi.\n")


if __name__ == '__main__':
    main()
