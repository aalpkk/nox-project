#!/usr/bin/env python3
"""
NOX Forward Test Aracı
=======================
output/ klasöründeki tarama CSV çıktılarını (NOX v3, SMC, Pine, Divergence) okuyup,
sinyal tarihinden itibaren gerçek fiyat getirilerini (1g, 3g, 5g) hesaplayarak
NOX dark-theme interaktif HTML raporu oluşturur.

Kullanım:
    python run_forward_test.py                  # en son CSV'ler
    python run_forward_test.py --date 20260220  # belirli tarih
    python run_forward_test.py --open           # raporu tarayıcıda aç
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from statistics import median

import numpy as np
import pandas as pd

from markets.bist import data as data_mod
from core.reports import _NOX_CSS, _sanitize


# =============================================================================
# 1. CSV KEŞİF & PARSE
# =============================================================================

# Dosya adı → screener tipi eşleştirme
_CSV_PATTERNS = [
    (re.compile(r'^nox_v3_signals_weekly_(\d{8})\.csv$'), 'nox_v3_weekly'),
    (re.compile(r'^nox_v3_signals_(\d{8})\.csv$'),        'nox_v3_daily'),
    (re.compile(r'^nox_smc_signals_(\d{8})\.csv$'),       'smc'),
    (re.compile(r'^pine_signals_(\d{8})\.csv$'),          'pine'),
    (re.compile(r'^nox_divergence_(\d{8})\.csv$'),        'divergence'),
]


def discover_csvs(output_dir, target_date=None):
    """output/ dizinindeki CSV'leri keşfet ve screener tipine göre grupla.
    target_date: 'YYYYMMDD' formatında; None ise en son tarih kullanılır."""
    found = {}  # {screener: [(date_str, path), ...]}
    for fname in os.listdir(output_dir):
        for pat, screener in _CSV_PATTERNS:
            m = pat.match(fname)
            if m:
                date_str = m.group(1)
                found.setdefault(screener, []).append((date_str, os.path.join(output_dir, fname)))
                break

    # Tarih filtresi veya en son tarih
    result = {}
    if target_date:
        for scr, items in found.items():
            for d, p in items:
                if d == target_date:
                    result[scr] = (d, p)
    else:
        for scr, items in found.items():
            items.sort(key=lambda x: x[0], reverse=True)
            result[scr] = items[0]

    return result


def _parse_nox_v3(path, screener_name):
    """NOX v3 daily/weekly CSV parse → normalize sinyal listesi."""
    df = pd.read_csv(path)
    signals = []
    for _, row in df.iterrows():
        sig = str(row.get('signal', '')).strip()
        if sig == 'PIVOT_AL':
            direction = 'AL'
        elif sig == 'PIVOT_SAT':
            direction = 'SAT'
        else:
            continue
        signals.append({
            'screener': screener_name,
            'ticker': str(row['ticker']).strip(),
            'signal_date': str(row['signal_date']).strip(),
            'direction': direction,
            'signal_type': sig,
            'entry_price': float(row['close']),
            'quality': None,
        })
    return signals


def _parse_smc(path):
    """SMC CSV parse → normalize sinyal listesi."""
    df = pd.read_csv(path)
    signals = []
    for _, row in df.iterrows():
        d = str(row.get('direction', '')).strip().upper()
        if d == 'BUY':
            direction = 'AL'
        elif d == 'SELL':
            direction = 'SAT'
        else:
            continue
        signals.append({
            'screener': 'smc',
            'ticker': str(row['ticker']).strip(),
            'signal_date': str(row['signal_date']).strip(),
            'direction': direction,
            'signal_type': str(row.get('pattern', '')).strip(),
            'entry_price': float(row['close']),
            'quality': int(row['quality']) if pd.notna(row.get('quality')) else None,
        })
    return signals


def _parse_pine(path, date_str):
    """Pine CSV parse → normalize sinyal listesi.
    Pine CSV'de tarih kolonu yok, dosya adındaki tarih kullanılır."""
    df = pd.read_csv(path)
    # Pine kolonları: Hisse, Sinyal, Devam, Rejim, Skor, Fiyat, Stop, TP, RR, RS
    sig_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
    signals = []
    for _, row in df.iterrows():
        signals.append({
            'screener': 'pine',
            'ticker': str(row['Hisse']).strip(),
            'signal_date': sig_date,
            'direction': 'AL',
            'signal_type': str(row.get('Sinyal', '')).strip(),
            'entry_price': float(row['Fiyat']),
            'quality': int(row['Skor']) if pd.notna(row.get('Skor')) else None,
        })
    return signals


def _parse_divergence(path):
    """Divergence CSV parse → normalize sinyal listesi."""
    df = pd.read_csv(path)
    signals = []
    for _, row in df.iterrows():
        d = str(row.get('direction', '')).strip().upper()
        if d == 'BUY':
            direction = 'AL'
        elif d == 'SELL':
            direction = 'SAT'
        else:
            continue
        signals.append({
            'screener': 'divergence',
            'ticker': str(row['ticker']).strip(),
            'signal_date': str(row['signal_date']).strip(),
            'direction': direction,
            'signal_type': str(row.get('div_type', '')).strip(),
            'entry_price': float(row['close']),
            'quality': int(row['quality']) if pd.notna(row.get('quality')) else None,
        })
    return signals


def parse_all_csvs(csv_map):
    """Tüm CSV'leri parse et → birleşik sinyal listesi."""
    all_signals = []
    for screener, (date_str, path) in csv_map.items():
        try:
            if screener in ('nox_v3_daily', 'nox_v3_weekly'):
                sigs = _parse_nox_v3(path, screener)
            elif screener == 'smc':
                sigs = _parse_smc(path)
            elif screener == 'pine':
                sigs = _parse_pine(path, date_str)
            elif screener == 'divergence':
                sigs = _parse_divergence(path)
            else:
                continue
            all_signals.extend(sigs)
            print(f"  {screener}: {len(sigs)} sinyal ({path})")
        except Exception as e:
            print(f"  ! {screener} parse hata: {e}")
    return all_signals


# =============================================================================
# 2. VERİ ÇEKME
# =============================================================================

def fetch_price_data(signals):
    """Sinyallerdeki unique ticker'lar için fiyat verisi çek."""
    tickers = sorted(set(s['ticker'] for s in signals))
    print(f"\n  {len(tickers)} unique hisse için veri çekiliyor...")
    all_data = data_mod.fetch_data(tickers, period="6mo")
    xu = data_mod.fetch_benchmark(period="6mo")
    return all_data, xu


# =============================================================================
# 3. FORWARD GETİRİ HESAPLAMA
# =============================================================================

WINDOWS = [1, 3, 5]


def compute_forward_returns(signals, all_data, xu_df):
    """Her sinyal için 1g, 3g, 5g forward getiri + XU100 kıyasla."""
    results = []
    for sig in signals:
        ticker = sig['ticker']
        df = all_data.get(ticker)
        if df is None or df.empty:
            continue

        # signal_date'i index'te bul
        try:
            sig_date = pd.Timestamp(sig['signal_date'])
        except Exception:
            continue

        idx = df.index.searchsorted(sig_date)
        # Tam eşleşme yoksa en yakın sonraki iş gününü bul
        if idx >= len(df):
            idx = len(df) - 1
        if abs((df.index[idx] - sig_date).days) > 5:
            continue  # çok uzak, atla

        entry_price = sig['entry_price']
        row = {**sig}
        status = 'bekliyor'

        for w in WINDOWS:
            fwd_idx = idx + w
            if fwd_idx < len(df):
                fwd_close = df['Close'].iloc[fwd_idx]
                ret = (fwd_close / entry_price - 1) * 100
                # SAT sinyalleri: düşüş = pozitif getiri
                if sig['direction'] == 'SAT':
                    ret = -ret
                row[f'ret_{w}d'] = round(ret, 2)

                # XU100 getirisi
                if xu_df is not None and not xu_df.empty:
                    xu_idx_start = xu_df.index.searchsorted(df.index[idx])
                    xu_idx_end = xu_df.index.searchsorted(df.index[fwd_idx])
                    if xu_idx_start < len(xu_df) and xu_idx_end < len(xu_df):
                        xu_start = xu_df['Close'].iloc[xu_idx_start]
                        xu_end = xu_df['Close'].iloc[xu_idx_end]
                        xu_ret = (xu_end / xu_start - 1) * 100
                        if sig['direction'] == 'SAT':
                            xu_ret = -xu_ret
                        row[f'xu_{w}d'] = round(xu_ret, 2)
                        row[f'excess_{w}d'] = round(ret - xu_ret, 2)
                    else:
                        row[f'xu_{w}d'] = None
                        row[f'excess_{w}d'] = None
                else:
                    row[f'xu_{w}d'] = None
                    row[f'excess_{w}d'] = None

                if w == 1:
                    status = 'kısmi'
            else:
                row[f'ret_{w}d'] = None
                row[f'xu_{w}d'] = None
                row[f'excess_{w}d'] = None

        # Tüm pencereler doluysa tamam
        if all(row.get(f'ret_{w}d') is not None for w in WINDOWS):
            status = 'tamam'
        row['status'] = status
        results.append(row)

    return results


# =============================================================================
# 4. ÖZET İSTATİSTİKLER
# =============================================================================

def compute_summary(results):
    """Screener bazlı özet istatistikler."""
    summary = {}
    # Genel
    all_screeners = sorted(set(r['screener'] for r in results))

    for scr in ['genel'] + all_screeners:
        if scr == 'genel':
            subset = results
        else:
            subset = [r for r in results if r['screener'] == scr]

        if not subset:
            continue

        stats = {'screener': scr, 'n': len(subset)}

        for w in WINDOWS:
            key = f'ret_{w}d'
            vals = [r[key] for r in subset if r.get(key) is not None]
            if vals:
                wins = sum(1 for v in vals if v > 0)
                stats[f'n_{w}d'] = len(vals)
                stats[f'wr_{w}d'] = round(wins / len(vals) * 100, 1)
                stats[f'avg_{w}d'] = round(sum(vals) / len(vals), 2)
                stats[f'med_{w}d'] = round(median(vals), 2)
                stats[f'best_{w}d'] = round(max(vals), 2)
                stats[f'worst_{w}d'] = round(min(vals), 2)
            else:
                stats[f'n_{w}d'] = 0
                stats[f'wr_{w}d'] = None
                stats[f'avg_{w}d'] = None
                stats[f'med_{w}d'] = None
                stats[f'best_{w}d'] = None
                stats[f'worst_{w}d'] = None

        # AL/SAT ayrımı
        for d in ['AL', 'SAT']:
            d_subset = [r for r in subset if r['direction'] == d]
            stats[f'n_{d}'] = len(d_subset)
            vals_5 = [r['ret_5d'] for r in d_subset if r.get('ret_5d') is not None]
            if vals_5:
                stats[f'wr_5d_{d}'] = round(sum(1 for v in vals_5 if v > 0) / len(vals_5) * 100, 1)
                stats[f'avg_5d_{d}'] = round(sum(vals_5) / len(vals_5), 2)
            else:
                stats[f'wr_5d_{d}'] = None
                stats[f'avg_5d_{d}'] = None

        summary[scr] = stats

    return summary


# =============================================================================
# 5. HTML RAPOR
# =============================================================================

_SCREENER_LABELS = {
    'genel': 'Genel',
    'nox_v3_daily': 'NOX v3 Günlük',
    'nox_v3_weekly': 'NOX v3 Haftalık',
    'smc': 'SMC',
    'pine': 'Pine',
    'divergence': 'Uyumsuzluk',
}

_SCREENER_TAB_ORDER = ['genel', 'nox_v3_daily', 'nox_v3_weekly', 'smc', 'pine', 'divergence']


def generate_html(results, summary, csv_map):
    """NOX dark-theme interaktif HTML rapor."""
    now = datetime.now().strftime('%d.%m.%Y %H:%M')
    dates = sorted(set(d for d, _ in csv_map.values()))
    date_label = ', '.join(dates)

    # Tab'ları oluştur (veri olan screener'lar)
    active_tabs = [t for t in _SCREENER_TAB_ORDER if t in summary]

    results_json = json.dumps(_sanitize(results), ensure_ascii=False)
    summary_json = json.dumps(_sanitize(summary), ensure_ascii=False)
    tabs_json = json.dumps(active_tabs)
    labels_json = json.dumps(_SCREENER_LABELS, ensure_ascii=False)

    # Genel özet kartları için
    gen = summary.get('genel', {})

    html = f"""<!DOCTYPE html>
<html lang="tr"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NOX — Forward Test · {date_label}</title>
<style>{_NOX_CSS}

/* TABS */
.nox-tabs {{ display: flex; gap: 4px; margin-bottom: 16px; flex-wrap: wrap; }}
.nox-tab {{
  padding: 10px 20px; border-radius: var(--radius) var(--radius) 0 0;
  background: var(--bg-card); border: 1px solid var(--border-subtle); border-bottom: none;
  color: var(--text-muted); font-weight: 600; font-size: 0.82rem;
  cursor: pointer; transition: all 0.2s; font-family: var(--font-display);
}}
.nox-tab:hover {{ color: var(--text-secondary); background: var(--bg-elevated); }}
.nox-tab.active {{
  color: var(--nox-cyan); background: var(--bg-elevated);
  border-color: var(--nox-cyan); border-bottom: 2px solid var(--bg-elevated);
}}
.nox-tab .cnt {{ font-family: var(--font-mono); font-size: 0.72rem; margin-left: 6px;
  padding: 2px 8px; border-radius: 10px; background: rgba(34,211,238,0.1); }}

/* SUMMARY CARDS */
.nox-cards {{ display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 20px; }}
.nox-card {{
  flex: 1; min-width: 140px; padding: 16px;
  background: var(--bg-card); border: 1px solid var(--border-subtle);
  border-radius: var(--radius); text-align: center;
}}
.nox-card .card-label {{
  font-size: 0.7rem; color: var(--text-muted); text-transform: uppercase;
  letter-spacing: 0.08em; margin-bottom: 6px; font-family: var(--font-display);
}}
.nox-card .card-value {{
  font-size: 1.4rem; font-weight: 700; font-family: var(--font-mono);
}}
.nox-card .card-sub {{
  font-size: 0.68rem; color: var(--text-muted); margin-top: 4px;
  font-family: var(--font-mono);
}}

/* STATS TABLE */
.stats-table {{ width: 100%; border-collapse: collapse; font-size: 0.75rem; margin-bottom: 20px; }}
.stats-table th {{
  background: var(--bg-elevated); color: var(--text-muted); font-weight: 600;
  font-size: 0.65rem; text-transform: uppercase; letter-spacing: 0.06em;
  padding: 8px; text-align: center; border-bottom: 1px solid var(--border-subtle);
}}
.stats-table td {{
  padding: 8px; text-align: center; border-bottom: 1px solid rgba(39,39,42,0.5);
  font-family: var(--font-mono); font-size: 0.72rem;
}}
.stats-table tr:hover {{ background: var(--bg-hover); }}

/* DIRECTION BADGES */
.dir-al {{ color: var(--nox-green); font-weight: 700; }}
.dir-sat {{ color: var(--nox-red); font-weight: 700; }}
.status-tamam {{ color: var(--nox-green); }}
.status-kismi {{ color: var(--nox-yellow); }}
.status-bekliyor {{ color: var(--text-muted); }}
</style>
</head><body>
<div class="nox-container">

<div class="nox-header">
  <div class="nox-logo">NOX<span class="proj">project</span><span class="mode">forward test</span></div>
  <div class="nox-meta">CSV: <b>{date_label}</b><br>{now}<br><b>{len(results)}</b> sinyal</div>
</div>

<!-- ÖZET KARTLARI -->
<div class="nox-cards" id="cards"></div>

<!-- TAB'LAR -->
<div class="nox-tabs" id="tabs"></div>

<!-- FİLTRE -->
<div class="nox-filters">
  <div><label>Hisse</label><input type="text" id="fS" placeholder="ARA" oninput="af()"></div>
  <div><label>Yön</label>
  <select id="fDir" onchange="af()"><option value="">Tümü</option>
  <option value="AL">AL</option><option value="SAT">SAT</option></select></div>
  <div><label>Durum</label>
  <select id="fSt" onchange="af()"><option value="">Tümü</option>
  <option value="tamam">Tamam</option><option value="kısmi">Kısmi</option>
  <option value="bekliyor">Bekliyor</option></select></div>
  <div><button class="nox-btn" onclick="resetF()">Sıfırla</button></div>
</div>

<!-- STATS TABLOSU -->
<div class="nox-table-wrap" style="margin-bottom:20px" id="statsWrap"></div>

<!-- DETAY TABLOSU -->
<div class="nox-table-wrap">
<table><thead><tr>
<th onclick="sb('ticker')">Hisse</th>
<th onclick="sb('screener')">Tarama</th>
<th onclick="sb('signal_type')">Sinyal</th>
<th onclick="sb('direction')">Yön</th>
<th onclick="sb('signal_date')">Tarih</th>
<th onclick="sb('entry_price')">Giriş</th>
<th onclick="sb('ret_1d')">1G%</th>
<th onclick="sb('ret_3d')">3G%</th>
<th onclick="sb('ret_5d')">5G%</th>
<th onclick="sb('xu_5d')">XU100 5G</th>
<th onclick="sb('excess_5d')">Fazla 5G</th>
<th onclick="sb('status')">Durum</th>
</tr></thead><tbody id="tb"></tbody></table>
</div>

<div class="nox-status" id="st"></div>
</div>

<script>
const D={results_json};
const S={summary_json};
const TABS={tabs_json};
const LBL={labels_json};
let curTab='genel';
let col='ret_5d', asc=false;

// ── TABS ──
function initTabs(){{
  const el=document.getElementById('tabs');
  TABS.forEach(t=>{{
    const d=document.createElement('div');
    d.className='nox-tab'+(t==='genel'?' active':'');
    d.id='tab-'+t;
    const n=t==='genel'?D.length:D.filter(r=>r.screener===t).length;
    d.innerHTML=(LBL[t]||t)+' <span class="cnt">'+n+'</span>';
    d.onclick=()=>{{curTab=t;
      document.querySelectorAll('.nox-tab').forEach(x=>x.classList.remove('active'));
      d.classList.add('active');
      renderStats();af()}};
    el.appendChild(d);
  }});
}}

// ── ÖZET KARTLARI ──
function renderCards(){{
  const el=document.getElementById('cards');
  const s=S[curTab];
  if(!s){{el.innerHTML='';return}}
  const mk=(label,key)=>{{
    const avg=s['avg_'+key];
    const wr=s['wr_'+key];
    const n=s['n_'+key]||0;
    const val=avg!=null?((avg>0?'+':'')+avg.toFixed(2)+'%'):'—';
    const cls=avg!=null?(avg>0?'var(--nox-green)':avg<0?'var(--nox-red)':'var(--text-muted)'):'var(--text-muted)';
    const wrStr=wr!=null?('WR: '+wr+'%'):'';
    return `<div class="nox-card"><div class="card-label">${{label}}</div>
      <div class="card-value" style="color:${{cls}}">${{val}}</div>
      <div class="card-sub">${{wrStr}} · ${{n}} sinyal</div></div>`;
  }};
  el.innerHTML=mk('1 Gün','1d')+mk('3 Gün','3d')+mk('5 Gün','5d');
}}

// ── STATS TABLOSU ──
function renderStats(){{
  renderCards();
  const el=document.getElementById('statsWrap');
  let h='<table class="stats-table"><thead><tr><th>Tarama</th><th>N</th>';
  h+='<th>1G Ort%</th><th>1G WR%</th><th>3G Ort%</th><th>3G WR%</th>';
  h+='<th>5G Ort%</th><th>5G WR%</th><th>5G Med%</th>';
  h+='<th>En İyi 5G</th><th>En Kötü 5G</th>';
  h+='<th>AL N</th><th>AL WR%</th><th>SAT N</th><th>SAT WR%</th>';
  h+='</tr></thead><tbody>';
  const tabs=curTab==='genel'?TABS:['genel',curTab];
  tabs.forEach(t=>{{
    const s=S[t];if(!s)return;
    const rc=(v)=>{{if(v==null)return'color:var(--text-muted)';return v>0?'color:var(--nox-green)':v<0?'color:var(--nox-red)':'color:var(--text-muted)'}};
    const fv=(v)=>v!=null?v.toFixed(2):'—';
    const fw=(v)=>v!=null?v.toFixed(1)+'%':'—';
    const bold=t===curTab?'font-weight:700':'';
    h+=`<tr style="${{bold}}">
      <td style="text-align:left;font-family:var(--font-display);font-weight:600">${{LBL[t]||t}}</td>
      <td>${{s.n}}</td>
      <td style="${{rc(s.avg_1d)}}">${{fv(s.avg_1d)}}</td><td>${{fw(s.wr_1d)}}</td>
      <td style="${{rc(s.avg_3d)}}">${{fv(s.avg_3d)}}</td><td>${{fw(s.wr_3d)}}</td>
      <td style="${{rc(s.avg_5d)}}">${{fv(s.avg_5d)}}</td><td>${{fw(s.wr_5d)}}</td>
      <td style="${{rc(s.med_5d)}}">${{fv(s.med_5d)}}</td>
      <td style="color:var(--nox-green)">${{fv(s.best_5d)}}</td>
      <td style="color:var(--nox-red)">${{fv(s.worst_5d)}}</td>
      <td>${{s.n_AL||0}}</td><td>${{fw(s.wr_5d_AL)}}</td>
      <td>${{s.n_SAT||0}}</td><td>${{fw(s.wr_5d_SAT)}}</td></tr>`;
  }});
  h+='</tbody></table>';
  el.innerHTML=h;
}}

// ── FİLTRE & SIRALAMA ──
function af(){{
  const sr=document.getElementById('fS').value.toUpperCase();
  const fd=document.getElementById('fDir').value;
  const fs=document.getElementById('fSt').value;
  let f=D.filter(r=>{{
    if(curTab!=='genel'&&r.screener!==curTab)return false;
    if(sr&&!r.ticker.includes(sr))return false;
    if(fd&&r.direction!==fd)return false;
    if(fs&&r.status!==fs)return false;
    return true;
  }});
  f.sort((a,b)=>{{
    let va=a[col],vb=b[col];
    if(va==null&&vb==null)return 0;
    if(va==null)return 1;if(vb==null)return -1;
    if(typeof va==='string')return asc?va.localeCompare(vb):vb.localeCompare(va);
    return asc?(va-vb):(vb-va);
  }});
  render(f);
}}

function sb(c){{if(col===c)asc=!asc;else{{col=c;asc=c==='ticker'||c==='signal_date'}};af()}}

function resetF(){{
  document.getElementById('fS').value='';
  document.getElementById('fDir').value='';
  document.getElementById('fSt').value='';
  af();
}}

// ── RENDER ──
function retCell(v){{
  if(v==null)return '<td style="color:var(--text-muted)">—</td>';
  const c=v>0?'var(--nox-green)':v<0?'var(--nox-red)':'var(--text-muted)';
  return '<td style="color:'+c+';font-weight:600">'+(v>0?'+':'')+v.toFixed(2)+'%</td>';
}}

function render(data){{
  const tb=document.getElementById('tb');tb.innerHTML='';
  data.forEach(r=>{{
    const tr=document.createElement('tr');
    const dirC=r.direction==='AL'?'dir-al':'dir-sat';
    const stC='status-'+(r.status==='tamam'?'tamam':r.status==='kısmi'?'kismi':'bekliyor');
    const stL=r.status==='tamam'?'Tamam':r.status==='kısmi'?'Kısmi':'Bekliyor';
    tr.innerHTML=`<td><a class="tv-link" href="https://www.tradingview.com/chart/?symbol=BIST:${{r.ticker}}" target="_blank">${{r.ticker}}</a></td>
      <td style="color:var(--text-muted);font-size:.68rem">${{LBL[r.screener]||r.screener}}</td>
      <td style="font-size:.68rem">${{r.signal_type}}</td>
      <td class="${{dirC}}">${{r.direction}}</td>
      <td style="color:var(--text-muted)">${{r.signal_date}}</td>
      <td>${{r.entry_price.toFixed(2)}}</td>
      ${{retCell(r.ret_1d)}}${{retCell(r.ret_3d)}}${{retCell(r.ret_5d)}}
      ${{retCell(r.xu_5d)}}${{retCell(r.excess_5d)}}
      <td class="${{stC}}" style="font-size:.7rem;font-weight:600">${{stL}}</td>`;
    tb.appendChild(tr);
  }});
  document.getElementById('st').innerHTML='<b>'+data.length+'</b> / '+D.length+' sinyal';
  renderStats();
}}

// ── INIT ──
initTabs();
renderStats();
af();
</script></body></html>"""
    return html


# =============================================================================
# 6. CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="NOX Forward Test Aracı")
    parser.add_argument('--output', default='output', help='CSV/HTML dizini (default: output)')
    parser.add_argument('--date', help='Belirli tarih (YYYYMMDD)')
    parser.add_argument('--open', action='store_true', help='Raporu tarayıcıda aç')
    args = parser.parse_args()

    output_dir = args.output
    if not os.path.isdir(output_dir):
        print(f"  HATA: {output_dir} dizini bulunamadı!")
        sys.exit(1)

    # ── 1. CSV Keşfet ──
    print(f"\n  CSV dosyaları taranıyor ({output_dir})...")
    csv_map = discover_csvs(output_dir, args.date)
    if not csv_map:
        print("  HATA: Hiçbir CSV bulunamadı!")
        sys.exit(1)
    for scr, (d, p) in csv_map.items():
        print(f"  {scr}: {d} → {os.path.basename(p)}")

    # ── 2. Parse ──
    print(f"\n  CSV'ler parse ediliyor...")
    signals = parse_all_csvs(csv_map)
    if not signals:
        print("  HATA: Hiçbir sinyal bulunamadı!")
        sys.exit(1)
    print(f"  Toplam: {len(signals)} sinyal")

    # ── 3. Veri Çek ──
    t0 = time.time()
    all_data, xu_df = fetch_price_data(signals)
    print(f"  Veri çekildi ({time.time() - t0:.1f}s)")

    # ── 4. Forward Getiri ──
    print(f"\n  Forward getiriler hesaplanıyor...")
    results = compute_forward_returns(signals, all_data, xu_df)
    n_tamam = sum(1 for r in results if r['status'] == 'tamam')
    n_kismi = sum(1 for r in results if r['status'] == 'kısmi')
    n_bekl = sum(1 for r in results if r['status'] == 'bekliyor')
    print(f"  {len(results)} sinyal: {n_tamam} tamam, {n_kismi} kısmi, {n_bekl} bekliyor")

    # ── 5. Özet ──
    summary = compute_summary(results)
    gen = summary.get('genel', {})
    print(f"\n  === GENEL ÖZET ===")
    for w in WINDOWS:
        avg = gen.get(f'avg_{w}d')
        wr = gen.get(f'wr_{w}d')
        if avg is not None:
            print(f"  {w}G: ort {avg:+.2f}%, WR {wr:.1f}%")

    # ── 6. HTML ──
    html = generate_html(results, summary, csv_map)
    dates = sorted(set(d for d, _ in csv_map.values()))
    date_str = dates[0] if dates else datetime.now().strftime('%Y%m%d')
    os.makedirs(output_dir, exist_ok=True)
    fname = f"forward_test_{date_str}.html"
    path = os.path.join(output_dir, fname)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"\n  HTML: {path}")

    if args.open:
        subprocess.Popen(['open', path])

    print(f"  Forward test tamamlandı.\n")


if __name__ == '__main__':
    main()
