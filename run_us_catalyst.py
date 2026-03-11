#!/usr/bin/env python3
"""
NOX Project — US Catalyst Screener Runner
ABD borsasında %30-60 potansiyelli hisseleri önceden tespit eder.

Kullanım:
  python run_us_catalyst.py                    # Konsol çıktı
  python run_us_catalyst.py --html             # HTML rapor
  python run_us_catalyst.py --html --notify    # HTML + Telegram
  python run_us_catalyst.py --debug AAPL       # Tek ticker debug
  python run_us_catalyst.py --no-enrich        # Sadece Faz 1 (hızlı)
  python run_us_catalyst.py --csv              # CSV çıktı
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd

from markets.us.data import (
    get_all_us_tickers, fetch_data, fetch_benchmark,
    fetch_ticker_info, fetch_insider_data, fetch_fda_calendar,
)
from markets.us.catalyst_screener import run_all_modules, compute_spy_regime
from markets.bist.regime_transition import (
    scan_regime_transition, compute_trade_state, calc_oe_score,
    find_last_transition, REGIME_NAMES,
)
from markets.us.config import RT_CFG_US
from core.reports import (
    _NOX_CSS, _sanitize,
    send_telegram, send_telegram_document, push_html_to_github,
)


# ══════════════════════════════════════════
# REGIME TRANSITION HELPERS
# ══════════════════════════════════════════

def _to_lower_cols(df):
    """Uppercase → lowercase kolon dönüşümü (regime_transition lowercase istiyor)."""
    rename = {}
    for c in df.columns:
        cl = c.lower()
        if cl in ('close', 'high', 'low', 'open', 'volume') and c != cl:
            rename[c] = cl
    return df.rename(columns=rename) if rename else df


def _to_weekly_us(df):
    """Günlük → haftalık resample (Cuma kapanış)."""
    if len(df) < 10:
        return None
    w = df.resample('W-FRI').agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'volume': 'sum',
    }).dropna(subset=['close'])
    return w if len(w) >= 10 else None


def _entry_window_us(days_since):
    """Giriş penceresi etiketleme."""
    if days_since <= 3:
        return 'FRESH'
    elif days_since <= 10:
        return 'RECENT'
    elif days_since <= 25:
        return '2ND_WAVE'
    else:
        return 'LATE'


def scan_all_regimes(stock_dfs, cfg=None):
    """Tüm stoklarda regime transition taraması."""
    cfg = cfg or RT_CFG_US
    regime_data = {}
    n_scanned = 0

    for ticker, df_raw in stock_dfs.items():
        if len(df_raw) < 60:
            continue
        n_scanned += 1

        try:
            df = _to_lower_cols(df_raw.copy())
            weekly_df = _to_weekly_us(df)

            result = scan_regime_transition(df, weekly_df, cfg)

            # Trade state
            trade_state = compute_trade_state(
                result['regime'], result['close'], result['ema21']
            )

            # OE score
            oe = calc_oe_score(df, result['ema21'], cfg)

            # Son bar değerleri
            last = len(df) - 1
            regime_val = int(result['regime'].iloc[last])
            exit_val = int(result['exit_stage'].iloc[last])
            in_trade = bool(trade_state['in_trade'].iloc[last])

            # Geçiş bilgisi
            trans_info = find_last_transition(
                result['regime'], result['close'], df.index
            )
            if trans_info and trans_info['direction'] == 'AL':
                days_since = last - trans_info['bar_idx']
                entry_close = trans_info['close_at_transition']
                gain_pct = (float(df['close'].iloc[last]) - entry_close) / entry_close * 100 if entry_close > 0 else 0.0
                transition_label = trans_info['transition']
                trade_date = str(trans_info['date'])[:10]
            else:
                days_since = 0
                gain_pct = 0.0
                transition_label = trans_info['transition'] if trans_info else 'TUT'
                trade_date = ''

            entry_window = _entry_window_us(days_since) if in_trade else '-'

            regime_data[ticker] = {
                'regime': regime_val,
                'regime_name': REGIME_NAMES.get(regime_val, '?'),
                'exit_stage': exit_val,
                'in_trade': in_trade,
                'entry_window': entry_window,
                'days_since': days_since,
                'trade_date': trade_date,
                'transition': transition_label,
                'gain_since_pct': round(gain_pct, 2),
                'oe_score': oe['oe_score'],
                'oe_tags': oe['oe_tags'],
                'oe_warning': oe['oe_warning'],
                'trend_score': int(result['trend_score'].iloc[last]),
                'participation_score': int(result['participation_score'].iloc[last]),
                'expansion_score': int(result['expansion_score'].iloc[last]),
                'close': float(df['close'].iloc[last]),
                'adx': float(result['adx'].iloc[last]) if pd.notna(result['adx'].iloc[last]) else 0.0,
                'cmf': float(result['cmf'].iloc[last]),
                'rvol': float(result['rvol'].iloc[last]),
            }
        except Exception as e:
            # Sessiz devam — tek hisse hatası tüm taramayı durdurmasın
            continue

    return regime_data, n_scanned


def classify_groups(all_results, regime_data):
    """Catalyst sinyallerini regime verisiyle cross-reference ederek 4 gruba ayır.

    A: Catalyst + regime active + FRESH/RECENT + exit≤1 + oe≤1
    B: Catalyst + regime active (diğer)
    C: Catalyst var ama regime aktif değil
    D: Regime active ama catalyst yok
    """
    grouped = {'A': [], 'B': [], 'C': [], 'D': []}

    # Catalyst sinyali olan ticker'lar (set)
    catalyst_tickers = set()
    for mod_key, signals in all_results.items():
        for s in signals:
            catalyst_tickers.add(s['ticker'])

    # Her catalyst sinyalini grupla
    for mod_key, signals in all_results.items():
        for s in signals:
            ticker = s['ticker']
            ri = regime_data.get(ticker)

            if ri is None:
                # Regime verisi yok → C
                s_copy = {**s, 'module': mod_key, 'group': 'C'}
                grouped['C'].append(s_copy)
                continue

            regime_active = ri['in_trade'] and ri['regime'] >= 2

            if regime_active:
                ew = ri['entry_window']
                if ew in ('FRESH', 'RECENT') and ri['exit_stage'] <= 1 and ri['oe_score'] <= 1:
                    group = 'A'
                else:
                    group = 'B'
            else:
                group = 'C'

            s_copy = {**s, 'module': mod_key, 'group': group}
            # Regime bilgilerini ekle
            s_copy.update({
                'regime_name': ri['regime_name'],
                'entry_window': ri['entry_window'],
                'exit_stage': ri['exit_stage'],
                'oe_score': ri['oe_score'],
                'oe_tags': ri['oe_tags'],
                'days_since': ri['days_since'],
                'trade_date': ri['trade_date'],
                'gain_since_pct': ri['gain_since_pct'],
                'regime': ri['regime'],
            })
            grouped[group].append(s_copy)

    # D grubu: regime active ama catalyst yok
    for ticker, ri in regime_data.items():
        if ticker in catalyst_tickers:
            continue
        if ri['in_trade'] and ri['regime'] >= 2:
            grouped['D'].append({
                'ticker': ticker,
                'group': 'D',
                'regime_name': ri['regime_name'],
                'regime': ri['regime'],
                'transition': ri['transition'],
                'entry_window': ri['entry_window'],
                'exit_stage': ri['exit_stage'],
                'oe_score': ri['oe_score'],
                'oe_tags': ri['oe_tags'],
                'days_since': ri['days_since'],
                'trade_date': ri['trade_date'],
                'gain_since_pct': ri['gain_since_pct'],
                'close': ri['close'],
                'adx': ri['adx'],
                'cmf': ri['cmf'],
                'rvol': ri['rvol'],
                'trend_score': ri['trend_score'],
                'participation_score': ri['participation_score'],
                'expansion_score': ri['expansion_score'],
            })

    # Sıralama
    for g in ('A', 'B', 'C'):
        grouped[g].sort(key=lambda x: x.get('score', 0), reverse=True)
    # D: regime desc → exit_stage asc
    grouped['D'].sort(key=lambda x: (-x.get('regime', 0), x.get('exit_stage', 0)))

    return grouped


# ══════════════════════════════════════════
# KONSOL ÇIKTI
# ══════════════════════════════════════════

MODULE_LABELS = {
    'VOLUME': '📈 Unusual Volume (reaktif)',
    'ACCUM': '🔍 Accumulation (prediktif)',
    'SQUEEZE': '🔴 Short Squeeze',
    'INSIDER': '💼 Insider Buying',
    'BIOTECH': '🧬 Biotech Catalyst',
    'EARNINGS': '📊 Earnings Momentum',
    'BREAKOUT': '🔺 Technical Breakout',
}

MODULE_EMOJI = {
    'VOLUME': '📈', 'ACCUM': '🔍', 'SQUEEZE': '🔴', 'INSIDER': '💼',
    'BIOTECH': '🧬', 'EARNINGS': '📊', 'BREAKOUT': '🔺',
}


def _print_results(all_results, n_scanned):
    """Konsol çıktı."""
    total = sum(len(v) for v in all_results.values())
    print(f"\n{'═' * 60}")
    print(f"  NOX US CATALYST — {n_scanned} taranan | {total} sinyal")
    print(f"{'═' * 60}")

    for mod_key, label in MODULE_LABELS.items():
        signals = all_results.get(mod_key, [])
        if not signals:
            continue
        print(f"\n{label} ({len(signals)})")
        print("─" * 50)

        for s in signals[:15]:  # Top 15 per module
            line = f"  {s['ticker']:<6} {s['close']:>8.2f}"

            if mod_key == 'VOLUME':
                line += f"  {s['change_pct']:>+6.1f}%  RVOL:{s['rvol']:.1f}  ${s['dollar_vol'] / 1e6:.0f}M"
                if s['consecutive_days'] > 1:
                    line += f"  [{s['consecutive_days']}g]"
            elif mod_key == 'ACCUM':
                line += f"  AvgRVOL:{s['avg_rvol']:.2f}  Range/ATR:{s['range_vs_atr']:.2f}  Slope:{s['vol_slope']:.2f}  {s['accum_days']}g"
            elif mod_key == 'SQUEEZE':
                line += f"  Short:{s['short_pct']:.0f}%  DTC:{s.get('short_ratio', 0) or 0:.1f}"
                if s.get('float_m'):
                    line += f"  Float:{s['float_m']:.0f}M"
            elif mod_key == 'INSIDER':
                line += f"  {s['n_buyers']} alıcı  ${s['buy_value_k']:.0f}K"
                if s['has_senior']:
                    line += "  [C-Level]"
            elif mod_key == 'BIOTECH':
                line += f"  MCap:${s.get('mcap_b', 0) or 0:.1f}B  VolTrend:{s['vol_trend']:.1f}"
                if s.get('days_to_earnings'):
                    line += f"  Earn:{s['days_to_earnings']}g"
            elif mod_key == 'EARNINGS':
                sub = s.get('subtype', '')
                line += f"  [{sub}]"
                if sub == 'PRE':
                    line += f"  {s['days_to_earnings']}g  BB:{s.get('bb_width', 0):.1f}%"
                elif sub == 'POST':
                    line += f"  Gap:{s.get('gap_pct', 0):.1f}%  Hold:{s.get('gap_hold_pct', 0):.1f}%"
            elif mod_key == 'BREAKOUT':
                sub = s.get('subtype', '')
                line += f"  [{sub}]  ATR:{s['atr_ratio']:.2f}  {s['consol_days']}g"
                if s.get('vol_ratio'):
                    line += f"  Vol:{s['vol_ratio']:.1f}x"

            line += f"  Skor:{s['score']}"
            print(line)


# ══════════════════════════════════════════
# CSV ÇIKTI
# ══════════════════════════════════════════

def _save_csv(all_results, date_str, grouped=None):
    """CSV çıktı — grouped varsa group+regime bilgileri eklenir."""
    rows = []

    if grouped:
        for group_key, signals in grouped.items():
            for s in signals:
                row = {'date': date_str, 'group': group_key}
                # oe_tags list → string
                s_copy = {**s}
                if 'oe_tags' in s_copy and isinstance(s_copy['oe_tags'], list):
                    s_copy['oe_tags'] = ','.join(s_copy['oe_tags'])
                row.update(s_copy)
                rows.append(row)
    else:
        for mod_key, signals in all_results.items():
            for s in signals:
                row = {'module': mod_key, 'date': date_str}
                row.update(s)
                rows.append(row)

    if not rows:
        print("⚠️ CSV: Sinyal yok")
        return None

    df = pd.DataFrame(rows)
    os.makedirs('output', exist_ok=True)
    path = f"output/us_catalyst_{date_str.replace('-', '')}.csv"
    df.to_csv(path, index=False)
    print(f"💾 CSV: {path} ({len(rows)} satır)")
    return path


# ══════════════════════════════════════════
# HTML RAPOR
# ══════════════════════════════════════════

def _generate_html(all_results, n_scanned, date_str, regime_info=None,
                   grouped=None):
    """Interaktif HTML rapor — 2 katmanlı tab: Grup (Özet/A/B/C/D) + Modül."""

    now_str = datetime.now(timezone(timedelta(hours=-5))).strftime('%d.%m.%Y %H:%M ET')
    total = sum(len(v) for v in all_results.values())

    # Regime badge
    ri = regime_info or {'regime': 'NEUTRAL', 'score': 3, 'max_score': 6}
    regime_label = ri['regime']
    regime_s = ri['score']
    regime_m = ri['max_score']
    regime_color_map = {'BULL': '#4ade80', 'NEUTRAL': '#facc15', 'RISK_OFF': '#f87171'}
    regime_emoji_map = {'BULL': '\U0001f7e2', 'NEUTRAL': '\U0001f7e1', 'RISK_OFF': '\U0001f534'}
    regime_color = regime_color_map.get(regime_label, '#facc15')
    regime_em = regime_emoji_map.get(regime_label, '')

    # Her modülün verisini JSON'a çevir
    data_json = json.dumps(_sanitize(all_results), ensure_ascii=False)

    # Grouped veriyi JSON'a çevir
    grp = grouped or {'A': [], 'B': [], 'C': [], 'D': []}
    grouped_json = json.dumps(_sanitize(grp), ensure_ascii=False)

    html = f"""<!DOCTYPE html>
<html lang="tr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NOX US Catalyst — {now_str}</title>
<style>
{_NOX_CSS}

/* TABS — Level 1 (Group) */
.nox-tabs {{
  display: flex; gap: 4px; flex-wrap: wrap;
  margin-bottom: 0;
  border-bottom: 1px solid var(--border-subtle);
  padding-bottom: 0;
}}
.nox-tab {{
  padding: 8px 16px;
  font-size: 0.78rem; font-weight: 600;
  color: var(--text-muted);
  cursor: pointer;
  border-bottom: 2px solid transparent;
  transition: all 0.2s;
  user-select: none;
}}
.nox-tab:hover {{ color: var(--text-secondary); }}
.nox-tab.active {{
  color: var(--nox-cyan);
  border-bottom-color: var(--nox-cyan);
}}
.nox-tab .cnt {{
  font-family: var(--font-mono); font-size: 0.72rem;
  margin-left: 4px; opacity: 0.7;
}}
.tab-content {{ display: none; }}
.tab-content.active {{ display: block; }}

/* TABS — Level 2 (Module sub-tabs) */
.nox-subtabs {{
  display: flex; gap: 3px; flex-wrap: wrap;
  margin: 12px 0 16px;
  border-bottom: 1px solid rgba(255,255,255,0.05);
  padding-bottom: 0;
}}
.nox-subtab {{
  padding: 6px 12px;
  font-size: 0.72rem; font-weight: 600;
  color: var(--text-muted);
  cursor: pointer;
  border-bottom: 2px solid transparent;
  transition: all 0.2s;
  user-select: none;
}}
.nox-subtab:hover {{ color: var(--text-secondary); }}
.nox-subtab.active {{
  color: var(--nox-cyan);
  border-bottom-color: var(--nox-cyan);
}}
.nox-subtab .cnt {{
  font-family: var(--font-mono); font-size: 0.68rem;
  margin-left: 3px; opacity: 0.7;
}}
.subtab-content {{ display: none; }}
.subtab-content.active {{ display: block; }}

/* SCORE BADGE */
.score-badge {{
  display: inline-block; padding: 2px 8px; border-radius: 10px;
  font-weight: 700; font-size: 0.7rem; font-family: var(--font-mono);
}}
.score-high {{ background: rgba(74,222,128,0.15); color: var(--nox-green); }}
.score-mid {{ background: rgba(250,204,21,0.15); color: var(--nox-yellow); }}
.score-low {{ background: rgba(251,146,60,0.15); color: var(--nox-orange); }}

/* MODULE BADGE */
.mod-badge {{
  display: inline-block; padding: 3px 10px; border-radius: 12px;
  font-weight: 700; font-size: 0.65rem;
  letter-spacing: 0.02em;
}}
.mod-ACCUM {{ background: rgba(251,146,60,0.15); color: var(--nox-orange); }}
.mod-VOLUME {{ background: rgba(96,165,250,0.15); color: var(--nox-blue); }}
.mod-SQUEEZE {{ background: rgba(248,113,113,0.15); color: var(--nox-red); }}
.mod-INSIDER {{ background: rgba(192,132,252,0.15); color: var(--nox-purple); }}
.mod-BIOTECH {{ background: rgba(74,222,128,0.15); color: var(--nox-green); }}
.mod-EARNINGS {{ background: rgba(250,204,21,0.15); color: var(--nox-yellow); }}
.mod-BREAKOUT {{ background: rgba(34,211,238,0.15); color: var(--nox-cyan); }}

/* GROUP BADGE */
.grp-badge {{
  display: inline-block; padding: 2px 8px; border-radius: 10px;
  font-weight: 700; font-size: 0.7rem; font-family: var(--font-mono);
}}
.grp-A {{ background: rgba(74,222,128,0.2); color: #4ade80; }}
.grp-B {{ background: rgba(250,204,21,0.2); color: #facc15; }}
.grp-C {{ background: rgba(96,165,250,0.2); color: #60a5fa; }}
.grp-D {{ background: rgba(156,163,175,0.2); color: #9ca3af; }}

/* OE BADGE */
.oe-0 {{ color: var(--nox-green); }}
.oe-1, .oe-2 {{ color: var(--nox-yellow); }}
.oe-3 {{ color: var(--nox-orange); }}
.oe-4 {{ color: var(--nox-red); }}

/* ENTRY WINDOW BADGE */
.ew-badge {{
  display: inline-block; padding: 2px 6px; border-radius: 8px;
  font-weight: 700; font-size: 0.65rem;
}}
.ew-FRESH {{ background: rgba(74,222,128,0.2); color: #4ade80; }}
.ew-RECENT {{ background: rgba(250,204,21,0.2); color: #facc15; }}
.ew-2ND_WAVE {{ background: rgba(251,146,60,0.2); color: #fb923c; }}
.ew-LATE {{ background: rgba(248,113,113,0.2); color: #f87171; }}

/* EXIT STAGE */
.exit-0 {{ color: var(--nox-green); }}
.exit-1 {{ color: var(--nox-yellow); }}
.exit-2 {{ color: var(--nox-orange); }}
.exit-3 {{ color: var(--nox-red); }}

/* REGIME NAME BADGE */
.rn-badge {{
  display: inline-block; padding: 2px 7px; border-radius: 8px;
  font-weight: 700; font-size: 0.65rem;
}}
.rn-FULL_TREND {{ background: rgba(74,222,128,0.2); color: #4ade80; }}
.rn-TREND {{ background: rgba(250,204,21,0.2); color: #facc15; }}
.rn-GRI_BOLGE {{ background: rgba(251,146,60,0.2); color: #fb923c; }}
.rn-CHOPPY {{ background: rgba(156,163,175,0.2); color: #9ca3af; }}

/* SUMMARY CARDS */
.summary-cards {{
  display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
  gap: 12px; margin-bottom: 20px;
}}
.summary-card {{
  background: var(--bg-card); border: 1px solid var(--border-subtle);
  border-radius: var(--radius); padding: 16px;
  text-align: center;
}}
.summary-card .card-num {{
  font-size: 1.8rem; font-weight: 800;
  font-family: var(--font-mono);
}}
.summary-card .card-label {{
  font-size: 0.72rem; color: var(--text-muted);
  margin-top: 4px; text-transform: uppercase;
  letter-spacing: 0.06em;
}}

/* DIRECTION BADGE */
.dir-up {{ color: var(--nox-green); }}
.dir-down {{ color: var(--nox-red); }}

/* REGIME BADGE */
.regime-badge {{
  display: inline-block; padding: 4px 14px; border-radius: 14px;
  font-weight: 700; font-size: 0.78rem; letter-spacing: 0.04em;
  margin-left: 12px;
}}

/* LIQUIDITY BADGE */
.liq-high {{ background: rgba(74,222,128,0.15); color: var(--nox-green); }}
.liq-mid {{ background: rgba(250,204,21,0.15); color: var(--nox-yellow); }}
.liq-low {{ background: rgba(248,113,113,0.15); color: var(--nox-red); }}

/* RISK % */
.risk-low {{ color: var(--nox-green); }}
.risk-mid {{ color: var(--nox-yellow); }}
.risk-high {{ color: var(--nox-red); }}
</style>
</head>
<body>
<div class="nox-container">

<!-- HEADER -->
<div class="nox-header">
  <div class="nox-logo">
    NOX<span class="proj"> Project</span>
    <span class="mode">US Catalyst Screener</span>
  </div>
  <div class="nox-meta">
    <div><b>{n_scanned}</b> taranan &middot; <b>{total}</b> sinyal
      <span class="regime-badge" style="background:rgba({{'74,222,128' if regime_label=='BULL' else '250,204,21' if regime_label=='NEUTRAL' else '248,113,113'}},0.15);color:{regime_color}">
        {regime_em} SPY: {regime_label} ({regime_s}/{regime_m})
      </span>
    </div>
    <div>{now_str}</div>
  </div>
</div>

<!-- GROUP TABS (Level 1) -->
<div class="nox-tabs" id="grp-tabs"></div>

<!-- TAB CONTENTS -->
<div id="grp-contents"></div>

<!-- STATUS -->
<div class="nox-status">
  NOX US Catalyst Screener &middot; <b>{n_scanned}</b> hisse tarandi &middot; {now_str}
</div>

</div><!-- /container -->

<script>
const D = {data_json};
const G = {grouped_json};
const N_SCANNED = {n_scanned};

const GRP_TABS = [
  {{id:'summary', label:'Ozet', emoji:'⭐'}},
  {{id:'A', label:'A Grubu', emoji:'🟢'}},
  {{id:'B', label:'B Grubu', emoji:'🟡'}},
  {{id:'C', label:'Watchlist', emoji:'🔵'}},
  {{id:'D', label:'Teknik', emoji:'⚪'}},
  // Modül tabları
  {{id:'ACCUM', label:'Birikim', emoji:'🔍'}},
  {{id:'BREAKOUT', label:'Breakout', emoji:'🔺'}},
  {{id:'SQUEEZE', label:'Squeeze', emoji:'🔴'}},
  {{id:'BIOTECH', label:'Biotech', emoji:'🧬'}},
  {{id:'EARNINGS', label:'Earnings', emoji:'📊'}},
  {{id:'INSIDER', label:'Insider', emoji:'💼'}},
  {{id:'VOLUME', label:'Volume', emoji:'📈'}},
];

// Regime kolonları (her modül tablosuna eklenecek)
const RT_COLS = [
  {{k:'group',l:'Grp'}}, {{k:'regime_name',l:'Regime'}}, {{k:'entry_window',l:'Pencere'}},
  {{k:'exit_stage',l:'Exit'}}, {{k:'oe_score',l:'OE'}}, {{k:'trade_date',l:'Baslangic'}},
];

const MOD_COLS = {{
  VOLUME: [
    {{k:'ticker',l:'Ticker'}}, {{k:'close',l:'Fiyat'}}, ...RT_COLS,
    {{k:'change_pct',l:'Degisim%'}},
    {{k:'rvol',l:'RVOL'}}, {{k:'dollar_vol',l:'Likidite'}}, {{k:'direction',l:'Yon'}},
    {{k:'consecutive_days',l:'Ardisik'}}, {{k:'trigger',l:'Trigger'}}, {{k:'stop',l:'Stop'}},
    {{k:'risk_pct',l:'Risk%'}}, {{k:'gap_risk',l:'Gap Risk'}},
    {{k:'rs',l:'RS'}}, {{k:'score',l:'Skor'}},
  ],
  ACCUM: [
    {{k:'ticker',l:'Ticker'}}, {{k:'close',l:'Fiyat'}}, ...RT_COLS,
    {{k:'avg_rvol',l:'Ort RVOL'}},
    {{k:'range_vs_atr',l:'Range/ATR'}}, {{k:'vol_slope',l:'Hacim Egimi'}},
    {{k:'accum_days',l:'Birikim Gun'}}, {{k:'range_pct',l:'Range%'}},
    {{k:'dollar_vol',l:'Likidite'}}, {{k:'trigger',l:'Trigger'}}, {{k:'stop',l:'Stop'}},
    {{k:'risk_pct',l:'Risk%'}}, {{k:'gap_risk',l:'Gap Risk'}},
    {{k:'rs',l:'RS'}}, {{k:'score',l:'Skor'}},
  ],
  SQUEEZE: [
    {{k:'ticker',l:'Ticker'}}, {{k:'name',l:'Ad'}}, {{k:'close',l:'Fiyat'}}, ...RT_COLS,
    {{k:'short_pct',l:'Short%'}}, {{k:'short_ratio',l:'DTC'}}, {{k:'float_m',l:'Float(M)'}},
    {{k:'mcap_b',l:'MCap($B)'}}, {{k:'above_ema21',l:'EMA21+'}}, {{k:'vol_trend',l:'VolTrend'}},
    {{k:'dollar_vol',l:'Likidite'}}, {{k:'trigger',l:'Trigger'}}, {{k:'stop',l:'Stop'}},
    {{k:'risk_pct',l:'Risk%'}}, {{k:'gap_risk',l:'Gap Risk'}},
    {{k:'sector',l:'Sektor'}}, {{k:'score',l:'Skor'}},
  ],
  INSIDER: [
    {{k:'ticker',l:'Ticker'}}, {{k:'name',l:'Ad'}}, {{k:'close',l:'Fiyat'}}, ...RT_COLS,
    {{k:'n_buyers',l:'Alicilar'}}, {{k:'buy_value_k',l:'Alim($K)'}},
    {{k:'n_sells',l:'Satislar'}}, {{k:'days_ago',l:'GunOnce'}},
    {{k:'has_senior',l:'C-Level'}}, {{k:'dollar_vol',l:'Likidite'}},
    {{k:'trigger',l:'Trigger'}}, {{k:'stop',l:'Stop'}}, {{k:'risk_pct',l:'Risk%'}},
    {{k:'gap_risk',l:'Gap Risk'}}, {{k:'sector',l:'Sektor'}}, {{k:'score',l:'Skor'}},
  ],
  BIOTECH: [
    {{k:'ticker',l:'Ticker'}}, {{k:'name',l:'Ad'}}, {{k:'close',l:'Fiyat'}}, ...RT_COLS,
    {{k:'mcap_b',l:'MCap($B)'}}, {{k:'vol_trend',l:'VolTrend'}},
    {{k:'vol_accel',l:'Ivme'}}, {{k:'atr_expansion',l:'ATR Exp'}},
    {{k:'fda_date',l:'FDA Tarih'}}, {{k:'fda_drug',l:'Ilac'}}, {{k:'days_to_fda',l:'FDA Gun'}},
    {{k:'days_to_earnings',l:'Bilanco'}}, {{k:'dollar_vol',l:'Likidite'}},
    {{k:'trigger',l:'Trigger'}}, {{k:'stop',l:'Stop'}}, {{k:'risk_pct',l:'Risk%'}},
    {{k:'gap_risk',l:'Gap Risk'}}, {{k:'industry',l:'Industry'}},
    {{k:'rs',l:'RS'}}, {{k:'score',l:'Skor'}},
  ],
  EARNINGS: [
    {{k:'ticker',l:'Ticker'}}, {{k:'name',l:'Ad'}}, {{k:'close',l:'Fiyat'}}, ...RT_COLS,
    {{k:'subtype',l:'Tip'}}, {{k:'days_to_earnings',l:'Gun'}},
    {{k:'bb_width',l:'BB%'}}, {{k:'gap_pct',l:'Gap%'}},
    {{k:'vol_trend',l:'VolTrend'}}, {{k:'earnings_date',l:'Tarih'}},
    {{k:'dollar_vol',l:'Likidite'}}, {{k:'trigger',l:'Trigger'}}, {{k:'stop',l:'Stop'}},
    {{k:'risk_pct',l:'Risk%'}}, {{k:'gap_risk',l:'Gap Risk'}},
    {{k:'sector',l:'Sektor'}}, {{k:'rs',l:'RS'}}, {{k:'score',l:'Skor'}},
  ],
  BREAKOUT: [
    {{k:'ticker',l:'Ticker'}}, {{k:'close',l:'Fiyat'}}, ...RT_COLS,
    {{k:'subtype',l:'Tip'}},
    {{k:'atr_ratio',l:'ATR Oran'}}, {{k:'consol_days',l:'Konsol(g)'}},
    {{k:'range_pct',l:'Range%'}}, {{k:'vol_ratio',l:'VolRatio'}},
    {{k:'prev_20_high',l:'20gHigh'}}, {{k:'dollar_vol',l:'Likidite'}},
    {{k:'trigger',l:'Trigger'}}, {{k:'stop',l:'Stop'}}, {{k:'risk_pct',l:'Risk%'}},
    {{k:'gap_risk',l:'Gap Risk'}}, {{k:'rs',l:'RS'}}, {{k:'score',l:'Skor'}},
  ],
}};

// A/B group columns
const AB_COLS = [
  {{k:'module',l:'Modul'}}, {{k:'ticker',l:'Ticker'}}, {{k:'close',l:'Fiyat'}},
  {{k:'regime_name',l:'Regime'}}, {{k:'entry_window',l:'Pencere'}},
  {{k:'exit_stage',l:'Exit'}}, {{k:'oe_score',l:'OE'}},
  {{k:'trade_date',l:'Baslangic'}}, {{k:'days_since',l:'Gun'}},
  {{k:'gain_since_pct',l:'Getiri%'}},
  {{k:'trigger',l:'Trigger'}}, {{k:'stop',l:'Stop'}}, {{k:'risk_pct',l:'Risk%'}},
  {{k:'dollar_vol',l:'Likidite'}}, {{k:'score',l:'Skor'}},
];

// C group columns
const C_COLS = [
  {{k:'module',l:'Modul'}}, {{k:'ticker',l:'Ticker'}}, {{k:'close',l:'Fiyat'}},
  {{k:'regime_name',l:'Regime'}}, {{k:'trend_score',l:'T'}},
  {{k:'participation_score',l:'P'}}, {{k:'expansion_score',l:'E'}},
  {{k:'trigger',l:'Trigger'}}, {{k:'stop',l:'Stop'}}, {{k:'risk_pct',l:'Risk%'}},
  {{k:'dollar_vol',l:'Likidite'}}, {{k:'score',l:'Skor'}},
];

// D group columns
const D_COLS = [
  {{k:'ticker',l:'Ticker'}}, {{k:'close',l:'Fiyat'}},
  {{k:'regime_name',l:'Regime'}}, {{k:'transition',l:'Gecis'}},
  {{k:'entry_window',l:'Pencere'}}, {{k:'exit_stage',l:'Exit'}},
  {{k:'oe_score',l:'OE'}}, {{k:'trade_date',l:'Baslangic'}},
  {{k:'days_since',l:'Gun'}}, {{k:'gain_since_pct',l:'Getiri%'}},
  {{k:'adx',l:'ADX'}}, {{k:'cmf',l:'CMF'}}, {{k:'rvol',l:'RVOL'}},
  {{k:'trend_score',l:'T'}}, {{k:'participation_score',l:'P'}},
  {{k:'expansion_score',l:'E'}},
];

let curGrpTab = 'summary';

function scoreBadge(s) {{
  if (s == null) return '';
  const cls = s >= 70 ? 'score-high' : s >= 45 ? 'score-mid' : 'score-low';
  return `<span class="score-badge ${{cls}}">${{s}}</span>`;
}}

function fmtVal(k, v) {{
  if (v == null || v === '') return '-';
  if (k === 'ticker') return `<a class="tv-link" href="https://www.tradingview.com/chart/?symbol=${{v}}" target="_blank">${{v}}</a>`;
  if (k === 'score') return scoreBadge(v);
  if (k === 'module') return `<span class="mod-badge mod-${{v}}">${{v}}</span>`;
  if (k === 'group') return `<span class="grp-badge grp-${{v}}">${{v}}</span>`;
  if (k === 'regime_name') return `<span class="rn-badge rn-${{v}}">${{v}}</span>`;
  if (k === 'entry_window') return v && v !== '-' ? `<span class="ew-badge ew-${{v}}">${{v}}</span>` : '-';
  if (k === 'exit_stage') {{
    const cls = 'exit-' + Math.min(v, 3);
    return `<span class="${{cls}}">${{v}}</span>`;
  }}
  if (k === 'oe_score') {{
    const cls = 'oe-' + Math.min(v, 4);
    return `<span class="${{cls}}">${{v}}</span>`;
  }}
  if (k === 'change_pct') return `<span class="${{v>0?'dir-up':'dir-down'}}">${{v>0?'+':''}}${{v.toFixed(1)}}%</span>`;
  if (k === 'gain_since_pct') return v != null ? `<span class="${{v>0?'dir-up':'dir-down'}}">${{v>0?'+':''}}${{v.toFixed(1)}}%</span>` : '-';
  if (k === 'direction') return `<span class="${{v==='UP'?'dir-up':'dir-down'}}">${{v}}</span>`;
  if (k === 'dollar_vol') {{
    const m = v / 1e6;
    const cls = m >= 50 ? 'liq-high' : m >= 10 ? 'liq-mid' : 'liq-low';
    return `<span class="score-badge ${{cls}}">${{m >= 1 ? '$' + m.toFixed(0) + 'M' : '$' + (v/1e3).toFixed(0) + 'K'}}</span>`;
  }}
  if (k === 'above_ema21') return v ? '✅' : '❌';
  if (k === 'has_senior') return v ? '⭐' : '-';
  if (k === 'rs') return v != null ? `<span class="${{v>1?'rs-pos':'rs-neg'}}">${{v.toFixed(2)}}</span>` : '-';
  if (k === 'subtype') return `<span class="mod-badge mod-${{v}}">${{v}}</span>`;
  if (k === 'risk_pct') {{
    const cls = v < 3 ? 'risk-low' : v <= 5 ? 'risk-mid' : 'risk-high';
    return `<span class="${{cls}}">${{v.toFixed(1)}}%</span>`;
  }}
  if (k === 'trigger' || k === 'stop' || k === 'invalidation') return v != null ? v.toFixed(2) : '-';
  if (k === 'gap_risk') return v != null ? v.toFixed(1) + '%' : '-';
  if (k === 'spread_proxy') return v != null ? v.toFixed(2) + '%' : '-';
  if (typeof v === 'number') return v % 1 === 0 ? v.toString() : v.toFixed(2);
  return v;
}}

function buildGenericTable(items, cols, sortKey) {{
  if (!items || !items.length) return '<p style="color:var(--text-muted);padding:20px">Sinyal bulunamadi</p>';
  let h = '<div class="nox-table-wrap"><table><thead><tr>';
  cols.forEach((c,i) => {{
    h += `<th onclick="sortGrpTable('${{sortKey}}',{{k:'${{c.k}}',i:${{i}}}})">` + c.l + '</th>';
  }});
  h += '</tr></thead><tbody>';
  items.forEach(r => {{
    const hl = (r.score != null && r.score >= 70) ? ' class="hl"' : '';
    h += `<tr${{hl}}>`;
    cols.forEach(c => {{
      h += `<td>${{fmtVal(c.k, r[c.k])}}</td>`;
    }});
    h += '</tr>';
  }});
  h += '</tbody></table></div>';
  return h;
}}

function buildModTable(modKey) {{
  const items = D[modKey] || [];
  return buildGenericTable(items, MOD_COLS[modKey] || [], modKey);
}}

function buildSummary() {{
  let h = '';

  // Grup kartları
  h += '<div class="summary-cards">';
  const grps = [
    {{id:'A', label:'A Grubu', color:'#4ade80', emoji:'🟢', desc:'Catalyst + Regime Active + Erken'}},
    {{id:'B', label:'B Grubu', color:'#facc15', emoji:'🟡', desc:'Catalyst + Regime Active'}},
    {{id:'C', label:'Watchlist', color:'#60a5fa', emoji:'🔵', desc:'Catalyst, Regime bekliyor'}},
    {{id:'D', label:'Teknik', color:'#9ca3af', emoji:'⚪', desc:'Sadece Regime Active'}},
  ];
  grps.forEach(g => {{
    const cnt = (G[g.id]||[]).length;
    h += `<div class="summary-card" style="cursor:pointer" onclick="switchGrpTab('${{g.id}}')">
      <div class="card-num" style="color:${{g.color}}">${{cnt}}</div>
      <div class="card-label">${{g.emoji}} ${{g.label}}</div>
      <div style="font-size:0.65rem;color:var(--text-muted);margin-top:4px">${{g.desc}}</div>
    </div>`;
  }});
  h += '</div>';

  // Modül kartları
  h += '<div class="summary-cards">';
  const modOrder = ['ACCUM','BREAKOUT','SQUEEZE','BIOTECH','EARNINGS','INSIDER','VOLUME'];
  const modColors = {{
    ACCUM:'var(--nox-orange)', VOLUME:'var(--nox-blue)', SQUEEZE:'var(--nox-red)', INSIDER:'var(--nox-purple)',
    BIOTECH:'var(--nox-green)', EARNINGS:'var(--nox-yellow)', BREAKOUT:'var(--nox-cyan)',
  }};
  const modEmoji = {{ACCUM:'🔍', VOLUME:'📈', SQUEEZE:'🔴', INSIDER:'💼', BIOTECH:'🧬', EARNINGS:'📊', BREAKOUT:'🔺'}};
  modOrder.forEach(m => {{
    const cnt = (D[m]||[]).length;
    h += `<div class="summary-card" style="cursor:pointer" onclick="switchGrpTab('${{m}}')">
      <div class="card-num" style="color:${{modColors[m]}}">${{cnt}}</div>
      <div class="card-label">${{modEmoji[m]}} ${{m}}</div>
    </div>`;
  }});
  h += '</div>';

  // Top A sinyalleri
  const aList = G['A'] || [];
  if (aList.length) {{
    h += '<h3 style="color:#4ade80;margin:16px 0 12px;font-size:0.9rem">🟢 A Grubu — En Iyi Sinyaller</h3>';
    h += buildGenericTable(aList.slice(0, 15), AB_COLS, 'sumA');
  }}

  // Top all sinyaller
  let all = [];
  modOrder.forEach(m => {{
    (D[m]||[]).forEach(r => all.push({{...r, _mod:m}}));
  }});
  all.sort((a,b) => (b.score||0) - (a.score||0));
  const top = all.slice(0, 20);

  if (top.length) {{
    h += '<h3 style="color:var(--nox-cyan);margin:16px 0 12px;font-size:0.9rem">⭐ En Guclu Sinyaller (Top 20)</h3>';
    h += '<div class="nox-table-wrap"><table><thead><tr>';
    h += '<th>Modul</th><th>Ticker</th><th>Fiyat</th><th>Detay</th><th>Trigger</th><th>Stop</th><th>Risk%</th><th>Likidite</th><th>Skor</th>';
    h += '</tr></thead><tbody>';
    top.forEach(r => {{
      const hl = r.score >= 70 ? ' class="hl"' : '';
      let detail = '';
      if (r._mod === 'ACCUM') detail = `RVOL:${{r.avg_rvol}} R/ATR:${{r.range_vs_atr}} ${{r.accum_days}}g birikim`;
      else if (r._mod === 'VOLUME') detail = `RVOL:${{r.rvol}} ${{r.change_pct>0?'+':''}}${{r.change_pct}}%`;
      else if (r._mod === 'SQUEEZE') detail = `Short:${{r.short_pct}}% DTC:${{r.short_ratio||'-'}}`;
      else if (r._mod === 'INSIDER') detail = `${{r.n_buyers}} alici ${{(r.buy_value_k||0).toFixed(0)}}K`;
      else if (r._mod === 'BIOTECH') detail = r.fda_date ? `FDA:${{r.fda_date}} ${{r.fda_drug||''}}` : `MCap:${{r.mcap_b||'-'}}B Vol:${{r.vol_trend}}`;
      else if (r._mod === 'EARNINGS') detail = `${{r.subtype||''}} ${{r.days_to_earnings||''}}g`;
      else if (r._mod === 'BREAKOUT') detail = `${{r.subtype||''}} ATR:${{r.atr_ratio}} ${{r.consol_days}}g`;

      h += `<tr${{hl}}>`;
      h += `<td><span class="mod-badge mod-${{r._mod}}">${{r._mod}}</span></td>`;
      h += `<td><a class="tv-link" href="https://www.tradingview.com/chart/?symbol=${{r.ticker}}" target="_blank">${{r.ticker}}</a></td>`;
      h += `<td>${{r.close}}</td>`;
      h += `<td style="font-size:0.72rem">${{detail}}</td>`;
      h += `<td>${{fmtVal('trigger', r.trigger)}}</td>`;
      h += `<td>${{fmtVal('stop', r.stop)}}</td>`;
      h += `<td>${{fmtVal('risk_pct', r.risk_pct)}}</td>`;
      h += `<td>${{fmtVal('dollar_vol', r.dollar_vol)}}</td>`;
      h += `<td>${{scoreBadge(r.score)}}</td>`;
      h += '</tr>';
    }});
    h += '</tbody></table></div>';
  }}
  return h;
}}

function buildGroupTab(grpKey) {{
  const items = G[grpKey] || [];
  if (grpKey === 'D') return buildGenericTable(items, D_COLS, 'D');
  if (grpKey === 'C') return buildGenericTable(items, C_COLS, 'C');
  return buildGenericTable(items, AB_COLS, grpKey);
}}

function switchGrpTab(t) {{
  curGrpTab = t;
  document.querySelectorAll('.nox-tab').forEach(x => x.classList.remove('active'));
  document.getElementById('gtab-' + t).classList.add('active');
  document.querySelectorAll('.tab-content').forEach(x => x.classList.remove('active'));
  document.getElementById('gtc-' + t).classList.add('active');
}}

let sortState = {{}};
function sortGrpTable(sortKey, col) {{
  // Determine data source
  const isGroup = ['A','B','C','D'].includes(sortKey);
  const isModTab = sortKey in D;
  let items = null;

  if (isGroup) items = G[sortKey];
  else if (isModTab) items = D[sortKey];
  else return;

  if (!items || !items.length) return;

  const key = col.k;
  const prev = sortState[sortKey + '_' + key];
  const asc = prev === 'desc';
  sortState[sortKey + '_' + key] = asc ? 'asc' : 'desc';
  items.sort((a,b) => {{
    let va = a[key], vb = b[key];
    if (va == null) return 1;
    if (vb == null) return -1;
    if (typeof va === 'string') return asc ? va.localeCompare(vb) : vb.localeCompare(va);
    return asc ? va - vb : vb - va;
  }});

  const elId = 'gtc-' + sortKey;
  const el = document.getElementById(elId);
  if (el) {{
    if (isGroup) el.innerHTML = buildGroupTab(sortKey);
    else el.innerHTML = buildModTable(sortKey);
  }}
}}

// Init
function init() {{
  const tabsEl = document.getElementById('grp-tabs');
  const contentsEl = document.getElementById('grp-contents');

  GRP_TABS.forEach(t => {{
    let cnt = '';
    if (['A','B','C','D'].includes(t.id)) cnt = (G[t.id]||[]).length;
    else if (t.id !== 'summary') cnt = (D[t.id]||[]).length;

    const d = document.createElement('div');
    d.className = 'nox-tab' + (t.id === 'summary' ? ' active' : '');
    d.id = 'gtab-' + t.id;
    d.innerHTML = t.emoji + ' ' + t.label + (cnt !== '' ? ` <span class="cnt">${{cnt}}</span>` : '');
    d.onclick = () => switchGrpTab(t.id);
    tabsEl.appendChild(d);

    const tc = document.createElement('div');
    tc.className = 'tab-content' + (t.id === 'summary' ? ' active' : '');
    tc.id = 'gtc-' + t.id;

    if (t.id === 'summary') tc.innerHTML = buildSummary();
    else if (['A','B','C','D'].includes(t.id)) tc.innerHTML = buildGroupTab(t.id);
    else tc.innerHTML = buildModTable(t.id);

    contentsEl.appendChild(tc);
  }});
}}

init();
</script>
</body>
</html>"""

    return html


# ══════════════════════════════════════════
# TELEGRAM FORMATLAMA
# ══════════════════════════════════════════

def _format_telegram(all_results, n_scanned, html_url=None, regime_info=None,
                     grouped=None):
    """Telegram mesajı formatla."""
    now_str = datetime.now(timezone(timedelta(hours=-5))).strftime('%Y-%m-%d %H:%M ET')
    total = sum(len(v) for v in all_results.values())

    ri = regime_info or {'regime': 'NEUTRAL', 'score': 3, 'max_score': 6}
    regime_em = {
        'BULL': '\U0001f7e2', 'NEUTRAL': '\U0001f7e1', 'RISK_OFF': '\U0001f534'
    }.get(ri['regime'], '')

    lines = []
    lines.append(f'<b>⬡ NOX US Catalyst — {now_str}</b>')
    lines.append(f"\U0001f4ca SPY: {regime_em} {ri['regime']} ({ri['score']}/{ri['max_score']})")

    # Grup sayıları
    if grouped:
        ga, gb, gc, gd = len(grouped['A']), len(grouped['B']), len(grouped['C']), len(grouped['D'])
        lines.append(f'🏷️ A:{ga} B:{gb} C:{gc} D:{gd}')

    if html_url:
        lines.append(f'🔗 <a href="{html_url}">Raporu Ac</a>')

    lines.append(f'\n📋 {n_scanned} taranan | {total} sinyal')

    # Modül sayıları
    counts = []
    for mod_key, emoji in MODULE_EMOJI.items():
        n = len(all_results.get(mod_key, []))
        if n > 0:
            counts.append(f'{emoji}{mod_key}:{n}')
    if counts:
        lines.append(' '.join(counts))

    # A Grubu sinyalleri
    if grouped and grouped['A']:
        lines.append(f'\n<b>⭐ A Grubu ({len(grouped["A"])})</b>')
        lines.append('─' * 20)

        for s in grouped['A'][:15]:
            mod = s.get('module', '')
            emoji = MODULE_EMOJI.get(mod, '')
            rn = s.get('regime_name', '')
            ew = s.get('entry_window', '')
            oe = s.get('oe_score', 0)
            t_val = s.get('trigger')
            s_val = s.get('stop')
            sc = s.get('score', 0)
            line = f'{emoji} {s["ticker"]} {s["close"]:.2f} [{mod}] {rn} {ew} OE:{oe}'
            if t_val and s_val:
                line += f' T:{t_val:.2f} S:{s_val:.2f}'
            line += f' S:{sc}'
            lines.append(line)
    elif grouped:
        # Grouped var ama A boş — top sinyalleri göster
        pass

    # A yoksa veya grouped yoksa top sinyaller
    if not grouped or not grouped['A']:
        all_sigs = []
        for mod_key, signals in all_results.items():
            for s in signals:
                all_sigs.append({**s, '_mod': mod_key})
        all_sigs.sort(key=lambda x: x.get('score', 0), reverse=True)
        top = all_sigs[:15]

        if top:
            lines.append(f'\n<b>⭐ One Cikanlar ({len(top)})</b>')
            lines.append('─' * 20)

            for s in top:
                mod = s['_mod']
                emoji = MODULE_EMOJI.get(mod, '')
                line = f'{emoji} {s["ticker"]} {s["close"]:.2f} [{mod}]'

                if mod == 'ACCUM':
                    line += f' RVOL:{s["avg_rvol"]:.2f} {s["accum_days"]}g'
                elif mod == 'VOLUME':
                    line += f' RVOL:{s["rvol"]:.1f} {s["change_pct"]:+.1f}%'
                elif mod == 'SQUEEZE':
                    line += f' Short:{s["short_pct"]:.0f}%'
                elif mod == 'INSIDER':
                    line += f' {s["n_buyers"]} alici ${s.get("buy_value_k", 0):.0f}K'
                elif mod == 'BIOTECH':
                    if s.get('fda_date'):
                        line += f' FDA:{s["fda_date"]}'
                    else:
                        line += f' MCap:${s.get("mcap_b", 0) or 0:.1f}B'
                elif mod == 'EARNINGS':
                    line += f' [{s.get("subtype", "")}] {s.get("days_to_earnings", "")}g'
                elif mod == 'BREAKOUT':
                    line += f' [{s.get("subtype", "")}] ATR:{s.get("atr_ratio", 0):.2f}'

                t_val = s.get('trigger')
                s_val = s.get('stop')
                r_val = s.get('risk_pct')
                if t_val and s_val:
                    line += f' T:{t_val:.2f} S:{s_val:.2f}'
                if r_val:
                    line += f' R:{r_val:.1f}%'

                line += f' S:{s["score"]}'
                lines.append(line)

    return '\n'.join(lines)


# ══════════════════════════════════════════
# BUCKET PRE-FILTER HELPER
# ══════════════════════════════════════════

def _compute_ticker_stats(stock_dfs, spy_df):
    """Her ticker için temel istatistikleri bir kez hesapla (bucket pre-filter)."""
    stats = {}
    for t, df in stock_dfs.items():
        if len(df) < 30:
            continue
        close = df['Close']
        vol = df['Volume']
        high = df['High']
        low = df['Low']
        avg_vol_20 = vol.rolling(20, min_periods=10).mean().iloc[-1]
        if pd.isna(avg_vol_20) or avg_vol_20 <= 0:
            continue

        # ATR ratio (sıkışma tespiti)
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr_14 = tr.rolling(14, min_periods=14).mean()
        atr_cur = atr_14.iloc[-1]
        atr_60_avg = atr_14.iloc[-60:].mean() if len(atr_14) >= 60 else atr_cur
        atr_ratio = atr_cur / atr_60_avg if pd.notna(atr_60_avg) and atr_60_avg > 0 else 1.0

        # Vol slope
        vol_first5 = vol.iloc[-10:-5].mean()
        vol_last5 = vol.iloc[-5:].mean()
        vol_slope = vol_last5 / vol_first5 if vol_first5 > 0 else 1.0

        stats[t] = {
            'rvol_1d': vol.iloc[-1] / avg_vol_20,
            'rvol_5d': vol.iloc[-5:].mean() / avg_vol_20,
            'change_1d': abs((close.iloc[-1] - close.iloc[-2]) / close.iloc[-2] * 100) if close.iloc[-2] != 0 else 0,
            'dollar_vol': (close * vol).rolling(20, min_periods=10).mean().iloc[-1],
            'price': close.iloc[-1],
            'atr_ratio': atr_ratio,
            'vol_slope': vol_slope,
        }
    return stats


# ══════════════════════════════════════════
# REGIME GATING
# ══════════════════════════════════════════

def _apply_regime_gating(all_results, regime_info):
    """Rejim bazlı skor ayarlaması. Modül kodlarına dokunmaz.

    BULL: tüm modüller aktif, skor aynen
    NEUTRAL: VOLUME ve BREAKOUT(aktif) sinyallerinde skor * 0.85
    RISK_OFF: VOLUME ve BREAKOUT(aktif) filtrelenir, SETUP ve ACCUM skor * 1.1
    """
    regime = regime_info.get('regime', 'NEUTRAL')
    if regime == 'BULL':
        return all_results  # Aynen

    for mod_key, signals in all_results.items():
        new_signals = []
        for s in signals:
            if regime == 'NEUTRAL':
                if mod_key == 'VOLUME':
                    s['score'] = int(s['score'] * 0.85)
                elif mod_key == 'BREAKOUT' and s.get('subtype') == 'BREAKOUT':
                    s['score'] = int(s['score'] * 0.85)
            elif regime == 'RISK_OFF':
                # VOLUME ve aktif BREAKOUT filtrelenir
                if mod_key == 'VOLUME':
                    continue
                if mod_key == 'BREAKOUT' and s.get('subtype') == 'BREAKOUT':
                    continue
                # SETUP ve ACCUM kontra fırsat bonusu
                if mod_key == 'ACCUM':
                    s['score'] = min(100, int(s['score'] * 1.1))
                elif mod_key == 'BREAKOUT' and s.get('subtype') == 'SETUP':
                    s['score'] = min(100, int(s['score'] * 1.1))
            new_signals.append(s)
        all_results[mod_key] = new_signals
        # Re-sort
        all_results[mod_key].sort(key=lambda x: x.get('score', 0), reverse=True)

    return all_results


# ══════════════════════════════════════════
# ANA RUNNER
# ══════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='NOX US Catalyst Screener')
    parser.add_argument('--html', action='store_true', help='HTML rapor üret')
    parser.add_argument('--csv', action='store_true', help='CSV çıktı')
    parser.add_argument('--notify', action='store_true', help='Telegram bildirim')
    parser.add_argument('--debug', type=str, help='Tek ticker debug')
    parser.add_argument('--no-enrich', action='store_true', help='Faz 2 atla (sadece teknik)')
    parser.add_argument('--min-mcap', type=float, default=300,
                        help='Min market cap (milyon $, varsayılan: 300)')
    parser.add_argument('--ics', type=str, default=None,
                        help='CatalystAlert ICS takvim dosyası yolu')
    args = parser.parse_args()

    t0 = time.time()
    print("⬡ NOX US Catalyst Screener")
    print("═" * 40)

    # ── 1. Ticker listesi ──
    if args.debug:
        tickers = [args.debug.upper()]
        print(f"🔍 Debug: {tickers[0]}")
    else:
        min_mcap = args.min_mcap * 1e6  # Milyon $ → $
        tickers = get_all_us_tickers(min_mcap=min_mcap)

    # ── 2. Batch OHLCV download ──
    stock_dfs = fetch_data(tickers, period="1y")
    spy_df = fetch_benchmark(period="1y")
    n_scanned = len(stock_dfs)

    if not stock_dfs:
        print("❌ Veri indirilemedi!")
        sys.exit(1)

    # ── 3. FDA/PDUFA Takvimi (enrichment'tan önce — ticker listesine eklenir) ──
    fda_calendar = {}
    if not args.no_enrich:
        ics_path = args.ics
        if ics_path is None:
            ics_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    'data', 'catalysts', 'catalyst-calendar.ics')
        fda_calendar = fetch_fda_calendar(ics_path=ics_path)

    # ── 4. Faz 2: Enrichment (opsiyonel) ──
    ticker_info = None
    insider_data = None

    if not args.no_enrich:
        # Bucket-based enrichment pre-filter
        enrich_tickers = list(stock_dfs.keys())

        if len(enrich_tickers) > 100 and not args.debug:
            stats = _compute_ticker_stats(stock_dfs, spy_df)
            event_set = set()   # BIOTECH, EARNINGS, INSIDER — düşük bar
            flow_set = set()    # VOLUME, ACCUM — hacim tabanlı
            tech_set = set()    # BREAKOUT, SQUEEZE — teknik

            for t, s in stats.items():
                # Event bucket: price > $5 AND dollar_vol > $500K
                if s['price'] > 5.0 and s['dollar_vol'] > 500_000:
                    event_set.add(t)
                # Flow bucket: rvol_5d >= 1.2 OR vol_slope rising
                if s['rvol_5d'] >= 1.2 or s.get('vol_slope', 1.0) > 1.1:
                    flow_set.add(t)
                # Technical bucket: atr_ratio < 0.7 OR rvol_1d >= 1.5
                if s.get('atr_ratio', 1.0) < 0.7 or s['rvol_1d'] >= 1.5:
                    tech_set.add(t)

            # Union → enrichment
            enrich_set = event_set | flow_set | tech_set
            print(f"📋 Bucket pre-filter: Event:{len(event_set)} Flow:{len(flow_set)} Tech:{len(tech_set)} → Union:{len(enrich_set)}")

            # FDA takvimindeki ticker'ları ön-filtreden bağımsız ekle
            if fda_calendar:
                fda_tickers = [t for t in fda_calendar if t in stock_dfs and t not in enrich_set]
                if fda_tickers:
                    enrich_set.update(fda_tickers)
                    print(f"  + {len(fda_tickers)} FDA/PDUFA ticker eklendi → toplam {len(enrich_set)}")

            # Cap: 300 ticker
            max_enrich = 300
            if len(enrich_set) > max_enrich:
                # Skora göre sırala — rvol_1d + change_1d
                scored = [(t, stats[t]['rvol_1d'] + stats[t]['change_1d']) for t in enrich_set if t in stats]
                scored.sort(key=lambda x: x[1], reverse=True)
                enrich_tickers = [t for t, _ in scored[:max_enrich]]
            else:
                enrich_tickers = list(enrich_set)

            if not enrich_tickers:
                enrich_tickers = list(stock_dfs.keys())[:80]
            print(f"📋 Enrichment adayı: {len(enrich_tickers)} ticker")

        ticker_info = fetch_ticker_info(enrich_tickers)
        insider_data = fetch_insider_data(enrich_tickers)

    # ── 4. SPY Rejim ──
    regime_info = compute_spy_regime(spy_df)
    regime = regime_info['regime']
    regime_score = regime_info['score']
    regime_max = regime_info['max_score']
    regime_emoji = {'BULL': '\U0001f7e2', 'NEUTRAL': '\U0001f7e1', 'RISK_OFF': '\U0001f534'}.get(regime, '')
    print(f"\n📊 SPY Rejim: {regime_emoji} {regime} ({regime_score}/{regime_max})")

    # ── 5. Tarama ──
    all_results = run_all_modules(stock_dfs, spy_df, ticker_info, insider_data,
                                   fda_calendar)

    # ── 6. Rejim Gating ──
    all_results = _apply_regime_gating(all_results, regime_info)

    # ── 6b. Per-stock Regime Scan ──
    t_regime = time.time()
    print(f"\n🔍 Per-stock regime taraması ({len(stock_dfs)} hisse)...")
    regime_data, n_regime = scan_all_regimes(stock_dfs)
    n_active = sum(1 for v in regime_data.values() if v['in_trade'] and v['regime'] >= 2)
    print(f"   {n_regime} tarandı, {n_active} regime active ({time.time() - t_regime:.1f}s)")

    # ── 6c. Cross-reference: 4-group classification ──
    grouped = classify_groups(all_results, regime_data)
    ga, gb, gc, gd = len(grouped['A']), len(grouped['B']), len(grouped['C']), len(grouped['D'])
    print(f"   🏷️ A:{ga} B:{gb} C:{gc} D:{gd}")

    # ── 6d. Modül verilerine regime bilgisi ekle (HTML modül tabları için) ──
    # Hızlı lookup: (ticker, module) → group
    _grp_lookup = {}
    for g in ('A', 'B', 'C'):
        for sig in grouped[g]:
            _grp_lookup[(sig.get('ticker'), sig.get('module'))] = g

    for mod_key, signals in all_results.items():
        for s in signals:
            ri = regime_data.get(s['ticker'])
            if ri:
                s['regime_name'] = ri['regime_name']
                s['entry_window'] = ri['entry_window']
                s['exit_stage'] = ri['exit_stage']
                s['oe_score'] = ri['oe_score']
                s['trade_date'] = ri['trade_date']
                s['days_since'] = ri['days_since']
                s['gain_since_pct'] = ri['gain_since_pct']
                s['group'] = _grp_lookup.get((s['ticker'], mod_key), 'C')

    # ── 7. Çıktı ──
    date_str = datetime.now().strftime('%Y-%m-%d')

    _print_results(all_results, n_scanned)

    if args.csv:
        _save_csv(all_results, date_str, grouped=grouped)

    html_url = None
    html_path = None
    if args.html:
        html = _generate_html(all_results, n_scanned, date_str, regime_info,
                              grouped=grouped)
        os.makedirs('output', exist_ok=True)
        html_path = f"output/us_catalyst_{date_str.replace('-', '')}.html"
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html)
        print(f"📄 HTML: {html_path}")

        # GitHub Pages
        html_url = push_html_to_github(html, 'us_catalyst.html', date_str)

    if args.notify:
        msg = _format_telegram(all_results, n_scanned, html_url, regime_info,
                               grouped=grouped)
        send_telegram(msg)
        if html_path:
            send_telegram_document(html_path)

    elapsed = time.time() - t0
    print(f"\n⏱️ Toplam süre: {elapsed:.1f}s")


if __name__ == '__main__':
    main()
