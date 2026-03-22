#!/usr/bin/env python3
"""
NOX Divergence Screener v2 — Runner
=====================================
BIST hisselerinde uyumsuzluk (divergence) taramasi.
RSI, MACD, OBV, MFI, ADX Exhaustion, Uclu Uyumsuzluk, Fiyat-Hacim.

Pivotsuz yaklasim: Lightweight swing detection (order=2).

Kullanim:
    python run_divergence.py                    # tum hisseler
    python run_divergence.py --debug THYAO      # tek hisse detayli
    python run_divergence.py --type TRIPLE      # sadece uclu uyumsuzluk
    python run_divergence.py --type OBV         # sadece OBV uyumsuzluk
    python run_divergence.py --type MFI         # sadece MFI uyumsuzluk
    python run_divergence.py --type ADX         # sadece ADX exhaustion
    python run_divergence.py --top 20           # en iyi 20
    python run_divergence.py --csv              # CSV kaydet
    python run_divergence.py --weekly           # haftalik veri ile tara
    python run_divergence.py THYAO EREGL SISE   # spesifik hisseler
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

from markets.bist import data as data_mod
from markets.bist.divergence import (
    scan_divergences, DIV_CFG, _find_swings, _find_structural_swings,
    _calc_atr, _calc_rsi, _calc_macd, _calc_obv, _calc_mfi, _calc_adx,
    DivergenceSetup, BUCKET_MAP,
)
from core.indicators import ema, resample_weekly, resample_monthly
from core.reports import send_telegram, push_html_to_github, _NOX_CSS, _sanitize


# =============================================================================
# YARDIMCI
# =============================================================================

def _to_lower_cols(df):
    return df.rename(columns={
        'Open': 'open', 'High': 'high', 'Low': 'low',
        'Close': 'close', 'Volume': 'volume',
    })


def _is_halted(df, n=3):
    """Son n bar tamami H==L ise halt/suspend."""
    if len(df) < n:
        return False
    tail = df.tail(n)
    return (tail['high'] == tail['low']).all()


# =============================================================================
# TIP ETIKETLERI (Turkish)
# =============================================================================

TYPE_LABELS = {
    'RSI_CLASSIC': 'RSI Klasik',
    'RSI_HIDDEN': 'RSI Gizli',
    'MACD_CLASSIC': 'MACD Klasik',
    'MACD_HIDDEN': 'MACD Gizli',
    'OBV_CLASSIC': 'OBV Klasik',
    'OBV_HIDDEN': 'OBV Gizli',
    'MFI_CLASSIC': 'MFI Klasik',
    'MFI_HIDDEN': 'MFI Gizli',
    'ADX_EXHAUST': 'ADX Bitisi',
    'TRIPLE': 'Uclu',
    'PRICE_VOLUME': 'Fiyat-Hacim',
}

BUCKET_LABELS = {
    'REVERSAL': 'DONUS',
    'CONTINUATION': 'DEVAM',
    'CONFIRMATION': 'TEYIT',
    'EXHAUSTION': 'BITIM',
}

STATE_LABELS = {
    'SETUP': 'KURULUM',
    'TRIGGERED': 'TETIK',
    'STALE': 'ESKI',
    'INVALIDATED': 'GECERSIZ',
}

SIGNAL_LABEL_SHORT = {
    'ENTRY_LONG': 'EL', 'ENTRY_SHORT': 'ES',
    'REDUCE': 'RED', 'COVER': 'COV', 'WATCH': 'W',
}

SECTION_TITLES = {
    'triple_buy': 'UCLU UYUMSUZLUK (RSI + MACD + Hacim) — AL',
    'triple_sell': 'UCLU UYUMSUZLUK (RSI + MACD + Hacim) — SAT',
    'rsi_buy': 'RSI UYUMSUZLUK — AL',
    'rsi_sell': 'RSI UYUMSUZLUK — SAT',
    'macd_buy': 'MACD UYUMSUZLUK — AL',
    'macd_sell': 'MACD UYUMSUZLUK — SAT',
    'obv_buy': 'OBV UYUMSUZLUK — AL',
    'obv_sell': 'OBV UYUMSUZLUK — SAT',
    'mfi_buy': 'MFI UYUMSUZLUK — AL',
    'mfi_sell': 'MFI UYUMSUZLUK — SAT',
    'adx_buy': 'ADX TREND BITISI — AL',
    'adx_sell': 'ADX TREND BITISI — SAT',
    'pv_buy': 'FIYAT-HACIM UYUMSUZLUGU — AL',
    'pv_sell': 'FIYAT-HACIM UYUMSUZLUGU — SAT',
}


# =============================================================================
# KURAL BAZLI SINIFLANDIRMA (Backtest 2y, trigger_bar fix, discovery/validation dogrulanmis)
# =============================================================================
# A kaldirildi — TRIGGERED tek basina edge yok (%48.5 WR, forward-looking bias idi)
# D yeniden tanimlandi — MACD_HIDDEN+RR>=2+YAPISAL (premium, dogrulanmis %62.5)
# C STALE kisitlamasi kaldirildi — dogrulama combo state gerektirmiyordu

RULE_DEFS = {
    'B': {
        'name': 'Kisa Setup',
        'desc': 'ENTRY_SHORT + SETUP + RR>=1.5 (SAT)',
        'color': 'var(--nox-red)',
        'stats': {'1g_wr': 62.7, '1g_avg': 0.24, '3g_wr': 59.7, '3g_avg': 0.48,
                  '5g_wr': 76.1, '5g_avg': 1.66, 'n_bt': 67},
    },
    'D': {
        'name': 'MACD Premium',
        'desc': 'MACD_HIDDEN + RR>=2.0 + Yapisal — dogrulanmis premium katman',
        'color': 'var(--nox-orange)',
        'stats': {'1g_wr': 54.2, '1g_avg': 0.52, '3g_wr': 57.1, '3g_avg': 1.57,
                  '5g_wr': 61.1, '5g_avg': 2.27, 'n_bt': 275},
    },
    'C': {
        'name': 'Yapisal Gizli',
        'desc': 'ENTRY_LONG + Yapisal + Hidden tip — ana AL motoru',
        'color': 'var(--nox-green)',
        'stats': {'1g_wr': 49.9, '1g_avg': 0.35, '3g_wr': 50.7, '3g_avg': 0.45,
                  '5g_wr': 59.2, '5g_avg': 1.50, 'n_bt': 507},
    },
    'E': {
        'name': 'Cover',
        'desc': 'COVER + STALE + RR>=1.0 — taktik pozisyon (dusuk agirlik)',
        'color': 'var(--nox-blue)',
        'stats': {'1g_wr': 44.2, '1g_avg': 0.24, '3g_wr': 53.8, '3g_avg': 1.40,
                  '5g_wr': 54.8, '5g_avg': 1.93, 'n_bt': 104},
    },
}

RULE_ORDER = ['B', 'D', 'C', 'E']

# Modifier (booster/caution) tag'leri — sinyalin yaninda gosterilir
MOD_DEFS = {
    'FT': {'label': '↑FT', 'desc': 'FULL_TREND rejim', 'color': 'var(--nox-green)'},
    'RR': {'label': '↑RR', 'desc': 'RR>=2.0', 'color': 'var(--nox-cyan)'},
    'RSI': {'label': '↑RSI', 'desc': 'RSI_HIDDEN (en tutarli div type)', 'color': 'var(--nox-cyan)'},
    'K60': {'label': '⚠K60', 'desc': 'K>=60 — dogrulamada kirilgan, dikkat', 'color': 'var(--nox-yellow)'},
}


def _classify_rule(sig):
    """Sinyali 4 kuraldan birine esle. None = gurultu, kesilecek.
    Oncelik: B > D > C > E (D daha spesifik, C'den once kontrol)."""
    state = getattr(sig, 'state', '')
    label = getattr(sig, 'signal_label', '')
    rr = getattr(sig, 'rr_ratio', 0) or 0
    struct = getattr(sig, 'structural', False)
    dt = getattr(sig, 'div_type', '')
    is_hidden = 'HIDDEN' in dt

    # B: Setup + Entry Short + RR>=1.5 (SAT)
    if state == 'SETUP' and label == 'ENTRY_SHORT' and rr >= 1.5:
        return 'B'
    # D: MACD_HIDDEN + RR>=2 + Yapisal (premium AL, C'den once)
    if dt == 'MACD_HIDDEN' and rr >= 2.0 and struct:
        return 'D'
    # C: Entry Long + Yapisal + Hidden (ana AL motoru, state kisitlamasi yok)
    if label == 'ENTRY_LONG' and struct and is_hidden:
        return 'C'
    # E: Cover + Stale + RR>=1 (taktik)
    if state == 'STALE' and label == 'COVER' and rr >= 1.0:
        return 'E'
    # Gurultu
    return None


def _classify_modifiers(sig):
    """Global modifier etiketleri — sinyalin kalitesini belirler."""
    mods = []
    regime = getattr(sig, 'regime', -1)
    if regime == 3:
        mods.append('FT')  # FULL_TREND boost
    rr = getattr(sig, 'rr_ratio', 0) or 0
    if rr >= 2.0:
        mods.append('RR')  # RR>=2 boost
    dt = getattr(sig, 'div_type', '')
    if dt == 'RSI_HIDDEN':
        mods.append('RSI')  # RSI_HIDDEN premium
    q = getattr(sig, 'quality', 0)
    if q >= 60:
        mods.append('K60')  # K>=60 caution
    return mods


# =============================================================================
# TARAMA
# =============================================================================

def _scan_all(stock_dfs, debug_ticker=None, scan_bars=10, min_bars=60, weekly_data=None):
    """Tum hisselerde divergence taramasi."""
    all_results = {
        'rsi': [], 'macd': [], 'obv': [], 'mfi': [], 'adx': [],
        'triple': [], 'pv': [],
        'primary': [], 'confirmation': [], 'exhaustion': [],
    }
    n_scanned = 0
    last_date = None

    for ticker, df in stock_dfs.items():
        if len(df) < min_bars:
            continue
        if _is_halted(df):
            continue

        try:
            if debug_ticker and ticker == debug_ticker:
                _print_debug(ticker, df)

            wdf = weekly_data.get(ticker) if weekly_data else None
            result = scan_divergences(df, scan_bars=scan_bars, weekly_df=wdf)
            n_scanned += 1

            if last_date is None and len(df) > 0:
                last_date = df.index[-1]

            for sig_type in ('rsi', 'macd', 'obv', 'mfi', 'adx', 'triple', 'pv',
                             'primary', 'confirmation', 'exhaustion'):
                for sig in result.get(sig_type, []):
                    if sig.bar_idx < len(df):
                        sig_date = df.index[sig.bar_idx]
                    else:
                        sig_date = df.index[-1]

                    all_results[sig_type].append({
                        'ticker': ticker,
                        'close': float(df['close'].iloc[-1]),
                        'signal': sig,
                        'signal_date': sig_date,
                        'signal_close': float(df['close'].iloc[sig.bar_idx]) if sig.bar_idx < len(df) else float(df['close'].iloc[-1]),
                        'bars_ago': len(df) - 1 - sig.bar_idx,
                    })

        except Exception as e:
            print(f"  ! {ticker}: {e}")
            continue

    date_str = last_date.strftime('%Y-%m-%d') if last_date else datetime.now().strftime('%Y-%m-%d')
    return all_results, n_scanned, date_str


# =============================================================================
# DEBUG
# =============================================================================

def _print_debug(ticker, df):
    """Tek hisse detayli swing + indicator + sinyal ciktisi."""
    cfg = DIV_CFG
    print(f"\n  {'=' * 80}")
    print(f"  DEBUG: {ticker}")
    print(f"  {'=' * 80}")

    atr = _calc_atr(df, cfg['atr_len'])
    rsi = _calc_rsi(df['close'], cfg['rsi_len'])
    macd_line, signal_line, macd_hist = _calc_macd(
        df['close'], cfg['macd_fast'], cfg['macd_slow'], cfg['macd_signal']
    )
    obv = _calc_obv(df)
    obv_ema_series = ema(obv, cfg['obv_ema_len'])
    mfi = _calc_mfi(df, cfg['mfi_len'])
    adx = _calc_adx(df, cfg['adx_len'])

    print(f"\n  Son ATR: {atr.iloc[-1]:.4f}")
    print(f"  Son Close: {df['close'].iloc[-1]:.2f}")
    print(f"  Son RSI: {rsi.iloc[-1]:.1f}")
    print(f"  Son MACD Hist: {macd_hist.iloc[-1]:.4f}")
    print(f"  Son MACD Line: {macd_line.iloc[-1]:.4f}")
    print(f"  Son MFI: {mfi.iloc[-1]:.1f}")
    print(f"  Son ADX: {adx.iloc[-1]:.1f}")

    # Structural swing'ler
    struct_lows, struct_highs = _find_structural_swings(
        df['close'], atr, cfg['swing_order'],
        cfg['structural_min_atr_dist'], cfg['structural_min_bar_gap']
    )

    print(f"\n  Structural Swing Lows ({sum(1 for s in struct_lows if s[3])} yapisal / {len(struct_lows)} toplam, son 10):")
    print(f"  {'SwingIdx':>8} {'SinyalIdx':>9} {'Fiyat':>10} {'Yapisal':>8}")
    print(f"  {'─' * 40}")
    for sw in struct_lows[-10:]:
        tag = '  ✓' if sw[3] else ''
        print(f"  {sw[0]:>8} {sw[1]:>9} {sw[2]:>10.4f} {tag:>8}")

    print(f"\n  Structural Swing Highs ({sum(1 for s in struct_highs if s[3])} yapisal / {len(struct_highs)} toplam, son 10):")
    print(f"  {'SwingIdx':>8} {'SinyalIdx':>9} {'Fiyat':>10} {'Yapisal':>8}")
    print(f"  {'─' * 40}")
    for sw in struct_highs[-10:]:
        tag = '  ✓' if sw[3] else ''
        print(f"  {sw[0]:>8} {sw[1]:>9} {sw[2]:>10.4f} {tag:>8}")

    # Son 15 bar indicator degerleri
    n = len(df)
    print(f"\n  Son 15 Bar Indicator Degerleri:")
    print(f"  {'Bar':>5} {'Tarih':>12} {'Close':>10} {'RSI':>7} {'MACD H':>10} "
          f"{'MFI':>7} {'ADX':>7} {'ATR':>10} {'Vol':>12}")
    print(f"  {'─' * 90}")
    for i in range(max(0, n - 15), n):
        date_str = df.index[i].strftime('%Y-%m-%d') if hasattr(df.index[i], 'strftime') else str(df.index[i])
        r = rsi.iloc[i] if not np.isnan(rsi.iloc[i]) else 0
        mh = macd_hist.iloc[i] if not np.isnan(macd_hist.iloc[i]) else 0
        m = mfi.iloc[i] if not np.isnan(mfi.iloc[i]) else 0
        ax = adx.iloc[i] if not np.isnan(adx.iloc[i]) else 0
        a = atr.iloc[i] if not np.isnan(atr.iloc[i]) else 0
        v = df['volume'].iloc[i]
        print(f"  {i:>5} {date_str:>12} {df['close'].iloc[i]:>10.2f} {r:>7.1f} "
              f"{mh:>10.4f} {m:>7.1f} {ax:>7.1f} {a:>10.4f} {v:>12.0f}")
    print(f"  {'─' * 90}")

    # Sinyalleri genis scan ile tara
    result = scan_divergences(df, scan_bars=30)
    # Legacy key'ler icin say
    legacy_total = sum(len(result.get(k, [])) for k in ('triple', 'rsi', 'macd', 'obv', 'mfi', 'adx', 'pv'))

    if legacy_total > 0:
        print(f"\n  Bulunan Sinyaller ({legacy_total}):")
        print(f"  {'Bar':>5} {'Yon':<5} {'Tip':<15} {'K':>3} {'Bucket':<7} {'State':<8} {'Trg':<6} {'S':>1} {'Detay'}")
        print(f"  {'─' * 90}")
        for sig_type in ('triple', 'rsi', 'macd', 'obv', 'mfi', 'adx', 'pv'):
            for sig in result.get(sig_type, []):
                det_parts = []
                d = sig.details
                if 'prev_rsi' in d:
                    det_parts.append(f"RSI: {d['prev_rsi']:.0f}→{d['curr_rsi']:.0f}")
                if 'prev_hist' in d:
                    det_parts.append(f"Hist: {d['prev_hist']:.4f}→{d['curr_hist']:.4f}")
                if 'prev_obv' in d:
                    det_parts.append(f"OBV: {d['prev_obv']:.0f}→{d['curr_obv']:.0f}")
                if 'prev_mfi' in d:
                    det_parts.append(f"MFI: {d['prev_mfi']:.0f}→{d['curr_mfi']:.0f}")
                if 'adx_slope' in d:
                    det_parts.append(f"ADX: {d['adx_value']:.0f} s={d['adx_slope']:.2f}")
                if 'sub_type' in d:
                    det_parts.append(f"{d['sub_type']}")
                    det_parts.append(f"VolR: {d.get('vol_ratio', 0):.1f}")
                    if d.get('limit_tag'):
                        det_parts.append(f"[{d['limit_tag']}]")
                elif 'vol_ratio' in d:
                    det_parts.append(f"VolR: {d['vol_ratio']:.1f}")
                if d.get('has_mfi'):
                    det_parts.append("MFI+")
                det_str = ', '.join(det_parts) if det_parts else '-'
                bucket = getattr(sig, 'bucket', '-')[:5]
                state = getattr(sig, 'state', '-')[:6]
                trg = getattr(sig, 'trigger_type', '-')[:6] if getattr(sig, 'trigger_type', 'NONE') != 'NONE' else '-'
                struct = '✓' if getattr(sig, 'structural', False) else ' '
                print(f"  {sig.bar_idx:>5} {sig.direction:<5} {sig.div_type:<15} "
                      f"{sig.quality:>3} {bucket:<7} {state:<8} {trg:<6} {struct:>1} {det_str}")

        # State dagilimi ozeti
        all_sigs = []
        for k in ('triple', 'rsi', 'macd', 'mfi', 'adx'):
            all_sigs.extend(result.get(k, []))
        state_counts = {}
        for s in all_sigs:
            st = getattr(s, 'state', 'SETUP')
            state_counts[st] = state_counts.get(st, 0) + 1
        if state_counts:
            sc_str = ' | '.join(f"{k}:{v}" for k, v in sorted(state_counts.items()))
            print(f"\n  State Dagilimi: {sc_str}")
        print(f"  {'─' * 90}")
    else:
        print(f"\n  Sinyal bulunamadi (son 30 bar).")

    print(f"  {'=' * 80}\n")


# =============================================================================
# KONSOL RAPOR
# =============================================================================

def _state_tag(sig):
    """DivergenceSetup icin kompakt state etiketi."""
    state = getattr(sig, 'state', None)
    trg = getattr(sig, 'trigger_type', 'NONE')
    label = getattr(sig, 'signal_label', 'WATCH')
    label_short = SIGNAL_LABEL_SHORT.get(label, '')
    parts = []
    if state == 'TRIGGERED':
        trg_short = {'SWING_BREAK': 'SB', 'EMA_RECLAIM': 'ER', 'VOLUME_REVERSAL': 'VR'}.get(trg, '?')
        parts.append(f"TETIK({trg_short})")
    elif state == 'STALE':
        parts.append('ESKI')
    elif state == 'INVALIDATED':
        parts.append('GCR')
    if label_short and label_short != 'W':
        parts.append(label_short)
    return (' ' + ' '.join(parts)) if parts else ''


def _print_section(title, items, section_type):
    """Bir bolum icin tablo yazdir."""
    if not items:
        return

    # INVALIDATED sinyalleri filtrele
    items = [i for i in items if getattr(i['signal'], 'state', 'SETUP') != 'INVALIDATED']
    if not items:
        return

    count = len(items)
    items.sort(key=lambda x: x['signal'].quality, reverse=True)

    print(f"\n  ◆ {title} ({count})")
    print(f"  {'─' * 85}")

    if section_type in ('triple',):
        print(f"  {'Hisse':<8} {'Fiyat':>8} {'Hacim':>6} {'MFI':>4} {'K':>3} {'Tarih':>6} {'Durum'}")
        print(f"  {'─' * 85}")
        for item in items:
            sig = item['signal']
            d = sig.details
            vol_r = d.get('vol_ratio', 0)
            vol_str = f"x{vol_r:.1f}" if vol_r > 0 else '-'
            mfi_str = '+' if d.get('has_mfi') else '-'
            date_str = _fmt_date(item['signal_date'])
            st = _state_tag(sig)
            print(f"  {item['ticker']:<8} {item['close']:>8.2f} {vol_str:>6} {mfi_str:>4} "
                  f"{sig.quality:>3} {date_str:>6}{st}")

    elif section_type in ('rsi',):
        print(f"  {'Hisse':<8} {'Fiyat':>8} {'Tip':>8} {'RSI':>12} {'K':>3} {'Tarih':>6} {'Durum'}")
        print(f"  {'─' * 85}")
        for item in items:
            sig = item['signal']
            d = sig.details
            tip = 'Klasik' if sig.div_type == 'RSI_CLASSIC' else 'Gizli'
            r1 = d.get('prev_rsi', 0)
            r2 = d.get('curr_rsi', 0)
            date_str = _fmt_date(item['signal_date'])
            st = _state_tag(sig)
            print(f"  {item['ticker']:<8} {item['close']:>8.2f} {tip:>8} "
                  f"{r1:>5.0f}→{r2:<5.0f} {sig.quality:>3} {date_str:>6}{st}")

    elif section_type in ('macd',):
        print(f"  {'Hisse':<8} {'Fiyat':>8} {'Tip':>8} {'Hist1':>9} {'Hist2':>9} {'K':>3} {'Tarih':>6} {'Durum'}")
        print(f"  {'─' * 85}")
        for item in items:
            sig = item['signal']
            d = sig.details
            tip = 'Klasik' if sig.div_type == 'MACD_CLASSIC' else 'Gizli'
            h1 = d.get('prev_hist', 0)
            h2 = d.get('curr_hist', 0)
            date_str = _fmt_date(item['signal_date'])
            st = _state_tag(sig)
            print(f"  {item['ticker']:<8} {item['close']:>8.2f} {tip:>8} "
                  f"{h1:>9.4f} {h2:>9.4f} {sig.quality:>3} {date_str:>6}{st}")

    elif section_type in ('obv',):
        print(f"  {'Hisse':<8} {'Fiyat':>8} {'Tip':>8} {'OBV1':>12} {'OBV2':>12} {'K':>3} {'Tarih':>6}")
        print(f"  {'─' * 85}")
        for item in items:
            sig = item['signal']
            d = sig.details
            tip = 'Klasik' if sig.div_type == 'OBV_CLASSIC' else 'Gizli'
            o1 = d.get('prev_obv', 0)
            o2 = d.get('curr_obv', 0)
            date_str = _fmt_date(item['signal_date'])
            print(f"  {item['ticker']:<8} {item['close']:>8.2f} {tip:>8} "
                  f"{o1:>12.0f} {o2:>12.0f} {sig.quality:>3} {date_str:>6}")

    elif section_type in ('mfi',):
        print(f"  {'Hisse':<8} {'Fiyat':>8} {'Tip':>8} {'MFI':>12} {'K':>3} {'Tarih':>6} {'Durum'}")
        print(f"  {'─' * 85}")
        for item in items:
            sig = item['signal']
            d = sig.details
            tip = 'Klasik' if sig.div_type == 'MFI_CLASSIC' else 'Gizli'
            m1 = d.get('prev_mfi', 0)
            m2 = d.get('curr_mfi', 0)
            date_str = _fmt_date(item['signal_date'])
            st = _state_tag(sig)
            print(f"  {item['ticker']:<8} {item['close']:>8.2f} {tip:>8} "
                  f"{m1:>5.0f}→{m2:<5.0f} {sig.quality:>3} {date_str:>6}{st}")

    elif section_type in ('adx',):
        print(f"  {'Hisse':<8} {'Fiyat':>8} {'ADX':>6} {'Slope':>7} {'F.Slope':>8} {'K':>3} {'Tarih':>6} {'Durum'}")
        print(f"  {'─' * 85}")
        for item in items:
            sig = item['signal']
            d = sig.details
            adx_val = d.get('adx_value', 0)
            adx_sl = d.get('adx_slope', 0)
            cs = d.get('close_slope', 0)
            date_str = _fmt_date(item['signal_date'])
            st = _state_tag(sig)
            print(f"  {item['ticker']:<8} {item['close']:>8.2f} {adx_val:>6.1f} "
                  f"{adx_sl:>7.2f} {cs:>8.3f} {sig.quality:>3} {date_str:>6}{st}")

    elif section_type in ('pv',):
        print(f"  {'Hisse':<8} {'Fiyat':>8} {'Tip':>8} {'VolR':>6} {'R/ATR':>7} {'K':>3} {'Tarih':>6}")
        print(f"  {'─' * 85}")
        for item in items:
            sig = item['signal']
            d = sig.details
            sub = d.get('sub_type', '?')[:6]
            vol_r = d.get('vol_ratio', 0)
            ra = d.get('range_atr', 0)
            lt = d.get('limit_tag')
            date_str = _fmt_date(item['signal_date'])
            vol_str = f"x{vol_r:.1f}"
            tag = f" [{lt}]" if lt else ""
            print(f"  {item['ticker']:<8} {item['close']:>8.2f} {sub:>8} "
                  f"{vol_str:>6} {ra:>7.3f} {sig.quality:>3} {date_str:>6}{tag}")

    print(f"  {'─' * 85}")


def _fmt_date(sig_date):
    """Tarih formatlama."""
    if hasattr(sig_date, 'strftime'):
        return sig_date.strftime('%m-%d')
    return str(sig_date)[-5:]


def _print_results(all_results, n_scanned, date_str, type_filter=None, top_n=None, tf_label=''):
    """Konsol rapor — kural bazli gruplama."""
    w = 85
    _LEGACY_KEYS = ('triple', 'rsi', 'macd', 'obv', 'mfi', 'adx', 'pv')

    print(f"\n{'═' * w}")
    print(f"  NOX UYUMSUZLUK TARAMASI v3{tf_label} — {date_str} — {n_scanned} hisse tarandi")
    print(f"{'═' * w}")

    # Tum sinyalleri topla ve kural siniflandir
    all_classified = []
    n_noise = 0
    for sig_type in _LEGACY_KEYS:
        for item in all_results.get(sig_type, []):
            sig = item['signal']
            rule = _classify_rule(sig)
            if rule is None:
                n_noise += 1
                continue
            all_classified.append({
                'rule': rule,
                'ticker': item['ticker'],
                'close': item['close'],
                'signal': sig,
                'signal_date': item['signal_date'],
                'sig_type': sig_type,
            })

    # Type filtresi
    if type_filter:
        tf = type_filter.upper()
        filter_map = {
            'TRIPLE': ['triple'], 'UCLU': ['triple'], 'RSI': ['rsi'],
            'MACD': ['macd'], 'OBV': ['obv'], 'MFI': ['mfi'],
            'ADX': ['adx'], 'PV': ['pv'], 'PRICE_VOLUME': ['pv'],
            'FH': ['pv'], 'FIYAT': ['pv'],
        }
        allowed = filter_map.get(tf, list(_LEGACY_KEYS))
        all_classified = [i for i in all_classified if i['sig_type'] in allowed]

    # Top N
    if top_n and top_n > 0:
        all_classified.sort(key=lambda x: x['signal'].quality, reverse=True)
        all_classified = all_classified[:top_n]

    # Kural bazli gruplama
    for rule_key in RULE_ORDER:
        rd = RULE_DEFS[rule_key]
        rule_items = [i for i in all_classified if i['rule'] == rule_key]
        if not rule_items:
            continue

        rule_items.sort(key=lambda x: x['signal'].quality, reverse=True)
        st = rd['stats']
        print(f"\n  ◆ KURAL {rule_key}: {rd['name'].upper()} ({len(rule_items)} sinyal)")
        print(f"    BT: 1G WR %{st['1g_wr']:.0f} (+{st['1g_avg']}%) | "
              f"3G WR %{st['3g_wr']:.0f} (+{st['3g_avg']}%) | "
              f"5G WR %{st['5g_wr']:.0f} (+{st['5g_avg']}%) | N={st['n_bt']}")
        print(f"  {'─' * 80}")
        print(f"  {'Hisse':<8} {'Yon':<5} {'Tip':<15} {'Fiyat':>8} {'K':>3} {'RR':>5} {'Tarih':>6}")
        print(f"  {'─' * 80}")

        for item in rule_items:
            sig = item['signal']
            dir_str = 'AL' if sig.direction == 'BUY' else 'SAT'
            tip = TYPE_LABELS.get(sig.div_type, sig.div_type)[:14]
            date_str_fmt = _fmt_date(item['signal_date'])
            rr = getattr(sig, 'rr_ratio', 0) or 0
            rr_str = f"{rr:.1f}" if rr > 0 else '  —'
            mods = _classify_modifiers(sig)
            mod_str = f" {''.join(MOD_DEFS[m]['label'] for m in mods)}" if mods else ''
            print(f"  {item['ticker']:<8} {dir_str:<5} {tip:<15} "
                  f"{item['close']:>8.2f} {sig.quality:>3} {rr_str:>5} {date_str_fmt:>6}{mod_str}")
        print(f"  {'─' * 80}")

    # Ozet
    total = len(all_classified)
    total_buy = sum(1 for i in all_classified if i['signal'].direction == 'BUY')
    total_sell = total - total_buy
    rule_counts = ' | '.join(
        f"{r}:{sum(1 for i in all_classified if i['rule'] == r)}"
        for r in RULE_ORDER
        if sum(1 for i in all_classified if i['rule'] == r) > 0
    )
    print(f"\n{'═' * w}")
    print(f"  OZET: {total} sinyal ({total_buy} AL + {total_sell} SAT) | {rule_counts}")
    print(f"  Elenen: {n_noise} gurultu sinyal")
    print(f"{'═' * w}")


# =============================================================================
# CSV
# =============================================================================

def _save_csv(all_results, date_str, output_dir):
    """Sinyalleri CSV dosyasina kaydet — sadece kural eslesen sinyaller."""
    os.makedirs(output_dir, exist_ok=True)
    rows = []

    for sig_type, items in all_results.items():
        # primary/confirmation/exhaustion key'lerini atla
        if sig_type in ('primary', 'confirmation', 'exhaustion'):
            continue
        for item in items:
            sig = item['signal']

            # Kural siniflandirmasi — gurultuyu kes
            rule = _classify_rule(sig)
            if rule is None:
                continue

            mods = _classify_modifiers(sig)
            sig_date = item['signal_date']

            row = {
                'rule': rule,
                'mods': ','.join(mods) if mods else '',
                'ticker': item['ticker'],
                'close': round(item['close'], 4),
                'div_type': sig.div_type,
                'direction': sig.direction,
                'quality': sig.quality,
                'signal_date': sig_date.strftime('%Y-%m-%d') if hasattr(sig_date, 'strftime') else str(sig_date),
            }

            row['bucket'] = getattr(sig, 'bucket', '')
            row['state'] = getattr(sig, 'state', '')
            row['age'] = getattr(sig, 'age', 0)
            row['trigger_type'] = getattr(sig, 'trigger_type', 'NONE')
            row['structural'] = getattr(sig, 'structural', False)
            row['confirmation_mod'] = getattr(sig, 'confirmation_mod', 0)
            row['location_q'] = getattr(sig, 'location_q', 0)
            row['regime'] = getattr(sig, 'regime', -1)
            row['regime_mod'] = getattr(sig, 'regime_mod', 0)
            row['signal_label'] = getattr(sig, 'signal_label', 'WATCH')
            row['risk_score'] = getattr(sig, 'risk_score', 0)
            row['rr_ratio'] = getattr(sig, 'rr_ratio', 0.0)

            d = sig.details
            if 'prev_rsi' in d:
                row['rsi_prev'] = d.get('prev_rsi')
                row['rsi_curr'] = d.get('curr_rsi')
                row['rsi_diff'] = d.get('rsi_diff')
            if 'prev_hist' in d:
                row['hist_prev'] = d.get('prev_hist')
                row['hist_curr'] = d.get('curr_hist')
            if 'prev_obv' in d:
                row['obv_prev'] = d.get('prev_obv')
                row['obv_curr'] = d.get('curr_obv')
            if 'prev_mfi' in d:
                row['mfi_prev'] = d.get('prev_mfi')
                row['mfi_curr'] = d.get('curr_mfi')
                row['mfi_diff'] = d.get('mfi_diff')
            if 'adx_slope' in d:
                row['adx_value'] = d.get('adx_value')
                row['adx_slope'] = d.get('adx_slope')
            if 'vol_ratio' in d:
                row['vol_ratio'] = d.get('vol_ratio')
            if 'range_atr' in d:
                row['range_atr'] = d.get('range_atr')
            if 'sub_type' in d:
                row['sub_type'] = d.get('sub_type')
            if d.get('limit_tag'):
                row['limit_tag'] = d.get('limit_tag')
            if 'close_slope' in d:
                row['close_slope'] = d.get('close_slope')
            if 'has_mfi' in d:
                row['has_mfi'] = d.get('has_mfi')

            rows.append(row)

    if rows:
        csv_df = pd.DataFrame(rows)
        csv_df.sort_values(['rule', 'quality'], ascending=[True, False], inplace=True)
        fname = f"nox_divergence_{date_str.replace('-', '')}.csv"
        path = os.path.join(output_dir, fname)
        csv_df.to_csv(path, index=False)
        print(f"\n  CSV: {path} ({len(rows)} sinyal)")
    else:
        print(f"\n  Sinyal yok, CSV olusturulmadi.")


# =============================================================================
# HTML RAPOR
# =============================================================================


def _flatten_results(all_results):
    """all_results dict -> flat list of dicts for JSON embed. Kural filtresi uygular."""
    rows = []
    n_noise = 0
    for sig_type, items in all_results.items():
        # primary/confirmation/exhaustion key'lerini atla
        if sig_type in ('primary', 'confirmation', 'exhaustion'):
            continue
        for item in items:
            sig = item['signal']

            # Kural siniflandirmasi
            rule = _classify_rule(sig)
            if rule is None:
                n_noise += 1
                continue  # gurultu, kes

            # Modifier tag'ler
            mods = _classify_modifiers(sig)

            d = sig.details
            sig_date = item['signal_date']

            row = {
                'ticker': item['ticker'],
                'close': round(float(item['close']), 2),
                'div_type': sig.div_type,
                'direction': sig.direction,
                'quality': sig.quality,
                'category': sig_type,
                'rule': rule,
                'mods': mods,
                'signal_date': sig_date.strftime('%Y-%m-%d') if hasattr(sig_date, 'strftime') else str(sig_date),
                # Faz 1 yeni alanlar
                'bucket': getattr(sig, 'bucket', ''),
                'state': getattr(sig, 'state', ''),
                'age': getattr(sig, 'age', 0),
                'trigger_type': getattr(sig, 'trigger_type', 'NONE'),
                'structural': getattr(sig, 'structural', False),
                'confirmation_mod': getattr(sig, 'confirmation_mod', 0),
                # Faz 2
                'location_q': getattr(sig, 'location_q', 0),
                'regime': getattr(sig, 'regime', -1),
                'regime_mod': getattr(sig, 'regime_mod', 0),
                'signal_label': getattr(sig, 'signal_label', 'WATCH'),
                'risk_score': getattr(sig, 'risk_score', 0),
                'rr_ratio': round(getattr(sig, 'rr_ratio', 0.0), 2),
            }

            if 'prev_rsi' in d:
                row['rsi_prev'] = round(float(d['prev_rsi']), 1) if d.get('prev_rsi') is not None else None
                row['rsi_curr'] = round(float(d['curr_rsi']), 1) if d.get('curr_rsi') is not None else None
            if 'prev_hist' in d:
                row['hist_prev'] = round(float(d['prev_hist']), 4) if d.get('prev_hist') is not None else None
                row['hist_curr'] = round(float(d['curr_hist']), 4) if d.get('curr_hist') is not None else None
            if 'prev_obv' in d:
                row['obv_prev'] = float(d['prev_obv']) if d.get('prev_obv') is not None else None
                row['obv_curr'] = float(d['curr_obv']) if d.get('curr_obv') is not None else None
            if 'prev_mfi' in d:
                row['mfi_prev'] = round(float(d['prev_mfi']), 1) if d.get('prev_mfi') is not None else None
                row['mfi_curr'] = round(float(d['curr_mfi']), 1) if d.get('curr_mfi') is not None else None
            if 'adx_slope' in d:
                row['adx_value'] = round(float(d['adx_value']), 1) if d.get('adx_value') is not None else None
                row['adx_slope'] = round(float(d['adx_slope']), 2) if d.get('adx_slope') is not None else None
            if 'vol_ratio' in d:
                row['vol_ratio'] = round(float(d['vol_ratio']), 1) if d.get('vol_ratio') is not None else None
            if 'range_atr' in d:
                row['range_atr'] = round(float(d['range_atr']), 3) if d.get('range_atr') is not None else None
            if 'sub_type' in d:
                row['sub_type'] = d.get('sub_type')
            if d.get('limit_tag'):
                row['limit_tag'] = d.get('limit_tag')
            if d.get('has_mfi'):
                row['has_mfi'] = True
            if 'close_slope' in d:
                row['close_slope'] = round(float(d['close_slope']), 3) if d.get('close_slope') is not None else None

            # ── Viability (freshness) hesapla ──
            signal_close = item.get('signal_close', item['close'])
            current_close = item['close']
            bars_ago = item.get('bars_ago', 0)

            if signal_close and signal_close > 0:
                if row['direction'] == 'BUY':
                    move_pct = (current_close - signal_close) / signal_close * 100
                else:
                    move_pct = (signal_close - current_close) / signal_close * 100
            else:
                move_pct = 0.0

            if bars_ago == 0:
                viability = 'TAZE'
            elif move_pct < -5:
                viability = 'KIRILMIS'
            elif move_pct >= 3:
                viability = 'GEC'
            else:
                viability = 'GIRILEBILIR'

            row['signal_close'] = round(signal_close, 2)
            row['move_pct'] = round(move_pct, 2)
            row['bars_ago'] = bars_ago
            row['viability'] = viability

            rows.append(row)

    if n_noise > 0:
        print(f"  Kural filtresi: {n_noise} gurultu sinyal elendi, {len(rows)} sinyal kaldi.")
    return rows


def _generate_html(all_results, n_scanned, date_str, tf_label=''):
    """Divergence HTML raporu olustur — kural bazli."""
    now = datetime.now().strftime('%d.%m.%Y %H:%M')
    flat = _flatten_results(all_results)
    rows_json = json.dumps(_sanitize(flat), ensure_ascii=False)

    tf_tag = tf_label.strip() if tf_label.strip() else ''
    title_suffix = f' · {tf_tag}' if tf_tag else ''

    dir_labels_json = json.dumps({'BUY': 'AL', 'SELL': 'SAT'})
    type_labels_json = json.dumps(TYPE_LABELS)
    rule_defs_json = json.dumps(RULE_DEFS, ensure_ascii=False)
    rule_order_json = json.dumps(RULE_ORDER)

    html = f"""<!DOCTYPE html>
<html lang="tr"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NOX — Divergence{title_suffix} · {date_str}</title>
<style>{_NOX_CSS}
.tab-bar {{
  display: flex; gap: 4px; flex-wrap: wrap; margin-bottom: 16px;
}}
.tab-btn {{
  display: flex; align-items: center; gap: 6px;
  background: var(--bg-card); border: 1px solid var(--border-subtle);
  border-radius: 20px; padding: 6px 14px;
  font-size: 0.78rem; font-weight: 500;
  cursor: pointer; transition: all 0.2s;
  user-select: none; color: var(--text-secondary);
  font-family: var(--font-display);
}}
.tab-btn:hover {{ border-color: var(--border-dim); background: var(--bg-elevated); }}
.tab-btn.active {{ border-color: var(--nox-cyan); background: var(--nox-cyan-dim); color: var(--nox-cyan); }}
.tab-btn .cnt {{ font-family: var(--font-mono); font-weight: 700; font-size: 0.85rem; }}
.tab-btn .dot {{ width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }}
.dir-buy {{ color: var(--nox-green); font-weight: 700; }}
.dir-sell {{ color: var(--nox-red); font-weight: 700; }}
.q-hi {{ color: var(--nox-green); }}
.q-mid {{ color: var(--nox-yellow); }}
.q-lo {{ color: var(--text-muted); }}
.badge {{
  display: inline-block; padding: 1px 6px; border-radius: 3px;
  font-size: 0.65rem; font-weight: 700; letter-spacing: 0.3px;
}}
.rr-hi {{ color: var(--nox-green); font-weight: 700; }}
.rr-mid {{ color: var(--nox-yellow); }}
.rr-lo {{ color: var(--nox-red); }}
/* Kural kartlari */
.rule-cards {{
  display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 16px;
}}
.r-card {{
  background: var(--bg-card); border: 1px solid var(--border-subtle);
  border-radius: 10px; padding: 14px 16px; min-width: 170px; flex: 1;
  border-top: 3px solid var(--border-subtle);
}}
.r-card .r-head {{
  display: flex; align-items: center; justify-content: space-between;
  margin-bottom: 8px;
}}
.r-card .r-title {{
  font-family: var(--font-display); font-size: 0.82rem; font-weight: 700;
}}
.r-card .r-count {{
  font-family: var(--font-mono); font-size: 1.3rem; font-weight: 800;
}}
.r-card .r-stats {{
  font-size: 0.68rem; color: var(--text-secondary);
  font-family: var(--font-mono); line-height: 1.6;
}}
.r-card .r-desc {{
  font-size: 0.65rem; color: var(--text-muted); margin-top: 6px;
  line-height: 1.3;
}}
.r-wr {{ font-weight: 700; }}
.r-avg {{ color: var(--nox-green); }}
/* Kural badge */
.rule-badge {{
  display: inline-block; padding: 2px 8px; border-radius: 4px;
  font-size: 0.7rem; font-weight: 800; letter-spacing: 0.5px;
  font-family: var(--font-mono);
}}
/* Viability badges */
.v-taze {{ color: var(--nox-cyan); font-weight: 700; }}
.v-ok   {{ color: var(--nox-green); font-weight: 700; }}
.v-gec  {{ color: var(--nox-yellow); font-weight: 700; }}
.v-brok {{ color: var(--nox-red); font-weight: 700; }}
</style>
</head><body>
<div class="nox-container">
<div class="nox-header">
  <div class="nox-logo">NOX<span class="proj">project</span><span class="mode">divergence{title_suffix}</span></div>
  <div class="nox-meta"><b>{len(flat)}</b> sinyal / {n_scanned} taranan<br>{now}</div>
</div>
<!-- Nasil Okunur -->
<details style="margin-bottom:16px;background:var(--bg-card);border:1px solid var(--border-subtle);border-radius:10px;padding:12px 16px;font-size:.78rem;color:var(--text-secondary);cursor:pointer;">
<summary style="font-weight:700;color:var(--nox-cyan);font-size:.82rem;cursor:pointer;user-select:none;">NASIL OKUNUR?</summary>
<div style="margin-top:10px;line-height:1.7;">
<b style="color:var(--text-primary);">Kural Bazli Filtreleme</b> — 56.389 sinyalden 2 yillik backtestle belirlenmis 4 kural ile yalnizca edge veren ~%2 sinyal gosterilir. Kesif/dogrulama ayrimiyla dogrulanmis kombinasyonlar.<br><br>
<b style="color:var(--nox-red);">B: Kisa Setup</b> — ENTRY_SHORT + SETUP + RR&ge;1.5. Sat yonlu kurulum sinyalleri. <b>5G WR %76, Avg +1.66%</b>.<br>
<b style="color:var(--nox-orange);">D: MACD Premium</b> — MACD_HIDDEN + RR&ge;2.0 + Yapisal. Dogrulanmis premium AL katmani. <b>5G WR %61, Avg +2.27%</b>. Dogrulama: %62.5.<br>
<b style="color:var(--nox-green);">C: Yapisal Gizli</b> — ENTRY_LONG + yapisal swing + gizli uyumsuzluk. Ana AL motoru. <b>5G WR %59, Avg +1.50%</b>. Dogrulama: %59.2. En buyuk N.<br>
<b style="color:var(--nox-blue);">E: Cover</b> — COVER + STALE + RR&ge;1.0. Taktik pozisyon, dusuk agirlik. <b>5G WR %55, Avg +1.93%</b>.<br><br>
<b style="color:var(--nox-cyan);">Modifier Etiketler:</b><br>
&bull; <span style="color:var(--nox-green);">&uarr;FT</span> = FULL_TREND rejim (guclendirici) &bull; <span style="color:var(--nox-cyan);">&uarr;RR</span> = RR&ge;2.0 &bull; <span style="color:var(--nox-cyan);">&uarr;RSI</span> = RSI_HIDDEN (en tutarli div type)<br>
&bull; <span style="color:var(--nox-yellow);">&rlhar;K60</span> = K&ge;60 dikkat — dogrulamada kirilgan, kucuk N kombinasyonlarda supheli<br><br>
<b style="color:var(--nox-orange);">Kalite (K) ve RR:</b><br>
&bull; <b>K</b>: 4 bilesen (Yapi + Momentum + Katilim + Konum, 0-25 her biri) + teyit bonusu. K&ge;60 <span class="q-hi">yesil</span>, K&ge;40 <span class="q-mid">sari</span>, K&lt;40 <span class="q-lo">gri</span>.<br>
&bull; <b>RR</b>: Risk/Odul orani. RR&ge;2 <span class="rr-hi">yesil</span>, RR&ge;1 <span class="rr-mid">sari</span>, RR&lt;1 <span class="rr-lo">kirmizi</span>.
</div>
</details>
<!-- Backtest Sonuclari -->
<details style="margin-bottom:16px;background:var(--bg-card);border:1px solid var(--border-subtle);border-radius:10px;padding:12px 16px;font-size:.78rem;color:var(--text-secondary);cursor:pointer;">
<summary style="font-weight:700;color:var(--nox-orange);font-size:.82rem;cursor:pointer;user-select:none;">BACKTEST SONUCLARI (2y, 100K TL)</summary>
<div style="margin-top:10px;line-height:1.8;">

<b style="color:var(--text-primary);font-size:.85rem;">Portfolio Backtest &mdash; 2 Yil, 100K TL, 1138 Sinyal</b><br>
<span style="color:var(--text-muted);font-size:.7rem;">Komisyon: %0.2 &bull; Max 10 es zamanli &bull; Pozisyon: %10 sermaye</span><br><br>

<b style="color:var(--nox-cyan);">Strateji Karsilastirma</b>
<table style="width:100%;border-collapse:collapse;font-family:var(--font-mono);font-size:.72rem;margin:8px 0 14px;">
<thead><tr style="border-bottom:1px solid var(--border-subtle);color:var(--text-muted);">
<th style="text-align:left;padding:4px 8px;">Strateji</th>
<th style="text-align:right;padding:4px 6px;">N</th>
<th style="text-align:right;padding:4px 6px;">WR</th>
<th style="text-align:right;padding:4px 6px;">PF</th>
<th style="text-align:right;padding:4px 6px;">Sharpe</th>
<th style="text-align:right;padding:4px 6px;">Equity</th>
<th style="text-align:right;padding:4px 6px;">DD</th>
</tr></thead>
<tbody>
<tr style="border-bottom:1px solid var(--border-subtle);">
<td style="padding:4px 8px;color:var(--text-muted);">FIXED</td>
<td style="text-align:right;padding:4px 6px;">579</td>
<td style="text-align:right;padding:4px 6px;">54.6%</td>
<td style="text-align:right;padding:4px 6px;">1.44</td>
<td style="text-align:right;padding:4px 6px;">0.79</td>
<td style="text-align:right;padding:4px 6px;">164K</td>
<td style="text-align:right;padding:4px 6px;color:var(--nox-red);">-20.1%</td>
</tr>
<tr style="border-bottom:1px solid var(--border-subtle);background:rgba(34,211,238,0.06);">
<td style="padding:4px 8px;color:var(--nox-cyan);font-weight:700;">TRAILING</td>
<td style="text-align:right;padding:4px 6px;">601</td>
<td style="text-align:right;padding:4px 6px;">54.9%</td>
<td style="text-align:right;padding:4px 6px;color:var(--nox-green);">1.73</td>
<td style="text-align:right;padding:4px 6px;color:var(--nox-green);">1.34</td>
<td style="text-align:right;padding:4px 6px;color:var(--nox-green);">202K</td>
<td style="text-align:right;padding:4px 6px;">-16.8%</td>
</tr>
<tr>
<td style="padding:4px 8px;color:var(--text-muted);">PARTIAL</td>
<td style="text-align:right;padding:4px 6px;">654</td>
<td style="text-align:right;padding:4px 6px;">52.0%</td>
<td style="text-align:right;padding:4px 6px;">1.34</td>
<td style="text-align:right;padding:4px 6px;">0.72</td>
<td style="text-align:right;padding:4px 6px;">170K</td>
<td style="text-align:right;padding:4px 6px;color:var(--nox-red);">-18.4%</td>
</tr>
</tbody></table>

<b style="color:var(--nox-cyan);">Kural Bazli Kirilim (TRAILING)</b>
<table style="width:100%;border-collapse:collapse;font-family:var(--font-mono);font-size:.72rem;margin:8px 0 14px;">
<thead><tr style="border-bottom:1px solid var(--border-subtle);color:var(--text-muted);">
<th style="text-align:left;padding:4px 8px;">Kural</th>
<th style="text-align:right;padding:4px 6px;">N</th>
<th style="text-align:right;padding:4px 6px;">WR</th>
<th style="text-align:right;padding:4px 6px;">PF</th>
<th style="text-align:right;padding:4px 6px;">PnL TL</th>
<th style="text-align:right;padding:4px 6px;">Trail</th>
<th style="text-align:right;padding:4px 6px;">Hold</th>
</tr></thead>
<tbody>
<tr style="border-bottom:1px solid var(--border-subtle);">
<td style="padding:4px 8px;color:var(--nox-red);font-weight:700;">B Kisa Setup</td>
<td style="text-align:right;padding:4px 6px;">57</td>
<td style="text-align:right;padding:4px 6px;">61.4%</td>
<td style="text-align:right;padding:4px 6px;">1.50</td>
<td style="text-align:right;padding:4px 6px;color:var(--nox-green);">+7,639</td>
<td style="text-align:right;padding:4px 6px;">4%/4G</td>
<td style="text-align:right;padding:4px 6px;">5G</td>
</tr>
<tr style="border-bottom:1px solid var(--border-subtle);">
<td style="padding:4px 8px;color:var(--nox-green);font-weight:700;">C Yapisal</td>
<td style="text-align:right;padding:4px 6px;">175</td>
<td style="text-align:right;padding:4px 6px;">48.0%</td>
<td style="text-align:right;padding:4px 6px;color:var(--nox-yellow);">0.94</td>
<td style="text-align:right;padding:4px 6px;color:var(--nox-red);">-6,552</td>
<td style="text-align:right;padding:4px 6px;">4%/3G</td>
<td style="text-align:right;padding:4px 6px;">10G</td>
</tr>
<tr style="border-bottom:1px solid var(--border-subtle);background:rgba(74,222,128,0.06);">
<td style="padding:4px 8px;color:var(--nox-orange);font-weight:700;">D AL (MACD)</td>
<td style="text-align:right;padding:4px 6px;">229</td>
<td style="text-align:right;padding:4px 6px;color:var(--nox-green);">58.1%</td>
<td style="text-align:right;padding:4px 6px;color:var(--nox-green);">2.27</td>
<td style="text-align:right;padding:4px 6px;color:var(--nox-green);">+79,685</td>
<td style="text-align:right;padding:4px 6px;">4%/3G</td>
<td style="text-align:right;padding:4px 6px;">10G</td>
</tr>
<tr style="border-bottom:1px solid var(--border-subtle);">
<td style="padding:4px 8px;color:var(--nox-orange);font-weight:700;">D SAT (MACD)</td>
<td style="text-align:right;padding:4px 6px;">147</td>
<td style="text-align:right;padding:4px 6px;">54.4%</td>
<td style="text-align:right;padding:4px 6px;">1.34</td>
<td style="text-align:right;padding:4px 6px;color:var(--nox-green);">+11,346</td>
<td style="text-align:right;padding:4px 6px;">2.5%/2G</td>
<td style="text-align:right;padding:4px 6px;">5G</td>
</tr>
<tr>
<td style="padding:4px 8px;color:var(--nox-blue);font-weight:700;">E Cover</td>
<td style="text-align:right;padding:4px 6px;">67</td>
<td style="text-align:right;padding:4px 6px;">46.3%</td>
<td style="text-align:right;padding:4px 6px;">1.63</td>
<td style="text-align:right;padding:4px 6px;color:var(--nox-green);">+14,517</td>
<td style="text-align:right;padding:4px 6px;">3%/2G</td>
<td style="text-align:right;padding:4px 6px;">5G</td>
</tr>
</tbody></table>

<details style="margin-top:8px;padding:8px 12px;border-radius:6px;background:rgba(34,211,238,0.04);border:1px solid rgba(34,211,238,0.12);">
<summary style="cursor:pointer;font-weight:700;color:var(--nox-cyan);font-size:.76rem;">C DARALTMA OPTIMIZASYONU</summary>
<div style="margin-top:8px;line-height:1.7;">
<span style="color:var(--text-muted);">C kurali 511 sinyalin %58&apos;i MFI_HIDDEN (WR %55.6 &mdash; zayif). 3 seviyeli filtre testi:</span>
<table style="width:100%;border-collapse:collapse;font-family:var(--font-mono);font-size:.7rem;margin:6px 0;">
<thead><tr style="border-bottom:1px solid var(--border-subtle);color:var(--text-muted);">
<th style="text-align:left;padding:3px 6px;">Filtre</th><th style="text-align:right;padding:3px 6px;">N</th>
<th style="text-align:right;padding:3px 6px;">WR</th><th style="text-align:right;padding:3px 6px;">PF</th>
<th style="text-align:right;padding:3px 6px;">Sharpe</th><th style="text-align:right;padding:3px 6px;">Equity</th>
<th style="text-align:right;padding:3px 6px;">C WR</th>
</tr></thead>
<tbody>
<tr style="border-bottom:1px solid var(--border-subtle);"><td style="padding:3px 6px;">Yok</td><td style="text-align:right;padding:3px 6px;">579</td><td style="text-align:right;padding:3px 6px;">54.6%</td><td style="text-align:right;padding:3px 6px;">1.44</td><td style="text-align:right;padding:3px 6px;">0.79</td><td style="text-align:right;padding:3px 6px;">163K</td><td style="text-align:right;padding:3px 6px;">48.0%</td></tr>
<tr style="border-bottom:1px solid var(--border-subtle);"><td style="padding:3px 6px;">Moderate</td><td style="text-align:right;padding:3px 6px;">547</td><td style="text-align:right;padding:3px 6px;">56.5%</td><td style="text-align:right;padding:3px 6px;">1.54</td><td style="text-align:right;padding:3px 6px;">0.92</td><td style="text-align:right;padding:3px 6px;">175K</td><td style="text-align:right;padding:3px 6px;">51.6%</td></tr>
<tr style="background:rgba(74,222,128,0.08);"><td style="padding:3px 6px;font-weight:700;color:var(--nox-green);">Aggressive</td><td style="text-align:right;padding:3px 6px;">514</td><td style="text-align:right;padding:3px 6px;color:var(--nox-green);">58.9%</td><td style="text-align:right;padding:3px 6px;color:var(--nox-green);">1.69</td><td style="text-align:right;padding:3px 6px;color:var(--nox-green);">1.08</td><td style="text-align:right;padding:3px 6px;color:var(--nox-green);">180K</td><td style="text-align:right;padding:3px 6px;color:var(--nox-green);">59.0%</td></tr>
</tbody></table>
<span style="color:var(--nox-green);">&bull;</span> Aggressive: MFI_HIDDEN + OBV_HIDDEN cikarilir &rarr; WR +4.3pp, PF +0.25, Sharpe +0.29
</div>
</details>

<details style="margin-top:8px;padding:8px 12px;border-radius:6px;background:rgba(251,146,60,0.04);border:1px solid rgba(251,146,60,0.12);">
<summary style="cursor:pointer;font-weight:700;color:var(--nox-orange);font-size:.76rem;">TRAIL PARAMETRE OPTIMIZASYONU</summary>
<div style="margin-top:8px;line-height:1.7;">
<span style="color:var(--text-muted);">Kural bazli trailing stop sweep (aggressive C filtre ile):</span>
<table style="width:100%;border-collapse:collapse;font-family:var(--font-mono);font-size:.7rem;margin:6px 0;">
<thead><tr style="border-bottom:1px solid var(--border-subtle);color:var(--text-muted);">
<th style="text-align:left;padding:3px 6px;">Kural</th>
<th style="text-align:right;padding:3px 6px;">Trail</th><th style="text-align:right;padding:3px 6px;">Start</th>
<th style="text-align:right;padding:3px 6px;">WR</th><th style="text-align:right;padding:3px 6px;">PF</th>
<th style="text-align:right;padding:3px 6px;">Sharpe</th>
</tr></thead>
<tbody>
<tr style="border-bottom:1px solid var(--border-subtle);"><td style="padding:3px 6px;color:var(--nox-red);">B</td><td style="text-align:right;padding:3px 6px;">4.0%</td><td style="text-align:right;padding:3px 6px;">4G</td><td style="text-align:right;padding:3px 6px;">55.2%</td><td style="text-align:right;padding:3px 6px;">1.75</td><td style="text-align:right;padding:3px 6px;">1.36</td></tr>
<tr style="border-bottom:1px solid var(--border-subtle);"><td style="padding:3px 6px;color:var(--nox-green);">C</td><td style="text-align:right;padding:3px 6px;">4.0%</td><td style="text-align:right;padding:3px 6px;">3G</td><td style="text-align:right;padding:3px 6px;">55.0%</td><td style="text-align:right;padding:3px 6px;">1.76</td><td style="text-align:right;padding:3px 6px;">1.37</td></tr>
<tr style="border-bottom:1px solid var(--border-subtle);background:rgba(74,222,128,0.06);"><td style="padding:3px 6px;color:var(--nox-orange);font-weight:700;">D_BUY</td><td style="text-align:right;padding:3px 6px;">4.0%</td><td style="text-align:right;padding:3px 6px;">3G</td><td style="text-align:right;padding:3px 6px;color:var(--nox-green);">55.6%</td><td style="text-align:right;padding:3px 6px;color:var(--nox-green);">1.80</td><td style="text-align:right;padding:3px 6px;color:var(--nox-green);">1.38</td></tr>
<tr style="border-bottom:1px solid var(--border-subtle);"><td style="padding:3px 6px;color:var(--nox-orange);">D_SELL</td><td style="text-align:right;padding:3px 6px;">2.5%</td><td style="text-align:right;padding:3px 6px;">2G</td><td style="text-align:right;padding:3px 6px;">54.9%</td><td style="text-align:right;padding:3px 6px;">1.73</td><td style="text-align:right;padding:3px 6px;">1.34</td></tr>
<tr><td style="padding:3px 6px;color:var(--nox-blue);">E</td><td style="text-align:right;padding:3px 6px;">3.0%</td><td style="text-align:right;padding:3px 6px;">2G</td><td style="text-align:right;padding:3px 6px;">54.8%</td><td style="text-align:right;padding:3px 6px;">1.74</td><td style="text-align:right;padding:3px 6px;">1.35</td></tr>
</tbody></table>

<b style="color:var(--nox-cyan);">Combined Optimized vs Baseline:</b>
<table style="width:100%;border-collapse:collapse;font-family:var(--font-mono);font-size:.7rem;margin:6px 0;">
<thead><tr style="border-bottom:1px solid var(--border-subtle);color:var(--text-muted);">
<th style="text-align:left;padding:3px 6px;"></th><th style="text-align:right;padding:3px 6px;">Baseline</th>
<th style="text-align:right;padding:3px 6px;">Optimized</th><th style="text-align:right;padding:3px 6px;">Delta</th>
</tr></thead>
<tbody>
<tr style="border-bottom:1px solid var(--border-subtle);"><td style="padding:3px 6px;">WR</td><td style="text-align:right;padding:3px 6px;">54.9%</td><td style="text-align:right;padding:3px 6px;color:var(--nox-green);">55.8%</td><td style="text-align:right;padding:3px 6px;color:var(--nox-green);">+0.9%</td></tr>
<tr style="border-bottom:1px solid var(--border-subtle);"><td style="padding:3px 6px;">PF</td><td style="text-align:right;padding:3px 6px;">1.73</td><td style="text-align:right;padding:3px 6px;color:var(--nox-green);">1.87</td><td style="text-align:right;padding:3px 6px;color:var(--nox-green);">+0.14</td></tr>
<tr style="border-bottom:1px solid var(--border-subtle);"><td style="padding:3px 6px;">Sharpe</td><td style="text-align:right;padding:3px 6px;">1.34</td><td style="text-align:right;padding:3px 6px;color:var(--nox-green);">1.46</td><td style="text-align:right;padding:3px 6px;color:var(--nox-green);">+0.12</td></tr>
<tr style="border-bottom:1px solid var(--border-subtle);"><td style="padding:3px 6px;">PnL TL</td><td style="text-align:right;padding:3px 6px;">126,793</td><td style="text-align:right;padding:3px 6px;color:var(--nox-green);">145,234</td><td style="text-align:right;padding:3px 6px;color:var(--nox-green);">+18,441</td></tr>
<tr><td style="padding:3px 6px;">Max DD</td><td style="text-align:right;padding:3px 6px;">-16.8%</td><td style="text-align:right;padding:3px 6px;color:var(--nox-green);">-16.5%</td><td style="text-align:right;padding:3px 6px;color:var(--nox-green);">+0.3%</td></tr>
</tbody></table>
<span style="color:var(--nox-orange);">&bull;</span> Ana bulgu: Tum kurallar 4% trail&apos;e yoneliyor &mdash; mevcut 3% erken kesiyor. D_BUY en guclu motor (PF 2.40).
</div>
</details>

<details style="margin-top:8px;padding:8px 12px;border-radius:6px;background:rgba(96,165,250,0.04);border:1px solid rgba(96,165,250,0.12);">
<summary style="cursor:pointer;font-weight:700;color:var(--nox-blue);font-size:.76rem;">EXECUTION REHBERI</summary>
<div style="margin-top:8px;line-height:1.7;">
<b style="color:var(--nox-red);">B (Kisa Setup):</b> 5G hedef. Trail 4%/4G. SAT yonlu &mdash; WR %61-76.<br>
<b style="color:var(--nox-orange);">D BUY (MACD Premium):</b> 7-10G hold. Trail 4%/3G. <span style="color:var(--nox-green);">En guclu motor</span> &mdash; PF 2.27, MFE/MAE 5.66x. K40-59 sweet spot (%67.4 WR).<br>
<b style="color:var(--nox-orange);">D SELL (MACD Premium):</b> 3-4G max 5G. Trail 2.5%/2G. Hizli &mdash; erken cikis.<br>
<b style="color:var(--nox-green);">C (Yapisal Gizli):</b> 5G+runner. Trail 4%/3G. MFI_HIDDEN cikar, RSI/MACD_HIDDEN tut. C zayif &mdash; dusuk agirlik.<br>
<b style="color:var(--nox-blue);">E (Cover):</b> 4-5G max 5G. Trail 3%/2G. Taktik pozisyon &mdash; dusuk agirlik.<br><br>
<span style="color:var(--text-muted);">&bull; Hicbir kural hizli degil &mdash; erken cikis 7-17pp WR kaybi. &bull; Trail cikislari WR %43 vs max_hold WR %81.</span>
</div>
</details>

</div>
</details>
<!-- Kural Kartlari -->
<div class="rule-cards" id="ruleCards"></div>
<!-- Tab Bar -->
<div class="tab-bar" id="tabs"></div>
<!-- Filtreler -->
<div class="nox-filters">
  <div><label>Yon</label>
  <select id="fDir" onchange="af()"><option value="">Tumu</option>
  <option value="BUY">AL</option><option value="SELL">SAT</option></select></div>
  <div><label>K&ge;</label><input type="number" id="fQ" value="0" step="10" min="0" oninput="af()"></div>
  <div><label>Hisse</label><input type="text" id="fS" placeholder="ARA" oninput="af()"></div>
  <div><button class="nox-btn" onclick="reset()">Sifirla</button></div>
</div>
<div class="nox-table-wrap">
<table><thead><tr>
<th onclick="sb('rule')">Kural</th>
<th onclick="sb('ticker')">Hisse</th>
<th onclick="sb('direction')">Yon</th>
<th onclick="sb('div_type')">Tip</th>
<th onclick="sb('close')">Fiyat</th>
<th onclick="sb('quality')">K</th>
<th onclick="sb('rr_ratio')">RR</th>
<th>Detay</th>
<th onclick="sb('signal_date')">Tarih</th>
<th onclick="sb('viability')">Durum</th>
</tr></thead><tbody id="tb"></tbody></table>
</div>
<div class="nox-status" id="st"><b>{len(flat)}</b> / {len(flat)}</div>
</div>
<script>
const D={rows_json};
const DL={dir_labels_json};
const TL={type_labels_json};
const RD={rule_defs_json};
const RO={rule_order_json};
const RP={{'B':1,'D':2,'C':3,'E':4}};
const MD={json.dumps(MOD_DEFS, ensure_ascii=False)};
let curTab='all',col='_rule',asc=true;

/* Siralama: kural onceligi (A>B>C>D>E), esitse kalite desc */
function ruleSort(a,b){{
  const pa=RP[a.rule]||9, pb=RP[b.rule]||9;
  if(pa!==pb) return pa-pb;
  const va=VP[a.viability]||9, vb=VP[b.viability]||9;
  if(va!==vb) return va-vb;
  return (b.quality||0)-(a.quality||0);
}}

/* Kural kartlarini doldur */
function initRuleCards(){{
  const box=document.getElementById('ruleCards');
  RO.forEach(r=>{{
    const rd=RD[r];
    if(!rd) return;
    const cnt=D.filter(d=>d.rule===r).length;
    const c=document.createElement('div');
    c.className='r-card';
    c.style.borderTopColor=rd.color;
    const st=rd.stats;
    c.innerHTML=`<div class="r-head"><span class="r-title" style="color:${{rd.color}}">${{r}}: ${{rd.name}}</span><span class="r-count" style="color:${{rd.color}}">${{cnt}}</span></div>`+
      `<div class="r-stats"><span class="r-wr">1G %${{st['1g_wr']}} &middot; 3G %${{st['3g_wr']}} &middot; 5G %${{st['5g_wr']}}</span><br>`+
      `<span class="r-avg">1G +${{st['1g_avg']}}% &middot; 3G +${{st['3g_avg']}}% &middot; 5G +${{st['5g_avg']}}%</span><br>`+
      `<span style="color:var(--text-muted)">BT: ${{st.n_bt}} sinyal</span></div>`+
      `<div class="r-desc">${{rd.desc}}</div>`;
    box.appendChild(c);
  }});
}}

function initTabs(){{
  const bar=document.getElementById('tabs');
  /* Tumu tab */
  const allBtn=document.createElement('div');
  allBtn.className='tab-btn active';allBtn.dataset.t='all';
  allBtn.innerHTML='<span class="dot" style="background:var(--text-muted)"></span>Tumu <span class="cnt">'+D.length+'</span>';
  allBtn.onclick=()=>setTab('all');
  bar.appendChild(allBtn);
  /* Girilebilir tab */
  const viaCnt=D.filter(d=>d.viability==='TAZE'||d.viability==='GIRILEBILIR').length;
  const viaBtn=document.createElement('div');
  viaBtn.className='tab-btn';viaBtn.dataset.t='viable';
  viaBtn.innerHTML='<span class="dot" style="background:var(--nox-green)"></span>\u2713 Girilebilir <span class="cnt">'+viaCnt+'</span>';
  viaBtn.onclick=()=>setTab('viable');
  bar.appendChild(viaBtn);
  /* Kural tab'lari */
  RO.forEach(r=>{{
    const rd=RD[r];
    if(!rd) return;
    const cnt=D.filter(d=>d.rule===r).length;
    if(!cnt) return;
    const btn=document.createElement('div');
    btn.className='tab-btn';btn.dataset.t=r;
    btn.innerHTML='<span class="dot" style="background:'+rd.color+'"></span>'+r+': '+rd.name+' <span class="cnt">'+cnt+'</span>';
    btn.onclick=()=>setTab(r);
    bar.appendChild(btn);
  }});
}}

function setTab(t){{
  curTab=t;
  document.querySelectorAll('.tab-btn').forEach(b=>b.classList.remove('active'));
  document.querySelector('.tab-btn[data-t="'+t+'"]').classList.add('active');
  col='_rule'; asc=true;
  af();
}}

function af(){{
  const dir=document.getElementById('fDir').value;
  const minQ=parseInt(document.getElementById('fQ').value)||0;
  const sr=document.getElementById('fS').value.toUpperCase();
  let f=D.filter(r=>{{
    if(curTab==='viable'){{ if(r.viability!=='TAZE'&&r.viability!=='GIRILEBILIR') return false; }}
    else if(curTab!=='all' && r.rule!==curTab) return false;
    if(dir&&r.direction!==dir) return false;
    if(r.quality<minQ) return false;
    if(sr&&!r.ticker.includes(sr)) return false;
    return true;
  }});
  if(col==='_rule'){{
    f.sort(ruleSort);
  }} else if(col==='viability'){{
    f.sort((a,b)=>{{
      const va=VP[a.viability]||9, vb=VP[b.viability]||9;
      return asc?va-vb:vb-va;
    }});
  }} else {{
    f.sort((a,b)=>{{
      let va=a[col],vb=b[col];
      if(typeof va==='string') return asc?va.localeCompare(vb):vb.localeCompare(va);
      return asc?(va||0)-(vb||0):(vb||0)-(va||0);
    }});
  }}
  render(f);
}}

function sb(c){{
  if(c==='_rule')return;
  if(col===c)asc=!asc;else{{col=c;asc=c==='ticker'||c==='signal_date'}};
  af();
}}

function reset(){{
  curTab='all'; col='_rule'; asc=true;
  document.querySelectorAll('.tab-btn').forEach(b=>b.classList.remove('active'));
  document.querySelector('.tab-btn[data-t="all"]').classList.add('active');
  document.getElementById('fDir').value='';
  document.getElementById('fQ').value='0';
  document.getElementById('fS').value='';
  af();
}}

function mkDetail(r){{
  const parts=[];
  if(r.rsi_prev!=null)parts.push('RSI: '+r.rsi_prev+'→'+r.rsi_curr);
  if(r.hist_prev!=null)parts.push('Hist: '+r.hist_prev+'→'+r.hist_curr);
  if(r.obv_prev!=null)parts.push('OBV: '+Math.round(r.obv_prev)+'→'+Math.round(r.obv_curr));
  if(r.mfi_prev!=null)parts.push('MFI: '+r.mfi_prev+'→'+r.mfi_curr);
  if(r.adx_value!=null)parts.push('ADX: '+r.adx_value+' s='+r.adx_slope);
  if(r.sub_type)parts.push(r.sub_type);
  if(r.vol_ratio!=null)parts.push('VolR: x'+r.vol_ratio);
  if(r.range_atr!=null)parts.push('R/ATR: '+r.range_atr);
  if(r.limit_tag)parts.push('['+r.limit_tag+']');
  if(r.has_mfi)parts.push('MFI+');
  if(r.structural)parts.push('S');
  return parts.length?'<span class="detail-cell">'+parts.join(', ')+'</span>':'<span style="color:var(--text-muted)">—</span>';
}}

function mkRule(rule){{
  const rd=RD[rule];
  if(!rd)return rule||'—';
  return '<span class="rule-badge" style="background:color-mix(in srgb,'+rd.color+' 20%,transparent);color:'+rd.color+'">'+rule+'</span>';
}}

function mkMods(mods){{
  if(!mods||!mods.length) return '';
  return ' '+mods.map(m=>{{
    const md=MD[m];
    if(!md) return m;
    return '<span style="font-size:.6rem;font-weight:700;color:'+md.color+';padding:0 2px" title="'+md.desc+'">'+md.label+'</span>';
  }}).join('');
}}

function mkRR(rr){{
  if(!rr||rr<=0)return '<span style="color:var(--text-muted)">—</span>';
  const cls=rr>=2?'rr-hi':rr>=1?'rr-mid':'rr-lo';
  return '<span class="'+cls+'">'+rr.toFixed(1)+'</span>';
}}

const VP={{'TAZE':1,'GIRILEBILIR':2,'GEC':3,'KIRILMIS':4}};
function mkViability(r){{
  const ba=r.bars_ago||0;
  const mp=r.move_pct||0;
  const v=r.viability||'TAZE';
  if(v==='TAZE') return '<span class="v-taze">\u25cf TAZE</span>';
  if(v==='GIRILEBILIR') return `<span class="v-ok">\u2713 ${{mp>0?'+':''}}${{mp.toFixed(1)}}% ${{ba}}G</span>`;
  if(v==='GEC') return `<span class="v-gec">\u26a0 ${{mp>0?'+':''}}${{mp.toFixed(1)}}% ${{ba}}G</span>`;
  if(v==='KIRILMIS') return `<span class="v-brok">\u2717 ${{mp>0?'+':''}}${{mp.toFixed(1)}}% ${{ba}}G</span>`;
  return '\u2014';
}}

function render(data){{
  const tb=document.getElementById('tb');tb.innerHTML='';
  data.forEach(r=>{{
    const tr=document.createElement('tr');
    const dirCls=r.direction==='BUY'?'dir-buy':'dir-sell';
    const qCls=r.quality>=60?'q-hi':r.quality>=40?'q-mid':'q-lo';
    const tipLabel=TL[r.div_type]||r.div_type;
    const dateStr=r.signal_date?r.signal_date.slice(5):'—';
    tr.innerHTML=`<td>${{mkRule(r.rule)}}${{mkMods(r.mods)}}</td>
<td><a class="tv-link" href="https://www.tradingview.com/chart/?symbol=BIST:${{r.ticker}}" target="_blank">${{r.ticker}}</a></td>
<td class="${{dirCls}}">${{DL[r.direction]||r.direction}}</td>
<td style="font-size:.72rem">${{tipLabel}}</td>
<td>${{r.close}}</td>
<td class="${{qCls}}" style="font-weight:700">${{r.quality}}</td>
<td>${{mkRR(r.rr_ratio||0)}}</td>
<td>${{mkDetail(r)}}</td>
<td style="color:var(--text-muted);font-size:.72rem">${{dateStr}}</td>
<td style="font-size:.72rem;white-space:nowrap">${{mkViability(r)}}</td>`;
    tb.appendChild(tr);
  }});
  document.getElementById('st').innerHTML='<b>'+data.length+'</b> / '+D.length;
}}

initRuleCards();
initTabs();
af();
</script></body></html>"""
    return html


def _save_div_html(html_content, date_str, output_dir, suffix=''):
    """HTML dosyasini kaydet."""
    os.makedirs(output_dir, exist_ok=True)
    fname = f"nox_divergence{suffix}_{date_str.replace('-', '')}.html"
    path = os.path.join(output_dir, fname)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"\n  HTML: {path}")
    return path


# =============================================================================
# TELEGRAM
# =============================================================================

def _format_telegram(all_results, n_scanned, date_str, tf_label='', html_url=None):
    """Telegram HTML mesaji olustur — kural bazli gruplama."""
    tf_tag = f" {tf_label.strip()}" if tf_label.strip() else ''
    lines = []
    if html_url:
        lines.append(f'🔗 <a href="{html_url}">Detayli Rapor</a>\n')
    lines.append(f"📊 <b>NOX Uyumsuzluk v3{tf_tag}</b> — {date_str}\n")

    _LEGACY_KEYS = ('triple', 'rsi', 'macd', 'obv', 'mfi', 'adx', 'pv')
    _RULE_EMOJI = {'B': '🔴', 'D': '🟠', 'C': '🟢', 'E': '🔄'}

    # Tum sinyalleri topla ve kural siniflandir
    all_classified = []
    for sig_type in _LEGACY_KEYS:
        for item in all_results.get(sig_type, []):
            sig = item['signal']
            rule = _classify_rule(sig)
            if rule is None:
                continue
            all_classified.append({
                'rule': rule,
                'ticker': item['ticker'],
                'close': item['close'],
                'signal': sig,
            })

    lines.append(f"📋 {n_scanned} hisse | {len(all_classified)} sinyal\n")

    # Kural bazli gruplama
    for rule_key in RULE_ORDER:
        rd = RULE_DEFS[rule_key]
        rule_items = [i for i in all_classified if i['rule'] == rule_key]
        if not rule_items:
            continue

        rule_items.sort(key=lambda x: x['signal'].quality, reverse=True)
        emoji = _RULE_EMOJI.get(rule_key, '•')
        st = rd['stats']
        lines.append(f"{emoji} <b>{rule_key}: {rd['name']}</b> ({len(rule_items)})")
        lines.append(f"   <i>5G WR %{st['5g_wr']:.0f} Avg +{st['5g_avg']}%</i>")

        for item in rule_items[:8]:  # max 8 per rule
            sig = item['signal']
            dir_str = '🟢' if sig.direction == 'BUY' else '🔴'
            rr = getattr(sig, 'rr_ratio', 0) or 0
            rr_str = f" RR:{rr:.1f}" if rr > 0 else ''
            mods = _classify_modifiers(sig)
            mod_str = f" {''.join(MOD_DEFS[m]['label'] for m in mods)}" if mods else ''
            lines.append(f"  <code>{item['ticker']:<6}</code> {item['close']:>8.2f} K:{sig.quality}{rr_str}{mod_str} {dir_str}")

        if len(rule_items) > 8:
            lines.append(f"  <i>+{len(rule_items) - 8} daha...</i>")
        lines.append('')

    total = len(all_classified)
    total_buy = sum(1 for i in all_classified if i['signal'].direction == 'BUY')
    total_sell = total - total_buy
    lines.append(f"<b>Toplam:</b> {total} sinyal ({total_buy} AL + {total_sell} SAT)")

    return '\n'.join(lines)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="NOX Divergence Screener v2")
    parser.add_argument('tickers', nargs='*', help='Spesifik ticker(lar)')
    parser.add_argument('--period', default='2y', help='Veri periyodu (default: 2y)')
    parser.add_argument('--debug', metavar='TICKER', help='Tek hisse detayli debug')
    parser.add_argument('--top', type=int, metavar='N', help='En iyi N sinyal')
    parser.add_argument('--type', metavar='TYPE', dest='type_filter',
                        help='Tip filtresi (TRIPLE, RSI, MACD, OBV, MFI, ADX, PV)')
    parser.add_argument('--csv', action='store_true', help='CSV kaydet')
    parser.add_argument('--output', default='output', help='CSV cikti dizini')
    parser.add_argument('--scan-bars', type=int, default=5,
                        help='Son kac bar taranacak (default: 5)')
    parser.add_argument('--weekly', action='store_true',
                        help='Haftalik veri ile tara (daily resample)')
    parser.add_argument('--monthly', action='store_true',
                        help='Aylik veri ile tara (daily resample)')
    parser.add_argument('--html', action='store_true',
                        help='HTML rapor uret')
    parser.add_argument('--notify', action='store_true',
                        help='Telegram bildirimi gonder')
    args = parser.parse_args()

    # ── 1. Ticker listesi ────────────────────────────────────────────────────
    if args.tickers:
        tickers = [t.upper() for t in args.tickers]
        print(f"  {len(tickers)} ticker belirtildi: {', '.join(tickers)}")
    else:
        print(f"  Ticker listesi aliniyor...")
        tickers = data_mod.get_all_bist_tickers()
        print(f"  {len(tickers)} ticker bulundu.")

    if args.debug:
        debug_ticker = args.debug.upper()
        if debug_ticker not in tickers:
            tickers.append(debug_ticker)
    else:
        debug_ticker = None

    # ── 2. Veri yukleme ──────────────────────────────────────────────────────
    print(f"\n  Veri yukleniyor (period={args.period})...")
    t0 = time.time()
    all_data = data_mod.fetch_data(tickers, period=args.period)
    print(f"  {len(all_data)} hisse yuklendi ({time.time() - t0:.1f}s)")

    if not all_data:
        print("  HATA: Hicbir hisse verisi yuklenemedi!")
        sys.exit(1)

    # ── 3. Lowercase donusum + halt filtresi ─────────────────────────────────
    if args.monthly:
        print(f"\n  Aylik resample yapiliyor...")
        monthly_data = {}
        today = pd.Timestamp.now().normalize()
        # Sonraki is gunu: Cuma ise +3, Ct +2, Pz +1, diger +1
        wd = today.weekday()
        next_bday = today + pd.Timedelta(days=3 if wd == 4 else (2 if wd == 5 else (1 if wd == 6 else 1)))
        month_closed = next_bday.month != today.month  # sonraki is gunu farkli ayda
        for ticker, df in all_data.items():
            mdf = resample_monthly(df)
            if len(mdf) < 12:
                continue
            # Kapanmamis ayi cikar (ayin son is gunuyse ay tamamlanmis)
            if len(mdf) >= 2 and not month_closed:
                last_date = mdf.index[-1]
                if last_date.month == today.month and last_date.year == today.year:
                    mdf = mdf.iloc[:-1]
            if len(mdf) >= 12:
                monthly_data[ticker] = mdf
        print(f"  {len(monthly_data)} hisse ayliga donusturuldu (kapanmis mumlar, month_closed={month_closed}).")
        stock_dfs = {ticker: _to_lower_cols(df) for ticker, df in monthly_data.items()}
    elif args.weekly:
        print(f"\n  Haftalik resample yapiliyor...")
        weekly_data = {}
        today = pd.Timestamp.now().normalize()
        week_closed = today.weekday() >= 4  # Cuma kapanisi sonrasi hafta tamamlanmis
        for ticker, df in all_data.items():
            wdf = resample_weekly(df)
            if len(wdf) < 60:
                continue
            # Kapanmamis haftayi cikar (Cuma veya sonrasiysa hafta tamamlanmis)
            if len(wdf) >= 2 and not week_closed:
                last_date = wdf.index[-1]
                if last_date.isocalendar()[1] == today.isocalendar()[1]:
                    wdf = wdf.iloc[:-1]
            if len(wdf) >= 60:
                weekly_data[ticker] = wdf
        print(f"  {len(weekly_data)} hisse haftaliga donusturuldu (kapanmis mumlar, week_closed={week_closed}).")
        stock_dfs = {ticker: _to_lower_cols(df) for ticker, df in weekly_data.items()}
    else:
        stock_dfs = {ticker: _to_lower_cols(df) for ticker, df in all_data.items()}

    # ── 3b. Haftalik veri hazirla (rejim hesabi icin, gunluk modda) ────────
    regime_weekly = None
    if not args.weekly and not args.monthly:
        regime_weekly = {}
        for ticker, df in all_data.items():
            wdf = resample_weekly(df)
            if len(wdf) >= 12:
                regime_weekly[ticker] = _to_lower_cols(wdf)
        print(f"  {len(regime_weekly)} hisse haftalik rejim verisi hazir.")

    # ── 4. Tarama ────────────────────────────────────────────────────────────
    print(f"\n  Divergence taramasi v2 (scan_bars={args.scan_bars}, swing_order={DIV_CFG['swing_order']})...")
    t1 = time.time()
    min_bars = 18 if args.monthly else 60
    all_results, n_scanned, date_str = _scan_all(
        stock_dfs, debug_ticker=debug_ticker, scan_bars=args.scan_bars,
        min_bars=min_bars, weekly_data=regime_weekly,
    )
    total_signals = sum(len(all_results.get(k, [])) for k in ('rsi', 'macd', 'obv', 'mfi', 'adx', 'triple', 'pv'))
    print(f"  {n_scanned} hisse tarandi, {total_signals} sinyal ({time.time() - t1:.1f}s)")

    # ── 5. Rapor ─────────────────────────────────────────────────────────────
    tf_label = ' (AYLIK)' if args.monthly else (' (HAFTALIK)' if args.weekly else '')
    _print_results(all_results, n_scanned, date_str,
                   type_filter=args.type_filter, top_n=args.top,
                   tf_label=tf_label)

    # ── 6. CSV ───────────────────────────────────────────────────────────────
    if args.csv:
        _save_csv(all_results, date_str, args.output)

    # ── 7. HTML ───────────────────────────────────────────────────────────
    html_url = None
    if args.html:
        html_content = _generate_html(all_results, n_scanned, date_str, tf_label)
        suffix = '_weekly' if args.weekly else ('_monthly' if args.monthly else '')
        _save_div_html(html_content, date_str, args.output, suffix)
        # GitHub Pages push
        gh_filename = f"nox_divergence{suffix}.html"
        html_url = push_html_to_github(html_content, gh_filename, date_str)

    #── 7. Telegram ───────────────────────────────────────────────────────────
    if args.notify:
        msg = _format_telegram(all_results, n_scanned, date_str, tf_label,
                               html_url=html_url)
        send_telegram(msg)

    print(f"\n  Toplam sure: {time.time() - t0:.1f}s")
    print(f"  NOX Divergence v2 tamamlandi.")


if __name__ == '__main__':
    main()
