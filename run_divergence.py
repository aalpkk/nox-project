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
    scan_divergences, DIV_CFG, _find_swings,
    _calc_atr, _calc_rsi, _calc_macd, _calc_obv, _calc_mfi, _calc_adx,
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
# TARAMA
# =============================================================================

def _scan_all(stock_dfs, debug_ticker=None, scan_bars=10, min_bars=60):
    """Tum hisselerde divergence taramasi."""
    all_results = {
        'rsi': [], 'macd': [], 'obv': [], 'mfi': [], 'adx': [],
        'triple': [], 'pv': [],
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

            result = scan_divergences(df, scan_bars=scan_bars)
            n_scanned += 1

            if last_date is None and len(df) > 0:
                last_date = df.index[-1]

            for sig_type in ('rsi', 'macd', 'obv', 'mfi', 'adx', 'triple', 'pv'):
                for sig in result[sig_type]:
                    if sig.bar_idx < len(df):
                        sig_date = df.index[sig.bar_idx]
                    else:
                        sig_date = df.index[-1]

                    all_results[sig_type].append({
                        'ticker': ticker,
                        'close': float(df['close'].iloc[-1]),
                        'signal': sig,
                        'signal_date': sig_date,
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
    print(f"\n  {'=' * 70}")
    print(f"  DEBUG: {ticker}")
    print(f"  {'=' * 70}")

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

    # Swing'ler (pivot yerine)
    swing_lows, swing_highs = _find_swings(df['close'], cfg['swing_order'])

    print(f"\n  Swing Lows ({len(swing_lows)} toplam, son 10):")
    print(f"  {'SwingIdx':>8} {'SinyalIdx':>9} {'Fiyat':>10}")
    print(f"  {'─' * 30}")
    for sw in swing_lows[-10:]:
        print(f"  {sw[0]:>8} {sw[1]:>9} {sw[2]:>10.4f}")

    print(f"\n  Swing Highs ({len(swing_highs)} toplam, son 10):")
    print(f"  {'SwingIdx':>8} {'SinyalIdx':>9} {'Fiyat':>10}")
    print(f"  {'─' * 30}")
    for sw in swing_highs[-10:]:
        print(f"  {sw[0]:>8} {sw[1]:>9} {sw[2]:>10.4f}")

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
    total = sum(len(v) for v in result.values())

    if total > 0:
        print(f"\n  Bulunan Sinyaller ({total}):")
        print(f"  {'Bar':>5} {'Yon':<5} {'Tip':<15} {'Kalite':>7} {'Detay'}")
        print(f"  {'─' * 70}")
        for sig_type in ('triple', 'rsi', 'macd', 'obv', 'mfi', 'adx', 'pv'):
            for sig in result[sig_type]:
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
                    det_parts.append(f"ADX: {d['adx_value']:.0f} slope={d['adx_slope']:.2f}")
                if 'sub_type' in d:
                    det_parts.append(f"{d['sub_type']}")
                    det_parts.append(f"VolR: {d.get('vol_ratio', 0):.1f}")
                    det_parts.append(f"R/ATR: {d.get('range_atr', 0):.3f}")
                    if d.get('limit_tag'):
                        det_parts.append(f"[{d['limit_tag']}]")
                elif 'vol_ratio' in d:
                    det_parts.append(f"VolR: {d['vol_ratio']:.1f}")
                if d.get('has_mfi'):
                    det_parts.append("MFI+")
                det_str = ', '.join(det_parts) if det_parts else '-'
                print(f"  {sig.bar_idx:>5} {sig.direction:<5} {sig.div_type:<15} "
                      f"{sig.quality:>7} {det_str}")
        print(f"  {'─' * 70}")
    else:
        print(f"\n  Sinyal bulunamadi (son 30 bar).")

    print(f"  {'=' * 70}\n")


# =============================================================================
# KONSOL RAPOR
# =============================================================================

def _print_section(title, items, section_type):
    """Bir bolum icin tablo yazdir."""
    if not items:
        return

    count = len(items)
    items.sort(key=lambda x: x['signal'].quality, reverse=True)

    print(f"\n  ◆ {title} ({count})")
    print(f"  {'─' * 75}")

    if section_type in ('triple',):
        print(f"  {'Hisse':<8} {'Fiyat':>8} {'RSI Div':>8} {'MACD Div':>9} "
              f"{'Hacim':>6} {'MFI':>4} {'Kalite':>7} {'Tarih':>6}")
        print(f"  {'─' * 75}")
        for item in items:
            sig = item['signal']
            d = sig.details
            vol_r = d.get('vol_ratio', 0)
            vol_str = f"x{vol_r:.1f}" if vol_r > 0 else '-'
            mfi_str = '+' if d.get('has_mfi') else '-'
            date_str = _fmt_date(item['signal_date'])
            print(f"  {item['ticker']:<8} {item['close']:>8.2f} {'Klasik':>8} "
                  f"{'Klasik':>9} {vol_str:>6} {mfi_str:>4} {sig.quality:>7} {date_str:>6}")

    elif section_type in ('rsi',):
        print(f"  {'Hisse':<8} {'Fiyat':>8} {'Tip':>8} {'RSI1':>6} {'RSI2':>6} "
              f"{'Fark':>6} {'Kalite':>7} {'Tarih':>6}")
        print(f"  {'─' * 75}")
        for item in items:
            sig = item['signal']
            d = sig.details
            tip = 'Klasik' if sig.div_type == 'RSI_CLASSIC' else 'Gizli'
            rsi1 = d.get('prev_rsi', 0)
            rsi2 = d.get('curr_rsi', 0)
            diff = d.get('rsi_diff', 0)
            date_str = _fmt_date(item['signal_date'])
            print(f"  {item['ticker']:<8} {item['close']:>8.2f} {tip:>8} "
                  f"{rsi1:>6.1f} {rsi2:>6.1f} {diff:>6.1f} "
                  f"{sig.quality:>7} {date_str:>6}")

    elif section_type in ('macd',):
        print(f"  {'Hisse':<8} {'Fiyat':>8} {'Tip':>8} {'Hist1':>9} {'Hist2':>9} "
              f"{'Kalite':>7} {'Tarih':>6}")
        print(f"  {'─' * 75}")
        for item in items:
            sig = item['signal']
            d = sig.details
            tip = 'Klasik' if sig.div_type == 'MACD_CLASSIC' else 'Gizli'
            h1 = d.get('prev_hist', 0)
            h2 = d.get('curr_hist', 0)
            date_str = _fmt_date(item['signal_date'])
            print(f"  {item['ticker']:<8} {item['close']:>8.2f} {tip:>8} "
                  f"{h1:>9.4f} {h2:>9.4f} "
                  f"{sig.quality:>7} {date_str:>6}")

    elif section_type in ('obv',):
        print(f"  {'Hisse':<8} {'Fiyat':>8} {'Tip':>8} {'OBV1':>12} {'OBV2':>12} "
              f"{'Kalite':>7} {'Tarih':>6}")
        print(f"  {'─' * 75}")
        for item in items:
            sig = item['signal']
            d = sig.details
            tip = 'Klasik' if sig.div_type == 'OBV_CLASSIC' else 'Gizli'
            o1 = d.get('prev_obv', 0)
            o2 = d.get('curr_obv', 0)
            date_str = _fmt_date(item['signal_date'])
            print(f"  {item['ticker']:<8} {item['close']:>8.2f} {tip:>8} "
                  f"{o1:>12.0f} {o2:>12.0f} "
                  f"{sig.quality:>7} {date_str:>6}")

    elif section_type in ('mfi',):
        print(f"  {'Hisse':<8} {'Fiyat':>8} {'Tip':>8} {'MFI1':>6} {'MFI2':>6} "
              f"{'Fark':>6} {'Kalite':>7} {'Tarih':>6}")
        print(f"  {'─' * 75}")
        for item in items:
            sig = item['signal']
            d = sig.details
            tip = 'Klasik' if sig.div_type == 'MFI_CLASSIC' else 'Gizli'
            m1 = d.get('prev_mfi', 0)
            m2 = d.get('curr_mfi', 0)
            diff = d.get('mfi_diff', 0)
            date_str = _fmt_date(item['signal_date'])
            print(f"  {item['ticker']:<8} {item['close']:>8.2f} {tip:>8} "
                  f"{m1:>6.1f} {m2:>6.1f} {diff:>6.1f} "
                  f"{sig.quality:>7} {date_str:>6}")

    elif section_type in ('adx',):
        print(f"  {'Hisse':<8} {'Fiyat':>8} {'ADX':>6} {'Slope':>7} {'F.Slope':>8} "
              f"{'Kalite':>7} {'Tarih':>6}")
        print(f"  {'─' * 75}")
        for item in items:
            sig = item['signal']
            d = sig.details
            adx_val = d.get('adx_value', 0)
            adx_sl = d.get('adx_slope', 0)
            cs = d.get('close_slope', 0)
            date_str = _fmt_date(item['signal_date'])
            print(f"  {item['ticker']:<8} {item['close']:>8.2f} {adx_val:>6.1f} "
                  f"{adx_sl:>7.2f} {cs:>8.3f} "
                  f"{sig.quality:>7} {date_str:>6}")

    elif section_type in ('pv',):
        print(f"  {'Hisse':<8} {'Fiyat':>8} {'Tip':>8} {'VolR':>6} {'R/ATR':>7} "
              f"{'Kalite':>7} {'Tarih':>6}")
        print(f"  {'─' * 75}")
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
                  f"{vol_str:>6} {ra:>7.3f} {sig.quality:>7} {date_str:>6}{tag}")

    print(f"  {'─' * 75}")


def _fmt_date(sig_date):
    """Tarih formatlama."""
    if hasattr(sig_date, 'strftime'):
        return sig_date.strftime('%m-%d')
    return str(sig_date)[-5:]


def _print_results(all_results, n_scanned, date_str, type_filter=None, top_n=None, tf_label=''):
    """Konsol rapor."""
    w = 75

    # Type filtresi
    if type_filter:
        tf = type_filter.upper()
        filter_map = {
            'TRIPLE': ['triple'],
            'UCLU': ['triple'],
            'RSI': ['rsi'],
            'MACD': ['macd'],
            'OBV': ['obv'],
            'MFI': ['mfi'],
            'ADX': ['adx'],
            'PV': ['pv'],
            'PRICE_VOLUME': ['pv'],
            'FH': ['pv'],
            'FIYAT': ['pv'],
        }
        allowed = filter_map.get(tf, list(all_results.keys()))
        for key in list(all_results.keys()):
            if key not in allowed:
                all_results[key] = []

    # Top N — tum sinyalleri birlestir, kaliteye gore sirala, sonra kes
    if top_n and top_n > 0:
        all_flat = []
        for sig_type, items in all_results.items():
            for item in items:
                all_flat.append((sig_type, item))
        all_flat.sort(key=lambda x: x[1]['signal'].quality, reverse=True)
        all_flat = all_flat[:top_n]
        for key in all_results:
            all_results[key] = []
        for sig_type, item in all_flat:
            all_results[sig_type].append(item)

    # Sayimlar
    total_buy = 0
    total_sell = 0
    type_counts = {}
    for sig_type, items in all_results.items():
        for item in items:
            d = item['signal'].direction
            if d == 'BUY':
                total_buy += 1
            else:
                total_sell += 1
            type_counts[sig_type] = type_counts.get(sig_type, 0) + 1

    print(f"\n{'═' * w}")
    print(f"  NOX UYUMSUZLUK TARAMASI v2{tf_label} — {date_str} — {n_scanned} hisse tarandi")
    print(f"{'═' * w}")

    # Bolumler: triple > rsi > macd > obv > mfi > adx > pv
    sections = [
        ('triple', 'triple_buy', 'triple_sell'),
        ('rsi', 'rsi_buy', 'rsi_sell'),
        ('macd', 'macd_buy', 'macd_sell'),
        ('obv', 'obv_buy', 'obv_sell'),
        ('mfi', 'mfi_buy', 'mfi_sell'),
        ('adx', 'adx_buy', 'adx_sell'),
        ('pv', 'pv_buy', 'pv_sell'),
    ]

    for sig_type, buy_key, sell_key in sections:
        items = all_results.get(sig_type, [])
        if not items:
            continue
        buy_items = [i for i in items if i['signal'].direction == 'BUY']
        sell_items = [i for i in items if i['signal'].direction == 'SELL']

        if buy_items:
            _print_section(SECTION_TITLES[buy_key], buy_items, sig_type)
        if sell_items:
            _print_section(SECTION_TITLES[sell_key], sell_items, sig_type)

    # Ozet
    type_labels_tr = {
        'triple': 'UCLU', 'rsi': 'RSI', 'macd': 'MACD',
        'obv': 'OBV', 'mfi': 'MFI', 'adx': 'ADX', 'pv': 'FH',
    }
    counts_parts = []
    for key in ('triple', 'rsi', 'macd', 'obv', 'mfi', 'adx', 'pv'):
        c = type_counts.get(key, 0)
        if c > 0:
            counts_parts.append(f"{c} {type_labels_tr[key]}")
    counts_str = ', '.join(counts_parts) if counts_parts else 'yok'

    print(f"\n{'═' * w}")
    print(f"  OZET: {total_buy} AL + {total_sell} SAT ({counts_str})")
    print(f"{'═' * w}")


# =============================================================================
# CSV
# =============================================================================

def _save_csv(all_results, date_str, output_dir):
    """Sinyalleri CSV dosyasina kaydet."""
    os.makedirs(output_dir, exist_ok=True)
    rows = []

    for sig_type, items in all_results.items():
        for item in items:
            sig = item['signal']
            sig_date = item['signal_date']

            row = {
                'ticker': item['ticker'],
                'close': round(item['close'], 4),
                'div_type': sig.div_type,
                'direction': sig.direction,
                'quality': sig.quality,
                'signal_date': sig_date.strftime('%Y-%m-%d') if hasattr(sig_date, 'strftime') else str(sig_date),
            }

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
        csv_df.sort_values(['quality', 'ticker'], ascending=[False, True], inplace=True)
        fname = f"nox_divergence_{date_str.replace('-', '')}.csv"
        path = os.path.join(output_dir, fname)
        csv_df.to_csv(path, index=False)
        print(f"\n  CSV: {path}")
    else:
        print(f"\n  Sinyal yok, CSV olusturulmadi.")


# =============================================================================
# HTML RAPOR
# =============================================================================

# Tab tanimlamalari: key -> (label, renk)
_DIV_TABS = {
    'triple': ('UCLU', 'var(--nox-purple)'),
    'rsi':    ('RSI',  'var(--nox-cyan)'),
    'macd':   ('MACD', 'var(--nox-orange)'),
    'obv':    ('OBV',  'var(--nox-green)'),
    'mfi':    ('MFI',  'var(--nox-blue)'),
    'adx':    ('ADX',  'var(--nox-yellow)'),
    'pv':     ('FH',   'var(--nox-red)'),
}
_TAB_ORDER = ['triple', 'rsi', 'macd', 'obv', 'mfi', 'adx', 'pv']


def _flatten_results(all_results):
    """all_results dict -> flat list of dicts for JSON embed."""
    rows = []
    for sig_type, items in all_results.items():
        for item in items:
            sig = item['signal']
            d = sig.details
            sig_date = item['signal_date']

            row = {
                'ticker': item['ticker'],
                'close': round(float(item['close']), 2),
                'div_type': sig.div_type,
                'direction': sig.direction,
                'quality': sig.quality,
                'category': sig_type,
                'signal_date': sig_date.strftime('%Y-%m-%d') if hasattr(sig_date, 'strftime') else str(sig_date),
            }

            # Tip-spesifik detaylar
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

            rows.append(row)
    return rows


def _generate_html(all_results, n_scanned, date_str, tf_label=''):
    """Divergence HTML raporu olustur."""
    now = datetime.now().strftime('%d.%m.%Y %H:%M')
    flat = _flatten_results(all_results)
    rows_json = json.dumps(_sanitize(flat), ensure_ascii=False)

    tf_tag = tf_label.strip() if tf_label.strip() else ''
    title_suffix = f' · {tf_tag}' if tf_tag else ''

    # Pre-compute JSON strings (f-string icinde dict comp kullanmamak icin)
    dir_labels_json = json.dumps({'BUY': 'AL', 'SELL': 'SAT'})
    type_labels_json = json.dumps(TYPE_LABELS)
    tabs_json = json.dumps(_TAB_ORDER)
    tab_labels_json = json.dumps({k: v[0] for k, v in _DIV_TABS.items()})
    tab_colors_json = json.dumps({k: v[1] for k, v in _DIV_TABS.items()})

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
</style>
</head><body>
<div class="nox-container">
<div class="nox-header">
  <div class="nox-logo">NOX<span class="proj">project</span><span class="mode">divergence{title_suffix}</span></div>
  <div class="nox-meta"><b>{len(flat)}</b> sinyal / {n_scanned} taranan<br>{now}</div>
</div>
<div class="tab-bar" id="tabs"></div>
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
<th onclick="sb('ticker')">Hisse</th>
<th onclick="sb('direction')">Yon</th>
<th onclick="sb('div_type')">Tip</th>
<th onclick="sb('close')">Fiyat</th>
<th onclick="sb('quality')">Kalite</th>
<th>Detay</th>
<th onclick="sb('signal_date')">Tarih</th>
</tr></thead><tbody id="tb"></tbody></table>
</div>
<div class="nox-status" id="st"><b>{len(flat)}</b> / {len(flat)}</div>
</div>
<script>
const D={rows_json};
const DL={dir_labels_json};
const TL={type_labels_json};
const TABS={tabs_json};
const TAB_LABELS={tab_labels_json};
const TAB_COLORS={tab_colors_json};
let curTab='all',col='quality',asc=false;

function initTabs(){{
  const bar=document.getElementById('tabs');
  // "Tumu" tab
  const allBtn=document.createElement('div');
  allBtn.className='tab-btn active';allBtn.dataset.t='all';
  allBtn.innerHTML='<span class="dot" style="background:var(--nox-cyan)"></span>Tumu <span class="cnt">'+D.length+'</span>';
  allBtn.onclick=()=>setTab('all');
  bar.appendChild(allBtn);
  TABS.forEach(t=>{{
    const cnt=D.filter(r=>r.category===t).length;
    if(!cnt)return;
    const btn=document.createElement('div');
    btn.className='tab-btn';btn.dataset.t=t;
    btn.innerHTML='<span class="dot" style="background:'+TAB_COLORS[t]+'"></span>'+TAB_LABELS[t]+' <span class="cnt">'+cnt+'</span>';
    btn.onclick=()=>setTab(t);
    bar.appendChild(btn);
  }});
}}

function setTab(t){{
  curTab=t;
  document.querySelectorAll('.tab-btn').forEach(b=>b.classList.remove('active'));
  document.querySelector('.tab-btn[data-t="'+t+'"]').classList.add('active');
  af();
}}

function af(){{
  const dir=document.getElementById('fDir').value;
  const minQ=parseInt(document.getElementById('fQ').value)||0;
  const sr=document.getElementById('fS').value.toUpperCase();
  let f=D.filter(r=>{{
    if(curTab!=='all'&&r.category!==curTab)return false;
    if(dir&&r.direction!==dir)return false;
    if(r.quality<minQ)return false;
    if(sr&&!r.ticker.includes(sr))return false;
    return true;
  }});
  f.sort((a,b)=>{{
    let va=a[col],vb=b[col];
    if(typeof va==='string')return asc?va.localeCompare(vb):vb.localeCompare(va);
    return asc?(va||0)-(vb||0):(vb||0)-(va||0);
  }});
  render(f);
}}

function sb(c){{if(col===c)asc=!asc;else{{col=c;asc=c==='ticker'||c==='signal_date'}};
  document.querySelectorAll('th').forEach(h=>h.classList.remove('sorted'));
  af();
}}

function reset(){{
  curTab='all';
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
  return parts.length?'<span class="detail-cell">'+parts.join(', ')+'</span>':'<span style="color:var(--text-muted)">—</span>';
}}

function render(data){{
  const tb=document.getElementById('tb');tb.innerHTML='';
  data.forEach(r=>{{
    const tr=document.createElement('tr');
    const dirCls=r.direction==='BUY'?'dir-buy':'dir-sell';
    const qCls=r.quality>=60?'q-hi':r.quality>=40?'q-mid':'q-lo';
    const tipLabel=TL[r.div_type]||r.div_type;
    const dateStr=r.signal_date?r.signal_date.slice(5):'—';
    tr.innerHTML=`<td><a class="tv-link" href="https://www.tradingview.com/chart/?symbol=BIST:${{r.ticker}}" target="_blank">${{r.ticker}}</a></td>
<td class="${{dirCls}}">${{DL[r.direction]||r.direction}}</td>
<td style="font-size:.72rem">${{tipLabel}}</td>
<td>${{r.close}}</td>
<td class="${{qCls}}" style="font-weight:700">${{r.quality}}</td>
<td>${{mkDetail(r)}}</td>
<td style="color:var(--text-muted);font-size:.72rem">${{dateStr}}</td>`;
    tb.appendChild(tr);
  }});
  document.getElementById('st').innerHTML='<b>'+data.length+'</b> / '+D.length;
}}

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
    """Telegram HTML mesaji olustur."""
    tf_tag = f" {tf_label.strip()}" if tf_label.strip() else ''
    lines = []
    if html_url:
        lines.append(f'🔗 <a href="{html_url}">Detayli Rapor</a>\n')
    lines.append(f"📊 <b>NOX Uyumsuzluk{tf_tag}</b> — {date_str}\n")

    # Sayimlar
    counts = {}
    type_labels = {'triple': 'UCLU', 'rsi': 'RSI', 'macd': 'MACD',
                   'obv': 'OBV', 'mfi': 'MFI', 'adx': 'ADX', 'pv': 'FH'}
    for sig_type, items in all_results.items():
        if items:
            counts[sig_type] = len(items)
    parts = [f"{c} {type_labels[k]}" for k, c in counts.items() if c > 0]
    lines.append(f"📋 {n_scanned} hisse | {' | '.join(parts)}\n")

    def _fmt_items(items, emoji, title, min_q, max_n, sig_type):
        """Bir bolum icin satir listesi."""
        filtered = [i for i in items if i['signal'].quality >= min_q]
        filtered.sort(key=lambda x: x['signal'].quality, reverse=True)
        filtered = filtered[:max_n]
        if not filtered:
            return
        lines.append(f"{emoji} <b>{title}</b>")
        for item in filtered:
            sig = item['signal']
            d = sig.details
            detail_parts = []
            if sig_type == 'triple':
                vol_r = d.get('vol_ratio', 0)
                if vol_r > 0:
                    detail_parts.append(f"RSI+MACD x{vol_r:.1f}")
                if d.get('has_mfi'):
                    detail_parts.append("MFI+")
            elif sig_type == 'rsi':
                tip = 'Klasik' if sig.div_type == 'RSI_CLASSIC' else 'Gizli'
                r1 = d.get('prev_rsi', 0)
                r2 = d.get('curr_rsi', 0)
                detail_parts.append(f"{tip} {r1:.1f}→{r2:.1f}")
            elif sig_type == 'macd':
                tip = 'Klasik' if sig.div_type == 'MACD_CLASSIC' else 'Gizli'
                detail_parts.append(tip)
            det = ' '.join(detail_parts) if detail_parts else ''
            det_str = f" {det}" if det else ''
            lines.append(f"  <code>{item['ticker']:<6}</code> {item['close']:>8.2f} K:{sig.quality}{det_str}")
        lines.append('')

    # Triple SAT/AL
    triple = all_results.get('triple', [])
    triple_sell = [i for i in triple if i['signal'].direction == 'SELL']
    triple_buy = [i for i in triple if i['signal'].direction == 'BUY']
    _fmt_items(triple_sell, '🔻', 'UCLU SAT (K≥50)', 50, 10, 'triple')
    _fmt_items(triple_buy, '🟢', 'UCLU AL (K≥50)', 50, 10, 'triple')

    # RSI SAT/AL
    rsi = all_results.get('rsi', [])
    rsi_sell = [i for i in rsi if i['signal'].direction == 'SELL']
    rsi_buy = [i for i in rsi if i['signal'].direction == 'BUY']
    _fmt_items(rsi_sell, '📉', 'RSI SAT (K≥40)', 40, 10, 'rsi')
    _fmt_items(rsi_buy, '📈', 'RSI AL (K≥40)', 40, 10, 'rsi')

    # MACD SAT/AL
    macd = all_results.get('macd', [])
    macd_sell = [i for i in macd if i['signal'].direction == 'SELL']
    macd_buy = [i for i in macd if i['signal'].direction == 'BUY']
    _fmt_items(macd_sell, '📉', 'MACD SAT (K≥50)', 50, 5, 'macd')
    _fmt_items(macd_buy, '📈', 'MACD AL (K≥50)', 50, 5, 'macd')

    # Toplam ozet
    total_buy = sum(1 for items in all_results.values() for i in items if i['signal'].direction == 'BUY')
    total_sell = sum(1 for items in all_results.values() for i in items if i['signal'].direction == 'SELL')
    lines.append(f"<b>Toplam:</b> {total_buy} AL + {total_sell} SAT")

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
    parser.add_argument('--scan-bars', type=int, default=10,
                        help='Son kac bar taranacak (default: 10)')
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

    # ── 4. Tarama ────────────────────────────────────────────────────────────
    print(f"\n  Divergence taramasi v2 (scan_bars={args.scan_bars}, swing_order={DIV_CFG['swing_order']})...")
    t1 = time.time()
    min_bars = 18 if args.monthly else 60
    all_results, n_scanned, date_str = _scan_all(
        stock_dfs, debug_ticker=debug_ticker, scan_bars=args.scan_bars,
        min_bars=min_bars
    )
    total_signals = sum(len(v) for v in all_results.values())
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
