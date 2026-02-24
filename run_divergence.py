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
from core.indicators import ema, resample_weekly


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

def _scan_all(stock_dfs, debug_ticker=None, scan_bars=10):
    """Tum hisselerde divergence taramasi."""
    all_results = {
        'rsi': [], 'macd': [], 'obv': [], 'mfi': [], 'adx': [],
        'triple': [], 'pv': [],
    }
    n_scanned = 0
    last_date = None

    for ticker, df in stock_dfs.items():
        if len(df) < 60:
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
                if 'close_slope' in d and 'vol_slope' in d:
                    det_parts.append(f"FSlope: {d['close_slope']:.3f}")
                    det_parts.append(f"VSlope: {d['vol_slope']:.3f}")
                if 'vol_ratio' in d:
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
        print(f"  {'Hisse':<8} {'Fiyat':>8} {'F.Slope':>8} {'H.Slope':>8} "
              f"{'Kalite':>7} {'Tarih':>6}")
        print(f"  {'─' * 75}")
        for item in items:
            sig = item['signal']
            d = sig.details
            cs = d.get('close_slope', 0)
            vs = d.get('vol_slope', 0)
            date_str = _fmt_date(item['signal_date'])
            print(f"  {item['ticker']:<8} {item['close']:>8.2f} {cs:>8.3f} "
                  f"{vs:>8.3f} {sig.quality:>7} {date_str:>6}")

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
            if 'close_slope' in d:
                row['close_slope'] = d.get('close_slope')
            if 'vol_slope' in d:
                row['vol_slope'] = d.get('vol_slope')
            if 'vol_ratio' in d:
                row['vol_ratio'] = d.get('vol_ratio')
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
    all_data = data_mod.fetch_data(tickers, period=args.period, batch_size=25)
    print(f"  {len(all_data)} hisse yuklendi ({time.time() - t0:.1f}s)")

    if not all_data:
        print("  HATA: Hicbir hisse verisi yuklenemedi!")
        sys.exit(1)

    # ── 3. Lowercase donusum + halt filtresi ─────────────────────────────────
    if args.weekly:
        print(f"\n  Haftalik resample yapiliyor...")
        weekly_data = {}
        for ticker, df in all_data.items():
            wdf = resample_weekly(df)
            if len(wdf) < 60:
                continue
            # Kapanmamis haftayi cikar
            if len(wdf) >= 2:
                last_date = wdf.index[-1]
                today = pd.Timestamp.now().normalize()
                if (last_date - today).days >= 0 or last_date.isocalendar()[1] == today.isocalendar()[1]:
                    wdf = wdf.iloc[:-1]
            if len(wdf) >= 60:
                weekly_data[ticker] = wdf
        print(f"  {len(weekly_data)} hisse haftaliga donusturuldu (kapanmis mumlar).")
        stock_dfs = {ticker: _to_lower_cols(df) for ticker, df in weekly_data.items()}
    else:
        stock_dfs = {ticker: _to_lower_cols(df) for ticker, df in all_data.items()}

    # ── 4. Tarama ────────────────────────────────────────────────────────────
    print(f"\n  Divergence taramasi v2 (scan_bars={args.scan_bars}, swing_order={DIV_CFG['swing_order']})...")
    t1 = time.time()
    all_results, n_scanned, date_str = _scan_all(
        stock_dfs, debug_ticker=debug_ticker, scan_bars=args.scan_bars
    )
    total_signals = sum(len(v) for v in all_results.values())
    print(f"  {n_scanned} hisse tarandi, {total_signals} sinyal ({time.time() - t1:.1f}s)")

    # ── 5. Rapor ─────────────────────────────────────────────────────────────
    tf_label = ' (HAFTALIK)' if args.weekly else ''
    _print_results(all_results, n_scanned, date_str,
                   type_filter=args.type_filter, top_n=args.top,
                   tf_label=tf_label)

    # ── 6. CSV ───────────────────────────────────────────────────────────────
    if args.csv:
        _save_csv(all_results, date_str, args.output)

    print(f"\n  Toplam sure: {time.time() - t0:.1f}s")
    print(f"  NOX Divergence v2 tamamlandi.")


if __name__ == '__main__':
    main()
