#!/usr/bin/env python3
"""
NOX SMC Pattern Screener — Runner
==================================
BIST hisselerinde SMC (Smart Money Concepts) pattern taramasi.
QM, Fakeout, Flag, 3-Drive, Compression, Can-Can, 2R/2S Fakeout.

Kullanim:
    python run_smc.py                    # tum hisseler
    python run_smc.py --debug THYAO      # tek hisse detayli
    python run_smc.py --pattern QM       # sadece QM sinyalleri
    python run_smc.py --top 20           # en iyi 20 sinyal
    python run_smc.py --csv              # CSV kaydet
    python run_smc.py THYAO EREGL SISE   # spesifik hisseler
"""

import argparse
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

from markets.bist import data as data_mod
from markets.bist.smc_patterns import (
    scan_patterns, detect_structure, get_market_bias, SMC_CFG,
    _calc_atr,
)


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
# PATTERN GRUPLAMA
# =============================================================================

PATTERN_GROUPS = {
    'QM': {
        'title_buy': 'QM RETEST — AL',
        'title_sell': 'QM RETEST — SAT',
        'patterns': ['QM_QUICK', 'QM_LATE'],
        'columns': ['Hisse', 'Fiyat', 'Seviye', 'Kalite', 'Stop', 'Hedef', 'Tip', 'Tarih'],
    },
    'FAKEOUT': {
        'title_buy': 'FAKEOUT — AL',
        'title_sell': 'FAKEOUT — SAT',
        'patterns': ['FAKEOUT_V1', 'FAKEOUT_V2'],
        'columns': ['Hisse', 'Fiyat', 'Sweep', 'Kalite', 'Stop', 'Hedef', 'Tip', 'Tarih'],
    },
    'FLAG': {
        'title_buy': 'FLAG BREAKOUT — AL',
        'title_sell': 'FLAG BREAKOUT — SAT',
        'patterns': ['FLAG_B'],
        'columns': ['Hisse', 'Fiyat', 'Level', 'Kalite', 'Stop', 'Hedef', 'Retrace', 'Tarih'],
    },
    '3DRIVE': {
        'title_buy': '3-DRIVE — AL',
        'title_sell': '3-DRIVE — SAT',
        'patterns': ['3DRIVE'],
        'columns': ['Hisse', 'Fiyat', 'Level', 'Kalite', 'Stop', 'Hedef', 'Ratio', 'Tarih'],
    },
    'COMPRESSION': {
        'title_buy': 'COMPRESSION — AL',
        'title_sell': 'COMPRESSION — SAT',
        'patterns': ['COMPRESSION'],
        'columns': ['Hisse', 'Fiyat', 'Range', 'Kalite', 'Yon', 'Bars', 'Tarih'],
    },
    'CANCAN': {
        'title_buy': 'CAN-CAN — AL',
        'title_sell': 'CAN-CAN — SAT',
        'patterns': ['CANCAN'],
        'columns': ['Hisse', 'Fiyat', 'Seviye', 'Kalite', 'Stop', 'Hedef', 'Flip', 'Tarih'],
    },
    '2RS': {
        'title_buy': '2S FAKEOUT — AL',
        'title_sell': '2R FAKEOUT — SAT',
        'patterns': ['2R_FAKEOUT', '2S_FAKEOUT'],
        'columns': ['Hisse', 'Fiyat', 'Seviye', 'Kalite', 'Stop', 'Hedef', 'Touch', 'Tarih'],
    },
}


# =============================================================================
# TARAMA
# =============================================================================

def _scan_all(stock_dfs, debug_ticker=None, scan_bars=15):
    """Tum hisselerde SMC pattern taramasi."""
    all_signals = []
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

            sigs = scan_patterns(df, scan_bars=scan_bars)
            n_scanned += 1

            if last_date is None and len(df) > 0:
                last_date = df.index[-1]

            for sig in sigs:
                # Tarih bilgisi ekle
                if sig.bar_idx < len(df):
                    sig_date = df.index[sig.bar_idx]
                else:
                    sig_date = df.index[-1]

                all_signals.append({
                    'ticker': ticker,
                    'close': float(df['close'].iloc[-1]),
                    'signal': sig,
                    'signal_date': sig_date,
                })

        except Exception as e:
            print(f"  ! {ticker}: {e}")
            continue

    date_str = last_date.strftime('%Y-%m-%d') if last_date else datetime.now().strftime('%Y-%m-%d')
    return all_signals, n_scanned, date_str


# =============================================================================
# DEBUG
# =============================================================================

def _print_debug(ticker, df):
    """Tek hisse detayli pivot + yapi + sinyal ciktisi."""
    print(f"\n  {'=' * 70}")
    print(f"  DEBUG: {ticker}")
    print(f"  {'=' * 70}")

    pivots = detect_structure(df, SMC_CFG['pivot_lb'])
    bias = get_market_bias(pivots)
    atr = _calc_atr(df, SMC_CFG['atr_len'])

    print(f"\n  Market Bias: {bias}")
    print(f"  Toplam pivot: {len(pivots)}")
    print(f"  Son ATR: {atr.iloc[-1]:.4f}")
    print(f"  Son Close: {df['close'].iloc[-1]:.2f}")

    # Son 15 pivot
    print(f"\n  Son 15 Pivot:")
    print(f"  {'Idx':>5} {'Onay':>5} {'Tip':<5} {'Fiyat':>10} {'Etiket':<5}")
    print(f"  {'─' * 35}")
    for p in pivots[-15:]:
        print(f"  {p.idx:>5} {p.confirm_idx:>5} {p.ptype:<5} {p.price:>10.4f} {p.label:<5}")
    print(f"  {'─' * 35}")

    # Sinyal taramasi
    sigs = scan_patterns(df, scan_bars=30)
    if sigs:
        print(f"\n  Bulunan Sinyaller ({len(sigs)}):")
        print(f"  {'Bar':>5} {'Yon':<5} {'Pattern':<15} {'Seviye':>10} {'Kalite':>7} {'Stop':>10} {'Hedef':>10}")
        print(f"  {'─' * 70}")
        for s in sigs:
            print(f"  {s.bar_idx:>5} {s.direction:<5} {s.pattern:<15} "
                  f"{s.key_level:>10.4f} {s.quality:>7} {s.stop:>10.4f} {s.target:>10.4f}")
            if s.details:
                det_str = ', '.join(f'{k}={v}' for k, v in s.details.items()
                                    if k not in ('choch',))
                print(f"         {det_str}")
        print(f"  {'─' * 70}")
    else:
        print(f"\n  Sinyal bulunamadi (son 30 bar).")

    print(f"  {'=' * 70}\n")


# =============================================================================
# KONSOL RAPOR
# =============================================================================

def _format_signal_row(ticker, close, sig, sig_date):
    """Tek sinyal icin row dict olustur."""
    date_str = sig_date.strftime('%m-%d') if hasattr(sig_date, 'strftime') else str(sig_date)[-5:]
    return {
        'ticker': ticker,
        'close': close,
        'key_level': sig.key_level,
        'quality': sig.quality,
        'stop': sig.stop,
        'target': sig.target,
        'pattern': sig.pattern,
        'direction': sig.direction,
        'date': date_str,
        'details': sig.details,
    }


def _print_pattern_group(group_key, group_info, signals_in_group, direction):
    """Bir pattern grubu icin tablo yazdir."""
    filtered = [s for s in signals_in_group if s['direction'] == direction]
    if not filtered:
        return

    title = group_info['title_buy'] if direction == 'BUY' else group_info['title_sell']
    count = len(filtered)

    # Kaliteye gore sirala
    filtered.sort(key=lambda s: s['quality'], reverse=True)

    direction_label = 'AL' if direction == 'BUY' else 'SAT'
    print(f"\n  ◆ {title} ({count} sinyal)")
    print(f"  {'─' * 75}")

    if group_key == 'QM':
        print(f"  {'Hisse':<8} {'Fiyat':>8} {'Seviye':>8} {'Kalite':>7} "
              f"{'Stop':>8} {'Hedef':>8} {'Tip':<8} {'Tarih':>6}")
        print(f"  {'─' * 75}")
        for s in filtered:
            tip = s['pattern'].replace('QM_', '')
            print(f"  {s['ticker']:<8} {s['close']:>8.2f} {s['key_level']:>8.2f} "
                  f"{s['quality']:>7} {s['stop']:>8.2f} {s['target']:>8.2f} "
                  f"{tip:<8} {s['date']:>6}")

    elif group_key == 'FAKEOUT':
        print(f"  {'Hisse':<8} {'Fiyat':>8} {'Sweep':>8} {'Kalite':>7} "
              f"{'Stop':>8} {'Hedef':>8} {'Tip':<8} {'Tarih':>6}")
        print(f"  {'─' * 75}")
        for s in filtered:
            tip = s['pattern'].replace('FAKEOUT_', '')
            print(f"  {s['ticker']:<8} {s['close']:>8.2f} {s['key_level']:>8.2f} "
                  f"{s['quality']:>7} {s['stop']:>8.2f} {s['target']:>8.2f} "
                  f"{tip:<8} {s['date']:>6}")

    elif group_key == 'FLAG':
        print(f"  {'Hisse':<8} {'Fiyat':>8} {'Level':>8} {'Kalite':>7} "
              f"{'Stop':>8} {'Hedef':>8} {'Ret%':>6} {'Tarih':>6}")
        print(f"  {'─' * 75}")
        for s in filtered:
            ret_pct = s['details'].get('retrace_pct', 0) * 100
            print(f"  {s['ticker']:<8} {s['close']:>8.2f} {s['key_level']:>8.2f} "
                  f"{s['quality']:>7} {s['stop']:>8.2f} {s['target']:>8.2f} "
                  f"{ret_pct:>5.1f}% {s['date']:>6}")

    elif group_key == '3DRIVE':
        print(f"  {'Hisse':<8} {'Fiyat':>8} {'Level':>8} {'Kalite':>7} "
              f"{'Stop':>8} {'Hedef':>8} {'Ratio':>6} {'Tarih':>6}")
        print(f"  {'─' * 75}")
        for s in filtered:
            ratio = s['details'].get('ratio', 0)
            print(f"  {s['ticker']:<8} {s['close']:>8.2f} {s['key_level']:>8.2f} "
                  f"{s['quality']:>7} {s['stop']:>8.2f} {s['target']:>8.2f} "
                  f"{ratio:>6.3f} {s['date']:>6}")

    elif group_key == 'COMPRESSION':
        print(f"  {'Hisse':<8} {'Fiyat':>8} {'Range':>8} {'Kalite':>7} "
              f"{'Yon':<8} {'Bars':>5} {'Tarih':>6}")
        print(f"  {'─' * 75}")
        for s in filtered:
            yon = 'YUKARI' if s['direction'] == 'BUY' else 'ASAGI'
            comp_range = s['details'].get('range', 0)
            comp_bars = s['details'].get('comp_bars', 0)
            print(f"  {s['ticker']:<8} {s['close']:>8.2f} {comp_range:>8.2f} "
                  f"{s['quality']:>7} {yon:<8} {comp_bars:>5} {s['date']:>6}")

    elif group_key == 'CANCAN':
        print(f"  {'Hisse':<8} {'Fiyat':>8} {'Seviye':>8} {'Kalite':>7} "
              f"{'Stop':>8} {'Hedef':>8} {'Flip':<5} {'Tarih':>6}")
        print(f"  {'─' * 75}")
        for s in filtered:
            flip = s['details'].get('flip', '?')
            print(f"  {s['ticker']:<8} {s['close']:>8.2f} {s['key_level']:>8.2f} "
                  f"{s['quality']:>7} {s['stop']:>8.2f} {s['target']:>8.2f} "
                  f"{flip:<5} {s['date']:>6}")

    elif group_key == '2RS':
        print(f"  {'Hisse':<8} {'Fiyat':>8} {'Seviye':>8} {'Kalite':>7} "
              f"{'Stop':>8} {'Hedef':>8} {'Touch':>6} {'Tarih':>6}")
        print(f"  {'─' * 75}")
        for s in filtered:
            touch = s['details'].get('touch_count', 0)
            print(f"  {s['ticker']:<8} {s['close']:>8.2f} {s['key_level']:>8.2f} "
                  f"{s['quality']:>7} {s['stop']:>8.2f} {s['target']:>8.2f} "
                  f"{touch:>6} {s['date']:>6}")

    print(f"  {'─' * 75}")


def _print_results(all_signals, n_scanned, date_str, pattern_filter=None, top_n=None):
    """Konsol rapor."""
    w = 75

    # Signal row'lari olustur
    rows = []
    for item in all_signals:
        row = _format_signal_row(
            item['ticker'], item['close'], item['signal'], item['signal_date']
        )
        rows.append(row)

    # Pattern filtresi
    if pattern_filter:
        pf = pattern_filter.upper()
        filtered_patterns = []
        for gk, gv in PATTERN_GROUPS.items():
            if pf in gk or any(pf in p for p in gv['patterns']):
                filtered_patterns.extend(gv['patterns'])
        if filtered_patterns:
            rows = [r for r in rows if r['pattern'] in filtered_patterns]

    # Top N
    if top_n and top_n > 0:
        rows.sort(key=lambda r: r['quality'], reverse=True)
        rows = rows[:top_n]

    buy_rows = [r for r in rows if r['direction'] == 'BUY']
    sell_rows = [r for r in rows if r['direction'] == 'SELL']

    print(f"\n{'═' * w}")
    print(f"  NOX SMC PATTERN SCREENER — {date_str} — {n_scanned} hisse tarandi")
    print(f"{'═' * w}")

    # Pattern bazinda gruplama
    for gk, gv in PATTERN_GROUPS.items():
        group_rows = [r for r in rows if r['pattern'] in gv['patterns']]
        if not group_rows:
            continue
        _print_pattern_group(gk, gv, group_rows, 'BUY')
        _print_pattern_group(gk, gv, group_rows, 'SELL')

    # Ozet
    pattern_counts = {}
    for r in rows:
        p = r['pattern']
        pattern_counts[p] = pattern_counts.get(p, 0) + 1

    counts_str = ', '.join(f'{c} {p}' for p, c in sorted(pattern_counts.items(), key=lambda x: -x[1]))

    print(f"\n{'═' * w}")
    print(f"  OZET: {len(buy_rows)} AL + {len(sell_rows)} SAT sinyal ({counts_str})")
    print(f"{'═' * w}")


# =============================================================================
# CSV
# =============================================================================

def _save_csv(all_signals, date_str, output_dir):
    """Sinyalleri CSV dosyasina kaydet."""
    os.makedirs(output_dir, exist_ok=True)
    rows = []
    for item in all_signals:
        sig = item['signal']
        sig_date = item['signal_date']
        rows.append({
            'ticker': item['ticker'],
            'close': round(item['close'], 4),
            'pattern': sig.pattern,
            'direction': sig.direction,
            'key_level': round(sig.key_level, 4),
            'quality': sig.quality,
            'stop': round(sig.stop, 4),
            'target': round(sig.target, 4),
            'signal_date': sig_date.strftime('%Y-%m-%d') if hasattr(sig_date, 'strftime') else str(sig_date),
        })

    if rows:
        csv_df = pd.DataFrame(rows)
        csv_df.sort_values(['quality', 'ticker'], ascending=[False, True], inplace=True)
        fname = f"nox_smc_signals_{date_str.replace('-', '')}.csv"
        path = os.path.join(output_dir, fname)
        csv_df.to_csv(path, index=False)
        print(f"\n  CSV: {path}")
    else:
        print(f"\n  Sinyal yok, CSV olusturulmadi.")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="NOX SMC Pattern Screener")
    parser.add_argument('tickers', nargs='*', help='Spesifik ticker(lar)')
    parser.add_argument('--period', default='2y', help='Veri periyodu (default: 2y)')
    parser.add_argument('--debug', metavar='TICKER', help='Tek hisse detayli debug')
    parser.add_argument('--top', type=int, metavar='N', help='En iyi N sinyal')
    parser.add_argument('--pattern', metavar='PATTERN', help='Pattern filtresi (QM, FAKEOUT, FLAG, ...)')
    parser.add_argument('--csv', action='store_true', help='CSV kaydet')
    parser.add_argument('--output', default='output', help='CSV cikti dizini')
    parser.add_argument('--scan-bars', type=int, default=15, help='Son kac bar taranacak (default: 15)')
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
    stock_dfs = {ticker: _to_lower_cols(df) for ticker, df in all_data.items()}

    # ── 4. Tarama ────────────────────────────────────────────────────────────
    print(f"\n  SMC pattern taramasi (scan_bars={args.scan_bars})...")
    t1 = time.time()
    all_signals, n_scanned, date_str = _scan_all(
        stock_dfs, debug_ticker=debug_ticker, scan_bars=args.scan_bars
    )
    print(f"  {n_scanned} hisse tarandi, {len(all_signals)} sinyal ({time.time() - t1:.1f}s)")

    # ── 5. Rapor ─────────────────────────────────────────────────────────────
    _print_results(all_signals, n_scanned, date_str,
                   pattern_filter=args.pattern, top_n=args.top)

    # ── 6. CSV ───────────────────────────────────────────────────────────────
    if args.csv:
        _save_csv(all_signals, date_str, args.output)

    print(f"\n  Toplam sure: {time.time() - t0:.1f}s")
    print(f"  NOX SMC tamamlandi.")


if __name__ == '__main__':
    main()
