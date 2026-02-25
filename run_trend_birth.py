#!/usr/bin/env python3
"""
NOX Trend Birth Screener — Runner
===================================
BIST hisselerinde trend dogum (erken trend) taramasi.
Katmanli sinyal sistemi: Hazirlik → Tetik → Teyit → Cikis.

Kullanim:
    python run_trend_birth.py                        # tum hisseler
    python run_trend_birth.py --tickers THYAO ASELS  # spesifik hisseler
    python run_trend_birth.py --csv                  # CSV kaydet
    python run_trend_birth.py --scan-bars 10         # son 10 bar tara
"""

import argparse
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

from markets.bist import data as data_mod
from markets.bist.trend_birth import (
    scan_trend_birth, TrendBirthSignal, TB_CFG,
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
    """Son n bar H==L ise halt/suspend."""
    if len(df) < n:
        return False
    tail = df.tail(n)
    return (tail['high'] == tail['low']).all()


# =============================================================================
# TARAMA
# =============================================================================

def _scan_all(stock_dfs, scan_bars=5):
    """Tum hisselerde trend birth taramasi."""
    results = {
        'prep': [],      # Hazirlik asmasinda (prep >= 40, tetik yok)
        'trigger': [],   # Tetik var (trigger_count >= 2)
        'confirmed': [], # Teyitli (confirmed == True)
    }
    n_scanned = 0
    last_date = None

    for ticker, df in stock_dfs.items():
        if len(df) < 150:  # Yeterli veri gerekli (BB pctile 120 + buffer)
            continue
        if _is_halted(df):
            continue

        try:
            out = scan_trend_birth(df)
            n_scanned += 1

            if last_date is None and len(df) > 0:
                last_date = df.index[-1]

            # Son scan_bars bar icerisinde sinyal ara
            tail_start = max(0, len(df) - scan_bars)
            for i in range(tail_start, len(df)):
                direction = out['direction'].iloc[i]
                trigger_count = int(out['trigger_count'].iloc[i])
                prep_score = int(out['prep_score'].iloc[i])
                confirmed = bool(out['confirmed'].iloc[i])

                # Trigger listesi olustur
                trig_list = []
                if direction == 'AL':
                    if out['trig_hist_slope_bull'].iloc[i]:
                        trig_list.append('HIST_SLOPE')
                    if out['trig_obv_bull'].iloc[i]:
                        trig_list.append('OBV_TURN')
                    if out['trig_choch_bull'].iloc[i]:
                        trig_list.append('CHOCH')
                elif direction == 'SAT':
                    if out['trig_hist_slope_bear'].iloc[i]:
                        trig_list.append('HIST_SLOPE')
                    if out['trig_obv_bear'].iloc[i]:
                        trig_list.append('OBV_TURN')
                    if out['trig_choch_bear'].iloc[i]:
                        trig_list.append('CHOCH')
                if out['trig_vol_spike'].iloc[i]:
                    trig_list.append('VOL_SPIKE')

                # Degerler
                adx_val = out['adx'].iloc[i]
                rsi_val = out['rsi'].iloc[i]
                atr_val = out['atr'].iloc[i]
                hist_val = out['histogram'].iloc[i]
                close_val = df['close'].iloc[i]
                stop_val = out['stop'].iloc[i]
                trail_val = out['trailing_stop_atr'].iloc[i]
                sig_date = df.index[i]

                signal = TrendBirthSignal(
                    ticker=ticker,
                    date=sig_date,
                    direction=direction if direction else 'AL',
                    close=float(close_val),
                    prep_score=prep_score,
                    squeeze_active=bool(out['squeeze_active'].iloc[i]),
                    volume_dryup=bool(out['volume_dryup'].iloc[i]),
                    adx_declining=bool(out['adx_declining'].iloc[i]),
                    trigger_count=trigger_count,
                    triggers=trig_list,
                    confirmed=confirmed,
                    adx_rising=bool(out['adx_rising'].iloc[i]),
                    rsi_above_50=bool(out['rsi_above_50'].iloc[i]),
                    stop=float(stop_val) if pd.notna(stop_val) else 0.0,
                    trailing_stop_atr=float(trail_val) if pd.notna(trail_val) else 0.0,
                    adx=float(adx_val) if pd.notna(adx_val) else 0.0,
                    rsi=float(rsi_val) if pd.notna(rsi_val) else 0.0,
                    atr=float(atr_val) if pd.notna(atr_val) else 0.0,
                    macd_hist=float(hist_val) if pd.notna(hist_val) else 0.0,
                )

                # Sinyal kategorize et
                if trigger_count >= TB_CFG['min_triggers'] and direction:
                    if confirmed:
                        results['confirmed'].append(signal)
                    else:
                        results['trigger'].append(signal)
                elif prep_score >= TB_CFG['prep_gate'] and trigger_count < TB_CFG['min_triggers']:
                    results['prep'].append(signal)

        except Exception as e:
            print(f"  ! {ticker}: {e}")
            continue

    date_str = last_date.strftime('%Y-%m-%d') if last_date else datetime.now().strftime('%Y-%m-%d')

    # Ayni ticker icin en gunceli tut
    for key in results:
        seen = {}
        for sig in results[key]:
            if sig.ticker not in seen or sig.date > seen[sig.ticker].date:
                seen[sig.ticker] = sig
        results[key] = sorted(seen.values(), key=lambda s: s.prep_score, reverse=True)

    return results, n_scanned, date_str


# =============================================================================
# KONSOL RAPOR
# =============================================================================

def _bool_mark(val):
    return '+' if val else '-'


def _print_prep_table(signals, w=80):
    """Hazirlik tablosu — prep_score >= 40 ama henuz tetik yok."""
    if not signals:
        return
    print(f"\n  {'▸'} HAZIRLIK — Bir Sey Pisiyor ({len(signals)} hisse)")
    print(f"  {'─' * w}")
    print(f"  {'Hisse':<8} {'Fiyat':>8} {'Prep':>5} {'Squeeze':>8} "
          f"{'VolDry':>7} {'ADXdwn':>7} {'ADX':>6} {'RSI':>6}")
    print(f"  {'─' * w}")
    for s in signals:
        print(f"  {s.ticker:<8} {s.close:>8.2f} {s.prep_score:>5} "
              f"{'  ' + _bool_mark(s.squeeze_active):>8} "
              f"{'  ' + _bool_mark(s.volume_dryup):>7} "
              f"{'  ' + _bool_mark(s.adx_declining):>7} "
              f"{s.adx:>6.1f} {s.rsi:>6.1f}")
    print(f"  {'─' * w}")


def _print_trigger_table(signals, w=80):
    """Tetik tablosu — trigger_count >= 2."""
    if not signals:
        return
    print(f"\n  {'▸'} TETIK — Giris Sinyali ({len(signals)} hisse)")
    print(f"  {'─' * w}")
    print(f"  {'Hisse':<8} {'Fiyat':>8} {'Yon':<4} {'Trig':>4} "
          f"{'Prep':>5} {'Detay':<28} {'Stop':>8} {'Tarih':>6}")
    print(f"  {'─' * w}")
    for s in signals:
        date_str = s.date.strftime('%m-%d') if hasattr(s.date, 'strftime') else str(s.date)[-5:]
        trig_str = '+'.join(s.triggers) if s.triggers else '-'
        print(f"  {s.ticker:<8} {s.close:>8.2f} {s.direction:<4} {s.trigger_count:>4} "
              f"{s.prep_score:>5} {trig_str:<28} {s.stop:>8.2f} {date_str:>6}")
    print(f"  {'─' * w}")


def _print_confirmed_table(signals, w=80):
    """Teyitli tablosu — confirmed == True."""
    if not signals:
        return
    print(f"\n  {'▸'} TEYITLI — Pozisyonu Koru ({len(signals)} hisse)")
    print(f"  {'─' * w}")
    print(f"  {'Hisse':<8} {'Fiyat':>8} {'Yon':<4} {'Trig':>4} "
          f"{'ADX+':>5} {'RSI+':>5} {'ADX':>6} {'RSI':>6} {'Stop':>8} {'Trail':>7} {'Tarih':>6}")
    print(f"  {'─' * w}")
    for s in signals:
        date_str = s.date.strftime('%m-%d') if hasattr(s.date, 'strftime') else str(s.date)[-5:]
        print(f"  {s.ticker:<8} {s.close:>8.2f} {s.direction:<4} {s.trigger_count:>4} "
              f"{'  ' + _bool_mark(s.adx_rising):>5} "
              f"{'  ' + _bool_mark(s.rsi_above_50):>5} "
              f"{s.adx:>6.1f} {s.rsi:>6.1f} {s.stop:>8.2f} {s.trailing_stop_atr:>7.2f} "
              f"{date_str:>6}")
    print(f"  {'─' * w}")


def _print_results(results, n_scanned, date_str):
    """Konsol rapor."""
    w = 80
    print(f"\n{'═' * w}")
    print(f"  NOX TREND BIRTH SCREENER — {date_str} — {n_scanned} hisse tarandi")
    print(f"{'═' * w}")

    _print_confirmed_table(results['confirmed'], w)
    _print_trigger_table(results['trigger'], w)
    _print_prep_table(results['prep'], w)

    n_conf = len(results['confirmed'])
    n_trig = len(results['trigger'])
    n_prep = len(results['prep'])

    print(f"\n{'═' * w}")
    print(f"  OZET: {n_conf} teyitli + {n_trig} tetik + {n_prep} hazirlik")
    print(f"{'═' * w}")


# =============================================================================
# CSV
# =============================================================================

def _save_csv(results, date_str, output_dir):
    """Sinyalleri CSV'ye kaydet."""
    os.makedirs(output_dir, exist_ok=True)
    rows = []
    for category, signals in results.items():
        for s in signals:
            rows.append({
                'ticker': s.ticker,
                'date': s.date.strftime('%Y-%m-%d') if hasattr(s.date, 'strftime') else str(s.date),
                'category': category,
                'direction': s.direction,
                'close': round(s.close, 2),
                'prep_score': s.prep_score,
                'squeeze': s.squeeze_active,
                'vol_dryup': s.volume_dryup,
                'adx_declining': s.adx_declining,
                'trigger_count': s.trigger_count,
                'triggers': '+'.join(s.triggers),
                'confirmed': s.confirmed,
                'adx_rising': s.adx_rising,
                'rsi_above_50': s.rsi_above_50,
                'adx': round(s.adx, 2),
                'rsi': round(s.rsi, 2),
                'atr': round(s.atr, 4),
                'stop': round(s.stop, 2),
                'trailing_atr': round(s.trailing_stop_atr, 4),
                'macd_hist': round(s.macd_hist, 6),
            })

    if rows:
        csv_df = pd.DataFrame(rows)
        fname = f"nox_trend_birth_{date_str.replace('-', '')}.csv"
        path = os.path.join(output_dir, fname)
        csv_df.to_csv(path, index=False)
        print(f"\n  CSV: {path}")
    else:
        print(f"\n  Sinyal yok, CSV olusturulmadi.")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="NOX Trend Birth Screener")
    parser.add_argument('--tickers', nargs='*', help='Spesifik ticker(lar)')
    parser.add_argument('--period', default='1y', help='Veri periyodu (default: 1y)')
    parser.add_argument('--scan-bars', type=int, default=5, help='Son kac bar taranacak (default: 5)')
    parser.add_argument('--csv', action='store_true', help='CSV kaydet')
    parser.add_argument('--output', default='output', help='CSV cikti dizini')
    args = parser.parse_args()

    w = 80
    print(f"\n{'═' * w}")
    print(f"  NOX TREND BIRTH SCREENER")
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

    # ── 4. Tarama ────────────────────────────────────────────────────────────
    print(f"\n  Trend birth taramasi (scan_bars={args.scan_bars})...")
    t1 = time.time()
    results, n_scanned, date_str = _scan_all(stock_dfs, scan_bars=args.scan_bars)
    print(f"  {n_scanned} hisse tarandi ({time.time() - t1:.1f}s)")

    # ── 5. Rapor ─────────────────────────────────────────────────────────────
    _print_results(results, n_scanned, date_str)

    # ── 6. CSV ───────────────────────────────────────────────────────────────
    if args.csv:
        _save_csv(results, date_str, args.output)

    print(f"\n  Toplam sure: {time.time() - t0:.1f}s")
    print(f"  NOX Trend Birth tamamlandi.\n")


if __name__ == '__main__':
    main()
