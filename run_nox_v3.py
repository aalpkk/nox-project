#!/usr/bin/env python3
"""
NOX v3 PIVOT AL/SAT Screener
=============================
Gunluk + Haftalik pivot tarama, tek seferde.
Son 10 barda (2*lb) onaylanan sinyalleri gosterir = TV'de gorunen yakin elmaslar.
Kirilmis pivotlar filtrelenir (fiyat pivot seviyesinin altina dustuyse).
Halt/suspend hisseler atlanir (Yahoo sahte bar tespit).

Kullanim:
    python run_nox_v3.py              # gunluk + haftalik
    python run_nox_v3.py --daily      # sadece gunluk
    python run_nox_v3.py --weekly     # sadece haftalik
    python run_nox_v3.py --csv
    python run_nox_v3.py --html
    python run_nox_v3.py --debug THYAO
    python run_nox_v3.py THYAO EREGL SISE
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
from markets.bist.nox_v3_signals import compute_nox_v3, detect_daily_triggers, calc_adx_with_di, _pine_rsi, calc_rs
from markets.bist.trend_birth import scan_trend_birth
from core.reports import _NOX_CSS, _sanitize, send_telegram, send_telegram_document, push_html_to_github


# =============================================================================
# YARDIMCI
# =============================================================================

def _get_past_fridays(n, last_date):
    """Son n Cumayi dondur (eskiden yeniye).
    last_date: verinin son gununun date objesi."""
    from datetime import timedelta
    d = last_date
    # Cuma = 4 (weekday)
    days_since_friday = (d.weekday() - 4) % 7
    last_fri = d - timedelta(days=days_since_friday)
    fridays = []
    for i in range(n):
        fridays.append(last_fri - timedelta(weeks=i))
    return sorted(fridays)  # eskiden yeniye

def _to_lower_cols(df):
    return df.rename(columns={
        'Open': 'open', 'High': 'high', 'Low': 'low',
        'Close': 'close', 'Volume': 'volume',
    })


def _is_halted(df, n=3):
    """Son n bar tamami H==L (sifir range) ise hisse halt/suspend durumunda.
    Yahoo halt gunlerinde sahte bar uretir (O=H=L=C ayni fiyat).
    TV'de bu barlar yok, pivot hesaplarini bozuyor."""
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
    # Son mum yarimsa at: W-FRI label = haftanin Cumasi.
    # Veri o Cumaya kadar ulasmamissa mum yarimdir.
    # Ancak tatil/yari gun durumunda (orn: Cuma tatil, Persembe son islem gunu)
    # mum yeterince tamamdir — Persembe veya Cuma verisi varsa kabul et.
    if len(weekly) > 0:
        last_friday = weekly.index[-1]
        last_data_date = df.index[-1]
        if last_data_date < last_friday:
            # Haftanin Persembe veya sonrasiysa mum yeterli (tatil/yari gun)
            days_before_friday = (last_friday - last_data_date).days
            if days_before_friday > 1:  # Carsamba veya oncesi = gercekten yarim
                weekly = weekly.iloc[:-1]
    return weekly


# =============================================================================
# TARAMA
# =============================================================================

def _scan(stock_dfs, debug_ticker=None, scan_bars=10, tf_label='daily'):
    """
    Tum hisselerde pivot tarama yap.
    Son `scan_bars` onay barini kontrol eder (varsayilan=2*lb=10).
    Bu, TV'de gorunen son ~2 haftanin elmaslarini yakalar.
    Kirilmis pivotlar filtrelenir (fiyat pivot seviyesinin altina dustuyse).
    Ticker basina sadece en son sinyal raporlanir.
    Ayrica ADAY (onaylanmamis) pivot low adaylarini da tarar.
    tf_label='weekly' ise haftalik momentum alanlari hesaplanir.
    """
    buys = []
    sells = []
    candidates = []  # ADAY: onaylanmamis pivot low adaylari
    n_scanned = 0
    last_date = None
    lb = 5

    for ticker, df in stock_dfs.items():
        if len(df) < 60:
            continue
        if _is_halted(df):
            continue
        try:
            result = compute_nox_v3(df, require_gate=False, min_sell_severity=0)
            n_scanned += 1

            last = len(df) - 1
            if last_date is None:
                last_date = df.index[-1]

            # Haftalik momentum alanlari
            wk_mom = False
            wk_rsi_up = False
            if tf_label == 'weekly' and last >= 1:
                wk_mom = float(result['close'].iloc[last]) > float(result['close'].iloc[last - 1])
                rsi_arr = result['rsi']
                wk_rsi_up = float(rsi_arr.iloc[last]) > float(rsi_arr.iloc[last - 1])

            if debug_ticker and ticker == debug_ticker:
                _print_debug(ticker, result)

            scan_start = max(2 * lb, last - scan_bars + 1)
            current_close = result['close'].iloc[last]

            # — AL: en son onaylanan ve KIRILMAMIS pivot low —
            found_buy = False
            for i in range(last, scan_start - 1, -1):
                if result['pivot_buy'].iloc[i]:
                    pivot_price = result['pivot_low_price'].iloc[i]
                    # Pivot kirildi mi? (fiyat pivot seviyesinin altina dustu mu?)
                    # Son bardan pivot barına kadar min low kontrol et
                    pivot_bar_i = i - lb
                    bars_after = df['low'].iloc[i:last + 1]
                    if bars_after.min() < pivot_price:
                        continue  # kirilmis pivot, atla
                    pivot_date = df.index[pivot_bar_i].strftime('%Y-%m-%d') if pivot_bar_i >= 0 else '?'
                    signal_date = df.index[i].strftime('%Y-%m-%d')
                    fresh = 'BUGUN' if i == last else 'YAKIN'
                    delta_pct = (current_close - pivot_price) / pivot_price * 100
                    buys.append({
                        'ticker': ticker,
                        'close': current_close,
                        'pivot': pivot_price,
                        'delta_pct': round(delta_pct, 1),
                        'br': result['br_score'].iloc[i],
                        'rg': result['rg_score'].iloc[i],
                        'gate': bool(result['gate_open'].iloc[i]),
                        'adx': result['adx'].iloc[last],
                        'slope': result['adx_slope'].iloc[last],
                        'rsi': result['rsi'].iloc[last],
                        'signal_date': signal_date,
                        'pivot_date': pivot_date,
                        'fresh': fresh,
                        'wk_mom': wk_mom,
                        'wk_rsi_up': wk_rsi_up,
                    })
                    found_buy = True
                    break  # ticker basina en son gecerli sinyal

            # — ADAY: onaylanmamis pivot low adaylari —
            # Son lb barda (onay gelmemis bolge) olusmus potansiyel pivot low
            if not found_buy:
                zone_start = max(0, last - 2 * lb)
                zone_lows = df['low'].iloc[zone_start:last + 1]
                min_pos = int(zone_lows.values.argmin()) + zone_start
                # Sadece onaylanmamis bolgede ise (son lb bar)
                if min_pos > last - lb:
                    candidate_price = df['low'].iloc[min_pos]
                    # Sol taraf: lookback'ten dusuk olmali
                    left_start = max(0, min_pos - lb)
                    left_ok = (min_pos == left_start or
                               candidate_price <= df['low'].iloc[left_start:min_pos].min())
                    # Sag taraf: sonraki barlar kirilmamis olmali
                    right_ok = (min_pos == last or
                                candidate_price <= df['low'].iloc[min_pos + 1:last + 1].min())
                    if left_ok and right_ok:
                        delta_pct = (current_close - candidate_price) / candidate_price * 100
                        bars_since = last - min_pos
                        candidates.append({
                            'ticker': ticker,
                            'close': current_close,
                            'pivot': candidate_price,
                            'delta_pct': round(delta_pct, 1),
                            'bars_since': bars_since,
                            'rg': result['rg_score'].iloc[last],
                            'gate': bool(result['gate_open'].iloc[last]),
                            'adx': result['adx'].iloc[last],
                            'slope': result['adx_slope'].iloc[last],
                            'rsi': result['rsi'].iloc[last],
                            'pivot_date': df.index[min_pos].strftime('%Y-%m-%d'),
                            'wk_mom': wk_mom,
                            'wk_rsi_up': wk_rsi_up,
                        })

            # — SAT: en son onaylanan pivot high —
            for i in range(last, scan_start - 1, -1):
                if result['pivot_sell'].iloc[i]:
                    pivot_bar_i = i - lb
                    pivot_date = df.index[pivot_bar_i].strftime('%Y-%m-%d') if pivot_bar_i >= 0 else '?'
                    signal_date = df.index[i].strftime('%Y-%m-%d')
                    fresh = 'BUGUN' if i == last else 'YAKIN'
                    sells.append({
                        'ticker': ticker,
                        'close': current_close,
                        'pivot': result['pivot_high_price'].iloc[i],
                        'severity': int(result['sell_severity'].iloc[i]),
                        'adx': result['adx'].iloc[last],
                        'slope': result['adx_slope'].iloc[last],
                        'rsi': result['rsi'].iloc[last],
                        'dd_pct': result['drawdown_pct'].iloc[last],
                        'signal_date': signal_date,
                        'pivot_date': pivot_date,
                        'fresh': fresh,
                    })
                    break
        except Exception as e:
            print(f"  ! {ticker}: {e}")
            continue

    # Siralama
    buys.sort(key=lambda x: (x['fresh'] != 'BUGUN', not x['gate'], x['delta_pct']))
    sells.sort(key=lambda x: (x['fresh'] != 'BUGUN', -x['severity'], x['dd_pct']))
    candidates.sort(key=lambda x: (-x['bars_since'], x['delta_pct']))

    date_str = last_date.strftime('%Y-%m-%d') if last_date else datetime.now().strftime('%Y-%m-%d')
    return buys, sells, candidates, n_scanned, date_str


# =============================================================================
# HAFTALIK ZENGINLESTIRME — Trend Birth + Durum
# =============================================================================

def _enrich_with_trend_birth(signals, stock_dfs, ref_date_str=None):
    """Haftalik AL/ADAY sinyallerini gunluk Trend Birth verisiyle zenginlestirir.
    Her sinyal icin tb_stage (OK/TRG/PREP/-) ve status (HAZIR/IZLE/BEKLE) ekler.
    HAZIR icin sinyal son 4 hafta icerisinde olmali."""
    from datetime import timedelta
    if ref_date_str:
        ref_date = datetime.strptime(ref_date_str, '%Y-%m-%d')
    else:
        ref_date = datetime.now()
    max_age = timedelta(weeks=4)

    for sig in signals:
        ticker = sig['ticker']
        df = stock_dfs.get(ticker)
        if df is None or len(df) < 60:
            sig.update(tb_stage='-', tb_prep=0, tb_triggers=0, status='BEKLE')
            continue
        try:
            tb = scan_trend_birth(df)
            last = len(df) - 1
            prep = int(tb['prep_score'].iloc[last])
            trig = int(tb['trigger_count'].iloc[last])
            conf = bool(tb['confirmed'].iloc[last])

            stage = 'OK' if conf else 'TRG' if trig >= 2 else 'PREP' if prep >= 40 else '-'

            # Tazelik kontrolu: sinyal tarihi son 4 hafta icerisinde mi?
            date_str = sig.get('signal_date') or sig.get('pivot_date', '')
            try:
                sig_date = datetime.strptime(date_str, '%Y-%m-%d')
                is_fresh = (ref_date - sig_date) <= max_age
            except (ValueError, TypeError):
                is_fresh = False

            wk_mom = sig.get('wk_mom', False)
            delta = sig.get('delta_pct', 999)
            if wk_mom and stage in ('TRG', 'OK') and delta <= 20 and is_fresh:
                status = 'HAZIR'
            elif wk_mom:
                status = 'İZLE'
            else:
                status = 'BEKLE'

            sig.update(tb_stage=stage, tb_prep=prep, tb_triggers=trig, status=status)
        except Exception:
            sig.update(tb_stage='-', tb_prep=0, tb_triggers=0, status='BEKLE')
    return signals


# =============================================================================
# GUNLUK TETIK — Haftalik pivot + gunluk tetik iki katmanli sistem
# =============================================================================

def _apply_daily_triggers(weekly_buys, weekly_candidates, daily_dfs,
                          max_delta_pct=15.0, xu_df=None):
    """
    Her haftalik AL/ADAY sinyali icin gunluk tetik ara.
    Tetik bulunursa → triggered, bulunamazsa → zone_only.
    xu_df verilirse tetik barindaki RS hesaplanir.

    Returns: (triggered, zone_only)
    """
    # XU100 close'u lowercase'e çevir (varsa)
    xu_close = None
    if xu_df is not None and not xu_df.empty:
        if 'close' in xu_df.columns:
            xu_close = xu_df['close']
        elif 'Close' in xu_df.columns:
            xu_close = xu_df['Close']

    triggered = []
    zone_only = []

    for sig in weekly_buys + weekly_candidates:
        ticker = sig['ticker']
        daily_df = daily_dfs.get(ticker)
        if daily_df is None or len(daily_df) < 30:
            zone_only.append(sig)
            continue

        # Pivot confirm date = signal_date (haftalik onay bari)
        confirm_date = sig.get('signal_date') or sig.get('pivot_date', '')
        pivot_price = sig['pivot']

        result = detect_daily_triggers(
            daily_df, pivot_price, confirm_date, max_delta_pct
        )

        if result['triggered']:
            # Tetik bulundu: signal_date ve close'u tetik gununun degerleriyle guncelle
            sig['trigger_type'] = result['trigger_type']
            sig['trigger_date'] = result['trigger_date']
            sig['orig_signal_date'] = sig.get('signal_date', sig.get('pivot_date', ''))
            sig['signal_date'] = result['trigger_date']
            sig['close'] = result['trigger_close']
            sig['delta_pct_at_trigger'] = result['delta_pct_at_trigger']
            # fresh etiketini tetik tarihine göre güncelle
            last_trading_day = daily_df.index[-1].strftime('%Y-%m-%d')
            sig['fresh'] = 'BUGUN' if result['trigger_date'] == last_trading_day else 'YAKIN'
            # RS hesapla
            _add_rs(sig, daily_df, result['trigger_date'], xu_close)
            # Gunluk indikatörler
            _add_daily_indicators(sig, daily_df, result['trigger_date'])
            triggered.append(sig)
        else:
            zone_only.append(sig)

    return triggered, zone_only


def _add_rs(entry, daily_df, trigger_date_str, xu_close):
    """Tetik barindaki RS (Relative Strength vs XU100) hesapla."""
    if xu_close is None:
        return
    try:
        trigger_ts = pd.Timestamp(trigger_date_str)
        rs_series = calc_rs(daily_df['close'], xu_close, period=20)
        idx = daily_df.index.searchsorted(trigger_ts)
        if idx >= len(daily_df):
            idx = len(daily_df) - 1
        val = rs_series.iloc[idx]
        if pd.notna(val):
            entry['rs_score'] = round(float(val), 3)
    except Exception:
        pass


def _add_daily_indicators(entry, daily_df, trigger_date_str):
    """Tetik barindaki gunluk ADX, ADX slope, RSI hesapla."""
    try:
        trigger_ts = pd.Timestamp(trigger_date_str)
        idx = daily_df.index.get_loc(trigger_ts)
    except (KeyError, TypeError):
        # Tam eslesme yoksa en yakin bari bul
        try:
            trigger_ts = pd.Timestamp(trigger_date_str)
            idx = daily_df.index.searchsorted(trigger_ts)
            if idx >= len(daily_df):
                idx = len(daily_df) - 1
        except Exception:
            return

    try:
        adx, _, _ = calc_adx_with_di(daily_df)
        rsi = _pine_rsi(daily_df['close'], 14)
        adx_slope = (adx - adx.shift(5)) / 5

        entry['d_adx'] = round(float(adx.iloc[idx]), 1) if pd.notna(adx.iloc[idx]) else None
        entry['d_adx_slope'] = round(float(adx_slope.iloc[idx]), 2) if pd.notna(adx_slope.iloc[idx]) else None
        entry['d_rsi'] = round(float(rsi.iloc[idx]), 1) if pd.notna(rsi.iloc[idx]) else None
    except Exception:
        entry['d_adx'] = None
        entry['d_adx_slope'] = None
        entry['d_rsi'] = None


def _save_csv_v2(triggered, sells, date_str, output_dir, suffix='',
                 zone_only=None):
    """
    Yeni CSV kaydet: triggered → PIVOT_AL, zone_only → ZONE_ONLY, sells → PIVOT_SAT.
    Ayni dosya adi pattern'i: nox_v3_signals{suffix}_{YYYYMMDD}.csv
    """
    os.makedirs(output_dir, exist_ok=True)
    rows = []
    for b in triggered:
        row = {
            'signal': 'PIVOT_AL',
            'ticker': b['ticker'],
            'pivot_date': b.get('pivot_date', ''),
            'signal_date': b['signal_date'],
            'close': b['close'],
            'pivot_price': b['pivot'],
            'delta_pct': b.get('delta_pct'),
            'gate_open': b.get('gate'),
            'rg_score': b.get('rg'),
            'adx': b.get('adx'),
            'adx_slope': b.get('slope'),
            'rsi': b.get('rsi'),
            'trigger_type': b.get('trigger_type', ''),
            'trigger_date': b.get('trigger_date', ''),
            'delta_pct_at_trigger': b.get('delta_pct_at_trigger'),
            'd_adx': b.get('d_adx'),
            'd_adx_slope': b.get('d_adx_slope'),
            'd_rsi': b.get('d_rsi'),
            'rs_score': b.get('rs_score'),
            'wl_status': b.get('status', ''),
            'tb_stage': b.get('tb_stage', ''),
            'tb_prep': b.get('tb_prep', ''),
        }
        rows.append(row)
    for z in (zone_only or []):
        row = {
            'signal': 'ZONE_ONLY',
            'ticker': z['ticker'],
            'pivot_date': z.get('pivot_date', ''),
            'signal_date': z.get('signal_date', z.get('pivot_date', '')),
            'close': z['close'],
            'pivot_price': z['pivot'],
            'delta_pct': z.get('delta_pct'),
            'gate_open': z.get('gate'),
            'rg_score': z.get('rg'),
            'adx': z.get('adx'),
            'adx_slope': z.get('slope'),
            'rsi': z.get('rsi'),
            'wl_status': z.get('status', ''),
            'tb_stage': z.get('tb_stage', ''),
            'tb_prep': z.get('tb_prep', ''),
        }
        rows.append(row)
    for s in sells:
        rows.append({
            'signal': 'PIVOT_SAT',
            'ticker': s['ticker'],
            'pivot_date': s['pivot_date'],
            'signal_date': s['signal_date'],
            'close': s['close'],
            'pivot_price': s['pivot'],
            'severity': s['severity'],
            'adx': s['adx'],
            'adx_slope': s['slope'],
            'rsi': s['rsi'],
            'drawdown_pct': s['dd_pct'],
        })
    if rows:
        csv_df = pd.DataFrame(rows)
        fname = f"nox_v3_signals{suffix}_{date_str.replace('-', '')}.csv"
        path = os.path.join(output_dir, fname)
        csv_df.to_csv(path, index=False)
        print(f"\n  CSV: {path}")


# =============================================================================
# RAPOR
# =============================================================================

def _print_triggered_results(triggered, sells, zone_only, n_scanned, date_str):
    """Tetik bazli konsol ciktisi — haftalik iki katmanli sistem icin."""
    _print_section(f"◆ NOX v3 [HAFTALIK+TETIK] — {date_str} — {n_scanned} hisse")

    if triggered:
        # Gate durumuna gore ayir
        gated = [t for t in triggered if t.get('gate')]
        ungated = [t for t in triggered if not t.get('gate')]

        for label, items in [("◆ PIVOT AL+TETIK — ONAYLI (Gate Acik)", gated),
                             ("◆ PIVOT AL+TETIK — Sadece Pivot (Gate Kapali)", ungated)]:
            if not items:
                print(f"\n  {label} (0)")
                continue
            print(f"\n  {label} ({len(items)})")
            w = 130
            print(f"  {'─' * w}")
            print(f"  {'Hisse':<8} {'TetikTrh':>10} {'Tetik':>5} {'Fiyat':>8} {'DipFiy':>8} {'Δ%':>6} "
                  f"{'RG':>5} {'dADX':>5} {'dSlp':>6} {'dRSI':>5} {'RS':>6} "
                  f"{'Mom':>4} {'TB':>4} {'Durum':>6}")
            print(f"  {'─' * w}")
            for b in items:
                mom = '✓' if b.get('wk_mom') else '✗'
                tb = b.get('tb_stage', '-')
                status = b.get('status', 'BEKLE')
                d_adx = f"{b['d_adx']:.1f}" if b.get('d_adx') is not None else '-'
                d_slp = f"{b['d_adx_slope']:+.2f}" if b.get('d_adx_slope') is not None else '-'
                d_rsi = f"{b['d_rsi']:.1f}" if b.get('d_rsi') is not None else '-'
                rs = f"{b['rs_score']:.2f}" if b.get('rs_score') is not None else '-'
                delta = b.get('delta_pct_at_trigger', b.get('delta_pct', 0))
                print(f"  {b['ticker']:<8} {b['signal_date']:>10} {b.get('trigger_type','?'):>5} "
                      f"{b['close']:>8.2f} {b['pivot']:>8.2f} {delta:>+5.1f}% "
                      f"{b.get('rg', 0):>5.0f} {d_adx:>5} {d_slp:>6} {d_rsi:>5} {rs:>6} "
                      f"{mom:>4} {tb:>4} {status:>6}")
            print(f"  {'─' * w}")
    else:
        print(f"\n  ◆ PIVOT AL+TETIK (0)")

    # Zone-only
    if zone_only:
        print(f"\n  ◆ ZONE_ONLY — Tetik Bekleniyor ({len(zone_only)} hisse)")
        for z in zone_only[:10]:
            print(f"    {z['ticker']:<8} ◆{z['pivot']:.2f} Δ{z.get('delta_pct', 0):+.1f}%")
        if len(zone_only) > 10:
            print(f"    ...ve {len(zone_only) - 10} hisse daha")
    else:
        print(f"\n  ◆ ZONE_ONLY (0)")

    # SAT
    severe = [s for s in sells if s['severity'] >= 2]
    mild = [s for s in sells if s['severity'] == 1]
    slope_only = [s for s in sells if s['severity'] == 0]
    _print_sell_group("◆ PIVOT SAT — SERT (Severity 2-3)", severe)
    _print_sell_group("◆ PIVOT SAT — Hafif (Severity 1)", mild)
    if slope_only:
        print(f"\n  ◆ PIVOT SAT — Sadece Slope ({len(slope_only)} hisse, listelenmedi)")


def _print_section(title):
    w = 80
    print(f"\n{'━' * w}")
    print(f"  {title}")
    print(f"{'━' * w}")


def _print_buy_group(label, items, weekly=False):
    if not items:
        print(f"\n  {label} (0)")
        return
    print(f"\n  {label} ({len(items)})")
    w = 90 if not weekly else 115
    print(f"  {'─' * w}")
    hdr = (f"  {'Hisse':<8} {'◆Elmas':>10} {'Fiyat':>8} {'DipFiy':>8} {'Δ%':>6} "
           f"{'RG':>5} {'ADX':>5} {'Slope':>6} {'RSI':>5}")
    if weekly:
        hdr += f" {'Mom':>4} {'TB':>4} {'Durum':>6}"
    hdr += f" {'':>6}"
    print(hdr)
    print(f"  {'─' * w}")
    for b in items:
        tag = '★BUGN' if b.get('fresh') == 'BUGUN' else ''
        line = (f"  {b['ticker']:<8} {b['pivot_date']:>10} "
                f"{b['close']:>8.2f} {b['pivot']:>8.2f} {b['delta_pct']:>+5.1f}% "
                f"{b['rg']:>5.0f} {b['adx']:>5.1f} {b['slope']:>+6.2f} {b['rsi']:>5.1f}")
        if weekly:
            mom = '✓' if b.get('wk_mom') else '✗'
            tb = b.get('tb_stage', '-')
            status = b.get('status', 'BEKLE')
            line += f" {mom:>4} {tb:>4} {status:>6}"
        line += f" {tag:>6}"
        print(line)
    print(f"  {'─' * w}")


def _print_sell_group(label, items):
    if not items:
        print(f"\n  {label} (0)")
        return
    print(f"\n  {label} ({len(items)})")
    print(f"  {'─' * 82}")
    print(f"  {'Hisse':<8} {'◆Elmas':>10} {'Fiyat':>8} {'TepeFiy':>8} "
          f"{'Sev':>4} {'ADX':>5} {'Slope':>6} {'RSI':>5} {'DD%':>6} {'':>6}")
    print(f"  {'─' * 82}")
    for s in items:
        tag = '★BUGN' if s.get('fresh') == 'BUGUN' else ''
        print(f"  {s['ticker']:<8} {s['pivot_date']:>10} "
              f"{s['close']:>8.2f} {s['pivot']:>8.2f} "
              f"{s['severity']:>4} {s['adx']:>5.1f} {s['slope']:>+6.2f} "
              f"{s['rsi']:>5.1f} {s['dd_pct']:>6.1f} {tag:>6}")
    print(f"  {'─' * 82}")


def _print_candidate_group(label, items, weekly=False):
    if not items:
        print(f"\n  {label} (0)")
        return
    print(f"\n  {label} ({len(items)})")
    w = 90 if not weekly else 115
    print(f"  {'─' * w}")
    hdr = (f"  {'Hisse':<8} {'◆Elmas':>10} {'Fiyat':>8} {'DipFiy':>8} {'Δ%':>6} "
           f"{'Bars':>4} {'RG':>5} {'ADX':>5} {'Slope':>6} {'RSI':>5}")
    if weekly:
        hdr += f" {'Mom':>4} {'TB':>4} {'Durum':>6}"
    print(hdr)
    print(f"  {'─' * w}")
    for c in items:
        gate_tag = '⬡' if c.get('gate') else ''
        line = (f"  {c['ticker']:<8} {c['pivot_date']:>10} "
                f"{c['close']:>8.2f} {c['pivot']:>8.2f} {c['delta_pct']:>+5.1f}% "
                f"{c['bars_since']:>4} {c['rg']:>5.0f} {c['adx']:>5.1f} "
                f"{c['slope']:>+6.2f} {c['rsi']:>5.1f}")
        if weekly:
            mom = '✓' if c.get('wk_mom') else '✗'
            tb = c.get('tb_stage', '-')
            status = c.get('status', 'BEKLE')
            line += f" {mom:>4} {tb:>4} {status:>6}"
        line += f" {gate_tag}"
        print(line)
    print(f"  {'─' * w}")


def _print_results(buys, sells, candidates, n_scanned, date_str, tf_label):
    _print_section(f"◆ NOX v3 [{tf_label}] — {date_str} — {n_scanned} hisse")
    is_weekly = tf_label == 'HAFTALIK'

    # AL sinyallerini gate durumuna gore ayir
    gated = [b for b in buys if b['gate']]
    ungated = [b for b in buys if not b['gate']]

    al_label = "◆ WATCHLIST AL" if is_weekly else "◆ PIVOT AL"
    _print_buy_group(f"{al_label} — ONAYLI (Gate Acik)", gated, weekly=is_weekly)
    _print_buy_group(f"{al_label} — Sadece Pivot (Gate Kapali)", ungated, weekly=is_weekly)

    # ADAY: onaylanmamis pivot adaylari
    _print_candidate_group("◆ ADAY — Taze Elmas (Onay Bekliyor)", candidates, weekly=is_weekly)

    # SAT sinyallerini severity'ye gore ayir
    severe = [s for s in sells if s['severity'] >= 2]
    mild = [s for s in sells if s['severity'] == 1]
    slope_only = [s for s in sells if s['severity'] == 0]

    _print_sell_group("◆ PIVOT SAT — SERT (Severity 2-3)", severe)
    _print_sell_group("◆ PIVOT SAT — Hafif (Severity 1)", mild)
    if slope_only:
        print(f"\n  ◆ PIVOT SAT — Sadece Slope ({len(slope_only)} hisse, listelenmedi)")


def _print_debug(ticker, result):
    print(f"\n  {'=' * 60}")
    print(f"  DEBUG: {ticker}")
    print(f"  {'=' * 60}")
    last = len(result['close']) - 1
    lb = 10
    start = max(0, last - lb)
    print(f"\n  Son {last - start + 1} bar:")
    print(f"  {'Bar':>4} {'Close':>8} {'BR':>5} {'RG':>5} {'Gate':<5} "
          f"{'ADX':>5} {'Slope':>6} {'RSI':>5} {'Sev':>4} {'PvtL':>8} {'PvtH':>8} {'BUY':<4} {'SELL':<4}")
    print(f"  {'─' * 85}")
    for i in range(start, last + 1):
        pvt_l = result['pivot_low_price'].iloc[i]
        pvt_h = result['pivot_high_price'].iloc[i]
        pvt_l_str = f"{pvt_l:.2f}" if pd.notna(pvt_l) else "—"
        pvt_h_str = f"{pvt_h:.2f}" if pd.notna(pvt_h) else "—"
        buy_str = "AL" if result['pivot_buy'].iloc[i] else ""
        sell_str = "SAT" if result['pivot_sell'].iloc[i] else ""
        gate_str = "ACIK" if result['gate_open'].iloc[i] else "KPLI"
        print(f"  {i:>4} {result['close'].iloc[i]:>8.2f} "
              f"{result['br_score'].iloc[i]:>5.0f} {result['rg_score'].iloc[i]:>5.0f} "
              f"{gate_str:<5} {result['adx'].iloc[i]:>5.1f} "
              f"{result['adx_slope'].iloc[i]:>+6.2f} {result['rsi'].iloc[i]:>5.1f} "
              f"{result['sell_severity'].iloc[i]:>4} "
              f"{pvt_l_str:>8} {pvt_h_str:>8} {buy_str:<4} {sell_str:<4}")
    print(f"  {'─' * 85}")


# =============================================================================
# CSV
# =============================================================================

def _save_csv(buys, sells, date_str, output_dir, suffix='', candidates=None):
    os.makedirs(output_dir, exist_ok=True)
    rows = []
    for b in buys:
        row = {
            'signal': 'PIVOT_AL',
            'ticker': b['ticker'],
            'pivot_date': b['pivot_date'],
            'signal_date': b['signal_date'],
            'close': b['close'],
            'pivot_price': b['pivot'],
            'delta_pct': b.get('delta_pct'),
            'gate_open': b['gate'],
            'rg_score': b['rg'],
            'adx': b['adx'],
            'adx_slope': b['slope'],
            'rsi': b['rsi'],
            'wl_status': b.get('status', ''),
            'tb_stage': b.get('tb_stage', ''),
            'tb_prep': b.get('tb_prep', ''),
        }
        rows.append(row)
    for c in (candidates or []):
        row = {
            'signal': 'ADAY',
            'ticker': c['ticker'],
            'pivot_date': c['pivot_date'],
            'signal_date': c.get('pivot_date', ''),
            'close': c['close'],
            'pivot_price': c['pivot'],
            'delta_pct': c.get('delta_pct'),
            'gate_open': c.get('gate'),
            'adx': c['adx'],
            'adx_slope': c['slope'],
            'rsi': c['rsi'],
            'wl_status': c.get('status', ''),
            'tb_stage': c.get('tb_stage', ''),
            'tb_prep': c.get('tb_prep', ''),
        }
        rows.append(row)
    for s in sells:
        rows.append({
            'signal': 'PIVOT_SAT',
            'ticker': s['ticker'],
            'pivot_date': s['pivot_date'],
            'signal_date': s['signal_date'],
            'close': s['close'],
            'pivot_price': s['pivot'],
            'severity': s['severity'],
            'adx': s['adx'],
            'adx_slope': s['slope'],
            'rsi': s['rsi'],
            'drawdown_pct': s['dd_pct'],
        })
    if rows:
        csv_df = pd.DataFrame(rows)
        fname = f"nox_v3_signals{suffix}_{date_str.replace('-', '')}.csv"
        path = os.path.join(output_dir, fname)
        csv_df.to_csv(path, index=False)
        print(f"\n  CSV: {path}")


# =============================================================================
# HTML RAPOR
# =============================================================================

def _generate_html(d_buys, d_sells, d_cands, d_n, d_date,
                   w_buys, w_sells, w_cands, w_n, w_date,
                   overlap_tickers):
    from datetime import timezone, timedelta
    tr_tz = timezone(timedelta(hours=3))
    now = datetime.now(tr_tz).strftime('%d.%m.%Y %H:%M')

    def _prep_buys(buys, weekly=False):
        rows = []
        for b in buys:
            row = {
                'ticker': b['ticker'],
                'close': round(b['close'], 2),
                'pivot': round(b['pivot'], 2),
                'delta_pct': b.get('delta_pct', 0),
                'rg': round(b.get('rg', 0), 1),
                'adx': round(b.get('adx', 0), 1),
                'slope': round(b.get('slope', 0), 2),
                'rsi': round(b.get('rsi', 0), 1),
                'gate': b.get('gate', False),
                'pivot_date': b.get('pivot_date', ''),
                'signal_date': b.get('signal_date', ''),
                'fresh': b.get('fresh', 'YAKIN'),
                'rs_score': round(b.get('rs_score', 0), 3) if b.get('rs_score') else None,
                'trigger_type': b.get('trigger_type', ''),
            }
            if weekly:
                row['wk_mom'] = b.get('wk_mom', False)
                row['tb_stage'] = b.get('tb_stage', '-')
                row['tb_prep'] = b.get('tb_prep', 0)
                row['status'] = b.get('status', 'BEKLE')
            rows.append(row)
        return rows

    def _prep_sells(sells):
        rows = []
        for s in sells:
            rows.append({
                'ticker': s['ticker'],
                'close': round(s['close'], 2),
                'pivot': round(s['pivot'], 2),
                'severity': s['severity'],
                'adx': round(s['adx'], 1),
                'slope': round(s['slope'], 2),
                'rsi': round(s['rsi'], 1),
                'dd_pct': round(s['dd_pct'], 1),
                'pivot_date': s['pivot_date'],
                'signal_date': s['signal_date'],
                'fresh': s['fresh'],
            })
        return rows

    def _prep_candidates(cands, weekly=False):
        rows = []
        for c in cands:
            row = {
                'ticker': c['ticker'],
                'close': round(c['close'], 2),
                'pivot': round(c['pivot'], 2),
                'delta_pct': c.get('delta_pct', 0),
                'bars_since': c.get('bars_since', 0),
                'rg': round(c.get('rg', 0), 1),
                'adx': round(c.get('adx', 0), 1),
                'slope': round(c.get('slope', 0), 2),
                'rsi': round(c.get('rsi', 0), 1),
                'gate': c.get('gate', False),
                'pivot_date': c.get('pivot_date', ''),
                'rs_score': round(c.get('rs_score', 0), 3) if c.get('rs_score') else None,
            }
            if weekly:
                row['wk_mom'] = c.get('wk_mom', False)
                row['tb_stage'] = c.get('tb_stage', '-')
                row['tb_prep'] = c.get('tb_prep', 0)
                row['status'] = c.get('status', 'BEKLE')
            rows.append(row)
        return rows

    data = {
        'daily': {
            'buys': _sanitize(_prep_buys(d_buys)) if d_buys else [],
            'sells': _sanitize(_prep_sells(d_sells)) if d_sells else [],
            'candidates': _sanitize(_prep_candidates(d_cands)) if d_cands else [],
            'n': d_n or 0, 'date': d_date or '',
        },
        'weekly': {
            'buys': _sanitize(_prep_buys(w_buys, weekly=True)) if w_buys else [],
            'sells': _sanitize(_prep_sells(w_sells)) if w_sells else [],
            'candidates': _sanitize(_prep_candidates(w_cands, weekly=True)) if w_cands else [],
            'n': w_n or 0, 'date': w_date or '',
        },
        'overlap': sorted(overlap_tickers) if overlap_tickers else [],
    }
    data_json = json.dumps(data, ensure_ascii=False)

    html = f"""<!DOCTYPE html>
<html lang="tr"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NOX v3 — PIVOT AL/SAT · {now}</title>
<style>{_NOX_CSS}

/* TABS */
.nox-tabs {{ display: flex; gap: 4px; margin-bottom: 16px; }}
.nox-tab {{
  padding: 10px 24px; border-radius: var(--radius) var(--radius) 0 0;
  background: var(--bg-card); border: 1px solid var(--border-subtle); border-bottom: none;
  color: var(--text-muted); font-weight: 600; font-size: 0.85rem;
  cursor: pointer; transition: all 0.2s; font-family: var(--font-display);
}}
.nox-tab:hover {{ color: var(--text-secondary); background: var(--bg-elevated); }}
.nox-tab.active {{
  color: var(--nox-cyan); background: var(--bg-elevated);
  border-color: var(--nox-cyan); border-bottom: 2px solid var(--bg-elevated);
}}
.nox-tab .cnt {{ font-family: var(--font-mono); font-size: 0.75rem; margin-left: 6px;
  padding: 2px 8px; border-radius: 10px; background: rgba(34,211,238,0.1); }}
.tab-content {{ display: none; }}
.tab-content.active {{ display: block; }}

/* SECTION */
.section-title {{
  font-size: 0.85rem; font-weight: 700; padding: 12px 0 8px;
  color: var(--text-secondary); font-family: var(--font-display);
  display: flex; align-items: center; gap: 8px;
}}
.section-title .icon {{ font-size: 1rem; }}
.section-count {{
  font-family: var(--font-mono); font-size: 0.72rem;
  padding: 2px 8px; border-radius: 10px;
}}
.cnt-buy {{ background: rgba(74,222,128,0.12); color: var(--nox-green); }}
.cnt-sell {{ background: rgba(248,113,113,0.12); color: var(--nox-red); }}
.cnt-gate {{ background: rgba(34,211,238,0.12); color: var(--nox-cyan); }}

/* GATE BADGE */
.gate-open {{ color: var(--nox-green); font-weight: 700; }}
.gate-closed {{ color: var(--text-muted); }}
.sev-badge {{
  display: inline-block; padding: 2px 8px; border-radius: var(--radius-sm);
  font-size: 0.68rem; font-weight: 700; font-family: var(--font-mono);
}}
.sev-3 {{ background: rgba(248,113,113,0.2); color: var(--nox-red); }}
.sev-2 {{ background: rgba(251,146,60,0.15); color: var(--nox-orange); }}
.sev-1 {{ background: rgba(250,204,21,0.12); color: var(--nox-yellow); }}
.sev-0 {{ background: rgba(113,113,122,0.12); color: var(--text-muted); }}
.overlap-badge {{
  display: inline-block; padding: 1px 6px; border-radius: 8px;
  font-size: 0.6rem; font-weight: 700; font-family: var(--font-mono);
  background: rgba(192,132,252,0.15); color: var(--nox-purple);
  margin-left: 4px; vertical-align: middle;
}}
.dd-neg {{ color: var(--nox-red); }}
.fresh-badge {{
  display: inline-block; padding: 2px 6px; border-radius: var(--radius-sm);
  font-size: 0.62rem; font-weight: 700; font-family: var(--font-mono);
}}
.fresh-bugun {{ background: rgba(74,222,128,0.18); color: var(--nox-green); }}
.fresh-yakin {{ background: rgba(250,204,21,0.12); color: var(--nox-yellow); }}
.fresh-aday {{ background: rgba(34,211,238,0.15); color: var(--nox-cyan); }}
.delta-lo {{ color: var(--nox-green); }}
.delta-mid {{ color: var(--nox-yellow); }}
.delta-hi {{ color: var(--nox-red); }}
.status-badge {{
  display: inline-block; padding: 2px 8px; border-radius: var(--radius-sm);
  font-size: 0.68rem; font-weight: 700; font-family: var(--font-mono);
}}
.status-hazir {{ background: rgba(74,222,128,0.18); color: var(--nox-green); }}
.status-izle  {{ background: rgba(250,204,21,0.15); color: var(--nox-yellow); }}
.status-bekle {{ background: rgba(113,113,122,0.15); color: var(--text-muted); }}
.tb-ok   {{ color: var(--nox-green); font-weight: 700; }}
.tb-trg  {{ color: var(--nox-cyan); font-weight: 700; }}
.tb-prep {{ color: var(--nox-yellow); }}
.tb-none {{ color: var(--text-muted); }}
</style>
</head><body>
<div class="nox-container">
<div class="nox-header">
  <div class="nox-logo">NOX<span class="proj">project</span><span class="mode">pivot al/sat · v3</span></div>
  <div class="nox-meta">{now}</div>
</div>

<div class="nox-filters" style="margin-bottom:12px">
  <div><label>Hisse</label><input type="text" id="fS" placeholder="ARA" oninput="render()"></div>
  <div><label>Gate</label>
  <select id="fGate" onchange="render()"><option value="">Tumü</option>
  <option value="open">Acik</option><option value="closed">Kapali</option></select></div>
  <div><label>Fresh</label>
  <select id="fFresh" onchange="render()"><option value="">Tumü</option>
  <option value="BUGUN">Bugün</option><option value="YAKIN">Yakın</option></select></div>
  <div><label>Δ%≤</label><input type="number" id="fDelta" value="" step="5" min="0" placeholder="max" oninput="render()"></div>
  <div><label>ADX≥</label><input type="number" id="fADX" value="0" step="5" min="0" oninput="render()"></div>
  <div><label>RSI</label>
  <select id="fRSI" onchange="render()"><option value="">Tumü</option>
  <option value="low">≤30 (Asırı Satım)</option><option value="mid">30-70</option>
  <option value="high">≥70 (Asırı Alım)</option></select></div>
  <div><label>Durum</label>
  <select id="fStatus" onchange="render()"><option value="">Tumü</option>
  <option value="HAZIR">HAZIR</option><option value="İZLE">İZLE</option>
  <option value="BEKLE">BEKLE</option></select></div>
  <div><button class="nox-btn" onclick="resetF()">Sifirla</button></div>
</div>

<details class="trade-guide" style="margin-bottom:14px;padding:10px 14px;border-radius:8px;
  background:rgba(34,211,238,0.06);border:1px solid rgba(34,211,238,0.15);
  font-size:0.78rem;color:var(--text-secondary);line-height:1.6">
<summary style="cursor:pointer;font-weight:700;color:var(--nox-cyan);font-size:0.82rem">
  Trade Rehberi</summary>
<div style="margin-top:8px">
<b style="color:var(--nox-green)">AL Checklist (Haftalik tab)</b><br>
1. <b>Durum: HAZIR</b> filtrele (WR %78, en kaliteli grup)<br>
2. <b>Gate: AÇIK</b> olanları seç (breadth+regime onaylı)<br>
3. <b>D+W badge</b> varsa öncelik ver (günlük+haftalık çakışma)<br>
4. <b>Δ% düşük</b> → pivot zonuna yakın = iyi risk/ödül<br>
5. Hisse adına tıkla → TradingView'da grafiği kontrol et<br>
6. Stop = pivot fiyatının %2-3 altı<br>
<br>
<b style="color:var(--nox-cyan)">Tarih Okuma</b><br>
<b>Sinyal</b> = tetik günü (giriş tarihi). <b>Elmas</b> = pivot dip oluşum tarihi.<br>
&#8226; BUGÜN → hemen değerlendir | 1-5 gün → fiyat yakınsa geçerli | >2 hafta → geç kalmış, atla<br>
&#8226; Elmas ne kadar tazeyse destek o kadar güvenilir (>2 ay → aşınmış olabilir)<br>
<br>
<b style="color:var(--nox-red)">SAT Sinyalleri</b><br>
&#8226; Sev 2-3 → pozisyon kapat/daralt | Sev 1 → stop sıkılaştır | Sev 0 → bilgi amaçlı<br>
<br>
<b style="color:var(--nox-yellow)">Kolon Açıklamaları</b>
<table style="width:100%;font-size:0.74rem;margin-top:6px;border-collapse:collapse">
<tr style="border-bottom:1px solid rgba(255,255,255,0.08)">
  <td style="padding:3px 6px;color:var(--nox-cyan);font-weight:700;white-space:nowrap">Hisse</td>
  <td style="padding:3px 6px">Ticker (tıkla → TradingView). <b>D+W</b> badge = günlük+haftalık çakışma</td></tr>
<tr style="border-bottom:1px solid rgba(255,255,255,0.08)">
  <td style="padding:3px 6px;color:var(--nox-cyan);font-weight:700">Sinyal</td>
  <td style="padding:3px 6px">Günlük tetik (BOS/HC2/EMA_R) ateşlenme tarihi = giriş günü. BUGÜN/YAKIN etiketi tazeliği gösterir. >2 hafta → atla</td></tr>
<tr style="border-bottom:1px solid rgba(255,255,255,0.08)">
  <td style="padding:3px 6px;color:var(--nox-cyan);font-weight:700">Elmas</td>
  <td style="padding:3px 6px">Haftalık pivot low (◆) oluşum tarihi. Yapısal destek seviyesi. Taze elmas daha güvenilir, >2 ay → aşınmış olabilir</td></tr>
<tr style="border-bottom:1px solid rgba(255,255,255,0.08)">
  <td style="padding:3px 6px;color:var(--nox-cyan);font-weight:700">Fiyat</td>
  <td style="padding:3px 6px">Tetik günü kapanış fiyatı (giriş fiyatı)</td></tr>
<tr style="border-bottom:1px solid rgba(255,255,255,0.08)">
  <td style="padding:3px 6px;color:var(--nox-cyan);font-weight:700">DipFiy</td>
  <td style="padding:3px 6px">Pivot low fiyatı (destek seviyesi). Stop loss için referans: %2-3 altı</td></tr>
<tr style="border-bottom:1px solid rgba(255,255,255,0.08)">
  <td style="padding:3px 6px;color:var(--nox-cyan);font-weight:700">Δ%</td>
  <td style="padding:3px 6px">Fiyatın pivot üzerindeki uzaklığı. Düşük = zona yakın, iyi risk/ödül. Yeşil ≤5% | Sarı ≤15% | Kırmızı >15%</td></tr>
<tr style="border-bottom:1px solid rgba(255,255,255,0.08)">
  <td style="padding:3px 6px;color:var(--nox-cyan);font-weight:700">RG</td>
  <td style="padding:3px 6px">Regime Gate skoru (0-100). Trend gücü ve yönü. Yüksek = güçlü trend ortamı</td></tr>
<tr style="border-bottom:1px solid rgba(255,255,255,0.08)">
  <td style="padding:3px 6px;color:var(--nox-cyan);font-weight:700">ADX</td>
  <td style="padding:3px 6px">Günlük ADX (14). Trend gücü. >25 = trend var, <20 = yatay piyasa</td></tr>
<tr style="border-bottom:1px solid rgba(255,255,255,0.08)">
  <td style="padding:3px 6px;color:var(--nox-cyan);font-weight:700">Slope</td>
  <td style="padding:3px 6px">ADX 5-bar eğimi. Yeşil (+) = momentum artıyor | Kırmızı (-) = momentum azalıyor</td></tr>
<tr style="border-bottom:1px solid rgba(255,255,255,0.08)">
  <td style="padding:3px 6px;color:var(--nox-cyan);font-weight:700">RSI</td>
  <td style="padding:3px 6px">RSI (14). <30 yeşil (aşırı satım, dönüş potansiyeli) | >70 kırmızı (aşırı alım)</td></tr>
<tr style="border-bottom:1px solid rgba(255,255,255,0.08)">
  <td style="padding:3px 6px;color:var(--nox-cyan);font-weight:700">RS</td>
  <td style="padding:3px 6px">Relative Strength vs XU100 (20G). Yeşil >1 = endeksten güçlü | Kırmızı ≤1 = zayıf. Pivot AL'da RS≤1 daha iyi (WR %72 vs %29) — geri kalmış hisse destek bulunca güçlü döner</td></tr>
<tr style="border-bottom:1px solid rgba(255,255,255,0.08)">
  <td style="padding:3px 6px;color:var(--nox-cyan);font-weight:700">Gate</td>
  <td style="padding:3px 6px">Breadth + Regime onayı. AÇIK = piyasa koşulları uygun, sinyal güvenilir. KAPALI = dikkatli ol</td></tr>
<tr style="border-bottom:1px solid rgba(255,255,255,0.08)">
  <td style="padding:3px 6px;color:var(--nox-cyan);font-weight:700">Mom</td>
  <td style="padding:3px 6px">Haftalık momentum. ✓ = son hafta pozitif kapanış | ✗ = negatif</td></tr>
<tr style="border-bottom:1px solid rgba(255,255,255,0.08)">
  <td style="padding:3px 6px;color:var(--nox-cyan);font-weight:700">TB</td>
  <td style="padding:3px 6px">Trend Birth aşaması. OK = trend doğmuş | TRG = tetik aşaması | PREP = hazırlık | - = yok</td></tr>
<tr style="border-bottom:1px solid rgba(255,255,255,0.08)">
  <td style="padding:3px 6px;color:var(--nox-cyan);font-weight:700">Prep</td>
  <td style="padding:3px 6px">Trend Birth hazırlık skoru (0-100). Yüksek = trend doğuşuna yakın</td></tr>
<tr>
  <td style="padding:3px 6px;color:var(--nox-cyan);font-weight:700">Durum</td>
  <td style="padding:3px 6px">Watchlist durumu. <b style="color:var(--nox-green)">HAZIR</b> (WR %78) → trade et | <b style="color:var(--nox-yellow)">İZLE</b> (WR %69) → takipte tut | BEKLE (WR %60) → erken</td></tr>
</table>
</div>
</details>

<div class="nox-tabs">
  <div class="nox-tab active" onclick="switchTab('daily')" id="tab-daily">
    Günlük <span class="cnt" id="cnt-daily"></span></div>
  <div class="nox-tab" onclick="switchTab('weekly')" id="tab-weekly">
    Haftalik <span class="cnt" id="cnt-weekly"></span></div>
</div>

<div id="tc-daily" class="tab-content active"></div>
<div id="tc-weekly" class="tab-content"></div>

<div class="nox-status" id="st"></div>
</div>

<script>
const D={data_json};
const OV=new Set(D.overlap);
let curTab='daily';

function switchTab(t){{
  curTab=t;
  document.querySelectorAll('.nox-tab').forEach(x=>x.classList.remove('active'));
  document.getElementById('tab-'+t).classList.add('active');
  document.querySelectorAll('.tab-content').forEach(x=>x.classList.remove('active'));
  document.getElementById('tc-'+t).classList.add('active');
}}

function resetF(){{
  document.getElementById('fS').value='';
  document.getElementById('fGate').value='';
  document.getElementById('fFresh').value='';
  document.getElementById('fDelta').value='';
  document.getElementById('fADX').value='0';
  document.getElementById('fRSI').value='';
  document.getElementById('fStatus').value='';
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
function doSort(tf, tbl, col){{
  const key=tf+'-'+tbl;
  if(!sortState[key]||sortState[key].col!==col)
    sortState[key]={{col:col,asc:col==='ticker'}};
  else sortState[key].asc=!sortState[key].asc;
  render();
}}

function applyGlobalFilters(rows){{
  const sr=document.getElementById('fS').value.toUpperCase();
  const ff=document.getElementById('fFresh').value;
  const maxDelta=parseFloat(document.getElementById('fDelta').value);
  const minADX=parseFloat(document.getElementById('fADX').value)||0;
  const fRSI=document.getElementById('fRSI').value;
  const fStatus=document.getElementById('fStatus').value;
  return rows.filter(r=>{{
    if(sr&&!r.ticker.includes(sr)) return false;
    if(ff&&r.fresh!==ff) return false;
    if(!isNaN(maxDelta)&&maxDelta>0&&r.delta_pct!=null&&r.delta_pct>maxDelta) return false;
    if(minADX>0&&r.adx<minADX) return false;
    if(fRSI==='low'&&r.rsi>30) return false;
    if(fRSI==='mid'&&(r.rsi<30||r.rsi>70)) return false;
    if(fRSI==='high'&&r.rsi<70) return false;
    if(fStatus&&r.status&&r.status!==fStatus) return false;
    return true;
  }});
}}
function mkFreshBadge(fresh){{
  if(fresh==='BUGUN') return '<span class="fresh-badge fresh-bugun">BUGÜN</span>';
  return '<span class="fresh-badge fresh-yakin">YAKIN</span>';
}}

function mkStatusBadge(s){{
  if(!s) return '';
  const cls=s==='HAZIR'?'status-hazir':s==='İZLE'?'status-izle':'status-bekle';
  return `<span class="status-badge ${{cls}}">${{s}}</span>`;
}}
function mkTbBadge(stage){{
  if(!stage||stage==='-') return '<span class="tb-none">-</span>';
  const cls=stage==='OK'?'tb-ok':stage==='TRG'?'tb-trg':'tb-prep';
  return `<span class="${{cls}}">${{stage}}</span>`;
}}

function mkBuyTable(buys, tf, label, cssClass){{
  const isW=tf==='weekly';
  const sk=sortState[tf+'-'+label];
  if(sk) buys=sortRows(buys, sk.col, sk.asc);
  else if(isW){{ // varsayilan siralama: HAZIR > IZLE > BEKLE, sonra delta_pct artan
    const so={{'HAZIR':0,'İZLE':1,'BEKLE':2}};
    buys=[...buys].sort((a,b)=>{{
      const sa=so[a.status]??2, sb=so[b.status]??2;
      if(sa!==sb) return sa-sb;
      return (a.delta_pct||0)-(b.delta_pct||0);
    }});
  }}
  const fg=document.getElementById('fGate').value;
  buys=applyGlobalFilters(buys);
  buys=buys.filter(r=>{{
    if(fg==='open'&&!r.gate) return false;
    if(fg==='closed'&&r.gate) return false;
    return true;
  }});
  if(!buys.length) return '';
  const srt=(c)=>`onclick="doSort('${{tf}}','${{label}}','${{c}}')"`;
  let h=`<div class="section-title"><span class="icon">${{cssClass==='cnt-gate'?'✅':'◆'}}</span>${{
    cssClass==='cnt-gate'?'ONAYLI (Gate Açık)':'Sadece Pivot (Gate Kapalı)'}}<span class="section-count ${{cssClass}}">${{buys.length}}</span></div>`;
  h+=`<div class="nox-table-wrap" style="margin-bottom:16px"><table><thead><tr>
  <th ${{srt('ticker')}}>Hisse</th><th ${{srt('signal_date')}}>Sinyal</th><th ${{srt('pivot_date')}}>Elmas</th>
  <th ${{srt('close')}}>Fiyat</th><th ${{srt('pivot')}}>DipFiy</th><th ${{srt('delta_pct')}}>Δ%</th>
  <th ${{srt('rg')}}>RG</th><th ${{srt('adx')}}>ADX</th>
  <th ${{srt('slope')}}>Slope</th><th ${{srt('rsi')}}>RSI</th>
  <th ${{srt('rs_score')}}>RS</th><th>Gate</th>`;
  if(isW) h+=`<th>Mom</th><th ${{srt('tb_stage')}}>TB</th><th ${{srt('tb_prep')}}>Prep</th><th ${{srt('status')}}>Durum</th>`;
  h+=`</tr></thead><tbody>`;
  buys.forEach(r=>{{
    const ovB=OV.has(r.ticker)?'<span class="overlap-badge">D+W</span>':'';
    const gateC=r.gate?'gate-open':'gate-closed';
    const gateT=r.gate?'AÇIK':'KAPALI';
    const slopeC=r.slope>0?'var(--nox-green)':r.slope<-0.3?'var(--nox-red)':'var(--text-muted)';
    const rsiC=r.rsi<30?'var(--nox-green)':r.rsi>70?'var(--nox-red)':'var(--text-primary)';
    const rsV=r.rs_score!=null?r.rs_score.toFixed(2):'-';
    const rsC=r.rs_score>1?'var(--nox-green)':r.rs_score!=null?'var(--nox-red)':'var(--text-muted)';
    const deltaC=r.delta_pct<=5?'delta-lo':r.delta_pct<=15?'delta-mid':'delta-hi';
    h+=`<tr${{r.gate?' class="hl"':''}}>
    <td><a class="tv-link" href="https://www.tradingview.com/chart/?symbol=BIST:${{r.ticker}}" target="_blank">${{r.ticker}}</a>${{ovB}}</td>
    <td>${{r.signal_date}} ${{mkFreshBadge(r.fresh)}}</td>
    <td style="color:var(--text-muted)">${{r.pivot_date}}</td>
    <td>${{r.close}}</td><td>${{r.pivot}}</td>
    <td class="${{deltaC}}">${{r.delta_pct>0?'+':''}}${{r.delta_pct}}%</td>
    <td>${{r.rg}}</td><td>${{r.adx}}</td>
    <td style="color:${{slopeC}}">${{r.slope>0?'+':''}}${{r.slope}}</td>
    <td style="color:${{rsiC}}">${{r.rsi}}</td>
    <td style="color:${{rsC}}">${{rsV}}</td>
    <td class="${{gateC}}">${{gateT}}</td>`;
    if(isW) h+=`<td style="color:${{r.wk_mom?'var(--nox-green)':'var(--text-muted)'}}">${{r.wk_mom?'✓':'✗'}}</td>
    <td>${{mkTbBadge(r.tb_stage)}}</td><td>${{r.tb_prep||0}}</td><td>${{mkStatusBadge(r.status)}}</td>`;
    h+=`</tr>`;
  }});
  h+=`</tbody></table></div>`;
  return h;
}}

function mkSellTable(sells, tf, label, cssClass){{
  const sk=sortState[tf+'-'+label];
  if(sk) sells=sortRows(sells, sk.col, sk.asc);
  sells=applyGlobalFilters(sells);
  if(!sells.length) return '';
  const srt=(c)=>`onclick="doSort('${{tf}}','${{label}}','${{c}}')"`;
  const titles={{'sert':'SERT (Severity 2-3)','hafif':'Hafif (Severity 1)','slope':'Sadece Slope (Severity 0)'}};
  const icons={{'sert':'🔴','hafif':'🟡','slope':'⚪'}};
  let h=`<div class="section-title"><span class="icon">${{icons[label]||'◆'}}</span>PIVOT SAT — ${{titles[label]||label}}<span class="section-count ${{cssClass}}">${{sells.length}}</span></div>`;
  h+=`<div class="nox-table-wrap" style="margin-bottom:16px"><table><thead><tr>
  <th ${{srt('ticker')}}>Hisse</th><th ${{srt('signal_date')}}>Sinyal</th><th ${{srt('pivot_date')}}>Elmas</th>
  <th ${{srt('close')}}>Fiyat</th><th ${{srt('pivot')}}>TepeFiy</th>
  <th ${{srt('severity')}}>Sev</th><th ${{srt('adx')}}>ADX</th>
  <th ${{srt('slope')}}>Slope</th><th ${{srt('rsi')}}>RSI</th>
  <th ${{srt('dd_pct')}}>DD%</th></tr></thead><tbody>`;
  sells.forEach(r=>{{
    const sevC='sev-'+r.severity;
    const slopeC=r.slope>0?'var(--nox-green)':r.slope<-0.3?'var(--nox-red)':'var(--text-muted)';
    h+=`<tr>
    <td><a class="tv-link" href="https://www.tradingview.com/chart/?symbol=BIST:${{r.ticker}}" target="_blank">${{r.ticker}}</a></td>
    <td>${{r.signal_date}} ${{mkFreshBadge(r.fresh)}}</td>
    <td style="color:var(--text-muted)">${{r.pivot_date}}</td>
    <td>${{r.close}}</td><td>${{r.pivot}}</td>
    <td><span class="sev-badge ${{sevC}}">${{r.severity}}</span></td>
    <td>${{r.adx}}</td>
    <td style="color:${{slopeC}}">${{r.slope>0?'+':''}}${{r.slope}}</td>
    <td>${{r.rsi}}</td>
    <td class="dd-neg">${{r.dd_pct}}%</td></tr>`;
  }});
  h+=`</tbody></table></div>`;
  return h;
}}

function mkCandidateTable(cands, tf){{
  const isW=tf==='weekly';
  const sk=sortState[tf+'-cand'];
  if(sk) cands=sortRows(cands, sk.col, sk.asc);
  else if(isW){{
    const so={{'HAZIR':0,'İZLE':1,'BEKLE':2}};
    cands=[...cands].sort((a,b)=>{{
      const sa=so[a.status]??2, sb=so[b.status]??2;
      if(sa!==sb) return sa-sb;
      return (a.delta_pct||0)-(b.delta_pct||0);
    }});
  }}
  cands=applyGlobalFilters(cands);
  if(!cands.length) return '';
  const srt=(c)=>`onclick="doSort('${{tf}}','cand','${{c}}')"`;
  let h=`<div class="section-title"><span class="icon">💎</span>ADAY — Taze Elmas (Onay Bekliyor)<span class="section-count" style="background:rgba(34,211,238,0.12);color:var(--nox-cyan)">${{cands.length}}</span></div>`;
  h+=`<div class="nox-table-wrap" style="margin-bottom:16px"><table><thead><tr>
  <th ${{srt('ticker')}}>Hisse</th><th ${{srt('pivot_date')}}>Elmas</th>
  <th ${{srt('close')}}>Fiyat</th><th ${{srt('pivot')}}>DipFiy</th><th ${{srt('delta_pct')}}>Δ%</th>
  <th ${{srt('bars_since')}}>Bars</th>
  <th ${{srt('rg')}}>RG</th><th ${{srt('adx')}}>ADX</th>
  <th ${{srt('slope')}}>Slope</th><th ${{srt('rsi')}}>RSI</th>
  <th ${{srt('rs_score')}}>RS</th><th>Gate</th>`;
  if(isW) h+=`<th>Mom</th><th ${{srt('tb_stage')}}>TB</th><th ${{srt('tb_prep')}}>Prep</th><th ${{srt('status')}}>Durum</th>`;
  h+=`</tr></thead><tbody>`;
  cands.forEach(r=>{{
    const deltaC=r.delta_pct<=5?'delta-lo':r.delta_pct<=15?'delta-mid':'delta-hi';
    const slopeC=r.slope>0?'var(--nox-green)':r.slope<-0.3?'var(--nox-red)':'var(--text-muted)';
    const rsiC=r.rsi<30?'var(--nox-green)':r.rsi>70?'var(--nox-red)':'var(--text-primary)';
    const gateC=r.gate?'gate-open':'gate-closed';
    const gateT=r.gate?'AÇIK':'KAPALI';
    const barsLabel=r.bars_since+'/5';
    const rsV=r.rs_score!=null?r.rs_score.toFixed(2):'-';
    const rsC=r.rs_score>1?'var(--nox-green)':r.rs_score!=null?'var(--nox-red)':'var(--text-muted)';
    h+=`<tr>
    <td><a class="tv-link" href="https://www.tradingview.com/chart/?symbol=BIST:${{r.ticker}}" target="_blank">${{r.ticker}}</a></td>
    <td>${{r.pivot_date}} <span class="fresh-badge fresh-aday">ADAY</span></td>
    <td>${{r.close}}</td><td>${{r.pivot}}</td>
    <td class="${{deltaC}}">${{r.delta_pct>0?'+':''}}${{r.delta_pct}}%</td>
    <td style="color:var(--nox-cyan)">${{barsLabel}}</td>
    <td>${{r.rg}}</td><td>${{r.adx}}</td>
    <td style="color:${{slopeC}}">${{r.slope>0?'+':''}}${{r.slope}}</td>
    <td style="color:${{rsiC}}">${{r.rsi}}</td>
    <td style="color:${{rsC}}">${{rsV}}</td>
    <td class="${{gateC}}">${{gateT}}</td>`;
    if(isW) h+=`<td style="color:${{r.wk_mom?'var(--nox-green)':'var(--text-muted)'}}">${{r.wk_mom?'✓':'✗'}}</td>
    <td>${{mkTbBadge(r.tb_stage)}}</td><td>${{r.tb_prep||0}}</td><td>${{mkStatusBadge(r.status)}}</td>`;
    h+=`</tr>`;
  }});
  h+=`</tbody></table></div>`;
  return h;
}}

function render(){{
  ['daily','weekly'].forEach(tf=>{{
    const td=D[tf];
    const el=document.getElementById('tc-'+tf);
    const tfLabel=tf==='daily'?'Günlük':'Haftalık';
    let html=`<div style="font-size:0.75rem;color:var(--text-muted);margin-bottom:8px;font-family:var(--font-mono)">${{tfLabel}} — ${{td.date}} — ${{td.n}} hisse tarandi</div>`;

    // AL
    const gated=td.buys.filter(r=>r.gate);
    const ungated=td.buys.filter(r=>!r.gate);
    const alTitle=tf==='weekly'?'WATCHLIST — Haftalık Pivot':'PIVOT AL';
    html+=`<div class="section-title" style="font-size:1rem;color:var(--nox-green)">${{alTitle}}<span class="section-count cnt-buy">${{td.buys.length}}</span></div>`;
    html+=mkBuyTable(gated,tf,'gated','cnt-gate');
    html+=mkBuyTable(ungated,tf,'ungated','cnt-buy');

    // ADAY
    html+=mkCandidateTable(td.candidates||[],tf);

    // SAT
    const sert=td.sells.filter(r=>r.severity>=2);
    const hafif=td.sells.filter(r=>r.severity===1);
    const slope=td.sells.filter(r=>r.severity===0);
    html+=`<div class="section-title" style="font-size:1rem;color:var(--nox-red);margin-top:24px">PIVOT SAT<span class="section-count cnt-sell">${{td.sells.length}}</span></div>`;
    html+=mkSellTable(sert,tf,'sert','cnt-sell');
    html+=mkSellTable(hafif,tf,'hafif','cnt-sell');
    html+=mkSellTable(slope,tf,'slope','cnt-sell');

    el.innerHTML=html;
  }});

  // Tab counts
  document.getElementById('cnt-daily').textContent=D.daily.buys.length+' AL / '+D.daily.sells.length+' SAT';
  document.getElementById('cnt-weekly').textContent=D.weekly.buys.length+' AL / '+D.weekly.sells.length+' SAT';

  // Overlap
  let st='';
  if(D.overlap.length)
    st+='<span style="color:var(--nox-purple)">★ ÇAKIŞMA (Günlük+Haftalık AL): <b>'+D.overlap.join(', ')+'</b></span><br>';
  st+='<b>'+D.daily.buys.length+'</b> günlük AL · <b>'+D.weekly.buys.length+'</b> haftalık AL';
  document.getElementById('st').innerHTML=st;
}}

render();
</script></body></html>"""
    return html


def _save_html(html_content, date_str, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    fname = f"nox_v3_pivot_{date_str.replace('-', '')}.html"
    path = os.path.join(output_dir, fname)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"\n  HTML: {path}")
    return path


# =============================================================================
# TELEGRAM FORMAT
# =============================================================================

def _format_weekly_telegram(buys, sells, candidates, n_scanned, date_str, html_url=None):
    """Haftalik sinyal Telegram mesaji — HAZIR/IZLE/BEKLE gruplu."""
    # AL + ADAY birlestir, duruma gore grupla
    all_signals = buys + candidates
    hazir = [s for s in all_signals if s.get('status') == 'HAZIR']
    izle = [s for s in all_signals if s.get('status') == 'İZLE']
    bekle = [s for s in all_signals if s.get('status', 'BEKLE') == 'BEKLE']

    severe = [s for s in sells if s['severity'] >= 2]
    mild = [s for s in sells if s['severity'] == 1]
    slope_only = [s for s in sells if s['severity'] == 0]

    lines = [f"<b>⬡ NOX v3 Haftalik — {date_str}</b>"]
    if html_url:
        lines.append(f'🔗 <a href="{html_url}">Raporu Aç</a>')
    lines.append("")
    lines.append(f"📋 {n_scanned} taranan | {len(buys)} AL / {len(sells)} SAT / {len(candidates)} ADAY")
    lines.append("")

    def _fmt_signal(s):
        tb = s.get('tb_stage', '-')
        prep = s.get('tb_prep', 0)
        return (f"<b>{s['ticker']}</b> {s['close']:.2f} ◆{s['pivot']:.2f} "
                f"Δ{s['delta_pct']:+.0f}% {tb} PREP:{prep}")

    # HAZIR — Giris Sinyali
    if hazir:
        lines.append(f"<b>🟢 HAZIR — Giris Sinyali [{len(hazir)}]</b>")
        lines.append("─────────────────")
        for s in hazir:
            lines.append(_fmt_signal(s))
        lines.append("")

    # IZLE — Tetik Bekleniyor
    if izle:
        lines.append(f"<b>🟡 IZLE — Tetik Bekleniyor [{len(izle)}]</b>")
        lines.append("─────────────────")
        shown = izle[:5]
        for s in shown:
            lines.append(_fmt_signal(s))
        if len(izle) > 5:
            lines.append(f"...ve {len(izle) - 5} hisse daha")
        lines.append("")

    # BEKLE — Momentum Yok
    if bekle:
        lines.append(f"⚪ BEKLE — Momentum Yok ({len(bekle)} adet)")
        lines.append("")

    # SAT — Sert (tam liste)
    if severe:
        lines.append(f"<b>◆ SAT — Sert [{len(severe)}]</b>")
        lines.append("─────────────────")
        for s in severe:
            fresh_tag = '★' if s['fresh'] == 'BUGUN' else ''
            lines.append(
                f"{fresh_tag}<b>{s['ticker']}</b> {s['close']:.2f} ◆{s['pivot']:.2f} "
                f"Sev:{s['severity']} ADX:{s['adx']:.1f} [{s['signal_date']}]"
            )
        lines.append("")

    # SAT — Hafif + Slope (sadece sayi)
    mild_slope_cnt = len(mild) + len(slope_only)
    if mild_slope_cnt:
        lines.append(f"◆ SAT — Hafif/Slope: {mild_slope_cnt} hisse")
        lines.append("")

    return "\n".join(lines)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="NOX v3 PIVOT AL/SAT Screener")
    parser.add_argument('tickers', nargs='*', help='Spesifik ticker(lar)')
    parser.add_argument('--period', default='2y', help='Veri periyodu (default: 2y)')
    parser.add_argument('--daily', action='store_true', help='Sadece gunluk')
    parser.add_argument('--weekly', action='store_true', help='Sadece haftalik')
    parser.add_argument('--csv', action='store_true', help='CSV kaydet')
    parser.add_argument('--html', action='store_true', help='HTML rapor kaydet ve ac')
    parser.add_argument('--output', default='output', help='CSV/HTML cikti dizini')
    parser.add_argument('--debug', metavar='TICKER', help='Debug modu')
    parser.add_argument('--notify', action='store_true', help='Telegram + GitHub Pages bildirim')
    parser.add_argument('--backtest', type=int, metavar='N',
                        help='Son N Cuma icin haftalik CSV uret (backtest modu)')
    args = parser.parse_args()

    # Her ikisi de belirtilmediyse, ikisini de calistir
    run_daily = not args.weekly or args.daily
    run_weekly = not args.daily or args.weekly

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
    xu_df = data_mod.fetch_benchmark(period=args.period)
    print(f"  {len(all_data)} hisse yuklendi ({time.time() - t0:.1f}s)")
    if xu_df is not None:
        xu_df_lc = _to_lower_cols(xu_df)
        print(f"  XU100: {len(xu_df_lc)} bar")
    else:
        xu_df_lc = None
        print("  [!] XU100 yuklenemedi, RS hesaplanamayacak")

    if not all_data:
        print("  HATA: Hicbir hisse verisi yuklenemedi!")
        sys.exit(1)

    stock_dfs = {ticker: _to_lower_cols(df) for ticker, df in all_data.items()}

    # ── 2b. BACKTEST MODU ───────────────────────────────────────────────────
    if args.backtest:
        if not args.csv:
            print("  HATA: --backtest icin --csv gerekli!")
            sys.exit(1)
        # Backtest sadece haftalik calisir, weekly'yi zorla ac
        run_weekly = True
    if args.backtest and run_weekly and args.csv:
        last_data_date = stock_dfs[next(iter(stock_dfs))].index[-1].date()
        fridays = _get_past_fridays(args.backtest, last_data_date)
        print(f"\n  Backtest: {args.backtest} Cuma icin CSV uretiliyor...")
        print(f"  Tarihler: {fridays[0]} → {fridays[-1]}")
        for fri in fridays:
            cutoff = pd.Timestamp(fri)
            # Gunluk veriyi kes
            cut_dfs = {t: df[df.index <= cutoff] for t, df in stock_dfs.items()}
            cut_dfs = {t: df for t, df in cut_dfs.items() if len(df) >= 60}
            # Haftalik resample
            wk_dfs = {t: _to_weekly(df) for t, df in cut_dfs.items()}
            wk_dfs = {t: df for t, df in wk_dfs.items() if len(df) >= 30}
            # Tarama
            wb, ws, wc, wn, wd = _scan(wk_dfs, tf_label='weekly')
            # Gunluk tetik uygula
            triggered, zone_only = _apply_daily_triggers(wb, wc, cut_dfs, xu_df=xu_df_lc)
            # TB zenginlestirme (triggered sinyallere)
            triggered = _enrich_with_trend_birth(triggered, cut_dfs, wd)
            zone_only = _enrich_with_trend_birth(zone_only, cut_dfs, wd)
            # CSV kaydet (yeni format)
            _save_csv_v2(triggered, ws, wd, args.output, suffix='_weekly',
                         zone_only=zone_only)
            n_trig = len(triggered)
            n_zone = len(zone_only)
            trig_types = {}
            for t in triggered:
                tt = t.get('trigger_type', '?')
                trig_types[tt] = trig_types.get(tt, 0) + 1
            trig_str = ' '.join(f"{k}:{v}" for k, v in sorted(trig_types.items()))
            print(f"    {wd}: {wn} hisse, {n_trig} TETIK + {n_zone} ZONE + {len(ws)} SAT | {trig_str}")
        print(f"\n  Backtest tamamlandi. ({time.time() - t0:.1f}s)")
        return

    # ── 3. GUNLUK TARAMA ────────────────────────────────────────────────────
    if run_daily:
        print(f"\n  Gunluk tarama...")
        t1 = time.time()
        d_buys, d_sells, d_cands, d_n, d_date = _scan(stock_dfs, debug_ticker, tf_label='daily')
        print(f"  Gunluk tamamlandi ({time.time() - t1:.1f}s)")
        _print_results(d_buys, d_sells, d_cands, d_n, d_date, 'GUNLUK')
        if args.csv:
            _save_csv(d_buys, d_sells, d_date, args.output)

    # ── 4. HAFTALIK TARAMA + GUNLUK TETIK ──────────────────────────────────
    w_triggered = []
    w_zone_only = []
    if run_weekly:
        print(f"\n  Haftalik resample + tarama...")
        t2 = time.time()
        weekly_dfs = {t: _to_weekly(df) for t, df in stock_dfs.items()}
        weekly_dfs = {t: df for t, df in weekly_dfs.items() if len(df) >= 30}
        w_buys, w_sells, w_cands, w_n, w_date = _scan(weekly_dfs, debug_ticker, tf_label='weekly')
        # Gunluk tetik uygula
        print(f"  Gunluk tetik taramasi...")
        t2t = time.time()
        w_triggered, w_zone_only = _apply_daily_triggers(w_buys, w_cands, stock_dfs, xu_df=xu_df_lc)
        print(f"  Tetik tamamlandi ({time.time() - t2t:.1f}s): {len(w_triggered)} tetik, {len(w_zone_only)} zone-only")
        # TB zenginlestirme (triggered sinyallere)
        print(f"  Trend Birth zenginlestirme...")
        t2b = time.time()
        w_triggered = _enrich_with_trend_birth(w_triggered, stock_dfs, w_date)
        w_zone_only = _enrich_with_trend_birth(w_zone_only, stock_dfs, w_date)
        print(f"  Haftalik tamamlandi ({time.time() - t2:.1f}s, TB: {time.time() - t2b:.1f}s)")
        _print_triggered_results(w_triggered, w_sells, w_zone_only, w_n, w_date)
        if args.csv:
            _save_csv_v2(w_triggered, w_sells, w_date, args.output,
                         suffix='_weekly', zone_only=w_zone_only)

    # ── 5. Ozet ─────────────────────────────────────────────────────────────
    print(f"\n{'=' * 80}")
    print(f"  Toplam sure: {time.time() - t0:.1f}s")

    # Cakisma analizi: hem gunluk hem haftalik AL veren hisseler
    overlap = set()
    if run_daily and run_weekly:
        d_tickers = {b['ticker'] for b in d_buys}
        w_tickers = {b['ticker'] for b in w_triggered}
        overlap = d_tickers & w_tickers
        if overlap:
            print(f"\n  ★ CAKISMA (Gunluk + Haftalik AL): {', '.join(sorted(overlap))}")

    # ── 6. HTML rapor ─────────────────────────────────────────────────────
    html_path = None
    if args.html:
        html = _generate_html(
            d_buys if run_daily else [], d_sells if run_daily else [],
            d_cands if run_daily else [],
            d_n if run_daily else 0, d_date if run_daily else '',
            w_triggered if run_weekly else [], w_sells if run_weekly else [],
            w_zone_only if run_weekly else [],
            w_n if run_weekly else 0, w_date if run_weekly else '',
            overlap,
        )
        date_for_file = d_date if run_daily else w_date
        html_path = _save_html(html, date_for_file, args.output)
        if not args.notify:
            subprocess.Popen(['open', html_path])

    # ── 7. Telegram + GitHub Pages bildirim ─────────────────────────────
    if run_weekly and args.notify:
        html_url = None
        if args.html and html_path:
            html_url = push_html_to_github(html, 'nox_v3_weekly.html', w_date)
            send_telegram_document(html_path)
        msg = _format_weekly_telegram(w_triggered, w_sells, w_zone_only, w_n, w_date, html_url)
        send_telegram(msg)

    print(f"\n  NOX v3 tamamlandi.")


if __name__ == '__main__':
    main()
