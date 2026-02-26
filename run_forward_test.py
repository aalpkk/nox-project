#!/usr/bin/env python3
"""
NOX Forward Test Aracı
=======================
Tüm tarama CSV çıktılarını (Trend, Dip, Sideways, Rejim v3, Combo,
NOX v3, SMC, Pine, Divergence) okuyup, sinyal tarihinden itibaren
gerçek fiyat getirilerini (1g, 3g, 5g) hesaplayarak NOX dark-theme
interaktif HTML raporu oluşturur.

Kaynaklar:
  - output/   : NOX v3, SMC, Pine, Divergence CSV'leri
  - proje kökü: main.py çıktıları (trend, dip, sideways)
  - GitHub    : bist-tavan-screener artifact'leri (rejim_v3, combo) [--gh]

Kullanım:
    python run_forward_test.py                  # en son CSV'ler
    python run_forward_test.py --date 20260220  # belirli tarih
    python run_forward_test.py --open           # raporu tarayıcıda aç
    python run_forward_test.py --gh --open      # GitHub artifact + aç
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
# output/ dizini için pattern'lar
_CSV_PATTERNS_OUTPUT = [
    (re.compile(r'^nox_v3_signals_weekly_(\d{8})\.csv$'), 'nox_v3_weekly'),
    (re.compile(r'^nox_v3_signals_(\d{8})\.csv$'),        'nox_v3_daily'),
    (re.compile(r'^nox_smc_signals_(\d{8})\.csv$'),       'smc'),
    (re.compile(r'^pine_signals_(\d{8})\.csv$'),          'pine'),
    (re.compile(r'^nox_divergence_(\d{8})\.csv$'),        'divergence'),
    # GitHub artifact'leri de output/'a indirilir
    (re.compile(r'^rejim_v3_signals_(\d{8})\.csv$'),      'rejim_v3'),
    (re.compile(r'^combo_signals_(\d{8})\.csv$'),         'combo'),
]

# Proje kökü için pattern'lar (main.py çıktıları)
_CSV_PATTERNS_ROOT = [
    (re.compile(r'^nox_bist_trend_(\d{8})\.csv$'),    'trend'),
    (re.compile(r'^nox_bist_dip_(\d{8})\.csv$'),       'dip'),
    (re.compile(r'^nox_bist_sideways_(\d{8})\.csv$'),  'sideways'),
]


def _scan_dir(directory, patterns):
    """Bir dizini verilen pattern listesiyle tara → {screener: [(date_str, path), ...]}"""
    found = {}
    if not os.path.isdir(directory):
        return found
    for fname in os.listdir(directory):
        for pat, screener in patterns:
            m = pat.match(fname)
            if m:
                date_str = m.group(1)
                found.setdefault(screener, []).append((date_str, os.path.join(directory, fname)))
                break
    return found


def discover_csvs(output_dir, target_date=None):
    """output/ ve proje kökündeki CSV'leri keşfet ve screener tipine göre grupla.
    target_date: 'YYYYMMDD' formatında; None ise tüm tarihler dahil edilir.
    Dönüş: {screener: [(date_str, path), ...]} — tarih sıralı (eskiden yeniye)."""
    # İki dizini tara
    found = _scan_dir(output_dir, _CSV_PATTERNS_OUTPUT)
    root_dir = os.path.dirname(os.path.abspath(__file__))
    root_found = _scan_dir(root_dir, _CSV_PATTERNS_ROOT)
    # Birleştir
    for scr, items in root_found.items():
        found.setdefault(scr, []).extend(items)

    # Tarih filtresi
    result = {}
    for scr, items in found.items():
        items.sort(key=lambda x: x[0])  # eskiden yeniye
        if target_date:
            filtered = [(d, p) for d, p in items if d == target_date]
            if filtered:
                result[scr] = filtered
        else:
            result[scr] = items

    return result


def _parse_nox_v3(path, screener_name):
    """NOX v3 daily/weekly CSV parse → normalize sinyal listesi."""
    df = pd.read_csv(path)
    signals = []
    for _, row in df.iterrows():
        sig = str(row.get('signal', '')).strip()
        if sig == 'PIVOT_AL':
            direction = 'AL'
        elif sig == 'ADAY':
            direction = 'AL'
        elif sig == 'PIVOT_SAT':
            direction = 'SAT'
        else:
            continue
        entry = {
            'screener': screener_name,
            'ticker': str(row['ticker']).strip(),
            'signal_date': str(row['signal_date']).strip(),
            'direction': direction,
            'signal_type': sig,
            'entry_price': float(row['close']),
            'quality': None,
        }
        # Haftalik watchlist alanlari
        wl = str(row.get('wl_status', '')).strip()
        if wl and wl != 'nan':
            entry['wl_status'] = wl
            entry['tb_stage'] = str(row.get('tb_stage', '')).strip()
            entry['delta_pct'] = float(row['delta_pct']) if pd.notna(row.get('delta_pct')) else None
        signals.append(entry)
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


def _parse_rejim_v3(path, date_str):
    """Rejim v3 CSV parse → normalize sinyal listesi + kalite alanları."""
    df = pd.read_csv(path)
    sig_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
    signals = []
    for _, row in df.iterrows():
        signals.append({
            'screener': 'rejim_v3',
            'ticker': str(row['ticker']).strip(),
            'signal_date': sig_date,
            'direction': 'AL',
            'signal_type': str(row.get('signal', '')).strip(),
            'entry_price': float(row['close']),
            'quality': int(row['quality']) if pd.notna(row.get('quality')) else None,
            'rs_score': round(float(row['rs_score']), 1) if pd.notna(row.get('rs_score')) else None,
            'rs_pass': str(row.get('rs_pass', '')).strip().lower() == 'true',
            'adx_val': round(float(row['adx']), 1) if pd.notna(row.get('adx')) else None,
            'adx_slope_val': round(float(row['adx_slope']), 2) if pd.notna(row.get('adx_slope')) else None,
            'regime': str(row.get('regime', '')).strip(),
        })
    return signals


def _parse_trend_dip_sideways(path, screener_name, date_str):
    """Trend/Dip/Sideways CSV parse → normalize sinyal listesi.
    Dosya adındaki tarih kullanılır, tümü AL sinyali."""
    df = pd.read_csv(path)
    sig_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
    signals = []
    for _, row in df.iterrows():
        signals.append({
            'screener': screener_name,
            'ticker': str(row['ticker']).strip(),
            'signal_date': sig_date,
            'direction': 'AL',
            'signal_type': str(row.get('signal', '')).strip(),
            'entry_price': float(row['close']),
            'quality': int(row['quality']) if pd.notna(row.get('quality')) else None,
        })
    return signals


def _parse_combo(path, date_str):
    """Combo CSV parse → normalize sinyal listesi (basit format)."""
    df = pd.read_csv(path)
    sig_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
    signals = []
    for _, row in df.iterrows():
        signals.append({
            'screener': 'combo',
            'ticker': str(row['ticker']).strip(),
            'signal_date': sig_date,
            'direction': 'AL',
            'signal_type': str(row.get('signal', '')).strip(),
            'entry_price': float(row['close']),
            'quality': None,
        })
    return signals


def parse_all_csvs(csv_map):
    """Tüm CSV'leri parse et → birleşik sinyal listesi.
    csv_map: {screener: [(date_str, path), ...]}"""
    all_signals = []
    for screener, entries in csv_map.items():
        scr_total = 0
        for date_str, path in entries:
            try:
                if screener in ('nox_v3_daily', 'nox_v3_weekly'):
                    sigs = _parse_nox_v3(path, screener)
                elif screener == 'smc':
                    sigs = _parse_smc(path)
                elif screener == 'pine':
                    sigs = _parse_pine(path, date_str)
                elif screener == 'divergence':
                    sigs = _parse_divergence(path)
                elif screener == 'rejim_v3':
                    sigs = _parse_rejim_v3(path, date_str)
                elif screener in ('trend', 'dip', 'sideways'):
                    sigs = _parse_trend_dip_sideways(path, screener, date_str)
                elif screener == 'combo':
                    sigs = _parse_combo(path, date_str)
                else:
                    continue
                all_signals.extend(sigs)
                scr_total += len(sigs)
            except Exception as e:
                print(f"  ! {screener}/{date_str} parse hata: {e}")
        if scr_total > 0:
            n_dates = len(entries)
            extra = f" ({n_dates} tarih)" if n_dates > 1 else ""
            print(f"  {screener}: {scr_total} sinyal{extra}")
    return all_signals


# =============================================================================
# 2. GITHUB ARTİFACT İNDİRME
# =============================================================================

_GH_REPO = 'aalpkk/bist-tavan-screener'
_GH_ARTIFACT_PREFIX = 'signals-'


def fetch_github_csvs(output_dir):
    """GitHub Actions'taki tüm signals-* artifact'lerinden CSV'leri indir.
    rejim_v3_signals_YYYYMMDD.csv ve combo_signals_YYYYMMDD.csv → output_dir'a kopyalar.
    Zaten mevcut dosyalar tekrar indirilmez."""
    import shutil
    import tempfile

    print(f"\n  GitHub artifact'leri indiriliyor ({_GH_REPO})...")

    # Tüm signals-* artifact ID'lerini al
    try:
        result = subprocess.run(
            ['gh', 'api', f'repos/{_GH_REPO}/actions/artifacts',
             '--paginate', '--jq',
             r'.artifacts[] | select(.name | startswith("signals-")) | "\(.id)"'],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            print(f"  ! gh api hata: {result.stderr.strip()}")
            return
        artifact_ids = [x.strip() for x in result.stdout.strip().split('\n') if x.strip()]
        if not artifact_ids:
            print("  ! signals-* artifact bulunamadı.")
            return
        print(f"  {len(artifact_ids)} artifact bulundu")
    except FileNotFoundError:
        print("  ! gh CLI bulunamadı. GitHub artifact'leri için gh CLI gerekli.")
        return
    except Exception as e:
        print(f"  ! GitHub artifact listesi alınamadı: {e}")
        return

    # Her artifact'i indir ve CSV'leri topla
    copied = 0
    for art_id in artifact_ids:
        tmp_dir = tempfile.mkdtemp(prefix='nox_gh_')
        try:
            result = subprocess.run(
                ['gh', 'api', f'repos/{_GH_REPO}/actions/artifacts/{art_id}/zip'],
                capture_output=True, timeout=30,
            )
            if result.returncode != 0:
                continue
            zip_path = os.path.join(tmp_dir, 'a.zip')
            with open(zip_path, 'wb') as f:
                f.write(result.stdout)
            # zip aç
            subprocess.run(['unzip', '-qo', zip_path, '-d', tmp_dir],
                           capture_output=True, timeout=15)
            # CSV'leri kopyala (zaten varsa atla)
            for fname in os.listdir(tmp_dir):
                if not fname.endswith('.csv'):
                    continue
                if 'rejim_v3_signals_' not in fname and 'combo_signals_' not in fname:
                    continue
                dst = os.path.join(output_dir, fname)
                if os.path.exists(dst):
                    continue
                shutil.copy2(os.path.join(tmp_dir, fname), dst)
                print(f"    ← {fname}")
                copied += 1
        except Exception:
            pass
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    if copied == 0:
        print("  Yeni CSV yok (hepsi zaten mevcut)")
    else:
        print(f"  {copied} yeni CSV indirildi → {output_dir}")


# =============================================================================
# 3. VERİ ÇEKME
# =============================================================================

def fetch_price_data(signals):
    """Sinyallerdeki unique ticker'lar için fiyat verisi çek."""
    tickers = sorted(set(s['ticker'] for s in signals))
    print(f"\n  {len(tickers)} unique hisse için veri çekiliyor...")
    all_data = data_mod.fetch_data(tickers, period="6mo")
    xu = data_mod.fetch_benchmark(period="6mo")
    return all_data, xu


# =============================================================================
# 4. FORWARD GETİRİ HESAPLAMA
# =============================================================================

WINDOWS = [1, 3, 5]


def _compute_indicators(df):
    """Bir hisse için RSI(14) ve MACD(12,26,9) histogram hesapla."""
    close = df['Close']
    # RSI(14) — Wilder's smoothing
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14).mean()
    rsi = 100 - 100 / (1 + avg_gain / avg_loss.replace(0, np.nan))
    # MACD(12,26,9)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = macd_line - signal_line
    return rsi, macd_hist


def compute_forward_returns(signals, all_data, xu_df):
    """Her sinyal için 1g, 3g, 5g forward getiri + XU100 kıyasla."""
    results = []
    _ind_cache = {}  # ticker -> (rsi_series, macd_hist_series)
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

        # Rejim v3 sinyalleri icin RSI/MACD hesapla
        if sig['screener'] == 'rejim_v3':
            if ticker not in _ind_cache:
                _ind_cache[ticker] = _compute_indicators(df)
            rsi_s, macd_s = _ind_cache[ticker]
            if idx < len(rsi_s):
                v = rsi_s.iloc[idx]
                if pd.notna(v):
                    row['rsi_at_signal'] = round(float(v), 1)
            if idx < len(macd_s):
                v = macd_s.iloc[idx]
                if pd.notna(v):
                    row['macd_hist'] = round(float(v), 4)

        results.append(row)

    return results


# =============================================================================
# 5. ÖZET İSTATİSTİKLER
# =============================================================================

def _calc_window_stats(subset, windows):
    """Alt küme için pencere bazlı istatistikler hesapla."""
    stats = {}
    for w in windows:
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
            for k in ['wr', 'avg', 'med', 'best', 'worst']:
                stats[f'{k}_{w}d'] = None
    return stats


def compute_summary(results):
    """Screener bazlı özet istatistikler."""
    summary = {}
    all_screeners = sorted(set(r['screener'] for r in results))

    for scr in ['genel'] + all_screeners:
        if scr == 'genel':
            subset = results
        else:
            subset = [r for r in results if r['screener'] == scr]

        if not subset:
            continue

        stats = {'screener': scr, 'n': len(subset)}
        stats.update(_calc_window_stats(subset, WINDOWS))

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

    # Haftalik watchlist durum bazli kirilim (HAZIR/IZLE/BEKLE)
    wl_results = [r for r in results if r.get('wl_status')]
    if wl_results:
        for wl_st in ['HAZIR', 'İZLE', 'BEKLE']:
            wl_sub = [r for r in wl_results if r.get('wl_status') == wl_st]
            if not wl_sub:
                continue
            key = f'wl_{wl_st}'
            stats = {'screener': key, 'n': len(wl_sub)}
            stats.update(_calc_window_stats(wl_sub, WINDOWS))
            for d in ['AL', 'SAT']:
                d_subset = [r for r in wl_sub if r['direction'] == d]
                stats[f'n_{d}'] = len(d_subset)
                vals_5 = [r['ret_5d'] for r in d_subset if r.get('ret_5d') is not None]
                if vals_5:
                    stats[f'wr_5d_{d}'] = round(sum(1 for v in vals_5 if v > 0) / len(vals_5) * 100, 1)
                    stats[f'avg_5d_{d}'] = round(sum(vals_5) / len(vals_5), 2)
                else:
                    stats[f'wr_5d_{d}'] = None
                    stats[f'avg_5d_{d}'] = None
            summary[key] = stats

    # Rejim v3 kalite bazli kirilim
    r3_results = [r for r in results if r.get('screener') == 'rejim_v3']
    if r3_results:
        # Filtre tanimlari: (anahtar, etiket, filtre fonksiyonu)
        r3_filters = [
            ('r3_q≥70',   lambda r: (r.get('quality') or 0) >= 70),
            ('r3_rs✓',    lambda r: r.get('rs_pass', False)),
            ('r3_adx≥25', lambda r: (r.get('adx_val') or 0) >= 25),
            ('r3_rsi≤40', lambda r: (r.get('rsi_at_signal') or 999) <= 40),
            ('r3_macd>0', lambda r: (r.get('macd_hist') or -1) > 0),
            ('r3_filtreli', lambda r: (
                (r.get('quality') or 0) >= 50
                and r.get('rs_pass', False)
                and (r.get('adx_val') or 0) >= 20
            )),
        ]
        for fkey, ffn in r3_filters:
            sub = [r for r in r3_results if ffn(r)]
            if not sub:
                continue
            stats = {'screener': fkey, 'n': len(sub)}
            stats.update(_calc_window_stats(sub, WINDOWS))
            for d in ['AL', 'SAT']:
                d_subset = [r for r in sub if r['direction'] == d]
                stats[f'n_{d}'] = len(d_subset)
                vals_5 = [r['ret_5d'] for r in d_subset if r.get('ret_5d') is not None]
                if vals_5:
                    stats[f'wr_5d_{d}'] = round(sum(1 for v in vals_5 if v > 0) / len(vals_5) * 100, 1)
                    stats[f'avg_5d_{d}'] = round(sum(vals_5) / len(vals_5), 2)
                else:
                    stats[f'wr_5d_{d}'] = None
                    stats[f'avg_5d_{d}'] = None
            summary[fkey] = stats

        # Sinyal tipi bazli kirilim
        sig_types = sorted(set(r.get('signal_type', '') for r in r3_results))
        for st in sig_types:
            if not st:
                continue
            sub = [r for r in r3_results if r.get('signal_type') == st]
            if not sub:
                continue
            fkey = f'r3s_{st}'
            stats = {'screener': fkey, 'n': len(sub)}
            stats.update(_calc_window_stats(sub, WINDOWS))
            for d in ['AL', 'SAT']:
                d_subset = [r for r in sub if r['direction'] == d]
                stats[f'n_{d}'] = len(d_subset)
                vals_5 = [r['ret_5d'] for r in d_subset if r.get('ret_5d') is not None]
                if vals_5:
                    stats[f'wr_5d_{d}'] = round(sum(1 for v in vals_5 if v > 0) / len(vals_5) * 100, 1)
                    stats[f'avg_5d_{d}'] = round(sum(vals_5) / len(vals_5), 2)
                else:
                    stats[f'wr_5d_{d}'] = None
                    stats[f'avg_5d_{d}'] = None
            summary[fkey] = stats

    return summary


# =============================================================================
# 6. HTML RAPOR
# =============================================================================

_SCREENER_LABELS = {
    'genel': 'Genel',
    'trend': 'Trend',
    'dip': 'Dip',
    'sideways': 'Sideways',
    'rejim_v3': 'Rejim v3',
    'r3_q≥70': 'R3 Q≥70',
    'r3_rs✓': 'R3 RS✓',
    'r3_adx≥25': 'R3 ADX≥25',
    'r3_rsi≤40': 'R3 RSI≤40',
    'r3_macd>0': 'R3 MACD>0',
    'r3_filtreli': 'R3 Filtreli',
    'r3s_GUCLU': 'R3 GÜÇLÜ',
    'r3s_CMB': 'R3 CMB',
    'r3s_CMB+': 'R3 CMB+',
    'r3s_BILESEN': 'R3 BİLEŞEN',
    'r3s_ZAYIF': 'R3 ZAYIF',
    'r3s_ERKEN': 'R3 ERKEN',
    'r3s_DONUS': 'R3 DÖNÜŞ',
    'r3s_MR': 'R3 MR',
    'r3s_PB': 'R3 PB',
    'combo': 'Combo',
    'nox_v3_daily': 'NOX v3 Günlük',
    'nox_v3_weekly': 'NOX v3 Haftalık',
    'wl_HAZIR': 'WL HAZIR',
    'wl_İZLE': 'WL İZLE',
    'wl_BEKLE': 'WL BEKLE',
    'smc': 'SMC',
    'pine': 'Pine',
    'divergence': 'Uyumsuzluk',
}

_SCREENER_TAB_ORDER = [
    'genel', 'trend', 'dip', 'sideways', 'rejim_v3',
    'r3_q≥70', 'r3_rs✓', 'r3_adx≥25', 'r3_rsi≤40', 'r3_macd>0', 'r3_filtreli',
    'r3s_GUCLU', 'r3s_CMB', 'r3s_CMB+', 'r3s_BILESEN', 'r3s_ZAYIF',
    'r3s_ERKEN', 'r3s_DONUS', 'r3s_MR', 'r3s_PB',
    'combo',
    'nox_v3_daily', 'nox_v3_weekly', 'wl_HAZIR', 'wl_İZLE', 'wl_BEKLE',
    'smc', 'pine', 'divergence',
]


def generate_html(results, summary, csv_map):
    """NOX dark-theme interaktif HTML rapor."""
    now = datetime.now().strftime('%d.%m.%Y %H:%M')
    all_dates = sorted(set(d for entries in csv_map.values() for d, _ in entries))
    if len(all_dates) <= 3:
        date_label = ', '.join(all_dates)
    else:
        date_label = f"{all_dates[0]} → {all_dates[-1]} ({len(all_dates)} gün)"

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
.wl-badge {{
  display: inline-block; padding: 2px 6px; border-radius: 4px;
  font-size: 0.62rem; font-weight: 700; font-family: var(--font-mono);
}}
.wl-hazir {{ background: rgba(74,222,128,0.18); color: var(--nox-green); }}
.wl-izle  {{ background: rgba(250,204,21,0.15); color: var(--nox-yellow); }}
.wl-bekle {{ background: rgba(113,113,122,0.15); color: var(--text-muted); }}
.tb-ok   {{ color: var(--nox-green); font-weight: 700; }}
.tb-trg  {{ color: var(--nox-cyan); font-weight: 700; }}
.tb-prep {{ color: var(--nox-yellow); }}
.tb-none {{ color: var(--text-muted); }}
.r3-sub {{ font-size: 0.72rem !important; padding: 7px 12px !important; }}
.q-badge {{
  display: inline-block; padding: 2px 6px; border-radius: 4px;
  font-size: 0.62rem; font-weight: 700; font-family: var(--font-mono);
}}
.q-high {{ background: rgba(74,222,128,0.18); color: var(--nox-green); }}
.q-mid  {{ background: rgba(250,204,21,0.15); color: var(--nox-yellow); }}
.q-low  {{ background: rgba(248,113,113,0.12); color: var(--nox-red); }}
.rs-pass {{ color: var(--nox-green); font-weight: 700; }}
.rs-fail {{ color: var(--text-muted); }}
.r3-note {{
  background: rgba(34,211,238,0.06); border: 1px solid rgba(34,211,238,0.2);
  border-radius: var(--radius); padding: 14px 18px; margin-bottom: 16px;
  font-size: 0.78rem; line-height: 1.6; display: none;
}}
.r3-note b {{ color: var(--nox-cyan); }}
.r3-note .rule {{ color: var(--nox-green); font-weight: 600; }}
.r3-note .warn {{ color: var(--nox-yellow); font-size: 0.72rem; }}
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
  <div><label>WL</label>
  <select id="fWL" onchange="af()"><option value="">Tümü</option>
  <option value="HAZIR">HAZIR</option><option value="İZLE">İZLE</option>
  <option value="BEKLE">BEKLE</option></select></div>
  <div><label>Q≥</label><input type="number" id="fQ" value="" step="10" min="0" max="100" style="width:55px" placeholder="min" oninput="af()"></div>
  <div><label>RS</label>
  <select id="fRS" onchange="af()"><option value="">Tümü</option>
  <option value="pass">Pass</option><option value="fail">Fail</option></select></div>
  <div><button class="nox-btn" onclick="resetF()">Sıfırla</button></div>
</div>

<!-- R3 HATIRLAT NOT -->
<div class="r3-note" id="r3note">
  <b>Rejim v3 — Forward Test Filtre Kuralları</b><br>
  <span class="rule">1) RS 30-60</span> — Relative Strength: düşük RS zayıf, yüksek RS aşırı uzamış; 30-60 bandı en iyi getiri<br>
  <span class="rule">2) MACD &gt; 0</span> — Momentum pozitif; MACD negatifken sinyaller kaybettiriyor<br>
  <span class="rule">3) Q ≥ 70</span> — Kalite skoru yüksek; düşük kalite sinyaller gürültü<br>
  3 kural birlikte → <b>5G +9.52%, WR %73.3</b> (N=15)<br>
  <span class="warn">⚠ XU100 düşüş döneminde RS 30-60 çok daha güçlü (+4.36% vs -0.47% uptrend)</span><br>
  <span class="warn">⚠ SMC SAT WR %62.2 güçlü — SAT tarafında SMC sinyallerini kullan</span>
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
<th onclick="sb('delta_pct')">Δ%</th>
<th onclick="sb('ret_1d')">1G%</th>
<th onclick="sb('ret_3d')">3G%</th>
<th onclick="sb('ret_5d')">5G%</th>
<th onclick="sb('xu_5d')">XU100 5G</th>
<th onclick="sb('excess_5d')">Fazla 5G</th>
<th onclick="sb('quality')">Q</th>
<th onclick="sb('rs_score')">RS</th>
<th onclick="sb('rsi_at_signal')">RSI</th>
<th onclick="sb('macd_hist')">MACD</th>
<th onclick="sb('adx_val')">ADX</th>
<th onclick="sb('wl_status')">WL</th>
<th onclick="sb('tb_stage')">TB</th>
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

// R3 kalite + sinyal tipi filtreleri
const R3F={{
  'r3_q≥70':r=>(r.quality||0)>=70,
  'r3_rs✓':r=>r.rs_pass===true,
  'r3_adx≥25':r=>(r.adx_val||0)>=25,
  'r3_rsi≤40':r=>(r.rsi_at_signal||999)<=40,
  'r3_macd>0':r=>(r.macd_hist||-1)>0,
  'r3_filtreli':r=>(r.quality||0)>=50&&r.rs_pass===true&&(r.adx_val||0)>=20,
  'r3s_GUCLU':r=>r.signal_type==='GUCLU',
  'r3s_CMB':r=>r.signal_type==='CMB',
  'r3s_CMB+':r=>r.signal_type==='CMB+',
  'r3s_BILESEN':r=>r.signal_type==='BILESEN',
  'r3s_ZAYIF':r=>r.signal_type==='ZAYIF',
  'r3s_ERKEN':r=>r.signal_type==='ERKEN',
  'r3s_DONUS':r=>r.signal_type==='DONUS',
  'r3s_MR':r=>r.signal_type==='MR',
  'r3s_PB':r=>r.signal_type==='PB',
}};

// ── TABS ──
function initTabs(){{
  const el=document.getElementById('tabs');
  TABS.forEach(t=>{{
    const d=document.createElement('div');
    const isR3=t.startsWith('r3_');
    d.className='nox-tab'+(t==='genel'?' active':'')+(isR3?' r3-sub':'');
    d.id='tab-'+t;
    let n;
    if(t==='genel') n=D.length;
    else if(t.startsWith('wl_')) n=D.filter(r=>r.wl_status===t.replace('wl_','')).length;
    else if(R3F[t]) n=D.filter(r=>r.screener==='rejim_v3'&&R3F[t](r)).length;
    else n=D.filter(r=>r.screener===t).length;
    d.innerHTML=(LBL[t]||t)+' <span class="cnt">'+n+'</span>';
    d.onclick=()=>{{curTab=t;
      document.querySelectorAll('.nox-tab').forEach(x=>x.classList.remove('active'));
      d.classList.add('active');
      updateR3Note();renderStats();af()}};
    el.appendChild(d);
  }});
}}

// ── R3 NOT GÖSTERİMİ ──
function updateR3Note(){{
  const el=document.getElementById('r3note');
  const show=curTab==='rejim_v3'||curTab.startsWith('r3_')||curTab.startsWith('r3s_');
  el.style.display=show?'block':'none';
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
  let tabs;
  if(curTab==='genel') tabs=TABS;
  else if(curTab==='rejim_v3') tabs=['genel','rejim_v3',...Object.keys(R3F).filter(k=>S[k])];
  else if(R3F[curTab]) tabs=['genel','rejim_v3',curTab];
  else tabs=['genel',curTab];
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
  const fwl=document.getElementById('fWL').value;
  const fq=parseFloat(document.getElementById('fQ').value);
  const frs=document.getElementById('fRS').value;
  let f=D.filter(r=>{{
    // Tab filtresi
    if(R3F[curTab]){{
      if(r.screener!=='rejim_v3')return false;
      if(!R3F[curTab](r))return false;
    }}else if(curTab.startsWith('wl_')){{
      const wlKey=curTab.replace('wl_','');
      if(r.wl_status!==wlKey)return false;
    }}else if(curTab!=='genel'&&r.screener!==curTab)return false;
    // Genel filtreler
    if(sr&&!r.ticker.includes(sr))return false;
    if(fd&&r.direction!==fd)return false;
    if(fs&&r.status!==fs)return false;
    if(fwl&&r.wl_status!==fwl)return false;
    // R3 kalite filtreleri
    if(!isNaN(fq)&&fq>0&&(r.quality==null||r.quality<fq))return false;
    if(frs==='pass'&&!r.rs_pass)return false;
    if(frs==='fail'&&r.rs_pass)return false;
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
  document.getElementById('fWL').value='';
  document.getElementById('fQ').value='';
  document.getElementById('fRS').value='';
  af();
}}

// ── RENDER ──
function retCell(v){{
  if(v==null)return '<td style="color:var(--text-muted)">—</td>';
  const c=v>0?'var(--nox-green)':v<0?'var(--nox-red)':'var(--text-muted)';
  return '<td style="color:'+c+';font-weight:600">'+(v>0?'+':'')+v.toFixed(2)+'%</td>';
}}

function mkQCell(v){{
  if(v==null)return '<td style="color:var(--text-muted)">—</td>';
  const cls=v>=70?'q-high':v>=40?'q-mid':'q-low';
  return '<td><span class="q-badge '+cls+'">'+v+'</span></td>';
}}
function mkRsCell(score,pass){{
  if(score==null)return '<td style="color:var(--text-muted)">—</td>';
  const cls=pass?'rs-pass':'rs-fail';
  return '<td class="'+cls+'">'+score+(pass?' ✓':'')+'</td>';
}}
function mkIndCell(v,digits){{
  if(v==null)return '<td style="color:var(--text-muted)">—</td>';
  return '<td style="font-family:var(--font-mono);font-size:.7rem">'+v.toFixed(digits||1)+'</td>';
}}
function mkMacdCell(v){{
  if(v==null)return '<td style="color:var(--text-muted)">—</td>';
  const c=v>0?'var(--nox-green)':v<0?'var(--nox-red)':'var(--text-muted)';
  return '<td style="color:'+c+';font-family:var(--font-mono);font-size:.7rem">'+(v>0?'+':'')+v.toFixed(2)+'</td>';
}}
function mkWlBadge(s){{
  if(!s)return'<td>—</td>';
  const cls=s==='HAZIR'?'wl-hazir':s==='İZLE'?'wl-izle':'wl-bekle';
  return '<td><span class="wl-badge '+cls+'">'+s+'</span></td>';
}}
function mkTbCell(s){{
  if(!s)return'<td style="color:var(--text-muted)">—</td>';
  const cls=s==='OK'?'tb-ok':s==='TRG'?'tb-trg':s==='PREP'?'tb-prep':'tb-none';
  return '<td class="'+cls+'">'+s+'</td>';
}}

function render(data){{
  const tb=document.getElementById('tb');tb.innerHTML='';
  data.forEach(r=>{{
    const tr=document.createElement('tr');
    const dirC=r.direction==='AL'?'dir-al':'dir-sat';
    const stC='status-'+(r.status==='tamam'?'tamam':r.status==='kısmi'?'kismi':'bekliyor');
    const stL=r.status==='tamam'?'Tamam':r.status==='kısmi'?'Kısmi':'Bekliyor';
    const deltaCell=r.delta_pct!=null?
      '<td style="color:'+(r.delta_pct<=10?'var(--nox-green)':r.delta_pct<=20?'var(--nox-yellow)':'var(--nox-red)')+'">'+
      (r.delta_pct>0?'+':'')+r.delta_pct.toFixed(1)+'%</td>':'<td style="color:var(--text-muted)">—</td>';
    tr.innerHTML=`<td><a class="tv-link" href="https://www.tradingview.com/chart/?symbol=BIST:${{r.ticker}}" target="_blank">${{r.ticker}}</a></td>
      <td style="color:var(--text-muted);font-size:.68rem">${{LBL[r.screener]||r.screener}}</td>
      <td style="font-size:.68rem">${{r.signal_type}}</td>
      <td class="${{dirC}}">${{r.direction}}</td>
      <td style="color:var(--text-muted)">${{r.signal_date}}</td>
      <td>${{r.entry_price.toFixed(2)}}</td>
      ${{deltaCell}}
      ${{retCell(r.ret_1d)}}${{retCell(r.ret_3d)}}${{retCell(r.ret_5d)}}
      ${{retCell(r.xu_5d)}}${{retCell(r.excess_5d)}}
      ${{mkQCell(r.quality)}}${{mkRsCell(r.rs_score,r.rs_pass)}}
      ${{mkIndCell(r.rsi_at_signal)}}${{mkMacdCell(r.macd_hist)}}${{mkIndCell(r.adx_val)}}
      ${{mkWlBadge(r.wl_status)}}${{mkTbCell(r.tb_stage)}}
      <td class="${{stC}}" style="font-size:.7rem;font-weight:600">${{stL}}</td>`;
    tb.appendChild(tr);
  }});
  document.getElementById('st').innerHTML='<b>'+data.length+'</b> / '+D.length+' sinyal';
  renderStats();
}}

// ── INIT ──
initTabs();
updateR3Note();
renderStats();
af();
</script></body></html>"""
    return html


# =============================================================================
# 7. CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="NOX Forward Test Aracı")
    parser.add_argument('--output', default='output', help='CSV/HTML dizini (default: output)')
    parser.add_argument('--date', help='Belirli tarih (YYYYMMDD)')
    parser.add_argument('--open', action='store_true', help='Raporu tarayıcıda aç')
    parser.add_argument('--gh', action='store_true',
                        help='GitHub artifact\'lerini indir (rejim_v3, combo)')
    args = parser.parse_args()

    output_dir = args.output
    if not os.path.isdir(output_dir):
        print(f"  HATA: {output_dir} dizini bulunamadı!")
        sys.exit(1)

    # ── 0. GitHub Artifact'leri ──
    if args.gh:
        fetch_github_csvs(output_dir)

    # ── 1. CSV Keşfet ──
    print(f"\n  CSV dosyaları taranıyor ({output_dir} + proje kökü)...")
    csv_map = discover_csvs(output_dir, args.date)
    if not csv_map:
        print("  HATA: Hiçbir CSV bulunamadı!")
        sys.exit(1)
    for scr, entries in csv_map.items():
        dates_list = [d for d, _ in entries]
        if len(entries) == 1:
            print(f"  {scr}: {dates_list[0]} → {os.path.basename(entries[0][1])}")
        else:
            print(f"  {scr}: {len(entries)} dosya ({dates_list[0]}..{dates_list[-1]})")

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
    all_dates = sorted(set(d for entries in csv_map.values() for d, _ in entries))
    date_str = all_dates[-1] if all_dates else datetime.now().strftime('%Y%m%d')
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
