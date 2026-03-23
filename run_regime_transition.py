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
    classify_volume_quality,
    RegimeTransitionSignal, RT_CFG, REGIME_NAMES,
    TIMEFRAME_CONFIGS,
)
from collections import Counter
from core.reports import _NOX_CSS, _sanitize, send_telegram, push_html_to_github


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


def _to_weekly(df, include_incomplete=False):
    weekly = df.resample('W-FRI').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
    }).dropna(subset=['close'])
    if not include_incomplete and len(weekly) > 0:
        last_friday = weekly.index[-1]
        last_data_date = df.index[-1]
        if last_data_date < last_friday:
            weekly = weekly.iloc[:-1]
    return weekly


def _to_monthly(df, include_incomplete=False):
    monthly = df.resample('ME').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
    }).dropna(subset=['close'])
    if not include_incomplete and len(monthly) > 0:
        last_month_end = monthly.index[-1]
        last_data_date = df.index[-1]
        if last_data_date < last_month_end:
            monthly = monthly.iloc[:-1]
    return monthly


def _entry_window_tf(days_since, timeframe='daily'):
    """Gecisten bu yana gecen gune gore giris penceresi etiketi (timeframe-aware)."""
    if timeframe == 'weekly':
        if days_since <= 7:
            return 'TAZE'
        elif days_since <= 28:
            return 'BEKLE'
        elif days_since <= 84:
            return '2.DALGA'
        else:
            return 'GEC'
    elif timeframe == 'monthly':
        if days_since <= 31:
            return 'TAZE'
        elif days_since <= 93:
            return 'BEKLE'
        elif days_since <= 279:
            return '2.DALGA'
        else:
            return 'GEC'
    else:  # daily
        if days_since <= 1:
            return 'TAZE'
        elif days_since <= 7:
            return 'BEKLE'
        elif days_since <= 20:
            return '2.DALGA'
        else:
            return 'GEC'


_TF_LABELS = {'daily': 'Gunluk', 'weekly': 'Haftalik', 'monthly': 'Aylik'}
_TF_DEFAULT_PERIOD = {'daily': '2y', 'weekly': '5y', 'monthly': '10y'}
_TF_MIN_BARS = {'daily': 100, 'weekly': 50, 'monthly': 24}

# Gunluk pullback esikleri
_PB_EMA_DIST_PCT = 3.0   # EMA21'e %3'ten yakin = pullback
_PB_RSI_MAX = 55          # RSI(14) <= 55 = asiri alimda degil


def _compute_weekly_al_with_daily_pb(weekly_dfs, monthly_dfs, daily_dfs, cfg=None):
    """Haftalik rejim taramasi yapip AL aktif ticker'lari bul,
    ardindan gunluk veride EMA21 pullback kontrolu yap.
    Returns: {ticker: {'weekly_al': True, 'pb': bool, 'ema_dist': float, 'rsi': float,
                        'weekly_transition': str, 'weekly_regime': str}}
    """
    from markets.bist.regime_transition import TIMEFRAME_CONFIGS
    w_cfg = cfg or TIMEFRAME_CONFIGS.get('weekly', RT_CFG)
    min_bars = _TF_MIN_BARS['weekly']
    result = {}

    for ticker, wdf in weekly_dfs.items():
        if len(wdf) < min_bars:
            continue
        try:
            htf_df = monthly_dfs.get(ticker) if monthly_dfs else None
            out = scan_regime_transition(wdf, weekly_df=htf_df, cfg=w_cfg)
            trade_state = compute_trade_state(out['regime'], out['close'], out['ema21'])
            last = len(wdf) - 1
            if not trade_state['in_trade'].iloc[last]:
                continue

            # Haftalik AL aktif — gecis bilgisi
            trade_start_idx = int(trade_state['trade_start_idx'].iloc[last])
            prev_r = int(out['regime'].iloc[trade_start_idx - 1]) if trade_start_idx > 0 else 0
            curr_r = int(out['regime'].iloc[trade_start_idx])
            transition = f"{REGIME_NAMES.get(prev_r, '?')}→{REGIME_NAMES.get(curr_r, '?')}"
            regime_name = REGIME_NAMES.get(int(out['regime'].iloc[last]), '?')

            # Gunluk pullback kontrolu
            ddf = daily_dfs.get(ticker)
            pb = False
            ema_dist = None
            rsi_val = None
            if ddf is not None and len(ddf) >= 30:
                close = ddf['close']
                ema21 = close.ewm(span=21, adjust=False).mean()
                last_close = float(close.iloc[-1])
                last_ema = float(ema21.iloc[-1])
                if last_ema > 0:
                    ema_dist = round((last_close - last_ema) / last_ema * 100, 1)
                    delta = close.diff()
                    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
                    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
                    rs = gain / loss.replace(0, np.nan)
                    rsi = 100 - 100 / (1 + rs)
                    rsi_val = round(float(rsi.iloc[-1]), 1) if pd.notna(rsi.iloc[-1]) else None
                    pb = abs(ema_dist) <= _PB_EMA_DIST_PCT and (rsi_val or 50) <= _PB_RSI_MAX

            result[ticker] = {
                'weekly_al': True,
                'pb': pb,
                'ema_dist': ema_dist,
                'rsi': rsi_val,
                'weekly_transition': transition,
                'weekly_regime': regime_name,
            }
        except Exception:
            continue

    return result


# =============================================================================
# TARAMA
# =============================================================================

def _scan_all(stock_dfs, higher_tf_dfs=None, scan_bars=60, cfg=None, timeframe='daily'):
    """Tum hisselerde regime transition taramasi.
    Sticky AL: trade aktif hisseleri listeler, OE skoru hesaplar."""
    cfg = cfg or RT_CFG
    min_bars = _TF_MIN_BARS.get(timeframe, 100)
    results = []
    n_scanned = 0
    last_date = None
    regime_dist = {0: 0, 1: 0, 2: 0, 3: 0}

    for ticker, df in stock_dfs.items():
        if len(df) < min_bars:
            continue
        if _is_halted(df):
            continue

        try:
            htf_df = higher_tf_dfs.get(ticker) if higher_tf_dfs else None
            out = scan_regime_transition(df, weekly_df=htf_df, cfg=cfg)
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
            atr_thresh = cfg.get('entry_atr_pct_thresh', 3.0)
            # 1. Dusuk volatilite (ATR% < threshold)
            if atr_pct < atr_thresh:
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
            oe = calc_oe_score(df, out['ema21'], cfg=cfg)

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
            window = _entry_window_tf(s.days_since)
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

def _save_csv(results, date_str, output_dir, timeframe='daily',
              weekly_al_pb=None):
    os.makedirs(output_dir, exist_ok=True)
    rows = []
    for s in results:
        row = {
            'ticker': s.ticker,
            'date': s.date.strftime('%Y-%m-%d') if hasattr(s.date, 'strftime') else str(s.date),
            'regime': s.regime,
            'regime_name': s.regime_name,
            'transition': s.transition,
            'transition_date': s.transition_date.strftime('%Y-%m-%d') if hasattr(s.transition_date, 'strftime') else '',
            'days_since': s.days_since,
            'entry_window': _entry_window_tf(s.days_since, timeframe),
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
        }
        # Hacim-donus tier
        _atr_pct_csv = round((s.atr / s.close * 100), 2) if s.close > 0 else 0
        _vt_csv, _ = classify_volume_quality(
            _atr_pct_csv, s.cmf, s.rvol, s.participation_score, s.oe_score)
        row['atr_pct'] = _atr_pct_csv
        row['vol_tier'] = _vt_csv
        # Badge: H+PB (haftalik AL + gunluk pullback), H+AL (haftalik AL aktif)
        if weekly_al_pb and s.ticker in weekly_al_pb:
            info = weekly_al_pb[s.ticker]
            row['badge'] = 'H+PB' if info['pb'] else 'H+AL'
        rows.append(row)
    if rows:
        csv_df = pd.DataFrame(rows)
        tf_suffix = f"_{timeframe}" if timeframe != 'daily' else ''
        fname = f"regime_transition{tf_suffix}_{date_str.replace('-', '')}.csv"
        path = os.path.join(output_dir, fname)
        csv_df.to_csv(path, index=False)
        print(f"\n  CSV: {path}")
    else:
        print(f"\n  Sinyal yok, CSV olusturulmadi.")


# =============================================================================
# HTML RAPOR
# =============================================================================

_TD = 'style="padding:2px 6px"'
_TD_G = 'style="padding:2px 6px;color:var(--nox-green)"'
_TD_R = 'style="padding:2px 6px;color:var(--nox-red)"'
_TD_M = 'style="padding:2px 6px;color:var(--text-muted)"'
_ROW_B = 'style="border-bottom:1px solid rgba(255,255,255,0.06)"'
_ROW_H = 'style="border-bottom:1px solid rgba(255,255,255,0.12)"'


def _wr_td(wr):
    if wr >= 58:
        return _TD_G
    elif wr < 50:
        return _TD_R
    return _TD


def _build_trade_guide_extra(timeframe):
    """Haftalik/aylik timeframe icin ek trade rehberi HTML blogu."""
    if timeframe == 'daily':
        return ''

    if timeframe == 'weekly':
        return f"""
<details class="trade-guide" style="margin-bottom:14px;padding:10px 14px;border-radius:8px;
  background:rgba(250,204,21,0.06);border:1px solid rgba(250,204,21,0.15);
  font-size:0.78rem;color:var(--text-secondary);line-height:1.6">
<summary style="cursor:pointer;font-weight:700;color:var(--nox-yellow);font-size:0.82rem">
  Haftalik Trade Rehberi</summary>
<div style="margin-top:8px">
<b style="color:var(--nox-green)">Nasil Trade Edilir? (Haftalik)</b><br>
&#8226; Haftalik rejim gecisi = <b>orta vadeli trend degisimi</b> (haftalar-aylar)<br>
&#8226; Giris: Haftalik mum kapanisinda rejim yukselince, gunluk grafige gecip <b>pullback/destek</b> noktasindan gir<br>
&#8226; Pozisyon buyuklugu: Gunluge gore <b>2-3x daha buyuk</b> pozisyon, cunku stop daha genis<br>
&#8226; Tutma suresi: <b>3-10 hafta</b> tipik | Exit stage 2+ olunca daralt<br>
&#8226; Stop: Haftalik swing low - 0.5*ATR (gunluge gore daha genis, %5-10 arasi normal)<br>
&#8226; Cikis: Haftalik kapanis EMA21 altinda veya Exit Stage >=2<br>
<br>
<b style="color:var(--nox-cyan)">Backtest Sonuclari (80 hafta, 531 hisse, N=3683 AL)</b>
<table style="width:100%;font-size:0.74rem;margin-top:4px;margin-bottom:8px;border-collapse:collapse">
<tr {_ROW_H}>
  <td colspan="4" style="padding:4px 6px;color:var(--nox-green);font-weight:700">Pencere Bazinda</td></tr>
<tr {_ROW_B}>
  <td {_TD}><b>3 Hafta</b></td>
  <td {_TD}>WR %50.3</td>
  <td {_TD}>Ort +2.23%</td>
  <td {_TD_M}>N=3699</td></tr>
<tr {_ROW_B}>
  <td {_TD}><b>5 Hafta</b></td>
  <td {_TD}>WR %50.4</td>
  <td {_TD}>Ort +6.64%</td>
  <td {_TD_M}>N=3572</td></tr>
<tr {_ROW_B}>
  <td {_TD}><b>10 Hafta</b></td>
  <td {_TD}>WR %50.6</td>
  <td {_TD}>Ort +7.27%</td>
  <td {_TD_M}>N=3240</td></tr>
</table>
<table style="width:100%;font-size:0.74rem;margin-top:4px;margin-bottom:8px;border-collapse:collapse">
<tr {_ROW_H}>
  <td colspan="4" style="padding:4px 6px;color:var(--nox-green);font-weight:700">Gecis Tipi (3H)</td></tr>
<tr {_ROW_B}>
  <td {_TD}><b>GRI&#8594;FULL &#9733;</b></td>
  <td {_TD_G}>WR %58.7</td>
  <td {_TD}>Ort +4.34%</td>
  <td {_TD_M}>N=254</td></tr>
<tr {_ROW_B}>
  <td {_TD}><b>CHOPPY&#8594;FULL</b></td>
  <td {_TD}>WR %51.7</td>
  <td {_TD}>Ort +3.50%</td>
  <td {_TD_M}>N=410</td></tr>
<tr {_ROW_B}>
  <td {_TD}><b>TREND&#8594;FULL</b></td>
  <td {_TD}>WR %49.9</td>
  <td {_TD}>Ort +2.44%</td>
  <td {_TD_M}>N=1017</td></tr>
<tr {_ROW_B}>
  <td {_TD}><b>GRI&#8594;TREND</b></td>
  <td {_TD}>WR %49.3</td>
  <td {_TD}>Ort +2.23%</td>
  <td {_TD_M}>N=542</td></tr>
<tr {_ROW_B}>
  <td {_TD}><b>CHOPPY&#8594;TREND</b></td>
  <td {_TD}>WR %48.9</td>
  <td {_TD}>Ort +1.57%</td>
  <td {_TD_M}>N=603</td></tr>
<tr {_ROW_B}>
  <td {_TD}><b>CHOPPY&#8594;GRI</b></td>
  <td {_TD}>WR %49.1</td>
  <td {_TD}>Ort +1.25%</td>
  <td {_TD_M}>N=857</td></tr>
</table>
<table style="width:100%;font-size:0.74rem;margin-top:4px;margin-bottom:8px;border-collapse:collapse">
<tr {_ROW_H}>
  <td colspan="4" style="padding:4px 6px;color:var(--nox-green);font-weight:700">Giris Skoru (3H)</td></tr>
<tr {_ROW_B}>
  <td {_TD}><b>Score 3</b></td>
  <td {_TD_R}>WR %48.5</td>
  <td {_TD}>Ort +1.37%</td>
  <td {_TD_M}>N=1211</td></tr>
<tr {_ROW_B}>
  <td {_TD}><b>Score 2</b></td>
  <td {_TD}>WR %50.8</td>
  <td {_TD}>Ort +2.36%</td>
  <td {_TD_M}>N=966</td></tr>
<tr {_ROW_B}>
  <td {_TD}><b>Score 1</b></td>
  <td {_TD}>WR %51.9</td>
  <td {_TD}>Ort +2.77%</td>
  <td {_TD_M}>N=1211</td></tr>
<tr {_ROW_B}>
  <td {_TD}><b>Score 0</b></td>
  <td {_TD}>WR %48.8</td>
  <td {_TD}>Ort +3.26%</td>
  <td {_TD_M}>N=293</td></tr>
</table>
<table style="width:100%;font-size:0.74rem;margin-top:4px;margin-bottom:8px;border-collapse:collapse">
<tr {_ROW_H}>
  <td colspan="5" style="padding:4px 6px;color:var(--nox-green);font-weight:700">Gecis x Skor Capraz (3H, N&#8805;10)</td></tr>
<tr {_ROW_B}>
  <td {_TD}><b>GRI&#8594;FULL S0</b></td>
  <td {_TD_G}>WR %61.5</td>
  <td {_TD}>Ort +4.71%</td>
  <td {_TD_M}>N=39</td>
  <td {_TD_M}>&#9733;</td></tr>
<tr {_ROW_B}>
  <td {_TD}><b>GRI&#8594;FULL S1</b></td>
  <td {_TD_G}>WR %58.9</td>
  <td {_TD}>Ort +4.45%</td>
  <td {_TD_M}>N=151</td>
  <td {_TD_M}>&#9733;</td></tr>
<tr {_ROW_B}>
  <td {_TD}><b>GRI&#8594;FULL S2</b></td>
  <td {_TD_G}>WR %56.2</td>
  <td {_TD}>Ort +3.85%</td>
  <td {_TD_M}>N=64</td>
  <td {_TD_M}></td></tr>
<tr {_ROW_B}>
  <td {_TD}><b>CHOPPY&#8594;FULL S2</b></td>
  <td {_TD_G}>WR %56.1</td>
  <td {_TD}>Ort +3.71%</td>
  <td {_TD_M}>N=41</td>
  <td {_TD_M}></td></tr>
<tr {_ROW_B}>
  <td {_TD}><b>CHOPPY&#8594;FULL S1</b></td>
  <td {_TD}>WR %54.2</td>
  <td {_TD}>Ort +4.24%</td>
  <td {_TD_M}>N=225</td>
  <td {_TD_M}></td></tr>
<tr {_ROW_B}>
  <td {_TD}><b>GRI&#8594;TREND S2</b></td>
  <td {_TD}>WR %52.2</td>
  <td {_TD}>Ort +3.15%</td>
  <td {_TD_M}>N=251</td>
  <td {_TD_M}></td></tr>
</table>
<b style="color:var(--nox-yellow)">Onemli Notlar</b><br>
&#8226; Haftalik genel WR ~%50 — <b>tek basina edge dusuk</b>, gecis tipi kritik<br>
&#8226; <b>GRI&#8594;FULL en iyi gecis</b>: WR %58.7, Ort +4.34% (N=254) — bu gecisi oncelikle takip et<br>
&#8226; Giris skoru haftalikta <b>ayirici degil</b> (Score3 %48.5 &lt; Score1 %51.9) — skor filtresi KULLANMA<br>
&#8226; Ortalama getiri pozitif (+2-7%) — kazananlar kaybedenlerden buyuk, risk/odul olumlu<br>
&#8226; Gunluk tarama ile birlestir: haftalik AL + gunluk pullback = en iyi kombinasyon<br>
</div>
</details>"""

    else:  # monthly
        return f"""
<details class="trade-guide" style="margin-bottom:14px;padding:10px 14px;border-radius:8px;
  background:rgba(74,222,128,0.06);border:1px solid rgba(74,222,128,0.15);
  font-size:0.78rem;color:var(--text-secondary);line-height:1.6">
<summary style="cursor:pointer;font-weight:700;color:var(--nox-green);font-size:0.82rem">
  Aylik Trade Rehberi</summary>
<div style="margin-top:8px">
<b style="color:var(--nox-green)">Nasil Trade Edilir? (Aylik)</b><br>
&#8226; Aylik rejim gecisi = <b>uzun vadeli yapisal trend degisimi</b> (aylar-yillar)<br>
&#8226; Giris: Aylik mum kapanisinda rejim yukselince, haftalik grafige gecip <b>ilk pullback</b>'ten gir<br>
&#8226; Pozisyon buyuklugu: Gunluge gore <b>daha buyuk ama daha az hisse</b>, portfoy agirlikli pozisyon<br>
&#8226; Tutma suresi: <b>2-6 ay</b> tipik | Piramitleme ile pozisyon buyut<br>
&#8226; Stop: Aylik swing low - 0.5*ATR (genis stop, %10-15 arasi normal)<br>
&#8226; Cikis: Aylik kapanis EMA21 altinda veya Exit Stage >=2<br>
<br>
<b style="color:var(--nox-cyan)">Backtest Sonuclari (60 ay, 490 hisse, N=3052 AL)</b>
<table style="width:100%;font-size:0.74rem;margin-top:4px;margin-bottom:8px;border-collapse:collapse">
<tr {_ROW_H}>
  <td colspan="4" style="padding:4px 6px;color:var(--nox-green);font-weight:700">Pencere Bazinda</td></tr>
<tr {_ROW_B}>
  <td {_TD}><b>2 Ay</b></td>
  <td {_TD_G}>WR %60.5</td>
  <td {_TD}>Ort +14.63%</td>
  <td {_TD_M}>N=3069</td></tr>
<tr {_ROW_B}>
  <td {_TD}><b>3 Ay</b></td>
  <td {_TD_G}>WR %60.7</td>
  <td {_TD}>Ort +19.85%</td>
  <td {_TD_M}>N=3006</td></tr>
<tr {_ROW_B}>
  <td {_TD}><b>6 Ay</b></td>
  <td {_TD_G}>WR %65.4</td>
  <td {_TD}>Ort +38.35%</td>
  <td {_TD_M}>N=2861</td></tr>
</table>
<table style="width:100%;font-size:0.74rem;margin-top:4px;margin-bottom:8px;border-collapse:collapse">
<tr {_ROW_H}>
  <td colspan="4" style="padding:4px 6px;color:var(--nox-green);font-weight:700">Gecis Tipi (2A)</td></tr>
<tr {_ROW_B}>
  <td {_TD}><b>CHOPPY&#8594;FULL &#9733;</b></td>
  <td {_TD_G}>WR %63.5</td>
  <td {_TD}>Ort +15.93%</td>
  <td {_TD_M}>N=822</td></tr>
<tr {_ROW_B}>
  <td {_TD}><b>CHOPPY&#8594;TREND &#9733;</b></td>
  <td {_TD_G}>WR %62.0</td>
  <td {_TD}>Ort +17.02%</td>
  <td {_TD_M}>N=376</td></tr>
<tr {_ROW_B}>
  <td {_TD}><b>GRI&#8594;TREND</b></td>
  <td {_TD_G}>WR %60.5</td>
  <td {_TD}>Ort +13.33%</td>
  <td {_TD_M}>N=299</td></tr>
<tr {_ROW_B}>
  <td {_TD}><b>CHOPPY&#8594;GRI</b></td>
  <td {_TD_G}>WR %59.0</td>
  <td {_TD}>Ort +12.92%</td>
  <td {_TD_M}>N=502</td></tr>
<tr {_ROW_B}>
  <td {_TD}><b>TREND&#8594;FULL</b></td>
  <td {_TD_G}>WR %58.8</td>
  <td {_TD}>Ort +14.53%</td>
  <td {_TD_M}>N=877</td></tr>
<tr {_ROW_B}>
  <td {_TD}><b>GRI&#8594;FULL</b></td>
  <td {_TD}>WR %56.8</td>
  <td {_TD}>Ort +11.63%</td>
  <td {_TD_M}>N=176</td></tr>
</table>
<table style="width:100%;font-size:0.74rem;margin-top:4px;margin-bottom:8px;border-collapse:collapse">
<tr {_ROW_H}>
  <td colspan="4" style="padding:4px 6px;color:var(--nox-green);font-weight:700">Giris Skoru (2A)</td></tr>
<tr {_ROW_B}>
  <td {_TD}><b>Score 3</b></td>
  <td {_TD_G}>WR %60.3</td>
  <td {_TD}>Ort +14.31%</td>
  <td {_TD_M}>N=716</td></tr>
<tr {_ROW_B}>
  <td {_TD}><b>Score 2</b></td>
  <td {_TD_G}>WR %62.9</td>
  <td {_TD}>Ort +15.22%</td>
  <td {_TD_M}>N=820</td></tr>
<tr {_ROW_B}>
  <td {_TD}><b>Score 1</b></td>
  <td {_TD_G}>WR %60.0</td>
  <td {_TD}>Ort +14.57%</td>
  <td {_TD_M}>N=1254</td></tr>
<tr {_ROW_B}>
  <td {_TD}><b>Score 0</b></td>
  <td {_TD}>WR %56.5</td>
  <td {_TD}>Ort +14.30%</td>
  <td {_TD_M}>N=262</td></tr>
</table>
<table style="width:100%;font-size:0.74rem;margin-top:4px;margin-bottom:8px;border-collapse:collapse">
<tr {_ROW_H}>
  <td colspan="5" style="padding:4px 6px;color:var(--nox-green);font-weight:700">Gecis x Skor Capraz (2A, N&#8805;10)</td></tr>
<tr {_ROW_B}>
  <td {_TD}><b>GRI&#8594;FULL S2</b></td>
  <td {_TD_G}>WR %69.0</td>
  <td {_TD}>Ort +15.07%</td>
  <td {_TD_M}>N=29</td>
  <td {_TD_M}>&#9733;</td></tr>
<tr {_ROW_B}>
  <td {_TD}><b>CHOPPY&#8594;FULL S2</b></td>
  <td {_TD_G}>WR %66.1</td>
  <td {_TD}>Ort +13.46%</td>
  <td {_TD_M}>N=186</td>
  <td {_TD_M}>&#9733;</td></tr>
<tr {_ROW_B}>
  <td {_TD}><b>CHOPPY&#8594;TREND S3</b></td>
  <td {_TD_G}>WR %65.7</td>
  <td {_TD}>Ort +22.45%</td>
  <td {_TD_M}>N=181</td>
  <td {_TD_M}>&#9733;</td></tr>
<tr {_ROW_B}>
  <td {_TD}><b>CHOPPY&#8594;FULL S1</b></td>
  <td {_TD_G}>WR %64.8</td>
  <td {_TD}>Ort +17.74%</td>
  <td {_TD_M}>N=463</td>
  <td {_TD_M}></td></tr>
<tr {_ROW_B}>
  <td {_TD}><b>GRI&#8594;TREND S2</b></td>
  <td {_TD_G}>WR %63.3</td>
  <td {_TD}>Ort +15.56%</td>
  <td {_TD_M}>N=210</td>
  <td {_TD_M}></td></tr>
<tr {_ROW_B}>
  <td {_TD}><b>TREND&#8594;FULL S2</b></td>
  <td {_TD_G}>WR %63.0</td>
  <td {_TD}>Ort +18.61%</td>
  <td {_TD_M}>N=165</td>
  <td {_TD_M}></td></tr>
</table>
<table style="width:100%;font-size:0.74rem;margin-top:4px;margin-bottom:8px;border-collapse:collapse">
<tr {_ROW_H}>
  <td colspan="4" style="padding:4px 6px;color:var(--nox-green);font-weight:700">Exit Stage (2A)</td></tr>
<tr {_ROW_B}>
  <td {_TD}><b>Exit 0</b></td>
  <td {_TD_G}>WR %60.9</td>
  <td {_TD}>Ort +15.23%</td>
  <td {_TD_M}>N=2798</td></tr>
<tr {_ROW_B}>
  <td {_TD}><b>Exit 1</b></td>
  <td {_TD}>WR %56.7</td>
  <td {_TD}>Ort +8.40%</td>
  <td {_TD_M}>N=254</td></tr>
</table>
<b style="color:var(--nox-green)">Onemli Notlar</b><br>
&#8226; Aylik en guclu timeframe — <b>WR %60-65, Ort +14-38%</b><br>
&#8226; Tum gecis tipleri karli (WR %57-64 arasi)<br>
&#8226; <b>CHOPPY&#8594;FULL + CHOPPY&#8594;TREND</b> en yuksek getirili gecisler (WR %62-64)<br>
&#8226; En iyi capraz: <b>CHOPPY&#8594;TREND S3</b> (WR %65.7, Ort +22.45%) ve <b>CHOPPY&#8594;FULL S2</b> (WR %66.1)<br>
&#8226; 6 aylik pencerede WR %65.4, Ort +38.35% — uzun tutma oduluyor<br>
&#8226; Giris skoru aylikta daha az ayirici (tum skorlar WR %57-63) — <b>gecis tipi oncelikli</b><br>
&#8226; Exit==0 filtrele: WR %60.9 vs Exit1 %56.7<br>
&#8226; Pozisyon yonetimi: Baslangicta %50 gir, haftalik pullback'te %50 ekle<br>
</div>
</details>"""


def _generate_html(results, n_scanned, date_str, regime_dist, timeframe='daily', weekly_al_pb=None):
    now = datetime.now().strftime('%d.%m.%Y %H:%M')
    tf_label = _TF_LABELS.get(timeframe, 'Gunluk')

    rows_data = []
    for s in results:
        # Stop-fiyat uzakligi %
        stop_dist_pct = round((s.close - s.stop) / s.close * 100, 1) if s.stop > 0 and s.close > 0 else 0
        # Hacim-donus tier
        _atr_pct = round((s.atr / s.close * 100), 2) if s.close > 0 else 0
        _vt, _vt_icon = classify_volume_quality(
            _atr_pct, s.cmf, s.rvol, s.participation_score, s.oe_score)
        rows_data.append({
            'ticker': s.ticker,
            'regime': s.regime,
            'regime_name': s.regime_name,
            'transition': s.transition,
            'transition_date': s.transition_date.strftime('%Y-%m-%d') if hasattr(s.transition_date, 'strftime') else '',
            'transition_date_iso': s.transition_date.strftime('%Y-%m-%d') if hasattr(s.transition_date, 'strftime') else '',
            'days_since': s.days_since,
            'entry_window': _entry_window_tf(s.days_since, timeframe),
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
            'atr_pct': _atr_pct,
            'vol_tier': _vt,
            'vol_tier_icon': _vt_icon,
        })
        # Haftalik AL + gunluk pullback bilgisi (sadece gunluk tarama icin)
        if weekly_al_pb and s.ticker in weekly_al_pb:
            info = weekly_al_pb[s.ticker]
            rows_data[-1]['weekly_al'] = True
            rows_data[-1]['weekly_pb'] = info['pb']
            rows_data[-1]['ema_dist'] = info['ema_dist']
            rows_data[-1]['daily_rsi'] = info['rsi']
            rows_data[-1]['w_transition'] = info['weekly_transition']
            rows_data[-1]['w_regime'] = info['weekly_regime']
        else:
            rows_data[-1]['weekly_al'] = False
            rows_data[-1]['weekly_pb'] = False

    data = {
        'rows': _sanitize(rows_data),
        'n_scanned': n_scanned,
        'date': date_str,
        'regime_dist': regime_dist,
    }
    data_json = json.dumps(data, ensure_ascii=False)
    trade_guide_extra = _build_trade_guide_extra(timeframe)

    html = f"""<!DOCTYPE html>
<html lang="tr"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NOX — Regime Transition ({tf_label}) · {now}</title>
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

/* WEEKLY AL + PULLBACK BADGES */
.wal-badge {{
  display: inline-block; padding: 1px 6px; border-radius: var(--radius-sm);
  font-size: 0.62rem; font-weight: 700; font-family: var(--font-mono);
  letter-spacing: 0.02em; margin-left: 4px; cursor: help;
}}
.wal-pb {{ background: rgba(74,222,128,0.2); color: var(--nox-green); }}
.wal-al {{ background: rgba(34,211,238,0.15); color: var(--nox-cyan); }}

/* VOL TIER BADGE (Hacim-donus kalitesi) */
.vt-badge {{
  display: inline-block; padding: 1px 6px; border-radius: var(--radius-sm);
  font-size: 0.62rem; font-weight: 700; font-family: var(--font-mono);
  letter-spacing: 0.02em; margin-left: 4px; cursor: help;
}}
.vt-ALTIN {{ background: rgba(250,204,21,0.2); color: #facc15; }}
.vt-GUMUS {{ background: rgba(192,192,192,0.2); color: #c0c0c0; }}
.vt-BRONZ {{ background: rgba(205,127,50,0.2); color: #cd7f32; }}
.vt-ELE {{ background: rgba(248,113,113,0.15); color: #f87171; }}
</style>
</head><body>
<div class="nox-container">
<div class="nox-header">
  <div class="nox-logo">NOX<span class="proj">project</span><span class="mode">regime transition screener ({tf_label.lower()})</span></div>
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
  <div><label>Hacim</label>
  <select id="fVT" onchange="render()"><option value="">Tumu</option>
  <option value="ALTIN">🥇 ALTIN</option><option value="GUMUS">🥈 GUMUS</option>
  <option value="BRONZ">🥉 BRONZ</option><option value="!ELE">ELE Haric</option></select></div>
  <div><label>Exit≤</label><input type="number" id="fExit" value="" step="1" min="0" max="3" placeholder="max" oninput="render()"></div>
  <div><label>ADX≥</label><input type="number" id="fADX" value="0" step="5" min="0" oninput="render()"></div>
  {'<div><label style="display:flex;align-items:center;gap:4px"><input type="checkbox" id="fWPB" onchange="render()"> H+PB</label></div><div><label style="display:flex;align-items:center;gap:4px"><input type="checkbox" id="fWAL" onchange="render()"> H-AL</label></div>' if timeframe == 'daily' else ''}
  <div><button class="nox-btn" onclick="resetF()">Sifirla</button></div>
</div>

<details class="trade-guide" style="margin-bottom:14px;padding:10px 14px;border-radius:8px;
  background:rgba(34,211,238,0.06);border:1px solid rgba(34,211,238,0.15);
  font-size:0.78rem;color:var(--text-secondary);line-height:1.6">
<summary style="cursor:pointer;font-weight:700;color:var(--nox-cyan);font-size:0.82rem">
  Trade Rehberi</summary>
<div style="margin-top:8px">
<b style="color:var(--nox-green)">Sticky AL Mantigi</b><br>
Rejim yukselince trade baslar, fiyat EMA21 altina kapanana kadar aktif kalir.<br>
Bu liste <b>simdi girilebilir</b> hisseleri gosterir — trade kapanmis hisseler listede yoktur.<br>
<br>
<b style="color:var(--nox-green)">AL Checklist</b><br>
1. <b>Giris skoru 3-4/4</b> filtrele (Score4 WR %62.5, Score3 WR %52.0)<br>
2. <b>OE skoru 0-1</b> olanları sec (overextended degil)<br>
3. <b>Pencere: TAZE/BEKLE</b> → erken giris firsati<br>
4. <b>Exit Stage 0</b> → cikis riski yok<br>
5. Hisse adina tikla → TradingView'da grafigi kontrol et<br>
6. Stop = tablodaki Stop fiyati (swing low - 0.5*ATR)<br>
<br>
<b style="color:var(--nox-cyan)">Backtest Sonuclari (250 bar, 537 hisse, N=12051 AL)</b><br>
<table style="width:100%;font-size:0.74rem;margin-top:4px;margin-bottom:8px;border-collapse:collapse">
<tr style="border-bottom:1px solid rgba(255,255,255,0.12)">
  <td colspan="4" style="padding:4px 6px;color:var(--nox-green);font-weight:700">Genel Giris Skoru (5G)</td></tr>
<tr style="border-bottom:1px solid rgba(255,255,255,0.06)">
  <td style="padding:2px 6px"><b>Score 4</b></td>
  <td style="padding:2px 6px;color:var(--nox-green)">WR %62.5</td>
  <td style="padding:2px 6px">Ort +1.75%</td>
  <td style="padding:2px 6px;color:var(--text-muted)">N=595</td></tr>
<tr style="border-bottom:1px solid rgba(255,255,255,0.06)">
  <td style="padding:2px 6px"><b>Score 3</b></td>
  <td style="padding:2px 6px">WR %52.0</td>
  <td style="padding:2px 6px">Ort +1.27%</td>
  <td style="padding:2px 6px;color:var(--text-muted)">N=3189</td></tr>
<tr style="border-bottom:1px solid rgba(255,255,255,0.06)">
  <td style="padding:2px 6px"><b>Score 2</b></td>
  <td style="padding:2px 6px">WR %50.8</td>
  <td style="padding:2px 6px">Ort +1.04%</td>
  <td style="padding:2px 6px;color:var(--text-muted)">N=3509</td></tr>
<tr style="border-bottom:1px solid rgba(255,255,255,0.06)">
  <td style="padding:2px 6px"><b>Score 1</b></td>
  <td style="padding:2px 6px">WR %50.4</td>
  <td style="padding:2px 6px">Ort +0.87%</td>
  <td style="padding:2px 6px;color:var(--text-muted)">N=3471</td></tr>
<tr style="border-bottom:1px solid rgba(255,255,255,0.08)">
  <td style="padding:2px 6px"><b>Score 0</b></td>
  <td style="padding:2px 6px;color:var(--nox-red)">WR %46.5</td>
  <td style="padding:2px 6px">Ort +0.39%</td>
  <td style="padding:2px 6px;color:var(--text-muted)">N=1110</td></tr>
</table>
<b style="color:var(--nox-cyan)">Gecis Tipi x Giris Skoru (5G WR)</b>
<table style="width:100%;font-size:0.74rem;margin-top:4px;margin-bottom:8px;border-collapse:collapse">
<tr style="border-bottom:1px solid rgba(255,255,255,0.12)">
  <td style="padding:4px 6px;color:var(--nox-yellow);font-weight:700">CHOPPY&#8594;GRI</td>
  <td colspan="3" style="padding:4px 6px;color:var(--text-muted)">Genel WR %52.1, Ort +1.20% (N=2587)</td></tr>
<tr style="border-bottom:1px solid rgba(255,255,255,0.06)">
  <td style="padding:2px 12px">Score 4</td>
  <td style="padding:2px 6px;color:var(--nox-green)">WR %60.7</td>
  <td style="padding:2px 6px">Ort +1.52%</td>
  <td style="padding:2px 6px;color:var(--text-muted)">N=308</td></tr>
<tr style="border-bottom:1px solid rgba(255,255,255,0.06)">
  <td style="padding:2px 12px">Score 3</td>
  <td style="padding:2px 6px">WR %51.1</td>
  <td style="padding:2px 6px">Ort +1.08%</td>
  <td style="padding:2px 6px;color:var(--text-muted)">N=1662</td></tr>
<tr style="border-bottom:1px solid rgba(255,255,255,0.08)">
  <td style="padding:2px 12px">Score 2</td>
  <td style="padding:2px 6px">WR %50.2</td>
  <td style="padding:2px 6px">Ort +1.26%</td>
  <td style="padding:2px 6px;color:var(--text-muted)">N=600</td></tr>
<tr style="border-bottom:1px solid rgba(255,255,255,0.12)">
  <td style="padding:4px 6px;color:var(--nox-cyan);font-weight:700">CHOPPY&#8594;TREND</td>
  <td colspan="3" style="padding:4px 6px;color:var(--text-muted)">Genel WR %50.4, Ort +1.18% (N=1455)</td></tr>
<tr style="border-bottom:1px solid rgba(255,255,255,0.06)">
  <td style="padding:2px 12px">Score 4</td>
  <td style="padding:2px 6px;color:var(--nox-green)">WR %71.0</td>
  <td style="padding:2px 6px">Ort +2.65%</td>
  <td style="padding:2px 6px;color:var(--text-muted)">N=93</td></tr>
<tr style="border-bottom:1px solid rgba(255,255,255,0.06)">
  <td style="padding:2px 12px">Score 3</td>
  <td style="padding:2px 6px">WR %53.6</td>
  <td style="padding:2px 6px">Ort +1.81%</td>
  <td style="padding:2px 6px;color:var(--text-muted)">N=603</td></tr>
<tr style="border-bottom:1px solid rgba(255,255,255,0.06)">
  <td style="padding:2px 12px">Score 2</td>
  <td style="padding:2px 6px;color:var(--nox-red)">WR %46.0</td>
  <td style="padding:2px 6px">Ort +0.53%</td>
  <td style="padding:2px 6px;color:var(--text-muted)">N=652</td></tr>
<tr style="border-bottom:1px solid rgba(255,255,255,0.08)">
  <td style="padding:2px 12px">Score 1</td>
  <td style="padding:2px 6px;color:var(--nox-red)">WR %41.1</td>
  <td style="padding:2px 6px">Ort +0.28%</td>
  <td style="padding:2px 6px;color:var(--text-muted)">N=107</td></tr>
<tr style="border-bottom:1px solid rgba(255,255,255,0.12)">
  <td style="padding:4px 6px;color:var(--nox-green);font-weight:700">CHOPPY&#8594;FULL</td>
  <td colspan="3" style="padding:4px 6px;color:var(--text-muted)">Genel WR %49.8, Ort +0.61% (N=997)</td></tr>
<tr style="border-bottom:1px solid rgba(255,255,255,0.06)">
  <td style="padding:2px 12px">Score 2</td>
  <td style="padding:2px 6px">WR %55.7</td>
  <td style="padding:2px 6px">Ort +0.82%</td>
  <td style="padding:2px 6px;color:var(--text-muted)">N=203</td></tr>
<tr style="border-bottom:1px solid rgba(255,255,255,0.06)">
  <td style="padding:2px 12px">Score 1</td>
  <td style="padding:2px 6px">WR %50.7</td>
  <td style="padding:2px 6px">Ort +0.76%</td>
  <td style="padding:2px 6px;color:var(--text-muted)">N=458</td></tr>
<tr style="border-bottom:1px solid rgba(255,255,255,0.08)">
  <td style="padding:2px 12px">Score 0</td>
  <td style="padding:2px 6px;color:var(--nox-red)">WR %44.1</td>
  <td style="padding:2px 6px">Ort +0.06%</td>
  <td style="padding:2px 6px;color:var(--text-muted)">N=329</td></tr>
<tr style="border-bottom:1px solid rgba(255,255,255,0.12)">
  <td style="padding:4px 6px;color:var(--nox-cyan);font-weight:700">GRI&#8594;TREND</td>
  <td colspan="3" style="padding:4px 6px;color:var(--text-muted)">Genel WR %51.3, Ort +1.06% (N=1903)</td></tr>
<tr style="border-bottom:1px solid rgba(255,255,255,0.06)">
  <td style="padding:2px 12px">Score 4</td>
  <td style="padding:2px 6px;color:var(--nox-green)">WR %61.3</td>
  <td style="padding:2px 6px">Ort +1.69%</td>
  <td style="padding:2px 6px;color:var(--text-muted)">N=194</td></tr>
<tr style="border-bottom:1px solid rgba(255,255,255,0.06)">
  <td style="padding:2px 12px">Score 3</td>
  <td style="padding:2px 6px">WR %50.2</td>
  <td style="padding:2px 6px">Ort +1.03%</td>
  <td style="padding:2px 6px;color:var(--text-muted)">N=836</td></tr>
<tr style="border-bottom:1px solid rgba(255,255,255,0.08)">
  <td style="padding:2px 12px">Score 2</td>
  <td style="padding:2px 6px">WR %50.6</td>
  <td style="padding:2px 6px">Ort +0.99%</td>
  <td style="padding:2px 6px;color:var(--text-muted)">N=812</td></tr>
<tr style="border-bottom:1px solid rgba(255,255,255,0.12)">
  <td style="padding:4px 6px;color:var(--nox-green);font-weight:700">GRI&#8594;FULL</td>
  <td colspan="3" style="padding:4px 6px;color:var(--text-muted)">Genel WR %48.1, Ort +0.79% (N=919)</td></tr>
<tr style="border-bottom:1px solid rgba(255,255,255,0.06)">
  <td style="padding:2px 12px">Score 3</td>
  <td style="padding:2px 6px;color:var(--nox-green)">WR %68.2</td>
  <td style="padding:2px 6px">Ort +2.22%</td>
  <td style="padding:2px 6px;color:var(--text-muted)">N=22</td></tr>
<tr style="border-bottom:1px solid rgba(255,255,255,0.06)">
  <td style="padding:2px 12px">Score 2</td>
  <td style="padding:2px 6px">WR %53.3</td>
  <td style="padding:2px 6px">Ort +1.20%</td>
  <td style="padding:2px 6px;color:var(--text-muted)">N=244</td></tr>
<tr style="border-bottom:1px solid rgba(255,255,255,0.06)">
  <td style="padding:2px 12px">Score 1</td>
  <td style="padding:2px 6px;color:var(--nox-red)">WR %45.8</td>
  <td style="padding:2px 6px">Ort +0.33%</td>
  <td style="padding:2px 6px;color:var(--text-muted)">N=456</td></tr>
<tr style="border-bottom:1px solid rgba(255,255,255,0.08)">
  <td style="padding:2px 12px">Score 0</td>
  <td style="padding:2px 6px;color:var(--nox-red)">WR %44.7</td>
  <td style="padding:2px 6px">Ort +1.20%</td>
  <td style="padding:2px 6px;color:var(--text-muted)">N=197</td></tr>
<tr style="border-bottom:1px solid rgba(255,255,255,0.12)">
  <td style="padding:4px 6px;color:var(--nox-green);font-weight:700">TREND&#8594;FULL</td>
  <td colspan="3" style="padding:4px 6px;color:var(--text-muted)">Genel WR %51.9, Ort +1.00% (N=4013)</td></tr>
<tr style="border-bottom:1px solid rgba(255,255,255,0.06)">
  <td style="padding:2px 12px">Score 3</td>
  <td style="padding:2px 6px;color:var(--nox-green)">WR %71.2</td>
  <td style="padding:2px 6px">Ort +2.97%</td>
  <td style="padding:2px 6px;color:var(--text-muted)">N=59</td></tr>
<tr style="border-bottom:1px solid rgba(255,255,255,0.06)">
  <td style="padding:2px 12px">Score 2</td>
  <td style="padding:2px 6px">WR %52.9</td>
  <td style="padding:2px 6px">Ort +1.28%</td>
  <td style="padding:2px 6px;color:var(--text-muted)">N=998</td></tr>
<tr style="border-bottom:1px solid rgba(255,255,255,0.06)">
  <td style="padding:2px 12px">Score 1</td>
  <td style="padding:2px 6px">WR %51.8</td>
  <td style="padding:2px 6px">Ort +1.00%</td>
  <td style="padding:2px 6px;color:var(--text-muted)">N=2372</td></tr>
<tr style="border-bottom:1px solid rgba(255,255,255,0.08)">
  <td style="padding:2px 12px">Score 0</td>
  <td style="padding:2px 6px;color:var(--nox-red)">WR %48.5</td>
  <td style="padding:2px 6px">Ort +0.31%</td>
  <td style="padding:2px 6px;color:var(--text-muted)">N=584</td></tr>
</table>
<br>
<b style="color:var(--nox-red)">OE (Overextended) Uyarisi</b><br>
&#8226; OE 0-1 = Guvenli | OE 2 = Dikkat | OE 3-4 = Riskli (geri cekilme beklenir)<br>
&#8226; Kriterler: RSI>80, BB ustu, 5G>%8 yukselis, EMA21'den >%5 uzak<br>
<br>
<b style="color:var(--nox-yellow)">Kolon Aciklamalari</b>
<table style="width:100%;font-size:0.74rem;margin-top:6px;border-collapse:collapse">
<tr style="border-bottom:1px solid rgba(255,255,255,0.08)">
  <td style="padding:3px 6px;color:var(--nox-cyan);font-weight:700;white-space:nowrap">Hisse</td>
  <td style="padding:3px 6px">Ticker (tikla → TradingView)</td></tr>
<tr style="border-bottom:1px solid rgba(255,255,255,0.08)">
  <td style="padding:3px 6px;color:var(--nox-cyan);font-weight:700">Regime</td>
  <td style="padding:3px 6px">Bugunki rejim seviyesi. CHOPPY(0) → GRI(1) → TREND(2) → FULL(3). Trade aktifken rejim dusebilir ama EMA21 ustundeyse trade devam eder</td></tr>
<tr style="border-bottom:1px solid rgba(255,255,255,0.08)">
  <td style="padding:3px 6px;color:var(--nox-cyan);font-weight:700">Gecis</td>
  <td style="padding:3px 6px">Trade basladigi andaki rejim degisimi (orn. CHOPPY→TREND)</td></tr>
<tr style="border-bottom:1px solid rgba(255,255,255,0.08)">
  <td style="padding:3px 6px;color:var(--nox-cyan);font-weight:700">Tarih</td>
  <td style="padding:3px 6px">Trade baslangic tarihi</td></tr>
<tr style="border-bottom:1px solid rgba(255,255,255,0.08)">
  <td style="padding:3px 6px;color:var(--nox-cyan);font-weight:700">Gun</td>
  <td style="padding:3px 6px">Trade kac gundur aktif</td></tr>
<tr style="border-bottom:1px solid rgba(255,255,255,0.08)">
  <td style="padding:3px 6px;color:var(--nox-cyan);font-weight:700">Pencere</td>
  <td style="padding:3px 6px">TAZE (0-1g) → en iyi giris | BEKLE (3-7g) → pullback bekle | 2.DALGA (10-20g) → ikinci sans | GEC (30+g) → gec kalmis</td></tr>
<tr style="border-bottom:1px solid rgba(255,255,255,0.08)">
  <td style="padding:3px 6px;color:var(--nox-cyan);font-weight:700">Giris</td>
  <td style="padding:3px 6px">Entry Score (0-4). GIR(4) / FIRSAT(3) / RISKLI(2) / GEC(0-1). Kriterler: dusuk vol, erken giris, buyume odasi, pump filtresi</td></tr>
<tr style="border-bottom:1px solid rgba(255,255,255,0.08)">
  <td style="padding:3px 6px;color:var(--nox-cyan);font-weight:700">OE</td>
  <td style="padding:3px 6px">Overextended skoru (0-4). Yuksek = asiri uzamis, geri cekilme riski. Kriterler: RSI>80, BB ustu, 5G momentum>%8, EMA21 uzakligi>%5</td></tr>
<tr style="border-bottom:1px solid rgba(255,255,255,0.08)">
  <td style="padding:3px 6px;color:var(--nox-cyan);font-weight:700">T / P / E</td>
  <td style="padding:3px 6px">Trend(0-3) / Participation(0-3) / Expansion(0-3) skorlari. Rejimi belirleyen 3 bilesen</td></tr>
<tr style="border-bottom:1px solid rgba(255,255,255,0.08)">
  <td style="padding:3px 6px;color:var(--nox-cyan);font-weight:700">Exit</td>
  <td style="padding:3px 6px">Cikis asamasi (0-3). 0=temiz, 1=structure break, 2+=momentum decay. Dusuk = daha guvenli</td></tr>
<tr style="border-bottom:1px solid rgba(255,255,255,0.08)">
  <td style="padding:3px 6px;color:var(--nox-cyan);font-weight:700">Stop</td>
  <td style="padding:3px 6px">Efektif stop fiyati = max(swing_low - 0.5*ATR, close - 2*ATR). Turuncu = fiyata cok yakin (%2 alti)</td></tr>
<tr style="border-bottom:1px solid rgba(255,255,255,0.08)">
  <td style="padding:3px 6px;color:var(--nox-cyan);font-weight:700">ADX / Slope</td>
  <td style="padding:3px 6px">ADX(14) trend gucu + 5-bar slope. ADX>25 = trend var. Slope yesil = momentum artiyor</td></tr>
<tr style="border-bottom:1px solid rgba(255,255,255,0.08)">
  <td style="padding:3px 6px;color:var(--nox-cyan);font-weight:700">CMF</td>
  <td style="padding:3px 6px">Chaikin Money Flow (20). Pozitif = birikim, negatif = dagilim</td></tr>
<tr style="border-bottom:1px solid rgba(255,255,255,0.08)">
  <td style="padding:3px 6px;color:var(--nox-cyan);font-weight:700">RVOL</td>
  <td style="padding:3px 6px">Relative Volume (20G ortalamaya gore). >=1.5 yesil = yuksek ilgi | <1.0 gri = dusuk hacim</td></tr>
<tr>
  <td style="padding:3px 6px;color:var(--nox-cyan);font-weight:700">DI±</td>
  <td style="padding:3px 6px">DI+ minus DI-. >5 yesil = boga baskisi | <-5 kirmizi = ayi baskisi</td></tr>
</table>
</div>
</details>

{trade_guide_extra}

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
  const wpbEl=document.getElementById('fWPB');
  if(wpbEl) wpbEl.checked=false;
  const walEl=document.getElementById('fWAL');
  if(walEl) walEl.checked=false;
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
    const fVT=document.getElementById('fVT');
    if(fVT&&fVT.value){{
      if(fVT.value==='!ELE'&&r.vol_tier==='ELE') return false;
      else if(fVT.value!=='!ELE'&&r.vol_tier!==fVT.value) return false;
    }}
    const wpbEl=document.getElementById('fWPB');
    if(wpbEl&&wpbEl.checked&&!r.weekly_pb) return false;
    const walEl=document.getElementById('fWAL');
    if(walEl&&walEl.checked&&!r.weekly_al) return false;
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

function mkVolTierBadge(tier, icon){{
  if(!tier||tier==='NORMAL') return '';
  const tips={{'ALTIN':'ATR≤3% + Part=3 + OE≤1 → 5G WR %75','GUMUS':'ATR≤3% + Part=3 → 5G WR %71','BRONZ':'ATR≤4% + Part≥3 → 5G WR %60','ELE':'ATR>5% / CMF<-0.1 / RVOL>5 → kotu hacim'}};
  return `<span class="vt-badge vt-${{tier}}" title="${{tips[tier]||''}}">${{icon||''}}${{tier}}</span>`;
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
    <td><a class="tv-link" href="https://www.tradingview.com/chart/?symbol=BIST:${{r.ticker}}" target="_blank">${{r.ticker}}</a>${{r.weekly_pb?'<span class="wal-badge wal-pb" title="Haftalik AL + Gunluk Pullback (EMA%'+r.ema_dist+', RSI '+r.daily_rsi+') — '+r.w_transition+'">H+PB</span>':(r.weekly_al?'<span class="wal-badge wal-al" title="Haftalik AL aktif — '+r.w_transition+' ('+r.w_regime+')">H-AL</span>':'')}}${{mkVolTierBadge(r.vol_tier, r.vol_tier_icon)}}</td>
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
  const vtCount={{}};
  D.rows.forEach(r=>{{vtCount[r.vol_tier]=(vtCount[r.vol_tier]||0)+1}});
  const vtSummary=['ALTIN','GUMUS','BRONZ','ELE'].filter(t=>vtCount[t])
    .map(t=>`${{t}}:${{vtCount[t]}}`).join(' · ');
  document.getElementById('st').innerHTML=
    `<b>${{D.rows.length}}</b> AL aktif · OE uyari: <b>${{oeWarn}}</b> · ${{vtSummary?' · '+vtSummary:''}} · ${{D.date}}`;
}}

render();
</script></body></html>"""
    return html


def _save_html(html_content, date_str, output_dir, timeframe='daily'):
    os.makedirs(output_dir, exist_ok=True)
    tf_suffix = f"_{timeframe}" if timeframe != 'daily' else ''
    fname = f"regime_transition{tf_suffix}_{date_str.replace('-', '')}.html"
    path = os.path.join(output_dir, fname)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"\n  HTML: {path}")
    return path


# =============================================================================
# BACKTEST
# =============================================================================

def _run_backtest(stock_dfs, higher_tf_dfs, N, cfg=None, timeframe='daily'):
    """Son N bar'daki AL sinyallerini cikar, forward return hesapla.
    Her ticker 1 kez scan edilir (O(ticker) karmasiklik).

    Returns: (signals_list, summary_dict)
    """
    cfg = cfg or RT_CFG
    min_bars = _TF_MIN_BARS.get(timeframe, 100)
    # Forward return pencereleri timeframe'e gore
    if timeframe == 'weekly':
        ret_windows = [('ret_3w', 3), ('ret_5w', 5), ('ret_10w', 10)]
    elif timeframe == 'monthly':
        ret_windows = [('ret_2m', 2), ('ret_3m', 3), ('ret_6m', 6)]
    else:
        ret_windows = [('ret_5d', 5), ('ret_10d', 10), ('ret_20d', 20)]
    ret_keys = [rw[0] for rw in ret_windows]

    signals = []
    n_scanned = 0
    last_date = None
    lb = cfg['stop_swing_lb']

    for ticker, df in stock_dfs.items():
        if len(df) < min_bars:
            continue
        if _is_halted(df):
            continue

        try:
            htf_df = higher_tf_dfs.get(ticker) if higher_tf_dfs else None
            out = scan_regime_transition(df, weekly_df=htf_df, cfg=cfg)
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
                if d != 'AL':
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
                stop = swing_low - cfg['stop_atr_initial'] * atr_val
                stop_dist_pct = (entry_price - stop) / entry_price * 100 if entry_price > 0 else 0.0

                # Entry Score (veri-odakli kriterler)
                entry_score = 0
                atr_thresh = cfg.get('entry_atr_pct_thresh', 3.0)
                # 1. Dusuk volatilite (ATR% < threshold)
                if atr_pct < atr_thresh:
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

                # Forward returns (dynamic windows)
                ret_vals = {}
                for rkey, window in ret_windows:
                    end_idx = i + window
                    if end_idx < n_bars:
                        r = (float(close_s.iloc[end_idx]) - entry_price) / entry_price * 100
                        ret_vals[rkey] = round(r, 2)
                    else:
                        ret_vals[rkey] = None

                # Stop hit (ilk pencere icerisinde)
                first_window = ret_windows[0][1]
                stop_hit = False
                stop_hit_bar = None
                if stop > 0:
                    for j in range(i + 1, min(i + first_window + 1, n_bars)):
                        if float(low_s.iloc[j]) < stop:
                            stop_hit = True
                            stop_hit_bar = j - i
                            break

                sig_row = {
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
                    'stop_hit': stop_hit,
                    'stop_hit_bar': stop_hit_bar,
                }
                sig_row.update(ret_vals)
                signals.append(sig_row)

        except Exception as e:
            print(f"  ! {ticker}: {e}")
            continue

    date_str = last_date.strftime('%Y-%m-%d') if last_date else datetime.now().strftime('%Y-%m-%d')
    summary = _compute_backtest_summary(signals, ret_keys)
    summary['n_scanned'] = n_scanned
    summary['date_str'] = date_str
    summary['ret_keys'] = ret_keys
    return signals, summary


def _compute_backtest_summary(signals, ret_keys=None):
    """Backtest sinyallerinden istatistik ozet cikar."""
    if not signals:
        return {'total': 0}
    if ret_keys is None:
        ret_keys = ['ret_5d', 'ret_10d', 'ret_20d']

    df = pd.DataFrame(signals)
    summary = {'total': len(df)}

    al = df[df['direction'] == 'AL']
    summary['n_al'] = len(al)

    first_ret_key = ret_keys[0]

    # Genel pencere istatistikleri
    for rkey in ret_keys:
        valid = df[df[rkey].notna()][rkey]
        n = len(valid)
        if n > 0:
            wr = (valid > 0).sum() / n * 100
            summary[f'all_{rkey}'] = {
                'n': n, 'wr': round(wr, 1),
                'avg': round(valid.mean(), 2),
                'med': round(valid.median(), 2),
            }
        else:
            summary[f'all_{rkey}'] = {'n': 0, 'wr': 0, 'avg': 0, 'med': 0}

    # Stop hit orani (AL, ilk pencere icinde)
    al_with_first = al[al[first_ret_key].notna()]
    if len(al_with_first) > 0:
        summary['stop_pct_first'] = round(al_with_first['stop_hit'].sum() / len(al_with_first) * 100, 1)
    else:
        summary['stop_pct_first'] = 0.0

    # AL istatistikleri (pencere bazli)
    summary['al_stats'] = {}
    for rkey in ret_keys:
        valid = al[al[rkey].notna()][rkey]
        n = len(valid)
        if n > 0:
            wr = (valid > 0).sum() / n * 100
            summary['al_stats'][rkey] = {'n': n, 'wr': round(wr, 1), 'avg': round(valid.mean(), 2)}
        else:
            summary['al_stats'][rkey] = {'n': 0, 'wr': 0, 'avg': 0}

    # Gecis tipi bazinda (AL, top 5)
    summary['by_transition_al'] = {}
    if len(al) > 0:
        trans_counts = al['transition'].value_counts()
        for trans_name in trans_counts.index[:8]:
            t_df = al[al['transition'] == trans_name]
            valid_5 = t_df[t_df[first_ret_key].notna()][first_ret_key]
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
            valid_first = s_df[s_df[first_ret_key].notna()]
            n = len(valid_first)
            if n > 0:
                wr = (valid_first[first_ret_key] > 0).sum() / n * 100
                stop_r = valid_first['stop_hit'].sum() / n * 100
                summary['by_score_al'][score] = {
                    'n': n, 'wr': round(wr, 1),
                    'avg': round(valid_first[first_ret_key].mean(), 2),
                    'stop_pct': round(stop_r, 1),
                }
        # <=1 grubu
        s_df = al[al['entry_score'] <= 1]
        valid_first = s_df[s_df[first_ret_key].notna()]
        n = len(valid_first)
        if n > 0:
            wr = (valid_first[first_ret_key] > 0).sum() / n * 100
            stop_r = valid_first['stop_hit'].sum() / n * 100
            summary['by_score_al']['<=1'] = {
                'n': n, 'wr': round(wr, 1),
                'avg': round(valid_first[first_ret_key].mean(), 2),
                'stop_pct': round(stop_r, 1),
            }

    # Exit stage bazinda (AL)
    summary['by_exit_al'] = {}
    if len(al) > 0:
        for exit_val in [0, 1]:
            e_df = al[al['exit_stage'] == exit_val]
            valid_first = e_df[e_df[first_ret_key].notna()]
            n = len(valid_first)
            if n > 0:
                wr = (valid_first[first_ret_key] > 0).sum() / n * 100
                stop_r = valid_first['stop_hit'].sum() / n * 100
                summary['by_exit_al'][exit_val] = {
                    'n': n, 'wr': round(wr, 1),
                    'avg': round(valid_first[first_ret_key].mean(), 2),
                    'stop_pct': round(stop_r, 1),
                }
        # >=2 grubu
        e_df = al[al['exit_stage'] >= 2]
        valid_first = e_df[e_df[first_ret_key].notna()]
        n = len(valid_first)
        if n > 0:
            wr = (valid_first[first_ret_key] > 0).sum() / n * 100
            stop_r = valid_first['stop_hit'].sum() / n * 100
            summary['by_exit_al']['>=2'] = {
                'n': n, 'wr': round(wr, 1),
                'avg': round(valid_first[first_ret_key].mean(), 2),
                'stop_pct': round(stop_r, 1),
            }

    # Filtreli (AL, Score>=3, Exit<=1)
    if len(al) > 0:
        filt = al[(al['entry_score'] >= 3) & (al['exit_stage'] <= 1)]
        summary['filtered_al'] = {}
        for rkey in ret_keys:
            valid = filt[filt[rkey].notna()][rkey]
            n = len(valid)
            if n > 0:
                wr = (valid > 0).sum() / n * 100
                summary['filtered_al'][rkey] = {
                    'n': n, 'wr': round(wr, 1), 'avg': round(valid.mean(), 2),
                }
            else:
                summary['filtered_al'][rkey] = {'n': 0, 'wr': 0, 'avg': 0}
        summary['filtered_al']['total'] = len(filt)
    else:
        summary['filtered_al'] = {'total': 0}

    return summary


def _print_backtest_summary(summary, N, n_scanned, date_str, timeframe='daily'):
    """Backtest ozet ciktisi."""
    tf_label = _TF_LABELS.get(timeframe, 'Gunluk')
    ret_keys = summary.get('ret_keys', ['ret_5d', 'ret_10d', 'ret_20d'])
    first_key = ret_keys[0]
    # Kisa etiketler: ret_5d→5G, ret_3w→3H, ret_2m→2A
    def _rk_label(rk):
        s = rk.replace('ret_', '').upper()
        return s.replace('D', 'G').replace('W', 'H').replace('M', 'A')

    w = 70
    print(f"\n{'═' * w}")
    print(f"  REGIME TRANSITION BACKTEST ({tf_label}) — Son {N} bar")
    print(f"{'═' * w}")
    print(f"  {n_scanned} hisse, {summary['total']} AL sinyal")

    # Genel
    print(f"\n  GENEL")
    print(f"  {'Pencere':<10} {'N':>6} {'WR%':>7} {'Ort%':>8} {'Med%':>8}")
    print(f"  {'─' * 45}")
    for rkey in ret_keys:
        s = summary.get(f'all_{rkey}', {})
        n = s.get('n', 0)
        label = _rk_label(rkey)
        if n > 0:
            print(f"  {label:<10} {n:>6} {s['wr']:>6.1f}% {s['avg']:>+7.2f} {s['med']:>+7.2f}")
        else:
            print(f"  {label:<10} {n:>6}    —       —       —")

    # AL pencere bazinda
    al_stats = summary.get('al_stats', {})
    if al_stats:
        print(f"\n  AL PENCERE BAZINDA")
        print(f"  {'Pencere':<10} {'N':>6} {'WR%':>7} {'Ort%':>8}")
        print(f"  {'─' * 35}")
        for rkey in ret_keys:
            s = al_stats.get(rkey, {})
            n = s.get('n', 0)
            label = _rk_label(rkey)
            if n > 0:
                print(f"  {label:<10} {n:>6} {s['wr']:>6.1f}% {s['avg']:>+7.2f}")
            else:
                print(f"  {label:<10} {n:>6}    —       —")

    # Gecis tipi (AL, top entries)
    first_label = _rk_label(first_key)
    by_trans = summary.get('by_transition_al', {})
    if by_trans:
        print(f"\n  GECIS TIPI (AL, Top {min(5, len(by_trans))})")
        print(f"  {'Gecis':<28} {'N':>5} {first_label+' WR%':>8} {first_label+' Ort%':>9}")
        print(f"  {'─' * 55}")
        sorted_trans = sorted(by_trans.items(), key=lambda x: x[1]['n'], reverse=True)
        for trans_name, ts in sorted_trans[:5]:
            print(f"  {trans_name:<28} {ts['n']:>5} {ts['wr']:>7.1f}% {ts['avg']:>+8.2f}")

    # Giris skoru (AL)
    by_score = summary.get('by_score_al', {})
    if by_score:
        print(f"\n  GIRIS SKORU (AL)")
        print(f"  {'Skor':<6} {'N':>5} {first_label+' WR%':>8} {first_label+' Ort%':>9} {'Stop%':>7}")
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
        print(f"  {'Exit':<6} {'N':>5} {first_label+' WR%':>8} {first_label+' Ort%':>9} {'Stop%':>7}")
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
        f1 = filt.get(ret_keys[0], {})
        f2 = filt.get(ret_keys[1], {}) if len(ret_keys) > 1 else {}
        k1_label = _rk_label(ret_keys[0])
        k2_label = _rk_label(ret_keys[1]) if len(ret_keys) > 1 else ''
        print(f"\n  FILTRELI (AL, Score>=3, Exit<=1)")
        print(f"  N: {filt_total}  "
              f"{k1_label}: WR {f1.get('wr', 0):.1f}% Ort {f1.get('avg', 0):+.2f}%  "
              f"{k2_label}: WR {f2.get('wr', 0):.1f}% Ort {f2.get('avg', 0):+.2f}%")

    print(f"{'═' * w}")


def _save_backtest_csv(signals, date_str, output_dir, timeframe='daily'):
    """Backtest sinyallerini CSV'ye kaydet."""
    if not signals:
        print(f"\n  Sinyal yok, CSV olusturulmadi.")
        return
    os.makedirs(output_dir, exist_ok=True)
    csv_df = pd.DataFrame(signals)
    tf_suffix = f"_{timeframe}" if timeframe != 'daily' else ''
    fname = f"regime_transition{tf_suffix}_backtest_{date_str.replace('-', '')}.csv"
    path = os.path.join(output_dir, fname)
    csv_df.to_csv(path, index=False)
    print(f"\n  Backtest CSV: {path} ({len(signals)} sinyal)")


# =============================================================================
# TELEGRAM
# =============================================================================

def _format_telegram(results, n_scanned, date_str, tf_label='', html_url=None):
    """Telegram HTML mesaji olustur."""
    tf_tag = f" ({tf_label.strip()})" if tf_label.strip() else ''
    lines = []
    if html_url:
        lines.append(f'🔗 <a href="{html_url}">Detayli Rapor</a>\n')
    lines.append(f"📊 <b>NOX Regime Transition{tf_tag}</b> — {date_str}\n")
    lines.append(f"✅ {len(results)} AL aktif | {n_scanned} hisse tarandi\n")

    # Gecis tipi dagilimi
    trans_counts = Counter(s.transition for s in results)
    top_trans = trans_counts.most_common(5)
    if top_trans:
        parts = [f"{t}: {c}" for t, c in top_trans]
        lines.append(f"⭐ {' | '.join(parts)}\n")

    # Top 10 ticker
    top = results[:10]
    if top:
        names = [f"{s.ticker}" for s in top]
        lines.append(f"📋 Top 10: {', '.join(names)}")

    return '\n'.join(lines)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="NOX Regime Transition Screener")
    parser.add_argument('--tickers', nargs='*', help='Spesifik ticker(lar)')
    parser.add_argument('--period', default=None, help='Veri periyodu (default: timeframe\'e gore)')
    parser.add_argument('--timeframe', '--tf', default='daily',
                        choices=['daily', 'weekly', 'monthly'],
                        help='Timeframe (default: daily)')
    parser.add_argument('--csv', action='store_true', help='CSV kaydet')
    parser.add_argument('--html', action='store_true', help='HTML rapor kaydet ve ac')
    parser.add_argument('--output', default='output', help='Cikti dizini')
    parser.add_argument('--backtest', type=int, metavar='N',
                        help='Backtest: son N bar icin forward return hesapla')
    parser.add_argument('--notify', action='store_true',
                        help='Telegram + GitHub Pages bildirim')
    args = parser.parse_args()

    timeframe = args.timeframe
    tf_label = _TF_LABELS.get(timeframe, 'Gunluk')
    cfg = TIMEFRAME_CONFIGS.get(timeframe, RT_CFG)
    period = args.period or _TF_DEFAULT_PERIOD.get(timeframe, '2y')

    w = 80
    print(f"\n{'═' * w}")
    print(f"  NOX REGIME TRANSITION SCREENER ({tf_label})")
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
    print(f"\n  Veri yukleniyor (period={period})...")
    t0 = time.time()
    all_data = data_mod.fetch_data(tickers, period=period)
    print(f"  {len(all_data)} hisse yuklendi ({time.time() - t0:.1f}s)")

    if not all_data:
        print("  HATA: Hicbir hisse verisi yuklenemedi!")
        sys.exit(1)

    # ── 3. Lowercase donusum ─────────────────────────────────────────────────
    daily_dfs = {ticker: _to_lower_cols(df) for ticker, df in all_data.items()}

    # ── 4. Timeframe'e gore primary + higher TF veri hazirligi ───────────────
    if timeframe == 'daily':
        stock_dfs = daily_dfs
        print(f"\n  Haftalik resample (higher TF)...")
        higher_tf_dfs = {t: _to_weekly(df) for t, df in daily_dfs.items()}
        higher_tf_dfs = {t: df for t, df in higher_tf_dfs.items() if len(df) >= 25}
    elif timeframe == 'weekly':
        print(f"\n  Haftalik resample (primary)...")
        stock_dfs = {t: _to_weekly(df, include_incomplete=True) for t, df in daily_dfs.items()}
        stock_dfs = {t: df for t, df in stock_dfs.items() if len(df) >= _TF_MIN_BARS['weekly']}
        print(f"\n  Aylik resample (higher TF)...")
        higher_tf_dfs = {t: _to_monthly(df) for t, df in daily_dfs.items()}
        higher_tf_dfs = {t: df for t, df in higher_tf_dfs.items() if len(df) >= 12}
    else:  # monthly
        print(f"\n  Aylik resample (primary)...")
        stock_dfs = {t: _to_monthly(df, include_incomplete=True) for t, df in daily_dfs.items()}
        stock_dfs = {t: df for t, df in stock_dfs.items() if len(df) >= _TF_MIN_BARS['monthly']}
        higher_tf_dfs = None  # monthly icin higher TF yok

    # ── 5. Backtest veya normal tarama ──────────────────────────────────────
    if args.backtest:
        print(f"\n  Backtest modu: son {args.backtest} bar...")
        t1 = time.time()
        signals, summary = _run_backtest(stock_dfs, higher_tf_dfs, args.backtest,
                                         cfg=cfg, timeframe=timeframe)
        n_scanned = summary.get('n_scanned', 0)
        date_str = summary.get('date_str', datetime.now().strftime('%Y-%m-%d'))
        print(f"  {n_scanned} hisse tarandi, {len(signals)} sinyal ({time.time() - t1:.1f}s)")
        _print_backtest_summary(summary, args.backtest, n_scanned, date_str, timeframe=timeframe)
        if args.csv:
            _save_backtest_csv(signals, date_str, args.output, timeframe=timeframe)
    else:
        print(f"\n  Regime transition taramasi ({tf_label.lower()})...")
        t1 = time.time()
        results, n_scanned, date_str, regime_dist = _scan_all(
            stock_dfs, higher_tf_dfs, cfg=cfg, timeframe=timeframe)
        print(f"  {n_scanned} hisse tarandi ({time.time() - t1:.1f}s)")

        # ── 6. Rapor ─────────────────────────────────────────────────────────
        _print_results(results, n_scanned, date_str, regime_dist)

        # ── 7. Haftalik AL + gunluk pullback (badge hesaplama, CSV'den once) ─
        weekly_al_pb = None
        if timeframe == 'daily' and results:
            weekly_al_pb = _compute_weekly_al_with_daily_pb(
                higher_tf_dfs, None, daily_dfs)
            n_wal = len(weekly_al_pb)
            n_pb = sum(1 for v in weekly_al_pb.values() if v['pb'])
            print(f"  Haftalik AL: {n_wal} hisse | H+PB: {n_pb} hisse")

        # ── 8. CSV (badge bilgisi dahil) ─────────────────────────────────────
        if args.csv:
            _save_csv(results, date_str, args.output, timeframe=timeframe,
                      weekly_al_pb=weekly_al_pb)

        # ── 9. HTML ──────────────────────────────────────────────────────────
        html_url = None
        if args.html:
            html = _generate_html(results, n_scanned, date_str, regime_dist,
                                  timeframe=timeframe, weekly_al_pb=weekly_al_pb)
            html_path = _save_html(html, date_str, args.output, timeframe=timeframe)
            # GitHub Pages push
            tf_suffix = f"_{timeframe}" if timeframe != 'daily' else ''
            gh_filename = f"regime_transition{tf_suffix}.html"
            html_url = push_html_to_github(html, gh_filename, date_str)
            if not args.notify:
                subprocess.Popen(['open', html_path])

        # ── 9. Telegram ────────────────────────────────────────────────────
        if args.notify:
            msg = _format_telegram(results, n_scanned, date_str, tf_label,
                                   html_url=html_url)
            send_telegram(msg)

    print(f"\n  Toplam sure: {time.time() - t0:.1f}s")
    print(f"  NOX Regime Transition ({tf_label}) tamamlandi.\n")


if __name__ == '__main__':
    main()
