#!/usr/bin/env python3
"""
NOX Project — Ana Orkestrator
--market bist|us  (default: bist)
--mode trend|dip  (default: auto — hafta içi=trend, hafta sonu=dip)
--debug           debug bilgileri göster
--ticker XXXXX    tek hisse analiz et
--date YYYY-MM-DD backtest tarihi
"""
import sys
import os
import datetime
import importlib
import pandas as pd
from collections import Counter

# Root path
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from core.config import (
    SIGNAL_EMOJI, SIGNAL_PRIORITY_TREND, SIGNAL_PRIORITY_DIP,
    SIGNAL_PRIORITY_SIDEWAYS, REGIME_SHORT,
)
from core.indicators import calc_xu100_market_state


def parse_args():
    args = {
        'market': 'bist',
        'mode': None,
        'debug': '--debug' in sys.argv,
        'ticker': None,
        'cutoff_date': None,
    }
    for i, arg in enumerate(sys.argv):
        if arg == '--market' and i + 1 < len(sys.argv):
            args['market'] = sys.argv[i + 1].lower()
        if arg == '--mode' and i + 1 < len(sys.argv):
            args['mode'] = sys.argv[i + 1].lower()
        if arg == '--ticker' and i + 1 < len(sys.argv):
            args['ticker'] = sys.argv[i + 1].upper()
        if arg == '--date' and i + 1 < len(sys.argv):
            args['cutoff_date'] = sys.argv[i + 1]
    return args


def determine_mode(forced_mode=None):
    if forced_mode in ('trend', 'regime', 'dip'):
        return 'dip' if forced_mode == 'dip' else 'trend'
    today = datetime.datetime.now().weekday()
    return 'dip' if today >= 5 else 'trend'


def load_market_modules(market_name):
    """Dinamik olarak market modüllerini yükle. 4-tuple: data, regime, dip, sideways."""
    data_mod = importlib.import_module(f'markets.{market_name}.data')
    regime_mod = importlib.import_module(f'markets.{market_name}.regime')
    dip_mod = importlib.import_module(f'markets.{market_name}.dip')
    try:
        sideways_mod = importlib.import_module(f'markets.{market_name}.sideways')
    except ImportError:
        sideways_mod = None
    return data_mod, regime_mod, dip_mod, sideways_mod


def run_trend(all_data, xu_df, usd_df, regime_mod, debug_mode, single_ticker, market_state=None):
    print("🔄 Trend analiz ediliyor...\n")
    dbg = {
        'total': 0, 'no_atr': 0, 'low_vol': 0, 'no_signal': 0, 'low_active': 0,
        'exception': 0, 'signal': 0,
        'regime': {0: 0, 1: 0, 2: 0, 3: 0},
        'wt_recent': 0, 'pmax_long': 0, 'bos': 0, 'choch': 0,
        'rs_fail': 0, 'q_fail': 0,
        'strong_check': 0, 'weak_check': 0, 'donus_check': 0,
        'early_check': 0, 'pb_check': 0, 'sq_check': 0, 'mr_check': 0,
        'combo_base': 0, 'combo_plus': 0, 'combo_bos': 0,
    }
    results = []
    for t, df in all_data.items():
        r = regime_mod.analyze_regime(t, df, xu_df, dbg if debug_mode else None, usd_df=usd_df, market_state=market_state)
        if r:
            results.append(r)
    if debug_mode:
        print(f"\n{'='*60}\n🔍 DEBUG — Trend\n{'='*60}")
        print(f"  Analiz: {dbg['total']} | Sinyal: {dbg['signal']} | Exception: {dbg['exception']}")
        if market_state:
            print(f"  Market State: risk_on={market_state.get('risk_on')} sideways={market_state.get('sideways')} weekly_st={market_state.get('weekly_st_up')}")
        for k in ('panic_block', 'sideways_block', 'riskoff_block', 'mode_filter', 'combo_filter', 'q_filter'):
            if dbg.get(k, 0) > 0:
                print(f"  {k}: {dbg[k]}")
    results.sort(key=lambda x: (SIGNAL_PRIORITY_TREND.get(x['signal'], 99), -x['rr']))
    return results


def run_dip(all_data, xu_df, usd_df, dip_mod, debug_mode, single_ticker):
    print("🔄 DIP analiz ediliyor...\n")
    dbg = {
        'total': 0, 'no_atr': 0, 'low_vol': 0, 'no_signal': 0,
        'exception': 0, 'signal': 0, 'regime': {},
    }
    results = []
    for t, df in all_data.items():
        r = dip_mod.analyze_dip(t, df, xu_df, usd_df=usd_df, dbg=dbg if debug_mode else None)
        if r:
            results.append(r)
    if debug_mode:
        print(f"\n🔍 DEBUG — DIP: {dbg['total']} analiz, {dbg['signal']} sinyal")
    results.sort(key=lambda x: (SIGNAL_PRIORITY_DIP.get(x['signal'], 99), -x['rr']))
    return results


def run_sideways(all_data, xu_df, usd_df, sideways_mod, debug_mode, single_ticker, market_state=None):
    print("🔄 Sideways analiz ediliyor...\n")
    dbg = {
        'total': 0, 'no_atr': 0, 'low_vol': 0, 'no_signal': 0,
        'exception': 0, 'signal': 0, 'panic_block': 0,
        'mr_checks': 0, 'mr_signal': 0, 'sq_checks': 0, 'sq_signal': 0,
    }
    results = []
    for t, df in all_data.items():
        r = sideways_mod.analyze_sideways(t, df, xu_df, dbg if debug_mode else None, usd_df=usd_df, market_state=market_state)
        if r:
            results.append(r)
    if debug_mode:
        print(f"\n{'='*60}\n🔍 DEBUG — Sideways\n{'='*60}")
        print(f"  Analiz: {dbg['total']} | Sinyal: {dbg['signal']} | Exception: {dbg.get('exception', 0)}")
        print(f"  MR checks: {dbg.get('mr_checks', 0)} signals: {dbg.get('mr_signal', 0)}")
        print(f"  SQ checks: {dbg.get('sq_checks', 0)} signals: {dbg.get('sq_signal', 0)}")
        if market_state:
            print(f"  Market State: sideways={market_state.get('sideways')} weekly_st={market_state.get('weekly_st_up')}")
        for k in ('panic_block', 'low_vol', 'no_atr', 'log_only'):
            if dbg.get(k, 0) > 0:
                print(f"  {k}: {dbg[k]}")
    results.sort(key=lambda x: (SIGNAL_PRIORITY_SIDEWAYS.get(x['signal'], 99), -x['rr']))
    return results


def print_results(results, mode):
    print(f"\n{'='*140}")
    for r in results:
        emoji = SIGNAL_EMOJI.get(r['signal'], "")
        rs = REGIME_SHORT.get(r['regime'], "?")
        oe = f" ⚠OE{r.get('overext_score', 0)}" if r.get('overext_warning') else ""
        kc_tags = ",".join(r.get('kc_tags', []))
        kc_str = f" [{kc_tags}]={r.get('kc_score', 0)}" if kc_tags else ""
        mode_str = r.get('trade_mode', '-')
        pos_str = f"{r.get('pos_size', '-')}"
        atr_pct = f"{r.get('atr_pctile', 0):.0%}" if r.get('atr_pctile') is not None else "-"
        print(f"{emoji} {r['ticker']:8s} [{r['signal']:12s}] {mode_str:8s} pos:{pos_str:4s} {rs:2s}  "
              f"Fiyat:{r['close']:>9.2f} S:{r['stop']:>9.2f} TP:{r['tp']:>9.2f}({r['tp_src']}) "
              f"R:R={r['rr']:.1f} RS:{r['rs_score']:+.1f} Q:{r['quality']} RVOL:{r['rvol']}x ATR%:{atr_pct}{oe}{kc_str}")


def save_outputs(results, mode, market_name, total):
    from core.reports import (
        generate_regime_html, generate_dip_html,
        generate_sideways_html,
        format_regime_telegram, format_dip_telegram,
        format_sideways_telegram,
        send_telegram, send_telegram_document,
        push_html_to_github,
    )
    date_str = datetime.datetime.now().strftime('%Y%m%d')

    if mode == 'sideways':
        html = generate_sideways_html(results, total, market_label=market_name.upper())
        html_file = f"nox_{market_name}_sideways_{date_str}.html"
        gh_filename = f"nox_{market_name}_sideways.html"
    elif mode == 'trend':
        html = generate_regime_html(results, total, market_label=market_name.upper())
        html_file = f"nox_{market_name}_trend_{date_str}.html"
        gh_filename = f"nox_{market_name}_trend.html"
    else:
        html = generate_dip_html(results, total, market_label=market_name.upper())
        html_file = f"nox_{market_name}_dip_{date_str}.html"
        gh_filename = f"nox_{market_name}_dip.html"

    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"📄 {html_file}")

    html_url = push_html_to_github(html, gh_filename, date_str)

    if results:
        df_out = pd.DataFrame(results)
        csv_file = f"nox_{market_name}_{mode}_{date_str}.csv"
        df_out.to_csv(csv_file, index=False)
        print(f"💾 {csv_file}")

    if mode == 'sideways':
        msg = format_sideways_telegram(results, total, html_url)
    elif mode == 'trend':
        msg = format_regime_telegram(results, total, html_url)
    else:
        msg = format_dip_telegram(results, total, html_url)
    send_telegram(msg)
    send_telegram_document(html_file)

    return html_file


def main():
    args = parse_args()
    market_name = args['market']
    mode = determine_mode(args['mode'])
    debug_mode = args['debug']
    single_ticker = args['ticker']
    cutoff_date = args['cutoff_date']

    market_label = market_name.upper()
    mode_label = f"⬡ NOX TREND · {market_label}" if mode == 'trend' else f"⬡ NOX DIP · {market_label}"
    print(f"{'='*60}")
    print(f"{mode_label}")
    print(f"{'='*60}")

    if cutoff_date:
        print(f"📅 Backtest modu: veri {cutoff_date} tarihine kadar kesilecek")

    # Market modüllerini yükle
    data_mod, regime_mod, dip_mod, sideways_mod = load_market_modules(market_name)

    # Ticker listesi
    if market_name == 'bist':
        tickers = data_mod.get_all_bist_tickers()
    elif market_name == 'us':
        tickers = data_mod.get_all_us_tickers()
    elif market_name == 'crypto':
        tickers = data_mod.get_all_crypto_tickers()
    elif market_name == 'commodity':
        tickers = data_mod.get_all_commodity_tickers()
    else:
        print(f"❌ Bilinmeyen market: {market_name}")
        return

    if single_ticker:
        if single_ticker not in tickers:
            print(f"⚠️ {single_ticker} listede yok!")
            similar = [t for t in tickers if single_ticker[:3] in t][:5]
            if similar:
                print(f"   Benzer: {similar}")
            return
        tickers = [single_ticker]
        debug_mode = True
        print(f"🔎 Tek hisse debug: {single_ticker}")

    # Veri çek
    period = "5y" if mode == 'dip' else "1y"
    all_data = data_mod.fetch_data(tickers, period=period)

    if cutoff_date:
        cut = pd.Timestamp(cutoff_date)
        all_data = {t: df[df.index <= cut] for t, df in all_data.items()}

    # Benchmark
    xu_df = data_mod.fetch_benchmark(period=period)
    if cutoff_date and xu_df is not None:
        xu_df = xu_df[xu_df.index <= pd.Timestamp(cutoff_date)]

    # USDTRY (sadece BIST)
    usd_df = None
    if market_name == 'bist' and hasattr(data_mod, 'fetch_usdtry'):
        usd_df = data_mod.fetch_usdtry(period="5y")
        if cutoff_date and usd_df is not None:
            usd_df = usd_df[usd_df.index <= pd.Timestamp(cutoff_date)]

    total = len(all_data)
    print(f"📡 {total} hisse analiz ediliyor ({market_label} {mode})...\n")

    # Market state (1 kere hesapla, tüm ticker'lara geç)
    market_state = None
    if xu_df is not None and mode == 'trend':
        market_state = calc_xu100_market_state(xu_df)
        if debug_mode:
            ms_print = {k: v for k, v in market_state.items() if k != 'sideways_flag_series'}
            print(f"🌐 Market State: {ms_print}")

    # Analiz — sideways router
    if mode == 'trend':
        if market_state and market_state.get('sideways') and sideways_mod:
            # TREND disabled during sideways, enable sideways modules
            results = run_sideways(all_data, xu_df, usd_df, sideways_mod, debug_mode, single_ticker, market_state=market_state)
            mode = 'sideways'  # switch mode for output
        else:
            results = run_trend(all_data, xu_df, usd_df, regime_mod, debug_mode, single_ticker, market_state=market_state)
    else:
        results = run_dip(all_data, xu_df, usd_df, dip_mod, debug_mode, single_ticker)

    # Sinyal özeti
    sig_counts = Counter(r['signal'] for r in results)
    print(f"\n📋 {total} taranan | {len(results)} sinyal")
    for sig, cnt in sig_counts.most_common():
        print(f"   {SIGNAL_EMOJI.get(sig, '')} {sig}: {cnt}")

    print_results(results, mode)
    save_outputs(results, mode, market_name, total)
    print(f"\n📋 Taranan: {total} | Toplam Sinyal: {len(results)}")
    return results


if __name__ == "__main__":
    main()
