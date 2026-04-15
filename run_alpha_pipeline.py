#!/usr/bin/env python3
"""
NYX Alpha Pipeline — BIST 5-Aşamalı Yatırım Sistemi
=====================================================
Kullanım:
    python run_alpha_pipeline.py                     # 2Y backtest
    python run_alpha_pipeline.py --live              # Bugünkü sinyal tarama
    python run_alpha_pipeline.py --period 5y         # 5Y backtest
    python run_alpha_pipeline.py --rebalance weekly  # Haftalık rebalance
    python run_alpha_pipeline.py --capital 500000    # Özel sermaye
"""

import argparse
import os
import sys
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from markets.bist.data import get_all_bist_tickers, fetch_data, fetch_benchmark
from alpha.config import (
    DATA_PERIOD, INITIAL_CAPITAL, MIN_VOLUME_TL,
    REBALANCE_FREQ, WF_STEP_DAYS, MIN_DATA_DAYS,
)
from alpha.stages import scan_universe
from alpha.portfolio import build_portfolio
from alpha.backtest_wf import WalkForwardBacktester
from alpha.metrics import compute_alpha_metrics, compute_monthly_returns, compute_trade_stats
from alpha.report import generate_alpha_report


def _filter_liquid(all_data: dict, min_vol_tl: float) -> dict:
    """Düşük hacimli hisseleri ele."""
    filtered = {}
    dropped = 0
    for ticker, df in all_data.items():
        if df is None or len(df) < MIN_DATA_DAYS:
            dropped += 1
            continue
        avg_vol = (df['Close'] * df['Volume']).tail(20).mean()
        if avg_vol >= min_vol_tl:
            filtered[ticker] = df
        else:
            dropped += 1
    if dropped:
        print(f"  {dropped} hisse elendi (hacim < {min_vol_tl/1e6:.0f}M TL veya yetersiz veri)")
    return filtered


def run_backtest(args):
    """Walk-forward backtest modu."""
    period = args.period
    capital = args.capital
    rebal = args.rebalance

    # Rebalance frekansına göre step ayarla
    step_map = {'weekly': 5, 'biweekly': 10, 'monthly': 21}
    import alpha.config as cfg
    cfg.WF_STEP_DAYS = step_map.get(rebal, 10)
    cfg.INITIAL_CAPITAL = capital

    print("=" * 60)
    print("  NYX ALPHA PIPELINE — BIST")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Periyot: {period} | Rebalance: {rebal} | Sermaye: {capital:,.0f} TL")
    print("=" * 60)

    # 1. Veri çek
    t0 = time.time()
    print("\n[1/5] Veri çekiliyor...")
    tickers = get_all_bist_tickers()
    print(f"  {len(tickers)} hisse bulundu")
    all_data = fetch_data(tickers, period=period)
    xu_df = fetch_benchmark(period=period)
    print(f"  {len(all_data)} hisse yüklendi ({time.time()-t0:.0f}s)")

    # 2. Likidite filtresi
    print("\n[2/5] Likidite filtresi...")
    all_data = _filter_liquid(all_data, MIN_VOLUME_TL)
    print(f"  {len(all_data)} hisse kaldı")

    # 3. Walk-forward backtest
    print(f"\n[3/5] Walk-forward backtest ({rebal})...")
    t1 = time.time()
    bt = WalkForwardBacktester(all_data, xu_df)
    result = bt.run()
    print(f"  {len(result['rebalance_events'])} rebalance, {len(result['trades'])} trade ({time.time()-t1:.0f}s)")

    # 4. Metrikler
    print("\n[4/5] Metrikler hesaplanıyor...")
    metrics = compute_alpha_metrics(result['equity_curve'], capital)
    monthly_rets = compute_monthly_returns(result['equity_curve'])
    trade_stats = compute_trade_stats(result['trades'])

    # 5. Rapor
    print("\n[5/5] HTML rapor üretiliyor...")
    filepath = generate_alpha_report(result, metrics, monthly_rets, trade_stats)

    _print_summary(metrics, trade_stats)
    print(f"\n  Rapor: {filepath}")
    print("=" * 60)


def run_live_scan(args):
    """Canlı tarama — bugünkü sinyaller ve önerilen portföy."""
    print("=" * 60)
    print("  NYX ALPHA PIPELINE — CANLI TARAMA")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    # Veri çek
    print("\n[1/3] Veri çekiliyor...")
    tickers = get_all_bist_tickers()
    all_data = fetch_data(tickers, period='1y')
    xu_df = fetch_benchmark(period='1y')
    all_data = _filter_liquid(all_data, MIN_VOLUME_TL)
    print(f"  {len(all_data)} hisse")

    # Aşama 1-3
    print("\n[2/3] Evren taranıyor (Aşama 1-3)...")
    candidates = scan_universe(all_data)
    passed = [c for c in candidates if c.get('passed')]

    print(f"  Aşama 1-2 adayları: {len(candidates)}")
    print(f"  Aşama 3 onaylı:     {len(passed)}")

    if candidates:
        print(f"\n  {'Hisse':<10} {'Tip':<14} {'Skor':>5} {'Eğim':>7} {'ADX':>5} {'CMF':>6} {'RSI':>5} {'Mum':<12} {'Durum':<6}")
        print("  " + "-" * 75)
        for c in candidates[:25]:
            m = c['momentum']
            s = c['slope']
            cf = c['confirmation']
            status = '✅' if c['passed'] else '❌'
            candle = cf.get('candle_pattern') or '—'
            print(f"  {c['ticker']:<10} {m['momentum_type'] or '—':<14} "
                  f"{c['composite_score']:>5.1f} {s['price_slope']:>+6.2f}% "
                  f"{cf['adx']:>5.1f} {cf['cmf']:>+5.3f} {cf['rsi']:>5.1f} "
                  f"{candle:<12} {status}")

    # Aşama 4-5
    if len(passed) >= 3:
        print("\n[3/3] Portföy optimizasyonu (Aşama 4-5)...")
        portfolio = build_portfolio(all_data, candidates)
        if portfolio['n_stocks'] > 0:
            print(f"\n  Max Sharpe Portföy (Sharpe: {portfolio['sharpe_ratio']:.2f})")
            print(f"  Beklenen Getiri: {portfolio['expected_return']:+.1f}%  Risk: {portfolio['expected_risk']:.1f}%")
            print(f"\n  {'Hisse':<10} {'Ağırlık':>8}")
            print("  " + "-" * 20)
            for t, w in sorted(portfolio['weights'].items(), key=lambda x: -x[1]):
                print(f"  {t:<10} {w*100:>7.1f}%")
        else:
            print("  Portföy oluşturulamadı (yetersiz aday)")
    else:
        print(f"\n  Aşama 3'ten {len(passed)} hisse geçti — minimum 3 gerekli, portföy oluşturulamıyor")

    print("\n" + "=" * 60)


def _print_summary(metrics: dict, trade_stats: dict):
    """Konsol özet."""
    m = metrics
    ts = trade_stats

    print("\n" + "─" * 50)
    print("  SONUÇLAR")
    print("─" * 50)
    print(f"  Toplam Getiri:  {m['total_return']:+.1f}%")
    print(f"  XU100:          {m['benchmark_total']:+.1f}%")
    print(f"  Alpha:          {m['alpha']:+.1f}%")
    print(f"  Jensen Alpha:   {m['jensens_alpha']:+.1f}%")
    print(f"  Sharpe:         {m['sharpe_ratio']:.3f}")
    print(f"  Sortino:        {m['sortino_ratio']:.3f}")
    print(f"  Info Ratio:     {m['information_ratio']:.3f}")
    print(f"  Max DD:         {m['max_drawdown']:.1f}%")
    print(f"  Calmar:         {m['calmar_ratio']:.3f}")
    print(f"  Beta:           {m['beta']:.3f}")
    print(f"  Win Rate:       {ts.get('win_rate',0):.0f}% ({ts.get('n_trades',0)} trade)")
    print(f"  Profit Factor:  {ts.get('profit_factor',0):.2f}")
    print(f"  Ort. Tutma:     {ts.get('avg_hold',0):.0f} gün")
    print(f"  Süre:           {m['years']:.1f} yıl ({m['n_trading_days']} gün)")
    print("─" * 50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NYX Alpha Pipeline — BIST')
    parser.add_argument('--live', action='store_true', help='Canlı sinyal tarama')
    parser.add_argument('--capital', type=float, default=INITIAL_CAPITAL, help='Başlangıç sermayesi')
    parser.add_argument('--rebalance', choices=['weekly', 'biweekly', 'monthly'],
                        default=REBALANCE_FREQ, help='Rebalance frekansı')
    parser.add_argument('--period', default=DATA_PERIOD, help='Veri periyodu (1y, 2y, 5y)')
    args = parser.parse_args()

    if args.live:
        run_live_scan(args)
    else:
        run_backtest(args)
