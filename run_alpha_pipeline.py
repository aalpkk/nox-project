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
from alpha.report import generate_alpha_report, generate_live_scan_report


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
    """Canlı tarama — bugünkü sinyaller ve önerilen portföy (ML destekli)."""
    from alpha.config import (
        ML_STAGE1_ENABLED, ML_SCORE_THRESHOLD, ML_SWING_THRESHOLD,
        ML_SLOPE_LOOKBACK, ML_SLOPE_MIN, ML_COMPOSITE_WEIGHT,
        CONFIRMATION_MIN_SCORE,
    )
    from core.indicators import calc_atr

    print("=" * 60)
    print("  NYX ALPHA PIPELINE — CANLI TARAMA")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    # Veri çek
    t0 = time.time()
    print("\n[1/4] Veri çekiliyor...")
    tickers = get_all_bist_tickers()
    all_data = fetch_data(tickers, period='1y')
    xu_df = fetch_benchmark(period='1y')
    all_data = _filter_liquid(all_data, MIN_VOLUME_TL)
    print(f"  {len(all_data)} hisse ({time.time()-t0:.0f}s)")

    # ML Tarama
    passed = []
    if ML_STAGE1_ENABLED:
        print("\n[2/4] ML tarama (Aşama 1-2)...")
        t1 = time.time()
        try:
            from ml.scorer import MLScorer
            from ml.features import compute_all_features
            ml_scorer = MLScorer()
            ml_scorer._load_models()
            if ml_scorer.loaded:
                from alpha.stages import stage3_confirmation
                ml_candidates = []
                for ticker, df in all_data.items():
                    if df is None or len(df) < 80:
                        continue
                    try:
                        feats = compute_all_features(df, xu_df=xu_df)
                        if feats.empty or len(feats) < 5:
                            continue
                        row = feats.iloc[-1]
                        vec = ml_scorer._make_feature_vector(row)
                        if vec is None:
                            continue
                        preds = ml_scorer._predict_all(vec)
                        ml_1g = preds['ml_a_1g']
                        ml_3g = preds['ml_a_3g']
                        if ml_1g is None or ml_1g < ML_SCORE_THRESHOLD:
                            continue
                        if ml_3g is not None and ml_3g < ML_SWING_THRESHOLD:
                            continue
                        # ML eğim kontrolü
                        if len(feats) > ML_SLOPE_LOOKBACK:
                            row_ago = feats.iloc[-1 - ML_SLOPE_LOOKBACK]
                            vec_ago = ml_scorer._make_feature_vector(row_ago)
                            if vec_ago is not None:
                                p_ago = ml_scorer._predict_all(vec_ago)
                                if p_ago['ml_a_1g'] is not None and (ml_1g - p_ago['ml_a_1g']) < ML_SLOPE_MIN:
                                    continue
                        # Stage 3 teknik onay
                        confirmation = stage3_confirmation(df)
                        if confirmation['score'] < CONFIRMATION_MIN_SCORE:
                            continue
                        # Composite skor
                        ml_avg = ml_1g if ml_3g is None else (ml_1g + ml_3g) / 2
                        composite = ml_avg * 100 * ML_COMPOSITE_WEIGHT + confirmation['score'] * 10 * (1 - ML_COMPOSITE_WEIGHT)
                        # ATR ve stop hesapla
                        atr_val = 0.0
                        if len(df) >= 20:
                            _atr = calc_atr(df)
                            if not pd.isna(_atr.iloc[-1]):
                                atr_val = float(_atr.iloc[-1])
                        close_px = float(df['Close'].iloc[-1])
                        stop_dist_pct = (atr_val * 2.0 / close_px * 100) if close_px > 0 else 0

                        ml_candidates.append({
                            'ticker': ticker.replace('.IS', ''),
                            'ml_1g': ml_1g,
                            'ml_3g': ml_3g,
                            'composite': round(min(100, composite), 1),
                            'adx': confirmation['adx'],
                            'cmf': confirmation['cmf'],
                            'rsi': confirmation['rsi'],
                            'conf_score': confirmation['score'],
                            'close': close_px,
                            'atr': atr_val,
                            'stop': round(close_px - 2.0 * atr_val, 2),
                            'stop_pct': round(stop_dist_pct, 1),
                            'trail_target': round(close_px + 1.5 * atr_val, 2),
                            'passed': True,
                        })
                    except Exception:
                        continue
                ml_candidates.sort(key=lambda x: x['composite'], reverse=True)
                passed = ml_candidates
                print(f"  {len(passed)} aday bulundu ({time.time()-t1:.0f}s)")
        except ImportError:
            print("  [UYARI] ML modülleri yüklenemedi, klasik taramaya düşüyor")

    # Klasik fallback
    if not passed:
        print("\n[2/4] Klasik tarama (Aşama 1-3)...")
        candidates = scan_universe(all_data)
        passed = [c for c in candidates if c.get('passed')]
        print(f"  {len(passed)} aday")

    # Sonuçları göster
    print(f"\n[3/4] Sonuçlar...")
    if passed:
        print(f"\n  {'Hisse':<8} {'ML1g':>5} {'ML3g':>5} {'Skor':>5} {'ADX':>5} {'CMF':>6} {'RSI':>5} {'Fiyat':>8} {'Stop':>8} {'Stop%':>6} {'Trail':>8}")
        print("  " + "-" * 85)
        for c in passed[:30]:
            ml3g_str = f"{c['ml_3g']:.2f}" if c.get('ml_3g') else "  —"
            print(f"  {c['ticker']:<8} {c['ml_1g']:>5.2f} {ml3g_str:>5} {c['composite']:>5.1f} "
                  f"{c['adx']:>5.1f} {c['cmf']:>+5.3f} {c['rsi']:>5.1f} "
                  f"{c['close']:>8.2f} {c['stop']:>8.2f} {c['stop_pct']:>5.1f}% "
                  f"{c['trail_target']:>8.2f}")

    # Portföy optimizasyonu
    if len(passed) >= 3:
        print(f"\n[4/4] Portföy optimizasyonu...")
        # passed'ı build_portfolio formatına çevir
        bp_candidates = [{'ticker': c['ticker'] + '.IS' if not c['ticker'].endswith('.IS') else c['ticker'],
                          'composite_score': c['composite'], 'passed': True} for c in passed]
        portfolio = build_portfolio(all_data, bp_candidates)
        if portfolio['n_stocks'] > 0:
            print(f"\n  Max Sharpe Portföy (Sharpe: {portfolio['sharpe_ratio']:.2f})")
            print(f"  Beklenen Getiri: {portfolio['expected_return']:+.1f}%  Risk: {portfolio['expected_risk']:.1f}%")
            print(f"\n  {'Hisse':<10} {'Ağırlık':>8}")
            print("  " + "-" * 20)
            for t, w in sorted(portfolio['weights'].items(), key=lambda x: -x[1]):
                print(f"  {t.replace('.IS',''):<10} {w*100:>7.1f}%")
        else:
            print("  Portföy oluşturulamadı")
            portfolio = None
    else:
        print(f"\n  {len(passed)} aday — minimum 3 gerekli, portföy oluşturulamıyor")
        portfolio = None

    # HTML rapor
    if passed:
        filepath = generate_live_scan_report(passed, portfolio)
        print(f"\n  Rapor: {filepath}")

        # GitHub Pages'e deploy
        try:
            from core.reports import push_html_to_github
            with open(filepath, 'r', encoding='utf-8') as f:
                html_content = f.read()
            push_html_to_github(html_content, 'alpha_scan.html',
                                datetime.now().strftime('%Y-%m-%d'))
        except Exception as e:
            print(f"  ⚠️ GitHub Pages deploy: {e}")

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
