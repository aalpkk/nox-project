"""
NOX Backtest — Runner
Ana çalıştırıcı. Komut satırından veya import ile kullanılır.

Kullanım:
  python -m backtest.run --mode trend [--max-tickers 50] [--period 10y]
  python -m backtest.run --mode dip
  python -m backtest.run --mode both
"""
import sys, os
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

import argparse
import pandas as pd
import time
from datetime import datetime

from markets.bist.data import get_all_bist_tickers, fetch_data, fetch_benchmark
from markets.bist import regime as regime_mod
from markets.bist import dip as dip_mod
from backtest.engine import run_backtest
from backtest.metrics import build_matrix, signal_breakdown, regime_breakdown
from backtest.report import generate_backtest_html, save_trades_csv


def parse_args():
    parser = argparse.ArgumentParser(description='NOX Backtest — BIST')
    parser.add_argument('--mode', choices=['trend', 'dip', 'both'], default='trend',
                        help='Test modu: trend, dip, veya both')
    parser.add_argument('--max-tickers', type=int, default=None,
                        help='Maks hisse sayısı (hızlı test için, ör: 50)')
    parser.add_argument('--period', default='10y',
                        help='Veri periyodu (default: 10y)')
    parser.add_argument('--output', default='.',
                        help='Çıktı dizini')
    parser.add_argument('--elite', action='store_true',
                        help='TREND Momentum modu (RS≥20 + sadece risk-on)')
    return parser.parse_args()


def run(mode='trend', max_tickers=None, period='10y', output_dir='.', elite=False):
    """Ana backtest akışı."""
    t0 = time.time()
    mode_label = f"{mode.upper()}" + (" MOMENTUM" if elite else "")
    print("=" * 60)
    print(f"  NOX BACKTEST — BIST — {mode_label}")
    print(f"  Period: {period} | Max tickers: {max_tickers or 'ALL'}")
    print("=" * 60)

    # 1. Veri çek
    print("\n📡 Veri çekiliyor...")
    tickers = get_all_bist_tickers()
    if max_tickers:
        tickers = tickers[:max_tickers]

    all_data = fetch_data(tickers, period=period)
    xu_df = fetch_benchmark(period=period)

    # USD verisi (BIST için gerekli)
    import yfinance as yf
    try:
        usd_df = yf.download("USDTRY=X", period=period, auto_adjust=True, progress=False)
        if isinstance(usd_df.columns, pd.MultiIndex):
            usd_df.columns = usd_df.columns.droplevel('Ticker')
        print(f"  [DEBUG] USDTRY columns: {usd_df.columns.tolist()}")
        print(f"  USDTRY: {len(usd_df)} gün")
    except:
        usd_df = None
        print("  ⚠️ USDTRY çekilemedi")

    print(f"✅ {len(all_data)} hisse yüklendi ({time.time()-t0:.1f}s)")

    # 2. Backtest çalıştır
    modes_to_run = [mode] if mode != 'both' else ['trend', 'dip']

    for m in modes_to_run:
        print(f"\n{'='*60}")
        print(f"  🔬 {m.upper()} Backtest başlıyor...")
        print(f"{'='*60}")

        all_trades = run_backtest(
            list(all_data.keys()), all_data, xu_df, usd_df,
            regime_mod, dip_mod,
            mode=m, max_tickers=max_tickers, elite=elite,
        )

        if not all_trades:
            print("⚠️ Hiç trade üretilemedi!")
            continue

        # 3. Rapor oluştur
        print(f"\n📊 Rapor oluşturuluyor ({len(all_trades)} trade)...")
        html_path = generate_backtest_html(all_trades, mode=m, output_dir=output_dir, elite=elite)
        csv_path = save_trades_csv(all_trades, output_dir=output_dir, elite=elite)

        # 4. Özet yazdır
        matrix, best = build_matrix(all_trades)
        sig_bk = signal_breakdown(all_trades, 'baseline')

        print(f"\n{'─'*50}")
        print(f"  ÖZET — {m.upper()}")
        print(f"{'─'*50}")
        print(f"  Toplam trade: {len(all_trades)}")
        print(f"  En iyi filtre: {best}")

        print(f"\n  Sinyal performansı (baseline):")
        for sig, met in sorted(sig_bk.items(), key=lambda x: -x[1].get('expectancy', 0)):
            if met['n_trades'] > 0:
                emoji = "🟢" if met['win_rate'] >= 55 else "🟡" if met['win_rate'] >= 45 else "🔴"
                print(f"    {emoji} {sig:12s} N={met['n_trades']:5d}  WR={met['win_rate']:5.1f}%  "
                      f"PF={met['profit_factor']:5.2f}  Exp={met['expectancy']:+.2f}%")

    elapsed = time.time() - t0
    print(f"\n✅ Toplam süre: {elapsed:.1f}s ({elapsed/60:.1f}dk)")
    return None


if __name__ == '__main__':
    args = parse_args()
    run(mode=args.mode, max_tickers=args.max_tickers,
        period=args.period, output_dir=args.output, elite=args.elite)
