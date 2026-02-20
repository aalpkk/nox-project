#!/usr/bin/env python3
"""
NOX Reversal Screener v2 - Runner Script
=========================================
BIST verilerini yükler, kolon isimlerini lowercase'e çevirir,
ReversalScreenerV2 ile makro rejim + swing tarama çalıştırır.

Kullanım:
    python run_reversal.py
    python run_reversal.py --debug
    python run_reversal.py --period 2y
"""

import argparse
import sys
import time

import pandas as pd

from markets.bist import data as data_mod
from markets.bist.reversal_v2 import ReversalScreenerV2


# =============================================================================
# KOLON ADAPTASYONU
# =============================================================================

def _to_lower_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Uppercase -> lowercase kolon donusumu (Open->open, Close->close, ...)"""
    return df.rename(columns={
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume',
    })


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="NOX Reversal Screener v2")
    parser.add_argument('--period', default='1y', help='Veri periyodu (default: 1y)')
    parser.add_argument('--debug', action='store_true', help='Debug modu')
    parser.add_argument('--top', type=int, default=20, help='Swing raporda gosterilecek max hisse (default: 20)')
    args = parser.parse_args()

    print("=" * 75)
    print("  NOX REVERSAL SCREENER v2")
    print("=" * 75)

    # ── 1. Veri Yukleme ──────────────────────────────────────────────────────
    print(f"\n  Ticker listesi aliniyor...")
    tickers = data_mod.get_all_bist_tickers()
    print(f"  {len(tickers)} ticker bulundu.")

    print(f"\n  Veri yukleniyor (period={args.period})...")
    t0 = time.time()
    all_data = data_mod.fetch_data(tickers, period=args.period)
    print(f"  {len(all_data)} hisse yuklendi ({time.time() - t0:.1f}s)")

    if not all_data:
        print("  HATA: Hicbir hisse verisi yuklenemedi!")
        sys.exit(1)

    print(f"  Benchmark (XU100) yukleniyor...")
    xu_df = data_mod.fetch_benchmark(period=args.period)
    if xu_df is None or xu_df.empty:
        print("  HATA: Benchmark verisi yuklenemedi!")
        sys.exit(1)
    print(f"  XU100: {len(xu_df)} bar yuklendi")

    # ── 2. Kolon Donusumu (Uppercase -> lowercase) ───────────────────────────
    xu_df = _to_lower_cols(xu_df)
    stock_dfs = {ticker: _to_lower_cols(df) for ticker, df in all_data.items()}

    if args.debug:
        sample_ticker = next(iter(stock_dfs))
        sample_df = stock_dfs[sample_ticker]
        print(f"\n  [DEBUG] Ornek: {sample_ticker}")
        print(f"  [DEBUG] Kolonlar: {list(sample_df.columns)}")
        print(f"  [DEBUG] Satir sayisi: {len(sample_df)}")
        print(f"  [DEBUG] XU100 kolonlar: {list(xu_df.columns)}")

    # ── 3. Screener Calistir ─────────────────────────────────────────────────
    screener = ReversalScreenerV2()

    # 3a. Makro Rejim
    print(f"\n  MAKRO REJIM ANALIZI...")
    t1 = time.time()
    macro = screener.macro_regime(xu_df, stock_dfs)
    print(f"  Makro analiz tamamlandi ({time.time() - t1:.1f}s)")
    screener.print_macro_report(macro)

    # 3b. Swing Tarama
    print(f"\n\n  SWING TARAMA...")
    t2 = time.time()
    entries = screener.swing_scan(stock_dfs, xu_df, macro)
    print(f"  Swing tarama tamamlandi ({time.time() - t2:.1f}s)")
    screener.print_swing_report(entries, macro, top_n=args.top)

    # ── 4. Ozet ──────────────────────────────────────────────────────────────
    actionable = [e for e in entries if e.state.value != 'NO_SETUP']
    print(f"\n  Toplam taranan: {len(entries)} | Aksiyonlu: {len(actionable)}")
    print(f"  Toplam sure: {time.time() - t0:.1f}s")
    print(f"\n  Reversal Screener v2 tamamlandi.")


if __name__ == '__main__':
    main()
