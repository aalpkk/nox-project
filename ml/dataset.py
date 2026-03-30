"""
ML Dataset Construction — Master Dataset Builder
Tüm BIST hisselerini çeker, feature + target hesaplar, parquet olarak kaydeder.
"""
import os
import time
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

from ml.features import (compute_all_features, compute_targets,
                         compute_breakout_targets, compute_macro_features)


# ═══════════════════════════════════════════
# VERİ ÇEKME
# ═══════════════════════════════════════════

def get_bist_tickers():
    """BIST hisse listesi — TradingView > İsyatırım > statik fallback."""
    import requests
    # Try TradingView first
    try:
        url = "https://scanner.tradingview.com/turkey/scan"
        payload = {
            "columns": ["name"],
            "filter": [
                {"left": "exchange", "operation": "equal", "right": "BIST"},
                {"left": "type", "operation": "equal", "right": "stock"},
                {"left": "is_primary", "operation": "equal", "right": True},
            ],
            "sort": {"sortBy": "name", "sortOrder": "asc"},
            "range": [0, 1000],
        }
        r = requests.post(url, json=payload, headers={
            'User-Agent': 'Mozilla/5.0', 'Content-Type': 'application/json'
        }, timeout=15)
        if r.status_code == 200:
            tickers = [item.get('s', '').split(':')[-1]
                       for item in r.json().get('data', [])
                       if item.get('s')]
            if len(tickers) > 100:
                print(f"  TradingView → {len(tickers)} hisse")
                return tickers
    except Exception:
        pass

    # Fallback: statik liste
    print("  Statik liste kullanılıyor")
    return _static_tickers()


def _static_tickers():
    return [
        "ACSEL","ADEL","AEFES","AFYON","AGESA","AGHOL","AKBNK","AKCNS","AKENR",
        "AKFGY","AKGRT","AKSA","AKSEN","AKSGY","ALARK","ALFAS","ALGYO","ALKIM",
        "ANHYT","ANSGR","ARCLK","ARDYZ","ARENA","ASELS","ASGYO","ASTOR","ASUZU",
        "ATAGY","ATAKP","ATLAS","AYEN","AYGAZ","BAGFS","BERA","BFREN","BIENY",
        "BIGCH","BIMAS","BIOEN","BJKAS","BOBET","BRISA","BRSAN","BRYAT","BTCIM",
        "BUCIM","CCOLA","CIMSA","CWENE","DOAS","DOHOL","ECILC","ECZYT","EGEEN",
        "EKGYO","ENJSA","ENKAI","ERBOS","EREGL","EUPWR","EUREN","FROTO","GARAN",
        "GENIL","GESAN","GLYHO","GOLTS","GSRAY","GUBRF","HEKTS","HLGYO","HUBVC",
        "IMASM","INDES","IPEKE","ISDMR","ISGYO","ISMEN","ISSEN","KARSN","KAYSE",
        "KCHOL","KLSER","KMPUR","KONTR","KONYA","KOZAA","KOZAL","KRDMD","KZBGY",
        "LMKDC","MAVI","MGROS","MIATK","MPARK","ODAS","OTKAR","OYAKC","PAPIL",
        "PEKGY","PETKM","PGSUS","REEDR","SAHOL","SASA","SISE","SKBNK","SOKM",
        "SRVGY","TABGD","TATGD","TAVHL","TCELL","THYAO","TKFEN","TKNSA","TOASO",
        "TRGYO","TTKOM","TTRAK","TUPRS","TURSG","ULKER","VAKBN","VERUS","VESTL",
        "YEOTK","YKBNK","YUNSA","ZOREN",
    ]


def _normalize_df(raw, t, yf_t, yf_syms):
    """yfinance raw output → tek hisse DataFrame."""
    try:
        if len(yf_syms) == 1:
            df = raw.copy()
        else:
            if isinstance(raw.columns, pd.MultiIndex):
                level_0 = raw.columns.get_level_values(0).unique().tolist()
                level_1 = raw.columns.get_level_values(1).unique().tolist()
                price_cols = {'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close'}
                if any(v in price_cols for v in level_0):
                    key = yf_t if yf_t in level_1 else (t if t in level_1 else None)
                    if key is None:
                        return None
                    df = raw.xs(key, level=1, axis=1).copy()
                else:
                    key = yf_t if yf_t in level_0 else (t if t in level_0 else None)
                    if key is None:
                        return None
                    df = raw[key].copy()
            else:
                return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(-1)

        col_map = {}
        for col in df.columns:
            cs = str(col).strip().lower()
            if cs in ('close', 'adj close'):
                col_map[col] = 'Close'
            elif cs == 'open':
                col_map[col] = 'Open'
            elif cs == 'high':
                col_map[col] = 'High'
            elif cs == 'low':
                col_map[col] = 'Low'
            elif cs == 'volume':
                col_map[col] = 'Volume'
        if col_map:
            df = df.rename(columns=col_map)

        df = df.dropna(how='all')
        if not df.empty and len(df) >= 80 and 'Close' in df.columns:
            return df
    except Exception:
        pass
    return None


def _normalize_index_df(xu):
    if isinstance(xu.columns, pd.MultiIndex):
        xu.columns = xu.columns.get_level_values(0)
    col_map = {}
    for col in xu.columns:
        cs = str(col).strip().lower()
        if cs in ('close', 'adj close'):
            col_map[col] = 'Close'
        elif cs == 'open':
            col_map[col] = 'Open'
        elif cs == 'high':
            col_map[col] = 'High'
        elif cs == 'low':
            col_map[col] = 'Low'
        elif cs == 'volume':
            col_map[col] = 'Volume'
    if col_map:
        xu = xu.rename(columns=col_map)
    return xu


def fetch_all_data(tickers, period="2y", batch_size=50):
    """Toplu yfinance download. Returns dict {ticker: DataFrame}."""
    print(f"📡 {len(tickers)} hisse çekiliyor (period={period})...")
    all_data = {}
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        yf_syms = [f"{t}.IS" for t in batch]
        try:
            raw = yf.download(" ".join(yf_syms), period=period,
                              progress=False, auto_adjust=True,
                              group_by='ticker', threads=True)
            if raw.empty:
                continue
            for t, yf_t in zip(batch, yf_syms):
                df = _normalize_df(raw, t, yf_t, yf_syms)
                if df is not None:
                    all_data[t] = df
        except Exception as e:
            print(f"  [!] Batch hata: {e}")
        if i + batch_size < len(tickers):
            time.sleep(1)
        print(f"  {min(i+batch_size, len(tickers))}/{len(tickers)}...", end='\r')
    print(f"\n✅ {len(all_data)}/{len(tickers)} hisse yüklendi")
    return all_data


def fetch_benchmark(period="2y"):
    try:
        xu = yf.download("XU100.IS", period=period, progress=False, auto_adjust=True)
        return _normalize_index_df(xu)
    except Exception as e:
        print(f"  [!] XU100 hata: {e}")
        return None


def fetch_macro_data(period="2y"):
    """VIX, DXY, USDTRY, SPY çek. Returns dict."""
    macro = {}
    symbols = {
        'VIX': '^VIX',
        'DXY': 'DX-Y.NYB',
        'USDTRY': 'USDTRY=X',
        'SPY': 'SPY',
    }
    for name, sym in symbols.items():
        try:
            df = yf.download(sym, period=period, progress=False, auto_adjust=True)
            df = _normalize_index_df(df)
            if not df.empty and len(df) > 30:
                macro[name] = df
                print(f"  ✅ {name}: {len(df)} gün")
        except Exception:
            print(f"  [!] {name} hata")
    return macro


# ═══════════════════════════════════════════
# DATASET OLUŞTURMA
# ═══════════════════════════════════════════

def build_dataset(all_data, xu_df, macro_dfs=None, min_days=80,
                   include_breakout_targets=False):
    """
    Tüm hisseler için feature + target hesapla, tek DataFrame'e birleştir.

    Args:
        include_breakout_targets: True ise tavan/rally target'ları da eklenir.

    Returns:
        DataFrame — MultiIndex (ticker, date), feature + target kolonları
    """
    # Macro features (tüm hisseler için aynı)
    macro_feats = compute_macro_features(xu_df, macro_dfs)

    all_rows = []
    total = len(all_data)
    for idx, (ticker, df) in enumerate(all_data.items()):
        if len(df) < min_days:
            continue
        try:
            feats = compute_all_features(df, xu_df=xu_df)
            targets = compute_targets(df)

            if feats.empty:
                continue

            combined = feats.join(targets)

            # Breakout target'ları (opsiyonel)
            if include_breakout_targets:
                brk_targets = compute_breakout_targets(df)
                combined = combined.join(brk_targets)

            # Macro features join
            if not macro_feats.empty:
                combined = combined.join(macro_feats, how='left')

            combined['ticker'] = ticker
            all_rows.append(combined)
        except Exception as e:
            print(f"  [!] {ticker}: {e}")

        if (idx + 1) % 50 == 0:
            print(f"  Feature hesaplama: {idx+1}/{total}...", end='\r')

    if not all_rows:
        print("❌ Hiç veri yok!")
        return pd.DataFrame()

    print(f"\n  Birleştiriliyor ({len(all_rows)} hisse)...")
    result = pd.concat(all_rows, ignore_index=False)
    result = result.set_index('ticker', append=True).swaplevel()
    result.index.names = ['ticker', 'date']
    result = result.sort_index()

    # Auto-cleanup: drop columns that are 100% NaN
    _target_cols = _ALL_TARGET_COLS
    feat_cols = [c for c in result.columns if c not in _target_cols]
    nan_pct = result[feat_cols].isna().mean()
    all_nan_cols = nan_pct[nan_pct >= 0.99].index.tolist()
    if all_nan_cols:
        print(f"  ⚠️ {len(all_nan_cols)} kolon >99% NaN — kaldırılıyor: {all_nan_cols}")
        result = result.drop(columns=all_nan_cols)

    print(f"✅ Dataset: {len(result)} satır, {len(result.columns)} kolon")
    return result


_ALL_TARGET_COLS = {
    'ret_1g', 'ret_3g', 'up_1g', 'up_3g',
    'tavan_1d', 'tavan_3d', 'tavan_5d', 'tavan_series',
    'rally_3d', 'rally_5d', 'rally_any',
}


# ═══════════════════════════════════════════
# WALK-FORWARD SPLIT
# ═══════════════════════════════════════════

def walk_forward_split(df, train_months=12, val_months=6, step_months=6):
    """
    Zaman bazlı walk-forward split.

    Yields:
        (fold_num, train_idx, val_idx) tuples
    """
    dates = df.index.get_level_values('date')
    min_date = dates.min()
    max_date = dates.max()

    train_start = min_date
    fold = 0
    while True:
        train_end = train_start + pd.DateOffset(months=train_months)
        val_end = train_end + pd.DateOffset(months=val_months)

        if train_end > max_date:
            break

        train_mask = (dates >= train_start) & (dates < train_end)
        val_mask = (dates >= train_end) & (dates < val_end)

        if train_mask.sum() > 0 and val_mask.sum() > 0:
            yield fold, train_mask, val_mask
            fold += 1

        train_start = train_start + pd.DateOffset(months=step_months)

        if val_end > max_date:
            break


def time_split(df, train_end='2025-06-30', val_end='2025-12-31'):
    """
    Basit zaman bazlı train/val/test split.

    Returns:
        (train_mask, val_mask, test_mask)
    """
    dates = df.index.get_level_values('date')
    train_mask = dates <= train_end
    val_mask = (dates > train_end) & (dates <= val_end)
    test_mask = dates > val_end
    return train_mask, val_mask, test_mask


def get_feature_columns(df):
    """Feature kolon listesi (target ve meta hariç)."""
    exclude = _ALL_TARGET_COLS | {'close'}
    return [c for c in df.columns if c not in exclude]


def dataset_summary(df):
    """Dataset özet istatistikleri yazdır."""
    dates = df.index.get_level_values('date')
    tickers = df.index.get_level_values('ticker')
    print(f"\n{'='*60}")
    print(f"DATASET ÖZETİ")
    print(f"{'='*60}")
    print(f"  Satır:        {len(df):,}")
    print(f"  Hisse:        {tickers.nunique()}")
    print(f"  Tarih Aralığı:{dates.min().strftime('%Y-%m-%d')} → {dates.max().strftime('%Y-%m-%d')}")
    feat_cols = get_feature_columns(df)
    print(f"  Feature:      {len(feat_cols)}")

    # Feature availability audit by category
    categories = {
        'Fiyat': ['close', 'returns_1d', 'returns_5d', 'returns_10d',
                   'close_position', 'gap_pct', 'ema21_dist_pct', 'ema55_dist_pct'],
        'Trend': ['adx_14', 'adx_slope_5', 'plus_di', 'minus_di', 'di_spread',
                   'ema_trend_up', 'supertrend_dir', 'pmax_dir', 'phase_above_ema21',
                   'htf_adx', 'htf_adx_slope', 'htf_trend_up'],
        'Momentum': ['rsi_14', 'rsi_2', 'macd_line', 'macd_signal', 'macd_hist',
                      'wt1', 'wt2', 'wt_bullish', 'squeeze_on', 'squeeze_mom'],
        'Volatilite': ['atr_14', 'atr_pct', 'bb_pctb', 'bb_width',
                        'drawdown_20', 'daily_move_atr'],
        'Hacim': ['vol_ratio_20', 'vol_ratio_30', 'cmf_20', 'obv_trend',
                   'obv_slope_5', 'mfi_14', 'consecutive_green'],
        'SMC/Yapı': ['swing_bias', 'bos_age', 'choch_age', 'structure_break',
                      'higher_low', 'near_40high'],
        'Q Score': ['q_rvol_s', 'q_clv_s', 'q_wick_s', 'q_range_s', 'q_total'],
        'NW Breadth': ['br_rsi_thrust', 'br_rsi_gradual', 'br_ad_proxy',
                        'br_ema_reclaim', 'br_score'],
        'NW Regime': ['rg_slope_score', 'rg_di_score', 'rg_ema_above',
                       'rg_adx_rebound', 'rg_was_trending', 'rg_score', 'gate_open'],
        'Sell Severity': ['red_count', 'drawdown_20_pct', 'decline_5d_atr',
                           'sell_severity', 'pivot_delta_pct'],
        'RT TPE': ['rt_ema_bull', 'rt_st_bull', 'rt_wk_trend_up',
                    'rt_cmf_pos', 'rt_rvol_high', 'rt_obv_slope_pos',
                    'rt_adx_slope_pos', 'rt_atr_expanding', 'rt_di_bull',
                    'trend_score', 'participation_score', 'expansion_score', 'tpe_total'],
        'RT Meta': ['regime_score', 'entry_score', 'oe_score', 'pb_ema_dist',
                     'pb_rsi_low', 'exit_stage', 'days_in_trade'],
        'Tavan': ['is_tavan', 'tavan_streak', 'close_to_high', 'tavan_locked',
                   'hit_tavan_intraday', 'recent_tavan_10d'],
        'RS': ['rs_10', 'rs_60', 'rs_composite'],
        'Pre-Breakout': [
            'range_contraction_5_20', 'range_contraction_5_40',
            'atr_contraction_5_20', 'bb_width_pctile_20',
            'vol_dryup_ratio', 'vol_surge_today', 'vol_acceleration',
            'tl_volume_20d_avg', 'vol_pattern_score',
            'rsi_momentum_5d', 'macd_hist_accel',
            'consecutive_higher_close', 'close_vs_20d_high_pct',
            'dist_to_52w_high_pct', 'dist_to_20d_high_pct',
            'near_52w_high', 'price_range_position_20',
            'near_tavan_miss', 'recent_near_tavan_5d', 'max_daily_ret_5d',
            'rs_acceleration', 'market_tavan_count_10d',
        ],
        'Makro': ['xu100_above_ema21', 'xu100_ret_5d', 'vix', 'vix_chg_5d',
                   'dxy_trend', 'usdtry_chg_1d', 'spy_trend', 'macro_risk_score'],
    }

    print(f"\n  Feature Audit:")
    print(f"  {'Kategori':<16} {'N':>3} {'Mevcut':>6} {'Ort NaN%':>9}")
    print(f"  {'-'*38}")
    for cat, cols in categories.items():
        present = [c for c in cols if c in df.columns]
        if present:
            avg_nan = df[present].isna().mean().mean()
            print(f"  {cat:<16} {len(present):>3} / {len(cols):<3}  {avg_nan:>7.1%}")

    # High NaN features
    nan_pct = df[feat_cols].isna().mean().sort_values(ascending=False)
    high_nan = nan_pct[nan_pct > 0.3]
    if len(high_nan) > 0:
        print(f"\n  ⚠️ Yüksek NaN (>30%):")
        for col, pct in high_nan.items():
            print(f"    {col}: {pct:.1%}")

    # Target dağılımı
    all_targets = ['up_1g', 'up_3g',
                   'tavan_1d', 'tavan_3d', 'tavan_5d', 'tavan_series',
                   'rally_3d', 'rally_5d', 'rally_any']
    for target in all_targets:
        if target in df.columns:
            valid = df[target].dropna()
            print(f"\n  {target}: N={len(valid):,}, pozitif={valid.mean():.1%}")

    print(f"{'='*60}\n")
