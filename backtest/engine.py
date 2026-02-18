"""
NOX Backtest — Engine (Optimized)
Göstergeleri BİR KEZ hesapla, tüm tarihleri tara.
Orijinal rolling-window yaklaşımına göre ~100x hızlı.
"""
import sys, os
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time
import traceback

from backtest.config import (
    REGIMES, REGIME_GROUPS, FILTER_TESTS, SIGNAL_GROUPS,
    ENTRY_METHOD, MAX_HOLD_DAYS, COMMISSION_PCT, SLIPPAGE_PCT,
)
from core.config import (
    ADX_LEN, ADX_TREND, ADX_CHOPPY, ADX_SLOPE_LEN, ADX_SLOPE_THRESH,
    EMA_FAST, EMA_SLOW, ST_LEN, ST_MULT,
    BOS_LOOKBACK, BOS_TIGHT, CHOCH_TIGHT,
    SQ_LEN, SQ_MULT_BB, SQ_MULT_KC,
    BB_LEN, BB_MULT, DONCH_LEN, MR_RSI_LEN, MR_RSI_THRESH,
    RS_LEN1, RS_LEN2, ATR_LEN,
    OVEREXT_WT1_THRESH, OVEREXT_RSI_THRESH, OVEREXT_MOMENTUM_PCT,
    OVEREXT_MOMENTUM_DAYS,
)
from markets.bist.config import (
    RVOL_THRESH, TREND_STOP, GRI_STOP, DONUS_STOP, COMBO_STOP,
    TREND_TP, GRI_TP, DONUS_TP, COMBO_TP, TRAIL_MULT,
    QUAL_MIN_GRI, QUAL_MIN_TREND, RS_THRESHOLD, MIN_AVG_VOLUME_TL,
    PINK_STOP_MULT, PINK_EMA89, PINK_EMA144, PINK_RSI_LEN,
    PINK_RSI_DIV_LOOKBACK, PINK_TOUCH_WINDOW, PINK_TOUCH_COUNT,
    ATR_PANIC_PCTILE, ATR_PCTILE_WINDOW,
    CORE_Q_MIN_PARTIAL_STRONG, CORE_Q_MIN_EARLY, CORE_Q_MIN_COMBO,
    COMBO_RS_MAX, COMBO_OE_MAX, COMBO_DIST_EMA_MAX,
    POS_SIZE_CORE, POS_SIZE_MOMENTUM,
    MOMENTUM_RS_THRESH,
)
from core.indicators import (
    ema, sma, rma, true_range, calc_atr, calc_adx, calc_supertrend,
    calc_wavetrend, calc_pmax, calc_smc, calc_cmf,
    calc_rsi, resample_weekly, calc_overextended,
    calc_atr_sma, calc_adx_ema, calc_rsi_sma, calc_macd, to_usd,
)


# ══════════════════════════════════════════════════════════════
# Yardımcı fonksiyonlar (değişmedi)
# ══════════════════════════════════════════════════════════════

def get_regime_for_date(dt):
    """Tarih hangi rejim dönemine düşüyor."""
    dt_str = dt.strftime('%Y-%m-%d') if hasattr(dt, 'strftime') else str(dt)[:10]
    for name, cfg in REGIMES.items():
        if cfg['start'] <= dt_str < cfg['end']:
            return name, cfg['type']
    return 'unknown', 'unknown'


def simulate_trade(df, entry_idx, stop_price, tp_price, direction='long',
                    trail_mult=None, atr_val=None):
    """
    Tek trade simülasyonu — trailing stop destekli.
    Entry → her gün trailing stop güncelle → stop/tp kontrolü → exit veya timeout.
    trail_mult ve atr_val verilirse, iz süren stop aktif olur:
      highest_close'dan trail_mult * atr_val mesafe ile stop yukarı çekilir.
    """
    n = len(df)
    if entry_idx >= n - 1:
        return None

    if ENTRY_METHOD == "next_open" and entry_idx + 1 < n:
        entry_price = float(df['Open'].iloc[entry_idx + 1])
        start_idx = entry_idx + 1
    else:
        entry_price = float(df['Close'].iloc[entry_idx])
        start_idx = entry_idx + 1

    if entry_price <= 0 or pd.isna(entry_price):
        return None

    # Entry validation: gap-up entry > TP veya gap-down entry < stop → geçersiz trade
    if entry_price >= tp_price or entry_price <= stop_price:
        return None

    entry_price *= (1 + SLIPPAGE_PCT)
    mae = 0.0
    mfe = 0.0
    current_stop = stop_price
    highest_close = entry_price
    use_trail = trail_mult is not None and atr_val is not None and atr_val > 0

    for i in range(start_idx, min(start_idx + MAX_HOLD_DAYS, n)):
        low = float(df['Low'].iloc[i])
        high = float(df['High'].iloc[i])
        close = float(df['Close'].iloc[i])

        if pd.isna(low) or pd.isna(high) or pd.isna(close):
            continue

        current_loss = (low / entry_price - 1) * 100
        current_gain = (high / entry_price - 1) * 100
        mae = min(mae, current_loss)
        mfe = max(mfe, current_gain)

        # Stop kontrolü (trailing stop dahil)
        if low <= current_stop:
            exit_price = current_stop * (1 - SLIPPAGE_PCT)
            pnl = (exit_price / entry_price - 1) * 100 - COMMISSION_PCT * 200
            pnl = max(pnl, -25.0)  # realistic fill cap
            exit_reason = 'TRAIL_STOP' if current_stop > stop_price else 'STOP'
            return {
                'exit_price': exit_price, 'exit_date': df.index[i],
                'exit_reason': exit_reason, 'pnl_pct': round(pnl, 2),
                'hold_days': i - start_idx + 1,
                'mae': round(mae, 2), 'mfe': round(mfe, 2),
            }

        # TP kontrolü
        if high >= tp_price:
            exit_price = tp_price * (1 - SLIPPAGE_PCT)
            pnl = (exit_price / entry_price - 1) * 100 - COMMISSION_PCT * 200
            return {
                'exit_price': exit_price, 'exit_date': df.index[i],
                'exit_reason': 'TP', 'pnl_pct': round(pnl, 2),
                'hold_days': i - start_idx + 1,
                'mae': round(mae, 2), 'mfe': round(mfe, 2),
            }

        # Trailing stop güncelle (gün kapanışından sonra)
        if use_trail and close > highest_close:
            highest_close = close
            new_stop = highest_close - trail_mult * atr_val
            if new_stop > current_stop:
                current_stop = new_stop

    last_idx = min(start_idx + MAX_HOLD_DAYS - 1, n - 1)
    exit_price = float(df['Close'].iloc[last_idx]) * (1 - SLIPPAGE_PCT)
    pnl = (exit_price / entry_price - 1) * 100 - COMMISSION_PCT * 200
    # Realistic fill cap: max loss -25% (gap/likidite riski)
    pnl = max(pnl, -25.0)
    return {
        'exit_price': exit_price, 'exit_date': df.index[last_idx],
        'exit_reason': 'TIMEOUT', 'pnl_pct': round(pnl, 2),
        'hold_days': last_idx - start_idx + 1,
        'mae': round(mae, 2), 'mfe': round(mfe, 2),
    }


def apply_filter(signal, filter_cfg):
    """Sinyal filtre kriterlerini geçiyor mu."""
    q = signal.get('quality', 0)
    rs = signal.get('rs_score', 0)
    rr = signal.get('rr', 0)
    oe = signal.get('overext_score', 0)

    if q < filter_cfg['quality']:
        return False
    if rs < filter_cfg['rs']:
        return False
    if rr < filter_cfg['rr']:
        return False
    if oe > filter_cfg['oe_max']:
        return False
    return True


# ══════════════════════════════════════════════════════════════
# VECTORIZED TREND SIGNAL GENERATION
# Göstergeleri 1 kez hesapla → tüm tarihleri tara
# ══════════════════════════════════════════════════════════════

def _fast_trend_signals(ticker, df, xu_df, usd_df, debug=False, elite=False):
    """
    Vectorized trend sinyal üretimi — Production Ruleset v1.
    Tüm göstergeleri BİR KEZ hesaplar, sonra her 5 günde bir tarar.
    Otomatik mod seçimi: risk-on + RS≥20 → MOMENTUM, else CORE.
    elite=True → sadece MOMENTUM sinyallerini döndür.
    elite=False → sadece CORE sinyallerini döndür.
    """
    n = len(df)
    min_days = 200
    if n < min_days:
        return []

    c = df['Close']
    h = df['High']
    l = df['Low']
    o = df['Open']
    v = df['Volume']

    # ── PRECOMPUTE: Pahalı göstergeler (1 kez) ──
    atr_s = calc_atr(df, ATR_LEN)
    vol_sma20 = sma(v, 20)
    avg_turnover = vol_sma20 * c

    ema_f = ema(c, EMA_FAST)
    ema_s_val = ema(c, EMA_SLOW)
    ema_trend_up = ema_f > ema_s_val

    st_dir = calc_supertrend(df, ST_LEN, ST_MULT)
    super_trend_up = (st_dir == 1)

    adx_full = calc_adx(df, ADX_LEN)
    adx_slope_s = (adx_full - adx_full.shift(ADX_SLOPE_LEN)) / ADX_SLOPE_LEN
    adx_rising_s = adx_slope_s > ADX_SLOPE_THRESH

    # Weekly (HTF)
    wdf = resample_weekly(df)
    if len(wdf) >= 20:
        htf_adx_w = calc_adx(wdf, ADX_LEN)
        htf_adx_slope_w = (htf_adx_w - htf_adx_w.shift(ADX_SLOPE_LEN)) / ADX_SLOPE_LEN
        htf_ema_f_w = ema(wdf['Close'], EMA_FAST)
        htf_ema_s_w = ema(wdf['Close'], EMA_SLOW)
        htf_adx_d = htf_adx_w.reindex(df.index, method='ffill').fillna(0)
        htf_slope_d = htf_adx_slope_w.reindex(df.index, method='ffill').fillna(0)
        htf_trend_up_d = (htf_ema_f_w > htf_ema_s_w).reindex(df.index, method='ffill').fillna(False)
    else:
        htf_adx_d = pd.Series(0.0, index=df.index)
        htf_slope_d = pd.Series(0.0, index=df.index)
        htf_trend_up_d = pd.Series(False, index=df.index)

    htf_rising_d = htf_slope_d > ADX_SLOPE_THRESH

    # ── Weekly SuperTrend (multi-timeframe doğrulama) ──
    if len(wdf) >= 20:
        wdf_st = calc_supertrend(wdf, ST_LEN, ST_MULT)
        weekly_st_up_d = (wdf_st == 1).reindex(df.index, method='ffill').fillna(False)
    else:
        weekly_st_up_d = pd.Series(False, index=df.index)

    # ── XU100 Dinamik Market Rejimi (EMA50 + slope) ──
    #   risk_on  (2): XU100 > EMA50 ve EMA50 slope > 0 → tüm sinyaller
    #   grey     (1): arada → sadece en iyi sinyaller (STRONG, COMBO+, EARLY)
    #   risk_off (0): XU100 < EMA50 ve EMA50 slope < 0 → trade yok
    if xu_df is not None and len(xu_df) >= 60:
        xu_close = xu_df['Close'].reindex(df.index, method='ffill')
        xu_ema50 = ema(xu_df['Close'], 50).reindex(df.index, method='ffill')
        xu_ema50_slope = (xu_ema50 - xu_ema50.shift(10)) / xu_ema50.shift(10) * 100  # 10-gün % değişim
        xu_above = xu_close > xu_ema50
        xu_slope_pos = xu_ema50_slope > 0
        xu_slope_neg = xu_ema50_slope < 0
        market_regime_s = pd.Series(np.select(
            [xu_above & xu_slope_pos, (~xu_above) & xu_slope_neg],
            [2, 0], default=1
        ), index=df.index)
    else:
        market_regime_s = pd.Series(2, index=df.index)  # default risk-on

    # ── CMF (Chaikin Money Flow) — birikim/dağıtım filtresi ──
    cmf_s = calc_cmf(df, 20)

    # ── ATR Percentile (Panic Filter) ──
    atr_pctile_s = atr_s.rolling(ATR_PCTILE_WINDOW, min_periods=max(ATR_PCTILE_WINDOW // 2, 1)).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    ).fillna(0.5)

    # ── Overextension Filter (distance-to-EMA / ATR) ──
    # Fiyat EMA20'den > 2 ATR uzaksa → koşmuş hisse, girme
    ema20 = ema(c, 20)
    dist_to_ema20 = (c - ema20) / atr_s.replace(0, np.nan)
    dist_to_ema20 = dist_to_ema20.fillna(0)

    # Regime
    trend_up_count = ema_trend_up.astype(int) + super_trend_up.astype(int) + htf_trend_up_d.astype(int)
    confirmed_trend_up = (trend_up_count >= 2) & (c > ema_s_val)

    htf_r = pd.Series(np.select(
        [(htf_adx_d > ADX_TREND) & htf_rising_d, htf_adx_d > ADX_TREND, htf_adx_d > ADX_CHOPPY],
        [2, 1, 0], default=-1
    ), index=df.index)
    daily_confirm = (adx_full > ADX_CHOPPY) & adx_rising_s

    regime_s = pd.Series(np.select(
        [~confirmed_trend_up, confirmed_trend_up & (htf_r == 2) & daily_confirm,
         confirmed_trend_up & (htf_r >= 1), confirmed_trend_up & (htf_r == 0)],
        [0, 3, 2, 1], default=0
    ), index=df.index)

    # WaveTrend
    wt = calc_wavetrend(df)
    wt_cross_up_s = wt['cross_up']
    wt_recent_s = wt['wt_recent']
    wt_bullish_s = wt['wt_bullish']

    # PMAX
    pm = calc_pmax(df)
    pmax_long_s = pd.Series(pm['pmax_long'], index=df.index)

    # SMC
    smc = calc_smc(df)
    idx_arr = np.arange(n)
    bos_bar_arr = smc['bos_bar']
    choch_bar_arr = smc['choch_bar']
    sb_arr = smc['swing_bias_arr']
    bos_age = idx_arr - bos_bar_arr
    choch_age = idx_arr - choch_bar_arr

    recent_bos_s = pd.Series((bos_age <= BOS_LOOKBACK) & (sb_arr == 1), index=df.index)
    recent_choch_s = pd.Series((choch_age <= BOS_LOOKBACK) & (sb_arr == 1), index=df.index)
    bos_tight_s = pd.Series((bos_age <= BOS_TIGHT) & (sb_arr == 1), index=df.index)
    choch_tight_s = pd.Series((choch_age <= CHOCH_TIGHT) & (sb_arr == 1), index=df.index)

    # Squeeze
    sq_basis = sma(c, SQ_LEN)
    sq_dev = c.rolling(SQ_LEN).std() * SQ_MULT_BB
    sq_rng = sma(true_range(df), SQ_LEN)
    sqz_on = ((sq_basis - sq_dev) > (sq_basis - SQ_MULT_KC * sq_rng)) & \
             ((sq_basis + sq_dev) < (sq_basis + SQ_MULT_KC * sq_rng))
    hh = h.rolling(SQ_LEN).max()
    ll = l.rolling(SQ_LEN).min()
    sq_mid = (hh + ll) / 2
    sq_mom = (c - (sq_mid + sq_basis) / 2).rolling(SQ_LEN).mean()
    sq_release = (~sqz_on) & sqz_on.shift(1) & (sq_mom > 0) & (sq_mom > sq_mom.shift(1))
    sq_release_recent = sq_release.rolling(3, min_periods=1).max().fillna(0).astype(bool)
    atr_ma = sma(atr_s, 20)
    atr_expanding_s = (atr_s > atr_ma * 1.05).rolling(3, min_periods=1).max().fillna(0).astype(bool)
    sq_mom_rising = (sq_mom > 0) & (sq_mom > sq_mom.shift(1)) & (~sqz_on)
    trend_sq_s = sq_release_recent | sq_mom_rising

    # BB / MR
    bb_basis_s = sma(c, BB_LEN)
    bb_dev_val = c.rolling(BB_LEN).std() * BB_MULT
    bb_upper_s = bb_basis_s + bb_dev_val
    bb_lower_s = bb_basis_s - bb_dev_val
    bb_pctb_s = (c - bb_lower_s) / (bb_upper_s - bb_lower_s)
    donch_lower_s = l.rolling(DONCH_LEN).min()
    rsi_short = calc_rsi(c, MR_RSI_LEN)

    # Quality (vectorized)
    rvol_s = v / vol_sma20.replace(0, np.nan)
    candle_range_s = h - l
    clv_s = ((c - l) / candle_range_s.replace(0, np.nan)).fillna(0.5)
    upper_wick_s = h - pd.concat([c, o], axis=1).max(axis=1)
    wick_ratio_s = (upper_wick_s / candle_range_s.replace(0, np.nan)).fillna(0)
    range_atr_s = (candle_range_s / atr_s.replace(0, np.nan)).fillna(0)

    rvol_score = pd.Series(np.select(
        [rvol_s >= 2, rvol_s >= RVOL_THRESH, rvol_s >= 1],
        [25, 20, 10], default=0
    ), index=df.index)
    clv_score = pd.Series(np.select(
        [clv_s >= 0.75, clv_s >= 0.5, clv_s >= 0.25],
        [25, 15, 5], default=0
    ), index=df.index)
    wick_score = pd.Series(np.select(
        [wick_ratio_s <= 0.15, wick_ratio_s <= 0.3, wick_ratio_s <= 0.5],
        [25, 15, 5], default=0
    ), index=df.index)
    range_score = pd.Series(np.select(
        [range_atr_s >= 1.2, range_atr_s >= 0.8, range_atr_s >= 0.5],
        [25, 15, 5], default=0
    ), index=df.index)
    quality_s = rvol_score + clv_score + wick_score + range_score

    # RS score (vectorized)
    rs_score_s = pd.Series(0.0, index=df.index)
    if xu_df is not None and len(xu_df) >= RS_LEN2 + 5:
        xu_aligned = xu_df['Close'].reindex(df.index, method='ffill')
        stock_ret1 = (c / c.shift(RS_LEN1) - 1) * 100
        stock_ret2 = (c / c.shift(RS_LEN2) - 1) * 100
        bench_ret1 = (xu_aligned / xu_aligned.shift(RS_LEN1) - 1) * 100
        bench_ret2 = (xu_aligned / xu_aligned.shift(RS_LEN2) - 1) * 100
        rs_score_s = ((stock_ret1 - bench_ret1) * 0.6 + (stock_ret2 - bench_ret2) * 0.4).fillna(0)
    rs_pass_s = rs_score_s > RS_THRESHOLD

    # Overextended (vectorized)
    rsi14 = calc_rsi(c, 14)
    oe_wt1_high = (wt['wt1'] > OVEREXT_WT1_THRESH).astype(int)
    oe_wt1_decline = (wt['wt1'] < wt['wt1'].shift(1)).astype(int)
    oe_rsi_high = (rsi14 > OVEREXT_RSI_THRESH).fillna(0).astype(int)
    oe_above_bb = (c > bb_upper_s).fillna(0).astype(int)
    oe_mom = ((c - c.shift(OVEREXT_MOMENTUM_DAYS)) / c.shift(OVEREXT_MOMENTUM_DAYS) * 100 > OVEREXT_MOMENTUM_PCT).fillna(0).astype(int)
    overext_score_s = oe_wt1_high + oe_wt1_decline + oe_rsi_high + oe_above_bb + oe_mom

    # ── PRECOMPUTE: Signal condition bileşenleri ──
    vol_high_s = v > vol_sma20 * RVOL_THRESH
    q_pass_gri_s = quality_s >= QUAL_MIN_GRI
    q_pass_trend_s = quality_s >= QUAL_MIN_TREND

    # Pullback
    pb_rsi = calc_rsi(c, 5)
    pb_dipped_s = ((pb_rsi < 40) & (pb_rsi > 20)).rolling(5, min_periods=1).max().fillna(0).astype(bool)
    pb_vol_dry_s = (v < vol_sma20 * 0.8).rolling(5, min_periods=1).max().fillna(0).astype(bool)
    ema_cross_up = (c > ema_f) & (c.shift(1) <= ema_f.shift(1))
    pb_reclaim_s = ema_cross_up.rolling(3, min_periods=1).max().fillna(0).astype(bool)

    # Reversal (tightened: ema55 dist narrowed to -2/+2)
    ema55_cross = (c > ema_s_val) & (c.shift(1) <= ema_s_val.shift(1))
    recent_e55_s = ema55_cross.rolling(10, min_periods=1).max().fillna(0).astype(bool)
    recent_wt_cross_s = wt_cross_up_s.rolling(10, min_periods=1).max().fillna(0).astype(bool)
    ema55_dist_s = (c - ema_s_val) / ema_s_val * 100
    approaching_s = (ema55_dist_s > -2) & (ema55_dist_s < 2) & (c > c.shift(3)) & (c.shift(3) > c.shift(6))

    # Early
    sw_hl = h.rolling(20).max().shift(1)
    struct_break_s = c > sw_hl
    mom_up5_s = (c - c.shift(5)) / c.shift(5) * 100
    green_cnt_s = (c > o).astype(int).rolling(5, min_periods=1).sum()
    adx_turn_s = (adx_full > adx_full.shift(1)) & (adx_full.shift(1) > adx_full.shift(2))
    early_rsi_s = calc_rsi(c, 14)
    highest5_s = c.rolling(5).max()
    early_vol_high = v > vol_sma20 * 1.2

    # ── SCAN: Her 5 günde 1 sinyal kontrol ──
    signals = []
    step = 5

    # Numpy arrays for fast indexing
    c_arr = c.values
    atr_arr = atr_s.values
    vol_sma20_arr = vol_sma20.values
    avg_turn_arr = avg_turnover.values
    regime_arr = regime_s.values
    ema_trend_arr = ema_trend_up.values
    st_up_arr = super_trend_up.values
    adx_arr = adx_full.values
    adx_slope_arr = adx_slope_s.values
    htf_adx_arr = htf_adx_d.values
    confirmed_arr = confirmed_trend_up.values
    wt_cross_arr = wt_cross_up_s.values
    wt_recent_arr = wt_recent_s.values
    wt_bull_arr = wt_bullish_s.values
    pmax_arr = pmax_long_s.values
    vol_high_arr = vol_high_s.values
    rs_pass_arr = rs_pass_s.values
    rs_score_arr = rs_score_s.values
    q_pass_gri_arr = q_pass_gri_s.values
    q_pass_trend_arr = q_pass_trend_s.values
    quality_arr = quality_s.values
    trend_sq_arr = trend_sq_s.values
    sq_release_arr = sq_release_recent.values
    atr_exp_arr = atr_expanding_s.values
    sq_basis_arr = sq_basis.values
    clv_arr = clv_s.values
    bb_lower_arr = bb_lower_s.values
    bb_basis_arr = bb_basis_s.values
    bb_pctb_arr = bb_pctb_s.values
    donch_arr = donch_lower_s.values
    rsi_short_arr = rsi_short.values
    overext_arr = overext_score_s.values
    rvol_arr = rvol_s.values
    ema_s_arr = ema_s_val.values
    bos_tight_arr = bos_tight_s.values
    choch_tight_arr = choch_tight_s.values
    recent_bos_arr = recent_bos_s.values
    recent_choch_arr = recent_choch_s.values
    pb_dipped_arr = pb_dipped_s.values
    pb_vol_dry_arr = pb_vol_dry_s.values
    pb_reclaim_arr = pb_reclaim_s.values
    recent_e55_arr = recent_e55_s.values
    recent_wt_cross_arr = recent_wt_cross_s.values
    ema55_dist_arr = ema55_dist_s.values
    approaching_arr = approaching_s.values
    struct_break_arr = struct_break_s.values
    mom_up5_arr = mom_up5_s.values
    green_cnt_arr = green_cnt_s.values
    adx_turn_arr = adx_turn_s.values
    early_rsi_arr = early_rsi_s.values
    highest5_arr = highest5_s.values
    early_vol_arr = early_vol_high.values
    wt1_arr = wt['wt1'].values
    wt2_arr = wt['wt2'].values

    # Yeni indikatör array'leri
    market_regime_arr = market_regime_s.values   # 0=risk-off, 1=grey, 2=risk-on
    weekly_st_up_arr = weekly_st_up_d.values     # haftalık SuperTrend UP
    cmf_arr = cmf_s.values                       # CMF: >0 birikim, <0 dağıtım
    atr_pctile_arr = atr_pctile_s.values         # ATR percentile (panic filter)
    dist_ema20_arr = dist_to_ema20.values        # distance-to-EMA20 / ATR

    # risk-on'da izin verilen tüm sinyaller, grey'de sadece en iyiler
    GREY_ALLOWED = {"STRONG", "COMBO+", "EARLY", "PARTIAL"}

    for day_idx in range(min_days, n, step):
        # Temel filtreler
        atr_val = atr_arr[day_idx]
        if np.isnan(atr_val) or atr_val == 0:
            continue
        if avg_turn_arr[day_idx] < MIN_AVG_VOLUME_TL:
            continue

        # ── SIDEWAYS hard-block (backtest rejim dönemleri) ──
        signal_date = df.index[day_idx]
        _rname, _rtype = get_regime_for_date(signal_date)
        if _rtype == 'sideways':
            continue

        # ── Dinamik XU100 market rejim filtresi ──
        mkt_regime = int(market_regime_arr[day_idx])
        if mkt_regime == 0:  # risk-off → trade yok
            continue

        # ── Panic filter: ATR percentile ≥ threshold → crash/panik, trade açma ──
        if float(atr_pctile_arr[day_idx]) >= ATR_PANIC_PCTILE:
            continue

        close_price = c_arr[day_idx]
        regime = int(regime_arr[day_idx])
        vol_high = bool(vol_high_arr[day_idx])
        rs_pass = bool(rs_pass_arr[day_idx])
        q_pass_gri = bool(q_pass_gri_arr[day_idx])
        q_pass_trend = bool(q_pass_trend_arr[day_idx])
        _ema_trend_up = bool(ema_trend_arr[day_idx])
        _super_trend_up = bool(st_up_arr[day_idx])
        _trend_sq = bool(trend_sq_arr[day_idx])
        _confirmed = bool(confirmed_arr[day_idx])

        # ── Signal detection ──

        # COMBO
        combo_base = (bool(wt_cross_arr[day_idx]) or bool(wt_recent_arr[day_idx])) and \
                     bool(wt_bull_arr[day_idx]) and bool(pmax_arr[day_idx])
        combo_plus = combo_base and bool(choch_tight_arr[day_idx])
        combo_bos = combo_base and bool(bos_tight_arr[day_idx]) and not bool(choch_tight_arr[day_idx])

        # STRONG / WEAK (RS filtresi kaldırıldı — "koşmuş hisse" tuzağı)
        strong = regime >= 2 and _ema_trend_up and _super_trend_up and _trend_sq and vol_high and q_pass_trend
        weak = regime >= 2 and _ema_trend_up and _super_trend_up and _trend_sq and (not vol_high) and q_pass_trend

        # EARLY
        _struct = bool(struct_break_arr[day_idx]) and bool(early_vol_arr[day_idx]) and bool(adx_turn_arr[day_idx])
        _mom = float(mom_up5_arr[day_idx]) > 5.0 and close_price >= float(highest5_arr[day_idx]) and \
               int(green_cnt_arr[day_idx]) >= 3 and bool(early_vol_arr[day_idx])
        _early_rsi = float(early_rsi_arr[day_idx])
        early = regime <= 1 and (_struct or _mom) and _early_rsi < 75

        # ── Precompute: ortak değerler ──
        _oe = int(overext_arr[day_idx])
        _dist_ema = float(dist_ema20_arr[day_idx])
        _rs = float(rs_score_arr[day_idx])
        _cmf = float(cmf_arr[day_idx])
        _quality = int(quality_arr[day_idx])
        _weekly_st = bool(weekly_st_up_arr[day_idx])

        # ── Sinyal önceliklendirme ──
        signal = None
        if combo_bos and _oe <= COMBO_OE_MAX and _rs < COMBO_RS_MAX and _dist_ema <= COMBO_DIST_EMA_MAX:
            signal = "COMBO"
        elif strong:
            signal = "STRONG"
        elif early:
            signal = "EARLY"

        if signal is None:
            has_wt = bool(wt_recent_arr[day_idx]) or bool(wt_cross_arr[day_idx])
            has_pmax = bool(pmax_arr[day_idx])
            has_smc = bool(recent_bos_arr[day_idx]) or bool(recent_choch_arr[day_idx])
            active = int(has_wt) + int(has_pmax) + int(has_smc)
            if active >= 2:
                signal = "PARTIAL"
            else:
                continue

        # ── Otomatik mod seçimi ──
        # risk-on + haftalık ST UP + RS≥20 → MOMENTUM, else CORE
        is_risk_on = (mkt_regime == 2) and _weekly_st
        is_momentum = is_risk_on and _rs >= MOMENTUM_RS_THRESH and signal in ("STRONG", "PARTIAL")
        trade_mode = "MOMENTUM" if is_momentum else "CORE"

        # ── elite flag: sadece istenen modu döndür ──
        if elite and trade_mode != "MOMENTUM":
            continue
        if not elite and trade_mode != "CORE":
            continue

        # ── CORE mod filtreleri ──
        if trade_mode == "CORE":
            # Grey market rejimde sadece en iyi sinyaller
            if mkt_regime == 1 and signal not in GREY_ALLOWED:
                continue
            # Weekly ST down → sadece EARLY, PARTIAL, STRONG geçebilir
            if not _weekly_st and signal not in ("EARLY", "PARTIAL", "STRONG"):
                continue
            # Quality gates
            if signal in ("EARLY", "COMBO") and _quality < CORE_Q_MIN_EARLY:
                continue
            if signal in ("PARTIAL", "STRONG") and _quality < CORE_Q_MIN_PARTIAL_STRONG:
                continue

        # ── MOMENTUM mod filtreleri ──
        if trade_mode == "MOMENTUM":
            # Borderline risk-on + düşük quality → reject
            xu_slope_val = float(xu_ema50_slope.iloc[day_idx]) if day_idx < len(xu_ema50_slope) else 0
            if xu_slope_val < 0.5 and _quality < 60:  # borderline
                continue

        # ── Ortak filtreler (her iki mod) ──
        # CMF: ağır dağıtım → sinyal reddet
        if _cmf < -0.2:
            continue
        # Overextension: fiyat EMA20'den > 2 ATR uzak
        if _dist_ema > 2.0:
            continue
        # Gap risk: son 3 günde %8+ gap
        if day_idx >= 3:
            gap_risk = False
            for gi in range(1, 4):
                prev_close = c_arr[day_idx - gi]
                curr_open = o.values[day_idx - gi + 1]
                if prev_close > 0 and not np.isnan(curr_open):
                    gap_pct = abs(curr_open / prev_close - 1) * 100
                    if gap_pct > 8.0:
                        gap_risk = True
                        break
            if gap_risk:
                continue

        # ── Stop / TP ──
        stop_mult = {
            "COMBO": COMBO_STOP,
            "STRONG": TREND_STOP,
            "EARLY": GRI_STOP,
            "PARTIAL": TREND_STOP,
        }[signal]

        if signal == "COMBO":
            tp_price = close_price + atr_val * COMBO_TP
        elif signal == "EARLY":
            tp_price = close_price + atr_val * TREND_TP * 0.8
        else:
            tp_price = close_price + atr_val * TREND_TP

        stop_price = close_price - atr_val * stop_mult
        risk = close_price - stop_price
        reward = tp_price - close_price
        rr = reward / risk if risk > 0 else 0

        # ── rr20 default filtre: R:R < 2.0 → trade açma ──
        if rr < 2.0:
            continue

        # ── Position sizing (sinyal bazlı risk ağırlığı) ──
        if trade_mode == "MOMENTUM":
            pos_size = POS_SIZE_MOMENTUM.get(signal, 1.0)
        else:
            pos_size = POS_SIZE_CORE.get(signal, 1.0)

        regime_name = {3: "FULL_TREND", 2: "TREND", 1: "GRI_BOLGE", 0: "CHOPPY"}[regime]

        bt_regime_name, bt_regime_type = get_regime_for_date(signal_date)

        signals.append({
            'ticker': ticker,
            'signal': signal,
            'trade_mode': trade_mode,
            'regime': regime_name,
            'regime_score': regime,
            'close': round(close_price, 2),
            'stop': round(stop_price, 2),
            'tp': round(tp_price, 2),
            'rr': round(rr, 2),
            'atr': round(atr_val, 2),
            'atr_val_raw': atr_val,
            'trail_mult': TRAIL_MULT,
            'quality': int(_quality),
            'rs_score': round(_rs, 1),
            'rvol': round(float(rvol_arr[day_idx]), 1) if not np.isnan(rvol_arr[day_idx]) else 0,
            'bb_pctb': round(float(bb_pctb_arr[day_idx]), 2) if not np.isnan(bb_pctb_arr[day_idx]) else None,
            'overext_score': _oe,
            'overext_tags': [],
            'overext_warning': _oe >= 3,
            'turnover_m': round(float(avg_turn_arr[day_idx]) / 1e6, 1),
            'cmf': round(_cmf, 3),
            'atr_pctile': round(float(atr_pctile_arr[day_idx]), 2),
            'dist_ema20': round(_dist_ema, 2),
            'market_regime': mkt_regime,
            'weekly_st_up': _weekly_st,
            'pos_size': pos_size,
            'date': signal_date,
            'day_idx': day_idx,
            'regime_period': bt_regime_name,
            'regime_type': bt_regime_type,
        })

    if debug:
        print(f"    {ticker}: signals={len(signals)}")
    return signals


# ══════════════════════════════════════════════════════════════
# VECTORIZED DIP SIGNAL GENERATION
# ══════════════════════════════════════════════════════════════

def _fast_dip_signals(ticker, df, xu_df, usd_df, dip_mod, debug=False):
    """
    DIP sinyalleri — pahalı göstergeleri 1 kez hesapla, cache ile analyze_dip çağır.
    calc_supertrend, calc_wavetrend, calc_adx_ema, resample_weekly cache'lenir.
    """
    n = len(df)
    min_days = 200
    if n < min_days:
        return []

    fridays = [i for i in range(min_days, n) if df.index[i].weekday() == 4]
    if not fridays:
        return []

    # ── Precompute pahalı göstergeler (1 kez, full data) ──
    full_st = calc_supertrend(df, ST_LEN, ST_MULT)
    full_wt = calc_wavetrend(df)
    full_adx_ema = calc_adx_ema(df, ADX_LEN)
    full_weekly = resample_weekly(df)

    # XU100 weekly (precompute)
    full_xu_weekly = None
    if xu_df is not None and len(xu_df) > 150:
        full_xu_weekly = resample_weekly(xu_df)

    # USD weekly (precompute)
    full_usd_weekly = None
    full_usd_price = None
    if usd_df is not None:
        full_usd_price = to_usd(df, usd_df)
        if full_usd_price is not None and len(full_usd_price) >= 100:
            full_usd_weekly = resample_weekly(full_usd_price)

    # ── Cached fonksiyon yaratıcıları ──
    def _cached_supertrend(df_arg, *a, **kw):
        return full_st.iloc[:len(df_arg)]

    def _cached_wavetrend(df_arg):
        m = len(df_arg)
        return {k: (v.iloc[:m] if isinstance(v, pd.Series) else v)
                for k, v in full_wt.items()}

    def _cached_adx_ema(df_arg, *a, **kw):
        if len(df_arg) == len(df) or (hasattr(df_arg.index, 'freq') and df_arg.index.freq is None and len(df_arg) <= len(df)):
            # Daily data — use cache if it's a slice of the original
            if len(df_arg) <= len(full_adx_ema):
                return full_adx_ema.iloc[:len(df_arg)]
        return calc_adx_ema(df_arg, *a, **kw)

    def _cached_resample_weekly(df_arg):
        # Daily stock data → use full_weekly cache
        if len(df_arg) <= len(df) and len(df_arg) > 0:
            last_date = df_arg.index[-1]
            # Check if this is stock data (same index as df)
            if df_arg.index[0] == df.index[0]:
                return full_weekly.loc[:last_date]
            # Check if this is XU100 data
            if full_xu_weekly is not None and xu_df is not None and len(df_arg) <= len(xu_df):
                if df_arg.index[0] == xu_df.index[0]:
                    return full_xu_weekly.loc[:last_date]
            # Check if this is USD-converted data
            if full_usd_weekly is not None and full_usd_price is not None and len(df_arg) <= len(full_usd_price):
                try:
                    if df_arg.index[0] == full_usd_price.index[0]:
                        return full_usd_weekly.loc[:last_date]
                except:
                    pass
        return resample_weekly(df_arg)

    def _cached_to_usd(df_arg, usd_arg):
        if full_usd_price is not None and len(df_arg) <= len(df):
            if df_arg.index[0] == df.index[0]:
                last_date = df_arg.index[-1]
                sliced = full_usd_price.loc[:last_date]
                if len(sliced) >= 55:
                    return sliced
        return to_usd(df_arg, usd_arg)

    # ── Monkey-patch: dip_mod namespace'ine cache'li versiyonları enjekte et ──
    orig_funcs = {
        'calc_supertrend': dip_mod.calc_supertrend,
        'calc_wavetrend': dip_mod.calc_wavetrend,
        'calc_adx_ema': dip_mod.calc_adx_ema,
        'resample_weekly': dip_mod.resample_weekly,
        'to_usd': dip_mod.to_usd,
    }

    dip_mod.calc_supertrend = _cached_supertrend
    dip_mod.calc_wavetrend = _cached_wavetrend
    dip_mod.calc_adx_ema = _cached_adx_ema
    dip_mod.resample_weekly = _cached_resample_weekly
    dip_mod.to_usd = _cached_to_usd

    signals = []
    errors = 0

    try:
        for day_idx in fridays:
            try:
                window = df.iloc[:day_idx + 1]
                xu_window = xu_df.iloc[:day_idx + 1] if xu_df is not None and day_idx < len(xu_df) else xu_df
                result = dip_mod.analyze_dip(
                    ticker, window, xu_window, usd_df=usd_df, dbg=None
                )
                if result:
                    signal_date = df.index[day_idx]
                    regime_name, regime_type = get_regime_for_date(signal_date)
                    result['date'] = signal_date
                    result['day_idx'] = day_idx
                    result['regime_period'] = regime_name
                    result['regime_type'] = regime_type
                    signals.append(result)
            except Exception as e:
                errors += 1
                if errors <= 3 and debug:
                    print(f"    [ERR] {ticker} day={day_idx}: {type(e).__name__}: {e}")
    finally:
        # Orijinal fonksiyonları geri yükle
        for fname, func in orig_funcs.items():
            setattr(dip_mod, fname, func)

    if debug and (signals or errors):
        print(f"    {ticker}: signals={len(signals)} errors={errors}")
    return signals


# ══════════════════════════════════════════════════════════════
# ANA BACKTEST
# ══════════════════════════════════════════════════════════════

def generate_signals_for_ticker(ticker, df, xu_df, usd_df, regime_mod, dip_mod, mode='trend', debug=False, elite=False):
    """
    Tek hisse için TÜM tarihlerde sinyal üret.
    Vectorized: göstergeleri 1 kez hesapla, tüm tarihleri tara.
    elite=True → TREND Momentum modu (RS≥20 + risk-on only)
    """
    if mode == 'trend':
        return _fast_trend_signals(ticker, df, xu_df, usd_df, debug=debug, elite=elite)
    elif mode == 'dip':
        return _fast_dip_signals(ticker, df, xu_df, usd_df, dip_mod, debug=debug)
    return []


def run_backtest(tickers, all_data, xu_df, usd_df, regime_mod, dip_mod,
                 mode='trend', max_tickers=None, progress_cb=None, elite=False):
    """Ana backtest fonksiyonu."""
    all_signals = []
    t0 = time.time()
    n_tickers = len(tickers) if max_tickers is None else min(max_tickers, len(tickers))
    total_errors = 0

    for idx, ticker in enumerate(tickers[:n_tickers]):
        if progress_cb:
            progress_cb(idx + 1, n_tickers, ticker)
        elif (idx + 1) % 20 == 0 or idx == 0:
            elapsed = time.time() - t0
            eta = (elapsed / (idx + 1)) * (n_tickers - idx - 1)
            print(f"  [{idx+1}/{n_tickers}] {ticker} ... ({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")

        df = all_data.get(ticker)
        if df is None or len(df) < 200:
            continue

        debug = (idx < 3)
        signals = generate_signals_for_ticker(
            ticker, df, xu_df, usd_df, regime_mod, dip_mod, mode, debug=debug, elite=elite
        )

        for sig in signals:
            day_idx = sig.get('day_idx')
            stop = sig.get('stop', 0)
            tp = sig.get('tp', 0)
            if day_idx is None or stop <= 0 or tp <= 0:
                continue

            trade = simulate_trade(
                df, day_idx, stop, tp,
                trail_mult=sig.get('trail_mult'),
                atr_val=sig.get('atr_val_raw'),
            )
            if trade is None:
                continue

            combined = {**sig, **trade}
            combined['entry_price'] = float(df['Close'].iloc[day_idx])
            all_signals.append(combined)

    elapsed = time.time() - t0
    print(f"\n✅ Backtest tamamlandı: {len(all_signals)} trade, {elapsed:.1f}s")

    return all_signals
