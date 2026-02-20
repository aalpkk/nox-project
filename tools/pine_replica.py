#!/usr/bin/env python3
"""
BIST Rejim Filtresi (ADX + Slope) — PineScript v6 Replica
Exact Python replica of the TradingView indicator.

Usage:
    python tools/pine_replica.py THYAO              # single ticker
    python tools/pine_replica.py THYAO EREGL        # multiple tickers
    python tools/pine_replica.py --all              # all BIST tickers
    python tools/pine_replica.py THYAO --days 30    # last N days history
"""
import sys
import os
import argparse
import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.indicators import (
    ema, sma, rma, true_range,
    calc_atr, calc_rsi, calc_adx,
    calc_supertrend, calc_wavetrend,
)
from markets.bist.data import fetch_data, fetch_benchmark, get_all_bist_tickers


def resample_weekly_fri(df):
    """Resample daily OHLCV to weekly with Friday-ending weeks (matches TradingView)."""
    wdf = df.resample('W-FRI').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min',
        'Close': 'last', 'Volume': 'sum'
    }).dropna()
    return wdf


# ═══════════════════════════════════════════════════════════════
# Section 1: PineScript Parameters (matching Pine defaults exactly)
# ═══════════════════════════════════════════════════════════════

PINE = {
    # ADX
    'adx_len':          14,
    'adx_trend':        20,
    'adx_choppy':       15,
    'adx_slope_len':    3,      # Pine=3 (NOX uses 5)
    'adx_slope_thresh': 0.5,

    # EMA
    'ema_fast':         21,
    'ema_slow':         55,

    # SuperTrend
    'st_period':        10,
    'st_mult':          3.0,

    # Squeeze
    'sq_len':           20,
    'sq_bb_mult':       2.0,
    'sq_kc_mult':       1.5,

    # Bollinger Bands
    'bb_len':           20,
    'bb_mult':          2.0,

    # Stochastic RSI
    'stoch_rsi_len':    14,
    'stoch_oversold':   20,

    # Relative Strength
    'rs_len1':          21,     # Pine=21 (NOX uses 10)
    'rs_len2':          63,     # Pine=63 (NOX uses 50)
    'rs_thresh':        0.0,    # Pine=0.0 (NOX uses 5.0)

    # WaveTrend
    'wt_ch_len':        10,
    'wt_avg_len':       21,
    'wt_ma_len':        4,

    # Volume
    'vol_sma_len':      20,
    'vol_breakout':     1.5,    # Pine=1.5x (NOX uses 0.5x)

    # ATR
    'atr_len':          14,

    # Stop/TP
    'stop_mult':        2.0,
    'tp_mult':          2.0,
}


# ═══════════════════════════════════════════════════════════════
# Section 3: New helper functions (not in core/indicators.py)
# ═══════════════════════════════════════════════════════════════

def rolling_linreg(series, length, offset=0):
    """
    Replicate ta.linreg(source, length, offset) from PineScript.
    For each bar: fit OLS on last `length` values, return predicted
    value at position (length - 1 - offset).

    Vectorized implementation using rolling window sums.
    """
    s = series.values.astype(float)
    n = len(s)
    result = np.full(n, np.nan)

    if length < 2:
        return pd.Series(result, index=series.index)

    # x = 0, 1, ..., length-1
    x_sum = length * (length - 1) / 2.0
    x2_sum = length * (length - 1) * (2 * length - 1) / 6.0
    x_eval = length - 1 - offset

    for i in range(length - 1, n):
        window = s[i - length + 1: i + 1]
        if np.any(np.isnan(window)):
            continue
        y_sum = np.sum(window)
        xy_sum = np.dot(np.arange(length, dtype=float), window)

        denom = length * x2_sum - x_sum * x_sum
        if denom == 0:
            continue

        slope = (length * xy_sum - x_sum * y_sum) / denom
        intercept = (y_sum - slope * x_sum) / length
        result[i] = slope * x_eval + intercept

    return pd.Series(result, index=series.index)


def calc_stoch_rsi(series, length):
    """
    Replicate Pine's ta.stoch(rsi, rsi, rsi, length) then ta.sma(..., 3) twice.
    Step 1: RSI
    Step 2: Stoch of RSI = (rsi - rsi_min) / (rsi_max - rsi_min) * 100
    Step 3: stochK = sma(stoch_raw, 3)
    Step 4: stochD = sma(stochK, 3)
    """
    rsi = calc_rsi(series, length)

    rsi_min = rsi.rolling(length).min()
    rsi_max = rsi.rolling(length).max()
    rsi_range = rsi_max - rsi_min
    stoch_raw = ((rsi - rsi_min) / rsi_range.replace(0, np.nan) * 100).fillna(50)

    stoch_k = sma(stoch_raw, 3)
    stoch_d = sma(stoch_k, 3)

    return stoch_k, stoch_d


# ═══════════════════════════════════════════════════════════════
# Section 4: Indicator computation
# ═══════════════════════════════════════════════════════════════

def compute_indicators(df, xu_df):
    """
    Compute all PineScript indicators for the full DataFrame.
    Returns a dict of indicator series keyed by name.
    """
    P = PINE
    c = df['Close']
    h = df['High']
    l = df['Low']
    v = df['Volume']
    ind = {}

    # ── 1. Daily ADX + slope ──
    adx_s = calc_adx(df, P['adx_len'])
    adx_slope_s = (adx_s - adx_s.shift(P['adx_slope_len'])) / P['adx_slope_len']
    adx_rising_s = adx_slope_s > P['adx_slope_thresh']
    ind['adx'] = adx_s
    ind['adx_slope'] = adx_slope_s
    ind['adx_rising'] = adx_rising_s

    # ── 2. Weekly ADX (Pine: request.security("W", adxValue)) ──
    # Use Friday-ending weeks to match TradingView.
    # No shift: on historical data, ffill naturally gives last completed week
    # for Mon-Thu, and current week's close value on Friday.
    wdf = resample_weekly_fri(df)
    if len(wdf) >= 20:
        w_adx_s = calc_adx(wdf, P['adx_len'])
        w_adx_slope = (w_adx_s - w_adx_s.shift(P['adx_slope_len'])) / P['adx_slope_len']
        w_adx_rising = w_adx_slope > P['adx_slope_thresh']

        # Map weekly to daily: forward-fill
        ind['w_adx'] = w_adx_s.reindex(df.index, method='ffill')
        ind['w_adx_slope'] = w_adx_slope.reindex(df.index, method='ffill')
        ind['w_adx_rising'] = w_adx_rising.reindex(df.index, method='ffill').fillna(False)
    else:
        ind['w_adx'] = pd.Series(0.0, index=df.index)
        ind['w_adx_slope'] = pd.Series(0.0, index=df.index)
        ind['w_adx_rising'] = pd.Series(False, index=df.index)

    # ── 3. Weekly EMA (Pine: request.security("W", ta.ema(close, N))) ──
    if len(wdf) >= P['ema_slow'] + 5:
        w_ema_f = ema(wdf['Close'], P['ema_fast'])
        w_ema_s = ema(wdf['Close'], P['ema_slow'])
        ind['w_ema_f'] = w_ema_f.reindex(df.index, method='ffill')
        ind['w_ema_s'] = w_ema_s.reindex(df.index, method='ffill')
    else:
        ind['w_ema_f'] = pd.Series(np.nan, index=df.index)
        ind['w_ema_s'] = pd.Series(np.nan, index=df.index)

    # ── 4. Daily EMA 21/55 ──
    ema_f = ema(c, P['ema_fast'])
    ema_s = ema(c, P['ema_slow'])
    ind['ema_f'] = ema_f
    ind['ema_s'] = ema_s
    ind['ema_trend_up'] = ema_f > ema_s

    # ── 5. SuperTrend ──
    st_dir = calc_supertrend(df, P['st_period'], P['st_mult'])
    ind['st_dir'] = st_dir
    ind['st_up'] = st_dir == 1

    # ── 6. Regime score ──
    # HTF trend: weekly EMA fast > slow
    htf_trend_up = ind['w_ema_f'] > ind['w_ema_s']
    # 2-of-3 rule + close > ema_slow
    trend_count = ind['ema_trend_up'].astype(int) + ind['st_up'].astype(int) + htf_trend_up.astype(int)
    confirmed_trend_up = (trend_count >= 2) & (c > ema_s)
    ind['confirmed_trend_up'] = confirmed_trend_up

    # HTF regime (from weekly ADX)
    # Pine: htfRegime_raw = htfADX > adxTrend and htfRising ? 2 :
    #                       htfADX > adxTrend ? 1 : htfADX > adxChoppy ? 0 : -1
    w_adx = ind['w_adx'].fillna(0)
    w_adx_rising = ind['w_adx_rising'].fillna(False)
    htf_r = pd.Series(-1, index=df.index, dtype=int)
    htf_r[w_adx > P['adx_choppy']] = 0
    htf_r[w_adx > P['adx_trend']] = 1
    htf_r[(w_adx > P['adx_trend']) & w_adx_rising] = 2

    # Daily confirm
    daily_confirm = (adx_s > P['adx_choppy']) & adx_rising_s

    # Regime score 0-3
    regime = pd.Series(0, index=df.index, dtype=int)
    mask_confirmed = confirmed_trend_up
    regime[mask_confirmed & (htf_r == 2) & daily_confirm] = 3
    regime[mask_confirmed & (htf_r >= 1) & ~((htf_r == 2) & daily_confirm)] = 2
    regime[mask_confirmed & (htf_r == 0) & ~(htf_r >= 1)] = 1
    # regime stays 0 where not confirmed or htf_r == -1

    ind['regime'] = regime
    ind['htf_trend_up'] = htf_trend_up

    # ── 7. Squeeze Momentum (with linreg — Pine exact) ──
    sq_basis = sma(c, P['sq_len'])
    sq_dev = c.rolling(P['sq_len']).std(ddof=0) * P['sq_bb_mult']  # Pine ta.stdev uses ddof=0
    sq_rng = sma(true_range(df), P['sq_len'])

    sq_bb_upper = sq_basis + sq_dev
    sq_bb_lower = sq_basis - sq_dev
    sq_kc_upper = sq_basis + P['sq_kc_mult'] * sq_rng
    sq_kc_lower = sq_basis - P['sq_kc_mult'] * sq_rng

    # Pine: sqzOn = sqLowerBB > sqLowerKC and sqUpperBB < sqUpperKC  (BB inside KC)
    # Pine: sqzOff = sqLowerBB < sqLowerKC and sqUpperBB > sqUpperKC (BB outside KC)
    # NOTE: NOT the complement! There's a 3rd state (partial overlap) where both are False.
    sqz_on = (sq_bb_lower > sq_kc_lower) & (sq_bb_upper < sq_kc_upper)
    sqz_off = (sq_bb_lower < sq_kc_lower) & (sq_bb_upper > sq_kc_upper)

    # Squeeze momentum using linreg (Pine exact)
    hh = h.rolling(P['sq_len']).max()
    ll = l.rolling(P['sq_len']).min()
    sq_mid = (hh + ll) / 2
    sq_mom_src = c - (sq_mid + sq_basis) / 2
    sq_mom = rolling_linreg(sq_mom_src, P['sq_len'], offset=0)

    sq_mom_rising = sq_mom > sq_mom.shift(1)
    sq_release = sqz_off & sqz_on.shift(1) & (sq_mom > 0) & sq_mom_rising

    ind['sqz_on'] = sqz_on
    ind['sqz_off'] = sqz_off
    ind['sq_mom'] = sq_mom
    ind['sq_mom_rising'] = sq_mom_rising
    ind['sq_release'] = sq_release

    # ── 8. BB %B ──
    bb_basis = sma(c, P['bb_len'])
    bb_dev_val = c.rolling(P['bb_len']).std(ddof=0) * P['bb_mult']  # Pine ta.stdev uses ddof=0
    bb_upper = bb_basis + bb_dev_val
    bb_lower = bb_basis - bb_dev_val
    bb_pctb = (c - bb_lower) / (bb_upper - bb_lower)
    ind['bb_pctb'] = bb_pctb

    # ── 9. Stochastic RSI ──
    stoch_k, stoch_d = calc_stoch_rsi(c, P['stoch_rsi_len'])
    # Pine: stochCrossUp = ta.crossover(stochK, stochD) and stochK < stochOversold
    stoch_cross_up = ((stoch_k > stoch_d) & (stoch_k.shift(1) <= stoch_d.shift(1))
                      & (stoch_k < P['stoch_oversold']))
    ind['stoch_k'] = stoch_k
    ind['stoch_d'] = stoch_d
    ind['stoch_cross_up'] = stoch_cross_up

    # ── 10. WaveTrend ──
    wt = calc_wavetrend(df)
    ind['wt1'] = wt['wt1']
    ind['wt2'] = wt['wt2']
    ind['wt_cross_up'] = wt['cross_up']
    ind['wt_cross_dn'] = wt['cross_dn']
    ind['wt_bullish'] = wt['wt_bullish']

    # ── 11. RS score (Pine periods: 21/63, threshold: 0.0) ──
    rs_score = pd.Series(0.0, index=df.index)
    rs_pass = pd.Series(True, index=df.index)
    if xu_df is not None and len(xu_df) >= P['rs_len2'] + 5:
        bench_c = xu_df['Close'].reindex(df.index, method='ffill')
        for i in range(P['rs_len2'] + 5, len(df)):
            sc = c.iloc[i]
            sc1 = c.iloc[i - P['rs_len1']]
            sc2 = c.iloc[i - P['rs_len2']]
            bc = bench_c.iloc[i]
            bc1 = bench_c.iloc[i - P['rs_len1']]
            bc2 = bench_c.iloc[i - P['rs_len2']]
            if sc1 > 0 and sc2 > 0 and bc1 > 0 and bc2 > 0:
                sp1 = (sc - sc1) / sc1 * 100
                sp2 = (sc - sc2) / sc2 * 100
                bp1 = (bc - bc1) / bc1 * 100
                bp2 = (bc - bc2) / bc2 * 100
                rs_val = (sp1 - bp1) * 0.6 + (sp2 - bp2) * 0.4
                rs_score.iloc[i] = rs_val
                rs_pass.iloc[i] = rs_val > P['rs_thresh']
    ind['rs_score'] = rs_score
    ind['rs_pass'] = rs_pass

    # ── 12. Volume ──
    vol_sma = sma(v, P['vol_sma_len'])
    vol_breakout = v > vol_sma * P['vol_breakout']
    ind['vol_sma'] = vol_sma
    ind['vol_breakout'] = vol_breakout

    # ── 13. ATR ──
    atr = calc_atr(df, P['atr_len'])
    ind['atr'] = atr

    return ind


# ═══════════════════════════════════════════════════════════════
# Section 5: Signal generation (per-bar, no state)
# ═══════════════════════════════════════════════════════════════

def compute_signals(df, ind):
    """
    For EVERY bar independently (no state):
    GUCLU, ZAYIF, PB, MR, DONUS, SQ

    Priority (highest wins): GUCLU > ZAYIF > PB > MR > DONUS > SQ
    Each bar gets at most 1 signal.
    """
    P = PINE
    n = len(df)
    c = df['Close']
    signals = pd.Series('', index=df.index, dtype=str)

    # Pre-extract series for vectorized access
    regime = ind['regime']
    ema_trend_up = ind['ema_trend_up']
    st_up = ind['st_up']
    sq_release = ind['sq_release']
    sq_mom = ind['sq_mom']
    sq_mom_rising = ind['sq_mom_rising']
    sqz_off = ind['sqz_off']
    vol_breakout = ind['vol_breakout']
    rs_pass = ind['rs_pass']
    bb_pctb = ind['bb_pctb']
    stoch_cross_up = ind['stoch_cross_up']
    ema_s = ind['ema_s']
    ema_f = ind['ema_f']
    wt_cross_up = ind['wt_cross_up']

    # Squeeze sub-condition: sq_release OR (sq_mom>0 AND sq_mom_rising AND sqz_off)
    sq_cond = sq_release | ((sq_mom > 0) & sq_mom_rising & sqz_off)

    # GUCLU: regime>=2 AND ema_trend_up AND st_up AND sq_cond AND vol_breakout AND rs_pass
    guclu = (regime >= 2) & ema_trend_up & st_up & sq_cond & vol_breakout & rs_pass

    # ZAYIF: regime>=2 AND ema_trend_up AND st_up AND sq_cond AND NOT vol_breakout AND rs_pass
    zayif = (regime >= 2) & ema_trend_up & st_up & sq_cond & (~vol_breakout) & rs_pass

    # PB: regime>=2 AND st_up AND rs_pass AND bb_pctb<0.3 AND bb_pctb>0.05 AND sq_mom_rising AND close>ema_s
    pb = ((regime >= 2) & st_up & rs_pass &
          (bb_pctb < 0.3) & (bb_pctb > 0.05) &
          sq_mom_rising & (c > ema_s))

    # MR: regime<=1 AND bb_pctb<0.1 AND stoch_cross_up AND close>ema_s*0.95 AND rs_pass
    mr = ((regime <= 1) & (bb_pctb < 0.1) & stoch_cross_up &
          (c > ema_s * 0.95) & rs_pass)

    # DONUS: recent_ema55_cross(10bar) AND recent_wt_cross(10bar) AND rs_pass
    #        AND close>ema_s AND close>ema_f
    ema55_cross = (c > ema_s) & (c.shift(1) <= ema_s.shift(1))
    recent_ema55 = pd.Series(False, index=df.index)
    recent_wt = pd.Series(False, index=df.index)
    for i in range(len(df)):
        for j in range(min(10, i + 1)):
            if i - j >= 0 and ema55_cross.iloc[i - j]:
                recent_ema55.iloc[i] = True
                break
        for j in range(min(10, i + 1)):
            if i - j >= 0 and wt_cross_up.iloc[i - j]:
                recent_wt.iloc[i] = True
                break

    donus = recent_ema55 & recent_wt & rs_pass & (c > ema_s) & (c > ema_f)

    # SQ: sq_release AND ema_trend_up AND st_up AND rs_pass AND sq_mom>0
    sq = sq_release & ema_trend_up & st_up & rs_pass & (sq_mom > 0)

    # Priority assignment (highest wins)
    signals[sq] = 'SQ'
    signals[donus] = 'DONUS'
    signals[mr] = 'MR'
    signals[pb] = 'PB'
    signals[zayif] = 'ZAYIF'
    signals[guclu] = 'GUCLU'

    return signals


# ═══════════════════════════════════════════════════════════════
# Section 6: Continuation tracking
# ═══════════════════════════════════════════════════════════════

def compute_continuation(signals):
    """
    For each bar:
    - guclu_devam: True if yesterday was GUCLU AND today has ANY signal
    - any_devam:   True if today's signal == yesterday's signal (same signal persists)
    """
    n = len(signals)
    guclu_devam = pd.Series(False, index=signals.index)
    any_devam = pd.Series(False, index=signals.index)
    devam_label = pd.Series('', index=signals.index, dtype=str)

    for i in range(1, n):
        prev = signals.iloc[i - 1]
        curr = signals.iloc[i]

        if prev == 'GUCLU' and curr != '':
            guclu_devam.iloc[i] = True
            devam_label.iloc[i] = 'DEVAM'
        elif prev != '' and curr == prev:
            any_devam.iloc[i] = True
            devam_label.iloc[i] = 'DEVAM'

    return {
        'guclu_devam': guclu_devam,
        'any_devam': any_devam,
        'devam_label': devam_label,
    }


# ═══════════════════════════════════════════════════════════════
# Section 7: Stop/TP computation
# ═══════════════════════════════════════════════════════════════

def compute_stops(df, ind, signals):
    """
    For each bar with a signal:
    stop = close - ATR * 2.0
    tp   = close + ATR * 2.0 (ZAYIF gets tp_mult * 0.5 = 1.0)
    rr   = (tp - close) / (close - stop)
    pos_size = regime: 3->100%, 2->75%, 1->40%, 0->20%
    """
    P = PINE
    c = df['Close']
    atr = ind['atr']
    regime = ind['regime']

    stop = pd.Series(np.nan, index=df.index)
    tp = pd.Series(np.nan, index=df.index)
    rr = pd.Series(np.nan, index=df.index)
    pos_pct = pd.Series(np.nan, index=df.index)

    pos_map = {3: 100, 2: 75, 1: 40, 0: 20}

    has_signal = signals != ''
    stop[has_signal] = c[has_signal] - atr[has_signal] * P['stop_mult']

    # TP: ZAYIF gets half the TP multiplier
    is_zayif = signals == 'ZAYIF'
    tp[has_signal & ~is_zayif] = c[has_signal & ~is_zayif] + atr[has_signal & ~is_zayif] * P['tp_mult']
    tp[has_signal & is_zayif] = c[has_signal & is_zayif] + atr[has_signal & is_zayif] * (P['tp_mult'] * 0.5)

    # R:R
    risk = c - stop
    reward = tp - c
    rr[has_signal] = (reward[has_signal] / risk[has_signal].replace(0, np.nan))

    # Position size
    for score, pct in pos_map.items():
        mask = has_signal & (regime == score)
        pos_pct[mask] = pct

    return {
        'stop': stop,
        'tp': tp,
        'rr': rr,
        'pos_pct': pos_pct,
    }


# ═══════════════════════════════════════════════════════════════
# Section 8: Output display
# ═══════════════════════════════════════════════════════════════

def regime_name(score):
    return {3: "FULL_TREND", 2: "TREND", 1: "GRI_BOLGE", 0: "CHOPPY"}.get(score, "?")


def arrow(val):
    if val is None or np.isnan(val):
        return "?"
    return "^" if val > 0 else "v"


def display_results(ticker, df, ind, signals, cont, stops, days=10):
    """Formatted CLI table showing last N days."""
    c = df['Close']
    n = len(df)
    last = min(days, n)

    # Current bar summary
    i = -1
    cur_regime = int(ind['regime'].iloc[i])
    cur_adx = ind['adx'].iloc[i]
    cur_adx_slope = ind['adx_slope'].iloc[i]
    cur_w_adx = ind['w_adx'].iloc[i]
    cur_w_adx_slope = ind['w_adx_slope'].iloc[i]
    cur_ema_up = ind['ema_trend_up'].iloc[i]
    cur_st_up = ind['st_up'].iloc[i]
    cur_htf_up = ind['htf_trend_up'].iloc[i]
    cur_rs = ind['rs_score'].iloc[i]
    cur_rs_pass = ind['rs_pass'].iloc[i]
    cur_vol_break = ind['vol_breakout'].iloc[i]
    cur_sqz = ind['sqz_on'].iloc[i]
    cur_sq_release = ind['sq_release'].iloc[i]
    cur_bb_pctb = ind['bb_pctb'].iloc[i]
    cur_atr = ind['atr'].iloc[i]

    # Squeeze status
    if cur_sq_release:
        sq_txt = "RELEASE!"
    elif cur_sqz:
        sq_txt = "SQUEEZE"
    else:
        sq_txt = "OFF"

    vol_txt = "YUKSEK" if cur_vol_break else "DUSUK"

    print()
    print(f"{'=' * 70}")
    print(f"  {ticker} -- BIST Rejim Filtresi (Pine Replica)")
    print(f"{'=' * 70}")
    print(f"  Rejim: {regime_name(cur_regime)} (skor={cur_regime})"
          f" | ADX: {cur_adx:.1f} {arrow(cur_adx_slope)}"
          f" | Haftalik ADX: {cur_w_adx:.1f} {arrow(cur_w_adx_slope)}")
    print(f"  Trend: EMA {'UP' if cur_ema_up else 'DN'}"
          f" ST {'UP' if cur_st_up else 'DN'}"
          f" HTF {'UP' if cur_htf_up else 'DN'}"
          f" | RS: {cur_rs:+.1f} {'OK' if cur_rs_pass else 'FAIL'}"
          f" | Hacim: {vol_txt}")
    print(f"  Squeeze: {sq_txt}"
          f" | BB%B: {cur_bb_pctb:.2f}"
          f" | ATR: {cur_atr:.2f}")
    print(f"{'-' * 70}")

    # Table header
    print(f"  {'Tarih':>12}  {'Sinyal':>10}  {'Devam':>5}"
          f"  {'Fiyat':>8}  {'Stop':>8}  {'TP':>8}"
          f"  {'R:R':>5}  {'Poz%':>5}")
    print(f"  {'-' * 12}  {'-' * 10}  {'-' * 5}"
          f"  {'-' * 8}  {'-' * 8}  {'-' * 8}"
          f"  {'-' * 5}  {'-' * 5}")

    for j in range(last, 0, -1):
        idx = n - j
        if idx < 0:
            continue
        date_str = df.index[idx].strftime('%Y-%m-%d')
        sig = signals.iloc[idx]
        dev = cont['devam_label'].iloc[idx]
        price = c.iloc[idx]
        st = stops['stop'].iloc[idx]
        tp_val = stops['tp'].iloc[idx]
        rr_val = stops['rr'].iloc[idx]
        pos = stops['pos_pct'].iloc[idx]

        sig_disp = f"{sig} AL" if sig else "-"
        dev_disp = dev if dev else "-"
        st_disp = f"{st:.2f}" if not np.isnan(st) else "-"
        tp_disp = f"{tp_val:.2f}" if not np.isnan(tp_val) else "-"
        rr_disp = f"{rr_val:.1f}" if not np.isnan(rr_val) else "-"
        pos_disp = f"{pos:.0f}%" if not np.isnan(pos) else "-"

        print(f"  {date_str:>12}  {sig_disp:>10}  {dev_disp:>5}"
              f"  {price:>8.2f}  {st_disp:>8}  {tp_disp:>8}"
              f"  {rr_disp:>5}  {pos_disp:>5}")

    print(f"{'-' * 70}")

    # Today's summary
    today_sig = signals.iloc[-1]
    yesterday_sig = signals.iloc[-2] if n >= 2 else ''
    today_devam = cont['devam_label'].iloc[-1]

    if today_sig:
        print(f"  Bugun sinyal: {today_sig} AL")
    else:
        print(f"  Bugun sinyal: YOK")

    if yesterday_sig == 'GUCLU':
        if today_devam == 'DEVAM':
            print(f"  Dun GUCLU sinyal vardi, bugun devam EDIYOR")
        else:
            print(f"  Dun GUCLU sinyal vardi, bugun devam ETMIYOR")
    elif yesterday_sig:
        if today_devam == 'DEVAM':
            print(f"  Dun {yesterday_sig} sinyal vardi, bugun devam EDIYOR")

    print()


# ═══════════════════════════════════════════════════════════════
# Section 9: Scan summary for --all mode
# ═══════════════════════════════════════════════════════════════

def display_scan_summary(results):
    """Show compact summary table for multi-ticker scan."""
    if not results:
        print("\nHic sinyal bulunamadi.\n")
        return

    # Sort by signal priority
    priority = {'GUCLU': 1, 'ZAYIF': 2, 'PB': 3, 'MR': 4, 'DONUS': 5, 'SQ': 6}
    results.sort(key=lambda r: (priority.get(r['signal'], 99), -r['regime']))

    print()
    print(f"{'=' * 85}")
    print(f"  BIST Rejim Filtresi -- Sinyal Ozeti ({len(results)} sinyal)")
    print(f"{'=' * 85}")
    print(f"  {'Hisse':>8}  {'Sinyal':>10}  {'Devam':>5}  {'Rejim':>12}"
          f"  {'Fiyat':>8}  {'Stop':>8}  {'TP':>8}  {'R:R':>5}  {'RS':>6}")
    print(f"  {'-' * 8}  {'-' * 10}  {'-' * 5}  {'-' * 12}"
          f"  {'-' * 8}  {'-' * 8}  {'-' * 8}  {'-' * 5}  {'-' * 6}")

    for r in results:
        sig_disp = f"{r['signal']} AL"
        dev_disp = r['devam'] if r['devam'] else "-"
        regime_disp = regime_name(r['regime'])
        st_disp = f"{r['stop']:.2f}" if r['stop'] else "-"
        tp_disp = f"{r['tp']:.2f}" if r['tp'] else "-"
        rr_disp = f"{r['rr']:.1f}" if r['rr'] else "-"
        rs_disp = f"{r['rs']:+.1f}"
        print(f"  {r['ticker']:>8}  {sig_disp:>10}  {dev_disp:>5}  {regime_disp:>12}"
              f"  {r['close']:>8.2f}  {st_disp:>8}  {tp_disp:>8}"
              f"  {rr_disp:>5}  {rs_disp:>6}")

    print(f"{'=' * 85}")
    print()


# ═══════════════════════════════════════════════════════════════
# Main pipeline
# ═══════════════════════════════════════════════════════════════

def display_debug(ticker, df, ind):
    """
    Print indicator values matching Pine's info table for side-by-side comparison.
    Run: python tools/pine_replica.py THYAO --debug
    Then compare each value with TradingView's info table.
    """
    P = PINE
    c = df['Close']
    i = -1  # last bar

    regime = int(ind['regime'].iloc[i])
    adx_val = float(ind['adx'].iloc[i])
    adx_slope = float(ind['adx_slope'].iloc[i])
    adx_rising = bool(ind['adx_rising'].iloc[i])
    w_adx = float(ind['w_adx'].iloc[i])
    w_slope = float(ind['w_adx_slope'].iloc[i])
    w_rising = bool(ind['w_adx_rising'].iloc[i])
    ema_up = bool(ind['ema_trend_up'].iloc[i])
    st_up = bool(ind['st_up'].iloc[i])
    htf_up = bool(ind['htf_trend_up'].iloc[i])
    sqz_on = bool(ind['sqz_on'].iloc[i])
    sqz_off = bool(ind['sqz_off'].iloc[i])
    sq_mom = float(ind['sq_mom'].iloc[i]) if not np.isnan(ind['sq_mom'].iloc[i]) else 0
    sq_mom_rising = bool(ind['sq_mom_rising'].iloc[i])
    sq_release = bool(ind['sq_release'].iloc[i])
    bb_pctb = float(ind['bb_pctb'].iloc[i])
    vol_break = bool(ind['vol_breakout'].iloc[i])
    rs_score = float(ind['rs_score'].iloc[i])
    rs_pass = bool(ind['rs_pass'].iloc[i])
    atr_val = float(ind['atr'].iloc[i])
    stoch_k_val = float(ind['stoch_k'].iloc[i]) if not np.isnan(ind['stoch_k'].iloc[i]) else 0
    stoch_d_val = float(ind['stoch_d'].iloc[i]) if not np.isnan(ind['stoch_d'].iloc[i]) else 0
    wt1_val = float(ind['wt1'].iloc[i]) if not np.isnan(ind['wt1'].iloc[i]) else 0
    wt2_val = float(ind['wt2'].iloc[i]) if not np.isnan(ind['wt2'].iloc[i]) else 0

    # Confirmed trend components
    trend_count = int(ema_up) + int(st_up) + int(htf_up)
    confirmed = trend_count >= 2 and float(c.iloc[i]) > float(ind['ema_s'].iloc[i])

    # htfRegime breakdown
    w_adx_f = float(ind['w_adx'].iloc[i])
    if w_adx_f > P['adx_trend'] and w_rising:
        htf_regime = 2
    elif w_adx_f > P['adx_trend']:
        htf_regime = 1
    elif w_adx_f > P['adx_choppy']:
        htf_regime = 0
    else:
        htf_regime = -1

    daily_confirm = adx_val > P['adx_choppy'] and adx_rising

    # momCond1 / momCond2
    mom_cond1 = regime >= 2 and ema_up and st_up
    mom_cond2 = sq_release or (sq_mom > 0 and sq_mom_rising and sqz_off)

    print()
    print(f"{'=' * 60}")
    print(f"  {ticker} -- DEBUG (Pine tablosu ile karsilastir)")
    print(f"{'=' * 60}")
    print(f"  Tarih:            {df.index[i].strftime('%Y-%m-%d')}")
    print(f"  Fiyat:            {c.iloc[i]:.2f}")
    print(f"{'─' * 60}")
    print(f"  REJIM:            {regime_name(regime)} (skor={regime})")
    print(f"  confirmedTrendUp: {confirmed}  (trendCount={trend_count}, close>ema55={float(c.iloc[i]):.2f}>{float(ind['ema_s'].iloc[i]):.2f})")
    print(f"  htfRegime:        {htf_regime}")
    print(f"  dailyConfirm:     {daily_confirm}")
    print(f"{'─' * 60}")
    print(f"  Gunluk ADX:       {adx_val:.1f}  (slope={adx_slope:.2f}, rising={adx_rising})")
    print(f"  Haftalik ADX:     {w_adx:.1f}  (slope={w_slope:.2f}, rising={w_rising})")
    print(f"{'─' * 60}")
    print(f"  EMA trend up:     {ema_up}  (ema21={float(ind['ema_f'].iloc[i]):.2f}, ema55={float(ind['ema_s'].iloc[i]):.2f})")
    print(f"  SuperTrend up:    {st_up}")
    print(f"  HTF trend up:     {htf_up}  (w_ema21={float(ind['w_ema_f'].iloc[i]):.2f}, w_ema55={float(ind['w_ema_s'].iloc[i]):.2f})")
    print(f"{'─' * 60}")
    print(f"  sqzOn:            {sqz_on}")
    print(f"  sqzOff:           {sqz_off}")
    print(f"  sqMom:            {sq_mom:.4f}  (rising={sq_mom_rising})")
    print(f"  sqRelease:        {sq_release}")
    print(f"{'─' * 60}")
    print(f"  BB %B:            {bb_pctb:.2f}")
    print(f"  Stoch K/D:        {stoch_k_val:.1f} / {stoch_d_val:.1f}")
    print(f"  WT1/WT2:          {wt1_val:.1f} / {wt2_val:.1f}")
    print(f"{'─' * 60}")
    print(f"  Hacim breakout:   {vol_break}")
    print(f"  RS skoru:         {rs_score:+.1f}  (pass={rs_pass})")
    print(f"  ATR:              {atr_val:.2f}")
    print(f"{'─' * 60}")
    print(f"  momCond1:         {mom_cond1}  (regime>={2}, ema_up, st_up)")
    print(f"  momCond2:         {mom_cond2}  (sqRelease OR (sqMom>0 AND rising AND sqzOff))")
    print(f"  GUCLU:            {mom_cond1 and mom_cond2 and vol_break and rs_pass}")
    print(f"  ZAYIF:            {mom_cond1 and mom_cond2 and (not vol_break) and rs_pass}")
    print()


def analyze_ticker(ticker, df, xu_df, days=10, quiet=False, debug=False):
    """Full analysis pipeline for a single ticker. Returns summary dict or None."""
    if df is None or len(df) < 100:
        if not quiet:
            print(f"  {ticker}: Yetersiz veri ({len(df) if df is not None else 0} gun)")
        return None

    ind = compute_indicators(df, xu_df)
    signals = compute_signals(df, ind)
    cont = compute_continuation(signals)
    stops = compute_stops(df, ind, signals)

    if debug:
        display_debug(ticker, df, ind)

    if not quiet:
        display_results(ticker, df, ind, signals, cont, stops, days=days)

    # Return summary for scan mode
    today_sig = signals.iloc[-1]
    if today_sig:
        return {
            'ticker': ticker,
            'signal': today_sig,
            'devam': cont['devam_label'].iloc[-1],
            'regime': int(ind['regime'].iloc[-1]),
            'close': float(df['Close'].iloc[-1]),
            'stop': float(stops['stop'].iloc[-1]) if not np.isnan(stops['stop'].iloc[-1]) else None,
            'tp': float(stops['tp'].iloc[-1]) if not np.isnan(stops['tp'].iloc[-1]) else None,
            'rr': float(stops['rr'].iloc[-1]) if not np.isnan(stops['rr'].iloc[-1]) else None,
            'rs': float(ind['rs_score'].iloc[-1]),
        }
    return None


def display_csv(results):
    """Print results as CSV to stdout."""
    import csv
    import io
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(['Hisse', 'Sinyal', 'Devam', 'Rejim', 'Skor', 'Fiyat', 'Stop', 'TP', 'RR', 'RS'])
    for r in results:
        writer.writerow([
            r['ticker'],
            r['signal'],
            r['devam'] if r['devam'] else '',
            regime_name(r['regime']),
            r['regime'],
            f"{r['close']:.2f}",
            f"{r['stop']:.2f}" if r['stop'] else '',
            f"{r['tp']:.2f}" if r['tp'] else '',
            f"{r['rr']:.1f}" if r['rr'] else '',
            f"{r['rs']:.1f}",
        ])
    print(out.getvalue(), end='')


def main():
    parser = argparse.ArgumentParser(
        description='BIST Rejim Filtresi — PineScript v6 Replica')
    parser.add_argument('tickers', nargs='*', help='Ticker symbols (e.g. THYAO EREGL)')
    parser.add_argument('--all', action='store_true', help='Scan all BIST tickers')
    parser.add_argument('--days', type=int, default=10, help='Number of days to display (default: 10)')
    parser.add_argument('--csv', action='store_true', help='Output results as CSV')
    parser.add_argument('--output', '-o', type=str, help='Save CSV to file (implies --csv)')
    parser.add_argument('--debug', action='store_true', help='Show indicator values for Pine comparison')
    args = parser.parse_args()

    if args.output:
        args.csv = True

    if not args.tickers and not args.all:
        parser.print_help()
        sys.exit(1)

    # Determine tickers
    if args.all:
        tickers = get_all_bist_tickers()
    else:
        tickers = [t.upper() for t in args.tickers]

    # Fetch data — in CSV mode, redirect stdout to suppress progress messages
    if args.csv:
        import contextlib, io as _io
        with contextlib.redirect_stdout(_io.StringIO()):
            data = fetch_data(tickers, period='2y')
            xu_df = fetch_benchmark(period='2y')
    else:
        print("Veri cekiliyor...")
        data = fetch_data(tickers, period='2y')
        xu_df = fetch_benchmark(period='2y')

    if not data:
        if not args.csv:
            print("Hic veri cekilemedi!")
        sys.exit(1)

    # Analyze
    scan_mode = args.all or len(tickers) > 3
    results = []

    for ticker in tickers:
        df = data.get(ticker)
        if df is None:
            if not scan_mode and not args.csv:
                print(f"  {ticker}: Veri bulunamadi")
            continue

        result = analyze_ticker(
            ticker, df, xu_df,
            days=args.days,
            quiet=scan_mode or args.csv,
            debug=args.debug,
        )
        if result:
            results.append(result)

    if args.csv:
        if args.output:
            import io as _io2, csv as _csv2
            os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
            with open(args.output, 'w', newline='') as f:
                writer = _csv2.writer(f)
                writer.writerow(['Hisse', 'Sinyal', 'Devam', 'Rejim', 'Skor', 'Fiyat', 'Stop', 'TP', 'RR', 'RS'])
                for r in results:
                    writer.writerow([
                        r['ticker'], r['signal'],
                        r['devam'] if r['devam'] else '',
                        regime_name(r['regime']), r['regime'],
                        f"{r['close']:.2f}",
                        f"{r['stop']:.2f}" if r['stop'] else '',
                        f"{r['tp']:.2f}" if r['tp'] else '',
                        f"{r['rr']:.1f}" if r['rr'] else '',
                        f"{r['rs']:.1f}",
                    ])
            print(f"CSV kaydedildi: {args.output} ({len(results)} sinyal)")
        else:
            display_csv(results)
    elif scan_mode:
        display_scan_summary(results)


if __name__ == '__main__':
    main()
