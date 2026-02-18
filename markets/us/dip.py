"""
NOX Project — US Market DIP + INSTITUTIONAL Analiz
Haftalık EMA dip tarama + Institutional accumulation sinyali.
BIST'teki RECOVER (USD EMA bazlı) yerine:
  INSTITUTIONAL = Kurumsal birikim + hacim analizi + fiyat sıkışması

Sinyaller:
  DIP+  = DIP + rejim overlap
  DIP   = Güvenli haftalık dip
  DIP_E = Erken dip
  DIP_W = İzle (teyit bekliyor)
  INSTITUTIONAL = Kurumsal birikim tespiti
"""
import sys, os
_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _root not in sys.path:
    sys.path.insert(0, _root)

import numpy as np
import pandas as pd

from markets.us.config import (
    MIN_AVG_VOLUME_USD, RVOL_THRESH,
    DONUS_TP, PINK_STOP_MULT,
    PINK_EMA89, PINK_EMA144, PINK_RSI_LEN, PINK_RSI_DIV_LOOKBACK,
    PINK_TOUCH_WINDOW, PINK_TOUCH_COUNT,
    INST_MIN_SCORE, INST_ACCUM_WINDOW, INST_VOL_SPIKE, INST_PRICE_RANGE,
)
from core.config import (
    EMA_FAST, EMA_SLOW, ST_LEN, ST_MULT,
    ADX_LEN, ADX_SLOPE_LEN, ADX_SLOPE_THRESH, ADX_TREND, ADX_CHOPPY,
    ATR_LEN, RS_LEN1, RS_LEN2,
)
from core.indicators import (
    ema, sma, true_range, calc_atr_sma, calc_adx_ema, calc_supertrend,
    calc_wavetrend, calc_rsi_sma, calc_macd, resample_weekly,
    calc_overextended,
)

# Alias
MIN_AVG_VOLUME_TL = MIN_AVG_VOLUME_USD


# ══════════════════════════════════════════
# INSTITUTIONAL SIGNAL (RECOVER replacement)
# ══════════════════════════════════════════

def calc_institutional_criteria(df, verbose=False):
    """
    Institutional accumulation tespiti.
    BIST'teki RECOVER (USD EMA + likidite) yerine:
    1. Volume accumulation: Yükselen hacim + dar fiyat aralığı = birikim
    2. Dark pool prints: Anormal hacim spike'ları
    3. Higher lows on volume: Kurumsal destek
    4. MACD momentum shift: Sıfır çizgisi yakınında bullish cross
    5. Relative volume persistence: Sürekli yüksek hacim
    6. Price compression: BB squeeze + artan OBV
    """
    result = {
        'accumulation': False, 'vol_persistence': False,
        'higher_lows': False, 'macd_shift': False,
        'price_compression': False, 'dark_pool_vol': False,
        'obv_trend': False,
        'inst_score': 0, 'inst_tags': [],
    }

    c = df['Close']
    h = df['High']
    l = df['Low']
    v = df['Volume']
    n = len(df)

    if n < 60:
        return result

    # 1. ACCUMULATION: Dar range + yüksek hacim
    avg_vol_20 = v.rolling(20).mean()
    price_range = (h - l) / c
    avg_range = price_range.rolling(20).mean()
    narrow_range = price_range.iloc[-1] < avg_range.iloc[-1] * 0.7 if not pd.isna(avg_range.iloc[-1]) else False
    high_vol = v.iloc[-1] > avg_vol_20.iloc[-1] * INST_VOL_SPIKE if not pd.isna(avg_vol_20.iloc[-1]) else False
    vol_accum = sum(1 for i in range(-INST_ACCUM_WINDOW, 0)
                    if i >= -n and not pd.isna(avg_vol_20.iloc[i])
                    and v.iloc[i] > avg_vol_20.iloc[i] * 1.2
                    and price_range.iloc[i] < avg_range.iloc[i] * 0.8)
    if vol_accum >= 5:
        result['accumulation'] = True
        result['inst_tags'].append('ACCUM')
        result['inst_score'] += 12

    # 2. VOLUME PERSISTENCE: Son 10 günün 6+'sında ortalamanın üstünde hacim
    recent_above = sum(1 for i in range(-10, 0) if not pd.isna(avg_vol_20.iloc[i]) and v.iloc[i] > avg_vol_20.iloc[i])
    if recent_above >= 6:
        result['vol_persistence'] = True
        result['inst_tags'].append('VOL_PERS')
        result['inst_score'] += 8

    # 3. HIGHER LOWS on volume: Son 20 günde higher low + hacim artışı
    lows_20 = l.iloc[-20:]
    swing_lows = []
    for i in range(2, len(lows_20) - 2):
        if lows_20.iloc[i] <= lows_20.iloc[i-1] and lows_20.iloc[i] <= lows_20.iloc[i-2] and \
           lows_20.iloc[i] <= lows_20.iloc[i+1] and lows_20.iloc[i] <= lows_20.iloc[i+2]:
            swing_lows.append((i, lows_20.iloc[i]))
    if len(swing_lows) >= 2 and swing_lows[-1][1] > swing_lows[-2][1]:
        result['higher_lows'] = True
        result['inst_tags'].append('HL')
        result['inst_score'] += 10

    # 4. MACD SHIFT: Sıfır çizgisi yakınında bullish cross
    macd_data = calc_macd(c)
    if macd_data is not None:
        macd_line = macd_data['macd']
        signal_line = macd_data['signal']
        hist = macd_data['hist']
        if len(macd_line) > 2:
            near_zero = abs(macd_line.iloc[-1]) < c.iloc[-1] * 0.01
            bull_cross = macd_line.iloc[-1] > signal_line.iloc[-1] and macd_line.iloc[-2] <= signal_line.iloc[-2]
            hist_rising = hist.iloc[-1] > hist.iloc[-2] if len(hist) > 2 else False
            if bull_cross:
                result['macd_shift'] = True
                result['inst_tags'].append('MACD+' if not near_zero else 'MACD0+')
                result['inst_score'] += 12 if near_zero else 8
            elif hist_rising and near_zero:
                result['inst_tags'].append('MACD↑')
                result['inst_score'] += 5

    # 5. PRICE COMPRESSION: BB squeeze
    bb_mid = sma(c, 20)
    bb_std = c.rolling(20).std()
    bb_width = (bb_std * 2) / bb_mid
    if len(bb_width.dropna()) > 20:
        current_width = bb_width.iloc[-1]
        min_width_50 = bb_width.iloc[-50:].min() if n >= 50 else bb_width.min()
        if not pd.isna(current_width) and not pd.isna(min_width_50):
            if current_width < min_width_50 * 1.1:
                result['price_compression'] = True
                result['inst_tags'].append('SQUEEZE')
                result['inst_score'] += 10

    # 6. OBV TREND: On-Balance Volume artış trendi
    obv = (np.sign(c.diff()) * v).cumsum()
    if len(obv) >= 20:
        obv_ema10 = ema(obv, 10)
        obv_ema20 = ema(obv, 20)
        if obv_ema10.iloc[-1] > obv_ema20.iloc[-1] and obv.iloc[-1] > obv.iloc[-5]:
            result['obv_trend'] = True
            result['inst_tags'].append('OBV+')
            result['inst_score'] += 8

    # 7. DARK POOL VOLUME: Anormal tek-gün hacim spike'ı (son 10 gün)
    vol_std = v.rolling(50).std()
    for i in range(-10, 0):
        if not pd.isna(avg_vol_20.iloc[i]) and not pd.isna(vol_std.iloc[i]):
            if v.iloc[i] > avg_vol_20.iloc[i] + 2 * vol_std.iloc[i]:
                result['dark_pool_vol'] = True
                result['inst_tags'].append('DPOOL')
                result['inst_score'] += 10
                break

    return result


# ══════════════════════════════════════════
# PINK V2 CORE (DIP detection — same as BIST)
# ══════════════════════════════════════════

def _pink_hidden_bullish_div(price, rsi_s, lookback=10):
    """Hidden bullish divergence: price higher low + RSI lower low."""
    n = len(price)
    if n < lookback + 5:
        return False
    p = price.iloc[-lookback:]
    r = rsi_s.iloc[-lookback:]
    p_min_idx = p.idxmin()
    p_prev = price.iloc[-lookback * 2:-lookback] if n >= lookback * 2 else price.iloc[:n - lookback]
    if len(p_prev) < 5:
        return False
    p_prev_min = p_prev.min()
    p_cur_min = p.min()
    r_at_cur = r.loc[p_min_idx] if p_min_idx in r.index else r.iloc[0]
    prev_r_window = rsi_s.iloc[-lookback * 2:-lookback] if n >= lookback * 2 else rsi_s.iloc[:n - lookback]
    r_at_prev = prev_r_window.min() if len(prev_r_window) > 0 else 50
    return p_cur_min > p_prev_min and r_at_cur < r_at_prev


def calc_pink_v2(df, verbose=False):
    """Haftalık EMA dip tespiti — BIST ile aynı mantık."""
    result = {
        'pink_signal': None, 'pink_ma': None, 'pink_candle': None,
        'pink_rsi_div': False, 'pink_touches': 0,
        'pink_stop': None, 'pink_tp': None,
        'pink_detail': '', 'pink_tags': [],
    }

    wdf = resample_weekly(df)
    if len(wdf) < max(PINK_EMA144, 50):
        return result

    wc = wdf['Close']
    wh = wdf['High']
    wl = wdf['Low']
    wo = wdf['Open']

    ema_configs = [
        (13, 'EMA13'), (21, 'EMA21'), (34, 'EMA34'),
        (55, 'EMA55'), (PINK_EMA89, 'EMA89'),
        (100, 'EMA100'), (PINK_EMA144, 'EMA144'),
    ]

    best = None
    for period, label in ema_configs:
        ema_s = ema(wc, period)
        if pd.isna(ema_s.iloc[-1]):
            continue
        ema_val = ema_s.iloc[-1]

        # Fiyat EMA'ya dokunuyor mu? (Low <= EMA + %1 tolerance)
        touch = wl.iloc[-1] <= ema_val * 1.01 and wc.iloc[-1] >= ema_val * 0.98
        if not touch:
            continue

        # Touch count
        touch_mask = (wl <= ema_s * 1.01) & (wc >= ema_s * 0.98)
        recent_touches = touch_mask.iloc[-PINK_TOUCH_WINDOW:].sum() if len(touch_mask) >= PINK_TOUCH_WINDOW else touch_mask.sum()

        # Fresh touch (son 3 haftada yeni temas)
        fresh = not any(touch_mask.iloc[-i] for i in range(2, min(8, len(touch_mask))) if -i >= -len(touch_mask))

        # Reversal candle patterns
        body = abs(wc.iloc[-1] - wo.iloc[-1])
        total_range = wh.iloc[-1] - wl.iloc[-1]
        lower_wick = min(wo.iloc[-1], wc.iloc[-1]) - wl.iloc[-1]
        candle_type = None
        if total_range > 0:
            if lower_wick / total_range > 0.5 and body / total_range < 0.3:
                candle_type = "TOPAC"
            elif body / total_range < 0.15:
                candle_type = "DOJI"
            elif wc.iloc[-1] > wo.iloc[-1] and body / total_range > 0.5:
                candle_type = "Vol+"

        # RSI
        rsi = calc_rsi_sma(wc, PINK_RSI_LEN)
        rsi_val = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50
        rsi_div = _pink_hidden_bullish_div(wc, rsi, PINK_RSI_DIV_LOOKBACK)

        # Trend confirmation
        ema_fast_w = ema(wc, 13)
        trend_ok = wc.iloc[-1] > ema_fast_w.iloc[-1] if not pd.isna(ema_fast_w.iloc[-1]) else False

        # Signal classification
        confirmed = candle_type is not None or rsi_div
        sig = None
        if fresh and confirmed and trend_ok and rsi_val < 60:
            sig = "DIP"
        elif fresh and (candle_type or rsi_div) and rsi_val < 65:
            sig = "DIP_E"
        elif touch:
            sig = "DIP_W"

        if sig and (best is None or ["DIP", "DIP_E", "DIP_W"].index(sig) < ["DIP", "DIP_E", "DIP_W"].index(best.get('pink_signal', 'DIP_W'))):
            # TP: sonraki swing high
            swing_highs = wh.iloc[-30:]
            tp = swing_highs.max() if len(swing_highs) > 0 else wc.iloc[-1] * 1.1
            # Stop: son swing low veya EMA altı
            stop = wl.iloc[-5:].min()

            detail_parts = [f"{label}({ema_val:.2f})"]
            if rsi_div:
                detail_parts.append("RSI-Div")
            if candle_type:
                detail_parts.append(candle_type)
            if trend_ok:
                detail_parts.append("TrendK")
            detail_parts.append(f"RSI:{rsi_val:.0f}")
            detail_parts.append(f"T:{int(recent_touches)}x")

            tags = []
            if rsi_val < 40:
                tags.append("RSI<40")
            if rsi_div:
                tags.append("DIV")

            best = {
                'pink_signal': sig, 'pink_ma': label,
                'pink_candle': candle_type, 'pink_rsi_div': rsi_div,
                'pink_touches': int(recent_touches),
                'pink_stop': float(stop), 'pink_tp': float(tp),
                'pink_detail': " ".join(detail_parts),
                'pink_tags': tags,
            }

    return best if best else result


# ══════════════════════════════════════════
# ANALYZE DIP (main entry point)
# ══════════════════════════════════════════

def analyze_dip(ticker, df, xu_df, usd_df=None, dbg=None):
    """US DIP analizi — haftalık EMA dip + INSTITUTIONAL."""
    try:
        if dbg:
            dbg['total'] = dbg.get('total', 0) + 1

        n = len(df)
        c = df['Close']
        h = df['High']
        l = df['Low']
        v = df['Volume']

        # ATR
        atr = calc_atr_sma(df, ATR_LEN)
        atr_val = atr.iloc[-1]
        if pd.isna(atr_val) or atr_val == 0:
            return None

        close_price = float(c.iloc[-1])

        # Volume filter
        avg_turnover = (c * v).iloc[-20:].mean()
        if pd.isna(avg_turnover) or avg_turnover < MIN_AVG_VOLUME_TL:
            return None

        rvol = float(v.iloc[-1] / v.iloc[-20:].mean()) if v.iloc[-20:].mean() > 0 else 0

        # Regime
        ema_f = ema(c, EMA_FAST)
        ema_s = ema(c, EMA_SLOW)
        adx_data = calc_adx_ema(df, ADX_LEN)
        adx_val = adx_data['adx'].iloc[-1] if not pd.isna(adx_data['adx'].iloc[-1]) else 15
        adx_slope = (adx_data['adx'].iloc[-1] - adx_data['adx'].iloc[-ADX_SLOPE_LEN]) / ADX_SLOPE_LEN if n > ADX_SLOPE_LEN else 0

        st_data = calc_supertrend(df, ST_LEN, ST_MULT)
        super_trend_up = bool(st_data['trend'].iloc[-1] == 1) if not pd.isna(st_data['trend'].iloc[-1]) else False
        ema_trend_up = float(ema_f.iloc[-1]) > float(ema_s.iloc[-1]) if not pd.isna(ema_f.iloc[-1]) else False

        if adx_val >= ADX_TREND and ema_trend_up and super_trend_up:
            regime = 3; regime_name = "FULL_TREND"
        elif adx_val >= ADX_TREND or (ema_trend_up and super_trend_up):
            regime = 2; regime_name = "TREND"
        elif adx_val >= ADX_CHOPPY:
            regime = 1; regime_name = "GRI_BOLGE"
        else:
            regime = 0; regime_name = "CHOPPY"

        # RS Score
        rs_score = 0.0
        if xu_df is not None and len(xu_df) >= RS_LEN2:
            stock_ret1 = (c.iloc[-1] / c.iloc[-RS_LEN1] - 1) * 100 if n > RS_LEN1 else 0
            stock_ret2 = (c.iloc[-1] / c.iloc[-RS_LEN2] - 1) * 100 if n > RS_LEN2 else 0
            bench_ret1 = (xu_df['Close'].iloc[-1] / xu_df['Close'].iloc[-RS_LEN1] - 1) * 100
            bench_ret2 = (xu_df['Close'].iloc[-1] / xu_df['Close'].iloc[-RS_LEN2] - 1) * 100
            rs_score = round(((stock_ret1 - bench_ret1) + (stock_ret2 - bench_ret2)) / 2, 1)

        # Quality score
        quality = 0
        if ema_trend_up: quality += 20
        if super_trend_up: quality += 15
        if adx_val >= ADX_TREND: quality += 15
        if rvol >= 1.0: quality += 10
        if rvol >= 1.5: quality += 10
        if adx_slope > ADX_SLOPE_THRESH: quality += 10
        if rs_score > 0: quality += 10
        if rs_score > RS_THRESHOLD: quality += 10
        quality = min(quality, 100)

        # Pink V2 (DIP detection)
        pink = calc_pink_v2(df)
        pink_signal = pink.get('pink_signal')

        # Institutional criteria
        inst = calc_institutional_criteria(df)

        # Signal decision
        signal = None

        if pink_signal:
            signal = pink_signal
            # Overlap check → DIP+
            wt = calc_wavetrend(df)
            wt_recent = bool(wt['wt_recent'].iloc[-1])
            ema55_cross = (c > ema_s) & (c.shift(1) <= ema_s.shift(1))
            recent_e55 = any(ema55_cross.iloc[-i] for i in range(1, 11) if -i >= -n)
            has_overlap = (wt_recent and recent_e55) or (wt_recent and regime >= 2)
            if has_overlap and pink_signal in ("DIP", "DIP_E"):
                signal = "DIP+"
        elif inst['inst_score'] >= INST_MIN_SCORE:
            signal = "INSTITUTIONAL"
        else:
            return None

        if signal is None:
            return None

        if dbg:
            dbg['signal'] = dbg.get('signal', 0) + 1

        # Stop / TP
        if signal == "INSTITUTIONAL":
            stop_price = l.iloc[-20:].min()
            tp_price = h.iloc[-50:].max() if n >= 50 else close_price * 1.15
            tp_src = "SWING"
        else:
            stop_price = pink['pink_stop'] if pink['pink_stop'] else close_price - atr_val * PINK_STOP_MULT
            tp_price = pink['pink_tp'] if pink['pink_tp'] else close_price + atr_val * DONUS_TP
            tp_src = "SWING" if pink.get('pink_ma') in ("EMA13", "EMA21", "EMA34") else "EMA21"

        risk = close_price - stop_price
        reward = tp_price - close_price
        rr = reward / risk if risk > 0 else 0

        overext = calc_overextended(df)

        detail = pink.get('pink_detail', '')
        if signal == "INSTITUTIONAL":
            detail = f"INST: {','.join(inst['inst_tags'])}"

        return {
            'ticker': ticker, 'signal': signal, 'regime': regime_name,
            'regime_score': regime, 'close': round(close_price, 2),
            'stop': round(stop_price, 2), 'tp': round(tp_price, 2),
            'tp_src': tp_src, 'rr': round(rr, 2), 'atr': round(atr_val, 2),
            'quality': int(quality), 'rs_score': round(rs_score, 1),
            'rvol': round(rvol, 1),
            'pink_ma': pink.get('pink_ma'),
            'pink_candle': pink.get('pink_candle'),
            'pink_rsi_div': pink.get('pink_rsi_div', False),
            'pink_touches': pink.get('pink_touches', 0),
            'pink_detail': detail,
            'pink_tags': pink.get('pink_tags', []),
            'kc_score': inst['inst_score'], 'kc_tags': inst['inst_tags'],
            'turnover_m': round(avg_turnover / 1e6, 1),
            'overext_score': overext['overext_score'],
            'overext_tags': overext['overext_tags'],
            'overext_warning': overext['overext_warning'],
        }
    except Exception as e:
        if dbg:
            dbg['exception'] = dbg.get('exception', 0) + 1
        return None
