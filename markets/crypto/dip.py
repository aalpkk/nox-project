"""
NOX Project — Crypto DIP + WHALE Analiz
Haftalık EMA dip tarama + Whale/Smart Money accumulation sinyali.

Sinyaller:
  DIP+  = DIP + rejim overlap
  DIP   = Güvenli haftalık dip
  DIP_E = Erken dip
  DIP_W = İzle (teyit bekliyor)
  WHALE = 🐋 Whale birikim tespiti (RECOVER/INSTITUTIONAL yerine)
"""
import sys, os
_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _root not in sys.path:
    sys.path.insert(0, _root)

import numpy as np
import pandas as pd

from markets.crypto.config import (
    MIN_AVG_VOLUME_USD, RVOL_THRESH,
    DONUS_TP, PINK_STOP_MULT,
    PINK_EMA89, PINK_EMA144, PINK_RSI_LEN, PINK_RSI_DIV_LOOKBACK,
    PINK_TOUCH_WINDOW, PINK_TOUCH_COUNT,
    WHALE_MIN_SCORE, WHALE_VOL_SPIKE, WHALE_ACCUM_WINDOW,
)
from core.config import (
    EMA_FAST, EMA_SLOW, ST_LEN, ST_MULT,
    ADX_LEN, ADX_SLOPE_LEN, ADX_SLOPE_THRESH, ADX_TREND, ADX_CHOPPY,
    ATR_LEN, RS_LEN1, RS_LEN2, BB_LEN, BB_MULT,
)
from core.indicators import (
    ema, sma, true_range, calc_atr_sma, calc_adx_ema, calc_supertrend,
    calc_wavetrend, calc_rsi_sma, calc_macd, resample_weekly,
    calc_overextended,
)

MIN_AVG_VOLUME_TL = MIN_AVG_VOLUME_USD


# ══════════════════════════════════════════
# WHALE SIGNAL
# ══════════════════════════════════════════

def calc_whale_criteria(df, verbose=False):
    """
    Whale / Smart Money accumulation tespiti.
    Kripto'ya özel: on-chain proxy'ler + volume profiling.

    1. Whale accumulation: Büyük hacim + dar range + fiyat tutunma
    2. Volume climax: Anormal hacim spike sonrası reversal
    3. Absorption: Satış baskısına rağmen fiyat düşmüyor
    4. MACD divergence: Fiyat düşerken MACD yükseliyor
    5. OBV breakout: On-Balance Volume yeni zirve
    6. Funding reset: Uzun süre negatif sonrası nötralize (proxy)
    7. Range compression: Volatilite sıkışması
    """
    result = {
        'whale_accum': False, 'vol_climax': False,
        'absorption': False, 'macd_div': False,
        'obv_breakout': False, 'funding_reset': False,
        'range_compress': False,
        'whale_score': 0, 'whale_tags': [],
    }

    c = df['Close']
    h = df['High']
    l = df['Low']
    o = df['Open']
    v = df['Volume']
    n = len(df)

    if n < 60:
        return result

    avg_vol_20 = v.rolling(20).mean()
    price_range = (h - l) / c
    avg_range = price_range.rolling(20).mean()

    # 1. WHALE ACCUMULATION: Yüksek hacim + dar range + yukarı kapanış
    accum_days = 0
    for i in range(-WHALE_ACCUM_WINDOW, 0):
        if i >= -n and not pd.isna(avg_vol_20.iloc[i]) and not pd.isna(avg_range.iloc[i]):
            high_vol = v.iloc[i] > avg_vol_20.iloc[i] * 1.3
            narrow = price_range.iloc[i] < avg_range.iloc[i] * 0.8
            bullish_close = c.iloc[i] > (o.iloc[i] + c.iloc[i]) / 2  # üst yarıda kapanış
            if high_vol and narrow and bullish_close:
                accum_days += 1
    if accum_days >= 4:
        result['whale_accum'] = True
        result['whale_tags'].append('🐋ACCUM')
        result['whale_score'] += 12

    # 2. VOLUME CLIMAX: Son 20 günde en yüksek hacim + sonra reversal
    vol_20 = v.iloc[-20:]
    max_vol_idx = vol_20.idxmax()
    if max_vol_idx is not None:
        max_pos = vol_20.index.get_loc(max_vol_idx)
        vol_std = v.rolling(50).std()
        if not pd.isna(vol_std.iloc[-1]) and not pd.isna(avg_vol_20.iloc[-1]):
            is_climax = vol_20.iloc[max_pos] > avg_vol_20.iloc[-1] + 2.5 * vol_std.iloc[-1]
            # Climax'tan sonra fiyat yükseldi mi?
            if is_climax and max_pos < len(vol_20) - 2:
                post_climax_return = (c.iloc[-1] / c.loc[max_vol_idx] - 1) if c.loc[max_vol_idx] > 0 else 0
                if post_climax_return > 0:
                    result['vol_climax'] = True
                    result['whale_tags'].append('CLIMAX')
                    result['whale_score'] += 10

    # 3. ABSORPTION: Son 10 günde satış hacmi yüksek ama fiyat düşmüyor
    sell_vol_days = sum(1 for i in range(-10, 0)
                        if c.iloc[i] < o.iloc[i]  # kırmızı mum
                        and not pd.isna(avg_vol_20.iloc[i])
                        and v.iloc[i] > avg_vol_20.iloc[i] * 1.2)
    price_change_10d = (c.iloc[-1] / c.iloc[-10] - 1) * 100 if n >= 10 else 0
    if sell_vol_days >= 4 and price_change_10d > -2:
        result['absorption'] = True
        result['whale_tags'].append('ABSORB')
        result['whale_score'] += 10

    # 4. MACD DIVERGENCE: Fiyat lower low + MACD higher low
    macd_data = calc_macd(c)
    if macd_data is not None:
        macd_line = macd_data['macd']
        signal_line = macd_data['signal']
        hist = macd_data['hist']
        if len(macd_line) > 30:
            # Son 30 günde fiyat lower low
            price_30 = c.iloc[-30:]
            macd_30 = macd_line.iloc[-30:]
            mid = len(price_30) // 2
            p_first_half_low = price_30.iloc[:mid].min()
            p_second_half_low = price_30.iloc[mid:].min()
            m_first_half_low = macd_30.iloc[:mid].min()
            m_second_half_low = macd_30.iloc[mid:].min()

            if p_second_half_low < p_first_half_low and m_second_half_low > m_first_half_low:
                result['macd_div'] = True
                result['whale_tags'].append('MACD_DIV')
                result['whale_score'] += 12

            # Bullish cross
            if len(hist) > 2 and macd_line.iloc[-1] > signal_line.iloc[-1] and macd_line.iloc[-2] <= signal_line.iloc[-2]:
                near_zero = abs(macd_line.iloc[-1]) < c.iloc[-1] * 0.02
                result['whale_tags'].append('MACD0+' if near_zero else 'MACD+')
                result['whale_score'] += 8 if near_zero else 5

    # 5. OBV BREAKOUT: OBV yeni 20 günlük zirve
    obv = (np.sign(c.diff()) * v).cumsum()
    if len(obv) >= 20:
        obv_20_max = obv.iloc[-20:-1].max()
        if obv.iloc[-1] > obv_20_max and not pd.isna(obv_20_max):
            result['obv_breakout'] = True
            result['whale_tags'].append('OBV↑')
            result['whale_score'] += 8

    # 6. FUNDING RESET PROXY: Uzun düşüş sonrası stabilizasyon
    # (Gerçek funding rate yok, proxy: RSI uzun süre oversold sonrası recovery)
    rsi = calc_rsi_sma(c, 14)
    if len(rsi) >= 20:
        oversold_days = sum(1 for i in range(-20, -5) if not pd.isna(rsi.iloc[i]) and rsi.iloc[i] < 35)
        current_rsi = rsi.iloc[-1]
        if oversold_days >= 5 and current_rsi > 40 and current_rsi < 55:
            result['funding_reset'] = True
            result['whale_tags'].append('RSI_RESET')
            result['whale_score'] += 8

    # 7. RANGE COMPRESSION: BB width minimum
    bb_mid = sma(c, 20)
    bb_std = c.rolling(20).std()
    bb_width = (bb_std * 2) / bb_mid
    if len(bb_width.dropna()) > 50:
        current_width = bb_width.iloc[-1]
        pct_rank = (bb_width.iloc[-100:] < current_width).sum() / min(100, len(bb_width)) if len(bb_width) >= 20 else 0.5
        if not pd.isna(current_width) and pct_rank < 0.1:
            result['range_compress'] = True
            result['whale_tags'].append('SQUEEZE')
            result['whale_score'] += 10

    return result


# ══════════════════════════════════════════
# PINK V2 (DIP detection — same as others)
# ══════════════════════════════════════════

def _pink_hidden_bullish_div(price, rsi_s, lookback=10):
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
    """Haftalık EMA dip tespiti."""
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

        touch = wl.iloc[-1] <= ema_val * 1.015 and wc.iloc[-1] >= ema_val * 0.975
        if not touch:
            continue

        touch_mask = (wl <= ema_s * 1.015) & (wc >= ema_s * 0.975)
        recent_touches = touch_mask.iloc[-PINK_TOUCH_WINDOW:].sum() if len(touch_mask) >= PINK_TOUCH_WINDOW else touch_mask.sum()
        fresh = not any(touch_mask.iloc[-i] for i in range(2, min(8, len(touch_mask))) if -i >= -len(touch_mask))

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

        rsi = calc_rsi_sma(wc, PINK_RSI_LEN)
        rsi_val = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50
        rsi_div = _pink_hidden_bullish_div(wc, rsi, PINK_RSI_DIV_LOOKBACK)

        ema_fast_w = ema(wc, 13)
        trend_ok = wc.iloc[-1] > ema_fast_w.iloc[-1] if not pd.isna(ema_fast_w.iloc[-1]) else False

        confirmed = candle_type is not None or rsi_div
        sig = None
        if fresh and confirmed and trend_ok and rsi_val < 60:
            sig = "DIP"
        elif fresh and (candle_type or rsi_div) and rsi_val < 65:
            sig = "DIP_E"
        elif touch:
            sig = "DIP_W"

        if sig and (best is None or ["DIP", "DIP_E", "DIP_W"].index(sig) < ["DIP", "DIP_E", "DIP_W"].index(best.get('pink_signal', 'DIP_W'))):
            swing_highs = wh.iloc[-30:]
            tp = swing_highs.max() if len(swing_highs) > 0 else wc.iloc[-1] * 1.15
            stop = wl.iloc[-5:].min()

            detail_parts = [f"{label}({ema_val:.2f})"]
            if rsi_div: detail_parts.append("RSI-Div")
            if candle_type: detail_parts.append(candle_type)
            if trend_ok: detail_parts.append("TrendK")
            detail_parts.append(f"RSI:{rsi_val:.0f}")
            detail_parts.append(f"T:{int(recent_touches)}x")

            tags = []
            if rsi_val < 40: tags.append("RSI<40")
            if rsi_div: tags.append("DIV")

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
# ANALYZE DIP
# ══════════════════════════════════════════

def analyze_dip(ticker, df, xu_df, usd_df=None, dbg=None):
    """Crypto DIP analizi — haftalık EMA dip + WHALE."""
    try:
        if dbg:
            dbg['total'] = dbg.get('total', 0) + 1

        n = len(df)
        c = df['Close']
        h = df['High']
        l = df['Low']
        v = df['Volume']

        atr = calc_atr_sma(df, ATR_LEN)
        atr_val = atr.iloc[-1]
        if pd.isna(atr_val) or atr_val == 0:
            return None

        close_price = float(c.iloc[-1])

        avg_turnover = (c * v).iloc[-20:].mean()
        if pd.isna(avg_turnover) or avg_turnover < MIN_AVG_VOLUME_TL:
            return None

        rvol = float(v.iloc[-1] / v.iloc[-20:].mean()) if v.iloc[-20:].mean() > 0 else 0

        # Regime
        ema_f = ema(c, EMA_FAST)
        ema_s = ema(c, EMA_SLOW)
        adx_data = calc_adx_ema(df, ADX_LEN)
        adx_val = adx_data['adx'].iloc[-1] if not pd.isna(adx_data['adx'].iloc[-1]) else 15
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

        # RS vs BTC
        rs_score = 0.0
        if xu_df is not None and len(xu_df) >= RS_LEN2:
            stock_ret1 = (c.iloc[-1] / c.iloc[-RS_LEN1] - 1) * 100 if n > RS_LEN1 else 0
            stock_ret2 = (c.iloc[-1] / c.iloc[-RS_LEN2] - 1) * 100 if n > RS_LEN2 else 0
            bench_ret1 = (xu_df['Close'].iloc[-1] / xu_df['Close'].iloc[-RS_LEN1] - 1) * 100
            bench_ret2 = (xu_df['Close'].iloc[-1] / xu_df['Close'].iloc[-RS_LEN2] - 1) * 100
            rs_score = round(((stock_ret1 - bench_ret1) + (stock_ret2 - bench_ret2)) / 2, 1)

        # Quality
        quality = 0
        if ema_trend_up: quality += 20
        if super_trend_up: quality += 15
        if adx_val >= ADX_TREND: quality += 15
        if rvol >= 1.0: quality += 10
        if rvol >= 1.5: quality += 10
        quality = min(quality, 100)

        # Pink V2
        pink = calc_pink_v2(df)
        pink_signal = pink.get('pink_signal')

        # Whale criteria
        whale = calc_whale_criteria(df)

        signal = None
        if pink_signal:
            signal = pink_signal
            wt = calc_wavetrend(df)
            wt_recent = bool(wt['wt_recent'].iloc[-1])
            ema55_cross = (c > ema_s) & (c.shift(1) <= ema_s.shift(1))
            recent_e55 = any(ema55_cross.iloc[-i] for i in range(1, 11) if -i >= -n)
            has_overlap = (wt_recent and recent_e55) or (wt_recent and regime >= 2)
            if has_overlap and pink_signal in ("DIP", "DIP_E"):
                signal = "DIP+"
        elif whale['whale_score'] >= WHALE_MIN_SCORE:
            signal = "WHALE"
        else:
            return None

        if dbg:
            dbg['signal'] = dbg.get('signal', 0) + 1

        # Stop / TP
        if signal == "WHALE":
            stop_price = l.iloc[-20:].min()
            tp_price = h.iloc[-50:].max() if n >= 50 else close_price * 1.20
            tp_src = "SWING"
        else:
            stop_price = pink['pink_stop'] if pink['pink_stop'] else close_price - atr_val * PINK_STOP_MULT
            tp_price = pink['pink_tp'] if pink['pink_tp'] else close_price + atr_val * DONUS_TP
            tp_src = "SWING"

        risk = close_price - stop_price
        reward = tp_price - close_price
        rr = reward / risk if risk > 0 else 0

        overext = calc_overextended(df)

        detail = pink.get('pink_detail', '')
        if signal == "WHALE":
            detail = f"WHALE: {','.join(whale['whale_tags'])}"

        return {
            'ticker': ticker, 'signal': signal, 'regime': regime_name,
            'regime_score': regime, 'close': round(close_price, 2),
            'stop': round(stop_price, 2), 'tp': round(tp_price, 2),
            'tp_src': tp_src, 'rr': round(rr, 2), 'atr': round(atr_val, 2),
            'quality': int(quality), 'rs_score': round(rs_score, 1),
            'rvol': round(rvol, 1),
            'pink_detail': detail,
            'pink_tags': pink.get('pink_tags', []),
            'kc_score': whale['whale_score'], 'kc_tags': whale['whale_tags'],
            'turnover_m': round(avg_turnover / 1e6, 1),
            'overext_score': overext['overext_score'],
            'overext_tags': overext['overext_tags'],
            'overext_warning': overext['overext_warning'],
        }
    except Exception as e:
        if dbg:
            dbg['exception'] = dbg.get('exception', 0) + 1
        return None
