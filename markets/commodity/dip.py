"""
NOX Project — Commodity DIP + SUPPLY Analiz
Haftalık EMA dip tarama + Arz/Talep dinamikleri sinyali.

Sinyaller:
  DIP+  = DIP + rejim overlap
  DIP   = Güvenli haftalık dip
  DIP_E = Erken dip
  DIP_W = İzle
  SUPPLY = 📦 Arz/Talep dengesizliği tespiti

Emtiaya özel: mevsimsellik, contango/backwardation proxy,
stok tükenme işaretleri, üretici hedge davranışı proxy'leri.
"""
import sys, os
_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _root not in sys.path:
    sys.path.insert(0, _root)

import numpy as np
import pandas as pd

from markets.commodity.config import (
    MIN_AVG_VOLUME_USD, RVOL_THRESH,
    DONUS_TP, PINK_STOP_MULT,
    PINK_EMA89, PINK_EMA144, PINK_RSI_LEN, PINK_RSI_DIV_LOOKBACK,
    PINK_TOUCH_WINDOW, PINK_TOUCH_COUNT,
    SUPPLY_MIN_SCORE, SUPPLY_SEASONAL_WINDOW,
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
# SUPPLY/DEMAND SIGNAL
# ══════════════════════════════════════════

def calc_supply_criteria(df, verbose=False):
    """
    Emtia arz/talep dengesizliği tespiti.

    1. Backwardation proxy: Kısa vadeli fiyat > uzun vadeli EMA (talep > arz)
    2. Seasonal strength: Yılın aynı döneminde tarihsel olarak güçlü mü?
    3. Inventory draw proxy: Sürekli artan fiyat + artan hacim = stok azalıyor
    4. Producer hedge unwind: Fiyat düşüşü sonrası recovery + düşük hacim
    5. Mean reversion setup: Uzun vadeli ortalamadan aşırı sapma sonrası dönüş
    6. Momentum divergence: Fiyat vs OBV/MACD ayrışması
    7. Volatility expansion: Sessiz dönem sonrası hacim patlaması
    """
    result = {
        'backwardation': False, 'seasonal': False,
        'inventory_draw': False, 'hedge_unwind': False,
        'mean_reversion': False, 'momentum_div': False,
        'vol_expansion': False,
        'supply_score': 0, 'supply_tags': [],
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

    # 1. BACKWARDATION PROXY: Kısa EMA > Uzun EMA spread genişliyor
    ema_21 = ema(c, 21)
    ema_55 = ema(c, 55)
    ema_144 = ema(c, 144)
    if not pd.isna(ema_21.iloc[-1]) and not pd.isna(ema_55.iloc[-1]) and not pd.isna(ema_144.iloc[-1]):
        spread_now = (ema_21.iloc[-1] - ema_55.iloc[-1]) / ema_55.iloc[-1] * 100
        spread_5ago = (ema_21.iloc[-5] - ema_55.iloc[-5]) / ema_55.iloc[-5] * 100 if n > 5 else 0
        above_long = c.iloc[-1] > ema_144.iloc[-1]
        if spread_now > 0 and spread_now > spread_5ago and above_long:
            result['backwardation'] = True
            result['supply_tags'].append('BACKW')
            result['supply_score'] += 10

    # 2. SEASONAL STRENGTH: Yılın bu döneminde fiyat tarihsel olarak yükseler mi?
    if n >= SUPPLY_SEASONAL_WINDOW:
        current_month_return = (c.iloc[-1] / c.iloc[-21] - 1) * 100 if n > 21 else 0
        # 1 yıl önceki aynı dönem
        year_ago_start = max(0, n - SUPPLY_SEASONAL_WINDOW - 21)
        year_ago_end = max(0, n - SUPPLY_SEASONAL_WINDOW)
        if year_ago_end > year_ago_start and year_ago_end < n:
            year_ago_return = (c.iloc[year_ago_end] / c.iloc[year_ago_start] - 1) * 100
            if year_ago_return > 2 and current_month_return > 0:
                result['seasonal'] = True
                result['supply_tags'].append('SEASONAL')
                result['supply_score'] += 7

    # 3. INVENTORY DRAW PROXY: Artan fiyat + artan hacim = stok azalması
    price_trend_20 = (c.iloc[-1] / c.iloc[-20] - 1) * 100 if n > 20 else 0
    vol_trend_20 = (avg_vol_20.iloc[-1] / avg_vol_20.iloc[-20] - 1) * 100 if n > 20 and not pd.isna(avg_vol_20.iloc[-20]) and avg_vol_20.iloc[-20] > 0 else 0
    if price_trend_20 > 3 and vol_trend_20 > 10:
        result['inventory_draw'] = True
        result['supply_tags'].append('INV_DRAW')
        result['supply_score'] += 12

    # 4. PRODUCER HEDGE UNWIND: Düşüş sonrası recovery, düşen hacimle
    if n >= 40:
        drawdown_20_40 = (c.iloc[-20] / c.iloc[-40] - 1) * 100
        recovery_0_20 = (c.iloc[-1] / c.iloc[-20] - 1) * 100
        vol_declining = avg_vol_20.iloc[-1] < avg_vol_20.iloc[-10] if not pd.isna(avg_vol_20.iloc[-10]) else False
        if drawdown_20_40 < -5 and recovery_0_20 > 2 and vol_declining:
            result['hedge_unwind'] = True
            result['supply_tags'].append('HEDGE_UW')
            result['supply_score'] += 8

    # 5. MEAN REVERSION: 200 EMA'dan aşırı sapma sonrası dönüş
    ema_200 = ema(c, 200)
    if not pd.isna(ema_200.iloc[-1]) and ema_200.iloc[-1] > 0:
        deviation = (c.iloc[-1] / ema_200.iloc[-1] - 1) * 100
        # 200 EMA altında ama yaklaşıyor
        if -15 < deviation < -3:
            dev_5ago = (c.iloc[-5] / ema_200.iloc[-5] - 1) * 100 if not pd.isna(ema_200.iloc[-5]) else deviation
            if deviation > dev_5ago:  # sapma azalıyor = dönüş
                result['mean_reversion'] = True
                result['supply_tags'].append('MR_200')
                result['supply_score'] += 10

    # 6. MOMENTUM DIVERGENCE: Fiyat düşerken MACD yükseliyor
    macd_data = calc_macd(c)
    if macd_data is not None:
        macd_line = macd_data['macd']
        signal_line = macd_data['signal']
        hist = macd_data['hist']
        if len(macd_line) > 30:
            price_30 = c.iloc[-30:]
            macd_30 = macd_line.iloc[-30:]
            mid = len(price_30) // 2
            p_first_low = price_30.iloc[:mid].min()
            p_second_low = price_30.iloc[mid:].min()
            m_first_low = macd_30.iloc[:mid].min()
            m_second_low = macd_30.iloc[mid:].min()

            if p_second_low < p_first_low and m_second_low > m_first_low:
                result['momentum_div'] = True
                result['supply_tags'].append('MACD_DIV')
                result['supply_score'] += 10

            # Bullish cross
            if len(hist) > 2 and macd_line.iloc[-1] > signal_line.iloc[-1] and macd_line.iloc[-2] <= signal_line.iloc[-2]:
                near_zero = abs(macd_line.iloc[-1]) < c.iloc[-1] * 0.015
                result['supply_tags'].append('MACD0+' if near_zero else 'MACD+')
                result['supply_score'] += 8 if near_zero else 5

    # 7. VOLATILITY EXPANSION: BB squeeze sonrası volume patlaması
    bb_std = c.rolling(20).std()
    bb_mid = sma(c, 20)
    bb_width = (bb_std * 2) / bb_mid if bb_mid is not None else None
    if bb_width is not None and len(bb_width.dropna()) > 50:
        # Son 50 günde en düşük BB width
        recent_min = bb_width.iloc[-50:].min()
        width_5ago = bb_width.iloc[-5] if not pd.isna(bb_width.iloc[-5]) else bb_width.iloc[-1]
        current = bb_width.iloc[-1]
        # Squeeze'den çıkış: width genişliyor + hacim artıyor
        if not pd.isna(current) and not pd.isna(width_5ago) and not pd.isna(recent_min):
            was_squeezed = width_5ago < recent_min * 1.15
            expanding = current > width_5ago * 1.2
            vol_up = v.iloc[-1] > avg_vol_20.iloc[-1] * 1.3 if not pd.isna(avg_vol_20.iloc[-1]) else False
            if was_squeezed and expanding and vol_up:
                result['vol_expansion'] = True
                result['supply_tags'].append('VOL_EXP')
                result['supply_score'] += 8

    return result


# ══════════════════════════════════════════
# PINK V2 (DIP) — aynı mantık
# ══════════════════════════════════════════

def _pink_hidden_bullish_div(price, rsi_s, lookback=10):
    n = len(price)
    if n < lookback + 5: return False
    p = price.iloc[-lookback:]
    p_prev = price.iloc[-lookback*2:-lookback] if n >= lookback*2 else price.iloc[:n-lookback]
    if len(p_prev) < 5: return False
    p_min_idx = p.idxmin()
    r_at_cur = rsi_s.loc[p_min_idx] if p_min_idx in rsi_s.index else rsi_s.iloc[-lookback]
    prev_r = rsi_s.iloc[-lookback*2:-lookback] if n >= lookback*2 else rsi_s.iloc[:n-lookback]
    r_at_prev = prev_r.min() if len(prev_r) > 0 else 50
    return p.min() > p_prev.min() and r_at_cur < r_at_prev


def calc_pink_v2(df):
    result = {'pink_signal': None, 'pink_ma': None, 'pink_candle': None,
              'pink_rsi_div': False, 'pink_touches': 0,
              'pink_stop': None, 'pink_tp': None, 'pink_detail': '', 'pink_tags': []}
    wdf = resample_weekly(df)
    if len(wdf) < max(PINK_EMA144, 50): return result
    wc, wh, wl, wo = wdf['Close'], wdf['High'], wdf['Low'], wdf['Open']

    best = None
    for period, label in [(13,'EMA13'),(21,'EMA21'),(34,'EMA34'),(55,'EMA55'),
                           (PINK_EMA89,'EMA89'),(100,'EMA100'),(PINK_EMA144,'EMA144')]:
        ema_s = ema(wc, period)
        if pd.isna(ema_s.iloc[-1]): continue
        ev = ema_s.iloc[-1]
        if not (wl.iloc[-1] <= ev*1.01 and wc.iloc[-1] >= ev*0.98): continue

        touch_mask = (wl <= ema_s*1.01) & (wc >= ema_s*0.98)
        touches = touch_mask.iloc[-PINK_TOUCH_WINDOW:].sum() if len(touch_mask) >= PINK_TOUCH_WINDOW else touch_mask.sum()
        fresh = not any(touch_mask.iloc[-i] for i in range(2, min(8, len(touch_mask))) if -i >= -len(touch_mask))

        body = abs(wc.iloc[-1] - wo.iloc[-1])
        tr = wh.iloc[-1] - wl.iloc[-1]
        lw = min(wo.iloc[-1], wc.iloc[-1]) - wl.iloc[-1]
        ct = None
        if tr > 0:
            if lw/tr > 0.5 and body/tr < 0.3: ct = "TOPAC"
            elif body/tr < 0.15: ct = "DOJI"
            elif wc.iloc[-1] > wo.iloc[-1] and body/tr > 0.5: ct = "Vol+"

        rsi = calc_rsi_sma(wc, PINK_RSI_LEN)
        rv = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50
        rd = _pink_hidden_bullish_div(wc, rsi, PINK_RSI_DIV_LOOKBACK)
        ef = ema(wc, 13)
        tok = wc.iloc[-1] > ef.iloc[-1] if not pd.isna(ef.iloc[-1]) else False

        sig = None
        if fresh and (ct or rd) and tok and rv < 60: sig = "DIP"
        elif fresh and (ct or rd) and rv < 65: sig = "DIP_E"
        elif True: sig = "DIP_W"

        if sig and (best is None or ["DIP","DIP_E","DIP_W"].index(sig) < ["DIP","DIP_E","DIP_W"].index(best.get('pink_signal','DIP_W'))):
            tp = wh.iloc[-30:].max() if len(wh) >= 30 else wc.iloc[-1]*1.1
            stop = wl.iloc[-5:].min()
            dp = [f"{label}({ev:.2f})"]
            if rd: dp.append("RSI-Div")
            if ct: dp.append(ct)
            if tok: dp.append("TrendK")
            dp.append(f"RSI:{rv:.0f}"); dp.append(f"T:{int(touches)}x")
            tags = []
            if rv < 40: tags.append("RSI<40")
            if rd: tags.append("DIV")
            best = {'pink_signal': sig, 'pink_ma': label, 'pink_candle': ct,
                    'pink_rsi_div': rd, 'pink_touches': int(touches),
                    'pink_stop': float(stop), 'pink_tp': float(tp),
                    'pink_detail': " ".join(dp), 'pink_tags': tags}
    return best if best else result


# ══════════════════════════════════════════
# ANALYZE DIP
# ══════════════════════════════════════════

def analyze_dip(ticker, df, xu_df, usd_df=None, dbg=None):
    """Commodity DIP analizi — haftalık EMA dip + SUPPLY."""
    try:
        if dbg: dbg['total'] = dbg.get('total', 0) + 1
        n = len(df)
        c, h, l, v = df['Close'], df['High'], df['Low'], df['Volume']

        atr = calc_atr_sma(df, ATR_LEN)
        atr_val = atr.iloc[-1]
        if pd.isna(atr_val) or atr_val == 0: return None
        close_price = float(c.iloc[-1])

        avg_turnover = (c * v).iloc[-20:].mean()
        if pd.isna(avg_turnover) or avg_turnover < MIN_AVG_VOLUME_TL: return None
        rvol = float(v.iloc[-1] / v.iloc[-20:].mean()) if v.iloc[-20:].mean() > 0 else 0

        # Regime
        ema_f, ema_s = ema(c, EMA_FAST), ema(c, EMA_SLOW)
        adx_data = calc_adx_ema(df, ADX_LEN)
        adx_val = adx_data['adx'].iloc[-1] if not pd.isna(adx_data['adx'].iloc[-1]) else 15
        st_data = calc_supertrend(df, ST_LEN, ST_MULT)
        st_up = bool(st_data['trend'].iloc[-1] == 1) if not pd.isna(st_data['trend'].iloc[-1]) else False
        ema_up = float(ema_f.iloc[-1]) > float(ema_s.iloc[-1]) if not pd.isna(ema_f.iloc[-1]) else False

        if adx_val >= ADX_TREND and ema_up and st_up: regime = 3; rn = "FULL_TREND"
        elif adx_val >= ADX_TREND or (ema_up and st_up): regime = 2; rn = "TREND"
        elif adx_val >= ADX_CHOPPY: regime = 1; rn = "GRI_BOLGE"
        else: regime = 0; rn = "CHOPPY"

        # RS vs DBC benchmark
        rs_score = 0.0
        if xu_df is not None and len(xu_df) >= RS_LEN2:
            sr1 = (c.iloc[-1]/c.iloc[-RS_LEN1]-1)*100 if n > RS_LEN1 else 0
            sr2 = (c.iloc[-1]/c.iloc[-RS_LEN2]-1)*100 if n > RS_LEN2 else 0
            br1 = (xu_df['Close'].iloc[-1]/xu_df['Close'].iloc[-RS_LEN1]-1)*100
            br2 = (xu_df['Close'].iloc[-1]/xu_df['Close'].iloc[-RS_LEN2]-1)*100
            rs_score = round(((sr1-br1)+(sr2-br2))/2, 1)

        quality = min(sum([20 if ema_up else 0, 15 if st_up else 0, 15 if adx_val>=ADX_TREND else 0,
                           10 if rvol>=1.0 else 0, 10 if rvol>=1.5 else 0, 10 if rs_score>0 else 0]), 100)

        pink = calc_pink_v2(df)
        pink_signal = pink.get('pink_signal')
        supply = calc_supply_criteria(df)

        signal = None
        if pink_signal:
            signal = pink_signal
            wt = calc_wavetrend(df)
            wt_recent = bool(wt['wt_recent'].iloc[-1])
            e55c = (c > ema_s) & (c.shift(1) <= ema_s.shift(1))
            re55 = any(e55c.iloc[-i] for i in range(1, 11) if -i >= -n)
            if (wt_recent and re55 or wt_recent and regime >= 2) and pink_signal in ("DIP","DIP_E"):
                signal = "DIP+"
        elif supply['supply_score'] >= SUPPLY_MIN_SCORE:
            signal = "SUPPLY"
        else:
            return None

        if dbg: dbg['signal'] = dbg.get('signal', 0) + 1

        if signal == "SUPPLY":
            stop_price = l.iloc[-20:].min()
            tp_price = h.iloc[-50:].max() if n >= 50 else close_price * 1.15
            tp_src = "SWING"
        else:
            stop_price = pink['pink_stop'] if pink['pink_stop'] else close_price - atr_val * PINK_STOP_MULT
            tp_price = pink['pink_tp'] if pink['pink_tp'] else close_price + atr_val * DONUS_TP
            tp_src = "SWING"

        risk = close_price - stop_price
        rr = (tp_price - close_price) / risk if risk > 0 else 0
        overext = calc_overextended(df)

        detail = pink.get('pink_detail', '')
        if signal == "SUPPLY":
            detail = f"SUPPLY: {','.join(supply['supply_tags'])}"

        return {
            'ticker': ticker, 'signal': signal, 'regime': rn,
            'regime_score': regime, 'close': round(close_price, 2),
            'stop': round(stop_price, 2), 'tp': round(tp_price, 2),
            'tp_src': tp_src, 'rr': round(rr, 2), 'atr': round(atr_val, 2),
            'quality': int(quality), 'rs_score': round(rs_score, 1), 'rvol': round(rvol, 1),
            'pink_detail': detail, 'pink_tags': pink.get('pink_tags', []),
            'kc_score': supply['supply_score'], 'kc_tags': supply['supply_tags'],
            'turnover_m': round(avg_turnover / 1e6, 1),
            'overext_score': overext['overext_score'],
            'overext_tags': overext['overext_tags'],
            'overext_warning': overext['overext_warning'],
        }
    except Exception as e:
        if dbg: dbg['exception'] = dbg.get('exception', 0) + 1
        return None
