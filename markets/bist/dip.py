"""
BIST Screener — DIP (Pink V2) Analiz Modulu
Haftalik EMA dip tarama + RECOVER (eski KC) sinyali.
Yeni terminoloji:
  DIP+  = DIP + rejim overlap
  DIP   = Guvenli haftalik dip (safe)
  DIP_E = Erken dip
  DIP_W = Izle (teyit bekliyor)
  RECOVER = USD EMA + likidite + MACD + kurumsal kapanis
"""
import numpy as np
import pandas as pd
from markets.bist.config import MIN_AVG_VOLUME_TL, DONUS_TP, PINK_STOP_MULT, PINK_EMA89, PINK_EMA144, PINK_RSI_LEN, PINK_RSI_DIV_LOOKBACK, PINK_TOUCH_WINDOW, PINK_TOUCH_COUNT, RVOL_THRESH
from core.config import (
    EMA_FAST, EMA_SLOW, ST_LEN, ST_MULT,
    ADX_LEN, ADX_SLOPE_LEN, ADX_SLOPE_THRESH, ADX_TREND, ADX_CHOPPY,
    ATR_LEN, RS_LEN1, RS_LEN2,
)
from core.indicators import (
    ema, sma, true_range, calc_atr_sma, calc_adx_ema, calc_supertrend,
    calc_wavetrend, calc_rsi_sma, calc_macd, resample_weekly, to_usd,
    calc_overextended,
)


# ── PINK V2 CORE helpers ──

def _pink_hidden_bullish_div(price, rsi_s, lookback=10):
    if len(price) < lookback + 2 or len(rsi_s) < lookback + 2:
        return False
    p = price.iloc[-lookback:]
    r = rsi_s.iloc[-lookback:]
    p1, p2 = p.iloc[-1], p.min()
    r1 = r.iloc[p.values.argmin()]
    r2 = r.iloc[-1]
    return p1 > p2 and r2 < r1


def _pink_reversal_candle(wdf, idx=-1):
    o, h, l, c = wdf['Open'].iloc[idx], wdf['High'].iloc[idx], wdf['Low'].iloc[idx], wdf['Close'].iloc[idx]
    tr = h - l
    if tr == 0:
        return False, ""
    body = abs(c - o)
    bp = body / tr
    us = h - max(c, o)
    ls = min(c, o) - l
    mid = (h + l) / 2
    if ls >= body * 2 and us < body * 0.5 and min(c, o) > mid and bp < 0.35:
        return True, "CEKIC"
    if bp < 0.05 and us < tr * 0.05 and ls > tr * 0.7:
        return True, "YUSUFCUK"
    if bp < 0.08 and ls > body * 1.5 and us > body * 1.5:
        return True, "DOJI"
    if bp < 0.20 and ls > body * 1.5 and us < ls * 0.8 and ls > tr * 0.30:
        return True, "TOPAC"
    if idx != 0 and abs(idx) < len(wdf):
        po, pc = wdf['Open'].iloc[idx - 1], wdf['Close'].iloc[idx - 1]
        if pc < po and c > o and c > po and o < pc:
            return True, "ENGULF"
    return False, ""


def _pink_count_touches(wc, wl, ma_s, window=52, pct=2.0):
    touches = 0
    for i in range(-min(window, len(wc)), 0):
        ma_val = ma_s.iloc[i]
        if np.isnan(ma_val):
            continue
        low_dist = abs(wl.iloc[i] - ma_val) / ma_val * 100
        close_dist = abs(wc.iloc[i] - ma_val) / ma_val * 100
        if low_dist < pct or close_dist < pct:
            touches += 1
    return touches


def calc_pink_v2(df, dbg=None, rs_score=0.0, avg_turnover=0.0, xu_df=None, verbose=False):
    """Pink V2 Dip Modulu — yeni terminoloji: DIP/DIP_E/DIP_W."""
    result = {
        'pink_signal': None, 'pink_ma': None, 'pink_candle': None,
        'pink_rsi_div': False, 'pink_touches': 0,
        'pink_stop': None, 'pink_tp': None, 'pink_detail': None,
        'pink_tags': [],
    }
    wdf = resample_weekly(df)
    if len(wdf) < 30:
        return result

    wc, wl, wh, wv = wdf['Close'], wdf['Low'], wdf['High'], wdf['Volume']
    wo = wdf['Open']
    w_ema13 = ema(wc, 13)
    w_ema21 = ema(wc, 21)
    w_ema34 = ema(wc, 34)
    w_ema55 = ema(wc, 55)
    w_ema89 = ema(wc, PINK_EMA89)
    w_ema100 = ema(wc, 100)
    w_ema144 = ema(wc, PINK_EMA144)
    w_ema200 = ema(wc, 200)
    w_rsi = calc_rsi_sma(wc, PINK_RSI_LEN)
    w_atr = calc_atr_sma(wdf, ATR_LEN)

    last_close = float(wc.iloc[-1])
    last_low = float(wl.iloc[-1])
    last_high = float(wh.iloc[-1])
    last_ema21 = float(w_ema21.iloc[-1]) if not np.isnan(w_ema21.iloc[-1]) else None
    last_atr = float(w_atr.iloc[-1]) if not np.isnan(w_atr.iloc[-1]) else 0
    last_rsi = float(w_rsi.iloc[-1]) if not np.isnan(w_rsi.iloc[-1]) else 50

    if last_rsi > 65 or last_rsi < 25:
        return result
    if rs_score < -25:
        return result

    # Endeks filtresi
    idx_bullish = True
    if xu_df is not None and len(xu_df) > 150:
        xu_wdf = resample_weekly(xu_df)
        if len(xu_wdf) > 21:
            xu_w_sma21 = sma(xu_wdf['Close'], 21)
            idx_bullish = float(xu_wdf['Close'].iloc[-1]) > float(xu_w_sma21.iloc[-1])

    # Aylik trend
    monthly_bullish = True
    mdf = df.resample('ME').agg({'Open': 'first', 'High': 'max', 'Low': 'min',
                                  'Close': 'last', 'Volume': 'sum'}).dropna()
    if len(mdf) >= 12:
        m_sma10 = sma(mdf['Close'], 10)
        monthly_bullish = float(mdf['Close'].iloc[-1]) > float(m_sma10.iloc[-1])

    # Fibonacci
    fib_levels = {}
    sw_high_52 = float(wh.iloc[-52:].max()) if len(wh) >= 52 else float(wh.max())
    sw_low_52 = float(wl.iloc[-52:].min()) if len(wl) >= 52 else float(wl.min())
    fib_range = sw_high_52 - sw_low_52
    if fib_range > 0:
        fib_levels = {
            'fib382': sw_high_52 - fib_range * 0.382,
            'fib500': sw_high_52 - fib_range * 0.500,
            'fib618': sw_high_52 - fib_range * 0.618,
        }

    # Hacim
    w_vol_avg5 = sma(wv, 5)
    last_vol = float(wv.iloc[-1]) if not np.isnan(wv.iloc[-1]) else 0
    last_vol_avg = float(w_vol_avg5.iloc[-1]) if not np.isnan(w_vol_avg5.iloc[-1]) else 1
    vol_rising = last_vol > last_vol_avg * 1.0

    # Gunluk trend kirilimi
    daily_trend_break = False
    weekly_green = last_close > float(wo.iloc[-1])
    if len(df) > 20:
        dc = df['Close']
        dh = df['High']
        recent_high5 = dh.rolling(5).max().shift(1)
        if not np.isnan(recent_high5.iloc[-1]):
            broke_high = dc.iloc[-1] > recent_high5.iloc[-1]
            green_last3 = sum(1 for i in range(3) if dc.iloc[-1 - i] > df['Open'].iloc[-1 - i]) >= 2
            daily_trend_break = (broke_high and green_last3) or weekly_green
    else:
        daily_trend_break = weekly_green

    # FVG
    has_fvg = False
    if len(wdf) >= 15:
        for i in range(-5, -13, -1):
            if abs(i) + 2 > len(wdf):
                continue
            gap_top = float(wl.iloc[i])
            gap_bot = float(wh.iloc[i - 2])
            if gap_top <= gap_bot:
                continue
            post_gap_high = float(wh.iloc[i:].max())
            if post_gap_high < gap_top * 1.02:
                continue
            if last_low <= gap_top * 1.03 and last_close >= gap_bot * 0.98:
                has_fvg = True
                break

    # Konsolidasyon
    is_consolidating = False
    if len(wdf) >= 4 and last_atr > 0:
        recent_ranges = [(float(wh.iloc[-1 - i]) - float(wl.iloc[-1 - i])) for i in range(3)]
        avg_range = sum(recent_ranges) / 3
        is_consolidating = avg_range < last_atr * 0.70

    # Volatilite squeeze
    vol_squeeze = False
    if len(wc) >= 20:
        w_bb_basis = sma(wc, 20)
        w_bb_dev = wc.rolling(20).std() * 2
        w_kc_rng = sma(calc_atr_sma(wdf, 10), 20) * 1.5
        if not np.isnan(w_bb_dev.iloc[-1]) and not np.isnan(w_kc_rng.iloc[-1]):
            vol_squeeze = float(w_bb_dev.iloc[-1]) < float(w_kc_rng.iloc[-1])

    w_swing_high = float(wh.rolling(10).max().iloc[-1]) if len(wdf) > 10 else last_close * 1.08

    ma_list = [
        ("EMA200", w_ema200, 5.0), ("EMA144", w_ema144, 5.0),
        ("EMA100", w_ema100, 4.0), ("EMA89", w_ema89, 4.0),
        ("EMA55", w_ema55, 3.0), ("EMA34", w_ema34, 2.5),
        ("EMA21", w_ema21, 2.5), ("EMA13", w_ema13, 2.0),
    ]

    all_ma_vals = {}
    for name, ser, _ in ma_list:
        vv = ser.iloc[-1]
        if not np.isnan(vv):
            all_ma_vals[name] = float(vv)

    best_sig, best_det = None, {}

    for ma_name, ma_ser, prox in ma_list:
        mv = ma_ser.iloc[-1]
        if np.isnan(mv):
            continue
        mv = float(mv)

        touching = last_low <= mv * (1 + prox / 100) and last_close >= mv * (1 - prox / 100)
        if not touching:
            continue

        # Taze temas
        weeks_above = 0
        for wi in range(2, min(6, len(wc))):
            prev_close = float(wc.iloc[-1 - wi])
            prev_ma = float(ma_ser.iloc[-1 - wi]) if not np.isnan(ma_ser.iloc[-1 - wi]) else mv
            if prev_close > prev_ma * 1.02:
                weeks_above += 1
        classic_fresh = weeks_above >= 2
        last_above_ma = last_close > mv
        bottom_bounce = last_above_ma and vol_rising
        fresh_touch = classic_fresh or bottom_bounce
        if not fresh_touch:
            continue

        has_candle, candle_name = _pink_reversal_candle(wdf, -1)
        has_div = _pink_hidden_bullish_div(wc, w_rsi, PINK_RSI_DIV_LOOKBACK)
        touches = _pink_count_touches(wc, wl, ma_ser, PINK_TOUCH_WINDOW, prox)

        if ma_name in ("EMA13", "EMA21", "EMA34"):
            if avg_turnover < 20_000_000:
                continue
            min_touches = 3
        else:
            min_touches = PINK_TOUCH_COUNT

        safe = vol_rising and daily_trend_break
        early = touches >= min_touches
        if not safe and not early:
            continue

        # Donus teyidi
        confirmed_safe = False
        confirmed_early = False
        if len(wc) >= 3:
            w1_above = float(wc.iloc[-1]) > mv
            w2_above = float(wc.iloc[-2]) > mv if not np.isnan(wc.iloc[-2]) else False
            hold_above = w1_above and w2_above
            broke_prev_high = float(wc.iloc[-1]) > float(wh.iloc[-2])
            green1 = float(wc.iloc[-1]) > float(wo.iloc[-1])
            green2 = float(wc.iloc[-2]) > float(wo.iloc[-2]) if len(wo) >= 2 else False
            two_green = green1 and green2
            conf_count = sum([hold_above, broke_prev_high, two_green])
            confirmed_safe = conf_count >= 2
            confirmed_early = conf_count >= 1

        if safe and confirmed_safe:
            sig = "DIP"
        elif safe:
            sig = "DIP_W"
        elif early and confirmed_early:
            sig = "DIP_E"
        else:
            sig = "DIP_W"

        # Stop
        lower_vals = sorted([vv for n, vv in all_ma_vals.items()
                             if n != ma_name and vv < mv], reverse=True)
        if lower_vals:
            stop_p = lower_vals[0] - last_atr * 0.3
        else:
            stop_p = mv * 0.97
        max_stop = last_close * 0.85
        if stop_p < max_stop:
            stop_p = max_stop

        # TP
        if ma_name in ("EMA13", "EMA21", "EMA34"):
            tp_p = max(w_swing_high, last_close * 1.05)
        else:
            tp_p = last_ema21 if (last_ema21 and last_ema21 > last_close) else last_close * 1.10

        risk = last_close - stop_p
        reward = tp_p - last_close
        rr = reward / risk if risk > 0 else 0
        rr_min = 0.5 if (safe and confirmed_safe) else 1.0
        if rr < rr_min:
            continue

        # Fib match
        fib_match = None
        for fib_name, fib_val in fib_levels.items():
            if abs(mv - fib_val) / mv < 0.03:
                fib_match = fib_name
                break

        tags = []
        if fib_match:       tags.append(f"FIB{fib_match[-3:]}")
        if has_fvg:          tags.append("FVG")
        if is_consolidating: tags.append("KONS")
        if vol_squeeze:      tags.append("SQZW")
        if idx_bullish:      tags.append("IDX+")
        if monthly_bullish:  tags.append("MTF+")
        if has_div:          tags.append("DIV")

        det = {
            'signal': sig, 'ma': ma_name, 'candle': candle_name if has_candle else None,
            'rsi_div': has_div, 'touches': touches, 'vol_rising': vol_rising,
            'rsi_level': round(last_rsi, 0), 'daily_break': daily_trend_break,
            'stop': round(stop_p, 2), 'tp': round(tp_p, 2), 'ma_val': round(mv, 2),
            'fib_match': fib_match, 'idx_bullish': idx_bullish,
            'monthly_bullish': monthly_bullish, 'tags': tags,
        }

        if best_sig is None or \
           (sig == "DIP" and best_det.get('signal') == "DIP_E") or \
           (sig == "DIP" and best_det.get('signal') == "DIP_W") or \
           ma_name in ("EMA200", "EMA144"):
            best_sig, best_det = sig, det

    if best_sig:
        if not best_det.get('idx_bullish', True):
            if best_sig == "DIP":
                best_sig = "DIP_E"
            best_det['tags'].append("IDX⚠")
        if not best_det.get('monthly_bullish', True):
            best_det['tags'].append("MTF⚠")

        result['pink_signal'] = best_sig
        result['pink_ma'] = best_det['ma']
        result['pink_candle'] = best_det.get('candle')
        result['pink_rsi_div'] = best_det['rsi_div']
        result['pink_touches'] = best_det['touches']
        result['pink_stop'] = best_det['stop']
        result['pink_tp'] = best_det['tp']
        result['pink_tags'] = best_det.get('tags', [])

        parts = [f"{best_det['ma']}({best_det['ma_val']})"]
        if best_det['rsi_div']:     parts.append("RSI-Div")
        if best_det.get('candle'):  parts.append(best_det['candle'])
        if best_det['vol_rising']:  parts.append("Vol+")
        if best_det['daily_break']: parts.append("TrendK")
        parts.append(f"RSI:{int(best_det['rsi_level'])}")
        parts.append(f"T:{best_det['touches']}x")
        if best_det.get('tags'):
            parts.append("|" + ",".join(best_det['tags']))
        result['pink_detail'] = " ".join(parts)

    return result


# ═══════════════════════════════════════════
# KC → RECOVER sinyali
# ═══════════════════════════════════════════

def calc_recover_criteria(df, usd_df_raw=None, verbose=False):
    """RECOVER (eski KC) kriterleri: USD EMA + likidite + MACD + kurumsal kapanis."""
    result = {
        'usd_ema_above': [], 'usd_ema_cross': False, 'usd_close': None,
        'liquidity_grab': False, 'liq_level': None,
        'macd_bull_cross': False, 'macd_near_zero': False,
        'institutional_close': False, 'inst_level': None,
        'vol_candle_divergence': False, 'double_bottom': False,
        'higher_low': False, 'kc_score': 0, 'kc_tags': [],
    }

    c = df['Close']
    wdf = resample_weekly(df)
    if len(wdf) < 30:
        return result

    wc = wdf['Close']
    wh = wdf['High']
    wl = wdf['Low']
    wo = wdf['Open']
    wv = wdf['Volume']

    # 1. USD EMA
    if usd_df_raw is not None:
        usd_price = to_usd(df, usd_df_raw)
        if usd_price is not None and len(usd_price) >= 100:
            usd_c = usd_price['Close']
            result['usd_close'] = round(float(usd_c.iloc[-1]), 4)
            usd_w = resample_weekly(usd_price)
            if len(usd_w) >= 20:
                usd_wc = usd_w['Close']
                usd_emas = {}
                for p in [14, 21, 55, 144, 233]:
                    if len(usd_wc) >= p + 5:
                        usd_emas[p] = ema(usd_wc, p)
                for p, e in usd_emas.items():
                    if float(usd_wc.iloc[-1]) > float(e.iloc[-1]):
                        result['usd_ema_above'].append(p)
                if 55 in usd_emas and 144 in usd_emas:
                    e55 = float(usd_emas[55].iloc[-1])
                    e144 = float(usd_emas[144].iloc[-1])
                    e55_prev = float(usd_emas[55].iloc[-2])
                    e144_prev = float(usd_emas[144].iloc[-2])
                    if e55 > e144 and e55_prev <= e144_prev:
                        result['usd_ema_cross'] = True
                    elif abs(e55 - e144) / e144 < 0.02 and e55 > e55_prev:
                        result['usd_ema_cross'] = True

    # 2. Likidite alimi
    if len(wdf) >= 20:
        lookback = min(52, len(wh) - 4)
        if lookback > 5:
            swing_highs = []
            for i in range(2, lookback):
                idx = -(i + 1)
                if abs(idx) >= len(wh):
                    continue
                h_val = float(wh.iloc[idx])
                is_swing = True
                for j in [1, 2]:
                    if abs(idx - j) < len(wh) and float(wh.iloc[idx - j]) > h_val:
                        is_swing = False
                    if abs(idx + j) <= len(wh) and idx + j < 0 and float(wh.iloc[idx + j]) > h_val:
                        is_swing = False
                if is_swing:
                    swing_highs.append(h_val)

            if swing_highs:
                lc = float(wc.iloc[-1])
                for sh in sorted(swing_highs):
                    if lc > sh and float(wc.iloc[-2]) <= sh:
                        result['liquidity_grab'] = True
                        result['liq_level'] = round(sh, 2)
                        break
                    elif lc > sh and float(wh.iloc[-1]) > sh:
                        result['liquidity_grab'] = True
                        result['liq_level'] = round(sh, 2)
                        break

    # 3. MACD ala gecis
    if len(wc) >= 35:
        macd_line, signal_line, histogram = calc_macd(wc)
        for i in range(4):
            idx = -(i + 1)
            if abs(idx) >= len(macd_line) or abs(idx + 1) >= len(macd_line):
                continue
            curr_hist = float(histogram.iloc[idx])
            prev_hist = float(histogram.iloc[idx - 1])
            if curr_hist > 0 and prev_hist <= 0:
                result['macd_bull_cross'] = True
                curr_macd = float(macd_line.iloc[idx])
                prev_macd = float(macd_line.iloc[idx - 1])
                macd_range = float(macd_line.max() - macd_line.min())
                if macd_range > 0:
                    zero_ratio = abs(curr_macd) / macd_range
                    if zero_ratio < 0.15 or (prev_macd <= 0 and curr_macd >= 0):
                        result['macd_near_zero'] = True
                elif prev_macd <= 0 and curr_macd >= 0:
                    result['macd_near_zero'] = True
                break

    # 4. Kurumsal kapanis
    if len(wdf) >= 20:
        lc = float(wc.iloc[-1])
        lookback = min(52, len(wh) - 2)
        resistance_zones = []
        for i in range(2, lookback):
            idx = -(i + 1)
            if abs(idx) >= len(wh):
                continue
            w_high = float(wh.iloc[idx])
            w_close = float(wc.iloc[idx])
            w_open = float(wo.iloc[idx])
            body_top = max(w_close, w_open)
            upper_wick = w_high - body_top
            body = abs(w_close - w_open)
            if body > 0 and upper_wick > body * 0.3:
                resistance_zones.append(w_high)

        if resistance_zones:
            for rz in sorted(resistance_zones):
                tolerance = rz * 0.01
                if lc >= rz - tolerance:
                    prev_below = all(float(wc.iloc[-(j + 1)]) < rz - tolerance
                                     for j in range(1, min(5, len(wc))))
                    if prev_below:
                        result['institutional_close'] = True
                        result['inst_level'] = round(rz, 2)
                        break

    # 5. Hacim-mum uyumsuzlugu
    if len(wdf) >= 10:
        lb = min(20, len(wdf))
        avg_body = np.mean([abs(float(wc.iloc[-i]) - float(wo.iloc[-i])) for i in range(4, lb)])
        avg_vol = np.mean([float(wv.iloc[-i]) for i in range(4, lb)])
        if avg_body > 0 and avg_vol > 0:
            for week_i in range(1, 4):
                if week_i >= len(wdf):
                    break
                w_body = abs(float(wc.iloc[-week_i]) - float(wo.iloc[-week_i]))
                w_vol = float(wv.iloc[-week_i])
                body_ratio = w_body / avg_body
                vol_ratio = w_vol / avg_vol
                if body_ratio < 0.5 and vol_ratio > 1.5:
                    result['vol_candle_divergence'] = True
                    break

    # 6. Ikili dip + yukselen dip
    if len(wdf) >= 20:
        lookback = min(30, len(wl) - 2)
        swing_lows = []
        for i in range(2, lookback):
            idx = -(i + 1)
            if abs(idx) >= len(wl) or abs(idx - 1) >= len(wl) or abs(idx - 2) >= len(wl):
                continue
            l_val = float(wl.iloc[idx])
            is_swing = True
            for j in [1, 2]:
                if abs(idx - j) < len(wl) and float(wl.iloc[idx - j]) < l_val:
                    is_swing = False
                if idx + j < 0 and abs(idx + j) < len(wl) and float(wl.iloc[idx + j]) < l_val:
                    is_swing = False
            if is_swing:
                swing_lows.append((i, l_val))

        if len(swing_lows) >= 2:
            for a in range(len(swing_lows)):
                for b in range(a + 1, len(swing_lows)):
                    weeks_a, low_a = swing_lows[a]
                    weeks_b, low_b = swing_lows[b]
                    if abs(weeks_a - weeks_b) >= 4:
                        avg_low = (low_a + low_b) / 2
                        diff_pct = abs(low_a - low_b) / avg_low
                        if diff_pct < 0.05:
                            result['double_bottom'] = True
                            break
                if result['double_bottom']:
                    break
            if len(swing_lows) >= 2:
                recent_low = swing_lows[0][1]
                prev_low = swing_lows[1][1]
                if recent_low > prev_low * 1.01:
                    result['higher_low'] = True

    # Skor
    score = 0
    tags = []
    usd_above = result['usd_ema_above']
    usd_levels = sorted([x for x in usd_above if x in [233, 144, 55, 21, 14]], reverse=True)
    if usd_levels:
        tags.append(f"USD{usd_levels[0]}+")
    if 233 in usd_above:     score += 15
    elif 144 in usd_above:   score += 12
    elif 55 in usd_above:    score += 8
    elif 21 in usd_above:    score += 4
    elif 14 in usd_above:    score += 2
    if result['usd_ema_cross']:         score += 10; tags.append("GOLDX")
    if result['liquidity_grab']:        score += 15; tags.append("LIQ✅")
    if result['macd_bull_cross']:
        if result['macd_near_zero']:    score += 15; tags.append("MACD0+")
        else:                           score += 8; tags.append("MACD+")
    if result['institutional_close']:   score += 12; tags.append("INST✅")
    if result['vol_candle_divergence']: score += 8; tags.append("BIRIKM")
    if result['double_bottom']:         score += 4; tags.append("2DIP")
    if result['higher_low']:            score += 3; tags.append("HL")

    result['kc_score'] = score
    result['kc_tags'] = tags
    return result


# ═══════════════════════════════════════════
# ANA DIP ANALİZ
# ═══════════════════════════════════════════

def analyze_dip(ticker, df, xu_df, usd_df=None, dbg=None):
    """DIP (Pink V2) + RECOVER analizi — tek hisse."""
    try:
        if dbg:
            dbg['total'] += 1
        n = len(df)
        c, h, l, o, v = df['Close'], df['High'], df['Low'], df['Open'], df['Volume']

        atr = calc_atr_sma(df, ATR_LEN)
        atr_val = atr.iloc[-1]
        if np.isnan(atr_val) or atr_val == 0:
            if dbg:
                dbg['no_atr'] += 1
            return None

        vol_sma20 = sma(v, 20)
        avg_turnover = vol_sma20.iloc[-1] * c.iloc[-1]
        if avg_turnover < MIN_AVG_VOLUME_TL:
            if dbg:
                dbg['low_vol'] += 1
            return None

        # Trend (rejim hesabi icin)
        ema_f = ema(c, EMA_FAST)
        ema_s = ema(c, EMA_SLOW)
        ema_trend_up = ema_f.iloc[-1] > ema_s.iloc[-1]
        st_dir = calc_supertrend(df, ST_LEN, ST_MULT)
        super_trend_up = st_dir.iloc[-1] == 1

        # HTF
        wdf = resample_weekly(df)
        if len(wdf) < 20:
            htf_adx, htf_slope, htf_rising, htf_trend_up = 0, 0, False, False
        else:
            htf_adx_s = calc_adx_ema(wdf, ADX_LEN)
            htf_adx = htf_adx_s.iloc[-1] if not np.isnan(htf_adx_s.iloc[-1]) else 0
            htf_slope = (htf_adx - htf_adx_s.iloc[-1 - ADX_SLOPE_LEN]) / ADX_SLOPE_LEN if len(htf_adx_s) > ADX_SLOPE_LEN else 0
            htf_rising = htf_slope > ADX_SLOPE_THRESH
            htf_ema_f = ema(wdf['Close'], EMA_FAST)
            htf_ema_s = ema(wdf['Close'], EMA_SLOW)
            htf_trend_up = htf_ema_f.iloc[-1] > htf_ema_s.iloc[-1]

        adx_s = calc_adx_ema(df, ADX_LEN)
        adx_val = adx_s.iloc[-1] if not np.isnan(adx_s.iloc[-1]) else 0
        adx_slope = (adx_val - adx_s.iloc[-1 - ADX_SLOPE_LEN]) / ADX_SLOPE_LEN if n > ADX_SLOPE_LEN else 0
        adx_rising = adx_slope > ADX_SLOPE_THRESH

        # Rejim
        trend_up_count = int(ema_trend_up) + int(super_trend_up) + int(htf_trend_up)
        confirmed_trend_up = trend_up_count >= 2 and c.iloc[-1] > ema_s.iloc[-1]
        if htf_adx > ADX_TREND and htf_rising:     htf_r = 2
        elif htf_adx > ADX_TREND:                   htf_r = 1
        elif htf_adx > ADX_CHOPPY:                   htf_r = 0
        else:                                        htf_r = -1
        daily_confirm = adx_val > ADX_CHOPPY and adx_rising
        if not confirmed_trend_up:                   regime = 0
        elif htf_r == 2 and daily_confirm:           regime = 3
        elif htf_r >= 1:                             regime = 2
        elif htf_r == 0:                             regime = 1
        else:                                        regime = 0
        regime_name = {3: "FULL_TREND", 2: "TREND", 1: "GRI_BOLGE", 0: "CHOPPY"}[regime]

        # Quality
        close_price = c.iloc[-1]
        rvol = v.iloc[-1] / vol_sma20.iloc[-1] if vol_sma20.iloc[-1] > 0 else 0
        candle_range = h.iloc[-1] - l.iloc[-1]
        clv = (c.iloc[-1] - l.iloc[-1]) / candle_range if candle_range > 0 else 0.5
        upper_wick = h.iloc[-1] - max(c.iloc[-1], o.iloc[-1])
        wick_ratio = upper_wick / candle_range if candle_range > 0 else 0
        range_atr = candle_range / atr_val if atr_val > 0 else 0
        rvol_s = 25 if rvol >= 2 else 20 if rvol >= 1.5 else 10 if rvol >= 1 else 0
        clv_s = 25 if clv >= 0.75 else 15 if clv >= 0.5 else 5 if clv >= 0.25 else 0
        wick_s = 25 if wick_ratio <= 0.15 else 15 if wick_ratio <= 0.3 else 5 if wick_ratio <= 0.5 else 0
        range_s = 25 if range_atr >= 1.2 else 15 if range_atr >= 0.8 else 5 if range_atr >= 0.5 else 0
        quality = rvol_s + clv_s + wick_s + range_s

        # RS
        rs_score = 0.0
        if xu_df is not None and len(xu_df) >= RS_LEN2 + 5:
            aligned = pd.DataFrame({'stock': c, 'bench': xu_df['Close']}).dropna()
            if len(aligned) >= RS_LEN2 + 5:
                sc, bc = aligned['stock'], aligned['bench']
                sp1 = (sc.iloc[-1] - sc.iloc[-1 - RS_LEN1]) / sc.iloc[-1 - RS_LEN1] * 100
                sp2 = (sc.iloc[-1] - sc.iloc[-1 - RS_LEN2]) / sc.iloc[-1 - RS_LEN2] * 100
                bp1 = (bc.iloc[-1] - bc.iloc[-1 - RS_LEN1]) / bc.iloc[-1 - RS_LEN1] * 100
                bp2 = (bc.iloc[-1] - bc.iloc[-1 - RS_LEN2]) / bc.iloc[-1 - RS_LEN2] * 100
                rs_score = (sp1 - bp1) * 0.6 + (sp2 - bp2) * 0.4

        # Pink V2
        _verbose = dbg.get('_verbose', False) if dbg else False
        pink = calc_pink_v2(df, dbg=dbg, rs_score=rs_score, avg_turnover=avg_turnover,
                            xu_df=xu_df, verbose=_verbose)
        pink_signal = pink['pink_signal']

        # RECOVER (eski KC)
        kc = calc_recover_criteria(df, usd_df_raw=usd_df, verbose=_verbose)

        # Pink sinyali yoksa ama RECOVER skoru yuksekse
        if not pink_signal:
            if kc['kc_score'] >= 45:
                # Dip filtresi
                wdf_dip = resample_weekly(df)
                dip_ok = False
                if len(wdf_dip) >= 4:
                    low_4w = float(wdf_dip['Low'].iloc[-4:].min())
                    dip_pct = (close_price - low_4w) / low_4w if low_4w > 0 else 999
                    if dip_pct <= 0.10:
                        dip_ok = True
                if not dip_ok and len(wdf_dip) >= 14:
                    w_close = wdf_dip['Close']
                    w_rsi = calc_rsi_sma(w_close, 14)
                    if len(w_rsi.dropna()) > 0 and float(w_rsi.iloc[-1]) < 60:
                        dip_ok = True

                if dip_ok:
                    signal = "RECOVER"
                    wdf_kc = resample_weekly(df)
                    kc_lookback = min(10, len(wdf_kc))
                    stop_price = float(wdf_kc['Low'].iloc[-kc_lookback:].min())
                    max_stop_dist = close_price * 0.15
                    if close_price - stop_price > max_stop_dist:
                        stop_price = close_price - max_stop_dist
                    tp_lookback = min(52, len(wdf_kc))
                    swing_high = float(wdf_kc['High'].iloc[-tp_lookback:].max())
                    risk_kc = close_price - stop_price
                    tp_price = max(swing_high, close_price + risk_kc * 2.0) if risk_kc > 0 else close_price + atr_val * 3.0
                    risk = close_price - stop_price
                    reward = tp_price - close_price
                    rr = reward / risk if risk > 0 else 0

                    overext = calc_overextended(df)

                    return {
                        'ticker': ticker, 'signal': signal, 'regime': regime_name,
                        'regime_score': regime, 'close': round(close_price, 2),
                        'stop': round(stop_price, 2), 'tp': round(tp_price, 2),
                        'tp_src': "SWING", 'rr': round(rr, 2), 'atr': round(atr_val, 2),
                        'quality': int(quality), 'rs_score': round(rs_score, 1),
                        'rvol': round(rvol, 1),
                        'pink_ma': None, 'pink_candle': None,
                        'pink_rsi_div': False, 'pink_touches': 0,
                        'pink_detail': f"RECOVER: {','.join(kc['kc_tags'])}",
                        'pink_tags': kc['kc_tags'],
                        'kc_score': kc['kc_score'], 'kc_tags': kc['kc_tags'],
                        'usd_close': kc.get('usd_close'),
                        'usd_ema_above': kc.get('usd_ema_above', []),
                        'turnover_m': round(avg_turnover / 1e6, 1),
                        'overext_score': overext['overext_score'],
                        'overext_tags': overext['overext_tags'],
                        'overext_warning': overext['overext_warning'],
                    }

            if dbg:
                dbg['no_signal'] += 1
            return None

        if dbg:
            dbg['signal'] += 1

        # Stop / TP
        stop_price = pink['pink_stop'] if pink['pink_stop'] else close_price - atr_val * PINK_STOP_MULT
        tp_price = pink['pink_tp'] if pink['pink_tp'] else close_price + atr_val * DONUS_TP
        tp_src = "SWING" if pink.get('pink_ma') in ("EMA13", "EMA21", "EMA34") else "EMA21"
        risk = close_price - stop_price
        reward = tp_price - close_price
        rr = reward / risk if risk > 0 else 0

        # Overlap check → DIP+
        wt = calc_wavetrend(df)
        wt_recent = bool(wt['wt_recent'].iloc[-1])
        ema55_cross = (c > ema_s) & (c.shift(1) <= ema_s.shift(1))
        recent_e55 = any(ema55_cross.iloc[-i] for i in range(1, 11) if -i >= -n)
        has_overlap = (wt_recent and recent_e55) or (wt_recent and regime >= 2)

        signal = pink_signal
        if has_overlap and pink_signal in ("DIP", "DIP_E"):
            signal = "DIP+"

        overext = calc_overextended(df, wt_data=wt)

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
            'pink_detail': pink.get('pink_detail'),
            'pink_tags': pink.get('pink_tags', []),
            'kc_score': kc['kc_score'], 'kc_tags': kc['kc_tags'],
            'usd_close': kc.get('usd_close'),
            'usd_ema_above': kc.get('usd_ema_above', []),
            'turnover_m': round(avg_turnover / 1e6, 1),
            'overext_score': overext['overext_score'],
            'overext_tags': overext['overext_tags'],
            'overext_warning': overext['overext_warning'],
        }
    except Exception as e:
        if dbg:
            dbg['exception'] += 1
            if dbg['exception'] <= 3:
                print(f"  [HATA] {ticker}: {type(e).__name__}: {e}")
        return None
