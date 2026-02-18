"""
BIST Screener — Rejim V3 Analiz Modülü
Günlük sinyaller: COMBO+, COMBO, STRONG, WEAK, REVERSAL, EARLY,
                  PULLBACK, SQUEEZE, MEANREV, PARTIAL
Orijinal mantık korunmuştur. Değişiklikler:
  - Terminoloji güncellemesi (CMB+→COMBO+, GUCLU→STRONG, vb.)
  - BOS_TIGHT=3, CHOCH_TIGHT=2 (config'den)
  - OVEREXTENDED metadata eklendi
"""
import numpy as np
import pandas as pd
from markets.bist.config import (
    RVOL_THRESH, TREND_STOP, GRI_STOP, MR_STOP, DONUS_STOP, COMBO_STOP,
    TREND_TP, GRI_TP, DONUS_TP, COMBO_TP, QUAL_MIN_GRI, QUAL_MIN_TREND,
    RS_THRESHOLD, MIN_AVG_VOLUME_TL,
    ATR_PANIC_PCTILE, ATR_PCTILE_WINDOW,
    CORE_Q_MIN_PARTIAL_STRONG, CORE_Q_MIN_EARLY, CORE_Q_MIN_COMBO,
    COMBO_RS_MAX, COMBO_OE_MAX, COMBO_DIST_EMA_MAX,
    POS_SIZE_CORE, POS_SIZE_MOMENTUM,
    MOMENTUM_RS_THRESH,
    CORE_SIGNALS, MOMENTUM_SIGNALS,
)
from core.config import (
    ADX_LEN, ADX_TREND, ADX_CHOPPY, ADX_SLOPE_LEN, ADX_SLOPE_THRESH,
    EMA_FAST, EMA_SLOW, ST_LEN, ST_MULT,
    BOS_LOOKBACK, BOS_TIGHT, CHOCH_TIGHT,
    SQ_LEN, SQ_MULT_BB, SQ_MULT_KC,
    BB_LEN, BB_MULT, DONCH_LEN, MR_RSI_LEN, MR_RSI_THRESH,
    RS_LEN1, RS_LEN2, ATR_LEN,
)
from core.indicators import (
    ema, sma, rma, true_range, calc_atr, calc_adx, calc_supertrend,
    calc_wavetrend, calc_pmax, calc_smc, calc_order_blocks,
    calc_rsi, resample_weekly, calc_overextended, calc_atr_percentile,
)


def analyze_regime(ticker, df, xu_df, dbg=None, usd_df=None, market_state=None):
    """Rejim V3 analizi — tek hisse. Production Ruleset v1."""
    try:
        if dbg:
            dbg['total'] += 1
        n = len(df)
        c = df['Close']
        h = df['High']
        l = df['Low']
        o = df['Open']
        v = df['Volume']

        # ATR
        atr = calc_atr(df, ATR_LEN)
        atr_val = atr.iloc[-1]
        if np.isnan(atr_val) or atr_val == 0:
            if dbg:
                dbg['no_atr'] += 1
            return None

        # Hacim filtresi
        vol_sma20 = sma(v, 20)
        avg_turnover = vol_sma20.iloc[-1] * c.iloc[-1]
        if avg_turnover < MIN_AVG_VOLUME_TL:
            if dbg:
                dbg['low_vol'] += 1
            return None

        # === PANIC BLOCK: ATR percentile ===
        atr_pctile_s = calc_atr_percentile(df, ATR_LEN, ATR_PCTILE_WINDOW)
        atr_pctile = float(atr_pctile_s.iloc[-1])
        if atr_pctile >= ATR_PANIC_PCTILE:
            if dbg:
                dbg['panic_block'] = dbg.get('panic_block', 0) + 1
            return None

        # === SIDEWAYS BLOCK ===
        if market_state and market_state.get('sideways'):
            if dbg:
                dbg['sideways_block'] = dbg.get('sideways_block', 0) + 1
            return None

        # === RISK-OFF BLOCK ===
        if market_state and market_state.get('risk_off'):
            if dbg:
                dbg['riskoff_block'] = dbg.get('riskoff_block', 0) + 1
            return None

        # === dist_ema_atr (fiyat/EMA20 mesafesi ATR cinsinden) ===
        ema20 = ema(c, 20)
        dist_ema_atr = float((c.iloc[-1] - ema20.iloc[-1]) / atr_val) if atr_val > 0 else 0.0

        # === TREND ===
        ema_f = ema(c, EMA_FAST)
        ema_s = ema(c, EMA_SLOW)
        ema_trend_up = ema_f.iloc[-1] > ema_s.iloc[-1]

        st_dir = calc_supertrend(df, ST_LEN, ST_MULT)
        super_trend_up = st_dir.iloc[-1] == 1

        # === HTF (haftalık) ===
        wdf = resample_weekly(df)
        if len(wdf) < 20:
            htf_adx, htf_slope, htf_rising, htf_trend_up = 0, 0, False, False
        else:
            htf_adx_s = calc_adx(wdf, ADX_LEN)
            htf_adx = htf_adx_s.iloc[-1] if not np.isnan(htf_adx_s.iloc[-1]) else 0
            htf_slope = (htf_adx - htf_adx_s.iloc[-1 - ADX_SLOPE_LEN]) / ADX_SLOPE_LEN if len(htf_adx_s) > ADX_SLOPE_LEN else 0
            htf_rising = htf_slope > ADX_SLOPE_THRESH
            htf_ema_f = ema(wdf['Close'], EMA_FAST)
            htf_ema_s = ema(wdf['Close'], EMA_SLOW)
            htf_trend_up = htf_ema_f.iloc[-1] > htf_ema_s.iloc[-1]

        # === GÜNLÜK ADX ===
        adx_s = calc_adx(df, ADX_LEN)
        adx_val = adx_s.iloc[-1] if not np.isnan(adx_s.iloc[-1]) else 0
        adx_slope = (adx_val - adx_s.iloc[-1 - ADX_SLOPE_LEN]) / ADX_SLOPE_LEN if n > ADX_SLOPE_LEN else 0
        adx_rising = adx_slope > ADX_SLOPE_THRESH

        # === REJİM ===
        trend_up_count = int(ema_trend_up) + int(super_trend_up) + int(htf_trend_up)
        confirmed_trend_up = trend_up_count >= 2 and c.iloc[-1] > ema_s.iloc[-1]

        if htf_adx > ADX_TREND and htf_rising:
            htf_r = 2
        elif htf_adx > ADX_TREND:
            htf_r = 1
        elif htf_adx > ADX_CHOPPY:
            htf_r = 0
        else:
            htf_r = -1

        daily_confirm = adx_val > ADX_CHOPPY and adx_rising

        if not confirmed_trend_up:
            regime = 0
        elif htf_r == 2 and daily_confirm:
            regime = 3
        elif htf_r >= 1:
            regime = 2
        elif htf_r == 0:
            regime = 1
        else:
            regime = 0

        regime_name = {3: "FULL_TREND", 2: "TREND", 1: "GRI_BOLGE", 0: "CHOPPY"}[regime]
        if dbg:
            dbg['regime'][regime] = dbg['regime'].get(regime, 0) + 1

        # === WAVETREND ===
        wt = calc_wavetrend(df)
        wt_cross_up = bool(wt['cross_up'].iloc[-1])
        wt_recent = bool(wt['wt_recent'].iloc[-1])
        wt_bullish = bool(wt['wt_bullish'].iloc[-1])
        wt_killed = bool(wt['wt_killed'].iloc[-1])

        # === PMAX ===
        pm = calc_pmax(df)
        pmax_long = bool(pm['pmax_long'][-1])

        # === SMC ===
        smc = calc_smc(df)
        last_idx = n - 1
        bos_bar = smc['bos_bar'][last_idx]
        choch_bar = smc['choch_bar'][last_idx]
        swing_bias = smc['swing_bias']

        recent_bos = (last_idx - bos_bar) <= BOS_LOOKBACK and smc['bull_bos'].any() and swing_bias == 1
        recent_choch = (last_idx - choch_bar) <= BOS_LOOKBACK and smc['bull_choch'].any() and swing_bias == 1
        # Tight pencereler (COMBO için) — config'den BOS_TIGHT=3, CHOCH_TIGHT=2
        bos_tight = (last_idx - bos_bar) <= BOS_TIGHT and swing_bias == 1
        choch_tight = (last_idx - choch_bar) <= CHOCH_TIGHT and swing_bias == 1

        bos_age = last_idx - bos_bar if bos_bar >= 0 and (last_idx - bos_bar) <= BOS_LOOKBACK else None
        choch_age = last_idx - choch_bar if choch_bar >= 0 and (last_idx - choch_bar) <= BOS_LOOKBACK else None

        # === OB ===
        ob = calc_order_blocks(df)
        close_price = c.iloc[-1]
        nearest_ob = np.nan
        if not np.isnan(ob['ob_resist_bot']) and ob['ob_resist_bot'] > close_price:
            nearest_ob = (ob['ob_resist_bot'] + ob['ob_resist_top']) / 2

        # === SQUEEZE ===
        sq_basis = sma(c, SQ_LEN)
        sq_dev = c.rolling(SQ_LEN).std() * SQ_MULT_BB
        sq_rng = sma(true_range(df), SQ_LEN)
        sqz_on = ((sq_basis - sq_dev) > (sq_basis - SQ_MULT_KC * sq_rng)) & \
                 ((sq_basis + sq_dev) < (sq_basis + SQ_MULT_KC * sq_rng))
        hh = h.rolling(SQ_LEN).max()
        ll = l.rolling(SQ_LEN).min()
        sq_mid = (hh + ll) / 2
        sq_mom_src = c - (sq_mid + sq_basis) / 2
        sq_mom = sq_mom_src.rolling(SQ_LEN).mean()
        sq_release = (~sqz_on) & sqz_on.shift(1) & (sq_mom > 0) & (sq_mom > sq_mom.shift(1))
        sq_release_last = any(bool(sq_release.iloc[-1 - i]) for i in range(3) if len(sq_release) > i)
        atr_ma = sma(atr, 20)
        atr_expanding = any(atr.iloc[-1 - i] > atr_ma.iloc[-1 - i] * 1.05 for i in range(3) if len(atr) > i)
        trend_sq = sq_release_last or (sq_mom.iloc[-1] > 0 and sq_mom.iloc[-1] > sq_mom.iloc[-2] and not sqz_on.iloc[-1])

        # === BB / MR ===
        bb_basis = sma(c, BB_LEN)
        bb_dev_val = c.rolling(BB_LEN).std() * BB_MULT
        bb_upper = bb_basis + bb_dev_val
        bb_lower = bb_basis - bb_dev_val
        bb_pctb = (c - bb_lower) / (bb_upper - bb_lower)
        donch_lower = l.rolling(DONCH_LEN).min()
        rsi_short = calc_rsi(c, MR_RSI_LEN)

        # === QUALITY ===
        rvol = v.iloc[-1] / vol_sma20.iloc[-1] if vol_sma20.iloc[-1] > 0 else 0
        candle_range = h.iloc[-1] - l.iloc[-1]
        clv = (c.iloc[-1] - l.iloc[-1]) / candle_range if candle_range > 0 else 0.5
        upper_wick = h.iloc[-1] - max(c.iloc[-1], o.iloc[-1])
        wick_ratio = upper_wick / candle_range if candle_range > 0 else 0
        range_atr = candle_range / atr_val if atr_val > 0 else 0

        rvol_s = 25 if rvol >= 2 else 20 if rvol >= RVOL_THRESH else 10 if rvol >= 1 else 0
        clv_s = 25 if clv >= 0.75 else 15 if clv >= 0.5 else 5 if clv >= 0.25 else 0
        wick_s = 25 if wick_ratio <= 0.15 else 15 if wick_ratio <= 0.3 else 5 if wick_ratio <= 0.5 else 0
        range_s = 25 if range_atr >= 1.2 else 15 if range_atr >= 0.8 else 5 if range_atr >= 0.5 else 0
        quality = rvol_s + clv_s + wick_s + range_s
        q_pass_gri = quality >= QUAL_MIN_GRI
        q_pass_trend = quality >= QUAL_MIN_TREND

        # === RS ===
        rs_pass = True
        rs_score = 0.0
        if xu_df is not None and len(xu_df) >= RS_LEN2 + 5:
            aligned = pd.DataFrame({'stock': c, 'bench': xu_df['Close']}).dropna()
            if len(aligned) >= RS_LEN2 + 5:
                sc = aligned['stock']
                bc = aligned['bench']
                sp1 = (sc.iloc[-1] - sc.iloc[-1 - RS_LEN1]) / sc.iloc[-1 - RS_LEN1] * 100
                sp2 = (sc.iloc[-1] - sc.iloc[-1 - RS_LEN2]) / sc.iloc[-1 - RS_LEN2] * 100
                bp1 = (bc.iloc[-1] - bc.iloc[-1 - RS_LEN1]) / bc.iloc[-1 - RS_LEN1] * 100
                bp2 = (bc.iloc[-1] - bc.iloc[-1 - RS_LEN2]) / bc.iloc[-1 - RS_LEN2] * 100
                rs_score = (sp1 - bp1) * 0.6 + (sp2 - bp2) * 0.4
                rs_pass = rs_score > RS_THRESHOLD

        # ============================================================
        # SİNYAL TESPİTİ (v3 mantığı birebir — yeni terminoloji)
        # ============================================================
        vol_high = v.iloc[-1] > vol_sma20.iloc[-1] * RVOL_THRESH

        # A: TREND (q_pass_trend kaldırıldı — mod filtresi quality gate uygulayacak)
        strong = regime >= 2 and ema_trend_up and super_trend_up and trend_sq and vol_high and rs_pass
        weak = regime >= 2 and ema_trend_up and super_trend_up and trend_sq and (not vol_high) and rs_pass

        # B: PULLBACK
        pb_rsi = calc_rsi(c, 5)
        pb_dipped = any(pb_rsi.iloc[-i] < 40 and pb_rsi.iloc[-i] > 20 for i in range(1, 6) if -i >= -n)
        pb_vol_dry = any(v.iloc[-i] < vol_sma20.iloc[-i] * 0.8 for i in range(1, 6) if -i >= -n)
        pb_reclaim = any(c.iloc[-1 - i] > ema_f.iloc[-1 - i] and c.iloc[-2 - i] <= ema_f.iloc[-2 - i] for i in range(3) if n > 2 + i)
        pullback = confirmed_trend_up and super_trend_up and rs_pass and pb_dipped and pb_vol_dry and pb_reclaim and (q_pass_gri if regime == 1 else q_pass_trend)

        # C: SQUEEZE EXPANSION
        sq_exp = regime >= 1 and confirmed_trend_up and rs_pass and sq_release_last and atr_expanding and c.iloc[-1] > sq_basis.iloc[-1] and clv >= 0.4 and q_pass_gri

        # D: MEAN REVERSION
        mr_bb = l.iloc[-1] <= bb_lower.iloc[-1] and c.iloc[-1] > bb_lower.iloc[-1]
        mr_donch = l.iloc[-1] <= donch_lower.iloc[-2] and c.iloc[-1] > donch_lower.iloc[-2] if len(donch_lower) > 1 else False
        mean_rev = regime <= 1 and (mr_bb or mr_donch) and rsi_short.iloc[-1] < MR_RSI_THRESH and rsi_short.iloc[-1] > rsi_short.iloc[-2] and c.iloc[-1] > ema_s.iloc[-1] * 0.90

        # E: REVERSAL (eski DONUS)
        ema55_cross = (c > ema_s) & (c.shift(1) <= ema_s.shift(1))
        recent_e55 = any(ema55_cross.iloc[-i] for i in range(1, 11) if -i >= -n)
        recent_wt_cross = any(wt['cross_up'].iloc[-i] for i in range(1, 11) if -i >= -n)
        ema55_dist = (c.iloc[-1] - ema_s.iloc[-1]) / ema_s.iloc[-1] * 100
        approaching = -3 < ema55_dist < 3 and c.iloc[-1] > c.iloc[-4] and c.iloc[-4] > c.iloc[-7] if n > 7 else False
        reversal = (recent_e55 and recent_wt_cross and rs_pass and c.iloc[-1] > ema_s.iloc[-1]) or \
                   (recent_wt_cross and rs_pass and approaching and v.iloc[-1] > vol_sma20.iloc[-1])

        # F: EARLY
        sw_hl = h.rolling(20).max().shift(1)
        struct_break = c.iloc[-1] > sw_hl.iloc[-1] if not np.isnan(sw_hl.iloc[-1]) else False
        mom_up5 = (c.iloc[-1] - c.iloc[-6]) / c.iloc[-6] * 100 if n > 6 else 0
        green_cnt = sum(1 for i in range(5) if c.iloc[-1 - i] > o.iloc[-1 - i])
        adx_turn = adx_val > adx_s.iloc[-2] and adx_s.iloc[-2] > adx_s.iloc[-3] if n > 3 else False
        early_rsi = calc_rsi(c, 14).iloc[-1]
        highest5 = c.rolling(5).max().iloc[-1]
        early_struct = struct_break and v.iloc[-1] > vol_sma20.iloc[-1] * 1.2 and adx_turn
        early_mom = mom_up5 > 5.0 and c.iloc[-1] >= highest5 and green_cnt >= 3 and v.iloc[-1] > vol_sma20.iloc[-1] * 1.2
        early = regime <= 1 and (early_struct or early_mom) and early_rsi < 75

        # G: COMBO
        combo_base = (wt_cross_up or wt_recent) and wt_bullish and pmax_long
        combo_plus = combo_base and choch_tight
        combo_bos = combo_base and bos_tight and not choch_tight

        # Debug
        if dbg:
            if wt_cross_up or wt_recent:
                dbg['wt_recent'] += 1
            if pmax_long:
                dbg['pmax_long'] += 1
            if recent_bos:
                dbg['bos'] += 1
            if recent_choch:
                dbg['choch'] += 1
            if combo_base:
                dbg['combo_base'] += 1
            if combo_plus:
                dbg['combo_plus'] += 1
            if combo_bos:
                dbg['combo_bos'] += 1
            if not rs_pass:
                dbg['rs_fail'] += 1
            if strong:
                dbg['strong_check'] += 1
            if weak:
                dbg['weak_check'] += 1
            if reversal:
                dbg['donus_check'] += 1
            if early:
                dbg['early_check'] += 1
            if pullback:
                dbg['pb_check'] += 1
            if sq_exp:
                dbg['sq_check'] += 1
            if mean_rev:
                dbg['mr_check'] += 1
            if confirmed_trend_up and super_trend_up:
                dbg['pb_trend_ok'] = dbg.get('pb_trend_ok', 0) + 1
                if pb_dipped:
                    dbg['pb_dipped'] = dbg.get('pb_dipped', 0) + 1
                if pb_vol_dry:
                    dbg['pb_vol_dry'] = dbg.get('pb_vol_dry', 0) + 1
                if pb_reclaim:
                    dbg['pb_reclaim'] = dbg.get('pb_reclaim', 0) + 1
            if regime >= 1 and confirmed_trend_up:
                dbg['sq_regime_ok'] = dbg.get('sq_regime_ok', 0) + 1
                if sq_release_last:
                    dbg['sq_release'] = dbg.get('sq_release', 0) + 1
                if atr_expanding:
                    dbg['sq_atr_exp'] = dbg.get('sq_atr_exp', 0) + 1
            if regime <= 1:
                dbg['mr_regime_ok'] = dbg.get('mr_regime_ok', 0) + 1
                if mr_bb:
                    dbg['mr_bb_hit'] = dbg.get('mr_bb_hit', 0) + 1
                if mr_donch:
                    dbg['mr_donch_hit'] = dbg.get('mr_donch_hit', 0) + 1
                if struct_break:
                    dbg['early_struct_brk'] = dbg.get('early_struct_brk', 0) + 1
                if early_mom:
                    dbg['early_mom_ok'] = dbg.get('early_mom_ok', 0) + 1

        # === SİNYAL PRİORİTE (yeni terminoloji) ===
        signal = None
        if combo_plus:
            signal = "COMBO+"
        elif combo_bos:
            signal = "COMBO"
        elif strong:
            signal = "STRONG"
        elif weak:
            signal = "WEAK"
        elif reversal:
            signal = "REVERSAL"
        elif early:
            signal = "EARLY"
        elif pullback:
            signal = "PULLBACK"
        elif sq_exp:
            signal = "SQUEEZE"
        elif mean_rev:
            signal = "MEANREV"

        if signal is None:
            has_wt = wt_recent or wt_cross_up
            has_pmax = pmax_long
            has_smc = recent_bos or recent_choch
            active = sum([has_wt, has_pmax, has_smc])
            if active >= 2 and rs_pass:
                signal = "PARTIAL"
            else:
                if dbg:
                    dbg['low_active'] += 1
                return None

        # === OVEREXTENDED (sinyal tespitinden hemen sonra, mod filtresinden önce) ===
        overext = calc_overextended(df, wt_data=wt)
        oe_score = overext['overext_score']

        # ============================================================
        # MOD SEÇİMİ + SİNYAL FİLTRELEME (Production Ruleset v1)
        # ============================================================
        ms_risk_on = market_state.get('risk_on', True) if market_state else True
        ms_weekly_st_up = market_state.get('weekly_st_up', True) if market_state else True

        # Mod seçimi
        if not ms_weekly_st_up:
            trade_mode = "CORE"
        elif ms_risk_on and rs_score >= MOMENTUM_RS_THRESH:
            trade_mode = "MOMENTUM"
        else:
            trade_mode = "CORE"

        # MOMENTUM mod filtreleri
        if trade_mode == "MOMENTUM":
            if signal not in MOMENTUM_SIGNALS:
                if dbg:
                    dbg['mode_filter'] = dbg.get('mode_filter', 0) + 1
                return None

        # CORE mod filtreleri
        if trade_mode == "CORE":
            if signal not in CORE_SIGNALS:
                if dbg:
                    dbg['mode_filter'] = dbg.get('mode_filter', 0) + 1
                return None
            # COMBO+ → CORE'da kapalı
            if signal == "COMBO+":
                if dbg:
                    dbg['mode_filter'] = dbg.get('mode_filter', 0) + 1
                return None
            # COMBO özel kurallar
            if signal == "COMBO":
                if rs_score >= COMBO_RS_MAX or oe_score > COMBO_OE_MAX or dist_ema_atr > COMBO_DIST_EMA_MAX or quality < CORE_Q_MIN_COMBO:
                    if dbg:
                        dbg['combo_filter'] = dbg.get('combo_filter', 0) + 1
                    return None
            # PARTIAL/STRONG quality gate
            if signal in ("PARTIAL", "STRONG"):
                if quality < CORE_Q_MIN_PARTIAL_STRONG:
                    if dbg:
                        dbg['q_filter'] = dbg.get('q_filter', 0) + 1
                    return None
            # EARLY quality gate
            if signal == "EARLY":
                if quality < CORE_Q_MIN_EARLY:
                    if dbg:
                        dbg['q_filter'] = dbg.get('q_filter', 0) + 1
                    return None

        if dbg:
            dbg['signal'] += 1

        # === POZİSYON BOYUTLANDIRMA ===
        if trade_mode == "MOMENTUM":
            pos_size = POS_SIZE_MOMENTUM.get(signal, 1.0)
        else:
            pos_size = POS_SIZE_CORE.get(signal, 1.0)

        # === STOP / TP ===
        stop_mult = {
            "COMBO+": COMBO_STOP, "COMBO": COMBO_STOP,
            "STRONG": TREND_STOP, "WEAK": TREND_STOP,
            "PULLBACK": GRI_STOP, "SQUEEZE": GRI_STOP, "EARLY": GRI_STOP,
            "REVERSAL": DONUS_STOP, "MEANREV": MR_STOP,
            "PARTIAL": TREND_STOP,
        }.get(signal, TREND_STOP)

        if signal in ("COMBO+", "COMBO"):
            atr_tp = close_price + atr_val * COMBO_TP
            if not np.isnan(nearest_ob) and nearest_ob > close_price and nearest_ob < atr_tp:
                tp_price = nearest_ob
                tp_src = "OB"
            else:
                tp_price = atr_tp
                tp_src = "ATR"
        elif signal == "MEANREV":
            tp_price = bb_basis.iloc[-1]
            tp_src = "BB"
        elif signal == "PULLBACK":
            tp_price = close_price + atr_val * GRI_TP
            tp_src = "ATR"
        elif signal == "REVERSAL":
            tp_price = close_price + atr_val * DONUS_TP
            tp_src = "ATR"
        elif signal == "WEAK":
            tp_price = close_price + atr_val * TREND_TP * 0.75
            tp_src = "ATR"
        elif signal == "EARLY":
            tp_price = close_price + atr_val * TREND_TP * 0.8
            tp_src = "ATR"
        else:
            tp_price = close_price + atr_val * TREND_TP
            tp_src = "ATR"

        stop_price = close_price - atr_val * stop_mult
        risk = close_price - stop_price
        reward = tp_price - close_price
        rr = reward / risk if risk > 0 else 0

        # === LOGGING TAGS ===
        rs_band = "hi" if rs_score >= 20 else "mid" if rs_score >= 5 else "lo"
        q_bucket = "A" if quality >= 75 else "B" if quality >= 50 else "C"
        atr_pct_bucket = "panic" if atr_pctile >= 0.85 else "high" if atr_pctile >= 0.7 else "normal"

        # === RECOVER (KC) metadata ===
        kc_score = 0
        kc_tags = []
        try:
            from dip import calc_recover_criteria
            kc_result = calc_recover_criteria(df, usd_df_raw=usd_df)
            kc_score = kc_result.get('kc_score', 0)
            kc_tags = kc_result.get('kc_tags', [])
        except Exception:
            pass

        return {
            'ticker': ticker,
            'signal': signal,
            'regime': regime_name,
            'regime_score': regime,
            'close': round(close_price, 2),
            'stop': round(stop_price, 2),
            'tp': round(tp_price, 2),
            'tp_src': tp_src,
            'rr': round(rr, 2),
            'atr': round(atr_val, 2),
            'quality': int(quality),
            'rs_score': round(rs_score, 1),
            'rs_pass': rs_pass,
            'rvol': round(rvol, 1),
            'bb_pctb': round(float(bb_pctb.iloc[-1]), 2) if not np.isnan(bb_pctb.iloc[-1]) else None,
            # Bileşenler
            'wt_cross': wt_cross_up,
            'wt_recent': wt_recent,
            'wt_bullish': wt_bullish,
            'wt_killed': wt_killed,
            'wt1': round(float(wt['wt1'].iloc[-1]), 1),
            'wt2': round(float(wt['wt2'].iloc[-1]), 1),
            'pmax_long': pmax_long,
            'smc_bos': recent_bos,
            'smc_choch': recent_choch,
            'bos_age': bos_age,
            'choch_age': choch_age,
            'swing_bias': swing_bias,
            'ob_resist': f"{ob['ob_resist_bot']:.2f}-{ob['ob_resist_top']:.2f}" if not np.isnan(ob.get('ob_resist_top', np.nan)) else None,
            # Trend
            'ema_trend': ema_trend_up,
            'supertrend': super_trend_up,
            'adx': round(adx_val, 1),
            'adx_slope': round(adx_slope, 2),
            'htf_adx': round(htf_adx, 1),
            'squeeze': bool(sqz_on.iloc[-1]),
            'sq_release': sq_release_last,
            # Overextended
            'overext_score': overext['overext_score'],
            'overext_tags': overext['overext_tags'],
            'overext_warning': overext['overext_warning'],
            # Meta
            'turnover_m': round(avg_turnover / 1e6, 1),
            'vol_ratio': round(rvol, 1),
            # Production Ruleset v1
            'trade_mode': trade_mode,
            'risk_on': ms_risk_on,
            'pos_size': pos_size,
            'atr_pctile': round(atr_pctile, 2),
            'dist_ema_atr': round(dist_ema_atr, 2),
            'rs_band': rs_band,
            'q_bucket': q_bucket,
            'atr_pct_bucket': atr_pct_bucket,
            # RECOVER metadata
            'kc_score': kc_score,
            'kc_tags': kc_tags,
        }
    except Exception as e:
        if dbg:
            dbg['exception'] += 1
            if dbg['exception'] <= 3:
                print(f"  [HATA] {ticker}: {type(e).__name__}: {e}")
        return None
