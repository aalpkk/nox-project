"""
NOX Project — Sideways Module
Module A: Range Mean Reversion (SIDEWAYS_MR)
Module B: Squeeze Breakout (SIDEWAYS_SQ)
Active only when market_state.sideways=True.
"""
import numpy as np
import pandas as pd
from markets.bist.config import (
    RVOL_THRESH, MIN_AVG_VOLUME_TL,
    ATR_PANIC_PCTILE, ATR_PCTILE_WINDOW,
    SIDEWAYS_DEPLOY_MODE,
    SIDEWAYS_MR_RSI_THRESH, SIDEWAYS_MR_ATR_PCTILE_MAX,
    SIDEWAYS_MR_RVOL_MIN, SIDEWAYS_MR_REWARD_ATR,
    SIDEWAYS_MR_SECTOR_RS_MAX, SIDEWAYS_MR_STOP_ATR, SIDEWAYS_MR_TIMEOUT,
    SIDEWAYS_SQ_BB_PCTILE_MAX, SIDEWAYS_SQ_ATR_PCTILE_MAX,
    SIDEWAYS_SQ_BODY_ATR, SIDEWAYS_SQ_HHV_PERIOD,
    SIDEWAYS_SQ_RR_TARGET, SIDEWAYS_SQ_TIMEOUT, SIDEWAYS_SQ_FT_MODE,
    POS_SIZE_SIDEWAYS,
)
from core.config import (
    BB_LEN, BB_MULT, ATR_LEN, RS_LEN1, RS_LEN2,
)
from markets.bist.config import RS_THRESHOLD
from core.indicators import (
    ema, sma, calc_atr, calc_rsi, calc_atr_percentile,
    calc_bb_width_percentile, calc_overextended, calc_wavetrend,
)


def analyze_sideways(ticker, df, xu_df, dbg=None, usd_df=None, market_state=None):
    """
    Sideways analiz -- tek hisse.
    Module A (SIDEWAYS_MR): Range Mean Reversion, only when weekly_st_up=True
    Module B (SIDEWAYS_SQ): Squeeze Breakout, active regardless of weekly_st
    Returns dict compatible with analyze_regime() output, or None.
    """
    try:
        if dbg:
            dbg['total'] = dbg.get('total', 0) + 1

        # === GATE: sideways must be True ===
        if not market_state or not market_state.get('sideways'):
            return None

        n = len(df)
        if n < 100:
            return None

        c = df['Close']
        h = df['High']
        l = df['Low']
        o = df['Open']
        v = df['Volume']

        # === ATR ===
        atr_s = calc_atr(df, ATR_LEN)
        atr_val = float(atr_s.iloc[-1])
        if np.isnan(atr_val) or atr_val == 0:
            if dbg:
                dbg['no_atr'] = dbg.get('no_atr', 0) + 1
            return None

        # === Volume filter ===
        vol_sma20 = sma(v, 20)
        avg_turnover = float(vol_sma20.iloc[-1] * c.iloc[-1])
        if avg_turnover < MIN_AVG_VOLUME_TL:
            if dbg:
                dbg['low_vol'] = dbg.get('low_vol', 0) + 1
            return None

        # === Panic block ===
        atr_pctile_s = calc_atr_percentile(df, ATR_LEN, ATR_PCTILE_WINDOW)
        atr_pctile = float(atr_pctile_s.iloc[-1])
        if atr_pctile >= ATR_PANIC_PCTILE:
            if dbg:
                dbg['panic_block'] = dbg.get('panic_block', 0) + 1
            return None

        # === Shared indicators ===
        close_price = float(c.iloc[-1])
        rvol = float(v.iloc[-1] / vol_sma20.iloc[-1]) if float(vol_sma20.iloc[-1]) > 0 else 0

        # BB
        bb_mid = sma(c, BB_LEN)
        bb_dev = c.rolling(BB_LEN).std() * BB_MULT
        bb_upper = bb_mid + bb_dev
        bb_lower = bb_mid - bb_dev

        # BB width percentile (stock-level)
        bb_w_pctile = calc_bb_width_percentile(df, BB_LEN, BB_MULT, 100)
        bb_w_pctile_val = float(bb_w_pctile.iloc[-1])

        # RSI
        rsi14 = calc_rsi(c, 14)
        rsi_val = float(rsi14.iloc[-1]) if not np.isnan(rsi14.iloc[-1]) else 50

        # RS score
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

        # Quality (same pattern as regime.py)
        candle_range = float(h.iloc[-1] - l.iloc[-1])
        clv = (close_price - float(l.iloc[-1])) / candle_range if candle_range > 0 else 0.5
        upper_wick = float(h.iloc[-1]) - max(close_price, float(o.iloc[-1]))
        wick_ratio = upper_wick / candle_range if candle_range > 0 else 0
        range_atr = candle_range / atr_val if atr_val > 0 else 0

        rvol_s = 25 if rvol >= 2 else 20 if rvol >= RVOL_THRESH else 10 if rvol >= 1 else 0
        clv_s = 25 if clv >= 0.75 else 15 if clv >= 0.5 else 5 if clv >= 0.25 else 0
        wick_s = 25 if wick_ratio <= 0.15 else 15 if wick_ratio <= 0.3 else 5 if wick_ratio <= 0.5 else 0
        range_s_val = 25 if range_atr >= 1.2 else 15 if range_atr >= 0.8 else 5 if range_atr >= 0.5 else 0
        quality = rvol_s + clv_s + wick_s + range_s_val

        # Overextended
        wt = calc_wavetrend(df)
        overext = calc_overextended(df, wt_data=wt)

        weekly_st_up = market_state.get('weekly_st_up', False)

        # ============================================================
        # MODULE A: Range Mean Reversion (SIDEWAYS_MR)
        # Only active when weekly_st_up=True
        # ============================================================
        mr_signal = None
        if weekly_st_up and SIDEWAYS_DEPLOY_MODE in ("full",):
            # Band reclaim: Low < BB_lower AND Close > BB_lower
            band_reclaim = (float(l.iloc[-1]) < float(bb_lower.iloc[-1]) and
                            close_price > float(bb_lower.iloc[-1]))

            mr_rsi_ok = rsi_val < SIDEWAYS_MR_RSI_THRESH
            mr_atr_ok = atr_pctile < SIDEWAYS_MR_ATR_PCTILE_MAX
            mr_rvol_ok = rvol >= SIDEWAYS_MR_RVOL_MIN
            mr_rs_ok = rs_score < SIDEWAYS_MR_SECTOR_RS_MAX

            # Reward: dist_to_BB_mid / ATR >= threshold
            bb_mid_val = float(bb_mid.iloc[-1])
            dist_to_mid = bb_mid_val - close_price
            reward_atr = dist_to_mid / atr_val if atr_val > 0 else 0
            mr_reward_ok = reward_atr >= SIDEWAYS_MR_REWARD_ATR

            if band_reclaim and mr_rsi_ok and mr_atr_ok and mr_rvol_ok and mr_rs_ok and mr_reward_ok:
                mr_signal = "SIDEWAYS_MR"

            if dbg:
                dbg['mr_band_reclaim'] = dbg.get('mr_band_reclaim', 0) + int(bool(band_reclaim))
                dbg['mr_checks'] = dbg.get('mr_checks', 0) + 1
                if mr_signal:
                    dbg['mr_signal'] = dbg.get('mr_signal', 0) + 1

        # ============================================================
        # MODULE B: Squeeze Breakout (SIDEWAYS_SQ)
        # Active regardless of weekly_st
        # ============================================================
        sq_signal = None
        if SIDEWAYS_DEPLOY_MODE in ("full", "squeeze_only"):
            sq_bb_ok = bb_w_pctile_val < SIDEWAYS_SQ_BB_PCTILE_MAX
            sq_atr_ok = atr_pctile < SIDEWAYS_SQ_ATR_PCTILE_MAX

            # Close > HHV(Close, 20) — breakout above recent range
            hhv = c.rolling(SIDEWAYS_SQ_HHV_PERIOD).max()
            # Breakout: current close > previous period's highest close
            sq_breakout = close_price > float(hhv.iloc[-2]) if n > 1 and not np.isnan(hhv.iloc[-2]) else False

            # Strong candle: body > 0.7 * ATR
            body = abs(close_price - float(o.iloc[-1]))
            sq_body_ok = body > SIDEWAYS_SQ_BODY_ATR * atr_val

            # Follow-through mode
            if SIDEWAYS_SQ_FT_MODE == "before_entry":
                # Require follow-through: yesterday also closed strong
                ft_ok = (n >= 2 and float(c.iloc[-2]) > float(o.iloc[-2]) and
                         float(c.iloc[-2]) > float(c.iloc[-3]) if n >= 3 else True)
            else:
                # after_entry: no FT filter at entry
                ft_ok = True

            if sq_bb_ok and sq_atr_ok and sq_breakout and sq_body_ok and ft_ok:
                sq_signal = "SIDEWAYS_SQ"

            if dbg:
                dbg['sq_bb_ok'] = dbg.get('sq_bb_ok', 0) + int(bool(sq_bb_ok))
                dbg['sq_breakout'] = dbg.get('sq_breakout', 0) + int(bool(sq_breakout))
                dbg['sq_checks'] = dbg.get('sq_checks', 0) + 1
                if sq_signal:
                    dbg['sq_signal'] = dbg.get('sq_signal', 0) + 1

        # ============================================================
        # SIGNAL SELECTION (Module B has priority over Module A)
        # ============================================================
        signal = sq_signal or mr_signal

        if signal is None:
            if dbg:
                dbg['no_signal'] = dbg.get('no_signal', 0) + 1
            return None

        # === DEPLOY GATE ===
        if SIDEWAYS_DEPLOY_MODE == "log_only":
            if dbg:
                dbg['log_only'] = dbg.get('log_only', 0) + 1
            # Log but return None — signal computed but not acted upon
            return None

        if dbg:
            dbg['signal'] = dbg.get('signal', 0) + 1

        # === STOP / TP ===
        if signal == "SIDEWAYS_MR":
            tp_price = float(bb_mid.iloc[-1])
            stop_price = float(l.iloc[-1]) - SIDEWAYS_MR_STOP_ATR * atr_val
            tp_src = "BB"
            timeout = SIDEWAYS_MR_TIMEOUT
        else:  # SIDEWAYS_SQ
            risk_pct = (close_price - float(l.iloc[-1])) / close_price if close_price > 0 else 0.02
            risk_pct = max(risk_pct, 0.005)  # minimum risk floor
            tp_price = close_price * (1 + SIDEWAYS_SQ_RR_TARGET * risk_pct)
            stop_price = float(l.iloc[-1])  # Day0 low
            tp_src = "RR"
            timeout = SIDEWAYS_SQ_TIMEOUT

        risk = close_price - stop_price
        reward = tp_price - close_price
        rr = reward / risk if risk > 0 else 0

        # Position sizing
        pos_size = POS_SIZE_SIDEWAYS.get(signal, 0.6)

        return {
            'ticker': ticker,
            'signal': signal,
            'regime': 'SIDEWAYS',
            'regime_score': 0,
            'close': round(close_price, 2),
            'stop': round(stop_price, 2),
            'tp': round(tp_price, 2),
            'tp_src': tp_src,
            'rr': round(rr, 2),
            'atr': round(atr_val, 2),
            'quality': int(quality),
            'rs_score': round(rs_score, 1),
            'rs_pass': rs_score > RS_THRESHOLD,
            'rvol': round(rvol, 1),
            'bb_pctb': round(float((c.iloc[-1] - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1])), 2) if float(bb_upper.iloc[-1] - bb_lower.iloc[-1]) > 0 else None,
            'bb_w_pctile': round(bb_w_pctile_val, 1),
            # Indicators
            'wt1': round(float(wt['wt1'].iloc[-1]), 1),
            'wt2': round(float(wt['wt2'].iloc[-1]), 1),
            'rsi': round(rsi_val, 1),
            # Overextended
            'overext_score': overext['overext_score'],
            'overext_tags': overext['overext_tags'],
            'overext_warning': overext['overext_warning'],
            # Meta
            'turnover_m': round(avg_turnover / 1e6, 1),
            'vol_ratio': round(rvol, 1),
            # Sideways-specific
            'trade_mode': 'SIDEWAYS',
            'risk_on': False,
            'pos_size': pos_size,
            'atr_pctile': round(atr_pctile, 2),
            'timeout': timeout,
            'weekly_st_up': weekly_st_up,
            'module': 'A' if signal == 'SIDEWAYS_MR' else 'B',
            # Compatibility fields
            'ob_resist': None,
            'bos_age': None,
            'choch_age': None,
            'kc_score': 0,
            'kc_tags': [],
            'dist_ema_atr': round(float((c.iloc[-1] - ema(c, 20).iloc[-1]) / atr_val), 2) if atr_val > 0 else 0,
        }
    except Exception as e:
        if dbg:
            dbg['exception'] = dbg.get('exception', 0) + 1
            if dbg.get('exception', 0) <= 3:
                print(f"  [HATA] {ticker} sideways: {type(e).__name__}: {e}")
        return None
