#!/usr/bin/env python3
"""
NOX Sideways Diagnostic — Detaylı funnel raporu.
Her kriterde kaç hisse eleniyor, hangisi nerede takılıyor gösterir.
"""
import sys, os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from core.config import BB_LEN, BB_MULT, ATR_LEN, RS_LEN1, RS_LEN2
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
)
from core.indicators import (
    ema, sma, calc_atr, calc_rsi, calc_atr_percentile,
    calc_bb_width_percentile, calc_xu100_market_state,
)
from markets.bist.data import fetch_data, fetch_benchmark, get_all_bist_tickers


def main():
    print("=" * 70)
    print("NOX SIDEWAYS DIAGNOSTIC — Detaylı Funnel Raporu")
    print("=" * 70)

    # --- Veri yükle ---
    tickers = get_all_bist_tickers()
    print(f"\n📡 {len(tickers)} ticker yükleniyor...")
    all_data = fetch_data(tickers, period="1y")
    xu_df = fetch_benchmark(period="1y")
    total = len(all_data)
    print(f"✅ {total} hisse verisi alındı\n")

    # --- Market state ---
    market_state = calc_xu100_market_state(xu_df) if xu_df is not None else {}
    market_state['sideways'] = True  # force

    weekly_st_up = market_state.get('weekly_st_up', False)
    print(f"🌐 Market State:")
    print(f"   weekly_st_up = {weekly_st_up}")
    print(f"   risk_on      = {market_state.get('risk_on')}")
    print(f"   sideways     = {market_state.get('sideways')}")
    print(f"   DEPLOY_MODE  = {SIDEWAYS_DEPLOY_MODE}")
    print()

    # --- Threshold bilgisi ---
    print("📐 Eşik Değerleri:")
    print(f"   [Genel]  MIN_AVG_VOLUME_TL = {MIN_AVG_VOLUME_TL:,.0f}")
    print(f"   [Genel]  ATR_PANIC_PCTILE  = {ATR_PANIC_PCTILE}")
    print(f"   [SQ] BB_W_PCTILE_MAX  = {SIDEWAYS_SQ_BB_PCTILE_MAX}")
    print(f"   [SQ] ATR_PCTILE_MAX   = {SIDEWAYS_SQ_ATR_PCTILE_MAX}")
    print(f"   [SQ] HHV_PERIOD       = {SIDEWAYS_SQ_HHV_PERIOD}")
    print(f"   [SQ] BODY_ATR         = {SIDEWAYS_SQ_BODY_ATR}")
    print(f"   [SQ] FT_MODE          = {SIDEWAYS_SQ_FT_MODE}")
    print(f"   [MR] RSI_THRESH       = {SIDEWAYS_MR_RSI_THRESH}")
    print(f"   [MR] ATR_PCTILE_MAX   = {SIDEWAYS_MR_ATR_PCTILE_MAX}")
    print(f"   [MR] RVOL_MIN         = {SIDEWAYS_MR_RVOL_MIN}")
    print(f"   [MR] REWARD_ATR       = {SIDEWAYS_MR_REWARD_ATR}")
    print(f"   [MR] SECTOR_RS_MAX    = {SIDEWAYS_MR_SECTOR_RS_MAX}")
    print(f"   [MR] weekly_st_up gerekli = True (şu an: {weekly_st_up})")
    print()

    # --- Funnel sayaçları ---
    gate_short_data = []
    gate_no_atr = []
    gate_low_vol = []
    gate_panic = []
    passed_common = []  # Ortak filtreleri geçenler

    # Module B (SQ) detay
    sq_candidates = []    # ortak geçenler (hepsi SQ'ya girer)
    sq_fail_bb = []
    sq_fail_atr = []
    sq_fail_breakout = []
    sq_fail_body = []
    sq_fail_ft = []
    sq_pass = []

    # Module A (MR) detay
    mr_candidates = []
    mr_fail_band = []
    mr_fail_rsi = []
    mr_fail_atr = []
    mr_fail_rvol = []
    mr_fail_rs = []
    mr_fail_reward = []
    mr_pass = []

    # Yakın kaçanlar için detaylı kayıt
    sq_near_miss = []
    mr_near_miss = []

    for ticker, df in all_data.items():
        n = len(df)
        if n < 100:
            gate_short_data.append(ticker)
            continue

        c = df['Close']
        h = df['High']
        l = df['Low']
        o = df['Open']
        v = df['Volume']

        # ATR
        atr_s = calc_atr(df, ATR_LEN)
        atr_val = float(atr_s.iloc[-1])
        if np.isnan(atr_val) or atr_val == 0:
            gate_no_atr.append(ticker)
            continue

        # Volume
        vol_sma20 = sma(v, 20)
        avg_turnover = float(vol_sma20.iloc[-1] * c.iloc[-1])
        if avg_turnover < MIN_AVG_VOLUME_TL:
            gate_low_vol.append(ticker)
            continue

        # Panic
        atr_pctile_s = calc_atr_percentile(df, ATR_LEN, ATR_PCTILE_WINDOW)
        atr_pctile = float(atr_pctile_s.iloc[-1])
        if atr_pctile >= ATR_PANIC_PCTILE:
            gate_panic.append(ticker)
            continue

        passed_common.append(ticker)

        # --- Ortak göstergeler ---
        close_price = float(c.iloc[-1])
        rvol = float(v.iloc[-1] / vol_sma20.iloc[-1]) if float(vol_sma20.iloc[-1]) > 0 else 0

        bb_mid = sma(c, BB_LEN)
        bb_dev = c.rolling(BB_LEN).std() * BB_MULT
        bb_upper = bb_mid + bb_dev
        bb_lower = bb_mid - bb_dev

        bb_w_pctile = calc_bb_width_percentile(df, BB_LEN, BB_MULT, 100)
        bb_w_pctile_val = float(bb_w_pctile.iloc[-1])

        rsi14 = calc_rsi(c, 14)
        rsi_val = float(rsi14.iloc[-1]) if not np.isnan(rsi14.iloc[-1]) else 50

        # RS
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

        # ============================================================
        # MODULE B: Squeeze Breakout (SIDEWAYS_SQ)
        # ============================================================
        sq_bb_ok = bb_w_pctile_val < SIDEWAYS_SQ_BB_PCTILE_MAX
        sq_atr_ok = atr_pctile < SIDEWAYS_SQ_ATR_PCTILE_MAX

        hhv = c.rolling(SIDEWAYS_SQ_HHV_PERIOD).max()
        sq_breakout = close_price > float(hhv.iloc[-2]) if n > 1 and not np.isnan(hhv.iloc[-2]) else False

        body = abs(close_price - float(o.iloc[-1]))
        sq_body_ok = body > SIDEWAYS_SQ_BODY_ATR * atr_val

        if SIDEWAYS_SQ_FT_MODE == "before_entry":
            ft_ok = (n >= 2 and float(c.iloc[-2]) > float(o.iloc[-2]) and
                     float(c.iloc[-2]) > float(c.iloc[-3]) if n >= 3 else True)
        else:
            ft_ok = True

        sq_checks = {
            'bb_w': bb_w_pctile_val,
            'bb_ok': sq_bb_ok,
            'atr_pct': atr_pctile,
            'atr_ok': sq_atr_ok,
            'breakout': sq_breakout,
            'body': round(body / atr_val, 2) if atr_val > 0 else 0,
            'body_ok': sq_body_ok,
            'ft_ok': ft_ok,
            'close': close_price,
            'hhv_prev': round(float(hhv.iloc[-2]), 2) if n > 1 and not np.isnan(hhv.iloc[-2]) else None,
        }

        # SQ eleme detayı
        sq_reasons = []
        if not sq_bb_ok:
            sq_fail_bb.append(ticker)
            sq_reasons.append(f"BB_W%={bb_w_pctile_val:.0f}>{SIDEWAYS_SQ_BB_PCTILE_MAX}")
        if not sq_atr_ok:
            sq_fail_atr.append(ticker)
            sq_reasons.append(f"ATR%={atr_pctile:.2f}>{SIDEWAYS_SQ_ATR_PCTILE_MAX}")
        if not sq_breakout:
            sq_fail_breakout.append(ticker)
            sq_reasons.append("breakout=NO")
        if not sq_body_ok:
            sq_fail_body.append(ticker)
            sq_reasons.append(f"body/ATR={body/atr_val:.2f}<{SIDEWAYS_SQ_BODY_ATR}")
        if not ft_ok:
            sq_fail_ft.append(ticker)
            sq_reasons.append("FT=NO")

        if sq_bb_ok and sq_atr_ok and sq_breakout and sq_body_ok and ft_ok:
            sq_pass.append(ticker)
        else:
            # Yakın kaçan: max 2 kriter fail
            fail_count = sum([not sq_bb_ok, not sq_atr_ok, not sq_breakout, not sq_body_ok, not ft_ok])
            if fail_count <= 2:
                sq_near_miss.append({
                    'ticker': ticker,
                    'fail_count': fail_count,
                    'reasons': sq_reasons,
                    **sq_checks,
                })

        # ============================================================
        # MODULE A: Range Mean Reversion (SIDEWAYS_MR)
        # ============================================================
        if weekly_st_up and SIDEWAYS_DEPLOY_MODE in ("full",):
            band_reclaim = (float(l.iloc[-1]) < float(bb_lower.iloc[-1]) and
                            close_price > float(bb_lower.iloc[-1]))
            mr_rsi_ok = rsi_val < SIDEWAYS_MR_RSI_THRESH
            mr_atr_ok = atr_pctile < SIDEWAYS_MR_ATR_PCTILE_MAX
            mr_rvol_ok = rvol >= SIDEWAYS_MR_RVOL_MIN
            mr_rs_ok = rs_score < SIDEWAYS_MR_SECTOR_RS_MAX

            bb_mid_val = float(bb_mid.iloc[-1])
            dist_to_mid = bb_mid_val - close_price
            reward_atr = dist_to_mid / atr_val if atr_val > 0 else 0
            mr_reward_ok = reward_atr >= SIDEWAYS_MR_REWARD_ATR

            mr_candidates.append(ticker)

            mr_reasons = []
            if not band_reclaim:
                mr_fail_band.append(ticker)
                mr_reasons.append(f"band_reclaim=NO (L={float(l.iloc[-1]):.2f} BBlo={float(bb_lower.iloc[-1]):.2f})")
            if not mr_rsi_ok:
                mr_fail_rsi.append(ticker)
                mr_reasons.append(f"RSI={rsi_val:.1f}>{SIDEWAYS_MR_RSI_THRESH}")
            if not mr_atr_ok:
                mr_fail_atr.append(ticker)
                mr_reasons.append(f"ATR%={atr_pctile:.2f}>{SIDEWAYS_MR_ATR_PCTILE_MAX}")
            if not mr_rvol_ok:
                mr_fail_rvol.append(ticker)
                mr_reasons.append(f"RVOL={rvol:.1f}<{SIDEWAYS_MR_RVOL_MIN}")
            if not mr_rs_ok:
                mr_fail_rs.append(ticker)
                mr_reasons.append(f"RS={rs_score:.1f}>{SIDEWAYS_MR_SECTOR_RS_MAX}")
            if not mr_reward_ok:
                mr_fail_reward.append(ticker)
                mr_reasons.append(f"reward/ATR={reward_atr:.2f}<{SIDEWAYS_MR_REWARD_ATR}")

            if band_reclaim and mr_rsi_ok and mr_atr_ok and mr_rvol_ok and mr_rs_ok and mr_reward_ok:
                mr_pass.append(ticker)
            else:
                fail_count = sum([not band_reclaim, not mr_rsi_ok, not mr_atr_ok, not mr_rvol_ok, not mr_rs_ok, not mr_reward_ok])
                if fail_count <= 2:
                    mr_near_miss.append({
                        'ticker': ticker,
                        'fail_count': fail_count,
                        'reasons': mr_reasons,
                        'rsi': rsi_val,
                        'atr_pct': atr_pctile,
                        'rvol': rvol,
                        'rs': rs_score,
                        'reward_atr': round(reward_atr, 2),
                        'band_reclaim': band_reclaim,
                        'close': close_price,
                        'bb_lower': round(float(bb_lower.iloc[-1]), 2),
                        'bb_mid': round(bb_mid_val, 2),
                    })

    # ============================================================
    # RAPOR
    # ============================================================
    W = 70
    print("=" * W)
    print("FUNNEL RAPORU — Ortak Filtreler")
    print("=" * W)
    print(f"  Toplam veri alınan     : {total}")
    print(f"  ❌ Yetersiz veri (<100) : {len(gate_short_data)}")
    print(f"  ❌ ATR yok/sıfır        : {len(gate_no_atr)}")
    print(f"  ❌ Düşük hacim          : {len(gate_low_vol)}")
    print(f"  ❌ Panic block (ATR%≥{ATR_PANIC_PCTILE})  : {len(gate_panic)}")
    print(f"  ✅ Ortak filtreleri geçen: {len(passed_common)}")
    if gate_panic:
        print(f"\n  Panic'e takılanlar: {', '.join(sorted(gate_panic)[:20])}{'...' if len(gate_panic)>20 else ''}")

    # --- MODULE B: SQUEEZE ---
    print()
    print("=" * W)
    print(f"MODULE B: SQUEEZE BREAKOUT — {len(passed_common)} hisse değerlendirildi")
    print("=" * W)

    # Kaç hisse hangi kriterde fail
    print(f"\n  Kriter bazında FAIL sayıları (bir hisse birden fazla kriterde fail olabilir):")
    print(f"  ❌ BB_W% ≥ {SIDEWAYS_SQ_BB_PCTILE_MAX}      : {len(sq_fail_bb):>4}  ({len(sq_fail_bb)/len(passed_common)*100:.0f}%)")
    print(f"  ❌ ATR% ≥ {SIDEWAYS_SQ_ATR_PCTILE_MAX}       : {len(sq_fail_atr):>4}  ({len(sq_fail_atr)/len(passed_common)*100:.0f}%)")
    print(f"  ❌ Breakout yok          : {len(sq_fail_breakout):>4}  ({len(sq_fail_breakout)/len(passed_common)*100:.0f}%)")
    print(f"  ❌ Body < {SIDEWAYS_SQ_BODY_ATR}×ATR         : {len(sq_fail_body):>4}  ({len(sq_fail_body)/len(passed_common)*100:.0f}%)")
    if SIDEWAYS_SQ_FT_MODE == "before_entry":
        print(f"  ❌ Follow-through yok    : {len(sq_fail_ft):>4}  ({len(sq_fail_ft)/len(passed_common)*100:.0f}%)")
    else:
        print(f"  ℹ️  Follow-through       : Devre dışı (FT_MODE=after_entry)")
    print(f"\n  ✅ SQ SİNYAL ÜRETTİ     : {len(sq_pass)}")
    if sq_pass:
        print(f"     → {', '.join(sorted(sq_pass))}")

    # SQ Yakın kaçanlar
    sq_near_miss.sort(key=lambda x: x['fail_count'])
    if sq_near_miss:
        print(f"\n  🔶 Yakın Kaçanlar (≤2 kriter fail, en yakınlar):")
        for nm in sq_near_miss[:15]:
            reasons_str = " | ".join(nm['reasons'])
            print(f"     {nm['ticker']:8s} [{nm['fail_count']} fail] BB_W%={nm['bb_w']:.0f} ATR%={nm['atr_pct']:.2f} "
                  f"body/ATR={nm['body']:.2f} brk={'✓' if nm['breakout'] else '✗'} → {reasons_str}")

    # --- MODULE A: MEAN REVERSION ---
    print()
    print("=" * W)
    if not weekly_st_up:
        print(f"MODULE A: MEAN REVERSION — DEVRE DIŞI (weekly_st_up=False)")
        print("=" * W)
        print(f"\n  ⚠️  MR modülü sadece weekly SuperTrend UP iken aktif.")
        print(f"  Şu an weekly SuperTrend DOWN olduğu için hiçbir hisse değerlendirilmedi.")
    elif SIDEWAYS_DEPLOY_MODE not in ("full",):
        print(f"MODULE A: MEAN REVERSION — DEVRE DIŞI (DEPLOY_MODE={SIDEWAYS_DEPLOY_MODE})")
        print("=" * W)
    else:
        print(f"MODULE A: MEAN REVERSION — {len(mr_candidates)} hisse değerlendirildi")
        print("=" * W)
        print(f"\n  Kriter bazında FAIL sayıları:")
        print(f"  ❌ Band reclaim yok      : {len(mr_fail_band):>4}  ({len(mr_fail_band)/max(len(mr_candidates),1)*100:.0f}%)")
        print(f"  ❌ RSI ≥ {SIDEWAYS_MR_RSI_THRESH}             : {len(mr_fail_rsi):>4}  ({len(mr_fail_rsi)/max(len(mr_candidates),1)*100:.0f}%)")
        print(f"  ❌ ATR% ≥ {SIDEWAYS_MR_ATR_PCTILE_MAX}         : {len(mr_fail_atr):>4}  ({len(mr_fail_atr)/max(len(mr_candidates),1)*100:.0f}%)")
        print(f"  ❌ RVOL < {SIDEWAYS_MR_RVOL_MIN}            : {len(mr_fail_rvol):>4}  ({len(mr_fail_rvol)/max(len(mr_candidates),1)*100:.0f}%)")
        print(f"  ❌ RS ≥ {SIDEWAYS_MR_SECTOR_RS_MAX}             : {len(mr_fail_rs):>4}  ({len(mr_fail_rs)/max(len(mr_candidates),1)*100:.0f}%)")
        print(f"  ❌ Reward/ATR < {SIDEWAYS_MR_REWARD_ATR}     : {len(mr_fail_reward):>4}  ({len(mr_fail_reward)/max(len(mr_candidates),1)*100:.0f}%)")
        print(f"\n  ✅ MR SİNYAL ÜRETTİ     : {len(mr_pass)}")
        if mr_pass:
            print(f"     → {', '.join(sorted(mr_pass))}")

        mr_near_miss.sort(key=lambda x: x['fail_count'])
        if mr_near_miss:
            print(f"\n  🔶 Yakın Kaçanlar (≤2 kriter fail):")
            for nm in mr_near_miss[:15]:
                reasons_str = " | ".join(nm['reasons'])
                print(f"     {nm['ticker']:8s} [{nm['fail_count']} fail] RSI={nm['rsi']:.1f} ATR%={nm['atr_pct']:.2f} "
                      f"RVOL={nm['rvol']:.1f} RS={nm['rs']:.1f} rew/ATR={nm['reward_atr']:.2f} "
                      f"band={'✓' if nm['band_reclaim'] else '✗'} → {reasons_str}")

    # --- ÖZET ---
    print()
    print("=" * W)
    print("ÖZET")
    print("=" * W)
    print(f"  {total} hisse tarandı")
    print(f"  {total - len(passed_common)} hisse ortak filtrelerde elendi:")
    print(f"      {len(gate_short_data)} yetersiz veri + {len(gate_no_atr)} ATR yok + {len(gate_low_vol)} düşük hacim + {len(gate_panic)} panic")
    print(f"  {len(passed_common)} hisse modüllere ulaştı")
    print(f"  Module B (SQ): {len(sq_pass)} sinyal")
    if weekly_st_up and SIDEWAYS_DEPLOY_MODE == "full":
        print(f"  Module A (MR): {len(mr_pass)} sinyal")
    else:
        print(f"  Module A (MR): DEVRE DIŞI")
    print(f"  Toplam sinyal: {len(sq_pass) + len(mr_pass)}")

    # En çok elenen kriter
    print(f"\n  📊 En büyük darboğazlar (SQ modülü):")
    bottlenecks = [
        (f"BB_W% ≥ {SIDEWAYS_SQ_BB_PCTILE_MAX}", len(sq_fail_bb)),
        (f"ATR% ≥ {SIDEWAYS_SQ_ATR_PCTILE_MAX}", len(sq_fail_atr)),
        ("Breakout yok", len(sq_fail_breakout)),
        (f"Body < {SIDEWAYS_SQ_BODY_ATR}×ATR", len(sq_fail_body)),
    ]
    bottlenecks.sort(key=lambda x: -x[1])
    for name, cnt in bottlenecks:
        bar = "█" * int(cnt / max(len(passed_common), 1) * 40)
        print(f"     {name:25s} {cnt:>4} {bar}")


if __name__ == "__main__":
    main()
