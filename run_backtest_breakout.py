#!/usr/bin/env python3
"""
Smart Breakout Targets — 2 Yıl Backtest
TP hit oranları, varma süreleri, 1G/3G/5G WR/ort/median
"""

import os
import sys
import warnings
import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ── Parametreler (run_smart_breakout.py ile aynı) ────────────
BB_LENGTH = 20
BB_MULT = 2.0
BB_WIDTH_THRESH = 1.60
ATR_LENGTH = 10
ATR_SMA_LENGTH = 20
ATR_SQUEEZE_RATIO = 2.00
MIN_SQUEEZE_BARS = 3
IMPULSE_ATR_MULT = 0.35
VOL_SMA_LENGTH = 20
VOL_MULT = 1.5
MAX_RANGE_ATR_MULT = 6.0
HTF_EMA_LENGTH = 50
ATR_SL_MULT = 0.5
TP_RATIOS = [1.0, 2.0, 3.0]

MAX_TRADE_BARS = 30  # TP/SL takibi max gün

SYMBOLS_FILE = Path(__file__).parent / "tools" / "bist_symbols.txt"


def load_symbols():
    if SYMBOLS_FILE.exists():
        return [
            line.strip() + ".IS"
            for line in SYMBOLS_FILE.read_text().splitlines()
            if line.strip()
        ]
    print("HATA: tools/bist_symbols.txt bulunamadı")
    sys.exit(1)


def calc_indicators(df):
    c, h, l, o, v = df["Close"], df["High"], df["Low"], df["Open"], df["Volume"]
    sma = c.rolling(BB_LENGTH).mean()
    std = c.rolling(BB_LENGTH).std()
    bb_upper = sma + BB_MULT * std
    bb_lower = sma - BB_MULT * std
    df["bb_width"] = (bb_upper - bb_lower) / sma
    df["bb_width_sma"] = df["bb_width"].rolling(BB_LENGTH).mean()
    tr = pd.concat(
        [h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1
    ).max(axis=1)
    df["atr"] = tr.rolling(ATR_LENGTH).mean()
    df["atr_sma"] = df["atr"].rolling(ATR_SMA_LENGTH).mean()
    df["vol_sma"] = v.rolling(VOL_SMA_LENGTH).mean()
    df["body"] = (c - o).abs()
    df["htf_ema"] = c.ewm(span=HTF_EMA_LENGTH, adjust=False).mean()
    df["sq_bb"] = df["bb_width"] < df["bb_width_sma"] * BB_WIDTH_THRESH
    df["sq_atr"] = df["atr"] < df["atr_sma"] * ATR_SQUEEZE_RATIO
    df["squeeze"] = df["sq_bb"] & df["sq_atr"]
    return df


def backtest_all(use_vol=True, use_htf=False, lookback_days=730):
    symbols = load_symbols()
    print(f"📡 {len(symbols)} sembol, {lookback_days} gün backtest...")
    print(f"   Vol filtre: {'ON' if use_vol else 'OFF'} | HTF filtre: {'ON' if use_htf else 'OFF'}")
    print()

    end = datetime.now()
    start = end - timedelta(days=lookback_days + 60)  # warmup

    all_trades = []
    errors = 0

    for i, sym in enumerate(symbols):
        ticker = sym.replace(".IS", "")
        if (i + 1) % 50 == 0:
            print(f"   ... {i+1}/{len(symbols)}")
        try:
            df = yf.download(sym, start=start, end=end, progress=False, timeout=10)
            if df is None or len(df) < 80:
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            trades = _backtest_single(df, ticker, use_vol, use_htf)
            all_trades.extend(trades)
        except Exception:
            errors += 1
            continue

    print(f"   Tamamlandı — {len(all_trades)} trade ({errors} hata)\n")

    if not all_trades:
        print("Trade bulunamadı.")
        return

    tdf = pd.DataFrame(all_trades)
    _print_results(tdf)
    _save_html_report(tdf)
    return tdf


def _backtest_single(df, ticker, use_vol, use_htf):
    df = calc_indicators(df).copy()
    df.reset_index(drop=True, inplace=True)
    n = len(df)
    trades = []

    # Squeeze dönemlerini bul
    squeezes = []
    in_sq = False
    sq_start = 0
    for i in range(n):
        if pd.isna(df["squeeze"].iloc[i]):
            continue
        if df["squeeze"].iloc[i]:
            if not in_sq:
                sq_start = i
                in_sq = True
        else:
            if in_sq:
                sq_len = i - sq_start
                if sq_len >= MIN_SQUEEZE_BARS:
                    squeezes.append({"start": sq_start, "end": i - 1, "length": sq_len})
                in_sq = False

    if not squeezes:
        return trades

    # Her squeeze sonrası kutu + kırılma ara
    used_until = 0  # çakışma önleme
    for sq in squeezes:
        sq_s, sq_e, sq_len = sq["start"], sq["end"], sq["length"]
        if sq_e < used_until:
            continue

        sq_high = df["High"].iloc[sq_s : sq_e + 1].max()
        sq_low = df["Low"].iloc[sq_s : sq_e + 1].min()
        atr_at_end = df["atr"].iloc[sq_e]

        if atr_at_end <= 0 or pd.isna(atr_at_end):
            continue

        # Aralık sınırlama
        if (sq_high - sq_low) > MAX_RANGE_ATR_MULT * atr_at_end:
            center = (sq_high + sq_low) / 2.0
            sq_high = center + 3.0 * atr_at_end
            sq_low = center - 3.0 * atr_at_end

        box_top = sq_high
        box_bot = sq_low

        # Kırılma ara (squeeze sonrasındaki barlar)
        for i in range(sq_e + 1, min(sq_e + 31, n)):  # max 30 bar bekle
            c_i = df["Close"].iloc[i]
            o_i = df["Open"].iloc[i]
            body_i = df["body"].iloc[i]
            atr_i = df["atr"].iloc[i]
            vol_i = df["Volume"].iloc[i]
            vol_sma_i = df["vol_sma"].iloc[i]
            htf_ema_i = df["htf_ema"].iloc[i]

            if atr_i <= 0 or pd.isna(atr_i):
                continue

            impulse_ok = body_i > atr_i * IMPULSE_ATR_MULT
            vol_ok = (not use_vol) or (vol_sma_i > 0 and vol_i > vol_sma_i * VOL_MULT)

            direction = None
            if c_i > box_top and c_i > o_i and impulse_ok and vol_ok:
                htf_ok = (not use_htf) or (c_i > htf_ema_i)
                if htf_ok:
                    direction = "LONG"
            elif c_i < box_bot and c_i < o_i and impulse_ok and vol_ok:
                htf_ok = (not use_htf) or (c_i < htf_ema_i)
                if htf_ok:
                    direction = "SHORT"

            if direction is None:
                continue

            # Kırılma bulundu — seviyeleri hesapla
            entry = c_i
            if direction == "LONG":
                sl = box_bot - atr_i * ATR_SL_MULT
                risk = abs(entry - sl)
                tp1 = entry + risk * TP_RATIOS[0]
                tp2 = entry + risk * TP_RATIOS[1]
                tp3 = entry + risk * TP_RATIOS[2]
            else:
                sl = box_top + atr_i * ATR_SL_MULT
                risk = abs(sl - entry)
                tp1 = entry - risk * TP_RATIOS[0]
                tp2 = entry - risk * TP_RATIOS[1]
                tp3 = entry - risk * TP_RATIOS[2]

            if risk <= 0:
                break

            # Sinyal gücü
            strength = 0
            if impulse_ok:
                strength += 1
            if use_vol and vol_sma_i > 0 and vol_i > vol_sma_i * VOL_MULT:
                strength += 1
            if use_htf and (
                (direction == "LONG" and c_i > htf_ema_i)
                or (direction == "SHORT" and c_i < htf_ema_i)
            ):
                strength += 1
            if sq_len >= 6:
                strength += 1

            # Forward analiz
            trade = {
                "ticker": ticker,
                "dir": direction,
                "entry": entry,
                "sl": sl,
                "tp1": tp1,
                "tp2": tp2,
                "tp3": tp3,
                "risk": risk,
                "risk_pct": risk / entry * 100,
                "strength": strength,
                "sq_bars": sq_len,
                "bo_bar": i,
            }

            # Tarih
            if isinstance(df.index, pd.RangeIndex):
                trade["date"] = ""
            else:
                trade["date"] = str(df.index[i].date()) if i < len(df.index) else ""

            # 1G, 3G, 5G returns
            for days, label in [(1, "1g"), (3, "3g"), (5, "5g")]:
                if i + days < n:
                    future_close = df["Close"].iloc[i + days]
                    if direction == "LONG":
                        ret = (future_close - entry) / entry * 100
                    else:
                        ret = (entry - future_close) / entry * 100
                    trade[f"ret_{label}"] = ret
                else:
                    trade[f"ret_{label}"] = np.nan

            # TP/SL hit analizi (max 30 gün)
            tp1_hit = tp2_hit = tp3_hit = sl_hit = False
            tp1_day = tp2_day = tp3_day = sl_day = np.nan
            trail_level = sl
            trail_result = "TIMEOUT"

            for j in range(1, min(MAX_TRADE_BARS + 1, n - i)):
                idx = i + j
                hj = df["High"].iloc[idx]
                lj = df["Low"].iloc[idx]

                if direction == "LONG":
                    # SL check (trail level)
                    if not sl_hit and lj <= trail_level:
                        sl_hit = True
                        sl_day = j
                        if not tp1_hit:
                            trail_result = "SL_LOSS"
                        else:
                            trail_result = "SL_TRAIL"
                        break
                    if not tp1_hit and hj >= tp1:
                        tp1_hit = True
                        tp1_day = j
                        trail_level = entry  # breakeven
                    if not tp2_hit and hj >= tp2:
                        tp2_hit = True
                        tp2_day = j
                        trail_level = tp1
                    if not tp3_hit and hj >= tp3:
                        tp3_hit = True
                        tp3_day = j
                        trail_level = tp2
                        trail_result = "TP3_WIN"
                        break
                else:
                    if not sl_hit and hj >= trail_level:
                        sl_hit = True
                        sl_day = j
                        if not tp1_hit:
                            trail_result = "SL_LOSS"
                        else:
                            trail_result = "SL_TRAIL"
                        break
                    if not tp1_hit and lj <= tp1:
                        tp1_hit = True
                        tp1_day = j
                        trail_level = entry
                    if not tp2_hit and lj <= tp2:
                        tp2_hit = True
                        tp2_day = j
                        trail_level = tp1
                    if not tp3_hit and lj <= tp3:
                        tp3_hit = True
                        tp3_day = j
                        trail_level = tp2
                        trail_result = "TP3_WIN"
                        break

            trade["tp1_hit"] = tp1_hit
            trade["tp2_hit"] = tp2_hit
            trade["tp3_hit"] = tp3_hit
            trade["sl_hit"] = sl_hit
            trade["tp1_day"] = tp1_day
            trade["tp2_day"] = tp2_day
            trade["tp3_day"] = tp3_day
            trade["sl_day"] = sl_day
            trade["trail_result"] = trail_result

            # Kâr hesabı (trail modunda)
            if trail_result == "TP3_WIN":
                # TP3'te çıkış — ancak trail ile TP2'de kilitlenmiş
                trade["trail_pnl_pct"] = (tp3 - entry) / entry * 100 if direction == "LONG" else (entry - tp3) / entry * 100
            elif trail_result == "SL_TRAIL":
                # Trail stop vuruldu — trail_level'da çıkış
                if tp2_hit:
                    exit_p = tp1
                elif tp1_hit:
                    exit_p = entry
                else:
                    exit_p = sl
                trade["trail_pnl_pct"] = (exit_p - entry) / entry * 100 if direction == "LONG" else (entry - exit_p) / entry * 100
            elif trail_result == "SL_LOSS":
                trade["trail_pnl_pct"] = (sl - entry) / entry * 100 if direction == "LONG" else (entry - sl) / entry * 100
            else:
                # Timeout — son kapanışta çık
                last_idx = min(i + MAX_TRADE_BARS, n - 1)
                last_c = df["Close"].iloc[last_idx]
                trade["trail_pnl_pct"] = (last_c - entry) / entry * 100 if direction == "LONG" else (entry - last_c) / entry * 100

            trades.append(trade)
            used_until = i + 5  # min 5 bar sonra yeni trade
            break  # bu squeeze'den bir kırılma yeter

    return trades


def _print_results(tdf):
    n = len(tdf)
    print("=" * 80)
    print(f"SMART BREAKOUT BACKTEST — {n} TRADE")
    print("=" * 80)

    # Yön dağılımı
    dir_counts = tdf["dir"].value_counts()
    print(f"\nYön: {dict(dir_counts)}")
    print(f"Güç dağılımı: {dict(tdf['strength'].value_counts().sort_index())}")
    print(f"Squeeze bar ort: {tdf['sq_bars'].mean():.1f} | med: {tdf['sq_bars'].median():.0f}")

    # ── 1G / 3G / 5G ─────────────────────────────────────────
    print("\n" + "─" * 60)
    print("FORWARD RETURN ANALİZİ")
    print("─" * 60)
    print(f"{'Dönem':<8} {'N':>6} {'WR%':>8} {'ORT%':>8} {'MED%':>8} {'STD%':>8}")
    print("─" * 60)

    for label in ["1g", "3g", "5g"]:
        col = f"ret_{label}"
        valid = tdf[col].dropna()
        if len(valid) == 0:
            continue
        wr = (valid > 0).mean() * 100
        avg = valid.mean()
        med = valid.median()
        std = valid.std()
        print(f"{label.upper():<8} {len(valid):>6} {wr:>7.1f}% {avg:>7.2f}% {med:>7.2f}% {std:>7.2f}%")

    # Yöne göre
    for d in ["LONG", "SHORT"]:
        sub = tdf[tdf["dir"] == d]
        if len(sub) < 10:
            continue
        print(f"\n  {d} ({len(sub)}):")
        for label in ["1g", "3g", "5g"]:
            col = f"ret_{label}"
            valid = sub[col].dropna()
            if len(valid) == 0:
                continue
            wr = (valid > 0).mean() * 100
            avg = valid.mean()
            med = valid.median()
            print(f"    {label.upper()}: WR {wr:.1f}% | ORT {avg:+.2f}% | MED {med:+.2f}%")

    # ── TP HIT ORANI ──────────────────────────────────────────
    print("\n" + "─" * 60)
    print("TP/SL HIT ANALİZİ (max 30 gün)")
    print("─" * 60)

    for tp_label, tp_col, day_col in [
        ("TP1 (1:1)", "tp1_hit", "tp1_day"),
        ("TP2 (1:2)", "tp2_hit", "tp2_day"),
        ("TP3 (1:3)", "tp3_hit", "tp3_day"),
    ]:
        hit = tdf[tp_col].sum()
        rate = hit / n * 100
        days_valid = tdf.loc[tdf[tp_col], day_col].dropna()
        avg_days = days_valid.mean() if len(days_valid) > 0 else 0
        med_days = days_valid.median() if len(days_valid) > 0 else 0
        print(
            f"  {tp_label:<12} Hit: {hit:>5}/{n} ({rate:>5.1f}%) | "
            f"Ort süre: {avg_days:>5.1f}G | Med: {med_days:>4.0f}G"
        )

    sl_before_tp1 = tdf[tdf["trail_result"] == "SL_LOSS"]
    print(f"\n  SL (TP1 öncesi): {len(sl_before_tp1)}/{n} ({len(sl_before_tp1)/n*100:.1f}%)")
    if len(sl_before_tp1) > 0:
        sl_days = sl_before_tp1["sl_day"].dropna()
        print(f"    SL varma süresi: ort {sl_days.mean():.1f}G | med {sl_days.median():.0f}G")

    # Trail sonuçları
    print("\n  Trail sonuç dağılımı:")
    for result, count in tdf["trail_result"].value_counts().items():
        pct = count / n * 100
        sub_pnl = tdf.loc[tdf["trail_result"] == result, "trail_pnl_pct"]
        avg_pnl = sub_pnl.mean()
        print(f"    {result:<12}: {count:>5} ({pct:>5.1f}%) | Ort PnL: {avg_pnl:+.2f}%")

    # Genel trail PnL
    avg_pnl = tdf["trail_pnl_pct"].mean()
    med_pnl = tdf["trail_pnl_pct"].median()
    print(f"\n  Genel Trail PnL: ORT {avg_pnl:+.2f}% | MED {med_pnl:+.2f}%")

    # ── YÖNE GÖRE TP ──────────────────────────────────────────
    print("\n" + "─" * 60)
    print("YÖNE GÖRE TP HIT")
    print("─" * 60)
    for d in ["LONG", "SHORT"]:
        sub = tdf[tdf["dir"] == d]
        if len(sub) < 10:
            continue
        ns = len(sub)
        print(f"\n  {d} ({ns}):")
        for tp_label, tp_col, day_col in [
            ("TP1", "tp1_hit", "tp1_day"),
            ("TP2", "tp2_hit", "tp2_day"),
            ("TP3", "tp3_hit", "tp3_day"),
        ]:
            hit = sub[tp_col].sum()
            rate = hit / ns * 100
            days_v = sub.loc[sub[tp_col], day_col].dropna()
            avg_d = days_v.mean() if len(days_v) > 0 else 0
            print(f"    {tp_label}: {hit}/{ns} ({rate:.1f}%) | ort {avg_d:.1f}G")

    # ── GÜCE GÖRE ─────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("SİNYAL GÜCÜNE GÖRE")
    print("─" * 60)
    for s in sorted(tdf["strength"].unique()):
        sub = tdf[tdf["strength"] == s]
        if len(sub) < 5:
            continue
        ns = len(sub)
        ret5 = sub["ret_5g"].dropna()
        wr5 = (ret5 > 0).mean() * 100 if len(ret5) > 0 else 0
        tp1r = sub["tp1_hit"].mean() * 100
        tp3r = sub["tp3_hit"].mean() * 100
        print(
            f"  Güç {s}/4 ({ns:>4}): 5G WR {wr5:>5.1f}% | "
            f"TP1 {tp1r:>5.1f}% | TP3 {tp3r:>5.1f}%"
        )

    # ── SQUEEZE SÜRESİNE GÖRE ────────────────────────────────
    print("\n" + "─" * 60)
    print("SQUEEZE SÜRESİNE GÖRE")
    print("─" * 60)
    bins = [(3, 5), (6, 10), (11, 20), (21, 50), (51, 999)]
    for lo, hi in bins:
        sub = tdf[(tdf["sq_bars"] >= lo) & (tdf["sq_bars"] <= hi)]
        if len(sub) < 5:
            continue
        ns = len(sub)
        ret5 = sub["ret_5g"].dropna()
        wr5 = (ret5 > 0).mean() * 100 if len(ret5) > 0 else 0
        tp1r = sub["tp1_hit"].mean() * 100
        tp3r = sub["tp3_hit"].mean() * 100
        label = f"{lo}-{hi}" if hi < 999 else f"{lo}+"
        print(
            f"  SQ {label:>6} ({ns:>4}): 5G WR {wr5:>5.1f}% | "
            f"TP1 {tp1r:>5.1f}% | TP3 {tp3r:>5.1f}%"
        )

    # ── RISK% GRUPLARI ────────────────────────────────────────
    print("\n" + "─" * 60)
    print("RİSK% GRUPLARINA GÖRE (Entry→SL mesafesi)")
    print("─" * 60)
    risk_bins = [(0, 5), (5, 10), (10, 20), (20, 50)]
    for lo, hi in risk_bins:
        sub = tdf[(tdf["risk_pct"] >= lo) & (tdf["risk_pct"] < hi)]
        if len(sub) < 5:
            continue
        ns = len(sub)
        ret5 = sub["ret_5g"].dropna()
        wr5 = (ret5 > 0).mean() * 100 if len(ret5) > 0 else 0
        tp1r = sub["tp1_hit"].mean() * 100
        avg_pnl = sub["trail_pnl_pct"].mean()
        print(
            f"  Risk {lo}-{hi}% ({ns:>4}): 5G WR {wr5:>5.1f}% | "
            f"TP1 {tp1r:>5.1f}% | Trail PnL {avg_pnl:+.2f}%"
        )


def _save_html_report(tdf):
    os.makedirs("output", exist_ok=True)
    csv_path = "output/backtest_breakout.csv"
    tdf.to_csv(csv_path, index=False)
    print(f"\n📄 CSV: {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smart Breakout Backtest")
    parser.add_argument("--no-vol", action="store_true", help="Hacim filtresini kapat")
    parser.add_argument("--htf", action="store_true", help="HTF EMA filtresi aç")
    parser.add_argument("--days", type=int, default=730, help="Backtest süresi (gün)")
    args = parser.parse_args()

    backtest_all(
        use_vol=not args.no_vol,
        use_htf=args.htf,
        lookback_days=args.days,
    )
