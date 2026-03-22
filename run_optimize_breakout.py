#!/usr/bin/env python3
"""
Smart Breakout — Parametre Optimizasyonu
Veriyi bir kez indir, parametre grid sweep yap.
"""

import os
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

SYMBOLS_FILE = Path(__file__).parent / "tools" / "bist_symbols.txt"
BB_LENGTH = 20
BB_MULT = 2.0
HTF_EMA_LENGTH = 50
TP_RATIOS = [1.0, 2.0, 3.0]
MAX_TRADE_BARS = 30


def load_symbols():
    return [l.strip() + ".IS" for l in SYMBOLS_FILE.read_text().splitlines() if l.strip()]


def download_all(lookback_days=730):
    symbols = load_symbols()
    print(f"📡 {len(symbols)} sembol indiriliyor...", flush=True)
    end = datetime.now()
    start = end - timedelta(days=lookback_days + 60)
    cache = {}
    for i, sym in enumerate(symbols):
        if (i + 1) % 100 == 0:
            print(f"   ... {i+1}/{len(symbols)}")
        try:
            df = yf.download(sym, start=start, end=end, progress=False, timeout=10)
            if df is None or len(df) < 80:
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            cache[sym.replace(".IS", "")] = df
        except Exception:
            continue
    print(f"   {len(cache)} sembol indirildi\n")
    return cache


def calc_base(df):
    """Parametre-bağımsız hesaplamalar (bir kez yapılır)."""
    c, h, l, o, v = df["Close"], df["High"], df["Low"], df["Open"], df["Volume"]
    sma = c.rolling(BB_LENGTH).mean()
    std = c.rolling(BB_LENGTH).std()
    bb_upper = sma + BB_MULT * std
    bb_lower = sma - BB_MULT * std
    df["bb_width"] = (bb_upper - bb_lower) / sma
    df["bb_width_sma"] = df["bb_width"].rolling(BB_LENGTH).mean()

    for atr_len in [10, 14]:
        tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
        df[f"atr_{atr_len}"] = tr.rolling(atr_len).mean()
    for sma_len in [20, 30]:
        df[f"atr_sma_{sma_len}"] = df["atr_10"].rolling(sma_len).mean()
        df[f"atr14_sma_{sma_len}"] = df["atr_14"].rolling(sma_len).mean()

    for vl in [20]:
        df[f"vol_sma_{vl}"] = v.rolling(vl).mean()

    df["body"] = (c - o).abs()
    df["htf_ema"] = c.ewm(span=HTF_EMA_LENGTH, adjust=False).mean()
    return df


def sweep_single(df, ticker, params):
    """Tek hisse, tek parametre seti ile backtest."""
    bb_thr = params["bb_thr"]
    atr_ratio = params["atr_ratio"]
    min_sq = params["min_sq"]
    max_sq = params["max_sq"]
    impulse_mul = params["impulse_mul"]
    vol_mul = params["vol_mul"]
    sl_atr_mul = params["sl_atr_mul"]
    dir_filter = params["dir_filter"]  # "BOTH", "LONG"
    atr_col = "atr_10"
    atr_sma_col = "atr_sma_20"

    n = len(df)
    # Squeeze
    sq_bb = df["bb_width"].values < df["bb_width_sma"].values * bb_thr
    sq_atr = df[atr_col].values < df[atr_sma_col].values * atr_ratio
    squeeze = sq_bb & sq_atr

    # Squeeze dönemleri
    squeezes = []
    in_sq = False
    sq_start = 0
    for i in range(n):
        if np.isnan(df["bb_width"].iloc[i]) or np.isnan(df[atr_sma_col].iloc[i]):
            if in_sq:
                in_sq = False
            continue
        if squeeze[i]:
            if not in_sq:
                sq_start = i
                in_sq = True
            else:
                # Max squeeze kontrolü
                if (i - sq_start + 1) > max_sq:
                    sq_len = max_sq
                    if sq_len >= min_sq:
                        squeezes.append({"start": sq_start, "end": sq_start + max_sq - 1, "length": sq_len})
                    sq_start = i  # yeni squeeze başlat
        else:
            if in_sq:
                sq_len = i - sq_start
                if sq_len >= min_sq:
                    squeezes.append({"start": sq_start, "end": i - 1, "length": min(sq_len, max_sq)})
                in_sq = False

    if not squeezes:
        return []

    close = df["Close"].values
    high = df["High"].values
    low = df["Low"].values
    opn = df["Open"].values
    atr = df[atr_col].values
    vol = df["Volume"].values
    vol_sma = df["vol_sma_20"].values
    body = df["body"].values
    htf = df["htf_ema"].values

    trades = []
    used_until = 0

    for sq in squeezes:
        sq_s, sq_e, sq_len = sq["start"], sq["end"], sq["length"]
        if sq_e < used_until:
            continue

        box_hi = high[sq_s:sq_e + 1].max()
        box_lo = low[sq_s:sq_e + 1].min()
        atr_e = atr[sq_e]
        if atr_e <= 0 or np.isnan(atr_e):
            continue

        # Aralık sınırlama
        rng = box_hi - box_lo
        if rng > 6.0 * atr_e:
            ctr = (box_hi + box_lo) / 2.0
            box_hi = ctr + 3.0 * atr_e
            box_lo = ctr - 3.0 * atr_e

        for i in range(sq_e + 1, min(sq_e + 31, n)):
            ci, oi = close[i], opn[i]
            bi, ai = body[i], atr[i]
            vi, vsi = vol[i], vol_sma[i]

            if ai <= 0 or np.isnan(ai):
                continue

            imp_ok = bi > ai * impulse_mul
            vol_ok = vsi > 0 and vi > vsi * vol_mul

            direction = None
            if ci > box_hi and ci > oi and imp_ok and vol_ok:
                direction = "LONG"
            elif ci < box_lo and ci < oi and imp_ok and vol_ok:
                direction = "SHORT"

            if direction is None:
                continue
            if dir_filter == "LONG" and direction != "LONG":
                continue

            entry = ci
            if direction == "LONG":
                sl = box_lo - ai * sl_atr_mul
                risk = abs(entry - sl)
                tp1 = entry + risk * 1.0
                tp2 = entry + risk * 2.0
                tp3 = entry + risk * 3.0
            else:
                sl = box_hi + ai * sl_atr_mul
                risk = abs(sl - entry)
                tp1 = entry - risk * 1.0
                tp2 = entry - risk * 2.0
                tp3 = entry - risk * 3.0

            if risk <= 0:
                break

            # Strength
            strength = 1  # impulse always true here
            if vi > vsi * vol_mul:
                strength += 1
            if sq_len >= 6:
                strength += 1

            # Forward returns
            ret = {}
            for days, lbl in [(1, "1g"), (3, "3g"), (5, "5g")]:
                if i + days < n:
                    fc = close[i + days]
                    if direction == "LONG":
                        ret[lbl] = (fc - entry) / entry * 100
                    else:
                        ret[lbl] = (entry - fc) / entry * 100
                else:
                    ret[lbl] = np.nan

            # TP/SL tracking
            tp1_hit = tp2_hit = tp3_hit = False
            tp1_day = tp2_day = tp3_day = np.nan
            trail_level = sl
            trail_result = "TIMEOUT"

            for j in range(1, min(MAX_TRADE_BARS + 1, n - i)):
                idx = i + j
                hj, lj = high[idx], low[idx]

                if direction == "LONG":
                    if lj <= trail_level:
                        trail_result = "SL_LOSS" if not tp1_hit else "SL_TRAIL"
                        break
                    if not tp1_hit and hj >= tp1:
                        tp1_hit = True; tp1_day = j; trail_level = entry
                    if not tp2_hit and hj >= tp2:
                        tp2_hit = True; tp2_day = j; trail_level = tp1
                    if not tp3_hit and hj >= tp3:
                        tp3_hit = True; tp3_day = j; trail_level = tp2
                        trail_result = "TP3_WIN"; break
                else:
                    if hj >= trail_level:
                        trail_result = "SL_LOSS" if not tp1_hit else "SL_TRAIL"
                        break
                    if not tp1_hit and lj <= tp1:
                        tp1_hit = True; tp1_day = j; trail_level = entry
                    if not tp2_hit and lj <= tp2:
                        tp2_hit = True; tp2_day = j; trail_level = tp1
                    if not tp3_hit and lj <= tp3:
                        tp3_hit = True; tp3_day = j; trail_level = tp2
                        trail_result = "TP3_WIN"; break

            # Trail PnL
            if trail_result == "TP3_WIN":
                pnl = (tp3 - entry) / entry * 100 if direction == "LONG" else (entry - tp3) / entry * 100
            elif trail_result == "SL_TRAIL":
                exit_p = tp1 if tp2_hit else (entry if tp1_hit else sl)
                pnl = (exit_p - entry) / entry * 100 if direction == "LONG" else (entry - exit_p) / entry * 100
            elif trail_result == "SL_LOSS":
                pnl = (sl - entry) / entry * 100 if direction == "LONG" else (entry - sl) / entry * 100
            else:
                last_idx = min(i + MAX_TRADE_BARS, n - 1)
                pnl = (close[last_idx] - entry) / entry * 100 if direction == "LONG" else (entry - close[last_idx]) / entry * 100

            trades.append({
                "ticker": ticker, "dir": direction, "entry": entry,
                "risk_pct": risk / entry * 100, "strength": strength, "sq_bars": sq_len,
                "ret_1g": ret.get("1g", np.nan),
                "ret_3g": ret.get("3g", np.nan),
                "ret_5g": ret.get("5g", np.nan),
                "tp1_hit": tp1_hit, "tp2_hit": tp2_hit, "tp3_hit": tp3_hit,
                "tp1_day": tp1_day, "tp2_day": tp2_day, "tp3_day": tp3_day,
                "trail_result": trail_result, "trail_pnl_pct": pnl,
            })
            used_until = i + 5
            break

    return trades


def evaluate(trades):
    """Trade listesinden skor hesapla."""
    if len(trades) < 30:
        return None
    tdf = pd.DataFrame(trades)
    n = len(tdf)

    ret5 = tdf["ret_5g"].dropna()
    wr5 = (ret5 > 0).mean() * 100 if len(ret5) > 0 else 50
    ret3 = tdf["ret_3g"].dropna()
    wr3 = (ret3 > 0).mean() * 100 if len(ret3) > 0 else 50
    ret1 = tdf["ret_1g"].dropna()
    wr1 = (ret1 > 0).mean() * 100 if len(ret1) > 0 else 50

    tp1_rate = tdf["tp1_hit"].mean() * 100
    tp2_rate = tdf["tp2_hit"].mean() * 100
    tp3_rate = tdf["tp3_hit"].mean() * 100

    avg_pnl = tdf["trail_pnl_pct"].mean()
    med_pnl = tdf["trail_pnl_pct"].median()

    sl_loss_rate = (tdf["trail_result"] == "SL_LOSS").mean() * 100

    # Composite score: WR + TP hit + PnL
    score = (
        wr5 * 0.25
        + wr3 * 0.15
        + wr1 * 0.10
        + tp1_rate * 0.20
        + avg_pnl * 0.15
        + med_pnl * 0.10
        - sl_loss_rate * 0.05
    )

    return {
        "n": n, "wr1": wr1, "wr3": wr3, "wr5": wr5,
        "avg1": ret1.mean() if len(ret1) > 0 else 0,
        "med1": ret1.median() if len(ret1) > 0 else 0,
        "avg5": ret5.mean() if len(ret5) > 0 else 0,
        "med5": ret5.median() if len(ret5) > 0 else 0,
        "tp1": tp1_rate, "tp2": tp2_rate, "tp3": tp3_rate,
        "tp1_day": tdf.loc[tdf["tp1_hit"], "tp1_day"].median() if tdf["tp1_hit"].sum() > 0 else 0,
        "sl_loss": sl_loss_rate,
        "avg_pnl": avg_pnl, "med_pnl": med_pnl,
        "score": score,
    }


def run_optimization(cache):
    # Parametre grid — daraltılmış (en etkili parametreler)
    grid = {
        "bb_thr":      [0.8, 1.0, 1.2],       # squeeze sıkılığı (en kritik)
        "atr_ratio":   [1.0, 1.3, 1.6],        # ATR squeeze (en kritik)
        "min_sq":      [3, 5],
        "max_sq":      [10, 20, 40],            # kutu genişliği sınırı (en kritik)
        "impulse_mul": [0.35, 0.6],
        "vol_mul":     [1.5],                   # sabit
        "sl_atr_mul":  [0.3, 0.5],
        "dir_filter":  ["LONG", "BOTH"],
    }

    keys = list(grid.keys())
    combos = list(product(*[grid[k] for k in keys]))
    print(f"🔬 {len(combos)} parametre kombinasyonu test ediliyor...\n")

    # Her hisse için base hesapla
    prepped = {}
    for ticker, df in cache.items():
        df_c = df.copy()
        try:
            df_c = calc_base(df_c)
            df_c.reset_index(drop=True, inplace=True)
            prepped[ticker] = df_c
        except Exception:
            continue

    results = []
    for ci, combo in enumerate(combos):
        params = dict(zip(keys, combo))
        if (ci + 1) % 50 == 0:
            print(f"   ... {ci+1}/{len(combos)}")

        all_trades = []
        for ticker, df in prepped.items():
            trades = sweep_single(df, ticker, params)
            all_trades.extend(trades)

        ev = evaluate(all_trades)
        if ev is None:
            continue

        ev.update(params)
        results.append(ev)

    if not results:
        print("Sonuç yok.")
        return

    rdf = pd.DataFrame(results)
    rdf.sort_values("score", ascending=False, inplace=True)

    # Top 20
    print("\n" + "=" * 120)
    print("TOP 20 PARAMETRE KOMBİNASYONU")
    print("=" * 120)
    print(
        f"{'#':>3} {'N':>5} {'WR1':>6} {'WR3':>6} {'WR5':>6} "
        f"{'TP1%':>6} {'TP2%':>6} {'TP3%':>6} {'TP1d':>5} "
        f"{'AvgPnL':>7} {'MedPnL':>7} {'SL%':>5} {'SKOR':>7} "
        f"{'bb':>5} {'atr':>5} {'msq':>4} {'xsq':>4} {'imp':>5} {'vol':>5} {'sl':>4} {'dir':>5}"
    )
    print("─" * 120)
    for idx, row in rdf.head(20).iterrows():
        rank = rdf.index.get_loc(idx) + 1
        print(
            f"{rank:>3} {row['n']:>5.0f} {row['wr1']:>5.1f}% {row['wr3']:>5.1f}% {row['wr5']:>5.1f}% "
            f"{row['tp1']:>5.1f}% {row['tp2']:>5.1f}% {row['tp3']:>5.1f}% {row['tp1_day']:>5.0f} "
            f"{row['avg_pnl']:>+6.2f}% {row['med_pnl']:>+6.2f}% {row['sl_loss']:>4.1f}% {row['score']:>7.2f} "
            f"{row['bb_thr']:>5.2f} {row['atr_ratio']:>5.2f} {row['min_sq']:>4.0f} {row['max_sq']:>4.0f} "
            f"{row['impulse_mul']:>5.2f} {row['vol_mul']:>5.2f} {row['sl_atr_mul']:>4.2f} {row['dir_filter']:>5}"
        )

    # En iyi parametrelerle detaylı rapor
    best = rdf.iloc[0]
    print("\n" + "=" * 80)
    print("EN İYİ PARAMETRE SETİ — DETAYLI RAPOR")
    print("=" * 80)
    best_params = {k: best[k] for k in keys}
    print(f"Parametreler: {best_params}")

    all_trades = []
    for ticker, df in prepped.items():
        all_trades.extend(sweep_single(df, ticker, best_params))

    tdf = pd.DataFrame(all_trades)
    _detailed_report(tdf)

    # CSV kaydet
    os.makedirs("output", exist_ok=True)
    rdf.to_csv("output/breakout_optimize.csv", index=False)
    tdf.to_csv("output/breakout_best_trades.csv", index=False)
    print(f"\n📄 output/breakout_optimize.csv ({len(rdf)} combo)")
    print(f"📄 output/breakout_best_trades.csv ({len(tdf)} trade)")

    return rdf, tdf


def _detailed_report(tdf):
    n = len(tdf)
    print(f"\nToplam: {n} trade")
    print(f"Yön: {dict(tdf['dir'].value_counts())}")
    print(f"Squeeze ort: {tdf['sq_bars'].mean():.1f} | med: {tdf['sq_bars'].median():.0f}")
    print(f"Risk% ort: {tdf['risk_pct'].mean():.1f}% | med: {tdf['risk_pct'].median():.1f}%")

    print(f"\n{'Dönem':<8} {'N':>6} {'WR%':>8} {'ORT%':>8} {'MED%':>8}")
    print("─" * 45)
    for lbl in ["1g", "3g", "5g"]:
        col = f"ret_{lbl}"
        v = tdf[col].dropna()
        if len(v) == 0:
            continue
        print(f"{lbl.upper():<8} {len(v):>6} {(v>0).mean()*100:>7.1f}% {v.mean():>+7.2f}% {v.median():>+7.2f}%")

    print(f"\n{'TP':<8} {'Hit':>8} {'Oran':>8} {'Med Süre':>10}")
    print("─" * 40)
    for tp, tc, dc in [("TP1", "tp1_hit", "tp1_day"), ("TP2", "tp2_hit", "tp2_day"), ("TP3", "tp3_hit", "tp3_day")]:
        hit = tdf[tc].sum()
        rate = hit / n * 100
        md = tdf.loc[tdf[tc], dc].median() if hit > 0 else 0
        print(f"{tp:<8} {hit:>6}/{n} {rate:>7.1f}% {md:>9.0f}G")

    sl_n = (tdf["trail_result"] == "SL_LOSS").sum()
    print(f"\nSL Loss: {sl_n}/{n} ({sl_n/n*100:.1f}%)")
    print(f"\nTrail sonuçları:")
    for res, cnt in tdf["trail_result"].value_counts().items():
        sub = tdf.loc[tdf["trail_result"] == res, "trail_pnl_pct"]
        print(f"  {res:<12}: {cnt:>5} ({cnt/n*100:>5.1f}%) | PnL: {sub.mean():+.2f}%")

    print(f"\nGenel Trail PnL: ORT {tdf['trail_pnl_pct'].mean():+.2f}% | MED {tdf['trail_pnl_pct'].median():+.2f}%")

    # Yöne göre
    for d in ["LONG", "SHORT"]:
        sub = tdf[tdf["dir"] == d]
        if len(sub) < 10:
            continue
        ns = len(sub)
        r5 = sub["ret_5g"].dropna()
        print(f"\n  {d} ({ns}): 5G WR {(r5>0).mean()*100:.1f}% | TP1 {sub['tp1_hit'].mean()*100:.1f}% | TP3 {sub['tp3_hit'].mean()*100:.1f}%")


if __name__ == "__main__":
    cache = download_all(730)
    run_optimization(cache)
