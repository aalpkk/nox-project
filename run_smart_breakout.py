#!/usr/bin/env python3
"""
Smart Breakout Targets — Tüm BIST Tarayıcı
Squeeze → Kutu → Kırılma → TP/SL seviyeleri
"""

import os
import sys
import warnings
import argparse
import base64
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests as req
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()

warnings.filterwarnings("ignore")

# ── Parametreler (optimized — 2Y backtest 2548 trade) ─────────
BB_LENGTH = 20
BB_MULT = 2.0
BB_WIDTH_THRESH = 0.80          # was 1.60 — sıkı squeeze
ATR_LENGTH = 10
ATR_SMA_LENGTH = 20
ATR_SQUEEZE_RATIO = 1.00        # was 2.00 — sıkı ATR filtre
MIN_SQUEEZE_BARS = 5            # was 3
MAX_SQUEEZE_BARS = 40           # yeni — kutu genişliğini sınırla
IMPULSE_ATR_MULT = 0.35
VOL_SMA_LENGTH = 20
VOL_MULT = 1.5
MAX_RANGE_ATR_MULT = 6.0
HTF_EMA_LENGTH = 50
ATR_SL_MULT = 0.3              # was 0.5 — daha sıkı SL
TP_RATIOS = [1.0, 2.0, 3.0]
LONG_ONLY = True                # SHORT sinyaller kaldırıldı (TP1 %1.3)

# ── Sembol listesi ────────────────────────────────────────────
SYMBOLS_FILE = Path(__file__).parent / "tools" / "bist_symbols.txt"


def load_symbols():
    if SYMBOLS_FILE.exists():
        symbols = [
            line.strip()
            for line in SYMBOLS_FILE.read_text().splitlines()
            if line.strip()
        ]
        return [s + ".IS" for s in symbols]
    print("HATA: tools/bist_symbols.txt bulunamadı")
    sys.exit(1)


# ── Teknik hesaplamalar ───────────────────────────────────────
def calc_indicators(df: pd.DataFrame) -> pd.DataFrame:
    c, h, l, o, v = df["Close"], df["High"], df["Low"], df["Open"], df["Volume"]

    # Bollinger Band width
    sma = c.rolling(BB_LENGTH).mean()
    std = c.rolling(BB_LENGTH).std()
    bb_upper = sma + BB_MULT * std
    bb_lower = sma - BB_MULT * std
    df["bb_width"] = (bb_upper - bb_lower) / sma
    df["bb_width_sma"] = df["bb_width"].rolling(BB_LENGTH).mean()

    # ATR
    tr = pd.concat(
        [h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1
    ).max(axis=1)
    df["atr"] = tr.rolling(ATR_LENGTH).mean()
    df["atr_sma"] = df["atr"].rolling(ATR_SMA_LENGTH).mean()

    # Volume SMA
    df["vol_sma"] = v.rolling(VOL_SMA_LENGTH).mean()

    # Body
    df["body"] = (c - o).abs()

    # HTF proxy — EMA50 on daily as weekly trend proxy
    df["htf_ema"] = c.ewm(span=HTF_EMA_LENGTH, adjust=False).mean()

    # Squeeze koşulları
    df["sq_bb"] = df["bb_width"] < df["bb_width_sma"] * BB_WIDTH_THRESH
    df["sq_atr"] = df["atr"] < df["atr_sma"] * ATR_SQUEEZE_RATIO
    df["squeeze"] = df["sq_bb"] & df["sq_atr"]

    # ── ML feature'lar için ek indikatörler ──────────────────
    df["ema20"] = c.ewm(span=20, adjust=False).mean()

    # RSI 14
    delta = c.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / 14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    df["rsi"] = 100 - (100 / (1 + rs))

    # ADX, +DI, -DI
    tr_raw = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    plus_dm = h.diff().clip(lower=0)
    minus_dm = (-l.diff()).clip(lower=0)
    both_pos = (plus_dm > 0) & (minus_dm > 0)
    plus_dm_c = plus_dm.copy()
    minus_dm_c = minus_dm.copy()
    plus_dm_c[both_pos & (plus_dm <= minus_dm)] = 0
    minus_dm_c[both_pos & (minus_dm < plus_dm)] = 0
    atr14 = tr_raw.ewm(alpha=1 / 14, adjust=False).mean()
    df["plus_di"] = 100 * plus_dm_c.ewm(alpha=1 / 14, adjust=False).mean() / (atr14 + 1e-10)
    df["minus_di"] = 100 * minus_dm_c.ewm(alpha=1 / 14, adjust=False).mean() / (atr14 + 1e-10)
    dx = 100 * (df["plus_di"] - df["minus_di"]).abs() / (df["plus_di"] + df["minus_di"] + 1e-10)
    df["adx"] = dx.ewm(alpha=1 / 14, adjust=False).mean()
    df["di_spread"] = df["plus_di"] - df["minus_di"]

    # Returns & slopes
    df["ret_5d"] = c.pct_change(5) * 100
    df["ret_10d"] = c.pct_change(10) * 100
    df["ret_20d"] = c.pct_change(20) * 100
    df["ema50_slope"] = (df["htf_ema"] - df["htf_ema"].shift(10)) / (df["htf_ema"].shift(10) + 1e-10) * 100
    df["ema20_slope"] = (df["ema20"] - df["ema20"].shift(5)) / (df["ema20"].shift(5) + 1e-10) * 100
    df["dist_ema50"] = (c - df["htf_ema"]) / (df["htf_ema"] + 1e-10) * 100
    df["low_10d"] = l.rolling(10).min()
    df["runup_10d"] = (c - df["low_10d"]) / (df["low_10d"] + 1e-10) * 100
    df["rs_10"] = c / c.shift(10)
    df["rs_60"] = c / c.shift(60)

    return df


# ── Squeeze → Kutu → Kırılma state machine ───────────────────
def scan_single(df: pd.DataFrame, use_vol: bool, use_htf: bool):
    """Tek hisse için squeeze/box/breakout taraması.
    Returns dict with status and levels, or None.
    """
    if len(df) < ATR_SMA_LENGTH + BB_LENGTH + 10:
        return None

    df = calc_indicators(df).copy()
    df.reset_index(drop=True, inplace=True)
    n = len(df)

    # ── Squeeze dönemlerini bul ───────────────────────────────
    squeezes = []
    in_sq = False
    sq_start = 0
    for i in range(n):
        if pd.isna(df["squeeze"].iloc[i]):
            if in_sq:
                in_sq = False
            continue
        if df["squeeze"].iloc[i]:
            if not in_sq:
                sq_start = i
                in_sq = True
            elif (i - sq_start + 1) > MAX_SQUEEZE_BARS:
                sq_len = MAX_SQUEEZE_BARS
                if sq_len >= MIN_SQUEEZE_BARS:
                    squeezes.append(
                        {"start": sq_start, "end": sq_start + MAX_SQUEEZE_BARS - 1, "length": sq_len}
                    )
                sq_start = i
        else:
            if in_sq:
                sq_len = i - sq_start
                if sq_len >= MIN_SQUEEZE_BARS:
                    squeezes.append(
                        {"start": sq_start, "end": i - 1, "length": min(sq_len, MAX_SQUEEZE_BARS)}
                    )
                in_sq = False

    # Hâlâ squeeze içindeyse
    if in_sq:
        sq_len = min(n - sq_start, MAX_SQUEEZE_BARS)
        if sq_len >= MIN_SQUEEZE_BARS:
            return {
                "status": "SQUEEZE",
                "squeeze_bars": sq_len,
                "squeeze_start": df.index[sq_start],
                "close": df["Close"].iloc[-1],
            }
        return None

    if not squeezes:
        return None

    # ── En son geçerli squeeze'den kutu oluştur ──────────────
    last_sq = squeezes[-1]
    sq_s, sq_e = last_sq["start"], last_sq["end"]
    sq_high = df["High"].iloc[sq_s : sq_e + 1].max()
    sq_low = df["Low"].iloc[sq_s : sq_e + 1].min()
    sq_range = sq_high - sq_low
    atr_at_end = df["atr"].iloc[sq_e]

    # Aralık sınırlama
    if atr_at_end > 0 and sq_range > MAX_RANGE_ATR_MULT * atr_at_end:
        center = (sq_high + sq_low) / 2.0
        sq_high = center + 3.0 * atr_at_end
        sq_low = center - 3.0 * atr_at_end
        sq_range = sq_high - sq_low

    box_top = sq_high
    box_bot = sq_low
    box_mid = (box_top + box_bot) / 2.0

    # ── Kırılma kontrolü (squeeze'den sonraki barlar) ────────
    breakout = None
    for i in range(sq_e + 1, n):
        c_i = df["Close"].iloc[i]
        o_i = df["Open"].iloc[i]
        h_i = df["High"].iloc[i]
        l_i = df["Low"].iloc[i]
        body_i = df["body"].iloc[i]
        atr_i = df["atr"].iloc[i]
        vol_i = df["Volume"].iloc[i]
        vol_sma_i = df["vol_sma"].iloc[i]
        htf_ema_i = df["htf_ema"].iloc[i]

        if atr_i <= 0 or pd.isna(atr_i):
            continue

        impulse_ok = body_i > atr_i * IMPULSE_ATR_MULT
        vol_ok = (not use_vol) or (
            vol_sma_i > 0 and vol_i > vol_sma_i * VOL_MULT
        )

        # LONG kırılma
        if c_i > box_top and c_i > o_i and impulse_ok and vol_ok:
            htf_ok = (not use_htf) or (c_i > htf_ema_i)
            if htf_ok:
                breakout = {"dir": "LONG", "bar": i, "atr": atr_i}
                break

        # SHORT kırılma (LONG_ONLY modda devre dışı)
        if not LONG_ONLY and c_i < box_bot and c_i < o_i and impulse_ok and vol_ok:
            htf_ok = (not use_htf) or (c_i < htf_ema_i)
            if htf_ok:
                breakout = {"dir": "SHORT", "bar": i, "atr": atr_i}
                break

    # ── Kutu var ama kırılma yok ──────────────────────────────
    if breakout is None:
        return {
            "status": "BOX",
            "box_top": box_top,
            "box_bot": box_bot,
            "box_mid": box_mid,
            "squeeze_bars": last_sq["length"],
            "bars_since": n - 1 - sq_e,
            "close": df["Close"].iloc[-1],
            "atr": df["atr"].iloc[-1],
        }

    # ── Kırılma bulundu → seviyeler ──────────────────────────
    bi = breakout["bar"]
    entry = df["Close"].iloc[bi]
    atr_bo = breakout["atr"]
    direction = breakout["dir"]

    if direction == "LONG":
        sl = box_bot - atr_bo * ATR_SL_MULT
        risk = abs(entry - sl)
        tp1 = entry + risk * TP_RATIOS[0]
        tp2 = entry + risk * TP_RATIOS[1]
        tp3 = entry + risk * TP_RATIOS[2]
    else:
        sl = box_top + atr_bo * ATR_SL_MULT
        risk = abs(sl - entry)
        tp1 = entry - risk * TP_RATIOS[0]
        tp2 = entry - risk * TP_RATIOS[1]
        tp3 = entry - risk * TP_RATIOS[2]

    # Sinyal gücü
    strength = 0
    body_bo = df["body"].iloc[bi]
    if body_bo > atr_bo * IMPULSE_ATR_MULT:
        strength += 1
    vol_bo = df["Volume"].iloc[bi]
    vol_sma_bo = df["vol_sma"].iloc[bi]
    if use_vol and vol_sma_bo > 0 and vol_bo > vol_sma_bo * VOL_MULT:
        strength += 1
    htf_ema_bo = df["htf_ema"].iloc[bi]
    if use_htf:
        if (direction == "LONG" and entry > htf_ema_bo) or (
            direction == "SHORT" and entry < htf_ema_bo
        ):
            strength += 1
    if last_sq["length"] >= 6:
        strength += 1

    # Trade durumu — kırılma barından bugüne
    trade_status = "OPEN"
    trail_level = sl
    current_close = df["Close"].iloc[-1]
    tp1_hit = tp2_hit = tp3_hit = False

    for j in range(bi + 1, n):
        cj = df["Close"].iloc[j]
        hj = df["High"].iloc[j]
        lj = df["Low"].iloc[j]

        if direction == "LONG":
            # SL kontrolü
            if lj <= trail_level:
                trade_status = "LOSS" if not tp1_hit else "WIN_TRAIL"
                break
            if not tp1_hit and hj >= tp1:
                tp1_hit = True
                trail_level = entry
            if not tp2_hit and hj >= tp2:
                tp2_hit = True
                trail_level = tp1
            if not tp3_hit and hj >= tp3:
                tp3_hit = True
                trail_level = tp2
        else:
            if hj >= trail_level:
                trade_status = "LOSS" if not tp1_hit else "WIN_TRAIL"
                break
            if not tp1_hit and lj <= tp1:
                tp1_hit = True
                trail_level = entry
            if not tp2_hit and lj <= tp2:
                tp2_hit = True
                trail_level = tp1
            if not tp3_hit and lj <= tp3:
                tp3_hit = True
                trail_level = tp2

    bars_ago = n - 1 - bi
    bo_date = df.index[bi] if isinstance(df.index, pd.DatetimeIndex) else None

    # ── ML Feature extraction (breakout barından) ────────────
    sq_len = last_sq["length"]
    sq_s, sq_e = last_sq["start"], last_sq["end"]
    box_range = box_top - box_bot
    box_mid = (box_top + box_bot) / 2.0

    def _sv(col):
        v = df[col].iloc[bi]
        return float(v) if pd.notna(v) else 0.0

    ml_features = {
        "squeeze_bars": sq_len,
        "box_range_atr": box_range / (atr_bo + 1e-10),
        "box_range_pct": box_range / (entry + 1e-10) * 100,
        "box_position": (entry - box_bot) / (box_range + 1e-10),
        "close_vs_box_top_pct": (entry - box_top) / (box_top + 1e-10) * 100,
        "days_since_sq_end": bi - sq_e,
        "box_tightness": float(df["bb_width"].iloc[sq_e] / (df["bb_width_sma"].iloc[sq_e] + 1e-10))
        if pd.notna(df["bb_width"].iloc[sq_e]) and pd.notna(df["bb_width_sma"].iloc[sq_e])
        else 0.0,
        "body_atr": float(body_bo / (atr_bo + 1e-10)),
        "close_to_high": float((entry - df["Low"].iloc[bi]) / (df["High"].iloc[bi] - df["Low"].iloc[bi] + 1e-10)),
        "gap_pct": float((df["Open"].iloc[bi] - df["Close"].iloc[bi - 1]) / (df["Close"].iloc[bi - 1] + 1e-10) * 100) if bi > 0 else 0.0,
        "vol_ratio": float(vol_bo / (vol_sma_bo + 1e-10)) if vol_sma_bo > 0 else 0.0,
        "bo_range_atr": float((df["High"].iloc[bi] - df["Low"].iloc[bi]) / (atr_bo + 1e-10)),
        "above_ema50": 1 if entry > htf_ema_bo else 0,
        "dist_ema50": _sv("dist_ema50"),
        "ema50_slope": _sv("ema50_slope"),
        "ema20_slope": _sv("ema20_slope"),
        "ret_5d": _sv("ret_5d"),
        "ret_10d": _sv("ret_10d"),
        "ret_20d": _sv("ret_20d"),
        "rsi": _sv("rsi"),
        "adx": _sv("adx"),
        "plus_di": _sv("plus_di"),
        "minus_di": _sv("minus_di"),
        "di_spread": _sv("di_spread"),
        "rs_10": _sv("rs_10") if _sv("rs_10") != 0 else 1.0,
        "rs_60": _sv("rs_60") if _sv("rs_60") != 0 else 1.0,
        "risk_pct": risk / (entry + 1e-10) * 100,
        "entry_to_box_mid_atr": (entry - box_mid) / (atr_bo + 1e-10),
        "runup_10d": _sv("runup_10d"),
        "strength": strength,
        "xu100_above_ema": 0,  # placeholder — scanner'da doldurulacak
        "xu100_ret_20d": 0.0,
    }

    return {
        "status": "BREAKOUT",
        "dir": direction,
        "entry": round(entry, 2),
        "sl": round(sl, 2),
        "tp1": round(tp1, 2),
        "tp2": round(tp2, 2),
        "tp3": round(tp3, 2),
        "risk": round(risk, 2),
        "rr_current": round(abs(current_close - entry) / risk, 2)
        if risk > 0
        else 0,
        "strength": strength,
        "strength_label": (
            "STRONG 🟢" if strength >= 4 else "MEDIUM 🟡" if strength >= 2 else "NORMAL ⚫"
        ),
        "trade_status": trade_status,
        "tp1_hit": tp1_hit,
        "tp2_hit": tp2_hit,
        "tp3_hit": tp3_hit,
        "trail_level": round(trail_level, 2),
        "bars_ago": bars_ago,
        "bo_date": bo_date,
        "squeeze_bars": last_sq["length"],
        "close": round(current_close, 2),
        "box_top": round(box_top, 2),
        "box_bot": round(box_bot, 2),
        "ml_features": ml_features,
    }


# ── Toplu tarama ──────────────────────────────────────────────
def run_scanner(use_vol=True, use_htf=False, lookback_days=180, max_bo_age=5, notify=False):
    symbols = load_symbols()
    print(f"📡 {len(symbols)} sembol taranıyor (son {lookback_days} gün)...")
    print(f"   Vol filtre: {'ON' if use_vol else 'OFF'} | HTF filtre: {'ON' if use_htf else 'OFF'}")
    print()

    end = datetime.now()
    start = end - timedelta(days=lookback_days)

    results = {"SQUEEZE": [], "BOX": [], "BREAKOUT": []}
    errors = 0

    for i, sym in enumerate(symbols):
        ticker = sym.replace(".IS", "")
        if (i + 1) % 50 == 0:
            print(f"   ... {i+1}/{len(symbols)} tamamlandı")
        try:
            df = yf.download(sym, start=start, end=end, progress=False, timeout=10)
            if df is None or len(df) < 50:
                continue
            # yfinance multi-level column fix
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            res = scan_single(df, use_vol, use_htf)
            if res is None:
                continue

            res["ticker"] = ticker
            status = res["status"]

            if status == "BREAKOUT":
                if res["bars_ago"] <= max_bo_age and res["trade_status"] == "OPEN":
                    results["BREAKOUT"].append(res)
            elif status == "BOX":
                results["BOX"].append(res)
            elif status == "SQUEEZE":
                results["SQUEEZE"].append(res)

        except Exception:
            errors += 1
            continue

    print(f"   Tarama tamamlandı ({errors} hata)\n")

    # ── ML Scoring ──────────────────────────────────────────
    _ml_score_breakouts(results["BREAKOUT"], start, end)

    # ── Sonuçları yazdır ──────────────────────────────────────
    _print_breakouts(results["BREAKOUT"])
    _print_boxes(results["BOX"])
    _print_squeezes(results["SQUEEZE"])

    # HTML rapor
    html = _generate_html(results)
    os.makedirs("output", exist_ok=True)
    out_path = "output/smart_breakout.html"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\n📄 HTML rapor: {out_path}")

    if notify:
        _notify_telegram(results, html, out_path)

    return results


# ── ML Scoring + Gate + Bucket ────────────────────────────────
ML_FEATURE_COLS = [
    "squeeze_bars", "box_range_atr", "box_range_pct", "box_position",
    "close_vs_box_top_pct", "days_since_sq_end", "box_tightness",
    "body_atr", "close_to_high", "gap_pct", "vol_ratio", "bo_range_atr",
    "above_ema50", "dist_ema50", "ema50_slope", "ema20_slope",
    "ret_5d", "ret_10d", "ret_20d",
    "rsi", "adx", "plus_di", "minus_di", "di_spread",
    "rs_10", "rs_60",
    "risk_pct", "entry_to_box_mid_atr", "runup_10d",
    "strength",
    "xu100_above_ema", "xu100_ret_20d",
]


def _ml_score_breakouts(breakouts, start, end):
    """ML scoring → hard gate → bucket assignment."""
    if not breakouts:
        return

    model_path = Path(__file__).parent / "output" / "lgb_tp1_10g.txt"
    config_path = Path(__file__).parent / "output" / "ml_breakout_config.json"

    if not model_path.exists():
        print("   ⚠️ ML model bulunamadı (output/lgb_tp1_10g.txt) — skor atlanıyor\n")
        for bo in breakouts:
            bo["ml_prob"] = None
            bo["ml_bucket"] = "?"
            bo["ml_gate"] = "?"
        return

    import json
    try:
        import lightgbm as lgb
    except ImportError:
        print("   ⚠️ lightgbm yüklü değil — skor atlanıyor\n")
        for bo in breakouts:
            bo["ml_prob"] = None
            bo["ml_bucket"] = "?"
            bo["ml_gate"] = "?"
        return

    model = lgb.Booster(model_file=str(model_path))
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    p95 = config.get("p95", 0.35)
    p85 = config.get("p85", 0.30)
    p65 = config.get("p65", 0.25)

    # XU100 regime
    xu100_above = 0
    xu100_ret = 0.0
    try:
        xu = yf.download("XU100.IS", start=start, end=end, progress=False, timeout=10)
        if xu is not None and len(xu) > 50:
            if isinstance(xu.columns, pd.MultiIndex):
                xu.columns = xu.columns.get_level_values(0)
            xu_ema = xu["Close"].ewm(span=50, adjust=False).mean()
            xu100_above = 1 if xu["Close"].iloc[-1] > xu_ema.iloc[-1] else 0
            xu100_ret = float((xu["Close"].iloc[-1] / xu["Close"].iloc[-21] - 1) * 100) if len(xu) > 21 else 0
    except Exception:
        pass

    print(f"   ML scoring: XU100 {'▲UP' if xu100_above else '▼DOWN'} ({xu100_ret:+.1f}%)")

    for bo in breakouts:
        feat = bo.get("ml_features")
        if not feat:
            bo["ml_prob"] = None
            bo["ml_bucket"] = "?"
            bo["ml_gate"] = "?"
            continue

        # XU100 regime inject
        feat["xu100_above_ema"] = xu100_above
        feat["xu100_ret_20d"] = xu100_ret

        # Score
        X = np.array([[feat.get(f, 0) for f in ML_FEATURE_COLS]])
        prob = float(model.predict(X)[0])
        bo["ml_prob"] = prob

        # Hard gates
        gates = []
        hard_fail = False
        if feat.get("above_ema50", 1) == 0:
            gates.append("EMA50↓")
            hard_fail = True
        if feat.get("risk_pct", 0) >= 20:
            gates.append("RISK↑")
            hard_fail = True
        if xu100_above == 0:
            gates.append("XU100↓")
            # soft gate — not hard fail

        bo["ml_gate"] = ",".join(gates) if gates else "OK"

        # Bucket
        if hard_fail:
            bo["ml_bucket"] = "X"
        elif prob >= p95:
            bkt = "A+"
            if bo["ml_gate"] == "OK":
                bkt = "A+✓"
            bo["ml_bucket"] = bkt
        elif prob >= p85:
            bo["ml_bucket"] = "A"
        elif prob >= p65:
            bo["ml_bucket"] = "B"
        else:
            bo["ml_bucket"] = "C"

    # Bucket'a göre sırala (A+✓ > A+ > A > B > C > X)
    bucket_order = {"A+✓": 0, "A+": 1, "A": 2, "B": 3, "C": 4, "X": 5, "?": 6}
    breakouts.sort(key=lambda x: (bucket_order.get(x.get("ml_bucket", "?"), 9), -(x.get("ml_prob") or 0)))

    n_scored = sum(1 for b in breakouts if b.get("ml_prob") is not None)
    n_gated = sum(1 for b in breakouts if b.get("ml_bucket") == "X")
    print(f"   {n_scored} skorlandı, {n_gated} gate'lendi (EMA50↓ / RISK↑)\n")


def _print_breakouts(items):
    if not items:
        print("🚀 KIRILMA: Yok\n")
        return

    # ML bucket sıralaması zaten _ml_score_breakouts'ta yapıldı
    print(f"🚀 KIRILMA ({len(items)} adet)")
    print("─" * 120)
    print(
        f"{'Sembol':<10} {'Yön':<6} {'Güç':<12} {'Giriş':>8} {'SL':>8} "
        f"{'TP1':>8} {'TP2':>8} {'TP3':>8} {'Durum':<10} {'Gün':>4} "
        f"{'ML':>6} {'Bucket':<6} {'Gate':<10}"
    )
    print("─" * 120)
    for r in items:
        tp_marks = ""
        if r["tp1_hit"]:
            tp_marks += "①"
        if r["tp2_hit"]:
            tp_marks += "②"
        if r["tp3_hit"]:
            tp_marks += "③"
        status = r["trade_status"]
        if tp_marks:
            status += " " + tp_marks

        prob = r.get("ml_prob")
        prob_str = f"{prob:.2f}" if prob is not None else "  - "
        bucket = r.get("ml_bucket", "?")
        gate = r.get("ml_gate", "?")

        print(
            f"{r['ticker']:<10} {r['dir']:<6} {r['strength_label']:<12} "
            f"{r['entry']:>8.2f} {r['sl']:>8.2f} {r['tp1']:>8.2f} "
            f"{r['tp2']:>8.2f} {r['tp3']:>8.2f} {status:<10} {r['bars_ago']:>4} "
            f"{prob_str:>6} {bucket:<6} {gate:<10}"
        )
    print()


def _print_boxes(items):
    if not items:
        print("📦 KUTU (kırılma bekleyen): Yok\n")
        return

    # Fiyatın kutu sınırına yakınlığına göre sırala
    for r in items:
        if r["close"] > r["box_mid"]:
            r["proximity"] = (r["close"] - r["box_mid"]) / (
                r["box_top"] - r["box_mid"]
            )
        else:
            r["proximity"] = (r["box_mid"] - r["close"]) / (
                r["box_mid"] - r["box_bot"]
            )
    items.sort(key=lambda x: abs(x["proximity"]), reverse=True)

    print(f"📦 KUTU — kırılma bekleyen ({len(items)} adet)")
    print("─" * 85)
    print(
        f"{'Sembol':<10} {'Kapanış':>9} {'Kutu Üst':>9} {'Kutu Alt':>9} "
        f"{'Merkez':>9} {'Mesafe%':>8} {'SQ Bar':>7} {'Bekleme':>8}"
    )
    print("─" * 85)
    for r in items[:30]:  # top 30
        dist_top = (r["box_top"] - r["close"]) / r["close"] * 100
        dist_bot = (r["close"] - r["box_bot"]) / r["close"] * 100
        closer = f"↑{dist_top:.1f}%" if dist_top < dist_bot else f"↓{dist_bot:.1f}%"
        print(
            f"{r['ticker']:<10} {r['close']:>9.2f} {r['box_top']:>9.2f} "
            f"{r['box_bot']:>9.2f} {r['box_mid']:>9.2f} {closer:>8} "
            f"{r['squeeze_bars']:>7} {r['bars_since']:>8}"
        )
    if len(items) > 30:
        print(f"   ... ve {len(items) - 30} daha")
    print()


def _print_squeezes(items):
    if not items:
        print("🟡 AKTİF SIKIŞMA: Yok\n")
        return

    items.sort(key=lambda x: x["squeeze_bars"], reverse=True)
    print(f"🟡 AKTİF SIKIŞMA ({len(items)} adet)")
    print("─" * 45)
    print(f"{'Sembol':<10} {'Kapanış':>9} {'SQ Bar':>7} {'Durum':<15}")
    print("─" * 45)
    for r in items:
        label = "⏳ Bekleniyor" if r["squeeze_bars"] < MIN_SQUEEZE_BARS else "✅ Geçerli"
        print(
            f"{r['ticker']:<10} {r['close']:>9.2f} {r['squeeze_bars']:>7} {label:<15}"
        )
    print()


# ── HTML Rapor ────────────────────────────────────────────────
def _generate_html(results):
    now_tr = datetime.now(timezone(timedelta(hours=3)))
    date_str = now_tr.strftime("%d.%m.%Y %H:%M")

    breakouts = results["BREAKOUT"]
    boxes = results["BOX"]
    squeezes = results["SQUEEZE"]

    TV = "https://www.tradingview.com/chart/?symbol=BIST:"

    def _tk(ticker):
        return f'<a href="{TV}{ticker}" target="_blank" class="tk">{ticker}</a>'

    def _bo_rows():
        rows = []
        # Breakout'lar zaten ML bucket sıralı
        for r in breakouts:
            cls = "long" if r["dir"] == "LONG" else "short"
            tp_marks = ""
            if r["tp1_hit"]:
                tp_marks += " ①"
            if r["tp2_hit"]:
                tp_marks += " ②"
            if r["tp3_hit"]:
                tp_marks += " ③"
            s = r["strength"]
            s_cls = "strong" if s >= 4 else "medium" if s >= 2 else "normal"

            prob = r.get("ml_prob")
            bucket = r.get("ml_bucket", "?")
            gate = r.get("ml_gate", "?")
            prob_str = f"{prob:.2f}" if prob is not None else "-"
            prob_val = f"{prob:.4f}" if prob is not None else "0"

            # Bucket renk
            bkt_cls = "strong" if bucket.startswith("A+") else "medium" if bucket == "A" else "normal"
            if bucket == "X":
                bkt_cls = "short"

            gate_html = f'<span style="color:#f85149">{gate}</span>' if gate != "OK" and gate != "?" else gate

            rows.append(
                f'<tr class="{cls}">'
                f"<td><b>{_tk(r['ticker'])}</b></td>"
                f'<td class="{cls}">{r["dir"]}</td>'
                f'<td class="{s_cls}" data-val="{s}">{s}/4</td>'
                f'<td data-val="{r["entry"]:.2f}">{r["entry"]:.2f}</td>'
                f'<td data-val="{r["sl"]:.2f}">{r["sl"]:.2f}</td>'
                f'<td data-val="{r["tp1"]:.2f}">{r["tp1"]:.2f}</td>'
                f'<td data-val="{r["tp2"]:.2f}">{r["tp2"]:.2f}</td>'
                f'<td data-val="{r["tp3"]:.2f}">{r["tp3"]:.2f}</td>'
                f"<td>{r['trade_status']}{tp_marks}</td>"
                f'<td data-val="{r["bars_ago"]}">{r["bars_ago"]}G</td>'
                f'<td data-val="{prob_val}">{prob_str}</td>'
                f'<td class="{bkt_cls}" data-val="{bucket}">{bucket}</td>'
                f"<td>{gate_html}</td>"
                f"</tr>"
            )
        return "\n".join(rows)

    def _box_rows():
        rows = []
        for r in sorted(boxes, key=lambda x: x.get("proximity", 0), reverse=True)[:40]:
            dist_top = (r["box_top"] - r["close"]) / r["close"] * 100
            dist_bot = (r["close"] - r["box_bot"]) / r["close"] * 100
            closer_val = dist_top if dist_top < dist_bot else -dist_bot
            closer = f"↑{dist_top:.1f}%" if dist_top < dist_bot else f"↓{dist_bot:.1f}%"
            rows.append(
                f"<tr>"
                f"<td><b>{_tk(r['ticker'])}</b></td>"
                f'<td data-val="{r["close"]:.2f}">{r["close"]:.2f}</td>'
                f'<td data-val="{r["box_top"]:.2f}">{r["box_top"]:.2f}</td>'
                f'<td data-val="{r["box_bot"]:.2f}">{r["box_bot"]:.2f}</td>'
                f'<td data-val="{closer_val:.2f}">{closer}</td>'
                f'<td data-val="{r["squeeze_bars"]}">{r["squeeze_bars"]}</td>'
                f'<td data-val="{r["bars_since"]}">{r["bars_since"]}G</td>'
                f"</tr>"
            )
        return "\n".join(rows)

    def _sq_rows():
        rows = []
        for r in sorted(squeezes, key=lambda x: x["squeeze_bars"], reverse=True):
            rows.append(
                f"<tr>"
                f"<td><b>{_tk(r['ticker'])}</b></td>"
                f'<td data-val="{r["close"]:.2f}">{r["close"]:.2f}</td>'
                f'<td data-val="{r["squeeze_bars"]}">{r["squeeze_bars"]}</td>'
                f"</tr>"
            )
        return "\n".join(rows)

    return f"""<!DOCTYPE html>
<html lang="tr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Smart Breakout Targets — {date_str}</title>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ font-family:'Segoe UI',system-ui,sans-serif; background:#0d1117; color:#c9d1d9; padding:20px; }}
  h1 {{ color:#58a6ff; margin-bottom:5px; }}
  .subtitle {{ color:#8b949e; margin-bottom:20px; }}
  .section {{ background:#161b22; border:1px solid #30363d; border-radius:8px; padding:16px; margin-bottom:20px; }}
  .section h2 {{ color:#f0f6fc; margin-bottom:12px; font-size:1.1em; }}
  table {{ width:100%; border-collapse:collapse; font-size:0.85em; }}
  th {{ background:#21262d; color:#8b949e; padding:8px 6px; text-align:left; border-bottom:2px solid #30363d; }}
  td {{ padding:6px; border-bottom:1px solid #21262d; }}
  tr:hover {{ background:#1c2128; }}
  .long {{ color:#3fb950; font-weight:bold; }}
  .short {{ color:#f85149; font-weight:bold; }}
  .strong {{ color:#3fb950; font-weight:bold; }}
  .medium {{ color:#d29922; }}
  .normal {{ color:#8b949e; }}
  .stat {{ display:inline-block; background:#21262d; border-radius:6px; padding:8px 16px; margin:4px; text-align:center; }}
  .stat .num {{ font-size:1.4em; font-weight:bold; color:#58a6ff; }}
  .stat .lbl {{ font-size:0.8em; color:#8b949e; }}
  a.tk {{ color:#58a6ff; text-decoration:none; }}
  a.tk:hover {{ text-decoration:underline; color:#79c0ff; }}
  th.sortable {{ cursor:pointer; user-select:none; position:relative; padding-right:18px; }}
  th.sortable:hover {{ color:#f0f6fc; background:#30363d; }}
  th.sortable::after {{ content:'⇅'; position:absolute; right:4px; opacity:0.4; font-size:0.8em; }}
  th.sortable.asc::after {{ content:'↑'; opacity:1; }}
  th.sortable.desc::after {{ content:'↓'; opacity:1; }}
</style>
<script>
function sortTable(th) {{
  const table = th.closest('table');
  const idx = Array.from(th.parentNode.children).indexOf(th);
  const tbody = table.querySelector('tbody') || table;
  const rows = Array.from(tbody.querySelectorAll('tr')).filter(r => r.querySelector('td'));
  const isAsc = th.classList.contains('asc');

  th.parentNode.querySelectorAll('th').forEach(h => h.classList.remove('asc','desc'));
  th.classList.add(isAsc ? 'desc' : 'asc');
  const dir = isAsc ? -1 : 1;

  rows.sort((a, b) => {{
    const ca = a.children[idx], cb = b.children[idx];
    let va = ca.getAttribute('data-val') || ca.textContent.trim();
    let vb = cb.getAttribute('data-val') || cb.textContent.trim();
    const na = parseFloat(va), nb = parseFloat(vb);
    if (!isNaN(na) && !isNaN(nb)) return (na - nb) * dir;
    return va.localeCompare(vb, 'tr') * dir;
  }});
  rows.forEach(r => tbody.appendChild(r));
}}
</script>
</head>
<body>
<h1>Smart Breakout Targets</h1>
<p class="subtitle">{date_str} — Tüm BIST</p>

<div>
  <span class="stat"><span class="num">{len(breakouts)}</span><br><span class="lbl">Kırılma</span></span>
  <span class="stat"><span class="num">{len(boxes)}</span><br><span class="lbl">Kutu</span></span>
  <span class="stat"><span class="num">{len(squeezes)}</span><br><span class="lbl">Sıkışma</span></span>
</div>

<div class="section">
  <h2>🚀 Kırılmalar (Aktif Trade)</h2>
  <table>
    <tr><th class="sortable" onclick="sortTable(this)">Sembol</th><th class="sortable" onclick="sortTable(this)">Yön</th><th class="sortable" onclick="sortTable(this)">Güç</th><th class="sortable" onclick="sortTable(this)">Giriş</th><th class="sortable" onclick="sortTable(this)">SL</th><th class="sortable" onclick="sortTable(this)">TP1</th><th class="sortable" onclick="sortTable(this)">TP2</th><th class="sortable" onclick="sortTable(this)">TP3</th><th class="sortable" onclick="sortTable(this)">Durum</th><th class="sortable" onclick="sortTable(this)">Gün</th><th class="sortable" onclick="sortTable(this)">ML</th><th class="sortable" onclick="sortTable(this)">Bucket</th><th>Gate</th></tr>
    {_bo_rows()}
  </table>
  {f'<p style="color:#8b949e;margin-top:8px;">Sonuç yok</p>' if not breakouts else ''}
</div>

<div class="section">
  <h2>📦 Kutu — Kırılma Bekleyen</h2>
  <table>
    <tr><th class="sortable" onclick="sortTable(this)">Sembol</th><th class="sortable" onclick="sortTable(this)">Kapanış</th><th class="sortable" onclick="sortTable(this)">Kutu Üst</th><th class="sortable" onclick="sortTable(this)">Kutu Alt</th><th class="sortable" onclick="sortTable(this)">Mesafe</th><th class="sortable" onclick="sortTable(this)">SQ Bar</th><th class="sortable" onclick="sortTable(this)">Bekleme</th></tr>
    {_box_rows()}
  </table>
  {f'<p style="color:#8b949e;margin-top:8px;">Sonuç yok</p>' if not boxes else ''}
</div>

<div class="section">
  <h2>🟡 Aktif Sıkışma</h2>
  <table>
    <tr><th class="sortable" onclick="sortTable(this)">Sembol</th><th class="sortable" onclick="sortTable(this)">Kapanış</th><th class="sortable" onclick="sortTable(this)">SQ Bar</th></tr>
    {_sq_rows()}
  </table>
  {f'<p style="color:#8b949e;margin-top:8px;">Sonuç yok</p>' if not squeezes else ''}
</div>

</body>
</html>"""


# ── Telegram + GitHub Pages ───────────────────────────────────
def _send_telegram(msg):
    token = os.environ.get("TG_BOT_TOKEN") or os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat = os.environ.get("TG_CHAT_ID") or os.environ.get("TELEGRAM_CHAT_ID", "")
    if not token or not chat:
        print("⚠️ Telegram token/chat yok, konsola yazdırılıyor:")
        print(msg)
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    for i in range(0, len(msg), 4000):
        chunk = msg[i : i + 4000]
        try:
            req.post(
                url,
                json={
                    "chat_id": chat,
                    "text": chunk,
                    "parse_mode": "HTML",
                    "disable_web_page_preview": True,
                },
                timeout=10,
            )
        except Exception:
            pass


def _send_telegram_document(filepath, caption=""):
    token = os.environ.get("TG_BOT_TOKEN") or os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat = os.environ.get("TG_CHAT_ID") or os.environ.get("TELEGRAM_CHAT_ID", "")
    if not token or not chat or not os.path.exists(filepath):
        return
    url = f"https://api.telegram.org/bot{token}/sendDocument"
    try:
        with open(filepath, "rb") as f:
            req.post(
                url,
                data={"chat_id": chat, "caption": caption},
                files={"document": f},
                timeout=30,
            )
        print("📤 HTML rapor Telegram'a gönderildi")
    except Exception as e:
        print(f"⚠️ Telegram document hata: {e}")


def _push_html_to_github(html_content, filename):
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN", "")
    repo = os.environ.get("GH_PAGES_REPO", "")
    if not token or not repo:
        return None
    api_url = f"https://api.github.com/repos/{repo}/contents/{filename}"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
    }
    content_b64 = base64.b64encode(html_content.encode("utf-8")).decode("ascii")
    sha = None
    try:
        resp = req.get(api_url, headers=headers, timeout=10)
        if resp.status_code == 200:
            sha = resp.json().get("sha")
    except Exception:
        pass
    now_tr = datetime.now(timezone(timedelta(hours=3)))
    payload = {
        "message": f"Smart Breakout — {now_tr.strftime('%d.%m.%Y')}",
        "content": content_b64,
        "branch": "main",
    }
    if sha:
        payload["sha"] = sha
    try:
        resp = req.put(api_url, headers=headers, json=payload, timeout=15)
        if resp.status_code in (200, 201):
            owner, name = repo.split("/")
            page_url = f"https://{owner}.github.io/{name}/{filename}"
            print(f"✅ HTML yayınlandı: {page_url}")
            return page_url
    except Exception:
        pass
    return None


def _build_telegram_message(results, html_url=None):
    now_tr = datetime.now(timezone(timedelta(hours=3)))
    date_str = now_tr.strftime("%d.%m.%Y %H:%M")
    bo = results["BREAKOUT"]
    bx = results["BOX"]
    sq = results["SQUEEZE"]

    lines = [
        f"<b>📊 Smart Breakout — {date_str}</b>",
        f"🚀 {len(bo)} kırılma | 📦 {len(bx)} kutu | 🟡 {len(sq)} sıkışma",
        "",
    ]

    if bo:
        lines.append("<b>🚀 Kırılmalar</b>")
        # ML bucket sıralı zaten
        for r in bo:
            s = r["strength"]
            bucket = r.get("ml_bucket", "?")
            prob = r.get("ml_prob")
            gate = r.get("ml_gate", "?")

            # Bucket ikonu
            if bucket.startswith("A+"):
                b_icon = "⭐"
            elif bucket == "A":
                b_icon = "🟢"
            elif bucket == "B":
                b_icon = "🟡"
            elif bucket == "X":
                b_icon = "⛔"
            else:
                b_icon = "⚪"

            prob_str = f"{prob:.2f}" if prob is not None else "-"
            gate_str = f" ⚠{gate}" if gate not in ("OK", "?") else ""

            lines.append(
                f"{b_icon}<b>{r['ticker']}</b> [{bucket}] {r['entry']:.2f} "
                f"ML:{prob_str} {r['bars_ago']}G{gate_str}"
            )
            lines.append(
                f"  SL:{r['sl']:.2f} TP1:{r['tp1']:.2f} TP2:{r['tp2']:.2f} TP3:{r['tp3']:.2f}"
            )
        lines.append("")

    if bx:
        lines.append(f"<b>📦 Kutu ({len(bx)})</b>")
        for r in sorted(bx, key=lambda x: x.get("proximity", 0), reverse=True)[:10]:
            dist_top = (r["box_top"] - r["close"]) / r["close"] * 100
            dist_bot = (r["close"] - r["box_bot"]) / r["close"] * 100
            closer = f"↑{dist_top:.1f}%" if dist_top < dist_bot else f"↓{dist_bot:.1f}%"
            lines.append(
                f"<b>{r['ticker']}</b> {r['close']:.2f} "
                f"[{r['box_bot']:.2f}-{r['box_top']:.2f}] {closer}"
            )
        if len(bx) > 10:
            lines.append(f"  ... +{len(bx) - 10} daha")
        lines.append("")

    if sq:
        top_sq = sorted(sq, key=lambda x: x["squeeze_bars"], reverse=True)[:15]
        tickers = " ".join(f"{r['ticker']}({r['squeeze_bars']})" for r in top_sq)
        lines.append(f"<b>🟡 Sıkışma ({len(sq)})</b>")
        lines.append(tickers)
        if len(sq) > 15:
            lines.append(f"  ... +{len(sq) - 15} daha")

    if html_url:
        lines.append(f"\n🔗 <a href=\"{html_url}\">Detaylı Rapor</a>")

    return "\n".join(lines)


def _notify_telegram(results, html_content, html_path):
    print("\n📤 Telegram'a gönderiliyor...")

    # GitHub Pages'e push
    html_url = _push_html_to_github(html_content, "smart_breakout.html")

    # Telegram mesajı
    msg = _build_telegram_message(results, html_url)
    _send_telegram(msg)

    # HTML dosyayı document olarak gönder
    _send_telegram_document(html_path, "📊 Smart Breakout Targets — Detaylı Rapor")

    print("✅ Telegram bildirim tamamlandı")


# ── CLI ───────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smart Breakout Targets — BIST Tarayıcı")
    parser.add_argument("--no-vol", action="store_true", help="Hacim filtresini kapat")
    parser.add_argument("--htf", action="store_true", help="HTF EMA filtresini aç")
    parser.add_argument("--days", type=int, default=180, help="Geriye bakış günü (varsayılan 180)")
    parser.add_argument("--max-age", type=int, default=5, help="Max kırılma yaşı (gün, varsayılan 5)")
    parser.add_argument("--notify", action="store_true", help="Telegram'a gönder + GitHub Pages push")
    args = parser.parse_args()

    run_scanner(
        use_vol=not args.no_vol,
        use_htf=args.htf,
        lookback_days=args.days,
        max_bo_age=args.max_age,
        notify=args.notify,
    )
