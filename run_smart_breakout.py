#!/usr/bin/env python3
"""
Smart Breakout Targets — Tüm BIST Tarayıcı
Squeeze → Kutu → Kırılma → TP/SL seviyeleri
"""

import os
import sys
import warnings
import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ── Parametreler ──────────────────────────────────────────────
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
            continue
        if df["squeeze"].iloc[i]:
            if not in_sq:
                sq_start = i
                in_sq = True
        else:
            if in_sq:
                sq_len = i - sq_start
                if sq_len >= MIN_SQUEEZE_BARS:
                    squeezes.append(
                        {"start": sq_start, "end": i - 1, "length": sq_len}
                    )
                in_sq = False

    # Hâlâ squeeze içindeyse
    if in_sq:
        sq_len = n - sq_start
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

        # SHORT kırılma
        if c_i < box_bot and c_i < o_i and impulse_ok and vol_ok:
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
    }


# ── Toplu tarama ──────────────────────────────────────────────
def run_scanner(use_vol=True, use_htf=False, lookback_days=180, max_bo_age=5):
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

    return results


def _print_breakouts(items):
    if not items:
        print("🚀 KIRILMA: Yok\n")
        return

    items.sort(key=lambda x: x["strength"], reverse=True)
    print(f"🚀 KIRILMA ({len(items)} adet)")
    print("─" * 100)
    print(
        f"{'Sembol':<10} {'Yön':<6} {'Güç':<12} {'Giriş':>8} {'SL':>8} "
        f"{'TP1':>8} {'TP2':>8} {'TP3':>8} {'Risk':>7} {'Durum':<12} {'Gün':>4} {'SQ':>3}"
    )
    print("─" * 100)
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

        print(
            f"{r['ticker']:<10} {r['dir']:<6} {r['strength_label']:<12} "
            f"{r['entry']:>8.2f} {r['sl']:>8.2f} {r['tp1']:>8.2f} "
            f"{r['tp2']:>8.2f} {r['tp3']:>8.2f} {r['risk']:>7.2f} "
            f"{status:<12} {r['bars_ago']:>4} {r['squeeze_bars']:>3}"
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
        for r in sorted(breakouts, key=lambda x: x["strength"], reverse=True):
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
    <tr><th class="sortable" onclick="sortTable(this)">Sembol</th><th class="sortable" onclick="sortTable(this)">Yön</th><th class="sortable" onclick="sortTable(this)">Güç</th><th class="sortable" onclick="sortTable(this)">Giriş</th><th class="sortable" onclick="sortTable(this)">SL</th><th class="sortable" onclick="sortTable(this)">TP1</th><th class="sortable" onclick="sortTable(this)">TP2</th><th class="sortable" onclick="sortTable(this)">TP3</th><th class="sortable" onclick="sortTable(this)">Durum</th><th class="sortable" onclick="sortTable(this)">Gün</th></tr>
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


# ── CLI ───────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smart Breakout Targets — BIST Tarayıcı")
    parser.add_argument("--no-vol", action="store_true", help="Hacim filtresini kapat")
    parser.add_argument("--htf", action="store_true", help="HTF EMA filtresini aç")
    parser.add_argument("--days", type=int, default=180, help="Geriye bakış günü (varsayılan 180)")
    parser.add_argument("--max-age", type=int, default=5, help="Max kırılma yaşı (gün, varsayılan 5)")
    args = parser.parse_args()

    run_scanner(
        use_vol=not args.no_vol,
        use_htf=args.htf,
        lookback_days=args.days,
        max_bo_age=args.max_age,
    )
