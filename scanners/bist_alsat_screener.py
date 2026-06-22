"""
BIST AL/SAT SCREENER — Forward Test Bulgularına Dayalı Filtreli Tarama
======================================================================
bist_rejim_v3_screener.py'nin analiz motorunu kullanır,
forward test sonuçlarına göre AL/İZLE/ATLA sınıflandırması yapar.

Sinyal filtreleri (12-25 Şub 2026 forward test):
  GUCLU : RS 30-60 → AL,  RS 0-30 veya 60+ → İZLE,  RS<0 → ATLA
  CMB   : Q≥80 → AL,  Q≥70+MACD>0 → İZLE,  geri kalan → ATLA
  ZAYIF : RS 20-60+MACD>0+Q≥50 → AL,  MACD>0+Q≥40 → İZLE,  geri kalan → ATLA
  BILESEN: Q≥70+MACD>0 → AL,  Q≥50+MACD>0 → İZLE,  geri kalan → ATLA
  DONUS : Her zaman AL
  PB    : RS>0+Q≥50 → AL,  geri kalan → İZLE
  SQ    : RS>0+Q≥50 → AL,  geri kalan → İZLE
  MR    : RS>0 → AL,  geri kalan → İZLE
  ERKEN : Her zaman ATLA
  CMB+  : Her zaman ATLA
"""

import sys
import os
import time
import pandas as pd
import numpy as np
from datetime import datetime

# ── bist_rejim_v3_screener'dan import ──
from bist_rejim_v3_screener import (
    get_all_bist_tickers,
    fetch_data,
    fetch_benchmark,
    analyze,
    push_html_to_github,
    send_telegram,
    send_telegram_document,
    TELEGRAM_BOT_TOKEN,
    TELEGRAM_CHAT_ID,
)


# ============================================================
# MACD HESAPLAMA
# ============================================================

def calc_macd(close, fast=12, slow=26, signal=9):
    """MACD line, signal line ve histogram döndür."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


# ============================================================
# OE SKORU — Overextended / Aşırı Uzanma (0-5, uyarı amaçlı)
# ============================================================

def calc_oe(df):
    """
    Overbought/Overextended skoru (0-5). Yüksek = aşırı uzamış, geç giriş riski.
    5 bileşen (her biri 0 veya 1):
      1. RSI(2) > 90       — kısa vadede aşırı alım
      2. RSI(14) > 70      — orta vadede overbought
      3. BB %B > 1.0       — Bollinger üst bandı aşmış
      4. ATR genişleme      — son 5G ATR > 20G ATR x 1.5
      5. EMA21 uzaklığı     — fiyat EMA21'den >2 ATR uzakta (mean reversion riski)
    Returns: (skor, detay_str)
    """
    close = df['Close']

    # RSI hesaplama (Wilder's smoothing)
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    # RSI(2)
    avg_gain2 = gain.ewm(alpha=1/2, adjust=False).mean()
    avg_loss2 = loss.ewm(alpha=1/2, adjust=False).mean()
    rs2 = avg_gain2 / avg_loss2.replace(0, np.nan)
    rsi2 = (100 - 100 / (1 + rs2)).iloc[-1]

    # RSI(14)
    avg_gain14 = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss14 = loss.ewm(alpha=1/14, adjust=False).mean()
    rs14 = avg_gain14 / avg_loss14.replace(0, np.nan)
    rsi14 = (100 - 100 / (1 + rs14)).iloc[-1]

    # BB %B
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    lower = sma20 - 2 * std20
    upper = sma20 + 2 * std20
    bb_pctb = ((close - lower) / (upper - lower)).iloc[-1]

    # ATR genişleme
    h, l, pc = df['High'], df['Low'], close.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    atr_5 = tr.rolling(5).mean().iloc[-1]
    atr_20 = tr.rolling(20).mean().iloc[-1]

    # EMA21 uzaklığı
    ema21 = close.ewm(span=21, adjust=False).mean()
    ema21_val = ema21.iloc[-1]
    close_val = close.iloc[-1]
    atr14 = tr.ewm(alpha=1/14, adjust=False).mean().iloc[-1]

    # NaN kontrol
    if pd.isna(rsi2): rsi2 = 50.0
    if pd.isna(rsi14): rsi14 = 50.0
    if pd.isna(bb_pctb): bb_pctb = 0.5
    if pd.isna(atr_5) or pd.isna(atr_20) or atr_20 == 0:
        atr_expanded = False
    else:
        atr_expanded = atr_5 > atr_20 * 1.5
    if pd.isna(ema21_val) or pd.isna(atr14) or atr14 == 0:
        ema_stretched = False
    else:
        ema_stretched = (close_val - ema21_val) > 2 * atr14

    # Skor: 0-5
    score = 0
    parts = []

    if rsi2 > 90:
        score += 1
        parts.append('RSI2>90')
    if rsi14 > 70:
        score += 1
        parts.append('RSI14>70')
    if bb_pctb > 1.0:
        score += 1
        parts.append('BB>1')
    if atr_expanded:
        score += 1
        parts.append('ATR exp')
    if ema_stretched:
        score += 1
        parts.append('EMA21 uzak')

    detail = ' '.join(parts) if parts else '-'
    return score, detail


# ============================================================
# SINYAL SINIFLANDIRMA (Forward Test Kuralları)
# ============================================================

def classify(result, macd_hist_val):
    """
    analyze() sonucunu AL/İZLE/ATLA olarak sınıflandır.
    Returns: (karar, sebep)
      karar: "AL", "İZLE", "ATLA"
      sebep: Kısa açıklama
    """
    sig = result['signal']
    rs = result.get('rs_score', 0.0)
    q = result.get('quality', 0)
    macd_pos = macd_hist_val > 0 if not np.isnan(macd_hist_val) else False

    # ── ERKEN ve CMB+ her zaman ATLA ──
    if sig == "ERKEN":
        return "ATLA", "ERKEN sinyal — forward testte düşük WR"
    if sig == "CMB+":
        return "ATLA", "CMB+ — forward testte yetersiz performans"

    # ── DÖNÜŞ her zaman AL ──
    if sig == "DONUS":
        return "AL", "DÖNÜŞ — forward testte en yüksek WR"

    # ── GÜÇLÜ: RS bazlı ──
    if sig == "GUCLU":
        if rs < 0:
            return "ATLA", f"RS negatif ({rs:.1f})"
        if 30 <= rs <= 60:
            return "AL", f"RS sweet-spot ({rs:.1f})"
        # RS 0-30 veya 60+
        return "İZLE", f"RS {rs:.1f} (optimal dışı)"

    # ── CMB: Quality bazlı ──
    if sig == "CMB":
        if q >= 80:
            return "AL", f"Q≥80 ({q})"
        if q >= 70 and macd_pos:
            return "İZLE", f"Q={q} + MACD>0"
        return "ATLA", f"Q={q}" + ("" if macd_pos else " + MACD≤0")

    # ── ZAYIF: RS + MACD + Quality ──
    if sig == "ZAYIF":
        if 20 <= rs <= 60 and macd_pos and q >= 50:
            return "AL", f"RS={rs:.1f} MACD>0 Q={q}"
        if macd_pos and q >= 40:
            return "İZLE", f"MACD>0 Q={q}"
        return "ATLA", f"RS={rs:.1f} Q={q}" + ("" if macd_pos else " MACD≤0")

    # ── BİLEŞEN: Quality + MACD ──
    if sig == "BILESEN":
        if q >= 70 and macd_pos:
            return "AL", f"Q={q} + MACD>0"
        if q >= 50 and macd_pos:
            return "İZLE", f"Q={q} + MACD>0"
        return "ATLA", f"Q={q}" + ("" if macd_pos else " + MACD≤0")

    # ── PB: RS + Quality ──
    if sig == "PB":
        if rs > 0 and q >= 50:
            return "AL", f"RS={rs:.1f} Q={q}"
        return "İZLE", f"RS={rs:.1f} Q={q}"

    # ── SQ: RS + Quality ──
    if sig == "SQ":
        if rs > 0 and q >= 50:
            return "AL", f"RS={rs:.1f} Q={q}"
        return "İZLE", f"RS={rs:.1f} Q={q}"

    # ── MR: RS bazlı ──
    if sig == "MR":
        if rs > 0:
            return "AL", f"RS pozitif ({rs:.1f})"
        return "İZLE", f"RS negatif ({rs:.1f})"

    # Fallback (bilinmeyen sinyal)
    return "İZLE", f"Bilinmeyen sinyal: {sig}"


# ============================================================
# HTML DASHBOARD (3 bölümlü)
# ============================================================

def generate_html(results, total):
    """AL/İZLE/ATLA bölümlü HTML dashboard oluştur."""
    now = datetime.now().strftime("%d.%m.%Y %H:%M")

    al = [r for r in results if r['karar'] == 'AL']
    izle = [r for r in results if r['karar'] == 'İZLE']
    atla = [r for r in results if r['karar'] == 'ATLA']

    # Sinyal dağılımı
    from collections import Counter
    sig_dist = Counter(r['signal'] for r in results)
    karar_dist = {'AL': len(al), 'İZLE': len(izle), 'ATLA': len(atla)}

    # JSON data (JS filtreleri için)
    import json
    json_data = json.dumps(results, ensure_ascii=False, default=str)

    def row_html(r, idx):
        """Tek satır HTML."""
        sig_colors = {
            "CMB+": "#e74c3c", "CMB": "#e67e22", "GUCLU": "#27ae60",
            "ZAYIF": "#f1c40f", "DONUS": "#9b59b6", "ERKEN": "#e67e22",
            "PB": "#3498db", "SQ": "#2ecc71", "MR": "#95a5a6", "BILESEN": "#1abc9c"
        }
        regime_colors = {
            "FULL_TREND": "#27ae60", "TREND": "#2ecc71",
            "GRI_BOLGE": "#f39c12", "CHOPPY": "#e74c3c"
        }
        regime_short = {"FULL_TREND": "FT", "TREND": "TR", "GRI_BOLGE": "GR", "CHOPPY": "CH"}
        karar_bg = {"AL": "rgba(39,174,96,0.12)", "İZLE": "rgba(241,196,15,0.12)", "ATLA": "rgba(149,165,166,0.12)"}
        karar_badge = {
            "AL": '<span style="background:#27ae60;color:#fff;padding:2px 8px;border-radius:4px;font-weight:bold">AL</span>',
            "İZLE": '<span style="background:#f39c12;color:#fff;padding:2px 8px;border-radius:4px;font-weight:bold">İZLE</span>',
            "ATLA": '<span style="background:#95a5a6;color:#fff;padding:2px 8px;border-radius:4px;font-weight:bold">ATLA</span>',
        }
        sig_c = sig_colors.get(r['signal'], '#888')
        reg_c = regime_colors.get(r['regime'], '#888')
        reg_s = regime_short.get(r['regime'], r['regime'][:2])
        bg = karar_bg.get(r['karar'], '')
        tv_url = f"https://www.tradingview.com/chart/?symbol=BIST:{r['ticker']}"

        macd_val = r.get('macd_hist', 0)
        macd_color = '#27ae60' if macd_val > 0 else '#e74c3c'

        # OE badge (overextended 0-5)
        oe_val = r.get('oe', 0)
        oe_tip = r.get('oe_detail', '-')
        if oe_val >= 4:
            oe_badge = f'<span style="background:#e74c3c;color:#fff;padding:2px 6px;border-radius:3px;font-size:0.8em" title="{oe_tip}">{oe_val}/5</span>'
        elif oe_val == 3:
            oe_badge = f'<span style="background:#e67e22;color:#fff;padding:2px 6px;border-radius:3px;font-size:0.8em" title="{oe_tip}">{oe_val}/5</span>'
        elif oe_val == 2:
            oe_badge = f'<span style="color:#f1c40f;font-size:0.85em" title="{oe_tip}">{oe_val}/5</span>'
        elif oe_val == 1:
            oe_badge = f'<span style="color:#888;font-size:0.85em" title="{oe_tip}">{oe_val}/5</span>'
        else:
            oe_badge = f'<span style="color:#555;font-size:0.85em">0</span>'

        return f'''<tr style="background:{bg}" data-karar="{r['karar']}" data-regime="{r['regime']}" data-signal="{r['signal']}" data-ticker="{r['ticker']}">
  <td>{idx}</td>
  <td><a href="{tv_url}" target="_blank" style="color:#3498db;text-decoration:none;font-weight:bold">{r['ticker']}</a></td>
  <td>{karar_badge[r['karar']]}</td>
  <td><span style="background:{sig_c};color:#fff;padding:2px 6px;border-radius:3px;font-size:0.85em">{r['signal']}</span></td>
  <td><span style="color:{reg_c};font-weight:bold">{reg_s}</span></td>
  <td style="text-align:right">{r['close']:.2f}</td>
  <td style="text-align:right;color:#e74c3c">{r['stop']:.2f}</td>
  <td style="text-align:right;color:#27ae60">{r['tp']:.2f}</td>
  <td style="text-align:right">{r['rr']:.1f}</td>
  <td style="text-align:right">{r['rs_score']:.1f}</td>
  <td style="text-align:right">{r['quality']}</td>
  <td style="text-align:right">{r['rvol']:.1f}</td>
  <td style="text-align:right;color:{macd_color}">{macd_val:.3f}</td>
  <td style="text-align:right">{oe_badge}</td>
  <td style="font-size:0.8em;color:#aaa">{r['sebep']}</td>
</tr>'''

    # Tüm satırlar
    rows_al = '\n'.join(row_html(r, i+1) for i, r in enumerate(al))
    rows_izle = '\n'.join(row_html(r, i+1) for i, r in enumerate(izle))
    rows_atla = '\n'.join(row_html(r, i+1) for i, r in enumerate(atla))

    # Sinyal chip'leri
    sig_chips = ' '.join(
        f'<span style="background:#333;padding:3px 8px;border-radius:12px;font-size:0.85em;margin:2px">'
        f'{s}: {c}</span>'
        for s, c in sorted(sig_dist.items(), key=lambda x: -x[1])
    )

    html = f'''<!DOCTYPE html>
<html lang="tr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>BIST AL/SAT Tarama — {now}</title>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
         background:#0d1117; color:#e6edf3; padding:16px; }}
  .header {{ text-align:center; margin-bottom:20px; }}
  .header h1 {{ font-size:1.6em; margin-bottom:4px; }}
  .header .sub {{ color:#8b949e; font-size:0.9em; }}
  .cards {{ display:flex; gap:12px; justify-content:center; margin:16px 0; flex-wrap:wrap; }}
  .card {{ padding:16px 28px; border-radius:10px; text-align:center; min-width:140px; }}
  .card-al {{ background:rgba(39,174,96,0.15); border:1px solid #27ae60; }}
  .card-izle {{ background:rgba(241,196,15,0.15); border:1px solid #f1c40f; }}
  .card-atla {{ background:rgba(149,165,166,0.15); border:1px solid #95a5a6; }}
  .card .num {{ font-size:2em; font-weight:bold; }}
  .card .lbl {{ font-size:0.85em; color:#8b949e; margin-top:4px; }}
  .card-al .num {{ color:#27ae60; }}
  .card-izle .num {{ color:#f1c40f; }}
  .card-atla .num {{ color:#95a5a6; }}
  .filters {{ display:flex; gap:8px; flex-wrap:wrap; margin:12px 0; align-items:center; justify-content:center; }}
  .filters select, .filters input {{
    background:#161b22; color:#e6edf3; border:1px solid #30363d;
    padding:6px 10px; border-radius:6px; font-size:0.9em;
  }}
  .chips {{ text-align:center; margin:8px 0; }}
  .section {{ margin:24px 0; }}
  .section h2 {{ font-size:1.2em; margin-bottom:8px; padding-left:4px;
                  border-left:3px solid; padding:4px 8px; }}
  .section-al h2 {{ border-color:#27ae60; color:#27ae60; }}
  .section-izle h2 {{ border-color:#f1c40f; color:#f1c40f; }}
  .section-atla h2 {{ border-color:#95a5a6; color:#95a5a6; }}
  table {{ width:100%; border-collapse:collapse; font-size:0.85em; }}
  th {{ background:#161b22; padding:8px 6px; text-align:left; cursor:pointer;
       border-bottom:2px solid #30363d; position:sticky; top:0; }}
  th:hover {{ background:#1c2333; }}
  td {{ padding:6px; border-bottom:1px solid #21262d; }}
  tr:hover {{ background:rgba(56,139,253,0.1) !important; }}
  .toggle-btn {{
    background:#161b22; color:#8b949e; border:1px solid #30363d;
    padding:8px 16px; border-radius:6px; cursor:pointer; margin:8px 4px;
    font-size:0.9em;
  }}
  .toggle-btn:hover {{ background:#1c2333; }}
  .toggle-btn.active {{ background:#1c2333; color:#e6edf3; border-color:#58a6ff; }}
  .notes {{ background:#161b22; border:1px solid #30363d; border-radius:8px;
            padding:16px; margin:16px 0; display:none; font-size:0.9em; line-height:1.6; }}
  .notes.show {{ display:block; }}
  a {{ color:#58a6ff; }}
  @media(max-width:768px) {{
    table {{ font-size:0.75em; }}
    .card {{ min-width:100px; padding:12px 16px; }}
  }}
</style>
</head>
<body>

<div class="header">
  <h1>BIST AL/SAT Tarama</h1>
  <div class="sub">{now} | Taranan: {total} | Sinyal: {len(results)}</div>
</div>

<div class="cards">
  <div class="card card-al"><div class="num">{karar_dist['AL']}</div><div class="lbl">AL</div></div>
  <div class="card card-izle"><div class="num">{karar_dist['İZLE']}</div><div class="lbl">İZLE</div></div>
  <div class="card card-atla"><div class="num">{karar_dist['ATLA']}</div><div class="lbl">ATLA</div></div>
</div>

<div class="chips">{sig_chips}</div>

<div class="filters">
  <select id="fKarar" onchange="applyFilters()">
    <option value="">Tümü</option>
    <option value="AL">AL</option>
    <option value="İZLE">İZLE</option>
    <option value="ATLA">ATLA</option>
  </select>
  <select id="fRejim" onchange="applyFilters()">
    <option value="">Tüm Rejimler</option>
    <option value="FULL_TREND">Full Trend</option>
    <option value="TREND">Trend</option>
    <option value="GRI_BOLGE">Gri Bölge</option>
    <option value="CHOPPY">Choppy</option>
  </select>
  <select id="fSignal" onchange="applyFilters()">
    <option value="">Tüm Sinyaller</option>
    <option value="GUCLU">GÜÇLÜ</option>
    <option value="CMB">CMB</option>
    <option value="ZAYIF">ZAYIF</option>
    <option value="BILESEN">BİLEŞEN</option>
    <option value="DONUS">DÖNÜŞ</option>
    <option value="PB">PB</option>
    <option value="SQ">SQ</option>
    <option value="MR">MR</option>
    <option value="ERKEN">ERKEN</option>
    <option value="CMB+">CMB+</option>
  </select>
  <input id="fTicker" type="text" placeholder="Hisse ara..." oninput="applyFilters()">
</div>

<div style="text-align:center">
  <button class="toggle-btn" onclick="toggleNotes(this)">Forward Test Notları</button>
  <button class="toggle-btn" onclick="toggleSection('atla-section',this)">ATLA Sinyallerini Göster</button>
</div>

<div id="fwd-notes" class="notes">
  <b>Forward Test Bulguları (12-25 Şubat 2026, XU100 düşüş dönemi)</b><br><br>
  <table style="width:100%;font-size:0.9em;border-collapse:collapse;margin:8px 0">
    <thead><tr style="border-bottom:1px solid #30363d;text-align:left">
      <th style="padding:4px 8px">Sinyal</th><th style="padding:4px 8px">Filtre</th>
      <th style="padding:4px 8px">N</th><th style="padding:4px 8px">5G WR</th>
      <th style="padding:4px 8px">5G Ort</th><th style="padding:4px 8px">Karar</th>
    </tr></thead>
    <tbody>
      <tr><td style="padding:3px 8px;color:#27ae60;font-weight:bold">GÜÇLÜ</td>
        <td style="padding:3px 8px">Tümü (filtresiz)</td><td>71</td><td>%33.8</td><td style="color:#e74c3c">-1.91%</td><td style="color:#555">—</td></tr>
      <tr style="background:rgba(39,174,96,0.08)"><td></td>
        <td style="padding:3px 8px">RS 30-60</td><td>16</td><td style="color:#27ae60;font-weight:bold">%68.8</td><td style="color:#27ae60">+4.08%</td><td style="color:#27ae60">AL</td></tr>
      <tr><td></td>
        <td style="padding:3px 8px">RS 0-30 veya 60+</td><td>55</td><td>%23.6</td><td style="color:#e74c3c">-3.66%</td><td style="color:#f1c40f">İZLE</td></tr>
      <tr style="border-top:1px solid #21262d"><td style="padding:3px 8px;color:#e67e22;font-weight:bold">CMB</td>
        <td style="padding:3px 8px">Tümü (filtresiz)</td><td>209</td><td>%25.8</td><td style="color:#e74c3c">-2.71%</td><td style="color:#555">—</td></tr>
      <tr style="background:rgba(39,174,96,0.08)"><td></td>
        <td style="padding:3px 8px">Q ≥ 80</td><td>25</td><td style="color:#27ae60;font-weight:bold">%44.0</td><td style="color:#27ae60">+2.67%</td><td style="color:#27ae60">AL</td></tr>
      <tr><td></td>
        <td style="padding:3px 8px">Q 70-79</td><td>15</td><td>%26.7</td><td style="color:#e74c3c">-2.22%</td><td style="color:#f1c40f">İZLE</td></tr>
      <tr><td></td>
        <td style="padding:3px 8px">Q &lt; 70</td><td>169</td><td>%23.1</td><td style="color:#e74c3c">-3.55%</td><td style="color:#95a5a6">ATLA</td></tr>
      <tr style="border-top:1px solid #21262d"><td style="padding:3px 8px;color:#f1c40f;font-weight:bold">ZAYIF</td>
        <td style="padding:3px 8px">Tümü (filtresiz)</td><td>164</td><td>%36.6</td><td style="color:#e74c3c">-0.77%</td><td style="color:#555">—</td></tr>
      <tr style="background:rgba(39,174,96,0.08)"><td></td>
        <td style="padding:3px 8px">RS 20-60 + MACD&gt;0 + Q≥50</td><td>17</td><td style="color:#27ae60;font-weight:bold">%70.6</td><td style="color:#27ae60">+9.22%</td><td style="color:#27ae60">AL</td></tr>
      <tr><td></td>
        <td style="padding:3px 8px">MACD&gt;0 + Q≥40</td><td>63</td><td>%28.6</td><td style="color:#e74c3c">-2.29%</td><td style="color:#f1c40f">İZLE</td></tr>
      <tr><td></td>
        <td style="padding:3px 8px">Geri kalan</td><td>84</td><td>%35.7</td><td style="color:#e74c3c">-1.65%</td><td style="color:#95a5a6">ATLA</td></tr>
      <tr style="border-top:1px solid #21262d"><td style="padding:3px 8px;color:#1abc9c;font-weight:bold">BİLEŞEN</td>
        <td style="padding:3px 8px">Tümü (filtresiz)</td><td>228</td><td>%39.9</td><td style="color:#e74c3c">-1.07%</td><td style="color:#555">—</td></tr>
      <tr style="background:rgba(39,174,96,0.08)"><td></td>
        <td style="padding:3px 8px">Q≥70 + MACD&gt;0</td><td>12</td><td style="color:#27ae60;font-weight:bold">%66.7</td><td style="color:#27ae60">+7.24%</td><td style="color:#27ae60">AL</td></tr>
      <tr><td></td>
        <td style="padding:3px 8px">Q≥50 + MACD&gt;0</td><td>21</td><td>%42.9</td><td style="color:#f1c40f">+0.53%</td><td style="color:#f1c40f">İZLE</td></tr>
      <tr><td></td>
        <td style="padding:3px 8px">Geri kalan</td><td>195</td><td>%37.9</td><td style="color:#e74c3c">-1.75%</td><td style="color:#95a5a6">ATLA</td></tr>
      <tr style="border-top:1px solid #21262d"><td style="padding:3px 8px;color:#9b59b6;font-weight:bold">DÖNÜŞ</td>
        <td style="padding:3px 8px">Tümü</td><td>25</td><td style="color:#27ae60;font-weight:bold">%52.0</td><td style="color:#27ae60">+2.86%</td><td style="color:#27ae60">Her zaman AL</td></tr>
      <tr style="border-top:1px solid #21262d"><td style="padding:3px 8px;color:#e67e22;font-weight:bold">ERKEN</td>
        <td style="padding:3px 8px">Tümü</td><td>85</td><td style="color:#e74c3c">%11.8</td><td style="color:#e74c3c">-7.43%</td><td style="color:#95a5a6">Her zaman ATLA</td></tr>
      <tr style="border-top:1px solid #21262d"><td style="padding:3px 8px;color:#e74c3c;font-weight:bold">CMB+</td>
        <td style="padding:3px 8px">Tümü</td><td>23</td><td style="color:#e74c3c">%21.7</td><td style="color:#e74c3c">-3.74%</td><td style="color:#95a5a6">Her zaman ATLA</td></tr>
    </tbody>
  </table>
  <br><i style="color:#8b949e">Dönem: 12 Ocak — 25 Şubat 2026 (10 tarama günü, N=809 sinyal, 5G=5 iş günü getirisi)</i>
</div>

<div class="section section-al">
  <h2>AL Sinyalleri ({len(al)})</h2>
  <table id="table-al">
    <thead><tr>
      <th>#</th><th>Hisse</th><th>Karar</th><th>Sinyal</th><th>Rejim</th>
      <th>Fiyat</th><th>Stop</th><th>TP</th><th>R:R</th><th>RS</th>
      <th>Q</th><th>RVOL</th><th>MACD</th><th>OE</th><th>Sebep</th>
    </tr></thead>
    <tbody>{rows_al}</tbody>
  </table>
</div>

<div class="section section-izle">
  <h2>İZLE Sinyalleri ({len(izle)})</h2>
  <table id="table-izle">
    <thead><tr>
      <th>#</th><th>Hisse</th><th>Karar</th><th>Sinyal</th><th>Rejim</th>
      <th>Fiyat</th><th>Stop</th><th>TP</th><th>R:R</th><th>RS</th>
      <th>Q</th><th>RVOL</th><th>MACD</th><th>OE</th><th>Sebep</th>
    </tr></thead>
    <tbody>{rows_izle}</tbody>
  </table>
</div>

<div id="atla-section" class="section section-atla" style="display:none">
  <h2>ATLA Sinyalleri ({len(atla)})</h2>
  <table id="table-atla">
    <thead><tr>
      <th>#</th><th>Hisse</th><th>Karar</th><th>Sinyal</th><th>Rejim</th>
      <th>Fiyat</th><th>Stop</th><th>TP</th><th>R:R</th><th>RS</th>
      <th>Q</th><th>RVOL</th><th>MACD</th><th>OE</th><th>Sebep</th>
    </tr></thead>
    <tbody>{rows_atla}</tbody>
  </table>
</div>

<script>
const DATA = {json_data};

function applyFilters() {{
  const fK = document.getElementById('fKarar').value;
  const fR = document.getElementById('fRejim').value;
  const fS = document.getElementById('fSignal').value;
  const fT = document.getElementById('fTicker').value.toUpperCase();
  document.querySelectorAll('tbody tr').forEach(tr => {{
    let show = true;
    if (fK && tr.dataset.karar !== fK) show = false;
    if (fR && tr.dataset.regime !== fR) show = false;
    if (fS && tr.dataset.signal !== fS) show = false;
    if (fT && !tr.dataset.ticker.includes(fT)) show = false;
    tr.style.display = show ? '' : 'none';
  }});
  // Karar filtresi seçilince ilgili bölümleri göster
  if (fK === 'ATLA') {{
    document.getElementById('atla-section').style.display = 'block';
  }}
}}

function toggleNotes(btn) {{
  const n = document.getElementById('fwd-notes');
  n.classList.toggle('show');
  btn.classList.toggle('active');
}}

function toggleSection(id, btn) {{
  const s = document.getElementById(id);
  const visible = s.style.display !== 'none';
  s.style.display = visible ? 'none' : 'block';
  btn.classList.toggle('active');
}}

// Tablo sıralama
document.querySelectorAll('th').forEach((th, colIdx) => {{
  th.addEventListener('click', () => {{
    const table = th.closest('table');
    const tbody = table.querySelector('tbody');
    const rows = Array.from(tbody.querySelectorAll('tr'));
    const asc = th.dataset.sort !== 'asc';
    th.dataset.sort = asc ? 'asc' : 'desc';
    rows.sort((a, b) => {{
      const aVal = a.children[colIdx]?.textContent.trim() || '';
      const bVal = b.children[colIdx]?.textContent.trim() || '';
      const aNum = parseFloat(aVal);
      const bNum = parseFloat(bVal);
      if (!isNaN(aNum) && !isNaN(bNum)) return asc ? aNum - bNum : bNum - aNum;
      return asc ? aVal.localeCompare(bVal) : bVal.localeCompare(aVal);
    }});
    rows.forEach(r => tbody.appendChild(r));
  }});
}});
</script>

</body>
</html>'''
    return html


# ============================================================
# TELEGRAM FORMAT
# ============================================================

def format_telegram(results, total, html_url=None):
    """AL odaklı Telegram mesajı oluştur."""
    now = datetime.now().strftime("%d.%m.%Y")

    al = [r for r in results if r['karar'] == 'AL']
    izle = [r for r in results if r['karar'] == 'İZLE']
    atla = [r for r in results if r['karar'] == 'ATLA']

    emoji_map = {
        "CMB+": "🔴", "CMB": "🟠", "GUCLU": "🟢", "ZAYIF": "🟡",
        "DONUS": "🟣", "ERKEN": "🟠", "PB": "🔵", "SQ": "🟩",
        "MR": "⚪", "BILESEN": "🔵"
    }
    regime_short = {"FULL_TREND": "FT", "TREND": "TR", "GRI_BOLGE": "GR", "CHOPPY": "CH"}

    lines = [f"<b>📊 BIST AL/SAT — {now}</b>", ""]

    # ── AL sinyalleri ──
    if al:
        lines.append(f"<b>✅ AL ({len(al)} sinyal)</b>")
        lines.append("─────────────────")
        # Öncelik sırasına göre sırala
        priority = {"CMB+": 0, "CMB": 1, "GUCLU": 2, "ZAYIF": 3, "DONUS": 4,
                     "ERKEN": 5, "PB": 6, "SQ": 7, "MR": 8, "BILESEN": 9}
        al_sorted = sorted(al, key=lambda r: (priority.get(r['signal'], 99), -r['rr']))
        for r in al_sorted:
            em = emoji_map.get(r['signal'], '⬜')
            reg = regime_short.get(r['regime'], r['regime'][:2])
            oe_val = r.get('oe', 0)
            oe_tag = ""
            if oe_val >= 4:
                oe_tag = " ⚠️OE%d" % oe_val
            elif oe_val == 3:
                oe_tag = " 🟠OE%d" % oe_val
            lines.append(
                f"{em} <b>{r['ticker']}</b> {r['close']:.2f} [{r['signal']}] {reg}{oe_tag}"
            )
            lines.append(
                f"  S:{r['stop']:.2f} TP:{r['tp']:.2f} R:R={r['rr']:.1f} "
                f"RS:{r['rs_score']:.1f} Q:{r['quality']}"
            )
    else:
        lines.append("✅ AL: yok")

    lines.append("")

    # ── İZLE özet ──
    if izle:
        lines.append(f"👁 <b>İZLE: {len(izle)} sinyal</b> (detay raporda)")
    else:
        lines.append("👁 İZLE: yok")

    # ── ATLA özet ──
    if atla:
        from collections import Counter
        atla_dist = Counter(r['signal'] for r in atla)
        dist_str = ', '.join(f"{s}:{c}" for s, c in atla_dist.most_common())
        lines.append(f"🚫 ATLA: {len(atla)} sinyal ({dist_str})")

    lines.append("")
    lines.append(f"📋 Taranan: {total} | Toplam: {len(results)}")

    if html_url:
        lines.append(f'🔗 <a href="{html_url}">Detaylı Rapor</a>')

    return '\n'.join(lines)


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("📊 BIST AL/SAT SCREENER")
    print("=" * 60)

    # 1. Veri çek
    tickers = get_all_bist_tickers()
    all_data = fetch_data(tickers, period="1y")
    xu_df = fetch_benchmark(period="1y")

    # 2. Analiz + sınıflandırma
    results = []
    for ticker, df in all_data.items():
        try:
            res = analyze(ticker, df, xu_df)
            if res is None:
                continue

            # MACD hesapla
            _, _, macd_hist = calc_macd(df['Close'])
            macd_hist_val = macd_hist.iloc[-1] if len(macd_hist) > 0 else 0.0
            if pd.isna(macd_hist_val):
                macd_hist_val = 0.0

            # OE skoru (overextended uyarı)
            oe_score, oe_detail = calc_oe(df)

            # Sınıflandır
            karar, sebep = classify(res, macd_hist_val)

            # Numpy tiplerini Python native'e çevir
            clean = {}
            for k, v in res.items():
                if isinstance(v, (np.bool_,)):
                    clean[k] = bool(v)
                elif isinstance(v, (np.integer,)):
                    clean[k] = int(v)
                elif isinstance(v, (np.floating,)):
                    clean[k] = float(v)
                else:
                    clean[k] = v

            clean['karar'] = karar
            clean['sebep'] = sebep
            clean['macd_hist'] = round(float(macd_hist_val), 4)
            clean['oe'] = oe_score
            clean['oe_detail'] = oe_detail
            results.append(clean)
        except Exception as e:
            print(f"  [!] {ticker}: {e}")

    # 3. Sıralama: AL > İZLE > ATLA, içinde sinyal önceliği + R:R
    karar_order = {"AL": 0, "İZLE": 1, "ATLA": 2}
    priority = {"CMB+": 0, "CMB": 1, "GUCLU": 2, "ZAYIF": 3, "DONUS": 4,
                "ERKEN": 5, "PB": 6, "SQ": 7, "MR": 8, "BILESEN": 9}
    results.sort(key=lambda r: (
        karar_order.get(r['karar'], 9),
        priority.get(r['signal'], 99),
        -r.get('rr', 0)
    ))

    # 4. Konsol çıktısı
    al_count = sum(1 for r in results if r['karar'] == 'AL')
    izle_count = sum(1 for r in results if r['karar'] == 'İZLE')
    atla_count = sum(1 for r in results if r['karar'] == 'ATLA')

    print(f"\n{'='*60}")
    print(f"📊 SONUÇ: {len(results)} sinyal — AL:{al_count} İZLE:{izle_count} ATLA:{atla_count}")
    print(f"{'='*60}\n")

    karar_emoji = {"AL": "✅", "İZLE": "👁", "ATLA": "🚫"}
    for r in results:
        ke = karar_emoji.get(r['karar'], '?')
        print(
            f"  {ke} {r['karar']:4s} | {r['ticker']:6s} {r['signal']:7s} "
            f"{r['regime']:11s} | {r['close']:8.2f} S:{r['stop']:.2f} TP:{r['tp']:.2f} "
            f"R:R={r['rr']:.1f} RS:{r['rs_score']:.1f} Q:{r['quality']} "
            f"MACD:{r['macd_hist']:.3f} OE:{r['oe']}/5 | {r['sebep']}"
        )

    # 5. HTML rapor
    date_str = datetime.now().strftime("%Y%m%d")
    html_content = generate_html(results, len(all_data))
    html_fname = f"alsat_{date_str}.html"
    with open(html_fname, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"\n📄 HTML: {html_fname}")

    # 6. GitHub Pages
    html_url = push_html_to_github(html_content, date_str)
    if html_url:
        print(f"🔗 {html_url}")

    # 7. Telegram
    msg = format_telegram(results, len(all_data), html_url)
    send_telegram(msg)
    send_telegram_document(html_fname)

    # 8. CSV
    csv_fname = f"alsat_signals_{date_str}.csv"
    if results:
        pd.DataFrame(results).to_csv(csv_fname, index=False)
        print(f"📊 CSV: {csv_fname}")

    print(f"\n✅ Tamamlandı — {len(results)} sinyal ({al_count} AL)")
    return results


if __name__ == "__main__":
    main()
