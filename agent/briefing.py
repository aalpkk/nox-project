"""
NOX Agent — Otomatik Brifing Pipeline
GH Actions context: CSV + makro → Claude API → Telegram + HTML rapor.

Kullanım:
    python -m agent.briefing              # lokal çalıştır
    python -m agent.briefing --notify     # Telegram + GitHub Pages
    python -m agent.briefing --no-ai      # Claude API kullanmadan (sadece veri)
"""
import argparse
import os
import sys
from datetime import datetime, timezone, timedelta

# Proje kökünü path'e ekle
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(ROOT, '.env'))

from agent.scanner_reader import (summarize_signals, SCREENER_NAMES,
                                   export_signals_json,
                                   fetch_mkk_data, fetch_takas_data,
                                   fetch_takas_history)
from agent.html_signals import fetch_all_html_signals
from agent.macro import (fetch_macro_data, fetch_macro_snapshot, assess_macro_regime,
                         format_macro_summary, calc_category_regimes,
                         fetch_market_news)
from agent.confluence import calc_all_confluence, format_confluence_summary
from agent.smart_money import (calc_batch_sms, format_sms_line, sms_icon,
                               classify_kurum_sms)
from agent.institutional import calc_batch_ice
from agent.institutional import format_sms_line as ice_format_line
from agent.institutional import sms_icon as ice_sms_icon
from agent.html_report import generate_briefing_html
from agent.prompts import BRIEFING_PROMPT
from core.reports import send_telegram, push_html_to_github

_TZ_TR = timezone(timedelta(hours=3))

# ── Portföy + Günlük Input ──
_AGENT_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_portfolio():
    """agent/portfolio.json'dan portföy bilgisi yükle."""
    path = os.path.join(_AGENT_DIR, 'portfolio.json')
    try:
        import json
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, ValueError):
        return None


def _load_daily_input():
    """agent/daily_input/ altındaki en güncel dosyayı yükle."""
    import glob
    input_dir = os.path.join(_AGENT_DIR, 'daily_input')
    # .md ve .txt dosyalarını ara
    files = sorted(glob.glob(os.path.join(input_dir, '*.md')))
    files += sorted(glob.glob(os.path.join(input_dir, '*.txt')))
    if not files:
        return None
    latest = files[-1]
    try:
        with open(latest, 'r', encoding='utf-8') as f:
            content = f.read()
        fname = os.path.basename(latest)
        return {"file": fname, "content": content}
    except Exception:
        return None

# ── Scanner HTML Link Registry ──
# base_url: None = GH_PAGES_BASE_URL (nox-signals), aksi halde özel URL
_SCANNER_HTML_LINKS = [
    {"name": "NOX v3 Haftalık Pivot", "file": "nox_v3_weekly.html"},
    {"name": "Regime Transition", "file": "regime_transition.html"},
    {"name": "RT Haftalık", "file": "regime_transition_weekly.html"},
    {"name": "US Catalyst", "file": "us_catalyst.html"},
    {"name": "Divergence", "file": "nox_divergence.html"},
    {"name": "Tavan Scanner", "file": "tavan.html", "base_url": "https://aalpkk.github.io/bist-signals"},
]


def _build_scanner_links():
    """GH Pages base URL'den scanner linkleri oluştur."""
    base = os.environ.get("GH_PAGES_BASE_URL", "").rstrip("/")
    if not base:
        return ""
    lines = ["", "<b>📊 Scanner Raporları:</b>"]
    for item in _SCANNER_HTML_LINKS:
        item_base = item.get("base_url", base)
        lines.append(f'  🔗 <a href="{item_base}/{item["file"]}">{item["name"]}</a>')
    return "\n".join(lines)


def _run_fresh_scanners():
    """NW + RT scanner'ları çalıştırıp güncel CSV üret."""
    import subprocess
    scanners = [
        ("NOX v3 Haftalık", [sys.executable, os.path.join(ROOT, "run_nox_v3.py"),
                              "--weekly", "--csv"]),
        ("Regime Transition", [sys.executable, os.path.join(ROOT, "run_regime_transition.py"),
                                "--csv"]),
    ]
    for name, cmd in scanners:
        print(f"  🔄 {name} çalıştırılıyor...")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True,
                                    timeout=600, cwd=ROOT)
            if result.returncode == 0:
                # Kaç sinyal üretildi
                lines = result.stdout.strip().split('\n')
                last_lines = [l for l in lines[-5:] if l.strip()]
                for l in last_lines:
                    print(f"    {l}")
            else:
                print(f"    ⚠️ Hata (exit {result.returncode})")
                if result.stderr:
                    print(f"    {result.stderr[:200]}")
        except subprocess.TimeoutExpired:
            print(f"    ⚠️ Timeout (10dk)")
        except Exception as e:
            print(f"    ⚠️ {e}")


def _get_sm_info(ticker, ice_results, sms_scores):
    """ICE/SMS bilgisini çek (ICE öncelikli, SMS fallback).
    Returns: (score_val, icon_str, mult_tag) veya (None, None, None)
    """
    active = ice_results or sms_scores
    if not active:
        return None, None, None
    sm = active.get(ticker)
    if not sm and ice_results and sms_scores:
        sm = sms_scores.get(ticker)
    if not sm:
        return None, None, None
    sm_val = sm.score if hasattr(sm, 'score') else sm
    sm_icon_str = (sm.icon if hasattr(sm, 'icon')
                   else (sms_icon(sm_val) if callable(sms_icon) else "⚪"))
    mult_tag = f"×{sm.multiplier:.2f}" if hasattr(sm, 'multiplier') else ""
    return sm_val, sm_icon_str, mult_tag


def _compute_4_lists(latest_signals, confluence_results=None,
                     sms_scores=None, ice_results=None):
    """4 kaynak bazlı öncelikli hisse listesi oluştur.

    Returns: dict with keys: 'alsat', 'tavan', 'nw', 'rt', 'tier1', 'tier2'
    Her value: [(ticker, score, reasons_list, signal_dict), ...] sıralı
    tier1: 2+ listede çakışan hisseler
    tier2: her listeden en kaliteli tekil hisseler
    """
    today = datetime.now(_TZ_TR)
    today_sd = today.strftime('%Y-%m-%d')
    today_cd = today.strftime('%Y%m%d')

    # Her screener'ın en son tarihi = "güncel veri" tarihi
    # Hafta sonu veya tatil günlerinde bugünün tarihi sinyallerle eşleşmez
    _screener_latest_sd = {}
    _screener_latest_cd = {}
    for s in latest_signals:
        scr = s.get('screener', '')
        sd = s.get('signal_date', '')
        cd = s.get('csv_date', '')
        if sd:
            if sd > _screener_latest_sd.get(scr, ''):
                _screener_latest_sd[scr] = sd
        if cd:
            if cd > _screener_latest_cd.get(scr, ''):
                _screener_latest_cd[scr] = cd

    def _is_today(s):
        sd = s.get('signal_date', '')
        cd = s.get('csv_date', '')
        if sd == today_sd or cd == today_cd:
            return True
        # Screener'ın en son tarihiyle karşılaştır (tatil/hafta sonu)
        scr = s.get('screener', '')
        return sd == _screener_latest_sd.get(scr, '') or cd == _screener_latest_cd.get(scr, '')

    # RT haritaları (CMF cross-ref için)
    rt_map = {}
    for s in latest_signals:
        if s.get('screener') == 'regime_transition':
            rt_map[s['ticker']] = s

    # ── LİSTE 1: AL/SAT Tarama ──
    # Sadece karar=AL (İZLE dahil değil), ERKEN hariç
    # Öncelik: ZAYIF (RS 20-60 + MACD>0 + Q≥50) > GUCLU (RS 30-60) > BILESEN (Q≥70 + MACD>0)
    alsat_items = []
    for s in latest_signals:
        if s.get('screener') != 'alsat':
            continue
        karar = s.get('karar', '')
        if karar != 'AL':
            continue
        sig_type = s.get('signal_type', '')
        if sig_type == 'ERKEN':
            continue
        q = s.get('quality', 0) or 0
        rs = s.get('rs_score', 0) or 0
        macd = s.get('macd', 0) or 0
        oe = s.get('oe', '')
        rr = s.get('rr', '')
        stop = s.get('stop_price', '')
        tp = s.get('target_price', '')

        # Kalite filtreleri — sadece kriterleri karşılayanlar listeye girer
        passes = False
        tier_label = ''
        if sig_type == 'ZAYIF' and 20 <= rs <= 60 and macd > 0 and q >= 50:
            passes = True
            tier_label = 'ZAYIF✓'
            score = 300 + q  # En yüksek öncelik
        elif sig_type in ('GUCLU', 'GÜÇLÜ') and 30 <= rs <= 60:
            passes = True
            tier_label = 'GÜÇLÜ✓'
            score = 200 + q
        elif sig_type in ('BILESEN', 'BİLEŞEN') and q >= 70 and macd > 0:
            passes = True
            tier_label = 'BİLEŞEN✓'
            score = 100 + q
        elif sig_type in ('CMB+', 'CMB'):
            passes = True
            tier_label = sig_type
            score = 400 + q  # CMB her zaman güçlü
        elif sig_type in ('DONUS', 'DÖNÜŞ') and q >= 70:
            passes = True
            tier_label = 'DÖNÜŞ'
            score = 50 + q
        # PB ve düşük kaliteli sinyaller dahil değil

        if not passes:
            continue

        reasons = [tier_label, f"Q={q}", f"RS={rs:.0f}", f"MACD={'+'if macd>0 else ''}{macd:.4f}"]
        if oe != '':
            reasons.append(f"OE={oe}")
        if rr != '':
            reasons.append(f"R:R={rr}")

        # ICE/SMS
        sm_val, sm_ic, sm_mult = _get_sm_info(s['ticker'], ice_results, sms_scores)
        if sm_val is not None:
            if sm_val >= 45:
                score += 5
            elif sm_val < 15:
                reasons.append("⚠️SM")

        alsat_items.append((s['ticker'], score, reasons, s))

    alsat_items.sort(key=lambda x: -x[1])

    # ── LİSTE 2: Tavan Tarayıcı ──
    # Tavan: skor + hacim düşüklüğü + CMF pozitif
    # Tavan kandidat: hacim yüksek olabilir, öncelik skor + CMF
    tavan_items = []
    tavan_tickers = {}
    for s in latest_signals:
        if s.get('screener') not in ('tavan', 'tavan_kandidat'):
            continue
        t = s['ticker']
        if t in tavan_tickers:
            if (s.get('skor', 0) or 0) <= (tavan_tickers[t].get('skor', 0) or 0):
                continue
        tavan_tickers[t] = s

    for ticker, s in tavan_tickers.items():
        skor = s.get('skor', 0) or 0
        vol = s.get('volume_ratio', 0) or 0
        seri = s.get('streak', 0) or 0
        rs = s.get('rs', 0) or 0
        is_kandidat = s.get('screener') == 'tavan_kandidat'

        # CMF cross-ref from RT
        cmf = None
        rt_sig = rt_map.get(ticker)
        if rt_sig:
            cmf = rt_sig.get('cmf')

        # Skor hesaplama
        score = skor * 10
        if is_kandidat:
            # Kandidat: hacim yüksek olabilir, öncelik skor + CMF
            if cmf is not None and cmf > 0:
                score += int(cmf * 100)  # CMF bonus
        else:
            # Tavan: düşük hacim bonus
            if vol < 1.0:
                score += 30  # Kilitli tavan
            elif vol < 1.5:
                score += 15
            # CMF pozitif bonus
            if cmf is not None and cmf > 0:
                score += int(cmf * 100)

        reasons = []
        if is_kandidat:
            reasons.append(f"KND skor:{skor}")
        elif skor >= 50 and vol < 1.0:
            reasons.append(f"🔒skor:{skor}")
        else:
            reasons.append(f"skor:{skor}")
        reasons.append(f"vol:{vol:.1f}x")
        if seri > 1:
            reasons.append(f"seri:{seri}")
        if rs != 0:
            reasons.append(f"RS{rs:+.0f}%")
        if cmf is not None:
            reasons.append(f"CMF{cmf:+.2f}")

        # ICE/SMS
        sm_val, sm_ic, sm_mult = _get_sm_info(ticker, ice_results, sms_scores)
        if sm_val is not None:
            if sm_val >= 45:
                score += 15
            elif sm_val < 15:
                reasons.append("⚠️SM")

        tavan_items.append((ticker, score, reasons, s))

    tavan_items.sort(key=lambda x: -x[1])

    # ── LİSTE 3: NW Pivot AL (Günlük) ──
    # SADECE bugünün yeni sinyalleri (fresh=BUGÜN) + gate=AÇIK
    # D+W overlap günceldir (bugünün daily + weekly overlap)
    nw_items = []
    for s in latest_signals:
        if s.get('screener') != 'nox_v3_daily':
            continue
        if s.get('direction') != 'AL':
            continue
        fresh = s.get('fresh', '')
        if fresh not in ('BUGUN', 'BUGÜN'):
            continue  # Sadece bugünün yeni sinyalleri
        if not s.get('gate'):
            continue  # Sadece gate açık (onaylı sinyal)
        delta = s.get('delta_pct')
        dw = s.get('dw_overlap', False)
        rs = s.get('rs_score')
        adx = s.get('adx')
        rsi = s.get('rsi')

        # Sıralama: D+W önce, sonra delta% (pivot yakınlığı)
        score = 30
        if dw:
            score += 50  # D+W en üstte
        # Delta düşük = pivota yakın = daha iyi
        if delta is not None:
            score += max(0, int(20 - delta))

        reasons = []
        if dw:
            reasons.append("⚡D+W")
        reasons.append("🔥BUGÜN")
        if delta is not None:
            reasons.append(f"δ{delta:.1f}%")
        if adx is not None:
            reasons.append(f"ADX={adx:.0f}")
        if rsi is not None:
            reasons.append(f"RSI={rsi:.0f}")
        if rs is not None:
            reasons.append(f"RS={rs:.1f}")

        # ICE/SMS
        sm_val, sm_ic, sm_mult = _get_sm_info(s['ticker'], ice_results, sms_scores)
        if sm_val is not None:
            if sm_val >= 45:
                score += 5
            elif sm_val < 15:
                reasons.append("⚠️SM")

        nw_items.append((s['ticker'], score, reasons, s))

    nw_items.sort(key=lambda x: -x[1])

    # ── LİSTE 4: Regime Transition ──
    # Tarih bugün olmalı, giriş ≥ 3, OE ≤ 2
    rt_items = []
    for s in latest_signals:
        if s.get('screener') != 'regime_transition' or s.get('direction') != 'AL':
            continue
        if not _is_today(s):
            continue  # Sadece bugünün sinyalleri
        window = s.get('entry_window', '')
        if window not in ('TAZE', '2.DALGA'):
            continue  # Sadece TAZE ve 2.DALGA — YAKIN/BEKLE/GEÇ dahil değil
        badge = s.get('badge', '')
        entry_score = int(s.get('quality', 0) or 0)
        if entry_score < 3:
            continue  # Giriş 3/4 veya 4/4 olmalı
        cmf = s.get('cmf', 0) or 0
        adx = s.get('adx', 0) or 0
        oe = int(s.get('oe', 0) or 0)
        if oe > 2:
            continue  # OE 0, 1, 2 olabilir — 3+ dahil değil

        # transition_date: rejim geçişinin gerçek tarihi
        t_date = s.get('transition_date', '')
        is_today_transition = (t_date == today_sd)

        # Sıralama
        score = 0
        if badge == 'H+PB':
            score += 100
        elif badge == 'H+AL':
            score += 80
        elif badge:
            score += 60
        score += entry_score * 10
        window_pts = {'TAZE': 20, '2.DALGA': 10, 'YAKIN': 5}.get(window, 0)
        score += window_pts
        # TAZE + aynı gün geçiş = en taze sinyal → bonus
        if window == 'TAZE' and is_today_transition:
            score += 15

        reasons = []
        if badge:
            reasons.append(f"🏅{badge}")
        reasons.append(window)
        # TAZE ise geçiş tarihini göster
        if window == 'TAZE' and t_date:
            short_date = t_date[5:].replace('-', '/')  # 03/17
            if is_today_transition:
                reasons.append("📍BUGÜN")
            else:
                reasons.append(f"📅{short_date}")
        reasons.append(f"F{entry_score}")
        if cmf != 0:
            reasons.append(f"CMF{cmf:+.2f}")
        if oe > 0:
            reasons.append(f"OE={oe}")
        if adx:
            reasons.append(f"ADX={adx:.0f}")

        # ICE/SMS
        sm_val, sm_ic, sm_mult = _get_sm_info(s['ticker'], ice_results, sms_scores)
        if sm_val is not None:
            if sm_val >= 45:
                score += 10
            elif sm_val < 15:
                reasons.append("⚠️SM")

        rt_items.append((s['ticker'], score, reasons, s))

    rt_items.sort(key=lambda x: -x[1])

    # ── Çapraz Çakışma Tagging ──
    list_data = {'alsat': alsat_items, 'tavan': tavan_items, 'nw': nw_items, 'rt': rt_items}
    _LIST_SHORT = {'alsat': 'AS', 'tavan': 'TVN', 'nw': 'NW', 'rt': 'RT'}
    ticker_list_count = {}
    for list_name, items in list_data.items():
        for ticker, _, _, _ in items:
            ticker_list_count.setdefault(ticker, set()).add(list_name)

    for list_name, items in list_data.items():
        for i, (ticker, score, reasons, sig) in enumerate(items):
            cnt = len(ticker_list_count.get(ticker, set()))
            if cnt >= 3:
                reasons.insert(0, f"⚡{cnt}LİSTE")
            elif cnt == 2:
                others = ticker_list_count[ticker] - {list_name}
                other_tags = "+".join(_LIST_SHORT.get(o, o) for o in sorted(others))
                reasons.insert(0, f"∩{other_tags}")

    # ── Tier 1: Çapraz çakışmalar (2+ listede) — kalite bazlı skor ──
    def _overlap_quality(ticker):
        """Normalize overlap quality: farklı kaynak kalitelerini eşit tartır."""
        quality = 0
        in_lists = []
        for list_name in ('alsat', 'tavan', 'nw', 'rt'):
            for t, sc, reas, sig in list_data[list_name]:
                if t != ticker:
                    continue
                in_lists.append(list_name)
                if list_name == 'rt':
                    badge = sig.get('badge', '')
                    if badge == 'H+PB':
                        quality += 50
                    elif badge == 'H+AL':
                        quality += 40
                    elif badge:
                        quality += 30
                    entry_s = int(sig.get('quality', 0) or 0)
                    quality += entry_s * 8  # F3=24, F4=32
                    if sig.get('entry_window') == 'TAZE':
                        quality += 10
                    cmf = sig.get('cmf', 0) or 0
                    if cmf > 0.1:
                        quality += 5
                elif list_name == 'nw':
                    if sig.get('dw_overlap'):
                        quality += 35  # D+W çok güçlü
                    else:
                        quality += 20
                    delta = sig.get('delta_pct')
                    if delta is not None and delta < 10:
                        quality += 10  # Pivota yakın
                elif list_name == 'alsat':
                    sig_type = sig.get('signal_type', '')
                    if sig_type in ('CMB', 'CMB+'):
                        quality += 35
                    elif sig_type in ('ZAYIF', 'GUCLU', 'GÜÇLÜ'):
                        quality += 30  # Kalite filtrelerini geçmiş
                    elif sig_type in ('BILESEN', 'BİLEŞEN'):
                        quality += 25
                    else:
                        quality += 10  # DÖNÜŞ
                    q = sig.get('quality', 0) or 0
                    quality += q // 10  # Q=100→10, Q=75→7
                elif list_name == 'tavan':
                    skor = sig.get('skor', 0) or 0
                    vol = sig.get('volume_ratio', 0) or 0
                    if skor >= 50 and vol < 1.0:
                        quality += 30  # Kilitli tavan
                    elif skor >= 50:
                        quality += 20
                    elif skor >= 40:
                        quality += 15
                    else:
                        quality += 8
                    # CMF cross-ref (tavan sinyalinde RT CMF varsa)
                    rt_s = rt_map.get(ticker)
                    if rt_s:
                        cmf_t = rt_s.get('cmf', 0) or 0
                        if cmf_t > 0.1:
                            quality += 5
                break
        # Çakışma çeşitliliği bonusu
        if len(in_lists) >= 3:
            quality += 20
        elif len(in_lists) >= 2:
            quality += 5
        # RT veya NW içeren çakışma daha anlamlı (farklı soru cevaplıyorlar)
        has_technical = bool({'alsat', 'tavan'} & set(in_lists))
        has_structural = bool({'nw', 'rt'} & set(in_lists))
        if has_technical and has_structural:
            quality += 15  # Teknik + yapısal çakışma bonusu
        return quality, in_lists

    tier1 = []
    for ticker, lists in ticker_list_count.items():
        if len(lists) < 2:
            continue
        quality, in_lists = _overlap_quality(ticker)
        # Kalite eşiği: sadece anlamlı çakışmalar
        # AS+TVN DÖNÜŞ düşük kalite = gürültü, minimum 40p gerekli
        if quality < 40:
            continue
        list_tags = "+".join(_LIST_SHORT.get(l, l) for l in sorted(in_lists))
        # Teknik+yapısal çakışma etiketi
        has_tech = bool({'alsat', 'tavan'} & set(in_lists))
        has_struct = bool({'nw', 'rt'} & set(in_lists))
        ty_tag = " 🔀T+Y" if has_tech and has_struct else ""
        reasons_all = [f"⚡{list_tags} [{quality}p]{ty_tag}"]
        for l in sorted(in_lists):
            for t, sc, reas, sig in list_data[l]:
                if t == ticker:
                    short_reason = f"[{_LIST_SHORT[l]}] {' '.join(reas[:4])}"
                    reasons_all.append(short_reason)
                    break
        tier1.append((ticker, quality, reasons_all, {
            'overlap_count': len(in_lists),
            'in_lists': in_lists,
        }))

    # ── Gevşek RT çakışma: güçlü sinyal + RT badge (F serbest) ──
    # Normal RT filtresi (F≥3 OE≤2) çok sıkı — güçlü AS/NW/TVN sinyali varsa
    # badge + TAZE/2.DALGA + OE≤3 yetsin, F serbest
    tier1_tickers = {t for t, _, _, _ in tier1}
    strong_tickers = {}  # ticker → (list_name, reasons, sig)
    # Güçlü AS sinyalleri (kalite filtrelerinden geçen)
    for t, sc, reas, sig in alsat_items:
        st = sig.get('signal_type', '')
        if st in ('ZAYIF', 'GUCLU', 'GÜÇLÜ', 'CMB', 'CMB+', 'BILESEN', 'BİLEŞEN'):
            strong_tickers[t] = ('alsat', reas, sig)
    # Kilitli tavan
    for t, sc, reas, sig in tavan_items:
        skor = sig.get('skor', 0) or 0
        vol = sig.get('volume_ratio', 0) or 0
        if skor >= 50 and vol < 1.0:
            strong_tickers.setdefault(t, ('tavan', reas, sig))
    # NW D+W
    for t, sc, reas, sig in nw_items:
        if sig.get('dw_overlap'):
            strong_tickers.setdefault(t, ('nw', reas, sig))

    # Tüm RT sinyallerinden gevşek tarama
    for s in latest_signals:
        if s.get('screener') != 'regime_transition' or s.get('direction') != 'AL':
            continue
        if not _is_today(s):
            continue
        ticker = s['ticker']
        if ticker in tier1_tickers:
            continue  # Zaten Tier 1'de
        if ticker not in strong_tickers:
            continue  # Güçlü karşı sinyal yok
        badge = s.get('badge', '')
        if not badge:
            continue  # Badge zorunlu
        window = s.get('entry_window', '')
        if window not in ('TAZE', '2.DALGA'):
            continue  # Window hala gerekli
        oe = int(s.get('oe', 0) or 0)
        if oe > 3:
            continue  # OE≤3 (gevşetilmiş)
        entry_score = int(s.get('quality', 0) or 0)
        cmf = s.get('cmf', 0) or 0

        # Kalite skoru
        src_name, src_reas, src_sig = strong_tickers[ticker]
        quality, _ = _overlap_quality(ticker)  # Base quality from lists
        # RT badge kalitesi ekle (gevşek giriş olduğu için biraz düşür)
        rt_quality = 20  # Base for relaxed badge
        if badge == 'H+PB':
            rt_quality += 15
        if window == 'TAZE':
            rt_quality += 5
        if cmf > 0.1:
            rt_quality += 5
        quality += rt_quality

        src_tag = _LIST_SHORT[src_name]
        src_short = ' '.join(src_reas[:3])
        # Transition date for RT↓
        t_date_r = s.get('transition_date', '')
        date_tag = ""
        if window == 'TAZE' and t_date_r:
            if t_date_r == today_sd:
                date_tag = " 📍BUGÜN"
            else:
                date_tag = f" 📅{t_date_r[5:].replace('-', '/')}"
        rt_reasons = f"🏅{badge} {window}{date_tag} F{entry_score} OE={oe} CMF{cmf:+.2f}"
        # Teknik+yapısal çakışma etiketi
        is_structural = True  # RT = yapısal
        is_technical = src_name in ('alsat', 'tavan')
        ty_tag = " 🔀T+Y" if is_technical and is_structural else ""
        reasons_all = [
            f"⚡{src_tag}+RT [{quality}p]{ty_tag}",
            f"[{src_tag}] {src_short}",
            f"[RT↓] {rt_reasons}",  # ↓ = gevşek filtre
        ]
        tier1.append((ticker, quality, reasons_all, {
            'overlap_count': 2,
            'in_lists': [src_name, 'rt'],
            'relaxed': True,
        }))
        tier1_tickers.add(ticker)

    tier1.sort(key=lambda x: -x[1])

    # ── Tier 2: Her listeden en kaliteli tekil hisseler ──
    tier1_tickers = {t for t, _, _, _ in tier1}
    tier2 = []
    for list_name in ('nw', 'rt', 'alsat', 'tavan'):  # NW/RT önce (daha anlamlı)
        items = list_data[list_name]
        count = 0
        for ticker, score, reasons, sig in items:
            if ticker in tier1_tickers:
                continue
            if ticker in {t for t, _, _, _ in tier2}:
                continue
            tag = _LIST_SHORT[list_name]
            tier2.append((ticker, score, [f"[{tag}]"] + reasons, sig))
            count += 1
            if count >= 5:
                break
    # Tier 2'yi liste etiketine göre grupla ama sıra koru
    # (NW → RT → AS → TVN sırasıyla zaten eklenmiş)

    result = dict(list_data)
    result['tier1'] = tier1
    result['tier2'] = tier2
    return result


def _push_priority_tickers(lists_dict):
    """4 listeden tüm unique ticker'ları priority_tickers.json olarak GitHub Pages'e push et.
    VDS takas scraper bu dosyayı okuyarak hangi hisseler için veri çekeceğini bilir."""
    import json
    seen = set()
    tickers = []
    for list_name in ('tier1', 'tier2', 'alsat', 'tavan', 'nw', 'rt'):
        for item in lists_dict.get(list_name, []):
            ticker = item[0]
            if ticker not in seen:
                seen.add(ticker)
                tickers.append(ticker)

    payload = json.dumps({
        "updated_at": datetime.now(_TZ_TR).strftime("%Y-%m-%dT%H:%M:%S+03:00"),
        "total": len(tickers),
        "tickers": tickers
    }, ensure_ascii=False)

    url = push_html_to_github(payload, 'priority_tickers.json',
                               datetime.now(_TZ_TR).strftime('%Y%m%d'))
    if url:
        print(f"  ✅ Priority tickers push OK ({len(tickers)} hisse)")
    else:
        print(f"  ⚠️ Priority tickers push başarısız")


def _fmt_lot(lot):
    """Lot sayısını kompakt formatla: 1500000 → +1.5M, -300000 → -300K."""
    if abs(lot) >= 1_000_000:
        return f"{lot / 1_000_000:+.1f}M"
    elif abs(lot) >= 1000:
        return f"{lot / 1000:+.0f}K"
    elif lot != 0:
        return f"{lot:+d}"
    return "0"



def _build_sm_inline(ticker, takas_data, mkk_data, sms_scores, ice_results=None):
    """Tek hisse için kompakt SM satırı (sinyal satırının altına eklenir).

    ICE varsa: 💰 ×1.15🟢 T=var B=guc KV=dest | YB+7.8M F+7.0M
    SMS fallback: 💰 🟢68 K%50↓2.8 | YB+7.8M F+7.0M [F:YAT.FONLARI]
    """
    parts = []

    # ICE öncelikli, SMS fallback
    ice_shown = False
    if ice_results:
        ice = ice_results.get(ticker)
        if ice:
            ice_shown = True
            t = ice.labels.get("kurumsal_teyit")
            b = ice.labels.get("tasinan_birikim")
            kv = ice.labels.get("kisa_vade")
            t_s = t.value[:3] if t else "?"
            b_s = b.value[:3] if b else "?"
            kv_s = kv.value[:4] if kv else "?"
            dt20 = ice.metrics.get("takas_20_change")
            dt_str = f" ΔT20={dt20:+,.0f}" if dt20 is not None else ""
            parts.append(f"×{ice.multiplier:.2f}{ice.icon} T={t_s} B={b_s} KV={kv_s}{dt_str}")

    if not ice_shown and sms_scores:
        sms = sms_scores.get(ticker)
        if sms:
            sv = sms.score if hasattr(sms, 'score') else sms
            parts.append(f"{sms_icon(sv)}{sv}")

    # MKK: kurumsal % + değişim
    if mkk_data:
        mkk = mkk_data.get(ticker)
        if mkk:
            k_pct = mkk.get('kurumsal_pct', 0)
            fark_5g = mkk.get('bireysel_fark_5g')
            fark_1g = mkk.get('bireysel_fark_1g')
            if fark_5g is not None:
                k_chg = -fark_5g
                arrow = "↑" if k_chg > 0.2 else ("↓" if k_chg < -0.2 else "→")
                parts.append(f"K%{k_pct:.0f}{arrow}{abs(k_chg):.1f}")
            elif fark_1g is not None:
                k_chg = -fark_1g
                arrow = "↑" if k_chg > 0.1 else ("↓" if k_chg < -0.1 else "→")
                parts.append(f"K%{k_pct:.0f}{arrow}{abs(k_chg):.1f}")
            else:
                parts.append(f"K%{k_pct:.0f}")

    # Takas: net akış by tip (haftalık)
    if takas_data:
        td = takas_data.get(ticker)
        if td:
            kurumlar = td.get('kurumlar', [])
            net_by_tip = {}
            top_buyer_name = None
            top_buyer_lot = 0
            for k in kurumlar:
                name = k.get('Aracı Kurum') or k.get('kurum') or ''
                h_lot = k.get('Haftalık Fark') or k.get('haftalik_fark') or 0
                tip = classify_kurum_sms(name)
                net_by_tip[tip] = net_by_tip.get(tip, 0) + h_lot
                if h_lot > top_buyer_lot:
                    top_buyer_name = name
                    top_buyer_lot = h_lot

            flow_parts = []
            for tip, short in [('yab_banka', 'YB'), ('fon', 'F'), ('prop', 'P')]:
                net = net_by_tip.get(tip, 0)
                if abs(net) >= 1000:
                    flow_parts.append(f"{short}{_fmt_lot(net)}")
            if flow_parts:
                parts.append("| " + " ".join(flow_parts))

            if top_buyer_name and top_buyer_lot > 0:
                tip = classify_kurum_sms(top_buyer_name)
                _TIP_SHORT = {'yab_banka': 'YB', 'fon': 'F', 'prop': 'P',
                              'yerli_banka': 'YrB', 'diger': ''}
                tip_tag = _TIP_SHORT.get(tip, '')
                words = top_buyer_name.split()[:2]
                short_name = " ".join(words)[:15]
                parts.append(f"[{tip_tag}:{short_name}]" if tip_tag else f"[{short_name}]")

    if parts:
        return "   💰 " + " ".join(parts)
    return None


def _build_shortlist_message(lists_dict, portfolio=None,
                             sms_scores=None, takas_data=None, mkk_data=None,
                             ice_results=None):
    """4 kaynak bazlı shortlist'i Telegram mesajı olarak formatla.
    Her hissenin altında kompakt SM (takas+MKK) satırı gösterir."""
    now = datetime.now(_TZ_TR)
    held = set()
    watched = set()
    if portfolio:
        held = {h['ticker'] for h in portfolio.get('holdings', [])}
        watched = set(portfolio.get('watchlist', []))

    def _tag(ticker):
        if ticker in held:
            return " 📌PF"
        elif ticker in watched:
            return " 👁️İL"
        return ""

    has_sm = bool(takas_data or mkk_data or ice_results)

    lines = [
        f"<b>⬡ NOX Ön Analiz — {now.strftime('%d.%m.%Y %H:%M')}</b>",
    ]

    # ── 1. AL/SAT Tarama ──
    alsat_list = lists_dict.get('alsat', [])
    if alsat_list:
        lines.append("")
        lines.append(f"📋 <b>1. AL/SAT Tarama</b> (Q skoru sıralı, {len(alsat_list)} sinyal)")
        lines.append("")
        for i, (ticker, score, reasons, sig) in enumerate(alsat_list[:8], 1):
            reasons_str = " ".join(reasons)
            lines.append(f"{i}. <b>{ticker}</b>{_tag(ticker)} {reasons_str}")
            if has_sm:
                sm_line = _build_sm_inline(ticker, takas_data, mkk_data, sms_scores, ice_results)
                if sm_line:
                    lines.append(sm_line)

    # ── 2. Tavan Tarayıcı ──
    tavan_list = lists_dict.get('tavan', [])
    if tavan_list:
        lines.append("")
        lines.append(f"🔺 <b>2. Tavan Tarayıcı</b> (skor/vol sıralı, {len(tavan_list)} hisse)")
        lines.append("")
        for i, (ticker, score, reasons, sig) in enumerate(tavan_list[:8], 1):
            reasons_str = " ".join(reasons)
            lines.append(f"{i}. <b>{ticker}</b>{_tag(ticker)} {reasons_str}")
            if has_sm:
                sm_line = _build_sm_inline(ticker, takas_data, mkk_data, sms_scores, ice_results)
                if sm_line:
                    lines.append(sm_line)

    # ── 3. NW Pivot AL ──
    nw_list = lists_dict.get('nw', [])
    if nw_list:
        lines.append("")
        lines.append(f"📊 <b>3. NW Pivot AL</b> (günlük gate açık, {len(nw_list)} sinyal)")
        lines.append("")
        for i, (ticker, score, reasons, sig) in enumerate(nw_list[:10], 1):
            reasons_str = " ".join(reasons)
            lines.append(f"{i}. <b>{ticker}</b>{_tag(ticker)} {reasons_str}")
            if has_sm:
                sm_line = _build_sm_inline(ticker, takas_data, mkk_data, sms_scores, ice_results)
                if sm_line:
                    lines.append(sm_line)

    # ── 4. Regime Transition ──
    rt_list = lists_dict.get('rt', [])
    if rt_list:
        lines.append("")
        lines.append(f"⚡ <b>4. Regime Transition</b> (F≥3 OE≤2, {len(rt_list)} sinyal)")
        lines.append("")
        for i, (ticker, score, reasons, sig) in enumerate(rt_list[:10], 1):
            reasons_str = " ".join(reasons)
            lines.append(f"{i}. <b>{ticker}</b>{_tag(ticker)} {reasons_str}")
            if has_sm:
                sm_line = _build_sm_inline(ticker, takas_data, mkk_data, sms_scores, ice_results)
                if sm_line:
                    lines.append(sm_line)

    # ── Tier 1: Çapraz Çakışmalar ──
    tier1 = lists_dict.get('tier1', [])
    if tier1:
        lines.append("")
        lines.append(f"🔥 <b>Tier 1 — Çakışmalar</b> ({len(tier1)} hisse)")
        lines.append("")
        for i, (ticker, score, reasons, _) in enumerate(tier1[:10], 1):
            reasons_str = " | ".join(reasons)
            lines.append(f"{i}. <b>{ticker}</b>{_tag(ticker)} {reasons_str}")
            if has_sm:
                sm_line = _build_sm_inline(ticker, takas_data, mkk_data, sms_scores, ice_results)
                if sm_line:
                    lines.append(sm_line)

    # ── Tier 2: En Kaliteli Tekil ──
    tier2 = lists_dict.get('tier2', [])
    if tier2:
        lines.append("")
        lines.append(f"⭐ <b>Tier 2 — Tekil Kalite</b> ({len(tier2)} hisse)")
        lines.append("")
        for i, (ticker, score, reasons, _) in enumerate(tier2[:10], 1):
            reasons_str = " ".join(reasons)
            lines.append(f"{i}. <b>{ticker}</b>{_tag(ticker)} {reasons_str}")
            if has_sm:
                sm_line = _build_sm_inline(ticker, takas_data, mkk_data, sms_scores, ice_results)
                if sm_line:
                    lines.append(sm_line)

    return "\n".join(lines)


def run_briefing(notify=False, use_ai=True, fresh=False, shortlist_only=False):
    """Ana brifing pipeline.

    shortlist_only=True: Sadece öncelikli hisse listesi oluştur + takas iste.
    """
    now = datetime.now(_TZ_TR)
    print(f"\n{'='*50}")
    print(f"⬡ NOX Brifing — {now.strftime('%d.%m.%Y %H:%M')}")
    print(f"{'='*50}\n")

    # 0. Fresh mode — scanner'ları önce çalıştır
    if fresh:
        print("🔄 Fresh mode — scanner'lar çalıştırılıyor...")
        _run_fresh_scanners()
        print()

    # 1. Scanner sinyallerini HTML raporlardan yükle (tek kaynak)
    print("📋 Scanner sinyalleri yükleniyor (HTML)...")
    latest_signals = fetch_all_html_signals()
    if not latest_signals:
        msg = "⚠️ Hiç sinyal bulunamadı — brifing üretilemiyor."
        print(msg)
        if notify:
            send_telegram(msg)
        return

    signal_summary = summarize_signals(latest_signals)

    # 1b. Sinyalleri JSON olarak export et + GitHub Pages'e push
    if notify and latest_signals:
        json_path = os.path.join(ROOT, 'output', 'latest_signals.json')
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        export_signals_json(latest_signals, json_path)
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                json_content = f.read()
            push_html_to_github(json_content, 'latest_signals.json',
                                now.strftime('%Y%m%d'))
        except Exception as e:
            print(f"  ⚠️ Sinyal JSON push hatası: {e}")

    # 2. MKK verisi (VDS auto-fetch — çakışma skorunda kullanılır)
    mkk_data_map = None
    try:
        mkk_json = fetch_mkk_data()
        if mkk_json and mkk_json.get('data'):
            mkk_data_map = mkk_json['data']
            print(f"  MKK: {len(mkk_data_map)} hisse (VDS/{mkk_json.get('extracted_at', '?')[:10]})")
    except Exception as e:
        print(f"  ⚠️ MKK veri hatası: {e}")

    # 3. Çakışma analizi (makro olmadan da çalışır)
    print("\n⬡ Çakışma analizi...")
    confluence_results = calc_all_confluence(
        latest_signals, None, min_score=1, mkk_data=mkk_data_map)
    print(f"  {len(confluence_results)} hisse çakışma skoru ≥ 1")

    # 2b. SMS hesapla (takas verisi varsa)
    sms_scores = None
    takas_data_map = None
    try:
        takas_json_sms = fetch_takas_data()
        if takas_json_sms and takas_json_sms.get('data'):
            takas_data_map = takas_json_sms['data']
            sms_scores = calc_batch_sms(None, takas_data_map, mkk_data_map)
            guclu = sum(1 for s in sms_scores.values() if s.score >= 45)
            dagitim = sum(1 for s in sms_scores.values() if s.score < 15)
            print(f"  SMS: {len(sms_scores)} hisse ({guclu}🟢 güçlü, {dagitim}🔴 dağıtım)")
    except Exception as e:
        print(f"  ⚠️ SMS hesaplama hatası: {e}")

    # 2c. ICE hesapla (takas history varsa — SMS'e öncelikli)
    ice_results = None
    try:
        takas_history = fetch_takas_history()
        if takas_history:
            signal_tickers = list({s['ticker'] for s in latest_signals})
            ice_results = calc_batch_ice(
                signal_tickers, takas_history, takas_data_map, mkk_data_map)
            if ice_results:
                guclu_ice = sum(1 for r in ice_results.values() if r.multiplier >= 1.15)
                red_ice = sum(1 for r in ice_results.values() if r.multiplier < 0.90)
                partial = sum(1 for r in ice_results.values() if r.status == "partial")
                print(f"  ICE: {len(ice_results)} hisse ({guclu_ice} güçlü teyit, {red_ice} dağıtım riski"
                      f"{f', {partial} kısmi veri' if partial else ''})")
                # ICE/SMS fallback loglama
                ice_tickers = set(ice_results.keys())
                sms_only = set((sms_scores or {}).keys()) - ice_tickers
                sms_fallback = len(sms_only & set(signal_tickers))
                if sms_fallback:
                    print(f"  ICE aktif: {len(ice_tickers)}/{len(signal_tickers)} | SMS fallback: {sms_fallback}")
                # A/B karşılaştırma (ICE vs SMS, kalibrasyon)
                if sms_scores and ice_results:
                    diffs = []
                    for t in sorted(ice_results.keys())[:10]:
                        sms = sms_scores.get(t)
                        ice_r = ice_results[t]
                        if sms:
                            sms_l = sms.label[0] if hasattr(sms, 'label') else '?'
                            ice_l = ice_r.label[0]
                            if sms_l != ice_l:
                                diffs.append(f"{t}:SMS={sms.score}{sms_l}/ICE={ice_r.score_100}{ice_l}")
                    if diffs:
                        print(f"  A/B fark: {' | '.join(diffs[:8])}")
    except Exception as e:
        print(f"  ⚠️ ICE hesaplama hatası: {e}")

    # ── SHORTLIST MODE: sadece öncelikli hisse listesi oluştur ──
    if shortlist_only:
        print("\n📋 4 kaynak bazlı öncelikli liste oluşturuluyor...")
        lists_dict = _compute_4_lists(
            latest_signals, confluence_results, sms_scores, ice_results)
        portfolio = _load_portfolio()

        _LIST_LABELS = [
            ('alsat', '📋 AL/SAT Tarama'),
            ('tavan', '🔺 Tavan Tarayıcı'),
            ('nw', '📊 NW Pivot AL (günlük gate açık)'),
            ('rt', '⚡ Regime Transition (F≥3 OE≤2)'),
        ]
        core_total = sum(len(lists_dict.get(k, [])) for k in ('alsat', 'tavan', 'nw', 'rt'))
        print(f"  Toplam: {core_total} sinyal (4 liste)\n")
        for key, label in _LIST_LABELS:
            items = lists_dict.get(key, [])
            if not items:
                print(f"  ── {label} (0) ──\n")
                continue
            limit = 10 if key in ('nw', 'rt') else 8
            print(f"  ── {label} ({len(items)}) ──")
            for i, (ticker, score, reasons, _sig) in enumerate(items[:limit], 1):
                print(f"  {i:2d}. {ticker:6s} [{score:3d}p] — {' '.join(reasons[:6])}")
            print()

        # Tier 1 + Tier 2
        tier1 = lists_dict.get('tier1', [])
        tier2 = lists_dict.get('tier2', [])
        if tier1:
            print(f"  ── 🔥 Tier 1: Çakışmalar ({len(tier1)}) ──")
            for i, (ticker, score, reasons, _) in enumerate(tier1[:10], 1):
                print(f"  {i:2d}. {ticker:6s} [{score:3d}p] — {' | '.join(reasons[:4])}")
            print()
        if tier2:
            print(f"  ── ⭐ Tier 2: Tekil Kalite ({len(tier2)}) ──")
            for i, (ticker, score, reasons, _) in enumerate(tier2[:10], 1):
                print(f"  {i:2d}. {ticker:6s} [{score:3d}p] — {' '.join(reasons[:5])}")
            print()
        if notify:
            msg = _build_shortlist_message(lists_dict, portfolio,
                                            sms_scores, takas_data_map, mkk_data_map,
                                            ice_results)
            send_telegram(msg)
            print(f"  ✅ Shortlist Telegram'a gönderildi")
            _push_priority_tickers(lists_dict)
        return {"lists": lists_dict}

    # 3. Makro veri çek
    print("\n🌍 Makro veri çekiliyor...")
    category_regimes = None
    try:
        macro_data = fetch_macro_data()
        snapshot = fetch_macro_snapshot()
        macro_result = assess_macro_regime(snapshot)
        category_regimes = calc_category_regimes(macro_data)
        macro_result['category_regimes'] = category_regimes
        print(f"  Rejim: {macro_result['regime']} (skor: {macro_result['risk_score']})")
        if category_regimes:
            cats = ", ".join(f"{c}={v['regime']}" for c, v in category_regimes.items())
            print(f"  Kategori: {cats}")
    except Exception as e:
        print(f"  ⚠️ Makro veri hatası: {e}")
        macro_result = None

    # Çakışmayı makro + SMS + ICE ile tekrar hesapla
    confluence_results = calc_all_confluence(
        latest_signals, macro_result, min_score=1, mkk_data=mkk_data_map,
        sms_scores=sms_scores, ice_results=ice_results)

    # 4. Claude ile brifing üret
    briefing_text = ""
    if use_ai:
        print("\n🤖 Claude ile brifing üretiliyor...")
        try:
            briefing_text = _generate_ai_briefing(
                signal_summary, macro_result, confluence_results, latest_signals)
            print(f"  ✅ Brifing üretildi ({len(briefing_text)} karakter)")
        except Exception as e:
            print(f"  ⚠️ Claude API hatası: {e}")
            briefing_text = _generate_fallback_briefing(
                signal_summary, macro_result, confluence_results)
    else:
        briefing_text = _generate_fallback_briefing(
            signal_summary, macro_result, confluence_results)

    # 4b. Shortlist listeleri + haberler (HTML rapor için)
    print("\n📋 4 liste + haberler hesaplanıyor...")
    lists_dict = _compute_4_lists(
        latest_signals, confluence_results, sms_scores, ice_results)
    news_items = fetch_market_news()
    if news_items:
        print(f"  📰 {len(news_items)} haber çekildi")

    # 5. HTML rapor oluştur
    print("\n📊 HTML rapor oluşturuluyor...")
    html = generate_briefing_html(
        briefing_text, macro_result, confluence_results, signal_summary,
        lists_dict=lists_dict, news_items=news_items)
    date_str = now.strftime('%Y%m%d')
    html_path = os.path.join(ROOT, 'output', f'nox_briefing_{date_str}.html')
    os.makedirs(os.path.dirname(html_path), exist_ok=True)
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"  ✅ {html_path}")

    # 6. Telegram + GitHub Pages
    html_url = None
    if notify:
        print("\n📤 Yayınlanıyor...")
        html_url = push_html_to_github(
            html, f'nox_briefing_{date_str}.html', date_str)

        tg_msg = _build_telegram_message(
            briefing_text, macro_result, confluence_results, html_url)
        send_telegram(tg_msg)
        print("  ✅ Telegram mesajı gönderildi")
    else:
        print(f"\n{'─'*50}")
        print(briefing_text)
        print(f"{'─'*50}")

    return {
        "briefing": briefing_text,
        "macro": macro_result,
        "confluence": confluence_results,
        "signal_summary": signal_summary,
        "html_path": html_path,
        "html_url": html_url,
    }


def _generate_ai_briefing(signal_summary, macro_result, confluence_results,
                           latest_signals=None):
    """Claude API ile brifing üret. Section 10 formatında."""
    from agent.claude_client import single_prompt

    data_context = []
    data_context.append(f"Tarih: {datetime.now(_TZ_TR).strftime('%d.%m.%Y')}")

    # ── Portföy bilgisi ──
    portfolio = _load_portfolio()
    if portfolio:
        holdings = portfolio.get('holdings', [])
        watchlist = portfolio.get('watchlist', [])
        tickers_held = [h['ticker'] for h in holdings]
        data_context.append(f"\n## 💼 PORTFÖY ({len(holdings)} hisse, güncelleme: {portfolio.get('updated', '?')})")
        for h in holdings:
            note = f" — {h['note']}" if h.get('note') else ""
            data_context.append(f"  {h['ticker']}{note}")
        if watchlist:
            data_context.append(f"  İzleme listesi: {', '.join(watchlist)}")
        data_context.append("  ⚠️ Portföydeki hisseler için: SAT sinyali varsa UYAR, AL sinyali varsa 'ZATEN PORTFÖYDE' not ekle.")

    # ── Günlük takas/kademe input ──
    daily = _load_daily_input()
    if daily:
        data_context.append(f"\n## 📋 GÜNLÜK TAKAS/KADEME VERİSİ (kaynak: {daily['file']})")
        data_context.append(daily['content'])

    # ── Veri tarihi uyarıları ──
    if latest_signals:
        today_str = datetime.now(_TZ_TR).strftime('%Y%m%d')
        screener_dates = {}
        for s in latest_signals:
            scr = s['screener']
            d = s.get('csv_date', '')
            if scr not in screener_dates or d > screener_dates[scr]:
                screener_dates[scr] = d
        stale = []
        for scr, d in screener_dates.items():
            if d and d < today_str:
                days_old = (datetime.strptime(today_str, '%Y%m%d') -
                           datetime.strptime(d, '%Y%m%d')).days
                name = SCREENER_NAMES.get(scr, scr)
                if days_old > 5:
                    stale.append(f"  ⚠️ {name}: {d[:4]}-{d[4:6]}-{d[6:]} ({days_old} gün eski!)")
                else:
                    data_context.append(f"  {name}: {d[:4]}-{d[4:6]}-{d[6:]} ({days_old} gün)")
        if stale:
            data_context.append("\n## ⚠️ ESKİ VERİ UYARISI — bu screener'lar güncel değil:")
            data_context.extend(stale)
            data_context.append("  Bu screener verilerinin çakışma sonuçlarını düşük güvenle değerlendir.")

    # ── Makro detaylı veri ──
    if macro_result:
        data_context.append(f"\n## Makro Rejim: {macro_result['regime']} (risk skor: {macro_result['risk_score']})")
        data_context.append("Rejim sinyalleri:")
        for sig in macro_result.get('signals', []):
            data_context.append(f"  {sig}")

        data_context.append("\nEnstrüman detayları:")
        for s in macro_result.get('snapshot', []):
            if s.get('price') is None:
                continue
            chg_1d = f"{s['chg_1d']:+.1f}%" if s.get('chg_1d') is not None else "?"
            chg_5d = f"{s['chg_5d']:+.1f}%" if s.get('chg_5d') is not None else "?"
            trend = s.get('trend', '?')
            data_context.append(
                f"  {s['name']}: {s['price']:,.2f} (1G:{chg_1d} 5G:{chg_5d} trend:{trend})")

    # ── Sinyal özeti (screener bazında) ──
    data_context.append(f"\n## Scanner Sinyalleri: {signal_summary.get('total', 0)} toplam")
    for scr, stats in signal_summary.get('screeners', {}).items():
        name = SCREENER_NAMES.get(scr, scr)
        data_context.append(f"  {name}: {stats['total']} ({stats.get('AL', 0)} AL, {stats.get('SAT', 0)} SAT)")

    # ── Öne çıkan sinyaller ──
    if latest_signals:
        nw_al_tickers = {s['ticker'] for s in latest_signals
                         if s.get('screener') == 'nox_v3_weekly'
                         and s.get('direction') == 'AL'
                         and s.get('signal_type') == 'PIVOT_AL'}
        rt_al_map = {}  # ticker → RT sinyal
        for s in latest_signals:
            if s.get('screener') == 'regime_transition' and s.get('direction') == 'AL':
                rt_al_map[s['ticker']] = s

        # ── BADGE: önce CSV'den (native), sonra çapraz hesaplama (fallback) ──
        badge_taze = []   # TAZE window
        badge_bekle = []  # BEKLE/2.DALGA/GEÇ window
        badge_seen = set()

        def _format_badge(ticker, badge_type, rt_sig, nw_sig=None):
            """Badge satırı formatla — sadece bilinen alanlar."""
            parts = [f"  {ticker}: {badge_type}"]
            window = rt_sig.get('entry_window', '') if rt_sig else ''
            if window:
                parts.append(f"[{window}]")
            # NW bilgileri (varsa)
            if nw_sig:
                wl = nw_sig.get('wl_status')
                if wl:
                    parts.append(f"WL={wl}")
                delta = nw_sig.get('delta_pct')
                if delta is not None:
                    parts.append(f"δ={delta:.1f}%")
            # RT bilgileri
            cmf = rt_sig.get('cmf') if rt_sig else None
            if cmf is not None:
                parts.append(f"CMF={cmf:+.3f}")
            adx = rt_sig.get('adx') if rt_sig else None
            if adx is not None:
                parts.append(f"ADX={adx:.0f}")
            oe = rt_sig.get('oe') if rt_sig else None
            if oe:
                parts.append(f"OE={oe}")
            return ' '.join(parts)

        # 1. CSV'den gelen native badge
        for s in latest_signals:
            if s.get('screener') == 'regime_transition' and s.get('badge'):
                ticker = s['ticker']
                badge_seen.add(ticker)
                nw_sig = next((ns for ns in latest_signals
                               if ns.get('screener') == 'nox_v3_weekly'
                               and ns['ticker'] == ticker
                               and ns.get('signal_type') == 'PIVOT_AL'), None)
                entry = _format_badge(ticker, s['badge'], s, nw_sig)
                window = s.get('entry_window', '')
                if window in ('TAZE', ''):
                    badge_taze.append(entry)
                else:
                    badge_bekle.append(entry)

        # 2. Çapraz badge fallback (NW AL ∩ RT AL)
        for ticker in nw_al_tickers & set(rt_al_map.keys()):
            if ticker in badge_seen:
                continue
            rt = rt_al_map[ticker]
            window = rt.get('entry_window', '')
            badge_type = 'H+AL' if window == 'TAZE' else ('H+PB' if window == '2.DALGA' else 'H+RT')
            nw_sig = next((ns for ns in latest_signals
                           if ns.get('screener') == 'nox_v3_weekly'
                           and ns['ticker'] == ticker
                           and ns.get('signal_type') == 'PIVOT_AL'), None)
            badge_seen.add(ticker)
            entry = _format_badge(ticker, badge_type, rt, nw_sig)
            if window in ('TAZE', ''):
                badge_taze.append(entry)
            else:
                badge_bekle.append(entry)

        if badge_taze or badge_bekle:
            data_context.append(f"\n## ⭐ BADGE Sinyalleri ({len(badge_taze)} TAZE + {len(badge_bekle)} diğer)")
            data_context.append("  H+AL = haftalık pivot AL + günlük geçiş, H+PB = pivot AL + pullback")
            if badge_taze:
                data_context.append("  -- TAZE (giriş penceresi açık):")
                data_context.extend(badge_taze)
            if badge_bekle:
                data_context.append("  -- BEKLE/2.DALGA/GEÇ (pencere henüz uygun değil):")
                data_context.extend(badge_bekle)

        # ── NW HAZIR + tetikli = ACİL SİNYAL ──
        wl_wr = {'HAZIR': '%77.8', 'İZLE': '%68.9', 'BEKLE': '%59.7'}
        nw_al_signals = [s for s in latest_signals
                         if s.get('screener') == 'nox_v3_weekly'
                         and s.get('direction') == 'AL'
                         and s.get('signal_type') == 'PIVOT_AL']
        # WL önceliğine göre sırala (HAZIR > İZLE > BEKLE)
        wl_order = {'HAZIR': 0, 'İZLE': 1, 'BEKLE': 2}
        nw_al_signals.sort(key=lambda s: (wl_order.get(s.get('wl_status', 'BEKLE'), 3),
                                           s.get('delta_pct') or 999))

        acil = [s for s in nw_al_signals
                if s.get('wl_status') == 'HAZIR' and s.get('trigger_type')]
        if acil:
            data_context.append("\n## 🚨 ACİL SİNYAL — NW HAZIR + Tetikli (WR %77.8):")
            for s in acil:
                delta = s.get('delta_pct')
                delta_str = f" δ={delta:.1f}%" if delta is not None else ""
                data_context.append(
                    f"  {s['ticker']}: HAZIR tetik={s.get('trigger_type')}{delta_str}")

        # ── NW Tüm AL sinyalleri (WL grubuna göre) ──
        if nw_al_signals:
            data_context.append(f"\n## NW Haftalık Pivot AL ({len(nw_al_signals)} tetikli sinyal):")
            data_context.append("  WR referans: HAZIR=%77.8, İZLE=%68.9, BEKLE=%59.7")
            for wl_grp in ['HAZIR', 'İZLE', 'BEKLE']:
                grp = [s for s in nw_al_signals if s.get('wl_status') == wl_grp]
                if not grp:
                    continue
                data_context.append(f"  [{wl_grp}] (WR {wl_wr.get(wl_grp, '?')}, {len(grp)} sinyal):")
                for s in grp[:15]:
                    delta = s.get('delta_pct')
                    delta_str = f"δ={delta:.1f}%" if delta is not None else ""
                    trigger = s.get('trigger_type', '-')
                    rs = s.get('rs_score')
                    rs_str = f" RS={rs:.1f}" if rs is not None else ""
                    data_context.append(
                        f"    {s['ticker']}: tetik={trigger} {delta_str}{rs_str}")

        # ── AL/SAT DÖNÜŞ sinyalleri ──
        donus_signals = [s for s in latest_signals
                         if s.get('screener') == 'alsat'
                         and s.get('signal_type') == 'DÖNÜŞ'
                         and s.get('karar') == 'AL']
        if donus_signals:
            data_context.append("\n## AL/SAT DÖNÜŞ Sinyalleri (En Yüksek WR):")
            for s in donus_signals[:15]:
                q = s.get('quality', '?')
                oe = s.get('oe', '?')
                data_context.append(
                    f"  {s['ticker']}: DÖNÜŞ Q={q} OE={oe}")

        # ── Tavan sinyalleri (detaylı) ──
        tavan_signals = [s for s in latest_signals
                         if s.get('screener') == 'tavan'
                         and s.get('direction') == 'AL']
        if tavan_signals:
            # Skora göre sırala
            tavan_signals.sort(key=lambda s: -(s.get('skor', 0) or 0))
            data_context.append("\n## Tavan Sinyalleri (skor≥50 + vol<1.0x = KİLİTLİ TAVAN — WR %84.6):")
            for s in tavan_signals[:15]:
                skor = s.get('skor', 0)
                streak = s.get('streak', 0)
                vol = s.get('volume_ratio', 0)
                yab = s.get('yabanci_degisim', 0)
                kilitli = "🔒 KİLİTLİ" if skor >= 50 and vol < 1.0 else ""
                yab_str = f" yab={yab:+.2f}%" if yab else ""
                data_context.append(
                    f"  {s['ticker']}: skor={skor} streak={streak} vol={vol:.1f}x{yab_str} {kilitli}")

        # ── RT AL sinyalleri (badge, fırsat, CMF, ADX dahil) ──
        rt_al_signals = [s for s in latest_signals
                         if s.get('screener') == 'regime_transition'
                         and s.get('direction') == 'AL']
        # Badge öne, sonra fırsat skoru yüksek, TAZE/2.DALGA öne
        rt_al_signals.sort(key=lambda s: (
            0 if s.get('badge') or s['ticker'] in badge_seen else 1,
            -(s.get('quality', 0) or 0),
            0 if s.get('entry_window') in ('TAZE', '2.DALGA') else 1,
        ))
        if rt_al_signals:
            data_context.append("\n## Regime Transition AL Sinyalleri:")
            data_context.append("  Fırsat: 0-4 (4=en iyi). TAZE/2.DALGA + Fırsat≥3 = öncelikli.")
            for s in rt_al_signals[:25]:
                window = s.get('entry_window', '?')
                firsat = s.get('quality', '')
                firsat_str = f" F{firsat}" if firsat else ""
                cmf = s.get('cmf')
                cmf_str = f" CMF={cmf:.3f}" if cmf is not None else ""
                adx = s.get('adx')
                adx_str = f" ADX={adx:.0f}" if adx is not None else ""
                oe = s.get('oe', '')
                oe_str = f" OE={oe}" if oe else ""
                badge = s.get('badge', '')
                badge_str = f" 🏅{badge}" if badge else (" ★NW_AL" if s['ticker'] in nw_al_tickers else "")
                data_context.append(
                    f"  {s['ticker']}: RT [{window}]{firsat_str}{badge_str}{cmf_str}{adx_str}{oe_str}")

        # ── Rejim v3 AL sinyalleri ──
        rejim_al_signals = [s for s in latest_signals
                            if s.get('screener') == 'rejim_v3'
                            and s.get('direction') == 'AL']
        if rejim_al_signals:
            data_context.append("\n## Rejim v3 AL Sinyalleri:")
            for s in rejim_al_signals[:15]:
                data_context.append(f"  {s['ticker']}: Rejim AL")

        # ── SAT sinyalleri toplu ──
        sat_signals = [s for s in latest_signals
                       if s.get('direction') == 'SAT' or s.get('karar') == 'SAT']
        if sat_signals:
            data_context.append(f"\n## SAT Sinyalleri ({len(sat_signals)} toplam):")
            for s in sat_signals[:20]:
                scr = SCREENER_NAMES.get(s.get('screener', ''), s.get('screener', '?'))
                data_context.append(f"  {s['ticker']}: {scr} SAT")

    # ── MKK + Takas verisi (VDS auto-fetch) ──
    mkk_json = fetch_mkk_data()
    takas_json = fetch_takas_data()
    if mkk_json and mkk_json.get('data'):
        mkk_data = mkk_json['data']
        data_context.append(f"\n## MKK Yatırımcı Dağılımı ({len(mkk_data)} hisse, tarih: {mkk_json.get('extracted_at', '?')}, {mkk_json.get('history_days', 0)} gün history)")
        data_context.append("  Gerçek MKK verisi — B+K=%100, bireysel düşüşü = kurumsal birikim")
        # Shortlist'teki hisseler için MKK göster
        shortlist_tickers = set()
        for s in latest_signals or []:
            if (s.get('screener') == 'nox_v3_weekly'
                    and s.get('signal_type') == 'PIVOT_AL'):
                shortlist_tickers.add(s['ticker'])
            if s.get('screener') == 'regime_transition' and s.get('badge'):
                shortlist_tickers.add(s['ticker'])
        for ticker in sorted(shortlist_tickers):
            mkk = mkk_data.get(ticker)
            if mkk:
                line = f"  {ticker}: bireysel=%{mkk.get('bireysel_pct', '?')} kurumsal=%{mkk.get('kurumsal_pct', '?')}"
                fark_1g = mkk.get('bireysel_fark_1g')
                if fark_1g is not None:
                    line += f" (1g: {fark_1g:+.2f}%)"
                data_context.append(line)

    if takas_json and takas_json.get('data'):
        takas_data = takas_json['data']
        data_context.append(f"\n## Takas Verisi ({len(takas_data)} hisse, VDS/{takas_json.get('extracted_at', '?')})")
        # Shortlist'teki hisseler için öne çıkan takas verisi
        for ticker in sorted(shortlist_tickers if 'shortlist_tickers' in dir() else []):
            td = takas_data.get(ticker)
            if td:
                kurumlar = td.get('kurumlar', [])
                top3 = sorted(kurumlar, key=lambda k: abs(k.get('Günlük Fark') or k.get('lot_fark') or 0), reverse=True)[:3]
                if top3:
                    parts = []
                    for k in top3:
                        name = k.get('Aracı Kurum') or k.get('kurum') or '?'
                        lot = k.get('Günlük Fark') or k.get('lot_fark') or 0
                        parts.append(f"{name}({lot:+,})")
                    data_context.append(f"  {ticker}: {', '.join(parts)}")

        # ── SMS skorları (takas verisi varsa) ──
        sms_results = calc_batch_sms(None, takas_data, mkk_data if 'mkk_data' in dir() else None)
        if sms_results:
            sorted_sms = sorted(sms_results.values(), key=lambda x: -x.score)
            guclu = [s for s in sorted_sms if s.score >= 45]
            dagitim = [s for s in sorted_sms if s.score < 15]
            data_context.append(f"\n## Smart Money Score ({len(sms_results)} hisse, {len(guclu)}🟢 güçlü, {len(dagitim)}🔴 dağıtım)")
            data_context.append("  SMS: Birikim+Yoğunlaşma+KarşıTaraf+Süreklilik+MKK (max 85p)")
            data_context.append("  ≥45🟢GÜÇLÜ | 30-44🟡ORTA | 15-29⚪ZAYIF | <15🔴DAĞITIM")
            for sms in sorted_sms[:25]:
                data_context.append(f"  {format_sms_line(sms)}")

    # ── Kategori rejimleri ──
    if macro_result and macro_result.get('category_regimes'):
        cat_reg = macro_result['category_regimes']
        data_context.append("\n## Kategori Rejimleri:")
        for cat in ["BIST", "US", "FX", "Emtia", "Kripto", "Faiz"]:
            cr = cat_reg.get(cat)
            if not cr:
                continue
            data_context.append(f"  {cat}: {cr['regime']} (skor: {cr['score']})")
            for inst in cr.get('instruments', []):
                ema_icon = "↑" if inst['above_ema'] else "↓"
                data_context.append(
                    f"    {inst['name']}: EMA21{ema_icon} RSI={inst['rsi']} Range={inst['range_pct']}%")

    # ── Çakışma top hisseler ──
    if confluence_results:
        data_context.append(f"\n## Çakışma Analizi — Top 25 ({len(confluence_results)} hisse ≥1 skor)")
        for r in confluence_results[:25]:
            details_str = "; ".join(r['details'][:5])
            src = r.get('source_count', 0)
            data_context.append(
                f"  {r['ticker']}: skor={r['score']} [{src} kaynak] ({r['recommendation']}) — {details_str}")

    # ── Çoklu kaynak çakışması ──
    multi_source = [r for r in (confluence_results or []) if r.get('source_count', 0) >= 3]
    if multi_source:
        data_context.append(f"\n## 3+ Kaynak Çakışması ({len(multi_source)} hisse):")
        for r in multi_source[:15]:
            data_context.append(f"  {r['ticker']}: {r['source_count']} kaynak, skor={r['score']}")

    # ── Top ticker cross-reference ──
    top = signal_summary.get('top_tickers', [])
    if top:
        data_context.append(f"\n## Çoklu Screener'da Çıkan Hisseler:")
        for t in top[:15]:
            data_context.append(f"  {t['ticker']}: {t['count']} screener")

    prompt = BRIEFING_PROMPT + "\n\n## Veri\n" + "\n".join(data_context)
    return single_prompt(prompt, max_tokens=8192)


def _generate_fallback_briefing(signal_summary, macro_result, confluence_results):
    """Claude API olmadan temel brifing oluştur."""
    now = datetime.now(_TZ_TR)
    lines = [f"⬡ NOX Brifing — {now.strftime('%d.%m.%Y %H:%M')}", ""]

    # Makro
    if macro_result:
        lines.append(format_macro_summary(macro_result))
        lines.append("")

    # Sinyal özeti + veri tarihleri
    total = signal_summary.get('total', 0)
    lines.append(f"📋 Toplam {total} sinyal")
    dates_info = signal_summary.get('screener_dates', {})
    for scr, stats in signal_summary.get('screeners', {}).items():
        name = SCREENER_NAMES.get(scr, scr)
        date_str = dates_info.get(scr, '')
        date_tag = f" [{date_str}]" if date_str else ""
        lines.append(f"  {name}: {stats['total']} ({stats.get('AL', 0)} AL, {stats.get('SAT', 0)} SAT){date_tag}")
    lines.append("")

    # Çakışma top
    if confluence_results:
        lines.append(format_confluence_summary(confluence_results, top_n=10))

    return "\n".join(lines)


def _build_telegram_message(briefing_text, macro_result, confluence_results,
                            html_url=None):
    """Telegram mesajını formatla."""
    now = datetime.now(_TZ_TR)

    # Başlık + link (4000 char chunk bölünmesinde kaybolmaması için üstte)
    lines = [f"<b>⬡ NOX Brifing — {now.strftime('%d.%m.%Y %H:%M')}</b>"]
    if html_url:
        lines.append(f'🔗 <a href="{html_url}">Detaylı Rapor</a>')
    lines.append("")

    # Makro rejim
    if macro_result:
        regime = macro_result.get("regime", "N/A")
        risk = macro_result.get("risk_score", 0)
        lines.append(f"🌍 Rejim: <b>{regime}</b> (skor: {risk})")
        lines.append("")

    # Brifing metni (kısalt)
    max_brief_len = 6000
    if len(briefing_text) > max_brief_len:
        briefing_text = briefing_text[:max_brief_len] + "..."
    lines.append(briefing_text)

    # Çakışma top 5
    if confluence_results:
        lines.append("")
        lines.append("<b>⬡ Top 5 Çakışma:</b>")
        rec_emoji = {
            "GÜÇLÜ_AL": "🟢🟢", "AL": "🟢", "İZLE": "🟡",
            "NÖTR": "⚪", "KAÇIN": "🔴",
        }
        for r in confluence_results[:5]:
            emoji = rec_emoji.get(r['recommendation'], '')
            src = r.get('source_count', 0)
            lines.append(
                f"{emoji} <b>{r['ticker']}</b> skor:{r['score']} [{src} kaynak] ({r['recommendation']})")

    # Scanner HTML linkleri
    scanner_links = _build_scanner_links()
    if scanner_links:
        lines.append(scanner_links)

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description='NOX Agent Brifing')
    parser.add_argument('--notify', action='store_true',
                        help='Telegram + GitHub Pages yayınla')
    parser.add_argument('--no-ai', action='store_true',
                        help='Claude API kullanma')
    parser.add_argument('--fresh', action='store_true',
                        help='Önce NW+RT scanner çalıştır (güncel CSV üret)')
    parser.add_argument('--shortlist', action='store_true',
                        help='Sadece öncelikli hisse listesi (takas verisi iste)')
    args = parser.parse_args()

    run_briefing(notify=args.notify, use_ai=not args.no_ai, fresh=args.fresh,
                 shortlist_only=args.shortlist)


if __name__ == '__main__':
    main()
