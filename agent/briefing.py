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

from agent.scanner_reader import (get_latest_signals, summarize_signals, get_latest_date_signals,
                                   SCREENER_NAMES, export_signals_json,
                                   fetch_mkk_data, fetch_takas_data)
from agent.macro import (fetch_macro_data, fetch_macro_snapshot, assess_macro_regime,
                         format_macro_summary, calc_category_regimes)
from agent.confluence import calc_all_confluence, format_confluence_summary
from agent.smart_money import calc_batch_sms, format_sms_line, sms_icon
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


def _compute_priority_shortlist(latest_signals, confluence_results=None,
                                sms_scores=None):
    """Tüm scanner sinyallerinden öncelikli hisse listesi oluştur.
    Kullanıcının takas/kademe verisi girmesi gereken hisseleri belirler.

    Freshness kuralları:
    - Bugün tetiklenen sinyaller bonus alır (+2)
    - Haftalık sinyal + bugün günlük tetik çakışması ekstra bonus (+3)
    - NW'de fresh=BUGÜN olan sinyaller öne çıkar

    SMS entegrasyonu:
    - Badge sinyal + SMS<15 → "⚠️ SM uyarısı" notu
    - NW İZLE + SMS<30 → shortlist'ten düşür

    Returns: (general_list, tavan_list) — [(ticker, score, reasons_list), ...] sıralı
    """
    ticker_scores = {}  # ticker → {'score': int, 'reasons': [], 'signals': {}}

    def _add(ticker, pts, reason):
        if ticker not in ticker_scores:
            ticker_scores[ticker] = {'score': 0, 'reasons': [], 'signals': {}}
        ticker_scores[ticker]['score'] += pts
        ticker_scores[ticker]['reasons'].append(reason)

    # Bugünün tarih string'leri (signal_date ve csv_date formatları)
    today = datetime.now(_TZ_TR)
    today_sd = today.strftime('%Y-%m-%d')   # signal_date formatı
    today_cd = today.strftime('%Y%m%d')     # csv_date formatı

    def _is_today(s):
        """Sinyal bugün mü üretildi?"""
        sd = s.get('signal_date', '')
        cd = s.get('csv_date', '')
        return sd == today_sd or cd == today_cd

    # NW AL tickers map
    nw_map = {}
    for s in latest_signals:
        if (s.get('screener') == 'nox_v3_weekly'
                and s.get('direction') == 'AL'
                and s.get('signal_type') == 'PIVOT_AL'):
            nw_map[s['ticker']] = s

    # RT AL map
    rt_map = {}
    for s in latest_signals:
        if s.get('screener') == 'regime_transition' and s.get('direction') == 'AL':
            rt_map[s['ticker']] = s

    # Bugün günlük tetik veren ticker'lar (NW fresh=BUGÜN veya günlük sinyal bugün)
    today_triggered = set()
    for s in latest_signals:
        if not _is_today(s):
            continue
        if s.get('fresh') == 'BUGUN' or s.get('fresh') == 'BUGÜN':
            today_triggered.add(s['ticker'])
        elif s.get('screener') in ('nox_v3_daily', 'regime_transition'):
            today_triggered.add(s['ticker'])

    # ── GENEL LİSTE (tavan hariç) ──

    # 1. BADGE sinyalleri (NW + RT çakışma) — EN YÜKSEK ÖNCELİK
    #    Giriş skoru 3+ ve OE ≤ 2 zorunlu
    for s in latest_signals:
        if s.get('screener') == 'regime_transition' and s.get('badge'):
            badge = s['badge']
            window = s.get('entry_window', '')
            if window not in ('TAZE', '2.DALGA'):
                continue
            entry_score = int(s.get('quality', 0) or 0)
            oe = int(s.get('oe', 0) or 0)
            if entry_score < 3:
                continue
            if oe > 2:
                continue
            cmf = s.get('cmf', 0) or 0
            # Badge base: 15 (TAZE) / 13 (2.DALGA) — D+W'nin üstünde kalması için
            pts = 15 if window == 'TAZE' else 13
            if cmf > 0.1:
                pts += 2
            elif cmf < -0.25:
                pts -= 4  # Ciddi negatif CMF
            elif cmf < -0.1:
                pts -= 2  # Negatif CMF = para akışı ters
            if badge == 'H+PB':
                pts += 3  # H+PB tarihsel olarak en yüksek WR setup
            is_fresh = _is_today(s)
            if is_fresh:
                pts += 2
            fresh_tag = " 🔥BUGÜN" if is_fresh else ""
            _add(s['ticker'], pts, f"🏅{badge} [{window}] F{entry_score} OE={oe} CMF{cmf:+.2f}{fresh_tag}")

    # 2. NW PIVOT_AL tetikli sinyaller
    for ticker, s in nw_map.items():
        wl = s.get('wl_status', 'BEKLE')
        trigger = s.get('trigger_type', '')
        delta = s.get('delta_pct')
        if not trigger:
            continue
        fresh = s.get('fresh', '')
        is_fresh_nw = fresh in ('BUGUN', 'BUGÜN')
        has_daily_today = ticker in today_triggered
        # Delta kontrolü — δ>%15 zone dışı, tetik eşiğini aşmış
        d_pct = abs(delta) if delta else 0
        if wl == 'HAZIR':
            if d_pct > 15:
                pts = 5  # Zone dışına çıkmış, HAZIR puanını hak etmiyor
                label = f"NW HAZIR {trigger} δ{delta:.1f}% ⚠️zone↑"
            else:
                pts = 10  # WR %77.8 — en güçlü tek sinyal
                label = f"NW HAZIR {trigger} δ{delta:.1f}%" if delta else f"NW HAZIR {trigger}"
        elif wl == 'İZLE':
            pts = 5
            label = f"NW İZLE {trigger}"
        else:
            pts = 3
            label = f"NW BEKLE {trigger}"
        if is_fresh_nw:
            pts += 2
            label += " 🔥BUGÜN"
        if has_daily_today and not is_fresh_nw:
            pts += 3
            label += " ⚡D+W"
        _add(ticker, pts, label)

    # 3. RT fırsat ≥ 3 + TAZE/2.DALGA sinyaller (badge olmayanlar)
    #    Giriş skoru 3+ ve OE ≤ 2 zorunlu
    for ticker, s in rt_map.items():
        if ticker in ticker_scores and any('🏅' in r for r in ticker_scores[ticker]['reasons']):
            continue
        window = s.get('entry_window', '')
        entry_score = int(s.get('quality', 0) or 0)
        cmf = s.get('cmf', 0) or 0
        oe = int(s.get('oe', 0) or 0)
        if window in ('TAZE', '2.DALGA') and entry_score >= 3 and oe <= 2:
            pts = 4 + (entry_score - 3)
            if cmf > 0.1:
                pts += 1
            is_fresh = _is_today(s)
            if is_fresh:
                pts += 2
            fresh_tag = " 🔥BUGÜN" if is_fresh else ""
            _add(ticker, pts, f"RT {window} F{entry_score} CMF{cmf:+.2f} OE={oe}{fresh_tag}")

    # 4. AL/SAT DÖNÜŞ — Q yüksek
    for s in latest_signals:
        if s.get('screener') == 'alsat' and s.get('direction') == 'AL':
            q = s.get('quality', 0) or s.get('q', 0) or 0
            if q >= 85:
                _add(s['ticker'], 4, f"DÖNÜŞ Q={q}")
            elif q >= 70:
                _add(s['ticker'], 2, f"DÖNÜŞ Q={q}")

    # 5. Çakışma bonus (tavan hariç screener'lardan gelen)
    #    Düşük çakışma skoru (<3) = kaynak bonusu yarıya iner + uyarı
    if confluence_results:
        for r in confluence_results:
            src = r.get('source_count', 0)
            score = r.get('score', 0)
            if src >= 3:
                pts = src - 1
                if score < 3:
                    pts = max(pts // 2, 1)  # Düşük skor = zayıf teyit
                    _add(r['ticker'], pts, f"Çakışma {src} kaynak skor={score} ⚠️zayıf")
                else:
                    _add(r['ticker'], pts, f"Çakışma {src} kaynak skor={score}")

    # SMS entegrasyonu: ikon + uyarı + filtre
    if sms_scores:
        for ticker in list(ticker_scores.keys()):
            sms = sms_scores.get(ticker)
            if not sms:
                continue
            sms_val = sms.score if hasattr(sms, 'score') else sms
            icon = sms_icon(sms_val) if callable(sms_icon) else "⚪"
            reasons = ticker_scores[ticker]['reasons']

            # Badge sinyal + SMS<15 → uyarı notu
            has_badge = any('🏅' in r for r in reasons)
            if has_badge and sms_val < 15:
                reasons.append(f"⚠️SM {sms_val}{icon}")

            # NW İZLE + SMS<30 → puan düşür
            has_izle = any('NW İZLE' in r for r in reasons)
            if has_izle and sms_val < 30:
                ticker_scores[ticker]['score'] -= 2
                reasons.append(f"SM↓ {sms_val}{icon}")

            # SMS≥45 → bonus
            if sms_val >= 45:
                ticker_scores[ticker]['score'] += 2
                reasons.append(f"SM {sms_val}{icon}")
            elif sms_val >= 30:
                reasons.append(f"SM {sms_val}{icon}")
            elif sms_val < 15:
                reasons.append(f"SM {sms_val}{icon}")

    # Genel liste: sırala
    ranked = sorted(ticker_scores.items(), key=lambda x: -x[1]['score'])
    general_list = [(t, info['score'], info['reasons']) for t, info in ranked]

    # ── TAVAN LİSTESİ (tavan + çapraz çakışma) ──
    tavan_scores = {}

    def _tadd(ticker, pts, reason):
        if ticker not in tavan_scores:
            tavan_scores[ticker] = {'score': 0, 'reasons': []}
        tavan_scores[ticker]['score'] += pts
        tavan_scores[ticker]['reasons'].append(reason)

    tavan_tickers = {}
    for s in latest_signals:
        if s.get('screener') in ('tavan', 'tavan_kandidat'):
            t = s['ticker']
            if t in tavan_tickers:
                existing_skor = tavan_tickers[t].get('skor', 0) or 0
                new_skor = s.get('skor', 0) or 0
                if new_skor <= existing_skor:
                    continue
            tavan_tickers[t] = s

    tavan_cross = {}
    for s in latest_signals:
        if s['ticker'] in tavan_tickers and s.get('screener') not in ('tavan', 'tavan_kandidat'):
            tavan_cross.setdefault(s['ticker'], set()).add(s['screener'])

    for ticker, s in tavan_tickers.items():
        skor = s.get('skor', 0) or 0
        vol = s.get('volume_ratio', 0) or 0
        cross_count = len(tavan_cross.get(ticker, set()))
        cross_has_rt = 'regime_transition' in tavan_cross.get(ticker, set())
        cross_has_nw = 'nox_v3_weekly' in tavan_cross.get(ticker, set())

        pts = 0
        label_parts = []

        if skor >= 50 and vol < 1.0:
            pts = 6
            label_parts.append(f"🔒TAVAN skor:{skor} vol:{vol:.1f}x")
        elif skor >= 40 and vol < 1.5:
            pts = 4
            label_parts.append(f"TAVAN skor:{skor} vol:{vol:.1f}x")
        elif skor >= 30 and cross_count >= 3:
            pts = 3
            label_parts.append(f"TAVAN skor:{skor} vol:{vol:.1f}x")
        elif skor >= 40:
            pts = 2
            label_parts.append(f"TAVAN skor:{skor} vol:{vol:.1f}x (yüksek hacim)")

        if pts > 0 and cross_count >= 2:
            bonus = min(cross_count - 1, 3)
            if cross_has_rt:
                bonus += 1
            if cross_has_nw:
                bonus += 1
            pts += bonus
            # Çakışan screener isimlerini göster
            cross_names = []
            _CROSS_SHORT = {
                'nox_v3_weekly': 'NW', 'nox_v3_daily': 'ND',
                'regime_transition': 'RT', 'rejim_v3': 'R3',
                'alsat': 'AS', 'divergence': 'DIV',
            }
            for scr in sorted(tavan_cross.get(ticker, set())):
                cross_names.append(_CROSS_SHORT.get(scr, scr[:3].upper()))
            label_parts.append(f"({'+'.join(cross_names)})")

        if pts > 0:
            # SMS entegrasyonu — tavan listesine de ikon + bonus/uyarı
            if sms_scores:
                sms = sms_scores.get(ticker)
                if sms:
                    sms_val = sms.score if hasattr(sms, 'score') else sms
                    icon = sms_icon(sms_val)
                    if sms_val >= 45:
                        pts += 2
                        label_parts.append(f"SM {sms_val}{icon}")
                    elif sms_val >= 30:
                        label_parts.append(f"SM {sms_val}{icon}")
                    elif sms_val < 15:
                        label_parts.append(f"⚠️SM {sms_val}{icon}")

            _tadd(ticker, pts, " ".join(label_parts))

    tavan_ranked = sorted(tavan_scores.items(), key=lambda x: -x[1]['score'])
    tavan_list = [(t, info['score'], info['reasons']) for t, info in tavan_ranked]

    return general_list, tavan_list


def _push_priority_tickers(general_list, tavan_list):
    """Shortlist ticker'larını priority_tickers.json olarak GitHub Pages'e push et.
    VDS takas scraper bu dosyayı okuyarak hangi hisseler için veri çekeceğini bilir."""
    import json
    tickers = []
    for ticker, _score, _reasons in general_list[:20]:
        tickers.append(ticker)
    for ticker, _score, _reasons in tavan_list[:10]:
        if ticker not in tickers:
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


def _build_shortlist_message(general_list, tavan_list, portfolio=None,
                             sms_scores=None):
    """İki shortlist'i Telegram mesajı olarak formatla."""
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

    lines = [
        f"<b>⬡ NOX Ön Analiz — {now.strftime('%d.%m.%Y %H:%M')}</b>",
    ]

    # ── Genel Liste (badge > D+W > NW > RT > DÖNÜŞ > çakışma) ──
    lines.append("")
    lines.append("📋 <b>Sinyal Listesi</b> (badge → D+W → NW → RT → çakışma)")
    lines.append("")
    takas_needed = []
    for i, (ticker, score, reasons) in enumerate(general_list[:20], 1):
        # SMS ikonu (reasons'da SM satırından veya sms_scores'dan)
        sm_tag = ""
        if sms_scores:
            sms = sms_scores.get(ticker)
            if sms:
                sm_val = sms.score if hasattr(sms, 'score') else sms
                sm_tag = f" {sms_icon(sm_val)}{sm_val}"
        reasons_short = " | ".join(r for r in reasons[:3] if not r.startswith("SM"))
        lines.append(f"{i}. <b>{ticker}</b> [{score}p]{sm_tag}{_tag(ticker)} — {reasons_short}")
        takas_needed.append(ticker)

    # ── Tavan Listesi ──
    if tavan_list:
        lines.append("")
        lines.append(f"🔺 <b>Tavan/Kandidat</b> ({len(tavan_list)} hisse)")
        lines.append("")
        for i, (ticker, score, reasons) in enumerate(tavan_list[:10], 1):
            sm_tag = ""
            if sms_scores:
                sms = sms_scores.get(ticker)
                if sms:
                    sm_val = sms.score if hasattr(sms, 'score') else sms
                    sm_tag = f" {sms_icon(sm_val)}{sm_val}"
            reasons_short = " | ".join(r for r in reasons[:2] if not r.startswith("SM"))
            lines.append(f"{i}. <b>{ticker}</b> [{score}p]{sm_tag}{_tag(ticker)} — {reasons_short}")
            if ticker not in takas_needed:
                takas_needed.append(ticker)

    lines.append("")
    lines.append(f"<b>📊 Bu {len(takas_needed)} hisse için kademe S/A + takas verisi yükle.</b>")

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

    # 1. Scanner sinyallerini yükle
    print("📋 Scanner sinyalleri yükleniyor...")
    signals, csv_map = get_latest_signals(fetch_gh=True)
    if not signals:
        msg = "⚠️ Hiç sinyal bulunamadı — brifing üretilemiyor."
        print(msg)
        if notify:
            send_telegram(msg)
        return

    latest_signals = get_latest_date_signals(signals)
    signal_summary = summarize_signals(latest_signals)
    print(f"  Son tarih: {len(latest_signals)} sinyal")

    # 1b. Sinyalleri JSON olarak export et + GitHub Pages'e push
    if notify and latest_signals:
        json_path = os.path.join(ROOT, 'output', 'latest_signals.json')
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
    try:
        takas_json_sms = fetch_takas_data()
        if takas_json_sms and takas_json_sms.get('data'):
            sms_scores = calc_batch_sms(None, takas_json_sms['data'], mkk_data_map)
            guclu = sum(1 for s in sms_scores.values() if s.score >= 45)
            dagitim = sum(1 for s in sms_scores.values() if s.score < 15)
            print(f"  SMS: {len(sms_scores)} hisse ({guclu}🟢 güçlü, {dagitim}🔴 dağıtım)")
    except Exception as e:
        print(f"  ⚠️ SMS hesaplama hatası: {e}")

    # ── SHORTLIST MODE: sadece öncelikli hisse listesi oluştur ──
    if shortlist_only:
        print("\n📋 Öncelikli hisse listesi oluşturuluyor...")
        general_list, tavan_list = _compute_priority_shortlist(
            latest_signals, confluence_results, sms_scores)
        portfolio = _load_portfolio()
        print(f"  Genel: {len(general_list)} hisse, Tavan: {len(tavan_list)} hisse\n")
        print("  ── Sinyal Listesi ──")
        for i, (ticker, score, reasons) in enumerate(general_list[:20], 1):
            print(f"  {i:2d}. {ticker:6s} [{score:2d}p] — {' | '.join(reasons[:3])}")
        if tavan_list:
            print(f"\n  ── Tavan ({len(tavan_list)}) ──")
            for i, (ticker, score, reasons) in enumerate(tavan_list[:10], 1):
                print(f"  {i:2d}. {ticker:6s} [{score:2d}p] — {' | '.join(reasons[:2])}")
        if notify:
            msg = _build_shortlist_message(general_list, tavan_list, portfolio,
                                            sms_scores)
            send_telegram(msg)
            print(f"\n  ✅ Shortlist Telegram'a gönderildi")
            # VDS takas scraper için ticker listesini GitHub Pages'e push
            _push_priority_tickers(general_list, tavan_list)
        return {"shortlist": general_list, "tavan": tavan_list}

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

    # Çakışmayı makro + SMS ile tekrar hesapla
    confluence_results = calc_all_confluence(
        latest_signals, macro_result, min_score=1, mkk_data=mkk_data_map,
        sms_scores=sms_scores)

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

    # 5. HTML rapor oluştur
    print("\n📊 HTML rapor oluşturuluyor...")
    html = generate_briefing_html(
        briefing_text, macro_result, confluence_results, signal_summary)
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
