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

import pandas as pd

# Proje kökünü path'e ekle
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(ROOT, '.env'))

from agent.scanner_reader import (summarize_signals, SCREENER_NAMES,
                                   export_signals_json,
                                   fetch_mkk_data)
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


# ═══════════════════════════════════════════
# ML SCORING OVERLAY v2 (3-Katmanlı, Feature flag: ML_SCORING_ENABLED)
# ═══════════════════════════════════════════

_YF_CACHE_PATH = os.path.join(ROOT, "output", "yf_price_cache.parquet")
_YF_XU_CACHE_PATH = os.path.join(ROOT, "output", "yf_xu100_cache.parquet")


def _normalize_yf_columns(df):
    """yfinance DataFrame kolon isimlerini standartlaştır."""
    col_map = {}
    for col in df.columns:
        cs = str(col).strip().lower()
        if cs in ('close', 'adj close'):
            col_map[col] = 'Close'
        elif cs == 'open':
            col_map[col] = 'Open'
        elif cs == 'high':
            col_map[col] = 'High'
        elif cs == 'low':
            col_map[col] = 'Low'
        elif cs == 'volume':
            col_map[col] = 'Volume'
    if col_map:
        df = df.rename(columns=col_map)
    return df


def _load_yf_cache():
    """Cache'den fiyat verisi yükle. Returns: (price_data dict, xu_df, cache_end_date)."""
    price_data = {}
    xu_df = None
    cache_end = None

    if os.path.exists(_YF_CACHE_PATH):
        try:
            cached = pd.read_parquet(_YF_CACHE_PATH)
            if 'ticker' in cached.columns:
                for t, grp in cached.groupby('ticker'):
                    df = grp.drop(columns=['ticker']).copy()
                    df.index = pd.to_datetime(df.index)
                    df = df.sort_index()
                    if len(df) >= 80 and 'Close' in df.columns:
                        price_data[t] = df
                if price_data:
                    cache_end = max(df.index.max() for df in price_data.values())
                    cache_end = cache_end.date() if hasattr(cache_end, 'date') else cache_end
                    print(f"  [Cache] {len(price_data)} hisse yüklendi (son: {cache_end})")
        except Exception as e:
            print(f"  [Cache] Okuma hatası: {e}")

    if os.path.exists(_YF_XU_CACHE_PATH):
        try:
            xu_df = pd.read_parquet(_YF_XU_CACHE_PATH)
            xu_df.index = pd.to_datetime(xu_df.index)
        except Exception:
            pass

    return price_data, xu_df, cache_end


def _save_yf_cache(price_data, xu_df):
    """Fiyat verisini parquet cache'e yaz."""
    os.makedirs(os.path.dirname(_YF_CACHE_PATH), exist_ok=True)
    try:
        frames = []
        for t, df in price_data.items():
            tmp = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
            tmp['ticker'] = t
            frames.append(tmp)
        if frames:
            combined = pd.concat(frames)
            combined.to_parquet(_YF_CACHE_PATH)
            print(f"  [Cache] {len(price_data)} hisse kaydedildi")
    except Exception as e:
        print(f"  [Cache] Yazma hatası: {e}")

    if xu_df is not None:
        try:
            xu_df.to_parquet(_YF_XU_CACHE_PATH)
        except Exception:
            pass


def _fetch_yf_price_data(tickers):
    """yfinance ile fiyat verisi çek — inkremental cache destekli.

    1. Cache varsa yükle, son tarihi bul
    2. Sadece eksik günleri indir (start=son_tarih+1)
    3. Cache'e append et
    4. İlk run: ~25 dk, sonraki: saniyeler

    Returns: (price_data: dict, xu_df: DataFrame or None)
    """
    import yfinance as yf
    from datetime import date

    today = date.today()

    # Cache yükle
    price_data, xu_df, cache_end = _load_yf_cache()

    # Cache yeterince güncel mi? (aynı gün veya dünkü iş günü)
    need_full = False
    need_delta = False
    delta_start = None

    if not price_data:
        need_full = True
        print(f"  [YF] Cache yok — tam indirme yapılacak ({len(tickers)} hisse)")
    elif cache_end and (today - cache_end).days <= 1:
        # Cache bugün veya dün — güncel
        # Eksik ticker'lar var mı?
        missing = [t for t in tickers if t not in price_data]
        if missing:
            print(f"  [YF] Cache güncel, {len(missing)} yeni hisse indiriliyor")
            need_full = False
            need_delta = False
            # Sadece eksikleri indir
            _download_and_merge(missing, price_data, yf, full=True)
        else:
            print(f"  [YF] Cache güncel ({cache_end}), indirme atlanıyor ✓")
    elif cache_end:
        delta_days = (today - cache_end).days
        if delta_days <= 10:
            need_delta = True
            delta_start = cache_end + timedelta(days=1)
            print(f"  [YF] Cache {delta_days} gün eski — delta indirme ({delta_start} → {today})")
        else:
            need_full = True
            print(f"  [YF] Cache {delta_days} gün eski — tam indirme gerekli")
    else:
        need_full = True

    if need_full:
        price_data = {}
        _download_and_merge(tickers, price_data, yf, full=True)

    elif need_delta and delta_start:
        # İnkremental: sadece eksik günler
        all_tickers = list(set(tickers) | set(price_data.keys()))
        yf_syms = [f"{t}.IS" for t in all_tickers]
        try:
            raw = yf.download(" ".join(yf_syms),
                              start=delta_start.strftime("%Y-%m-%d"),
                              end=(today + timedelta(days=1)).strftime("%Y-%m-%d"),
                              progress=False, auto_adjust=True,
                              group_by='ticker', threads=True)
            if not raw.empty:
                new_count = 0
                for t, yf_t in zip(all_tickers, yf_syms):
                    try:
                        df_new = _extract_ticker_df(raw, t, yf_t, len(yf_syms))
                        if df_new is not None and not df_new.empty:
                            if t in price_data:
                                # Append — yeni günleri ekle
                                existing = price_data[t]
                                combined = pd.concat([existing, df_new])
                                combined = combined[~combined.index.duplicated(keep='last')]
                                combined = combined.sort_index()
                                # Son 1 yılı tut (memory tasarrufu)
                                cutoff = combined.index.max() - pd.Timedelta(days=370)
                                combined = combined[combined.index >= cutoff]
                                price_data[t] = combined
                                new_count += 1
                            else:
                                # Yeni ticker — yeterli veri varsa ekle
                                if len(df_new) >= 5:
                                    price_data[t] = df_new
                                    new_count += 1
                    except Exception:
                        continue
                print(f"  [YF] Delta: {new_count} hisse güncellendi")
        except Exception as e:
            print(f"  [YF] Delta indirme hatası: {e}")

        # Delta'da eksik kalan yeni ticker'ları tam indir
        missing = [t for t in tickers if t not in price_data]
        if missing:
            print(f"  [YF] {len(missing)} yeni hisse tam indiriliyor")
            _download_and_merge(missing, price_data, yf, full=True)

    # XU100 güncelle
    if xu_df is None or xu_df.empty or need_full:
        try:
            xu = yf.download("XU100.IS", period="1y", progress=False, auto_adjust=True)
            if isinstance(xu.columns, pd.MultiIndex):
                xu.columns = xu.columns.get_level_values(0)
            xu = _normalize_yf_columns(xu)
            if not xu.empty and 'Close' in xu.columns:
                xu_df = xu
        except Exception:
            pass
    elif need_delta and delta_start:
        try:
            xu_new = yf.download("XU100.IS",
                                 start=delta_start.strftime("%Y-%m-%d"),
                                 progress=False, auto_adjust=True)
            if isinstance(xu_new.columns, pd.MultiIndex):
                xu_new.columns = xu_new.columns.get_level_values(0)
            xu_new = _normalize_yf_columns(xu_new)
            if not xu_new.empty and 'Close' in xu_new.columns:
                xu_df = pd.concat([xu_df, xu_new])
                xu_df = xu_df[~xu_df.index.duplicated(keep='last')].sort_index()
        except Exception:
            pass

    # Cache kaydet
    _save_yf_cache(price_data, xu_df)

    print(f"  ✅ YF: {len(price_data)} hisse hazır")
    return price_data, xu_df


def _extract_ticker_df(raw, ticker, yf_ticker, total_syms):
    """yfinance raw DataFrame'den tek ticker çıkar ve normalize et."""
    try:
        if total_syms == 1:
            df = raw.copy()
        elif isinstance(raw.columns, pd.MultiIndex):
            level_0 = raw.columns.get_level_values(0).unique().tolist()
            level_1 = raw.columns.get_level_values(1).unique().tolist()
            price_cols = {'Open', 'High', 'Low', 'Close', 'Volume'}
            if any(v in price_cols for v in level_0):
                key = yf_ticker if yf_ticker in level_1 else ticker
                df = raw.xs(key, level=1, axis=1).copy()
            else:
                key = yf_ticker if yf_ticker in level_0 else ticker
                df = raw[key].copy()
        else:
            return None

        df = _normalize_yf_columns(df)
        df = df.dropna(how='all')
        return df
    except Exception:
        return None


def _download_and_merge(tickers, price_data, yf, full=True):
    """Tam 1y indirme yap ve price_data dict'e merge et."""
    if not tickers:
        return
    yf_syms = [f"{t}.IS" for t in tickers]
    try:
        raw = yf.download(" ".join(yf_syms), period="1y",
                          progress=False, auto_adjust=True,
                          group_by='ticker', threads=True)
        if not raw.empty:
            for t, yf_t in zip(tickers, yf_syms):
                df = _extract_ticker_df(raw, t, yf_t, len(yf_syms))
                if df is not None and len(df) >= 80 and 'Close' in df.columns:
                    price_data[t] = df
            print(f"  ✅ TradingView → {len(price_data)} hisse")
    except Exception as e:
        print(f"  ⚠️ YF indirme hatası: {e}")


def _calc_taban_risk(df):
    """1Y OHLCV'den likidite-bazlı taban riski hesapla.

    Returns: dict with consec_max, taban_days, avg_tl_vol, price
    """
    returns = df['Close'].pct_change()
    close = df['Close']
    volume = df['Volume']

    # 1) Ardışık taban günleri (en uzun seri)
    taban_mask = (returns <= -0.09)
    consec_max = 0
    consec_cur = 0
    for is_taban in taban_mask:
        if is_taban:
            consec_cur += 1
            consec_max = max(consec_max, consec_cur)
        else:
            consec_cur = 0

    # 2) Toplam taban günü sayısı
    taban_days = int(taban_mask.sum())

    # 3) Ortalama günlük TL hacim (son 20 gün)
    tl_vol = (close * volume).tail(20)
    avg_tl_vol = float(tl_vol.mean()) if len(tl_vol) > 0 else 0

    # 4) Son fiyat
    price = float(close.iloc[-1])

    return {
        'consec_max': consec_max,
        'taban_days': taban_days,
        'avg_tl_vol': avg_tl_vol,
        'price': price,
    }


def _taban_risk_overlay(lists_dict):
    """Tüm listelere taban riski bilgi label'ı ekle (skor cezası yok).

    Backtest verisi taban risk metriklerinin WR'ı düşürmediğini gösterdi.
    Bu overlay sadece bilgilendirme amaçlı: ardışık taban geçmişi veya
    düşük likidite olan hisselere ⚠️ notu ekler (pozisyon boyutu kararı
    kullanıcıda).
    """
    # 1) Ticker toplama
    all_tickers = set()
    for key in ('tier1', 'tier2', 'tier2a', 'tier2b', 'alsat', 'tavan', 'nw', 'rt', 'sbt'):
        for item in lists_dict.get(key, []):
            all_tickers.add(item[0])
    if not all_tickers:
        return

    # 2) Fiyat verisi çek
    price_data, _ = _fetch_yf_price_data(sorted(all_tickers))

    # 3) Risk hesapla
    taban_risks = {}
    for t, df in price_data.items():
        try:
            taban_risks[t] = _calc_taban_risk(df)
        except Exception:
            continue

    if not taban_risks:
        return

    warned = 0

    # 4) Her listeye bilgi label'ı ekle (skor değişmez, sıralama değişmez)
    for key in ('tier1', 'tier2', 'tier2a', 'tier2b', 'alsat', 'tavan', 'nw', 'rt', 'sbt'):
        items = lists_dict.get(key, [])
        for (ticker, score, reasons, sig) in items:
            tr = taban_risks.get(ticker)
            if not tr:
                continue
            # Sadece ardışık taban ≥ 2 (gerçek tuzak: çıkış yok)
            if tr['consec_max'] >= 2:
                reasons.append(f"⚠️taban(ardışık{tr['consec_max']}x,{tr['taban_days']}gün)")
                warned += 1

    if warned:
        print(f"  ⚠️ Taban uyarı: {warned} hisse (bilgi notu, skor cezası yok)")


def _fetch_sbt_data():
    """SBT (Smart Breakout Targets) verisini GH Pages'ten çek.

    Returns: {ticker: {'ml_prob': float, 'ml_bucket': str, 'ml_gate': bool}} veya {}
    """
    import requests
    from html.parser import HTMLParser

    nox_base = os.environ.get("GH_PAGES_BASE_URL",
                               "https://aalpkk.github.io/nox-signals").rstrip("/")
    url = f"{nox_base}/smart_breakout.html"

    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code != 200:
            return {}
        return _parse_sbt_html(resp.text)
    except Exception:
        return {}


class _SBTTableParser:
    """SBT HTML tablosundan ticker + ML prob + bucket parse eder."""

    def __init__(self):
        from html.parser import HTMLParser
        self._parser_cls = HTMLParser

    @staticmethod
    def parse(html_text):
        """HTML'den SBT verisi çıkar.

        SBT HTML'de const D formatı veya tablo olabilir.
        Returns: {ticker: {'ml_prob': float, 'ml_bucket': str, 'ml_gate': bool}}
        """
        import json
        import re

        result = {}

        # Önce const D JSON formatını dene
        m = re.search(r'const\s+D\s*=\s*(\{.+?\})\s*;', html_text, re.DOTALL)
        if m:
            try:
                raw = m.group(1)
                raw = re.sub(r',\s*([\]}])', r'\1', raw)
                d = json.loads(raw)
                # D.rows formatı
                for row in d.get('rows', []):
                    ticker = row.get('ticker', '')
                    if not ticker:
                        continue
                    ml_prob = row.get('ml_prob', row.get('prob', 0))
                    ml_bucket = row.get('ml_bucket', row.get('bucket', ''))
                    result[ticker] = {
                        'ml_prob': float(ml_prob) if ml_prob else 0.0,
                        'ml_bucket': str(ml_bucket),
                        'ml_gate': str(ml_bucket) in ('A+', 'A'),
                    }
                return result
            except (json.JSONDecodeError, KeyError):
                pass

        # Fallback: data-ticker table parse
        from html.parser import HTMLParser

        class _Parser(HTMLParser):
            def __init__(self):
                super().__init__()
                self.rows = []
                self._cells = []
                self._in_td = False
                self._td_text = ""
                self._in_table = False
                self._in_a_tk = False
                self._row_ticker = None

            def handle_starttag(self, tag, attrs):
                attrs_d = dict(attrs)
                if tag in ('table', 'tbody'):
                    self._in_table = True
                elif tag == 'tr' and self._in_table:
                    self._cells = []
                    self._row_ticker = attrs_d.get('data-ticker')
                elif tag == 'td' and self._in_table:
                    self._in_td = True
                    self._td_text = ""
                elif tag == 'a' and self._in_td:
                    if 'tk' in (attrs_d.get('class', '')):
                        self._in_a_tk = True

            def handle_endtag(self, tag):
                if tag == 'table':
                    self._in_table = False
                elif tag == 'a' and self._in_a_tk:
                    self._in_a_tk = False
                elif tag == 'td' and self._in_td:
                    self._in_td = False
                    self._cells.append(self._td_text.strip())
                elif tag == 'tr' and self._in_table and self._cells:
                    # Ticker: data-ticker attr veya ilk hücredeki text
                    ticker = self._row_ticker or self._cells[0].strip()
                    if ticker and len(ticker) >= 3 and not ticker.startswith('Sembol'):
                        self.rows.append((ticker, self._cells))
                    self._cells = []
                    self._row_ticker = None

            def handle_data(self, data):
                if self._in_a_tk and self._in_td:
                    self._row_ticker = data.strip()
                if self._in_td:
                    self._td_text += data

        parser = _Parser()
        parser.feed(html_text)
        for ticker, cells in parser.rows:
            # Hücre yapısı değişken — bucket ve prob bulmaya çalış
            ml_prob = 0.0
            ml_bucket = ''
            for cell in cells:
                cell_s = cell.strip()
                if cell_s in ('A+', 'A', 'B', 'C', 'X'):
                    ml_bucket = cell_s
                try:
                    v = float(cell_s.replace('%', ''))
                    if 0 < v <= 1:
                        ml_prob = v
                    elif 1 < v <= 100:
                        ml_prob = v / 100
                except ValueError:
                    pass
            if ml_bucket or ml_prob > 0:
                result[ticker] = {
                    'ml_prob': ml_prob,
                    'ml_bucket': ml_bucket,
                    'ml_gate': ml_bucket in ('A+', 'A'),
                }

        return result


def _parse_sbt_html(html_text):
    """SBT HTML'den veri çıkar. Wrapper for _SBTTableParser."""
    return _SBTTableParser.parse(html_text)


def _calc_gate_penalty(ticker, sig, sbt_data):
    """SBT gate penalty hesapla.

    Args:
        ticker: str
        sig: dict — sinyal bilgisi
        sbt_data: dict — _fetch_sbt_data() sonucu

    Returns: int — 0=OK, 1=soft gate, 99=hard gate (eleme)
    """
    sbt = sbt_data.get(ticker)
    if not sbt:
        return 0  # SBT verisi yok — ceza yok

    bucket = sbt.get('ml_bucket', '')

    # Hard gate: SBT X bucket + EMA50 altında (proxy: screener quality düşük)
    if bucket == 'X':
        # EMA50 altı proxy: quality düşükse veya OE yüksekse
        oe = sig.get('oe', 0) if isinstance(sig, dict) else 0
        quality = sig.get('quality', 0) if isinstance(sig, dict) else 0
        oe_val = int(oe) if oe != '' and oe is not None else 0
        q_val = int(quality) if quality else 0
        if oe_val >= 3 or q_val < 40:
            return 99  # Hard gate — eleme
        return 1  # Soft gate — uyarı

    if bucket in ('C', 'B'):
        return 0  # OK — ceza yok

    return 0  # A, A+ — OK


def _ml_overlay_v2(lists_dict, latest_signals):
    """3-Katmanlı ML overlay: dual skor, composite scoring, SBT gate, rerank.

    Katman A: Rule engine (mevcut _compute_4_lists) — aday üretir
    Katman B: Shortlist ML — dual skor (1g+3g), reranker
    Katman C: Scanner-specific ML — SBT bucket/gate

    Feature flag: ML_SCORING_ENABLED env var kontrol edilir.
    Hata durumunda sessiz fail — mevcut sistemi bozmaz.

    Returns: ml_scores dict veya None
    """
    if not os.getenv('ML_SCORING_ENABLED', '').lower() in ('1', 'true', 'yes'):
        return None

    try:
        from ml.scorer import (MLScorer, ml_badge_dual, calc_ml_rerank_bonus,
                                calc_ml_rerank_bonus_v2, ML_EFFECT_ICONS,
                                calc_ml_effect_label,
                                calc_source_quality_bonus, calc_overlap_bonus)

        print("\n🤖 ML skorlama v2 yapılıyor...")
        scorer = MLScorer()

        # ── Step 1: Ticker toplama ──
        all_tickers = set()
        for key in ('tier1', 'tier2', 'tier2a', 'tier2b', 'alsat', 'tavan', 'nw', 'rt', 'sbt'):
            for item in lists_dict.get(key, []):
                all_tickers.add(item[0])

        if not all_tickers:
            return None

        tickers = sorted(all_tickers)
        print(f"  [ML] {len(tickers)} ticker için fiyat verisi çekiliyor...")

        # ── Step 2-3: Fiyat verisi + XU100 ──
        price_data, xu_df = _fetch_yf_price_data(tickers)

        # ── Step 4: Dual ML skorla (score_tickers_dual lazy-load tetikler) ──
        ml_scores = scorer.score_tickers_dual(tickers, price_data, xu_df)
        if ml_scores:
            print(f"  [ML] {len(ml_scores)}/{len(tickers)} ticker skorlandı (dual)")
        else:
            print("  [ML] ML skorlama başarısız veya model yok — composite SBT-only")

        # ── Step 4b: Breakout ML (tüm BIST evreni, opsiyonel) ──
        from ml.scorer import is_breakout_ml_enabled
        breakout_scores = {}
        if is_breakout_ml_enabled():
            try:
                # Tüm BIST evrenini tara (shortlist değil)
                from markets.bist.data import get_all_bist_tickers
                all_bist = get_all_bist_tickers()
                # Mevcut price_data'yı genişlet (eksik hisseleri indir)
                missing = [t for t in all_bist if t not in price_data]
                if missing:
                    print(f"  [BRK] {len(missing)} ek hisse indiriliyor...")
                    extra_price, _ = _fetch_yf_price_data(missing)
                    price_data.update(extra_price)
                    print(f"  [BRK] Toplam: {len(price_data)} hisse")
                breakout_scores = scorer.score_breakout(all_bist, price_data, xu_df)
                if breakout_scores:
                    print(f"  [BRK] {len(breakout_scores)}/{len(all_bist)} breakout skorlandı")
                    # Shortlist ticker'ları ile çapraz referans
                    shortlist_tickers = set()
                    for key in ('tier1', 'tier2a', 'tier2b'):
                        for item in lists_dict.get(key, []):
                            shortlist_tickers.add(item[0])
                    # Top 5 (yüksek güven) + Top 10 (izle) — fusion_score ile sıralı
                    alerts = []
                    for t, bdata in sorted(breakout_scores.items(),
                                            key=lambda x: x[1].get('fusion_score', 0),
                                            reverse=True):
                        if bdata.get('alert'):
                            alert = dict(bdata)
                            alert['ticker'] = t
                            if t in shortlist_tickers:
                                alert['in_shortlist'] = 'Shortlist\'te mevcut'
                            alerts.append(alert)
                    if alerts:
                        lists_dict['_breakout_alerts'] = alerts[:10]
                        n_top5 = sum(1 for a in alerts if a.get('tier') == 'top5')
                        n_top10 = sum(1 for a in alerts if a.get('tier') == 'top10')
                        print(f"  [BRK] {len(alerts)} alert (top5={n_top5}, top10={n_top10})")
            except Exception as e:
                print(f"  [BRK] Breakout skorlama hatası: {e}")

        # ── Step 5: SBT verisi (opportunistic, hata={}) ──
        print("  [ML] SBT verisi çekiliyor...")
        sbt_data = _fetch_sbt_data()
        if sbt_data:
            print(f"  [ML] SBT: {len(sbt_data)} hisse")
        else:
            print("  [ML] SBT verisi yok — atlanıyor")

        # ── Step 6: Pre-ML sıralama snapshot ──
        _LIST_SHORT = {'alsat': 'AS', 'tavan': 'TVN', 'nw': 'NW', 'rt': 'RT', 'sbt': 'SBT'}
        pre_ml_ranks = {}  # {key: {ticker: rank_index}}
        for key in ('tier1', 'tier2', 'tier2a', 'tier2b', 'alsat', 'tavan', 'nw', 'rt', 'sbt'):
            items = lists_dict.get(key, [])
            pre_ml_ranks[key] = {item[0]: i for i, item in enumerate(items)}

        # Ticker → kaç listede mevcut (Tier1 dahil değil — tier1 zaten overlap)
        ticker_source_count = {}
        for key in ('alsat', 'tavan', 'nw', 'rt', 'sbt'):
            for item in lists_dict.get(key, []):
                ticker_source_count.setdefault(item[0], set()).add(key)
        for t, sources in ticker_source_count.items():
            ticker_source_count[t] = len(sources)

        # ── Step 7-8: Composite score + ML enjeksiyon ──
        ml_filtered = []   # Hard gate / weak filter tarafından elenen
        rank_changes = []  # Rerank sonrası sıra değişiklikleri

        for key in ('tier1', 'tier2', 'tier2a', 'tier2b', 'alsat', 'tavan', 'nw', 'rt', 'sbt'):
            items = lists_dict.get(key, [])
            for i, (ticker, score, reasons, sig_or_meta) in enumerate(items):
                ml_data = ml_scores.get(ticker)

                # ML skorları hesapla (dual)
                ml_short = None  # 1g
                ml_swing = None  # 3g

                if ml_data:
                    ml_short = ml_data.get('ml_score_short')
                    ml_swing = ml_data.get('ml_score_swing')

                    # ── Step 7: Model D — TVN conditional blend (her iki horizon) ──
                    is_tvn = key == 'tavan' or (isinstance(sig_or_meta, dict)
                             and sig_or_meta.get('screener') == 'tavan')
                    if is_tvn:
                        if ml_data.get('ml_b_1g') is not None and ml_data.get('ml_a_1g') is not None:
                            ml_short = 0.65 * ml_data['ml_b_1g'] + 0.35 * ml_data['ml_a_1g']
                        if ml_data.get('ml_b_3g') is not None and ml_data.get('ml_a_3g') is not None:
                            ml_swing = 0.65 * ml_data['ml_b_3g'] + 0.35 * ml_data['ml_a_3g']

                    # Sig'e dual ML skorları enjekte et
                    if isinstance(sig_or_meta, dict):
                        sig_or_meta['ml_score'] = round(ml_short, 3) if ml_short is not None else None
                        sig_or_meta['ml_score_short'] = round(ml_short, 3) if ml_short is not None else None
                        sig_or_meta['ml_score_swing'] = round(ml_swing, 3) if ml_swing is not None else None

                # Breakout ML skorunu enjekte et
                if isinstance(sig_or_meta, dict) and breakout_scores:
                    brk = breakout_scores.get(ticker)
                    if brk:
                        sig_or_meta['breakout_master'] = round(brk.get('breakout_master', 0), 3)
                        sig_or_meta['breakout_fusion'] = round(brk.get('fusion_score', 0), 3)
                        sig_or_meta['breakout_tier'] = brk.get('tier')
                        # BRK detay skorları
                        if brk.get('ml_s_score') is not None:
                            sig_or_meta['brk_ml_s'] = round(brk['ml_s_score'], 3)
                        if brk.get('tavan_prob') is not None:
                            sig_or_meta['brk_tavan_prob'] = round(brk['tavan_prob'], 3)
                        if brk.get('rally_prob') is not None:
                            sig_or_meta['brk_rally_prob'] = round(brk['rally_prob'], 3)
                        # ALMA flag: breakout potansiyeli yok + kısa momentum zayıf
                        bm = brk.get('breakout_master', 0)
                        ms = brk.get('ml_s_score')
                        if bm < 0.08 and (ms is not None and ms < 0.43):
                            sig_or_meta['brk_avoid'] = True

                # SBT verisi enjekte et
                sbt_info = sbt_data.get(ticker, {})
                sbt_bucket = sbt_info.get('ml_bucket', '')
                if isinstance(sig_or_meta, dict):
                    if sbt_bucket:
                        sig_or_meta['sbt_bucket'] = sbt_bucket
                    sig_or_meta['_rule_score'] = score  # Orijinal rule score sakla

                # ── Step 8: Composite score hesapla ──
                # Tier1 için overlap_bonus=0 (zaten overlap quality baked-in)
                is_tier1 = key == 'tier1'
                src_count = ticker_source_count.get(ticker, 1)
                overlap_b = 0 if is_tier1 else calc_overlap_bonus(src_count)
                ml_rerank = calc_ml_rerank_bonus_v2(ml_swing)
                sq_bonus = calc_source_quality_bonus(sbt_bucket)
                # SBT gate sadece SBT listesine uygulanır — AS/NW/RT/TVN kendi sinyalleri
                if key == 'sbt':
                    gate_pen = _calc_gate_penalty(ticker, sig_or_meta if isinstance(sig_or_meta, dict) else {}, sbt_data)
                else:
                    gate_pen = 0

                # ML effect hesapla
                ml_effect = calc_ml_effect_label(ml_swing, ml_rerank)

                # Gate penalty — hard gate (99) filtre olarak işaretlenir
                if isinstance(sig_or_meta, dict):
                    sig_or_meta['gate_penalty'] = gate_pen
                    sig_or_meta['ml_effect'] = ml_effect

                if gate_pen >= 99:
                    # Hard gate — eleme (sadece SBT listesinde)
                    final_score = -9999
                else:
                    final_score = score + overlap_b + ml_rerank + sq_bonus - gate_pen

                # Tuple'ı güncelle (score alanı composite olur)
                items[i] = (ticker, final_score, reasons, sig_or_meta)

                # Dual badge + ML effect reason'a ekle
                badge = ml_badge_dual(ml_short, ml_swing)
                if badge and isinstance(reasons, list):
                    reasons.append(badge)
                if ml_effect != 'neutral' and isinstance(reasons, list):
                    reasons.append(ML_EFFECT_ICONS[ml_effect])

                # SBT bucket tag ekle (? ve boş değerleri atla)
                if sbt_bucket and sbt_bucket not in ('?', '') and isinstance(reasons, list):
                    reasons.append(f"SBT:{sbt_bucket}")

        # ── Step 9: Re-sort + rank delta ──
        for key in ('alsat', 'tavan', 'nw', 'rt', 'sbt'):
            items = lists_dict.get(key, [])
            items.sort(key=lambda x: -x[1])
            # Rank delta hesapla
            for new_rank, (ticker, _, _, _) in enumerate(items):
                old_rank = pre_ml_ranks.get(key, {}).get(ticker, new_rank)
                delta = old_rank - new_rank  # Pozitif = yükseldi
                if abs(delta) >= 2:
                    tag = _LIST_SHORT.get(key, key)
                    rank_changes.append({
                        'ticker': ticker,
                        'list': key,
                        'list_tag': tag,
                        'old_rank': old_rank + 1,
                        'new_rank': new_rank + 1,
                        'delta': delta,
                    })

        # Tier1/Tier2 de re-sort
        for key in ('tier1', 'tier2', 'tier2a', 'tier2b'):
            items = lists_dict.get(key, [])
            items.sort(key=lambda x: -x[1])

        # ── Step 10: 3-Zone Decision Tree filtresi (W-score bazlı) ──
        tier1_tickers_set = {item[0] for item in lists_dict.get('tier1', [])}

        for key in ('alsat', 'tavan', 'nw', 'rt', 'sbt', 'tier2', 'tier2a', 'tier2b'):
            items = lists_dict.get(key, [])
            kept = []
            for ticker, final_score, reasons, sig_or_meta in items:
                gate_pen = 0
                ml_w = None
                src_cnt = ticker_source_count.get(ticker, 1)
                sbt_b = ''
                if isinstance(sig_or_meta, dict):
                    gate_pen = sig_or_meta.get('gate_penalty', 0)
                    ml_w = sig_or_meta.get('ml_score_swing')
                    sbt_b = sig_or_meta.get('sbt_bucket', '')

                is_tier = ticker in tier1_tickers_set
                has_quality = sbt_b in ('A+', 'A')

                # Hard gate eleme
                if gate_pen >= 99:
                    ml_filtered.append({
                        'ticker': ticker,
                        'reason': 'SBT X + zayıf',
                        'rule_score': sig_or_meta.get('_rule_score', final_score) if isinstance(sig_or_meta, dict) else final_score,
                        'list': key,
                    })
                    continue

                # WEAK zone: W < 0.40 (gevşetildi, 0.45→0.40)
                rule_s = sig_or_meta.get('_rule_score', final_score) if isinstance(sig_or_meta, dict) else final_score
                if ml_w is not None and ml_w < 0.40:
                    if src_cnt >= 2 or is_tier or has_quality or rule_s >= 150:
                        # Multi-source, tier1, SBT A+/A, veya güçlü rule → koru
                        kept.append((ticker, final_score, reasons, sig_or_meta))
                        continue
                    else:
                        # Tek kaynak + W<0.40 + kalitesiz + düşük rule → eleme
                        ml_filtered.append({
                            'ticker': ticker,
                            'reason': f'tek kaynak + W<0.40 ({int(ml_w*100)})',
                            'rule_score': rule_s,
                            'list': key,
                        })
                        continue

                kept.append((ticker, final_score, reasons, sig_or_meta))
            lists_dict[key] = kept

        # Tier2 = tier2a + tier2b (güncellenmiş)
        lists_dict['tier2'] = lists_dict.get('tier2a', []) + lists_dict.get('tier2b', [])

        # Store extra data for message building
        lists_dict['_ml_filtered'] = ml_filtered
        lists_dict['_ml_rank_changes'] = rank_changes
        lists_dict['_sbt_data'] = sbt_data

        n_filtered = len(ml_filtered)
        n_rank_changes = len(rank_changes)
        print(f"  [ML] Composite: {n_filtered} elendi, {n_rank_changes} rerank değişimi")

        return ml_scores

    except Exception as e:
        import traceback
        print(f"  [ML] Hata: {e} — atlanıyor")
        traceback.print_exc()
        return None


# ═══════════════════════════════════════════
# SEKTÖR REGIME OVERLAY (Soft gate + badge)
# ═══════════════════════════════════════════

def _sector_regime_overlay(lists_dict):
    """Sektör endeksi regime overlay — soft gate + uyarı badge.

    1) Shortlist'teki hisselerin sektör endekslerini çekip badge enjekte eder.
    2) Tüm BIST endekslerini (şehir hariç) çekip özet için saklar.
    Hiçbir sinyal elenmez — sadece bilgi.
    """
    from agent.sector_regime import (
        _ALL_INDEX_CODES, _ALL_INDICES,
        load_sector_map, fetch_index_regimes, get_ticker_sector_regime,
    )

    try:
        print("\n📊 Endeks & sektör regime analizi yapılıyor...")

        ticker_to_sector, sector_indexes = load_sector_map()

        # Shortlist'teki unique sektör endekslerini topla
        needed_sectors = set()
        for key in ('tier1', 'tier2', 'tier2a', 'tier2b', 'alsat', 'tavan', 'nw', 'rt', 'sbt'):
            for item in lists_dict.get(key, []):
                ticker = item[0]
                si = ticker_to_sector.get(ticker)
                if si:
                    needed_sectors.add(si)

        # Tüm endeksleri tek seferde çek (sektör + piyasa + tematik + katılım)
        all_codes = list(set(_ALL_INDEX_CODES) | needed_sectors)
        print(f"  [ENDEKS] {len(all_codes)} endeks çekiliyor (tvDatafeed)...")
        all_regimes = fetch_index_regimes(all_codes)

        if not all_regimes:
            print("  [ENDEKS] Endeks verisi çekilemedi — atlanıyor")
            return

        # Sektör alt kümesi (badge'ler için)
        sector_regimes = {k: v for k, v in all_regimes.items() if k in needed_sectors}
        al_count = sum(1 for r in sector_regimes.values() if r.get('in_trade', False))
        pasif_count = len(sector_regimes) - al_count
        print(f"  [SEKTÖR] {len(sector_regimes)} sektör: {al_count} AL, {pasif_count} pasif")

        # Grup özeti
        for group_name, group_codes in _ALL_INDICES.items():
            group_results = {c: all_regimes[c] for c in group_codes if c in all_regimes}
            al = sum(1 for v in group_results.values() if v.get('in_trade', False))
            total = len(group_results)
            if total > 0:
                print(f"  [{group_name.upper():8s}] {total} endeks: {al} AL, {total-al} pasif")

        # Her sinyal'e badge enjekte
        for key in ('tier1', 'tier2', 'tier2a', 'tier2b', 'alsat', 'tavan', 'nw', 'rt', 'sbt'):
            items = lists_dict.get(key, [])
            for i, (ticker, score, reasons, sig_or_meta) in enumerate(items):
                info = get_ticker_sector_regime(ticker, sector_regimes, ticker_to_sector)
                if not info:
                    continue

                # Badge enjekte
                if isinstance(reasons, list):
                    reasons.append(info['badge'])

                # Score adjust: AL aktif → +3, pasif → -2
                if info.get('in_trade', False):
                    new_score = score + 3
                else:
                    new_score = score - 2

                # Metadata enjekte
                if isinstance(sig_or_meta, dict):
                    sig_or_meta['sector_index'] = info['sector_index']
                    sig_or_meta['sector_regime'] = info['regime_label']
                    sig_or_meta['sector_in_trade'] = info.get('in_trade', False)

                items[i] = (ticker, new_score, reasons, sig_or_meta)

        # Store for message + HTML
        lists_dict['_sector_regimes'] = sector_regimes
        lists_dict['_index_regimes'] = all_regimes
        lists_dict['_sector_stats'] = {
            'al': al_count,
            'pasif': pasif_count,
            'no_data': len(needed_sectors) - len(sector_regimes),
        }

    except Exception as e:
        print(f"  [ENDEKS] Hata: {e} — atlanıyor")
        import traceback
        traceback.print_exc()


# ── Günlük Input ──
_AGENT_DIR = os.path.dirname(os.path.abspath(__file__))


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


def _extract_list_tickers(lists_dict):
    """Shortlist'teki benzersiz hisse kodlarını çıkar."""
    tickers = set()
    for key in ('tier1', 'tier2', 'tier2a', 'tier2b', 'alsat', 'tavan', 'nw', 'rt', 'sbt'):
        for item in lists_dict.get(key, []):
            tickers.add(item[0])
    return list(tickers)


def _fetch_matriks_pipeline(tickers, mkk_data_map, matriks_enabled, matriks_api_key, now):
    """Matriks kurumsal veri çek + SMS/ICE hesapla.

    Args:
        tickers: Sadece bu hisseler için veri çekilir
        mkk_data_map: MKK verisi
        matriks_enabled: Feature flag
        matriks_api_key: API key
        now: Zaman damgası

    Returns:
        (sms_scores, takas_data_map, ice_results, cost_data_map)
    """
    sms_scores = None
    takas_data_map = None
    ice_results = None
    cost_data_map = None

    if not matriks_enabled or not matriks_api_key:
        if not matriks_enabled:
            print("\n  ℹ️ MATRIKS_ENABLED=0 — kurumsal veri atlanıyor")
        elif not matriks_api_key:
            print("\n  ⚠️ MATRIKS_API_KEY eksik — kurumsal veri atlanıyor")
        return sms_scores, takas_data_map, ice_results, cost_data_map

    # Tarihsel flow: MATRIKS_HISTORY_DAYS env var (default 5, 0=kapalı)
    # 5g × 50 hisse = ~250 çağrı (~5dk). 10g = ~500 çağrı (~10dk).
    history_days = int(os.environ.get("MATRIKS_HISTORY_DAYS", "5"))
    include_history = history_days > 0

    mode_str = f", {history_days}g tarihçe" if include_history else ""
    print(f"\n⬡ Matriks kurumsal veri çekiliyor ({len(tickers)} hisse{mode_str})...")

    takas_history = None
    try:
        import concurrent.futures
        from agent.matriks_client import MatriksClient
        from agent.matriks_adapter import process_matriks_batch

        matriks_timeout = int(os.environ.get("MATRIKS_TIMEOUT", "600"))  # 10 dk hard limit
        client = MatriksClient()

        def _fetch():
            return client.fetch_batch(
                tickers, include_history=include_history, history_days=history_days)

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            future = ex.submit(_fetch)
            try:
                matriks_raw = future.result(timeout=matriks_timeout)
            except concurrent.futures.TimeoutError:
                print(f"  ⚠️ Matriks {matriks_timeout}s timeout — kısmi veriyle devam")
                matriks_raw = None

        if matriks_raw:
            takas_data_map, cost_data_map, takas_history = process_matriks_batch(matriks_raw)
            print(f"  Matriks: {len(matriks_raw)} hisse veri alındı")

            if takas_history:
                hist_days = len(takas_history)
                hist_tickers = len(set(t for d in takas_history.values() for t in d))
                print(f"  Tarihçe: {hist_days} gün × {hist_tickers} hisse")

            if cost_data_map:
                cost_vals = [c["value"] for c in cost_data_map.values() if c["value"] != "veri_yok"]
                print(f"  Maliyet avantajı: {len(cost_data_map)} hisse ({len(cost_vals)} cost verisi)")

            # SMS hesapla
            if takas_data_map:
                sms_scores = calc_batch_sms(None, takas_data_map, mkk_data_map)
                guclu = sum(1 for s in sms_scores.values() if s.score >= 45)
                dagitim = sum(1 for s in sms_scores.values() if s.score < 15)
                print(f"  SMS: {len(sms_scores)} hisse ({guclu}🟢 güçlü, {dagitim}🔴 dağıtım)")
    except Exception as e:
        print(f"  ⚠️ Matriks veri hatası: {e}")

    # ICE hesapla
    try:
        if takas_data_map or cost_data_map or takas_history:
            ice_results = calc_batch_ice(
                tickers, takas_history, takas_data_map, mkk_data_map,
                cost_data_map=cost_data_map)
            if ice_results:
                full = sum(1 for r in ice_results.values() if r.status == "ok")
                partial = sum(1 for r in ice_results.values() if r.status == "partial")
                no_hist = sum(1 for r in ice_results.values() if r.status in ("no_history", "cost_only"))
                guclu_ice = sum(1 for r in ice_results.values() if r.multiplier >= 1.15)
                red_ice = sum(1 for r in ice_results.values() if r.multiplier < 0.90)
                print(f"  ICE: {len(ice_results)} hisse (tam={full}, kısmi={partial}, snapshot={no_hist}"
                      f" | {guclu_ice}🟢 güçlü, {red_ice}🔴 dağıtım)")
    except Exception as e:
        print(f"  ⚠️ ICE hesaplama hatası: {e}")

    return sms_scores, takas_data_map, ice_results, cost_data_map


def _inject_ice_data(lists_dict, ice_results):
    """ICE verilerini sinyal dict'lerine inject et (HTML rapor için)."""
    if not ice_results:
        return
    for key in ('tier1', 'tier2', 'tier2a', 'tier2b', 'alsat', 'tavan', 'nw', 'rt', 'sbt'):
        for item in lists_dict.get(key, []):
            ticker = item[0]
            sig_or_meta = item[3] if len(item) > 3 else {}
            if isinstance(sig_or_meta, dict):
                ice = ice_results.get(ticker)
                if ice:
                    sig_or_meta['ice_mult'] = ice.multiplier
                    sig_or_meta['ice_icon'] = ice.icon
                    cr = ice.metrics.get("cost_ratio")
                    if cr:
                        sig_or_meta['cost_ratio'] = cr
                        ma = ice.labels.get("maliyet_avantaji")
                        sig_or_meta['cost_value'] = ma.value if ma else ""
                    streak = ice.metrics.get("streak_days", 0)
                    if streak >= 3:
                        sig_or_meta['streak_days'] = streak
                        sig_or_meta['streak_momentum'] = ice.metrics.get("streak_momentum", "")
                    dpoz = ice.metrics.get("position_change_pct")
                    if dpoz is not None:
                        sig_or_meta['position_change_pct'] = dpoz


def _compute_4_lists(latest_signals, confluence_results=None):
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
    # ZAYIF çıkarıldı (backtest WR %33). Öncelik: CMB > GUCLU > BILESEN > DONUS
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
        # ZAYIF çıkarıldı — backtest: 1G WR %33, 3G WR %0 (session_20260321)
        if sig_type in ('GUCLU', 'GÜÇLÜ') and 30 <= rs <= 60:
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
            # DÖNÜŞ hard gate — RVOL>5 eleme (backtest: %33 WR)
            _d_rvol = s.get('rvol', 0) or 0
            if _d_rvol > 5:
                continue  # hard eleme — aşırı spike, çöp sinyal
            passes = True
            # ALTIN badge — ATR<3% + Part=3 (backtest: %63-70 WR)
            _d_atr_pct = s.get('atr_pct', 99) or 99
            _d_part = s.get('part_score', 0) or 0
            if _d_atr_pct < 3 and _d_part == 3:
                tier_label = '🥇DÖNÜŞ'
                score = 150 + q  # ALTIN DÖNÜŞ — GÜÇLÜ seviyesine yakın
            else:
                tier_label = 'DÖNÜŞ'
                score = 50 + q
        # PB ve düşük kaliteli sinyaller dahil değil

        if not passes:
            continue

        # OE cezası — düşük OE daha kaliteli (LILAK OE=0 > REEDR OE=5)
        oe_val = int(oe) if oe != '' else 0
        if oe_val >= 5:
            score -= 30  # Aşırı uzamış — pullback bekle
        elif oe_val >= 4:
            score -= 15  # Geç kalma riski
        elif oe_val >= 3:
            score -= 5

        # Decay etiketi: DONUS=swing, diğer=3G hold
        decay = '🔄SW' if sig_type in ('DONUS', 'DÖNÜŞ') else '⏳3G'
        reasons = [decay, tier_label, f"Q={q}", f"RS={rs:.0f}", f"MACD={'+'if macd>0 else ''}{macd:.4f}"]
        if oe != '':
            reasons.append(f"OE={oe}")
        if rr != '':
            reasons.append(f"R:R={rr}")

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

        # Skor hesaplama — data-driven (5Y backtest, N=23763)
        score = skor * 5  # Base yarıya (skor prediktif değil, streak önemli)

        # A) Streak (en güçlü prediktör — 1G WR +18pp)
        if seri >= 3:
            score += 150   # 77.2% 1G, 71.0% 3G
        elif seri >= 2:
            score += 80    # 66.0% 1G, 59.5% 3G

        # B) Skor zone (counter-intuitive ama 5Y data-driven)
        if skor <= 49:
            score += 40    # Non-kilitli: 75.3% 1G — EN İYİ
        elif 60 <= skor <= 79:
            score -= 60    # Yarı-kilitli: 44.2% 1G — EN KÖTÜ
        elif 50 <= skor <= 59:
            score -= 30    # 50.0% 1G — baseline altı
        # skor >= 80: +0 (59.5% 1G — genel baseline)

        # C) Volume (güçlendirilmiş — düşük vol = kaliteli tavan)
        if vol < 1.0:
            score += 50
        elif vol < 1.5:
            score += 30
        elif vol < 2.0:
            pass           # baseline
        elif vol < 3.0:
            score -= 20
        elif vol < 5.0:
            score -= 50
        elif vol < 10:
            score -= 80
        elif vol < 20:
            score -= 120
        else:
            score -= 200   # Premium dışı — kovalanmaz

        # D) CMF (küçültülmüş, tüm sinyallere uygula)
        if cmf is not None and cmf > 0:
            score += int(cmf * 50)

        reasons = ['⚡1G']  # Tavan = fast decay (intraday çıkış şart)
        if seri >= 3:
            reasons.append(f"🔥seri:{seri}")
        elif seri >= 2:
            reasons.append(f"seri:{seri}")
        if skor <= 49:
            reasons.append(f"▲skor:{skor}")      # Momentum
        elif 60 <= skor <= 79:
            reasons.append(f"⚠️Y-KLT:{skor}")   # Yarı-kilitli uyarı
        elif skor >= 80:
            reasons.append(f"🔒skor:{skor}")     # Kilitli (bilgi)
        else:
            reasons.append(f"skor:{skor}")
        reasons.append(f"vol:{vol:.1f}x")
        if vol > 10:
            reasons.append("⚠️KOVALANMAZ")
        if rs != 0:
            reasons.append(f"RS{rs:+.0f}%")
        if cmf is not None:
            reasons.append(f"CMF{cmf:+.2f}")

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

        reasons = ['⚡1G']  # NW daily = fast trade (1G WR %64, 3G decay)
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
        # Badge varsa entry_score ≥ 2 yeterli, yoksa ≥ 3 gerekli
        if badge:
            if entry_score < 2:
                continue  # Badge var ama çok zayıf giriş
        else:
            if entry_score < 3:
                continue  # Badge yok → güçlü giriş şart (FIRSAT 3/4+)
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

        # Hacim-donus tier
        vol_tier = s.get('vol_tier', '')
        vol_tier_icon = s.get('vol_tier_icon', '')

        # ELE filtresi — kotu hacim profili, listeye almiyoruz
        if vol_tier == 'ELE':
            continue

        # Tier bonusu
        if vol_tier == 'ALTIN':
            score += 50
        elif vol_tier == 'GUMUS':
            score += 30
        elif vol_tier == 'BRONZ':
            score += 10

        # Decay: H+AL = swing, diğer = 3G hold
        decay = '🔄SW' if badge == 'H+AL' else '⏳3G'
        reasons = [decay]
        # Vol tier badge (en basta)
        if vol_tier_icon:
            reasons.insert(0, f"{vol_tier_icon}{vol_tier}")
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

        rt_items.append((s['ticker'], score, reasons, s))

    rt_items.sort(key=lambda x: -x[1])

    # ── LİSTE 5: SBT Breakout (taze kırılma) ──
    _BUCKET_SCORE = {'A+': 200, 'A': 150, 'B': 100, 'C': 50}
    sbt_items = []
    for s in latest_signals:
        if s.get('screener') != 'sbt':
            continue
        bucket = s.get('sbt_bucket', '') or ''
        if bucket == 'X':
            continue
        if bucket == '?':
            bucket = ''  # ML modeli yokken — bucket bilinmiyor
        strength = s.get('quality', 0) or 0
        ml_prob = s.get('sbt_ml_prob', 0) or 0

        score = _BUCKET_SCORE.get(bucket, 50) + strength * 20 + int(ml_prob * 100)
        prob_pct = int(ml_prob * 100)
        reasons = ['⚡1G']
        if bucket:
            reasons.append(f'SBT:{bucket}')
        reasons += [f'💪{strength}/4', f'ML%{prob_pct}']

        sbt_items.append((s['ticker'], score, reasons, s))

    sbt_items.sort(key=lambda x: -x[1])

    # ── Çapraz Çakışma Tagging ──
    list_data = {'alsat': alsat_items, 'tavan': tavan_items, 'nw': nw_items, 'rt': rt_items, 'sbt': sbt_items}
    _LIST_SHORT = {'alsat': 'AS', 'tavan': 'TVN', 'nw': 'NW', 'rt': 'RT', 'sbt': 'SBT'}
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
        for list_name in ('alsat', 'tavan', 'nw', 'rt', 'sbt'):
            for t, sc, reas, sig in list_data[list_name]:
                if t != ticker:
                    continue
                in_lists.append(list_name)
                if list_name == 'sbt':
                    sb = sig.get('sbt_bucket', '')
                    if sb == 'A+':
                        quality += 35
                    elif sb == 'A':
                        quality += 25
                    elif sb == 'B':
                        quality += 15
                    else:
                        quality += 5
                    quality += (sig.get('quality', 0) or 0) * 3  # strength*3
                elif list_name == 'rt':
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
                    # Hacim-donus tier bonusu
                    vt = sig.get('vol_tier', '')
                    if vt == 'ALTIN':
                        quality += 25
                    elif vt == 'GUMUS':
                        quality += 15
                    elif vt == 'BRONZ':
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
                    elif sig_type in ('GUCLU', 'GÜÇLÜ'):
                        quality += 30  # Kalite filtrelerini geçmiş
                    elif sig_type in ('BILESEN', 'BİLEŞEN'):
                        quality += 25
                    elif sig_type in ('DONUS', 'DÖNÜŞ'):
                        # ALTIN DÖNÜŞ: ATR<3% + Part=3 (backtest: %63-70 WR)
                        _d_atr = sig.get('atr_pct', 99) or 99
                        _d_part = sig.get('part_score', 0) or 0
                        if _d_atr < 3 and _d_part == 3:
                            quality += 25  # ALTIN DÖNÜŞ
                        else:
                            quality += 10  # Normal DÖNÜŞ
                    else:
                        quality += 10
                    q = sig.get('quality', 0) or 0
                    quality += q // 10  # Q=100→10, Q=75→7
                elif list_name == 'tavan':
                    skor = sig.get('skor', 0) or 0
                    vol = sig.get('volume_ratio', 0) or 0
                    seri = sig.get('streak', 0) or 0
                    # Streak-dominant quality (5Y data-driven)
                    if seri >= 3:
                        quality += 40
                    elif seri >= 2:
                        quality += 25
                    else:
                        quality += 10
                    # Skor zone
                    if skor <= 49:
                        quality += 15   # Non-kilitli momentum
                    elif 60 <= skor <= 79:
                        quality -= 15   # Yarı-kilitli en kötü zone
                    # Volume
                    if vol < 1.5:
                        quality += 10
                    elif vol > 3.0:
                        quality -= 10
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

    # Premium overlap ikililer — backtest kanıtlı
    _PREMIUM_OVERLAPS = [
        {'rt', 'tavan'}, {'nw', 'tavan'}, {'nw', 'rt'}, {'alsat', 'nw'},
        {'sbt', 'rt'}, {'sbt', 'nw'},
    ]

    tier1 = []
    for ticker, lists in ticker_list_count.items():
        if len(lists) < 2:
            continue
        quality, in_lists = _overlap_quality(ticker)
        ls = set(in_lists)
        # Saf AS+RT 2-li çakışma → Tier 1'e ekleme (WR %37-48, session_20260321)
        if ls == {'alsat', 'rt'}:
            continue

        # OE>=4 → Tier 1 yasağı — AMA kilitli tavan (skor≥50) için OE yarı ağırlık
        # Tavan serisinde OE tek başına güvenilir uzama göstergesi değil;
        # kilitli veya yakın tavanlarda ağırlığı düşürülmeli
        has_high_oe = False
        is_kilitli_tavan = False
        tavan_vol = 0
        for l in in_lists:
            for t, sc, reas, sig in list_data[l]:
                if t == ticker:
                    if l == 'tavan':
                        skor_t = sig.get('skor', 0) or 0
                        if skor_t >= 50:
                            is_kilitli_tavan = True
                        tavan_vol = sig.get('volume_ratio', 0) or 0
                    else:
                        oe_raw = sig.get('oe', 0)
                        if oe_raw != '' and oe_raw is not None and int(oe_raw or 0) >= 4:
                            has_high_oe = True
                    break
        if has_high_oe:
            if is_kilitli_tavan and tavan_vol <= 3:
                pass  # Kilitli tavan + düşük vol → OE eleme yapma (sadece uyarı)
            elif is_kilitli_tavan:
                # Kilitli ama yüksek vol → OE=5+ hâlâ engeller
                # Sadece en aşırı OE (5+) kontrol et
                oe_max = 0
                for l in in_lists:
                    for t, sc, reas, sig in list_data[l]:
                        if t == ticker and l != 'tavan':
                            oe_raw = sig.get('oe', 0)
                            if oe_raw != '' and oe_raw is not None:
                                oe_max = max(oe_max, int(oe_raw or 0))
                            break
                if oe_max >= 5:
                    continue  # Kilitli tavan bile olsa OE=5+ → yasak
            else:
                continue  # Normal sinyal — OE>=4 Tier 1 yasağı

        # Horizon mismatch tespiti: TVN=⚡1G, AS/RT=⏳3G/🔄SW
        # Farklı horizon'lu overlap → tactical (premium değil)
        _1G_LISTS = {'tavan', 'nw', 'sbt'}      # ⚡1G horizon
        _SWING_LISTS = {'alsat', 'rt'}   # ⏳3G / 🔄SW horizon
        has_1g = bool(ls & _1G_LISTS)
        has_swing = bool(ls & _SWING_LISTS)
        horizon_mismatch = has_1g and has_swing and len(in_lists) == 2
        # AS+TVN saf 2-li her zaman tactical (horizon mismatch + zayıf overlap)
        is_tactical = horizon_mismatch and 'tavan' in ls
        if is_tactical:
            quality -= 20  # Tactical ceza — premium altında sırala

        # Kalite eşiği: sadece anlamlı çakışmalar
        if quality < 40:
            continue

        list_tags = "+".join(_LIST_SHORT.get(l, l) for l in sorted(in_lists))
        # Overlap decay etiketi — horizon mismatch varsa 1G'ye düşür
        if len(in_lists) >= 3:
            ol_decay = '⚡1G'  # 3+ liste = FAST_DECAY
        elif horizon_mismatch:
            ol_decay = '⚡1G'  # Horizon mismatch = kısa horizon kazanır
        elif ls in _PREMIUM_OVERLAPS:
            ol_decay = '⏳3G'  # Premium ikililer = HOLD_TO_3D
        else:
            ol_decay = '⚡1G'  # Bilinmeyen = kısa horizon
        # Teknik+yapısal çakışma etiketi
        has_tech = bool({'alsat', 'tavan'} & ls)
        has_struct = bool({'nw', 'rt'} & ls)
        ty_tag = " 🔀T+Y" if has_tech and has_struct else ""
        tactical_tag = " 🔶TAKTİK" if is_tactical else ""
        reasons_all = [f"{ol_decay} ⚡{list_tags} [{quality}p]{ty_tag}{tactical_tag}"]
        for l in sorted(in_lists):
            for t, sc, reas, sig in list_data[l]:
                if t == ticker:
                    short_reason = f"[{_LIST_SHORT[l]}] {' '.join(reas[:4])}"
                    reasons_all.append(short_reason)
                    break
        # RT vol_tier bilgisini meta'ya aktar
        _vt = ''
        _vt_icon = ''
        if 'rt' in ls:
            for t, sc, reas, sig in list_data['rt']:
                if t == ticker:
                    _vt = sig.get('vol_tier', '')
                    _vt_icon = sig.get('vol_tier_icon', '')
                    break
        tier1.append((ticker, quality, reasons_all, {
            'overlap_count': len(in_lists),
            'in_lists': in_lists,
            'tactical': is_tactical,
            'vol_tier': _vt,
            'vol_tier_icon': _vt_icon,
        }))

    # ── Gevşek RT çakışma: güçlü sinyal + RT badge (F serbest) ──
    # Normal RT filtresi (F≥3 OE≤2) çok sıkı — güçlü AS/NW/TVN sinyali varsa
    # badge + TAZE/2.DALGA + OE≤3 yetsin, F serbest
    tier1_tickers = {t for t, _, _, _ in tier1}
    strong_tickers = {}  # ticker → (list_name, reasons, sig)
    # Güçlü AS sinyalleri (kalite filtrelerinden geçen)
    for t, sc, reas, sig in alsat_items:
        st = sig.get('signal_type', '')
        if st in ('GUCLU', 'GÜÇLÜ', 'CMB', 'CMB+', 'BILESEN', 'BİLEŞEN'):
            strong_tickers[t] = ('alsat', reas, sig)
    # Streak/momentum tavan (5Y data-driven)
    for t, sc, reas, sig in tavan_items:
        seri = sig.get('streak', 0) or 0
        skor = sig.get('skor', 0) or 0
        vol = sig.get('volume_ratio', 0) or 0
        if seri >= 2 or (skor <= 49 and vol < 2.0):
            strong_tickers.setdefault(t, ('tavan', reas, sig))
    # NW D+W
    for t, sc, reas, sig in nw_items:
        if sig.get('dw_overlap'):
            strong_tickers.setdefault(t, ('nw', reas, sig))
    # SBT A+/A
    for t, sc, reas, sig in sbt_items:
        sb = sig.get('sbt_bucket', '')
        if sb in ('A+', 'A'):
            strong_tickers.setdefault(t, ('sbt', reas, sig))

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
        entry_score = int(s.get('quality', 0) or 0)
        # Badge varsa entry_score ≥ 2 yeterli, yoksa ≥ 3 gerekli (çakışma gevşek)
        if badge:
            if entry_score < 2:
                continue
        else:
            if entry_score < 3:
                continue
        window = s.get('entry_window', '')
        if window not in ('TAZE', '2.DALGA'):
            continue  # Window hala gerekli
        oe = int(s.get('oe', 0) or 0)
        if oe > 3:
            continue  # OE≤3 (gevşetilmiş)
        cmf = s.get('cmf', 0) or 0

        # Saf AS+RT gevşek çakışma → Tier 1'e ekleme (WR %37-48)
        # Sadece 3+ liste overlap'ta AS+RT gevşek izin verilir
        src_name, src_reas, src_sig = strong_tickers[ticker]
        if src_name == 'alsat' and ticker not in {t for t, _, _, _ in nw_items} \
                and ticker not in {t for t, _, _, _ in tavan_items}:
            continue

        # Kalite skoru
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
            f"⏳3G ⚡{src_tag}+RT [{quality}p]{ty_tag}",
            f"[{src_tag}] {src_short}",
            f"[RT↓] {rt_reasons}",  # ↓ = gevşek filtre
        ]
        # RT vol_tier
        _rvt = s.get('vol_tier', '')
        _rvt_icon = s.get('vol_tier_icon', '')
        tier1.append((ticker, quality, reasons_all, {
            'overlap_count': 2,
            'in_lists': [src_name, 'rt'],
            'relaxed': True,
            'vol_tier': _rvt,
            'vol_tier_icon': _rvt_icon,
        }))
        tier1_tickers.add(ticker)

    tier1.sort(key=lambda x: -x[1])

    # ── Tier 2: Her listeden en kaliteli tekil hisseler ──
    # Score >= 100 soft gate, max 3 per list
    # Horizon'a göre 2 alt gruba ayrılır: tier2a (⚡1G tactical), tier2b (⏳3G/🔄SW swing-lite)
    _TIER2_MIN_SCORE = 100
    _TIER2_PER_LIST = 3
    tier1_tickers = {t for t, _, _, _ in tier1}
    tier2_all = []
    for list_name in ('nw', 'rt', 'alsat', 'tavan', 'sbt'):
        items = list_data[list_name]
        count = 0
        for ticker, score, reasons, sig in items:
            if ticker in tier1_tickers:
                continue
            if score < _TIER2_MIN_SCORE:
                continue
            if ticker in {t for t, _, _, _ in tier2_all}:
                continue
            tag = _LIST_SHORT[list_name]
            tier2_all.append((ticker, score, [f"[{tag}]"] + reasons, sig))
            count += 1
            if count >= _TIER2_PER_LIST:
                break

    # Horizon split: decay etiketi ilk reasons'tan çek
    tier2a = []  # ⚡1G — tactical (tavan, NW daily)
    tier2b = []  # ⏳3G / 🔄SW — swing-lite (RT, AS)
    for item in tier2_all:
        _, _, reas, _ = item
        # İlk non-tag reason'da decay etiketini bul
        decay_1g = any('⚡1G' in r for r in reas)
        if decay_1g:
            tier2a.append(item)
        else:
            tier2b.append(item)

    result = dict(list_data)
    result['tier1'] = tier1
    result['tier2a'] = tier2a
    result['tier2b'] = tier2b
    # Backward compat: birleşik tier2 (toplam)
    result['tier2'] = tier2a + tier2b
    return result


def _push_priority_tickers(lists_dict):
    """Tüm listelerden unique ticker'ları priority_tickers.json olarak GitHub Pages'e push et.
    VDS takas scraper bu dosyayı okuyarak hangi hisseler için veri çekeceğini bilir.
    Enriched format: tickers + details (ML breakdown)."""
    import json
    seen = set()
    tickers = []
    details = []
    for list_name in ('tier1', 'tier2', 'alsat', 'tavan', 'nw', 'rt', 'sbt'):
        for item in lists_dict.get(list_name, []):
            ticker = item[0]
            if ticker not in seen:
                seen.add(ticker)
                tickers.append(ticker)
                # ML detail extraction
                sig = item[3] if len(item) > 3 else {}
                if isinstance(sig, dict):
                    detail = {
                        'ticker': ticker,
                        'rule_score': sig.get('_rule_score', item[1]),
                        'ml_short': sig.get('ml_score_short'),
                        'ml_swing': sig.get('ml_score_swing'),
                        'ml_flag': sig.get('ml_effect', 'neutral'),
                        'sbt_bucket': sig.get('sbt_bucket'),
                        'sector_index': sig.get('sector_index'),
                        'sector_regime': sig.get('sector_regime'),
                        'final_score': item[1],
                    }
                else:
                    detail = {
                        'ticker': ticker,
                        'rule_score': item[1],
                        'ml_short': None,
                        'ml_swing': None,
                        'ml_flag': 'neutral',
                        'sbt_bucket': None,
                        'sector_index': None,
                        'sector_regime': None,
                        'final_score': item[1],
                    }
                details.append(detail)

    payload = json.dumps({
        "updated_at": datetime.now(_TZ_TR).strftime("%Y-%m-%dT%H:%M:%S+03:00"),
        "total": len(tickers),
        "tickers": tickers,
        "details": details,
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
            ma = ice.labels.get("maliyet_avantaji")
            t_s = t.value[:3] if t else "?"
            b_s = b.value[:3] if b else "?"
            kv_s = kv.value[:4] if kv else "?"
            dt20 = ice.metrics.get("takas_20_change")
            dt_str = f" ΔT20={dt20:+,.0f}" if dt20 is not None else ""

            # Maliyet + trend ek bilgileri
            extras = []
            cr = ice.metrics.get("cost_ratio")
            if cr:
                _CR_TAG = {"guclu": "💪", "avantaj": "✅", "notr": "", "risk": "⚠", "yuksek_risk": "🔻"}
                ma_tag = _CR_TAG.get(ma.value, "") if ma else ""
                extras.append(f"r={cr:.2f}{ma_tag}")
            streak = ice.metrics.get("streak_days", 0)
            if streak >= 3:
                mom = ice.metrics.get("streak_momentum", "")
                mom_short = "💪" if mom == "GÜÇLÜ" else ""
                extras.append(f"SM{streak}g{mom_short}")
            dpoz = ice.metrics.get("position_change_pct")
            if dpoz is not None and abs(dpoz) >= 0.5:
                extras.append(f"Δpoz={dpoz:+.1f}%")
            ext_str = f" [{' '.join(extras)}]" if extras else ""

            parts.append(f"×{ice.multiplier:.2f}{ice.icon} T={t_s} B={b_s} KV={kv_s}{dt_str}{ext_str}")

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

    # Takas: net akış by tip (haftalık varsa, yoksa günlük)
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
                g_lot = k.get('Günlük Fark') or k.get('gunluk_fark') or 0
                lot = h_lot if h_lot != 0 else g_lot
                tip = classify_kurum_sms(name)
                net_by_tip[tip] = net_by_tip.get(tip, 0) + lot
                if lot > top_buyer_lot:
                    top_buyer_name = name
                    top_buyer_lot = lot

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


def _build_shortlist_message(lists_dict,
                             sms_scores=None, takas_data=None, mkk_data=None,
                             ice_results=None):
    """4 kaynak bazlı shortlist'i Telegram mesajı olarak formatla.
    3-Katmanlı ML: dual badge, rerank bölümü, SBT taktik, filtre bölümü.
    Her hissenin altında kompakt SM (takas+MKK) satırı gösterir."""
    now = datetime.now(_TZ_TR)

    has_sm = bool(takas_data or mkk_data or ice_results)
    _LIST_SHORT = {'alsat': 'AS', 'tavan': 'TVN', 'nw': 'NW', 'rt': 'RT', 'sbt': 'SBT'}

    lines = [
        f"<b>⬡ NOX Ön Analiz — {now.strftime('%d.%m.%Y %H:%M')}</b>",
    ]

    def _sig_line(ticker, reasons, sig):
        """Sinyal satırını formatla — SBT + dual ML + gate + BRK + sektör bilgisi."""
        # Kaynak tag'leri topla
        source_tags = []
        sbt_bucket = ''
        gate_tag = ''
        ml_badge = ''
        sector_badge = ''
        other_reasons = []

        for r in reasons:
            if r.startswith('SBT:'):
                sbt_bucket = r.split(':')[1]
            elif r.startswith('🤖'):
                ml_badge = r
            elif 'soft gate' in r.lower():
                gate_tag = 'soft gate'
            elif r.startswith('✅') and 'X' in r:
                sector_badge = r
            elif r.startswith('⚠️') and '↓' in r:
                sector_badge = r
            else:
                other_reasons.append(r)

        # BRK badge (sig dict'ten)
        brk_badge = ''
        if isinstance(sig, dict):
            if sig.get('brk_avoid'):
                brk_badge = '⛔ALMA'
            elif sig.get('breakout_tier') == 'top5':
                brk_badge = 'BRK🎯T5'
            elif sig.get('breakout_tier') == 'top10':
                brk_badge = 'BRK⚡T10'

        # Kaynak bilgisi
        parts = [f"<b>{ticker}</b>"]
        reason_compact = " ".join(other_reasons[:4])
        if reason_compact:
            parts.append(f"— {reason_compact}")
        if sbt_bucket:
            parts.append(f"SBT({sbt_bucket})")
        if ml_badge:
            parts.append(f"| {ml_badge}")
        if gate_tag:
            parts.append(f"| {gate_tag}")
        if brk_badge:
            parts.append(f"| {brk_badge}")
        if sector_badge:
            parts.append(sector_badge)

        return " ".join(parts)

    # ── A. ML Destekli Ana Shortlist ──
    # Scanner HTML linkleri
    _nox_base = os.environ.get("GH_PAGES_BASE_URL", "https://aalpkk.github.io/nox-signals").rstrip("/")
    _bist_base = os.environ.get("BIST_PAGES_BASE_URL", "https://aalpkk.github.io/bist-signals").rstrip("/")
    _SCAN_URL = {
        'alsat': f'{_bist_base}/',
        'tavan': f'{_bist_base}/tavan.html',
        'nw': f'{_nox_base}/nox_v3_weekly.html',
        'rt': f'{_nox_base}/regime_transition.html',
        'sbt': f'{_nox_base}/smart_breakout.html',
    }

    # ── 1. AL/SAT Tarama ──
    alsat_list = lists_dict.get('alsat', [])
    if alsat_list:
        lines.append("")
        lines.append(f'📋 <a href="{_SCAN_URL["alsat"]}"><b>1. AL/SAT Tarama</b></a> (Q skoru sıralı, {len(alsat_list)} sinyal)')
        lines.append("")
        for i, (ticker, score, reasons, sig) in enumerate(alsat_list[:15], 1):
            lines.append(f"{i}. {_sig_line(ticker, reasons, sig)}")
            if has_sm:
                sm_line = _build_sm_inline(ticker, takas_data, mkk_data, sms_scores, ice_results)
                if sm_line:
                    lines.append(sm_line)

    # ── 2. Tavan Tarayıcı ──
    tavan_list = lists_dict.get('tavan', [])
    if tavan_list:
        lines.append("")
        lines.append(f'🔺 <a href="{_SCAN_URL["tavan"]}"><b>2. Tavan Tarayıcı</b></a> (skor/vol sıralı, {len(tavan_list)} hisse)')
        lines.append("")
        for i, (ticker, score, reasons, sig) in enumerate(tavan_list[:15], 1):
            lines.append(f"{i}. {_sig_line(ticker, reasons, sig)}")
            if has_sm:
                sm_line = _build_sm_inline(ticker, takas_data, mkk_data, sms_scores, ice_results)
                if sm_line:
                    lines.append(sm_line)

    # ── 3. NW Pivot AL ──
    nw_list = lists_dict.get('nw', [])
    if nw_list:
        lines.append("")
        lines.append(f'📊 <a href="{_SCAN_URL["nw"]}"><b>3. NW Pivot AL</b></a> (günlük gate açık, {len(nw_list)} sinyal)')
        lines.append("")
        for i, (ticker, score, reasons, sig) in enumerate(nw_list[:15], 1):
            lines.append(f"{i}. {_sig_line(ticker, reasons, sig)}")
            if has_sm:
                sm_line = _build_sm_inline(ticker, takas_data, mkk_data, sms_scores, ice_results)
                if sm_line:
                    lines.append(sm_line)

    # ── 4. Regime Transition ──
    rt_list = lists_dict.get('rt', [])
    if rt_list:
        lines.append("")
        lines.append(f'⚡ <a href="{_SCAN_URL["rt"]}"><b>4. Regime Transition</b></a> (F≥3 OE≤2, {len(rt_list)} sinyal)')
        lines.append("")
        for i, (ticker, score, reasons, sig) in enumerate(rt_list[:15], 1):
            lines.append(f"{i}. {_sig_line(ticker, reasons, sig)}")
            if has_sm:
                sm_line = _build_sm_inline(ticker, takas_data, mkk_data, sms_scores, ice_results)
                if sm_line:
                    lines.append(sm_line)

    # ── 5. SBT Breakout ──
    sbt_list = lists_dict.get('sbt', [])
    if sbt_list:
        lines.append("")
        lines.append(f'🚀 <a href="{_SCAN_URL["sbt"]}"><b>5. SBT Breakout</b></a> (taze kırılma, {len(sbt_list)} sinyal)')
        lines.append("")
        for i, (ticker, score, reasons, sig) in enumerate(sbt_list[:15], 1):
            lines.append(f"{i}. {_sig_line(ticker, reasons, sig)}")
            if has_sm:
                sm_line = _build_sm_inline(ticker, takas_data, mkk_data, sms_scores, ice_results)
                if sm_line:
                    lines.append(sm_line)
    elif not lists_dict.get('_sbt_data'):
        lines.append("")
        lines.append("ℹ️ SBT ML: veri yok, breakout-quality bonus uygulanmadı")

    # ── 6. ML Güçlü (≥0.50) ──
    ml_strong = []
    _ml_seen = set()
    for key in ('alsat', 'tavan', 'nw', 'rt', 'sbt'):
        for ticker, score, reasons, sig in lists_dict.get(key, []):
            if ticker in _ml_seen:
                continue
            if not isinstance(sig, dict):
                continue
            s_val = sig.get('ml_score_short')
            w_val = sig.get('ml_score_swing')
            if s_val is None and w_val is None:
                continue
            best = max(s_val or 0, w_val or 0)
            if best >= 0.50:
                _ml_seen.add(ticker)
                s_pct = int(s_val * 100) if s_val else 0
                w_pct = int(w_val * 100) if w_val else 0
                # Hangi listelerde var
                src = []
                for ln in ('alsat', 'tavan', 'nw', 'rt', 'sbt'):
                    for t2, _, _, _ in lists_dict.get(ln, []):
                        if t2 == ticker:
                            src.append({'alsat': 'AS', 'tavan': 'TVN', 'nw': 'NW', 'rt': 'RT', 'sbt': 'SBT'}[ln])
                            break
                ml_strong.append((ticker, best, s_pct, w_pct, src))
    ml_strong.sort(key=lambda x: -x[1])
    if ml_strong:
        lines.append("")
        lines.append(f"🤖 <b>6. ML Güçlü</b> (skor≥50, {len(ml_strong)} hisse)")
        lines.append("")
        for i, (ticker, best, s_pct, w_pct, src) in enumerate(ml_strong[:15], 1):
            src_str = "+".join(src)
            lines.append(f"{i}. <b>{ticker}</b> — S{s_pct}·W{w_pct} [{src_str}]")

    # ── Tier 1: Çapraz Çakışmalar ──
    tier1 = lists_dict.get('tier1', [])
    if tier1:
        lines.append("")
        lines.append(f"🔥 <b>Tier 1 — Çakışmalar</b> ({len(tier1)} hisse)")
        lines.append("")
        for i, (ticker, score, reasons, _) in enumerate(tier1[:15], 1):
            reasons_str = " | ".join(reasons)
            lines.append(f"{i}. <b>{ticker}</b> {reasons_str}")
            if has_sm:
                sm_line = _build_sm_inline(ticker, takas_data, mkk_data, sms_scores, ice_results)
                if sm_line:
                    lines.append(sm_line)

    # ── Tier 2A: Tactical (⚡1G) ──
    tier2a = lists_dict.get('tier2a', [])
    if tier2a:
        lines.append("")
        lines.append(f"⚡ <b>Tier 2A — Tactical ⚡1G</b> ({len(tier2a)} hisse)")
        lines.append("")
        for i, (ticker, score, reasons, _) in enumerate(tier2a[:15], 1):
            lines.append(f"{i}. {_sig_line(ticker, reasons, _)}")
            if has_sm:
                sm_line = _build_sm_inline(ticker, takas_data, mkk_data, sms_scores, ice_results)
                if sm_line:
                    lines.append(sm_line)

    # ── Tier 2B: Swing-Lite (⏳3G/🔄SW) ──
    tier2b = lists_dict.get('tier2b', [])
    if tier2b:
        lines.append("")
        lines.append(f"⭐ <b>Tier 2B — Swing-Lite ⏳3G/🔄SW</b> ({len(tier2b)} hisse)")
        lines.append("")
        for i, (ticker, score, reasons, _) in enumerate(tier2b[:15], 1):
            lines.append(f"{i}. {_sig_line(ticker, reasons, _)}")
            if has_sm:
                sm_line = _build_sm_inline(ticker, takas_data, mkk_data, sms_scores, ice_results)
                if sm_line:
                    lines.append(sm_line)

    # ── B. ML Rerank Değişimi ──
    rank_changes = lists_dict.get('_ml_rank_changes', [])
    if rank_changes:
        up_changes = [r for r in rank_changes if r['delta'] > 0]
        down_changes = [r for r in rank_changes if r['delta'] < 0]
        up_changes.sort(key=lambda x: -x['delta'])
        down_changes.sort(key=lambda x: x['delta'])

        lines.append("")
        lines.append(f"🧠 <b>ML Rerank Değişimi</b>")
        if up_changes:
            lines.append("  ↑ Yükselenler:")
            for r in up_changes[:5]:
                lines.append(f"    <b>{r['ticker']}</b> [{r['list_tag']}] {r['old_rank']}→{r['new_rank']} (+{r['delta']})")
        if down_changes:
            lines.append("  ↓ Düşenler:")
            for r in down_changes[:5]:
                lines.append(f"    <b>{r['ticker']}</b> [{r['list_tag']}] {r['old_rank']}→{r['new_rank']} ({r['delta']})")

    # ── C. Filtreyle Elenenler ──
    ml_filtered = lists_dict.get('_ml_filtered', [])
    if ml_filtered:
        lines.append("")
        lines.append(f"⚠️ <b>Filtreyle Elenenler</b> ({len(ml_filtered)} hisse — rule güçlü ama ML zayıf)")
        for f in ml_filtered[:6]:
            tag = _LIST_SHORT.get(f['list'], f['list'])
            lines.append(f"  <b>{f['ticker']}</b> [{tag}] rule:{f['rule_score']}p — {f['reason']}")

    # ── D. Endeks & Sektör Durumu ──
    all_regimes = lists_dict.get('_index_regimes', {})
    if all_regimes:
        from agent.sector_regime import _ALL_INDICES
        lines.append("")
        lines.append("📊 <b>Endeks & Sektör Durumu</b>")

        group_labels = {
            'piyasa': '📈 Piyasa',
            'sektor': '🏭 Sektör',
            'tematik': '🏷 Tematik',
            'katilim': '☪️ Katılım',
        }
        for group, codes in _ALL_INDICES.items():
            group_data = {c: all_regimes[c] for c in codes if c in all_regimes}
            if not group_data:
                continue
            al = sorted(k for k, v in group_data.items() if v.get('in_trade', False))
            pasif = sorted(k for k, v in group_data.items() if not v.get('in_trade', False))
            label = group_labels.get(group, group)
            parts = []
            for s in al:
                parts.append(f"{s}✅")
            for s in pasif:
                parts.append(f"{s}⚠️")
            lines.append(f"{label}: {' '.join(parts)}")

    return "\n".join(lines)


# ═══════════════════════════════════════════
# TEMPLATE-BASED BRİFİNG (AI yerine kod ile üretim)
# ═══════════════════════════════════════════

def _build_template_briefing(macro_result, signal_summary, lists_dict,
                              latest_signals, news_items=None,
                              confluence_results=None):
    """Kod tabanlı brifing — AI olmadan tüm verileri template ile formatla.

    AI sadece limit order stratejisi için çağrılır (ayrı fonksiyon).
    Bu fonksiyon: makro + tarama + shortlist + çelişki + sektör + strateji üretir.
    """
    now = datetime.now(_TZ_TR)
    _LIST_SHORT = {'alsat': 'AS', 'tavan': 'TVN', 'nw': 'NW', 'rt': 'RT', 'sbt': 'SBT'}

    lines = [f"<b>⬡ NOX Brifing — {now.strftime('%d.%m.%Y %H:%M')}</b>", ""]

    # ── 1. KÜRESEL ORTAM ──
    if macro_result:
        regime = macro_result.get('regime', 'N/A')
        risk = macro_result.get('risk_score', 0)
        regime_emoji = {
            'GÜÇLÜ_RISK_ON': '🟢🟢', 'RISK_ON': '🟢', 'NÖTR': '⚪',
            'RISK_OFF': '🔴', 'GÜÇLÜ_RISK_OFF': '🔴🔴',
        }
        lines.append(f"🌍 <b>Küresel Ortam: {regime_emoji.get(regime, '')} {regime}</b> (skor: {risk})")

        # Kategori bazlı kısa özet
        snapshot = macro_result.get('snapshot', [])
        by_cat = {}
        for s in snapshot:
            by_cat.setdefault(s.get('category', ''), []).append(s)

        for cat in ['BIST', 'US', 'FX', 'Emtia']:
            items = by_cat.get(cat, [])
            if not items:
                continue
            parts = []
            for item in items:
                price = item.get('price')
                if price is None:
                    continue
                chg_1d = item.get('chg_1d')
                trend = item.get('trend', '')
                icon = '↑' if trend == 'UP' else ('↓' if trend == 'DOWN' else '→')
                chg_str = f"{chg_1d:+.1f}%" if chg_1d is not None else ""
                parts.append(f"{icon}{item['name']}:{price:,.1f}({chg_str})")
            if parts:
                lines.append(f"  <b>{cat}</b>: {' '.join(parts)}")

        # Kategori rejimleri
        cat_reg = macro_result.get('category_regimes', {})
        if cat_reg:
            cat_parts = []
            for cat in ['BIST', 'US', 'FX', 'Emtia', 'Kripto', 'Faiz']:
                cr = cat_reg.get(cat)
                if cr:
                    cat_parts.append(f"{cat}={cr['regime']}")
            if cat_parts:
                lines.append(f"  Kategori: {', '.join(cat_parts)}")

        # Rejim sinyalleri (kısa)
        sinyaller = macro_result.get('signals', [])
        if sinyaller:
            lines.append(f"  Sinyaller: {' | '.join(sinyaller[:4])}")

        lines.append("")

    # Scanner HTML linkleri (tüm bölümlerde kullanılır)
    _nox_base2 = os.environ.get("GH_PAGES_BASE_URL", "https://aalpkk.github.io/nox-signals").rstrip("/")
    _bist_base2 = os.environ.get("BIST_PAGES_BASE_URL", "https://aalpkk.github.io/bist-signals").rstrip("/")
    _SCAN_URL2 = {
        'alsat': f'{_bist_base2}/',
        'tavan': f'{_bist_base2}/tavan.html',
        'nw': f'{_nox_base2}/nox_v3_weekly.html',
        'rt': f'{_nox_base2}/regime_transition.html',
        'sbt': f'{_nox_base2}/smart_breakout.html',
    }

    # ── 2. TARAMA ÖZETİ ──
    total = signal_summary.get('total', 0)
    if lists_dict:
        n_as = len(lists_dict.get('alsat', []))
        n_tvn = len(lists_dict.get('tavan', []))
        n_nw = len(lists_dict.get('nw', []))
        n_rt = len(lists_dict.get('rt', []))
        n_sbt = len(lists_dict.get('sbt', []))
        n_tier1 = len(lists_dict.get('tier1', []))
        n_tier2a = len(lists_dict.get('tier2a', []))
        n_tier2b = len(lists_dict.get('tier2b', []))
        core_total = n_as + n_tvn + n_nw + n_rt + n_sbt
        lines.append(f"📋 <b>Tarama</b>: {total} sinyal → {core_total} shortlist "
                     f'(<a href="{_SCAN_URL2["alsat"]}">AS:{n_as}</a> '
                     f'<a href="{_SCAN_URL2["tavan"]}">TVN:{n_tvn}</a> '
                     f'<a href="{_SCAN_URL2["nw"]}">NW:{n_nw}</a> '
                     f'<a href="{_SCAN_URL2["rt"]}">RT:{n_rt}</a> '
                     f'<a href="{_SCAN_URL2["sbt"]}">SBT:{n_sbt}</a>)')
        lines.append(f"  Tier1:{n_tier1} çakışma | Tier2A:{n_tier2a} taktik | Tier2B:{n_tier2b} swing")
    lines.append("")

    # ── Fiyat bilgisi helper ──
    def _collect_price(ticker):
        """Ticker için fiyat/ATR/stop/target bilgisi topla."""
        pi = {}
        for ln in ('alsat', 'rt', 'nw', 'tavan', 'sbt'):
            for t, sc, reas, sig in lists_dict.get(ln, []):
                if t != ticker or not isinstance(sig, dict):
                    continue
                if not pi.get('close'):
                    ep = sig.get('entry_price', 0)
                    if ep:
                        pi['close'] = ep
                if not pi.get('atr_pct'):
                    atr = sig.get('atr_pct', 0)
                    if atr:
                        pi['atr_pct'] = atr
                if not pi.get('stop'):
                    sp = sig.get('stop_price', 0)
                    if sp:
                        pi['stop'] = sp
                if not pi.get('target'):
                    tp = sig.get('target_price', 0)
                    if tp:
                        pi['target'] = tp
                if not pi.get('oe') and pi.get('oe') != 0:
                    oe = sig.get('oe', '')
                    if oe != '':
                        pi['oe'] = oe
        return pi

    def _price_tag(ticker):
        pi = _collect_price(ticker)
        if not pi.get('close'):
            return ""
        parts = [f"₺{pi['close']:.2f}"]
        if pi.get('atr_pct'):
            parts.append(f"ATR%{pi['atr_pct']:.1f}")
        if pi.get('stop'):
            parts.append(f"SL:{pi['stop']:.2f}")
        if pi.get('target'):
            parts.append(f"TP:{pi['target']:.2f}")
        return " ".join(parts)

    # ── 3. SHORTLIST ANALİZ ──

    # Tier 1 — çakışmalar
    tier1 = lists_dict.get('tier1', [])
    if tier1:
        lines.append(f"🔥 <b>Tier 1 — Çakışmalar</b> ({len(tier1)} hisse)")
        lines.append("")
        for i, (ticker, quality, reasons, meta) in enumerate(tier1[:15], 1):
            in_lists = meta.get('in_lists', []) if isinstance(meta, dict) else []
            list_tags = "+".join(_LIST_SHORT.get(l, l) for l in sorted(in_lists))
            relaxed = " [RT↓]" if (isinstance(meta, dict) and meta.get('relaxed')) else ""
            # Metrik detayları
            metrics = []
            for l in sorted(in_lists):
                for t, sc, reas, sig in lists_dict.get(l, []):
                    if t != ticker:
                        continue
                    if l == 'rt':
                        badge = sig.get('badge', '')
                        if badge:
                            metrics.append(f"[{badge}]")
                    elif l == 'nw':
                        if sig.get('dw_overlap'):
                            metrics.append("D+W")
                        delta = sig.get('delta_pct')
                        if delta is not None:
                            metrics.append(f"δ={delta:.1f}%")
                    elif l == 'alsat':
                        st = sig.get('signal_type', '')
                        if st:
                            metrics.append(st)
                    elif l == 'tavan':
                        skor = sig.get('skor', 0) or 0
                        if skor:
                            metrics.append(f"tvn={skor}")
                    break
            # ML badge
            ml_badge = ""
            for r in reasons:
                if r.startswith('🤖'):
                    ml_badge = r
                    break
            pt = _price_tag(ticker)
            metric_str = " ".join(metrics)
            line = f"{i}. <b>{ticker}</b> [{list_tags}]{relaxed} Q={quality} {metric_str}"
            if ml_badge:
                line += f" {ml_badge}"
            if pt:
                line += f" | {pt}"
            lines.append(line)
        lines.append("")

    # Tier 2B — swing
    tier2b = lists_dict.get('tier2b', [])
    if tier2b:
        lines.append(f"⭐ <b>Tier 2B — Swing ⏳3G/🔄SW</b> ({len(tier2b)} hisse)")
        lines.append("")
        for i, (ticker, score, reasons, sig) in enumerate(tier2b[:15], 1):
            reason_parts = [r for r in reasons if not r.startswith('🤖')][:3]
            ml_badge = next((r for r in reasons if r.startswith('🤖')), "")
            pt = _price_tag(ticker)
            line = f"{i}. <b>{ticker}</b> [{score}p] {' '.join(reason_parts)}"
            if ml_badge:
                line += f" {ml_badge}"
            if pt:
                line += f" | {pt}"
            lines.append(line)
        lines.append("")

    # Tier 2A — tactical
    tier2a = lists_dict.get('tier2a', [])
    if tier2a:
        lines.append(f"⚡ <b>Tier 2A — Tactical ⚡1G</b> ({len(tier2a)} hisse)")
        lines.append("")
        for i, (ticker, score, reasons, sig) in enumerate(tier2a[:15], 1):
            reason_parts = [r for r in reasons if not r.startswith('🤖')][:3]
            ml_badge = next((r for r in reasons if r.startswith('🤖')), "")
            pt = _price_tag(ticker)
            line = f"{i}. <b>{ticker}</b> [{score}p] {' '.join(reason_parts)}"
            if ml_badge:
                line += f" {ml_badge}"
            if pt:
                line += f" | {pt}"
            lines.append(line)
        lines.append("")

    # ── 4. DİKKAT / ÇELİŞKİ ──
    if latest_signals and lists_dict:
        sat_tickers = {s['ticker'] for s in latest_signals
                       if s.get('direction') == 'SAT' or s.get('karar') == 'SAT'}
        shortlist_set = set()
        for ln in ('tier1', 'tier2a', 'tier2b', 'alsat', 'tavan', 'nw', 'rt'):
            for item in lists_dict.get(ln, []):
                shortlist_set.add(item[0])
        conflicts = shortlist_set & sat_tickers
        if conflicts:
            lines.append(f"⚠️ <b>Çelişki</b> ({len(conflicts)} hisse — AL+SAT aynı anda)")
            for ticker in sorted(conflicts):
                sat_src = []
                for s in latest_signals:
                    if s['ticker'] == ticker and (s.get('direction') == 'SAT' or s.get('karar') == 'SAT'):
                        scr = SCREENER_NAMES.get(s.get('screener', ''), s.get('screener', '?'))
                        sat_src.append(scr)
                al_src = []
                for ln in ('alsat', 'tavan', 'nw', 'rt', 'sbt'):
                    for t, sc, reas, sig in lists_dict.get(ln, []):
                        if t == ticker:
                            al_src.append(_LIST_SHORT.get(ln, ln))
                lines.append(f"  {ticker}: AL={'+'.join(al_src)} ↔ SAT={','.join(sat_src)} → BEKLE")
            lines.append("")

    # SAT sinyalleri (kısa)
    if latest_signals:
        sat_signals = [s for s in latest_signals
                       if s.get('direction') == 'SAT' or s.get('karar') == 'SAT']
        if sat_signals:
            lines.append(f"🔴 <b>SAT Sinyalleri</b> ({len(sat_signals)})")
            sat_str = ", ".join(s['ticker'] for s in sat_signals[:15])
            lines.append(f"  {sat_str}")
            lines.append("")

    # ── 5. ENDEKS & SEKTÖR DURUMU ──
    all_regimes = lists_dict.get('_index_regimes', {})
    if all_regimes:
        from agent.sector_regime import _ALL_INDICES
        lines.append("📊 <b>Endeks & Sektör Durumu</b>")
        group_labels = {
            'piyasa': '📈 Piyasa', 'sektor': '🏭 Sektör',
            'tematik': '🏷 Tematik', 'katilim': '☪️ Katılım',
        }
        for group, codes in _ALL_INDICES.items():
            group_data = {c: all_regimes[c] for c in codes if c in all_regimes}
            if not group_data:
                continue
            al = sorted(k for k, v in group_data.items() if v.get('in_trade', False))
            pasif = sorted(k for k, v in group_data.items() if not v.get('in_trade', False))
            label = group_labels.get(group, group)
            parts = []
            for s in al:
                parts.append(f"{s}✅")
            for s in pasif:
                parts.append(f"{s}⚠️")
            lines.append(f"  {label}: {' '.join(parts)}")
        lines.append("")

    # ── ML Güçlü (≥0.50) ──
    ml_strong2 = []
    _ml_seen2 = set()
    for key2 in ('alsat', 'tavan', 'nw', 'rt', 'sbt'):
        for ticker, score, reasons, sig in lists_dict.get(key2, []):
            if ticker in _ml_seen2 or not isinstance(sig, dict):
                continue
            s_val = sig.get('ml_score_short')
            w_val = sig.get('ml_score_swing')
            if s_val is None and w_val is None:
                continue
            best = max(s_val or 0, w_val or 0)
            if best >= 0.50:
                _ml_seen2.add(ticker)
                s_pct = int(s_val * 100) if s_val else 0
                w_pct = int(w_val * 100) if w_val else 0
                src = []
                for ln2 in ('alsat', 'tavan', 'nw', 'rt', 'sbt'):
                    for t2, _, _, _ in lists_dict.get(ln2, []):
                        if t2 == ticker:
                            src.append(_LIST_SHORT.get(ln2, ln2))
                            break
                ml_strong2.append((ticker, best, s_pct, w_pct, src))
    ml_strong2.sort(key=lambda x: -x[1])
    if ml_strong2:
        lines.append(f"🤖 <b>ML Güçlü</b> (skor≥50, {len(ml_strong2)} hisse)")
        for i, (ticker, best, s_pct, w_pct, src) in enumerate(ml_strong2[:15], 1):
            src_str = "+".join(src)
            lines.append(f"  {i}. <b>{ticker}</b> — S{s_pct}·W{w_pct} [{src_str}]")
        lines.append("")

    # ── ML Rerank Değişimi ──
    rank_changes = lists_dict.get('_ml_rank_changes', [])
    if rank_changes:
        up_changes = sorted([r for r in rank_changes if r['delta'] > 0], key=lambda x: -x['delta'])
        down_changes = sorted([r for r in rank_changes if r['delta'] < 0], key=lambda x: x['delta'])
        if up_changes or down_changes:
            lines.append("🧠 <b>ML Rerank Değişimi</b>")
            if up_changes:
                up_str = " ".join(f"{r['ticker']}↑{r['delta']}" for r in up_changes[:5])
                lines.append(f"  Yükselen: {up_str}")
            if down_changes:
                dn_str = " ".join(f"{r['ticker']}↓{abs(r['delta'])}" for r in down_changes[:5])
                lines.append(f"  Düşen: {dn_str}")
            lines.append("")

    # ── Filtreyle Elenenler ──
    ml_filtered = lists_dict.get('_ml_filtered', [])
    if ml_filtered:
        lines.append(f"⚠️ <b>ML Filtre</b> ({len(ml_filtered)} elendi)")
        for f in ml_filtered[:6]:
            tag = _LIST_SHORT.get(f['list'], f['list'])
            lines.append(f"  {f['ticker']}[{tag}] rule:{f['rule_score']}p — {f['reason']}")
        lines.append("")

    # ── Birikim→Breakout Tahmini ──
    breakout_alerts = lists_dict.get('_breakout_alerts', [])
    if breakout_alerts:
        top5 = [a for a in breakout_alerts if a.get('tier') == 'top5']
        top10 = [a for a in breakout_alerts if a.get('tier') == 'top10']

        lines.append("🎯 <b>Birikim→Breakout (ML)</b>")

        if top5:
            lines.append("  <b>▸ Yüksek Güven</b> (Top 5)")
            for i, ba in enumerate(top5, 1):
                ticker = ba['ticker']
                fusion = int(ba.get('fusion_score', 0) * 100)
                tvn_pct = int(ba.get('tavan_prob', 0) * 100) if ba.get('tavan_prob') else 0
                ralli_pct = int(ba.get('rally_prob', 0) * 100) if ba.get('rally_prob') else 0
                ml_s = int(ba.get('ml_s_score', 0) * 100) if ba.get('ml_s_score') else 0
                xref = f" ★SL" if ba.get('in_shortlist') else ""
                lines.append(f"  {i}. <b>{ticker}</b> F{fusion} "
                             f"(TVN:{tvn_pct} RLI:{ralli_pct} S:{ml_s}){xref}")

        if top10:
            lines.append("  <b>▸ İzle</b> (Top 6-10)")
            for i, ba in enumerate(top10, 6):
                ticker = ba['ticker']
                fusion = int(ba.get('fusion_score', 0) * 100)
                tvn_pct = int(ba.get('tavan_prob', 0) * 100) if ba.get('tavan_prob') else 0
                ralli_pct = int(ba.get('rally_prob', 0) * 100) if ba.get('rally_prob') else 0
                xref = f" ★SL" if ba.get('in_shortlist') else ""
                lines.append(f"  {i}. {ticker} F{fusion} "
                             f"(TVN:{tvn_pct} RLI:{ralli_pct}){xref}")
        lines.append("")

    # ── Limit Order TP ──
    limit_tp = _compute_limit_tp_signals(lists_dict)
    if limit_tp:
        lines.append(f"🎯 <b>Limit Order TP</b> ({len(limit_tp)} adet)")
        lines.append("  Filtre: score≥400 · streak≥2 · ML S≥58")
        lines.append("  Strateji: -%1.5 limit giriş → %4 TP → 1g hold")
        lines.append("  Backtest: WR %93 · PF 9.3 · Ort +%3.26/trade")
        lines.append("")
        for i, sig in enumerate(limit_tp, 1):
            lines.append(
                f"  {i}. <b>{sig['ticker']}</b> — "
                f"Limit: {sig['limit_price']:.2f} TL | "
                f"TP: {sig['tp_price']:.2f} TL | "
                f"+%{sig['net_pct']:.2f} net | "
                f"streak={sig['streak']} | S{sig['ml_s']}"
            )
        lines.append("")

    # ── 6. STRATEJİ ──
    if macro_result:
        regime = macro_result.get('regime', 'NÖTR')
        if regime in ('RISK_OFF', 'GÜÇLÜ_RISK_OFF'):
            lines.append("📌 <b>Strateji</b>: Defansif — küçük pozisyon, seçici giriş, "
                         "rejim düzelene kadar Tier1 odaklı.")
        elif regime in ('RISK_ON', 'GÜÇLÜ_RISK_ON'):
            lines.append("📌 <b>Strateji</b>: Momentum takibi — Tier1+Tier2B swing, "
                         "sektör uyumlu hisseler öncelikli.")
        else:
            lines.append("📌 <b>Strateji</b>: Dengeli — Tier1 çakışma öncelikli, "
                         "taktik fırsatlar seçici değerlendir.")

    # ── Haberler ──
    if news_items:
        lines.append("")
        lines.append(f"📰 <b>Haberler</b> ({len(news_items)})")
        for item in news_items[:5]:
            title = item.get('title', '')[:80]
            lines.append(f"  • {title}")

    return "\n".join(lines)


def _compute_limit_tp_signals(lists_dict):
    """Tavan sinyallerinden limit order TP adaylarını hesapla.

    Backtest'te kanıtlanmış strateji: score≥400 + streak≥2 + ML S≥0.58
    → kapanıştan -%1.5 limit giriş → +%4 TP → 1 gün hold.
    1 yıl backtest: 148 trade, WR %93.2, PF 9.3, ort +%3.26/trade.
    """
    _LIST_SHORT = {'alsat': 'AS', 'tavan': 'TVN', 'nw': 'NW', 'rt': 'RT', 'sbt': 'SBT'}
    candidates = []
    seen = set()

    # Tavan öncelikli, sonra diğer listeler
    for list_key in ('tavan', 'alsat', 'nw', 'rt', 'sbt'):
        for ticker, score, reasons, sig in lists_dict.get(list_key, []):
            if ticker in seen or not isinstance(sig, dict):
                continue
            # Filtre: score≥400 AND streak≥2 AND ml_score_short≥0.58
            if score < 400:
                continue
            streak = sig.get('streak', 0) or 0
            if streak < 2:
                continue
            ml_s = sig.get('ml_score_short')
            if ml_s is None or ml_s < 0.58:
                continue
            entry_price = sig.get('entry_price', 0)
            if not entry_price or entry_price <= 0:
                continue

            seen.add(ticker)
            limit_price = round(entry_price * 0.985, 2)
            tp_price = round(limit_price * 1.04, 2)
            net_pct = round(((tp_price / entry_price) - 1) * 100, 2)

            candidates.append({
                'ticker': ticker,
                'entry_close': entry_price,
                'limit_price': limit_price,
                'tp_price': tp_price,
                'net_pct': net_pct,
                'streak': streak,
                'ml_s': round(ml_s * 100),
                'score': score,
                'list_source': _LIST_SHORT.get(list_key, list_key),
            })

    # En yüksek streak'e göre sırala, eşitlikte score'a bak
    candidates.sort(key=lambda x: (-x['streak'], -x['score']))
    return candidates


def _generate_limit_order_ai(lists_dict):
    """AI ile sadece limit order / giriş stratejisi üret.

    Tüm shortlist verisini gönderir, AI sadece entry strategy yazar.
    Template brifingden ayrı çağrılır — maliyet ve hata riski düşük.
    """
    from agent.claude_client import single_prompt

    # Tier1 + Tier2 hisselerini topla
    tickers_data = []
    for ticker, quality, reasons, meta in lists_dict.get('tier1', [])[:15]:
        in_lists = meta.get('in_lists', []) if isinstance(meta, dict) else []
        pi = _get_price_info_for_ai(ticker, lists_dict)
        if pi:
            tickers_data.append(f"  {ticker} [Tier1 Q={quality} {'+'.join(in_lists)}] {pi}")

    for tier_name, label in [('tier2b', 'Tier2B'), ('tier2a', 'Tier2A')]:
        for ticker, score, reasons, sig in lists_dict.get(tier_name, [])[:10]:
            pi = _get_price_info_for_ai(ticker, lists_dict)
            if pi:
                tickers_data.append(f"  {ticker} [{label} {score}p] {pi}")

    if not tickers_data:
        return ""

    prompt = """Her hisse icin sabah acilisa GIRIS STRATEJISI yaz. SADECE strateji — piyasa yorumu yapma.

## FORMAT (Telegram — HTML tag)
Her hisse 3-4 satir:
<b>TICKER</b>
  Normal: limit [fiyat]
  Gap-up: [strateji]
  Gap-down: [strateji]
  SL:[fiyat] TP1:[fiyat] TP2:[fiyat]

## KURALLAR
- ATR% kullan: <3 dar spread, >5 genis spread
- OE=0 momentum taze → gap-up olasi → limit kapanis+ATR%/2
- OE>=3 uzamis → pullback olasi → limit kapanis-1%
- SL: stop verideyse kullan, yoksa 1.5xATR alti
- TP: target verideyse kullan, yoksa TP1=2xATR TP2=3xATR
- Kisa yaz, tablo KULLANMA

## HISSELER
"""
    prompt += "\n".join(tickers_data)
    return single_prompt(prompt, max_tokens=4096)


def _get_price_info_for_ai(ticker, lists_dict):
    """Ticker fiyat bilgisini AI prompt için formatla."""
    pi = {}
    for ln in ('alsat', 'rt', 'nw', 'tavan', 'sbt'):
        for t, sc, reas, sig in lists_dict.get(ln, []):
            if t != ticker or not isinstance(sig, dict):
                continue
            if not pi.get('close'):
                ep = sig.get('entry_price', 0)
                if ep:
                    pi['close'] = ep
            if not pi.get('atr_pct'):
                atr = sig.get('atr_pct', 0)
                if atr:
                    pi['atr_pct'] = atr
            if not pi.get('stop'):
                sp = sig.get('stop_price', 0)
                if sp:
                    pi['stop'] = sp
            if not pi.get('target'):
                tp = sig.get('target_price', 0)
                if tp:
                    pi['target'] = tp
            if not pi.get('oe') and pi.get('oe') != 0:
                oe = sig.get('oe', '')
                if oe != '':
                    pi['oe'] = oe
    if not pi.get('close'):
        return ""
    parts = [f"fiyat={pi['close']:.2f}"]
    if pi.get('atr_pct'):
        parts.append(f"ATR%={pi['atr_pct']:.1f}")
    if pi.get('stop'):
        parts.append(f"SL={pi['stop']:.2f}")
    if pi.get('target'):
        parts.append(f"TP={pi['target']:.2f}")
    if pi.get('oe') is not None and pi.get('oe') != '':
        parts.append(f"OE={pi['oe']}")
    return " ".join(parts)


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

    # 2. MKK verisi (GitHub Pages — çakışma skorunda kullanılır)
    mkk_data_map = None
    try:
        mkk_json = fetch_mkk_data()
        if mkk_json and mkk_json.get('data'):
            mkk_data_map = mkk_json['data']
            print(f"  MKK: {len(mkk_data_map)} hisse ({mkk_json.get('extracted_at', '?')[:10]})")
    except Exception as e:
        print(f"  ⚠️ MKK veri hatası: {e}")

    # 3. Çakışma analizi (makro olmadan da çalışır)
    print("\n⬡ Çakışma analizi...")
    confluence_results = calc_all_confluence(
        latest_signals, None, min_score=1, mkk_data=mkk_data_map)
    print(f"  {len(confluence_results)} hisse çakışma skoru ≥ 1")

    # ── Matriks kurumsal veri değişkenleri ──
    sms_scores = None
    takas_data_map = None
    takas_extracted_at = None
    ice_results = None
    cost_data_map = None

    matriks_enabled = os.environ.get("MATRIKS_ENABLED", "0") == "1"
    matriks_api_key = os.environ.get("MATRIKS_API_KEY", "")

    # ── SHORTLIST MODE: sadece öncelikli hisse listesi oluştur ──
    if shortlist_only:
        print("\n📋 4 kaynak bazlı öncelikli liste oluşturuluyor...")
        lists_dict = _compute_4_lists(latest_signals, confluence_results)

        # Taban risk overlay (1Y fiyat verisi ile)
        _taban_risk_overlay(lists_dict)

        # ML overlay v2 (feature flag kontrollü, 3-katmanlı)
        _ml_overlay_v2(lists_dict, latest_signals)

        # Sektör regime overlay (soft gate + badge)
        _sector_regime_overlay(lists_dict)

        # Shortlistteki benzersiz hisseleri çıkar (Matriks sadece bunlar için çekilecek)
        shortlist_tickers = _extract_list_tickers(lists_dict)

        # 2b. Matriks kurumsal veri (sadece shortlist hisseleri — ~50 hisse, ~2.5 dk)
        sms_scores, takas_data_map, ice_results, cost_data_map = _fetch_matriks_pipeline(
            shortlist_tickers, mkk_data_map, matriks_enabled, matriks_api_key, now)

        # ICE verilerini sinyallere inject et
        _inject_ice_data(lists_dict, ice_results)

        _LIST_LABELS = [
            ('alsat', '📋 AL/SAT Tarama'),
            ('tavan', '🔺 Tavan Tarayıcı'),
            ('nw', '📊 NW Pivot AL (günlük gate açık)'),
            ('rt', '⚡ Regime Transition (F≥3 OE≤2)'),
            ('sbt', '🚀 SBT Breakout'),
        ]
        core_total = sum(len(lists_dict.get(k, [])) for k in ('alsat', 'tavan', 'nw', 'rt', 'sbt'))
        print(f"  Toplam: {core_total} sinyal (5 liste)\n")
        for key, label in _LIST_LABELS:
            items = lists_dict.get(key, [])
            if not items:
                print(f"  ── {label} (0) ──\n")
                continue
            print(f"  ── {label} ({len(items)}) ──")
            for i, (ticker, score, reasons, _sig) in enumerate(items[:15], 1):
                print(f"  {i:2d}. {ticker:6s} [{score:3d}p] — {' '.join(reasons[:6])}")
            print()

        # Tier 1 + Tier 2A/2B
        tier1 = lists_dict.get('tier1', [])
        tier2a = lists_dict.get('tier2a', [])
        tier2b = lists_dict.get('tier2b', [])
        if tier1:
            print(f"  ── 🔥 Tier 1: Çakışmalar ({len(tier1)}) ──")
            for i, (ticker, score, reasons, _) in enumerate(tier1[:15], 1):
                print(f"  {i:2d}. {ticker:6s} [{score:3d}p] — {' | '.join(reasons[:4])}")
            print()
        if tier2a:
            print(f"  ── ⚡ Tier 2A: Tactical ⚡1G ({len(tier2a)}) ──")
            for i, (ticker, score, reasons, _) in enumerate(tier2a[:15], 1):
                print(f"  {i:2d}. {ticker:6s} [{score:3d}p] — {' '.join(reasons[:5])}")
            print()
        if tier2b:
            print(f"  ── ⭐ Tier 2B: Swing-Lite ({len(tier2b)}) ──")
            for i, (ticker, score, reasons, _) in enumerate(tier2b[:15], 1):
                print(f"  {i:2d}. {ticker:6s} [{score:3d}p] — {' '.join(reasons[:5])}")
            print()
        if notify:
            msg = _build_shortlist_message(lists_dict,
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

    # Çakışmayı makro ile tekrar hesapla (SMS/ICE shortlist sonrası eklenecek)
    confluence_results = calc_all_confluence(
        latest_signals, macro_result, min_score=1, mkk_data=mkk_data_map)

    # 4. Shortlist ÖNCE hesapla (AI'a shortlist verisini göndermek için)
    print("\n📋 4 liste + haberler hesaplanıyor...")
    lists_dict = _compute_4_lists(latest_signals, confluence_results)

    # Taban risk overlay (1Y fiyat verisi ile)
    _taban_risk_overlay(lists_dict)

    # ML overlay v2 (feature flag kontrollü, 3-katmanlı)
    _ml_overlay_v2(lists_dict, latest_signals)

    # Sektör regime overlay (soft gate + badge)
    _sector_regime_overlay(lists_dict)

    # 4a. Matriks kurumsal veri (sadece shortlist hisseleri — ~50 hisse, ~2.5 dk)
    shortlist_tickers = _extract_list_tickers(lists_dict)
    sms_scores, takas_data_map, ice_results, cost_data_map = _fetch_matriks_pipeline(
        shortlist_tickers, mkk_data_map, matriks_enabled, matriks_api_key, now)

    # ICE verilerini sinyallere inject et (HTML rapor için)
    _inject_ice_data(lists_dict, ice_results)

    # Limit Order TP sinyalleri hesapla
    limit_tp = _compute_limit_tp_signals(lists_dict)
    lists_dict['_limit_tp'] = limit_tp
    if limit_tp:
        print(f"  🎯 Limit TP: {len(limit_tp)} sinyal")

    news_items = fetch_market_news()
    if news_items:
        print(f"  📰 {len(news_items)} haber çekildi")

    # 4b. Template brifing üret (kod tabanlı — AI yorumu yok)
    print("\n📝 Template brifing oluşturuluyor...")
    briefing_text = _build_template_briefing(
        macro_result, signal_summary, lists_dict,
        latest_signals, news_items=news_items,
        confluence_results=confluence_results)
    print(f"  ✅ Template brifing ({len(briefing_text)} karakter)")

    # 4c. AI sadece limit order stratejisi için (use_ai=True ise)
    limit_order_text = ""
    if use_ai:
        print("\n🤖 AI limit order stratejisi üretiliyor...")
        try:
            limit_order_text = _generate_limit_order_ai(lists_dict)
            if limit_order_text:
                print(f"  ✅ Limit order stratejisi ({len(limit_order_text)} karakter)")
                briefing_text += "\n\n" + "💰 <b>Giriş Stratejisi (AI)</b>\n\n" + limit_order_text
        except Exception as e:
            print(f"  ⚠️ Limit order AI hatası: {e}")

    # 5. HTML rapor oluştur — shortlist + çelişki verisini hazırla
    print("\n📊 HTML rapor oluşturuluyor...")
    _shortlist_tickers = set()
    if lists_dict:
        for _ln in ('tier1', 'tier2', 'alsat', 'tavan', 'nw', 'rt', 'sbt'):
            for _item in lists_dict.get(_ln, []):
                _shortlist_tickers.add(_item[0])
    _sat_tickers = {s['ticker'] for s in latest_signals
                    if s.get('direction') == 'SAT' or s.get('karar') == 'SAT'}
    html = generate_briefing_html(
        briefing_text, macro_result, confluence_results, signal_summary,
        lists_dict=lists_dict, news_items=news_items,
        shortlist_tickers=_shortlist_tickers, sat_tickers=_sat_tickers,
        limit_order_text=limit_order_text)
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
                           latest_signals=None, lists_dict=None):
    """Claude API ile brifing üret. Shortlist bazlı veri bağlamı."""
    from agent.claude_client import single_prompt

    _LIST_SHORT = {'alsat': 'AS', 'tavan': 'TVN', 'nw': 'NW', 'rt': 'RT'}

    data_context = []
    data_context.append(f"Tarih: {datetime.now(_TZ_TR).strftime('%d.%m.%Y')}")

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

    # ── Sinyal Pipeline Özeti (shortlist bazlı) ──
    total_signals = signal_summary.get('total', 0)
    if lists_dict:
        n_as = len(lists_dict.get('alsat', []))
        n_tvn = len(lists_dict.get('tavan', []))
        n_nw = len(lists_dict.get('nw', []))
        n_rt = len(lists_dict.get('rt', []))
        n_sbt = len(lists_dict.get('sbt', []))
        n_tier1 = len(lists_dict.get('tier1', []))
        n_tier2 = len(lists_dict.get('tier2', []))
        shortlist_total = n_as + n_tvn + n_nw + n_rt + n_sbt
        data_context.append(f"\n## 📊 Sinyal Pipeline ({total_signals} tarandı → {shortlist_total} shortlist geçti)")
        data_context.append(f"  AL/SAT: {n_as} sinyal (filtre: karar=AL, tip≠ERKEN/ZAYIF, kalite gate)")
        data_context.append(f"  Tavan: {n_tvn} sinyal (filtre: skor formülü, dedup)")
        data_context.append(f"  NW Pivot: {n_nw} sinyal (filtre: BUGÜN + gate=AÇIK, daily)")
        data_context.append(f"  RT: {n_rt} sinyal (filtre: TAZE/2.DALGA + badge + F≥3 + OE≤2)")
        data_context.append(f"  SBT: {n_sbt} sinyal (filtre: bars≤3 + OPEN/TRAIL + bucket≠X)")
        n_tier2a = len(lists_dict.get('tier2a', []))
        n_tier2b = len(lists_dict.get('tier2b', []))
        data_context.append(f"  Tier 1 Çakışma: {n_tier1} hisse (2+ liste, quality≥40)")
        data_context.append(f"  Tier 2A Tactical ⚡1G: {n_tier2a} hisse (tavan/NW/SBT, intraday çıkış)")
        data_context.append(f"  Tier 2B Swing-Lite: {n_tier2b} hisse (RT/AS, 3G-5G tutma)")
    else:
        data_context.append(f"\n## Scanner Sinyalleri: {total_signals} toplam")
        for scr, stats in signal_summary.get('screeners', {}).items():
            name = SCREENER_NAMES.get(scr, scr)
            data_context.append(f"  {name}: {stats['total']} ({stats.get('AL', 0)} AL, {stats.get('SAT', 0)} SAT)")

    # ── Ticker → fiyat bilgisi toplama (tüm listelerden) ──
    def _collect_price_info(ticker):
        """Ticker için fiyat/ATR/stop/target bilgisi topla (ilk bulunan)."""
        price_info = {}
        for ln in ('alsat', 'rt', 'nw', 'tavan', 'sbt'):
            for t, sc, reas, sig in lists_dict.get(ln, []):
                if t != ticker or not isinstance(sig, dict):
                    continue
                if not price_info.get('close'):
                    ep = sig.get('entry_price', 0)
                    if ep:
                        price_info['close'] = ep
                if not price_info.get('atr_pct'):
                    atr = sig.get('atr_pct', 0)
                    if atr:
                        price_info['atr_pct'] = atr
                if not price_info.get('stop'):
                    sp = sig.get('stop_price', 0)
                    if sp:
                        price_info['stop'] = sp
                if not price_info.get('target'):
                    tp = sig.get('target_price', 0)
                    if tp:
                        price_info['target'] = tp
                if not price_info.get('oe'):
                    oe = sig.get('oe', '')
                    if oe != '':
                        price_info['oe'] = oe
        return price_info

    def _price_str(ticker):
        """Fiyat bilgisi string — data context için."""
        pi = _collect_price_info(ticker)
        if not pi.get('close'):
            return ""
        parts = [f"fiyat={pi['close']:.2f}"]
        if pi.get('atr_pct'):
            parts.append(f"ATR%={pi['atr_pct']:.1f}")
        if pi.get('stop'):
            parts.append(f"SL={pi['stop']:.2f}")
        if pi.get('target'):
            parts.append(f"TP={pi['target']:.2f}")
        if pi.get('oe') is not None and pi.get('oe') != '':
            parts.append(f"OE={pi['oe']}")
        return " ".join(parts)

    # ── Tier 1: Çapraz Çakışmalar (detaylı) ──
    if lists_dict and lists_dict.get('tier1'):
        tier1 = lists_dict['tier1']
        data_context.append(f"\n## ⭐ ÖNCELİKLİ: Tier 1 Çakışmalar ({len(tier1)} hisse)")
        data_context.append("  Kalite filtrelerinden geçmiş 2+ liste çakışma. Quality skoru sıralı.")
        for ticker, quality, reasons, meta in tier1:
            in_lists = meta.get('in_lists', []) if isinstance(meta, dict) else []
            list_tags = "+".join(_LIST_SHORT.get(l, l) for l in sorted(in_lists))
            relaxed = " [RT↓]" if (isinstance(meta, dict) and meta.get('relaxed')) else ""
            # Her çakışmanın temel metrikleri
            metrics = []
            for l in sorted(in_lists):
                for list_name in ('alsat', 'tavan', 'nw', 'rt'):
                    if list_name != l:
                        continue
                    for t, sc, reas, sig in lists_dict.get(list_name, []):
                        if t != ticker:
                            continue
                        if list_name == 'rt':
                            badge = sig.get('badge', '')
                            if badge:
                                metrics.append(f"[{badge}]")
                            cmf = sig.get('cmf')
                            if cmf is not None:
                                metrics.append(f"CMF{cmf:+.2f}")
                        elif list_name == 'nw':
                            if sig.get('dw_overlap'):
                                metrics.append("D+W")
                            delta = sig.get('delta_pct')
                            if delta is not None:
                                metrics.append(f"δ={delta:.1f}%")
                        elif list_name == 'alsat':
                            st = sig.get('signal_type', '')
                            metrics.append(st)
                            q = sig.get('quality', 0)
                            if q:
                                metrics.append(f"Q={q}")
                        elif list_name == 'tavan':
                            skor = sig.get('skor', 0) or 0
                            metrics.append(f"tavan_skor={skor}")
                        break
            metrics_str = " ".join(metrics)
            ps = _price_str(ticker)
            data_context.append(
                f"  {ticker}: {list_tags}{relaxed} quality={quality} {metrics_str} {ps}".rstrip())

    # ── Tier 2: Tekil Kalite Sinyaller (horizon bazlı) ──
    def _format_tier2_items(items, label):
        if not items:
            return
        data_context.append(f"\n## {label} ({len(items)} hisse)")
        by_list = {}
        for ticker, score, reasons, sig in items:
            tag = reasons[0] if reasons else '[?]'
            by_list.setdefault(tag, []).append((ticker, score, reasons, sig))
        for tag, group in by_list.items():
            tickers_str = []
            for ticker, score, reasons, sig in group:
                detail_parts = [ticker]
                if isinstance(sig, dict):
                    if sig.get('dw_overlap'):
                        detail_parts.append("D+W")
                    delta = sig.get('delta_pct')
                    if delta is not None:
                        detail_parts.append(f"δ={delta:.1f}%")
                    if sig.get('gate'):
                        detail_parts.append("gate=AÇIK")
                    st = sig.get('signal_type', '')
                    if st:
                        detail_parts.append(st)
                    q = sig.get('quality', 0) or 0
                    if q:
                        detail_parts.append(f"Q={q}")
                    badge = sig.get('badge', '')
                    if badge:
                        detail_parts.append(badge)
                    cmf = sig.get('cmf')
                    if cmf is not None:
                        detail_parts.append(f"CMF{cmf:+.2f}")
                    skor = sig.get('skor')
                    if skor is not None:
                        detail_parts.append(f"skor={skor}")
                ps = _price_str(ticker)
                if ps:
                    detail_parts.append(ps)
                tickers_str.append(" ".join(detail_parts))
            data_context.append(f"  {tag} ({len(group)} hisse): {', '.join(tickers_str)}")

    if lists_dict:
        _format_tier2_items(lists_dict.get('tier2a', []),
                           "⚡ Tier 2A — Tactical ⚡1G (intraday/1G çıkış)")
        _format_tier2_items(lists_dict.get('tier2b', []),
                           "⭐ Tier 2B — Swing-Lite ⏳3G/🔄SW (3-5G tutma)")

    # ── Çelişki Tespiti (AL + SAT aynı hissede) ──
    if latest_signals and lists_dict:
        sat_tickers = {s['ticker'] for s in latest_signals
                       if s.get('direction') == 'SAT' or s.get('karar') == 'SAT'}
        shortlist_tickers = set()
        for list_name in ('tier1', 'tier2', 'alsat', 'tavan', 'nw', 'rt'):
            for item in lists_dict.get(list_name, []):
                shortlist_tickers.add(item[0])
        conflicts = shortlist_tickers & sat_tickers
        if conflicts:
            data_context.append(f"\n## ⚠️ ÇELİŞKİLER (shortlist'te AL ama SAT sinyali de var — {len(conflicts)} hisse)")
            for ticker in sorted(conflicts):
                # SAT sinyal kaynağını bul
                sat_sources = []
                for s in latest_signals:
                    if s['ticker'] == ticker and (s.get('direction') == 'SAT' or s.get('karar') == 'SAT'):
                        scr = SCREENER_NAMES.get(s.get('screener', ''), s.get('screener', '?'))
                        sat_sources.append(scr)
                # AL sinyal kaynağını bul
                al_sources = []
                for list_name in ('alsat', 'tavan', 'nw', 'rt'):
                    for t, sc, reas, sig in lists_dict.get(list_name, []):
                        if t == ticker:
                            al_sources.append(_LIST_SHORT.get(list_name, list_name))
                data_context.append(
                    f"  {ticker}: AL kaynak={'+'.join(al_sources)} AMA SAT={', '.join(sat_sources)} → BEKLE")

    # ── SAT sinyalleri (kısa — risk farkındalığı) ──
    if latest_signals:
        sat_signals = [s for s in latest_signals
                       if s.get('direction') == 'SAT' or s.get('karar') == 'SAT']
        if sat_signals:
            data_context.append(f"\n## SAT Sinyalleri ({len(sat_signals)} toplam — risk farkındalığı):")
            for s in sat_signals[:10]:
                scr = SCREENER_NAMES.get(s.get('screener', ''), s.get('screener', '?'))
                data_context.append(f"  {s['ticker']}: {scr} SAT")

    # ── MKK kompakt özeti (sadece bilgi, skor etkisi yok) ──
    shortlist_tickers = set()
    if lists_dict:
        for list_name in ('tier1', 'tier2', 'alsat', 'tavan', 'nw', 'rt'):
            for item in lists_dict.get(list_name, []):
                shortlist_tickers.add(item[0])

    mkk_json = fetch_mkk_data()
    has_sm_data = False
    if mkk_json and mkk_json.get('data'):
        mkk_data = mkk_json['data']
        mkk_lines = []
        for ticker in sorted(shortlist_tickers):
            mkk = mkk_data.get(ticker)
            if mkk:
                k_pct = mkk.get('kurumsal_pct', 0)
                fark_1g = mkk.get('bireysel_fark_1g')
                fark_str = f" {fark_1g:+.1f}%" if fark_1g is not None else ""
                mkk_lines.append(f"  {ticker}: K%{k_pct:.0f}{fark_str}")
        if mkk_lines:
            has_sm_data = True
            data_context.append(f"\n## MKK Bilgi (tarih: {mkk_json.get('extracted_at', '?')[:10]}, sadece bilgi — skor etkisi yok)")
            data_context.extend(mkk_lines)

    if not has_sm_data:
        data_context.append("\n## SM/Takas/MKK verisi YOK — bu verilere referans yapma")

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
    """Telegram mesajını formatla — template brifing zaten tam içerik."""
    lines = []

    # HTML rapor linki (üstte — chunk bölünmesinde kaybolmasın)
    if html_url:
        lines.append(f'🔗 <a href="{html_url}">Detaylı Rapor</a>')
        lines.append("")

    # Brifing metni (template + AI limit order zaten birleşik)
    max_brief_len = 6000
    if len(briefing_text) > max_brief_len:
        briefing_text = briefing_text[:max_brief_len] + "..."
    lines.append(briefing_text)

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
