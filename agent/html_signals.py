"""
NOX Agent — HTML Raporlardan Sinyal Okuyucu
=============================================
GitHub Pages'te yayınlanan scanner HTML raporlarından sinyal çeker.
CSV artifact'lerine bağımlılığı ortadan kaldırır — tek kaynak: HTML.

Veri kaynakları:
  - nox-signals/nox_v3_weekly.html      → NW haftalık pivot AL/SAT
  - nox-signals/regime_transition.html   → RT günlük rejim geçiş
  - nox-signals/regime_transition_weekly.html → RT haftalık (badge tespiti için)
  - bist-signals/tavan.html             → Tavan serisi
  - bist-signals/ (index.html)          → AL/SAT günlük
"""
import json
import os
import re
from html.parser import HTMLParser

import requests

from markets.bist.regime_transition import classify_volume_quality, is_donus_transition

# -- URL Yapılandırması --
_NOX_BASE = "https://aalpkk.github.io/nox-signals"
_BIST_BASE = "https://aalpkk.github.io/bist-signals"

_HTML_SOURCES = {
    'nw': f'{_NOX_BASE}/nox_v3_weekly.html',
    'rt': f'{_NOX_BASE}/regime_transition.html',
    'rt_weekly': f'{_NOX_BASE}/regime_transition_weekly.html',
    'tavan': f'{_BIST_BASE}/tavan.html',
    'alsat': f'{_BIST_BASE}/',  # index.html
    'sbt': f'{_BIST_BASE}/smart_breakout.html',  # SBT — Katman C, ML overlay'de consume edilir
}

_TIMEOUT = 20


# ============================================================================
# Yardımcı Fonksiyonlar
# ============================================================================

def _get_base_urls():
    """Ortam değişkenlerinden base URL'leri al (override desteği)."""
    nox = os.environ.get("GH_PAGES_BASE_URL", _NOX_BASE).rstrip("/")
    bist = os.environ.get("BIST_PAGES_BASE_URL", _BIST_BASE).rstrip("/")
    return nox, bist


def _fetch_html(url):
    """URL'den HTML içeriği indir."""
    try:
        resp = requests.get(url, timeout=_TIMEOUT)
        if resp.status_code != 200:
            print(f"  ⚠️ {url} → HTTP {resp.status_code}")
            return None
        return resp.text
    except Exception as e:
        print(f"  ⚠️ {url} → {e}")
        return None


def _extract_const_d(html_text):
    """HTML'den const D={...}; JavaScript verisini çıkar.

    Hem 'const D={...};' hem de 'const D ={...};' formatlarını destekler.
    Returns: dict veya None
    """
    # const D= sonrası, son }; kapanışına kadar al
    m = re.search(r'const\s+D\s*=\s*(\{.+?\})\s*;', html_text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        # Bazı HTML'lerde JavaScript obje literal'i JSON değil (trailing comma vs.)
        # Temizleme dene
        raw = m.group(1)
        # Trailing comma temizliği
        raw = re.sub(r',\s*([\]}])', r'\1', raw)
        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            print(f"  ⚠️ const D parse hatası: {e}")
            return None


def _extract_const_data(html_text):
    """HTML'den const DATA=[...]; JavaScript verisini çıkar.

    bist-signals index.html formatı: const DATA = [...];
    Returns: list veya None
    """
    m = re.search(r'const\s+DATA\s*=\s*(\[.+?\])\s*;', html_text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        raw = m.group(1)
        raw = re.sub(r',\s*([\]}])', r'\1', raw)
        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            print(f"  ⚠️ const DATA parse hatası: {e}")
            return None


class _TavanTableParser(HTMLParser):
    """Tavan HTML'deki <table> satırlarını parse eder.

    Her <tr data-ticker="XXX"> satırından hücre değerlerini çıkarır.
    Kolon sırası: Hisse, Skor, Karar, Seri, Fiyat, Hacim, EMA21, RSI, RS, Yabancı
    """
    def __init__(self):
        super().__init__()
        self.rows = []
        self._current_ticker = None
        self._current_cells = []
        self._in_td = False
        self._td_text = ""
        self._in_tbody = False

    def handle_starttag(self, tag, attrs):
        attrs_d = dict(attrs)
        if tag == 'tbody':
            self._in_tbody = True
        elif tag == 'tr' and self._in_tbody:
            ticker = attrs_d.get('data-ticker')
            if ticker:
                self._current_ticker = ticker
                self._current_cells = []
        elif tag == 'td' and self._current_ticker is not None:
            self._in_td = True
            self._td_text = ""

    def handle_endtag(self, tag):
        if tag == 'tbody':
            self._in_tbody = False
        elif tag == 'td' and self._in_td:
            self._in_td = False
            self._current_cells.append(self._td_text.strip())
        elif tag == 'tr' and self._current_ticker is not None:
            if self._current_cells:
                self.rows.append({
                    'ticker': self._current_ticker,
                    'cells': self._current_cells,
                })
            self._current_ticker = None
            self._current_cells = []

    def handle_data(self, data):
        if self._in_td:
            self._td_text += data


def _safe_float(val, default=0.0):
    """Güvenli float dönüşümü."""
    try:
        # "2.5x" → 2.5, "+22.2%" → 22.2, "%10.1 (+0.10)" → 10.1
        if isinstance(val, str):
            val = val.replace('x', '').replace('%', '').replace('+', '').strip()
            # Parantez içi varsa sadece ilk sayıyı al
            val = val.split('(')[0].strip()
            val = val.split()[0]  # boşluktan sonrasını at
        return float(val)
    except (ValueError, TypeError, IndexError):
        return default


def _safe_int(val, default=0):
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return default


# ============================================================================
# NW Weekly Sinyaller
# ============================================================================

def fetch_nw_signals(base_url=None):
    """NW Weekly HTML'den sinyal listesi çek.

    D.weekly.buys → PIVOT_AL (trigger_type varsa) veya ZONE_ONLY
    D.weekly.sells → PIVOT_SAT
    D.weekly.candidates → ADAY
    D.daily.buys/sells de dahil (günlük tetik çakışması + NW günlük sinyaller)

    Returns: scanner_reader formatında signal dicts listesi
    """
    nox_base, _ = _get_base_urls()
    url = f"{nox_base}/nox_v3_weekly.html" if not base_url else f"{base_url}/nox_v3_weekly.html"

    html = _fetch_html(url)
    if not html:
        return []

    d = _extract_const_d(html)
    if not d:
        print("  ⚠️ NW HTML: const D bulunamadı")
        return []

    signals = []
    report_date = d.get('weekly', {}).get('date', d.get('daily', {}).get('date', ''))

    # Overlap set'i (D+W çakışma)
    overlap = set(d.get('overlap', []))

    # Weekly buys — D.weekly.buys HER ZAMAN PIVOT_AL (tetikli sinyaller)
    # trigger_type HTML'de yoksa bile sinyal tipi PIVOT_AL
    for b in d.get('weekly', {}).get('buys', []):
        trigger = b.get('trigger_type', '')

        entry = {
            'screener': 'nox_v3_weekly',
            'ticker': b['ticker'],
            'signal_date': b.get('signal_date', report_date),
            'direction': 'AL',
            'signal_type': 'PIVOT_AL',
            'entry_price': b.get('close', 0),
            'quality': None,
            'trigger_type': trigger,
            'wl_status': b.get('status', 'BEKLE'),
            'delta_pct': b.get('delta_pct'),
            'rs_score': b.get('rs_score'),
            'fresh': b.get('fresh', 'YAKIN'),
            'gate': b.get('gate', False),
            'tb_stage': b.get('tb_stage', '-'),
            'dw_overlap': b['ticker'] in overlap,
            'csv_date': report_date.replace('-', '') if report_date else '',
        }
        signals.append(entry)

    # Weekly sells
    for s in d.get('weekly', {}).get('sells', []):
        signals.append({
            'screener': 'nox_v3_weekly',
            'ticker': s['ticker'],
            'signal_date': s.get('signal_date', report_date),
            'direction': 'SAT',
            'signal_type': 'PIVOT_SAT',
            'entry_price': s.get('close', 0),
            'quality': None,
            'fresh': s.get('fresh', 'YAKIN'),
            'csv_date': report_date.replace('-', '') if report_date else '',
        })

    # Weekly candidates (ADAY — izleme, trade yok)
    for c in d.get('weekly', {}).get('candidates', []):
        signals.append({
            'screener': 'nox_v3_weekly',
            'ticker': c['ticker'],
            'signal_date': report_date,
            'direction': 'AL',
            'signal_type': 'ADAY',
            'entry_price': c.get('close', 0),
            'quality': None,
            'wl_status': c.get('status', 'BEKLE'),
            'delta_pct': c.get('delta_pct'),
            'fresh': 'ADAY',
            'csv_date': report_date.replace('-', '') if report_date else '',
        })

    # Daily buys — NW günlük sinyaller (D+W çakışma tespiti için)
    daily_date = d.get('daily', {}).get('date', '')
    for b in d.get('daily', {}).get('buys', []):
        signals.append({
            'screener': 'nox_v3_daily',
            'ticker': b['ticker'],
            'signal_date': b.get('signal_date', daily_date),
            'direction': 'AL',
            'signal_type': 'PIVOT_AL',
            'entry_price': b.get('close', 0),
            'quality': None,
            'fresh': b.get('fresh', 'YAKIN'),
            'gate': b.get('gate', False),
            'delta_pct': b.get('delta_pct'),
            'adx': b.get('adx'),
            'rsi': b.get('rsi'),
            'rg': b.get('rg'),
            'slope': b.get('slope'),
            'rs_score': b.get('rs_score'),
            'dw_overlap': b['ticker'] in overlap,
            'csv_date': daily_date.replace('-', '') if daily_date else '',
        })

    n_weekly_al = sum(1 for s in signals if s['screener'] == 'nox_v3_weekly' and s['direction'] == 'AL')
    n_weekly_sat = sum(1 for s in signals if s['screener'] == 'nox_v3_weekly' and s['direction'] == 'SAT')
    n_daily_al = sum(1 for s in signals if s['screener'] == 'nox_v3_daily')
    n_daily_gate = sum(1 for s in signals if s['screener'] == 'nox_v3_daily' and s.get('gate'))
    print(f"  NW: {n_weekly_al} W-AL + {n_weekly_sat} W-SAT | {n_daily_al} D-AL ({n_daily_gate} gate) ({daily_date})")
    return signals


# ============================================================================
# RT Sinyaller (Badge tespiti dahil)
# ============================================================================

def fetch_rt_signals(base_url=None):
    """RT Daily + Weekly HTML'den sinyal listesi çek.

    Badge tespiti:
      1. RT Weekly HTML → haftalık AL hisseleri
      2. RT Daily HTML → günlük sinyaller + weekly_al/weekly_pb flag'leri
      3. Günlük D.rows'ta weekly_pb=true → H+PB, weekly_al=true → H+AL

    Returns: scanner_reader formatında signal dicts listesi
    """
    nox_base, _ = _get_base_urls()

    # 1. RT Daily HTML
    rt_url = f"{nox_base}/regime_transition.html" if not base_url else f"{base_url}/regime_transition.html"
    html = _fetch_html(rt_url)
    if not html:
        return []

    d = _extract_const_d(html)
    if not d:
        print("  ⚠️ RT HTML: const D bulunamadı")
        return []

    report_date = d.get('date', '')
    # Tarih formatı "13.03.2026 15:58" olabilir → ISO'ya çevir
    if report_date and '.' in report_date and len(report_date) > 10:
        try:
            parts = report_date.split()[0].split('.')
            report_date = f"{parts[2]}-{parts[1]}-{parts[0]}"
        except (IndexError, ValueError):
            pass
    elif report_date and '.' in report_date:
        try:
            parts = report_date.split('.')
            report_date = f"{parts[2]}-{parts[1]}-{parts[0]}"
        except (IndexError, ValueError):
            pass

    signals = []
    for r in d.get('rows', []):
        entry_window = r.get('entry_window', '')
        if entry_window in ('TAZE', '2.DALGA', 'YAKIN'):
            direction = 'AL'
        else:
            direction = 'SAT'

        # Badge tespiti — RT Daily HTML'de weekly_al/weekly_pb zaten gömülü
        badge = ''
        if r.get('weekly_pb'):
            badge = 'H+PB'
        elif r.get('weekly_al'):
            badge = 'H+AL'

        # Hacim-donus tier — sadece DONUS sinyallerinde (backtest kanitli)
        _atr_pct = r.get('atr_pct', 0) or 0
        _rvol = r.get('rvol', 0) or 0
        _cmf = r.get('cmf', 0) or 0
        _part = r.get('participation_score', 0) or 0
        _oe = r.get('oe_score', 0) or 0
        _transition = r.get('transition', '')
        if is_donus_transition(_transition):
            vol_tier, vol_tier_icon = classify_volume_quality(
                float(_atr_pct), float(_cmf), float(_rvol), int(_part), int(_oe))
        else:
            vol_tier, vol_tier_icon = '', ''

        entry = {
            'screener': 'regime_transition',
            'ticker': r['ticker'],
            'signal_date': report_date,
            'direction': direction,
            'signal_type': r.get('transition', ''),
            'entry_price': r.get('close', 0),
            'quality': r.get('entry_score', 0),
            'entry_window': entry_window,
            'oe': r.get('oe_score', 0),
            'oe_detail': r.get('oe_tags', ''),
            'cmf': r.get('cmf', 0),
            'adx': r.get('adx', 0),
            'trend_score': r.get('trend_score', 0),
            'regime': r.get('regime_name', ''),
            'transition_date': r.get('transition_date', r.get('transition_date_iso', '')),
            'csv_date': report_date.replace('-', '') if report_date else '',
            # Hacim-donus tier
            'atr_pct': _atr_pct,
            'rvol': _rvol,
            'participation_score': _part,
            'vol_tier': vol_tier,
            'vol_tier_icon': vol_tier_icon,
        }
        if badge:
            entry['badge'] = badge
        signals.append(entry)

    n_al = sum(1 for s in signals if s['direction'] == 'AL')
    n_badge = sum(1 for s in signals if s.get('badge'))
    print(f"  RT: {n_al} AL ({n_badge} badge) ({report_date})")
    return signals


# ============================================================================
# Tavan Sinyaller
# ============================================================================

def fetch_tavan_signals(base_url=None):
    """Tavan HTML'den tavan serisi sinyallerini çek.

    HTML table parse → data-ticker + hücre indeksleri.
    Kolon sırası: Hisse, Skor, Karar, Seri, Fiyat, Hacim, EMA21, RSI, RS, Yabancı

    Returns: scanner_reader formatında signal dicts listesi
    """
    _, bist_base = _get_base_urls()
    url = f"{bist_base}/tavan.html" if not base_url else f"{base_url}/tavan.html"

    html = _fetch_html(url)
    if not html:
        return []

    parser = _TavanTableParser()
    parser.feed(html)

    if not parser.rows:
        print("  ⚠️ Tavan HTML: tablo satırı bulunamadı")
        return []

    # Tarih: HTML başlığından veya bugünün tarihi
    date_m = re.search(r'(\d{2})\.(\d{2})\.(\d{4})', html)
    if date_m:
        report_date = f"{date_m.group(3)}-{date_m.group(2)}-{date_m.group(1)}"
    else:
        from datetime import datetime, timezone, timedelta
        report_date = datetime.now(timezone(timedelta(hours=3))).strftime('%Y-%m-%d')

    signals = []
    for row in parser.rows:
        cells = row['cells']
        if len(cells) < 7:
            continue

        # Hücre sırası: [#, Hisse, Skor, Karar, Seri, Fiyat, Hacim, EMA21, RSI, RS, Yabancı, ...]
        # cells[0] = sıra no, cells[1] = ticker (zaten data-ticker'da), cells[2] = skor
        skor = _safe_int(cells[2])
        karar = cells[3].strip() if len(cells) > 3 else ''
        seri = _safe_int(cells[4]) if len(cells) > 4 else 1
        fiyat = _safe_float(cells[5]) if len(cells) > 5 else 0
        hacim = _safe_float(cells[6]) if len(cells) > 6 else 1.0
        rsi = _safe_float(cells[8]) if len(cells) > 8 else 50
        rs = _safe_float(cells[9]) if len(cells) > 9 else 0

        direction = 'AL' if rs >= 0 else 'SAT'

        entry = {
            'screener': 'tavan',
            'ticker': row['ticker'],
            'signal_date': report_date,
            'direction': direction,
            'signal_type': 'TAVAN',
            'entry_price': fiyat,
            'quality': skor,
            'skor': skor,
            'streak': seri,
            'volume_ratio': hacim,
            'rs': rs,
            'csv_date': report_date.replace('-', ''),
        }
        signals.append(entry)

    print(f"  Tavan: {len(signals)} hisse ({report_date})")
    return signals


# ============================================================================
# AL/SAT (bist-signals) Sinyaller
# ============================================================================

def fetch_alsat_signals(base_url=None):
    """AL/SAT (bist-signals) HTML'den sinyal çek.

    const DATA = [...]; formatı — her entry bir hisse.
    Sadece karar=AL olanlar + DÖNÜŞ tipi güçlü sinyaller.

    Returns: scanner_reader formatında signal dicts listesi
    """
    _, bist_base = _get_base_urls()
    url = f"{bist_base}/" if not base_url else f"{base_url}/"

    html = _fetch_html(url)
    if not html:
        return []

    data = _extract_const_data(html)
    if not data:
        print("  ⚠️ AL/SAT HTML: const DATA bulunamadı")
        return []

    # Tarih: URL veya HTML'den
    date_m = re.search(r'(\d{2})\.(\d{2})\.(\d{4})', html)
    if date_m:
        report_date = f"{date_m.group(3)}-{date_m.group(2)}-{date_m.group(1)}"
    else:
        from datetime import datetime, timezone, timedelta
        report_date = datetime.now(timezone(timedelta(hours=3))).strftime('%Y-%m-%d')

    signals = []
    for row in data:
        karar = row.get('karar', '')
        if karar == 'ATLA':
            continue

        entry = {
            'screener': 'alsat',
            'ticker': row['ticker'],
            'signal_date': report_date,
            'direction': 'AL',
            'signal_type': row.get('signal', ''),
            'entry_price': row.get('close', 0),
            'quality': row.get('quality'),
            'karar': karar,
            'csv_date': report_date.replace('-', ''),
        }
        if row.get('rs_score') is not None:
            entry['rs_score'] = round(row['rs_score'], 1) if isinstance(row['rs_score'], float) else row['rs_score']
        if row.get('regime'):
            entry['regime'] = row['regime']
        if row.get('oe') is not None:
            entry['oe'] = row['oe']
            entry['oe_detail'] = row.get('oe_detail', '')
        if row.get('rr') is not None:
            entry['rr'] = round(row['rr'], 2) if isinstance(row['rr'], float) else row['rr']
        if row.get('stop') is not None:
            entry['stop_price'] = round(row['stop'], 2) if isinstance(row['stop'], float) else row['stop']
        if row.get('tp') is not None:
            entry['target_price'] = round(row['tp'], 2) if isinstance(row['tp'], float) else row['tp']
        if row.get('macd_hist') is not None:
            entry['macd'] = round(row['macd_hist'], 4) if isinstance(row['macd_hist'], float) else row['macd_hist']
        signals.append(entry)

    n_al = sum(1 for s in signals if s['karar'] == 'AL')
    n_izle = sum(1 for s in signals if s['karar'] == 'İZLE')
    print(f"  AL/SAT: {n_al} AL + {n_izle} İZLE ({report_date})")
    return signals


# ============================================================================
# SBT (Smart Breakout Targets) Sinyaller — Stub
# ============================================================================

class _SBTBreakoutParser(HTMLParser):
    """SBT Breakout HTML tablosunu parse eder.

    Ticker: <a class="tk">TICKER</a> link text'inden.
    Kolonlar: Sembol, Yön, Güç, Giriş, SL, TP1, TP2, TP3, Durum, Gün, ML, Bucket, Gate
    data-val attribute varsa kullan, yoksa text content.
    """
    def __init__(self):
        super().__init__()
        self.rows = []
        self._current_row = []
        self._in_td = False
        self._td_text = ""
        self._td_data_val = None
        self._in_tbody = False
        self._current_ticker = None

    def handle_starttag(self, tag, attrs):
        attrs_d = dict(attrs)
        if tag == 'tbody':
            self._in_tbody = True
        elif tag == 'tr' and self._in_tbody:
            self._current_row = []
            self._current_ticker = None
        elif tag == 'td' and self._in_tbody:
            self._in_td = True
            self._td_text = ""
            self._td_data_val = attrs_d.get('data-val')
        elif tag == 'a' and self._in_td:
            # <a class="tk">TICKER</a> — ticker link
            if 'tk' in (attrs_d.get('class', '')):
                pass  # text content'i handle_data'da yakalanır

    def handle_endtag(self, tag):
        if tag == 'tbody':
            self._in_tbody = False
        elif tag == 'td' and self._in_td:
            self._in_td = False
            val = self._td_data_val if self._td_data_val is not None else self._td_text.strip()
            self._current_row.append(val)
            self._td_data_val = None
        elif tag == 'tr' and self._in_tbody and self._current_row:
            if len(self._current_row) >= 10:
                # İlk kolon = sembol (ticker)
                ticker = self._current_row[0].strip()
                if ticker and len(ticker) >= 3:
                    self.rows.append({
                        'ticker': ticker,
                        'cells': self._current_row,
                    })
            self._current_row = []

    def handle_data(self, data):
        if self._in_td:
            self._td_text += data


def fetch_sbt_signals(base_url=None):
    """SBT HTML'den breakout sinyallerini çek.

    Filtre: bars_ago <= 3, trade_status OPEN/WIN_TRAIL, bucket != 'X'

    Returns: scanner_reader formatında signal dicts listesi
    """
    _, bist_base = _get_base_urls()
    url = f"{bist_base}/smart_breakout.html" if not base_url else f"{base_url}/smart_breakout.html"

    html = _fetch_html(url)
    if not html:
        return []

    # Önce const D JSON formatını dene
    d = _extract_const_d(html)
    if d and d.get('rows'):
        return _parse_sbt_json(d)

    # Fallback: HTML tablo parse
    parser = _SBTBreakoutParser()
    parser.feed(html)

    if not parser.rows:
        print("  ⚠️ SBT HTML: tablo satırı bulunamadı")
        return []

    # Tarih
    date_m = re.search(r'(\d{2})\.(\d{2})\.(\d{4})', html)
    if date_m:
        report_date = f"{date_m.group(3)}-{date_m.group(2)}-{date_m.group(1)}"
    else:
        from datetime import datetime, timezone, timedelta
        report_date = datetime.now(timezone(timedelta(hours=3))).strftime('%Y-%m-%d')

    signals = []
    for row in parser.rows:
        cells = row['cells']
        # Kolon sırası: Sembol(0), Yön(1), Güç(2), Giriş(3), SL(4),
        # TP1(5), TP2(6), TP3(7), Durum(8), Gün(9), ML(10), Bucket(11), Gate(12)
        ticker = row['ticker']
        direction_raw = cells[1].strip() if len(cells) > 1 else ''
        strength = _safe_int(cells[2]) if len(cells) > 2 else 0
        entry_price = _safe_float(cells[3]) if len(cells) > 3 else 0
        sl_price = _safe_float(cells[4]) if len(cells) > 4 else 0
        tp1_price = _safe_float(cells[5]) if len(cells) > 5 else 0
        tp2_price = _safe_float(cells[6]) if len(cells) > 6 else 0
        tp3_price = _safe_float(cells[7]) if len(cells) > 7 else 0
        trade_status = cells[8].strip() if len(cells) > 8 else ''
        bars_ago = _safe_int(cells[9]) if len(cells) > 9 else 99
        ml_prob_raw = cells[10].strip() if len(cells) > 10 else '0'
        bucket = cells[11].strip() if len(cells) > 11 else ''
        gate = cells[12].strip() if len(cells) > 12 else ''

        # ML prob parse
        ml_prob = _safe_float(ml_prob_raw)
        if ml_prob > 1:
            ml_prob = ml_prob / 100

        # Filtreler
        if bars_ago > 3:
            continue
        if not any(k in trade_status.upper() for k in ('OPEN', 'WIN_TRAIL', 'ACIK')):
            continue
        if bucket == 'X':
            continue

        signals.append({
            'screener': 'sbt',
            'ticker': ticker,
            'signal_date': report_date,
            'direction': 'AL',
            'signal_type': 'BREAKOUT',
            'entry_price': entry_price,
            'quality': strength,
            'sl_price': sl_price,
            'tp1_price': tp1_price,
            'tp2_price': tp2_price,
            'tp3_price': tp3_price,
            'sbt_bucket': bucket,
            'sbt_ml_prob': ml_prob,
            'sbt_gate': gate,
            'bars_ago': bars_ago,
            'csv_date': report_date.replace('-', ''),
        })

    print(f"  SBT: {len(signals)} breakout ({report_date})")
    return signals


def _parse_sbt_json(d):
    """const D JSON formatından SBT sinyallerini parse et."""
    report_date = d.get('date', d.get('report_date', ''))
    if report_date and '.' in report_date:
        try:
            parts = report_date.split()[0].split('.')
            report_date = f"{parts[2]}-{parts[1]}-{parts[0]}"
        except (IndexError, ValueError):
            pass
    if not report_date:
        from datetime import datetime, timezone, timedelta
        report_date = datetime.now(timezone(timedelta(hours=3))).strftime('%Y-%m-%d')

    signals = []
    for row in d.get('rows', []):
        ticker = row.get('ticker', '')
        if not ticker:
            continue
        bars_ago = row.get('bars_ago', row.get('gun', 99))
        trade_status = str(row.get('trade_status', row.get('durum', '')))
        bucket = str(row.get('ml_bucket', row.get('bucket', '')))

        # Filtreler
        if bars_ago > 3:
            continue
        if not any(k in trade_status.upper() for k in ('OPEN', 'WIN_TRAIL', 'ACIK')):
            continue
        if bucket == 'X':
            continue

        ml_prob = row.get('ml_prob', row.get('prob', 0))
        ml_prob = float(ml_prob) if ml_prob else 0.0
        if ml_prob > 1:
            ml_prob = ml_prob / 100

        signals.append({
            'screener': 'sbt',
            'ticker': ticker,
            'signal_date': report_date,
            'direction': 'AL',
            'signal_type': 'BREAKOUT',
            'entry_price': float(row.get('entry', row.get('giris', 0)) or 0),
            'quality': int(row.get('strength', row.get('guc', 0)) or 0),
            'sl_price': float(row.get('sl', 0) or 0),
            'tp1_price': float(row.get('tp1', 0) or 0),
            'tp2_price': float(row.get('tp2', 0) or 0),
            'tp3_price': float(row.get('tp3', 0) or 0),
            'sbt_bucket': bucket,
            'sbt_ml_prob': ml_prob,
            'sbt_gate': str(row.get('gate', '')),
            'bars_ago': int(bars_ago),
            'csv_date': report_date.replace('-', ''),
        })

    print(f"  SBT: {len(signals)} breakout ({report_date})")
    return signals


# ============================================================================
# Birleşik Fonksiyon
# ============================================================================

def fetch_all_html_signals():
    """Tüm HTML kaynaklardan sinyal çek, birleştir.

    Sıra: NW → RT → Tavan → AL/SAT
    Her kaynak bağımsız — bir kaynak fail olursa diğerleri çalışmaya devam eder.

    Returns: scanner_reader formatıyla uyumlu signal dicts listesi
    """
    print("📡 HTML raporlardan sinyal çekiliyor...")
    signals = []

    for name, fetcher in [
        ('NW', fetch_nw_signals),
        ('RT', fetch_rt_signals),
        ('Tavan', fetch_tavan_signals),
        ('AL/SAT', fetch_alsat_signals),
        ('SBT', fetch_sbt_signals),
    ]:
        try:
            sigs = fetcher()
            signals.extend(sigs)
        except Exception as e:
            print(f"  ⚠️ {name} hatası: {e}")

    print(f"  Toplam: {len(signals)} sinyal")
    return signals
