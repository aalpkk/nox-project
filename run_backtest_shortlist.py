#!/usr/bin/env python3
"""
NOX Shortlist Backtest
======================
Günlük brifing pipeline'ının ürettiği 4-liste shortlist (AL/SAT, Tavan, NW, RT
+ Tier1/Tier2) sisteminin gerçek performansını ölçer.

Faz 1: GitHub Pages HTML commit history → tarihsel sinyaller → shortlist reconstruct
Faz 2: --full flag → 4 screener modülünü tarihsel OHLCV üzerinde çalıştır (5Y)

Kullanım:
    python run_backtest_shortlist.py                       # Faz 1 (GitHub HTML, ~20 gün)
    python run_backtest_shortlist.py --full                 # Faz 2 (5 yıl, default)
    python run_backtest_shortlist.py --full --period 3y     # Faz 2 (3 yıl)
    python run_backtest_shortlist.py --full --period 1y     # Faz 2 (1 yıl hızlı test)
    python run_backtest_shortlist.py --full --open          # Raporu browser'da aç
"""

import argparse
import base64
import csv
import json
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from html.parser import HTMLParser
from statistics import median

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from markets.bist import data as data_mod
from core.reports import _NOX_CSS, _sanitize

WINDOWS = [1, 3, 5]
_TZ_TR = timezone(timedelta(hours=3))


# =============================================================================
# 1. GITHUB HTML GEÇMİŞİNDEN VERİ TOPLAMA
# =============================================================================

def _gh_api(endpoint, timeout=30):
    """gh CLI ile GitHub API çağrısı."""
    result = subprocess.run(
        ['gh', 'api', endpoint],
        capture_output=True, text=True, timeout=timeout,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _fetch_commits(repo, path, max_pages=3):
    """Bir repo+dosya için commit listesi çek.
    Returns: [(sha, date_str_YYYYMMDD), ...]
    """
    commits = []
    for page in range(1, max_pages + 1):
        endpoint = (f'repos/{repo}/commits?path={path}'
                    f'&per_page=100&page={page}')
        raw = _gh_api(endpoint)
        if not raw:
            break
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            break
        if not data:
            break
        for item in data:
            sha = item.get('sha', '')
            date_iso = item.get('commit', {}).get('committer', {}).get('date', '')
            if not sha or not date_iso:
                continue
            # ISO format: 2026-03-17T15:30:00Z → YYYYMMDD
            try:
                dt = datetime.fromisoformat(date_iso.replace('Z', '+00:00'))
                # UTC+3'e çevir (BIST saati)
                dt_tr = dt.astimezone(_TZ_TR)
                date_str = dt_tr.strftime('%Y%m%d')
                commits.append((sha, date_str))
            except (ValueError, TypeError):
                continue
    return commits


def _fetch_file_at_sha(repo, path, sha):
    """Belirli bir SHA'daki dosya içeriğini base64 decode et."""
    endpoint = f'repos/{repo}/contents/{path}?ref={sha}'
    raw = _gh_api(endpoint)
    if not raw:
        return None
    try:
        data = json.loads(raw)
        content_b64 = data.get('content', '')
        if not content_b64:
            return None
        return base64.b64decode(content_b64).decode('utf-8')
    except (json.JSONDecodeError, UnicodeDecodeError, Exception):
        return None


def _dedup_by_date(commits):
    """Gün başına son commit'i al."""
    date_sha = {}
    for sha, date_str in commits:
        date_sha[date_str] = sha  # son gelen kazanır
    return sorted(date_sha.items())


def fetch_all_html_history():
    """2 GitHub repo'dan 4 HTML dosyasının commit geçmişini çek.

    Returns: {date_str: {'nw': html, 'rt': html, 'alsat': html, 'tavan': html}}
    """
    print("\n[1] GitHub HTML geçmişi çekiliyor...")

    repos = {
        'nw': ('aalpkk/nox-signals', 'nox_v3_weekly.html'),
        'rt': ('aalpkk/nox-signals', 'regime_transition.html'),
        'alsat': ('aalpkk/bist-signals', 'index.html'),
        'tavan': ('aalpkk/bist-signals', 'tavan.html'),
    }

    # Her kaynak için commit listesi çek
    all_dates = set()
    source_commits = {}
    for key, (repo, path) in repos.items():
        print(f"  {key}: commits from {repo}/{path}...")
        commits = _fetch_commits(repo, path)
        deduped = _dedup_by_date(commits)
        source_commits[key] = dict(deduped)
        dates = set(d for d, _ in deduped)
        all_dates |= dates
        print(f"    {len(deduped)} gün")

    # Ortak tarihleri bul (en az 2 kaynak olan günler)
    date_counts = defaultdict(int)
    for key, dmap in source_commits.items():
        for d in dmap:
            date_counts[d] += 1
    valid_dates = sorted(d for d, c in date_counts.items() if c >= 2)
    print(f"  {len(valid_dates)} ortak gün bulundu")

    # Her tarih için HTML'leri çek
    history = {}
    for i, date_str in enumerate(valid_dates):
        day_data = {}
        for key, (repo, path) in repos.items():
            sha = source_commits[key].get(date_str)
            if not sha:
                continue
            html = _fetch_file_at_sha(repo, path, sha)
            if html:
                day_data[key] = html
        if day_data:
            history[date_str] = day_data
            n_sources = len(day_data)
            sig_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
            print(f"  {sig_date}: {n_sources} kaynak", end="")
            if (i + 1) % 5 == 0:
                print()
            else:
                print("  |  ", end="")
        # Rate limiting
        if (i + 1) % 10 == 0:
            time.sleep(1)

    print(f"\n  Toplam: {len(history)} gün tarihsel veri")
    return history


# =============================================================================
# 2. HTML PARSE FONKSİYONLARI
# =============================================================================

def _extract_const_d(html_text):
    """HTML'den const D={...}; verisini çıkar."""
    m = re.search(r'const\s+D\s*=\s*(\{.+?\})\s*;', html_text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        raw = m.group(1)
        raw = re.sub(r',\s*([\]}])', r'\1', raw)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None


def _extract_const_data(html_text):
    """HTML'den const DATA=[...]; verisini çıkar."""
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
        except json.JSONDecodeError:
            return None


class _TavanTableParser(HTMLParser):
    """Tavan HTML table parse."""
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
    try:
        if isinstance(val, str):
            val = val.replace('x', '').replace('%', '').replace('+', '').strip()
            val = val.split('(')[0].strip()
            val = val.split()[0]
        return float(val)
    except (ValueError, TypeError, IndexError):
        return default


def _safe_int(val, default=0):
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return default


def parse_nw_from_html(html, date_str):
    """NW HTML'den sinyal listesi parse et.
    Returns: signal dicts listesi (html_signals.py formatı)
    """
    d = _extract_const_d(html)
    if not d:
        return []

    sig_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
    signals = []
    overlap = set(d.get('overlap', []))

    # Daily buys (shortlist için günlük gate açık sinyaller kullanılır)
    daily_date = d.get('daily', {}).get('date', sig_date)
    for b in d.get('daily', {}).get('buys', []):
        signals.append({
            'screener': 'nox_v3_daily',
            'ticker': b['ticker'],
            'signal_date': daily_date if daily_date else sig_date,
            'direction': 'AL',
            'signal_type': 'PIVOT_AL',
            'entry_price': b.get('close', 0),
            'fresh': b.get('fresh', 'YAKIN'),
            'gate': b.get('gate', False),
            'delta_pct': b.get('delta_pct'),
            'adx': b.get('adx'),
            'rsi': b.get('rsi'),
            'rg': b.get('rg'),
            'slope': b.get('slope'),
            'rs_score': b.get('rs_score'),
            'dw_overlap': b['ticker'] in overlap,
            'csv_date': date_str,
        })

    # Weekly buys (tam sinyal listesi)
    report_date = d.get('weekly', {}).get('date', sig_date)
    for b in d.get('weekly', {}).get('buys', []):
        signals.append({
            'screener': 'nox_v3_weekly',
            'ticker': b['ticker'],
            'signal_date': b.get('signal_date', report_date),
            'direction': 'AL',
            'signal_type': 'PIVOT_AL',
            'entry_price': b.get('close', 0),
            'trigger_type': b.get('trigger_type', ''),
            'wl_status': b.get('status', 'BEKLE'),
            'delta_pct': b.get('delta_pct'),
            'rs_score': b.get('rs_score'),
            'fresh': b.get('fresh', 'YAKIN'),
            'gate': b.get('gate', False),
            'dw_overlap': b['ticker'] in overlap,
            'csv_date': date_str,
        })

    # Weekly sells
    for s in d.get('weekly', {}).get('sells', []):
        signals.append({
            'screener': 'nox_v3_weekly',
            'ticker': s['ticker'],
            'signal_date': s.get('signal_date', report_date),
            'direction': 'SAT',
            'signal_type': 'PIVOT_SAT',
            'entry_price': s.get('close', 0),
            'fresh': s.get('fresh', 'YAKIN'),
            'csv_date': date_str,
        })

    return signals


def parse_rt_from_html(html, date_str):
    """RT HTML'den sinyal listesi parse et."""
    d = _extract_const_d(html)
    if not d:
        return []

    sig_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
    report_date = d.get('date', sig_date)
    # Tarih formatı "13.03.2026 15:58" → ISO
    if report_date and '.' in report_date and len(report_date) > 10:
        try:
            parts = report_date.split()[0].split('.')
            report_date = f"{parts[2]}-{parts[1]}-{parts[0]}"
        except (IndexError, ValueError):
            report_date = sig_date
    elif report_date and '.' in report_date:
        try:
            parts = report_date.split('.')
            report_date = f"{parts[2]}-{parts[1]}-{parts[0]}"
        except (IndexError, ValueError):
            report_date = sig_date

    signals = []
    for r in d.get('rows', []):
        entry_window = r.get('entry_window', '')
        if entry_window in ('TAZE', '2.DALGA', 'YAKIN'):
            direction = 'AL'
        else:
            direction = 'SAT'

        badge = ''
        if r.get('weekly_pb'):
            badge = 'H+PB'
        elif r.get('weekly_al'):
            badge = 'H+AL'

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
            'csv_date': date_str,
        }
        if badge:
            entry['badge'] = badge
        signals.append(entry)

    return signals


def parse_alsat_from_html(html, date_str):
    """AL/SAT HTML'den sinyal listesi parse et."""
    data = _extract_const_data(html)
    if not data:
        return []

    sig_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
    # Tarih: HTML'den
    date_m = re.search(r'(\d{2})\.(\d{2})\.(\d{4})', html)
    if date_m:
        report_date = f"{date_m.group(3)}-{date_m.group(2)}-{date_m.group(1)}"
    else:
        report_date = sig_date

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
            'csv_date': date_str,
        }
        if row.get('rs_score') is not None:
            entry['rs_score'] = round(row['rs_score'], 1) if isinstance(row['rs_score'], float) else row['rs_score']
        if row.get('macd_hist') is not None:
            entry['macd'] = round(row['macd_hist'], 4) if isinstance(row['macd_hist'], float) else row['macd_hist']
        if row.get('oe') is not None:
            entry['oe'] = row['oe']
        if row.get('rr') is not None:
            entry['rr'] = round(row['rr'], 2) if isinstance(row['rr'], float) else row['rr']
        signals.append(entry)

    return signals


def parse_tavan_from_html(html, date_str):
    """Tavan HTML'den sinyal listesi parse et."""
    parser = _TavanTableParser()
    parser.feed(html)
    if not parser.rows:
        return []

    sig_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
    date_m = re.search(r'(\d{2})\.(\d{2})\.(\d{4})', html)
    if date_m:
        report_date = f"{date_m.group(3)}-{date_m.group(2)}-{date_m.group(1)}"
    else:
        report_date = sig_date

    signals = []
    for row in parser.rows:
        cells = row['cells']
        if len(cells) < 7:
            continue
        skor = _safe_int(cells[2])
        karar = cells[3].strip() if len(cells) > 3 else ''
        seri = _safe_int(cells[4]) if len(cells) > 4 else 1
        fiyat = _safe_float(cells[5]) if len(cells) > 5 else 0
        hacim = _safe_float(cells[6]) if len(cells) > 6 else 1.0
        rsi = _safe_float(cells[8]) if len(cells) > 8 else 50
        rs = _safe_float(cells[9]) if len(cells) > 9 else 0

        signals.append({
            'screener': 'tavan',
            'ticker': row['ticker'],
            'signal_date': report_date,
            'direction': 'AL' if rs >= 0 else 'SAT',
            'signal_type': 'TAVAN',
            'entry_price': fiyat,
            'quality': skor,
            'skor': skor,
            'streak': seri,
            'volume_ratio': hacim,
            'rs': rs,
            'csv_date': date_str,
        })

    return signals


def parse_day_signals(day_htmls, date_str):
    """Bir günün tüm HTML'lerini parse et → birleşik sinyal listesi."""
    signals = []
    parsers = {
        'nw': parse_nw_from_html,
        'rt': parse_rt_from_html,
        'alsat': parse_alsat_from_html,
        'tavan': parse_tavan_from_html,
    }
    for key, parser_fn in parsers.items():
        html = day_htmls.get(key)
        if not html:
            continue
        try:
            sigs = parser_fn(html, date_str)
            signals.extend(sigs)
        except Exception as e:
            print(f"  ⚠️ {date_str}/{key} parse hatası: {e}")
    return signals


# =============================================================================
# 3. SHORTLIST RECONSTRUCT (briefing.py _compute_4_lists replikası)
# =============================================================================

def _compute_4_lists_backtest(latest_signals, date_str, tier2_min_score=100):
    """briefing.py:_compute_4_lists() replikası — SM verileri olmadan.

    Returns: dict with keys: 'alsat', 'tavan', 'nw', 'rt', 'tier1', 'tier2'
    Her value: [(ticker, score, list_name, signal_dict), ...]
    """
    sig_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"

    # Screener bazlı en son tarih
    _screener_latest_sd = {}
    for s in latest_signals:
        scr = s.get('screener', '')
        sd = s.get('signal_date', '')
        if sd and sd > _screener_latest_sd.get(scr, ''):
            _screener_latest_sd[scr] = sd

    def _is_today(s):
        sd = s.get('signal_date', '')
        cd = s.get('csv_date', '')
        if sd == sig_date or cd == date_str:
            return True
        scr = s.get('screener', '')
        return sd == _screener_latest_sd.get(scr, '')

    # RT haritası (CMF cross-ref için)
    rt_map = {}
    for s in latest_signals:
        if s.get('screener') == 'regime_transition':
            rt_map[s['ticker']] = s

    # ── LİSTE 1: AL/SAT Tarama ──
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

        passes = False
        tier_label = ''
        # ZAYIF çıkarıldı — backtest: 1G WR %33, 3G WR %0
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
            score = 400 + q
        elif sig_type in ('DONUS', 'DÖNÜŞ') and q >= 70:
            passes = True
            tier_label = 'DÖNÜŞ'
            score = 50 + q

        if not passes:
            continue

        alsat_items.append((s['ticker'], score, [tier_label], s))

    alsat_items.sort(key=lambda x: -x[1])

    # ── LİSTE 2: Tavan Tarayıcı ──
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

        cmf = None
        rt_sig = rt_map.get(ticker)
        if rt_sig:
            cmf = rt_sig.get('cmf')

        # Skor hesaplama — data-driven (5Y backtest, N=23763)
        score = skor * 5

        # A) Streak (en güçlü prediktör)
        if seri >= 3:
            score += 150
        elif seri >= 2:
            score += 80

        # B) Skor zone
        if skor <= 49:
            score += 40
        elif 60 <= skor <= 79:
            score -= 60
        elif 50 <= skor <= 59:
            score -= 30

        # C) Volume
        if vol < 1.0:
            score += 50
        elif vol < 1.5:
            score += 30
        elif vol < 2.0:
            pass
        elif vol < 3.0:
            score -= 20
        elif vol < 5.0:
            score -= 50
        elif vol < 10:
            score -= 80
        elif vol < 20:
            score -= 120
        else:
            score -= 200

        # D) CMF
        if cmf is not None and cmf > 0:
            score += int(cmf * 50)

        reasons = []
        if seri >= 3:
            reasons.append(f"🔥seri:{seri}")
        elif seri >= 2:
            reasons.append(f"seri:{seri}")
        if skor <= 49:
            reasons.append(f"▲skor:{skor}")
        elif skor >= 80:
            reasons.append(f"🔒skor:{skor}")

        tavan_items.append((ticker, score, reasons, s))

    tavan_items.sort(key=lambda x: -x[1])

    # ── LİSTE 3: NW Pivot AL (Günlük) ──
    nw_items = []
    for s in latest_signals:
        if s.get('screener') != 'nox_v3_daily':
            continue
        if s.get('direction') != 'AL':
            continue
        fresh = s.get('fresh', '')
        if fresh not in ('BUGUN', 'BUGÜN'):
            continue
        if not s.get('gate'):
            continue
        delta = s.get('delta_pct')
        dw = s.get('dw_overlap', False)

        score = 30
        if dw:
            score += 50
        if delta is not None:
            score += max(0, int(20 - delta))

        reasons = []
        if dw:
            reasons.append('D+W')

        nw_items.append((s['ticker'], score, reasons, s))

    nw_items.sort(key=lambda x: -x[1])

    # ── LİSTE 4: Regime Transition ──
    rt_items = []
    for s in latest_signals:
        if s.get('screener') != 'regime_transition' or s.get('direction') != 'AL':
            continue
        if not _is_today(s):
            continue
        window = s.get('entry_window', '')
        if window not in ('TAZE', '2.DALGA'):
            continue
        badge = s.get('badge', '')
        if not badge:
            continue  # no_badge RT zayıf — backtest: 1G med -0.34
        entry_score = int(s.get('quality', 0) or 0)
        if entry_score < 3:
            continue
        oe = int(s.get('oe', 0) or 0)
        if oe > 2:
            continue

        t_date = s.get('transition_date', '')
        is_today_transition = (t_date == sig_date)

        score = 0
        if badge == 'H+PB':
            score += 100
        elif badge == 'H+AL':
            score += 80
        elif badge:
            score += 60
        score += entry_score * 10
        window_pts = {'TAZE': 20, '2.DALGA': 10}.get(window, 0)
        score += window_pts
        if window == 'TAZE' and is_today_transition:
            score += 15

        reasons = []
        if badge:
            reasons.append(badge)
        reasons.append(window)

        rt_items.append((s['ticker'], score, reasons, s))

    rt_items.sort(key=lambda x: -x[1])

    # ── Çapraz Çakışma ──
    list_data = {'alsat': alsat_items, 'tavan': tavan_items, 'nw': nw_items, 'rt': rt_items}
    _LIST_SHORT = {'alsat': 'AS', 'tavan': 'TVN', 'nw': 'NW', 'rt': 'RT'}
    ticker_list_count = {}
    for list_name, items in list_data.items():
        for ticker, _, _, _ in items:
            ticker_list_count.setdefault(ticker, set()).add(list_name)

    def _overlap_quality(ticker):
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
                    quality += entry_s * 8
                    if sig.get('entry_window') == 'TAZE':
                        quality += 10
                    cmf = sig.get('cmf', 0) or 0
                    if cmf > 0.1:
                        quality += 5
                elif list_name == 'nw':
                    if sig.get('dw_overlap'):
                        quality += 35
                    else:
                        quality += 20
                    delta = sig.get('delta_pct')
                    if delta is not None and delta < 10:
                        quality += 10
                elif list_name == 'alsat':
                    sig_type = sig.get('signal_type', '')
                    if sig_type in ('CMB', 'CMB+'):
                        quality += 35
                    elif sig_type in ('GUCLU', 'GÜÇLÜ'):
                        quality += 30
                    elif sig_type in ('BILESEN', 'BİLEŞEN'):
                        quality += 25
                    else:
                        quality += 10
                    q = sig.get('quality', 0) or 0
                    quality += q // 10
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
                    if skor <= 49:
                        quality += 15
                    elif 60 <= skor <= 79:
                        quality -= 15
                    if vol < 1.5:
                        quality += 10
                    elif vol > 3.0:
                        quality -= 10
                break
        if len(in_lists) >= 3:
            quality += 20
        elif len(in_lists) >= 2:
            quality += 5
        has_technical = bool({'alsat', 'tavan'} & set(in_lists))
        has_structural = bool({'nw', 'rt'} & set(in_lists))
        if has_technical and has_structural:
            if set(in_lists) == {'alsat', 'rt'}:
                quality -= 25  # AS+RT penalty
            else:
                quality += 15
        return quality, in_lists

    # Tier 1: 2+ liste çakışma
    tier1 = []
    for ticker, lists in ticker_list_count.items():
        if len(lists) < 2:
            continue
        quality, in_lists = _overlap_quality(ticker)
        if quality < 40:
            continue
        has_tech = bool({'alsat', 'tavan'} & set(in_lists))
        has_struct = bool({'nw', 'rt'} & set(in_lists))
        list_tags = "+".join(_LIST_SHORT.get(l, l) for l in sorted(in_lists))
        tier1.append((ticker, quality, [list_tags], {
            'overlap_count': len(in_lists),
            'in_lists': in_lists,
            'ty': has_tech and has_struct,
        }))

    # Gevşek RT çakışma
    tier1_tickers = {t for t, _, _, _ in tier1}
    strong_tickers = {}
    for t, sc, reas, sig in alsat_items:
        st = sig.get('signal_type', '')
        if st in ('GUCLU', 'GÜÇLÜ', 'CMB', 'CMB+', 'BILESEN', 'BİLEŞEN'):
            strong_tickers[t] = ('alsat', sig)
    for t, sc, reas, sig in tavan_items:
        seri = sig.get('streak', 0) or 0
        skor = sig.get('skor', 0) or 0
        vol = sig.get('volume_ratio', 0) or 0
        if seri >= 2 or (skor <= 49 and vol < 2.0):
            strong_tickers.setdefault(t, ('tavan', sig))
    for t, sc, reas, sig in nw_items:
        if sig.get('dw_overlap'):
            strong_tickers.setdefault(t, ('nw', sig))

    for s in latest_signals:
        if s.get('screener') != 'regime_transition' or s.get('direction') != 'AL':
            continue
        if not _is_today(s):
            continue
        ticker = s['ticker']
        if ticker in tier1_tickers or ticker not in strong_tickers:
            continue
        badge = s.get('badge', '')
        if not badge:
            continue
        window = s.get('entry_window', '')
        if window not in ('TAZE', '2.DALGA'):
            continue
        oe = int(s.get('oe', 0) or 0)
        if oe > 3:
            continue

        src_name, src_sig = strong_tickers[ticker]
        quality, _ = _overlap_quality(ticker)
        rt_quality = 20
        if badge == 'H+PB':
            rt_quality += 15
        if window == 'TAZE':
            rt_quality += 5
        quality += rt_quality

        src_tag = _LIST_SHORT[src_name]
        is_structural = True
        is_technical = src_name in ('alsat', 'tavan')
        tier1.append((ticker, quality, [f"{src_tag}+RT"], {
            'overlap_count': 2,
            'in_lists': [src_name, 'rt'],
            'relaxed': True,
            'ty': is_technical and is_structural,
        }))
        tier1_tickers.add(ticker)

    tier1.sort(key=lambda x: -x[1])

    # Tier 2: her listeden max 5 tekil
    # Score >= 100 soft gate: tekil/Tier2 için düşük kalite kuyrukları budanır
    _TIER2_MIN_SCORE = tier2_min_score
    tier1_tickers = {t for t, _, _, _ in tier1}
    tier2 = []
    for list_name in ('nw', 'rt', 'alsat', 'tavan'):
        items = list_data[list_name]
        count = 0
        for ticker, score, reasons, sig in items:
            if ticker in tier1_tickers:
                continue
            if score < _TIER2_MIN_SCORE:
                continue  # Düşük score tekil sinyal filtrelenir
            if ticker in {t for t, _, _, _ in tier2}:
                continue
            tag = _LIST_SHORT[list_name]
            tier2.append((ticker, score, [tag], sig))
            count += 1
            if count >= 5:
                break

    result = dict(list_data)
    result['tier1'] = tier1
    result['tier2'] = tier2
    return result


# =============================================================================
# 3b. TABAN RİSKİ HESAPLAMA (backtest mirror)
# =============================================================================

def _calc_taban_risk_bt(df, sig_date=None):
    """Backtest OHLCV'den taban riski hesapla.

    sig_date verilirse sadece o tarihe kadarki veriyi kullanır (look-ahead yok).
    Returns: dict with taban_days, atr_pct, max_drop_60d, risk (0-7)
    """
    if sig_date is not None:
        df = df.loc[:sig_date]
    if len(df) < 60:
        return {'taban_days': 0, 'atr_pct': 0.0, 'max_drop_60d': 0.0, 'risk': 0}

    returns = df['Close'].pct_change()

    # 1) Geçmiş taban sayısı (günlük düşüş >= 9% ≈ taban)
    taban_days = int((returns <= -0.09).sum())

    # 2) ATR% (son 14 günlük)
    high = df['High']
    low = df['Low']
    close = df['Close']
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    atr14 = tr.rolling(14).mean().iloc[-1]
    atr_pct = (atr14 / close.iloc[-1]) * 100 if close.iloc[-1] > 0 else 0.0

    # 3) Son 60 günde en sert tek gün düşüş
    recent = returns.tail(60)
    max_drop = float(recent.min()) * 100 if len(recent) > 0 else 0.0

    # 4) Risk skoru (0-7)
    risk = 0
    if taban_days >= 3:   risk += 3
    elif taban_days >= 1: risk += 1
    if atr_pct > 8:       risk += 2
    elif atr_pct > 5:     risk += 1
    if max_drop < -7:     risk += 2
    elif max_drop < -5:   risk += 1

    return {
        'taban_days': taban_days,
        'atr_pct': round(atr_pct, 1),
        'max_drop_60d': round(max_drop, 1),
        'risk': risk,
    }


def _enrich_trades_taban_risk(results, all_data):
    """Trade listesine taban_risk bilgisi ekle (in-place)."""
    for trade in results:
        ticker = trade.get('ticker')
        df = all_data.get(ticker)
        if df is None or df.empty:
            trade['taban_risk'] = 0
            continue
        try:
            sig_date = pd.Timestamp(trade['signal_date'])
            tr = _calc_taban_risk_bt(df, sig_date=sig_date)
            trade['taban_risk'] = tr['risk']
            trade['taban_days'] = tr['taban_days']
            trade['taban_atr_pct'] = tr['atr_pct']
            trade['taban_max_drop'] = tr['max_drop_60d']
        except Exception:
            trade['taban_risk'] = 0


# =============================================================================
# 4. FORWARD GETİRİ HESAPLAMA
# =============================================================================

def compute_forward_returns(trades, all_data, xu_df):
    """Her trade için 1G/3G/5G forward return hesapla."""
    results = []
    for trade in trades:
        ticker = trade['ticker']
        df = all_data.get(ticker)
        if df is None or df.empty:
            continue

        try:
            sig_date = pd.Timestamp(trade['signal_date'])
        except Exception:
            continue

        idx = df.index.searchsorted(sig_date)
        if idx >= len(df):
            idx = len(df) - 1
        if abs((df.index[idx] - sig_date).days) > 5:
            continue

        # Entry price: her zaman fiyat verisinden al (HTML parse hataları olabilir)
        actual_close = float(df['Close'].iloc[idx])
        entry_price = trade.get('entry_price', 0)
        if not entry_price or entry_price <= 0:
            entry_price = actual_close
        # Fiyat verisiyle çok farklıysa (>%50), fiyat verisini kullan
        elif actual_close > 0 and abs(entry_price - actual_close) / actual_close > 0.5:
            entry_price = actual_close

        row = {**trade, 'entry_price': entry_price}
        status = 'bekliyor'

        for w in WINDOWS:
            fwd_idx = idx + w
            if fwd_idx < len(df):
                fwd_close = float(df['Close'].iloc[fwd_idx])
                ret = (fwd_close / entry_price - 1) * 100
                # Aşırı getiri cap: ±50% (veri hatası koruması)
                ret = max(-50, min(50, ret))
                row[f'ret_{w}d'] = round(ret, 2)

                if xu_df is not None and not xu_df.empty:
                    xu_s = xu_df.index.searchsorted(df.index[idx])
                    xu_e = xu_df.index.searchsorted(df.index[fwd_idx])
                    if xu_s < len(xu_df) and xu_e < len(xu_df):
                        xu_ret = (float(xu_df['Close'].iloc[xu_e]) /
                                  float(xu_df['Close'].iloc[xu_s]) - 1) * 100
                        row[f'xu_{w}d'] = round(xu_ret, 2)
                        row[f'excess_{w}d'] = round(ret - xu_ret, 2)
                    else:
                        row[f'xu_{w}d'] = None
                        row[f'excess_{w}d'] = None
                else:
                    row[f'xu_{w}d'] = None
                    row[f'excess_{w}d'] = None

                if w == 1:
                    status = 'kısmi'
            else:
                row[f'ret_{w}d'] = None
                row[f'xu_{w}d'] = None
                row[f'excess_{w}d'] = None

        if all(row.get(f'ret_{w}d') is not None for w in WINDOWS):
            status = 'tamam'
        row['status'] = status
        results.append(row)

    return results


# =============================================================================
# 5. ANALİZ METRİKLERİ
# =============================================================================

def _calc_stats(subset, windows=None):
    """Alt küme için pencere bazlı istatistikler."""
    if windows is None:
        windows = WINDOWS
    stats = {}
    for w in windows:
        key = f'ret_{w}d'
        vals = [r[key] for r in subset if r.get(key) is not None]
        if vals:
            wins = sum(1 for v in vals if v > 0)
            stats[f'n_{w}d'] = len(vals)
            stats[f'wr_{w}d'] = round(wins / len(vals) * 100, 1)
            stats[f'avg_{w}d'] = round(sum(vals) / len(vals), 2)
            stats[f'med_{w}d'] = round(median(vals), 2)
            stats[f'best_{w}d'] = round(max(vals), 2)
            stats[f'worst_{w}d'] = round(min(vals), 2)
        else:
            stats[f'n_{w}d'] = 0
            for k in ['wr', 'avg', 'med', 'best', 'worst']:
                stats[f'{k}_{w}d'] = None
        # XU100 excess
        ex_key = f'excess_{w}d'
        ex_vals = [r[ex_key] for r in subset if r.get(ex_key) is not None]
        if ex_vals:
            stats[f'excess_avg_{w}d'] = round(sum(ex_vals) / len(ex_vals), 2)
        else:
            stats[f'excess_avg_{w}d'] = None
    return stats


def compute_all_analysis(all_trades):
    """Tam detay analiz metrikleri hesapla.

    Returns: dict with analysis sections
    """
    analysis = {}

    # ── Liste bazlı ──
    list_stats = {}
    for list_name in ('alsat', 'tavan', 'nw', 'rt'):
        sub = [t for t in all_trades if t.get('list') == list_name]
        if sub:
            list_stats[list_name] = {'n': len(sub), **_calc_stats(sub)}
    analysis['lists'] = list_stats

    # ── Tier bazlı ──
    tier_stats = {}
    for tier in ('tier1', 'tier2', 'list_only'):
        sub = [t for t in all_trades if t.get('tier') == tier]
        if sub:
            tier_stats[tier] = {'n': len(sub), **_calc_stats(sub)}
    analysis['tiers'] = tier_stats

    # ── Overlap detay ──
    overlap_stats = {}
    # 2-list vs 3+ list
    ol2 = [t for t in all_trades if t.get('tier') == 'tier1' and t.get('overlap_count', 0) == 2]
    ol3 = [t for t in all_trades if t.get('tier') == 'tier1' and t.get('overlap_count', 0) >= 3]
    if ol2:
        overlap_stats['2_list'] = {'n': len(ol2), **_calc_stats(ol2)}
    if ol3:
        overlap_stats['3+_list'] = {'n': len(ol3), **_calc_stats(ol3)}

    # Çift kombinasyonlar
    combos = defaultdict(list)
    for t in all_trades:
        if t.get('tier') == 'tier1':
            in_lists = t.get('in_lists', [])
            if len(in_lists) >= 2:
                # Her çift
                for i in range(len(in_lists)):
                    for j in range(i + 1, len(in_lists)):
                        pair = tuple(sorted([in_lists[i], in_lists[j]]))
                        combos[pair].append(t)
    for pair, trades in combos.items():
        key = '+'.join(pair)
        if trades:
            overlap_stats[key] = {'n': len(trades), **_calc_stats(trades)}

    analysis['overlaps'] = overlap_stats

    # ── T+Y analiz ──
    ty_trades = [t for t in all_trades if t.get('ty')]
    no_ty_trades = [t for t in all_trades if t.get('tier') == 'tier1' and not t.get('ty')]
    ty_stats = {}
    if ty_trades:
        ty_stats['ty'] = {'n': len(ty_trades), **_calc_stats(ty_trades)}
    if no_ty_trades:
        ty_stats['no_ty'] = {'n': len(no_ty_trades), **_calc_stats(no_ty_trades)}
    analysis['ty'] = ty_stats

    # ── Sinyal tipi kırılımı ──
    sig_type_stats = {}
    # AL/SAT sinyal tipleri
    for st in ('CMB', 'CMB+', 'ZAYIF', 'GUCLU', 'BILESEN', 'DONUS'):
        sub = [t for t in all_trades if t.get('list') == 'alsat'
               and t.get('signal_type', '') == st]
        if sub:
            sig_type_stats[f'AS_{st}'] = {'n': len(sub), **_calc_stats(sub)}
    # RT badge
    for badge in ('H+PB', 'H+AL', ''):
        label = badge if badge else 'no_badge'
        sub = [t for t in all_trades if t.get('list') == 'rt'
               and t.get('badge', '') == badge]
        if sub:
            sig_type_stats[f'RT_{label}'] = {'n': len(sub), **_calc_stats(sub)}
    # NW D+W vs daily-only
    dw = [t for t in all_trades if t.get('list') == 'nw' and t.get('dw_overlap')]
    nondw = [t for t in all_trades if t.get('list') == 'nw' and not t.get('dw_overlap')]
    if dw:
        sig_type_stats['NW_DW'] = {'n': len(dw), **_calc_stats(dw)}
    if nondw:
        sig_type_stats['NW_daily'] = {'n': len(nondw), **_calc_stats(nondw)}
    # Tavan kilitli (skor>=50) vs normal
    kilitli = [t for t in all_trades if t.get('list') == 'tavan'
               and (t.get('skor', 0) or 0) >= 50]
    non_kilitli = [t for t in all_trades if t.get('list') == 'tavan'
                   and (t.get('skor', 0) or 0) < 50]
    if kilitli:
        sig_type_stats['TVN_kilitli'] = {'n': len(kilitli), **_calc_stats(kilitli)}
    if non_kilitli:
        sig_type_stats['TVN_normal'] = {'n': len(non_kilitli), **_calc_stats(non_kilitli)}
    analysis['signal_types'] = sig_type_stats

    # ── Score band ──
    score_bands = {}
    bands = [(400, 999, '400+'), (300, 399, '300-399'), (200, 299, '200-299'),
             (100, 199, '100-199'), (50, 99, '50-99'), (0, 49, '0-49')]
    for lo, hi, label in bands:
        sub = [t for t in all_trades if lo <= t.get('score', 0) <= hi]
        if sub:
            score_bands[label] = {'n': len(sub), **_calc_stats(sub)}
    analysis['score_bands'] = score_bands

    # ── Ek özellik kırılımları (1G/3G analiz) ──
    feature_stats = {}

    # Quality buckets (alsat)
    for lo, hi, label in [(75, 100, 'q_75_100'), (50, 74, 'q_50_74'),
                          (25, 49, 'q_25_49'), (0, 24, 'q_0_24')]:
        sub = [t for t in all_trades if lo <= (t.get('quality') or 0) <= hi
               and t.get('quality') is not None]
        if sub:
            feature_stats[label] = {'n': len(sub), **_calc_stats(sub)}

    # Tavan skor buckets
    for lo, hi, label in [(80, 100, 'tvn_skor_80+'), (60, 79, 'tvn_skor_60_79'),
                          (50, 59, 'tvn_skor_50_59'), (30, 49, 'tvn_skor_30_49'),
                          (0, 29, 'tvn_skor_0_29')]:
        sub = [t for t in all_trades if t.get('list') in ('tavan',)
               and lo <= (t.get('skor') or 0) <= hi]
        if sub:
            feature_stats[label] = {'n': len(sub), **_calc_stats(sub)}

    # Volume ratio buckets (tavan)
    for lo, hi, label in [(3.0, 999, 'tvn_vr_3+'), (2.0, 2.99, 'tvn_vr_2_3'),
                          (1.5, 1.99, 'tvn_vr_1.5_2'), (1.0, 1.49, 'tvn_vr_1_1.5'),
                          (0, 0.99, 'tvn_vr_<1')]:
        sub = [t for t in all_trades if t.get('list') in ('tavan',)
               and t.get('volume_ratio') is not None
               and lo <= (t.get('volume_ratio') or 0) <= hi]
        if sub:
            feature_stats[label] = {'n': len(sub), **_calc_stats(sub)}

    # Streak buckets (tavan)
    for lo, hi, label in [(3, 99, 'tvn_streak_3+'), (2, 2, 'tvn_streak_2'),
                          (1, 1, 'tvn_streak_1')]:
        sub = [t for t in all_trades if t.get('list') in ('tavan',)
               and lo <= (t.get('streak') or 0) <= hi]
        if sub:
            feature_stats[label] = {'n': len(sub), **_calc_stats(sub)}

    # RT entry_window
    for ew in ('TAZE', 'YAKIN', ''):
        label = f"rt_ew_{ew}" if ew else 'rt_ew_none'
        sub = [t for t in all_trades if t.get('list') in ('rt',)
               and t.get('entry_window', '') == ew]
        if sub:
            feature_stats[label] = {'n': len(sub), **_calc_stats(sub)}

    # RT oe (overextension) buckets
    for lo, hi, label in [(3, 4, 'rt_oe_3_4'), (1, 2, 'rt_oe_1_2'), (0, 0, 'rt_oe_0')]:
        sub = [t for t in all_trades if t.get('list') in ('rt',)
               and lo <= (t.get('oe') or 0) <= hi]
        if sub:
            feature_stats[label] = {'n': len(sub), **_calc_stats(sub)}

    # NW gate
    for gv, label in [(True, 'nw_gate_open'), (False, 'nw_gate_closed')]:
        sub = [t for t in all_trades if t.get('list') in ('nw',)
               and bool(t.get('gate')) == gv]
        if sub:
            feature_stats[label] = {'n': len(sub), **_calc_stats(sub)}

    # NW delta_pct buckets
    for lo, hi, label in [(0, 3, 'nw_delta_0_3'), (3, 6, 'nw_delta_3_6'),
                          (6, 10, 'nw_delta_6_10'), (10, 99, 'nw_delta_10+')]:
        sub = [t for t in all_trades if t.get('list') in ('nw',)
               and t.get('delta_pct') is not None
               and lo <= abs(t.get('delta_pct') or 0) < hi]
        if sub:
            feature_stats[label] = {'n': len(sub), **_calc_stats(sub)}

    # RS score buckets (alsat)
    for lo, hi, label in [(1.5, 99, 'as_rs_1.5+'), (1.0, 1.49, 'as_rs_1_1.5'),
                          (0.5, 0.99, 'as_rs_0.5_1'), (0, 0.49, 'as_rs_<0.5')]:
        sub = [t for t in all_trades if t.get('list') in ('alsat',)
               and t.get('rs_score') is not None
               and lo <= (t.get('rs_score') or 0) <= hi]
        if sub:
            feature_stats[label] = {'n': len(sub), **_calc_stats(sub)}

    # CMF buckets (RT)
    for lo, hi, label in [(0.1, 1.0, 'rt_cmf_0.1+'), (0, 0.099, 'rt_cmf_0_0.1'),
                          (-1.0, -0.001, 'rt_cmf_neg')]:
        sub = [t for t in all_trades if t.get('list') in ('rt',)
               and t.get('cmf') is not None
               and lo <= (t.get('cmf') or 0) <= hi]
        if sub:
            feature_stats[label] = {'n': len(sub), **_calc_stats(sub)}

    # Taban risk buckets
    for lo, hi, label in [(4, 7, 'taban_risk_high'), (2, 3, 'taban_risk_med'),
                          (0, 1, 'taban_risk_low')]:
        sub = [t for t in all_trades
               if lo <= (t.get('taban_risk') or 0) <= hi]
        if sub:
            feature_stats[label] = {'n': len(sub), **_calc_stats(sub)}

    analysis['features'] = feature_stats

    # ── Günlük performans ──
    daily_perf = {}
    all_dates = sorted(set(t.get('csv_date', '') for t in all_trades if t.get('csv_date')))
    for d in all_dates:
        sub = [t for t in all_trades if t.get('csv_date') == d]
        if sub:
            daily_perf[d] = {'n': len(sub), **_calc_stats(sub)}
    analysis['daily'] = daily_perf

    # ── En iyi/kötü 20 ──
    with_5d = [t for t in all_trades if t.get('ret_5d') is not None]
    with_5d.sort(key=lambda x: x['ret_5d'], reverse=True)
    analysis['top20'] = with_5d[:20]
    analysis['bottom20'] = with_5d[-20:]

    # ── Genel ──
    analysis['overall'] = {'n': len(all_trades), **_calc_stats(all_trades)}

    return analysis


# =============================================================================
# 6. FAZ 1: GITHUB HTML GEÇMİŞİNDEN BACKTEST
# =============================================================================

def run_phase1():
    """GitHub HTML commit geçmişinden tarihsel shortlist backtest."""
    print("\n" + "=" * 60)
    print("  FAZ 1: GitHub HTML Geçmişinden Backtest")
    print("=" * 60)

    # 1. HTML geçmişini çek
    history = fetch_all_html_history()
    if not history:
        print("  HATA: Tarihsel veri bulunamadı!")
        return [], {}

    # 2. Her gün için sinyalleri parse et + shortlist oluştur
    print("\n[2] Sinyaller parse ediliyor + shortlist oluşturuluyor...")
    all_trades = []
    daily_stats = {}
    _LIST_SHORT = {'alsat': 'AS', 'tavan': 'TVN', 'nw': 'NW', 'rt': 'RT'}

    for date_str, day_htmls in sorted(history.items()):
        signals = parse_day_signals(day_htmls, date_str)
        if not signals:
            continue

        lists = _compute_4_lists_backtest(signals, date_str)
        sig_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"

        day_count = 0

        # Tier 1 trades
        for ticker, quality, reasons, meta in lists.get('tier1', []):
            in_lists = meta.get('in_lists', [])
            # Bu ticker'ın giriş fiyatını bul
            entry_price = 0
            entry_sig = None
            for ln in in_lists:
                for t, sc, r, sig in lists.get(ln, []):
                    if t == ticker:
                        entry_price = sig.get('entry_price', 0)
                        entry_sig = sig
                        break
                if entry_price:
                    break

            trade = {
                'ticker': ticker,
                'signal_date': sig_date,
                'csv_date': date_str,
                'entry_price': entry_price,
                'score': quality,
                'tier': 'tier1',
                'list': 'tier1',
                'overlap_count': meta.get('overlap_count', 0),
                'in_lists': in_lists,
                'ty': meta.get('ty', False),
                'relaxed': meta.get('relaxed', False),
                'list_tags': '+'.join(_LIST_SHORT.get(l, l) for l in sorted(in_lists)),
                'source': 'github_html',
            }
            # Sinyal detaylarını ekle
            if entry_sig:
                for k in ('signal_type', 'badge', 'dw_overlap', 'entry_window',
                           'quality', 'skor', 'volume_ratio', 'karar', 'rs_score', 'macd'):
                    if k in entry_sig:
                        trade[k] = entry_sig[k]
            all_trades.append(trade)
            day_count += 1

        # Tier 2 trades
        for ticker, score, reasons, sig in lists.get('tier2', []):
            trade = {
                'ticker': ticker,
                'signal_date': sig_date,
                'csv_date': date_str,
                'entry_price': sig.get('entry_price', 0),
                'score': score,
                'tier': 'tier2',
                'list': reasons[0] if reasons else '?',
                'source': 'github_html',
            }
            for k in ('signal_type', 'badge', 'dw_overlap', 'entry_window',
                       'quality', 'skor', 'volume_ratio', 'karar', 'rs_score', 'macd'):
                if k in sig:
                    trade[k] = sig[k]
            all_trades.append(trade)
            day_count += 1

        # Tüm liste sinyalleri (tier'a girmeyen)
        tier_tickers = {t['ticker'] for t in all_trades if t.get('csv_date') == date_str}
        for list_name in ('alsat', 'tavan', 'nw', 'rt'):
            for ticker, score, reasons, sig in lists.get(list_name, []):
                if ticker in tier_tickers:
                    continue
                trade = {
                    'ticker': ticker,
                    'signal_date': sig_date,
                    'csv_date': date_str,
                    'entry_price': sig.get('entry_price', 0),
                    'score': score,
                    'tier': 'list_only',
                    'list': list_name,
                    'source': 'github_html',
                }
                for k in ('signal_type', 'badge', 'dw_overlap', 'entry_window',
                           'quality', 'skor', 'volume_ratio', 'karar', 'rs_score', 'macd'):
                    if k in sig:
                        trade[k] = sig[k]
                all_trades.append(trade)
                tier_tickers.add(ticker)
                day_count += 1

        n_t1 = len(lists.get('tier1', []))
        n_t2 = len(lists.get('tier2', []))
        daily_stats[date_str] = {
            'n_signals': len(signals),
            'n_trades': day_count,
            'n_tier1': n_t1,
            'n_tier2': n_t2,
            'n_as': len(lists.get('alsat', [])),
            'n_tvn': len(lists.get('tavan', [])),
            'n_nw': len(lists.get('nw', [])),
            'n_rt': len(lists.get('rt', [])),
        }

    print(f"\n  Toplam: {len(all_trades)} trade, {len(daily_stats)} gün")

    # 3. Fiyat verisi çek
    if not all_trades:
        return [], {}, None, None, None

    print("\n[3] Fiyat verisi çekiliyor...")
    tickers = sorted(set(t['ticker'] for t in all_trades))
    print(f"  {len(tickers)} unique ticker")
    t0 = time.time()
    all_data = data_mod.fetch_data(tickers, period="6mo")
    xu_df = data_mod.fetch_benchmark(period="6mo")
    print(f"  Veri çekildi ({time.time() - t0:.1f}s)")

    # 4. Forward getiri hesapla
    print("\n[4] Forward getiriler hesaplanıyor...")
    results = compute_forward_returns(all_trades, all_data, xu_df)
    n_tamam = sum(1 for r in results if r['status'] == 'tamam')
    n_kismi = sum(1 for r in results if r['status'] == 'kısmi')
    print(f"  {len(results)} trade: {n_tamam} tamam, {n_kismi} kısmi")

    # 5. Taban risk enrichment
    _enrich_trades_taban_risk(results, all_data)

    return results, daily_stats, history, all_data, xu_df


# =============================================================================
# 7. FAZ 2: SCREENER TARİHSEL BACKTEST (--full)
# =============================================================================

# ── AL/SAT Signal Generator ──

def _generate_alsat_signals(all_data, xu_df, start_date, end_date):
    """regime.py:analyze_regime() mantığını her hisse için tek geçişte çalıştır.

    Her trading day için df'i o güne kadar slice edip analyze_regime() çağırır.
    Yavaş ama doğru — tam fidelity, regime.py ile birebir eşleşir.
    """
    from markets.bist.regime import analyze_regime

    xu_lower = None
    if xu_df is not None and not xu_df.empty:
        xu_lower = xu_df.copy()

    signals = []
    tickers = sorted(all_data.keys())
    total = len(tickers)
    n_signals = 0

    # AL/SAT signal type → shortlist signal_type mapping
    _SIG_MAP = {
        'COMBO+': 'CMB', 'COMBO': 'CMB', 'STRONG': 'GUCLU', 'WEAK': 'ZAYIF',
        'PARTIAL': 'BILESEN', 'REVERSAL': 'DONUS', 'EARLY': 'ERKEN',
        'BUILDUP': 'BILESEN', 'PULLBACK': 'BILESEN', 'SQUEEZE': 'BILESEN',
        'MEANREV': 'DONUS',
    }

    # İşlem günlerini bul (tüm hisseler üzerinden)
    all_dates = set()
    for df in all_data.values():
        all_dates.update(df.index)
    trading_days = sorted(d for d in all_dates if start_date <= d <= end_date)

    # Her hisse için: sadece piyasa kapanış günlerinde analyze_regime çağır
    # Optimizasyon: haftada 1 kez sample al (her Cuma veya haftanın son günü)
    # Bu şekilde ~609 × ~260 = ~158K çağrı / yıl
    sample_days = []
    week_days = defaultdict(list)
    for d in trading_days:
        yw = d.isocalendar()[:2]
        week_days[yw].append(d)
    for yw in sorted(week_days.keys()):
        sample_days.append(week_days[yw][-1])  # haftanın son günü

    print(f"  AL/SAT: {total} hisse × {len(sample_days)} hafta (~{total * len(sample_days) // 1000}K çağrı)")

    for idx, ticker in enumerate(tickers):
        if (idx + 1) % 100 == 0:
            print(f"    [{idx+1}/{total}] {ticker} — {n_signals} sinyal")

        daily_raw = all_data[ticker]
        if daily_raw is None or len(daily_raw) < 120:
            continue

        for sample_date in sample_days:
            # df'i sample_date'e kadar slice et
            mask = daily_raw.index <= sample_date
            df_slice = daily_raw.loc[mask]
            if len(df_slice) < 80:
                continue

            try:
                result = analyze_regime(ticker, df_slice, xu_lower)
            except Exception:
                continue

            if result is None:
                continue

            sig_type = result.get('signal')
            if not sig_type:
                continue

            mapped = _SIG_MAP.get(sig_type, sig_type)
            karar = 'AL'

            signals.append({
                'screener': 'alsat',
                'ticker': ticker,
                'signal_date': sample_date.strftime('%Y-%m-%d'),
                'direction': karar,
                'karar': karar,
                'signal_type': mapped,
                'quality': result.get('quality', 0),
                'rs_score': result.get('rs_score', 0),
                'macd': result.get('macd_hist') if 'macd_hist' in result else None,
                'entry_price': result.get('close', 0),
                'oe': result.get('overext_score', 0),
            })
            n_signals += 1

    print(f"    AL/SAT: {n_signals} sinyal üretildi")
    return signals


# ── NW Pivot AL Signal Generator ──

def _generate_nw_signals(all_data, xu_df, start_date, end_date):
    """run_backtest_elmas.py pattern — NW pivot + daily trigger."""
    from markets.bist.nox_v3_signals import (
        compute_nox_v3, detect_daily_triggers, calc_adx_with_di,
        _pine_rsi, calc_rs, NOX_V3, NOX_V3_TRIGGER
    )

    LB = NOX_V3['pivot_lb']

    def _to_lower(df):
        return df.rename(columns={
            'Open': 'open', 'High': 'high', 'Low': 'low',
            'Close': 'close', 'Volume': 'volume',
        })

    def _to_weekly(df):
        return df.resample('W-FRI').agg({
            'open': 'first', 'high': 'max', 'low': 'min',
            'close': 'last', 'volume': 'sum',
        }).dropna(subset=['close'])

    xu_close = None
    if xu_df is not None and not xu_df.empty:
        xu_close = _to_lower(xu_df)['close']

    signals = []
    tickers = sorted(all_data.keys())
    total = len(tickers)
    n_triggered = 0

    print(f"  NW: {total} hisse taranıyor...")

    for idx, ticker in enumerate(tickers):
        if (idx + 1) % 100 == 0:
            print(f"    [{idx+1}/{total}] {ticker} — {n_triggered} sinyal")

        daily_raw = all_data[ticker]
        if daily_raw is None or len(daily_raw) < 120:
            continue

        daily = _to_lower(daily_raw)
        weekly = _to_weekly(daily)
        if len(weekly) < 60:
            continue

        try:
            w_result = compute_nox_v3(weekly, require_gate=False, min_sell_severity=0)
        except Exception:
            continue

        # Günlük pivot (D+W tespiti)
        try:
            d_result = compute_nox_v3(daily, require_gate=False, min_sell_severity=0)
            daily_pivot_buy = d_result['pivot_buy']
        except Exception:
            daily_pivot_buy = None

        for i in range(2 * LB, len(weekly)):
            if not w_result['pivot_buy'].iloc[i]:
                continue

            pivot_price = w_result['pivot_low_price'].iloc[i]
            confirm_date = weekly.index[i]
            if confirm_date < start_date or confirm_date > end_date:
                continue

            # Pivot kırılma kontrolü
            bars_after = weekly['low'].iloc[i:min(i + 11, len(weekly))]
            if len(bars_after) > 1 and bars_after.iloc[1:].min() < pivot_price:
                continue

            w_close = weekly['close'].iloc[i]
            w_delta = (w_close - pivot_price) / pivot_price * 100
            w_gate = bool(w_result['gate_open'].iloc[i])

            # Daily trigger ara
            trigger = detect_daily_triggers(
                daily, pivot_price, confirm_date.strftime('%Y-%m-%d'),
                max_delta_pct=NOX_V3_TRIGGER['max_delta_pct']
            )
            if not trigger['triggered']:
                continue
            tt = trigger['trigger_type']
            if tt not in ('HC2', 'BOS'):
                continue

            trigger_date_str = trigger['trigger_date']
            trigger_close = trigger['trigger_close']

            # D+W tespiti
            is_dw = False
            if daily_pivot_buy is not None:
                try:
                    t_ts = pd.Timestamp(trigger_date_str)
                    t_idx = daily.index.searchsorted(t_ts)
                    for offset in range(-2, 3):
                        ci = t_idx + offset
                        if 0 <= ci < len(daily_pivot_buy) and daily_pivot_buy.iloc[ci]:
                            is_dw = True
                            break
                except Exception:
                    pass

            delta_at = trigger.get('delta_pct_at_trigger')
            if delta_at is None:
                delta_at = (trigger_close - pivot_price) / pivot_price * 100

            n_triggered += 1
            signals.append({
                'screener': 'nox_v3_daily',
                'ticker': ticker,
                'signal_date': trigger_date_str,
                'direction': 'AL',
                'fresh': 'BUGUN',
                'gate': w_gate,
                'delta_pct': round(delta_at, 1),
                'dw_overlap': is_dw,
                'entry_price': trigger_close,
            })

    print(f"    NW: {n_triggered} tetikli sinyal (HC2/BOS)")
    return signals


# ── RT Signal Generator ──

def _generate_rt_signals(all_data, xu_df, start_date, end_date):
    """run_backtest_rt_volume.py pattern — regime transition."""
    from markets.bist.regime_transition import (
        scan_regime_transition, calc_oe_score, RT_CFG, REGIME_NAMES
    )

    def _to_lower(df):
        return df.rename(columns={
            'Open': 'open', 'High': 'high', 'Low': 'low',
            'Close': 'close', 'Volume': 'volume',
        })

    def _to_weekly(df):
        return df.resample('W-FRI').agg({
            'open': 'first', 'high': 'max', 'low': 'min',
            'close': 'last', 'volume': 'sum',
        }).dropna(subset=['close'])

    signals = []
    tickers = sorted(all_data.keys())
    total = len(tickers)
    n_signals = 0

    print(f"  RT: {total} hisse taranıyor...")

    for idx, ticker in enumerate(tickers):
        if (idx + 1) % 100 == 0:
            print(f"    [{idx+1}/{total}] {ticker} — {n_signals} sinyal")

        daily_raw = all_data[ticker]
        if daily_raw is None or len(daily_raw) < 120:
            continue

        daily = _to_lower(daily_raw)
        weekly = _to_weekly(daily)

        try:
            rt = scan_regime_transition(daily, weekly, RT_CFG)
        except Exception:
            continue

        regime = rt['regime']
        close = rt['close']
        ema21 = rt['ema21']
        cmf = rt['cmf']
        rvol = rt['rvol']

        last_signal_i = -999
        for i in range(1, len(daily)):
            curr_r = int(regime.iloc[i]) if pd.notna(regime.iloc[i]) else None
            prev_r = int(regime.iloc[i-1]) if pd.notna(regime.iloc[i-1]) else None
            if curr_r is None or prev_r is None or curr_r <= prev_r:
                continue
            if i - last_signal_i < 5:
                continue

            sig_date = daily.index[i]
            if sig_date < start_date or sig_date > end_date:
                continue

            last_signal_i = i
            _close = float(close.iloc[i])
            _cmf = float(cmf.iloc[i]) if pd.notna(cmf.iloc[i]) else 0
            _rvol = float(rvol.iloc[i]) if pd.notna(rvol.iloc[i]) else 0

            # Badge: CMF + RVOL koşulları
            badge = ''
            if _cmf > 0.1 and 1.2 <= _rvol <= 5:
                badge = 'H+PB'
            elif _cmf > 0:
                badge = 'H+AL'

            # Entry score (kalite): ATR<3%, ADX_slope<0, regime<=2, RVOL<2
            entry_score = 4
            atr_series = rt.get('atr')
            adx_slope_series = rt.get('adx_slope')
            if atr_series is not None and pd.notna(atr_series.iloc[i]):
                atr_pct = float(atr_series.iloc[i]) / _close * 100 if _close > 0 else 99
                if atr_pct >= 3:
                    entry_score -= 1
            if adx_slope_series is not None and pd.notna(adx_slope_series.iloc[i]):
                if float(adx_slope_series.iloc[i]) < 0:
                    entry_score -= 1
            if curr_r <= 2:
                entry_score -= 0  # regime 2+ is ok
            if _rvol < 0.8:
                entry_score -= 1

            # OE score
            try:
                oe = calc_oe_score(daily.iloc[:i+1], ema21.iloc[:i+1], RT_CFG)
                _oe_score = oe['oe_score']
            except Exception:
                _oe_score = 0

            n_signals += 1
            signals.append({
                'screener': 'regime_transition',
                'ticker': ticker,
                'signal_date': sig_date.strftime('%Y-%m-%d'),
                'direction': 'AL',
                'entry_window': 'TAZE',
                'badge': badge,
                'quality': entry_score,
                'oe': _oe_score,
                'transition_date': sig_date.strftime('%Y-%m-%d'),
                'cmf': round(_cmf, 3),
                'entry_price': _close,
            })

    print(f"    RT: {n_signals} sinyal üretildi")
    return signals


# ── Tavan Signal Generator ──

def _generate_tavan_signals(all_data, start_date, end_date):
    """bist_tavan_scanner_v2.py pattern — OHLCV heuristic tavan tespiti."""
    signals = []
    tickers = sorted(all_data.keys())
    n_signals = 0

    print(f"  Tavan: {len(tickers)} hisse taranıyor...")

    TAVAN_ESIK = 0.095

    for ticker in tickers:
        df = all_data[ticker]
        if df is None or len(df) < 20:
            continue

        c = df['Close']
        h = df['High']
        lo = df['Low']
        v = df['Volume']
        prev_c = c.shift(1)
        daily_ret = (c - prev_c) / prev_c

        # Tavan tespiti
        is_tavan = daily_ret >= TAVAN_ESIK
        daily_range = (h - lo).replace(0, np.nan)
        close_pos = (c - lo) / daily_range

        # Volume ratio
        vol_sma20 = v.rolling(20).mean()
        vol_ratio = v / vol_sma20.replace(0, np.nan)

        # Streak hesapla
        streak_arr = []
        count = 0
        for val in is_tavan:
            count = count + 1 if val else 0
            streak_arr.append(count)
        streak_s = pd.Series(streak_arr, index=df.index)

        # Skor (skor≥50 → kilitli tanım)
        skor = pd.Series(np.where(
            close_pos >= 0.99, 80,
            np.where(close_pos >= 0.95, 60,
                     np.where(close_pos >= 0.90, 50, 30))
        ), index=df.index)

        # Streak bonusu
        skor = skor + np.where(streak_s >= 4, 15,
                      np.where(streak_s >= 3, 10,
                      np.where(streak_s >= 2, 5, 0)))

        # Volume bonusu (düşük hacim = kilitli = iyi)
        skor = skor + np.where(vol_ratio < 1.0, 5, 0)

        # Tavan günlerini filtrele
        tavan_mask = is_tavan & (df.index >= start_date) & (df.index <= end_date)
        for i in df.index[tavan_mask]:
            s = int(skor.loc[i])
            vr = float(vol_ratio.loc[i]) if pd.notna(vol_ratio.loc[i]) else 1.0
            st = int(streak_s.loc[i])

            screener_name = 'tavan' if s >= 50 else 'tavan_kandidat'
            n_signals += 1
            signals.append({
                'screener': screener_name,
                'ticker': ticker,
                'signal_date': i.strftime('%Y-%m-%d'),
                'direction': 'AL',
                'signal_type': 'TAVAN',
                'skor': s,
                'streak': st,
                'volume_ratio': round(vr, 2),
                'entry_price': float(c.loc[i]),
            })

    print(f"    Tavan: {n_signals} sinyal üretildi")
    return signals


# ── Phase 2 Orchestrator ──

def run_phase2(period='5y', years=5.0):
    """4 screener'ı tarihsel OHLCV üzerinde çalıştırarak shortlist backtest yap.

    Returns: (results_list, all_data, xu_df) — results forward return dahil
    """
    print("\n" + "=" * 60)
    print(f"  FAZ 2: Screener Tarihsel Backtest ({period})")
    print("=" * 60)

    t0 = time.time()
    end_date = pd.Timestamp.now()
    start_date = end_date - timedelta(days=int(years * 365.25))
    print(f"  Dönem: {start_date.strftime('%Y-%m-%d')} → {end_date.strftime('%Y-%m-%d')}")

    # 1. Veri çek
    print("\n[P2-1] Veri çekiliyor...")
    tickers = data_mod.get_all_bist_tickers()
    all_data = data_mod.fetch_data(tickers, period=period)
    xu_df = data_mod.fetch_benchmark(period=period)
    print(f"  {len(all_data)} hisse yüklendi ({time.time() - t0:.0f}s)")

    # 2. Sinyal üretimi (4 screener)
    print("\n[P2-2] Sinyal üretimi...")
    t1 = time.time()

    alsat_signals = _generate_alsat_signals(all_data, xu_df, start_date, end_date)
    nw_signals = _generate_nw_signals(all_data, xu_df, start_date, end_date)
    rt_signals = _generate_rt_signals(all_data, xu_df, start_date, end_date)
    tavan_signals = _generate_tavan_signals(all_data, start_date, end_date)

    total_sigs = len(alsat_signals) + len(nw_signals) + len(rt_signals) + len(tavan_signals)
    print(f"\n  Toplam: {total_sigs} sinyal ({time.time() - t1:.0f}s)")
    print(f"    AS={len(alsat_signals)} NW={len(nw_signals)} RT={len(rt_signals)} TVN={len(tavan_signals)}")

    # 3. Günlük sinyal toplama + shortlist reconstruct
    print("\n[P2-3] Shortlist reconstruct...")
    all_sigs = alsat_signals + nw_signals + rt_signals + tavan_signals
    day_signals = defaultdict(list)
    for sig in all_sigs:
        day_signals[sig['signal_date']].append(sig)

    _LIST_SHORT = {'alsat': 'AS', 'tavan': 'TVN', 'nw': 'NW', 'rt': 'RT'}
    _META_KEYS = ('signal_type', 'badge', 'dw_overlap', 'entry_window',
                  'quality', 'skor', 'volume_ratio', 'karar', 'rs_score', 'macd',
                  'oe', 'cmf', 'delta_pct', 'gate', 'streak', 'fresh')
    all_trades = []
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')

    for date_str_iso in sorted(day_signals.keys()):
        if date_str_iso < start_str or date_str_iso > end_str:
            continue

        # csv_date format: YYYYMMDD
        csv_date = date_str_iso.replace('-', '')
        sigs = day_signals[date_str_iso]

        lists = _compute_4_lists_backtest(sigs, csv_date)

        # Tier 1 trades
        for ticker, quality, reasons, meta in lists.get('tier1', []):
            in_lists = meta.get('in_lists', [])
            entry_price = 0
            entry_sig = None
            for ln in in_lists:
                for t, sc, r, sig in lists.get(ln, []):
                    if t == ticker:
                        entry_price = sig.get('entry_price', 0)
                        entry_sig = sig
                        break
                if entry_price:
                    break

            trade = {
                'ticker': ticker,
                'signal_date': date_str_iso,
                'csv_date': csv_date,
                'entry_price': entry_price,
                'score': quality,
                'tier': 'tier1',
                'list': 'tier1',
                'overlap_count': meta.get('overlap_count', 0),
                'in_lists': in_lists,
                'ty': meta.get('ty', False),
                'relaxed': meta.get('relaxed', False),
                'list_tags': '+'.join(_LIST_SHORT.get(l, l) for l in sorted(in_lists)),
                'source': 'phase2',
            }
            if entry_sig:
                for k in _META_KEYS:
                    if k in entry_sig:
                        trade[k] = entry_sig[k]
            all_trades.append(trade)

        # Tier 2 trades
        for ticker, score, reasons, sig in lists.get('tier2', []):
            trade = {
                'ticker': ticker,
                'signal_date': date_str_iso,
                'csv_date': csv_date,
                'entry_price': sig.get('entry_price', 0),
                'score': score,
                'tier': 'tier2',
                'list': reasons[0] if reasons else '?',
                'source': 'phase2',
            }
            for k in _META_KEYS:
                if k in sig:
                    trade[k] = sig[k]
            all_trades.append(trade)

        # Liste tekil sinyaller
        tier_tickers = {t['ticker'] for t in all_trades if t.get('csv_date') == csv_date}
        for list_name in ('alsat', 'tavan', 'nw', 'rt'):
            for ticker, score, reasons, sig in lists.get(list_name, []):
                if ticker in tier_tickers:
                    continue
                trade = {
                    'ticker': ticker,
                    'signal_date': date_str_iso,
                    'csv_date': csv_date,
                    'entry_price': sig.get('entry_price', 0),
                    'score': score,
                    'tier': 'list_only',
                    'list': list_name,
                    'source': 'phase2',
                }
                for k in _META_KEYS:
                    if k in sig:
                        trade[k] = sig[k]
                all_trades.append(trade)
                tier_tickers.add(ticker)

    print(f"  {len(all_trades)} trade oluşturuldu ({len(day_signals)} gün)")

    # 4. Forward getiri
    print("\n[P2-4] Forward getiriler hesaplanıyor...")
    results = compute_forward_returns(all_trades, all_data, xu_df)
    n_tamam = sum(1 for r in results if r.get('status') == 'tamam')
    print(f"  {len(results)} trade: {n_tamam} tamam")

    # Taban risk enrichment
    _enrich_trades_taban_risk(results, all_data)

    # Trade-level parquet kaydet
    try:
        import pandas as _pd
        _df = _pd.DataFrame(results)
        # in_lists list tipinde → string'e çevir
        if 'in_lists' in _df.columns:
            _df['in_lists'] = _df['in_lists'].apply(
                lambda x: '+'.join(x) if isinstance(x, list) else str(x) if x else '')
        _pq_path = os.path.join('output', 'shortlist_5y_trades.parquet')
        _df.to_parquet(_pq_path, index=False)
        print(f"\n  Parquet: {_pq_path} ({len(_df)} trade)")
    except Exception as e:
        print(f"\n  Parquet kayıt hatası: {e}")

    elapsed = time.time() - t0
    print(f"\n  Phase 2 tamamlandı: {elapsed:.0f}s ({elapsed/60:.1f} dk)")

    return results, all_data, xu_df


# =============================================================================
# 8. ÖZET YAZDIRMA (Konsol)
# =============================================================================

def print_console_summary(results, analysis):
    """Konsola özet istatistikleri yazdır."""
    gen = analysis.get('overall', {})
    if not gen:
        return

    print(f"\n{'═' * 70}")
    print(f"  SHORTLIST BACKTEST ÖZETİ")
    print(f"{'═' * 70}")
    print(f"  {gen['n']} trade")

    for w in WINDOWS:
        avg = gen.get(f'avg_{w}d')
        wr = gen.get(f'wr_{w}d')
        exc = gen.get(f'excess_avg_{w}d')
        if avg is not None:
            exc_str = f"  XU100+: {exc:+.2f}%" if exc is not None else ""
            print(f"  {w}G: ort {avg:+.2f}%, WR {wr:.1f}%{exc_str}")

    # Liste bazlı
    print(f"\n  --- LİSTE BAZLI (5G) ---")
    _LABELS = {'alsat': 'AL/SAT', 'tavan': 'Tavan', 'nw': 'NW Pivot', 'rt': 'Rejim'}
    for ln, label in _LABELS.items():
        st = analysis.get('lists', {}).get(ln)
        if st:
            wr = st.get('wr_5d')
            avg = st.get('avg_5d')
            exc = st.get('excess_avg_5d')
            wr_s = f"WR={wr:.1f}%" if wr is not None else "WR=—"
            avg_s = f"ort={avg:+.2f}%" if avg is not None else "ort=—"
            exc_s = f"XU100+={exc:+.2f}%" if exc is not None else ""
            print(f"  {label:10s}  N={st['n']:4d}  {avg_s}  {wr_s}  {exc_s}")

    # Tier bazlı
    print(f"\n  --- TIER BAZLI (5G) ---")
    _TIER_LABELS = {'tier1': 'Tier 1', 'tier2': 'Tier 2', 'list_only': 'Liste Tekil'}
    for tier, label in _TIER_LABELS.items():
        st = analysis.get('tiers', {}).get(tier)
        if st:
            wr = st.get('wr_5d')
            avg = st.get('avg_5d')
            wr_s = f"WR={wr:.1f}%" if wr is not None else "WR=—"
            avg_s = f"ort={avg:+.2f}%" if avg is not None else "ort=—"
            print(f"  {label:12s}  N={st['n']:4d}  {avg_s}  {wr_s}")

    # Overlap
    print(f"\n  --- OVERLAP (5G) ---")
    for combo, st in analysis.get('overlaps', {}).items():
        wr = st.get('wr_5d')
        avg = st.get('avg_5d')
        if wr is not None:
            print(f"  {combo:15s}  N={st['n']:4d}  ort={avg:+.2f}%  WR={wr:.1f}%")

    # T+Y
    ty = analysis.get('ty', {})
    if ty.get('ty'):
        st = ty['ty']
        wr = st.get('wr_5d')
        avg = st.get('avg_5d')
        if wr is not None:
            print(f"\n  T+Y Çakışma:  N={st['n']:4d}  ort={avg:+.2f}%  WR={wr:.1f}%")

    # Ek özellik kırılımları
    features = analysis.get('features', {})
    if features:
        base_1g = gen.get('wr_1d', 50)
        base_3g = gen.get('wr_3d', 50)
        print(f"\n  --- ÖZELLİK KIRILIMI (1G/3G) ---")
        print(f"  {'Özellik':22s} {'N':>6s}  {'1G WR':>7s} {'Δ1G':>6s}  {'3G WR':>7s} {'Δ3G':>6s}")
        rows = []
        for feat, st in features.items():
            wr1 = st.get('wr_1d')
            wr3 = st.get('wr_3d')
            if wr1 is not None and st['n'] >= 20:
                d1 = wr1 - base_1g
                d3 = (wr3 - base_3g) if wr3 is not None else 0
                rows.append((feat, st['n'], wr1, d1, wr3, d3))
        # 1G delta'ya göre sırala
        rows.sort(key=lambda x: -x[3])
        for feat, n, wr1, d1, wr3, d3 in rows:
            m1 = '🟢' if d1 > 2 else ('🟡' if d1 > 0 else '🔴')
            wr3_s = f"{wr3:.1f}%" if wr3 is not None else "—"
            d3_s = f"{d3:+.1f}" if wr3 is not None else "—"
            print(f"  {m1} {feat:20s} {n:6d}  {wr1:6.1f}% {d1:+5.1f}  {wr3_s:>7s} {d3_s:>5s}")


# =============================================================================
# 9. HTML RAPOR (6 TAB) — Python-rendered
# =============================================================================

def _html_rc(v):
    """Return color style for value."""
    if v is None:
        return 'color:var(--text-muted)'
    return 'color:var(--nox-green)' if v > 0 else ('color:var(--nox-red)' if v < 0 else 'color:var(--text-muted)')


def _html_fv(v):
    return f'{v:.2f}' if v is not None else '—'


def _html_fw(v):
    return f'{v:.1f}%' if v is not None else '—'


def _html_ret_cell(v):
    if v is None:
        return '<td style="color:var(--text-muted)">—</td>'
    c = 'var(--nox-green)' if v > 0 else ('var(--nox-red)' if v < 0 else 'var(--text-muted)')
    sign = '+' if v > 0 else ''
    return f'<td style="color:{c};font-weight:600">{sign}{v:.2f}%</td>'


def _html_stats_header():
    return ('<tr><th>Kategori</th><th>N</th>'
            '<th>1G Ort%</th><th>1G WR%</th><th>3G Ort%</th><th>3G WR%</th>'
            '<th>5G Ort%</th><th>5G WR%</th><th>5G Med%</th>'
            '<th>En Iyi</th><th>En Kotu</th><th>XU100+</th></tr>')


def _html_stats_row(label, st, bold=False):
    if not st:
        return ''
    style = ' style="font-weight:700"' if bold else ''
    return (f'<tr{style}><td>{label}</td><td>{st.get("n", 0)}</td>'
            f'<td style="{_html_rc(st.get("avg_1d"))}">{_html_fv(st.get("avg_1d"))}</td>'
            f'<td>{_html_fw(st.get("wr_1d"))}</td>'
            f'<td style="{_html_rc(st.get("avg_3d"))}">{_html_fv(st.get("avg_3d"))}</td>'
            f'<td>{_html_fw(st.get("wr_3d"))}</td>'
            f'<td style="{_html_rc(st.get("avg_5d"))}">{_html_fv(st.get("avg_5d"))}</td>'
            f'<td>{_html_fw(st.get("wr_5d"))}</td>'
            f'<td style="{_html_rc(st.get("med_5d"))}">{_html_fv(st.get("med_5d"))}</td>'
            f'<td style="color:var(--nox-green)">{_html_fv(st.get("best_5d"))}</td>'
            f'<td style="color:var(--nox-red)">{_html_fv(st.get("worst_5d"))}</td>'
            f'<td style="{_html_rc(st.get("excess_avg_5d"))}">{_html_fv(st.get("excess_avg_5d"))}</td></tr>')


def _html_card(label, avg, wr, n_val=None):
    if n_val is not None:
        return (f'<div class="nox-card"><div class="card-label">{label}</div>'
                f'<div class="card-value" style="color:var(--nox-cyan)">{n_val}</div>'
                f'<div class="card-sub">&nbsp;</div></div>')
    c = _html_rc(avg)
    v = f'{avg:+.2f}%' if avg is not None else '—'
    wr_s = f'WR: {wr:.1f}%' if wr is not None else ''
    return (f'<div class="nox-card"><div class="card-label">{label}</div>'
            f'<div class="card-value" style="{c}">{v}</div>'
            f'<div class="card-sub">{wr_s}</div></div>')


def _html_trade_table(trades_list, title):
    if not trades_list:
        return ''
    h = f'<h3 style="color:var(--nox-cyan);margin:16px 0 8px;font-size:0.85rem">{title}</h3>'
    h += '<table class="stats-table"><thead><tr><th>Hisse</th><th>Tarih</th><th>Tier</th><th>Liste</th><th>Skor</th><th>5G%</th><th>XU100+ 5G</th></tr></thead><tbody>'
    for r in trades_list:
        h += (f'<tr><td><a href="https://www.tradingview.com/chart/?symbol=BIST:{r["ticker"]}" '
              f'target="_blank" style="color:var(--nox-cyan)">{r["ticker"]}</a></td>'
              f'<td style="font-family:var(--font-mono)">{r.get("signal_date", "")}</td>'
              f'<td>{r.get("tier", "")}</td><td>{r.get("list_tags", r.get("list", ""))}</td>'
              f'<td>{r.get("score", "")}</td>'
              f'{_html_ret_cell(r.get("ret_5d"))}{_html_ret_cell(r.get("excess_5d"))}</tr>')
    h += '</tbody></table>'
    return h


def _html_hm_cell(v):
    if v is None:
        return '<span style="padding:4px 8px;border-radius:4px;font-size:0.68rem;font-weight:700;font-family:var(--font-mono);background:rgba(113,113,122,0.15);color:var(--text-muted)">—</span>'
    if v >= 70:
        bg, fg = 'rgba(74,222,128,0.25)', 'var(--nox-green)'
    elif v >= 60:
        bg, fg = 'rgba(74,222,128,0.12)', 'var(--nox-green)'
    elif v >= 50:
        bg, fg = 'rgba(250,204,21,0.12)', 'var(--nox-yellow)'
    else:
        bg, fg = 'rgba(248,113,113,0.12)', 'var(--nox-red)'
    return f'<span style="padding:4px 8px;border-radius:4px;font-size:0.68rem;font-weight:700;font-family:var(--font-mono);background:{bg};color:{fg}">{v:.1f}%</span>'


def generate_html_report(results, analysis, phase='phase1'):
    """NOX dark-theme 6-tab HTML rapor — tamamen Python-rendered."""
    now = datetime.now(_TZ_TR).strftime('%d.%m.%Y %H:%M')
    gen = analysis.get('overall', {})
    n_trades = gen.get('n', 0)
    dates = sorted(set(r.get('csv_date', '') for r in results if r.get('csv_date')))
    date_range = f"{dates[0]} - {dates[-1]}" if len(dates) >= 2 else (dates[0] if dates else '?')
    n_days = len(dates)

    # ── TAB 1: Overview ──
    overview = '<div class="nox-cards">'
    overview += _html_card('Trade', None, None, n_val=n_trades)
    for label, key in [('1 Gun', '1d'), ('3 Gun', '3d'), ('5 Gun', '5d')]:
        overview += _html_card(label, gen.get(f'avg_{key}'), gen.get(f'wr_{key}'))
    overview += '</div>'

    overview += '<h3 style="color:var(--nox-cyan);margin:16px 0 8px;font-size:0.85rem">Liste Karsilastirma</h3>'
    overview += f'<table class="stats-table"><thead>{_html_stats_header()}</thead><tbody>'
    overview += _html_stats_row('Genel', gen, bold=True)
    for key, label in [('alsat', 'AL/SAT'), ('tavan', 'Tavan'), ('nw', 'NW Pivot'), ('rt', 'Rejim')]:
        overview += _html_stats_row(label, analysis.get('lists', {}).get(key))
    overview += '</tbody></table>'

    overview += '<h3 style="color:var(--nox-cyan);margin:16px 0 8px;font-size:0.85rem">Tier Karsilastirma</h3>'
    overview += f'<table class="stats-table"><thead>{_html_stats_header()}</thead><tbody>'
    for key, label in [('tier1', 'Tier 1 (Cakisma)'), ('tier2', 'Tier 2 (Tekil)'), ('list_only', 'Liste Tekil')]:
        overview += _html_stats_row(label, analysis.get('tiers', {}).get(key))
    overview += '</tbody></table>'

    overview += _html_trade_table(analysis.get('top20', []), 'En Iyi 20 Trade (5G)')
    overview += _html_trade_table(analysis.get('bottom20', []), 'En Kotu 20 Trade (5G)')

    # ── TAB 2: Lists ──
    lists_html = ''
    sig_types = analysis.get('signal_types', {})

    sections = [
        ('AL/SAT Sinyal Tipi', ['AS_CMB', 'AS_CMB+', 'AS_ZAYIF', 'AS_GUCLU', 'AS_BILESEN', 'AS_DONUS']),
        ('NW Detay', ['NW_DW', 'NW_daily']),
        ('RT Badge', ['RT_H+PB', 'RT_H+AL', 'RT_no_badge']),
        ('Tavan Detay', ['TVN_kilitli', 'TVN_normal']),
    ]
    for title, keys in sections:
        has_data = any(sig_types.get(k) for k in keys)
        if not has_data:
            continue
        lists_html += f'<h3 style="color:var(--nox-cyan);margin:16px 0 8px;font-size:0.85rem">{title}</h3>'
        lists_html += f'<table class="stats-table"><thead>{_html_stats_header()}</thead><tbody>'
        for k in keys:
            st = sig_types.get(k)
            if st:
                short = k.split('_', 1)[1] if '_' in k else k
                lists_html += _html_stats_row(short, st)
        lists_html += '</tbody></table>'

    score_bands = analysis.get('score_bands', {})
    if score_bands:
        lists_html += '<h3 style="color:var(--nox-cyan);margin:16px 0 8px;font-size:0.85rem">Score Band</h3>'
        lists_html += f'<table class="stats-table"><thead>{_html_stats_header()}</thead><tbody>'
        for band, st in score_bands.items():
            lists_html += _html_stats_row(band, st)
        lists_html += '</tbody></table>'

    # Ek özellik kırılımları
    features = analysis.get('features', {})
    if features:
        feat_sections = [
            ('Tavan Skor', ['tvn_skor_80+', 'tvn_skor_60_79', 'tvn_skor_50_59', 'tvn_skor_30_49', 'tvn_skor_0_29']),
            ('Tavan Volume Ratio', ['tvn_vr_3+', 'tvn_vr_2_3', 'tvn_vr_1.5_2', 'tvn_vr_1_1.5', 'tvn_vr_<1']),
            ('Tavan Streak', ['tvn_streak_3+', 'tvn_streak_2', 'tvn_streak_1']),
            ('Quality', ['q_75_100', 'q_50_74', 'q_25_49', 'q_0_24']),
            ('RT Entry Window', ['rt_ew_TAZE', 'rt_ew_YAKIN', 'rt_ew_none']),
            ('RT OE', ['rt_oe_3_4', 'rt_oe_1_2', 'rt_oe_0']),
            ('RT CMF', ['rt_cmf_0.1+', 'rt_cmf_0_0.1', 'rt_cmf_neg']),
            ('NW Gate', ['nw_gate_open', 'nw_gate_closed']),
            ('NW Delta%', ['nw_delta_0_3', 'nw_delta_3_6', 'nw_delta_6_10', 'nw_delta_10+']),
            ('AL/SAT RS Score', ['as_rs_1.5+', 'as_rs_1_1.5', 'as_rs_0.5_1', 'as_rs_<0.5']),
            ('Taban Riski', ['taban_risk_high', 'taban_risk_med', 'taban_risk_low']),
        ]
        for title, keys in feat_sections:
            has_data = any(features.get(k) for k in keys)
            if not has_data:
                continue
            lists_html += f'<h3 style="color:var(--nox-cyan);margin:16px 0 8px;font-size:0.85rem">{title}</h3>'
            lists_html += f'<table class="stats-table"><thead>{_html_stats_header()}</thead><tbody>'
            for k in keys:
                st = features.get(k)
                if st:
                    lists_html += _html_stats_row(k, st)
            lists_html += '</tbody></table>'

    # ── TAB 3: Tiers ──
    tiers_html = '<h3 style="color:var(--nox-cyan);margin:16px 0 8px;font-size:0.85rem">Tier Performansi</h3>'
    tiers_html += f'<table class="stats-table"><thead>{_html_stats_header()}</thead><tbody>'
    tiers_html += _html_stats_row('Genel', gen, bold=True)
    for key, label in [('tier1', 'Tier 1'), ('tier2', 'Tier 2'), ('list_only', 'Tekil')]:
        tiers_html += _html_stats_row(label, analysis.get('tiers', {}).get(key))
    tiers_html += '</tbody></table>'

    ty = analysis.get('ty', {})
    if ty.get('ty') or ty.get('no_ty'):
        tiers_html += '<h3 style="color:var(--nox-cyan);margin:16px 0 8px;font-size:0.85rem">Teknik+Yapisal (T+Y)</h3>'
        tiers_html += f'<table class="stats-table"><thead>{_html_stats_header()}</thead><tbody>'
        if ty.get('ty'):
            tiers_html += _html_stats_row('T+Y Cakisma', ty['ty'])
        if ty.get('no_ty'):
            tiers_html += _html_stats_row('Diger Cakisma', ty['no_ty'])
        tiers_html += '</tbody></table>'

    # ── TAB 4: Overlap ──
    overlaps = analysis.get('overlaps', {})
    overlap_html = ''
    if overlaps:
        overlap_html += '<h3 style="color:var(--nox-cyan);margin:16px 0 8px;font-size:0.85rem">Cakisma Derinligi</h3>'
        overlap_html += f'<table class="stats-table"><thead>{_html_stats_header()}</thead><tbody>'
        if overlaps.get('2_list'):
            overlap_html += _html_stats_row('2 Liste', overlaps['2_list'])
        if overlaps.get('3+_list'):
            overlap_html += _html_stats_row('3+ Liste', overlaps['3+_list'])
        overlap_html += '</tbody></table>'

        # Heatmap
        pairs = ['alsat+nw', 'alsat+rt', 'alsat+tavan', 'nw+rt', 'nw+tavan', 'rt+tavan']
        overlap_html += '<h3 style="color:var(--nox-cyan);margin:16px 0 8px;font-size:0.85rem">Cift Kombinasyon WR (5G)</h3>'
        overlap_html += '<div style="display:flex;gap:12px;flex-wrap:wrap;margin-bottom:16px">'
        for p in pairs:
            st = overlaps.get(p)
            if st:
                overlap_html += (f'<div style="text-align:center"><div style="font-size:0.68rem;color:var(--text-muted);margin-bottom:4px">{p}</div>'
                                 f'{_html_hm_cell(st.get("wr_5d"))}'
                                 f'<div style="font-size:0.62rem;color:var(--text-muted);margin-top:2px">N={st["n"]} avg={_html_fv(st.get("avg_5d"))}%</div></div>')
        overlap_html += '</div>'

        overlap_html += f'<table class="stats-table"><thead>{_html_stats_header()}</thead><tbody>'
        for combo, st in overlaps.items():
            overlap_html += _html_stats_row(combo, st)
        overlap_html += '</tbody></table>'
    else:
        overlap_html = '<p style="color:var(--text-muted)">Cakisma verisi yok.</p>'

    # ── TAB 5: Daily ──
    daily = analysis.get('daily', {})
    daily_html = '<table class="stats-table"><thead><tr>'
    daily_html += '<th>Tarih</th><th>N</th><th>1G WR%</th><th>1G Ort%</th><th>3G WR%</th><th>3G Ort%</th><th>5G WR%</th><th>5G Ort%</th><th>XU100+ 5G</th>'
    daily_html += '</tr></thead><tbody>'
    for d in sorted(daily.keys()):
        st = daily[d]
        sd = f'{d[:4]}-{d[4:6]}-{d[6:8]}'
        daily_html += (f'<tr><td style="font-family:var(--font-mono)">{sd}</td><td>{st["n"]}</td>'
                       f'<td>{_html_fw(st.get("wr_1d"))}</td><td style="{_html_rc(st.get("avg_1d"))}">{_html_fv(st.get("avg_1d"))}</td>'
                       f'<td>{_html_fw(st.get("wr_3d"))}</td><td style="{_html_rc(st.get("avg_3d"))}">{_html_fv(st.get("avg_3d"))}</td>'
                       f'<td>{_html_fw(st.get("wr_5d"))}</td><td style="{_html_rc(st.get("avg_5d"))}">{_html_fv(st.get("avg_5d"))}</td>'
                       f'<td style="{_html_rc(st.get("excess_avg_5d"))}">{_html_fv(st.get("excess_avg_5d"))}</td></tr>')
    daily_html += '</tbody></table>'

    # ── TAB 6: Trades (JS-rendered for filtering) ──
    results_json = json.dumps(_sanitize(results), ensure_ascii=False)

    trades_filters = """<div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:12px;align-items:center">
    <input id="fTicker" placeholder="Hisse" oninput="af()" style="width:80px;padding:6px;border-radius:6px;border:1px solid var(--border-subtle);background:var(--bg-card);color:var(--text-primary);font-size:0.75rem">
    <select id="fList" onchange="af()" style="padding:6px;border-radius:6px;border:1px solid var(--border-subtle);background:var(--bg-card);color:var(--text-primary);font-size:0.75rem">
      <option value="">Tum Listeler</option>
      <option value="tier1">Tier 1</option><option value="tier2">Tier 2</option>
      <option value="alsat">AL/SAT</option><option value="tavan">Tavan</option>
      <option value="nw">NW</option><option value="rt">RT</option>
    </select>
    <select id="fTier" onchange="af()" style="padding:6px;border-radius:6px;border:1px solid var(--border-subtle);background:var(--bg-card);color:var(--text-primary);font-size:0.75rem">
      <option value="">Tum Tier</option>
      <option value="tier1">Tier 1</option><option value="tier2">Tier 2</option>
      <option value="list_only">Tekil</option>
    </select>
    <span id="st" style="font-size:0.72rem;color:var(--text-muted)"></span>
  </div>"""

    trades_table = """<table class="stats-table"><thead><tr>
    <th onclick="sb('ticker')">Hisse</th>
    <th onclick="sb('signal_date')">Tarih</th>
    <th onclick="sb('tier')">Tier</th>
    <th onclick="sb('list')">Liste</th>
    <th onclick="sb('score')">Skor</th>
    <th onclick="sb('entry_price')">Giris</th>
    <th onclick="sb('ret_1d')">1G%</th>
    <th onclick="sb('ret_3d')">3G%</th>
    <th onclick="sb('ret_5d')">5G%</th>
    <th onclick="sb('excess_5d')">XU100+ 5G</th>
    <th>Detay</th>
    </tr></thead><tbody id="tb"></tbody></table>"""

    # ── TAB content map ──
    tabs = [
        ('overview', 'Genel Bakis', overview),
        ('lists', 'Listeler', lists_html),
        ('tiers', 'Tier Analizi', tiers_html),
        ('overlap', 'Cakisma', overlap_html),
        ('daily', 'Gunluk', daily_html),
        ('trades', "Trade'ler", trades_filters + trades_table),
    ]

    # Build tab bar
    tab_bar = '<div class="nox-tabs">'
    for i, (tid, tlabel, _) in enumerate(tabs):
        active = ' active' if i == 0 else ''
        tab_bar += f'<div class="nox-tab{active}" onclick="st(\'{tid}\')" id="t-{tid}">{tlabel}</div>'
    tab_bar += '</div>'

    # Build tab contents
    tab_contents = ''
    for i, (tid, _, content) in enumerate(tabs):
        active = ' active' if i == 0 else ''
        tab_contents += f'<div class="tab-content{active}" id="c-{tid}">{content}</div>'

    html = f"""<!DOCTYPE html>
<html lang="tr"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NOX Shortlist Backtest - {date_range}</title>
<style>{_NOX_CSS}
.nox-tabs {{ display: flex; gap: 4px; margin-bottom: 16px; flex-wrap: wrap; }}
.nox-tab {{
  padding: 10px 20px; border-radius: var(--radius) var(--radius) 0 0;
  background: var(--bg-card); border: 1px solid var(--border-subtle); border-bottom: none;
  color: var(--text-muted); font-weight: 600; font-size: 0.82rem;
  cursor: pointer; transition: all 0.2s; font-family: var(--font-display);
}}
.nox-tab:hover {{ color: var(--text-secondary); background: var(--bg-elevated); }}
.nox-tab.active {{
  color: var(--nox-cyan); background: var(--bg-elevated);
  border-color: var(--nox-cyan); border-bottom: 2px solid var(--bg-elevated);
}}
.nox-cards {{ display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 20px; }}
.nox-card {{
  flex: 1; min-width: 130px; padding: 16px;
  background: var(--bg-card); border: 1px solid var(--border-subtle);
  border-radius: var(--radius); text-align: center;
}}
.nox-card .card-label {{ font-size: 0.7rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 6px; font-family: var(--font-display); }}
.nox-card .card-value {{ font-size: 1.4rem; font-weight: 700; font-family: var(--font-mono); }}
.nox-card .card-sub {{ font-size: 0.68rem; color: var(--text-muted); margin-top: 4px; font-family: var(--font-mono); }}
.stats-table {{ width: 100%; border-collapse: collapse; font-size: 0.75rem; margin-bottom: 20px; }}
.stats-table th {{ background: var(--bg-elevated); color: var(--text-muted); font-weight: 600; font-size: 0.65rem; text-transform: uppercase; letter-spacing: 0.06em; padding: 8px; text-align: center; border-bottom: 1px solid var(--border-subtle); cursor:pointer; }}
.stats-table td {{ padding: 8px; text-align: center; border-bottom: 1px solid rgba(39,39,42,0.5); font-family: var(--font-mono); font-size: 0.72rem; }}
.stats-table tr:hover {{ background: var(--bg-hover); }}
.stats-table td:first-child {{ text-align: left; font-family: var(--font-display); font-weight: 600; }}
.tab-content {{ display: none; }}
.tab-content.active {{ display: block; }}
</style>
</head><body>
<div class="nox-container">
<div class="nox-header">
  <div class="nox-logo">NOX<span class="proj">project</span><span class="mode">shortlist backtest</span></div>
  <div class="nox-meta">Tarih: <b>{date_range}</b><br>{now}<br><b>{n_trades}</b> trade / {n_days} gun</div>
</div>
{tab_bar}
{tab_contents}
</div>
<script>
var D={results_json};
var col='ret_5d',asc=false;
function st(id){{
  document.querySelectorAll('.nox-tab').forEach(function(x){{x.classList.remove('active')}});
  document.querySelectorAll('.tab-content').forEach(function(x){{x.classList.remove('active')}});
  document.getElementById('t-'+id).classList.add('active');
  document.getElementById('c-'+id).classList.add('active');
  if(id==='trades')af();
}}
function rc(v){{if(v==null)return'color:var(--text-muted)';return v>0?'color:var(--nox-green)':v<0?'color:var(--nox-red)':'color:var(--text-muted)';}}
function retC(v){{
  if(v==null)return'<td style="color:var(--text-muted)">—</td>';
  var c=v>0?'var(--nox-green)':v<0?'var(--nox-red)':'var(--text-muted)';
  return'<td style="color:'+c+';font-weight:600">'+(v>0?'+':'')+v.toFixed(2)+'%</td>';
}}
function af(){{
  var tk=(document.getElementById('fTicker').value||'').toUpperCase();
  var fl=document.getElementById('fList').value;
  var ft=document.getElementById('fTier').value;
  var f=D.filter(function(r){{
    if(tk&&r.ticker.indexOf(tk)<0)return false;
    if(fl){{if(fl==='tier1'||fl==='tier2'){{if(r.tier!==fl)return false;}}else{{if(r.list!==fl)return false;}}}}
    if(ft&&r.tier!==ft)return false;
    return true;
  }});
  f.sort(function(a,b){{
    var va=a[col],vb=b[col];
    if(va==null&&vb==null)return 0;
    if(va==null)return 1;if(vb==null)return-1;
    if(typeof va==='string')return asc?va.localeCompare(vb):vb.localeCompare(va);
    return asc?(va-vb):(vb-va);
  }});
  var tb=document.getElementById('tb');
  if(!tb)return;
  var h='';
  for(var i=0;i<f.length;i++){{
    var r=f[i];
    var det=[];
    if(r.signal_type)det.push(r.signal_type);
    if(r.badge)det.push(r.badge);
    if(r.dw_overlap)det.push('D+W');
    if(r.ty)det.push('T+Y');
    if(r.relaxed)det.push('RT↓');
    h+='<tr><td><a href="https://www.tradingview.com/chart/?symbol=BIST:'+r.ticker+'" target="_blank" style="color:var(--nox-cyan)">'+r.ticker+'</a></td>';
    h+='<td style="color:var(--text-muted)">'+r.signal_date+'</td>';
    h+='<td>'+(r.tier||'')+'</td>';
    h+='<td style="font-size:.68rem">'+(r.list_tags||r.list||'')+'</td>';
    h+='<td>'+(r.score||'')+'</td>';
    h+='<td>'+(r.entry_price?r.entry_price.toFixed(2):'—')+'</td>';
    h+=retC(r.ret_1d)+retC(r.ret_3d)+retC(r.ret_5d)+retC(r.excess_5d);
    h+='<td style="font-size:.62rem;color:var(--text-muted)">'+det.join(' ')+'</td></tr>';
  }}
  tb.innerHTML=h;
  document.getElementById('st').textContent=f.length+' / '+D.length+' trade';
}}
function sb(c){{if(col===c)asc=!asc;else{{col=c;asc=c==='ticker'||c==='signal_date';}};af();}}
af();
</script></body></html>"""
    return html


# =============================================================================
# 10. CSV EXPORT
# =============================================================================

def _csv_row(label, st):
    """Tek satır CSV: Grup,N,1G_ORT,1G_MED,1G_WR,..."""
    if not st:
        return None
    def _f(v): return f'{v:.2f}' if v is not None else ''
    def _fw(v): return f'{v:.1f}' if v is not None else ''
    return [
        label, str(st.get('n', 0)),
        _f(st.get('avg_1d')), _f(st.get('med_1d')), _fw(st.get('wr_1d')),
        _f(st.get('avg_3d')), _f(st.get('med_3d')), _fw(st.get('wr_3d')),
        _f(st.get('avg_5d')), _f(st.get('med_5d')), _fw(st.get('wr_5d')),
        _f(st.get('excess_avg_5d')),
    ]


def export_csv(analysis, output_dir):
    """Tüm kırılım gruplarını tek CSV'ye yaz."""
    header = ['Grup', 'N',
              '1G_ORT%', '1G_MED%', '1G_WR%',
              '3G_ORT%', '3G_MED%', '3G_WR%',
              '5G_ORT%', '5G_MED%', '5G_WR%',
              '5G_XU100+%']

    rows = []

    # Genel
    rows.append(_csv_row('GENEL', analysis.get('overall')))

    # Liste bazlı
    rows.append([])  # boş satır
    _LABELS = {'alsat': 'AL/SAT', 'tavan': 'Tavan', 'nw': 'NW_Pivot', 'rt': 'Rejim'}
    for ln, label in _LABELS.items():
        st = analysis.get('lists', {}).get(ln)
        if st:
            rows.append(_csv_row(label, st))

    # Tier bazlı
    rows.append([])
    _TIER = {'tier1': 'Tier1', 'tier2': 'Tier2', 'list_only': 'Liste_Tekil'}
    for tier, label in _TIER.items():
        st = analysis.get('tiers', {}).get(tier)
        if st:
            rows.append(_csv_row(label, st))

    # Overlap
    rows.append([])
    for combo, st in analysis.get('overlaps', {}).items():
        rows.append(_csv_row(f'OL_{combo}', st))

    # T+Y
    ty = analysis.get('ty', {})
    if ty.get('ty'):
        rows.append(_csv_row('T+Y', ty['ty']))
    if ty.get('no_ty'):
        rows.append(_csv_row('Tier1_noTY', ty['no_ty']))

    # Sinyal tipi
    rows.append([])
    for stype, st in analysis.get('signal_types', {}).items():
        rows.append(_csv_row(stype, st))

    # Score band
    rows.append([])
    for band, st in analysis.get('score_bands', {}).items():
        rows.append(_csv_row(f'Score_{band}', st))

    # Günlük
    rows.append([])
    for d, st in sorted(analysis.get('daily', {}).items()):
        d_fmt = f'{d[:4]}-{d[4:6]}-{d[6:8]}' if len(d) == 8 else d
        rows.append(_csv_row(d_fmt, st))

    # Yaz
    os.makedirs(output_dir, exist_ok=True)
    date_str = datetime.now(_TZ_TR).strftime('%Y%m%d')
    path = os.path.join(output_dir, f'shortlist_backtest_{date_str}.csv')
    with open(path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            if r:
                w.writerow(r)
            else:
                w.writerow([])
    return path


# =============================================================================
# 11. CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='NOX Shortlist Backtest')
    parser.add_argument('--full', action='store_true',
                        help='Faz 2: screener tarihsel backtest')
    parser.add_argument('--period', default='5y',
                        help='Faz 2 veri periyodu (default: 5y)')
    parser.add_argument('--years', type=float, default=None,
                        help='Faz 2 backtest yıl sayısı (default: period ile aynı)')
    parser.add_argument('--open', action='store_true',
                        help='Raporu tarayıcıda aç')
    parser.add_argument('--output', default='output',
                        help='Çıktı dizini (default: output)')
    args = parser.parse_args()

    # Period'dan yıl sayısı çıkar
    if args.years is None:
        p = args.period.lower()
        if p.endswith('y'):
            args.years = float(p[:-1])
        elif p.endswith('mo'):
            args.years = float(p[:-2]) / 12
        else:
            args.years = 5.0

    print("\n" + "=" * 60)
    print("  NOX SHORTLIST BACKTEST")
    print("=" * 60)

    # Faz 2 standalone modu: --full flag
    if args.full:
        results_p2, _all_data, _xu_df = run_phase2(
            period=args.period, years=args.years
        )
        all_results = results_p2
        if not all_results:
            print("\n  HATA: Hiç trade bulunamadı!")
            sys.exit(1)

        # Analiz
        print("\n[5] Analiz hesaplanıyor...")
        analysis = compute_all_analysis(all_results)
        print_console_summary(all_results, analysis)

        # CSV export
        csv_path = export_csv(analysis, args.output)
        print(f"\n[6] CSV: {csv_path}")

        # HTML rapor
        print("\n[7] HTML rapor oluşturuluyor...")
        html = generate_html_report(all_results, analysis, phase='phase2')
        os.makedirs(args.output, exist_ok=True)
        date_str = datetime.now(_TZ_TR).strftime('%Y%m%d')
        period_tag = args.period.replace('.', '')
        fname = f"shortlist_backtest_{period_tag}_{date_str}.html"
        path = os.path.join(args.output, fname)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(html)
        print(f"  HTML: {path}")

        if args.open:
            subprocess.Popen(['open', path])

        print(f"\n  Phase 2 backtest tamamlandı. {len(all_results)} trade analiz edildi.\n")
        return

    # Faz 1: GitHub HTML (default)
    results_p1, daily_stats, _history, _all_data, _xu_df = run_phase1()

    all_results = results_p1
    if not all_results:
        print("\n  HATA: Hiç trade bulunamadı!")
        sys.exit(1)

    # Analiz
    print("\n[5] Analiz hesaplanıyor...")
    analysis = compute_all_analysis(all_results)

    # Konsol özeti
    print_console_summary(all_results, analysis)

    # ── Production-Eşleniği A/B Test: Tier 2 Score >= 100 ──
    # A: Layer 1 mevcut sistem (Tier 2 score filtresi YOK)
    # B: Aynı + Tier 2'de score >= 100 (production)
    # Tier 1 overlapper her iki durumda da muaf
    print("\n" + "=" * 60)
    print("  PRODUCTION A/B TEST: Tier2 Score>=100")
    print("=" * 60)

    _LIST_SHORT_AB = {'alsat': 'AS', 'tavan': 'TVN', 'nw': 'NW', 'rt': 'RT'}

    if _history:
        # A tarafı: tier2_min_score=0 ile shortlist yeniden oluştur
        print("\n  A tarafı: Tier2 filtre YOK ile yeniden hesaplanıyor...")
        trades_a = []
        for date_str, day_htmls in sorted(_history.items()):
            signals = parse_day_signals(day_htmls, date_str)
            if not signals:
                continue
            lists = _compute_4_lists_backtest(signals, date_str, tier2_min_score=0)
            sig_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
            # Sadece shortlist = tier1 + tier2 (list_only dahil DEĞİL)
            for ticker, quality, reasons, meta in lists.get('tier1', []):
                in_lists = meta.get('in_lists', [])
                entry_sig = None
                for ln in in_lists:
                    for t, sc, r, sig in lists.get(ln, []):
                        if t == ticker:
                            entry_sig = sig
                            break
                    if entry_sig:
                        break
                trades_a.append({
                    'ticker': ticker, 'signal_date': sig_date, 'csv_date': date_str,
                    'entry_price': entry_sig.get('entry_price', 0) if entry_sig else 0,
                    'score': quality, 'tier': 'tier1', 'list': 'tier1',
                    'in_lists': in_lists,
                    'list_tags': '+'.join(_LIST_SHORT_AB.get(l, l) for l in sorted(in_lists)),
                })
            for ticker, score, reasons, sig in lists.get('tier2', []):
                trades_a.append({
                    'ticker': ticker, 'signal_date': sig_date, 'csv_date': date_str,
                    'entry_price': sig.get('entry_price', 0),
                    'score': score, 'tier': 'tier2',
                    'list': reasons[0] if reasons else '?',
                })

        # B tarafı: tier2'den score < 100 olanları çıkar
        trades_b = [t for t in trades_a if not (t['tier'] == 'tier2' and t['score'] < 100)]

        # Forward return hesapla (price data cached)
        print(f"  A: {len(trades_a)} trade → B: {len(trades_b)} trade ({len(trades_a)-len(trades_b)} çıkarıldı)")
        print("  Forward return hesaplanıyor...")
        results_a = compute_forward_returns(trades_a, _all_data, _xu_df)
        results_b = compute_forward_returns(trades_b, _all_data, _xu_df)

        def _ab_row(label, data_a, data_b):
            """Tek satır A/B karşılaştırma."""
            sa = _calc_stats(data_a)
            sb = _calc_stats(data_b)
            na, nb = len(data_a), len(data_b)
            def _fmt(st, key):
                v = st.get(key)
                return f"{v:.1f}%" if v is not None else '  -  '
            def _fmtm(st, key):
                v = st.get(key)
                return f"{v:+.2f}" if v is not None else '  -  '
            print(f"  {label}")
            print(f"    {'':14s} {'N':>5s} {'1G WR':>8s} {'1G MED':>8s} {'3G WR':>8s} {'3G MED':>8s} {'5G WR':>8s} {'5G MED':>8s}")
            print(f"    {'A (filtre yok)':14s} {na:5d} {_fmt(sa,'wr_1d'):>8s} {_fmtm(sa,'med_1d'):>8s} {_fmt(sa,'wr_3d'):>8s} {_fmtm(sa,'med_3d'):>8s} {_fmt(sa,'wr_5d'):>8s} {_fmtm(sa,'med_5d'):>8s}")
            print(f"    {'B (score≥100)':14s} {nb:5d} {_fmt(sb,'wr_1d'):>8s} {_fmtm(sb,'med_1d'):>8s} {_fmt(sb,'wr_3d'):>8s} {_fmtm(sb,'med_3d'):>8s} {_fmt(sb,'wr_5d'):>8s} {_fmtm(sb,'med_5d'):>8s}")
            # Delta
            d1 = (sb.get('wr_1d') or 0) - (sa.get('wr_1d') or 0)
            d3 = (sb.get('wr_3d') or 0) - (sa.get('wr_3d') or 0)
            d5 = (sb.get('wr_5d') or 0) - (sa.get('wr_5d') or 0)
            print(f"    {'Δ (B-A)':14s} {nb-na:+5d} {d1:+7.1f}p {'':>8s} {d3:+7.1f}p {'':>8s} {d5:+7.1f}p")

        # Genel shortlist (Tier1 + Tier2)
        print()
        _ab_row("GENEL (Tier1+Tier2)", results_a, results_b)

        # Sadece Tier2
        t2_a = [r for r in results_a if r.get('tier') == 'tier2']
        t2_b = [r for r in results_b if r.get('tier') == 'tier2']
        print()
        _ab_row("TIER2 tümü", t2_a, t2_b)

        # NW-only Tier2
        nw_t2_a = [r for r in t2_a if r.get('list') == 'NW']
        nw_t2_b = [r for r in t2_b if r.get('list') == 'NW']
        print()
        _ab_row("TIER2 NW", nw_t2_a, nw_t2_b)

        # RT-only Tier2
        rt_t2_a = [r for r in t2_a if r.get('list') == 'RT']
        rt_t2_b = [r for r in t2_b if r.get('list') == 'RT']
        print()
        _ab_row("TIER2 RT", rt_t2_a, rt_t2_b)

        # Tier1 (kontrol — değişmemeli)
        t1_a = [r for r in results_a if r.get('tier') == 'tier1']
        t1_b = [r for r in results_b if r.get('tier') == 'tier1']
        print()
        _ab_row("TIER1 (kontrol)", t1_a, t1_b)

        # Elenen trade'lerin profili
        elenen = [r for r in results_a if r['tier'] == 'tier2' and r['score'] < 100]
        if elenen:
            es = _calc_stats(elenen)
            print(f"\n  ── Elenen Trade Profili (N={len(elenen)}) ──")
            wr1 = f"{es.get('wr_1d',0):.1f}%" if es.get('wr_1d') is not None else '-'
            m1 = f"{es.get('med_1d',0):+.2f}" if es.get('med_1d') is not None else '-'
            wr3 = f"{es.get('wr_3d',0):.1f}%" if es.get('wr_3d') is not None else '-'
            m3 = f"{es.get('med_3d',0):+.2f}" if es.get('med_3d') is not None else '-'
            print(f"    1G WR: {wr1}, MED: {m1}")
            print(f"    3G WR: {wr3}, MED: {m3}")
            # Liste dağılımı
            from collections import Counter
            dist = Counter(r.get('list', '?') for r in elenen)
            print(f"    Dağılım: {dict(dist)}")
    else:
        print("  (history verisi yok — A/B test atlanıyor)")

    print()

    # CSV export
    csv_path = export_csv(analysis, args.output)
    print(f"\n[6] CSV: {csv_path}")

    # HTML rapor
    print("\n[7] HTML rapor oluşturuluyor...")
    html = generate_html_report(all_results, analysis)
    os.makedirs(args.output, exist_ok=True)
    date_str = datetime.now(_TZ_TR).strftime('%Y%m%d')
    fname = f"shortlist_backtest_{date_str}.html"
    path = os.path.join(args.output, fname)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"  HTML: {path}")

    if args.open:
        subprocess.Popen(['open', path])

    print(f"\n  Backtest tamamlandı. {len(all_results)} trade analiz edildi.\n")


if __name__ == '__main__':
    main()
