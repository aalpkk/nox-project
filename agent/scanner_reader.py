"""
NOX Agent — Scanner Sonuc Okuyucu
Tum scanner CSV'lerini kesfet, parse et, normalize et.
GitHub artifact'lerini iki repo'dan indirir:
  - aalpkk/bist-tavan-screener (alsat, tavan)
  - aalpkk/nox-project (nox_v3, divergence, regime_transition)
Render/remote ortamda: latest_signals.json'dan HTTP ile yükler.
"""
import os
import re
import json
import shutil
import subprocess
import tempfile
import pandas as pd

# -- CSV Dosya Adi Pattern'lari --
# Aktif workflow ciktilari:
#   bist-tavan-screener: daily_scan (alsat, tavan)
#   nox-project: nox-weekly, nox-divergence, nox-regime
_CSV_PATTERNS_OUTPUT = [
    # nox-project workflows
    # ONEMLI: weekly_ prefix'li olan HAFTALIK (wl_status+trigger dolu, WR %66.9+)
    #         prefix'siz olan GUNLUK (wl_status yok, WR %41.6 — zayif)
    (re.compile(r'^nox_v3_signals_weekly_(\d{8})\.csv$'), 'nox_v3_weekly'),
    (re.compile(r'^nox_v3_signals_(\d{8})\.csv$'),        'nox_v3_daily'),
    # (re.compile(r'^nox_divergence_(\d{8})\.csv$'),       'divergence'),  # PASİF — geliştirme bekliyor
    (re.compile(r'^regime_transition_(\d{8})\.csv$'),      'regime_transition'),
    # bist-tavan-screener workflows
    (re.compile(r'^alsat_signals_(\d{8})\.csv$'),          'alsat'),
    # rejim_v3 kaldirildi — regime_transition (nox-project HTML) tek kaynak
    (re.compile(r'^tavan_devam_(\d{8})_\d{4}\.csv$'),      'tavan'),
    (re.compile(r'^tavan_kandidat_(\d{8})_\d{4}\.csv$'),   'tavan_kandidat'),
    (re.compile(r'^tavan_signals_(\d{8})\.csv$'),          'tavan'),
    (re.compile(r'^tavan_serisi_(\d{8})\.csv$'),           'tavan'),
    # kademe: otomatik CSV tarama YOK — kullanıcı Telegram'dan foto/xls yükler
    # (re.compile(r'^nox_kademe_(\d{8})\.csv$'),           'kademe'),
]

# Insan okunabilir screener adlari
SCREENER_NAMES = {
    'nox_v3_weekly': 'NOX v3 Haftalik Pivot',
    'nox_v3_daily': 'NOX v3 Gunluk (zayif)',
    'divergence': 'Divergence',
    'regime_transition': 'Regime Transition',
    'alsat': 'AL/SAT Screener',
    'tavan': 'Tavan Scanner',
    'tavan_kandidat': 'Tavan Kandidat',
    'kademe': 'Kademe S/A',
}


def _scan_dir(directory, patterns):
    """Bir dizini tara -> {screener: [(date_str, path), ...]}"""
    found = {}
    if not os.path.isdir(directory):
        return found
    for fname in os.listdir(directory):
        for pat, screener in patterns:
            m = pat.match(fname)
            if m:
                date_str = m.group(1)
                found.setdefault(screener, []).append(
                    (date_str, os.path.join(directory, fname)))
                break
    return found


def discover_csvs(output_dir=None, target_date=None):
    """CSV'leri kesfet -> {screener: [(date_str, path), ...]}"""
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), 'output')

    found = _scan_dir(output_dir, _CSV_PATTERNS_OUTPUT)

    # Tarih filtresi, dedup (ayni tarihte en yeni dosya), siralama
    result = {}
    for scr, items in found.items():
        # Ayni tarihte birden fazla dosya varsa (orn: tavan_devam_20260310_1609 vs _1637)
        # sadece en yeni dosyayi al (dosya adi alfabetik -> son = en yeni saat)
        by_date = {}
        for d, p in items:
            if d not in by_date or p > by_date[d]:
                by_date[d] = p
        deduped = sorted(by_date.items())

        if target_date:
            filtered = [(d, p) for d, p in deduped if d == target_date]
            if filtered:
                result[scr] = filtered
        else:
            result[scr] = deduped

    return result


# -- CSV Parse Fonksiyonlari --

def _parse_nox_v3(path, screener_name):
    """NOX v3 daily/weekly CSV parse."""
    df = pd.read_csv(path)
    signals = []
    for _, row in df.iterrows():
        sig = str(row.get('signal', '')).strip()
        if sig == 'PIVOT_AL':
            direction = 'AL'
        elif sig == 'ADAY':
            direction = 'AL'
        elif sig == 'PIVOT_SAT':
            direction = 'SAT'
        elif sig == 'ZONE_ONLY':
            direction = 'AL'
        else:
            continue
        entry = {
            'screener': screener_name,
            'ticker': str(row['ticker']).strip(),
            'signal_date': str(row['signal_date']).strip(),
            'direction': direction,
            'signal_type': sig,
            'entry_price': float(row['close']),
            'quality': None,
        }
        tt = str(row.get('trigger_type', '')).strip()
        if tt and tt != 'nan':
            entry['trigger_type'] = tt
        if pd.notna(row.get('rs_score')):
            entry['rs_score'] = round(float(row['rs_score']), 3)
        wl = str(row.get('wl_status', '')).strip()
        if wl and wl != 'nan':
            entry['wl_status'] = wl
            entry['tb_stage'] = str(row.get('tb_stage', '')).strip()
            entry['delta_pct'] = float(row['delta_pct']) if pd.notna(row.get('delta_pct')) else None
        # fresh etiketi
        fresh = str(row.get('fresh', '')).strip()
        if fresh and fresh != 'nan':
            entry['fresh'] = fresh
        # gate durumu
        gate = str(row.get('gate', '')).strip()
        if gate and gate != 'nan':
            entry['gate'] = gate
        signals.append(entry)
    return signals


def _parse_divergence(path):
    """Divergence CSV parse."""
    df = pd.read_csv(path)
    signals = []
    for _, row in df.iterrows():
        d = str(row.get('direction', '')).strip().upper()
        if d == 'BUY':
            direction = 'AL'
        elif d == 'SELL':
            direction = 'SAT'
        else:
            continue
        signals.append({
            'screener': 'divergence',
            'ticker': str(row['ticker']).strip(),
            'signal_date': str(row['signal_date']).strip(),
            'direction': direction,
            'signal_type': str(row.get('div_type', '')).strip(),
            'entry_price': float(row['close']),
            'quality': int(row['quality']) if pd.notna(row.get('quality')) else None,
        })
    return signals


def _parse_alsat(path, date_str):
    """AL/SAT Screener CSV parse.
    Sinyal tipleri: DONUS, CMB, BILESEN, ERKEN, ZAYIF, PB
    Karar: AL / IZLE / ATLA
    """
    df = pd.read_csv(path)
    sig_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
    signals = []
    for _, row in df.iterrows():
        karar = str(row.get('karar', '')).strip()
        # ATLA = skip
        if karar == 'ATLA':
            continue
        direction = 'AL'

        entry = {
            'screener': 'alsat',
            'ticker': str(row['ticker']).strip(),
            'signal_date': sig_date,
            'direction': direction,
            'signal_type': str(row.get('signal', '')).strip(),
            'entry_price': float(row['close']),
            'quality': int(row['quality']) if pd.notna(row.get('quality')) else None,
            'karar': karar,
        }
        if pd.notna(row.get('rs_score')):
            entry['rs_score'] = round(float(row['rs_score']), 1)
        regime = str(row.get('regime', '')).strip()
        if regime and regime != 'nan':
            entry['regime'] = regime
        if pd.notna(row.get('oe')):
            entry['oe'] = int(row['oe'])
            entry['oe_detail'] = str(row.get('oe_detail', '')).strip()
        if pd.notna(row.get('rr')):
            entry['rr'] = round(float(row['rr']), 2)
        # MACD
        macd_val = row.get('macd')
        if pd.notna(macd_val):
            entry['macd'] = round(float(macd_val), 4)
        # Stop ve hedef seviyeleri
        if pd.notna(row.get('stop')):
            entry['stop_price'] = round(float(row['stop']), 2)
        if pd.notna(row.get('target')):
            entry['target_price'] = round(float(row['target']), 2)
        signals.append(entry)
    return signals




def _parse_regime_transition(path, date_str):
    """Regime Transition CSV parse (nox-regime workflow).
    entry_window: TAZE/YAKIN = giris penceresi acik, BEKLE/GEC = kapali
    Badge'ler: H+AL, H+PB (en yuksek WR setup'lari)
    """
    df = pd.read_csv(path)
    sig_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
    signals = []
    for _, row in df.iterrows():
        transition = str(row.get('transition', '')).strip()
        entry_window = str(row.get('entry_window', '')).strip()

        # TAZE/2.DALGA = giris penceresi acik -> AL, BEKLE/GEC = SAT
        if entry_window in ('TAZE', '2.DALGA', 'YAKIN'):
            direction = 'AL'
        else:
            direction = 'SAT'

        entry = {
            'screener': 'regime_transition',
            'ticker': str(row['ticker']).strip(),
            'signal_date': sig_date,
            'direction': direction,
            'signal_type': transition,
            'entry_price': float(row['close']) if pd.notna(row.get('close')) else 0,
            'quality': int(row['entry_score']) if pd.notna(row.get('entry_score')) else None,
            'entry_window': entry_window,
        }
        regime = str(row.get('regime_name', '')).strip()
        if regime and regime != 'nan':
            entry['regime'] = regime
        if pd.notna(row.get('oe_score')):
            entry['oe'] = int(row['oe_score'])
            entry['oe_detail'] = str(row.get('oe_tags', '')).strip()
        if pd.notna(row.get('trend_score')):
            entry['trend_score'] = int(row['trend_score'])
        # CMF
        cmf_val = row.get('cmf')
        if pd.notna(cmf_val):
            entry['cmf'] = round(float(cmf_val), 4)
        # ADX
        adx_val = row.get('adx')
        if pd.notna(adx_val):
            entry['adx'] = round(float(adx_val), 2)
        # Badge (H+AL, H+PB)
        badge = str(row.get('badge', '')).strip()
        if badge and badge != 'nan':
            entry['badge'] = badge
        signals.append(entry)
    return signals


def _parse_tavan(path, date_str, screener_name='tavan'):
    """Tavan Scanner CSV parse (tavan_devam + tavan_kandidat + eski format).
    Kolon: score/skor, streak (devam), vol_ratio/volume_ratio, rs, close.
    """
    df = pd.read_csv(path)
    sig_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
    signals = []
    for _, row in df.iterrows():
        # score vs skor (eski/yeni format)
        skor = 0
        for col in ('score', 'skor'):
            if pd.notna(row.get(col)):
                skor = int(row[col])
                break
        # RS
        rs_val = float(row['rs']) if pd.notna(row.get('rs')) else 0
        direction = 'AL' if rs_val >= 0 else 'SAT'
        # vol_ratio vs volume_ratio
        vol = 1.0
        for col in ('vol_ratio', 'volume_ratio'):
            if pd.notna(row.get(col)):
                vol = round(float(row[col]), 2)
                break

        entry = {
            'screener': screener_name,
            'ticker': str(row['ticker']).strip(),
            'signal_date': sig_date,
            'direction': direction,
            'signal_type': 'TAVAN' if screener_name == 'tavan' else 'KANDIDAT',
            'entry_price': float(row['close']) if pd.notna(row.get('close')) else 0,
            'quality': skor,
            'skor': skor,
        }
        if pd.notna(row.get('streak')):
            entry['streak'] = int(row['streak'])
        entry['volume_ratio'] = vol
        entry['rs'] = rs_val
        if pd.notna(row.get('cmf')):
            entry['cmf'] = round(float(row['cmf']), 4)
        if pd.notna(row.get('yab_degisim')):
            entry['yabanci_degisim'] = round(float(row['yab_degisim']), 2)
        elif pd.notna(row.get('yabanci_degisim')):
            entry['yabanci_degisim'] = round(float(row['yabanci_degisim']), 2)
        signals.append(entry)
    return signals


def _parse_kademe(path, date_str):
    """Kademe scanner CSV parse (MatriksIQ nox_kademe_*.csv).
    S/A oranı, bid/ask depth, karar.
    """
    df = pd.read_csv(path)
    sig_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
    signals = []
    for _, row in df.iterrows():
        sa = float(row.get('sat_al', 1.0)) if pd.notna(row.get('sat_al')) else 1.0
        karar = str(row.get('karar', '')).strip()
        kilitli = str(row.get('kilitli', 'False')).strip().lower() == 'true'

        if karar in ('GUCLU_AL', 'GEC'):
            direction = 'AL'
        elif karar in ('DIKKAT', 'ELEME'):
            direction = 'SAT'
        elif kilitli:
            direction = 'AL'
        else:
            direction = 'AL' if sa < 1.0 else 'SAT'

        entry = {
            'screener': 'kademe',
            'ticker': str(row['ticker']).strip(),
            'signal_date': sig_date,
            'direction': direction,
            'signal_type': karar or ('KILITLI' if kilitli else 'NOTR'),
            'entry_price': 0,
            'quality': None,
            'sa_ratio': round(sa, 3),
            'kilitli': kilitli,
        }
        if pd.notna(row.get('bid_depth')):
            entry['bid_depth'] = int(row['bid_depth'])
        if pd.notna(row.get('ask_depth')):
            entry['ask_depth'] = int(row['ask_depth'])
        signals.append(entry)
    return signals


def parse_all_csvs(csv_map):
    """Tum CSV'leri parse et -> birlesik sinyal listesi."""
    all_signals = []
    for screener, entries in csv_map.items():
        scr_total = 0
        for date_str, path in entries:
            try:
                if screener in ('nox_v3_weekly', 'nox_v3_daily'):
                    sigs = _parse_nox_v3(path, screener)
                elif screener == 'divergence':
                    sigs = _parse_divergence(path)
                elif screener == 'alsat':
                    sigs = _parse_alsat(path, date_str)
                elif screener == 'regime_transition':
                    sigs = _parse_regime_transition(path, date_str)
                elif screener in ('tavan', 'tavan_kandidat'):
                    sigs = _parse_tavan(path, date_str, screener)
                elif screener == 'kademe':
                    sigs = _parse_kademe(path, date_str)
                else:
                    continue
                for s in sigs:
                    s['csv_date'] = date_str
                all_signals.extend(sigs)
                scr_total += len(sigs)
            except Exception as e:
                print(f"  ! {screener}/{date_str} parse hata: {e}")
        if scr_total > 0:
            n_dates = len(entries)
            extra = f" ({n_dates} tarih)" if n_dates > 1 else ""
            print(f"  {screener}: {scr_total} sinyal{extra}")
    return all_signals


def build_xref(all_signals):
    """Capraz referans: (ticker, csv_date) -> {screener: [signals]}"""
    xref = {}
    for s in all_signals:
        key = (s['ticker'], s.get('csv_date', ''))
        xref.setdefault(key, {}).setdefault(s['screener'], []).append(s)
    return xref


# -- GitHub Artifact Indirme --

# Iki repo'dan artifact indir
_GH_ARTIFACT_SOURCES = [
    {
        "repo": "aalpkk/bist-tavan-screener",
        "prefix": "signals-",
        "csv_patterns": None,  # alsat CSV'leri
    },
    {
        "repo": "aalpkk/bist-tavan-screener",
        "prefix": "scan-results-",
        "csv_patterns": ['tavan_'],
    },
    {
        "repo": "aalpkk/nox-project",
        "prefix": "weekly-signals-",
        "csv_patterns": None,  # nox_v3 daily + weekly CSV'leri
    },
    {
        "repo": "aalpkk/nox-project",
        "prefix": "regime-daily-",
        "csv_patterns": ['regime_transition_'],
    },
    # divergence: PASİF — geliştirme bekliyor
    # {
    #     "repo": "aalpkk/nox-project",
    #     "prefix": "divergence-daily-",
    #     "csv_patterns": ['nox_divergence_'],
    # },
]


def fetch_github_artifacts(output_dir=None, max_artifacts=10):
    """GitHub Actions artifact'lerinden CSV'leri indir (her iki repo)."""
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), 'output')
    os.makedirs(output_dir, exist_ok=True)

    total_copied = 0
    for source in _GH_ARTIFACT_SOURCES:
        repo = source["repo"]
        prefix = source["prefix"]
        csv_prefixes = source["csv_patterns"]

        print(f"\n  {repo} / {prefix}*")
        try:
            result = subprocess.run(
                ['gh', 'api', f'repos/{repo}/actions/artifacts',
                 '--paginate', '--jq',
                 f'.artifacts[] | select(.name | startswith("{prefix}")) | "\\(.id)"'],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode != 0:
                print(f"    ! gh api hata: {result.stderr.strip()}")
                continue
            artifact_ids = [x.strip() for x in result.stdout.strip().split('\n')
                           if x.strip()]
            if not artifact_ids:
                print(f"    ! {prefix}* artifact bulunamadi")
                continue
            # En yeni N artifact'i al (GH API yeniden eskiye siralar)
            artifact_ids = artifact_ids[:max_artifacts]
            print(f"    {len(artifact_ids)} artifact bulundu")
        except FileNotFoundError:
            print("    ! gh CLI bulunamadi")
            continue
        except Exception as e:
            print(f"    ! Hata: {e}")
            continue

        for art_id in artifact_ids:
            tmp_dir = tempfile.mkdtemp(prefix='nox_gh_')
            try:
                result = subprocess.run(
                    ['gh', 'api', f'repos/{repo}/actions/artifacts/{art_id}/zip'],
                    capture_output=True, timeout=30,
                )
                if result.returncode != 0:
                    continue
                zip_path = os.path.join(tmp_dir, 'a.zip')
                with open(zip_path, 'wb') as f:
                    f.write(result.stdout)
                subprocess.run(['unzip', '-qo', zip_path, '-d', tmp_dir],
                               capture_output=True, timeout=15)
                for fname in os.listdir(tmp_dir):
                    if not fname.endswith('.csv'):
                        continue
                    if csv_prefixes and not any(fname.startswith(p) for p in csv_prefixes):
                        continue
                    dst = os.path.join(output_dir, fname)
                    if os.path.exists(dst):
                        continue
                    shutil.copy2(os.path.join(tmp_dir, fname), dst)
                    print(f"      <- {fname}")
                    total_copied += 1
            except Exception:
                pass
            finally:
                shutil.rmtree(tmp_dir, ignore_errors=True)

    if total_copied == 0:
        print("  Yeni CSV yok (hepsi zaten mevcut)")
    else:
        print(f"  {total_copied} yeni CSV indirildi")
    return total_copied


def get_latest_signals(output_dir=None, target_date=None, fetch_gh=False):
    """En son CSV'leri kesfedip parse et -> (signals, csv_map).
    fetch_gh=True ise once GitHub artifact'lerini indir."""
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), 'output')

    if fetch_gh:
        print("\nGitHub artifact'leri indiriliyor...")
        fetch_github_artifacts(output_dir)

    csv_map = discover_csvs(output_dir=output_dir, target_date=target_date)
    if not csv_map:
        print("Hic CSV bulunamadi")
        return [], csv_map
    print(f"{len(csv_map)} screener bulundu: {', '.join(csv_map.keys())}")
    signals = parse_all_csvs(csv_map)
    print(f"Toplam {len(signals)} sinyal parse edildi")
    return signals, csv_map


def get_signals_for_ticker(signals, ticker):
    """Belirli bir hisse icin tum sinyalleri filtrele."""
    ticker = ticker.upper().strip()
    candidates = [ticker, f"{ticker}.IS", ticker.replace('.IS', '')]
    return [s for s in signals if s['ticker'] in candidates]


def get_latest_date_signals(signals):
    """Her screener'in en son tarihli sinyallerini dondur."""
    if not signals:
        return []
    latest_by_screener = {}
    for s in signals:
        scr = s['screener']
        d = s.get('csv_date', '')
        if scr not in latest_by_screener or d > latest_by_screener[scr]:
            latest_by_screener[scr] = d
    return [s for s in signals
            if s.get('csv_date', '') == latest_by_screener.get(s['screener'], '')]


def summarize_signals(signals):
    """Sinyal ozetini dondur -- Claude'a gonderilecek kompakt format."""
    if not signals:
        return {"total": 0, "screeners": {}, "top_tickers": []}

    from collections import Counter

    by_screener = {}
    ticker_counts = Counter()
    for s in signals:
        scr = s['screener']
        by_screener.setdefault(scr, {"AL": 0, "SAT": 0, "total": 0})
        by_screener[scr][s['direction']] = by_screener[scr].get(s['direction'], 0) + 1
        by_screener[scr]["total"] += 1
        if s['direction'] == 'AL':
            ticker_counts[s['ticker']] += 1

    top_tickers = ticker_counts.most_common(10)

    # Her screener'in son tarihini bul
    screener_dates = {}
    for s in signals:
        scr = s['screener']
        d = s.get('csv_date', '')
        if scr not in screener_dates or d > screener_dates[scr]:
            screener_dates[scr] = d

    return {
        "total": len(signals),
        "screeners": by_screener,
        "top_tickers": [{"ticker": t, "count": c} for t, c in top_tickers],
        "screener_dates": screener_dates,
    }


# -- Signals JSON Export/Import (Render entegrasyonu) --

def export_signals_json(signals, output_path):
    """Sinyalleri JSON dosyasına yaz (GitHub Pages'e push için)."""
    from datetime import datetime, timezone, timedelta
    _TZ_TR = timezone(timedelta(hours=3))
    now = datetime.now(_TZ_TR)

    data = {
        "exported_at": now.isoformat(),
        "total": len(signals),
        "signals": signals,
    }
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, default=str)
    print(f"  ✅ {len(signals)} sinyal → {output_path}")
    return output_path


def fetch_signals_from_url(base_url=None, filename="latest_signals.json"):
    """GitHub Pages'ten sinyal JSON'u indir (Render ortamı için).

    Returns: (signals_list, None) veya ([], None) hata durumunda.
    """
    import requests

    if base_url is None:
        base_url = os.environ.get("GH_PAGES_BASE_URL", "").rstrip("/")
    if not base_url:
        return [], None

    url = f"{base_url}/{filename}"
    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code != 200:
            print(f"  ⚠️ Sinyal JSON indirilemedi: HTTP {resp.status_code}")
            return [], None

        data = resp.json()
        signals = data.get("signals", [])
        exported_at = data.get("exported_at", "?")
        print(f"  ✅ {len(signals)} sinyal indirildi (güncelleme: {exported_at})")
        return signals, None
    except Exception as e:
        print(f"  ⚠️ Sinyal JSON hatası: {e}")
        return [], None


# -- VDS JSON Fetch (MKK, Takas, Kademe) --

def _fetch_vds_json(filename, base_url=None):
    """GitHub Pages'ten VDS JSON dosyası indir.

    Args:
        filename: ör. "mkk_data.json", "takas_data.json", "kademe_data.json"
        base_url: GitHub Pages base URL (None ise env'den al)

    Returns:
        dict veya None
    """
    import requests

    if base_url is None:
        base_url = os.environ.get("GH_PAGES_BASE_URL", "").rstrip("/")
    if not base_url:
        return None

    url = f"{base_url}/{filename}"
    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code != 200:
            print(f"  ⚠️ {filename} indirilemedi: HTTP {resp.status_code}")
            return None
        data = resp.json()
        extracted = data.get("extracted_at", "?")
        total = data.get("total", 0)
        print(f"  ✅ {filename}: {total} kayıt (güncelleme: {extracted})")
        return data
    except Exception as e:
        print(f"  ⚠️ {filename} hatası: {e}")
        return None


def fetch_mkk_data(base_url=None):
    """GitHub Pages'ten MKK history JSON indir ve en son snapshot'ı döndür.

    mkk_history.json formatı:
        {"2026-03-13": {"GARAN": {"b": 5.33, "k": 94.67, "ys": 141313}, ...}, ...}

    Returns:
        dict: {extracted_at, source, total, data: {TICKER: {bireysel_pct, kurumsal_pct, ...}}}
        veya None
    """
    import requests

    if base_url is None:
        base_url = os.environ.get("GH_PAGES_BASE_URL", "").rstrip("/")
    if not base_url:
        return None

    url = f"{base_url}/mkk_history.json"
    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code != 200:
            print(f"  ⚠️ mkk_history.json indirilemedi: HTTP {resp.status_code}")
            return None

        history = resp.json()
        if not history:
            return None

        dates = sorted(history.keys())
        latest_date = dates[-1]
        latest_snap = history[latest_date]

        # Bir önceki günü bul (fark hesabı için)
        prev_snap = history[dates[-2]] if len(dates) >= 2 else {}

        # 5 gün önceki snapshot (haftalık fark)
        week_snap = {}
        for d in reversed(dates):
            if d < latest_date and len([x for x in dates if x <= d]) <= len(dates) - 4:
                week_snap = history[d]
                break

        # Normalize: b/k/ys -> bireysel_pct/kurumsal_pct/yatirimci_sayisi
        data = {}
        for ticker, vals in latest_snap.items():
            entry = {
                "bireysel_pct": vals.get("b", 0),
                "kurumsal_pct": vals.get("k", 0),
                "yatirimci_sayisi": vals.get("ys", 0),
                "tarih": latest_date,
            }

            # Günlük fark
            prev = prev_snap.get(ticker)
            if prev:
                entry["bireysel_fark_1g"] = round(vals.get("b", 0) - prev.get("b", 0), 2)

            # Haftalık fark
            wk = week_snap.get(ticker)
            if wk:
                entry["bireysel_fark_5g"] = round(vals.get("b", 0) - wk.get("b", 0), 2)

            data[ticker] = entry

        print(f"  ✅ MKK: {len(data)} hisse, son tarih: {latest_date} ({len(dates)} gün history)")
        return {
            "extracted_at": latest_date,
            "source": "matriks_iq_mkk",
            "total": len(data),
            "history_days": len(dates),
            "data": data,
        }
    except Exception as e:
        print(f"  ⚠️ MKK history hatası: {e}")
        return None


def fetch_takas_data(base_url=None):
    """GitHub Pages'ten Takas (aracı kurum pozisyon) JSON indir.

    Returns:
        dict: {extracted_at, source, total, data: {TICKER: {...}}} veya None
    """
    return _fetch_vds_json("takas_data.json", base_url)


def fetch_takas_history(base_url=None):
    """GitHub Pages'ten Takas history JSON indir.

    takas_history.json formatı:
        {"2026-03-14": {"GARAN": {"top_alici": [...], "net_tip": {...}, ...}}}

    Returns:
        dict veya None
    """
    import requests

    if base_url is None:
        base_url = os.environ.get("GH_PAGES_BASE_URL", "").rstrip("/")
    if not base_url:
        return None

    url = f"{base_url}/takas_history.json"
    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code != 200:
            print(f"  ⚠️ takas_history.json indirilemedi: HTTP {resp.status_code}")
            return None
        history = resp.json()
        if history:
            dates = sorted(history.keys())
            print(f"  ✅ Takas history: {len(dates)} gün ({dates[0]} → {dates[-1]})")
        return history
    except Exception as e:
        print(f"  ⚠️ Takas history hatası: {e}")
        return None


def fetch_kademe_data(base_url=None):
    """GitHub Pages'ten Kademe (emir defteri) JSON indir.

    Returns:
        dict: {extracted_at, source, total, data: {TICKER: {...}}} veya None
    """
    return _fetch_vds_json("kademe_data.json", base_url)
