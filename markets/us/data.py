"""
NOX Project — US Market Data
NASDAQ Screener API + yfinance veri indirme.
Catalyst screener için enrichment fonksiyonları.
"""
import time
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

pd.set_option('future.no_silent_downcasting', True)


# ── NASDAQ SCREENER API ──
_NASDAQ_API_URL = "https://api.nasdaq.com/api/screener/stocks"

# ── STATIC FALLBACKS ──
_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
_NDX100_URL = "https://en.wikipedia.org/wiki/NASDAQ-100#Components"


def get_nasdaq_screener_tickers(min_mcap=300e6, min_price=5.0):
    """NASDAQ Screener API'den tüm NYSE + NASDAQ hisselerini çek.

    API ~7000 ticker döndürür. min_mcap ve min_price ile filtrelenir.
    Midas (TR broker) üzerinde alınıp satılabilen tüm US hisseleri bu listede.

    Returns: sorted list of ticker symbols
    """
    import requests as req

    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
        'Accept': 'application/json',
    }
    params = {
        'tableonly': 'true',
        'limit': '25',
        'offset': '0',
        'download': 'true',
    }

    try:
        resp = req.get(_NASDAQ_API_URL, headers=headers, params=params, timeout=30)
        if resp.status_code != 200:
            print(f"⚠️ NASDAQ Screener API {resp.status_code}, fallback'e dönülüyor")
            return None

        data = resp.json()
        rows = data.get('data', {}).get('rows', [])
        if not rows:
            print("⚠️ NASDAQ API boş veri döndürdü, fallback'e dönülüyor")
            return None

        print(f"📡 NASDAQ API: {len(rows)} toplam hisse alındı")

        # Filtrele: market cap, price, symbol kalitesi
        tickers = []
        skipped_mcap = 0
        skipped_price = 0
        skipped_symbol = 0
        skipped_spac = 0
        skipped_etf = 0

        _SPAC_KEYWORDS = ['ACQUISITION', 'SPAC', 'BLANK CHECK']
        _ETF_KEYWORDS = [' ETF', ' FUND', ' TRUST', ' INDEX']

        for row in rows:
            symbol = row.get('symbol', '').strip()

            # Sembol kalitesi: sadece basit ticker'lar (warrants, units, rights hariç)
            if not symbol or not symbol.isalpha() or len(symbol) > 5:
                skipped_symbol += 1
                continue

            # SPAC filtresi
            name = row.get('name', '')
            name_upper = name.upper() if name else ''
            if any(w in name_upper for w in _SPAC_KEYWORDS):
                skipped_spac += 1
                continue

            # ETF/Fund filtresi
            if any(w in name_upper for w in _ETF_KEYWORDS):
                skipped_etf += 1
                continue

            # Fiyat filtresi
            price_str = str(row.get('lastsale', '$0')).replace('$', '').replace(',', '').strip()
            try:
                price = float(price_str)
            except (ValueError, TypeError):
                price = 0
            if price < min_price:
                skipped_price += 1
                continue

            # Market cap filtresi
            mcap_str = str(row.get('marketCap', '0')).replace(',', '').strip()
            try:
                mcap = float(mcap_str)
            except (ValueError, TypeError):
                mcap = 0
            if mcap < min_mcap:
                skipped_mcap += 1
                continue

            tickers.append(symbol)

        tickers = sorted(set(tickers))
        print(f"✅ NASDAQ Screener: {len(tickers)} hisse (mcap>=${min_mcap/1e6:.0f}M, fiyat>=${min_price})")
        print(f"   Atlanan: {skipped_mcap} mcap, {skipped_price} fiyat, {skipped_symbol} sembol, {skipped_spac} SPAC, {skipped_etf} ETF")
        return tickers

    except Exception as e:
        print(f"⚠️ NASDAQ Screener API hatası: {e}")
        return None


def get_sp500_tickers():
    """S&P 500 ticker listesi — Wikipedia'dan (fallback)."""
    try:
        tables = pd.read_html(_SP500_URL)
        df = tables[0]
        tickers = sorted(df['Symbol'].str.replace('.', '-', regex=False).tolist())
        print(f"✅ S&P 500 → {len(tickers)} hisse")
        return tickers
    except Exception as e:
        print(f"⚠️ S&P 500 Wikipedia hatası: {e}")
        return _SP500_STATIC


def get_ndx100_tickers():
    """NASDAQ-100 ticker listesi — Wikipedia'dan (fallback)."""
    try:
        tables = pd.read_html(_NDX100_URL)
        for t in tables:
            if 'Ticker' in t.columns:
                tickers = sorted(t['Ticker'].str.replace('.', '-', regex=False).tolist())
                print(f"✅ NASDAQ-100 → {len(tickers)} hisse")
                return tickers
            if 'Symbol' in t.columns:
                tickers = sorted(t['Symbol'].str.replace('.', '-', regex=False).tolist())
                print(f"✅ NASDAQ-100 → {len(tickers)} hisse")
                return tickers
        print("⚠️ NASDAQ-100 tablosu bulunamadı, fallback kullanılıyor")
        return _NDX100_STATIC
    except Exception as e:
        print(f"⚠️ NASDAQ-100 Wikipedia hatası: {e}, fallback kullanılıyor")
        return _NDX100_STATIC


def get_all_us_tickers(min_mcap=300e6, min_price=5.0):
    """Tüm US hisseleri: NASDAQ API → Wikipedia fallback → static fallback.

    NASDAQ API: ~2000-3000 hisse (mcap filtrelenmiş)
    Wikipedia fallback: S&P 500 + NASDAQ 100 (~550 hisse)
    Static fallback: 169 hisse
    """
    # Önce NASDAQ API dene — en geniş evren
    tickers = get_nasdaq_screener_tickers(min_mcap=min_mcap, min_price=min_price)
    if tickers and len(tickers) >= 100:
        return tickers

    # Fallback: Wikipedia S&P 500 + NASDAQ 100
    print("📋 NASDAQ API başarısız, Wikipedia'ya dönülüyor...")
    sp = get_sp500_tickers()
    ndx = get_ndx100_tickers()
    combined = sorted(set(sp + ndx))
    print(f"📊 Toplam US: {len(combined)} benzersiz ticker ({len(sp)} S&P + {len(ndx)} NDX)")
    return combined


def _normalize_df(df, ticker=None):
    """yfinance MultiIndex → düz DataFrame."""
    if isinstance(df.columns, pd.MultiIndex):
        level_names = list(df.columns.names)
        if ticker:
            try:
                # Yeni yfinance: levels = ['Ticker', 'Price']
                if 'Ticker' in level_names:
                    sub = df.xs(ticker, level='Ticker', axis=1)
                else:
                    # Eski format: level 0 = Price, level 1 = ticker
                    level0_vals = df.columns.get_level_values(0).unique().tolist()
                    if 'Price' in level0_vals or 'Close' in level0_vals:
                        sub = df.xs(ticker, level=1, axis=1)
                    else:
                        sub = df.xs(ticker, level=0, axis=1)
                sub = sub[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
                return sub
            except Exception:
                pass
        # Tek ticker: sadece level düşür
        if 'Ticker' in level_names:
            df.columns = df.columns.droplevel('Ticker')
        else:
            try:
                df.columns = df.columns.droplevel(0)
            except Exception:
                try:
                    df.columns = df.columns.droplevel(1)
                except Exception:
                    pass
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col not in df.columns:
            return None
    return df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()


def fetch_data(tickers, period="1y"):
    """Batch download US hisseleri."""
    n_batches = (len(tickers) + 99) // 100
    print(f"📡 {len(tickers)} US hisse verisi çekiliyor ({n_batches} batch, period={period})...")
    result = {}
    batch_size = 100
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        batch_num = i // batch_size + 1
        try:
            raw = yf.download(batch, period=period, group_by='ticker',
                              auto_adjust=True, progress=False, threads=True)
            if raw.empty:
                continue
            if batch_num % 5 == 0 or batch_num == n_batches:
                print(f"  ... batch {batch_num}/{n_batches} ({len(result)} hisse yüklendi)")
            for t in batch:
                try:
                    if len(batch) == 1:
                        sub = _normalize_df(raw)
                    else:
                        sub = _normalize_df(raw, t)
                    if sub is not None and len(sub) >= 60:
                        result[t] = sub
                except:
                    pass
        except Exception as e:
            print(f"  ⚠️ Batch hata: {e}")

    print(f"✅ {len(result)}/{len(tickers)} hisse yüklendi")
    return result


def fetch_benchmark(period="1y"):
    """SPY benchmark verisi."""
    try:
        df = yf.download("SPY", period=period, auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            # Yeni yfinance: Level 0='Price', Level 1='Ticker'
            # Ticker seviyesini düşür, Price seviyesini tut
            if 'Ticker' in df.columns.names:
                df.columns = df.columns.droplevel('Ticker')
            elif 'Price' in df.columns.names:
                df.columns = df.columns.droplevel('Price')
                # Eğer Price düştüyse sadece ticker kalır, tekrar dene
                if 'Open' not in df.columns:
                    df = yf.download("SPY", period=period, auto_adjust=True, progress=False)
                    df.columns = df.columns.droplevel(1)
            else:
                df.columns = df.columns.droplevel(1)
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
        print(f"✅ SPY benchmark: {len(df)} gün")
        return df
    except Exception as e:
        print(f"⚠️ SPY benchmark hatası: {e}")
        return None


# ── FDA / PDUFA TAKVİMİ ──

def _parse_fda_ics(ics_path):
    """CatalystAlert ICS dosyasından FDA/biyotek takvimi parse et.

    Sadece PDUFA kayıtlarını döndürür (mevcut scan_biotech_catalyst uyumu).
    AI-extracted kayıtlar ai_extracted=True ile işaretlenir.

    Returns: {ticker: {date, drug, phase, catalyst_type, ai_extracted}} veya None
    """
    import re

    try:
        with open(ics_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"  ⚠️ ICS dosyası okunamadı: {e}")
        return None

    # VEVENT bloklarını ayır
    blocks = content.split('BEGIN:VEVENT')
    if len(blocks) < 2:
        print("  ⚠️ ICS dosyasında VEVENT bulunamadı")
        return None

    today = datetime.now().strftime('%Y%m%d')
    result = {}

    for block in blocks[1:]:
        block = block.split('END:VEVENT')[0]

        # DTSTART çıkar
        dt_match = re.search(r'DTSTART;VALUE=DATE:(\d{8})', block)
        if not dt_match:
            dt_match = re.search(r'DTSTART:(\d{8})', block)
        if not dt_match:
            continue
        date_raw = dt_match.group(1)

        # Geçmiş tarihli kayıtları atla
        if date_raw < today:
            continue

        date_str = f"{date_raw[:4]}-{date_raw[4:6]}-{date_raw[6:8]}"

        # SUMMARY çıkar (satır devamı — unfold)
        summary_match = re.search(r'SUMMARY:(.*?)(?:\r?\n(?! ))', block, re.DOTALL)
        if not summary_match:
            continue
        summary = summary_match.group(1).replace('\r\n ', '').replace('\n ', '').strip()

        # Ticker: SUMMARY başı
        colon_idx = summary.find(':')
        if colon_idx < 0 or colon_idx > 10:
            continue
        ticker = summary[:colon_idx].strip().upper()
        if not ticker or not ticker.isalpha():
            continue
        rest = summary[colon_idx + 1:].strip()

        # Catalyst type tespiti
        if 'FDA PDUFA Date' in rest:
            catalyst_type = 'PDUFA'
        elif re.search(r'Phase\s+\S+\s+Results?\s+Expected', rest, re.IGNORECASE):
            catalyst_type = 'PHASE_RESULT'
        elif 'Earnings' in rest:
            catalyst_type = 'EARNINGS'
        elif 'Advisory Committee' in rest:
            catalyst_type = 'ADVISORY'
        else:
            catalyst_type = 'OTHER'

        # Sadece PDUFA kayıtlarını döndür
        if catalyst_type != 'PDUFA':
            continue

        # İlaç adı: "FDA PDUFA Date " sonrası, parantez öncesi
        drug = ''
        drug_match = re.search(r'FDA PDUFA Date\s+(.+?)(?:\s*\(|$)', rest)
        if drug_match:
            drug = drug_match.group(1).strip()

        # Review type: parantez içinden
        phase = ''
        phase_match = re.search(r'\(([^)]+)\)', rest)
        if phase_match:
            phase = phase_match.group(1).strip().lower()

        # AI-extracted tespiti
        ai_extracted = 'AI-extracted' in block or 'ai-extracted' in block.lower()

        # DESCRIPTION kontrolü (unfold)
        desc_match = re.search(r'DESCRIPTION:(.*?)(?:\r?\n(?! ))', block, re.DOTALL)
        if desc_match:
            desc = desc_match.group(1).replace('\r\n ', '').replace('\n ', '')
            if 'AI-extracted' in desc or 'ai-extracted' in desc.lower():
                ai_extracted = True

        # Aynı ticker birden fazla PDUFA → en yakın tarihlisini al
        if ticker in result:
            if date_str >= result[ticker]['date']:
                continue

        result[ticker] = {
            'date': date_str,
            'drug': drug,
            'phase': phase,
            'catalyst_type': 'PDUFA',
            'ai_extracted': ai_extracted,
        }

    return result if result else None


def _fetch_fda_catalystalert():
    """CatalystAlert.io Supabase API'den PDUFA takvimi çek.

    Birincil kaynak — yapılandırılmış veri, ticker eşleştirmeli, ~50+ PDUFA.
    Returns: dict {ticker: {date, drug, phase, catalyst_type}} veya None (hata)
    """
    import re
    import requests as req

    try:
        # Supabase anon key'i frontend JS'ten al
        headers_js = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'}
        resp = req.get('https://catalystalert.io/_next/static/chunks/app/layout-bf927e3762d10073.js',
                       headers=headers_js, timeout=15)
        if resp.status_code != 200:
            # Layout chunk URL değişmiş olabilir — ana sayfadan bul
            main_resp = req.get('https://catalystalert.io', headers=headers_js, timeout=15)
            layout_match = re.search(r'(/_next/static/chunks/app/layout-[a-f0-9]+\.js)', main_resp.text)
            if not layout_match:
                return None
            resp = req.get(f'https://catalystalert.io{layout_match.group(1)}',
                           headers=headers_js, timeout=15)
            if resp.status_code != 200:
                return None

        keys = re.findall(r'(eyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+)', resp.text)
        if not keys:
            return None
        anon_key = keys[0]

        # JWT'den Supabase ref'i çıkar
        import base64, json
        parts = anon_key.split('.')
        payload_b64 = parts[1] + '=' * (4 - len(parts[1]) % 4)
        payload = json.loads(base64.urlsafe_b64decode(payload_b64))
        sb_ref = payload.get('ref', '')
        if not sb_ref:
            return None

        sb_url = f'https://{sb_ref}.supabase.co'
        headers = {'apikey': anon_key, 'Authorization': f'Bearer {anon_key}'}

        # PDUFA kayıtlarını company ticker'ı ile birlikte çek
        today_str = datetime.now().strftime('%Y-%m-%d')
        api_url = (f'{sb_url}/rest/v1/catalysts'
                   f'?select=type,title,expected_date,outcome,importance,'
                   f'companies:company_id(ticker,name)'
                   f'&type=eq.PDUFA'
                   f'&expected_date=gte.{today_str}'
                   f'&order=expected_date.asc'
                   f'&limit=100')

        resp = req.get(api_url, headers=headers, timeout=15)
        if resp.status_code != 200:
            return None

        data = resp.json()
        if not data:
            return None

        result = {}
        for entry in data:
            comp = entry.get('companies') or {}
            ticker = comp.get('ticker', '')
            if not ticker or not ticker.isalpha():
                continue

            date_str = (entry.get('expected_date') or '')[:10]
            if not date_str:
                continue

            # İlaç adını title'dan çıkar
            title = entry.get('title', '')
            drug = title.replace('FDA PDUFA Date ', '').strip()

            result[ticker] = {
                'date': date_str,
                'drug': drug,
                'phase': f"importance:{entry.get('importance', '-')}",
                'catalyst_type': 'PDUFA',
            }

        return result if result else None

    except Exception as e:
        print(f"  ⚠️ CatalystAlert hatası: {e}")
        return None


def _scrape_rttnews_fda():
    """RTTNews FDA takviminden PDUFA verisi scrape et (fallback).

    Returns: dict {ticker: {date, drug, phase, catalyst_type}} veya None
    """
    import re
    import requests as req
    import time as _time

    try:
        all_entries = []
        page = 1
        while True:
            url = f"https://www.rttnews.com/corpinfo/fdacalendar.aspx?PageNum={page}"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
                              'AppleWebKit/537.36 (KHTML, like Gecko) '
                              'Chrome/122.0.0.0 Safari/537.36',
            }
            resp = req.get(url, headers=headers, timeout=15)
            if resp.status_code != 200:
                break

            html = resp.text
            idx = html.find('dvCalendar')
            if idx < 0:
                break

            calendar_html = html[idx:]
            chunks = re.split(r'<div class="grid-row border-bottom[^"]*">', calendar_html)

            page_entries = []
            for chunk in chunks[1:]:
                entry = {}
                tick = re.search(r'symbol=([A-Z]+)', chunk)
                if tick:
                    entry['ticker'] = tick.group(1)
                date_m = re.search(r'class="bg-purple"[^>]*>\s*(\d{2}/\d{2}/\d{4})', chunk)
                if date_m:
                    entry['pdufa_date'] = date_m.group(1)
                drug = re.search(r'class="tblcontent2">\s*(.*?)\s*</div', chunk, re.DOTALL)
                if drug:
                    entry['drug'] = re.sub(r'<[^>]+>', '', drug.group(1)).strip()
                ev = re.search(r'class="tblcontent3">(.*?)</div', chunk, re.DOTALL)
                if ev:
                    text = re.sub(r'<[^>]+>', ' ', ev.group(1))
                    entry['event'] = re.sub(r'\s+', ' ', text).strip()
                if entry.get('ticker') and entry.get('pdufa_date'):
                    page_entries.append(entry)

            all_entries.extend(page_entries)

            # Sayfa navigasyonu
            pages = re.findall(r'PageNum=(\d+)', html)
            max_page = max(int(p) for p in pages) if pages else 1
            if page >= max_page:
                break
            page += 1
            _time.sleep(1)

        if not all_entries:
            return None

        today = datetime.now()
        result = {}
        for e in all_entries:
            try:
                pdufa_dt = datetime.strptime(e['pdufa_date'], '%m/%d/%Y')
            except ValueError:
                continue
            if pdufa_dt < today:
                continue
            result[e['ticker']] = {
                'date': pdufa_dt.strftime('%Y-%m-%d'),
                'drug': e.get('drug', ''),
                'phase': e.get('event', ''),
                'catalyst_type': 'PDUFA',
            }

        return result if result else None

    except Exception as e:
        print(f"  ⚠️ RTTNews hatası: {e}")
        return None


def fetch_fda_calendar(ics_path=None):
    """FDA/PDUFA takvimi — üç kaynaklı (fallback zinciri).

    1. ICS dosyası (varsa) — en kapsamlı, 66+ PDUFA
    2. CatalystAlert.io (Supabase API) — yapılandırılmış veri
    3. RTTNews (HTML scrape) — fallback

    Döndürür: {ticker: {date, drug, phase, catalyst_type}}
    """
    # Kaynak 1: ICS
    if ics_path:
        result = _parse_fda_ics(ics_path)
        if result:
            clean = {k: v for k, v in result.items() if not v.get('ai_extracted')}
            ai_count = len(result) - len(clean)
            msg = f"✅ FDA takvimi: {len(clean)} yaklaşan PDUFA (ICS)"
            if ai_count:
                msg += f" (+{ai_count} AI-extracted atlandı)"
            print(msg)
            return clean

    # Kaynak 2: CatalystAlert.io
    result = _fetch_fda_catalystalert()
    if result:
        print(f"✅ FDA takvimi: {len(result)} yaklaşan PDUFA (CatalystAlert)")
        return result

    # Kaynak 3: RTTNews (fallback)
    print("  ℹ️ CatalystAlert başarısız, RTTNews deneniyor...")
    result = _scrape_rttnews_fda()
    if result:
        print(f"✅ FDA takvimi: {len(result)} yaklaşan PDUFA (RTTNews fallback)")
        return result

    print("⚠️ FDA takvimi çekilemedi (tüm kaynaklar başarısız)")
    return {}


# ── ENRICHMENT (Catalyst Screener — Faz 2) ──

def fetch_ticker_info(tickers):
    """Bireysel ticker bilgileri: market cap, float, short interest, sector.
    Sadece Faz 1'i geçen ticker'lar için çağrılır (~50 ticker).
    """
    print(f"📡 {len(tickers)} ticker detay bilgisi çekiliyor...")
    result = {}
    fail_count = 0
    for i, t in enumerate(tickers):
        for attempt in range(2):  # 1 retry
            try:
                info = yf.Ticker(t).info or {}
                if not info or not info.get('marketCap'):
                    if attempt == 0:
                        time.sleep(1)
                        continue
                    break
                result[t] = {
                    'market_cap': info.get('marketCap'),
                    'float_shares': info.get('floatShares'),
                    'short_pct': info.get('shortPercentOfFloat'),
                    'short_ratio': info.get('shortRatio'),  # days to cover
                    'sector': info.get('sector', ''),
                    'industry': info.get('industry', ''),
                    'name': info.get('shortName', t),
                    'earnings_date': None,
                }
                # Earnings tarihi — birden fazla field dene
                ed = info.get('earningsTimestamp')
                if not ed:
                    # Alternatif: earningsDate veya earningsDates
                    ed = info.get('mostRecentQuarter')
                if ed and isinstance(ed, (int, float)):
                    result[t]['earnings_date'] = datetime.fromtimestamp(ed)
                break
            except Exception as e:
                if attempt == 0:
                    time.sleep(1)
                    continue
                fail_count += 1
        time.sleep(0.5)
        if (i + 1) % 20 == 0:
            print(f"  ... {i + 1}/{len(tickers)} tamamlandı ({len(result)} başarılı)")
    if fail_count > 0:
        print(f"  ⚠️ {fail_count} ticker info alınamadı")
    print(f"✅ {len(result)}/{len(tickers)} ticker bilgisi alındı")
    return result


def fetch_insider_data(tickers):
    """Son 90 gün insider işlemleri.
    Sadece Faz 1'i geçen ticker'lar için çağrılır.
    """
    print(f"📡 {len(tickers)} ticker insider verisi çekiliyor...")
    result = {}
    cutoff = datetime.now() - timedelta(days=90)
    for i, t in enumerate(tickers):
        try:
            tk = yf.Ticker(t)
            txns = tk.insider_transactions
            if txns is not None and not txns.empty and 'Text' in txns.columns:
                # Son 90 gün filtre
                if 'Start Date' in txns.columns:
                    txns = txns.copy()
                    txns['Start Date'] = pd.to_datetime(txns['Start Date'], errors='coerce')
                    txns = txns.dropna(subset=['Start Date'])
                    txns = txns[txns['Start Date'] >= cutoff]
                if not txns.empty:
                    result[t] = txns
            time.sleep(0.3)
        except Exception:
            pass
        if (i + 1) % 20 == 0:
            print(f"  ... {i + 1}/{len(tickers)} tamamlandı ({len(result)} veri)")
    print(f"✅ {len(result)}/{len(tickers)} insider verisi alındı")
    return result


# ── STATIC FALLBACKS (top 50 S&P + NDX for offline) ──
_SP500_STATIC = [
    "AAPL","ABBV","ABT","ACN","ADBE","AIG","AMD","AMGN","AMT","AMZN",
    "AVGO","AXP","BA","BAC","BLK","BMY","BRK-B","C","CAT","CHTR",
    "CL","CMCSA","COF","COP","COST","CRM","CSCO","CVX","DE","DHR",
    "DIS","DOW","DUK","EMR","F","FDX","GD","GE","GILD","GM",
    "GOOG","GOOGL","GS","HD","HON","IBM","INTC","INTU","ISRG","JNJ",
    "JPM","KHC","KO","LIN","LLY","LMT","LOW","MA","MCD","MDLZ",
    "MDT","MET","META","MMM","MO","MRK","MS","MSFT","NEE","NFLX",
    "NKE","NOW","NVDA","ORCL","PEP","PFE","PG","PM","PYPL","QCOM",
    "RTX","SBUX","SCHW","SO","SPG","T","TGT","TMO","TMUS","TSLA",
    "TXN","UNH","UNP","UPS","V","VZ","WBA","WFC","WMT","XOM",
]

_NDX100_STATIC = [
    "AAPL","ABNB","ADBE","ADI","ADP","ADSK","AEP","AMAT","AMD","AMGN",
    "AMZN","ANSS","ARM","ASML","AVGO","AZN","BIIB","BKNG","BKR","CCEP",
    "CDNS","CDW","CEG","CHTR","CMCSA","COST","CPRT","CRWD","CSCO","CSGP",
    "CSX","CTAS","CTSH","DASH","DDOG","DLTR","DXCM","EA","EXC","FANG",
    "FAST","FTNT","GEHC","GFS","GILD","GOOG","GOOGL","HON","IDXX","ILMN",
    "INTC","INTU","ISRG","KDP","KHC","KLAC","LIN","LRCX","LULU","MAR",
    "MCHP","MDB","MDLZ","MELI","META","MNST","MRNA","MRVL","MSFT","MU",
    "NFLX","NVDA","NXPI","ODFL","ON","ORLY","PANW","PAYX","PCAR","PDD",
    "PEP","PYPL","QCOM","REGN","ROP","ROST","SBUX","SNPS","TEAM","TMUS",
    "TSLA","TTD","TTWO","TXN","VRSK","VRTX","WBD","WDAY","XEL","ZS",
]
