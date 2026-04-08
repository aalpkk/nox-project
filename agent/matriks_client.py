"""
NOX Agent — Matriks MCP HTTP Client
Doğrudan HTTP POST ile Matriks MCP API'ye istek atar.
GitHub Actions'ta MCP stdio proxy'ye gerek kalmadan çalışır.

Kullanım:
    client = MatriksClient()
    flow = client.get_institutional_flow("GARAN")
    settlement = client.get_settlement("GARAN")
    price = client.get_market_price("GARAN")
"""
import os
import time
import json
import requests
from datetime import datetime, timedelta, timezone
from typing import Optional


MCP_URL = "https://mcp.matriks.ai/mcp"
_DEFAULT_CLIENT_ID = "33667"
_RATE_LIMIT_SEC = 0.5   # başlangıç — adaptive olarak ayarlanır
_RATE_MIN = 0.3         # minimum bekleme (429 gelmezse buraya kadar düşer)
_RATE_MAX = 2.0         # maksimum bekleme (429 gelince buraya kadar çıkar)
_RATE_STEP_DOWN = 0.05  # her başarılı çağrıda bu kadar düş
_RATE_STEP_UP = 0.5     # her 429'da bu kadar artır
_TIMEOUT = 30
_MSG_ID_COUNTER = 0
_429_COUNT = 0  # global 429 sayacı
_429_MAX = 15   # circuit breaker — limit artırıldı (eskiden 10)

_TZ_TR = timezone(timedelta(hours=3))

# Çoklu periyot: (key, gün sayısı)
_PERIODS = [
    ("daily", 1),
    ("weekly", 5),
    ("monthly", 20),
    ("quarterly", 60),
]


class MatriksClient:
    """Matriks MCP API HTTP client — session yönetimi + rate limiting."""

    def __init__(self, api_key: str = None, client_id: str = None):
        self.api_key = api_key or os.environ.get("MATRIKS_API_KEY", "")
        self.client_id = client_id or os.environ.get("MATRIKS_CLIENT_ID", _DEFAULT_CLIENT_ID)
        self.session_id: Optional[str] = None
        self._last_call = 0.0
        self._initialized = False
        self._partial_results: Optional[dict] = None  # timeout'ta kısmi veri kurtarma
        # Debug: credentials doğrulama
        key_preview = self.api_key[:12] + "..." if len(self.api_key) > 12 else self.api_key
        print(f"  [Matriks] Client-ID: {self.client_id[:8]}..., Key: {key_preview}")

    # ──────────────────────────────────────────
    # Low-level MCP JSON-RPC
    # ──────────────────────────────────────────

    def _next_id(self) -> int:
        global _MSG_ID_COUNTER
        _MSG_ID_COUNTER += 1
        return _MSG_ID_COUNTER

    def _rate_wait(self):
        global _RATE_LIMIT_SEC
        elapsed = time.time() - self._last_call
        if elapsed < _RATE_LIMIT_SEC:
            time.sleep(_RATE_LIMIT_SEC - elapsed)
        self._last_call = time.time()

    @staticmethod
    def _rate_success():
        """429 gelmedi — rate limit'i kademeli düşür."""
        global _RATE_LIMIT_SEC
        if _RATE_LIMIT_SEC > _RATE_MIN:
            _RATE_LIMIT_SEC = max(_RATE_MIN, _RATE_LIMIT_SEC - _RATE_STEP_DOWN)

    @staticmethod
    def _rate_backoff():
        """429 geldi — rate limit'i kademeli artır."""
        global _RATE_LIMIT_SEC
        _RATE_LIMIT_SEC = min(_RATE_MAX, _RATE_LIMIT_SEC + _RATE_STEP_UP)

    def _send(self, msg: dict) -> Optional[dict]:
        """JSON-RPC mesajı gönder, yanıt döndür. 429 rate limit'te otomatik retry."""
        global _429_COUNT

        # Global circuit breaker — çok fazla 429 aldıysa artık deneme
        if _429_COUNT >= _429_MAX:
            return None

        headers = {
            "Content-Type": "application/json",
            "X-Client-ID": self.client_id,
            "X-API-Key": self.api_key,
        }
        if self.session_id:
            headers["MCP-Session-ID"] = self.session_id

        max_retries = 3
        for attempt in range(max_retries + 1):
            self._rate_wait()
            try:
                resp = requests.post(MCP_URL, headers=headers, json=msg, timeout=_TIMEOUT)
            except Exception as e:
                print(f"    ⚠️ HTTP hatası: {e}")
                return None

            if resp.status_code == 429:
                _429_COUNT += 1
                self._rate_backoff()
                if _429_COUNT >= _429_MAX:
                    print(f"    🛑 Circuit breaker: {_429_COUNT} adet 429 — Matriks API devre dışı")
                    return None
                wait = min(10 * (2 ** attempt), 60)  # 10, 20, 40, 60 saniye
                if attempt < max_retries:
                    print(f"    ⏳ Rate limit (429 #{_429_COUNT}), rate→{_RATE_LIMIT_SEC:.2f}s, {wait}s bekleniyor...")
                    time.sleep(wait)
                    continue
                else:
                    print(f"    ⚠️ Rate limit aşılamadı ({max_retries} retry)")
                    return None
            self._rate_success()
            break

        sid = resp.headers.get("mcp-session-id") or resp.headers.get("MCP-Session-ID")
        if sid:
            self.session_id = sid

        if resp.status_code == 204:
            return None
        try:
            return resp.json()
        except Exception:
            return None

    def _ensure_init(self):
        """MCP initialize + notifications/initialized handshake."""
        if self._initialized:
            return
        if _429_COUNT >= _429_MAX:
            return
        init_msg = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "nox-agent", "version": "1.0"}
            }
        }
        self._send(init_msg)
        # initialized notification
        self._send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        self._initialized = True

    def call_tool(self, tool_name: str, arguments: dict) -> Optional[dict]:
        """MCP tool çağrısı yap, result döndür."""
        self._ensure_init()
        msg = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments,
            }
        }
        resp = self._send(msg)
        if not resp:
            return None

        # JSON-RPC result → content text parse
        result = resp.get("result", {})
        contents = result.get("content", [])
        for c in contents:
            if c.get("type") == "text":
                text = c["text"]
                try:
                    return json.loads(text)
                except (json.JSONDecodeError, TypeError):
                    return {"_raw_text": text}
        return result

    # ──────────────────────────────────────────
    # Tarih yardımcıları
    # ──────────────────────────────────────────

    @staticmethod
    def _date_range(days: int) -> tuple:
        """Son N iş günü için startDate/endDate hesapla.

        Returns: (start_str, end_str) YYYY-MM-DD formatında
        """
        today = datetime.now(_TZ_TR).date()
        # Hafta sonu düzeltmesi: iş günü say
        cal_days = int(days * 1.5) + 5  # tahmini takvim günü (hafta sonları dahil)
        start = today - timedelta(days=cal_days)
        return start.strftime("%Y-%m-%d"), today.strftime("%Y-%m-%d")

    # ──────────────────────────────────────────
    # Public API — tool wrappers
    # ──────────────────────────────────────────

    def get_institutional_flow(self, symbol: str, top: int = 10,
                               start_date: str = None, end_date: str = None,
                               include_investor: bool = False) -> Optional[dict]:
        """Kurumsal akış verisi (topBuyers/topSellers/byVolume/moneyFlow).

        Args:
            symbol: Hisse kodu
            top: En çok N broker
            start_date: Başlangıç tarihi (YYYY-MM-DD), None ise bugün
            end_date: Bitiş tarihi (YYYY-MM-DD), None ise bugün
            include_investor: MKK yatırımcı dağılımı (bireysel/kurumsal %) dahil et

        Returns:
            dict: {topBuyers, topSellers, byVolume, moneyFlow, summary, ...}
                  include_investor=True ise investorCount/investorHistoric alanları eklenir
        """
        args = {
            "symbol": symbol,
            "top": top,
            "includeDetails": True,
            "includeMoneyFlow": True,
        }
        if start_date:
            args["startDate"] = start_date
        if end_date:
            args["endDate"] = end_date
        if include_investor:
            args["includeInvestorCount"] = True
            args["includeInvestorHistoric"] = True
        return self.call_tool("institutionalFlow", args)

    def get_institutional_flow_periods(self, symbol: str, top: int = 10) -> dict:
        """4 periyot için kurumsal akış verisi çek (G/H/A/3A).

        Daily çağrısı MKK yatırımcı dağılımını da içerir (0 extra call).

        Returns:
            dict: {daily: flow, weekly: flow, monthly: flow, quarterly: flow}
                  daily flow'da investorCount/investorHistoric alanları bulunur
        """
        results = {}
        for key, days in _PERIODS:
            try:
                start, end = self._date_range(max(days, 2))
                # Sadece daily'de MKK yatırımcı dağılımını iste (0 extra call)
                inv = (key == "daily")
                flow = self.get_institutional_flow(symbol, top=top,
                                                   start_date=start, end_date=end,
                                                   include_investor=inv)
                if flow and (flow.get("topBuyers") or flow.get("topSellers")):
                    results[key] = flow
            except Exception as e:
                print(f"    ⚠️ {symbol} {key} flow hatası: {e}")
        return results

    def get_settlement(self, symbol: str, top: int = 10,
                        dates: list = None) -> Optional[dict]:
        """Takas analizi — broker pozisyonları + maliyet.

        Args:
            symbol: Hisse kodu
            top: En çok N kurum
            dates: Tarih listesi (YYYY-MM-DD, max 6) — tarihsel karşılaştırma

        Returns:
            dict: {analysis: str} (metin formatında analiz)
        """
        args = {
            "symbol": symbol,
            "mode": "symbol",
            "top": top,
        }
        if dates:
            args["dates"] = dates
        return self.call_tool("settlementAnalysis", args)

    def get_settlement_trend(self, agents: list,
                              trend_type: str = "increasing") -> Optional[dict]:
        """SM broker'ların ardışık pozisyon trendleri.

        Batch başına 1 kez çağrılır — tüm hisseler için trend bilgisi döner.

        Args:
            agents: Broker kodları listesi (örn: ['CIY', 'MLB', 'YATFON'])
            trend_type: 'increasing' (artan) veya 'decreasing' (azalan)

        Returns:
            dict: {analysis: str} (metin formatında trend analizi)
        """
        return self.call_tool("settlementAnalysis", {
            "mode": "trend",
            "agents": agents,
            "trendType": trend_type,
            "top": 50,
        })

    def get_investor_data(self, symbol: str, start_date: str = None,
                          end_date: str = None) -> Optional[list]:
        """MKK yatırımcı dağılımı (bireysel/kurumsal %, yatırımcı sayısı).

        historicalData tool'u ile çekilir — historicalInvestorData alanı.

        Args:
            symbol: Hisse kodu
            start_date: Başlangıç tarihi (YYYY-MM-DD)
            end_date: Bitiş tarihi (YYYY-MM-DD)

        Returns:
            list: [{d, rQ, rP, iQ, iP, c}, ...] veya None
                  d=tarih, rP=bireysel%, iP=kurumsal%, c=yatırımcı sayısı
        """
        if not start_date or not end_date:
            start_date, end_date = self._date_range(10)
        try:
            resp = self.call_tool("historicalData", {
                "symbol": symbol,
                "startDate": start_date,
                "endDate": end_date,
                "includeHistoricalInvestorData": True,
                "rawBars": False,       # bar verisi gönderme (response küçülsün)
            })
            if resp and isinstance(resp, dict):
                inv = resp.get("historicalInvestorData")
                if inv:
                    return inv
        except Exception:
            pass
        return None

    def get_market_price(self, symbol: str) -> Optional[dict]:
        """Güncel fiyat bilgisi.

        Returns:
            dict: {symbol, data: {price, change, changePercent, volume, ...}}
        """
        return self.call_tool("marketPrice", {
            "action": "price",
            "symbol": symbol,
            "includeDetails": True,
        })

    @staticmethod
    def _business_days(n: int) -> list:
        """Son N iş gününün tarih listesini döndür (dünden geriye).

        Bugün henüz kapanmamış olabilir, dünden başlar.
        """
        today = datetime.now(_TZ_TR).date()
        days = []
        d = today - timedelta(days=1)
        while len(days) < n:
            if d.weekday() < 5:  # Pzt-Cum
                days.append(d)
            d -= timedelta(days=1)
        return days

    def get_daily_flow_history(self, symbol: str, days: int = 20,
                                top: int = 10,
                                cached_dates: set = None) -> dict:
        """Son N iş günü için günlük flow verisi çek.

        Her gün ayrı ayrı çekilir — ICE takas_history için.
        cached_dates varsa o günler atlanır (delta çekme).

        Returns:
            {date_str: flow_response} — sadece yeni çekilen günler
        """
        bdays = self._business_days(days)
        cached_dates = cached_dates or set()
        result = {}
        skipped = 0
        for d in bdays:
            ds = d.strftime("%Y-%m-%d")
            if ds in cached_dates:
                skipped += 1
                continue
            try:
                flow = self.get_institutional_flow(
                    symbol, top=top, start_date=ds, end_date=ds)
                if flow and (flow.get("topBuyers") or flow.get("topSellers")):
                    result[ds] = flow
            except Exception:
                pass  # Tatil veya hata — atla
        return result

    # SM broker kodları — settlement trend çağrısı için
    _SM_AGENTS = [
        "CIY", "MLB", "DBY", "GSM", "JPM", "UBS", "MRL", "HSB",
        "NOM", "MOR", "BAR", "BNP", "MAC", "WOD", "VIR", "CTD",
        "YATFON", "EMKFON",
    ]

    def fetch_batch(self, tickers: list, include_settlement: bool = True,
                    include_history: bool = False, history_days: int = 20,
                    history_cache: dict = None) -> dict:
        """Toplu veri çek — her ticker için 4 periyot flow + settlement + price.

        Hisse başına: 4 flow (G/H/A/3A) + 1 settlement(+dates) + 1 price = 6 çağrı.
        include_history=True ise: + eksik günlük daily flow çekilir.
        + batch başına 1 trend çağrısı (SM ardışık birikim).

        Args:
            tickers: Hisse kodu listesi
            include_settlement: Settlement analizi dahil et (maliyet avantajı için)
            include_history: Günlük flow tarihçesi çek (ICE history için)
            history_days: Kaç iş günü geriye git (default 20)
            history_cache: {TICKER: {date_str: flow}} — daha önce çekilmiş tarihçe

        Returns:
            dict: {TICKER: {flows, settlement, price, daily_flows?},
                   _trend: {analysis: str}}
        """
        results = {}
        self._partial_results = results  # referans paylaş — timeout'ta kısmi veri okunabilir
        history_cache = history_cache or {}
        _batch_start = time.time()

        # Batch başına 1 kez: SM ardışık birikim trendleri
        try:
            trend = self.get_settlement_trend(self._SM_AGENTS)
            if trend:
                results["_trend"] = trend
                print("  Matriks: SM trend verisi alındı")
        except Exception as e:
            print(f"  ⚠️ Matriks trend hatası: {e}")

        # Settlement tarih parametresi: son 2 iş günü (hafta sonu düzeltmeli)
        bdays = self._business_days(5)  # son 5 iş günü
        settle_dates = [bdays[-1].strftime("%Y-%m-%d"),  # 1 hafta önceki iş günü
                        bdays[0].strftime("%Y-%m-%d")]   # son iş günü

        total = len(tickers)
        total_api_calls = 0
        total_cache_hits = 0

        for i, ticker in enumerate(tickers, 1):
            try:
                data = {}

                # 4 periyot flow (G/H/A/3A) — her zaman taze
                flows = self.get_institutional_flow_periods(ticker)
                if flows:
                    data["flows"] = flows

                if include_settlement:
                    settlement = self.get_settlement(ticker, dates=settle_dates)
                    if settlement:
                        data["settlement"] = settlement

                price = self.get_market_price(ticker)
                if price:
                    data["price"] = price

                # MKK yatırımcı dağılımı (1 extra call — son 10 iş günü)
                investor = self.get_investor_data(ticker)
                if investor:
                    data["investor"] = investor

                total_api_calls += 7  # flow(4) + settlement(1) + price(1) + investor(1)
                total_investor = sum(1 for t in results.values()
                                     if isinstance(t, dict) and "investor" in t)

                # Günlük flow tarihçesi (ICE history) — delta çekme
                if include_history and history_days > 0 and _429_COUNT < _429_MAX:
                    # Cache'teki tarihleri al — sadece eksikleri çek
                    ticker_cached = history_cache.get(ticker, {})
                    cached_dates = set(ticker_cached.keys())

                    new_flows = self.get_daily_flow_history(
                        ticker, history_days, cached_dates=cached_dates)

                    # Cache + yeni → birleştir
                    merged = dict(ticker_cached)
                    merged.update(new_flows)
                    if merged:
                        data["daily_flows"] = merged
                    total_api_calls += len(new_flows)
                    total_cache_hits += len(cached_dates)

                elif include_history and _429_COUNT >= _429_MAX:
                    if i == 1 or (i > 1 and _429_COUNT == _429_MAX):
                        print(f"  ⚠️ {_429_COUNT} adet 429 — kalan hisseler için history atlanıyor")
                    # Cache'ten olabildiğini yükle
                    ticker_cached = history_cache.get(ticker, {})
                    if ticker_cached:
                        data["daily_flows"] = dict(ticker_cached)
                    include_history = False

                if data:
                    results[ticker] = data
                    if i % 10 == 0 or i == total:
                        df = data.get("daily_flows", {})
                        elapsed_total = time.time() - _batch_start
                        eta = (elapsed_total / i) * (total - i)
                        print(f"  Matriks: {i}/{total} hisse, "
                              f"rate={_RATE_LIMIT_SEC:.2f}s, "
                              f"API={total_api_calls}, MKK={total_investor}, "
                              f"ETA={eta/60:.0f}dk")
            except Exception as e:
                print(f"  ⚠️ Matriks {ticker} hatası: {e}")
                continue

        if total_cache_hits > 0:
            print(f"  Matriks: {total_api_calls} API çağrısı, {total_cache_hits} gün cache'ten")

        return results
