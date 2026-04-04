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
_RATE_LIMIT_SEC = 0.5
_TIMEOUT = 30
_MSG_ID_COUNTER = 0

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

    # ──────────────────────────────────────────
    # Low-level MCP JSON-RPC
    # ──────────────────────────────────────────

    def _next_id(self) -> int:
        global _MSG_ID_COUNTER
        _MSG_ID_COUNTER += 1
        return _MSG_ID_COUNTER

    def _rate_wait(self):
        elapsed = time.time() - self._last_call
        if elapsed < _RATE_LIMIT_SEC:
            time.sleep(_RATE_LIMIT_SEC - elapsed)
        self._last_call = time.time()

    def _send(self, msg: dict) -> Optional[dict]:
        """JSON-RPC mesajı gönder, yanıt döndür."""
        headers = {
            "Content-Type": "application/json",
            "X-Client-ID": self.client_id,
            "X-API-Key": self.api_key,
        }
        if self.session_id:
            headers["MCP-Session-ID"] = self.session_id

        self._rate_wait()
        resp = requests.post(MCP_URL, headers=headers, json=msg, timeout=_TIMEOUT)

        sid = resp.headers.get("mcp-session-id") or resp.headers.get("MCP-Session-ID")
        if sid:
            self.session_id = sid

        if resp.status_code == 204:
            return None
        return resp.json()

    def _ensure_init(self):
        """MCP initialize + notifications/initialized handshake."""
        if self._initialized:
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
                               start_date: str = None, end_date: str = None) -> Optional[dict]:
        """Kurumsal akış verisi (topBuyers/topSellers/byVolume/moneyFlow).

        Args:
            symbol: Hisse kodu
            top: En çok N broker
            start_date: Başlangıç tarihi (YYYY-MM-DD), None ise bugün
            end_date: Bitiş tarihi (YYYY-MM-DD), None ise bugün

        Returns:
            dict: {topBuyers, topSellers, byVolume, moneyFlow, summary, ...}
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
        return self.call_tool("institutionalFlow", args)

    def get_institutional_flow_periods(self, symbol: str, top: int = 10) -> dict:
        """4 periyot için kurumsal akış verisi çek (G/H/A/3A).

        Returns:
            dict: {daily: flow, weekly: flow, monthly: flow, quarterly: flow}
        """
        results = {}
        for key, days in _PERIODS:
            try:
                # Her zaman tarih aralığı geç — piyasa kapalıyken de dünün verisi gelsin
                start, end = self._date_range(max(days, 2))
                flow = self.get_institutional_flow(symbol, top=top,
                                                   start_date=start, end_date=end)
                # Boş veri kontrolü (topBuyers/topSellers boş olabilir)
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

    # SM broker kodları — settlement trend çağrısı için
    _SM_AGENTS = [
        "CIY", "MLB", "DBY", "GSM", "JPM", "UBS", "MRL", "HSB",
        "NOM", "MOR", "BAR", "BNP", "MAC", "WOD", "VIR", "CTD",
        "YATFON", "EMKFON",
    ]

    def fetch_batch(self, tickers: list, include_settlement: bool = True) -> dict:
        """Toplu veri çek — her ticker için 4 periyot flow + settlement + price.

        Hisse başına: 4 flow (G/H/A/3A) + 1 settlement(+dates) + 1 price = 6 çağrı.
        + batch başına 1 trend çağrısı (SM ardışık birikim).
        50 hisse ≈ 155 saniye (0.5s rate limit).

        Args:
            tickers: Hisse kodu listesi
            include_settlement: Settlement analizi dahil et (maliyet avantajı için)

        Returns:
            dict: {TICKER: {flows, settlement, price},
                   _trend: {analysis: str}}
        """
        results = {}

        # Batch başına 1 kez: SM ardışık birikim trendleri
        try:
            trend = self.get_settlement_trend(self._SM_AGENTS)
            if trend:
                results["_trend"] = trend
                print("  Matriks: SM trend verisi alındı")
        except Exception as e:
            print(f"  ⚠️ Matriks trend hatası: {e}")

        # Settlement tarih parametresi: bugün + 1 hafta önce
        today = datetime.now(_TZ_TR).date()
        week_ago = today - timedelta(days=9)  # ~5 iş günü + hafta sonu
        settle_dates = [week_ago.strftime("%Y-%m-%d"), today.strftime("%Y-%m-%d")]

        total = len(tickers)
        for i, ticker in enumerate(tickers, 1):
            try:
                data = {}

                # 4 periyot flow (G/H/A/3A)
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

                if data:
                    results[ticker] = data
                    if i % 10 == 0 or i == total:
                        print(f"  Matriks: {i}/{total} hisse tamamlandı")
            except Exception as e:
                print(f"  ⚠️ Matriks {ticker} hatası: {e}")
                continue
        return results
