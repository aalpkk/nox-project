"""
Matriks MCP historicalData rawBars testi.

Amaç: BIST EOD OHLCV için Matriks historicalData tool'unu bulk-history
kaynağı olarak kullanabilir miyiz?

Test senaryoları:
    1. GARAN 30 gün (sanity)
    2. GARAN 1 yıl
    3. GARAN 5 yıl (asıl hedef)

Her senaryoda:
    - Response süresi
    - Bar sayısı
    - Kolon anahtarları
    - İlk/son tarih
    - OHLCV alanları var mı
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.matriks_client import MatriksClient  # noqa: E402


def _date_range(days: int) -> tuple[str, str]:
    end = datetime.now().date()
    start = end - timedelta(days=days)
    return start.isoformat(), end.isoformat()


def _probe_bars(resp: dict) -> dict:
    """Response içinde bar listesini bul ve özetle."""
    out = {'found_key': None, 'bar_count': 0, 'keys': [], 'first': None, 'last': None}
    if not isinstance(resp, dict):
        return out
    out['keys'] = list(resp.keys())
    candidates = ['bars', 'rawBars', 'data', 'historicalBars', 'ohlcv',
                  'priceData', 'historicalPriceData', 'prices']
    for k in candidates:
        if k in resp and isinstance(resp[k], list) and resp[k]:
            out['found_key'] = k
            out['bar_count'] = len(resp[k])
            out['first'] = resp[k][0]
            out['last'] = resp[k][-1]
            return out
    # Nested arama (content -> text -> json)
    if 'content' in resp:
        for c in resp.get('content', []):
            if isinstance(c, dict) and c.get('type') == 'text':
                try:
                    inner = json.loads(c.get('text', ''))
                    if isinstance(inner, dict):
                        sub = _probe_bars(inner)
                        if sub['bar_count']:
                            return sub
                except Exception:
                    pass
    return out


def test_scenario(client: MatriksClient, symbol: str, days: int, label: str):
    start, end = _date_range(days)
    print(f"\n═══ {label}: {symbol} ({start} → {end}, ~{days}g) ═══")

    t0 = time.time()
    try:
        resp = client.call_tool('historicalData', {
            'symbol': symbol,
            'startDate': start,
            'endDate': end,
            'rawBars': True,
            'includeHistoricalInvestorData': False,
        })
    except Exception as e:
        print(f"  ❌ EXCEPTION: {type(e).__name__}: {e}")
        return
    dt = time.time() - t0

    if resp is None:
        print(f"  ❌ NO RESPONSE ({dt:.2f}s)")
        return

    if not isinstance(resp, dict):
        print(f"  ⚠ Response type: {type(resp).__name__}")
        print(f"  Content: {str(resp)[:300]}")
        return

    probe = _probe_bars(resp)
    print(f"  ⏱ {dt:.2f}s")
    print(f"  Top-level keys: {probe['keys']}")
    if probe['found_key']:
        print(f"  ✅ Bars bulundu '{probe['found_key']}' → {probe['bar_count']} bar")
        print(f"  First: {json.dumps(probe['first'], ensure_ascii=False)[:300]}")
        print(f"  Last:  {json.dumps(probe['last'], ensure_ascii=False)[:300]}")
    else:
        print(f"  ⚠ Bar listesi bulunamadı")
        # Raw payload özet
        s = json.dumps(resp, ensure_ascii=False)
        print(f"  Raw (first 600): {s[:600]}")


def main():
    if not os.environ.get('MATRIKS_API_KEY'):
        print("❌ MATRIKS_API_KEY yok")
        return 1

    client = MatriksClient()

    # Test 1: 30 gün
    test_scenario(client, 'GARAN', 35, '30 GÜN')
    # Test 2: 1 yıl
    test_scenario(client, 'GARAN', 365, '1 YIL')
    # Test 3: 5 yıl (asıl hedef)
    test_scenario(client, 'GARAN', 1825, '5 YIL')
    # Test 4: Sektör endeksi (XBANK)
    test_scenario(client, 'XBANK', 1825, 'SEKTÖR ENDEKSİ 5y')

    print("\n✅ Testler tamamlandı")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
