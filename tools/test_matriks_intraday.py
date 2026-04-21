"""
Matriks MCP historicalData — intraday probe.

Amaç: BIST 5dk/15dk/1h intraday bar desteği var mı?
Önce tool schema'sına bakar (hangi param isimleri resmi), sonra olası
interval parametrelerini sistematik dener.

Kullanım:
    export MATRIKS_API_KEY=sk_live_...
    export MATRIKS_CLIENT_ID=33667    # opsiyonel, default 33667
    python tools/test_matriks_intraday.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.matriks_client import MatriksClient  # noqa: E402


SYMBOL = "GARAN"
DAYS_SHORT = 10   # intraday testte 10 iş günü yeterli — bar count'ı görmek için
DAYS_LONG = 365   # historical kapsam testi

# Denenecek param isimleri + değer
PARAM_CANDIDATES = [
    ("interval", "5m"),
    ("interval", "5"),
    ("interval", "5min"),
    ("interval", "M5"),
    ("timeframe", "5m"),
    ("timeframe", "5min"),
    ("period", "5m"),
    ("resolution", "5"),
    ("resolution", "5m"),
    ("granularity", "5m"),
    ("barType", "5m"),
    ("barInterval", "5"),
]


def _raw_call(client: MatriksClient, tool_name: str, arguments: dict) -> dict:
    client._ensure_init()
    msg = {
        'jsonrpc': '2.0',
        'id': client._next_id(),
        'method': 'tools/call',
        'params': {'name': tool_name, 'arguments': arguments},
    }
    return client._send(msg) or {}


def _list_tools(client: MatriksClient) -> list[dict]:
    client._ensure_init()
    msg = {
        'jsonrpc': '2.0',
        'id': client._next_id(),
        'method': 'tools/list',
        'params': {},
    }
    resp = client._send(msg) or {}
    return resp.get('result', {}).get('tools', [])


def _parse_content(raw: dict) -> dict | None:
    result = raw.get('result', {})
    for c in result.get('content', []) or []:
        if isinstance(c, dict) and c.get('type') == 'text':
            try:
                return json.loads(c.get('text', ''))
            except Exception:
                return {'_raw_text': c.get('text', '')[:300]}
    return result


def _probe_bars(resp) -> dict:
    out = {'found_key': None, 'bar_count': 0, 'first': None, 'last': None, 'sample3': []}
    if not isinstance(resp, dict):
        return out
    for k in ('bars', 'rawBars', 'data', 'historicalBars', 'ohlcv', 'priceData', 'historicalPriceData', 'prices', 'candles'):
        v = resp.get(k)
        if isinstance(v, list) and v:
            out['found_key'] = k
            out['bar_count'] = len(v)
            out['first'] = v[0]
            out['last'] = v[-1]
            out['sample3'] = v[:3]
            return out
    return out


def _infer_bar_spacing(bars: list) -> str:
    """İlk birkaç bar'ın timestamp'lerinden interval çıkar."""
    if not bars or len(bars) < 2:
        return 'n/a'
    times = []
    for b in bars[:5]:
        if not isinstance(b, dict):
            continue
        for tk in ('time', 'timestamp', 'date', 'datetime', 't'):
            if tk in b:
                times.append(b[tk])
                break
    if len(times) < 2:
        return f'timestamps bulunamadı, keys={list(bars[0].keys()) if isinstance(bars[0], dict) else type(bars[0]).__name__}'
    return f'timestamps={times[:3]}'


def _date_range(days: int) -> tuple[str, str]:
    end = datetime.now().date()
    start = end - timedelta(days=days)
    return start.isoformat(), end.isoformat()


def step1_list_tools(client):
    print("\n═══ STEP 1: tools/list — historicalData schema ═══")
    tools = _list_tools(client)
    print(f"  Toplam tool: {len(tools)}")
    for t in tools:
        if 'histor' in (t.get('name') or '').lower():
            print(f"\n  Tool: {t.get('name')}")
            print(f"  Description: {(t.get('description') or '')[:300]}")
            schema = t.get('inputSchema', {})
            props = schema.get('properties', {})
            required = schema.get('required', [])
            print(f"  Required params: {required}")
            print(f"  All params ({len(props)}):")
            for pname, pdef in props.items():
                typ = pdef.get('type', '?')
                desc = (pdef.get('description') or '')[:120]
                enum = pdef.get('enum')
                extras = f" enum={enum}" if enum else ''
                print(f"    - {pname} ({typ}){extras}: {desc}")


def step2_baseline(client):
    print("\n═══ STEP 2: BASELINE — parametresiz (daily varsayım) ═══")
    start, end = _date_range(DAYS_SHORT)
    raw = _raw_call(client, 'historicalData', {
        'symbol': SYMBOL, 'startDate': start, 'endDate': end, 'rawBars': True,
    })
    if 'error' in raw:
        print(f"  ❌ JSON-RPC error: {json.dumps(raw['error'])[:300]}")
        return
    resp = _parse_content(raw)
    probe = _probe_bars(resp)
    if probe['found_key']:
        print(f"  ✅ {probe['bar_count']} bar (key='{probe['found_key']}')")
        print(f"  Spacing: {_infer_bar_spacing([probe['first'], probe['last']] if probe['bar_count']>=2 else [])}")
        print(f"  First bar: {json.dumps(probe['first'], default=str)[:250]}")
        print(f"  Last bar:  {json.dumps(probe['last'], default=str)[:250]}")
    else:
        print(f"  ⚠ bar listesi yok — parsed resp keys: {list(resp.keys()) if isinstance(resp, dict) else type(resp).__name__}")
        print(f"  Payload (first 400): {json.dumps(resp, default=str)[:400]}")


def step3_try_intraday(client):
    print("\n═══ STEP 3: INTRADAY PROBES — param ismi + '5m' değeri ═══")
    start, end = _date_range(DAYS_SHORT)
    base = {
        'symbol': SYMBOL, 'startDate': start, 'endDate': end, 'rawBars': True,
    }
    best = None
    for pname, pval in PARAM_CANDIDATES:
        args = dict(base)
        args[pname] = pval
        label = f"{pname}={pval!r}"
        print(f"\n  → {label}")
        t0 = time.time()
        raw = _raw_call(client, 'historicalData', args)
        dt = time.time() - t0
        if 'error' in raw:
            err = raw.get('error', {})
            print(f"    ❌ JSON-RPC error ({dt:.1f}s): code={err.get('code')} msg={str(err.get('message'))[:200]}")
            continue
        resp = _parse_content(raw)
        probe = _probe_bars(resp)
        if probe['found_key']:
            bar_n = probe['bar_count']
            # Intraday heuristic: 10 iş gününde 5m barlar ~= 10 * (480/5) = ~960 bar olmalı
            # Daily: ~10 bar. Aradaki fark net.
            is_intraday = bar_n > 50
            tag = "🎯 INTRADAY" if is_intraday else f"daily-ish (N={bar_n})"
            print(f"    ✅ {dt:.1f}s  N={bar_n}  {tag}")
            print(f"    First: {json.dumps(probe['first'], default=str)[:200]}")
            if bar_n >= 2:
                print(f"    Last:  {json.dumps(probe['last'], default=str)[:200]}")
            if is_intraday and best is None:
                best = (pname, pval, bar_n)
        else:
            print(f"    ⚠ ({dt:.1f}s) bar yok — resp keys: {list(resp.keys()) if isinstance(resp, dict) else '?'}")
            if isinstance(resp, dict) and '_raw_text' in resp:
                print(f"    raw_text: {resp['_raw_text'][:200]}")

    print("\n─── ÖZET ───")
    if best:
        pname, pval, n = best
        print(f"  ✅ INTRADAY DESTEKLİ: {pname}={pval!r} → {n} bar (10 iş günü)")
        print(f"  Sıradaki: 1 yıllık kapsam testi için bu param ile {DAYS_LONG}g dene.")
    else:
        print("  ❌ Hiçbir intraday param kombinasyonu bar üretmedi.")
        print("  Sonuç: Matriks historicalData sadece daily. Intraday için bu API kapısı kapalı.")


def step4_long_history(client, pname: str, pval: str):
    print(f"\n═══ STEP 4: 1 YIL KAPSAM — {pname}={pval!r} ═══")
    start, end = _date_range(DAYS_LONG)
    args = {
        'symbol': SYMBOL, 'startDate': start, 'endDate': end, 'rawBars': True,
        pname: pval,
    }
    t0 = time.time()
    raw = _raw_call(client, 'historicalData', args)
    dt = time.time() - t0
    if 'error' in raw:
        print(f"  ❌ JSON-RPC error ({dt:.1f}s): {json.dumps(raw['error'])[:300]}")
        return
    resp = _parse_content(raw)
    probe = _probe_bars(resp)
    if probe['found_key']:
        n = probe['bar_count']
        print(f"  ✅ {dt:.1f}s  N={n} bar")
        print(f"  First: {json.dumps(probe['first'], default=str)[:250]}")
        print(f"  Last:  {json.dumps(probe['last'], default=str)[:250]}")
        # Tahmini: 250 işgünü × 96 barx5m = 24000, × 32x15m = 8000, × 8x1h = 2000
        exp_5m = 250 * 96
        exp_15m = 250 * 32
        exp_1h = 250 * 8
        print(f"  Beklenen yaklaşık bar sayısı (250 işgünü): 5m≈{exp_5m}, 15m≈{exp_15m}, 1h≈{exp_1h}")
    else:
        print(f"  ⚠ bar yok")


def main():
    if not os.environ.get('MATRIKS_API_KEY'):
        print("❌ MATRIKS_API_KEY env değişkeni set değil")
        print("   Önce: export MATRIKS_API_KEY=sk_live_...")
        return 1

    client = MatriksClient()
    step1_list_tools(client)
    step2_baseline(client)
    step3_try_intraday(client)
    # step4 manuel: step3'te intraday bulunursa terminalden tekrar elle çağır
    print("\n✅ Testler tamamlandı. Intraday bulunduysa step4 için tekrar koştur.")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
