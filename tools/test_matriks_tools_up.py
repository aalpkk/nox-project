"""Hangi Matriks tool'ları çalışıyor? historicalData down mu, geneli mi?"""
from __future__ import annotations
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent.matriks_client import MatriksClient  # noqa: E402


def _probe(client: MatriksClient, tool: str, args: dict) -> str:
    client._ensure_init()
    msg = {'jsonrpc': '2.0', 'id': client._next_id(),
           'method': 'tools/call', 'params': {'name': tool, 'arguments': args}}
    t0 = time.time()
    resp = client._send(msg) or {}
    dt = time.time() - t0
    if 'error' in resp:
        return f"[{dt:5.2f}s] ❌ {tool}: {resp['error'].get('message', '?')}"
    result = resp.get('result', {})
    contents = result.get('content', [])
    if contents:
        txt = ''
        for c in contents:
            if isinstance(c, dict) and c.get('type') == 'text':
                txt = c.get('text', '')
                break
        return f"[{dt:5.2f}s] ✅ {tool}: len={len(txt)}"
    return f"[{dt:5.2f}s] ⚠ {tool}: empty result keys={list(result.keys())}"


def main():
    if not os.environ.get('MATRIKS_API_KEY'):
        return 1
    client = MatriksClient()

    print(_probe(client, 'marketPrice', {'action': 'price', 'symbol': 'GARAN'}))
    print(_probe(client, 'institutionalFlow', {
        'symbol': 'GARAN', 'top': 5,
        'includeDetails': True, 'includeMoneyFlow': True,
    }))
    print(_probe(client, 'settlementAnalysis', {
        'symbol': 'GARAN', 'mode': 'symbol', 'top': 5,
    }))
    print(_probe(client, 'historicalData', {
        'symbol': 'GARAN', 'startDate': '2026-04-01', 'endDate': '2026-04-18',
        'rawBars': True,
    }))


if __name__ == '__main__':
    raise SystemExit(main())
