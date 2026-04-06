#!/usr/bin/env python3
"""
Matriks MCP Proxy — streamable-http → stdio bridge
Claude Code stdio MCP olarak çalışır, istekleri POST ile Matriks sunucusuna iletir.
"""
import sys
import json
import requests

MCP_URL = "https://mcp.matriks.ai/mcp"
CLIENT_ID = "33667"
API_KEY = "***REDACTED_KEY***"

session_id = None


def send_to_server(msg):
    global session_id
    headers = {
        "Content-Type": "application/json",
        "X-Client-ID": CLIENT_ID,
        "X-API-Key": API_KEY,
    }
    if session_id:
        headers["MCP-Session-ID"] = session_id

    resp = requests.post(MCP_URL, headers=headers, json=msg, timeout=30)

    # Session ID yönetimi
    sid = resp.headers.get("mcp-session-id") or resp.headers.get("MCP-Session-ID")
    if sid:
        session_id = sid

    if resp.status_code == 204:
        return None
    return resp.json()


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue

        result = send_to_server(msg)
        if result is not None:
            sys.stdout.write(json.dumps(result) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
