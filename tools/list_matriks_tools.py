"""Matriks MCP tool listesini çek ve yazdır.

Ayrıca MKK/investor ile ilgili olası tool isimlerini dener.
"""
import json, sys, os, time, requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MCP_URL = "https://mcp.matriks.ai/mcp"
API_KEY = os.environ.get("MATRIKS_API_KEY", "")
CLIENT_ID = os.environ.get("MATRIKS_CLIENT_ID", "33667")

headers = {
    "Content-Type": "application/json",
    "X-Client-ID": CLIENT_ID,
    "X-API-Key": API_KEY,
}
session_id = None
msg_id = 0


def send(msg):
    global session_id, msg_id
    msg_id += 1
    msg["id"] = msg_id
    resp = requests.post(MCP_URL, headers=headers, json=msg, timeout=60)
    sid = resp.headers.get("mcp-session-id") or resp.headers.get("MCP-Session-ID")
    if sid:
        session_id = sid
        headers["MCP-Session-ID"] = sid
    if resp.status_code == 204:
        return None
    return resp.json()


# 1. Initialize
print("=== Initialize ===")
resp = send({"jsonrpc": "2.0", "method": "initialize",
             "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                        "clientInfo": {"name": "nox-list", "version": "1.0"}}})
print(f"  Session: {session_id}")
send({"jsonrpc": "2.0", "method": "notifications/initialized"})
time.sleep(1)

# 2. tools/list
print("\n=== Tools List ===")
resp = send({"jsonrpc": "2.0", "method": "tools/list", "params": {}})
if resp and "result" in resp:
    tools = resp["result"].get("tools", [])
    print(f"\n  Toplam {len(tools)} tool:\n")
    for t in tools:
        name = t.get("name", "")
        desc = t.get("description", "")[:120]
        schema = t.get("inputSchema", {})
        props = list(schema.get("properties", {}).keys())
        print(f"  {name}: {desc}")
        if props:
            print(f"    params: {', '.join(props)}")
        print()
else:
    print("  tools/list hata:", json.dumps(resp, indent=2, ensure_ascii=False) if resp else "None")

    # 3. Fallback: olası MKK tool isimlerini dene
    print("\n=== MKK Tool Denemeleri ===")
    candidates = [
        "investorDistribution", "mkkDistribution", "shareholderStructure",
        "investorProfile", "mkkData", "investorData", "stockOwnership",
        "ownershipDistribution", "investorTypes", "shareholderDistribution",
    ]
    for name in candidates:
        time.sleep(0.5)
        try:
            resp = send({"jsonrpc": "2.0", "method": "tools/call",
                         "params": {"name": name, "arguments": {"symbol": "GARAN"}}})
            err = resp.get("error", {}) if resp else {}
            result = resp.get("result", {}) if resp else {}
            if err:
                code = err.get("code", "")
                msg_text = err.get("message", "")[:80]
                print(f"  {name}: ERR {code} — {msg_text}")
            elif result:
                print(f"  {name}: ✅ FOUND! {json.dumps(result, ensure_ascii=False)[:200]}")
            else:
                print(f"  {name}: boş yanıt")
        except Exception as e:
            print(f"  {name}: exception {e}")
