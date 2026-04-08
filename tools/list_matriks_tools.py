"""Matriks MCP tool listesini çek ve yazdır."""
import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent.matriks_client import MatriksClient

client = MatriksClient()
client._ensure_init()

msg = {
    "jsonrpc": "2.0",
    "id": 999,
    "method": "tools/list",
    "params": {}
}
resp = client._send(msg)
if resp and "result" in resp:
    tools = resp["result"].get("tools", [])
    print(f"\n=== {len(tools)} Matriks MCP Tool ===\n")
    for t in tools:
        name = t.get("name", "")
        desc = t.get("description", "")
        schema = t.get("inputSchema", {})
        props = schema.get("properties", {})
        required = schema.get("required", [])
        params = []
        for k, v in props.items():
            req = "*" if k in required else ""
            params.append(f"{k}{req}({v.get('type','?')})")
        print(f"  {name}: {desc[:120]}")
        if params:
            print(f"    params: {', '.join(params)}")
        print()
else:
    print("Hata:", json.dumps(resp, indent=2) if resp else "None")
