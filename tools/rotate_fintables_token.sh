#!/usr/bin/env bash
# Rotate the Fintables MCP token GitHub secret from the local Claude Code keychain.
#
# Fintables MCP uses short-lived OAuth tokens (~7 days, NO refresh token), so the
# CI secret expires regularly. After re-authenticating on the Mac via `/mcp`, run
# this to push the fresh token into the FINTABLES_MCP_TOKEN GitHub secret.
#
# Usage:
#   tools/rotate_fintables_token.sh          # rotate + run CI probe to confirm
#   tools/rotate_fintables_token.sh --no-probe   # rotate only
set -euo pipefail

SERVICE="Claude Code-credentials"   # macOS keychain item holding mcpOAuth tokens
SECRET="FINTABLES_MCP_TOKEN"

# Pull token + expiry from keychain (entry key looks like "fintables|<hash>";
# matched by prefix so a re-auth that changes the hash still resolves).
read -r TOKEN EXP < <(
  security find-generic-password -s "$SERVICE" -w 2>/dev/null | python3 -c '
import json, sys
d = json.load(sys.stdin)
m = d.get("mcpOAuth", {})
k = next((k for k in m if k.startswith("fintables")), None)
if not k:
    sys.exit("no fintables entry in keychain mcpOAuth — run /mcp on the Mac first")
e = m[k]
print(e["accessToken"], e.get("expiresAt", 0))
'
)

if [[ -z "${TOKEN:-}" ]]; then
  echo "FAIL: could not read fintables token from keychain" >&2
  exit 1
fi

# Warn if the token we are about to push is already expired / near expiry.
python3 - "$EXP" <<'PYEOF'
import sys, time
exp = int(sys.argv[1])
if not exp:
    print("note: token has no expiry field"); raise SystemExit
secs = exp/1000 if exp > 1e12 else exp
left = secs - time.time()
when = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(secs))
if left <= 0:
    print(f"WARNING: local token already EXPIRED ({when}) — re-auth via /mcp before rotating")
else:
    print(f"local token valid until {when} (~{left/86400:.1f} days left)")
PYEOF

printf '%s' "$TOKEN" | gh secret set "$SECRET"
echo "rotated GitHub secret: $SECRET"

if [[ "${1:-}" == "--no-probe" ]]; then
  exit 0
fi

echo "running CI probe to confirm..."
gh workflow run fintables-token-probe.yml
sleep 25
RUN=$(gh run list --workflow fintables-token-probe.yml --limit 1 --json databaseId -q '.[0].databaseId')
gh run view "$RUN" --log 2>/dev/null | grep -E "status:|OK:|FAIL" || true
gh run view "$RUN" --json status,conclusion -q '.status+" / "+(.conclusion//"running")'
