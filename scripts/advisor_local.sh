#!/bin/zsh
# NOX Advisor — lokal akşam koşusu (headless Claude Code sentezi, API key'siz).
# launchd 21:45 TR hafta içi tetikler (com.nox.advisor.plist).
#
# Akış: en güncel DE artifact'ını indir → advisor'ı claude -p (abonelik) ile koş
# → Telegram + private repoya advisory_latest. CI'nın deterministik raporu zaten
# gitmiş olur; bu koşu zengin (LLM-gerekçeli) raporu üstüne gönderir.
set -euo pipefail

REPO_DIR="${NOX_REPO_DIR:-$HOME/Projects/nox_project}"
LOG_DIR="$REPO_DIR/logs"
mkdir -p "$LOG_DIR"

cd "$REPO_DIR"
# .env (GH_TOKEN, TG_*, NOX_PORTFOLIO_REPO) zaten load_dotenv ile okunur;
# claude CLI PATH'i launchd ortamında eksik olabilir:
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"

{
  echo "=== NOX advisor local $(date '+%F %T') ==="
  # ANTHROPIC_API_KEY'i bilinçli sıfırla → auto modu CLI'ya (claude -p) düşer
  ANTHROPIC_API_KEY= python3 - <<'PYEOF'
from agent.advisor.signals import fetch_latest_de_artifact
from agent.advisor import run_advisor

asof, kind = fetch_latest_de_artifact()
print(f"DE artifact: asof={asof} kind={kind}")
run_advisor(asof=asof, notify=True, llm_mode="cli")
PYEOF
  echo "=== bitti $(date '+%F %T') ==="
} >> "$LOG_DIR/advisor_local.log" 2>&1
