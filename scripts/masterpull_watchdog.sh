#!/bin/zsh
# NOX master-data-pull WATCHDOG — GitHub Actions cron'u güvenilmez (scheduled
# tetikleri geciktirir/düşürür). Bu watchdog hafta içi 18:50 TR'de çalışır:
# master-data-pull BUGÜN koşmadıysa (cron kaçırdıysa) close-mode tetikler.
# Koştuysa hiçbir şey yapmaz (çift koşu YOK). DE→advisor zinciri otomatik gelir.
#
# Kurulum:
#   cp scripts/com.nox.masterpull.plist ~/Library/LaunchAgents/
#   launchctl load ~/Library/LaunchAgents/com.nox.masterpull.plist
set -uo pipefail
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"
REPO="aalpkk/nox-project"
LOG="$HOME/Projects/nox_project/logs/masterpull_watchdog.log"
mkdir -p "$(dirname "$LOG")"
today=$(date +%F)
# bugün master-data-pull koştu mu (schedule veya dispatch)
n=$(gh run list --repo "$REPO" --workflow=master-data-pull.yml \
      --created="$today" --json databaseId --jq 'length' 2>/dev/null || echo "ERR")
ts=$(date '+%F %T')
if [ "$n" = "ERR" ]; then
  echo "$ts  gh hata — kontrol edilemedi" >> "$LOG"; exit 0
fi
if [ "$n" = "0" ]; then
  echo "$ts  bugün koşu YOK → close tetikleniyor (cron kaçırmış)" >> "$LOG"
  gh workflow run master-data-pull.yml --repo "$REPO" -f mode=close >> "$LOG" 2>&1
else
  echo "$ts  bugün $n koşu var → watchdog no-op" >> "$LOG"
fi
