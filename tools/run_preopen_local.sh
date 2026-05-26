#!/usr/bin/env bash
# Local cron wrapper for pre-open snapshot + validate (US).
#
# Reads INTRADAY_SID / INTRADAY_SIGN (+ optional TG_BOT_TOKEN / TG_CHAT_ID)
# from .env, then runs:
#   1. tools/preopen_pm_snapshot_us.py  → output/premarket_snapshot_us_<date>.csv
#   2. tools/preopen_validator_v0.py    → output/preopen_validated_us_<date>.csv
#   3. Telegram digest (if TG_* set)
#
# Schedule via crontab (system TZ = Europe/Istanbul):
#   20 16 * * 1-5  /Users/alpkarakasli/Projects/nox_project/tools/run_preopen_local.sh
#
# Logs to logs/preopen_local_<date>.log (rotated per day, kept ~30d via
# cron's natural overwrite — old files retained until manual cleanup).
#
# Exit codes:
#   0  success (snapshot + validate done)
#   1  missing cookies / candidates / dependencies
#   2  snapshot or validator failed
set -euo pipefail

REPO="/Users/alpkarakasli/Projects/nox_project"
cd "$REPO"

# ---------- env ----------
if [ -f "$REPO/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  . "$REPO/.env"
  set +a
fi

if [ -z "${INTRADAY_SID:-}" ] || [ -z "${INTRADAY_SIGN:-}" ]; then
  echo "!! INTRADAY_SID / INTRADAY_SIGN missing from .env (TV cookies needed)." >&2
  echo "   add them to $REPO/.env then re-run." >&2
  exit 1
fi

# ---------- paths ----------
SNAP_DATE="${SNAP_DATE:-$(TZ=America/New_York date +%F)}"
D_US="${SNAP_DATE//-/_}"

CAND_PATH="${CAND_PATH:-}"
if [ -z "$CAND_PATH" ]; then
  CAND_PATH="$(ls -1 output/tclose_candidates_us_*.csv 2>/dev/null | sort | tail -n 1 || true)"
fi
if [ -z "$CAND_PATH" ] || [ ! -f "$CAND_PATH" ]; then
  echo "!! candidates file missing — set CAND_PATH or place output/tclose_candidates_us_*.csv" >&2
  exit 1
fi

SNAP_OUT="output/premarket_snapshot_us_${D_US}.csv"
VAL_OUT="output/preopen_validated_us_${D_US}.csv"

mkdir -p logs
LOG="logs/preopen_local_${D_US}.log"

PY="${PY:-python3}"

# ---------- run ----------
{
  echo "==== preopen-local $(date -Is) ===="
  echo "[cfg] snap_date=$SNAP_DATE  cand=$CAND_PATH"
  echo "[cfg] snap_out=$SNAP_OUT  val_out=$VAL_OUT"
  echo

  echo "[1/2] snapshot ..."
  if ! "$PY" tools/preopen_pm_snapshot_us.py \
        --candidates "$CAND_PATH" \
        --output     "$SNAP_OUT" \
        --snapshot-date "$SNAP_DATE" \
        --pm-start "${PM_START:-04:00}" \
        --pm-end   "${PM_END:-09:30}"; then
    echo "!! snapshot failed" >&2
    exit 2
  fi

  echo
  echo "[2/2] validate ..."
  if ! "$PY" tools/preopen_validator_v0.py \
        --candidates "$CAND_PATH" \
        --premarket  "$SNAP_OUT" \
        --output     "$VAL_OUT"; then
    echo "!! validator failed" >&2
    exit 2
  fi

  # Digest
  SUMMARY=$("$PY" - <<PY
import pandas as pd
df = pd.read_csv("$VAL_OUT")
col = "preopen_status" if "preopen_status" in df.columns else (
      "status" if "status" in df.columns else None)
if col is None:
    print(f"NO_STATUS_COL N={len(df)}")
else:
    c = df[col].value_counts().to_dict()
    print(f"PASS={c.get('PASS_PREOPEN',0)} "
          f"WATCH={c.get('WATCH_OPEN',0)} "
          f"REJECT={c.get('REJECT_PREOPEN',0)} (N={len(df)})")
PY
)
  echo
  echo "[digest] $SUMMARY"

  # Telegram (optional)
  if [ -n "${TG_BOT_TOKEN:-}" ] && [ -n "${TG_CHAT_ID:-}" ]; then
    MSG="preopen-validate (local) $SNAP_DATE
candidates: $CAND_PATH
$SUMMARY
out: $VAL_OUT"
    if "$PY" - <<PY
import os, urllib.request, urllib.parse, sys
tok = "$TG_BOT_TOKEN"
chat = "$TG_CHAT_ID"
msg = """$MSG"""
url = f"https://api.telegram.org/bot{tok}/sendMessage"
data = urllib.parse.urlencode({"chat_id": chat, "text": msg,
                               "disable_web_page_preview": "true"}).encode()
req = urllib.request.Request(url, data=data)
try:
    with urllib.request.urlopen(req, timeout=15) as r:
        print("[tg]", r.status)
except Exception as e:
    print("[tg] FAIL", e, file=sys.stderr)
    sys.exit(1)
PY
    then
      echo "[tg] sent"
    else
      echo "[tg] send failed (continuing)"
    fi
  else
    echo "[tg] TG_BOT_TOKEN/CHAT_ID not set — skip"
  fi

  echo
  echo "==== done $(date -Is) ===="
} 2>&1 | tee -a "$LOG"
