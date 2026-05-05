#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

WAIT_FOR_SESSION="${WAIT_FOR_SESSION:-opentq_overnight_native_20260505T002533Z}"
POLL_SECONDS="${POLL_SECONDS:-300}"
MAX_WAIT_SECONDS="${MAX_WAIT_SECONDS:-172800}"
STAMP="${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ROOT="${RUN_ROOT:-artifacts/overnight-research-ui/$STAMP}"
WATCH_LOG="$RUN_ROOT/wait-for-first-run.log"

mkdir -p "$RUN_ROOT"

timestamp() {
  date "+%Y-%m-%d %H:%M:%S"
}

session_running() {
  screen -list 2>/dev/null | grep -q "[.]$WAIT_FOR_SESSION"
}

waited=0
echo "[$(timestamp)] waiting_for=$WAIT_FOR_SESSION" >"$WATCH_LOG"
while session_running; do
  if [[ "$waited" -ge "$MAX_WAIT_SECONDS" ]]; then
    echo "[$(timestamp)] timeout after ${waited}s waiting for $WAIT_FOR_SESSION" >>"$WATCH_LOG"
    exit 3
  fi
  echo "[$(timestamp)] still running: $WAIT_FOR_SESSION waited=${waited}s" >>"$WATCH_LOG"
  sleep "$POLL_SECONDS"
  waited=$((waited + POLL_SECONDS))
done

SESSION_NAME="opentq_overnight_research_ui_$STAMP"
echo "[$(timestamp)] launching=$SESSION_NAME run_root=$RUN_ROOT" >>"$WATCH_LOG"
screen -dmS "$SESSION_NAME" bash -lc "cd '$ROOT_DIR' && RUN_STAMP='$STAMP' RUN_ROOT='$RUN_ROOT' bash ./scripts/run_qwen36_overnight_research_ui.sh > '$RUN_ROOT/screen.log' 2>&1"
echo "[$(timestamp)] launched=$SESSION_NAME" >>"$WATCH_LOG"
