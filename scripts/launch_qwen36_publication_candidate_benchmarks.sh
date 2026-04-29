#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

OUTPUT_ROOT="${1:-artifacts/qwen3.6-27b-publication-candidate-benchmarks}"
STAMP="${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ROOT="$OUTPUT_ROOT/$STAMP"
SESSION_NAME="${SESSION_NAME:-opentq_qwen36_publication_candidate}"
mkdir -p "$RUN_ROOT"

RUN_LOG="$RUN_ROOT/publication-candidate.log"
SESSION_FILE="$RUN_ROOT/screen.session"
SCREEN_PID_FILE="$RUN_ROOT/screen.pid"

if ! command -v screen >/dev/null 2>&1; then
  echo "screen is required for detached publication-candidate benchmark runs"
  exit 1
fi

if screen -list | grep -q "[.]$SESSION_NAME"; then
  echo "screen session already running: $SESSION_NAME"
  echo "attach: screen -r $SESSION_NAME"
  exit 0
fi

screen -dmS "$SESSION_NAME" bash -lc "cd '$ROOT_DIR' && RUN_ROOT='$RUN_ROOT' exec caffeinate -dimsu ./scripts/run_qwen36_publication_candidate_benchmarks.sh > '$RUN_LOG' 2>&1"
echo "$SESSION_NAME" >"$SESSION_FILE"
screen -list >"$RUN_ROOT/screen.list" 2>/dev/null || true
awk -v session="$SESSION_NAME" '$0 ~ "[.]" session { split($1, parts, "."); print parts[1] }' "$RUN_ROOT/screen.list" >"$SCREEN_PID_FILE"

echo "run_root=$RUN_ROOT"
echo "screen_session=$SESSION_NAME"
echo "screen_pid=$(cat "$SCREEN_PID_FILE")"
echo "log=$RUN_LOG"
echo "summary=$RUN_ROOT/RUN_SUMMARY.md"
echo "attach=screen -r $SESSION_NAME"
