#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SESSION="${SESSION:-opentq_qwen36_bf16_paired_quality}"
STAMP="${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ROOT="${RUN_ROOT:-artifacts/qwen3.6-27b-bf16-paired-quality/$STAMP}"
LOG_PATH="$RUN_ROOT/bf16-paired-quality.log"

mkdir -p "$RUN_ROOT"

if screen -list | grep -q "[.]$SESSION"; then
  echo "screen session already exists: $SESSION"
  echo "attach: screen -r $SESSION"
  exit 1
fi

{
  echo "# Qwen3.6 BF16 Paired Quality Launcher"
  echo
  echo "- Started: $(date '+%Y-%m-%d %H:%M:%S')"
  echo "- Screen: \`$SESSION\`"
  echo "- Run root: \`$RUN_ROOT\`"
  echo "- Log: \`$LOG_PATH\`"
} >"$RUN_ROOT/LAUNCH.md"

screen -dmS "$SESSION" bash -lc "cd '$ROOT_DIR' && RUN_ROOT='$RUN_ROOT' exec caffeinate -dimsu ./scripts/run_qwen36_bf16_paired_quality.sh > '$LOG_PATH' 2>&1"
screen -list >"$RUN_ROOT/screen.list" 2>&1 || true
printf "%s\n" "$SESSION" >"$RUN_ROOT/screen.session"

echo "launched screen: $SESSION"
echo "run root: $RUN_ROOT"
echo "log: $LOG_PATH"
echo "attach: screen -r $SESSION"
