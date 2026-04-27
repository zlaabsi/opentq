#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION_NAME="${SESSION_NAME:-opentq_dynamic_release}"
LOG_ROOT="${LOG_ROOT:-$ROOT_DIR/artifacts/logs/dynamic-release}"
mkdir -p "$LOG_ROOT"

if ! command -v screen >/dev/null 2>&1; then
  echo "screen is required for detached long-running dynamic GGUF releases"
  exit 1
fi

if screen -list | grep -q "[.]$SESSION_NAME"; then
  echo "screen session already running: $SESSION_NAME"
  echo "attach: screen -r $SESSION_NAME"
  exit 0
fi

screen -dmS "$SESSION_NAME" bash -lc "cd '$ROOT_DIR' && ./scripts/release_qwen36_dynamic_gguf.sh"

echo "started: $SESSION_NAME"
echo "attach: screen -r $SESSION_NAME"
echo "logs: $LOG_ROOT"
