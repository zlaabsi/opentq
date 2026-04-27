#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

GGUF_ROOT="${GGUF_ROOT:-artifacts/qwen3.6-27b-gguf}"
LOG_ROOT="${LOG_ROOT:-artifacts/logs/gguf}"
SESSION_NAME="${SESSION_NAME:-opentq_gguf_exports}"

echo "== screen =="
screen_output="$(screen -list || true)"
if ! grep "$SESSION_NAME" <<<"$screen_output"; then
  echo "not running"
fi

echo
echo "== disk =="
df -h "$ROOT_DIR"

echo
echo "== gguf files =="
find "$GGUF_ROOT" -maxdepth 3 -type f -name "*.gguf" -exec ls -lh {} \; 2>/dev/null || true

echo
echo "== latest logs =="
for log in "$LOG_ROOT"/*.log; do
  [[ -f "$log" ]] || continue
  echo "-- $log"
  tail -n 8 "$log"
done
