#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SOURCE="${BASE_GGUF:-artifacts/qwen3.6-27b-source/Qwen3.6-27B-BF16.gguf}"
OUT_ROOT="${OUT_ROOT:-artifacts/qwen3.6-27b-dynamic-gguf}"
VALIDATION_ROOT="${VALIDATION_ROOT:-artifacts/qwen3.6-27b-dynamic-validation}"
LOG_ROOT="${LOG_ROOT:-artifacts/logs/dynamic-gguf}"

echo "disk:"
df -h "$ROOT_DIR" | sed -n '1,2p'
echo

echo "screen:"
screen -ls 2>/dev/null || true
echo

echo "source:"
if [[ -s "$SOURCE" ]]; then
  ls -lh "$SOURCE"
else
  echo "missing: $SOURCE"
fi
echo

echo "dynamic artifacts:"
find "$OUT_ROOT" -maxdepth 2 -type f \( -name '*.gguf' -o -name 'dry-run.log' -o -name 'plan.json' \) 2>/dev/null \
  -exec ls -lh {} + | awk '{print $5, $9}' | sort || true
echo

echo "validation:"
find "$VALIDATION_ROOT" -maxdepth 1 -type f -name '*.json' 2>/dev/null \
  -exec ls -lh {} + | awk '{print $5, $9}' | sort || true
echo

echo "recent logs:"
find "$LOG_ROOT" -maxdepth 1 -type f -name '*.log' 2>/dev/null -print0 \
  | xargs -0 ls -lt 2>/dev/null \
  | sed -n '1,8p' || true
echo

latest_log="$(find "$LOG_ROOT" -maxdepth 1 -type f -name '*.log' 2>/dev/null -print0 | xargs -0 ls -t 2>/dev/null | head -1 || true)"
if [[ -n "$latest_log" ]]; then
  echo "tail: $latest_log"
  tail -40 "$latest_log"
fi
