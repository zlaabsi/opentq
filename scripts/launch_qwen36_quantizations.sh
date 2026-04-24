#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

OUTPUT_ROOT="${1:-artifacts/qwen3.6-27b}"
mkdir -p "$OUTPUT_ROOT"

RUN_LOG="$OUTPUT_ROOT/overnight.log"
PID_FILE="$OUTPUT_ROOT/overnight.pid"

nohup caffeinate -dimsu "$ROOT_DIR/scripts/run_qwen36_quantizations.sh" "$OUTPUT_ROOT" >"$RUN_LOG" 2>&1 &
echo $! >"$PID_FILE"

echo "pid=$(cat "$PID_FILE")"
echo "log=$RUN_LOG"
