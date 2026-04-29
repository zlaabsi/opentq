#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

OUTPUT_ROOT="${1:-artifacts/qwen3.6-27b}"
mkdir -p "$OUTPUT_ROOT"

RUN_LOG="$OUTPUT_ROOT/quantization.log"
PID_FILE="$OUTPUT_ROOT/quantization.pid"
CAFFEINATE_PID_FILE="$OUTPUT_ROOT/caffeinate.pid"

nohup /bin/bash -lc "cd '$ROOT_DIR' && exec ./scripts/run_qwen36_quantizations.sh '$OUTPUT_ROOT'" >"$RUN_LOG" 2>&1 &
BATCH_PID=$!
nohup caffeinate -dimsu -w "$BATCH_PID" >/dev/null 2>&1 &
echo "$BATCH_PID" >"$PID_FILE"
echo $! >"$CAFFEINATE_PID_FILE"

echo "pid=$(cat "$PID_FILE")"
echo "caffeinate_pid=$(cat "$CAFFEINATE_PID_FILE")"
echo "log=$RUN_LOG"
