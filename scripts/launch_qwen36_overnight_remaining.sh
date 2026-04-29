#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

OUTPUT_ROOT="${1:-artifacts/qwen3.6-27b-overnight}"
STAMP="${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ROOT="$OUTPUT_ROOT/$STAMP"
mkdir -p "$RUN_ROOT"

RUN_LOG="$RUN_ROOT/overnight.log"
PID_FILE="$RUN_ROOT/overnight.pid"
CAFFEINATE_PID_FILE="$RUN_ROOT/caffeinate.pid"

RUN_ROOT="$RUN_ROOT" nohup /bin/bash -lc "cd '$ROOT_DIR' && exec ./scripts/run_qwen36_overnight_remaining.sh" >"$RUN_LOG" 2>&1 &
BATCH_PID=$!
nohup caffeinate -dimsu -w "$BATCH_PID" >/dev/null 2>&1 &
echo "$BATCH_PID" >"$PID_FILE"
echo $! >"$CAFFEINATE_PID_FILE"

echo "run_root=$RUN_ROOT"
echo "pid=$(cat "$PID_FILE")"
echo "caffeinate_pid=$(cat "$CAFFEINATE_PID_FILE")"
echo "log=$RUN_LOG"
echo "summary=$RUN_ROOT/RUN_SUMMARY.md"
