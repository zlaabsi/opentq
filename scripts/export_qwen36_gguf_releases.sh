#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

PACKED_ROOT="${PACKED_ROOT:-artifacts/qwen3.6-27b-packed}"
GGUF_ROOT="${GGUF_ROOT:-artifacts/qwen3.6-27b-gguf}"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-/Users/zlaabsi/Documents/GitHub/llama.cpp}"
LOG_ROOT="${LOG_ROOT:-artifacts/logs/gguf}"
FORCE="${FORCE:-0}"

if [[ -n "${OPENTQ_RELEASES:-}" ]]; then
  read -r -a RELEASES <<<"$OPENTQ_RELEASES"
else
  RELEASES=(
    "Qwen3.6-27B-TQ4_BAL_V2"
    "Qwen3.6-27B-TQ3_SB4"
    "Qwen3.6-27B-TQ4_SB4"
    "Qwen3.6-27B-TQ4R2"
    "Qwen3.6-27B-TQ4R4"
  )
fi

mkdir -p "$GGUF_ROOT" "$LOG_ROOT"

timestamp() {
  date "+%Y-%m-%d %H:%M:%S"
}

run_export() {
  local release="$1"
  local packed="$PACKED_ROOT/$release"
  local output_dir="$GGUF_ROOT/$release"
  local output="$output_dir/$release.gguf"
  local log="$LOG_ROOT/$release.log"

  if [[ ! -f "$packed/opentq-pack.json" ]]; then
    echo "[$(timestamp)] missing packed release: $packed" | tee -a "$log"
    return 1
  fi

  mkdir -p "$output_dir"
  if [[ "$FORCE" != "1" && -s "$output" ]]; then
    echo "[$(timestamp)] skip $release (GGUF exists: $output)" | tee -a "$log"
    return 0
  fi

  echo "[$(timestamp)] start GGUF export $release" | tee -a "$log"
  local cmd=(
    uv run python -m opentq.cli export-gguf
    --packed "$packed"
    --output "$output"
    --llama-cpp "$LLAMA_CPP_DIR"
  )
  if [[ -n "${MAX_TENSORS:-}" ]]; then
    cmd+=(--max-tensors "$MAX_TENSORS")
  fi

  if command -v caffeinate >/dev/null 2>&1; then
    PYTHONUNBUFFERED=1 caffeinate -dimsu "${cmd[@]}" >>"$log" 2>&1
  else
    PYTHONUNBUFFERED=1 "${cmd[@]}" >>"$log" 2>&1
  fi
  echo "[$(timestamp)] done GGUF export $release" | tee -a "$log"
}

for release in "${RELEASES[@]}"; do
  run_export "$release"
done
