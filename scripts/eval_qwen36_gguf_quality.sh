#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

GGUF_ROOT="${GGUF_ROOT:-artifacts/qwen3.6-27b-gguf}"
EVAL_ROOT="${EVAL_ROOT:-artifacts/qwen3.6-27b-eval}"
SUITE="${SUITE:-benchmarks/qwen36_quality_samples.jsonl}"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-/Users/zlaabsi/Documents/GitHub/llama.cpp}"
NGL="${NGL:-99}"
CTX_SIZE="${CTX_SIZE:-2048}"
TIMEOUT="${TIMEOUT:-900}"
FLASH_ATTN="${FLASH_ATTN:-on}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
SAMPLE_ID="${SAMPLE_ID:-}"
REFERENCE="${REFERENCE:-}"
PROMPT_FORMAT="${PROMPT_FORMAT:-raw}"

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

mkdir -p "$EVAL_ROOT"

for release in "${RELEASES[@]}"; do
  gguf="$GGUF_ROOT/$release/$release.gguf"
  output="$EVAL_ROOT/$release.json"
  if [[ ! -f "$gguf" ]]; then
    echo "skip $release (missing GGUF: $gguf)"
    continue
  fi

  args=(
    python -m opentq.cli eval-gguf
    --gguf "$gguf"
    --output "$output"
    --suite "$SUITE"
    --llama-cpp "$LLAMA_CPP_DIR"
    --ngl "$NGL"
    --ctx-size "$CTX_SIZE"
    --timeout "$TIMEOUT"
    --flash-attn "$FLASH_ATTN"
    --prompt-format "$PROMPT_FORMAT"
  )
  if [[ -n "$MAX_SAMPLES" ]]; then
    args+=(--max-samples "$MAX_SAMPLES")
  fi
  if [[ -n "$SAMPLE_ID" ]]; then
    args+=(--sample-id "$SAMPLE_ID")
  fi
  if [[ -n "$REFERENCE" ]]; then
    args+=(--reference "$REFERENCE")
  fi

  uv run "${args[@]}"
done
