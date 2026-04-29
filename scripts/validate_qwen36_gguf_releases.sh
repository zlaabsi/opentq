#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

GGUF_ROOT="${GGUF_ROOT:-artifacts/qwen3.6-27b-gguf}"
VALIDATION_ROOT="${VALIDATION_ROOT:-artifacts/qwen3.6-27b-validation}"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-../llama.cpp}"
NGL="${NGL:-0}"
CTX_SIZE="${CTX_SIZE:-256}"
N_PREDICT="${N_PREDICT:-4}"
TIMEOUT="${TIMEOUT:-600}"
FLASH_ATTN="${FLASH_ATTN:-off}"
PROMPT="${PROMPT:-Write one short sentence about quantization.}"
BENCH="${BENCH:-0}"

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

mkdir -p "$VALIDATION_ROOT"

for release in "${RELEASES[@]}"; do
  gguf="$GGUF_ROOT/$release/$release.gguf"
  output="$VALIDATION_ROOT/$release.json"
  if [[ ! -f "$gguf" ]]; then
    echo "skip $release (missing GGUF: $gguf)"
    continue
  fi

  args=(
    python -m opentq.cli validate-gguf
    --gguf "$gguf"
    --output "$output"
    --llama-cpp "$LLAMA_CPP_DIR"
    --ngl "$NGL"
    --ctx-size "$CTX_SIZE"
    --n-predict "$N_PREDICT"
    --timeout "$TIMEOUT"
    --flash-attn "$FLASH_ATTN"
    --prompt "$PROMPT"
  )
  if [[ "$BENCH" == "1" ]]; then
    args+=(--bench)
  fi

  uv run "${args[@]}"
done
