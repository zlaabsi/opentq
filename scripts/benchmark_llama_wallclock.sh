#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 MODEL.gguf"
  exit 2
fi

MODEL="$1"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-../llama.cpp}"
PROMPT_TOKENS="${PROMPT_TOKENS:-8192}"
GEN_TOKENS="${GEN_TOKENS:-128}"
NGL="${NGL:-99}"
FLASH_ATTN="${FLASH_ATTN:-1}"

BENCH="$LLAMA_CPP_DIR/build/bin/llama-bench"
if [[ ! -x "$BENCH" ]]; then
  echo "missing llama-bench: $BENCH"
  echo "build llama.cpp first: cmake -B build -DGGML_METAL=ON && cmake --build build -j"
  exit 1
fi

echo "model=$MODEL"
echo "prompt_tokens=$PROMPT_TOKENS"
echo "gen_tokens=$GEN_TOKENS"
echo "ngl=$NGL"
echo "flash_attn=$FLASH_ATTN"

start="$(date +%s)"
"$BENCH" \
  -m "$MODEL" \
  -ngl "$NGL" \
  -fa "$FLASH_ATTN" \
  -p "$PROMPT_TOKENS" \
  -n "$GEN_TOKENS"
end="$(date +%s)"

echo "wall_clock_seconds=$((end - start))"
