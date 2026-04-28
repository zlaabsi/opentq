#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-/Users/zlaabsi/Documents/GitHub/llama.cpp}"
OUTPUT_ROOT="${OUTPUT_ROOT:-artifacts/qwen3.6-27b-otq-eval}"
SUITE="${SUITE:-benchmarks/qwen36_release_extended_samples.jsonl}"
CTX_SIZE="${CTX_SIZE:-8192}"
NGL="${NGL:-99}"
FLASH_ATTN="${FLASH_ATTN:-on}"
TIMEOUT="${TIMEOUT:-1800}"
PROMPT_FORMAT="${PROMPT_FORMAT:-qwen3-no-think}"
MAX_SAMPLES="${MAX_SAMPLES:-}"

model_path_for_label() {
  case "$1" in
    OTQ-DYN-Q3_K_M)
      printf '%s\n' "artifacts/hf-gguf-canonical/Qwen3.6-27B-OTQ-GGUF/Qwen3.6-27B-OTQ-DYN-Q3_K_M.gguf"
      ;;
    OTQ-DYN-Q4_K_M)
      printf '%s\n' "artifacts/hf-gguf-canonical/Qwen3.6-27B-OTQ-GGUF/Qwen3.6-27B-OTQ-DYN-Q4_K_M.gguf"
      ;;
    *)
      return 1
      ;;
  esac
}

mkdir -p "$OUTPUT_ROOT"

for label in OTQ-DYN-Q3_K_M OTQ-DYN-Q4_K_M; do
  gguf="$(model_path_for_label "$label")"
  output="$OUTPUT_ROOT/$label.json"
  if [[ ! -f "$gguf" ]]; then
    echo "skip $label: missing $gguf"
    continue
  fi

  args=(
    uv run opentq eval-gguf
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

  echo "eval $label -> $output"
  "${args[@]}"
done

uv run python scripts/build_qwen36_release_report.py \
  --repo artifacts/hf-gguf-canonical/Qwen3.6-27B-OTQ-GGUF \
  --quant-eval-root "$OUTPUT_ROOT"
