#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
source "$REPO_DIR/scripts/qwen36_dynamic_common.sh"

LLAMA_CPP="${LLAMA_CPP:-/Users/zlaabsi/Documents/GitHub/llama.cpp}"
BASE_GGUF="${BASE_GGUF:-artifacts/qwen3.6-27b-source/Qwen3.6-27B-BF16.gguf}"
OUT_ROOT="${OUT_ROOT:-artifacts/qwen3.6-27b-dynamic-gguf}"
PROFILES="${PROFILES:-OTQ-DYN-Q4_K_M OTQ-DYN-Q3_K_M OTQ-DYN-Q5_K_M OTQ-DYN-IQ4_NL}"

for profile in $PROFILES; do
  slug="Qwen3.6-27B-${profile}-GGUF"
  out_dir="$OUT_ROOT/$slug"
  target="$out_dir/$(qwen36_dynamic_public_filename "$profile")"
  uv run opentq dynamic-gguf-plan \
    --profile "$profile" \
    --output "$out_dir" \
    --llama-cpp "$LLAMA_CPP" \
    --source-gguf "$BASE_GGUF" \
    --target-gguf "$target"
done
