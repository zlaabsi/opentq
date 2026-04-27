#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

HF_USER="${HF_USER:-zlaabsi}"
GGUF_ROOT="${GGUF_ROOT:-artifacts/qwen3.6-27b-gguf}"
HF_ROOT="${HF_ROOT:-artifacts/qwen3.6-27b-hf-gguf}"
RUNTIME_REPO="${RUNTIME_REPO:-https://github.com/zlaabsi/llama.cpp-opentq}"
LINK_MODE="${LINK_MODE:-hardlink}"

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

mkdir -p "$HF_ROOT"

for release in "${RELEASES[@]}"; do
  gguf="$GGUF_ROOT/$release/$release.gguf"
  output="$HF_ROOT/$release"
  repo_id="$HF_USER/$release-GGUF"

  if [[ ! -f "$gguf" ]]; then
    echo "skip $release (missing GGUF: $gguf)"
    continue
  fi

  uv run python -m opentq.cli prepare-hf-gguf \
    --gguf "$gguf" \
    --output "$output" \
    --repo-id "$repo_id" \
    --runtime-repo "$RUNTIME_REPO" \
    --link-mode "$LINK_MODE"
done
