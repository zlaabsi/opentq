#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

SOURCE_ROOT="${1:-artifacts/qwen3.6-27b}"
PACK_ROOT="${2:-artifacts/qwen3.6-27b-packed}"

RELEASES=(
  "Qwen3.6-27B-TQ4_SB4"
  "Qwen3.6-27B-TQ3_SB4"
  "Qwen3.6-27B-TQ4R2"
  "Qwen3.6-27B-TQ4R4"
  "Qwen3.6-27B-TQ4_BAL_V2"
)

for release in "${RELEASES[@]}"; do
  input="$SOURCE_ROOT/$release"
  output="$PACK_ROOT/$release"
  if [[ ! -f "$input/manifest.json" ]]; then
    echo "skip $release: missing manifest"
    continue
  fi
  if [[ -f "$output/opentq-pack.json" ]]; then
    echo "skip $release: packed manifest exists"
    continue
  fi
  echo "pack $release"
  uv run python -m opentq.cli pack-release --input "$input" --output "$output"
  uv run python -m opentq.cli gguf-plan --packed "$output" --output "$output/gguf-plan.json"
done
