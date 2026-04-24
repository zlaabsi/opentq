#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

OUTPUT_ROOT="${1:-artifacts/qwen3.6-27b}"
mkdir -p "$OUTPUT_ROOT/logs"

RELEASES=(
  "Qwen3.6-27B-TQ4_SB4"
  "Qwen3.6-27B-TQ3_SB4"
  "Qwen3.6-27B-TQ4R2"
  "Qwen3.6-27B-TQ4R4"
  "Qwen3.6-27B-TQ4_BAL_V2"
)

timestamp() {
  date "+%Y-%m-%d %H:%M:%S"
}

for release in "${RELEASES[@]}"; do
  release_dir="$OUTPUT_ROOT/$release"
  log_path="$OUTPUT_ROOT/logs/$release.log"

  if [[ -f "$release_dir/manifest.json" ]]; then
    echo "[$(timestamp)] skip $release (manifest exists)"
    continue
  fi

  echo "[$(timestamp)] start $release" | tee -a "$log_path"
  PYTHONUNBUFFERED=1 uv run python -m opentq.cli quantize-release \
    --recipe qwen3.6-27b \
    --release "$release" \
    --output "$release_dir" \
    >>"$log_path" 2>&1
  echo "[$(timestamp)] done $release" | tee -a "$log_path"
done
