#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

HF_USER="${HF_USER:-${1:-}}"
PACK_ROOT="${2:-artifacts/qwen3.6-27b-packed}"
STAGE_ROOT="${3:-artifacts/qwen3.6-27b-hf}"

if [[ -z "$HF_USER" ]]; then
  echo "usage: HF_USER=<user-or-org> $0 [hf-user] [pack-root] [stage-root]" >&2
  exit 2
fi

for manifest in "$PACK_ROOT"/Qwen3.6-27B-*/opentq-pack.json; do
  [[ -f "$manifest" ]] || continue
  release="$(basename "$(dirname "$manifest")")"
  repo_id="$HF_USER/$release"
  output="$STAGE_ROOT/$release"
  echo "stage $repo_id"
  uv run python -m opentq.cli prepare-hf \
    --packed "$(dirname "$manifest")" \
    --output "$output" \
    --repo-id "$repo_id" \
    --link-mode hardlink
done
