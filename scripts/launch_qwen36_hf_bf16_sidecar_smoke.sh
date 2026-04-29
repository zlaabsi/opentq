#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODEL_ID="${MODEL_ID:-Qwen/Qwen3.6-27B}"
FLAVOR="${FLAVOR:-a100-large}"
TIMEOUT="${TIMEOUT:-45m}"
BENCHMARKS="${BENCHMARKS:-mmlu_pro,gpqa,aime}"
NO_THINK_SAMPLES="${NO_THINK_SAMPLES:-4}"
THINKING_SAMPLES="${THINKING_SAMPLES:-2}"
NO_THINK_MAX_TOKENS="${NO_THINK_MAX_TOKENS:-512}"
THINKING_MAX_TOKENS="${THINKING_MAX_TOKENS:-2048}"
UPLOAD_REPO="${UPLOAD_REPO:-zlaabsi/opentq-qwen36-bf16-sidecar}"
OPENTQ_REF="${OPENTQ_REF:-$(git rev-parse HEAD)}"

if [ "$MODEL_ID" != "Qwen/Qwen3.6-27B" ]; then
  echo "Refusing wrong model id: $MODEL_ID" >&2
  exit 1
fi

hf jobs uv run \
  --detach \
  --flavor "$FLAVOR" \
  --timeout "$TIMEOUT" \
  --secrets HF_TOKEN \
  --env "OPENTQ_REF=$OPENTQ_REF" \
  --env HF_HUB_ENABLE_HF_TRANSFER=1 \
  scripts/run_qwen36_hf_bf16_sidecar.py \
  --model-id "$MODEL_ID" \
  --benchmarks "$BENCHMARKS" \
  --no-think-samples "$NO_THINK_SAMPLES" \
  --thinking-samples "$THINKING_SAMPLES" \
  --no-think-max-tokens "$NO_THINK_MAX_TOKENS" \
  --thinking-max-tokens "$THINKING_MAX_TOKENS" \
  --upload-repo "$UPLOAD_REPO"
