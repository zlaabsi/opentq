#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

LLAMA_CPP="${LLAMA_CPP:-/Users/zlaabsi/Documents/GitHub/llama.cpp}"
HF_USER="${HF_USER:-zlaabsi}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3.6-27B}"
PROFILES="${PROFILES:-${PROFILE:-OTQ-DYN-Q3_XL}}"

OUT_ROOT="${OUT_ROOT:-artifacts/qwen3.6-27b-dynamic-gguf}"
VALIDATION_ROOT="${VALIDATION_ROOT:-artifacts/qwen3.6-27b-dynamic-validation}"
EVAL_ROOT="${EVAL_ROOT:-artifacts/qwen3.6-27b-dynamic-eval}"
HF_STAGE_ROOT="${HF_STAGE_ROOT:-artifacts/hf-gguf-dynamic}"
LOG_ROOT="${LOG_ROOT:-artifacts/logs/dynamic-release}"

NGL="${NGL:-99}"
FLASH_ATTN="${FLASH_ATTN:-on}"
QUALITY_CTX_SIZE="${QUALITY_CTX_SIZE:-8192}"
BENCH_CTX_SIZE="${BENCH_CTX_SIZE:-8192}"
BENCH_PROMPT_TOKENS="${BENCH_PROMPT_TOKENS:-8192}"
BENCH_GEN_TOKENS="${BENCH_GEN_TOKENS:-128}"
BENCH_N_PREDICT="${BENCH_N_PREDICT:-128}"
TIMEOUT="${TIMEOUT:-2400}"
QUALITY_TIMEOUT="${QUALITY_TIMEOUT:-1200}"
WAIT_FOR_SMOKE="${WAIT_FOR_SMOKE:-1}"
WAIT_INTERVAL="${WAIT_INTERVAL:-60}"
MAX_WAIT_SECONDS="${MAX_WAIT_SECONDS:-0}"
UPLOAD="${UPLOAD:-1}"
FORCE_QUALITY="${FORCE_QUALITY:-0}"
FORCE_BENCH="${FORCE_BENCH:-0}"

mkdir -p "$VALIDATION_ROOT" "$EVAL_ROOT" "$HF_STAGE_ROOT" "$LOG_ROOT"

timestamp() {
  date "+%Y-%m-%d %H:%M:%S"
}

run_logged() {
  local log="$1"
  shift
  echo "[$(timestamp)] $*" | tee -a "$log"
  if command -v caffeinate >/dev/null 2>&1; then
    PYTHONUNBUFFERED=1 caffeinate -dimsu "$@" >>"$log" 2>&1
  else
    PYTHONUNBUFFERED=1 "$@" >>"$log" 2>&1
  fi
}

json_passed() {
  local file="$1"
  python - "$file" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists() or path.stat().st_size == 0:
    raise SystemExit(1)
payload = json.loads(path.read_text(encoding="utf-8"))
raise SystemExit(0 if payload.get("overall_pass") is True else 1)
PY
}

wait_for_smoke() {
  local profile="$1"
  local smoke="$2"
  local log="$3"
  local waited=0

  if [[ "$WAIT_FOR_SMOKE" != "1" ]]; then
    return 0
  fi

  until json_passed "$smoke"; do
    if [[ -s "$smoke" ]]; then
      echo "[$(timestamp)] smoke exists but did not pass: $smoke" | tee -a "$log"
      return 1
    fi
    if (( MAX_WAIT_SECONDS > 0 && waited >= MAX_WAIT_SECONDS )); then
      echo "[$(timestamp)] timed out waiting for smoke for $profile after ${waited}s" | tee -a "$log"
      return 1
    fi
    echo "[$(timestamp)] waiting for smoke gate for $profile: $smoke" | tee -a "$log"
    sleep "$WAIT_INTERVAL"
    waited=$((waited + WAIT_INTERVAL))
  done
  echo "[$(timestamp)] smoke gate passed for $profile: $smoke" | tee -a "$log"
}

for profile in $PROFILES; do
  slug="Qwen3.6-27B-${profile}-GGUF"
  gguf="$OUT_ROOT/$slug/Qwen3.6-27B-${profile}.gguf"
  smoke="$VALIDATION_ROOT/$slug-smoke.json"
  release_bench="$VALIDATION_ROOT/$slug-release-bench.json"
  quality="$EVAL_ROOT/$slug-quality-qwen3-no-think.json"
  stage="$HF_STAGE_ROOT/$slug"
  repo_id="$HF_USER/$slug"
  log="$LOG_ROOT/$slug.log"

  echo "[$(timestamp)] release pipeline start: $profile" | tee -a "$log"

  if [[ ! -s "$gguf" ]]; then
    echo "[$(timestamp)] missing GGUF artifact: $gguf" | tee -a "$log"
    exit 1
  fi

  wait_for_smoke "$profile" "$smoke" "$log"

  if [[ "$FORCE_QUALITY" != "1" ]] && json_passed "$quality"; then
    echo "[$(timestamp)] skip quality eval $profile (passed: $quality)" | tee -a "$log"
  else
    run_logged "$log" \
      uv run opentq eval-gguf \
        --gguf "$gguf" \
        --output "$quality" \
        --suite benchmarks/qwen36_quality_samples.jsonl \
        --llama-cpp "$LLAMA_CPP" \
        --ngl "$NGL" \
        --flash-attn "$FLASH_ATTN" \
        --ctx-size "$QUALITY_CTX_SIZE" \
        --timeout "$QUALITY_TIMEOUT" \
        --prompt-format qwen3-no-think
  fi

  if [[ "$FORCE_BENCH" != "1" ]] && json_passed "$release_bench"; then
    echo "[$(timestamp)] skip release bench $profile (passed: $release_bench)" | tee -a "$log"
  else
    run_logged "$log" \
      uv run opentq validate-gguf \
        --gguf "$gguf" \
        --output "$release_bench" \
        --llama-cpp "$LLAMA_CPP" \
        --ngl "$NGL" \
        --flash-attn "$FLASH_ATTN" \
        --ctx-size "$BENCH_CTX_SIZE" \
        --n-predict "$BENCH_N_PREDICT" \
        --timeout "$TIMEOUT" \
        --bench \
        --bench-prompt-tokens "$BENCH_PROMPT_TOKENS" \
        --bench-gen-tokens "$BENCH_GEN_TOKENS" \
        --prompt "Write a concise paragraph about Apple Silicon LLM inference."
  fi

  run_logged "$log" \
    uv run opentq prepare-hf-gguf \
      --gguf "$gguf" \
      --output "$stage" \
      --repo-id "$repo_id" \
      --base-model "$BASE_MODEL" \
      --stock-compatible \
      --validation "$release_bench" \
      --quality-eval "$quality"

  if [[ "$UPLOAD" == "1" ]]; then
    echo "[$(timestamp)] hf repo create $repo_id --type model -y" | tee -a "$log"
    hf repo create "$repo_id" --type model -y >>"$log" 2>&1 || {
      echo "[$(timestamp)] repo create returned non-zero; continuing in case the repo already exists: $repo_id" | tee -a "$log"
    }
    run_logged "$log" hf upload-large-folder "$repo_id" "$stage"
  else
    echo "[$(timestamp)] upload disabled for $profile; staged at $stage" | tee -a "$log"
  fi

  echo "[$(timestamp)] release pipeline complete: $repo_id" | tee -a "$log"
done
