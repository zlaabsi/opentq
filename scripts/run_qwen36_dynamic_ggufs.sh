#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
source "$REPO_DIR/scripts/qwen36_dynamic_common.sh"

LLAMA_CPP="${LLAMA_CPP:-${LLAMA_CPP_DIR:-../llama.cpp}}"
HF_CACHE_ROOT="${HF_CACHE_ROOT:-${HF_HOME:-$HOME/.cache/huggingface}/hub/models--Qwen--Qwen3.6-27B}"
HF_REVISION="${HF_REVISION:-}"
HF_MODEL_DIR="${HF_MODEL_DIR:-}"
BASE_GGUF="${BASE_GGUF:-artifacts/qwen3.6-27b-source/Qwen3.6-27B-BF16.gguf}"
OUT_ROOT="${OUT_ROOT:-artifacts/qwen3.6-27b-dynamic-gguf}"
VALIDATION_ROOT="${VALIDATION_ROOT:-artifacts/qwen3.6-27b-dynamic-validation}"
LOG_ROOT="${LOG_ROOT:-artifacts/logs/dynamic-gguf}"
PROFILES="${PROFILES:-OTQ-DYN-Q4_K_M}"
MIN_FREE_GIB_SOURCE="${MIN_FREE_GIB_SOURCE:-55}"
MIN_FREE_GIB_QUANT="${MIN_FREE_GIB_QUANT:-18}"
RUN_SMOKE="${RUN_SMOKE:-1}"
NGL="${NGL:-99}"
FLASH_ATTN="${FLASH_ATTN:-on}"
CTX_SIZE="${CTX_SIZE:-256}"
N_PREDICT="${N_PREDICT:-4}"
TIMEOUT="${TIMEOUT:-900}"

mkdir -p "$(dirname "$BASE_GGUF")" "$OUT_ROOT" "$VALIDATION_ROOT" "$LOG_ROOT"

timestamp() {
  date "+%Y-%m-%d %H:%M:%S"
}

free_gib() {
  df -g "$REPO_DIR" | awk 'NR==2 {print $4}'
}

require_free_gib() {
  local min_gib="$1"
  local label="$2"
  local free
  free="$(free_gib)"
  if (( free < min_gib )); then
    echo "[$(timestamp)] insufficient free disk for $label: ${free} GiB available, need ${min_gib} GiB" >&2
    exit 1
  fi
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

resolve_hf_model_dir() {
  if [[ -n "$HF_MODEL_DIR" ]]; then
    echo "$HF_MODEL_DIR"
    return 0
  fi
  if [[ -z "$HF_REVISION" && -f "$HF_CACHE_ROOT/refs/main" ]]; then
    HF_REVISION="$(cat "$HF_CACHE_ROOT/refs/main")"
  fi
  if [[ -n "$HF_REVISION" && -d "$HF_CACHE_ROOT/snapshots/$HF_REVISION" ]]; then
    echo "$HF_CACHE_ROOT/snapshots/$HF_REVISION"
    return 0
  fi
  python - <<'PY'
from huggingface_hub import snapshot_download
print(snapshot_download("Qwen/Qwen3.6-27B"))
PY
}

ensure_source_gguf() {
  local log="$LOG_ROOT/source-bf16.log"
  if [[ -s "$BASE_GGUF" ]]; then
    echo "[$(timestamp)] source GGUF exists: $BASE_GGUF" | tee -a "$log"
    return 0
  fi
  require_free_gib "$MIN_FREE_GIB_SOURCE" "BF16 source GGUF conversion"
  local model_dir
  model_dir="$(resolve_hf_model_dir)"
  if [[ ! -d "$model_dir" ]]; then
    echo "[$(timestamp)] missing HF model dir: $model_dir" >&2
    exit 1
  fi
  echo "[$(timestamp)] converting source BF16 GGUF from $model_dir -> $BASE_GGUF" | tee -a "$log"
  run_logged "$log" \
    python "$LLAMA_CPP/convert_hf_to_gguf.py" "$model_dir" \
      --outtype bf16 \
      --outfile "$BASE_GGUF" \
      --model-name Qwen3.6-27B
  echo "[$(timestamp)] source GGUF ready: $(ls -lh "$BASE_GGUF" | awk '{print $5, $9}')" | tee -a "$log"
}

ensure_source_gguf

for profile in $PROFILES; do
  slug="Qwen3.6-27B-${profile}-GGUF"
  out_dir="$OUT_ROOT/$slug"
  target="$out_dir/$(qwen36_dynamic_public_filename "$profile")"
  log="$LOG_ROOT/$slug.log"
  mkdir -p "$out_dir"
  qwen36_dynamic_ensure_public_alias "$out_dir" "$profile"

  echo "[$(timestamp)] planning $profile" | tee -a "$log"
  uv run opentq dynamic-gguf-plan \
    --profile "$profile" \
    --output "$out_dir" \
    --llama-cpp "$LLAMA_CPP" \
    --source-gguf "$BASE_GGUF" \
    --target-gguf "$target" >>"$log" 2>&1

  if [[ ! -s "$out_dir/dry-run.log" ]]; then
    echo "[$(timestamp)] dry-run $profile" | tee -a "$log"
    DRY_RUN=1 "$out_dir/quantize.sh" >"$out_dir/dry-run.log" 2>&1 || {
      cat "$out_dir/dry-run.log" >>"$log"
      exit 1
    }
  fi

  if [[ -s "$target" ]]; then
    echo "[$(timestamp)] skip quantization $profile (exists: $target)" | tee -a "$log"
  else
    require_free_gib "$MIN_FREE_GIB_QUANT" "$profile quantization"
    echo "[$(timestamp)] quantizing $profile -> $target" | tee -a "$log"
    run_logged "$log" "$out_dir/quantize.sh"
    echo "[$(timestamp)] quantized $profile: $(ls -lh "$target" | awk '{print $5, $9}')" | tee -a "$log"
  fi

  if [[ "$RUN_SMOKE" == "1" ]]; then
    validation="$VALIDATION_ROOT/$slug-smoke.json"
    if [[ -s "$validation" ]]; then
      echo "[$(timestamp)] skip smoke $profile (exists: $validation)" | tee -a "$log"
    else
      echo "[$(timestamp)] smoke validate $profile" | tee -a "$log"
      uv run opentq validate-gguf \
        --gguf "$target" \
        --output "$validation" \
        --llama-cpp "$LLAMA_CPP" \
        --ngl "$NGL" \
        --flash-attn "$FLASH_ATTN" \
        --ctx-size "$CTX_SIZE" \
        --n-predict "$N_PREDICT" \
        --timeout "$TIMEOUT" \
        --prompt "The capital of France is" >>"$log" 2>&1
    fi
  fi
done
