#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

STAMP="${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ROOT="${RUN_ROOT:-artifacts/overnight-native/$STAMP}"
LOG_ROOT="$RUN_ROOT/logs"
SUMMARY="$RUN_ROOT/RUN_SUMMARY.md"

DATASET_REPO_ID="${DATASET_REPO_ID:-zlaabsi/Qwen3.6-27B-OTQ-GGUF-benchmarks}"
HF_UPLOAD_DATASET="${HF_UPLOAD_DATASET:-1}"
CLEANUP_APPLY="${CLEANUP_APPLY:-1}"
BENCHMARK_SAMPLES_PER_FAMILY="${BENCHMARK_SAMPLES_PER_FAMILY:-64}"
BENCHMARK_MODELS="${BENCHMARK_MODELS:-q3,q4,q5}"
BENCHMARK_TIMEOUT="${BENCHMARK_TIMEOUT:-3600}"
PUBLICATION_CANDIDATE_MIN_FREE_GIB="${PUBLICATION_CANDIDATE_MIN_FREE_GIB:-25}"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-../llama.cpp}"

mkdir -p "$LOG_ROOT"

timestamp() {
  date "+%Y-%m-%d %H:%M:%S"
}

append_summary() {
  printf "%s\n" "${1:-}" >>"$SUMMARY"
}

init_summary() {
  {
    echo "# OpenTQ Qwen3.6 Overnight Native Run"
    echo
    echo "- Started: $(timestamp)"
    echo "- Run root: \`$RUN_ROOT\`"
    echo "- Branch: \`$(git branch --show-current)\`"
    echo "- Commit: \`$(git rev-parse --short HEAD)\`"
    echo "- Dataset repo: \`$DATASET_REPO_ID\`"
    echo "- Benchmark models: \`$BENCHMARK_MODELS\`"
    echo "- Benchmark samples/family: \`$BENCHMARK_SAMPLES_PER_FAMILY\`"
    echo "- Cleanup apply: \`$CLEANUP_APPLY\`"
    echo
    echo "## Phases"
  } >"$SUMMARY"
}

run_step() {
  local required="$1"
  local name="$2"
  local command="$3"
  local log_path="$LOG_ROOT/$name.log"

  echo "[$(timestamp)] start $name"
  append_summary "- RUNNING \`$name\`: \`$command\`"
  eval "$command" >"$log_path" 2>&1
  local status=$?
  if [[ "$status" -eq 0 ]]; then
    echo "[$(timestamp)] pass $name"
    append_summary "- PASS \`$name\`"
  else
    echo "[$(timestamp)] fail $name status=$status"
    append_summary "- FAIL \`$name\` status=$status; see \`$log_path\`"
    if [[ "$required" == "required" ]]; then
      finalize_summary
      exit "$status"
    fi
  fi
}

finalize_summary() {
  {
    echo
    echo "## Final State"
    echo
    echo '```text'
    echo "git:"
    git status --short --branch
    echo
    echo "disk:"
    df -h .
    echo
    echo "run files:"
    find "$RUN_ROOT" -maxdepth 2 -type f | sort
    echo '```'
    echo
    echo "- Finished: $(timestamp)"
  } >>"$SUMMARY"
}

write_environment() {
  {
    echo "## Git"
    git status --short --branch
    git log -5 --oneline
    echo
    echo "## Disk"
    df -h .
    du -sh artifacts/* 2>/dev/null | sort -h | tail -80 || true
    echo
    echo "## HF Auth"
    hf auth whoami || true
    echo
    echo "## llama.cpp"
    ls -la "$LLAMA_CPP_DIR/build/bin" 2>/dev/null || true
  } >"$RUN_ROOT/environment.txt" 2>&1
}

dataset_viewer_check() {
  local encoded="${DATASET_REPO_ID/\//%2F}"
  local base="https://datasets-server.huggingface.co"
  local out="$RUN_ROOT/hf-dataset-viewer"
  mkdir -p "$out"
  curl -fsS "$base/is-valid?dataset=$encoded" >"$out/is-valid.json"
  curl -fsS "$base/splits?dataset=$encoded" >"$out/splits.json"

  for config in paired_samples paired_summary; do
    local ok=0
    for _ in $(seq 1 20); do
      if curl -fsS "$base/first-rows?dataset=$encoded&config=$config&split=train" >"$out/first-rows-$config.json"; then
        ok=1
        break
      fi
      sleep 60
    done
    if [[ "$ok" != "1" ]]; then
      echo "Dataset Viewer did not return first rows for $config" >&2
      return 1
    fi
  done
}

stage_and_upload_dataset() {
  local dataset_stage="$RUN_ROOT/hf-dataset-stage"
  uv run python scripts/stage_qwen36_benchmark_repro_dataset.py --output "$dataset_stage"
  if [[ "$HF_UPLOAD_DATASET" == "1" ]]; then
    hf upload "$DATASET_REPO_ID" "$dataset_stage" . \
      --repo-type dataset \
      --commit-message "Refresh browseable Qwen3.6 OTQ benchmark reproducibility dataset"
  fi
}

build_llama_cpp_runtime_targets() {
  if [[ ! -d "$LLAMA_CPP_DIR/build" ]]; then
    echo "missing llama.cpp build directory: $LLAMA_CPP_DIR/build" >&2
    return 1
  fi
  cmake --build "$LLAMA_CPP_DIR/build" --target opentq-dequant-probe llama-gguf llama-cli llama-bench -j 4
}

init_summary
write_environment

run_step required "pytest-core" "uv run pytest -q"
run_step required "stage-upload-browseable-dataset" "stage_and_upload_dataset"
run_step optional "hf-dataset-viewer-check" "dataset_viewer_check"

run_step optional "publication-candidate-benchmarks-long" \
  "WAIT_FOR_SESSIONS='' MODELS='$BENCHMARK_MODELS' SAMPLES_PER_FAMILY='$BENCHMARK_SAMPLES_PER_FAMILY' TIMEOUT='$BENCHMARK_TIMEOUT' PUBLICATION_CANDIDATE_MIN_FREE_GIB='$PUBLICATION_CANDIDATE_MIN_FREE_GIB' RUN_ROOT='$RUN_ROOT/publication-candidate-benchmarks' ./scripts/run_qwen36_publication_candidate_benchmarks.sh"

run_step optional "build-llama-cpp-runtime-targets" "build_llama_cpp_runtime_targets"
run_step optional "packed-runtime-full-audit" \
  "OUT_ROOT='$RUN_ROOT/packed-runtime-gates' RUN_PACKED=1 RUN_METAL_SMOKE=0 FULL_PACK_AUDIT=1 AUDIT_DEQUANTIZE_SAMPLES=8 PROBE_TIMEOUT=300 LLAMA_CPP_DIR='$LLAMA_CPP_DIR' ./scripts/run_qwen36_native_runtime_gates.sh"
run_step optional "metal-native-matrix" \
  "OUT_ROOT='$RUN_ROOT/metal-matrix' LLAMA_CPP_DIR='$LLAMA_CPP_DIR' METAL_CTX_SIZE=4096 METAL_N_PREDICT=256 METAL_BENCH_PROMPT_TOKENS=2048 METAL_BENCH_GEN_TOKENS=128 METAL_TIMEOUT=3600 ./scripts/run_qwen36_metal_matrix.sh"

run_step required "cleanup-manifest" \
  "uv run python scripts/build_qwen36_cleanup_manifest.py --output '$RUN_ROOT/cleanup-manifest.json'"
if [[ "$CLEANUP_APPLY" == "1" ]]; then
  run_step optional "cleanup-apply-safe-published-artifacts" \
    "uv run python scripts/build_qwen36_cleanup_manifest.py --output '$RUN_ROOT/cleanup-manifest-applied.json' --apply"
else
  append_summary "- SKIP \`cleanup-apply-safe-published-artifacts\`: CLEANUP_APPLY=$CLEANUP_APPLY"
fi

finalize_summary
echo "[$(timestamp)] summary=$SUMMARY"
