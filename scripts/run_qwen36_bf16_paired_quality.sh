#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

STAMP="${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ROOT="${RUN_ROOT:-artifacts/qwen3.6-27b-bf16-paired-quality/$STAMP}"
LOG_ROOT="$RUN_ROOT/logs"
SUMMARY="$RUN_ROOT/RUN_SUMMARY.md"
MODELS="${MODELS:-bf16_gguf,q3,q4,q5}"
CRITICAL_BENCHMARK_IDS="${CRITICAL_BENCHMARK_IDS:-mmlu_pro gpqa aime}"
BROAD_BENCHMARK_IDS="${BROAD_BENCHMARK_IDS:-mmlu arc hellaswag gsm8k math bbh truthfulqa winogrande drop piqa commonsenseqa}"
CRITICAL_SAMPLES="${CRITICAL_SAMPLES:-8}"
BROAD_SAMPLES="${BROAD_SAMPLES:-8}"
THINKING_SAMPLES="${THINKING_SAMPLES:-4}"
NO_THINK_MAX_TOKENS="${NO_THINK_MAX_TOKENS:-2048}"
THINKING_MAX_TOKENS="${THINKING_MAX_TOKENS:-4096}"
TIMEOUT="${TIMEOUT:-3600}"
MIN_FREE_GIB="${BF16_PAIRED_MIN_FREE_GIB:-20}"
RUN_BROAD="${RUN_BROAD:-1}"
RUN_THINKING="${RUN_THINKING:-1}"
RUN_PERF="${RUN_PERF:-1}"
LLAMA_CPP="${LLAMA_CPP:-/Users/zlaabsi/Documents/GitHub/llama.cpp}"

mkdir -p "$LOG_ROOT"

timestamp() {
  date "+%Y-%m-%d %H:%M:%S"
}

append_summary() {
  printf "%s\n" "${1:-}" >>"$SUMMARY"
}

benchmark_args() {
  for benchmark_id in $1; do
    printf " --benchmark-id %q" "$benchmark_id"
  done
}

init_summary() {
  {
    echo "# Qwen3.6 BF16-vs-OTQ Paired Quality Run"
    echo
    echo "- Started: $(timestamp)"
    echo "- DRI: Codex"
    echo "- Run root: \`$RUN_ROOT\`"
    echo "- Models: \`$MODELS\`"
    echo "- Critical benchmarks: \`$CRITICAL_BENCHMARK_IDS\`"
    echo "- Broad benchmarks: \`$BROAD_BENCHMARK_IDS\`"
    echo "- Policy: direct degradation claims only against local BF16 GGUF on identical task ids; official scores are anchors only."
    echo "- Stop condition: command failures are recorded; required preflight failures stop the run."
    echo
    echo "## Steps"
  } >"$SUMMARY"
}

run_step() {
  required="$1"
  name="$2"
  command="$3"
  log_path="$LOG_ROOT/$name.log"

  echo "[$(timestamp)] start $name"
  append_summary "- RUNNING \`$name\`: \`$command\`"
  /bin/bash -lc "$command" >"$log_path" 2>&1
  status=$?

  if [[ "$status" -eq 0 ]]; then
    echo "[$(timestamp)] pass $name"
    append_summary "- PASS \`$name\`"
  else
    echo "[$(timestamp)] fail $name status=$status"
    append_summary "- FAIL \`$name\` status=$status; see \`$log_path\`"
    if [[ "$required" == "required" ]]; then
      write_final_status
      exit "$status"
    fi
  fi
}

check_disk_gate() {
  free_kib="$(df -Pk . | awk 'NR==2 {print $4}')"
  min_kib=$((MIN_FREE_GIB * 1024 * 1024))
  free_gib=$((free_kib / 1024 / 1024))
  append_summary "- Disk before BF16 paired work: \`${free_gib} GiB\` free."
  echo "[$(timestamp)] free_disk=${free_gib}GiB min=${MIN_FREE_GIB}GiB"
  if [[ "$free_kib" -lt "$min_kib" ]]; then
    append_summary "- STOP: free disk below \`${MIN_FREE_GIB} GiB\`; BF16 paired run skipped."
    write_final_status
    exit 2
  fi
}

require_bf16_gguf() {
  if [[ ! -f artifacts/qwen3.6-27b-source/Qwen3.6-27B-BF16.gguf ]]; then
    append_summary "- STOP: missing local BF16 GGUF at \`artifacts/qwen3.6-27b-source/Qwen3.6-27B-BF16.gguf\`."
    write_final_status
    exit 4
  fi
}

write_report_pointer() {
  report_root="$1"
  append_summary
  append_summary "## Report"
  append_summary
  append_summary "- Paired quality report: \`$report_root/paired-quality-report.md\`"
}

run_perf_benchmarks() {
  if [[ "$RUN_PERF" != "1" ]]; then
    append_summary "- SKIP performance benchmarks because RUN_PERF=$RUN_PERF."
    return 0
  fi
  mkdir -p "$RUN_ROOT/perf"
  for model_key in bf16_gguf q3 q4 q5; do
    case "$model_key" in
      bf16_gguf) model_path="artifacts/qwen3.6-27b-source/Qwen3.6-27B-BF16.gguf" ;;
      q3) model_path="artifacts/hf-gguf-canonical/Qwen3.6-27B-OTQ-GGUF/Qwen3.6-27B-OTQ-DYN-Q3_K_M.gguf" ;;
      q4) model_path="artifacts/hf-gguf-canonical/Qwen3.6-27B-OTQ-GGUF/Qwen3.6-27B-OTQ-DYN-Q4_K_M.gguf" ;;
      q5) model_path="artifacts/hf-gguf-canonical/Qwen3.6-27B-OTQ-GGUF/Qwen3.6-27B-OTQ-DYN-Q5_K_M.gguf" ;;
    esac
    run_step optional "perf-$model_key" "'$LLAMA_CPP/build/bin/llama-bench' -m '$model_path' -ngl 99 -fa 1 -p 128 -n 64 -r 2 -o json > '$RUN_ROOT/perf/$model_key.json'"
  done
}

write_final_status() {
  append_summary
  append_summary "## Final Status"
  append_summary
  append_summary '```text'
  {
    echo "git:"
    git status --short --branch
    echo
    echo "disk:"
    df -h .
    echo
    echo "screens:"
    screen -list || true
  } >>"$SUMMARY"
  append_summary '```'
  append_summary
  append_summary "- Finished: $(timestamp)"
}

init_summary
require_bf16_gguf
run_step required "git-status" "git status --short --branch"
run_step required "pytest-benchmark-matrix" "uv run pytest tests/test_qwen36_benchmark_matrix.py tests/test_qwen36_paired_quality_report.py -q"
run_step required "dry-run-bf16-targets" "uv run python scripts/run_qwen36_benchmark_subsets.py --dry-run --models '$MODELS' --sample-mode quick"
check_disk_gate

run_step required "paired-critical-no-think" "uv run python scripts/run_qwen36_benchmark_subsets.py --models '$MODELS' $(benchmark_args "$CRITICAL_BENCHMARK_IDS") --max-samples-per-family '$CRITICAL_SAMPLES' --max-tokens '$NO_THINK_MAX_TOKENS' --prompt-format qwen3-no-think --temperature 0 --timeout '$TIMEOUT' --output-root '$RUN_ROOT/paired-critical-no-think/subsets'"
run_step required "paired-critical-no-think-report" "uv run python scripts/build_qwen36_paired_quality_report.py --subset-root '$RUN_ROOT/paired-critical-no-think/subsets' --output-root '$RUN_ROOT/paired-critical-no-think/report'"
write_report_pointer "$RUN_ROOT/paired-critical-no-think/report"

if [[ "$RUN_THINKING" == "1" ]]; then
  run_step optional "paired-critical-thinking" "uv run python scripts/run_qwen36_benchmark_subsets.py --models '$MODELS' $(benchmark_args "$CRITICAL_BENCHMARK_IDS") --max-samples-per-family '$THINKING_SAMPLES' --max-tokens '$THINKING_MAX_TOKENS' --prompt-format qwen3-thinking --temperature 0 --timeout '$TIMEOUT' --output-root '$RUN_ROOT/paired-critical-thinking/subsets'"
  run_step optional "paired-critical-thinking-report" "uv run python scripts/build_qwen36_paired_quality_report.py --subset-root '$RUN_ROOT/paired-critical-thinking/subsets' --output-root '$RUN_ROOT/paired-critical-thinking/report'"
  write_report_pointer "$RUN_ROOT/paired-critical-thinking/report"
else
  append_summary "- SKIP thinking-mode critical run because RUN_THINKING=$RUN_THINKING."
fi

if [[ "$RUN_BROAD" == "1" ]]; then
  run_step optional "paired-broad-no-think" "uv run python scripts/run_qwen36_benchmark_subsets.py --models '$MODELS' $(benchmark_args "$BROAD_BENCHMARK_IDS") --max-samples-per-family '$BROAD_SAMPLES' --max-tokens '$NO_THINK_MAX_TOKENS' --prompt-format qwen3-no-think --temperature 0 --timeout '$TIMEOUT' --output-root '$RUN_ROOT/paired-broad-no-think/subsets'"
  run_step optional "paired-broad-no-think-report" "uv run python scripts/build_qwen36_paired_quality_report.py --subset-root '$RUN_ROOT/paired-broad-no-think/subsets' --output-root '$RUN_ROOT/paired-broad-no-think/report'"
  write_report_pointer "$RUN_ROOT/paired-broad-no-think/report"
else
  append_summary "- SKIP broad no-think run because RUN_BROAD=$RUN_BROAD."
fi

run_perf_benchmarks
write_final_status
echo "[$(timestamp)] summary=$SUMMARY"
