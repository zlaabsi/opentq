#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

STAMP="${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ROOT="${RUN_ROOT:-artifacts/qwen3.6-27b-publication-candidate-benchmarks/$STAMP}"
LOG_ROOT="$RUN_ROOT/logs"
SUMMARY="$RUN_ROOT/RUN_SUMMARY.md"
MODELS="${MODELS:-q3,q4,q5}"
SAMPLES_PER_FAMILY="${SAMPLES_PER_FAMILY:-24}"
TIMEOUT="${TIMEOUT:-2400}"
MIN_FREE_GIB="${PUBLICATION_CANDIDATE_MIN_FREE_GIB:-25}"
WAIT_FOR_SESSIONS="${WAIT_FOR_SESSIONS:-opentq_qwen36_representative}"
WAIT_POLL_SECONDS="${WAIT_POLL_SECONDS:-300}"
WAIT_MAX_SECONDS="${WAIT_MAX_SECONDS:-86400}"
BENCHMARK_IDS="${BENCHMARK_IDS:-mmlu mmlu_pro arc hellaswag gsm8k math aime humaneval mbpp livecodebench bbh gpqa ifeval truthfulqa winogrande drop piqa commonsenseqa}"
SWE_SAMPLES="${SWE_SAMPLES:-3}"
SWE_MAX_TOKENS="${SWE_MAX_TOKENS:-4096}"

mkdir -p "$LOG_ROOT"

timestamp() {
  date "+%Y-%m-%d %H:%M:%S"
}

append_summary() {
  printf "%s\n" "${1:-}" >>"$SUMMARY"
}

benchmark_args() {
  for benchmark_id in $BENCHMARK_IDS; do
    printf " --benchmark-id %q" "$benchmark_id"
  done
}

init_summary() {
  {
    echo "# Qwen3.6 Publication-Candidate Benchmark Run"
    echo
    echo "- Started: $(timestamp)"
    echo "- Runner: OpenTQ release tooling"
    echo "- Run root: \`$RUN_ROOT\`"
    echo "- Models: \`$MODELS\`"
    echo "- Samples per benchmark cap: \`$SAMPLES_PER_FAMILY\`"
    echo "- Token policy: adapter-native max tokens, no global cap."
    echo "- Benchmarks: \`$BENCHMARK_IDS\`"
    echo "- Scope: publication-candidate evidence only; not an official full-benchmark claim."
    echo
    echo "## Steps"
  } >"$SUMMARY"
}

session_running() {
  session="$1"
  screen -list >"$RUN_ROOT/screen-wait.list" 2>/dev/null || true
  grep -q "[.]$session" "$RUN_ROOT/screen-wait.list"
}

wait_for_sessions() {
  if [[ -z "$WAIT_FOR_SESSIONS" ]]; then
    append_summary "- No prerequisite sessions configured."
    return 0
  fi
  waited=0
  while true; do
    running=""
    for session in $WAIT_FOR_SESSIONS; do
      if session_running "$session"; then
        running="$running $session"
      fi
    done
    if [[ -z "$running" ]]; then
      append_summary "- Prerequisite sessions clear."
      return 0
    fi
    if [[ "$waited" -ge "$WAIT_MAX_SECONDS" ]]; then
      append_summary "- STOP: waited \`${waited}s\` for prerequisite sessions but still running:\`$running\`."
      exit 3
    fi
    echo "[$(timestamp)] waiting for sessions:$running"
    append_summary "- WAIT: prerequisite sessions still running:\`$running\`; waited \`${waited}s\`."
    sleep "$WAIT_POLL_SECONDS"
    waited=$((waited + WAIT_POLL_SECONDS))
  done
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
      exit "$status"
    fi
  fi
}

check_disk_gate() {
  free_kib="$(df -Pk . | awk 'NR==2 {print $4}')"
  min_kib=$((MIN_FREE_GIB * 1024 * 1024))
  free_gib=$((free_kib / 1024 / 1024))
  append_summary "- Disk before benchmark work: \`${free_gib} GiB\` free."
  echo "[$(timestamp)] free_disk=${free_gib}GiB min=${MIN_FREE_GIB}GiB"
  if [[ "$free_kib" -lt "$min_kib" ]]; then
    append_summary "- STOP: free disk below \`${MIN_FREE_GIB} GiB\`; benchmark run skipped."
    exit 2
  fi
}

write_benchmark_summary() {
  append_summary
  append_summary "## Results"
  append_summary
  if ls "$RUN_ROOT/subsets"/*.json >/dev/null 2>&1; then
    jq -r '.model.key as $model | .benchmarks[] | "- " + $model + " " + .benchmark_id + ": " + (.summary.passed|tostring) + "/" + (.summary.total|tostring) + " (" + ((.summary.pass_rate * 10000 | round / 100)|tostring) + "%)."' "$RUN_ROOT/subsets"/*.json >>"$SUMMARY"
  else
    append_summary "- No subset JSONs found."
  fi
}

write_swe_summary() {
  append_summary
  append_summary "## SWE-bench Patch Evidence"
  append_summary
  if ls "$RUN_ROOT/swe-bench-patches"/*.json >/dev/null 2>&1; then
    for path in "$RUN_ROOT/swe-bench-patches"/*.json; do
      append_summary "- Patch-generation JSON: \`$path\`."
    done
  else
    append_summary "- No SWE-bench patch JSONs found."
  fi
  append_summary "- These are not public scores. Pass/fail still requires the official SWE-bench harness."
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
  } >>"$SUMMARY"
  append_summary '```'
  append_summary
  append_summary "- Finished: $(timestamp)"
}

init_summary
wait_for_sessions
run_step required "git-status" "git status --short --branch"
run_step required "pytest-benchmark-matrix" "uv run pytest tests/test_qwen36_benchmark_matrix.py -q"
run_step required "dry-run" "uv run python scripts/run_qwen36_benchmark_subsets.py --dry-run --models '$MODELS' --sample-mode quick"
check_disk_gate

run_step required "publication-candidate-subsets" "uv run python scripts/run_qwen36_benchmark_subsets.py --models '$MODELS' $(benchmark_args) --max-samples-per-family '$SAMPLES_PER_FAMILY' --timeout '$TIMEOUT' --output-root '$RUN_ROOT/subsets'"
run_step required "degradation-report" "uv run python scripts/build_qwen36_degradation_report.py --subset-root '$RUN_ROOT/subsets' --output-root '$RUN_ROOT/degradation-report'"
write_benchmark_summary

run_step optional "swe-bench-patch-generation-q3-q4-q5" "uv run python scripts/run_qwen36_benchmark_subsets.py --models '$MODELS' --benchmark-id swe_bench --allow-external-harness --max-samples-per-family '$SWE_SAMPLES' --max-tokens '$SWE_MAX_TOKENS' --timeout '$TIMEOUT' --output-root '$RUN_ROOT/swe-bench-patches'"
write_swe_summary
write_final_status
echo "[$(timestamp)] summary=$SUMMARY"
