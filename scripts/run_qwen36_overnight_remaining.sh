#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

STAMP="${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ROOT="${RUN_ROOT:-artifacts/qwen3.6-27b-overnight/$STAMP}"
LOG_ROOT="$RUN_ROOT/logs"
SUMMARY="$RUN_ROOT/RUN_SUMMARY.md"
MIN_FREE_GIB="${OVERNIGHT_MIN_FREE_GIB:-25}"
LCB_SAMPLES="${LCB_SAMPLES:-12}"
LCB_MAX_TOKENS="${LCB_MAX_TOKENS:-4096}"
LCB_TIMEOUT="${LCB_TIMEOUT:-1800}"
SWE_SAMPLES="${SWE_SAMPLES:-3}"
SWE_MAX_TOKENS="${SWE_MAX_TOKENS:-2048}"
SWE_TIMEOUT="${SWE_TIMEOUT:-1800}"

mkdir -p "$LOG_ROOT"

timestamp() {
  date "+%Y-%m-%d %H:%M:%S"
}

append_summary() {
  printf "%s\n" "${1:-}" >>"$SUMMARY"
}

init_summary() {
  {
    echo "# Qwen3.6 Overnight Remaining Run"
    echo
    echo "- Started: $(timestamp)"
    echo "- DRI: Codex"
    echo "- Run root: \`$RUN_ROOT\`"
    echo "- Policy: no upload, no deletion, no BF16 sidecar, no synthetic SWE-bench score."
    echo "- Disk hard gate: stop before heavy work if free space is below \`${MIN_FREE_GIB} GiB\`."
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
      append_summary
      append_summary "Stopped after required step failure: \`$name\`."
      exit "$status"
    fi
  fi
}

check_disk_gate() {
  free_kib="$(df -Pk . | awk 'NR==2 {print $4}')"
  min_kib=$((MIN_FREE_GIB * 1024 * 1024))
  free_gib=$((free_kib / 1024 / 1024))
  echo "[$(timestamp)] free_disk=${free_gib}GiB min=${MIN_FREE_GIB}GiB"
  append_summary "- Disk before heavy work: \`${free_gib} GiB\` free."
  if [[ "$free_kib" -lt "$min_kib" ]]; then
    append_summary "- STOP: free disk below \`${MIN_FREE_GIB} GiB\`; heavy benchmark run skipped."
    exit 2
  fi
}

write_livecodebench_summary() {
  if [[ ! -f "$RUN_ROOT/livecodebench-v6/q3.json" || ! -f "$RUN_ROOT/livecodebench-v6/q4.json" ]]; then
    append_summary
    append_summary "## LiveCodeBench v6"
    append_summary
    append_summary "- Missing Q3 or Q4 JSON; no summary generated."
    return
  fi

  append_summary
  append_summary "## LiveCodeBench v6"
  append_summary
  jq -r '.model.key as $model | .benchmarks[] | "- `" + $model + "` `" + .benchmark_id + "`: `" + (.summary.passed|tostring) + "/" + (.summary.total|tostring) + "` pass@1 subset; first task checked `" + (.results[0].score.checked_cases|tostring) + "/" + (.results[0].score.total_cases|tostring) + "` stdin cases."' \
    "$RUN_ROOT/livecodebench-v6/q3.json" \
    "$RUN_ROOT/livecodebench-v6/q4.json" >>"$SUMMARY"
}

write_swe_summary() {
  append_summary
  append_summary "## SWE-bench Verified"
  append_summary
  if [[ -f "$RUN_ROOT/swe-bench-patches/q3.json" ]]; then
    append_summary "- Patch-generation JSON exists for Q3 at \`$RUN_ROOT/swe-bench-patches/q3.json\`."
  fi
  if [[ -f "$RUN_ROOT/swe-bench-patches/q4.json" ]]; then
    append_summary "- Patch-generation JSON exists for Q4 at \`$RUN_ROOT/swe-bench-patches/q4.json\`."
  fi
  append_summary "- These files are not public scores. Pass/fail still requires the official SWE-bench harness."
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

run_step required "git-status" "git status --short --branch"
run_step required "ruff-benchmark-runner" "uv run ruff check scripts/run_qwen36_benchmark_subsets.py tests/test_qwen36_benchmark_matrix.py"
run_step required "pytest-focused" "uv run pytest tests/test_qwen36_benchmark_matrix.py tests/test_qwen36_runtime_repos.py tests/test_shell_scripts.py -q"
run_step required "benchmark-dry-run" "uv run python scripts/run_qwen36_benchmark_subsets.py --dry-run --models q3,q4 --sample-mode quick"
run_step required "livecodebench-fetch-smoke" "uv run python -c 'from scripts.run_qwen36_benchmark_subsets import ADAPTERS, samples_for_adapter; s=samples_for_adapter(ADAPTERS[\"livecodebench\"], max_samples=1)[0]; print(s[\"source_question_id\"], len(s[\"public_test_cases\"]), len(s[\"private_test_cases\"]))'"
run_step required "swe-fetch-smoke" "uv run python -c 'from scripts.run_qwen36_benchmark_subsets import ADAPTERS, samples_for_adapter; s=samples_for_adapter(ADAPTERS[\"swe_bench\"], max_samples=1)[0]; print(s[\"source_instance_id\"], s[\"repo\"], s[\"harness_required\"])'"

check_disk_gate

run_step optional "livecodebench-v6-q3-q4" "uv run python scripts/run_qwen36_benchmark_subsets.py --models q3,q4 --benchmark-id livecodebench --max-samples-per-family '$LCB_SAMPLES' --max-tokens '$LCB_MAX_TOKENS' --timeout '$LCB_TIMEOUT' --output-root '$RUN_ROOT/livecodebench-v6'"
write_livecodebench_summary

run_step optional "swe-bench-patch-generation-q3-q4" "uv run python scripts/run_qwen36_benchmark_subsets.py --models q3,q4 --benchmark-id swe_bench --allow-external-harness --max-samples-per-family '$SWE_SAMPLES' --max-tokens '$SWE_MAX_TOKENS' --timeout '$SWE_TIMEOUT' --output-root '$RUN_ROOT/swe-bench-patches'"
write_swe_summary

run_step optional "runtime-repo-restage-check" "PYTHONPATH=src uv run python scripts/stage_qwen36_otq_runtime_repos.py --output-root '$RUN_ROOT/hf-runtime-restage' --link-mode hardlink"
run_step optional "runtime-repo-tests" "uv run pytest tests/test_qwen36_runtime_repos.py -q"

write_final_status
echo "[$(timestamp)] summary=$SUMMARY"
