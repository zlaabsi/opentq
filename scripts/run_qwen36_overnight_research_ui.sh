#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

STAMP="${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ROOT="${RUN_ROOT:-artifacts/overnight-research-ui/$STAMP}"
LOG_ROOT="$RUN_ROOT/logs"
SUMMARY="$RUN_ROOT/RUN_SUMMARY.md"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3.6-27B}"

mkdir -p "$LOG_ROOT"

timestamp() {
  date "+%Y-%m-%d %H:%M:%S"
}

append_summary() {
  printf "%s\n" "${1:-}" >>"$SUMMARY"
}

init_summary() {
  {
    echo "# OpenTQ Qwen3.6 Research/UI Overnight Run"
    echo
    echo "- Started: $(timestamp)"
    echo "- Run root: \`$RUN_ROOT\`"
    echo "- Branch: \`$(git branch --show-current)\`"
    echo "- Commit: \`$(git rev-parse --short HEAD)\`"
    echo "- Model: \`$MODEL_ID\`"
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
  /bin/bash -lc "$command" >"$log_path" 2>&1
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
    find "$RUN_ROOT" -maxdepth 3 -type f | sort
    echo '```'
    echo
    echo "- Finished: $(timestamp)"
  } >>"$SUMMARY"
}

init_summary

run_step required "pytest-research-ui" \
  "uv run pytest tests/test_kv_cache.py tests/test_pruning.py tests/test_allocation_ui.py -q"

run_step required "dynamic-plan-q4" \
  "uv run opentq dynamic-gguf-plan --profile OTQ-DYN-Q4_K_M --recipe qwen3.6-27b --output '$RUN_ROOT/dynamic-plan-q4' --no-converter-mapping"

run_step required "kv-cache-layer-policy" \
  "uv run opentq kv-cache-plan --model-id '$MODEL_ID' --weight-plan '$RUN_ROOT/dynamic-plan-q4/plan.json' --output '$RUN_ROOT/kv-cache-policy' --default-dtype fp8_e4m3 --promote-dtype bf16 --edge-layers 2 --periodic-stride 8"

run_step required "quantization-aware-pruning-candidates" \
  "uv run opentq pruning-candidates --plan '$RUN_ROOT/dynamic-plan-q4/plan.json' --output '$RUN_ROOT/pruning-candidates' --max-candidates 256"

run_step required "allocation-ui-artifact" \
  "uv run opentq allocation-ui --plan '$RUN_ROOT/dynamic-plan-q4/plan.json' --output '$RUN_ROOT/allocation-ui' --title 'OpenTQ Qwen3.6-27B Allocation Explorer'"

run_step optional "full-pytest-regression" "uv run pytest -q"

finalize_summary
echo "[$(timestamp)] summary=$SUMMARY"
