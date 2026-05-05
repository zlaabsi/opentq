#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

STAMP="${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT_ROOT="${OUT_ROOT:-artifacts/native-metal-matrix/$STAMP}"
SUMMARY="$OUT_ROOT/RUN_SUMMARY.md"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-../llama.cpp}"
METAL_PRIMARY_ROOT="${METAL_PRIMARY_ROOT:-artifacts/hf-runtime/Qwen3.6-27B-OTQ-Metal-GGUF}"
METAL_FALLBACK_ROOT="${METAL_FALLBACK_ROOT:-artifacts/qwen3.6-27b-gguf}"
METAL_VARIANTS="${METAL_VARIANTS:-TQ3_SB4 TQ4_SB4 TQ4R2 TQ4R4 TQ4_BAL_V2}"

METAL_CTX_SIZE="${METAL_CTX_SIZE:-4096}"
METAL_N_PREDICT="${METAL_N_PREDICT:-256}"
METAL_NGL="${METAL_NGL:-99}"
METAL_FLASH_ATTN="${METAL_FLASH_ATTN:-on}"
METAL_TIMEOUT="${METAL_TIMEOUT:-3600}"
METAL_BENCH_PROMPT_TOKENS="${METAL_BENCH_PROMPT_TOKENS:-2048}"
METAL_BENCH_GEN_TOKENS="${METAL_BENCH_GEN_TOKENS:-128}"
METAL_PROMPT="${METAL_PROMPT:-You are Qwen3.6-27B running through OpenTQ. Explain in two paragraphs why mixed precision tensor allocation can preserve quality while reducing memory.}"

mkdir -p "$OUT_ROOT"

timestamp() {
  date "+%Y-%m-%d %H:%M:%S"
}

append_summary() {
  printf "%s\n" "${1:-}" >>"$SUMMARY"
}

init_summary() {
  {
    echo "# Qwen3.6-27B OpenTQ Metal Matrix"
    echo
    echo "- Started: $(timestamp)"
    echo "- llama.cpp: \`$LLAMA_CPP_DIR\`"
    echo "- Primary root: \`$METAL_PRIMARY_ROOT\`"
    echo "- Fallback root: \`$METAL_FALLBACK_ROOT\`"
    echo "- Variants: \`$METAL_VARIANTS\`"
    echo "- Context / generation: \`${METAL_CTX_SIZE}\` ctx, \`${METAL_N_PREDICT}\` generated tokens"
    echo "- Bench: \`${METAL_BENCH_PROMPT_TOKENS}\` prompt tokens, \`${METAL_BENCH_GEN_TOKENS}\` generated tokens"
    echo "- Metal: \`-ngl $METAL_NGL\`, FlashAttention \`$METAL_FLASH_ATTN\`"
    echo
    echo "## Results"
  } >"$SUMMARY"
}

variant_path() {
  local variant="$1"
  local primary="$METAL_PRIMARY_ROOT/Qwen3.6-27B-OTQ-${variant}-Metal.gguf"
  local fallback="$METAL_FALLBACK_ROOT/Qwen3.6-27B-${variant}/Qwen3.6-27B-${variant}.gguf"
  if [[ -f "$primary" ]]; then
    printf "%s" "$primary"
    return 0
  fi
  if [[ -f "$fallback" ]]; then
    printf "%s" "$fallback"
    return 0
  fi
  return 1
}

run_variant() {
  local variant="$1"
  local gguf
  if ! gguf="$(variant_path "$variant")"; then
    echo "[$(timestamp)] missing $variant"
    append_summary "- MISSING \`$variant\`: no custom Metal/OpenTQ GGUF found in primary or fallback roots."
    return 0
  fi

  local output="$OUT_ROOT/${variant}.json"
  local log="$OUT_ROOT/${variant}.log"
  echo "[$(timestamp)] start $variant $gguf"
  append_summary "- RUNNING \`$variant\`: \`$gguf\`"
  uv run python -m opentq.cli validate-gguf \
    --gguf "$gguf" \
    --output "$output" \
    --llama-cpp "$LLAMA_CPP_DIR" \
    --prompt "$METAL_PROMPT" \
    --ctx-size "$METAL_CTX_SIZE" \
    --n-predict "$METAL_N_PREDICT" \
    --ngl "$METAL_NGL" \
    --flash-attn "$METAL_FLASH_ATTN" \
    --timeout "$METAL_TIMEOUT" \
    --bench \
    --bench-prompt-tokens "$METAL_BENCH_PROMPT_TOKENS" \
    --bench-gen-tokens "$METAL_BENCH_GEN_TOKENS" >"$log" 2>&1
  local status=$?
  if [[ "$status" -eq 0 ]]; then
    append_summary "- PASS \`$variant\`: \`$output\`"
  else
    append_summary "- FAIL \`$variant\` status=$status: see \`$log\` and \`$output\` if present."
  fi
  return 0
}

write_machine_summary() {
  uv run python - "$OUT_ROOT" "$SUMMARY" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
summary = Path(sys.argv[2])
rows = []
for path in sorted(root.glob("*.json")):
    payload = json.loads(path.read_text())
    rows.append(
        {
            "variant": path.stem,
            "overall_pass": payload.get("overall_pass"),
            "artifact": (payload.get("artifact") or {}).get("filename"),
            "bytes": (payload.get("artifact") or {}).get("bytes"),
            "phases": [
                {
                    "label": phase.get("label"),
                    "passed": phase.get("passed"),
                    "duration_seconds": phase.get("duration_seconds"),
                    "failure_reason": phase.get("failure_reason"),
                }
                for phase in payload.get("phases", [])
            ],
        }
    )
with summary.open("a", encoding="utf-8") as handle:
    handle.write("\n## Machine-readable summary\n\n")
    handle.write("```json\n")
    handle.write(json.dumps(rows, indent=2))
    handle.write("\n```\n")
PY
}

init_summary
for variant in $METAL_VARIANTS; do
  run_variant "$variant"
done
write_machine_summary
append_summary
append_summary "- Finished: $(timestamp)"
echo "[$(timestamp)] summary=$SUMMARY"
