#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

PACKED_ROOT="${PACKED_ROOT:-artifacts/hf-runtime/Qwen3.6-27B-OTQ-Packed}"
METAL_GGUF="${METAL_GGUF:-artifacts/hf-runtime/Qwen3.6-27B-OTQ-Metal-GGUF/Qwen3.6-27B-OTQ-TQ3_SB4-Metal.gguf}"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-../llama.cpp}"
PROBE_BINARY="${PROBE_BINARY:-$LLAMA_CPP_DIR/build/bin/opentq-dequant-probe}"
OUT_ROOT="${OUT_ROOT:-artifacts/runtime-gates}"

RUN_PACKED="${RUN_PACKED:-1}"
RUN_METAL_SMOKE="${RUN_METAL_SMOKE:-1}"
RUN_METAL_BENCH="${RUN_METAL_BENCH:-0}"
FULL_PACK_AUDIT="${FULL_PACK_AUDIT:-0}"

AUDIT_MAX_TENSORS="${AUDIT_MAX_TENSORS:-20}"
AUDIT_DEQUANTIZE_SAMPLES="${AUDIT_DEQUANTIZE_SAMPLES:-4}"
MAX_FIXTURES_PER_VARIANT="${MAX_FIXTURES_PER_VARIANT:-1}"
PROBE_TIMEOUT="${PROBE_TIMEOUT:-120}"

METAL_CTX_SIZE="${METAL_CTX_SIZE:-256}"
METAL_N_PREDICT="${METAL_N_PREDICT:-8}"
METAL_NGL="${METAL_NGL:-99}"
METAL_FLASH_ATTN="${METAL_FLASH_ATTN:-on}"
METAL_TIMEOUT="${METAL_TIMEOUT:-600}"
METAL_PROMPT="${METAL_PROMPT:-The capital of France is}"
METAL_BENCH_PROMPT_TOKENS="${METAL_BENCH_PROMPT_TOKENS:-64}"
METAL_BENCH_GEN_TOKENS="${METAL_BENCH_GEN_TOKENS:-4}"

if [[ -n "${PACKED_RELEASES:-}" ]]; then
  read -r -a RELEASES <<<"$PACKED_RELEASES"
else
  RELEASES=(
    "Qwen3.6-27B-OTQ-TQ3_SB4"
    "Qwen3.6-27B-OTQ-TQ4_SB4"
    "Qwen3.6-27B-OTQ-TQ4R2"
    "Qwen3.6-27B-OTQ-TQ4R4"
    "Qwen3.6-27B-OTQ-TQ4_BAL_V2"
  )
fi

mkdir -p "$OUT_ROOT"
SUMMARY="$OUT_ROOT/RUN_SUMMARY.md"

timestamp() {
  date "+%Y-%m-%d %H:%M:%S"
}

append_summary() {
  printf "%s\n" "${1:-}" >>"$SUMMARY"
}

init_summary() {
  {
    echo "# Qwen3.6-27B Native OpenTQ Runtime Gates"
    echo
    echo "- Started: $(timestamp)"
    echo "- Packed root: \`$PACKED_ROOT\`"
    echo "- Metal GGUF: \`$METAL_GGUF\`"
    echo "- llama.cpp dir: \`$LLAMA_CPP_DIR\`"
    echo "- Probe binary: \`$PROBE_BINARY\`"
    echo "- Scope: Packed .otq runtime probes and custom OpenTQ GGUF Metal smoke checks."
    echo
    echo "## Results"
  } >"$SUMMARY"
}

run_step() {
  local name="$1"
  shift
  echo "[$(timestamp)] start $name"
  if "$@"; then
    echo "[$(timestamp)] pass $name"
    append_summary "- PASS \`$name\`"
  else
    local status=$?
    echo "[$(timestamp)] fail $name status=$status"
    append_summary "- FAIL \`$name\` status=$status"
    return "$status"
  fi
}

probe_packed_release() {
  local release="$1"
  local packed="$PACKED_ROOT/$release"
  local label="${release#Qwen3.6-27B-OTQ-}"
  local output="$OUT_ROOT/$label/pack-runtime-probe.json"
  local fixtures="$OUT_ROOT/$label/fixtures"
  local args=(
    uv run python -m opentq.cli probe-pack-runtime
    --packed "$packed"
    --fixtures-output "$fixtures"
    --probe-binary "$PROBE_BINARY"
    --output "$output"
    --audit-dequantize-samples "$AUDIT_DEQUANTIZE_SAMPLES"
    --max-fixtures-per-variant "$MAX_FIXTURES_PER_VARIANT"
    --timeout "$PROBE_TIMEOUT"
  )

  if [[ "$FULL_PACK_AUDIT" != "1" ]]; then
    args+=(--audit-max-tensors "$AUDIT_MAX_TENSORS")
  fi

  if [[ ! -f "$packed/opentq-pack.json" ]]; then
    echo "missing packed release: $packed" >&2
    return 2
  fi
  "${args[@]}"
}

validate_metal_smoke() {
  local output="$OUT_ROOT/metal/TQ3_SB4-metal-smoke.json"
  if [[ "$RUN_METAL_BENCH" == "1" ]]; then
    output="$OUT_ROOT/metal/TQ3_SB4-metal-bench.json"
  fi
  local args=(
    uv run python -m opentq.cli validate-gguf
    --gguf "$METAL_GGUF"
    --output "$output"
    --llama-cpp "$LLAMA_CPP_DIR"
    --prompt "$METAL_PROMPT"
    --ctx-size "$METAL_CTX_SIZE"
    --n-predict "$METAL_N_PREDICT"
    --ngl "$METAL_NGL"
    --flash-attn "$METAL_FLASH_ATTN"
    --timeout "$METAL_TIMEOUT"
  )

  if [[ "$RUN_METAL_BENCH" == "1" ]]; then
    args+=(--bench --bench-prompt-tokens "$METAL_BENCH_PROMPT_TOKENS" --bench-gen-tokens "$METAL_BENCH_GEN_TOKENS")
  fi

  if [[ ! -f "$METAL_GGUF" ]]; then
    echo "missing Metal GGUF: $METAL_GGUF" >&2
    return 2
  fi
  "${args[@]}"
}

write_machine_summary() {
  uv run python - "$OUT_ROOT" "$SUMMARY" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
summary = Path(sys.argv[2])
rows = []
for path in sorted(root.glob("*/pack-runtime-probe.json")):
    payload = json.loads(path.read_text())
    audit = payload.get("audit", {})
    rows.append(
        {
            "gate": path.parent.name,
            "pass": payload.get("overall_pass"),
            "variants": audit.get("variant_counts", {}),
            "payload_bytes": (audit.get("totals") or {}).get("payload_bytes"),
            "probe_rows": len(payload.get("probe_rows", [])),
        }
    )
metal_path = root / "metal" / "TQ3_SB4-metal-bench.json"
if not metal_path.exists():
    metal_path = root / "metal" / "TQ3_SB4-metal-smoke.json"
metal = None
if metal_path.exists():
    payload = json.loads(metal_path.read_text())
    metal = {
        "pass": payload.get("overall_pass"),
        "gates": payload.get("gates"),
        "bytes": (payload.get("artifact") or {}).get("bytes"),
    }
with summary.open("a", encoding="utf-8") as handle:
    handle.write("\n## Machine-readable summary\n\n")
    handle.write("```json\n")
    handle.write(json.dumps({"packed": rows, "metal": metal}, indent=2))
    handle.write("\n```\n")
PY
}

init_summary
failures=0

if [[ "$RUN_PACKED" == "1" ]]; then
  for release in "${RELEASES[@]}"; do
    run_step "packed-${release#Qwen3.6-27B-OTQ-}" probe_packed_release "$release" || failures=$((failures + 1))
  done
else
  append_summary "- SKIP \`packed\`"
fi

if [[ "$RUN_METAL_SMOKE" == "1" ]]; then
  metal_step="metal-TQ3_SB4-smoke"
  if [[ "$RUN_METAL_BENCH" == "1" ]]; then
    metal_step="metal-TQ3_SB4-bench"
  fi
  run_step "$metal_step" validate_metal_smoke || failures=$((failures + 1))
else
  append_summary "- SKIP \`metal-smoke\`"
fi

write_machine_summary
append_summary
append_summary "- Finished: $(timestamp)"
echo "[$(timestamp)] summary=$SUMMARY"

if [[ "$failures" -gt 0 ]]; then
  exit 1
fi
