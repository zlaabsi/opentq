# Overnight Native OpenTQ Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run an autonomous overnight pass that advances OpenTQ beyond stock GGUFs into packed/runtime validation, Metal-native evidence, dataset reproducibility cleanup, safe disk cleanup, longer benchmarks, and next research/UI workstreams.

**Architecture:** Keep destructive work behind verifiable gates. The overnight runner writes all outputs under `artifacts/overnight-native/<stamp>`, updates only source-controlled docs/scripts/tests before launch, and deletes only local staging directories that match public HF repo files by name and size.

**Tech Stack:** Bash orchestration, Python manifest builders, `uv`/pytest, Hugging Face CLI/API, stock/patched `llama.cpp`, OpenTQ packed runtime probes, GGUF validation, Dataset Viewer API.

---

## File Structure

- Modify: `scripts/stage_qwen36_benchmark_repro_dataset.py`
  - Normalize complex values to Dataset Viewer-stable scalar columns.
  - Pin dataset configs to `data/*.jsonl` so raw JSON evidence does not break Arrow inference.
- Create: `scripts/build_qwen36_cleanup_manifest.py`
  - Verify public HF model/dataset staging directories before deletion.
  - Delete only safe candidates when `--apply` is provided.
- Create: `scripts/run_qwen36_metal_matrix.sh`
  - Run long Metal + FlashAttention validation across native OpenTQ variants when artifacts exist.
  - Mark missing variants explicitly instead of pretending coverage exists.
- Create: `scripts/run_qwen36_overnight_native.sh`
  - End-to-end unattended runner with logs, summary, dataset upload/check, benchmarks, packed gates, Metal gates, and cleanup.
- Create: `docs/research/kv-cache-layer-policy.md`
  - Define the KV-cache mixed precision roadmap.
- Create: `docs/research/quantization-aware-pruning.md`
  - Define the pruning + quantization coupling roadmap.
- Create: `docs/allocation-ui.md`
  - Define the allocation UI and explain how it should expose tensor-family and layer sensitivity.
- Test: `tests/test_benchmark_repro_dataset.py`
  - Validate viewer-stable serialization and dataset card configs.
- Test: `tests/test_cleanup_manifest.py`
  - Validate remote-size matching and safe deletion behavior.

---

### Task 1: Dataset Viewer Fix

- [ ] Convert variable-type benchmark fields (`answer`, `score.actual`, `score.expected`, `stdout_tail`) into stable scalar strings when needed.
- [ ] Add explicit Hugging Face dataset card configs for `paired_samples` and `paired_summary`.
- [ ] Re-stage the dataset under the overnight run directory.
- [ ] Upload to `zlaabsi/Qwen3.6-27B-OTQ-GGUF-benchmarks` when HF auth is available.
- [ ] Poll Dataset Viewer `/first-rows` for both configs.
- [ ] If indexing still fails, keep the failure JSON under `artifacts/overnight-native/<stamp>/hf-dataset-viewer`.

### Task 2: Long Benchmarks

- [ ] Run `tests/test_qwen36_benchmark_matrix.py`.
- [ ] Run dry-run benchmark adapter validation.
- [ ] Launch publication-candidate subsets with `MODELS=q3,q4,q5` and `SAMPLES_PER_FAMILY=64`.
- [ ] Include SWE-bench patch-generation evidence as non-score evidence.
- [ ] Include LiveCodeBench through the existing lite exact-match adapter.
- [ ] Generate degradation report from the longer subset outputs.
- [ ] Keep summary clear: these are still release evidence, not official leaderboard claims.

### Task 3: Packed `.otq` Runtime Public Gate

- [ ] Rebuild patched `llama.cpp` runtime targets if the build directory exists.
- [ ] Run packed probes for `TQ3_SB4`, `TQ4_SB4`, `TQ4R2`, `TQ4R4`, and `TQ4_BAL_V2`.
- [ ] Use full pack audit, not only a small local smoke.
- [ ] Export fixture JSON and C++ probe logs.
- [ ] Mark a public runtime blocker if `opentq-dequant-probe` cannot be built from the current `llama.cpp` checkout.

### Task 4: Metal-native Matrix

- [ ] Search for native OpenTQ GGUFs in `artifacts/hf-runtime/Qwen3.6-27B-OTQ-Metal-GGUF`.
- [ ] Fall back to `artifacts/qwen3.6-27b-gguf/<variant>/<variant>.gguf` when present.
- [ ] Run long bounded generation with Metal + FlashAttention.
- [ ] Run `llama-bench` with `2048` prompt tokens and `128` generated tokens.
- [ ] Cover `TQ3_SB4`, `TQ4_SB4`, `TQ4R2`, `TQ4R4`, `TQ4_BAL_V2` where artifacts exist.
- [ ] Mark missing artifacts explicitly; do not claim full matrix if exports are absent.

### Task 5: Safe Disk Cleanup

- [ ] Build cleanup manifest with candidate sizes and reasons.
- [ ] Verify public HF repo file names and sizes before marking a staging directory safe.
- [ ] Delete only safe-by-default published staging directories.
- [ ] Keep raw native workdirs and `hf-runtime` while Packed/Metal gates are active.
- [ ] Write applied cleanup manifest with exact deleted paths and bytes.

### Task 6: KV-cache Quantization Roadmap

- [ ] Document why KV cache should be optimized per layer, separately from weights.
- [ ] Define policy schema for hybrid BF16/FP8/int-style KV allocation.
- [ ] Define target integrations: OpenTQ policy planner first, vLLM-style skip-layer execution later.
- [ ] Define metrics: prefill/decode throughput, long-context quality, memory footprint, and exact policy trace.

### Task 7: Quantization-aware Pruning Roadmap

- [ ] Document the coupled decision rule: keep, quantize lower, quantize higher, or prune.
- [ ] Separate safe structured candidates from research-only candidates.
- [ ] Define measurements: tensor-family error, head/MLP saliency, latency gain, quality regression.
- [ ] Keep pruning out of public release claims until a reversible experiment harness exists.

### Task 8: Allocation UI

- [ ] Specify an allocation dashboard with tensor treemap, layer filters, family filters, error metrics, and policy diff.
- [ ] Ensure the UI reads OpenTQ `plan.json` / `tensor-types.annotated.tsv`.
- [ ] Present it as a decision tool, not only a visualization.
- [ ] Keep the terminal monitor documented for long quantization runs.

### Task 9: Verification And Push

- [ ] Run targeted tests for dataset staging and cleanup manifest.
- [ ] Run full `uv run pytest -q`.
- [ ] Commit source-controlled changes.
- [ ] Push the overnight branch before launching the detached job.
- [ ] Launch `scripts/run_qwen36_overnight_native.sh` with `nohup`.
- [ ] Record PID and log path in the final response.

---

## Overnight Run Command

```bash
RUN_STAMP="$(date -u +%Y%m%dT%H%M%SZ)" \
HF_UPLOAD_DATASET=1 \
CLEANUP_APPLY=1 \
BENCHMARK_SAMPLES_PER_FAMILY=64 \
LLAMA_CPP_DIR="../llama.cpp" \
nohup ./scripts/run_qwen36_overnight_native.sh \
  > "artifacts/overnight-native/$RUN_STAMP/nohup.log" 2>&1 &
```

## Stop Conditions

- Required source tests fail before runtime work.
- Dataset staging inputs are missing.
- Cleanup manifest cannot verify public HF files.

Runtime gates and long benchmarks are optional phases inside the runner: they log failures and continue so the overnight pass still produces a complete diagnostic package.
