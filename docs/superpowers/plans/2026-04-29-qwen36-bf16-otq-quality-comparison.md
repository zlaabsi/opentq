# Qwen3.6 BF16-vs-OTQ Quality Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce defensible BF16-vs-Q3/Q4/Q5 quality and performance evidence for Qwen3.6-27B OTQ GGUF artifacts.

**Architecture:** Treat official Qwen numbers as external anchors, not quantization deltas. Direct degradation is measured only by running the local BF16 GGUF and each quantized GGUF through the same runner, task ids, prompt mode, generation settings, and scoring rule, then producing a paired report.

**Tech Stack:** llama.cpp GGUF runtime, pinned Hugging Face datasets via existing adapters, Python JSON/Markdown report builders, bash screen/caffeinate long-running launchers, pytest/ruff verification.

---

## Research Decision

- Hugging Face quantization guidance compares quantized models by accuracy, throughput, memory, and hardware-specific behavior; it explicitly says to benchmark on the target task and hardware.
- Qwen3.6 official scores are not a BF16 sidecar for our runner. They use benchmark-specific harnesses and Qwen thinking behavior; the model card says Qwen3.6 operates in thinking mode by default and documents benchmark-specific settings for SWE-bench, Terminal-Bench, SkillsBench, and AIME26.
- Unsloth's GGUF benchmark methodology combines real evals, throughput, disk size, PPL, and KL divergence. Their docs also warn that PPL/KLD can be misleading and should not replace real task evals such as MMLU-Pro and LiveCodeBench.
- Therefore the correct local decision is: run paired local BF16 GGUF diagnostics first, then expand only if the BF16 local anchor behaves sanely. If BF16 local is slow or scores far below official, report a harness/prompt/runtime mismatch instead of blaming quantization.

## Files

- Modify: `scripts/run_qwen36_benchmark_subsets.py`
- Create: `scripts/build_qwen36_paired_quality_report.py`
- Create: `scripts/run_qwen36_bf16_paired_quality.sh`
- Create: `scripts/launch_qwen36_bf16_paired_quality.sh`
- Modify: `scripts/run_qwen36_representative_benchmarks.sh`
- Modify: `scripts/run_qwen36_publication_candidate_benchmarks.sh`
- Modify: `tests/test_qwen36_benchmark_matrix.py`
- Create: `tests/test_qwen36_paired_quality_report.py`
- Modify: `tests/test_shell_scripts.py`

## Task 1: Stop Misleading Publication-Candidate Run

- [x] Verify whether `opentq_qwen36_publication_candidate` is still waiting or already computing.
- [x] Stop it because it compares quantized GGUF subsets to official Qwen scores without a paired BF16 baseline.
- [x] Verify no `run_qwen36_publication`, `run_qwen36_benchmark_subsets`, or `llama-completion` process remains from that screen.

## Task 2: Add Local BF16 GGUF and Prompt Controls

- [x] Add `bf16_gguf` to `MODEL_PATHS`, pointing to `artifacts/qwen3.6-27b-source/Qwen3.6-27B-BF16.gguf`.
- [x] Mark `bf16_gguf` as `gguf_reference` in model payloads.
- [x] Keep `bf16` as the unsupported HF source model key so the local GGUF runner does not accidentally treat a Hub id as a file path.
- [x] Add `--prompt-format` override with `qwen3-no-think` and `qwen3-thinking`.
- [x] Add generation metadata controls: `--temperature`, `--top-p`, `--top-k`, `--seed`, `--context-size`, `--gpu-layers`.
- [x] Save generation metadata into each model JSON.
- [x] Parse final answers after `</think>` so thinking traces do not pollute multiple-choice, numeric, exact-text, and code extraction.

## Task 3: Add Paired Quality Report

- [x] Build `scripts/build_qwen36_paired_quality_report.py`.
- [x] Load one baseline JSON, default `bf16_gguf.json`, and candidate JSONs, default `q3/q4/q5`.
- [x] Join results by identical `task_id`.
- [x] Report local BF16 pass rate, quant pass rate, delta in percentage points, retention, BF16-only failures, quant-only recoveries, both-correct, both-wrong, and Wilson interval for candidate pass rate.
- [x] Include official Qwen scores only as anchors and label `n < 30` as `diagnostic_only_sample_lt_30`.

## Task 4: Add Long-Running Paired Runner

- [x] Build `scripts/run_qwen36_bf16_paired_quality.sh`.
- [x] Preflight: git status, BF16 GGUF existence, pytest, dry-run target parsing, disk gate.
- [x] Phase A: critical no-think paired run for `mmlu_pro gpqa aime`, 8 samples, BF16/Q3/Q4/Q5.
- [x] Phase B: critical thinking paired run for `mmlu_pro gpqa aime`, 4 samples, BF16/Q3/Q4/Q5.
- [x] Phase C: broad no-think paired run for non-agentic text benchmarks, 8 samples, BF16/Q3/Q4/Q5.
- [x] Phase D: llama.cpp `llama-bench` performance pass for BF16/Q3/Q4/Q5.
- [x] Generate paired reports after each phase.
- [x] Use optional phases for long work so one slow section records failure without erasing earlier evidence.

## Task 5: Verification

- [x] Run `uv run pytest tests/test_qwen36_benchmark_matrix.py tests/test_qwen36_paired_quality_report.py tests/test_shell_scripts.py -q`.
- [x] Run `uv run ruff check scripts/run_qwen36_benchmark_subsets.py scripts/build_qwen36_paired_quality_report.py tests/test_qwen36_benchmark_matrix.py tests/test_qwen36_paired_quality_report.py`.
- [x] Run `bash -n scripts/run_qwen36_bf16_paired_quality.sh scripts/launch_qwen36_bf16_paired_quality.sh`.
- [x] Run a dry-run with `--models bf16_gguf,q3,q4,q5`.

## Task 6: Execute Overnight

- [x] Launch `./scripts/launch_qwen36_bf16_paired_quality.sh`.
- [x] Record the screen session, run root, log path, and summary path.
- [x] Leave no HF upload, no artifact deletion, and no official degradation claim in the runner.

## Interpretation Rules

- If local BF16 is also low on MMLU-Pro/AIME/GPQA, the observed failure is primarily prompt/scoring/runtime mismatch, not quantization degradation.
- If local BF16 is high and Q3/Q4/Q5 are much lower on identical items, then the quantization or GGUF conversion is the likely culprit.
- If no-think is bad but thinking improves both BF16 and quants, public quality claims must use thinking-mode results or explicitly say no-think practical mode.
- If BF16 GGUF is too slow or times out on M1 Max 32GB, the next autonomous path is a remote BF16 sidecar on HF Jobs or another GPU runner; without `HF_TOKEN`, local evidence remains diagnostic.
- Publish only paired deltas with matching task ids, prompt mode, scoring rule, and generation config; official Qwen model-card scores remain context, not the denominator for quantization loss.
