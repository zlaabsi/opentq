# Qwen3.6 Long-Running TODO

This is the operational checklist for the Qwen3.6-27B release/evaluation run. The detailed plan is `docs/superpowers/plans/2026-04-28-qwen36-long-running-release-evals.md`.

## Current State

- Canonical stock-compatible public repo name: `zlaabsi/Qwen3.6-27B-OTQ-GGUF`.
- Public stock GGUF files should stay named with `OTQ-DYN` plus stock quant tokens, for example `Qwen3.6-27B-OTQ-DYN-Q3_K_M.gguf`.
- Native technical variants stay `TQ*` internally and can be publicly namespaced as `OTQ-TQ*` when released.
- `Q3_K_M` and `Q4_K_M` are the current validated stock GGUF targets.
- `Q5_K_M` is now disk-unblocked but still unpublished until generation and runtime gates pass.
- `IQ4_NL` is a stock llama.cpp nonlinear 4-bit experiment and requires imatrix evidence before publication.
- Packed and Metal tracks are separate from stock llama.cpp GGUF and must stay gated until public runtime evidence exists.
- Disk cleanup deleted only regenerable Hugging Face caches. Local BF16 GGUF source and release artifacts were preserved.
- The canonical HF GGUF repo has been refreshed with hardware compatibility and practical mini-subset scores.

## Autonomous Rules

- Continue to the next checked phase when the previous command succeeds.
- Stop only for missing credentials, paid remote BF16 hardware, destructive deletion without an explicit current instruction, free disk below `15 GiB`, or repeated command failure.
- Do not claim degradation versus BF16 unless the benchmark is officially comparable or a matching BF16 mini-run exists.
- Do not publish `XL`, `OTQ3`, `OTQ4`, or ambiguous labels.
- Do not run multimodal MMMU/MathVista claims on text-only GGUF artifacts.

## Phase Checklist

- [x] Phase 0: Preflight repo state, disk headroom, llama.cpp binaries, and focused tests.
- [x] Phase 1: Refresh artifact audit, M1 Max 32GB runtime gates, canonical GGUF staging, and naming checks.
- [x] Phase 2: Re-run deterministic OTQ micro-evals and rebuild GGUF benchmark plots/CSVs.
- [x] Phase 3: Add `scripts/run_qwen36_benchmark_subsets.py` dry-run planner plus tests for the benchmark matrix.
- [x] Phase 4a: Implement real benchmark adapters with pinned dataset, split, revision, task ids, prompt format, and scoring rule.
- [x] Phase 4b: Run quick OTQ benchmark subsets for Q3 and Q4 across the requested benchmark families.
- [x] Phase 5a: Add degradation report builder with no-fake-delta gates.
- [x] Phase 5b: Produce final degradation report after Q3/Q4 subset JSONs exist.
- [ ] Phase 6: Generate and validate Q5_K_M now that free disk is at the threshold; do not publish until runtime gates pass.
- [ ] Phase 7: Re-stage and gate `Qwen3.6-27B-OTQ-Packed`; upload only with `HF_UPLOAD=1` and runtime evidence.
- [ ] Phase 8: Re-stage and gate `Qwen3.6-27B-OTQ-Metal-GGUF`; upload only with `HF_UPLOAD=1` and runtime evidence.
- [x] Phase 9: Build cleanup manifest and delete only approved regenerable cache paths.
- [ ] Phase 10: Write final release report with git hash, HF inventory, runtime evidence, benchmark subset summary, and cleanup decisions.

## Benchmark Groups

- Official-comparable if scoring matches: MMLU-Pro, AIME26, SWE-bench Verified/Pro/Multilingual, LiveCodeBench v6, GPQA Diamond.
- OTQ subset plus BF16 mini-run required for degradation: MMLU, ARC, HellaSwag, GSM8K, MATH, HumanEval, MBPP, BBH, IFEval, TruthfulQA, WinoGrande, DROP, PIQA, CommonsenseQA.
- Judge-based sentinel only unless a BF16 sidecar and pinned judge exist: MT-Bench, Chatbot Arena style, AlpacaEval.
- Blocked for current text-only release: MMMU, MathVista.

## Resume Commands

```bash
cd /Users/zlaabsi/Documents/GitHub/opentq
git status --short --branch
sed -n '1,260p' docs/qwen36-long-running-todo.md
sed -n '1,260p' docs/superpowers/plans/2026-04-28-qwen36-long-running-release-evals.md
cat benchmarks/qwen36_long_running_benchmark_matrix.json | python -m json.tool >/dev/null
```

## Stop Conditions

- HF token missing when an upload is requested.
- Free disk below `15 GiB`, or below `80 GiB` before Q5 generation.
- A command would delete artifacts without `ALLOW_DELETE=1`.
- A command would use paid remote BF16 hardware without `RUN_REMOTE_BF16=1`.
- A benchmark harness is missing a pinned split, task id list, prompt format, or scoring rule.
