# Qwen3.6 Long-Running TODO

This is the operational checklist for the Qwen3.6-27B release/evaluation run. The detailed plan is `docs/superpowers/plans/2026-04-28-qwen36-long-running-release-evals.md`.

## Current State

- Canonical stock-compatible public repo name: `zlaabsi/Qwen3.6-27B-OTQ-GGUF`.
- Public stock GGUF files should stay named with `OTQ-DYN` plus stock quant tokens, for example `Qwen3.6-27B-OTQ-DYN-Q3_K_M.gguf`.
- Native technical variants stay `TQ*` internally and can be publicly namespaced as `OTQ-TQ*` when released.
- `Q3_K_M` and `Q4_K_M` are the current validated stock GGUF targets.
- `Q5_K_M` is a future quality-first stock GGUF candidate after disk cleanup and runtime gates.
- `IQ4_NL` is a stock llama.cpp nonlinear 4-bit experiment and requires imatrix evidence before publication.
- Packed and Metal tracks are separate from stock llama.cpp GGUF and must stay gated until public runtime evidence exists.
- Disk pressure is high; cleanup must be manifest-driven and must not remove unpublished only copies.

## Autonomous Rules

- Continue to the next checked phase when the previous command succeeds.
- Stop only for missing credentials, paid remote BF16 hardware, destructive deletion, release upload without `HF_UPLOAD=1`, free disk below `15 GiB`, or repeated command failure.
- Do not claim degradation versus BF16 unless the benchmark is officially comparable or a matching BF16 mini-run exists.
- Do not publish `XL`, `OTQ3`, `OTQ4`, or ambiguous labels.
- Do not run multimodal MMMU/MathVista claims on text-only GGUF artifacts.

## Phase Checklist

- [x] Phase 0: Preflight repo state, disk headroom, llama.cpp binaries, and focused tests.
- [x] Phase 1: Refresh artifact audit, M1 Max 32GB runtime gates, canonical GGUF staging, and naming checks.
- [x] Phase 2: Re-run deterministic OTQ micro-evals and rebuild GGUF benchmark plots/CSVs.
- [x] Phase 3: Add `scripts/run_qwen36_benchmark_subsets.py` dry-run planner plus tests for the benchmark matrix.
- [ ] Phase 4: Implement benchmark adapters, then run quick OTQ benchmark subsets for Q3 and Q4 across the requested benchmark families.
- [ ] Phase 5: Add degradation report builder and produce `artifacts/qwen3.6-27b-degradation-report/`.
- [ ] Phase 6: Decide whether Q5_K_M can be generated; skip if free disk is below `80 GiB`.
- [ ] Phase 7: Re-stage and gate `Qwen3.6-27B-OTQ-Packed`; upload only with `HF_UPLOAD=1` and runtime evidence.
- [ ] Phase 8: Re-stage and gate `Qwen3.6-27B-OTQ-Metal-GGUF`; upload only with `HF_UPLOAD=1` and runtime evidence.
- [ ] Phase 9: Build cleanup manifest and delete only `release_verified` paths with `ALLOW_DELETE=1`.
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
