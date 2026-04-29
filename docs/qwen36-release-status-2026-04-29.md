# Qwen3.6 Release Status 2026-04-29

## Completed

- Pushed `main` through the current release work.
- Implemented pinned benchmark adapters for the text-only Phase 4 set.
- Added the remaining agentic/coding adapters:
  - SWE-bench Verified pinned to `princeton-nlp/SWE-bench_Verified` at `c104f840cc67f8b6eec6f759ebc8b2693d585d4a`, status `requires_external_harness`, no synthetic pass/fail;
  - LiveCodeBench lite v6 pinned to `livecodebench/code_generation_lite` at `0fe84c3912ea0c4d4a78037083943e8f0c4dd505`, raw `test6.jsonl`, stdin exact scoring over public and private tests.
- Ran a LiveCodeBench v6 smoke on `abc387_b`: Q3 and Q4 both passed `1/1`, each with `43/43` stdin tests checked, under `artifacts/qwen3.6-27b-benchmark-subsets-livecodebench-smoke/`.
- Ran practical Q3/Q4 mini-subsets: `Q3_K_M` `39/68`, `Q4_K_M` `39/68`.
- Generated the practical degradation report under `artifacts/qwen3.6-27b-degradation-report-practical/`.
- Refreshed `zlaabsi/Qwen3.6-27B-OTQ-GGUF` with:
  - hardware compatibility;
  - Q5 pending status at first, then local Q5-ready staging after validation;
  - practical mini-subset scores;
  - no fake BF16 degradation claim.
- Cleaned only regenerable Hugging Face caches and recovered disk to about 80 GiB free.
- Generated `Q5_K_M` and completed local release gates:
  - smoke validation: passed;
  - runtime recheck with `qwen3-no-think`: bounded generation passed with `Paris`;
  - `llama-bench` 8K: `pp8192` `102.51 +/- 1.23` t/s and `tg128` `8.97 +/- 0.11` t/s in `artifacts/release-audit/runtime-Q5_K_M-M1_Max_32GB-qwen3-no-think.json`;
  - release validation: passed with `pp8192` `93.94` t/s and `tg128` `8.87` t/s in `artifacts/qwen3.6-27b-dynamic-validation/Qwen3.6-27B-OTQ-DYN-Q5_K_M-GGUF-release-bench.json`;
  - quality eval: `5/5`;
  - release extended eval: `10/10`.
- Regenerated local canonical staging at `artifacts/hf-gguf-canonical/Qwen3.6-27B-OTQ-GGUF` with `Q3_K_M`, `Q4_K_M`, and `Q5_K_M`.
- Uploaded the refreshed canonical staging to `zlaabsi/Qwen3.6-27B-OTQ-GGUF`; remote SHA after verification: `ff2df91ff002d059a8f6c567d257a23566eac9ae`.
- Verified the remote now contains `Qwen3.6-27B-OTQ-DYN-Q5_K_M.gguf`, `evidence/Q5_K_M/validation.json`, `evidence/Q5_K_M/quality-eval.json`, and `evidence/Q5_K_M/release-eval.json`.
- Re-staged Packed and Metal runtime repos locally. Packed remains `public_release_ready=false`; Metal has TQ3_SB4 validation evidence but remains `public_release_ready=false` pending runtime loader/kernel readiness.
- Ran the paired BF16 sidecar on Hugging Face Jobs H200 for `Qwen/Qwen3.6-27B`, job `69f22ef7d70108f37ace1773`, and uploaded `runs/69f22ef7d70108f37ace1773/no_think.json` to `zlaabsi/opentq-qwen36-bf16-sidecar`.
- Ran the matching local GGUF paired mini-subset for `Q3_K_M`, `Q4_K_M`, and `Q5_K_M` under `artifacts/qwen3.6-27b-benchmark-subsets-release-core/`.
- Generated the paired BF16-vs-GGUF report under `artifacts/qwen3.6-27b-paired-bf16-quant-report/`: BF16 `40/52`, Q3 `41/52`, Q4 `39/52`, Q5 `41/52` on the same pinned no-think mini-subset.

## Cleanup Decision

Deleted:

- `~/.cache/huggingface/hub/models--Qwen--Qwen3.6-27B`;
- `~/.cache/huggingface/xet`;
- `~/.cache/huggingface/hub/models--BAAI--bge-m3`.
- remaining small regenerable HF model caches after disk dipped below the Q5 threshold.

Preserved:

- `artifacts/qwen3.6-27b-source/Qwen3.6-27B-BF16.gguf`;
- canonical GGUF staging;
- dynamic GGUF source artifacts;
- Packed/Metal runtime staging.

The decision is documented in `docs/qwen36-disk-cleanup-arbitrage.md`.

## Still Gated

- Q5 HF publication: complete and verified on `zlaabsi/Qwen3.6-27B-OTQ-GGUF`.
- SWE-bench: adapter is pinned, but public pass/fail still requires the official external SWE-bench harness.
- LiveCodeBench: adapter is pinned for v6 stdin tasks; official-delta publication still requires review that the selected subset matches the model-card v6 protocol.
- MT-Bench, Chatbot Arena style, AlpacaEval: require a pinned judge setup.
- MMMU and MathVista: blocked for the current text-only GGUF track.
- Packed: not public-release-ready until runtime/tooling is public and validated.
- Metal/custom GGUF: not public-release-ready until loader and Metal kernels are validated.

## Public Claim Boundary

The HF card may claim stock GGUF usability, M1 Max measured runtime gates, Q5 local validation, and paired practical no-think mini-subset deltas. It must not present the 52-sample sidecar as a full official benchmark replacement or compare its MMLU-Pro/GPQA percentages directly to Qwen's full-harness official scores.
