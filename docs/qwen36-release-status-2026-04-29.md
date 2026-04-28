# Qwen3.6 Release Status 2026-04-29

## Completed

- Pushed `main` through the current release work.
- Implemented pinned benchmark adapters for the text-only Phase 4 set.
- Ran practical Q3/Q4 mini-subsets: `Q3_K_M` `39/68`, `Q4_K_M` `39/68`.
- Generated the practical degradation report under `artifacts/qwen3.6-27b-degradation-report-practical/`.
- Refreshed `zlaabsi/Qwen3.6-27B-OTQ-GGUF` with:
  - hardware compatibility;
  - Q5 pending status;
  - practical mini-subset scores;
  - no fake BF16 degradation claim.
- Cleaned only regenerable Hugging Face caches and recovered disk to about 80 GiB free.

## Cleanup Decision

Deleted:

- `~/.cache/huggingface/hub/models--Qwen--Qwen3.6-27B`;
- `~/.cache/huggingface/xet`;
- `~/.cache/huggingface/hub/models--BAAI--bge-m3`.

Preserved:

- `artifacts/qwen3.6-27b-source/Qwen3.6-27B-BF16.gguf`;
- canonical GGUF staging;
- dynamic GGUF source artifacts;
- Packed/Metal runtime staging.

The decision is documented in `docs/qwen36-disk-cleanup-arbitrage.md`.

## Still Gated

- `Q5_K_M`: disk-unblocked, but not generated or release-valid yet.
- SWE-bench and LiveCodeBench: require real harness adapters.
- MT-Bench, Chatbot Arena style, AlpacaEval: require a pinned judge setup.
- MMMU and MathVista: blocked for the current text-only GGUF track.
- Packed: not public-release-ready until runtime/tooling is public and validated.
- Metal/custom GGUF: not public-release-ready until loader and Metal kernels are validated.

## Public Claim Boundary

The HF card may claim stock GGUF usability, M1 Max measured runtime gates, and practical OTQ mini-subset scores. It must not claim full benchmark parity or BF16 degradation unless the matching benchmark setup is run.
