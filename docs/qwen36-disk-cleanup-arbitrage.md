# Qwen3.6 Disk Cleanup Arbitrage

This note records the cleanup decision for the Qwen3.6 release workspace.

## Current Constraint

The Mac has less than 20 GiB free. That is below the threshold for:

- generating `Q5_K_M`;
- running larger benchmark subsets;
- staging release refreshes with comfortable rollback space;
- keeping partial outputs safe when a command fails.

## Options

| Option | Immediate disk gain | Future option value | Regret if wrong | Decision |
| --- | ---: | --- | --- | --- |
| Keep all Hugging Face caches | 0 GiB | Keeps local safetensors cache for possible BF16 work | Blocks Q5 and larger jobs now | Reject |
| Delete local BF16 GGUF source | ~50 GiB | Removes the source needed for Q5 and re-quantization | High: Q5 becomes harder and more expensive | Reject |
| Delete regenerable HF caches only | ~60 GiB | Requires re-download if needed later | Moderate: network time only | Accept |
| Delete staged GGUF / runtime artifacts | 30-125 GiB apparent | Risks losing release evidence or upload candidates | High until HF verification is repeated | Reject for now |

## Chosen Strategy

Delete only regenerable caches:

- `~/.cache/huggingface/hub/models--Qwen--Qwen3.6-27B`;
- `~/.cache/huggingface/xet`;
- `~/.cache/huggingface/hub/models--BAAI--bge-m3`.

Preserve:

- `artifacts/qwen3.6-27b-source/Qwen3.6-27B-BF16.gguf`;
- `artifacts/hf-gguf-canonical/Qwen3.6-27B-OTQ-GGUF`;
- `artifacts/qwen3.6-27b-dynamic-gguf`;
- `artifacts/hf-runtime` until Packed/Metal gates are reviewed.

## Rationale

This is a minimax/regret decision. The cache deletion unlocks the next valuable actions now while keeping the expensive local BF16 GGUF artifact. If a BF16 sidecar is later required, the deleted safetensors cache can be re-downloaded; if Q5 or HF staging is blocked today, the release stalls immediately.
