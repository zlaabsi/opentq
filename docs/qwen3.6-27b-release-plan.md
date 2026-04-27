# Qwen3.6-27B Release Plan

This is the model-specific execution plan for `Qwen/Qwen3.6-27B`.

## Why this model needs a custom plan

`Qwen3.6-27B` is not a plain dense transformer stack. It is a dense hybrid model with a repeating layout of three linear-attention blocks followed by one full gated-attention block. That changes both the weight-quantization risk profile and the runtime plan:

- weight policies do not need to be uniform across all blocks
- KV-cache work only applies to the gated-attention layers
- speculative decoding integration is possible, but hybrid-cache rollback is more complex than on plain full-attention models

## Release Matrix

| Priority | Release | Kind | Target size | Goal |
| --- | --- | --- | --- | --- |
| 1 | `Qwen3.6-27B-TQ4_SB4` | uniform | `~19.2-19.6 GiB` | uniform 4-bit quality baseline |
| 2 | `Qwen3.6-27B-TQ3_SB4` | uniform | `~16.0-16.3 GiB` | compact public 3-bit-family release |
| 3 | `Qwen3.6-27B-TQ4R2` | residual | `~32.2-32.5 GiB` | quality-first release |
| 4 | `Qwen3.6-27B-TQ4_SB2` | uniform | `~16.0-16.3 GiB` | redesigned 4-bit release for the 16 GiB class |
| 5 | `Qwen3.6-27B-TQ4_BAL_V2` | mixed | `~18.6-18.9 GiB` | flagship dense-hybrid profile |
| 6 | `Qwen3.6-27B-TQ2_0` | uniform | `~9.6-9.8 GiB` | aggressive memory floor |
| 7 | `Qwen3.6-27B-TQ4R4` | residual | `~38.6-39.0 GiB` | near-lossless reference |
| 8 | `Qwen3.6-27B-TQ1_0` | uniform | `~6.3-6.6 GiB` | research lower bound |
| 9 | `Qwen3.6-27B-OTQ-GGUF` | dynamic-compatible | staged from validated artifacts | canonical stock llama.cpp public repo with multiple GGUF files |
| 10 | `Qwen3.6-27B-OTQ-Packed` | OpenTQ packed weights | pack manifest required | separate `.otq` OpenTQ/TurboQuant artifact repo |
| 11 | `Qwen3.6-27B-OTQ-Metal-GGUF` | custom-runtime GGUF | runtime gate required | separate OpenTQ/Metal GGUF release track |

## Mixed flagship policy

`Qwen3.6-27B-TQ4_BAL_V2` is the model-aware redesign for the dense-hybrid stack:

- embeddings and `lm_head`: `TQ4R2`
- full-attention projections: `TQ4R2`
- linear-attention projections: `TQ4_SB2`
- MLP projections: `TQ4_SB2`
- visual tower weights: `TQ4_SB2`
- norms, state scalars, bias-like tensors: keep high precision

The hypothesis is simple: pay residual bits where routing and output sensitivity are highest, while moving the bulk MLP and linear-attention weights to the lighter `TQ4_SB2` path. The residual path stores both primary scales and residual scales, so the packed target is closer to the 19 GiB class than the original 16 GiB planning target.

## Runtime split

### Dynamic-compatible stock GGUF track

- emits normal GGUFs with only standard llama.cpp tensor types
- uses `llama-quantize --tensor-type-file` for per-tensor allocation
- does not require `llama.cpp-opentq`, custom GGML types, or OpenTQ Metal kernels
- is the correct first public Hugging Face release path while native OpenTQ kernels are not release-grade
- keeps the same smoke, quality, and long-context wall-clock gates as native OpenTQ

See `docs/dynamic-compatible-gguf.md`.

### llama.cpp track

- weight quantization first
- KV kept separate from weight-format claims
- baseline KV policy should start simple on this hybrid architecture

### opentq-metal track

- int4 compressed-domain KV first
- DFlash after the base runtime is stable
- DDTree only after DFlash, because hybrid models constrain tree verification gains

## Overnight order

1. tensor inventory and calibration set
2. `TQ4_SB4`
3. `TQ3_SB4`
4. `TQ4R2`
5. `TQ4R4`
6. `TQ4_BAL_V2`
7. dynamic-compatible stock GGUF profiles
8. runtime patchsets

## CLI

```bash
uv run opentq recipe qwen3.6-27b --format markdown
uv run opentq inventory --model-id Qwen/Qwen3.6-27B
uv run opentq dynamic-gguf-profiles
uv run python scripts/stage_qwen36_otq_gguf_repo.py
uv run python scripts/stage_qwen36_otq_runtime_repos.py
```
