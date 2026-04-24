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
| 1 | `Qwen3.6-27B-TQ4_SB4` | uniform | `~15.5-16.8 GiB` | first-wave balanced public release |
| 2 | `Qwen3.6-27B-TQ3_SB4` | uniform | `~12.6-13.3 GiB` | compact 32 GB Mac release |
| 3 | `Qwen3.6-27B-TQ4R2` | residual | `~18.5-20.5 GiB` | quality-first release |
| 4 | `Qwen3.6-27B-TQ_BAL_DENSE` | mixed | `~14.5-16.0 GiB` | flagship dense-hybrid profile |
| 5 | `Qwen3.6-27B-TQ2_0` | uniform | `~8.0-9.5 GiB` | aggressive memory floor |
| 6 | `Qwen3.6-27B-TQ4R4` | residual | `~24.0-28.0 GiB` | near-lossless reference |
| 7 | `Qwen3.6-27B-TQ1_0` | uniform | `~5.5-6.5 GiB` | research lower bound |

## Mixed flagship policy

`Qwen3.6-27B-TQ_BAL_DENSE` is the only profile here that is explicitly model-aware:

- embeddings and `lm_head`: `TQ4R4`
- full-attention projections: `TQ4R2`
- linear-attention projections: `TQ4_SB4`
- MLP projections: `TQ3_SB4`
- visual tower weights: `TQ4_SB4`
- norms, state scalars, bias-like tensors: keep high precision

The hypothesis is simple: pay bits where routing and output sensitivity are highest, save bits where the model is over-parameterized.
For the complete multimodal release, the visual tower stays on a conservative path instead of inheriting the more aggressive text-tower policy.

## Runtime split

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
5. `TQ_BAL_DENSE`
6. `TQ2_0`
7. `TQ4R4`
8. `TQ1_0`
9. runtime patchsets

## CLI

```bash
uv run opentq recipe qwen3.6-27b --format markdown
uv run opentq inventory --model-id Qwen/Qwen3.6-27B
```
