# KV Cache Layer Policy Roadmap

OpenTQ currently optimizes **weights** by tensor family. The next complementary axis is the KV cache: optimize **runtime state** by layer.

## Why This Matters

Weight quantization reduces model storage and memory load. KV-cache quantization reduces long-context memory pressure and can improve decode throughput. These two decisions interact: a model with aggressive weight compression may need more conservative KV cache precision in sensitive layers, while a quality-first weight profile can spend less KV precision in stable layers.

OpenTQ should expose both decisions in one auditable policy:

```yaml
weights:
  profile: OTQ-DYN-Q4_K_M

kv_cache:
  default_dtype: fp8
  layer_overrides:
    0: bf16
    1: bf16
    45: bf16
    46: bf16
  attention_group_overrides:
    linear_attention_state: bf16
```

## First Milestone

Add a planner-only KV policy artifact:

| Output | Purpose |
| --- | --- |
| `kv-cache-policy.json` | Resolved per-layer KV dtype plan. |
| `kv-cache-policy.tsv` | Human-readable layer table. |
| `kv-cache-rationale.md` | Why layers were promoted or compressed. |

This does not require a new runtime yet. It makes policy design explicit and lets later vLLM/llama.cpp/native Metal integrations consume a stable file.

## Runtime Targets

| Runtime | Candidate Integration |
| --- | --- |
| vLLM | Hybrid FP8/BF16 layer skip policy, similar in spirit to `--kv-cache-dtype-skip-layers`. |
| llama.cpp | Future per-layer KV cache type hooks if exposed upstream or in OpenTQ fork. |
| OpenTQ Metal-native | Direct mixed-precision KV allocation coupled to custom kernels. |

## Metrics

- Prefill throughput at 8K, 16K, 32K contexts.
- Decode throughput at fixed context.
- Peak unified memory.
- Long-context quality samples.
- Policy trace: layer, dtype, reason, measured sensitivity.

## Release Rule

Do not claim KV-cache quality preservation from weight-only results. KV cache needs paired runs with identical prompts, context length, runtime, and scoring.
