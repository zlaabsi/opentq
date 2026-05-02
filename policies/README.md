# OpenTQ Policy Files

Policy files let users define custom dynamic allocation without editing OpenTQ source.

```bash
uv run opentq dynamic-gguf-plan \
  --policy-file policies/qwen36-custom-dyn-q4.yaml \
  --output artifacts/qwen36-custom-q4 \
  --llama-cpp /path/to/llama.cpp
```

Supported formats: YAML and JSON.

Core fields:

| Field | Meaning |
| --- | --- |
| `name` | policy name written into the generated plan |
| `base_ftype` | fallback GGUF quantization type passed to `llama-quantize` |
| `target` | human-readable target such as `custom 32GB Apple Silicon profile` |
| `requires_imatrix` | whether the policy needs an imatrix |
| `category_types` | default GGUF tensor type per tensor family |
| `edge_layers` | number of first/last layers promoted by `edge_overrides` |
| `edge_overrides` | precision overrides for first/last layers |
| `periodic_stride` | stride for periodic layer promotion |
| `periodic_overrides` | precision overrides applied at the stride |

The stock-compatible GGUF track customizes allocation across standard `llama.cpp` tensor types. New OpenTQ kernels belong to the native runtime track.
