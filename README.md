# OpenTQ

`OpenTQ` is an open quantization lab for low-bit weight formats inspired by the 2026 TurboQuant line of work, but designed as a public family with explicit specs, reproducible tooling, and two runtime targets:

- `llama.cpp` via GGUF export plus patchsets
- `opentq-metal`, an Apple Silicon runtime built around compressed-domain decode, KV compression, and optional speculative decoding

## Why this repo exists

The public `TQ3_4S` ecosystem proves that low-bit WHT-rotated weight formats can land at an attractive quality/size point, but the public runtime forks do not ship the quantizer itself. `OpenTQ` fills that gap with an auditable family of variants, benchmark scripts, and a quantizer core you can extend.

## Family goals

- multiple formats, not a single `TQ3`
- explicit naming and stable specs
- `llama.cpp` compatibility as a first-class target
- Apple Silicon runtime work that can go beyond `llama.cpp`
- room for DFlash and compressed-domain attention on the runtime side

## Initial family

| Variant | Intent | Notes |
| --- | --- | --- |
| `TQ1_0` | minimum footprint | ternary / near-ternary baseline |
| `TQ2_0` | very low memory | 2-bit scalar path |
| `TQ3_SB4` | compact general-purpose | 3-bit WHT with four sub-block scales |
| `TQ4_SB4` | daily driver | 4-bit WHT with four sub-block scales |
| `TQ4R2` | quality-first 6-bit total | 4+2 residual quantization |
| `TQ4R4` | near-lossless 8-bit total | 4+4 residual quantization |
| `TQ_BAL_DENSE` | dense-hybrid mixed profile | model-aware release recipe for Qwen3.6-27B-class dense models |
| `TQ_MIX_MOE` | MoE-aware release profile | tensor-role-aware mixed precision |

`SB4` means "4 sub-block scales per block". The naming is deliberate: it describes the format instead of inheriting an opaque revision suffix.

## Current status

This initial commit gives you:

- a working Python package with:
  - Gaussian Lloyd-Max codebook generation
  - fast Walsh-Hadamard rotation
  - block quantization and residual quantization for `.npy` tensors
  - a CLI to inspect variants, estimate size, quantize demo tensors, inspect a Hugging Face safetensors index, print a model release matrix, and run a full release quantization over HF safetensors
- architecture docs for:
  - the `llama.cpp` path
  - the Apple Silicon runtime path
  - the March/April 2026 landscape that informed the design

What it does **not** do yet:

- end-to-end Hugging Face model conversion
- GGUF tensor emission
- Metal kernels for the custom runtime
- model-wide calibration and tensor-role heuristics

## Quick start

```bash
uv sync
uv run opentq variants
uv run opentq plan TQ4R2 --shape 8192 8192
uv run opentq quantize weights.npy --variant TQ3_SB4 --output artifacts/q_proj
uv run opentq recipe qwen3.6-27b --format markdown
uv run opentq inventory --model-id Qwen/Qwen3.6-27B
uv run opentq release-plan --recipe qwen3.6-27b --release Qwen3.6-27B-TQ4_SB4
uv run opentq quantize-release --recipe qwen3.6-27b --release Qwen3.6-27B-TQ4_SB4 --output artifacts/qwen36-tq4sb4 --max-tensors 8
```

For unattended overnight runs on Apple Silicon, launch the resumable Qwen3.6-27B batch with:

```bash
./scripts/launch_qwen36_quantizations.sh
python ./scripts/status_qwen36_quantizations.py
uv run opentq status
uv run opentq status --watch
```

## Repo layout

```text
src/opentq/          quantizer core and CLI
tests/               unit tests for codebooks and quantization logic
docs/                architecture, variants, research notes
patches/llama.cpp/   planned upstream integration notes and patch strategy
```

## Next steps

1. add tensor-role-aware calibration over Hugging Face safetensors
2. emit GGUF custom tensor payloads for `llama.cpp`
3. land Metal decode kernels for `TQ3_SB4`, `TQ4_SB4`, and `TQ4R2`
4. benchmark on M1 Max, M2 Max, M3 Max, and M4 Max with long-context agentic prompts
