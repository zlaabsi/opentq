

`OpenTQ` is an open quantization lab for low-bit weight formats inspired by the 2026 TurboQuant line of work, but designed as a public family with explicit specs, reproducible tooling, and three runtime/release targets:

- stock `llama.cpp` via dynamic-compatible GGUF allocation over existing tensor types
- `llama.cpp-opentq` via GGUF export plus custom OpenTQ tensor patchsets
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
| `TQ4_SB2` | balanced 16 GiB path | 4-bit WHT with two sub-block scales |
| `TQ4_SB4` | daily driver | 4-bit WHT with four sub-block scales |
| `TQ4R2` | quality-first 6-bit total | 4+2 residual quantization |
| `TQ4R4` | near-lossless 8-bit total | 4+4 residual quantization |
| `TQ4_BAL_V2` | dense-hybrid mixed profile | model-aware flagship recipe built around `TQ4_SB2` + `TQ4R2` |
| `TQ_MIX_MOE` | MoE-aware release profile | tensor-role-aware mixed precision |

`SB4` means "4 sub-block scales per block". The naming is deliberate: it describes the format instead of inheriting an opaque revision suffix.

## Current status

This initial commit gives you:

- a working Python package with:
  - Gaussian Lloyd-Max codebook generation
  - fast Walsh-Hadamard rotation
  - block quantization and residual quantization for `.npy` tensors
  - a CLI to inspect variants, estimate size, quantize demo tensors, inspect a Hugging Face safetensors index, print a model release matrix, create stock-compatible dynamic GGUF plans, and run a full release quantization over HF safetensors
- architecture docs for:
  - the `llama.cpp` path
  - the Apple Silicon runtime path
  - the March/April 2026 landscape that informed the design

What it does **not** do yet:

- upstream stock `llama.cpp` support for OpenTQ tensor types
- optimized Metal kernels for the custom runtime
- model-wide KLD calibration beyond the initial tensor-role dynamic heuristics

## Quick start

```bash
uv sync
uv run opentq variants
uv run opentq plan TQ4R2 --shape 8192 8192
uv run opentq quantize weights.npy --variant TQ3_SB4 --output artifacts/q_proj
uv run opentq recipe qwen3.6-27b --format markdown
uv run opentq inventory --model-id Qwen/Qwen3.6-27B
uv run opentq dynamic-gguf-profiles
uv run opentq dynamic-gguf-plan --profile OTQ-DYN-Q4_K_M --output artifacts/qwen36-otq-dyn-q4-k-m --llama-cpp /Users/zlaabsi/Documents/GitHub/llama.cpp
uv run python scripts/build_qwen36_release_report.py
./scripts/launch_qwen36_dynamic_ggufs.sh
./scripts/status_qwen36_dynamic_ggufs.sh
uv run opentq release-plan --recipe qwen3.6-27b --release Qwen3.6-27B-TQ4_BAL_V2
uv run opentq quantize-release --recipe qwen3.6-27b --release Qwen3.6-27B-TQ4_BAL_V2 --output artifacts/qwen36-tq4balv2 --max-tensors 8
uv run opentq pack-release --input artifacts/qwen36-tq4balv2 --output artifacts/qwen36-tq4balv2-packed
uv run opentq gguf-plan --packed artifacts/qwen36-tq4balv2-packed --output artifacts/qwen36-tq4balv2-packed/gguf-plan.json
uv run opentq export-gguf --packed artifacts/qwen36-tq4balv2-packed --output artifacts/gguf/Qwen3.6-27B-TQ4_BAL_V2.gguf --llama-cpp /Users/zlaabsi/Documents/GitHub/llama.cpp
uv run opentq prepare-hf-gguf --gguf artifacts/gguf/Qwen3.6-27B-TQ4_BAL_V2.gguf --output artifacts/hf-gguf/qwen36-tq4balv2 --repo-id zlaabsi/Qwen3.6-27B-TQ4_BAL_V2-GGUF
```

For unattended overnight runs on Apple Silicon, launch the resumable Qwen3.6-27B batch with:

```bash
./scripts/launch_qwen36_quantizations.sh
python ./scripts/status_qwen36_quantizations.py
uv run opentq status
uv run opentq status --watch
uv run opentq monitor
uv run opentq monitor --watch
```

`opentq monitor --watch` uses a full-screen Rich TUI with colored panels, live progress bars, current-tensor inspection, recent timeline, and category aggregates.

For the release path after quantization:

```bash
./scripts/pack_qwen36_releases.sh
./scripts/launch_qwen36_gguf_exports.sh
HF_USER=zlaabsi ./scripts/stage_qwen36_gguf_releases.sh
```

The OpenTQ `.otq` packs are the `OTQ-Packed` release track. Public stock-compatible Hugging Face releases should use `Qwen3.6-27B-OTQ-GGUF`; custom OpenTQ GGUF artifacts remain tied to `llama.cpp-opentq` and belong in the `OTQ-Metal-GGUF` track once the custom tensor types and Metal kernels are release-grade.

See [inference-release-checklist.md](/Users/zlaabsi/Documents/GitHub/opentq/docs/inference-release-checklist.md) for the OpenTQ packed format, GGUF staging, and runtime validation flow.
See [dynamic-compatible-gguf.md](/Users/zlaabsi/Documents/GitHub/opentq/docs/dynamic-compatible-gguf.md) for the stock-compatible GGUF path.

## Repo layout

```text
src/opentq/          quantizer core and CLI
tests/               unit tests for codebooks and quantization logic
docs/                architecture, variants, research notes
patches/llama.cpp/   planned upstream integration notes and patch strategy
```

## Next steps

1. add tensor-role-aware calibration over Hugging Face safetensors
2. release stock-compatible dynamic GGUFs with standard llama.cpp tensor types
3. land Metal decode kernels for `TQ3_SB4`, `TQ4_SB4`, and `TQ4R2`
4. benchmark on M1 Max, M2 Max, M3 Max, and M4 Max with long-context agentic prompts
