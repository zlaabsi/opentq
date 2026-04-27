# llama.cpp patch strategy

This directory tracks the narrow patchset that teaches `llama.cpp` how to:

- recognize `OpenTQ` GGUF tensor types
- allocate the right scale / index payload views
- dispatch CPU reference paths and later optimized Metal kernels for `TQ3_SB4`, `TQ4_SB4`, and `TQ4R2`

The deliberate choice is to keep the patches small and reviewable before creating a long-lived public fork.

## Current bridge artifact

`opentq pack-release` writes private `.otq` tensor payloads. `opentq export-gguf` converts those packs into public GGUF files with custom OpenTQ tensor types.

The active local patch under `/Users/zlaabsi/Documents/GitHub/llama.cpp` implements:

- `OPENTQ_*` entries in `ggml_type`
- type traits for block size and byte size
- CPU reference `dequantize_row` paths
- CPU `Q8_0` vec-dot fallback paths
- GGUF Python constants for custom type IDs
- Qwen3.6 tokenizer mapping in `convert_hf_to_gguf.py`

Still missing for performance:

- compressed-domain Metal matmul kernels
- loader-side hard validation of `opentq.schema`, `opentq.release_slug`, and `opentq.required_runtime`
- upstreamable cleanup of custom type ID allocation
