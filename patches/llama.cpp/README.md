# llama.cpp patch strategy

This directory will hold the narrow patchsets that teach `llama.cpp` how to:

- recognize `OpenTQ` GGUF tensor types
- allocate the right scale / index payload views
- dispatch Metal kernels for `TQ3_SB4`, `TQ4_SB4`, and `TQ4R2`

The deliberate choice is to keep the patches small and reviewable before creating a long-lived public fork.

## Current bridge artifact

`opentq pack-release` writes `.otq` tensor payloads and `opentq gguf-plan` writes the `gguf-plan.json` contract consumed by this patch track.

The GGUF patch must implement:

- `OPENTQ_*` entries in `ggml_type`
- type traits for block size and byte size
- CPU reference `dequantize_row` paths
- Metal unpack/dequant kernels
- a writer that embeds `.otq` sections as GGUF tensor payloads
- loader checks for `opentq.schema`, `opentq.release_slug`, and `general.quantization_version`
