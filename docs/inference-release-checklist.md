# Inference and Release Checklist

This checklist tracks the gap between completed OpenTQ quantization runs and runnable/public releases.

## Current completed inputs

The Qwen3.6-27B quantization runs produce intermediate tensor chunks under:

```bash
artifacts/qwen3.6-27b/Qwen3.6-27B-*
```

Those directories are research artifacts: resumable, inspectable, and monitor-friendly, but not final runtime payloads.

## Stage 1: OpenTQ packed payloads

Pack a completed release:

```bash
uv run python -m opentq.cli pack-release \
  --input artifacts/qwen3.6-27b/Qwen3.6-27B-TQ4_BAL_V2 \
  --output artifacts/qwen3.6-27b-packed/Qwen3.6-27B-TQ4_BAL_V2
```

Pack all completed Qwen3.6-27B releases:

```bash
./scripts/pack_qwen36_releases.sh
```

Output:

- `opentq-pack.json`: release index, tensor metadata, offsets, checksums
- `tensors/*.otq`: bit-packed tensor payloads
- `gguf-plan.json`: runtime integration manifest for the GGUF track

Packed payload details:

- primary indices are bit-packed at `weight_bits`
- residual indices are bit-packed at `residual_bits`
- primary scales and residual scales are stored as `float16`
- copied tensors are stored as `float16` by default

## Stage 2: Hugging Face staging

Prepare one staged repository folder:

```bash
uv run python -m opentq.cli prepare-hf \
  --packed artifacts/qwen3.6-27b-packed/Qwen3.6-27B-TQ4_BAL_V2 \
  --output artifacts/qwen3.6-27b-hf/Qwen3.6-27B-TQ4_BAL_V2 \
  --repo-id zlaabsi/Qwen3.6-27B-TQ4_BAL_V2
```

Prepare all packed releases:

```bash
HF_USER=zlaabsi ./scripts/stage_qwen36_hf_releases.sh
```

Upload a staged release:

```bash
hf upload-large-folder zlaabsi/Qwen3.6-27B-TQ4_BAL_V2 artifacts/qwen3.6-27b-hf/Qwen3.6-27B-TQ4_BAL_V2
```

## Stage 3: GGUF

`gguf-plan.json` is not a runnable GGUF file. It is the bridge contract for the `llama.cpp` integration.

Missing runtime work:

- register `OPENTQ_*` tensor types in `ggml_type`
- define type size and block size tables
- implement CPU `dequantize_row` reference paths
- implement Metal unpack/dequant kernels
- write a GGUF emitter that embeds `.otq` payload sections as custom typed tensors
- validate `opentq.*` metadata during load

The GGUF output becomes publishable only after the loader can run the payload. Until then, Hugging Face releases should be labeled as OpenTQ packed releases, not stock `llama.cpp` GGUF releases.

## Recommended public release order

1. `Qwen3.6-27B-TQ4_BAL_V2`
2. `Qwen3.6-27B-TQ3_SB4`
3. `Qwen3.6-27B-TQ4_SB4`
4. `Qwen3.6-27B-TQ4R2`
5. `Qwen3.6-27B-TQ4R4`

`TQ4_BAL_V2` should be the flagship first because it is model-aware and targets the useful 32 GB Mac class.
