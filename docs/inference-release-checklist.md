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

## Stage 2: private OpenTQ staging

This stage is useful for internal testing only. Do not publish these folders yet.

Prepare one staged OpenTQ repository folder:

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

## Stage 3: GGUF public artifacts

`gguf-plan.json` is a bridge contract. Public releases should use real `.gguf` files generated from the packed OpenTQ payloads.

Export one GGUF:

```bash
uv run python -m opentq.cli export-gguf \
  --packed artifacts/qwen3.6-27b-packed/Qwen3.6-27B-TQ4_BAL_V2 \
  --output artifacts/qwen3.6-27b-gguf/Qwen3.6-27B-TQ4_BAL_V2/Qwen3.6-27B-TQ4_BAL_V2.gguf \
  --llama-cpp /Users/zlaabsi/Documents/GitHub/llama.cpp
```

Export all GGUF releases in a detached session:

```bash
./scripts/launch_qwen36_gguf_exports.sh
```

Stage public Hugging Face GGUF folders:

```bash
HF_USER=zlaabsi ./scripts/stage_qwen36_gguf_releases.sh
```

Upload a staged GGUF release:

```bash
hf upload-large-folder zlaabsi/Qwen3.6-27B-TQ4_BAL_V2-GGUF artifacts/qwen3.6-27b-hf-gguf/Qwen3.6-27B-TQ4_BAL_V2
```

Runtime status:

- OpenTQ tensor types are registered in the patched `llama.cpp` fork
- GGUF metadata and custom typed tensor payloads are emitted by `opentq export-gguf`
- CPU reference dequant / vec-dot paths are available for correctness and fallback
- Metal builds with the patch; optimized compressed-domain Metal kernels remain a later performance pass
- Stock upstream `llama.cpp` remains unsupported until the patchset is upstreamed
- Current GGUF export is text-only; vision tensors are intentionally skipped

## Recommended public release order

1. `Qwen3.6-27B-TQ4_BAL_V2`
2. `Qwen3.6-27B-TQ3_SB4`
3. `Qwen3.6-27B-TQ4_SB4`
4. `Qwen3.6-27B-TQ4R2`
5. `Qwen3.6-27B-TQ4R4`

`TQ4_BAL_V2` should be the flagship first because it is model-aware and targets the useful 32 GB Mac class.
