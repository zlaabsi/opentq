# OpenTQ Packed And Metal GGUF Release Plan

## Naming Direction

Do not use `Native` as the public repo suffix. It is technically accurate but does not explain the user value.

Recommended public repo family:

- `Qwen3.6-27B-OTQ-Packed`
- `Qwen3.6-27B-OTQ-Metal-GGUF`

Meaning:

- `OTQ-Packed`: OpenTQ bit-packed `.otq` payloads plus `opentq-pack.json`.
- `OTQ-Metal-GGUF`: custom OpenTQ GGUFs that require the OpenTQ/Metal runtime path.

Avoid:

- `Qwen3.6-27B-OTQ-Native`
- `Qwen3.6-27B-OTQ-TurboQuant`
- `Qwen3.6-27B-TQ-*` as the primary repo brand, because `TQ` conflicts with existing third-party TurboQuant naming such as `TQ3_4S`.

## Public Meaning

- `OTQ`: OpenTQ brand and toolchain.
- `TurboQuant`: explicit algorithm family in model cards and visual assets for users who do not know the abbreviation.
- Variant suffixes such as `TQ4_SB4`, `TQ4R2`, `TQ4_BAL_V2`: native OpenTQ/TurboQuant weight profiles.

## Separation From The GGUF Repo

`Qwen3.6-27B-OTQ-GGUF` is the stock-compatible repo. It must contain only normal llama.cpp GGUF tensor types.

`Qwen3.6-27B-OTQ-Packed` can contain OpenTQ `.otq` payloads. It is a format release, not a stock inference release.

`Qwen3.6-27B-OTQ-Metal-GGUF` can contain custom-runtime GGUFs, but only after:

- smoke generation passes;
- long-context wall-clock benchmarks pass;
- quality suite and extended release suite pass;
- runtime compatibility is explicitly documented;
- Metal kernels are reliable enough for a public release;
- a clear install path exists for the required runtime.

## Current Staging

The runtime repos are staged locally, not published:

```bash
uv run python scripts/stage_qwen36_otq_runtime_repos.py
```

Outputs:

- `artifacts/hf-runtime/Qwen3.6-27B-OTQ-Packed`
- `artifacts/hf-runtime/Qwen3.6-27B-OTQ-Metal-GGUF`

`OTQ-Packed` includes the generated `.otq` tensor packs and manifests for all completed packed variants.

`OTQ-Metal-GGUF` includes only `TQ3_SB4` as a release candidate because it is the only custom GGUF with complete Metal validation evidence so far. `TQ4_BAL_V2` and `TQ4_SB4` remain blocked until validation/export issues are resolved.

## Candidate Packed And Metal Releases

| Variant | Role | Current status |
| --- | --- | --- |
| `TQ3_SB4` | compact native TurboQuant release | packed; Metal GGUF smoke validated; long-context gate still required |
| `TQ4_SB4` | uniform 4-bit baseline | packed; GGUF export size inconsistent, needs audit |
| `TQ4R2` | residual quality-first release | packed; runtime validation required |
| `TQ4_BAL_V2` | model-aware mixed flagship | packed; GGUF exists but needs Metal validation |
| `TQ4R4` | reference-grade regression target | packed; large, high-end validation only |

## Release Gate

Packed and Metal GGUF releases must not be published as ordinary stock GGUFs.

They need their own card section:

- required runtime;
- unsupported runtimes;
- exact file format;
- Metal backend status;
- benchmark conditions;
- recommended context/KV settings;
- known limitations.
