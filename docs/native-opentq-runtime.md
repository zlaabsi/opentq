# Native OpenTQ Runtime Track

OpenTQ has two runtime paths for Qwen3.6-27B:

1. **Stock GGUF release**: public `Q3_K_M`, `Q4_K_M`, and `Q5_K_M` files that run in stock `llama.cpp`.
2. **Native OpenTQ runtime track**: packed `.otq` payloads and custom OpenTQ GGUF tensors that are meant for custom runtime kernels.

The stock GGUF path is the current public user-facing release. The native path is where OpenTQ-specific efficiency work happens: custom tensor payloads, runtime probes, Metal kernels, and eventually compressed-domain execution.

## Current Runtime Evidence

The native runtime gate script is:

```bash
./scripts/run_qwen36_native_runtime_gates.sh
```

It runs two classes of checks:

| Gate | Artifact | What It Proves |
| --- | --- | --- |
| Packed probe | `artifacts/hf-runtime/Qwen3.6-27B-OTQ-Packed/*/opentq-pack.json` | manifest integrity, tensor checksums, Python dequantization, and C++ dequant/dot fixture parity |
| Metal smoke/bench | `artifacts/hf-runtime/Qwen3.6-27B-OTQ-Metal-GGUF/Qwen3.6-27B-OTQ-TQ3_SB4-Metal.gguf` | custom OpenTQ GGUF metadata read, bounded generation, and optional `llama-bench` execution on Metal |

Run with the short Metal benchmark enabled:

```bash
RUN_METAL_BENCH=1 ./scripts/run_qwen36_native_runtime_gates.sh
```

The latest local gate covered these packed releases:

| Packed Release | Quantized Variant Mix | Runtime Gate |
| --- | --- | --- |
| `Qwen3.6-27B-OTQ-TQ3_SB4` | `TQ3_SB4` | packed probe passed |
| `Qwen3.6-27B-OTQ-TQ4_SB4` | `TQ4_SB4` | packed probe passed |
| `Qwen3.6-27B-OTQ-TQ4R2` | `TQ4R2` | packed probe passed |
| `Qwen3.6-27B-OTQ-TQ4R4` | `TQ4R4` | packed probe passed |
| `Qwen3.6-27B-OTQ-TQ4_BAL_V2` | `TQ4_SB2` + `TQ4R2` | packed probe passed |

The first custom Metal GGUF gate covered:

| Artifact | Runtime | Gate |
| --- | --- | --- |
| `Qwen3.6-27B-OTQ-TQ3_SB4-Metal.gguf` | patched `llama.cpp` with Metal and FlashAttention | metadata read, bounded generation, and short benchmark passed |

## What The Packed Probe Checks

`opentq probe-pack-runtime` is the bridge between the Python quantizer and an external runtime implementation.

It performs:

- `.otq` manifest schema validation
- selected tensor SHA256 checks against `opentq-pack.json`
- Python dequantization for selected tensors
- finite-value and shape checks
- runtime fixture export: packed block bytes, expected fp32 decode, activation vector, and expected dot product
- external C++ probe execution for `dequant` and `dot`

Example:

```bash
uv run opentq probe-pack-runtime \
  --packed artifacts/hf-runtime/Qwen3.6-27B-OTQ-Packed/Qwen3.6-27B-OTQ-TQ4_BAL_V2 \
  --fixtures-output artifacts/runtime-gates/TQ4_BAL_V2/fixtures \
  --probe-binary ../llama.cpp/build/bin/opentq-dequant-probe \
  --output artifacts/runtime-gates/TQ4_BAL_V2/pack-runtime-probe.json
```

## What Is Not Release-Ready Yet

The native track is not the same as the stock GGUF release.

Before publishing Packed or Metal-native artifacts as user-facing releases, OpenTQ still needs:

- a documented public runtime installation path
- a stable runtime repository or patchset users can build without local-only assumptions
- longer Metal gates beyond short bounded generation
- custom GGUF export coverage beyond the current `TQ3_SB4` Metal artifact
- agentic workload tests that match the intended daily harness use case

Until those gates are met, the native track should be described as runtime evidence and research infrastructure, not as a polished public inference release.

