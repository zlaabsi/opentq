# OpenTQ Dynamic-Compatible GGUF

This track is deliberately separate from native OpenTQ.

Custom-runtime OpenTQ GGUFs store custom `OPENTQ_*` tensor payloads and require `llama.cpp-opentq`. Dynamic-compatible GGUFs store only standard llama.cpp tensor types (`Q3_K`, `Q4_K`, `Q5_K`, `Q6_K`, `Q8_0`, `IQ4_NL`, `F16`, etc.). The “OpenTQ” part is the allocation policy and validation harness, not a new runtime format.

## Why this exists

Public Hugging Face releases need a compatible path while the native OpenTQ Metal runtime is still being optimized. Dynamic-compatible GGUFs can load in stock `llama.cpp`, `llama-server`, and downstream tools that support normal GGUFs.

The tradeoff is clear:

- dynamic-compatible GGUF: no custom runtime, publishable sooner, uses mature Metal kernels
- native OpenTQ GGUF: more algorithmic novelty, smaller/custom payloads, requires custom kernels before public release

## Profiles

| Profile | Base ftype | Intent | Allocation summary |
| --- | --- | --- | --- |
| `OTQ-DYN-Q3_K_M` | `Q3_K_M` | compact 32 GB candidate | Q3 bulk MLP, Q4 attention, Q5/Q6 anchors |
| `OTQ-DYN-Q4_K_M` | `Q4_K_M` | primary public stock-GGUF candidate | Q4 bulk MLP, Q5/Q6 attention, Q8 output head |
| `OTQ-DYN-Q5_K_M` | `Q5_K_M` | quality-first stock-GGUF baseline | Q5 bulk MLP, Q6/Q8 attention and anchors |
| `OTQ-DYN-IQ4_NL` | `IQ4_NL` | calibrated nonlinear 4-bit experiment | IQ4_NL bulk MLP, Q5/Q6/Q8 anchors; imatrix required |

`Q5_K_M` is a future quality-first stock GGUF candidate after disk cleanup and runtime gates.
`IQ4_NL is a stock llama.cpp nonlinear 4-bit quant type`, reported by llama.cpp as roughly 4.5 bpw. In OpenTQ release planning it is a deferred experiment because it requires an imatrix before release consideration.

Each profile also promotes first/last layers and periodic attention anchors. That is the current OpenTQ dynamic policy. Later calibration can replace these heuristics with measured KLD/per-layer sensitivity.

## Generate A Plan

```bash
cd /Users/zlaabsi/Documents/GitHub/opentq

uv run opentq dynamic-gguf-profiles

uv run opentq dynamic-gguf-plan \
  --profile OTQ-DYN-Q4_K_M \
  --output artifacts/qwen3.6-27b-dynamic-gguf/Qwen3.6-27B-OTQ-DYN-Q4_K_M-GGUF \
  --llama-cpp /Users/zlaabsi/Documents/GitHub/llama.cpp \
  --source-gguf artifacts/qwen3.6-27b-source/Qwen3.6-27B-BF16.gguf \
  --target-gguf artifacts/qwen3.6-27b-dynamic-gguf/Qwen3.6-27B-OTQ-DYN-Q4_K_M-GGUF/Qwen3.6-27B-OTQ-DYN-Q4_K_M.gguf
```

Outputs:

- `plan.json`: full allocation plan and compatibility metadata
- `tensor-types.txt`: stock `llama-quantize --tensor-type-file` input, no comments
- `tensor-types.annotated.tsv`: readable tensor allocation table
- `quantize.sh`: runnable stock llama.cpp quantization script

## Quantize

```bash
artifacts/qwen3.6-27b-dynamic-gguf/Qwen3.6-27B-OTQ-DYN-Q4_K_M-GGUF/quantize.sh
```

For the Qwen3.6-27B release path, use the guarded sequential runner instead. By default it converts the BF16 source GGUF if missing, runs only `OTQ-DYN-Q4_K_M`, writes logs, performs a dry-run, quantizes, then smoke-validates:

```bash
./scripts/launch_qwen36_dynamic_ggufs.sh
./scripts/status_qwen36_dynamic_ggufs.sh
```

To run more profiles later:

```bash
PROFILES="OTQ-DYN-Q3_K_M OTQ-DYN-Q5_K_M" ./scripts/launch_qwen36_dynamic_ggufs.sh
```

After a profile has been quantized and smoke-tested, run the gated release pipeline. It waits for the smoke JSON from the quantization runner, then executes quality eval, long-context bench, HF staging, repo creation, and upload:

```bash
PROFILES="OTQ-DYN-Q3_K_M" ./scripts/launch_qwen36_dynamic_release.sh
./scripts/status_qwen36_dynamic_ggufs.sh
```

For the canonical multi-file Hugging Face repo, stage from existing validated artifacts:

```bash
uv run python scripts/stage_qwen36_otq_gguf_repo.py \
  --banner "/Users/zlaabsi/Downloads/ChatGPT Image Apr 28, 2026, 01_45_35 AM.png"
```

This also generates benchmark plots and CSV tables from the evidence JSON.

Disable upload for a local dry release:

```bash
UPLOAD=0 PROFILES="OTQ-DYN-Q3_K_M" ./scripts/release_qwen36_dynamic_gguf.sh
```

Force a quality or benchmark refresh after changing the gate:

```bash
FORCE_QUALITY=1 FORCE_BENCH=1 PROFILES="OTQ-DYN-Q3_K_M" ./scripts/launch_qwen36_dynamic_release.sh
```

Run a dry-run first to get the real final GGUF size:

```bash
DRY_RUN=1 artifacts/qwen3.6-27b-dynamic-gguf/Qwen3.6-27B-OTQ-DYN-Q4_K_M-GGUF/quantize.sh
```

For `OTQ-DYN-IQ4_NL`, pass an imatrix:

```bash
IMATRIX=artifacts/qwen3.6-27b-imatrix/qwen36.imatrix \
  artifacts/qwen3.6-27b-dynamic-gguf/Qwen3.6-27B-OTQ-DYN-IQ4_NL-GGUF/quantize.sh
```

## Release Gate

Dynamic-compatible GGUFs use the same release gate as native OpenTQ GGUFs:

- smoke generation must pass
- quality samples must pass against the same suite
- long-context wall-clock benchmark must pass
- stock llama.cpp compatibility must be tested without the OpenTQ fork
- Hugging Face staging must include validation JSON, quality JSON, benchmark output, SHA256, and model card

The important distinction: these artifacts are safe to publish as normal GGUFs if they pass, because their tensor types are standard.
