<p align="center">
  <img src="docs/assets/opentq-hero.png" alt="OpenTQ" width="947">
</p>

<p align="center">
  <strong>Open quantization tooling for TurboQuant-style low-bit LLM releases, stock GGUF deployment, and Apple Silicon runtime experiments.</strong>
</p>

<p align="center">
  <a href="https://github.com/zlaabsi/opentq">
    <img src="https://img.shields.io/badge/GitHub-opentq-0a1628?style=flat-square" alt="GitHub repository">
  </a>
  <a href="https://github.com/zlaabsi/opentq/stargazers">
    <img src="https://img.shields.io/github/stars/zlaabsi/opentq?style=flat-square" alt="GitHub stars">
  </a>
  <a href="https://github.com/zlaabsi/opentq/network/members">
    <img src="https://img.shields.io/github/forks/zlaabsi/opentq?style=flat-square" alt="GitHub forks">
  </a>
  <a href="https://github.com/zlaabsi/opentq/commits/main">
    <img src="https://img.shields.io/github/last-commit/zlaabsi/opentq?style=flat-square" alt="Last commit">
  </a>
  <a href="https://github.com/zlaabsi/opentq/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/license-Apache--2.0-4b5563?style=flat-square" alt="Apache-2.0 license">
  </a>
  <a href="https://www.python.org/downloads/">
    <img src="https://img.shields.io/badge/python-3.11%2B-3776ab?style=flat-square" alt="Python 3.11+">
  </a>
  <a href="https://huggingface.co/zlaabsi/Qwen3.6-27B-OTQ-GGUF">
    <img src="https://img.shields.io/badge/HF-Qwen3.6--27B--OTQ--GGUF-f59e0b?style=flat-square" alt="Hugging Face GGUF release">
  </a>
  <a href="https://huggingface.co/datasets/zlaabsi/Qwen3.6-27B-OTQ-GGUF-benchmarks">
    <img src="https://img.shields.io/badge/HF%20dataset-reproducibility-10b981?style=flat-square" alt="Hugging Face benchmark dataset">
  </a>
  <a href="https://github.com/ggml-org/llama.cpp">
    <img src="https://img.shields.io/badge/GGUF-stock%20llama.cpp-2563eb?style=flat-square" alt="Stock llama.cpp compatible">
  </a>
</p>

<p align="center">
  <img src="docs/assets/qwen36-paired-bf16-quant-delta.png" alt="Qwen3.6 paired BF16 vs GGUF quality signal" width="860">
</p>

<p align="center">
  Same-task practical mini-subset for <code>Qwen/Qwen3.6-27B</code>: BF16 <code>157/232</code>, Q3_K_M <code>154/232</code>, Q4_K_M <code>155/232</code>, Q5_K_M <code>155/232</code>. This is a release regression signal, not a leaderboard replacement.
</p>

`OpenTQ` is an open quantization lab for low-bit weight formats. It turns quantization from an opaque "one setting for the whole model" step into an auditable release process: plan tensor allocations, generate artifacts, run runtime gates, publish evidence, and keep the claims tied to reproducible files.

The current flagship public artifact is the stock-compatible [`Qwen3.6-27B-OTQ-GGUF`](https://huggingface.co/zlaabsi/Qwen3.6-27B-OTQ-GGUF) release. It uses standard GGUF tensor types, so users can run it with stock `llama.cpp`, while OpenTQ controls the tensor-family allocation policy and validation harness.

## At A Glance

- **Stock GGUF track:** `Q3_K_M`, `Q4_K_M`, and `Q5_K_M` for `Qwen/Qwen3.6-27B`; no custom runtime required.
- **Transparent allocation:** norms/state remain high precision; projection-heavy families absorb most compression.
- **Practical quality checks:** paired BF16-vs-GGUF mini-subsets with pinned task IDs and public reproducibility data.
- **Runtime gates:** local Apple Silicon checks with `llama.cpp`/Metal, bounded generation, release evals, and 8K prefill/decode measurements.
- **Custom policies:** users can define their own dynamic allocation policy with YAML/JSON via `--policy-file`.
- **Run monitor:** long quantization jobs can be watched through the terminal dashboard and machine-readable status output.
- **Research runtime tracks:** native OpenTQ payloads and compressed-domain runtime work are tracked separately from stock GGUF releases.

## Practical Workflows

| Workflow | Command / Doc | What It Gives You |
| --- | --- | --- |
| List built-in allocation profiles | `uv run opentq dynamic-gguf-profiles` | discover the bundled Q3/Q4/Q5 dynamic GGUF policies |
| Generate a stock-compatible plan | `uv run opentq dynamic-gguf-plan --profile OTQ-DYN-Q4_K_M ...` | `plan.json`, `tensor-types.txt`, annotated tensor map, runnable `quantize.sh` |
| Use a custom allocation policy | `uv run opentq dynamic-gguf-plan --policy-file policies/qwen36-custom-dyn-q4.yaml ...` | define where precision is spent without editing OpenTQ source |
| Watch a long run | `uv run opentq monitor --root artifacts/qwen3.6-27b --watch` | terminal dashboard for active profile, tensor, category progress, and release state |
| Read the full guide | [`docs/cookbook.md`](docs/cookbook.md) | end-to-end examples for profiles, policies, monitoring, validation, and release evidence |
| Inspect future UI direction | [`docs/allocation-ui.md`](docs/allocation-ui.md) | tensor treemap, layer/family filters, error overlays, and policy diff design |

OpenTQ's stock-compatible path supports custom **allocation** policies, not arbitrary new GGUF kernels. You can choose which tensor families use `F16`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K`, `Q8_0`, etc.; the resulting GGUF still relies on tensor types supported by `llama.cpp`.

<p align="center">
  <img src="docs/assets/opentq-monitor-overview.png" alt="OpenTQ terminal quantization monitor" width="860">
</p>

## Public Release Matrix

| Track | Repo / Artifact | Runtime | Status | Use It For |
| --- | --- | --- | --- | --- |
| Stock GGUF | [`zlaabsi/Qwen3.6-27B-OTQ-GGUF`](https://huggingface.co/zlaabsi/Qwen3.6-27B-OTQ-GGUF) | stock `llama.cpp` | public | local text inference with standard GGUF loaders |
| Reproducibility dataset | [`zlaabsi/Qwen3.6-27B-OTQ-GGUF-benchmarks`](https://huggingface.co/datasets/zlaabsi/Qwen3.6-27B-OTQ-GGUF-benchmarks) | JSON/CSV assets | public | pinned BF16-vs-GGUF samples, raw outputs, reports |
| BF16 sidecar | [`zlaabsi/opentq-qwen36-bf16-sidecar`](https://huggingface.co/datasets/zlaabsi/opentq-qwen36-bf16-sidecar) | HF Jobs H200 output | public | matching BF16 baseline for the practical subset |
| Packed `.otq` | local `Qwen3.6-27B-OTQ-Packed` artifacts | OpenTQ runtime probes | gated locally | custom-runtime integration and parity testing |
| Custom OpenTQ GGUF | local `Qwen3.6-27B-OTQ-TQ3_SB4-Metal.gguf` | patched `llama.cpp` Metal path | gated locally | native Metal runtime experiments |

## Qwen3.6-27B GGUF Variants

| File | Quant | Size | Recommended Target | Role |
| --- | --- | ---: | --- | --- |
| `Qwen3.6-27B-OTQ-DYN-Q3_K_M.gguf` | `Q3_K_M` | 13.48 GiB | 32 GB Apple Silicon first pick | compact local run |
| `Qwen3.6-27B-OTQ-DYN-Q4_K_M.gguf` | `Q4_K_M` | 16.82 GiB | 32 GB with care; 48 GB+ preferred | balanced default |
| `Qwen3.6-27B-OTQ-DYN-Q5_K_M.gguf` | `Q5_K_M` | 19.92 GiB | 48 GB+ preferred; measured on M1 Max 32 GB with tight headroom | quality-first local run |

Quick download:

```bash
hf download zlaabsi/Qwen3.6-27B-OTQ-GGUF \
  Qwen3.6-27B-OTQ-DYN-Q4_K_M.gguf \
  --local-dir models/Qwen3.6-27B-OTQ-GGUF
```

Run with stock `llama.cpp`:

```bash
llama-cli \
  -m models/Qwen3.6-27B-OTQ-GGUF/Qwen3.6-27B-OTQ-DYN-Q4_K_M.gguf \
  -p "Explain OpenTQ in three concise bullet points." \
  -ngl 999
```

## Allocation Transparency

<p align="center">
  <img src="docs/assets/qwen36-tensor-allocation.png" alt="Dynamic GGUF tensor-type allocation" width="860">
</p>

<p align="center">
  OpenTQ does not apply a flat quantization recipe. It assigns standard GGUF tensor types per tensor family, then publishes the allocation map so users can inspect where precision was spent.
</p>

| Variant | Mapped Tensors | F16 | Q3_K | Q4_K | Q5_K | Q6_K | Q8_0 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `Q3_K_M` | 851 | 353 | 180 | 252 | 65 | 1 | 0 |
| `Q4_K_M` | 851 | 353 | 0 | 180 | 237 | 80 | 1 |
| `Q5_K_M` | 851 | 353 | 0 | 0 | 180 | 237 | 81 |

<p align="center">
  <img src="docs/assets/qwen36-allocation-policy.png" alt="Where OpenTQ spends precision" width="860">
</p>

The compact profile spends fewer bits on bulk projection tensors, while normalization, state, embeddings, self-attention anchors, and output-sensitive tensors stay higher precision. The goal is not just a smaller file; it is a quantization policy that can be audited.

## Quality Signal

The paired subset uses the same pinned task IDs, prompt format `qwen3-no-think`, deterministic decoding, and local scoring rules for BF16 and the GGUF artifacts.

| Metric | BF16 | Q3_K_M | Q4_K_M | Q5_K_M |
| --- | ---: | ---: | ---: | ---: |
| Practical mini-subset | 157/232 | 154/232 | 155/232 | 155/232 |
| Aggregate score | 67.7% | 66.4% | 66.8% | 66.8% |
| Delta vs BF16 | baseline | -1.3 pp | -0.9 pp | -0.9 pp |

Important boundary: this is a paired release-regression signal. Official Qwen full-harness scores remain the capability baseline; the mini-subset should not be used as a leaderboard claim.

## Runtime And Release Gates

<p align="center">
  <img src="docs/assets/qwen36-release-scorecard.png" alt="Qwen3.6 release decision scorecard" width="860">
</p>

<p align="center">
  <img src="docs/assets/qwen36-runtime-frontier.png" alt="Qwen3.6 runtime frontier" width="860">
</p>

| Variant | Prefill Gate | Decode Gate | Release Eval | Measured Hardware |
| --- | ---: | ---: | --- | --- |
| `Q3_K_M` | 107.09 tok/s | 10.19 tok/s | passed | M1 Max 32 GB |
| `Q4_K_M` | 106.98 tok/s | 9.62 tok/s | passed | M1 Max 32 GB |
| `Q5_K_M` | 93.94 tok/s | 8.87 tok/s | passed | M1 Max 32 GB, tight headroom |

Use `Q3_K_M` first on 32 GB Macs. Use `Q4_K_M` for the best balance. Use `Q5_K_M` when quality is worth the extra memory pressure.

## What This Repo Contains

| Area | Files | Purpose |
| --- | --- | --- |
| Quantizer core | `src/opentq/` | codebooks, rotations, tensor quantization, packing, CLI |
| Release scripts | `scripts/` | GGUF staging, HF release reports, runtime checks |
| Benchmarks | `benchmarks/` | pinned benchmark matrix, paired BF16-vs-GGUF summaries |
| Docs | `docs/` | architecture, format notes, benchmark methodology |
| Tests | `tests/` | unit tests and release tooling checks |

## Format Family

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

## Developer Quick Start

```bash
uv sync
uv run opentq variants
uv run opentq recipe qwen3.6-27b --format markdown
uv run opentq inventory --model-id Qwen/Qwen3.6-27B
uv run opentq dynamic-gguf-profiles
```

Create a stock-compatible GGUF allocation plan:

```bash
uv run opentq dynamic-gguf-plan \
  --profile OTQ-DYN-Q4_K_M \
  --output artifacts/qwen36-otq-dyn-q4-k-m \
  --llama-cpp /path/to/llama.cpp
```

Create a custom allocation plan from an external policy file:

```bash
uv run opentq dynamic-gguf-plan \
  --policy-file policies/qwen36-custom-dyn-q4.yaml \
  --output artifacts/qwen36-custom-q4 \
  --llama-cpp /path/to/llama.cpp
```

Build release reports and status pages:

```bash
uv run python scripts/build_qwen36_release_report.py
./scripts/status_qwen36_dynamic_ggufs.sh
```

## Release Scope

OpenTQ separates stock-compatible GGUF releases from native runtime research. The public Qwen3.6-27B OTQ GGUF release uses standard `llama.cpp` tensor types and requires no custom OpenTQ runtime. Native packed payloads and custom Metal/runtime work are separate research tracks.

Practical mini-subsets are reported as quantization-regression signals for the released artifacts. They are not replacements for official full-harness benchmark results.

## Further Reading

- [Cookbook](docs/cookbook.md)
- [Architecture](docs/architecture.md)
- [Variant naming](docs/variants.md)
- [Dynamic-compatible GGUF path](docs/dynamic-compatible-gguf.md)
- [Native OpenTQ runtime track](docs/native-opentq-runtime.md)
- [Allocation UI roadmap](docs/allocation-ui.md)
- [KV-cache layer policy roadmap](docs/research/kv-cache-layer-policy.md)
- [Quantization-aware pruning roadmap](docs/research/quantization-aware-pruning.md)
- [Benchmark methodology](docs/llm-benchmark-protocol.md)
