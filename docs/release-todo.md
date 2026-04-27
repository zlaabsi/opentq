# OpenTQ Release TODO

## Benchmark Plots

Add matplotlib benchmark plots to the canonical Hugging Face GGUF repo.

Required charts:

- `assets/benchmark-throughput.png`: `pp8192` and `tg128` comparison by variant.
- `assets/eval-latency.png`: quality/release eval mean and p95 latency by variant.
- `assets/artifact-size.png`: artifact size in GiB/GB by variant.
- `assets/eval-pass-rate.png`: category pass-rate heatmap for release suites.

Rules:

- Use the JSON evidence already published in `evidence/<quant>/`.
- Plot wall-clock/eval latency separately from decode-only throughput.
- Export PNG and SVG when possible.
- Keep charts reproducible from a script, not manually generated.

## Hugging Face Organization

- Canonical stock-compatible GGUF repo: `zlaabsi/Qwen3.6-27B-OTQ-GGUF`.
- Public GGUF filenames must not contain internal profile labels.
- Public GGUF filenames must contain a valid Hugging Face/llama.cpp quant token.
- Existing per-variant repos should be treated as legacy once the canonical repo is live.

## Naming Rules

- `OTQ`: OpenTQ brand and release namespace.
- `TurboQuant`: written in full in user-facing copy and visual assets.
- `DYN`: dynamic tensor-level allocation with standard GGUF tensor types.
- `Q3_K_M`, `Q4_K_M`, etc.: stock GGUF quant names.
- Avoid internal quality labels in filenames unless they are documented and necessary.
