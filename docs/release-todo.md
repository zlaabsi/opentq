# OpenTQ Release TODO

## Benchmark Plots

Matplotlib benchmark plots are implemented for the canonical Hugging Face GGUF repo.

Generated charts:

- `assets/benchmark-throughput.{svg,pdf,png}`: `pp8192` and `tg128` comparison by variant.
- `assets/eval-latency.{svg,pdf,png}`: release eval mean and p95 latency by variant.
- `assets/artifact-size.{svg,pdf,png}`: artifact size in GiB by variant.
- `assets/eval-pass-rate.{svg,pdf,png}`: category pass-rate heatmap for release suites.
- `assets/tensor-allocation.{svg,pdf,png}`: GGUF tensor-type allocation by variant.
- `assets/official-language-baseline.{svg,pdf,png}`: official Qwen3.6-27B reference scores.

Generate them with:

```bash
uv run python scripts/build_qwen36_release_report.py
```

Rules:

- Use the JSON evidence already published in `evidence/<quant>/`.
- Plot wall-clock/eval latency separately from decode-only throughput.
- Export SVG/PDF/PNG plus CSV tables under `benchmarks/`.
- Keep charts reproducible from a script, not manually generated.

## Official Baseline

Do not run the BF16 GGUF locally just to obtain quality baselines. Use Qwen's official scores as the external baseline and run OTQ artifacts only.

Run OTQ-only comparison/eval probes with:

```bash
./scripts/run_qwen36_otq_eval.sh
```

Baseline data lives in `benchmarks/qwen36_official_language_baseline.json`. See `docs/llm-benchmark-protocol.md`.

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
