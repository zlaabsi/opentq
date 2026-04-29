# Benchmark Methodology

OpenTQ reports two different kinds of evidence:

1. **Release gates**: smoke generation, runtime throughput, bounded generation, and artifact integrity checks. These show that a release is runnable.
2. **Quality signals**: paired benchmark subsets or full benchmark harnesses. These estimate quantization regression relative to a reference model.

The two should not be conflated. A runnable GGUF release is not automatically a full benchmark claim.

## Paired Mini-Subsets

The Qwen3.6-27B GGUF release includes a paired BF16-vs-GGUF mini-subset. It uses:

- identical pinned task IDs;
- the same prompt format;
- deterministic decoding;
- the same local scoring rules;
- machine-readable raw outputs and summary files.

This makes the subset useful for regression detection across quantized artifacts. It is intentionally described as a practical quality signal, not as a replacement for official full-harness scores.

## Full Benchmark Claims

OpenTQ treats full benchmark claims as valid only when the quantized artifact is evaluated with the matching benchmark definition:

- dataset and revision;
- split and task IDs;
- prompt format;
- scoring rule or judge;
- sample count or full-run setting;
- runtime and decoding parameters.

For leaderboard-style comparisons, the quantized model should be run through the same task definition as the BF16/source model. Official upstream model-card scores remain useful context, but they are not a substitute for running the quantized artifact under the matching harness.

## Reported Metadata

Every benchmark report should include:

- model artifact and checksum;
- benchmark name, dataset revision, split, and task IDs;
- prompt format and decoding settings;
- runtime backend and hardware;
- sample count;
- aggregate score;
- per-task raw outputs when redistribution is allowed;
- whether the result is a paired mini-subset, a release gate, or a full benchmark run.

This keeps public claims auditable and makes it clear which numbers are regression signals and which are benchmark claims.
