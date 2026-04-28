# Qwen3.6-27B LLM Benchmark Protocol

This protocol separates public release gates from quality claims.

## Baseline Policy

Do not rerun the BF16 model locally just to obtain quality baselines. Qwen already publishes the Qwen3.6-27B reference scores in the official model card. Those values are the external BF16/source baseline for public reporting.

Local BF16 runs are only useful for hardware-specific runtime questions, for example M1 Max prefill wall-clock behavior. They are not required for the model quality release report.

## What Can Be Claimed Today

The public `Qwen3.6-27B-OTQ-GGUF` repo has:

- smoke generation gates;
- long-context `llama-bench` prefill/decode throughput;
- deterministic release micro-suite results;
- per-category pass-rate plots from the release suite;
- imported official Qwen baseline scores for context.

That is enough to publish a usable GGUF release. It is not enough to claim parity on MMLU-Pro, GPQA Diamond, SWE-bench, LiveCodeBench, or any other official benchmark until those benchmark tasks are run on the OTQ artifacts.

## OTQ-vs-Official Comparison

The comparison flow is:

1. Import the official Qwen baseline table from `benchmarks/qwen36_official_language_baseline.json`.
2. Run benchmark subsets on the OTQ GGUF files only.
3. Compare OTQ scores to the official baseline only when the task name, prompt format, scoring rule, and sample policy match.

Use:

```bash
./scripts/run_qwen36_otq_eval.sh
```

It evaluates only the OTQ release artifacts. It does not run the BF16 GGUF.

## Public Benchmark Families To Add

The next quality gate should run at least a fixed-limit subset of:

- MMLU-Pro for broad knowledge/reasoning;
- GPQA Diamond for hard STEM reasoning;
- LiveCodeBench or MBPP/HumanEval for coding;
- SWE-bench style agentic coding if the harness is available;
- IFEval for instruction following;
- LongBench or a local long-context needle set for context retention.

For every task, record:

- exact dataset/task name and revision;
- sample limit or full-run flag;
- prompt format;
- decoding parameters;
- llama.cpp commit;
- hardware and OS;
- wall-clock time, not just decode tok/s;
- score delta versus the official Qwen baseline when comparable.

## Reporting Rule

A Hugging Face card can say "release-gated" after smoke, release-suite and wall-clock gates pass.

It can say "delta versus official Qwen baseline" only after the matching benchmark task has been run on the OTQ artifact.
