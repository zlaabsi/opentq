# Qwen3.6 Benchmark Representativeness

The previous practical report is a smoke signal, not a representative quality claim. It used `4` samples per benchmark family and should be described as regression detection only.

## Current Tiers

| Tier | Purpose | Models | Samples | Publishable Claim |
| --- | --- | --- | --- | --- |
| Smoke | Catch broken inference or scoring quickly | Q3/Q4/Q5 | `1-4` per selected benchmark | No quality claim |
| Practical | Cheap sanity across many families | Q3/Q4 | `4` per implemented family | OTQ-only smoke percentages |
| Representative Local | Better local signal using spread offsets | Q3/Q4/Q5 | `8` per selected text/reasoning family by default | Local subset only, no full benchmark parity |
| Publication Candidate | Stronger public appendix | Q3/Q4/Q5 plus BF16 sidecar or official-compatible review | `16-24` per family where feasible | Limited claim with exact subset metadata |
| Full Benchmark | Real leaderboard-style claim | selected release models | official harness/sample count | Public benchmark claim |

## Immediate Fixes

- Benchmark task ids are now deterministic spread offsets across pinned splits instead of `offset:0..N`.
- Q5 is now a first-class benchmark runner target.
- Representative local runs use a separate artifact root and do not overwrite the older practical smoke report.
- SWE-bench remains patch-only until the official harness is run.
- Judge-based benchmarks remain blocked until a pinned judge and BF16 sidecar exist.

## Launch

```bash
cd /Users/zlaabsi/Documents/GitHub/opentq
./scripts/launch_qwen36_representative_benchmarks.sh
```

Defaults:

- models: `q3,q4,q5`
- samples per selected benchmark: `8`
- max tokens: `1024`
- benchmarks: `mmlu mmlu_pro arc hellaswag gsm8k math aime bbh gpqa ifeval truthfulqa winogrande drop piqa commonsenseqa`

Monitor:

```bash
screen -r opentq_qwen36_representative
tail -f artifacts/qwen3.6-27b-representative-benchmarks/*/representative.log
```

The result is a local report under:

```text
artifacts/qwen3.6-27b-representative-benchmarks/<timestamp>/degradation-report/degradation-report.md
```
