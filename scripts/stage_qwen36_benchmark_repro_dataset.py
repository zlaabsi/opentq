#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


DATASET_REPO_ID = "zlaabsi/Qwen3.6-27B-OTQ-GGUF-benchmarks"
MODEL_REPO_ID = "zlaabsi/Qwen3.6-27B-OTQ-GGUF"
BASE_MODEL = "Qwen/Qwen3.6-27B"
BF16_JOB = "69f235d2d2c8bd8662bd320e"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage public Qwen3.6 OTQ benchmark reproducibility dataset.")
    parser.add_argument("--output", default="artifacts/hf-datasets/Qwen3.6-27B-OTQ-GGUF-benchmarks")
    parser.add_argument("--paired-report-root", default="artifacts/qwen3.6-27b-paired-bf16-quant-report-232")
    parser.add_argument("--quant-root", default="artifacts/qwen3.6-27b-benchmark-subsets-release-core-232")
    parser.add_argument(
        "--bf16-json",
        default=f"artifacts/qwen3.6-27b-bf16-hf-sidecar/{BF16_JOB}/bf16_remote_no_think.json",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def index_run(payload: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    for benchmark in payload.get("benchmarks", []):
        benchmark_id = benchmark["benchmark_id"]
        samples = {sample["task_id"]: sample for sample in benchmark.get("samples", [])}
        for result in benchmark.get("results", []):
            task_id = result["task_id"]
            indexed[(benchmark_id, task_id)] = {
                "sample": samples.get(task_id, {}),
                "result": result,
            }
    return indexed


def score_fields(result: dict[str, Any]) -> dict[str, Any]:
    score = result.get("score", {})
    return {
        "passed": bool(result.get("passed", False)),
        "actual": score.get("actual"),
        "expected": score.get("expected"),
        "stdout_tail": result.get("stdout_tail"),
        "elapsed_seconds": result.get("elapsed_seconds"),
    }


def paired_sample_rows(bf16: dict[str, Any], q3: dict[str, Any], q4: dict[str, Any], q5: dict[str, Any]) -> list[dict[str, Any]]:
    runs = {
        "bf16": index_run(bf16),
        "q3_k_m": index_run(q3),
        "q4_k_m": index_run(q4),
        "q5_k_m": index_run(q5),
    }
    keys = sorted(set().union(*(set(index.keys()) for index in runs.values())))
    rows: list[dict[str, Any]] = []
    for benchmark_id, task_id in keys:
        sample = next((runs[name].get((benchmark_id, task_id), {}).get("sample") for name in runs if runs[name].get((benchmark_id, task_id), {}).get("sample")), {})
        row: dict[str, Any] = {
            "benchmark_id": benchmark_id,
            "task_id": task_id,
            "dataset": sample.get("dataset"),
            "config": sample.get("config"),
            "split": sample.get("split"),
            "revision": sample.get("revision"),
            "category": sample.get("category"),
            "prompt_format": sample.get("prompt_format"),
            "scoring_rule": sample.get("scoring_rule"),
            "max_tokens": sample.get("max_tokens"),
            "answer": sample.get("answer"),
            "prompt": sample.get("prompt"),
        }
        for name, index in runs.items():
            result = index.get((benchmark_id, task_id), {}).get("result", {})
            row[name] = score_fields(result) if result else {}
        rows.append(row)
    return rows


def copy_file(source: Path, target: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def readme() -> str:
    return f"""---
pretty_name: Qwen3.6-27B OTQ GGUF benchmark reproducibility
license: other
tags:
- opentq
- qwen3.6
- qwen
- gguf
- benchmark
- reproducibility
- quantization
task_categories:
- text-generation
---

# Qwen3.6-27B OTQ GGUF Benchmark Reproducibility

This dataset contains the compact paired benchmark evidence used by [`{MODEL_REPO_ID}`](https://huggingface.co/{MODEL_REPO_ID}).

It is a reproducibility dataset, not a leaderboard dataset. The rows are small practical release signals run on pinned task IDs with prompt format `qwen3-no-think`, deterministic decoding and local scoring rules.

## Contents

| Path | Meaning |
| --- | --- |
| `data/paired_samples.jsonl` | Flattened 232-row paired sample table with prompts, task IDs, scoring metadata and BF16/Q3/Q4/Q5 outputs. |
| `data/paired_summary.jsonl` | One row per benchmark plus `TOTAL`, matching the model card table. |
| `raw/bf16/no_think.json` | BF16 sidecar output from Hugging Face Jobs H200 run `{BF16_JOB}` on `{BASE_MODEL}`. |
| `raw/quant/q3.json` | Local stock `llama.cpp`/Metal/FlashAttention run for `Q3_K_M`. |
| `raw/quant/q4.json` | Local stock `llama.cpp`/Metal/FlashAttention run for `Q4_K_M`. |
| `raw/quant/q5.json` | Local stock `llama.cpp`/Metal/FlashAttention run for `Q5_K_M`. |
| `reports/paired_summary.csv` | Benchmark summary CSV. |
| `reports/paired_summary.json` | Benchmark summary JSON. |
| `reports/paired_report.md` | Human-readable paired subset report. |

## Scope

- Base model: `{BASE_MODEL}`
- GGUF model repo: `{MODEL_REPO_ID}`
- BF16 runtime: Hugging Face Jobs H200 with Transformers
- GGUF runtime: stock `llama.cpp` / `llama-server`, Metal, FlashAttention
- Prompt format: `qwen3-no-think`
- Sample count: 232

These files include prompts derived from upstream benchmark datasets. Respect the upstream dataset licenses and terms for any reuse beyond reproducibility of this release evaluation.
"""


def main() -> int:
    args = parse_args()
    output = Path(args.output)
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)

    paired_root = Path(args.paired_report_root)
    quant_root = Path(args.quant_root)
    bf16_path = Path(args.bf16_json)

    bf16 = load_json(bf16_path)
    q3 = load_json(quant_root / "q3.json")
    q4 = load_json(quant_root / "q4.json")
    q5 = load_json(quant_root / "q5.json")
    paired = load_json(paired_root / "paired_summary.json")

    rows = paired_sample_rows(bf16, q3, q4, q5)
    write_jsonl(output / "data" / "paired_samples.jsonl", rows)
    write_jsonl(output / "data" / "paired_summary.jsonl", list(paired.get("rows", [])))

    copy_file(bf16_path, output / "raw" / "bf16" / "no_think.json")
    for name in ("q3", "q4", "q5"):
        copy_file(quant_root / f"{name}.json", output / "raw" / "quant" / f"{name}.json")
    copy_file(paired_root / "paired_summary.csv", output / "reports" / "paired_summary.csv")
    copy_file(paired_root / "paired_summary.json", output / "reports" / "paired_summary.json")
    copy_file(paired_root / "README.md", output / "reports" / "paired_report.md")

    write_json(
        output / "metadata" / "dataset_inventory.json",
        {
            "schema": "opentq.qwen36_benchmark_repro_dataset.v1",
            "dataset_repo_id": DATASET_REPO_ID,
            "model_repo_id": MODEL_REPO_ID,
            "base_model": BASE_MODEL,
            "bf16_job": BF16_JOB,
            "sample_rows": len(rows),
            "source_files": {
                "bf16": str(bf16_path),
                "q3": str(quant_root / "q3.json"),
                "q4": str(quant_root / "q4.json"),
                "q5": str(quant_root / "q5.json"),
                "paired_summary": str(paired_root / "paired_summary.json"),
            },
        },
    )
    (output / "README.md").write_text(readme(), encoding="utf-8")
    print(f"staged {DATASET_REPO_ID} at {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
