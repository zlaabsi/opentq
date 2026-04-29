#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "accelerate>=1.10.0",
#   "datasets>=4.0.0",
#   "hf-transfer>=0.1.9",
#   "huggingface-hub>=0.34.0",
#   "requests>=2.32.0",
#   "safetensors>=0.5.0",
#   "sentencepiece>=0.2.0",
#   "torch>=2.7.0",
#   "transformers>=4.57.0",
# ]
# ///
from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


DEFAULT_MODEL_ID = "Qwen/Qwen3.6-27B"
DEFAULT_REPO_URL = "https://github.com/zlaabsi/opentq.git"
DEFAULT_REPO_REF = "main"
DEFAULT_UPLOAD_REPO = "zlaabsi/opentq-qwen36-bf16-sidecar"
DATASET_VIEWER = "https://datasets-server.huggingface.co"
DATASET_VIEWER_RETRIES = 8
DATASET_VIEWER_RETRY_STATUSES = {429, 500, 502, 503, 504}
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
BEGIN_MARKER = "BEGIN_OPENTQ_QWEN36_BF16_SIDECAR_JSON"
END_MARKER = "END_OPENTQ_QWEN36_BF16_SIDECAR_JSON"


@dataclass(frozen=True)
class ModeConfig:
    name: str
    prompt_format: str
    samples_per_family: int
    max_tokens: int
    token_policy: str


@dataclass(frozen=True)
class EmbeddedBenchmarkAdapter:
    benchmark_id: str
    dataset: str
    config: str
    split: str
    revision: str
    task_ids: tuple[str, ...]
    prompt_format: str
    scoring_rule: str
    max_tokens: int

    def as_plan_payload(self) -> dict[str, Any]:
        return {
            "benchmark_id": self.benchmark_id,
            "dataset": self.dataset,
            "config": self.config,
            "split": self.split,
            "revision": self.revision,
            "task_ids": list(self.task_ids),
            "prompt_format": self.prompt_format,
            "scoring_rule": self.scoring_rule,
            "max_tokens": self.max_tokens,
            "backend": "embedded_dataset_viewer",
        }


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def spread_offsets(total: int, count: int) -> tuple[str, ...]:
    target_count = min(total, count)
    if target_count == 1:
        return ("offset:0",)
    selected = []
    seen = set()
    for index in range(target_count):
        offset = round(index * (total - 1) / (target_count - 1))
        if offset in seen:
            continue
        selected.append(f"offset:{offset}")
        seen.add(offset)
    return tuple(selected)


def letter_for_index(index: int) -> str:
    if index < 0 or index >= len(LETTERS):
        raise ValueError(f"choice index out of range: {index}")
    return LETTERS[index]


def multiple_choice_prompt(question: str, choices: list[str], labels: list[str] | None = None) -> str:
    if labels is None:
        labels = [letter_for_index(index) for index in range(len(choices))]
    rendered = "\n".join(f"({label}) {choice}" for label, choice in zip(labels, choices, strict=True))
    return f"{question}\n\n{rendered}\n\nReturn only the letter of the correct answer."


def parse_task_offset(task_id: str) -> int:
    if not task_id.startswith("offset:"):
        raise ValueError(f"unsupported task id {task_id!r}; expected offset:<int>")
    return int(task_id.split(":", 1)[1])


def extract_boxed_answer(text: str) -> str:
    match = re.search(r"\\boxed\{([^{}]+)\}", text)
    return match.group(1).strip() if match else text.strip()


def extract_hash_answer(answer: str) -> str:
    match = re.search(r"####\s*([-+]?\d+(?:\.\d+)?)", answer)
    return match.group(1) if match else answer.strip()


def clean_generated_text(output: str) -> str:
    cleaned = output.strip()
    while cleaned.endswith("[end of text]"):
        cleaned = cleaned[: -len("[end of text]")].strip()
    return cleaned


def final_answer_text(output: str) -> str:
    cleaned = clean_generated_text(output)
    if "</think>" in cleaned:
        cleaned = cleaned.rsplit("</think>", 1)[1].strip()
    for marker in ("<|im_end|>", "<|endoftext|>"):
        if marker in cleaned:
            cleaned = cleaned.split(marker, 1)[0].strip()
    return cleaned


def extract_multiple_choice_letter(output: str) -> str | None:
    cleaned = final_answer_text(output).upper()
    boxed = re.search(r"\\BOXED\{([A-Z])\}", cleaned)
    if boxed:
        return boxed.group(1)
    parenthesized = re.findall(r"\(([A-Z])\)", cleaned)
    if parenthesized:
        return parenthesized[-1]
    standalone = re.findall(r"\b([A-Z])\b", cleaned)
    return standalone[-1] if standalone else None


def extract_number(output: str) -> str | None:
    matches = re.findall(r"[-+]?\d+(?:\.\d+)?", final_answer_text(output).replace(",", ""))
    return matches[-1] if matches else None


def embedded_adapters() -> dict[str, EmbeddedBenchmarkAdapter]:
    return {
        "mmlu": EmbeddedBenchmarkAdapter(
            "mmlu",
            dataset="cais/mmlu",
            config="all",
            split="test",
            revision="c30699e8356da336a370243923dbaf21066bb9fe",
            task_ids=spread_offsets(total=14042, count=16),
            prompt_format="qwen3-no-think",
            scoring_rule="multiple_choice_letter",
            max_tokens=16,
        ),
        "mmlu_pro": EmbeddedBenchmarkAdapter(
            "mmlu_pro",
            dataset="TIGER-Lab/MMLU-Pro",
            config="default",
            split="test",
            revision="54611cde22c74cca43dd78732198de6abe971398",
            task_ids=spread_offsets(total=12032, count=24),
            prompt_format="qwen3-no-think",
            scoring_rule="multiple_choice_letter",
            max_tokens=16,
        ),
        "arc": EmbeddedBenchmarkAdapter(
            "arc",
            dataset="allenai/ai2_arc",
            config="ARC-Challenge",
            split="validation",
            revision="210d026faf9955653af8916fad021475a3f00453",
            task_ids=spread_offsets(total=299, count=16),
            prompt_format="qwen3-no-think",
            scoring_rule="multiple_choice_letter",
            max_tokens=16,
        ),
        "hellaswag": EmbeddedBenchmarkAdapter(
            "hellaswag",
            dataset="Rowan/hellaswag",
            config="default",
            split="validation",
            revision="218ec52e09a7e7462a5400043bb9a69a41d06b76",
            task_ids=spread_offsets(total=10042, count=16),
            prompt_format="qwen3-no-think",
            scoring_rule="multiple_choice_letter",
            max_tokens=16,
        ),
        "gsm8k": EmbeddedBenchmarkAdapter(
            "gsm8k",
            dataset="openai/gsm8k",
            config="main",
            split="test",
            revision="740312add88f781978c0658806c59bc2815b9866",
            task_ids=spread_offsets(total=1319, count=16),
            prompt_format="qwen3-no-think",
            scoring_rule="numeric_exact",
            max_tokens=1024,
        ),
        "math": EmbeddedBenchmarkAdapter(
            "math",
            dataset="EleutherAI/hendrycks_math",
            config="algebra",
            split="test",
            revision="21a5633873b6a120296cce3e2df9d5550074f4a3",
            task_ids=spread_offsets(total=1187, count=16),
            prompt_format="qwen3-no-think",
            scoring_rule="math_boxed_exact",
            max_tokens=2048,
        ),
        "gpqa": EmbeddedBenchmarkAdapter(
            "gpqa",
            dataset="hendrydong/gpqa_diamond_mc",
            config="default",
            split="test",
            revision="284143babc24a94fbac45d143333b2307e64ff80",
            task_ids=spread_offsets(total=198, count=24),
            prompt_format="qwen3-no-think",
            scoring_rule="multiple_choice_letter",
            max_tokens=16,
        ),
        "bbh": EmbeddedBenchmarkAdapter(
            "bbh",
            dataset="lukaemon/bbh",
            config="boolean_expressions",
            split="test",
            revision="982bb89fd79532a8ac676a61fc42eb1aeec63f99",
            task_ids=spread_offsets(total=250, count=24),
            prompt_format="qwen3-no-think",
            scoring_rule="exact_text",
            max_tokens=32,
        ),
        "truthfulqa": EmbeddedBenchmarkAdapter(
            "truthfulqa",
            dataset="truthfulqa/truthful_qa",
            config="multiple_choice",
            split="validation",
            revision="741b8276f2d1982aa3d5b832d3ee81ed3b896490",
            task_ids=spread_offsets(total=817, count=16),
            prompt_format="qwen3-no-think",
            scoring_rule="multiple_choice_letter",
            max_tokens=16,
        ),
        "winogrande": EmbeddedBenchmarkAdapter(
            "winogrande",
            dataset="allenai/winogrande",
            config="winogrande_debiased",
            split="validation",
            revision="01e74176c63542e6b0bcb004dcdea22d94fb67b5",
            task_ids=spread_offsets(total=1267, count=16),
            prompt_format="qwen3-no-think",
            scoring_rule="multiple_choice_letter",
            max_tokens=16,
        ),
        "drop": EmbeddedBenchmarkAdapter(
            "drop",
            dataset="ucinlp/drop",
            config="default",
            split="validation",
            revision="95cda593fae71b60b5b19f82de3fcf3298c1239c",
            task_ids=spread_offsets(total=9535, count=16),
            prompt_format="qwen3-no-think",
            scoring_rule="contains_any",
            max_tokens=256,
        ),
        "piqa": EmbeddedBenchmarkAdapter(
            "piqa",
            dataset="lighteval/piqa",
            config="plain_text",
            split="validation",
            revision="41782e6bf0ef7de82a2ca8a9feb1dca042837fae",
            task_ids=spread_offsets(total=1838, count=16),
            prompt_format="qwen3-no-think",
            scoring_rule="multiple_choice_letter",
            max_tokens=16,
        ),
        "commonsenseqa": EmbeddedBenchmarkAdapter(
            "commonsenseqa",
            dataset="tau/commonsense_qa",
            config="default",
            split="validation",
            revision="94630fe30dad47192a8546eb75f094926d47e155",
            task_ids=spread_offsets(total=1221, count=16),
            prompt_format="qwen3-no-think",
            scoring_rule="multiple_choice_letter",
            max_tokens=16,
        ),
        "aime": EmbeddedBenchmarkAdapter(
            "aime",
            dataset="MathArena/aime_2026",
            config="default",
            split="train",
            revision="10b4e45b7a503075d4da8a0d57916a4f06ce6bd2",
            task_ids=spread_offsets(total=30, count=24),
            prompt_format="qwen3-no-think",
            scoring_rule="numeric_exact",
            max_tokens=4096,
        ),
    }


class EmbeddedSmokeRunner:
    SCHEMA = "opentq.qwen36_benchmark_subset_eval.v1"

    def __init__(self) -> None:
        self.ADAPTERS = embedded_adapters()
        self._row_cache: dict[tuple[str, str], dict[str, Any]] = {}

    def fetch_row(self, adapter: EmbeddedBenchmarkAdapter, task_id: str) -> dict[str, Any]:
        import requests

        cache_key = (adapter.benchmark_id, task_id)
        cached = self._row_cache.get(cache_key)
        if cached is not None:
            return dict(cached)
        params = {
            "dataset": adapter.dataset,
            "config": adapter.config,
            "split": adapter.split,
            "revision": adapter.revision,
            "offset": parse_task_offset(task_id),
            "length": 1,
        }
        response = None
        for attempt in range(1, DATASET_VIEWER_RETRIES + 1):
            response = requests.get(f"{DATASET_VIEWER}/rows", params=params, timeout=60)
            if response.status_code not in DATASET_VIEWER_RETRY_STATUSES:
                break
            if attempt < DATASET_VIEWER_RETRIES:
                time.sleep(min(45, 2**attempt))
        assert response is not None
        response.raise_for_status()
        rows = response.json().get("rows", [])
        if not rows:
            raise ValueError(f"no row returned for {adapter.benchmark_id} {task_id}")
        row = dict(rows[0]["row"])
        self._row_cache[cache_key] = row
        return dict(row)

    def build_sample_from_row(self, adapter: EmbeddedBenchmarkAdapter, task_id: str, row: dict[str, Any]) -> dict[str, Any]:
        sample = {
            "benchmark_id": adapter.benchmark_id,
            "task_id": task_id,
            "dataset": adapter.dataset,
            "config": adapter.config,
            "split": adapter.split,
            "revision": adapter.revision,
            "prompt_format": adapter.prompt_format,
            "scoring_rule": adapter.scoring_rule,
            "max_tokens": adapter.max_tokens,
        }
        if adapter.benchmark_id == "mmlu_pro":
            choices = [str(item) for item in row["options"]]
            sample.update(
                prompt=multiple_choice_prompt(str(row["question"]), choices),
                answer=letter_for_index(int(row["answer_index"])),
                category=row.get("category"),
            )
            return sample
        if adapter.benchmark_id == "mmlu":
            choices = [str(item) for item in row["choices"]]
            sample.update(
                prompt=multiple_choice_prompt(str(row["question"]), choices),
                answer=letter_for_index(int(row["answer"])),
                category=row.get("subject"),
            )
            return sample
        if adapter.benchmark_id == "arc":
            choices = [str(item) for item in row["choices"]["text"]]
            labels = [str(item) for item in row["choices"]["label"]]
            sample.update(prompt=multiple_choice_prompt(str(row["question"]), choices, labels), answer=str(row["answerKey"]))
            return sample
        if adapter.benchmark_id == "hellaswag":
            sample.update(
                prompt=multiple_choice_prompt(str(row["ctx"]), [str(item) for item in row["endings"]]),
                answer=letter_for_index(int(row["label"])),
            )
            return sample
        if adapter.benchmark_id == "gsm8k":
            sample.update(
                prompt=f"{row['question']}\n\nReturn only the final numeric answer.",
                answer=extract_hash_answer(str(row["answer"])),
            )
            return sample
        if adapter.benchmark_id == "math":
            sample.update(
                prompt=f"{row['problem']}\n\nReturn only the final answer.",
                answer=extract_boxed_answer(str(row["solution"])),
                category=row.get("type"),
            )
            return sample
        if adapter.benchmark_id == "bbh":
            sample.update(prompt=f"{row['input']}\n\nReturn only the answer.", answer=str(row["target"]))
            return sample
        if adapter.benchmark_id == "gpqa":
            sample.update(
                prompt=str(row["problem"]),
                answer=extract_boxed_answer(str(row["solution"])).replace("\\boxed{", "").replace("}", ""),
            )
            return sample
        if adapter.benchmark_id == "truthfulqa":
            choices = [str(item) for item in row["mc1_targets"]["choices"]]
            labels = [int(item) for item in row["mc1_targets"]["labels"]]
            sample.update(prompt=multiple_choice_prompt(str(row["question"]), choices), answer=letter_for_index(labels.index(1)))
            return sample
        if adapter.benchmark_id == "winogrande":
            question = str(row["sentence"]).replace("_", "____")
            sample.update(
                prompt=multiple_choice_prompt(question, [str(row["option1"]), str(row["option2"])]),
                answer=letter_for_index(int(row["answer"]) - 1),
            )
            return sample
        if adapter.benchmark_id == "drop":
            sample.update(
                prompt=f"{row['passage']}\n\nQuestion: {row['question']}\n\nReturn only the answer.",
                answer=[str(item) for item in row["answers_spans"]["spans"]],
            )
            return sample
        if adapter.benchmark_id == "piqa":
            sample.update(
                prompt=multiple_choice_prompt(str(row["goal"]), [str(row["sol1"]), str(row["sol2"])]),
                answer=letter_for_index(int(row["label"])),
            )
            return sample
        if adapter.benchmark_id == "commonsenseqa":
            choices = [str(item) for item in row["choices"]["text"]]
            labels = [str(item) for item in row["choices"]["label"]]
            sample.update(prompt=multiple_choice_prompt(str(row["question"]), choices, labels), answer=str(row["answerKey"]))
            return sample
        if adapter.benchmark_id == "aime":
            sample.update(prompt=f"{row['problem']}\n\nReturn only the final integer answer.", answer=str(row["answer"]))
            return sample
        raise ValueError(f"unsupported embedded adapter: {adapter.benchmark_id}")

    def samples_for_adapter(self, adapter: EmbeddedBenchmarkAdapter, max_samples: int | None = None) -> list[dict[str, Any]]:
        task_ids = adapter.task_ids[:max_samples] if max_samples is not None else adapter.task_ids
        return [self.build_sample_from_row(adapter, task_id, self.fetch_row(adapter, task_id)) for task_id in task_ids]

    def format_qwen_prompt(self, prompt: str, prompt_format: str) -> str:
        if prompt_format == "qwen3-no-think":
            return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        if prompt_format == "qwen3-thinking":
            return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n"
        if prompt_format == "raw":
            return prompt
        raise ValueError(f"unsupported prompt_format: {prompt_format}")

    def score_benchmark_output(self, sample: dict[str, Any], output: str) -> dict[str, Any]:
        expected = sample.get("answer")
        if sample["scoring_rule"] == "multiple_choice_letter":
            actual = extract_multiple_choice_letter(output)
            return {"passed": actual == expected, "expected": expected, "actual": actual}
        if sample["scoring_rule"] in {"numeric_exact", "math_boxed_exact"}:
            final_text = final_answer_text(output)
            actual = extract_boxed_answer(final_text)
            if actual == final_text:
                actual = extract_number(output) or actual
            return {"passed": str(actual).strip() == str(expected).strip(), "expected": expected, "actual": actual}
        if sample["scoring_rule"] == "exact_text":
            actual = final_answer_text(output)
            return {"passed": actual.lower() == str(expected).lower(), "expected": expected, "actual": actual}
        if sample["scoring_rule"] == "contains_any":
            actual = final_answer_text(output).lower()
            expected_values = expected if isinstance(expected, list) else [expected]
            return {
                "passed": any(str(item).lower() in actual for item in expected_values),
                "expected": expected_values,
            }
        raise ValueError(f"unsupported embedded scoring rule: {sample['scoring_rule']}")

    def summarize_results(self, results: list[dict[str, Any]]) -> dict[str, Any]:
        total = len(results)
        passed = sum(1 for result in results if result.get("passed"))
        return {
            "total": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate": round(passed / total, 4) if total else 0.0,
        }


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def run_command(command: list[str], cwd: Path | None = None) -> None:
    completed = subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n"
            f"stdout:\n{completed.stdout[-4000:]}\n"
            f"stderr:\n{completed.stderr[-4000:]}"
        )


def repo_root_from_local_script() -> Path | None:
    candidate = Path.cwd()
    if (candidate / "scripts" / "run_qwen36_benchmark_subsets.py").exists():
        return candidate
    script_parent = Path(__file__).resolve().parent
    candidate = script_parent.parent
    if (candidate / "scripts" / "run_qwen36_benchmark_subsets.py").exists():
        return candidate
    return None


def checkout_opentq(repo_url: str, repo_ref: str, checkout_dir: Path) -> Path:
    local_root = repo_root_from_local_script()
    if local_root is not None:
        return local_root
    if checkout_dir.exists():
        run_command(["git", "fetch", "--depth", "1", "origin", repo_ref], cwd=checkout_dir)
        run_command(["git", "checkout", "FETCH_HEAD"], cwd=checkout_dir)
        return checkout_dir
    run_command(["git", "clone", "--depth", "1", repo_url, str(checkout_dir)])
    if repo_ref != DEFAULT_REPO_REF:
        run_command(["git", "fetch", "--depth", "1", "origin", repo_ref], cwd=checkout_dir)
        run_command(["git", "checkout", "FETCH_HEAD"], cwd=checkout_dir)
    return checkout_dir


def load_runner(repo_url: str, repo_ref: str, checkout_dir: Path) -> Any:
    local_root = repo_root_from_local_script()
    if local_root is not None:
        sys.path.insert(0, str(local_root))
        return importlib.import_module("scripts.run_qwen36_benchmark_subsets")
    if os.environ.get("OPENTQ_USE_EMBEDDED_RUNNER", "1") != "0":
        return EmbeddedSmokeRunner()
    repo_root = checkout_opentq(repo_url, repo_ref, checkout_dir)
    sys.path.insert(0, str(repo_root))
    return importlib.import_module("scripts.run_qwen36_benchmark_subsets")


def selected_adapters(runner: Any, benchmark_ids: list[str]) -> list[Any]:
    missing = [benchmark_id for benchmark_id in benchmark_ids if benchmark_id not in runner.ADAPTERS]
    if missing:
        raise ValueError(f"unsupported benchmark id(s): {', '.join(missing)}")
    return [runner.ADAPTERS[benchmark_id] for benchmark_id in benchmark_ids]


def configured_sample(runner: Any, sample: dict[str, Any], mode: ModeConfig) -> dict[str, Any]:
    updated = dict(sample)
    updated["prompt_format"] = mode.prompt_format
    if mode.token_policy == "cap":
        updated["max_tokens"] = min(int(updated["max_tokens"]), mode.max_tokens)
    elif mode.token_policy == "floor":
        updated["max_tokens"] = max(int(updated["max_tokens"]), mode.max_tokens)
    else:
        raise ValueError(f"unsupported token policy: {mode.token_policy}")
    _ = runner
    return updated


def collect_samples(runner: Any, adapters: list[Any], mode: ModeConfig) -> list[dict[str, Any]]:
    samples = []
    for adapter in adapters:
        for sample in runner.samples_for_adapter(adapter, max_samples=mode.samples_per_family):
            samples.append(configured_sample(runner, sample, mode))
    return samples


def model_payload(model_key: str, model_id: str, model_revision: str | None) -> dict[str, Any]:
    return {
        "key": model_key,
        "path": model_id,
        "exists": None,
        "kind": "hf_transformers_bf16",
        "revision": model_revision,
    }


def empty_eval_payload(
    runner: Any,
    model_key: str,
    model_id: str,
    model_revision: str | None,
    mode: ModeConfig,
    benchmarks: list[dict[str, Any]],
    load_seconds: float | None,
) -> dict[str, Any]:
    return {
        "schema": runner.SCHEMA,
        "created_at": now_iso(),
        "model": model_payload(model_key, model_id, model_revision),
        "generation": {
            "runtime": "hf_jobs_transformers",
            "dtype": "bf16",
            "prompt_format_override": mode.prompt_format,
            "temperature": 0.0,
            "top_p": None,
            "top_k": None,
            "seed": None,
            "token_policy": mode.token_policy,
            "max_tokens_override": mode.max_tokens,
            "samples_per_family": mode.samples_per_family,
            "load_seconds": load_seconds,
            "job_id": os.environ.get("JOB_ID"),
            "accelerator": os.environ.get("ACCELERATOR"),
            "cpu_cores": os.environ.get("CPU_CORES"),
            "memory": os.environ.get("MEMORY"),
        },
        "benchmarks": benchmarks,
    }


def load_bf16_model(model_id: str, model_revision: str | None) -> tuple[Any, Any, float]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    token = os.environ.get("HF_TOKEN")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    started = time.perf_counter()
    common_kwargs = {
        "trust_remote_code": True,
        "token": token,
    }
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=model_revision, **common_kwargs)
    model_kwargs = {
        **common_kwargs,
        "revision": model_revision,
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
        "low_cpu_mem_usage": True,
    }
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, attn_implementation="sdpa", **model_kwargs)
    except (TypeError, ValueError):
        model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    model.eval()
    return tokenizer, model, time.perf_counter() - started


def eos_token_ids(tokenizer: Any) -> list[int] | int | None:
    ids = []
    if tokenizer.eos_token_id is not None:
        ids.append(int(tokenizer.eos_token_id))
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end, int) and im_end >= 0:
        ids.append(im_end)
    unique = list(dict.fromkeys(ids))
    if not unique:
        return None
    return unique if len(unique) > 1 else unique[0]


def generate_one(runner: Any, tokenizer: Any, model: Any, sample: dict[str, Any]) -> dict[str, Any]:
    import torch

    prompt = runner.format_qwen_prompt(str(sample["prompt"]), str(sample["prompt_format"]))
    inputs = tokenizer(prompt, return_tensors="pt")
    first_device = next(model.parameters()).device
    inputs = {key: value.to(first_device) for key, value in inputs.items()}
    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": int(sample["max_tokens"]),
        "do_sample": False,
        "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
        "eos_token_id": eos_token_ids(tokenizer),
    }
    started = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(**inputs, **generation_kwargs)
    elapsed = time.perf_counter() - started
    generated_ids = output_ids[0, inputs["input_ids"].shape[-1] :]
    output = tokenizer.decode(generated_ids, skip_special_tokens=False).strip()
    score = runner.score_benchmark_output(sample, output)
    return {
        "task_id": sample["task_id"],
        "benchmark_id": sample["benchmark_id"],
        "returncode": 0,
        "stdout_tail": output[-4000:],
        "stderr_tail": "",
        "score": score,
        "passed": bool(score.get("passed")),
        "elapsed_seconds": round(elapsed, 3),
        "generated_tokens": int(generated_ids.numel()),
    }


def run_mode(
    runner: Any,
    tokenizer: Any,
    model: Any,
    model_id: str,
    model_revision: str | None,
    adapters: list[Any],
    mode: ModeConfig,
    load_seconds: float,
) -> dict[str, Any]:
    benchmark_payloads = []
    for adapter in adapters:
        samples = [
            configured_sample(runner, sample, mode)
            for sample in runner.samples_for_adapter(adapter, max_samples=mode.samples_per_family)
        ]
        results = [generate_one(runner, tokenizer, model, sample) for sample in samples]
        benchmark_payloads.append(
            {
                "benchmark_id": adapter.benchmark_id,
                "adapter": adapter.as_plan_payload(),
                "samples": samples,
                "results": results,
                "summary": runner.summarize_results(results),
            }
        )
    return empty_eval_payload(
        runner=runner,
        model_key=f"bf16_remote_{mode.name}",
        model_id=model_id,
        model_revision=model_revision,
        mode=mode,
        benchmarks=benchmark_payloads,
        load_seconds=round(load_seconds, 3),
    )


def upload_outputs(
    payloads: dict[str, dict[str, Any]],
    upload_repo: str,
    upload_private: bool,
    optional: bool,
) -> list[str]:
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    job_id = os.environ.get("JOB_ID") or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    uploaded = []
    try:
        api.create_repo(upload_repo, repo_type="dataset", private=upload_private, exist_ok=True)
        for name, payload in payloads.items():
            path_in_repo = f"runs/{job_id}/{name}.json"
            api.upload_file(
                repo_id=upload_repo,
                repo_type="dataset",
                path_in_repo=path_in_repo,
                path_or_fileobj=(json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"),
            )
            uploaded.append(f"hf://datasets/{upload_repo}/{path_in_repo}")
    except Exception as exc:
        if not optional:
            raise
        print(f"optional upload failed: {exc}", file=sys.stderr)
    return uploaded


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a low-cost remote BF16 sidecar for Qwen3.6-27B on Hugging Face Jobs.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--model-revision")
    parser.add_argument("--repo-url", default=os.environ.get("OPENTQ_REPO_URL", DEFAULT_REPO_URL))
    parser.add_argument("--repo-ref", default=os.environ.get("OPENTQ_REF", DEFAULT_REPO_REF))
    parser.add_argument("--checkout-dir", type=Path, default=Path("/tmp/opentq"))
    parser.add_argument("--benchmarks", default="mmlu_pro,gpqa,aime")
    parser.add_argument("--no-think-samples", type=int, default=4)
    parser.add_argument("--thinking-samples", type=int, default=2)
    parser.add_argument("--no-think-max-tokens", type=int, default=512)
    parser.add_argument("--thinking-max-tokens", type=int, default=2048)
    parser.add_argument("--skip-thinking", action="store_true")
    parser.add_argument("--upload-repo", default=DEFAULT_UPLOAD_REPO)
    parser.add_argument("--no-upload", action="store_true")
    parser.add_argument("--upload-public", action="store_true")
    parser.add_argument("--strict-upload", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.model_id != DEFAULT_MODEL_ID:
        raise ValueError(f"refusing to run the wrong Qwen model: {args.model_id!r}; expected {DEFAULT_MODEL_ID!r}")
    runner = load_runner(args.repo_url, args.repo_ref, args.checkout_dir)
    adapters = selected_adapters(runner, parse_csv(args.benchmarks))
    modes = [
        ModeConfig(
            name="no_think",
            prompt_format="qwen3-no-think",
            samples_per_family=args.no_think_samples,
            max_tokens=args.no_think_max_tokens,
            token_policy="cap",
        )
    ]
    if not args.skip_thinking:
        modes.append(
            ModeConfig(
                name="thinking",
                prompt_format="qwen3-thinking",
                samples_per_family=args.thinking_samples,
                max_tokens=args.thinking_max_tokens,
                token_policy="floor",
            )
        )
    print(
        json.dumps(
            {
                "event": "sidecar_start",
                "model_id": args.model_id,
                "model_revision": args.model_revision,
                "repo_ref": args.repo_ref,
                "benchmarks": [adapter.benchmark_id for adapter in adapters],
                "modes": [mode.__dict__ for mode in modes],
                "job_id": os.environ.get("JOB_ID"),
                "accelerator": os.environ.get("ACCELERATOR"),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if isinstance(runner, EmbeddedSmokeRunner):
        for mode in modes:
            sample_count = len(collect_samples(runner, adapters, mode))
            print(
                json.dumps(
                    {
                        "event": "sidecar_prefetch_complete",
                        "mode": mode.name,
                        "sample_count": sample_count,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    tokenizer, model, load_seconds = load_bf16_model(args.model_id, args.model_revision)
    payloads = {
        mode.name: run_mode(
            runner=runner,
            tokenizer=tokenizer,
            model=model,
            model_id=args.model_id,
            model_revision=args.model_revision,
            adapters=adapters,
            mode=mode,
            load_seconds=load_seconds,
        )
        for mode in modes
    }
    uploaded = []
    if not args.no_upload:
        uploaded = upload_outputs(
            payloads=payloads,
            upload_repo=args.upload_repo,
            upload_private=not args.upload_public,
            optional=not args.strict_upload,
        )
    envelope = {
        "schema": "opentq.qwen36_bf16_remote_sidecar.v1",
        "created_at": now_iso(),
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "repo_url": args.repo_url,
        "repo_ref": args.repo_ref,
        "upload_repo": None if args.no_upload else args.upload_repo,
        "uploaded": uploaded,
        "payloads": payloads,
    }
    print(BEGIN_MARKER)
    print(json.dumps(envelope, sort_keys=True))
    print(END_MARKER)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
