#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import os
import pickle
import socket
import subprocess
import tempfile
import re
import time
import zlib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests


SCHEMA = "opentq.qwen36_benchmark_subset_eval.v1"
DATASET_VIEWER = "https://datasets-server.huggingface.co"
HF_DATASETS_RESOLVE = "https://huggingface.co/datasets"
DATASET_VIEWER_RETRIES = 5
DATASET_VIEWER_RETRY_STATUSES = {429, 500, 502, 503, 504}
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
EXTERNAL_HARNESS_SCORING_RULES = {"swe_bench_verified_harness"}
OVERSIZED_LOCAL_MODEL_RAM_FRACTION = 0.9

MODEL_PATHS = {
    "q3": Path("artifacts/hf-gguf-canonical/Qwen3.6-27B-OTQ-GGUF/Qwen3.6-27B-OTQ-DYN-Q3_K_M.gguf"),
    "q4": Path("artifacts/hf-gguf-canonical/Qwen3.6-27B-OTQ-GGUF/Qwen3.6-27B-OTQ-DYN-Q4_K_M.gguf"),
    "q5": Path("artifacts/hf-gguf-canonical/Qwen3.6-27B-OTQ-GGUF/Qwen3.6-27B-OTQ-DYN-Q5_K_M.gguf"),
    "bf16_gguf": Path("artifacts/qwen3.6-27b-source/Qwen3.6-27B-BF16.gguf"),
    "bf16": Path("Qwen/Qwen3.6-27B"),
}


@dataclass(frozen=True)
class ModelTarget:
    key: str
    path: Path

    def as_payload(self) -> dict[str, Any]:
        kind = "hf_source" if self.key == "bf16" else "gguf_reference" if self.key == "bf16_gguf" else "gguf"
        return {
            "key": self.key,
            "path": str(self.path),
            "exists": self.path.exists() if self.key != "bf16" else None,
            "kind": kind,
        }


@dataclass(frozen=True)
class BenchmarkAdapter:
    benchmark_id: str
    dataset: str
    config: str
    split: str
    revision: str
    task_ids: tuple[str, ...]
    prompt_format: str
    scoring_rule: str
    max_tokens: int
    backend: str = "dataset_viewer"

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
            "backend": self.backend,
        }


def _offsets(count: int) -> tuple[str, ...]:
    return tuple(f"offset:{offset}" for offset in range(count))


def _spread_offsets(total: int, count: int) -> tuple[str, ...]:
    if total <= 0:
        raise ValueError("total must be positive")
    if count <= 0:
        raise ValueError("count must be positive")
    if count == 1:
        return ("offset:0",)
    selected = []
    seen = set()
    target_count = min(total, count)
    for index in range(target_count):
        offset = round(index * (total - 1) / (target_count - 1))
        if offset in seen:
            continue
        selected.append(f"offset:{offset}")
        seen.add(offset)
    return tuple(selected)


ADAPTERS: dict[str, BenchmarkAdapter] = {
    "mmlu": BenchmarkAdapter(
        "mmlu",
        dataset="cais/mmlu",
        config="all",
        split="test",
        revision="c30699e8356da336a370243923dbaf21066bb9fe",
        task_ids=_spread_offsets(total=14042, count=16),
        prompt_format="qwen3-no-think",
        scoring_rule="multiple_choice_letter",
        max_tokens=16,
    ),
    "mmlu_pro": BenchmarkAdapter(
        "mmlu_pro",
        dataset="TIGER-Lab/MMLU-Pro",
        config="default",
        split="test",
        revision="54611cde22c74cca43dd78732198de6abe971398",
        task_ids=_spread_offsets(total=12032, count=24),
        prompt_format="qwen3-no-think",
        scoring_rule="multiple_choice_letter",
        max_tokens=16,
    ),
    "arc": BenchmarkAdapter(
        "arc",
        dataset="allenai/ai2_arc",
        config="ARC-Challenge",
        split="validation",
        revision="210d026faf9955653af8916fad021475a3f00453",
        task_ids=_spread_offsets(total=299, count=16),
        prompt_format="qwen3-no-think",
        scoring_rule="multiple_choice_letter",
        max_tokens=16,
    ),
    "hellaswag": BenchmarkAdapter(
        "hellaswag",
        dataset="Rowan/hellaswag",
        config="default",
        split="validation",
        revision="218ec52e09a7e7462a5400043bb9a69a41d06b76",
        task_ids=_spread_offsets(total=10042, count=16),
        prompt_format="qwen3-no-think",
        scoring_rule="multiple_choice_letter",
        max_tokens=16,
    ),
    "gsm8k": BenchmarkAdapter(
        "gsm8k",
        dataset="openai/gsm8k",
        config="main",
        split="test",
        revision="740312add88f781978c0658806c59bc2815b9866",
        task_ids=_spread_offsets(total=1319, count=16),
        prompt_format="qwen3-no-think",
        scoring_rule="numeric_exact",
        max_tokens=1024,
    ),
    "math": BenchmarkAdapter(
        "math",
        dataset="EleutherAI/hendrycks_math",
        config="algebra",
        split="test",
        revision="21a5633873b6a120296cce3e2df9d5550074f4a3",
        task_ids=_spread_offsets(total=1187, count=16),
        prompt_format="qwen3-no-think",
        scoring_rule="math_boxed_exact",
        max_tokens=2048,
    ),
    "aime": BenchmarkAdapter(
        "aime",
        dataset="MathArena/aime_2026",
        config="default",
        split="train",
        revision="10b4e45b7a503075d4da8a0d57916a4f06ce6bd2",
        task_ids=_spread_offsets(total=30, count=24),
        prompt_format="qwen3-no-think",
        scoring_rule="numeric_exact",
        max_tokens=4096,
    ),
    "humaneval": BenchmarkAdapter(
        "humaneval",
        dataset="openai/openai_humaneval",
        config="openai_humaneval",
        split="test",
        revision="7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544",
        task_ids=_spread_offsets(total=164, count=12),
        prompt_format="qwen3-no-think",
        scoring_rule="python_unit_tests",
        max_tokens=2048,
    ),
    "mbpp": BenchmarkAdapter(
        "mbpp",
        dataset="google-research-datasets/mbpp",
        config="sanitized",
        split="test",
        revision="4bb6404fdc6cacfda99d4ac4205087b89d32030c",
        task_ids=_spread_offsets(total=257, count=12),
        prompt_format="qwen3-no-think",
        scoring_rule="python_unit_tests",
        max_tokens=2048,
    ),
    "swe_bench": BenchmarkAdapter(
        "swe_bench",
        dataset="princeton-nlp/SWE-bench_Verified",
        config="default",
        split="test",
        revision="c104f840cc67f8b6eec6f759ebc8b2693d585d4a",
        task_ids=_spread_offsets(total=500, count=3),
        prompt_format="qwen3-no-think",
        scoring_rule="swe_bench_verified_harness",
        max_tokens=8192,
    ),
    "livecodebench": BenchmarkAdapter(
        "livecodebench",
        dataset="livecodebench/code_generation_lite",
        config="v6",
        split="test",
        revision="0fe84c3912ea0c4d4a78037083943e8f0c4dd505",
        task_ids=_spread_offsets(total=175, count=12),
        prompt_format="qwen3-no-think",
        scoring_rule="livecodebench_v6_stdin_exact",
        max_tokens=4096,
        backend="hf_raw_jsonl",
    ),
    "bbh": BenchmarkAdapter(
        "bbh",
        dataset="lukaemon/bbh",
        config="boolean_expressions",
        split="test",
        revision="982bb89fd79532a8ac676a61fc42eb1aeec63f99",
        task_ids=_spread_offsets(total=250, count=24),
        prompt_format="qwen3-no-think",
        scoring_rule="exact_text",
        max_tokens=32,
    ),
    "gpqa": BenchmarkAdapter(
        "gpqa",
        dataset="hendrydong/gpqa_diamond_mc",
        config="default",
        split="test",
        revision="284143babc24a94fbac45d143333b2307e64ff80",
        task_ids=_spread_offsets(total=198, count=24),
        prompt_format="qwen3-no-think",
        scoring_rule="multiple_choice_letter",
        max_tokens=16,
    ),
    "ifeval": BenchmarkAdapter(
        "ifeval",
        dataset="google/IFEval",
        config="default",
        split="train",
        revision="966cd89545d6b6acfd7638bc708b98261ca58e84",
        task_ids=_spread_offsets(total=541, count=24),
        prompt_format="qwen3-no-think",
        scoring_rule="ifeval_partial",
        max_tokens=4096,
    ),
    "truthfulqa": BenchmarkAdapter(
        "truthfulqa",
        dataset="truthfulqa/truthful_qa",
        config="multiple_choice",
        split="validation",
        revision="741b8276f2d1982aa3d5b832d3ee81ed3b896490",
        task_ids=_spread_offsets(total=817, count=16),
        prompt_format="qwen3-no-think",
        scoring_rule="multiple_choice_letter",
        max_tokens=16,
    ),
    "winogrande": BenchmarkAdapter(
        "winogrande",
        dataset="allenai/winogrande",
        config="winogrande_debiased",
        split="validation",
        revision="01e74176c63542e6b0bcb004dcdea22d94fb67b5",
        task_ids=_spread_offsets(total=1267, count=16),
        prompt_format="qwen3-no-think",
        scoring_rule="multiple_choice_letter",
        max_tokens=16,
    ),
    "drop": BenchmarkAdapter(
        "drop",
        dataset="ucinlp/drop",
        config="default",
        split="validation",
        revision="95cda593fae71b60b5b19f82de3fcf3298c1239c",
        task_ids=_spread_offsets(total=9535, count=16),
        prompt_format="qwen3-no-think",
        scoring_rule="contains_any",
        max_tokens=256,
    ),
    "piqa": BenchmarkAdapter(
        "piqa",
        dataset="lighteval/piqa",
        config="plain_text",
        split="validation",
        revision="41782e6bf0ef7de82a2ca8a9feb1dca042837fae",
        task_ids=_spread_offsets(total=1838, count=16),
        prompt_format="qwen3-no-think",
        scoring_rule="multiple_choice_letter",
        max_tokens=16,
    ),
    "commonsenseqa": BenchmarkAdapter(
        "commonsenseqa",
        dataset="tau/commonsense_qa",
        config="default",
        split="validation",
        revision="94630fe30dad47192a8546eb75f094926d47e155",
        task_ids=_spread_offsets(total=1221, count=16),
        prompt_format="qwen3-no-think",
        scoring_rule="multiple_choice_letter",
        max_tokens=16,
    ),
}


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def load_matrix(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "opentq.qwen36_long_running_benchmark_matrix.v1":
        raise ValueError(f"unsupported benchmark matrix schema: {path}")
    return payload


def parse_models(value: str) -> list[ModelTarget]:
    targets: list[ModelTarget] = []
    for raw in value.split(","):
        key = raw.strip().lower()
        if not key:
            continue
        if key not in MODEL_PATHS:
            allowed = ", ".join(sorted(MODEL_PATHS))
            raise ValueError(f"unsupported model key {key!r}; expected one of: {allowed}")
        targets.append(ModelTarget(key=key, path=MODEL_PATHS[key]))
    if not targets:
        raise ValueError("at least one model key is required")
    return targets


def parse_task_offset(task_id: str) -> int:
    if not task_id.startswith("offset:"):
        raise ValueError(f"unsupported task id {task_id!r}; expected offset:<int>")
    return int(task_id.split(":", 1)[1])


def letter_for_index(index: int) -> str:
    if index < 0 or index >= len(LETTERS):
        raise ValueError(f"choice index out of range: {index}")
    return LETTERS[index]


def multiple_choice_prompt(question: str, choices: list[str], labels: list[str] | None = None) -> str:
    if labels is None:
        labels = [letter_for_index(index) for index in range(len(choices))]
    rendered = "\n".join(f"({label}) {choice}" for label, choice in zip(labels, choices, strict=True))
    return f"{question}\n\n{rendered}\n\nReturn only the letter of the correct answer."


def extract_hash_answer(answer: str) -> str:
    match = re.search(r"####\s*([-+]?\d+(?:\.\d+)?)", answer)
    return match.group(1) if match else answer.strip()


def extract_boxed_answer(text: str) -> str:
    match = re.search(r"\\boxed\{([^{}]+)\}", text)
    return match.group(1).strip() if match else text.strip()


def parse_json_list(value: Any) -> list[dict[str, Any]]:
    if value in (None, ""):
        return []
    if isinstance(value, list):
        return [dict(item) for item in value]
    raw = str(value)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        # LiveCodeBench lite stores private tests as base64(zlib(pickle(JSON))).
        # The dataset revision is pinned in ADAPTERS before this path is used.
        unpacked = pickle.loads(zlib.decompress(base64.b64decode(raw.encode("utf-8"))))
        if isinstance(unpacked, bytes):
            unpacked = unpacked.decode("utf-8")
        parsed = json.loads(unpacked) if isinstance(unpacked, str) else unpacked
    if not isinstance(parsed, list):
        raise ValueError(f"expected JSON list, got {type(parsed).__name__}")
    return [dict(item) for item in parsed]


def build_sample_from_row(adapter: BenchmarkAdapter, task_id: str, row: dict[str, Any]) -> dict[str, Any]:
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
    benchmark_id = adapter.benchmark_id
    if benchmark_id == "mmlu":
        choices = [str(item) for item in row["choices"]]
        sample.update(
            prompt=multiple_choice_prompt(str(row["question"]), choices),
            answer=letter_for_index(int(row["answer"])),
            category=row.get("subject"),
        )
    elif benchmark_id == "mmlu_pro":
        choices = [str(item) for item in row["options"]]
        sample.update(
            prompt=multiple_choice_prompt(str(row["question"]), choices),
            answer=letter_for_index(int(row["answer_index"])),
            category=row.get("category"),
        )
    elif benchmark_id == "arc":
        choices = [str(item) for item in row["choices"]["text"]]
        labels = [str(item) for item in row["choices"]["label"]]
        sample.update(prompt=multiple_choice_prompt(str(row["question"]), choices, labels), answer=str(row["answerKey"]))
    elif benchmark_id == "hellaswag":
        sample.update(prompt=multiple_choice_prompt(str(row["ctx"]), [str(item) for item in row["endings"]]), answer=letter_for_index(int(row["label"])))
    elif benchmark_id == "gsm8k":
        sample.update(prompt=f"{row['question']}\n\nReturn only the final numeric answer.", answer=extract_hash_answer(str(row["answer"])))
    elif benchmark_id == "math":
        sample.update(prompt=f"{row['problem']}\n\nReturn only the final answer.", answer=extract_boxed_answer(str(row["solution"])), category=row.get("type"))
    elif benchmark_id == "aime":
        sample.update(prompt=f"{row['problem']}\n\nReturn only the final integer answer.", answer=str(row["answer"]))
    elif benchmark_id == "humaneval":
        sample.update(
            prompt=f"Complete this Python function. Return only Python code.\n\n{row['prompt']}",
            answer=None,
            python_tests=str(row["test"]),
            entry_point=row.get("entry_point"),
            source_task_id=row.get("task_id"),
        )
    elif benchmark_id == "mbpp":
        sample.update(
            prompt=f"{row['prompt']}\n\nReturn only Python code.",
            answer=None,
            python_tests="\n".join(str(item) for item in row["test_list"]),
            source_task_id=row.get("task_id"),
        )
    elif benchmark_id == "swe_bench":
        hints = str(row.get("hints_text") or "").strip()
        hints_block = f"\n\nHints:\n{hints}" if hints else ""
        sample.update(
            prompt=(
                "You are fixing a real GitHub issue from SWE-bench Verified.\n"
                "Return only a unified diff patch that applies to the repository at the base commit.\n\n"
                f"Repository: {row['repo']}\n"
                f"Base commit: {row['base_commit']}\n"
                f"Instance id: {row['instance_id']}\n\n"
                f"Issue:\n{row['problem_statement']}{hints_block}"
            ),
            answer=None,
            source_instance_id=row.get("instance_id"),
            repo=row.get("repo"),
            base_commit=row.get("base_commit"),
            fail_to_pass=row.get("FAIL_TO_PASS"),
            pass_to_pass=row.get("PASS_TO_PASS"),
            harness_required=True,
            scoring_note="SWE-bench Verified requires the official repository checkout and test harness; no pass/fail is computed by this GGUF runner.",
        )
    elif benchmark_id == "livecodebench":
        public_tests = parse_json_list(row.get("public_test_cases"))
        private_tests = parse_json_list(row.get("private_test_cases"))
        starter_code = str(row.get("starter_code") or "").strip()
        starter_block = f"\n\nStarter code:\n```python\n{starter_code}\n```" if starter_code else ""
        sample.update(
            prompt=(
                "Write a Python 3 solution for this programming contest problem. "
                "Read from stdin and write to stdout. Return only executable Python code, no Markdown.\n\n"
                f"Title: {row['question_title']}\n"
                f"Platform: {row['platform']}\n"
                f"Question id: {row['question_id']}\n"
                f"Contest date: {row['contest_date']}\n"
                f"Difficulty: {row['difficulty']}\n\n"
                f"{row['question_content']}{starter_block}"
            ),
            answer=None,
            source_question_id=row.get("question_id"),
            contest_date=row.get("contest_date"),
            platform=row.get("platform"),
            difficulty=row.get("difficulty"),
            public_test_cases=public_tests,
            private_test_cases=private_tests,
            scoring_note="LiveCodeBench lite v6 stdin exact-match scorer using the pinned test6.jsonl public and private test cases.",
        )
    elif benchmark_id == "bbh":
        sample.update(prompt=f"{row['input']}\n\nReturn only the answer.", answer=str(row["target"]))
    elif benchmark_id == "gpqa":
        sample.update(prompt=str(row["problem"]), answer=extract_boxed_answer(str(row["solution"])).replace("\\boxed{", "").replace("}", ""))
    elif benchmark_id == "ifeval":
        sample.update(prompt=str(row["prompt"]), answer=None, instruction_id_list=row.get("instruction_id_list", []), kwargs=row.get("kwargs", []))
    elif benchmark_id == "truthfulqa":
        choices = [str(item) for item in row["mc1_targets"]["choices"]]
        labels = [int(item) for item in row["mc1_targets"]["labels"]]
        sample.update(prompt=multiple_choice_prompt(str(row["question"]), choices), answer=letter_for_index(labels.index(1)))
    elif benchmark_id == "winogrande":
        question = str(row["sentence"]).replace("_", "____")
        sample.update(prompt=multiple_choice_prompt(question, [str(row["option1"]), str(row["option2"])]), answer=letter_for_index(int(row["answer"]) - 1))
    elif benchmark_id == "drop":
        sample.update(prompt=f"{row['passage']}\n\nQuestion: {row['question']}\n\nReturn only the answer.", answer=[str(item) for item in row["answers_spans"]["spans"]])
    elif benchmark_id == "piqa":
        sample.update(prompt=multiple_choice_prompt(str(row["goal"]), [str(row["sol1"]), str(row["sol2"])]), answer=letter_for_index(int(row["label"])))
    elif benchmark_id == "commonsenseqa":
        choices = [str(item) for item in row["choices"]["text"]]
        labels = [str(item) for item in row["choices"]["label"]]
        sample.update(prompt=multiple_choice_prompt(str(row["question"]), choices, labels), answer=str(row["answerKey"]))
    else:
        raise ValueError(f"unsupported adapter: {benchmark_id}")
    return sample


def fetch_dataset_viewer_row(adapter: BenchmarkAdapter, task_id: str) -> dict[str, Any]:
    offset = parse_task_offset(task_id)
    params = {
        "dataset": adapter.dataset,
        "config": adapter.config,
        "split": adapter.split,
        "revision": adapter.revision,
        "offset": offset,
        "length": 1,
    }
    response = None
    for attempt in range(1, DATASET_VIEWER_RETRIES + 1):
        try:
            response = requests.get(f"{DATASET_VIEWER}/rows", params=params, timeout=60)
            if response.status_code in DATASET_VIEWER_RETRY_STATUSES and attempt < DATASET_VIEWER_RETRIES:
                time.sleep(min(2 ** (attempt - 1), 8))
                continue
            response.raise_for_status()
            break
        except requests.RequestException:
            if attempt >= DATASET_VIEWER_RETRIES:
                raise
            time.sleep(min(2 ** (attempt - 1), 8))
    if response is None:
        raise RuntimeError("dataset viewer request did not return a response")
    rows = response.json().get("rows", [])
    if not rows:
        raise ValueError(f"no row returned for {adapter.benchmark_id} {task_id}")
    return dict(rows[0]["row"])


def fetch_datasets_library_row(adapter: BenchmarkAdapter, task_id: str) -> dict[str, Any]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("datasets package is required for this adapter") from exc
    offset = parse_task_offset(task_id)
    dataset = load_dataset(adapter.dataset, adapter.config, split=adapter.split, revision=adapter.revision)
    return dict(dataset[offset])


def raw_jsonl_filename(adapter: BenchmarkAdapter) -> str:
    if adapter.dataset == "livecodebench/code_generation_lite" and adapter.config == "v6":
        return "test6.jsonl"
    raise ValueError(f"unsupported raw JSONL adapter: {adapter.benchmark_id}")


def fetch_hf_raw_jsonl_row(adapter: BenchmarkAdapter, task_id: str) -> dict[str, Any]:
    offset = parse_task_offset(task_id)
    dataset = quote(adapter.dataset, safe="/")
    filename = quote(raw_jsonl_filename(adapter), safe="/")
    url = f"{HF_DATASETS_RESOLVE}/{dataset}/resolve/{adapter.revision}/{filename}"
    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()
    for index, line in enumerate(response.iter_lines(decode_unicode=True)):
        if index == offset:
            if not line:
                raise ValueError(f"empty JSONL row for {adapter.benchmark_id} {task_id}")
            return dict(json.loads(line))
    raise ValueError(f"no row returned for {adapter.benchmark_id} {task_id}")


def fetch_row(adapter: BenchmarkAdapter, task_id: str) -> dict[str, Any]:
    if adapter.backend == "hf_raw_jsonl":
        return fetch_hf_raw_jsonl_row(adapter, task_id)
    if adapter.backend == "datasets_library":
        return fetch_datasets_library_row(adapter, task_id)
    return fetch_dataset_viewer_row(adapter, task_id)


def format_qwen_prompt(prompt: str, prompt_format: str) -> str:
    if prompt_format == "raw":
        return prompt
    if prompt_format == "qwen3-no-think":
        return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    if prompt_format == "qwen3-thinking":
        return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n"
    raise ValueError(f"unsupported prompt_format: {prompt_format}")


def generation_binary(llama_cpp: Path) -> Path:
    completion = llama_cpp / "build" / "bin" / "llama-completion"
    if completion.exists():
        return completion
    return llama_cpp / "build" / "bin" / "llama-cli"


def server_binary(llama_cpp: Path) -> Path:
    return llama_cpp / "build" / "bin" / "llama-server"


def physical_memory_bytes() -> int | None:
    try:
        return int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, OSError, ValueError):
        pass
    try:
        completed = subprocess.run(["sysctl", "-n", "hw.memsize"], text=True, capture_output=True, check=False, timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    try:
        return int(completed.stdout.strip())
    except ValueError:
        return None


def gib(value: int) -> float:
    return value / float(1024**3)


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


def extract_python_code(output: str) -> str:
    cleaned = final_answer_text(output)
    fenced = re.search(r"```(?:python|py)?\s*(.*?)```", cleaned, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        return fenced.group(1).strip()
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


def run_python_unit_tests(code: str, tests: str) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "candidate_test.py"
        path.write_text(f"{extract_python_code(code)}\n\n{tests}\n", encoding="utf-8")
        completed = subprocess.run(["python", str(path)], text=True, capture_output=True, check=False, timeout=20)
    return {
        "passed": completed.returncode == 0,
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-2000:],
        "stderr_tail": completed.stderr[-2000:],
    }


def normalize_stdio(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.strip().splitlines())


def run_livecodebench_stdin_tests(code: str, sample: dict[str, Any]) -> dict[str, Any]:
    cases = [*sample.get("public_test_cases", []), *sample.get("private_test_cases", [])]
    if not cases:
        return {"passed": None, "reason": "no_test_cases"}
    unsupported = [case.get("testtype") for case in cases if case.get("testtype") != "stdin"]
    if unsupported:
        return {
            "passed": None,
            "reason": "unsupported_livecodebench_testtype",
            "unsupported_testtypes": sorted({str(item) for item in unsupported}),
        }
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "solution.py"
        path.write_text(extract_python_code(code) + "\n", encoding="utf-8")
        failures = []
        for index, case in enumerate(cases):
            completed = subprocess.run(
                ["python", str(path)],
                input=str(case.get("input", "")),
                text=True,
                capture_output=True,
                check=False,
                timeout=10,
            )
            expected = normalize_stdio(str(case.get("output", "")))
            actual = normalize_stdio(completed.stdout)
            if completed.returncode != 0 or actual != expected:
                failures.append(
                    {
                        "case_index": index,
                        "returncode": completed.returncode,
                        "expected_tail": expected[-500:],
                        "actual_tail": actual[-500:],
                        "stderr_tail": completed.stderr[-500:],
                    }
                )
                if len(failures) >= 3:
                    break
    return {
        "passed": not failures,
        "total_cases": len(cases),
        "checked_cases": len(cases) if not failures else failures[-1]["case_index"] + 1,
        "failures": failures,
    }


def score_ifeval_partial(sample: dict[str, Any], output: str) -> dict[str, Any]:
    text = clean_generated_text(output)
    checks = []
    for instruction, kwargs in zip(sample.get("instruction_id_list", []), sample.get("kwargs", []), strict=False):
        kwargs = kwargs or {}
        if instruction == "punctuation:no_comma":
            checks.append({"instruction": instruction, "passed": "," not in text})
        elif instruction == "detectable_format:number_highlighted_sections":
            expected = int(kwargs.get("num_highlights") or 0)
            found = len(re.findall(r"\*[^*]+\*", text))
            checks.append({"instruction": instruction, "passed": found >= expected, "found": found, "expected": expected})
        elif instruction == "length_constraints:number_words":
            expected = int(kwargs.get("num_words") or 0)
            words = len(re.findall(r"\S+", text))
            checks.append({"instruction": instruction, "passed": words >= expected, "found": words, "expected": expected})
        else:
            checks.append({"instruction": instruction, "passed": None, "reason": "unsupported_partial_checker"})
    supported = [item for item in checks if item["passed"] is not None]
    return {
        "passed": bool(supported) and all(bool(item["passed"]) for item in supported),
        "checks": checks,
        "unsupported_count": len(checks) - len(supported),
        "scoring_note": "partial IFEval checker; unsupported instruction ids are reported and not used for public deltas",
    }


def score_benchmark_output(sample: dict[str, Any], output: str) -> dict[str, Any]:
    scoring_rule = str(sample["scoring_rule"])
    expected = sample.get("answer")
    if scoring_rule == "multiple_choice_letter":
        actual = extract_multiple_choice_letter(output)
        return {"passed": actual == expected, "expected": expected, "actual": actual}
    if scoring_rule in {"numeric_exact", "math_boxed_exact"}:
        final_text = final_answer_text(output)
        actual = extract_boxed_answer(final_text)
        if actual == final_text:
            actual = extract_number(output) or actual
        return {"passed": str(actual).strip() == str(expected).strip(), "expected": expected, "actual": actual}
    if scoring_rule == "exact_text":
        actual = final_answer_text(output)
        return {"passed": actual.lower() == str(expected).lower(), "expected": expected, "actual": actual}
    if scoring_rule == "contains_any":
        actual = final_answer_text(output).lower()
        expected_values = expected if isinstance(expected, list) else [expected]
        return {"passed": any(str(item).lower() in actual for item in expected_values), "expected": expected_values}
    if scoring_rule == "python_unit_tests":
        return run_python_unit_tests(clean_generated_text(output), str(sample["python_tests"]))
    if scoring_rule == "ifeval_partial":
        return score_ifeval_partial(sample, output)
    if scoring_rule == "swe_bench_verified_harness":
        return {
            "passed": None,
            "reason": "requires_external_swe_bench_verified_harness",
            "scoring_note": "Patch generation can be captured, but pass/fail must come from the official SWE-bench harness.",
        }
    if scoring_rule == "livecodebench_v6_stdin_exact":
        return run_livecodebench_stdin_tests(output, sample)
    raise ValueError(f"unsupported scoring rule: {scoring_rule}")


def sample_count(row: dict[str, Any], defaults: dict[str, Any], sample_mode: str) -> int:
    if sample_mode == "smoke":
        return 4
    explicit_count = re.match(r"^(\d+)\b", str(row.get("subset_policy", "")))
    if explicit_count:
        return int(explicit_count.group(1))
    family = str(row.get("family", ""))
    if "agentic" in family:
        return int(defaults.get("agentic_samples_per_family", 3))
    if "coding" in family:
        return int(defaults.get("coding_samples_per_family", 12))
    if "judge" in family or family == "arena":
        return int(defaults.get("judge_samples_per_family", 12))
    if "reasoning" in family or family in {"math", "stem_reasoning"}:
        return int(defaults.get("reasoning_samples_per_family", 24))
    return int(defaults.get("quick_samples_per_family", 16))


def adapter_requires_external_harness(adapter: BenchmarkAdapter | None) -> bool:
    return bool(adapter and adapter.scoring_rule in EXTERNAL_HARNESS_SCORING_RULES)


def row_status(row: dict[str, Any], allow_judge: bool, allow_external_harness: bool = False) -> str:
    mode = str(row.get("comparison_mode", ""))
    if mode == "blocked_modality":
        return "blocked_modality"
    if mode == "judge_based" and not allow_judge:
        return "skipped_judge"
    adapter = ADAPTERS.get(str(row.get("id")))
    if adapter is None:
        return "missing_adapter"
    if adapter_requires_external_harness(adapter) and not allow_external_harness:
        return "requires_external_harness"
    return "planned"


def build_plan(
    matrix: dict[str, Any],
    models: list[ModelTarget],
    output_root: Path,
    llama_cpp: Path,
    sample_mode: str,
    dry_run: bool,
    allow_judge: bool,
    allow_external_harness: bool,
) -> dict[str, Any]:
    defaults = dict(matrix.get("default_subset_policy", {}))
    benchmarks = []
    for row in matrix.get("benchmarks", []):
        status = row_status(row, allow_judge, allow_external_harness)
        adapter = ADAPTERS.get(str(row["id"]))
        benchmarks.append(
            {
                "id": row["id"],
                "name": row["name"],
                "family": row["family"],
                "comparison_mode": row["comparison_mode"],
                "status": status,
                "sample_count": 0
                if status in {"blocked_modality", "skipped_judge", "requires_external_harness"}
                else sample_count(row, defaults, sample_mode),
                "official_baseline": row.get("official_baseline"),
                "subset_policy": row["subset_policy"],
                "claim_rule": row["claim_rule"],
                "adapter": adapter.as_plan_payload() if adapter else None,
            }
        )
    return {
        "schema": SCHEMA,
        "created_at": now_iso(),
        "dry_run": dry_run,
        "sample_mode": sample_mode,
        "llama_cpp": str(llama_cpp),
        "output_root": str(output_root),
        "models": [model.as_payload() for model in models],
        "benchmarks": benchmarks,
    }


def print_plan(plan: dict[str, Any]) -> None:
    for model in plan["models"]:
        exists = model["exists"]
        exists_text = "remote" if exists is None else str(bool(exists)).lower()
        print(f"model {model['key']} kind={model['kind']} exists={exists_text} path={model['path']}")
    for row in plan["benchmarks"]:
        print(
            f"{row['status']} {row['id']} mode={row['comparison_mode']} "
            f"samples={row['sample_count']} claim={row['claim_rule']}"
        )
        adapter = row.get("adapter")
        if adapter:
            print(
                f"  adapter dataset={adapter['dataset']} config={adapter['config']} "
                f"split={adapter['split']} revision={adapter['revision']} scoring={adapter['scoring_rule']}"
            )


def selected_adapters(
    matrix: dict[str, Any],
    allow_judge: bool,
    allow_external_harness: bool,
    benchmark_ids: set[str] | None,
) -> list[BenchmarkAdapter]:
    adapters = []
    for row in matrix.get("benchmarks", []):
        benchmark_id = str(row["id"])
        if benchmark_ids is not None and benchmark_id not in benchmark_ids:
            continue
        if row_status(row, allow_judge, allow_external_harness) != "planned":
            continue
        adapter = ADAPTERS.get(benchmark_id)
        if adapter is not None:
            adapters.append(adapter)
    return adapters


def samples_for_adapter(adapter: BenchmarkAdapter, max_samples: int | None = None) -> list[dict[str, Any]]:
    task_ids = adapter.task_ids[:max_samples] if max_samples is not None else adapter.task_ids
    samples = []
    for task_id in task_ids:
        row = fetch_row(adapter, task_id)
        samples.append(build_sample_from_row(adapter, task_id, row))
    return samples


def apply_generation_overrides(sample: dict[str, Any], max_tokens: int | None, prompt_format: str | None) -> dict[str, Any]:
    updated = dict(sample)
    if max_tokens is not None:
        updated["max_tokens"] = min(int(updated["max_tokens"]), max_tokens)
    if prompt_format is not None:
        updated["prompt_format"] = prompt_format
    return updated


def apply_max_tokens(sample: dict[str, Any], max_tokens: int | None) -> dict[str, Any]:
    return apply_generation_overrides(sample, max_tokens=max_tokens, prompt_format=None)


def generation_command(
    model: ModelTarget,
    llama_cpp: Path,
    sample: dict[str, Any],
    timeout_seconds: int,
    temperature: float = 0.0,
    top_p: float | None = None,
    top_k: int | None = None,
    seed: int | None = None,
    context_size: int = 8192,
    gpu_layers: int = 99,
) -> list[str]:
    _ = timeout_seconds
    command = [
        str(generation_binary(llama_cpp)),
        "-m",
        str(model.path),
        "-ngl",
        str(gpu_layers),
        "-c",
        str(context_size),
        "-n",
        str(sample["max_tokens"]),
        "-p",
        format_qwen_prompt(str(sample["prompt"]), str(sample["prompt_format"])),
        "-fa",
        "on",
        "-no-cnv",
        "--simple-io",
        "--no-warmup",
        "--no-display-prompt",
        "--log-verbosity",
        "1",
        "--temp",
        str(temperature),
    ]
    if top_p is not None:
        command.extend(["--top-p", str(top_p)])
    if top_k is not None:
        command.extend(["--top-k", str(top_k)])
    if seed is not None:
        command.extend(["--seed", str(seed)])
    return command


def run_generation(
    model: ModelTarget,
    llama_cpp: Path,
    sample: dict[str, Any],
    timeout_seconds: int,
    temperature: float,
    top_p: float | None,
    top_k: int | None,
    seed: int | None,
    context_size: int,
    gpu_layers: int,
) -> dict[str, Any]:
    command = generation_command(
        model,
        llama_cpp,
        sample,
        timeout_seconds,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        seed=seed,
        context_size=context_size,
        gpu_layers=gpu_layers,
    )
    started = time.perf_counter()
    completed = subprocess.run(command, text=True, capture_output=True, check=False, timeout=timeout_seconds)
    elapsed = time.perf_counter() - started
    stdout = completed.stdout.strip()
    score = score_benchmark_output(sample, stdout) if completed.returncode == 0 else {"passed": False, "reason": "runtime_error"}
    return {
        "task_id": sample["task_id"],
        "benchmark_id": sample["benchmark_id"],
        "returncode": completed.returncode,
        "stdout_tail": stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
        "score": score,
        "passed": completed.returncode == 0 and bool(score.get("passed")),
        "elapsed_seconds": round(elapsed, 3),
    }


def available_tcp_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def llama_server_command(
    model: ModelTarget,
    llama_cpp: Path,
    host: str,
    port: int,
    timeout_seconds: int,
    context_size: int,
    gpu_layers: int,
) -> list[str]:
    return [
        str(server_binary(llama_cpp)),
        "-m",
        str(model.path),
        "-ngl",
        str(gpu_layers),
        "-c",
        str(context_size),
        "-fa",
        "on",
        "--host",
        host,
        "--port",
        str(port),
        "--no-webui",
        "--log-verbosity",
        "1",
        "-to",
        str(timeout_seconds),
    ]


def wait_for_llama_server(base_url: str, process: subprocess.Popen[Any], start_timeout_seconds: int) -> None:
    deadline = time.monotonic() + start_timeout_seconds
    last_error = ""
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"llama-server exited before readiness with status {process.returncode}")
        try:
            response = requests.get(f"{base_url}/health", timeout=5)
            if response.status_code == 200:
                return
            last_error = f"HTTP {response.status_code}: {response.text[-500:]}"
        except requests.RequestException as exc:
            last_error = str(exc)
        time.sleep(2)
    raise TimeoutError(f"llama-server did not become ready within {start_timeout_seconds}s: {last_error}")


def start_llama_server(
    model: ModelTarget,
    llama_cpp: Path,
    log_path: Path,
    host: str,
    port: int | None,
    timeout_seconds: int,
    start_timeout_seconds: int,
    context_size: int,
    gpu_layers: int,
) -> tuple[str, subprocess.Popen[Any], Any]:
    selected_port = port if port is not None else available_tcp_port(host)
    command = llama_server_command(
        model=model,
        llama_cpp=llama_cpp,
        host=host,
        port=selected_port,
        timeout_seconds=timeout_seconds,
        context_size=context_size,
        gpu_layers=gpu_layers,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("w", encoding="utf-8")
    log_file.write(" ".join(command) + "\n")
    log_file.flush()
    process = subprocess.Popen(command, text=True, stdout=log_file, stderr=subprocess.STDOUT)
    base_url = f"http://{host}:{selected_port}"
    wait_for_llama_server(base_url, process, start_timeout_seconds)
    return base_url, process, log_file


def stop_llama_server(process: subprocess.Popen[Any], log_file: Any) -> None:
    try:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=30)
    finally:
        log_file.close()


def run_server_generation(
    base_url: str,
    sample: dict[str, Any],
    timeout_seconds: int,
    temperature: float,
    top_p: float | None,
    top_k: int | None,
    seed: int | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "prompt": format_qwen_prompt(str(sample["prompt"]), str(sample["prompt_format"])),
        "n_predict": int(sample["max_tokens"]),
        "temperature": temperature,
        "cache_prompt": False,
        "stream": False,
    }
    if top_p is not None:
        payload["top_p"] = top_p
    if top_k is not None:
        payload["top_k"] = top_k
    if seed is not None:
        payload["seed"] = seed
    started = time.perf_counter()
    response = requests.post(f"{base_url}/completion", json=payload, timeout=timeout_seconds)
    elapsed = time.perf_counter() - started
    if response.status_code != 200:
        return {
            "task_id": sample["task_id"],
            "benchmark_id": sample["benchmark_id"],
            "returncode": response.status_code,
            "stdout_tail": "",
            "stderr_tail": response.text[-4000:],
            "score": {"passed": False, "reason": "server_http_error"},
            "passed": False,
            "elapsed_seconds": round(elapsed, 3),
        }
    data = response.json()
    stdout = str(data.get("content", "")).strip()
    score = score_benchmark_output(sample, stdout)
    return {
        "task_id": sample["task_id"],
        "benchmark_id": sample["benchmark_id"],
        "returncode": 0,
        "stdout_tail": stdout[-4000:],
        "stderr_tail": "",
        "server_timings": data.get("timings", {}),
        "score": score,
        "passed": bool(score.get("passed")),
        "elapsed_seconds": round(elapsed, 3),
    }


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    passed = sum(1 for result in results if result.get("passed"))
    return {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": round(passed / total, 4) if total else 0.0,
    }


def enforce_local_memory_gate(model: ModelTarget, allow_oversized_local_model: bool) -> None:
    if allow_oversized_local_model or not model.path.exists():
        return
    memory_bytes = physical_memory_bytes()
    if memory_bytes is None:
        return
    model_bytes = model.path.stat().st_size
    threshold = int(memory_bytes * OVERSIZED_LOCAL_MODEL_RAM_FRACTION)
    if model_bytes > threshold:
        raise MemoryError(
            f"refusing to load {model.key} locally: model file is {gib(model_bytes):.1f} GiB, "
            f"physical memory is {gib(memory_bytes):.1f} GiB, safety threshold is "
            f"{OVERSIZED_LOCAL_MODEL_RAM_FRACTION:.0%} of RAM. Use remote BF16 sidecar or pass "
            "--allow-oversized-local-model to override."
        )


def require_runtime(model: ModelTarget, llama_cpp: Path, runtime: str, allow_oversized_local_model: bool) -> None:
    if model.key == "bf16":
        raise ValueError("BF16 benchmark execution is intentionally not supported locally by this GGUF runner")
    if not model.path.exists():
        raise FileNotFoundError(f"missing model file: {model.path}")
    enforce_local_memory_gate(model, allow_oversized_local_model)
    binary = server_binary(llama_cpp) if runtime == "server" else generation_binary(llama_cpp)
    if not binary.exists():
        raise FileNotFoundError(f"missing llama.cpp binary: {binary}")


def run_model_benchmarks(
    model: ModelTarget,
    adapters: list[BenchmarkAdapter],
    output_root: Path,
    llama_cpp: Path,
    max_samples: int | None,
    max_tokens: int | None,
    timeout_seconds: int,
    prompt_format: str | None,
    temperature: float,
    top_p: float | None,
    top_k: int | None,
    seed: int | None,
    context_size: int,
    gpu_layers: int,
    runtime: str,
    server_host: str,
    server_port: int | None,
    server_start_timeout: int,
    allow_oversized_local_model: bool,
) -> dict[str, Any]:
    require_runtime(model, llama_cpp, runtime, allow_oversized_local_model)
    benchmark_payloads = []
    payload = {
        "schema": SCHEMA,
        "created_at": now_iso(),
        "model": model.as_payload(),
        "generation": {
            "prompt_format_override": prompt_format,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "seed": seed,
            "context_size": context_size,
            "gpu_layers": gpu_layers,
            "timeout_seconds": timeout_seconds,
            "runtime": runtime,
            "server_host": server_host if runtime == "server" else None,
            "server_port": server_port if runtime == "server" else None,
            "server_start_timeout_seconds": server_start_timeout if runtime == "server" else None,
            "allow_oversized_local_model": allow_oversized_local_model,
        },
        "benchmarks": benchmark_payloads,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    target = output_root / f"{model.key}.json"
    server: tuple[str, subprocess.Popen[Any], Any] | None = None
    if runtime == "server":
        server = start_llama_server(
            model=model,
            llama_cpp=llama_cpp,
            log_path=output_root / "server-logs" / f"{model.key}.log",
            host=server_host,
            port=server_port,
            timeout_seconds=timeout_seconds,
            start_timeout_seconds=server_start_timeout,
            context_size=context_size,
            gpu_layers=gpu_layers,
        )
        payload["generation"]["server_url"] = server[0]
        target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    try:
        for adapter in adapters:
            samples = [
                apply_generation_overrides(sample, max_tokens=max_tokens, prompt_format=prompt_format)
                for sample in samples_for_adapter(adapter, max_samples=max_samples)
            ]
            if runtime == "server":
                assert server is not None
                results = [
                    run_server_generation(
                        server[0],
                        sample,
                        timeout_seconds,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        seed=seed,
                    )
                    for sample in samples
                ]
            else:
                results = [
                    run_generation(
                        model,
                        llama_cpp,
                        sample,
                        timeout_seconds,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        seed=seed,
                        context_size=context_size,
                        gpu_layers=gpu_layers,
                    )
                    for sample in samples
                ]
            benchmark_payloads.append(
                {
                    "benchmark_id": adapter.benchmark_id,
                    "adapter": adapter.as_plan_payload(),
                    "samples": samples,
                    "results": results,
                    "summary": summarize_results(results),
                }
            )
            target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    finally:
        if server is not None:
            stop_llama_server(server[1], server[2])
    print(target)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan Qwen3.6 benchmark subset runs from the release matrix.")
    parser.add_argument("--matrix", type=Path, default=Path("benchmarks/qwen36_long_running_benchmark_matrix.json"))
    parser.add_argument("--models", default="q3,q4")
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/qwen3.6-27b-benchmark-subsets"))
    parser.add_argument("--llama-cpp", type=Path, default=Path("/Users/zlaabsi/Documents/GitHub/llama.cpp"))
    parser.add_argument("--sample-mode", choices=("smoke", "quick"), default="quick")
    parser.add_argument("--benchmark-id", action="append", default=[])
    parser.add_argument("--max-samples-per-family", type=int)
    parser.add_argument("--max-tokens", type=int, help="Cap generation tokens per sample for practical local subsets.")
    parser.add_argument("--prompt-format", choices=("raw", "qwen3-no-think", "qwen3-thinking"), help="Override adapter prompt format.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--context-size", type=int, default=8192)
    parser.add_argument("--gpu-layers", type=int, default=99)
    parser.add_argument("--runtime", choices=("completion", "server"), default="completion")
    parser.add_argument("--server-host", default="127.0.0.1")
    parser.add_argument("--server-port", type=int)
    parser.add_argument("--server-start-timeout", type=int, default=900)
    parser.add_argument(
        "--allow-oversized-local-model",
        action="store_true",
        help="Override the RAM safety gate for local GGUF files larger than the configured RAM threshold.",
    )
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-judge", action="store_true")
    parser.add_argument(
        "--allow-external-harness",
        action="store_true",
        help="Include adapters that require an external official harness; this runner still records no synthetic pass/fail.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    matrix = load_matrix(args.matrix)
    models = parse_models(args.models)
    plan = build_plan(
        matrix=matrix,
        models=models,
        output_root=args.output_root,
        llama_cpp=args.llama_cpp,
        sample_mode=args.sample_mode,
        dry_run=args.dry_run,
        allow_judge=args.allow_judge,
        allow_external_harness=args.allow_external_harness,
    )
    print_plan(plan)
    if not args.dry_run:
        benchmark_ids = set(args.benchmark_id) if args.benchmark_id else None
        adapters = selected_adapters(
            matrix,
            allow_judge=args.allow_judge,
            allow_external_harness=args.allow_external_harness,
            benchmark_ids=benchmark_ids,
        )
        if not adapters:
            raise ValueError("no executable benchmark adapters selected")
        for model in models:
            run_model_benchmarks(
                model=model,
                adapters=adapters,
                output_root=args.output_root,
                llama_cpp=args.llama_cpp,
                max_samples=args.max_samples_per_family,
                max_tokens=args.max_tokens,
                timeout_seconds=args.timeout,
                prompt_format=args.prompt_format,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                seed=args.seed,
                context_size=args.context_size,
                gpu_layers=args.gpu_layers,
                runtime=args.runtime,
                server_host=args.server_host,
                server_port=args.server_port,
                server_start_timeout=args.server_start_timeout,
                allow_oversized_local_model=args.allow_oversized_local_model,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
