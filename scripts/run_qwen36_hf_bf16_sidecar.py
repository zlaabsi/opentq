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
BEGIN_MARKER = "BEGIN_OPENTQ_QWEN36_BF16_SIDECAR_JSON"
END_MARKER = "END_OPENTQ_QWEN36_BF16_SIDECAR_JSON"


@dataclass(frozen=True)
class ModeConfig:
    name: str
    prompt_format: str
    samples_per_family: int
    max_tokens: int
    token_policy: str


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


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
