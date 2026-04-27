from __future__ import annotations

import json
import math
import re
import subprocess
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .hf_release import dump_json, load_json


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _tail(text: str, max_chars: int = 12_000) -> str:
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil((percentile / 100.0) * len(ordered)) - 1))
    return round(ordered[index], 3)


@dataclass(frozen=True)
class GGUFQualityEvalOptions:
    gguf: Path
    output: Path
    suite: Path
    llama_cpp_dir: Path = Path("../llama.cpp")
    gpu_layers: str = "99"
    ctx_size: int = 2048
    flash_attn: str = "on"
    timeout_seconds: float = 600.0
    threads: int | None = None
    temperature: float = 0.0
    top_p: float | None = None
    max_samples: int | None = None
    sample_ids: tuple[str, ...] = field(default_factory=tuple)
    reference: Path | None = None


def _load_suite(path: Path, max_samples: int | None, sample_ids: tuple[str, ...]) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"missing quality suite: {path}")
    selected = set(sample_ids)
    samples = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        sample = json.loads(stripped)
        if "id" not in sample:
            raise ValueError(f"sample on line {line_number} is missing id")
        if "prompt" not in sample:
            raise ValueError(f"sample {sample['id']} is missing prompt")
        if selected and sample["id"] not in selected:
            continue
        samples.append(sample)
        if max_samples is not None and len(samples) >= max_samples:
            break
    if selected:
        found = {sample["id"] for sample in samples}
        missing = sorted(selected - found)
        if missing:
            raise ValueError(f"sample ids not found in {path}: {', '.join(missing)}")
    if not samples:
        raise ValueError(f"no samples selected from {path}")
    return samples


def _require_executable(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"missing llama.cpp binary: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"not a llama.cpp binary: {path}")


def _generation_binary(bin_dir: Path) -> Path:
    completion = bin_dir / "llama-completion"
    if completion.exists():
        return completion
    return bin_dir / "llama-cli"


def _sample_max_tokens(sample: dict[str, Any]) -> int:
    value = int(sample.get("max_tokens", 32))
    if value < 1:
        raise ValueError(f"sample {sample['id']} has invalid max_tokens: {value}")
    return value


def _generation_command(options: GGUFQualityEvalOptions, binary: Path, sample: dict[str, Any]) -> list[str]:
    command = [
        str(binary),
        "-m",
        str(options.gguf),
        "-ngl",
        str(options.gpu_layers),
        "-c",
        str(int(sample.get("ctx_size", options.ctx_size))),
        "-n",
        str(_sample_max_tokens(sample)),
        "-p",
        str(sample["prompt"]),
        "-fa",
        options.flash_attn,
        "-no-cnv",
        "--simple-io",
        "--no-warmup",
        "--no-display-prompt",
        "--log-verbosity",
        "1",
        "--temp",
        str(options.temperature),
        "--ignore-eos",
    ]
    if options.top_p is not None:
        command.extend(["--top-p", str(options.top_p)])
    if options.threads is not None:
        command.extend(["-t", str(options.threads)])
    return command


def _expected_strings(sample: dict[str, Any]) -> list[str]:
    expected = sample.get("expected", [])
    if isinstance(expected, str):
        return [expected]
    return [str(item) for item in expected]


def score_output(sample: dict[str, Any], output: str) -> dict[str, Any]:
    scorer = str(sample.get("scorer", "contains")).lower()
    expected = _expected_strings(sample)
    normalized_output = output.strip()
    lowered_output = normalized_output.lower()

    passed = False
    detail: dict[str, Any] = {"scorer": scorer, "expected": expected}
    if scorer == "contains":
        passed = any(item.lower() in lowered_output for item in expected)
    elif scorer == "contains_all":
        passed = all(item.lower() in lowered_output for item in expected)
    elif scorer == "exact":
        passed = bool(expected) and normalized_output == expected[0]
    elif scorer == "regex":
        if not expected:
            raise ValueError(f"sample {sample['id']} uses regex scorer without expected pattern")
        passed = re.search(expected[0], normalized_output, flags=re.IGNORECASE | re.DOTALL) is not None
    elif scorer == "json_valid":
        try:
            parsed = json.loads(normalized_output)
            passed = True
            detail["parsed_type"] = type(parsed).__name__
        except json.JSONDecodeError as exc:
            detail["json_error"] = str(exc)
    else:
        raise ValueError(f"unsupported scorer for sample {sample['id']}: {scorer}")

    detail["passed"] = passed
    return detail


def _run_sample(options: GGUFQualityEvalOptions, binary: Path, sample: dict[str, Any]) -> dict[str, Any]:
    command = _generation_command(options, binary, sample)
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=options.timeout_seconds,
        )
        duration = time.monotonic() - started
        output = completed.stdout.strip()
        scored = score_output(sample, output) if completed.returncode == 0 and output else {
            "scorer": str(sample.get("scorer", "contains")).lower(),
            "expected": _expected_strings(sample),
            "passed": False,
            "reason": "no_output" if completed.returncode == 0 else "runtime_error",
        }
        passed = bool(completed.returncode == 0 and output and scored["passed"])
        return {
            "id": sample["id"],
            "category": sample.get("category", "uncategorized"),
            "description": sample.get("description"),
            "returncode": completed.returncode,
            "timed_out": False,
            "duration_seconds": round(duration, 3),
            "max_tokens": _sample_max_tokens(sample),
            "ctx_size": int(sample.get("ctx_size", options.ctx_size)),
            "stdout": _tail(completed.stdout),
            "stderr_tail": _tail(completed.stderr),
            "score": scored,
            "passed": passed,
        }
    except subprocess.TimeoutExpired as exc:
        duration = time.monotonic() - started
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode(errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode(errors="replace")
        return {
            "id": sample["id"],
            "category": sample.get("category", "uncategorized"),
            "description": sample.get("description"),
            "returncode": None,
            "timed_out": True,
            "duration_seconds": round(duration, 3),
            "max_tokens": _sample_max_tokens(sample),
            "ctx_size": int(sample.get("ctx_size", options.ctx_size)),
            "stdout": _tail(stdout),
            "stderr_tail": _tail(stderr),
            "score": {
                "scorer": str(sample.get("scorer", "contains")).lower(),
                "expected": _expected_strings(sample),
                "passed": False,
                "reason": "timeout",
            },
            "passed": False,
        }


def _summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    durations = [float(result["duration_seconds"]) for result in results]
    passed = [result for result in results if result["passed"]]
    categories: dict[str, dict[str, Any]] = {}
    for result in results:
        category = str(result.get("category") or "uncategorized")
        row = categories.setdefault(category, {"total": 0, "passed": 0, "pass_rate": 0.0})
        row["total"] += 1
        row["passed"] += int(bool(result["passed"]))
    for row in categories.values():
        row["pass_rate"] = round(row["passed"] / row["total"], 4) if row["total"] else 0.0
    return {
        "total": len(results),
        "passed": len(passed),
        "failed": len(results) - len(passed),
        "pass_rate": round(len(passed) / len(results), 4) if results else 0.0,
        "duration_seconds_total": round(sum(durations), 3),
        "latency_seconds_mean": round(sum(durations) / len(durations), 3) if durations else None,
        "latency_seconds_p50": _percentile(durations, 50.0),
        "latency_seconds_p95": _percentile(durations, 95.0),
        "categories": categories,
    }


def _compare_to_reference(current: dict[str, Any], reference_path: Path) -> dict[str, Any]:
    reference = load_json(reference_path)
    if reference.get("schema") != "opentq.gguf_quality_eval.v1":
        raise ValueError(f"unsupported reference eval schema: {reference_path}")
    current_by_id = {sample["id"]: sample for sample in current["samples"]}
    reference_by_id = {sample["id"]: sample for sample in reference["samples"]}
    shared_ids = sorted(set(current_by_id) & set(reference_by_id))
    transitions = []
    for sample_id in shared_ids:
        current_passed = bool(current_by_id[sample_id]["passed"])
        reference_passed = bool(reference_by_id[sample_id]["passed"])
        if current_passed != reference_passed:
            transitions.append(
                {
                    "id": sample_id,
                    "reference_passed": reference_passed,
                    "current_passed": current_passed,
                }
            )
    current_summary = current["summary"]
    reference_summary = reference["summary"]
    return {
        "reference": str(reference_path),
        "shared_samples": len(shared_ids),
        "pass_rate_delta": round(float(current_summary["pass_rate"]) - float(reference_summary["pass_rate"]), 4),
        "latency_mean_ratio": (
            round(float(current_summary["latency_seconds_mean"]) / float(reference_summary["latency_seconds_mean"]), 4)
            if current_summary.get("latency_seconds_mean") and reference_summary.get("latency_seconds_mean")
            else None
        ),
        "status_transitions": transitions,
    }


def run_quality_eval(options: GGUFQualityEvalOptions) -> dict[str, Any]:
    if not options.gguf.exists():
        raise FileNotFoundError(f"missing GGUF artifact: {options.gguf}")
    if options.gguf.suffix != ".gguf":
        raise ValueError(f"expected a .gguf artifact: {options.gguf}")
    if options.flash_attn not in {"on", "off", "auto", "1", "0"}:
        raise ValueError("flash_attn must be one of: on, off, auto, 1, 0")

    samples = _load_suite(options.suite, options.max_samples, options.sample_ids)
    bin_dir = options.llama_cpp_dir / "build" / "bin"
    generator = _generation_binary(bin_dir)
    _require_executable(generator)

    results = [_run_sample(options, generator, sample) for sample in samples]
    payload = {
        "schema": "opentq.gguf_quality_eval.v1",
        "created_at": _now_iso(),
        "overall_pass": all(result["passed"] for result in results),
        "artifact": {
            "path": str(options.gguf),
            "filename": options.gguf.name,
            "bytes": options.gguf.stat().st_size,
        },
        "suite": {
            "path": str(options.suite),
            "samples": len(samples),
        },
        "runtime": {
            "llama_cpp_dir": str(options.llama_cpp_dir),
            "gpu_layers": options.gpu_layers,
            "flash_attn": options.flash_attn,
            "ctx_size": options.ctx_size,
            "threads": options.threads,
            "temperature": options.temperature,
            "top_p": options.top_p,
        },
        "summary": _summarize(results),
        "samples": results,
    }
    if options.reference is not None:
        payload["comparison"] = _compare_to_reference(payload, options.reference)
    dump_json(options.output, payload)
    return payload
