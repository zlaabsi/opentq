from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .hf_release import dump_json


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _tail(text: str, max_chars: int = 12_000) -> str:
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


@dataclass(frozen=True)
class GGUFValidationOptions:
    gguf: Path
    output: Path
    llama_cpp_dir: Path = Path("../llama.cpp")
    prompt: str = "Write one short sentence about quantization."
    ctx_size: int = 256
    n_predict: int = 4
    gpu_layers: str = "0"
    threads: int | None = None
    timeout_seconds: float = 600.0
    flash_attn: str = "off"
    run_bench: bool = False
    bench_prompt_tokens: int = 512
    bench_gen_tokens: int = 16


def _run_phase(label: str, command: list[str], timeout_seconds: float) -> dict[str, Any]:
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        duration = time.monotonic() - started
        payload = {
            "label": label,
            "command": command,
            "returncode": completed.returncode,
            "timed_out": False,
            "duration_seconds": round(duration, 3),
            "stdout_tail": _tail(completed.stdout),
            "stderr_tail": _tail(completed.stderr),
            "passed": completed.returncode == 0,
        }
        payload["failure_reason"] = _failure_reason(payload)
        return payload
    except subprocess.TimeoutExpired as exc:
        duration = time.monotonic() - started
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode(errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode(errors="replace")
        payload = {
            "label": label,
            "command": command,
            "returncode": None,
            "timed_out": True,
            "duration_seconds": round(duration, 3),
            "stdout_tail": _tail(stdout),
            "stderr_tail": _tail(stderr),
            "passed": False,
        }
        payload["failure_reason"] = _failure_reason(payload)
        return payload


def _failure_reason(phase: dict[str, Any]) -> str | None:
    if phase.get("passed"):
        return None
    if phase.get("timed_out"):
        return "timeout"
    text = f"{phase.get('stdout_tail', '')}\n{phase.get('stderr_tail', '')}"
    if "not implemented" in text:
        return "backend_not_implemented"
    if "not supported" in text:
        return "unsupported_runtime_mode"
    if phase.get("returncode") is not None:
        return "runtime_error"
    return "unknown"


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


def _generation_command(options: GGUFValidationOptions, binary: Path) -> list[str]:
    command = [
        str(binary),
        "-m",
        str(options.gguf),
        "-ngl",
        str(options.gpu_layers),
        "-c",
        str(options.ctx_size),
        "-n",
        str(options.n_predict),
        "-p",
        options.prompt,
        "-fa",
        options.flash_attn,
        "-no-cnv",
        "--simple-io",
        "--no-warmup",
        "--no-display-prompt",
        "--log-verbosity",
        "1",
        "--temp",
        "0",
        "--ignore-eos",
    ]
    if options.threads is not None:
        command.extend(["-t", str(options.threads)])
    return command


def _bench_flash_attn_value(value: str) -> str:
    if value in {"on", "auto", "1"}:
        return "1"
    return "0"


def validate_gguf(options: GGUFValidationOptions) -> dict[str, Any]:
    gguf = options.gguf
    if not gguf.exists():
        raise FileNotFoundError(f"missing GGUF artifact: {gguf}")
    if gguf.suffix != ".gguf":
        raise ValueError(f"expected a .gguf artifact: {gguf}")
    if options.flash_attn not in {"on", "off", "auto", "1", "0"}:
        raise ValueError("flash_attn must be one of: on, off, auto, 1, 0")

    bin_dir = options.llama_cpp_dir / "build" / "bin"
    llama_gguf = bin_dir / "llama-gguf"
    generator = _generation_binary(bin_dir)
    llama_bench = bin_dir / "llama-bench"
    _require_executable(llama_gguf)
    _require_executable(generator)
    if options.run_bench:
        _require_executable(llama_bench)

    phases = [
        _run_phase(
            "gguf_metadata_read",
            [str(llama_gguf), str(gguf), "r", "n"],
            min(options.timeout_seconds, 180.0),
        )
    ]

    cli_phase = _run_phase("bounded_generation", _generation_command(options, generator), options.timeout_seconds)
    generated = cli_phase["stdout_tail"].strip()
    cli_phase["generated_chars"] = len(generated)
    cli_phase["passed"] = bool(cli_phase["passed"] and generated)
    if not cli_phase["passed"] and cli_phase["failure_reason"] is None:
        cli_phase["failure_reason"] = "no_generation"
    phases.append(cli_phase)

    if options.run_bench and cli_phase["passed"]:
        phases.append(
            _run_phase(
                "llama_bench",
                [
                    str(llama_bench),
                    "-m",
                    str(gguf),
                    "-ngl",
                    str(options.gpu_layers),
                    "-fa",
                    _bench_flash_attn_value(options.flash_attn),
                    "-p",
                    str(options.bench_prompt_tokens),
                    "-n",
                    str(options.bench_gen_tokens),
                    "-r",
                    "1",
                    "--no-warmup",
                ],
                options.timeout_seconds,
            )
        )

    passed = all(phase["passed"] for phase in phases)
    payload = {
        "schema": "opentq.gguf_validation.v1",
        "created_at": _now_iso(),
        "overall_pass": passed,
        "artifact": {
            "path": str(gguf),
            "filename": gguf.name,
            "bytes": gguf.stat().st_size,
        },
        "runtime": {
            "llama_cpp_dir": str(options.llama_cpp_dir),
            "gpu_layers": options.gpu_layers,
            "flash_attn": options.flash_attn,
            "ctx_size": options.ctx_size,
            "n_predict": options.n_predict,
            "threads": options.threads,
            "bench_prompt_tokens": options.bench_prompt_tokens if options.run_bench else None,
            "bench_gen_tokens": options.bench_gen_tokens if options.run_bench else None,
        },
        "gates": {
            "gguf_metadata_read": phases[0]["passed"],
            "bounded_generation": cli_phase["passed"],
            "benchmark": None if not options.run_bench else phases[-1]["passed"],
        },
        "phases": phases,
    }
    dump_json(options.output, payload)
    return payload


def assert_validation_matches(
    validation: dict[str, Any],
    gguf: Path,
    *,
    require_benchmark: bool = True,
    min_benchmark_prompt_tokens: int = 8192,
    min_benchmark_gen_tokens: int = 128,
) -> None:
    if validation.get("schema") != "opentq.gguf_validation.v1":
        raise ValueError("validation payload has an unsupported schema")
    if not validation.get("overall_pass"):
        raise ValueError("GGUF validation did not pass")
    artifact = validation.get("artifact") or {}
    if artifact.get("filename") != gguf.name:
        raise ValueError(f"validation artifact mismatch: {artifact.get('filename')} != {gguf.name}")
    if int(artifact.get("bytes", -1)) != gguf.stat().st_size:
        raise ValueError("validation artifact size does not match current GGUF")
    if require_benchmark and (validation.get("gates") or {}).get("benchmark") is not True:
        raise ValueError("GGUF validation is missing a passing benchmark gate")
    runtime = validation.get("runtime") or {}
    if require_benchmark and int(runtime.get("bench_prompt_tokens") or 0) < min_benchmark_prompt_tokens:
        raise ValueError("GGUF validation benchmark prompt length is below the release gate")
    if require_benchmark and int(runtime.get("bench_gen_tokens") or 0) < min_benchmark_gen_tokens:
        raise ValueError("GGUF validation benchmark generation length is below the release gate")
