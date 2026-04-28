from __future__ import annotations

import argparse
import json
import platform
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RuntimeConfig:
    llama_cpp: Path
    threads: int = 8
    ctx_size: int = 8192
    predict: int = 64
    timeout_seconds: int = 600


def build_cli_command(model: Path, prompt: str, config: RuntimeConfig) -> list[str]:
    return [
        str(config.llama_cpp / "build" / "bin" / "llama-cli"),
        "-m",
        str(model),
        "-p",
        prompt,
        "-n",
        str(config.predict),
        "-c",
        str(config.ctx_size),
        "-t",
        str(config.threads),
        "-ngl",
        "99",
        "-fa",
        "on",
        "-no-cnv",
        "--simple-io",
    ]


def build_bench_command(model: Path, config: RuntimeConfig) -> list[str]:
    return [
        str(config.llama_cpp / "build" / "bin" / "llama-bench"),
        "-m",
        str(model),
        "-p",
        str(config.ctx_size),
        "-n",
        "128",
        "-t",
        str(config.threads),
        "-ngl",
        "99",
        "-fa",
        "1",
    ]


def write_runtime_result(output: Path, payload: dict[str, Any]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_command(command: list[str], timeout_seconds: int) -> dict[str, Any]:
    try:
        completed = subprocess.run(command, text=True, capture_output=True, check=False, timeout=timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        return {
            "command": command,
            "returncode": -1,
            "timed_out": True,
            "timeout_seconds": timeout_seconds,
            "stdout_tail": stdout[-4000:],
            "stderr_tail": stderr[-4000:],
        }
    return {
        "command": command,
        "returncode": completed.returncode,
        "timed_out": False,
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local Qwen3.6 GGUF runtime checks.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--llama-cpp", type=Path, default=Path("/Users/zlaabsi/Documents/GitHub/llama.cpp"))
    parser.add_argument("--machine", default="M1 Max 32GB")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt", default="Réponds uniquement par la capitale de la France.")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--ctx-size", type=int, default=8192)
    parser.add_argument("--predict", type=int, default=64)
    parser.add_argument("--timeout-seconds", type=int, default=600)
    args = parser.parse_args()

    config = RuntimeConfig(
        llama_cpp=args.llama_cpp,
        threads=args.threads,
        ctx_size=args.ctx_size,
        predict=args.predict,
        timeout_seconds=args.timeout_seconds,
    )
    cli_result = run_command(build_cli_command(args.model, args.prompt, config), config.timeout_seconds)
    bench_result = run_command(build_bench_command(args.model, config), config.timeout_seconds)
    payload = {
        "model": str(args.model),
        "machine": args.machine,
        "platform": platform.platform(),
        "bounded_generation_passed": cli_result["returncode"] == 0,
        "bench_passed": bench_result["returncode"] == 0,
        "cli": cli_result,
        "bench": bench_result,
    }
    write_runtime_result(args.output, payload)
    raise SystemExit(0 if payload["bounded_generation_passed"] and payload["bench_passed"] else 1)


if __name__ == "__main__":
    main()
