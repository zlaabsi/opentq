from __future__ import annotations

import argparse
import json
import platform
import re
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


def generation_binary(llama_cpp: Path) -> Path:
    completion = llama_cpp / "build" / "bin" / "llama-completion"
    if completion.exists():
        return completion
    return llama_cpp / "build" / "bin" / "llama-cli"


def format_prompt(prompt: str, prompt_format: str) -> str:
    if prompt_format == "raw":
        return prompt
    if prompt_format == "qwen3-no-think":
        return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    raise ValueError(f"unsupported prompt format: {prompt_format}")


def build_cli_command(model: Path, prompt: str, config: RuntimeConfig) -> list[str]:
    binary = generation_binary(config.llama_cpp)
    command = [
        str(binary),
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
        "--no-warmup",
    ]
    if binary.name == "llama-completion":
        command.extend(
            [
                "-no-cnv",
                "--simple-io",
                "--no-display-prompt",
                "--log-verbosity",
                "1",
                "--temp",
                "0",
            ]
        )
    return command


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


def tail_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")[-4000:]
    return value[-4000:]


def run_command(command: list[str], timeout_seconds: int) -> dict[str, Any]:
    try:
        completed = subprocess.run(command, text=True, capture_output=True, check=False, timeout=timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        return {
            "command": command,
            "returncode": -1,
            "timed_out": True,
            "timeout_seconds": timeout_seconds,
            "stdout_tail": tail_text(exc.stdout),
            "stderr_tail": tail_text(exc.stderr),
        }
    return {
        "command": command,
        "returncode": completed.returncode,
        "timed_out": False,
        "stdout_tail": tail_text(completed.stdout),
        "stderr_tail": tail_text(completed.stderr),
    }


def generation_passed(result: dict[str, Any], expected_output_regex: str | None) -> bool:
    if result["returncode"] != 0 or result["timed_out"]:
        return False
    stdout = str(result.get("stdout_tail", ""))
    stderr = str(result.get("stderr_tail", ""))
    combined = f"{stdout}\n{stderr}".lower()
    unsupported_markers = (
        "not supported",
        "please use llama-completion",
        "available commands:",
    )
    if any(marker in combined for marker in unsupported_markers):
        return False
    if not stdout.strip():
        return False
    if expected_output_regex and re.search(expected_output_regex, stdout) is None:
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local Qwen3.6 GGUF runtime checks.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--llama-cpp", type=Path, default=Path("/Users/zlaabsi/Documents/GitHub/llama.cpp"))
    parser.add_argument("--machine", default="M1 Max 32GB")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt", default="Réponds uniquement par la capitale de la France.")
    parser.add_argument("--prompt-format", choices=["raw", "qwen3-no-think"], default="qwen3-no-think")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--ctx-size", type=int, default=8192)
    parser.add_argument("--predict", type=int, default=64)
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--expected-output-regex", default=r"\bParis\b")
    args = parser.parse_args()

    config = RuntimeConfig(
        llama_cpp=args.llama_cpp,
        threads=args.threads,
        ctx_size=args.ctx_size,
        predict=args.predict,
        timeout_seconds=args.timeout_seconds,
    )
    cli_result = run_command(
        build_cli_command(args.model, format_prompt(args.prompt, args.prompt_format), config),
        config.timeout_seconds,
    )
    bench_result = run_command(build_bench_command(args.model, config), config.timeout_seconds)
    bounded_generation_passed = generation_passed(cli_result, args.expected_output_regex)
    payload = {
        "model": str(args.model),
        "machine": args.machine,
        "platform": platform.platform(),
        "prompt_format": args.prompt_format,
        "expected_output_regex": args.expected_output_regex,
        "bounded_generation_passed": bounded_generation_passed,
        "bench_passed": bench_result["returncode"] == 0,
        "cli": cli_result,
        "bench": bench_result,
    }
    write_runtime_result(args.output, payload)
    raise SystemExit(0 if payload["bounded_generation_passed"] and payload["bench_passed"] else 1)


if __name__ == "__main__":
    main()
