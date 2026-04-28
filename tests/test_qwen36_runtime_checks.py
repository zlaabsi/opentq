from __future__ import annotations

import json
import sys
from pathlib import Path

from scripts.run_qwen36_runtime_checks import (
    RuntimeConfig,
    build_bench_command,
    build_cli_command,
    format_prompt,
    generation_passed,
    run_command,
    write_runtime_result,
)


def test_build_cli_command_prefers_completion_and_uses_metal_fa(tmp_path: Path) -> None:
    model = tmp_path / "model.gguf"
    completion = tmp_path / "llama.cpp" / "build" / "bin" / "llama-completion"
    completion.parent.mkdir(parents=True)
    completion.write_text("#!/usr/bin/env sh\n", encoding="utf-8")
    config = RuntimeConfig(llama_cpp=tmp_path / "llama.cpp", threads=8, ctx_size=8192, predict=32)

    command = build_cli_command(model, "Paris?", config)

    assert command == [
        str(completion),
        "-m",
        str(model),
        "-p",
        "Paris?",
        "-n",
        "32",
        "-c",
        "8192",
        "-t",
        "8",
        "-ngl",
        "99",
        "-fa",
        "on",
        "--no-warmup",
        "-no-cnv",
        "--simple-io",
        "--no-display-prompt",
        "--log-verbosity",
        "1",
        "--temp",
        "0",
    ]


def test_build_cli_command_falls_back_to_non_chat_cli(tmp_path: Path) -> None:
    model = tmp_path / "model.gguf"
    config = RuntimeConfig(llama_cpp=tmp_path / "llama.cpp", threads=4, ctx_size=1024, predict=8)

    command = build_cli_command(model, "Paris?", config)

    assert command[0] == str(tmp_path / "llama.cpp" / "build" / "bin" / "llama-cli")
    assert "-no-cnv" not in command
    assert "--single-turn" not in command


def test_format_prompt_supports_qwen_no_think() -> None:
    assert format_prompt("Paris?", "raw") == "Paris?"
    assert format_prompt("Paris?", "qwen3-no-think") == (
        "<|im_start|>user\nParis?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


def test_build_bench_command_sets_prefill_and_decode(tmp_path: Path) -> None:
    model = tmp_path / "model.gguf"
    config = RuntimeConfig(llama_cpp=tmp_path / "llama.cpp", threads=8, ctx_size=8192, predict=32)

    command = build_bench_command(model, config)

    assert command[:4] == [str(tmp_path / "llama.cpp" / "build" / "bin" / "llama-bench"), "-m", str(model), "-p"]
    assert "8192" in command
    assert "128" in command
    assert "-fa" in command
    assert "1" in command


def test_write_runtime_result_records_evidence(tmp_path: Path) -> None:
    output = tmp_path / "runtime.json"
    write_runtime_result(
        output,
        {
            "model": "Qwen3.6-27B-OTQ-DYN-Q3_K_M.gguf",
            "machine": "M1 Max 32GB",
            "bounded_generation_passed": True,
        },
    )

    assert json.loads(output.read_text(encoding="utf-8")) == {
        "model": "Qwen3.6-27B-OTQ-DYN-Q3_K_M.gguf",
        "machine": "M1 Max 32GB",
        "bounded_generation_passed": True,
    }


def test_run_command_records_timeout() -> None:
    result = run_command([sys.executable, "-c", "import time; time.sleep(2)"], timeout_seconds=1)

    assert result["returncode"] == -1
    assert result["timed_out"] is True
    assert result["timeout_seconds"] == 1


def test_generation_passed_requires_expected_text() -> None:
    result = {"returncode": 0, "timed_out": False, "stdout_tail": "Paris\n", "stderr_tail": ""}

    assert generation_passed(result, r"\bParis\b") is True
    assert generation_passed(result, r"\bLyon\b") is False


def test_generation_passed_rejects_unsupported_cli_mode() -> None:
    result = {
        "returncode": 0,
        "timed_out": False,
        "stdout_tail": "available commands:\n> prompt",
        "stderr_tail": "please use llama-completion instead",
    }

    assert generation_passed(result, r"\bParis\b") is False
