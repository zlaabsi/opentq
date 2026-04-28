from __future__ import annotations

import json
from pathlib import Path

from scripts.run_qwen36_runtime_checks import RuntimeConfig, build_bench_command, build_cli_command, write_runtime_result


def test_build_cli_command_uses_metal_fa_and_prompt(tmp_path: Path) -> None:
    model = tmp_path / "model.gguf"
    config = RuntimeConfig(llama_cpp=tmp_path / "llama.cpp", threads=8, ctx_size=8192, predict=32)

    command = build_cli_command(model, "Paris?", config)

    assert command == [
        str(tmp_path / "llama.cpp" / "build" / "bin" / "llama-cli"),
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
    ]


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
