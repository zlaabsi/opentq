from __future__ import annotations

import json
from pathlib import Path

from opentq.gguf_validate import GGUFValidationOptions, _bench_flash_attn_value, validate_gguf


def _write_executable(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    path.chmod(path.stat().st_mode | 0o111)


def test_validate_gguf_writes_passing_gate(tmp_path: Path) -> None:
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"GGUF-smoke")
    bin_dir = tmp_path / "llama.cpp" / "build" / "bin"
    _write_executable(bin_dir / "llama-gguf", "#!/usr/bin/env sh\nexit 0\n")
    _write_executable(bin_dir / "llama-cli", "#!/usr/bin/env sh\nprintf 'validated output\\n'\n")

    output = tmp_path / "validation.json"
    payload = validate_gguf(
        GGUFValidationOptions(
            gguf=gguf,
            output=output,
            llama_cpp_dir=tmp_path / "llama.cpp",
            timeout_seconds=5.0,
        )
    )

    assert payload["overall_pass"] is True
    assert payload["gates"]["gguf_metadata_read"] is True
    assert payload["gates"]["bounded_generation"] is True
    assert json.loads(output.read_text(encoding="utf-8"))["artifact"]["filename"] == gguf.name


def test_validate_gguf_fails_without_generation(tmp_path: Path) -> None:
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"GGUF-smoke")
    bin_dir = tmp_path / "llama.cpp" / "build" / "bin"
    _write_executable(bin_dir / "llama-gguf", "#!/usr/bin/env sh\nexit 0\n")
    _write_executable(bin_dir / "llama-cli", "#!/usr/bin/env sh\nexit 0\n")

    output = tmp_path / "validation.json"
    payload = validate_gguf(
        GGUFValidationOptions(
            gguf=gguf,
            output=output,
            llama_cpp_dir=tmp_path / "llama.cpp",
            timeout_seconds=5.0,
        )
    )

    assert payload["overall_pass"] is False
    assert payload["gates"]["bounded_generation"] is False


def test_bench_flash_attention_uses_llama_bench_boolean_values() -> None:
    assert _bench_flash_attn_value("on") == "1"
    assert _bench_flash_attn_value("auto") == "1"
    assert _bench_flash_attn_value("1") == "1"
    assert _bench_flash_attn_value("off") == "0"
    assert _bench_flash_attn_value("0") == "0"
