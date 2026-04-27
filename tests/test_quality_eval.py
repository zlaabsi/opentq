from __future__ import annotations

import json
from pathlib import Path

from opentq.quality_eval import GGUFQualityEvalOptions, clean_output_for_scoring, format_prompt, run_quality_eval, score_output


def _write_executable(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    path.chmod(path.stat().st_mode | 0o111)


def test_score_output_contains_all() -> None:
    sample = {
        "id": "tool",
        "scorer": "contains_all",
        "expected": ['"action"', '"query"'],
    }

    score = score_output(sample, '{"action":"search","query":"metal"}')

    assert score["passed"] is True


def test_score_output_json_valid() -> None:
    sample = {"id": "json", "scorer": "json_valid"}

    score = score_output(sample, '{"ok": true}')

    assert score["passed"] is True
    assert score["parsed_type"] == "dict"


def test_score_output_json_contains() -> None:
    sample = {
        "id": "tool",
        "scorer": "json_contains",
        "expected": {"action": "search", "query": "metal"},
    }

    score = score_output(sample, '{"action":"search","query":"metal"}')

    assert score["passed"] is True
    assert score["missing_or_mismatched"] == {}


def test_score_output_json_contains_strips_llama_cpp_eos_marker() -> None:
    sample = {
        "id": "tool",
        "scorer": "json_contains",
        "expected": {"action": "search", "query": "metal"},
    }

    score = score_output(sample, '{"action":"search","query":"metal"} [end of text]\n')

    assert score["passed"] is True


def test_score_output_json_contains_rejects_trailing_text() -> None:
    sample = {
        "id": "tool",
        "scorer": "json_contains",
        "expected": {"action": "search"},
    }

    score = score_output(sample, '{"action":"search"}\\n```json')

    assert score["passed"] is False
    assert "json_error" in score


def test_clean_output_for_scoring_only_strips_terminal_marker() -> None:
    assert clean_output_for_scoring("Paris [end of text]\n") == "Paris"
    assert clean_output_for_scoring("Paris [end of text]\nextra") == "Paris [end of text]\nextra"


def test_format_prompt_qwen3_no_think() -> None:
    prompt = format_prompt("Return 26.", "qwen3-no-think")

    assert prompt.startswith("<|im_start|>user\nReturn 26.<|im_end|>")
    assert prompt.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")


def test_run_quality_eval_writes_payload(tmp_path: Path) -> None:
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"GGUF-quality")
    suite = tmp_path / "suite.jsonl"
    suite.write_text(
        json.dumps(
            {
                "id": "knowledge_capital_fr",
                "category": "knowledge",
                "prompt": "The capital of France is",
                "expected": ["Paris"],
                "scorer": "contains",
                "max_tokens": 8,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    bin_dir = tmp_path / "llama.cpp" / "build" / "bin"
    _write_executable(bin_dir / "llama-completion", "#!/usr/bin/env sh\nprintf 'Paris\\n'\n")

    output = tmp_path / "eval.json"
    payload = run_quality_eval(
        GGUFQualityEvalOptions(
            gguf=gguf,
            output=output,
            suite=suite,
            llama_cpp_dir=tmp_path / "llama.cpp",
            timeout_seconds=5.0,
        )
    )

    written = json.loads(output.read_text(encoding="utf-8"))
    assert payload["overall_pass"] is True
    assert payload["summary"]["pass_rate"] == 1.0
    assert written["samples"][0]["id"] == "knowledge_capital_fr"


def test_run_quality_eval_compares_reference(tmp_path: Path) -> None:
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"GGUF-quality")
    suite = tmp_path / "suite.jsonl"
    suite.write_text(
        json.dumps(
            {
                "id": "sample",
                "category": "knowledge",
                "prompt": "prompt",
                "expected": ["Paris"],
                "scorer": "contains",
                "max_tokens": 8,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    reference = tmp_path / "reference.json"
    reference.write_text(
        json.dumps(
            {
                "schema": "opentq.gguf_quality_eval.v1",
                "summary": {"pass_rate": 0.0, "latency_seconds_mean": 2.0},
                "samples": [{"id": "sample", "passed": False}],
            }
        ),
        encoding="utf-8",
    )
    bin_dir = tmp_path / "llama.cpp" / "build" / "bin"
    _write_executable(bin_dir / "llama-completion", "#!/usr/bin/env sh\nprintf 'Paris\\n'\n")

    payload = run_quality_eval(
        GGUFQualityEvalOptions(
            gguf=gguf,
            output=tmp_path / "eval.json",
            suite=suite,
            llama_cpp_dir=tmp_path / "llama.cpp",
            timeout_seconds=5.0,
            reference=reference,
        )
    )

    assert payload["comparison"]["pass_rate_delta"] == 1.0
    assert payload["comparison"]["status_transitions"] == [
        {"id": "sample", "reference_passed": False, "current_passed": True}
    ]
