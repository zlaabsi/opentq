from __future__ import annotations

import base64
import json
import pickle
import subprocess
import zlib
from pathlib import Path

import requests
import scripts.run_qwen36_benchmark_subsets as runner
from scripts.run_qwen36_benchmark_subsets import (
    ADAPTERS,
    MODEL_PATHS,
    build_sample_from_row,
    apply_generation_overrides,
    apply_max_tokens,
    fetch_dataset_viewer_row,
    fetch_hf_raw_jsonl_row,
    final_answer_text,
    format_qwen_prompt,
    generation_binary,
    generation_command,
    enforce_local_memory_gate,
    llama_server_command,
    ModelTarget,
    parse_json_list,
    parse_models,
    run_livecodebench_stdin_tests,
    run_server_generation,
    score_benchmark_output,
    server_binary,
    _spread_offsets,
)


MATRIX = Path("benchmarks/qwen36_benchmark_matrix.json")


REQUESTED_IDS = {
    "mmlu",
    "mmlu_pro",
    "arc",
    "hellaswag",
    "gsm8k",
    "math",
    "aime",
    "humaneval",
    "mbpp",
    "swe_bench",
    "livecodebench",
    "bbh",
    "gpqa",
    "mt_bench",
    "chatbot_arena",
    "alpacaeval",
    "ifeval",
    "mmmu",
    "mathvista",
    "truthfulqa",
    "winogrande",
    "drop",
    "piqa",
    "commonsenseqa",
}

PHASE4_ADAPTER_IDS = {
    "mmlu",
    "mmlu_pro",
    "arc",
    "hellaswag",
    "gsm8k",
    "math",
    "aime",
    "humaneval",
    "mbpp",
    "swe_bench",
    "livecodebench",
    "bbh",
    "gpqa",
    "ifeval",
    "truthfulqa",
    "winogrande",
    "drop",
    "piqa",
    "commonsenseqa",
}


def _matrix_rows() -> list[dict[str, object]]:
    payload = json.loads(MATRIX.read_text(encoding="utf-8"))
    return list(payload["benchmarks"])


def test_matrix_contains_all_requested_benchmark_families() -> None:
    ids = {str(row["id"]) for row in _matrix_rows()}

    assert ids == REQUESTED_IDS


def test_matrix_rows_have_claim_controls() -> None:
    for row in _matrix_rows():
        assert row["requested_by_user"] is True
        assert row["comparison_mode"] in {
            "official_comparable",
            "mini_bf16_required",
            "judge_based",
            "blocked_modality",
        }
        assert row["subset_policy"]
        assert row["claim_rule"]


def test_multimodal_rows_are_blocked_for_text_only_gguf() -> None:
    rows = {str(row["id"]): row for row in _matrix_rows()}

    assert rows["mmmu"]["comparison_mode"] == "blocked_modality"
    assert rows["mathvista"]["comparison_mode"] == "blocked_modality"


def test_official_comparable_rows_carry_official_baselines() -> None:
    rows = {str(row["id"]): row for row in _matrix_rows()}

    for benchmark_id in ("mmlu_pro", "gpqa", "aime", "swe_bench", "livecodebench"):
        assert rows[benchmark_id]["comparison_mode"] == "official_comparable"
        assert rows[benchmark_id]["official_baseline"]


def test_benchmark_subset_runner_dry_run_reports_skips(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            "python",
            "scripts/run_qwen36_benchmark_subsets.py",
            "--matrix",
            str(MATRIX),
            "--models",
            "q3,q4",
            "--output-root",
            str(tmp_path),
            "--llama-cpp",
            str(tmp_path / "llama.cpp"),
            "--sample-mode",
            "quick",
            "--dry-run",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "mmlu_pro" in completed.stdout
    assert "planned mmlu_pro mode=official_comparable samples=24" in completed.stdout
    assert "requires_external_harness swe_bench mode=official_comparable samples=0" in completed.stdout
    assert "planned livecodebench mode=official_comparable samples=12" in completed.stdout
    assert "planned ifeval mode=mini_bf16_required samples=24" in completed.stdout
    assert "skipped_judge" in completed.stdout
    assert "blocked_modality" in completed.stdout


def test_benchmark_subset_runner_refuses_non_dry_run_without_adapters(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            "python",
            "scripts/run_qwen36_benchmark_subsets.py",
            "--matrix",
            str(MATRIX),
            "--models",
            "q3",
            "--output-root",
            str(tmp_path),
            "--llama-cpp",
            str(tmp_path / "missing-llama.cpp"),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "missing llama.cpp binary" in completed.stderr
    assert not (tmp_path / "q3.json").exists()


def test_phase4_adapters_are_defined_with_pinned_metadata() -> None:
    assert set(ADAPTERS) == PHASE4_ADAPTER_IDS

    for benchmark_id, adapter in ADAPTERS.items():
        assert adapter.benchmark_id == benchmark_id
        assert adapter.dataset
        assert adapter.config
        assert adapter.split
        assert len(adapter.revision) == 40
        assert adapter.revision != "main"
        assert adapter.task_ids
        assert adapter.prompt_format == "qwen3-no-think"
        assert adapter.scoring_rule


def test_task_ids_are_spread_across_pinned_splits_not_first_rows_only() -> None:
    assert _spread_offsets(total=100, count=5) == ("offset:0", "offset:25", "offset:50", "offset:74", "offset:99")
    assert ADAPTERS["mmlu"].task_ids[:4] == ("offset:0", "offset:936", "offset:1872", "offset:2808")
    assert ADAPTERS["livecodebench"].task_ids[:4] == ("offset:0", "offset:16", "offset:32", "offset:47")


def test_benchmark_runner_supports_q5_model_target() -> None:
    targets = parse_models("q3,q4,q5")

    assert [target.key for target in targets] == ["q3", "q4", "q5"]
    assert MODEL_PATHS["q5"].name == "Qwen3.6-27B-OTQ-DYN-Q5_K_M.gguf"


def test_benchmark_runner_supports_local_bf16_gguf_reference() -> None:
    targets = parse_models("bf16_gguf,q3")

    assert [target.key for target in targets] == ["bf16_gguf", "q3"]
    assert targets[0].as_payload()["kind"] == "gguf_reference"
    assert MODEL_PATHS["bf16_gguf"].name == "Qwen3.6-27B-BF16.gguf"


def test_local_memory_gate_refuses_bf16_file_larger_than_ram_threshold(tmp_path: Path, monkeypatch) -> None:
    model_path = tmp_path / "bf16.gguf"
    with model_path.open("wb") as handle:
        handle.truncate(50 * 1024**3)
    monkeypatch.setattr(runner, "physical_memory_bytes", lambda: 32 * 1024**3)

    try:
        enforce_local_memory_gate(ModelTarget("bf16_gguf", model_path), allow_oversized_local_model=False)
    except MemoryError as exc:
        assert "refusing to load bf16_gguf locally" in str(exc)
    else:
        raise AssertionError("expected oversized local model to be refused")

    enforce_local_memory_gate(ModelTarget("bf16_gguf", model_path), allow_oversized_local_model=True)


def test_build_sample_from_mmlu_row_pins_task_metadata() -> None:
    sample = build_sample_from_row(
        ADAPTERS["mmlu"],
        "offset:0",
        {
            "question": "Find the degree for the field extension.",
            "choices": ["0", "4", "2", "6"],
            "answer": 1,
            "subject": "abstract_algebra",
        },
    )

    assert sample["benchmark_id"] == "mmlu"
    assert sample["task_id"] == "offset:0"
    assert sample["dataset"] == "cais/mmlu"
    assert sample["split"] == "test"
    assert sample["answer"] == "B"
    assert "(B) 4" in sample["prompt"]


def test_build_sample_from_swe_bench_row_records_harness_requirement() -> None:
    sample = build_sample_from_row(
        ADAPTERS["swe_bench"],
        "offset:0",
        {
            "repo": "astropy/astropy",
            "instance_id": "astropy__astropy-12907",
            "base_commit": "d16bfe05a744909de4b27f5875fe0d4ed41ce607",
            "problem_statement": "Nested compound models are not separable.",
            "hints_text": "",
            "FAIL_TO_PASS": "[\"test_a\"]",
            "PASS_TO_PASS": "[\"test_b\"]",
        },
    )

    assert sample["benchmark_id"] == "swe_bench"
    assert sample["dataset"] == "princeton-nlp/SWE-bench_Verified"
    assert sample["harness_required"] is True
    assert sample["scoring_rule"] == "swe_bench_verified_harness"
    assert "Return only a unified diff patch" in sample["prompt"]


def test_build_sample_from_livecodebench_row_records_v6_stdin_tests() -> None:
    sample = build_sample_from_row(
        ADAPTERS["livecodebench"],
        "offset:0",
        {
            "question_title": "A+B",
            "question_content": "Read two integers and print their sum.",
            "platform": "atcoder",
            "question_id": "abc000_a",
            "contest_id": "abc000",
            "contest_date": "2025-01-01T00:00:00",
            "starter_code": "",
            "difficulty": "easy",
            "public_test_cases": '[{"input": "1 2\\n", "output": "3\\n", "testtype": "stdin"}]',
            "private_test_cases": '[{"input": "4 5\\n", "output": "9\\n", "testtype": "stdin"}]',
            "metadata": "{}",
        },
    )

    assert sample["benchmark_id"] == "livecodebench"
    assert sample["dataset"] == "livecodebench/code_generation_lite"
    assert sample["config"] == "v6"
    assert sample["source_question_id"] == "abc000_a"
    assert len(sample["public_test_cases"]) == 1
    assert len(sample["private_test_cases"]) == 1
    assert sample["scoring_rule"] == "livecodebench_v6_stdin_exact"


def test_livecodebench_private_tests_can_use_official_compressed_encoding() -> None:
    payload = json.dumps([{"input": "1\n", "output": "1\n", "testtype": "stdin"}])
    compressed = base64.b64encode(zlib.compress(pickle.dumps(payload))).decode("utf-8")

    parsed = parse_json_list(compressed)

    assert parsed == [{"input": "1\n", "output": "1\n", "testtype": "stdin"}]


def test_score_benchmark_output_handles_multiple_choice_and_numeric_answers() -> None:
    mc = {
        "scoring_rule": "multiple_choice_letter",
        "answer": "D",
    }
    numeric = {
        "scoring_rule": "numeric_exact",
        "answer": "18",
    }

    assert score_benchmark_output(mc, "The answer is D.")["passed"] is True
    assert score_benchmark_output(mc, "The answer is A.")["passed"] is False
    assert score_benchmark_output(mc, "<think>\nMaybe A.\n</think>\nThe answer is D.")["passed"] is True
    assert score_benchmark_output(numeric, "Final answer: 18")["passed"] is True


def test_final_answer_text_ignores_qwen_thinking_trace() -> None:
    assert final_answer_text("<think>\nA tempting wrong answer.\n</think>\nFinal answer: B") == "Final answer: B"
    assert "<think>" in format_qwen_prompt("Hello", "qwen3-thinking")


def test_swe_bench_scoring_refuses_synthetic_pass_fail() -> None:
    score = score_benchmark_output({"scoring_rule": "swe_bench_verified_harness", "answer": None}, "diff --git a/x b/x")

    assert score["passed"] is None
    assert score["reason"] == "requires_external_swe_bench_verified_harness"


def test_livecodebench_stdin_scorer_executes_generated_python() -> None:
    sample = {
        "public_test_cases": [{"input": "1 2\n", "output": "3\n", "testtype": "stdin"}],
        "private_test_cases": [{"input": "4 5\n", "output": "9\n", "testtype": "stdin"}],
    }
    output = "```python\nimport sys\nnums=list(map(int, sys.stdin.read().split()))\nprint(sum(nums))\n```"

    score = run_livecodebench_stdin_tests(output, sample)

    assert score["passed"] is True
    assert score["total_cases"] == 2


def test_generation_command_prefers_llama_completion(tmp_path: Path) -> None:
    bin_dir = tmp_path / "build" / "bin"
    bin_dir.mkdir(parents=True)
    completion = bin_dir / "llama-completion"
    cli = bin_dir / "llama-cli"
    completion.write_text("", encoding="utf-8")
    cli.write_text("", encoding="utf-8")
    sample = {
        "prompt": "Return B.",
        "prompt_format": "qwen3-no-think",
        "max_tokens": 8,
    }

    command = generation_command(
        ModelTarget("q3", tmp_path / "model.gguf"),
        tmp_path,
        sample,
        timeout_seconds=10,
    )

    assert generation_binary(tmp_path) == completion
    assert command[0] == str(completion)
    assert "-no-cnv" in command
    assert "--no-display-prompt" in command


def test_llama_server_command_uses_persistent_runtime_flags(tmp_path: Path) -> None:
    bin_dir = tmp_path / "build" / "bin"
    bin_dir.mkdir(parents=True)
    server = bin_dir / "llama-server"
    server.write_text("", encoding="utf-8")

    command = llama_server_command(
        ModelTarget("q3", tmp_path / "model.gguf"),
        tmp_path,
        host="127.0.0.1",
        port=18080,
        timeout_seconds=120,
        context_size=4096,
        gpu_layers=99,
    )

    assert server_binary(tmp_path) == server
    assert command[0] == str(server)
    assert "--no-webui" in command
    assert command[command.index("--port") + 1] == "18080"


def test_run_server_generation_scores_completion_response(monkeypatch) -> None:
    class FakeResponse:
        status_code = 200
        text = '{"content": "D"}'

        def json(self) -> dict[str, object]:
            return {"content": "D", "timings": {"predicted_ms": 12}}

    calls = []

    def fake_post(url, json, timeout):
        calls.append((url, json, timeout))
        return FakeResponse()

    monkeypatch.setattr(runner.requests, "post", fake_post)
    sample = {
        "task_id": "offset:0",
        "benchmark_id": "mmlu_pro",
        "prompt": "Return D.",
        "prompt_format": "qwen3-no-think",
        "max_tokens": 16,
        "scoring_rule": "multiple_choice_letter",
        "answer": "D",
    }

    result = run_server_generation("http://127.0.0.1:18080", sample, 60, temperature=0.0, top_p=None, top_k=None, seed=None)

    assert result["passed"] is True
    assert result["server_timings"] == {"predicted_ms": 12}
    assert calls[0][0] == "http://127.0.0.1:18080/completion"
    assert calls[0][1]["n_predict"] == 16


def test_apply_max_tokens_caps_long_samples() -> None:
    sample = {"max_tokens": 4096}

    capped = apply_max_tokens(sample, 512)

    assert capped["max_tokens"] == 512
    assert sample["max_tokens"] == 4096


def test_apply_generation_overrides_can_switch_prompt_mode() -> None:
    sample = {"max_tokens": 4096, "prompt_format": "qwen3-no-think"}

    updated = apply_generation_overrides(sample, max_tokens=1024, prompt_format="qwen3-thinking")

    assert updated["max_tokens"] == 1024
    assert updated["prompt_format"] == "qwen3-thinking"
    assert sample["prompt_format"] == "qwen3-no-think"


def test_fetch_dataset_viewer_row_retries_transient_errors(monkeypatch) -> None:
    calls = []
    responses = [
        {"status": 502, "payload": {}},
        {"status": 200, "payload": {"rows": [{"row": {"question": "q", "choices": ["a"], "answer": 0}}]}},
    ]

    class FakeResponse:
        def __init__(self, status: int, payload: dict[str, object]) -> None:
            self.status_code = status
            self._payload = payload

        def raise_for_status(self) -> None:
            if self.status_code >= 400:
                raise requests.HTTPError(f"{self.status_code} error")

        def json(self) -> dict[str, object]:
            return self._payload

    def fake_get(*args, **kwargs):
        calls.append((args, kwargs))
        item = responses.pop(0)
        return FakeResponse(int(item["status"]), dict(item["payload"]))

    monkeypatch.setattr(runner.requests, "get", fake_get)
    monkeypatch.setattr(runner.time, "sleep", lambda _seconds: None)

    row = fetch_dataset_viewer_row(ADAPTERS["mmlu"], "offset:0")

    assert row["question"] == "q"
    assert len(calls) == 2


def test_fetch_hf_raw_jsonl_row_reads_pinned_livecodebench_file(monkeypatch) -> None:
    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def iter_lines(self, decode_unicode: bool = False):
            assert decode_unicode is True
            yield '{"question_id": "first"}'
            yield '{"question_id": "second"}'

    calls = []

    def fake_get(url, *args, **kwargs):
        calls.append((url, kwargs))
        return FakeResponse()

    monkeypatch.setattr(runner.requests, "get", fake_get)

    row = fetch_hf_raw_jsonl_row(ADAPTERS["livecodebench"], "offset:1")

    assert row["question_id"] == "second"
    assert "livecodebench/code_generation_lite/resolve/0fe84c3912ea0c4d4a78037083943e8f0c4dd505/test6.jsonl" in calls[0][0]
    assert calls[0][1]["stream"] is True
