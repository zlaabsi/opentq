from __future__ import annotations

import json
import subprocess
from pathlib import Path

from scripts.run_qwen36_benchmark_subsets import (
    ADAPTERS,
    build_sample_from_row,
    apply_max_tokens,
    generation_binary,
    generation_command,
    ModelTarget,
    score_benchmark_output,
)


MATRIX = Path("benchmarks/qwen36_long_running_benchmark_matrix.json")


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
            "/Users/zlaabsi/Documents/GitHub/llama.cpp",
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
    assert score_benchmark_output(numeric, "Final answer: 18")["passed"] is True


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


def test_apply_max_tokens_caps_long_samples() -> None:
    sample = {"max_tokens": 4096}

    capped = apply_max_tokens(sample, 512)

    assert capped["max_tokens"] == 512
    assert sample["max_tokens"] == 4096
