from __future__ import annotations

import json
import subprocess
from pathlib import Path


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
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 2
    assert "execution adapters are not implemented yet" in completed.stdout
    assert not (tmp_path / "q3.json").exists()
