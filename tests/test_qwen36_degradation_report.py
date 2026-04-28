from __future__ import annotations

import json
import subprocess
from pathlib import Path


def test_degradation_report_marks_missing_subset_scores_as_pending(tmp_path: Path) -> None:
    output_root = tmp_path / "report"
    completed = subprocess.run(
        [
            "python",
            "scripts/build_qwen36_degradation_report.py",
            "--matrix",
            "benchmarks/qwen36_long_running_benchmark_matrix.json",
            "--official-baseline",
            "benchmarks/qwen36_official_language_baseline.json",
            "--subset-root",
            str(tmp_path / "missing-subsets"),
            "--output-root",
            str(output_root),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads((output_root / "degradation-report.json").read_text(encoding="utf-8"))
    rows = {row["benchmark_id"]: row for row in payload["rows"]}
    assert rows["mmlu_pro"]["claim_status"] == "pending_otq_subset"
    assert rows["mmlu"]["claim_status"] == "pending_bf16_or_otq_only"
    assert "Official benchmark deltas are not claimed" in (output_root / "degradation-report.md").read_text(encoding="utf-8")


def test_degradation_report_markdown_includes_subset_scores(tmp_path: Path) -> None:
    subset_root = tmp_path / "subsets"
    subset_root.mkdir()
    (subset_root / "q3.json").write_text(
        json.dumps(
            {
                "schema": "opentq.qwen36_benchmark_subset_eval.v1",
                "model": {"key": "q3"},
                "benchmarks": [
                    {
                        "benchmark_id": "mmlu_pro",
                        "summary": {"total": 4, "passed": 3, "failed": 1, "pass_rate": 0.75},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    output_root = tmp_path / "report"

    completed = subprocess.run(
        [
            "python",
            "scripts/build_qwen36_degradation_report.py",
            "--matrix",
            "benchmarks/qwen36_long_running_benchmark_matrix.json",
            "--official-baseline",
            "benchmarks/qwen36_official_language_baseline.json",
            "--subset-root",
            str(subset_root),
            "--output-root",
            str(output_root),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads((output_root / "degradation-report.json").read_text(encoding="utf-8"))
    rows = {row["benchmark_id"]: row for row in payload["rows"]}
    markdown = (output_root / "degradation-report.md").read_text(encoding="utf-8")
    assert rows["mmlu_pro"]["claim_status"] == "official_delta_candidate_review_required"
    assert "`q3` 3/4 (75.0%), candidate -11.2 pp" in markdown
    assert "MMLU-Pro: 86.2 score" in markdown
