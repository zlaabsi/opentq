from __future__ import annotations

import json
import subprocess
from pathlib import Path


SCHEMA = "opentq.qwen36_benchmark_subset_eval.v1"


def _write_model(path: Path, key: str, results: list[dict[str, object]]) -> None:
    path.write_text(
        json.dumps(
            {
                "schema": SCHEMA,
                "model": {"key": key},
                "benchmarks": [
                    {
                        "benchmark_id": "mmlu_pro",
                        "adapter": {"prompt_format": "qwen3-no-think", "scoring_rule": "multiple_choice_letter"},
                        "summary": {
                            "total": len(results),
                            "passed": sum(1 for item in results if item["passed"]),
                            "failed": sum(1 for item in results if not item["passed"]),
                            "pass_rate": sum(1 for item in results if item["passed"]) / len(results),
                        },
                        "results": results,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_paired_quality_report_compares_identical_task_ids(tmp_path: Path) -> None:
    subset_root = tmp_path / "subsets"
    subset_root.mkdir()
    _write_model(
        subset_root / "bf16_gguf.json",
        "bf16_gguf",
        [
            {"task_id": "offset:0", "passed": True, "score": {"actual": "A"}},
            {"task_id": "offset:1", "passed": True, "score": {"actual": "B"}},
            {"task_id": "offset:2", "passed": False, "score": {"actual": "C"}},
        ],
    )
    _write_model(
        subset_root / "q4.json",
        "q4",
        [
            {"task_id": "offset:0", "passed": True, "score": {"actual": "A"}},
            {"task_id": "offset:1", "passed": False, "score": {"actual": "C"}},
            {"task_id": "offset:2", "passed": True, "score": {"actual": "D"}},
        ],
    )
    output_root = tmp_path / "report"

    completed = subprocess.run(
        [
            "python",
            "scripts/build_qwen36_paired_quality_report.py",
            "--subset-root",
            str(subset_root),
            "--candidate-key",
            "q4",
            "--output-root",
            str(output_root),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads((output_root / "paired-quality-report.json").read_text(encoding="utf-8"))
    row = payload["rows"][0]
    comparison = row["comparisons"]["q4"]
    assert comparison["baseline_passed"] == 2
    assert comparison["candidate_passed"] == 2
    assert comparison["delta_pp"] == 0
    assert comparison["baseline_only"] == 1
    assert comparison["candidate_only"] == 1
    assert comparison["publishability"] == "diagnostic_only_sample_lt_30"
    markdown = (output_root / "paired-quality-report.md").read_text(encoding="utf-8")
    assert "Official Qwen scores are anchors only" in markdown
    assert "BF16-only 1, quant-only 1" in markdown
