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
