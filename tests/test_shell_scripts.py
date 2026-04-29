from __future__ import annotations

import subprocess
from pathlib import Path


def test_run_qwen36_otq_eval_script_is_bash_3_compatible() -> None:
    script = Path("scripts/run_qwen36_otq_eval.sh")

    completed = subprocess.run(["bash", "-n", str(script)], text=True, capture_output=True, check=False)

    assert completed.returncode == 0, completed.stderr


def test_qwen36_release_scripts_are_bash_3_compatible() -> None:
    scripts = [
        Path("scripts/run_qwen36_representative_benchmarks.sh"),
        Path("scripts/launch_qwen36_representative_benchmarks.sh"),
        Path("scripts/run_qwen36_publication_candidate_benchmarks.sh"),
        Path("scripts/launch_qwen36_publication_candidate_benchmarks.sh"),
        Path("scripts/run_qwen36_bf16_paired_quality.sh"),
        Path("scripts/launch_qwen36_bf16_paired_quality.sh"),
        Path("scripts/launch_qwen36_hf_bf16_sidecar_smoke.sh"),
    ]

    for script in scripts:
        completed = subprocess.run(["bash", "-n", str(script)], text=True, capture_output=True, check=False)

        assert completed.returncode == 0, f"{script}: {completed.stderr}"
