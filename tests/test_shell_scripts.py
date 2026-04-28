from __future__ import annotations

import subprocess
from pathlib import Path


def test_run_qwen36_otq_eval_script_is_bash_3_compatible() -> None:
    script = Path("scripts/run_qwen36_otq_eval.sh")

    completed = subprocess.run(["bash", "-n", str(script)], text=True, capture_output=True, check=False)

    assert completed.returncode == 0, completed.stderr
