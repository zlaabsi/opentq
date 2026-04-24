from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any


def progress_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def count_parts(root: Path) -> int:
    tensors_root = root / "tensors"
    if not tensors_root.exists():
        return 0
    return sum(1 for _ in tensors_root.rglob("part-*"))


def count_tensor_dirs(root: Path) -> int:
    tensors_root = root / "tensors"
    if not tensors_root.exists():
        return 0
    return sum(1 for child in tensors_root.iterdir() if child.is_dir())


def build_status_payload(root: str | Path = "artifacts/qwen3.6-27b") -> dict[str, Any]:
    status_root = Path(root)
    if not status_root.exists():
        return {"root": str(status_root), "exists": False}

    releases = []
    for child in sorted(status_root.iterdir()):
        if not child.is_dir() or not child.name.startswith("Qwen3.6-27B-"):
            continue
        manifest_path = child / "manifest.json"
        progress_path = child / "progress.jsonl"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            releases.append(
                {
                    "release": child.name,
                    "state": "done",
                    "processed_tensors": manifest.get("processed_tensors"),
                    "elapsed_seconds": manifest.get("elapsed_seconds"),
                    "quantized_tensors": manifest.get("quantized_tensors"),
                }
            )
            continue
        releases.append(
            {
                "release": child.name,
                "state": "running",
                "processed_tensors": progress_lines(progress_path),
                "active_tensor_dirs": count_tensor_dirs(child),
                "written_part_files": count_parts(child),
            }
        )

    return {
        "root": str(status_root),
        "releases": releases,
    }


def print_status(payload: dict[str, Any], clear: bool = False) -> None:
    if clear and sys.stdout.isatty():
        print("\033[2J\033[H", end="")
    print(json.dumps(payload, indent=2), flush=True)


def watch_status(root: str | Path = "artifacts/qwen3.6-27b", interval: float = 10.0) -> int:
    try:
        while True:
            print_status(build_status_payload(root), clear=True)
            time.sleep(interval)
    except KeyboardInterrupt:
        return 130
