from __future__ import annotations

import json
import sys
from pathlib import Path


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


def main() -> int:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("artifacts/qwen3.6-27b")
    if not root.exists():
        print(json.dumps({"root": str(root), "exists": False}, indent=2))
        return 0

    releases = []
    for child in sorted(root.iterdir()):
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

    payload = {
        "root": str(root),
        "releases": releases,
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
