from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


BLOCKED_NAMES = {"Qwen3.6-27B-BF16.gguf"}


def classify_path(path: str, uploaded: bool, regenerable: bool) -> str:
    name = Path(path).name
    if uploaded:
        return "uploaded-verified"
    if name in BLOCKED_NAMES:
        return "blocked"
    if regenerable:
        return "regenerable"
    return "investigate"


def inspect_path(path: Path) -> dict[str, Any]:
    stat = path.stat()
    if path.is_file():
        total = stat.st_size
    else:
        total = sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
    return {
        "path": str(path),
        "bytes": total,
        "inode": stat.st_ino,
        "hardlink_count": stat.st_nlink,
    }


def build_manifest(paths: list[Path]) -> list[dict[str, Any]]:
    records = []
    for path in paths:
        info = inspect_path(path)
        info["classification"] = classify_path(str(path), uploaded=False, regenerable=False)
        info["delete_command"] = None
        records.append(info)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Qwen3.6 cleanup manifest without deleting files.")
    parser.add_argument("--output", type=Path, default=Path("artifacts/release-audit/qwen36-cleanup-manifest.json"))
    parser.add_argument("paths", nargs="*", type=Path)
    args = parser.parse_args()

    paths = args.paths or [
        Path("artifacts/qwen3.6-27b"),
        Path("artifacts/hf-runtime"),
        Path("artifacts/qwen3.6-27b-source"),
        Path("artifacts/hf-gguf-canonical"),
        Path("artifacts/qwen3.6-27b-gguf"),
    ]
    existing = [path for path in paths if path.exists()]
    payload = build_manifest(existing)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
