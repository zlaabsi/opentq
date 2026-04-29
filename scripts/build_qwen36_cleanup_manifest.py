from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


BLOCKED_NAMES = {"Qwen3.6-27B-BF16.gguf"}
BLOCKED_PATH_PATTERNS = (
    "artifacts/qwen3.6-27b-source",
)
UPLOADED_VERIFIED_PATTERNS = (
    "artifacts/hf-gguf-canonical/Qwen3.6-27B-OTQ-GGUF",
    "artifacts/hf-datasets/Qwen3.6-27B-OTQ-GGUF-benchmarks",
    "artifacts/qwen3.6-27b-dynamic-gguf",
    "artifacts/qwen3.6-27b-benchmark-subsets-release-core-232",
    "artifacts/qwen3.6-27b-bf16-hf-sidecar",
    "artifacts/qwen3.6-27b-paired-bf16-quant-report-232",
)
REGENERABLE_LOCAL_PATTERNS = (
    "artifacts/hf-gguf-canonical/Qwen3.6-27B-OTQ-GGUF-next",
)
GATED_LOCAL_PATTERNS = (
    "artifacts/qwen3.6-27b",
    "artifacts/hf-runtime/Qwen3.6-27B-OTQ-Packed",
    "artifacts/hf-runtime/Qwen3.6-27B-OTQ-Metal-GGUF",
    "artifacts/qwen3.6-27b-gguf",
)
REGENERABLE_CACHE_PATTERNS = (
    ".cache/huggingface/hub/models--Qwen--Qwen3.6-27B",
    ".cache/huggingface/xet",
    ".cache/huggingface/hub/models--BAAI--bge-m3",
)


def normalize_path(path: str) -> str:
    return str(Path(path).expanduser())


def path_contains(path: str, patterns: tuple[str, ...]) -> bool:
    normalized = normalize_path(path)
    for pattern in patterns:
        normalized_pattern = normalize_path(pattern)
        if pattern.startswith("artifacts/"):
            if normalized == normalized_pattern or normalized.startswith(f"{normalized_pattern}/"):
                return True
            continue
        if normalized_pattern in normalized:
            return True
    return False


def classify_path(path: str, uploaded: bool, regenerable: bool) -> str:
    normalized = normalize_path(path)
    name = Path(normalized).name
    if uploaded:
        return "uploaded-verified"
    if name in BLOCKED_NAMES or path_contains(normalized, BLOCKED_PATH_PATTERNS):
        return "blocked"
    if path_contains(normalized, REGENERABLE_CACHE_PATTERNS):
        return "regenerable-cache"
    if path_contains(normalized, UPLOADED_VERIFIED_PATTERNS):
        return "uploaded-verified"
    if path_contains(normalized, REGENERABLE_LOCAL_PATTERNS):
        return "regenerable"
    if path_contains(normalized, GATED_LOCAL_PATTERNS):
        return "gated-local-artifact"
    if regenerable:
        return "regenerable"
    return "investigate"


def inspect_path(path: Path) -> dict[str, Any]:
    stat = path.stat()
    if path.is_file():
        apparent_total = stat.st_size
        disk_total = getattr(stat, "st_blocks", 0) * 512
        files = 1
        unique_inodes = 1
        max_hardlink_count = stat.st_nlink
    else:
        apparent_total = 0
        disk_total = 0
        files = 0
        seen_inodes: set[tuple[int, int]] = set()
        max_hardlink_count = stat.st_nlink
        for item in path.rglob("*"):
            if not item.is_file():
                continue
            item_stat = item.stat()
            files += 1
            max_hardlink_count = max(max_hardlink_count, item_stat.st_nlink)
            key = (item_stat.st_dev, item_stat.st_ino)
            if key in seen_inodes:
                continue
            seen_inodes.add(key)
            apparent_total += item_stat.st_size
            disk_total += getattr(item_stat, "st_blocks", 0) * 512
        unique_inodes = len(seen_inodes)
    return {
        "path": str(path),
        "bytes": apparent_total,
        "apparent_bytes": apparent_total,
        "disk_bytes": disk_total,
        "file_count": files,
        "unique_file_inodes": unique_inodes,
        "inode": stat.st_ino,
        "hardlink_count": stat.st_nlink,
        "max_file_hardlink_count": max_hardlink_count,
    }


def rationale_for(classification: str) -> str:
    if classification == "blocked":
        return "Preserve. This is the local BF16 source anchor or another explicitly blocked artifact."
    if classification == "uploaded-verified":
        return "Remote/public evidence exists. Candidate for later cleanup only after a fresh HF inventory check."
    if classification == "regenerable-cache":
        return (
            "Regenerable Hugging Face cache. Local BF16 GGUF source is preserved separately, "
            "so this cache has lower option value than the disk it blocks."
        )
    if classification == "regenerable":
        return "Local staging output that can be rebuilt from preserved sources, but do not delete without an explicit cleanup decision."
    if classification == "gated-local-artifact":
        return "Preserve for now. This belongs to Packed/Metal/native work that is not public-release-ready yet."
    return "Needs manual inspection before any cleanup decision."


def suggested_action_for(classification: str) -> str:
    if classification == "regenerable-cache":
        return "delete-if-disk-pressure"
    if classification == "uploaded-verified":
        return "candidate-after-hf-reverification"
    if classification == "regenerable":
        return "candidate-after-local-rebuild-check"
    if classification in {"blocked", "gated-local-artifact"}:
        return "preserve"
    return "inspect"


def approved_reclaim_bytes(classification: str, disk_bytes: int) -> int:
    if classification == "regenerable-cache":
        return disk_bytes
    return 0


def reclaim_note_for(info: dict[str, Any]) -> str:
    classification = str(info["classification"])
    if classification == "regenerable-cache":
        return "Approved only when disk pressure returns; still run the delete command manually."
    if classification == "blocked":
        return "Not reclaimable under the current release policy."
    if classification == "gated-local-artifact":
        return "Potential size is not approved reclaim because runtime/publication gates are still open."
    if classification == "uploaded-verified":
        return "Potential size is an upper bound; repeat HF inventory and account for hardlinks before deletion."
    if classification == "regenerable":
        return "Potential size is an upper bound; verify local rebuild path before deletion."
    return "Manual inspection required before estimating reclaim."


def build_manifest(paths: list[Path]) -> list[dict[str, Any]]:
    records = []
    for path in paths:
        info = inspect_path(path)
        info["classification"] = classify_path(str(path), uploaded=False, regenerable=False)
        info["suggested_action"] = suggested_action_for(info["classification"])
        info["potential_reclaimable_bytes_upper_bound"] = 0 if info["classification"] == "blocked" else info["disk_bytes"]
        info["approved_reclaimable_bytes_now"] = approved_reclaim_bytes(info["classification"], info["disk_bytes"])
        info["reclaim_estimate_note"] = reclaim_note_for(info)
        info["delete_command"] = None
        if info["classification"] == "regenerable-cache":
            info["delete_command"] = f"rm -rf {path}"
        info["decision_rationale"] = rationale_for(info["classification"])
        records.append(info)
    return records


def default_paths() -> list[Path]:
    return [
        Path("artifacts/qwen3.6-27b"),
        Path("artifacts/qwen3.6-27b-source"),
        Path("artifacts/qwen3.6-27b-dynamic-gguf"),
        Path("artifacts/hf-gguf-canonical/Qwen3.6-27B-OTQ-GGUF"),
        Path("artifacts/hf-gguf-canonical/Qwen3.6-27B-OTQ-GGUF-next"),
        Path("artifacts/hf-datasets/Qwen3.6-27B-OTQ-GGUF-benchmarks"),
        Path("artifacts/hf-runtime/Qwen3.6-27B-OTQ-Packed"),
        Path("artifacts/hf-runtime/Qwen3.6-27B-OTQ-Metal-GGUF"),
        Path("artifacts/qwen3.6-27b-gguf"),
        Path("artifacts/qwen3.6-27b-benchmark-subsets-release-core-232"),
        Path("artifacts/qwen3.6-27b-bf16-hf-sidecar"),
        Path("artifacts/qwen3.6-27b-paired-bf16-quant-report-232"),
        Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3.6-27B",
        Path.home() / ".cache/huggingface/xet",
        Path.home() / ".cache/huggingface/hub/models--BAAI--bge-m3",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Qwen3.6 cleanup manifest without deleting files.")
    parser.add_argument("--output", type=Path, default=Path("artifacts/release-audit/qwen36-cleanup-manifest.json"))
    parser.add_argument("paths", nargs="*", type=Path)
    args = parser.parse_args()

    paths = args.paths or default_paths()
    existing = [path for path in paths if path.exists()]
    payload = build_manifest(existing)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
