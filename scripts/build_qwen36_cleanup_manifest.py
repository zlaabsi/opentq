#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests


MODEL_REPO_ID = "zlaabsi/Qwen3.6-27B-OTQ-GGUF"
DATASET_REPO_ID = "zlaabsi/Qwen3.6-27B-OTQ-GGUF-benchmarks"
HF_API = "https://huggingface.co/api"


@dataclass(frozen=True)
class CleanupCandidate:
    path: Path
    kind: str
    reason: str
    repo_id: str | None = None
    repo_type: str | None = None
    require_remote_match: bool = False
    safe_by_default: bool = False


DEFAULT_CANDIDATES = [
    CleanupCandidate(
        path=Path("artifacts/hf-gguf-canonical/Qwen3.6-27B-OTQ-GGUF"),
        kind="published_hf_model_stage",
        reason="Canonical local staging directory for the public stock GGUF model repo.",
        repo_id=MODEL_REPO_ID,
        repo_type="model",
        require_remote_match=True,
        safe_by_default=True,
    ),
    CleanupCandidate(
        path=Path("artifacts/hf-datasets/Qwen3.6-27B-OTQ-GGUF-benchmarks"),
        kind="published_hf_dataset_stage",
        reason="Local staging directory for the public benchmark reproducibility dataset.",
        repo_id=DATASET_REPO_ID,
        repo_type="dataset",
        require_remote_match=True,
        safe_by_default=False,
    ),
    CleanupCandidate(
        path=Path("artifacts/qwen3.6-27b-source"),
        kind="bf16_source_gguf",
        reason="Large local BF16 GGUF source. Useful for future sidecars and conversions; do not delete automatically.",
        safe_by_default=False,
    ),
    CleanupCandidate(
        path=Path("artifacts/qwen3.6-27b"),
        kind="raw_quantization_workdir",
        reason="Raw tensor workdir for native packed releases. Keep until Packed/Metal-native public runtime gates are finished.",
        safe_by_default=False,
    ),
    CleanupCandidate(
        path=Path("artifacts/hf-runtime"),
        kind="native_runtime_stage",
        reason="Packed and Metal-native staging area. Keep while native runtime gates are active.",
        safe_by_default=False,
    ),
    CleanupCandidate(
        path=Path("artifacts/qwen3.6-27b-gguf"),
        kind="custom_opentq_gguf_stage",
        reason="Custom OpenTQ GGUF variants used by native Metal gates. Keep unless superseded by hf-runtime artifacts.",
        safe_by_default=False,
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and optionally apply a safe Qwen3.6 cleanup manifest.")
    parser.add_argument("--output", default="artifacts/release-audit/qwen36-cleanup-manifest.json")
    parser.add_argument("--apply", action="store_true", help="Delete candidates that are proven safe by this manifest.")
    parser.add_argument("--repo-root", default=".")
    return parser.parse_args()


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def human_gib(value: int) -> str:
    return f"{value / (1024 ** 3):.2f} GiB"


def local_files(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for child in sorted(path.rglob("*")):
        if not child.is_file():
            continue
        if any(part in {".cache", ".git"} for part in child.parts):
            continue
        rows.append(
            {
                "relative_path": child.relative_to(path).as_posix(),
                "bytes": child.stat().st_size,
            }
        )
    return rows


def fetch_repo_files(repo_id: str, repo_type: str) -> dict[str, int]:
    if repo_type == "model":
        url = f"{HF_API}/models/{repo_id}"
        params = {"blobs": "true"}
    elif repo_type == "dataset":
        url = f"{HF_API}/datasets/{repo_id}"
        params = {"blobs": "true"}
    else:
        raise ValueError(f"unsupported repo type: {repo_type}")
    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()
    payload = response.json()
    rows: dict[str, int] = {}
    for sibling in payload.get("siblings", []):
        name = sibling.get("rfilename")
        size = sibling.get("size")
        if name is not None and size is not None:
            rows[str(name)] = int(size)
    return rows


def remote_match(candidate: CleanupCandidate, files: list[dict[str, Any]]) -> dict[str, Any]:
    if not candidate.require_remote_match:
        return {"required": False, "ok": None, "missing": [], "size_mismatches": []}
    if candidate.repo_id is None or candidate.repo_type is None:
        return {"required": True, "ok": False, "missing": ["missing repo metadata"], "size_mismatches": []}

    remote = fetch_repo_files(candidate.repo_id, candidate.repo_type)
    missing: list[str] = []
    mismatches: list[dict[str, Any]] = []
    for row in files:
        rel = row["relative_path"]
        remote_size = remote.get(rel)
        if remote_size is None:
            missing.append(rel)
            continue
        if int(row["bytes"]) != int(remote_size):
            mismatches.append({"path": rel, "local_bytes": row["bytes"], "remote_bytes": remote_size})
    return {
        "required": True,
        "ok": not missing and not mismatches and bool(files),
        "repo_id": candidate.repo_id,
        "repo_type": candidate.repo_type,
        "missing": missing,
        "size_mismatches": mismatches,
    }


def build_manifest(repo_root: Path) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for candidate in DEFAULT_CANDIDATES:
        path = repo_root / candidate.path
        files = local_files(path)
        total = sum(int(row["bytes"]) for row in files)
        match = remote_match(candidate, files) if path.exists() else {"required": candidate.require_remote_match, "ok": False, "missing": ["local path missing"], "size_mismatches": []}
        safe = bool(path.exists() and candidate.safe_by_default and (not candidate.require_remote_match or match.get("ok") is True))
        records.append(
            {
                "path": str(candidate.path),
                "exists": path.exists(),
                "kind": candidate.kind,
                "reason": candidate.reason,
                "file_count": len(files),
                "bytes": total,
                "human_size": human_gib(total),
                "remote_match": match,
                "safe_to_delete": safe,
            }
        )
    return {
        "schema": "opentq.qwen36_cleanup_manifest.v1",
        "created_at": now_iso(),
        "records": records,
        "safe_delete_bytes": sum(row["bytes"] for row in records if row["safe_to_delete"]),
        "safe_delete_human": human_gib(sum(row["bytes"] for row in records if row["safe_to_delete"])),
    }


def apply_manifest(repo_root: Path, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    deleted: list[dict[str, Any]] = []
    artifact_root = (repo_root / "artifacts").resolve()
    for row in manifest["records"]:
        if not row.get("safe_to_delete"):
            continue
        target = (repo_root / row["path"]).resolve()
        if artifact_root not in target.parents:
            raise RuntimeError(f"refusing to delete outside artifacts: {target}")
        if not target.exists():
            continue
        shutil.rmtree(target)
        deleted.append({"path": row["path"], "bytes": row["bytes"], "human_size": row["human_size"]})
    return deleted


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    manifest = build_manifest(repo_root)
    if args.apply:
        manifest["deleted"] = apply_manifest(repo_root, manifest)
        manifest["applied_at"] = now_iso()
    output = repo_root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)
    print(f"safe_to_delete={manifest['safe_delete_human']}")
    if args.apply:
        print(f"deleted={human_gib(sum(row['bytes'] for row in manifest.get('deleted', [])))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
