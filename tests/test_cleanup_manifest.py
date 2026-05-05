from __future__ import annotations

import json
from pathlib import Path

from scripts import build_qwen36_cleanup_manifest as cleanup


def test_manifest_marks_canonical_stage_safe_when_remote_sizes_match(tmp_path: Path, monkeypatch) -> None:
    repo = tmp_path
    stage = repo / "artifacts/hf-gguf-canonical/Qwen3.6-27B-OTQ-GGUF"
    stage.mkdir(parents=True)
    (stage / "README.md").write_text("card\n", encoding="utf-8")
    (stage / "Qwen3.6-27B-OTQ-DYN-Q3_K_M.gguf").write_bytes(b"abc")

    def fake_fetch(repo_id: str, repo_type: str) -> dict[str, int]:
        assert repo_id == cleanup.MODEL_REPO_ID
        assert repo_type == "model"
        return {
            "README.md": 5,
            "Qwen3.6-27B-OTQ-DYN-Q3_K_M.gguf": 3,
        }

    monkeypatch.setattr(cleanup, "fetch_repo_files", fake_fetch)

    manifest = cleanup.build_manifest(repo)
    row = next(item for item in manifest["records"] if item["path"] == "artifacts/hf-gguf-canonical/Qwen3.6-27B-OTQ-GGUF")

    assert row["safe_to_delete"] is True
    assert row["remote_match"]["ok"] is True
    assert manifest["safe_delete_bytes"] == 8


def test_manifest_blocks_canonical_stage_when_remote_file_is_missing(tmp_path: Path, monkeypatch) -> None:
    repo = tmp_path
    stage = repo / "artifacts/hf-gguf-canonical/Qwen3.6-27B-OTQ-GGUF"
    stage.mkdir(parents=True)
    (stage / "README.md").write_text("card\n", encoding="utf-8")
    (stage / "Qwen3.6-27B-OTQ-DYN-Q3_K_M.gguf").write_bytes(b"abc")

    monkeypatch.setattr(cleanup, "fetch_repo_files", lambda repo_id, repo_type: {"README.md": 5})

    manifest = cleanup.build_manifest(repo)
    row = next(item for item in manifest["records"] if item["path"] == "artifacts/hf-gguf-canonical/Qwen3.6-27B-OTQ-GGUF")

    assert row["safe_to_delete"] is False
    assert row["remote_match"]["ok"] is False
    assert row["remote_match"]["missing"] == ["Qwen3.6-27B-OTQ-DYN-Q3_K_M.gguf"]


def test_apply_manifest_only_deletes_safe_artifacts(tmp_path: Path) -> None:
    repo = tmp_path
    safe = repo / "artifacts/safe"
    keep = repo / "artifacts/keep"
    safe.mkdir(parents=True)
    keep.mkdir(parents=True)
    (safe / "file.txt").write_text("safe", encoding="utf-8")
    (keep / "file.txt").write_text("keep", encoding="utf-8")
    manifest = {
        "records": [
            {"path": "artifacts/safe", "safe_to_delete": True, "bytes": 4, "human_size": "0.00 GiB"},
            {"path": "artifacts/keep", "safe_to_delete": False, "bytes": 4, "human_size": "0.00 GiB"},
        ]
    }

    deleted = cleanup.apply_manifest(repo, manifest)

    assert deleted == [{"path": "artifacts/safe", "bytes": 4, "human_size": "0.00 GiB"}]
    assert not safe.exists()
    assert keep.exists()
