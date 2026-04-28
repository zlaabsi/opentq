from __future__ import annotations

from pathlib import Path

from scripts.build_qwen36_cleanup_manifest import classify_path, inspect_path


def test_classify_path_marks_uploaded_verified() -> None:
    assert classify_path("artifacts/hf-gguf-canonical/Qwen3.6-27B-OTQ-GGUF", True, True) == "uploaded-verified"
    assert classify_path("artifacts/qwen3.6-27b-source/Qwen3.6-27B-BF16.gguf", False, False) == "blocked"
    assert classify_path("~/.cache/huggingface/hub/models--Qwen--Qwen3.6-27B", False, False) == "regenerable-cache"
    assert classify_path("artifacts/tmp", False, True) == "regenerable"


def test_inspect_path_reports_inode_and_links(tmp_path: Path) -> None:
    first = tmp_path / "first.bin"
    second = tmp_path / "second.bin"
    first.write_bytes(b"1234")
    second.hardlink_to(first)

    info = inspect_path(first)

    assert info["bytes"] == 4
    assert info["hardlink_count"] >= 2
    assert isinstance(info["inode"], int)
