from __future__ import annotations

from pathlib import Path

from scripts.build_qwen36_cleanup_manifest import build_manifest, classify_path, inspect_path


def test_classify_path_marks_uploaded_verified() -> None:
    assert classify_path("artifacts/hf-gguf-canonical/Qwen3.6-27B-OTQ-GGUF", True, True) == "uploaded-verified"
    assert classify_path("artifacts/qwen3.6-27b-source/Qwen3.6-27B-BF16.gguf", False, False) == "blocked"
    assert classify_path("~/.cache/huggingface/hub/models--Qwen--Qwen3.6-27B", False, False) == "regenerable-cache"
    assert classify_path("artifacts/qwen3.6-27b-dynamic-gguf", False, False) == "uploaded-verified"
    assert classify_path("artifacts/hf-gguf-canonical/Qwen3.6-27B-OTQ-GGUF-next", False, False) == "regenerable"
    assert classify_path("artifacts/qwen3.6-27b", False, False) == "gated-local-artifact"
    assert classify_path("artifacts/qwen3.6-27b/Qwen3.6-27B-TQ4R2", False, False) == "gated-local-artifact"
    assert classify_path("artifacts/qwen3.6-27b-source", False, False) == "blocked"
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


def test_inspect_path_counts_unique_inodes_inside_directory(tmp_path: Path) -> None:
    first = tmp_path / "first.bin"
    second = tmp_path / "second.bin"
    third = tmp_path / "third.bin"
    first.write_bytes(b"1234")
    second.hardlink_to(first)
    third.write_bytes(b"ab")

    info = inspect_path(tmp_path)

    assert info["file_count"] == 3
    assert info["unique_file_inodes"] == 2
    assert info["apparent_bytes"] == 6
    assert info["bytes"] == 6
    assert info["max_file_hardlink_count"] >= 2


def test_build_manifest_is_non_destructive(tmp_path: Path) -> None:
    cache_dir = tmp_path / ".cache" / "huggingface" / "xet"
    cache_dir.mkdir(parents=True)
    (cache_dir / "chunk").write_bytes(b"x")

    manifest = build_manifest([cache_dir])

    assert manifest[0]["classification"] == "regenerable-cache"
    assert manifest[0]["suggested_action"] == "delete-if-disk-pressure"
    assert manifest[0]["approved_reclaimable_bytes_now"] > 0
    assert manifest[0]["potential_reclaimable_bytes_upper_bound"] >= manifest[0]["approved_reclaimable_bytes_now"]
    assert manifest[0]["delete_command"].startswith("rm -rf ")
    assert (cache_dir / "chunk").exists()
