from pathlib import Path

from opentq.status import build_status_payload


def test_build_status_payload_for_running_release(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    release = root / "Qwen3.6-27B-TQ4_SB4"
    tensor_dir = release / "tensors" / "tensor_a"
    tensor_dir.mkdir(parents=True)
    (release / "progress.jsonl").write_text('{"name": "a"}\n{"name": "b"}\n', encoding="utf-8")
    (tensor_dir / "part-00000.npz").write_bytes(b"x")
    (tensor_dir / "part-00001.npz").write_bytes(b"x")

    payload = build_status_payload(root)

    assert payload["root"] == str(root)
    assert len(payload["releases"]) == 1
    release_payload = payload["releases"][0]
    assert release_payload["release"] == "Qwen3.6-27B-TQ4_SB4"
    assert release_payload["state"] == "running"
    assert release_payload["processed_tensors"] == 2
    assert release_payload["active_tensor_dirs"] == 1
    assert release_payload["written_part_files"] == 2


def test_build_status_payload_for_missing_root(tmp_path: Path) -> None:
    root = tmp_path / "missing"
    payload = build_status_payload(root)
    assert payload == {"root": str(root), "exists": False}
