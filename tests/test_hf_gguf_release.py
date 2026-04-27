import json
from pathlib import Path

from opentq.hf_gguf_release import prepare_hf_gguf_release


def test_prepare_hf_gguf_release_stages_only_public_files(tmp_path: Path) -> None:
    gguf = tmp_path / "Qwen3.6-27B-TQ4_BAL_V2.gguf"
    gguf.write_bytes(b"GGUF-smoke")
    stage = tmp_path / "hf"

    summary = prepare_hf_gguf_release(
        gguf,
        stage,
        "zlaabsi/Qwen3.6-27B-TQ4_BAL_V2-GGUF",
        link_mode="copy",
        compute_sha256=False,
    )

    assert summary["schema"] == "opentq.hf_gguf_release.v1"
    assert summary["release"]["excluded_private_artifacts"] == ["*.otq", "opentq-pack.json"]
    assert (stage / gguf.name).exists()
    assert (stage / "README.md").exists()
    metadata = json.loads((stage / "opentq-gguf-release.json").read_text(encoding="utf-8"))
    assert metadata["artifact"]["filename"] == gguf.name
