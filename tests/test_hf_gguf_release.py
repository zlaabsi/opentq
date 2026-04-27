import json
from pathlib import Path

from opentq.hf_gguf_release import prepare_hf_gguf_release


def test_prepare_hf_gguf_release_stages_only_public_files(tmp_path: Path) -> None:
    gguf = tmp_path / "Qwen3.6-27B-TQ4_BAL_V2.gguf"
    gguf.write_bytes(b"GGUF-smoke")
    validation = tmp_path / "validation.json"
    validation.write_text(
        json.dumps(
            {
                "schema": "opentq.gguf_validation.v1",
                "created_at": "2026-04-27T00:00:00+00:00",
                "overall_pass": True,
                "artifact": {
                    "filename": gguf.name,
                    "bytes": gguf.stat().st_size,
                },
                "gates": {
                    "gguf_metadata_read": True,
                    "bounded_generation": True,
                    "benchmark": True,
                },
                "runtime": {
                    "bench_prompt_tokens": 8192,
                    "bench_gen_tokens": 128,
                },
            }
        ),
        encoding="utf-8",
    )
    stage = tmp_path / "hf"

    summary = prepare_hf_gguf_release(
        gguf,
        stage,
        "zlaabsi/Qwen3.6-27B-TQ4_BAL_V2-GGUF",
        link_mode="copy",
        compute_sha256=False,
        validation_path=validation,
    )

    assert summary["schema"] == "opentq.hf_gguf_release.v1"
    assert summary["release"]["excluded_private_artifacts"] == ["*.otq", "opentq-pack.json"]
    assert (stage / gguf.name).exists()
    assert (stage / "README.md").exists()
    metadata = json.loads((stage / "opentq-gguf-release.json").read_text(encoding="utf-8"))
    assert metadata["artifact"]["filename"] == gguf.name
    assert metadata["validation"]["overall_pass"] is True


def test_prepare_hf_gguf_release_blocks_unvalidated_artifacts(tmp_path: Path) -> None:
    gguf = tmp_path / "Qwen3.6-27B-TQ3_SB4.gguf"
    gguf.write_bytes(b"GGUF-smoke")

    try:
        prepare_hf_gguf_release(
            gguf,
            tmp_path / "hf",
            "zlaabsi/Qwen3.6-27B-TQ3_SB4-GGUF",
            link_mode="copy",
            compute_sha256=False,
        )
    except ValueError as exc:
        assert "missing required validation" in str(exc)
    else:
        raise AssertionError("expected unvalidated GGUF staging to fail")
