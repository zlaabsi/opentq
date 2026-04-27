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
    assert (stage / "validation.json").exists()


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


def test_prepare_hf_gguf_release_supports_stock_compatible_dynamic(tmp_path: Path) -> None:
    gguf = tmp_path / "Qwen3.6-27B-OTQ-DYN-Q4_K_M.gguf"
    gguf.write_bytes(b"GGUF-dynamic")
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
    quality = tmp_path / "quality.json"
    quality.write_text(
        json.dumps(
            {
                "schema": "opentq.gguf_quality_eval.v1",
                "created_at": "2026-04-27T00:00:01+00:00",
                "overall_pass": True,
                "artifact": {"filename": gguf.name},
                "summary": {"pass_rate": 1.0},
                "samples": [],
            }
        ),
        encoding="utf-8",
    )

    summary = prepare_hf_gguf_release(
        gguf,
        tmp_path / "hf",
        "zlaabsi/Qwen3.6-27B-OTQ-DYN-Q4_K_M-GGUF",
        link_mode="copy",
        compute_sha256=False,
        validation_path=validation,
        quality_eval_path=quality,
        stock_compatible=True,
    )

    assert summary["runtime"]["stock_compatible"] is True
    assert summary["runtime"]["requires"] == "stock llama.cpp"
    assert summary["quality_eval"]["pass_rate"] == 1.0
    assert (tmp_path / "hf" / "quality-eval.json").exists()
    assert "validation.json" in summary["release"]["public_files"]
    assert "quality-eval.json" in summary["release"]["public_files"]
    readme = (tmp_path / "hf" / "README.md").read_text(encoding="utf-8")
    assert "standard GGUF tensor types only" in readme
    assert "Custom OpenTQ runtime: not required" in readme
