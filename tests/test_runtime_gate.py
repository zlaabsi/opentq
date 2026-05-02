from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from opentq.release_pack import pack_release
from opentq.run import TensorArtifactResult, write_copy_tensor, write_quantized_tensor
from opentq.runtime_gate import (
    PackAuditOptions,
    RuntimeProbeOptions,
    audit_packed_runtime,
    run_runtime_probe,
    write_runtime_fixtures,
)


class DummySlice:
    def __init__(self, array: np.ndarray):
        self.array = array

    def get_shape(self):
        return self.array.shape

    def get_dtype(self):
        return "f32"

    def __getitem__(self, item):
        return self.array[item]


class DummyReader:
    def __init__(self, tensors: dict[str, np.ndarray]):
        self.tensors = tensors

    def get_slice(self, name: str):
        return DummySlice(self.tensors[name])

    def get_tensor(self, name: str):
        return self.tensors[name]


def write_runtime_release(root: Path) -> None:
    release_slug = "Qwen3.6-27B-TQ4_SB4"
    quant_name = "model.language_model.layers.0.mlp.up_proj.weight"
    copy_name = "model.language_model.layers.0.input_layernorm.weight"
    values = np.linspace(-1.0, 1.0, 256, dtype=np.float32).reshape(2, 128)
    copied = np.ones((128,), dtype=np.float32)
    reader = DummyReader({quant_name: values, copy_name: copied})
    quant_action = type("Action", (), {"category": "mlp_proj", "mode": "quantize", "variant_name": "TQ4_SB4"})()
    copy_action = type("Action", (), {"category": "layernorm", "mode": "copy", "variant_name": None})()

    quant_result: TensorArtifactResult = write_quantized_tensor(reader, quant_name, "dummy.safetensors", quant_action, root, release_slug)
    copy_result: TensorArtifactResult = write_copy_tensor(reader, copy_name, "dummy.safetensors", copy_action, root)
    manifest = {
        "model_id": "dummy/model",
        "release_slug": release_slug,
        "results": [quant_result.__dict__, copy_result.__dict__],
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def test_audit_packed_runtime_passes_for_valid_pack(tmp_path: Path) -> None:
    source = tmp_path / "release"
    packed = tmp_path / "packed"
    write_runtime_release(source)
    pack_release(source, packed)

    payload = audit_packed_runtime(PackAuditOptions(packed, dequantize_samples=2))

    assert payload["overall_pass"] is True
    assert payload["schema_pass"] is True
    assert payload["mode_counts"] == {"copy": 1, "quantize": 1}
    assert payload["variant_counts"] == {"TQ4_SB4": 1}
    assert payload["checksum"]["failures"] == []
    assert payload["dequantize"]["failures"] == []


def test_runtime_fixtures_emit_probe_inputs(tmp_path: Path) -> None:
    source = tmp_path / "release"
    packed = tmp_path / "packed"
    fixtures = tmp_path / "fixtures"
    write_runtime_release(source)
    pack_release(source, packed)

    payload = write_runtime_fixtures(packed, fixtures)

    assert payload["schema"] == "opentq.runtime_fixtures.v1"
    assert len(payload["fixtures"]) == 1
    fixture = payload["fixtures"][0]
    assert fixture["variant"] == "TQ4_SB4"
    assert fixture["block_bytes"] == 100
    assert Path(fixture["block"]).stat().st_size == 100
    assert np.fromfile(fixture["expected"], dtype=np.float32).shape == (128,)
    assert np.fromfile(fixture["activation"], dtype=np.float32).shape == (128,)
    assert np.fromfile(fixture["expected_dot"], dtype=np.float32).shape == (1,)


def test_runtime_probe_requires_external_probe_binary(tmp_path: Path) -> None:
    source = tmp_path / "release"
    packed = tmp_path / "packed"
    output = tmp_path / "probe.json"
    write_runtime_release(source)
    pack_release(source, packed)

    payload = run_runtime_probe(
        RuntimeProbeOptions(
            packed_dir=packed,
            fixtures_output=tmp_path / "fixtures",
            probe_binary=tmp_path / "missing-probe",
            output=output,
        )
    )

    assert payload["overall_pass"] is False
    assert payload["audit"]["overall_pass"] is True
    assert payload["probe_status"] == "probe-binary-not-found"
    assert output.exists()
