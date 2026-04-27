from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from opentq.release_pack import pack_release
from opentq.run import TensorArtifactResult, write_quantized_tensor
from opentq.runtime import OpenTQPack


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


def test_runtime_dequantizes_packed_tensor(tmp_path: Path):
    source = tmp_path / "source"
    release_slug = "Qwen3.6-27B-TQ4_SB4"
    tensor_name = "model.language_model.layers.0.mlp.up_proj.weight"
    values = np.linspace(-1.0, 1.0, 256, dtype=np.float32).reshape(2, 128)
    reader = DummyReader({tensor_name: values})
    action = type("Action", (), {"category": "mlp_proj", "mode": "quantize", "variant_name": "TQ4_SB4"})()

    result: TensorArtifactResult = write_quantized_tensor(reader, tensor_name, "dummy.safetensors", action, source, release_slug)
    manifest = {
        "model_id": "dummy/model",
        "release_slug": release_slug,
        "results": [result.__dict__],
    }
    (source / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    packed_dir = tmp_path / "packed"
    pack_release(source, packed_dir)
    decoded = OpenTQPack(packed_dir).dequantize_tensor(tensor_name)

    assert decoded.shape == values.shape
    assert np.mean((decoded - values) ** 2) < 0.01
