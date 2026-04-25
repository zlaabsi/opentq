from pathlib import Path

import json
import numpy as np

from opentq.gguf import build_gguf_plan
from opentq.release_pack import pack_release


def write_dummy_release(root: Path) -> None:
    quant_dir = root / "tensors" / "a__weight"
    quant_dir.mkdir(parents=True)
    np.savez_compressed(
        quant_dir / "part-00000.npz",
        indices=(np.arange(64, dtype=np.uint8) % 16).reshape(2, 32),
        scales=np.ones((2, 4), dtype=np.float32),
        residual_indices=np.array([], dtype=np.uint8),
        residual_scales=np.array([], dtype=np.float32),
        shape=np.array([2, 4], dtype=np.int64),
        row_start=np.array([0], dtype=np.int64),
        row_stop=np.array([2], dtype=np.int64),
    )
    manifest = {
        "model_id": "Qwen/Qwen3.6-27B",
        "release_slug": "Qwen3.6-27B-TQ4_SB4",
        "results": [
            {
                "name": "a.weight",
                "category": "mlp_proj",
                "mode": "quantize",
                "variant_name": "TQ4_SB4",
                "dtype": "bf16",
                "shape": [2, 4],
                "source_file": "model-00001.safetensors",
                "tensor_dir": "tensors/a__weight",
                "part_count": 1,
                "num_values": 8,
                "mse": 0.1,
                "max_abs_error": 0.2,
                "sum_squared_error": 0.8,
                "skipped": False,
            }
        ],
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def test_build_gguf_plan_maps_opentq_tensor_types(tmp_path: Path) -> None:
    source = tmp_path / "release"
    packed = tmp_path / "packed"
    write_dummy_release(source)
    pack_release(source, packed)

    plan = build_gguf_plan(packed)

    assert plan["schema"] == "opentq.gguf_plan.v1"
    assert "OPENTQ_TQ4_SB4" in plan["custom_tensor_types"]
    assert plan["required_metadata"]["general.quantization_version"] == 1
