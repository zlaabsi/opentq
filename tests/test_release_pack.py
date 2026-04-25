import json
from pathlib import Path

import numpy as np

from opentq.bitpack import unpack_bits
from opentq.hf_release import prepare_hf_release
from opentq.release_pack import pack_release


def write_dummy_release(root: Path) -> None:
    quant_dir = root / "tensors" / "a__weight"
    copy_dir = root / "tensors" / "b__weight"
    quant_dir.mkdir(parents=True)
    copy_dir.mkdir(parents=True)

    indices = (np.arange(64, dtype=np.uint8) % 16).reshape(2, 32)
    np.savez_compressed(
        quant_dir / "part-00000.npz",
        indices=indices,
        scales=np.ones((2, 4), dtype=np.float32),
        residual_indices=np.array([], dtype=np.uint8),
        residual_scales=np.array([], dtype=np.float32),
        shape=np.array([2, 4], dtype=np.int64),
        row_start=np.array([0], dtype=np.int64),
        row_stop=np.array([2], dtype=np.int64),
    )
    np.save(copy_dir / "part-00000.npy", np.array([1.0, 2.0], dtype=np.float32), allow_pickle=False)

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
            },
            {
                "name": "b.weight",
                "category": "layernorm",
                "mode": "copy",
                "variant_name": None,
                "dtype": "bf16",
                "shape": [2],
                "source_file": "model-00001.safetensors",
                "tensor_dir": "tensors/b__weight",
                "part_count": 1,
                "num_values": 2,
                "mse": None,
                "max_abs_error": None,
                "sum_squared_error": None,
                "skipped": False,
            },
        ],
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def test_pack_release_emits_bitpacked_payload(tmp_path: Path) -> None:
    source = tmp_path / "release"
    output = tmp_path / "packed"
    write_dummy_release(source)

    payload = pack_release(source, output)

    assert payload["schema"] == "opentq.pack.v1"
    assert payload["totals"]["tensors"] == 2
    assert payload["totals"]["quantized_tensors"] == 1
    tensor = payload["tensors"][0]
    tensor_file = output / tensor["file"]
    assert tensor_file.exists()

    section = tensor["sections"][0]["indices"]
    data = tensor_file.read_bytes()[section["offset"] : section["offset"] + section["bytes"]]
    restored = unpack_bits(data, 4, tensor["sections"][0]["index_count"])
    expected = np.arange(64, dtype=np.uint8) % 16
    np.testing.assert_array_equal(restored, expected)


def test_prepare_hf_release_writes_card_and_summary(tmp_path: Path) -> None:
    source = tmp_path / "release"
    packed = tmp_path / "packed"
    stage = tmp_path / "hf"
    write_dummy_release(source)
    pack_release(source, packed)

    summary = prepare_hf_release(packed, stage, "zlaabsi/Qwen3.6-27B-TQ4_SB4", link_mode="none")

    assert summary["repo_id"] == "zlaabsi/Qwen3.6-27B-TQ4_SB4"
    assert (stage / "README.md").exists()
    assert (stage / "opentq-release.json").exists()
    assert (stage / "opentq-pack.json").exists()
