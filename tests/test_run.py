import json
from pathlib import Path

from opentq.run import seed_existing_results


def test_seed_existing_results_backfills_meta_not_yet_in_progress(tmp_path: Path) -> None:
    output_root = tmp_path / "release"
    progress_path = output_root / "progress.jsonl"
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path.write_text(
        '{"name":"a.weight","category":"mlp_proj","mode":"quantize","variant_name":"TQ4_SB4","dtype":"bf16","shape":[4,4],"source_file":"model-00001.safetensors","tensor_dir":"tensors/a__weight","part_count":1,"num_values":16,"mse":0.1,"max_abs_error":0.2,"sum_squared_error":1.6,"skipped":false}\n',
        encoding="utf-8",
    )

    recovered_dir = output_root / "tensors" / "b__weight"
    recovered_dir.mkdir(parents=True)
    (recovered_dir / "meta.json").write_text(
        json.dumps(
            {
                "name": "b.weight",
                "category": "layernorm",
                "mode": "copy",
                "shape": [4],
                "dtype": "bf16",
                "storage_dtype": "float32",
                "source_file": "model-00001.safetensors",
                "part_count": 1,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    plan = {
        "tensors": [
            {"name": "a.weight", "mode": "quantize"},
            {"name": "b.weight", "mode": "copy"},
            {"name": "c.weight", "mode": "quantize"},
        ]
    }

    results = seed_existing_results(output_root, plan, progress_path)

    assert len(results) == 2
    assert {result.name for result in results} == {"a.weight", "b.weight"}
    recovered = next(result for result in results if result.name == "b.weight")
    assert recovered.num_values == 4
    assert recovered.mse is None

    lines = progress_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
