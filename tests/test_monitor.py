from pathlib import Path
import zlib

import numpy as np

from opentq.monitor import build_monitor_payload, render_monitor, summarize_active_parts


def test_build_monitor_payload_for_running_release(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    release = root / "Qwen3.6-27B-TQ4_SB4"
    tensors = release / "tensors"
    logs = root / "logs"
    tensors.mkdir(parents=True)
    logs.mkdir(parents=True)

    (logs / "Qwen3.6-27B-TQ4_SB4.log").write_text("[2026-04-24 05:00:00] start Qwen3.6-27B-TQ4_SB4\n", encoding="utf-8")
    (release / "plan.json").write_text(
        """
        {
          "model_id": "Qwen/Qwen3.6-27B",
          "summary": {"total_tensors": 2, "by_action": {"copy:copy": 0, "quantize:TQ4_SB4": 2}},
          "tensors": [
            {"name": "a.weight", "source_file": "model-00001.safetensors", "category": "mlp_proj", "mode": "quantize", "variant_name": "TQ4_SB4"},
            {"name": "b.weight", "source_file": "model-00001.safetensors", "category": "linear_attn_proj", "mode": "quantize", "variant_name": "TQ4_SB4"}
          ]
        }
        """.strip(),
        encoding="utf-8",
    )
    (release / "progress.jsonl").write_text(
        '{"name":"a.weight","category":"mlp_proj","mode":"quantize","variant_name":"TQ4_SB4","dtype":"bf16","shape":[4,4],"source_file":"model-00001.safetensors","tensor_dir":"tensors/a__weight","part_count":1,"num_values":16,"mse":0.1,"max_abs_error":0.2,"sum_squared_error":1.6,"skipped":false}\n',
        encoding="utf-8",
    )
    done_tensor = tensors / "a__weight"
    done_tensor.mkdir()
    (done_tensor / "meta.json").write_text('{"name": "a.weight"}\n', encoding="utf-8")

    active_tensor = tensors / "b__weight"
    active_tensor.mkdir()
    np.savez_compressed(
        active_tensor / "part-00000.npz",
        indices=np.zeros((2, 32), dtype=np.uint8),
        scales=np.zeros((2, 4), dtype=np.float32),
        residual_indices=np.zeros((0,), dtype=np.uint8),
        residual_scales=np.zeros((0,), dtype=np.float32),
        shape=np.array([2, 4], dtype=np.int64),
        row_start=np.array([0], dtype=np.int64),
        row_stop=np.array([2], dtype=np.int64),
    )

    payload = build_monitor_payload(
        root,
        shape_resolver=lambda _model, _file, name: (((4, 4), "bf16") if name == "b.weight" else ((4, 4), "bf16")),
        now=1_000.0,
    )

    assert payload["selected_release"] == "Qwen3.6-27B-TQ4_SB4"
    release_payload = payload["releases"][0]
    assert release_payload["summary"]["tensors_done"] == 1
    assert release_payload["summary"]["done_quantize"] == 1
    assert release_payload["current"]["name"] == "b.weight"
    assert release_payload["current"]["written_values"] == 8
    assert release_payload["current"]["written_blocks"] == 2


def test_render_monitor_empty_state(tmp_path: Path) -> None:
    text = render_monitor(build_monitor_payload(tmp_path / "missing"))
    assert "No runs found" in text


def test_summarize_active_parts_ignores_transient_zlib_errors(tmp_path: Path, monkeypatch) -> None:
    good = tmp_path / "part-00000.npz"
    bad = tmp_path / "part-00001.npz"
    np.savez_compressed(
        good,
        indices=np.zeros((2, 32), dtype=np.uint8),
        scales=np.zeros((2, 4), dtype=np.float32),
        shape=np.array([2, 4], dtype=np.int64),
        row_start=np.array([0], dtype=np.int64),
        row_stop=np.array([2], dtype=np.int64),
    )
    bad.write_bytes(b"not-a-real-npz")

    real_np_load = np.load

    def flaky_np_load(path: Path, *args: object, **kwargs: object):
        if Path(path) == bad:
            raise zlib.error("invalid block type")
        return real_np_load(path, *args, **kwargs)

    monkeypatch.setattr(np, "load", flaky_np_load)

    summary = summarize_active_parts([good, bad], mode="quantize")
    assert summary["observed_part_count"] == 2
    assert summary["readable_part_count"] == 1
    assert summary["written_values"] == 8
    assert summary["written_blocks"] == 2
