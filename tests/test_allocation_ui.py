from pathlib import Path

from opentq.allocation_ui import AllocationUIOptions, build_allocation_ui_data, write_allocation_ui


def _write_plan(path: Path) -> None:
    path.write_text(
        """
{
  "model_id": "Qwen/Qwen3.6-27B",
  "profile": {"name": "OTQ-DYN-Q4_K_M"},
  "tensors": [
    {
      "hf_name": "lm_head.weight",
      "gguf_name": "output.weight",
      "category": "lm_head",
      "mode": "set_type",
      "ggml_type": "Q8_0",
      "layer_index": null,
      "reason": "profile category allocation"
    },
    {
      "hf_name": "model.language_model.layers.3.self_attn.q_proj.weight",
      "gguf_name": "blk.3.attn_q.weight",
      "category": "self_attn_proj",
      "mode": "set_type",
      "ggml_type": "Q6_K",
      "layer_index": 3,
      "reason": "periodic anchor override every 4 layers"
    },
    {
      "hf_name": "model.visual.blocks.0.attn.qkv.weight",
      "category": "visual_attn",
      "mode": "skip",
      "ggml_type": null,
      "layer_index": null,
      "reason": "text-only release skips vision tensors"
    }
  ]
}
""",
        encoding="utf-8",
    )


def test_allocation_ui_data_filters_skipped_tensors(tmp_path: Path) -> None:
    plan = tmp_path / "plan.json"
    _write_plan(plan)

    payload = build_allocation_ui_data(AllocationUIOptions(plan_path=plan, output_dir=tmp_path / "ui"))

    assert payload["summary"]["tensor_count"] == 2
    assert payload["summary"]["type_counts"]["Q8_0"] == 1
    assert payload["summary"]["type_counts"]["Q6_K"] == 1
    assert payload["tensors"][0]["color"]


def test_write_allocation_ui_outputs_data_and_html(tmp_path: Path) -> None:
    plan = tmp_path / "plan.json"
    _write_plan(plan)

    payload = write_allocation_ui(AllocationUIOptions(plan_path=plan, output_dir=tmp_path / "ui"))

    assert (tmp_path / "ui" / "allocation-ui-data.json").exists()
    assert (tmp_path / "ui" / "index.html").exists()
    assert (tmp_path / "ui" / "README.md").exists()
    assert "OpenTQ Allocation Explorer" in (tmp_path / "ui" / "index.html").read_text(encoding="utf-8")
    assert payload["outputs"]["html"].endswith("index.html")
