from pathlib import Path

from opentq.pruning import PruningCandidateOptions, build_pruning_candidates, write_pruning_candidates


def _write_plan(path: Path) -> None:
    path.write_text(
        """
{
  "model_id": "Qwen/Qwen3.6-27B",
  "profile": {"name": "OTQ-DYN-Q3_K_M"},
  "tensors": [
    {
      "hf_name": "model.language_model.layers.0.mlp.down_proj.weight",
      "category": "mlp_proj",
      "mode": "set_type",
      "ggml_type": "Q4_K",
      "layer_index": 0,
      "reason": "edge-layer override for first/last 2 layers"
    },
    {
      "hf_name": "model.language_model.layers.10.mlp.down_proj.weight",
      "category": "mlp_proj",
      "mode": "set_type",
      "ggml_type": "Q3_K",
      "layer_index": 10,
      "reason": "profile category allocation"
    },
    {
      "hf_name": "model.language_model.layers.10.self_attn.q_proj.weight",
      "category": "self_attn_proj",
      "mode": "set_type",
      "ggml_type": "Q5_K",
      "layer_index": 10,
      "reason": "profile category allocation"
    }
  ]
}
""",
        encoding="utf-8",
    )


def test_pruning_candidates_rank_middle_mlp_above_edge_layer(tmp_path: Path) -> None:
    plan = tmp_path / "plan.json"
    _write_plan(plan)

    payload = build_pruning_candidates(PruningCandidateOptions(plan_path=plan, output_dir=tmp_path / "out"))
    by_unit = {row["unit_id"]: row for row in payload["candidates"]}

    assert by_unit["layer.10.mlp_proj"]["action"] in {"prune_candidate", "quantize_aggressive"}
    assert by_unit["layer.10.mlp_proj"]["score"] > by_unit["layer.0.mlp_proj"]["score"]
    assert by_unit["layer.0.mlp_proj"]["action"] == "keep_high_precision"


def test_write_pruning_candidates_outputs_policy_and_report(tmp_path: Path) -> None:
    plan = tmp_path / "plan.json"
    _write_plan(plan)

    payload = write_pruning_candidates(PruningCandidateOptions(plan_path=plan, output_dir=tmp_path / "out"))

    assert (tmp_path / "out" / "pruning-candidates.json").exists()
    assert (tmp_path / "out" / "pruning-candidates.jsonl").exists()
    assert (tmp_path / "out" / "pruning-policy.yaml").exists()
    assert (tmp_path / "out" / "paired-pruning-report.md").exists()
    assert payload["summary"]["candidate_count"] == 3
