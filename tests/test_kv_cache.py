from pathlib import Path

from opentq.kv_cache import KVCachePlanOptions, build_kv_cache_policy, write_kv_cache_policy


def test_kv_cache_policy_promotes_edge_and_periodic_layers() -> None:
    payload = build_kv_cache_policy(
        KVCachePlanOptions(
            output_dir=Path("unused"),
            num_layers=10,
            default_dtype="fp8_e4m3",
            promote_dtype="bf16",
            edge_layers=1,
            periodic_stride=4,
        )
    )

    by_layer = {row["layer"]: row for row in payload["layers"]}

    assert by_layer[0]["key_dtype"] == "bf16"
    assert by_layer[3]["key_dtype"] == "bf16"
    assert by_layer[4]["key_dtype"] == "fp8_e4m3"
    assert by_layer[9]["key_dtype"] == "bf16"
    assert payload["summary"]["promoted_layers"] == [0, 3, 7, 9]


def test_kv_cache_policy_couples_to_promoted_weight_plan_layers(tmp_path: Path) -> None:
    plan = tmp_path / "plan.json"
    plan.write_text(
        """
{
  "tensors": [
    {
      "hf_name": "model.language_model.layers.5.self_attn.q_proj.weight",
      "category": "self_attn_proj",
      "mode": "set_type",
      "ggml_type": "Q6_K",
      "layer_index": 5,
      "reason": "periodic anchor override every 4 layers"
    },
    {
      "hf_name": "model.language_model.layers.6.mlp.down_proj.weight",
      "category": "mlp_proj",
      "mode": "set_type",
      "ggml_type": "Q3_K",
      "layer_index": 6,
      "reason": "profile category allocation"
    }
  ]
}
""",
        encoding="utf-8",
    )

    payload = build_kv_cache_policy(
        KVCachePlanOptions(
            output_dir=tmp_path / "kv",
            num_layers=8,
            edge_layers=0,
            periodic_stride=0,
            weight_plan=plan,
        )
    )

    by_layer = {row["layer"]: row for row in payload["layers"]}
    assert by_layer[5]["key_dtype"] == "bf16"
    assert "coupled to weight policy" in by_layer[5]["reason"]
    assert by_layer[6]["key_dtype"] == "fp8_e4m3"


def test_write_kv_cache_policy_outputs_json_tsv_and_rationale(tmp_path: Path) -> None:
    payload = write_kv_cache_policy(KVCachePlanOptions(output_dir=tmp_path, num_layers=2))

    assert (tmp_path / "kv-cache-policy.json").exists()
    assert (tmp_path / "kv-cache-policy.tsv").exists()
    assert (tmp_path / "kv-cache-rationale.md").exists()
    assert payload["outputs"]["json"].endswith("kv-cache-policy.json")
