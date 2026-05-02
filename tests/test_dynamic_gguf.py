from pathlib import Path

from opentq.dynamic_gguf import (
    DynamicGGUFPlanOptions,
    build_dynamic_tensor_plans,
    build_quantize_script,
    exact_regex_pattern,
    get_dynamic_profile,
    load_dynamic_policy_file,
    resolve_dynamic_tensor_action,
    tensor_type_lines,
    write_dynamic_gguf_plan,
)


def test_dynamic_profile_applies_edge_and_periodic_overrides() -> None:
    profile = get_dynamic_profile("OTQ-DYN-Q4_K_M")

    edge = resolve_dynamic_tensor_action(profile, "model.language_model.layers.0.mlp.down_proj.weight")
    middle = resolve_dynamic_tensor_action(profile, "model.language_model.layers.10.mlp.down_proj.weight")
    periodic = resolve_dynamic_tensor_action(profile, "model.language_model.layers.3.self_attn.q_proj.weight")

    assert edge.ggml_type == "Q5_K"
    assert edge.reason.startswith("edge-layer override")
    assert middle.ggml_type == "Q4_K"
    assert periodic.ggml_type == "Q6_K"
    assert periodic.reason.startswith("periodic anchor override")


def test_dynamic_profile_keeps_stock_gguf_names_and_types() -> None:
    profile = get_dynamic_profile("otq-dyn-q3-k-m")
    weight_map = {
        "lm_head.weight": "model-00001.safetensors",
        "model.language_model.embed_tokens.weight": "model-00001.safetensors",
        "model.language_model.layers.0.self_attn.q_proj.weight": "model-00001.safetensors",
        "model.language_model.layers.10.mlp.down_proj.weight": "model-00002.safetensors",
        "model.visual.blocks.0.attn.qkv.weight": "model-00003.safetensors",
    }

    plans = build_dynamic_tensor_plans(
        model_id="Qwen/Qwen3.6-27B",
        weight_map=weight_map,
        profile=profile,
        include_vision=False,
    )
    by_name = {row.hf_name: row for row in plans}

    assert by_name["lm_head.weight"].gguf_name == "output.weight"
    assert by_name["lm_head.weight"].ggml_type == "Q6_K"
    assert by_name["model.language_model.embed_tokens.weight"].gguf_name == "token_embd.weight"
    assert by_name["model.language_model.layers.0.self_attn.q_proj.weight"].gguf_name == "blk.0.attn_q.weight"
    assert by_name["model.language_model.layers.10.mlp.down_proj.weight"].ggml_type == "Q3_K"
    assert by_name["model.visual.blocks.0.attn.qkv.weight"].mode == "skip"


def test_tensor_type_file_uses_exact_regex_patterns() -> None:
    profile = get_dynamic_profile("OTQ-DYN-Q4_K_M")
    weight_map = {
        "lm_head.weight": "model-00001.safetensors",
        "model.language_model.layers.3.self_attn.q_proj.weight": "model-00001.safetensors",
    }
    plans = build_dynamic_tensor_plans(model_id="Qwen/Qwen3.6-27B", weight_map=weight_map, profile=profile)

    lines = tensor_type_lines(plans)

    assert f"{exact_regex_pattern('output.weight')}=q8_0" in lines
    assert f"{exact_regex_pattern('blk.3.attn_q.weight')}=q6_k" in lines


def test_quantize_script_supports_dry_run_and_imatrix() -> None:
    profile = get_dynamic_profile("OTQ-DYN-IQ4_NL")
    script = build_quantize_script(
        profile=profile,
        llama_cpp_dir=Path("../llama.cpp"),
        source_gguf=Path("base.gguf"),
        target_gguf=Path("out.gguf"),
    )

    assert "--tensor-type-file" in script
    assert "--dry-run" in script
    assert "--imatrix" in script
    assert "IQ4_NL" in script


def test_load_dynamic_policy_file_supports_yaml_profiles(tmp_path: Path) -> None:
    policy = tmp_path / "my-dyn-q4.yaml"
    policy.write_text(
        """
name: MY-DYN-Q4
base_ftype: Q4_K_M
target: custom 32GB Apple Silicon profile
requires_imatrix: false
category_types:
  embeddings: Q6_K
  lm_head: Q8_0
  self_attn_proj: Q6_K
  linear_attn_proj: Q5_K
  linear_attn_conv: F16
  mlp_proj: Q3_K
edge_layers: 2
edge_overrides:
  mlp_proj: Q5_K
  self_attn_proj: Q8_0
periodic_stride: 4
periodic_overrides:
  self_attn_proj: Q6_K
notes: user supplied policy
""",
        encoding="utf-8",
    )

    profile = load_dynamic_policy_file(policy)
    edge = resolve_dynamic_tensor_action(profile, "model.language_model.layers.0.self_attn.q_proj.weight")
    middle = resolve_dynamic_tensor_action(profile, "model.language_model.layers.10.mlp.down_proj.weight")

    assert profile.name == "MY-DYN-Q4"
    assert profile.base_ftype == "Q4_K_M"
    assert profile.target == "custom 32GB Apple Silicon profile"
    assert profile.category_types["mlp_proj"] == "Q3_K"
    assert edge.ggml_type == "Q8_0"
    assert middle.ggml_type == "Q3_K"


def test_load_dynamic_policy_file_supports_json_profiles(tmp_path: Path) -> None:
    policy = tmp_path / "my-dyn-q5.json"
    policy.write_text(
        """
{
  "name": "MY-DYN-Q5",
  "base_ftype": "Q5_K_M",
  "target": "custom quality-first profile",
  "category_types": {
    "embeddings": "Q8_0",
    "lm_head": "Q8_0",
    "self_attn_proj": "Q6_K",
    "linear_attn_proj": "Q6_K",
    "linear_attn_conv": "F16",
    "mlp_proj": "Q5_K"
  }
}
""",
        encoding="utf-8",
    )

    profile = load_dynamic_policy_file(policy)

    assert profile.name == "MY-DYN-Q5"
    assert profile.requires_imatrix is False
    assert profile.edge_overrides == {}
    assert profile.periodic_overrides == {}
    assert profile.edge_layers == 2
    assert profile.periodic_stride == 4


def test_load_dynamic_policy_file_rejects_unknown_categories_and_types(tmp_path: Path) -> None:
    policy = tmp_path / "bad-policy.json"
    policy.write_text(
        """
{
  "name": "BAD",
  "base_ftype": "Q4_K_M",
  "target": "bad",
  "requires_imatrix": false,
  "category_types": {
    "not_a_category": "Q4_K",
    "mlp_proj": "NOT_A_GGML_TYPE"
  }
}
""",
        encoding="utf-8",
    )

    try:
        load_dynamic_policy_file(policy)
    except ValueError as exc:
        message = str(exc)
    else:
        raise AssertionError("expected invalid custom policy to fail")

    assert "not_a_category" in message
    assert "NOT_A_GGML_TYPE" in message


def test_write_dynamic_gguf_plan_accepts_external_policy_file(tmp_path: Path, monkeypatch) -> None:
    policy = tmp_path / "my-dyn-q4.yaml"
    policy.write_text(
        """
name: MY-DYN-Q4
base_ftype: Q4_K_M
target: custom local experiment
category_types:
  embeddings: Q6_K
  lm_head: Q8_0
  self_attn_proj: Q6_K
  linear_attn_proj: Q5_K
  linear_attn_conv: F16
  mlp_proj: Q3_K
edge_overrides:
  self_attn_proj: Q8_0
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "opentq.dynamic_gguf.fetch_safetensors_index",
        lambda model_id: {
            "weight_map": {
                "lm_head.weight": "model-00001.safetensors",
                "model.language_model.layers.0.self_attn.q_proj.weight": "model-00001.safetensors",
                "model.language_model.layers.10.mlp.down_proj.weight": "model-00002.safetensors",
            }
        },
    )

    output_dir = tmp_path / "plan"
    payload = write_dynamic_gguf_plan(
        DynamicGGUFPlanOptions(
            recipe_key="qwen3.6-27b",
            output_dir=output_dir,
            policy_file=policy,
            use_converter_mapping=False,
        )
    )

    tensor_types = (output_dir / "tensor-types.txt").read_text(encoding="utf-8")

    assert payload["profile"]["name"] == "MY-DYN-Q4"
    assert payload["policy_source"]["kind"] == "policy_file"
    assert payload["policy_source"]["path"] == str(policy)
    assert f"{exact_regex_pattern('blk.0.attn_q.weight')}=q8_0" in tensor_types
    assert f"{exact_regex_pattern('blk.10.ffn_down.weight')}=q3_k" in tensor_types
    assert (output_dir / "plan.json").exists()
    assert (output_dir / "quantize.sh").exists()


def test_dynamic_docs_explain_iq4_nl_and_do_not_use_xl() -> None:
    text = Path("docs/dynamic-compatible-gguf.md").read_text(encoding="utf-8")

    assert "XL" not in text
    assert "IQ4_NL is a stock llama.cpp nonlinear 4-bit quant type" in text
    assert "requires an imatrix before release consideration" in text
