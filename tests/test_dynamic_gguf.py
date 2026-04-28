from pathlib import Path

from opentq.dynamic_gguf import (
    build_dynamic_tensor_plans,
    build_quantize_script,
    exact_regex_pattern,
    get_dynamic_profile,
    resolve_dynamic_tensor_action,
    tensor_type_lines,
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


def test_dynamic_docs_explain_iq4_nl_and_do_not_use_xl() -> None:
    text = Path("docs/dynamic-compatible-gguf.md").read_text(encoding="utf-8")

    assert "XL" not in text
    assert "IQ4_NL is a stock llama.cpp nonlinear 4-bit quant type" in text
    assert "requires an imatrix before release consideration" in text
