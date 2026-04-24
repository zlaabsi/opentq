from opentq.policies import resolve_tensor_action


def test_balanced_policy_routes_core_tensor_roles() -> None:
    assert resolve_tensor_action("Qwen3.6-27B-TQ4_BAL_V2", "lm_head.weight").variant_name == "TQ4R2"
    assert resolve_tensor_action("Qwen3.6-27B-TQ4_BAL_V2", "model.language_model.layers.11.self_attn.q_proj.weight").variant_name == "TQ4R2"
    assert resolve_tensor_action("Qwen3.6-27B-TQ4_BAL_V2", "model.language_model.layers.0.mlp.up_proj.weight").variant_name == "TQ4_SB2"
    assert resolve_tensor_action("Qwen3.6-27B-TQ4_BAL_V2", "model.language_model.layers.0.linear_attn.A_log").mode == "copy"
