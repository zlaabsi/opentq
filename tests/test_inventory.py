from opentq.inventory import classify_qwen36_27b_tensor


def test_classify_qwen36_attention_and_linear_roles() -> None:
    assert classify_qwen36_27b_tensor("model.language_model.layers.11.self_attn.q_proj.weight") == "self_attn_proj"
    assert classify_qwen36_27b_tensor("model.language_model.layers.10.linear_attn.in_proj_qkv.weight") == "linear_attn_proj"
    assert classify_qwen36_27b_tensor("model.language_model.layers.10.linear_attn.A_log") == "linear_attn_state"

