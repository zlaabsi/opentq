from __future__ import annotations

from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True)
class InventoryRow:
    category: str
    count: int
    samples: tuple[str, ...]


def classify_qwen36_27b_tensor(name: str) -> str:
    if name == "lm_head.weight":
        return "lm_head"
    if name == "model.language_model.embed_tokens.weight":
        return "embeddings"
    if name.startswith("model.visual.patch_embed."):
        return "visual_patch_embed"
    if name.startswith("model.visual.merger."):
        return "visual_merger"
    if name.startswith("model.visual.blocks."):
        if ".attn." in name:
            return "visual_attn"
        if ".mlp." in name:
            return "visual_mlp"
        if ".norm" in name:
            return "visual_norm"
        return "visual_misc"
    if name.startswith("model.vision_tower."):
        return "vision_tower"
    if name.startswith("model.mm_projector."):
        return "mm_projector"

    if ".self_attn." in name:
        if name.endswith((".q_proj.weight", ".k_proj.weight", ".v_proj.weight", ".o_proj.weight")):
            return "self_attn_proj"
        if name.endswith((".q_norm.weight", ".k_norm.weight")):
            return "self_attn_norm"
        return "self_attn_misc"

    if ".linear_attn." in name:
        if name.endswith((".in_proj_a.weight", ".in_proj_b.weight", ".in_proj_qkv.weight", ".in_proj_z.weight", ".out_proj.weight")):
            return "linear_attn_proj"
        if name.endswith(".conv1d.weight"):
            return "linear_attn_conv"
        if name.endswith(".norm.weight"):
            return "linear_attn_norm"
        if name.endswith((".A_log", ".dt_bias")):
            return "linear_attn_state"
        return "linear_attn_misc"

    if ".mlp." in name:
        if name.endswith((".gate_proj.weight", ".up_proj.weight", ".down_proj.weight")):
            return "mlp_proj"
        return "mlp_misc"

    if name.endswith(("input_layernorm.weight", "post_attention_layernorm.weight", "norm.weight")):
        return "layernorm"

    return "other"


def build_inventory(weight_map: dict[str, str], sample_limit: int = 3) -> list[InventoryRow]:
    grouped: dict[str, list[str]] = {}
    for name in sorted(weight_map):
        category = classify_qwen36_27b_tensor(name)
        grouped.setdefault(category, []).append(name)

    rows = []
    for category in sorted(grouped):
        names = grouped[category]
        rows.append(InventoryRow(category=category, count=len(names), samples=tuple(names[:sample_limit])))
    return rows


def inventory_summary(weight_map: dict[str, str]) -> dict[str, int]:
    counter = Counter(classify_qwen36_27b_tensor(name) for name in weight_map)
    return dict(sorted(counter.items()))
