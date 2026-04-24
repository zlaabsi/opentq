from __future__ import annotations

from dataclasses import dataclass

from .inventory import classify_qwen36_27b_tensor


@dataclass(frozen=True)
class TensorAction:
    category: str
    mode: str
    variant_name: str | None = None


_UNIFORM_QUANTIZED_CATEGORIES = {
    "embeddings",
    "lm_head",
    "linear_attn_proj",
    "linear_attn_conv",
    "self_attn_proj",
    "mlp_proj",
    "visual_attn",
    "visual_mlp",
    "visual_patch_embed",
    "visual_merger",
    "visual_pos_embed",
    "mtp_proj",
}

_COPY_CATEGORIES = {
    "layernorm",
    "linear_attn_norm",
    "linear_attn_state",
    "self_attn_norm",
    "visual_norm",
    "mtp_norm",
    "vision_tower",
    "mm_projector",
    "visual_misc",
    "other",
}

_UNIFORM_RELEASES = {
    "QWEN3.6-27B-TQ1_0": "TQ1_0",
    "QWEN3.6-27B-TQ2_0": "TQ2_0",
    "QWEN3.6-27B-TQ3_SB4": "TQ3_SB4",
    "QWEN3.6-27B-TQ4_SB4": "TQ4_SB4",
    "QWEN3.6-27B-TQ4R2": "TQ4R2",
    "QWEN3.6-27B-TQ4R4": "TQ4R4",
}

_BALANCED_VARIANTS = {
    "embeddings": "TQ4R4",
    "lm_head": "TQ4R4",
    "self_attn_proj": "TQ4R2",
    "linear_attn_proj": "TQ4_SB4",
    "linear_attn_conv": "TQ4_SB4",
    "mlp_proj": "TQ3_SB4",
    "visual_attn": "TQ4_SB4",
    "visual_mlp": "TQ4_SB4",
    "visual_patch_embed": "TQ4_SB4",
    "visual_merger": "TQ4_SB4",
    "visual_pos_embed": "TQ4_SB4",
    "mtp_proj": "TQ4_SB4",
}


def resolve_tensor_action(release_slug: str, tensor_name: str, include_vision: bool = True, include_language: bool = True) -> TensorAction:
    category = classify_qwen36_27b_tensor(tensor_name)
    is_visual = category.startswith("visual")
    is_language = not is_visual and category not in {"vision_tower", "mm_projector"}

    if is_visual and not include_vision:
        return TensorAction(category=category, mode="skip")
    if is_language and not include_language:
        return TensorAction(category=category, mode="skip")

    normalized_slug = release_slug.upper()

    if normalized_slug in _UNIFORM_RELEASES:
        if category in _UNIFORM_QUANTIZED_CATEGORIES:
            return TensorAction(category=category, mode="quantize", variant_name=_UNIFORM_RELEASES[normalized_slug])
        if category in _COPY_CATEGORIES:
            return TensorAction(category=category, mode="copy")
        return TensorAction(category=category, mode="copy")

    if normalized_slug == "QWEN3.6-27B-TQ_BAL_DENSE":
        if category in _BALANCED_VARIANTS:
            return TensorAction(category=category, mode="quantize", variant_name=_BALANCED_VARIANTS[category])
        if category in _COPY_CATEGORIES:
            return TensorAction(category=category, mode="copy")
        return TensorAction(category=category, mode="copy")

    raise KeyError(f"unsupported release slug: {release_slug}")
