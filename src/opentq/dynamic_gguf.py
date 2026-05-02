from __future__ import annotations

import json
import os
import re
import shlex
import stat
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from .gguf_export import (
    load_converter_module,
    mapped_text_name,
    qwen35_transform_kind,
    snapshot_metadata,
    strip_text_name,
)
from .hf import fetch_safetensors_index
from .inventory import classify_qwen36_27b_tensor
from .recipes import get_recipe


GGML_TYPE_BPW = {
    "F32": 32.0,
    "F16": 16.0,
    "BF16": 16.0,
    "Q8_0": 8.5,
    "Q6_K": 6.5625,
    "Q5_K": 5.5,
    "Q4_K": 4.5,
    "IQ4_NL": 4.5,
    "IQ4_XS": 4.25,
    "Q3_K": 3.4375,
    "IQ3_S": 3.44,
    "IQ3_XXS": 3.06,
    "Q2_K": 2.625,
    "IQ2_XS": 2.31,
}


COPY_CATEGORIES = {
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


TEXT_SKIP_CATEGORIES = {
    "visual_attn",
    "visual_mlp",
    "visual_patch_embed",
    "visual_merger",
    "visual_pos_embed",
    "visual_norm",
    "visual_misc",
    "vision_tower",
    "mm_projector",
    "mtp_proj",
    "mtp_norm",
}


DYNAMIC_POLICY_CATEGORIES = {
    "embeddings",
    "lm_head",
    "self_attn_proj",
    "self_attn_norm",
    "self_attn_misc",
    "linear_attn_proj",
    "linear_attn_conv",
    "linear_attn_norm",
    "linear_attn_state",
    "linear_attn_misc",
    "mlp_proj",
    "mlp_misc",
    "layernorm",
    "other",
    "mtp",
    "mtp_proj",
    "mtp_norm",
    "visual_attn",
    "visual_mlp",
    "visual_patch_embed",
    "visual_merger",
    "visual_pos_embed",
    "visual_norm",
    "visual_misc",
    "vision_tower",
    "mm_projector",
}


BASE_FTYPES = {
    "F32",
    "F16",
    "BF16",
    "Q4_0",
    "Q4_1",
    "Q5_0",
    "Q5_1",
    "Q8_0",
    "Q2_K",
    "Q3_K_S",
    "Q3_K_M",
    "Q3_K_L",
    "Q4_K_S",
    "Q4_K_M",
    "Q5_K_S",
    "Q5_K_M",
    "Q6_K",
    "IQ4_NL",
    "IQ4_XS",
    "IQ3_S",
    "IQ3_XXS",
    "IQ2_XS",
}


@dataclass(frozen=True)
class DynamicGGUFProfile:
    name: str
    base_ftype: str
    target: str
    requires_imatrix: bool
    category_types: dict[str, str]
    edge_overrides: dict[str, str]
    periodic_overrides: dict[str, str]
    edge_layers: int = 2
    periodic_stride: int = 4
    notes: str = ""


@dataclass(frozen=True)
class DynamicTensorAction:
    category: str
    mode: str
    ggml_type: str | None
    reason: str
    layer_index: int | None = None


@dataclass(frozen=True)
class DynamicTensorPlan:
    hf_name: str
    gguf_name: str | None
    source_file: str
    category: str
    mode: str
    ggml_type: str | None
    layer_index: int | None
    reason: str


@dataclass(frozen=True)
class DynamicGGUFPlanOptions:
    recipe_key: str
    output_dir: Path
    profile_name: str | None = None
    policy_file: Path | None = None
    llama_cpp_dir: Path = Path("../llama.cpp")
    source_gguf: Path | None = None
    target_gguf: Path | None = None
    include_vision: bool = False
    include_language: bool = True
    use_converter_mapping: bool = True


QWEN36_DYNAMIC_PROFILES: dict[str, DynamicGGUFProfile] = {
    "OTQ-DYN-Q3_K_M": DynamicGGUFProfile(
        name="OTQ-DYN-Q3_K_M",
        base_ftype="Q3_K_M",
        target="32 GB Apple Silicon compact stock-GGUF release",
        requires_imatrix=False,
        category_types={
            "embeddings": "Q5_K",
            "lm_head": "Q6_K",
            "self_attn_proj": "Q4_K",
            "linear_attn_proj": "Q4_K",
            "linear_attn_conv": "F16",
            "mlp_proj": "Q3_K",
        },
        edge_overrides={
            "mlp_proj": "Q4_K",
            "linear_attn_proj": "Q4_K",
        },
        periodic_overrides={
            "self_attn_proj": "Q5_K",
        },
        notes="Aggressive stock-compatible profile: Q3 bulk MLP, Q4 attention, Q5/Q6 anchors.",
    ),
    "OTQ-DYN-Q4_K_M": DynamicGGUFProfile(
        name="OTQ-DYN-Q4_K_M",
        base_ftype="Q4_K_M",
        target="primary public stock-GGUF release for 32 GB / 48 GB Macs",
        requires_imatrix=False,
        category_types={
            "embeddings": "Q6_K",
            "lm_head": "Q8_0",
            "self_attn_proj": "Q5_K",
            "linear_attn_proj": "Q5_K",
            "linear_attn_conv": "F16",
            "mlp_proj": "Q4_K",
        },
        edge_overrides={
            "mlp_proj": "Q5_K",
            "linear_attn_proj": "Q6_K",
            "self_attn_proj": "Q6_K",
        },
        periodic_overrides={
            "self_attn_proj": "Q6_K",
            "linear_attn_proj": "Q6_K",
        },
        notes="Balanced stock-compatible flagship: Q4 bulk MLP, Q5/Q6 attention, Q8 output head.",
    ),
    "OTQ-DYN-Q5_K_M": DynamicGGUFProfile(
        name="OTQ-DYN-Q5_K_M",
        base_ftype="Q5_K_M",
        target="quality-first stock-GGUF release for 48 GB+ Macs",
        requires_imatrix=False,
        category_types={
            "embeddings": "Q8_0",
            "lm_head": "Q8_0",
            "self_attn_proj": "Q6_K",
            "linear_attn_proj": "Q6_K",
            "linear_attn_conv": "F16",
            "mlp_proj": "Q5_K",
        },
        edge_overrides={
            "mlp_proj": "Q6_K",
            "linear_attn_proj": "Q8_0",
            "self_attn_proj": "Q8_0",
        },
        periodic_overrides={
            "self_attn_proj": "Q8_0",
        },
        notes="High-fidelity stock-compatible profile for regression and high-end Apple Silicon.",
    ),
    "OTQ-DYN-IQ4_NL": DynamicGGUFProfile(
        name="OTQ-DYN-IQ4_NL",
        base_ftype="IQ4_NL",
        target="imatrix-driven nonlinear 4-bit stock-GGUF experiment",
        requires_imatrix=True,
        category_types={
            "embeddings": "Q6_K",
            "lm_head": "Q8_0",
            "self_attn_proj": "Q5_K",
            "linear_attn_proj": "Q5_K",
            "linear_attn_conv": "F16",
            "mlp_proj": "IQ4_NL",
        },
        edge_overrides={
            "mlp_proj": "Q5_K",
            "linear_attn_proj": "Q6_K",
            "self_attn_proj": "Q6_K",
        },
        periodic_overrides={
            "self_attn_proj": "Q6_K",
        },
        notes="Uses standard llama.cpp IQ4_NL for bulk MLP; should be calibrated with an imatrix before release.",
    ),
}


def normalize_profile_name(name: str) -> str:
    return name.upper().replace("-", "_")


def get_dynamic_profile(name: str) -> DynamicGGUFProfile:
    normalized = normalize_profile_name(name)
    for profile_name, profile in QWEN36_DYNAMIC_PROFILES.items():
        if normalize_profile_name(profile_name) == normalized:
            return profile
    available = ", ".join(sorted(QWEN36_DYNAMIC_PROFILES))
    raise KeyError(f"unknown dynamic GGUF profile {name!r}; available: {available}")


def _load_policy_payload(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")
    if suffix == ".json":
        payload = json.loads(text)
    elif suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - dependency is declared, this is a packaging guard
            raise RuntimeError("YAML policy files require PyYAML; install opentq with project dependencies") from exc
        payload = yaml.safe_load(text)
    else:
        raise ValueError(f"unsupported dynamic GGUF policy extension {suffix!r}; use .json, .yaml, or .yml")
    if not isinstance(payload, dict):
        raise ValueError(f"dynamic GGUF policy {path} must contain a JSON/YAML object")
    return payload


def _string_field(payload: dict[str, Any], key: str, errors: list[str], *, default: str | None = None) -> str:
    value = payload.get(key, default)
    if value is None:
        errors.append(f"missing required field {key!r}")
        return ""
    if not isinstance(value, str):
        errors.append(f"field {key!r} must be a string")
        return ""
    return value.strip()


def _bool_field(payload: dict[str, Any], key: str, errors: list[str], *, default: bool) -> bool:
    value = payload.get(key, default)
    if not isinstance(value, bool):
        errors.append(f"field {key!r} must be a boolean")
        return default
    return value


def _int_field(payload: dict[str, Any], key: str, errors: list[str], *, default: int) -> int:
    value = payload.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        errors.append(f"field {key!r} must be a non-negative integer")
        return default
    return value


def _policy_type_map(payload: dict[str, Any], key: str, errors: list[str], *, required: bool = False) -> dict[str, str]:
    value = payload.get(key)
    if value is None:
        if required:
            errors.append(f"missing required field {key!r}")
        return {}
    if not isinstance(value, dict):
        errors.append(f"field {key!r} must be an object mapping tensor categories to GGUF tensor types")
        return {}

    resolved: dict[str, str] = {}
    for category, ggml_type in value.items():
        category_ok = isinstance(category, str)
        ggml_type_ok = isinstance(ggml_type, str)
        category_name = category.strip() if category_ok else repr(category)
        tensor_type = ggml_type.strip().upper() if ggml_type_ok else repr(ggml_type)

        if not category_ok or category_name not in DYNAMIC_POLICY_CATEGORIES:
            allowed = ", ".join(sorted(DYNAMIC_POLICY_CATEGORIES))
            errors.append(f"{key}.{category_name} is not a known tensor category; allowed: {allowed}")
            category_ok = False
        if not ggml_type_ok or tensor_type not in GGML_TYPE_BPW:
            allowed = ", ".join(sorted(GGML_TYPE_BPW))
            errors.append(f"{key}.{category_name} uses unknown GGUF tensor type {tensor_type}; allowed: {allowed}")
            ggml_type_ok = False
        if category_ok and ggml_type_ok:
            resolved[category_name] = tensor_type
    return resolved


def load_dynamic_policy_file(path: str | Path) -> DynamicGGUFProfile:
    policy_path = Path(path)
    payload = _load_policy_payload(policy_path)
    errors: list[str] = []

    name = _string_field(payload, "name", errors)
    base_ftype = _string_field(payload, "base_ftype", errors).upper()
    target = _string_field(payload, "target", errors)
    notes = _string_field(payload, "notes", errors, default="")
    requires_imatrix = _bool_field(payload, "requires_imatrix", errors, default=False)
    edge_layers = _int_field(payload, "edge_layers", errors, default=2)
    periodic_stride = _int_field(payload, "periodic_stride", errors, default=4)
    category_types = _policy_type_map(payload, "category_types", errors, required=True)
    edge_overrides = _policy_type_map(payload, "edge_overrides", errors)
    periodic_overrides = _policy_type_map(payload, "periodic_overrides", errors)

    if base_ftype and base_ftype not in BASE_FTYPES:
        allowed = ", ".join(sorted(BASE_FTYPES))
        errors.append(f"base_ftype {base_ftype!r} is not supported by the stock GGUF planner; allowed: {allowed}")
    if not category_types:
        errors.append("category_types must define at least one tensor category")

    if errors:
        joined = "; ".join(errors)
        raise ValueError(f"invalid dynamic GGUF policy {policy_path}: {joined}")

    return DynamicGGUFProfile(
        name=name,
        base_ftype=base_ftype,
        target=target,
        requires_imatrix=requires_imatrix,
        category_types=category_types,
        edge_layers=edge_layers,
        edge_overrides=edge_overrides,
        periodic_stride=periodic_stride,
        periodic_overrides=periodic_overrides,
        notes=notes,
    )


def resolve_dynamic_profile_source(options: DynamicGGUFPlanOptions) -> tuple[DynamicGGUFProfile, dict[str, str]]:
    if (options.profile_name is None) == (options.policy_file is None):
        raise ValueError("dynamic GGUF planning requires exactly one of profile_name or policy_file")
    if options.policy_file is not None:
        profile = load_dynamic_policy_file(options.policy_file)
        return profile, {"kind": "policy_file", "path": str(options.policy_file)}
    assert options.profile_name is not None
    profile = get_dynamic_profile(options.profile_name)
    return profile, {"kind": "builtin_profile", "name": profile.name}


def dynamic_profiles_payload() -> list[dict[str, Any]]:
    return [
        {
            "name": profile.name,
            "base_ftype": profile.base_ftype,
            "target": profile.target,
            "requires_imatrix": profile.requires_imatrix,
            "category_types": profile.category_types,
            "edge_layers": profile.edge_layers,
            "edge_overrides": profile.edge_overrides,
            "periodic_stride": profile.periodic_stride,
            "periodic_overrides": profile.periodic_overrides,
            "notes": profile.notes,
        }
        for profile in sorted(QWEN36_DYNAMIC_PROFILES.values(), key=lambda item: item.name)
    ]


def layer_index_from_name(name: str) -> int | None:
    match = re.search(r"(?:^|\.)layers\.(\d+)\.", name)
    if match:
        return int(match.group(1))
    match = re.search(r"(?:^|\.)blk\.(\d+)\.", name)
    if match:
        return int(match.group(1))
    return None


def resolve_dynamic_tensor_action(
    profile: DynamicGGUFProfile,
    tensor_name: str,
    *,
    n_layers: int = 64,
    include_vision: bool = False,
    include_language: bool = True,
) -> DynamicTensorAction:
    if tensor_name.startswith("mtp."):
        return DynamicTensorAction("mtp", "skip", None, "MTP tensors are not exported in the text GGUF artifact")

    category = classify_qwen36_27b_tensor(tensor_name)
    is_visual = category.startswith("visual") or category in {"vision_tower", "mm_projector"}
    is_language = not is_visual
    if is_visual and not include_vision:
        return DynamicTensorAction(category, "skip", None, "text-only release skips vision tensors")
    if is_language and not include_language:
        return DynamicTensorAction(category, "skip", None, "language tensors disabled")
    if category in TEXT_SKIP_CATEGORIES and not include_vision:
        return DynamicTensorAction(category, "skip", None, "not part of the text GGUF artifact")

    layer_index = layer_index_from_name(tensor_name)
    ggml_type = profile.category_types.get(category)
    reason = "profile category allocation"

    if ggml_type is None and category in COPY_CATEGORIES:
        return DynamicTensorAction(category, "copy", "F16", "kept high precision or handled by llama.cpp fallback", layer_index)
    if ggml_type is None:
        return DynamicTensorAction(category, "copy", "F16", "unclassified tensor kept high precision", layer_index)

    if layer_index is not None and profile.edge_layers > 0:
        is_edge = layer_index < profile.edge_layers or layer_index >= max(n_layers - profile.edge_layers, 0)
        if is_edge and category in profile.edge_overrides:
            ggml_type = profile.edge_overrides[category]
            reason = f"edge-layer override for first/last {profile.edge_layers} layers"

    if layer_index is not None and profile.periodic_stride > 0 and category in profile.periodic_overrides:
        if (layer_index + 1) % profile.periodic_stride == 0:
            ggml_type = profile.periodic_overrides[category]
            reason = f"periodic anchor override every {profile.periodic_stride} layers"

    return DynamicTensorAction(category, "set_type", ggml_type, reason, layer_index)


def exact_regex_pattern(name: str) -> str:
    return "^" + re.escape(name.lower()) + "$"


def heuristic_gguf_name(hf_name: str) -> str | None:
    name = strip_text_name(hf_name)
    if name is None:
        return None
    name = name.replace("model.", "", 1) if name.startswith("model.") else name
    if name == "lm_head.weight":
        return "output.weight"
    if name == "embed_tokens.weight":
        return "token_embd.weight"
    if name == "norm.weight":
        return "output_norm.weight"

    patterns = (
        (r"layers\.(\d+)\.self_attn\.q_proj\.weight$", "blk.{layer}.attn_q.weight"),
        (r"layers\.(\d+)\.self_attn\.k_proj\.weight$", "blk.{layer}.attn_k.weight"),
        (r"layers\.(\d+)\.self_attn\.v_proj\.weight$", "blk.{layer}.attn_v.weight"),
        (r"layers\.(\d+)\.self_attn\.o_proj\.weight$", "blk.{layer}.attn_output.weight"),
        (r"layers\.(\d+)\.mlp\.gate_proj\.weight$", "blk.{layer}.ffn_gate.weight"),
        (r"layers\.(\d+)\.mlp\.up_proj\.weight$", "blk.{layer}.ffn_up.weight"),
        (r"layers\.(\d+)\.mlp\.down_proj\.weight$", "blk.{layer}.ffn_down.weight"),
        (r"layers\.(\d+)\.linear_attn\.in_proj_qkv\.weight$", "blk.{layer}.attn_qkv.weight"),
        (r"layers\.(\d+)\.linear_attn\.in_proj_z\.weight$", "blk.{layer}.attn_gate.weight"),
        (r"layers\.(\d+)\.linear_attn\.in_proj_a\.weight$", "blk.{layer}.ssm_alpha.weight"),
        (r"layers\.(\d+)\.linear_attn\.in_proj_b\.weight$", "blk.{layer}.ssm_beta.weight"),
        (r"layers\.(\d+)\.linear_attn\.out_proj\.weight$", "blk.{layer}.ssm_out.weight"),
        (r"layers\.(\d+)\.linear_attn\.conv1d\.weight$", "blk.{layer}.ssm_conv1d.weight"),
        (r"layers\.(\d+)\.linear_attn\.A_log$", "blk.{layer}.ssm_a"),
        (r"layers\.(\d+)\.linear_attn\.dt_bias$", "blk.{layer}.ssm_dt.bias"),
        (r"layers\.(\d+)\.linear_attn\.norm\.weight$", "blk.{layer}.ssm_norm.weight"),
        (r"layers\.(\d+)\.input_layernorm\.weight$", "blk.{layer}.attn_norm.weight"),
        (r"layers\.(\d+)\.post_attention_layernorm\.weight$", "blk.{layer}.ffn_norm.weight"),
    )
    for pattern, template in patterns:
        match = re.search(pattern, name)
        if match:
            return template.format(layer=match.group(1))
    return None


def build_converter_name_mapper(model_id: str, llama_cpp_dir: Path) -> Callable[[str], str | None]:
    converter = load_converter_module(llama_cpp_dir)
    snapshot = snapshot_metadata(model_id)
    config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    arch = config.get("architectures", [None])[0]
    if arch is None:
        raise ValueError("missing architecture in config.json")
    model_cls = converter.ModelBase.from_model_architecture(arch, converter.ModelType.TEXT)
    model = model_cls(
        snapshot,
        converter.gguf.LlamaFileType.MOSTLY_F16,
        Path(os.devnull),
        use_temp_file=False,
        remote_hf_model_id=model_id,
        model_name="opentq-dynamic-gguf-map",
    )

    def map_name(hf_name: str) -> str | None:
        stripped = strip_text_name(hf_name)
        if stripped is None:
            return None
        transform = qwen35_transform_kind(stripped)
        return mapped_text_name(model, stripped, transform)

    return map_name


def build_dynamic_tensor_plans(
    *,
    model_id: str,
    weight_map: dict[str, str],
    profile: DynamicGGUFProfile,
    include_vision: bool = False,
    include_language: bool = True,
    name_mapper: Callable[[str], str | None] | None = None,
) -> list[DynamicTensorPlan]:
    rows: list[DynamicTensorPlan] = []
    for hf_name in sorted(weight_map):
        action = resolve_dynamic_tensor_action(
            profile,
            hf_name,
            include_vision=include_vision,
            include_language=include_language,
        )
        gguf_name = None
        if action.mode != "skip":
            if name_mapper is not None:
                try:
                    gguf_name = name_mapper(hf_name)
                except Exception:
                    gguf_name = None
            if gguf_name is None:
                gguf_name = heuristic_gguf_name(hf_name)
        rows.append(
            DynamicTensorPlan(
                hf_name=hf_name,
                gguf_name=gguf_name,
                source_file=weight_map[hf_name],
                category=action.category,
                mode=action.mode,
                ggml_type=action.ggml_type,
                layer_index=action.layer_index,
                reason=action.reason,
            )
        )
    return rows


def tensor_type_lines(rows: list[DynamicTensorPlan]) -> list[str]:
    lines = []
    seen: set[str] = set()
    for row in rows:
        if row.mode == "skip" or row.gguf_name is None or row.ggml_type is None:
            continue
        if row.mode not in {"set_type", "copy"}:
            continue
        key = row.gguf_name.lower()
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"{exact_regex_pattern(row.gguf_name)}={row.ggml_type.lower()}")
    return lines


def build_summary(rows: list[DynamicTensorPlan]) -> dict[str, Any]:
    by_mode = Counter(row.mode for row in rows)
    by_type = Counter(row.ggml_type or "none" for row in rows if row.mode != "skip")
    by_category_type = Counter(f"{row.category}:{row.ggml_type or row.mode}" for row in rows)
    unmapped = [row.hf_name for row in rows if row.mode != "skip" and row.gguf_name is None]
    return {
        "tensor_count": len(rows),
        "by_mode": dict(sorted(by_mode.items())),
        "by_type": dict(sorted(by_type.items())),
        "by_category_type": dict(sorted(by_category_type.items())),
        "mapped_tensor_types": len(tensor_type_lines(rows)),
        "unmapped_count": len(unmapped),
        "unmapped_samples": unmapped[:20],
    }


def build_quantize_script(
    *,
    profile: DynamicGGUFProfile,
    llama_cpp_dir: Path,
    source_gguf: Path | None,
    target_gguf: Path | None,
) -> str:
    quantize_bin = llama_cpp_dir / "build" / "bin" / "llama-quantize"
    if source_gguf is None:
        source_assignment = ': "${SOURCE_GGUF:?set SOURCE_GGUF to the BF16/F16 source GGUF}"'
    else:
        source_assignment = f'SOURCE_GGUF="${{SOURCE_GGUF:-{shlex.quote(str(source_gguf))}}}"'
    if target_gguf is None:
        target_assignment = ': "${TARGET_GGUF:?set TARGET_GGUF to the output GGUF}"'
    else:
        target_assignment = f'TARGET_GGUF="${{TARGET_GGUF:-{shlex.quote(str(target_gguf))}}}"'
    return f"""#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
LLAMA_QUANTIZE="${{LLAMA_QUANTIZE:-{shlex.quote(str(quantize_bin))}}}"
{source_assignment}
{target_assignment}
TENSOR_TYPES="${{TENSOR_TYPES:-$SCRIPT_DIR/tensor-types.txt}}"

args=()
if [[ "${{ALLOW_REQUANTIZE:-0}}" == "1" ]]; then
  args+=(--allow-requantize)
fi
if [[ -n "${{IMATRIX:-}}" ]]; then
  args+=(--imatrix "$IMATRIX")
fi
if [[ "${{DRY_RUN:-0}}" == "1" ]]; then
  args+=(--dry-run)
fi

mkdir -p "$(dirname "$TARGET_GGUF")"
args+=(--tensor-type-file "$TENSOR_TYPES" "$SOURCE_GGUF" "$TARGET_GGUF" {shlex.quote(profile.base_ftype)})
if [[ -n "${{NTHREADS:-}}" ]]; then
  args+=("$NTHREADS")
fi

exec "$LLAMA_QUANTIZE" "${{args[@]}}"
"""


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def make_executable(path: Path) -> None:
    current = path.stat().st_mode
    path.chmod(current | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def write_dynamic_gguf_plan(options: DynamicGGUFPlanOptions) -> dict[str, Any]:
    recipe = get_recipe(options.recipe_key)
    profile, policy_source = resolve_dynamic_profile_source(options)
    index_data = fetch_safetensors_index(recipe.model_id)
    weight_map = index_data["weight_map"]

    name_mapper = None
    mapping_error = None
    if options.use_converter_mapping:
        try:
            name_mapper = build_converter_name_mapper(recipe.model_id, options.llama_cpp_dir)
        except Exception as exc:  # pragma: no cover - exercised only when local converter/env is unavailable
            mapping_error = repr(exc)

    rows = build_dynamic_tensor_plans(
        model_id=recipe.model_id,
        weight_map=weight_map,
        profile=profile,
        include_vision=options.include_vision,
        include_language=options.include_language,
        name_mapper=name_mapper,
    )
    tensor_lines = tensor_type_lines(rows)
    output_dir = options.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    tensor_types_path = output_dir / "tensor-types.txt"
    annotated_path = output_dir / "tensor-types.annotated.tsv"
    plan_path = output_dir / "plan.json"
    script_path = output_dir / "quantize.sh"

    write_text(tensor_types_path, "\n".join(tensor_lines) + ("\n" if tensor_lines else ""))
    write_text(
        annotated_path,
        "gguf_name\tggml_type\tcategory\tlayer\tmode\treason\thf_name\n"
        + "\n".join(
            "\t".join(
                [
                    row.gguf_name or "",
                    row.ggml_type or "",
                    row.category,
                    "" if row.layer_index is None else str(row.layer_index),
                    row.mode,
                    row.reason,
                    row.hf_name,
                ]
            )
            for row in rows
            if row.mode != "skip"
        )
        + "\n",
    )
    write_text(
        script_path,
        build_quantize_script(
            profile=profile,
            llama_cpp_dir=options.llama_cpp_dir,
            source_gguf=options.source_gguf,
            target_gguf=options.target_gguf,
        ),
    )
    make_executable(script_path)

    payload = {
        "schema": "opentq.dynamic_gguf_plan.v1",
        "model_id": recipe.model_id,
        "recipe_key": recipe.key,
        "policy_source": policy_source,
        "profile": asdict(profile),
        "compatibility": {
            "gguf_tensor_types": "standard llama.cpp GGML types only",
            "requires_custom_llama_cpp": False,
            "requires_opentq_runtime": False,
            "requires_imatrix": profile.requires_imatrix,
        },
        "source_gguf": None if options.source_gguf is None else str(options.source_gguf),
        "target_gguf": None if options.target_gguf is None else str(options.target_gguf),
        "llama_cpp_dir": str(options.llama_cpp_dir),
        "include_vision": options.include_vision,
        "include_language": options.include_language,
        "mapping_error": mapping_error,
        "outputs": {
            "plan": str(plan_path),
            "tensor_type_file": str(tensor_types_path),
            "annotated_tensor_type_file": str(annotated_path),
            "quantize_script": str(script_path),
        },
        "summary": build_summary(rows),
        "tensors": [asdict(row) for row in rows],
    }
    write_text(plan_path, json.dumps(payload, indent=2) + "\n")
    return payload
