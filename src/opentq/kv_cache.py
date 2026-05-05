from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .dynamic_gguf import layer_index_from_name


KV_DTYPE_BITS = {
    "bf16": 16.0,
    "fp16": 16.0,
    "fp8": 8.0,
    "fp8_e4m3": 8.0,
    "fp8_e5m2": 8.0,
    "int8": 8.0,
    "int4": 4.0,
}


@dataclass(frozen=True)
class KVCachePlanOptions:
    output_dir: Path
    model_id: str = "Qwen/Qwen3.6-27B"
    num_layers: int = 64
    default_dtype: str = "fp8_e4m3"
    promote_dtype: str = "bf16"
    edge_layers: int = 2
    periodic_stride: int = 8
    weight_plan: Path | None = None


def _validate_dtype(name: str) -> str:
    dtype = name.strip().lower()
    if dtype not in KV_DTYPE_BITS:
        allowed = ", ".join(sorted(KV_DTYPE_BITS))
        raise ValueError(f"unsupported KV cache dtype {name!r}; allowed: {allowed}")
    return dtype


def _load_weight_plan(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _infer_num_layers(weight_plan: dict[str, Any] | None, fallback: int) -> int:
    if not weight_plan:
        return fallback
    layer_indexes = []
    for row in weight_plan.get("tensors", []):
        layer = row.get("layer_index")
        if isinstance(layer, int):
            layer_indexes.append(layer)
            continue
        name = str(row.get("hf_name") or row.get("gguf_name") or "")
        parsed = layer_index_from_name(name)
        if parsed is not None:
            layer_indexes.append(parsed)
    if not layer_indexes:
        return fallback
    return max(layer_indexes) + 1


def _weight_policy_sensitive_layers(weight_plan: dict[str, Any] | None) -> dict[int, set[str]]:
    sensitive: dict[int, set[str]] = {}
    if not weight_plan:
        return sensitive
    for row in weight_plan.get("tensors", []):
        if row.get("mode") == "skip":
            continue
        category = str(row.get("category") or "")
        if category not in {"self_attn_proj", "linear_attn_proj"}:
            continue
        reason = str(row.get("reason") or "")
        if not (reason.startswith("edge-layer override") or reason.startswith("periodic anchor override")):
            continue
        layer = row.get("layer_index")
        if not isinstance(layer, int):
            continue
        sensitive.setdefault(layer, set()).add(f"coupled to weight policy: {category} {reason}")
    return sensitive


def build_kv_cache_policy(options: KVCachePlanOptions) -> dict[str, Any]:
    default_dtype = _validate_dtype(options.default_dtype)
    promote_dtype = _validate_dtype(options.promote_dtype)
    if options.edge_layers < 0:
        raise ValueError("edge_layers must be non-negative")
    if options.periodic_stride < 0:
        raise ValueError("periodic_stride must be non-negative")

    weight_plan = _load_weight_plan(options.weight_plan)
    num_layers = _infer_num_layers(weight_plan, options.num_layers)
    sensitive_layers = _weight_policy_sensitive_layers(weight_plan)

    layers: list[dict[str, Any]] = []
    for layer in range(num_layers):
        reasons: list[str] = []
        dtype = default_dtype
        if options.edge_layers and (layer < options.edge_layers or layer >= max(num_layers - options.edge_layers, 0)):
            dtype = promote_dtype
            reasons.append(f"edge layer preserved in {promote_dtype}")
        if options.periodic_stride and (layer + 1) % options.periodic_stride == 0:
            dtype = promote_dtype
            reasons.append(f"periodic KV anchor every {options.periodic_stride} layers")
        if layer in sensitive_layers:
            dtype = promote_dtype
            reasons.extend(sorted(sensitive_layers[layer]))
        if not reasons:
            reasons.append(f"default runtime KV cache dtype {default_dtype}")

        layers.append(
            {
                "layer": layer,
                "key_dtype": dtype,
                "value_dtype": dtype,
                "estimated_bits_per_kv_value": KV_DTYPE_BITS[dtype],
                "reason": "; ".join(reasons),
            }
        )

    dtype_counts = Counter(row["key_dtype"] for row in layers)
    promoted = [row["layer"] for row in layers if row["key_dtype"] != default_dtype]
    payload = {
        "schema": "opentq.kv_cache_policy.v1",
        "model_id": options.model_id,
        "source_weight_plan": str(options.weight_plan) if options.weight_plan else None,
        "policy": {
            "default_dtype": default_dtype,
            "promote_dtype": promote_dtype,
            "edge_layers": options.edge_layers,
            "periodic_stride": options.periodic_stride,
        },
        "summary": {
            "num_layers": num_layers,
            "dtype_counts": dict(sorted(dtype_counts.items())),
            "promoted_layer_count": len(promoted),
            "promoted_layers": promoted,
            "average_bits_per_kv_value": round(
                sum(float(row["estimated_bits_per_kv_value"]) for row in layers) / max(len(layers), 1),
                3,
            ),
        },
        "runtime_targets": {
            "vllm": {
                "status": "adapter_plan",
                "kv_cache_dtype": "fp8" if default_dtype.startswith("fp8") else default_dtype,
                "kv_cache_dtype_skip_layers": promoted,
                "note": "Use equivalent runtime support when available; validate quality with paired long-context prompts.",
            },
            "opentq_metal_native": {
                "status": "target_policy",
                "note": "Metal kernels should consume this layer plan when mixed-precision KV cache support lands.",
            },
        },
        "layers": layers,
    }
    return payload


def write_kv_cache_policy(options: KVCachePlanOptions) -> dict[str, Any]:
    payload = build_kv_cache_policy(options)
    output_dir = options.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "kv-cache-policy.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    tsv_lines = ["layer\tkey_dtype\tvalue_dtype\testimated_bits_per_kv_value\treason"]
    for row in payload["layers"]:
        tsv_lines.append(
            "\t".join(
                [
                    str(row["layer"]),
                    row["key_dtype"],
                    row["value_dtype"],
                    str(row["estimated_bits_per_kv_value"]),
                    row["reason"],
                ]
            )
        )
    (output_dir / "kv-cache-policy.tsv").write_text("\n".join(tsv_lines) + "\n", encoding="utf-8")

    summary = payload["summary"]
    rationale = [
        "# OpenTQ KV Cache Layer Policy",
        "",
        f"- Model: `{payload['model_id']}`",
        f"- Default dtype: `{payload['policy']['default_dtype']}`",
        f"- Promoted dtype: `{payload['policy']['promote_dtype']}`",
        f"- Layers: `{summary['num_layers']}`",
        f"- Promoted layers: `{summary['promoted_layer_count']}`",
        f"- Average bits per KV value: `{summary['average_bits_per_kv_value']}`",
        "",
        "This artifact is runtime-facing policy evidence. It does not claim quality preservation until paired long-context runs pass with the same prompts and scoring rules.",
        "",
        "## Runtime Handoff",
        "",
        "- vLLM-style runtimes can treat `promoted_layers` as BF16/FP16 skip layers while using FP8 for the default layers.",
        "- OpenTQ Metal-native should consume the same layer table when mixed-precision KV kernels are wired in.",
    ]
    (output_dir / "kv-cache-rationale.md").write_text("\n".join(rationale) + "\n", encoding="utf-8")

    payload["outputs"] = {
        "json": str(json_path),
        "tsv": str(output_dir / "kv-cache-policy.tsv"),
        "rationale": str(output_dir / "kv-cache-rationale.md"),
    }
    return payload
