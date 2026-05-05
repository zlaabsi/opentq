from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .dynamic_gguf import GGML_TYPE_BPW


HIGH_PRECISION_TYPES = {"F16", "BF16", "Q8_0", "Q6_K"}
PRUNABLE_CATEGORIES = {"mlp_proj", "self_attn_proj", "linear_attn_proj"}


@dataclass(frozen=True)
class PruningCandidateOptions:
    plan_path: Path
    output_dir: Path
    max_candidates: int = 256
    prune_threshold: float = 0.78
    aggressive_threshold: float = 0.56


def _load_plan(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _precision_score(ggml_type: str | None) -> float:
    if ggml_type is None:
        return 0.0
    bpw = GGML_TYPE_BPW.get(ggml_type.upper(), 16.0)
    # Lower assigned precision means the existing allocation policy already
    # treats this family as less fragile.
    return max(0.0, min(1.0, (8.5 - bpw) / 5.5))


def _category_score(category: str) -> float:
    if category == "mlp_proj":
        return 0.24
    if category == "linear_attn_proj":
        return 0.12
    if category == "self_attn_proj":
        return 0.04
    return 0.0


def _layer_score(layer: int | None, num_layers: int) -> float:
    if layer is None:
        return 0.0
    edge_distance = min(layer, max(num_layers - layer - 1, 0))
    if edge_distance <= 1:
        return -0.35
    if edge_distance <= 3:
        return -0.12
    return 0.08


def _reason_penalty(reason: str, ggml_type: str | None) -> float:
    penalty = 0.0
    if reason.startswith("edge-layer override"):
        penalty -= 0.32
    if reason.startswith("periodic anchor override"):
        penalty -= 0.18
    if ggml_type and ggml_type.upper() in HIGH_PRECISION_TYPES:
        penalty -= 0.16
    return penalty


def _action(score: float, options: PruningCandidateOptions) -> str:
    if score >= options.prune_threshold:
        return "prune_candidate"
    if score >= options.aggressive_threshold:
        return "quantize_aggressive"
    if score >= 0.30:
        return "quantize_standard"
    return "keep_high_precision"


def build_pruning_candidates(options: PruningCandidateOptions) -> dict[str, Any]:
    plan = _load_plan(options.plan_path)
    tensors = [row for row in plan.get("tensors", []) if row.get("mode") != "skip"]
    layer_indexes = [row["layer_index"] for row in tensors if isinstance(row.get("layer_index"), int)]
    num_layers = max(layer_indexes) + 1 if layer_indexes else 0

    grouped: dict[tuple[int | None, str], list[dict[str, Any]]] = defaultdict(list)
    for row in tensors:
        category = str(row.get("category") or "")
        if category not in PRUNABLE_CATEGORIES:
            continue
        grouped[(row.get("layer_index"), category)].append(row)

    candidates: list[dict[str, Any]] = []
    for (layer, category), rows in grouped.items():
        precision = sum(_precision_score(row.get("ggml_type")) for row in rows) / max(len(rows), 1)
        reason_penalty = sum(_reason_penalty(str(row.get("reason") or ""), row.get("ggml_type")) for row in rows) / max(len(rows), 1)
        score = max(0.0, min(1.0, precision + _category_score(category) + _layer_score(layer, num_layers) + reason_penalty))
        action = _action(score, options)
        candidates.append(
            {
                "unit_id": f"layer.{layer}.{category}" if layer is not None else category,
                "layer": layer,
                "category": category,
                "action": action,
                "score": round(score, 4),
                "tensor_count": len(rows),
                "assigned_types": sorted({str(row.get("ggml_type")) for row in rows}),
                "sample_tensors": [str(row.get("hf_name")) for row in rows[:4]],
                "rationale": _candidate_rationale(action, category, layer, rows),
            }
        )

    action_order = {
        "prune_candidate": 0,
        "quantize_aggressive": 1,
        "quantize_standard": 2,
        "keep_high_precision": 3,
    }
    candidates.sort(key=lambda row: (action_order[row["action"]], -float(row["score"]), row["unit_id"]))
    candidates = candidates[: options.max_candidates]

    counts: dict[str, int] = {}
    for row in candidates:
        counts[row["action"]] = counts.get(row["action"], 0) + 1

    return {
        "schema": "opentq.quantization_aware_pruning_candidates.v1",
        "source_plan": str(options.plan_path),
        "model_id": plan.get("model_id"),
        "profile": plan.get("profile", {}).get("name"),
        "num_layers": num_layers,
        "policy": {
            "prune_threshold": options.prune_threshold,
            "aggressive_threshold": options.aggressive_threshold,
            "release_rule": "offline candidates only; no public pruned artifact without paired validation",
        },
        "summary": {
            "candidate_count": len(candidates),
            "action_counts": dict(sorted(counts.items())),
        },
        "candidates": candidates,
    }


def _candidate_rationale(action: str, category: str, layer: int | None, rows: list[dict[str, Any]]) -> str:
    types = sorted({str(row.get("ggml_type")) for row in rows})
    location = f"layer {layer}" if layer is not None else "global"
    if action == "prune_candidate":
        return f"{location} {category} is middle-layer, projection-heavy, and already assigned low precision ({', '.join(types)}). Validate before pruning."
    if action == "quantize_aggressive":
        return f"{location} {category} is a lower-risk compression target before structural pruning."
    if action == "quantize_standard":
        return f"{location} {category} should follow the current allocation policy unless paired metrics improve."
    return f"{location} {category} is preserved because the policy or layer position suggests sensitivity."


def write_pruning_candidates(options: PruningCandidateOptions) -> dict[str, Any]:
    payload = build_pruning_candidates(options)
    output_dir = options.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "pruning-candidates.json"
    jsonl_path = output_dir / "pruning-candidates.jsonl"
    policy_path = output_dir / "pruning-policy.yaml"
    report_path = output_dir / "paired-pruning-report.md"

    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in payload["candidates"]:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    policy_lines = [
        "schema: opentq.pruning_policy.v1",
        f"source_plan: {payload['source_plan']}",
        "release_rule: offline candidates only; paired validation required before publishing pruned artifacts",
        "units:",
    ]
    for row in payload["candidates"][:64]:
        policy_lines.extend(
            [
                f"  - unit_id: {row['unit_id']}",
                f"    action: {row['action']}",
                f"    score: {row['score']}",
                f"    category: {row['category']}",
                f"    layer: {row['layer']}",
            ]
        )
    policy_path.write_text("\n".join(policy_lines) + "\n", encoding="utf-8")

    report_lines = [
        "# Quantization-Aware Pruning Candidates",
        "",
        f"- Source plan: `{payload['source_plan']}`",
        f"- Profile: `{payload['profile']}`",
        f"- Candidate count: `{payload['summary']['candidate_count']}`",
        f"- Action counts: `{payload['summary']['action_counts']}`",
        "",
        "These are offline candidates. They are not a release claim and they do not modify model weights yet.",
        "",
        "| Unit | Action | Score | Assigned types |",
        "| --- | --- | ---: | --- |",
    ]
    for row in payload["candidates"][:24]:
        report_lines.append(
            f"| `{row['unit_id']}` | `{row['action']}` | {row['score']} | `{', '.join(row['assigned_types'])}` |"
        )
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    payload["outputs"] = {
        "json": str(json_path),
        "jsonl": str(jsonl_path),
        "policy": str(policy_path),
        "report": str(report_path),
    }
    return payload
