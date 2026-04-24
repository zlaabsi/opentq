from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .hf import base_weight_size_gib, fetch_safetensors_index
from .inventory import build_inventory, inventory_summary
from .quantize import quantize_tensor
from .recipes import get_recipe, recipe_markdown, recipe_to_dict
from .variants import VARIANTS, get_variant


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="opentq")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("variants", help="List built-in quantization variants.")

    plan = sub.add_parser("plan", help="Estimate bit budget and storage for a variant.")
    plan.add_argument("variant")
    plan.add_argument("--shape", nargs="+", type=int, required=True)

    quantize = sub.add_parser("quantize", help="Quantize a .npy tensor and emit a manifest.")
    quantize.add_argument("tensor")
    quantize.add_argument("--variant", required=True)
    quantize.add_argument("--output", required=True)
    quantize.add_argument("--seed", type=int, default=42)

    recipe = sub.add_parser("recipe", help="Show the release matrix and work phases for a model recipe.")
    recipe.add_argument("key")
    recipe.add_argument("--format", choices=("json", "markdown"), default="json")

    inventory = sub.add_parser("inventory", help="Inspect the safetensors index of a Hugging Face model.")
    inventory.add_argument("--model-id", default="Qwen/Qwen3.6-27B")
    inventory.add_argument("--sample-limit", type=int, default=3)

    return parser


def cmd_variants() -> int:
    rows = []
    for name, variant in sorted(VARIANTS.items()):
        rows.append(
            {
                "name": name,
                "total_bits": variant.total_bits,
                "estimated_bpw": round(variant.estimated_bpw(), 2),
                "intended_use": variant.intended_use,
                "notes": variant.notes,
            }
        )
    print(json.dumps(rows, indent=2))
    return 0


def cmd_plan(variant_name: str, shape: list[int]) -> int:
    variant = get_variant(variant_name)
    num_values = int(np.prod(shape))
    estimated_bits = int(num_values * variant.estimated_bpw())
    plan = {
        "variant": variant.name,
        "shape": shape,
        "values": num_values,
        "estimated_bpw": variant.estimated_bpw(),
        "estimated_bytes": estimated_bits // 8,
        "estimated_mib": round((estimated_bits / 8) / (1024 * 1024), 2),
        "runtime_targets": variant.runtime_targets,
    }
    print(json.dumps(plan, indent=2))
    return 0


def cmd_quantize(tensor_path: str, variant_name: str, output: str, seed: int) -> int:
    variant = get_variant(variant_name)
    data = np.load(tensor_path)
    result = quantize_tensor(data, variant, seed=seed)

    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    np.save(output_path / "reconstruction.npy", result.reconstruction)
    np.savez_compressed(
        output_path / "quantized_blocks.npz",
        indices=np.stack([block.indices for block in result.packed.blocks]),
        scales=np.stack([block.scales for block in result.packed.blocks]),
        residual_indices=np.stack([block.residual_indices for block in result.packed.blocks if block.residual_indices is not None])
        if any(block.residual_indices is not None for block in result.packed.blocks)
        else np.array([], dtype=np.uint8),
    )
    manifest_path = output_path / "manifest.json"
    manifest_path.write_text(json.dumps(result.to_manifest(), indent=2) + "\n", encoding="utf-8")
    print(manifest_path)
    return 0


def cmd_recipe(key: str, output_format: str) -> int:
    recipe = get_recipe(key)
    if output_format == "json":
        print(json.dumps(recipe_to_dict(recipe), indent=2))
        return 0
    if output_format == "markdown":
        print(recipe_markdown(recipe))
        return 0
    raise ValueError(f"unsupported format: {output_format}")


def cmd_inventory(model_id: str, sample_limit: int) -> int:
    index_data = fetch_safetensors_index(model_id)
    weight_map = index_data["weight_map"]
    payload = {
        "model_id": model_id,
        "base_weight_size_gib": round(base_weight_size_gib(index_data), 2),
        "num_tensors": len(weight_map),
        "summary": inventory_summary(weight_map),
        "inventory": [
            {
                "category": row.category,
                "count": row.count,
                "samples": list(row.samples),
            }
            for row in build_inventory(weight_map, sample_limit=sample_limit)
        ],
    }
    print(json.dumps(payload, indent=2))
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "variants":
        return cmd_variants()
    if args.command == "plan":
        return cmd_plan(args.variant, args.shape)
    if args.command == "quantize":
        return cmd_quantize(args.tensor, args.variant, args.output, args.seed)
    if args.command == "recipe":
        return cmd_recipe(args.key, args.format)
    if args.command == "inventory":
        return cmd_inventory(args.model_id, args.sample_limit)
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
