from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .hf import base_weight_size_gib, fetch_safetensors_index
from .inventory import build_inventory, inventory_summary
from .quantize import quantize_tensor
from .recipes import get_recipe, recipe_markdown, recipe_to_dict
from .run import build_release_plan, quantize_release
from .status import build_status_payload, print_status, watch_status
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

    release_plan = sub.add_parser("release-plan", help="Build the tensor-level plan for a model release.")
    release_plan.add_argument("--recipe", required=True)
    release_plan.add_argument("--release", required=True)
    release_plan.add_argument("--language-only", action="store_true")
    release_plan.add_argument("--vision-only", action="store_true")

    quantize_release_parser = sub.add_parser("quantize-release", help="Run a full HF safetensors -> OpenTQ quantization.")
    quantize_release_parser.add_argument("--recipe", required=True)
    quantize_release_parser.add_argument("--release", required=True)
    quantize_release_parser.add_argument("--output", required=True)
    quantize_release_parser.add_argument("--max-tensors", type=int)
    quantize_release_parser.add_argument("--only-shard")
    quantize_release_parser.add_argument("--language-only", action="store_true")
    quantize_release_parser.add_argument("--vision-only", action="store_true")
    quantize_release_parser.add_argument("--no-skip-existing", action="store_true")

    status = sub.add_parser("status", help="Inspect or watch the overnight Qwen3.6-27B quantization batch.")
    status.add_argument("--root", default="artifacts/qwen3.6-27b")
    status.add_argument("--watch", action="store_true")
    status.add_argument("--interval", type=float, default=10.0)

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
    if result.reconstruction is not None:
        np.save(output_path / "reconstruction.npy", result.reconstruction)
    np.savez_compressed(
        output_path / "quantized_blocks.npz",
        indices=result.packed.indices,
        scales=result.packed.scales,
        residual_indices=result.packed.residual_indices if result.packed.residual_indices is not None else np.array([], dtype=np.uint8),
        residual_scales=result.packed.residual_scales if result.packed.residual_scales is not None else np.array([], dtype=np.float32),
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


def cmd_release_plan(recipe_key: str, release_slug: str, language_only: bool, vision_only: bool) -> int:
    payload = build_release_plan(
        recipe_key,
        release_slug,
        include_vision=not language_only,
        include_language=not vision_only,
    )
    print(json.dumps(payload, indent=2))
    return 0


def cmd_quantize_release(
    recipe_key: str,
    release_slug: str,
    output: str,
    max_tensors: int | None,
    only_shard: str | None,
    language_only: bool,
    vision_only: bool,
    no_skip_existing: bool,
) -> int:
    payload = quantize_release(
        recipe_key,
        release_slug,
        output,
        include_vision=not language_only,
        include_language=not vision_only,
        max_tensors=max_tensors,
        only_shard=only_shard,
        skip_existing=not no_skip_existing,
    )
    print(json.dumps(payload, indent=2))
    return 0


def cmd_status(root: str, watch: bool, interval: float) -> int:
    if watch:
        return watch_status(root=root, interval=interval)
    print_status(build_status_payload(root))
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
    if args.command == "release-plan":
        return cmd_release_plan(args.recipe, args.release, args.language_only, args.vision_only)
    if args.command == "quantize-release":
        return cmd_quantize_release(
            args.recipe,
            args.release,
            args.output,
            args.max_tensors,
            args.only_shard,
            args.language_only,
            args.vision_only,
            args.no_skip_existing,
        )
    if args.command == "status":
        return cmd_status(args.root, args.watch, args.interval)
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
