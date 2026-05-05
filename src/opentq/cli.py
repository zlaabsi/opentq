from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .allocation_ui import AllocationUIOptions, write_allocation_ui
from .hf import base_weight_size_gib, fetch_safetensors_index
from .gguf import write_gguf_plan
from .gguf_export import GGUFExportOptions, export_gguf
from .gguf_validate import GGUFValidationOptions, validate_gguf
from .dynamic_gguf import DynamicGGUFPlanOptions, dynamic_profiles_payload, write_dynamic_gguf_plan
from .hf_gguf_release import prepare_hf_gguf_release
from .hf_release import prepare_hf_release
from .inventory import build_inventory, inventory_summary
from .kv_cache import KVCachePlanOptions, write_kv_cache_policy
from .monitor import build_monitor_payload, print_monitor, watch_monitor
from .pruning import PruningCandidateOptions, write_pruning_candidates
from .quantize import quantize_tensor
from .quality_eval import GGUFQualityEvalOptions, run_quality_eval
from .release_pack import pack_release
from .recipes import get_recipe, recipe_markdown, recipe_to_dict
from .run import build_release_plan, quantize_release
from .runtime_gate import PackAuditOptions, RuntimeProbeOptions, audit_packed_runtime, run_runtime_probe, write_runtime_fixtures
from .status import build_status_payload, print_status, watch_status
from .variants import VARIANTS, get_variant


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="opentq")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("variants", help="List built-in quantization variants.")

    sub.add_parser("dynamic-gguf-profiles", help="List stock-compatible dynamic GGUF profiles.")

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

    status = sub.add_parser("status", help="Inspect or watch a Qwen3.6-27B quantization batch.")
    status.add_argument("--root", default="artifacts/qwen3.6-27b")
    status.add_argument("--watch", action="store_true")
    status.add_argument("--interval", type=float, default=10.0)

    monitor = sub.add_parser("monitor", help="Render a dense terminal monitor for the active quantization run.")
    monitor.add_argument("--root", default="artifacts/qwen3.6-27b")
    monitor.add_argument("--watch", action="store_true")
    monitor.add_argument("--interval", type=float, default=5.0)

    pack_release_parser = sub.add_parser("pack-release", help="Bit-pack a completed OpenTQ release directory.")
    pack_release_parser.add_argument("--input", required=True)
    pack_release_parser.add_argument("--output", required=True)
    pack_release_parser.add_argument("--force", action="store_true")
    pack_release_parser.add_argument("--max-tensors", type=int)
    pack_release_parser.add_argument("--copy-dtype", default="float16")

    hf_release_parser = sub.add_parser("prepare-hf", help="Create a Hugging Face staging folder from a packed OpenTQ release.")
    hf_release_parser.add_argument("--packed", required=True)
    hf_release_parser.add_argument("--output", required=True)
    hf_release_parser.add_argument("--repo-id", required=True)
    hf_release_parser.add_argument("--link-mode", choices=("hardlink", "copy", "symlink", "none"), default="hardlink")

    audit_pack_runtime = sub.add_parser("audit-pack-runtime", help="Validate an OpenTQ packed release without running inference.")
    audit_pack_runtime.add_argument("--packed", required=True)
    audit_pack_runtime.add_argument("--output")
    audit_pack_runtime.add_argument("--max-tensors", type=int)
    audit_pack_runtime.add_argument("--dequantize-samples", type=int, default=4)

    runtime_fixtures = sub.add_parser("runtime-fixtures", help="Emit OpenTQ block fixtures for external runtime probes.")
    runtime_fixtures.add_argument("--packed", required=True)
    runtime_fixtures.add_argument("--output", required=True)
    runtime_fixtures.add_argument("--max-fixtures-per-variant", type=int, default=1)

    probe_pack_runtime = sub.add_parser("probe-pack-runtime", help="Run packed-release fixtures against an external OpenTQ dequant probe.")
    probe_pack_runtime.add_argument("--packed", required=True)
    probe_pack_runtime.add_argument("--fixtures-output", required=True)
    probe_pack_runtime.add_argument("--probe-binary", required=True)
    probe_pack_runtime.add_argument("--output", required=True)
    probe_pack_runtime.add_argument("--audit-max-tensors", type=int)
    probe_pack_runtime.add_argument("--audit-dequantize-samples", type=int, default=4)
    probe_pack_runtime.add_argument("--max-fixtures-per-variant", type=int, default=1)
    probe_pack_runtime.add_argument("--timeout", type=float, default=120.0)

    gguf_plan = sub.add_parser("gguf-plan", help="Write the GGUF integration plan for a packed OpenTQ release.")
    gguf_plan.add_argument("--packed", required=True)
    gguf_plan.add_argument("--output", required=True)

    gguf_export = sub.add_parser("export-gguf", help="Export a packed OpenTQ release to a llama.cpp GGUF container.")
    gguf_export.add_argument("--packed", required=True)
    gguf_export.add_argument("--output", required=True)
    gguf_export.add_argument("--llama-cpp", default="../llama.cpp")
    gguf_export.add_argument("--max-tensors", type=int)
    gguf_export.add_argument("--include-vision", action="store_true")

    dynamic_gguf = sub.add_parser("dynamic-gguf-plan", help="Create a stock llama.cpp GGUF dynamic allocation plan.")
    dynamic_gguf.add_argument("--recipe", default="qwen3.6-27b")
    dynamic_gguf_source = dynamic_gguf.add_mutually_exclusive_group(required=True)
    dynamic_gguf_source.add_argument("--profile")
    dynamic_gguf_source.add_argument("--policy-file")
    dynamic_gguf.add_argument("--output", required=True)
    dynamic_gguf.add_argument("--llama-cpp", default="../llama.cpp")
    dynamic_gguf.add_argument("--source-gguf")
    dynamic_gguf.add_argument("--target-gguf")
    dynamic_gguf.add_argument("--include-vision", action="store_true")
    dynamic_gguf.add_argument("--vision-only", action="store_true")
    dynamic_gguf.add_argument("--no-converter-mapping", action="store_true")

    kv_cache_plan = sub.add_parser("kv-cache-plan", help="Write a per-layer KV cache precision policy.")
    kv_cache_plan.add_argument("--output", required=True)
    kv_cache_plan.add_argument("--model-id", default="Qwen/Qwen3.6-27B")
    kv_cache_plan.add_argument("--num-layers", type=int, default=64)
    kv_cache_plan.add_argument("--default-dtype", default="fp8_e4m3")
    kv_cache_plan.add_argument("--promote-dtype", default="bf16")
    kv_cache_plan.add_argument("--edge-layers", type=int, default=2)
    kv_cache_plan.add_argument("--periodic-stride", type=int, default=8)
    kv_cache_plan.add_argument("--weight-plan")

    pruning_candidates = sub.add_parser("pruning-candidates", help="Rank quantization-aware structured pruning candidates.")
    pruning_candidates.add_argument("--plan", required=True)
    pruning_candidates.add_argument("--output", required=True)
    pruning_candidates.add_argument("--max-candidates", type=int, default=256)
    pruning_candidates.add_argument("--prune-threshold", type=float, default=0.78)
    pruning_candidates.add_argument("--aggressive-threshold", type=float, default=0.56)

    allocation_ui = sub.add_parser("allocation-ui", help="Generate an inspectable tensor allocation dashboard artifact.")
    allocation_ui.add_argument("--plan", required=True)
    allocation_ui.add_argument("--output", required=True)
    allocation_ui.add_argument("--title", default="OpenTQ Allocation Explorer")
    allocation_ui.add_argument("--metrics")

    hf_gguf_release_parser = sub.add_parser("prepare-hf-gguf", help="Create a Hugging Face staging folder from a GGUF artifact.")
    hf_gguf_release_parser.add_argument("--gguf", required=True)
    hf_gguf_release_parser.add_argument("--output", required=True)
    hf_gguf_release_parser.add_argument("--repo-id", required=True)
    hf_gguf_release_parser.add_argument("--base-model", default="Qwen/Qwen3.6-27B")
    hf_gguf_release_parser.add_argument("--runtime-repo", default="https://github.com/zlaabsi/llama.cpp-opentq")
    hf_gguf_release_parser.add_argument("--link-mode", choices=("hardlink", "copy", "symlink"), default="hardlink")
    hf_gguf_release_parser.add_argument("--include-vision", action="store_true")
    hf_gguf_release_parser.add_argument("--no-sha256", action="store_true")
    hf_gguf_release_parser.add_argument("--validation")
    hf_gguf_release_parser.add_argument("--allow-unvalidated", action="store_true")
    hf_gguf_release_parser.add_argument("--allow-smoke-validation", action="store_true")
    hf_gguf_release_parser.add_argument("--min-benchmark-prompt-tokens", type=int, default=8192)
    hf_gguf_release_parser.add_argument("--min-benchmark-gen-tokens", type=int, default=128)
    hf_gguf_release_parser.add_argument("--stock-compatible", action="store_true")
    hf_gguf_release_parser.add_argument("--quality-eval")

    validate_gguf_parser = sub.add_parser("validate-gguf", help="Run release-gating GGUF runtime checks.")
    validate_gguf_parser.add_argument("--gguf", required=True)
    validate_gguf_parser.add_argument("--output", required=True)
    validate_gguf_parser.add_argument("--llama-cpp", default="../llama.cpp")
    validate_gguf_parser.add_argument("--prompt", default="Write one short sentence about quantization.")
    validate_gguf_parser.add_argument("--ctx-size", type=int, default=256)
    validate_gguf_parser.add_argument("--n-predict", type=int, default=4)
    validate_gguf_parser.add_argument("--ngl", default="0")
    validate_gguf_parser.add_argument("--threads", type=int)
    validate_gguf_parser.add_argument("--timeout", type=float, default=600.0)
    validate_gguf_parser.add_argument("--flash-attn", default="off")
    validate_gguf_parser.add_argument("--bench", action="store_true")
    validate_gguf_parser.add_argument("--bench-prompt-tokens", type=int, default=512)
    validate_gguf_parser.add_argument("--bench-gen-tokens", type=int, default=16)

    eval_gguf_parser = sub.add_parser("eval-gguf", help="Run small quality samples against a GGUF artifact.")
    eval_gguf_parser.add_argument("--gguf", required=True)
    eval_gguf_parser.add_argument("--output", required=True)
    eval_gguf_parser.add_argument("--suite", default="benchmarks/qwen36_quality_samples.jsonl")
    eval_gguf_parser.add_argument("--llama-cpp", default="../llama.cpp")
    eval_gguf_parser.add_argument("--ctx-size", type=int, default=2048)
    eval_gguf_parser.add_argument("--ngl", default="99")
    eval_gguf_parser.add_argument("--threads", type=int)
    eval_gguf_parser.add_argument("--timeout", type=float, default=600.0)
    eval_gguf_parser.add_argument("--flash-attn", default="on")
    eval_gguf_parser.add_argument("--temperature", type=float, default=0.0)
    eval_gguf_parser.add_argument("--top-p", type=float)
    eval_gguf_parser.add_argument("--max-samples", type=int)
    eval_gguf_parser.add_argument("--sample-id", action="append", default=[])
    eval_gguf_parser.add_argument("--reference")
    eval_gguf_parser.add_argument("--prompt-format", choices=("raw", "qwen3-no-think"), default="raw")
    eval_gguf_parser.add_argument("--ignore-eos", action="store_true")

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


def cmd_dynamic_gguf_profiles() -> int:
    print(json.dumps(dynamic_profiles_payload(), indent=2))
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


def cmd_monitor(root: str, watch: bool, interval: float) -> int:
    if watch:
        return watch_monitor(root=root, interval=interval)
    print_monitor(build_monitor_payload(root))
    return 0


def cmd_pack_release(input_dir: str, output: str, force: bool, max_tensors: int | None, copy_dtype: str) -> int:
    payload = pack_release(input_dir, output, force=force, max_tensors=max_tensors, copy_dtype=copy_dtype)
    summary = {
        "manifest": str(Path(output) / "opentq-pack.json"),
        "release_slug": payload["release_slug"],
        "model_id": payload["model_id"],
        "schema": payload["schema"],
        "totals": payload["totals"],
    }
    print(json.dumps(summary, indent=2))
    return 0


def cmd_prepare_hf(packed: str, output: str, repo_id: str, link_mode: str) -> int:
    payload = prepare_hf_release(packed, output, repo_id, link_mode=link_mode)
    print(json.dumps(payload, indent=2))
    return 0


def cmd_audit_pack_runtime(packed: str, output: str | None, max_tensors: int | None, dequantize_samples: int) -> int:
    payload = audit_packed_runtime(
        PackAuditOptions(
            packed_dir=Path(packed),
            max_tensors=max_tensors,
            dequantize_samples=dequantize_samples,
        )
    )
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if payload["overall_pass"] else 1


def cmd_runtime_fixtures(packed: str, output: str, max_fixtures_per_variant: int) -> int:
    payload = write_runtime_fixtures(packed, output, max_per_variant=max_fixtures_per_variant)
    print(json.dumps(payload, indent=2))
    return 0


def cmd_probe_pack_runtime(
    packed: str,
    fixtures_output: str,
    probe_binary: str,
    output: str,
    audit_max_tensors: int | None,
    audit_dequantize_samples: int,
    max_fixtures_per_variant: int,
    timeout: float,
) -> int:
    payload = run_runtime_probe(
        RuntimeProbeOptions(
            packed_dir=Path(packed),
            fixtures_output=Path(fixtures_output),
            probe_binary=Path(probe_binary),
            output=Path(output),
            audit_max_tensors=audit_max_tensors,
            audit_dequantize_samples=audit_dequantize_samples,
            max_fixtures_per_variant=max_fixtures_per_variant,
            timeout_seconds=timeout,
        )
    )
    print(json.dumps(payload, indent=2))
    return 0 if payload["overall_pass"] else 1


def cmd_gguf_plan(packed: str, output: str) -> int:
    payload = write_gguf_plan(packed, output)
    summary = {
        "manifest": output,
        "schema": payload["schema"],
        "release_slug": payload["release_slug"],
        "status": payload["status"],
        "custom_tensor_types": payload["custom_tensor_types"],
        "tensor_count": len(payload["tensors"]),
    }
    print(json.dumps(summary, indent=2))
    return 0


def cmd_export_gguf(packed: str, output: str, llama_cpp: str, max_tensors: int | None, include_vision: bool) -> int:
    payload = export_gguf(
        GGUFExportOptions(
            packed_dir=Path(packed),
            output=Path(output),
            llama_cpp_dir=Path(llama_cpp),
            text_only=not include_vision,
            max_tensors=max_tensors,
        )
    )
    print(json.dumps(payload, indent=2))
    return 0


def cmd_dynamic_gguf_plan(
    recipe: str,
    profile: str | None,
    policy_file: str | None,
    output: str,
    llama_cpp: str,
    source_gguf: str | None,
    target_gguf: str | None,
    include_vision: bool,
    vision_only: bool,
    no_converter_mapping: bool,
) -> int:
    payload = write_dynamic_gguf_plan(
        DynamicGGUFPlanOptions(
            recipe_key=recipe,
            profile_name=profile,
            policy_file=Path(policy_file) if policy_file else None,
            output_dir=Path(output),
            llama_cpp_dir=Path(llama_cpp),
            source_gguf=Path(source_gguf) if source_gguf else None,
            target_gguf=Path(target_gguf) if target_gguf else None,
            include_vision=include_vision,
            include_language=not vision_only,
            use_converter_mapping=not no_converter_mapping,
        )
    )
    print(
        json.dumps(
            {
                "schema": payload["schema"],
                "model_id": payload["model_id"],
                "profile": payload["profile"]["name"],
                "policy_source": payload["policy_source"],
                "base_ftype": payload["profile"]["base_ftype"],
                "compatibility": payload["compatibility"],
                "outputs": payload["outputs"],
                "summary": payload["summary"],
            },
            indent=2,
        )
    )
    return 0 if payload["summary"]["unmapped_count"] == 0 else 1


def cmd_kv_cache_plan(
    output: str,
    model_id: str,
    num_layers: int,
    default_dtype: str,
    promote_dtype: str,
    edge_layers: int,
    periodic_stride: int,
    weight_plan: str | None,
) -> int:
    payload = write_kv_cache_policy(
        KVCachePlanOptions(
            output_dir=Path(output),
            model_id=model_id,
            num_layers=num_layers,
            default_dtype=default_dtype,
            promote_dtype=promote_dtype,
            edge_layers=edge_layers,
            periodic_stride=periodic_stride,
            weight_plan=Path(weight_plan) if weight_plan else None,
        )
    )
    print(
        json.dumps(
            {
                "schema": payload["schema"],
                "model_id": payload["model_id"],
                "summary": payload["summary"],
                "outputs": payload["outputs"],
            },
            indent=2,
        )
    )
    return 0


def cmd_pruning_candidates(
    plan: str,
    output: str,
    max_candidates: int,
    prune_threshold: float,
    aggressive_threshold: float,
) -> int:
    payload = write_pruning_candidates(
        PruningCandidateOptions(
            plan_path=Path(plan),
            output_dir=Path(output),
            max_candidates=max_candidates,
            prune_threshold=prune_threshold,
            aggressive_threshold=aggressive_threshold,
        )
    )
    print(
        json.dumps(
            {
                "schema": payload["schema"],
                "model_id": payload["model_id"],
                "profile": payload["profile"],
                "summary": payload["summary"],
                "outputs": payload["outputs"],
            },
            indent=2,
        )
    )
    return 0


def cmd_allocation_ui(plan: str, output: str, title: str, metrics: str | None) -> int:
    payload = write_allocation_ui(
        AllocationUIOptions(
            plan_path=Path(plan),
            output_dir=Path(output),
            title=title,
            metrics_path=Path(metrics) if metrics else None,
        )
    )
    print(
        json.dumps(
            {
                "schema": payload["schema"],
                "model_id": payload["model_id"],
                "profile": payload["profile"],
                "summary": payload["summary"],
                "outputs": payload["outputs"],
            },
            indent=2,
        )
    )
    return 0


def cmd_prepare_hf_gguf(
    gguf: str,
    output: str,
    repo_id: str,
    base_model: str,
    runtime_repo: str,
    link_mode: str,
    include_vision: bool,
    no_sha256: bool,
    validation: str | None,
    allow_unvalidated: bool,
    allow_smoke_validation: bool,
    min_benchmark_prompt_tokens: int,
    min_benchmark_gen_tokens: int,
    stock_compatible: bool,
    quality_eval: str | None,
) -> int:
    payload = prepare_hf_gguf_release(
        gguf,
        output,
        repo_id,
        base_model=base_model,
        runtime_repo=runtime_repo,
        link_mode=link_mode,
        text_only=not include_vision,
        compute_sha256=not no_sha256,
        validation_path=validation,
        require_validation=not allow_unvalidated,
        require_benchmark=not allow_smoke_validation,
        min_benchmark_prompt_tokens=min_benchmark_prompt_tokens,
        min_benchmark_gen_tokens=min_benchmark_gen_tokens,
        stock_compatible=stock_compatible,
        quality_eval_path=quality_eval,
    )
    print(json.dumps(payload, indent=2))
    return 0


def cmd_validate_gguf(
    gguf: str,
    output: str,
    llama_cpp: str,
    prompt: str,
    ctx_size: int,
    n_predict: int,
    ngl: str,
    threads: int | None,
    timeout: float,
    flash_attn: str,
    bench: bool,
    bench_prompt_tokens: int,
    bench_gen_tokens: int,
) -> int:
    payload = validate_gguf(
        GGUFValidationOptions(
            gguf=Path(gguf),
            output=Path(output),
            llama_cpp_dir=Path(llama_cpp),
            prompt=prompt,
            ctx_size=ctx_size,
            n_predict=n_predict,
            gpu_layers=ngl,
            threads=threads,
            timeout_seconds=timeout,
            flash_attn=flash_attn,
            run_bench=bench,
            bench_prompt_tokens=bench_prompt_tokens,
            bench_gen_tokens=bench_gen_tokens,
        )
    )
    print(json.dumps(payload, indent=2))
    return 0 if payload["overall_pass"] else 1


def cmd_eval_gguf(
    gguf: str,
    output: str,
    suite: str,
    llama_cpp: str,
    ctx_size: int,
    ngl: str,
    threads: int | None,
    timeout: float,
    flash_attn: str,
    temperature: float,
    top_p: float | None,
    max_samples: int | None,
    sample_ids: list[str],
    reference: str | None,
    prompt_format: str,
    ignore_eos: bool,
) -> int:
    payload = run_quality_eval(
        GGUFQualityEvalOptions(
            gguf=Path(gguf),
            output=Path(output),
            suite=Path(suite),
            llama_cpp_dir=Path(llama_cpp),
            ctx_size=ctx_size,
            gpu_layers=ngl,
            threads=threads,
            timeout_seconds=timeout,
            flash_attn=flash_attn,
            temperature=temperature,
            top_p=top_p,
            max_samples=max_samples,
            sample_ids=tuple(sample_ids),
            reference=Path(reference) if reference else None,
            prompt_format=prompt_format,
            ignore_eos=ignore_eos,
        )
    )
    print(json.dumps(payload, indent=2))
    return 0 if payload["overall_pass"] else 1


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "variants":
        return cmd_variants()
    if args.command == "dynamic-gguf-profiles":
        return cmd_dynamic_gguf_profiles()
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
    if args.command == "monitor":
        return cmd_monitor(args.root, args.watch, args.interval)
    if args.command == "pack-release":
        return cmd_pack_release(args.input, args.output, args.force, args.max_tensors, args.copy_dtype)
    if args.command == "prepare-hf":
        return cmd_prepare_hf(args.packed, args.output, args.repo_id, args.link_mode)
    if args.command == "audit-pack-runtime":
        return cmd_audit_pack_runtime(args.packed, args.output, args.max_tensors, args.dequantize_samples)
    if args.command == "runtime-fixtures":
        return cmd_runtime_fixtures(args.packed, args.output, args.max_fixtures_per_variant)
    if args.command == "probe-pack-runtime":
        return cmd_probe_pack_runtime(
            args.packed,
            args.fixtures_output,
            args.probe_binary,
            args.output,
            args.audit_max_tensors,
            args.audit_dequantize_samples,
            args.max_fixtures_per_variant,
            args.timeout,
        )
    if args.command == "gguf-plan":
        return cmd_gguf_plan(args.packed, args.output)
    if args.command == "export-gguf":
        return cmd_export_gguf(args.packed, args.output, args.llama_cpp, args.max_tensors, args.include_vision)
    if args.command == "dynamic-gguf-plan":
        return cmd_dynamic_gguf_plan(
            args.recipe,
            args.profile,
            args.policy_file,
            args.output,
            args.llama_cpp,
            args.source_gguf,
            args.target_gguf,
            args.include_vision,
            args.vision_only,
            args.no_converter_mapping,
        )
    if args.command == "kv-cache-plan":
        return cmd_kv_cache_plan(
            args.output,
            args.model_id,
            args.num_layers,
            args.default_dtype,
            args.promote_dtype,
            args.edge_layers,
            args.periodic_stride,
            args.weight_plan,
        )
    if args.command == "pruning-candidates":
        return cmd_pruning_candidates(
            args.plan,
            args.output,
            args.max_candidates,
            args.prune_threshold,
            args.aggressive_threshold,
        )
    if args.command == "allocation-ui":
        return cmd_allocation_ui(args.plan, args.output, args.title, args.metrics)
    if args.command == "prepare-hf-gguf":
        return cmd_prepare_hf_gguf(
            args.gguf,
            args.output,
            args.repo_id,
            args.base_model,
            args.runtime_repo,
            args.link_mode,
            args.include_vision,
            args.no_sha256,
            args.validation,
            args.allow_unvalidated,
            args.allow_smoke_validation,
            args.min_benchmark_prompt_tokens,
            args.min_benchmark_gen_tokens,
            args.stock_compatible,
            args.quality_eval,
        )
    if args.command == "validate-gguf":
        return cmd_validate_gguf(
            args.gguf,
            args.output,
            args.llama_cpp,
            args.prompt,
            args.ctx_size,
            args.n_predict,
            args.ngl,
            args.threads,
            args.timeout,
            args.flash_attn,
            args.bench,
            args.bench_prompt_tokens,
            args.bench_gen_tokens,
        )
    if args.command == "eval-gguf":
        return cmd_eval_gguf(
            args.gguf,
            args.output,
            args.suite,
            args.llama_cpp,
            args.ctx_size,
            args.ngl,
            args.threads,
            args.timeout,
            args.flash_attn,
            args.temperature,
            args.top_p,
            args.max_samples,
            args.sample_id,
            args.reference,
            args.prompt_format,
            args.ignore_eos,
        )
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
