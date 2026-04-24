from __future__ import annotations

import hashlib
import json
import math
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from itertools import count
from typing import Any

import numpy as np
from huggingface_hub import hf_hub_download
from safetensors import safe_open
import torch

from .hf import fetch_safetensors_index
from .policies import TensorAction, resolve_tensor_action
from .quantize import quantize_tensor
from .recipes import get_recipe
from .variants import get_variant


@dataclass
class TensorArtifactResult:
    name: str
    category: str
    mode: str
    variant_name: str | None
    dtype: str
    shape: tuple[int, ...]
    source_file: str
    tensor_dir: str
    part_count: int
    num_values: int
    mse: float | None = None
    max_abs_error: float | None = None
    sum_squared_error: float | None = None
    skipped: bool = False


def sanitize_tensor_name(name: str) -> str:
    return name.replace(".", "__")


def tensor_seed(release_slug: str, tensor_name: str) -> int:
    digest = hashlib.sha256(f"{release_slug}:{tensor_name}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little")


def tensor_dir(root: Path, tensor_name: str) -> Path:
    return root / "tensors" / sanitize_tensor_name(tensor_name)


def choose_chunk_rows(shape: tuple[int, ...], target_elements: int = 8_388_608) -> int:
    if not shape:
        return 1
    if len(shape) == 1:
        return max(1, min(shape[0], target_elements))
    row_width = int(np.prod(shape[1:]))
    return max(1, target_elements // max(row_width, 1))


def slice_shape(tensor_slice: Any) -> tuple[int, ...]:
    try:
        return tuple(int(dim) for dim in tensor_slice.get_shape())
    except AttributeError:
        return tuple(int(dim) for dim in np.asarray(tensor_slice).shape)


def to_numpy(tensor: Any) -> np.ndarray:
    if isinstance(tensor, torch.Tensor):
        materialized = tensor.detach().cpu()
        if torch.is_floating_point(materialized) and materialized.dtype != torch.float32:
            materialized = materialized.to(torch.float32)
        return materialized.numpy()
    return np.asarray(tensor)


def load_tensor_chunk(reader: Any, tensor_name: str, start: int | None = None, stop: int | None = None) -> np.ndarray:
    tensor_slice = reader.get_slice(tensor_name)
    shape = slice_shape(tensor_slice)
    if start is None and stop is None:
        return to_numpy(reader.get_tensor(tensor_name))
    if len(shape) == 0:
        return to_numpy(reader.get_tensor(tensor_name))
    if len(shape) == 1:
        return to_numpy(tensor_slice[slice(start, stop)])
    index = (slice(start, stop),) + tuple(slice(None) for _ in shape[1:])
    return to_numpy(tensor_slice[index])


def get_tensor_shape(reader: Any, tensor_name: str) -> tuple[int, ...]:
    return slice_shape(reader.get_slice(tensor_name))


def get_tensor_dtype(reader: Any, tensor_name: str) -> str:
    return str(reader.get_slice(tensor_name).get_dtype()).lower()


def iter_tensor_chunks(reader: Any, tensor_name: str, shape: tuple[int, ...]):
    rows = choose_chunk_rows(shape)
    if len(shape) == 0:
        yield 0, None, None, load_tensor_chunk(reader, tensor_name)
        return
    for part_index, start in zip(count(), range(0, shape[0], rows)):
        stop = min(start + rows, shape[0])
        yield part_index, start, stop, load_tensor_chunk(reader, tensor_name, start, stop)


def dump_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_copy_tensor(reader: Any, tensor_name: str, source_file: str, action: TensorAction, output_root: Path) -> TensorArtifactResult:
    destination = tensor_dir(output_root, tensor_name)
    destination.mkdir(parents=True, exist_ok=True)

    shape = get_tensor_shape(reader, tensor_name)
    source_dtype = get_tensor_dtype(reader, tensor_name)
    part_count = 0
    storage_dtype = ""

    for part_index, _start, _stop, array in iter_tensor_chunks(reader, tensor_name, shape):
        storage_dtype = str(array.dtype)
        np.save(destination / f"part-{part_index:05d}.npy", array, allow_pickle=False)
        part_count += 1

    dump_json(
        destination / "meta.json",
        {
            "name": tensor_name,
            "category": action.category,
            "mode": action.mode,
            "shape": list(shape),
            "dtype": source_dtype,
            "storage_dtype": storage_dtype,
            "source_file": source_file,
            "part_count": part_count,
        },
    )

    return TensorArtifactResult(
        name=tensor_name,
        category=action.category,
        mode=action.mode,
        variant_name=None,
        dtype=source_dtype,
        shape=shape,
        source_file=source_file,
        tensor_dir=str(destination.relative_to(output_root)),
        part_count=part_count,
        num_values=int(np.prod(shape)),
    )


def write_quantized_tensor(reader: Any, tensor_name: str, source_file: str, action: TensorAction, output_root: Path, release_slug: str) -> TensorArtifactResult:
    if action.variant_name is None:
        raise ValueError(f"quantize action missing variant for {tensor_name}")

    variant = get_variant(action.variant_name)
    destination = tensor_dir(output_root, tensor_name)
    destination.mkdir(parents=True, exist_ok=True)

    shape = get_tensor_shape(reader, tensor_name)
    source_dtype = get_tensor_dtype(reader, tensor_name)
    part_count = 0
    seed = tensor_seed(release_slug, tensor_name)
    total_values = int(np.prod(shape))
    total_sse = 0.0
    total_abs_error = 0.0
    storage_dtype = ""

    for part_index, start, stop, array in iter_tensor_chunks(reader, tensor_name, shape):
        storage_dtype = str(array.dtype)
        result = quantize_tensor(array, variant, seed=seed + part_index * 104729, return_reconstruction=False)
        residual_payload = result.packed.residual_indices if result.packed.residual_indices is not None else np.array([], dtype=np.uint8)
        residual_scales = result.packed.residual_scales if result.packed.residual_scales is not None else np.array([], dtype=np.float32)
        np.savez_compressed(
            destination / f"part-{part_index:05d}.npz",
            indices=result.packed.indices,
            scales=result.packed.scales,
            residual_indices=residual_payload,
            residual_scales=residual_scales,
            shape=np.array(array.shape, dtype=np.int64),
            row_start=np.array([-1 if start is None else start], dtype=np.int64),
            row_stop=np.array([-1 if stop is None else stop], dtype=np.int64),
        )
        total_sse += result.packed.sum_squared_error
        total_abs_error = max(total_abs_error, result.packed.max_abs_error)
        part_count += 1

    dump_json(
        destination / "meta.json",
        {
            "name": tensor_name,
            "category": action.category,
            "mode": action.mode,
            "variant_name": action.variant_name,
            "shape": list(shape),
            "dtype": source_dtype,
            "storage_dtype": storage_dtype,
            "source_file": source_file,
            "part_count": part_count,
            "seed": seed,
            "group_size": variant.group_size,
            "block_size": variant.block_size,
            "sub_block_size": variant.sub_block_size,
        },
    )

    return TensorArtifactResult(
        name=tensor_name,
        category=action.category,
        mode=action.mode,
        variant_name=action.variant_name,
        dtype=source_dtype,
        shape=shape,
        source_file=source_file,
        tensor_dir=str(destination.relative_to(output_root)),
        part_count=part_count,
        num_values=total_values,
        mse=total_sse / max(total_values, 1),
        max_abs_error=total_abs_error,
        sum_squared_error=total_sse,
    )


def build_release_plan(recipe_key: str, release_slug: str, include_vision: bool = True, include_language: bool = True) -> dict[str, Any]:
    recipe = get_recipe(recipe_key)
    index_data = fetch_safetensors_index(recipe.model_id)
    weight_map = index_data["weight_map"]

    tensors = []
    counter = Counter()
    for tensor_name in sorted(weight_map):
        action = resolve_tensor_action(release_slug, tensor_name, include_vision=include_vision, include_language=include_language)
        row = {
            "name": tensor_name,
            "source_file": weight_map[tensor_name],
            "category": action.category,
            "mode": action.mode,
            "variant_name": action.variant_name,
        }
        tensors.append(row)
        counter[(action.mode, action.variant_name or "copy")] += 1

    summary = {
        "total_tensors": len(tensors),
        "by_action": {f"{mode}:{variant}": count for (mode, variant), count in sorted(counter.items())},
    }
    return {
        "recipe_key": recipe_key,
        "model_id": recipe.model_id,
        "release_slug": release_slug,
        "include_vision": include_vision,
        "include_language": include_language,
        "summary": summary,
        "tensors": tensors,
    }


def quantize_release(
    recipe_key: str,
    release_slug: str,
    output_dir: str,
    *,
    include_vision: bool = True,
    include_language: bool = True,
    max_tensors: int | None = None,
    only_shard: str | None = None,
    skip_existing: bool = True,
) -> dict[str, Any]:
    recipe = get_recipe(recipe_key)
    plan = build_release_plan(recipe_key, release_slug, include_vision=include_vision, include_language=include_language)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    dump_json(output_root / "plan.json", plan)
    progress_path = output_root / "progress.jsonl"

    tensors_by_shard: dict[str, list[dict[str, Any]]] = {}
    for row in plan["tensors"]:
        if row["mode"] == "skip":
            continue
        tensors_by_shard.setdefault(row["source_file"], []).append(row)

    processed_results: list[TensorArtifactResult] = []
    processed_count = 0
    started_at = time.time()

    for shard_file in sorted(tensors_by_shard):
        if only_shard and shard_file != only_shard:
            continue

        local_path = hf_hub_download(repo_id=recipe.model_id, filename=shard_file)
        with safe_open(local_path, framework="pt") as reader:
            for row in tensors_by_shard[shard_file]:
                if max_tensors is not None and processed_count >= max_tensors:
                    break

                destination = tensor_dir(output_root, row["name"])
                if skip_existing and (destination / "meta.json").exists():
                    continue

                action = TensorAction(category=row["category"], mode=row["mode"], variant_name=row["variant_name"])
                if action.mode == "copy":
                    result = write_copy_tensor(reader, row["name"], shard_file, action, output_root)
                elif action.mode == "quantize":
                    result = write_quantized_tensor(reader, row["name"], shard_file, action, output_root, release_slug)
                else:
                    continue

                processed_results.append(result)
                processed_count += 1
                progress_path.parent.mkdir(parents=True, exist_ok=True)
                with progress_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(asdict(result)) + "\n")

            if max_tensors is not None and processed_count >= max_tensors:
                break

    summary_counter = Counter(result.mode for result in processed_results)
    quantized = [result for result in processed_results if result.mode == "quantize"]
    manifest = {
        "recipe_key": recipe_key,
        "model_id": recipe.model_id,
        "release_slug": release_slug,
        "output_dir": str(output_root),
        "include_vision": include_vision,
        "include_language": include_language,
        "processed_tensors": processed_count,
        "elapsed_seconds": round(time.time() - started_at, 2),
        "counts": dict(summary_counter),
        "quantized_tensors": len(quantized),
        "avg_quant_mse": None if not quantized else float(sum(result.mse for result in quantized if result.mse is not None) / len(quantized)),
        "max_abs_error": None if not quantized else float(max(result.max_abs_error or 0.0 for result in quantized)),
        "results": [asdict(result) for result in processed_results],
    }
    dump_json(output_root / "manifest.json", manifest)
    return manifest
