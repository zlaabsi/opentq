from __future__ import annotations

import hashlib
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .bitpack import pack_bits
from .variants import get_variant


PACK_SCHEMA = "opentq.pack.v1"


@dataclass(frozen=True)
class PackOptions:
    source: Path
    output: Path
    force: bool = False
    max_tensors: int | None = None
    copy_dtype: str = "float16"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_section(handle: Any, name: str, payload: bytes) -> dict[str, Any]:
    offset = handle.tell()
    handle.write(payload)
    return {"name": name, "offset": offset, "bytes": len(payload)}


def sanitized_tensor_file(tensor_dir: str) -> str:
    return Path(tensor_dir).name + ".otq"


def part_sort_key(path: Path) -> str:
    return path.name


def pack_copy_tensor(tensor_root: Path, output_file: Path, dtype: str) -> dict[str, Any]:
    np_dtype = np.dtype(dtype)
    sections = []
    values = 0
    with output_file.open("wb") as handle:
        for part in sorted(tensor_root.glob("part-*.npy"), key=part_sort_key):
            array = np.load(part, allow_pickle=False)
            values += int(array.size)
            sections.append(
                {
                    "part": part.name,
                    "shape": list(array.shape),
                    "dtype": str(np_dtype),
                    "data": write_section(handle, "data", np.asarray(array, dtype=np_dtype).tobytes()),
                }
            )
    return {"sections": sections, "values": values, "bytes": output_file.stat().st_size}


def pack_quant_tensor(tensor_root: Path, output_file: Path, variant_name: str) -> dict[str, Any]:
    variant = get_variant(variant_name)
    sections = []
    values = 0
    blocks = 0
    with output_file.open("wb") as handle:
        for part in sorted(tensor_root.glob("part-*.npz"), key=part_sort_key):
            with np.load(part) as payload:
                shape = tuple(int(item) for item in payload["shape"].tolist())
                indices = np.asarray(payload["indices"], dtype=np.uint8).reshape(-1)
                scales = np.asarray(payload["scales"], dtype=np.float16)
                values += int(np.prod(shape, dtype=np.int64))
                blocks += int(payload["indices"].shape[0])
                part_sections = {
                    "part": part.name,
                    "shape": list(shape),
                    "row_start": int(payload["row_start"][0]) if "row_start" in payload.files else None,
                    "row_stop": int(payload["row_stop"][0]) if "row_stop" in payload.files else None,
                    "indices": write_section(handle, "indices", pack_bits(indices, variant.weight_bits)),
                    "scales": write_section(handle, "scales", scales.tobytes()),
                    "index_count": int(indices.size),
                    "scale_shape": list(scales.shape),
                    "scale_dtype": "float16",
                }
                if variant.residual_bits is not None and "residual_indices" in payload.files:
                    residual_indices = np.asarray(payload["residual_indices"], dtype=np.uint8).reshape(-1)
                    residual_scales = np.asarray(payload["residual_scales"], dtype=np.float16)
                    part_sections["residual_indices"] = write_section(handle, "residual_indices", pack_bits(residual_indices, variant.residual_bits))
                    part_sections["residual_scales"] = write_section(handle, "residual_scales", residual_scales.tobytes())
                    part_sections["residual_index_count"] = int(residual_indices.size)
                    part_sections["residual_scale_shape"] = list(residual_scales.shape)
                    part_sections["residual_scale_dtype"] = "float16"
                sections.append(part_sections)
    return {"sections": sections, "values": values, "blocks": blocks, "bytes": output_file.stat().st_size}


def pack_release(source: str | Path, output: str | Path, *, force: bool = False, max_tensors: int | None = None, copy_dtype: str = "float16") -> dict[str, Any]:
    options = PackOptions(source=Path(source), output=Path(output), force=force, max_tensors=max_tensors, copy_dtype=copy_dtype)
    manifest_path = options.source / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing source manifest: {manifest_path}")
    manifest = load_json(manifest_path)

    if options.output.exists() and force:
        shutil.rmtree(options.output)
    options.output.mkdir(parents=True, exist_ok=True)
    tensor_output = options.output / "tensors"
    tensor_output.mkdir(parents=True, exist_ok=True)

    started_at = time.time()
    tensor_rows = []
    totals = {
        "tensors": 0,
        "quantized_tensors": 0,
        "copied_tensors": 0,
        "values": 0,
        "payload_bytes": 0,
    }

    results = manifest.get("results", [])
    for row in results[: options.max_tensors]:
        tensor_file = tensor_output / sanitized_tensor_file(row["tensor_dir"])
        source_tensor_root = options.source / row["tensor_dir"]
        if tensor_file.exists() and not force:
            raise FileExistsError(f"packed tensor already exists: {tensor_file}")

        if row["mode"] == "copy":
            packed = pack_copy_tensor(source_tensor_root, tensor_file, options.copy_dtype)
            totals["copied_tensors"] += 1
            layout = {"copy_dtype": options.copy_dtype}
        elif row["mode"] == "quantize":
            packed = pack_quant_tensor(source_tensor_root, tensor_file, row["variant_name"])
            totals["quantized_tensors"] += 1
            variant = get_variant(row["variant_name"])
            layout = {
                "variant": variant.name,
                "weight_bits": variant.weight_bits,
                "residual_bits": variant.residual_bits,
                "group_size": variant.group_size,
                "block_size": variant.block_size,
                "sub_block_size": variant.sub_block_size,
                "scale_dtype": "float16",
            }
        else:
            continue

        tensor_rows.append(
            {
                "name": row["name"],
                "category": row["category"],
                "mode": row["mode"],
                "variant_name": row.get("variant_name"),
                "shape": row["shape"],
                "dtype": row["dtype"],
                "file": str(tensor_file.relative_to(options.output)),
                "sha256": sha256_file(tensor_file),
                "bytes": packed["bytes"],
                "num_values": row["num_values"],
                "layout": layout,
                "sections": packed["sections"],
            }
        )
        totals["tensors"] += 1
        totals["values"] += int(row["num_values"])
        totals["payload_bytes"] += int(packed["bytes"])

    packed_manifest = {
        "schema": PACK_SCHEMA,
        "source_release": str(options.source),
        "source_manifest_sha256": sha256_file(manifest_path),
        "model_id": manifest["model_id"],
        "release_slug": manifest["release_slug"],
        "created_at": int(time.time()),
        "elapsed_seconds": round(time.time() - started_at, 2),
        "copy_dtype": options.copy_dtype,
        "totals": totals,
        "gguf": {
            "status": "requires-runtime-patch",
            "metadata_required": ["general.architecture", "general.quantization_version", "opentq.schema", "opentq.release_slug"],
            "tensor_payload": "OpenTQ packed tensor files are GGUF-ready payloads but not a stock GGUF container.",
        },
        "tensors": tensor_rows,
    }
    dump_json(options.output / "opentq-pack.json", packed_manifest)
    return packed_manifest
