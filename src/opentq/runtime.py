from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .bitpack import unpack_bits
from .codebooks import lloyd_max_gaussian
from .hadamard import hadamard_unrotate_groups
from .run import tensor_seed
from .variants import QuantVariant, get_variant


@dataclass(frozen=True)
class TensorPart:
    tensor_name: str
    part_name: str
    shape: tuple[int, ...]
    row_start: int | None
    row_stop: int | None
    sections: dict[str, dict[str, Any]]


class OpenTQPack:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.manifest_path = self.root / "opentq-pack.json"
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"missing OpenTQ pack manifest: {self.manifest_path}")
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.tensors = {row["name"]: row for row in self.manifest["tensors"]}

    def tensor(self, name: str) -> dict[str, Any]:
        try:
            return self.tensors[name]
        except KeyError as exc:
            raise KeyError(f"tensor not found in OpenTQ pack: {name}") from exc

    def verify_tensor(self, name: str) -> bool:
        row = self.tensor(name)
        digest = hashlib.sha256()
        with (self.root / row["file"]).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest() == row["sha256"]

    def read_section(self, tensor_row: dict[str, Any], section: dict[str, Any]) -> bytes:
        with (self.root / tensor_row["file"]).open("rb") as handle:
            handle.seek(int(section["offset"]))
            return handle.read(int(section["bytes"]))

    def dequantize_tensor(self, name: str, *, dtype: np.dtype | str = np.float32) -> np.ndarray:
        row = self.tensor(name)
        if row["mode"] == "copy":
            return self.read_copy_tensor(row, dtype=dtype)
        if row["mode"] != "quantize":
            raise ValueError(f"unsupported tensor mode for {name}: {row['mode']}")
        return self.read_quant_tensor(row, dtype=dtype)

    def read_copy_tensor(self, row: dict[str, Any], *, dtype: np.dtype | str = np.float32) -> np.ndarray:
        arrays = []
        for part in row["sections"]:
            payload = self.read_section(row, part["data"])
            array = np.frombuffer(payload, dtype=np.dtype(part.get("dtype", row["layout"].get("copy_dtype", "float16"))))
            arrays.append(array.reshape(tuple(part["shape"])))
        if len(arrays) == 1:
            return arrays[0].astype(dtype, copy=False)
        return np.concatenate(arrays, axis=0).astype(dtype, copy=False)

    def read_quant_tensor(self, row: dict[str, Any], *, dtype: np.dtype | str = np.float32) -> np.ndarray:
        variant = get_variant(row["variant_name"])
        parts = []
        for part_index, part in enumerate(row["sections"]):
            parts.append(self.dequantize_quant_part(row, part, variant, part_index))
        tensor = np.concatenate(parts, axis=0) if len(parts) > 1 and len(parts[0].shape) > 0 else parts[0]
        return tensor.reshape(tuple(row["shape"])).astype(dtype, copy=False)

    def dequantize_quant_part(self, row: dict[str, Any], part: dict[str, Any], variant: QuantVariant, part_index: int) -> np.ndarray:
        shape = tuple(int(item) for item in part["shape"])
        values = int(np.prod(shape, dtype=np.int64))
        padded_values = ((values + variant.group_size - 1) // variant.group_size) * variant.group_size
        indices = unpack_bits(self.read_section(row, part["indices"]), variant.weight_bits, int(part["index_count"]))
        if indices.size != padded_values:
            raise ValueError(f"{row['name']}:{part['part']} index count mismatch: {indices.size} != {padded_values}")
        scales = np.frombuffer(self.read_section(row, part["scales"]), dtype=np.float16).astype(np.float32).reshape(tuple(part["scale_shape"]))
        rotated = reconstruct_rotated(indices, scales, variant.weight_bits, variant)

        if variant.residual_bits is not None and "residual_indices" in part:
            residual_indices = unpack_bits(
                self.read_section(row, part["residual_indices"]),
                variant.residual_bits,
                int(part["residual_index_count"]),
            )
            residual_scales = (
                np.frombuffer(self.read_section(row, part["residual_scales"]), dtype=np.float16)
                .astype(np.float32)
                .reshape(tuple(part["residual_scale_shape"]))
            )
            rotated += reconstruct_rotated(residual_indices, residual_scales, variant.residual_bits, variant)

        groups = rotated.reshape(-1, variant.group_size)
        seed_base = int(row.get("seed", tensor_seed(self.manifest["release_slug"], row["name"]))) + part_index * 104729
        seeds = seed_base + (np.arange(groups.shape[0], dtype=np.uint64) * np.uint64(variant.group_size))
        if variant.use_wht:
            decoded = hadamard_unrotate_groups(groups, seeds)
        else:
            decoded = groups
        return decoded.reshape(-1)[:values].reshape(shape)


def reconstruct_rotated(indices: np.ndarray, scales: np.ndarray, bits: int, variant: QuantVariant) -> np.ndarray:
    codebook, _ = lloyd_max_gaussian(bits)
    blocks = np.asarray(indices, dtype=np.uint8).reshape(-1, variant.block_size)
    sub_blocks_per_block = variant.block_size // variant.sub_block_size
    scale_rows = np.asarray(scales, dtype=np.float32).reshape(-1, sub_blocks_per_block)
    if blocks.shape[0] != scale_rows.shape[0]:
        raise ValueError(f"block/scale count mismatch: {blocks.shape[0]} != {scale_rows.shape[0]}")
    decoded = codebook[blocks].astype(np.float32).reshape(-1, sub_blocks_per_block, variant.sub_block_size)
    decoded *= scale_rows[:, :, None]
    return decoded.reshape(-1)
