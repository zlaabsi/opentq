from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .codebooks import lloyd_max_gaussian
from .hadamard import hadamard_rotate_groups, hadamard_unrotate_groups
from .packing import PackedTensor
from .variants import QuantVariant


@dataclass
class QuantizeResult:
    packed: PackedTensor
    reconstruction: np.ndarray | None

    def to_manifest(self) -> dict[str, object]:
        return {
            "shape": list(self.packed.shape),
            "variant": self.packed.variant,
            "mse": self.packed.mse,
            "max_abs_error": self.packed.max_abs_error,
            "num_blocks": self.packed.num_blocks,
            "sum_squared_error": self.packed.sum_squared_error,
        }


@lru_cache(maxsize=16)
def gaussian_quantizer(bits: int) -> tuple[np.ndarray, np.ndarray]:
    return lloyd_max_gaussian(bits)


def quantize_sub_blocks(values: np.ndarray, bits: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    codebook, boundaries = gaussian_quantizer(bits)
    max_level = float(np.max(np.abs(codebook)))
    sub_blocks = np.asarray(values, dtype=np.float32)
    scales = np.maximum(np.max(np.abs(sub_blocks), axis=-1, keepdims=True) / max_level, 1e-8).astype(np.float32)
    normalized = sub_blocks / scales
    indices = np.searchsorted(boundaries[1:-1], normalized, side="right").astype(np.uint8)
    reconstructed = codebook[indices] * scales
    return indices, reconstructed.astype(np.float32), scales.astype(np.float32)


def quantize_tensor(tensor: np.ndarray, variant: QuantVariant, seed: int = 42, return_reconstruction: bool = True) -> QuantizeResult:
    source = np.asarray(tensor, dtype=np.float32)
    flat = source.reshape(-1)
    padded_size = ((flat.size + variant.group_size - 1) // variant.group_size) * variant.group_size
    padded = np.zeros(padded_size, dtype=np.float32)
    padded[: flat.size] = flat

    groups = padded.reshape(-1, variant.group_size)
    group_count = groups.shape[0]
    blocks_per_group = variant.group_size // variant.block_size
    sub_blocks_per_block = variant.block_size // variant.sub_block_size
    seeds = seed + (np.arange(group_count, dtype=np.uint64) * np.uint64(variant.group_size))

    if variant.use_wht:
        rotated_groups = hadamard_rotate_groups(groups, seeds)
    else:
        rotated_groups = groups.copy()

    sub_blocks = rotated_groups.reshape(group_count, blocks_per_group, sub_blocks_per_block, variant.sub_block_size)
    primary_indices, primary_reconstruction, primary_scales = quantize_sub_blocks(sub_blocks, variant.weight_bits)
    rotated_reconstruction = primary_reconstruction.reshape(group_count, variant.group_size)

    residual_indices = None
    residual_scales = None
    if variant.residual_bits is not None:
        residual = sub_blocks - primary_reconstruction
        residual_indices_raw, residual_reconstruction, residual_scales_raw = quantize_sub_blocks(residual, variant.residual_bits)
        residual_indices = residual_indices_raw.reshape(-1, variant.block_size)
        residual_scales = residual_scales_raw.reshape(-1, sub_blocks_per_block)
        rotated_reconstruction = rotated_reconstruction + residual_reconstruction.reshape(group_count, variant.group_size)

    if variant.use_wht:
        reconstructed_groups = hadamard_unrotate_groups(rotated_reconstruction, seeds)
    else:
        reconstructed_groups = rotated_reconstruction

    valid_reconstruction = reconstructed_groups.reshape(-1)[: flat.size]
    diff = flat - valid_reconstruction
    total_sse = float(np.sum(diff * diff))
    max_abs_error = 0.0 if diff.size == 0 else float(np.max(np.abs(diff)))
    reconstruction = None if not return_reconstruction else valid_reconstruction.reshape(source.shape)

    indices = primary_indices.reshape(-1, variant.block_size)
    scales = primary_scales.reshape(-1, sub_blocks_per_block)
    mse = float(total_sse / max(flat.size, 1))
    packed_tensor = PackedTensor(
        shape=source.shape,
        variant=variant.name,
        indices=indices,
        scales=scales,
        residual_indices=residual_indices,
        residual_scales=residual_scales,
        mse=mse,
        max_abs_error=max_abs_error,
        sum_squared_error=float(total_sse),
    )
    return QuantizeResult(packed=packed_tensor, reconstruction=reconstruction)
