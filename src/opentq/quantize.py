from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .codebooks import lloyd_max_gaussian
from .hadamard import hadamard_rotate
from .packing import PackedBlock, PackedTensor
from .variants import QuantVariant


@dataclass
class QuantizeResult:
    packed: PackedTensor
    reconstruction: np.ndarray

    def to_manifest(self) -> dict[str, object]:
        return {
            "shape": list(self.packed.shape),
            "variant": self.packed.variant,
            "mse": self.packed.mse,
            "max_abs_error": self.packed.max_abs_error,
            "num_blocks": len(self.packed.blocks),
        }


def nearest_codebook_index(values: np.ndarray, codebook: np.ndarray) -> np.ndarray:
    distances = np.abs(values[:, None] - codebook[None, :])
    return np.argmin(distances, axis=1).astype(np.uint8)


def quantize_sub_block(values: np.ndarray, bits: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    codebook, _ = lloyd_max_gaussian(bits)
    max_level = float(np.max(np.abs(codebook)))
    scale = max(np.max(np.abs(values)) / max_level, 1e-8)
    normalized = values / scale
    indices = nearest_codebook_index(normalized.astype(np.float32), codebook)
    reconstructed = codebook[indices] * scale
    return indices, reconstructed.astype(np.float32), np.array([scale], dtype=np.float32)


def quantize_block(block: np.ndarray, variant: QuantVariant, seed: int) -> tuple[PackedBlock, np.ndarray]:
    work = np.asarray(block, dtype=np.float32).copy()
    if variant.use_wht:
        work, _ = hadamard_rotate(work, seed)

    sub_blocks = []
    recon_parts = []
    scales = []

    for start in range(0, work.size, variant.sub_block_size):
        sub = work[start : start + variant.sub_block_size]
        idx, recon, scale = quantize_sub_block(sub, variant.weight_bits)
        sub_blocks.append(idx)
        recon_parts.append(recon)
        scales.append(scale)

    primary_indices = np.concatenate(sub_blocks).astype(np.uint8)
    reconstruction = np.concatenate(recon_parts).astype(np.float32)
    packed = PackedBlock(indices=primary_indices, scales=np.concatenate(scales).astype(np.float32))

    if variant.residual_bits is not None:
        residual = work - reconstruction
        residual_parts = []
        residual_indices_parts = []
        for start in range(0, residual.size, variant.sub_block_size):
            sub = residual[start : start + variant.sub_block_size]
            idx, recon, _ = quantize_sub_block(sub, variant.residual_bits)
            residual_indices_parts.append(idx)
            residual_parts.append(recon)
        residual_reconstruction = np.concatenate(residual_parts).astype(np.float32)
        reconstruction = reconstruction + residual_reconstruction
        packed.residual_indices = np.concatenate(residual_indices_parts).astype(np.uint8)

    return packed, reconstruction


def quantize_tensor(tensor: np.ndarray, variant: QuantVariant, seed: int = 42) -> QuantizeResult:
    array = np.asarray(tensor, dtype=np.float32)
    flat = array.reshape(-1)
    blocks = []
    reconstruction = np.zeros_like(flat)

    for offset in range(0, flat.size, variant.block_size):
        block = flat[offset : offset + variant.block_size]
        if block.size < variant.block_size:
            padded = np.zeros(variant.block_size, dtype=np.float32)
            padded[: block.size] = block
            block = padded
        packed_block, recon_block = quantize_block(block, variant, seed + offset)
        blocks.append(packed_block)
        reconstruction[offset : offset + min(variant.block_size, flat.size - offset)] = recon_block[: min(variant.block_size, flat.size - offset)]

    mse = float(np.mean((flat - reconstruction) ** 2))
    max_abs_error = float(np.max(np.abs(flat - reconstruction)))
    packed_tensor = PackedTensor(
        shape=array.shape,
        variant=variant.name,
        blocks=blocks,
        mse=mse,
        max_abs_error=max_abs_error,
    )
    return QuantizeResult(packed=packed_tensor, reconstruction=reconstruction.reshape(array.shape))

