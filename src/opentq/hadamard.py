from __future__ import annotations

import math

import numpy as np


def next_power_of_two(value: int) -> int:
    if value < 1:
        return 1
    return 1 << (value - 1).bit_length()


def fwht(vector: np.ndarray) -> np.ndarray:
    work = np.asarray(vector, dtype=np.float32).copy()
    n = work.shape[-1]
    if n & (n - 1):
        raise ValueError("FWHT input length must be a power of two")

    original_shape = work.shape
    work = work.reshape(-1, n)
    h = 1
    while h < n:
        work = work.reshape(-1, n // (2 * h), 2, h)
        left = work[:, :, 0, :].copy()
        right = work[:, :, 1, :].copy()
        work[:, :, 0, :] = left + right
        work[:, :, 1, :] = left - right
        work = work.reshape(-1, n)
        h *= 2

    work /= math.sqrt(n)
    return work.reshape(original_shape)


def random_signs_matrix(length: int, seeds: np.ndarray) -> np.ndarray:
    seed_values = np.asarray(seeds, dtype=np.uint64).reshape(-1, 1)
    positions = np.arange(length, dtype=np.uint64).reshape(1, -1)
    mixed = seed_values ^ (positions + np.uint64(0x9E3779B97F4A7C15))
    mixed ^= mixed >> np.uint64(30)
    mixed *= np.uint64(0xBF58476D1CE4E5B9)
    mixed ^= mixed >> np.uint64(27)
    mixed *= np.uint64(0x94D049BB133111EB)
    mixed ^= mixed >> np.uint64(31)
    return np.where((mixed & np.uint64(1)) == 0, -1.0, 1.0).astype(np.float32)


def random_signs(length: int, seed: int) -> np.ndarray:
    return random_signs_matrix(length, np.array([seed], dtype=np.uint64))[0]


def hadamard_rotate(values: np.ndarray, seed: int) -> tuple[np.ndarray, int]:
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    padded_length = next_power_of_two(flat.size)
    padded = np.zeros(padded_length, dtype=np.float32)
    padded[: flat.size] = flat
    signs = random_signs(padded_length, seed)
    rotated = fwht(padded * signs)
    return rotated[: flat.size], padded_length


def hadamard_rotate_groups(values: np.ndarray, seeds: np.ndarray) -> np.ndarray:
    groups = np.asarray(values, dtype=np.float32)
    if groups.ndim != 2:
        raise ValueError("hadamard_rotate_groups expects a 2D array")
    if groups.shape[1] & (groups.shape[1] - 1):
        raise ValueError("group width must be a power of two")
    signs = random_signs_matrix(groups.shape[1], seeds)
    return fwht(groups * signs)


def hadamard_unrotate(values: np.ndarray, seed: int, original_length: int | None = None) -> np.ndarray:
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    padded_length = next_power_of_two(flat.size)
    padded = np.zeros(padded_length, dtype=np.float32)
    padded[: flat.size] = flat
    signs = random_signs(padded_length, seed)
    unrotated = fwht(padded) * signs
    length = flat.size if original_length is None else original_length
    return unrotated[:length]


def hadamard_unrotate_groups(values: np.ndarray, seeds: np.ndarray) -> np.ndarray:
    groups = np.asarray(values, dtype=np.float32)
    if groups.ndim != 2:
        raise ValueError("hadamard_unrotate_groups expects a 2D array")
    if groups.shape[1] & (groups.shape[1] - 1):
        raise ValueError("group width must be a power of two")
    signs = random_signs_matrix(groups.shape[1], seeds)
    return fwht(groups) * signs
