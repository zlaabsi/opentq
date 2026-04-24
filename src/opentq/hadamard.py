from __future__ import annotations

import math

import numpy as np


def next_power_of_two(value: int) -> int:
    if value < 1:
        return 1
    return 1 << (value - 1).bit_length()


def fwht(vector: np.ndarray) -> np.ndarray:
    work = np.asarray(vector, dtype=np.float32).copy()
    n = work.shape[0]
    if n & (n - 1):
        raise ValueError("FWHT input length must be a power of two")

    h = 1
    while h < n:
        for start in range(0, n, h * 2):
            left = work[start : start + h].copy()
            right = work[start + h : start + (2 * h)].copy()
            work[start : start + h] = left + right
            work[start + h : start + (2 * h)] = left - right
        h *= 2

    work /= math.sqrt(n)
    return work


def random_signs(length: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=length)


def hadamard_rotate(values: np.ndarray, seed: int) -> tuple[np.ndarray, int]:
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    padded_length = next_power_of_two(flat.size)
    padded = np.zeros(padded_length, dtype=np.float32)
    padded[: flat.size] = flat
    signs = random_signs(padded_length, seed)
    rotated = fwht(padded * signs)
    return rotated[: flat.size], padded_length

