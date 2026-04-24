from __future__ import annotations

from math import erf, exp, inf, pi, sqrt
from statistics import NormalDist

import numpy as np


_STD_NORMAL = NormalDist(mu=0.0, sigma=1.0)


def gaussian_pdf(x: float) -> float:
    return exp(-0.5 * x * x) / sqrt(2.0 * pi)


def gaussian_cdf(x: float) -> float:
    if x == inf:
        return 1.0
    if x == -inf:
        return 0.0
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def truncated_normal_mean(lower: float, upper: float) -> float:
    z = gaussian_cdf(upper) - gaussian_cdf(lower)
    if z < 1e-12:
        if lower == -inf:
            return upper - 1.0
        if upper == inf:
            return lower + 1.0
        return (lower + upper) * 0.5
    return (gaussian_pdf(lower) - gaussian_pdf(upper)) / z


def initial_centroids(bits: int) -> np.ndarray:
    levels = 2**bits
    probs = (np.arange(levels) + 0.5) / levels
    return np.array([_STD_NORMAL.inv_cdf(p) for p in probs], dtype=np.float64)


def lloyd_max_gaussian(bits: int, max_iter: int = 128, tol: float = 1e-8) -> tuple[np.ndarray, np.ndarray]:
    if bits <= 0:
        raise ValueError("bits must be positive")

    centroids = initial_centroids(bits)

    for _ in range(max_iter):
        boundaries = np.concatenate(
            [
                np.array([-np.inf], dtype=np.float64),
                (centroids[:-1] + centroids[1:]) * 0.5,
                np.array([np.inf], dtype=np.float64),
            ]
        )
        updated = np.array(
            [truncated_normal_mean(lower, upper) for lower, upper in zip(boundaries[:-1], boundaries[1:], strict=True)],
            dtype=np.float64,
        )
        if np.max(np.abs(updated - centroids)) < tol:
            centroids = updated
            break
        centroids = updated

    boundaries = np.concatenate(
        [
            np.array([-np.inf], dtype=np.float64),
            (centroids[:-1] + centroids[1:]) * 0.5,
            np.array([np.inf], dtype=np.float64),
        ]
    )
    return centroids.astype(np.float32), boundaries.astype(np.float32)

