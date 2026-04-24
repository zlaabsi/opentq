from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PackedBlock:
    indices: np.ndarray
    scales: np.ndarray
    residual_indices: np.ndarray | None = None


@dataclass
class PackedTensor:
    shape: tuple[int, ...]
    variant: str
    blocks: list[PackedBlock]
    mse: float
    max_abs_error: float

    @property
    def num_values(self) -> int:
        return int(np.prod(self.shape))

