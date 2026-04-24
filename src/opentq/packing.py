from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PackedTensor:
    shape: tuple[int, ...]
    variant: str
    indices: np.ndarray
    scales: np.ndarray
    residual_indices: np.ndarray | None
    residual_scales: np.ndarray | None
    mse: float
    max_abs_error: float
    sum_squared_error: float

    @property
    def num_values(self) -> int:
        return int(np.prod(self.shape))

    @property
    def num_blocks(self) -> int:
        return int(self.indices.shape[0])
