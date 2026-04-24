import numpy as np

from opentq.quantize import quantize_tensor
from opentq.variants import get_variant


def test_residual_quantization_improves_mse() -> None:
    tensor = np.linspace(-2.0, 2.0, 256, dtype=np.float32).reshape(16, 16)
    base = quantize_tensor(tensor, get_variant("TQ4_SB4"))
    residual = quantize_tensor(tensor, get_variant("TQ4R2"))
    assert residual.packed.mse <= base.packed.mse


def test_quantize_tensor_preserves_shape() -> None:
    tensor = np.random.default_rng(42).normal(size=(8, 8)).astype(np.float32)
    result = quantize_tensor(tensor, get_variant("TQ3_SB4"))
    assert result.reconstruction.shape == tensor.shape
    assert result.packed.shape == tensor.shape

