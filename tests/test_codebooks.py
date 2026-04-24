from opentq.codebooks import lloyd_max_gaussian


def test_lloyd_max_gaussian_is_monotonic() -> None:
    codebook, boundaries = lloyd_max_gaussian(4)
    assert len(codebook) == 16
    assert len(boundaries) == 17
    assert all(codebook[i] < codebook[i + 1] for i in range(len(codebook) - 1))
    assert all(boundaries[i] < boundaries[i + 1] for i in range(len(boundaries) - 1))

