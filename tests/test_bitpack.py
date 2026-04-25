import numpy as np
import pytest

from opentq.bitpack import pack_bits, unpack_bits


@pytest.mark.parametrize("bits", [1, 2, 3, 4, 8])
def test_pack_bits_roundtrip(bits: int) -> None:
    values = (np.arange(37, dtype=np.uint16) % (1 << bits)).astype(np.uint8)
    payload = pack_bits(values, bits)
    restored = unpack_bits(payload, bits, values.size)
    np.testing.assert_array_equal(restored, values)


def test_pack_bits_rejects_out_of_range_values() -> None:
    with pytest.raises(ValueError):
        pack_bits(np.array([0, 4], dtype=np.uint8), 2)
