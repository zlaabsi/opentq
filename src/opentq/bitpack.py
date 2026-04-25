from __future__ import annotations

import numpy as np


def _validate_bits(bits: int) -> None:
    if bits not in {1, 2, 3, 4, 8}:
        raise ValueError(f"unsupported bit width: {bits}")


def pack_bits(values: np.ndarray, bits: int) -> bytes:
    _validate_bits(bits)
    flat = np.asarray(values, dtype=np.uint8).reshape(-1)
    if flat.size == 0:
        return b""
    if int(flat.max()) >= (1 << bits):
        raise ValueError(f"value out of range for {bits}-bit packing")
    if bits == 8:
        return flat.tobytes()
    if bits == 4:
        padded = np.pad(flat, (0, (-flat.size) % 2), constant_values=0).reshape(-1, 2)
        return (padded[:, 0] | (padded[:, 1] << 4)).astype(np.uint8).tobytes()
    if bits == 2:
        padded = np.pad(flat, (0, (-flat.size) % 4), constant_values=0).reshape(-1, 4)
        packed = padded[:, 0] | (padded[:, 1] << 2) | (padded[:, 2] << 4) | (padded[:, 3] << 6)
        return packed.astype(np.uint8).tobytes()
    if bits == 1:
        padded = np.pad(flat, (0, (-flat.size) % 8), constant_values=0).reshape(-1, 8)
        shifts = np.arange(8, dtype=np.uint8)
        return np.sum(padded << shifts, axis=1, dtype=np.uint8).astype(np.uint8).tobytes()

    padded = np.pad(flat, (0, (-flat.size) % 8), constant_values=0).reshape(-1, 8).astype(np.uint32)
    accum = (
        padded[:, 0]
        | (padded[:, 1] << 3)
        | (padded[:, 2] << 6)
        | (padded[:, 3] << 9)
        | (padded[:, 4] << 12)
        | (padded[:, 5] << 15)
        | (padded[:, 6] << 18)
        | (padded[:, 7] << 21)
    )
    out = np.empty(accum.size * 3, dtype=np.uint8)
    out[0::3] = accum & 0xFF
    out[1::3] = (accum >> 8) & 0xFF
    out[2::3] = (accum >> 16) & 0xFF
    return out.tobytes()


def unpack_bits(payload: bytes, bits: int, count: int) -> np.ndarray:
    _validate_bits(bits)
    if count == 0:
        return np.array([], dtype=np.uint8)
    data = np.frombuffer(payload, dtype=np.uint8)
    if bits == 8:
        return data[:count].copy()
    if bits == 4:
        out = np.empty(data.size * 2, dtype=np.uint8)
        out[0::2] = data & 0x0F
        out[1::2] = data >> 4
        return out[:count]
    if bits == 2:
        out = np.empty(data.size * 4, dtype=np.uint8)
        out[0::4] = data & 0x03
        out[1::4] = (data >> 2) & 0x03
        out[2::4] = (data >> 4) & 0x03
        out[3::4] = (data >> 6) & 0x03
        return out[:count]
    if bits == 1:
        out = np.empty(data.size * 8, dtype=np.uint8)
        for shift in range(8):
            out[shift::8] = (data >> shift) & 0x01
        return out[:count]

    triples = np.pad(data, (0, (-data.size) % 3), constant_values=0).reshape(-1, 3).astype(np.uint32)
    accum = triples[:, 0] | (triples[:, 1] << 8) | (triples[:, 2] << 16)
    out = np.empty(accum.size * 8, dtype=np.uint8)
    for index in range(8):
        out[index::8] = (accum >> (index * 3)) & 0x07
    return out[:count]
