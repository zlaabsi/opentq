import numpy as np

from opentq.gguf_export import opentq_group_type_size, pack_group_record
from opentq.variants import get_variant


def test_opentq_gguf_type_sizes_match_llama_cpp_blocks() -> None:
    assert opentq_group_type_size(get_variant("TQ3_SB4")) == 84
    assert opentq_group_type_size(get_variant("TQ4_SB2")) == 84
    assert opentq_group_type_size(get_variant("TQ4_SB4")) == 100
    assert opentq_group_type_size(get_variant("TQ4R2")) == 164
    assert opentq_group_type_size(get_variant("TQ4R4")) == 196


def test_pack_group_record_uses_fixed_block_layout() -> None:
    variant = get_variant("TQ4R2")
    record = pack_group_record(
        123,
        np.arange(128, dtype=np.uint8) % 16,
        np.ones(16, dtype=np.float16),
        variant,
        np.arange(128, dtype=np.uint8) % 4,
        np.ones(16, dtype=np.float16),
    )

    assert len(record) == opentq_group_type_size(variant)
