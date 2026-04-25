from opentq.variants import get_variant


def test_residual_variants_count_primary_and_residual_scale_overhead() -> None:
    assert get_variant("TQ4_SB4").estimated_bpw() == 6.0
    assert get_variant("TQ4R2").estimated_bpw() == 10.0
    assert get_variant("TQ4R4").estimated_bpw() == 12.0
