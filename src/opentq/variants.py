from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QuantVariant:
    name: str
    weight_bits: int
    residual_bits: int | None = None
    group_size: int = 128
    block_size: int = 32
    sub_block_size: int = 8
    sub_block_scales: int = 4
    use_wht: bool = True
    codebook: str = "lloyd-max"
    runtime_targets: tuple[str, ...] = ("llama.cpp", "opentq-metal")
    intended_use: str = "general"
    notes: str = ""

    @property
    def total_bits(self) -> int:
        return self.weight_bits + (self.residual_bits or 0)

    @property
    def has_residual(self) -> bool:
        return self.residual_bits is not None

    def estimated_bpw(self) -> float:
        scale_overhead = self.sub_block_scales * 16 / self.block_size
        return float(self.total_bits + scale_overhead)


VARIANTS: dict[str, QuantVariant] = {
    "TQ1_0": QuantVariant(
        name="TQ1_0",
        weight_bits=1,
        group_size=256,
        block_size=64,
        sub_block_size=16,
        sub_block_scales=4,
        use_wht=False,
        intended_use="minimum-footprint",
        notes="Ternary-leaning baseline for aggressive RAM budgets.",
    ),
    "TQ2_0": QuantVariant(
        name="TQ2_0",
        weight_bits=2,
        group_size=256,
        block_size=64,
        sub_block_size=16,
        sub_block_scales=4,
        intended_use="very-low-memory",
        notes="2-bit scalar codebook path, intended for 32 GB unified-memory Macs.",
    ),
    "TQ3_SB4": QuantVariant(
        name="TQ3_SB4",
        weight_bits=3,
        group_size=128,
        block_size=32,
        sub_block_size=8,
        sub_block_scales=4,
        intended_use="compact-general-purpose",
        notes="Closest open analogue to the public 3.5-bit WHT family, without opaque revision labels.",
    ),
    "TQ4_SB2": QuantVariant(
        name="TQ4_SB2",
        weight_bits=4,
        group_size=128,
        block_size=32,
        sub_block_size=16,
        sub_block_scales=2,
        intended_use="balanced-16gib",
        notes="4-bit WHT with two sub-block scales per 32-weight block; the first practical redesign for ~16 GiB Qwen3.6-27B releases.",
    ),
    "TQ4_SB4": QuantVariant(
        name="TQ4_SB4",
        weight_bits=4,
        group_size=128,
        block_size=32,
        sub_block_size=8,
        sub_block_scales=4,
        intended_use="daily-driver",
        notes="Primary quality/latency target for 27B on Apple Silicon.",
    ),
    "TQ4R2": QuantVariant(
        name="TQ4R2",
        weight_bits=4,
        residual_bits=2,
        group_size=128,
        block_size=32,
        sub_block_size=8,
        sub_block_scales=4,
        intended_use="quality-first",
        notes="Residual format tuned for a better quality/RAM point than plain 6-bit scalar quantization.",
    ),
    "TQ4R4": QuantVariant(
        name="TQ4R4",
        weight_bits=4,
        residual_bits=4,
        group_size=128,
        block_size=32,
        sub_block_size=8,
        sub_block_scales=4,
        intended_use="near-lossless",
        notes="Residual format for top-end Apple Silicon and regression baselines.",
    ),
    "TQ_MIX_MOE": QuantVariant(
        name="TQ_MIX_MOE",
        weight_bits=4,
        residual_bits=2,
        group_size=128,
        block_size=32,
        sub_block_size=8,
        sub_block_scales=4,
        intended_use="moe-mixed",
        notes="Release profile placeholder: experts aggressive, attention conservative, embeddings/output anchored high.",
    ),
}


def get_variant(name: str) -> QuantVariant:
    try:
        return VARIANTS[name.upper()]
    except KeyError as exc:
        available = ", ".join(sorted(VARIANTS))
        raise KeyError(f"unknown variant {name!r}; available: {available}") from exc
