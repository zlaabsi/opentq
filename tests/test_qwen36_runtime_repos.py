from __future__ import annotations

from scripts.stage_qwen36_otq_runtime_repos import metal_gate_markdown, packed_gate_markdown


def test_packed_gate_markdown_says_not_stock_inference() -> None:
    text = packed_gate_markdown()

    assert "not a stock llama.cpp inference release" in text
    assert "opentq-pack.json" in text


def test_metal_gate_markdown_blocks_unvalidated_variants() -> None:
    text = metal_gate_markdown()

    assert "`TQ3_SB4` is the first candidate" in text
    assert "`TQ4_SB4` remains blocked until the inconsistent GGUF export size is audited" in text
    assert "required OpenTQ/Metal runtime" in text
