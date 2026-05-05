from __future__ import annotations

from scripts.stage_qwen36_benchmark_repro_dataset import readme, score_fields, viewer_scalar


def test_viewer_scalar_preserves_simple_values() -> None:
    assert viewer_scalar("A") == "A"
    assert viewer_scalar(3) == 3
    assert viewer_scalar(0.25) == 0.25
    assert viewer_scalar(True) is True
    assert viewer_scalar(None) is None


def test_viewer_scalar_serializes_complex_values() -> None:
    assert viewer_scalar(["A", "B"]) == '["A", "B"]'
    assert viewer_scalar({"choices": ["A", "B"]}) == '{"choices": ["A", "B"]}'


def test_score_fields_uses_viewer_stable_expected_value() -> None:
    row = score_fields(
        {
            "passed": True,
            "score": {"actual": "B", "expected": ["A", "B"]},
            "stdout_tail": {"raw": "B"},
            "elapsed_seconds": 1.25,
        }
    )

    assert row == {
        "passed": True,
        "actual": "B",
        "expected": '["A", "B"]',
        "stdout_tail": '{"raw": "B"}',
        "elapsed_seconds": 1.25,
    }


def test_dataset_card_pins_viewer_configs_to_jsonl_files() -> None:
    card = readme()

    assert "configs:" in card
    assert "config_name: paired_samples" in card
    assert "path: data/paired_samples.jsonl" in card
    assert "config_name: paired_summary" in card
    assert "path: data/paired_summary.jsonl" in card
