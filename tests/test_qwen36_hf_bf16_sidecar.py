from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPT = Path("scripts/run_qwen36_hf_bf16_sidecar.py")


def load_sidecar_module():
    spec = importlib.util.spec_from_file_location("run_qwen36_hf_bf16_sidecar", SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_sidecar_refuses_model_drift_by_default() -> None:
    module = load_sidecar_module()

    assert module.DEFAULT_MODEL_ID == "Qwen/Qwen3.6-27B"


def test_mode_configs_encode_low_cost_smoke_shape() -> None:
    module = load_sidecar_module()

    no_think = module.ModeConfig(
        name="no_think",
        prompt_format="qwen3-no-think",
        samples_per_family=4,
        max_tokens=512,
        token_policy="cap",
    )
    thinking = module.ModeConfig(
        name="thinking",
        prompt_format="qwen3-thinking",
        samples_per_family=2,
        max_tokens=2048,
        token_policy="floor",
    )

    assert no_think.token_policy == "cap"
    assert thinking.token_policy == "floor"


def test_configured_sample_caps_no_think_and_extends_thinking_tokens() -> None:
    module = load_sidecar_module()
    sample = {"prompt_format": "qwen3-no-think", "max_tokens": 16}

    no_think = module.configured_sample(
        None,
        sample,
        module.ModeConfig("no_think", "qwen3-no-think", 4, 512, "cap"),
    )
    thinking = module.configured_sample(
        None,
        sample,
        module.ModeConfig("thinking", "qwen3-thinking", 2, 2048, "floor"),
    )

    assert no_think["max_tokens"] == 16
    assert thinking["max_tokens"] == 2048
    assert thinking["prompt_format"] == "qwen3-thinking"


def test_empty_eval_payload_is_report_compatible() -> None:
    module = load_sidecar_module()

    class Runner:
        SCHEMA = "opentq.qwen36_benchmark_subset_eval.v1"

    payload = module.empty_eval_payload(
        runner=Runner,
        model_key="bf16_remote_no_think",
        model_id="Qwen/Qwen3.6-27B",
        model_revision="6a9e13bd6fc8f0983b9b99948120bc37f49c13e9",
        mode=module.ModeConfig("no_think", "qwen3-no-think", 4, 512, "cap"),
        benchmarks=[],
        load_seconds=None,
    )

    assert payload["schema"] == "opentq.qwen36_benchmark_subset_eval.v1"
    assert payload["model"]["key"] == "bf16_remote_no_think"
    assert payload["model"]["path"] == "Qwen/Qwen3.6-27B"
    assert payload["model"]["kind"] == "hf_transformers_bf16"
