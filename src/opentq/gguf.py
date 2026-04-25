from __future__ import annotations

import json
from pathlib import Path
from typing import Any


VARIANT_GGUF_TYPES = {
    "TQ1_0": "OPENTQ_TQ1_0",
    "TQ2_0": "OPENTQ_TQ2_0",
    "TQ3_SB4": "OPENTQ_TQ3_SB4",
    "TQ4_SB2": "OPENTQ_TQ4_SB2",
    "TQ4_SB4": "OPENTQ_TQ4_SB4",
    "TQ4R2": "OPENTQ_TQ4R2",
    "TQ4R4": "OPENTQ_TQ4R4",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def build_gguf_plan(packed_dir: str | Path) -> dict[str, Any]:
    packed = Path(packed_dir)
    pack_manifest = load_json(packed / "opentq-pack.json")
    tensors = []
    used_types = set()
    for tensor in pack_manifest["tensors"]:
        if tensor["mode"] == "copy":
            gguf_type = "F16"
        else:
            gguf_type = VARIANT_GGUF_TYPES.get(tensor["variant_name"], f"OPENTQ_{tensor['variant_name']}")
            used_types.add(gguf_type)
        tensors.append(
            {
                "name": tensor["name"],
                "mode": tensor["mode"],
                "variant_name": tensor.get("variant_name"),
                "gguf_type": gguf_type,
                "shape": tensor["shape"],
                "source_file": tensor["file"],
                "bytes": tensor["bytes"],
            }
        )

    return {
        "schema": "opentq.gguf_plan.v1",
        "status": "requires llama.cpp patchset",
        "model_id": pack_manifest["model_id"],
        "release_slug": pack_manifest["release_slug"],
        "source_pack_manifest": str(packed / "opentq-pack.json"),
        "required_metadata": {
            "general.architecture": "qwen3",
            "general.quantization_version": 1,
            "opentq.schema": pack_manifest["schema"],
            "opentq.release_slug": pack_manifest["release_slug"],
            "opentq.pack_manifest_sha256": pack_manifest["source_manifest_sha256"],
        },
        "custom_tensor_types": sorted(used_types),
        "patch_points": [
            "ggml_type enum entries for each OPENTQ_* type",
            "type-size and block-size tables",
            "dequantize_row implementations",
            "Metal unpack/dequant kernels",
            "GGUF writer that embeds .otq tensor payloads under the mapped type",
            "loader validation for opentq.* metadata",
        ],
        "tensors": tensors,
    }


def write_gguf_plan(packed_dir: str | Path, output: str | Path) -> dict[str, Any]:
    plan = build_gguf_plan(packed_dir)
    dump_json(Path(output), plan)
    return plan
