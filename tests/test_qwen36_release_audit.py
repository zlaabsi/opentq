from __future__ import annotations

import json
from pathlib import Path

from scripts.audit_qwen36_release_state import (
    audit_public_names,
    build_artifact_inventory,
    classify_dynamic_profile,
)


def test_audit_public_names_rejects_xl_and_otq3() -> None:
    names = [
        "Qwen3.6-27B-OTQ-DYN-Q3_K_M.gguf",
        "Qwen3.6-27B-OTQ-DYN-Q3_XL-Q3_K_M.gguf",
        "Qwen3.6-27B-OTQ3_SB4",
        "XLAHooksInterface.h",
    ]

    findings = audit_public_names(names)

    assert findings == [
        {
            "severity": "error",
            "name": "Qwen3.6-27B-OTQ-DYN-Q3_XL-Q3_K_M.gguf",
            "message": "Public name contains forbidden XL profile label.",
        },
        {
            "severity": "error",
            "name": "Qwen3.6-27B-OTQ3_SB4",
            "message": "Use OTQ-TQ* for branded native artifacts; OTQ3/OTQ4 is ambiguous.",
        },
    ]


def test_classify_dynamic_profile_separates_release_and_experiment() -> None:
    assert classify_dynamic_profile("OTQ-DYN-Q3_K_M") == "current-release"
    assert classify_dynamic_profile("OTQ-DYN-Q4_K_M") == "current-release"
    assert classify_dynamic_profile("OTQ-DYN-Q5_K_M") == "future-candidate"
    assert classify_dynamic_profile("OTQ-DYN-IQ4_NL") == "deferred-imatrix-experiment"


def test_build_artifact_inventory_counts_files_and_bytes(tmp_path: Path) -> None:
    artifact = tmp_path / "Qwen3.6-27B-OTQ-DYN-Q3_K_M.gguf"
    artifact.write_bytes(b"123456")
    evidence = tmp_path / "evidence" / "Q3_K_M"
    evidence.mkdir(parents=True)
    (evidence / "validation.json").write_text(json.dumps({"passed": True}), encoding="utf-8")

    inventory = build_artifact_inventory(tmp_path)

    assert inventory["root"] == str(tmp_path)
    assert inventory["file_count"] == 2
    assert inventory["total_bytes"] == 6 + len(json.dumps({"passed": True}))
    assert inventory["gguf_files"] == [
        {
            "path": str(artifact),
            "name": "Qwen3.6-27B-OTQ-DYN-Q3_K_M.gguf",
            "bytes": 6,
        }
    ]
