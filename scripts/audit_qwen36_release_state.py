from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


CURRENT_DYNAMIC_PROFILES = {"OTQ-DYN-Q3_K_M", "OTQ-DYN-Q4_K_M"}
FUTURE_DYNAMIC_PROFILES = {"OTQ-DYN-Q5_K_M"}
DEFERRED_DYNAMIC_PROFILES = {"OTQ-DYN-IQ4_NL"}


def audit_public_names(names: list[str]) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    for name in names:
        if "XL" in name:
            findings.append(
                {
                    "severity": "error",
                    "name": name,
                    "message": "Public name contains forbidden XL profile label.",
                }
            )
        if "OTQ3" in name or "OTQ4" in name:
            findings.append(
                {
                    "severity": "error",
                    "name": name,
                    "message": "Use OTQ-TQ* for branded native artifacts; OTQ3/OTQ4 is ambiguous.",
                }
            )
    return findings


def classify_dynamic_profile(profile: str) -> str:
    if profile in CURRENT_DYNAMIC_PROFILES:
        return "current-release"
    if profile in FUTURE_DYNAMIC_PROFILES:
        return "future-candidate"
    if profile in DEFERRED_DYNAMIC_PROFILES:
        return "deferred-imatrix-experiment"
    return "unknown"


def build_artifact_inventory(root: Path) -> dict[str, Any]:
    files = [path for path in root.rglob("*") if path.is_file()]
    gguf_files = [
        {"path": str(path), "name": path.name, "bytes": path.stat().st_size}
        for path in sorted(files)
        if path.suffix == ".gguf"
    ]
    return {
        "root": str(root),
        "file_count": len(files),
        "total_bytes": sum(path.stat().st_size for path in files),
        "gguf_files": gguf_files,
        "naming_findings": audit_public_names([path.name for path in files]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Qwen3.6 OpenTQ release state.")
    parser.add_argument("--root", type=Path, default=Path("artifacts/hf-gguf-canonical/Qwen3.6-27B-OTQ-GGUF"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/release-audit/qwen36-release-audit.json"))
    args = parser.parse_args()

    report = build_artifact_inventory(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
