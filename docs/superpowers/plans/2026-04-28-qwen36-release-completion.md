# Qwen3.6 Release Completion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish the Qwen3.6-27B OpenTQ release gates for stock GGUF, Packed, Metal/custom-runtime, hardware compatibility, and safe cleanup.

**Architecture:** Add small release-audit utilities that produce machine-readable JSON and human-readable Markdown, then wire their outputs into existing HF staging scripts. Keep publication and deletion as separate explicit gates; no script will upload or delete by default.

**Tech Stack:** Python 3.11+, `uv`, `pytest`, `huggingface_hub`, local `llama.cpp` binaries, existing OpenTQ scripts under `scripts/`.

---

## Files

- Create: `scripts/audit_qwen36_release_state.py`  
  Local/HF release-state audit: naming, artifacts, evidence, repo existence, and cleanup candidates.
- Create: `scripts/run_qwen36_runtime_checks.py`  
  Runtime validator for GGUF files using `llama-cli` and `llama-bench`.
- Create: `scripts/build_qwen36_cleanup_manifest.py`  
  Disk cleanup manifest generator with hardlink/inode awareness and no deletion behavior.
- Modify: `scripts/stage_qwen36_otq_gguf_repo.py`  
  Add hardware compatibility and clarify future `Q5_K_M` / deferred `IQ4_NL`.
- Modify: `scripts/stage_qwen36_otq_runtime_repos.py`  
  Strengthen Packed and Metal release-status wording and gate output.
- Modify: `docs/dynamic-compatible-gguf.md`  
  Clarify `IQ4_NL` as stock llama.cpp nonlinear 4-bit and not a primary release target.
- Modify: `docs/release-todo.md`  
  Replace loose checklist wording with gated next actions.
- Create: `tests/test_qwen36_release_audit.py`
- Create: `tests/test_qwen36_runtime_checks.py`
- Create: `tests/test_qwen36_cleanup_manifest.py`
- Modify: `tests/test_hf_gguf_release.py`
- Modify: `tests/test_dynamic_gguf.py`

## Task 1: Release Audit Script

**Files:**
- Create: `scripts/audit_qwen36_release_state.py`
- Test: `tests/test_qwen36_release_audit.py`

- [ ] **Step 1: Write failing tests for naming and artifact audit**

```python
# tests/test_qwen36_release_audit.py
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
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
uv run pytest tests/test_qwen36_release_audit.py -q
```

Expected: FAIL because `scripts.audit_qwen36_release_state` does not exist.

- [ ] **Step 3: Implement minimal audit script**

```python
# scripts/audit_qwen36_release_state.py
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
```

- [ ] **Step 4: Run audit tests**

Run:

```bash
uv run pytest tests/test_qwen36_release_audit.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/audit_qwen36_release_state.py tests/test_qwen36_release_audit.py
git commit -m "tools: add qwen36 release audit"
```

## Task 2: Runtime Check Harness

**Files:**
- Create: `scripts/run_qwen36_runtime_checks.py`
- Test: `tests/test_qwen36_runtime_checks.py`

- [ ] **Step 1: Write failing tests for command construction and JSON output**

```python
# tests/test_qwen36_runtime_checks.py
from __future__ import annotations

import json
from pathlib import Path

from scripts.run_qwen36_runtime_checks import RuntimeConfig, build_bench_command, build_cli_command, write_runtime_result


def test_build_cli_command_uses_metal_fa_and_prompt(tmp_path: Path) -> None:
    model = tmp_path / "model.gguf"
    config = RuntimeConfig(llama_cpp=tmp_path / "llama.cpp", threads=8, ctx_size=8192, predict=32)

    command = build_cli_command(model, "Paris?", config)

    assert command == [
        str(tmp_path / "llama.cpp" / "build" / "bin" / "llama-cli"),
        "-m",
        str(model),
        "-p",
        "Paris?",
        "-n",
        "32",
        "-c",
        "8192",
        "-t",
        "8",
        "-ngl",
        "99",
        "-fa",
    ]


def test_build_bench_command_sets_prefill_and_decode(tmp_path: Path) -> None:
    model = tmp_path / "model.gguf"
    config = RuntimeConfig(llama_cpp=tmp_path / "llama.cpp", threads=8, ctx_size=8192, predict=32)

    command = build_bench_command(model, config)

    assert command[:4] == [str(tmp_path / "llama.cpp" / "build" / "bin" / "llama-bench"), "-m", str(model), "-p"]
    assert "8192" in command
    assert "128" in command
    assert "-fa" in command


def test_write_runtime_result_records_evidence(tmp_path: Path) -> None:
    output = tmp_path / "runtime.json"
    write_runtime_result(
        output,
        {
            "model": "Qwen3.6-27B-OTQ-DYN-Q3_K_M.gguf",
            "machine": "M1 Max 32GB",
            "bounded_generation_passed": True,
        },
    )

    assert json.loads(output.read_text(encoding="utf-8")) == {
        "model": "Qwen3.6-27B-OTQ-DYN-Q3_K_M.gguf",
        "machine": "M1 Max 32GB",
        "bounded_generation_passed": True,
    }
```

- [ ] **Step 2: Run tests and verify they fail**

```bash
uv run pytest tests/test_qwen36_runtime_checks.py -q
```

Expected: FAIL because the module does not exist.

- [ ] **Step 3: Implement runtime harness**

```python
# scripts/run_qwen36_runtime_checks.py
from __future__ import annotations

import argparse
import json
import platform
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RuntimeConfig:
    llama_cpp: Path
    threads: int = 8
    ctx_size: int = 8192
    predict: int = 64


def build_cli_command(model: Path, prompt: str, config: RuntimeConfig) -> list[str]:
    return [
        str(config.llama_cpp / "build" / "bin" / "llama-cli"),
        "-m",
        str(model),
        "-p",
        prompt,
        "-n",
        str(config.predict),
        "-c",
        str(config.ctx_size),
        "-t",
        str(config.threads),
        "-ngl",
        "99",
        "-fa",
    ]


def build_bench_command(model: Path, config: RuntimeConfig) -> list[str]:
    return [
        str(config.llama_cpp / "build" / "bin" / "llama-bench"),
        "-m",
        str(model),
        "-p",
        str(config.ctx_size),
        "-n",
        "128",
        "-t",
        str(config.threads),
        "-ngl",
        "99",
        "-fa",
    ]


def write_runtime_result(output: Path, payload: dict[str, Any]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_command(command: list[str]) -> dict[str, Any]:
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local Qwen3.6 GGUF runtime checks.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--llama-cpp", type=Path, default=Path("/Users/zlaabsi/Documents/GitHub/llama.cpp"))
    parser.add_argument("--machine", default="M1 Max 32GB")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt", default="Réponds uniquement par la capitale de la France.")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--ctx-size", type=int, default=8192)
    parser.add_argument("--predict", type=int, default=64)
    args = parser.parse_args()

    config = RuntimeConfig(llama_cpp=args.llama_cpp, threads=args.threads, ctx_size=args.ctx_size, predict=args.predict)
    cli_result = run_command(build_cli_command(args.model, args.prompt, config))
    bench_result = run_command(build_bench_command(args.model, config))
    payload = {
        "model": str(args.model),
        "machine": args.machine,
        "platform": platform.platform(),
        "bounded_generation_passed": cli_result["returncode"] == 0,
        "bench_passed": bench_result["returncode"] == 0,
        "cli": cli_result,
        "bench": bench_result,
    }
    write_runtime_result(args.output, payload)
    raise SystemExit(0 if payload["bounded_generation_passed"] and payload["bench_passed"] else 1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run runtime harness tests**

```bash
uv run pytest tests/test_qwen36_runtime_checks.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/run_qwen36_runtime_checks.py tests/test_qwen36_runtime_checks.py
git commit -m "tools: add qwen36 runtime checks"
```

## Task 3: Hardware Compatibility In HF Card

**Files:**
- Modify: `scripts/stage_qwen36_otq_gguf_repo.py`
- Modify: `tests/test_hf_gguf_release.py`

- [ ] **Step 1: Add failing README assertions**

Append to `tests/test_hf_gguf_release.py`:

```python
def test_canonical_readme_contains_hardware_compatibility_table(tmp_path: Path) -> None:
    from scripts.stage_qwen36_otq_gguf_repo import hardware_compatibility_markdown

    markdown = hardware_compatibility_markdown()

    assert "| Hardware | Status | Recommended artifact | Notes |" in markdown
    assert "| M1 Max 32 GB | Measured | `Q3_K_M`" in markdown
    assert "| 16 GB Apple Silicon | Not recommended | None" in markdown
    assert "Expected rows are capacity guidance, not measured benchmark claims." in markdown
```

- [ ] **Step 2: Run test and verify failure**

```bash
uv run pytest tests/test_hf_gguf_release.py::test_canonical_readme_contains_hardware_compatibility_table -q
```

Expected: FAIL because `hardware_compatibility_markdown` does not exist.

- [ ] **Step 3: Implement hardware table helper**

Add to `scripts/stage_qwen36_otq_gguf_repo.py` near the README helpers:

```python
def hardware_compatibility_markdown() -> str:
    rows = [
        ("M1 Max 32 GB", "Measured", "`Q3_K_M`; `Q4_K_M` with tighter context", "Local release validation target."),
        ("32 GB Apple Silicon", "Expected", "`Q3_K_M`", "Capacity guidance for M-series systems with similar usable unified memory."),
        ("48 GB Apple Silicon", "Expected", "`Q4_K_M`; future `Q5_K_M` after generation", "No benchmark claim until measured."),
        ("64 GB+ Apple Silicon", "Expected", "`Q4_K_M`; larger native/custom candidates after runtime gates", "Quality-first track once artifacts are validated."),
        ("16 GB Apple Silicon", "Not recommended", "None", "Current 27B artifacts leave too little memory headroom."),
    ]
    lines = [
        "## Hardware Compatibility",
        "",
        "| Hardware | Status | Recommended artifact | Notes |",
        "| --- | --- | --- | --- |",
    ]
    lines.extend(f"| {hardware} | {status} | {artifact} | {notes} |" for hardware, status, artifact, notes in rows)
    lines.extend(
        [
            "",
            "Expected rows are capacity guidance, not measured benchmark claims.",
            "",
        ]
    )
    return "\n".join(lines)
```

Then insert `{hardware_compatibility_markdown()}` into the canonical README before the runtime section.

- [ ] **Step 4: Run targeted test**

```bash
uv run pytest tests/test_hf_gguf_release.py::test_canonical_readme_contains_hardware_compatibility_table -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/stage_qwen36_otq_gguf_repo.py tests/test_hf_gguf_release.py
git commit -m "docs: add qwen36 hardware compatibility table"
```

## Task 4: IQ4_NL And Q5_K_M Wording

**Files:**
- Modify: `docs/dynamic-compatible-gguf.md`
- Modify: `docs/release-todo.md`
- Modify: `tests/test_dynamic_gguf.py`

- [ ] **Step 1: Add failing docs/naming test**

Append to `tests/test_dynamic_gguf.py`:

```python
def test_dynamic_docs_explain_iq4_nl_and_do_not_use_xl() -> None:
    text = Path("docs/dynamic-compatible-gguf.md").read_text(encoding="utf-8")

    assert "XL" not in text
    assert "IQ4_NL is a stock llama.cpp nonlinear 4-bit quant type" in text
    assert "requires an imatrix before release consideration" in text
```

- [ ] **Step 2: Run test and verify failure if wording is missing**

```bash
uv run pytest tests/test_dynamic_gguf.py::test_dynamic_docs_explain_iq4_nl_and_do_not_use_xl -q
```

Expected: FAIL until docs contain exact wording.

- [ ] **Step 3: Patch docs**

Update `docs/dynamic-compatible-gguf.md` profile text so `OTQ-DYN-IQ4_NL` says:

```markdown
`IQ4_NL` is a stock llama.cpp nonlinear 4-bit quant type, reported by llama.cpp as roughly 4.5 bpw. In OpenTQ release planning it is a deferred experiment because it requires an imatrix before release consideration.
```

Update `docs/release-todo.md` so future stock candidates say:

```markdown
- `Q5_K_M`: future quality-first stock GGUF candidate after disk cleanup and runtime gates.
- `IQ4_NL`: deferred stock llama.cpp nonlinear 4-bit experiment; do not publish without imatrix evidence.
```

- [ ] **Step 4: Run targeted docs test**

```bash
uv run pytest tests/test_dynamic_gguf.py::test_dynamic_docs_explain_iq4_nl_and_do_not_use_xl -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add docs/dynamic-compatible-gguf.md docs/release-todo.md tests/test_dynamic_gguf.py
git commit -m "docs: clarify future stock gguf candidates"
```

## Task 5: Cleanup Manifest Generator

**Files:**
- Create: `scripts/build_qwen36_cleanup_manifest.py`
- Test: `tests/test_qwen36_cleanup_manifest.py`

- [ ] **Step 1: Write failing tests for classification and hardlinks**

```python
# tests/test_qwen36_cleanup_manifest.py
from __future__ import annotations

from pathlib import Path

from scripts.build_qwen36_cleanup_manifest import classify_path, inspect_path


def test_classify_path_marks_uploaded_verified() -> None:
    assert classify_path("artifacts/hf-gguf-canonical/Qwen3.6-27B-OTQ-GGUF", True, True) == "uploaded-verified"
    assert classify_path("artifacts/qwen3.6-27b-source/Qwen3.6-27B-BF16.gguf", False, False) == "blocked"
    assert classify_path("artifacts/tmp", False, True) == "regenerable"


def test_inspect_path_reports_inode_and_links(tmp_path: Path) -> None:
    first = tmp_path / "first.bin"
    second = tmp_path / "second.bin"
    first.write_bytes(b"1234")
    second.hardlink_to(first)

    info = inspect_path(first)

    assert info["bytes"] == 4
    assert info["hardlink_count"] >= 2
    assert isinstance(info["inode"], int)
```

- [ ] **Step 2: Run tests and verify failure**

```bash
uv run pytest tests/test_qwen36_cleanup_manifest.py -q
```

Expected: FAIL because module does not exist.

- [ ] **Step 3: Implement cleanup manifest**

```python
# scripts/build_qwen36_cleanup_manifest.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


BLOCKED_NAMES = {"Qwen3.6-27B-BF16.gguf"}


def classify_path(path: str, uploaded: bool, regenerable: bool) -> str:
    name = Path(path).name
    if uploaded:
        return "uploaded-verified"
    if name in BLOCKED_NAMES:
        return "blocked"
    if regenerable:
        return "regenerable"
    return "investigate"


def inspect_path(path: Path) -> dict[str, Any]:
    stat = path.stat()
    if path.is_file():
        total = stat.st_size
    else:
        total = sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
    return {
        "path": str(path),
        "bytes": total,
        "inode": stat.st_ino,
        "hardlink_count": stat.st_nlink,
    }


def build_manifest(paths: list[Path]) -> list[dict[str, Any]]:
    records = []
    for path in paths:
        info = inspect_path(path)
        info["classification"] = classify_path(str(path), uploaded=False, regenerable=False)
        info["delete_command"] = None
        records.append(info)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Qwen3.6 cleanup manifest without deleting files.")
    parser.add_argument("--output", type=Path, default=Path("artifacts/release-audit/qwen36-cleanup-manifest.json"))
    parser.add_argument("paths", nargs="*", type=Path)
    args = parser.parse_args()

    paths = args.paths or [
        Path("artifacts/qwen3.6-27b"),
        Path("artifacts/hf-runtime"),
        Path("artifacts/qwen3.6-27b-source"),
        Path("artifacts/hf-gguf-canonical"),
        Path("artifacts/qwen3.6-27b-gguf"),
    ]
    existing = [path for path in paths if path.exists()]
    payload = build_manifest(existing)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run cleanup tests**

```bash
uv run pytest tests/test_qwen36_cleanup_manifest.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/build_qwen36_cleanup_manifest.py tests/test_qwen36_cleanup_manifest.py
git commit -m "tools: add qwen36 cleanup manifest"
```

## Task 6: Packed And Metal Gate Wording

**Files:**
- Modify: `scripts/stage_qwen36_otq_runtime_repos.py`
- Create: `tests/test_qwen36_runtime_repos.py`

- [ ] **Step 1: Write failing tests for release status text**

```python
# tests/test_qwen36_runtime_repos.py
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
```

- [ ] **Step 2: Run test and verify failure**

```bash
uv run pytest tests/test_qwen36_runtime_repos.py -q
```

Expected: FAIL because helpers do not exist.

- [ ] **Step 3: Add gate helpers and insert in README generation**

Add to `scripts/stage_qwen36_otq_runtime_repos.py`:

```python
def packed_gate_markdown() -> str:
    return "\n".join(
        [
            "## Release Boundary",
            "",
            "`OTQ-Packed` is not a stock llama.cpp inference release. It contains OpenTQ `.otq` payloads and `opentq-pack.json` manifests for runtime/tooling integration.",
            "",
        ]
    )


def metal_gate_markdown() -> str:
    return "\n".join(
        [
            "## Metal Runtime Gate",
            "",
            "`TQ3_SB4` is the first candidate for the required OpenTQ/Metal runtime path.",
            "`TQ4_SB4` remains blocked until the inconsistent GGUF export size is audited.",
            "`TQ4_BAL_V2`, `TQ4R2`, and `TQ4R4` remain blocked until fresh runtime validation exists.",
            "",
        ]
    )
```

Insert `packed_gate_markdown()` in the Packed README and `metal_gate_markdown()` in the Metal README.

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_qwen36_runtime_repos.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/stage_qwen36_otq_runtime_repos.py tests/test_qwen36_runtime_repos.py
git commit -m "docs: tighten packed and metal release gates"
```

## Task 7: Run Local Release Evidence

**Files:**
- Runtime outputs under `artifacts/release-audit/`
- No code changes expected unless previous tasks expose bugs.

- [ ] **Step 1: Generate release audit JSON**

```bash
uv run python scripts/audit_qwen36_release_state.py \
  --root artifacts/hf-gguf-canonical/Qwen3.6-27B-OTQ-GGUF \
  --output artifacts/release-audit/qwen36-release-audit.json
```

Expected: JSON exists and `naming_findings` is empty for public files.

- [ ] **Step 2: Run runtime check for Q3_K_M**

```bash
uv run python scripts/run_qwen36_runtime_checks.py \
  --model artifacts/hf-gguf-canonical/Qwen3.6-27B-OTQ-GGUF/Qwen3.6-27B-OTQ-DYN-Q3_K_M.gguf \
  --output artifacts/release-audit/runtime-Q3_K_M-M1_Max_32GB.json
```

Expected: command exits 0 and JSON has `bounded_generation_passed: true`, `bench_passed: true`.

- [ ] **Step 3: Run runtime check for Q4_K_M**

```bash
uv run python scripts/run_qwen36_runtime_checks.py \
  --model artifacts/hf-gguf-canonical/Qwen3.6-27B-OTQ-GGUF/Qwen3.6-27B-OTQ-DYN-Q4_K_M.gguf \
  --output artifacts/release-audit/runtime-Q4_K_M-M1_Max_32GB.json
```

Expected: command exits 0 and JSON records runtime evidence or a clear failure to document.

- [ ] **Step 4: Generate cleanup manifest**

```bash
uv run python scripts/build_qwen36_cleanup_manifest.py \
  --output artifacts/release-audit/qwen36-cleanup-manifest.json
```

Expected: manifest exists and contains no deletion commands.

- [ ] **Step 5: Do not commit large artifacts**

```bash
git status --short
```

Expected: generated files under `artifacts/` are untracked or ignored. Do not add them to git.

## Task 8: Regenerate HF Staging And Review

**Files:**
- Generated staging under `artifacts/hf-gguf-canonical/Qwen3.6-27B-OTQ-GGUF`
- No source commit unless generation reveals missing source changes.

- [ ] **Step 1: Regenerate canonical HF staging**

```bash
uv run python scripts/stage_qwen36_otq_gguf_repo.py \
  --banner "/Users/zlaabsi/Downloads/ChatGPT Image Apr 28, 2026, 01_45_35 AM.png"
```

Expected: staging completes and README contains hardware compatibility.

- [ ] **Step 2: Search generated public files for forbidden names**

```bash
rg -n "XL|OTQ3|OTQ4" artifacts/hf-gguf-canonical/Qwen3.6-27B-OTQ-GGUF
```

Expected: no matches.

- [ ] **Step 3: Verify current source tests**

```bash
uv run pytest \
  tests/test_dynamic_gguf.py \
  tests/test_hf_gguf_release.py \
  tests/test_qwen36_release_audit.py \
  tests/test_qwen36_runtime_checks.py \
  tests/test_qwen36_cleanup_manifest.py \
  tests/test_qwen36_runtime_repos.py \
  -q
```

Expected: PASS.

- [ ] **Step 4: Upload only after manual review**

Do not upload in this task. HF upload is a separate gate after README, artifacts, and runtime JSONs are reviewed.

## Task 9: Claude Code Worker Use

**Files:**
- No direct source ownership for Claude Code unless a patch is explicitly reviewed.

- [ ] **Step 1: Run Claude Code only for a read-only naming audit**

```bash
claude -p \
  --permission-mode plan \
  --max-budget-usd 1 \
  "Read the OpenTQ repo. Report only public naming violations involving XL, OTQ3, OTQ4, Native as a public suffix, or custom OpenTQ GGUFs described as stock llama.cpp. Do not edit files."
```

Expected: plain-text report. Treat it as advisory.

- [ ] **Step 2: Review Claude output manually**

Check any reported files with:

```bash
rg -n "XL|OTQ3|OTQ4|OTQ-Native|stock llama.cpp" README.md docs scripts tests
```

Expected: only allowed historical/internal references or issues that Codex patches in a normal task.

- [ ] **Step 3: Do not accept Claude edits automatically**

If Claude is used with edit permissions in a separate approved step, review with:

```bash
git diff
uv run pytest tests/test_dynamic_gguf.py tests/test_hf_gguf_release.py -q
```

Expected: no unreviewed patch is committed.

## Final Verification

- [ ] **Step 1: Run targeted source tests**

```bash
uv run pytest \
  tests/test_dynamic_gguf.py \
  tests/test_hf_gguf_release.py \
  tests/test_qwen36_release_audit.py \
  tests/test_qwen36_runtime_checks.py \
  tests/test_qwen36_cleanup_manifest.py \
  tests/test_qwen36_runtime_repos.py \
  tests/test_recipes.py \
  tests/test_variants.py \
  tests/test_gguf.py \
  tests/test_gguf_export.py \
  -q
```

Expected: PASS.

- [ ] **Step 2: Check source tree**

```bash
git status --short --branch
git log --oneline --max-count=8
```

Expected: branch `qwen36-release-completion`, only intended commits.

- [ ] **Step 3: Summarize release gates**

Prepare a concise status table:

```text
OTQ-GGUF: validated / blocked reason
OTQ-Packed: staged / upload gated / blocked reason
OTQ-Metal-GGUF: candidate list / blocked variants
Cleanup: manifest generated / no deletion performed
Claude Code: used read-only / not used
```

Expected: user can decide upload and cleanup gates from evidence.
