from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def human_gib(value: int) -> str:
    return f"{value / (1024**3):.2f} GiB"


def link_file(source: Path, destination: Path, mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    if mode == "copy":
        shutil.copy2(source, destination)
    elif mode == "symlink":
        destination.symlink_to(source)
    elif mode == "hardlink":
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)
    elif mode == "none":
        return
    else:
        raise ValueError(f"unknown link mode: {mode}")


def model_card(pack_manifest: dict[str, Any], repo_id: str) -> str:
    totals = pack_manifest["totals"]
    release_slug = pack_manifest["release_slug"]
    return f"""---
base_model: {pack_manifest["model_id"]}
tags:
- opentq
- quantized
- gguf-ready
- qwen3.6
---

# {release_slug}

This repository is a staged OpenTQ weight-quantized release for `{pack_manifest["model_id"]}`.

## Status

- Format: `OpenTQ` packed tensor payloads
- Runtime: requires an OpenTQ-aware loader or `llama.cpp` patchset
- Stock `llama.cpp`: not supported until the custom OpenTQ tensor types and dequant kernels are registered
- Target repo: `{repo_id}`

## Size

- Packed payload: {human_gib(int(totals["payload_bytes"]))}
- Tensors: {totals["tensors"]}
- Quantized tensors: {totals["quantized_tensors"]}
- Copied tensors: {totals["copied_tensors"]}
- Weight values: {totals["values"]:,}

## Files

- `opentq-pack.json`: tensor index, payload offsets, shapes, variants and checksums
- `tensors/*.otq`: bit-packed OpenTQ tensor payloads
- `opentq-release.json`: release summary for tooling

## Upload

```bash
hf upload-large-folder {repo_id} .
```

For smaller test uploads, `hf upload {repo_id} .` is also valid.
"""


def prepare_hf_release(packed_dir: str | Path, output_dir: str | Path, repo_id: str, *, link_mode: str = "hardlink") -> dict[str, Any]:
    packed = Path(packed_dir)
    output = Path(output_dir)
    pack_manifest_path = packed / "opentq-pack.json"
    if not pack_manifest_path.exists():
        raise FileNotFoundError(f"missing pack manifest: {pack_manifest_path}")
    pack_manifest = load_json(pack_manifest_path)
    output.mkdir(parents=True, exist_ok=True)

    linked_files = 0
    if link_mode != "none":
        link_file(pack_manifest_path, output / "opentq-pack.json", link_mode)
        linked_files += 1
        for source in sorted((packed / "tensors").glob("*.otq")):
            link_file(source, output / "tensors" / source.name, link_mode)
            linked_files += 1
    else:
        shutil.copy2(pack_manifest_path, output / "opentq-pack.json")
        linked_files += 1

    summary = {
        "schema": "opentq.hf_release.v1",
        "repo_id": repo_id,
        "release_slug": pack_manifest["release_slug"],
        "model_id": pack_manifest["model_id"],
        "packed_dir": str(packed),
        "link_mode": link_mode,
        "linked_files": linked_files,
        "upload": {
            "large_folder": f"hf upload-large-folder {repo_id} {output}",
            "standard": f"hf upload {repo_id} {output} .",
        },
        "runtime_status": "requires OpenTQ runtime or llama.cpp patchset",
    }
    dump_json(output / "opentq-release.json", summary)
    (output / "README.md").write_text(model_card(pack_manifest, repo_id), encoding="utf-8")
    return summary
