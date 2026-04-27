from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from .hf_release import dump_json, human_gib, link_file


def sha256_file(path: Path, chunk_size: int = 32 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def infer_variant_slug(path: Path) -> str:
    stem = path.name
    if stem.endswith(".gguf"):
        stem = stem[:-5]
    return stem


def gguf_model_card(
    *,
    repo_id: str,
    base_model: str,
    gguf_name: str,
    gguf_size: int,
    gguf_sha256: str,
    runtime_repo: str,
    text_only: bool,
) -> str:
    modality = "text-only" if text_only else "text + vision"
    return f"""---
base_model: {base_model}
tags:
- gguf
- llama.cpp
- opentq
- quantized
- qwen3.6
pipeline_tag: text-generation
---

# {gguf_name}

This is the public GGUF release for `{base_model}` using OpenTQ weight quantization.

## Runtime

- Format: GGUF with custom OpenTQ tensor types
- Runtime: patched `llama.cpp` / `{runtime_repo}`
- Stock upstream `llama.cpp`: not supported until the OpenTQ tensor types are upstreamed
- Modality in this release: {modality}

The OpenTQ `.otq` research packs are not part of this Hugging Face release. The published artifact is the GGUF file only.

## File

- GGUF: `{gguf_name}.gguf`
- Size: {human_gib(gguf_size)}
- SHA256: `{gguf_sha256}`

## Build Runtime

```bash
git clone {runtime_repo}
cd llama.cpp
cmake -B build -DGGML_METAL=ON -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=ON
cmake --build build -j
```

## Run

```bash
./build/bin/llama-cli \\
  -m {gguf_name}.gguf \\
  -ngl 99 \\
  -fa \\
  -c 8192 \\
  -p "Explain the tradeoff between prefill and decode throughput."
```

## Benchmark

Measure wall-clock total time, not only decode `tok/s`:

```bash
./build/bin/llama-bench -m {gguf_name}.gguf -ngl 99 -fa 1 -p 8192 -n 128
```

Target repo: `{repo_id}`
"""


def prepare_hf_gguf_release(
    gguf: str | Path,
    output_dir: str | Path,
    repo_id: str,
    *,
    base_model: str = "Qwen/Qwen3.6-27B",
    runtime_repo: str = "https://github.com/zlaabsi/llama.cpp-opentq",
    link_mode: str = "hardlink",
    text_only: bool = True,
    compute_sha256: bool = True,
) -> dict[str, Any]:
    gguf_path = Path(gguf)
    if not gguf_path.exists():
        raise FileNotFoundError(f"missing GGUF artifact: {gguf_path}")
    if gguf_path.suffix != ".gguf":
        raise ValueError(f"expected a .gguf artifact: {gguf_path}")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    target_name = gguf_path.name
    link_file(gguf_path, output / target_name, link_mode)

    size = gguf_path.stat().st_size
    digest = sha256_file(gguf_path) if compute_sha256 else "not-computed"
    slug = infer_variant_slug(gguf_path)
    summary = {
        "schema": "opentq.hf_gguf_release.v1",
        "repo_id": repo_id,
        "base_model": base_model,
        "artifact": {
            "path": str(gguf_path),
            "filename": target_name,
            "bytes": size,
            "gib": round(size / (1024**3), 2),
            "sha256": digest,
        },
        "runtime": {
            "repo": runtime_repo,
            "stock_llama_cpp": "unsupported until OpenTQ tensor types are upstreamed",
            "requires": "llama.cpp-opentq",
        },
        "release": {
            "text_only": text_only,
            "public_files": [target_name, "README.md", "opentq-gguf-release.json"],
            "excluded_private_artifacts": ["*.otq", "opentq-pack.json"],
        },
        "upload": {
            "large_folder": f"hf upload-large-folder {repo_id} {output}",
            "standard": f"hf upload {repo_id} {output} .",
        },
    }
    dump_json(output / "opentq-gguf-release.json", summary)
    (output / "README.md").write_text(
        gguf_model_card(
            repo_id=repo_id,
            base_model=base_model,
            gguf_name=slug,
            gguf_size=size,
            gguf_sha256=digest,
            runtime_repo=runtime_repo,
            text_only=text_only,
        ),
        encoding="utf-8",
    )
    return summary


def clear_hf_gguf_stage(output_dir: str | Path) -> None:
    output = Path(output_dir)
    if output.exists():
        shutil.rmtree(output)
