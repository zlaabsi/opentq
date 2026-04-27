from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from .gguf_validate import assert_validation_matches
from .hf_release import dump_json, human_gib, link_file, load_json


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
    validation: dict[str, Any] | None,
    stock_compatible: bool = False,
    quality_eval: dict[str, Any] | None = None,
) -> str:
    modality = "text-only" if text_only else "text + vision"
    validation_status = "passed" if validation is not None else "not attached"
    validation_created_at = validation.get("created_at", "unknown") if validation else "n/a"
    quality_status = "passed" if quality_eval is not None and quality_eval.get("overall_pass") else "not attached"
    quality_summary = quality_eval.get("summary", {}) if quality_eval else {}
    if stock_compatible:
        runtime_section = f"""- Format: standard GGUF tensor types only
- Runtime: stock `llama.cpp` / `llama-server`
- Custom OpenTQ runtime: not required
- Modality in this release: {modality}

This artifact is the OpenTQ dynamic-compatible track: OpenTQ provides the tensor allocation policy and validation harness, while the GGUF payload uses existing llama.cpp quantization types.
"""
        build_section = """## Build Runtime

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -B build -DGGML_METAL=ON -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=ON
cmake --build build -j
```
"""
    else:
        runtime_section = f"""- Format: GGUF with custom OpenTQ tensor types
- Runtime: patched `llama.cpp` / `{runtime_repo}`
- Stock upstream `llama.cpp`: not supported until the OpenTQ tensor types are upstreamed
- Modality in this release: {modality}

The OpenTQ `.otq` research packs are not part of this Hugging Face release. The published artifact is the GGUF file only.
"""
        build_section = f"""## Build Runtime

```bash
git clone {runtime_repo}
cd llama.cpp
cmake -B build -DGGML_METAL=ON -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=ON
cmake --build build -j
```
"""
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

This is the public GGUF release for `{base_model}` using OpenTQ dynamic weight allocation.

## Runtime

{runtime_section}

## Validation

- Release gate: {validation_status}
- Validation date: {validation_created_at}
- Required checks: GGUF metadata read + bounded `llama-cli` generation
- Quality eval: {quality_status}
- Quality pass rate: {quality_summary.get("pass_rate", "n/a")}

## File

- GGUF: `{gguf_name}.gguf`
- Size: {human_gib(gguf_size)}
- SHA256: `{gguf_sha256}`

{build_section}

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
    validation_path: str | Path | None = None,
    require_validation: bool = True,
    require_benchmark: bool = True,
    min_benchmark_prompt_tokens: int = 8192,
    min_benchmark_gen_tokens: int = 128,
    stock_compatible: bool = False,
    quality_eval_path: str | Path | None = None,
) -> dict[str, Any]:
    gguf_path = Path(gguf)
    if not gguf_path.exists():
        raise FileNotFoundError(f"missing GGUF artifact: {gguf_path}")
    if gguf_path.suffix != ".gguf":
        raise ValueError(f"expected a .gguf artifact: {gguf_path}")
    validation: dict[str, Any] | None = None
    if validation_path is not None:
        validation = load_json(Path(validation_path))
        assert_validation_matches(
            validation,
            gguf_path,
            require_benchmark=require_benchmark,
            min_benchmark_prompt_tokens=min_benchmark_prompt_tokens,
            min_benchmark_gen_tokens=min_benchmark_gen_tokens,
        )
    elif require_validation:
        raise ValueError("missing required validation payload; pass validation_path or disable require_validation")
    quality_eval: dict[str, Any] | None = None
    if quality_eval_path is not None:
        quality_eval = load_json(Path(quality_eval_path))
        if quality_eval.get("schema") != "opentq.gguf_quality_eval.v1":
            raise ValueError(f"unsupported quality eval schema: {quality_eval_path}")
        if quality_eval.get("artifact", {}).get("filename") != gguf_path.name:
            raise ValueError("quality eval artifact filename does not match GGUF")
        if quality_eval.get("overall_pass") is not True:
            raise ValueError("quality eval did not pass")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    target_name = gguf_path.name
    link_file(gguf_path, output / target_name, link_mode)
    public_files = [target_name, "README.md", "opentq-gguf-release.json"]
    validation_public_name = None
    if validation_path is not None:
        validation_public_name = "validation.json"
        link_file(Path(validation_path), output / validation_public_name, link_mode)
        public_files.append(validation_public_name)
    quality_public_name = None
    if quality_eval_path is not None:
        quality_public_name = "quality-eval.json"
        link_file(Path(quality_eval_path), output / quality_public_name, link_mode)
        public_files.append(quality_public_name)

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
            "repo": "https://github.com/ggml-org/llama.cpp" if stock_compatible else runtime_repo,
            "stock_llama_cpp": "supported" if stock_compatible else "unsupported until OpenTQ tensor types are upstreamed",
            "requires": "stock llama.cpp" if stock_compatible else "llama.cpp-opentq",
            "stock_compatible": stock_compatible,
        },
        "release": {
            "text_only": text_only,
            "public_files": public_files,
            "excluded_private_artifacts": ["*.otq", "opentq-pack.json"],
        },
        "validation": {
            "required": require_validation,
            "benchmark_required": require_benchmark,
            "min_benchmark_prompt_tokens": min_benchmark_prompt_tokens if require_benchmark else None,
            "min_benchmark_gen_tokens": min_benchmark_gen_tokens if require_benchmark else None,
            "attached": validation is not None,
            "created_at": validation.get("created_at") if validation else None,
            "overall_pass": validation.get("overall_pass") if validation else None,
            "source": str(validation_path) if validation_path else None,
            "public_file": validation_public_name,
        },
        "quality_eval": {
            "attached": quality_eval is not None,
            "created_at": quality_eval.get("created_at") if quality_eval else None,
            "overall_pass": quality_eval.get("overall_pass") if quality_eval else None,
            "pass_rate": quality_eval.get("summary", {}).get("pass_rate") if quality_eval else None,
            "source": str(quality_eval_path) if quality_eval_path else None,
            "public_file": quality_public_name,
        },
        "upload": {
            "large_folder": f"hf upload-large-folder {repo_id} {output} --repo-type model",
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
            validation=validation,
            stock_compatible=stock_compatible,
            quality_eval=quality_eval,
        ),
        encoding="utf-8",
    )
    return summary


def clear_hf_gguf_stage(output_dir: str | Path) -> None:
    output = Path(output_dir)
    if output.exists():
        shutil.rmtree(output)
