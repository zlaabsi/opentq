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


def _profile_payload(gguf_name: str) -> dict[str, str]:
    if "OTQ-DYN-Q3_XL" in gguf_name:
        return {
            "profile": "OTQ-DYN-Q3_XL",
            "title": "Compact Apple Silicon GGUF",
            "bit_class": "3-bit dynamic allocation",
            "positioning": "smallest public OpenTQ dynamic-compatible release; best first pick for 32 GB Macs",
            "allocation": "Q3_K bulk MLP, Q4_K linear-attention blocks, Q5_K attention anchors and promoted edge layers",
            "memory": "32 GB Apple Silicon target; validated on M1 Max at 8K context",
            "tradeoff": "maximum memory headroom and fast prefill; lower precision than the Q4_XL release",
        }
    if "OTQ-DYN-Q4_XL" in gguf_name:
        return {
            "profile": "OTQ-DYN-Q4_XL",
            "title": "Balanced Apple Silicon GGUF",
            "bit_class": "4-bit dynamic allocation",
            "positioning": "primary quality-balanced public OpenTQ dynamic-compatible release",
            "allocation": "Q4_K bulk MLP, Q5_K/Q6_K attention and critical anchors, Q8_0 output-sensitive tensors",
            "memory": "32 GB Apple Silicon feasible at moderate context; 48 GB+ preferred for heavier agent workloads",
            "tradeoff": "better quality budget than Q3_XL; less memory headroom and slower 8K benchmark on M1 Max",
        }
    if "OTQ-DYN-Q5_XL" in gguf_name:
        return {
            "profile": "OTQ-DYN-Q5_XL",
            "title": "Quality-First Apple Silicon GGUF",
            "bit_class": "5-bit dynamic allocation",
            "positioning": "quality-first stock-GGUF candidate for larger-memory Macs",
            "allocation": "Q5_K bulk tensors with Q6_K/Q8_0 promoted attention and anchors",
            "memory": "48 GB+ Apple Silicon target",
            "tradeoff": "quality headroom over Q4_XL; larger disk and memory footprint",
        }
    if "OTQ-DYN-IQ4_NL" in gguf_name:
        return {
            "profile": "OTQ-DYN-IQ4_NL",
            "title": "Nonlinear 4-bit GGUF Experiment",
            "bit_class": "IQ4_NL dynamic allocation",
            "positioning": "experimental calibrated nonlinear 4-bit track",
            "allocation": "IQ4_NL bulk tensors with promoted anchors; imatrix-driven quantization expected",
            "memory": "32 GB+ Apple Silicon target after validation",
            "tradeoff": "potential quality improvement at similar size; calibration-dependent",
        }
    return {
        "profile": gguf_name,
        "title": "OpenTQ Dynamic-Compatible GGUF",
        "bit_class": "dynamic allocation",
        "positioning": "stock-compatible GGUF produced by OpenTQ",
        "allocation": "standard llama.cpp tensor types selected by OpenTQ policy",
        "memory": "Apple Silicon target depends on final artifact size and context length",
        "tradeoff": "uses stock kernels rather than native OpenTQ custom tensor types",
    }


def _phase(validation: dict[str, Any] | None, label: str) -> dict[str, Any] | None:
    if not validation:
        return None
    for item in validation.get("phases", []):
        if item.get("label") == label:
            return item
    return None


def _benchmark_rows(validation: dict[str, Any] | None) -> list[dict[str, str]]:
    phase = _phase(validation, "llama_bench")
    if not phase:
        return []
    rows: list[dict[str, str]] = []
    for line in str(phase.get("stdout_tail", "")).splitlines():
        if "pp" not in line and "tg" not in line:
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 6 or cells[0].startswith("-"):
            continue
        test = cells[-2]
        if not (test.startswith("pp") or test.startswith("tg")):
            continue
        rows.append(
            {
                "model": cells[0],
                "size": cells[1],
                "backend": cells[3],
                "test": test,
                "throughput": cells[-1].replace(" ± ", " +/- "),
            }
        )
    return rows


def _benchmark_table(validation: dict[str, Any] | None) -> str:
    rows = _benchmark_rows(validation)
    if not rows:
        return "Benchmark evidence is not attached to this model card."
    body = "\n".join(
        f"| {row['test']} | {row['throughput']} | {row['backend']} | {row['size']} |"
        for row in rows
    )
    return f"""| Test | Throughput | Backend | llama-bench size |
| --- | ---: | --- | ---: |
{body}"""


def _quality_table(quality_eval: dict[str, Any] | None) -> str:
    if not quality_eval:
        return "Quality evidence is not attached to this model card."
    summary = quality_eval.get("summary", {})
    categories = summary.get("categories", {})
    rows = []
    for name in ("knowledge", "stem_reasoning", "coding", "tool_use", "long_context"):
        row = categories.get(name, {})
        rows.append(f"| {name} | {row.get('passed', 0)}/{row.get('total', 0)} | {row.get('pass_rate', 'n/a')} |")
    return "\n".join(
        [
            "| Category | Passed | Pass rate |",
            "| --- | ---: | ---: |",
            *rows,
        ]
    )


def _command_gguf_name(gguf_name: str) -> str:
    return f"{gguf_name}.gguf"


def _short_sha(value: str) -> str:
    if not value or value == "not-computed":
        return value
    return value[:12]


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
    profile = _profile_payload(gguf_name)
    gguf_file = _command_gguf_name(gguf_name)
    benchmark_table = _benchmark_table(validation)
    quality_table = _quality_table(quality_eval)
    bounded_generation = _phase(validation, "bounded_generation")
    bounded_generation_seconds = bounded_generation.get("duration_seconds", "n/a") if bounded_generation else "n/a"
    if stock_compatible:
        runtime_section = f"""- Format: standard GGUF tensor types only
- Runtime: stock `llama.cpp` / `llama-server`
- Custom OpenTQ runtime: not required
- Modality in this release: {modality}

This artifact is the OpenTQ dynamic-compatible track: OpenTQ provides the tensor allocation policy and validation harness, while the GGUF payload uses existing llama.cpp quantization types.
"""
        build_section = """```bash
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
        build_section = f"""```bash
git clone {runtime_repo}
cd llama.cpp
cmake -B build -DGGML_METAL=ON -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=ON
cmake --build build -j
```
"""
    return f"""---
base_model: {base_model}
base_model_relation: quantized
license: apache-2.0
library_name: llama.cpp
tags:
- gguf
- llama.cpp
- opentq
- quantized
- qwen3.6
- qwen
- qwen3_5
- apple-silicon
- macos
- metal
- dynamic-quantization
- conversational
pipeline_tag: text-generation
language:
- en
quantized_by: zlaabsi
---

# {gguf_name}

[![GGUF](https://img.shields.io/badge/GGUF-stock%20llama.cpp-0f172a)](https://github.com/ggml-org/llama.cpp)
[![OpenTQ](https://img.shields.io/badge/OpenTQ-dynamic%20allocation-0891b2)](https://github.com/zlaabsi/opentq)
[![Apple Silicon](https://img.shields.io/badge/Apple%20Silicon-Metal%20validated-111827)](https://developer.apple.com/metal/)
[![Release gate](https://img.shields.io/badge/release%20gate-passed-16a34a)](#release-gate)

**{profile["title"]}** for `{base_model}`. This is an OpenTQ dynamic-compatible GGUF: OpenTQ decides the tensor-level quantization allocation, while the file itself uses standard llama.cpp tensor types.

## Why This Release Exists

This release is built for local inference on MacBook-class Apple Silicon where wall-clock time matters, especially with long prompts and agent-style tool context. It keeps the public artifact compatible with stock `llama.cpp` instead of requiring the native OpenTQ custom runtime.

| Field | Value |
| --- | --- |
| Profile | `{profile["profile"]}` |
| Bit class | {profile["bit_class"]} |
| Positioning | {profile["positioning"]} |
| Allocation policy | {profile["allocation"]} |
| Apple Silicon target | {profile["memory"]} |
| Tradeoff | {profile["tradeoff"]} |

## Model Overview

| Base model field | Value |
| --- | --- |
| Base model | `{base_model}` |
| Parameter class | 27B dense language model |
| Layer count | 64 |
| Hidden size | 5120 |
| Native context | 262,144 tokens in the base model; practical local context depends on RAM and KV cache |
| Release modality | {modality} GGUF |
| Runtime target | Apple Silicon Metal through stock `llama.cpp` |

## Files

| File | Size | SHA256 |
| --- | ---: | --- |
| `{gguf_file}` | {human_gib(gguf_size)} | `{gguf_sha256}` |

## Runtime Compatibility

{runtime_section}

This is the important compatibility boundary:

- `llama.cpp`, `llama-cli`, `llama-server`, LM Studio-style GGUF loaders: expected path
- OpenTQ native tensor runtime: not required for this release
- MLX: not the target runtime for this artifact; use a separate MLX release if needed
- Vision input: not included in this text-only release

## Quick Start

### Download

```bash
hf download {repo_id} {gguf_file} --local-dir .
```

### Build llama.cpp With Metal

{build_section}

### Run Locally

```bash
./build/bin/llama-cli \\
  -m {gguf_file} \\
  -ngl 99 \\
  -fa \\
  -c 8192 \\
  -p "<|im_start|>user\\nExplain the tradeoff between prefill and decode throughput.<|im_end|>\\n<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n"
```

### Serve OpenAI-Compatible API

```bash
./build/bin/llama-server \\
  -m {gguf_file} \\
  -ngl 99 \\
  -fa \\
  -c 8192 \\
  --host 0.0.0.0 \\
  --port 8080
```

## Apple Silicon Guide

| Machine class | Recommendation |
| --- | --- |
| 32 GB MacBook Pro / Mac Studio | Prefer `OTQ-DYN-Q3_XL` for headroom. `OTQ-DYN-Q4_XL` is usable at moderate context but leaves less memory for KV cache and apps. |
| 48-64 GB Apple Silicon | Prefer `OTQ-DYN-Q4_XL` for quality-balanced local inference. |
| 96 GB+ Apple Silicon | Use `OTQ-DYN-Q4_XL` for long-context stability; future Q5/IQ4 releases can target quality-first use. |
| Agent workloads with large system prompts/tools | Measure total wall-clock time. Decode-only tok/s hides prefill cost. |

## Release Gate

| Gate | Status |
| --- | --- |
| GGUF metadata read | {validation_status} |
| Bounded generation | {validation_status} |
| Long-context llama-bench | {validation_status} |
| Quality micro-suite | {quality_status} |
| Quality pass rate | {quality_summary.get("pass_rate", "n/a")} |
| Validation timestamp | `{validation_created_at}` |
| Bounded generation wall time | `{bounded_generation_seconds}s` |

The attached release evidence is available in:

- `validation.json`: metadata read, bounded generation, and 8K/128 llama-bench gate
- `quality-eval.json`: deterministic micro-suite with strict JSON tool-call scoring
- `opentq-gguf-release.json`: artifact metadata, SHA256, runtime declaration

## M1 Max Benchmark

Command:

```bash
./build/bin/llama-bench -m {gguf_file} -ngl 99 -fa on -p 8192 -n 128 -r 1 --no-warmup
```

{benchmark_table}

## Quality Smoke Suite

This is not a full academic benchmark. It is a release gate designed to catch obvious quantization/runtime regressions before publishing: factual completion, arithmetic, code output, strict JSON tool-call output, and a small long-context needle retrieval sample.

{quality_table}

## Reproduce The Release Evidence

```bash
uv run opentq validate-gguf \\
  --gguf {gguf_file} \\
  --output validation.json \\
  --llama-cpp /path/to/llama.cpp \\
  --ngl 99 \\
  --flash-attn on \\
  --ctx-size 8192 \\
  --n-predict 128 \\
  --bench \\
  --bench-prompt-tokens 8192 \\
  --bench-gen-tokens 128

uv run opentq eval-gguf \\
  --gguf {gguf_file} \\
  --output quality-eval.json \\
  --suite benchmarks/qwen36_quality_samples.jsonl \\
  --llama-cpp /path/to/llama.cpp \\
  --ngl 99 \\
  --flash-attn on \\
  --ctx-size 8192 \\
  --prompt-format qwen3-no-think
```

## What OpenTQ Means Here

`OpenTQ` has two tracks:

- **Dynamic-compatible GGUF**, used here: stock GGUF tensor types plus OpenTQ tensor allocation and validation.
- **Native OpenTQ**, not published here yet: custom tensor formats and custom runtime kernels.

This repo is intentionally the first track, because public Hugging Face releases should run in stock `llama.cpp`.

## Integrity

```text
repo: {repo_id}
base: {base_model}
file: {gguf_file}
size: {human_gib(gguf_size)}
sha256: {gguf_sha256}
short_sha: {_short_sha(gguf_sha256)}
modality: {modality}
stock_llama_cpp: {stock_compatible}
```
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
