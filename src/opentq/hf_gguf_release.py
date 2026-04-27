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
    if "OTQ-DYN-Q3_K_M" in gguf_name:
        return {
            "profile": "OTQ-DYN-Q3_K_M",
            "title": "Compact Apple Silicon GGUF",
            "bit_class": "3-bit dynamic allocation",
            "positioning": "smallest public OpenTQ dynamic-compatible release; best first pick for 32 GB Macs",
            "allocation": "Q3_K bulk MLP, Q4_K linear-attention blocks, Q5_K attention anchors and promoted edge layers",
            "memory": "32 GB Apple Silicon target; validated on M1 Max at 8K context",
            "tradeoff": "maximum memory headroom and fast prefill; lower precision than the Q4_K_M release",
        }
    if "OTQ-DYN-Q4_K_M" in gguf_name:
        return {
            "profile": "OTQ-DYN-Q4_K_M",
            "title": "Balanced Apple Silicon GGUF",
            "bit_class": "4-bit dynamic allocation",
            "positioning": "primary quality-balanced public OpenTQ dynamic-compatible release",
            "allocation": "Q4_K bulk MLP, Q5_K/Q6_K attention and critical anchors, Q8_0 output-sensitive tensors",
            "memory": "32 GB Apple Silicon feasible at moderate context; 48 GB+ preferred for heavier agent workloads",
            "tradeoff": "better quality budget than Q3_K_M; less memory headroom and slower 8K benchmark on M1 Max",
        }
    if "OTQ-DYN-Q5_K_M" in gguf_name:
        return {
            "profile": "OTQ-DYN-Q5_K_M",
            "title": "Quality-First Apple Silicon GGUF",
            "bit_class": "5-bit dynamic allocation",
            "positioning": "quality-first stock-GGUF candidate for larger-memory Macs",
            "allocation": "Q5_K bulk tensors with Q6_K/Q8_0 promoted attention and anchors",
            "memory": "48 GB+ Apple Silicon target",
            "tradeoff": "quality headroom over Q4_K_M; larger disk and memory footprint",
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


def _quality_eval_table(quality_eval: dict[str, Any] | None) -> str:
    if not quality_eval:
        return "No evaluation payload attached."
    summary = quality_eval.get("summary", {})
    categories = summary.get("categories", {})
    rows = []
    for name, row in sorted(categories.items()):
        rows.append(f"| {name} | {row.get('passed', 0)}/{row.get('total', 0)} | {row.get('pass_rate', 'n/a')} |")
    if not rows:
        rows.append("| uncategorized | 0/0 | n/a |")
    return "\n".join(
        [
            "| Category | Passed | Pass rate |",
            "| --- | ---: | ---: |",
            *rows,
        ]
    )


def _eval_summary_line(quality_eval: dict[str, Any] | None) -> str:
    if not quality_eval:
        return "not attached"
    summary = quality_eval.get("summary", {})
    return (
        f"{summary.get('passed', 0)}/{summary.get('total', 0)} "
        f"({summary.get('pass_rate', 'n/a')}); "
        f"mean {summary.get('latency_seconds_mean', 'n/a')}s; "
        f"p95 {summary.get('latency_seconds_p95', 'n/a')}s"
    )


def _counts_table(counts: dict[str, Any] | None, *, label: str) -> str:
    if not counts:
        return f"No {label} counts attached."
    rows = [
        f"| `{name}` | {value} |"
        for name, value in sorted(counts.items(), key=lambda item: (-int(item[1]), str(item[0])))
    ]
    return "\n".join(
        [
            f"| {label} | Count |",
            "| --- | ---: |",
            *rows,
        ]
    )


def _plan_summary_table(plan: dict[str, Any] | None) -> str:
    if not plan:
        return "Allocation plan is not attached."
    profile = plan.get("profile", {})
    summary = plan.get("summary", {})
    return f"""| Field | Value |
| --- | --- |
| Profile | `{profile.get("name", "n/a")}` |
| Base ftype | `{profile.get("base_ftype", "n/a")}` |
| Target | {profile.get("target", "n/a")} |
| Tensor rows | {summary.get("tensor_count", "n/a")} |
| Mapped tensor types | {summary.get("mapped_tensor_types", "n/a")} |
| Unmapped tensors | {summary.get("unmapped_count", "n/a")} |
| Requires imatrix | {profile.get("requires_imatrix", "n/a")} |"""


def _family_table(current_profile: str) -> str:
    rows = [
        {
            "profile": "OTQ-DYN-Q3_K_M",
            "repo": "zlaabsi/Qwen3.6-27B-OTQ-GGUF",
            "file": "Qwen3.6-27B-OTQ-DYN-Q3_K_M.gguf",
            "size": "13.48 GiB / 14.5 GB",
            "target": "32 GB Apple Silicon compact",
            "role": "smallest public OpenTQ dynamic-compatible release",
        },
        {
            "profile": "OTQ-DYN-Q4_K_M",
            "repo": "zlaabsi/Qwen3.6-27B-OTQ-GGUF",
            "file": "Qwen3.6-27B-OTQ-DYN-Q4_K_M.gguf",
            "size": "16.82 GiB / 18.1 GB",
            "target": "32 GB moderate context; 48 GB+ preferred",
            "role": "quality-balanced public release",
        },
    ]
    body = []
    for row in rows:
        marker = "current" if row["profile"] == current_profile else "sibling"
        body.append(
            f"| `{row['profile']}` | `{row['file']}` | {row['size']} | {row['target']} | {row['role']} | {marker} |"
        )
    return "\n".join(
        [
            "| Profile | GGUF file | Size | Apple Silicon target | Role | Status |",
            "| --- | --- | ---: | --- | --- | --- |",
            *body,
        ]
    )


def _release_banner_svg(profile: dict[str, str], gguf_name: str, gguf_size: int) -> str:
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="1400" height="420" viewBox="0 0 1400 420" role="img" aria-label="OpenTQ Qwen3.6 GGUF release banner">
  <defs>
    <linearGradient id="bg" x1="0" x2="1" y1="0" y2="1">
      <stop offset="0%" stop-color="#07111f"/>
      <stop offset="48%" stop-color="#0f2b36"/>
      <stop offset="100%" stop-color="#111827"/>
    </linearGradient>
    <linearGradient id="line" x1="0" x2="1">
      <stop offset="0%" stop-color="#22d3ee"/>
      <stop offset="50%" stop-color="#34d399"/>
      <stop offset="100%" stop-color="#f59e0b"/>
    </linearGradient>
    <filter id="glow" x="-20%" y="-20%" width="140%" height="140%">
      <feGaussianBlur stdDeviation="14" result="blur"/>
      <feMerge>
        <feMergeNode in="blur"/>
        <feMergeNode in="SourceGraphic"/>
      </feMerge>
    </filter>
  </defs>
  <rect width="1400" height="420" rx="34" fill="url(#bg)"/>
  <path d="M76 296 C238 190 336 340 500 230 S790 150 948 220 1196 234 1322 120" fill="none" stroke="url(#line)" stroke-width="10" opacity="0.95" filter="url(#glow)"/>
  <circle cx="1130" cy="118" r="88" fill="#22d3ee" opacity="0.13"/>
  <circle cx="1238" cy="208" r="130" fill="#34d399" opacity="0.10"/>
  <text x="78" y="108" fill="#e5f6ff" font-size="32" font-family="ui-monospace, SFMono-Regular, Menlo, Consolas, monospace" letter-spacing="3">OpenTQ Dynamic-Compatible GGUF</text>
  <text x="78" y="180" fill="#ffffff" font-size="58" font-family="Inter, ui-sans-serif, system-ui, sans-serif" font-weight="800">{gguf_name}</text>
  <text x="80" y="244" fill="#cbd5e1" font-size="28" font-family="Inter, ui-sans-serif, system-ui, sans-serif">{profile["title"]} for Qwen3.6-27B</text>
  <g transform="translate(80 304)">
    <rect width="246" height="52" rx="16" fill="#082f49" stroke="#38bdf8" stroke-opacity="0.55"/>
    <text x="24" y="34" fill="#e0f2fe" font-size="22" font-family="ui-monospace, SFMono-Regular, Menlo, Consolas, monospace">{profile["profile"]}</text>
  </g>
  <g transform="translate(352 304)">
    <rect width="262" height="52" rx="16" fill="#064e3b" stroke="#34d399" stroke-opacity="0.55"/>
    <text x="24" y="34" fill="#dcfce7" font-size="22" font-family="ui-monospace, SFMono-Regular, Menlo, Consolas, monospace">stock llama.cpp</text>
  </g>
  <g transform="translate(640 304)">
    <rect width="236" height="52" rx="16" fill="#451a03" stroke="#f59e0b" stroke-opacity="0.55"/>
    <text x="24" y="34" fill="#ffedd5" font-size="22" font-family="ui-monospace, SFMono-Regular, Menlo, Consolas, monospace">{human_gib(gguf_size)}</text>
  </g>
</svg>
"""


def _usage_markdown(
    *,
    repo_id: str,
    gguf_file: str,
    profile: dict[str, str],
    stock_compatible: bool,
) -> str:
    runtime_note = (
        "This release uses standard llama.cpp tensor types. No OpenTQ runtime patch is required."
        if stock_compatible
        else "This release requires an OpenTQ-aware runtime patch."
    )
    ollama_name = repo_id.split("/")[-1].replace("-GGUF", "").lower()
    return f"""# Usage

## Runtime Boundary

{runtime_note}

Target runtime for this Hugging Face release:

- `llama.cpp` with Metal on Apple Silicon
- `llama-server` for OpenAI-compatible local serving
- LM Studio or any GGUF app that accepts the same llama.cpp quant type
- Ollama via a local GGUF import

MLX is intentionally not the target for this GGUF artifact. If an MLX build is released later, it should be a separate repo.

## Download

```bash
hf download {repo_id} {gguf_file} --local-dir models/{repo_id.split('/')[-1]}
```

## llama.cpp CLI

```bash
./build/bin/llama-cli \\
  -m models/{repo_id.split('/')[-1]}/{gguf_file} \\
  -ngl 99 \\
  -fa \\
  -c 8192 \\
  -p "<|im_start|>user\\nGive a short benchmark plan for a local GGUF model.<|im_end|>\\n<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n"
```

## llama-server

```bash
./build/bin/llama-server \\
  -m models/{repo_id.split('/')[-1]}/{gguf_file} \\
  -ngl 99 \\
  -fa \\
  -c 8192 \\
  --host 0.0.0.0 \\
  --port 8080
```

Then point an OpenAI-compatible client at `http://localhost:8080/v1`.

## LM Studio

1. Download `{gguf_file}` from the Files tab.
2. Import it as a local GGUF model.
3. Use the model's embedded chat template if LM Studio detects it.
4. Keep context conservative on 32 GB Macs; start with 4096 or 8192 tokens.
5. Enable Metal/GPU offload if available.

## Ollama Local Import

Create a `Modelfile` next to the GGUF:

```text
FROM ./{gguf_file}
PARAMETER num_ctx 8192
PARAMETER temperature 0.6
```

Then import:

```bash
ollama create {ollama_name} -f Modelfile
ollama run {ollama_name}
```

## Thinking / No-Thinking Prompting

For deterministic checks, the release scripts use a no-thinking scaffold:

```text
<|im_start|>user
Your prompt here
<|im_end|>
<|im_start|>assistant
<think>

</think>

```

For normal chat, use your application's Qwen chat template. For agent workloads, measure total wall-clock time, not only decode tokens per second, because long prefill can dominate.

## Variant Guidance

Profile: `{profile["profile"]}`.

{profile["positioning"]}.
"""


def _benchmarks_markdown(
    *,
    gguf_file: str,
    validation: dict[str, Any] | None,
    quality_eval: dict[str, Any] | None,
    release_eval: dict[str, Any] | None,
) -> str:
    return f"""# Benchmarks And Release Evidence

These are release gates for local usability, not a replacement for a full academic benchmark run.

## llama-bench

Command:

```bash
./build/bin/llama-bench -m {gguf_file} -ngl 99 -fa on -p 8192 -n 128 -r 1 --no-warmup
```

{_benchmark_table(validation)}

## Quality Smoke Suite

File: `quality-eval.json`

{_quality_eval_table(quality_eval)}

Summary: {_eval_summary_line(quality_eval)}

## Extended Release Suite

File: `release-eval.json`

{_quality_eval_table(release_eval)}

Summary: {_eval_summary_line(release_eval)}

The extended suite covers small factual recall, French instruction following, arithmetic, decimal comparison, Python/JavaScript output, JSON tool-call shape, agentic ordering, needle retrieval, and constrained markdown formatting.

## Hardware Notes

The attached evaluations were produced with Metal GPU offload and FlashAttention enabled on Apple Silicon. On 32 GB machines, prefer measuring complete wall-clock latency for long prompts because prefill can dominate decode throughput.
"""


def _release_notes_markdown(
    *,
    gguf_name: str,
    gguf_file: str,
    gguf_size: int,
    gguf_sha256: str,
    profile: dict[str, str],
    validation: dict[str, Any] | None,
    quality_eval: dict[str, Any] | None,
    release_eval: dict[str, Any] | None,
    plan: dict[str, Any] | None,
) -> str:
    validation_status = validation.get("overall_pass") if validation else "not attached"
    quality_status = quality_eval.get("overall_pass") if quality_eval else "not attached"
    release_eval_status = release_eval.get("overall_pass") if release_eval else "not attached"
    plan_summary = plan.get("summary", {}) if plan else {}
    return f"""# Release Notes

## {gguf_name}

- File: `{gguf_file}`
- Size: {human_gib(gguf_size)}
- SHA256: `{gguf_sha256}`
- Profile: `{profile["profile"]}`
- Positioning: {profile["positioning"]}
- Runtime: stock llama.cpp GGUF

## Release Gate

- Validation: {validation_status}
- Quality smoke suite: {quality_status}
- Extended release suite: {release_eval_status}
- Quality smoke summary: {_eval_summary_line(quality_eval)}
- Extended suite summary: {_eval_summary_line(release_eval)}

## Allocation Transparency

- Tensor rows in plan: {plan_summary.get("tensor_count", "n/a")}
- Mapped tensor types: {plan_summary.get("mapped_tensor_types", "n/a")}
- Unmapped tensors: {plan_summary.get("unmapped_count", "n/a")}
- Plan file: `opentq-plan.json`
- Tensor type map: `tensor-types.txt`
- Annotated tensor map: `tensor-types.annotated.tsv`

## Compatibility Notes

This release is the OpenTQ dynamic-compatible track. OpenTQ selects the tensor-level allocation, but the published artifact uses recognized llama.cpp GGUF tensor types so that users do not need a custom OpenTQ runtime.
"""


def _infer_release_eval_path(gguf_path: Path, quality_eval_path: str | Path | None) -> Path | None:
    candidates: list[Path] = []
    if quality_eval_path is not None:
        quality_path = Path(quality_eval_path)
        candidates.append(quality_path.with_name(f"{gguf_path.parent.name}-release-extended.json"))
        if "-quality-qwen3-no-think" in quality_path.name:
            candidates.append(quality_path.with_name(quality_path.name.replace("-quality-qwen3-no-think", "-release-extended")))
        if "-quality" in quality_path.name:
            candidates.append(quality_path.with_name(quality_path.name.replace("-quality", "-release-extended")))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _load_release_eval(gguf_path: Path, quality_eval_path: str | Path | None) -> tuple[dict[str, Any] | None, Path | None]:
    path = _infer_release_eval_path(gguf_path, quality_eval_path)
    if path is None:
        return None, None
    payload = load_json(path)
    if payload.get("schema") != "opentq.gguf_quality_eval.v1":
        raise ValueError(f"unsupported release eval schema: {path}")
    if payload.get("artifact", {}).get("filename") != gguf_path.name:
        raise ValueError("release eval artifact filename does not match GGUF")
    if payload.get("overall_pass") is not True:
        raise ValueError("release eval did not pass")
    return payload, path


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
    release_eval: dict[str, Any] | None = None,
    plan: dict[str, Any] | None = None,
) -> str:
    modality = "text-only" if text_only else "text + vision"
    validation_status = "passed" if validation is not None else "not attached"
    validation_created_at = validation.get("created_at", "unknown") if validation else "n/a"
    quality_status = "passed" if quality_eval is not None and quality_eval.get("overall_pass") else "not attached"
    release_eval_status = "passed" if release_eval is not None and release_eval.get("overall_pass") else "not attached"
    quality_summary = quality_eval.get("summary", {}) if quality_eval else {}
    release_eval_summary = release_eval.get("summary", {}) if release_eval else {}
    profile = _profile_payload(gguf_name)
    gguf_file = _command_gguf_name(gguf_name)
    benchmark_table = _benchmark_table(validation)
    quality_table = _quality_table(quality_eval)
    release_eval_table = _quality_eval_table(release_eval)
    plan_summary_table = _plan_summary_table(plan)
    type_summary_table = _counts_table(plan.get("summary", {}).get("by_type") if plan else None, label="GGUF tensor type")
    family_table = _family_table(profile["profile"])
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

![OpenTQ Qwen3.6 banner](assets/opentq-qwen36-banner.svg)

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

## Variant Family

{family_table}

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

Additional release files:

- `USAGE.md`: direct llama.cpp, llama-server, LM Studio and Ollama recipes
- `BENCHMARKS.md`: benchmark and evaluation evidence
- `RELEASE_NOTES.md`: compatibility notes and release gate summary
- `opentq-plan.json`: OpenTQ dynamic allocation plan
- `tensor-types.txt`: llama.cpp tensor-type mapping
- `tensor-types.annotated.tsv`: annotated tensor allocation table
- `release-eval.json`: extended 10-sample release evaluation

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
| 32 GB MacBook Pro / Mac Studio | Prefer `OTQ-DYN-Q3_K_M` for headroom. `OTQ-DYN-Q4_K_M` is usable at moderate context but leaves less memory for KV cache and apps. |
| 48-64 GB Apple Silicon | Prefer `OTQ-DYN-Q4_K_M` for quality-balanced local inference. |
| 96 GB+ Apple Silicon | Use `OTQ-DYN-Q4_K_M` for long-context stability; future Q5/IQ4 releases can target quality-first use. |
| Agent workloads with large system prompts/tools | Measure total wall-clock time. Decode-only tok/s hides prefill cost. |

## Release Gate

| Gate | Status |
| --- | --- |
| GGUF metadata read | {validation_status} |
| Bounded generation | {validation_status} |
| Long-context llama-bench | {validation_status} |
| Quality micro-suite | {quality_status} |
| Quality pass rate | {quality_summary.get("pass_rate", "n/a")} |
| Extended release suite | {release_eval_status} |
| Extended release pass rate | {release_eval_summary.get("pass_rate", "n/a")} |
| Validation timestamp | `{validation_created_at}` |
| Bounded generation wall time | `{bounded_generation_seconds}s` |

The attached release evidence is available in:

- `validation.json`: metadata read, bounded generation, and 8K/128 llama-bench gate
- `quality-eval.json`: deterministic micro-suite with strict JSON tool-call scoring
- `release-eval.json`: extended release suite with 10 deterministic samples
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

## Extended Release Suite

The extended suite adds more deterministic samples across factual recall, French instruction following, arithmetic, decimal comparison, Python/JavaScript output, JSON tool-call shape, agentic ordering, needle retrieval, and constrained markdown formatting.

{release_eval_table}

## Allocation Transparency

{plan_summary_table}

{type_summary_table}

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

uv run opentq eval-gguf \\
  --gguf {gguf_file} \\
  --output release-eval.json \\
  --suite benchmarks/qwen36_release_extended_samples.jsonl \\
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
    release_eval, release_eval_path = _load_release_eval(gguf_path, quality_eval_path)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for stale_name in (
        "USAGE.md",
        "BENCHMARKS.md",
        "RELEASE_NOTES.md",
        "opentq-plan.json",
        "tensor-types.txt",
        "tensor-types.annotated.tsv",
        "quantize-dry-run.log",
        "release-eval.json",
    ):
        stale_path = output / stale_name
        if stale_path.exists():
            stale_path.unlink()
    stale_asset = output / "assets" / "opentq-qwen36-banner.svg"
    if stale_asset.exists():
        stale_asset.unlink()
    target_name = gguf_path.name
    for stale_gguf in output.glob("*.gguf"):
        if stale_gguf.name != target_name:
            stale_gguf.unlink()
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
    release_eval_public_name = None
    if release_eval_path is not None:
        release_eval_public_name = "release-eval.json"
        link_file(release_eval_path, output / release_eval_public_name, link_mode)
        public_files.append(release_eval_public_name)

    plan: dict[str, Any] | None = None
    transparency_files: list[dict[str, str]] = []
    for source_name, public_name in (
        ("plan.json", "opentq-plan.json"),
        ("tensor-types.txt", "tensor-types.txt"),
        ("tensor-types.annotated.tsv", "tensor-types.annotated.tsv"),
        ("dry-run.log", "quantize-dry-run.log"),
    ):
        source = gguf_path.parent / source_name
        if not source.exists():
            continue
        link_file(source, output / public_name, link_mode)
        public_files.append(public_name)
        transparency_files.append({"source": str(source), "public_file": public_name})
        if source_name == "plan.json":
            plan = load_json(source)

    size = gguf_path.stat().st_size
    digest = sha256_file(gguf_path) if compute_sha256 else "not-computed"
    slug = infer_variant_slug(gguf_path)
    profile = _profile_payload(slug)
    assets_dir = output / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    (assets_dir / "opentq-qwen36-banner.svg").write_text(
        _release_banner_svg(profile, slug, size),
        encoding="utf-8",
    )
    (output / "USAGE.md").write_text(
        _usage_markdown(
            repo_id=repo_id,
            gguf_file=target_name,
            profile=profile,
            stock_compatible=stock_compatible,
        ),
        encoding="utf-8",
    )
    (output / "BENCHMARKS.md").write_text(
        _benchmarks_markdown(
            gguf_file=target_name,
            validation=validation,
            quality_eval=quality_eval,
            release_eval=release_eval,
        ),
        encoding="utf-8",
    )
    (output / "RELEASE_NOTES.md").write_text(
        _release_notes_markdown(
            gguf_name=slug,
            gguf_file=target_name,
            gguf_size=size,
            gguf_sha256=digest,
            profile=profile,
            validation=validation,
            quality_eval=quality_eval,
            release_eval=release_eval,
            plan=plan,
        ),
        encoding="utf-8",
    )
    public_files.extend(["USAGE.md", "BENCHMARKS.md", "RELEASE_NOTES.md", "assets/opentq-qwen36-banner.svg"])
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
        "release_eval": {
            "attached": release_eval is not None,
            "created_at": release_eval.get("created_at") if release_eval else None,
            "overall_pass": release_eval.get("overall_pass") if release_eval else None,
            "pass_rate": release_eval.get("summary", {}).get("pass_rate") if release_eval else None,
            "source": str(release_eval_path) if release_eval_path else None,
            "public_file": release_eval_public_name,
        },
        "transparency": {
            "attached": bool(transparency_files),
            "files": transparency_files,
            "plan_schema": plan.get("schema") if plan else None,
            "tensor_count": plan.get("summary", {}).get("tensor_count") if plan else None,
            "mapped_tensor_types": plan.get("summary", {}).get("mapped_tensor_types") if plan else None,
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
            release_eval=release_eval,
            plan=plan,
        ),
        encoding="utf-8",
    )
    return summary


def clear_hf_gguf_stage(output_dir: str | Path) -> None:
    output = Path(output_dir)
    if output.exists():
        shutil.rmtree(output)
