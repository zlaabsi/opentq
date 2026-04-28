#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from opentq.hf_gguf_release import sha256_file
from opentq.hf_release import human_gib, link_file


REPO_ID = "zlaabsi/Qwen3.6-27B-OTQ-GGUF"
BASE_MODEL = "Qwen/Qwen3.6-27B"
DEFAULT_BANNER = Path("docs/assets/qwen36-opentq-hero.png")


@dataclass(frozen=True)
class Variant:
    quant: str
    profile: str
    title: str
    role: str
    target: str
    allocation: str
    source_prefix: str

    @property
    def filename(self) -> str:
        return f"Qwen3.6-27B-OTQ-DYN-{self.quant}.gguf"


VARIANTS = (
    Variant(
        quant="Q3_K_M",
        profile="OTQ-DYN-Q3_K_M",
        title="Compact TurboQuant dynamic GGUF",
        role="smallest public OpenTQ dynamic-compatible release",
        target="32 GB Apple Silicon first pick",
        allocation="Q3_K bulk MLP, Q4_K linear-attention blocks, Q5_K attention anchors, Q6_K output head",
        source_prefix="Q3",
    ),
    Variant(
        quant="Q4_K_M",
        profile="OTQ-DYN-Q4_K_M",
        title="Balanced TurboQuant dynamic GGUF",
        role="quality-balanced public release",
        target="32 GB moderate context; 48 GB+ preferred",
        allocation="Q4_K bulk MLP, Q5_K/Q6_K attention and critical anchors, Q8_0 output-sensitive tensors",
        source_prefix="Q4",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage the canonical multi-file Qwen3.6-27B OpenTQ GGUF repo.")
    parser.add_argument("--output", default="artifacts/hf-gguf-canonical/Qwen3.6-27B-OTQ-GGUF")
    parser.add_argument("--source-root", default="artifacts/qwen3.6-27b-dynamic-gguf")
    parser.add_argument("--validation-root", default="artifacts/qwen3.6-27b-dynamic-validation")
    parser.add_argument("--eval-root", default="artifacts/qwen3.6-27b-dynamic-eval")
    parser.add_argument("--link-mode", choices=("hardlink", "copy", "symlink"), default="hardlink")
    parser.add_argument("--banner", default=os.environ.get("OPENTQ_QWEN36_BANNER"))
    parser.add_argument("--practical-report", default="artifacts/qwen3.6-27b-degradation-report-practical/degradation-report.json")
    parser.add_argument("--skip-report", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def public_replacements(variant: Variant) -> list[tuple[str, str]]:
    legacy_profile = f"OTQ-DYN-{variant.source_prefix}_XL"
    legacy_dir = f"Qwen3.6-27B-{legacy_profile}-GGUF"
    legacy_file = f"Qwen3.6-27B-{legacy_profile}.gguf"
    legacy_quant_file = f"Qwen3.6-27B-{legacy_profile}-{variant.quant}.gguf"
    return [
        (legacy_quant_file, variant.filename),
        (legacy_file, variant.filename),
        (legacy_dir, "Qwen3.6-27B-OTQ-GGUF"),
        (legacy_profile, variant.profile),
    ]


def sanitize_public_text(text: str, variant: Variant) -> str:
    for old, new in public_replacements(variant):
        text = text.replace(old, new)
    return text


def sanitize_public_value(value: Any, variant: Variant) -> Any:
    if isinstance(value, str):
        return sanitize_public_text(value, variant)
    if isinstance(value, list):
        return [sanitize_public_value(item, variant) for item in value]
    if isinstance(value, dict):
        return {key: sanitize_public_value(item, variant) for key, item in value.items()}
    return value


def first_existing(patterns: list[str]) -> Path:
    matches: list[Path] = []
    for pattern in patterns:
        if Path(pattern).is_absolute():
            matches.extend(Path(match) for match in glob.glob(pattern))
        else:
            matches.extend(Path().glob(pattern))
    existing = sorted(path for path in matches if path.exists())
    if not existing:
        raise FileNotFoundError("missing artifact for patterns: " + ", ".join(patterns))
    return existing[0]


def source_dir(source_root: Path, variant: Variant) -> Path:
    return first_existing([str(source_root / f"Qwen3.6-27B-OTQ-DYN-{variant.source_prefix}*-GGUF")])


def source_gguf(source: Path, variant: Variant) -> Path:
    preferred = sorted(path for path in source.glob(f"*{variant.quant}.gguf") if path.is_file())
    if preferred:
        return preferred[0]
    return first_existing([str(source / f"Qwen3.6-27B-OTQ-DYN-{variant.source_prefix}*.gguf")])


def evidence_paths(validation_root: Path, eval_root: Path, variant: Variant) -> dict[str, Path]:
    stem = f"Qwen3.6-27B-OTQ-DYN-{variant.source_prefix}*-GGUF"
    return {
        "validation": first_existing([str(validation_root / f"{stem}-release-bench.json")]),
        "quality": first_existing([str(eval_root / f"{stem}-quality-qwen3-no-think.json")]),
        "release_eval": first_existing([str(eval_root / f"{stem}-release-extended.json")]),
    }


def rewrite_artifact_payload(path: Path, destination: Path, variant: Variant, gguf: Path) -> dict[str, Any]:
    payload = sanitize_public_value(load_json(path), variant)
    artifact = dict(payload.get("artifact", {}))
    artifact["filename"] = variant.filename
    artifact["path"] = variant.filename
    artifact["bytes"] = gguf.stat().st_size
    payload["artifact"] = artifact
    dump_json(destination, payload)
    return payload


def rewrite_plan(source: Path, destination: Path, variant: Variant) -> dict[str, Any]:
    payload = sanitize_public_value(load_json(source), variant)
    profile = dict(payload.get("profile", {}))
    profile["name"] = variant.profile
    payload["profile"] = profile
    payload["public_filename"] = variant.filename
    payload["public_repo"] = REPO_ID
    dump_json(destination, payload)
    return payload


def copy_transparency(source: Path, target: Path, variant: Variant) -> None:
    plan = source / "plan.json"
    if plan.exists():
        rewrite_plan(plan, target / "opentq-plan.json", variant)
    for name in ("tensor-types.txt", "tensor-types.annotated.tsv", "dry-run.log"):
        if not (source / name).exists():
            continue
        public_name = "quantize-dry-run.log" if name == "dry-run.log" else name
        text = (source / name).read_text(encoding="utf-8", errors="replace")
        (target / public_name).write_text(sanitize_public_text(text, variant), encoding="utf-8")


def bench_rows(validation: dict[str, Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for phase in validation.get("phases", []):
        if phase.get("label") != "llama_bench":
            continue
        for line in str(phase.get("stdout_tail", "")).splitlines():
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            if len(cells) < 7 or not cells[-2].startswith(("pp", "tg")):
                continue
            rows.append(
                {
                    "test": cells[-2],
                    "throughput": cells[-1].replace(" ± ", " +/- "),
                    "backend": cells[3],
                    "size": cells[1],
                }
            )
    return rows


def md_bench_table(records: list[dict[str, Any]]) -> str:
    lines = ["| Variant | Test | Throughput | Backend | Size |", "| --- | --- | ---: | --- | ---: |"]
    for record in records:
        for row in bench_rows(record["validation"]):
            lines.append(
                f"| `{record['variant'].quant}` | {row['test']} | {row['throughput']} | {row['backend']} | {row['size']} |"
            )
    return "\n".join(lines)


def md_eval_table(records: list[dict[str, Any]]) -> str:
    lines = ["| Variant | Suite | Passed | Pass rate | Mean latency | p95 latency |", "| --- | --- | ---: | ---: | ---: | ---: |"]
    for record in records:
        for label, payload in (("smoke", record["quality"]), ("release", record["release_eval"])):
            summary = payload.get("summary", {})
            lines.append(
                f"| `{record['variant'].quant}` | {label} | {summary.get('passed', 0)}/{summary.get('total', 0)} | "
                f"{summary.get('pass_rate', 'n/a')} | {summary.get('latency_seconds_mean', 'n/a')}s | "
                f"{summary.get('latency_seconds_p95', 'n/a')}s |"
            )
    return "\n".join(lines)


def phase_duration(validation: dict[str, Any], label: str) -> str:
    for phase in validation.get("phases", []):
        if phase.get("label") == label:
            return f"{phase.get('duration_seconds', 'n/a')}s"
    return "n/a"


def pass_text(value: bool) -> str:
    return "passed" if value else "failed"


def md_release_gate_table(records: list[dict[str, Any]]) -> str:
    lines = [
        "| Variant | Metadata | Bounded generation | 8K llama-bench | Smoke gate | Release gate | Timestamp |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for record in records:
        validation = record["validation"]
        gates = validation.get("gates", {})
        smoke = record["quality"].get("summary", {})
        release = record["release_eval"].get("summary", {})
        lines.append(
            f"| `{record['variant'].quant}` | {pass_text(bool(gates.get('gguf_metadata_read')))} | "
            f"{pass_text(bool(gates.get('bounded_generation')))} ({phase_duration(validation, 'bounded_generation')}) | "
            f"{pass_text(bool(gates.get('benchmark')))} ({phase_duration(validation, 'llama_bench')}) | "
            f"{smoke.get('passed', 0)}/{smoke.get('total', 0)} | "
            f"{release.get('passed', 0)}/{release.get('total', 0)} | "
            f"`{validation.get('created_at', 'n/a')}` |"
        )
    return "\n".join(lines)


def md_family_table(records: list[dict[str, Any]]) -> str:
    lines = ["| File | Quant | Size | Apple Silicon target | Role |", "| --- | --- | ---: | --- | --- |"]
    for record in records:
        variant = record["variant"]
        lines.append(
            f"| `{variant.filename}` | `{variant.quant}` | {human_gib(record['bytes'])} | {variant.target} | {variant.role} |"
        )
    return "\n".join(lines)


def md_allocation_summary(records: list[dict[str, Any]]) -> str:
    tensor_types = ["F16", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "Q8_0"]
    lines = ["| Variant | Mapped tensors | " + " | ".join(tensor_types) + " |", "| --- | ---: | " + " | ".join(["---:"] * len(tensor_types)) + " |"]
    for record in records:
        summary = record.get("plan", {}).get("summary", {})
        by_type = summary.get("by_type", {})
        lines.append(
            f"| `{record['variant'].quant}` | {summary.get('mapped_tensor_types', 'n/a')} | "
            + " | ".join(str(by_type.get(tensor_type, 0)) for tensor_type in tensor_types)
            + " |"
        )
    return "\n".join(lines)


def md_category_table(records: list[dict[str, Any]], key: str) -> str:
    categories = sorted(
        {
            category
            for record in records
            for category in record[key].get("summary", {}).get("categories", {}).keys()
        }
    )
    lines = ["| Variant | " + " | ".join(category.replace("_", " ") for category in categories) + " |", "| --- | " + " | ".join(["---:"] * len(categories)) + " |"]
    for record in records:
        by_category = record[key].get("summary", {}).get("categories", {})
        values = []
        for category in categories:
            payload = by_category.get(category, {})
            values.append(f"{payload.get('passed', 0)}/{payload.get('total', 0)}")
        lines.append(f"| `{record['variant'].quant}` | " + " | ".join(values) + " |")
    return "\n".join(lines)


def find_font(candidates: list[str], size: int) -> ImageFont.FreeTypeFont:
    for candidate in candidates:
        path = Path(candidate).expanduser()
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default(size=size)


def dither_mask(width: int, height: int) -> Image.Image:
    y, x = np.mgrid[0:height, 0:width]
    pattern = ((x * 7 + y * 13) % 19) / 18.0
    waves = (np.sin(x / 53.0) + np.cos(y / 41.0) + np.sin((x + y) / 77.0)) / 3.0
    mask = ((pattern + waves * 0.34) > 0.58).astype(np.uint8) * 255
    return Image.fromarray(mask, mode="L").filter(ImageFilter.GaussianBlur(0.15))


def make_banner(path: Path) -> None:
    width, height = 1600, 520
    y, x = np.mgrid[0:height, 0:width]
    left = np.array([255, 112, 67], dtype=np.float32)
    mid = np.array([238, 138, 255], dtype=np.float32)
    right = np.array([38, 101, 255], dtype=np.float32)
    t = x / (width - 1)
    gradient = np.zeros((height, width, 3), dtype=np.float32)
    first = t < 0.55
    a = np.clip(t / 0.55, 0, 1)[..., None]
    b = np.clip((t - 0.55) / 0.45, 0, 1)[..., None]
    gradient[first] = left * (1 - a[first]) + mid * a[first]
    gradient[~first] = mid * (1 - b[~first]) + right * b[~first]
    vertical = (0.92 - 0.36 * (y / (height - 1)))[..., None]
    image = Image.fromarray(np.clip(gradient * vertical, 0, 255).astype(np.uint8), mode="RGB")

    overlay = Image.new("RGB", (width, height), "#0c0b11")
    mask = dither_mask(width, height)
    image = Image.composite(overlay, image, mask.point(lambda px: int(px * 0.34)))

    draw = ImageDraw.Draw(image, "RGBA")
    draw.rounded_rectangle((24, 24, width - 24, height - 24), radius=46, outline=(255, 255, 255, 64), width=2)
    draw.ellipse((980, -120, 1480, 380), fill=(18, 17, 18, 58))
    draw.ellipse((1120, 110, 1690, 680), fill=(255, 255, 255, 24))
    draw.rounded_rectangle((78, 82, 386, 132), radius=24, fill=(18, 17, 18, 210))
    draw.rounded_rectangle((406, 82, 676, 132), radius=24, fill=(255, 255, 255, 46))

    pixel_font = find_font(
        [
            "~/Library/Fonts/GeistPixel-Regular.ttf",
            "/System/Library/Fonts/SFNSMono.ttf",
            "/System/Library/Fonts/Menlo.ttc",
        ],
        34,
    )
    title_font = find_font(
        [
            "~/Library/Fonts/KHTeka-Regular.ttf",
            "/System/Library/Fonts/Avenir Next Condensed.ttc",
            "/System/Library/Fonts/Supplemental/DIN Condensed Bold.ttf",
            "/System/Library/Fonts/HelveticaNeue.ttc",
        ],
        96,
    )
    subtitle_font = find_font(
        [
            "~/Library/Fonts/KHTeka-Regular.ttf",
            "/System/Library/Fonts/Avenir Next Condensed.ttc",
            "/System/Library/Fonts/HelveticaNeue.ttc",
        ],
        40,
    )
    small_font = find_font(["/System/Library/Fonts/SFNSMono.ttf", "/System/Library/Fonts/Menlo.ttc"], 28)

    draw.text((104, 91), "OpenTQ", font=pixel_font, fill=(255, 255, 255, 238))
    draw.text((430, 92), "TurboQuant", font=pixel_font, fill=(18, 17, 18, 230))
    draw.text((78, 182), "Qwen3.6 27B", font=title_font, fill=(255, 255, 255, 246))
    draw.text((82, 286), "Dynamic-compatible GGUFs for stock llama.cpp on Apple Silicon", font=subtitle_font, fill=(255, 255, 255, 222))

    badges = [("Q3_K_M", "13.48 GiB"), ("Q4_K_M", "16.82 GiB"), ("Metal + FA", "M1 Max validated")]
    x0 = 82
    for label, value in badges:
        text = f"{label}  /  {value}"
        bbox = draw.textbbox((0, 0), text, font=small_font)
        w = bbox[2] - bbox[0] + 42
        draw.rounded_rectangle((x0, 382, x0 + w, 438), radius=18, fill=(18, 17, 18, 210), outline=(255, 255, 255, 54), width=1)
        draw.text((x0 + 22, 397), text, font=small_font, fill=(255, 255, 255, 232))
        x0 += w + 18

    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, optimize=True)


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
            "`Q5_K_M` is pending until disk cleanup, generation and runtime validation are complete.",
            "",
        ]
    )
    return "\n".join(lines)


def format_subset_score(summary: dict[str, Any]) -> str:
    total = int(summary.get("total", 0) or 0)
    passed = int(summary.get("passed", 0) or 0)
    pass_rate = float(summary.get("pass_rate", 0.0) or 0.0) * 100
    return f"{passed}/{total} ({pass_rate:.1f}%)"


def practical_subset_markdown(path: Path) -> str:
    if not path.exists():
        return "Practical mini-subset report not staged in this local checkout."
    payload = load_json(path)
    rows = []
    for row in payload.get("rows", []):
        summaries = row.get("subset_summaries") or {}
        if not summaries:
            continue
        q3 = format_subset_score(summaries["q3"]) if "q3" in summaries else "pending"
        q4 = format_subset_score(summaries["q4"]) if "q4" in summaries else "pending"
        rows.append((row["benchmark_id"], q3, q4, row["claim_status"]))
    if not rows:
        return "Practical mini-subset report has no scored rows."
    q3_passed = q3_total = q4_passed = q4_total = 0
    for _benchmark, q3, q4, _status in rows:
        if q3 != "pending":
            passed, total = q3.split(" ", 1)[0].split("/", 1)
            q3_passed += int(passed)
            q3_total += int(total)
        if q4 != "pending":
            passed, total = q4.split(" ", 1)[0].split("/", 1)
            q4_passed += int(passed)
            q4_total += int(total)
    lines = [
        "These are small local release signals, not full benchmark replacements.",
        f"Practical total: `Q3_K_M` {q3_passed}/{q3_total}; `Q4_K_M` {q4_passed}/{q4_total}.",
        "",
        "| Benchmark | Q3_K_M | Q4_K_M | Claim status |",
        "| --- | ---: | ---: | --- |",
    ]
    lines.extend(f"| `{benchmark}` | {q3} | {q4} | `{status}` |" for benchmark, q3, q4, status in rows)
    lines.extend(
        [
            "",
            "Official degradation claims remain blocked unless the benchmark task, split, prompt format and scoring rule match the official setup, or a matching BF16 mini sidecar exists.",
            "",
        ]
    )
    return "\n".join(lines)


def readme(records: list[dict[str, Any]], practical_report: Path) -> str:
    file_rows = []
    for record in records:
        variant = record["variant"]
        file_rows.append(
            f"| `{variant.filename}` | `{variant.quant}` | {human_gib(record['bytes'])} | `{record['sha256']}` | {variant.target} |"
        )
    file_table = "\n".join(["| File | Quant | Size | SHA256 | Target |", "| --- | --- | ---: | --- | --- |", *file_rows])
    return f"""---
base_model: {BASE_MODEL}
base_model_relation: quantized
license: apache-2.0
library_name: llama.cpp
tags:
- gguf
- llama.cpp
- opentq
- turboquant
- quantized
- qwen3.6
- qwen
- qwen3_5
- apple-silicon
- macos
- metal
- dynamic-quantization
- conversational
pipeline_tag: image-text-to-text
language:
- en
quantized_by: zlaabsi
---

# Qwen3.6-27B-OTQ-GGUF

![OpenTQ TurboQuant Qwen3.6 banner](assets/opentq-qwen36-hero.png)

[![GGUF](https://img.shields.io/badge/GGUF-stock%20llama.cpp-0a1628)](https://github.com/ggml-org/llama.cpp)
[![OpenTQ](https://img.shields.io/badge/OpenTQ-TurboQuant%20allocation-0b3d73)](https://github.com/zlaabsi/opentq)
[![Apple Silicon](https://img.shields.io/badge/Apple%20Silicon-Metal%20%2B%20FA-125ea5)](https://developer.apple.com/metal/)
[![Release gate](https://img.shields.io/badge/release%20gate-passed-2f80c2)](#release-gate)
[![Base model](https://img.shields.io/badge/base-Qwen%2FQwen3.6--27B-7cc7ff)](https://huggingface.co/Qwen/Qwen3.6-27B)

**OpenTQ TurboQuant dynamic-compatible GGUFs** for `{BASE_MODEL}`.

This is the stock `llama.cpp` release track. OpenTQ chooses the tensor-level allocation policy, but the files themselves use standard GGUF tensor types (`Q3_K_M`, `Q4_K_M`, `Q5_K`, `Q6_K`, `Q8_0`, `F16`). No custom OpenTQ runtime is required for these GGUF files.

> The Hugging Face `pipeline_tag` follows the official Qwen3.6-27B card (`image-text-to-text`). These GGUF artifacts are validated here for local text inference with stock `llama.cpp`; vision tensors are not part of this text-focused release track.

## Why This Release Exists

These builds target MacBook-class Apple Silicon where wall-clock time matters, especially with long prompts, large system messages and agent/tool context. The goal is not to publish another uniform quant; it is to provide a stock-compatible GGUF family where OpenTQ spends precision on the tensors that matter more for local inference.

| Field | Value |
| --- | --- |
| Release track | `Qwen3.6-27B-OTQ-GGUF` |
| Method | OpenTQ / TurboQuant-inspired dynamic tensor allocation |
| Runtime | stock `llama.cpp` with Metal and FlashAttention |
| Compatibility boundary | standard GGUF only; no native OpenTQ kernel required |
| Current public variants | `Q3_K_M` compact and `Q4_K_M` balanced |
| Validation machine | M1 Max, 8K prefill gate, bounded generation, deterministic release suites |

## Files

{file_table}

## Variant Family

{md_family_table(records)}

## Naming

- `OTQ`: OpenTQ, the release/tooling brand.
- `TurboQuant`: the quantization family and design direction.
- `DYN`: dynamic tensor-level allocation; different tensor families receive different GGUF quant types.
- `Q3_K_M` / `Q4_K_M`: standard GGUF quant names recognized by Hugging Face and stock `llama.cpp`.

## Which File Should I Use?

- `Q3_K_M`: first pick for 32 GB Apple Silicon and larger app/tool contexts.
- `Q4_K_M`: quality-balanced pick; usable on 32 GB at moderate context, more comfortable on 48 GB+.

{hardware_compatibility_markdown()}

## Model Overview

| Base model field | Value |
| --- | --- |
| Base model | `{BASE_MODEL}` |
| Parameter class | 27B dense model |
| HF architecture | `Qwen3_5ForConditionalGeneration` |
| Layer count | 64 language layers |
| Hidden size | 5120 |
| Native context | 262,144 tokens in the base model; practical local context depends on RAM, KV/cache settings and apps |
| Public GGUF modality | text inference release track |
| Runtime target | Apple Silicon Metal through stock `llama.cpp` |

## Runtime Compatibility

- `llama.cpp`, `llama-cli`, `llama-server`: supported.
- LM Studio and Ollama local GGUF import: expected to work as standard GGUF loaders.
- OpenTQ custom runtime: not required for this repo.
- Native TurboQuant/OpenTQ tensor formats: separate release track, not mixed into this GGUF repo.
- MLX: not the target runtime for this GGUF track.

## Quick Start

### 1. Download A GGUF

```bash
hf download {REPO_ID} Qwen3.6-27B-OTQ-DYN-Q3_K_M.gguf --local-dir models/Qwen3.6-27B-OTQ-GGUF
```

Use `Q3_K_M` first on 32 GB Macs. Use `Q4_K_M` when you can afford the extra memory.

### 2. Build llama.cpp With Metal

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -B build -DGGML_METAL=ON -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=ON
cmake --build build -j
```

### 3. Run Locally

```bash
./build/bin/llama-cli \\
  -m models/Qwen3.6-27B-OTQ-GGUF/Qwen3.6-27B-OTQ-DYN-Q3_K_M.gguf \\
  -ngl 99 \\
  -fa \\
  -c 8192 \\
  --temp 0.6 \\
  --top-p 0.95 \\
  -p "<|im_start|>user\\nExplain the tradeoff between prefill and decode throughput.<|im_end|>\\n<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n"
```

### 4. Serve An OpenAI-Compatible API

```bash
./build/bin/llama-server \\
  -m models/Qwen3.6-27B-OTQ-GGUF/Qwen3.6-27B-OTQ-DYN-Q3_K_M.gguf \\
  -ngl 99 \\
  -fa \\
  -c 8192 \\
  --host 0.0.0.0 \\
  --port 8080
```

```bash
curl http://localhost:8080/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{{"model":"qwen3.6-27b-otq","messages":[{{"role":"user","content":"Give me a 3-bullet summary of OpenTQ."}}],"temperature":0.6}}'
```

## llama.cpp Settings

| Setting | Recommended value | Why |
| --- | --- | --- |
| GPU layers | `-ngl 99` | Offload all supported layers to Metal on Apple Silicon |
| FlashAttention | `-fa` / `-fa on` | Critical for long-context prefill wall-clock |
| Context | `-c 8192` first | Validated release gate; increase only after checking memory headroom |
| Prompt format | Qwen chat template | Keep `<|im_start|>` / `<|im_end|>` formatting |
| Sampling | `--temp 0.6 --top-p 0.95` | Good default for general chat; tighten for deterministic evals |
| Server | `llama-server` | Use for OpenAI-compatible local apps and agents |

## Apple Silicon Guide

| Machine class | Recommendation |
| --- | --- |
| 32 GB MacBook Pro / Mac Studio | Prefer `Q3_K_M` for headroom, especially with agentic prompts and other apps open. |
| 48-64 GB Apple Silicon | Prefer `Q4_K_M` for quality-balanced local inference. |
| 96 GB+ Apple Silicon | `Q4_K_M` is the current quality pick; future Q5/IQ4 variants can target quality-first use. |
| Agent workloads with large tool context | Measure total wall-clock time. Decode-only tok/s hides prefill cost. |

## Benchmarks

{md_bench_table(records)}

![Runtime frontier](assets/runtime-frontier.png)

![Prefill decode tradeoff](assets/prefill-decode-tradeoff.png)

![Release scorecard](assets/release-scorecard.png)

The plots compare the quantized OTQ artifacts against each other on measured release data. Official Qwen scores are kept as a reference table, not plotted as a fake delta.

## Practical Mini-Subset Quality Signals

{practical_subset_markdown(practical_report)}

## Release Evaluation

{md_eval_table(records)}

## Release Gate

{md_release_gate_table(records)}

![Release gate latency](assets/release-gate-latency.png)

![Release gate coverage](assets/release-gate-coverage.png)

## Official Baseline vs OTQ Claims

| Item | Status |
| --- | --- |
| Official Qwen3.6-27B source scores | Imported from the official model card into `benchmarks/official_qwen36_baseline.csv` |
| OTQ `Q3_K_M` / `Q4_K_M` runtime | Measured with `llama-bench` on M1 Max |
| OTQ functional release gates | Measured with deterministic smoke and extended suites |
| Official benchmark deltas | Not claimed yet; requires running the same tasks/scoring on the GGUF artifacts |

## Allocation Transparency

{md_allocation_summary(records)}

![Tensor allocation](assets/tensor-allocation.png)

![Allocation policy](assets/allocation-policy.png)

The allocation plots show where OpenTQ spends precision. For example, the compact profile pushes bulk MLP tensors lower while preserving attention anchors and output-sensitive tensors at higher precision.

## Transparency Files

Each variant has full release evidence under `evidence/<quant>/`:

- `validation.json`
- `quality-eval.json`
- `release-eval.json`
- `opentq-plan.json`
- `tensor-types.txt`
- `tensor-types.annotated.tsv`
- `quantize-dry-run.log`

## Reproduce Release Evidence

```bash
git clone https://github.com/zlaabsi/opentq
cd opentq
uv sync
uv run python scripts/stage_qwen36_otq_gguf_repo.py
uv run python scripts/build_qwen36_release_report.py --repo artifacts/hf-gguf-canonical/Qwen3.6-27B-OTQ-GGUF
```

Run the same style of OTQ release evaluation:

```bash
LLAMA_CPP_DIR=/path/to/llama.cpp ./scripts/run_qwen36_otq_eval.sh
```

Run the long-context benchmark directly:

```bash
./build/bin/llama-bench \\
  -m models/Qwen3.6-27B-OTQ-GGUF/Qwen3.6-27B-OTQ-DYN-Q3_K_M.gguf \\
  -ngl 99 \\
  -fa on \\
  -p 8192 \\
  -n 128 \\
  -r 1 \\
  --no-warmup
```
"""


def usage_md() -> str:
    return f"""# Usage

This repo contains stock-compatible GGUF files. Use the filename that matches your memory budget.

## Download

```bash
hf download {REPO_ID} Qwen3.6-27B-OTQ-DYN-Q3_K_M.gguf --local-dir models/Qwen3.6-27B-OTQ-GGUF
hf download {REPO_ID} Qwen3.6-27B-OTQ-DYN-Q4_K_M.gguf --local-dir models/Qwen3.6-27B-OTQ-GGUF
```

## llama-server

```bash
./build/bin/llama-server \\
  -m models/Qwen3.6-27B-OTQ-GGUF/Qwen3.6-27B-OTQ-DYN-Q3_K_M.gguf \\
  -ngl 99 \\
  -fa \\
  -c 8192 \\
  --host 0.0.0.0 \\
  --port 8080
```

## llama.cpp Settings

| Setting | Value |
| --- | --- |
| Metal offload | `-ngl 99` |
| FlashAttention | `-fa` / `-fa on` |
| Validated context | `-c 8192` |
| First 32 GB pick | `Qwen3.6-27B-OTQ-DYN-Q3_K_M.gguf` |
| Quality-balanced pick | `Qwen3.6-27B-OTQ-DYN-Q4_K_M.gguf` |

For Qwen-style no-thinking checks, use the same scaffold as the release gate:

```text
<|im_start|>user
Your prompt
<|im_end|>
<|im_start|>assistant
<think>

</think>

```
"""


def benchmarks_md(records: list[dict[str, Any]], practical_report: Path) -> str:
    return f"""# Benchmarks

Benchmarks were run with stock `llama.cpp`, Metal offload and FlashAttention enabled. These are measured OTQ GGUF release results; official Qwen scores are included as a reference CSV, not as a claimed delta.

## Measured OTQ Runtime

{md_bench_table(records)}

![Runtime frontier](assets/runtime-frontier.png)

![Prefill decode tradeoff](assets/prefill-decode-tradeoff.png)

![Release scorecard](assets/release-scorecard.png)

## Functional Release Suites

{md_eval_table(records)}

## Practical Mini-Subset Quality Signals

{practical_subset_markdown(practical_report)}

![Release gate latency](assets/release-gate-latency.png)

![Release gate coverage](assets/release-gate-coverage.png)

## Allocation

{md_allocation_summary(records)}

![Tensor allocation](assets/tensor-allocation.png)

![Allocation policy](assets/allocation-policy.png)

## Official Reference

The official Qwen3.6-27B source-model benchmark table is exported to `benchmarks/official_qwen36_baseline.csv`.

Do not interpret those rows as OTQ deltas. A delta is valid only after the same benchmark task, prompt format, sample policy and scoring rule are run on the OTQ GGUF artifacts.

These suites are release gates, not full academic benchmarks. They cover small factual recall, reasoning, code output, JSON tool-call output, formatting and needle retrieval.
"""


def release_notes(records: list[dict[str, Any]]) -> str:
    rows = []
    for record in records:
        rows.append(
            f"- `{record['variant'].filename}`: {human_gib(record['bytes'])}, SHA256 `{record['sha256']}`"
        )
    return "\n".join(
        [
            "# Release Notes",
            "",
            "Canonical OpenTQ GGUF repo for Qwen3.6-27B.",
            "",
            "## Files",
            "",
            *rows,
            "",
            "## Packed And Metal GGUF Tracks",
            "",
            "OpenTQ Packed and Metal GGUF releases are intentionally not mixed into this repo. They require a dedicated runtime path and are staged under separate repo names.",
        ]
    )


def main() -> int:
    args = parse_args()
    output = Path(args.output)
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    (output / "evidence").mkdir()

    records: list[dict[str, Any]] = []
    for variant in VARIANTS:
        source = source_dir(Path(args.source_root), variant)
        gguf = source_gguf(source, variant)
        target = output / variant.filename
        link_file(gguf, target, args.link_mode)
        evidence = output / "evidence" / variant.quant
        evidence.mkdir(parents=True, exist_ok=True)
        paths = evidence_paths(Path(args.validation_root), Path(args.eval_root), variant)
        validation = rewrite_artifact_payload(paths["validation"], evidence / "validation.json", variant, target)
        quality = rewrite_artifact_payload(paths["quality"], evidence / "quality-eval.json", variant, target)
        release_eval = rewrite_artifact_payload(paths["release_eval"], evidence / "release-eval.json", variant, target)
        copy_transparency(source, evidence, variant)
        plan = load_json(evidence / "opentq-plan.json") if (evidence / "opentq-plan.json").exists() else {}
        records.append(
            {
                "variant": variant,
                "source": str(gguf),
                "bytes": target.stat().st_size,
                "sha256": sha256_file(target),
                "validation": validation,
                "quality": quality,
                "release_eval": release_eval,
                "plan": plan,
            }
        )

    banner_output = output / "assets" / "opentq-qwen36-hero.png"
    banner_output.parent.mkdir(parents=True, exist_ok=True)
    banner_source = Path(args.banner) if args.banner else DEFAULT_BANNER
    if banner_source.exists():
        shutil.copy2(banner_source, banner_output)
    else:
        make_banner(banner_output)
    practical_report = Path(args.practical_report)
    (output / "README.md").write_text(readme(records, practical_report), encoding="utf-8")
    (output / "USAGE.md").write_text(usage_md(), encoding="utf-8")
    (output / "BENCHMARKS.md").write_text(benchmarks_md(records, practical_report), encoding="utf-8")
    (output / "RELEASE_NOTES.md").write_text(release_notes(records), encoding="utf-8")
    dump_json(
        output / "opentq-gguf-release.json",
        {
            "schema": "opentq.hf_multi_gguf_release.v1",
            "repo_id": REPO_ID,
            "base_model": BASE_MODEL,
            "runtime": {
                "stock_llama_cpp": True,
                "requires": "stock llama.cpp",
                "native_opentq_required": False,
            },
            "artifacts": [
                {
                    "filename": record["variant"].filename,
                    "quant": record["variant"].quant,
                    "profile": record["variant"].profile,
                    "bytes": record["bytes"],
                    "gib": round(record["bytes"] / (1024**3), 2),
                    "sha256": record["sha256"],
                }
                for record in records
            ],
            "todo": {
                "benchmark_plots": "generated by scripts/build_qwen36_release_report.py",
                "packed_and_metal_releases": "planned; see docs/otq-packed-metal-release-plan.md",
            },
        },
    )
    if not args.skip_report:
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("build_qwen36_release_report.py")),
                "--repo",
                str(output),
            ],
            check=True,
        )
    print(f"staged {REPO_ID} at {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
