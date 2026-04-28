#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from opentq.hf_gguf_release import sha256_file
from opentq.hf_release import human_gib


BASE_MODEL = "Qwen/Qwen3.6-27B"
PACKED_REPO_ID = "zlaabsi/Qwen3.6-27B-OTQ-Packed"
METAL_REPO_ID = "zlaabsi/Qwen3.6-27B-OTQ-Metal-GGUF"


@dataclass(frozen=True)
class PackedVariant:
    source_slug: str
    public_slug: str
    role: str
    release_status: str


PACKED_VARIANTS = (
    PackedVariant("Qwen3.6-27B-TQ3_SB4", "Qwen3.6-27B-OTQ-TQ3_SB4", "compact packed TurboQuant candidate", "runtime-prepared"),
    PackedVariant("Qwen3.6-27B-TQ4_BAL_V2", "Qwen3.6-27B-OTQ-TQ4_BAL_V2", "balanced packed TurboQuant candidate", "runtime-prepared"),
    PackedVariant("Qwen3.6-27B-TQ4_SB4", "Qwen3.6-27B-OTQ-TQ4_SB4", "SB4 packed TurboQuant candidate", "runtime-prepared"),
    PackedVariant("Qwen3.6-27B-TQ4R2", "Qwen3.6-27B-OTQ-TQ4R2", "R2 packed TurboQuant experiment", "experimental"),
    PackedVariant("Qwen3.6-27B-TQ4R4", "Qwen3.6-27B-OTQ-TQ4R4", "R4 packed TurboQuant experiment", "experimental"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage Qwen3.6-27B OpenTQ runtime release repos without uploading.")
    parser.add_argument("--output-root", default="artifacts/hf-runtime")
    parser.add_argument("--packed-root", default="artifacts/qwen3.6-27b-packed")
    parser.add_argument("--metal-gguf-root", default="artifacts/qwen3.6-27b-gguf")
    parser.add_argument("--validation-root", default="artifacts/qwen3.6-27b-validation")
    parser.add_argument("--link-mode", choices=("hardlink", "copy", "symlink"), default="hardlink")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def link_file(source: Path, target: Path, mode: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        target.unlink()
    if mode == "hardlink":
        os.link(source, target)
    elif mode == "symlink":
        target.symlink_to(source.resolve())
    else:
        shutil.copy2(source, target)


def link_tree(source: Path, target: Path, mode: str) -> None:
    for item in source.rglob("*"):
        if item.is_dir():
            continue
        if ".cache" in item.parts:
            continue
        rel = item.relative_to(source)
        link_file(item, target / rel, mode)


def dir_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def count_files(path: Path, suffix: str | None = None) -> int:
    return sum(1 for item in path.rglob("*") if item.is_file() and (suffix is None or item.name.endswith(suffix)))


def find_font(candidates: list[str], size: int) -> ImageFont.FreeTypeFont:
    for candidate in candidates:
        path = Path(candidate).expanduser()
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default(size=size)


def dither_mask(width: int, height: int) -> Image.Image:
    y, x = np.mgrid[0:height, 0:width]
    pattern = ((x * 11 + y * 17) % 23) / 22.0
    waves = (np.sin(x / 47.0) + np.cos(y / 61.0) + np.sin((x - y) / 89.0)) / 3.0
    mask = ((pattern + waves * 0.31) > 0.57).astype(np.uint8) * 255
    return Image.fromarray(mask, mode="L").filter(ImageFilter.GaussianBlur(0.16))


def make_banner(path: Path, title: str, subtitle: str, badges: list[tuple[str, str]]) -> None:
    width, height = 1600, 520
    y, x = np.mgrid[0:height, 0:width]
    left = np.array([255, 118, 60], dtype=np.float32)
    mid = np.array([238, 144, 255], dtype=np.float32)
    right = np.array([28, 94, 255], dtype=np.float32)
    t = x / (width - 1)
    gradient = np.zeros((height, width, 3), dtype=np.float32)
    first = t < 0.52
    a = np.clip(t / 0.52, 0, 1)[..., None]
    b = np.clip((t - 0.52) / 0.48, 0, 1)[..., None]
    gradient[first] = left * (1 - a[first]) + mid * a[first]
    gradient[~first] = mid * (1 - b[~first]) + right * b[~first]
    vertical = (0.98 - 0.38 * (y / (height - 1)))[..., None]
    image = Image.fromarray(np.clip(gradient * vertical, 0, 255).astype(np.uint8), mode="RGB")
    overlay = Image.new("RGB", (width, height), "#121112")
    image = Image.composite(overlay, image, dither_mask(width, height).point(lambda px: int(px * 0.32)))
    draw = ImageDraw.Draw(image, "RGBA")

    draw.rounded_rectangle((26, 26, width - 26, height - 26), radius=46, outline=(255, 255, 255, 62), width=2)
    draw.ellipse((980, -130, 1480, 370), fill=(18, 17, 18, 55))
    draw.ellipse((1128, 108, 1690, 670), fill=(255, 255, 255, 22))
    draw.rounded_rectangle((78, 82, 386, 132), radius=24, fill=(18, 17, 18, 212))
    draw.rounded_rectangle((406, 82, 676, 132), radius=24, fill=(255, 255, 255, 48))

    pixel_font = find_font(["~/Library/Fonts/GeistPixel-Regular.ttf", "/System/Library/Fonts/SFNSMono.ttf", "/System/Library/Fonts/Menlo.ttc"], 34)
    title_font = find_font(
        ["~/Library/Fonts/KHTeka-Regular.ttf", "/System/Library/Fonts/Avenir Next Condensed.ttc", "/System/Library/Fonts/HelveticaNeue.ttc"],
        84,
    )
    subtitle_font = find_font(
        ["~/Library/Fonts/KHTeka-Regular.ttf", "/System/Library/Fonts/Avenir Next Condensed.ttc", "/System/Library/Fonts/HelveticaNeue.ttc"],
        39,
    )
    small_font = find_font(["/System/Library/Fonts/SFNSMono.ttf", "/System/Library/Fonts/Menlo.ttc"], 26)

    draw.text((104, 91), "OpenTQ", font=pixel_font, fill=(255, 255, 255, 238))
    draw.text((430, 92), "TurboQuant", font=pixel_font, fill=(18, 17, 18, 230))
    draw.text((78, 182), title, font=title_font, fill=(255, 255, 255, 246))
    draw.text((82, 286), subtitle, font=subtitle_font, fill=(255, 255, 255, 224))

    x0 = 82
    for label, value in badges:
        text = f"{label}  /  {value}"
        bbox = draw.textbbox((0, 0), text, font=small_font)
        w = bbox[2] - bbox[0] + 42
        draw.rounded_rectangle((x0, 382, x0 + w, 438), radius=18, fill=(18, 17, 18, 212), outline=(255, 255, 255, 54), width=1)
        draw.text((x0 + 22, 397), text, font=small_font, fill=(255, 255, 255, 234))
        x0 += w + 18

    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, optimize=True)


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


def staged_packed_readme(records: list[dict[str, Any]]) -> str:
    rows = "\n".join(
        f"| `{row['public_slug']}/` | {human_gib(row['bytes'])} | {row['otq_files']} | {row['role']} | `{row['release_status']}` |"
        for row in records
    )
    return f"""---
base_model: {BASE_MODEL}
base_model_relation: quantized
license: apache-2.0
tags:
- opentq
- turboquant
- packed
- custom-runtime
- qwen3.6
- apple-silicon
- metal
pipeline_tag: image-text-to-text
quantized_by: zlaabsi
---

# Qwen3.6-27B-OTQ-Packed

![OpenTQ Packed TurboQuant banner](assets/opentq-qwen36-packed.png)

OpenTQ Packed contains native TurboQuant weight artifacts for `{BASE_MODEL}`.

This is **not** the stock llama.cpp GGUF track. These files require an OpenTQ loader/runtime path and should stay private or gated until the runtime is public and benchmark-gated.

{packed_gate_markdown()}

## Files

| Directory | Size | `.otq` tensors | Role | Status |
| --- | ---: | ---: | --- | --- |
{rows}

## Runtime Contract

- `.otq` files are OpenTQ packed tensor payloads.
- `opentq-pack.json` is the per-variant manifest.
- `gguf-plan.json` records the intended GGUF/custom-runtime export plan when available.
- These artifacts are useful for runtime development, regression testing and reproducible packaging.
- They are not importable by LM Studio, Ollama or stock llama.cpp.

## Naming

- `OTQ`: OpenTQ release/tooling brand.
- `TQ`: TurboQuant weight quantization family.
- `SB4`, `BAL_V2`, `R2`, `R4`: OpenTQ scheme/revision labels.
"""


def staged_packed_usage() -> str:
    return """# Usage

This repo is staged for OpenTQ runtime development.

```bash
python -m json.tool Qwen3.6-27B-OTQ-TQ3_SB4/opentq-pack.json | head -80
```

Do not present these artifacts as stock GGUFs. Public inference requires the OpenTQ runtime path to be released and documented first.
"""


def stage_packed(output_root: Path, packed_root: Path, link_mode: str) -> dict[str, Any]:
    output = output_root / "Qwen3.6-27B-OTQ-Packed"
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)

    records: list[dict[str, Any]] = []
    for variant in PACKED_VARIANTS:
        source = packed_root / variant.source_slug
        if not source.exists():
            continue
        target = output / variant.public_slug
        link_tree(source, target, link_mode)
        pack_manifest = target / "opentq-pack.json"
        payload = load_json(pack_manifest)
        totals = payload.get("totals", {})
        records.append(
            {
                "source_slug": variant.source_slug,
                "public_slug": variant.public_slug,
                "role": variant.role,
                "release_status": variant.release_status,
                "bytes": dir_size(target),
                "otq_files": count_files(target, ".otq"),
                "manifest_sha256": sha256_file(pack_manifest),
                "total_values": totals.get("values"),
                "payload_bytes": totals.get("payload_bytes"),
            }
        )

    make_banner(
        output / "assets" / "opentq-qwen36-packed.png",
        "Qwen3.6 27B Packed",
        "Native TurboQuant tensor packs for the OpenTQ runtime track",
        [("Packed", "custom runtime"), ("Variants", str(len(records))), ("Status", "staged private")],
    )
    (output / "README.md").write_text(staged_packed_readme(records), encoding="utf-8")
    (output / "USAGE.md").write_text(staged_packed_usage(), encoding="utf-8")
    dump_json(
        output / "opentq-packed-release.json",
        {
            "schema": "opentq.hf_packed_release.v1",
            "repo_id": PACKED_REPO_ID,
            "base_model": BASE_MODEL,
            "public_release_ready": False,
            "reason": "requires public OpenTQ loader/runtime and release benchmarks",
            "artifacts": records,
        },
    )
    return {"repo_id": PACKED_REPO_ID, "path": str(output), "artifacts": records}


def validation_pass(path: Path) -> bool:
    payload = load_json(path)
    return bool(payload.get("overall_pass"))


def staged_metal_readme(record: dict[str, Any], blocked: list[dict[str, Any]]) -> str:
    blocked_rows = "\n".join(f"| `{row['filename']}` | {human_gib(row['bytes'])} | {row['reason']} |" for row in blocked)
    return f"""---
base_model: {BASE_MODEL}
base_model_relation: quantized
license: apache-2.0
library_name: llama.cpp
tags:
- gguf
- opentq
- turboquant
- custom-runtime
- metal
- apple-silicon
- qwen3.6
pipeline_tag: image-text-to-text
quantized_by: zlaabsi
---

# Qwen3.6-27B-OTQ-Metal-GGUF

![OpenTQ Metal TurboQuant banner](assets/opentq-qwen36-metal.png)

This staged repo is for OpenTQ custom GGUFs that require the OpenTQ Metal runtime path.

It is **not** the public stock llama.cpp repo. If users want a standard GGUF today, use `zlaabsi/Qwen3.6-27B-OTQ-GGUF`.

{metal_gate_markdown()}

## Release Candidate

| File | Size | SHA256 | Validation |
| --- | ---: | --- | --- |
| `{record['filename']}` | {human_gib(record['bytes'])} | `{record['sha256']}` | `{record['validation_status']}` |

## Runtime Contract

- Requires OpenTQ custom runtime support.
- Requires Metal kernels for efficient Apple Silicon inference.
- Not expected to run in stock LM Studio, Ollama or unpatched llama.cpp.
- Public release should stay blocked until long-context runtime benchmarks pass.

## Not Staged As Release Candidates

| Local file | Size | Reason |
| --- | ---: | --- |
{blocked_rows}
"""


def staged_metal_usage(filename: str) -> str:
    return f"""# Usage

This staged repo targets OpenTQ runtime development, not stock GGUF loading.

```bash
python -m json.tool evidence/TQ3_SB4/Qwen3.6-27B-TQ3_SB4-metal.json | head -80
```

Before public release, rerun smoke, long-context and quality gates against the current OpenTQ Metal runtime.
"""


def stage_metal(output_root: Path, gguf_root: Path, validation_root: Path, link_mode: str) -> dict[str, Any]:
    output = output_root / "Qwen3.6-27B-OTQ-Metal-GGUF"
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)

    source = gguf_root / "Qwen3.6-27B-TQ3_SB4" / "Qwen3.6-27B-TQ3_SB4.gguf"
    target_name = "Qwen3.6-27B-OTQ-TQ3_SB4-Metal.gguf"
    target = output / target_name
    link_file(source, target, link_mode)

    evidence_dir = output / "evidence" / "TQ3_SB4"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    validations = sorted(validation_root.glob("Qwen3.6-27B-TQ3_SB4-metal*.json"))
    for validation in validations:
        shutil.copy2(validation, evidence_dir / validation.name)
    validation_status = "passed" if validations and all(validation_pass(path) for path in validations) else "blocked"

    record = {
        "filename": target_name,
        "source_filename": source.name,
        "bytes": target.stat().st_size,
        "sha256": sha256_file(target),
        "validation_status": validation_status,
        "validation_files": [str((evidence_dir / path.name).relative_to(output)) for path in validations],
    }
    blocked: list[dict[str, Any]] = []
    for path in sorted(gguf_root.glob("*/*.gguf")):
        if path == source:
            continue
        reason = "not release-gated with Metal validation"
        if path.name == "Qwen3.6-27B-TQ4_SB4.gguf":
            reason = "size is inconsistent with packed payload; requires re-export audit"
        blocked.append({"filename": path.name, "bytes": path.stat().st_size, "reason": reason})

    make_banner(
        output / "assets" / "opentq-qwen36-metal.png",
        "Qwen3.6 27B Metal",
        "Custom TurboQuant GGUFs for OpenTQ kernels on Apple Silicon",
        [("TQ3_SB4", human_gib(record["bytes"])), ("Runtime", "OpenTQ Metal"), ("Status", validation_status)],
    )
    (output / "README.md").write_text(staged_metal_readme(record, blocked), encoding="utf-8")
    (output / "USAGE.md").write_text(staged_metal_usage(target_name), encoding="utf-8")
    dump_json(
        output / "opentq-metal-gguf-release.json",
        {
            "schema": "opentq.hf_metal_gguf_release.v1",
            "repo_id": METAL_REPO_ID,
            "base_model": BASE_MODEL,
            "public_release_ready": False,
            "reason": "requires public OpenTQ Metal runtime and long-context release benchmark",
            "artifact": record,
            "blocked_local_artifacts": blocked,
        },
    )
    return {"repo_id": METAL_REPO_ID, "path": str(output), "artifact": record, "blocked": blocked}


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    packed = stage_packed(output_root, Path(args.packed_root), args.link_mode)
    metal = stage_metal(output_root, Path(args.metal_gguf_root), Path(args.validation_root), args.link_mode)
    dump_json(
        output_root / "qwen3.6-27b-otq-runtime-staging.json",
        {
            "schema": "opentq.runtime_staging.v1",
            "base_model": BASE_MODEL,
            "repos": [packed, metal],
        },
    )
    print(f"staged {PACKED_REPO_ID} at {packed['path']}")
    print(f"staged {METAL_REPO_ID} at {metal['path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
