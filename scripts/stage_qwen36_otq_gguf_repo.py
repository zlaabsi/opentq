#!/usr/bin/env python3
from __future__ import annotations

import argparse
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


def readme(records: list[dict[str, Any]]) -> str:
    rows = []
    for record in records:
        variant = record["variant"]
        rows.append(
            f"| `{variant.filename}` | `{variant.quant}` | {human_gib(record['bytes'])} | `{record['sha256']}` | {variant.target} |"
        )
    file_table = "\n".join(["| File | Quant | Size | SHA256 | Target |", "| --- | --- | ---: | --- | --- |", *rows])
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
pipeline_tag: text-generation
language:
- en
quantized_by: zlaabsi
---

# Qwen3.6-27B-OTQ-GGUF

![OpenTQ TurboQuant Qwen3.6 banner](assets/opentq-qwen36-hero.png)

**OpenTQ TurboQuant dynamic-compatible GGUFs** for `{BASE_MODEL}`.

This repository is the stock `llama.cpp` track: OpenTQ applies a TurboQuant-inspired tensor-level allocation policy, but the published files use standard GGUF quantization types. No custom OpenTQ runtime is required for these GGUFs.

## Files

{file_table}

## Naming

- `OTQ`: OpenTQ, the release/tooling brand.
- `TurboQuant`: the quantization family and design direction.
- `DYN`: dynamic tensor-level allocation; different tensor families receive different GGUF quant types.
- `Q3_K_M` / `Q4_K_M`: standard GGUF quant names recognized by Hugging Face and stock `llama.cpp`.

## Which File Should I Use?

- `Q3_K_M`: first pick for 32 GB Apple Silicon and larger app/tool contexts.
- `Q4_K_M`: quality-balanced pick; usable on 32 GB at moderate context, more comfortable on 48 GB+.

## Runtime Compatibility

- `llama.cpp`, `llama-cli`, `llama-server`: supported.
- LM Studio and Ollama local GGUF import: expected to work as standard GGUF loaders.
- OpenTQ custom runtime: not required for this repo.
- Native TurboQuant/OpenTQ tensor formats: separate release track, not mixed into this GGUF repo.

## Benchmarks

{md_bench_table(records)}

## Release Evaluation

{md_eval_table(records)}

## Benchmark Plots

![Throughput](assets/benchmark-throughput.png)

![Release latency](assets/eval-latency.png)

![Category pass rate](assets/eval-pass-rate.png)

![Artifact size](assets/artifact-size.png)

![Official Qwen baseline](assets/official-language-baseline.png)

Official Qwen3.6-27B scores are used as the external BF16/source-model baseline. OpenTQ does not need to rerun BF16 locally for release plotting; deltas should only be claimed after matching benchmark tasks are run on the OTQ artifacts.

## Transparency Files

Each variant has full release evidence under `evidence/<quant>/`:

- `validation.json`
- `quality-eval.json`
- `release-eval.json`
- `opentq-plan.json`
- `tensor-types.txt`
- `tensor-types.annotated.tsv`
- `quantize-dry-run.log`

## Usage

```bash
hf download {REPO_ID} Qwen3.6-27B-OTQ-DYN-Q3_K_M.gguf --local-dir models/Qwen3.6-27B-OTQ-GGUF

./build/bin/llama-cli \\
  -m models/Qwen3.6-27B-OTQ-GGUF/Qwen3.6-27B-OTQ-DYN-Q3_K_M.gguf \\
  -ngl 99 \\
  -fa \\
  -c 8192 \\
  -p "<|im_start|>user\\nExplain prefill vs decode throughput in one paragraph.<|im_end|>\\n<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n"
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


def benchmarks_md(records: list[dict[str, Any]]) -> str:
    return f"""# Benchmarks

Benchmarks were run with stock `llama.cpp`, Metal offload and FlashAttention enabled.

## Throughput

{md_bench_table(records)}

## Functional Release Suites

{md_eval_table(records)}

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
        records.append(
            {
                "variant": variant,
                "source": str(gguf),
                "bytes": target.stat().st_size,
                "sha256": sha256_file(target),
                "validation": validation,
                "quality": quality,
                "release_eval": release_eval,
            }
        )

    banner_output = output / "assets" / "opentq-qwen36-hero.png"
    banner_output.parent.mkdir(parents=True, exist_ok=True)
    if args.banner:
        shutil.copy2(Path(args.banner), banner_output)
    else:
        make_banner(banner_output)
    (output / "README.md").write_text(readme(records), encoding="utf-8")
    (output / "USAGE.md").write_text(usage_md(), encoding="utf-8")
    (output / "BENCHMARKS.md").write_text(benchmarks_md(records), encoding="utf-8")
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
