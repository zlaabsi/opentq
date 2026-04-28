#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


PALETTE = {
    "q3": "#7cc7ff",
    "q4": "#0b3d73",
    "q5": "#2f80c2",
    "q6": "#125ea5",
    "q8": "#061a33",
    "f16": "#cfe9ff",
    "ref": "#6f93b7",
    "ink": "#0a1628",
    "muted": "#5d7692",
    "grid": "#d8e8f7",
    "paper": "#f8fbff",
}


@dataclass(frozen=True)
class VariantEvidence:
    name: str
    root: Path
    validation: dict[str, Any]
    quality: dict[str, Any]
    release_eval: dict[str, Any]
    plan: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build publication-quality Qwen3.6 OpenTQ release plots and tables.")
    parser.add_argument("--repo", default="artifacts/hf-gguf-canonical/Qwen3.6-27B-OTQ-GGUF")
    parser.add_argument("--quant-eval-root", help="Optional directory with OTQ-only eval JSONs.")
    parser.add_argument("--comparison-root", help="Deprecated alias for --quant-eval-root.")
    parser.add_argument(
        "--official-baseline",
        default="benchmarks/qwen36_official_language_baseline.json",
        help="Official Qwen baseline JSON. Used as an external reference, not rerun locally.",
    )
    return parser.parse_args()


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Helvetica Neue", "Avenir Next"],
            "font.size": 8.6,
            "axes.titlesize": 10.2,
            "axes.labelsize": 9.1,
            "xtick.labelsize": 8.1,
            "ytick.labelsize": 8.1,
            "legend.fontsize": 8.2,
            "figure.dpi": 150,
            "savefig.dpi": 320,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.06,
            "lines.linewidth": 2.0,
            "lines.markersize": 6,
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.75,
            "axes.axisbelow": True,
            "legend.frameon": False,
            "figure.constrained_layout.use": False,
        }
    )


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def evidence_variants(repo: Path) -> list[VariantEvidence]:
    evidence = repo / "evidence"
    variants: list[VariantEvidence] = []
    for root in sorted(path for path in evidence.iterdir() if path.is_dir()):
        variants.append(
            VariantEvidence(
                name=root.name,
                root=root,
                validation=load_json(root / "validation.json"),
                quality=load_json(root / "quality-eval.json"),
                release_eval=load_json(root / "release-eval.json"),
                plan=load_json(root / "opentq-plan.json"),
            )
        )
    if not variants:
        raise FileNotFoundError(f"no evidence variants under {evidence}")
    return variants


def variant_color(name: str) -> str:
    if "Q3" in name:
        return PALETTE["q3"]
    if "Q4" in name:
        return PALETTE["q4"]
    return PALETTE["ref"]


def clean_label(label: str) -> str:
    return label.replace("OTQ-DYN-", "").replace("_", "-")


def parse_bench_rows(validation: dict[str, Any], variant: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for phase in validation.get("phases", []):
        if phase.get("label") != "llama_bench":
            continue
        for line in str(phase.get("stdout_tail", "")).splitlines():
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            if len(cells) < 7 or not cells[-2].startswith(("pp", "tg")):
                continue
            match = re.search(r"([0-9]+(?:\.[0-9]+)?)", cells[-1])
            rows.append(
                {
                    "variant": variant,
                    "test": cells[-2],
                    "tokens_per_second": float(match.group(1)) if match else 0.0,
                    "throughput_text": cells[-1].replace(" +/- ", " +/- ").replace(" ± ", " +/- "),
                    "backend": cells[3],
                    "size": cells[1],
                }
            )
    return rows


def eval_summary_rows(variants: list[VariantEvidence]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for variant in variants:
        for suite, payload in (("smoke", variant.quality), ("release", variant.release_eval)):
            summary = payload.get("summary", {})
            rows.append(
                {
                    "variant": variant.name,
                    "suite": suite,
                    "total": summary.get("total", 0),
                    "passed": summary.get("passed", 0),
                    "pass_rate": summary.get("pass_rate", 0.0),
                    "latency_seconds_mean": summary.get("latency_seconds_mean", 0.0),
                    "latency_seconds_p50": summary.get("latency_seconds_p50", 0.0),
                    "latency_seconds_p95": summary.get("latency_seconds_p95", 0.0),
                    "duration_seconds_total": summary.get("duration_seconds_total", 0.0),
                }
            )
    return rows


def category_rows(variants: list[VariantEvidence]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for variant in variants:
        for suite, payload in (("smoke", variant.quality), ("release", variant.release_eval)):
            for category, summary in payload.get("summary", {}).get("categories", {}).items():
                rows.append(
                    {
                        "variant": variant.name,
                        "suite": suite,
                        "category": category,
                        "total": summary.get("total", 0),
                        "passed": summary.get("passed", 0),
                        "pass_rate": summary.get("pass_rate", 0.0),
                    }
                )
    return rows


def artifact_rows(repo: Path) -> list[dict[str, Any]]:
    manifest = load_json(repo / "opentq-gguf-release.json")
    return [
        {
            "variant": item["quant"],
            "filename": item["filename"],
            "bytes": item["bytes"],
            "gib": item["gib"],
            "sha256": item["sha256"],
        }
        for item in manifest.get("artifacts", [])
    ]


def tensor_rows(variants: list[VariantEvidence]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for variant in variants:
        for tensor_type, count in variant.plan.get("summary", {}).get("by_type", {}).items():
            rows.append({"variant": variant.name, "tensor_type": tensor_type, "count": count})
    return rows


def category_tensor_rows(variants: list[VariantEvidence]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for variant in variants:
        for key, count in variant.plan.get("summary", {}).get("by_category_type", {}).items():
            category, _, tensor_type = str(key).partition(":")
            rows.append(
                {
                    "variant": variant.name,
                    "category": category,
                    "tensor_type": tensor_type or "unknown",
                    "count": int(count),
                }
            )
    return rows


def official_baseline_rows(path: Path | None) -> list[dict[str, Any]]:
    if not path or not path.exists():
        return []
    payload = load_json(path)
    rows: list[dict[str, Any]] = []
    for row in payload.get("scores", []):
        rows.append(
            {
                "model": payload.get("model_id", "Qwen/Qwen3.6-27B"),
                "source": payload.get("source_url", ""),
                "category": row.get("category", ""),
                "benchmark": row.get("benchmark", ""),
                "score": row.get("score", ""),
                "unit": row.get("unit", "score"),
                "higher_is_better": row.get("higher_is_better", True),
                "notes": row.get("notes", ""),
            }
        )
    return rows


def quant_eval_rows(quant_eval_root: Path | None) -> list[dict[str, Any]]:
    if not quant_eval_root or not quant_eval_root.exists():
        return []
    rows: list[dict[str, Any]] = []
    for path in sorted(quant_eval_root.glob("*.json")):
        payload = load_json(path)
        summary = payload.get("summary", {})
        rows.append(
            {
                "model": path.stem,
                "total": summary.get("total", 0),
                "passed": summary.get("passed", 0),
                "pass_rate": summary.get("pass_rate", 0.0),
                "latency_seconds_mean": summary.get("latency_seconds_mean", 0.0),
                "latency_seconds_p95": summary.get("latency_seconds_p95", 0.0),
            }
        )
    return rows


def polish_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", which="major", linestyle="-", linewidth=0.75, alpha=0.24)
    ax.minorticks_on()
    ax.grid(True, axis="y", which="minor", linestyle="-", linewidth=0.25, alpha=0.12)
    ax.set_axisbelow(True)


def save_figure(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.7)
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.05)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.05)
    fig.savefig(stem.with_suffix(".png"), dpi=320, facecolor="white", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def bench_metric(rows: list[dict[str, Any]], variant: str, test: str) -> float:
    for row in rows:
        if row["variant"] == variant and row["test"] == test:
            return float(row["tokens_per_second"])
    return 0.0


def size_metric(rows: list[dict[str, Any]], variant: str) -> float:
    for row in rows:
        if row["variant"] == variant:
            return float(row["gib"])
    return 0.0


def plot_runtime_frontier(bench: list[dict[str, Any]], artifacts: list[dict[str, Any]], stem: Path) -> None:
    variants = sorted({row["variant"] for row in bench})
    sizes = [size_metric(artifacts, variant) for variant in variants]
    prefill = [bench_metric(bench, variant, "pp8192") for variant in variants]
    decode = [bench_metric(bench, variant, "tg128") for variant in variants]

    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    for variant, x, y, tg in zip(variants, sizes, prefill, decode):
        ax.scatter(
            x,
            y,
            s=240 + tg * 18,
            color=variant_color(variant),
            edgecolor=PALETTE["ink"],
            linewidth=0.9,
            zorder=3,
        )
        ax.annotate(
            f"{clean_label(variant)}\n{tg:.2f} tok/s decode",
            (x, y),
            textcoords="offset points",
            xytext=(12, -4 if "Q3" in variant else -24),
            fontsize=8.5,
            color=PALETTE["ink"],
        )
    ax.plot(sizes, prefill, color="#7fb5df", linewidth=1.4, alpha=0.8, zorder=2)
    ax.set_xlabel("GGUF artifact size (GiB)")
    ax.set_ylabel("prefill throughput at 8K context (tok/s)")
    ax.set_title("M1 Max size / prefill / decode frontier", loc="left", fontweight="semibold")
    ax.set_xlim(max(0, min(sizes) - 1.2), max(sizes) + 2.0)
    ax.set_ylim(max(0, min(prefill) - 2.0), max(prefill) + 4.0)
    ax.text(
        0.99,
        0.04,
        "bubble label = measured tg128 decode throughput",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color=PALETTE["muted"],
    )
    polish_axes(ax)
    save_figure(fig, stem)


def plot_prefill_decode_tradeoff(bench: list[dict[str, Any]], stem: Path) -> None:
    variants = sorted({row["variant"] for row in bench})
    tests = [("pp8192", "Prefill 8K"), ("tg128", "Decode 128")]
    y = np.arange(len(tests))

    fig, ax = plt.subplots(figsize=(7.0, 2.85))
    for idx, variant in enumerate(variants):
        values = [bench_metric(bench, variant, test) for test, _ in tests]
        offset = (idx - (len(variants) - 1) / 2) * 0.18
        ax.scatter(values, y + offset, s=95, color=variant_color(variant), edgecolor=PALETTE["ink"], linewidth=0.7, label=clean_label(variant), zorder=3)
        for value, yi in zip(values, y + offset):
            ax.annotate(f"{value:.2f}", (value, yi), textcoords="offset points", xytext=(8, -3), fontsize=8.2, color=PALETTE["ink"])
    ax.set_yticks(y, [label for _, label in tests])
    ax.set_xlabel("tokens / second")
    ax.set_title("Measured prefill vs decode tradeoff", loc="left", fontweight="semibold")
    ax.legend(loc="upper right", ncol=len(variants), handletextpad=0.35, columnspacing=0.9)
    polish_axes(ax)
    save_figure(fig, stem)


def plot_eval_latency(rows: list[dict[str, Any]], stem: Path) -> None:
    release = [row for row in rows if row["suite"] == "release"]
    variants = [row["variant"] for row in release]
    labels = [clean_label(v) for v in variants]
    mean = [float(row["latency_seconds_mean"]) for row in release]
    p95 = [float(row["latency_seconds_p95"]) for row in release]
    x = np.arange(len(variants))

    fig, ax = plt.subplots(figsize=(4.1, 3.0))
    bars = ax.bar(x, mean, color=[variant_color(v) for v in variants], edgecolor="#2d2d2d", linewidth=0.7, label="mean")
    ax.plot(x, p95, color="#0f5f9f", marker="o", linewidth=2.0, label="p95", zorder=3)
    ax.bar_label(bars, labels=[f"{value:.1f}s" for value in mean], padding=2, fontsize=8.5)
    for xi, value in zip(x, p95):
        ax.annotate(f"{value:.1f}s", (xi, value), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8.5, color=PALETTE["ink"])

    ax.set_xticks(x, labels)
    ax.set_ylabel("seconds / sample")
    ax.set_title("Release-gate latency by quantized artifact", loc="left", fontweight="semibold")
    ax.legend(loc="upper right", ncol=2, columnspacing=0.9)
    polish_axes(ax)
    save_figure(fig, stem)


def plot_pass_rate(rows: list[dict[str, Any]], stem: Path) -> None:
    release = [row for row in rows if row["suite"] == "release"]
    variants = sorted({row["variant"] for row in release})
    categories = sorted({row["category"] for row in release})
    matrix = np.zeros((len(variants), len(categories)), dtype=float)
    lookup = {(row["variant"], row["category"]): float(row["pass_rate"]) for row in release}
    for i, variant in enumerate(variants):
        for j, category in enumerate(categories):
            matrix[i, j] = lookup.get((variant, category), np.nan)

    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    image = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap="Blues", aspect="auto")
    ax.set_xticks(np.arange(len(categories)), [c.replace("_", " ") for c in categories], rotation=28, ha="right")
    ax.set_yticks(np.arange(len(variants)), [clean_label(v) for v in variants])
    for i in range(len(variants)):
        for j in range(len(categories)):
            value = matrix[i, j]
            ax.text(j, i, "n/a" if np.isnan(value) else f"{value:.0%}", ha="center", va="center", fontsize=9, color=PALETTE["ink"])

    ax.set_title("Release-gate coverage by category", loc="left", fontweight="semibold")
    cbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    cbar.ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
    save_figure(fig, stem)


def plot_release_scorecard(
    artifacts: list[dict[str, Any]],
    evals: list[dict[str, Any]],
    bench: list[dict[str, Any]],
    stem: Path,
) -> None:
    release = {row["variant"]: row for row in evals if row["suite"] == "release"}
    variants = sorted(release)
    labels = [clean_label(variant) for variant in variants]
    metrics = [
        ("Size", [size_metric(artifacts, variant) for variant in variants], "GiB", False),
        ("Prefill", [bench_metric(bench, variant, "pp8192") for variant in variants], "tok/s", True),
        ("Decode", [bench_metric(bench, variant, "tg128") for variant in variants], "tok/s", True),
        ("p95 latency", [float(release[variant]["latency_seconds_p95"]) for variant in variants], "s", False),
    ]
    normalized = np.zeros((len(metrics), len(variants)))
    for row_idx, (_, values, _, higher_is_better) in enumerate(metrics):
        arr = np.array(values, dtype=float)
        if np.allclose(arr.max(), arr.min()):
            normalized[row_idx] = 1.0
            continue
        scaled = (arr - arr.min()) / (arr.max() - arr.min())
        normalized[row_idx] = scaled if higher_is_better else 1.0 - scaled

    fig, ax = plt.subplots(figsize=(7.0, 3.1))
    image = ax.imshow(normalized, vmin=0.0, vmax=1.0, cmap="Blues", aspect="auto")
    ax.set_xticks(np.arange(len(variants)), labels)
    ax.set_yticks(np.arange(len(metrics)), [metric[0] for metric in metrics])
    for row_idx, (_, values, unit, _) in enumerate(metrics):
        for col_idx, value in enumerate(values):
            text_color = "white" if normalized[row_idx, col_idx] >= 0.58 else PALETTE["ink"]
            ax.text(col_idx, row_idx, f"{value:.2f} {unit}", ha="center", va="center", fontsize=8.2, color=text_color)
    ax.set_title("Release decision scorecard", loc="left", fontweight="semibold")
    cbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("relative fit for local release", fontsize=8.2)
    save_figure(fig, stem)


def plot_tensor_allocation(rows: list[dict[str, Any]], stem: Path) -> None:
    variants = sorted({row["variant"] for row in rows})
    tensor_types = ["F16", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "Q8_0"]
    lookup = {(row["variant"], row["tensor_type"]): int(row["count"]) for row in rows}
    totals = np.array([sum(lookup.get((variant, t), 0) for t in tensor_types) for variant in variants], dtype=float)
    fig, ax = plt.subplots(figsize=(7.0, 3.15))
    bottom = np.zeros(len(variants))
    colors = {"F16": PALETTE["f16"], "Q3_K": PALETTE["q3"], "Q4_K": "#4494d1", "Q5_K": PALETTE["q5"], "Q6_K": PALETTE["q6"], "Q8_0": PALETTE["q8"]}
    for tensor_type in tensor_types:
        counts = np.array([lookup.get((variant, tensor_type), 0) for variant in variants], dtype=float)
        values = np.divide(counts, totals, out=np.zeros_like(counts), where=totals > 0)
        ax.bar([clean_label(v) for v in variants], values, bottom=bottom, label=tensor_type, color=colors[tensor_type], edgecolor="white", linewidth=0.6)
        bottom += values
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
    ax.set_ylabel("share of mapped tensors")
    ax.set_title("Dynamic GGUF tensor-type allocation", loc="left", fontweight="semibold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=len(tensor_types), columnspacing=0.8, handletextpad=0.3)
    polish_axes(ax)
    save_figure(fig, stem)


def plot_allocation_policy(rows: list[dict[str, Any]], stem: Path) -> None:
    priority = [
        "mlp_proj",
        "linear_attn_proj",
        "self_attn_proj",
        "embeddings",
        "lm_head",
        "linear_attn_state",
        "layernorm",
    ]
    variants = sorted({row["variant"] for row in rows})
    categories = [category for category in priority if any(row["category"] == category for row in rows)]
    tensor_types = ["F16", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "Q8_0"]
    type_index = {tensor_type: idx for idx, tensor_type in enumerate(tensor_types)}
    fig, axes = plt.subplots(1, len(variants), figsize=(7.0, 3.45), sharey=True)
    if len(variants) == 1:
        axes = [axes]
    for ax, variant in zip(axes, variants):
        matrix = np.full((len(categories), len(tensor_types)), np.nan)
        for i, category in enumerate(categories):
            category_rows = [row for row in rows if row["variant"] == variant and row["category"] == category]
            total = sum(int(row["count"]) for row in category_rows)
            if total == 0:
                continue
            for row in category_rows:
                tensor_type = row["tensor_type"]
                if tensor_type in type_index:
                    matrix[i, type_index[tensor_type]] = int(row["count"]) / total
        ax.imshow(np.nan_to_num(matrix), vmin=0.0, vmax=1.0, cmap="Blues", aspect="auto")
        ax.set_title(clean_label(variant), fontweight="semibold")
        ax.set_xticks(np.arange(len(tensor_types)), tensor_types, rotation=35, ha="right")
        ax.set_yticks(np.arange(len(categories)), [category.replace("_", " ") for category in categories])
        for i in range(len(categories)):
            for j in range(len(tensor_types)):
                value = matrix[i, j]
                if np.isnan(value) or value <= 0:
                    continue
                text_color = "white" if value >= 0.75 else PALETTE["ink"]
                ax.text(j, i, f"{value:.0%}", ha="center", va="center", fontsize=7.0, color=text_color)
    axes[0].set_ylabel("tensor family")
    fig.suptitle("Where OpenTQ spends precision", x=0.02, y=1.0, ha="left", fontweight="semibold", fontsize=10.2)
    save_figure(fig, stem)


def plot_official_baseline(rows: list[dict[str, Any]], stem: Path) -> None:
    plot_rows = [row for row in rows if isinstance(row.get("score"), (int, float)) and float(row["score"]) <= 100.0]
    if not plot_rows:
        return
    category_rank = {"Coding Agent": 0, "Knowledge": 1, "STEM & Reasoning": 2}
    plot_rows = sorted(plot_rows, key=lambda row: (category_rank.get(str(row["category"]), 99), -float(row["score"])))
    selected: list[dict[str, Any]] = []
    for category in ("Coding Agent", "Knowledge", "STEM & Reasoning"):
        category_rows = [row for row in plot_rows if row["category"] == category]
        selected.extend(category_rows[:5])
    plot_rows = selected
    labels = [str(row["benchmark"]) for row in plot_rows]
    scores = [float(row["score"]) for row in plot_rows]
    colors = [("#9cd5ff" if row["category"] == "Coding Agent" else "#3187c7" if row["category"] == "Knowledge" else "#0b3d73") for row in plot_rows]
    y = np.arange(len(plot_rows))

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    bars = ax.barh(y, scores, color=colors, edgecolor="#2d2d2d", linewidth=0.55)
    ax.bar_label(bars, labels=[f"{score:.1f}" for score in scores], padding=3, fontsize=8.4)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("official Qwen score")
    ax.set_title("Official Qwen3.6-27B BF16 reference", loc="left", fontweight="semibold")
    polish_axes(ax)
    save_figure(fig, stem)


def plot_quant_eval(rows: list[dict[str, Any]], stem: Path) -> None:
    if not rows:
        return
    labels = [row["model"].replace("OTQ-DYN-", "").replace("_", "-") for row in rows]
    pass_rate = [float(row["pass_rate"]) for row in rows]
    p95 = [float(row["latency_seconds_p95"]) for row in rows]
    x = np.arange(len(rows))

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    bars = axes[0].bar(x, pass_rate, color=[variant_color(row["model"]) for row in rows], edgecolor="#2d2d2d", linewidth=0.7)
    axes[0].bar_label(bars, labels=[f"{value:.0%}" for value in pass_rate], padding=2, fontsize=8.5)
    axes[0].set_xticks(x, labels, rotation=18, ha="right")
    axes[0].set_ylim(0, 1.05)
    axes[0].yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
    axes[0].set_title("OTQ eval pass rate", loc="left", fontweight="semibold")
    polish_axes(axes[0])

    bars = axes[1].bar(x, p95, color=[variant_color(row["model"]) for row in rows], edgecolor="#2d2d2d", linewidth=0.7)
    axes[1].bar_label(bars, labels=[f"{value:.1f}s" for value in p95], padding=2, fontsize=8.5)
    axes[1].set_xticks(x, labels, rotation=18, ha="right")
    axes[1].set_ylabel("p95 seconds")
    axes[1].set_title("OTQ eval latency", loc="left", fontweight="semibold")
    polish_axes(axes[1])
    save_figure(fig, stem)


def figure_links(name: str) -> str:
    return f"![{name}](assets/{name}.png)\n\nVector: [`SVG`](assets/{name}.svg) | [`PDF`](assets/{name}.pdf)"


def write_report_md(
    repo: Path,
    official_rows: list[dict[str, Any]],
    quant_evals: list[dict[str, Any]],
) -> None:
    official_note = (
        "Official Qwen3.6-27B language benchmark scores are imported as an external reference table in `benchmarks/official_qwen36_baseline.csv`. They are not plotted against OTQ until matching benchmark tasks are run on these GGUF files."
        if official_rows
        else "No official baseline JSON was available when this report was generated."
    )
    quant_note = (
        "OTQ-only comparison JSONs are available in `benchmarks/quant_eval.csv`."
        if quant_evals
        else "No additional OTQ benchmark subset is attached yet. Add OTQ-only task runs when comparing against official benchmark families."
    )
    lines = [
        "# Benchmarks",
        "",
        "Benchmarks are split into measured OTQ runtime frontiers, release-gate checks, allocation transparency, and the official Qwen reference table.",
        "",
        "## Measured OTQ Runtime",
        "",
        figure_links("runtime-frontier"),
        "",
        figure_links("prefill-decode-tradeoff"),
        "",
        figure_links("release-scorecard"),
        "",
        figure_links("tensor-allocation"),
        "",
        figure_links("allocation-policy"),
        "",
        "## Release Gates",
        "",
        figure_links("release-gate-latency"),
        "",
        figure_links("release-gate-coverage"),
        "",
        "These release suites are deterministic guardrails, not substitutes for full academic benchmarks.",
        "",
        "## Official Qwen Baseline",
        "",
        official_note,
        "",
        "Deltas versus the official Qwen baseline must only be reported for tasks that are actually run on the OTQ artifacts with the same task definition and scoring rule.",
        "",
        "## OTQ Task Runs",
        "",
        quant_note,
        "",
        "## CSV Data",
        "",
        "- `benchmarks/throughput.csv`",
        "- `benchmarks/eval_summary.csv`",
        "- `benchmarks/category_pass_rate.csv`",
        "- `benchmarks/artifacts.csv`",
        "- `benchmarks/tensor_allocation.csv`",
        "- `benchmarks/category_tensor_allocation.csv`",
        "- `benchmarks/official_qwen36_baseline.csv`",
        "- `benchmarks/quant_eval.csv` when OTQ-only task runs are present",
    ]
    (repo / "BENCHMARKS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    configure_matplotlib()
    repo = Path(args.repo)
    variants = evidence_variants(repo)
    assets = repo / "assets"
    data_dir = repo / "benchmarks"
    quant_eval_arg = args.quant_eval_root or args.comparison_root

    bench = [row for variant in variants for row in parse_bench_rows(variant.validation, variant.name)]
    evals = eval_summary_rows(variants)
    categories = category_rows(variants)
    artifacts = artifact_rows(repo)
    tensors = tensor_rows(variants)
    category_tensors = category_tensor_rows(variants)
    official_rows = official_baseline_rows(Path(args.official_baseline) if args.official_baseline else None)
    quant_evals = quant_eval_rows(Path(quant_eval_arg) if quant_eval_arg else None)

    write_csv(data_dir / "throughput.csv", bench, ["variant", "test", "tokens_per_second", "throughput_text", "backend", "size"])
    write_csv(
        data_dir / "eval_summary.csv",
        evals,
        [
            "variant",
            "suite",
            "total",
            "passed",
            "pass_rate",
            "latency_seconds_mean",
            "latency_seconds_p50",
            "latency_seconds_p95",
            "duration_seconds_total",
        ],
    )
    write_csv(data_dir / "category_pass_rate.csv", categories, ["variant", "suite", "category", "total", "passed", "pass_rate"])
    write_csv(data_dir / "artifacts.csv", artifacts, ["variant", "filename", "bytes", "gib", "sha256"])
    write_csv(data_dir / "tensor_allocation.csv", tensors, ["variant", "tensor_type", "count"])
    write_csv(data_dir / "category_tensor_allocation.csv", category_tensors, ["variant", "category", "tensor_type", "count"])
    write_csv(
        data_dir / "official_qwen36_baseline.csv",
        official_rows,
        ["model", "source", "category", "benchmark", "score", "unit", "higher_is_better", "notes"],
    )
    if quant_evals:
        write_csv(data_dir / "quant_eval.csv", quant_evals, ["model", "total", "passed", "pass_rate", "latency_seconds_mean", "latency_seconds_p95"])

    plot_runtime_frontier(bench, artifacts, assets / "runtime-frontier")
    plot_prefill_decode_tradeoff(bench, assets / "prefill-decode-tradeoff")
    plot_release_scorecard(artifacts, evals, bench, assets / "release-scorecard")
    plot_eval_latency(evals, assets / "release-gate-latency")
    plot_pass_rate(categories, assets / "release-gate-coverage")
    plot_tensor_allocation(tensors, assets / "tensor-allocation")
    plot_allocation_policy(category_tensors, assets / "allocation-policy")
    plot_quant_eval(quant_evals, assets / "quant-eval-summary")
    write_report_md(repo, official_rows, quant_evals)
    print(f"wrote publication-grade benchmark report assets under {repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
