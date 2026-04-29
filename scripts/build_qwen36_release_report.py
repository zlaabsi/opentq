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
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle
import numpy as np


PALETTE = {
    "paper": "#fbfaf6",
    "panel": "#f5f2eb",
    "ink": "#171717",
    "muted": "#68645d",
    "faint": "#a8a29a",
    "grid": "#e6e0d6",
    "spine": "#8e887e",
    "cream": "#e8e2d4",
    "copper": "#c86f24",
    "rust": "#8d4b3f",
    "maroon": "#5a3430",
    "blue": "#2f73b7",
    "sky": "#76a9d4",
    "green": "#4d8a58",
    "olive": "#76865c",
    "black": "#211817",
}
TYPE_COLORS = {
    "F16": "#e8e2d4",
    "Q3_K": "#2f73b7",
    "Q4_K": "#76a9d4",
    "Q5_K": "#c86f24",
    "Q6_K": "#8d4b3f",
    "Q8_0": "#211817",
}
SCORE_CMAP = LinearSegmentedColormap.from_list(
    "opentq_score",
    ["#fbfaf6", "#e7e0d2", "#b6c6a2", "#4d8a58"],
)
DELTA_CMAP = LinearSegmentedColormap.from_list(
    "opentq_delta",
    [PALETTE["rust"], "#eadfd5", PALETTE["paper"], "#d5dfc3", PALETTE["green"]],
)
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
            "font.sans-serif": ["DejaVu Sans", "Helvetica Neue", "Avenir Next", "Inter"],
            "font.size": 8.0,
            "axes.titlesize": 10.8,
            "axes.labelsize": 8.4,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.3,
            "figure.dpi": 150,
            "savefig.dpi": 320,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.08,
            "lines.linewidth": 1.6,
            "lines.markersize": 5.0,
            "axes.linewidth": 0.72,
            "axes.grid": True,
            "grid.color": PALETTE["grid"],
            "grid.alpha": 0.72,
            "grid.linewidth": 0.72,
            "grid.linestyle": "--",
            "axes.axisbelow": True,
            "axes.facecolor": PALETTE["paper"],
            "figure.facecolor": PALETTE["paper"],
            "savefig.facecolor": PALETTE["paper"],
            "axes.edgecolor": PALETTE["spine"],
            "axes.labelcolor": PALETTE["ink"],
            "xtick.color": PALETTE["muted"],
            "ytick.color": PALETTE["muted"],
            "text.color": PALETTE["ink"],
            "legend.frameon": True,
            "legend.facecolor": PALETTE["paper"],
            "legend.edgecolor": "#d4cec4",
            "legend.framealpha": 0.95,
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
        return PALETTE["blue"]
    if "Q4" in name:
        return PALETTE["copper"]
    if "Q5" in name:
        return PALETTE["maroon"]
    return PALETTE["faint"]


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
    ax.set_facecolor(PALETTE["paper"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(PALETTE["spine"])
    ax.spines["bottom"].set_color(PALETTE["spine"])
    ax.tick_params(colors=PALETTE["muted"], length=3.5, width=0.75)
    ax.grid(True, axis="y", which="major", linestyle="--", linewidth=0.72, alpha=0.78)
    ax.minorticks_on()
    ax.grid(True, axis="y", which="minor", linestyle=":", linewidth=0.45, alpha=0.35)
    ax.set_axisbelow(True)


def add_title(ax: plt.Axes, title: str, subtitle: str | None = None) -> None:
    ax.set_title(title, loc="left", fontweight="semibold", fontsize=11.2, pad=15 if subtitle else 8)
    if subtitle:
        ax.text(
            0.0,
            1.02,
            subtitle,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=7.7,
            color=PALETTE["muted"],
        )


def add_cell_grid(ax: plt.Axes, rows: int, cols: int, *, color: str = "#e7e1d8") -> None:
    ax.grid(False)
    ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
    ax.grid(which="minor", color=color, linestyle="-", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)


def save_figure(fig: plt.Figure, stem: Path, caption: str | None = None, *, tight: bool = True) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    if caption:
        fig.text(0.012, 0.014, caption, ha="left", va="bottom", fontsize=6.8, color=PALETTE["muted"])
        if tight:
            fig.tight_layout(pad=0.9, rect=(0.0, 0.075, 1.0, 1.0))
    elif tight:
        fig.tight_layout(pad=0.9)
    for suffix in (".svg", ".pdf"):
        fig.savefig(stem.with_suffix(suffix), bbox_inches="tight", pad_inches=0.06, facecolor=PALETTE["paper"])
    fig.savefig(stem.with_suffix(".png"), dpi=320, facecolor=PALETTE["paper"], bbox_inches="tight", pad_inches=0.06)
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

    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    order = np.argsort(sizes)
    ax.plot(
        np.array(sizes)[order],
        np.array(prefill)[order],
        color=PALETTE["faint"],
        linewidth=1.45,
        alpha=0.85,
        zorder=1,
    )
    for variant, x, y, tg in zip(variants, sizes, prefill, decode):
        marker = "o" if "Q3" in variant else "s" if "Q4" in variant else "D"
        ax.scatter(
            x,
            y,
            s=170 + tg * 15,
            color=variant_color(variant),
            marker=marker,
            edgecolor=PALETTE["paper"],
            linewidth=1.8,
            zorder=3,
        )
        ax.scatter(x, y, s=205 + tg * 15, facecolor="none", edgecolor=variant_color(variant), linewidth=1.05, zorder=2)
        label_offset = (20, -8) if "Q5" in variant else (14, -2 if "Q3" in variant else -24)
        label_align = "left"
        ax.annotate(
            f"{clean_label(variant)}\n{x:.1f} GiB · {tg:.2f} tok/s decode",
            (x, y),
            textcoords="offset points",
            xytext=label_offset,
            fontsize=7.4,
            color=PALETTE["ink"],
            ha=label_align,
            arrowprops={"arrowstyle": "-", "color": PALETTE["faint"], "lw": 0.8},
        )
    ax.set_xlabel("GGUF artifact size (GiB)")
    ax.set_ylabel("8K prefill throughput (tokens/s)")
    add_title(
        ax,
        "M1 Max runtime frontier",
        "Each point is a released GGUF artifact; marker size tracks measured tg128 decode throughput.",
    )
    ax.set_xlim(max(0, min(sizes) - 1.2), max(sizes) + 2.8)
    ax.set_ylim(max(0, min(prefill) - 2.0), max(prefill) + 4.0)
    ax.text(
        0.015,
        0.06,
        "Smaller and higher is better for local agent workloads with long context.",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.2,
        color=PALETTE["muted"],
    )
    polish_axes(ax)
    save_figure(fig, stem, "M1 Max 32 GB, llama.cpp Metal backend, 8192-token prefill and 128-token decode checks.")


def plot_prefill_decode_tradeoff(bench: list[dict[str, Any]], stem: Path) -> None:
    variants = sorted({row["variant"] for row in bench})
    tests = [("pp8192", "Prefill 8K"), ("tg128", "Decode 128")]
    y = np.arange(len(tests))

    fig, ax = plt.subplots(figsize=(7.6, 3.05))
    ax.axvspan(0, 12, color=PALETTE["panel"], zorder=0)
    for idx, variant in enumerate(variants):
        values = [bench_metric(bench, variant, test) for test, _ in tests]
        offset = (idx - (len(variants) - 1) / 2) * 0.17
        for value, yi in zip(values, y + offset):
            ax.hlines(yi, 0, value, color=variant_color(variant), linewidth=1.15, alpha=0.52, zorder=1)
        ax.scatter(
            values,
            y + offset,
            s=98,
            color=variant_color(variant),
            edgecolor=PALETTE["paper"],
            linewidth=1.4,
            label=clean_label(variant),
            zorder=3,
        )
        for value, yi in zip(values, y + offset):
            ax.annotate(f"{value:.2f}", (value, yi), textcoords="offset points", xytext=(8, -3), fontsize=8.1, color=PALETTE["ink"])
    ax.set_yticks(y, [label for _, label in tests])
    ax.set_xlabel("tokens per second")
    add_title(ax, "Prefill / decode tradeoff", "Same runtime, same prompt sizes; no cross-runtime score mixing.")
    ax.legend(loc="lower right", ncol=len(variants), handletextpad=0.35, columnspacing=0.9)
    ax.set_ylim(-0.55, len(tests) - 0.45)
    polish_axes(ax)
    ax.grid(True, axis="x", which="major")
    save_figure(fig, stem, "Prefill dominates long-context wall-clock; decode-only throughput is not sufficient for release claims.")


def plot_eval_latency(rows: list[dict[str, Any]], stem: Path) -> None:
    release = [row for row in rows if row["suite"] == "release"]
    variants = [row["variant"] for row in release]
    labels = [clean_label(v) for v in variants]
    mean = [float(row["latency_seconds_mean"]) for row in release]
    p95 = [float(row["latency_seconds_p95"]) for row in release]
    x = np.arange(len(variants))

    fig, ax = plt.subplots(figsize=(5.0, 3.35))
    bars = ax.bar(
        x,
        mean,
        color=[variant_color(v) for v in variants],
        edgecolor=PALETTE["paper"],
        linewidth=1.2,
        width=0.55,
        label="mean",
    )
    ax.plot(x, p95, color=PALETTE["ink"], marker="o", linewidth=2.0, label="p95", zorder=3)
    ax.bar_label(bars, labels=[f"{value:.1f}s" for value in mean], padding=3, fontsize=7.4, color=PALETTE["ink"])
    for xi, value in zip(x, p95):
        ax.annotate(f"{value:.1f}s", (xi, value), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=7.4, color=PALETTE["ink"])

    ax.set_xticks(x, labels)
    ax.set_ylabel("seconds / sample")
    add_title(ax, "Release-gate latency", "Mean bars with p95 overlay from deterministic local checks.")
    ax.legend(loc="upper right", ncol=2, columnspacing=0.9, handlelength=1.4)
    polish_axes(ax)
    save_figure(fig, stem, "Latency is a release guardrail, not a substitute for benchmark quality evaluation.")


def plot_pass_rate(rows: list[dict[str, Any]], stem: Path) -> None:
    release = [row for row in rows if row["suite"] == "release"]
    variants = sorted({row["variant"] for row in release})
    categories = sorted({row["category"] for row in release})
    matrix = np.zeros((len(variants), len(categories)), dtype=float)
    lookup = {(row["variant"], row["category"]): float(row["pass_rate"]) for row in release}
    for i, variant in enumerate(variants):
        for j, category in enumerate(categories):
            matrix[i, j] = lookup.get((variant, category), np.nan)

    fig, ax = plt.subplots(figsize=(7.6, 2.95))
    image = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap=SCORE_CMAP, aspect="auto")
    ax.set_xticks(np.arange(len(categories)), [c.replace("_", " ") for c in categories], rotation=28, ha="right")
    ax.set_yticks(np.arange(len(variants)), [clean_label(v) for v in variants])
    add_cell_grid(ax, len(variants), len(categories))
    for i in range(len(variants)):
        for j in range(len(categories)):
            value = matrix[i, j]
            ax.text(j, i, "n/a" if np.isnan(value) else f"{value:.0%}", ha="center", va="center", fontsize=7.9, color=PALETTE["ink"])

    add_title(ax, "Release-gate coverage", "Per-category pass rate; darker cells mean stronger release-suite coverage.")
    cbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    cbar.ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
    cbar.outline.set_edgecolor(PALETTE["spine"])
    save_figure(fig, stem, "Deterministic release suites catch runtime regressions; they are intentionally smaller than academic benchmarks.")


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

    fig, ax = plt.subplots(figsize=(7.6, 3.35))
    image = ax.imshow(normalized, vmin=0.0, vmax=1.0, cmap=SCORE_CMAP, aspect="auto")
    ax.set_xticks(np.arange(len(variants)), labels)
    ax.set_yticks(np.arange(len(metrics)), [metric[0] for metric in metrics])
    add_cell_grid(ax, len(metrics), len(variants))
    for row_idx, (_, values, unit, _) in enumerate(metrics):
        for col_idx, value in enumerate(values):
            ax.text(col_idx, row_idx, f"{value:.2f} {unit}", ha="center", va="center", fontsize=7.3, color=PALETTE["ink"])
    add_title(ax, "Release decision scorecard", "Each row is normalized independently; darker cells mean stronger local release fit.")
    cbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("relative fit for local release", fontsize=7.4)
    cbar.outline.set_edgecolor(PALETTE["spine"])
    save_figure(fig, stem, "This matrix compares release ergonomics, not model intelligence. Quality is reported by paired benchmark subsets.")


def plot_tensor_allocation(rows: list[dict[str, Any]], stem: Path) -> None:
    variants = sorted({row["variant"] for row in rows})
    tensor_types = ["F16", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "Q8_0"]
    lookup = {(row["variant"], row["tensor_type"]): int(row["count"]) for row in rows}
    totals = np.array([sum(lookup.get((variant, t), 0) for t in tensor_types) for variant in variants], dtype=float)
    fig, ax = plt.subplots(figsize=(7.6, 3.45))
    bottom = np.zeros(len(variants))
    labels = [clean_label(v) for v in variants]
    for tensor_type in tensor_types:
        counts = np.array([lookup.get((variant, tensor_type), 0) for variant in variants], dtype=float)
        values = np.divide(counts, totals, out=np.zeros_like(counts), where=totals > 0)
        bars = ax.bar(
            labels,
            values,
            bottom=bottom,
            label=tensor_type,
            color=TYPE_COLORS[tensor_type],
            edgecolor=PALETTE["paper"],
            linewidth=1.15,
            width=0.76,
        )
        for bar, value, base in zip(bars, values, bottom):
            if value < 0.055:
                continue
            text_color = "white" if tensor_type in {"Q8_0", "Q6_K", "Q3_K"} else PALETTE["ink"]
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                base + value / 2,
                f"{value:.0%}",
                ha="center",
                va="center",
                fontsize=8.0,
                color=text_color,
            )
        bottom += values
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
    ax.set_ylabel("share of mapped tensors")
    add_title(
        ax,
        "Dynamic GGUF tensor-type allocation",
        "Count-based tensor map produced by the OpenTQ policy; skipped vision tensors are excluded.",
    )
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), ncol=1, columnspacing=0.8, handletextpad=0.5)
    ax.set_ylim(0, 1.0)
    polish_axes(ax)
    save_figure(fig, stem, "The allocation intentionally keeps normalization/state tensors high precision while lowering dense projection tensors.")


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
    fig, axes = plt.subplots(1, len(variants), figsize=(7.8, 3.75), sharey=True)
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
        ax.set_xlim(-0.5, len(tensor_types) - 0.5)
        ax.set_ylim(len(categories) - 0.5, -0.5)
        ax.set_facecolor(PALETTE["paper"])
        for i in range(len(categories)):
            for j, tensor_type in enumerate(tensor_types):
                value = matrix[i, j]
                base = Rectangle(
                    (j - 0.5, i - 0.5),
                    1.0,
                    1.0,
                    facecolor="#f2eee6",
                    edgecolor="#e1dbd0",
                    linewidth=0.8,
                )
                ax.add_patch(base)
                if np.isnan(value) or value <= 0:
                    continue
                color = TYPE_COLORS[tensor_type]
                ax.add_patch(
                    Rectangle(
                        (j - 0.5, i - 0.5),
                        1.0,
                        1.0,
                        facecolor=color,
                        edgecolor="#e1dbd0",
                        linewidth=0.8,
                        alpha=0.18 + 0.74 * value,
                    )
                )
                text_color = "white" if value >= 0.82 and tensor_type in {"Q3_K", "Q6_K", "Q8_0"} else PALETTE["ink"]
                ax.text(j, i, f"{value:.0%}", ha="center", va="center", fontsize=6.8, color=text_color)
        ax.set_title(clean_label(variant), fontweight="semibold", fontsize=10.2, pad=9)
        ax.set_xticks(np.arange(len(tensor_types)), tensor_types, rotation=35, ha="right")
        ax.set_yticks(np.arange(len(categories)), [category.replace("_", " ") for category in categories])
        ax.tick_params(axis="both", length=0, colors=PALETTE["ink"])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(PALETTE["spine"])
            spine.set_linewidth(0.85)
    axes[0].set_ylabel("tensor family")
    fig.suptitle(
        "Where OpenTQ spends precision",
        x=0.02,
        y=1.02,
        ha="left",
        fontweight="semibold",
        fontsize=12.0,
        color=PALETTE["ink"],
    )
    fig.text(
        0.02,
        0.935,
        "Rows show tensor families; columns show GGUF tensor types. Cell opacity is the share within each tensor family.",
        ha="left",
        va="top",
        fontsize=7.8,
        color=PALETTE["muted"],
    )
    save_figure(fig, stem, "Family-level allocation makes the policy auditable: state and norms stay F16, projections absorb most quantization.")


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
    colors = [
        (PALETTE["green"] if row["category"] == "Coding Agent" else PALETTE["blue"] if row["category"] == "Knowledge" else PALETTE["copper"])
        for row in plot_rows
    ]
    y = np.arange(len(plot_rows))

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    bars = ax.barh(y, scores, color=colors, edgecolor=PALETTE["paper"], linewidth=1.0)
    ax.bar_label(bars, labels=[f"{score:.1f}" for score in scores], padding=3, fontsize=8.3, color=PALETTE["ink"])
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("official Qwen score")
    add_title(ax, "Official Qwen3.6-27B BF16 reference", "External vendor scores; not mixed with OTQ unless task definitions match.")
    polish_axes(ax)
    save_figure(fig, stem, "Use this only as a reference table until a paired harness runs the same task and scoring rule.")


def plot_quant_eval(rows: list[dict[str, Any]], stem: Path) -> None:
    if not rows:
        return
    labels = [row["model"].replace("OTQ-DYN-", "").replace("_", "-") for row in rows]
    pass_rate = [float(row["pass_rate"]) for row in rows]
    p95 = [float(row["latency_seconds_p95"]) for row in rows]
    x = np.arange(len(rows))

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.15))
    bars = axes[0].bar(x, pass_rate, color=[variant_color(row["model"]) for row in rows], edgecolor=PALETTE["paper"], linewidth=1.1)
    axes[0].bar_label(bars, labels=[f"{value:.0%}" for value in pass_rate], padding=2, fontsize=7.4)
    axes[0].set_xticks(x, labels, rotation=18, ha="right")
    axes[0].set_ylim(0, 1.05)
    axes[0].yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
    add_title(axes[0], "OTQ eval pass rate")
    polish_axes(axes[0])

    bars = axes[1].bar(x, p95, color=[variant_color(row["model"]) for row in rows], edgecolor=PALETTE["paper"], linewidth=1.1)
    axes[1].bar_label(bars, labels=[f"{value:.1f}s" for value in p95], padding=2, fontsize=7.4)
    axes[1].set_xticks(x, labels, rotation=18, ha="right")
    axes[1].set_ylabel("p95 seconds")
    add_title(axes[1], "OTQ eval latency")
    polish_axes(axes[1])
    save_figure(fig, stem, "OTQ task runs are only comparable when prompts, split, runtime, and scoring rule are pinned.")


def paired_summary_rows(repo: Path) -> list[dict[str, Any]]:
    path = repo / "benchmarks" / "paired_bf16_quant_summary.json"
    if not path.exists():
        return []
    payload = load_json(path)
    return list(payload.get("rows", []))


def plot_paired_bf16_quant(rows: list[dict[str, Any]], stem: Path) -> None:
    if not rows:
        return
    total = next((row for row in rows if row.get("benchmark") == "TOTAL"), None)
    benchmarks = [row for row in rows if row.get("benchmark") != "TOTAL"]
    if not total or not benchmarks:
        return

    variants = [
        ("BF16", "bf16_rate", PALETTE["cream"]),
        ("Q3_K_M", "q3_rate", TYPE_COLORS["Q3_K"]),
        ("Q4_K_M", "q4_rate", TYPE_COLORS["Q4_K"]),
        ("Q5_K_M", "q5_rate", TYPE_COLORS["Q5_K"]),
    ]
    deltas = np.array(
        [
            [
                float(row.get("q3_delta_vs_bf16", 0.0)) * 100,
                float(row.get("q4_delta_vs_bf16", 0.0)) * 100,
                float(row.get("q5_delta_vs_bf16", 0.0)) * 100,
            ]
            for row in benchmarks
        ]
    )
    display_names = {
        "mmlu": "MMLU",
        "mmlu_pro": "MMLU-Pro",
        "arc": "ARC",
        "hellaswag": "HellaSwag",
        "gsm8k": "GSM8K",
        "math": "MATH",
        "bbh": "BBH",
        "gpqa": "GPQA",
        "truthfulqa": "TruthfulQA",
        "winogrande": "WinoGrande",
        "drop": "DROP",
        "piqa": "PIQA",
        "commonsenseqa": "CommonSenseQA",
    }
    names = [display_names.get(str(row["benchmark"]), str(row["benchmark"]).replace("_", "-")) for row in benchmarks]
    total_rates = [float(total.get(rate_key, 0.0)) * 100 for _label, rate_key, _color in variants]

    fig = plt.figure(figsize=(8.15, 4.3))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.9, 1.85, 0.055], wspace=0.48)
    ax_bar = fig.add_subplot(gs[0, 0])
    ax_heat = fig.add_subplot(gs[0, 1])
    cax = fig.add_subplot(gs[0, 2])
    fig.subplots_adjust(left=0.085, right=0.965, top=0.82, bottom=0.18, wspace=0.48)
    fig.text(
        0.085,
        0.94,
        "Paired BF16-vs-GGUF quality signal",
        ha="left",
        va="top",
        fontsize=10.7,
        fontweight="semibold",
        color=PALETTE["ink"],
    )
    fig.text(
        0.085,
        0.895,
        "232 identical pinned prompts, deterministic decoding, local scoring rules. Deltas are percentage points vs BF16.",
        ha="left",
        va="top",
        fontsize=7.0,
        color=PALETTE["muted"],
    )

    y = np.arange(len(variants))
    bars = ax_bar.barh(
        y,
        total_rates,
        color=[color for _label, _key, color in variants],
        edgecolor=PALETTE["paper"],
        linewidth=1.0,
        height=0.58,
    )
    for bar, value, (label, _key, color) in zip(bars, total_rates, variants):
        text_color = PALETTE["paper"] if label in {"Q3_K_M", "Q5_K_M"} else PALETTE["ink"]
        ax_bar.text(
            value - 1.4,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1f}%",
            ha="right",
            va="center",
            fontsize=7.0,
            fontweight="semibold",
            color=text_color,
        )
        if label == "BF16":
            bar.set_edgecolor(PALETTE["spine"])
        elif color == TYPE_COLORS["Q4_K"]:
            bar.set_edgecolor("#5d8aad")
    ax_bar.set_yticks(y, [label for label, _key, _color in variants])
    ax_bar.invert_yaxis()
    ax_bar.set_xlim(0, 75.0)
    ax_bar.set_xlabel("aggregate pass rate", fontsize=7.2)
    ax_bar.set_title("Aggregate", loc="left", fontweight="semibold", fontsize=8.8, pad=7)
    ax_bar.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=100))
    ax_bar.tick_params(axis="both", labelsize=6.9)
    ax_bar.grid(True, axis="x", which="major", linestyle="--", linewidth=0.62, alpha=0.55)
    ax_bar.grid(False, axis="y")
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)
    ax_bar.spines["left"].set_color(PALETTE["spine"])
    ax_bar.spines["bottom"].set_color(PALETTE["spine"])

    max_abs = max(12.5, float(np.nanmax(np.abs(deltas))) if deltas.size else 12.5)
    norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)
    image = ax_heat.imshow(deltas, aspect="auto", cmap=DELTA_CMAP, norm=norm)
    ax_heat.set_xticks([0, 1, 2], ["Q3_K_M", "Q4_K_M", "Q5_K_M"])
    ax_heat.set_yticks(np.arange(len(names)), names)
    ax_heat.tick_params(axis="x", labelrotation=0)
    ax_heat.tick_params(axis="both", labelsize=6.7, pad=2.2)
    for row_idx in range(deltas.shape[0]):
        for col_idx in range(deltas.shape[1]):
            value = deltas[row_idx, col_idx]
            color = PALETTE["paper"] if abs(value) >= max_abs * 0.62 else PALETTE["ink"]
            ax_heat.text(
                col_idx,
                row_idx,
                f"{value:+.1f}",
                ha="center",
                va="center",
                fontsize=6.45,
                fontweight="semibold",
                color=color,
            )
    ax_heat.grid(False)
    ax_heat.set_xticks(np.arange(-0.5, 3, 1), minor=True)
    ax_heat.set_yticks(np.arange(-0.5, len(names), 1), minor=True)
    ax_heat.grid(which="minor", color=PALETTE["grid"], linewidth=0.62)
    ax_heat.tick_params(which="minor", bottom=False, left=False)
    ax_heat.set_xlabel("delta vs BF16 (pp)", fontsize=7.2)
    ax_heat.set_title("Per-benchmark delta", loc="left", fontweight="semibold", fontsize=8.8, pad=7)
    for spine in ax_heat.spines.values():
        spine.set_color(PALETTE["spine"])
        spine.set_linewidth(0.72)
    cbar = fig.colorbar(image, cax=cax)
    cbar.set_label("pp", rotation=90, labelpad=4, fontsize=6.8)
    cbar.ax.tick_params(labelsize=6.3, colors=PALETTE["muted"], length=2.8)
    cbar.outline.set_edgecolor(PALETTE["spine"])
    save_figure(fig, stem, "Paired BF16-vs-GGUF subset: aggregate rates and per-benchmark percentage-point deltas.", tight=False)


def figure_links(name: str) -> str:
    return f"![{name}](assets/{name}.png)\n\nVector: [`SVG`](assets/{name}.svg) | [`PDF`](assets/{name}.pdf)"


def write_report_md(
    repo: Path,
    official_rows: list[dict[str, Any]],
    quant_evals: list[dict[str, Any]],
) -> None:
    paired_csv = repo / "benchmarks" / "paired_bf16_quant_summary.csv"
    paired_json = repo / "benchmarks" / "paired_bf16_quant_summary.json"
    has_paired = paired_csv.exists() and paired_json.exists()
    official_note = (
        "Official Qwen3.6-27B language benchmark scores are imported as an external reference table in `benchmarks/official_qwen36_baseline.csv`. They are not plotted against OTQ until matching benchmark tasks are run on these GGUF files."
        if official_rows
        else "No official baseline JSON was available when this report was generated."
    )
    quant_note = (
        "OTQ-only comparison JSONs are available in `benchmarks/quant_eval.csv`."
        if quant_evals
        else "No separate OTQ-only benchmark subset is attached beyond the paired BF16-vs-GGUF mini-subset."
        if has_paired
        else "No additional OTQ benchmark subset is attached yet. Add OTQ-only task runs when comparing against official benchmark families."
    )
    paired_note = (
        [
            "",
            "## Paired BF16-vs-GGUF Mini-Subset",
            "",
            "The staged repo includes a same-task, same-prompt, deterministic 232-sample practical subset comparing the BF16 sidecar against `Q3_K_M`, `Q4_K_M`, and `Q5_K_M`.",
            "",
            figure_links("paired-bf16-quant-delta"),
            "",
            "Files:",
            "",
            "- `benchmarks/paired_bf16_quant_summary.csv`",
            "- `benchmarks/paired_bf16_quant_summary.json`",
            "- `benchmarks/paired_bf16_quant_report.md`",
            "",
            "This is a quantization-regression signal, not a full official benchmark replacement. Do not compare its small-subset `mmlu_pro` or `gpqa` rates directly to the Qwen model-card full-harness scores.",
        ]
        if has_paired
        else []
    )
    csv_rows = [
        "- `benchmarks/throughput.csv`",
        "- `benchmarks/eval_summary.csv`",
        "- `benchmarks/category_pass_rate.csv`",
        "- `benchmarks/artifacts.csv`",
        "- `benchmarks/tensor_allocation.csv`",
        "- `benchmarks/category_tensor_allocation.csv`",
        "- `benchmarks/official_qwen36_baseline.csv`",
    ]
    if has_paired:
        csv_rows.extend(
            [
                "- `benchmarks/paired_bf16_quant_summary.csv`",
                "- `benchmarks/paired_bf16_quant_summary.json`",
                "- `benchmarks/paired_bf16_quant_report.md`",
            ]
        )
    csv_rows.append("- `benchmarks/quant_eval.csv` when separate OTQ-only task runs are present")
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
        *paired_note,
        "",
        "## CSV Data",
        "",
        *csv_rows,
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
    paired_rows = paired_summary_rows(repo)

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
    plot_paired_bf16_quant(paired_rows, assets / "paired-bf16-quant-delta")
    write_report_md(repo, official_rows, quant_evals)
    print(f"wrote publication-grade benchmark report assets under {repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
