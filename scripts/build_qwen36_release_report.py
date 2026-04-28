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
    "q3": "#648fff",
    "q4": "#fe6100",
    "ref": "#8a8a8a",
    "good": "#6acc64",
    "warn": "#f0c75e",
    "bad": "#d65f5f",
    "ink": "#242424",
    "grid": "#d8d8d8",
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
            "font.size": 8.8,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.3,
            "xtick.labelsize": 8.3,
            "ytick.labelsize": 8.3,
            "legend.fontsize": 8.2,
            "figure.dpi": 150,
            "savefig.dpi": 320,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.06,
            "lines.linewidth": 2.0,
            "lines.markersize": 6,
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.alpha": 0.25,
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


def plot_throughput(rows: list[dict[str, Any]], stem: Path) -> None:
    variants = sorted({row["variant"] for row in rows})
    tests = ["pp8192", "tg128"]
    values = {(row["variant"], row["test"]): row["tokens_per_second"] for row in rows}
    x = np.arange(len(variants))
    width = 0.34

    fig, ax = plt.subplots(figsize=(7.0, 3.25))
    test_colors = {"pp8192": "#648fff", "tg128": "#ffb000"}
    labels = {"pp8192": "Prefill 8K", "tg128": "Decode 128"}
    for offset, test in zip((-width / 2, width / 2), tests):
        ys = [values.get((variant, test), 0.0) for variant in variants]
        bars = ax.bar(x + offset, ys, width=width, label=labels[test], color=test_colors[test], edgecolor="#2d2d2d", linewidth=0.7)
        ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=8.5)

    ax.set_xticks(x, [clean_label(v) for v in variants])
    ax.set_ylabel("tokens / second")
    ax.set_title("M1 Max runtime throughput", loc="left", fontweight="semibold")
    ax.legend(loc="upper right", ncol=2, columnspacing=1.0, handletextpad=0.4)
    polish_axes(ax)
    save_figure(fig, stem)


def plot_eval_latency(rows: list[dict[str, Any]], stem: Path) -> None:
    release = [row for row in rows if row["suite"] == "release"]
    variants = [row["variant"] for row in release]
    labels = [clean_label(v) for v in variants]
    mean = [float(row["latency_seconds_mean"]) for row in release]
    p95 = [float(row["latency_seconds_p95"]) for row in release]
    x = np.arange(len(variants))

    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    bars = ax.bar(x, mean, color=[variant_color(v) for v in variants], edgecolor="#2d2d2d", linewidth=0.7, label="mean")
    ax.plot(x, p95, color="#dc267f", marker="o", linewidth=2.0, label="p95", zorder=3)
    ax.bar_label(bars, labels=[f"{value:.1f}s" for value in mean], padding=2, fontsize=8.5)
    for xi, value in zip(x, p95):
        ax.annotate(f"{value:.1f}s", (xi, value), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8.5, color="#7a1244")

    ax.set_xticks(x, labels)
    ax.set_ylabel("seconds / sample")
    ax.set_title("Release-suite latency", loc="left", fontweight="semibold")
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
    image = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap="Greens", aspect="auto")
    ax.set_xticks(np.arange(len(categories)), [c.replace("_", " ") for c in categories], rotation=28, ha="right")
    ax.set_yticks(np.arange(len(variants)), [clean_label(v) for v in variants])
    for i in range(len(variants)):
        for j in range(len(categories)):
            value = matrix[i, j]
            ax.text(j, i, "n/a" if np.isnan(value) else f"{value:.0%}", ha="center", va="center", fontsize=9, color="#152015")

    ax.set_title("Release-suite pass rate by category", loc="left", fontweight="semibold")
    cbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    cbar.ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
    save_figure(fig, stem)


def plot_artifacts(rows: list[dict[str, Any]], stem: Path) -> None:
    variants = [row["variant"] for row in rows]
    sizes = [float(row["gib"]) for row in rows]
    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    y = np.arange(len(variants))
    bars = ax.barh(y, sizes, color=[variant_color(v) for v in variants], edgecolor="#2d2d2d", linewidth=0.7)
    ax.bar_label(bars, labels=[f"{value:.2f} GiB" for value in sizes], padding=3, fontsize=8.5)
    ax.axvline(32, color=PALETTE["ref"], linewidth=1.2, linestyle="--")
    ax.text(32, len(variants) - 0.35, "32 GB", color=PALETTE["ref"], fontsize=8.5, ha="right")
    ax.set_yticks(y, [clean_label(v) for v in variants])
    ax.set_xlabel("artifact size (GiB)")
    ax.set_title("Final GGUF size", loc="left", fontweight="semibold")
    ax.set_xlim(0, max(34, max(sizes) * 1.2))
    polish_axes(ax)
    save_figure(fig, stem)


def plot_tensor_allocation(rows: list[dict[str, Any]], stem: Path) -> None:
    variants = sorted({row["variant"] for row in rows})
    tensor_types = sorted({row["tensor_type"] for row in rows})
    lookup = {(row["variant"], row["tensor_type"]): int(row["count"]) for row in rows}
    totals = np.array([sum(lookup.get((variant, t), 0) for t in tensor_types) for variant in variants], dtype=float)
    fig, ax = plt.subplots(figsize=(7.0, 3.15))
    bottom = np.zeros(len(variants))
    palette = ["#648fff", "#fe6100", "#ffb000", "#785ef0", "#dc267f", "#6acc64", "#8a8a8a"]
    for idx, tensor_type in enumerate(tensor_types):
        counts = np.array([lookup.get((variant, tensor_type), 0) for variant in variants], dtype=float)
        values = np.divide(counts, totals, out=np.zeros_like(counts), where=totals > 0)
        ax.bar([clean_label(v) for v in variants], values, bottom=bottom, label=tensor_type, color=palette[idx % len(palette)], edgecolor="white", linewidth=0.6)
        bottom += values
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
    ax.set_ylabel("share of tensors")
    ax.set_title("GGUF tensor-type allocation", loc="left", fontweight="semibold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=min(5, len(tensor_types)), columnspacing=0.8, handletextpad=0.3)
    polish_axes(ax)
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
    colors = [("#648fff" if row["category"] == "Coding Agent" else "#fe6100" if row["category"] == "Knowledge" else "#785ef0") for row in plot_rows]
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
        "Official Qwen3.6-27B language benchmark scores are imported as an external BF16 reference. We do not rerun BF16 locally for release plotting."
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
        "Benchmarks are split into runtime throughput, release-gate checks, artifact metrics, and the official Qwen reference baseline.",
        "",
        "## Runtime",
        "",
        figure_links("benchmark-throughput"),
        "",
        figure_links("artifact-size"),
        "",
        figure_links("tensor-allocation"),
        "",
        "## Release Gates",
        "",
        figure_links("eval-latency"),
        "",
        figure_links("eval-pass-rate"),
        "",
        "These release suites are deterministic guardrails, not substitutes for full academic benchmarks.",
        "",
        "## Official Baseline",
        "",
        figure_links("official-language-baseline") if official_rows else official_note,
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
    write_csv(
        data_dir / "official_qwen36_baseline.csv",
        official_rows,
        ["model", "source", "category", "benchmark", "score", "unit", "higher_is_better", "notes"],
    )
    if quant_evals:
        write_csv(data_dir / "quant_eval.csv", quant_evals, ["model", "total", "passed", "pass_rate", "latency_seconds_mean", "latency_seconds_p95"])

    plot_throughput(bench, assets / "benchmark-throughput")
    plot_eval_latency(evals, assets / "eval-latency")
    plot_pass_rate(categories, assets / "eval-pass-rate")
    plot_artifacts(artifacts, assets / "artifact-size")
    plot_tensor_allocation(tensors, assets / "tensor-allocation")
    plot_official_baseline(official_rows, assets / "official-language-baseline")
    plot_quant_eval(quant_evals, assets / "quant-eval-summary")
    write_report_md(repo, official_rows, quant_evals)
    print(f"wrote publication-grade benchmark report assets under {repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
