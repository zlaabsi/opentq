#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


SCHEMA = "opentq.qwen36_paired_quality_report.v1"


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def official_by_benchmark(matrix_path: Path) -> dict[str, dict[str, Any]]:
    if not matrix_path.exists():
        return {}
    matrix = load_json(matrix_path)
    return {
        str(row["id"]): dict(row.get("official_baseline") or {})
        for row in matrix.get("benchmarks", [])
        if row.get("official_baseline")
    }


def load_model_payloads(subset_root: Path) -> dict[str, dict[str, Any]]:
    payloads = {}
    for path in sorted(subset_root.glob("*.json")):
        payload = load_json(path)
        if payload.get("schema") != "opentq.qwen36_benchmark_subset_eval.v1":
            continue
        key = str((payload.get("model") or {}).get("key") or path.stem)
        payloads[key] = payload
    return payloads


def benchmark_map(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(item["benchmark_id"]): item for item in payload.get("benchmarks", [])}


def result_map(benchmark: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(item["task_id"]): item for item in benchmark.get("results", [])}


def pass_rate(passed: int, total: int) -> float | None:
    if total <= 0:
        return None
    return passed / total


def wilson_interval(passed: int, total: int, z: float = 1.96) -> dict[str, float | None]:
    if total <= 0:
        return {"low": None, "high": None}
    p = passed / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denominator
    return {"low": round(max(0.0, center - margin), 4), "high": round(min(1.0, center + margin), 4)}


def compare_benchmark(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    baseline_results = result_map(baseline)
    candidate_results = result_map(candidate)
    common_ids = sorted(set(baseline_results) & set(candidate_results))
    both_correct = 0
    baseline_only = 0
    candidate_only = 0
    both_wrong = 0
    rows = []
    for task_id in common_ids:
        baseline_passed = bool(baseline_results[task_id].get("passed"))
        candidate_passed = bool(candidate_results[task_id].get("passed"))
        if baseline_passed and candidate_passed:
            both_correct += 1
        elif baseline_passed and not candidate_passed:
            baseline_only += 1
        elif candidate_passed and not baseline_passed:
            candidate_only += 1
        else:
            both_wrong += 1
        rows.append(
            {
                "task_id": task_id,
                "baseline_passed": baseline_passed,
                "candidate_passed": candidate_passed,
                "baseline_score": baseline_results[task_id].get("score"),
                "candidate_score": candidate_results[task_id].get("score"),
            }
        )
    total = len(common_ids)
    baseline_passed_count = both_correct + baseline_only
    candidate_passed_count = both_correct + candidate_only
    baseline_rate = pass_rate(baseline_passed_count, total)
    candidate_rate = pass_rate(candidate_passed_count, total)
    delta_pp = None if baseline_rate is None or candidate_rate is None else round((candidate_rate - baseline_rate) * 100, 2)
    retention = None
    if baseline_rate and candidate_rate is not None:
        retention = round(candidate_rate / baseline_rate, 4)
    return {
        "total": total,
        "baseline_passed": baseline_passed_count,
        "candidate_passed": candidate_passed_count,
        "baseline_pass_rate": None if baseline_rate is None else round(baseline_rate, 4),
        "candidate_pass_rate": None if candidate_rate is None else round(candidate_rate, 4),
        "delta_pp": delta_pp,
        "retention": retention,
        "both_correct": both_correct,
        "baseline_only": baseline_only,
        "candidate_only": candidate_only,
        "both_wrong": both_wrong,
        "candidate_wilson_95": wilson_interval(candidate_passed_count, total),
        "publishability": "candidate" if total >= 30 else "diagnostic_only_sample_lt_30",
        "paired_results": rows,
    }


def build_report(
    subset_root: Path,
    baseline_key: str,
    candidate_keys: list[str],
    matrix_path: Path,
) -> dict[str, Any]:
    payloads = load_model_payloads(subset_root)
    if baseline_key not in payloads:
        raise ValueError(f"missing baseline model JSON for {baseline_key!r} in {subset_root}")
    baseline_benchmarks = benchmark_map(payloads[baseline_key])
    official = official_by_benchmark(matrix_path)
    rows = []
    for benchmark_id, baseline_benchmark in sorted(baseline_benchmarks.items()):
        comparisons = {}
        for candidate_key in candidate_keys:
            candidate_payload = payloads.get(candidate_key)
            if not candidate_payload:
                comparisons[candidate_key] = {"status": "missing_candidate"}
                continue
            candidate_benchmark = benchmark_map(candidate_payload).get(benchmark_id)
            if not candidate_benchmark:
                comparisons[candidate_key] = {"status": "missing_benchmark"}
                continue
            comparison = compare_benchmark(baseline_benchmark, candidate_benchmark)
            comparison["status"] = "paired"
            comparisons[candidate_key] = comparison
        baseline_summary = baseline_benchmark.get("summary", {})
        official_score = official.get(benchmark_id, {}).get("score")
        local_baseline_pct = float(baseline_summary.get("pass_rate", 0.0)) * 100
        rows.append(
            {
                "benchmark_id": benchmark_id,
                "baseline_summary": baseline_summary,
                "official_baseline": official.get(benchmark_id, {}),
                "official_anchor_delta_pp": None if official_score is None else round(local_baseline_pct - float(official_score), 2),
                "adapter": baseline_benchmark.get("adapter", {}),
                "comparisons": comparisons,
            }
        )
    return {
        "schema": SCHEMA,
        "created_at": now_iso(),
        "subset_root": str(subset_root),
        "baseline_key": baseline_key,
        "candidate_keys": candidate_keys,
        "matrix": str(matrix_path),
        "methodology": {
            "direct_delta": "candidate pass rate minus BF16 GGUF pass rate on identical task_ids, prompt format, scoring rule, and runner",
            "official_scores": "reported as anchors only; not used as quantization deltas",
            "small_sample_rule": "n < 30 is diagnostic only",
        },
        "rows": rows,
    }


def pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.1f}%"


def comparison_cell(comparison: dict[str, Any]) -> str:
    if comparison.get("status") != "paired":
        return str(comparison.get("status", "missing"))
    return (
        f"{comparison['candidate_passed']}/{comparison['total']} ({pct(comparison['candidate_pass_rate'])}), "
        f"delta {comparison['delta_pp']:+.1f} pp, "
        f"BF16-only {comparison['baseline_only']}, quant-only {comparison['candidate_only']}"
    )


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    candidate_keys = list(payload["candidate_keys"])
    header = ["Benchmark", "BF16 GGUF Local", "Official Anchor", *candidate_keys, "Status"]
    lines = [
        "# Qwen3.6 BF16-vs-OTQ Paired Quality Report",
        "",
        "This is the quantization-quality report to use for degradation claims.",
        "Official Qwen scores are anchors only; direct deltas are computed only against the local BF16 GGUF on identical task ids.",
        "",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for row in payload["rows"]:
        baseline = row["baseline_summary"]
        official = row.get("official_baseline") or {}
        official_text = "none"
        if official:
            official_text = f"{official.get('benchmark')}: {official.get('score')}"
            if row.get("official_anchor_delta_pp") is not None:
                official_text += f" (local BF16 {row['official_anchor_delta_pp']:+.1f} pp)"
        cells = [
            f"`{row['benchmark_id']}`",
            f"{baseline.get('passed', 0)}/{baseline.get('total', 0)} ({pct(baseline.get('pass_rate'))})",
            official_text,
        ]
        cells.extend(comparison_cell(row["comparisons"].get(candidate_key, {})) for candidate_key in candidate_keys)
        statuses = sorted(
            {
                str(comparison.get("publishability", comparison.get("status", "missing")))
                for comparison in row["comparisons"].values()
            }
        )
        cells.append(", ".join(statuses))
        lines.append("| " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a paired BF16-vs-OTQ quality report.")
    parser.add_argument("--subset-root", type=Path, required=True)
    parser.add_argument("--baseline-key", default="bf16_gguf")
    parser.add_argument("--candidate-key", action="append")
    parser.add_argument("--matrix", type=Path, default=Path("benchmarks/qwen36_long_running_benchmark_matrix.json"))
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = build_report(
        subset_root=args.subset_root,
        baseline_key=args.baseline_key,
        candidate_keys=args.candidate_key or ["q3", "q4", "q5"],
        matrix_path=args.matrix,
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "paired-quality-report.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(args.output_root / "paired-quality-report.md", payload)
    print(args.output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
