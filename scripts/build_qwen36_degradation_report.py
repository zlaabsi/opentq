#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


SCHEMA = "opentq.qwen36_degradation_report.v1"


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def official_by_benchmark(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    payload = load_json(path)
    return {str(row["benchmark"]): row for row in payload.get("scores", [])}


def load_subset_summaries(root: Path) -> dict[str, dict[str, Any]]:
    summaries: dict[str, dict[str, Any]] = {}
    if not root.exists():
        return summaries
    for path in sorted(root.glob("*.json")):
        payload = load_json(path)
        if payload.get("schema") != "opentq.qwen36_benchmark_subset_eval.v1":
            continue
        model_key = str((payload.get("model") or {}).get("key") or path.stem)
        for benchmark in payload.get("benchmarks", []):
            summaries.setdefault(str(benchmark["benchmark_id"]), {})[model_key] = benchmark.get("summary", {})
    return summaries


def row_claim_status(matrix_row: dict[str, Any], subset_summaries: dict[str, dict[str, Any]]) -> str:
    benchmark_id = str(matrix_row["id"])
    mode = str(matrix_row["comparison_mode"])
    if mode == "blocked_modality":
        return "blocked_modality"
    if mode == "judge_based":
        return "judge_required"
    if benchmark_id not in subset_summaries:
        if mode == "official_comparable":
            return "pending_otq_subset"
        return "pending_bf16_or_otq_only"
    if mode == "official_comparable":
        return "official_delta_candidate_review_required"
    return "otq_only_ready_bf16_delta_pending"


def build_rows(matrix: dict[str, Any], official_rows: dict[str, dict[str, Any]], subset_summaries: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for matrix_row in matrix.get("benchmarks", []):
        official = matrix_row.get("official_baseline") or {}
        official_name = official.get("benchmark")
        rows.append(
            {
                "benchmark_id": matrix_row["id"],
                "name": matrix_row["name"],
                "comparison_mode": matrix_row["comparison_mode"],
                "official_baseline": official,
                "official_source_score": official_rows.get(str(official_name), {}) if official_name else {},
                "subset_summaries": subset_summaries.get(str(matrix_row["id"]), {}),
                "claim_status": row_claim_status(matrix_row, subset_summaries),
                "claim_rule": matrix_row["claim_rule"],
            }
        )
    return rows


def format_model_summary(summary: dict[str, Any], official_score: float | None) -> str:
    total = int(summary.get("total", 0) or 0)
    passed = int(summary.get("passed", 0) or 0)
    pass_rate = summary.get("pass_rate")
    if not total or pass_rate is None:
        return "pending"
    pct = float(pass_rate) * 100
    text = f"{passed}/{total} ({pct:.1f}%)"
    if official_score is not None:
        text += f", candidate {pct - official_score:+.1f} pp"
    return text


def format_subset_summaries(row: dict[str, Any]) -> str:
    summaries = row.get("subset_summaries") or {}
    if not summaries:
        return "pending"
    official = row.get("official_source_score") or {}
    official_score = float(official["score"]) if "score" in official else None
    parts = []
    for model_key in sorted(summaries):
        parts.append(f"`{model_key}` {format_model_summary(summaries[model_key], official_score)}")
    return "; ".join(parts)


def format_official_reference(row: dict[str, Any]) -> str:
    official = row.get("official_source_score") or {}
    if "score" not in official:
        return "none"
    benchmark = official.get("benchmark") or (row.get("official_baseline") or {}).get("benchmark") or row["benchmark_id"]
    return f"{benchmark}: {official['score']} {official.get('unit', 'score')}"


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Qwen3.6 OTQ Degradation Report",
        "",
        "Official benchmark deltas are not claimed unless the matching OTQ subset exists and the task/split/scoring match is reviewed.",
        "Mini-subset percentages are local smoke/practical signals, not full benchmark replacements.",
        "",
        "| Benchmark | Mode | OTQ Mini-Subset | Official Reference | Status | Rule |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in payload["rows"]:
        lines.append(
            f"| `{row['benchmark_id']}` | `{row['comparison_mode']}` | {format_subset_summaries(row)} | "
            f"{format_official_reference(row)} | `{row['claim_status']}` | {row['claim_rule']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Qwen3.6 OTQ benchmark degradation report.")
    parser.add_argument("--matrix", type=Path, default=Path("benchmarks/qwen36_benchmark_matrix.json"))
    parser.add_argument("--official-baseline", type=Path, default=Path("benchmarks/qwen36_official_language_baseline.json"))
    parser.add_argument("--subset-root", type=Path, default=Path("artifacts/qwen3.6-27b-benchmark-subsets"))
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/qwen3.6-27b-degradation-report"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    matrix = load_json(args.matrix)
    payload = {
        "schema": SCHEMA,
        "created_at": now_iso(),
        "matrix": str(args.matrix),
        "official_baseline": str(args.official_baseline),
        "subset_root": str(args.subset_root),
        "rows": build_rows(matrix, official_by_benchmark(args.official_baseline), load_subset_summaries(args.subset_root)),
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "degradation-report.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(args.output_root / "degradation-report.md", payload)
    print(args.output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
