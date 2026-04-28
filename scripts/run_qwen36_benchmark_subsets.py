#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


SCHEMA = "opentq.qwen36_benchmark_subset_plan.v1"

MODEL_PATHS = {
    "q3": Path("artifacts/hf-gguf-canonical/Qwen3.6-27B-OTQ-GGUF/Qwen3.6-27B-OTQ-DYN-Q3_K_M.gguf"),
    "q4": Path("artifacts/hf-gguf-canonical/Qwen3.6-27B-OTQ-GGUF/Qwen3.6-27B-OTQ-DYN-Q4_K_M.gguf"),
    "bf16": Path("Qwen/Qwen3.6-27B"),
}


@dataclass(frozen=True)
class ModelTarget:
    key: str
    path: Path

    def as_payload(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "path": str(self.path),
            "exists": self.path.exists() if self.key != "bf16" else None,
            "kind": "hf_source" if self.key == "bf16" else "gguf",
        }


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def load_matrix(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "opentq.qwen36_long_running_benchmark_matrix.v1":
        raise ValueError(f"unsupported benchmark matrix schema: {path}")
    return payload


def parse_models(value: str) -> list[ModelTarget]:
    targets: list[ModelTarget] = []
    for raw in value.split(","):
        key = raw.strip().lower()
        if not key:
            continue
        if key not in MODEL_PATHS:
            allowed = ", ".join(sorted(MODEL_PATHS))
            raise ValueError(f"unsupported model key {key!r}; expected one of: {allowed}")
        targets.append(ModelTarget(key=key, path=MODEL_PATHS[key]))
    if not targets:
        raise ValueError("at least one model key is required")
    return targets


def sample_count(row: dict[str, Any], defaults: dict[str, Any], sample_mode: str) -> int:
    if sample_mode == "smoke":
        return 4
    explicit_count = re.match(r"^(\d+)\b", str(row.get("subset_policy", "")))
    if explicit_count:
        return int(explicit_count.group(1))
    family = str(row.get("family", ""))
    if "agentic" in family:
        return int(defaults.get("agentic_samples_per_family", 3))
    if "coding" in family:
        return int(defaults.get("coding_samples_per_family", 12))
    if "judge" in family or family == "arena":
        return int(defaults.get("judge_samples_per_family", 12))
    if "reasoning" in family or family in {"math", "stem_reasoning"}:
        return int(defaults.get("reasoning_samples_per_family", 24))
    return int(defaults.get("quick_samples_per_family", 16))


def row_status(row: dict[str, Any], allow_judge: bool) -> str:
    mode = str(row.get("comparison_mode", ""))
    if mode == "blocked_modality":
        return "blocked_modality"
    if mode == "judge_based" and not allow_judge:
        return "skipped_judge"
    return "planned"


def build_plan(
    matrix: dict[str, Any],
    models: list[ModelTarget],
    output_root: Path,
    llama_cpp: Path,
    sample_mode: str,
    dry_run: bool,
    allow_judge: bool,
) -> dict[str, Any]:
    defaults = dict(matrix.get("default_subset_policy", {}))
    benchmarks = []
    for row in matrix.get("benchmarks", []):
        status = row_status(row, allow_judge)
        benchmarks.append(
            {
                "id": row["id"],
                "name": row["name"],
                "family": row["family"],
                "comparison_mode": row["comparison_mode"],
                "status": status,
                "sample_count": 0 if status in {"blocked_modality", "skipped_judge"} else sample_count(row, defaults, sample_mode),
                "official_baseline": row.get("official_baseline"),
                "subset_policy": row["subset_policy"],
                "claim_rule": row["claim_rule"],
            }
        )
    return {
        "schema": SCHEMA,
        "created_at": now_iso(),
        "dry_run": dry_run,
        "sample_mode": sample_mode,
        "llama_cpp": str(llama_cpp),
        "output_root": str(output_root),
        "models": [model.as_payload() for model in models],
        "benchmarks": benchmarks,
    }


def print_plan(plan: dict[str, Any]) -> None:
    for model in plan["models"]:
        exists = model["exists"]
        exists_text = "remote" if exists is None else str(bool(exists)).lower()
        print(f"model {model['key']} kind={model['kind']} exists={exists_text} path={model['path']}")
    for row in plan["benchmarks"]:
        print(
            f"{row['status']} {row['id']} mode={row['comparison_mode']} "
            f"samples={row['sample_count']} claim={row['claim_rule']}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan Qwen3.6 benchmark subset runs from the release matrix.")
    parser.add_argument("--matrix", type=Path, default=Path("benchmarks/qwen36_long_running_benchmark_matrix.json"))
    parser.add_argument("--models", default="q3,q4")
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/qwen3.6-27b-benchmark-subsets"))
    parser.add_argument("--llama-cpp", type=Path, default=Path("/Users/zlaabsi/Documents/GitHub/llama.cpp"))
    parser.add_argument("--sample-mode", choices=("smoke", "quick"), default="quick")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-judge", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    matrix = load_matrix(args.matrix)
    models = parse_models(args.models)
    plan = build_plan(
        matrix=matrix,
        models=models,
        output_root=args.output_root,
        llama_cpp=args.llama_cpp,
        sample_mode=args.sample_mode,
        dry_run=args.dry_run,
        allow_judge=args.allow_judge,
    )
    print_plan(plan)
    if not args.dry_run:
        print(
            "benchmark subset execution adapters are not implemented yet; "
            "rerun with --dry-run or implement Phase 4 adapters before producing score JSONs"
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
