from __future__ import annotations

import json
import math
import sys
import time
import zipfile
import zlib
from collections import defaultdict
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import numpy as np
from huggingface_hub import hf_hub_download
from rich.box import HEAVY, ROUNDED, SIMPLE_HEAVY
from rich.columns import Columns
from rich.console import Console, Group, RenderableType
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from safetensors import safe_open
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text


ShapeResolver = Callable[[str, str, str], tuple[tuple[int, ...] | None, str | None]]


def sanitize_tensor_name(name: str) -> str:
    return name.replace(".", "__")


def choose_chunk_rows(shape: tuple[int, ...], target_elements: int = 8_388_608) -> int:
    if not shape:
        return 1
    if len(shape) == 1:
        return max(1, min(shape[0], target_elements))
    row_width = int(np.prod(shape[1:]))
    return max(1, target_elements // max(row_width, 1))


def product(shape: tuple[int, ...]) -> int:
    return int(np.prod(shape, dtype=np.int64)) if shape else 1


def human_number(value: float | int | None) -> str:
    if value is None:
        return "-"
    units = ["", "K", "M", "B", "T"]
    value = float(value)
    index = 0
    while abs(value) >= 1000 and index < len(units) - 1:
        value /= 1000.0
        index += 1
    if index == 0:
        return f"{value:.0f}"
    return f"{value:.2f}{units[index]}"


def human_bytes(value: int | None) -> str:
    if value is None:
        return "-"
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    value = float(value)
    index = 0
    while value >= 1024 and index < len(units) - 1:
        value /= 1024.0
        index += 1
    if index == 0:
        return f"{int(value)}{units[index]}"
    return f"{value:.2f} {units[index]}"


def human_duration(value: float | None) -> str:
    if value is None:
        return "-"
    total = max(0, int(value))
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def format_metric(value: float | None) -> str:
    if value is None:
        return "-"
    if value == 0:
        return "0"
    if abs(value) < 0.001 or abs(value) >= 1000:
        return f"{value:.2e}"
    return f"{value:.6f}".rstrip("0").rstrip(".")


def format_percent(numerator: int | float | None, denominator: int | float | None) -> str:
    if numerator is None or denominator in (None, 0):
        return "-"
    return f"{(float(numerator) / float(denominator)) * 100:.1f}%"


def truncate(text: str, width: int) -> str:
    if len(text) <= width:
        return text
    if width <= 3:
        return text[:width]
    return text[: width - 3] + "..."


def compact_tensor_name(name: str, keep: int = 4) -> str:
    parts = name.split(".")
    if len(parts) <= keep:
        return name
    return "..." + ".".join(parts[-keep:])


def format_shape(shape: tuple[int, ...] | None) -> str:
    if not shape:
        return "-"
    return " x ".join(str(dim) for dim in shape)


def render_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(header) for header in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))
    header = "  ".join(value.ljust(widths[index]) for index, value in enumerate(headers))
    divider = "  ".join("-" * widths[index] for index in range(len(headers)))
    body = ["  ".join(value.ljust(widths[index]) for index, value in enumerate(row)) for row in rows]
    return "\n".join([header, divider, *body])


def progress_ratio(numerator: int | float | None, denominator: int | float | None) -> float:
    if numerator is None or denominator in (None, 0):
        return 0.0
    return max(0.0, min(float(numerator) / float(denominator), 1.0))


def state_style(state: str) -> str:
    return {
        "running": "cyan",
        "done": "green",
        "error": "red",
    }.get(state, "white")


def category_style(category: str | None) -> str:
    return {
        "mlp_proj": "magenta",
        "linear_attn_proj": "cyan",
        "self_attn_proj": "blue",
        "embeddings": "yellow",
        "layernorm": "green",
        "copy": "green",
    }.get(category or "", "white")


def metric_table(rows: list[tuple[RenderableType, RenderableType]], *, expand: bool = True) -> Table:
    table = Table.grid(expand=expand, padding=(0, 1))
    table.add_column(style="bold white")
    table.add_column(style="white")
    for label, value in rows:
        table.add_row(label, value)
    return table


def rich_progress_row(
    label: str,
    numerator: int | float | None,
    denominator: int | float | None,
    *,
    color: str,
    detail: str,
) -> Table:
    ratio = progress_ratio(numerator, denominator)
    bar = ProgressBar(
        total=100,
        completed=ratio * 100,
        width=None,
        complete_style=color,
        finished_style=color,
        pulse_style=f"{color} dim",
        style="grey27",
    )
    table = Table.grid(expand=True, padding=(0, 1))
    table.add_column(width=10, style="bold white")
    table.add_column(ratio=1)
    table.add_column(justify="right", style="white")
    table.add_row(label, bar, detail)
    return table


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def parse_release_start(log_path: Path) -> float | None:
    if not log_path.exists():
        return None
    with log_path.open("r", encoding="utf-8") as handle:
        first_line = handle.readline().strip()
    if not first_line.startswith("[") or "]" not in first_line:
        return None
    timestamp = first_line[1 : first_line.index("]")]
    try:
        return datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S").timestamp()
    except ValueError:
        return None


@lru_cache(maxsize=512)
def resolve_tensor_metadata(model_id: str, source_file: str, tensor_name: str) -> tuple[tuple[int, ...] | None, str | None]:
    try:
        local_path = hf_hub_download(repo_id=model_id, filename=source_file, local_files_only=True)
        with safe_open(local_path, framework="np") as reader:
            tensor_slice = reader.get_slice(tensor_name)
            shape = tuple(int(dim) for dim in tensor_slice.get_shape())
            dtype = str(tensor_slice.get_dtype()).lower()
            return shape, dtype
    except Exception:
        return None, None


def chunk_value_counts(shape: tuple[int, ...]) -> list[int]:
    if not shape:
        return [1]
    rows = choose_chunk_rows(shape)
    if len(shape) == 1:
        return [min(rows, shape[0] - start) for start in range(0, shape[0], rows)]
    row_width = int(np.prod(shape[1:]))
    return [min(rows, shape[0] - start) * row_width for start in range(0, shape[0], rows)]


def estimate_part_count(shape: tuple[int, ...]) -> int:
    return len(chunk_value_counts(shape))


def estimate_block_count(shape: tuple[int, ...], block_size: int) -> int:
    return sum(math.ceil(values / block_size) for values in chunk_value_counts(shape))


def estimate_group_count(shape: tuple[int, ...], group_size: int) -> int:
    return sum(math.ceil(values / group_size) for values in chunk_value_counts(shape))


def summarize_active_parts(part_paths: list[Path], mode: str) -> dict[str, Any]:
    summary = {
        "observed_part_count": len(part_paths),
        "readable_part_count": 0,
        "written_values": 0,
        "written_blocks": 0,
        "written_bytes": 0,
        "latest_part": None,
        "latest_row_start": None,
        "latest_row_stop": None,
        "latest_chunk_shape": None,
        "latest_modified_at": None,
    }
    latest_loaded_path: Path | None = None
    for path in sorted(part_paths):
        try:
            stat = path.stat()
        except OSError:
            continue
        summary["written_bytes"] += stat.st_size
        try:
            with np.load(path) as payload:
                summary["readable_part_count"] += 1
                chunk_shape = tuple(int(value) for value in payload["shape"].tolist()) if "shape" in payload.files else ()
                summary["written_values"] += product(chunk_shape)
                if mode == "quantize" and "indices" in payload.files:
                    summary["written_blocks"] += int(payload["indices"].shape[0])
                if latest_loaded_path is None or stat.st_mtime >= latest_loaded_path.stat().st_mtime:
                    latest_loaded_path = path
                    summary["latest_part"] = path.name
                    summary["latest_modified_at"] = stat.st_mtime
                    summary["latest_chunk_shape"] = chunk_shape
                    summary["latest_row_start"] = int(payload["row_start"][0]) if "row_start" in payload.files else None
                    summary["latest_row_stop"] = int(payload["row_stop"][0]) if "row_stop" in payload.files else None
        except (EOFError, ValueError, OSError, zipfile.BadZipFile, zlib.error):
            continue
    return summary


def build_recent_timeline(entries: list[dict[str, Any]], release_dir: Path, limit: int = 8) -> list[dict[str, Any]]:
    rows = []
    for entry in entries:
        meta_path = release_dir / entry["tensor_dir"] / "meta.json"
        completed_at = meta_path.stat().st_mtime if meta_path.exists() else None
        rows.append(
            {
                "completed_at": completed_at,
                "name": entry["name"],
                "category": entry["category"],
                "part_count": entry["part_count"],
                "num_values": entry["num_values"],
                "mse": entry.get("mse"),
                "max_abs_error": entry.get("max_abs_error"),
            }
        )
    rows.sort(key=lambda row: row["completed_at"] or 0, reverse=True)
    return rows[:limit]


def build_category_summary(entries: list[dict[str, Any]], limit: int = 8) -> list[dict[str, Any]]:
    stats: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"tensors": 0, "values": 0, "sum_squared_error": 0.0, "quant_values": 0, "max_abs_error": 0.0, "modes": defaultdict(int)}
    )
    for entry in entries:
        bucket = stats[entry["category"]]
        bucket["tensors"] += 1
        bucket["values"] += int(entry["num_values"])
        bucket["modes"][entry["mode"]] += 1
        if entry["mode"] == "quantize" and entry.get("sum_squared_error") is not None:
            bucket["sum_squared_error"] += float(entry["sum_squared_error"])
            bucket["quant_values"] += int(entry["num_values"])
            bucket["max_abs_error"] = max(bucket["max_abs_error"], float(entry.get("max_abs_error") or 0.0))

    rows = []
    for category, bucket in stats.items():
        rows.append(
            {
                "category": category,
                "tensors": bucket["tensors"],
                "values": bucket["values"],
                "avg_mse": None if bucket["quant_values"] == 0 else bucket["sum_squared_error"] / bucket["quant_values"],
                "max_abs_error": None if bucket["quant_values"] == 0 else bucket["max_abs_error"],
                "quantized": bucket["modes"].get("quantize", 0),
                "copied": bucket["modes"].get("copy", 0),
            }
        )
    rows.sort(key=lambda row: row["values"], reverse=True)
    return rows[:limit]


def build_current_tensor_summary(
    release_dir: Path,
    plan: dict[str, Any],
    entries: list[dict[str, Any]],
    shape_resolver: ShapeResolver,
) -> dict[str, Any] | None:
    tensors_root = release_dir / "tensors"
    if not tensors_root.exists():
        return None

    completed_names = {entry["name"] for entry in entries}
    plan_rows = {sanitize_tensor_name(row["name"]): row for row in plan["tensors"]}
    active_dirs = [child for child in tensors_root.iterdir() if child.is_dir() and not (child / "meta.json").exists()]
    if not active_dirs:
        return None

    active_dir = max(active_dirs, key=lambda path: path.stat().st_mtime)
    plan_row = plan_rows.get(active_dir.name)
    part_paths = sorted(active_dir.glob("part-*"))
    active_parts = summarize_active_parts(part_paths, plan_row["mode"] if plan_row else "quantize")

    tensor_name = plan_row["name"] if plan_row else active_dir.name.replace("__", ".")
    if tensor_name in completed_names:
        return None

    full_shape = None
    full_dtype = None
    if plan_row:
        full_shape, full_dtype = shape_resolver(plan["model_id"], plan_row["source_file"], plan_row["name"])

    summary: dict[str, Any] = {
        "name": tensor_name,
        "category": None if plan_row is None else plan_row["category"],
        "mode": None if plan_row is None else plan_row["mode"],
        "variant_name": None if plan_row is None else plan_row.get("variant_name"),
        "source_file": None if plan_row is None else plan_row["source_file"],
        "tensor_dir": str(active_dir.relative_to(release_dir)),
        "dtype": full_dtype,
        "shape": full_shape,
        "written_values": active_parts["written_values"],
        "written_blocks": active_parts["written_blocks"],
        "written_bytes": active_parts["written_bytes"],
        "part_count_done": active_parts["readable_part_count"],
        "part_count_observed": active_parts["observed_part_count"],
        "latest_part": active_parts["latest_part"],
        "latest_row_start": active_parts["latest_row_start"],
        "latest_row_stop": active_parts["latest_row_stop"],
        "latest_chunk_shape": active_parts["latest_chunk_shape"],
        "latest_modified_at": active_parts["latest_modified_at"],
    }

    if full_shape is not None:
        total_values = product(full_shape)
        summary["total_values"] = total_values
        summary["value_progress"] = active_parts["written_values"] / total_values if total_values else None
        summary["expected_parts"] = estimate_part_count(full_shape)
        if len(full_shape) >= 1:
            summary["rows_total"] = full_shape[0]
            summary["rows_done"] = active_parts["latest_row_stop"]
            summary["row_progress"] = (
                None
                if (not full_shape[0] or active_parts["latest_row_stop"] is None)
                else active_parts["latest_row_stop"] / full_shape[0]
            )
        if plan_row and plan_row["mode"] == "quantize" and plan_row.get("variant_name"):
            from .variants import get_variant

            variant = get_variant(plan_row["variant_name"])
            summary["group_size"] = variant.group_size
            summary["block_size"] = variant.block_size
            summary["sub_block_size"] = variant.sub_block_size
            summary["expected_blocks"] = estimate_block_count(full_shape, variant.block_size)
            summary["expected_groups"] = estimate_group_count(full_shape, variant.group_size)
            summary["block_progress"] = (
                None if summary["expected_blocks"] == 0 else active_parts["written_blocks"] / summary["expected_blocks"]
            )
    return summary


def build_next_tensor_summary(
    plan: dict[str, Any],
    entries: list[dict[str, Any]],
    current: dict[str, Any] | None,
    shape_resolver: ShapeResolver,
) -> dict[str, Any] | None:
    completed_names = {entry["name"] for entry in entries}
    current_name = None if current is None else current["name"]
    for row in plan["tensors"]:
        if row["name"] in completed_names or row["name"] == current_name:
            continue
        shape, dtype = shape_resolver(plan["model_id"], row["source_file"], row["name"])
        payload: dict[str, Any] = {
            "name": row["name"],
            "category": row["category"],
            "mode": row["mode"],
            "variant_name": row.get("variant_name"),
            "source_file": row["source_file"],
            "shape": shape,
            "dtype": dtype,
        }
        if shape is not None and row["mode"] == "quantize" and row.get("variant_name"):
            from .variants import get_variant

            variant = get_variant(row["variant_name"])
            payload["total_values"] = product(shape)
            payload["expected_parts"] = estimate_part_count(shape)
            payload["expected_blocks"] = estimate_block_count(shape, variant.block_size)
        return payload
    return None


def build_release_monitor(
    release_dir: Path,
    shape_resolver: ShapeResolver = resolve_tensor_metadata,
    now: float | None = None,
) -> dict[str, Any]:
    if now is None:
        now = time.time()
    plan = load_json(release_dir / "plan.json")
    entries = load_jsonl(release_dir / "progress.jsonl")
    current = build_current_tensor_summary(release_dir, plan, entries, shape_resolver)
    upcoming = build_next_tensor_summary(plan, entries, current, shape_resolver)

    by_mode = defaultdict(int)
    planned_by_mode = defaultdict(int)
    planned_by_variant = plan["summary"]["by_action"]
    for key, count in planned_by_variant.items():
        mode, _variant = key.split(":", 1)
        planned_by_mode[mode] += count
    for entry in entries:
        by_mode[entry["mode"]] += 1

    completed_values = sum(int(entry["num_values"]) for entry in entries)
    total_values_done = completed_values + int(current["written_values"]) if current else completed_values
    total_parts_done = sum(int(entry["part_count"]) for entry in entries) + (0 if current is None else int(current["part_count_done"]))
    quant_entries = [entry for entry in entries if entry["mode"] == "quantize" and entry.get("sum_squared_error") is not None]
    total_sse = sum(float(entry["sum_squared_error"]) for entry in quant_entries)
    total_quant_values = sum(int(entry["num_values"]) for entry in quant_entries)

    manifest_path = release_dir / "manifest.json"
    log_path = release_dir.parent / "logs" / f"{release_dir.name}.log"
    start_ts = parse_release_start(log_path)
    elapsed_seconds = None if start_ts is None else max(now - start_ts, 0.0)
    if manifest_path.exists():
        manifest = load_json(manifest_path)
        elapsed_seconds = manifest.get("elapsed_seconds", elapsed_seconds)

    release = {
        "release": release_dir.name,
        "state": "done" if manifest_path.exists() else "running",
        "model_id": plan["model_id"],
        "summary": {
            "tensors_total": plan["summary"]["total_tensors"],
            "tensors_done": len(entries),
            "tensors_started": len(entries) + (0 if current is None else 1),
            "planned_quantize": planned_by_mode.get("quantize", 0),
            "planned_copy": planned_by_mode.get("copy", 0),
            "done_quantize": by_mode.get("quantize", 0),
            "done_copy": by_mode.get("copy", 0),
            "values_done": total_values_done,
            "part_files_done": total_parts_done,
            "avg_quant_mse": None if total_quant_values == 0 else total_sse / total_quant_values,
            "max_abs_error": None if not quant_entries else max(float(entry.get("max_abs_error") or 0.0) for entry in quant_entries),
            "elapsed_seconds": elapsed_seconds,
            "values_per_second": None if not elapsed_seconds else total_values_done / elapsed_seconds,
        },
        "current": current,
        "next": upcoming,
        "recent": build_recent_timeline(entries, release_dir),
        "categories": build_category_summary(entries),
    }
    return release


def build_monitor_payload(
    root: str | Path = "artifacts/qwen3.6-27b",
    *,
    shape_resolver: ShapeResolver = resolve_tensor_metadata,
    now: float | None = None,
) -> dict[str, Any]:
    monitor_root = Path(root)
    if not monitor_root.exists():
        return {"root": str(monitor_root), "exists": False}

    release_dirs = [
        child
        for child in sorted(monitor_root.iterdir())
        if child.is_dir() and child.name.startswith("Qwen3.6-27B-") and (child / "plan.json").exists()
    ]
    releases = [build_release_monitor(child, shape_resolver=shape_resolver, now=now) for child in release_dirs]

    selected = None
    for release in releases:
        if release["state"] == "running":
            selected = release["release"]
            break
    if selected is None and releases:
        selected = releases[-1]["release"]

    return {
        "root": str(monitor_root),
        "updated_at": now if now is not None else time.time(),
        "selected_release": selected,
        "releases": releases,
    }


def build_header_renderable(payload: dict[str, Any]) -> Panel:
    selected_release = payload.get("selected_release") or "-"
    left = Columns(
        [
            Spinner("dots", style="cyan"),
            Text("OpenTQ Monitor", style="bold white"),
            Text(selected_release, style="bold cyan"),
        ],
        expand=False,
        padding=(0, 1),
    )
    right = Group(
        Text(payload["root"], style="bright_black", justify="right"),
        Text(datetime.fromtimestamp(payload["updated_at"]).strftime("%Y-%m-%d %H:%M:%S"), style="bright_black", justify="right"),
    )
    grid = Table.grid(expand=True)
    grid.add_column(ratio=1)
    grid.add_column(justify="right", width=40)
    grid.add_row(left, right)
    return Panel(grid, box=HEAVY, border_style="cyan", padding=(0, 1))


def build_release_panel(releases: list[dict[str, Any]], selected_release: str | None) -> Panel:
    table = Table(box=SIMPLE_HEAVY, expand=True, header_style="bold cyan")
    table.add_column("release", ratio=2)
    table.add_column("state", width=9)
    table.add_column("tensors", width=20)
    table.add_column("quant", width=11)
    table.add_column("copy", width=10)
    table.add_column("values", width=10, justify="right")
    table.add_column("parts", width=9, justify="right")
    table.add_column("elapsed", width=10)
    table.add_column("rate", width=11, justify="right")
    for release in releases:
        summary = release["summary"]
        style = "bold white on rgb(25,35,60)" if release["release"] == selected_release else ""
        table.add_row(
            truncate(release["release"], 26),
            Text(release["state"], style=f"bold {state_style(release['state'])}"),
            f"{summary['tensors_done']}/{summary['tensors_total']} ({format_percent(summary['tensors_done'], summary['tensors_total'])})",
            f"{summary['done_quantize']}/{summary['planned_quantize']}",
            f"{summary['done_copy']}/{summary['planned_copy']}",
            human_number(summary["values_done"]),
            human_number(summary["part_files_done"]),
            human_duration(summary["elapsed_seconds"]),
            human_number(summary["values_per_second"]) + "/s" if summary["values_per_second"] is not None else "-",
            style=style,
        )
    return Panel(table, title="Releases", border_style="cyan", box=ROUNDED, padding=(0, 1))


def build_overview_panel(selected: dict[str, Any]) -> Panel:
    summary = selected["summary"]
    metrics = metric_table(
        [
            ("state", Text(selected["state"], style=f"bold {state_style(selected['state'])}")),
            ("elapsed", human_duration(summary["elapsed_seconds"])),
            ("rate", human_number(summary["values_per_second"]) + "/s" if summary["values_per_second"] is not None else "-"),
            ("values", human_number(summary["values_done"])),
            ("avg mse", format_metric(summary["avg_quant_mse"])),
            ("max err", format_metric(summary["max_abs_error"])),
        ]
    )
    progress = Group(
        rich_progress_row(
            "tensors",
            summary["tensors_done"],
            summary["tensors_total"],
            color="cyan",
            detail=f"{summary['tensors_done']}/{summary['tensors_total']}",
        ),
        rich_progress_row(
            "quant",
            summary["done_quantize"],
            summary["planned_quantize"],
            color="magenta",
            detail=f"{summary['done_quantize']}/{summary['planned_quantize']}",
        ),
        rich_progress_row(
            "copy",
            summary["done_copy"],
            summary["planned_copy"],
            color="green",
            detail=f"{summary['done_copy']}/{summary['planned_copy']}",
        ),
    )
    content = Group(
        Text(selected["release"], style="bold white"),
        Text(selected["model_id"], style="bright_black"),
        Text(""),
        metrics,
        Text(""),
        progress,
    )
    return Panel(content, title="Overview", border_style="bright_blue", box=ROUNDED, padding=(0, 1))


def build_current_panel(selected: dict[str, Any]) -> Panel:
    current = selected.get("current")
    upcoming = selected.get("next")
    border = category_style(current["category"] if current else upcoming["category"] if upcoming else None)

    if current is None:
        if upcoming is None:
            body: RenderableType = Group(
                Text("No active tensor directory.", style="bold white"),
                Text("The runner is between tensors or the release is complete.", style="bright_black"),
            )
        else:
            body = Group(
                Text(compact_tensor_name(upcoming["name"], keep=5), style="bold white"),
                Text(
                    f"{upcoming['mode']}  {upcoming.get('variant_name') or '-'}  {upcoming['category']}  {upcoming.get('dtype') or '-'}",
                    style=f"bold {category_style(upcoming['category'])}",
                ),
                Text(f"shape {format_shape(upcoming.get('shape'))}  shard {upcoming.get('source_file') or '-'}", style="bright_black"),
                Text(""),
                rich_progress_row("parts", 0, upcoming.get("expected_parts"), color="cyan", detail=f"0/{upcoming.get('expected_parts', '-')}"),
                rich_progress_row(
                    "values",
                    0,
                    upcoming.get("total_values"),
                    color="magenta",
                    detail=f"0/{human_number(upcoming.get('total_values'))}",
                ),
                rich_progress_row(
                    "blocks",
                    0,
                    upcoming.get("expected_blocks"),
                    color="blue",
                    detail=f"0/{human_number(upcoming.get('expected_blocks'))}",
                ),
            )
        return Panel(body, title="Current Tensor", border_style=border, box=ROUNDED, padding=(0, 1))

    progress_rows: list[RenderableType] = [
        rich_progress_row(
            "parts",
            current["part_count_done"],
            current.get("expected_parts"),
            color="cyan",
            detail=f"{current['part_count_done']} ok / {current['part_count_observed']} seen / {current.get('expected_parts', '-')}",
        ),
        rich_progress_row(
            "values",
            current["written_values"],
            current.get("total_values"),
            color="magenta",
            detail=f"{human_number(current['written_values'])}/{human_number(current.get('total_values'))}",
        ),
    ]
    if current.get("rows_total") is not None:
        progress_rows.append(
            rich_progress_row(
                "rows",
                current.get("rows_done"),
                current["rows_total"],
                color="green",
                detail=f"{current.get('rows_done') if current.get('rows_done') is not None else '-'} / {current['rows_total']}",
            )
        )
    if current.get("expected_blocks") is not None:
        progress_rows.append(
            rich_progress_row(
                "blocks",
                current["written_blocks"],
                current["expected_blocks"],
                color="blue",
                detail=f"{human_number(current['written_blocks'])}/{human_number(current['expected_blocks'])}",
            )
        )

    details = metric_table(
        [
            ("mode", f"{current.get('mode') or '-'} / {current.get('variant_name') or '-'}"),
            ("category", Text(current.get("category") or "-", style=f"bold {category_style(current.get('category'))}")),
            ("shape", format_shape(current.get("shape"))),
            ("dtype", current.get("dtype") or "-"),
            ("shard", current.get("source_file") or "-"),
            ("latest", current.get("latest_part") or "-"),
            ("range", f"{current.get('latest_row_start')}:{current.get('latest_row_stop')}"),
            ("chunk", format_shape(current.get("latest_chunk_shape"))),
            ("bytes", human_bytes(current.get("written_bytes"))),
            ("groups", f"{human_number(math.ceil(current['written_values'] / current['group_size']))}/{human_number(current['expected_groups'])}" if current.get("expected_groups") is not None else "-"),
            ("block cfg", f"{current.get('group_size', '-')} / {current.get('block_size', '-')} / {current.get('sub_block_size', '-')}"),
            ("tensor dir", current.get("tensor_dir") or "-"),
        ]
    )

    body = Group(
        Text(compact_tensor_name(current["name"], keep=5), style="bold white"),
        Text(datetime.fromtimestamp(current["latest_modified_at"]).strftime("%H:%M:%S") if current.get("latest_modified_at") else "preparing...", style="bright_black"),
        Text(""),
        *progress_rows,
        Text(""),
        details,
    )
    return Panel(body, title="Current Tensor", border_style=border, box=ROUNDED, padding=(0, 1))


def build_recent_panel(selected: dict[str, Any]) -> Panel:
    table = Table(box=SIMPLE_HEAVY, expand=True, header_style="bold cyan")
    table.add_column("time", width=8)
    table.add_column("tensor", ratio=3)
    table.add_column("category", width=18)
    table.add_column("values", width=10, justify="right")
    table.add_column("parts", width=7, justify="right")
    table.add_column("mse", width=10, justify="right")
    table.add_column("maxerr", width=10, justify="right")
    recent = selected["recent"]
    if not recent:
        table.add_row("-", "No completed tensors yet.", "-", "-", "-", "-", "-")
    for event in recent:
        timestamp = "-" if event["completed_at"] is None else datetime.fromtimestamp(event["completed_at"]).strftime("%H:%M:%S")
        table.add_row(
            timestamp,
            truncate(compact_tensor_name(event["name"], keep=4), 44),
            Text(event["category"], style=f"bold {category_style(event['category'])}"),
            human_number(event["num_values"]),
            str(event["part_count"]),
            format_metric(event["mse"]),
            format_metric(event["max_abs_error"]),
        )
    return Panel(table, title="Recent Timeline", border_style="cyan", box=ROUNDED, padding=(0, 1))


def build_category_panel(selected: dict[str, Any]) -> Panel:
    table = Table(box=SIMPLE_HEAVY, expand=True, header_style="bold cyan")
    table.add_column("category", ratio=2)
    table.add_column("tensors", width=8, justify="right")
    table.add_column("values", width=10, justify="right")
    table.add_column("quant", width=7, justify="right")
    table.add_column("copy", width=7, justify="right")
    table.add_column("avg_mse", width=10, justify="right")
    table.add_column("maxerr", width=10, justify="right")
    categories = selected["categories"]
    if not categories:
        table.add_row("No category stats yet.", "-", "-", "-", "-", "-", "-")
    for row in categories:
        table.add_row(
            Text(row["category"], style=f"bold {category_style(row['category'])}"),
            str(row["tensors"]),
            human_number(row["values"]),
            str(row["quantized"]),
            str(row["copied"]),
            format_metric(row["avg_mse"]),
            format_metric(row["max_abs_error"]),
        )
    return Panel(table, title="By Category", border_style="cyan", box=ROUNDED, padding=(0, 1))


def build_footer_renderable() -> Panel:
    footer = Columns(
        [
            Text("ctrl-c", style="bold cyan"),
            Text("stop watch", style="bright_black"),
            Text("uv run opentq status", style="bold green"),
            Text("machine-readable JSON", style="bright_black"),
        ],
        expand=True,
        equal=False,
        padding=(0, 2),
    )
    return Panel(footer, border_style="grey35", box=ROUNDED, padding=(0, 1))


def build_monitor_renderable(payload: dict[str, Any], *, width: int | None = None) -> RenderableType:
    if payload.get("exists") is False:
        body = Group(
            Text("No runs found.", style="bold white"),
            Text(f"root: {payload['root']}", style="bright_black"),
            Text("Start a quantization batch, then rerun this command.", style="bright_black"),
        )
        return Panel(body, title="OpenTQ Monitor", border_style="yellow", box=ROUNDED, padding=(1, 2))

    releases: list[dict[str, Any]] = payload["releases"]
    if not releases:
        body = Group(
            Text("No release directories found.", style="bold white"),
            Text(f"root: {payload['root']}", style="bright_black"),
        )
        return Panel(body, title="OpenTQ Monitor", border_style="yellow", box=ROUNDED, padding=(1, 2))

    selected = next((release for release in releases if release["release"] == payload.get("selected_release")), releases[0])

    if width is None:
        width = 160

    layout = Layout(name="root")
    layout.split_column(
        Layout(build_header_renderable(payload), name="header", size=5),
        Layout(name="main", ratio=1),
        Layout(build_footer_renderable(), name="footer", size=3),
    )
    if width < 140:
        layout["main"].split_column(
            Layout(build_release_panel(releases, payload.get("selected_release")), name="releases", size=10),
            Layout(build_overview_panel(selected), name="overview", size=13),
            Layout(build_current_panel(selected), name="current", size=18),
            Layout(build_recent_panel(selected), name="recent", size=14),
            Layout(build_category_panel(selected), name="categories", ratio=1),
        )
        return layout

    layout["main"].split_row(
        Layout(name="left", ratio=5),
        Layout(name="right", ratio=6),
    )
    layout["left"].split_column(
        Layout(build_release_panel(releases, payload.get("selected_release")), name="releases", size=10),
        Layout(build_overview_panel(selected), name="overview", size=13),
        Layout(build_current_panel(selected), name="current", ratio=1),
    )
    layout["right"].split_column(
        Layout(build_recent_panel(selected), name="recent", ratio=3),
        Layout(build_category_panel(selected), name="categories", ratio=2),
    )
    return layout


def render_monitor(payload: dict[str, Any]) -> str:
    if payload.get("exists") is False:
        return f"OpenTQ Monitor\n\nNo runs found under {payload['root']}.\nStart a quantization batch, then rerun this command."

    releases: list[dict[str, Any]] = payload["releases"]
    if not releases:
        return f"OpenTQ Monitor\n\nNo release directories found under {payload['root']}."

    selected = next((release for release in releases if release["release"] == payload.get("selected_release")), releases[0])
    lines = [
        "OpenTQ Monitor",
        f"root: {payload['root']}",
        f"updated: {datetime.fromtimestamp(payload['updated_at']).strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "Releases",
    ]

    release_rows = []
    for release in releases:
        summary = release["summary"]
        release_rows.append(
            [
                truncate(release["release"], 24),
                release["state"],
                f"{summary['tensors_done']}/{summary['tensors_total']} ({format_percent(summary['tensors_done'], summary['tensors_total'])})",
                f"{summary['done_quantize']}/{summary['planned_quantize']}",
                f"{summary['done_copy']}/{summary['planned_copy']}",
                human_number(summary["values_done"]),
                human_number(summary["part_files_done"]),
                human_duration(summary["elapsed_seconds"]),
                human_number(summary["values_per_second"]) + "/s" if summary["values_per_second"] is not None else "-",
            ]
        )
    lines.append(
        render_table(
            ["release", "state", "tensors", "quant", "copy", "values", "parts", "elapsed", "rate"],
            release_rows,
        )
    )

    lines.extend(["", f"Current Release: {selected['release']}"])
    summary = selected["summary"]
    lines.append(
        f"done={summary['tensors_done']}/{summary['tensors_total']}  "
        f"quant={summary['done_quantize']}/{summary['planned_quantize']}  "
        f"copy={summary['done_copy']}/{summary['planned_copy']}  "
        f"avg_mse={format_metric(summary['avg_quant_mse'])}  "
        f"max_abs_error={format_metric(summary['max_abs_error'])}"
    )

    current = selected.get("current")
    lines.extend(["", "Current Tensor"])
    if current is None:
        if selected.get("next") is None:
            lines.append("No active tensor directory yet. The runner is between tensors or the release is complete.")
        else:
            upcoming = selected["next"]
            lines.append("No active tensor directory yet. The runner is between tensors.")
            lines.append(
                f"next={compact_tensor_name(upcoming['name'], keep=5)}  mode={upcoming['mode']}  "
                f"variant={upcoming.get('variant_name') or '-'}  category={upcoming['category']}"
            )
            lines.append(
                f"shape={format_shape(upcoming.get('shape'))}  dtype={upcoming.get('dtype') or '-'}  "
                f"source={upcoming.get('source_file') or '-'}"
            )
            if upcoming.get("total_values") is not None:
                lines.append(
                    f"expected_values={human_number(upcoming['total_values'])}  "
                    f"expected_parts={upcoming.get('expected_parts', '-')}  "
                    f"expected_blocks={human_number(upcoming.get('expected_blocks'))}"
                )
    else:
        lines.append(compact_tensor_name(current["name"], keep=5))
        lines.append(
            f"mode={current.get('mode') or '-'}  variant={current.get('variant_name') or '-'}  "
            f"category={current.get('category') or '-'}  dtype={current.get('dtype') or '-'}"
        )
        lines.append(f"shape={format_shape(current.get('shape'))}  shard={current.get('source_file') or '-'}")
        lines.append(
            f"parts={current['part_count_done']} readable / {current['part_count_observed']} observed / "
            f"{current.get('expected_parts', '-') or '-'} expected  "
            f"values={human_number(current['written_values'])}/{human_number(current.get('total_values'))} "
            f"({format_percent(current['written_values'], current.get('total_values'))})"
        )
        if current.get("rows_total") is not None:
            lines.append(
                f"rows={current.get('rows_done') if current.get('rows_done') is not None else '-'}"
                f"/{current['rows_total']} "
                f"({format_percent(current.get('rows_done'), current['rows_total'])})  "
                f"latest_range={current.get('latest_row_start')}:{current.get('latest_row_stop')}"
            )
        if current.get("expected_blocks") is not None:
            lines.append(
                f"blocks={human_number(current['written_blocks'])}/{human_number(current['expected_blocks'])} "
                f"({format_percent(current['written_blocks'], current['expected_blocks'])})  "
                f"groups={human_number(math.ceil(current['written_values'] / current['group_size']))}/{human_number(current['expected_groups'])}"
            )
            lines.append(
                f"group_size={current['group_size']}  block_size={current['block_size']}  "
                f"sub_block_size={current['sub_block_size']}"
            )
        lines.append(
            f"latest_part={current.get('latest_part') or '-'}  chunk_shape={format_shape(current.get('latest_chunk_shape'))}  "
            f"written_bytes={human_bytes(current.get('written_bytes'))}"
        )

    lines.extend(["", "Recent Timeline"])
    recent_rows = []
    for event in selected["recent"]:
        timestamp = "-" if event["completed_at"] is None else datetime.fromtimestamp(event["completed_at"]).strftime("%H:%M:%S")
        recent_rows.append(
            [
                timestamp,
                truncate(compact_tensor_name(event["name"], keep=4), 42),
                event["category"],
                human_number(event["num_values"]),
                str(event["part_count"]),
                format_metric(event["mse"]),
                format_metric(event["max_abs_error"]),
            ]
        )
    if recent_rows:
        lines.append(render_table(["time", "tensor", "category", "values", "parts", "mse", "maxerr"], recent_rows))
    else:
        lines.append("No completed tensors yet.")

    lines.extend(["", "By Category"])
    category_rows = []
    for row in selected["categories"]:
        category_rows.append(
            [
                row["category"],
                str(row["tensors"]),
                human_number(row["values"]),
                str(row["quantized"]),
                str(row["copied"]),
                format_metric(row["avg_mse"]),
                format_metric(row["max_abs_error"]),
            ]
        )
    if category_rows:
        lines.append(render_table(["category", "tensors", "values", "quant", "copy", "avg_mse", "maxerr"], category_rows))
    else:
        lines.append("No category stats yet.")

    lines.extend(
        [
            "",
            "Actions",
            "ctrl-c to stop watch mode",
            "use `uv run opentq status` for machine-readable JSON",
        ]
    )
    return "\n".join(lines)


def print_monitor(payload: dict[str, Any], clear: bool = False) -> None:
    console = Console()
    if clear and console.is_terminal:
        console.clear()
    console.print(build_monitor_renderable(payload, width=console.size.width))


def watch_monitor(root: str | Path = "artifacts/qwen3.6-27b", interval: float = 5.0) -> int:
    console = Console()
    try:
        with Live(
            build_monitor_renderable(build_monitor_payload(root), width=console.size.width),
            console=console,
            screen=True,
            auto_refresh=True,
            refresh_per_second=8,
        ) as live:
            while True:
                live.update(build_monitor_renderable(build_monitor_payload(root), width=console.size.width), refresh=True)
                time.sleep(interval)
    except KeyboardInterrupt:
        return 130
