from __future__ import annotations

import hashlib
import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .bitpack import unpack_bits
from .gguf_export import pack_group_record
from .hadamard import hadamard_unrotate_groups
from .release_pack import PACK_SCHEMA
from .runtime import OpenTQPack, reconstruct_rotated
from .run import tensor_seed
from .variants import QuantVariant, get_variant


@dataclass(frozen=True)
class PackAuditOptions:
    packed_dir: Path
    max_tensors: int | None = None
    dequantize_samples: int = 4


@dataclass(frozen=True)
class RuntimeProbeOptions:
    packed_dir: Path
    fixtures_output: Path
    probe_binary: Path | None = None
    output: Path | None = None
    audit_max_tensors: int | None = None
    audit_dequantize_samples: int = 4
    max_fixtures_per_variant: int = 1
    timeout_seconds: float = 120.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _variant_counts(tensors: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in tensors:
        if row.get("mode") != "quantize":
            continue
        variant = str(row.get("variant_name"))
        counts[variant] = counts.get(variant, 0) + 1
    return dict(sorted(counts.items()))


def _mode_counts(tensors: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in tensors:
        mode = str(row.get("mode"))
        counts[mode] = counts.get(mode, 0) + 1
    return dict(sorted(counts.items()))


def audit_packed_runtime(options: PackAuditOptions) -> dict[str, Any]:
    started = time.time()
    pack = OpenTQPack(options.packed_dir)
    manifest = pack.manifest
    tensors = list(manifest.get("tensors", []))
    selected_tensors = tensors[: options.max_tensors] if options.max_tensors is not None else tensors

    schema_pass = manifest.get("schema") == PACK_SCHEMA
    checksum_rows: list[dict[str, Any]] = []
    missing_files: list[str] = []
    checksum_failures: list[str] = []
    for row in selected_tensors:
        rel = Path(row["file"])
        path = options.packed_dir / rel
        if not path.exists():
            missing_files.append(str(rel))
            checksum_rows.append({"name": row["name"], "file": str(rel), "status": "missing"})
            continue
        actual = sha256_file(path)
        expected = row.get("sha256")
        ok = actual == expected
        if not ok:
            checksum_failures.append(row["name"])
        checksum_rows.append(
            {
                "name": row["name"],
                "file": str(rel),
                "bytes": path.stat().st_size,
                "sha256_pass": ok,
            }
        )

    dequant_samples: list[dict[str, Any]] = []
    dequant_failures: list[str] = []
    for row in selected_tensors:
        if len(dequant_samples) >= options.dequantize_samples:
            break
        try:
            decoded = pack.dequantize_tensor(row["name"], dtype=np.float32)
            finite = bool(np.isfinite(decoded).all())
            shape_pass = list(decoded.shape) == list(row["shape"])
            sample = {
                "name": row["name"],
                "mode": row.get("mode"),
                "variant": row.get("variant_name"),
                "shape": list(decoded.shape),
                "finite": finite,
                "min": float(np.min(decoded)) if decoded.size else 0.0,
                "max": float(np.max(decoded)) if decoded.size else 0.0,
                "mean": float(np.mean(decoded)) if decoded.size else 0.0,
                "shape_pass": shape_pass,
            }
            if not finite or not shape_pass:
                dequant_failures.append(row["name"])
        except Exception as exc:  # pragma: no cover - exercised by integration failures
            sample = {
                "name": row.get("name"),
                "mode": row.get("mode"),
                "variant": row.get("variant_name"),
                "error": str(exc),
            }
            dequant_failures.append(str(row.get("name")))
        dequant_samples.append(sample)

    payload = {
        "schema": "opentq.runtime_pack_audit.v1",
        "packed_dir": str(options.packed_dir),
        "release_slug": manifest.get("release_slug"),
        "model_id": manifest.get("model_id"),
        "pack_schema": manifest.get("schema"),
        "schema_pass": schema_pass,
        "totals": manifest.get("totals", {}),
        "mode_counts": _mode_counts(tensors),
        "variant_counts": _variant_counts(tensors),
        "selected_tensor_count": len(selected_tensors),
        "checksum": {
            "checked": len(checksum_rows),
            "missing_files": missing_files,
            "failures": checksum_failures,
            "rows": checksum_rows[:16],
        },
        "dequantize": {
            "sample_count": len(dequant_samples),
            "failures": dequant_failures,
            "samples": dequant_samples,
        },
        "elapsed_seconds": round(time.time() - started, 3),
    }
    payload["overall_pass"] = bool(schema_pass and not missing_files and not checksum_failures and not dequant_failures)
    return payload


def _first_quant_sections(pack: OpenTQPack, max_per_variant: int) -> list[tuple[dict[str, Any], dict[str, Any], int]]:
    counts: dict[str, int] = {}
    selected: list[tuple[dict[str, Any], dict[str, Any], int]] = []
    for row in pack.manifest.get("tensors", []):
        if row.get("mode") != "quantize":
            continue
        variant = row["variant_name"]
        if counts.get(variant, 0) >= max_per_variant:
            continue
        section = row["sections"][0]
        selected.append((row, section, 0))
        counts[variant] = counts.get(variant, 0) + 1
    return selected


def _group_payload(pack: OpenTQPack, row: dict[str, Any], part: dict[str, Any], part_index: int, group_index: int = 0) -> dict[str, Any]:
    variant = get_variant(row["variant_name"])
    indices = unpack_bits(pack.read_section(row, part["indices"]), variant.weight_bits, int(part["index_count"])).reshape(-1, variant.group_size)
    if group_index >= indices.shape[0]:
        raise IndexError(f"group index {group_index} is outside tensor part with {indices.shape[0]} groups")

    scales = np.frombuffer(pack.read_section(row, part["scales"]), dtype=np.float16).reshape(tuple(part["scale_shape"]))
    scales = scales.reshape(indices.shape[0], -1)
    residual_indices = None
    residual_scales = None
    if variant.residual_bits is not None:
        residual_indices = unpack_bits(
            pack.read_section(row, part["residual_indices"]),
            variant.residual_bits,
            int(part["residual_index_count"]),
        ).reshape(-1, variant.group_size)
        residual_scales = np.frombuffer(pack.read_section(row, part["residual_scales"]), dtype=np.float16).reshape(tuple(part["residual_scale_shape"]))
        residual_scales = residual_scales.reshape(indices.shape[0], -1)

    seed_base = int(row.get("seed", tensor_seed(pack.manifest["release_slug"], row["name"])))
    seed = seed_base + part_index * 104729 + group_index * variant.group_size
    block = pack_group_record(
        seed,
        indices[group_index],
        scales[group_index],
        variant,
        None if residual_indices is None else residual_indices[group_index],
        None if residual_scales is None else residual_scales[group_index],
    )
    expected = _decode_group(indices[group_index], scales[group_index], variant, seed, residual_indices, residual_scales, group_index)
    activation = _probe_activation(variant.group_size)
    expected_dot = np.array([float(np.dot(expected, _q8_0_dequantize(activation)))], dtype=np.float32)
    return {
        "variant": variant.name,
        "seed": seed,
        "block": block,
        "expected": expected.astype(np.float32),
        "activation": activation.astype(np.float32),
        "expected_dot": expected_dot,
    }


def _decode_group(
    indices: np.ndarray,
    scales: np.ndarray,
    variant: QuantVariant,
    seed: int,
    residual_indices: np.ndarray | None,
    residual_scales: np.ndarray | None,
    group_index: int,
) -> np.ndarray:
    rotated = reconstruct_rotated(indices, scales, variant.weight_bits, variant)
    if variant.residual_bits is not None and residual_indices is not None and residual_scales is not None:
        rotated += reconstruct_rotated(residual_indices[group_index], residual_scales[group_index], variant.residual_bits, variant)
    return hadamard_unrotate_groups(rotated.reshape(1, -1), np.array([seed], dtype=np.uint64))[0]


def _probe_activation(width: int) -> np.ndarray:
    x = np.linspace(-0.85, 0.95, width, dtype=np.float32)
    wiggle = 0.125 * np.sin(np.arange(width, dtype=np.float32) * 0.37)
    return x + wiggle.astype(np.float32)


def _q8_0_dequantize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size % 32:
        raise ValueError("q8_0 reference expects multiples of 32")
    out = np.empty_like(values)
    for offset in range(0, values.size, 32):
        block = values[offset : offset + 32]
        d = np.float16(float(np.max(np.abs(block))) / 127.0).astype(np.float32)
        if float(d) == 0.0:
            out[offset : offset + 32] = 0.0
            continue
        q = np.sign(block / d) * np.floor(np.abs(block / d) + 0.5)
        q = np.clip(q, -128, 127).astype(np.int8)
        out[offset : offset + 32] = q.astype(np.float32) * d
    return out


def write_runtime_fixtures(packed_dir: str | Path, output_dir: str | Path, *, max_per_variant: int = 1) -> dict[str, Any]:
    pack = OpenTQPack(packed_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    fixtures = []
    for index, (row, part, part_index) in enumerate(_first_quant_sections(pack, max_per_variant), start=1):
        group = _group_payload(pack, row, part, part_index)
        stem = f"{index:03d}-{group['variant']}-{Path(row['file']).stem}"
        block_path = output / f"{stem}.block.bin"
        expected_path = output / f"{stem}.expected.f32"
        activation_path = output / f"{stem}.activation.f32"
        expected_dot_path = output / f"{stem}.expected-dot.f32"
        block_path.write_bytes(group["block"])
        group["expected"].astype("<f4").tofile(expected_path)
        group["activation"].astype("<f4").tofile(activation_path)
        group["expected_dot"].astype("<f4").tofile(expected_dot_path)
        fixtures.append(
            {
                "name": row["name"],
                "tensor_file": row["file"],
                "variant": group["variant"],
                "seed": group["seed"],
                "block": str(block_path),
                "expected": str(expected_path),
                "activation": str(activation_path),
                "expected_dot": str(expected_dot_path),
                "block_bytes": len(group["block"]),
            }
        )

    payload = {
        "schema": "opentq.runtime_fixtures.v1",
        "packed_dir": str(packed_dir),
        "release_slug": pack.manifest.get("release_slug"),
        "model_id": pack.manifest.get("model_id"),
        "fixtures": fixtures,
    }
    _json_dump(output / "runtime-fixtures.json", payload)
    return payload


def run_runtime_probe(options: RuntimeProbeOptions) -> dict[str, Any]:
    started = time.time()
    audit = audit_packed_runtime(
        PackAuditOptions(
            options.packed_dir,
            max_tensors=options.audit_max_tensors,
            dequantize_samples=options.audit_dequantize_samples,
        )
    )
    fixtures = write_runtime_fixtures(options.packed_dir, options.fixtures_output, max_per_variant=options.max_fixtures_per_variant)
    probe_rows = []
    probe_pass = True
    if options.probe_binary is None:
        probe_pass = False
        probe_status = "missing-probe-binary"
    elif not options.probe_binary.exists():
        probe_pass = False
        probe_status = "probe-binary-not-found"
    else:
        probe_status = "executed"
        for fixture in fixtures["fixtures"]:
            for mode, args in (
                ("dequant", [fixture["variant"], fixture["block"], fixture["expected"]]),
                ("dot", [fixture["variant"], fixture["block"], fixture["activation"], fixture["expected_dot"]]),
            ):
                command = [str(options.probe_binary), mode, *args]
                try:
                    completed = subprocess.run(command, text=True, capture_output=True, timeout=options.timeout_seconds, check=False)
                    row = {
                        "mode": mode,
                        "variant": fixture["variant"],
                        "name": fixture["name"],
                        "returncode": completed.returncode,
                        "timed_out": False,
                        "stdout": completed.stdout.strip(),
                        "stderr": completed.stderr.strip(),
                    }
                    ok = completed.returncode == 0
                except subprocess.TimeoutExpired as exc:
                    stdout = exc.stdout or ""
                    stderr = exc.stderr or ""
                    if isinstance(stdout, bytes):
                        stdout = stdout.decode(errors="replace")
                    if isinstance(stderr, bytes):
                        stderr = stderr.decode(errors="replace")
                    row = {
                        "mode": mode,
                        "variant": fixture["variant"],
                        "name": fixture["name"],
                        "returncode": None,
                        "timed_out": True,
                        "stdout": stdout.strip(),
                        "stderr": stderr.strip(),
                    }
                    ok = False
                probe_pass = probe_pass and ok
                probe_rows.append(row)

    payload = {
        "schema": "opentq.runtime_probe.v1",
        "packed_dir": str(options.packed_dir),
        "fixtures_output": str(options.fixtures_output),
        "probe_binary": None if options.probe_binary is None else str(options.probe_binary),
        "probe_status": probe_status,
        "audit": audit,
        "fixtures": fixtures,
        "probe_rows": probe_rows,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    payload["overall_pass"] = bool(audit["overall_pass"] and probe_pass)
    if options.output is not None:
        _json_dump(options.output, payload)
    return payload
