from __future__ import annotations

import importlib.util
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Iterable

import numpy as np
from huggingface_hub import snapshot_download

from .bitpack import pack_bits, unpack_bits
from .runtime import OpenTQPack
from .run import tensor_seed
from .variants import QuantVariant, get_variant


OPENTQ_GGML_TYPES = {
    "TQ3_SB4": 42,
    "TQ4_SB2": 43,
    "TQ4_SB4": 44,
    "TQ4R2": 45,
    "TQ4R4": 46,
}

OPENTQ_TYPE_NAMES = {value: key for key, value in OPENTQ_GGML_TYPES.items()}


@dataclass(frozen=True)
class GGUFExportOptions:
    packed_dir: Path
    output: Path
    llama_cpp_dir: Path
    text_only: bool = True
    max_tensors: int | None = None


@dataclass(frozen=True)
class TensorExport:
    source: dict[str, Any]
    gguf_name: str
    gguf_shape: tuple[int, ...]
    ggml_type: int
    nbytes: int
    transform: str


def opentq_group_type_size(variant: QuantVariant) -> int:
    scale_bytes = variant.sub_block_scales * 4 * 2
    index_bytes = variant.group_size * variant.weight_bits // 8
    residual_index_bytes = 0 if variant.residual_bits is None else variant.group_size * variant.residual_bits // 8
    residual_scale_bytes = 0 if variant.residual_bits is None else variant.sub_block_scales * 4 * 2
    return 4 + scale_bytes + index_bytes + residual_scale_bytes + residual_index_bytes


def load_converter_module(llama_cpp_dir: Path):
    gguf_py = llama_cpp_dir / "gguf-py"
    if str(gguf_py) not in sys.path:
        sys.path.insert(0, str(gguf_py))
    if str(llama_cpp_dir) not in sys.path:
        sys.path.insert(0, str(llama_cpp_dir))
    module_path = llama_cpp_dir / "convert_hf_to_gguf.py"
    spec = importlib.util.spec_from_file_location("opentq_llama_convert_hf_to_gguf", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import llama.cpp converter: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def snapshot_metadata(model_id: str) -> Path:
    return Path(
        snapshot_download(
            model_id,
            allow_patterns=[
                "config.json",
                "generation_config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "added_tokens.json",
                "merges.txt",
                "vocab.json",
                "README.md",
            ],
        )
    )


def build_converter_model(pack: OpenTQPack, output: Path, llama_cpp_dir: Path):
    converter = load_converter_module(llama_cpp_dir)
    model_id = pack.manifest["model_id"]
    snapshot = snapshot_metadata(model_id)
    config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    arch = config.get("architectures", [None])[0]
    if arch is None:
        raise ValueError("missing architecture in config.json")
    model_cls = converter.ModelBase.from_model_architecture(arch, converter.ModelType.TEXT)
    return model_cls(
        snapshot,
        converter.gguf.LlamaFileType.MOSTLY_F16,
        output,
        use_temp_file=False,
        remote_hf_model_id=model_id,
        model_name=pack.manifest["release_slug"],
    )


def strip_text_name(name: str) -> str | None:
    if name.startswith("model.visual.") or name.startswith("visual."):
        return None
    if name.startswith("mtp."):
        return None
    return name.replace("language_model.", "")


def first_block_id(name: str) -> int | None:
    for part in name.split("."):
        if part.isdecimal():
            return int(part)
    return None


def reorder_v_indices(num_k_heads: int, num_v_heads: int, head_dim: int) -> np.ndarray:
    num_v_per_k = num_v_heads // num_k_heads
    values = np.arange(num_v_heads * head_dim, dtype=np.int64).reshape(num_k_heads, num_v_per_k, head_dim)
    return values.transpose(1, 0, 2).reshape(-1)


def qwen35_transform_kind(name: str) -> str:
    if ".linear_attn.in_proj_qkv.weight" in name:
        return "linear_attn_qkv_rows"
    if ".linear_attn.in_proj_z.weight" in name:
        return "linear_attn_v_rows"
    if ".linear_attn.in_proj_a.weight" in name or ".linear_attn.in_proj_b.weight" in name:
        return "linear_attn_head_rows"
    if ".linear_attn.out_proj.weight" in name:
        return "linear_attn_out_cols"
    if ".linear_attn.conv1d.weight" in name:
        return "linear_attn_conv_f32"
    if ".linear_attn.A_log" in name:
        return "linear_attn_a_log_f32"
    if ".linear_attn.dt_bias" in name:
        return "linear_attn_dt_bias_f32"
    if name.endswith("norm.weight") and not name.endswith("linear_attn.norm.weight"):
        return "norm_plus_one_f32"
    return "identity"


def mapped_text_name(model: Any, hf_name: str, transform: str) -> str:
    name = hf_name
    if transform == "linear_attn_dt_bias_f32":
        name = name.rpartition(".dt_bias")[0] + ".dt_proj.bias"
    return model.map_tensor_name(name)


def transformed_shape(shape: tuple[int, ...], transform: str) -> tuple[int, ...]:
    if transform == "linear_attn_conv_f32" and len(shape) == 3 and shape[1] == 1:
        return (shape[0], shape[2])
    return shape


def tensor_is_quant_export(row: dict[str, Any], transform: str) -> bool:
    if row["mode"] != "quantize":
        return False
    return transform not in {"linear_attn_conv_f32"}


def build_tensor_exports(pack: OpenTQPack, model: Any, max_tensors: int | None = None) -> list[TensorExport]:
    exports: list[TensorExport] = []
    for row in pack.manifest["tensors"]:
        hf_name = strip_text_name(row["name"])
        if hf_name is None:
            continue
        transform = qwen35_transform_kind(hf_name)
        gguf_name = mapped_text_name(model, hf_name, transform)
        shape = transformed_shape(tuple(row["shape"]), transform)
        if tensor_is_quant_export(row, transform):
            variant = get_variant(row["variant_name"])
            ggml_type = OPENTQ_GGML_TYPES[variant.name]
            groups = math.ceil(int(np.prod(shape, dtype=np.int64)) / variant.group_size)
            if len(shape) >= 2 and int(np.prod(shape[1:], dtype=np.int64)) % variant.group_size != 0:
                raise ValueError(f"cannot export quantized tensor with non-aligned row width: {row['name']} {shape}")
            nbytes = groups * opentq_group_type_size(variant)
        else:
            ggml_type = 0
            nbytes = int(np.prod(shape, dtype=np.int64)) * 4
        exports.append(TensorExport(row, gguf_name, shape, ggml_type, nbytes, transform))
        if max_tensors is not None and len(exports) >= max_tensors:
            break
    return exports


def pack_group_record(seed: int, indices: np.ndarray, scales: np.ndarray, variant: QuantVariant, residual_indices: np.ndarray | None = None, residual_scales: np.ndarray | None = None) -> bytes:
    chunks = [
        np.array([seed], dtype="<u4").tobytes(),
        np.asarray(scales, dtype="<f2").reshape(-1).tobytes(),
        pack_bits(np.asarray(indices, dtype=np.uint8).reshape(-1), variant.weight_bits),
    ]
    if variant.residual_bits is not None:
        if residual_indices is None or residual_scales is None:
            raise ValueError(f"variant {variant.name} requires residual payload")
        chunks.append(np.asarray(residual_scales, dtype="<f2").reshape(-1).tobytes())
        chunks.append(pack_bits(np.asarray(residual_indices, dtype=np.uint8).reshape(-1), variant.residual_bits))
    return b"".join(chunks)


def iter_part_group_records(pack: OpenTQPack, row: dict[str, Any], part: dict[str, Any], variant: QuantVariant, part_index: int) -> Iterable[bytes]:
    indices = unpack_bits(pack.read_section(row, part["indices"]), variant.weight_bits, int(part["index_count"])).reshape(-1, variant.group_size)
    scales = np.frombuffer(pack.read_section(row, part["scales"]), dtype=np.float16).reshape(-1, variant.sub_block_scales)
    scales = scales.reshape(indices.shape[0], -1)
    residual_indices = None
    residual_scales = None
    if variant.residual_bits is not None:
        residual_indices = unpack_bits(pack.read_section(row, part["residual_indices"]), variant.residual_bits, int(part["residual_index_count"])).reshape(-1, variant.group_size)
        residual_scales = np.frombuffer(pack.read_section(row, part["residual_scales"]), dtype=np.float16).reshape(indices.shape[0], -1)
    seed_base = int(row.get("seed", tensor_seed(pack.manifest["release_slug"], row["name"]))) + part_index * 104729
    for group_index in range(indices.shape[0]):
        yield pack_group_record(
            int(seed_base + group_index * variant.group_size),
            indices[group_index],
            scales[group_index],
            variant,
            None if residual_indices is None else residual_indices[group_index],
            None if residual_scales is None else residual_scales[group_index],
        )


def part_group_records(pack: OpenTQPack, row: dict[str, Any], part: dict[str, Any], variant: QuantVariant, part_index: int) -> list[bytes]:
    return list(iter_part_group_records(pack, row, part, variant, part_index))


def row_group_matrix(pack: OpenTQPack, row: dict[str, Any], variant: QuantVariant) -> list[list[bytes]]:
    shape = tuple(row["shape"])
    row_width = int(np.prod(shape[1:], dtype=np.int64))
    if row_width % variant.group_size != 0:
        raise ValueError(f"row width is not OpenTQ aligned for {row['name']}: {row_width}")
    groups_per_row = row_width // variant.group_size
    rows: list[list[bytes]] = []
    for part_index, part in enumerate(row["sections"]):
        records = part_group_records(pack, row, part, variant, part_index)
        if len(records) % groups_per_row != 0:
            raise ValueError(f"group records do not divide into rows for {row['name']}")
        rows.extend([records[index : index + groups_per_row] for index in range(0, len(records), groups_per_row)])
    return rows


def write_quant_stream(pack: OpenTQPack, export: TensorExport, handle: BinaryIO) -> None:
    row = export.source
    variant = get_variant(row["variant_name"])
    transform = export.transform
    if transform == "identity":
        for part_index, part in enumerate(row["sections"]):
            for record in iter_part_group_records(pack, row, part, variant, part_index):
                handle.write(record)
        return

    matrix = row_group_matrix(pack, row, variant)
    if transform == "linear_attn_qkv_rows":
        q_dim = 128 * 16
        k_dim = 128 * 16
        v_perm = reorder_v_indices(16, 48, 128)
        order = list(range(q_dim + k_dim)) + [q_dim + k_dim + int(idx) for idx in v_perm]
        matrix = [matrix[index] for index in order]
    elif transform == "linear_attn_v_rows":
        matrix = [matrix[int(index)] for index in reorder_v_indices(16, 48, 128)]
    elif transform == "linear_attn_head_rows":
        matrix = [matrix[int(index)] for index in reorder_v_indices(16, 48, 1)]
    elif transform == "linear_attn_out_cols":
        perm = reorder_v_indices(16, 48, 128)
        group_perm = (perm.reshape(-1, 128)[:, 0] // 128).astype(np.int64)
        matrix = [[row_groups[int(index)] for index in group_perm] for row_groups in matrix]
    else:
        raise ValueError(f"unsupported quant transform: {transform}")

    for row_groups in matrix:
        for record in row_groups:
            handle.write(record)


def transformed_f32_tensor(pack: OpenTQPack, export: TensorExport) -> np.ndarray:
    data = pack.dequantize_tensor(export.source["name"], dtype=np.float32)
    transform = export.transform
    if transform == "linear_attn_conv_f32":
        data = np.squeeze(data, axis=1)
        qk_channels = 128 * 16 * 2
        order = list(range(qk_channels)) + [qk_channels + int(idx) for idx in reorder_v_indices(16, 48, 128)]
        data = data[order]
    elif transform == "linear_attn_a_log_f32":
        data = -np.exp(data)
        data = data[reorder_v_indices(16, 48, 1)]
    elif transform == "linear_attn_dt_bias_f32":
        data = data[reorder_v_indices(16, 48, 1)]
    elif transform == "norm_plus_one_f32":
        data = data + 1.0
    return np.ascontiguousarray(data.astype(np.float32, copy=False))


def write_padding(handle: BinaryIO, n: int, alignment: int = 32) -> None:
    pad = ((n + alignment - 1) // alignment) * alignment - n
    if pad:
        handle.write(b"\x00" * pad)


def log_export_event(**payload: Any) -> None:
    print(json.dumps(payload, sort_keys=True), file=sys.stderr, flush=True)


def export_gguf(options: GGUFExportOptions) -> dict[str, Any]:
    if not options.text_only:
        raise NotImplementedError("GGUF export is currently text-only; vision tensors are intentionally skipped")
    pack = OpenTQPack(options.packed_dir)
    model = build_converter_model(pack, options.output, options.llama_cpp_dir)
    writer = model.gguf_writer
    exports = build_tensor_exports(pack, model, options.max_tensors)
    log_export_event(
        event="export_start",
        release_slug=pack.manifest["release_slug"],
        tensor_count=len(exports),
        output=str(options.output),
    )

    for item in exports:
        writer.add_tensor_info(
            item.gguf_name,
            item.gguf_shape,
            np.dtype("int8"),
            item.nbytes,
            raw_dtype=item.ggml_type,
        )

    model.prepare_metadata(vocab_only=False)
    writer.add_string("opentq.schema", "opentq.gguf.v1")
    writer.add_string("opentq.release_slug", pack.manifest["release_slug"])
    writer.add_string("opentq.source_pack_schema", pack.manifest["schema"])
    writer.add_string("opentq.required_runtime", "llama.cpp-opentq")

    writer.write_header_to_file(path=options.output)
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    assert writer.fout is not None
    handle = writer.fout[0]
    write_padding(handle, handle.tell())
    for index, item in enumerate(exports, start=1):
        log_export_event(
            event="tensor_start",
            index=index,
            total=len(exports),
            name=item.gguf_name,
            transform=item.transform,
            ggml_type=OPENTQ_TYPE_NAMES.get(item.ggml_type, "F32"),
            bytes=item.nbytes,
        )
        if item.ggml_type in OPENTQ_TYPE_NAMES:
            write_quant_stream(pack, item, handle)
        else:
            transformed_f32_tensor(pack, item).tofile(handle)
        write_padding(handle, item.nbytes)
        log_export_event(
            event="tensor_done",
            index=index,
            total=len(exports),
            name=item.gguf_name,
            file_offset=handle.tell(),
        )
    writer.close()
    log_export_event(
        event="export_done",
        release_slug=pack.manifest["release_slug"],
        bytes=options.output.stat().st_size,
        output=str(options.output),
    )

    return {
        "schema": "opentq.gguf_export.v1",
        "output": str(options.output),
        "release_slug": pack.manifest["release_slug"],
        "model_id": pack.manifest["model_id"],
        "tensor_count": len(exports),
        "bytes": options.output.stat().st_size,
        "custom_types": sorted({OPENTQ_TYPE_NAMES[item.ggml_type] for item in exports if item.ggml_type in OPENTQ_TYPE_NAMES}),
    }
