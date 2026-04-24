from __future__ import annotations

import json
from typing import Any
from urllib.parse import quote
from urllib.request import urlopen


def hf_resolve_url(model_id: str, filename: str, revision: str = "main") -> str:
    quoted_model = quote(model_id, safe="/")
    quoted_file = quote(filename, safe="/")
    return f"https://huggingface.co/{quoted_model}/resolve/{revision}/{quoted_file}"


def fetch_json(url: str) -> dict[str, Any]:
    with urlopen(url, timeout=30) as response:
        return json.load(response)


def fetch_safetensors_index(model_id: str, filename: str = "model.safetensors.index.json") -> dict[str, Any]:
    return fetch_json(hf_resolve_url(model_id, filename))


def base_weight_size_gib(index_data: dict[str, Any]) -> float:
    total_size = float(index_data.get("metadata", {}).get("total_size", 0.0))
    return total_size / (1024.0**3)

