#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi


DEFAULT_ITEMS = [
    {
        "repo_id": "zlaabsi/Qwen3.6-27B-OTQ-DYN-Q3_XL-GGUF",
        "note": "Compact 13.48 GiB / 14.5 GB. Best first pick for 32 GB Apple Silicon.",
    },
    {
        "repo_id": "zlaabsi/Qwen3.6-27B-OTQ-DYN-Q4_XL-GGUF",
        "note": "Balanced 16.82 GiB / 18.1 GB. Quality-balanced profile for 32-48 GB+ Apple Silicon.",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create or update the OpenTQ Qwen3.6 Hugging Face collection.")
    parser.add_argument("--namespace", default="zlaabsi")
    parser.add_argument("--title", default="OpenTQ Qwen3.6 GGUF Releases")
    parser.add_argument(
        "--description",
        default="OpenTQ dynamic-compatible Qwen3.6-27B GGUFs for stock llama.cpp on Apple Silicon.",
        help="Hugging Face currently enforces a short collection description.",
    )
    parser.add_argument("--theme", default="blue")
    parser.add_argument("--output", default="artifacts/hf-gguf-dynamic/collection.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    api = HfApi()
    collection = api.create_collection(
        args.title,
        namespace=args.namespace,
        description=args.description,
        exists_ok=True,
    )
    collection = api.update_collection_metadata(
        collection.slug,
        description=args.description,
        theme=args.theme,
    )
    for item in DEFAULT_ITEMS:
        collection = api.add_collection_item(
            collection.slug,
            item["repo_id"],
            "model",
            note=item["note"],
            exists_ok=True,
        )

    payload = {
        "slug": collection.slug,
        "title": collection.title,
        "description": args.description,
        "items": [
            {
                "repo_id": item.item_id,
                "type": item.item_type,
                "note": getattr(item, "note", None),
            }
            for item in collection.items
        ],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
