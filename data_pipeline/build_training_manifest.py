"""Build a training manifest from tokenized shard metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_tokenized_metadata(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if "shards" not in metadata:
        raise ValueError(f"{path} does not look like tokenized metadata")
    return metadata


def source_from_metadata_path(path: Path) -> str:
    return path.stem


def bucket_for_source(source: str) -> str:
    if source == "makhzan_urdu":
        return "urdu_literature"
    if source == "fineweb2_urd_arab":
        return "urdu_web"
    return "unknown"


def build_manifest(metadata_paths: list[Path], output_path: Path) -> None:
    sources = []
    total_tokens = 0
    total_docs = 0

    for metadata_path in metadata_paths:
        metadata = load_tokenized_metadata(metadata_path)
        source = source_from_metadata_path(metadata_path)
        shards = metadata["shards"]
        source_tokens = sum(int(shard["tokens"]) for shard in shards)
        source_docs = int(metadata.get("total_docs", 0))
        total_tokens += source_tokens
        total_docs += source_docs
        sources.append(
            {
                "source": source,
                "bucket": bucket_for_source(source),
                "tokens": source_tokens,
                "docs": source_docs,
                "dtype": metadata["dtype"],
                "vocab_size": metadata["vocab_size"],
                "tokenizer_model": metadata["tokenizer_model"],
                "shards": shards,
            }
        )

    manifest = {
        "total_tokens": total_tokens,
        "total_docs": total_docs,
        "sources": sources,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build training manifest from tokenized metadata JSON files.")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("metadata", nargs="+", type=Path)
    args = parser.parse_args()
    build_manifest(args.metadata, args.output)


if __name__ == "__main__":
    main()

