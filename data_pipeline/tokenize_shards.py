"""Tokenize normalized JSONL into fixed-size binary shards.

The output is a simple contiguous uint16/uint32 token stream plus JSON metadata.
Training frameworks can either consume this directly or convert it to their
preferred indexed dataset format.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import sentencepiece as spm


def dtype_for_vocab(vocab_size: int) -> np.dtype:
    return np.uint16 if vocab_size <= 65535 else np.uint32


def tokenize_jsonl(
    input_path: Path,
    output_prefix: Path,
    *,
    model_path: Path,
    text_key: str,
    shard_tokens: int,
) -> None:
    sp = spm.SentencePieceProcessor()
    sp.load(str(model_path))
    dtype = dtype_for_vocab(sp.get_piece_size())

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    buffer: list[int] = []
    shard_index = 0
    total_tokens = 0
    total_docs = 0
    metadata = {
        "tokenizer_model": str(model_path),
        "vocab_size": sp.get_piece_size(),
        "dtype": np.dtype(dtype).name,
        "shard_tokens": shard_tokens,
        "shards": [],
    }

    def flush() -> None:
        nonlocal buffer, shard_index
        if not buffer:
            return
        shard_path = output_prefix.with_name(f"{output_prefix.name}-{shard_index:05d}.bin")
        np.asarray(buffer, dtype=dtype).tofile(shard_path)
        metadata["shards"].append({"path": str(shard_path), "tokens": len(buffer)})
        buffer = []
        shard_index += 1

    with input_path.open("r", encoding="utf-8", errors="replace") as fin:
        for line in fin:
            if not line.strip():
                continue
            record = json.loads(line)
            text = str(record.get(text_key, ""))
            if not text:
                continue
            ids = [sp.bos_id()] + sp.encode(text, out_type=int) + [sp.eos_id()]
            if buffer and len(buffer) + len(ids) > shard_tokens:
                flush()
            buffer.extend(ids)
            total_tokens += len(ids)
            total_docs += 1
            if len(buffer) >= shard_tokens:
                flush()

    flush()
    metadata["total_tokens"] = total_tokens
    metadata["total_docs"] = total_docs
    metadata_path = output_prefix.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"docs={total_docs} tokens={total_tokens} shards={len(metadata['shards'])}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Tokenize normalized JSONL into binary shards.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-prefix", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--text-key", default="normalized_text")
    parser.add_argument("--shard-tokens", type=int, default=134_217_728)
    args = parser.parse_args()
    tokenize_jsonl(
        args.input,
        args.output_prefix,
        model_path=args.model,
        text_key=args.text_key,
        shard_tokens=args.shard_tokens,
    )


if __name__ == "__main__":
    main()
