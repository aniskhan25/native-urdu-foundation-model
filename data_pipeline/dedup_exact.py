"""Exact deduplication for normalized text JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

try:
    from .normalize_urdu import normalize_urdu
except ImportError:  # pragma: no cover - direct script execution
    from normalize_urdu import normalize_urdu


def digest_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def dedup_jsonl(input_path: Path, output_path: Path, *, text_key: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    seen_hashes: set[str] = set()
    seen = 0
    kept = 0
    with input_path.open("r", encoding="utf-8", errors="replace") as fin:
        with output_path.open("w", encoding="utf-8") as fout:
            for line in fin:
                if not line.strip():
                    continue
                seen += 1
                record = json.loads(line)
                normalized = normalize_urdu(str(record.get(text_key, "")))
                digest = digest_text(normalized)
                if digest in seen_hashes:
                    continue
                seen_hashes.add(digest)
                record[text_key] = normalized
                record["dedup_hash"] = digest
                kept += 1
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"seen={seen} kept={kept} exact_duplicates={seen - kept}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Remove exact duplicate JSONL documents.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--text-key", default="normalized_text")
    args = parser.parse_args()
    dedup_jsonl(args.input, args.output, text_key=args.text_key)


if __name__ == "__main__":
    main()
