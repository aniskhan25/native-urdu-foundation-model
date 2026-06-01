"""Summarize canonical corpus JSONL shards."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def summarize_file(path: Path) -> dict[str, Any]:
    docs = 0
    chars = 0
    words = 0
    bucket_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    script_scores: list[float] = []
    quality_scores: list[float] = []

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            text = str(record.get("normalized_text", ""))
            docs += 1
            chars += len(text)
            words += len(text.split())
            bucket_counts[str(record.get("source_bucket", "unknown"))] += 1
            source_counts[str(record.get("source", path.stem))] += 1
            if record.get("language_score_ur") is not None:
                script_scores.append(float(record["language_score_ur"]))
            if record.get("quality_score") is not None:
                quality_scores.append(float(record["quality_score"]))

    return {
        "path": str(path),
        "docs": docs,
        "normalized_chars": chars,
        "whitespace_words": words,
        "avg_chars_per_doc": chars / max(1, docs),
        "avg_words_per_doc": words / max(1, docs),
        "avg_urdu_script_ratio": sum(script_scores) / max(1, len(script_scores)),
        "min_urdu_script_ratio": min(script_scores) if script_scores else None,
        "avg_quality_score": sum(quality_scores) / max(1, len(quality_scores)),
        "bucket_counts": dict(bucket_counts),
        "source_counts": dict(source_counts),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize compiled corpus JSONL shards.")
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()

    file_summaries = [summarize_file(path) for path in args.paths]
    totals: dict[str, Any] = {
        "files": len(file_summaries),
        "docs": sum(item["docs"] for item in file_summaries),
        "normalized_chars": sum(item["normalized_chars"] for item in file_summaries),
        "whitespace_words": sum(item["whitespace_words"] for item in file_summaries),
        "bucket_counts": defaultdict(int),
        "source_counts": defaultdict(int),
        "files_detail": file_summaries,
    }
    for item in file_summaries:
        for bucket, count in item["bucket_counts"].items():
            totals["bucket_counts"][bucket] += count
        for source, count in item["source_counts"].items():
            totals["source_counts"][source] += count
    totals["bucket_counts"] = dict(totals["bucket_counts"])
    totals["source_counts"] = dict(totals["source_counts"])
    print(json.dumps(totals, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

