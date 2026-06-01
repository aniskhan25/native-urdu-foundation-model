"""Fast script-level Urdu language filter.

This is a first-stage filter. It should be paired with a model-based Urdu vs.
Arabic/Persian classifier before full pretraining data is frozen.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .normalize_urdu import normalize_urdu, text_stats
except ImportError:  # pragma: no cover - direct script execution
    from normalize_urdu import normalize_urdu, text_stats


def keep_document(text: str, *, min_chars: int, min_script_ratio: float) -> bool:
    normalized = normalize_urdu(text)
    stats = text_stats(normalized)
    return len(normalized) >= min_chars and stats.urdu_script_ratio >= min_script_ratio


def filter_jsonl(
    input_path: Path,
    output_path: Path,
    *,
    text_key: str,
    min_chars: int,
    min_script_ratio: float,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    seen = 0
    with input_path.open("r", encoding="utf-8", errors="replace") as fin:
        with output_path.open("w", encoding="utf-8") as fout:
            for line in fin:
                if not line.strip():
                    continue
                seen += 1
                record = json.loads(line)
                text = str(record.get(text_key, ""))
                normalized = normalize_urdu(text)
                stats = text_stats(normalized)
                record["normalized_text"] = normalized
                record["language_score_ur"] = stats.urdu_script_ratio
                if len(normalized) < min_chars or stats.urdu_script_ratio < min_script_ratio:
                    continue
                kept += 1
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"seen={seen} kept={kept} dropped={seen - kept}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter JSONL by Urdu script ratio.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--text-key", default="raw_text")
    parser.add_argument("--min-chars", type=int, default=200)
    parser.add_argument("--min-script-ratio", type=float, default=0.65)
    args = parser.parse_args()

    filter_jsonl(
        args.input,
        args.output,
        text_key=args.text_key,
        min_chars=args.min_chars,
        min_script_ratio=args.min_script_ratio,
    )


if __name__ == "__main__":
    main()
