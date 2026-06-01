"""Heuristic document quality filters for Urdu corpus construction."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

try:
    from .normalize_urdu import normalize_urdu, text_stats
except ImportError:  # pragma: no cover - direct script execution
    from normalize_urdu import normalize_urdu, text_stats

SYMBOL_RE = re.compile(r"[^\w\s\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF.,،؛؟!۔:;?'\"()\[\]{}%/-]")
REPEATED_CHAR_RE = re.compile(r"(.)\1{4,}")
URLISH_RE = re.compile(r"https?://|www\.|\.com|\.pk|click here|subscribe|login", re.I)


@dataclass(frozen=True)
class QualityMetrics:
    char_count: int
    urdu_script_ratio: float
    repeated_char_ratio: float
    symbol_ratio: float
    boilerplate_hits: int
    quality_score: float


def quality_metrics(text: str) -> QualityMetrics:
    normalized = normalize_urdu(text)
    stats = text_stats(normalized)
    char_count = len(normalized)
    repeated_chars = sum(len(match.group(0)) for match in REPEATED_CHAR_RE.finditer(normalized))
    symbols = len(SYMBOL_RE.findall(normalized))
    boilerplate_hits = len(URLISH_RE.findall(normalized))

    repeated_char_ratio = repeated_chars / max(1, char_count)
    symbol_ratio = symbols / max(1, char_count)
    quality_score = (
        0.45 * stats.urdu_script_ratio
        + 0.40 * min(1.0, char_count / 1200)
        - 0.10 * min(1.0, repeated_char_ratio * 5)
        - 0.05 * min(1.0, symbol_ratio * 5)
        - 0.05 * min(1.0, boilerplate_hits / 5)
    )
    return QualityMetrics(
        char_count=char_count,
        urdu_script_ratio=stats.urdu_script_ratio,
        repeated_char_ratio=repeated_char_ratio,
        symbol_ratio=symbol_ratio,
        boilerplate_hits=boilerplate_hits,
        quality_score=max(0.0, min(1.0, quality_score)),
    )


def passes_quality(
    metrics: QualityMetrics,
    *,
    min_chars: int,
    min_script_ratio: float,
    max_repeated_char_ratio: float,
    max_symbol_ratio: float,
) -> bool:
    return (
        metrics.char_count >= min_chars
        and metrics.urdu_script_ratio >= min_script_ratio
        and metrics.repeated_char_ratio <= max_repeated_char_ratio
        and metrics.symbol_ratio <= max_symbol_ratio
    )


def filter_jsonl(input_path: Path, output_path: Path, *, text_key: str, args: argparse.Namespace) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    seen = 0
    kept = 0
    with input_path.open("r", encoding="utf-8", errors="replace") as fin:
        with output_path.open("w", encoding="utf-8") as fout:
            for line in fin:
                if not line.strip():
                    continue
                seen += 1
                record = json.loads(line)
                text = str(record.get(text_key, ""))
                metrics = quality_metrics(text)
                record["quality_metrics"] = asdict(metrics)
                record["quality_score"] = metrics.quality_score
                if not passes_quality(
                    metrics,
                    min_chars=args.min_chars,
                    min_script_ratio=args.min_script_ratio,
                    max_repeated_char_ratio=args.max_repeated_char_ratio,
                    max_symbol_ratio=args.max_symbol_ratio,
                ):
                    continue
                kept += 1
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"seen={seen} kept={kept} dropped={seen - kept}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply heuristic Urdu quality filters to JSONL.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--text-key", default="normalized_text")
    parser.add_argument("--min-chars", type=int, default=200)
    parser.add_argument("--min-script-ratio", type=float, default=0.65)
    parser.add_argument("--max-repeated-char-ratio", type=float, default=0.20)
    parser.add_argument("--max-symbol-ratio", type=float, default=0.15)
    args = parser.parse_args()
    filter_jsonl(args.input, args.output, text_key=args.text_key, args=args)


if __name__ == "__main__":
    main()
