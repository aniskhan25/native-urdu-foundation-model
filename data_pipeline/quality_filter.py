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
URLISH_RE = re.compile(r"https?://|www\.|[A-Za-z0-9-]+\.(com|org|net|pk|edu|gov)\b", re.I)
BOILERPLATE_RE = re.compile(
    r"کلک کریں|بٹن|لنک|عنوان پر جائیں|تلاش کریں|دستیاب ہے|لاگ ان|سائن اپ|"
    r"click here|subscribe|login|sign up|download|button|link",
    re.I,
)


@dataclass(frozen=True)
class QualityMetrics:
    char_count: int
    urdu_script_ratio: float
    repeated_char_ratio: float
    repeated_4gram_ratio: float
    repeated_6gram_ratio: float
    longest_repeated_ngram: int
    symbol_ratio: float
    url_hits: int
    boilerplate_hits: int
    quality_score: float


def word_tokens(text: str) -> list[str]:
    return re.findall(r"\S+", text)


def repeated_ngram_ratio(words: list[str], n: int) -> float:
    if len(words) < n:
        return 0.0
    ngrams = [tuple(words[index : index + n]) for index in range(len(words) - n + 1)]
    counts: dict[tuple[str, ...], int] = {}
    for ngram in ngrams:
        counts[ngram] = counts.get(ngram, 0) + 1
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    return repeated / max(1, len(ngrams))


def longest_repeated_ngram(words: list[str], max_n: int = 8) -> int:
    longest = 0
    for n in range(2, max_n + 1):
        counts: dict[tuple[str, ...], int] = {}
        for index in range(max(0, len(words) - n + 1)):
            ngram = tuple(words[index : index + n])
            counts[ngram] = counts.get(ngram, 0) + 1
        if any(count > 1 for count in counts.values()):
            longest = n
    return longest


def quality_metrics(text: str) -> QualityMetrics:
    normalized = normalize_urdu(text)
    stats = text_stats(normalized)
    char_count = len(normalized)
    words = word_tokens(normalized)
    repeated_chars = sum(len(match.group(0)) for match in REPEATED_CHAR_RE.finditer(normalized))
    symbols = len(SYMBOL_RE.findall(normalized))
    repeated_4gram = repeated_ngram_ratio(words, 4)
    repeated_6gram = repeated_ngram_ratio(words, 6)
    longest_repeat = longest_repeated_ngram(words)
    url_hits = len(URLISH_RE.findall(normalized))
    boilerplate_hits = len(BOILERPLATE_RE.findall(normalized))

    repeated_char_ratio = repeated_chars / max(1, char_count)
    symbol_ratio = symbols / max(1, char_count)
    quality_score = (
        0.45 * stats.urdu_script_ratio
        + 0.40 * min(1.0, char_count / 1200)
        - 0.10 * min(1.0, repeated_char_ratio * 5)
        - 0.10 * min(1.0, repeated_4gram * 8)
        - 0.05 * min(1.0, symbol_ratio * 5)
        - 0.05 * min(1.0, url_hits / 3)
        - 0.05 * min(1.0, boilerplate_hits / 3)
    )
    return QualityMetrics(
        char_count=char_count,
        urdu_script_ratio=stats.urdu_script_ratio,
        repeated_char_ratio=repeated_char_ratio,
        repeated_4gram_ratio=repeated_4gram,
        repeated_6gram_ratio=repeated_6gram,
        longest_repeated_ngram=longest_repeat,
        symbol_ratio=symbol_ratio,
        url_hits=url_hits,
        boilerplate_hits=boilerplate_hits,
        quality_score=max(0.0, min(1.0, quality_score)),
    )


def passes_quality(
    metrics: QualityMetrics,
    *,
    min_chars: int,
    min_script_ratio: float,
    max_repeated_char_ratio: float,
    max_repeated_4gram_ratio: float,
    max_repeated_6gram_ratio: float,
    max_longest_repeated_ngram: int,
    max_symbol_ratio: float,
    max_url_hits: int,
    max_boilerplate_hits: int,
) -> bool:
    return (
        metrics.char_count >= min_chars
        and metrics.urdu_script_ratio >= min_script_ratio
        and metrics.repeated_char_ratio <= max_repeated_char_ratio
        and metrics.repeated_4gram_ratio <= max_repeated_4gram_ratio
        and metrics.repeated_6gram_ratio <= max_repeated_6gram_ratio
        and metrics.longest_repeated_ngram <= max_longest_repeated_ngram
        and metrics.symbol_ratio <= max_symbol_ratio
        and metrics.url_hits <= max_url_hits
        and metrics.boilerplate_hits <= max_boilerplate_hits
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
                    max_repeated_4gram_ratio=args.max_repeated_4gram_ratio,
                    max_repeated_6gram_ratio=args.max_repeated_6gram_ratio,
                    max_longest_repeated_ngram=args.max_longest_repeated_ngram,
                    max_symbol_ratio=args.max_symbol_ratio,
                    max_url_hits=args.max_url_hits,
                    max_boilerplate_hits=args.max_boilerplate_hits,
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
    parser.add_argument("--max-repeated-4gram-ratio", type=float, default=0.08)
    parser.add_argument("--max-repeated-6gram-ratio", type=float, default=0.04)
    parser.add_argument("--max-longest-repeated-ngram", type=int, default=8)
    parser.add_argument("--max-symbol-ratio", type=float, default=0.15)
    parser.add_argument("--max-url-hits", type=int, default=3)
    parser.add_argument("--max-boilerplate-hits", type=int, default=2)
    args = parser.parse_args()
    filter_jsonl(args.input, args.output, text_key=args.text_key, args=args)


if __name__ == "__main__":
    main()
