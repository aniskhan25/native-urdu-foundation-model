"""Simple character n-gram contamination scanner.

This is intended for small held-out eval sets. For full-corpus scanning, replace
the in-memory sets with a MinHash/LSH index or a distributed implementation.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def char_ngrams(text: str, n: int) -> set[str]:
    compact = " ".join(text.split())
    if len(compact) < n:
        return {compact} if compact else set()
    return {compact[index : index + n] for index in range(len(compact) - n + 1)}


def max_jaccard(eval_text: str, train_texts: list[str], *, n: int) -> float:
    eval_grams = char_ngrams(eval_text, n)
    if not eval_grams:
        return 0.0
    best = 0.0
    for train_text in train_texts:
        train_grams = char_ngrams(train_text, n)
        union = len(eval_grams | train_grams)
        if union == 0:
            continue
        best = max(best, len(eval_grams & train_grams) / union)
    return best


def read_lines(path: Path, limit: int | None) -> list[str]:
    values: list[str] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if line:
                values.append(line)
            if limit is not None and len(values) >= limit:
                break
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description="Check eval text overlap against training text sample.")
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--eval", required=True, type=Path)
    parser.add_argument("--ngram", type=int, default=13)
    parser.add_argument("--threshold", type=float, default=0.20)
    parser.add_argument("--train-limit", type=int, default=100000)
    args = parser.parse_args()

    train_texts = read_lines(args.train, args.train_limit)
    eval_texts = read_lines(args.eval, None)
    for index, text in enumerate(eval_texts):
        score = max_jaccard(text, train_texts, n=args.ngram)
        if score >= args.threshold:
            print(f"FLAG eval_line={index + 1} max_jaccard={score:.4f}")


if __name__ == "__main__":
    main()

