"""Score generated Urdu samples for common base-model failure modes."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import re
from pathlib import Path
from statistics import mean
from typing import Any


ARABIC_SCRIPT_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]")
LETTER_RE = re.compile(r"[^\W\d_]", re.UNICODE)
URL_RE = re.compile(r"(https?://|www\.|[A-Za-z0-9-]+\.(com|org|net|pk|edu|gov)\b)", re.IGNORECASE)
UI_BOILERPLATE_RE = re.compile(
    r"(کلک کریں|بٹن|لنک|عنوان پر جائیں|تلاش کریں|دستیاب ہے|لاگ ان|سائن اپ|download|click|button|link)",
    re.IGNORECASE,
)
MATH_DISCOUNT_RE = re.compile(r"1200.*15|15.*1200")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def token_words(text: str) -> list[str]:
    return re.findall(r"\S+", text)


def ngram_repetition_ratio(words: list[str], n: int) -> float:
    if len(words) < n:
        return 0.0
    ngrams = [tuple(words[index : index + n]) for index in range(len(words) - n + 1)]
    if not ngrams:
        return 0.0
    counts = Counter(ngrams)
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    return repeated / len(ngrams)


def longest_repeated_ngram(words: list[str], max_n: int = 8) -> int:
    longest = 0
    for n in range(2, max_n + 1):
        counts = Counter(tuple(words[index : index + n]) for index in range(max(0, len(words) - n + 1)))
        if any(count > 1 for count in counts.values()):
            longest = n
    return longest


def urdu_script_ratio(text: str) -> float:
    letters = LETTER_RE.findall(text)
    if not letters:
        return 0.0
    urdu_chars = ARABIC_SCRIPT_RE.findall("".join(letters))
    return len(urdu_chars) / len(letters)


def prompt_copy_ratio(prompt: str, text: str) -> float:
    if not prompt:
        return 0.0
    return min(len(text), len(prompt)) / len(text) if text.startswith(prompt) and text else 0.0


def math_discount_correct(prompt: str, text: str) -> bool | None:
    if not MATH_DISCOUNT_RE.search(prompt):
        return None
    normalized = text.replace(",", "")
    return "1020" in normalized or "۱،۰۲۰" in text or "۱۰۲۰" in text


def score_record(record: dict[str, Any]) -> dict[str, Any]:
    prompt = str(record.get("prompt", ""))
    text = str(record.get("text") or prompt + " " + str(record.get("completion", "")))
    completion = str(record.get("completion", ""))
    words = token_words(completion or text)
    flags = []

    rep4 = ngram_repetition_ratio(words, 4)
    rep6 = ngram_repetition_ratio(words, 6)
    longest_repeat = longest_repeated_ngram(words)
    url_artifacts = len(URL_RE.findall(text))
    boilerplate_hits = len(UI_BOILERPLATE_RE.findall(text))
    script_ratio = urdu_script_ratio(text)
    copy_ratio = prompt_copy_ratio(prompt, text)
    math_ok = math_discount_correct(prompt, text)

    if rep4 > 0.08 or rep6 > 0.04 or longest_repeat >= 6:
        flags.append("repetition")
    if url_artifacts:
        flags.append("url_or_domain")
    if boilerplate_hits:
        flags.append("ui_boilerplate")
    if script_ratio < 0.85:
        flags.append("low_urdu_script_ratio")
    if copy_ratio > 0.35:
        flags.append("prompt_copying")
    if math_ok is False:
        flags.append("math_failed")

    return {
        "prompt": prompt,
        "chars": len(text),
        "completion_words": len(words),
        "urdu_script_ratio": round(script_ratio, 4),
        "prompt_copy_ratio": round(copy_ratio, 4),
        "repetition_4gram": round(rep4, 4),
        "repetition_6gram": round(rep6, 4),
        "longest_repeated_ngram": longest_repeat,
        "url_artifacts": url_artifacts,
        "boilerplate_hits": boilerplate_hits,
        "math_correct": math_ok,
        "flags": flags,
    }


def summarize(scores: list[dict[str, Any]]) -> dict[str, Any]:
    flag_counts = Counter(flag for score in scores for flag in score["flags"])
    math_scores = [score["math_correct"] for score in scores if score["math_correct"] is not None]
    return {
        "samples": len(scores),
        "avg_completion_words": round(mean(score["completion_words"] for score in scores), 2) if scores else 0,
        "avg_urdu_script_ratio": round(mean(score["urdu_script_ratio"] for score in scores), 4) if scores else 0,
        "avg_repetition_4gram": round(mean(score["repetition_4gram"] for score in scores), 4) if scores else 0,
        "flag_counts": dict(flag_counts),
        "math_accuracy": (sum(1 for item in math_scores if item) / len(math_scores)) if math_scores else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Score generated Urdu samples.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--scores-output", type=Path, default=None)
    parser.add_argument("--summary-output", type=Path, default=None)
    args = parser.parse_args()

    records = load_jsonl(args.input)
    scores = [score_record(record) for record in records]
    summary = summarize(scores)

    scores_output = args.scores_output or args.input.with_name(args.input.stem + ".scores.jsonl")
    summary_output = args.summary_output or args.input.with_name(args.input.stem + ".summary.json")

    with scores_output.open("w", encoding="utf-8") as handle:
        for score in scores:
            handle.write(json.dumps(score, ensure_ascii=False) + "\n")
    summary_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote scores to {scores_output}")
    print(f"Wrote summary to {summary_output}")


if __name__ == "__main__":
    main()
