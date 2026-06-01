"""Urdu Unicode normalization utilities.

The policy is intentionally explicit and versioned because tokenizer training
and pretraining data must use the same normalization behavior.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, TextIO

NORMALIZATION_VERSION = "custom_urdu_nfkc_then_nfc_v1"

ARABIC_DIACRITICS_RE = re.compile(
    r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]"
)
BIDI_CONTROLS_RE = re.compile(r"[\u200E\u200F\u202A-\u202E\u2066-\u2069]")
WHITESPACE_RE = re.compile(r"\s+")

LETTER_RE = re.compile(r"[^\W\d_]", re.UNICODE)

URDU_CHAR_MAP = {
    # Yeh family
    "\u064A": "\u06CC",  # Arabic Yeh -> Farsi Yeh
    "\u0649": "\u06CC",  # Alef Maksura -> Farsi Yeh
    # Kaf family
    "\u0643": "\u06A9",  # Arabic Kaf -> Keheh
    "\u06AA": "\u06A9",  # Swash Kaf -> Keheh for strict Urdu corpora
    # Heh family
    "\u0647": "\u06C1",  # Arabic Heh -> Heh Goal
    "\u0629": "\u06C1",  # Teh Marbuta -> Heh Goal
    "\u06C0": "\u06C1",  # Heh with Yeh Above -> Heh Goal
    # Alef family, conservative
    "\u0625": "\u0627",  # Alef with Hamza Below -> Alef
    "\u0623": "\u0627",  # Alef with Hamza Above -> Alef
    "\u0671": "\u0627",  # Alef Wasla -> Alef
    # Waw variants
    "\u06C4": "\u0648",
    "\u06C5": "\u0648",
    "\u06C6": "\u0648",
    "\u06C7": "\u0648",
    "\u06C8": "\u0648",
    "\u06C9": "\u0648",
    # Tatweel
    "\u0640": "",
    # Arabic numeric punctuation
    "\u066B": ".",
    "\u066C": ",",
}

EASTERN_DIGITS = str.maketrans(
    {
        "۰": "0",
        "۱": "1",
        "۲": "2",
        "۳": "3",
        "۴": "4",
        "۵": "5",
        "۶": "6",
        "۷": "7",
        "۸": "8",
        "۹": "9",
        "٠": "0",
        "١": "1",
        "٢": "2",
        "٣": "3",
        "٤": "4",
        "٥": "5",
        "٦": "6",
        "٧": "7",
        "٨": "8",
        "٩": "9",
    }
)


@dataclass(frozen=True)
class UrduTextStats:
    total_letters: int
    perso_arabic_chars: int
    urdu_script_ratio: float
    sha256: str


def normalize_urdu(text: str, *, remove_diacritics: bool = True) -> str:
    """Normalize Urdu text for tokenizer training and LM pretraining."""

    if not text:
        return ""

    text = unicodedata.normalize("NFKC", text)

    # Compose common decomposed forms before stripping marks.
    text = text.replace("ا\u0653", "آ")
    text = text.replace("و\u0654", "ؤ")
    text = text.replace("ی\u0654", "ئ")
    text = text.replace("ي\u0654", "ئ")

    text = "".join(URDU_CHAR_MAP.get(ch, ch) for ch in text)
    text = text.translate(EASTERN_DIGITS)
    text = BIDI_CONTROLS_RE.sub("", text)
    text = text.replace("\u200c", " ")
    text = text.replace("\u200d", "")

    if remove_diacritics:
        text = ARABIC_DIACRITICS_RE.sub("", text)

    text = WHITESPACE_RE.sub(" ", text).strip()
    return unicodedata.normalize("NFC", text)


def text_stats(text: str) -> UrduTextStats:
    """Return basic script-ratio stats for normalized or raw text."""

    letters = LETTER_RE.findall(text)
    perso_arabic = [
        ch
        for ch in letters
        if "\u0600" <= ch <= "\u06FF"
        or "\u0750" <= ch <= "\u077F"
        or "\u08A0" <= ch <= "\u08FF"
    ]
    ratio = len(perso_arabic) / max(1, len(letters))
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return UrduTextStats(
        total_letters=len(letters),
        perso_arabic_chars=len(perso_arabic),
        urdu_script_ratio=ratio,
        sha256=digest,
    )


def iter_jsonl(handle: TextIO) -> Iterable[dict]:
    for line_number, line in enumerate(handle, start=1):
        line = line.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL on line {line_number}: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"JSONL line {line_number} is not an object")
        yield value


def normalize_text_file(input_path: Path, output_path: Path, *, keep_diacritics: bool) -> None:
    with input_path.open("r", encoding="utf-8", errors="replace") as fin:
        with output_path.open("w", encoding="utf-8") as fout:
            for line in fin:
                normalized = normalize_urdu(line, remove_diacritics=not keep_diacritics)
                if normalized:
                    fout.write(normalized + "\n")


def normalize_jsonl_file(
    input_path: Path,
    output_path: Path,
    *,
    text_key: str,
    normalized_key: str,
    keep_diacritics: bool,
) -> None:
    with input_path.open("r", encoding="utf-8", errors="replace") as fin:
        with output_path.open("w", encoding="utf-8") as fout:
            for record in iter_jsonl(fin):
                raw_text = str(record.get(text_key, ""))
                normalized = normalize_urdu(raw_text, remove_diacritics=not keep_diacritics)
                if not normalized:
                    continue
                stats = text_stats(normalized)
                record[normalized_key] = normalized
                record["normalization_version"] = NORMALIZATION_VERSION
                record["language_score_ur"] = stats.urdu_script_ratio
                record["dedup_hash"] = stats.sha256
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize Urdu plain text or JSONL.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--jsonl", action="store_true", help="Read and write JSONL records")
    parser.add_argument("--text-key", default="raw_text")
    parser.add_argument("--normalized-key", default="normalized_text")
    parser.add_argument("--keep-diacritics", action="store_true")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.jsonl:
        normalize_jsonl_file(
            args.input,
            args.output,
            text_key=args.text_key,
            normalized_key=args.normalized_key,
            keep_diacritics=args.keep_diacritics,
        )
    else:
        normalize_text_file(args.input, args.output, keep_diacritics=args.keep_diacritics)


if __name__ == "__main__":
    main()
