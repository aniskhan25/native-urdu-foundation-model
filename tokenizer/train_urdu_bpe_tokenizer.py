"""Train and evaluate a 32k SentencePiece BPE tokenizer for Urdu."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Iterable

import sentencepiece as spm
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "data_pipeline"))

from normalize_urdu import normalize_urdu  # noqa: E402

REQUIRED_CHARS = (
    "اآبپتٹثجچحخدڈذرڑزژسشصضطظعغفقکگلمنںوؤہھءئیے"
    "0123456789"
    "،؛؟!۔:()[]{}«»\"'%-–—"
)


def iter_text_files(paths: list[str]) -> Iterable[Path]:
    for value in paths:
        path = Path(value)
        if path.is_file():
            yield path
            continue
        for subpath in path.rglob("*"):
            if subpath.is_file() and subpath.suffix.lower() in {".txt", ".jsonl"}:
                yield subpath


def iter_training_texts(paths: list[str], *, text_key: str) -> Iterable[str]:
    for file_path in iter_text_files(paths):
        with file_path.open("r", encoding="utf-8", errors="replace") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                if file_path.suffix.lower() != ".jsonl":
                    yield line
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL in {file_path}:{line_number}: {exc}") from exc
                if text_key not in record:
                    raise KeyError(f"Missing {text_key!r} in {file_path}:{line_number}")
                yield str(record[text_key])


def reservoir_sample_lines(
    input_paths: list[str],
    output_path: Path,
    *,
    max_lines: int,
    min_chars: int,
    seed: int,
    text_key: str,
) -> None:
    random.seed(seed)
    reservoir: list[str] = []
    n_seen = 0
    files = list(iter_text_files(input_paths))
    print(f"found_files={len(files)}")

    for text in tqdm(iter_training_texts(input_paths, text_key=text_key), desc="sampling tokenizer text"):
        text = normalize_urdu(text)
        if len(text) < min_chars:
            continue
        n_seen += 1
        if len(reservoir) < max_lines:
            reservoir.append(text)
            continue
        index = random.randint(0, n_seen - 1)
        if index < max_lines:
            reservoir[index] = text

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as out:
        for line in reservoir:
            out.write(line + "\n")
    print(f"seen_valid_lines={n_seen} sampled_lines={len(reservoir)} sample_file={output_path}")


def train_sentencepiece_bpe(input_file: Path, model_prefix: str, *, vocab_size: int) -> None:
    spm.SentencePieceTrainer.train(
        input=str(input_file),
        model_prefix=model_prefix,
        model_type="bpe",
        vocab_size=vocab_size,
        unk_id=0,
        bos_id=1,
        eos_id=2,
        pad_id=3,
        unk_piece="<unk>",
        bos_piece="<s>",
        eos_piece="</s>",
        pad_piece="<pad>",
        character_coverage=1.0,
        byte_fallback=False,
        required_chars=REQUIRED_CHARS,
        normalization_rule_name="identity",
        remove_extra_whitespaces=True,
        split_by_unicode_script=True,
        split_by_whitespace=True,
        split_by_number=True,
        max_sentence_length=16384,
        input_sentence_size=5_000_000,
        shuffle_input_sentence=True,
        hard_vocab_limit=False,
        train_extremely_large_corpus=True,
    )


def evaluate_tokenizer(model_path: Path, eval_paths: list[str], *, max_lines: int, text_key: str) -> None:
    sp = spm.SentencePieceProcessor()
    sp.load(str(model_path))

    total_words = 0
    total_pieces = 0
    unk_count = 0
    line_count = 0

    for text in iter_training_texts(eval_paths, text_key=text_key):
        text = normalize_urdu(text)
        if not text:
            continue
        ids = sp.encode(text, out_type=int)
        total_words += max(1, len(text.split()))
        total_pieces += len(ids)
        unk_count += sum(1 for token_id in ids if token_id == sp.unk_id())
        line_count += 1
        if line_count >= max_lines:
            break

    fertility = total_pieces / max(1, total_words)
    unk_rate = unk_count / max(1, total_pieces)
    print("Tokenizer evaluation")
    print("--------------------")
    print(f"Lines:     {line_count:,}")
    print(f"Words:     {total_words:,}")
    print(f"Pieces:    {total_pieces:,}")
    print(f"Fertility: {fertility:.3f} pieces/word")
    print(f"UNK rate:  {unk_rate:.8f}")
    if fertility > 2.5:
        print("WARNING: high fertility. Inspect normalization, corpus noise, and vocab size.")
    if unk_rate > 0.0001:
        print("WARNING: non-trivial UNK rate. Inspect rare Unicode characters.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a SentencePiece BPE tokenizer for Urdu.")
    parser.add_argument("--input", nargs="+", required=True, help="Input txt/jsonl files or directories")
    parser.add_argument("--model-prefix", required=True)
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--max-lines", type=int, default=2_000_000)
    parser.add_argument("--min-chars", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-file", type=Path)
    parser.add_argument("--eval-lines", type=int, default=10_000)
    parser.add_argument("--text-key", default="normalized_text")
    args = parser.parse_args()

    sample_file = args.sample_file or Path(args.model_prefix + ".tokenizer_sample.txt")
    reservoir_sample_lines(
        args.input,
        sample_file,
        max_lines=args.max_lines,
        min_chars=args.min_chars,
        seed=args.seed,
        text_key=args.text_key,
    )
    train_sentencepiece_bpe(sample_file, args.model_prefix, vocab_size=args.vocab_size)
    evaluate_tokenizer(
        Path(args.model_prefix + ".model"),
        args.input,
        max_lines=args.eval_lines,
        text_key=args.text_key,
    )


if __name__ == "__main__":
    main()
