"""Prompt/response SFT dataset with response-only labels."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from data_pipeline.normalize_urdu import normalize_urdu


IGNORE_INDEX = -100


class Tokenizer(Protocol):
    def encode(self, text: str, out_type: type = int) -> list[int]: ...

    def eos_id(self) -> int: ...


@dataclass(frozen=True)
class SftExample:
    input_ids: np.ndarray
    labels: np.ndarray


def format_prompt(prompt: str) -> str:
    return f"ہدایت:\n{prompt.strip()}\n\nجواب:\n"


def load_sft_records(path: Path) -> list[dict[str, str]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            prompt = str(record.get("prompt", "")).strip()
            response = str(record.get("response", "")).strip()
            if not prompt or not response:
                raise ValueError(f"{path}:{line_number} requires non-empty prompt and response")
            records.append({"prompt": prompt, "response": response})
    if not records:
        raise ValueError(f"No SFT records found: {path}")
    return records


def encode_sft_record(record: dict[str, str], tokenizer: Tokenizer, sequence_length: int) -> SftExample:
    prompt = normalize_urdu(record["prompt"])
    response = normalize_urdu(record["response"])
    prefix_ids = tokenizer.encode(format_prompt(prompt), out_type=int)
    response_ids = tokenizer.encode(response, out_type=int)
    eos_id = tokenizer.eos_id()
    if eos_id >= 0:
        response_ids = response_ids + [eos_id]

    input_ids = prefix_ids + response_ids
    labels = [IGNORE_INDEX] * len(prefix_ids) + response_ids
    if len(input_ids) > sequence_length:
        overflow = len(input_ids) - sequence_length
        input_ids = input_ids[overflow:]
        labels = labels[overflow:]
    if len(input_ids) < sequence_length:
        pad = sequence_length - len(input_ids)
        pad_id = eos_id if eos_id >= 0 else 0
        input_ids = input_ids + [pad_id] * pad
        labels = labels + [IGNORE_INDEX] * pad

    return SftExample(
        input_ids=np.asarray(input_ids, dtype=np.int64),
        labels=np.asarray(labels, dtype=np.int64),
    )


class SftDataset:
    def __init__(self, path: Path, tokenizer: Tokenizer, sequence_length: int) -> None:
        self.examples = [encode_sft_record(record, tokenizer, sequence_length) for record in load_sft_records(path)]

    def __len__(self) -> int:
        return len(self.examples)

    def batch(self, indices: list[int]) -> tuple[np.ndarray, np.ndarray]:
        input_ids = np.stack([self.examples[index].input_ids for index in indices])
        labels = np.stack([self.examples[index].labels for index in indices])
        return input_ids, labels


class SftBatchSampler:
    def __init__(self, dataset_size: int, batch_size: int, seed: int) -> None:
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.rng = random.Random(seed)
        self.indices = list(range(dataset_size))
        self.position = 0
        self.rng.shuffle(self.indices)

    def batch_indices(self) -> list[int]:
        if self.position + self.batch_size > self.dataset_size:
            self.rng.shuffle(self.indices)
            self.position = 0
        batch = self.indices[self.position : self.position + self.batch_size]
        self.position += self.batch_size
        return batch


def dataset_size(path: Path) -> int:
    return len(load_sft_records(path))
