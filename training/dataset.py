"""Random sequence sampler over tokenized binary shards."""

from __future__ import annotations

import bisect
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TokenShard:
    path: Path
    tokens: int
    dtype: np.dtype


@dataclass
class TokenSource:
    name: str
    weight: float
    shards: list[TokenShard]
    cumulative_tokens: list[int]

    @property
    def total_tokens(self) -> int:
        return self.cumulative_tokens[-1]


class RandomTokenSampler:
    """Sample contiguous `seq_len + 1` token windows from weighted sources."""

    def __init__(self, manifest_path: Path, source_weights: dict[str, float], seq_len: int, seed: int) -> None:
        self.seq_len = seq_len
        self.rng = random.Random(seed)
        self.sources = self._load_sources(manifest_path, source_weights)
        self.source_names = [source.name for source in self.sources]
        self.source_weights = [source.weight for source in self.sources]
        self._arrays: dict[Path, np.memmap] = {}

    def _load_sources(self, manifest_path: Path, source_weights: dict[str, float]) -> list[TokenSource]:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        sources = []
        for source in manifest["sources"]:
            name = source["source"]
            weight = float(source_weights.get(name, 0.0))
            if weight <= 0:
                continue
            dtype = np.dtype(source["dtype"])
            shards = [
                TokenShard(path=Path(shard["path"]), tokens=int(shard["tokens"]), dtype=dtype)
                for shard in source["shards"]
                if int(shard["tokens"]) > self.seq_len + 1
            ]
            if not shards:
                continue
            cumulative = []
            running = 0
            for shard in shards:
                running += shard.tokens
                cumulative.append(running)
            sources.append(TokenSource(name=name, weight=weight, shards=shards, cumulative_tokens=cumulative))
        if not sources:
            raise ValueError(f"No trainable sources in manifest: {manifest_path}")
        return sources

    def _array(self, shard: TokenShard) -> np.memmap:
        if shard.path not in self._arrays:
            self._arrays[shard.path] = np.memmap(shard.path, mode="r", dtype=shard.dtype)
        return self._arrays[shard.path]

    def sample(self) -> tuple[np.ndarray, str]:
        source = self.rng.choices(self.sources, weights=self.source_weights, k=1)[0]
        token_index = self.rng.randrange(0, source.total_tokens - self.seq_len - 1)
        shard_index = bisect.bisect_right(source.cumulative_tokens, token_index)
        shard_start = 0 if shard_index == 0 else source.cumulative_tokens[shard_index - 1]
        shard = source.shards[shard_index]
        offset = token_index - shard_start
        if offset + self.seq_len + 1 > shard.tokens:
            offset = shard.tokens - self.seq_len - 1
        window = np.asarray(self._array(shard)[offset : offset + self.seq_len + 1], dtype=np.int64)
        return window, source.name

    def batch(self, batch_size: int) -> tuple[np.ndarray, dict[str, int]]:
        windows = []
        counts = {name: 0 for name in self.source_names}
        for _ in range(batch_size):
            window, source_name = self.sample()
            windows.append(window)
            counts[source_name] += 1
        return np.stack(windows), counts

