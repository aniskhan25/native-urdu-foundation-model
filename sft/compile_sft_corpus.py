"""Compile normalized, deduplicated Urdu SFT data from configured sources."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Iterable

import yaml

from data_pipeline.normalize_urdu import normalize_urdu, text_stats
from data_pipeline.quality_filter import quality_metrics
from eval.generate_samples import load_prompts
from sft.prepare_seed_sft import SEED_RECORDS


PASHTO_SPECIFIC_CHARS = frozenset("ځڅډړږښګڼټۍې")
SFT_LABEL_REPLACEMENTS = {
    "Reasoning:": "حل:",
    "Answer:": "جواب:",
}


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict) or not isinstance(config.get("sources"), list):
        raise ValueError(
            f"{path} is not an SFT source config: missing sources list. "
            "Use configs/sft_sources_v1.yaml or set SFT_SOURCE_CONFIG."
        )
    if not isinstance(config.get("quality"), dict):
        raise ValueError(f"{path} is not an SFT source config: missing quality mapping")
    return config


def record_matches_filters(record: dict[str, Any], filters: dict[str, Any]) -> bool:
    return all(record.get(field) == value for field, value in filters.items())


def canonicalize_record(record: dict[str, Any], source: dict[str, Any]) -> dict[str, str]:
    prompt = str(record.get(source["prompt_field"], "")).strip()
    response = str(record.get(source["response_field"], "")).strip()
    input_field = source.get("input_field")
    if input_field and record.get(input_field):
        prompt = f"{prompt}\n\n{str(record[input_field]).strip()}"
    category_field = source.get("category_field")
    category = str(record.get(category_field, "unknown")) if category_field else "unknown"
    response = normalize_urdu(response)
    for old, new in SFT_LABEL_REPLACEMENTS.items():
        response = response.replace(old, new)
    return {
        "prompt": normalize_urdu(prompt),
        "response": response,
        "source": str(source["id"]),
        "category": category or "unknown",
        "license": str(source["license"]),
        "provenance": str(source.get("provenance", "unknown")),
    }


def rejection_reason(record: dict[str, str], quality: dict[str, Any], excluded_prompts: set[str]) -> str | None:
    prompt = record["prompt"]
    response = record["response"]
    if not prompt or not response:
        return "empty"
    if prompt in excluded_prompts:
        return "eval_overlap"
    if len(prompt) < int(quality["min_prompt_chars"]):
        return "short_prompt"
    if len(response) < int(quality["min_response_chars"]):
        return "short_response"
    if len(prompt) > int(quality["max_prompt_chars"]):
        return "long_prompt"
    if len(response) > int(quality["max_response_chars"]):
        return "long_response"
    if prompt == response:
        return "identical_pair"
    ratio = text_stats(f"{prompt} {response}").urdu_script_ratio
    if ratio < float(quality["min_combined_urdu_script_ratio"]):
        return "low_urdu_script_ratio"
    pashto_chars = sum(char in PASHTO_SPECIFIC_CHARS for char in f"{prompt}{response}")
    if pashto_chars > int(quality["max_pashto_specific_chars"]):
        return "pashto_specific_chars"
    response_metrics = quality_metrics(response)
    if response_metrics.repeated_4gram_ratio > float(quality["max_response_repeated_4gram_ratio"]):
        return "response_repetition_4gram"
    if response_metrics.repeated_6gram_ratio > float(quality["max_response_repeated_6gram_ratio"]):
        return "response_repetition_6gram"
    if response_metrics.longest_repeated_ngram > int(quality["max_response_longest_repeated_ngram"]):
        return "response_repetition_length"
    if response_metrics.url_hits > int(quality["max_response_url_hits"]):
        return "response_url"
    if response_metrics.boilerplate_hits > int(quality["max_response_boilerplate_hits"]):
        return "response_boilerplate"
    return None


def pair_digest(record: dict[str, str]) -> str:
    value = f"{record['prompt']}\0{record['response']}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def accept_records(
    records: Iterable[dict[str, str]],
    quality: dict[str, Any],
    excluded_prompts: set[str],
) -> tuple[list[dict[str, str]], Counter]:
    accepted = []
    rejected = Counter()
    prompt_hashes = set()
    pair_hashes = set()
    for record in records:
        reason = rejection_reason(record, quality, excluded_prompts)
        if reason:
            rejected[reason] += 1
            continue
        prompt_hash = text_stats(record["prompt"]).sha256
        digest = pair_digest(record)
        if digest in pair_hashes:
            rejected["duplicate_pair"] += 1
            continue
        if prompt_hash in prompt_hashes:
            rejected["duplicate_prompt"] += 1
            continue
        prompt_hashes.add(prompt_hash)
        pair_hashes.add(digest)
        accepted.append(record)
    return accepted, rejected


def split_records(
    records: list[dict[str, str]], validation_fraction: float, seed: int
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for record in records:
        groups[(record["source"], record["category"])].append(record)

    train = []
    validation = []
    rng = random.Random(seed)
    for key in sorted(groups):
        group = list(groups[key])
        rng.shuffle(group)
        validation_count = round(len(group) * validation_fraction)
        if len(group) >= 20:
            validation_count = max(1, validation_count)
        validation.extend(group[:validation_count])
        train.extend(group[validation_count:])
    rng.shuffle(train)
    rng.shuffle(validation)
    if not validation and train:
        validation.append(train.pop())
    return train, validation


def write_jsonl(path: Path, records: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def iter_hf_source(source: dict[str, Any]) -> Iterable[dict[str, Any]]:
    from datasets import load_dataset

    dataset = load_dataset(
        source["dataset"],
        source.get("config"),
        split=source.get("split", "train"),
        streaming=True,
    )
    shuffle_buffer = int(source.get("shuffle_buffer", 0))
    if shuffle_buffer > 0:
        dataset = dataset.shuffle(seed=int(source.get("seed", 42)), buffer_size=shuffle_buffer)
    yield from dataset


def load_excluded_prompts(config_path: Path, paths: list[str]) -> set[str]:
    excluded = set()
    for value in paths:
        path = Path(value)
        if not path.is_absolute():
            path = config_path.parent.parent / path
        excluded.update(normalize_urdu(prompt) for prompt in load_prompts(path))
    return excluded


def curated_seed_source() -> dict[str, Any]:
    return {
        "id": "curated_seed_v1",
        "prompt_field": "prompt",
        "response_field": "response",
        "category_field": "category",
        "license": "project-curated",
        "provenance": "human_curated",
    }


def compile_corpus(
    config: dict[str, Any],
    config_path: Path,
    selected_sources: set[str] | None = None,
    max_records_override: int | None = None,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    excluded_prompts = load_excluded_prompts(config_path, config.get("exclude_prompt_files", []))
    candidates = []
    source_stats: dict[str, dict[str, Any]] = {}

    if config.get("include_curated_seed", True):
        source = curated_seed_source()
        seed_records = [canonicalize_record(record, source) for record in SEED_RECORDS]
        candidates.extend(seed_records)
        source_stats[source["id"]] = {
            "dataset": "local",
            "license": source["license"],
            "provenance": source["provenance"],
            "seen": len(seed_records),
            "matched": len(seed_records),
        }

    for source in config["sources"]:
        source_id = str(source["id"])
        if selected_sources and source_id not in selected_sources:
            continue
        max_records = int(max_records_override or source["max_records"])
        max_scanned = int(source.get("max_scanned", max_records * 10))
        seen = 0
        matched = 0
        for raw_record in iter_hf_source(source):
            if seen >= max_scanned or matched >= max_records:
                break
            seen += 1
            if not record_matches_filters(raw_record, source.get("filters", {})):
                continue
            candidates.append(canonicalize_record(raw_record, source))
            matched += 1
        source_stats[source_id] = {
            "dataset": source["dataset"],
            "license": source["license"],
            "provenance": source.get("provenance", "unknown"),
            "seen": seen,
            "matched": matched,
        }

    accepted, rejected = accept_records(candidates, config["quality"], excluded_prompts)
    accepted_by_source = Counter(record["source"] for record in accepted)
    categories = Counter(record["category"] for record in accepted)
    licenses = Counter(record["license"] for record in accepted)
    provenance = Counter(record["provenance"] for record in accepted)
    for source_id, stats in source_stats.items():
        stats["accepted"] = accepted_by_source[source_id]
    summary = {
        "candidates": len(candidates),
        "accepted": len(accepted),
        "rejected": dict(sorted(rejected.items())),
        "sources": source_stats,
        "categories": dict(sorted(categories.items())),
        "licenses": dict(sorted(licenses.items())),
        "provenance": dict(sorted(provenance.items())),
        "excluded_prompts": len(excluded_prompts),
    }
    return accepted, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile the Urdu SFT v1 corpus.")
    parser.add_argument("--config", type=Path, default=Path("configs/sft_sources_v1.yaml"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source", action="append", dest="sources")
    parser.add_argument("--max-records-per-source", type=int, default=None)
    parser.add_argument("--min-total-records", type=int, default=5000)
    args = parser.parse_args()

    config = load_config(args.config)
    accepted, summary = compile_corpus(
        config,
        args.config.resolve(),
        selected_sources=set(args.sources) if args.sources else None,
        max_records_override=args.max_records_per_source,
    )
    if len(accepted) < args.min_total_records:
        raise ValueError(f"Accepted {len(accepted)} records; require at least {args.min_total_records}")

    train, validation = split_records(
        accepted,
        validation_fraction=float(config["validation_fraction"]),
        seed=int(config.get("seed", 42)),
    )
    write_jsonl(args.output_dir / "sft_train.jsonl", train)
    write_jsonl(args.output_dir / "sft_val.jsonl", validation)
    summary.update({"train": len(train), "validation": len(validation)})
    summary_path = args.output_dir / "sft_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
