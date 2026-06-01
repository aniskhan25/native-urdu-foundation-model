"""Compile selected open datasets into canonical normalized JSONL shards.

This script is intentionally streaming-first so it can run against large
Hugging Face datasets without downloading the full corpus up front.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

# Avoid leaving hf-xet helper processes around for short local pilot runs.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

from tqdm import tqdm

try:
    import yaml
except ImportError as exc:  # pragma: no cover - dependency guard
    raise SystemExit("Missing dependency: PyYAML. Run `pip install -r requirements.txt`.") from exc

try:
    from datasets import load_dataset
except ImportError as exc:  # pragma: no cover - dependency guard
    raise SystemExit("Missing dependency: datasets. Run `pip install -r requirements.txt`.") from exc

try:
    from .normalize_urdu import NORMALIZATION_VERSION, normalize_urdu, text_stats
    from .quality_filter import passes_quality, quality_metrics
except ImportError:  # pragma: no cover - direct script execution
    from normalize_urdu import NORMALIZATION_VERSION, normalize_urdu, text_stats
    from quality_filter import passes_quality, quality_metrics


SOURCE_LOADERS: dict[str, dict[str, Any]] = {
    "fineweb2_urd_arab": {
        "bucket": "urdu_web",
        "hf_dataset": "HuggingFaceFW/fineweb-2",
        "hf_config": "urd_Arab",
        "split": "train",
        "text_fields": ["text"],
        "url_fields": ["url", "source_url"],
        "timestamp_fields": ["date", "timestamp", "crawl_date"],
    },
    "hplt_ur_cleaned": {
        "bucket": "urdu_web",
        "hf_dataset": "HPLT/hplt_monolingual_v1_2",
        "hf_config": "ur_cleaned",
        "split": "train",
        "trust_remote_code": True,
        "text_fields": ["text"],
        "url_fields": ["url"],
        "timestamp_fields": ["timestamp", "date"],
    },
    "mc4_ur": {
        "bucket": "urdu_web",
        "hf_dataset": "allenai/c4",
        "hf_config": "ur",
        "split": "train",
        "text_fields": ["text"],
        "url_fields": ["url"],
        "timestamp_fields": ["timestamp"],
    },
    "makhzan_urdu": {
        "bucket": "urdu_literature",
        "hf_dataset": "ReySajju742/makhzan-urdu",
        "hf_config": None,
        "split": "train",
        "text_fields": ["xml"],
        "url_fields": ["url"],
        "timestamp_fields": ["timestamp", "date"],
    },
    "wikimedia_wikisource_ur": {
        "bucket": "urdu_literature",
        "hf_dataset": "wikimedia/wikisource",
        "hf_config": None,
        "split": "train",
        "text_fields": ["text"],
        "url_fields": ["url"],
        "timestamp_fields": ["timestamp", "date"],
        "language_fields": ["language", "lang"],
        "language_values": ["ur", "urd", "Urdu"],
    },
}

XML_TAG_RE = re.compile(r"<[^>]+>")
BODY_RE = re.compile(r"<body[^>]*>(.*?)</body>", re.DOTALL | re.IGNORECASE)


def first_present(record: dict[str, Any], field_names: Iterable[str]) -> Any:
    for field_name in field_names:
        value = record.get(field_name)
        if value is not None and value != "":
            return value
    return None


def stable_doc_id(source_id: str, text: str, fallback_index: int) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]
    return f"{source_id}:{fallback_index}:{digest}"


def load_source_plan(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def source_metadata(source_plan: dict[str, Any], source_id: str) -> dict[str, Any]:
    for bucket_name, bucket in source_plan["selected_sources"].items():
        for source in bucket.get("sources", []):
            if source.get("id") == source_id:
                return {
                    "bucket": bucket_name,
                    "name": source.get("name", source_id),
                    "url": source.get("url"),
                    "license": source.get("license"),
                    "priority": source.get("priority"),
                }
    raise KeyError(f"Source {source_id!r} is not present in configs/data_sources.yaml")


def record_language_allowed(record: dict[str, Any], loader: dict[str, Any]) -> bool:
    language_fields = loader.get("language_fields") or []
    if not language_fields:
        return True
    allowed_values = set(loader.get("language_values") or [])
    value = first_present(record, language_fields)
    return value in allowed_values


def load_hf_stream(loader: dict[str, Any]) -> Iterable[dict[str, Any]]:
    kwargs = {
        "path": loader["hf_dataset"],
        "split": loader.get("split", "train"),
        "streaming": True,
    }
    if loader.get("hf_config"):
        kwargs["name"] = loader["hf_config"]
    if loader.get("trust_remote_code"):
        kwargs["trust_remote_code"] = True
    return load_dataset(**kwargs)


def strip_xml_text(text: str) -> str:
    body_match = BODY_RE.search(text)
    if body_match:
        text = body_match.group(1)
    text = XML_TAG_RE.sub(" ", text)
    return text


def extract_raw_text(source_id: str, loader: dict[str, Any], record: dict[str, Any]) -> str | None:
    raw_text = first_present(record, loader["text_fields"])
    if raw_text is None:
        available = ", ".join(sorted(record.keys()))
        raise KeyError(f"No configured text field for {source_id}; available fields: {available}")
    raw_text = str(raw_text)
    if source_id == "makhzan_urdu":
        return strip_xml_text(raw_text)
    return raw_text


def canonicalize_record(
    *,
    source_id: str,
    source_meta: dict[str, Any],
    loader: dict[str, Any],
    raw_record: dict[str, Any],
    index: int,
    min_chars: int,
    min_script_ratio: float,
    max_repeated_char_ratio: float,
    max_symbol_ratio: float,
) -> dict[str, Any] | None:
    if not record_language_allowed(raw_record, loader):
        return None

    raw_text = extract_raw_text(source_id, loader, raw_record)
    normalized_text = normalize_urdu(raw_text)
    if not normalized_text:
        return None

    metrics = quality_metrics(normalized_text)
    if not passes_quality(
        metrics,
        min_chars=min_chars,
        min_script_ratio=min_script_ratio,
        max_repeated_char_ratio=max_repeated_char_ratio,
        max_symbol_ratio=max_symbol_ratio,
    ):
        return None

    stats = text_stats(normalized_text)
    return {
        "doc_id": str(raw_record.get("id") or stable_doc_id(source_id, normalized_text, index)),
        "source": source_id,
        "source_bucket": source_meta["bucket"],
        "source_name": source_meta["name"],
        "url": first_present(raw_record, loader.get("url_fields", [])) or source_meta.get("url"),
        "license": source_meta.get("license"),
        "timestamp": first_present(raw_record, loader.get("timestamp_fields", [])),
        "raw_text": raw_text,
        "normalized_text": normalized_text,
        "normalization_version": NORMALIZATION_VERSION,
        "quality_score": metrics.quality_score,
        "quality_metrics": asdict(metrics),
        "language_score_ur": stats.urdu_script_ratio,
        "dedup_hash": stats.sha256,
    }


def compile_source(
    *,
    source_id: str,
    source_plan: dict[str, Any],
    output_dir: Path,
    max_docs: int | None,
    max_scanned: int | None,
    min_chars: int,
    min_script_ratio: float,
    max_repeated_char_ratio: float,
    max_symbol_ratio: float,
) -> dict[str, Any]:
    if source_id not in SOURCE_LOADERS:
        raise KeyError(f"No loader is implemented for source {source_id!r}")

    loader = SOURCE_LOADERS[source_id]
    source_meta = source_metadata(source_plan, source_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{source_id}.jsonl"
    seen_hashes: set[str] = set()
    seen = 0
    kept = 0
    duplicates = 0
    filtered = 0

    dataset = load_hf_stream(loader)
    with output_path.open("w", encoding="utf-8") as out:
        progress = tqdm(dataset, desc=f"compiling {source_id}", unit="docs")
        for raw_record in progress:
            seen += 1
            canonical = canonicalize_record(
                source_id=source_id,
                source_meta=source_meta,
                loader=loader,
                raw_record=raw_record,
                index=seen,
                min_chars=min_chars,
                min_script_ratio=min_script_ratio,
                max_repeated_char_ratio=max_repeated_char_ratio,
                max_symbol_ratio=max_symbol_ratio,
            )
            if canonical is None:
                filtered += 1
            elif canonical["dedup_hash"] in seen_hashes:
                duplicates += 1
            else:
                seen_hashes.add(canonical["dedup_hash"])
                out.write(json.dumps(canonical, ensure_ascii=False) + "\n")
                kept += 1
                progress.set_postfix(kept=kept, filtered=filtered, duplicates=duplicates)

            if max_docs is not None and kept >= max_docs:
                break
            if max_scanned is not None and seen >= max_scanned:
                break

    return {
        "source_id": source_id,
        "output_path": str(output_path),
        "seen": seen,
        "kept": kept,
        "filtered": filtered,
        "duplicates": duplicates,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile selected open Urdu corpus sources.")
    parser.add_argument("--config", type=Path, default=Path("configs/data_sources.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/compiled"))
    parser.add_argument(
        "--source",
        action="append",
        dest="sources",
        help="Source id to compile. May be repeated. Defaults to the web pilot source.",
    )
    parser.add_argument("--max-docs-per-source", type=int, default=1000)
    parser.add_argument(
        "--max-scanned-per-source",
        type=int,
        default=None,
        help="Stop after scanning this many raw records even if max kept docs has not been reached.",
    )
    parser.add_argument("--min-chars", type=int, default=200)
    parser.add_argument("--min-script-ratio", type=float, default=0.65)
    parser.add_argument("--max-repeated-char-ratio", type=float, default=0.20)
    parser.add_argument("--max-symbol-ratio", type=float, default=0.15)
    parser.add_argument(
        "--force-exit",
        action="store_true",
        help="Exit via os._exit after writing outputs. Useful if HF streaming leaves helper threads alive.",
    )
    args = parser.parse_args()

    source_plan = load_source_plan(args.config)
    sources = args.sources or ["fineweb2_urd_arab"]
    summaries = []
    for source_id in sources:
        summaries.append(
            compile_source(
                source_id=source_id,
                source_plan=source_plan,
                output_dir=args.output_dir,
                max_docs=args.max_docs_per_source,
                max_scanned=args.max_scanned_per_source,
                min_chars=args.min_chars,
                min_script_ratio=args.min_script_ratio,
                max_repeated_char_ratio=args.max_repeated_char_ratio,
                max_symbol_ratio=args.max_symbol_ratio,
            )
        )

    summary_path = args.output_dir / "compile_summary.json"
    summary_path.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summaries, ensure_ascii=False, indent=2))
    sys.stdout.flush()
    sys.stderr.flush()
    if args.force_exit:
        os._exit(0)


if __name__ == "__main__":
    main()
