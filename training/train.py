"""Training entrypoint and preflight checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def preflight(config: dict) -> None:
    data = config.get("data", {})
    tokenizer = config.get("tokenizer", {})

    manifest_path = Path(data.get("manifest", ""))
    tokenizer_path = Path(tokenizer.get("path", ""))
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing data manifest: {manifest_path}")
    if not tokenizer_path.is_file():
        raise FileNotFoundError(f"Missing tokenizer model: {tokenizer_path}")

    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    total_tokens = int(manifest.get("total_tokens", 0))
    if total_tokens <= 0:
        raise ValueError(f"Manifest has no tokens: {manifest_path}")

    missing_shards = []
    for source in manifest.get("sources", []):
        for shard in source.get("shards", []):
            shard_path = Path(shard["path"])
            if not shard_path.is_file():
                missing_shards.append(str(shard_path))
    if missing_shards:
        preview = "\n".join(missing_shards[:10])
        raise FileNotFoundError(f"Missing token shards:\n{preview}")

    seq_len = int(data.get("sequence_length", config.get("model", {}).get("seq_len", 0)))
    batch_tokens = int(config.get("training", {}).get("global_batch_tokens", 0))
    estimated_steps = total_tokens // max(1, batch_tokens)
    print("Preflight OK")
    print(f"manifest={manifest_path}")
    print(f"tokenizer={tokenizer_path}")
    print(f"total_tokens={total_tokens}")
    print(f"sequence_length={seq_len}")
    print(f"global_batch_tokens={batch_tokens}")
    print(f"estimated_steps={estimated_steps}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Native Urdu LM training entrypoint.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--preflight", action="store_true", help="Validate config, tokenizer, manifest, and shards.")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.preflight:
        preflight(config)
        return
    raise NotImplementedError(
        "Training loop is not implemented yet. Loaded config for "
        f"{config.get('project', 'unknown_project')}; integrate FSDP/Megatron here."
    )


if __name__ == "__main__":
    main()
