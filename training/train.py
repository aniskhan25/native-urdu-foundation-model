"""Training entrypoint placeholder.

This repository currently focuses on data preparation and tokenizer training.
Use this file as the integration point for PyTorch FSDP or a Megatron-style
trainer once tokenized shards and the final model implementation are ready.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(description="Native Urdu LM training entrypoint.")
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    raise NotImplementedError(
        "Training loop is not implemented yet. Loaded config for "
        f"{config.get('project', 'unknown_project')}; integrate FSDP/Megatron here."
    )


if __name__ == "__main__":
    main()

