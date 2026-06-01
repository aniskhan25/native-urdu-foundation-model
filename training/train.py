"""Training entrypoint and preflight checks."""

from __future__ import annotations

import argparse
from functools import partial
import json
import math
import os
import time
from pathlib import Path

import yaml

from training.dataset import RandomTokenSampler
from training.model import DecoderBlock, LlamaDecoder, parameter_count


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


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value is not None else default


def init_distributed() -> tuple[int, int, int]:
    import torch
    import torch.distributed as dist

    rank = env_int("RANK", 0)
    world_size = env_int("WORLD_SIZE", 1)
    local_rank = env_int("LOCAL_RANK", 0)
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def cleanup_distributed() -> None:
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def current_lr(step: int, *, peak_lr: float, min_lr: float, warmup_steps: int, total_steps: int) -> float:
    if step < warmup_steps:
        return peak_lr * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return min_lr + (peak_lr - min_lr) * cosine


def set_optimizer_lr(optimizer: object, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def maybe_wrap_fsdp(model: object, enabled: bool, device: object) -> object:
    if not enabled:
        return model
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

    import torch

    auto_wrap_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={DecoderBlock})
    mixed_precision = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )
    return FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=mixed_precision,
        auto_wrap_policy=auto_wrap_policy,
        device_id=device,
        use_orig_params=True,
    )


def save_checkpoint(model: object, output_dir: Path, step: int, tokens_seen: int, rank: int) -> None:
    import torch
    import torch.distributed as dist

    checkpoint_path = output_dir / f"step_{step:06d}.pt"
    if dist.is_available() and dist.is_initialized():
        from torch.distributed.fsdp import FullStateDictConfig, FullyShardedDataParallel as FSDP, StateDictType

        if isinstance(model, FSDP):
            full_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, full_config):
                state_dict = model.state_dict()
        else:
            state_dict = model.state_dict()
    else:
        state_dict = model.state_dict()
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"step": step, "tokens_seen": tokens_seen, "model": state_dict}, checkpoint_path)
        latest_path = output_dir / "latest.json"
        latest_path.write_text(json.dumps({"step": step, "tokens_seen": tokens_seen, "checkpoint": str(checkpoint_path)}))


def train(config: dict, *, max_steps_override: int | None = None) -> None:
    import numpy as np
    import torch
    import torch.distributed as dist
    import torch.nn.functional as F

    rank, world_size, local_rank = init_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    data_config = config["data"]
    model_config = config["model"]
    training_config = config["training"]
    infra_config = config.get("infrastructure", {})

    seq_len = int(data_config["sequence_length"])
    target_tokens = int(config.get("target_tokens", 0))
    global_batch_tokens = int(training_config["global_batch_tokens"])
    per_rank_batch = max(1, global_batch_tokens // max(1, world_size * seq_len))
    actual_global_tokens = per_rank_batch * world_size * seq_len
    total_steps = int(max_steps_override or training_config.get("estimated_steps") or target_tokens // max(1, actual_global_tokens))
    warmup_steps = max(1, int(training_config.get("warmup_tokens", 0)) // max(1, actual_global_tokens))
    checkpoint_interval = int(training_config.get("checkpoint_interval_tokens", 0))
    checkpoint_steps = max(1, checkpoint_interval // max(1, actual_global_tokens)) if checkpoint_interval else total_steps + 1

    seed = int(training_config.get("seed", 42)) + rank
    torch.manual_seed(seed)
    np.random.seed(seed)

    sampler = RandomTokenSampler(
        Path(data_config["manifest"]),
        {name: float(weight) for name, weight in data_config.get("source_weights", {}).items()},
        seq_len=seq_len,
        seed=seed,
    )

    model = LlamaDecoder(model_config, int(config["tokenizer"]["vocab_size"])).to(device)
    use_fsdp = world_size > 1 and infra_config.get("parallelism") == "fsdp_full_shard"
    model = maybe_wrap_fsdp(model, enabled=use_fsdp, device=device)
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config["lr"]),
        betas=(float(training_config.get("beta1", 0.9)), float(training_config.get("beta2", 0.95))),
        eps=float(training_config.get("eps", 1e-8)),
        weight_decay=float(training_config.get("weight_decay", 0.1)),
    )

    output_dir = Path(infra_config.get("output_dir", "runs/dress_rehearsal"))
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"parameters={parameter_count(model):,}")
        print(f"world_size={world_size} per_rank_batch={per_rank_batch} actual_global_tokens={actual_global_tokens}")
        print(f"total_steps={total_steps} warmup_steps={warmup_steps} checkpoint_steps={checkpoint_steps}")

    peak_lr = float(training_config["lr"])
    min_lr = peak_lr * float(training_config.get("min_lr_ratio", 0.1))
    grad_clip = float(training_config.get("grad_clip", 1.0))
    checkpoint_blocks = bool(training_config.get("activation_checkpointing", False))
    tokens_seen = 0
    start_time = time.time()

    for step in range(total_steps):
        lr = current_lr(step, peak_lr=peak_lr, min_lr=min_lr, warmup_steps=warmup_steps, total_steps=total_steps)
        set_optimizer_lr(optimizer, lr)

        batch, source_counts = sampler.batch(per_rank_batch)
        batch_tensor = torch.from_numpy(batch).to(device=device, dtype=torch.long, non_blocking=True)
        input_ids = batch_tensor[:, :-1]
        labels = batch_tensor[:, 1:]

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(input_ids, checkpoint_blocks=checkpoint_blocks)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        tokens_seen += actual_global_tokens
        if rank == 0 and (step == 0 or (step + 1) % 10 == 0):
            elapsed = max(1e-6, time.time() - start_time)
            tokens_per_second = tokens_seen / elapsed
            counts = " ".join(f"{name}={count}" for name, count in source_counts.items())
            print(
                f"step={step + 1}/{total_steps} loss={loss.item():.4f} lr={lr:.6g} "
                f"tokens={tokens_seen} tok_s={tokens_per_second:.0f} {counts}",
                flush=True,
            )

        if (step + 1) % checkpoint_steps == 0 or step + 1 == total_steps:
            save_checkpoint(model, output_dir, step + 1, tokens_seen, rank)
            if dist.is_available() and dist.is_initialized():
                dist.barrier()

    cleanup_distributed()


def main() -> None:
    parser = argparse.ArgumentParser(description="Native Urdu LM training entrypoint.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--preflight", action="store_true", help="Validate config, tokenizer, manifest, and shards.")
    parser.add_argument("--max-steps", type=int, default=None, help="Override training steps for smoke tests.")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.preflight:
        preflight(config)
        return
    train(config, max_steps_override=args.max_steps)


if __name__ == "__main__":
    main()
