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
from training.progress import resume_progress


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


def checkpoint_state_dict(model: object, optimizer: object) -> tuple[dict, dict]:
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        from torch.distributed.fsdp import (
            FullOptimStateDictConfig,
            FullStateDictConfig,
            FullyShardedDataParallel as FSDP,
            StateDictType,
        )

        if isinstance(model, FSDP):
            state_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            optim_config = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, state_config, optim_config):
                return model.state_dict(), FSDP.optim_state_dict(model, optimizer)
    return model.state_dict(), optimizer.state_dict()


def load_checkpoint_state(model: object, optimizer: object, checkpoint_path: Path, device: object) -> tuple[int, int]:
    import torch
    import torch.distributed as dist

    checkpoint = torch.load(checkpoint_path, map_location="cpu", mmap=True)
    if dist.is_available() and dist.is_initialized():
        from torch.distributed.fsdp import (
            FullOptimStateDictConfig,
            FullStateDictConfig,
            FullyShardedDataParallel as FSDP,
            StateDictType,
        )

        if isinstance(model, FSDP):
            state_config = FullStateDictConfig(offload_to_cpu=False, rank0_only=False)
            optim_config = FullOptimStateDictConfig(offload_to_cpu=False, rank0_only=False)
            with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, state_config, optim_config):
                model.load_state_dict(checkpoint["model"])
                if "optimizer" in checkpoint:
                    optimizer_state = FSDP.optim_state_dict_to_load(model, optimizer, checkpoint["optimizer"])
                    optimizer.load_state_dict(optimizer_state)
            return int(checkpoint["step"]), int(checkpoint["tokens_seen"])

    model.load_state_dict(checkpoint["model"])
    if "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return int(checkpoint["step"]), int(checkpoint["tokens_seen"])


def resolve_resume_path(output_dir: Path, resume: str | None) -> Path | None:
    if not resume:
        return None
    if resume.lower() in {"1", "true", "latest"}:
        latest_path = output_dir / "latest.json"
        latest = json.loads(latest_path.read_text(encoding="utf-8"))
        return Path(latest["checkpoint"])
    return Path(resume)


def prune_checkpoints(output_dir: Path, keep_last: int) -> None:
    if keep_last <= 0:
        return
    checkpoints = sorted(output_dir.glob("step_*.pt"))
    for checkpoint in checkpoints[:-keep_last]:
        checkpoint.unlink()


def save_checkpoint(
    model: object,
    optimizer: object,
    output_dir: Path,
    step: int,
    tokens_seen: int,
    rank: int,
    keep_last: int,
) -> None:
    import torch
    import torch.distributed as dist

    checkpoint_path = output_dir / f"step_{step:06d}.pt"
    model_state, optimizer_state = checkpoint_state_dict(model, optimizer)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"step": step, "tokens_seen": tokens_seen, "model": model_state, "optimizer": optimizer_state},
            checkpoint_path,
        )
        latest_path = output_dir / "latest.json"
        latest_path.write_text(json.dumps({"step": step, "tokens_seen": tokens_seen, "checkpoint": str(checkpoint_path)}))
        prune_checkpoints(output_dir, keep_last)
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def loss_for_batch(model: object, batch: object, device: object, checkpoint_blocks: bool) -> object:
    import torch
    import torch.nn.functional as F

    batch_tensor = torch.from_numpy(batch).to(device=device, dtype=torch.long, non_blocking=True)
    input_ids = batch_tensor[:, :-1]
    labels = batch_tensor[:, 1:]
    logits = model(input_ids, checkpoint_blocks=checkpoint_blocks)
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))


def validate(
    model: object,
    manifest_path: Path,
    source_weights: dict[str, float],
    seq_len: int,
    batch_size: int,
    batches_per_source: int,
    seed: int,
    device: object,
    rank: int,
) -> tuple[dict[str, float], dict[str, float]]:
    import torch
    import torch.distributed as dist

    model.eval()
    source_losses: dict[str, float] = {}
    source_buckets: dict[str, str] = {}
    with torch.no_grad():
        for source_name in source_weights:
            sampler = RandomTokenSampler(manifest_path, {source_name: 1.0}, seq_len=seq_len, seed=seed + rank)
            source_buckets.update(sampler.source_buckets())
            total = torch.zeros(2, device=device)
            for _ in range(batches_per_source):
                batch, _ = sampler.batch(batch_size)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    loss = loss_for_batch(model, batch, device, checkpoint_blocks=False)
                total[0] += loss.detach()
                total[1] += 1
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(total, op=dist.ReduceOp.SUM)
            source_losses[source_name] = (total[0] / total[1].clamp_min(1)).item()
    model.train()

    bucket_totals: dict[str, list[float]] = {}
    for source_name, loss in source_losses.items():
        bucket = source_buckets.get(source_name, source_name)
        bucket_totals.setdefault(bucket, [0.0, 0.0])
        bucket_totals[bucket][0] += loss
        bucket_totals[bucket][1] += 1
    bucket_losses = {bucket: total / count for bucket, (total, count) in bucket_totals.items()}
    return source_losses, bucket_losses


def train(config: dict, *, max_steps_override: int | None = None, resume_override: str | None = None) -> None:
    import numpy as np
    import torch
    import torch.distributed as dist

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
    validation_interval = int(training_config.get("validation_interval_tokens", 0))
    validation_steps = max(1, validation_interval // max(1, actual_global_tokens)) if validation_interval else 0
    validation_batches = int(training_config.get("validation_batches_per_source", 2))
    keep_last_checkpoints = int(training_config.get("keep_last_checkpoints", 3))

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
    total_parameters = parameter_count(model)
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
    resume_path = resolve_resume_path(output_dir, resume_override or training_config.get("resume_from_checkpoint"))
    start_step = 0
    tokens_seen = 0
    loaded_step = None
    loaded_tokens = None
    if resume_path is not None:
        loaded_step, loaded_tokens = load_checkpoint_state(model, optimizer, resume_path, device)
        start_step, tokens_seen = resume_progress(
            loaded_step,
            loaded_tokens,
            reset=bool(training_config.get("resume_reset_progress", False)),
        )
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"parameters={total_parameters:,}")
        print(f"world_size={world_size} per_rank_batch={per_rank_batch} actual_global_tokens={actual_global_tokens}")
        print(
            f"total_steps={total_steps} start_step={start_step} warmup_steps={warmup_steps} "
            f"checkpoint_steps={checkpoint_steps} validation_steps={validation_steps}"
        )
        if resume_path is not None:
            print(f"resumed_from={resume_path}")
            if loaded_step is not None and loaded_tokens is not None:
                print(f"checkpoint_progress step={loaded_step} tokens={loaded_tokens}")

    peak_lr = float(training_config["lr"])
    min_lr = peak_lr * float(training_config.get("min_lr_ratio", 0.1))
    grad_clip = float(training_config.get("grad_clip", 1.0))
    checkpoint_blocks = bool(training_config.get("activation_checkpointing", False))
    start_time = time.time()

    for step in range(start_step, total_steps):
        lr = current_lr(step, peak_lr=peak_lr, min_lr=min_lr, warmup_steps=warmup_steps, total_steps=total_steps)
        set_optimizer_lr(optimizer, lr)

        batch, source_counts = sampler.batch(per_rank_batch)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            loss = loss_for_batch(model, batch, device, checkpoint_blocks)
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

        if validation_steps and ((step + 1) % validation_steps == 0 or step + 1 == total_steps):
            source_losses, bucket_losses = validate(
                model=model,
                manifest_path=Path(data_config["manifest"]),
                source_weights={name: float(weight) for name, weight in data_config.get("source_weights", {}).items()},
                seq_len=seq_len,
                batch_size=per_rank_batch,
                batches_per_source=validation_batches,
                seed=int(training_config.get("validation_seed", 9001)) + step,
                device=device,
                rank=rank,
            )
            if rank == 0:
                source_text = " ".join(f"val/source/{name}={loss:.4f}" for name, loss in source_losses.items())
                bucket_text = " ".join(f"val/bucket/{name}={loss:.4f}" for name, loss in bucket_losses.items())
                print(f"step={step + 1}/{total_steps} {source_text} {bucket_text}", flush=True)

        if (step + 1) % checkpoint_steps == 0 or step + 1 == total_steps:
            save_checkpoint(model, optimizer, output_dir, step + 1, tokens_seen, rank, keep_last_checkpoints)

    cleanup_distributed()


def main() -> None:
    parser = argparse.ArgumentParser(description="Native Urdu LM training entrypoint.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--preflight", action="store_true", help="Validate config, tokenizer, manifest, and shards.")
    parser.add_argument("--max-steps", type=int, default=None, help="Override training steps for smoke tests.")
    parser.add_argument("--resume", default=None, help="Checkpoint path, or 'latest' to use output_dir/latest.json.")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.preflight:
        preflight(config)
        return
    train(config, max_steps_override=args.max_steps, resume_override=args.resume)


if __name__ == "__main__":
    main()
