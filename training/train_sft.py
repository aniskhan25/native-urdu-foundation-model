"""Supervised fine-tuning entrypoint for prompt/response JSONL data."""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Any

import sentencepiece as spm
import yaml

from training.model import LlamaDecoder, parameter_count
from training.sft_dataset import IGNORE_INDEX, SftBatchSampler, SftDataset, dataset_size
from training.train import cleanup_distributed
from training.train import current_lr
from training.train import init_distributed
from training.train import load_checkpoint_state
from training.train import maybe_wrap_fsdp
from training.train import resolve_resume_path
from training.train import save_checkpoint
from training.train import set_optimizer_lr


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def preflight(config: dict[str, Any]) -> None:
    tokenizer_path = Path(config["tokenizer"]["path"])
    train_path = Path(config["data"]["train_jsonl"])
    validation_path = config["data"].get("validation_jsonl")
    base_checkpoint = Path(config["training"]["base_checkpoint"])
    for path in [tokenizer_path, train_path, base_checkpoint]:
        if not path.is_file():
            raise FileNotFoundError(path)
    if validation_path and not Path(validation_path).is_file():
        raise FileNotFoundError(validation_path)

    print("SFT preflight OK")
    print(f"tokenizer={tokenizer_path}")
    print(f"base_checkpoint={base_checkpoint}")
    print(f"train_jsonl={train_path} examples={dataset_size(train_path)}")
    if validation_path:
        path = Path(validation_path)
        print(f"validation_jsonl={path} examples={dataset_size(path)}")


def load_model_weights(model: object, checkpoint_path: Path) -> None:
    import torch
    import torch.distributed as dist

    checkpoint = torch.load(checkpoint_path, map_location="cpu", mmap=True)
    if dist.is_available() and dist.is_initialized():
        from torch.distributed.fsdp import FullStateDictConfig, FullyShardedDataParallel as FSDP, StateDictType

        if isinstance(model, FSDP):
            state_config = FullStateDictConfig(offload_to_cpu=False, rank0_only=False)
            with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, state_config):
                model.load_state_dict(checkpoint["model"])
            return
    model.load_state_dict(checkpoint["model"])


def loss_for_sft_batch(
    model: object,
    input_ids: object,
    labels: object,
    device: object,
    checkpoint_blocks: bool,
) -> object:
    import torch
    import torch.nn.functional as F

    input_tensor = torch.from_numpy(input_ids).to(device=device, dtype=torch.long, non_blocking=True)
    label_tensor = torch.from_numpy(labels).to(device=device, dtype=torch.long, non_blocking=True)
    logits = model(input_tensor[:, :-1], checkpoint_blocks=checkpoint_blocks)
    targets = label_tensor[:, 1:]
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=IGNORE_INDEX)


def validate(
    model: object,
    dataset: SftDataset,
    batch_size: int,
    batches: int,
    seed: int,
    device: object,
    rank: int,
) -> float:
    import torch
    import torch.distributed as dist

    model.eval()
    sampler = SftBatchSampler(len(dataset), batch_size, seed + rank)
    total = torch.zeros(2, device=device)
    with torch.no_grad():
        for _ in range(batches):
            input_ids, labels = dataset.batch(sampler.batch_indices())
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                loss = loss_for_sft_batch(model, input_ids, labels, device, checkpoint_blocks=False)
            total[0] += loss.detach()
            total[1] += 1
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(total, op=dist.ReduceOp.SUM)
    model.train()
    return (total[0] / total[1].clamp_min(1)).item()


def train(config: dict[str, Any], *, max_steps_override: int | None = None, resume_override: str | None = None) -> None:
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
    global_batch_examples = int(training_config["global_batch_examples"])
    per_rank_batch = max(1, global_batch_examples // max(1, world_size))
    actual_global_examples = per_rank_batch * world_size
    epochs = float(training_config.get("epochs", 1))
    train_examples = dataset_size(Path(data_config["train_jsonl"]))
    total_steps = int(
        max_steps_override
        or training_config.get("estimated_steps")
        or math.ceil(train_examples * epochs / max(1, actual_global_examples))
    )
    warmup_steps = int(training_config.get("warmup_steps") or total_steps * float(training_config.get("warmup_ratio", 0.03)))
    warmup_steps = max(1, warmup_steps)
    checkpoint_steps = int(training_config.get("checkpoint_interval_steps", max(1, total_steps // 4)))
    validation_steps = int(training_config.get("validation_interval_steps", checkpoint_steps))
    validation_batches = int(training_config.get("validation_batches", 8))
    keep_last_checkpoints = int(training_config.get("keep_last_checkpoints", 3))

    seed = int(training_config.get("seed", 42)) + rank
    torch.manual_seed(seed)
    np.random.seed(seed)

    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(str(config["tokenizer"]["path"]))
    train_dataset = SftDataset(Path(data_config["train_jsonl"]), tokenizer, seq_len)
    train_sampler = SftBatchSampler(len(train_dataset), per_rank_batch, seed)
    validation_dataset = None
    if data_config.get("validation_jsonl"):
        validation_dataset = SftDataset(Path(data_config["validation_jsonl"]), tokenizer, seq_len)

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
        weight_decay=float(training_config.get("weight_decay", 0.0)),
    )

    output_dir = Path(infra_config.get("output_dir", "runs/sft"))
    resume_path = resolve_resume_path(output_dir, resume_override or training_config.get("resume_from_checkpoint"))
    start_step = 0
    examples_seen = 0
    if resume_path is not None:
        start_step, examples_seen = load_checkpoint_state(model, optimizer, resume_path, device)
        loaded_from = resume_path
    else:
        loaded_from = Path(training_config["base_checkpoint"])
        load_model_weights(model, loaded_from)
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"parameters={total_parameters:,}")
        print(f"world_size={world_size} per_rank_batch={per_rank_batch} actual_global_examples={actual_global_examples}")
        print(
            f"train_examples={train_examples} total_steps={total_steps} start_step={start_step} "
            f"warmup_steps={warmup_steps} checkpoint_steps={checkpoint_steps} validation_steps={validation_steps}"
        )
        print(f"loaded_from={loaded_from}")

    peak_lr = float(training_config["lr"])
    min_lr = peak_lr * float(training_config.get("min_lr_ratio", 0.1))
    grad_clip = float(training_config.get("grad_clip", 1.0))
    checkpoint_blocks = bool(training_config.get("activation_checkpointing", False))
    start_time = time.time()

    for step in range(start_step, total_steps):
        lr = current_lr(step, peak_lr=peak_lr, min_lr=min_lr, warmup_steps=warmup_steps, total_steps=total_steps)
        set_optimizer_lr(optimizer, lr)

        input_ids, labels = train_dataset.batch(train_sampler.batch_indices())
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            loss = loss_for_sft_batch(model, input_ids, labels, device, checkpoint_blocks)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        examples_seen += actual_global_examples
        if rank == 0 and (step == 0 or (step + 1) % 10 == 0):
            elapsed = max(1e-6, time.time() - start_time)
            examples_per_second = examples_seen / elapsed
            print(
                f"step={step + 1}/{total_steps} loss={loss.item():.4f} lr={lr:.6g} "
                f"examples={examples_seen} ex_s={examples_per_second:.1f}",
                flush=True,
            )

        if validation_dataset is not None and validation_steps and (
            (step + 1) % validation_steps == 0 or step + 1 == total_steps
        ):
            validation_loss = validate(
                model=model,
                dataset=validation_dataset,
                batch_size=per_rank_batch,
                batches=validation_batches,
                seed=int(training_config.get("validation_seed", 9001)) + step,
                device=device,
                rank=rank,
            )
            if rank == 0:
                print(f"step={step + 1}/{total_steps} val/sft_loss={validation_loss:.4f}", flush=True)

        if (step + 1) % checkpoint_steps == 0 or step + 1 == total_steps:
            save_checkpoint(model, optimizer, output_dir, step + 1, examples_seen, rank, keep_last_checkpoints)

    cleanup_distributed()


def main() -> None:
    parser = argparse.ArgumentParser(description="Native Urdu SFT training entrypoint.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--preflight", action="store_true", help="Validate SFT config and input files.")
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
