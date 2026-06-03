"""Generate fixed Urdu samples from a trained checkpoint."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import sentencepiece as spm
import yaml


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_checkpoint(config: dict[str, Any], checkpoint: str) -> Path:
    if checkpoint != "latest":
        return Path(checkpoint)
    latest_path = Path(config["infrastructure"]["output_dir"]) / "latest.json"
    latest = json.loads(latest_path.read_text(encoding="utf-8"))
    return Path(latest["checkpoint"])


def load_prompts(path: Path) -> list[str]:
    prompts = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            prompts.append(line)
    if not prompts:
        raise ValueError(f"No prompts found in {path}")
    return prompts


def load_model(config: dict[str, Any], checkpoint_path: Path, device: object) -> object:
    import torch

    from training.model import LlamaDecoder

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = LlamaDecoder(config["model"], int(config["tokenizer"]["vocab_size"]))
    model.load_state_dict(checkpoint["model"])
    if device.type == "cuda":
        model.to(device=device, dtype=torch.bfloat16)
    else:
        model.to(device)
    model.eval()
    return model


def sample_next_token(logits: object, temperature: float, top_p: float, top_k: int) -> int:
    import torch

    logits = logits.float()
    if temperature <= 0:
        return int(torch.argmax(logits).item())
    logits = logits / temperature
    if top_k > 0:
        values, _ = torch.topk(logits, min(top_k, logits.numel()))
        logits = torch.where(logits < values[-1], torch.full_like(logits, float("-inf")), logits)
    probabilities = torch.softmax(logits, dim=-1)
    if 0 < top_p < 1:
        sorted_probs, sorted_indices = torch.sort(probabilities, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        keep = cumulative <= top_p
        keep[0] = True
        filtered = torch.zeros_like(probabilities)
        filtered[sorted_indices[keep]] = probabilities[sorted_indices[keep]]
        probabilities = filtered / filtered.sum()
    return int(torch.multinomial(probabilities, num_samples=1).item())


def generate(
    *,
    model: object,
    tokenizer: object,
    prompt: str,
    device: object,
    seq_len: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
) -> str:
    import torch

    bos_id = tokenizer.bos_id()
    eos_id = tokenizer.eos_id()
    token_ids = tokenizer.encode(prompt, out_type=int)
    if bos_id >= 0:
        token_ids = [bos_id] + token_ids

    with torch.no_grad():
        for _ in range(max_new_tokens):
            context = token_ids[-seq_len:]
            input_ids = torch.tensor([context], dtype=torch.long, device=device)
            logits = model(input_ids)[0, -1]
            next_id = sample_next_token(logits, temperature=temperature, top_p=top_p, top_k=top_k)
            if next_id == eos_id:
                break
            token_ids.append(next_id)
    return tokenizer.decode(token_ids[1:] if bos_id >= 0 and token_ids and token_ids[0] == bos_id else token_ids)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Urdu samples from a native LM checkpoint.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", default="latest", help="Checkpoint path or 'latest'.")
    parser.add_argument("--prompts", type=Path, default=Path("eval/prompts_urdu.txt"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import torch

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    config = load_config(args.config)
    checkpoint_path = resolve_checkpoint(config, args.checkpoint)
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(str(config["tokenizer"]["path"]))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_model(config, checkpoint_path, device)
    seq_len = int(config["data"]["sequence_length"])
    prompts = load_prompts(args.prompts)

    output_path = args.output
    if output_path is None:
        output_path = Path(config["infrastructure"]["output_dir"]) / "samples.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        for prompt in prompts:
            text = generate(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                device=device,
                seq_len=seq_len,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
            )
            record = {
                "prompt": prompt,
                "completion": text[len(prompt) :].strip() if text.startswith(prompt) else text,
                "text": text,
                "checkpoint": str(checkpoint_path),
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
            }
            print(json.dumps(record, ensure_ascii=False))
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Wrote samples to {output_path}")


if __name__ == "__main__":
    main()
