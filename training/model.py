"""Minimal LLaMA-style decoder model."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return normed * self.weight


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position: int, theta: float) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        positions = torch.arange(max_position, dtype=torch.float)
        freqs = torch.einsum("i,j->ij", positions, inv_freq)
        emb = torch.repeat_interleave(freqs, repeats=2, dim=-1)
        self.register_buffer("cos", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cos = self.cos[:, :, : q.size(-2), :]
        sin = self.sin[:, :, : q.size(-2), :]
        return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


class SelfAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, num_kv_heads: int, max_position: int, rope_theta: float) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads
        self.kv_repeats = num_heads // num_kv_heads
        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=False)
        self.rope = RotaryEmbedding(self.head_dim, max_position, rope_theta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q, k = self.rope(q, k)
        if self.kv_repeats > 1:
            k = k.repeat_interleave(self.kv_repeats, dim=1)
            v = v.repeat_interleave(self.kv_repeats, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(batch, seq_len, self.num_heads * self.head_dim)
        return self.o_proj(y)


class MLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderBlock(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        hidden_size = int(config["hidden_size"])
        self.input_norm = RMSNorm(hidden_size)
        self.self_attn = SelfAttention(
            hidden_size=hidden_size,
            num_heads=int(config["attention_heads"]),
            num_kv_heads=int(config["kv_heads"]),
            max_position=int(config["seq_len"]),
            rope_theta=float(config.get("rope_theta", 10000)),
        )
        self.post_attention_norm = RMSNorm(hidden_size)
        self.mlp = MLP(hidden_size, int(config["intermediate_size"]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.input_norm(x))
        return x + self.mlp(self.post_attention_norm(x))


class LlamaDecoder(nn.Module):
    def __init__(self, config: dict, vocab_size: int) -> None:
        super().__init__()
        hidden_size = int(config["hidden_size"])
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([DecoderBlock(config) for _ in range(int(config["layers"]))])
        self.norm = RMSNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        if config.get("tie_embeddings", True):
            self.lm_head.weight = self.embed_tokens.weight
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, *, checkpoint_blocks: bool = False) -> torch.Tensor:
        if input_ids.size(1) > self.layers[0].self_attn.rope.cos.size(-2):
            raise ValueError("input sequence is longer than configured seq_len")
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            if checkpoint_blocks and self.training:
                x = torch.utils.checkpoint.checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
        return self.lm_head(self.norm(x))


def parameter_count(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())
