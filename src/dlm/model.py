"""One decoder transformer for both models. Only the attention pattern differs (ADR-0002)."""

import math
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

AttentionKind = Literal["causal", "bidirectional", "block_causal"]


def attention_mask(kind: str, t: int, block: int = 0) -> Tensor:
    """Boolean (t, t) matrix, True where query row i may attend to key column j."""
    if kind == "causal":
        return torch.tril(torch.ones(t, t, dtype=torch.bool))
    if kind == "bidirectional":
        return torch.ones(t, t, dtype=torch.bool)
    if kind == "block_causal":
        if block <= 0:
            raise ValueError("block_causal needs a positive block size")
        blocks = torch.arange(t) // block
        return blocks[None, :] <= blocks[:, None]
    raise ValueError(f"unknown attention kind {kind!r}")


def _rope_tables(t: int, head_dim: int, base: float = 10000.0) -> tuple[Tensor, Tensor]:
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    ang = torch.outer(torch.arange(t).float(), inv)
    return ang.cos(), ang.sin()


def _apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
    return out.flatten(-2)


class Attention(nn.Module):
    def __init__(self, d_model: int, n_head: int, kind: str, block: int) -> None:
        super().__init__()
        assert d_model % n_head == 0 and (d_model // n_head) % 2 == 0
        self.n_head, self.kind, self.block = n_head, kind, block
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        b, t, c = x.shape
        q, k, v = self.qkv(x).view(b, t, 3, self.n_head, c // self.n_head).unbind(2)
        q, k, v = (a.transpose(1, 2) for a in (q, k, v))
        q, k = _apply_rope(q, cos[:t], sin[:t]), _apply_rope(k, cos[:t], sin[:t])
        if self.kind == "causal":
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        elif self.kind == "bidirectional":
            y = F.scaled_dot_product_attention(q, k, v)
        else:
            m = attention_mask(self.kind, t, self.block).to(x.device)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=m)
        out: Tensor = self.proj(y.transpose(1, 2).reshape(b, t, c))
        return out


class Block(nn.Module):
    def __init__(self, d_model: int, n_head: int, kind: str, block: int) -> None:
        super().__init__()
        self.n1, self.n2 = nn.RMSNorm(d_model), nn.RMSNorm(d_model)
        self.attn = Attention(d_model, n_head, kind, block)
        self.fc = nn.Linear(d_model, 4 * d_model, bias=False)
        self.out = nn.Linear(4 * d_model, d_model, bias=False)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        x = x + self.attn(self.n1(x), cos, sin)
        y: Tensor = x + self.out(F.gelu(self.fc(self.n2(x))))
        return y


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        n_layer: int,
        n_head: int,
        d_model: int,
        attention: str,
        attn_block: int = 0,
        abs_pos: bool = False,
    ) -> None:
        super().__init__()
        self.block_size = block_size
        self.wte = nn.Embedding(vocab_size, d_model)
        # Optional learned absolute positions. Off for real runs (RoPE only); the toy
        # memorisation task needs them because bidirectional RoPE has no position anchor.
        self.wpe = nn.Embedding(block_size, d_model) if abs_pos else None
        self.blocks = nn.ModuleList(
            Block(d_model, n_head, attention, attn_block) for _ in range(n_layer)
        )
        self.norm = nn.RMSNorm(d_model)
        cos, sin = _rope_tables(block_size, d_model // n_head)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.apply(self._init)
        for name, p in self.named_parameters():
            if name.endswith(("attn.proj.weight", "out.weight")):
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * n_layer))

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear | nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx: Tensor) -> Tensor:
        if idx.shape[1] > self.block_size:
            raise ValueError("sequence longer than block_size")
        x = self.wte(idx)
        if self.wpe is not None:
            x = x + self.wpe(torch.arange(idx.shape[1], device=idx.device))
        cos, sin = self.get_buffer("cos"), self.get_buffer("sin")
        for blk in self.blocks:
            x = blk(x, cos, sin)
        logits: Tensor = self.norm(x) @ self.wte.weight.T  # tied output embedding
        return logits
