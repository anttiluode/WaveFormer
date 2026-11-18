"""V2: causal local attention + optional genuinely cross-token recurrent head.

The v1 implementation remains unchanged. With two depth-2 attention blocks
and a window W, a query cannot directly receive a value more than
2*(W-1) positions away. Long-range content must use the recurrent branch.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from .model import AttentionHead, RecurrentHead, rotate_positions, VOCAB_SIZE, NUM_LABELS


class WindowAttentionHead(AttentionHead):
    def __init__(self, width: int, head_width: int, window: int | None):
        super().__init__(width, head_width)
        if window is not None and window < 1:
            raise ValueError("Window must be positive")
        self.window = window

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = rotate_positions(self.q(x)), rotate_positions(self.k(x)), self.v(x)
        scores = (q @ k.transpose(-1, -2)) * self.scale
        t = x.shape[1]
        i = torch.arange(t, device=x.device)
        delta = i[:, None] - i[None, :]
        allowed = (delta >= 0)
        if self.window is not None:
            allowed = allowed & (delta < self.window)
        scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
        return F.softmax(scores, dim=-1) @ v


class WindowHybridBlock(nn.Module):
    def __init__(self, width: int, head_width: int, window: int | None,
                 recurrence: str | None = None):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.attend = WindowAttentionHead(width, head_width, window)
        self.second = (
            RecurrentHead(width, head_width, oscillatory=(recurrence == "wave"))
            if recurrence else WindowAttentionHead(width, head_width, window)
        )
        self.mix = nn.Linear(2 * head_width, width, bias=False)
        self.ff = nn.Sequential(nn.Linear(width, 2 * width), nn.GELU(),
                                nn.Linear(2 * width, width))

    def forward(self, x: torch.Tensor, ablate_recurrent: bool = False) -> torch.Tensor:
        h = self.norm1(x)
        a, b = self.attend(h), self.second(h)
        if ablate_recurrent and isinstance(self.second, RecurrentHead):
            b = torch.zeros_like(b)
        x = x + self.mix(torch.cat((a, b), dim=-1))
        return x + self.ff(self.norm2(x))


class LocalWaveFormer(nn.Module):
    """Four controlled model families; all but full_attention use local heads.

    full_attention: global causal attention throughout (oracle architecture)
    local_attention: all attention limited to window W
    leaky: local attention + one long-range real recurrent head (last block)
    wave: local attention + one long-range phase-rotating head (last block)
    """
    VARIANTS = {"full_attention", "local_attention", "leaky", "wave"}

    def __init__(self, variant: str, width: int = 48, depth: int = 2,
                 window: int = 4, vocab_size: int = VOCAB_SIZE,
                 num_labels: int = NUM_LABELS):
        super().__init__()
        if variant not in self.VARIANTS:
            raise ValueError(f"Unknown variant: {variant}")
        if depth != 2:
            raise ValueError("V2 receptive-field proof and controls require depth=2")
        if width < 8 or width % 4:
            raise ValueError("Width must be divisible by 4")
        if window < 1:
            raise ValueError("window must be positive")
        self.variant, self.window, self.depth = variant, window, depth
        self.embed = nn.Embedding(vocab_size, width)
        self.blocks = nn.ModuleList([
            WindowHybridBlock(
                width, width//2,
                window=None if variant == "full_attention" else window,
                recurrence=variant if variant in {"leaky", "wave"} and index == depth-1 else None,
            ) for index in range(depth)
        ])
        self.norm = nn.LayerNorm(width)
        self.classifier = nn.Linear(width, num_labels)

    @property
    def guaranteed_no_attention_reach(self) -> int:
        """Any input > 2*(window-1) positions before query is unreachable."""
        return self.depth * (self.window - 1)

    def forward(self, token_ids: torch.Tensor, ablate_recurrent: bool = False) -> torch.Tensor:
        x = self.embed(token_ids)
        for block in self.blocks:
            x = block(x, ablate_recurrent=ablate_recurrent)
        return self.classifier(self.norm(x[:, -1]))

    @property
    def trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
