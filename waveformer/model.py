"""Tiny, causal hybrid attention/recurrent model. No wave advantage is presumed."""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F

VOCAB_SIZE = 10
NUM_LABELS = 4


def rotate_positions(t: torch.Tensor) -> torch.Tensor:
    """RoPE on the last dimension of a (batch,time,even_width) tensor."""
    width = t.shape[-1]
    if width % 2:
        raise ValueError("RoPE width must be even")
    positions = torch.arange(t.shape[1], device=t.device, dtype=torch.float32)
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(0, width, 2, device=t.device, dtype=torch.float32) / width
    )
    ang = positions[:, None] * freqs[None, :]
    c, s = ang.cos().to(t.dtype), ang.sin().to(t.dtype)
    even, odd = t[..., ::2], t[..., 1::2]
    return torch.stack((even*c-odd*s, even*s+odd*c), dim=-1).flatten(-2)


class AttentionHead(nn.Module):
    """One ordinary, masked Q/K/V head with RoPE."""
    def __init__(self, width: int, head_width: int):
        super().__init__()
        if head_width % 2:
            raise ValueError("Head width must be even")
        self.q = nn.Linear(width, head_width, bias=False)
        self.k = nn.Linear(width, head_width, bias=False)
        self.v = nn.Linear(width, head_width, bias=False)
        self.scale = head_width ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = rotate_positions(self.q(x)), rotate_positions(self.k(x)), self.v(x)
        scores = (q @ k.transpose(-1, -2)) * self.scale
        t = x.shape[1]
        causal = torch.ones(t, t, device=x.device, dtype=torch.bool).tril()
        scores = scores.masked_fill(~causal, torch.finfo(scores.dtype).min)
        return F.softmax(scores, dim=-1) @ v


class RecurrentHead(nn.Module):
    """Parallel real (leaky) or coupled phase-rotating state pairs.

    Learned decay, input, input gate, read gate and output transformation are
    shared by both variants. Only the wave variant rotates each state pair.
    Computation is deliberately token-sequential; no speed claim is made.
    """
    def __init__(self, width: int, head_width: int, oscillatory: bool):
        super().__init__()
        if head_width % 2:
            raise ValueError("Head width must be even")
        self.oscillatory = oscillatory
        self.head_width = head_width
        self.write = nn.Linear(width, head_width, bias=False)
        self.write_gate = nn.Linear(width, head_width, bias=False)
        self.read_gate = nn.Linear(width, head_width, bias=False)
        self.readout = nn.Linear(head_width, head_width, bias=False)
        self.decay_logits = nn.Parameter(torch.linspace(1.5, 4.0, head_width))
        self.phase = nn.Parameter(torch.linspace(0.05, 0.8, head_width//2))
        if not oscillatory:
            self.phase.requires_grad_(False)

    def zero_state(self, batch: int, ref: torch.Tensor) -> torch.Tensor:
        return ref.new_zeros(batch, self.head_width)

    def step(self, x_t: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if state.shape != (x_t.shape[0], self.head_width):
            raise ValueError("Bad recurrent state shape")
        decay = torch.sigmoid(self.decay_logits).to(state.dtype)
        if self.oscillatory:
            a, b = state[..., ::2], state[..., 1::2]
            theta = torch.tanh(self.phase).to(state.dtype) * math.pi
            c, s = theta.cos(), theta.sin()
            state = torch.stack((a*c-b*s, a*s+b*c), dim=-1).flatten(-2)
        drive = torch.tanh(self.write(x_t)) * torch.sigmoid(self.write_gate(x_t))
        state = decay * state + drive
        output = self.readout(torch.tanh(state)) * torch.sigmoid(self.read_gate(x_t))
        return output, state

    def forward(self, x: torch.Tensor, initial_state: torch.Tensor | None = None,
                return_state: bool = False):
        state = self.zero_state(x.shape[0], x) if initial_state is None else initial_state
        outputs = []
        for t in range(x.shape[1]):
            y, state = self.step(x[:, t], state)
            outputs.append(y)
        values = torch.stack(outputs, dim=1)
        return (values, state) if return_state else values


class HybridBlock(nn.Module):
    def __init__(self, width: int, head_width: int, variant: str, use_recurrent: bool):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.attend = AttentionHead(width, head_width)
        self.second = (RecurrentHead(width, head_width, oscillatory=(variant == "wave"))
                       if use_recurrent else AttentionHead(width, head_width))
        self.mix = nn.Linear(head_width * 2, width, bias=False)
        self.ff = nn.Sequential(nn.Linear(width, width * 2), nn.GELU(),
                                nn.Linear(width * 2, width))

    def forward(self, x: torch.Tensor, ablate_recurrent: bool = False) -> torch.Tensor:
        h = self.norm1(x)
        a, b = self.attend(h), self.second(h)
        if ablate_recurrent and isinstance(self.second, RecurrentHead):
            b = torch.zeros_like(b)
        x = x + self.mix(torch.cat((a, b), dim=-1))
        return x + self.ff(self.norm2(x))


class TinyWaveFormer(nn.Module):
    """Replace exactly one head of the final two-head block."""
    def __init__(self, variant: str, width: int = 48, depth: int = 2,
                 vocab_size: int = VOCAB_SIZE, num_labels: int = NUM_LABELS):
        super().__init__()
        if variant not in {"attention", "leaky", "wave"}:
            raise ValueError("variant must be attention, leaky, or wave")
        if width < 8 or width % 4:
            raise ValueError("Model width must be a multiple of 4")
        if depth < 1:
            raise ValueError("Depth must be >= 1")
        self.variant = variant
        self.embed = nn.Embedding(vocab_size, width)
        self.blocks = nn.ModuleList([
            HybridBlock(width, width//2, variant, use_recurrent=(variant != "attention" and i == depth-1))
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(width)
        self.classifier = nn.Linear(width, num_labels)

    def forward(self, token_ids: torch.Tensor, ablate_recurrent: bool = False) -> torch.Tensor:
        x = self.embed(token_ids)
        for block in self.blocks:
            x = block(x, ablate_recurrent=ablate_recurrent)
        return self.classifier(self.norm(x[:, -1]))

    @property
    def trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
