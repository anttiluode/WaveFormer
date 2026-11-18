"""V2 two-role classification episodes with provably inaccessible distant writes.

Local-only depth-2, window-W attention at final query reads at most the past
2*(W-1) tokens. Setting min_value_gap above this radius guarantees the last
query cannot directly see either write/value marker through local attention.
"""
from __future__ import annotations
import numpy as np
import torch
from .data import VALUES, WRITE_A, WRITE_B, ASK_A, ASK_B, FILL_0, FILL_1


def sample_far_batch(batch_size: int, length: int, rng: np.random.Generator,
                     min_value_gap: int) -> tuple[torch.Tensor, torch.Tensor]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if min_value_gap < 1:
        raise ValueError("min_value_gap must be >= 1")
    # Position of write marker can go up to: last index - gap - one value token.
    upper = length - 2 - min_value_gap
    if upper < 2:
        raise ValueError("sequence length too small for two distant nonoverlapping writes")
    tokens = np.empty((batch_size, length), dtype=np.int64)
    labels = np.empty(batch_size, dtype=np.int64)
    for i in range(batch_size):
        row = rng.choice([0, 1, 2, 3, FILL_0, FILL_1], size=length)
        a = int(rng.integers(VALUES))
        b = int((a + rng.integers(1, VALUES)) % VALUES)
        p_a = int(rng.integers(0, upper + 1))
        p_b = int(rng.integers(0, upper + 1))
        while abs(p_a - p_b) < 2:
            p_b = int(rng.integers(0, upper + 1))
        row[p_a:p_a+2] = [WRITE_A, a]
        row[p_b:p_b+2] = [WRITE_B, b]
        ask_a = bool(rng.integers(2))
        row[-1] = ASK_A if ask_a else ASK_B
        tokens[i] = row
        labels[i] = a if ask_a else b
    return torch.from_numpy(tokens), torch.from_numpy(labels)
