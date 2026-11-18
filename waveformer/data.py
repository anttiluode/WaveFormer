"""Balanced, newly sampled two-slot delayed-retrieval worlds.

Vocabulary 0..3: value symbols, 4/5: WRITE_A/WRITE_B, 6/7: ASK_A/ASK_B,
8/9: neutral distractors. Every sample contains both writes, each followed
by one value; the final query is chosen independently of target identity.
"""
from __future__ import annotations
import numpy as np
import torch

VALUES = 4
WRITE_A, WRITE_B, ASK_A, ASK_B, FILL_0, FILL_1 = 4, 5, 6, 7, 8, 9


def sample_batch(batch_size: int, length: int, rng: np.random.Generator):
    if length < 10:
        raise ValueError("length must be at least 10")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    sequence = np.empty((batch_size, length), dtype=np.int64)
    label = np.empty(batch_size, dtype=np.int64)
    for i in range(batch_size):
        tokens = rng.choice([0, 1, 2, 3, FILL_0, FILL_1], size=length, replace=True)
        a = int(rng.integers(VALUES))
        b = int((a + rng.integers(1, VALUES)) % VALUES)
        p_a = int(rng.integers(0, length - 2))
        p_b = int(rng.integers(0, length - 2))
        while abs(p_a - p_b) < 2:
            p_b = int(rng.integers(0, length - 2))
        tokens[p_a:p_a+2] = [WRITE_A, a]
        tokens[p_b:p_b+2] = [WRITE_B, b]
        ask_a = bool(rng.integers(2))
        tokens[-1] = ASK_A if ask_a else ASK_B
        label[i] = a if ask_a else b
        sequence[i] = tokens
    return torch.from_numpy(sequence), torch.from_numpy(label)
