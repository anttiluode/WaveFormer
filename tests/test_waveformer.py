"""Fast CPU causal, gradient, control and data checks; no pretrained weights."""
import unittest
import numpy as np
import torch
from waveformer.data import sample_batch, WRITE_A, WRITE_B, ASK_A, ASK_B
from waveformer.model import TinyWaveFormer, RecurrentHead, AttentionHead

class WaveFormerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)

    def test_head_shapes_and_future_is_causal(self):
        for cls in (lambda: AttentionHead(24, 12),
                    lambda: RecurrentHead(24, 12, oscillatory=False),
                    lambda: RecurrentHead(24, 12, oscillatory=True)):
            head = cls().eval()
            x = torch.randn(2, 9, 24)
            y = x.clone()
            y[:, 6:] = torch.randn_like(y[:, 6:]) * 10
            with torch.no_grad():
                original, modified = head(x), head(y)
            self.assertEqual(original.shape, (2, 9, 12))
            self.assertTrue(torch.allclose(original[:, :6], modified[:, :6], atol=1e-6))

    def test_recurrent_step_matches_full_and_chunked(self):
        for wave in (False, True):
            head = RecurrentHead(24, 12, oscillatory=wave).eval()
            x = torch.randn(3, 12, 24)
            with torch.no_grad():
                full, final = head(x, return_state=True)
                first, state = head(x[:, :5], return_state=True)
                second, finish = head(x[:, 5:], initial_state=state, return_state=True)
            self.assertTrue(torch.allclose(full, torch.cat([first, second], 1), atol=1e-6))
            self.assertTrue(torch.allclose(final, finish, atol=1e-6))

    def test_wave_phase_has_gradient(self):
        head = RecurrentHead(24, 12, oscillatory=True)
        x = torch.randn(2, 8, 24)
        loss = head(x).square().mean()
        loss.backward()
        self.assertIsNotNone(head.phase.grad)
        self.assertGreater(float(head.phase.grad.abs().sum()), 0)

    def test_variants_and_ablation(self):
        tokens, _ = sample_batch(3, 12, np.random.default_rng(5))
        for kind in ("attention", "leaky", "wave"):
            model = TinyWaveFormer(kind, width=24, depth=2)
            out = model(tokens)
            self.assertEqual(tuple(out.shape), (3, 4))
            self.assertTrue(bool(torch.isfinite(out).all()))
            if kind != "attention":
                self.assertFalse(torch.allclose(out, model(tokens, ablate_recurrent=True)))
        counts = [TinyWaveFormer(kind, width=24).trainable_parameters
                  for kind in ("attention", "leaky", "wave")]
        self.assertLess((max(counts) - min(counts)) / max(counts), 0.15)

    def test_balanced_writes_and_labels(self):
        x, y = sample_batch(400, 24, np.random.default_rng(88))
        self.assertEqual(x.shape, (400, 24))
        self.assertTrue(bool((y >= 0).all() and (y < 4).all()))
        for row, target in zip(x.tolist(), y.tolist()):
            self.assertEqual(row.count(WRITE_A), 1)
            self.assertEqual(row.count(WRITE_B), 1)
            a, b = row[row.index(WRITE_A)+1], row[row.index(WRITE_B)+1]
            self.assertNotEqual(a, b)
            self.assertIn(row[-1], (ASK_A, ASK_B))
            self.assertEqual(target, a if row[-1] == ASK_A else b)

if __name__ == "__main__":
    unittest.main()
