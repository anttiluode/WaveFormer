"""Strict causal reach, role-memory generator and recurrent interventions."""
import unittest
import numpy as np
import torch
from waveformer.v2_data import sample_far_batch
from waveformer.data import WRITE_A, WRITE_B, ASK_A, ASK_B
from waveformer.v2_model import LocalWaveFormer, WindowAttentionHead


class LocalMemoryTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)

    def test_window_attention_is_causal_and_local(self):
        x = torch.randn(2, 12, 24)
        head = WindowAttentionHead(24, 12, window=3).eval()
        with torch.no_grad():
            a = head(x)
            later = x.clone()
            later[:, 8:] = torch.randn_like(later[:, 8:])
            self.assertTrue(torch.allclose(a[:, :8], head(later)[:, :8], atol=1e-7))
            old = x.clone()
            old[:, :4] = torch.randn_like(old[:, :4])
            self.assertTrue(torch.allclose(a[:, 7:], head(old)[:, 7:], atol=1e-7))

    def test_local_two_block_model_cannot_see_remote_value(self):
        model = LocalWaveFormer("local_attention", width=24, window=3).eval()
        x, _ = sample_far_batch(8, 20, np.random.default_rng(51), min_value_gap=7)
        other = x.clone()
        other[:, :10] = (other[:, :10] + 1) % 4
        with torch.no_grad():
            self.assertTrue(torch.allclose(model(x), model(other), atol=1e-6))

    def test_generator_writes_remain_outside_receptive_field(self):
        rng = np.random.default_rng(98)
        x, y = sample_far_batch(250, 24, rng, min_value_gap=9)
        self.assertEqual(tuple(x.shape), (250,24))
        for row,label in zip(x.tolist(), y.tolist()):
            self.assertEqual(row.count(WRITE_A), 1)
            self.assertEqual(row.count(WRITE_B), 1)
            self.assertGreaterEqual((len(row)-1) - (row.index(WRITE_A)+1), 9)
            self.assertGreaterEqual((len(row)-1) - (row.index(WRITE_B)+1), 9)
            self.assertIn(row[-1], (ASK_A, ASK_B))
            v_a, v_b = row[row.index(WRITE_A)+1],row[row.index(WRITE_B)+1]
            self.assertNotEqual(v_a,v_b)
            self.assertEqual(label,v_a if row[-1]==ASK_A else v_b)

    def test_local_attention_is_identical_after_remote_swaps(self):
        x,_ = sample_far_batch(4,24,np.random.default_rng(2),min_value_gap=9)
        x2=x.clone()
        # Replace every distant token while preserving the last 9 tokens and query.
        x2[:,:-9] = torch.randint(0,4, x2[:,:-9].shape)
        for variant in ("local_attention","leaky","wave"):
            model=LocalWaveFormer(variant,width=24,window=4).eval()
            with torch.no_grad():
                intact=model(x)
                changed=model(x2)
                ablated=model(x,ablate_recurrent=True)
                ablated_changed=model(x2,ablate_recurrent=True)
            if variant=="local_attention":
                self.assertTrue(torch.allclose(intact,changed,atol=1e-6))
            else:
                # Removing the only long-range pathway makes distant edits invisible.
                self.assertTrue(torch.allclose(ablated,ablated_changed,atol=1e-6))
                self.assertGreater(float((intact-changed).abs().max()),1e-8)

    def test_wave_phase_gradient_and_parameter_counts(self):
        x,_=sample_far_batch(3,16,np.random.default_rng(9),min_value_gap=7)
        counts=[]
        for kind in ("full_attention","local_attention","leaky","wave"):
            m=LocalWaveFormer(kind,width=24,window=3)
            out=m(x)
            self.assertEqual(tuple(out.shape),(3,4))
            self.assertTrue(bool(torch.isfinite(out).all()))
            counts.append(m.trainable_parameters)
            if kind=="wave":
                out.square().mean().backward()
                self.assertGreater(float(m.blocks[-1].second.phase.grad.abs().sum()),0)
        self.assertLess((max(counts)-min(counts))/max(counts),0.15)


if __name__=="__main__":
    unittest.main()
