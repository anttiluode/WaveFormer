"""V2 causal-reach-controlled memory benchmark.

Contrast global attention, strictly local attention, local+leaky recurrence,
and local+phase-rotating recurrence. Writes are deliberately placed outside
the two-layer local attention receptive field. Results are not fabricated.
"""
from __future__ import annotations
import argparse
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from waveformer.v2_data import sample_far_batch
from waveformer.v2_model import LocalWaveFormer

VARIANTS = ("full_attention", "local_attention", "leaky", "wave")


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def eval_variant(model: LocalWaveFormer, length: int, batch_size: int,
                 batches: int, min_gap: int, seed: int, device: str,
                 ablate: bool = False) -> dict:
    model.eval()
    rng = np.random.default_rng(seed)
    n_correct, n = 0, 0
    for _ in range(batches):
        x, y = sample_far_batch(batch_size, length, rng, min_gap)
        x, y = x.to(device), y.to(device)
        prediction = model(x, ablate_recurrent=ablate).argmax(-1)
        n_correct += int((prediction == y).sum())
        n += y.numel()
    return {"correct": n_correct, "n": n, "accuracy": n_correct / n}


def train_one(variant: str, args, device: str) -> dict:
    seed_all(args.seed)
    model = LocalWaveFormer(variant, width=args.width, window=args.window).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    rng = np.random.default_rng(args.seed + 10000)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    tic = time.perf_counter()
    ema = None
    for step in range(1, args.steps+1):
        model.train()
        x, y = sample_far_batch(args.batch_size, args.train_length, rng, args.min_gap)
        logits = model(x.to(device))
        loss = F.cross_entropy(logits, y.to(device))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        v = float(loss.item())
        ema = v if ema is None else 0.96*ema + 0.04*v
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            print(f"{variant:15s} step {step:4d}/{args.steps} loss={v:.4f} ema={ema:.4f}", flush=True)
    training_s = time.perf_counter() - tic
    scores = {}
    for length in args.test_lengths:
        eval_seed = args.seed * 100003 + length * 101 + 919
        scores[str(length)] = {
            "intact": eval_variant(model, length, args.eval_batch_size,
                                   args.eval_batches, args.min_gap, eval_seed, device)
        }
        if variant in {"wave", "leaky"}:
            scores[str(length)]["recurrent_zeroed"] = eval_variant(
                model, length, args.eval_batch_size, args.eval_batches,
                args.min_gap, eval_seed, device, ablate=True
            )
    record = {
        "variant": variant,
        "trainable_parameters": model.trainable_parameters,
        "final_train_ema_loss": ema,
        "accuracy": scores,
        "training_wall_seconds": training_s,
        "peak_cuda_allocated_bytes": (
            int(torch.cuda.max_memory_allocated()) if device == "cuda" else None
        )
    }
    del optimizer, model
    if device == "cuda":
        torch.cuda.empty_cache()
    return record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--width", type=int, default=48)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--train-length", type=int, default=24)
    p.add_argument("--test-lengths", default="24,48,96")
    p.add_argument("--window", type=int, default=4)
    p.add_argument("--min-gap", type=int, default=9)
    p.add_argument("--lr", type=float, default=0.003)
    p.add_argument("--eval-batch-size", type=int, default=64)
    p.add_argument("--eval-batches", type=int, default=8)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--device", choices=("auto","cpu","cuda"), default="auto")
    p.add_argument("--output-dir", type=Path, default=Path("results/v2"))
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.smoke:
        args.steps = 2
        args.batch_size = min(args.batch_size, 8)
        args.eval_batch_size = min(args.eval_batch_size, 8)
        args.eval_batches = 1
        args.train_length = 16
        args.test_lengths = [16, 24]
        args.window = 3
        args.min_gap = 7
        args.log_every = 1
    else:
        args.test_lengths = [int(q) for q in args.test_lengths.split(",") if q.strip()]
    if min(args.steps, args.batch_size, args.eval_batches, args.eval_batch_size, args.log_every) < 1:
        p.error("steps, batches, evaluation counts and logging interval must be positive")
    if args.min_gap <= 2 * (args.window - 1):
        p.error("min-gap must exceed 2*(window-1): otherwise local attention can see the answer")
    if min([args.train_length]+args.test_lengths) < args.min_gap + 4:
        p.error("sequence lengths must exceed min-gap+3 to place two nonoverlapping writes")
    if args.device == "cuda" and not torch.cuda.is_available():
        p.error("CUDA requested but PyTorch does not have CUDA available")
    device = "cuda" if args.device=="auto" and torch.cuda.is_available() else (
        "cpu" if args.device=="auto" else args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dest = args.output_dir / f"{'smoke' if args.smoke else 'benchmark'}_seed{args.seed}.json"
    if dest.exists() and not args.overwrite:
        p.error(f"{dest} exists. Use --overwrite to replace intentionally.")
    print(f"device={device} train_length={args.train_length} local_window={args.window} "
          f"min_gap={args.min_gap}; local-only memory cutoff={2*(args.window-1)}",flush=True)
    variants = [train_one(v, args, device) for v in VARIANTS]
    report = {
        "status":"smoke only" if args.smoke else "source-reported trained synthetic benchmark",
        "seed": args.seed,
        "steps": args.steps,
        "train_length": args.train_length,
        "test_lengths": args.test_lengths,
        "eval_n_each": args.eval_batch_size*args.eval_batches,
        "window": args.window,
        "min_gap": args.min_gap,
        "local_attention_max_reach": 2*(args.window-1),
        "device": device,
        "data":"Two marked writes with distinct values; all writes outside local-query receptive field.",
        "limits":[
            "No pretrained language model, text generation, native persistent external state or imagined environment.",
            "Three seeds are exploratory; a phase advantage needs paired, head-essential comparisons.",
            "Full-attention is an architectural reference, while local-attention proves the local-only task is impossible beyond the cutoff.",
            "Every variant trains on the same sampled world batches and is tested on the same fresh examples.",
            "The recurrent head is sequential Python, not an optimized kernel; do not infer GPU speed advantages."
        ],
        "variants":variants
    }
    temp = dest.with_suffix(".tmp")
    temp.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    temp.replace(dest)
    print(json.dumps({"path":str(dest),"results":[{
        "variant":v["variant"],
        "params":v["trainable_parameters"],
        "accuracy":v["accuracy"],
        "training_seconds":round(v["training_wall_seconds"],2)
    } for v in variants]},indent=2),flush=True)


if __name__ == "__main__":
    main()
