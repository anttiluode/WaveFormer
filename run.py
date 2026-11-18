"""Train three controlled tiny architectures on paired role-memory worlds.

Results are produced only when this script is executed. No result is bundled
with the repository until a genuine run has been performed.
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
from waveformer.data import sample_batch
from waveformer.model import TinyWaveFormer


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model: TinyWaveFormer, length: int, batches: int,
             batch_size: int, seed: int, device: str, ablate: bool = False) -> float:
    model.eval()
    rng = np.random.default_rng(seed)
    hits = total = 0
    for _ in range(batches):
        x, labels = sample_batch(batch_size, length, rng)
        logits = model(x.to(device), ablate_recurrent=ablate)
        hits += int((logits.argmax(-1) == labels.to(device)).sum().item())
        total += batch_size
    return hits / total


def run_variant(variant: str, args: argparse.Namespace, device: str):
    # Identical RNG seeds and fresh world samples across all three arms.
    seed_all(args.seed)
    model = TinyWaveFormer(variant, width=args.width, depth=args.depth).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    rng = np.random.default_rng(args.seed + 10000)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    loss_ema = None
    t0 = time.perf_counter()
    for step in range(1, args.steps + 1):
        model.train()
        x, y = sample_batch(args.batch_size, args.train_length, rng)
        logits = model(x.to(device))
        loss = F.cross_entropy(logits, y.to(device))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        lv = float(loss.item())
        loss_ema = lv if loss_ema is None else 0.96 * loss_ema + 0.04 * lv
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            print(f"{variant:9s} step {step:5d}/{args.steps} loss={lv:.4f} ema={loss_ema:.4f}", flush=True)
    duration = time.perf_counter() - t0
    evals = {}
    ablation = {}
    for length in args.test_lengths:
        # Evaluation data are matched across conditions, lengths and models.
        fixed_eval_seed = args.seed * 100003 + length * 101 + 919
        evals[str(length)] = evaluate(
            model, length, args.eval_batches, args.eval_batch_size,
            fixed_eval_seed, device
        )
        if variant != "attention":
            ablation[str(length)] = evaluate(
                model, length, args.eval_batches, args.eval_batch_size,
                fixed_eval_seed, device, ablate=True
            )
    result = {
        "variant": variant,
        "trainable_parameters": model.trainable_parameters,
        "train_loss_ema": loss_ema,
        "heldout_exact_accuracy": evals,
        "without_recurrent_head": ablation if ablation else None,
        "training_wall_seconds": duration,
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()) if device == "cuda" else None,
    }
    del optimizer, model
    if device == "cuda":
        torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-length", type=int, default=24)
    parser.add_argument("--test-lengths", default="24,48,96")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--width", type=int, default=48)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Two-step plumbing test; not a benchmark.")
    args = parser.parse_args()
    if args.smoke:
        args.steps = 2
        args.batch_size = min(args.batch_size, 8)
        args.eval_batch_size = min(args.eval_batch_size, 8)
        args.eval_batches = 1
        args.test_lengths = [12, 24]
        args.train_length = 12
        args.log_every = 1
    else:
        args.test_lengths = [int(x) for x in args.test_lengths.split(",") if x.strip()]
    if min(args.steps, args.batch_size, args.eval_batch_size, args.eval_batches, args.log_every) < 1:
        parser.error("All steps, batch sizes, evaluation batches and logging interval must be positive")
    if min([args.train_length] + args.test_lengths) < 10:
        parser.error("Sequence lengths must be at least 10")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA device requested but PyTorch CUDA is not available")
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / (("smoke" if args.smoke else "benchmark") + f"_seed{args.seed}.json")
    if path.exists() and not args.overwrite:
        parser.error(f"Refusing to overwrite {path}; use --overwrite to intentionally replace")
    print(f"device={device}, parameter-matched variants, train_length={args.train_length}", flush=True)
    variants = [run_variant(name, args, device) for name in ("attention", "leaky", "wave")]
    report = {
        "status": "smoke-only, no scientific inference" if args.smoke else "trained synthetic-world comparison",
        "task": "Two independently marked value slots; late query A/B; four possible outputs.",
        "controls": "same training examples and held-out seeds; wave versus phase-disabled real recurrence",
        "caveats": [
            "The model does not use pretrained language-model weights.",
            "A single seed is not a replication; run multiple seeds and retain all outcomes.",
            "Python recurrent time loop is not an optimized streaming kernel, so training-time comparisons are not architectural throughput conclusions.",
            "The attention and recurrent heads have similar but non-identical parameter counts; counts are reported.",
            "No hidden state survives between independently sampled examples; recurrence is within each episode.",
            "Only final classification is scored; no autonomous imagination, external world model, or text generation is shown."
        ],
        "seed": args.seed,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "train_length": args.train_length,
        "test_lengths": args.test_lengths,
        "eval_examples_per_length": args.eval_batches * args.eval_batch_size,
        "device": device,
        "variants": variants,
    }
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)
    print(json.dumps({"path": str(path), "results": [
        {"variant": v["variant"], "parameters": v["trainable_parameters"],
         "heldout": v["heldout_exact_accuracy"],
         "ablation": v["without_recurrent_head"]} for v in variants
    ]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
