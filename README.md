# WaveFormer

**One attention head versus one resonant recurrent head.** A deliberately small, falsifiable comparison proposed from the WaveComputer, GeometricNeuron, TuftSheet, OneTinyTransformer and Qwenimagination research lines.

**V1 status: three trained seed results now committed (42–44).** All arms reach 100% at the 24-token training length. At length 48 the means are attention 76.95%, leaky 78.45%, wave 83.01%; at 96 they are 56.12%, 51.24%, 54.62%. **Zeroing the trained recurrent head changes accuracy by at most ~0.13 percentage points on average**: V1 does **not** establish useful wave computation. Measured training wall-time was roughly 14 s attention, 37 s leaky, 56 s wave per seed on the user's CUDA setup; this compares unoptimized implementations, not optimized throughput.

**V2 status: local-attention cutoff and far-memory experiment implemented; trained results pending.** V1 source and receipts are preserved unchanged.

## V2: make recurrence essential before comparing resonance

V1 trained all three variants on the same synthetic two-slot task, but ablation proved the second recurrent head is nearly unused. This is unsurprising: V1 has two *full-attention* heads in the first block and another full-attention head in the final block. The model can solve the task through those paths.

V2 keeps the V1 files and results intact, but adds a **predefined receptive-field intervention**:

- `full_attention`: two blocks of ordinary global Q/K/V attention, a retrieval reference.
- `local_attention`: two blocks of attention restricted to a window of 4 tokens, no long-range recurrent head.
- `leaky`: the same local heads plus a single real-valued recurrent head in the last block.
- `wave`: the same local heads plus a single damped phase-rotating recurrent head in the last block.

At depth 2 and window 4, local-only attention at the final token can see at most **6 prior positions**. The generator ensures both write/value pairs end **at least 9 positions before the query**. Their information therefore **cannot** reach the final query through local attention alone; if either recurrent hybrid succeeds, its recurrent pathway must be involved. The first block's local head can see the write marker directly before the value token.

The task and token semantics remain the same as V1. Training and evaluation use paired seeds, 24-token training sequences and fresh 24/48/96-token episodes. The runner reports both intact and recurrent-zeroed accuracy on **the exact same examples**. Tests explicitly verify that when the recurrent pathway is removed, modifying distant tokens cannot influence the answer.

```powershell
git pull --ff-only
python3.13 -m unittest discover -s tests -v
python3.13 run_v2.py --smoke
python3.13 run_v2.py --steps 1000 --seed 42
python3.13 run_v2.py --steps 1000 --seed 43
python3.13 run_v2.py --steps 1000 --seed 44
```

Results are written under `results/v2/` and are never allowed to overwrite V1 receipts. A smoke run is only a check that training and evaluation execute.

The V2 **first gate** is that `local_attention` remains near the four-way chance baseline (25%) while a hybrid exceeds it, and zeroing that hybrid's recurrent head removes the advantage. Only after that gate passes is a **wave > leaky** result evidence worth investigating. If the wave arm does not beat leaky on paired seeds, recurrence alone suffices for this task. A full-attention model is retained as an information-access upper reference. A single seed or a narrow training distribution is not a generalization claim.

Note that this benchmark does not implement continuous real-time operating state across episodes, active imagination, or any pretrained-language-model replacement. Training speed comparisons remain implementation-dependent.

## What is actually replaced?

A two-block, two-head-per-block causal transformer is trained from scratch on a synthetic sequence classification task. The final block always keeps its first conventional RoPE attention head. Its second head is one of:

| Arm | Final block's second head |
| --- | --- |
| `attention` | Normal causal Q/K/V head |
| `leaky` | Learned recurrent state with decays but **no oscillatory phase rotation** |
| `wave` | Identical recurrent head with learned phase rotations on state pairs |

All arms use token embeddings, residual pathways, LayerNorm, a GELU feed-forward network, and a final four-way classifier. The first block retains **two attention heads** in every arm. The recurrent arms have the same write, input-gate, read-gate and readout parameterization. Their phase dynamics are the principal controlled difference.

The wave recurrence is

```
s_t = sigmoid(decay) * R(theta) s_(t-1) + tanh(write(x_t)) * sigmoid(write_gate(x_t))
y_t = readout(tanh(s_t)) * sigmoid(read_gate(x_t))
```

`R(theta)` is a block-diagonal rotation over state pairs, with learned angles in the wave arm, and identity in the leaky arm. The recurrence is token-sequential Python, **not an optimized GPU wave solver**. We report total trainable parameter counts and do not assume either recurrent variant is computationally faster than attention.

## The first task: two-slot memory

Each 24-token training episode contains a `WRITE_A, value` pair, a `WRITE_B, different_value` pair, distracting symbols, and a final `ASK_A` or `ASK_B`. The model must output the requested value, one of four possible labels. The two assignments, their order, their locations, and the final question vary. The same deterministic batch stream is replayed for every architecture.

Held-out episodes are freshly generated with a separate RNG seed. Evaluate lengths **24, 48 and 96**, without changing the training length (24) or choosing the best run afterwards. At longer lengths, attention can still read all tokens while the recurrent state compresses their history.

This is deliberately a more demanding role-binding task than reading a single keyword, but much smaller than language modeling or real-world imagination.

## Run locally

Python 3.11+ and PyTorch (GPU optional):

```powershell
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python run.py --smoke
python run.py --steps 800 --seed 42
python run.py --steps 800 --seed 43
python run.py --steps 800 --seed 44
```

On a Windows machine with Python 3.13, replace `python` with `python3.13` if desired. `--smoke` takes two training steps per arm and checks execution only. It cannot measure performance. Regular runs write `results/benchmark_seed42.json` and so on, refusing to overwrite an existing file unless `--overwrite` is specified.

Other parameters: `--batch-size 64 --train-length 24 --test-lengths 24,48,96 --width 48 --depth 2 --device auto`.

## Planned success and failure criteria

Before reviewing any trained results, the following comparisons matter:

1. Does `wave` outperform `leaky` on **fresh longer sequences**, over multiple seeds? If not, wave-specific dynamics have not earned their place.
2. Does either recurrent hybrid match or exceed `attention` under roughly matched parameter count and training-example budget?
3. Does zeroing the trained recurrent head at evaluation reduce accuracy? If not, the preserved attention head may be doing all the work.
4. Are results robust to changed symbol mappings, query and write distributions, and longer gaps? The current generator does **not yet** test all those shifts.

Report every seed including failures. The first experiment is a benchmark of a small hybrid attention/recurrent design, **not a claim of new state-of-the-art language modeling or persistent inner imagination**.

A future version may connect the Qwenimagination concept reader to a learned persistent state; that is not implemented here. Previous [WaveCortex](https://github.com/anttiluode/WaveCortex) and [WaveComputerAstra](https://github.com/anttiluode/WaveComputerAstra) wave machines often lost to scalar controls, so a nonoscillatory baseline is mandatory.

## Research lineage

- [OneTinyTransformer](https://github.com/anttiluode/OneTinyTransformer): inspectable causal attention and RoPE baseline.
- [GeometricNeuronV30](https://github.com/anttiluode/GeometricNeuronV30): decay, delay and phase-bearing memory.
- [WaveComputer](https://github.com/anttiluode/WaveComputer): content separated from prediction error.
- [WaveCortex](https://github.com/anttiluode/WaveCortex): learned context-controlled relay, with a stronger scalar copy.
- [TuftSheet](https://github.com/anttiluode/TuftSheet): local context state rather than one global belief.
- [Qwenimagination](https://github.com/anttiluode/Qwenimagination): external concept transfer works at answer-time but delayed native reads fail under the tested writes.

MIT license. Antti Luode, research direction; Sol / ChatGPT, initial code and protocol, 18 November 2026.
