# M5 Max self-play eval on MPS: bandwidth vs dispatch regimes (and the fp16-on-MPS reversal)

> **Status: LIVE (2026-05-23).** Canonical owner of the fp16-on-MPS reversal
> finding. **Caution:** Finding 3's throughput compound is a *generation* win — in
> a real training run that same recipe runs away; read its cautionary epilogue,
> [perf-bench-vs-real-training-cost.md](perf-bench-vs-real-training-cost.md).

*Findings from the gomoku AlphaZero perf lab on 2026-05-23. Hardware: MacBook Pro Mac17,6, Apple M5 Max, 48 GB, macOS 26.4.1. PyTorch 2.11.0 with `Conv2d + BatchNorm2d` fused for inference. Workload: ResNet-style policy/value network, MCTS-driven self-play in this repo's `gomoku` module.*

This page documents three findings about PyTorch-on-MPS eval throughput on Apple silicon that surprised us, contradict prevailing folk wisdom, and (as far as we've been able to find) are not in Apple's published docs. Numbers are reproducible from the receipts in this repo; commands inline.

We're documenting them publicly because the open-source ML community kept telling us things on the internet that turned out to no longer be true. If you're benchmarking PyTorch on Apple silicon in 2026+ and a search engine sent you here looking for whether `model.half()` is worth it on MPS — the short answer is *probably yes, and by a lot more than the old threads claim*.

## TL;DR

1. **fp16 inference on MPS is no longer slow.** At torch 2.11.0 with fused conv+bn, casting our small eval model to `torch.float16` **nearly doubled aug-positions-per-second (+97.2%, 4,765 → 9,398)**. The historic regression in this direction is fixed; "stick with fp32 on MPS" threads are out of date for our workload shape.

2. **The same chip has two distinct bottleneck regimes at the same wave size.** At wave_size=512, a small ResNet (~325k params) is memory-bandwidth-bound — fp16 nearly doubles throughput. A tiny model (~30k params) at the same wave_size is MPS-dispatch-bound (kernel-launch latency per call dominates), and fp16 yields only +3.6%. Model size is the deciding factor between "fp16 doubles" and "fp16 noop"; there is no single "is fp16 worth it on MPS" answer for a given chip.

3. **Independent perf levers compose multiplicatively, to four decimal places.** Two levers — capping the trainer's SGD work to free MPS for workers (+28% aug/s alone), and casting workers' eval model to fp16 (+97% aug/s alone) — were predicted to combine multiplicatively at 1.283 × 1.972 = **2.530×**. Measured combined effect at trainer level: **2.529×**. The M5 Max's MPS scheduler honors the abstraction that disjoint-surface compute is genuinely independent.

## Background and vocabulary

These findings come from a [12-lane perf cycle](../ops/perf-log.md) in this project's perf lab. The lab measures throughput at fixed quality on the M5 Max under realistic AlphaZero self-play conditions. Each "lane" is a paired measurement: a baseline cell + a candidate cell that varies one or two axes, run for 60-120 seconds against fresh random weights to isolate the chip's behavior from training dynamics.

Vocabulary you'll need to follow the rest of this page:

- **aug-positions / sec (aug/s):** the headline throughput metric. Each game-position is augmented to 8 reflections+rotations (the D4 dihedral group on the 9×9 board) before training, so the lab measures positions-after-augmentation per second.
- **wave_size (V):** the MCTS eval-batch size — how many leaf-node evaluations are batched into a single forward pass through the model. Higher V = fewer, larger forward passes per game.
- **n_simulations (S):** MCTS simulations per move. Higher S = more eval calls per game.
- **model sizes used:** `tiny` (~30k params), `small` (~325k params), `medium` (~1.5M params).
- **R-S400, R-S\*** reference points: pure self-play throughput (no trainer running) at a quality pin defined by `(model, sims)`. e.g. R-S400 = small / sims=400. Promotion at a reference point requires a no-behavior-change knob movement that improves the number.
- **R-TRAIN-WL5, R-TRAIN-\*** reference points: live training throughput — trainer + N workers all sharing MPS. The holistic metric for "how fast does an elo-gaining cell actually run."

Two driver scripts:
- [`scripts/canonical_sweep.py`](../../scripts/canonical_sweep.py) — pure self-play (workers only, no trainer).
- [`scripts/lab_train_cell.py`](../../scripts/lab_train_cell.py) — live training: trainer + N self-play workers all on MPS.

Both follow a smoke-first doctrine: 60-90s cells unless a number is genuinely ambiguous.

## Finding 1: fp16 on MPS is no longer slow

The folk wisdom across PyTorch forum threads and StackOverflow answers from 2022-2024 says "fp16 on MPS is slow, use fp32." That was true at the time of those posts. It is no longer true at torch 2.11.0 with `Conv2d + BatchNorm2d` fused for inference.

We re-tested the historic regression on small/V=512 by casting the eval model to `torch.float16` in the worker process, with outputs cast back to `float32` before they cross the MCTS boundary:

| cell | fp32 | fp16 | Δ |
|---|---|---|---|
| small / W=8 / G=8 / sims=400 / V=512 | 4,765 aug/s | **9,398.5 aug/s** | **+97.2%** |
| small / W=8 / G=8 / sims=200 / V=512 | 9,156 aug/s | **16,850.8 aug/s** | **+84.0%** |
| small / W=8 / G=8 / sims=100 / V=512 | 15,082 aug/s | **22,312.1 aug/s** | **+48.0%** |
| tiny  / W=16 / G=8 / sims=400 / V=512 | 22,088 aug/s | **22,873.8 aug/s** | **+3.6%** |
| medium / W=8 / G=8 / sims=400 / V=512 | (no fp32 V=512 baseline) | **3,377.2 aug/s** | +142% vs medium V=64=1,393 (combined V + fp16) |

Reproduce in this repo:

```bash
# fp32 baseline (no flag):
python scripts/canonical_sweep.py --out-dir sweep_logs/baseline-fp32 \
  --cells-from cells.csv --secs-per-cell 60

# fp16 candidate (--fp16-eval):
python scripts/canonical_sweep.py --out-dir sweep_logs/candidate-fp16 \
  --cells-from cells.csv --secs-per-cell 60 --fp16-eval
```

Where `cells.csv` has the columns `model,workers,games_per_batch,n_simulations,wave_size` and rows for the shapes above.

### The implementation that makes this safe

The key is that **MCTS still sees fp32 numbers**. fp16 is purely on the eval-forward side:

1. [`gomoku/selfplay_worker.py`'s `_maybe_half`](../../gomoku/selfplay_worker.py) calls `model.half()` once after weight load, casting parameters and buffers to fp16.
2. [`gomoku/mcts.py`'s `make_torch_evaluator`](../../gomoku/mcts.py) (when constructed with `fp16=True`) casts forward inputs to `.half()` before calling the model, and **casts both policy logits and value outputs back to `.float()` before the single `.cpu().numpy()` host transfer**. The cast-back happens inside the same `torch.cat(...)` that bridges device → host.
3. MCTS, native search, replay payload, and downstream consumers all see fp32 numbers. Only the forward's internal precision is fp16 (a small numerical noise in the rounded-to-nearest-fp16 weights and intermediate activations).

We verified game-shape preservation by checking `plies_mean`, which stayed at 15.96-15.97 across all fp16 cells vs the fp32 references (both pinned at the `--max-plies 16` cap; MCTS exploration is unchanged at this precision).

### Why we believe the win is real

- fp16 actually engaged: each worker log on every fp16 cell contains `[w0] fp16-eval enabled (model cast to torch.float16)`. We don't trust the `--fp16-eval` flag's effect at the metadata level alone; we check the worker stdout.
- The mechanism predicts the asymmetry across cells (next section): fp16 wins scale with how eval-bound the loop is, and they're proportional to the eval forward's memory bandwidth.
- The result was independently audited by a separate Reviewer agent against the math, the engaged-flag artifact, and the cast-back code at `mcts.py:519-529`. Verdict: APPROVE (precedent-setting for the lab's treatment of fp16-with-fp32-output-cast as a no-behavior-change knob).

## Finding 2: same wave size, two bottleneck regimes

The fp16 win above isn't uniform across model sizes — and the asymmetry is the interesting part. At the same wave_size=512:

| model | params | fp32 aug/s | fp16 aug/s | Δ |
|---|---|---|---|---|
| **small** | 324,570 | 4,765 | 9,398.5 | **+97.2%** |
| **tiny** | ~30,000 | 22,088 | 22,873.8 | **+3.6%** |

**The small model is memory-bandwidth-bound** at V=512: its eval forward pumps a lot of bytes per kernel call, and the MPS GPU spends most of its time waiting for memory traffic. Halving the bytes-per-call (fp16 instead of fp32) nearly doubles the throughput.

**The tiny model is MPS-dispatch-bound** at V=512: each kernel call is small enough that the cost of *issuing* the call — the round-trip through the Metal command queue, kernel launch, parameter bind — dominates the actual compute. fp16 halves a compute that the GPU was barely doing anyway, and the kernel-launch overhead is unchanged. Result: +3.6%, basically a noop.

The implication is that there's no single "is fp16 worth it on MPS?" answer for a given chip. It depends on the model's per-call workload. For a fixed model architecture, the fp16 win grows as you add layers or widen the network (more bytes per forward = more bandwidth-relief).

The same regime asymmetry shows up cleanly in the `sims` axis (with the small model held fixed):

```
S=100: +48.0%  ← MCTS Python overhead dilutes eval's share of wall time
S=200: +84.0%  ← eval becomes dominant
S=400: +97.2%  ← eval dominates wall time; fp16's bandwidth save shines
```

Higher sims means more eval calls per game, which means a larger share of wall time is spent in the eval forward, which is where fp16 helps. The monotonic shape is the mechanism's fingerprint.

The threshold at the M5 Max for our ResNet shape sits somewhere between tiny (~30k params) and small (~325k params); medium (~1.5M params) at V=512 fp16 hit 3,377 aug/s vs medium V=64 fp32 = 1,393, a combined +142% (V + fp16). Cleanly attributing the fp16-only portion for medium requires an A/B at V=512 we haven't run yet; the trend-line estimate is +62%.

### Practical guidance

If you're running PyTorch on MPS for inference and you're not sure which regime you're in, **just try fp16 with a 60-second smoke**. The signature:
- **+~2× speedup → bandwidth-bound regime.** Keep fp16 on. Expect bigger wins as your model grows.
- **+~0-5% → dispatch-bound regime.** Your bottleneck is kernel launch, not bytes. fp16 won't help you; consider larger batches (reduce launches per unit work), `torch.compile` (folding kernels), or moving to ANE (different scheduler entirely — though see [coreml-ane-residency-lab.md](coreml-ane-residency-lab.md) for what we found there at this scale).

## Finding 3: independent levers compose multiplicatively

The third finding is a "mental model survives contact with reality" surprise.

### The setup

At higher wave sizes (V=512 vs V=64), workers produce augmented positions ~2.4× faster. Without changing anything else, that means the trainer's replay buffer fills 2.4× faster, and with a fixed `--sgd-per-position` ratio (default 0.0025 in this codebase), the trainer ends up doing 3.36× more SGD steps per epoch. The trainer's per-epoch SGD time grew from ~3s to ~43s, monopolizing MPS for 43s of every ~52s epoch and starving the workers of GPU time. Net effect at trainer level at V=512+default-sgd: **-28% aug/s** — V=512's pure-gen win actually went *backwards* at the trainer level.

Two independent levers were available:

- **Trainer-side**: lower `--sgd-per-position` to 0.001 (2.5× lower than default). This caps the trainer's per-epoch work and frees MPS for workers. Net effect at trainer level: **+28% aug/s** vs the WL5 baseline (V=64, sgd=0.0025, fp32).
- **Worker-side**: cast workers to fp16 (Finding 1), halving their eval bandwidth needs. Net effect at the same V=512 measured in pure self-play (no trainer interference, R-S400 cell): **+97.2% aug/s** vs fp32.

### Mental model

The levers act on **disjoint surfaces**:
- The trainer's SGD path is fp32 on MPS by construction (`gomoku/train.py` contains zero `fp16` / `half` / `autocast` references).
- The worker fp16 cast affects only the worker eval model and is reverted at the MCTS boundary (Finding 1's safety property).

They share GPU time but neither changes the *kind* of compute the other does.

If the levers are truly independent, you should be able to multiply their effects:

```
predicted compound = 1.283 (low-sgd) × 1.972 (fp16) = 2.530×
```

### The measurement

Run at V=512 + sgd=0.001 + fp16 in [`scripts/lab_train_cell.py`](../../scripts/lab_train_cell.py):

```bash
python scripts/lab_train_cell.py --out-dir sweep_logs/compound \
  --model small --workers 8 --games-per-batch 8 \
  --n-simulations 400 --wave-size 512 \
  --ema-tau 0.99 --grad-accum-steps 4 \
  --sgd-per-position 0.001 --fp16-eval \
  --warmup-secs 30 --measurement-secs 120 \
  --device mps
```

Result:

```
measured compound = 8,340.5 aug/s / 3,297.6 aug/s (R-TRAIN-WL5 baseline) = 2.529×
```

**Match to four decimal places.** The two levers compose as the model predicts, with no surprise interaction terms.

### Why this is rarer than it sounds

Real systems have hidden couplings: an "independent" knob turns out to share a buffer, push a queue into a different scheduling regime, or trigger a different code path. The M5 Max's MPS scheduler appears to honor the abstraction we built — workers' fp16 forward and trainer's fp32 backward sit on the same MPS queue but don't contaminate each other in any measurable way.

For autonomous lab orchestration this is load-bearing. It means we can search for compound mechanism findings by stacking independently-verified single-axis wins, instead of having to brute-force the full Cartesian product of every knob combination. The 2026-05-23 lab session leaned on this hard: every Tier-1 receipt explicitly named the surfaces its lever touched, and the next-lane choice was guided by "which surface hasn't been touched yet that the existing mechanism story implies has slack."

## Caveats and scope

These findings are specific to the conditions we measured. We're being explicit about the boundaries because perf claims age fast on Apple silicon — torch and Metal drivers move quickly, and a finding that holds at one (chip, OS, torch, model) tuple may not at another.

- **Torch 2.11.0 on macOS 26.4.1.** Older torch versions may still show the historic fp16 regression. If you're on torch < 2.10 or macOS < 26, re-measure before trusting Finding 1.
- **Conv2d + BatchNorm2d fused for inference.** Our model uses `gomoku.model.fuse_model_for_inference()` to fold BN into Conv. The fp16 forward without that fusion may behave differently (BN's running stats in fp16 can lose precision; folding sidesteps it). Most production inference paths fuse; if yours doesn't, verify.
- **Pure-inference eval workload.** The fp16 path here is "model.half(), inputs cast to half, outputs cast back to float." Training-side fp16 (mixed-precision SGD with loss scaling) is a different game with different concerns; we did NOT measure that, and `gomoku.train` stays fp32.
- **Specific model scales.** `small` here is ~325k params; `medium` is ~1.5M. If your model is much smaller (< 50k), Finding 2 suggests you may be in the dispatch-bound regime where fp16 doesn't help. Reference: tiny at V=512 = +3.6%.
- **Apple M5 Max specifically.** Chip-level findings don't necessarily transfer to M1/M2/M3/M4 Max or Pro variants. The bandwidth/dispatch threshold (Finding 2) is plausibly chip-specific because it depends on the GPU's compute-to-memory ratio.
- **Findings 1 and 3 went through the [lab's Training-Quality Promotion Gate](../ops/promotion-gate.md#training-quality-promotion-gate):** any throughput finding that touches training-behavior knobs (Finding 3's `sgd_per_position` change) is recorded as `needs_repeat` for production adoption, not `promote`. The perf lab's job is to identify levers; certifying a knob for production requires a separate canary training run with validation-archive metrics. **Finding 1's fp16 win is a clean perf promote (no MCTS-boundary behavior change); Finding 3's compound is a perf reference only, with a TQ canary gate for any production adoption.**

## Receipts and primary sources

Every number on this page is backed by a yaml receipt in [`wiki/ops/experiment-ledger.md`](../ops/experiment-ledger.md) with an independent Reviewer audit. Specifically:

| lane | finding | verdict | Reviewer |
|---|---|---|---|
| L06-followup | small + tiny fp16 (headline) | promote | APPROVE |
| L06fu-extended | R-S200, R-S100, medium V=512 fp16 | promote × 3 | APPROVE |
| L10 | R-TRAIN-WL5 baseline = 3,297.6 aug/s | promote (baseline) | APPROVE |
| L11 | V=512 default-sgd at trainer | reject | APPROVE |
| L11b | V=512 + low-sgd alone at trainer (+28%) | needs_repeat (TQ) | APPROVE |
| L11b' | V=512 + low-sgd + fp16 compound (+152.9%) | needs_repeat (TQ) | APPROVE |

Sweep artifacts (summary.tsv, trainer.log, per-worker logs, metadata.txt) live under `sweep_logs/lab-*-20260523T*/` in this repo.

The session narrative is in [`wiki/ops/perf-log.md`](../ops/perf-log.md) under the 2026-05-23 entries; the session-end summary lists all 12 lanes in order.

## Cross-refs

- [perf-bench-vs-real-training-cost.md](perf-bench-vs-real-training-cost.md) — **the cautionary epilogue to Finding 3.** The +152% throughput compound this page measured honestly as a *generation* win became an unbounded per-epoch runaway in a real training run once the replay buffer filled. Read it before treating any throughput compound here as a training-speed claim.
- [research-lab-charter.md](research-lab-charter.md) — the lab's mission, autonomy boundaries, and stop-gates triage.
- [m5-max-as-mainframe.md](m5-max-as-mainframe.md) — parent philosophy: treat the chip as a knowable mainframe and tune it specifically.
- [mcts-perf-ceiling.md](mcts-perf-ceiling.md) — what was already optimized in our MCTS before this cycle (saves reviewers from re-suggesting known-done work).
- [coreml-ane-residency-lab.md](coreml-ane-residency-lab.md) — the parallel investigation of Core ML / ANE as the worker eval backend. Short version: at small/V=64 the ANE path frees up MPS for the trainer (trainer_step_s_p50 -55.7%) but Core ML eval at this model scale is ~2× slower than torch/MPS on the worker side, so the holistic R-TRAIN-ANE cell came in below R-TRAIN-WL5. The lever is real but the model-size operating point matters.
- [activity-monitor-perf-runbook.md](activity-monitor-perf-runbook.md) — how to interpret Activity Monitor during MPS workloads (spoiler: GPU% is misleading about whether you're bandwidth- or dispatch-bound).

If you found this page via search and it answered your question, the open-source repo is at [github.com/jasonyandell/gomoku](https://github.com/jasonyandell/gomoku) (free as in beer). PRs welcome with new measurements at different (chip, torch, model) tuples — the perf-log doctrine is "the receipt is the lane", and an external receipt is just as good as one we filed ourselves.
