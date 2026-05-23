# M5 Max as Mainframe: the 9×9 Perf Proving Ground

Captured 2026-05-21 during WL5 monitoring as guiding philosophy for the
post-WL5 perf era and the long arc toward 15×15 + Gomocup submission.

## Thesis

Treat Jason's M5 Max as a **single, knowable mainframe** rather than a
generic ML target. Invest in chip-specific tuning — parameter sweeps to
find the exact batch sizes, sim counts, worker counts, kernel shapes,
memory layouts that make *this specific SKU* sing. The 9×9 gomoku
pipeline is explicitly the **perf proving ground**: we don't care that
9×9 is small, we care that it's a tractable workload on which to learn
the chip in deep detail before committing to a month-long 15×15 run.

The deliverable from this era is not "the strongest 9×9 player." It is
**a calibrated model of how the M5 Max behaves under AZ-style workloads**,
captured as a reusable chart that lets us pick the right knobs
confidently for 15×15.

## The reference frame

Jason came up on a 4.77 MHz IBM XT with a 5 MB hard drive where he
"knew every byte." That machine was ~200,000× slower than the M5 Max
GPU on raw compute, and people wrote real software on it — Commander
Keen, ZZT, Turbo Pascal compilers — precisely *because* they understood
where every cycle and every byte went. The constraint wasn't compute;
it was understanding.

The M5 Max in 2026 is a single coherent machine: ~14 TFLOPS GPU + ~38
TOPS ANE + AMX coprocessor + 48 GB unified memory at ~400 GB/s. It is
a knowable artifact. The "old days" approach — own the machine, push it
to its limits, write code that knows the hardware — applies. This is
the opposite stance from "just throw PyTorch defaults at it and scale
later," which assumes the chip is interchangeable. It isn't.

## What "know every byte" means in 2026

| 1985 XT | 2026 M5 Max |
|---|---|
| Memory map: IVT, BIOS data, conventional 640K, A000/B000/B800 video | Unified pool: which tensors live where; what's pinned for ANE; what's GPU-resident; SLC working set per worker |
| Cycle map: 8088 4.77 MHz, 6 µs per multiply | Per-epoch profile: MCTS select ms, eval batch ms, backprop ms, optimizer step ms, wandb log ms — and *why* each number is what it is |
| Topology map: 8088 + 8087 FPU + DMA + interrupt controllers | 12P + 4E cores, GPU compute units, ANE 16×16 tiles, AMX block size — which engine each piece of the workload actually lands on |
| Bandwidth map: ISA bus 8 MB/s, disk 80 KB/s | 400 GB/s unified bandwidth budget — what fraction self-play is consuming vs training |

We don't need any of this to *use* the chip. We need it to make it sing.

## What to sweep

Generic ML perf rules of thumb from CUDA literature should be
**re-validated here, not transplanted**. MPS + unified memory have
different optima. Prefer a sweep over a guess.

The first canonical sweep:

| axis | values |
|---|---|
| workers | 1, 2, 4, 8, 12, 16 |
| games-per-worker | 4, 8, 16 |
| n-simulations | 100, 200, 400, 800 |
| batch-size (eval) | 64, 128, 256, 512 |
| model size | tiny, small, medium |

Run each cell for ~5 min of self-play, chart augmented-positions/sec
and GPU/ANE/CPU utilization. ~4 × 3 × 4 × 4 × 3 = 576 cells, but the
useful contour is captured with a fractional design (e.g. ~30-50 cells
sampled across the corners + diagonals). Total budget: ~4-6 hours.

The deliverable is a **contour plot** showing the M5 Max's gomoku-AZ
behavior. That goes on the wall.

## The chip-specific levers worth disproportionate investment

Unified-memory pipelining is the most M5-Max-specific lever and has
almost no published prior art for AZ workloads:

- **ANE for inference, GPU for training, AMX for eval — simultaneously**.
  All three share tensors at zero copy cost thanks to unified memory.
  Almost no one does this because CUDA can't. We can.
- **MPS fallbacks to CPU**: every op that silently falls back is a
  latency hit. Audit + port the worst offenders to Metal Shading Language
  if needed.
- **Custom Metal kernels for the MCTS hot path** (selection, virtual loss
  accounting, leaf batching). Currently in C; some pieces probably belong
  in Metal where they can co-locate with the eval tensors.
- **Quantization-aware training** so INT8 ANE inference is lossless
  (see [ane-int8-inference.md](ane-int8-inference.md)).
- **MPS heap sizing + INT_MAX awareness** so we don't accidentally
  fragment the unified pool (see [project-gomoku-perf-ceiling]
  in memory and [mcts-perf-ceiling.md](mcts-perf-ceiling.md)).

2026-05-22 scout correction: the first Core ML harness says to phrase this
as **engine isolation**, not guaranteed zero-copy tensor sharing. PyTorch
MPS, Core ML, and BNNS have separate runtime/layout boundaries even on
unified memory. The run still supports the thesis direction: raw Core ML
eval was slower than fused PyTorch/MPS, but Core ML eval pressure hurt MPS
trainer steps far less than a competing PyTorch/MPS eval process. See
[ane-int8-inference.md](ane-int8-inference.md) for the receipt.

2026-05-22 rail-meter correction: `CPU_AND_NE` is not enough evidence that
the ANE is active. Apple Vision person segmentation drove the ANE rail to
~4.47 W, proving the meter is live, but the fresh Gomoku shape-scout was
blocked by unavailable cached/passwordless sudo before same-window rail
sampling could run. The mainframe discipline here is literal: trust the
rail, not the requested compute-unit label.

## What this enables

The 9×9 → 15×15 + Gomocup arc only works if perf claws back the
~20-50× scale-up cost. Realistic envelope, all chip-specific levers
pulled:

| lever | throughput multiplier |
|---|---|
| ANE INT8 inference | 3-5× |
| Bit-packed wide buffer | indirect — reduces wallclock-per-elo-gain |
| Eliminate MPS fallbacks + custom Metal | 1.5-2× |
| Pipeline ANE + GPU + AMX simultaneously | 1.5-2× |
| **Compounded** | **10-25× over current baseline** |

That puts 15×15 at the same per-epoch wallclock as current 9×9, or
better. Over a month of uninterrupted training, that's 20k-30k epochs
at 15×15 — real Gomocup-trajectory territory.

Without this perf era, a 15×15 month-long run produces a weak engine
that nobody can rank. With it, the same wallclock produces something
worth submitting.

## Anti-patterns

- **"It runs fine on any Mac"** is not the goal. We're not building a
  portable library. "We extracted the last 10% out of this exact SKU"
  is the goal.
- **Single-process micro-benches** mislead about production throughput.
  See [project-perf-bench-lesson] in memory. Always sweep the
  production shape (multi-worker, multi-process, with real model + real
  MCTS), not the unit-test shape.
- **Don't transplant CUDA recipes**. CUDA assumes host↔device copies are
  expensive; MPS doesn't. CUDA recommends large batches to amortize
  kernel launch; MPS launch overhead is different. CUDA's "always
  prefer fp16" assumes Tensor Cores; M5 Max has different tradeoffs.
- **Don't optimize for the wrong workload phase**. Self-play, training,
  and eval have different bottlenecks. Optimize each separately, then
  optimize their interleaving.

## Sequencing

1. Finish WL5 (currently running, ends ~e9000).
2. Cheap-test buffer-width ablation
   (per [buffer-bit-packing.md](buffer-bit-packing.md)).
3. ANE INT8 port + validation
   (per [ane-int8-inference.md](ane-int8-inference.md)).
4. Run the canonical sweep above; produce the contour chart.
5. Eliminate top-3 MPS fallbacks identified by sweep.
6. Pipeline ANE + GPU + AMX prototype; validate on a 200-epoch run.
7. Port to 15×15 (renju ruleset — free-style is solved, black wins
   per Allis/Wu).
8. Implement Gomocup stdin/stdout protocol.
9. Calibrate ELO vs published open-source engines (Pela, Embryo, Rapfi).
10. Commit a month-long 15×15 run.

Each step is independently shippable. Each compounds the chip-knowledge
that makes the next step cheaper.

## Cross-refs

- [m5-max-fp16-and-throughput-regimes.md](m5-max-fp16-and-throughput-regimes.md)
  — concrete findings from the 2026-05-23 perf cycle: fp16 on MPS is no
  longer slow (nearly doubles R-S400 small at V=512); the same chip has
  bandwidth-bound and dispatch-bound regimes depending on model size;
  independent perf levers compose multiplicatively to four decimals.
  Numbers, reproduction commands, Reviewer-audited receipts.
- [mcts-perf-ceiling.md](mcts-perf-ceiling.md) — current perf state,
  what's already been ported from zeb, what hasn't.
- [activity-monitor-perf-runbook.md](activity-monitor-perf-runbook.md) —
  practical Activity Monitor usage for these sweeps.
- [ane-int8-inference.md](ane-int8-inference.md) — first chip-specific
  lever in the sequence.
- [buffer-bit-packing.md](buffer-bit-packing.md) — second chip-specific
  lever; CPU-side, frees MPS budget for everything else.
- [az-at-scale-vs-laptop.md](az-at-scale-vs-laptop.md) — why
  laptop-scale isn't toy-scale; supports the "real engine on one
  machine" thesis.
- Memory: [[feedback-know-the-machine]] — short-form version of this
  philosophy.
- Memory: [[project-perf-bench-lesson]] — single-process benches
  mislead, always sweep production shape.
- Memory: [[user-hardware]] — the specific SKU spec we're targeting.
