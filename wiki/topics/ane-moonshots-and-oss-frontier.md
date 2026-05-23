# ANE Moonshots And OSS Frontier

Captured 2026-05-23 after a fresh web pass on OSS Apple Neural Engine work.
Source record: [../sources/ane-oss-frontier-2026-05-23.md](../sources/ane-oss-frontier-2026-05-23.md).

This page is a lab route map. It is not a claim that the current Gomoku Core ML
worker is production-ready. The point is to give the M5 Max perf lab a moon to
shoot at: max out the whole machine, learn where the classic ANE fits, and turn
Gomoku into the clean little proving ground for that knowledge.

## Thesis

The mainstream LLM runners did not ignore ANE because it is worthless. They
went to Metal/MLX/GPU because generic LLM serving needs flexible kernels,
dynamic shapes, KV-cache traffic, sampling, many quant formats, and fast model
swaps. The public ANE path is mostly Core ML: a compiled graph system that
likes static shapes, dense tensor work, supported ops, long-lived weights, and
careful buffer ownership.

That is inconvenient for a general LLM runner. It may be exactly the kind of
constraint this Gomoku lab can exploit.

Gomoku's dynamic part is the tree. Its neural part is small, pure inference,
and naturally batched at the leaf-eval boundary. The lab question becomes:

> Can we sculpt the evaluator so the ANE sees a dense, static, long-lived
> sidecar workload while CPU/native MCTS and MPS training keep doing the
> dynamic work?

## What The Web Pass Changed

Before this pass, the local evidence could be read as: "Core ML is slower than
MPS for our tiny model; maybe ANE is a dead end except as a curiosity."

After the pass, the better read is sharper:

- **Generic decode belongs on GPU/MLX today.** Apple itself frames LLM decode
  as memory-bandwidth-bound, while prefill/first-token is compute-bound and
  benefits from newer M5 GPU Neural Accelerators. That is not the same thing as
  classic Core ML ANE.
- **Classic ANE wins when the graph is shaped for it.** Apple's
  `ml-ane-transformers`, ANEMLL, and the profiler/bench ecosystem all point in
  the same direction: avoid fallback islands, use Core ML-compatible static
  graphs, manage IOSurface/buffer synchronization, and reduce host round trips.
- **The OSS frontier is not "full arbitrary LLM on ANE."** It is prefill-like
  dense matmuls, app-deployment models, monolithic/chunked Core ML conversion,
  in-model argmax/top-k reductions, static expert/offload islands, and private
  API research runtimes.
- **Fallback is the clue.** CPU/GPU fallback is not failure by itself; it is the
  current shape of useful ANE systems. The trick is to decide which graph
  islands deserve to be on ANE and stop paying switch penalties for tiny pieces.

## Current Local Evidence

The lab has already mapped the naive public Core ML path far enough to avoid
repeating it blindly:

- `R-TRAIN-ANE` at small/V=64 lost holistically even though trainer-side MPS
  relief was real. Worker eval was too slow.
- `R-TRAIN-TINY-ANE` won at tiny/V=64, proving the Core ML worker can help in
  at least one live-training corner.
- Tiny/V=512 and medium/V=512 did not become broader ANE wins. Bigger batch or
  bigger model alone is not the magic key in the current exported graph.
- Detached 934b rail receipts plus Vision positive controls say the ANE rail
  can be moved on this machine. The standing rule remains: rail proof needs
  same-window `powermetrics`, not a `CPU_AND_NE` label.

The conclusion is not "stop." It is "the easy surface is exhausted; shape the
workload."

## Moonshot Routes

### Route A — Rail And Profiler Control Loop

Goal: make ANE residency visible and debuggable before arguing about speed.

Steps:

1. Use `scripts/ane_vision_furnace.swift` as a known-good Vision positive
   control. It is a lab helper, not Gomoku evidence.
2. Run Gomoku Core ML cells with same-window `powermetrics`.
3. Add ANEMLL's `anemll-profile` to the toolbox and inspect Core ML compute
   plans for op placement, fallback reasons, graph interruptions, and measured
   prediction time.
4. Store profiler JSON next to each Core ML receipt.

Win condition: every ANE claim has both rail movement and op-placement evidence.

### Route B — ANE-Shaped Gomoku Evaluator

Goal: stop exporting the PyTorch model shape verbatim and design a model whose
eval boundary is ANE-friendly.

Ideas to test:

- Fixed batch dimensions first (`B=32/64/128`) rather than flexible `RangeDim`.
- Channel/filter counts aligned to ANE-friendly minima; avoid tiny matmuls.
- Larger single Core ML function with fewer host round trips.
- Policy/value heads shaped to reduce CPU/GPU fallback islands.
- In-model legality mask, top-k, or argmax-like reductions if profiler shows
  host transfer or unsupported head ops are meaningful.
- Output exactly what MCTS needs for a candidate set, not necessarily a full
  generic training-time tensor interface.

Win condition: a Gomoku eval package has a continuous ANE-heavy compute-plan
timeline and nonzero rail at the production batch size.

### Route C — Long-Lived Worker Weights

Goal: stop fighting Core ML's compile-once design center.

The public Core ML path bakes too much work into export/load. For live training
we should test deliberate staleness:

- ANE self-play workers run one checkpoint behind the trainer.
- Workers refresh every N epochs or when a background converter has the next
  package ready.
- Trainer publishes PyTorch weights on every epoch as usual, but Core ML
  workers consume a slower `.mlpackage` ring.
- Measure Elo/plies drift separately from throughput; stale eval may be fine
  for self-play if the window is short.

Win condition: compile/export falls below 5% of worker wall time and self-play
quality does not obviously collapse.

### Route D — INT8/LUT/ANE Quantization

Goal: chase the hardware path ANE is actually built for.

ANEMLL's release notes point at LUT quantization, FP16 scaling, in-model
argmax, and ANE-specific model packaging. For Gomoku:

- Build a calibration set from the WL5 validation archive.
- Compare FP16, weight-only INT8/LUT, and W8A8-style experiments where Core ML
  supports them.
- Score by three gates: rail, throughput, and archive policy/value drift.

Win condition: an ANE-metered quantized evaluator gives acceptable archive
drift and beats the FP16 Core ML path at the same batch.

### Route E — Prefill/Decode Analogy For MCTS

Goal: exploit the same split LLM systems keep rediscovering.

MCTS tree work is dynamic decode-like control flow; leaf evaluation is dense
prefill-like inference. Keep them on different engines:

| Work | Engine candidate | Reason |
|---|---|---|
| Selection/backprop/tree mutation | CPU native extension | Dynamic pointer/control flow. |
| Leaf-eval batches | ANE/Core ML | Dense, static inference if shaped correctly. |
| Trainer forward/backward | MPS/MLX GPU | Gradient path and flexible training. |
| Validation archive/match sidecar | ANE or CPU-only Core ML | Sidecar can be paced and isolated. |

Win condition: ANE leaf eval overlaps MPS trainer work without starving the
native MCTS workers.

### Route F — Private API Research Sandbox

Goal: learn the real constraints without committing production code to private
APIs.

Study and, if safe, run isolated benches from:

- `arozanov/ggml-ane`: `MUL_MAT` only, prefill-sized fp16 matmuls via private
  Core ML/ANE APIs, IOSurface I/O, compile-cache lessons.
- `maderix/ANE`: training/backprop proof, SRAM/INT8/IOSurface experiments,
  explicit limitations.
- `mechramc/Orion`: compiler/constraint validation, delta compilation,
  weight-reload tricks, benchmark harnesses.

This is an archaeology lane, not a production dependency. The deliverable is a
constraint table and a set of public-API design ideas to steal back into Core ML
or Metal/MLX.

Win condition: one page of constraints that directly changes our public Gomoku
export shape.

### Route G — Classic ANE vs M5 GPU Neural Accelerators

Goal: keep the two moons separate.

M5 has classic Core ML ANE and also GPU Neural Accelerators exposed through
Metal 4/TensorOps/MLX. The latter may be the right route for training and large
matmul throughput even if classic ANE becomes a sidecar engine.

Win condition: the lab has separate scoreboards for:

- Classic ANE/Core ML rail-backed inference.
- MPS/Metal/MLX GPU neural-accelerator throughput.
- Whole-machine overlap when both are active.

## Near-Term Lane Cards

```yaml
id: ANE-M1-profile-gomoku-coreml
tier: 1
hypothesis: The current Gomoku Core ML package loses because of fallback islands, graph interruptions, or tiny unsupported head ops; anemll-profile can name the exact blockers.
work: install/use anemll-profile; export fixed-batch Gomoku packages at B=32/64/128; save JSON compute-plan reports; summarize fallback reasons.
success: profiler JSON plus a ranked list of graph changes for Route B.
```

```yaml
id: ANE-M2-rail-positive-control
tier: 1
hypothesis: A same-window Vision furnace plus Gomoku fixed-batch Core ML cell can prove the meter and compare known-good ANE load to Gomoku load.
work: compile scripts/ane_vision_furnace.swift; run one-worker Vision positive control; run Gomoku B=32/B=128 cells; sample powermetrics in adjacent windows.
success: rail-backed receipt with Vision positive, CPU_ONLY negative, and Gomoku candidate.
```

```yaml
id: ANE-M3-ane-shaped-evaluator
tier: 1
hypothesis: A deliberately ANE-shaped Gomoku evaluator beats the verbatim PyTorch export by reducing fallback islands and dispatch overhead.
work: prototype an eval-only architecture/export variant; keep behavior comparable through archive logits/value checks; profile before benchmarking.
success: continuous ANE-heavy compute plan, nonzero ANE rail, and better positions/sec than current Core ML evaluator.
```

```yaml
id: ANE-M4-stale-package-ring
tier: 2
hypothesis: Core ML becomes viable when workers consume long-lived packages and refresh less often than PyTorch weights.
work: add package ring + background conversion; run stale-window A/B against torch workers; check plies/archive drift.
success: export/load cost below 5% wall and no obvious self-play quality regression.
```

```yaml
id: ANE-M5-private-api-constraint-scout
tier: 3
hypothesis: Private API projects have already discovered constraints that explain our public Core ML failures.
work: clone/read/run minimal benches from ggml-ane, maderix/ANE, and Orion in isolated scratch space; do not vendor.
success: a constraint table that informs public Core ML/MLX/Gomoku design.
```

## Stop Conditions

Do not stop at "the easy Core ML worker lost." Stop only when one of these is
true:

- Profiler plus rail evidence shows the public ANE path cannot keep Gomoku eval
  resident without unsupported private APIs.
- An ANE-shaped evaluator is built and loses to MPS on both raw throughput and
  trainer-overlap despite clean residency.
- A private-API scout shows the missing capability is unavailable through public
  APIs and cannot be approximated with package rings, fixed shapes, or in-model
  reductions.

Until then, the lab has a real moon: make the ANE a useful sidecar in a
whole-machine Gomoku pipeline, even if it never becomes the main runner.
