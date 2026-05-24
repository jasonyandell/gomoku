# M5 Max cross-engine coupling — where Core ML runs, and the shared power envelope

*Captured 2026-05-23 (session-resume) after the hollance/neural-engine absorption (L09e') resolved where Core ML actually runs our gomoku model, and a GPU-load stress test surfaced a surprising cross-engine coupling: saturating the GPU collapses the throughput of self-play workers that are running on the CPU.*

Sibling to [m5-max-fp16-and-throughput-regimes.md](m5-max-fp16-and-throughput-regimes.md) (single-engine MPS findings) and [m5-max-as-mainframe.md](m5-max-as-mainframe.md) (the philosophy). This page is about what happens **between** the engines: which engine Core ML actually uses for our workload, and how loading one engine affects work running on another.

Open-source-repo rationale, same as the throughput-regimes page: someone searching "does Apple Silicon CPU throttle when the GPU is busy" or "is Core ML CPU_AND_NE actually using the ANE" should land here and get measured answers instead of folk wisdom.

## TL;DR

- **Core ML runs our gomoku model on the CPU (BNNS), not the ANE** — confirmed by process sampling (L09e'). Despite `--coreml-compute-units CPU_AND_NE`, no `H11ANEServicesThread` spawns and the hot path is `E5RT::Ops::BnnsCpuInferenceOperation::ExecuteSync`. Under `CPU_AND_GPU` it *still* picks CPU. Core ML's cost model chooses CPU for our tiny/V=64 model regardless of the routing hint.
- **The L09c "engine-isolation" win (+33.9%) is real but load-fragile.** It works because our tiny-model trainer barely loads the GPU, leaving headroom for the CPU workers. **Saturate the GPU with a synthetic hog and the CPU workers collapse −82%** (10,432 → 1,905 aug/s) while the trainer barely moves (+14% trainer_step). The engines are compute-isolated but **share a package-level resource**. **Mechanism now RESOLVED (Lpwr2 + Lpwr2b): the throttle tracks the GPU's working-set size / sustained occupancy, NOT its compute throughput.** A 3.5×-higher-FLOP fp16 hog at matched matrix size throttled the workers no more than the fp32 hog (−14.8% vs −15.9%, inside noise), while throttle scales with matrix *size* (−8.8%→−26%, 2048→8192). **Compute-power-draw is ruled out**; the lever is GPU memory footprint/occupancy, not FLOP count (see Part 2). The qualitative coupling was always solid; the mechanism is now pinned.
- **ANE-resident workers also throttle under GPU load, and the coupling is bidirectional.** A clean pure-self-play A/B (L09i-fix-load-v2, no trainer → no wave-barrier confound) shows ANE workers **−35%** under a GPU hog (3,548 → 2,307 aug/s) — the ANE is **not** contention-immune. Notably the coupling runs both ways: the 16 busy ANE-resident worker processes (Core ML eval + CPU-side MCTS — total worker package load, not the ANE engine specifically; this cell can't isolate which) throttle the GPU hog from ~10.7 down to **~2.72 TFLOP/s** (~4×), so worker↔GPU package load browns out in both directions and settles at an equilibrium (workers −35%, hog −75%). Because the hog intensities aren't matched (~2.72 here vs ~11 in Lpwr), the ANE reads gentler than the CPU/BNNS −82% but **the intensities differ — not a clean fragility ranking**. (This supersedes the earlier "generation held / positive lean" reading, which was a trainer-barrier artifact.) See `wiki/ops/experiment-ledger.md` 2026-05-23 "L09i-fix-load-v2".
- **Production implication:** a heavier production trainer (15×15 board, bigger net) that actually saturates the GPU would throttle the CPU self-play workers, erasing the L09c win. The engine-isolation benefit is conditional on GPU power headroom.
- This is the "where does the machine break" the mainframe philosophy is built to find. We can't light all three engines at full tilt simultaneously — they draw from one power budget.

## Part 1 — Where does Core ML actually run our model?

### The question

The L09c PROMOTE (tiny / W=16 / V=64 / Core ML `CPU_AND_NE` = 10,762.6 aug/s, +33.9% vs torch baseline under live training) was filed at the `coreml-isolated` cap — we knew the workers vacated the GPU (trainer_step relief proved it) but not *which* engine they landed on. Per [coreml-ane-residency-lab.md](coreml-ane-residency-lab.md), `CPU_AND_NE` is a *request*, not proof of ANE residency.

### The no-sudo technique (from hollance/neural-engine)

[hollance/neural-engine § is-model-using-ane.md](https://github.com/hollance/neural-engine/blob/master/docs/is-model-using-ane.md) documents that Core ML dispatches through the private Espresso framework's three engines:

- `Espresso::ANERuntimeEngine` → ANE
- `Espresso::MPSEngine` / `Espresso::MetalLowmemEngine` → GPU
- `Espresso::BNNSEngine` → CPU (BNNS = Basic Neural Network Subroutines, part of Accelerate)

And that a thread named `H11ANEServicesThread` spawns in the process iff the ANE is in use. Both are observable **without sudo** via macOS `sample <pid>` (or `lldb`), unlike `powermetrics ane_power` which needs root.

### The finding (L09e', 2026-05-23)

Re-ran the L09c shape and sampled worker processes mid-measurement:

- **No `H11ANEServicesThread`** in any worker. All threads generically named.
- **Hot path: `E5RT::Ops::BnnsCpuInferenceOperation::ExecuteSync`** — Espresso's BNNS/CPU engine.
- `com.apple.ANEServices` + `com.apple.ANECompiler` frameworks are lazy-linked into the process (Core ML always links them) but no ANE thread or `ANERuntimeEngine` symbol appears.
- Cell reproduced L09c cleanly: 10,431.6 aug/s (within session-thermal noise of 10,762.6).

**Verdict: L09c is `coreml-isolated` via the CPU/BNNS path, NOT ANE residency.** The "tiny model fits the ANE design center" hypothesis is **falsified** — Core ML chose CPU for tiny just as it does for small/medium. The reason tiny *wins* and small/medium *lose* is that Core ML's BNNS CPU path is fast enough at tiny/V=64 to beat torch+MPS-contended workers; at larger shapes BNNS-CPU is too slow. ANE was never in play.

### Routing doesn't change the engine

| routing | worker engine (sampled) | worker aug/s | trainer_step_s_p50 |
|---|---|---|---|
| `CPU_AND_NE` (L09c / L09e') | CPU / BNNS | 10,432–10,763 | 0.0267 |
| `CPU_AND_GPU` (L09c-cpugpu) | CPU / BNNS | 10,202 | 0.0268 |

Even with the GPU explicitly allowed, Core ML keeps our tiny model on the CPU. Its internal cost model decides the GPU dispatch overhead isn't worth it for a model this small. The MPS frameworks load into the worker (link artifacts) but never carry the hot path.

## Part 2 — The shared power envelope (GPU load throttles CPU workers)

### The experiment

If the L09c workers run on the CPU, are they isolated from GPU load? We saturated the GPU with a synthetic fp32 matmul hog (`scripts/gpu_load_generator.py`, 8192² matmuls, ~11–14 TFLOP/s sustained) running concurrently with the L09c CPU-worker cell, and measured worker throughput with vs without the hog.

Engine placement confirmed by sampling during the run:
- Trainer → GPU (`AGXMetalG`, the Apple GPU hardware driver; `MPSGraph`)
- Hog → GPU (`AGXMetalG`, `MTLCommandBuffer`)
- Workers → CPU (`BnnsCpuInferenceOperation`)

### The result

| arm | worker aug/s | games/s | trainer_step_s_p50 |
|---|---|---|---|
| baseline (GPU = light trainer only) | 10,431.6 | 46.0 | 0.0267 |
| stress (GPU = trainer + ~11 TFLOP/s hog) | **1,905.2** | 7.55 | 0.0305 |
| delta | **−81.7%** | −84% | +14% |

**The CPU workers collapsed 82% when the GPU was saturated — even though they run on the CPU.** The asymmetry is the diagnostic tell: the trainer (GPU, competing directly with the hog for GPU compute) slowed only +14%, while the workers (a "different" processor) slowed −82%. That rules out simple GPU-compute contention (which would hit the trainer hardest) and points at a **shared power/thermal budget**: a GPU pinned at ~11 TFLOP/s consumes the package power envelope, the CPU cores throttle to stay within it, and the BNNS convolutions in the workers run slow.

### The companion result: ANE workers throttle too, and the coupling is bidirectional (L09i-fix-load-v2, 2026-05-23)

Does the same brownout hit the ANE? The first attempt (L09i-fix-load) was confounded by the wave barrier — the trainer stalled under the hog and gated worker output, so the headline −96% measured the trainer, not the workers. **L09i-fix-load-v2 redid the test cleanly:** PURE self-play (`canonical_sweep`, NO trainer → no wave barrier, so aug/s is a *direct* worker rate), interleaved A/B, tiny / W16 / G8 / S400 / V64, ANE-resident workers, with vs without a GPU hog.

| arm | worker aug/s | hog TFLOP/s |
|---|---|---|
| A (no hog) | 3,548 | — |
| B (+ GPU hog) | **2,307** | ~2.72 |
| delta | **−35%** | — |

- **ANE workers DO throttle under GPU load: −35%.** The ANE is **not** contention-immune. This **supersedes** the earlier "positive lean / generation held" reading, which was a trainer-barrier artifact, not real immunity.
- **The coupling is bidirectional.** The GPU hog reached only **~2.72 TFLOP/s** here, vs ~10.7 alongside a light trainer in fix-load — i.e. the 16 busy ANE-resident worker processes (Core ML eval + CPU-side MCTS — total worker package load, not the ANE engine specifically; this cell can't isolate which) throttle the GPU hog roughly **4×**. The busy workers and the GPU brown *each other* out through the shared package power budget and settle at an equilibrium (workers −35%, hog −75%).
- **This reconciles fix-load:** there the trainer stalled, the workers idled on the wave barrier, and package power was therefore free — which is why the hog reached 10.7 TFLOP/s and generation "held" (only during the non-stalled bursts). Remove the stall confound and the true ANE↔GPU brownout shows up in both directions.

**Caveat — not a clean fragility ranking vs the CPU.** The hog intensities are NOT matched (~2.72 TFLOP/s here vs ~11 in Lpwr, because the busy workers suppressed the hog), so the ANE's −35% reads **gentler than the CPU/BNNS −82%, but the intensities differ** — do not claim a clean ANE-vs-CPU fragility ranking from these two numbers. What is solid: both engines throttle under GPU load, and the busy ANE-resident worker processes (total package load — Core ML eval + CPU-side MCTS, not the ANE engine specifically) throttle the GPU back. Full receipt: `wiki/ops/experiment-ledger.md` 2026-05-23 "L09i-fix-load-v2".

### Intensity sweep — attempt to pin the mechanism (confounded by thermal drift)

To distinguish power/thermal coupling (smooth with load) from a scheduling cliff (step), we swept the hog intensity (matrix dimension → GPU power draw) and watched worker throughput. Cells: tiny / W=16 / V=64 / Core ML `CPU_AND_NE`, 30s warmup + 60s measure (the GPU-stress effect is steady within ~2s, so 60s is plenty — confirmed 2026-05-23). The sweep ran 4 cells sequentially — which, as it turned out, is the wrong design for a thermally-sensitive signal.

| hog matrix | hog TFLOP/s | worker aug/s | games/s | trainer_step | vs baseline |
|---|---|---|---|---|---|
| 0 (baseline) | — | 8,531 | 33.5 | 0.0332 | — |
| 2048 | 9.34 | 3,552 | 15.4 | 0.0369 | −58% |
| 4096 | 11.88 | 1,172 | 4.95 | 0.0343 | −86% |
| 8192 | 11.65 | 4,884 | 18.8 | 0.0294 | −43% |

*(`sweep_logs/lab-Lpwr-gpu-coupling-20260523T183502Z/`; 4 cells, 30s warmup + 60s measure each, run sequentially.)*

**The sweep is thermally confounded — and that is itself the finding.** Two problems broke the clean intensity story:

1. **The baseline itself fell to 8,531**, down from L09e''s 10,431 measured ~20 minutes earlier. The chip is heat-soaked from ~30 min of continuous cells. The thermal floor drifted *during the sweep*.
2. **Non-monotonic with load:** the 8192 hog (11.65 TFLOP/s) produced *higher* worker throughput (4,884) than the 4096 hog (11.88 TFLOP/s, 1,172) — two near-identical GPU loads, 4× different worker output. Ordering by hog TFLOP/s doesn't clean it up either. The cells differ more by *when in the heat-soak cycle they ran* than by *how hard the hog pushed*.

**What survives the confound:** the coupling is **real and large** — every hog cell sits far below even the depressed baseline (−43% to −86%). GPU load unambiguously hurts the CPU workers. **What does NOT survive:** the intensity-scaling shape, and therefore the power-vs-scheduling mechanism distinction. On a heat-soaked M5 Max, thermal state dominates over the hog-intensity axis, swamping the signal we were trying to sweep.

**Two different measurement intents (don't conflate them):**

- **For isolating a non-thermal variable** (here: the GPU-load *intensity* axis), the cool-chip tight back-to-back A/B (10,431 → 1,905, run seconds apart) is the trustworthy design; the spread-out sequential sweep is wrong because thermal drift confounds cross-cell comparison. Methodology fix: interleaved A/B pairs (hog / no-hog back-to-back) on a cooled chip with cooldowns between intensities (lane Lpwr2).

- **For production-representative throughput, heat-soak is a FEATURE, not a confound** — per Jason 2026-05-23: "heat soaked numbers are not bad to know, training will be heat soaked." So we measured it (lane **Lhot**): 8 back-to-back R-S400 cells to thermal steady state, then heat-soaked R-TRAIN-WL5. **Result: NO heat-soak haircut on the production shapes.** R-S400 steady state ≈ 9,783 aug/s (vs cool-start 9,398.5, *+4%*); R-TRAIN-WL5 heat-soaked ≈ 3,381 (vs cool 3,297.6, *+2.5%*). The R-S400 curve wobbles through warmup (9,388–10,029) then settles stable at ~9,783 — it does not decay. **The M5 Max sustains production throughput indefinitely under realistic self-play/training load; the cold-start references are trustworthy.**

**Correction (this supersedes an earlier claim on this page):** an initial reading attributed a "~18% haircut" to heat-soak, from the tiny/V=64 baseline falling 10,431 → 8,531. That number was **NOT representative**: it was a Core ML *CPU/BNNS-worker* shape measured *right after the synthetic 14-TFLOP fp32 hog* — an artificial extreme GPU thermal load on a non-production shape. Under real production load (Lhot), the GPU-resident production shapes show no haircut. The withdrawn 18% is retained here only as a cautionary worked example.

**The surviving nuance:** the haircut may be **engine-specific** — GPU-resident work (R-S400, R-TRAIN-WL5) sustains its clocks; CPU/BNNS-resident work (tiny/V=64 Core ML workers) *may* throttle under sustained heat/power load. That fits the Lpwr power-coupling story (the CPU is the thermally-sensitive path). It's one messy post-hog data point on a non-production shape — needs a clean re-test (Lhot2) before it's a claim. See [[feedback-heat-soaked-is-production]].

### Mechanism RESOLVED — occupancy/working-set, not compute power (Lpwr2, Lpwr2b, 2026-05-23)

The intensity sweep above was confounded by thermal drift; the clean re-tests pin the mechanism. **Lpwr2** swept hog matrix size on a cooled chip with interleaved A/B pairs: the throttle scales with **matrix SIZE** (−8.8% at 2048 → −26% at 8192), the GPU's memory footprint / sustained occupancy — not a step (no scheduling cliff).

**Lpwr2b** then ran the decisive discriminator — fp16 vs fp32 hog at matched matrix 4096 (pure self-play, ANE workers, bracketed no-hog, 120s cells, which cut cross-cell noise to 6.6%):

| hog | TFLOP/s | byte-traffic | ANE workers |
|---|---|---|---|
| fp32 @ 4096 | 1.98 | 1× | **−15.9%** |
| fp16 @ 4096 | **7.03** (3.5×) | ~1.75× | **−14.8%** |

**3.5× more FLOPs produced ~zero extra throttle** (−15.9% vs −14.8% — 1.3% apart, inside the 6.6% no-hog noise). So the cross-engine throttle is **FLOP-rate-INDEPENDENT** — and since fp16 also carried ~1.75× the byte-traffic-rate with no extra throttle, it isn't cleanly byte-bandwidth-rate-driven either. Combined with Lpwr2's size-scaling, the coupling tracks the GPU's **working-set size / sustained occupancy** ("is the GPU pinned busy, and how big is its memory footprint"), NOT its compute throughput. **Compute-power-draw is ruled out as the driver.**

**Actionable takeaway:** to cut the contention a heavy GPU trainer inflicts on CPU/ANE workers, **shrink the GPU's memory working-set / occupancy, not its FLOP count** — a lower-FLOP trainer at the same footprint won't help; a smaller-footprint trainer (or one that yields occupancy) will. Full receipt: `wiki/ops/experiment-ledger.md` 2026-05-23 "Lpwr2b".

### Independent confirmation: ANE is 0% (system GPU monitor)

A system GPU monitor (Stats-style menu-bar reading, captured 2026-05-23 during the load tests) independently corroborates the L09e' `sample` finding:

- **ANE utilization: 0%** — workers are NOT on the Apple Neural Engine. Two independent tools (process `sample` showing `BnnsCpuInferenceOperation`, and the system monitor showing ANE 0%) agree.
- **GPU utilization: ~75%**, Render 1%, Tiler 0% — the GPU compute is busy (trainer + hog) with headroom, and it's pure compute (matmuls), not graphics. Confirms the "GPU has headroom" observation — but the hog test proves that headroom isn't free to fill: pushing GPU work steals power budget from the CPU workers.

## Part 3 — Two FULL production runs at once (the real co-tenancy number, 2026-05-24)

*Jason: "run 2 trainings at once, 2 full 8-process training runs, wandb and everything, at the same time. I want to measure perf degradation."* This is the production-scale version of the synthetic-hog tests above: not a hog vs workers, but **two complete AlphaZero pipelines** sharing the M5 Max. Each run = `run_sweep.py` cell (1 trainer + 8 self-play workers + 1 eval = 10 procs), so the concurrent state is **20 processes** all driving MPS. Two throwaway clones of the WL4 production recipe (`PERFA`/`PERFB`, fresh weights, own wandb runs `dkqo29v3`/`pd8yzq7a`), torn down after measurement.

**Method.** The trainer log's per-epoch `(tot: gen=X train=Y)` split decomposes self-play generation (worker MPS load) from SGD (trainer MPS load). Metrics are regime-independent **rates**, not epoch wall (which is confounded by wave-mode tile adaptivity and by game-length drift as the fresh models degrade): **moves/sec** = games/sec × plies (generation MPS work, invariant to game length since each move = one MCTS search at fixed sims) and **SGD steps/sec** = steps / train-phase-seconds (batch=512 always, so per-step cost is buffer-size-invariant). Solo baseline = PERFA alone (epochs 10–20); concurrent = PERFA epochs 31–42 and PERFB epochs 6–15, windows that fully overlap with both runs at 8-worker load. Buffers were still filling (245k–890k of 1.5M), so this is the **compute-contention** number; the dual-full-1.5M-buffer (~16 GB) memory-pressure regime was not measured.

| metric | solo (1 run) | PERFA concurrent | PERFB concurrent | per-run degradation |
|---|---|---|---|---|
| generation (moves/sec) | 392.7 | 205.9 | 219.2 | **−48% / −44%** |
| SGD (steps/sec) | 26.64 | 10.94 | 13.21 | **−59% / −50%** |
| epochs/min | 8.84 | 5.22 | 4.64 | −41% / −48% |
| epoch wall (s) | 6.79 | 11.49 | 12.93 | +69% / +90% |

**The headline: each run runs at roughly HALF speed when you run two at once.** Generation drops ~45%, SGD drops ~55%. The two runs share the machine fairly evenly (PERFB marginally ahead — within window/epoch-range noise).

**Aggregate machine throughput (sum of both runs vs one run) tells the deeper story — and it's an asymmetry:**

| | 1 run (solo) | 2 runs (sum) | aggregate Δ |
|---|---|---|---|
| generation (moves/sec) | 392.7 | 425.1 | **+8%** |
| SGD (steps/sec) | 26.64 | 24.15 | **−9%** |

- **Self-play generation has slack: two runs produce ~8% MORE total moves than one.** Workers alternate Python MCTS tree-traversal with MPS forward passes, leaving MPS-occupancy gaps; the second run's workers fill them. This is the production-path (torch/MPS workers) confirmation of [[project-perf-bench-lesson]] — production OS scheduling interleaves across MPS processes; a single run does **not** saturate MPS generation.
- **SGD training has NO slack: two trainers do ~9% LESS total work than one.** The trainer runs a tight back-to-back forward+backward loop on batch=512 that pins MPS occupancy high (no gaps), so two trainers collide directly and you lose a little to context-switching.
- This is exactly the **working-set/occupancy** mechanism from Lpwr2b: self-play = low/gappy MPS occupancy (interleavable → slack); SGD = sustained high occupancy (collides). The gen-vs-train asymmetry is the same lever seen from the production side.

**Practical takeaway.** Running two full runs concurrently is **not** a free doubling, but it's **not catastrophic either**: total machine throughput stays roughly flat (gen +8%, SGD −9%), just split ~50/50 across two runs that each take ~2× longer per epoch. If you need two recipes' results and can tolerate each finishing in ~2× wall-clock, co-tenancy is fine and slightly net-positive on generation. If you need one result fast, run it alone — the SGD side in particular gets nothing from sharing.

## Implications

### For the L09c finding

The L09c +33.9% engine-isolation win is **conditional on the GPU having spare power budget**. It's real on our current tiny-model trainer (which barely loads the GPU). It would **not** survive a production trainer heavy enough to saturate the GPU. The win is best read as "Core ML CPU offload pays when the GPU is underutilized" — a narrower claim than "engine isolation pays."

### For the three-engine-pipeline dream

The [ane-int8-inference.md](ane-int8-inference.md) vision (self-play on ANE, trainer on GPU, MCTS on CPU) assumed the three engines run independently. They do not: the Lpwr finding shows the CPU and GPU share a power envelope (GPU work brownout-throttled the CPU/BNNS workers −82%), and **L09i-fix-load-v2 now confirms the ANE is in the same budget** — ANE workers throttle −35% under a GPU hog, and the busy worker processes (total package load — Core ML eval + CPU-side MCTS, not the ANE engine specifically) throttle the hog ~4× back (bidirectional). So **all three NN engines draw from one package power budget**; you can't run them at full tilt simultaneously. The pipeline can still help by *balancing* load across engines, but the total throughput ceiling is set by the package power/thermal budget, not the sum of the three engines' peak rates. "Make the Mac sing" has a power-budget ceiling.

### For "lighting the GPU up too"

The GPU has headroom during a light-trainer Core ML-CPU-worker cell, but filling that headroom with synthetic work *steals from the CPU workers*. Useful GPU work (a heavier trainer, or torch/MPS workers sharing the GPU) would do the same. The lab's job is to find the load balance across engines that maximizes total aug/s under the shared power ceiling — not to max out any single engine.

## Methodology notes

- **`scripts/gpu_load_generator.py`** — synthetic GPU load via continuous large fp32 matmuls on MPS; prints effective TFLOP/s as a power-draw proxy. Reusable for future isolation/coupling tests.
- **Engine attribution via `sample <pid>`** — no sudo. Grep the call graph for `BnnsCpuInferenceOperation` (CPU), `AGXMetal` / `MPSGraph` / `MTLCommandBuffer` (GPU), `ANERuntimeEngine` / `H11ANEServicesThread` (ANE).
- **Cell budget** — 30s warmup + 60s measure is sufficient for steady-signal coupling tests (the GPU-stress effect stabilizes within ~2s). Reserve longer windows for epoch-rate measurements where the trainer's per-epoch wall matters.
- **Why not powermetrics** — `powermetrics` (which would directly show CPU/GPU/ANE power draw and confirm the throttle hypothesis) needs sudo, which wasn't available this session. The `sample`-based engine attribution + the throughput asymmetry are the available evidence. A future sudo-enabled session should run the stress test under `powermetrics` to directly observe the CPU frequency / power throttle.

## Open questions

- ~~Is the coupling power/thermal (smooth with load) or scheduling (cliff)?~~ **Resolved (Lpwr2 + Lpwr2b): neither compute-power nor a scheduling cliff — it's smooth with GPU working-set *size*/occupancy and FLOP-rate-independent. See Part 2 "Mechanism RESOLVED."**
- ~~Does the same throttle hit ANE workers (if we ever get the model ANE-resident)?~~ **Answered (L09i-fix-load-v2): yes — ANE workers throttle −35% under a GPU hog in clean pure self-play, and the busy worker processes (total package load — Core ML eval + CPU-side MCTS, not the ANE engine specifically) throttle the hog ~4× back (bidirectional package-power coupling). The ANE is not contention-immune. See Part 2.** (Open follow-up: where the workers↔GPU equilibrium lands under a *real* trainer's GPU load vs the synthetic hog.)
- What's the optimal load balance? E.g., a heterogeneous worker pool (some torch/MPS on GPU, some Core ML/CPU) tuned to the power ceiling. A real lane once the mechanism is confirmed.
- ~~Does fp16 GPU load (lower power per FLOP) throttle the CPU less than fp32 at the same TFLOP/s? Would separate "power" from "FLOPs" as the coupling variable.~~ **Answered (Lpwr2b): at matched matrix size, the 3.5×-FLOP fp16 hog throttled no more than fp32 (−14.8% vs −15.9%, inside noise) — FLOPs are not the coupling variable; working-set/occupancy is. See Part 2.**

## Cross-refs

- [coreml-design-envelope-and-our-fit.md](coreml-design-envelope-and-our-fit.md) — the ANE entry point; L09e' result folded into its Current State.
- [coreml-ane-residency-lab.md](coreml-ane-residency-lab.md) — cap ladder; L09e' pins L09c at `coreml-isolated` (CPU/BNNS), not `ane-metered`.
- [m5-max-fp16-and-throughput-regimes.md](m5-max-fp16-and-throughput-regimes.md) — single-engine MPS findings.
- [m5-max-as-mainframe.md](m5-max-as-mainframe.md) — the philosophy; this page is a concrete "where it breaks" finding.
- [perf-lab-charter.md](perf-lab-charter.md) — lab operating rules.
- Memory: [[project-light-all-engines]] — the multi-engine utilization goal this finding constrains.
- External: [hollance/neural-engine](https://github.com/hollance/neural-engine) — the ANE community resource that unblocked the residency check.
