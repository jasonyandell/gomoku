# Core ML / ANE design envelope and where our workload fits

**This is the canonical entry point for the ANE research strand.** Read here first; the other ANE pages are scoped narrower:

- [coreml-ane-residency-lab.md](coreml-ane-residency-lab.md) — evidence-discipline control plane (Cap ladder for ANE-residency claims; powermetrics receipt schema). Read when you need to know what level of claim a receipt can support.
- [ane-int8-inference.md](ane-int8-inference.md) — **historical** scoping doc from WL5 era (May 2026). Implementation plan was partly executed (Core ML evaluator shipped 2026-05-23); current-state material has moved here.
- [m5-max-fp16-and-throughput-regimes.md](m5-max-fp16-and-throughput-regimes.md) — chip-level MPS findings; Finding 2 (bandwidth-bound vs dispatch-bound regimes) is the MPS analog of L09g for the Core ML path.

*Captured 2026-05-23 after L09 R-TRAIN-ANE came in below R-TRAIN-WL5 holistically (-41.5% aug/s) despite confirming the trainer-side MPS-relief mechanism the lane was designed to test (-55.7% trainer_step_s_p50). Updated 2026-05-23 (session-resume) after L09c PROMOTE / L09d, L09c-V512, L09e REJECT mapped the engine envelope's edges. The original asymmetry pointed at a framing question: what is Core ML actually optimized for, and how does our workload look through that lens? The L09c-through-L09e results pin down the answer empirically.*

We're documenting this for the same reason as the throughput-regimes page: open-source repo, mostly. People searching "what is Core ML for" or "is ANE good for PyTorch inference" or "Core ML small model overhead" should land here and get a real characterization instead of folk wisdom in either direction.

## Current state — what we know after L09c through L09e (2026-05-23 session-resume)

The L09 / L09c / L09d / L09c-V512 / L09e measured comparison points map a tight ANE engine envelope. **One important caveat up front:** none of these receipts include `powermetrics ane_power` evidence, so by [coreml-ane-residency-lab.md](coreml-ane-residency-lab.md)'s cap discipline they all sit at `coreml-scheduled` or `coreml-isolated` — *not* `ane-metered`. The "ANE pays" narrative below should be read as **"Core ML at `CPU_AND_NE` routing pays" — engine isolation, not proven ANE residency.** Core ML may be running on the ANE, the GPU, the CPU, or a mix; we have no rail-level evidence either way.

With that caveat, the empirical envelope as of 2026-05-23:

| model / shape | engine | aug/s | vs matched baseline | cap |
|---|---|---|---|---|
| **tiny / V=64** (L09c) | **Core ML @ CPU_AND_NE** | **10,762.6** | **+33.9%** vs tiny torch — the lone holistic win | `coreml-isolated` (trainer_step -16%) |
| tiny / V=64 (L09c-baseline) | torch | 8,039.1 | — | n/a |
| tiny / V=512 (L09c-V512) | Core ML @ CPU_AND_NE | 10,609.8 | -24.0% vs tiny torch+fp16 | `coreml-isolated` (trainer_step -62%) |
| tiny / V=512 (L09c-V512 baseline) | torch+fp16 | 13,968.6 | — | n/a |
| small / V=64 (L09) | Core ML @ CPU_AND_NE | 1,930.3 | -41.5% vs R-TRAIN-WL5 | `coreml-isolated` (trainer_step -56%) |
| small / V=64 (L09e CPU_AND_GPU) | Core ML @ CPU_AND_GPU | 1,908.3 | -42.1% vs R-TRAIN-WL5 | `coreml-isolated` (trainer_step -56%) |
| small / V=64 (L09e ALL) | Core ML @ ALL | 1,989.8 | -39.7% vs R-TRAIN-WL5 | `coreml-isolated` (trainer_step -56%) |
| medium / V=512 (L09d) | Core ML @ CPU_AND_NE | 591.7 | -59.6% vs medium torch+fp16 | `coreml-isolated` (trainer_step -81%) |

Three things the data settles, at today's stack (today's Core ML version + today's gomoku evaluator pipeline + today's small/tiny/medium model-arch family):

1. **Engine-isolation (`coreml-isolated` cap) is universally real for our workload.** Every measured Core ML candidate dropped trainer_step_s_p50 by 16-81% versus the matched torch baseline. The MPS-relief mechanism — workers vacate MPS so the trainer doesn't fight them — is mechanistically clean in every cell's trainer log.
2. **Holistic aug/s pays only at tiny + V=64.** The engine-isolation gain dominates at this single shape because Core ML's worker-side throughput is competitive there (both backends are pipeline-overhead-bound at tiny). At every other measured shape, Core ML's worker eval is 2-6× slower than torch+fp16, and the worker-side loss outweighs the trainer-side gain.
3. **Compute-units routing is null-to-marginal.** L09e measured CPU_AND_NE vs CPU_AND_GPU vs ALL at the L09 reference shape; across-routing spread was 4.3%, with ALL the marginal winner at +3.1%. No routing rescues a rejected shape.

**Open question (load-bearing for any future ANE work):** is the L09c PROMOTE actually running on the ANE? The `coreml-isolated` cap is cleared by the data; the `ane-metered` cap requires `powermetrics ane_power` evidence we don't have. If it turns out L09c is actually running on the GPU portion of CPU_AND_NE (Core ML's silent op routing), the win is still real as engine-isolation but the "tiny model fits the ANE design center better than small" narrative collapses — it would be "small enough that Core ML's CPU+GPU dispatch beats MPS contention." See L09e' below in the research queue for the residency-elevation lane.

**The L09c PROMOTE is a `coreml-isolated` win at the engine-isolation level, NOT an ANE-residency claim.** Anything downstream that reads R-TRAIN-TINY-ANE = 10,762.6 aug/s should know that's the cap.

## Future-shape framing — this is a snapshot, not a verdict

Per Jason 2026-05-23: "definitely any of your findings are valid, it's future shape I'm encouraging you to remain optimistic about." The empirical envelope above is time-stamped to today's Core ML version + today's evaluator pipeline + today's model-arch family. Specific re-measurement triggers that should reactivate the queued lanes:

- New Core ML major version lands (each Core ML version has historically shifted op-residency coverage and overhead floors)
- New ANE features (new `compute_precision` options, new `MLComputePolicy` modes, new ANE op coverage)
- Evaluator-pipeline changes (different `.mlpackage` export path, different fp32-cast strategy at the boundary, batched-prediction API changes)
- Model-arch family changes (different residual block depth, different stem padding, new ops introduced)
- Larger-model training runs (the 15×15 board with a deeper/wider net is a different point in the envelope)
- The inbound ANE research Jason flagged 2026-05-23 — when it drops, see [§ Inbound research landing zone](#inbound-research-landing-zone) below

The single-point envelope is the right reading of today's stack, **not a structural ANE limit**.

## TL;DR

- **Core ML's design center is the iOS/macOS app ML stack** — Vision framework features (image classification, OCR, pose), Siri ASR, FaceID, AR tracking, photo / video filtering. Models shipped inside an app bundle, called paced to UI events or video frames, optimized for power efficiency and concurrency with the rest of the running app. Apple's "small model" in this context is something like MobileNet or BERT-tiny — 1–50M parameters, lifecycle measured in app installs.
- **Our gomoku workload looks very different through that lens.** 325k-param custom ResNet, called thousands of times per second per worker, with weights changing every few seconds as the trainer publishes new versions. By Apple's design-center framing this is research-compute that happens to use ML primitives — not the app-inference shape Core ML was built around.
- **The L09 result is consistent with the misalignment, not a Core ML failure.** Core ML returned correct numbers and didn't crash. It just couldn't outrun PyTorch/MPS on the worker side at our model scale, because the per-call pipeline overhead doesn't amortize over a forward that small.
- **The lever the L09 cell mechanically validated** ("workers vacate MPS, trainer-side step time drops 56%") **is real** and we exploited it through a different door at L11b' (`sgd_per_position` capping). ANE-as-concurrent-compute-stream is still a viable framing; ANE-as-faster-eval-than-MPS is not, at this model scale.
- **Where ANE could still pay for us** is documented at the end of this page, queued as concrete research lanes — most promisingly: medium-model on Core ML (where the bandwidth-bound regime might kick in), routing-units sweeps, and the deployment story (shipping the trained model in a phone app, where Core ML on ANE is exactly the right tool).

## What Core ML / ANE is designed for

Inferred from Apple's public materials (Core ML developer docs, WWDC sessions, Core ML Tools, MLX framework, the Vision framework's API surface) — none of this is privileged Apple-insider information; it's the design center any developer can read off the framework's shape, examples, and where Apple's own apps use it.

### Primary customer: iOS / macOS app developers

Core ML originated as Apple's framework for shipping ML inside iOS apps. Its API surface, model packaging format (`.mlpackage`), tooling (`coremltools`), and integration points (Vision framework, Natural Language framework, Speech framework) are all shaped around the app-developer use case:

- **One model shipped per app, baked into the bundle at build time.** Lifecycle is "compile once when the app installs, infer many times across the app's lifetime."
- **Forward passes paced to UI events or media frames** — not "as many as the CPU can issue per second." A photo classification model is called when the user taps a photo; an AR pose model is called at 60 FPS. Both are "modest call rate per second," not "thousands per second per process."
- **Power efficiency matters more than wall-clock throughput.** Phones run on batteries; laptops often do too. The ANE was designed in this constraint — it's a separate compute unit so that ML inference doesn't burn the GPU (which is busy rendering UI) or the CPU (which is busy with everything else).
- **Static models** — the model's weights don't change after install. Core ML's compile pipeline expects "do the work once, then run."

### What Apple's own software does with it

The clearest signal of design intent is which Apple features ship on Core ML / ANE:

- **Vision framework**: image classification, OCR, face detection, hand pose, body pose, animal recognition. Mostly convolutional models, batched-of-1 or small batches, called on photos / camera frames.
- **Natural Language framework**: tokenization, language identification, sentiment analysis.
- **Speech framework**: on-device ASR (Siri's offline path), VoiceOver.
- **FaceID**: a custom neural network running on ANE for the unlock-the-phone use case.
- **Photos app**: scene classification, people recognition, "Memories" curation.
- **AR Kit**: pose tracking, scene understanding.
- **More recently**, Apple has been pushing larger models — their Llama-on-Apple-Silicon demos, MLX-based transformers, etc. — through ANE, but those models are 1–13B params (not 325k), and the workload shape is still "one inference per user prompt," not thousands per second.

The pattern: small-to-medium models (in 2026 parlance: tens of MB to hundreds of MB), batched-of-1 or modest batches, modest call rate, static weights, on-device, power-sensitive.

### What Apple offers for what we're doing instead

If our workload is "thousands of small-model forwards per second with live weight updates," Apple's framework landscape suggests a different tool:

- **PyTorch on MPS (the path we use)**: research-flexible, supports arbitrary models, fast iteration. The MPS backend is what Apple has invested in to make PyTorch a first-class citizen on Apple silicon. It's the right tool for our shape.
- **MLX** (Apple's open-source ML framework, [ml-explore/mlx](https://github.com/ml-explore/mlx)): designed for research workloads on Apple silicon, with first-class support for training, unified memory, and dynamic graphs. We haven't measured MLX vs PyTorch-MPS for our specific shape; it would be a legitimate scout lane, though porting our codebase from PyTorch is non-trivial.
- **Metal Performance Shaders (MPS) kernels directly**: lowest-level option, write your own Metal compute kernels. Used internally by PyTorch-MPS; we'd only reach for it if we hit a structural ceiling in the PyTorch backend.
- **Core ML for the eventual deployment**: when the trained model ships in an app for a user to play against, Core ML on ANE is exactly the right tool. Different lifecycle — install once, infer many times, power-sensitive. The training-time use of Core ML is the misfit; the deployment-time use isn't.

## Our workload through that lens

Our gomoku self-play workload, in the dimensions Core ML optimizes for:

| Dimension | Core ML design center | Our workload | Fit |
|---|---|---|---|
| Model size | 1M–1B params (MobileNet to phone-Llama) | 325k params (small) / 30k (tiny) / ~1.5M (medium) | **Below design envelope** for small and tiny. Medium is closer. |
| Per-call work | Image / utterance / frame (MB of activation) | A handful of 9×9 board states (KB of activation) | **Far below design center**; per-call overhead dominates compute. |
| Call rate | UI-paced or frame-paced (10s/sec) | Thousands per second per worker | **20–100× above design center**; pipeline overhead is paid per call. |
| Weight lifecycle | Compile once at install, infer many times | Re-export `.mlpackage` every trainer epoch (potentially seconds-scale) | **Adversarial to design**; Core ML's compile pipeline expects long-lived models. |
| Concurrency goal | One model running while UI / camera / Siri runs separately | Multiple worker processes running concurrent eval, sharing the chip with a trainer | **Compatible** — this is exactly the concurrency story Core ML supports, even if the per-process throughput is lower than MPS torch. |
| Power profile | Battery sensitive | Plugged in, full-throttle | **Compatible** but the ANE's power-efficiency advantage doesn't help us — we have the watts. |
| Output dtype | Whatever the app wants | fp32 for MCTS (precision-sensitive search) | **Compatible** — Core ML's internal FLOAT16 + a fp32 cast at the boundary handles this. |

The most important rows are the top three. We're using Core ML for a workload that's **20–100× outside its design call rate** and **3–30× below its design model size** — which combine to put us in the worst possible spot for ANE's parallel-compute throughput vs the pipeline-overhead floor. The L09 result (Core ML eval ~2× slower than MPS torch at this scale) is the predictable consequence, not a bug.

The third-from-top row (weight lifecycle) is also relevant: every time the trainer publishes a new `worker_weights.pt`, our worker re-exports the `.mlpackage` from scratch. That export takes a few seconds. Core ML's design assumes "compile once when the app installs" — our pattern "compile every few seconds in steady state" is fundamentally adversarial.

## Where the chip breaks — and where ANE could still pay

Per Jason 2026-05-23: "part of M5 as mainframe is learning just where it breaks. even if we dont directly leverage it in the end, we'll know." This page is the framing; the perf-queue lanes below are the measurement.

### Where ANE might cross into "pays off" territory

The L09 mechanism (MPS-relief for the trainer) is real; the cost (worker-side eval slowdown) scales with model size. As model size grows, the per-call compute grows roughly with parameter count; the pipeline overhead is roughly constant per call. So at some model size, the ANE's parallel throughput will catch up to its overhead, and the worker-side numbers will start competing with MPS torch — and the trainer-side gain (which is mechanically real and significant) will still be there.

Concretely, we expect three things to be measurable:

1. **As model size grows from tiny (~30k) → small (~325k) → medium (~1.5M) → larger**, the ANE worker-side gap to MPS torch should narrow and eventually invert. *(Hypothesis; not yet measured.)*
2. **At larger wave sizes** (V=512+, V=1024+), each Core ML forward does more work per call, which amortizes the pipeline overhead better. The L09 cell used V=64 — Core ML's worst case for our workload. *(Hypothesis; not yet measured.)*
3. **Different `--coreml-compute-units` routing** (CPU_AND_GPU vs CPU_AND_NE vs ALL) may matter more than we think. We only measured CPU_AND_NE in L09; we don't know if our model is even fully ANE-resident (only `ane-metered` via `powermetrics` would prove that; see [coreml-ane-residency-lab.md](coreml-ane-residency-lab.md) on the residency cap discipline). *(Hypothesis; not yet measured.)*

### Where ANE almost certainly is the right tool

- **Deployment**: the trained gomoku model ships in a phone app for someone to play against. Core ML on ANE is the right tool — install-once lifecycle, modest call rate (one inference per move), battery-sensitive. This is the canonical Core ML use case and we have no reason to expect it not to work cleanly.
- **A separate "match-eval sidecar"**: if we want to run an evaluator probe (e.g. tournament games against a reference model) *during* training without stealing GPU time, an ANE-resident sidecar process is exactly the concurrency-story Core ML was designed for. The eval probe is paced ("play 8 games every 10 minutes"), not 1000/sec, so it fits the design envelope. We haven't built this yet; it's a natural follow-on.

### Where ANE probably doesn't pay (current best evidence)

- **Live training-time worker eval at small / V=64**: L09 measured -41% holistic. The mechanism (MPS-relief) is real but the price (slower workers) outweighs the gain.
- **Anything where we need fp32 throughout the forward**: Core ML uses FLOAT16 internally; if we needed fp32 precision in intermediate activations, Core ML wouldn't be the path (though for our MCTS-bounded use, the fp32 cast at the output is sufficient).
- **Workloads where the model changes faster than Core ML can compile**: the `.mlpackage` export takes seconds; if the trainer publishes new weights every 11 seconds (L11b''s steady-state epoch), most of the worker's wall time would be spent re-exporting, not inferring.

## Research lanes — status and findings

The lanes that map the envelope's edges. Each lane below has a status header: **COMPLETED**, **QUEUED**, or **REACTIVATABLE** (queued, will move up in priority on a re-measurement trigger). Per-lane cross-refs point to the canonical receipt in [experiment-ledger.md](../ops/experiment-ledger.md).

Cap notes per lane reference the [coreml-ane-residency-lab.md](coreml-ane-residency-lab.md) ladder: `coreml-scheduled` < `coreml-isolated` < `ane-metered` < `ane-resident-candidate` < `production-ready`. Every L09* receipt so far sits at `coreml-isolated` or below — we have engine-isolation evidence but not ANE-residency evidence.

### L09 — Small model on Core ML CPU_AND_NE (the original ANE-offload prototype)

**Status: COMPLETED 2026-05-23 — REJECT holistic, partial mechanism-confirmation. Cap: `coreml-isolated`.**

Hypothesis: workers on Core ML free the trainer from MPS contention; even with slower raw eval, the holistic aug/s beats R-TRAIN-WL5.

Measurement: `lab_train_cell` at small / W=8 / G=8 / S=400 / V=64 / CPU_AND_NE.

Result: **1,930.3 aug/s (-41.5% vs R-TRAIN-WL5 3,297.6)** holistic. But **trainer_step_s_p50 -55.7%** (0.0512 → 0.0227s) — the MPS-relief mechanism CONFIRMED on the trainer side. Worker-side: Core ML eval ~2× slower than torch/MPS at this scale; loss dominates the trainer gain. Receipt: experiment-ledger.md 2026-05-23 "L09 R-TRAIN-ANE rejects".

### L09c — Tiny model on Core ML CPU_AND_NE

**Status: COMPLETED 2026-05-23 (session-resume) — PROMOTE. Cap: `coreml-isolated`.**

Hypothesis: at tiny (~30k params), per-call pipeline overhead may amortize differently; under live-training pressure even a slower ANE worker might pay because the alternative (workers on MPS torch fighting the trainer) is worse.

Measurement: matched-shape A/B at tiny / W=16 / G=8 / S=400 / V=64. Candidate `--evaluator coreml --coreml-compute-units CPU_AND_NE` vs baseline `--evaluator torch`.

Result: **10,762.6 aug/s (+33.9% vs 8,039.1 baseline)**. trainer_step_s_p50 -16.3% (0.0319 → 0.0267s) replicating L09's MPS-relief at smaller magnitude. **The lone holistic Core ML win in the measured envelope.** Mechanism: at tiny, per-eval compute is so light that both backends are pipeline-overhead-bound; the trainer-side MPS-relief tips the holistic balance positive. Opens new envelope-mapping refs R-TRAIN-TINY-ANE (10,762.6) and R-TRAIN-TINY (8,039.1, torch baseline arm). Receipt: experiment-ledger.md 2026-05-23 "L09c R-TRAIN-TINY-ANE PROMOTE".

### L09c-V512 — Does V-axis amortize Core ML pipeline overhead at tiny?

**Status: COMPLETED 2026-05-23 (session-resume) — REJECT. Cap: `coreml-isolated`.**

Hypothesis: if L09c's win is "tiny model = pipeline-overhead-bound where Core ML can compete," then V=512 should compound (each forward does more work per pipeline-overhead unit). Auto-queued from L09c PROMOTE.

Measurement: matched-shape A/B at tiny / W=16 / G=8 / S=400 / V=512. Candidate `--evaluator coreml --coreml-compute-units CPU_AND_NE` vs baseline `--evaluator torch --fp16-eval`.

Result: **10,609.8 aug/s (-24.0% vs 13,968.6 torch+fp16)**. trainer_step_s_p50 -62.5% (MPS-relief still real and larger here). **V-axis amortization FALSIFIED at tiny.** Mechanism: torch+fp16 already extracts most of V=512's bandwidth-bound value at tiny (per L06-followup, tiny+V=512+fp16 was only +3.6% over fp32 because tiny is MPS-dispatch-limited, not bandwidth-bound). Core ML can't match torch+fp16's bandwidth utilization at this operating point. Receipt: experiment-ledger.md 2026-05-23 "L09c-V512 REJECT".

### L09d — Medium model on Core ML CPU_AND_NE (the high-prior model-size case)

**Status: COMPLETED 2026-05-23 (session-resume) — REJECT. Cap: `coreml-isolated`.**

Hypothesis: medium (~1.5M params) is closer to Core ML's design envelope; per-call compute is larger so pipeline overhead amortizes better. Combined with the MPS-relief mechanism, this is where ANE-offload might actually pay holistically.

Measurement: matched-shape A/B at medium / W=8 / G=8 / S=400 / V=512 (240s measurement windows to capture ≥3 trainer epochs). Candidate `--evaluator coreml --coreml-compute-units CPU_AND_NE` vs baseline `--evaluator torch --fp16-eval`.

Result: **591.7 aug/s (-59.6% vs 1,463.3 torch+fp16)** holistic. trainer_step_s_p50 **-81.4%** (0.2391 → 0.0444s) — MPS-relief amplified at medium V=512 because the medium trainer has more compute per SGD step. But Core ML's worker gen time at medium V=512 is 5-7× slower than torch+fp16 (gen 30-40s/epoch vs ~6s/epoch). **"Larger compute amortizes pipeline overhead" hypothesis FALSIFIED in our envelope** — the opposite is closer to true: larger per-call workloads expose Core ML's lower per-forward throughput vs torch+fp16. Opens new envelope-mapping refs R-TRAIN-MEDIUM (1,463.3 torch+fp16 baseline) and R-TRAIN-MEDIUM-ANE (rejected, 591.7). Receipt: experiment-ledger.md 2026-05-23 "L09d R-TRAIN-MEDIUM-ANE REJECT".

### L09e — Compute-units routing sweep at the L09 reference shape

**Status: COMPLETED 2026-05-23 (session-resume) — REJECT (axis null). Cap: `coreml-isolated`.**

Hypothesis: L09 used `CPU_AND_NE`. Could a different compute-units routing (CPU_AND_GPU, ALL) rescue the L09 reject? This also partially addresses the residency question — if `CPU_AND_GPU` ≈ `CPU_AND_NE`, Core ML's routing decisions are roughly equivalent and the ANE isn't differentiated; if they differ, the routing hint matters.

Measurement: 2 cells at small / W=8 / G=8 / S=400 / V=64 with `--coreml-compute-units CPU_AND_GPU` and `ALL`. (CPU_ONLY skipped — predictably slow; CPU_AND_NE = L09 reference.)

Result: CPU_AND_GPU = 1,908.3 (-1.1% vs L09 CPU_AND_NE); ALL = 1,989.8 (+3.1%). **Across-routing spread 4.3% — within natural noise; ALL is marginal winner.** All three routings still ~40% below R-TRAIN-WL5. The L09 reject stands at this Core ML + evaluator combination. Trainer_step_s_p50 clustered at 0.0197-0.0227s across all three routings (MPS-relief is structural to Core ML offload at this shape, not routing-specific). Receipt: experiment-ledger.md 2026-05-23 "L09e REJECT".

**Important diagnostic gap exposed by L09e:** the 4.3% across-routing spread tells us that for our workload at small/V=64, Core ML's effective compute is similar across CPU_AND_NE, CPU_AND_GPU, and ALL routings. This is *consistent with* "Core ML uses similar hardware regardless of routing hint" but doesn't prove ANE residency one way or the other. The `ane-metered` cap remains unproven for any L09* result.

### L09e' — ANE residency proof via powermetrics (REACTIVATABLE)

**Status: QUEUED — newly added 2026-05-23 (session-resume).**

Hypothesis: elevate at least one L09* receipt from `coreml-isolated` to `ane-metered` by re-running with `powermetrics ane_power` evidence in a matched window. Most informative target: re-run L09c (the lone holistic win) under powermetrics to determine whether the win is actually ANE-resident or running on CPU/GPU portions of the CPU_AND_NE routing.

Pre-requisite: cached or passwordless `sudo` for `powermetrics` (per `coreml-ane-residency-lab.md`'s blocked-2026-05-22 lane 03 note).

**Why this matters:** the L09c PROMOTE narrative is currently capped at `coreml-isolated` — if L09c's win turns out to NOT be ANE-resident, the "tiny model fits the ANE design center" framing collapses to "tiny model fits Core ML's CPU+GPU dispatch better than MPS contention." Both are valid engine-isolation wins, but they motivate different next moves.

### L09f — Larger wave sizes on Core ML beyond L09c-V512 (REACTIVATABLE)

**Status: QUEUED — downweighted by L09c-V512 reject; reactivates on Core ML version change or new ANE features.**

Original hypothesis: V=512+ batches more leaf evals per forward, amortizing pipeline overhead. L09c-V512 falsified this at tiny (the only model where ANE wins). At small/medium the prior is even weaker since both are already worse than tiny. Lane stays queued but is not load-bearing under today's stack.

Cells (when reactivated): small/V=1024 and small/V=2048 (if model max_batch supports) on Core ML CPU_AND_NE.

### L09g — Model-size sweep at V=512 (pure self-play, no trainer; REACTIVATABLE)

**Status: QUEUED — downweighted by L09d/L09c-V512 rejects; reactivates on Core ML version change.**

Hypothesis: map Core ML's bandwidth-bound transition along the model-size axis under pure self-play (no trainer contention), the chip-level analog of Finding 2 in [m5-max-fp16-and-throughput-regimes.md](m5-max-fp16-and-throughput-regimes.md). With the L09c/L09d data points already in hand under live training, the pure-self-play sweep would isolate Core ML's worker-side throughput from the engine-isolation effect.

Cells: tiny / V=512 / Core ML CPU_AND_NE; small / V=512 / Core ML CPU_AND_NE; medium / V=512 / Core ML CPU_AND_NE. All canonical_sweep (60s smoke each).

**Note:** `canonical_sweep` doesn't currently support `--evaluator coreml` flags — would need a small CPU-queue patch first (~30 LOC; mirror the lab_train_cell flag passthrough).

### L09h — `.mlpackage` re-export cost amortization diagnostic

**Status: QUEUED — priority 1.0; cheap (1 cell) and informative under any future ANE-payoff scenario.**

Hypothesis: in live training, Core ML re-exports the model on every weight version. That overhead could dominate cell wall time if epochs are short. Measure directly; propose a caching scheme if > 5% of wall time.

Approach: instrument `gomoku/coreml_evaluator.py` to log per-export wall time; re-run an existing L09 cell.

### L09d' — Larger-than-medium model under live training (REACTIVATABLE)

**Status: QUEUED — not yet card-formalized; reactivates when we move to 15×15 or a deeper network.**

Hypothesis (per the L09d card's notes): if Core ML's worker-side gap to torch+fp16 narrows with model size up to medium and then maybe inverts at larger sizes, a 15×15-board net or a deeper 9×9 net might be where ANE actually pays holistically. Today's medium V=512 = 1.5M params; the next inflection point in our codebase would be ~3-5M params (deeper residual blocks) or the 15×15 board (larger spatial dimension).

When this becomes load-bearing: when the project's primary training shape moves beyond today's small model (e.g., post-WL5 redesign with a 4M-param net, or when 15×15 Gomocup play becomes the target).

## Inbound research landing zone

Jason flagged inbound ANE research on 2026-05-23 (origin TBD when it lands — Apple WWDC, internal scout, external paper, etc.). When it drops:

1. **Drop new findings into a new dated subsection of "Current state"** above (e.g., `### After [inbound research] (YYYY-MM-DD)`). Preserve the prior-state subsections as historical.
2. **For any L09* lane the new research suggests reactivating**, move its "Research lanes" status from `REACTIVATABLE` or `QUEUED` to a new dispatch-queue entry in [perf-queue.md](../ops/perf-queue.md) with priority updated per the new prior.
3. **For new lane ideas** (e.g., a new compute-units mode, a new ANE feature), open a new `L09[i,j,k...]` card. Use the existing L09a/L09b/L09c naming convention.
4. **Update the cap-status table** in Current state with any newly-elevated receipts (e.g., if the new research provides `powermetrics ane_power` for an L09* cell, elevate it from `coreml-isolated` to `ane-metered`).

The framework is set up to absorb new findings without requiring a structural overhaul of this page.

## Caveats — what this page is and isn't

- **This is interpretation of Apple's design intent based on public materials**, not a privileged statement of Apple's strategy. We have no insider information; we're reading the framework's shape, examples, and where Apple uses it. Apple may use Core ML / ANE for things this page doesn't anticipate, and Apple may extend the framework in directions that change the design center over time.
- **The chip-level findings are specific to the M5 Max in 2026.** Older Apple silicon (M1/M2/M3/M4) has different ANE characteristics; iPhones and iPads have very different ANE profiles. Numbers don't transfer cleanly.
- **The "Core ML doesn't pay at our scale" claim is workload-specific.** A different architecture (different layer types, different op mix) might fit ANE's eligibility gates differently. Our small ResNet has standard Conv2d + ReLU + GroupNorm + a small policy/value head; mostly ANE-friendly ops, but not measured with the full residency cap discipline yet.
- **MLX is an open question for our codebase.** If we ever hit a wall where PyTorch-MPS can't go further and we still want to research on the M5 Max, MLX is a plausible next-framework scout. It would require porting our model + training loop, so it's a larger investment.

## Cross-refs

- [coreml-ane-residency-lab.md](coreml-ane-residency-lab.md) — the control-plane page: evidence caps, residency proof discipline, JSON-receipt schema. This page is the design context; that page is the measurement contract.
- [m5-max-as-mainframe.md](m5-max-as-mainframe.md) — parent philosophy: treat the chip as a knowable mainframe, including where it breaks.
- [m5-max-fp16-and-throughput-regimes.md](m5-max-fp16-and-throughput-regimes.md) — the chip-level findings page (fp16 reversal, bandwidth/dispatch regimes, multiplicative composition). Finding 2 is the analog of L09g for the torch/MPS path.
- [ane-int8-inference.md](ane-int8-inference.md) — earlier lane-of-thought on ANE with int8 quantization (different precision path, may interact with the design-envelope analysis here).
- [activity-monitor-perf-runbook.md](activity-monitor-perf-runbook.md) — practical guide to interpreting MPS / ANE / GPU residency from Activity Monitor and `powermetrics`.
- [perf-lab-charter.md](perf-lab-charter.md) — autonomous-lab operating rules; the L09[c-h] lanes above will be dispatched per the standard receipt-and-Reviewer protocol.
- Memory: [[project-coreml-reality]] — the personalized index pointer for this page.
- Memory: [[feedback-know-the-machine]] — why mapping the envelope matters even when we won't directly leverage the lever.
- External: [Apple Core ML developer documentation](https://developer.apple.com/documentation/coreml), [coremltools on PyPI](https://pypi.org/project/coremltools/), [MLX framework (Apple's open-source research framework)](https://github.com/ml-explore/mlx).
