# Core ML / ANE design envelope and where our workload fits

*Captured 2026-05-23 after the L09 R-TRAIN-ANE cell came in below R-TRAIN-WL5 holistically (-41.5% aug/s) despite confirming the trainer-side MPS-relief mechanism the lane was designed to test (-55.7% trainer_step_s_p50). The asymmetry pointed at a framing question: what is Core ML actually optimized for, and how does our workload look through that lens?*

This page is the design-context companion to [coreml-ane-residency-lab.md](coreml-ane-residency-lab.md) (which is the control-plane / evidence-discipline page) and [m5-max-fp16-and-throughput-regimes.md](m5-max-fp16-and-throughput-regimes.md) (which is the public chip-findings writeup). The piece this page adds: characterize Core ML/ANE's design center as inferable from Apple's public framing, place our workload on that map, and pre-queue the research that would map the envelope's edges.

We're documenting this for the same reason as the throughput-regimes page: open-source repo, mostly. People searching "what is Core ML for" or "is ANE good for PyTorch inference" or "Core ML small model overhead" should land here and get a real characterization instead of folk wisdom in either direction.

## TL;DR

- **Core ML's design center is the iOS/macOS app ML stack** — Vision framework features (image classification, OCR, pose), Siri ASR, FaceID, AR tracking, photo / video filtering. Models shipped inside an app bundle, called paced to UI events or video frames, optimized for power efficiency and concurrency with the rest of the running app. Apple's "small model" in this context is something like MobileNet or BERT-tiny — 1–50M parameters, lifecycle measured in app installs.
- **Our gomoku workload looks very different through that lens.** 325k-param custom ResNet, called thousands of times per second per worker, with weights changing every few seconds as the trainer publishes new versions. By Apple's design-center framing this is research-compute that happens to use ML primitives — not the app-inference shape Core ML was built around.
- **The L09 result is consistent with the misalignment, not a Core ML failure.** Core ML returned correct numbers and didn't crash. It just couldn't outrun PyTorch/MPS on the worker side at our model scale, because the per-call pipeline overhead doesn't amortize over a forward that small.
- **The lever the L09 cell mechanically validated** ("workers vacate MPS, trainer-side step time drops 56%") **is real** and we exploited it through a different door at L11b' (`sgd_per_position` capping). ANE-as-concurrent-compute-stream is still a viable framing; ANE-as-faster-eval-than-MPS is not, at this model scale.
- **Where ANE could still pay for us** is documented at the end of this page, queued as concrete research lanes — most promisingly: medium-model on Core ML (where the bandwidth-bound regime might kick in), routing-units sweeps, and the deployment story (shipping the trained model in a phone app, where Core ML on ANE is exactly the right tool).

## 2026-05-23 OSS Frontier Update

Later 2026-05-23 measurements narrowed the simple public-Core-ML envelope:
ANE won one live-training corner (tiny/V=64), but small/V=64, tiny/V=512,
and medium/V=512 all lost holistically against torch/MPS or torch+fp16
baselines. That falsifies the easy "bigger model or bigger batch will
automatically amortize Core ML overhead" hypothesis for the current export.

The web pass does **not** make ANE less interesting. It makes the next
question more specific. Mainstream LLM runners use Metal/MLX/GPU because
generic decode is dynamic and memory-bandwidth-bound; ANE-specific OSS
projects win by shaping static dense graph islands, prefill-sized matmuls,
Core ML packages, IOSurface buffers, profiler-guided fallback removal,
and in some cases private APIs.

For the lab route map, see
[ane-moonshots-and-oss-frontier.md](ane-moonshots-and-oss-frontier.md).
For the web source record, see
[../sources/ane-oss-frontier-2026-05-23.md](../sources/ane-oss-frontier-2026-05-23.md).

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

## Research queue — mapping the envelope's edges

These lanes are queued in [perf-queue.md](../ops/perf-queue.md) for future autonomous sessions. The goal is to map the ANE envelope's edges along the dimensions we expect to matter, so we know where it breaks and where it might pay.

### L09c — Tiny model on Core ML CPU_AND_NE

**Hypothesis:** at tiny (~30k params), the model is so small that ANE pipeline overhead is even worse per call than at small. But under live-training pressure where the trainer is running on MPS, even a slow ANE worker might pay because the alternative (workers on MPS torch competing with trainer) gives up so much MPS time. Measure to confirm or rule out.

**Cell:** `python scripts/lab_train_cell.py --model tiny --workers 8 --games-per-batch 8 --n-simulations 400 --wave-size 64 --evaluator coreml --coreml-compute-units CPU_AND_NE --warmup-secs 30 --measurement-secs 120 --device mps`

**Expected outcome:** unclear. If the trainer-side gain dominates (MPS-relief regime), modest improvement vs L09. If the per-call overhead dominates (dispatch-bound at smaller scale), worse than L09.

### L09d — Medium model on Core ML CPU_AND_NE

**Hypothesis:** medium (~1.5M params) is closer to Core ML's design envelope. The per-call compute is larger, so pipeline overhead amortizes better. Combined with the MPS-relief trainer-side effect, this is where ANE-offload might actually pay holistically.

**Cells:**
- Baseline: medium on torch/MPS at V=512 fp16 (we have this from L06fu-extended = 3,377 aug/s pure-gen; need the trainer-loaded version)
- Candidate: medium on Core ML CPU_AND_NE at V=512

**Expected outcome:** the high-prior case for ANE actually winning. If +10% holistic, the lever is real and worth productionizing. If still negative, the model-size sweet spot for ANE in our codebase may be even larger (e.g. when we eventually try 15×15 with a bigger network).

### L09e — Compute-units routing sweep at small / V=64

**Hypothesis:** L09 used `CPU_AND_NE`. We don't know if our model is fully ANE-resident or if Core ML silently demoted some ops to CPU/GPU. Sweep `CPU_AND_GPU`, `ALL`, `CPU_AND_NE`, `CPU_ONLY` to map.

**Cells (lab_train_cell):**
- Small / V=64 / `--coreml-compute-units CPU_AND_NE` (= L09 reference)
- Small / V=64 / `--coreml-compute-units CPU_AND_GPU`
- Small / V=64 / `--coreml-compute-units ALL`
- Small / V=64 / `--coreml-compute-units CPU_ONLY`

**Expected outcome:** primarily diagnostic. If `CPU_AND_GPU` ≈ `CPU_AND_NE`, the ANE isn't doing much for us (Core ML is mostly using the CPU+GPU path anyway). If `CPU_AND_GPU` > `CPU_AND_NE`, the ANE path is actively slower than the GPU path for this model. Either result is informative. **Pre-requisite:** wire `powermetrics` `ane_power` into the metadata so we can move past the `coreml-scheduled` cap (see [coreml-ane-residency-lab.md](coreml-ane-residency-lab.md)).

### L09f — Larger wave sizes on Core ML

**Hypothesis:** V=64 is Core ML's worst case for our workload (low per-call work). V=512+ batches more leaf evals per forward, which gives the ANE more compute per pipeline-overhead unit. The amortization may shift the regime.

**Cells:**
- Small / V=512 / Core ML CPU_AND_NE (vs the L06-followup torch/MPS reference of 9,398.5 aug/s)
- Small / V=1024 / Core ML CPU_AND_NE
- Small / V=2048 / Core ML CPU_AND_NE (if the model's max_batch supports it)

**Expected outcome:** the V-axis is the cheapest way to test amortization without changing model size. If Core ML at V=1024 closes the gap or wins, that's a structurally interesting finding (Core ML being competitive at large wave sizes implies it's a viable path for the trainer-side-relief argument).

### L09g — Model-size sweep at fixed V=512 fp16

**Hypothesis:** map the bandwidth-bound transition for Core ML as we did for MPS torch (Finding 2 in [m5-max-fp16-and-throughput-regimes.md](m5-max-fp16-and-throughput-regimes.md)). Where does Core ML's per-call overhead stop dominating?

**Cells (canonical_sweep, pure self-play, no trainer):**
- Tiny / V=512 / Core ML CPU_AND_NE
- Small / V=512 / Core ML CPU_AND_NE
- Medium / V=512 / Core ML CPU_AND_NE

**Expected outcome:** the chip-level analog of Finding 2 but for Core ML. Compare aug/s ratios across model sizes vs the same shape on torch/MPS. Tells us where (model size × V) Core ML actually starts competing.

### L09h — `.mlpackage` re-export cost amortization

**Hypothesis:** in live training, Core ML re-exports the model on every weight version. That overhead could dominate the cell's wall time if epochs are short. Measure the re-export cost directly and propose a caching scheme if it's significant.

**Approach:** instrument `gomoku/coreml_evaluator.py` to log per-export wall time. Then compute the ratio of re-export-wall to inference-wall over a 120s window. If re-export is > 5% of wall time, propose a delta-encoding or differential-compile cache.

**Expected outcome:** diagnostic, possibly motivating a cache layer.

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
