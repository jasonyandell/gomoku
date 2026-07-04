# Core ML / ANE design envelope and where our workload fits

**Status: DORMANT** (as of 2026-07-04). The ANE/Core ML strand has been paused since May 2026; this page is a **snapshot, not a verdict** — every empirical claim is time-stamped to that stack (Core ML version + evaluator pipeline + model-arch family of the day). The design-envelope framing and the L09i residency finding are durable; the detailed research-lane catalog was cut from this page *(removed 2026-07-04; recover: `git show ca76350:wiki/_archive/topics/coreml-research-lanes.md`)* (reactivatable on the re-measurement triggers below).

**This is the canonical entry point for the ANE research strand.** Read here first; the other ANE pages are scoped narrower:

- [coreml-ane-residency-lab.md](coreml-ane-residency-lab.md) — evidence-discipline control plane (Cap ladder for ANE-residency claims; powermetrics receipt schema). Read when you need to know what level of claim a receipt can support.
- [m5-max-cross-engine-coupling.md](m5-max-cross-engine-coupling.md) — the cross-engine power-coupling finding (GPU load collapses CPU workers −82%). **NOTE (2026-05-23, L09i): the "Core ML runs on CPU/BNNS, not ANE" verdict on that page was correct *for the lab export* but is no longer the whole story — L09i found the cause (a symbolic `ct.RangeDim` batch dim silently demoted the program to CPU/BNNS) and L09i-fix restored genuine ANE residency by switching to a fixed static batch. See "Current state" below.**
- [ane-int8-inference.md](ane-int8-inference.md) — **historical** scoping doc from WL5 era (May 2026). Implementation plan was partly executed (Core ML evaluator shipped 2026-05-23); current-state material has moved here.
- [m5-max-fp16-and-throughput-regimes.md](m5-max-fp16-and-throughput-regimes.md) — chip-level MPS findings; Finding 2 (bandwidth-bound vs dispatch-bound regimes) is the MPS analog of L09g for the Core ML path.

*Captured 2026-05-23 after L09 R-TRAIN-ANE came in below R-TRAIN-WL5 holistically (-41.5% aug/s) despite confirming the trainer-side MPS-relief mechanism the lane was designed to test (-55.7% trainer_step_s_p50). Updated 2026-05-23 (session-resume) after L09c PROMOTE / L09d, L09c-V512, L09e REJECT mapped the engine envelope's edges. The original asymmetry pointed at a framing question: what is Core ML actually optimized for, and how does our workload look through that lens? The L09c-through-L09e results pin down the answer empirically.*

We're documenting this for the same reason as the throughput-regimes page: open-source repo, mostly. People searching "what is Core ML for" or "is ANE good for PyTorch inference" or "Core ML small model overhead" should land here and get a real characterization instead of folk wisdom in either direction.

## Current state — RESIDENCY RESOLVED: the prior envelope was all CPU/BNNS (2026-05-23, L09i + L09i-fix)

**The residency question is now resolved — in a new direction. ANE residency IS achievable for our model; the blocker was a one-line export bug; and every prior L09* "ANE" result was actually running on CPU/BNNS, never the Apple Neural Engine.**

L09i (a diagnostic) found that the lab's Core ML self-play export declared a **symbolic `ct.RangeDim` batch dimension** (`gomoku/coreml_evaluator.py:267`, `export_model_to_coreml`). The ANE requires fully static input shapes; a symbolic batch dim **silently demotes the entire Core ML program to CPU/BNNS**. The proof is tight: the lab export and a known-ANE-resident scout export (`scripts/coreml_ane_residency_scout.py --batch-shape fixed`, which hit the ANE power rail on 2026-05-22) emit **byte-identical MIL op graphs** — the *only* difference is RangeDim (symbolic) vs a fixed (static) batch. So L09 (small, −41.5%), L09c (tiny, +33.9%), L09d (medium, −59.6%), L09c-V512, and L09e (routing) were **all CPU/BNNS** — the "ANE" framing on every one of them was wrong. The `coreml-isolated` engine-isolation wins those cells measured are still real (workers vacated MPS), but they were *CPU* isolation, not ANE isolation.

**L09i-fix** swapped `RangeDim` for a single **fixed static batch** (the evaluator pads each leaf-batch up to it and slices outputs back; chunks larger ones). This **restored genuine ANE residency** — the first in the lab's history — confirmed via the hollance no-sudo `sample` technique TWICE: an isolated micro-probe and under the *live* self-play worker. Hot path showed `AneInferenceOperationImplUsingAnefAPIs` / `_ANEClient doEvaluateDirect` / `AppleNeuralEngine` with **zero BNNS lines**. Note: `ct.EnumeratedShapes` (a few discrete sizes) does NOT stay ANE-placeable either — it also falls back to BNNS. **A single fixed batch is the only ANE-resident export option.**

**But throughput is still a reject at tiny/V=64.** The static export pads every eval to one fixed size. At fixed batch 1024 (= wave×G×2), every eval is padded over a ~140-leaf wave tile (~7× tax) → only **2,303.9 aug/s**. Sizing the fixed batch to the tile (wave×3 = 192, ~1.37× pad) recovered to **7,697.7 aug/s (+234%)** — but that is still **−4.2% vs the torch baseline (8,039.1)** and **−28.5% vs the CPU/BNNS L09c (10,762.6)**. So on raw worker throughput, genuine ANE residency loses to both torch/MPS and the (mislabeled-but-real) CPU/BNNS path.

**Where the value re-opens:** the ANE-resident workers **fully vacate the GPU** (separate engine), yielding the best `trainer_step` (0.0172s) and most epochs/window (**18**, vs 6 torch / 7 L09c) the lab has ever measured. So the ANE's plausible value is **contention-immunity under a heavy GPU trainer**, NOT raw eval throughput. This re-opens the envelope along a different axis than the original (throughput) framing. Full receipts: [experiment-ledger.md](../ops/experiment-ledger.md) 2026-05-23 "L09i + L09i-fix" and [perf-log.md](../ops/perf-log.md).

Re-opened follow-up lanes (see Research lanes below): **L09i-fix-load** (does the contention-immunity advantage hold/widen under a deliberately heavy GPU trainer?), **L09i-fix-c** (tune the fixed-batch size to minimize pad tax while staying ANE-resident), **L09-ANE-resident-reopen** (re-map the throughput envelope now that residency is real and tied to a fixed-batch export).

### Prior snapshot — what we thought after L09c through L09e (2026-05-23 session-resume) — NOW SUPERSEDED ON RESIDENCY

*The subsection below is preserved as the pre-L09i reading. Its `coreml-isolated` engine-isolation deltas are still valid, but read every "Core ML @ CPU_AND_NE" / "ANE" label in it as **CPU/BNNS** — the RangeDim bug meant none of these cells ever reached the ANE.*

The L09 / L09c / L09d / L09c-V512 / L09e measured comparison points map a tight ANE engine envelope. **One important caveat up front:** none of these receipts include `powermetrics ane_power` evidence, so by [coreml-ane-residency-lab.md](coreml-ane-residency-lab.md)'s cap discipline they all sit at `coreml-scheduled` or `coreml-isolated` — *not* `ane-metered`. The "ANE pays" narrative below should be read as **"Core ML at `CPU_AND_NE` routing pays" — engine isolation, not proven ANE residency.** Core ML may be running on the ANE, the GPU, the CPU, or a mix; we have no rail-level evidence either way. *(2026-05-23 update: L09i resolved this — it was CPU/BNNS, via the RangeDim bug. See the resolved snapshot above.)*

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

### External research: hollance/neural-engine (folded in 2026-05-23, post-session-resume)

Jason flagged [hollance/neural-engine](https://github.com/hollance/neural-engine) — Matthijs Hollance's community resource ("Everything we actually know about the Apple Neural Engine"). Hollance authored the *Core ML Survival Guide* and has been one of the longest-active third-party voices on ANE. The repo is a curated collection of what's been learned about ANE behavior through experimentation, since Apple doesn't document the framework's internals.

Folded into our model below as updated priors. Findings unchanged; framing tightened.

**What hollance confirms about our setup:**

- **Our model is structurally ANE-friendly.** Hollance's "problematic layers" list (custom layers, RNN/LSTM/GRU, gather, dilated convs, broadcastable/ND layers, big pools, weird upsampling) — none of these appear in `gomoku/model.py`. We use Conv2d + BatchNorm (fused at eval via `fuse_conv_bn_eval`) + ReLU + Linear. ResBlock + stem + policy/value heads. All standard, all on the supported list.
- **ANE is fp16 throughout** — confirms what we knew about Core ML's `compute_precision=FLOAT16`. Our `--fp16-eval` flag is correctly off when `--evaluator coreml` (per the L09b fix). Activations in our trained gomoku resnet should stay in the `1e-2` to `1e1` band — well inside fp16's safe range.
- **Quantization is for storage, not compute, on ANE.** INT8 weights save disk but compute still happens at fp16. This *retroactively explains* why the ane-int8-inference.md plan never produced a clear INT8-vs-FP16 inference-speed win — the speedup we hoped for from INT8 ops doesn't materialize on ANE (it's an iOS deployment-size benefit, not an inference-throughput benefit).

**What hollance changes about our framing:**

- **Core ML's GPU path goes through MPS** (Metal Performance Shaders) — same Metal infrastructure as our torch/MPS trainer. So our `coreml-isolated` cap clearance (trainer_step_s_p50 down 16-81% across L09 family) tells us *the Core ML workers were not on the GPU* — if they were, the trainer would still be fighting MPS contention. The workers must be on either ANE or CPU. This sharpens the residency question: it's ANE-vs-CPU, not three-way.
- **`.all` is Apple's recommended setting for "I want ANE if possible."** CPU_AND_NE excludes the GPU entirely — so when Core ML hits an unsupported op, the fallback goes to *CPU* (slow), not GPU. This may explain L09e ALL's +3.1% over CPU_AND_NE: with ALL, fallback ops route to the GPU instead of the slow CPU. **Implication for any future ANE work: try `.all` first.** Our L09c PROMOTE was at CPU_AND_NE; a re-run at ALL might widen the win.
- **Core ML can split a model across processors.** A "Core ML ran on the ANE" claim doesn't mean the *whole* forward ran on ANE — Core ML may run part on ANE and part on CPU at every call, with switching overhead. This is consistent with our pipeline-overhead-bound interpretation at small/V=64.
- **The new ML-program / `mlprogram` converter (which we use via `convert_to="mlprogram"`) prefers broadcastable/ND layers** that don't run on ANE. We might be inadvertently generating ANE-hostile ops in the `.mlpackage` even though our PyTorch source code is clean. **Inspectable via coremltools** — L09i lane added to queue.

**No-sudo techniques for residency proof — major L09e' unblocker:**

Hollance documents three ways to check whether Core ML is using the ANE *without* `powermetrics` (i.e., no sudo required):

1. **Thread-name check.** Pause the process in a debugger (or `ps -M <pid>`); look for a thread named `H11ANEServicesThread`. If it exists, Core ML is using the Neural Engine for at least some portion. **No sudo. No debugger if `ps -M` works.**
2. **Symbolic breakpoint on `-[_ANEModel program]`.** If it hits, ANE is used.
3. **Espresso engine attribution.** Core ML internally dispatches through `Espresso::ANERuntimeEngine` (ANE), `Espresso::MPSEngine` / `Espresso::MetalLowmemEngine` (GPU), or `Espresso::BNNSEngine` (CPU). Per-engine stack frames in `lldb` tell us which parts of the model ran where.

The thread-name check (option 1) is the cheapest. It resolves the L09c residency question **today**, without waiting on cached sudo.

**Implications for the lab's research lanes:**

- **L09e' (residency proof for L09c) is now UNBLOCKED** — swap the powermetrics dependency for the thread-name check. New dispatch shape: re-run L09c with `lab_train_cell`, while workers are running, `ps -M <worker_pid>` and check for `H11ANEServicesThread`. Per-worker thread audit captured in artifact. Caps elevated from `coreml-isolated` → `ane-metered` (if rail bright) or pinned at `coreml-isolated` (if rail dark).
- **L09i (mlpackage op inspection) added** — diagnostic to check whether `convert_to="mlprogram"` is generating ND-broadcastable ops that prevent full ANE residency. Code-queue lane; no GPU needed.
- **L09c-ALL (re-run L09c at `--coreml-compute-units ALL`) added** — given hollance's `.all` recommendation and L09e's marginal +3.1% finding, retesting L09c at ALL might widen the +33.9% win further. Cheap (1 cell, 3 min wall).
- **The "tiny model fits the ANE design center" narrative is now testable.** If L09e' shows ANE-resident at tiny/V=64, the design-envelope hypothesis (small per-call work + pipeline-overhead-bound regime) holds. If ANE is dark and CPU is bright, the win is engine-isolation via Core ML's CPU fallback — different mechanism, different framing for what shapes might pay in the future.

**Open questions still open after this absorption:**

- Does our medium model's `.mlpackage` have more ND-broadcastable ops than tiny's? (Could explain L09d's -59.6% — if Core ML is silently demoting more of the medium model to slow CPU than tiny's, the worker-side throughput collapse makes sense.) L09i diagnostic would resolve this.
- Why does L09e ALL's marginal +3.1% over CPU_AND_NE not extend to a bigger gap? Either Core ML's auto-routing is conservative, or the GPU portions are small relative to the rest of the work. L09e' could illuminate this through Espresso engine attribution.

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
- **MAJOR UPDATE (2026-05-23, L09i):** the entire prior "ANE" envelope was actually **CPU/BNNS** — a symbolic `ct.RangeDim` batch dim in our export silently demoted Core ML off the ANE. **L09i-fix** (RangeDim → fixed static batch) restored the first genuine, `sample`-confirmed ANE residency in the lab. It is **still a throughput reject** (−4.2% vs torch, −28.5% vs the old CPU/BNNS L09c), but ANE-resident workers fully vacate the GPU → best `trainer_step` (0.0172s) and most epochs/window (18) ever. **The ANE's plausible value is contention-immunity under a heavy GPU trainer, not raw eval speed** — see "Current state" and the re-opened L09i-fix-load / L09i-fix-c / L09-ANE-resident-reopen lanes.
- **Where ANE could still pay for us** is documented at the end of this page, queued as concrete research lanes — now reframed around contention-immunity (engine separation under a heavy trainer) plus the deployment story (shipping the trained model in a phone app, where Core ML on ANE is exactly the right tool). The pre-L09i "medium-model / routing-units" lanes are moot for residency (they were CPU/BNNS) and fold into L09-ANE-resident-reopen.

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

Per Jason 2026-05-23: "part of M5 as mainframe is learning just where it breaks. even if we dont directly leverage it in the end, we'll know." This page is the framing; the gpu-queue lanes below are the measurement.

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

## Research lanes — archived (dormant, reactivatable)

The full L09-family research-lane catalog (L09 / L09c / L09c-V512 / L09d / L09e / L09e' / L09c-ALL / L09i / L09i-fix / L09i-fix-load / L09i-fix-c / L09-ANE-resident-reopen / L09f / L09g / L09h / L09d') and the inbound-research landing zone were cut from this page to keep it at synthesis altitude *(removed; see note above)*.

Those lanes are **DORMANT but reactivatable** — the strand has been paused since May 2026. Re-measurement triggers that should move a lane back up in priority (from the archive):

- New Core ML major version (op-residency coverage + overhead floors shift each version)
- New ANE features (new `compute_precision` options, `MLComputePolicy` modes, expanded ANE op coverage)
- Evaluator-pipeline changes (different `.mlpackage` export path, fp32-cast strategy, batched-prediction API)
- Model-arch family changes (residual depth/width, stem padding, new ops)
- Larger-model training runs (a 15×15 / deeper net is a different envelope point)
- Inbound ANE research landing (Jason flagged 2026-05-23; see the archive's landing-zone protocol)

The load-bearing conclusions those lanes produced already live above in **Current state** (L09i RangeDim root-cause + L09i-fix ANE residency restored, throughput still a reject, value re-opens on contention-immunity) — the archive holds the per-lane dispatch shapes and receipts.

## Caveats — what this page is and isn't

- **This is interpretation of Apple's design intent based on public materials**, not a privileged statement of Apple's strategy. We have no insider information; we're reading the framework's shape, examples, and where Apple uses it. Apple may use Core ML / ANE for things this page doesn't anticipate, and Apple may extend the framework in directions that change the design center over time.
- **The chip-level findings are specific to the M5 Max in 2026.** Older Apple silicon (M1/M2/M3/M4) has different ANE characteristics; iPhones and iPads have very different ANE profiles. Numbers don't transfer cleanly.
- **The "Core ML doesn't pay at our scale" claim is workload-specific.** A different architecture (different layer types, different op mix) might fit ANE's eligibility gates differently. Our small ResNet has standard Conv2d + ReLU + BatchNorm2d (fused via `fuse_conv_bn_eval`) + a small policy/value head; mostly ANE-friendly ops, but not measured with the full residency cap discipline yet.
- **MLX is an open question for our codebase.** If we ever hit a wall where PyTorch-MPS can't go further and we still want to research on the M5 Max, MLX is a plausible next-framework scout. It would require porting our model + training loop, so it's a larger investment.

## Cross-refs

- [coreml-ane-residency-lab.md](coreml-ane-residency-lab.md) — the control-plane page: evidence caps, residency proof discipline, JSON-receipt schema. This page is the design context; that page is the measurement contract.
- [m5-max-as-mainframe.md](m5-max-as-mainframe.md) — parent philosophy: treat the chip as a knowable mainframe, including where it breaks.
- [m5-max-fp16-and-throughput-regimes.md](m5-max-fp16-and-throughput-regimes.md) — the chip-level findings page (fp16 reversal, bandwidth/dispatch regimes, multiplicative composition). Finding 2 is the analog of L09g for the torch/MPS path.
- [ane-int8-inference.md](ane-int8-inference.md) — earlier lane-of-thought on ANE with int8 quantization (different precision path, may interact with the design-envelope analysis here).
- [activity-monitor-perf-runbook.md](activity-monitor-perf-runbook.md) — practical guide to interpreting MPS / ANE / GPU residency from Activity Monitor and `powermetrics`.
- [research-lab-charter.md](research-lab-charter.md) — autonomous-lab operating rules; the L09[c-h] lanes above will be dispatched per the standard receipt-and-Reviewer protocol.
- Memory: [[project-coreml-reality]] — the personalized index pointer for this page.
- Memory: [[feedback-know-the-machine]] — why mapping the envelope matters even when we won't directly leverage the lever.
- External: [Apple Core ML developer documentation](https://developer.apple.com/documentation/coreml), [coremltools on PyPI](https://pypi.org/project/coremltools/), [MLX framework (Apple's open-source research framework)](https://github.com/ml-explore/mlx).
