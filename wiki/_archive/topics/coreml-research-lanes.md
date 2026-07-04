# Core ML / ANE research lanes (archived, dormant)

**Status: DORMANT / ARCHIVE — full-fidelity extract, 2026-07-04.** This is the complete L09-family research-lane catalog and inbound-research landing zone, moved verbatim out of [coreml-design-envelope-and-our-fit.md](../../topics/coreml-design-envelope-and-our-fit.md) during the 2026-07-04 wiki curation to keep the design-envelope page at synthesis altitude. The ANE strand has been paused since May 2026; these lanes are **reactivatable** on any of the re-measurement triggers listed on the parent page (new Core ML version, new ANE features, evaluator-pipeline changes, model-arch changes, larger-model runs, inbound research). No facts were changed — only relocated.

---

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

### L09e' — ANE residency proof via thread-name (UNBLOCKED post-hollance-absorption)

**Status: UNBLOCKED 2026-05-23 (post-hollance-absorption) — ready to dispatch.**

Hypothesis: elevate L09c (lone PROMOTE) from `coreml-isolated` to `ane-metered` by detecting `H11ANEServicesThread` in worker process threads during the measurement window. If the thread exists, Core ML is using ANE for at least part of the forward pass. Per [hollance/neural-engine § Is my model using the ANE?](https://github.com/hollance/neural-engine/blob/master/docs/is-model-using-ane.md), this is a no-sudo equivalent of the `powermetrics ane_power` check.

**Dispatch shape:**
1. `lab_train_cell.py` at L09c shape: tiny / W=16 / G=8 / S=400 / V=64 / `--evaluator coreml --coreml-compute-units CPU_AND_NE` / 30s warmup + 120s measure.
2. During the measurement window, `ps -M <worker_pid>` for each of the 16 worker PIDs. Capture thread list for each.
3. Optional: also run `lldb -p <worker_pid>` and `image list Espresso` to inspect which Espresso engines were loaded; helps disambiguate ANE-only vs mixed-engine.
4. Receipt: tabulate per-worker thread audit results in the L09e' yaml. Cap elevation:
   - All 16 workers show `H11ANEServicesThread` → ANE is being used; cap elevates to `ane-metered`.
   - 0/16 workers show it → ANE is dark; L09c PROMOTE narrative reframes to "Core ML CPU+ANE dispatch beats MPS contention at tiny/V=64" — engine-isolation only, not ANE-residency. Both are real engine wins.
   - Mixed (some show, some don't) → flaky thread spawn; need longer observation window or multiple `ps -M` samples.

**Pre-requisite:** none beyond the existing `--evaluator coreml` flag and `ps`. (The original powermetrics-based version of this lane was blocked on sudo; the thread-name technique bypasses that.)

**Why this matters:** the L09c PROMOTE narrative is currently capped at `coreml-isolated`. If L09e' confirms ANE residency, the "tiny model fits the ANE design center better than small" framing has empirical support. If ANE is dark, the L09c win is via Core ML's CPU fallback being faster than MPS-contended torch — different mechanism, different future-research implications. Both readings are durable; both inform what shapes might pay when the inbound new ANE research lands.

### L09c-ALL — Re-run L09c at `--coreml-compute-units ALL` (auto-queued post-hollance-absorption)

**Status: QUEUED — small follow-up; cheap (1 cell, 3 min wall).**

Hypothesis: per hollance's "How do I make my model run on the ANE?" page, `.all` (ALL) is Apple's recommended setting for "I want ANE if possible" — with `CPU_AND_NE`, Core ML can only fall back to slow CPU for unsupported ops, but with `ALL` it can fall back to the GPU. L09e measured ALL marginally beating CPU_AND_NE by +3.1% at small/V=64. At tiny/V=64 (where L09c PROMOTE landed at +33.9% under CPU_AND_NE), ALL might widen the win further if any micro-fallback ops are routing to slow CPU under CPU_AND_NE.

**Dispatch shape:** `lab_train_cell.py` at the L09c recipe but with `--coreml-compute-units ALL` instead of `CPU_AND_NE`. Compare aug/s and trainer_step_s_p50 vs L09c's CPU_AND_NE numbers.

**Expected outcome:** marginal-to-modest improvement (likely +0% to +5%) if any fallback ops exist. If the gain is large (>10%), L09c's mechanism narrative shifts toward "fallback to GPU" rather than "ANE wins at tiny."

**Constraint:** trainer must still be on MPS (default for `lab_train_cell`); ALL routing in Core ML means workers could share the GPU with the trainer — defeating the engine-isolation property. **Need to inspect trainer_step_s_p50 carefully** — if it doesn't drop the L09c-equivalent ~16% relative to a torch baseline, ALL is putting workers on the GPU and L09c-ALL is not a fair compare. Add a matched torch baseline if Core ML auto-routes to GPU.

### L09i — `.mlpackage` op inspection → ROOT-CAUSE of the false-ANE strand

**Status: COMPLETED 2026-05-23 — RESOLVED the residency question (in a new direction). The blocker was not ND-broadcastable ops; it was a symbolic `ct.RangeDim` batch dimension.**

Original hypothesis: `convert_to="mlprogram"` may emit ANE-hostile broadcastable / ND ops even though `gomoku/model.py` is structurally clean. **Falsified — and the real cause was more fundamental.**

Finding: the op graph is clean, but the lab export at `gomoku/coreml_evaluator.py:267` (`export_model_to_coreml`) declared a **symbolic `ct.RangeDim` batch dimension**. The ANE requires fully static input shapes; a symbolic batch dim **silently demotes the whole Core ML program to CPU/BNNS**. Proof: the lab export and the known-ANE-resident scout export (`scripts/coreml_ane_residency_scout.py --batch-shape fixed`, ANE-rail-positive 2026-05-22) emit **byte-identical MIL op graphs** — RangeDim (symbolic) vs fixed (static) batch is the *only* delta. Therefore L09, L09c, L09d, L09c-V512, and L09e were **all CPU/BNNS, not ANE.** Receipt: [experiment-ledger.md](../ops/experiment-ledger.md) 2026-05-23 "L09i + L09i-fix".

**Implication for L09d's -59.6%:** the ND-op explanation is dead; both tiny and medium exports were on CPU/BNNS, so L09d's reject is a CPU/BNNS-vs-torch+fp16 throughput gap, not an ANE-hostile-op story.

### L09i-fix — switch RangeDim → fixed static batch (restores genuine ANE residency)

**Status: COMPLETED 2026-05-23 — ANE RESIDENCY RESTORED (first in lab history). Throughput still a REJECT; value re-opens on contention-immunity. Cap: `sample`-confirmed ANE residency (above `coreml-isolated`; strict `ane-metered` via powermetrics still pending sudo).**

Mechanism: replaced `ct.RangeDim` with a single **fixed static batch** — the evaluator pads each leaf-batch up to it and slices outputs back, chunking larger ones. Residency confirmed via the hollance no-sudo `sample` technique TWICE (isolated micro-probe + under the live self-play worker): hot path `AneInferenceOperationImplUsingAnefAPIs` / `_ANEClient doEvaluateDirect` / `AppleNeuralEngine`, **zero BNNS lines**. Note: `ct.EnumeratedShapes` also falls back to BNNS — a single fixed batch is the *only* ANE-resident option.

Throughput: at fixed batch 1024 (= wave×G×2), every eval is padded over a ~140-leaf wave tile (~7× tax) → **2,303.9 aug/s**. Sizing the fixed batch to the tile (wave×3 = 192, ~1.37× pad) → **7,697.7 aug/s (+234%)**, but still **−4.2% vs torch baseline (8,039.1)** and **−28.5% vs CPU/BNNS L09c (10,762.6)**.

Where it pays: ANE-resident workers **fully vacate the GPU** → best `trainer_step` measured (**0.0172s**) and most epochs/window ever (**18** vs 6 torch / 7 L09c). The ANE's value is **contention-immunity under a heavy GPU trainer**, not raw eval throughput. Receipt: [experiment-ledger.md](../ops/experiment-ledger.md) 2026-05-23 "L09i + L09i-fix" and [perf-log.md](../ops/perf-log.md).

### L09i-fix-load — does contention-immunity hold under a deliberately heavy GPU trainer? (re-opened)

**Status: QUEUED — the load-bearing test for the re-framed ANE value. Priority high.**

Hypothesis: L09i-fix's contention-immunity (workers off the GPU entirely → 18 epochs/window) is the ANE's only durable edge over torch+fp16, since raw throughput loses. Under a heavier GPU trainer (larger model, bigger SGD batch, more `sgd_per_position`), the engine-separation advantage should widen — the torch/CPU baselines fight the trainer for the GPU/MPS, the ANE workers don't. Measure trainer epochs/window and holistic aug/s for ANE-resident workers vs torch+fp16 vs CPU/BNNS as trainer GPU pressure scales up.

### L09i-fix-c — fixed-batch-size tuning to minimize pad tax while staying ANE-resident (re-opened)

**Status: QUEUED — cheap; directly attacks the −4.2%/−28.5% throughput gap.**

Hypothesis: the pad tax is the whole throughput story (7× at batch 1024 → 1.37× at wave×3=192). Sweep the fixed batch around the wave-tile size (e.g. wave×2, wave×3, wave×4) to find the minimum-pad point that still stays ANE-resident (re-confirm with `sample` each cell — recall a *too-small or non-fixed* shape falls back to BNNS). The win condition is closing the gap to torch+fp16 while keeping zero BNNS lines.

### L09-ANE-resident-reopen — re-map the throughput envelope on the genuine-ANE export (re-opened)

**Status: QUEUED — supersedes the pre-L09i envelope map.**

Hypothesis: every prior envelope point (L09/L09c/L09d/L09c-V512/L09e) was CPU/BNNS, so the model-size and V-axis envelope must be re-measured on the fixed-batch ANE-resident export before any "where ANE pays" claim can stand. Re-run the model-size sweep (tiny/small/medium) and the V-axis sweep, all on the fixed-batch export, all `sample`-confirmed ANE-resident, reporting both holistic aug/s and trainer epochs/window. This rebuilds the [§ Current state] table on genuine-ANE evidence.

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
2. **For any L09* lane the new research suggests reactivating**, move its "Research lanes" status from `REACTIVATABLE` or `QUEUED` to a new dispatch-queue entry in [gpu-queue.md](../ops/gpu-queue.md) with priority updated per the new prior.
3. **For new lane ideas** (e.g., a new compute-units mode, a new ANE feature), open a new `L09[i,j,k...]` card. Use the existing L09a/L09b/L09c naming convention.
4. **Update the cap-status table** in Current state with any newly-elevated receipts (e.g., if the new research provides `powermetrics ane_power` for an L09* cell, elevate it from `coreml-isolated` to `ane-metered`).

The framework is set up to absorb new findings without requiring a structural overhaul of this page.

