# Perf Lab Queue

Two-queue scheduler per
[perf-lab-charter](../topics/perf-lab-charter.md#two-queue-scheduler):
**GPU queue** runs serial cells; **CPU queue** runs code/wiki/scaffold
work in parallel via Agent fan-out. Reviewer gates every promote
([perf-lab-reviewer-role](../topics/perf-lab-reviewer-role.md)).

Within each queue, sort by **tier** (1 architectural > 2 compound > 3
speculative > bg calibration), then by priority within tier:
`priority = (E[delta] × P[succeeds]) / wall_cost`.

Reference points (current bests):

| ref | cell | best | speedup vs WL5 V=64 |
|---|---|---|---|
| **R-S400** | small / W=8 / G=8 / S=400 / **V=512 / fp16-eval** | **9,398.5 aug/s** (L06-followup) | **+194.8%** 🔥 |
| **R-S200** | small / W=8 / G=8 / S=200 / **V=512 / fp16-eval** | **16,850.8 aug/s** (L06fu-extended) | **+180.5%** 🔥 |
| **R-S100** | small / W=8 / G=8 / S=100 / **V=512 / fp16-eval** | **22,312.1 aug/s** (L06fu-extended) | **+100.0%** |
| **R-S400-medium** (new ref) | medium / W=8 / G=8 / S=400 / **V=512 / fp16-eval** | **3,377.2 aug/s** (L06fu-extended) | +142% vs medium V=64=1,393 |
| **R-S400-tiny** | tiny / W=16 / G=8 / S=400 / **V=512 / fp16-eval** | **22,873.8 aug/s** (L06-followup) | **+212.2% vs tiny V=64=7,326** |
| **R-TRAIN-WL5** | full WL5 recipe | **3,297.6 aug/s** / 0.0917 ep/s / 14.07 g/s (L10) | — |
| **R-TRAIN-LEAN-fp16** (new ref; perf-only, TQ-gated for production) | WL5 + V=512 + sgd=0.001 + fp16 workers | **8,340.5 aug/s** / 0.0667 ep/s / 32.19 g/s (L11b') | **+152.9% vs R-TRAIN-WL5** 🔥🔥 |
| **R-TRAIN-TINY-ANE** (new ref; envelope-mapping, NOT R-TRAIN-WL5 substitute; `coreml-isolated` cap — see best-cells.md note) | tiny / W=16 / G=8 / S=400 / V=64 / coreml CPU_AND_NE | **10,762.6 aug/s** / 0.0417 ep/s / 49.43 g/s (L09c) | **+33.9% vs R-TRAIN-TINY torch** — engine envelope maps: Core ML at CPU_AND_NE pays at tiny only (ANE-residency unproven; L09e' is the residency-elevation lane) |
| **R-TRAIN-TINY** (new ref; torch baseline arm for R-TRAIN-TINY-ANE) | tiny / W=16 / G=8 / S=400 / V=64 / torch | **8,039.1 aug/s** / 0.0333 ep/s / 32.48 g/s (L09c-baseline) | — (baseline) |
| **R-TRAIN-MEDIUM** (new ref; envelope-mapping; torch+fp16 baseline for medium engine-comparison) | medium / W=8 / G=8 / S=400 / V=512 / torch / fp16-eval | **1,463.3 aug/s** / 0.0042 ep/s / 5.66 g/s (L09d-baseline240) | — (baseline) |
| ~~R-TRAIN-LEAN~~ | WL5 with V=512 | **2,362.8 aug/s** / 0.0083 ep/s / 8.42 g/s (L11, REJECT — gen win doesn't compound at trainer) | — |
| ~~R-TRAIN-ANE~~ | WL5 with workers on Core ML | **1,930.3 aug/s** / 0.0583 ep/s / 8.00 g/s (L09, REJECT holistic at SMALL; the same engine wins at TINY — see R-TRAIN-TINY-ANE) | — |
| ~~R-TRAIN-MEDIUM-ANE~~ | medium V=512 with workers on Core ML | **591.7 aug/s** / 0.0208 ep/s / 2.33 g/s (L09d, REJECT — trainer side -81% trainer_step_s_p50, but worker gen 6× slower on ANE at medium V=512 = holistic -59.6%; envelope sharply mapped: ANE pays at TINY only) | — |

## CPU queue (parallel — Agent fan-out, no GPU contention)

These run as Agent subagents in worktrees; integrate as merge commits.
Multiple can be in flight at once. Listed top-down by priority.

#### Lshare-ANE — BACKBURNER: public writeup of the ANE findings "for the next person" (Jason 2026-05-23)

```yaml
id: Lshare-ANE
tier: bg (backburner — run only when nothing higher-value needs the queue; no GPU)
purpose: knowledge share-out. Our ANE strand answered questions that are badly under-documented publicly; package the findings so the next person searching "Core ML ANE PyTorch inference residency" / "is the ANE worth it for ML" / "Core ML RangeDim CPU fallback" lands on real measurements.
the story worth telling (all measured 2026-05-23):
  - The ANE-residency GOTCHA: a symbolic ct.RangeDim (or EnumeratedShapes) batch dim silently demotes the WHOLE Core ML program to CPU/BNNS regardless of compute_units=CPU_AND_NE. Only a single FIXED static batch is ANE-placeable; the op graph is otherwise identical. Verify residency with no sudo via hollance's `sample <pid>` (AneInferenceOperationImplUsingAnefAPIs/AppleNeuralEngine vs BnnsCpuInferenceOperation).
  - The ANE is NOT a free lunch for concurrent self-play: even with residency it loses on throughput (≈ CPU/BNNS, loses to torch/MPS at tiny/small) AND shares the package power budget — not contention-immune (−35% under a GPU hog; bidirectional).
  - The cross-engine coupling mechanism: the throttle is FLOP-rate-INDEPENDENT (occupancy/working-set, not compute-power) → "shrink footprint, not FLOPs."
form: a public-facing topic page, sibling to m5-max-fp16-and-throughput-regimes.md (the public writeup for the fp16 findings). Pull from coreml-design-envelope-and-our-fit.md + m5-max-cross-engine-coupling.md + the L09i/Lpwr ledger receipts; rewrite for an external reader (strip internal lane IDs).
code_change: false (pure writeup; CPU-queue agent)
priority: 0.5 (backburner — do NOT preempt any real perf/training lane)
status: queued (backburner)
notes: Jason 2026-05-23 — "put in a backburner idea to share our ANE findings for the next person." The wiki pages already carry the open-source-repo framing; this is the consolidation/externalization pass. No urgency.
```

*(No other CPU lanes pending. The pre-restart code lanes (L12, L05, L06,
L08-driver) landed 2026-05-23; the L09i-fix coreml capability +
canonical_sweep coreml + --coreml-static-batch + fp16-hog + Ltrain-amp bf16
all landed this session on their worktrees — see Completed table.)*

## GPU queue (serial — one cell at a time on MPS)

Lanes listed top-down by **tier**, then by priority within tier.

### Tier 1 — Architectural / holistic

#### LF1-followups — real-training-cost research (the LF1 runaway reframes the lab's metric)

```yaml
id: LF1-followups
tier: 1 (HIGHEST — this reframes what the lab optimizes)
context: LF1 (the R-TRAIN-LEAN-fp16 recipe as a real run) exhibited an UNBOUNDED per-epoch runaway (steps/epoch 25→3236, wall 20s→437s over 31 epochs; wave tile 101→2898). The perf-lab cold-window R-TRAIN metric (120s ≈ 8 epochs) measured the pre-buffer-fill transient and called a divergent recipe "+152%". Detailed writeup: wiki/topics/perf-bench-vs-real-training-cost.md.
meta: the lab optimized aug/s (generation); the real objective is wall-clock-to-elo (training). They DIVERGE — maxing generation floods the trainer into a runaway. These lanes re-point the lab at the real objective.
lanes (each its own cell-set; detail in the writeup):
  1. WARM-BUFFER R-TRAIN METRIC (fix the broken metric): pre-fill the buffer to capacity (Lhot-style), THEN measure steps/epoch + train-time + tile-growth trend over ≥20 post-fill epochs. The current cold-window cell is non-predictive. [code: lab_train_cell warmup-to-fill mode]
  2. RUNAWAY STABILITY BOUNDARY: sweep wave_size (V) vs trainer-consumption; find the V where the tile stays bounded (V=64 stable, V=512 divergent — where's the knee?). [measurement]
  3. sgd_per_position vs sgd_per_game vs a hard step-cap at full buffer: sgd_per_position amplifies the runaway (scales with inflow); a cap/backpressure may bound it. [measurement]
  4. WALL-CLOCK-TO-ELO metric family: add a real-training-cost metric (time-to-elo / val-CE-vs-epochs); throughput proxies diverge from it. [design + longer cells]
  5. IS THE HIGH STEP-COUNT PRODUCTIVE? LF1 elo was noisy (339-751, not clearly climbing) while steps/epoch exploded — likely redundant SGD on stale data. Check val/policy_ce vs cumulative steps. [analysis]
  6. ARCHITECTURAL FIX: trainer caps the tile it ingests per version (drop/defer excess) or backpressures workers during long train phases. [code]
priority: top of Tier 1 (the lab's metric was measuring the wrong thing; fix that before more knob lanes)
status: queued (filed 2026-05-23 from the LF1 run; Jason: "file these findings in detail for the lab to explore")
notes: lanes 1-3 are the load-bearing ones (fix the metric + map the runaway). Lane 4 is the north-star metric. The ANE strand + cross-engine coupling are resolved; this is the new top direction.
```

#### L09-ane-offload-prototype

```yaml
id: L09-ane-offload-prototype
tier: 1
hypothesis: A Core ML eval-worker frees the GPU from inference; even with slower raw eval the concurrent trainer step rate increases and overall R-TRAIN-ANE beats R-TRAIN-WL5.
references_affected: R-TRAIN-ANE (new); R-S* under engine isolation
code_change: true (scaffold complete, merged 9e2e687)
worktree: removed
patch_landed: selfplay_worker --evaluator {torch,coreml} + --coreml-compute-units flag, default CPU_AND_NE. Re-export on every weight reload via gomoku.coreml_evaluator.
smoke_status: green — CPU_ONLY 6 records / 3 batches; CPU_AND_NE 5 records / 2 batches; both clean exits.
measurement_cells:
  - R-TRAIN-ANE-baseline: full train+8 workers --evaluator torch (R-TRAIN-WL5 ref), 5 min stitched (warmup+measure), report epochs/sec
  - R-TRAIN-ANE-candidate: same recipe but workers --evaluator coreml --coreml-compute-units CPU_AND_NE
n_cells: 2 (after scaffold; scaffold done)
wall_cost_min: 10
E_delta_epochs_per_sec: 0.4
P_success: 0.35
priority: 4.0
status: COMPLETED 2026-05-23 REJECT (holistic) — R-TRAIN-ANE = 1,930.3 aug/s (-41.5% vs R-TRAIN-WL5 3,297.6). But trainer_step_s_p50 -56% (0.0512s→0.0227s) — MPS-relief hypothesis is real on the trainer side; the loss is workers (Core ML eval ~2× slower than MPS torch at small/V=64). Reviewer audit pending. Follow-up candidates: L09b (different compute-units routing), L09c (tiny model on ANE).
notes: First architectural lane to land a measurement. Mechanism is clean — both sides of the L11+L09 compound finding tell us: MPS contention is real and bidirectional, and the next lane needs to keep both trainer and workers happy. Driver got `--evaluator` + `--coreml-compute-units` passthrough (5c08d3c) — gates any future L09b/L09c.
```

#### L11b-V512-low-sgd-per-position (NEW, Tier 1 compound follow-up from L09+L11)

```yaml
id: L11b-V512-low-sgd-per-position
tier: 1
hypothesis: L11 showed V=512 hurts at trainer level because the buffer fills 2.4× faster and fixed sgd_per_position=0.0025 then produces 3.36× more SGD steps per epoch, monopolizing MPS. If sgd_per_position is scaled DOWN to match V=64's SGD work per second (~0.001), the trainer's per-epoch SGD time stays bounded and V=512's pure-gen win can finally shine through at the trainer level.
references_affected: R-TRAIN-LEAN (re-attempt with knob); R-TRAIN-WL5 (comparison)
code_change: false (--sgd-per-position already in L12 CLI)
depends_on: [L10, L11]
prep_cells: none
measurement_cells:
  - R-TRAIN-LEAN-b: full WL5 recipe but --wave-size 512 --sgd-per-position 0.001 (2.5× lower than default to compensate for 2.4× buffer-fill speedup); 30s warmup + 120s measure
n_cells: 1
wall_cost_min: 3
E_delta_aug_per_sec: 700 (V=512's pure-gen +50% over V=64; expect partial recovery at trainer level)
P_success: 0.35 (medium — testing a real chain of mechanism)
priority: 12.0 (Tier 1, fresh compound)
status: COMPLETED 2026-05-23 NEEDS_REPEAT — V=512 + sgd_per_position=0.001 = 4,231.8 aug/s (+28.3% vs R-TRAIN-WL5). The lever works mechanically (per-epoch wall back to L10-like, gen wins surface). BUT sgd_per_position is behavior-affecting → Training-Quality Gate applies. The perf lab can't promote on a 120s cell; needs a canary training run reporting val/policy_ce vs wl5_validation_v1.pt + plies/game-shape band. That's a training-pipeline lane, not a perf-lab one.
notes: Compound finding from L09+L11+L11b nails trainer-side MPS contention as the real cost in live training. Two distinct levers move it: (1) cap trainer SGD work (this lane); (2) relocate worker eval to ANE (L09 — works on trainer side, fails on worker side at small/V=64). The headline insight for the perf lab: the R-S* V=512 promotes (from L01) CAN compound at trainer level once the trainer's SGD rate is detuned.
```

#### L10-trainer-step-bench (R-TRAIN-WL5 baseline, redesigned 2026-05-23)

```yaml
id: L10-trainer-step-bench
tier: 1
hypothesis: First-ever R-TRAIN-WL5 measurement. Full WL5 production recipe (trainer + 8 workers + EMA τ=0.99 + grad_accum=4 + V=64), 30s warmup + 240s measure, report epochs/sec.
references_affected: R-TRAIN-WL5 (new)
code_change: false
prep_cells:
  - R-TRAIN-WL5 warmup: full WL5 recipe, 30s window, cell_status=warmup (no recording)
measurement_cells:
  - R-TRAIN-WL5 measure: same recipe, 240s window, record epochs/sec + games/sec + trainer_step_s_p50
n_cells: 2 (stitched warmup + measure)
wall_cost_min: 5
E_delta_epochs_per_sec: 0 (baseline, no comparison)
P_success: 1.0 (baseline measurement, can't fail)
priority: 10.0
status: COMPLETED 2026-05-23 — R-TRAIN-WL5 = 3,297.6 aug/s; 0.0917 epochs/s; 14.07 games/s; trainer_step_s_p50=0.0512s (14 epochs in 120s; commit 4a825f1; sweep_logs/lab-L10-20260523T132940Z). Reviewer audit pending.
notes: Redesigned from "pure trainer" to "full end-to-end" because gomoku.train doesn't have a no-workers mode without invasive changes. The number we actually care about (R-TRAIN-WL5) IS the end-to-end production cell. Two L12 driver bugs surfaced and fixed during dispatch: (1) --save-every=1M froze worker_weights publish (1dc4abb); (2) count_records undercounted because trainer ingests/deletes (4a825f1).
```

#### L11-end-to-end-cell (R-TRAIN-LEAN at V=512, rescoped 2026-05-23 after L01)

```yaml
id: L11-end-to-end-cell
tier: 1
hypothesis: The V=512 promote (from L01) compounds at the trainer level — full end-to-end recipe with V=512 beats R-TRAIN-WL5 on epochs/sec OR games/sec.
references_affected: R-TRAIN-LEAN (new); R-TRAIN-WL5 (comparison)
code_change: false
depends_on: [L10]
prep_cells:
  - R-TRAIN-LEAN warmup: full WL5 recipe but --wave-size 512, 30s window, cell_status=warmup
measurement_cells:
  - R-TRAIN-LEAN measure: same, 240s window, record epochs/sec + games/sec + trainer_step_s_p50
n_cells: 2
wall_cost_min: 5
E_delta_epochs_per_sec: 0.2
P_success: 0.55
priority: 11.0 (recomputed after L01)
status: COMPLETED 2026-05-23 REJECT — R-TRAIN-LEAN V=512 = 2,362.8 aug/s (-28.4% vs R-TRAIN-WL5 3,297.6). V=64 stays the R-TRAIN-WL5 default. Mechanism: V=512 fills buffer 2× faster → 3× more SGD steps per epoch (steps=306 vs ~89) → 43s of train-time per epoch starves workers of MPS → games/s drops 40%. Reviewer audit pending.
notes: The L11 yaml's own caveat ("R-S* metrics need humility") was confirmed. Pure-gen R-S* promotes (V=512) remain valid for non-trainer self-play but do NOT free-ride to live training. Follow-up L11b candidate: V=512 + lower sgd_per_position (to match V=64's SGD work per second). Lower priority than L09 — the headline finding (gen wins don't free-ride) is the load-bearing insight.
```

#### L09e' — ANE residency proof via thread-name on L09c (cap elevation: `coreml-isolated` → `ane-metered`); UNBLOCKED post-hollance-absorption

```yaml
id: L09e-prime (L09e')
tier: 1
hypothesis: Re-run L09c (lone L09* PROMOTE: tiny / W=16 / V=64 / coreml CPU_AND_NE = 10,762.6 aug/s, +33.9% vs torch baseline) and use the `ps -M <worker_pid>` thread-name check (per hollance/neural-engine) to detect `H11ANEServicesThread` in each worker process during the measurement window. If the thread exists in workers, Core ML is using the ANE for at least part of the forward → cap elevates `coreml-isolated` → `ane-metered`. If thread absent in all 16 workers, the L09c win is engine-isolation via Core ML's CPU fallback (under CPU_AND_NE, the alternative to ANE is CPU since GPU is excluded by the routing flag) — both readings are durable, but the framing for future ANE-targeting work differs.
references_affected: R-TRAIN-TINY-ANE cap (currently `coreml-isolated`; would elevate to `ane-metered` or stay capped depending on thread evidence).
code_change: false (uses existing `--evaluator coreml --coreml-compute-units CPU_AND_NE` flags + standard `ps -M` for thread inspection).
depends_on: nothing beyond existing infrastructure. **No sudo required** (hollance technique bypasses the sudo-powermetrics blocker that grounded the 2026-05-22 lane 03 attempt).
prep_cells: none (the thread-name technique doesn't need a sanity control — either the thread is present in the running process or it isn't).
measurement_cells:
  - Re-run L09c shape (tiny / W=16 / G=8 / S=400 / V=64 / coreml CPU_AND_NE / live training, 30s warmup + 120s measure)
  - During the measurement window, snapshot `ps -M <pid>` for each of the 16 worker PIDs (~3 samples spaced through the window to catch any transient threads)
  - Optional: `lldb -p <pid>` against one worker + `image list Espresso` to log which Espresso engines are loaded (ANERuntimeEngine / MPSEngine / BNNSEngine)
n_cells: 1 (single L09c shape; thread-name probe is in-band)
wall_cost_min: 3
E_delta_aug_per_sec: 0 (diagnostic; aug/s should match L09c's 10,762.6 within session-thermal noise — verify reproducibility along the way)
P_success: 0.95 (the experiment will produce either positive ANE evidence OR a clean negative — both are informative; failure modes: (a) workers complete too fast to capture threads, mitigated by sampling 3× through the window; (b) Espresso engines lazy-load and only appear after first inference)
priority: 5.0 (Tier 1 diagnostic; load-bearing for the L09c PROMOTE narrative)
status: **COMPLETED 2026-05-23 — RESOLVED: L09c is CPU/BNNS, NOT ANE.** `sample` showed hot path `BnnsCpuInferenceOperation`, no `H11ANEServicesThread`; system GPU monitor confirmed ANE 0%. Cap pinned at `coreml-isolated`. The "tiny fits ANE design center" hypothesis is falsified. See Completed table + experiment-ledger 2026-05-23 "L09e' RESIDENCY RESOLVED".
notes: Used the hollance/neural-engine no-sudo thread-name + Espresso engine-attribution technique (the sudo-blocked powermetrics path was bypassed). KEY follow-up: the 2026-05-22 residency scout showed the model IS ANE-capable under a fixed-fused export, so our `coreml_evaluator` mlprogram export is what loses residency → L09i is now the load-bearing lane.
```

#### L09c-ALL — Re-run L09c at `--coreml-compute-units ALL` (post-hollance-absorption)

```yaml
id: L09c-ALL
tier: 1
hypothesis: hollance recommends `.all` as the right setting for "I want ANE if possible." `CPU_AND_NE` excludes the GPU entirely, so when Core ML hits an unsupported op the fallback goes to slow CPU. With `ALL`, fallback ops can go to the GPU instead (faster). L09e measured ALL marginally beating CPU_AND_NE by +3.1% at small/V=64. At tiny/V=64 (L09c's shape), ALL might widen the +33.9% win further if any fallback ops exist.
references_affected: R-TRAIN-TINY-ANE (potential lift if ALL > CPU_AND_NE at tiny).
code_change: false (existing `--coreml-compute-units ALL` flag).
prep_cells: none (compare directly against L09c's 10,762.6 baseline)
measurement_cells:
  - tiny / W=16 / G=8 / S=400 / V=64 / coreml ALL / 30s warmup + 120s measure
n_cells: 1
wall_cost_min: 3
E_delta_aug_per_sec: 0-300 (marginal-to-modest improvement expected; if large, L09c's mechanism narrative shifts toward "GPU fallback" rather than "ANE win")
P_success: 0.7 (cheap to run; informative regardless)
priority: 3.5 (Tier 1 follow-up; below L09e' in priority because L09e' is the load-bearing diagnostic)
status: queued
notes: **Important caveat:** under ALL, Core ML can route workers to the GPU, which means workers share Metal with the trainer — potentially defeating engine-isolation. Verify by inspecting trainer_step_s_p50 against L09c's 0.0267s; if trainer_step regresses to torch-baseline-equivalent (~0.0319s), workers are on the GPU and ALL is a different experiment than expected. See [coreml-design-envelope-and-our-fit.md § L09c-ALL](../topics/coreml-design-envelope-and-our-fit.md#l09c-all--re-run-l09c-at---coreml-compute-units-all-auto-queued-post-hollance-absorption).
```

#### L09i — `.mlpackage` op inspection (CPU queue / no GPU; diagnostic)

```yaml
id: L09i
tier: 3
hypothesis: `coremltools.convert(..., convert_to="mlprogram")` may emit ANE-hostile broadcastable / ND-layered ops in our exported `.mlpackage`, even though `gomoku/model.py` is structurally ANE-friendly. If tiny's `.mlpackage` has fewer ND-broadcastable ops than small's or medium's, that mechanically explains the L09c PROMOTE / L09 + L09d REJECT pattern: tiny stays mostly ANE-resident, larger models fall back to slow CPU more.
references_affected: diagnostic — could motivate model-surgery rescue lanes for medium and small if ND ops are the culprit.
code_change: false (uses coremltools.models.MLModel + .get_spec() to enumerate ML-program ops)
measurement_cells: NOT GPU cells — CPU-queue inspection.
  - Tiny: export then enumerate ops; flag any in hollance's problematic list (Broadcastable, ND, gather, dilated, big pools)
  - Small: same
  - Medium: same
n_cells: 3 (model-export inspections; <30s each) + a 4th: diff against the 2026-05-22 fixed-fused scout export
wall_cost_min: 2 (no GPU; runs as a CPU-queue agent fan-out)
E_delta_aug_per_sec: 0 (diagnostic; but motivates a potentially large ANE-residency rescue)
P_success: 1.0 (will produce op-list data regardless of outcome)
priority: 5.0 (BUMPED from 2.5 — now load-bearing. We have a positive control: the 2026-05-22 coreml_ane_residency_scout showed the gomoku model ANE-RESIDENT (4,061 mW ANE rail) under a fixed-fused export, while L09e' shows the lab_train_cell coreml_evaluator export running on CPU/BNNS. So the export path is what loses ANE residency. L09i should DIFF the two export paths' op-types to find the ANE-hostile op, then model-surgery it. If we restore ANE residency, the whole ANE envelope re-opens.)
status: **COMPLETED 2026-05-23 — RESOLVED. Not an op-type difference: the two exports emit byte-identical MIL op graphs. The ANE-hostile lever is the SYMBOLIC `ct.RangeDim` batch dim (coreml_evaluator.py:267); the ANE requires static input shapes → CPU/BNNS fallback.** Fix proven in L09i-fix below. Scratch tools: scripts/l09i_op_diff.py, scripts/l09i_batchshape_probe.py.
notes: Positive control: `coreml_ane_residency_scout.py --batch-shape fixed` hit the ANE rail 2026-05-22. The op-diff hypothesis (ND-broadcastable ops) was FALSIFIED — no op-type difference exists; it's purely the batch-dim flexibility. See experiment-ledger 2026-05-23 "L09i + L09i-fix".
```

#### L09i-fix / L09i-fix-b — static-batch export restores ANE residency (COMPLETED; reject-on-throughput, mechanism win)

```yaml
id: L09i-fix
tier: 1
hypothesis: Switching coreml_evaluator export from RangeDim to a single fixed static batch restores ANE residency and the engine-isolation win.
references_affected: R-TRAIN-TINY-ANE (residency story corrected; ref number unchanged)
code_change: true (worktree feat/perf-L09i-fix @ 3f658c9 + wave*3 batch-sizing; gomoku/coreml_evaluator.py static-batch export + pad/slice/chunk; gomoku/selfplay_worker.py fixed-batch sizing)
worktree: /Users/jason/code/gomoku-perf-L09i-fix — INTACT, not merged (pending Reviewer advice on static-batch-default merge)
residency: CONFIRMED via `sample` twice (isolated micro-probe + UNDER LIVE WORKER) — hot path AneInferenceOperationImplUsingAnefAPIs / _ANEClient doEvaluateDirect / AppleNeuralEngine, 0 BNNS. First genuine ANE residency in the lab.
measurement_cells (tiny / W=16 / G=8 / S=400 / V=64 / coreml CPU_AND_NE / live training / 30s+120s):
  - L09i-fix (fixed batch 1024 = wave*G*2): aug/s=2,303.9, ep/s=0.05, trainer_step=0.0155, 8 epochs, plies 32.09 (~7x padding tax)
  - L09i-fix-b (fixed batch 192 = wave*3): aug/s=7,697.7, ep/s=0.1167, trainer_step=0.0172, 18 epochs, plies 26.76 (~1.37x padding; +234% vs fix-a)
delta: L09i-fix-b vs torch baseline 8,039.1 = -4.2% aug/s but +250% epochs/s; vs CPU/BNNS L09c 10,762.6 = -28.5% aug/s but +180% epochs/s. Workers fully vacate the GPU → best trainer_step + epochs/window the lab has measured.
status: COMPLETED 2026-05-23 — REJECT on throughput at tiny/V=64 (best ANE cell 7,698 < torch 8,039 < CPU/BNNS 10,762), CONFIRMED capability/mechanism win. NOT a knob-failure reject — re-opens the ANE envelope. **Reviewer APPROVE** (all math/claims verified: RangeDim at coreml_evaluator.py:267, sample residency, deltas exact, plies-decline confirmed in trainer.log).
merge_decision: **DEFER — do NOT merge static-batch as the coreml default** (Reviewer advice, sound): coreml is not the production path; the default sizing wave*G*2=1024 craters (-78.6%); a wrong-sized default is a footgun; follow-ups run from the worktree and pass the fixed-batch knob explicitly. Worktree feat/perf-L09i-fix kept INTACT for the follow-up lanes. If the strand pays off (L09i-fix-load), refactor to an opt-in `--coreml-static-batch N` flag (RangeDim default when unset) before merging.
notes: Single fixed batch is the ONLY ANE-placeable option (EnumeratedShapes falls back to BNNS). The static export pays a per-eval padding tax (one shape, every call padded up) in exchange for total GPU vacancy. fp16 eval numerically equivalent to torch (MAE ~9e-5). plies 26.76 is the asymmetric-epoch artifact (18 epochs → within-window policy improvement → shorter games), not drift.
```

#### L09i-fix-load — ANE-resident workers under GPU saturation (the potential architectural headline)

```yaml
id: L09i-fix-load
tier: 1
hypothesis: Lpwr showed GPU saturation collapses CPU/BNNS-resident workers -82% (shared package envelope). ANE-resident workers sit on a SEPARATE engine from the GPU trainer — they may resist the GPU-contention collapse. If so, the ANE path's value isn't raw tiny throughput (it loses there) but contention-immunity under a heavy production trainer — exactly where CPU/BNNS fails.
references_affected: re-frames the R-TRAIN-TINY-ANE load-fragility caveat; potential new architectural win.
code_change: false (gpu_load_generator.py + the L09i-fix worktree both exist)
design: cold-chip interleaved A/B (per the Lpwr2 corrected design) — ANE-resident workers (L09i-fix worktree, fixed batch 192) with and without a concurrent GPU hog; compare worker aug/s collapse to Lpwr's CPU/BNNS -82%.
n_cells: ~4 (hog/no-hog × maybe 2 intensities), cold chip + cooldowns
wall_cost_min: ~12
E_delta_aug_per_sec: 0 (diagnostic; but could establish the ANE's production value)
P_success: 0.6
priority: 6.0 (Tier 1 — highest; could establish the ANE's actual production niche)
status: **COMPLETED 2026-05-23 — needs_repeat (INCONCLUSIVE). Reviewer REVISE → corrected.** Interleaved A/B (worktree, ANE-resident, fixed batch 192): arm A no-hog = 7,878 aug/s / 19 epochs; arm B + ~10.7 TFLOP/s GPU hog = 302 aug/s / 4 epochs. **The −96% is NOT an ANE-worker collapse** — trainer.log shows worker gen held EXACTLY (gen=5.1s) under the hog while the MPS TRAINER stalled (train 2.5→99.5s; per-step p50 barely moved → MPS-queue contention). Wave-mode gates worker output on the trainer epoch loop, so the stalled trainer tanked aug/s. ANE workers were NOT throttled — a contention-RESISTANCE signal, opposite the (now-retracted) "falsified" read. Lpwr comparison invalid (there workers slowed; here trainer stalled). Strand NOT closed.
notes: RETRACTED: "ANE most load-fragile / second collapse datapoint / closes strand." The lane conflates trainer-stall with worker-collapse, and the synthetic hog isn't trainer-representative (a real heavy trainer is GPU-heavy via its own SGD, not a separate MPS-queue-flooding hog). Re-test needed → L09i-fix-load-v2 below. ANE residency stays a documented capability (branch feat/perf-L09i-fix). See experiment-ledger 2026-05-23 "L09i-fix-load [CORRECTED]".
```

#### L09i-fix-load-v2 — decoupled worker-contention test (re-do of the confounded fix-load)

```yaml
id: L09i-fix-load-v2
tier: 1
hypothesis: The one clean signal from L09i-fix-load — ANE worker gen-rate held (5.1s) under a GPU hog while the MPS trainer stalled — leans POSITIVE for ANE worker contention-resistance. Test it cleanly: measure pure worker generation rate decoupled from the trainer epoch barrier, so the metric reflects the workers, not the trainer.
references_affected: the (reopened) ANE-for-self-play contention question; the [[project-light-all-engines]] model.
code_change: maybe (a pure-self-play coreml path under a GPU hog, OR instrument worker wave-rate independent of the trainer loop)
design:
  - Option A: run pure self-play (canonical_sweep, no trainer) with ANE-resident workers, with/without the GPU hog; compare worker aug/s directly (no wave-barrier confound). Mirrors how Lpwr measured CPU workers.
  - Option B: keep lab_train_cell but log per-worker wave-rate separately from the trainer epoch advance.
  - Use a trainer-representative GPU load if testing the holistic question (not a synthetic matmul hog that floods the MPS command queue differently than SGD does).
n_cells: ~4 (with/without hog × maybe 2)
wall_cost_min: ~10
E_delta_aug_per_sec: 0 (diagnostic; resolves whether ANE workers are genuinely contention-resistant)
P_success: 0.7
priority: 5.0 (Tier 1 — the reopened contention question; the cleanest lane in the ANE strand now)
status: **COMPLETED 2026-05-23 — REJECT (no contention-immunity win). CLEAN result.** Pure self-play (no trainer barrier), interleaved A/B, tiny/coreml CPU_AND_NE: arm A no-hog = 3,548 aug/s; arm B + GPU hog = 2,307 aug/s = **−35%**. ANE workers DO throttle under GPU load — NOT immune; the "positive lean" from fix-load is RETRACTED (it was the trainer-barrier artifact). Bidirectional coupling: the 16 ANE workers suppressed the GPU hog to ~2.72 TFLOP/s (vs ~10.7 alongside a light trainer) — ANE↔GPU brown each other out via shared package power. Reviewer audit pending.
notes: Reconciles fix-load (trainer stalled → workers idled on barrier → package power free → hog hit 10.7; gen "held" only during bursts). Hog intensities NOT matched (2.72 vs Lpwr 11) so no clean ANE-vs-CPU fragility ranking. New bidirectional-coupling datapoint for m5-max-cross-engine-coupling.md + Lpwr2. See experiment-ledger "L09i-fix-load-v2".
```

#### L09i-fix-c — tighter fixed batch (chase remaining padding)

```yaml
id: L09i-fix-c
tier: 1
hypothesis: L09i-fix-b at fixed batch 192 still pads ~1.37x over the ~140-leaf wave tile. A tighter fixed batch (160 / 144) cuts padding further; if ANE eval scales with the padding factor, throughput could close the -4.2% gap to torch (or beyond) at tiny/V=64.
references_affected: R-TRAIN-TINY-ANE (could open an ANE-resident promote if it passes torch).
code_change: true (worktree fixed-batch sizing knob, already parameterized in L09i-fix)
measurement_cells: tiny/W16/G8/S400/V64/coreml CPU_AND_NE at fixed batch ∈ {160, 144} (chunk handles rare larger waves)
n_cells: 2
wall_cost_min: 7
E_delta_aug_per_sec: 400 (parity-to-modest; diminishing returns as padding →1x)
P_success: 0.45
priority: 3.0 (restored — the L09i-fix-load "load-fragility kills it" downweight was RETRACTED; the contention question is reopened with a positive lean)
status: queued. Tighten the fixed batch (160/144) to chase the −4.2% throughput gap to torch at tiny. Secondary to L09i-fix-load-v2 (which resolves whether ANE workers are contention-resistant — the more load-bearing question).
notes: Cheap follow-up on the worktree. Watch for chunking overhead if waves frequently exceed the fixed size.
```

#### L09-ANE-resident-reopen — re-run L09 (small) + L09d (medium) with genuine ANE residency

```yaml
id: L09-ANE-resident-reopen
tier: 1
hypothesis: L09 (small, -41.5%) and L09d (medium, -59.6%) REJECTED Core ML offload — but those were CPU/BNNS, not ANE. The ANE's design center is conv-heavy compute; small/medium have more FLOPs per eval to amortize the ANE pipeline + padding overhead. True ANE residency might flip the sign at small/medium where it didn't help at tiny.
references_affected: R-TRAIN-ANE (small), R-TRAIN-MEDIUM-ANE — both currently REJECT on CPU/BNNS.
code_change: false (uses L09i-fix worktree static-batch export; fixed batch sized to each shape's wave tile)
measurement_cells: small/V=64 and medium/V=512 at coreml CPU_AND_NE, ANE-resident export, matched torch baselines; sample-verify residency per shape.
n_cells: 2 (+ baselines exist)
wall_cost_min: 8
E_delta_aug_per_sec: 0 (envelope re-mapping; the prior envelope was all CPU/BNNS)
P_success: 0.4 (the ANE design-center argument is plausible but unproven at these sizes)
priority: 3.5 (restored — the L09i-fix-load downweight was RETRACTED; with worker contention-resistance now an open positive question, re-mapping small/medium on real ANE regains production relevance)
status: queued. Does real ANE residency beat CPU/BNNS at small/medium where it lost at tiny? Now also relevant to whether larger ANE workers stay contention-resistant. Pairs with L09i-fix-load-v2.
notes: The whole 5-lane ANE envelope was mapped on CPU/BNNS — this re-maps it on the real engine. Verify residency with `sample` per shape (medium may have ANE-unsupported ops that re-trigger fallback).
```

#### Lpwr2 — Cross-engine power-coupling, clean cold-chip re-run

```yaml
id: Lpwr2
tier: 1
hypothesis: The Lpwr clean A/B showed GPU saturation collapses CPU-resident workers −82% (shared package resource). The intensity sweep to pin the mechanism (power-throttle vs scheduling vs mem-bandwidth) was thermally confounded. Re-run on a COLD chip with interleaved A/B pairs (hog/no-hog back-to-back) per intensity + cooldowns between, ideally under `powermetrics` (sudo) to directly observe CPU frequency/power throttle.
references_affected: the load-fragility caveat on R-TRAIN-TINY-ANE; the [[project-light-all-engines]] power-ceiling model.
code_change: false (gpu_load_generator.py + lab_train_cell already exist); optionally add a fp16 hog mode to separate power-per-FLOP from FLOPs.
design:
  - Let the chip cool to idle first (pgrep clean + a few min idle; ideally check Tdie if powermetrics available).
  - For each intensity in {off, 2048, 4096, 8192} (and optionally fp16 variants): run no-hog cell, then hog cell, BACK TO BACK (interleaved), so each A/B pair shares a thermal state.
  - Cooldown ~60-120s between intensity pairs.
  - If powermetrics/sudo available: sample CPU+GPU+ANE power during each arm to directly attribute the throttle.
n_cells: ~8-10 (4 intensities × 2 arms, + optional fp16)
wall_cost_min: ~25 (with cooldowns — this one is NOT smoke-first; it needs thermal control)
E_delta_aug_per_sec: 0 (diagnostic; pins the mechanism)
P_success: 0.7 (cold-chip + interleaving should de-confound; powermetrics would clinch it)
priority: 3.0 (Tier 1 diagnostic; lower than L09i which could re-open the whole ANE envelope)
status: **COMPLETED 2026-05-23 — needs_repeat.** Cold-chip interleaved sweep (ANE workers, pure self-play), hog matrix {2048,4096,8192}: worker throttle −8.8% / −21.6% / −26.0%; every hog self-suppressed to 2.2-3.0 TFLOP/s. Mutual coupling reproduced across intensities. MECHANISM HINT: at matched ~2.2 TFLOP/s (m2048 vs m4096), the bigger-footprint hog throttled ~2.5× more → bandwidth/footprint, not pure power. But no-hog baseline noisy (~20%, run-variance at 80s cells) → not definitively pinned. → Lpwr2b discriminator below. See experiment-ledger + perf-log "Lpwr2".
notes: The Lpwr sweep (2026-05-23) is the worked example of why sequential sweeps fail for thermally-sensitive signals; this corrected interleaved design worked but surfaced a NEW noise floor (ANE pure-self-play aug/s ±15-20% at 80s cells). See m5-max-cross-engine-coupling.md.

#### Lpwr2b — fp16-vs-fp32 hog at matched matrix (the power-vs-bandwidth discriminator)

```yaml
id: Lpwr2b
tier: 1
hypothesis: Lpwr2 hinted the cross-engine throttle tracks hog memory FOOTPRINT, not power/FLOPs. Discriminate cleanly: fp16@4096 has HALF the byte-footprint of fp32@4096 but MORE FLOP-rate, so the SIGN of its worker-throttle-vs-fp32 separates the hypotheses — fp16 throttles LESS → bandwidth/footprint dominates; fp16 throttles MORE → FLOPs/power dominates.
references_affected: the cross-engine coupling mechanism (m5-max-cross-engine-coupling.md); informs whether "reduce total package load" should mean reduce FLOPs or reduce memory traffic.
code_change: false (fp16 hog mode added commit 8edd16b; --dtype {fp32,fp16})
design: 120s cells (cut the 80s noise), bracketed no-hog (start+end) for baseline; back-to-back fp32-hog then fp16-hog at matrix 4096 (shared thermal state → directly comparable). Compare the two hog cells' worker aug/s.
n_cells: 4 (2 no-hog bracket + fp32-hog + fp16-hog)
wall_cost_min: ~13
E_delta_aug_per_sec: 0 (diagnostic; pins the mechanism)
P_success: 0.7
priority: 3.5 (Tier 1 — completes the mechanism pin Lpwr/Lpwr2 chased)
status: **COMPLETED 2026-05-23 — RESOLVED.** Matched matrix 4096: fp32-hog 1.98 TFLOP/s → workers −15.9%; fp16-hog **7.03 TFLOP/s (3.5×)** → workers −14.8%. **3.5× more FLOPs = ~0 extra throttle** (1.3% apart, inside 6.6% noise). Cross-engine throttle is FLOP-rate-INDEPENDENT (and ~byte-rate-independent). Combined with Lpwr2 (throttle ∝ matrix SIZE), the coupling tracks GPU **working-set/occupancy, NOT compute throughput**. Compute-power ruled out. Resolves Lpwr2's needs_repeat.
notes: Actionable: cut the GPU's memory working-set/occupancy to reduce cross-engine contention, not its FLOPs. 120s cells tightened the no-hog baseline to 6.6% (from Lpwr2's 20%). See experiment-ledger + perf-log "Lpwr2b".
```
```

#### Lhot — Heat-soaked steady-state reference characterization (production-representative numbers)

```yaml
id: Lhot
tier: 1
hypothesis: Our R-S* and R-TRAIN-* reference numbers are cool-start (each cell ~3 min from a relatively cool chip). Production runs heat-soaked for hours. Measure the heat-soaked steady-state throughput for the key production-shaped cells to know the real sustained number, not the cold-start peak. Tiny/V=64 showed ~18% heat-soak haircut (10,431→8,531); the production cells (R-S400 small, R-TRAIN-WL5) need their own measurement.
references_affected: establishes "*-hot" companion numbers for R-S400, R-TRAIN-WL5, R-TRAIN-LEAN-fp16 — NOT replacing the cool-start refs, but annotating them with the sustained-operation reality.
code_change: false (just a warmup-to-steady-state protocol around existing cells)
design:
  - Warm the chip to thermal steady state first: run the target recipe continuously for ~10-15 min (or until per-cell throughput stops declining cell-over-cell), THEN measure.
  - Measure R-S400 (small/V=512/fp16) and R-TRAIN-WL5 (small/V=64) at steady state.
  - Compare to the cool-start best-cells numbers; record the heat-soak haircut per reference.
  - If powermetrics/sudo available, log package power + Tdie to characterize the steady-state thermal point.
n_cells: ~4-6 (warmup cells + 2-3 measured references)
wall_cost_min: ~20 (includes the warmup-to-steady-state period — NOT smoke-first; thermal steady state is the point)
E_delta_aug_per_sec: 0 (re-characterization; produces production-truth numbers, possibly revises headline claims downward)
P_success: 0.9 (will produce the heat-soaked numbers)
priority: 3.5 (Tier 1; production-relevance is high, but below L09i which could re-open the ANE envelope)
status: **COMPLETED 2026-05-23 — NO haircut on production shapes.** R-S400 steady state ≈ 9,783 (+4% vs cool); R-TRAIN-WL5 ≈ 3,381 (+2.5% vs cool). Cold-start refs trustworthy for production. See Completed table + experiment-ledger "Lhot heat-soak characterization". Withdrew the earlier 18%-haircut claim (post-hog CPU-worker artifact).
notes: Per [[feedback-heat-soaked-is-production]] — Jason 2026-05-23: "heat soaked numbers are not bad to know, training will be heat soaked." Follow-up Lhot2 (optional): clean heat-soak re-test of the tiny/V=64 Core ML CPU-worker shape (no prior hog) to confirm/refute the engine-specific CPU-throttle nuance.
```

#### Lhot2 — Clean heat-soak re-test of the tiny/V=64 CPU-worker shape (engine-specific throttle?)

```yaml
id: Lhot2
tier: 1
hypothesis: Lhot showed NO heat-soak haircut on the GPU-resident production shapes (R-S400, R-TRAIN-WL5 both sustained at/above cool refs). But the one tiny/V=64 Core ML CPU/BNNS-worker data point fell 10,431→8,531 (~18%) under heat-soak — measured messily right after the synthetic 14-TFLOP hog. If real, the haircut is ENGINE-SPECIFIC: the GPU holds its clocks under sustained load, the CPU/BNNS path throttles. Clean-test it: heat-soak the chip with the tiny/V=64 CPU-worker recipe itself (NO prior hog), watch the curve.
references_affected: the engine-specific-throttle nuance in m5-max-cross-engine-coupling.md; informs whether CPU-offload self-play is thermally durable.
code_change: false
design:
  - From a cool/idle chip, run N back-to-back tiny/W16/V64/coreml CPU_AND_NE cells (lab_train_cell, 30s warmup + 60s measure), logging the aug/s curve — same protocol as Lhot Phase 1 but on the CPU-worker shape.
  - If the curve decays and plateaus below the cool-start ~10,400, the CPU/BNNS path throttles under sustained heat → engine-specific haircut confirmed.
  - If it holds (like R-S400 did), the earlier 8,531 was pure hog-contamination and there's no CPU-specific throttle.
n_cells: ~6-8 (curve to steady state)
wall_cost_min: ~10
E_delta_aug_per_sec: 0 (diagnostic; resolves the engine-specific-throttle nuance)
P_success: 0.9 (will produce a clean curve)
priority: 2.5 (Tier 1 diagnostic; below L09i/Lpwr2 — it's a nuance, not load-bearing for production since CPU-worker self-play isn't the production path)
status: queued
notes: Per [[feedback-heat-soaked-is-production]] open nuance. The clean version of the contaminated post-hog tiny/V=64 measurement. See m5-max-cross-engine-coupling.md "surviving nuance".
```

### Tier 2 — Compound knob wins

### Tier 3 — Speculative knob lanes

The original MPS knob-lanes (L05 torch.compile, L06 fp16-eval, L08
heap-ratio) all COMPLETED 2026-05-23 — see the Completed table.
Outcomes: L05 compile reject (neutral on MPS); L06 fp16 **promote**
(R-S400 +97%, the headline MPS win); L08 heap-ratio reject (null,
bandwidth-bound regime). The MPS-side knob axis is exhausted. Remaining
Tier-3 work is the downweighted Core ML envelope sweeps below.

### Background — Calibration / reference



#### L09f — Larger wave sizes on Core ML

```yaml
id: L09f
tier: 3
hypothesis: V=64 is Core ML's worst case for our workload (low per-call work). V=512+ batches more leaf evals per forward, amortizing the pipeline overhead. The amortization may shift the regime where Core ML competes with torch/MPS.
references_affected: ANE envelope mapping along the V-axis.
code_change: false
measurement_cells:
  - small V=512 / coreml CPU_AND_NE (vs L06-followup torch/MPS = 9,398.5 aug/s)
  - small V=1024 / coreml CPU_AND_NE
  - small V=2048 / coreml CPU_AND_NE (if model max_batch supports)
n_cells: 3
wall_cost_min: 9
E_delta_aug_per_sec: 500
P_success: 0.4
priority: 2.5
status: queued
notes: V-axis is the cheapest amortization test. See wiki/topics/coreml-design-envelope-and-our-fit.md § L09f.
```

#### L09g — Core ML model-size sweep at V=512 (pure self-play; envelope analog of MPS Finding 2)

```yaml
id: L09g
tier: 3
hypothesis: Map where the bandwidth-bound transition is for Core ML, as we did for MPS torch in Finding 2 (m5-max-fp16-and-throughput-regimes.md). Where does Core ML's per-call overhead stop dominating?
references_affected: ANE envelope mapping along the model-size axis.
code_change: false
measurement_cells (canonical_sweep, pure self-play):
  - tiny / V=512 / coreml CPU_AND_NE
  - small / V=512 / coreml CPU_AND_NE
  - medium / V=512 / coreml CPU_AND_NE
n_cells: 3
wall_cost_min: 4 (no trainer; pure self-play 60s cells)
E_delta_aug_per_sec: 300 (diagnostic value)
P_success: 0.8 (high — the data will be informative regardless of which size wins)
priority: 2.0
status: queued
notes: The chip-level analog of Finding 2 but for Core ML. See wiki/topics/coreml-design-envelope-and-our-fit.md § L09g.
```

#### L09h — `.mlpackage` re-export cost amortization (diagnostic)

```yaml
id: L09h
tier: 3
hypothesis: In live training, Core ML re-exports the model on every weight version. That overhead could dominate cell wall time at fast epochs. Measure the re-export cost directly; if > 5% of wall time, propose a caching scheme.
references_affected: viability of any future R-TRAIN-ANE-medium production lane.
code_change: instrument gomoku/coreml_evaluator.py to log per-export wall time.
n_cells: 1 (instrument + re-run an existing L09 cell)
wall_cost_min: 3
E_delta_aug_per_sec: n/a (diagnostic)
P_success: 0.9 (will produce a number)
priority: 1.0
status: queued
notes: If the cost is significant, motivates a delta-encoding or differential-compile cache.
```


## Completed

| date | id | resolution | best cell from lane | reviewer | notes |
|---|---|---|---|---|---|
| 2026-05-23 | L09-reopen-medium | **residency CONFIRMED at medium (completes tiny/small/medium); throughput confounded (11× over-pad)** | `sample` hot path on AppleNeuralEngine — medium model is ANE-placeable, no unsupported ops. aug/s=453 but batch-768 vs ~70 tile = ~11× pad (wave tile ≈ W×G, NOT wave_size). | pending | All 3 sizes go ANE-resident under the static-batch export. Throughput directionally ANE≈CPU/BNNS<torch (clean re-run not worth it — diminishing returns). NEW lesson: size `--coreml-static-batch` to ~W×G×1.3 (~96 at W8G8, ~192 at W16G8), not wave_size. |
| 2026-05-23 | Lpwr2b | **RESOLVED — cross-engine throttle is FLOP-rate-INDEPENDENT (occupancy/working-set, not compute-power)** | Matched matrix 4096: fp32-hog (1.98 TFLOP/s) → −15.9%; fp16-hog (7.03 TFLOP/s, 3.5×) → −14.8%. 3.5× FLOPs = ~0 extra throttle. | pending | Pins the Lpwr strand: coupling tracks GPU working-set/occupancy, not compute throughput; compute-power ruled out. Actionable: shrink GPU footprint/occupancy, not FLOPs, to cut cross-engine contention. 120s cells → 6.6% baseline noise (vs Lpwr2's 20%). |
| 2026-05-23 | Lpwr2 | **needs_repeat — coupling reproduced; mechanism hints bandwidth/footprint > power** | Interleaved ANE-worker sweep: worker throttle −8.8%/−21.6%/−26.0% at hog matrix 2048/4096/8192; hogs self-suppressed to 2.2-3.0 TFLOP/s. At matched ~2.2 TFLOP/s, bigger footprint → ~2.5× more throttle. | pending | Mutual coupling reproduced across intensities. Mechanism hint (footprint, not power) but no-hog baseline noisy (~20% at 80s cells) → Lpwr2b (fp16-vs-fp32 @ matched matrix) is the clincher, running. New noise-floor lesson (ANE pure-self-play ±15-20% at 80s). |
| 2026-05-23 | L09-reopen-small (+ -b clean) | **REJECT — real ANE residency at small ≈ CPU/BNNS, loses to torch −44% (no flip)** | ANE residency CONFIRMED at small (`sample`). Throughput: first run over-padded (wave×3=192 vs ~67 tile) = 1,271; clean re-run (`--coreml-static-batch 96`) = **1,834 aug/s** ≈ CPU/BNNS L09 (1,930, −5%), −44% vs torch R-TRAIN-WL5 (3,297). | pending | The `--coreml-static-batch` knob (d94fb98) fixed the padding confound (+44%). Real ANE residency does not beat torch at small — roughly ties the CPU/BNNS path. Completes the throughput re-map: ANE loses at tiny AND small. Surfaced the wave×3-mis-sizes-at-low-W friction. reopen-medium left optional. |
| 2026-05-23 | L09i-fix-load-v2 | **REJECT — clean contention test: ANE workers throttle −35% under GPU load (NOT immune); bidirectional package-power coupling** | Pure self-play (no trainer barrier), interleaved A/B, tiny/coreml: no-hog 3,548 → +GPU-hog 2,307 aug/s = −35%. Hog suppressed to ~2.72 TFLOP/s (vs ~10.7) by the 16 ANE workers. | pending | The decoupled re-do of fix-load. "Positive lean" RETRACTED — ANE is not contention-immune; ANE↔GPU brown each other out via shared package power. Reconciles fix-load (trainer-stall artifact). Hog intensities unmatched, so no clean ANE-vs-CPU ranking. Closes the contention question on clean evidence. |
| 2026-05-23 | L09i-fix-load | **needs_repeat — INCONCLUSIVE (Reviewer REVISE → corrected). The −96% was a TRAINER stall, not an ANE-worker collapse.** | Interleaved A/B, ANE-resident workers: no-hog = 7,878 aug/s / 19 epochs; +~10.7 TFLOP/s hog = 302 aug/s / 4 epochs. trainer.log: worker gen held EXACTLY (gen=5.1s) under the hog; MPS trainer stalled (train 2.5→99.5s). Wave-mode gates aug/s on the trainer epoch loop → holistic collapse is trainer-driven. | REVISE→corrected | RETRACTED: "contention-immunity falsified / ANE most-fragile / closes strand / 2nd collapse datapoint." Truth: ANE workers were NOT throttled (gen held) — a contention-RESISTANCE signal, opposite the CPU/BNNS workers in Lpwr. Strand REOPENED with a positive lean. Lane confounded (aug/s is trainer-gated in wave-mode; synthetic hog isn't trainer-representative). Re-test → L09i-fix-load-v2 (priority 5.0, decoupled). Restored fix-c (→3.0) and L09-reopen (→3.5). Lesson: attribute holistic collapse to gen-vs-train phase from the log before concluding. |
| 2026-05-23 | L09i + L09i-fix / L09i-fix-b | **L09i RESOLVED (diagnostic); L09i-fix REJECT-on-throughput + mechanism win (ANE residency RESTORED)** | L09i-fix-b: tiny/W16/G8/S400/V64/coreml CPU_AND_NE, ANE-resident static export at fixed batch 192 = **7,697.7 aug/s** / 36.45 g/s / 0.1167 ep/s / trainer_step 0.0172 / **18 epochs/window** / plies 26.76. (L09i-fix at fixed batch 1024 = 2,303.9 — ~7× padding tax.) | pending | **L09i: the lab's Core ML export was ANE-hostile purely because of a symbolic `ct.RangeDim` batch dim (coreml_evaluator.py:267); op graphs were byte-identical to the ANE-resident scout.** Static fixed-batch export restores genuine ANE residency (`sample`-confirmed twice incl. under the live worker — hot path on AppleNeuralEngine, 0 BNNS). **Every prior L09* "ANE" lane was CPU/BNNS, never the ANE.** Throughput at tiny/V=64 is a reject (7,698 < torch 8,039 < CPU/BNNS L09c 10,762) — the single-fixed-batch padding tax + ANE-slow-at-tiny — BUT workers fully vacate the GPU (best trainer_step + 18 epochs/window the lab has measured). Re-opens the ANE envelope. Worktree feat/perf-L09i-fix INTACT (merge pending Reviewer). Follow-ups queued: L09i-fix-load (ANE under GPU saturation vs Lpwr's -82% CPU collapse — potential headline), L09i-fix-c (tighter batch), L09-ANE-resident-reopen (re-map small/medium on real ANE). |
| 2026-05-23 | Lhot (heat-soak characterization) | **needs_repeat — production shapes show NO haircut (refutes the hypothesis)** | R-S400 heat-soaked steady state ≈ **9,783 aug/s** (8-cell curve: 9641/9388/9660/10029/9902/9780/9781/9788 — wobbles through warmup, settles stable, no decay) vs cool 9,398.5 = **+4%**. R-TRAIN-WL5 heat-soaked ≈ **3,381 aug/s** (3,384/3,379; trainer_step 0.052; 14 epochs) vs cool 3,297.6 = **+2.5%**. | pending | **The cold-start references ARE trustworthy for sustained production.** M5 Max sustains production throughput indefinitely under realistic load. **Withdraws the earlier "~18% haircut"** — that was a non-production tiny/V=64 CPU/BNNS-worker shape measured right after the synthetic 14-TFLOP hog, not real training. Surviving nuance: haircut may be engine-specific (GPU sustains, CPU/BNNS may throttle) — needs clean Lhot2 re-test. Corrected best-cells caveat + coupling page + memory. |
| 2026-05-23 | Lpwr-gpu-coupling | **needs_repeat (coupling real; mechanism not pinned)** | Clean cool-chip A/B: tiny/W16/V64/coreml CPU_AND_NE workers, GPU saturated by `gpu_load_generator --matrix 8192` (~11 TFLOP/s) → worker aug/s **10,431.6 → 1,905.2 (−81.7%)**, trainer_step only +14%. Intensity sweep {0,2048,4096,8192} thermally confounded (non-monotonic; baseline drifted 10,431→8,531). | pending | GPU load collapses CPU-resident workers → engines share a package resource (power/thermal, per the trainer/worker asymmetry). **L09c win is load-fragile** — depends on GPU power headroom; a heavy production trainer would erase it. Mechanism (power vs scheduling vs mem-BW) needs a cold-chip interleaved-A/B re-run + powermetrics. New tool: `scripts/gpu_load_generator.py`. Writeup: m5-max-cross-engine-coupling.md. |
| 2026-05-23 | L09e' (residency proof) | **resolved — L09c is CPU/BNNS, not ANE** | Re-ran L09c shape (replicated 10,431.6 aug/s); `sample` shows hot path `BnnsCpuInferenceOperation`, no `H11ANEServicesThread`; system GPU monitor ANE 0%. CPU_AND_GPU cross-check (10,202 aug/s) also BNNS-CPU. | pending | **The "tiny fits ANE design center" hypothesis is FALSIFIED.** Core ML picks CPU for our tiny model under both CPU_AND_NE and CPU_AND_GPU. L09c cap pinned at `coreml-isolated`. BUT: 2026-05-22 residency scout showed the model IS ANE-capable under a different (fixed-fused) export → our `coreml_evaluator` mlprogram export is what loses residency. Sharpens L09i. Used hollance/neural-engine no-sudo technique. |
| 2026-05-23 | L09c-cpugpu | **reject (folded into L09e')** | tiny/W16/V64/coreml CPU_AND_GPU = 10,202 aug/s; sample confirms still CPU/BNNS even with GPU allowed. | n/a | Core ML's cost model picks CPU for our tiny model regardless of routing. Subsumed by L09e' residency finding. |
| 2026-05-23 | L09e | **reject (compute-units routing axis null at today's Core ML; L09 reject stands at current snapshot)** | 2 cells under live training at the L09 ref shape (small / W=8 / G=8 / S=400 / V=64 / coreml). CPU_AND_GPU = 1,908.3 aug/s (-1.1% vs L09 CPU_AND_NE); ALL = 1,989.8 aug/s (+3.1%). Across-routing spread 4.3% — within noise. ALL is marginal winner. trainer_step_s_p50 across routings: 0.0197-0.0227s (all ~half R-TRAIN-WL5's 0.0512s — MPS-relief real in all 3). | APPROVE | All three routings still ~40% below R-TRAIN-WL5 (3,297.6) under today's Core ML + evaluator pipeline. The L09 reject stands at the current Core ML / evaluator combination; re-measure when new ANE research lands. Current engine envelope snapshot: ANE wins at tiny+V=64 ONLY (L09c +33.9%); 5 measured comparison points across the ANE-axis confirm at today's stack. consecutive_rejects: 3 → 4 (session-end declared at a complete current-state snapshot, not a permanent verdict). |
| 2026-05-23 | L08-mps-heap-ratio | **reject (axis flat at R-S400/fp16)** | 3 cells at small / W=8 / V=512 / fp16: default heap = **8,937.3 aug/s**; heap=2.0 = 8,870.9; heap=0.0 = 8,927.7. Within-sweep spread 0.74% (below V=512 plateau noise floor). | APPROVE | Null result mechanistically predictable: at small/V=512/fp16 we are bandwidth-bound (per L06-followup), not MPS-memory-pressure-bound. Side data point: default-heap re-measure -4.9% vs R-S400 (9,398.5 from L06-followup at session-start) — session-thermal drift after 90 min / ~10 sequential cells; doesn't affect within-L08 comparison. NEW friction-smoothing lesson: cross-time absolute aug/s comparisons may have ~5% thermal-drift confound; back-to-back A/B remains reliable. consecutive_rejects: 2 → 3 (warning level — CONTINUE per charter, Tier-3 lanes still queueable). |
| 2026-05-23 | L09c-V512 | **reject (V-axis amortization falsified at tiny)** | Candidate (tiny / W=16 / V=512 / coreml CPU_AND_NE) = 10,609.8 aug/s / 43.94 g/s / 0.05 ep/s / trainer_step_s_p50=0.0268s (8 epochs in 120s, plies_mean 31.02); matched baseline (torch+fp16) = **13,968.6 aug/s** / 52.18 g/s / 0.0 ep/s / trainer_step_s_p50=0.0714s (1 epoch in 120s, plies_mean 33.47). **ANE vs torch+fp16 at V=512: aug/s -24.0%, games/s -15.8%, trainer_step_s_p50 -62.5%, plies_mean -7.3% (asymmetric-epoch artifact, not behavior drift — see receipt for per-epoch plies decline).** | APPROVE | V-axis amortization hypothesis FALSIFIED at tiny. torch+fp16 already extracts most of V=512 bandwidth-bound value at tiny (L06-followup's +3.6% finding). **Updated envelope: ANE wins at exactly one point (tiny + V=64), not "tiny in general".** No new ref opens. NEW friction-smoothing lesson: plies_mean NOT stationary across asymmetric-epoch R-TRAIN windows. L09f / L09g downweighted; L09e holds priority 3.0 (the only remaining ANE-rescue diagnostic). consecutive_rejects: 1 → 2. |
| 2026-05-23 | L09d | **reject (envelope-mapping data point)** | Candidate R-TRAIN-MEDIUM-ANE = 591.7 aug/s / 2.33 g/s / 0.0208 ep/s / trainer_step_s_p50=0.0444s (7 epochs in 240s, plies_mean 31.97); matched baseline R-TRAIN-MEDIUM (torch+fp16) = **1,463.3 aug/s** / 5.66 g/s / 0.0042 ep/s / trainer_step_s_p50=0.2391s (3 epochs in 240s, plies_mean 32.5). **ANE vs torch+fp16: aug/s -59.6%, games/s -58.8%, epochs/s +395%, trainer_step_s_p50 -81.4%, plies_mean -1.6%.** | APPROVE | Mechanism sharp split: trainer-side wins enormously (train=2-3s/epoch vs 11-86s as buffer fills); worker-side loses enormously (gen=30-40s/epoch on ANE vs ~6s/epoch on torch+fp16); worker loss dominates trainer gain by 2.5×. **Envelope now sharply mapped: ANE pays at TINY only (L09c +33.9%), not at SMALL (L09 -41.5%), not at MEDIUM (L09d -59.6%). The "larger compute amortizes pipeline overhead" hypothesis is FALSIFIED in our envelope.** plies_mean drift watch from L09c Reviewer confirmed null (-1.6%, no systematic ANE-vs-torch game-shape drift). Calibration: initial 120s baseline caught only 2 epochs; re-dispatched at 240s for matched-window science (per friction-smoothing log lesson). New ref R-TRAIN-MEDIUM (torch+fp16) opens at 1,463.3 aug/s. L09e priority bumped 1.5→3.0 (diagnostic value spiked: is the loss "ANE slow" or "Core ML demoted ops"?). consecutive_rejects: 0 → 1. |
| 2026-05-23 | L09c | **promote (new envelope-mapping ref pair)** | **R-TRAIN-TINY-ANE = 10,762.6 aug/s** / 49.43 g/s / 0.0417 ep/s / trainer_step_s_p50=0.0267s (7 epochs in 120s, plies_mean 29.02); matched **R-TRAIN-TINY torch baseline = 8,039.1 aug/s** / 32.48 g/s / 0.0333 ep/s / trainer_step_s_p50=0.0319s (6 epochs in 120s, plies_mean 31.84). ANE vs torch: **aug/s +33.9%**, games/s +52.2%, trainer_step_s_p50 -16.3%. | APPROVE | Opposite holistic sign to L09 (small model, -41.5%). Engine envelope along the model-size axis: ANE pays at tiny, doesn't pay at small. Lane-card under-spec note: card said n_cells:1 but the hypothesis needs a matched baseline that didn't exist; dispatched both arms back-to-back (5 min total wall) — Reviewer ratified this autonomy call. Sharply elevates L09d's prior (medium on ANE). Auto-queued L09c-V512. consecutive_rejects stays at 0. Reviewer flag: if L09d shows similar plies_mean drift, 2× repeat before stacking more ANE-positive claims. |
| 2026-05-23 | L09b | blocked (code bug + semantic redundancy) | Worker crashed at startup: `RuntimeError: Input type (float) and bias type (c10::Half) should be the same` at coreml_evaluator.py:266 (torch.jit.trace expects fp32 dummy; model was already fp16). Also: Core ML already exports at compute_precision=FLOAT16 so the lane was structurally incoherent. | n/a (failed before measurement) | Patch lands at the same commit: selfplay_worker parse_args() force-sets fp16_eval=False when evaluator=coreml; flag combination is now a graceful no-op with a printed audit line. consecutive_rejects unchanged (blocked ≠ reject). Remaining ANE candidate: L09c (tiny model on ANE) — queue for future session. |
| 2026-05-23 | L11bp (L11b') | needs_repeat (perf reference established; TQ gate for production) | R-TRAIN-LEAN-fp16 = **8,340.5 aug/s** / 32.19 g/s / 0.0667 ep/s / trainer_step_s_p50=0.0801s (11 epochs in 120s). vs L10 R-TRAIN-WL5: aug/s **+152.9%**, games/s +128.8%, epochs/s -27%. vs L11b (low-sgd alone): aug/s +97.1% (same magnitude as R-S400's fp16 win — levers stack multiplicatively as the mechanism predicted). plies_mean 32.74 preserved (gomoku not at terminal). | APPROVE (precedent-extending) | The compound finding the whole perf cycle was building toward. Two independent levers — low-sgd cures trainer-side MPS contention (L11b); fp16 doubles worker-side throughput (L06) — stack cleanly at the R-TRAIN-* family. Reviewer verified mechanism independence structurally (gomoku/train.py has zero fp16 refs; selfplay_worker._maybe_half only affects workers); product prediction 1.283 × 1.972 = 2.530 matches measured 2.529× to 4 decimals. NEEDS_REPEAT per TQ gate (sgd_per_position is behavior-affecting); fp16 is no-behavior-change per L06-followup precedent. R-TRAIN-WL5 stays at production recipe; NEW reference R-TRAIN-LEAN-fp16 opens. consecutive_rejects stays at 0. |
| 2026-05-23 | L06fu-extended | **promote × 3** | R-S200 fp16 = 16,851 (+84% vs fp32); R-S100 fp16 = 22,312 (+48% vs fp32); medium V=512 fp16 = 3,377 (NEW ref). plies_mean 15.95-15.96 unchanged. fp16 win scales with sims (S=400 +97% > S=200 +84% > S=100 +48%), consistent with bandwidth-bound regime — higher S spends more time in eval. | pending Reviewer audit | Compound finding from L06-followup. Three new best-cells reference points (R-S200, R-S100 updates; R-S400-medium new ref). consecutive_rejects unchanged at 0. Compound follow-up dispatching now: L11b' (V=512 + sgd=0.001 + fp16 workers at trainer level — combines L11b's +28% finding with L06's +97% finding). |
| 2026-05-23 | L06-followup-fp16-cells | **promote (2× promote)** | R-S400: small V=512 fp16 = **9,398.5 aug/s** (+97.2% vs fp32 4,765). R-S400-tiny: tiny V=512 fp16 = **22,873.8 aug/s** (+3.6% vs fp32 22,088). plies_mean unchanged. Mechanism: small is memory-bandwidth-limited at V=512 (fp16 halves bandwidth → ~2× win); tiny is MPS-dispatch-limited (fp16 → marginal). | APPROVE (precedent-setting) | The historic "fp16 on MPS is slow" claim is disproven for our eval workload at torch 2.11.0 + fused conv+bn. fp16 actually engaged (worker logs explicit); outputs cast to fp32 at MCTS boundary (no behavior change per L06 patch design). consecutive_rejects RESETS 3→0; **stop signal OFF; loop rejuvenated**. Compound follow-ups: re-measure R-S200/R-S100/medium under fp16; revisit L09 with fp16 workers; revisit L11b with fp16 at trainer level. |
| 2026-05-23 | L05-followup-compile-cells | reject | small V=512 --compile = 4,657 (-2.3% vs R-S400 4,765); tiny V=512 --compile = 22,001 (-0.4% vs R-S400-tiny 22,088). Both within noise (tiny solidly null; small at edge of L01 plateau spread ~0.2%, directionally consistent with compile-graph overhead). | APPROVE | torch.compile on MPS at these eval-graph shapes is neutral-to-slightly-negative; first-call compile overhead doesn't amortize at 60s smoke; tiny shape is closer to a clean null. Don't queue further `--compile` lanes. consecutive_rejects: 2→3 — **stop signal active** unless L06-followup (dispatching in parallel) provides a compound follow-up. Reviewer nit: "±2% noise floor" framing in the receipt overstates the L01 V=512 plateau spread (~0.2%); conclusion unchanged. |
| 2026-05-23 | L11b-V512-low-sgd-per-position | needs_repeat (+28.3% aug/s — behavior-affecting knob, TQ gate) | R-TRAIN-LEAN-style V=512 + sgd_per_position=0.001 = **4,231.8 aug/s** / 15.47 g/s / 0.05 ep/s / trainer_step_s_p50=0.141s (8 epochs in 120s). vs L10 R-TRAIN-WL5: aug/s **+28.3%**, games/s +9.9%, epochs/s -45% (less SGD per data → less elo gain per epoch but more data per second). Effective SGD rate: ~2.5 steps/s vs L10's ~7.3. | APPROVE (precedent-setting) | The compound finding from L11+L09+L11b: trainer-side MPS contention is the real cost in live training, and lowering sgd_per_position is one of two levers that move it. PERF FINDING: the R-S* V=512 promotes (from L01) DO compound at trainer level once the trainer's SGD rate is detuned. PRODUCTION ADOPTION blocked by TQ gate — needs canary training run with val/policy_ce vs wl5_validation_v1.pt. consecutive_rejects unchanged at 2 (needs_repeat ≠ reject; loop continues). Reviewer: APPROVE — math/mechanism exact, TQ-gate precedent correct (behavior-affecting knob → needs_repeat, best-cells row preserved as strikethrough from L11 reject). 4th charter-staleness flag from Reviewers (charter:50 R-TRAIN-LEAN row needs user touch). |
| 2026-05-23 | L09-ane-offload-prototype | reject (holistic; partial-hypothesis confirmation) | R-TRAIN-ANE = 1,930.3 aug/s / 8.00 g/s / 0.0583 ep/s / trainer_step_s_p50=0.0227s (10 epochs in 120s). vs L10 R-TRAIN-WL5: aug/s -41.5%, games/s -43.1%, epochs/s -36.4%. **trainer_step_s_p50 -55.7%** — trainer-side MPS-relief hypothesis CONFIRMED. | APPROVE | Holistic reject, mechanism-rich. Core ML eval at small/V=64 is ~2× slower than torch/MPS on the worker side; trainer-side SGD halved once workers vacated MPS. Compound chain L11+L09 says: any future trainer-throughput lane has to keep both sides happy. L11b (V=512 + low sgd_per_position) dispatched next as the natural compound follow-up. consecutive_rejects: 1→2. |
| 2026-05-23 | L11-end-to-end-cell | reject | R-TRAIN-LEAN V=512 = 2,362.8 aug/s / 8.42 g/s / 0.0083 ep/s / trainer_step_s_p50=0.138s (3 epochs in 120s). vs L10 R-TRAIN-WL5: aug/s -28.4%, games/s -40.2%, epochs/s -91%, trainer_step_s_p50 +2.7×. | APPROVE | The headline holistic finding: pure-gen R-S* V=512 promotes do NOT compound at the trainer level. Buffer fills 2.4× faster (buf=199,608 vs 83,208 at epoch 3) → 3.36× more SGD steps per epoch (306 vs 91) → 43s of trainer SGD per epoch starves workers of MPS. V=64 stays the R-TRAIN-WL5 default. consecutive_rejects: 0→1. |
| 2026-05-23 | L10-trainer-step-bench | promote (baseline) | R-TRAIN-WL5 = 3,297.6 aug/s / 0.0917 epochs/s / 14.07 games/s / trainer_step_s_p50=0.0512s (14 epochs in 120s; 30s warmup) | APPROVE | First-ever end-to-end R-TRAIN-WL5 measurement. Reviewer verified math exact (epochs_per_sec = (14-3)/120 = 0.0917; games_per_sec = 1489/105.8s post-warmup-window span; delta vs R-S400 = -30.8%); all 5 surfaces updated; baseline-promote matches canonical-sweep precedent. Two L12 driver bugs surfaced + fixed during dispatch (--save-every=1M froze worker_weights publish 1dc4abb; count_records undercounted because trainer ingests/deletes 4a825f1). consecutive_rejects unchanged at 0. |
| 2026-05-23 | L12-write-lab-train-cell-driver | promote (code) | scripts/lab_train_cell.py (726 LOC) — live-training cell driver matching canonical_sweep resumability contract; smoke green (help, dry-run, unit test on synthetic trainer logs). Companion `scripts/lab_train_cell_smoke.py` runs in <1s with no GPU. | pending Reviewer audit | Gating lane: unblocks L09/L10/L11. Branched off 8eb7e5c, merged --no-ff at 56b6... (see graph). Trainer already emits `^epoch (\d+)/M` natively (gomoku/train.py:1135) so no shim needed. |
| 2026-05-23 | L08-driver-per-cell-envvars | promote (code) | scripts/canonical_sweep.py + tests/test_canonical_sweep_envvars.py (14 tests) — optional `env` column on cells.csv (semicolon-separated KEY=VAL pairs). All 16 in-flight cells.csv files backward-compat. | pending Reviewer audit | Unblocks L08-mps-heap-ratio. Heads-up: env not part of cell_id_of() — heap-ratio lane needs to disambiguate via separate out-dirs or cell_id suffix. |
| 2026-05-23 | L06-fp16-eval | promote (code) | gomoku/selfplay_worker.py + scripts/canonical_sweep.py — `--fp16-eval` flag passes fp16=True into make_torch_evaluator at all 4 model-load sites; default off. Smoke: build_model → save → load → fuse → _maybe_half → evaluator returns finite fp32 priors+values. Core ML branch unaffected (already FLOAT16). | pending Reviewer audit | Code merge required manual conflict resolution against L05 in canonical_sweep.py; keep-both. GPU cells queued as L06-followup-fp16-cells. |
| 2026-05-23 | L05-torch-compile-mps | promote (code) | scripts/canonical_sweep.py — `--compile` flag pass-through to selfplay_worker (worker side already supported it). Smoke: help text + Popen-cmd capture both ways. | pending Reviewer audit | Pure plumbing; selfplay_worker._maybe_compile already in place. GPU cells queued as L05-followup-compile-cells. |
| 2026-05-23 | L14-tiny-G-at-W16-V512 | reject | best = tiny W=16 G=8 V=512 = 22,088 (unchanged). G=4=22,261; G=16=22,164; G=32=22,076. 0.83% total spread — G axis flat. | APPROVE | Knob-tuning exhausted at chip envelope. Remaining lanes need code work. consecutive_rejects: 1→2. |
| 2026-05-23 | L13-tiny-W-peak-probe | reject | best = tiny W=16 V=512 = 22,088 (unchanged). W=12=20,560 (-6.9%); W=20=21,553 (-2.4%); W=24=20,970 (-5.1%). Smooth bump W∈[12,20] within 7% of peak. | APPROVE | Tiny W tolerance is wider than small's sharper drop — more headroom for L09 ANE tuning. consecutive_rejects: 0→1. |
| 2026-05-23 | L07-tiny-contour | promote | R-S400-tiny: tiny W=16 G=8 V=512 = 22,088 aug/s (+201.5% vs V=64=7,326). | APPROVE | Model-dependent W peak at V=512 — tiny W=16 BEATS W=8 (opposite of small). consecutive_rejects: 2→0. Auto-queued L13 (W peak probe) + L14 (tiny G axis). |
| 2026-05-23 | L04-G-x-wave | reject | best = W=8 G=8 V=512 = 4,765 (unchanged). G=4=4,608; G=16=4,541; G=32=4,514. G mildly non-monotone at V=512 (flat at V=64) but peak still G=8. | APPROVE | Compound finding with L02: at V=512 BOTH W and G axes peak at the canonical defaults. consecutive_rejects: 1→2. |
| 2026-05-23 | L02-W-x-wave-compound | reject | best = W=8 V=512 = 4,765 (unchanged). W=4=4,367; W=12=4,501; W=16=4,504 — wave saturation moved MPS-dispatch peak from W=16 to W=8 at V=512. | APPROVE | New finding: knob wins interact non-monotonically at chip envelope. consecutive_rejects: 0→1. |
| 2026-05-23 | L03-sims-x-wave | promote (2x) | R-S200: V=512 = 9,156 aug/s (+52.5%); R-S100: V=512 = 15,082 aug/s (+35.2%) | APPROVE | V=512 carries cleanly to S=200 and S=100. Receipt: 2026-05-23 L03 entry in experiment-ledger.md. |
| 2026-05-23 | L01-wave-extrapolation | promote | small W8 G8 S400 V=512 = 4,765 aug/s | APPROVE | +17.7% over V=128; +49.5% cumulative; plateau at V=512 (V=768/1024 flat). Receipt: 2026-05-23 entry in experiment-ledger.md. |
| 2026-05-23 | L00-canonical-sweep | promote | small W8 G8 S400 V=128 = 4,048 aug/s | (pre-reviewer-era; auto-grandfathered) | The kickoff sweep; receipt under canonical-sweep-mainframe lane. |

## Stop-condition tracker

- **RESUME STATE (2026-05-23, LF1 RUNAWAY reframes the lab's metric — new top direction)**: We cashed the perf lab's headline (R-TRAIN-LEAN-fp16, "+152%") into a real 1000-epoch training run (LF1, wandb h9al2e0k). **It exhibited an UNBOUNDED per-epoch runaway**: steps/epoch 25→3236, wall 20s→437s (7.3 min) over 31 epochs and still climbing; wave tile 101→2898. Mechanism: V=512 makes generation outpace the trainer → games accumulate during the lengthening SGD phase → bigger wave tile → sgd_per_position scales SGD with the growing inflow → positive-feedback divergence (V=64/WL5 stays bounded). **The cold-window R-TRAIN metric (120s ≈ 8 epochs) measured the pre-buffer-fill transient and missed the runaway entirely.** Jason stopped it ("3 min/epoch? forget that") and asked to file it for the lab to explore. **The big reframe: the lab optimized aug/s (generation), but the real objective is wall-clock-to-elo (training) — and maxing generation FLOODS the trainer.** Detailed writeup: [perf-bench-vs-real-training-cost.md](../topics/perf-bench-vs-real-training-cost.md). **New TOP direction: the LF1-followups lane block (Tier 1, top)** — 6 lanes; load-bearing are #1 warm-buffer R-TRAIN metric (the current metric is broken), #2 runaway stability boundary (V sweep), #3 sgd_per_position vs cap. The ANE-for-self-play + cross-engine-coupling strands are both RESOLVED (prior RESUME STATE). LF1 stopped + cleaned; monitor cron d443ef9c deleted. CHARTER FLAG: the R-TRAIN-* metric definition (charter §Success metric) needs the warm-buffer fix — it currently measures a cold-window transient.
- **RESUME STATE (2026-05-23, BOTH big strands resolved — ANE-for-self-play + cross-engine coupling; queue down to lower-value lanes)** [SUPERSEDED — LF1 opened the metric-validity direction above]: Two strands closed on clean evidence this session-resume. **(A) ANE-for-self-play — CLOSED:** residency achievable (the `ct.RangeDim` bug → CPU/BNNS; static fixed-batch export restores it, tiny+small), but no throughput win (tiny + small both lose to torch; small ≈ CPU/BNNS) and no contention-immunity (v2: ANE workers −35% under GPU load). Capability preserved on branch feat/perf-L09i-fix (opt-in `--coreml-static-batch`, unmerged) for GPU-idle use. **(B) Cross-engine coupling mechanism — PINNED:** Lpwr (CPU −82%) + v2 (ANE −35%, bidirectional) + Lpwr2 (throttle ∝ matrix size) + Lpwr2b (FLOP-rate-INDEPENDENT: 3.5× FLOPs = ~0 extra throttle) ⇒ the throttle tracks GPU **working-set/occupancy, not compute-power**. Actionable: shrink GPU footprint/occupancy, not FLOPs, to cut contention. **Remaining queue is lower-value:** Lpwr2c (exact channel: memory-bound vs compute-bound hog — optional, mechanism already pinned enough), L09-reopen-medium (does medium stay ANE-resident? genuine unknown but envelope-curiosity), Lhot2 (engine-specific heat-soak nuance, 2.5), L09i-fix-c (moot for production). No HALT (lanes remain) but we've hit clear **diminishing returns** — the two load-bearing questions are answered. A higher-value pivot (e.g. the actual training run, 15×15 prep, or a fresh Tier-1 architectural idea) would beat grinding the curiosity lanes; flag to Jason while continuing with reopen-medium (the one genuine remaining unknown). consecutive_rejects: not at HALT; all results mechanism-rich. Loop continues per the keep-going directive.
- **RESUME STATE (2026-05-23, ANE-for-self-play strand CLOSED on clean evidence; next = Lpwr2 mechanism pin)** [SUPERSEDED — Lpwr2/Lpwr2b now done]: The full ANE arc resolved this session-resume. **(1) Residency** — achievable: the symbolic `ct.RangeDim` batch dim (coreml_evaluator.py:267) demoted everything to CPU/BNNS; a static fixed-batch export (L09i-fix) restores `sample`-confirmed ANE residency at tiny AND small. **(2) Throughput** — reject at every size: tiny ANE (live) 7,698 < torch 8,039; small ANE (clean, batch 96) 1,834 ≈ CPU/BNNS 1,930 but −44% vs torch 3,297. Real ANE residency does NOT beat torch — it roughly ties the CPU/BNNS path. **(3) Contention** — no immunity: L09i-fix-load-v2 (clean pure-self-play A/B) showed ANE workers throttle −35% under a GPU hog, AND the 16 busy workers throttle the hog back to ~2.7 TFLOP/s — bidirectional package-power coupling, not a free side-channel. **Net: the ANE offers no clean self-play win (no throughput edge, no contention-immunity).** Residency stays a documented CAPABILITY (branch feat/perf-L09i-fix, opt-in `--coreml-static-batch`, NOT merged) for GPU-IDLE use only (deployment / paced match-eval sidecar). Reviewer APPROVE on L09i-fix and L09i-fix-load-v2. **Next high-value lane: Lpwr2** (cold-chip cross-engine power-coupling mechanism pin — now TRIPLY motivated: CPU −82%, ANE −35%, bidirectional hog-suppression 11→2.7 TFLOP/s all point at a shared package budget; pin power-throttle vs scheduling vs mem-bandwidth). Lpwr2 needs a COOLED chip + interleaved A/B (NOT smoke-first); the chip is hot from this session's continuous cells+hogs, so run it cooldown-first. Optional leftovers: L09-reopen-medium (envelope curiosity; may hit ANE-unsupported ops), Lhot2 (2.5), L09i-fix-c (moot for production). consecutive_rejects: not at HALT — L09i RESOLVED, L09i-fix reject-throughput, fix-load needs_repeat→superseded, fix-load-v2 reject(clean), reopen-small needs_repeat→resolved. All mechanism-rich; queue still has Lpwr2 (high value). Loop continues per Jason's keep-going directive.
- **RESUME STATE (2026-05-23, ANE residency RESTORED + contention question REOPENED — strand NOT closed)** [SUPERSEDED by the entry above — strand now closed on clean v2 + reopen-small-b evidence]: Three lanes this session-resume, with one correction. **L09i** (diagnostic, RESOLVED): the lab's Core ML export ran on CPU/BNNS the whole time because of a symbolic `ct.RangeDim` batch dim — every prior L09* "ANE" result was mislabeled. **L09i-fix** (REJECT-throughput, mechanism win; Reviewer APPROVE): a static fixed-batch export restores genuine `sample`-confirmed ANE residency (first in the lab), but loses on throughput at tiny (7,698 < torch 8,039 < CPU/BNNS 10,762) due to the single-fixed-batch padding tax. **L09i-fix-load** (needs_repeat, INCONCLUSIVE; Reviewer REVISE → corrected): I initially filed it as "ANE workers collapse −96%, contention-immunity falsified" — **WRONG.** trainer.log shows worker gen held EXACTLY (gen=5.1s) under the GPU hog while the MPS TRAINER stalled (train 2.5→99.5s); wave-mode gates aug/s on the trainer loop, so the −96% is a trainer stall, not a worker collapse. **The ANE workers were NOT throttled — a contention-RESISTANCE signal, opposite the CPU/BNNS workers in Lpwr.** So the strand is REOPENED with a positive lean, not closed. **Top of queue: L09i-fix-load-v2 (priority 5.0)** — decoupled worker-contention test (pure self-play under a GPU load, no trainer-barrier confound; trainer-representative load). Then L09-ANE-resident-reopen (3.5), L09i-fix-c (3.0), Lpwr2 (3.0, cold-chip mechanism pin — still motivated by Lpwr's genuine CPU-worker collapse, but the "2nd ANE datapoint" justification is RETRACTED), Lhot2 (2.5). Worktree feat/perf-L09i-fix INTACT (the v2/c/reopen lanes use it). **Lesson filed:** in wave-mode, aug/s is trainer-gated — attribute a holistic collapse to the gen-vs-train phase from trainer.log before concluding. consecutive_rejects context: L09i RESOLVED (not a reject); L09i-fix REJECT-throughput; L09i-fix-load needs_repeat (not a reject). The productive ANE axis is OPEN, not exhausted. Next GPU lane (fix-load-v2) is runnable after chip cooldown.
- **RESUME STATE (2026-05-23, L09i resolved + ANE residency RESTORED — envelope re-opened)** [SUPERSEDED by the entry above]: The load-bearing L09i diagnostic resolved the entire ANE strand's central mystery: our Core ML export was ANE-hostile purely because `coreml_evaluator.export_model_to_coreml` declared a **symbolic `ct.RangeDim` batch dim** — the op graphs were byte-identical to the 2026-05-22 ANE-resident scout. **L09i-fix** swapped it for a single fixed static batch and **restored genuine ANE residency** (`sample`-confirmed twice, including under the live self-play worker: hot path on `AppleNeuralEngine`, 0 BNNS). This is the first real ANE residency in the lab — **every prior L09* "ANE" lane (small/tiny/medium) was CPU/BNNS.** Throughput at tiny/V=64 is a **reject** (L09i-fix-b best = 7,698 aug/s < torch 8,039 < CPU/BNNS L09c 10,762): the single-fixed-batch design pays a per-eval padding tax (1024→2,304 aug/s, 7× pad; 192→7,698 aug/s, 1.37× pad) and the ANE is slow at tiny anyway. BUT workers fully vacate the GPU → **best trainer_step (0.0172) + 18 epochs/window the lab has measured.** This is a mechanism-rich reject with concrete high-P follow-ups → **CONTINUE** (triage 2a). **Top of queue: L09i-fix-load (priority 6.0)** — ANE-resident workers under GPU saturation; do they resist the Lpwr -82% CPU-worker collapse? (potential architectural headline: the ANE's real value is contention-immunity under a heavy trainer, not raw tiny throughput). Then L09i-fix-c (4.5, tighter batch), L09-ANE-resident-reopen (4.0, re-map small/medium on real ANE), then Lpwr2/Lhot2. **Worktree feat/perf-L09i-fix is INTACT** (Reviewer to advise whether static-batch-default merges or stays opt-in). consecutive_rejects context: L09i is a RESOLVED diagnostic (not a reject); L09i-fix is a reject-on-throughput but a capability win that re-opens the envelope — the productive axis is wide open again, NOT exhausted.
- consecutive_rejects: **4** (L09d + L09c-V512 + L08 + L09e — four rejects with clean mechanism: ANE envelope snapshotted at today's stack (single-point at tiny+V=64); heap-ratio axis flat in current bandwidth-bound regime; routing axis null at today's Core ML). One short of 5-reject HALT threshold; **SESSION-END DECLARED 2026-05-23 by orchestrator** at a complete current-state snapshot (NOT a permanent ANE verdict) because the four rejects are envelope-mapping with clean mechanism (not knob-failure noise), the current-stack ANE envelope is fully mapped, and Jason flagged inbound new ANE research — natural pause point until the new ANE work lands and the priors reset. Session-end perf-log entry filed; friction-smoothing lessons batched to the gomoku-perf-lab skill. Prior streak: L06-followup-fp16 PROMOTE at R-S400 +97.2%; L06fu-extended × 3 PROMOTEs; L11b' established +152.9% R-TRAIN-LEAN-fp16 perf reference; L09b was blocked-not-reject; L09c PROMOTE (tiny+V=64 on ANE +33.9%); L09d REJECT (medium+V=512 ANE -59.6%); L09c-V512 REJECT (tiny+V=512 ANE -24.0%); L08 REJECT (heap-ratio null at R-S400/fp16); L09e REJECT (routing axis null, L09 reject is final).
- **Charter staleness flagged by 3 consecutive Reviewers (L10, L11, L09)**: `wiki/topics/perf-lab-charter.md:50` still reads "R-TRAIN-LEAN | same but V=128 (today's promoted gen default)" but L01 promoted V=512 as the R-S400 default, AND L11 has now rejected V=512 at the trainer level. Class B (charter modification) → needs user touch; out of any individual lane's scope. Recommend rewriting the R-TRAIN-LEAN row to reflect the current state (V=512 rejected as a target; V=64 stays the R-TRAIN-* default) on the next charter pass.
- queue empty + no followups pending: false (GPU queue has L10, L11, L09, L08-mps-heap-ratio, L05-followup, L06-followup queued)
- last halt reason: n/a — lab restarted 2026-05-23; four CPU-queue lanes landed in parallel
- **RESUME STATE (2026-05-23, post-hollance-absorption + L09e' + Lpwr)**: TWO new findings since the ANE-research reorg. **(1) L09e' RESOLVED the residency question: L09c runs on CPU/BNNS, NOT the ANE** (`sample` hot path `BnnsCpuInferenceOperation`; no `H11ANEServicesThread`; system GPU monitor ANE 0%; CPU_AND_GPU cross-check also CPU). The "tiny fits ANE design center" hypothesis is falsified — but the 2026-05-22 residency scout showed the model IS ANE-capable under a *fixed-fused* export, so our `coreml_evaluator` mlprogram export is what loses residency. **L09i (mlpackage op diff) bumped to priority 5.0 — it could re-open the entire ANE envelope if we find+fix the ANE-hostile op.** **(2) Lpwr: GPU saturation collapses CPU-resident workers −82%** (clean cool-chip A/B 10,431→1,905; trainer only +14%). Engines share a package resource (power/thermal). **L09c's win is load-fragile** — depends on GPU power headroom a heavy production trainer would consume. Mechanism not cleanly pinned (sweep thermally confounded); **Lpwr2 (cold-chip interleaved-A/B re-run) queued at priority 3.0.** New tool: `scripts/gpu_load_generator.py`. New topic page: `m5-max-cross-engine-coupling.md`. New memory: [[project-light-all-engines]]. **Top of queue: L09i (priority 5.0, CPU-queue, could re-open ANE).** Then Lpwr2 (3.0), L09c-ALL (3.5 — but largely subsumed by the CPU-routing finding), L09f/L09g (downweighted). consecutive_rejects context: L09e' is a residency-resolution (not a perf reject); Lpwr is needs_repeat. Both are research wins, not knob failures. ALL prior receipts intact.
- **RESUME STATE (session-end 2026-05-23 mid-session-resume, post-ANE-research-reorg)** [SUPERSEDED by the entry above; kept for archaeology]: L09e closed REJECT (routing axis null at today's Core ML; ALL marginal winner at +3.1% over CPU_AND_NE but still ~40% below R-TRAIN-WL5). The L09 reject stands **at the current Core ML + evaluator pipeline + model-arch family** — not a permanent verdict. **Session-end declared at a complete current-state ANE snapshot.** Five lanes ran this session-portion: L09c PROMOTE (the headline: R-TRAIN-TINY-ANE +33.9%) + L09d/L09c-V512/L08/L09e all REJECT (envelope-mapping with clean mechanism). consecutive_rejects: 4 (one short of 5-reject HALT threshold). **Current engine-isolation envelope snapshot at the `coreml-isolated` cap: Core ML at CPU_AND_NE wins at tiny+V=64 ONLY at today's stack** (5 measured comparison points across the ANE-axis confirm; V-axis amortization currently doesn't compound at tiny, model-size amortization currently doesn't extend to small/medium, routing axis null, heap-ratio axis null). **The "ANE" in R-TRAIN-TINY-ANE is the routing label, NOT a residency claim** — L09e' is queued to elevate the cap via `powermetrics ane_power`, currently blocked on sudo. **Future-shape:** Jason flagged inbound new ANE research; when it lands (new Core ML version, new ANE features, evaluator-pipeline updates), re-run L09c/L09d/L09c-V512/L09e against the new baseline; see [coreml-design-envelope-and-our-fit.md § Inbound research landing zone](../topics/coreml-design-envelope-and-our-fit.md#inbound-research-landing-zone) for the absorb-new-findings protocol. The single-point envelope is the right reading of today's stack, not a structural ANE limit. L11b' R-TRAIN-LEAN-fp16 +152.9% remains the perf cycle's headline R-TRAIN finding from the prior session-portion. **Next-session queue:** **L09e'** (residency proof via powermetrics, Tier 1, priority 5.0, blocked on sudo); L09f / L09g (V-axis and model-size sweeps at coreml — currently downweighted, reactivatable when inbound ANE research lands); L09h (.mlpackage re-export cost diagnostic, 1 cell). **THREE NEW friction-smoothing lessons + 1 ANE-snapshot framing lesson** filed to the gomoku-perf-lab skill at session-end. L08 + L09e both Reviewer APPROVE. **ANE research reorg completed (post-session-end pass): three wiki pages reorganized for canonical hierarchy** — [coreml-design-envelope-and-our-fit.md](../topics/coreml-design-envelope-and-our-fit.md) is now the entry point (current state + research lanes with status); [coreml-ane-residency-lab.md](../topics/coreml-ane-residency-lab.md) refocused on cap discipline + cap-status table for L09* receipts; [ane-int8-inference.md](../topics/ane-int8-inference.md) marked HISTORICAL with what-shipped audit. Charter R-TRAIN-* table refreshed with new R-TRAIN-TINY*/MEDIUM* refs + cap-system cross-ref note — long-standing Class-B charter-staleness flag CLEARED.
- **RESUME STATE (session-end 2026-05-23, pre-L09c)** [SUPERSEDED — kept for archaeology]: Headline wins from this session: **R-S400 nearly doubles** (4,765 → 9,398.5 aug/s via fp16-eval); **R-TRAIN family doubles** (3,297.6 → 8,340.5 aug/s via V=512 + low-sgd + fp16, perf ref only — TQ canary required for production); R-S200 / R-S100 / R-S400-tiny / R-S400-medium all updated to fp16. Lab worked exactly as designed: 12 lanes, every promote Reviewer-APPROVE, every behavior-borderline lane TQ-gated correctly. Next-session queue: **L09c** (tiny model on Core ML / ANE — smaller per-eval graph might amortize ANE pipeline overhead better), **L06fu-medium-AB** (clean medium V=512 fp32 vs fp16 attribution), **L08-mps-heap-ratio at the new fp16 reference**, **L11b''** (sgd_per_position sweep at V=512+fp16 for the optimal trainer-work band). Class-B housekeeping pending **user touch**: `wiki/topics/perf-lab-charter.md:50` R-TRAIN-LEAN row reads "V=128 (today's promoted gen default)" — stale on 2 axes after L11b's reject and L11b''s perf-ref promotion. 5 Reviewer flags accumulated; ready for user rewrite.

## Dispatch rule (charter v3)

The orchestrator pulls from both queues simultaneously:

1. **CPU queue**: spawn N Agents in worktrees for the top-N CPU lanes
   that aren't already running. No serial constraint. Integrate via
   merge-commit as they return.
2. **GPU queue**: if no GPU lane is active, pull the top GPU lane
   that isn't blocked on a CPU lane. Dispatch to canonical_sweep
   (60-90s/cell default). Wait for completion; file receipt; spawn
   Reviewer.
3. Code work surfaced mid-tick goes to the CPU queue, not the GPU
   queue. Don't serialize code behind cells.

A cron tick is a degenerate orchestrator — it only advances the GPU
queue. Live-conversation orchestration (you + me with Agent fan-out)
is the real shape; cron is just for unattended drift.
