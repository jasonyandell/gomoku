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
| **R-TRAIN-TINY-ANE** (new ref; envelope-mapping, NOT R-TRAIN-WL5 substitute) | tiny / W=16 / G=8 / S=400 / V=64 / coreml CPU_AND_NE | **10,762.6 aug/s** / 0.0417 ep/s / 49.43 g/s (L09c) | **+33.9% vs R-TRAIN-TINY torch** — engine envelope maps: ANE pays at tiny only |
| **R-TRAIN-TINY** (new ref; torch baseline arm for R-TRAIN-TINY-ANE) | tiny / W=16 / G=8 / S=400 / V=64 / torch | **8,039.1 aug/s** / 0.0333 ep/s / 32.48 g/s (L09c-baseline) | — (baseline) |
| **R-TRAIN-MEDIUM** (new ref; envelope-mapping; torch+fp16 baseline for medium engine-comparison) | medium / W=8 / G=8 / S=400 / V=512 / torch / fp16-eval | **1,463.3 aug/s** / 0.0042 ep/s / 5.66 g/s (L09d-baseline240) | — (baseline) |
| ~~R-TRAIN-LEAN~~ | WL5 with V=512 | **2,362.8 aug/s** / 0.0083 ep/s / 8.42 g/s (L11, REJECT — gen win doesn't compound at trainer) | — |
| ~~R-TRAIN-ANE~~ | WL5 with workers on Core ML | **1,930.3 aug/s** / 0.0583 ep/s / 8.00 g/s (L09, REJECT holistic at SMALL; the same engine wins at TINY — see R-TRAIN-TINY-ANE) | — |
| ~~R-TRAIN-MEDIUM-ANE~~ | medium V=512 with workers on Core ML | **591.7 aug/s** / 0.0208 ep/s / 2.33 g/s (L09d, REJECT — trainer side -81% trainer_step_s_p50, but worker gen 6× slower on ANE at medium V=512 = holistic -59.6%; envelope sharply mapped: ANE pays at TINY only) | — |

## CPU queue (parallel — Agent fan-out, no GPU contention)

These run as Agent subagents in worktrees; integrate as merge commits.
Multiple can be in flight at once. Listed top-down by priority.

*Empty.* All four pre-restart CPU lanes (L12, L05, L06, L08-driver)
landed on 2026-05-23 — see Completed table below. New code-lane
follow-ups (e.g. trainer-side `--compile` once trainer-compile is
desired) will be added here as they surface.

## GPU queue (serial — one cell at a time on MPS)

Lanes listed top-down by **tier**, then by priority within tier.

### Tier 1 — Architectural / holistic

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

### Tier 2 — Compound knob wins

### Tier 3 — Speculative knob lanes

(All Tier-3 lanes in this queue are now CPU-queue tasks that produce
GPU-queue cells once their patch lands; see CPU queue above. L05/L06
land code → become GPU cells. L08 unblocks when L08-driver lands.)

#### L08-mps-heap-ratio (post-L08-driver)

```yaml
id: L08-mps-heap-ratio
tier: 3
hypothesis: PYTORCH_MPS_HIGH_WATERMARK_RATIO at default may cap throughput; nondefault could help.
reference: R-S400 (now W=8 G=8 V=512 = 4,765)
cells (after L08-driver lands per-cell env var support):
  - small W=8 G=8 S=400 V=512 PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
  - small W=8 G=8 S=400 V=512 PYTORCH_MPS_HIGH_WATERMARK_RATIO=1.4
  - small W=8 G=8 S=400 V=512 PYTORCH_MPS_HIGH_WATERMARK_RATIO=2.0
n_cells: 3
wall_cost_min: 5 (60s/cell smoke-first per charter v3)
E_delta_aug_per_sec: 150
P_success: 0.3
priority: 2.6
status: queued (L08-driver landed 2026-05-23 — env column ready). Heads-up: the three cells collapse to one cell_id because env isn't in cell_id_of(); disambiguate via three separate out-dirs or a cell_id suffix when running.
```

#### L05-followup-compile-cells (post-L05 GPU follow-up)

```yaml
id: L05-followup-compile-cells
tier: 3
hypothesis: torch.compile on eval-only model improves aug/s without quality change.
reference: R-S400 (small W=8 G=8 V=512 = 4,765) and R-S400-tiny (W=16 V=512 = 22,088)
cells:
  - small W=8 G=8 S=400 V=512 --compile vs --no-compile (60s smoke)
  - tiny  W=16 G=8 S=400 V=512 --compile vs --no-compile (60s smoke)
n_cells: 4
wall_cost_min: 5
E_delta_aug_per_sec: 200
P_success: 0.25 (torch.compile + MPS is hit-or-miss)
priority: 2.5
status: queued (L05 driver flag landed 2026-05-23)
```

#### L06-followup-fp16-cells (post-L06 GPU follow-up)

```yaml
id: L06-followup-fp16-cells
tier: 3
hypothesis: fp16 eval reduces memory bandwidth and improves aug/s on MPS without behavior change.
reference: R-S400 (small W=8 G=8 V=512 = 4,765) and R-S400-tiny (W=16 V=512 = 22,088)
cells:
  - small W=8 G=8 S=400 V=512 --fp16-eval vs fp32 (60s smoke)
  - tiny  W=16 G=8 S=400 V=512 --fp16-eval vs fp32 (60s smoke)
n_cells: 4
wall_cost_min: 5
E_delta_aug_per_sec: 200
P_success: 0.3 (mature MPS + fused conv+bn may have closed the historic fp16 gap)
priority: 2.5
status: queued (L06 driver flag landed 2026-05-23)
```

### Background — Calibration / reference


#### L09e — Core ML compute-units routing sweep (bumped post-L09d — diagnostic value spiked)

```yaml
id: L09e
tier: 3
hypothesis: L09 used CPU_AND_NE. We don't know if our model is fully ANE-resident or if Core ML silently demoted some ops to CPU/GPU. L09d's medium-V=512 reject (-59.6%) could be "ANE is just slow at medium" OR "Core ML demoted ops to CPU/GPU silently at medium" — these are different conclusions with different implications. Sweep compute-units routings at the L09 reference shape (small/V=64) AND at L09d's medium V=512 shape; compare. If a non-CPU_AND_NE routing beats CPU_AND_NE at medium V=512, the L09d reject is "routing wrong", not "ANE just slow".
references_affected: diagnostic — disambiguates the L09 + L09d reject mechanism.
code_change: false (already supported by L12 driver)
measurement_cells:
  - small / W=8 / G=8 / S=400 / V=64 / coreml / CPU_AND_NE (re-measure L09 reference)
  - small / W=8 / G=8 / S=400 / V=64 / coreml / CPU_AND_GPU
  - small / W=8 / G=8 / S=400 / V=64 / coreml / ALL
  - small / W=8 / G=8 / S=400 / V=64 / coreml / CPU_ONLY
  - medium / W=8 / G=8 / S=400 / V=512 / coreml / CPU_AND_GPU (NEW post-L09d — diagnostic at the L09d-fail shape)
  - medium / W=8 / G=8 / S=400 / V=512 / coreml / ALL (NEW post-L09d)
n_cells: 6 (was 4; +2 medium-shape cells added post-L09d)
wall_cost_min: 16 (was 12; +6 for the 2 medium cells at 240s each)
E_delta_aug_per_sec: 300 (diagnostic value; could rescue L09d if routing is the issue)
P_success: 0.7 (informative regardless of outcome)
priority: 3.0 (was 1.5; bumped post-L09d — diagnostic value spiked because the L09d-reject mechanism is ambiguous between "ANE slow" and "ops demoted")
status: queued
notes: Pre-requisite for elevating any ANE receipt past the `coreml-scheduled` cap is wiring powermetrics ane_power into metadata (see coreml-ane-residency-lab.md "Definition Of Cap"). Without that, L09e is informative-but-not-residency-proving — but still diagnostically valuable for distinguishing the two L09d-reject readings.
```

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

- consecutive_rejects: **2** (L09d + L09c-V512 — both envelope-mapping rejects with clean mechanism; ANE win is now a single-point envelope at tiny+V=64). Far below the 5-reject charter threshold; lab continues per autonomous-loop charter. Prior streak: L06-followup-fp16 PROMOTE at R-S400 +97.2%; L06fu-extended × 3 PROMOTEs; L11b' established +152.9% R-TRAIN-LEAN-fp16 perf reference; L09b was blocked-not-reject; L09c PROMOTE (tiny + V=64 on ANE +33.9%); L09d REJECT (medium + V=512 on ANE -59.6%); L09c-V512 REJECT (tiny + V=512 on ANE -24.0%, V-axis amortization falsified at tiny).
- **Charter staleness flagged by 3 consecutive Reviewers (L10, L11, L09)**: `wiki/topics/perf-lab-charter.md:50` still reads "R-TRAIN-LEAN | same but V=128 (today's promoted gen default)" but L01 promoted V=512 as the R-S400 default, AND L11 has now rejected V=512 at the trainer level. Class B (charter modification) → needs user touch; out of any individual lane's scope. Recommend rewriting the R-TRAIN-LEAN row to reflect the current state (V=512 rejected as a target; V=64 stays the R-TRAIN-* default) on the next charter pass.
- queue empty + no followups pending: false (GPU queue has L10, L11, L09, L08-mps-heap-ratio, L05-followup, L06-followup queued)
- last halt reason: n/a — lab restarted 2026-05-23; four CPU-queue lanes landed in parallel
- **RESUME STATE (2026-05-23 mid-session, post-L09c-V512)**: L09c-V512 closed REJECT (-24.0% at tiny+V=512 vs torch+fp16). **Engine envelope: 4 measured comparison points; ANE wins at EXACTLY ONE point (tiny + V=64, L09c +33.9%); single-point envelope, not a region.** L09d "larger compute amortizes" FALSIFIED at medium; L09c-V512 "V-axis amortizes" FALSIFIED at tiny. L09f / L09g downweighted to "queue for completeness". **Top-of-queue for next dispatch:** the only remaining ANE-rescue diagnostic is **L09e** (compute-units routing sweep at small/V=64 and medium/V=512; Tier 3, priority 3.0, ~16 min wall, 6 cells). Behind L09e: **L08-mps-heap-ratio** (Tier 3, priority 2.6, 3 cells, ~5 min — MPS-side knob tuning, now next-best lever with ANE-axis nearly exhausted). L09c-V512 Reviewer pending. consecutive_rejects: 2 (still far below 5-reject charter threshold). NEW friction-smoothing lesson from this session: plies_mean is NOT stationary across asymmetric-epoch R-TRAIN cells; future Reviewers' drift-watch should check per-epoch trainer.log plies values when arms have very different epochs_in_window. Class-B housekeeping continues to accumulate (5 new R-TRAIN-* refs since session-restart) — pending **user touch** on `wiki/topics/perf-lab-charter.md:50`.
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
