# Perf Lab Queue

Live, ordered queue of perf experiments. The autonomous loop reads from
the **Active** section top-down within each tier; lanes finished or
rejected move to **Completed** with their resolution.

See [perf-lab-charter](../topics/perf-lab-charter.md) for the vision,
the tier system, and the priority function:
`priority = (E[delta] × P[succeeds]) / wall_cost_seconds`
**gated by tier** (Tier-1 always before Tier-2 before Tier-3).

Reviewer gates every promote: see
[perf-lab-reviewer-role](../topics/perf-lab-reviewer-role.md).

Reference points (current bests):

| ref | cell | best |
|---|---|---|
| **R-S400** | small / W=8 / G=8 / S=400 / V=128 | 4,048 aug/s |
| **R-S200** | small / W=8 / G=8 / S=200 / V=64  | 6,006 aug/s |
| **R-S100** | small / W=8 / G=8 / S=100 / V=64  | 11,151 aug/s |
| **R-TRAIN-WL5** | full WL5 recipe | TBD (first measure in L10) |
| **R-TRAIN-LEAN** | WL5 with V=128 | TBD (L11) |
| **R-TRAIN-ANE** | WL5 with workers on Core ML | TBD (L09) |

## Active

Lanes listed top-down by **tier**, then by priority within tier.

### Tier 1 — Architectural / holistic

#### L09-ane-offload-prototype

```yaml
id: L09-ane-offload-prototype
tier: 1
hypothesis: A Core ML eval-worker frees the GPU from inference; even with slower raw eval the concurrent trainer step rate increases and overall R-TRAIN-ANE beats R-TRAIN-WL5.
references_affected: R-TRAIN-ANE (new); R-S* under engine isolation
code_change: true
worktree: ~/code/gomoku-perf-L09-coreml
patch: |
  selfplay_worker: --evaluator {torch_mps, coreml} flag. coreml path uses gomoku.coreml_evaluator. Worker reports which engine it used in its log.
prep_cells:
  - smoke: 1 worker, tiny, S=100, V=32, coreml evaluator, 5 min — verify a worker produces records via Core ML end-to-end
measurement_cells:
  - R-TRAIN-ANE warmup: full train+8 workers (coreml), 5 min, no recording
  - R-TRAIN-ANE measure: full train+8 workers (coreml), 5 min, report epochs/sec + games/sec
n_cells: 3 (1 scaffold smoke + 2 stitched train cells)
wall_cost_min: 15 (+ ~30 min code scaffold, parallel-with-L01)
E_delta_epochs_per_sec: 0.4
P_success: 0.35
priority: 4.0
status: scaffolding
notes: The "holistic" lever Jason called out 2026-05-23. Even if Core ML eval is slower (it is — see ane-int8-inference receipts), freeing the GPU from worker eval is the win. The scaffold can run parallel to L01 since it's code work not GPU.
```

#### L10-trainer-step-bench

```yaml
id: L10-trainer-step-bench
tier: 1
hypothesis: First-ever R-TRAIN-WL5 measurement. Pure trainer (no self-play workers, replay-buffer warmed from archive), measure epochs/sec at the WL5 recipe.
references_affected: R-TRAIN-WL5 (new)
code_change: false (trainer already supports --no-eval and bounded epochs via timeout)
prep_cells:
  - R-TRAIN-WL5 warmup: trainer with batch=512 grad_accum=4 LR=1e-3, 30s warmup window, no recording
measurement_cells:
  - R-TRAIN-WL5 measure: same trainer, 4-min measurement window, report epochs/sec + step_s_p50
n_cells: 2 (warmup + measure)
wall_cost_min: 10
E_delta_epochs_per_sec: 0 (baseline, no comparison)
P_success: 1.0 (baseline measurement, can't fail)
priority: 10.0
status: queued
notes: Required as the R-TRAIN-WL5 reference before any R-TRAIN-* compound makes sense. Trainer feeds from archives/wl5_validation_v1.pt-backed warmed buffer to avoid the self-play dependency.
```

#### L11-end-to-end-cell

```yaml
id: L11-end-to-end-cell
tier: 1
hypothesis: The full production-shape end-to-end cell (trainer + 8 self-play workers + eval) at V=128 (today's promote) beats the V=64 baseline on epochs/sec + games/sec jointly.
references_affected: R-TRAIN-LEAN (new); R-TRAIN-WL5 (comparison)
code_change: false
depends_on: [L10]
prep_cells:
  - R-TRAIN-WL5 baseline warmup: full WL5 recipe (V=64), 30s no-record
  - R-TRAIN-WL5 baseline measure: 4 min, record epochs/sec + games/sec
  - R-TRAIN-LEAN warmup: same but V=128
  - R-TRAIN-LEAN measure: 4 min, record
n_cells: 4
wall_cost_min: 20
E_delta_epochs_per_sec: 0.15
P_success: 0.55
priority: 7.5
status: queued
notes: First end-to-end validation that the V=64 -> V=128 promote (from canonical sweep) actually improves the joint trainer+gen number, not just isolated self-play. If this fails to compound the win, R-S* metrics need a humility correction.
```

### Tier 2 — Compound knob wins

#### L02-W-x-wave-compound

```yaml
id: L02-W-x-wave-compound
tier: 2
hypothesis: V=128 promote compounds at higher worker counts.
reference: R-S400
code_change: false
cells:
  - small W=4  G=8 S=400 V=128
  - small W=12 G=8 S=400 V=128
  - small W=16 G=8 S=400 V=128
  - small W=4  G=8 S=400 V=256
  - small W=12 G=8 S=400 V=256
  - small W=16 G=8 S=400 V=256
n_cells: 6
wall_cost_min: 33
E_delta_aug_per_sec: 800
P_success: 0.6
priority: 2.4
status: queued
notes: Drop the redundant W=8 cells (already measured); just run the gaps.
```

#### L03-sims-x-wave

```yaml
id: L03-sims-x-wave
tier: 2
hypothesis: S=200 V=256 (or V=128) opens a new throughput regime faster than R-S200.
reference: R-S200 + R-S100
code_change: false
cells:
  - small W=8 G=8 S=100 V=128
  - small W=8 G=8 S=100 V=256
  - small W=8 G=8 S=200 V=128
  - small W=8 G=8 S=200 V=256
n_cells: 4
wall_cost_min: 22
E_delta_aug_per_sec: 1500
P_success: 0.55
priority: 5.6
status: queued
notes: Highest E[delta] in Tier 2; would open promoted-default at faster quality.
```

#### L04-G-x-wave

```yaml
id: L04-G-x-wave
tier: 2
hypothesis: G axis was flat at V=64; wider waves may unflatten it because each worker now needs more games to fill the eval batch.
reference: R-S400
code_change: false
cells:
  - small W=8 G=4  S=400 V=128
  - small W=8 G=16 S=400 V=128
  - small W=8 G=32 S=400 V=128
  - small W=8 G=4  S=400 V=256
  - small W=8 G=16 S=400 V=256
n_cells: 5
wall_cost_min: 28
E_delta_aug_per_sec: 400
P_success: 0.4
priority: 1.4
status: queued
notes: G=32 is novel territory.
```

### Tier 3 — Speculative knob lanes

#### L01-wave-extrapolation

```yaml
id: L01-wave-extrapolation
tier: 3
hypothesis: Wave gains continue past V=256; find the plateau.
reference: R-S400
code_change: false
cells:
  - small W=8 G=8 S=400 V=384
  - small W=8 G=8 S=400 V=512
  - small W=8 G=8 S=400 V=768
  - small W=8 G=8 S=400 V=1024
n_cells: 4
wall_cost_min: 22
E_delta_aug_per_sec: 600
P_success: 0.45
priority: 4.5
status: running (kicked 2026-05-23 ~01:56 UTC; ETA ~02:18 UTC)
notes: Was Tier-1 before the tier refactor; demoted to Tier-3 (single-axis speculation past today's win).
```

#### L05-torch-compile-mps

```yaml
id: L05-torch-compile-mps
tier: 3
hypothesis: torch.compile regressed under MPS historically; current torch + fused eval might be neutral or a win.
reference: R-S400 + R-S100
code_change: true
worktree: ~/code/gomoku-perf-L05-compile
patch: canonical_sweep --compile flag passes through to selfplay_worker.
cells:
  - small W=8 G=8 S=400 V=128 (--compile)
  - small W=8 G=8 S=100 V=64  (--compile)
n_cells: 2
wall_cost_min: 12
E_delta_aug_per_sec: 500
P_success: 0.3
priority: 12.5
status: queued
notes: High priority within Tier 3 because cheap.
```

#### L06-fp16-eval

```yaml
id: L06-fp16-eval
tier: 3
hypothesis: fp16 eval on MPS historic regression; may now be small win with mature fused conv+bn.
reference: R-S400
code_change: true
worktree: ~/code/gomoku-perf-L06-fp16
patch: --fp16-eval flag on selfplay_worker.
cells:
  - small W=8 G=8 S=400 V=128 (--fp16-eval)
n_cells: 1
wall_cost_min: 6
E_delta_aug_per_sec: 200
P_success: 0.25
priority: 8.3
status: queued
notes: Tiny.
```

#### L08-mps-heap-ratio

```yaml
id: L08-mps-heap-ratio
tier: 3
hypothesis: PYTORCH_MPS_HIGH_WATERMARK_RATIO at default (1.7) may cap throughput; nondefault could help.
reference: R-S400
code_change: false (env var only)
cells:
  - small W=8 G=8 S=400 V=128 PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
  - small W=8 G=8 S=400 V=128 PYTORCH_MPS_HIGH_WATERMARK_RATIO=1.4
  - small W=8 G=8 S=400 V=128 PYTORCH_MPS_HIGH_WATERMARK_RATIO=2.0
n_cells: 3
wall_cost_min: 17
E_delta_aug_per_sec: 150
P_success: 0.3
priority: 2.6
status: queued
notes: Almost no prior art for MPS heap tuning on M5 Max under AZ.
```

### Background — Calibration / reference

#### L07-tiny-contour

```yaml
id: L07-tiny-contour
tier: bg
hypothesis: Tiny model contour is the speed ceiling reference for ANE/engine-overlap planning.
reference: new R-S400-tiny
code_change: false
cells:
  - tiny W=8  G=8 S=400 V=128
  - tiny W=8  G=8 S=400 V=256
  - tiny W=16 G=8 S=400 V=128
  - tiny W=16 G=8 S=400 V=256
  - tiny W=12 G=8 S=400 V=128
  - tiny W=8  G=16 S=400 V=128
  - tiny W=8  G=16 S=400 V=256
n_cells: 7
wall_cost_min: 39
E_delta_aug_per_sec: 4000
P_success: 0.7
priority: 12.0
status: queued
notes: Calibrates the ANE work. Runs when nothing in Tier 1-3 needs GPU.
```

## Completed

| date | id | resolution | best cell from lane | reviewer | notes |
|---|---|---|---|---|---|
| 2026-05-23 | L00-canonical-sweep | promote | small W8 G8 S400 V=128 = 4,048 aug/s | (pre-reviewer-era; auto-grandfathered) | The kickoff sweep; receipt under canonical-sweep-mainframe lane. |

## Stop-condition tracker

- consecutive_rejects: 0
- queue empty + no followups pending: false
- last halt reason: n/a (loop has not yet started auto-dispatch)
