# Perf Lab Queue

Live, ordered queue of perf experiments. The autonomous loop reads from
the **Active** section top-down; lanes finished or rejected move to
**Completed** with their resolution. New compound follow-ups generated
by completed lanes get pushed back into Active at their computed
priority.

See [perf-lab-charter](../topics/perf-lab-charter.md) for the vision
and the priority function:
`priority = (E[delta_aug_pos] × P[lane_succeeds]) / wall_cost_seconds`.

Reference points (current bests):

| ref | cell | aug/s |
|---|---|---|
| R-S400 | small / W=8 / G=8 / S=400 / V=128 | 4,048 |
| R-S200 | small / W=8 / G=8 / S=200 / V=64  | 6,006 |
| R-S100 | small / W=8 / G=8 / S=100 / V=64  | 11,151 |

## Active

Lanes listed top-down by current priority. Edit in place when
rearranging; move to **Completed** with a one-line resolution when
done.

### L01-wave-extrapolation

```yaml
id: L01-wave-extrapolation
hypothesis: Wave gains continue past V=256; find the plateau or the inflection.
reference: R-S400
code_change: false
cells:
  - small / W=8 / G=8 / S=400 / V=384
  - small / W=8 / G=8 / S=400 / V=512
  - small / W=8 / G=8 / S=400 / V=768
  - small / W=8 / G=8 / S=400 / V=1024
n_cells: 4
wall_cost_min: 22
E_delta_aug_per_sec: 600
P_success: 0.55
priority: 15.0
status: queued
notes: Cheapest direct test of today's V=128 promotion. If V>=512 wins, auto-queue W×V_winner and S×V_winner.
```

### L02-W-x-wave-compound

```yaml
id: L02-W-x-wave-compound
hypothesis: The V=128 win compounds at higher worker counts (W=12, W=16).
reference: R-S400
code_change: false
depends_on: []
cells:
  - small / W=4  / G=8 / S=400 / V=128
  - small / W=8  / G=8 / S=400 / V=128   # already measured as 4,048; re-measure for parity
  - small / W=12 / G=8 / S=400 / V=128
  - small / W=16 / G=8 / S=400 / V=128
  - small / W=4  / G=8 / S=400 / V=256
  - small / W=8  / G=8 / S=400 / V=256   # already 4,409
  - small / W=12 / G=8 / S=400 / V=256
  - small / W=16 / G=8 / S=400 / V=256
n_cells: 8
wall_cost_min: 44
E_delta_aug_per_sec: 800
P_success: 0.6
priority: 11.4
status: queued
notes: Validates today's wave promotion across the workers axis.
```

### L03-sims-x-wave

```yaml
id: L03-sims-x-wave
hypothesis: S=200 V=256 (or V=128) might open a new throughput regime that's faster than R-S200 while still useful for quality.
reference: R-S200 (and possibly a new R-S200-V128)
code_change: false
cells:
  - small / W=8 / G=8 / S=100 / V=128
  - small / W=8 / G=8 / S=100 / V=256
  - small / W=8 / G=8 / S=200 / V=128
  - small / W=8 / G=8 / S=200 / V=256
  - small / W=8 / G=8 / S=400 / V=128   # already 4,048
  - small / W=8 / G=8 / S=400 / V=256   # already 4,409
n_cells: 6
wall_cost_min: 33
E_delta_aug_per_sec: 1500
P_success: 0.55
priority: 25.0
status: queued
notes: Opens promoting wave at quality points faster than R-S400. Has the highest E[delta] / cost in the day-1 queue.
```

### L04-G-x-wave

```yaml
id: L04-G-x-wave
hypothesis: G axis was flat at V=64, but wider waves may need more games per worker to fill the eval batch and unflatten G.
reference: R-S400
code_change: false
cells:
  - small / W=8 / G=4  / S=400 / V=128
  - small / W=8 / G=8  / S=400 / V=128   # already 4,048
  - small / W=8 / G=16 / S=400 / V=128
  - small / W=8 / G=32 / S=400 / V=128
  - small / W=8 / G=4  / S=400 / V=256
  - small / W=8 / G=16 / S=400 / V=256
n_cells: 6
wall_cost_min: 33
E_delta_aug_per_sec: 400
P_success: 0.4
priority: 4.5
status: queued
notes: Cheap follow-up; G=32 is new territory. If G=16/V=256 dominates, that's a new R-S400 candidate.
```

### L05-torch-compile-mps

```yaml
id: L05-torch-compile-mps
hypothesis: torch.compile historically regressed under MPS; with current torch + fused eval it may now be neutral or a win.
reference: R-S400 + R-S100 (compile pays back more on small kernels)
code_change: true
worktree: ~/code/gomoku-perf-L05-compile
patch: |
  scripts/canonical_sweep.py: add `--compile` flag passed through to selfplay_worker.
  gomoku/selfplay_worker.py: --compile already exists; just wire it through.
cells:
  - small / W=8 / G=8 / S=400 / V=128 (--compile)
  - small / W=8 / G=8 / S=100 / V=64  (--compile)
  - small / W=8 / G=8 / S=400 / V=128 (no compile, parity)   # already 4,048
  - small / W=8 / G=8 / S=100 / V=64  (no compile, parity)   # already 11,151
n_cells: 4
wall_cost_min: 22
E_delta_aug_per_sec: 500
P_success: 0.3
priority: 6.8
status: queued
notes: Kill-or-promote. If win, compounds with every cell.
```

### L06-fp16-eval

```yaml
id: L06-fp16-eval
hypothesis: fp16 eval on MPS has historically regressed; with mature MPS + fused conv+bn this may now be a small win.
reference: R-S400
code_change: true
worktree: ~/code/gomoku-perf-L06-fp16
patch: |
  gomoku/selfplay_worker: add --fp16-eval that calls make_torch_evaluator(..., fp16=True).
cells:
  - small / W=8 / G=8 / S=400 / V=128 (--fp16-eval)
  - small / W=8 / G=8 / S=400 / V=128 (no fp16, parity)   # already 4,048
n_cells: 2
wall_cost_min: 12
E_delta_aug_per_sec: 200
P_success: 0.25
priority: 4.2
status: queued
notes: Very cheap. Compounds with compile if both land.
```

### L07-tiny-contour

```yaml
id: L07-tiny-contour
hypothesis: Tiny model contour is needed as the speed ceiling reference for the ANE/engine-overlap planning work.
reference: new R-S400-tiny
code_change: false
cells:
  - tiny / W=8  / G=8 / S=400 / V=64    # already 7,326
  - tiny / W=8  / G=8 / S=400 / V=128
  - tiny / W=8  / G=8 / S=400 / V=256
  - tiny / W=16 / G=8 / S=400 / V=128
  - tiny / W=16 / G=8 / S=400 / V=256
  - tiny / W=12 / G=8 / S=400 / V=128
  - tiny / W=8  / G=16 / S=400 / V=128
  - tiny / W=8  / G=16 / S=400 / V=256
n_cells: 8
wall_cost_min: 44
E_delta_aug_per_sec: 4000
P_success: 0.7
priority: 63.6
status: queued
notes: Tiny was 7,326 at V=64; expected V=128/256 gains of 25-40% suggest 9,000-10,000+ aug/s. Calibrates the ANE work and adds a fast-mode default the trainer can pick if quality holds.
```

### L08-mps-heap-ratio

```yaml
id: L08-mps-heap-ratio
hypothesis: PYTORCH_MPS_HIGH_WATERMARK_RATIO at the default (1.7) may silently cap throughput under our worker count; lowering or raising it could help.
reference: R-S400
code_change: false (env var only)
cells:
  - small / W=8 / G=8 / S=400 / V=128 PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0   # disable cap
  - small / W=8 / G=8 / S=400 / V=128 PYTORCH_MPS_HIGH_WATERMARK_RATIO=1.4
  - small / W=8 / G=8 / S=400 / V=128 PYTORCH_MPS_HIGH_WATERMARK_RATIO=1.7   # default, already 4,048
  - small / W=8 / G=8 / S=400 / V=128 PYTORCH_MPS_HIGH_WATERMARK_RATIO=2.0
n_cells: 4
wall_cost_min: 22
E_delta_aug_per_sec: 150
P_success: 0.3
priority: 2.0
status: queued
notes: Almost no prior art for this knob under AZ workloads on M5 Max. If a non-default ratio wins, compounds with everything.
```

## Completed

(History of lanes that have run, with their resolution. Newest at top.)

| date | id | resolution | best cell from lane | notes |
|---|---|---|---|---|
| 2026-05-23 | L00-canonical-sweep | promote | small W8 G8 S400 V=**128** = 4,048 aug/s (+27%) | The kickoff sweep; receipt under canonical-sweep-mainframe lane. |

## Stop-condition tracker

- consecutive_rejects: 0
- queue empty + no followups pending: false
- last halt reason: n/a (loop has not yet started)
