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

| ref | cell | best | speedup vs WL5 V=64 |
|---|---|---|---|
| **R-S400** | small / W=8 / G=8 / S=400 / **V=512** | **4,765 aug/s** (L01) | **+49.5%** |
| **R-S200** | small / W=8 / G=8 / S=200 / **V=512** | **9,156 aug/s** (L03, reviewer APPROVE) | **+52.5%** |
| **R-S100** | small / W=8 / G=8 / S=100 / **V=512** | **15,082 aug/s** (L03, reviewer APPROVE) | **+35.2%** |
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
status: blocked-on-driver (needs scripts/lab_train_cell.py; see task #24)
notes: The "holistic" lever. Scaffold merged. Measurement cells need the live-training cell driver. The loop should skip this until L10 / L11 / L09 driver lands.
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
status: blocked-on-driver (needs scripts/lab_train_cell.py; see task #24)
notes: Redesigned from "pure trainer" to "full end-to-end" because gomoku.train doesn't have a no-workers mode without invasive changes. The number we actually care about (R-TRAIN-WL5) IS the end-to-end production cell.
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
status: blocked-on-driver (needs scripts/lab_train_cell.py; see task #24)
notes: Rescoped from V=128 to V=512 after L01 promoted V=512 as the R-S400 default. First end-to-end validation that today's wave promote compounds at the trainer level. If it doesn't, R-S* metrics need humility — gen throughput isn't the whole story.
```

#### L12-write-lab-train-cell-driver (NEW, prerequisite)

```yaml
id: L12-write-lab-train-cell-driver
tier: 1
hypothesis: Writing the live-training cell driver unblocks all R-TRAIN-* perf cells (L09, L10, L11). The driver is a code-only task (no GPU); estimated <100 LOC.
references_affected: enables R-TRAIN-* family
code_change: true (worktree at feat/perf-L12-train-cell-driver recommended)
patch: |
  scripts/lab_train_cell.py: subprocess.Popen(gomoku.train) + N selfplay_worker children; --warmup-secs 30 + --measurement-secs 240; SIGTERM all; parse trainer.log for epoch lines (regex "^epoch (\d+)/"); compute (last - first) / measurement_secs = epochs/sec. Count game*.pt in records dir for games/sec. Write summary.tsv row matching the canonical_sweep schema with extra columns epochs_per_sec, trainer_step_s_p50.
prep_cells: none
measurement_cells: none (this lane lands a driver; doesn't run cells)
n_cells: 0
wall_cost_min: 0 (code only; no GPU)
E_delta_epochs_per_sec: gates L09/L10/L11 (multiplier ~1)
P_success: 0.8
priority: 100 (gating)
status: queued
notes: Highest priority in Tier 1 because it unblocks three other Tier 1 lanes. Pure code work — can be done parallel to any Tier 2 lane that's running.
```

### Tier 2 — Compound knob wins

### Tier 3 — Speculative knob lanes

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

#### L08-mps-heap-ratio (rescoped 2026-05-23 after L02/L04 — V=512 not V=128)

```yaml
id: L08-mps-heap-ratio
tier: 3
hypothesis: PYTORCH_MPS_HIGH_WATERMARK_RATIO at default may cap throughput; nondefault could help.
reference: R-S400 (now W=8 G=8 V=512 = 4,765)
code_change: true (canonical_sweep needs per-cell env var support; add to L12 driver scope or carve out an L08-driver task)
cells:
  - small W=8 G=8 S=400 V=512 PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
  - small W=8 G=8 S=400 V=512 PYTORCH_MPS_HIGH_WATERMARK_RATIO=1.4
  - small W=8 G=8 S=400 V=512 PYTORCH_MPS_HIGH_WATERMARK_RATIO=2.0
n_cells: 3
wall_cost_min: 17
E_delta_aug_per_sec: 150
P_success: 0.3
priority: 2.6
status: blocked-on-driver (canonical_sweep needs per-cell env var support; see notes)
notes: Rescoped from V=128 to V=512. Blocked-on-driver because cells.csv schema doesn't carry env vars and PYTORCH_MPS_HIGH_WATERMARK_RATIO must be set before python starts. Workaround: run as 3 separate canonical_sweep invocations with `PYTORCH_MPS_HIGH_WATERMARK_RATIO=X python scripts/canonical_sweep.py ...` — but that's a manual orchestration task. Add per-cell env var column to cells.csv as part of L12 driver work.
```

### Background — Calibration / reference

#### L13-tiny-W-peak-probe (auto-queued 2026-05-23 after L07 promote)

```yaml
id: L13-tiny-W-peak-probe
tier: bg
hypothesis: L07 showed tiny W=16 V=512 beats tiny W=8 V=512 (+29.4%). Probe finer around W=16 to find the actual peak — W ∈ {12, 16, 20, 24} all at tiny G=8 S=400 V=512.
reference: R-S400-tiny (current best = tiny W=16 G=8 S=400 V=512 = 22,088 aug/s after L07)
code_change: false
cells:
  - tiny W=12 G=8 S=400 V=512
  - tiny W=20 G=8 S=400 V=512
  - tiny W=24 G=8 S=400 V=512
n_cells: 3 (W=16 already measured at 22,088)
wall_cost_min: 17
E_delta_aug_per_sec: 2000
P_success: 0.5
priority: 58.8 (highest unblocked after L07 promote; +29% jump at W=16 suggests further compounds possible)
status: queued
notes: New auto-queued follow-up to L07. If W=20 or W=24 beats W=16, the W-axis "saturation point" at tiny is even higher than expected, which is important for L09 ANE worker-count tuning.
```

#### L14-tiny-G-at-W16-V512 (auto-queued 2026-05-23 after L07)

```yaml
id: L14-tiny-G-at-W16-V512
tier: bg
hypothesis: L07 promote was at G=8. G axis at tiny W=16 V=512 might also be non-monotone with a different peak than G=8 (recall L04 found G axis is mildly non-monotone at V=512 even on small).
reference: R-S400-tiny (W=16 V=512 G=8 = 22,088)
code_change: false
cells:
  - tiny W=16 G=4  S=400 V=512
  - tiny W=16 G=16 S=400 V=512
  - tiny W=16 G=32 S=400 V=512
n_cells: 3
wall_cost_min: 17
E_delta_aug_per_sec: 800
P_success: 0.35
priority: 16.5
status: queued
notes: Lower priority than L13 (peak-probe is more targeted) but cheap.
```

## Completed

| date | id | resolution | best cell from lane | reviewer | notes |
|---|---|---|---|---|---|
| 2026-05-23 | L07-tiny-contour | promote | R-S400-tiny: tiny W=16 G=8 V=512 = 22,088 aug/s (+201.5% vs V=64=7,326). | APPROVE | Model-dependent W peak at V=512 — tiny W=16 BEATS W=8 (opposite of small). consecutive_rejects: 2→0. Auto-queued L13 (W peak probe) + L14 (tiny G axis). |
| 2026-05-23 | L04-G-x-wave | reject | best = W=8 G=8 V=512 = 4,765 (unchanged). G=4=4,608; G=16=4,541; G=32=4,514. G mildly non-monotone at V=512 (flat at V=64) but peak still G=8. | APPROVE | Compound finding with L02: at V=512 BOTH W and G axes peak at the canonical defaults. consecutive_rejects: 1→2. |
| 2026-05-23 | L02-W-x-wave-compound | reject | best = W=8 V=512 = 4,765 (unchanged). W=4=4,367; W=12=4,501; W=16=4,504 — wave saturation moved MPS-dispatch peak from W=16 to W=8 at V=512. | APPROVE | New finding: knob wins interact non-monotonically at chip envelope. consecutive_rejects: 0→1. |
| 2026-05-23 | L03-sims-x-wave | promote (2x) | R-S200: V=512 = 9,156 aug/s (+52.5%); R-S100: V=512 = 15,082 aug/s (+35.2%) | APPROVE | V=512 carries cleanly to S=200 and S=100. Receipt: 2026-05-23 L03 entry in experiment-ledger.md. |
| 2026-05-23 | L01-wave-extrapolation | promote | small W8 G8 S400 V=512 = 4,765 aug/s | APPROVE | +17.7% over V=128; +49.5% cumulative; plateau at V=512 (V=768/1024 flat). Receipt: 2026-05-23 entry in experiment-ledger.md. |
| 2026-05-23 | L00-canonical-sweep | promote | small W8 G8 S400 V=128 = 4,048 aug/s | (pre-reviewer-era; auto-grandfathered) | The kickoff sweep; receipt under canonical-sweep-mainframe lane. |

## Stop-condition tracker

- consecutive_rejects: **0** (L07 promote reset the counter)
- queue empty + no followups pending: false (L13 + L14 auto-queued post-L07; L05/L06/L08 blocked-on-driver-equivalents; L12 needs human session)
- last halt reason: n/a (loop running; cron ce6fb88e scheduled `7,17,27,37,47,57 * * * *`)

## Loop dispatch rule under "blocked-on-driver"

Tier-1 lanes with `status: blocked-on-driver` are skipped by the dispatch
heuristic — they don't count as "Tier-1 ready". The loop falls through to:
1. L12 if not yet started (it unblocks L09/L10/L11 — pure code work).
2. Tier 2 (L03 highest priority within tier at 16.7 after L01 rescope).
3. Tier 3 / bg only if everything above is in flight or blocked.

This avoids the loop stalling because Tier-1 needs human-curated code
work it can't do in a 10-min tick.
