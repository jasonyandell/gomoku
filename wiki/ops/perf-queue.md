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
| **R-S400** | small / W=8 / G=8 / S=400 / **V=512** | **4,765 aug/s** (L01) | **+49.5%** |
| **R-S200** | small / W=8 / G=8 / S=200 / **V=512** | **9,156 aug/s** (L03) | **+52.5%** |
| **R-S100** | small / W=8 / G=8 / S=100 / **V=512** | **15,082 aug/s** (L03) | **+35.2%** |
| **R-S400-tiny** | tiny / W=16 / G=8 / S=400 / **V=512** | **22,088 aug/s** (L07) | **+201.5% vs tiny V=64=7,326** |
| **R-TRAIN-WL5** | full WL5 recipe | TBD (L10) | — |
| **R-TRAIN-LEAN** | WL5 with V=512 | TBD (L11) | — |
| **R-TRAIN-ANE** | WL5 with workers on Core ML | TBD (L09) | — |

## CPU queue (parallel — Agent fan-out, no GPU contention)

These run as Agent subagents in worktrees; integrate as merge commits.
Multiple can be in flight at once. Listed top-down by priority.

### L12-write-lab-train-cell-driver (priority: gating)

```yaml
id: L12-write-lab-train-cell-driver
class: A (scripts/, no external effects)
unblocks: L09 (R-TRAIN-ANE), L10 (R-TRAIN-WL5 baseline), L11 (R-TRAIN-LEAN)
patch: |
  scripts/lab_train_cell.py: subprocess.Popen(gomoku.train) + N
  selfplay_worker children; --warmup-secs 30 + --measurement-secs 60-120;
  SIGTERM all; parse trainer.log for `^epoch (\d+)/` lines; compute
  epochs/sec, games/sec, trainer_step_s_p50. Write summary.tsv row
  matching canonical_sweep schema + epochs_per_sec column.
estimate: ~100 LOC, ~20-40 min Opus-time
notes: Gating — three GPU-queue Tier-1 lanes are blocked on it. Highest priority CPU lane.
```

### L05-torch-compile-mps (priority: med)

```yaml
id: L05-torch-compile-mps
class: A (worktree on feat/perf-L05-compile)
patch: --compile flag pass-through in canonical_sweep to selfplay_worker
estimate: ~30 min Opus-time (mostly already exists in selfplay_worker; just wire and smoke)
followup_cells: small W=8 G=8 V=512 (--compile) vs (no compile); R-S100-tiny too
notes: Cheap code; if cells show a win, compounds with everything.
```

### L06-fp16-eval (priority: med)

```yaml
id: L06-fp16-eval
class: A (worktree on feat/perf-L06-fp16)
patch: --fp16-eval flag passing fp16=True into make_torch_evaluator
estimate: ~20 min Opus-time
followup_cells: small W=8 G=8 V=512 (--fp16) vs (fp32); 2-cell smoke
notes: Cheap. Historic regression; worth re-checking with mature MPS + fused conv+bn.
```

### L08-driver-per-cell-envvars (priority: low)

```yaml
id: L08-driver-per-cell-envvars
class: A (canonical_sweep.py edit)
patch: extend cells.csv schema with optional `env` column; driver applies env vars in the Popen call
estimate: ~15 min Opus-time
unblocks: L08-mps-heap-ratio (GPU lane) and any future env-var experiments
notes: Strictly an infra enabler. After this lands, L08 becomes a normal GPU-queue lane.
```

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
status: blocked on CPU-queue L08-driver
```

### Background — Calibration / reference


## Completed

| date | id | resolution | best cell from lane | reviewer | notes |
|---|---|---|---|---|---|
| 2026-05-23 | L14-tiny-G-at-W16-V512 | reject | best = tiny W=16 G=8 V=512 = 22,088 (unchanged). G=4=22,261; G=16=22,164; G=32=22,076. 0.83% total spread — G axis flat. | APPROVE | Knob-tuning exhausted at chip envelope. Remaining lanes need code work. consecutive_rejects: 1→2. |
| 2026-05-23 | L13-tiny-W-peak-probe | reject | best = tiny W=16 V=512 = 22,088 (unchanged). W=12=20,560 (-6.9%); W=20=21,553 (-2.4%); W=24=20,970 (-5.1%). Smooth bump W∈[12,20] within 7% of peak. | APPROVE | Tiny W tolerance is wider than small's sharper drop — more headroom for L09 ANE tuning. consecutive_rejects: 0→1. |
| 2026-05-23 | L07-tiny-contour | promote | R-S400-tiny: tiny W=16 G=8 V=512 = 22,088 aug/s (+201.5% vs V=64=7,326). | APPROVE | Model-dependent W peak at V=512 — tiny W=16 BEATS W=8 (opposite of small). consecutive_rejects: 2→0. Auto-queued L13 (W peak probe) + L14 (tiny G axis). |
| 2026-05-23 | L04-G-x-wave | reject | best = W=8 G=8 V=512 = 4,765 (unchanged). G=4=4,608; G=16=4,541; G=32=4,514. G mildly non-monotone at V=512 (flat at V=64) but peak still G=8. | APPROVE | Compound finding with L02: at V=512 BOTH W and G axes peak at the canonical defaults. consecutive_rejects: 1→2. |
| 2026-05-23 | L02-W-x-wave-compound | reject | best = W=8 V=512 = 4,765 (unchanged). W=4=4,367; W=12=4,501; W=16=4,504 — wave saturation moved MPS-dispatch peak from W=16 to W=8 at V=512. | APPROVE | New finding: knob wins interact non-monotonically at chip envelope. consecutive_rejects: 0→1. |
| 2026-05-23 | L03-sims-x-wave | promote (2x) | R-S200: V=512 = 9,156 aug/s (+52.5%); R-S100: V=512 = 15,082 aug/s (+35.2%) | APPROVE | V=512 carries cleanly to S=200 and S=100. Receipt: 2026-05-23 L03 entry in experiment-ledger.md. |
| 2026-05-23 | L01-wave-extrapolation | promote | small W8 G8 S400 V=512 = 4,765 aug/s | APPROVE | +17.7% over V=128; +49.5% cumulative; plateau at V=512 (V=768/1024 flat). Receipt: 2026-05-23 entry in experiment-ledger.md. |
| 2026-05-23 | L00-canonical-sweep | promote | small W8 G8 S400 V=128 = 4,048 aug/s | (pre-reviewer-era; auto-grandfathered) | The kickoff sweep; receipt under canonical-sweep-mainframe lane. |

## Stop-condition tracker

- consecutive_rejects: **2** (L13 + L14; will reset on next promote)
- queue empty + no followups pending: false (CPU queue has L12/L05/L06/L08-driver; GPU queue is paused awaiting those)
- last halt reason: n/a — cron cancelled by user 2026-05-23; lab will resume on charter v3 model when restarted
- **RESUME STATE**: charter v3 landed. Next session: orchestrator fans out CPU-queue lanes (L12 highest), GPU queue restarts once L12 unblocks R-TRAIN-*. Default cell time is now 60-90s smoke-first.

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
