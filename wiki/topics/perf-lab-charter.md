# Perf Lab Charter — "Fastest Generator in the West"

Captured 2026-05-23 at the launch of the autonomous perf era. Sets the
goal, the success metric, the operating loop, and the autonomy
boundaries for the post-canonical-sweep work. Any future session
operating the lab should read this page first.

## Mission

Push **measured aug-positions-per-second at fixed quality** as high as
the M5 Max will allow under realistic gomoku self-play conditions.
Treat the chip as the [[feedback-know-the-machine]] mainframe it is:
sweep it, calibrate it, find its corners, write the receipts. Stop
when the ceiling is structural (not when the loop is bored).

This is **perf work, not training work**. The lab does not run epochs;
it runs ≤ 5-minute production-shape self-play cells against
fresh-random fused checkpoints, measures throughput, and chains
follow-ups from the results.

## Success metric

Two metric families, each with reference points. A win at *either*
moves the lab toward the mission.

### R-S* — generator throughput (aug-positions per second)

Pure self-play under `scripts/canonical_sweep.py`. Fresh random fused
checkpoint, no trainer running. Default knobs:
`--secs-per-cell 300 --max-plies 16 --device mps`.

| ref point | cell shape | current best (2026-05-23) |
|---|---|---|
| **R-S400** | small / W=8 / G=8 / sims=400 / wave=128 | 4,048 aug/s |
| **R-S200** | small / W=8 / G=8 / sims=200 / wave=64 | 6,006 aug/s |
| **R-S100** | small / W=8 / G=8 / sims=100 / wave=64 | 11,151 aug/s |

### R-TRAIN-* — trainer + concurrent generator (the holistic metric)

Live training with self-play workers via `gomoku.train`. Bounded ≤ 5
min/cell: ~30s warmup, then a 60-90s measurement window, then SIGTERM.
Measures the END-TO-END throughput that matters for actual elo gain.
Per Jason's 2026-05-23 directive: training-side wins compound across
the entire run, so this family is **higher tier than knob lanes on
the generator side**.

| ref point | cell shape | current best |
|---|---|---|
| **R-TRAIN-WL5** | small / W=8 / G=8 / sims=400 / V=64 / EMA τ=0.99 / grad_accum=4 (WL5 production recipe) | TBD (first measurement in L10) |
| **R-TRAIN-LEAN** | same but V=128 (today's promoted gen default) | TBD (L11) |
| **R-TRAIN-ANE** | same but workers on Core ML eval | TBD (L09) |

Reported per cell: `epochs/sec`, `games/sec`, `aug_pos/sec`,
`trainer_step_s_p50`, `worker_wave_s_p50`. The product
`epochs/sec × steps_per_epoch` is the trainer's true throughput.

### Promotion rules

A promotion at any reference point requires a *no-behavior-change*
knob movement that improves the number. Lanes that change sims, model
size, or anything semantic are explorations, not promotions.

A secondary visible number per session: receipts filed, promotes,
rejects. Three consecutive rejects with no compound follow-up = stop
signal.

### Cell time ceiling

**Strict 5 min per cell.** For measurements that need a longer
warmup (trainer cells, torch.compile graph capture, ANE first-load),
**split into two back-to-back cells**: a warmup cell (no measurement
recorded; `cell_status: warmup`) + a measure cell. The driver runs
them under the same lane; the receipt aggregates.

Per Jason 2026-05-23: "we can tell perf after a warmup, some seconds.
but you may want more time" — multi-cell stitch honors the 5-min
boundary while letting any measurement breathe.

## Operating loop (autonomous)

```
while queue:
    lane = pick_top_unblocked(queue)
    if box_busy(): break             # never compete with other tenants
    if lane.needs_code_change:
        wt = create_worktree(feat/perf-<lane.id>)
        apply_lane_patch(wt)
        run_cells(wt, lane.cells)
        if win: git merge --no-ff feat/perf-<lane.id>   # always merge, never rebase
        else:   git worktree remove + branch -D
    else:
        run_cells(main, lane.cells)
    receipt = score(lane.results, baseline_at_quality_point)
    file_receipt(receipt)
    update_perf_log(receipt)
    update_best_cells(receipt)
    queue.append(generate_followups(receipt))
    queue.rerank(by=priority_function)
    if consecutive_rejects >= 3 and no_compound_followup:
        break
```

## Priority function and tier system

```
priority = (E[delta] × P[lane succeeds]) / wall_cost_seconds
```

But raw priority is *gated by tier*. A Tier-2 lane cannot leapfrog a
Tier-1 lane on score alone — Tier-1 always runs first when unblocked.
This is the explicit "architectural levers > knob tuning" rule from
Jason's 2026-05-23 directive.

### Tiers

| Tier | What lives here | Examples |
|---|---|---|
| **1 — Architectural / holistic** | Lanes that change which engine, runtime, or workload split the chip uses. Wins here compound across every other lane. | ANE-offload, trainer step bench (new R-TRAIN family), end-to-end production cell, custom Metal kernel, fp16/bf16 training, model parallelism. |
| **2 — Compound knob wins** | Lanes that cross two existing axes near a known win to verify it compounds. | W × wave, sims × wave, G × wave, model × wave. |
| **3 — Speculative knob lanes** | Single-axis lanes with low prior probability of moving the number, but cheap. | torch.compile, fp16 eval, heap ratio, exotic env vars. |
| **bg — Calibration / reference** | Lanes that exist to ground future work (e.g., tiny model contour as the ANE-comparison reference). Run when nothing else needs the GPU. | Tiny contour, medium contour. |

Within a tier, the priority function picks the next lane. Re-rank
after every completed lane. A just-landed Tier-1 win generates Tier-1
compound follow-ups, which immediately go to the top.

### Default seeds per lane

- `E[delta]` — gut estimate in the relevant unit (aug/s for R-S*,
  epochs/sec for R-TRAIN-*). Refined as similar lanes resolve.
- `P[lane succeeds]` — single-axis pivot near a recent win: high
  (0.5-0.8). Speculative cross-axis: 0.2-0.4. Architectural: 0.1-0.3
  but with very high delta when they hit.
- `wall_cost_seconds` — `n_cells × secs_per_cell + 30 × n_cells`
  (setup margin).

## Autonomy boundaries

| Autonomous | Manual confirm |
|---|---|
| Designing cells | Promoting a candidate into a live training run (Training-Quality Gate applies) |
| Running ≤ 30 cells per session unattended | Custom Metal kernels, native C extensions, model architecture changes |
| Live-training cells ≤ 5 min each (trainer + workers + eval) | Long training runs (>5 min, anything epoch-counted) |
| Worktree create / merge / remove on `feat/perf-*` branches | Anything that changes `pyproject.toml`, CI, or external deps |
| Scaffolding evaluator backends (Core ML, BNNS, CPU) behind existing CLI flags | Modifying wandb project, archives, or trained-model artifacts |
| `git merge --no-ff` integrations | `git rebase`, `git push --force`, `git reset --hard` — never |
| Promoting a new "best cell" at a quality point | Modifying the charter (this page) — surface it first |
| Filing receipts, baseline rows, perf-log entries | Stopping the buffer-curation / external-engines / ANE-rail lanes that another workstream owns |
| Opening, closing, reprioritizing lanes | |

### Reviewer gate

**No promote, no commit-touching-receipt without Reviewer
sign-off.** After every lane (and on a periodic discipline check),
spawn a Reviewer agent per
[perf-lab-reviewer-role](perf-lab-reviewer-role.md). The Reviewer
returns APPROVE / REVISE / BLOCK; BLOCK surfaces to the user. The
loop does not commit a `promote` decision until the Reviewer
approves.

## File and directory contract

- `wiki/topics/perf-lab-charter.md` — this page (the why).
- `wiki/topics/perf-lab-session-runbook.md` — the per-session procedure
  (the how).
- `wiki/ops/perf-queue.md` — the live queue. Source of truth for what
  to run next.
- `wiki/ops/best-cells.md` — current best cell at each reference
  point; updated on promotion.
- `wiki/ops/perf-log.md` — narrative timeline of what we tried.
- `wiki/ops/experiment-ledger.md` — formal receipts.
- `wiki/ops/baselines.md` — citeable baseline rows.
- `.frontier/lanes.json` — coarse lane registry for board projection.
- `scripts/canonical_sweep.py` — workhorse driver, takes `--cells-from
  <csv> --lane <label>` for ad-hoc cell lists; resumable per
  [perf-lab-session-runbook](perf-lab-session-runbook.md).
- `scripts/plot_canonical_sweep.py` — chart producer.
- `sweep_logs/lab-<lane-id>-<TS>/` — per-lane artifact dir.
- Worktrees: `~/code/gomoku-perf-<lane-id>/` on branch
  `feat/perf-<lane-id>`.

## Worktree discipline

Worktrees exist for **code-change lanes** (anything that requires
editing python or env defaults beyond what the CLI surface supports).
Pure cell-list sweeps stay on `main`.

Lifecycle for a code-change lane:

```bash
git worktree add ../gomoku-perf-<lane-id> -b feat/perf-<lane-id>
# apply the patch in the worktree
# run the cells from inside the worktree, sweep_dir under main repo's sweep_logs/
# score against the relevant reference point
if win:
    cd ~/code/gomoku
    git merge --no-ff feat/perf-<lane-id> -m "lane: <one-line summary>"
    git worktree remove ../gomoku-perf-<lane-id>
    git branch -d feat/perf-<lane-id>
else:
    git worktree remove ../gomoku-perf-<lane-id>
    git branch -D feat/perf-<lane-id>
```

Never rebase. Never fast-forward. Per [[feedback-merge-commits]].

## Stop conditions

The loop halts when **any** of:

1. Queue is empty AND there are no pending auto-generated follow-ups
   from recent lanes.
2. Three consecutive lanes return `reject` AND no compound follow-up
   is queueable (signal that the productive axis-family is exhausted
   for this session).
3. Box becomes busy with another tenant (`pgrep` finds
   `gomoku.train`, `selfplay_worker` outside our lab dirs, or
   `eval_worker`).
4. A code-change lane's worktree fails to build (import error,
   missing dep). File a `blocked` receipt, queue follow-up that
   resolves the dep, halt this loop iteration.
5. An MPS / CUDA error appears in cell logs across all workers of a
   cell. File `failed` for that cell, retry once via
   `--retry-failed`. If still failing: halt and surface to the user.

At halt: append a session-end entry to
[perf-log.md](../ops/perf-log.md) with the leaderboard delta, the
queue state, and the headline finding.

## Anti-patterns

- **Compounding too many axes at once.** A V × W × S × G all-changed
  cell is a guess, not an experiment. Compound at most two axes per
  lane.
- **Long cells that aren't info-dense.** 5 minutes is the contract.
  Longer cells only make sense for power-rail measurements
  (`powermetrics`) and that's a separate lane family with its own
  rules.
- **Promoting on a single trial.** A lane is a tuple of cells, not
  one cell. The cell list is the experiment.
- **Treating canonical_sweep.py as a configuration file.** Add
  cell-list expressivity through `--cells-from` and per-lane csvs;
  don't accumulate one-off branches inside the driver itself.
- **Forgetting receipts.** A lane that ran without filing
  experiment-ledger.md + baselines.md + perf-log.md is invisible to
  the next session. The receipt is the lane.

## Cross-refs

- [m5-max-as-mainframe](m5-max-as-mainframe.md) — the parent
  philosophy; this charter operationalizes it.
- [perf-lab-session-runbook](perf-lab-session-runbook.md) — the
  per-session mechanics (lock, --status, --retry-failed, etc.).
- [mcts-perf-ceiling](mcts-perf-ceiling.md) — what's already been
  optimized; don't re-port these from other codebases.
- Memory: [[feedback-know-the-machine]],
  [[project-perf-bench-lesson]], [[feedback-merge-commits]],
  [[user-hardware]].
