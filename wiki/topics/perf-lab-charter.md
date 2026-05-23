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

Single primary number: `aug_pos_per_sec` measured by
`scripts/canonical_sweep.py` under default knobs (`--secs-per-cell 300
--max-plies 16 --device mps`).

Tracked at three quality reference points:

| ref point | cell shape | current best (2026-05-23) |
|---|---|---|
| **R-S400** | small / W=8 / G=8 / sims=400 / wave=128 | 4,048 aug/s (just promoted) |
| **R-S200** | small / W=8 / G=8 / sims=200 / wave=64 | 6,006 aug/s (canonical-sweep baseline) |
| **R-S100** | small / W=8 / G=8 / sims=100 / wave=64 | 11,151 aug/s (canonical-sweep baseline) |

Promotion at a quality point requires a *no-behavior-change* knob
movement that improves the number. Lanes that change sims, model size,
or anything semantic are explorations, not promotions to a quality
point.

A secondary visible number per session: number of receipts filed,
number of promotes, number of rejects. Three consecutive rejects with
no compound follow-up = stop signal.

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

## Priority function

```
priority = (E[delta_aug_pos] × P[lane succeeds]) / wall_cost_seconds
```

Default seeds per lane in the queue:
- `E[delta_aug_pos]` — gut estimate in aug/s. Refined as similar lanes
  resolve.
- `P[lane succeeds]` — gut estimate of probability the lane improves
  the relevant reference point. Single-axis pivots near recent wins:
  high (0.5-0.8). Speculative cross-axis combinations: 0.2-0.4.
  Architectural changes: 0.1-0.3 but with very high delta when they
  hit.
- `wall_cost_seconds` — `n_cells × secs_per_cell + 30 × n_cells`
  (setup margin).

Re-rank after every completed lane. A new high-value compound from a
just-promoted axis can leapfrog older speculative lanes.

## Autonomy boundaries

| Autonomous | Manual confirm |
|---|---|
| Designing cells | Promoting a candidate into a live training run (Training-Quality Gate applies) |
| Running ≤ 30 cells per session unattended | Structural code changes (custom Metal kernels, ANE port, native code rewrites) |
| Worktree create / merge / remove on `feat/perf-*` branches | Anything that changes `pyproject.toml`, CI, or external deps |
| `git merge --no-ff` integrations | `git rebase`, `git push --force`, `git reset --hard` — never |
| Promoting a new "best cell" at a quality point | Modifying wandb project, archives, or trained-model artifacts |
| Filing receipts, baseline rows, perf-log entries | Modifying the charter (this page) — surface it first |
| Opening, closing, reprioritizing lanes | Stopping the buffer-curation / external-engines / ANE-rail lanes that another workstream owns |

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
