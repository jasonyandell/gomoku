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
| **R-TRAIN-WL5** | small / W=8 / G=8 / sims=400 / V=64 / EMA τ=0.99 / grad_accum=4 (WL5 production recipe) | **3,297.6 aug/s** / 0.0917 epochs/s / 14.07 games/s (L10, 2026-05-23, Reviewer APPROVE) |
| ~~R-TRAIN-LEAN~~ at default sgd | same but V=512 with WL5's sgd_per_position=0.0025 | **2,362.8 aug/s** (L11, REJECT — gen win doesn't free-ride to trainer; V=512 fills buffer 2.4× faster → 3× more SGD steps/epoch → trainer monopolizes MPS) |
| **R-TRAIN-LEAN-fp16** (perf reference only — TQ canary required for production) | WL5 recipe + V=512 + sgd_per_position=0.001 + fp16 workers | **8,340.5 aug/s** / 0.0667 epochs/s / 32.19 games/s (L11b', **+152.9% vs R-TRAIN-WL5**; needs_repeat per TQ gate for production adoption; two independent levers stacked multiplicatively as the mechanism predicted) |
| ~~R-TRAIN-ANE~~ (naive) | WL5 recipe + workers on Core ML CPU_AND_NE | **1,930.3 aug/s** (L09, REJECT holistic; trainer_step_s_p50 -55.7% **confirms MPS-relief hypothesis** but Core ML worker eval ~2× slower than torch/MPS at this model size — net loss. L09c tiny-on-ANE is the remaining candidate) |

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

**Default 60-90s per cell.** Hard cap 5 min when escalation is genuinely
needed. See the [Smoke-first doctrine](#smoke-first-doctrine) section
above for the rule of thumb table.

For measurements that need a longer warmup (trainer cells, torch.compile
graph capture, ANE first-load), **split into two back-to-back cells**:
a warmup cell (no measurement recorded; `cell_status: warmup`) + a
measure cell. The driver runs them under the same lane; the receipt
aggregates.

## Two-queue scheduler

The lab is **not** a single-queue "dispatch one cell, wait, dispatch
next" loop. It's a two-queue scheduler:

| Queue | Concurrency | What goes here | Default cell wall |
|---|---|---|---|
| **GPU queue (serial)** | one at a time on MPS | live cells: self-play sweeps, R-TRAIN-* training cells, eval probes | 60-90s (smoke-first; see below) |
| **CPU queue (parallel)** | many at once via Agent fan-out | code (new scripts, evaluator backends, drivers); wiki edits; charter updates; plot generation; reviewer audits; worktree code work | n/a (subagent time) |

**Orchestrator's job**: keep both queues turning. Block only on GPU;
never block on code. When a code task surfaces, spawn an Agent in a
worktree (CPU queue) rather than serializing behind the next GPU cell.
When the GPU is busy with cell N, the orchestrator is using that wall
time to fan out code/wiki/review work in parallel.

Mid-conversation, the orchestrator is the live session (multiple `Agent`
calls in one message). For unattended drift, a cron tick is the MVP —
but it's a degenerate scheduler that only advances the GPU queue. Prefer
live-session orchestration when possible.

## Smoke-first doctrine

A 60-90s cell at ~90% confidence beats a 5-min cell at 99.99%
confidence almost always. Default `--secs-per-cell 60` (or 90). Escalate
to 5 min only when the smoke result is genuinely ambiguous (delta
within ~2x of the experimental noise floor).

**Why:** the 23-cell canonical sweep was a one-time map and reasonably
used 5-min cells. Ongoing lanes inherited that default by reflex; they
shouldn't have. We were spending 83-min lane budgets to confirm things
the 17-min version would have shown clearly.

**Rule of thumb for cell time:**

| Need | Cell wall |
|---|---|
| Single-axis pivot near a known peak | 60s |
| New-axis exploration | 90s |
| Resolving an ambiguous smoke read | 5 min (escalation, not default) |
| Trainer-loop measurement (R-TRAIN-*) | 30s warmup + 60-120s measure = ≤3 min total |
| One-time chip-map (e.g. canonical contour) | 5 min per cell, but rare |

**Smoke first; repeat only when needed.** If a lane's first cell is
clearly above the reference and clearly above noise, that's a promote
candidate — file the receipt and let the Reviewer audit. Don't run 4
more cells "for confidence" when the first one already settled it.

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

**Default-allow.** Code work is autonomous, full stop. The deny-list
below is exhaustive — if an action isn't on it, just do it. See
[conventions.md](conventions.md) for the full deny-list-as-allow-list
principle and the risk-class taxonomy (Class A/B/C).

| Class | Examples that apply to this lab | Policy |
|---|---|---|
| **A — local, reversible** | files under `scripts/`, `tests/`, `wiki/`; worktrees + merge-commits on `feat/perf-*`; per-cell artifact dirs under `sweep_logs/`; live-training cells ≤ 5 min; opening/closing/reprioritizing lanes; filing receipts | **Just do it. No size limit.** |
| **B — hard to reverse / shared state** | git push, wandb writes, archive mutations, `pyproject`/CI/deps, settings.json, modifying the charter (this page) | **Confirm with the user.** |
| **C — architectural / multi-day** | custom Metal kernels, native C extensions, model architecture changes, replacing the trainer or evaluator backend wholesale | **Discuss before starting.** |

Important corollary: **don't conflate timing/context with permission**.
The cron tick is the wrong *context* for a 100-LOC code task, but this
charter still *permits* the code. Right move: surface the task as a
CPU-queue lane (next section) and let the orchestrator fan out an Agent
to do the code in parallel.

### Reviewer gate

**No promote without Reviewer sign-off.** After every lane (and on a
periodic discipline check), spawn a Reviewer per
[perf-lab-reviewer-role](perf-lab-reviewer-role.md). Verdict APPROVE /
REVISE / BLOCK; BLOCK surfaces to the user. The loop does not commit a
`promote` decision until the Reviewer approves. Reviewer audits a
*reject* receipt too (catches confounded knobs, missed surfaces).

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

## Stop gates and escalation protocol

The lab is designed to **run forever autonomously**. The stop conditions
above are not "halt and ask the user" by default — they're a triage
list. For each, the orchestrator picks one of three actions:

- **CONTINUE** (loop self-handles via documented protocol; no user
  attention needed). The orchestrator updates the receipt / queue /
  perf-log and pulls the next lane.
- **ESCALATE** (one-line PushNotification to the user; loop pauses
  until user resumes or directs). Used for shared-state collisions,
  potential hardware issues, or decisions outside the lab's scope.
- **HALT** (clean session-end; perf-log session-end entry filed; loop
  ends gracefully). Used when there's genuinely nothing left to do
  that doesn't require user direction.

### Triage matrix

| # | Condition | Action | Self-handle protocol |
|---|---|---|---|
| **1** | Queue empty AND no auto-generated follow-ups | **HALT** | File a session-end perf-log entry summarizing the cycle. Note follow-ups that *could* be added next session (Tier-3 lanes, attribution gaps, etc.). User can re-open the queue when ready. |
| **2a** | Three consecutive `reject` results | **CONTINUE** if any unblocked Tier-2/3/bg lane is queueable | A reject doesn't mean the lab is exhausted — only that the specific lever doesn't pay. Pull the next Tier-3 lane or open a compound follow-up from the mechanism findings of the prior rejects. The 2026-05-23 session had 3 rejects (L11, L09, L05-followup) and then L06-followup nearly doubled R-S400 — the original "3-reject halt" would have missed the headline win. |
| **2b** | Three consecutive rejects AND no plausible Tier-3 or compound lane exists | **HALT** | Same as #1. |
| **3** | Box busy with another tenant (`pgrep` finds non-lab `gomoku.train`/`selfplay_worker`/`eval_worker`) | **ESCALATE** | This is a real shared-state signal — the user (or a cron) started something. PushNotification: `gomoku perf lab: box busy with <process>; pausing`. Wait. |
| **4** | Code-change lane fails to build (import, missing dep) | **CONTINUE** with patch | File a `blocked` receipt, **patch the bug in flight**, continue. The L12 driver had 4 bugs surfaced and patched mid-session 2026-05-23 (--save-every, count_records, --evaluator passthrough, --fp16-eval+coreml). Each was a 5–10 LOC fix; none warranted a session halt. Only escalate if the fix needs an architectural decision (Class C). |
| **5** | MPS / CUDA error across all workers of a cell | **CONTINUE** (1 retry) | File `failed`, `--retry-failed` once. If still failing after the retry, **ESCALATE** with the worker log tail. |
| **6** | Cell hits the wall-time cap mid-warmup (zero epochs in trainer log) | **CONTINUE** with longer measurement | The 2026-05-23 L10 first dispatch hit this — 60s measure was shorter than the trainer's first epoch (~12s warmup + ~11s/epoch); only 1 epoch in window so `epochs_per_sec=0`. Re-run with `--measurement-secs 120` (the charter's R-TRAIN-* upper-end). Don't escalate. |
| **7** | Reviewer returns `REVISE` | **CONTINUE** with fixes | Read the numbered list, apply the corrections (Class A edits to receipt / surfaces), re-spawn the Reviewer. Same lane, same lane-id, same receipt. |
| **8** | Reviewer returns `BLOCK` | **ESCALATE** | This is the only Reviewer verdict that pauses the loop. PushNotification with the BLOCK reason. |
| **9** | Charter staleness flagged by ≥ 3 consecutive Reviewers, AND the suggested fix is mechanical (text change in a doc page, no policy shift) | **CONTINUE** with charter edit | Jason 2026-05-23: "the lab should run forever autonomously". Mechanical doc-fixes are Class A even though they touch the charter, because they're just synchronizing the doc with measured reality. If the fix would change lab *policy* (a new tier, a new promotion rule, a new TQ-gate carveout), that's Class B → ESCALATE. |
| **10** | A `promote` decision requires the Training-Quality Promotion Gate (val/policy_ce + plies/game-shape + canary run) and the lab has only perf evidence | **CONTINUE** with `needs_repeat` | The lab does *not* run quality canaries. File `needs_repeat`, surface the perf finding as a new perf reference (e.g. R-TRAIN-LEAN-fp16), explicitly note "TQ canary required for production adoption — not the lab's job". Do NOT escalate; the lab's job is to find the lever, not to certify it for production. |
| **11** | A decision would change the production training recipe (R-TRAIN-WL5's current default) | **ESCALATE** | The perf lab proposes operating points; only the user decides when to flip a WL release. |
| **12** | Class C work surfaces (custom Metal kernel, native C extension, model architecture change, replacing trainer/evaluator backend wholesale) | **ESCALATE** | These are multi-day deep dives outside the lab's normal scope. Surface the proposal; don't start the work. |

### When in doubt

A useful heuristic: **the lab can autonomously do anything that's
reversible at the file/branch level and doesn't change the production
training default.** Worktrees, merge commits, receipts, charter doc-
syncs, lab follow-up queueing — all CONTINUE. Anything that affects
shared state with humans (the WL release lineage, third-party services,
the user's calendar) → ESCALATE.

### Escalation format

When ESCALATING, send a one-line PushNotification:

```
gomoku perf lab: <one-line situation>. <one-line action requested>.
```

Examples:
- `gomoku perf lab: box busy with gomoku.train PID 12345; pausing.`
- `gomoku perf lab: Reviewer BLOCK on L99: <reason>. Awaiting your call.`
- `gomoku perf lab: L20 found +50% perf at Class C model arch change; needs your design call before continuing.`

Then **pause the loop** (no ScheduleWakeup). Resume on next user prompt.

## Vibe footer (optional, per commit)

Lab commit messages MAY end with a two-line **Vibe footer** when a
lane's finding earns it. Optional, never required. The moment it
becomes required it dies (see "mando-fun" — forced fun is anti-fun).

**Format**: two lines, max ~250 chars total. The first line is a wry
observation about platform opacity (Apple silence, undocumented MPS
behavior, Core ML's three-blog-posts-disagree problem, the Metal team's
buffer-class boundaries nobody named). The second line is what we
measured or built *anyway* — earned optimism, not generic
self-congratulation. Both lines must react to something specific the
lane actually found.

**Worked examples** (shape, not template):

> Apple ships 38 TOPS of ANE marketing and zero docs on how to
> actually use it. We exported a Core ML eval model anyway, smoke
> green on CPU_ONLY and CPU_AND_NE.

> Eight workers should saturate a 14-TFLOPS GPU instantly; somehow
> we're at 30% utilization. Doesn't matter — wave=512 just got us
> +49.5% off the floor.

> Core ML's ANE scheduling is documented across three blog posts that
> disagree. We exported anyway, it ran, here's the smoke.

> The Metal team picked a buffer size class boundary at exactly 512
> elements years ago and never told anyone. We discovered it by
> hitting it. Cheers.

> PyTorch says "fp16 on MPS is slow" without ever defining "slow."
> We measured it: at V=512 it's X aug/s vs fp32 Y aug/s. There. Now
> it's a number.

**Anti-patterns**:
- Generic Apple-bashing not tied to today's result. The vibe footer
  is earned commentary, not a daily op-ed.
- Self-congratulation without specifics ("we crushed it!"). Cite the
  measurement.
- Two cynical lines or two optimistic lines. The pair tension is the
  shape — wry observation + earned optimism.
- Putting one on every commit because past commits had one. Optional
  means optional. A commit with no vibe footer is a commit that
  didn't earn one this round, and that's fine.
- Trying to be funny. The vibe footer is observational, not stand-up.
  If a real joke emerges from the observation, fine; manufacturing
  one isn't.

**Reviewer policy**: the Vibe footer is **not** part of the receipt
audit. Reviewer ignores it (no points awarded or deducted). It's pure
commit-message texture.

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
- [m5-max-fp16-and-throughput-regimes](m5-max-fp16-and-throughput-regimes.md)
  — public-facing writeup of the three surprising chip findings from
  the 2026-05-23 cycle: fp16 reversal, bandwidth/dispatch regimes,
  multiplicative lever composition. Use as the "what did the lab find?"
  reference for external readers.
- [perf-lab-session-runbook](perf-lab-session-runbook.md) — the
  per-session mechanics (lock, --status, --retry-failed, etc.).
- [mcts-perf-ceiling](mcts-perf-ceiling.md) — what's already been
  optimized; don't re-port these from other codebases.
- Memory: [[feedback-know-the-machine]],
  [[project-perf-bench-lesson]], [[feedback-merge-commits]],
  [[user-hardware]].
