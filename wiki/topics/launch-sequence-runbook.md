# Launch Sequence Runbook

Reusable playbook for kicking off a new training run on this project,
distilled from the WL1 and WL2 launches in May 2026. Both runs were
cell-based sweep launches (`scripts/run_sweep.py`) using wave-of-lockstep
distributed self-play; the steps below assume that pattern unless noted.

This page is what a future session should follow when the user says
"start a run", "kick off WL3", "ship the next training run", etc.

## Phase 0 — Pre-launch (the cell must exist)

1. **Read the wiki first** — `wiki/index.md`, then the relevant design doc
   (e.g. `topics/wave-of-lockstep-design.md`, `topics/wl2-scale-emulation-design.md`),
   then the previous run's "live run log" in `TRAINING_WIKI.md`. This
   project's design-then-implementation gap is short; the next-run
   design is usually already written.
2. **Confirm the cell** in `scripts/run_sweep.py`. Cell name is
   load-bearing — it sets the wandb run name, sweep_runs/ dir,
   sweep_logs/ dir, and is grepped throughout the wiki.
3. **Pre-launch state check**:
   ```bash
   pgrep -fa "gomoku.train|selfplay_worker" | grep -v pgrep
   git status                                      # clean working tree
   git log --oneline -5                            # commits that anchor the run
   ls sweep_runs/<cell-name> 2>/dev/null || echo "(clean)"
   ```
4. **Two known gotchas to check first** before committing to launch:
   - **MPS INT_MAX**: any replay buffer where `capacity × N_INPUT_PLANES
     × BOARD_SIZE²` exceeds 2.147e9 will crash on the first
     `buffer.shape_stats()` call. At 17 planes × 81 cells, that's
     ~1.56M positions max on MPS. 1.5M is the standard "safe ceiling."
     If you need bigger, force buffer to CPU or shrink the cell.
   - **Wave-mode worker race**: greedy-fill workers can race with the
     trainer's `_records/v{N}/` cleanup. Already fixed (commit
     `0d2c106`): worker catches ENOENT, drops the in-flight game,
     continues. Verify the fix is in the current selfplay_worker.py
     (grep for "drop wave game"). If it's missing, do NOT launch —
     a single dead worker stalls the barrier forever.

## Phase 1 — Title card → user ACK

Before pushing the launch button, paste a title card in chat. The
user reviews it and says go. Template:

```
┌──────────────────────────────────────────────────────────────────┐
│  RUN:    <cell-name>                                              │
│  WHEN:   <date>                                                   │
│  DEVICE: MPS (M5 Max)                                             │
│  BUDGET: N epochs (~Wh wall-clock estimate)                       │
└──────────────────────────────────────────────────────────────────┘
```

Plus:
- **Hypothesis** (one paragraph, link to design doc)
- **What's different vs prior run** (table)
- **Everything else identical** (table)
- **Topology** (trainer / N workers / eval)
- **Leading indicators to watch** (table — see [#leading-indicators](#leading-indicators))
- **Failure modes that warrant stopping mid-run** (bullet list)
- **Controls** (tail / stop / wandb URLs)
- **Commits anchoring the run** (3-5 most recent that landed the changes)

The user says "go". Don't launch without it.

## Phase 2 — Smoke (highly recommended for any new lever)

For any run that adds a new lever or touches the trainer/worker plumbing,
run a smoke first. 30 epochs is enough:

```bash
nohup python scripts/run_sweep.py --cell <CELL> --epochs 30 > /tmp/smoke.out 2>&1 &
```

Wait for e30 with a Monitor:

```python
Monitor(
    description="<CELL> smoke: epoch 30 or crash",
    timeout_ms=300000,
    persistent=False,
    command='cd ~/code/gomoku && until grep -qE "^epoch 30/|Traceback|RuntimeError" sweep_logs/<CELL>/trainer.log 2>/dev/null; do sleep 3; done && echo "DONE"',
)
```

Smoke validation checklist:
- All `epoch N/30` lines present, no Traceback in trainer or worker logs
- Each new lever's signature visible (e.g. "ema model initialized",
  "[wN] weights poll interval = X.Xs", "wave vN weights=current (mix=self)")
- No race-drop logs (`grep "drop wave game" sweep_logs/*/w*.log`) — a
  few are fine and benign; floods mean something else is wrong
- Procs alive (`pgrep -fl "sweep_runs/<CELL>"` returns ~9)

After smoke passes:
```bash
pkill -f 'sweep_runs/<CELL>/'
rm -rf sweep_runs/<CELL> sweep_logs/<CELL>           # clean for real launch
```

## Phase 3 — Real launch

```bash
nohup python scripts/run_sweep.py --cell <CELL> --epochs 5000 > /tmp/launch.out 2>&1 &
```

Then verify spin-up with another Monitor (5min timeout, wait for `epoch 2/`).
Grab the wandb URL from the trainer.log:
```bash
grep "View run" sweep_logs/<CELL>/trainer.log
```

Report back: wandb URL, run ID, first two epoch lines.

## Phase 4 — Wiki + workspace

Two writes, both atomic:

1. **`TRAINING_WIKI.md`** — append a "<CELL> live run log (date, wandb <id>)"
   section. Mirror the existing WL1/WL2 sections: Setup, smoke read,
   live milestone table (one row per eval), signals to watch,
   predictions-to-falsify.
2. **`wiki/log.md`** — one maintenance entry: `## [date] notebook | <CELL>
   launched — <one-line hypothesis>`. Cross-ref the design doc + the
   TRAINING_WIKI section.

Then update the wandb workspace to include the new run:
```bash
# Edit scripts/wandb_workspace.py: prepend new run to RUNS dict
python scripts/wandb_workspace.py
# Note the new URL; replace the workspace URL in TRAINING_WIKI's run log
# and in the title card.
```

Commit all three: cell config, wiki entries, workspace script.

## Phase 5 — Monitoring

Start a `/loop` self-paced monitor. The cadence pattern that works:
**3min for first 30min** (catch any early crashes / spin-up issues),
**15min for next 3h** (arc behavior visible at this resolution),
**30min indefinitely** (steady-state).

Per check:
- `tail` trainer.log for last 3 epoch lines + last 3 eval lines
- Verify all 10 procs alive (trainer + 8 workers + eval)
- `grep -c Traceback` on the trainer log; `grep -c "drop wave game"` on
  worker logs to track race-fix activity
- Compare key metrics (pl/vl/elo, plies, win rates) to the prior check
  AND to the baseline run at the same epoch
- Persist a small state JSON to `$CLAUDE_JOB_DIR/wl_last_check.json` so
  the next check can diff against it

Push notification only on:
- Stall (trainer log hasn't advanced an epoch in 2+ min)
- Crash (Traceback in trainer or workers, or proc count drops below 10)
- Heuristic crossing (eval first goes >50% sustained)
- Plies regrowth (`selfplay/plies_p90` > 30 sustained — defense regime
  has kicked in)
- Arc start (heuristic or la4 drops more than 20pp eval-to-eval)

Avoid push for routine progress or single-eval noise.

## Phase 6 — Run end

When the user calls the run (plateau, regression, completed budget):

1. **Stop cleanly** (SIGTERM, never -9):
   ```bash
   pkill -f 'sweep_runs/<CELL>/'
   # Verify all procs gone
   pgrep -fa "gomoku" | grep -vE "claude daemon|pgrep"
   ```
2. **Stop the /loop monitor** (omit ScheduleWakeup, send final
   PushNotification).
3. **Run-end entry in `TRAINING_WIKI.md`**: arc table, validated/refuted
   hypotheses, reframe, pointer to the next-run design. Preserve the
   run dirs (`sweep_runs/<CELL>/checkpoints/` may be needed as
   past-checkpoint source for the next run).
4. **`wiki/log.md`** maintenance entry summarizing.
5. **Commit** with the run-end story in the message.
6. The wandb run stays on the server — don't delete.

## Leading indicators

| signal | what it means | when to act |
|---|---|---|
| `selfplay/plies_mean` falling | model entering fast-attack regime | normal early; alarming if it never regrows past e500 |
| `selfplay/plies_p90` > 30 | defense regime forming | celebrate / push notification |
| `time/eval_vs_heuristic_s` climbing | **hidden plies-regrowth signal**: model fights longer vs external opponents *before* selfplay plies grow (because selfplay still fast-wins vs self). 16 games per eval at constant per-move cost → wall-clock growth ≈ per-game move count growth. Surfaces real defensive capability that selfplay metrics miss. | celebrate; leads `selfplay/plies_mean` by hundreds of epochs |
| `loss/policy` stagnating + `loss/value` near 0 | model converged on its current strategy; needs perturbation | check eval bouncing pattern |
| `eval/vs_lookahead4_winrate` REGRESSING from a prior peak | catastrophic forgetting; opponent-diversity failure | flag in run log; consider stopping if it doesn't recover in ~200 epochs |
| `wave/wait_for_slowest_s` > 50% of `wave/total_s` and growing | barrier rot; one worker is stalling | investigate the slow worker; check for race-drops |
| `model_elo` oscillating ±200 across consecutive evals | the WL1 oscillation failure mode | strong signal that the run is in trouble |
| `train/ema_l2_distance` climbing unbounded (WL2+) | EMA isn't keeping up; the brains are decoupling | tune τ down OR investigate gradient pathology |
| `wave/mix_self_frac` not matching configured fraction (WL2+) | past-checkpoint loader failing silently and falling back to "self" | check checkpoint dir + worker fallback logs |

## Failure modes seen in practice

- **MPS INT_MAX** (WL1 first launch, `i2pek12v`): 5M buffer overflowed
  MPSGraph. Fix: shrink to ≤1.5M or move buffer to CPU.
- **Greedy-fill race** (WL1 second launch, `wo9py6m4` killed at e97):
  worker mkdir, trainer rm -rf, worker torch.save crashed; barrier
  stalled forever because the dead worker's per-version count never
  reached the minimum. Fix: commit `0d2c106` — worker catches ENOENT,
  drops the game, continues.
- **High-frequency strength oscillation** (WL1 third launch, `l8mbntcm`):
  per-version uniformity removed the implicit version diversity Z had
  accidentally. Model bounced 600-1280 elo across single-eval intervals,
  la4 regressed 52%→5%. Fix: WL2 scale-emulation levers (EMA, past-mix,
  jitter, grad-accum).

## Fan-out playbook for implementing a multi-lever next-run

When the next-run design has multiple independent levers (like WL2 had 4),
parallelize implementation across background agents:

1. Group levers by file to avoid merge conflicts. WL2 grouping:
   - Agent A: `gomoku/train.py` (EMA + grad accumulation)
   - Agent B: `gomoku/selfplay_worker.py` (past-mix + poll jitter)
2. Use worktree isolation (`isolation: "worktree"` on the Agent tool) so
   each agent's edits land in a separate git worktree.
3. Each agent gets a prompt that includes: scope (which files), constraint
   (don't touch other agent's files), CLI flags to add, defaults
   (always disable the new behavior so existing cells are unchanged),
   tests to write, the design doc to read first.
4. After both finish: cherry-pick each commit onto main, run pytest,
   add the cell wiring + new Cell fields, smoke, launch.

This pattern worked for WL2: two agents in parallel, ~10min wall total,
clean commits, no conflicts, 88 tests passing.

## Cross-refs

- `wiki/topics/wave-of-lockstep-design.md` — WL1 design
- `wiki/topics/wl2-scale-emulation-design.md` — WL2 design (the model
  for future next-run design pages)
- `TRAINING_WIKI.md` — live run logs for all runs in the WL series
- `scripts/run_sweep.py` — Cell definitions; CELLS dict is canonical
- `scripts/wandb_workspace.py` — workspace generator
