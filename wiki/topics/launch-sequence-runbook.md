# Launch Sequence Runbook

> ✅ **Status: LIVE** *(2026-07-04)* — **the procedure is current.** The worked *examples* are WL-era
> (era-1, 9×9 May 2026): cell names, wandb ids, and reference metrics below are
> illustrative of the era they were distilled from. For the **current era's** cell
> surface and recipe, use [training-run-reference.md](training-run-reference.md)
> (the parameter dictionary) and [sound-world-recipe.md](sound-world-recipe.md)
> (the current frontier recipe); this page is the launch/monitor/stop *procedure*
> those runs still follow.

Reusable playbook for kicking off a new training run on this project,
distilled from the WL1 and WL2 launches in May 2026. Both runs were
cell-based sweep launches (`scripts/run_sweep.py`) using wave-of-lockstep
distributed self-play; the steps below assume that pattern unless noted.

This page is what a future session should follow when the user says
"start a run", "kick off WL3", "ship the next training run", etc.

## Phase 0 — Pre-launch (the cell must exist)

1. **Read the wiki first** — `wiki/index.md`, then the relevant design doc
   (e.g. `topics/wave-of-lockstep-design.md`; older design pages like WL2's
   have since been rotated out — preserved in git history, not a parallel
   directory),
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

Two distinct cadence patterns depending on whether the user is around:

### 5a — Active monitoring (user present)

Start a `/loop` self-paced monitor. The cadence pattern that works:
**3min for first 30min** (catch any early crashes / spin-up issues),
**15min for next 3h** (arc behavior visible at this resolution),
**30min indefinitely** (steady-state).

### 5b — Overnight / unattended monitoring (CronCreate, fixed cadence)

For long unattended runs (overnight, multi-hour cap chases), prefer
`CronCreate` over `/loop`. Each tick is independent; the harness doesn't
re-enter your prompt every time, the cron fires the same self-contained
check prompt verbatim. Validated against WL5 phase-2 overnight (12.5 hr,
50 ticks, zero false alarms).

```
CronCreate({
  cron: "7,22,37,52 * * * *",                        // every 15 min, off the :00/:15/:30/:45 marks
  recurring: true,
  prompt: "<the self-contained check prompt, see below>"
})
```

Pick an off-15-minute offset (`7,22,37,52` not `*/15`) for politeness —
every monitor that fires "on the 15" lands on the API at the same
instant.

The cron prompt **must be self-contained** because each fire is a fresh
context. Template:

```
<RUN> health check. wandb=<id>, cell=<cell-name>. Resumed at e<N> with cap e<M>.
Steps:
(1) ps -A | grep -E 'gomoku\.(train|selfplay_worker|eval_worker)' | grep <cell-name> | grep -v grep | wc -l   → should be 10
(2) tail -8 sweep_logs/<cell-name>/trainer.log   → read epoch, pl, vl, plies, elo
(3) tail -200 sweep_logs/<cell-name>/trainer.log | grep -iE 'nan|error|traceback|crash' | tail -5   → should be empty
Reference state: pl~X.XX, vl~X.XX, plies~XX, elo bouncing AAAA-BBBB. Prior ATH elo CCCC.
Push via PushNotification ONLY on:
  - process count != 10
  - NaN/traceback in last 200 lines
  - sustained plies < 25 (collapse)
  - new elo > <ATH> ATH
  - epoch >= <cap> (run end)
Otherwise report one short text line with epoch/elo/plies and stay quiet — no further scheduling
(cron handles cadence).
```

Critical points the WL5 era proved out:

- **Filter `ps` by cell name**, not just `gomoku.*`. Other sessions
  (frontier perf benches, contour sweeps) routinely spawn additional
  `gomoku.train` + worker processes in sibling worktrees. A bare
  `pgrep -fc 'gomoku\.(train|selfplay)'` reports 15 procs when WL5 has
  10 + a perf bench has 5 — that's a phantom alert. The cell-name
  filter scopes the count to *this* run.
- **`ps -A | grep ... | wc -l`**, not `pgrep -fc`. On macOS, `pgrep -fc`
  with no matches returns empty string (NOT "0"), which breaks
  `[ "$(pgrep -fc ...)" = "0" ]` polling loops. `wc -l` always returns
  a number.
- **One push trigger per protocol section**: only the listed conditions
  push. Everything else is an inline one-line report. The cron fires
  again in 15 min; if the operator is around they'll see the report,
  and if not, only the real alerts wake them.

### 5c — The overnight operating contract (workhorse + narrator split)

For multi-hour / overnight autonomous pushes Jason's standing contract is
**work to a stated horizon, not to the next clean milestone** ("work until
tomorrow" = keep the lab turning, file receipts, spawn Reviewers; don't park at
a green checkpoint). He wants to leave — movie, sleep — and trust the lab runs
itself and reports back. (First established 2026-05-24 at the Δelo Derby v1
launch; the dispatch-and-verify framing is [cockpit-vs-autopilot](cockpit-vs-autopilot.md).)

The architecture that survives a Claude session restart is a **two-part split**,
not one in-context loop:

- **Workhorse** — a standalone, crash-resumable python scheduler launched via
  `nohup` so it outlives the Claude session. Build `--resume` into it from the
  start (reconcile from on-disk state, no cold refill). Example: `delo_derby.py`
  (crash-resumable, per-chunk failure isolation).
- **Narrator** — a lightweight cron (the `7,22,37,52 * * * *` cadence of §5b)
  that **reports from disk state, never from in-context orchestration**. Each
  fire is a fresh context (§5b), so it must read run state off disk, not hold it.
  The narrator **doubles as a watchdog**: if the workhorse PID is gone (and the
  run isn't at its cap), **relaunch it with `--resume`** rather than just
  alerting. PushNotify on transitions only (per the §5b tight-list); on
  completion it files the results receipt and `CronDelete`s itself.

Two more parts of the overnight contract:

- **Fan out** the build across background `Agent`s, **grouped by file** to avoid
  edit conflicts (pair edits with `isolation: worktree`).
- **Improve the skills as friction appears** — the self-improvement clause is
  expected, not optional; a remembered procedure that keeps re-surfacing becomes
  a janitor + gauge ([worktree-hygiene](worktree-hygiene.md) pattern), and a recurring narration a
  cron.

### Per-check actions (both modes)

- `tail` trainer.log for last 3-8 epoch lines + most recent eval line
- Verify procs alive (count + cell-name filter as above)
- Last-200-lines error scan: `grep -iE 'nan|error|traceback|crash'`
- Compare key metrics (pl/vl/elo, plies) to phase-reference state
- For active mode only: persist a small state JSON to
  `$CLAUDE_JOB_DIR/wl_last_check.json` so the next check can diff

### Push (PushNotification) triggers — keep the list tight

- Stall (trainer log hasn't advanced an epoch in 2+ min)
- Crash (Traceback in trainer or workers, or **cell-filtered** proc
  count drops below 10)
- New ATH crossed (eval-side strength milestone)
- Sustained plies collapse (`< 25` for 3+ evals AND `vl < 0.04` — see
  [[feedback_absorption_phase]] in session memory: only push on
  plies-collapsing-with-low-vl, not on h/la2 dips, which oscillate
  routinely)
- Run-end (epoch ≥ cap, or process count drops to 0 from a clean exit)

Avoid push for: routine progress, single-eval noise, h/la2 dips
without plies collapse, or proc count going *up* (that's concurrent
worktree activity, not a problem).

## Phase 6 — Run end

Three flavors of "end" — they look similar but the procedure differs:

| Flavor | What happened | Process state |
|---|---|---|
| **Cap-reached** | Trainer hit `--epochs N`, exited cleanly, wandb finalized | Trainer gone; **8 workers + 1 eval still alive and polling** |
| **User-stopped** | Operator decided to stop | All 10 still alive until SIGTERM'd |
| **Crash** | Trainer crashed mid-run | Trainer gone; workers + eval polling indefinitely against a stale `worker_weights.pt` |

The non-obvious case is **cap-reached**: the trainer exits cleanly and
prints the wandb finalize banner, but the 8 self-play workers and the
1 eval worker keep polling for a new model version that will never
come. Process count drops 10 → 9 (no trainer) → still consuming MPS.
This is what the cron monitor will see as `proc count != 10` at the
tick after the cap.

### Run-end procedure

1. **Stop everything cleanly** (SIGTERM, never -9):
   ```bash
   pkill -TERM -f 'sweep_runs/<CELL>/'
   sleep 3
   # Verify all procs gone (cell-name-filtered to avoid false positives
   # from concurrent worktrees)
   ps -A | grep -E 'gomoku\.(train|selfplay_worker|eval_worker)' | grep <CELL> | grep -v grep | wc -l   # expect 0
   ```
2. **Stop the monitor**:
   - `/loop` mode: omit ScheduleWakeup at end of turn
   - Cron mode: `CronDelete({id: "<cron-job-id>"})`
   - Send a final `PushNotification` summarizing the run end so the
     operator (if away) hears the news.
3. **Compute the run-end stats with python**, not awk. macOS's `awk`
   does not support 3-argument `match()`. Sample script:

   ```python
   import re
   from pathlib import Path
   log = Path("sweep_logs/<CELL>/trainer.log").read_text().splitlines()
   ep_re = re.compile(r'epoch (\d+)/\d+ games=(\d+) buf=\d+ new=\d+ steps=\d+ pl=([\d.]+) vl=([\d.]+) plies=([\d.]+)')
   wr_re = re.compile(r'wr\[random=(\d+)% heuristic=(\d+)% vs_lookahead2=(\d+)% vs_lookahead4=(\d+)%\] elo=(\d+)')
   # filter by START_EPOCH, compute min/mean/max for pl/vl/plies, and
   # min/median/max + best-epoch for the elo series
   ```

   (zsh also chokes on bash compound commands containing `==` or `===`,
   so don't shell-script the stats summary — write it as a `.py` and
   invoke it.)
4. **Run-end entry in `TRAINING_WIKI.md`** using the **phase-N close-out
   template** below. WL5 phase-1 and phase-2 closes are the canonical
   examples; mirror their structure.
5. **`wiki/log.md`** maintenance entry summarizing.
6. **Commit + push** carefully — see [#commit--push-checklist](#commit--push-checklist)
   below for the deploy-trigger checks.
7. The wandb run stays on the server — don't delete. The
   `sweep_runs/<CELL>/checkpoints/` directory may be needed as a
   past-checkpoint source for the next cell.

### Phase-N close-out template (for TRAINING_WIKI.md)

Used for WL5 phase-1 close (2026-05-21) and WL5 phase-2 close (2026-05-22).
Mirror this structure:

```
### <CELL> phase-N close — <one-line subtitle>, <date> (<wall-time>)

<2-3 sentence framing paragraph: what bounded this phase, why it deserves
its own write-up>

**Phase N final state (e<NNNN>):**
- <epochs run>, <games generated>
- <buffer state, run-end metric snapshot>
- 0 NaN, 0 worker deaths, 0 barrier stalls (or actual counts)

**Phase N stats (n=<epochs> epoch lines):**
| metric | value | vs prior reference |
|---|---:|---|
| wall/epoch (median) | X.Xs | |
| pl (mean) | X.XXX | |
| vl (mean) | X.XXX | |
| plies (mean) | X.X | |
| elo (mean) | XXXX | |
| epochs/hr | XXX | |

**Eval scoreboard (<N> eval cycles in segment):**
| metric | value |
|---|---:|
| elo min/median/max | XXXX / XXXX / **XXXX** |
| la4 min/median/max | XX% / XX% / XX% |
| ... | |
| **Best elo: XXXX at e<N>** | la4=X% la2=X% h=X% |

**Run shape, summarized:**
| sub-phase | epochs | story |
|---|---|---|
| <name> | e<N>-e<M> | <one-line behavioral summary> |
| ... | | |

**What got validated:** <bullets>

**What didn't (yet) happen that we were hoping for:** <bullets>

**Run artifacts:**
- wandb: `<run-id>` <continuity note>
- Trainer log: `sweep_logs/<cell>/trainer.log` (this-phase lines: e<N>-e<M>)
- Last checkpoint: <path> (note keep_last_n pruning if relevant)
- Commits anchoring this phase: `<sha>` ..., `<sha>` ...

**Cross-refs:** <bullets — design doc, prior phase close, related topic pages>
```

### Commit + push checklist

Run-end commits often land alongside other-session changes. Defend
against accidental cross-contamination:

1. `git status` first. If there are uncommitted changes from another
   session, **commit them as their own commit** before your run-end
   commit. Their changes have nothing to do with the run-end and
   reviewing them mixed together is harder.
2. `git add` specific files, **never** `git add -A` or `git add .`
   (catches `.env`, secrets, sweep_runs/ binaries, etc.).
3. **Pre-push deploy-trigger check.** Push to `main` triggers the CF
   deploy workflow when *any* `app/**` path changes:
   ```bash
   git diff --stat origin/main..main -- app/ .github/ | tail -10
   ```
   If you see `app/*` files in the diff, decide whether a deploy is
   intended. If not — don't push (or push the run-end commits to a
   branch, leave `app/` changes for a separate PR).
4. `git push origin main`. The 28-commits-ahead state after an
   overnight run is normal (frontier merges + run-end commits stack);
   push happily.

## Leading indicators

| signal | what it means | when to act |
|---|---|---|
| `selfplay/plies_mean` falling | model entering fast-attack regime | normal early; alarming if it never regrows past e500 |
| `selfplay/plies_p90` > 30 | defense regime forming | celebrate / push notification |
| `time/eval_vs_heuristic_s` climbing | **hidden plies-regrowth signal**: model fights longer vs external opponents *before* selfplay plies grow (because selfplay still fast-wins vs self). 16 games per eval at constant per-move cost → wall-clock growth ≈ per-game move count growth. Surfaces real defensive capability that selfplay metrics miss. | celebrate; leads `selfplay/plies_mean` by hundreds of epochs |
| All three baselines (h / la2 / la4) climbing *together* at one eval | **balanced strength profile** — distinct from the single-baseline-spike pattern (heuristic 88% while la4 0%) that all WL1/WL2 arcs showed at their peaks. WL3 first showed this at e515: h50/la2:25/la4:38. Hypothesis: balanced-then-sustained is the leading indicator of "real" learning rather than baseline-specific overfit. | celebrate; track whether the next eval holds the *profile*, not just one baseline |
| Single-eval reading bounces 0% → 50% → 0% across consecutive cycles | **sample-size variance, not real bouncing** — 16 games per eval has wide CI. A single 16-game eval reading is a hint, not a fact. WL3 e361 showed h=5% in trainer eval but h=35% on a 40-game ad-hoc retest. Don't react to single bounces; wait 2-3 evals before calling an arc. For forensic checks, snapshot the checkpoint aside (keep_last_n prunes fast) and rerun with `--n-games 40` to halve the CI. | don't react to one bounce; check trend |
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

## Handoff friction — gotchas that will bite the next session

Distilled from the WL3 / WL3.1 / WL4 incidents on 2026-05-21. These
are non-obvious things a future-Claude (or Jason after sleep) needs
to know before touching a paused or crashed run.

### Checkpoint anatomy: `latest.pt` vs `epochNNNN.pt`

- `epochNNNN.pt` (slim, ~5MB): model + EMA + optimizer state + wandb_run_id.
  Written every `save_every` epochs (default 1). Subject to `keep_last_n=3`
  pruning — only the last 3 survive.
- `latest.pt` (heavy, ~8GB): same as epoch checkpoints PLUS the full
  1.5M-position replay buffer. Written every `save_buffer_every` epochs
  (default 100). The slim epoch checkpoints between buffer-saves don't
  include the buffer.
- **To resume a run with its replay buffer**, you MUST use the latest
  `save_buffer_every` checkpoint. `--resume sweep_runs/<cell>/checkpoints/latest.pt`
  reads the model state at whatever epoch latest.pt was last saved
  (which can be lower than the highest-numbered epoch checkpoint).
- The on-disk epoch number ≠ the epoch in the checkpoint payload when
  resuming from `latest.pt`. WL4 resume requested e1536 but actually
  loaded e1500 because `latest.pt` was last saved at e1500.

### `keep_last_n=3` is brutally short — snapshot aside immediately

- The default cell config keeps only the last 3 per-epoch checkpoints.
  At ~5s/cycle in production this is **15 seconds of training history**.
- If you need to preserve a specific epoch (e.g. for forensic comparison,
  a known-good fallback during recovery, or a milestone marker), copy
  it aside the moment you decide it's important:
  `cp sweep_runs/<cell>/checkpoints/epoch1234.pt sweep_runs/<cell>/checkpoints/milestone-e1234.pt`
- The `latest.pt` is preserved through `mv` of the cell dir, so renaming
  to `<cell>.paused-eN` keeps both the slim checkpoints AND the buffer
  checkpoint that's the actual resume point.

### `--resume` continues the OLD wandb run id

- The trainer pulls `wandb_run_id` from the checkpoint payload during
  `--resume`. There's no flag to override; resume always continues the
  prior wandb timeline.
- **Implications:**
  - If you want a clean separate wandb run for an experiment branched
    from an existing checkpoint, you'd have to strip the wandb_run_id
    from the checkpoint before resuming (or edit train.py to accept a
    `--wandb-new-run` flag — not currently a thing).
  - WL4 inherited 44cxzc9d (was WL3.1's). For the K=2→K=0 curriculum
    experiment this turned out to be a *feature* — the chart shows the
    transition at step 1537 as one continuous trajectory.
  - If wandb resume seems to lose history or step numbers go weird,
    the resume is probably overlaying an existing run. Check wandb_run_id.

### Cell rename + path divergence

- `scripts/run_sweep.py`'s cell dirs come from `cell.name`. Renaming a
  cell (e.g. WL3.1 → WL4 in CELLS) means a *new* `sweep_runs/<new>/`
  and `sweep_logs/<new>/` are created. The old dirs aren't touched —
  but `--resume` lets you point at the old cell's checkpoint while the
  new run's artifacts go to the new dir.
- This is the pattern for "branch an experiment from a paused run":
  preserve the old dirs as `<cell>.paused-eN/`, add a new cell with the
  experimental knob change, `--resume` from the paused checkpoint.

### Workspaces API doesn't update in place

- `python scripts/wandb_workspace.py` creates a NEW view URL every
  time. There's no API to update an existing view.
- The URL printed by the script is the latest authoritative one — any
  URLs in wiki/TRAINING_WIKI.md from earlier runs are stale and just
  show the wandb default workspace if followed.
- The cure: the wiki entries always mention the URL exists in
  `scripts/wandb_workspace.py` — regenerate when needed.

### macOS `pgrep` quirks

- `pgrep -fl "name"` returns lines (PID + cmdline), `pgrep -f "name"`
  returns just PIDs.
- `pgrep -f "WL3.1\b"` does NOT work as expected on macOS — the `\b`
  word boundary returns 0 matches even when there are obvious hits.
  Use the pattern without anchors: `pgrep -f "WL3.1"`.
- `pgrep` can transiently include your own bash subshell as a "matching"
  PID. After kill operations, `sleep 3-5` then re-check before
  declaring the run dead.

### Old `/loop` chains keep firing

- When you change the prompt to a `/loop` (e.g. switch from monitoring
  WL3.1 to WL4), the *previous* scheduled wakeup still fires once with
  the old prompt. Recognize stale loops and don't reschedule them
  (just don't call ScheduleWakeup at the end of the turn).
- They're harmless other than one extra check with the wrong context.

### Resume + wave-mode: confirm worker_weights.pt epoch tag is right

- The initial `_publish_worker_weights()` call in `gomoku/train.py`
  USED to default to `epoch=0`. For a fresh run that's fine — the
  wave-mode barrier expects v0 and workers write v0. For a `--resume`
  from a non-zero epoch, this was a STALL BUG: worker_weights.pt
  tagged epoch=0, workers write to v0/, trainer barrier waits on
  v{start_epoch}, they never meet.
- Fixed in commit `e8e0cef`: `_publish_worker_weights(epoch=start_epoch)`.
- **Verification step on every wave-mode resume:**
  ```bash
  python -c "
  import torch
  p = torch.load('sweep_runs/<cell>/checkpoints/worker_weights.pt', map_location='cpu', weights_only=False)
  print('worker_weights.pt epoch tag:', p.get('epoch'))
  "
  ```
  The epoch should match what the trainer log says it resumed at.
  If it shows `0` after a resume, the fix is missing.
- **Smell test mid-run:** if you see a `v0/` directory growing rapidly
  after a resume, you have this bug. Check `ls sweep_runs/<cell>/checkpoints/_records/`.

### Snapshot redundancy: `$CLAUDE_JOB_DIR/` is ephemeral

- `$CLAUDE_JOB_DIR/` is `/Users/jason/.claude/jobs/<job_id>/` and gets
  cleaned up when the job ends. Files there will NOT persist across
  sessions.
- Diagnoses, state files, and forensic notes can go there during a
  session. **Anything you want a future session to find should live in
  the repo** — either as wiki content, or in `sweep_runs/<cell>.dead-eN/`,
  or as a committed file.

### Concurrent worktree procs inflate `pgrep` (WL5 phase-2 overnight)

- Frontier perf experiments routinely launch their own gomoku training
  + worker processes from `.frontier/worktrees/...` directories. A
  monitor that just counts `pgrep -fc 'gomoku\.(train|selfplay_worker)'`
  will see 15 when WL5 is healthy at 10 + a 5-proc contour sweep is
  alongside. That looks like an alert; it isn't.
- Always scope proc counts to the cell: `ps -A | grep -E
  'gomoku\.(train|selfplay_worker|eval_worker)' | grep <CELL-NAME> |
  grep -v grep | wc -l`. The cell name appears in `--worker-input-dir`
  / `--worker-weights-path` / `--checkpoint-dir`, so it's reliably in
  every WL5-owned process's cmdline.
- Proc count going *up* from N is concurrent activity. Proc count
  going *down* is the real alert.

### macOS `awk` doesn't have 3-arg `match()`

- GNU awk supports `match(string, regex, array)` to capture groups.
  macOS BSD awk only supports `match(string, regex)` returning position +
  setting `RSTART`/`RLENGTH`. Scripts that worked on Linux silently
  syntax-error here.
- For any non-trivial log parsing or stats summary, **write a `python3`
  one-liner via heredoc**, not awk. Example pattern lives in the run-end
  procedure above.

### zsh chokes on `==` / `===` inside bash compound commands

- When constructing inline status reports via Bash, zsh (Jason's
  default shell) errors `(eval):1: == not found` on tests that use
  bash-style `=` operators. Symptom: command output truncated at the
  first `=`.
- Workaround: put the comparison in a python `-c` block, or split the
  Bash call into separate independent commands.

### Buffer snapshot lags slim checkpoints by `save_buffer_every` epochs

- `latest.pt` (8.8 GB, includes buffer) is written every
  `save_buffer_every` epochs (default 100). `epochNNNN.pt` (5.3 MB,
  weights only) is written every `save_every` (default 1).
- Resuming from `latest.pt` rolls the trainer back to whatever epoch
  the buffer snapshot was last written — up to 100 epochs behind the
  most recent slim checkpoint on disk.
- **This trade-off is fine and often the right call**: 100 epochs of
  weight drift is recoverable in <30 minutes of wall time; rebuilding
  the 1.5M-position buffer from empty takes hours. WL5 phase-2 resume
  burned 49 epochs to keep the buffer warm — no regret.
- Verify the actual resume-target epoch from trainer.log's first line
  after resume; don't trust the on-disk slim checkpoint number.

## Cross-refs

- `wiki/topics/wave-of-lockstep-design.md` — WL1 design
- wl2-scale-emulation-design.md — WL2 design (the model for future next-run
  design pages) *(removed 2026-07-04; recover: `git show ca76350:wiki/_archive/topics/wl2-scale-emulation-design.md`)*
- `TRAINING_WIKI.md` — live run logs for all runs in the WL series
- `scripts/run_sweep.py` — Cell definitions; CELLS dict is canonical
- `scripts/wandb_workspace.py` — workspace generator
