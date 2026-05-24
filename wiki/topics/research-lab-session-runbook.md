# Research Lab Session Runbook

How to run a GPU-required lab item (perf cell, training slice, or sweep)
end-to-end without rediscovering the convention. Sister page to
[launch-sequence-runbook.md](launch-sequence-runbook.md) (for training
runs). Aligned with
[m5-max-as-mainframe.md](m5-max-as-mainframe.md) (the philosophy).

## When to use this page

A "perf cell" is one paired measurement at a fixed code commit on the
M5 Max: a single parameter combination, run long enough to be stable,
captured as a receipt. A "sweep" is N cells run together to map a
contour. A "training slice" is a time-capped, resumable training run
dispatched via `run_sweep --max-wall-secs --final-eval`. All three are
GPU-required lab items and use this procedure. Use this page when you
intend to file evidence — not for ad-hoc tinkering.

## Pre-flight

1. **Box must be idle.** No live WL run, no eval cycle, no other
   tenant on MPS or ANE. Verify with:

   ```bash
   pgrep -fl 'selfplay_worker|gomoku.train|run_sweep|eval_worker' || true
   ```

   If anything turns up, either stop it or postpone the session.

2. **Working tree clean** (so the commit hash you record actually
   describes the bench). `git status --short` should be empty.

3. **WL5 contention disclaimer.** If the box is *not* idle (e.g.
   sampling something during a live run), say so in the receipt
   under `hardware:` / `confidence:`. The
   `2026-05-22 current-main baseline receipts microbench` row in
   [../ops/experiment-ledger.md](../ops/experiment-ledger.md) is the
   reference example.

## Naming and locations

| Thing | Location |
|---|---|
| Raw cell artifacts | `sweep_logs/<lane-id>-<UTC stamp>/` |
| Receipt | [../ops/experiment-ledger.md](../ops/experiment-ledger.md) |
| Baseline-row entries | [../ops/baselines.md](../ops/baselines.md) |
| Verifier commands | [../ops/test-ledger.md](../ops/test-ledger.md) |
| Narrative entry | [../ops/perf-log.md](../ops/perf-log.md) |
| Lane registration | `.frontier/lanes.json` + [../ops/frontier.md](../ops/frontier.md) |

UTC stamp format: `YYYYMMDDTHHMMSSZ`, e.g. `20260522T180000Z`. Match
what `.frontier/runs/` and `sweep_logs/` already use.

## The session loop

### 1. Claim the lane

If the lane already exists in `.frontier/lanes.json`, change its
`stage` to `"in-progress"` and its `heat` to `"hot"`. Otherwise add a
new entry with `goal:`, `nextAction:`, and `doneShape:` from the
[../ops/frontier.md](../ops/frontier.md) template.

### 2. Pin the command

Every cell needs an exact, paste-able command (env vars too). The
canonical surfaces are:

- `scripts/perf_microbench.py` — single-process bounded MCTS
  generation. Use for narrow MPS-only spot checks.
- `scripts/canonical_sweep.py` — multi-worker sweep driver. Spawns
  N `selfplay_worker` subprocesses per cell, bounds wall time, counts
  records, writes `summary.tsv`. Use for the canonical 5-axis sweep
  and other production-shape sweeps.
- `scripts/run_sweep.py` — full trainer + workers + eval. Use only
  if the lane requires real training cycles (most perf lanes don't).
- `python -m gomoku.selfplay_worker` — direct worker invocation. Use
  if you want one bespoke worker (e.g. evaluator-boundary profile).

If you reach for a 5th surface, file it here first.

### 3. Capture the environment

Write a `metadata.txt` next to the artifacts containing at least:

```
git commit:    <hash>
git status:    <`git status --porcelain` snapshot>
hostname:      <hostname>
hardware:      <chip + memory + cores>
python:        <`python --version`>
torch:         <torch.__version__>
date:          <UTC ISO timestamp>
contention:    idle | live <run-name>
env-flags:     GOMOKU_DISABLE_NATIVE_MCTS=...  PYTORCH_ENABLE_MPS_FALLBACK=...
```

Without this the receipt is not reproducible.

### 4. Run the cells

Run the cells. Capture raw stdout/stderr per cell under
`<sweep-dir>/per-cell-logs/<cell-id>.log`. The driver should write
`summary.tsv` incrementally so a mid-run failure leaves usable data
and the driver is resumable (skip cells already present).

For long sweeps (hours), prefer `run_in_background` on Bash so you can
work on parallel items while it runs. Don't poll — the harness will
notify you when the job exits.

### 5. Reduce + chart

After the sweep, produce one chart per question being asked. The
canonical sweep produces `contour.png` (workers x games-per-worker
faceted by model + sims). Save the script that produced the chart
alongside it so the chart can be regenerated.

### 6. File the receipt

Required fields per
[../ops/experiment-ledger.md](../ops/experiment-ledger.md) schema. For
training-quality-affecting changes, the
**Training-Quality Promotion Gate** also applies — no `promote`
without a fixed external baseline or validation-archive read, a
plies/game-shape check, a noise caveat, and reproducibility IDs.

Pure throughput changes (worker count, wave size, etc. with no
behavior change) can skip the quality gate but still need the rest.

### 7. Update the surfaces

- Append a row per cell-class to [../ops/baselines.md](../ops/baselines.md).
- Update the named-cell defaults in
  [../ops/status.md](../ops/status.md) if the winner displaced the
  prior throughput default.
- Mark the lane completed in `.frontier/lanes.json` and
  [../ops/frontier.md](../ops/frontier.md).
- Add a narrative entry to [../ops/perf-log.md](../ops/perf-log.md).
- Update [log.md](../log.md) only for wiki-structure changes (new
  topic page, new index route, etc.).

## Cell-design defaults

When designing a new cell list, copy these defaults unless you're
explicitly testing a deviation:

| knob | default | reason |
|---|---|---|
| `--max-plies` | 16 | bounded, matches the production-shape contour artifacts |
| `--wave-mode` | on | greedy-fill is the production wave-lockstep mode |
| `--device` | `mps` | M5 Max GPU |
| `fused_eval` | on | always; Conv+BN fusion is a 1.5x win and there's no reason to leave it off |
| `native_mcts` | on | likewise; `GOMOKU_DISABLE_NATIVE_MCTS=1` is for A/B receipts |
| model | small, stem_padding=1 | matches WL5 + production-contour default |
| per-cell wall | **60-90 s** (charter v3 smoke-first) | escalate to 300 s only when smoke is ambiguous (delta within ~2x noise floor) |

## Smoke-first pattern

Per charter v3, the default cell time is **60-90s, not 5 min**. A
60-90s cell at ~90% confidence beats a 5-min cell at 99.99% confidence
almost always.

Workflow:

1. **Smoke (60-90s/cell)**: run the lane with `--secs-per-cell 60`. If
   the result is clearly above the reference and clearly above noise,
   file the receipt as a promote candidate. If clearly below: reject.
2. **Escalate only when ambiguous**: if the result is within ~2x of
   the experimental noise floor (≈±2% on the established benches),
   re-run the same cells with `--secs-per-cell 300` for a stable read.
3. **Don't run the long version first**. Past-Claude defaulted to
   300s because the canonical sweep used it; subsequent lanes
   inherited that without justification. Charter v3 fixes this.

Example smoke-first lane (typical case):

```bash
# 3-cell smoke at 60s/cell: ~4 min wall total
python scripts/canonical_sweep.py \
  --out-dir sweep_logs/lab-<lane-id>-<TS> \
  --cells-from <path>/cells.csv \
  --lane <lane-id> \
  --secs-per-cell 60
```

If a cell's delta vs the reference is clear (>5% in either direction),
that's the answer. File the receipt.

## Fan-out pattern (everything-else queue)

When a lane is **code-only** (Class A under
[conventions.md](conventions.md)) — new script, evaluator backend,
driver, wiki edit, plot generation — it goes on the everything-else
queue, not the GPU-required queue. The orchestrator spawns an Agent
in a worktree to do the code work in parallel with whatever GPU item
is currently running.

Pattern for the orchestrator (live session):

```
# Suppose GPU is running cell N (~5 min wall).
# In the same turn, spawn parallel Agents for CPU lanes:

Agent(L12 driver):  worktree feat/perf-L12 → write scripts/lab_train_cell.py → smoke → merge
Agent(L05 compile): worktree feat/perf-L05-compile → wire flag → smoke → merge
Agent(L06 fp16):    worktree feat/perf-L06-fp16 → wire flag → smoke → merge

# When GPU cell N finishes, file receipt + spawn Reviewer in parallel
# with the still-running CPU Agents.
```

Three CPU Agents + one GPU cell + one Reviewer Agent = 5 parallel
streams of work. The wall-clock cost is dominated by whichever
single task is longest — usually the GPU cell at ~5 min.

**Anti-patterns:**
- Serializing CPU work behind GPU cells. Code doesn't need MPS.
- Spawning more parallel Agents than the laptop's effective CPU
  parallelism — 3-5 concurrent code Agents is a reasonable ceiling
  on the M5 Max.
- Holding the GPU cell back to "let code lanes finish first" — the
  GPU lane has its own constraint (one at a time) but should always
  be advancing if there's a ready lane.

## Resumability contract

Any multi-hour perf driver should be resumable. The canonical sweep
driver (`scripts/canonical_sweep.py`) is the reference implementation
of the contract below; new drivers (ANE rail proof, packed-buffer
ablation, etc.) should match it instead of reinventing it.

A driver satisfies the contract when all of these hold:

1. **Stable cell IDs.** Each cell's identity is derived purely from
   its parameters (not from list position). Adding cells later must
   not renumber existing rows.
2. **Atomic source of truth.** A single `summary.tsv` (or equivalent)
   under the run dir. Each completed cell appends one row and fsyncs.
   Rewrites use write-temp-then-rename. Partial rows never appear.
3. **Per-cell status flag.** Each row carries a `cell_status` column
   (e.g. `ok` / `failed`). Resume by default skips both; pass
   `--retry-failed` to clear failed rows + wipe their cell dirs +
   re-run.
4. **Lock file.** `<out-dir>/.sweep.lock` holds the PID of the
   running driver. A second invocation aborts unless the held PID is
   dead. Dead-PID locks are reclaimed automatically.
5. **`--status` mode.** Prints `done / failed / pending / ETA` and
   exits without spawning any GPU work. ETA uses the median wall
   seconds of completed cells, so it gets more accurate as the sweep
   progresses.
6. **Wall-time budget.** `--max-wall-secs N` exits cleanly after the
   budget. Pair with `--max-cells M` for cell-count caps. Both let
   you do a 10-minute top-up between meetings.
7. **Signal-clean shutdown.** SIGINT and SIGTERM terminate live
   workers (via the start-new-session pgroup), do not record the
   interrupted cell, and release the lock. The next resume re-runs
   only the interrupted cell, not everything since.
8. **`latest` symlink.** Each session refreshes
   `sweep_logs/<lane>-latest` → run dir. `--out-dir latest`
   shortcuts the most recent sweep so you never have to copy a
   timestamped path from history.

### Practical recipes for the canonical sweep

```bash
# Start a fresh sweep:
python scripts/canonical_sweep.py \
  --out-dir sweep_logs/canonical-sweep-$(date -u +%Y%m%dT%H%M%SZ)

# Pick up where you left off (most recent sweep):
python scripts/canonical_sweep.py --out-dir latest

# How far along am I, what's the ETA?
python scripts/canonical_sweep.py --out-dir latest --status

# Re-run cells that failed (e.g. crashed on MPS contention):
python scripts/canonical_sweep.py --out-dir latest --retry-failed

# 30-minute top-up between meetings:
python scripts/canonical_sweep.py --out-dir latest --max-wall-secs 1800

# Hard cap on cells this session (useful for testing):
python scripts/canonical_sweep.py --out-dir latest --max-cells 3

# Just the tiny + medium cells (substring match on cell_id):
python scripts/canonical_sweep.py --out-dir latest --only tiny,medium
```

If the lock file is stale because a SIGKILL or crash left it
behind on a dead PID, the next invocation reclaims it
automatically. If something pathological happens, `rm
<out-dir>/.sweep.lock` and retry — but check `pgrep` first to be
sure nothing is actually running.

## Anti-patterns

- **Single-process intuition transplanted to multi-worker production.**
  See [[project-perf-bench-lesson]] in memory. A 1-worker microbench
  number does not predict 8-worker production throughput.
- **GPU-percent as the score.** See
  [activity-monitor-perf-runbook.md](activity-monitor-perf-runbook.md).
  Wall seconds, games/sec, and aug-positions/sec are the score.
- **Promoting a throughput win without a quality check** on
  training-behavior-affecting changes. The
  Training-Quality Promotion Gate exists because we got burned by
  exactly this in the WL series.
- **Forgetting to record env flags.** `GOMOKU_DISABLE_NATIVE_MCTS`,
  `PYTORCH_ENABLE_MPS_FALLBACK`, `GOMOKU_DEVICE` etc. all materially
  change results. If they're not in the receipt, the receipt is
  unreproducible.

## Cross-refs

- [m5-max-as-mainframe.md](m5-max-as-mainframe.md) — why the lab
  exists at all.
- [activity-monitor-perf-runbook.md](activity-monitor-perf-runbook.md)
  — how to interpret what Activity Monitor shows during a cell.
- [mcts-perf-ceiling.md](mcts-perf-ceiling.md) — what's already
  been optimized; don't re-port these.
- [launch-sequence-runbook.md](launch-sequence-runbook.md) — sister
  runbook for training runs.
- [research-lab-charter.md](research-lab-charter.md) — the charter
  that governs the research lab operating loop.
- Memory: [[feedback-know-the-machine]], [[project-perf-bench-lesson]],
  [[user-hardware]].
