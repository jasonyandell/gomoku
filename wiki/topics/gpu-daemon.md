# The GPU daemon — the serial GPU-lane scheduler

`scripts/gpu_daemon.py` is the lab's **single serial GPU-lane scheduler**. The M5
Max runs exactly one MPS tenant at a time (train / gen / eval / perf). Historically
*Claude-the-orchestrator* was that scheduler, dispatching cells by hand off
[gpu-queue.md](../ops/gpu-queue.md). The daemon formalises the role: **jobs are
submitted as config files, queued on disk, and run serially on the GPU.**
`delo_derby` and manual dispatch become *clients* that submit; nobody else launches
an MPS tenant directly.

This is the machine that the [two-queue scheduler](research-lab-charter.md#two-queue-scheduler)
always described in prose — the GPU-required queue, now an actual daemon. (The
everything-else queue stays as parallel Agent fan-out; the daemon owns only the
serial GPU lane.)

## Why this shape

Decided with Jason 2026-05-24:

- **It becomes THE GPU scheduler** (not a side-car for training only). Train, gen,
  eval, and perf cells all flow through it, so there is one authority for "who is on
  the box" instead of an honour system + a `pgrep` preflight in every script.
- **A job is a config file.** Trackable, auditable, diff-able, and simpler than any
  bespoke submission API — cell-name jobs, inline params, and overrides are all just
  fields in the file. Any submission style can be expressed as a file.

## The queue is a directory (maildir-style)

A job's *state is its directory*. Moving the file IS the transition, so the whole
queue is greppable, `git`-diff-able, and survives a daemon restart:

```
lab_queue/                      # default; override with --queue-dir or $GOMOKU_LAB_QUEUE
  pending/    <id>.json         # waiting; sorted by (tier, -priority, submitted_at)
  running/    <id>.json         # the one tenant on the box right now
  done/       <id>.json
  failed/     <id>.json
  cancelled/  <id>.json
  logs/       <id>.log          # the job's subprocess output (run_sweep / raw cmd)
  events.jsonl                  # append-only audit log of every transition
  daemon.pid                    # {pid, started, queue_dir}; flock'd → one daemon only
```

The runtime dir is gitignored (like `sweep_runs/`); the audit trail lives in
`events.jsonl` plus the job config files themselves, which you can copy/commit if a
particular job spec is worth keeping.

## Two job kinds

| kind    | builds | for |
|---------|--------|-----|
| `train` | `run_sweep.py --cell C [--resume latest.pt] --max-wall-secs N --final-eval` | the established clean-stop training slice ([[project-training-slices]]) |
| `raw`   | an arbitrary argv on the serial lane | gen / eval / perf cells — `canonical_sweep`, `eval_worker`, anything GPU |

A `train` job is the [time-capped resumable slice](research-lab-charter.md#training-runs-as-gpu-required-items):
the bundle self-caps on an epoch boundary, clean-saves a resumable `latest.pt` (buffer
embedded), tears down workers, runs one `--final-eval` cycle, and the daemon reads the
final `eval/model_elo` from `eval_results.jsonl` into the job's `result`. `raw` is the
escape hatch so the daemon is genuinely the scheduler for *all* GPU work without
hard-coding a builder per tool.

Job config fields: `kind`, `cell`, `max_wall_secs`, `final_eval`, `resume_from`
(`"auto"` = resume `sweep_runs/<cell>/checkpoints/latest.pt` if present, else fresh),
`overrides` (`{epochs: N}`), `cmd` (raw), `tier` (1 architectural > 2 compound > 3
speculative), `priority` (higher first within a tier), `note`. The daemon stamps
`status`, `started_at`/`ended_at`, `returncode`, `wall_secs`, `result`.

## Serial + polite + resumable

- **Serial.** Exactly one job runs at a time.
- **Polite.** When idle the daemon preflights for *foreign* MPS tenants (`pgrep` on
  `selfplay_worker|gomoku.train|run_sweep|eval_worker`, excluding its own children)
  and **waits** rather than barging in — so it coexists with a hand-launched derby
  during the migration. `--no-preflight` only if you know you own the box.
- **Clean stop.** `gpu_daemon.py stop` (SIGTERM to the daemon) forwards SIGTERM to
  the running job's process group; `run_sweep` tears down its trainer+workers and
  saves `latest.pt`. An interrupted `train` job is re-queued with `resume_from=auto`,
  so the next start continues it with no cold buffer refill.
- **Crash resume.** A hard crash leaves the job in `running/`; the next daemon start
  *reconciles* it back to `pending` the same way. No work is silently lost.

## Usage

```bash
# run the scheduler (background it yourself; see launchd note below)
nohup python scripts/gpu_daemon.py daemon > lab_queue/daemon.out 2>&1 &

# submit a training slice (writes an auditable config file into pending/)
python scripts/gpu_daemon.py submit --kind train --cell WL5 \
    --max-wall-secs 600 --tier 1 --note "WL5 600s slice"

# resume a cell's latest.pt automatically
python scripts/gpu_daemon.py submit --kind train --cell derby-c0 \
    --max-wall-secs 1200 --resume auto

# any GPU work via a raw command
python scripts/gpu_daemon.py submit --kind raw --note "S400 gen sweep" \
    --cmd "python -u scripts/canonical_sweep.py --out-dir sweep_logs/... --cells-from ..."

# submit a pre-written config file
python scripts/gpu_daemon.py submit path/to/job.json

python scripts/gpu_daemon.py status            # daemon + queue snapshot (--json for machine)
python scripts/gpu_daemon.py show <id>         # one job's config + log tail
python scripts/gpu_daemon.py cancel <id>       # drop pending / clean-stop running
python scripts/gpu_daemon.py stop              # drain current job + exit
```

## Migration path (not yet done)

The daemon ships standalone and ready; flipping the lab to route everything through
it is a follow-up, sequenced so nothing in flight is disturbed:

1. **delo_derby → client.** Its chunk command already *is* a `train` job
   (`run_sweep --cell X --max-wall-secs N --final-eval`). Have the derby `submit` a
   chunk and poll the queue for completion instead of `subprocess`-ing `run_sweep`
   itself. The Δelo priority logic stays in the derby; it just stops being its own
   tenant. Do this when no derby is mid-flight.
2. **gpu-queue.md → feed.** Today the daemon's `pending/` *is* the machine-readable
   queue; `gpu-queue.md` stays the human narrative + RESUME STATE. A small parser
   that turns queued lanes into `submit` calls can make the markdown a literal feed
   later, but the queue itself already lives in the daemon.
3. **launchd.** A user-LaunchAgent plist can auto-start the daemon on login. Until
   then, `nohup` + `gpu_daemon.py stop` is the lifecycle.

## Cross-refs

- [research-lab-charter.md § Two-queue scheduler](research-lab-charter.md#two-queue-scheduler) — the model the daemon implements
- [gpu-queue.md](../ops/gpu-queue.md) — the human-readable lane narrative
- `scripts/run_sweep.py` — the clean-stop training slice the daemon dispatches
- `scripts/delo_derby.py` — the serial dispatcher this generalises; future client
- Tests: `tests/test_gpu_daemon.py` (GPU-free: queue lifecycle, ordering, reconcile)
- Memories: [[project-training-slices]], [[feedback-lab-scheduler]], [[feedback-autonomy-denylist]]
