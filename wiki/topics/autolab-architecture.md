# The Autolab — ledger-spine architecture

**Status:** epic [#53]. Formalizes Jason's spec (2026-06-18) of a self-driving lab
that treats the M5 Max as a mainframe. Supersedes the *framing* of #2 (three-tier
queue) and folds in #19 (nonstop derby daemon) — same pieces, one cleaner spine.
P1 (the ledger spine, #54) is **built + tested** (`gomoku/lab/ledger.py`); the rest
is the phased plan below.

## Thesis (the ~80%-there finding)

The repo already has every load-bearing mechanism:

- a resumable, wall-capped trainer — `train.py --max-wall-secs` self-caps at the
  epoch boundary and force-saves a **buffer-embedded** `latest.pt`; `--resume`
  reloads model+optimizer+EMA+buffer+wandb-run-id with **no cold refill** (the
  LF1 trap is already avoided);
- a proven priority chunk loop (`delo_derby.pick_priority`: Δelo/hr peak-progress
  with a starvation floor);
- an append-only JSONL writer (`scripts/lab_log.py`);
- a mac-native Gomocup arena (`gomocup_brain.py` + `external_engine.py` + the
  native Rapfi-NNUE anchor — **no wine**);
- a HuggingFace push (`gomoku/hf.py`);
- a calibration-immune 3-way gate (`scripts/sliding_gate.py`);
- the worker tier (`gh_worktree.py`, `gh_prime.sh`) — #2 phase 1 DONE.

**The one missing piece is the SPINE:** there is no single external, out-of-git,
append-only, financial-correction ledger that all loops read to pick first-priority
work and append results to. Today the "ledger" is smeared across five surfaces
(git-tracked `experiment-ledger.md`/`events.jsonl`, in-place-mutated
`derby_state.json`, per-version `derby_vN_board.json`, GitHub issues,
`verdicts.jsonl`). Build the spine first; then trainer/arena/research/worker
collapse into one shared shape. Everything else is wiring, not invention.

## The spine: the ledger (`gomoku/lab/ledger.py`)

### Where it lives — the `~/data/autolab/` home (out of git)

Everything autolab lives under **`~/data/autolab/`** (env `AUTOLAB_HOME`):

```
~/data/autolab/
  ledger.jsonl              # the spine (--ledger arg defaults here)
  runs/<lane>/sweep_runs/<cell>/checkpoints/{latest.pt, …}   # the DATA, incl. the buffer
  worktrees/<row_id>/       # ephemeral CODE checkout for one slice (removed after)
  daemon-<role>.lock        # the flock singleton + its metadata
```

It is **never git-tracked**, so worktree merges, branch switches, and
`reclaim_worktrees` can never clobber it — Jason's "flatfile so git doesn't stomp
it", literally. The Mac is the mainframe and 10 MB is nothing, so the reducer
reads the **whole file** every tick: no index, no DB, nothing to get stale.

The **`~/data` convention** is the answer to "we need to handle buffers": big
artifacts (the buffer-embedded `latest.pt`, worker records) live under
`~/data/autolab/runs/<lane>/` — local, fast on this machine, easy to inspect or
copy. HuggingFace only ever gets the *slimmed* per-slice weights; buffers never go
to HF. Config that benefits from review stays a small git-tracked file. This
resolves the doctrine tension head-on: *"git is the durability"* was right for
human receipts (the wiki), and wrong for a machine writing every hour.

The lane's run DATA is decoupled from the **ephemeral code worktree** the trainer
checks out per `commit`: `run_sweep` takes a backward-compatible
`--run-base`/`GOMOKU_RUN_DIR` (default `REPO_ROOT`, so the derby is untouched) and
the trainer points it at `~/data/autolab/runs/<lane>/`. Teardown deletes only the
code worktree; the checkpoints persist. That is what makes per-commit checkout
clean instead of a symlink dance.

### Row types (one JSON object per line)

| type | role | key fields |
|---|---|---|
| `experiment` | a unit of work | `id, role, commit, base, config, priority, status, note` |
| `claim` | a lease on work | `ref, by, lease_until` |
| `result` | outcome of work | `ref, status, model, buffer, metrics, wall_s, error` |
| `correction` | supersede fields | `ref, set{}, reason` ← the financial-journal move |
| `eval` | an arena report | `ref, model, panel, metrics` |
| `verdict` | a gate decision | `ref, gate, win_rate, ci, n` |
| `event` | log-only note | `scope, summary, data` |

`base` is the "model to start from": `scratch`, an HF revision (`hf://…@rev`), or a
local `latest.pt`. `commit` is the "commit to run". Every appended row gets an
auto-assigned monotonic `seq` (its file ordinal, assigned under an flock) + `ts`.

### Append-only, corrected like financial transactions

`append()` is flock-guarded and `fsync`s; it **never edits or deletes** a prior
line. To change anything, append a `correction` row that supersedes fields of a
prior entity by `ref` (last-writer-wins in the reducer). Setting
`{"status": "open"}` via a correction is how a failed lane is reopened (e.g. by
the research loop). The history *is* the audit trail.

### The reducer + priority pick

`fold(read_all(path)) → LedgerState` replays every row in order into current
state. `LedgerState.pick(role, now, priority_fn=None)` returns the first-priority
**open** item for a role; the default policy is `priority` desc → never-*completed*
(starvation floor, keyed on `_results`) → oldest (FIFO). Passing a different
`priority_fn` is the clean seam for porting `delo_derby.pick_priority` (Δelo/hr) in
P5 once eval rows feed it.

## The shared loop (trainer + arena are the same shape)

**No `claim` rows.** Mutual exclusion is the OS flock (auto-frees on death);
recovery is just *re-pick* (the in-flight lane's progress is safe in its
`latest.pt`). The flocked lockfile carries `{pid, role, item, started_at}` so
`autolab status` is legible without a claim. (Implemented in `gomoku/lab/daemon.py`,
P2.)

```
acquire flock singleton (role)           # exit if another daemon holds it
loop until stop:
  state = ledger.fold(ledger.read_all(path))
  item  = state.pick(my_role, now)       # first-priority OPEN for my role
  if not item: write lockfile{idle}; sleep; continue
  write lockfile{item, started_at}
  preflight()                            # foreign-tenant guard (PreflightDeferred → re-poll)
  try:
    res = run_chunk(item)                # checkout commit → run_sweep → HF → ChunkResult
    ledger.append(result(item, res.artifact_ref, res.metrics))
    for f in res.followups: ledger.append(f)   # flywheel: continuation + arena eval
  except: ledger.append(result(item, status=FAILED, error=…))   # terminal unless reopened
```

Trainer and arena differ only in `run_chunk()`. A finished training slice
auto-enqueues its continuation **and** an arena eval — the flywheel — which is why
**one shared ledger** beats one-per-role: the arena picks up the trainer's enqueued
eval rows with zero glue, and the cockpit is a single `fold()`. On a crash the
flock auto-frees and a fresh daemon re-picks the now-first-priority item; nothing to
reclaim.

## Code-shape contract (so a 1h slice never wastes a cycle)

The trained code is already mostly shaped for this; the contract makes it explicit:

1. **1h hard cap, no minimum** — `--max-wall-secs min(config, 3600)`; the MVP uses
   a **1-epoch cap** to prove plumbing in seconds. Keep the cap well above one
   epoch's wall so a slice never produces zero epochs.
2. **Bounded epoch wall is mandatory** so the epoch-boundary cap lands near the
   deadline (`--max-sgd-steps-per-epoch`/fixed steps). At 9×9 (~5s epochs) the
   overshoot is seconds = the accepted cycle-waste. Sub-epoch checkpointing is a
   15×15-era concern (#21), not now.
3. **Force-save a fully-resumable, buffer-embedded `latest.pt`** at every
   cap/SIGTERM (already `train.py._save_resumable`). Resume never cold-refills the
   buffer. Only the HF-delivered copy is slimmed (`hf.py` drops optimizer+buffer).
4. **Resume from an arbitrary model** — local `latest.pt` is a true resume; a
   resume-from-HF shim is a **warm-start** (the HF copy has no optimizer/buffer →
   re-warms the buffer = permitted cycle-waste). Prefer local for same-lane
   continuation; reserve HF-base for genuinely new lanes.
5. **Embed provenance** into the checkpoint (`git_sha`, `recipe_flags`,
   `ledger_row_id`, `parent`, Δelo) so resume is self-checking.
6. **Idempotent restart = re-pick** (no claim rows) — a crash leaves the lane
   `open` (no result was appended), so a fresh daemon simply re-picks the
   now-first-priority item and resumes from the lane's last `latest.pt`. Nothing
   to reclaim; no orphaned claim to clean up.
7. **Guaranteed singleton** via an OS `flock` that **auto-releases on death** (the
   fd closes) — so a SIGKILL needs no stale-lock cleanup, the failure mode a
   PID-file has. `FD_CLOEXEC` keeps `run_chunk`'s subprocesses from inheriting and
   pinning the lock. Preflight for foreign-MPS tenants before any GPU dispatch
   (`PreflightDeferred` → log an event + re-poll, never FAIL the lane) — *constantly*
   means "whenever the box is the lab's", not "unconditionally".
8. **Clean exit only** — every slice exits via the wandb-clean path, never
   SIGKILL, or it poisons the run id for the next `--resume`.

## Arena (mac-native, the trainer's twin) — built (P4 #59)

Same daemon shape. `ArenaRole.run_chunk(item)` resolves the candidate model
(`item.base`, HF revision or local), resolves the **current champion from the HF
`champion` tag** (`hf_hub_download(..., revision="champion")`; absent → the first
candidate seeds the ladder), and calls `scripts/sliding_gate.run_gate(...,
dry_run=True)` head-to-head vs the champion. `dry_run=True` is deliberate: we want
the gate's calibration-immune **PROMOTE / REVERT / AMBIGUOUS** verdict + its
verdict-log, but the champion is an **HF tag**, not `run_gate`'s local peer file —
so the arena does promotion itself by **moving the `champion` tag** to the
candidate's revision on PROMOTE. It appends `eval` + `verdict` rows (shared `ref`).
A live trainer slice (`probe_alive("train")`) **shrinks `n_games`** — the
co-tenancy guard. `run_gate`'s `eval_fn` is injectable, so the whole role is
GPU-free testable.

The richer **native Rapfi-NNUE panel** (logged-but-not-gated, via
`eval_vs_rapfi.run_rapfi_eval` + `gomocup_brain.py` with **`incremental=1`** — board
re-dumps crater the history-conditioned net to ~25% OOD) is a follow-up on top of
the H2H gate. The live gate proof (candidate vs a real champion) is deferred until
the box is free + a real model exists.

> **Operating contract (P5–P7, shipped in [#64]).** This page is the *design*;
> the *running* unattended-overnight contract — the four launchd jobs, the seed
> row, the monitor digest, and the research-lite priority invariant, all with
> absolute paths and literal config — lives in
> [autolab-supervisor-and-monitor.md](autolab-supervisor-and-monitor.md). Read
> that page to bring the lab up / down and read the morning digest.

## Research + worker lanes (the agent loops)

- **Research** — not a daemon; a Claude workflow on a long `ScheduleWakeup`/cron.
  Reads the folded ledger + W&B + HF, ideates, appends `experiment` rows **with
  priorities** (priority is how ideation steers the singleton), then **waits
  hours**, wakes, reads new `result` rows, writes wiki/issues, enqueues
  follow-ups. Until the wall-clock-to-elo gate (P5) exists it must declare it
  ranks on proxies it knows lie (the LF1 +152% throughput-runaway is the warning).
- **Worker** — GitHub issues stay the human-facing intake; the existing
  `gh_worktree.py`/`gh_prime.sh` flow pulls ready issues into worktrees, merges
  `--no-ff`, pushes. Worker claim/close mirror into the ledger so the cockpit sees
  all four loops in one stream.

## M5 saturation — the measured reality

The optimistic "pipeline GPU+ANE+AMX simultaneously" table is **partly falsified**:
the cross-engine receipts show all three NN engines draw from **one package power
budget**; **one heavy SGD trainer max** (a second nets −9%); self-play *generation*
has ~8% occupancy slack. So the singleton trainer is the official GPU tenant
(which *resolves* "don't compete with live tenants"), and the arena runs
**concurrently under a co-tenancy guard** (n_games auto-shrinks while a slice is
live). The concurrent path is **unvalidated at 15×15** — measuring it is a P4 task;
until then, conservative = arena takes idle-GPU gaps.

## Cockpit overlay (doctrine: more autopilot needs more cockpit)

Each loop ships three instruments before it runs unattended:

- **Gate** — trainer+arena share `sliding_gate.py` (3-way PROMOTE/REVERT/AMBIGUOUS,
  Wilson CI vs 0.5, never act when CI < δ; first promotion human-gated). Worker
  gates on the fresh-per-issue Reviewer. Research gates on the R-ELO/MTTE
  adjudication (P5).
- **One-glance status** — the unified ledger **is** the dispatch surface: an
  `autolab status` reader folds the flatfile into a per-lane board (lane / elo+CI /
  Δelo/hr / chunks-since-peak / last-verdict / HF-rev / owning-loop / PID-alive).
  Wire into `gh_prime.sh` SessionStart. The wiki stays the synthesis rollup on top.
- **Escalation** — the existing `human-gated`/`deferred` labels + a `needs:jason`
  status that floats to the top of `autolab status`.

## Locked decisions (2026-06-18 / 19)

- Home: everything out-of-git under **`~/data/autolab/`** (`AUTOLAB_HOME`), `--ledger`
  arg. Cross-machine fork (→ HF dataset repo + leases) deferred — single-machine
  mainframe for now.
- **No claim/lease rows.** Singleton = flock (auto-frees on death); recovery =
  re-pick. (The `claim` primitive stays in the ledger lib unused.)
- Buffer stays **local** in `~/data` (`latest.pt` = true resume); HF carries slimmed
  weights only.
- Data↔code decoupled: `run_sweep --run-base`/`GOMOKU_RUN_DIR` (default `REPO_ROOT`);
  per-commit checkout into an ephemeral code worktree.
- Arena: **concurrent + co-tenancy guard**; measure concurrent at 15×15 as P4.
- HF delivery: **per-slice revision + a moving `champion` tag** (arena bumps on
  PROMOTE).
- Priority: keep Δelo/hr `pick_priority` + starvation floor (port in P5).
- 1h hard cap is production; **MVP uses a 1-epoch cap**.

## Phased plan

| Phase | Issue | Goal | Status |
|---|---|---|---|
| **P1** spine | #54 | ledger reducer + corrections + priority pick + tests | **DONE** |
| **P2** daemon | #56 | flock singleton (no-claim re-pick) + `run_daemon` + `autolab status` | **DONE** |
| **P3** trainer | #57 | trainer role + `run_sweep --run-base` + `hf.push_slice` + 1-epoch proof | **CODE DONE** (ran once attended on 2026-06-19; artifacts since cleaned — `~/data/autolab` empty, HF mvp revisions gone — see [#58]) |
| **P4** arena | #59 | `ArenaRole`: `run_gate` (dry_run) vs the HF champion + `eval`/`verdict` rows + champion-tag bump + co-tenancy guard | **DONE** (live gate proof deferred — needs real models) |
| **P5** research-lite | #61 (partial) | deterministic `gomoku/lab/research.py` ideate→append-≤2-rows-below-seed→note loop (proxy-ranked; anchored gate still unbuilt) | **SHIPPED in [#64]** (pending live launch) |
| **P6** cockpit / monitor | #64 | `scripts/autolab_monitor.py` digest + `gomoku/lab/status.py` lane board + notify-on-change | **SHIPPED in [#64]** (pending live launch) |
| **P7** supervisor | #64 | `gomoku/lab/up.py` (up/down/status/restart) + four launchd plists + ledger seed | **SHIPPED in [#64]** (pending live launch) |

## Contradictions & risks (the honest tensions)

- **Ledger-outside-git** means it is not versioned/reviewable and not shared by git
  across machines — correct for the single-machine mainframe today; the
  cross-machine fork is an HF dataset repo.
- **Singleton-always-running** vs don't-compete: flock guarantees one trainer but
  cannot see a foreign MPS tenant Jason starts in his IDE → preflight politeness
  before each slice.
- **1h cap rounds up at the epoch boundary** — fine at 9×9, real overshoot at
  15×15 (minutes-long epochs) → sub-epoch checkpointing deferred to #21.
- **Resume-from-HF is a warm-start, not a true resume** — pay one buffer re-warm
  per new lane; keep same-lane continuation on local `latest.pt`.
- **Four loops, one GPU** under an envelope where the dual-full-1.5M-buffer memory
  regime (~16GB) was never measured — arena concurrency is an empirical bet that
  needs a measurement before the loop trusts it.

## Cross-refs

- [cockpit-vs-autopilot.md](cockpit-vs-autopilot.md) — the acceptance test for
  every new loop.
- [m5-max-as-mainframe.md](m5-max-as-mainframe.md) +
  [m5-max-cross-engine-coupling.md](m5-max-cross-engine-coupling.md) — the
  saturation philosophy and its measured envelope.
- [research-lab-charter.md](research-lab-charter.md) — the two-queue scheduler the
  autolab generalizes; the slice contract and priority function.
- [sliding-derby-design.md](sliding-derby-design.md) +
  [engine-panel-derby-design.md](engine-panel-derby-design.md) — the arena's gate
  and panel lineage.
- [wall-clock-to-elo-metric.md](wall-clock-to-elo-metric.md) — the P5 research gate
  (R-ELO/MTTE).
- Issues: [#53] epic · [#54] P1 spine · #2/#19 (superseded framing) · #21 (15×15
  sub-epoch).
