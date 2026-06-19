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

### Where it lives — out of git

Default a flatfile under `~/code` (e.g. `~/code/gomoku-autolab/ledger.jsonl`),
**configurable as the `--ledger` argument** to every loop. It is **never
git-tracked**, so worktree merges, branch switches, and `reclaim_worktrees` can
never clobber it — Jason's "flatfile so git doesn't stomp it", literally. The Mac
is the mainframe and 10 MB is nothing, so the reducer reads the **whole file**
every tick: no index, no DB, nothing to get stale.

Config that benefits from review (slice cap, panel, patience) stays a small
git-tracked file. Only the high-frequency work+result stream goes out-of-tree.
This resolves the doctrine tension head-on: *"git is the durability"* was right
for human receipts (the wiki), and wrong for a machine writing every hour.

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
`{"status": "open"}` via a correction is how a crashed/dead-lease lane is
reclaimed. The history *is* the audit trail.

### The reducer + priority pick

`fold(read_all(path)) → LedgerState` replays every row in order into current
state. `LedgerState.claimable(role, now)` returns work a role may take — `open`,
or `claimed` with an **expired lease and no result** (crash recovery).
`pick(role, now, priority_fn=None)` returns the first-priority claimable item; the
default policy is `priority` desc → never-attempted (starvation floor) → oldest
(FIFO). Passing a different `priority_fn` is the clean seam for porting
`delo_derby.pick_priority` (Δelo/hr) in P5 once eval rows feed it.

## The shared loop (trainer + arena are the same shape)

```
loop forever (holding a singleton flock):
  rows  = ledger.read_all(path)            # whole file; 10MB is nothing
  state = ledger.fold(rows)                # append-only → current view
  item  = state.pick(my_role, now)         # first-priority open/lease-expired
  if not item: sleep; continue
  ledger.append(claim(item, lease=slice_budget))   # singleton made legible
  workdir = git worktree-checkout <item.commit> into my own dir
  base    = hf.fetch(item.base)  if item.base != scratch else None
  artifact, metrics = run_chunk(workdir, base, item.config, deadline)
  ref     = hf.push(artifact, item.id)             # per-slice revision
  ledger.append(result(item, ref, metrics))
  ledger.append_followups(...)             # the flywheel: next slice + eval
  teardown worktree
```

Trainer and arena differ only in `run_chunk()` / `followups()`. A finished
training slice auto-enqueues its continuation **and** an arena eval — the flywheel
— which is why **one shared ledger** beats one-per-role: the arena watches the
trainer's `result` rows with zero glue, and the cockpit is a single `fold()`.

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
6. **Idempotent restart** — a crash after `claim` before `result` leaves an
   orphaned claim whose lease expires; the loop reclaims via a **correction**
   (never edits the claim row) and resumes from the lane's last `latest.pt`.
7. **Guaranteed singleton** via a real OS flock + PID file (not `pgrep`
   convention) that self-reclaims on a dead PID (else a SIGKILL deadlocks the
   lab). Preflight for foreign-MPS tenants before any GPU dispatch — *constantly*
   means "whenever the box is the lab's", not "unconditionally".
8. **Clean exit only** — every slice exits via the wandb-clean path, never
   SIGKILL, or it poisons the run id for the next `--resume`.

## Arena (mac-native, the trainer's twin)

Same loop. `run_chunk()` loads a model, plays a bounded match block against the
panel (champion + native Rapfi-NNUE @5s), computes a `sliding_gate.py` 3-way
verdict, appends `eval`+`verdict` rows. Reuses `gomocup_brain.py`,
`external_engine.py`, `match.py`/`eval.py`, `panel_tournament.py` (resumable —
skips completed pairs). **Net-as-engine must pass `incremental=1`** (board re-dumps
crater the history-conditioned net to ~25% OOD).

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

## Locked decisions (2026-06-18)

- Ledger: local flatfile in `~/code`, `--ledger` arg. Cross-machine fork (→ HF
  dataset repo) deferred — single-machine mainframe for now.
- Buffer stays **local** (`latest.pt` = true resume); HF carries slimmed weights.
- Arena: **concurrent + co-tenancy guard**; measure concurrent at 15×15 as P4.
- HF delivery: **per-slice revision + a moving `champion` tag** (arena bumps on
  PROMOTE).
- Priority: keep Δelo/hr `pick_priority` + starvation floor (port in P5).
- 1h hard cap is production; **MVP uses a 1-epoch cap**.

## Phased plan

| Phase | Issue | Goal | Status |
|---|---|---|---|
| **P1** spine | #54 | ledger reducer + corrections + priority pick + tests | **DONE** |
| P2 daemon | (file) | shared loop contract (flock+lease, claim/run/deliver/record) + `autolab status` | next |
| P3 trainer | (file) | wire ledger + HF push + per-slice checkout → **prove a 1-epoch slice end-to-end** | |
| P4 arena | (file) | gomocup_brain + panel + gate + HF-pull, co-tenancy guard | |
| P5 research | (file) | ideate→append→wait loop + the wall-clock-to-elo gate (`delta_e_harness` 5-gap list) | |
| P6 cockpit | (file) | status surface + escalation + retire the fragmented queue surfaces | |

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
