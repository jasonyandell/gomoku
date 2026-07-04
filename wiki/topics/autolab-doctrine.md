# The Autolab Doctrine — hard walls around a sandbox

> **Status: LIVE** *(2026-07-04)* — the era-independent thesis under the
> [Autolab hub](../autolab.md); the *why* that decides whether a change belongs.

**What this is.** The *why* under the autolab, distilled in a vision session
(2026-06, weekend). [autolab-architecture.md](autolab-architecture.md) is the
*what* (the pieces); this is the *thesis* that makes the pieces cohere and tells
you whether a proposed change belongs. One sentence:

> **The lab is a pure function from an append-only log to a set of actionable
> items, handed to stateless workers — the dumb ones (trainer/arena) and the
> smart one (Claude) — inside walls a simulator proves hold.**

Hard walls, simple hardened tools, no state anywhere, the big brain on tap
exactly when evidence demands it.

## 1. Hard walls around a sandbox

The lab is **hard walls around a sandbox.** The walls are the contract:
append-only ledger (can't corrupt history), `flock` singleton (can't
double-run), the 1h hard cap (can't monopolize the GPU), board-size as a
process-start constant (can't cross eras), reproducible execution (can't run
un-pinned code), and the load-bearing one — **no state in a process.** The
sandbox is everything the Claude researcher and worker get to *try* inside those
walls.

The hardening is not bureaucracy. **You harden the walls precisely so the sandbox
can be wild.** A deterministic, predictable substrate is what earns the right to
drop a *non-deterministic* LLM into the middle of it: nothing it does can breach
the walls. This is the real reason the loop-hardening pass matters — see §6.

## 2. The core doctrine: no state in a process

Jason's design lens (memory: `feedback-stateless-delegate-design`): **state never
lives in a process. It lives in a durable append-only log, and every component is
a stateless reducer that reconstructs what it needs on demand.** Corollary:
**delegate the generic, hard, solved sub-problem to a battle-tested tool; keep
only the domain glue.** The lab applies this same move four times:

| Generic hard problem | Hand-rolled (rejected) | Delegated to |
|---|---|---|
| mutual exclusion | a lease/claim manager | OS `flock` (auto-frees on death) |
| durable state | a database | one append-only JSONL flatfile (`ledger.py`) |
| artifact storage | a model registry | HuggingFace (slimmed weights + a `champion` tag) |
| **code execution / envs** | worktree + editable-install dance | **`uv`** (see §5) |

Every component is the same shape: **fold the durable log → do the next right
thing → append.** The trainer and arena already are (`daemon.run_daemon`); §3–§4
extend it to research, and §5 extends it to *execution itself* (the env is
reconstructed from a ref, not persisted).

## 3. Deterministic WHEN, intelligent WHAT

The substrate's only job is to answer, deterministically and cheaply, **"is there
something to do, and what is available?"** That is a *pure function over the
folded ledger*:

```
actionable(ledger) → { trainer:    [...open commits-to-run],
                        arena:      [...open evals],
                        researcher: [...threads whose evidence just landed],
                        worker:     [...open issues] }
```

`gomoku/lab/health.py` `scan(state)` was the seed; **`gomoku/lab/actionable.py`
`actionable(state)` is now that function** — one pure fold bundling
`state.pick('train')`, `state.pick('arena')`, the research threads (§4), and
`scan`'s alerts into a single read-surface. It is built *on* `state.pick` (never a
second pick policy), so the monitor, a cron trigger, and the daemon can't drift —
the sim asserts `actionable(state).train` agrees with `state.pick('train')`. It
draws the clean line:

- **Substrate = WHEN + what's-available (determinism).** The dumb part is
  reliably dumb.
- **Claude = WHAT-to-do (judgment).** The smart part is invoked *only* at the
  moments judgment is actually required, and is otherwise not running. You never
  pay for the big brain to sit in a loop.

## 4. "Waits" is deleted — resume on evidence, not a blocked thread

The original sketch said the research lane "waits hours" for results. **Delete the
word.** A component that sleeps holding a thread open is holding *state in a
process* — the one thing the architecture refuses everywhere else. Decompose what
"wait" was standing in for, and none of it is a wait:

- *remember what you started* → an **experiment row in the ledger** (the intent is
  durable data; the process remembers nothing).
- *the information arrives* → a **result row gets appended** (an event, not a
  timer).
- *make an informed choice* → a **fresh invocation** that folds the ledger, finds
  threads whose question now has an answer but no follow-up decision, and advances
  them — full context reconstructed from the log.

So research is the **same read→pick→act→append shape** as the trainer; it differs
only in that its "actionable" is *an open question whose evidence has landed* and
its "act" is an LLM call (ideate/analyze/enqueue), not a GPU chunk. The Claude
researcher is therefore **itself stateless**: handed a fully-reconstructed thread
at the moment judgment is needed, it decides, appends, and is gone. The ledger is
its memory. No research process to manage, no thread to leak, no "wait" to get
wrong.

Where the reframe bites: **hardest on research** (long-latency, irregular
evidence — the only place a timed sleep was ever tempting), lightly on the arena,
**not at all on the trainer** (it never waits on anything external; it always has
a queue).

**Built:** `research.research_threads(state)` is the WHEN (lanes whose evidence
landed undecided); `research.resume(ledger, decide=…)` is the reducer — fold, find
those threads, decide each, append a `research-decision` event (+ followups like a
correction that *parks* a dead fork), idempotent on a re-fire. `decide` is the
pluggable WHAT: a dumb deterministic default (park-declining / keep / flag-rising)
ships so the loop self-runs, and Claude drops in as `decide=` with zero structural
change. The research tick now resumes-then-proposes; the sim drives a fork to
evidence and asserts decide-once-per-arrival + park-takes-effect.

> **2026-06-20 refinement — mechanical vs. *epistemic* WHEN.** A
> [deep literature pass](../sources/autolab-agentic-research-lessons-2026-06-20.md)
> caught that `research_threads()` is the **mechanical** WHEN ("a new terminal slice
> landed for this lane") — *not yet* the **epistemic** WHEN ("enough of the *right*
> evidence for *this* question arrived"). A hypothesis needing *2 slices **and** an
> arena verdict* would fire the instant the slices land, before the verdict exists.
> The fix is a per-proposal **evidence contract** + a real `decision_due()`, and a
> **typed-intent boundary** so Claude proposes *meaning* while only the substrate
> writes rows. That work — the load-bearing next step, ahead of any richer Claude
> conversation — has its own design note:
> [autolab-researcher-contract.md](autolab-researcher-contract.md) (it also adds the
> *three-zone* governance — science / adaptive policy / protected instrument — and an
> upgraded one-sentence thesis).

## 5. Triggers are pluggable *because* they are not load-bearing

You may have loops/cron/`ScheduleWakeup`, **but you do not build on them.** The
property that makes this safe: a trigger only ever does the identical thing — call
`actionable(ledger)` and hand the result to a worker. So cron ≡ an event on
result-append ≡ a file-watch on the ledger ≡ a human running a command. They are
interchangeable. **Kill the loop and you lose timeliness, not correctness** — the
next trigger of any kind reconciles from the log.

Consequence: the easy default is **pull-via-cadence** (a cheap tick that
re-scans), and it is fine *precisely because the scan is the truth, not the tick.*
Moving to push later (the appender of a result enqueues the actionable item) is a
drop-in over the same `actionable`. Be lazy about triggers forever.

## 6. Why the simulator certifies the walls

`tests/test_lab_sim.py` drives the **real** loop (`daemon`/`ledger`/`trainer`/
`arena`) with the ML replaced by `random()`, and throws chaos at it — crashes
mid-slice (a `BaseException` past `except Exception` = a true SIGKILL: lock
auto-frees, no result row), HF outages, era crossings, foreign GPU tenants —
asserting loop invariants every tick (no lost progress, no silent stall, bounded
worktrees, gated first-promotion, no cross-era contamination, deterministic
fold). Every fix is *falsified* (turned off, confirmed the test goes red).

This is the job the sim actually does: **it is not testing gomoku, it is
certifying the cage is LLM-proof.** Which is the license you need to "plug the
Claude researcher in and let it try things." The invariants are about *the loop*,
not *how code lands on disk* — so a substrate swap (e.g. worktree → `uv`, §5/§
execution) is cheap to try: make the swap behind the `run_chunk` seam, run the
sim, watch it stay green.

## 5b. Lean all the way into uv (trainer BUILT + validated; mainline dev still direction)

`uv` is the execution leg of §2. For the autolab it *focuses* the trainer down to
"translate a ledger row into a `uv run` against a ref, record what came back" —
deleting the bespoke checkout / teardown / reclaim subsystem and the
shared-editable-install gotcha (a worktree slice today runs the commit's
`scripts/` against **main's** `gomoku` package — the per-commit checkout is partly
fiction; `uv` makes "a commit" actually mean a commit).

**The scope-creep that simplifies more:** go uv-native for *mainline dev too*.
Each worktree/clone gets its own `uv sync`'d locked env; `uv run` runs that tree's
code against that tree's deps, so the
[editable-install worktree gotcha](../../README.md) (PEP-660 finder pointing
`gomoku` at the wrong checkout; PYTHONPATH can't shadow it) becomes
*structurally impossible*. This is **negative scope creep** — it deletes the
repoint recipe, a memory note, and per-session venv confusion, and unifies **dev
env ≡ lab env ≡ CI env** from one lockfile: one mechanism for "run code here."

The one open question — **how the native `.so` are produced** — was scouted
(2026-06-19) and the answer says **clean afternoon, not fiddly.** `setup.py`
declares **4 C extensions** (`gomoku/_state_ops_native{,15}.c`,
`gomoku/_mcts_native{,15}.c`) built by `setuptools.build_meta` at install time; the
`.c` sources are **committed**, the `.so` are not. So it's neither "committed
binaries to copy" nor "a manual build step" — it's the standard PEP 517 path `uv`
already drives: `uv sync` runs the build backend → compiles the `.c` → `.so`, **no
custom build hook needed.** Because the `.c` are pinned per-SHA, a `uv run` against
a ref builds the *exact* native code for that commit (reproducible-per-SHA, the
property we wanted — better than committing platform binaries), and uv's build
cache compiles each (SHA, platform) once. The only cost is a few seconds of C
compilation on a cold env. Verdict: **the uv swap is unblocked.**

**Built + validated (2026-06-19).** The trainer's execution leg now *is* uv:
`_checkout` does `git archive <commit> | tar -x` (a standalone snapshot, no shared
`.git`), and `_run_slice` runs `uv run python scripts/run_sweep.py …` *inside* that
tree — so the slice builds and runs the commit's own `gomoku` (deps + C extensions
reconstructed from the ref), never main's editable install. The `git worktree
add/remove/prune` subsystem is deleted; teardown is a plain `rmtree`. Proven by a
real 1-epoch SMOKE slice end-to-end (`git archive HEAD` → fresh `uv` env + C build
→ trained, self-capped at 90 s, final-eval `model_elo≈389`, `DONE` + flywheel
follow-ups), GPU-tenancy checked free first. **Remaining:** commit a `uv.lock`
(deps currently resolve from `pyproject` per run — reproducible *source* per SHA,
not yet reproducible *deps*); optionally cache the per-SHA env instead of
re-syncing each slice; and the bigger swing — go uv-native for mainline dev too.

## Status

- **Built (in `feat/autolab-sim`, unmerged):** the loop simulator
  (`tests/test_lab_sim.py`) + the shared detector (`gomoku/lab/health.py`), plus
  the hardening fixes the sim found (HF-push decouple, 1h-cap enforcement,
  first-promotion gate, era-namespaced champion tag, worktree self-prune,
  era-cross retire). **`gomoku/lab/actionable.py`** — the unified `actionable()`
  read-surface (the molecule over `state.pick` + research threads + `scan`),
  consumed by the monitor. **Research as resume-on-evidence** —
  `research.{research_threads,resume,default_decide}` with the pluggable `decide`
  seam, wired into the research tick (resume-then-propose). New sim invariants
  (`actionable`-vs-`pick` consistency, decide-once-per-arrival) + scenarios
  (`research_resume_on_evidence`, `research_park_takes_effect`), every fix
  falsified RED-when-off.
- **Doctrine, partially realized:** the stateless-reducer shape now covers all
  three lanes (trainer/arena via `daemon.run_daemon`; research via `resume`). What
  remains is the *cadence/trigger* wiring — `resume` runs inside the research tick
  today; promoting it to a first-class trigger (cron ≡ event ≡ manual over
  `actionable`, §5) is unbuilt. And `actionable.worker` (open GitHub issues as a
  lane) is not yet a field.
- **Built (uv execution leg, §5b):** the trainer runs slices via `git archive` +
  `uv run` against the ref — the worktree subsystem deleted, the editable-install
  gotcha gone, validated by a real 1-epoch slice. `test_lab_trainer` asserts the
  new mechanism (archive, not worktree; `uv run`, not the daemon's interpreter).
- **Direction, not built:** a committed `uv.lock` (reproducible deps per SHA);
  per-SHA env caching; uv-native mainline dev (the editable-install gotcha gone for
  *every* worktree/clone, one mechanism for "run code here").

## Cross-refs

- [autolab-researcher-contract.md](autolab-researcher-contract.md) — the smart lane's
  I/O contract (evidence contract · typed-intent wall · continuation policy · three-zone
  governance), the #61 design note that realizes the "intelligent WHAT."
- [../sources/autolab-agentic-research-lessons-2026-06-20.md](../sources/autolab-agentic-research-lessons-2026-06-20.md)
  — the agentic-research literature pass these refinements mine.
- [autolab-architecture.md](autolab-architecture.md) — the pieces (the *what*).
- [autolab-supervisor-and-monitor.md](autolab-supervisor-and-monitor.md) — the
  running operating contract.
- [cockpit-vs-autopilot.md](cockpit-vs-autopilot.md) — the acceptance test for any
  new loop (gate · status · escalation); the cockpit is the thin supervisable
  layer over the autopilot these walls make safe.
- [conventions.md](conventions.md) — the cross-cutting project conventions.
- memory: `feedback-stateless-delegate-design` (the cross-project design lens).
