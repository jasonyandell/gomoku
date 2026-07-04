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

## 7. Lean all the way into uv (the execution leg of §2)

`uv` is the execution leg of §2: it *focuses* the trainer down to "translate a
ledger row into a `uv run` against a ref, record what came back" — deleting the
bespoke checkout / teardown / reclaim subsystem and the shared-editable-install
gotcha (a worktree slice otherwise runs the commit's `scripts/` against **main's**
`gomoku` package, so "a commit" isn't really a commit). Going uv-native for
*mainline dev too* is **negative scope creep**: each tree gets its own `uv sync`'d
locked env, so the [editable-install worktree gotcha](../../README.md) becomes
*structurally impossible* and **dev env ≡ lab env ≡ CI env** collapse to one
lockfile — one mechanism for "run code here."

**Built + validated (2026-06-19):** the trainer's execution leg now *is* uv —
`_checkout` does `git archive <commit> | tar -x` (a standalone snapshot, no shared
`.git`) and `_run_slice` runs `uv run python scripts/run_sweep.py …` inside that
tree, building the commit's own `gomoku` (deps + C extensions reconstructed from the
ref), never main's editable install. The `git worktree` subsystem is deleted;
teardown is a plain `rmtree`. Proven by a real 1-epoch smoke slice (fresh uv env + C
build → trained, self-capped 90 s, `model_elo≈389`, flywheel follow-ups). The
native-`.so` question was scouted clean — the 4 pinned per-SHA `.c` sources compile
via the standard PEP 517 path `uv sync` already drives (reproducible native code
per commit, no custom build hook). **Direction, not built:** a committed `uv.lock`
(reproducible *deps* per SHA), per-SHA env caching, and uv-native mainline dev.

## Status — doctrine realized (2026-07-04)

The stateless-reducer shape now covers all three lanes (trainer/arena via
`daemon.run_daemon`; research via `research.resume`), and the loop **simulator**
(`tests/test_lab_sim.py` + `gomoku/lab/health.py`) certifies the walls — every
hardening fix falsified RED-when-off. `gomoku/lab/actionable.py` is the unified
`actionable()` read-surface (§3); research resume-on-evidence (§4) ships with the
pluggable `decide` seam; the uv execution leg (§7) is built. **Not yet:** promoting
`resume` to a first-class trigger (cron ≡ event ≡ manual over `actionable`, §5),
`actionable.worker` as a lane, and the uv-native follow-ups of §7. Per-phase build
state lives on [autolab-architecture.md](autolab-architecture.md); this list is
only what bears on the doctrine.

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
