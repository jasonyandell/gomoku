# Claude Code Workflows in the Lab

**Role:** How the Claude Code *Workflow* feature maps onto this lab's
orchestration, where it fits, and — just as important — where it does **not**.
The first real workflow lives at
[`.claude/workflows/reviewer-gated-fanout.js`](../../.claude/workflows/reviewer-gated-fanout.js).

> **Historical / two explored approaches.** This Claude-workflow composite and the
> launchd-daemon autolab ([autolab-architecture.md](autolab-architecture.md)) are
> **two explored design approaches to the same autonomous-derby goal** — neither
> supersedes the other. The autonomous derby is **stopped** (see
> [derby.md](../derby.md) for status); both remain as design records.

## The one thing to remember

> **Workflow is the deterministic encoding of the *everything-else* lane of the
> two-queue scheduler. It never touches the GPU lane.**

A Workflow is a JavaScript script you hand to one tool call. It runs in the
background and orchestrates **subagents** (LLM reasoning agents) with ordinary
control flow — loops, conditionals, fan-out — that is *deterministic JS*, not
model-decided prose.

## It is deterministic chaining, not the looper

The easy confusion: this is **not `/loop`**. `/loop` re-invokes the model on a
timer (the looper — good for "keep checking until X", overnight ticks, the
derby-runner cadence). A Workflow is **deterministic chaining**: a fixed script
says *agent A's output feeds agent B, fan these five out at once, retry that one
until it passes* — and the JS, not a model's judgement, owns that wiring. You
reach for the looper when *when-to-act* is the question; you reach for Workflow
when *how-to-route-the-agents* is the question and you want it the same every
time.

Primitives, in lab vocabulary:

| Primitive | Lab analogue |
|---|---|
| `agent(prompt, {schema, isolation:'worktree'})` | one subagent; `schema` = a structured verdict (the Reviewer's `VERDICT/ONE-LINE/DETAILS` block); `isolation:'worktree'` = worktree-per-unit-of-work, enforced |
| `parallel([...])` | "fan out ≥2 in one message" — but a barrier, awaits all |
| `pipeline(items, s1, s2…)` | no barrier: lane A hits the Reviewer while lane B is still building. The Reviewer-per-lane shape |
| loop-until / judge / adversarial-verify | the `REVISE → re-spawn → APPROVE` retry; the round-robin H2H verdict |

## The dividing line: GPU lane vs everything-else lane

The lab is already split by **hardware**. That split is exactly the fit boundary.

**MISFIT — do not use Workflow here.** Workflow agents reason; they are *not* MPS
processes, cannot hold the serial GPU lock, and do not persist across sessions:

- The GPU serial queue — `delo_derby.py`, `run_sweep.py`, eval probes
  (`probe_100pct.py`). Already deterministic Python that holds the MPS lock.
- The derby-runner cron + `derby_watchdog.sh`. Cross-session daemons; Workflow
  is per-invocation.
- The Jason-gate (`deferred` label). Human-gated by design.
- derby-register's single config/code fork. One `if` — orchestration overkill.

**FIT — these are prose today and want to be control flow:**

| Surface | Pattern | Status |
|---|---|---|
| research-lab everything-else fan-out + Reviewer gate | `pipeline(implement → review)` + worktree isolation + schema verdict + loop-until-APPROVE | **Built:** `reviewer-gated-fanout.js` |
| bead/issue runner (claim → dispatch → verify → mirror) | `parallel` over claimed issues, worktree-per-issue, schema DECLINE-routing | Candidate (next) |
| fan-out-to-preserve-context convention | literally `parallel(agent())` + worktree + schema | The most Workflow-shaped idea in the wiki |
| multi-lever run *design* (train/research-lab) | group-by-file `parallel`, integrate | Builds code, never touches GPU |

## Why it matters: this is the cockpit, not a speedup

The recurring friction — *missed Reviewer audits, stale-base worktrees, a worker
editing shared `main`* — is not a knowledge gap. It is **discipline that lives in
prose and is therefore skippable.** Workflow's value here is not wall-clock; it
is that a `pipeline` whose second stage is the Reviewer means **no lane reaches
"promote" without an audit** — the verify-gate *is* the control flow. That is the
autopilot → cockpit move (see [`cockpit-vs-autopilot`](cockpit-vs-autopilot.md)):
turning skippable judgement into non-skippable structure. Worktree isolation per
agent structurally prevents the edit-shared-`main` failure mode that the
[worktree-hygiene](worktree-hygiene.md) cleanup existed to mop up after (the
auto-janitor `reclaim_worktrees.py` is **retired** as of 2026-07-01 — cleanup is
manual, ps-check-then-remove).

## The orchestrator session's discipline (the workflow-master)

The cockpit move has a mirror image: if a Workflow turns skippable judgement into
non-skippable structure, the **orchestrator session** must keep itself lean enough
to *run* that structure indefinitely. The failure mode this prevents is concrete
and was paid for: a session that holds all the work **in its own head** — building
a parallel apparatus, debugging inline, growing a deep narrative — hits the
context wall (~400k) and degrades into an instance attending poorly over its own
history. That is the autopilot-eats-the-cockpit failure.

> **The workflow-master runs and tunes the workflows; the *workflows* do the
> work.** The session holds the decision thread (which hypothesis is top
> priority, the verdicts, what to tune next) and **fans out everything
> context-heavy** — broad reads, multi-file builds, scoring, GPU slices — to
> subagents and workflows, keeping the conclusion, not the file dumps.

Two verbs, concretely:

- **Run** = kick a workflow and *watch*, not do. The GPU train-cycle
  (`sliding-derby-composite`) and the code/everything-else drain
  (`implement-backlog`) are the two standing lanes; the
  [research-lab-charter](research-lab-charter.md) triage matrix is the concrete
  rule for *what* each lane may touch.
- **Tune** = between invocations, edit the workflow `.js` from friction. The
  workflow file is the instrument; the master sharpens it rather than *becoming*
  it. The self-check: catching yourself hand-coding or debugging *inline* for more
  than a beat is the smell of holding work that should have been delegated.

**An empty drain is a real signal, not a failure.** (Evidence, 2026-06-16: an
`implement-backlog` run triaged all 15 ready issues and picked *zero* — the
code-only backlog was drained; everything remaining was `needs-live-validation`,
and even the white-defense eval suite (#45) turned out to need a *measurement*
run to prove its positive control, so it isn't pure-code-auto-mergeable.) The
"code is free, fan it out" lane has a floor: **it cannot manufacture code work
that isn't there.** When the code drain returns empty, that *is* the finding — the
single GPU / measurement lane is the binding constraint, and the master's job
shifts from draining code to running bounded measured-outcome GPU cycles.

Because the session's lifetime is bounded (context, or a crash), durability is not
optional: **flush to the substrate before the session dies** — friction → skill,
findings → wiki, work-state → issues, tacit feel → handoff — so a fresh
workflow-master re-adopts running work from state and loses almost nothing. *The
skill is the memory; the session is a cache.*

## Resilience: the workflow degrades, it does not crash

The autopilot is only trustworthy if it fails *gracefully*. The subagents a
workflow orchestrates are remote LLM calls — they **will** die on a transient API
overload (a 529) or a user-skip, and `agent()` is documented to return `null` in
exactly that case (after its own internal retries). So a workflow's resilience is a
property of its **deterministic JS**, not of the agents. The failure that taught
this (2026-06-16, #50): an overload window killed `implement-backlog`'s triage
agent, the script did `triage.picks` on `null`, and the whole run aborted with a
`TypeError` — a network blip turned into a stack trace. Every workflow had the same
shape: a bare `await agent(...)` whose result was dereferenced unguarded.

The fix is two simple, deterministic layers (nothing fancy):

1. **Bounded re-spawn for idempotent chokepoints.** A tiny `agentTry(prompt, opts,
   tries=3)` re-calls `agent()` on `null` — a fresh spawn gets a fresh internal-retry
   budget, so a brief blip rides out. Use it **only** for side-effect-free agents
   (triage / review / score).
2. **Graceful degradation everywhere else.** A null result is converted into a clean
   structured outcome (`{aborted:true}`, a `BLOCK` verdict) plus a `log()` line —
   never a dereference. `parallel`/`pipeline` results are `.filter(Boolean)`-ed.

The load-bearing rule: **never retry a side-effectful agent.** The composite derby's
train-*launch* agent stays a single bare `agent()` — re-spawning it could
double-launch a detached GPU slice. It degrades and leans on the
re-invocation/re-adoption model instead (a clean abort loses nothing because a later
invocation re-adopts the running slice from state — see
[`workflow-harness-capabilities`](workflow-harness-capabilities.md)). This is the
same insight as "the session is a cache": **a clean exit is always safe when the
state lives on disk.**

The gauge for this class of entropy (per the lab's janitor+gauge rule) is
[`scripts/check_workflow_resilience.mjs`](../../scripts/check_workflow_resilience.mjs):
a no-network smoke test that runs every workflow under agent-death / happy /
stage2-death stubs and asserts no-throw. It is verified **red** on the un-hardened
code (`reading 'picks'`) and **green** after — so a future edit that reintroduces an
unguarded `agent()` deref is caught deterministically, without waiting for the next
real outage.

The gauge is also **self-firing** (#51, the janitor+gauge rule's second half): an
opt-in committed `pre-commit` hook (`scripts/hooks/pre-commit`) runs the checker
automatically — but **only** when the commit stages a top-level
`.claude/workflows/*.js`; any other commit short-circuits to exit 0 without
spawning node. Enable it per clone/worktree with `git config core.hooksPath
scripts/hooks` (it does not override anyone's existing hooks; see
[`scripts/hooks/README.md`](../../scripts/hooks/README.md)). A non-zero checker
exit blocks the commit; a missing `node` warns and lets it through.

## The honest boundary the workflow keeps

`reviewer-gated-fanout.js` deliberately **stops at "APPROVED, branch ready"**. The
serial `git merge --no-ff` and `git push` stay the operator's call (Class-B-
adjacent), so the workflow never touches shared `main`. APPROVE branches come back
merge-ready; BLOCK and revise-exhausted lanes come back flagged for a human.

## Opt-in

Workflow can spawn many agents and spend a lot of tokens. It is **opt-in**: an
`ultracode` session, an explicit "use a workflow", or a skill that invokes it.
Do not infer it for tasks that merely *would* benefit.

## Next candidates

- **Issue-runner workflow** — encode the bead-runner's poll → claim → dispatch-
  worktree-worker → verify → mirror loop. CPU-only by rule, so no GPU conflict.
- Wire `reviewer-gated-fanout` into the `gomoku-research-lab` skill so the
  everything-else queue stops being hand-coordinated.
