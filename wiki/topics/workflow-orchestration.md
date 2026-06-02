# Claude Code Workflows in the Lab

**Role:** How the Claude Code *Workflow* feature maps onto this lab's
orchestration, where it fits, and — just as important — where it does **not**.
The first real workflow lives at
[`.claude/workflows/reviewer-gated-fanout.js`](../../.claude/workflows/reviewer-gated-fanout.js).

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
[worktree-hygiene](worktree-hygiene.md) janitor exists to clean up after.

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
