# Cockpit vs Autopilot

**Role:** The lab's operating lens for *when to build more autonomy vs. more
control*. It is a synthesis of Sid Bidasaria's **"Stop babysitting your agents"**
talk ([source](../sources/sid-bidasaria-stop-babysitting-agents-2026-05-20.md))
onto Jason's own situation: the machinery that runs without you (**autopilot**)
vs. the thin attention layer that makes that machinery *trustworthy and legible*
(**cockpit**). This page was referenced by `workflow-orchestration.md` and
`log.md` for three weeks before it existed — that absence was itself a small
instance of the gap it describes.

## The one thing to remember

> **More autopilot without more cockpit just gives you more to babysit.** The win
> isn't autonomy; it's *trustworthy* autonomy you can supervise at a glance.
> Attention is the scarce resource, not compute.

## Sid's stack: the three layers build on each other

The talk's load-bearing claim is the *ordering* — you cannot safely skip a rung:

1. **Verification first.** Teach the agent to check its own work — get it into a
   *loop* (write → build → run → check → fix → repeat until a success state).
   Package the loop as a **self-improving skill** that edits itself every time it
   hits a blocker, so the next run doesn't. Without this rung, nothing above it is
   safe.
2. **Then parallelize.** *Only once work is verifiable* can you run many agents at
   once and trust them. The hard limit is human: "more than four to five
   simultaneous sessions is a big load on my brain — **attention is scarce**." So
   the multi-agent tooling (desktop app, Agent View, remote control) is really
   *attention-protection*: it **sorts sessions by how much attention they need**,
   surfacing the blocked-on-input ones to the top.
3. **Then background loops.** Take your keyboard out of the hot path entirely —
   `/loop` and routines run the bookkeeping (PR babysitting, doc updates, triage,
   CI) that *must happen daily but doesn't need you in the loop, just in some
   loop.*

Stacked, you get "a system that does a lot of work without you on the keyboard."
The **autopilot** is rungs 2–3 (parallel agents, background loops); the
**cockpit** is rung 1 plus the attention-sorting that makes 2–3 supervisable.

## The lens, in lab vocabulary

Jason built the autopilot well and early — `gpu_daemon`, the Δelo derby,
`derby_watchdog.sh`, `run_sweep`'s self-capping slices, the re-invocation crons.
The persistent *gap* has been the cockpit: the attention layer. Three concrete
cockpit instruments, and where the lab stands on each:

| Cockpit instrument | What it is | Lab instantiation |
|---|---|---|
| **A trustworthy gate** | verify *before* you parallelize or promote; never act on noise (`CI < delta`) | the frozen-reference [`sliding_gate.py`](../../scripts/sliding_gate.py) (Wilson CI + an **AMBIGUOUS band** = "can't tell, don't act"); the Reviewer-gated fan-out; the measured-outcome **title card** (pre-stated confirm/refute) |
| **A one-glance status surface** | see all running work + what needs you, without reading logs | the [lab event log / dashboard](event-log.md) as a *log viewer*; the derby board (`composite_derby_board.jsonl`); `gh_prime.sh`'s ready-queue |
| **An escalation "needs you" line** | the few things that genuinely require a human, surfaced and *only* those | the `human-gated` / `deferred` labels; "no gate only try" routes everything else away from Jason |

**The judging question for any new lab infra:** *does this strengthen the cockpit
(trust, legibility, attention-economy) or just add more autopilot (autonomous
capability)?* Capability you can't supervise at a glance is a liability — it is
exactly the thing you end up babysitting.

## Why verification is the keystone (and the gate's design follows from it)

Sid's ordering explains a lab rule that otherwise looks like mere caution:
**short evals are a hint, not a verdict** ([conventions](conventions.md)). If you
parallelize or promote on a noisy signal, parallelism *multiplies* the error
faster than you can catch it — the cockpit fails silently. So the gate is
deliberately **calibration-immune** (anchor-free H2H win-rate, never reads a
broken absolute Elo) and **three-way** (PROMOTE / REVERT / AMBIGUOUS): it refuses
to turn a coin-flip into a decision. That `CI < delta` discipline *is* the "verify
before you parallelize" rung made into structure rather than vigilance.

## Today's instantiation: the workflow-master IS the cockpit

The [workflow-master operating model](workflow-orchestration.md#the-orchestrator-sessions-discipline-the-workflow-master)
is this page applied to a single session: the orchestrator *runs and tunes the
workflows* (autopilot) while holding only the decision thread and fanning out
everything context-heavy (cockpit — protecting its own scarce attention/context).
A Workflow whose second stage is the Reviewer makes the verify-gate
**non-skippable structure** instead of skippable discipline — the autopilot→cockpit
move in one primitive (see [workflow-orchestration](workflow-orchestration.md)).

## Related

- Source: [Sid Bidasaria — Stop babysitting your agents](../sources/sid-bidasaria-stop-babysitting-agents-2026-05-20.md)
- [workflow-orchestration](workflow-orchestration.md) — Workflows as the cockpit primitive; the workflow-master session discipline.
- [research-lab-charter](research-lab-charter.md) — the two-queue scheduler + triage/escalation matrix (the cockpit's routing rules).
- [event-log](event-log.md) — the status surface as a log viewer (record ≠ report).
