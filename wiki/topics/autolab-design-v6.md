# The Autolab — design v6

> **Status: WORKING DRAFT v6** *(2026-07-05)* — the standalone statement
> (Jason + Fable session). Supersedes prior walk-through drafts;
> [v1](autolab-primary-design.md) remains design-of-record until a vN is
> blessed. Substrate: [the runner](runner-spec.md). Red-team record:
> [A1–A26](autolab-design-adversarial-review.md) plus the 2026-07-05
> session exchange (unfiled).

## One breath

A laptop runs a research lab on itself. Truth lives in three adjacent
stores: the **client repo** (code + wiki), the **ledger repo** (one
append-only file per experiment, durable from its first row), and the
**runner root** (a work queue that is also its own history). All
execution flows through **the runner** — a boring utility with one
promise: *throw work at it and that work runs, eventually*, one item at
a time, in order. An experiment is a **slip**: a hypothesis stated
before any result exists, one composite run (smoke → train → eval), a
verdict, maybe a landing, always a lesson. Claude does the thinking in
two roles — a **researcher** who proposes and judges, a **curator** who
smelts every finished experiment into the wiki — spawned as agents by a
cron-shaped **orchestrator** that polls, folds, and decides. Stochastic
actors author; deterministic substrate mutates; a failed experiment is
ore, not waste. **The wiki is the product; the player is a byproduct.**

## The entities

| | Entity | Job |
|---|---|---|
| **G** | client repo | code + wiki — what runs and what's known; code advances only by apply, wiki only by curation |
| **ledger** | ledger repo | one append-only slip file per experiment, named by slip_id, durable from the first row; every row cites its evidence; the lab's only load-bearing state |
| **runner** | the substrate | one queue, one lane, an append-only log; owns order, exclusion, witness; knows tickets, never experiments |
| **orch** | the orchestrator | a Claude Code workflow on a cadence: poll the runner, fold the ledger, take the deterministic actions, spawn the agents; **all policy lives here**, and it holds no state — killed at any line, the next tick re-derives everything |
| **R** | researcher (agent) | stochastic proposer: mints, states the bet, authors in a throwaway scratch clone, pins a SHA, submits, judges; **never mutates canon** |
| **A** | applicator | a deterministic script submitted as a runner item: `--no-ff` merge + canon smoke, commit iff green |
| **C** | curator (agent) | single, serialized: consumes finished-but-uncurated slips → wiki prose + one lesson; **the closing transition** |

The champion is *not* an entity — it trains outside the lab, as a
sibling cron that appends one training slice whenever the queue is
empty. The lab never knows it exists; its evidence lives in W&B and the
training notebook.

## The slip id — the whole threading mechanism

```
within an experiment : one slip_id, one file in the ledger repo
                       → disjoint by uniqueness; appends never conflict
across experiments   : optional prev_id — an immutable chain; lineage,
                       retries, and challenges are all just links
retrieval            : slip_id IS the file's name → deterministic open;
                       no search, no snapshot-consistency question
runner run_ids       : <slip_id>-<label>-a<attempt> — deterministic, so
                       submission is idempotent; the fold joins tickets
                       to slips by name
ledger row ids       : {kind}:{slip_id}:{label}:{attempt} — the fold's
                       first-wins key; duplicates and stragglers vanish
```

A retry is a fresh attempt (new run_id) or a fresh mint chaining the old
slip; a challenge to a settled lesson is a new mint linking it. One
primitive.

## The shape of an experiment

```
mint     R states the bet — hypothesis, scope, intended change — as the
         slip's FIRST row, before any result exists
author   R clones a throwaway scratch, edits, commits, pins the SHA
         under refs/autolab/runs/<run_id>; the scratch is disposable —
         authoring is not a protocol step
run      ONE composite runner item: smoke → train → eval, each step
         time-boxed, all sharing ctx/ (train writes the checkpoint
         reference there; eval reads its cwd). The item's continuation
         payload names the slip — an address, never a program. Every
         step writes a done-marker into ctx/ as its last act.
verdict  R reads the ticket and the slip, judges: land | park — with
         the headline numbers INLINE in the rationale (a finished slip
         must never depend on its exhaust to be understood). Continuing
         the line is orthogonal: a follow-on is a new mint chained by
         prev_id.
land     verdict land → the orchestrator submits the apply item: merge
         the pinned SHA --no-ff into the client repo, run the canon
         smoke on the merged result, commit iff green. Textual conflict
         → recorded-not-landed + a conflict manifest in the ledger; the
         re-land is a fresh mint that resolves in its own scratch and
         revalidates. The lab never rebases, never fast-forwards, never
         lets a stochastic actor resolve a merge.
curate   C consumes finished-but-uncurated slips oldest-first — land
         status is an attribute, not a gate; crashed runs and parked
         ideas pass the same smelter — writes wiki prose + one lesson
         {scope tags, claim, status ∈ confirmed|refuted|open, refs},
         every lesson citing the ledger rows and run_ids that ground
         it, and commits the wiki. Terminal: Curated. Every mint
         reaches it. A failed apply is ore, not waste.
```

**The row kinds** — a closed vocabulary; the `lab` tool lints every
append and fails loudly on unknown kinds or keys:

```
mint       {slip_id, hypothesis, prev_id?, scope}    the bet, stated FIRST
submit     {slip_id, run_id, sha, pin_ref, label}    one per item submitted
verdict    {slip_id, land|park, rationale, run_ids}  headline numbers inline
apply      {slip_id, run_id, outcome}                landed | recorded-not-landed
curate     {slip_id, lesson}                         the closing row
correction {human, …}                                the one human override
```

## The orchestrator — poll, fold, decide, spawn

A Claude Code workflow on a cadence — cron-shaped, stateless,
rebuildable from ledger + tickets; cadence, never load-bearing. Its own
actions are deliberately dumb; the thinking is delegated to the agents
it spawns. One tick:

```
poll     runner harvest / runner queue — the tickets, via bash
fold     replay every slip file (first-wins per row id); join tickets
         by run_id → ALL lab state, derived
decide   — the complete list:
  1  route — completed item, no verdict row → verdict owed: spawn R
     with the slip + ticket injected as its dossier
  2  branch — failed | killed_box → spawn R to judge or re-mint;
     crashed → consult the ctx/ done-markers: present → treat as
     completed; absent → unknown — fresh attempt (new run_id) or R
  3  admit — unterminated slips < K → a mint may proceed: spawn R
  4  enqueue — head of the lab's own queue → submit to the runner only
     while queue depth ≈ 1 (the share and ordering rules below)
  5  land — verdict land → submit the apply item
  6  curate — finished-but-uncurated → spawn C, oldest first
  7  orphans — ORPHANED ticket → wedge-aware kill-vs-wait; wedged →
     stop submitting, dead-letter the human
  8  dead-letter — a chain with N consecutive dead attempts stops
     auto-retrying; only a human `correction` row revives it
  9  pins — before every submit: refs/autolab/runs/<run_id> at the SHA
 10  idle — nothing owed and room to admit → fresh eyes: spawn R to
     reread the wiki and propose, or escalate
spawn    the chosen agents, dossiers injected; exit the tick
```

## The agents — context in, bash out

R and C are Claude agents spawned per task by the orchestrator, with
their dossier **injected into the prompt** — the slip file, the relevant
ticket JSON, the lesson index; they do not go fishing for state. Their
hands are **bash**: `runner submit|status|harvest|queue` for tickets,
`lab append` for ledger rows (the lint lives in the tool, not in
vigilance), `git` for scratch clones and — C only — the wiki commit. No
MCP surface; a CLI is simpler and the transcript shows every command.

Binding writes are idempotent by reducer semantics: appends carry
deterministic row ids and the fold takes the first — a duplicated or
straggling agent write simply vanishes. **Mint guards**, enforced by
`lab` at the door: admission (unterminated slips < K) and the lessons
wall — best-effort scope match against the lesson index; a hit on a
refuted lesson must be linked as `prev` with a rationale.

Supervision: agents are bounded by the orchestrator's own turn
structure and retries, not by the runner — an agent is a conversation,
not a ticket. The curator is spawned one-at-a-time, always; that
discipline, plus git's own index lock as a loud backstop, is the wiki's
serialization.

## Properties — what can't go wrong, and why

```
one lane      everything that executes — training, eval, landings —
              passes through the runner's single lane, kernel-locked,
              in log order (runner C3, C6). Serializing the GPU, the
              repo, and the queue's own history is ONE mechanism.
nothing lost  a submitted item reaches a terminal, eventually (C1);
              the log is durable, the frontier derived, dispatch
              idempotent. Throw work at it; it runs.
born durable  the ledger is append-only from the first row — there is
              no landing step for knowledge, nothing in-flight to lose;
              a crashed agent costs a scratch clone, never the story.
first wins    deterministic ids + the fold = idempotency without locks,
              leases, or coordination, on both sides of the seam.
land gate     code lands ⟺ the --no-ff merge is textually clean AND the
              merged canon smokes green — deterministic script, total
              order with all other work, no stochastic merges ever.
one terminal  every mint reaches exactly ONE terminal: Curated. Land
              status and run outcomes are attributes, not terminals.
```

Liveness is human-guaranteed, not structural: no auto-parker; an
abandoned slip sits in the ledger until a human parks it. The board's
vitals make that visible, not silent.

## Scheduling — enqueue discipline

The runner's queue is dumb FIFO, constitutionally. All scheduling power
is *what the orchestrator chooses to append, and when*:

- **Queue depth ≈ 1.** The lab's real queue is a projection of the
  fold; releasing one item at a time re-decides the head every tick;
  batching would donate the schedule to arrival order.
- **Share** — exploration < ⅓ of the last M GPU-hours → the oldest
  eligible slip goes next; human-only **nice**, then age.
- **Admission** — K unterminated slips, counted by the fold at mint.
- **The champion** — a sibling cron appends one training slice when the
  queue is empty. No slip, no ledger row; not the lab's concern.

The orchestrator never reads a performance number — Δelo/Δt steers
spend only through R's land/park verdicts. Humans keep one override:
`correction`.

## Evidence and synthesis

Three stores, three lifetimes. **The ledger is the evidence spine** —
per-slip, append-only, durable from birth; heavyweight stores are
*referenced, never protocol-bearing* (rows cite run_ids, W&B run IDs,
checkpoint content-hashes; the bytes stay put). **The exhaust is the
ticket** — spec, step logs, sentinel, `ctx/`; forensics and freight,
never the story. **The wiki is the synthesis layer — the product** —
written only by C, citing ledger rows.

**The lessons wall is best-effort — say it plainly.** Lesson *existence*
is complete by construction (everything passes through C); *enforcement*
is a scope-tag match between stochastic authors — no ontology fixes
that; the domain is too complex for simple sets. A **missed match**
costs one duplicate experiment — cheap, self-healing. A **poisoned
lesson** — a wrong `refuted` — self-reinforcingly deflects search,
because nobody re-runs a refuted claim. Tolerate misses, engineer
against poison — two guards: every lesson cites the ledger rows that
ground it (a challenge is a mechanical audit, not archaeology), and the
board surfaces **load-bearing lessons** — the ones that actually fired
at a mint gate — where human eyes go.

## Stated risks — accepted prices, not unknowns

1. **Semantic staleness.** A textually clean, smoke-green merge can
   still be semantically wrong — validated at one SHA, landing on a
   canon it never trained against. Accepted; backstop: the canon smoke
   runs on the *merged* result, and later experiments run on the merged
   truth.
2. **Conflict re-lands cost a full lane.** Textual conflict → recorded-
   not-landed → a fresh mint resolves, revalidates, re-spends GPU.
3. **Everything shares one lane.** A landing waits behind a training
   slice; a fresh SHA costs seconds at the head of an item (the per-SHA
   delta is the top-level package alone — a few thousand lines of
   Python plus one ccache-able native compile; dep wheels are
   cache-shared across SHAs). Accepted at laptop scale — total order
   kills whole classes of races, and the GPU was the bottleneck anyway.
4. **`crashed` means unknown** — possibly completed; the ctx/
   done-markers disambiguate, retries are new run_ids, runs tolerate
   re-execution.
5. **Admission liveness is human-guaranteed.** An abandoned slip is
   ledger rows, not a held resource; still, only a human parks it.
   Vitals: **age-of-oldest-open-slip**.
6. **The lessons wall misses** — the tolerated, self-healing failure.
7. **Exhaust GC and ledger growth.** Retention is a named lab chore,
   with one hard gate: **exhaust is reapable only for Curated slips**
   (verdict rows carry their headline numbers, so a finished slip never
   needs its tickets back).

## Observability — the board

Projections over ledger + `runner queue` + `runner harvest`: the
frontier and whether it's advancing (**the one vital that matters**),
queue depth, the running item and its step, the curation backlog
(nothing rots silently), **age-of-oldest-open-slip**, chains with
mounting aborts, orphaned items awaiting kill-vs-wait, **never-matched
lessons**, **load-bearing lessons** (the poison watchlist). Append-only
stores make the overnight diff free; the watermark belongs to the
viewer.

## Before building — the tabletop, and the clean slate

**The tabletop.** Before any lab code, run the loop **by hand**: scratch
repos, a runner root, `sleep 2` as train. Play every entity — mint,
submit the composite, watch steps pipe through `ctx/`, verdict, land,
curate. **Walk the crash paths**: kill the wrapper mid-item → `crashed`
→ walk the done-marker branch both ways; kill the orchestrator mid-tick
→ the next tick re-derives; append a duplicate row → the fold ignores
it; conflict two slips on purpose → apply refuses; orphan an item →
rehearse kill-vs-wait. **The primary quarry: the agent fiddly bits** —
dossier size and shape, agent-spawn ergonomics, how an agent behaves
against the CLIs, orchestrator cadence — we will only find these by
trying it.

**Clean slate.** Existing ledgers and experiment records are **not
used, not migrated** — mined for ideas at most. Migration, eventually,
is schema-registry-shaped. The lab starts fresh.

## Design issues we still have

1. **The execution jail.** Runner steps execute with full process
   authority — filesystem, network, keychain. Smoke gates *function*,
   not *safety*. Boundary named; jail unbuilt.
2. **In-flight ledger tamper.** Same uid — `echo >>` works; tamper-
   evident only after commit; cross-checked by the exhaust. Unsolved.
3. **The agent fiddly bits.** Dossier construction from the fold,
   context size, supervision without runner boxes, orchestrator
   liveness and cost, agent misbehavior against the CLIs — the
   tabletop's primary quarry. The orchestrator-as-Claude-Code-workflow
   is the current best idea, not a settled fact.
4. **Two writers touch the client repo** — apply (through the runner)
   and C's wiki commit (direct). Disjoint paths make textual conflict
   impossible; git's index lock is the loud backstop; single-curator
   discipline is the real guard. Watch it at the tabletop.
5. **GPU share accounting.** Box sizes, the share window M, what counts
   as exploration — pure enqueue policy; tabletop knobs.
6. **Pin-ref and scratch hygiene.** `refs/autolab/runs/*` accretes one
   ref per run; scratch clones accrete per researcher. Sweepable
   chores; no policy yet.
7. **C batching.** A single serialized curator trades cross-experiment
   pattern-spotting for simplicity. Policy knob on the consumer;
   revisit with real traces.
8. **The methodology.** Design iteration in a synthesis wiki sprawls;
   versioned pages are the interim answer; rotate the losers out once a
   vN is blessed.

## Cross-refs

[the runner](runner-spec.md) — the substrate ·
[v1 (design-of-record)](autolab-primary-design.md) ·
[red-team A1–A26](autolab-design-adversarial-review.md) ·
[doctrine](autolab-doctrine.md) ·
[as-built architecture](autolab-architecture.md).
