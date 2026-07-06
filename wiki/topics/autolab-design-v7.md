# The Autolab — design v7

> **Status: WORKING DRAFT v7** *(2026-07-05)* — v3's gestalt on v6's
> substrate: the Petri-shaped walls restored (deterministic policy, one
> door per store, drawable invariants) over the runner, the born-durable
> ledger, and first-wins idempotency. Supersedes
> [v6](autolab-design-v6.md); [v1](autolab-primary-design.md) remains
> design-of-record until a vN is blessed. Substrate:
> [the runner](runner-spec.md). Red-team:
> [A1–A26](autolab-design-adversarial-review.md).

## One breath

A laptop runs a research lab on itself. Truth lives in three adjacent
stores: the **client repo** (code + wiki — canon), the **ledger repo**
(one append-only file per experiment, durable from its first row), and
the **runner root** (a work queue that is also its own history).
Everything that executes — training, eval, code landings, **wiki
landings** — passes through the runner's one lane, in order. Policy is
**the tick**: a dumb, flocked, deterministic reducer that polls, folds,
decides by fixed rules, and spawns Claude exactly where judgment lives —
a **researcher** who states the bet and judges the result, a **curator**
who smelts every finished experiment into the wiki. Stochastic actors
author; deterministic substrate mutates; hypothesis precedes evidence; a
failed experiment is ore, not waste. **The wiki is the product; the
player is a byproduct.**

## The entities

| | Entity | Job |
|---|---|---|
| **G** | client repo | code + wiki — canon; mutated **only by items in the lane** (code apply, wiki apply); never committed to directly by anyone |
| **ledger** | ledger repo | one append-only slip file per experiment, named by slip_id, durable from the first row; every row cites its evidence; the lab's only load-bearing state |
| **runner** | the substrate | one queue, one lane, an append-only log; owns order, exclusion, witness; knows tickets, never experiments ([spec](runner-spec.md)) |
| **tick** | the policy reducer | `lab tick`: a deterministic script on a cadence, flocked singleton; polls, folds, decides by the fixed list below, spawns agents, submits items; **holds no state and contains no model** — killed at any line, the next tick re-derives everything |
| **R** | researcher (agent) | stochastic proposer: mints, states the bet, authors in a throwaway scratch clone, pins a SHA, submits, judges; **never mutates canon** |
| **A** | applicator | a deterministic script submitted as a runner item: `--no-ff` merge + a green gate, commit iff green — **the same shape for code and wiki** |
| **C** | curator (agent) | single, serialized: consumes finished-but-uncurated slips → wiki prose + one lesson, then **submits its wiki landing as a lane item**; the closing transition |

The champion is *not* an entity — a sibling cron appends one training
slice whenever the queue is empty. No slip, no ledger row; its evidence
lives in W&B and the training notebook.

**Claude appears in exactly two places — R and C — and in neither does
it hold the pen on canon or policy.** The tick's decide list is short
enough to be a script, so it is one: putting a model where no judgment
is exercised buys stochastic rule-following and a per-tick token bill,
and nothing else.

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
         time-boxed, all sharing ctx/; the continuation payload names
         the slip — an address, never a program; every step writes a
         done-marker into ctx/ as its last act
verdict  R reads the ticket and the slip, judges: land | park — with
         the headline numbers INLINE in the rationale (a finished slip
         must never depend on its exhaust to be understood). A follow-on
         is a new mint chained by prev_id.
land     verdict land → the tick submits the code-apply item: merge the
         pinned SHA --no-ff into the client repo, run the canon smoke on
         the merged result, commit iff green. Textual conflict →
         recorded-not-landed + a conflict manifest; the re-land is a
         fresh mint that resolves in its own scratch and revalidates.
         The lab never rebases, never fast-forwards, never lets a
         stochastic actor resolve a merge.
curate   C consumes finished-but-uncurated slips oldest-first — land
         status is an attribute, not a gate; crashed runs and parked
         ideas pass the same smelter. C authors wiki prose + one lesson
         {scope tags, claim, status ∈ confirmed|refuted|open, refs},
         every lesson citing the ledger rows and run_ids that ground
         it, in a scratch clone — then submits the WIKI-APPLY item
         (--no-ff merge, lint gate, commit iff green). Terminal:
         Curated, when the wiki apply lands. Every mint reaches it. A
         failed apply is ore, not waste.
```

**The row kinds** — a closed vocabulary; the `lab` tool lints every
append and fails loudly on unknown kinds or keys:

```
mint       {slip_id, hypothesis, prev_id?, scope}    the bet, stated FIRST
submit     {slip_id, run_id, sha, pin_ref, label}    one per item submitted
verdict    {slip_id, land|park, rationale, run_ids}  headline numbers inline
apply      {slip_id, run_id, target: code|wiki,      landed |
            outcome}                                  recorded-not-landed
curate     {slip_id, lesson, wiki_run_id}            the closing row
correction {human, …}                                the one human override
```

## The tick — poll, fold, decide, spawn (no model inside)

`lab tick` is a deterministic script, kicked by cron, running under its
own flock (overlapping ticks are structurally impossible, same shape as
the runner's `dispatch.lock`). It is a **total function of (ledger,
tickets)**: killed at any line, the next tick re-derives everything.
One tick:

```
poll     runner harvest / runner queue — the tickets
fold     replay every slip file (first-wins per row id); join tickets
         by run_id → ALL lab state, derived
decide   — the complete list, fixed, in order:
  1  route — completed item, no verdict row → verdict owed: spawn R
     with `lab dossier <slip_id>` injected (slip + ticket + lessons)
  2  branch — failed | killed_box → spawn R to judge or re-mint;
     crashed → consult ctx/ done-markers: present → treat as
     completed; absent → unknown — fresh attempt (new run_id) or R
  3  admit — unterminated slips < K → a mint may proceed: spawn R
  4  enqueue — head of the lab's own queue → submit to the runner only
     while queue depth ≈ 1 (share and ordering rules below)
  5  land — verdict land, no apply row → submit the code-apply item
  6  curate — finished-but-uncurated → spawn C, oldest first, one at
     a time; curate verdict in → submit the wiki-apply item
  7  orphans — ORPHANED ticket → wedge-aware kill-vs-wait; wedged →
     stop submitting, dead-letter the human
  8  dead-letter — a chain with N consecutive dead attempts stops
     auto-retrying; only a human `correction` row revives it
  9  pins — before every submit: refs/autolab/runs/<run_id> at the SHA
 10  idle — nothing owed and room to admit → fresh eyes: spawn R to
     reread the wiki and propose, or escalate
spawn    the chosen agents, dossiers injected; exit the tick
```

Dossiers are themselves deterministic: `lab dossier` renders a
projection of the fold — the slip file, the ticket JSON, the lesson
index — so what an agent saw is reproducible from (ledger, tickets,
watermark). The tick never reads a performance number — Δelo/Δt steers
spend only through R's land/park verdicts. Humans keep one override:
`correction`.

## The agents — context in, bash out

R and C are Claude agents spawned per task by the tick, dossier
**injected into the prompt**; they do not go fishing for state. Their
hands are **bash**: `runner submit|status|harvest|queue` for tickets,
`lab append` for ledger rows (the lint lives in the tool, not in
vigilance), `git` for scratch clones. **Neither agent ever commits to
canon** — R's code and C's prose both land as lane items through A. No
MCP surface; a CLI is simpler and the transcript shows every command.

Binding writes are idempotent by reducer semantics: deterministic row
ids, first-wins fold — a duplicated or straggling agent write simply
vanishes. **Mint guards**, enforced by `lab` at the door: admission
(unterminated slips < K) and the lessons wall — scope match against the
lesson index; a hit on a refuted lesson must be linked as `prev` with a
rationale. Agents are bounded by the tick's spawn-and-retry structure,
not by the runner — an agent is a conversation, not a ticket.

## Invariants — the walls, and which kind each is

```
I1  one lane        everything that executes — training, eval, code
                    landings, wiki landings — passes through the
                    runner's single lane, kernel-locked, in log order
                    (runner C3, C6). STRUCTURAL.
I2  born durable    the ledger is append-only from the first row; no
                    landing step for knowledge, nothing in-flight to
                    lose. STRUCTURAL.
I3  first wins      deterministic ids + the fold = idempotency without
                    locks or leases, both sides of the seam. STRUCTURAL.
I4  one door/store  `lab` is the ledger's sole mutator; THE LANE is
                    canon's sole mutator (code apply and wiki apply are
                    items); nobody commits to canon directly. STRUCTURAL
                    for the lane; convention for `lab` (see issue 2).
I5  land gate       canon changes ⟺ --no-ff merge textually clean AND
                    the merged result passes its green gate (canon
                    smoke for code, lint for wiki). Deterministic
                    script; no stochastic merges ever. STRUCTURAL.
I6  single tick     policy is a flocked, deterministic total function
                    of (ledger, tickets); overlapping ticks can't run;
                    Claude appears only in R and C. STRUCTURAL.
I7  admission       mints pass `lab`'s guard (K, lessons wall) and only
                    the singleton tick grants mint permission — the
                    double-mint of two racing deciders can't be drawn.
                    STRUCTURAL, given I6.
I8  one terminal    every mint reaches exactly ONE terminal: Curated.
                    Land status and run outcomes are attributes.
                    CHECKED by the fold; liveness is human-guaranteed
                    (no auto-parker; vitals make abandonment visible).
I9  bet first       the mint row precedes every result row — hypothesis
                    stated before evidence exists. CHECKED by `lab`
                    (a slip's first row must be a mint).
```

Marking each wall STRUCTURAL vs CHECKED is deliberate: the sim (and the
tabletop) certify the checked ones; the structural ones are properties
of flock, link, append, and the fold — drawn, not patrolled.

## Scheduling — enqueue discipline

The runner's queue is dumb FIFO, constitutionally. All scheduling power
is *what the tick chooses to append, and when*:

- **Queue depth ≈ 1.** The lab's real queue is a projection of the
  fold; releasing one item at a time re-decides the head every tick.
- **Share** — exploration < ⅓ of the last M GPU-hours (denominator from
  runner history — the champion has no ledger rows) → the oldest
  eligible slip goes next; human-only **nice**, then age.
- **Admission** — K unterminated slips, counted by the fold at mint.
- **The champion** — a sibling cron appends one training slice when the
  queue is empty. Not the lab's concern.

## Evidence and synthesis

Three stores, three lifetimes. **The ledger is the evidence spine** —
per-slip, append-only, durable from birth; heavyweight stores are
referenced, never protocol-bearing (rows cite run_ids, W&B run IDs,
checkpoint content-hashes; the bytes stay put). **The exhaust is the
ticket** — forensics and freight, never the story. **The wiki is the
synthesis layer — the product** — written only by C, landed only
through the lane, citing ledger rows.

**The lessons wall is best-effort — say it plainly.** Lesson *existence*
is complete by construction (everything passes through C); *enforcement*
is a scope-tag match between stochastic authors — no ontology fixes
that. A **missed match** costs one duplicate experiment — cheap,
self-healing. A **poisoned lesson** — a wrong `refuted` —
self-reinforcingly deflects search. Tolerate misses, engineer against
poison: every lesson cites the ledger rows that ground it (a challenge
is a mechanical audit, not archaeology), and the board surfaces
**load-bearing lessons** — the ones that actually fired at a mint gate —
where human eyes go.

## Stated risks — accepted prices, not unknowns

1. **Semantic staleness.** A textually clean, smoke-green merge can
   still be semantically wrong. Accepted; the canon smoke runs on the
   *merged* result, and later experiments run on the merged truth.
2. **Conflict re-lands cost a full lane.** Textual conflict → recorded-
   not-landed → a fresh mint resolves, revalidates, re-spends GPU.
3. **Everything shares one lane.** A landing waits behind a training
   slice; a wiki landing waits behind both. Accepted at laptop scale —
   total order kills whole classes of races, and the GPU was the
   bottleneck anyway.
4. **`crashed` means unknown** — possibly completed; ctx/ done-markers
   disambiguate; retries are new run_ids; runs tolerate re-execution.
5. **Liveness is human-guaranteed.** An abandoned slip is ledger rows,
   not a held resource; only a human parks it. Vitals:
   **age-of-oldest-open-slip**.
6. **The lessons wall misses** — the tolerated, self-healing failure.
7. **Exhaust GC and ledger growth.** Retention is a named chore, one
   hard gate: **exhaust is reapable only for Curated slips** (verdicts
   carry their headline numbers).
8. **A typo'd experiment spends lane time to fail its smoke** (smoke is
   step 1 of the composite, not a mint gate). At depth ≈ 1 the wait is
   bounded by one item; accepted for the simpler mint.

## Observability — the board

Projections over ledger + `runner queue` + `runner harvest`: the
frontier and whether it's advancing (**the one vital that matters**),
queue depth, the running item and its step, the curation backlog,
**age-of-oldest-open-slip**, chains with mounting aborts, orphaned items
awaiting kill-vs-wait, never-matched lessons, load-bearing lessons (the
poison watchlist). Append-only stores make the overnight diff free; the
watermark belongs to the viewer.

## Before building — the tabletop, and the clean slate

**The tabletop.** Before any lab code, run the loop **by hand**: scratch
repos, a runner root, `sleep 2` as train. Play every entity — mint,
submit the composite, watch steps pipe through ctx/, verdict, land,
curate-and-land-the-wiki. **Walk the crash paths**: kill the wrapper
mid-item; kill the tick mid-decide (the flock releases, the next tick
re-derives); append a duplicate row (the fold ignores it); conflict two
slips on purpose (apply refuses); orphan an item (rehearse
kill-vs-wait). **The primary quarry: the agent fiddly bits** — dossier
size and shape, spawn ergonomics, agent behavior against the CLIs, tick
cadence — found only by trying it.

**Clean slate.** Existing ledgers and experiment records are **not
used, not migrated** — mined for ideas at most. Migration, eventually,
is schema-registry-shaped. The lab starts fresh.

## Design issues we still have

1. **The execution jail.** Runner steps execute with full process
   authority. Smoke gates *function*, not *safety*. Boundary named;
   jail unbuilt.
2. **In-flight ledger tamper.** Same uid — `echo >>` works; tamper-
   evident only after commit; cross-checked by the exhaust. Unsolved
   (I4 is convention on the `lab` side).
3. **The agent fiddly bits.** Dossier construction, context size,
   supervision without runner boxes, agent misbehavior against the
   CLIs — the tabletop's primary quarry.
4. **Tick implementation.** `lab tick` as a deterministic script is the
   design; during the tabletop a human (or a Claude session) *plays*
   the tick by following the decide list verbatim — the list is the
   spec, and any ambiguity a player hits is a bug in this page.
5. **GPU share accounting.** Box sizes, the share window M, what counts
   as exploration — pure enqueue policy; tabletop knobs.
6. **Pin-ref and scratch hygiene.** `refs/autolab/runs/*` and scratch
   clones accrete. Sweepable chores; no policy yet.
7. **C batching.** Single serialized C trades cross-experiment
   pattern-spotting for simplicity. Policy knob; revisit with traces.
8. **The methodology.** Design iteration in a synthesis wiki sprawls;
   versioned pages are the interim answer; rotate the losers once a vN
   is blessed.

## Cross-refs

[the runner](runner-spec.md) — the substrate ·
[v6](autolab-design-v6.md) (superseded) ·
[v3](autolab-design-v3.md) (the Petri statement) ·
[v1 (design-of-record)](autolab-primary-design.md) ·
[red-team A1–A26](autolab-design-adversarial-review.md) ·
[doctrine](autolab-doctrine.md).
