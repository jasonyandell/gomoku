# The Autolab — design v5 (first client of the runner)

> **Status: WORKING DRAFT v5** *(2026-07-05)* — the v4 concepts
> refactored onto [the runner](runner-spec.md). Supersedes
> [v4](autolab-design-v4.md) as the walk-through draft;
> [v1](autolab-primary-design.md) remains design-of-record until a vN is
> blessed. Red-team record: [A1–A26](autolab-design-adversarial-review.md)
> plus the 2026-07-05 session exchange (unfiled).

## One breath

A laptop runs a research lab on itself, on a substrate that knows only
tickets. Three adjacent stores: the **client repo** (code + wiki), the
**ledger repo** (one append-only slip file per experiment, durable from
its first row), the **runner root** (spool + exhaust). One noun executes
everything — **the run**: smoke (`null`), train and eval (`gpu`), apply
(`canon`), curate (`wiki`) — every serialization point is the runner's
flock. A **cron-shaped, stateless lab loop** folds ledger + `harvest()`
into a decision and enqueues just-in-time; binding writes are appends
with deterministic ids, duplicates ignored by the fold — first wins. The
researcher bets before any result exists; the applicator lands code only
if git's own merge is clean and the merged canon smokes green; the
ledger is born durable — nothing to land. The champion trains outside,
unseen. **The wiki is the product; the player is a byproduct.**

## What changed since v4

1. **The substrate is extracted** — watchdog → wrapper + box (C1); U →
   the `gpu` tag lock (C2); `al` → flock; grant order → the spool (C5).
   The lab keeps policy; the runner keeps order, exclusion, witness.
2. **Worktrees are gone as protocol entities** — authoring in throwaway
   scratch clones; runs execute at pinned git SHAs via
   `uvx --from git+file:///…@<sha> cmd`. No Wslots, no Wlive tokens, no
   orphaned-worktree reconciliation — that class died with the entity.
3. **The trainer/evaluator split is gone** — one noun: the run.
4. **The ledger gets its own repo, durable from birth** — nothing to
   land (I7 dissolves); rows cite run_ids; **the exhaust is the
   ticket**. The runner knows tickets; only the lab knows experiments.
5. **Idempotency by reducer semantics** — appends with deterministic
   ids; duplicates ignored by the fold; no locks, no leases lab-side.
6. **Pipelines pre-declared at submit**; stages advance from harvested
   payloads; **no conditionals in payloads** — branching is the reducer.
7. **Scheduling is enqueue discipline** — dumb FIFO spool; the lab's
   real queue in the ledger; spool depth ≈ 1 on `gpu`.
8. **The champion is a sibling client** — its own cron, one slice when
   `probe(gpu)` shows empty; the lab never knows it exists.
9. **Claude sessions as runs** (proposed default) — headless `claude -p`
   with a folded-ledger dossier; boxes, exhaust capture, supervision.
10. **Runner blemishes become lab semantics** — B1: `crashed` =
    "unknown, possibly completed" (done-marker, new-run_id retries);
    B2: orphan settling is wedge-aware lab policy.
11. **Three adjacent stores**; exhaust retention is a named lab chore.

## The entities

| | Entity | Job |
|---|---|---|
| **G** | client repo | code + wiki — truth of what runs and what's known; advances only by apply (`canon`) and curate (`wiki`) commits |
| **ledger** | ledger repo | one append-only slip file per experiment, named by slip_id, durable from the first row; rows cite run_ids; the lab's only load-bearing state |
| **runner** | the substrate | spool + exhaust; owns order (C5), exclusion (C2), witness (C1); knows tickets, never experiments |
| **R** | researcher (Claude) | stochastic proposer: mints, states the bet, authors in a throwaway scratch clone, pins a SHA, submits, verdicts; **never mutates canon** |
| **A** | applicator | a deterministic script **run under tag `canon`**: `--no-ff` merge + canon smoke, commit iff green |
| **C** | curator (Claude) | single, serialized **under tag `wiki`**: consumes terminal-uncurated slips → wiki prose + one lesson; **the closing transition** |
| **loop** | the lab loop | cron-shaped, stateless: fold ledger + harvest → decide → enqueue; **all policy lives here** |

Gone from v4's table: **W**, **T/E** (just runs), **U** (tag `gpu`),
**al** (append discipline + fold), **watchdog** (box + harvest). The
champion is *not* an entity. Stochastic actors author; substrate
mutates; everybody reads the ledger and the tickets.

## The slip id — the whole threading mechanism

```
within an experiment : one slip_id, one file in the ledger repo
                       → disjoint by uniqueness; appends never conflict
across experiments   : optional prev_id — an immutable chain; lineage,
                       retries, and challenges are all just links
retrieval            : slip_id IS the file's name → deterministic open;
                       no search, no snapshot-consistency question
runner run_ids       : <slip_id>-s<stage>-a<attempt> — deterministic, so
                       submit is idempotent (C3); the fold joins tickets
                       to slips by name
ledger row ids       : {kind}:{slip_id}:{stage}:{attempt} — the fold's
                       first-wins key; duplicates and stragglers vanish
```

A retry is a fresh attempt (new run_id) or a fresh mint chaining the
aborted one; a challenge is a new mint linking the lesson. One primitive.

## The lab loop — the reducer

Cron-shaped, stateless, rebuildable from ledger + tickets — cadence,
never load-bearing; killed at any line, the next pass re-derives the
same decisions from the same fold. One pass:

```
fold     replay every slip file (first-wins per row id); join
         harvest()/status() by run_id → ALL lab state, derived
decide   — the complete list of what the loop decides:
  1  stage advance — stage N completed → submit stage N+1, argv
     parameterized from stage N's ctx/ (train's checkpoint hash → eval)
  2  branch — failed | killed_box → route to R; crashed → consult the
     ctx/ done-marker: present → completed; absent → unknown — fresh
     attempt (new run_id) or route to R
  3  verdict owed — final stage terminal, no verdict row → spawn R with
     the slip's fold as dossier
  4  admit — unterminated slips < K → a mint may proceed (spawn R)
  5  enqueue — head of the lab's own queue → submit to gpu only while
     spool depth ≤ 1 (⅓ exploration share; human-only nice, then age)
  6  land — verdict land → enqueue apply under canon; park → no apply
  7  curate — terminal-but-uncurated → enqueue C under wiki, oldest first
  8  orphans — ORPHANED → wedge-aware kill-vs-wait; wedged → quarantine
     the gpu tag, dead-letter the human
  9  dead-letter — a chain with N consecutive dead attempts stops
     auto-retrying; only a human `correction` row revives it
 10  pins — before every submit: refs/autolab/runs/<run_id> at the SHA
 11  idle — nothing owed and room to admit → fresh eyes: spawn R to
     reread the wiki and propose, or escalate
enqueue  submit() the chosen specs; exit. No in-process state survives.
```

## Pipelines and stages — mint, submit, verdict

**The row kinds** — a closed vocabulary; the mint lint fails loudly:

```
mint       {slip_id, hypothesis, prev_id?, scope}    the bet, stated FIRST
submit     {slip_id, run_id, sha, pin_ref, stage}    one per stage attempt
verdict    {slip_id, land|park, rationale, run_ids}  cites tickets judged;
                                                     headline numbers INLINE
apply      {slip_id, run_id, outcome}                landed | recorded-not-landed
curate     {slip_id, lesson}                         the closing row
correction {human, …}                                the one human override
```

**Mint** — guards: admission (unterminated slips < K) and the lessons
wall (best-effort scope match; a hit on a refuted lesson must be linked
as prev with a rationale). **The bet is the first row, before any result
exists.** Authoring is not a protocol step: R clones a throwaway scratch,
edits, commits, pins `refs/autolab/runs/<run_id>` — disposable after.

**Submit** — the pipeline is pre-declared: stage 0 **smoke** (`null`;
doubles as the env pre-warm — cold-SHA uv builds happen off the gpu
tag), stage 1 **train** (`gpu`), stage 2 **eval** (`gpu`), each its own
box. The continuation payload (opaque, ≤ 64 KiB, returned verbatim as a
courtesy) names {slip_id, next_stage} — an address, never a program.
Runs write a **done-marker** into `ctx/` as their last act (the B1
disambiguator). New shapes require a new mint.

**Verdict** — land|park, citing the run_ids judged, with the headline
numbers inline in the rationale (a terminated slip must never depend on
its exhaust to be understood). Land makes the slip eligible for apply;
park goes straight to curation. Continuing the line is orthogonal to
either — a follow-on is a new mint chained by prev_id.

**Apply** — a deterministic script under tag `canon`: attempt a
`--no-ff` merge of the experiment SHA into the client repo, run the
canon smoke on the merged result, commit iff green. Textual conflict →
recorded-not-landed + a conflict manifest in the ledger; re-land = a
fresh mint that resolves in its own scratch and revalidates. The lab
never rebases, never fast-forwards, never lets a stochastic actor merge.

**Curate** — a single serialized Claude session under tag `wiki`
consumes terminal-uncurated slips oldest-first (land-status is an
attribute, not a gate — crashed runs and parked ideas pass the same
smelter): wiki prose + a one-line lesson {scope tags, claim, status ∈
confirmed|refuted|open, refs}, every lesson citing the ledger rows /
run_ids that ground it; the wiki commit happens inside the tagged run.

**Claude sessions as runs** — proposed default, tabletop quarry: R and C
sessions are themselves runs (argv = headless `claude -p` with a
folded-ledger dossier) — boxes, exhaust capture, uniform supervision.

## Invariants — where v4's walls went

```
BECAME RUNNER CLAIMS — bought, not maintained
I4   U ∈ {0,1}                → C2 on tag gpu — kernel flock, fd-inherited
I9   at most one C in flight  → C2 on tag wiki (one A: C2 on canon)
I10  every slice returns a result (the watchdog)
                              → C1 single terminal + harvest settling
one binding write, first wins → C3 below the seam; the fold above it
grant order                   → C5 — FIFO per tag by sequence number

BECAME LAB POLICY — the reducer's law, not drawn walls
I3   land ⟺ clean --no-ff merge AND merged canon smokes green —
     verbatim from v4, now a deterministic script under tag canon
I8   every mint reaches exactly ONE terminal: Curated — liveness
     human-guaranteed; vitals: age-of-oldest-open-slip
K    admission — a count over the fold, not a token supply
share / nice / age            → enqueue discipline (§ Scheduling)

DIED WITH WORKTREES
I1   Wslot/Wlive conservation — no entity, no leak, no reconciliation;
     an abandoned experiment is ledger rows, not a held resource
I2   base generation on W — the SHA pin rides every submit row
I5   atomic W+L birth — mint is one append; nothing else is born
I6   three doors — became tags; serialization moved into flock
I7   L always lands — dissolved, stronger: the ledger is born durable
```

## Scheduling as enqueue discipline

The spool is dumb FIFO per tag — constitutional (runner §5). All
scheduling power is *what the lab chooses to enqueue, and when*:

- **Spool depth ≈ 1 on `gpu`** — the lab's queue is a projection of the
  fold; releasing one run at a time re-decides the head every pass;
  batching would donate the schedule to arrival order.
- **Share** — exploration < ⅓ of the last M GPU-hours → oldest eligible
  stage next; human-only **nice**, then age. Eval before train (short
  runs unblock verdicts; time-to-verdict denominates Δelo/Δt).
- **Admission** — K unterminated slips, counted by the fold at mint.
- **The champion is external** — a sibling client cron enqueues one
  training slice when `probe(gpu)` shows empty. No slip, no ledger row;
  its evidence lives in W&B and the training notebook.

The loop never reads a performance number — Δelo/Δt steers spend only
through R's land/park verdicts. Humans keep one override: `correction`.

## Evidence and synthesis

Three stores, three lifetimes. **The ledger is the evidence spine** —
per-slip, append-only, durable from the first row; heavyweight stores
are *referenced, never protocol-bearing* (rows cite run_ids, W&B run
IDs, checkpoint content-hashes; the bytes stay put). **The exhaust is
the ticket** — spec, logs, exit.json, `ctx/`; forensics and freight,
never the story. **The wiki is the synthesis layer — the product** —
written only by C, citing ledger rows.

**The lessons wall is best-effort — say it plainly.** Lesson *existence*
is complete by construction (everything passes through C); *enforcement*
is a scope-tag match between stochastic authors — no ontology fixes it;
the domain is too complex for simple sets. A **missed match** costs one
duplicate experiment — cheap, self-healing; a **poisoned lesson** — a
wrong `refuted` — self-reinforcingly deflects search, because nobody
re-runs a refuted claim. Tolerate misses, engineer against poison — two
guards: every lesson cites the ledger rows that ground it (a challenge
is a mechanical audit, not archaeology), and the board surfaces
**load-bearing lessons** — fired at a mint gate, where human eyes go.

## Stated risks — accepted prices, not unknowns

1. **Semantic staleness.** A textually clean, smoke-green merge can
   still be semantically wrong — validated at one SHA, landing on a
   canon it never trained against. Accepted (v3's head-equality CAS
   stays dead); backstop: A's canon smoke on the merged result.
2. **Conflict re-lands cost a full lane.** Textual conflict → recorded-
   not-landed → a fresh mint resolves, revalidates, re-spends GPU.
3. **Admission liveness is human-guaranteed, not structural.** No
   auto-parker (deliberate); an abandoned slip sits in the ledger until
   a human parks it. MUCH softer than v4 — no slot leaks, just rows —
   still human-guaranteed. Vitals: **age-of-oldest-open-slip**.
4. **`crashed` means unknown** (B1): possibly completed; the done-marker
   disambiguates, retries are new run_ids, runs tolerate re-execution.
5. **The lessons wall misses** — the tolerated, self-healing failure.
6. **SHA reachability.** Pin `refs/autolab/runs/<run_id>` before submit
   or uv cannot materialize the env later — fails late, against the box.
7. **Exhaust GC and ledger-repo growth.** Exhaust is bulk; the ledger
   grows forever. Retention is a named lab chore — with one hard gate:
   **exhaust is reapable only for Curated slips** (verdict rows carry
   their headline numbers inline, so a terminated slip never needs its
   tickets back).

## Observability — the board

The board renders projections over ledger + `probe()` + `harvest()`:
running and on-deck per tag, spool depth, the curation backlog (nothing
rots silently), **age-of-oldest-open-slip**, chains with mounting
aborts, orphaned runs awaiting kill-vs-wait, **never-matched lessons**,
**load-bearing lessons** (fired at a mint gate — the poison watchlist).
Append-only makes the overnight diff free; the watermark is the viewer's.

## Before building — the tabletop, and the clean slate

**The tabletop.** Before any code, run this lab **by hand**: scratch
repos, a runner root, `sleep 2` as train. Play every entity — mint,
submit the pipeline, watch stages advance from harvested payloads,
verdict, apply, curate. **Walk the crash paths**: kill the wrapper
mid-run → `crashed` → walk the done-marker branch both ways; kill the
loop mid-pass → the next pass re-derives; append a duplicate row → the
fold ignores it; conflict two lanes → apply refuses; orphan a run →
rehearse kill-vs-wait. **The primary quarry: the Claude-session fiddly
bits** — headless auth, MCP availability, dossier size, cost — decide
whether sessions-as-runs stays the default.

**Clean slate.** Existing ledgers are **not used, not migrated** — mined
for ideas at most; migration, eventually, is schema-registry-shaped.

## Known unknowns

1. **The execution jail.** Runs execute with full process authority —
   filesystem, network, keychain. Smoke gates *function*, not *safety*.
   Boundary named; jail unbuilt.
2. **In-flight ledger tamper.** Same uid — `echo >>` works; tamper-
   evident only after commit. Smaller blast radius than v4 (one slip,
   cross-checked by the exhaust), still unsolved.
3. **Headless Claude sessions.** Auth, MCP, cost, dossier construction
   under a box — the tabletop's primary quarry; proposed, not blessed.
4. **GPU quantum and share accounting.** Box sizes, eval-before-train,
   the share window M — pure enqueue policy now; tabletop knobs.
5. **Pin-ref hygiene.** `refs/autolab/runs/*` accretes one ref per run;
   scratch clones accrete per researcher. Sweepable chores, no policy.
6. **C batching.** Serialized C trades cross-experiment pattern-spotting
   for simplicity. Policy knob on the consumer; revisit with traces.
7. **The methodology.** v1–v5 pages are the sprawl evidence; rotate the
   losers out once a vN is blessed.

## Cross-refs

[v4](autolab-design-v4.md) · [v3](autolab-design-v3.md) ·
[v2](autolab-design-next.md) ·
[v1 (design-of-record)](autolab-primary-design.md) ·
[red-team A1–A26](autolab-design-adversarial-review.md) ·
[doctrine](autolab-doctrine.md) · [runner spec](runner-spec.md).
