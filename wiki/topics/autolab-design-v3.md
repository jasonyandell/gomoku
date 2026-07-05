# The Autolab — design v3 (the Petri model)

> **Status: WORKING DRAFT v3** *(2026-07-05)* — from-the-top rewrite on the
> Petri-shaped execution model (Jason + Opus noodle, refined in session).
> Supersedes [autolab-design-next.md](autolab-design-next.md) (v2) as the
> walk-through draft; [autolab-primary-design.md](autolab-primary-design.md)
> (v1) remains design-of-record until a vN is blessed. Red-team record:
> [A1–A26](autolab-design-adversarial-review.md).

## One breath

A laptop runs a research lab on itself. Truth lives in git: the code in
canon, plus one small append-only **ledger per experiment**, named by its
routing-slip id, landed into canon no matter what. A researcher session
mints a worktree and its ledger together, proposes, works, and never touches
canon; the trainer holds the one GPU; the evaluator scores; a deterministic
**Applicator** is the only thing that mutates main — it *always* lands the
ledger, and lands the code only if the base still matches. The experiment
isn't finished at landing: a single **curator** reads each landed ledger and
smelts it into the wiki — prose plus a one-line lesson — so even a failed
apply becomes knowledge, and nothing rots in a heap. The only real
contention is GPU time; wrong states can't be drawn. **The wiki is the
product; the player is a byproduct.**

## The entities

| | Entity | Job |
|---|---|---|
| **G** | git canon | truth of code *and* landed ledgers; advances by generation (commit) |
| **W** | worktree | a linear token: an isolated checkout minted at base generation `v` |
| **L** | experiment ledger | minted **with** W, named by the unique slip id; append-only; the whole story of one experiment; optional `prev_id` links a predecessor |
| **R** | researcher (Claude) | stochastic proposer: mints, hypothesizes, works in W, closes; **never mutates canon** |
| **T** | trainer | the one physically serialized actor; runs on the GPU; appends perf/failure to L |
| **E** | evaluator | scores after T; appends eval results to L |
| **A** | Applicator | deterministic lander; **the only thing that mutates main** |
| **C** | curator (Claude) | single, serialized: consumes landed ledgers → wiki prose + a lesson; **the closing transition** |
| **U** | the GPU | one holder at a time |
| **al** | the one tool | sole mutator of any L; enforces every mint guard |

Two doors, both deterministic substrate: **`al` is the door to L** (during
the experiment); **A is the door to G** (at landing). Nothing else writes
anything. Nobody reads anybody — everybody reads git and the ledgers.

## The slip id — the whole threading mechanism

```
within an experiment : one slip_id, one L, three writers (R@mint, T, E)
                       → disjoint by uniqueness; appends never conflict
across experiments   : optional prev_id forms an immutable chain
                       → lineage, retries, and challenges are all just links
retrieval            : slip_id IS L's name → deterministic open;
                       no search, no snapshot-consistency question
```

A retry is a new mint chaining the aborted one; a challenge to a settled
lesson is a new mint linking the lesson's experiment. One primitive.

## Transitions (guard → effect)

```
R_mint :  guard — a free Wslot (supply = K, the admission cap as token
          supply) AND the mint lint (unknown kinds/keys fail loudly) AND
          the lessons wall (scope hits a refuted lesson → must link it as
          prev with a rationale) AND SMOKE. al enforces all of it.
          → produce Wlive@v AND L[slip_id]
          → append: hypothesis + intended change (the recipe, at mint)

T      :  guard — U granted by the arbiter (see scheduler)
          consume U → run slice → produce U
          → append perf / failure to L → emit to E's queue

E      :  → score → append eval results to L → emit to R's queue

R_close:  L now holds the full story → release W to A (R does NOT apply)

A      :  ALWAYS commit L[slip_id] to G (unique name → conflict-free)  [I7]
          code: if W.base == G.head → land code at G@(v+1)   [landed]
                else → append "code didn't land + why + provenance"
                       [recorded-not-landed]; branch retained as cache
          free the Wslot either way
          (the base check is a compare-and-swap on G.head, implemented as
          a --no-ff merge — this repo never rebases or fast-forwards)

C      :  consume the oldest landed-but-uncurated L
          → synthesize prose into the wiki + file the one-line lesson
            {scope tags, claim, status ∈ confirmed|refuted|open, refs}
          → prose lands through A, same base guard (C is serialized, so
            the guard never fires in practice)
          → terminal: Curated

Reconcile: crashed R ⟹ orphaned Wlive found via `git worktree list`
           (git is the liveness detector — no spawn rows, no leases)
           → resolve by A's guards → [aborted | re-land]; L durable anyway
```

Route shapes are free: the queues support eval-only routes (register a
model, run the gamut) or any reorder of the same closed vocabulary. New
transition *kinds* are gated, human-visible changes.

## Invariants — the wrong states that can't be drawn

```
I1  Wlive + Wslot = 1 per slot; slot supply = K   -- no orphan, no double-mint,
                                                     WIP cap is the token supply
I2  every Wlive carries its base generation v      -- R reasons against a pin
I3  A lands code ⟺ W.base == G.head                -- stochastic actors never
                                                     mutate canon; stale apply
                                                     structurally barred
I4  U ∈ {0,1}                                      -- one training at a time
I5  L named by unique slip_id                      -- appends never conflict
I6  al sole L-mutator; A sole G-mutator            -- two doors, both substrate
I7  L always lands into G                          -- knowledge unconditionally
                                                     durable, even on code failure
I8  every mint reaches exactly ONE terminal: Curated
    (land-status ∈ {landed, recorded-not-landed, aborted} is an ATTRIBUTE,
     not a terminal — a failed apply is ore, not waste)
I9  at most one C in flight                        -- single curator, serialized;
                                                     simplicity over throughput
```

The sim asserts these directly — they are place invariants of the net, so
"certify the walls" becomes checking drawn properties, not hunting bugs.

## The scheduler — the U-arbiter

Who gets the GPU is a deterministic policy at the U place:

1. **Quantum** — a running slice finishes; nothing preempts mid-slice.
2. **Share** — if an exploration W wants U and exploration's share of the
   last M GPU-hours < T (default ⅓): grant the oldest — order is human-only
   **nice**, then age. Nice is best-effort, never a guarantee; only a human
   sets it.
3. **Idle task** — otherwise, train the champion: the one infinite route,
   and it lives here in the arbiter, never in a slip. "Trains constantly"
   is structural.

(Rule 4, admission, moved into R_mint's guard — the Wslot supply *is* the
cap.) **The arbiter never reads a performance number** — Δelo/Δt steers
spend only through R's keep/park decisions. Humans keep one override: a
human-only correction entry in L.

## The researcher and the curator — one at a time, at least once

Both Claude roles run the same discipline: a trigger polls `al` for the head
item (a pure projection — no issuance, no lease, no release), spawns a
session with the dossier in its prompt, and the session's **one binding
write** is idempotent — first wins; a late duplicate is rejected as a
fixable tool result. R's queue: closed slips awaiting a verdict, idea
intake, and — idle — **fresh eyes** (reread the wiki, propose or escalate).
C's queue: landed-uncurated Ls, oldest first. Retries are new mints chained
by prev_id; a chain with N aborted links **dead-letters** to a human. There
is no deadline auto-parker. Backlog is bounded by construction (verdicts are
owed only by ≤ K admitted lanes; champion-cranking owes none).

The researcher authors contracts and intents — never raw anything. Its MCP
shim exposes exactly `propose` and `park`; C's exposes `curate`. Protected
surfaces get tools; sandbox surfaces (the worktree) get ordinary hands.

## Evidence and synthesis

**L is the evidence layer** — per-experiment, sharded, durable at A.
**The wiki is the synthesis layer — the product** — written only by C,
landed only through A, citing only landed content (the prose/intent seam of
A21 is structurally dead: one store, one lander). Lessons are born at C and
*only* at C — every experiment passes through it, whatever its land-status,
so the lessons wall's coverage is complete by construction. The parked and
the failed flow through the same smelter: an `open` lesson is fresh-eyes
bait, not a heap entry.

## Contention — the two serialization points

```
U        physical: GPU time. The real bottleneck. As intended.
G.head   logical: the code CAS. Ls land conflict-free (unique names), so
         canon contention is ~zero for KNOWLEDGE — but each landed code
         change stales every other live W's base: strict I3 converts
         concurrency into recorded-not-landed terminals. Accepted: most
         experiments change recipes/config, not code; the branch+recipe
         make re-landing cheap; and this is the honest price of never
         letting a stochastic actor resolve a merge.
```

Everything else — minting, dispatching, evaluating, curating — is parallel
around the single GPU slot.

## Observability — the board

The board renders projections of git + the ledgers: running, on deck, the
curation-queue depth (a growing queue is a vitals signal — nothing can rot
silently, every uncurated L is a visible token), chains with mounting
aborts, branches of dead experiments, never-matched lessons. Append-only
makes the overnight diff free; the watermark belongs to the viewer.

## Before building — the tabletop, and the clean slate

**The tabletop.** Before any code, run this net **by hand**: a scratch
repo, real Ls, us playing every entity — mint, train (faked), eval, apply,
curate. Domain: **perf research** — experiments take a smoke, not an hour,
so the full mint → land → curate loop runs at conversation speed, and every
vague row schema gets caught by the act of writing it down.

**Clean slate.** Existing ledgers are **not used, not migrated** — mined
for ideas at most. The lab starts fresh.

**Migration, eventually.** Schema-registry-shaped (versioned row schemas +
declared compatibility rules). Noted, not a priority.

## Known unknowns

1. **Locking L down (A17).** During the experiment L lives on W, same uid —
   `echo >>` still works. Landed-into-G is tamper-evident (content-addressed
   history) but in-flight L is convention + harness config + fold-side
   detection. Prevention unsolved; smaller blast radius than before (one
   experiment, not the lab).
2. **Code-CAS cost at velocity.** If the worker lane gets busy, I3's
   strictness manufactures recorded-not-landed terminals. Watch it at the
   tabletop; the relief valve (a serial re-land lane) is known but unbuilt.
3. **Branch GC (A26).** Contained, not solved: branches are namespaced
   caches, rebuildable from the recipe in L; the board surfaces branches of
   dead experiments; sweeping is a manual chore.
4. **C batching.** Single serialized C trades curation quality (cross-
   experiment pattern-spotting) and throughput for simplicity. Policy knob
   on the consumer; revisit with real traces.
5. **The methodology.** Design iteration stored in a synthesis wiki causes
   sprawl (v1, v2, v3 pages are the evidence). Versioned pages are the
   interim answer; rotate the losers out once a vN is blessed.

## Cross-refs

[v2 — design-next](autolab-design-next.md) ·
[v1 — primary design (design-of-record)](autolab-primary-design.md) ·
[red-team A1–A26](autolab-design-adversarial-review.md) ·
[doctrine](autolab-doctrine.md) ·
[as-built architecture](autolab-architecture.md).
