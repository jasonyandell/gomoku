# The Autolab — design v4 (the Petri model, hardened)

> **Status: WORKING DRAFT v4** *(2026-07-05)* — from-the-top consolidation
> after the v3 red-team session (Jason + Fable, 2026-07-05). Supersedes
> [autolab-design-v3.md](autolab-design-v3.md) as the walk-through draft;
> [autolab-primary-design.md](autolab-primary-design.md) (v1) remains
> design-of-record until a vN is blessed. Red-team record:
> [A1–A26](autolab-design-adversarial-review.md); this session's exchange
> is not yet filed there.

## One breath

A laptop runs a research lab on itself. Truth lives in git: the code in
canon, plus one small append-only **ledger per experiment**, named by its
routing-slip id, landed into canon no matter what. A researcher session
mints a worktree and its ledger together, states its bet before any result
exists, works, submits; the trainer holds the one GPU and **every granted
slice returns a result — CRASHED is a result**, not a happy one, but a
known one; the evaluator scores; a deterministic **Applicator** is the only
thing that touches code and ledgers in main — it *always* lands the ledger,
and lands the code only if git's own merge is clean and the merged canon
still smokes green. A single **curator** smelts each landed ledger into the
wiki through its own door, so even a crashed run becomes knowledge and
nothing rots in a heap. The champion trains **outside the lab entirely** —
the lab only lends it the idle GPU. The only real contention is GPU time;
wrong states can't be drawn. **The wiki is the product; the player is a
byproduct.**

## What changed since v3

The 2026-07-05 red-team pass, all in one place:

1. **Landing guard**: head-equality CAS (`W.base == G.head`) → **git's own
   merge + post-merge canon smoke**. v3's guard was self-defeating: A
   *always* lands every L, and C's prose landed through A too, so every
   knowledge landing advanced `G.head` and staled every live worktree —
   recorded-not-landed would have been the *default* terminal at any
   concurrency. Now: textual conflict is what's structurally barred;
   semantic staleness is an accepted, stated risk (§ Stated risks).
2. **C split fully from A** — three doors into three disjoint path-stores.
   The A21 seam stays dead by disjointness instead of by "one lander".
3. **SMOKE moved from mint to a new `submit` step.** Mint-time SMOKE only
   ever tested the base, which canon already vouches for. Mint gates
   admission; submit gates GPU spend — each guard at its own door.
4. **T always returns a result; CRASHED is a result.** A deterministic
   watchdog is T's writer of last resort: the U token cannot be lost to a
   crash. (It *can* be deliberately parked by a Metal wedge — see the
   watchdog transition.)
5. **E consumes U.** Eval is GPU work in this lab (MCTS + NN inference);
   I4 now covers T and E together.
6. **The champion is out of scope.** Orthogonal, external, untracked by
   the lab; the arbiter's idle rule just lends it the GPU. Its evidence
   lives where it always has (W&B, the training notebook). Bootstrap
   dissolves: no champion process → U idles.
7. **L lands as a fresh commit of its content**, never by surgery on W's
   history — entanglement in the branch can't hold knowledge hostage.
8. **The Wslot is returned by A and only by A** (invariant), and the
   honest consequence is stated: admission liveness is human-guaranteed,
   not structural.
9. **The lessons wall is demoted to best-effort matching** — no ontology.
   Misses are the cheap, self-healing failure; **poisoned lessons** are
   the focused risk, with two concrete guards (§ Evidence and synthesis).
10. **Crashed R is promoted to headline known unknown.** Git can
    enumerate worktrees but cannot decide liveness; the tempting
    deterministic teardown (remove the worktree, let it be re-minted)
    would lose the in-flight L — the one unforgivable loss. Rule until
    solved: **never tear down a Wlive without landing its L first.**

## The entities

| | Entity | Job |
|---|---|---|
| **G** | git canon | truth of code, ledgers, *and* wiki — three path-stores behind three doors; advances by generation (commit) |
| **W** | worktree | a linear token: an isolated checkout minted at base generation `v` |
| **L** | experiment ledger | minted **with** W, named by the unique slip id; append-only; lives on W in flight, lands as content; optional `prev_id` links a predecessor |
| **R** | researcher (Claude) | stochastic proposer: mints, hypothesizes, works in W, submits, verdicts; **never mutates canon** |
| **T** | trainer | the serialized GPU actor; **every granted slice returns a result** |
| **E** | evaluator | scores after T; **also a U consumer** — eval is GPU work here |
| **A** | Applicator | deterministic lander of code + ledgers; **the only thing that returns Wslots** |
| **C** | curator (Claude) | single, serialized: consumes landed ledgers → wiki prose + a lesson; **the closing transition** |
| **U** | the GPU | one holder at a time — T or E |
| **al** | the one tool | sole mutator of any in-flight L; enforces the mint and submit guards |
| **watchdog** | arbiter substrate | T's deterministic writer of last resort: CRASHED is a result |

Three doors, all deterministic substrate: **`al` is the door to L** (during
the experiment); **A is the door to code + ledgers in G**; **`curate` is
the door to the wiki in G**. Stochastic actors author; substrate mutates.
Nobody reads anybody — everybody reads git and the ledgers.

## The slip id — the whole threading mechanism

```
within an experiment : one slip_id, one L, writers (R, T/watchdog, E)
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
R_mint  : guard — a free Wslot (supply = K, the admission cap as token
          supply) AND the mint lint (unknown kinds/keys fail loudly) AND
          the lessons wall (best-effort scope match; a hit on a refuted
          lesson → must link it as prev with a rationale).
          One ATOMIC al operation → Wlive@v AND L[slip_id]
          → append: hypothesis + intended change — the bet is stated
            before any result exists

R_work  : not a transition — sandbox, ordinary hands. R edits W;
          L accretes context through al.

R_submit: guard — SMOKE passes on W AS MODIFIED.
          → append the recipe-as-built; the slip becomes eligible for U.
          (Mint gates admission; submit gates GPU spend. v3's mint-time
          SMOKE tested a pristine checkout of the base — worthless.)

T       : guard — slip submitted AND U granted by the arbiter
          consume U → run slice under the watchdog → produce U
          → append result: completed | failed  (+ checkpoint hash emitted)
          → completed → E's queue; failed → R's queue

watchdog: deterministic, part of the arbiter. A slice that dies or
          overruns quantum + grace → append result: CRASHED → regenerate
          U → emit to R's queue. Kill discipline is wedge-aware: poll for
          compile-idle before killing; an UNKILLABLE process means a
          wedged Metal stack — do NOT regenerate U; dead-letter the
          human (a wedged GPU parks the whole lab by design; nothing
          else could run anyway). A crashed run is ore, same as a
          failed apply.

E       : guard — U granted (short quantum)
          consume U → score → produce U
          → append eval rows, citing the checkpoint hash they measured
          → emit to R's queue

R_close : L holds the full story → verdict: keep (a follow-on mint
          chained by prev_id) or park → release W to A (R does NOT apply)

A       : ALWAYS land L — a FRESH COMMIT of L's content onto canon
          (unique name → conflict-free; entanglement in W's commit
          history can't hold knowledge hostage)                    [I7]
          code: --no-ff merge attempted in a scratch worktree
                merge clean AND merged canon SMOKEs green → land [landed]
                textual conflict → append the conflict manifest
                  (files, both sides' commits) → [recorded-not-landed]
                merge clean, smoke red → append the failure
                  → [recorded-not-landed]
                A never RESOLVES anything: it takes git's clean result
                or refuses. (This repo never rebases or fast-forwards.)
          free the Wslot — A is the ONLY slot-returner            [I1]

C       : consume the oldest landed-but-uncurated L
          → synthesize prose into the wiki + file the one-line lesson
            {scope tags, claim, status ∈ confirmed|refuted|open, refs —
             a lesson MUST cite the L rows that ground it}
          → the curate tool commits to the wiki: C's OWN door, disjoint
            by path from A's stores; C serialized → conflict-free by
            construction
          → terminal: Curated
```

There is **no crashed-R transition** — that is Known Unknown #1, the real
open problem of this design.

Route shapes are free: the queues support eval-only routes (register a
model, run the gamut — submit's guard degenerates to the registration
sanity check) or any reorder of the same closed vocabulary. New transition
*kinds* are gated, human-visible changes.

## Invariants — the wrong states that can't be drawn

```
I1   Wlive + Wslot = 1 per slot; slot supply = K; the slot is returned
     by A and ONLY by A                            -- no orphan, no
                                                      double-mint; WIP cap
                                                      is the token supply
I2   every Wlive carries its base generation v     -- R reasons against
                                                      a pin
I3   A lands code ⟺ the --no-ff merge is textually clean AND the merged
     canon smokes green                            -- stochastic actors
                                                      never RESOLVE a
                                                      merge; A takes
                                                      git's result or
                                                      refuses
I4   U ∈ {0,1} — held by T or E, never both        -- one GPU tenant
I5   L named by unique slip_id; W and L born together in one atomic al
     op                                            -- appends never
                                                      conflict; every
                                                      Wlive has an L
                                                      from birth
I6   three doors, all deterministic substrate: al → in-flight L;
     A → code + ledgers in G; curate → wiki in G   -- disjoint stores;
                                                      nothing stochastic
                                                      mutates any store
I7   L always lands into G, as a fresh commit of its content
                                                   -- knowledge
                                                      unconditionally
                                                      durable, whatever
                                                      happened to the code
I8   every mint reaches exactly ONE terminal: Curated
     (land-status ∈ {landed, recorded-not-landed, parked, aborted} and
      run results ∈ {completed, failed, crashed} are ATTRIBUTES, not
      terminals — a failed apply and a crashed run are ore, not waste)
I9   at most one C in flight                       -- single curator,
                                                      serialized
I10  every granted slice produces a result — the watchdog is T's writer
     of last resort                                -- the U token cannot
                                                      be lost to a crash,
                                                      only parked by a
                                                      wedge, deliberately
```

The sim asserts these directly — they are place invariants of the net, so
"certify the walls" becomes checking drawn properties, not hunting bugs.

## The scheduler — the U-arbiter

Who gets the GPU is a deterministic policy at the U place:

1. **Quantum** — a running slice finishes; nothing preempts mid-slice.
   The quantum's outer bound is now *enforced* by the watchdog, not
   aspirational.
2. **Share** — if slip work (T or E) wants U and exploration's share of
   the last M GPU-hours < T (default ⅓): grant the oldest — order is
   human-only **nice**, then age. Default within the share: E before T
   (evals are short and unblock verdicts; time-to-verdict is the
   denominator of Δelo/Δt). Nice is best-effort, never a guarantee; only
   a human sets it.
3. **Idle task** — otherwise, lend U to the **external champion
   process**. The champion is not the lab's: no slip, no L, no tracking —
   its evidence lives where it always has (W&B, the training notebook).
   No champion process running → U idles. Bootstrap needs nothing.

(Admission lives in R_mint's guard — the Wslot supply *is* the cap.)
**The arbiter never reads a performance number** — Δelo/Δt steers spend
only through R's keep/park verdicts. Humans keep one override: a
human-only correction entry in L.

## The researcher and the curator — one at a time, at least once

Both Claude roles run the same discipline: a trigger polls `al` for the
head item (a pure projection — no issuance, no lease, no release), spawns
a session with the dossier in its prompt, and the session's **one binding
write** is idempotent — first wins; a late duplicate is rejected as a
fixable tool result. R's binding writes: `propose` (mint), `submit`,
`verdict` (keep/park). C's: `curate`. R's queue: runs-complete slips
awaiting a verdict, idea intake, and — idle — **fresh eyes** (reread the
wiki, propose or escalate). C's queue: landed-uncurated Ls, oldest first.
Retries are new mints chained by prev_id; a chain with N aborted links
**dead-letters** to a human. There is no deadline auto-parker. Backlog is
bounded by construction (verdicts are owed only by ≤ K admitted lanes).

The researcher authors contracts and intents — never raw anything.
Protected surfaces get tools; sandbox surfaces (the worktree) get
ordinary hands.

## Evidence and synthesis

**L is the evidence spine** — per-experiment, sharded, durable at A.
Heavyweight stores are *referenced, never protocol-bearing*: L records
W&B run IDs and checkpoint content-hashes (T emits the hash, E cites the
hash it measured); the bytes stay in W&B and the checkpoints directory.
**The wiki is the synthesis layer — the product** — written only by C,
through `curate`, citing only landed content. Lessons are born at C and
*only* at C — every experiment passes through it whatever its attributes,
so a crashed run and a parked idea flow through the same smelter; an
`open` lesson is fresh-eyes bait, not a heap entry.

**The lessons wall is best-effort — say it plainly.** Coverage of lesson
*existence* is complete by construction (everything passes through C);
*enforcement* is a scope-tag match between two stochastic authors, and no
ontology will fix that — the domain is too complex for simple sets. The
asymmetry that makes best-effort acceptable: a **missed match** costs one
duplicate experiment, cheap and self-healing (the rediscovery lands and
gets curated like everything else); a **poisoned lesson** — a wrong
`refuted` — systematically deflects future search and is self-reinforcing,
because nobody re-runs a refuted claim. So the wall tolerates misses and
engineers against poison, with two guards: every lesson must cite the L
rows that ground it (a challenge session is a mechanical audit, not
archaeology), and the board surfaces **load-bearing lessons** — the ones
that have actually fired at a mint gate — because a poisoned lesson only
does damage when it fires, and that small set is where human eyes are
worth spending.

## Contention — the two serialization points

```
U   physical: GPU time, now shared by T and E. The real bottleneck.
    As intended.
G   logical: knowledge and prose land conflict-free BY CONSTRUCTION —
    unique ledger names, disjoint path-stores, serialized C. Code
    contention exists only when two live experiments touch the same
    lines; git detects it, A refuses it, and the re-land is a fresh
    mint. v3's price (every landing staled every base) is gone; the
    residual price is semantic staleness — stated below, not hidden.
```

Everything else — minting, dispatching, curating — is parallel around the
single GPU slot.

## Stated risks — accepted prices, not unknowns

1. **Semantic staleness.** A merge can be textually clean and smoke green
   yet semantically wrong — code validated at base `v` landing onto
   `v+n` it never trained against. Accepted deliberately: the
   alternative (v3's head-equality CAS) manufactured false staleness at
   any concurrency. Backstops: A's post-merge canon smoke, and every
   subsequent experiment runs on the merged truth.
2. **Conflict re-lands cost a full lane.** Textual conflict →
   recorded-not-landed → the retry is a fresh mint where R resolves the
   merge in its own sandbox and revalidates, re-spending GPU. Honest and
   rare — the disjoint stores removed the false positives, so what's
   left is real code-vs-code overlap.
3. **Admission liveness is human-guaranteed, not structural.** No
   deadline auto-parker (deliberate), slots return only at A (invariant),
   and nothing forces an abandoned experiment to A. K abandoned lanes
   freeze admission until a human notices the board. Accepted while K is
   small and the board is looked at daily; the vitals number is
   **age-of-oldest-Wlive**.
4. **The lessons wall misses.** Best-effort matching by design; see
   § Evidence and synthesis for why the tolerated failure is the cheap
   one.

## Observability — the board

The board renders projections of git + the ledgers: running, on deck, the
curation-queue depth (a growing queue is a vitals signal — nothing can rot
silently, every uncurated L is a visible token), **age-of-oldest-Wlive**
(the admission-liveness vitals), chains with mounting aborts, branches of
dead experiments, never-matched lessons, and **load-bearing lessons**
(fired at a mint gate — the poison watchlist). Append-only makes the
overnight diff free; the watermark belongs to the viewer.

## Before building — the tabletop, and the clean slate

**The tabletop.** Before any code, run this net **by hand**: a scratch
repo, real Ls, us playing every entity — mint, submit, train (faked),
eval, apply, curate. **Walk the crash paths deliberately**: kill the
faked T mid-slice and watch the watchdog write CRASHED; abandon an R
session and try to reconcile it; conflict two lanes on purpose and watch
A refuse. Domain: **perf research** — experiments take a smoke, not an
hour, so the full mint → land → curate loop runs at conversation speed,
and every vague row schema gets caught by the act of writing it down.

**Clean slate.** Existing ledgers are **not used, not migrated** — mined
for ideas at most. The lab starts fresh.

**Migration, eventually.** Schema-registry-shaped (versioned row schemas +
declared compatibility rules). Noted, not a priority.

## Known unknowns

1. **Crashed R — the orphaned Wlive.** The headline open problem. Git can
   enumerate Wlives but cannot decide liveness: a session three hours
   into legitimate work, a slip idle between sessions (normal — mint
   session ≠ close session), and a corpse are byte-identical in
   `git worktree list`. The tempting deterministic fix — a guard that
   removes the worktree on crash and lets it be re-minted — **loses the
   in-flight L**, the one unforgivable loss. Rules so far: **never tear
   down a Wlive without landing its L first** (A's fresh-commit-of-content
   makes this always possible while the file exists); reconcile is a
   human-triggered chore for now, not an automatic transition; candidate
   liveness primitive: flock on a slot file — held while a session works,
   released by the kernel on crash, no timestamps, no lease-by-convention.
   Unsolved.
2. **Locking L down in flight (A17).** During the experiment L lives on
   W, same uid — `echo >>` still works. Landed-into-G is tamper-evident
   (content-addressed history) but in-flight L is convention + harness
   config + fold-side detection. Prevention unsolved; blast radius one
   experiment, not the lab.
3. **The execution jail.** T runs R's arbitrary code with T's full
   process authority — filesystem, network, keychain credentials,
   absolute paths to the main checkout. Submit's SMOKE gates *function*,
   not *safety*. The walls of this design guard state stores; execution
   authority has no wall yet. Boundary now named; jail unbuilt.
4. **Eval scheduling policy.** E-on-U is drawn (I4 covers both); quantum
   size, share accounting, E-before-T priority, or folding eval into T's
   slice are knobs for the tabletop.
5. **Branch GC (A26).** Contained, not solved: branches are namespaced
   caches, rebuildable from the recipe in L; the board surfaces branches
   of dead experiments; sweeping is a manual chore.
6. **C batching.** Single serialized C trades curation quality (cross-
   experiment pattern-spotting) and throughput for simplicity. Policy
   knob on the consumer; revisit with real traces.
7. **The methodology.** Design iteration stored in a synthesis wiki
   causes sprawl (v1–v4 pages are the evidence). Versioned pages are the
   interim answer; rotate the losers out once a vN is blessed.

## Cross-refs

[v3 — the Petri model](autolab-design-v3.md) ·
[v2 — design-next](autolab-design-next.md) ·
[v1 — primary design (design-of-record)](autolab-primary-design.md) ·
[red-team A1–A26](autolab-design-adversarial-review.md) ·
[doctrine](autolab-doctrine.md) ·
[as-built architecture](autolab-architecture.md).
