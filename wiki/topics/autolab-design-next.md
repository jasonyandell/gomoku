# The Autolab — design walk-through (working draft)

> **Status: SUPERSEDED** *(same day, 2026-07-05)* — this was v2; the
> walk-through draft is now **[autolab-design-v3.md](autolab-design-v3.md)**
> (the Petri execution model: per-experiment ledgers, the Applicator, the
> curator). Kept for the adjudication trail of the pass-5 red-team
> ([A16–A26](autolab-design-adversarial-review.md)).
> [autolab-primary-design.md](autolab-primary-design.md) (v1) remains
> design-of-record until a vN is blessed.

## One breath

A laptop runs a research lab on itself. Every fact the lab learns is a line
in one append-only ledger — corrections are new entries, never erasures —
and one small tool, `al`, is the only door for anyone who isn't the
substrate. An admitted experiment gets a routing slip: an id and the ordered
**mechanical** steps — train, gate — that dumb loops execute; a scheduler
hands out the GPU hours — exploration gets its share, the champion soaks the
rest. There is no *decide* step: a finished slip simply **lands in the
researcher's queue**, a projection of the fold. One Claude session at a time
takes the head item's dossier — itself a projection, never an artifact — and
submits the one binding write: *propose*, or *park with what we learned*.
Retries are safe (at-least-once; the first decision wins); an item that
keeps failing dead-letters to a human. Findings become one-line lessons and
prose in the wiki; new items may link predecessors, and re-arguing a refuted
lesson without linking it is inadmissible. `al board` shows what's cooking
and what got learned overnight. **The wiki is the product; the player is a
byproduct.**

**The names.** *Autolab* = the method. **`al`** = the one tool. *Ledger* =
the file. *Slip* = an experiment's mechanical plan. *Dossier* = a decision
briefing (a projection). *Board* = the observability view.

## The cast

| Character | Job | Reads → writes |
|---|---|---|
| **Ledger** | one append-only JSONL outside git, corrected like a financial ledger; the only coordination surface | — |
| **Wiki — the product** | where learning compounds; also the durable home of open wounds | all read · researcher writes |
| **Models** | slim per-slice weights on HF + a `champion` tag; buffers stay local | — |
| **Trainer** | flock singleton; ≤1h slice of a pinned commit in a per-SHA uv env | ledger → ledger + HF |
| **Arena** | the protected instrument: H2H **gate** + anchor-pinned **panel** | ledger + HF → ledger |
| **Researcher** | dossier in → one idempotent decision out; prose → wiki | ledger + wiki → ledger (via the shim) + wiki (via git) |
| **Worker** | implements issues; merge SHA = a citable, trainable commit | ledger + issues → ledger + repo |

The daemons **are** the substrate — `al up` *is* them; they share the
plumbing in-process (A16). The one-door rule binds everything **outside**
the substrate: humans and researchers. Zero coupling, visibly: **nobody
reads anybody — everybody reads the ledger.**

## The ledger, and the one door

**Facts, not commands.** One fact = one fsync'd append = one whole
transaction. Balances are derived, never stored — the next open step, the
researcher's queue, the scoreboard are all computed by the fold at read
time. External surfaces (the HF champion tag) are projections, reconciled
after the commit, never read back as authority.

**Corrections** (minimal rule, A24): a correction cites the seq it corrects;
the fold applies corrections in ledger order; a human correction yields only
to a later human correction. The board surfaces dangling and conflicting
corrections — the reconciliation analog.

**One door: `al <verb>`** — `propose · park · result · dossier · board ·
log · status · up · down`. Reads are pure: `dossier` issues nothing, claims
nothing, appends nothing (A18 — it is a projection). The verb set is a
protected instrument. `gomoku/lab/` is plumbing, internal to the substrate.

## The routing slip — mechanical steps only

Admission compiles a proposal's evidence contract into a slip: **an id plus
the ordered mechanical steps** — vocabulary today `{train, gate}`; `panel`
and `implement` join when their producers exist. Every step has a dumb
executor; **judgment never appears in a slip** (A19 — `decide` is deleted
from the vocabulary). A result at step N opens step N+1; `pick(role)` = the
oldest open step that role executes; a slip with no open steps is **closed**
— and every closed slip lands in the researcher's queue *structurally* (a
projection can't be forgotten; stronger than the old "slips always end at
decide" promise).

The slip id is the join key of the whole ledger. Ids are immutable and never
continued; any item may optionally **link predecessors** by id (A23 — this
one primitive replaces `challenges`, carries thread memory, and chains
lesson supersession). New route *shapes* are cheap — same vocabulary,
different order; new step kinds are gated, human-visible changes.

## The researcher — one at a time, at least once

The researcher's queue is a projection of the fold, three item kinds:
**closed slips awaiting a verdict**, **idea intake** (a human or a session
dropped an idea via `al propose`; the dossier is "here's the wiki, here's
the idea — look into it"), and — when the queue is empty — **fresh eyes**,
the researcher's idle task (read the wiki anew; propose or escalate), the
exact symmetry of the scheduler's train-the-champion.

A trigger runs `al dossier` (pure read: the head item's briefing — thread
memory via predecessor links, active lessons, scoreboard, immutable
citations, the wiki doorway; empty queue → clean exit) and spawns a session
with the dossier in its prompt. The session's write path is **one MCP
tool**, a thin shim over `al`, exposing exactly `propose` and `park`. The
decision is **the one binding write** and it is idempotent: it carries its
own citation (dossier hash + evidence-cutoff seq); the first decision for an
item wins; a late or duplicate submit is rejected as a fixable tool result.
Spawning a session appends a **spawn row** — a fact, not a claim: no
exclusivity, no release — which feeds the retry counter: `attempts(item) ≥ N`
with no decision → **dead-letter** (needs-human; the head unclogs). There is
no deadline auto-parker (deleted — a bounded queue drained in order only
stalls via failure → DLQ, or Claude-down → correctly waits).

Backlog is bounded by construction: verdicts are owed only by admitted
exploration lanes (≤ K), and days of champion-cranking owe none.

The researcher authors **contracts and intents — never raw anything**.
Protected surfaces get tools; sandbox surfaces (wiki, repo) get ordinary
hands (git). Daemons never execute archived-commit code for control-plane
functions — archived commits are data.

## The scheduler — four rules

1. **Quantum** — a running slice finishes; nothing preempts mid-slice.
2. **Share** — if an exploration step is open and exploration's share of the
   last M GPU-hours < T (default ⅓): run the oldest admitted exploration
   item. Order = human-only **nice**, then age — nice is best-effort, never
   a guarantee, and only a human sets it (the retired-priority hole stays
   closed).
3. **Idle task** — otherwise, train the champion. "Trains constantly" is
   structural, not promised.
4. **Admission** — ≤ K immature lanes in flight; a proposal passes the
   contract lint (unknown kinds or keys fail loudly), the lessons wall, and
   SMOKE; then it queues.

**The scheduler never reads a performance number** — Δelo/Δt steers spend
only through keep/park decisions at contract boundaries. Humans keep one
override: a human-only correction entry.

## Lessons — never silently re-argue

An adjudicated decision carries a one-line lesson on its intent —
`{scope tags, claim, status, evidence refs, wiki commit+path}` — prose *why*
in the wiki. The dossier always carries the active lessons; admission
rejects a proposal whose scope hits a `refuted` lesson unless it **links
that lesson's item as a predecessor** with a rationale (A23 — the wall is an
admission rule over links, not a special verb). A superseding lesson links
the lesson it replaces; the newest supersession wins, deterministically.
Silent re-proposal is impossible for adjudicated questions; loud
re-litigation is science.

## Observability — the board

One fold, three windows: `pick` (worker), `dossier` (researcher), `board`
(human). `al board` renders the fold — running, on deck, decided, learned
overnight, plus the audits: dangling corrections, never-matched lessons,
items with mounting attempts, branches belonging to dead items. Append-only
makes the overnight diff free (`seq > watermark`; the watermark belongs to
the viewer). Panels are implementation, not design.

## Before building — the tabletop, and the clean slate

**The tabletop.** Before any code, run the design **by hand**: get it out of
heads and simulate it ourselves, step by step — a scratch ledger, real rows,
us playing every character (proposer, admission, scheduler, trainer, arena,
researcher, the board). The domain for the exercise: **perf research**. It's
the perfect first tenant — perf experiments don't take an hour-long GPU
slice, they take a smoke, so the whole loop (propose → admit → slip → steps
→ closed → verdict → lesson) gets exercised at conversation speed, and every
vague row schema gets caught by the act of having to write it down.

**Clean slate.** Any and all existing ledgers are **not to be used or
migrated** — mined for ideas at most. When the lab starts, it starts on a
fresh ledger. (This supersedes A20's era-cut: no coexistence window at all.)

**Migration, eventually.** Once real ledgers accumulate across design
versions, we'll want a migration plan — schema-registry-shaped (versioned
row schemas + declared compatibility rules, à la Kafka's schema registry).
Noted, not a priority.

## Known unknowns

The honest list — gaps acknowledged without a blessed solution, each with
its current best candidate:

1. **Locking the ledger down (A17).** `echo >> ledger.jsonl` works — same
   uid, no substrate user. Whiteboard answer (remote ledger, access via the
   net) trades the gap for hosting + auth. Today's 80%: the spawned
   session's harness config denies the ledger path (the shim is the only
   granted writer), and the fold quarantines rows missing the shim's stamp
   — **detection now, prevention unsolved.**
2. **The prose/intent seam (A21).** Wiki commit and ledger intent are two
   writes to two stores with no spanning transaction; every clean fix is
   two-phase commit in costume. 80%: **push-before-cite** (the shim
   validates cited commits are reachable from origin; a pushed branch
   suffices). Residual: pushed-never-merged prose — folds into #4.
3. **Park's background heap (A22).** Park is necessary (some questions —
   white-defense — resist resolution while still teaching us), but parked
   concepts drift into a heap that's forgotten or becomes maintenance.
   Leading candidate: a park files an **`open`-status lesson** — parked
   knowledge rides the same lessons rail, the wiki holds the wound, and
   fresh-eyes is the structural anti-forgetting loop (open lessons are its
   bait). DLQ, by contrast, settles nothing and files nothing (infra
   failure, not epistemics). **Tradeoff not yet chosen.**
4. **The branch graveyard (A26).** Every dead-lettered or parked item can
   orphan a pushed branch. Containment: lab branches are namespaced
   (`al/<item-id>`) so the board derives "branches of dead items" from the
   fold; sweeping stays a manual chore. No auto-delete.
5. **Settlement paths without lessons (A22 sibling).** A human park via
   correction files no lesson today; candidate: human corrections may carry
   the same lesson payload (a human is an adjudicator). Unblessed.
6. **The methodology itself.** Design iteration is being stored in a wiki
   built for settled synthesis — causing sprawl, staleness, and noise
   (this page is itself evidence). No better home chosen yet; the wiki's
   own curation rules say working notes should rotate out once a page of
   record absorbs them.

## Cross-refs

[Primary design (design-of-record until this is blessed)](autolab-primary-design.md) ·
[red-team A1–A26](autolab-design-adversarial-review.md) ·
[doctrine](autolab-doctrine.md) ·
[DR failure-mode map](autolab-dr-tabletop.md) ·
[as-built architecture](autolab-architecture.md).
