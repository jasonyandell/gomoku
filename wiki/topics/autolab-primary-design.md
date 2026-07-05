# The Autolab — the design

> **Status: LIVE** *(2026-07-04, reframed same day)* — the complete design of
> record; nothing is designed-for-later; where pages disagree, this page wins.
> Prior working notes: `git show c0681dc:wiki/topics/autolab-primary-design.md`
> (three passes, inline review refs); the pre-reframe clean rewrite:
> `git show 7bb410c:wiki/topics/autolab-primary-design.md`. Red-team:
> [autolab-design-adversarial-review.md](autolab-design-adversarial-review.md).
> As-built record: [autolab-architecture.md](autolab-architecture.md).
> Under the [Autolab hub](../autolab.md).

## One breath

A laptop runs a research lab on itself. Every fact the lab learns is a line in
one append-only ledger — corrections are new entries, never erasures — and one
small tool, `al`, is the only door. An admitted experiment gets a routing
slip: an id and the ordered steps that will settle it. Dumb loops read the
ledger and do the next open step; a scheduler hands out the GPU hours —
exploration gets its share, the champion soaks the rest. When a slip reaches
*decide*, a Claude session is spawned with a dossier — the ids, the evidence,
the standing lessons — and answers through the same door: propose or park.
Settled findings become one-line lessons the ledger enforces and prose in the
wiki. `al board` shows what's cooking and what got learned overnight. **The
wiki is the product; the player is a byproduct.**

The autolab is its own method — peer of the Derby and the ad-hoc lab, not
built on either. All three compound into the same wiki.

**The names.** *Autolab* = the method. **`al`** = the one tool. *Ledger* = the
file. *Slip* = an experiment's plan. *Dossier* = a decision briefing.
*Board* = the observability view.

## The cast

| Character | Job | Reads → writes |
|---|---|---|
| **Ledger** | one append-only JSONL outside git, corrected like a financial ledger; the only coordination surface | — |
| **Wiki — the product** | where learning compounds | all read · researcher writes |
| **Models** | slim per-slice weights on HF + a `champion` tag; buffers stay local | — |
| **Trainer** | flock singleton; ≤1h slice of a pinned commit in a per-SHA uv env | ledger → ledger + HF |
| **Arena** | the protected instrument: H2H **gate** + anchor-pinned **panel** | ledger + HF → ledger |
| **Researcher** | dossier in → typed intent out; prose → wiki | ledger + wiki → ledger (via `al`) + wiki (via git) |
| **Worker** | implements issues; merge SHA = a citable, trainable commit | ledger + issues → ledger + repo |

Zero coupling, visibly: **nobody reads anybody — everybody reads the ledger.**

## The ledger, and the one door

**Facts, not commands.** One fact = one fsync'd append = one whole
transaction. Balances are derived, never stored: everything a fact implies —
the next open step, the unblock, the scoreboard — is computed by the fold at
read time. External surfaces (the HF champion tag) are projections of the
fold, reconciled after the commit, never read back as authority.

**One door: `al <verb>`.** Git-shaped: you stay out of `.git/`, you use git —
same here. The porcelain verbs (`propose · park · result · dossier · board ·
log · status · up · down`) are the only way anything touches the ledger;
`gomoku/lab/` is plumbing, internal to the substrate. The verb set is a
protected instrument: small, semantic, nothing fancy. A challenge is
`propose --challenges <slip-id>`; a lesson rides the decide-time intent —
neither earns a verb.

## The routing slip

Admission compiles a proposal's evidence contract into a slip: **an id plus
the ordered steps that will settle the experiment** — e.g.
`[train, train, gate, decide]`. The slip says *what happens*, not whose turn
it is; each step kind has exactly one executor, so "whose turn" is derived.
The slip id is the join key of the whole ledger — results, corrections,
dossiers, and lessons all cite it.

A result at step N opens step N+1; `pick(role)` = the oldest open step that
role executes. Slips are linear, finite, and always end at `decide`. The step
vocabulary is a **closed, protected instrument** — today `{train, gate,
decide}`; `panel` and `implement` join when their producers (the panel #84,
the worker) exist, the same loud-producer rule admission applies to evidence
kinds. New route *shapes* (an eval-only slip) are cheap — same vocabulary,
different order; new *verbs* are gated, human-visible changes. One slip per
admitted proposal: exceptions — failure, budget, deadline, park — re-route
the same slip to `decide` via a correction entry; a challenge is a **new**
slip citing the old one. Nothing invokes anything: **control = data (the
slip) + cadence (a dumb tick)**, and any trigger is interchangeable with any
other because the fold is the truth.

## The scheduler — four rules

1. **Quantum** — a running slice finishes; nothing preempts mid-slice.
2. **Share** — if an exploration step is open and exploration's share of the
   last M GPU-hours < T (default ⅓): run the *oldest* admitted exploration
   item. FIFO — age is the order.
3. **Idle task** — otherwise, train the champion. "Trains constantly" is
   structural, not promised.
4. **Admission** — ≤ K immature lanes in flight; a proposal passes the
   contract lint (unknown kinds or keys fail loudly), the lessons wall, and
   SMOKE (an admission check, not a queue item); then it queues FIFO.

Fairness is denominated in **verdicts, not slices**: every admitted proposal
reaches its decision within its declared budget; an overdue decision
auto-parks and escalates. **The scheduler never reads a performance number** —
Δelo/Δt steers spend only through keep/park decisions at contract boundaries.
Determinism makes the queue forecastable: verdict ETAs ride the dossier and
the board. Humans keep one override: a human-only correction entry.

## The researcher — nothing raw, ever

Claude doesn't wake; it gets spawned. A trigger — a dumb poll to start;
triggers are interchangeable — runs **`al dossier`**, which returns **at most
one** briefing: the oldest open `decide` step (none → clean empty exit, so
the poll costs nothing). Issuing a dossier is itself a ledger entry —
`{slip id, dossier hash, evidence-cutoff seq}` — so every decision cites its
exact briefing and two sessions can't grab the same one. The trigger spawns a
session **with the dossier in its prompt**: thread memory (its own prior
rationales), the bounded refuted trail, the standing lessons, the scoreboard
+ schedule projection, immutable citations (HF revision+digest, wiki
commit+path), the wiki doorway. The dossier carries ids and pointers; the
session reads the wiki for the long tail.

The researcher authors **contracts and intents — never raw anything**: not
entries, not slips, not priorities. Its write path is **one MCP tool**, a
thin shim over `al` exposing exactly `propose` and `park`; validate →
compile → append, and a reject returns as a fixable tool result. A session
that never submits can't wedge a thread — the deadline auto-parks. The shim
runs as the substrate and owns the ledger's file permissions — the kernel is
the wall, as flock is the mutex. **Protected surfaces get tools; sandbox
surfaces (wiki, repo) get ordinary hands** (git). Daemons never execute
archived-commit code for control-plane functions — archived commits are data.

## Lessons — never re-argue a settled question

An adjudicated decision carries a one-line lesson on its intent —
`{scope tags, claim, status, evidence refs, wiki commit+path}` — and the
prose *why* lives in the wiki. Consumption is deterministic: the dossier
always carries the active lessons, and admission rejects a proposal whose
scope hits a `refuted` lesson unless it declares `challenges: <slip-id>` with
a rationale. Silent re-proposal is impossible; loud re-litigation is science.
The board surfaces untagged proposals and never-matched lessons.

## Observability — the board

One fold, three windows: `pick` (worker), `dossier` (researcher), `board`
(human). `al board` renders the fold — what's running, what's on deck, what
got decided and learned overnight. The overnight diff is free — append-only
means "since I last looked" is `seq > watermark`, and the watermark belongs
to the viewer, not the lab. Panels are implementation, not design; the audit
is a view of the board, not a role.

## Build order

1. Ledger hardening: the two walls, the routing slip, the torn-tail truncate,
   the `al` porcelain skeleton — each with its sim invariant.
2. The board. 3. The scheduler. 4. The dossier + lessons. 5. The MCP shim +
   the permissions wall. 6. **Live Claude — phase 2 begins**; real traces
   drive everything after. 7. The panel (#84). 8. The worker.

Rejected, not deferred: scouts, reviewer-as-role, auditor-as-role (the board
audits; the arena is the independent check), any separate lesson subsystem.
Past this list, a new wall needs a real trace pointing at it.

## Cross-refs

[Red-team](autolab-design-adversarial-review.md) ·
[DR failure-mode map](autolab-dr-tabletop.md) ·
[researcher contract](autolab-researcher-contract.md) ·
[arena lane](autolab-arena-eval-lane.md) ·
[doctrine](autolab-doctrine.md) ·
[as-built architecture](autolab-architecture.md) ·
issues #53 · #61 · #83 · #84.
