# The Autolab — the design

> **Status: LIVE** *(2026-07-04)* — the complete design of record; nothing is
> designed-for-later; where pages disagree, this page wins. Clean rewrite —
> the working notes it distills (three same-day passes, inline review refs):
> `git show c0681dc:wiki/topics/autolab-primary-design.md`. Red-team:
> [autolab-design-adversarial-review.md](autolab-design-adversarial-review.md).
> As-built record: [autolab-architecture.md](autolab-architecture.md).
> Under the [Autolab hub](../autolab.md).

## One breath

A laptop runs a research lab on itself. Everything the lab coordinates through
lives on one long append-only tape. Dumb, deterministic loops — a trainer, a
referee — read the tape, do the next right thing, and append what happened.
Each experiment carries a routing slip saying whose turn is next, so no piece
knows any other exists. A scheduler built like an OS scheduler, not like an
AI, hands out the GPU hours. Claude wakes only when a decision is due, gets a
briefing packet, and answers in typed form: *propose* or *park*. Settled
findings become one-line lessons the walls enforce — the lab never silently
re-argues a dead idea. A one-page TV shows what's cooking and what got learned
overnight. A chaos simulator proves the walls hold. **The wiki is the
product; the player is a byproduct.**

The autolab is its own method — peer of the Derby and the ad-hoc lab, not
built on either. All three compound into the same wiki.

## The cast

| Character | Job | Reads → writes |
|---|---|---|
| **Ledger — the tape** | append-only JSONL outside git, corrected like financial transactions; the only coordination surface | — |
| **Wiki — the product** | where learning compounds | all read · researcher writes |
| **Models** | slim per-slice weights on HF + a `champion` tag; buffers stay local | — |
| **Trainer** | flock singleton; ≤1h slice of a pinned commit in a per-SHA uv env | tape → tape + HF |
| **Arena** | the protected instrument: H2H **gate** + anchor-pinned **panel** | tape + HF → tape |
| **Researcher** | packet in → typed intent out; prose → wiki | tape + wiki → tape (via tool) + wiki (via git) |
| **Worker** | implements issues; merge SHA = a citable, trainable commit | tape + issues → tape + repo |

Zero coupling, visibly: **nobody reads anybody — everybody reads the tape.**

## The tape, and the one tool

**Facts, not commands.** One fact = one fsync'd append = one whole
transaction. Everything a fact implies — the continuation, the eval, the
unblock — is derived by the fold at read time. External surfaces (the HF
champion tag) are projections of the fold, reconciled after the commit, never
read back as authority.

**One door: `autolab <verb>`.** Git-shaped: you stay out of `.git/`, you use
git — same here. The porcelain verbs (`propose · park · challenge · submit ·
result · lesson · packet · board · log · up · down · status`) are the only way
anything touches the tape; `gomoku/lab/` is plumbing, internal to the
substrate. The verb set is a protected instrument: small, semantic, nothing
fancy.

**The routing slip.** Admission compiles a proposal's evidence contract into
an itinerary on the row — `[train, train, gate, decide]`. A result at leg N
opens leg N+1; `pick(role)` = the oldest open leg naming me. Slips are linear,
finite, and always end at `decide`; the leg vocabulary
`{train, gate, panel, implement, decide}` is protected; exceptions — failure,
budget, deadline, park — re-route to `decide` via a correction row. Nothing
invokes anything: **control = data (the slip) + cadence (a dumb tick)**, and
any trigger is interchangeable with any other because the scan is the truth.

## The scheduler — four rules

1. **Quantum** — a running slice finishes; nothing preempts mid-slice.
2. **Share** — if an exploration leg is open and exploration's share of the
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
Determinism makes the queue forecastable: verdict ETAs ride the packet and the
TV. Humans keep one override: a human-only correction row.

## The researcher — nothing raw, ever

The researcher authors **contracts and intents — never raw anything**: not
rows, not slips, not priorities. Its I/O is two MCP tools, a trivial proxy
onto the porcelain — Claude doesn't know the ledger, it knows the tool:

- **`autolab_packet()`** → the hydrated decision packet: thread memory (its
  own prior rationales), the bounded refuted trail, the standing lessons, the
  scoreboard + schedule projection, immutable citations (HF revision+digest,
  wiki commit+path), the wiki doorway.
- **`autolab_submit(intent)`** → validate → compile → append; a reject returns
  as a fixable tool result. A session that never submits can't wedge a thread
  — the deadline auto-parks.

The proxy runs as the substrate and owns the tape's file permissions — the
kernel is the wall, as flock is the mutex. **Protected surfaces get tools;
sandbox surfaces (wiki, repo) get ordinary hands** (git). Daemons never
execute archived-commit code for control-plane functions — archived commits
are data. Packets are pure-then-hydrated (`dossier_plan → hydrate`), carry
`{packet_hash, evidence_cutoff_seq}`, and every decision cites its exact
packet.

## Lessons — never re-argue a settled question

An adjudicated decision files a one-line `lesson` row —
`{scope tags, claim, status, evidence refs, wiki commit+path}` — and the prose
*why* lives in the wiki. Consumption is deterministic: the packet always
carries the active lessons, and admission rejects a proposal whose scope hits
a `refuted` lesson unless it declares `challenges: <id>` with a rationale.
Silent re-proposal is impossible; loud re-litigation is science. The TV's
audit panel surfaces untagged proposals and never-matched lessons.

## The TV

One fold, three windows: `pick` (worker), `packet` (researcher), `board`
(human). Panels: **NOW · ON DECK · THE DESK · SCOREBOARD · OVERNIGHT ·
VITALS**. The overnight diff is free — append-only means "since I last looked"
is `seq > watermark`, and the watermark belongs to the viewer, not the lab.
The audit is a panel, not a role.

## Build order

1. Tape hardening: the two walls, the routing slip, the torn-tail truncate,
   the `autolab` porcelain skeleton — each with its sim invariant.
2. The TV. 3. The scheduler. 4. The packet + lessons. 5. The MCP proxy + the
   permissions wall. 6. **Live Claude — phase 2 begins**; real traces drive
   everything after. 7. The panel (#84). 8. The worker.

Rejected, not deferred: scouts, reviewer-as-role, auditor-as-role (the TV
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
