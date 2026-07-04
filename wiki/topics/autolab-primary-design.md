# The Autolab Primary Design — a little lab on a laptop

> **Status: LIVE** *(2026-07-04)* — the **PRIMARY target design** for the autolab:
> the 2026-06 vision completed by the 2026-07-04 design session (the deterministic
> scheduler, the researcher packet, two new ledger walls, the invocation shape),
> hardened by an [adversarial review](autolab-design-adversarial-review.md), then
> **unified the same day**: the scheduler collapsed to four rules (elo steers only
> through decisions), the **TV** (§5) and **compounding lessons** (§6) designed in,
> the worker designed (§7) — **the design is complete; nothing is left designed-
> for-later** (implementation follows the build order below). Where this page and
> [autolab-architecture.md](autolab-architecture.md) (the as-BUILT record of
> P1–P7) disagree, **this page wins**. Per-piece built-state is marked inline.
> Under the [Autolab hub](../autolab.md).

## The one-breath version (tell your friends)

> A laptop runs a research lab on itself. Everything the lab knows lives in one
> append-only file. Dumb, deterministic loops — a trainer and a referee — read
> that file, do the next right thing, and append what happened. A scheduler built
> like an OS scheduler, not like an AI, decides which idea gets the next GPU
> hour. Claude is woken only when a decision is actually due, handed a briefing
> packet, and may only answer in a typed form: *propose an idea* or *park one*.
> Settled findings become one-line **lessons** the walls enforce — the lab never
> silently re-argues a dead idea. A one-page **TV** shows what's cooking, what's
> queued, and what got learned overnight. A chaos simulator proves the walls
> hold. And whether it ever produces a great gomoku player doesn't matter — the
> product is what we learn and write down. The lab is a stable factory for that.

Every piece below is individually simple and exists to close a specific, named
failure mode. The composition is what's new, not the parts.

## Why (the point, and the enemy)

The point is the **learning factory**, not the player: *we try, we learn, we
write it down* — the lab exists to make that loop run unattended, with evidence
you can trust days later. The enemy is equally specific (the phase gate, Jason
2026-07-04):

> The failure mode is not models doing badly — it's a **vague harness** letting
> each model's best-effort guess diverge from the last one's. Maximum
> permissiveness for research inside perfect clean efficient walls.

Phase 1 finishes the cage; phase 2 plugs a live Claude in and refines from
**real traces, not anticipated ones**. Throughout: training runs nonstop,
deterministically, on pinned code that cannot conflict with the code under
development.

## The vision, restated (2026-06 original + the 2026-07-04 corrections)

The six pieces, from Jason's original spec — with the two corrections that
completed it:

| Piece | One line | State |
|---|---|---|
| **Ledger** | append-only flatfile outside git, corrected like financial transactions; every loop folds the whole file | BUILT (`gomoku/lab/ledger.py`) — two wall-rules to add (§1) |
| **Models → HF** | slimmed weights per slice + a champion tag; buffers stay local | BUILT — tag becomes a *projection* (§1, rule 2) |
| **Trainer** | guaranteed singleton; ≤1h slices; pick → `git archive` + per-SHA uv env → run → deliver → append | BUILT + LIVE-PROVEN 2026-06-19 |
| **Arena** | mac-native gomocup: cheap H2H **gate** every slice + heavy anchor-pinned **panel** at coarse cadence | gate BUILT; panel DESIGN ([arena lane](autolab-arena-eval-lane.md), #84) |
| **Researcher** | *resume-on-evidence* (never "waits"): evidence contract fires → packet in → typed intent out | walls BUILT ([contract](autolab-researcher-contract.md)); packet + live Claude are phase-1/2 work (§3) |
| **Worker** | works GitHub issues; merge SHA becomes a citable, trainable `commit` | DESIGNED (§7) — built last; does not gate the cage |

**Correction 1 (2026-06-20):** "waits for results" is deleted — nothing sleeps
holding a thread; a fresh invocation folds the ledger when an evidence contract
is satisfied ([doctrine §4](autolab-doctrine.md)).
**Correction 2 (2026-07-04):** **researcher-set `priority` is OUT.** An
LLM-written priority field is an LLM writing directly to the control plane —
the exact hole the typed-intent wall closes everywhere else. WHICH lane gets
the next GPU hour is a deterministic function of the folded ledger (§2);
researchers get exactly two levers — **propose** and **park**.

## §1 — The ledger: facts, not commands (two walls) — DESIGN, fixes a verified RED class

The ledger's commit primitive is one fsync'd row, but the as-built code runs
several *logical* transactions as multiple appends (result + followups;
decision watermark + side-effects) or as an external effect before its commit
(the champion tag move). Every verified crash-window failure on the
[DR failure-mode map](autolab-dr-tabletop.md) (rows 3, 7, 8) is an instance of
one of two doctrinal inversions. Close the inversions, not the instances:

- **Wall A — one fact, one transaction.** A writer appends **facts** (a result,
  a decision); everything the fact *implies* — the continuation, the arena
  eval, the unblock, the escalation — is **derived by the fold**, not appended
  as extra command rows. One fsync'd append IS the whole transaction;
  crash-anywhere leaves either the whole fact or nothing. This also kills the
  park-while-running resurrect (Y2): derivation happens at *read* time, so a
  parked lane simply derives no continuation.
- **Wall B — external surfaces are projections, never authority.** Today the
  arena *reads the champion back from the HF tag* and mutates the tag
  mid-`run_chunk` — authoritative state living outside the fold, at the most
  protected spot in the system. Invert it: **the fold determines the champion**
  (from verdict rows); the HF tag is a projection, reconciled idempotently
  *after* the verdict commit and re-asserted on any restart. Closes #83
  structurally. Rule of thumb: *read your own ledger, never read back your own
  effects.*

Costs, stated honestly (from the [review](autolab-design-adversarial-review.md)):
derived work items need **stable IDs** — `f(lane, source-row seq)`, documented
as ledger semantics (protected-instrument zone) so corrections can target them;
and the determinism claim is **scoped**: same *(fold code, ledger, clock)* →
same pick. What *ran* is always plain facts in result rows; only *pending*
obligations are derived, so history never needs replaying through new code.
Both walls land as sim invariants, falsified RED-when-off: *no multi-row
transaction exists*; *kill between any two appends → derived obligations
unchanged or a prefix*.

## §2 — The scheduler: an OS scheduler, not an AI — DESIGN (v0 cut below)

"I want the trainer to be deterministic, like an OS thread scheduler" (Jason,
2026-07-04). The mapping is closer than analogy — fifty years of scheduler
results (MLFQ, proportional share, aging) drop in directly:

| OS concept | Autolab realization |
|---|---|
| single core | the GPU (`flock` = the core's mutex) |
| run queue | fold-derived open work items (§1 Wall A) |
| quantum | the 1h hard cap; preemption **only** at quantum boundaries (clean-exit rule) |
| context-switch cost | resume-from-`latest.pt` — the accepted cycle waste |
| priority classes | the [maturity ladder](autolab-researcher-contract.md): SMOKE → SCOUT → PILOT → ADJUDICATE |
| proportional share | a fixed production : exploration split of GPU-hours, counted from result-row `wall_s` (no new state) |
| aging | wait-time promotion in the exploration pool — the starvation valve |
| admission control | a WIP cap on immature lanes; excess proposals queue FIFO and age |
| **the idle task** | **the champion continuation** — when nothing else is runnable, train the champion |

The idle-task row is the load-bearing one: **"trains constantly" stops being a
promise the queue must keep and becomes structural** — the idle task is
productive, so the GPU never waits on the exploration queue being non-empty.

**Fairness is denominated in verdicts, not slices.** An OS thread wants CPU
forever; a research lane wants CPU *until its
[evidence contract](autolab-researcher-contract.md) is satisfied*, then it
leaves the run queue for the researcher's desk. The scheduler's guarantee:
**every admitted proposal reaches its first decision point within its declared
budget** — never evicted mid-contract, never granted more without a decision.
"It might take 2 hours to reach a verdict rather than one — fine." The contract
and the ladder quantum are the same object seen from two sides.

**The whole policy is four rules** *(2026-07-04 unification pass — this
replaces the earlier v0-cut + Δelo-bucket scheduler; see the
[review addendum](autolab-design-adversarial-review.md))*:

1. **Quantum:** a running slice finishes; nothing preempts mid-slice
   (clean-exit rule).
2. **Share:** if a runnable exploration item exists **and** exploration's share
   of the last **M** GPU-hours (from result-row `wall_s`; M fixed) is below the
   target **T** (default ⅓) → run the **oldest** admitted exploration item.
   FIFO by admission seq — age *is* the order, so no separate aging mechanism.
   The bounded window makes catch-up bursts structurally impossible (subsumes
   review A1's bounded credit).
3. **Idle task:** otherwise, run the champion continuation.
4. **Admission:** at most **K** immature lanes in flight; a new proposal must
   pass the contract lint (unknown evidence kinds / budget keys / kinds with no
   live producer are rejected loudly — A11), the **lessons wall** (§6), and
   **SMOKE** — which is an *admission check* (does it run, seconds, no GPU
   hour), not a queue item. Then it queues FIFO.

Every contract carries a **decision deadline**: overdue → the dumb decider
auto-parks + `needs_jason` escalation, so a stuck decision can't leak a WIP
slot (A2; auto-park is reversible by a correction row). The clock is an
explicit input — `pick(fold, now)` — and the result row records it (A12).

**Where did Δelo/Δt go? Into the decisions — the derby lives there, not in
queue math.** The scheduler never reads a performance number. Measured Δelo/Δt
steers spend through exactly one channel: **keep/park at contract boundaries**
— production lanes run `continuous` until a decision parks them; exploratory
forks are born BLOCKED until a decision frees them. That is the *same* lever
system humans and the researcher use, so there is one steering channel total,
and it is honest at small n (a judgment call against a pre-stated falsifier,
not a queue ranked by noise). This resolves review A6 permanently rather than
deferring it: elo never enters the pick function, panel or no panel. The
**maturity ladder** likewise stops being a priority class and becomes pure
**budget vocabulary** — SCOUT/PILOT/ADJUDICATE name contract sizes, and rung
promotion is a researcher decision, not a queue effect.

What this consciously gives up: proportional ranking *among* production lanes.
With one GPU and one champion lineage per era that case is empty; if two
production lanes ever contend, FIFO alternates them — accepted. The policy
remains **the single pick policy** behind the existing `priority_fn` seam
(`actionable()` discipline — never a second one).

Determinism's free gift: with FIFO + share the queue is trivially
**forecastable** — "when does my idea reach its verdict" ≈ queue position ×
quantum ÷ T, recomputed each fold. The **schedule projection** goes in the
packet and on [the TV](#5--the-tv-the-dashboard-is-a-third-window-on-the-fold)
(an estimate, never a promise — A3).

The operator keeps an override: a **human-only** correction row that reorders
or re-budgets a lane — audit-visible, outside the LLM's reach.

## §3 — The researcher: packet in, typed intent out — walls BUILT; packet is phase-1 work

The [contract](autolab-researcher-contract.md) built the safety core (evidence
contracts, typed-intent wall, continuation policy, watermarks — all
sim-certified). Two pieces complete the lane:

**The packet (the dossier) is cage-readiness, not presentation** *(2026-07-04
correction — reverses the earlier "presentation, not safety" ranking)*. A vague
or ad-hoc briefing is precisely the one-model-guesses-one-way mess generator;
the packet is the researcher's **entire sensory input**. Keep the built
design's split — `dossier_plan(state, decision_due) → DossierPlan` (**pure**;
provable: rebuilt after total process death ⇒ same hash) then
`hydrate(plan) → DecisionPacket` (**effectful**; records per-artifact
success/failure). The packet carries `{packet_hash, evidence_cutoff_seq,
hydration_status, schema_version}` and the decision cites that exact packet.
Contents, in priority order — this is "set Claude up for success":

1. **Thread memory** — `fold_thread` reconstructs the conversation-with-itself:
   proposal, title-card prediction, slices, verdicts, *its own prior
   rationales*, days apart.
2. **The refuted trail** — what was tried, what killed it, evidence refs —
   **bounded** (top-K recent + per-thread summary; the wiki carries the long
   tail — A8), or the researcher re-proposes its greatest misses forever.
3. **The standing lessons** — every active `lesson` row (§6), one line each:
   the cross-thread half of "react to your own past." The refuted trail is
   *this thread's* memory; the lessons are *the lab's*.
4. **The scoreboard** — per-lane rung, budget spent/remaining, last verdict,
   and the schedule projection (§2).
5. **Immutable citations** — HF *revision + digest* (never the moving
   `champion` tag), wiki *commit + path* (never "current wiki"), telemetry
   snapshot refs. Blessed 2026-07-04.
6. **The wiki doorway** — cross-thread synthesis stays **prose → wiki,
   executable → ledger**; the packet links the pages, it doesn't inline them.

**The invocation shape IS the cage for a real model.** `validate_intent` walls
off a well-behaved callable; an agentic Claude with tools could simply call
`ledger.append`. The practical wall is the invocation: the researcher is handed
a hydrated packet and **returns a typed `DecisionIntent`; it never holds a
ledger-append tool** — only the substrate compiles intents to rows. Enforce it
with the OS, the same way flock is the mutex: the phase-2 researcher process
runs **without write access to `AUTOLAB_HOME`** (packet read in, intent on
stdout). The repo/wiki stay writable — they are sandbox by design; the
corresponding wall is that **daemons never execute archived-commit code for
control-plane functions** (fold/pick/gate run from the lab's own install;
archived commits are data — A9, verify + sim-assert).

## §4 — The arena: gate + panel, protected instrument — gate BUILT, panel DESIGN

Unchanged from the [arena lane design](autolab-arena-eval-lane.md), blessed
2026-07-04: the cheap H2H **gate** vs the champion every slice (built, crowns
via PROMOTE/REVERT/AMBIGUOUS) and the heavy **panel** vs a version-pinned
anchor set at coarse cadence (`eval_kind`, #84). Panel composition + anchor
operating points + the Elo anchor-pin rule are a **protected instrument** —
a researcher may be measured by the yardstick, never move it. The panel is
also what upgrades the scheduler's Δelo input from decorative to real (§2).

## §5 — The TV: the dashboard is a third window on the fold — DESIGN

There is **one fold and three windows onto it**: the worker's (`pick`), the
researcher's (the packet), and the human's (**the TV**). The dashboard is not a
new subsystem — it is `board(state, now, since_seq) → Board`, a pure projection
over the same fold the machines act on, so **watching the TV is watching the
truth** (no second bookkeeping to drift). Refresh = re-fold; zero state; any
renderer (terminal `autolab board`, an auto-refreshing web page riding the
existing `gomoku-web` FastAPI, a phone) draws the same `Board` object.

The panels answer Jason's actual questions:

| "Ooh —" | Panel | Contents |
|---|---|---|
| what's cooking? | **NOW** | the running slice: lane, rung, elapsed vs cap, PID-alive, per-era champion + HF rev |
| what's queued? | **ON DECK** | runnable items in true pick order (the board calls the same `pick`), each with its schedule projection; the admission queue below |
| what needs a brain? | **THE DESK** | decisions due / in-flight; `needs_jason` floats to the top |
| how's everyone doing? | **SCOREBOARD** | per-lane: rung, budget spent/left, last gate verdict, review policy, thread link |
| what got learned overnight? | **OVERNIGHT** | everything since seq N: slices done, verdicts, decisions **with rationales**, lessons filed (§6, linked to their wiki pages), champions crowned, escalations |
| is it healthy? | **VITALS** | daemon locks, last-append age, `health.scan` alerts, GPU tenant status |

**OVERNIGHT is free because the ledger is append-only**: "since I last looked"
is just `seq > watermark` — the watermark is the viewer's (a query param / a
client cookie), never lab state. The same panel doubles as the **conversion
audit** the researcher contract wanted (repeated failures with no lesson,
lessons never consulted, decisions without evidence refs) — the auditor is a
TV panel, not a role.

## §6 — Lessons compound: the lab must never re-argue a settled question — DESIGN

The knowledge loop, end to end. Split by the doctrine's existing rule —
**executable → ledger, prose → wiki** — and consumed deterministically, which
is what earns the ledger half its place:

- **A `lesson` row** (new row type; appended only via `compile_intent` from an
  ADJUDICATE-tier decision, or by Jason):
  `{id, scope: [tags], claim, status: supported|refuted|retired,
  evidence_refs, wiki: commit+path}`. One line, machine-checkable. Corrections
  retire or supersede a lesson like any other row — a lesson is a standing
  *verdict*, not scripture.
- **The prose lives in the wiki** — the run's narrative, the why, the caveats —
  and the lesson row **cites the wiki page at a commit**. The wiki stays the
  compounding synthesis layer; the ledger carries only the one-liner a
  deterministic check can consume.
- **The lessons wall (admission-time, the deterministic consumer).** Every
  proposal declares `scope` tags (hypothesis-class labels — e.g.
  `white-domination`). `validate_intent` **rejects** a proposal whose scope
  intersects a `refuted` lesson — *unless* the intent explicitly carries
  `challenges: <lesson-id>` with a rationale. Silent re-proposal of a dead idea
  is impossible; **deliberate re-litigation is loud, cited, and allowed** —
  that's science, and the escape hatch is what keeps the wall from fossilizing
  a wrong lesson.
- **The packet always carries the active lessons** (they're one-liners, bounded
  by nature) — so the researcher tags its proposals *knowing* the standing
  verdicts, and reacts to its own past instead of repeating it.

Honest limit: the wall is as good as the tagging — a mistagged proposal slips
past. Mitigations, all cheap: the packet shows the lesson list at propose time;
the TV's OVERNIGHT/audit panel surfaces proposals with no scope tags and
lessons that never matched anything; and the reviewer *trigger* (a decision
that smells of self-confirmation) escalates to Jason. The wall converts the
failure from *silent and systemic* to *visible and case-by-case* — that is the
standard the whole cage holds itself to.

## §7 — The worker lane: designed now, built last

No hand-waving left: the worker is the existing `gh` issue flow
(`gh_worktree.py` / ready-queue labels) run as a lane. The researcher may emit
a `propose-work` intent (compiled to a `worker` experiment row + a GitHub
issue); a worker invocation claims the issue, implements in an isolated
worktree, lands via the normal merge gate, and the **merge commit SHA is the
result row** — which makes the new code a *citable, trainable `commit`* for a
follow-up training proposal. That closes the last loop: research → code →
training → evidence, all through the one ledger. Authority boundary unchanged:
the worker **cannot declare scientific success** (only the arena produces
measurements) and cannot touch protected-instrument code paths. It is built
last because nothing else depends on it — not because it is undesigned.

## The decisions of record (blessed 2026-07-04)

1. **Scheduler**: deterministic, as §2. Researcher-priority is OUT.
2. **Evidence cites immutable identities** — HF revision + digest, git
   commit/path; only the gate moves the champion tag, and the tag-move lands
   AFTER the verdict commit or is idempotent (§1 Wall B; #83).
3. **"Gomocup" = gate + panel, split** (§4; #84).
4. **"Waits" = resume-on-evidence** — nothing sleeps holding a thread.
5. **Worker lane is designed (§7), built last** — does not gate the cage.

Added by the same-day unification pass (Jason's directive: *no parts of the
design left for future us — implementation will follow design*):

6. **The scheduler never reads a performance number** — the derby lives in
   keep/park decisions; four rules total; the ladder is budget vocabulary, not
   a priority class; SMOKE is an admission check, not a queue item.
7. **The TV is a design piece** (§5) — one fold, three windows; the auditor is
   a TV panel, not a role.
8. **Lessons are facts** (§6) — `lesson` rows consumed by an admission wall
   with a loud `challenges:` escape; prose compounds in the wiki; the packet
   always carries the standing lessons.

## The build order (the design is complete; implementation is staged)

Nothing below is designed-later — every item is specified above or in its
linked page. The order optimizes for **traces you can trust, then eyes, then
the animal**:

1. **The two ledger walls** (§1) + the torn-tail **truncate** fix
   ([DR rows 7–8](autolab-dr-tabletop.md)) — so a trace is never a lie about
   what happened. Lands with its sim invariants (no multi-row transaction;
   kill-anywhere → obligations unchanged).
2. **The TV** (§5) — `board()` over the existing fold + a terminal and web
   renderer. Cheap, and it is the best debugging instrument for everything
   after it.
3. **The scheduler** (§2, four rules) — replaces the priority-desc pick;
   `actionable()` keeps asserting single-pick-policy agreement.
4. **The packet + lessons** (§3 + §6) — `dossier_plan`/`hydrate`, the `lesson`
   row, the admission lessons-wall, scope tags. Same-hash-after-death purity
   proof; each wall falsified RED-when-off.
5. **The invocation shape** (§3) — OS-permissions wall + the doctrine
   paragraph stating it plainly; the A9 control-plane/data invariant asserted.
6. **The live Claude `decide=`** — phase 2 begins: real experiments, real
   traces; the wiki's lab pages start compounding (§6 prose half).
7. **The panel** (#84, [arena lane](autolab-arena-eval-lane.md)) — the
   absolute-ish yardstick; upgrades what decisions (not the scheduler) see.
8. **The worker lane** (§7) — closes research → code → training.

**Rejected, with reasons — not deferred** (the review's committee-attack,
A7's spirit): *scouts* and *reviewer-as-a-role* (one bounded researcher + the
protected arena **is** the independent check; the reviewer *trigger* on
self-confirmation escalates to Jason); *auditor-as-a-role* (it's the TV's
OVERNIGHT/audit panel, §5); *a separate research-lesson subsystem* (unified
into §6's lesson rows + wiki prose). The standing caution survives: the cage
is more certified than the animal is real — past this list, new walls need a
real trace pointing at them.

## Cross-refs

- [autolab-design-adversarial-review.md](autolab-design-adversarial-review.md) —
  the 2026-07-04 red-team of this page (A1–A12); accepted fixes folded in above.
- [autolab-doctrine.md](autolab-doctrine.md) — the why (this page is its §3–§5
  made concrete); [autolab-architecture.md](autolab-architecture.md) — the
  as-built record this page steers.
- [autolab-researcher-contract.md](autolab-researcher-contract.md) ·
  [autolab-arena-eval-lane.md](autolab-arena-eval-lane.md) ·
  [autolab-dr-tabletop.md](autolab-dr-tabletop.md) — the smart lane, the
  measurement leg, the failure-mode map.
- Issues: #53 (epic) · #61 (researcher lane) · #83 (tag ordering → Wall B) ·
  #84 (panel) · #85 (W&B poisoning, self-heals).
- memory: `feedback-stateless-delegate-design` (the design lens);
  `feedback-challenge-ideas` (why the review page exists).
