# The Autolab Primary Design — a little lab on a laptop

> **Status: LIVE** *(2026-07-04)* — the **PRIMARY target design** for the autolab:
> the 2026-06 vision completed by the 2026-07-04 design session (the deterministic
> scheduler, the researcher packet, two new ledger walls, the invocation shape),
> then hardened by an [adversarial review](autolab-design-adversarial-review.md)
> whose accepted fixes are folded in below. Where this page and
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
> A chaos simulator proves the walls hold. And whether it ever produces a great
> gomoku player doesn't matter — the product is what we learn and write down.
> The lab is a stable factory for that.

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
| **Worker** | works GitHub issues | **phase-2+** — does not gate the cage |

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

**Δelo/Δt governs less than the slogan implies — deliberately.** Measured
Δelo/Δt is exactly the signal that doesn't exist for young ideas (gate n =
12–40 games; the panel is unbuilt). So the roles split three ways:

- the **scheduler** decides WHEN / HOW MUCH — pool split, ladder rung, age.
  Coarse, humble, noise-free.
- the **researcher** decides WHETHER — keep/park at contract boundaries, where
  Δelo evidence is actually *read*, against a pre-stated falsifier.
- **Δelo/Δt as a number** ranks only *production-pool* lanes and informs
  *rung promotion* — the places where n supports even coarse buckets
  {rising, flat, falling, unknown} (Wilson-gated; small n → `unknown` → fall
  back to ladder + age). Until the panel (#84) exists, this input is nearly
  decorative and the honest v0 scheduler is **ladder + age + contracts** — do
  not pitch it as a bandit yet ([review](autolab-design-adversarial-review.md) A6).

**Review-forced details** (each closes a named attack): the deficit counter
carries **bounded credit** (≈ a few quanta) so an empty exploration queue never
banks a champion-starving catch-up burst (A1); every contract carries a
**decision deadline** — an overdue decision auto-parks via the dumb decider and
escalates `needs_jason`, so a stuck decision can't leak a WIP slot forever
(A2); contracts are **linted at admission** — an evidence kind with no live
producer (e.g. `panel` before #84) is rejected loudly, the Y4 strict-keys rule
generalized (A11); aging uses **coarse hour-buckets** and the pick records its
clock input (A12).

**The v0 minimal cut (A7).** At today's scale (one GPU, a handful of concurrent
ideas) most of the machinery above is scale-insurance. v0 is:

1. champion continuation as the idle task;
2. exploration-first whenever a runnable exploration item exists *and*
   exploration's trailing share < its target (default ⅓), measured over
   result-row `wall_s`;
3. within exploration: ladder rung, then FIFO + age;
4. the verdict-guarantee + decision deadline from the contracts already built.

Deficit sophistication, per-rung weights, and Δelo-ranked production wait for
their named triggers (queue contention; the panel). The policy stays behind the
existing `priority_fn` seam and remains **the single pick policy**
(`actionable()` discipline — never a second one). One free consequence of
determinism worth building: the queue is **forecastable** — "when will my idea
reach its verdict" is computable at propose time and belongs in the packet as a
**schedule projection** (recomputed each fold, labeled an estimate, never a
promise — A3).

The operator keeps an override: a **human-only** correction row that bumps a
rung — audit-visible, outside the LLM's reach.

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
3. **The scoreboard** — per-lane rung, budget spent/remaining, Δelo bucket, and
   the schedule projection (§2).
4. **Immutable citations** — HF *revision + digest* (never the moving
   `champion` tag), wiki *commit + path* (never "current wiki"), telemetry
   snapshot refs. Blessed 2026-07-04.
5. **The wiki doorway** — cross-thread synthesis stays **prose → wiki,
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

## The decisions of record (blessed 2026-07-04)

1. **Scheduler**: deterministic, derby-style, as §2. Researcher-priority is OUT.
2. **Evidence cites immutable identities** — HF revision + digest, git
   commit/path; only the gate moves the champion tag, and the tag-move lands
   AFTER the verdict commit or is idempotent (§1 Wall B; #83).
3. **"Gomocup" = gate + panel, split** (§4; #84).
4. **"Waits" = resume-on-evidence** — nothing sleeps holding a thread.
5. **Worker lane is phase-2+** — does not gate the cage.

## Phase-1 finish line (cage-readiness), in order

The finish line is **traces you can trust**, not walls complete:

1. **The two ledger walls** (§1) + the torn-tail **truncate** fix
   ([DR rows 7–8](autolab-dr-tabletop.md)) — so a trace is never a lie about
   what happened.
2. **Scheduler v0** (§2 minimal cut) — so GPU allocation is replayable and
   Claude-independent.
3. **Packet v0** (§3) — thread memory, refuted trail, immutable refs,
   schedule projection.
4. **The invocation shape** (§3) — OS-permissions wall + the doctrine
   paragraph stating it plainly.

Then phase 2: plug the live Claude `decide=` in and let real traces drive
everything else. **Deferred until traces justify** (the contract's own
discipline): the panel build (#84), worker lane, scouts/reviewer-as-role/
auditor, the research-lesson system. The standing caution: the cage is already
more certified than the animal is real — after this list, resist further
wall-building until real decision traces exist.

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
