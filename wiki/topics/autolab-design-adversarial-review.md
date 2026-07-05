# Adversarial Review — the Autolab Primary Design

> **Status: HISTORICAL** *(conducted 2026-07-04)* — the red-team of
> [autolab-primary-design.md](autolab-primary-design.md), same session the
> design was banked. Accepted fixes are **folded into that page** (marked
> A1–A12 there); this is the dated record of the attacks and verdicts. Under
> the [Autolab hub](../autolab.md).

**Verdict summary.** Twelve attacks: **5 BROKE** the design as first stated
(each got a fix, folded in), **4 BENT** it (a claim survived only after being
scoped down), **3 HELD** (the design answered, with a residual to watch). The
two most consequential outcomes: the **v0 minimal cut** (A7 — most of the
scheduler machinery is scale-insurance with named triggers, not day-one build)
and the **Δelo/Δt honesty clause** (A6 — until the panel exists, the scheduler
is ladder + age + contracts; don't market a bandit you can't measure).

House rule this page serves (memory: `feedback-challenge-ideas`): a design is
banked with its strongest counter-arguments attached, not as a sales page.

> **Same-day addendum (2026-07-04, the unification pass).** Jason's directive —
> *no parts of the design left for future us* — pushed the design past several
> dispositions below, always by **unifying rather than adding**:
> **A1** (bounded credit) is now structural — the share is computed over a
> bounded window M, so there is no credit to bound. **A6** (Δelo honesty) is
> resolved permanently, not scoped: the scheduler **never** reads a performance
> number; the derby lives in keep/park decisions — one steering channel. **A7**
> (v0 cut with named triggers) is superseded: the simple form is **the** form
> (four rules); deficit counters, rung-priorities and Δelo-ranked pools are
> deleted, not deferred — the ladder is budget vocabulary only, SMOKE moved to
> admission. **A8**'s bound gains a second layer: cross-thread memory rides
> one-line `lesson` rows (design §6), which are bounded by nature. The
> **auditor residual** below is closed: it is the TV's OVERNIGHT/audit panel
> (design §5). All per [autolab-primary-design.md](autolab-primary-design.md).
>
> **Third pass (same day — the routing slip + MCP shape) adds two attacks:**
> **A13 — the slip becomes a workflow engine.** Itineraries invite branches,
> loops, conditions; six months later the slip is a DSL and comprehensibility
> is gone. *Disposition: walled in the design* — slips are linear · finite ·
> end at `decide`; leg vocabulary is a protected instrument; researchers
> author contracts, never raw slips (an LLM writing itineraries = an LLM
> writing the control plane, the retired-priority hole again); the idle task
> is the only infinite route and lives in the scheduler, not a slip.
> **A14 — the MCP server is a new trusted surface.** A bug in the server is a
> bug holding the pen. *Disposition: HELD* — the server adds **no new
> authority**: it wraps the same `validate_intent`/`compile_intent` the sim
> already certifies, runs as the substrate user, and its schema is the
> `DecisionIntent` type; it is *more* testable than the stdout-parsing shape
> it replaces (pytest drives the functions directly). Watch item: the server
> must stay a thin wrapper — logic that creeps into it escapes the sim's
> certification.
>
> **Fourth pass (the porcelain + the clean rewrite) adds one attack:**
> **A15 — the porcelain becomes a kitchen sink.** One-stop CLIs accrete verbs
> until the "one small tool" is a sprawling app and the semantic surface is as
> wide as the raw file was. *Disposition: walled* — the verb set is a
> **protected instrument** (small, semantic; a new verb is a gated,
> human-visible change, same as a new slip leg); plumbing stays in
> `gomoku/lab/`, and the MCP proxy exposes a *subset* of verbs, never a
> superset. The rewrite also moved this page to sole custody of the attack
> record — the design page (`git show c0681dc:…` for the last inline-ref
> version) now states the design without archaeology.

## The attacks

| # | Attack | Verdict | Disposition |
|---|---|---|---|
| **A1** | **Deficit-counter windup.** Exploration queue sits empty for days (champion soaks 100%); counters bank a huge exploration credit; a burst of new proposals then starves the champion for a catch-up marathon. | **BROKE** | Bounded credit: the deficit saturates at a few quanta. Classic leaky-bucket fix; in the design. |
| **A2** | **A stuck decision leaks a WIP slot forever.** The verdict-guarantee shifts starvation to admission: if in-flight lanes never resolve (researcher invocation keeps failing; evidence never arrives), the WIP cap fills with zombies and aging in the admission queue does nothing. | **BROKE** | Every contract carries a **decision deadline**; overdue → dumb-decider auto-parks + `needs_jason` escalation. Auto-park is reversible by a correction row (financial-journal recovery), so a wrong auto-park costs a re-open, not a loss. |
| **A3** | **Forecastability overclaim.** ETAs assume future slice durations and no arrivals; aging and promotions reshuffle the queue; a researcher that *plans* against a promised ETA breaks. | **BENT** | Renamed **schedule projection**: recomputed each fold, deterministic *at that fold*, labeled an estimate in the packet. Never a promise. |
| **A4** | **"Replayable from the log" breaks across code versions.** Fold-*derived* obligations (Wall A) change when derivation code changes; replaying an old ledger under new fold code yields different picks — "entirely deterministic, replayable" is false across upgrades. | **BENT** | Claim scoped: deterministic = same *(fold code, ledger, clock)* → same pick. What **ran** is always plain facts (result rows) — history never needs replaying through new code; only *pending* obligations are derived. Fold/schema version stamped on decision rows. |
| **A5** | **Derived work items need stable identities.** Corrections must target the continuation the fold derived; if the ID scheme shifts, parks silently miss. | **HELD** (with a rule) | Derived ID = pure `f(lane, source-row seq)`, documented as **ledger semantics** — protected-instrument zone, so no autonomous change can move it. |
| **A6** | **Δelo/Δt is nearly decorative in v0.** Gate n = 12–40 games → win-rate CI ±15–25% → elo CI swamps any per-slice trend; the panel that would measure real Δelo is unbuilt (#84). A "Δelo/Δt scheduler" would be ordering lanes by noise. | **BENT** | Honesty clause in the design: v0 scheduling is **ladder + age + contracts**; Δelo buckets are Wilson-gated (small n → `unknown`) and only rank the production pool / inform rung promotion. The panel is the named trigger that makes the signal real. |
| **A7** | **Complexity disproportionate to scale.** Pools + deficit + aging + WIP + ladder for one GPU and ~2–5 concurrent ideas? An OS scheduler serves thousands of threads; this could be round-robin with a champion fallback. | **BROKE** (the day-one scope) | The **v0 minimal cut**: idle-task champion, exploration-first-under-share, rung + FIFO + age, contracts. Each deferred mechanism keeps a *named trigger* (queue contention → deficit counters; panel → Δelo ranking). The full design stays banked as the scaling path, not the build order. |
| **A8** | **Packet growth is unbounded.** The refuted trail grows with history; a giant packet degrades the very decisions it exists to inform (context dilution). | **BROKE** | Packet is **bounded**: top-K recent per thread + per-thread summary + wiki doorway for the long tail. `packet_hash` covers the bounded form, so the purity proof still holds. |
| **A9** | **The write-surface escape hatch.** The OS-permissions wall covers `AUTOLAB_HOME`, but the researcher legitimately writes the **repo and wiki** (prose→wiki is its job) — and proposals cite commits. Can't a proposed commit smuggle code into the control plane? | **HELD** (with a verify task) | The repo is sandbox **by design**: archived commits are *data* — the trainer runs them as GPU workload under the cap; the arena measures their artifacts with protected code. The wall to assert: **daemons never execute archived-commit code for control-plane functions** (fold/pick/gate run from the lab's own install). Currently true by construction (`run_chunk` subprocesses only); needs a stated invariant + audit so a refactor can't quietly break it. |
| **A10** | **Park-while-running still wastes a slice.** No preemption mid-quantum, so a park during a running slice burns up to 1h of GPU on a lane already judged dead. | **HELD** (accepted cost) | ≤1 quantum per park, by design (clean-exit rule forbids mid-slice kills; Metal-wedge scar). Wall A already kills the worse half (the parked lane's *resurrect* via post-park followups). Documented as accepted cycle-waste — "balance, don't chase zero." |
| **A11** | **Contracts can demand evidence nothing produces.** A contract requiring `panel` evidence before #84 exists waits forever; the Y4 scar (a mistyped budget key silently ignored) shows misspelled/unknown kinds fail silent, not loud. | **BROKE** | **Admission-time contract lint**: unknown evidence kinds, kinds with no live producer, and unknown budget keys are rejected loudly at propose time (guesses fail loudly — the vague-harness enemy, mechanized away). |
| **A12** | **Aging makes the pick time-dependent.** `pick(fold, now)` — the clock is a hidden input; two audits at different times "replay" differently, and fine-grained age invites flappy ordering. | **BENT** | Clock admitted as an explicit input (it already was, via lease/ts); aging uses **coarse hour-buckets**; the result row records the pick's clock. Determinism claim reads *(fold code, ledger, clock)* everywhere. |

## What stays on watch (residuals, no fix scheduled)

- **A9's invariant is asserted nowhere yet** — it's true by code shape today.
  Phase-1 item 4 should land the statement + a grep-able audit.
- **A10's accepted waste compounds** if parks become frequent (a thrashing
  researcher). The conversion-audit idea in the
  [contract](autolab-researcher-contract.md) is the watch instrument: parks/
  week is a cheap health metric.
- **A6 rechecks when the panel lands** — the moment real Δelo exists, the
  temptation returns to let it order the exploration pool. It shouldn't:
  noise-humility is a property, not a stopgap.
- **Comprehensibility as a constraint** (the meta-attack): the design survives
  a bar conversation only via the one-breath version. Rule of thumb adopted:
  any *new* mechanism must be justifiable in one sentence naming its failure
  mode, or it waits.

## Cross-refs

- [autolab-primary-design.md](autolab-primary-design.md) — the reviewed design;
  fixes folded in, marked A1–A12.
- [autolab-dr-tabletop.md](autolab-dr-tabletop.md) — the sibling review for the
  *built* system (power-pull failure modes; rows 7–8 landed the same day).
- [autolab-researcher-contract.md](autolab-researcher-contract.md) — the walls
  several attacks lean on (contracts, typed intents, three-zone governance).
- memory: `feedback-challenge-ideas` · `feedback-stateless-delegate-design`.
