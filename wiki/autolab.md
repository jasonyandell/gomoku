# The Autolab — the self-driving lab

The machine that runs the research loop **unattended**: one out-of-git **ledger
spine** (a durable log, not in-process state) read by four loops —
**trainer · arena · researcher · worker** — under launchd supervision. The
doctrine in one sentence: **hard walls around a sandbox** — deterministic *WHEN*
(pluggable triggers, never load-bearing), intelligent *WHAT* (a model proposes
meaning; the substrate writes the rows). It is the *autopilot* the
[Derby](derby.md) always wanted: the Δelo/Δt engine that keeps racing while
you sleep.

> **Status: DORMANT (built + proven live 2026-06-19; then stopped).**
> Went **LIVE 2026-06-19** (epic #53, #64) — ran **6 real 9×9 slices** then a
> full **15×15 lane** unattended (0 failures), crowned the **first 9×9 champion**
> (`9x9-champ-recipe@0`) **and** the **first 15×15 champion** (`15x15-wdl@0`,
> internal elo 1918), then was **stopped** (work moved to 15×15 training +
> VCT-science). The measurement/smart/DR lanes below are **design-of-record
> notes** — some built, some not-yet-built; each page's banner says which.

> **← Hubs:** [index](index.md) · sibling hubs: [AlphaZero](alphazero.md) ·
> [Experiments](experiments.md) · [Derby](derby.md) ·
> [M5-as-Mainframe](m5-mainframe.md) · [Reference](reference.md)

## The idea (why it exists)

Read [autolab-doctrine.md](topics/autolab-doctrine.md) first if you're deciding
whether a change *belongs*. The thesis: no state lives in a process — reducers
fold a durable log; loops are **cadence, never load-bearing**; the generic-hard
parts (locking, atomicity, resume) are delegated to hardened tools (flock,
flatfile, HF, uv); and a **sim certifies the walls** rather than a human trusting
them. Everything below is an application of that thesis.

## The triad — trainer trains · researcher researches · arena evals

Three lanes fold onto one ledger spine; a fourth (worker) does the GPU work.

| Lane | Job | Design page | Status |
|---|---|---|---|
| **Trainer** | Run a time-capped GPU slice; append the result row | [autolab-architecture.md](topics/autolab-architecture.md) | **BUILT + LIVE-PROVEN** |
| **Arena** | Register a model → run the gamut → append a *relative* Elo (the measurement leg) | [autolab-arena-eval-lane.md](topics/autolab-arena-eval-lane.md) | **DESIGN-OF-RECORD** |
| **Researcher** | Evidence in → *typed intent* out; the epistemic WHEN + continuation policy (#61) | [autolab-researcher-contract.md](topics/autolab-researcher-contract.md) | **DESIGN-OF-RECORD** |
| **Supervisor** | launchd plists, the seed config/cell/cap, monitor digest + notification (P5–P7) | [autolab-supervisor-and-monitor.md](topics/autolab-supervisor-and-monitor.md) | **BUILT (DORMANT)** |

## Start → Now (chronological)

- **Weekend vision (2026-06):** the **doctrine** was distilled — hard walls
  around a sandbox, reducers over a durable log, the sim certifies the cage
  ([autolab-doctrine.md](topics/autolab-doctrine.md)).
- **Built + LIVE 2026-06-19 (#53/#64):** the ledger-spine architecture (P1–P7)
  ran unattended — 6 real 9×9 slices + a full 15×15 lane, **0 failures**,
  crowned the first 9×9 **and** 15×15 champions. #65 threaded board-size as a
  process-start constant; #67 fixed the arena artifact-ref contract.
- **Design deepened (2026-06):** the **researcher contract** (#61 — evidence
  contract, typed-intent wall, three-zone governance, claim-maturity ladder,
  mining the 2026-06 agentic-research literature) and the **arena eval lane**
  (gomocup-protocol contestant contract, anchor-pinned panel, two-layer
  determinism, cached O(panel) baseline).
- **DR tabletop (2026-06-24):** pulled the power at each table of the
  research→train→eval ring; the torn-line tail-guard was **FIXED (read path** —
  the 2026-07-04 review found the write-path sibling, [DR rows 7–8](topics/autolab-dr-tabletop.md)**)**
  and an end-to-end `triad_resume_under_crash` scenario **BUILT**; #83/#84/#85
  filed for the design-y remainder ([autolab-dr-tabletop.md](topics/autolab-dr-tabletop.md)).
- **Primary design banked (2026-07-04):** the vision completed into a
  [PRIMARY target design](topics/autolab-primary-design.md) — deterministic
  OS-style scheduler (researcher-set priority **retired**; propose/park only),
  the researcher packet promoted to cage-readiness, two ledger walls
  (facts-not-commands; champion-tag-as-projection), the invocation shape — and
  immediately [adversarially reviewed](topics/autolab-design-adversarial-review.md)
  (12 attacks). A same-day **unification pass** then made the design COMPLETE:
  the scheduler collapsed to four rules (elo steers only through keep/park
  decisions), **the TV** (dashboard = a third window on the fold) and
  **compounding lessons** (`lesson` rows + admission wall + wiki prose)
  designed in, the worker designed; scouts/reviewer-as-role/auditor-as-role
  **rejected, not deferred**.
- **Now (DORMANT):** the loop is stopped; the primary design is the relight
  plan. The live-status source of truth for the racing it drove is the
  [Derby hub](derby.md) + [Ops hub](ops.md).

## The pages

| Page | Role |
|---|---|
| [autolab-primary-design.md](topics/autolab-primary-design.md) | **The PRIMARY design (start here)** — the 2026-07-04 target design, COMPLETE (nothing designed-for-later): the four-rule deterministic scheduler, the two ledger walls, the researcher packet, **the TV**, **compounding lessons**, the invocation shape, the worker. Wins over any page it disagrees with. |
| [autolab-design-adversarial-review.md](topics/autolab-design-adversarial-review.md) | **The red-team of the primary design** — 12 attacks, verdicts, what got fixed vs. scoped (A1–A12). |
| [autolab-doctrine.md](topics/autolab-doctrine.md) | **The why** — the thesis that decides whether a change belongs (hard walls around a sandbox). |
| [autolab-architecture.md](topics/autolab-architecture.md) | **The what as BUILT** — the ledger-spine record of P1–P7 (steered by the primary design where they differ). |
| [autolab-supervisor-and-monitor.md](topics/autolab-supervisor-and-monitor.md) | **Operating appendix** — launchd plists, seed cell/cap, monitor digest, `autolab up`/`down` runbook. |
| [autolab-researcher-contract.md](topics/autolab-researcher-contract.md) | **The smart lane (#61)** — evidence in, typed intent out; the epistemic WHEN + continuation policy. |
| [autolab-arena-eval-lane.md](topics/autolab-arena-eval-lane.md) | **The measurement leg** — register a model, run the gamut, read a relative Elo. |
| [autolab-dr-tabletop.md](topics/autolab-dr-tabletop.md) | **Survive weeks unattended** — the power-pull failure-mode map + what the sim certifies. |
| [cockpit-vs-autopilot.md](topics/cockpit-vs-autopilot.md) | **The operating lens** — when to build more autonomy vs. more control. |

## Relationship to the Derby

The **[Derby](derby.md)** is the *charter* (race recipes in time-capped slices,
score by Δelo/Δt, three roles, Reviewer gate). The **autolab is the machine that
runs that charter unattended** — the autopilot. The Derby hub keeps the racing,
scoring, and role pages; this hub owns the self-driving substrate. Live ops
surfaces (GPU queue, bests registry, promotion gate) live on the
**[Ops hub](ops.md)**.
