# The Researcher-Claude Contract — evidence in, typed intent out

**What this is.** The design note for the autolab's *smart lane* (issue **#61**):
what researcher-Claude is **handed**, what it is **allowed to emit**, and the
deterministic walls between the two. It is the concrete realization of the
[doctrine](autolab-doctrine.md)'s "intelligent WHAT," and it absorbs the
[2026-06-20 deep literature pass](../sources/autolab-agentic-research-lessons-2026-06-20.md)
(weighted, not gospel — see that source for credibility caveats).

The headline, which **re-orders** an earlier (richer-conversation) sketch:

> The next step is **not** a richer Claude conversation. It is to make *"enough of
> the right evidence has arrived for this exact question"* a first-class
> deterministic contract, and to make every model decision a **constrained,
> evidence-cited intent** rather than arbitrary control-plane output.

A rich dossier and an independent judge without those two walls just let Claude make
a *confident wrong decision on partial evidence and write it straight to the ledger.*
So the walls come first.

## The correction that motivates this page: mechanical vs. epistemic WHEN

The branch's `research.research_threads()` fires when a research lane gains a new
terminal training slice — it groups `result` rows by lane and compares the count to
the last covered count. That is the **mechanical WHEN**:

```
a new training result exists
```

It is *not* the **epistemic WHEN**:

```
enough of the RIGHT evidence exists to decide THIS question
```

A hypothesis that needs *2 slices **and** an arena verdict* would today wake Claude
the instant the 2 slices land — before the verdict exists. (Prior synthesis called
this WHEN "correct"; it is mechanically correct, epistemically not. Corrected here.)

## The load-bearing core (build now)

### 1. The evidence contract — a real `decision_due()`

Every proposal declares **what makes its decision due**. Lightweight: a `required`
list of evidence kinds + a budget + kill conditions. Confirmatory example:

```jsonc
{ "type": "research-proposal", "thread_id": "t-freeze-value-head", "mode": "confirmatory",
  "question": "Does freezing the value head stabilize the teacher shift?",
  "expected_signal": "white loss improves without plies collapse",
  "falsifier": "no anchored improvement OR collapse detector fires",
  "budget": { "max_slices": 2, "max_wall_seconds": 7200 },
  "evidence_contract": {
    "required": [ {"kind":"train-result","count":2}, {"kind":"arena-verdict","artifact":"latest"} ],
    "optional": [ {"kind":"training-telemetry"}, {"kind":"external-anchor"} ],
    "decision_cadence": "after-budget" } }
```

A `scout` proposal is lighter (`max_slices:1`, "any non-noisy signal worth
adjudicating?"). The rule is **not** "predict everything" — it is *pre-state why the
experiment is informative, what evidence is required, and what ends or escalates it.*
Then `decision_due(state, thread_id)` fires only when required evidence is present /
the experiment failed and cannot produce it / budget expired / a kill condition fired
/ a human is required. This is on-doctrine: still a pure fold, still deterministic
WHEN — just the epistemically-correct version. It is the machine-readable form of the
pre-registered title-card already proven in `sliding-derby-composite`.

### 2. The typed-intent boundary — Claude proposes meaning, the substrate compiles commands

Today `research.Decision` carries arbitrary `followups` that the reducer appends —
fine for the deterministic Python decider, **too much authority for an LLM.** Split it:

```
researcher(packet) -> DecisionIntent          # typed: action, evidence_refs, rationale, uncertainty, requested_followups
validate_intent(packet, intent) -> CommandPlan # action allowed? refs actually present? budget respected?
                                               # era matches? protected surfaces untouched? structurally valid? not a dup?
compile(CommandPlan) -> [ledger rows]          # deterministic; the ONLY thing that writes
```

**The model proposes meaning; the substrate creates valid ledger commands.** This is
the technical heart of "LLM-proof cage" — the concrete answer to *how a deterministic
substrate safely accepts a non-deterministic Claude.* No LLM intent may directly
construct a `correction`, `verdict`, or `eval` row.

### 3. Continuation policy — researcher judgment causally upstream of spend

A real cycle-waste bug in our code: the trainer's flywheel enqueues a continuation at
**seed priority**, so the singleton picks it up next tick — potentially burning
another GPU-hour on an exploratory fork *before the researcher can park it.* Fix: a
proposal declares a `review_policy`:

```
continuous        # trusted production lane — keep rolling
after_each_slice  # exploratory fork — continuation is BLOCKED_FOR_DECISION until evidence + decision
after_budget
on_anomaly
```

For an exploratory fork: `slice completes → next continuation BLOCKED_FOR_DECISION →
required evidence arrives → thread actionable → decision opens or cancels the
continuation.` This keeps Claude *upstream* of additional research spend instead of
commenting after the hour is already spent. Directly serves "never waste a cycle."

## Supporting design (the dossier, done honestly)

The dossier I sketched earlier fetched live W&B — which **breaks the pure-fold
property** the doctrine promises. Split it:

```
dossier_plan(state, decision_due) -> DossierPlan   # PURE: exact ledger rows, exact artifact revisions,
                                                   # exact wiki commit/path, telemetry snapshot refs, missing evidence, allowed actions
hydrate(plan) -> DecisionPacket                    # EFFECTFUL: dereference; record per-artifact success/failure
```

The packet carries `{packet_hash, evidence_cutoff_seq, hydration_status,
schema_version}` and the decision **cites that exact packet**. Two consequences worth
the cost:

- **Immutable identities, not moving handles.** Cite the HF **revision + digest**, not
  the moving `champion` tag; a git **commit/path**, not "current wiki"; a W&B run **+
  exported snapshot**, not a mutable dashboard URL. *(This exposes a latent bug: an
  arena decision citing the `champion` tag isn't reproducible — the tag can move
  between hydration and audit.)*
- **Evidence watermark.** `actionable()` returns a small `DecisionDue(thread_id,
  reason, evidence_refs, covers_through_seq)` — **not** a giant thread. Claude may take
  minutes to decide while results arrive; its decision states *"considered evidence
  through seq N,"* so an append at N+1 creates a **later** decision point instead of
  racing the current one. (Our count-based high-water mark proves idempotence; exact
  refs + a seq cutoff are strictly stronger under concurrent appends — and our whole
  architecture is async.)

**Thread ≠ lane.** Make `thread` = the research *question* and `lane` = one execution
path within it; persist `thread_id` on every related event; never persist a mutable
thread document; derive `fold_thread(state, thread_id) -> ThreadView`. One thread can
then own lane-A, a modified lane-B retry, arena evals, a worker issue, the decision,
and a follow-up — all reconstructable, all stateless.

## Three-zone governance (upgrades "hard walls around a sandbox")

The pass adds the missing *middle* zone between our hard walls and wild sandbox:

| Zone | What lives here | Who may change it |
|---|---|---|
| **Autonomous science** | new configs, candidate code, branches, experiments, follow-up hypotheses | Claude, freely |
| **Adaptive research policy** | soft rules: "dedup exact proposals", "run a post-fill check for family X", "broaden after 3 stagnant decisions", "require an arena result before 3 more slices" | the lab may *learn* them — **versioned, reversible, observable** |
| **Protected instrument** | ledger semantics, runner/sandbox, evaluator, fixed datasets & anchors, promotion math, resource ceilings, escalation rules | **separately gated** — a model may append a `harness-change-proposal`; it may **not** silently change the walls its own performance is measured against |

## Claim-maturity / fidelity ladder

Make fidelity explicit on every experiment and never let a cheap tier carry an
expensive conclusion (this is the formal version of our LF1 / bigbuf / small-n scars):

```
SMOKE       does it run?                  -> may eliminate a broken proposal
SCOUT       enough signal to spend more?  -> may earn more compute
PILOT       reproduces over a regime?     -> may justify another slice
ADJUDICATE  beats baseline under the protected objective?  -> the ONLY tier that changes the recipe/conclusion
```

## Lessons: executable → ledger, prose → wiki

The pass wants a `research-lesson` event (`{scope, applies_when, rule, evidence_refs,
confidence, status}`) so failures change later behavior. This **partly conflicts with
our doctrine** that synthesis lives in the wiki, not scattered memory (Jason,
2026-06-16). Resolution:

> A ledger lesson earns its place **only if a deterministic check consumes it** (a gate
> pre-filter, a dedup rule, a scheduler nudge). **Executable → ledger; prose → wiki.**

The pass's **conversion audit** is the guard that keeps it honest (and doubles as the
guard against decorative memory): periodically flag *repeated failure with no candidate
lesson · an active lesson that never changed an action · a decision with no evidence
refs · a lesson contradicted by newer evidence · a global rule from one noisy incident.*

## Sim invariants to add (the operationalization)

Our [simulator](autolab-doctrine.md) already certifies the cage, not Gomoku. Each
recommendation above becomes a new wall it can assert (falsified RED-when-off, per
house practice):

- A decision **cannot cover evidence it did not receive**.
- The **same evidence cutoff cannot be decided twice**; new evidence after a decision
  creates a **new** actionable point.
- A **required arena verdict cannot be silently omitted** from a packet; missing
  required external evidence yields **BLOCKED**, not implicit absence.
- A fork marked `after_each_slice` **cannot continue before its decision**.
- An **LLM intent cannot directly construct** `correction`/`verdict`/`eval` rows; only
  the **protected evaluator** produces authoritative evaluation evidence.
- A proposal revision **cannot retroactively change the meaning** of prior evidence.
- A global lesson **cannot activate without its promotion rule**; a harness version
  change is **visible in every later result and decision**.
- **The dossier rebuilt after total process death has the same hash.** *(The purity
  capstone — and the cheapest proof that `dossier_plan` is honestly pure.)*

## Agent topology — adopt the authority boundaries, defer the roles

| Role | Shape | Authority |
|---|---|---|
| **Trainer** | deterministic worker | produce artifacts + telemetry |
| **Arena** | protected evaluator | produce measurements + verdicts |
| **Researcher** | one fresh bounded invocation per decision | interpret + propose **typed intents** |
| Scouts | optional parallel read-only | return independent candidate proposals |
| Reviewer | triggered, not universal | challenge interpretation; **cannot alter evidence** |
| Auditor | periodic retrospective | find bad decisions, unused lessons, wasted compute |
| **Worker** | one writer per branch/issue | implement code; **cannot declare scientific success** |
| **Human** | escalated | change walls, metrics, evaluator, doctrine |

Adopt the **boundaries** now (worker can't declare success; researcher emits intents
not rows; reviewer is *triggered* — on AMBIGUOUS gate / expensive next experiment /
semantic-causal conclusion / instrument-changing action / detected self-confirmation).
**Defer the extra roles** (scouts, reviewer-as-role, auditor) until real decision
traces show one bounded researcher misses those dimensions — building the committee
first is exactly failure-cluster #5. Note too: *our arena result is already a stronger
independent measurement than an LLM reviewer* for most decisions.

## Build order

1. **Evidence contract** (`decision_due`) — fixes the epistemic WHEN. *Highest value.*
2. **Typed-intent boundary** (`researcher → validate_intent → compile`) — makes Claude's
   output safe to append.
3. **Continuation policy** (`review_policy`) — stops the flywheel out-spending judgment.
4. **`dossier_plan`/`hydrate`** with immutable identities + the evidence watermark.
5. Each of the above lands **with its sim invariant**.

**Deferred until traces justify:** the `research-lesson` system (start executable-only),
the conversion audit as a role, scouts, reviewer-as-role, the auditor, and the full
maturity *state machine* (use the vocabulary first). Adopting the whole schema zoo at
once would violate both our minimalism and the pass's own failure-cluster #8 ("take
interfaces and failure lessons, not framework size").

## The doctrine sentence, upgraded

> **The autolab is an event-sourced experimental control plane. It deterministically
> identifies decision points from explicit evidence contracts, reconstructs a bounded
> and versioned evidence packet, hands that packet to a stateless researcher, validates
> the researcher's typed intent against hard walls, and appends the resulting facts and
> decisions. Research policy may learn from trial and error inside the sandbox; changes
> to the measuring instrument require separate evidence, simulation, and approval.**

A legitimate upgrade to the doctrine's one-sentence thesis — it adds the evidence
contract, the versioned packet, the typed-intent wall, and the three-zone learning
governance, without turning a single M5 into a pretend research institute.

## Cross-refs
- [../sources/autolab-agentic-research-lessons-2026-06-20.md](../sources/autolab-agentic-research-lessons-2026-06-20.md) — the literature pass mined here (+ raw verbatim).
- [autolab-doctrine.md](autolab-doctrine.md) — the *why* (refined by this pass).
- [autolab-architecture.md](autolab-architecture.md) — the *what* (ledger spine, the four lanes).
- [cockpit-vs-autopilot.md](cockpit-vs-autopilot.md) — the supervisability lens the three-zone governance extends.
- Issue **#61** (research lane) — this page is its design pass.
