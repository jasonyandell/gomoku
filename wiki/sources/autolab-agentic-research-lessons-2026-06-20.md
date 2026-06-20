# Source Record: agentic-research-lab lessons (deep literature pass, 2026-06-20)

**What this is.** A deep, multi-million-token, ~1-hour-wall-clock literature pass
run by **ChatGPT 5.5 Pro** at Jason's request (2026-06-20), reading **our actual
`feat/autolab-sim` branch** alongside the 2025–2026 agentic-research / AutoML-agent
/ self-driving-lab literature. Raw verbatim output preserved at
[autolab-agentic-research-lessons-2026-06-20-raw.md](autolab-agentic-research-lessons-2026-06-20-raw.md).

**Jason's framing (the weight to give it):** *"Don't take any of this as gospel,
especially its specific protocol recommendations. But do take it as solid informed
reasoning and mine it for valuable information."* This page is that mining; the
operationalized synthesis lives in
[../topics/autolab-researcher-contract.md](../topics/autolab-researcher-contract.md).

## Credibility caveats (read before citing)

- **I (Claude) have NOT independently verified the arXiv IDs, venues, or that the
  cited papers say what the pass claims.** Treat the *reasoning patterns and failure
  clusters* as load-bearing; treat the *citations* as unverified pointers.
- The pass **did read our code** — its three concrete corrections (mechanical-vs-
  epistemic WHEN, `Decision.followups` over-authority, uncommitted `uv.lock`) each
  independently match what a separate Explore agent found in the branch. That
  convergence is why its *specifics* earn more weight than a generic survey would.
- The pass self-corrected several bibliographic errors mid-document (e.g.
  AgentEvolver = 2511.10395 not …10915; "Manus whitepaper" is an independent
  overview, not a primary company report) — a mild positive signal on care, but not
  a substitute for verification.
- Most cited material is **very recent preprint** evidence. Per the pass's own
  weighting, the Nature *AI Scientist* publication and Cloudflare's operational
  report deserve more weight than newly-released framework papers; **none is settled
  doctrine.**

## The named sources, by the pass's own weighting

**Most directly load-bearing**
- **AutoLab** (2606.05080) — persistence is *validated research turns*, not uptime;
  large harness effects; **model rankings non-transitive across harnesses**;
  protected evaluators matter. *(This is our own [broken-yardstick reckoning](../topics/alphazero-lessons-15x15-gomoku.md) seen in the literature — H2H overturned the Rapfi rankings; non-transitivity is lived experience here.)*
- **Sibyl-AutoResearch** (2605.22343) — an executable pipeline ≠ good judgment; weak
  evidence becomes polished prose; **failures must convert to changed later
  behavior**, not just textual memories.
- **iML** (2602.13937) — modular executable contracts + recoverability beat
  monolithic code generation.
- **Multi-agent AutoML coordination study** (2603.29632) — isolated parallel scouts
  are robust for breadth; **shared-code specialist teams are operationally fragile**.
- **Agentic Evaluation** (2602.22442) — final score alone hides bad *intermediate*
  decisions (leakage, ungrounded, not counterfactually useful).
- **Cloudflare Glasswing/Mythos** ([blog](https://blog.cloudflare.com/cyber-frontier-models/)) — pointing one agent at a big repo failed; the win was **narrow tasking, executable validation, deduplication, constrained independent review**.

**Strong adjacent**
- **AgentEvolver** (2511.10395) — applicability-conditioned lessons (`when X → do Y`),
  executable task validation, dedup, **breadth-then-depth** with a recent-performance
  window (guard against premature convergence on the historical best).
- **Yunque DeepResearch** (2601.19578) — active-thread detail + compressed
  completed-subgoal memory; **supervisor intervenes on anomaly/stagnation, not on a
  rigid cadence**.
- **AutoLLMResearch / LLMConfig-Gym** (2605.11518) — make **fidelity explicit**;
  learn whether cheap/short evidence predicts expensive outcomes.
- **EvoScientist** (2603.08127) — separate **ideation memory** (directions) from
  **experimentation memory** (techniques that worked/failed).
- **AI Scientist** ([Nature](https://www.nature.com/articles/s41586-026-10265-5)) —
  idea→paper automation is possible, but **human validation + replication seeds
  remained necessary**; a manuscript is not proof the conclusion is correct.
- **Tongyi DeepResearch** (2510.24701) — training specialized long-horizon research
  models + synthetic envs; weaker on lab *governance*.

**Implementation inspiration, not architectural proof**
- **slime** (THUDM) — async producer/trainer separation, common rollout interface
  — *but solves distributed post-training at a scale our single M5 doesn't have;
  importing Ray-style machinery would be cargo-culting.*
- **Mistral Vibe** — minimal CLI core, explicit tools/hooks, resumable sessions,
  **committed dependency lock**; session state ≠ authoritative lab memory.
- **DeerFlow v2** — useful sandbox/worker boundaries, but a *warning*: generic
  agent-orchestration platforms become products unto themselves with their own churn.
- **Alibaba-NLP/DeepResearch** — eval environments + mundane ops failures
  (tool-service latency, QPS limits).

**Lower weight:** AutoML-Agent (2410.02958, limited negative-result evidence);
"Manus overview" (2505.02024, orientation only).

## The seven cross-project failure clusters (the durable part)

These recur across the corpus and are the lessons worth carrying forward:

1. **One giant intelligent loop accumulates sludge** — raw tool traces pile up,
   intent dilutes, errors cascade. → bounded subtasks + compressed memory + a
   supervisor that triggers on anomaly. *Do not hand Claude "the ledger, wiki, W&B,
   repo — go research Gomoku."* Give it **one bounded decision packet, one thread,
   one evidence cutoff.**
2. **Long runtime ≠ persistence** — failures split into premature termination vs.
   budget-exhaustion-with-no-grounded-progress. Measure **validated research turns**;
   every invocation ends in a typed ledger transition or it was wasted.
3. **Completed execution easily becomes an overstated claim** — a pilot becomes a
   general claim. Distinguish maturity: EXECUTED / PILOT_SIGNAL / REPLICATED /
   ADJUDICATED. A slice may justify another slice; it must not auto-justify "it
   worked." *(Our LF1 throughput-runaway is the local, stronger version of this.)*
4. **Memory that doesn't change behavior is decorative** — a lesson must be
   `applies_when`-scoped and *provably* alter a later proposal/gate/scheduler choice.
5. **More agents → more coordination failure** — default to one researcher per
   thread; add scouts/reviewer only when traces show a single bounded researcher
   misses dimensions. Don't stand up a committee preemptively.
6. **Cheap fidelity is useful but dangerous as truth** — proxy wins skew behavior;
   keep reference-based verification; only adjudication changes the conclusion.
7. **Self-evolution needs two speeds** — free *science* in the sandbox; versioned,
   reversible *policy* learning; and a **protected instrument** (evaluator, anchors,
   promotion math, ledger semantics) the model may only propose changes to, never
   silently alter.

## What we took (and what we filtered)

The operationalization — which of these we adopt now, which we defer until we have
real decision traces, and where a recommendation conflicts with our own doctrine
(executable lessons vs. the wiki; minimalism vs. the proposed schema zoo) — is in
[../topics/autolab-researcher-contract.md](../topics/autolab-researcher-contract.md).
The doctrine-level shift (mechanical → epistemic WHEN; three-zone governance) is
folded into [../topics/autolab-doctrine.md](../topics/autolab-doctrine.md).

## Cross-refs
- [../topics/autolab-researcher-contract.md](../topics/autolab-researcher-contract.md) — the design note that mines this for #61.
- [../topics/autolab-doctrine.md](../topics/autolab-doctrine.md) — the doctrine, refined by this pass.
- [sid-bidasaria-stop-babysitting-agents-2026-05-20.md](sid-bidasaria-stop-babysitting-agents-2026-05-20.md) + [../topics/cockpit-vs-autopilot.md](../topics/cockpit-vs-autopilot.md) — the prior agentic-operations source the three-zone governance extends.
