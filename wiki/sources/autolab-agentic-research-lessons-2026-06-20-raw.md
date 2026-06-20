# Autolab deep pass: research specifics and lessons

# Deep-pass verdict

After reading the specifics, I would change the diagnosis slightly:

**Your doctrine is not missing a richer agent hierarchy. It is missing a formal evidence contract and a controlled path by which lessons alter later behavior.**

The branch already has the right durable substrate:

```text
ledger → fold → actionable work → stateless worker → append
```

That is the important architectural win. fileciteturn30file0L10-L15

But the pasted assessment overstates one point:

> “The open question to decide: `research_threads(state)` — the WHEN exists and is correct.”

The **mechanical WHEN** exists. The **epistemic WHEN** is not yet correct.

On the branch, `research_threads()` becomes actionable when a research lane gains a new terminal training slice. It groups completed or failed train rows by lane and compares their count with the last covered count. It does not appear to ask whether the proposal’s required arena verdict, fixed-anchor result, held-out evaluation, or telemetry window has arrived. fileciteturn32file0L115-L142

So it currently means:

```text
new training result exists
```

not necessarily:

```text
enough of the right evidence exists to make this decision
```

That is the most important subtle correction I found.

---

## Bibliographic corrections

A few items in the reading list need relabeling:

- **AgentEvolver** is arXiv **2511.10395**, not 2511.10915. Its November 2025 paper is from Alibaba’s Tongyi Lab and Shanghai AI Lab. ([arxiv.org](https://arxiv.org/abs/2511.10395))
- **Yunque DeepResearch** is arXiv **2601.19578**, released in January 2026 rather than mid-2026. ([arxiv.org](https://arxiv.org/abs/2601.19578))
- **“From Mind to Machine: The Rise of Manus AI”** is an independent overview paper, not an official Manus company whitepaper. It is useful for orientation but weak evidence for architecture decisions. ([arxiv.org](https://arxiv.org/abs/2505.02024))
- The current iML title is **“A Multi-Agent Framework for Code-Guided, Modular, and Verifiable Automated Machine Learning.”** ([arxiv.org](https://arxiv.org/abs/2602.13937))

Most of the 2026 material here is extremely recent preprint evidence. The Nature AI Scientist publication and Cloudflare’s operational report deserve somewhat more weight than newly released framework papers; none should be treated as settled doctrine.

# What repeatedly failed

Across these projects, the failed approaches cluster remarkably well.

## 1. One giant intelligent loop accumulates sludge

Yunque’s authors explicitly identify problems in monolithic ReAct-style research: raw tool traces accumulate, original intent becomes diluted, minor errors cascade, and the architecture becomes hard to extend. Their answer is active-subgoal detail plus compressed completed-subgoal memory, with a supervisor that intervenes on detected anomaly or stagnation rather than reflecting at a rigid cadence. Removing either the memory mechanism or supervisor materially reduced benchmark results, although the paper also acknowledges that it lacks systematic token and latency analysis. ([arxiv.org](https://arxiv.org/html/2601.19578))

iML reports the analogous coding failure: monolithic code generation creates entangled logic and runtime failures that are difficult to isolate or recover. Its response is modular execution, strict interfaces, profiling-grounded planning, and dynamic contract checking. ([arxiv.org](https://arxiv.org/abs/2602.13937))

Cloudflare’s first, simpler attempt—pointing a general coding agent at a large repository—could produce findings, but it did not provide the meaningful coverage, validation, and signal-to-noise needed for operational security research. Their eventual Glasswing/Mythos harness split reconnaissance, scoped hunting, reproduction, validation, deduplication, gap analysis, tracing, and reporting. ([blog.cloudflare.com](https://blog.cloudflare.com/cyber-frontier-models/))

**Lesson for you:** do not hand Claude the ledger, wiki, W&B, logs, and repository and ask it to “research Gomoku.” Give it one bounded decision packet around one thread and one evidence cutoff.

---

## 2. Long runtime is not the same as persistence

The AutoLab benchmark’s strongest result is not “models need more context” or “models need more agents.” It is that successful models repeatedly interact with the executable benchmark, edit, measure, and continue. Many failures fell into two opposite categories:

```text
premature termination
or
budget exhaustion with little grounded progress
```

The study also found large harness effects: the same model’s score could move substantially under different agent harnesses, and model rankings were not consistently transitive across harnesses. ([arxiv.org](https://arxiv.org/abs/2606.05080))

**Lesson for you:** measure persistence as **validated research turns**, not process uptime, token use, or number of thoughts.

A healthy researcher invocation should produce one of:

```text
proposal
decision
request-for-missing-evidence
human escalation
```

It should not end with pages of analysis but no ledger transition.

Also reserve a finalization phase. Once the invocation approaches its budget, it must stop opening new lines of inquiry and emit a valid typed result.

---

## 3. Completed execution easily becomes an overstated claim

Sibyl-AutoResearch’s central criticism is that an executable pipeline does not guarantee good research judgment. Weak evidence becomes polished prose; a pilot result becomes a general claim; failures remain textual memories that do not modify later behavior. Sibyl’s important proposal is therefore not merely a paper-writing pipeline but **trial-to-behavior conversion**: recurring failures should alter later experiment generation, validation, or harness behavior. The paper is commendably explicit that its traces demonstrate auditable conversion and recoverability, not comparative scientific superiority. ([arxiv.org](https://arxiv.org/abs/2605.22343))

The agentic-evaluation paper reaches a related conclusion from AutoML systems: published systems overwhelmingly evaluate final outcomes but rarely evaluate whether intermediate agent decisions were valid, evidence-grounded, leakage-free, or counterfactually useful. Its observer framework is promising, but still a proof of concept rather than mature evidence that another evaluator agent solves the problem. ([arxiv.org](https://arxiv.org/abs/2602.22442))

The AI Scientist work shows that end-to-end idea-to-paper automation is possible, but human authors still inspected implementations, replication used additional seeds, and producing a manuscript was not itself evidence that the scientific conclusion was correct. ([nature.com](https://www.nature.com/articles/s41586-026-10265-5))

**Lesson for you:** distinguish claim maturity:

```text
EXECUTED
PILOT_SIGNAL
REPLICATED
ADJUDICATED
```

A completed slice may justify another slice. It should not automatically justify “the intervention worked.”

---

## 4. Memory that does not affect behavior is decorative

AgentEvolver specifically rejects random, redundant exploration and unstructured memory. It creates environment profiles, deduplicates generated tasks, tests whether tasks are executable, uses successes and failures, and stores experience in a form resembling:

```text
When this lesson applies
What to do
```

It also uses breadth-first exploration before deeper exploitation and guards against premature convergence by emphasizing a recent performance window rather than blindly following the historical best. ([arxiv.org](https://arxiv.org/abs/2511.10395))

EvoScientist similarly separates:

- **Ideation memory:** feasible, promising, and failed scientific directions.
- **Experimentation memory:** implementation and training techniques that worked or failed.

Its motivating failures include static pipelines repeating failed experiments, missing promising directions, and pursuing infeasible ideas. ([arxiv.org](https://arxiv.org/abs/2603.08127))

**Lesson for you:** a note saying “large buffers diluted learning” is not yet research memory. A useful lesson must say:

```json
{
  "scope": "science",
  "applies_when": "fork increases replay capacity without matching sample consumption",
  "rule": "require post-fill slope and held-out quality before favoring",
  "evidence_refs": ["result:...", "eval:..."],
  "confidence": "medium"
}
```

And later you should be able to prove that the lesson actually changed a proposal, gate, or scheduler choice.

---

## 5. More agents often create more coordination failure

The multi-agent AutoML study compared a single agent, isolated parallel subagents merged afterward, and specialist teams with pre-execution handoffs. Parallel isolated subagents were robust and good at broad, shallow exploration. Specialist teams could make deeper architectural changes when given enough time, but were operationally fragile because multiple agents modified a shared implementation and had to coordinate intermediate assumptions. ([arxiv.org](https://arxiv.org/abs/2603.29632))

Yunque supports specialists, but routes bounded subtasks through a central controller and retains a single structured memory rather than asking several peers to maintain a shared conversational worldview. ([arxiv.org](https://arxiv.org/html/2601.19578))

AutoML-Agent’s earlier design also uses retrieval-assisted planning, parallel specialists, and staged verification, but provides less negative-result evidence than the newer controlled study, so I would treat it as architectural inspiration rather than strong empirical support. ([arxiv.org](https://arxiv.org/abs/2410.02958))

**Lesson for you:**

```text
Default: one researcher deciding one thread.
For breadth: several read-only scouts returning proposals.
For code: one writer per branch.
For high-stakes interpretation: one separate reviewer.
```

Do not create a standing committee of proposer, theorist, critic, statistician, planner, and manager until actual traces show that a single bounded researcher repeatedly misses those dimensions.

---

## 6. Cheap fidelity is useful, but dangerous when mistaken for truth

AutoLLMResearch formalizes LLM configuration research as a long-horizon, multi-fidelity problem. LLMConfig-Gym draws on a large database of training trajectories so an agent can learn whether cheap configurations or short runs predict expensive outcomes, while evaluation checks whether that reasoning transfers across fidelities. ([arxiv.org](https://arxiv.org/abs/2605.11518))

AgentEvolver likewise mixes synthetic tasks with real target-distribution data because optimizing only generated proxy tasks can skew the learned behavior away from the real objective. It treats LLM judging as a fallback proxy and retains reference-based verification where possible. ([arxiv.org](https://arxiv.org/abs/2511.10395))

Your own LF1 throughput runaway is a local, stronger warning than either paper: a recipe won the apparent throughput contest while becoming worse under real wall-clock training dynamics.

**Lesson for you:** make fidelity explicit in every experiment:

```text
SMOKE       Does it run?
SCOUT       Is there enough signal to spend more?
PILOT       Does the effect reproduce over a meaningful regime?
ADJUDICATE  Does it beat the baseline under the protected objective?
```

A smoke result may eliminate a broken proposal. A scout may earn more compute. Only adjudication should change the scientific conclusion or production recipe.

---

## 7. Self-evolution needs two different speeds

Sibyl’s compelling idea is that failures should change future behavior and sometimes the harness itself. But the paper does not establish that unconstrained self-modifying harnesses outperform fixed ones. AutoLab demonstrates that harness changes can dramatically alter outcomes and even model rankings. ([arxiv.org](https://arxiv.org/abs/2605.22343))

That means you need three zones:

### Autonomous science

Claude may freely propose:

- New training configurations.
- New candidate code.
- New branches.
- New experiments.
- New follow-up hypotheses.

### Adaptive research policy

The lab may automatically learn soft rules such as:

- Avoid exact duplicate proposals.
- Run a post-fill check for this family.
- Broaden search after three stagnant decisions.
- Require an arena result before spending another three slices.

These rules must be versioned, reversible, and observable.

### Protected instrument

Changes to these remain separately gated:

- Ledger semantics.
- Runner and sandbox.
- Evaluator.
- Fixed datasets and anchors.
- Promotion math.
- Resource ceilings.
- Human escalation rules.

A model may append a `harness-change-proposal`. It may not silently change the walls against which its own performance is measured.

---

## 8. Generic agent platforms frequently become products unto themselves

DeerFlow v2 is a ground-up rewrite that no longer shares code with its original deep-research framework. It has become a broad “super-agent” platform with subagents, memory, skills, sandboxing, gateway services, web UI, and deployment options. That may be useful software, but it is also evidence that generalized agent orchestration can become a separate product with its own architectural churn. ([github.com](https://github.com/bytedance/deer-flow))

Mistral Vibe offers a cleaner implementation lesson: small CLI core, explicit tools and hooks, resumable sessions, delegated subagents, read-only exploration, and a committed dependency lock. Its session state is useful ergonomics, but should not become your laboratory’s authoritative memory. ([github.com](https://github.com/mistralai/mistral-vibe))

THUDM’s slime is important for a different reason. It decouples training, rollout generation, routing, and buffering, and permits custom tool-using or verifier-driven generation functions behind a common interface. Its asynchronous architecture prevents long-tail generation from serially blocking training. But it solves distributed post-training at a much larger scale than your single M5; importing its Ray-style machinery would be cargo culting. ([github.com](https://github.com/THUDM/slime))

Alibaba’s DeepResearch repository is primarily useful as an evaluation and synthetic-data environment around Tongyi DeepResearch, not as a direct self-driving experimental-lab architecture. Its operational notes also expose mundane but important failure modes such as tool-service latency and QPS limits. ([arxiv.org](https://arxiv.org/abs/2510.24701))

**Lesson for you:** take interfaces and failure lessons, not framework size.

# The architecture I now recommend

## 1. Replace “new evidence exists” with an evidence contract

Every research proposal should declare what makes a decision due.

For example:

```json
{
  "type": "research-proposal",
  "thread_id": "t-freeze-value-head",
  "mode": "confirmatory",
  "question": "Does freezing the value head stabilize the teacher shift?",
  "intervention": "Freeze value-head updates for two slices.",
  "expected_signal": "White loss improves without plies collapse.",
  "falsifier": "No anchored improvement or collapse detector fires.",
  "budget": {
    "max_slices": 2,
    "max_wall_seconds": 7200
  },
  "evidence_contract": {
    "required": [
      {"kind": "train-result", "count": 2},
      {"kind": "arena-verdict", "artifact": "latest"}
    ],
    "optional": [
      {"kind": "training-telemetry"},
      {"kind": "external-anchor"}
    ],
    "decision_cadence": "after-budget"
  }
}
```

An exploratory proposal can be lighter:

```json
{
  "mode": "scout",
  "question": "Does this direction produce any non-noisy signal worth adjudicating?",
  "informative_observations": [
    "clear improvement",
    "clear regression",
    "novel failure signature"
  ],
  "budget": {"max_slices": 1}
}
```

The rule is not “predict everything.” It is:

> **Pre-state why the experiment is informative, what evidence is required, and what ends or escalates it.**

Then:

```python
decision_due(state, thread_id)
```

fires when:

- Required evidence is present.
- The experiment failed and cannot produce it.
- The budget expired.
- A registered kill condition fired.
- A human decision is required.

That is a much stronger deterministic WHEN.

---

## 2. Use a stable thread ID; keep thread state derived

The current `Thread` projection uses a research lane as identity and terminal slices as evidence. fileciteturn31file0L105-L147

Change the relationship to:

```text
thread = research question
lane   = one execution path within that question
```

One thread may contain:

```text
proposal
├── lane A: two training slices
├── lane B: one modified retry
├── arena evaluations
├── external anchor result
├── worker issue
├── decision
└── follow-up proposal
```

Persist `thread_id` on every related event. Do not persist a mutable thread document. Derive:

```python
fold_thread(state, thread_id) -> ThreadView
```

from the ledger.

---

## 3. Make `actionable()` return a decision token, not a giant thread

The unified read surface is a good design. fileciteturn33file0L13-L24

For research, I would return:

```python
DecisionDue(
    thread_id="t-freeze-value-head",
    reason="evidence-contract-satisfied",
    evidence_refs=(...),
    covers_through_seq=1842,
)
```

The evidence watermark matters.

Claude might take several minutes to decide while more results arrive. The resulting decision should state:

```text
I considered evidence through ledger sequence 1842.
```

New evidence at 1843 then creates a later decision point instead of racing ambiguously with the current one.

The present `evidence_n` high-water mark proves idempotence, but exact references and a sequence cutoff are stronger than a count. fileciteturn32file0L101-L142

---

## 4. Split pure dossier planning from effectful hydration

The doctrine calls the lab a pure function over the ledger. Preserve that honestly.

A dossier that fetches W&B, Hugging Face, GitHub, and wiki content is not pure.

Use:

```python
dossier_plan(state, decision_due) -> DossierPlan
```

This pure function identifies:

- Exact ledger rows.
- Exact artifact revisions.
- Exact wiki commit/path references.
- Exact telemetry snapshots.
- Missing required evidence.
- Allowed actions.

Then:

```python
hydrate(plan) -> DecisionPacket
```

dereferences those artifacts and records success or failure.

The packet is saved or hashed:

```text
packet_hash
evidence cutoff
hydration status
schema version
```

The decision cites that exact packet.

Important external information should use immutable identities:

- HF revision and digest, not moving `champion`.
- Git commit/path, not “current wiki.”
- W&B run plus a snapshot or exported artifact, not only a mutable dashboard URL.

---

## 5. Do not allow Claude to append arbitrary ledger rows

The current `Decision` supports arbitrary `followups`, which the reducer appends. That is convenient for a deterministic Python decider, but too much authority for an LLM adapter. fileciteturn32file0L153-L190

Use:

```python
researcher(packet) -> DecisionIntent
```

For example:

```json
{
  "action": "park",
  "evidence_refs": ["result:...", "eval:..."],
  "rationale": "The required two slices regressed and the collapse guard fired.",
  "uncertainty": "low",
  "requested_followups": []
}
```

Then:

```python
validate_intent(packet, intent) -> CommandPlan
```

checks:

- Action is allowed.
- Evidence refs were actually present.
- Budget is respected.
- Board era matches.
- Protected components are untouched.
- Proposed experiment is structurally valid.
- No exact duplicate already exists.

Finally, a deterministic compiler creates corrections, experiments, or escalation rows.

The model proposes meaning. The substrate creates valid ledger commands.

---

## 6. Add lessons, but make them scoped and auditable

Add a derived or explicit `research-lesson` event:

```json
{
  "type": "research-lesson",
  "kind": "science",
  "scope": "replay-buffer-expansion",
  "applies_when": "buffer size is increased without proportional consumption",
  "rule": "require post-fill EPWH and held-out CE before favoring",
  "evidence_refs": ["result:...", "eval:..."],
  "confidence": "medium",
  "status": "candidate"
}
```

Useful lesson kinds:

```text
science
execution
evaluation
orchestration
```

Govern them by scope:

- A same-thread lesson may apply immediately.
- A project-wide heuristic should require repeated evidence or explicit promotion.
- A change to the protected harness requires simulation and human approval.

Then implement a **conversion audit**:

```text
Repeated failure with no candidate lesson?
Active lesson that never affected a later action?
Decision without evidence references?
Lesson contradicted by newer evidence?
Global rule based on one noisy incident?
```

That captures Sibyl’s best idea without adopting an enormous self-editing pipeline.

---

## 7. Make continuation policy explicit

There is another timing issue hiding here.

When a training slice completes, the trainer currently creates a continuation and arena work as part of its flywheel. If the continuation remains immediately runnable, it may consume another hour before the researcher has a chance to park the thread.

A proposal should declare:

```text
review_policy:
  continuous
  after_each_slice
  after_budget
  on_anomaly
```

For an exploratory research fork:

```text
slice completes
→ next continuation is BLOCKED_FOR_DECISION
→ required arena evidence arrives
→ thread becomes actionable
→ decision opens or cancels continuation
```

For a trusted production lane:

```text
continuous
```

This keeps researcher judgment causally upstream of additional research spending rather than merely commenting afterward.

---

## 8. Use a staged and adaptive scheduler

The literature does not support starting with a large hypothesis tree or swarm.

The strongest practical policy is:

```text
focused incumbent exploitation
+ guaranteed challenger allocation
+ broaden after measured stagnation
```

AgentEvolver’s breadth-then-depth approach and the controlled multi-agent/AutoML findings both support adaptive rather than permanently broad search. ([arxiv.org](https://arxiv.org/abs/2511.10395))

For your M5:

```text
production lane gets continuity
challengers get guaranteed bounded slots
adjudication gets protected evaluator time
stagnation increases challenger share
promising evidence deepens one thread
```

A priority may influence scheduling. It should not permanently starve an entire class of work.

Also deduplicate proposals before GPU allocation:

```text
same intervention
same base
same evaluator
same meaningful parameters
```

AgentEvolver and Cloudflare both found deduplication necessary once generation became parallel or long-running. ([arxiv.org](https://arxiv.org/abs/2511.10395))

# Recommended agent topology

| Role | Default shape | Authority |
|---|---|---|
| **Trainer** | Deterministic process worker | Produce artifacts and telemetry |
| **Arena** | Protected deterministic evaluator | Produce measurements and verdicts |
| **Researcher** | One fresh bounded invocation per decision | Interpret and propose typed intents |
| **Scouts** | Optional parallel read-only invocations | Return independent candidate proposals |
| **Reviewer** | Triggered only for ambiguous or high-cost decisions | Challenge interpretation; cannot alter evidence |
| **Auditor** | Periodic retrospective pass | Find bad decisions, unused lessons, wasted compute |
| **Worker** | One writer per branch/issue | Implement code; cannot declare scientific success |
| **Human** | Escalated protected-surface decisions | Change walls, metrics, evaluator, doctrine |

The reviewer should not be universal. Use it when:

- Protected metrics conflict.
- Gate says `AMBIGUOUS`.
- The next experiment is expensive.
- The conclusion is primarily semantic or causal.
- The proposed action changes the instrument.
- Actual traces reveal proposer self-confirmation.

Cloudflare’s independent validator is strongly justified because security findings are noisy semantic claims that require reproduction. Your arena result is often already a stronger independent measurement. ([blog.cloudflare.com](https://blog.cloudflare.com/cyber-frontier-models/))

# Add these simulator invariants

Your simulator already tests the cage rather than Gomoku itself, which is exactly right. fileciteturn30file0L130-L145

I would add:

```text
A decision cannot cover evidence it did not receive.
The same evidence cutoff cannot be decided twice.
New evidence after a decision creates a new actionable point.
A required arena verdict cannot be silently omitted from a packet.
Missing required external evidence causes BLOCKED, not implicit absence.
A research fork marked after_each_slice cannot continue before decision.
An LLM intent cannot directly construct correction or verdict rows.
Only a protected evaluator can produce authoritative evaluation evidence.
A proposal revision cannot retroactively change the meaning of prior evidence.
A global lesson cannot activate without its required promotion rule.
A harness version change is visible in every later result and decision.
The dossier rebuilt after total process death has the same hash.
```

Also commit the `uv.lock`; the branch doctrine itself correctly notes that source is pinned per SHA while dependencies are not yet fully reproducible. fileciteturn30file0L178-L189

# How I weight the named sources

## Most directly load-bearing

- **AutoLab:** persistence, time-management failure, protected evaluators, and enormous harness effects. ([arxiv.org](https://arxiv.org/abs/2606.05080))
- **Sibyl-AutoResearch:** evidence maturity, failure-to-behavior conversion, and the warning that traces do not themselves prove scientific superiority. ([arxiv.org](https://arxiv.org/abs/2605.22343))
- **iML:** modular executable contracts and recoverability over monolithic code generation. ([arxiv.org](https://arxiv.org/abs/2602.13937))
- **Multi-agent coordination study:** isolated breadth can help; coordinated shared-code teams are fragile. ([arxiv.org](https://arxiv.org/abs/2603.29632))
- **Agentic Evaluation:** final score alone does not reveal bad intermediate decisions. ([arxiv.org](https://arxiv.org/abs/2602.22442))
- **Cloudflare Glasswing/Mythos:** narrow tasking, executable validation, deduplication, and constrained independent review. ([blog.cloudflare.com](https://blog.cloudflare.com/cyber-frontier-models/))

## Strong adjacent lessons

- **AgentEvolver:** applicability-conditioned lessons, executable task validation, deduplication, and adaptive breadth/depth. ([arxiv.org](https://arxiv.org/abs/2511.10395))
- **Yunque:** active-thread detail, completed-subgoal compression, and anomaly-triggered intervention. ([arxiv.org](https://arxiv.org/html/2601.19578))
- **AutoLLMResearch / LLMConfig-Gym:** explicit fidelity and learning when cheap evidence predicts expensive outcomes. ([arxiv.org](https://arxiv.org/abs/2605.11518))
- **EvoScientist:** separate memory for scientific directions and implementation techniques. ([arxiv.org](https://arxiv.org/abs/2603.08127))
- **AI Scientist:** replication and end-to-end artifact generation, accompanied by a reminder that human validation remained necessary. ([nature.com](https://www.nature.com/articles/s41586-026-10265-5))
- **Tongyi DeepResearch:** strong evidence about training specialized long-horizon research models and constructing synthetic environments, but less direct evidence about experimental-lab governance. ([arxiv.org](https://arxiv.org/abs/2510.24701))

## Implementation inspiration, not architectural proof

- **slime:** asynchronous producer/trainer separation and a common rollout interface. ([github.com](https://github.com/THUDM/slime))
- **Alibaba-NLP/DeepResearch:** evaluation environments and operational tool-service lessons. ([github.com](https://github.com/Alibaba-NLP/DeepResearch))
- **Mistral Vibe:** minimal tool, hook, subagent, session, and lockfile conventions. ([github.com](https://github.com/mistralai/mistral-vibe))
- **DeerFlow:** useful sandbox and worker boundaries, but also a warning about general-agent platform growth and rewrites. ([github.com](https://github.com/bytedance/deer-flow))

## Lower evidentiary weight

- **AutoML-Agent:** useful early framework ideas, limited controlled negative-result evidence. ([arxiv.org](https://arxiv.org/abs/2410.02958))
- **Manus overview:** orientation only; not an official primary-source system report. ([arxiv.org](https://arxiv.org/abs/2505.02024))

# The doctrine sentence I would use now

> **The autolab is an event-sourced experimental control plane. It deterministically identifies decision points from explicit evidence contracts, reconstructs a bounded and versioned evidence packet, hands that packet to a stateless researcher, validates the researcher’s typed intent against hard walls, and appends the resulting facts and decisions. Research policy may learn from trial and error inside the sandbox; changes to the measuring instrument require separate evidence, simulation, and approval.**

That incorporates the strongest lessons from the field without turning your M5 into a pretend research institute.

The branch’s central direction—stateless workers, resume on evidence, unified `actionable()`, and a simulator-certified cage—is right. The next crucial step is not a richer Claude conversation.

It is to make **“enough evidence has arrived for this exact question”** a first-class deterministic contract, and to make every model decision a constrained, evidence-cited intent rather than arbitrary control-plane output.
