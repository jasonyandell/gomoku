# The Idea Pile — research seeds for the autolab

> **Status: LIVE (seed pile).** Un-run seeds below stay open. **Dated banner
> (2026-07-04) — settled/superseded through-lines:** the seeding framing (*"stop
> treating Rapfi-at-full-strength as a fixed wall to imitate"*, 2026-06-25) predates
> the **closed 9×9 chapter** and the **sound-world recipe** — read the Rapfi-wall
> ideas (#1/#3/#4/#7) as *historically motivated*, not the current front. Three ideas
> have **graduated** and are now one-line stubs + verdicts (full narratives were
> moved to a separate page and have since been removed *(2026-07-04; recover:
> `git show ca76350:wiki/_archive/topics/idea-pile-graduated-results.md`)*):
> **#9** batch-VCF → BUILT (GPU-VCF/VCT oracle, real-time); **#10** threat-molecules →
> thesis alive but first probe NEGATIVE, big swing un-built; **#11** VCT-terminus →
> throughput-win/robustness-loss, **superseded by [sound-world-recipe.md](sound-world-recipe.md)**.

A parking lot for **wild-but-grounded research directions**: bigger swings than a
single derby cell (which races *training recipes* on the [research board](../ops/research-board.md)).
Each entry is a candidate the **autolab** can pick up and give a real shot — so each
carries enough mechanism, a rough cost, and *how we'd know it worked* to be runnable
without re-deriving the idea. Add freely; promote to a `derby-idea` issue or an autolab
lane when one earns a run. Seeded 2026-06-25 from a brainstorm; the through-line of that
batch: **stop treating Rapfi-at-full-strength as a fixed wall to imitate** — every good
idea breaks that frame.

Grounding facts these lean on (see [eval-teacher-sensei](eval-teacher-sensei.md),
[rapfi-idx2-distillation-mine](rapfi-idx2-distillation-mine.md), `TRAINING_WIKI`
2026-06-25): think-**time** is Rapfi's strength dial (node budget never reaches the wall —
even 2M nodes loses to a matured net); the net at sims=32 beats rapfi@25ms, loses at 50ms+;
the 0.5-crossing migrates up the ms-ladder as the net climbs; Rapfi exposes only a *move*
(no value/policy) but emits a per-move winrate map (node-bounded) and a principal variation.

---

## Active thread: idx-2 DAgger
The live experiment (`gomoku/rapfimine/dagger.py`): on-book DAgger from ep250 — roll out the
net vs Rapfi (both seats), relabel visited states with the master, aggregate, re-fit by pure
supervised imitation, gate vs the frozen parent. Loop is built + perf-passed; a history-less
train/inference bug was found+fixed (TRAINING_WIKI 2026-06-25). Several ideas below
*compose* with it (esp. #1, #4, #5).

---

## The pile

### 1. Out-*search* yourself, then distill it back  ⭐ (highest-upside)
The net loses to rapfi@1000ms at sims=32 — but sims=32 is almost nothing. Give the net a
**huge test-time budget at idx-2 only** (sims=5k+, or MCTS with a VCF/threat solver in the
tree), find the lines where deep-net-search beats Rapfi, and **distill those policies into
the fast net**. This is AlphaZero's own expert-iteration engine, but the "expert" is the net
thinking 100× harder. Hypothesis: the knowledge to beat Rapfi may already be *in* the net;
search surfaces it, distillation makes it cheap.
- **Cost:** medium. Reuses MCTS + the distillation pipeline; the new piece is a heavy-search
  picker + (optionally) an in-tree VCF solver.
- **Measure:** does the distilled fast net's ms-ladder crossing climb (idea #4)? Gate vs frozen parent.
- **Composes with:** DAgger (swap the Rapfi label for the net's-own-deep-search label).

### 2. Actually *solve* idx-2  ⭐ (most on-thesis)
Freestyle gomoku (no forbidden moves) is almost certainly a first-player win, and we're pinned
to ONE opening. **Allis-style threat-space / VCF-VCT search + a transposition table** over the
idx-2 subtree → a (partial) **tablebase**: a *perfect* teacher with true values, not Rapfi's
node-bounded guesses. The most literal reading of "practice one kick 10,000 times" is to
*solve the kick*.
- **Cost:** high but bounded by the narrow tree; threat-space search is the classic tool and
  CPU-only (composes with the warm-pool/co-tenancy model).
- **Measure:** solved-fraction of the idx-2 subtree; distill solved values → does the net
  inherit them?

### 3. Judo — steer into Rapfi's blind spots
Rapfi-NNUE has weights, so it has systematic mis-evaluations. Search for idx-2 continuations
where Rapfi's claimed winrate **diverges from deep-search truth**, and train the net to
*navigate the game toward that terrain*. You don't have to be stronger everywhere — just
better at dragging the master onto a map where his map is wrong. (This is genuinely how
weaker engines upset stronger ones.)
- **Cost:** medium. Needs a "truth" oracle (deep net search or #2) to diff against Rapfi.
- **Measure:** win-rate vs Rapfi specifically from blind-spot-seeded lines.

### 4. The think-time ladder as a self-paced sparring partner  ⭐ (the metric that makes the rest measurable)
Stop fighting the 0/48 wall at rapfi@1000ms. Make the opponent's **think-time track the net's
frontier** (always face a Rapfi *just* above your level), and make the north-star metric
**"push the 0.5-crossing from 50ms → 1000ms."** Turns a brutal binary wall into a smooth dial
you can do gradient-ascent on — and a continuous Δelo-style progress signal for every other idea.
- **Cost:** low. `fast_eval` already plays the graded ladder; wire it as a curriculum + metric.
- **Measure:** is *itself* the measurement layer.

### 5. Turn the move-only oracle into a value oracle
Rapfi exposes only a move — but reconstruct a full-strength **Q-over-actions** by 1-ply
lookahead *through* it: play each candidate move, ask Rapfi to evaluate the resulting position
at **full think-time** (its winrate from the opponent's seat). Expensive in general, cheap at a
narrow idx-2 tree — gives the strong soft policy+value the node-bounded `analyze` couldn't.
- **Cost:** medium (branching × think-time per labelled state).
- **Composes with:** DAgger's value side (the node-bounded soft value is weak; this is the honest fix).

### 6. A dojo, not a duel
Keep a **museum of every past checkpoint**; the net must beat *all* of them at idx-2
(non-transitivity insurance + a richer gradient than one frozen parent). And **label by
disagreement**: train two nets from different seeds; query the expert only where they disagree
on idx-2 lines — disagreement is a free uncertainty signal pointing exactly at the states worth
a label (active learning, very sample-efficient).
- **Cost:** low–medium. Reuses the eval-panel (net-vs-net) + the gather/label loop.
- **Measure:** round-robin vs the museum; labels-per-elo (sample efficiency).

### 7. Moonshot — distill Rapfi-NNUE's *evaluation function* directly
Rapfi-NNUE is a neural net and we have its `.bin`. Rather than treating it as a black-box move
oracle, read its **NNUE eval as a dense teacher signal** over many positions, distill the
*evaluation function* itself, then let net+MCTS **out-search the distilled eval**. Steal the
master's intuition; add your own search.
- **Cost:** high (requires reading the NNUE weight format).

### 8. Moonshot — test-time training
At eval, let the net take a few gradient steps on Rapfi-labelled rollouts of the **current
game** before committing its move — adapting to the specific opponent line in real time. Wild,
but TTT is a live idea in the literature and idx-2's narrowness makes it cheap to try.
- **Cost:** medium; changes the inference path (per-move adaptation).

---

*Promotion path:* an idea earns a run → file a `derby-idea` issue (code-heavy) or hand it to an
autolab lane → record the verdict on the [research board](../ops/research-board.md) and a dated
`TRAINING_WIKI` entry. Keep this pile append-friendly; prune only when an idea is genuinely
settled (with evidence), not merely untried.

---

## 2026-06-25 addendum — measured perf reality + a new idea (batch-VCF guard-rail)

**DAgger (imitation) was tried and is a NEGATIVE for the Rapfi wall** (TRAINING_WIKI 2026-06-25):
it sharpens the prior but can't cross a *search-depth* wall. That promotes **#1 (out-search-and-
distill)** to top priority — and surfaced two hard facts that shape how #1/#9 must be built.

### Measured fact — the card is COMPUTE-bound, not memory-bound (microbench, M5 Max, 128×10 net)
Net-eval throughput is **FLAT at ~11,500 boards/sec from batch 64 → 65,536** (no OOM even at 65k).
So: you can hold *tens of thousands* of boards on the card, but **batching more buys ZERO
throughput** — the GPU is compute-saturated at batch≈64. **The hard currency is net-evals/sec
(~11.5k), not board count.** Consequences: (a) "search really hard" (#1) is bounded by total
evals = states × sims; wave-batching cuts wall-clock LATENCY, not total compute. (b) The way to
buy more search is FEWER, SMARTER evals (tactical pruning) or a cheaper search-net — not a bigger
batch.

### ⚠️ Open caveat on #1 (being measured 2026-06-25): does the net's DEEP search even beat Rapfi?
Quick test: net-MCTS at sims 32 vs 200 vs the Rapfi gradient was **identical** (beats rapfi@25ms,
loses 50ms+) — 6× more search didn't move the wall. sims=800/1600 pending. **If deeper search
does NOT cross a higher Rapfi rung, the net's EVALUATION (policy+value) is the ceiling, not its
search depth — and #1 (distill your own deep search) can't help, because there's no better search
result to distill.** In that case the lever shifts to a better *evaluator* (#7 distill Rapfi-NNUE's
eval; or #9's hard tactical truth) rather than more search. This caveat must be resolved before
committing to #1.

### 9. Batch-VCF "certain-death" guard-rail — ⭐ GRADUATED
**Idea (one line):** use CPU/GPU VCF as a hard, *certain* tactical teacher / "certain-death"
guard-rail (value ±1 + the forcing move) that sidesteps the ~11.5k net-eval GPU ceiling.
**VERDICT — BUILT & PROVEN.** Batched-frontier GPU-VCF shipped at **~2,500× CPU, 100%
agreement** over ~5,900 positions (plain VCF is pure OR/reachability → batchable); it then
generalized to the full AND/OR **GPU VCT oracle** `solve_vct_mega_bb`. VCF/VCT tactical truth
is now real-time. Full living record: [gpu-vct-feasibility.md](gpu-vct-feasibility.md) +
[mega-vct-solver.md](mega-vct-solver.md). Full original narrative:
idea-pile-graduated-results.md #9 *(removed; see note above)*.

### 10. Represent the board as threat-molecules, not stone-atoms — ⭐⭐ PARTLY GRADUATED
**Idea (one line):** represent *relations* (a molecule ⊋ a line), not the 4 line directions —
threat "chemistry" (react / catalyze / quench / inhibit) as a learned representation, with
VCF/VCT as the exactly-computable boundary; hunt non-line **offensive fields** a line-vocabulary
is provably blind to.
**VERDICT — THESIS ALIVE, first probe NEGATIVE, big swing UN-BUILT.** The "rediscover the claw"
sandbox (`scripts/claw_rediscovery.py`) ran NEGATIVE (CNN **+10.2 SD** vs relation-model
**+0.06 SD** — the blocking task is itself a windowed *line*-convolution, so it never pressured
non-line vision; pivot pre-approved). Load-bearing nuance kept: a mod-5 linear probe hit **27%
vs a 20% floor** — the relation substrate CAN hold mod-5 structure; the line-shaped task just
never demanded it.
The threat-graph + attention architecture is still a moonshot seed. Living homes:
[molecule-discovery-toolkit.md](molecule-discovery-toolkit.md) (the training-free detectors —
the recommended next stabs), [the-claw.md](the-claw.md) (the worked defense-axis example),
[shape-library-engine.md](shape-library-engine.md) (the "molecule ⊋ line" program realized).
Full original narrative + the claw-rediscovery v0 result:
idea-pile-graduated-results.md #10 *(removed; see note above)*.

### 11. Play self-play games TO the first VCT, not to five-in-a-row — ⭐⭐ GRADUATED
**Idea (one line):** end self-play at the first oracle-proven VCT (median ply ~19–20, cap50/cap25
near-complete) and take the exact oracle verdict+move — halving plies and handing the value head
a *verified* terminal instead of a bootstrapped one.
**VERDICT — THROUGHPUT WIN, ROBUSTNESS LOSS → superseded by the sound-world recipe.** Terminus-only
reaches equal strength at ~45% wall-clock but **never learns defense** (loses H2H to its five-played
control; #101 shows a stable ~9-ply attractor, not VCT-avoidance). The opponent-independent
VCT-defense **aux head** learned the representation but nothing made the POLICY act on it. The
defense half was solved instead by **oracle-in-the-environment** — see
[sound-world-recipe.md](sound-world-recipe.md). Living records:
[vct-terminus-selfplay-result.md](vct-terminus-selfplay-result.md) +
[vct-defense-aux-head-result.md](vct-defense-aux-head-result.md). Full original narrative:
idea-pile-graduated-results.md #11 *(removed; see note above)*.

---

## 2026-07-04 addendum — white-VCT rail redux (query-loop filing: two independent Opus reads)

**Provenance:** Jason proposed (verbatim intent): new 15×15 net on the Bruce-Lee opener with a
"lookahead-2 for VCT" for white @25 nodes — never allow an opponent VCT unless forced, always
stay on a VCT once held. Two independent Opus agents (one wiki-guided, one flat-footed) answered
from the wiki and **converged**; filed per [curation.md](../curation.md) § Query.

**Settled mapping (both agents, independently):** the proposal ≡ **rails-v0** —
`--oracle-veto` ("never allow a VCT unless forced") + `--attacker-preserve` ("stay on a VCT") at
cap25 (#114), which **already ran on this exact board** (#116, wandb `vraf0b6e`, 234k games,
2026-07-03): starvation cured, then **value-poisoning re-collapse** (white-share ~0.015,
`loss/value` 0.052 < the 0.08 death-tell) because idx-2 plays black-tilted — a soundness filter
can't invent a defense the position doesn't contain. See
[sound-world-recipe.md](sound-world-recipe.md) § rails.

**Live distinctions + seeds worth keeping:**
- **"Lookahead-2" is ambiguous and the ambiguity is load-bearing:** a 2-ply net-leaf lookahead is
  a *soft wall* that leaks on deep forcing chains (the K-cap ablation says shallow guards
  re-admit the fast-attack attractor); a cap25-bounded **oracle solve** is sound and already
  landed. Any spec must say which it is.
- **Seed A — idx-2 holdability probe (cheap falsifier):** ~90 min of mega-VCT oracle aimed at
  idx-2 white-to-move ("is this holdable at all?") before ANY training run; if idx-2 is a
  forced/heavy black win, every white-soundness recipe on it re-poisons by construction.
  Re-aim the [idx2-vct-frontier-map](idx2-vct-frontier-map.md) machinery as a fairness verdict.
- **Seed B — the inference-time actuator (untried):** prune white's MCTS **root moves at
  inference** with a bounded white-VCT check (never expand a move that hands black a tactic;
  keep white's own live threat). #103 proved we have the sensor and lack the actuator; this is
  an actuator, and nobody has run it.
- **If re-running the rail on idx-2:** ONE lever on the rails-v0 cell — pair with
  `--tail-subsample` (#118, built, Jason-gated) — and pre-state the bet: plies RISE + white H2H
  climbs off 0 past e2000 with vl staying ≥0.10, the signature no prior attempt produced.
  Do not stack a new net shape onto the same run (confound; the 96×8 trained-backwards incident).

**Cost:** Seed A ≈ one oracle session, no training. Seed B = code-heavy (derby-idea sized).
