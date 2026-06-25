# The Idea Pile — research seeds for the autolab

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
