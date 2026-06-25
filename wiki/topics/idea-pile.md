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

### 9. Batch-VCF "certain-death" guard-rail (Jason, 2026-06-25)  ⭐ — sidesteps the GPU ceiling
`solve_vcf` is CPU/numpy (zero GPU evals) and CPU-parallelizable across cores, so it **does not
touch the 11.5k eval ceiling** — it's "free" search relative to the card. Two uses, both fusing
#1 and #2:
- **Hard ground-truth teacher.** For states the net reaches at idx-2, run massively-parallel VCF
  (CPU fan-out, the rapfimine harness pattern): if the side-to-move has a forced four-win → value
  +1 and the policy target is that forcing move; if the OPPONENT has a VCF → this state is
  **certain death**, value −1. These are *certain* labels (not Rapfi's heuristic), a perfect
  tactical teacher exactly where tactics decide the game.
- **Anti-guard-rail / move ranking.** Score candidate moves "good idea → terrible idea": a move
  that hands the opponent a VCF is a **terrible idea** (prune it, teach the net to avoid it); a
  move that gives *us* a VCF is #1. Use as an MCTS prior shaper or a distillation target — a cheap
  decisiveness lever (cf. the research-board's "lookahead4 draw→win" gap, which was diagnosed as a
  *decisiveness* problem, not a maturity one).
- **The batchable primitive:** four/open-three detection is a **batched convolution over the 4
  directions** — GPU-friendly if we ever want it on-card, but the CPU solver already sidesteps the
  bottleneck. Reuses `gomoku/vcf.py` (`solve_vcf`, `has_four_threat`) + the rapfimine multiprocess
  fan-out. Cost: low–medium; composes directly with the (built, tested) DAgger gather loop —
  swap the Rapfi label for a VCF label.

#### Measured (2026-06-25) — GPU threat-detection is ~free; the TREE is the cost → batch it
Jason's "put VCF on the GPU and measure it, the handoff may not be free" brainstorm, with hard
numbers (M5 Max, 15×15; bench `scratchpad/bench_gpu_vcf.py`). The batchable VCF core =
directional length-5 line-convolutions (four/five detection in the 4 directions):
- **GPU threat-detect, data resident on MPS:** 0.66M b/s @256 → **12.7M b/s @65,536** — *light*
  kernel, scales with batch (unlike the compute-bound net which flatlines at 11.5k). ~1100× the
  net-eval rate and ~12,000× the CPU primitive.
- **With the CPU↔GPU handoff** (transfer→conv→back→sync): ~20–35% slower (282k @256 →
  **10.0M @65,536**) — the handoff is *real but negligible*; even shuttled, detection costs ~nothing
  vs the 11.5k net bottleneck. **Inline-on-GPU detection wins; "CPU is free" is beaten by "GPU is
  ~free AND resident."**
- **CPU baselines:** `has_four_threat` primitive **1,057 b/s**; `solve_vcf` full forcing-tree search
  **53 b/s**. So detection is NOT the cost — the **sequential forcing-TREE recursion** is.
- **The opportunity:** detection being 12.7M/s means a **batched-frontier GPU-VCF** (run thousands
  of VCF searches in lockstep; each ply, advance every search's forcing-move frontier through the
  GPU detection kernel) could turn VCF tactical truth from too-slow into a first-class real-time
  teacher/guard-rail. This is the concrete build #9 points to; the per-ply primitive is proven free.

#### ✅ PROVEN (2026-06-25) — batched-frontier GPU-VCF BUILT and MEASURED: ~2,500× CPU, 100% correct
Built the spike: `scripts/gpu_vcf_prototype.py` (`solve_vcf_batch(boards (B,2,15,15) bool) ->
(won, hit_cap)`, MPS). **It crushed the projection** — not ~100×, but **~2,500×**.
- **THE INSIGHT (why it's exact AND batchable):** plain VCF is a pure **OR / reachability** search,
  not a real AND/OR tree — the defender is ALWAYS forced to the *unique* completion square of the
  attacker's four, so every defender node has exactly one child. So "is there a forced win?" ==
  "from the root, can the attacker reach (within max_depth) a node with an immediate five or a sound
  double-four?". Run that as a **breadth-first frontier**: every node in the current frontier is an
  attacker-to-move board at the *same depth*, so the whole frontier advances **in lockstep** and ALL
  threat detection batches across the B searches at once (directional shift-products over `(F,15,15)`
  bool planes). One host sync per BFS level (the child-gather `nonzero`); no `.item()` in the loop.
- Four-detection is provably equal to `vcf._five_completions`: a four-move m + completion c ⟺ a
  length-5 window of 3 own + the two empties {m,c} + 0 opp; each (direction, signed-offset) pair maps
  to a *unique* completion cell, so the count of firing pairs == CPU's `len(comps)` (single vs double
  four) and the single-four block square is `m+δ·d`. Forcing test == `_has_immediate_five(defender)`.
- **CORRECTNESS: 100% agreement vs CPU `solve_vcf` across 5,300 positions** (500 spec + 2,400 random
  midgame + 2,400 dense), incl. **121 deep mates up to mate-distance 15** — every verdict matched;
  CPU `hit_cap`=0, the one GPU frontier-truncation case still agreed.
- **THROUGHPUT (M5 Max MPS, random midgame mix):** scales to ~**130–146k FULL solves/sec at
  B≈16k–65k** vs CPU's **53/s** = **~2,500× (peak 2,749× @ B=65,536, depth 8)**. Saturates near
  B≈16k; small B is launch-overhead-bound (~8–10k/s @256). depth has little effect (random wins are
  shallow). Throughput is position-mix-dependent — a batch of deep-tree forced-wins branches the
  frontier wider and costs more (capped at `max_frontier=4M`, ~1.8 GB, flagged `hit_cap`).
- **Open items:** returns the *verdict* not `winning_move`/`mate_distance` yet (the block-index
  machinery is already there — easy add); **no child-board dedup yet** (a per-level hash dedup is the
  obvious next win on tactical batches); plain VCF only, **not VCT** (the threes solver). The
  node-budget accounting differs (CPU DFS calls vs BFS frontier nodes) so the only *theoretically*
  possible disagreements are cap-boundary cases — none observed. **Verdict: VCF tactical truth is now
  real-time → #9's ground-truth teacher / certain-death guard-rail is unblocked on throughput.**
