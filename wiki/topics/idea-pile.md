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
  node-budget accounting differs (CPU DFS calls vs BFS frontier nodes) so the only possible
  disagreements are cap-boundary cases — and across ~5,900 total positions **exactly ONE** appeared
  (a later 600-dense-board run): a dense board where the CPU **hit its 200k-node DFS cap and returned
  an *unproven* `False`** (`hit_cap=True` = "unproven", not proven-safe) while the GPU completed and
  **proved a genuine forced win**. So the divergence is the GPU being strictly *more complete* than
  the cap-limited CPU, **not a false positive** — clean (non-capped) agreement stays **100%**.
  **Verdict: VCF tactical truth is now real-time → #9's ground-truth teacher / certain-death
  guard-rail is unblocked on throughput.**

---

## 2026-06-25 — the big swing: a threat-CHEMISTRY representation (Jason's nose)

### 10. Represent the board as threat-molecules, not stone-atoms  ⭐⭐ (representation-level swing)

> **Headline principle (hold this tightest): a molecule ⊋ a line.** Line-threats (fours, threes,
> VCF/VCT) are one *species* of molecule, not the genus — *"not all animals are dogs."* The moment a
> representation is organized around the 4 line directions it has decided non-line molecules don't
> exist — and is *provably blind* to them ([the-claw.md](the-claw.md)). So: **represent *relations*,
> not lines; let lines *emerge* as the common species.** Classify molecules by **function** — *offense*
> (force a line), *denial* (prevent lines: the claw), *shaping* (bias which molecules form next: "2s for
> options") — and treat **geometry** (line / lattice / cluster / field) as *orthogonal* to function.
> **The line is the win-*condition*, not the offense itself:** offense is the forcing pressure that
> *collapses into* a line, and may itself be a **field** (double-threat branch-points, VCF-dense regions,
> influence/thickness, catalytic gradients) that no line-vocabulary ever named (Jason, 2026-06-25 — "there
> may be yet-unknown-to-us *offensive* fields"). **Unit test for non-line vision:** can a representation
> tell the claw from a random same-count scatter? (A line rep provably cannot.) **The prize:** a
> relation-general representation is not just a better *describer* — it's a **discovery engine** for
> molecule-species we have no names for. Point it at high-value, *zero-line-content* residuals to surface
> unknown **offensive fields** — offense that 30 years of line-shaped theory (human *and* engine) was
> structurally unable to look at.

The frame that reorganizes everything above. **Gomoku, as humans play it, is not about stones — it's
about a stone's *relationship to future threats*.** You build 2s because 2s give you options; you
*avoid* building 2s when they only feed the opponent's threats; the skill is balancing those. The
stone-by-stone, cell-by-cell view is an **artifact of the search algorithm and of "the board is a grid
of positions"** — it's the convenient atomization, not the real object. The real object is an **overlay
of threat patterns with a learnable chemistry**: functional groups (twos, open-threes, fours) that
**react, catalyze, and quench** in combinations that are emphatically *not* set-addition.

**The crisp statement of "not set-addition":** two open-threes that *share a stone* (the fork) vs two
open-threes in different corners — same stone count, same "two threes," utterly different value. **The
value is in the bond, not the count.** Set-union of lines throws away exactly the information that
decides the game.

**This is not just metaphor — there's formal grounding we can build on (vocabulary + theory):**
Allis's **threat-space search** (the framework VCF/VCT are special cases of, Victor Allis 1994,
*Searching for Solutions in Games and AI* / the threat-space-search papers) already models every threat
as a **gain square** (where you play) + **cost squares** (the cells the line requires / the opponent
must answer in). That gives a real *reaction algebra*:
- threats **react** (forcing combination) when one's gain square is another's cost square, or they share
  squares such that a *single* defense can't cover both (→ double-threat / fork);
- threats are **inert** (genuinely additive, boring) when their squares don't interact;
- threats are **mutually quenched** when the defense to one kills the other;
- opponent stones are **inhibitors** — one enemy stone in a line poisons that whole species (no five
  possible there) permanently;
- a two is a **precursor / activation energy** — not a threat, but one move from igniting into an
  open three. Hence a **reactivity gradient** empty < stone < two < three < open-three < four < double,
  each rung "closer to ignition." A *value* is what integrates that gradient (mine minus theirs).

**Why this is also the DIAGNOSIS of our positional wall.** We concluded the wall is positional, not
tactical (TRAINING_WIKI 2026-06-25: net+root-VCF identical to net-only, 0 forced-win hits; deeper MCTS
search didn't move it). Restated in this frame: **our value head is computing chemistry over the wrong
primitives.** It's fed 17 raw-stone planes (`game.py to_planes()`: current + 7-ply history per side +
const) into a conv tower. Convolution is *good* at LOCAL pattern detection — a length-5 four IS a conv
feature (that's literally our GPU-VCF detector). But the chemistry — two threats far apart that share a
square — is a **long-range RELATIONAL** computation, and conv-over-a-grid is structurally weak at exactly
that. The net sees atoms everywhere and must **re-derive every bond, every forward pass**, through
stacked receptive fields — with no 5000-TPU budget to bake that re-derivation into the weights.
*Testable corollary:* NNUE engines for renju/gomoku typically feed **line-pattern / shape features**,
not raw stones (Rapfi's exact feature set unconfirmed — verify). If so, "Rapfi out-evaluates us
positionally" has a concrete mechanism: **Rapfi does chemistry over molecules; we do it over atoms.**

**The scrappy-team move — hand-code the cheap part, learn the expensive part:**
- **Cheap part we already own (this-week experiment):** detecting atoms-and-bonds is just convolution —
  the exact GPU machinery from #9. Precompute per board a stack of **threat-channel input planes**:
  open-three-here, four-here, double-four-gain-square-here, **cost-square-overlap heat** (this cell is a
  cost square for N of my threats = the bond/connectivity signal), and — the new toy — **"playing here
  hands the opponent a VCF" / "playing here wins by VCF"** (from `solve_vcf_batch`, now real-time).
  Feed those *alongside* the raw stones. The net stops re-deriving fours and spends capacity on the
  reaction dynamics. **Cost: low** — detectors exist; it's a `to_planes()` augmentation + retrain.
- **The expensive part worth net capacity:** the *reaction dynamics* — how functional groups combine
  into wins — is the thing we DON'T know how to hand-code past VCF/VCT, and it's where the net should
  spend its weights.
- **The real swing (moonshot):** stop feeding a grid for the relational part. **Nodes = detected
  threats, edges = their square-interactions (shared cost square / gain=cost coupling); run
  attention / message-passing over that graph.** Every threat attends to every other; the model
  *learns the reaction table*. Attention is the natural fit because "which pair/triple of threats is
  reactive" is a learned, non-additive, all-pairs function — catalysis and double-threats fall right
  out. **Cost: high** (new architecture + threat-graph builder), but it's the honest form of the idea.

**The one risk, named:** engineered threat features can cap the net at *our* taxonomy (the classic
"handcrafted features eventually lose to learned ones"). **Resolution — hybrid, not a cage:** keep the
raw-stone planes in the input too, so the net can still discover species outside our vocabulary; the
threat channels are a *scaffold and strong prior*, not a replacement. We hand it the atoms and bonds; it
learns the chemistry and stays free to find new elements. Respects both the nose and the from-scratch
dream, on an M5 budget.

- **VCF/VCT are the boundary conditions, not the equation.** They are the two *fully-forcing* reactions
  where the chemistry is exactly computable (and now, post-#9, real-time). The general equation this idea
  names is: **position value as a learned function over the threat reaction-network**, with VCF/VCT as
  the known-exact boundary. To our knowledge this has been used for *solving* (threat-space search) but
  **not as a learned *representation*** — that's the open gap.
- **Measure:** (a) does adding threat-channel input planes move the Rapfi ms-ladder crossing (idea #4)
  vs the raw-stone baseline, holding net size fixed? Cleanest first probe. (b) ablate raw-stones-only
  vs +threat-channels vs threat-graph. (c) the Rapfi-feature-set check (is its edge representational?).
- **Composes with:** #9 (the threat/VCF detectors ARE the channel generators), #7 (if Rapfi's edge is
  its shape features, distilling its eval and feeding shape features are two routes to the same fix),
  #1/#4 (better evaluator → search has something better to surface and distill).
- **Runnable seed:** augment `gomoku/game.py to_planes()` with K threat channels computed by the #9 conv
  detectors (+ a `solve_vcf_batch` certain-death/birth channel); retrain at fixed net size; gate on the
  ms-ladder crossing. The moonshot (threat-graph + attention) is a separate, larger lane.
- **Worked DEFENSE-axis example — "the claw"** ([the-claw.md](the-claw.md)): Jason's 1990s knight's-move
  defensive pattern = the lattice `2x+y≡0 (mod 5)`, *proven* the unique-up-to-symmetry perfect 5-in-a-row
  blocker — and *proven* invisible to Rapfi's line-organized representation (radius-4 classical engine
  aliases it to lone stones; radius-1-conv mix9svq near-blind). Its invariant is a **periodic congruence
  field**, not a line/shape feature, and a small conv can't express "position mod k" without absolute
  coords. **Lesson:** the defense axis likely needs **explicit periodic/modular positional channels**, not
  only threat detectors — and the claw is a concrete **#3 (judo)** target (a region where Rapfi's map is
  structurally wrong). Caveat: covering *set* ≠ drawing *strategy* (k=5 is a first-player win, Allis '94) —
  it's a representation/shaping prior, never a static policy.

- **Methods to do the discovering** → [molecule-discovery-toolkit.md](molecule-discovery-toolkit.md):
  a raid of computational-genetics tools mapped onto "discover non-line molecules / offensive fields" —
  **DCA** (training-free non-line *bond map* of cells, disentangles direct from line-induced indirect
  correlation), **cryo-EM 2-D class averaging** (unsupervised template dictionary, pose = the finite D4
  group), **reciprocal-lattice spectral** detector (the claw = a Bragg peak at frequency (2/5,1/5); lines
  light *different* bins), **TF-MoDISco** (discover motifs in value-head *importance* space → value-grounded
  molecules), **MAP-Elites/novelty** (illuminate a periodic table of fields). Killer reframe: a board needs
  no MSA, so these transfer *better* to us than to biology. (Turing reaction-diffusion = a lens only, NOT a
  mechanism — the claw's period-5 is number-theoretic, not dynamical; see that page's caveat.)

#### First stab — the "rediscover the claw" sandbox (the bitter-lesson probe)
Goal: prove a **relation-general** representation can rediscover a KNOWN non-line molecule (the claw)
*from compute*, where a line-based one provably can't — a measurable proving ground before hunting
UNKNOWN offensive fields. **Bitter-lesson stance:** don't hand-code the claw or mod-5; hand-code only a
*task* whose optimum IS the claw, and see whether the structure **emerges** in a learned representation.
The M5 Max is plenty for this (small board, tiny models, labels free) — *and* the efficiencies hide in
surprising places (cf. GPU-VCF): the claw is a **frequency-domain** object, so a periodic positional
encoding may rediscover it as a *spatial frequency*.
- **Task (cheap, GPU-resident):** blocking score of a defender stone-set = #(5-windows, all 4 dirs,
  containing ≥1 stone), via the length-5 directional conv kernels already in
  `scratchpad/bench_gpu_vcf.py`. The claw (`2x+y≡0 mod 5`, density 1/5) is the *provable* optimum.
- **Two representations predict the field-task from raw stone planes:** (a) **line/CNN baseline** (small
  conv over stones — should plateau; it's the blind one); (b) **relation-general** (small self-attention
  over stone/cell tokens with *relative* or *Fourier/periodic* positional encodings — capacity for
  arbitrary offsets + periodicity).
- **Did the claw emerge? (the dream signals):** (1) gradient-ascend a board through each learned model's
  blocking-landscape → does the relation-general model lay down a mod-5/knight lattice while the CNN
  can't? (2) linear-probe `2x+y mod 5` (cell residue) from hidden states → decodable for relation-general,
  not for the line baseline; (3) claw-vs-scatter discriminability; (4) Fourier PE: does period-5 light up
  in the learned positional spectrum?
- **Cost: low**; falsifiable (no emergence → architecture insufficient, pivot — Jason pre-approved the
  pivot). **Then** remove the scaffold and point the validated representation at real idx-2 positions:
  high-value, zero-line-content residuals = candidate **unknown offensive fields**. Tracked as the first
  runnable step of #10's relation-general lane.
- **v0 RESULT (2026-06-25): NEGATIVE — and the negative is load-bearing.** A relation-general transformer
  (Fourier/relative PE) did NOT beat a plain CNN at the blocking task; the **CNN won every behavioral
  signal** (claw-vs-scatter **+10.2 SD** vs the relation model's +0.06 SD = pure chance; the relation
  model's gradient-ascent **collapsed 45 stones into a central blob**, the opposite of a density-1/5
  lattice). **Why (the finding):** *blocking-score is itself a windowed line-convolution* — a length-5
  directional conv literally computes it — so the task is **line-shaped**, plays straight to the
  translation-equivariant CNN, and never *pressures* the relation substrate. The claw's line-invisibility
  is about lack of **offensive** line content; *blocking* is still a per-window **line** quantity, so this
  task can't force non-line vision. One weak PARTIAL for the thesis: a linear probe of `(2x+y) mod 5`
  reached **27%** from the relation model's final layer (vs the CNN pinned at the **20%** chance floor —
  translation-equivariance *structurally* can't hold absolute residue — and vs 16% in the relation model's
  own pre-training substrate), so the substrate *can* hold mod-5; the task gives it no reason to. Method is
  sound: TRUE-objective gradient-ascent recovers the claw (0.80 claw-ness, blocks 0.97/1.0). Script:
  `scripts/claw_rediscovery.py`.
- **PIVOT (what the negative dictates):** the discriminating task must be **position-dependent /
  non-translation-invariant** — a translation-invariant *count* can never separate the architectures. E.g.
  a per-cell "can the attacker ever make 5 through here" potential, or the **adversarial tempo-aware**
  defensive question from [the-claw.md](the-claw.md) §3 ("attacker has the move — which cells stay safe?").
  **Better: sidestep the learned-task trap entirely** with the **training-free** detectors in
  [molecule-discovery-toolkit.md](molecule-discovery-toolkit.md) — #3 spectral (detect the claw by its
  *frequency*, no task to accidentally make line-shaped) and #1 DCA (a non-line *bond map*). Those are the
  recommended next stabs; run order on that page.

---

## 2026-06-30 — measured from the full-corpus VCT cascade (#97)

### 11. Play self-play games TO the first VCT, not to five-in-a-row  ⭐⭐ (self-play efficiency)
> ⚙️ **BUILT 2026-06-30 (#98), default-OFF.** `--vct-terminus` / `--vct-terminus-budget 50` in
> `selfplay_worker.py` → `self_play.configure_vct_terminus()`: a batched cap50 `solve_vct_mega_bb`
> across the whole wave each ply (native + Python paths) ends any game whose side-to-move has a forced
> VCT, recording the oracle's winning move (one-hot) + exact terminal value. **Prereq done:** the
> `mega_vct_bb` oracle was 15×15-only (word-3 `TOPMASK`); ported N-general (byte-identical @15, validated
> @9). E2E smoke (random net, 9×9): plies 36.0→**19.9** (0.55×), median terminus ply **20** (= the corpus
> finding), 64/64 games VCT-ended. **Eval side BUILT too (#99):** `eval.vct_finish_picker` — a hybrid
> player that plays by policy to the VCT then lets the GPU oracle hammer it out to a REAL five
> (`match.py` `model:...,vct_finish=50`, `eval_worker --vct-finish-nodes`), so the net wins genuine games
> vs any opponent through the standard harness (no special "VCT=win" harness) — also the deployable
> web-UI player.
> ⭐ **TESTED 2026-06-30 (#100) — THROUGHPUT WIN, ROBUSTNESS LOSS.** Matched 9×9 A/B (`vctsci-terminus` vs
> `vctsci-control`), grown to e500. The terminus reaches **equal fixed-baseline strength at ~45% of the
> control's wall-clock** (throughput claim CONFIRMED) — but **loses head-to-head to the control 75–25 (0
> wins in 120 games, every config)** and **0–40** to the champion, because ending every game at the first
> VCT (≈ply 9) means it **never learns to defend** and, vs any opponent that denies a VCT, it never reaches
> one (finisher fires = 0) and collapses. **This caveat (below) is the DOMINANT effect** — attack-only
> specialization, a cousin of fast-attack collapse. The seek-VCT *objective* survives; the missing half is
> DEFENSE. Full result: [vct-terminus-selfplay-result.md](vct-terminus-selfplay-result.md); next probe #101.
**Measured fact** ([vct-cascade-run-2026-06-30.md](vct-cascade-run-2026-06-30.md)):
labeling all 56.1M unique rapfi positions with the GPU VCT oracle and joining back
to games shows the **first VCT arrives at median ply 19 / mean 21.6** (p10–p90 =
12–31), in **96.4%** of games. Rapfi games run ~40–60 plies, so the first *proven
forced win* exists less than halfway in.
**The idea:** a VCT is a forced win the oracle both **detects and terminally values**
(exact win + winning move via `solve_vct_mega_bb(return_move=True)`). So self-play
need not play out to an actual five — **stop at the first VCT and take the oracle
verdict as the result.** Roughly **halves the plies per game** (≈40–60 → ~19 to a
*labeled* winner) and replaces a bootstrapped value with an **exact** one at the
terminus.
- **Why it matters:** self-play rollout is the training bottleneck (MCTS-expanded
  plies). Cutting trajectory length >2× and handing the value head a verified target
  near the decisive moment should mean cheaper *and* cleaner data. Composes with the
  whole seek-VCT thesis ([shape-library-engine](shape-library-engine.md),
  [phi-distance-field-learnability](phi-distance-field-learnability.md)) — the game's
  real objective becomes "reach a VCT", which is exactly what those L2 targets regress.
- **Watch / caveat:** terminating at VCT changes the value target distribution and
  removes the defender's "play it out" learning past onset; the
  [vct-reachability-mining](vct-reachability-mining.md) knife-edge result (≈80% of
  alternative moves already lose by force near onset) suggests the pre-VCT band is
  where the hard learning is — so VCT-termination should *sharpen* the signal, not
  lose it, but measure. Defender still needs the "no VCT for me, avoid giving you one"
  lessons, which this preserves (the terminal credits the side that achieved the VCT).
- **Cost:** low-ish. The solver already runs in/near the MCTS loop (batch-VCF
  guard-rail, idea #9); reuse it as a terminal test each ply at **cap50** and end the
  game on the first hit.
- **⭐ MEASURED 2026-06-30 — cap50 is a near-complete first-VCT detector, so this is
  CHEAP:** 98.8% of all 27.4M corpus VCTs resolve at 50 nodes; a 50-node seeker finds
  a forced win in **96.1% of all games** (99.7% of vct-games), about as early as an
  infinitely-patient solver (median first-VCT ply 20 @ cap50 vs 19 @ cap2000 — 40×
  the budget buys ~1 ply). Only 0.3% of vct-games have a VCT invisible at 50 nodes.
  ⇒ a **cap50 mate-seeker** gives up almost nothing (0.3% of decidable games, ~1 ply)
  for 40–850× faster iteration than deep search. The deep tail (>2000 nodes, 9.8% of
  the corpus) is a wall dominated by hard-to-confirm no-wins, not cheap deep wins —
  skip it. Full numbers: [vct-cascade-run-2026-06-30.md](vct-cascade-run-2026-06-30.md)
  § "50 nodes is a near-complete first-VCT detector".
- **Measure:** plies/game in self-play (expect ~2× drop), value-head calibration on
  held-out oracle verdicts, and the #4 ms-ladder crossing vs a five-terminated
  control at equal wall-clock.
