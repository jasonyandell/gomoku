# VCT-mining research — the synthesis hub for the "seek-VCT" program

**What this page is.** A navigable map of the whole research program that mines
**VCT (Victory-by-Continuous-Threats) structure** out of the game corpus and the GPU
oracle, and turns it into a learnable, sound gomoku AI. Many threads have accumulated
(mining, labeling, reducing to shapes, three learnability probes); this hub ties them
into one story and says **which page holds what**. It summarizes and links — the deep
pages keep the evidence.

**The one thesis they all serve — "seek-VCT" (Jason, 2026-06-26).** *Don't search toward
five-in-a-row; steer play toward a position where a forced win (a VCT) exists, then hand
the tactical finish to the exact oracle.* The whole program rests on one asymmetry —
**anti-correlated tractability**:

- **Positional/strategic play** — intractable to *solve*, but **tolerant of approximation**
  (a slightly-wrong quiet move rarely loses on the spot) ⇒ give it to a **net** (the
  approximation-tolerant STEERING).
- **Tactical/forcing finish** — **intolerant** of approximation (one wrong move and the
  forced win evaporates) but **tractable to solve exactly** (the forcing constraint
  collapses the tree — why the oracle does tens of thousands/s) ⇒ give it to the **oracle**
  (the approximation-intolerant FORCING FINISH).

This is **the same division of labor Rapfi already ships**: its quantized shape network does
cheap approximate recognition and **never** confirms a win — VCF/VCT is pure search
([rapfi-mix9svq-architecture.md](rapfi-mix9svq-architecture.md) §6, "Division of labor"). The
seek-VCT bet is to move *more* of the game onto that clean side of the line: learn the
steering, solve the finish. (The task brief calls this page `net-architecture-and-representation.md`;
that file does not exist — the division-of-labor argument lives in the Rapfi teardown above.)

---

## The through-line (read this first)

1. **An oracle makes it all possible.** The on-device GPU VCT megakernel
   ([mega-vct-solver.md](mega-vct-solver.md), feasibility in
   [gpu-vct-feasibility.md](gpu-vct-feasibility.md) §8) runs the *whole* AND/OR proof search
   on-device, **~1600× CPU**, **0 FP / 0 FN**, and — the load-bearing fact — its **call-cost
   law** makes the wall nearly flat in batch size (one *tail*, set by the single hardest
   board). Million-position VCT labeling and per-cell ablation are tractable **only** because
   of this. Every thread below is a consumer of that oracle.

2. **Mine the corpus for where forced wins live.** Given a half-million strong Rapfi-vs-Rapfi
   games, several threads extract VCT structure: walk won games *back* to the enabling shape
   ([vct-backward-mining.md](vct-backward-mining.md)); read the *free* per-ply verdict field and
   fan off-path to find the knife-edge and the non-VCF "molecule" gold
   ([vct-reachability-mining.md](vct-reachability-mining.md)); label **all 56.1M** unique
   positions with exact verdicts ([vct-cascade-labeler.md](vct-cascade-labeler.md) +
   [vct-cascade-run-2026-06-30.md](vct-cascade-run-2026-06-30.md)); and forward-expand the idx-2
   opening as an AND/OR danger map ([idx2-vct-frontier-map.md](idx2-vct-frontier-map.md)).

3. **Reduce the wins to reusable structure.** The Shape-Library Engine
   ([shape-library-engine.md](shape-library-engine.md)) compiles Allis's threat theory *out of
   data*: mine the first-VCT, reduce each to a minimal **stencil**, match structurally, and let
   the oracle verify every leaf so the engine **never hallucinates a win** — only ever goes
   blind. The formal grammar it operates on is [allis-threat-theory.md](allis-threat-theory.md);
   the hunt for **non-line** structure the line-based grammar misses is
   [molecule-discovery-toolkit.md](molecule-discovery-toolkit.md), whose canonical exhibit is
   [the-claw.md](the-claw.md).

4. **Prove a net can carry the steering half.** The **learnability trilogy** shows, on
   held-out shard-disjoint games, that a net can **SEE** a VCT
   ([vct-recognition-learnability.md](vct-recognition-learnability.md)), **STEER** toward one
   ([seeker-steering-learnability.md](seeker-steering-learnability.md)), and **REGRESS** the
   proof-frontier field ([phi-distance-field-learnability.md](phi-distance-field-learnability.md),
   the first real L2 model). All three: yes-learnable, and a CNN beats attention every time.

**The punchline the threads converge on:** learn the approximation-tolerant STEERING with a net,
solve the approximation-intolerant FORCING FINISH with the oracle. The mining threads supply the
*verifiable* targets (distances, stencils, verdicts — never bootstrapped self-play value); the
learnability threads confirm the net can perceive them; the oracle guarantees soundness at the leaf.

## Which page for what

| If you need… | Go to |
|---|---|
| The oracle's API + the **call-cost law** (the enabling constraint) | [mega-vct-solver.md](mega-vct-solver.md) · [gpu-vct-feasibility.md](gpu-vct-feasibility.md) §8 |
| The threat **grammar** (four/three, gain/cost/rest, dependency/conflict, VCF vs VCT) | [allis-threat-theory.md](allis-threat-theory.md) |
| **The AI plan** — stencils (L1), the meta-VCT field (L2), the fork-seeking player | [shape-library-engine.md](shape-library-engine.md) |
| Mining the **enabling shape** / first-VCT move (move-labeled gold) | [vct-backward-mining.md](vct-backward-mining.md) |
| The **knife-edge**, the off-path fan, the free Φ field, the non-VCF molecule gold | [vct-reachability-mining.md](vct-reachability-mining.md) |
| **Corpus-scale exact verdicts** + throughput knees + the median-ply-19 headline | [vct-cascade-labeler.md](vct-cascade-labeler.md) · [vct-cascade-run-2026-06-30.md](vct-cascade-run-2026-06-30.md) |
| The **idx-2 opening** danger map / forward frontier | [idx2-vct-frontier-map.md](idx2-vct-frontier-map.md) |
| **Non-line** structure — methods, and the canonical example | [molecule-discovery-toolkit.md](molecule-discovery-toolkit.md) · [the-claw.md](the-claw.md) |
| Can a net **see / steer / regress** a VCT? (the learnability evidence) | [vct-recognition-learnability.md](vct-recognition-learnability.md) · [seeker-steering-learnability.md](seeker-steering-learnability.md) · [phi-distance-field-learnability.md](phi-distance-field-learnability.md) |
| Why the whole net/oracle split is right (Rapfi already does it) | [rapfi-mix9svq-architecture.md](rapfi-mix9svq-architecture.md) §6 · [net-architecture-and-representation.md](net-architecture-and-representation.md) (our net + line-planes representation) |

## The threads, in a few sentences each

### The oracle (L0) — the enabling tech
[mega-vct-solver.md](mega-vct-solver.md) / [gpu-vct-feasibility.md](gpu-vct-feasibility.md) §8.
A bitboard (`ulong[4]`) Metal megakernel runs one whole AND/OR VCT search per GPU thread —
**~1600× CPU**, **0 FP / 0 FN** over 320 VCT + 360 VCF real positions. The **call-cost law**:
one call = one *tail*, the wall set by the single hardest board and ~flat in batch size (B=16 →
24.6 s, B=16384 → 71.7 s), so every consumer is **bulk-synchronous** and caps `max_nodes`. Passive
proof outputs (`move`, `support`, `carriers`, `w`, `max_depth`→md) turn it from a yes/no oracle into
a stencil/label factory without extra search.

### Mining the corpus for VCT structure
- **[vct-backward-mining.md](vct-backward-mining.md)** — walk each *won* game BACKWARD to the
  earliest still-forced winner-to-move position = the **first true VCT move**, the *setup* not the
  line-bound kill. The flat-batch GPU miner ("just be sloppy, solve everything from the back" — the
  tail-bound kernel makes exhaustive cheaper than clever early-stopping) banked **200k** move-labeled
  enabling shapes, run-lengths to 17, **0 FP/FN/extra over 258 clean GPU-vs-CPU** checks. *Superseded
  as the canonical target* by the forward first-VCT miner (existence isn't monotone across plies), but
  it validated the substrate.
- **[vct-reachability-mining.md](vct-reachability-mining.md)** — two search-free ways to mine steering
  signal, and **the biggest single thesis update**: the pre-onset band we assumed was the net's *quiet,
  forgiving* region is a **KNIFE-EDGE** — ~**80% of alternative moves lose by force** (~99% one ply
  before onset, ~half even 6 plies out), so the net/solver boundary is fuzzy and *earlier* than assumed.
  Also: the off-path fan is a **defense/blunder** miner (not offense — a VCT belongs to the side to
  move); the **VCF-triviality split** (96% of fanned wins are trivial four-blocks, the 3.5% **non-VCF
  VCT** gold is the real molecule, concentrated on the *winner's* side); the free **Φ distance-to-VCT
  field**; and a 146,655-board move-labeled molecule-gold bank.
- **[vct-cascade-labeler.md](vct-cascade-labeler.md)** + **[vct-cascade-run-2026-06-30.md](vct-cascade-run-2026-06-30.md)**
  — label **all 56.1M** unique D4-canonical corpus positions with exact GPU-VCT verdicts via a
  node-budget ladder (reducer-over-a-log, resumable Parquet). Results: **48.9%** of positions are
  VCT-wins; the **first VCT arrives at median ply 19** (mean 21.6), so the entire back half of nearly
  every game is VCT-saturated — which is *why* the win rate is so high. **cap50 is a near-complete
  first-VCT detector** (98.8% of all VCTs, a forced win in 96% of games, ~as early as an
  infinitely-patient solver) ⇒ build the AI as a **cap50 mate-seeker**. Actionable spin-off: **play
  self-play TO the first VCT, not to five** (idea #11). (Also the load-bearing ops lesson: a
  `timeout`-SIGKILL of MLX mid-Metal-compile wedged the compiler service and crashed WindowServer —
  bound GPU runs by WORK, never an external timeout.)
- **[idx2-vct-frontier-map.md](idx2-vct-frontier-map.md)** — forward-expand the idx-2 opening as an
  AND/OR frontier (Rapfi top-8 both sides, the GPU oracle the only verdict): black VCT = win-terminus,
  white VCT = fumble loss-terminus. run-a: 9.6M nodes / depth-11 / TIME-bound (throughput dead-flat
  ~1,750 nodes/s). Produces a depths-0→7 **danger map** and the honest bounds (a harvest of winning
  positions, *not* a backed-up strategy; the AND-node top-8 gap makes it an approximation, never a solve).

### Reducing wins to reusable structure
- **[shape-library-engine.md](shape-library-engine.md)** — *the gomoku AI Jason wants to build.*
  Compile Allis out of data: mine the guaranteed-first-VCT → reduce to a minimal **stencil** (via
  md-invariant context ablation — the all-white "meanest board" certificate FAILED) → match structurally
  → **L0 verifies every match** (L1 proposes, L0 proves) so the engine **never hallucinates a win**, only
  goes blind. Measured **certificate property** (#88/#89): a stencil winning *in isolation* transfers by
  the same forcing line — **660/660** self-contained, **0** tempo-safe placements refute; the exact
  breaker is immediate defender `def_tempo`. **L2** = the AlphaZero layer that regresses stencil-
  reachability into the fog on verifiable targets. The player = a fork-seeking two-player pursuit (df-pn),
  not Dijkstra.
- **[allis-threat-theory.md](allis-threat-theory.md)** — the citation-backed formal grammar the whole
  program runs on: the taxonomy (five / straight-four / four / three / broken-three), the **gain/cost/rest**
  square formalism, **dependency vs conflict**, the winning-combination (fork) condition, and **VCF
  (OR-only) vs VCT (AND/OR)**. Allis proved 15×15 freestyle is a first-player win (138,790-node solution
  tree) — the theorem behind the white-defense wound and the claw's "no drawing strategy at k=5".
- **[molecule-discovery-toolkit.md](molecule-discovery-toolkit.md)** — methods raided from computational
  genetics/cryo-EM to discover **non-line "molecules"** (DCA bond-maps, cryo-EM class-averaging,
  reciprocal-lattice spectral detector, TF-MoDISco importance motifs, MAP-Elites). Key split: forcing
  threats are collinear *by construction*, so non-line structure lives only in the dependency-graph
  *topology* or the *residual* (won, no forcing proof) — point the bio tools at the residual.
- **[the-claw.md](the-claw.md)** — the canonical non-line molecule: Jason's 1990s knight's-move defensive
  crystal `2x+y ≡ 0 (mod 5)`, *proven* the unique optimal static 5-in-a-row blocker AND *proven* invisible
  to Rapfi's line-organized eval. The existence proof that non-line structure is real and that a line
  grammar (and a small conv) is structurally blind to it.

### The learnability trilogy (can a net carry the steering half?)
All three share the honest methodology — shard-disjoint held-out split, labels reused from the miner (no
re-solve), small/untuned go/no-go — and all three answer **yes**, with a **CNN beating attention** each time.
- **[vct-recognition-learnability.md](vct-recognition-learnability.md)** — can a net **SEE** "you have a
  forced VCT"? Yes, AUROC 0.92+ on unseen games. But recognition is *count-dominated* (logreg-on-counts
  0.946 beats attention 0.924; CNN wins with half the params) ⇒ leave recognition to the exact oracle;
  attention's real audition is the seeker.
- **[seeker-steering-learnability.md](seeker-steering-learnability.md)** — can a net **STEER** toward a VCT?
  Yes: behaviorally-clones the pre-onset moves of the side that reaches the first VCT, held-out top-1 0.386 /
  top-5 0.696 vs 0.025/0.121 for the adjacency prior. A weak-but-honest local proxy; doesn't yet settle
  attention's global-receptive-field bet (that's the Phase-C hybrid-play eval).
- **[phi-distance-field-learnability.md](phi-distance-field-learnability.md)** — can a net **REGRESS** the
  proof-frontier field Φ? Yes: CNN offense ρ=0.72 / defense ρ=0.76, reach-AUROC ≈0.91 on unseen games —
  **the first real L2 model** (verifiable, non-bootstrapped targets). Two sharp results: Φ is **NOT
  count-dominated** (spatial structure lives in *distance*, not *presence*), and **CNN beats attention a
  third time** — now param-matched on the global target with 3× the epochs — so the global-receptive-field
  bet does not cash out at this scale.

## Key evidence headlines (preserve on edit)
- **Oracle:** ~1600× CPU, **0 FP / 0 FN**; call-cost law (one tail, ~flat in B).
- **Corpus:** **56.1M** unique D4-canonical positions labeled; **48.9%** are VCT-wins.
- **First VCT at median ply 19** (mean 21.6); cap50 finds a forced win in 96% of games.
- **Knife-edge: ~80%** of pre-onset alternative moves lose by force (~99% one ply before onset).
- **Certificate property: 660/660** self-contained, **0** tempo-safe placements refute.
- **Trilogy:** recognition AUROC 0.92+, seeker top-1 0.386, Φ CNN ρ≈0.72–0.76 — all held-out; CNN > attention ×3.

**Cross-links:** [mega-vct-solver.md](mega-vct-solver.md) · [gpu-vct-feasibility.md](gpu-vct-feasibility.md) §8 ·
[shape-library-engine.md](shape-library-engine.md) · [allis-threat-theory.md](allis-threat-theory.md) ·
[vct-backward-mining.md](vct-backward-mining.md) · [vct-reachability-mining.md](vct-reachability-mining.md) ·
[vct-cascade-labeler.md](vct-cascade-labeler.md) · [vct-cascade-run-2026-06-30.md](vct-cascade-run-2026-06-30.md) ·
[idx2-vct-frontier-map.md](idx2-vct-frontier-map.md) · [molecule-discovery-toolkit.md](molecule-discovery-toolkit.md) ·
[the-claw.md](the-claw.md) · [vct-recognition-learnability.md](vct-recognition-learnability.md) ·
[seeker-steering-learnability.md](seeker-steering-learnability.md) ·
[phi-distance-field-learnability.md](phi-distance-field-learnability.md) ·
[rapfi-mix9svq-architecture.md](rapfi-mix9svq-architecture.md) §6.
