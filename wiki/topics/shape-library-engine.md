# The Shape-Library Engine — the gomoku AI: steer toward proven winning shapes, deny the opponent theirs

**One-line thesis.** Compile Allis's threat theory *out of data* instead of hand-building
it: mine every game for its proven winning **shapes**, reduce each to the exact minimal
configuration that makes the win inevitable, and play by finding the shortest *forcing*
path to an un-blockable **fork** of those shapes while denying the opponent theirs. The
distinguishing property — and the whole bet — is **the engine never hallucinates a win.**
Every winning claim traces to a proven shape and is verified at the leaf by the exact
solver. Its only failure mode is *blindness* (a win not yet in its vocabulary), never
*overconfidence*. That is the opposite of an AlphaZero value net, and for an interpretable
engine it is the property worth everything.

**This is the gomoku AI Jason wants to build** (2026-06-26 brainstorm). This page is the
plan outline; we refine from here.

Date: 2026-06-26. Hardware: M5 Max, 48 GB, MPS/MLX-Metal. Game: **freestyle** 9×9/15×15
(exactly-5-or-more wins, no overline/renju restriction — this matters for §3's monotonicity).

---

## 0. Working principles (Jason's — binding on every session that touches this)

- **Go all the way or fall on our face — then write it down, study it, get back up.** No
  tedious safe half-steps toward intermediate subgoals that "prove soundness in principle."
  Build the real thing; we will know it fails when it fails, and we'll diagnose it *from
  that vantage* with far more known variables than from here. **A negative result is a
  result we are happy with.** Jason is patient on this path; we don't have to crack it today.
- **Each design choice is stated decisively as "best we can think of today; there may be a
  better way."** No hedging with safe alternatives. Pick, write it down, move.
- **Full-board, not local** (§3). **Forks are a day-1 goal, not a later refinement** (§5).
- **Telemetry rides along the build for free; it is NOT a go/no-go gate.** We instrument
  (does monotonicity hold? how big are shape supports? does the library saturate?) *while*
  building the whole pipeline — we do not pause to certify a subgoal before proceeding.
- **Bitter lesson.** Don't hand-engineer structure (a locality window, a pattern grammar)
  the compute can find. Previously-impossible amounts of VCT compute are now tractable
  (§1); spend them. Mac first; rent B200s ($4/hr, ~couple grand acceptable) only if we must.

## 1. Where it sits — what already exists vs. what's new

| Layer | State | Where |
|---|---|---|
| **L0 — exact GPU VCT solver** | ✅ DONE | [gpu-vct-feasibility.md](gpu-vct-feasibility.md) §8 |
| **Stage-1 games + Stage-2 enabling-shape miner** | ✅ DONE — 63k shapes banked | [vct-backward-mining.md](vct-backward-mining.md) |
| **L1 — minimal shapes / the library** | 🔲 NEW (§3) | this page |
| **L2 — learned meta-VCT field** | 🔲 NEW (§4) | this page |
| **The player — fork-seeking pursuit search** | 🔲 NEW (§5) | this page |

What's already in hand is the *enabling tech and the raw material*:

- **L0** is the bitboard `ulong[4]` megakernel `solve_vct_mega_bb` — the whole AND/OR
  search on-device, **~1600× CPU labeling throughput** (~850–1020 solves/s @ B=16k–32k),
  **0 FP / 0 FN** over 320 VCT + 360 VCF real positions. This is what makes million-position
  VCT *and per-cell ablation* tractable — the engine that was "previously impossible."
- **The enabling-shape miner** already walks every won game back to its **first true VCT
  move** (the *setup*, not the line-bound kill) and has banked **63k full-board enabling
  shapes** (`scratchpad/vct_gpu_flat/`), run-lengths to 15. Open bottleneck there:
  **catalyst-move extraction** (verdict-only kernel ⇒ shapes carry `move=-1`; see
  [vct-backward-mining.md](vct-backward-mining.md) §5).

**The new work starts at: take those 63k enabling shapes and reduce each to its minimal
sufficient form.** That is L1.

## 2. The architecture — three layers, failing safe

- **L0 — exact VCT (built).** Leaf truth. Never wrong. Has the last word at every leaf of
  the player's search, so nothing downstream can make the engine claim a false win.
- **L1 — the shape library.** The minimal full-board **prime implicants** of won positions.
  This *is* the **monotone DNF of "the side to move has a VCT."** Matching is exact (a
  bitmask containment test), so an L1 hit is a *proof*, not an estimate.
- **L2 — the learned meta-VCT field** (where we AlphaZero — Jason's instinct). In the
  *fog* (no shape matches and exact search is too deep), regress the gradient toward
  shape-reachability under strong play. Trained on **verifiable** targets from L0/L1, so no
  value-overestimation. **Guides and orders only**; L0 still verifies the leaves.

The failure modes **compound safely**: L2 can be wrong but only steers *where we look*; L1
matches are exact by construction; L0 verifies. The engine's worst case is looking in the
wrong place, never believing a lie.

## 3. L1 — minimal shapes (the new core)

### The gift: VCT-win is monotone ⇒ a minimal shape is a prime implicant

In **freestyle**, `VCT-win(board)` is **monotone increasing in attacker-occupancy** (an
extra attacker stone never breaks a forced threat-win — it only ever makes/advances a five,
and can't hand the defender tempo) and **monotone decreasing in defender-occupancy** (an
extra defender stone can only block or counter). *(Renju's overline-forbidden rule would
break attacker-monotonicity — freestyle is why this holds. We **verify it empirically** as
free telemetry over the 63k shapes: random attacker-add / defender-add ablations should
*never* flip a win the wrong way.)*

Monotonicity makes the "exact minimum set of stones that makes the VCT inevitable" precisely
a **prime implicant** of a monotone boolean function (ML name: an *Anchor* — Ribeiro's
minimal sufficient explanation). Two clean consequences:

- **Three cell roles, not four.** Jason's `black / white / blank / either` collapses to
  **attacker-required / defender-forbidden / don't-care**. "Must be blank" (exactly empty)
  never survives as a minimal constraint: since an attacker stone never breaks the win,
  "defender-forbidden" (allow attacker *or* empty) always suffices and is **strictly more
  general** — it matches more boards, so the library compresses and generalizes for free.
- **Validating a candidate shape is a single solver call.** By monotonicity the worst case
  for a shape `(R = required-attacker, F = defender-forbidden)` is one specific board; if it
  wins, *every* board matching the shape wins. Extraction is then **greedy sequential
  ablation** (QuickXplain / Anchors): relax a cell, re-solve, commit if still a win. Both the
  per-cell tests and the across-shape sweep are **embarrassingly batchable** on the
  tail-bound L0 kernel — flip bits, re-solve a big batch, classify.

### Full-board, NOT local — for *soundness*, not just bitter-lesson compute

A far-away opponent stone can be a latent counter-four that decides whether the VCT is truly
forced. A local window would ignore it and emit an **unsound** shape (a claimed win that
isn't there). So shapes see the **whole board**. The elegance: a full-board representation
does **not** make shapes dense — the prime-implicant minimization turns everything that
doesn't matter into don't-care, so we get locality **exactly when the proof warrants it** and
globality **exactly when it warrants it**, with no hand-set window. *Don't impose the
structure; let minimization find it.* Two consequences fall out:

- **Matching = 256-bit bitmask containment:** `R ⊆ attacker(B)` and `F ∩ defender(B) = ∅` —
  two ANDs/compares, GPU-trivial.
- **Drop translation-invariance; keep D4.** You can't translate a full-board shape (it falls
  off the edge), but the board's dihedral symmetry is real — canonicalize under D4 for **8×
  dedup**.

### The library = the monotone DNF of "you have a VCT"

Dedup by D4-canonical key; **subsume** (a shape whose `R`,`F` are subsets of another's and
still wins makes the bigger one redundant — keep only maximally-general prime implicants).
The resulting set is literally the disjunctive normal form of the winning predicate. **Its
size is the complexity of gomoku's winning vocabulary** — thousands of terms, or millions?
Nobody knows. Mining it answers that. *(Discovery curve — unique minimal shapes vs games
mined — rides along as free telemetry, §0; saturation would say the vocabulary is finite,
growth says shapes aren't reducing enough and the representation needs rethinking. Not a
gate — an observation we collect while building.)*

## 4. L2 — the meta-VCT field (where we AlphaZero)

Most positions are not within forced distance of any shape — they're in the **fog**, too
deep for exact search. There we want Jason's "**meta-VCT**": *moving here has a higher chance
of reaching a VCT under strong play than moving there.* That is a learned potential:

- **Targets are verifiable, from L0/L1** — distance-to-nearest-shape, fork-reachability,
  `P(reach a winning shape under strong play)`. Not bootstrapped self-play value ⇒ **no
  overestimation**; the labels are proofs or measured distances.
- **Missed-VCT positions are L2's training set for free**: games where a VCT existed but even
  Rapfi didn't play it are exactly the fog L2 must learn to see into (and a Rapfi-strength
  probe besides).
- **L2 guides and orders search; L0 verifies leaves** ⇒ soundness preserved. L2 is plain
  PyTorch — it rents to B200s cleanly (§6).

## 5. The player — a two-player pursuit, not Dijkstra

"Min path between state *s* and a winning shape" is the right instinct, but the opponent
moves too, so it's a **two-player pursuit / race**, i.e. **threat-space search / df-pn**, not
single-source shortest path.

- **Potential-field framing.** L1 gives the *exact* potential near proven shapes; L2 regresses
  it into the fog; the move policy is its **gradient**; the opponent's field is the **dual**.
- **Distance must be tempo-aware.** Raw "stones still needed" is hopeless (unforced, the
  opponent blocks every other move). The real distance is the **length of a forcing sequence**
  (each step a four/open-three the opponent must answer) that establishes the shape. *(Open
  fork in §7: cheap admissible proxy vs. true recursive forcing-length.)*
- **Forks are first-class from day 1.** Single shapes get blocked. Strength is the **double
  threat** — reaching a position that is distance-0 from shape A **or** shape B with
  **disjoint refutations** so the opponent can't block both. The minimal shapes are the
  *atoms*; the player assembles **un-blockable conjunctions** of them. This is where the
  strength lives and where it is hardest.
- **Defense falls out symmetrically.** Move score ≈ `Δ(opp distance) + Δ(my distance)`; the
  dream move simultaneously advances my fork **and** occupies one of the opponent's
  required/forbidden cells.
- **v0-strong-soon vs optimal-later.** A v0 that matches shapes, verifies with L0, and uses
  raw shape-distance as eval is already a real (blunt) player — *not* a soundness toy. Full
  df-pn + L2 + fork assembly is the optimal-optimal we converge toward. **We expect something
  pretty strong long before it is optimal.**

## 6. Compute & portability bounds

Mac-first. The solver is **Metal/MSL — Apple-only**, so **mining + minimization stay on the
Mac** unless we CUDA-port the megakernel. **L2 is plain PyTorch** → rents to B200s ($4/hr,
~couple grand acceptable). Natural split if we outgrow the Mac: **mine/minimize here, train
L2 there.** Framing the envelope, not committing.

## 7. How it falls on its face (write it down NOW)

- **Library too specific** ⇒ poor coverage on novel boards. *Bet:* minimization's don't-cares
  + L2 generalization cover it. *Tell:* discovery curve never saturates.
- **The monotone DNF is effectively infinite** ⇒ throw compute, or the representation is wrong.
- **Distance heuristic not admissible** ⇒ search misled. *Mitigation:* L0 verifies leaves;
  distance only *focuses*, never *decides*.
- **Fork combinatorics explode** ⇒ need smart candidate generation for the conjunctions.
- **Off-distribution opponent** steers into fog L2 hasn't covered ⇒ adversarial self-play to
  grow the library where it's thin.
- **Monotonicity assumption false** in some freestyle edge case ⇒ the prime-implicant machinery
  needs a guard. *(That's why §3 verifies it empirically first thing.)*

We'll know which one bites when it bites — and diagnose it with the whole system standing.

## 8. Status / next

- **DONE:** L0 ([gpu-vct-feasibility.md](gpu-vct-feasibility.md) §8); Stage-1+2 miner, 63k
  enabling shapes ([vct-backward-mining.md](vct-backward-mining.md)).
- **OPEN (inherited):** catalyst-move extraction for the 63k shapes (GPU root-move output vs.
  parallel-CPU positive proofs — [vct-backward-mining.md](vct-backward-mining.md) §5).
- **NEXT (this plan):** (1) verify monotonicity (free telemetry); (2) **minimal-shape
  extraction** over the 63k → the D4-canonical, subsumption-reduced **library**; (3) the v0
  distance-field + fork player (leaf-verified by L0); (4) L2 meta-VCT field on verifiable
  targets.

**Cross-links:** [vct-backward-mining.md](vct-backward-mining.md) (the L1 mining substrate) ·
[gpu-vct-feasibility.md](gpu-vct-feasibility.md) §8 (L0, the enabling tech) ·
[allis-threat-theory.md](allis-threat-theory.md) (VCF/VCT formalism, forks/winning-combination) ·
[molecule-discovery-toolkit.md](molecule-discovery-toolkit.md) + [idea-pile.md](idea-pile.md) #10
(the "molecule ⊋ line" program this realizes) · [the-claw.md](the-claw.md) (non-line structure).
