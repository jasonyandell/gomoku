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
| **Stage-1 games + backward enabling-shape miner** | ✅ DONE — 63k shapes banked | [vct-backward-mining.md](vct-backward-mining.md) |
| **First-VCT forward miner** (the §3 mining input) | ✅ DONE (2026-06-26) — `mine_first_vct.py`, append-only | §3 / §8 |
| **L1 — stencil extraction / the library** | 🔲 NEW (§3) | this page |
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

## 3. L1 — the stencil library (the new core)

*(2026-06-26 brainstorm with Jason replaced the original "prime-implicant / monotone-DNF"
framing of this section. The math underneath — monotonicity, the single-worst-case-board
certificate — is unchanged; the framing is now first-VCT mining + a conservative, explicitly-
typed, single-orientation **stencil**. The boolean-logic lineage is recorded under Naming.)*

### What we mine: the guaranteed-FIRST VCT, from real games

The backward miner ([vct-backward-mining.md](vct-backward-mining.md)) walked won games from
the END and read the contiguous won-suffix — an engineering shortcut from when proving "no
VCT yet" on every position was unaffordable. But **VCT-existence is not monotone across
plies**: a forced win opens at ply k, closes at k+2 (the player didn't take it; the
opponent's reply blocked it), reopens later. The backward walk only ever sees the *last*
window. With the tail-bound megakernel we can afford the honest target: **scan every position
from move 0 and take the first ply where the side-to-move has a forced VCT** — the moment the
game was first theoretically over, whoever was to move. Scope is **any VCT by anyone**:
`GameState.board` is side-to-move-relative, so `board[0]` is the attacker frame at *every*
ply (no swap), and both colors — decisive games and draws alike — fall out of one forward
scan. `stm != winner` on an emitted shape ⇒ a **missed VCT** (a player had a forced win and
didn't take it — exactly L2's fog set, §4).

"First" is trustworthy only with a **clean negative prefix**. Per position the kernel returns
a **trit** — WIN / NO-WIN-proven / CAP (node budget exhausted, unknown) — and the first WIN
at ply *w* counts as *first* only if every earlier ply is **proven** NO-WIN. A CAP in the
prefix could itself hide an earlier VCT, so we **defer the whole game** to a fatter-budget
queue rather than emit a maybe-wrong first. Pure `(game, budget) → first | defer`; the defer
pile is re-mined later at a bigger budget (the deferred set *is* the population of genuinely
deep games). Fail-safe: the kernel is 0-FP, so we never report a *false-early* first, we only
postpone the hard games. Code: `scripts/threat_shapes/mine_first_vct.py` — append-only
`first_vct.jsonl.gz` + `defer.jsonl.gz` + `manifest.txt`, trivially resumable.

### What we store: a relative, explicitly-typed, single-orientation stencil

A mined first-VCT is a full board; we reduce it to the **minimum specification that still
guarantees the VCT** — a small **stencil** in *relative* coordinates. Absolute position is
throwaway provenance (`p=(3,4)` doesn't matter); "`.BBBBp` fits here" is the asset. Three
**conservative** cell types:

- **`B`** — attacker stone required.
- **`.`** — must be **empty** (open). These reserve the room the threat needs to develop.
- **`p`** — the catalyst move (empty, distinguished).
- everything else — implicit **don't-care** (not even stored).

We deliberately take "empty" over the looser "attacker-or-empty": it matches fewer boards but
every match is sound, and loosening can only *add* matches later — never invalidate a banked
one. The `.` cells are load-bearing: they are what makes the stencil self-contained and
**movable**.

### The minimization, and why one solver call certifies it

Monotonicity (freestyle, no overline rule): VCT-win is monotone **up** in attacker stones (an
extra attacker stone never breaks a forced win) and monotone **down** in defender stones (an
extra defender stone only blocks/counters). So a candidate stencil has exactly one **meanest
board**: keep `B` black and `.`/`p` empty, fill **every don't-care cell with a white
(defender) stone** — the worst possible surroundings. If the attacker still wins *that*
board, every board the stencil matches wins. **One solver call certifies the whole
(exponential) match-set.** *(Monotonicity itself rides along as free telemetry — random
attacker-add / defender-remove ablations over real shapes must never flip a win the wrong
way; a violation means a freestyle edge case breaks the machinery, §7.)*

Extraction is **greedy demotion, anchored on the move p** (so it can't drift onto an
unrelated VCT elsewhere on the board): start from the mined board (every attacker stone `B`,
every empty `.`); try demoting each cell to don't-care; rebuild the meanest board; re-solve;
keep the demotion if it still wins. What survives is the stencil. Batched across thousands of
shapes through the tail-bound kernel — the same throughput trick as mining. Walk the open
four: `.BBBBp` → demote the far `.` → meanest board makes it white `OBBBBp` → p still makes
five → drop it → land on `BBBBp`; try dropping any `B` → only a four, p makes no five → keep.
The algorithm *finds* which cells matter; we never hand-set a window (bitter lesson, §0).

### Prove once, match everywhere — and mobility falls out

The minimization's final state **is** the proof: it won in the meanest environment. So a
**structural fit** of the stencil anywhere on any board is sound for free — a real placement
has the same `B`-black and `.`-empty cells, and its don't-cares are never meaner than the
all-white board we already beat. "Where is this a VCT" becomes a **bitmask scan**, not a
solver call per location. A stencil is **mobile to every on-board placement that fits**: the
open four wins in many specific locations, while a *true* open four (six collinear cells)
cannot even be **placed** on a 5×5 → not a guaranteed win there, exactly as the geometry
demands. Edge-leaning wins handle themselves — their `.`/`p` cells fall off-board at interior
placements, so they simply fit in fewer spots. This kills the per-placement solver sweep: one
proof, structural matching forever.

### Conservative on symmetry — on purpose

We use **translation only**, which is provably safe (the meanest-test verdict depends on the
stencil's relative geometry, invariant under translation as long as it fits on-board). We do
**not** fold D4 (reflections / rotations / transposes) yet — not because it's wrong, but to
keep correctness on ground we trust. Each found orientation is its own entry; an open four
seen horizontally and vertically is two stencils today. We lose nothing: symmetry is a purely
**additive** 8× dedup we can fold in later without invalidating a single banked stencil.
*(Discovery curve — distinct stencils vs games mined — rides along as free telemetry, §0:
saturation says the vocabulary is finite; runaway growth says the stencils aren't reducing
enough and the representation needs rethinking.)*

> **Naming (resolved 2026-06-26: stencil).** We wanted a word for "relative, explicitly-typed,
> single-orientation reduction that is a *provable* VCT." Surveyed, as recorded lineage, none a
> perfect single fit: *prime implicant* / *implicant* (boolean logic — a partial assignment that
> forces a function true; exact, but drags in the monotone-DNF program Jason set aside); *Anchor*
> (Ribeiro, ML interpretability — a minimal sufficient rule guaranteeing a prediction;
> near-exact, but Anchors permit <100% precision and ours is *exact*); *certificate / witness*
> (complexity — a compact proof object); *stencil* (grid computing — a relative-offset pattern
> slid over a grid; captures the spatial/mobility form). Chosen: **stencil** — a relative-offset
> typed mask laid down wherever it fits, which is exactly the mobility property; it cleanly
> disambiguates from *shape* (the raw mined board). The framing: we reduce a board containing a
> VCT to a stencil. *implicant* remains available as the formal handle.

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

- **DONE:** L0 ([gpu-vct-feasibility.md](gpu-vct-feasibility.md) §8); backward Stage-1+2 miner,
  63k enabling shapes ([vct-backward-mining.md](vct-backward-mining.md)).
- **DONE (2026-06-26):** **first-VCT forward miner** — `scripts/threat_shapes/mine_first_vct.py`,
  the §3 mining input. Forward scan, both colors, trit verdict + defer-on-prefix-CAP,
  append-only/resumable. Smoke (100 games, `max_nodes=6000`): 12 certified firsts, 88 deferred
  (the deferred rate confirms certifying "first" wants a fatter budget than the backward walk
  ever needed — exactly the negative-prefix cost from [vct-backward-mining.md](vct-backward-mining.md) §3).
  The catalyst move p is extracted at emit time (anchors minimization), so L1 does **not** wait
  on the inherited move-extraction bottleneck.
- **NEXT (this plan):** (1) **stencil minimization** (§3) — greedy demote-to-don't-care under
  the meanest-board certificate, anchored on p → relative, explicitly-typed, single-orientation
  stencils, cross-validated against the independent CPU `solve_vct`; (2) the structural-match
  library + mobility sweep; (3) the v0 distance-field + fork player (leaf-verified by L0);
  (4) L2 meta-VCT field on verifiable targets. Monotonicity telemetry rides along (1).
- **DEFERRED (conservative, §3):** D4 / reflections / rotations — purely-additive 8× dedup,
  folded in later; and the "empty → attacker-or-empty" loosening.

**Cross-links:** [vct-backward-mining.md](vct-backward-mining.md) (the L1 mining substrate) ·
[gpu-vct-feasibility.md](gpu-vct-feasibility.md) §8 (L0, the enabling tech) ·
[allis-threat-theory.md](allis-threat-theory.md) (VCF/VCT formalism, forks/winning-combination) ·
[molecule-discovery-toolkit.md](molecule-discovery-toolkit.md) + [idea-pile.md](idea-pile.md) #10
(the "molecule ⊋ line" program this realizes) · [the-claw.md](the-claw.md) (non-line structure).
