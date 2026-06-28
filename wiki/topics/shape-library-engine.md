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
- **Performance is the enabling feature — the CALL-COST LAW (binding on every solver
  consumer).** One `solve_vct_mega_bb` call costs **one tail**: ~24–72 s set by the *single
  hardest board* in the batch, nearly **flat in batch size** up to ~16 k boards (compile is
  ~0.1 s — not the cost). **Throughput is free; latency is fixed.** So every consumer is
  **bulk-synchronous** — gather all boards into one ≤16 k call, *never* solve in a loop on a
  small batch — and caps `max_nodes` to shrink the tail. This is not tuning trivia: million-
  position VCT labeling and per-cell ablation are tractable *only* because the wall is ~flat in
  B. Headline + numbers: [gpu-vct-feasibility.md](gpu-vct-feasibility.md) (top banner).

## 1. Where it sits — what already exists vs. what's new

| Layer | State | Where |
|---|---|---|
| **L0 — exact GPU VCT solver** | ✅ DONE | [gpu-vct-feasibility.md](gpu-vct-feasibility.md) §8 |
| **Stage-1 games + backward enabling-shape miner** | ✅ DONE — 63k shapes banked | [vct-backward-mining.md](vct-backward-mining.md) |
| **First-VCT forward miner** (the §3 mining input) | ✅ DONE (2026-06-26) — `mine_first_vct.py`, append-only | §3 / §8 |
| **md-extraction** (the §3 ablation prerequisite) | ✅ DONE (2026-06-28, #91) — `max_depth`/`solve_md_min`, GPU-only, order-independent | [mega-vct-solver.md](mega-vct-solver.md) `max_depth` |
| **L1 — md-invariant stencil minimizer** | 🟡 BUILT + measured (2026-06-28, §3/§8) — `md_minimize.py` | this page |
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
- **L1 — the stencil library.** The minimal **stencils** of won positions — each a relative,
  explicitly-typed (`B` / `.` / `p`), single-orientation reduction of a board that is a
  *provable* VCT (formal handle: a monotone-function *implicant*; see §3 Naming). Matching is
  exact — a structural bitmask fit — so an L1 hit is a *proof*, not an estimate. (We mine one
  stencil per VCT and bank others as free augmentation; we do **not** chase the full DNF, §3.)
- **L2 — the learned meta-VCT field** (where we AlphaZero — Jason's instinct). In the
  *fog* (no stencil matches and exact search is too deep), regress the gradient toward
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
guarantees *this* VCT** — a small **stencil** in *relative* coordinates. Absolute position is
throwaway provenance (`p=(3,4)` doesn't matter); "`.BBBBp` fits here" is the asset. **A stencil's
identity is `(cells, mate-distance)`** — the cells *and* how long the forced win is (see the §3
correction below: a load-bearing stone is one whose removal changes md, swapping the win for a
different one). Four cell types:

- **`B`** — attacker stone required.
- **`.`** — must be **empty** (open). These reserve the room the threat needs to develop.
- **`p`** — the catalyst move (empty, distinguished).
- **`W`** — defender stone required — a **load-bearing** opponent stone that pins *this* line
  (by monotonicity it is never needed for a win to *exist*; it is needed for the win to be
  *this* win — it carries **identity, not existence**). Corrected in 2026-06-26 — see below.
- everything else — implicit **don't-care** (not even stored).

We take "empty" for `.` over the looser "attacker-or-empty": it matches fewer boards but every
match is more conservative, and loosening can only *add* matches later. The `.` cells reserve the
threat's room; the `W` cells fix which forced line is operative.

### The minimization — context ablation (the all-white certificate FAILED; corrected 2026-06-26)

> **Correction (probed 2026-06-26).** The original plan here was a one-call certificate: label
> cells `B` / `.` / don't-care, build the **meanest board** by filling every don't-care with a
> white (defender) stone, and let a single solve certify the whole match-set. **It does not
> work.** A first-VCT board has ~200 empty cells; filling them all white hands the *defender*
> ~400 five-in-a-rows — the defender has simply **won** — so the solver correctly returns
> NO-WIN for *every* shape and no cell is ever removable. The monotonicity is real ("more
> defender never helps the attacker") but **vacuous past the point the defender wins**: the
> abstract worst case is an illegal, defender-won board that no legal matched position could
> be. The lesson: the worst case must stay a **legal** board. (Evidence: `probe_meanest.py`,
> 6 shapes, all-white → win=False, ~390–476 defender-5s each.)

So we minimize by **context ablation** — remove stones from the *real, legal* mined board,
never synthesizing a white sea. Start from the mined first-VCT board P0 (a proven win); greedily
try removing each stone; re-solve the **still-legal** board; keep the removal if the stone was
genuinely irrelevant. **But the test is mate-distance invariance, NOT "still a win"** — see the
correction below.

> **Correction #2 — existence is the wrong invariant; ablate on mate-distance (2026-06-26,
> Jason).** "Remove it if it's still a win" is broken, and the monotonicity lemma is exactly
> why. *Existence*-monotonicity (proved: in freestyle, a white stone never makes a black win
> *appear* — adding white only ever hurts or is neutral; the one exception is **renju**, where a
> white stone can defuse black's *forbidden* double-three and so *enable* a win — which is why
> freestyle is load-bearing) tells us only whether **some** win survives. It says nothing about
> **which** win. A **load-bearing** stone is one whose removal **changes the mate distance**:
> pull the white stone that was blocking the fast kills and a **17-ply** forced win collapses to
> a **2-ply** one — *a different stencil entirely*, often one that doesn't even match the board it
> came from (its now-required-`.` cell is where that white blocker sat). And by monotonicity this
> is a **white-only** phenomenon (removing white can only shorten), so **load-bearing white is the
> rule, not the exception, in the long VCTs** — the run-15s are long *precisely because* white
> denies the short wins. Naive win-invariant ablation would shred every hard-won long VCT into a
> trivial short stencil. So the criterion is **md-invariance**, three outcomes per removal:
> - removal → **still wins at the same md** ⇒ irrelevant, drop it (a true don't-care);
> - removal → **wins at a shorter md** ⇒ **load-bearing**, keep it (it forces *this* line — if a
>   `W` stone, it becomes a `W` cell);
> - removal → **no win** ⇒ existence-critical, keep it.
>
> The anchor is therefore the pair **`(p, md)`**, not just `p`: we are stencilizing "the 17-ply
> VCT," and md is a coordinate of the object. **Hard dependency this creates:** the test needs
> *depth*, but `solve_vct_mega_bb` returns only `(win, hit_cap)` and caps *nodes*, not depth. So
> md-extraction (the move/depth-out-of-the-kernel work open since
> [vct-backward-mining.md](vct-backward-mining.md) §5) moves from "nice to have" to a **blocking
> prerequisite for L1** — or we depth-cap solves (e.g. "still win at `md−1`?" detects a
> shortening). *Not cracked yet; written down. We try, we learn, we write it down.*

What survives md-invariant ablation is the stencil: the attacker skeleton, the empties the
forcing sequence needs, and the load-bearing `W` stones that pin this exact line. The algorithm
*finds* which cells matter; we never hand-set a window (§0).

### Soundness: a candidate index that L0 verifies (corrected)

Context ablation certifies a stencil **in the context it was mined / in isolation**, not against
an adversary free to pre-place defender stones anywhere — that worst case is the illegal
all-defender board above, untestable in one solve. For a short VCT (mate-in-1/2, the bulk of the
corpus) the gap is nil: the attacker wins before the defender can act. For a longer VCT a
pre-existing opponent counter-threat could in principle out-race it. So a stencil is best read as
a **fast, high-precision index of known winning motifs, not a standalone proof** — which is
exactly the §2 architecture: **L1 proposes, L0 verifies the leaf.** A stencil match flags "a known
win is very likely here"; one cheap L0 solve on the real board confirms it before the engine ever
claims the win. The **never-hallucinates** property lives entirely in L0, so an imperfect stencil
costs a little search, never a false claim. Matching stays a structural bitmask scan over relative
coords — every fit is a *candidate*, and L0 turns the candidate into truth. (This is a deliberate
downgrade from the original "an L1 hit is a proof" claim, forced by the correction above.)

> **Performance — the minimizer must be bulk-synchronous (measured 2026-06-26).** Both the
> ablation sweep and match-verification are **tail-bound**: the call wall is set by the single
> hardest board, ~24–72 s nearly independent of batch up to ~16k boards
> ([gpu-vct-feasibility.md](gpu-vct-feasibility.md) §8; compile/import is ~0.1 s — *not* the
> cost). Per-board cost swings **~350×** between a 25-board call (1.5 s/board) and a 16k-board
> call (0.0044 s/board), so row-at-a-time minimization is the trap. The minimizer marches **all
> shapes in lockstep — one candidate-removal per shape per call, ~16k boards/call** — and **caps
> `max_nodes`** to shrink the tail; a CAP verdict just keeps a cell (fail-safe, over-specific
> never unsound).

### Empirical certificate property — measured, not just argued (2026-06-27, #88)

The "candidate index, not a proof" verdict above stands *in general*, but we can now say
**precisely when a stencil IS proof-grade**, and the answer is stronger than expected. Two
things made it testable: (a) the solver's new **`carriers`** output (#88,
[mega-vct-solver.md](mega-vct-solver.md)) — the load-bearing OWN stones, so the extracted
shape `support ∪ carriers` is a *complete, replayable* object (not just the openings); and
(b) the tail-bound oracle, which stamps a shape onto thousands of boards in a few calls.

**Experiment** (`scripts/threat_shapes/certificate_falsification.py`; pool = 4096 real Rapfi
positions, `max_nodes=500`):

1. Mine clean attacker VCTs → **660** wins.
2. **Self-containment** — rebuild each as *carriers alone* on an empty board (support empty,
   nothing else): **660 / 660 (100%) still win.** For offense, `(carriers, support)` is a
   *complete* description — no hidden own stone, no surrounding context needed.
3. **Transfer** — bolt ≤12 random opponent stones (off support & carriers) onto each
   self-contained shape, keep boards where the defender has **no VCT of its own**, re-solve:
   **0 / 2913 refuted.** The attacker always still wins. **Control:** when the defender *does*
   get a counter-VCT the attacker loses ~7% (266/287 won) — the filter is not vacuous, and
   **counter-tempo is the sole breaker.**

**What it means.** A stencil that wins in isolation wins on *any* board where it fits and the
defender has no counter-threat — **by the same forcing line**, so the expensive AND/OR search
runs **once per shape** and every placement wins *for the same reason*. This is the soundness
of **Allis's dependency-based / threat-space search**
([allis-threat-theory.md](allis-threat-theory.md) §3) — not new to the field, but now
**operational and falsifiable** for our mined stencils, and **confirmed** (0 counterexamples).
It promotes L1, *for the self-contained subset*, from "candidate index" toward a **certificate
engine**: match the shape (structural) + confirm the defender has no counter-VCT (one cheap
flipped L0 solve) ⇒ the win is proven, **without re-running the attacker search**.

**Honest bounds.** (i) "0/2913" is strong evidence, not a mechanized QED — the proof *is* the
forcing-sequence-composition argument (Allis), trusted but not formalized here. (ii) The safety
filter used ("defender has no VCT") is *stronger than necessary*; the control shows even a
defender-VCT usually fails to refute (the attacker moves first), so the true threshold is
*immediate* defender tempo (a four/five arising mid-sequence) — the next thing to pin down
(push the harness to the exact-tempo filter, near-edge translations, millions of placements).
(iii) This is the **self-contained / offensive** subset (here 100% of attacker VCTs); a
*defense*-flavored mining will surface **`W`-dependent** shapes that do NOT win in isolation —
those need the `W` channel before they earn a certificate. *(Update 2026-06-27, #90: the
over-inclusive `W` channel now SHIPS — `solve_vct_mega_bb(return_w=True)`, the `opp` mirror of
`carriers`; see [mega-vct-solver.md](mega-vct-solver.md) `w` section. It over-approximates the
minimal load-bearing `W`, which still awaits md-extraction. And `W` is corroborated as
identity-not-existence: removing ALL defender stones from clean attacker VCTs preserved
**660/660** wins — `scripts/threat_shapes/w_channel_probe.py`.)* (iv) The shape tested is the
**over-inclusive** `support ∪ carriers`, available **today** — the certificate property does
**not** wait on the md-extraction blocker that gates *minimal* ablation.

### Hardened — the EXACT immediate-tempo boundary, and the #88 filter was unsound (2026-06-27, #89)

#88's honest bound (ii) flagged the "no defender VCT" filter as *stronger than necessary* and
named the real threshold as **immediate defender tempo**. #89 replaces the filter with that
exact condition, scales the falsification, and adds an **adversarial** test for the one case a
start-of-board filter cannot see. The result is both a **harder-passed theorem** and a
**correction to #88**: the start-of-board immediate-tempo theorem **held — 0 counterexamples**,
and the adversarial mid-sequence attack did **not** break it.

**The exact boundary.** A board is *tempo-unsafe* iff the defender has an **immediate four-move
or five-completion** — precisely the megakernel's own `def_tempo(opp, empty) =
completion_mask(opp) ∪ gen_forcing(opp)` (the guard it already uses to reject a non-forcing
attacker three). #89 lifts this out as a **cheap per-board probe**
`defender_has_four_or_five(boards)` (one set-algebra kernel, **no AND/OR search** — far cheaper
than the flipped VCT solve #88 used), validated against `bb.probe`'s `completion_mask` plus a
deterministic unit self-test (`--self-test`).

**Numbers** (`certificate_falsification.py --pool 4096 --seeds 0,1,2 --cap 400`, `max_nodes=500`):
- **Self-containment:** **2036 / 2037** (99.95%) attacker VCTs win from carriers alone (one
  miss, seed 1) — confirming #88's 100% at scale, honestly not *literally* 100%.
- **Random transfer, EXACT tempo filter:** **0 / 6613** tempo-safe placements refute.
- **Adversarial mid-sequence:** **0 / 31636** tempo-safe placements refute (mechanism below).
- **Near-edge:** **0 / 9075** fitting flush edge/corner translations fail.
- **Theorem, machine-checked each run:** *every* refutation observed carries immediate defender
  tempo; **tempo-safe refutations = 0** in every block.

**The #88 filter is not sound (the correction).** A **lone defender four-move** (e.g. an
incidental open-three from the bolted-on stones) is *immediate tempo but not a VCT*, so #88's
"no defender VCT" filter **admits it** — seed 2 caught **2 / 8756** such boards refuting under
#88, **both with immediate tempo**, exactly the ones #89 excludes. Symmetrically, #89 **admits**
the boards #88 wrongly dropped: **11** boards (across seeds) are tempo-safe *but* hand the
defender a **slower** VCT, and the attacker **wins all 11** — the attacker-moves-first dominance
#88 only conjectured. So `def_tempo` is not merely a more-permissive filter; it is the
**correct** decision boundary, and "no defender VCT" was simultaneously **too strong** (drops
safe boards) *and* **too weak** (admits tempo refuters).

**The mid-sequence subtlety, and why the start probe suffices (the part-C attack).** A forced
defender **block** can itself create a counter-four mid-sequence — invisible to a start-of-board
check. We attacked this head-on: for every support cell `s` (a forced-block candidate), place
two defender stones collinear to `s` so that **if** the defender is ever forced to block at `s`,
the block completes a defender three = a four-move (`def_tempo`). The construction is
**verifiably live**: **~93%** (25776/27740) of these start-tempo-safe boards flip to
tempo-**unsafe** the instant the anchor is blocked. Yet **0 / 31636** refute. Two reasons, one
**proved** and one **measured**:
- *(proved)* A defender **real four** (a *five-completion* — the only thing that defeats an
  attacker **four**) built from a **single** forced block needs **three** pre-placed collinear
  stones, which is already a `(3-own, 2-empty)` window = **immediate tempo at the start**. So
  single-block real-fours are *always* start-visible; the probe is **exactly right** for
  four-only (VCF-style) lines, which dominate the short mined corpus.
- *(measured)* The only residual evasion — a single-block four-**move** (two pre-placed)
  defeating an attacker **forcing-three** — never materialized across 31636 primed boards: where
  `s` is genuinely the operative forced block, the attacker's continuation is a four or has an
  alternative winning move. (Every refutation in the adversarial set — 134 in seed 2 — was a
  start-tempo-**unsafe** board the probe correctly flags; the probe is a *perfect* refutability
  classifier on this set.)

**Near-edge (part D).** Stamping self-contained shapes **flush** against each edge and corner
(breathing room cut by the **boundary**, not a stone), **0 / 9075** that fit (all
`support ∪ carriers` in-bounds) lose. So **"the footprint fits on-board" is a sufficient
placement-legality test** — the support window captures the room the forcing line needs; the
boundary removes nothing extra.

**Honest bounds (unchanged in spirit).** Still **measured-0, not a mechanized QED** — the proof
remains the Allis forcing-composition argument. The one gap *not* closed by construction: a
**two-block** real-four manufactured mid-sequence (forcing the defender through two specific
blocks that together complete a four) would need move/line extraction to target deliberately; we
reached it only incidentally. Defense-flavored **`W`-dependent** shapes remain out of scope
(they don't self-contain). Net for the self-contained / offensive subset: the certificate is
**match the shape + one cheap `def_tempo` probe ⇒ proven win, no attacker search** — and the
probe boundary is now the *exact* one, not a conservative proxy.

### Conservative on symmetry — on purpose

We use **translation only**: the same relative stencil laid down elsewhere is just another
*candidate* placement, and L0 verifies each match anyway, so translation costs us nothing in
soundness. We do **not** fold D4 (reflections / rotations / transposes) yet — not because it's
wrong, but to keep the candidate set on ground we trust. Each found orientation is its own
entry; an open four seen horizontally and vertically is two stencils today. We lose nothing:
symmetry is a purely **additive** 8× dedup (more candidate generators) we can fold in later
without invalidating a single banked stencil.
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
- **The single-call "meanest board" certificate is vacuous** (DISCOVERED 2026-06-26): the
  all-white worst case is a defender win, not a usable bound — minimization had to fall back to
  legal-board **context ablation** (§3 correction). The deeper risk it exposes: a stencil is only
  truly proof-grade *in isolation*; in real context a faster opponent threat can refute it ⇒
  **L0 verifies every match** (§2), so the engine still never hallucinates. **MEASURED 2026-06-27
  (§3 certificate property):** the in-isolation property is real and strong — 660/660 attacker
  VCTs win from carriers alone, and **0/2913 *tempo-safe* placements refute**; the refuter is
  *exclusively* defender counter-tempo, exactly as dependency-based-search theory predicts. So
  the "risk" is now bounded and named: a stencil is a certificate up to defender counter-tempo.
  **HARDENED 2026-06-27 (#89):** the exact boundary is **immediate `def_tempo`** (a cheap
  four/five probe, not a flipped VCT solve); **0** refutations across **38k** tempo-safe random +
  adversarial placements and **9k** near-edge translations, the mid-sequence-block attack did not
  break it, and the #88 "no-VCT" filter was shown *unsound* (it admits lone-four-move refuters).

We'll know which one bites when it bites — and diagnose it with the whole system standing.

## 8. Status / next

- **DONE:** L0 ([gpu-vct-feasibility.md](gpu-vct-feasibility.md) §8); backward Stage-1+2 miner,
  63k enabling shapes ([vct-backward-mining.md](vct-backward-mining.md)).
- **DONE (2026-06-27, #88):** solver **`return_carriers`** (the load-bearing-stone `B` channel;
  [mega-vct-solver.md](mega-vct-solver.md)) ⇒ `support ∪ carriers` is a complete, replayable
  stencil **now**, *without* the md blocker. And the **certificate property is measured** (§3,
  2026-06-27): **660/660** attacker VCTs self-contained, **0/2913** tempo-safe placements refuted
  ⇒ L1 is a *certificate engine* for the self-contained subset. Harness:
  `scripts/threat_shapes/certificate_falsification.py`; stencil miner:
  `scripts/threat_shapes/mine_support_shapes.py`.
- **DONE (2026-06-27, #89):** **hardened** the certificate property (§3) — the exact boundary is
  the megakernel's own **`def_tempo`** (immediate defender four/five), exposed as a cheap probe
  `defender_has_four_or_five` (no AND/OR search). 3 seeds × pool 4096: **2036/2037**
  self-contained, **0/6613** random + **0/31636** *adversarial* (mid-sequence forced-block)
  tempo-safe placements refute, **0/9075** near-edge translations fail; **every** refutation
  carries immediate tempo. Correction: the #88 "no-VCT" filter is **unsound** (admits
  lone-four-move refuters) — `def_tempo` is the *correct* boundary. Still measured-0, not a
  formal QED.
- **DONE (2026-06-27, #90):** the **`W` channel** ships — `solve_vct_mega_bb(return_w=True)`,
  the `opp` mirror of `carriers` (`w = opp ∩ ⋃_support COLLIN`, the over-inclusive load-bearing
  defender stones), additive + default byte-identical across all 8 prior variants; tests +
  [mega-vct-solver.md](mega-vct-solver.md) `w` section. Empirically `W` is **identity, not
  existence** (removing all defender stones kept **660/660** attacker VCTs;
  `scripts/threat_shapes/w_channel_probe.py`). The *minimal* load-bearing `W` still needs the
  md-extraction blocker below.
- **DONE (2026-06-26):** **first-VCT forward miner** — `scripts/threat_shapes/mine_first_vct.py`,
  the §3 mining input. Forward scan, both colors, trit verdict + defer-on-prefix-CAP,
  append-only/resumable. Smoke (100 games, `max_nodes=6000`): 12 certified firsts, 88 deferred
  (the deferred rate confirms certifying "first" wants a fatter budget than the backward walk
  ever needed — exactly the negative-prefix cost from [vct-backward-mining.md](vct-backward-mining.md) §3).
  The catalyst move p is extracted at emit time (anchors minimization), so L1 does **not** wait
  on the inherited move-extraction bottleneck.
- **DONE (2026-06-28, #91) — md-extraction, the blocker is GONE.** The depth-cap variant
  `solve_vct_mega_bb(max_depth=)` + the wrapper `solve_md_min` give the **order-independent**
  mate distance md_min on GPU, by binary-searching a per-board frame cap (the cut returns clean
  `ret=0` without `hit_cap`, before any move ⇒ default verdict byte-identical, carriers/w safe).
  See [mega-vct-solver.md](mega-vct-solver.md) `max_depth` / invariant #9. **GPU-self validated**
  (byte-identical-vs-HEAD, depth-monotonicity, md bracket, md_min == an independent linear scan);
  **NO CPU cross-oracle** — a *live* CPU md is mis-calibrated (kernel `candidate_own` own-only is
  narrower than CPU's any-stone candidate set ⇒ `md_gpu > md_cpu` with no bug) and would
  re-summon the retired solver. md is in **FRAME units** (four=+1, three=+2, inline win collapses)
  — sufficient for shortening-detection, never reconciled to the CPU `mate_distance`.
- **DONE (2026-06-28, #91) — the md-invariant minimizer, BUILT + MEASURED** (`md_minimize.py`).
  Cumulative lockstep ablation, directional single-cap tests (freestyle monotonicity): OWN stone
  at cap `md0` (clean-win → redundant DROP / nowin → load-bearing `B` KEEP); OPP stone at cap
  `md0−1` (clean-win → a shorter mate opened → load-bearing `W` KEEP / nowin → DROP); cap → KEEP
  (fail-safe). No windowing (sound; sidesteps the found-line-vs-shortest-line risk). **Results on
  `molecule_gold` (16,345 boards):** md_min in 19.6 s, 0 ceiling pressure; orig 13.2 → **4.91**
  (B+W) stones (63% ↓); **load-bearing W is the long-VCT phenomenon, MEASURED** — W-rate by md0
  is **0% at md0=1** (degenerate inline win), **72% at md0=2, 100% at md0≥4**; and the cheap `w`
  channel (#90) is a **~10× over-approximation** of the true load-bearing W (88,637 → 8,694).
  Honest bounds banked: **73% of `molecule_gold` is md0=1** (root fork-three collapse ⇒ poor W
  substrate; the deeper real-game `enable_serial` corpus is the cleaner control); the corpus is
  defender-perturbed (inflates W); vocabulary **not fully saturated** at 16k (902 distinct,
  decelerating). Full record: [TRAINING_WIKI.md](../../TRAINING_WIKI.md) 2026-06-28.
- **NEXT (this plan):** (1) ✅ **stencil minimization** — DONE above (md-invariant, GPU sole
  oracle, bulk-synchronous). Refinements: the `enable_serial` deep-VCT contrast; **windowing**
  for the dense/deep regime (no-window ablation is `maxlen`-round-bound — the perf cost seen on
  `enable_serial`); ablate the **support openings** too (only stones are ablated today); emit
  `md = sp + leaf_offset` to undo the inline collapse. (2) the structural-match index + mobility
  sweep, each match **L0-verified** (L1 proposes, L0 proves); (3) the v0 distance-field + fork
  player; (4) L2 meta-VCT field on verifiable targets.
- **DEFERRED (conservative, §3):** D4 / reflections / rotations — purely-additive 8× dedup,
  folded in later; and the "empty → attacker-or-empty" loosening.

**Cross-links:** [vct-backward-mining.md](vct-backward-mining.md) (the L1 mining substrate) ·
[gpu-vct-feasibility.md](gpu-vct-feasibility.md) §8 (L0, the enabling tech) ·
[allis-threat-theory.md](allis-threat-theory.md) (VCF/VCT formalism, forks/winning-combination) ·
[molecule-discovery-toolkit.md](molecule-discovery-toolkit.md) + [idea-pile.md](idea-pile.md) #10
(the "molecule ⊋ line" program this realizes) · [the-claw.md](the-claw.md) (non-line structure).
