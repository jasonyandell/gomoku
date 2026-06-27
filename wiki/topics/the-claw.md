# The Claw — a line-decorrelated defensive crystal (and a representational lever)

**Provenance & credit.** "The claw" is **Jason's name** for a defensive pattern he found
**independently, by hand, through play, in the 1990s** — long before this project. The underlying
*object* is known to combinatorial game theory — periodic blocking/pairing tilings of Z² for
k-in-a-row (Györffy–Makay–Pluhár, Szeged; Beck, *Tic-Tac-Toe Theory*, 2008) — and credit for it goes
to that prior art, loudly. (**Jason's rule:** *if it's been discovered, it goes to the discoverer, every
time.*) Jason's contribution is the **independent rediscovery, the name, and — new — its use as a
representational lever**: the claw as the canonical proof that *non-line molecules exist and are invisible
to line-based engines.* He described it as:
*impenetrable; no good player lets you keep it long, but it can confound them and shape
the board in powerful ways.* Both halves of that lived intuition turn out to be **theorems**
(see §2–§3). Surfaced into the wiki 2026-06-25 during a representation brainstorm; the math
and the engine-internals were verified by two subagents (credited in §5).

This page is the canonical home for the claw: what it is, why it's provably special, why a
line-based engine is blind to it, and what it tells us to build. It is the worked **defense
axis** example feeding [idea-pile.md](idea-pile.md) **#10 (threat-chemistry representation)**
and the **existence proof** for **#3 (judo — steer into Rapfi's blind spots)**.

---

## 1. What it is — the mod-5 knight's-move lattice
A scatter of stones on **knight's-move spacing** through a region. The exact object:

> **L = { (x, y) ∈ Z² : 2x + y ≡ 0 (mod 5) }**

- **Density 1/5** — one stone per 5 cells (provably minimal: each disjoint 5-window needs ≥1 blocker).
- Nearest neighbors sit at **knight's-move / (1,2)-type offsets** — that's the visual "claw."
- The defining property: **knight spacing = zero shared lines.** Two stones a knight's move
  apart share *no* run of 5 consecutive collinear cells in any of the 4 directions. So the
  claw is **line-decorrelated by construction** — and that single fact drives everything below.

## 2. Why it's a *perfect* blocker — and *unique*
**Perfect-blocker theorem.** Every 5-window (5 consecutive collinear cells) in all four
directions contains **exactly one** point of L.
*Proof.* Let f(x,y)=2x+y mod 5. Along a direction with step d, f changes by a constant
c = f(d): horizontal c=2, vertical c=1, main-diag c=3, anti-diag c=1 — all coprime to 5. The
five f-values a, a+c, …, a+4c over a 5-window are a bijection of Z/5 ⇒ residue 0 appears
**exactly once** ⇒ exactly one stone per window. ∎
**Corollary — "no lines of its own":** exactly-one-per-window ⇒ at-most-one ⇒ no two L-points
share a 5-window ⇒ L contains and threatens **no** 5-in-a-row. It is an independent set of the
5-in-a-row hypergraph that simultaneously covers every winning line. A frozen wall that stops
both colors.

**Uniqueness.** A linear blocker ax+by≡0 (mod 5) is a perfect simultaneous blocker iff
a, b, a+b, a−b are all ≢0 (mod 5). Counting and quotienting the redundant scalars leaves
**exactly two** lattices — {2x+y≡0} and {x+2y≡0} — which are **mirror images under x↔y**. So
**up to the board's D4 symmetry, the claw is THE unique optimal static blocker.** Jason didn't
find *a* defensive trick; he found *the* minimal-density crystal the game allows. k=5 is the
smallest k for which a period-k modulus makes all four direction-steps invertible.

## 3. The crucial subtlety — a covering SET is not a drawing STRATEGY
This is exactly Jason's "no good player lets you keep it long," promoted to a theorem.
- The perfect-blocker result is a statement about a **completed, frozen coloring** of the
  plane: *if* all of L were already on the board, no 5 can pass. Mutual dead position.
- A **playable defense** is far harder: you place one stone per turn, reactively, while the
  attacker (with the move) can build a **double-four / four-three in a single move** — two
  threats, one reply, you lose by a tempo. L tells the defender *where* the safe cells are but
  not *which to grab first*, and it carries no overload guarantee. It is a **covering set**, not
  a **pairing strategy**.
- **Rigorous status of k-in-a-row defense** (positional game theory): pairing/blocking
  *strategies* exist for **k ≥ 8** (Brouwer/"Zetters" tiling; Hales–Jewett for k≥9); **k = 6,7
  are OPEN**; **k = 5 is a first-player WIN** — Allis (1994) proved freestyle 5-in-a-row is a
  Black win on 15×15 (threat-space search + proof-number search), and the infinite board
  follows a fortiori. So for k=5 **no drawing strategy exists at all.** The claw is the
  optimal-but-doomed k=5 ghost of the pairing strategies that only begin to exist at k≥8.
- **Therefore, precisely:** "no one can win or lose through it" is **true** of the static
  lattice (frozen wall), **false** as a realizable defense — tempo breaks it. Used dynamically,
  the claw is a **tempo-and-shaping tool**: hold it to deny options and dictate terrain, then
  convert off it. Exactly how Jason played it.

## 4. Why Rapfi is blind to it — and the representational lesson
**Rapfi is line-organized** (source-verified, `github.com/dhbloo/rapfi`, evaluator `mix9svq` +
the classical pattern engine). Substrate of both engines = **4 per-cell line keys** (H, V, two
diagonals). There is **no input feature keyed on a non-collinear stone relationship.**
- **Classical pattern engine:** radius-4 line window → threat classes (F3/B4/F5…). Claw spacing
  is **5 > 4** ⇒ two claw stones never co-occur in a window ⇒ **aliased to N lone `DEAD`
  stones. Perfectly invisible.**
- **mix9svq NNUE** (value/policy): per-cell sum of 4 radius-5 line embeddings (raw 11-trit shape
  codes), then a **3×3 depthwise conv (radius 1)** on *half* its 64 channels + coarse 3×3/
  quadrant value pooling. Radius-1 conv **cannot bridge spacing-5**; the claw's neighbors land
  exactly at the ±5 line-window edge (a degenerate marginal co-encoding) plus a coarse density
  signal. **Near-invisible — never the knight-offset coordination as a structure.** (Notably,
  32/64 value channels are *never* spatially convolved — half of Rapfi's positional judgment is
  pure line features.)
- **Verdict:** the claw is a **genuine blind spot** of the line-organized design — perfectly so
  for the classical engine, near-so for mix9svq.

**The representational lesson (feeds #10's defense axis).** The claw's invariant is
**2x+y mod 5 — a global periodic congruence field**, per-cell, invariant only under the
period-5 sublattice. This is the orthogonal complement of every line/shape detector (which fire
on local stone geometry). Critically, **a small conv kernel cannot express "position mod 5"
without absolute coordinates or a period-k positional encoding** — so the claw is invisible not
just to *line* features but to *vanilla local convolution* too. Implication: the **defense axis**
of a threat-chemistry representation likely needs **explicit periodic / modular positional
channels** (mod-k coordinate features), not only threat detectors. Offense lives on lines;
defense can live on *fields*.

## 5. Status, uses, and credits
- **Concrete #3 (judo) target.** The claw is an existence proof of "a region where Rapfi's map
  is structurally wrong," with a source citation. Open question worth a cell: can the net be
  trained to *herd* games onto claw-terrain where Rapfi mis-evaluates? (Respect §3 — the claw
  is a shaping/tempo prior, **never** a static policy; tempo, not coverage, decides k=5.)
- **Feeds #10.** Use as a curriculum/representation probe: does adding mod-k periodic positional
  channels (and/or a "can-the-opponent-ever-make-5-through-here" potential field) move the Rapfi
  ms-ladder crossing vs a raw-stone / line-only baseline?
- **Boundary, restated so no one over-claims:** perfect blocking *set* ≠ drawing *strategy*; for
  k=5 the latter provably doesn't exist. The claw's value is representational and dynamic, not a
  fortress.

**Credits.** Pattern: **Jason Yandell, 1990s.** Math verification + CGT grounding: subagent
"Defensive-lattice CGT grounding" (mod-5 proof by hand; framework = positional game theory /
pairing strategies; **József Beck, *Combinatorial Games: Tic-Tac-Toe Theory*, Cambridge 2008**;
Hales–Jewett 1963; Allis 1994; Györffy–Makay–Pluhár, Szeged — periodic pairing tilings of Z²).
Rapfi internals: subagent "Rapfi representation deep-dive" (`dhbloo/rapfi`, `eval/mix9svqnnue.cpp`,
`game/pattern.h`, `core/types.h:71` — line substrate; radius-4 classical / radius-5 + 3×3-conv
mix9svq).

See also: [idea-pile.md](idea-pile.md) #10 (threat-chemistry representation), #3 (judo),
#9 (GPU batch-VCF — the "no-five-possible-here" field is computable with that machinery).
