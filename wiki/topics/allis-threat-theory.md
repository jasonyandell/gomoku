# Allis's threat theory — the gomoku threat formalism (reference)

Tight, citation-backed reference for Victor Allis's formal theory of go-moku threats: the
taxonomy, the **gain / cost / rest** square formalism, the **dependency vs conflict** relations
between threats, the winning-combination condition, and the **VCF (OR-only) vs VCT (AND/OR)**
distinction. This is the formal object our "threat-chemistry" work operates on.

**Application** (how we use it): [molecule-discovery-toolkit.md](molecule-discovery-toolkit.md)
(FANMOD on the dependency graph; the forcing/residual split). **Thesis it serves:**
[idea-pile.md](idea-pile.md) #10 (threats as molecules), [the-claw.md](the-claw.md) (the canonical
non-line molecule). **Code:** `gomoku/vcf.py` (`solve_vcf`, `solve_vct`).

**Sources.** **[AHH93]** L.V. Allis, H.J. van den Herik, M.P.H. Huntjens, *"Go-Moku and
Threat-Space Search"* (AAAI Fall Symp. FS-93-02, 1993) — definitions in §3.1 and §4; full PDF:
<https://www.mimuw.edu.pl/~awojna/SID/referaty/Go-Moku.pdf>. **Allis 1994 PhD thesis**,
*Searching for Solutions in Games and Artificial Intelligence* (Univ. Limburg / Maastricht), ch.
on Go-Moku — adds proof-number search as the completeness backstop;
<https://cris.maastrichtuniversity.nl/en/publications/searching-for-solutions-in-games-and-artificial-intelligence/>.
**Eppstein CGT / Go-Moku page** (Billings' [AHH93] summary):
<https://ics.uci.edu/~eppstein/cgt/gomoku.html>.

> **Terminology note.** Allis's own vocabulary is *five, four, straight four, three, broken three*.
> The names *open four / simple four / open three* are the later engine community's; the abbreviations
> **VCF / VCT do not appear in Allis** — they are post-Allis community terms whose objects Allis
> describes exactly (a winning threat sequence of fours only = VCF; of threes-and-fours = VCT).

---

## 1. Threat taxonomy ([AHH93] §3.1)

Allis's verbatim definitions: the **four** is a line of five squares of which the attacker occupies
any four with the fifth empty; the **straight four** is a line of six with the four centre squares
occupied and both outer squares empty; the **three** is either a line of seven with the three centre
squares occupied (the other four empty) or a line of six with three *consecutive* of the four centre
squares occupied (the other three empty); the **broken three** is a line of six with three
*non-consecutive* of the four centre squares occupied.

| Allis term | Engine name | Geometry | Empty cells | Forcing | # cost (defensive) squares |
|---|---|---|---|---|---|
| **five** | five / win | 5-in-a-row | 0 | game over | — (terminal) |
| **straight four** | open four | 6-line, 4 centre filled, both ends empty | 2 | already won — two winning squares, defender too late | unblockable (a double threat in itself) |
| **four** | simple four | 5-line, any 4 filled, 1 gap | 1 | wins next move; must be blocked | **1** (single forced block) |
| **three** (both-side extendable) | open three | 7-line, 3 centre filled | 4 | threatens a straight four next move | **2** |
| **three** (one-side blocked) | blocked three | 6-line, 3 *consecutive* filled | 3 | threatens a straight four | **3** |
| **broken three** | gap / broken three | 6-line, 3 *non-consecutive* filled | 3 | threatens a straight four | **3** |

**Strength asymmetry Allis exploits.** A **four** has exactly **one** refutation (fully forced); a
**three / broken three** has **two or three** refutations; a **straight four** is already a guaranteed
win (two winning squares — the defender cannot block both). A three's threat has depth two (it
threatens a *straight* four next move) yet "must be countered immediately" ([AHH93] §3.1). **To win
against any defence a player must create a double threat — either a straight four, or two separate
threats.**

## 2. The gain / cost / rest square formalism ([AHH93] §4, verbatim)

Allis describes each threat by three square sets:

> 1. The **gain square** of a threat is the square played by the attacker.
> 2. The **cost squares** of a threat are the squares played by the defender, in response to the threat.
> 3. The **rest squares** of a threat are the squares containing a threat possibility; the gain square excepted.

So for a single threat: the **gain square** is the one new attacker stone that *creates* the threat;
the **rest squares** are the attacker's *already-placed* stones forming the rest of the pattern (the
line minus the gain square); the **cost squares** are the empty square(s) the defender must occupy to
refute it (1 for a four, 2–3 for a three).

**Worked example ([AHH93] §4):** playing **e15** creates a **four** with **gain square e15**, **cost
square d15**, and **rest squares {a15, b15, c15}**. After e15 and d15, playing **i11** creates a four
with rest squares {e15, f14, g13}. Note that a rest square (e15) can itself be the gain square of an
*earlier* threat — this is the hook for dependency (§3).

## 3. Dependency, conflict, and threat-space search ([AHH93] §4)

**The threat-space-search reduction.** Instead of branching over the defender's choices, the attacker
"allow[s] the opponent to play **all possible countermoves at the same time**" — every cost square of
each threat is occupied at once. If a winning threat sequence still exists, the defender's choices were
inessential. This linearises the attacker's search into a single line. It is **sound but not complete**
(some real wins are missed); the thesis closes the gap with **proof-number search** plus, at the root,
the **null-move heuristic** to tame the ~200 branching factor (a move counts only if, after a null move
by the defender, it is followed by a winning threat sequence — i.e. it is a *hidden threat*).

**Dependency (verbatim).**
> 4. Threat **A is dependent on threat B**, if a rest square of A is the gain square of B.
> 5. The **dependency tree** of a threat A is the tree with root A and consisting of dependent nodes
>    only, viz. the children of each node J are the threats dependent on J.

Worked: the four with gain square **i11** is dependent on the four with gain square **e15**, because
e15 is a rest square of the i11 threat. e15's own rest squares are already on the board, so it depends
on nothing — it is a leaf.

**Conflict (verbatim).**
> 6. Two dependency trees P and Q are **in conflict** if within P a threat A exists and within Q a
>    threat B, such that (1) the gain square of A is a cost square in B, or (2) vice versa, or (3) a
>    cost square in A is also a cost square in B.

Worked: a four (gain e15, cost d15) and a four (gain d15, cost e15) are in conflict — they cannot both
appear in one winning sequence.

**Two search principles ([AHH93] §4):** (1) a threat **independent** of B is not allowed in the search
tree of B; (2) the threat-space tree contains only *attacker* threats — after a candidate winning
sequence is found, it is separately checked against counter-attack.

> **Empirically confirmed on our GPU-mined stencils (2026-06-27, #88).** Principle (2) *is* the
> certificate property we now measure: a VCT shape that wins **in isolation** transfers to any board
> it fits **by the same forcing line** — 660/660 mined attacker VCTs win from their carrier stones
> alone, and **0/2913 placements with non-attacking defender stones refute the win**; the *only*
> refuter is a defender **counter-attack** (counter-tempo) — exactly the "separately checked against
> counter-attack" caveat. So the forcing sequence is context-independent (compose it once, reuse it
> everywhere); only the counter-attack check is board-specific. See
> [shape-library-engine.md](shape-library-engine.md) §3 *Empirical certificate property* and
> `scripts/threat_shapes/certificate_falsification.py`.

## 4. Winning threat sequence / winning combination ([AHH93] §3.1, §4)

A **winning threat sequence** is a sequence of threats each of which forces the defender's reply,
ending in a **double threat** (a straight four, or two separate threats the defender cannot both
answer). When the winning continuation depends on *which* refutation the defender picks (true for
threes, which have 2–3 cost squares), Allis upgrades from a **winning threat sequence** to a **winning
threat tree**.

**Independent-threat combination (the fork condition).** When no single dependency chain wins, the gain
squares of **two or three independent threats** can be combined to create a new threat — provided their
gain squares **lie on a single line and close together** *and* their **dependency trees are not in
conflict**. (Allis's diagram-4 example: combining {h10, i9} yields a straight four at f12; combinations
whose dependency trees conflict are rejected.)

## 5. VCF vs VCT — why one is OR-only and the other AND/OR

| | **VCF** (Victory by Continuous Fours) | **VCT** (Victory by Continuous Threats) |
|---|---|---|
| Allis object | winning threat sequence of **fours only** ([AHH93] §3.1, diag. 2a) | winning sequence/tree of **threes interrupted by fours** (diag. 2b) |
| Defender reply | every four has **one** cost square ⇒ reply is **unique/forced** | a three has **2–3** cost squares ⇒ defender **genuinely branches**, and may insert his own fours to delay (§3.2) |
| Search shape | **OR-only single forced line** (no defender branching) | real **AND/OR proof search** — OR over attacker threats, AND over defender refutations |
| Soundness | sound + complete on the fours-fragment; cheap | the all-cost-squares reduction is sound, **not complete**; proof-number search restores completeness |

This is exactly the shape of `gomoku/vcf.py`: `solve_vcf` recurses on the single forced block (OR-only);
`solve_vct` opens a genuine defender AND-node (`_vct_defend`) with a tempo guard (a defender counter-four
breaks the line). Both are deliberately conservative — a cap-hit "no win" is *unproven*, never trusted as
safe — because a false forced-win poisons training.

## 6. The headline result ([AHH93] §7; thesis ch. on Go-Moku)

By exhaustive solution-tree construction (the program **Victoria**, threat-space search + proof-number
search), Allis proved that **the first player (Black) wins 15×15 free-style Go-Moku** — and the
no-overline variant too. Free-style: ~1.1M CPU-seconds, 5.3M positions investigated, **solution tree
138,790 nodes, depth 35 ply (18 Black moves) against optimal defence**. (No-overline variant: 153,284
nodes, also 35 ply.) The infinite-board and 19×19 cases follow a fortiori. *Victoria* won gold at the
1992 4th Computer Olympiad.

> **Consequence for us.** k = 5 is a first-player win, so for k = 5 **no drawing/pairing strategy
> exists** — the claw is a perfect blocking *set*, never a realisable drawing *strategy*
> ([the-claw.md](the-claw.md) §3). Tempo, not coverage, decides k = 5.

---

**See also:** [molecule-discovery-toolkit.md](molecule-discovery-toolkit.md) (the bio-algorithm
application + the FANMOD-on-the-dependency-graph first experiment), [the-claw.md](the-claw.md),
[idea-pile.md](idea-pile.md) #10.
