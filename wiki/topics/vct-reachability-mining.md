# Mining VCT-reachability from the Rapfi corpus — the free distance field, the off-path fan, the knife-edge, and non-VCF gold

**One-line.** Two cheap, search-free (or GPU-only) ways to mine VCT-reachability signal from
the half-million **Rapfi-vs-Rapfi** games, toward the **seek-VCT** plan ([seeker-steering-learnability.md](seeker-steering-learnability.md),
[vct-recognition-learnability.md](vct-recognition-learnability.md)). The headline finding is a
**thesis update**: the pre-onset band we assumed was the net's *forgiving, quiet* steering region
is actually a **knife-edge** — ~**80% of alternative moves lose by force** there — so the clean
"net steers the quiet region, solver finishes the sharp tactics" split has a **fuzzy boundary that
starts well before the VCT**. Plus a usable bounty: a free **distance-to-VCT field**, and a
**non-VCF VCT** harvest (the ~3.5% of forced wins that need a *three*, not just fours — the
combinational "molecules", concentrated on the **winner's** side).

Date: 2026-06-26. Hardware: M5 Max; all solving on the **Metal/GPU** megakernel
([gpu-vct-feasibility.md](gpu-vct-feasibility.md) §8) — **zero contention** with the live
`collect_rapfi` fleet (CPU-only). **Both seats are strong Rapfi-NNUE** — every "losing move" below
is a counterfactual *we inject*, never a move a player chose. Code: `scripts/threat_shapes/vct_fan.py`.

---

## 1. The free distance-to-VCT field (no re-solve)

The forward puzzle miner ([shape-library-engine.md](shape-library-engine.md) §3) already wrote a
per-ply verdict for **every ply of every game** (`~/data/puzzle_miner/`): `win&~cap` plies present,
`cap` plies present, proven-no-VCT plies *absent* (within `manifest.txt` shards). So the full
per-game VCT-window structure is already on disk. Characterized over 400 shards / 40,000 games:

- **Terminal VCT is near-universal:** **99.0%** of decisive games end with the winner holding a
  proven VCT at their last move — "there's always one at the end" is true.
- **Multi-window (a side loses then re-finds a VCT) is real but modest:** 86.7% of sides-with-a-VCT
  have one window, 10.6% two, 2.7% three+ → **11.6% of games** carry a multi-path side. (VCT
  non-monotonicity, as `mine_first_vct.py` documents — the *holder* failing to execute, never the
  opponent "stopping" it.)
- **The offense gradient is well-populated** (of positions with a forward same-parity VCT): 34.9%
  are *at* a VCT (the solver's "you've won" set), then **~4–5% each at 1,2,3,4,5 my-moves out**, with
  20.9% at 6–10 and 22.1% at 11+. A usable slope in the near band.
- **Coverage is the catch — ~49%.** Only ~half of (mover) positions ever reach a same-parity VCT
  *in their own realized game* (the loser's positions mostly never do; plus draws); defense field
  (distance to the *opponent's* next VCT) covers 54.7%; **13.9%** of positions are `cap` holes (mask).

**The proposed target (TRAINED 2026-06-27 → [phi-distance-field-learnability.md](phi-distance-field-learnability.md)).**
A discounted potential
**`Φ(s) = γ^(my-moves-to-nearest-future-VCT)`**, `Φ=0` where none (the floor — "no forcing win
reachable here, this game"), `cap` masked, with a **second channel** for the defense field.
*(Result: the field is learnable + generalizes — held-out CNN offense ρ=0.72/reach-AUROC=0.91,
defense ρ=0.76/0.92; NOT count-dominated; CNN beats attention a third time even on this global target.)* Modulo
the runtime solver, *having a VCT **is** the value* (VCT ⟹ win), so Φ is a value function for the
subgoal "force a win" — and because closeness-to-a-fork is a whole-board fact, the target is
**global by construction** (the fair arena to re-audition attention vs a locality-biased CNN).
**Honesty:** distance-*along-Rapfi's-realized-play* is an **upper bound** on the true distance (the
game found *a* path; a shorter one it didn't take may exist), and the Φ=0 floor is a **lower bound**
(missed wins) — the free target brackets the truth and is noisy. Tightening it is what §2 is for.

## 2. The off-path fan, the framing, and the knife-edge

**Method (`vct_fan.py`).** Ride each game; at known-non-VCT pre-onset nodes, fan every alternative
move the side did *not* play and solve VCT on each resulting board. No recursion.

**The framing — load-bearing, and code-verified.** A VCT belongs to the **side to move**. After
side S plays alternative `m`, it is the **opponent's** turn, so a VCT on the fanned board is the
**opponent's** forced win — i.e. `m` is a forced-*losing* move for S. **The fan is a defense/blunder
detector + a VCT-board miner, never an offense ("S missed a win") detector** (S would need its own
next turn; that needs the expensive ∀-reply 2-ply search). Integrity check (GPU-only): **0.000%** of
fanned nodes are themselves VCT (n≈3k) and **0** parity violations — so we only ever fan true
non-VCT positions; "one side already had a VCT and we found another" never happens. *(A one-time
CPU-oracle spot-check also confirmed 120/120 wins / 80/80 clears before we retired the CPU solver as
a cross-check — see [why we don't cross-check with the CPU solver](#); the kernel's standing
0 FP/0 FN validation is the basis going forward.)*

**Worked example (the whole story in one board).** Ply 32, black to move, correctly non-VCT. White
already has a **gap-four** on row 8 (stones at cols 4,6,7,8; hole at col 5) — one move from five at
(8,5). Black isn't attacking; it's **defending a four**. Black's *real* move = **(8,5)** blocks it
(white then has no VCT) — and that same square sets up black's *own* VCT two plies later (the
**defensive point is the offensive point**). Black's *alternative* (8,5)→ e.g. a corner pass → white
plays (8,5) = five. So the "VCT after black's bad move" is just white finishing an existing four.

**The knife-edge (the thesis update).** Fraction of a side's alternatives that lose by force, by
who's to move × distance-to-onset (n≈300k):

| dist | VCT-holder to move | opponent to move |
|---|---|---|
| 1 | — | **98.3%** |
| 2 | **89.4%** | — |
| 3 | — | 92.7% |
| 4 | 52.7% | — |
| 5 | — | 84.6% |
| 6 | 45.7% | — |

Even **6 plies before the VCT, ~half of all moves lose by force**; one ply before, ~99%. The
*winner* at onset−2 loses 89% of the time if it deviates — **both players walk a tightrope**, and the
winner is the one who navigates the narrow safe set (~11% of moves). **Sharpness ramps up well
before the onset**, so the pre-onset region is *not* approximation-tolerant. The net's genuinely
forgiving domain is **further back than onset−6**; the solver/lookahead should own the whole sharp
ramp — which the "consult the oracle every ply" hybrid design already does (now with evidence for
*why*, not just hygiene).

## 3. The triviality split — 96% trivial, and the gold is the winner's combinations

Run the fast **VCF** kernel on the VCT-wins (VCF ⊆ VCT). VCF = forced win using only *fours* (the
trivial extreme is the length-1 "you didn't block my five"); a **non-VCF VCT** *needs a three* = a
real combination, the molecule-shaped tactic ([shape-library-engine.md](shape-library-engine.md)).

- Of the **406,202** fanned VCT-wins (81.1% of all fanned boards): **VCF 96.1%**, **non-VCF VCT
  3.5%** (14,380), VCF-capped 0.3%. So the eye-popping 81% is **mostly a four-blocking drill** —
  don't be fooled by the big number.
- **But the 3.5% gold has clean structure** — it splits by parity. non-VCF rate among VCT-wins:

  | dist | who'd get the VCT | non-VCF |
  |---|---|---|
  | 1 / 3 / 5 | **winner** (defender perturbed) | 1.9% / **6.0%** / **6.2%** |
  | 2 / 4 / 6 | opponent (winner perturbed) | 0.0% / 0.7% / 1.2% |

  **Almost all the non-VCF gold is the *winner's* combinational wins** — surfaced by perturbing the
  *defender*. Reason: the winner is *building* an attack (latent three-based combinations), the loser
  is only *defending* (their post-blunder "wins" are cheap fours they happened to hold). **Combinations
  belong to the side with the initiative.** (Within the winner's rows the rate also grows with
  distance, 1.9→6.0→6.2: closer to onset it's an immediate four-slam; further out the forced win more
  often *needs* a three.)

## 4. What this means + the harvest spec

- **Harvest the gold by perturbing the *defender*.** Fan the **opponent-to-move pre-onset nodes**
  (skip/heavily-downsample the winner-side fan — it's a trivial four-drill) → keep the **VCT∩¬VCF**
  boards. ~6% of those wins are gold; from 2,500 nodes we already get 14,380, so a corpus-scale
  defender-side fan (~50k nodes → a few **free**-GPU hours) yields **hundreds of thousands** of
  combinational forced-win boards.
  - **RAN 2026-06-27 — banked to `~/data/molecule_gold/`** (`harvest_molecules.py`, the corpus
    writer). 20,000 defender-side nodes / 68 shards / 25 min GPU → 4.01M fanned → **3.71M VCT
    (92.4% — the defender-side knife-edge, sharper than §3's both-sides 81%)** → **146,655 non-VCF
    gold (3.95% of VCT), 99.0% distinct, 100% move-labeled** (passive `return_move`). Boards are
    **sparse** (winner mean 6.2 stones) and the gold **grows with distance** (dist-1/3/5 =
    28.8k/51.7k/66.2k → the deeper win more genuinely *needs a three* = the purer molecule).
    Only 68/400 shards consumed at the node cap ⇒ resumable, ~60× headroom on the full set.
- **Each gold board pays twice:** it's a **non-trivial offense terminus** for the §1 distance field
  *and* a **hard defense lesson** (a natural-looking defensive move that walks into a combination) —
  aimed at the project's binding wound, the white-side **defense** gap ([white-side-defense-plan.md](white-side-defense-plan.md)).
- **No solver in the learned model.** Train the net as a pure potential/policy; bolt the batch
  solver on at runtime (every ply: take any VCT you have, dodge any the opponent has). The 2.38M
  move-labeled puzzles in 7 minutes ([vct-backward-mining.md](vct-backward-mining.md) §5) is proof
  the runtime tack-on is trivial.

## 5. What we thought vs. what we found (banked negatives)

- **Thought:** pre-onset = the net's *quiet, forgiving* steering region (the original seek-VCT
  split). **Found:** it's a knife-edge (~80% of moves lose), sharpness ramping *before* onset → the
  net/solver boundary is fuzzy and earlier than assumed. **The most valuable thing here.**
- **Predicted (both wrong):** *Claude* bet fanned VCTs would be common but caps would dominate by
  count; *Jason* bet more caps than VCTs. **Reality: 81% VCT, 5% cap** — forced losses are abundant
  and *shallow*.
- **Thought:** the 81% is a rich tactical signal. **Found:** 96% trivial four-blocks; the real signal
  is the 3.5% non-VCF tail. *A big number that's mostly trivial.*
- **Surprise nobody predicted:** the non-VCF gold isn't uniform — it's the **winner's** combinations
  (the parity split).
- **Methodology scar:** a momentary "we found a VCT where one already existed — impossible!" alarm was
  a **labeling** confusion (the "VCT-holder to move" column meant *parity bookkeeping*, not "has a VCT
  now"; all fanned nodes are pre-onset, non-VCT). Name things for what they *are*.

## 6. Artifacts

| Path | What |
|---|---|
| `scripts/threat_shapes/vct_fan.py` | the consolidated probe: framing integrity + knife-edge breakdown + VCF/non-VCF split (reproduces §2–§3) |
| `scripts/threat_shapes/harvest_molecules.py` | the §4 corpus **writer** — defender-side fan → VCT∩¬VCF → move-labeled, chunked/incremental/resumable |
| `~/data/molecule_gold/` (`gold.jsonl.gz`) | **CANONICAL** non-VCF combinational forced-win corpus — **146,655** move-labeled gold boards (first bank, 2026-06-27) + README |
| `~/data/puzzle_miner/` (`puzzles.jsonl.gz` + `manifest.txt`) | the free per-ply verdicts (§1) |
| (scratch this session) the `analyze_vct_field` distance-field characterization (§1 numbers) | not committed — reproducible from `puzzles.jsonl.gz`; promote to a real builder when we train Φ |

**Cross-links:** [seeker-steering-learnability.md](seeker-steering-learnability.md) (the BC steering
half) · [vct-recognition-learnability.md](vct-recognition-learnability.md) (recognition + the
thesis) · [gpu-vct-feasibility.md](gpu-vct-feasibility.md) §8 (the oracle / L0) ·
[vct-backward-mining.md](vct-backward-mining.md) §5 (move extraction) ·
[shape-library-engine.md](shape-library-engine.md) (non-VCF VCT = the molecules; L2 = the potential
field) · [white-side-defense-plan.md](white-side-defense-plan.md) (where the defense gold is aimed).
