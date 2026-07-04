# White-side (second-player) defense — the investigation that became swap2

> **Status: HISTORICAL / SUPERSEDED-BY [swap2-opening-protocol.md](swap2-opening-protocol.md) (concluded 2026-06-20).**
> This page is the compressed synthesis of a full day's white-side-defense
> investigation (2026-06-15 → 2026-06-20). The full verbatim chronicle — every dated
> UPDATE banner, table, probe result, and file:line code audit — is preserved at
> [_archive/topics/white-side-defense-plan-full.md](../_archive/topics/white-side-defense-plan-full.md).
> The canonical statement of the theorem this investigation discovered is
> [alphazero-lessons-15x15-gomoku.md](alphazero-lessons-15x15-gomoku.md) §15.

## The conclusion (2026-06-20 STOP banner)

**The 15×15 freestyle white "defense weakness" is NOT a net flaw — it is the
first-player-win theorem.** From an empty/random opening, the second player (white) is
in a (near-)solved-lost role, so **no policy/value teacher can make it win — there is no
error to fix.** The clincher: Rapfi(1s) vs Rapfi(1s) from 4-stone openings → white 1-9
(~10%); even the #1 engine playing *itself* is crushed as the second player. The fix is
not a better defender — it is to **delete the forced-lost role: swap2** (Gomocup's
balancing protocol, where a player is never *forced* onto the lost side). That also makes
the Rapfi yardstick honest (Rapfi is a swap2 engine). → build **[#72 swap2](swap2-opening-protocol.md)**.

## Durable takeaways (the lessons kept)

- **White defense is a TRAINING gap, search-invariant — not an eval flag.** The weakness
  survives every eval-side lever: FPU-reduction (`c=0.45`) and 4× more search (sims
  200→800) both leave white at **0-6 / 100% loss vs a real strong attacker (zetor17)**.
  A weakness invariant to every search/eval knob lives in the **weights**, not the search.
  (The 9×9 "FPU alone fixes white-loss" claim **does not transfer** to 15×15 real engines —
  it only closed a small residual tail vs a *weak* depth-4 searcher.)
- **The color dissociation names the cause.** Same net, same search, same opponent —
  only the color flips: champion is **perfect as black** (attacking, the forced-win side
  it learned to convert) and **helpless as white** (0-12 vs Rapfi-NNUE). The entire
  strength shortfall vs the #1 engine is the white-side defense gap, confirmed across 5+
  independent measurements.
- **Value-only defense teaching poisons the value head (#36/#42, FAILED).** Relabeling
  proven-lost white positions `z=-1` with no fire-rate bound saturates the value head
  (`vl 0.16→0.06`, the canonical value-poisoning death-tell) → it contradicts the untouched
  attacking policy → shared trunk corrupts → policy degrades. Value-only teaching is
  *structurally* wrong for "never lose as white": it teaches "you were already lost," never
  the **draw/loss boundary** where the white job actually lives.
- **The #43 saving-move-on-policy lever is SOUND but was killed on buffer dilution.**
  Stamping the unique VCF refutation on the *policy* head (value left at natural outcome)
  did NOT poison the value head (`pl` bounded ~1.19-1.22, `vl` clean ~0.13-0.14) — the
  mechanism worked. But the signal was **un-readable**: fresh stamped games accumulated at
  only ~0.16-0.3%/hr of a 1.5M warm buffer, so the lesson drowned in the attacker-biased
  mass. Root cause (profiled): the per-ply VCF solve made gen crawl (~7 s/game, ~32-78×
  slowdown), so even a tiny fresh buffer couldn't be kept dense. Fix in flight was
  **sparse-bite** (`--defense-detect-frac 0.1` — sample the expensive solve on 10% of
  danger plies; ~10× cheaper, ~1000× denser in a 150k buffer).
- **The hole is RECIPE-DEEP.** A from-scratch 0.44M net (no warm-start, no teacher,
  different value head) reproduces the champion's white sweep to the game (0-20 white) —
  exonerating warm-start, capacity, and the WDL head. The same recipe defends nearly
  perfectly on 9×9 (white-loss ≤5%), so the hole **scales with board size** — fast-attack
  collapse wearing a white-defense mask.

## What was tried (the map — one line each)

| arm | what | verdict |
|---|---|---|
| **Step A** — I0 FPU + H3 search budget (eval-only) | `fpu_reduction_c ∈ {0,0.45}`, sims ∈ {200,800} vs zetor17 | **FALSIFIED** — 0% white at every setting; defense is a training gap |
| **#45** — white-defense probe suite | 80 white-to-move-threat fixture + `white_loss_rate` + Wilson CI + gate primitive | BUILT; v1 weak-attacker → champion at its floor (3.75%), no headroom → #49 strong-attacker variant needed |
| **Strong-attacker read** — Rapfi-NNUE | champion 5W-19L, **white 0-12 = 0%**, black 42% | the whole shortfall is white; TC-tier calibration shows a cliff-then-white-plateau |
| **#36/#42** — value-only `--defense-teacher` (`z=-1`) | relabel proven-lost white positions | **FAILED** — poisons value head (`vl→0.06`); structurally wrong |
| **#43 (I2)** — stamp the SAVING move on policy | `vcf.vcf_refutations` + soft policy target, value untouched | mechanism SOUND, killed on **buffer dilution** (signal un-readable); sparse-bite + fresh-buffer is the unlock |
| **Finding: recipe-deep** | from-scratch `wdl@0` reproduces 0-20 white | warm-start / capacity / value-head all exonerated → fix the DATA |

**Root-cause map:** H1 (teaching gap — a lost white game is labeled only `z=-1`, never *which
move* saved it) **STANDS**; H2 (value-target asymmetry — white wins→0 so the value head
can't distinguish drawable from lost) **STANDS**; H3 (search too shallow) **RULED OUT**;
H4/H5 (opening/covariate-shift brittleness) plausible framing. Intervention ranking was
I0 (falsified) → I1/I2 (defense teacher, value then policy) → I3 (loss-side upweighting)
→ I4 (balanced openings) → I5 (eval compensators). All detail — the file:line ground-truth
audit of the training loop, the H1-H5 evidence, the I0-I5 leverage÷cost ranking, the
eval-cadence-fit plan — is in the [full archive](../_archive/topics/white-side-defense-plan-full.md).

## Why this matters / where it went

The investigation is the reason swap2 exists: white was playing a **-EV seat**, a rigged
game no training can win. That reframed white weakness from a *training* gap to a *game
imbalance* → the fix is upstream (rebalance the game so self-play generates winnable white
positions), which is exactly [swap2](swap2-opening-protocol.md)'s bootstrap mechanism. The
`--defense-detect-frac` / `--defense-teacher-policy` levers built here are sound, tested,
and default-off-byte-identical — kept as evidence, not a path forward.
