# The Seek-VCT program

**The thesis:** the net *steers* (learn the approximation-tolerant steering),
the GPU oracle *finishes* (solve the approximation-intolerant forcing kill) —
**anti-correlated tractability**. A VCT is a forced win the oracle both detects
and terminally values, so the game's objective becomes "reach a VCT." This is
the current live research frontier, and the wiki's densest region.

> **← Hubs:** [index](index.md) · parent: [Experiments](experiments.md) ·
> synthesis HUB page: [vct-mining-research.md](topics/vct-mining-research.md)

## The one-line headline (2026-06-30)

**The first VCT arrives at median ply 19; play self-play TO the VCT, not to
five.** A VCT exists by ~move 20 and games run 40–60 plies, so the back half is
VCT-saturated (48.9% of all positions are VCT-wins). A **cap50 mate-seeker** is a
near-complete detector (finds a forced win in 96.1% of games, ~as early as an
infinite-patience solver) ⇒ **build the AI as a cap50 mate-seeker.**
[vct-cascade-run-2026-06-30.md](topics/vct-cascade-run-2026-06-30.md)

## The oracle (the enabling tool)

| Page | Role |
|---|---|
| **[mega-vct-solver.md](topics/mega-vct-solver.md)** | **`mega_vct_bb` — the canonical API / the contract.** On-device GPU VCT solver, ~1600× CPU, 0 FP/0 FN over 320 VCT + 360 VCF. Outputs: `move`, `support`, `carriers`, `w`, `complete`, `max_depth`→md_min. |
| [gpu-vct-feasibility.md](topics/gpu-vct-feasibility.md) | The call-cost law (flat-in-batch, latency fixed by the single hardest board ⇒ bulk-synchronous only); §8 overturns the v0 CPU-bound narrative. |

## Mining the corpus

| Page | Role |
|---|---|
| [vct-backward-mining.md](topics/vct-backward-mining.md) | Walk won games back to the "first true VCT move" = the setup, not the line-bound kill. |
| [vct-reachability-mining.md](topics/vct-reachability-mining.md) | Distance field, off-path fan, the **knife-edge** (~80% of alt moves lose by force typically, up to ~98% one ply before onset), non-VCF gold. |
| [idx2-vct-frontier-map.md](topics/idx2-vct-frontier-map.md) | Forward-expand idx-2 as an AND/OR frontier — "solve the Bruce-Lee board for black"; the danger map. |
| [vct-cascade-labeler.md](topics/vct-cascade-labeler.md) | Corpus-scale verdict ledger + at-scale deepening curve (label all 56.1M positions). |
| [vct-cascade-run-2026-06-30.md](topics/vct-cascade-run-2026-06-30.md) | The live at-scale run + the median-ply-19 / cap50 findings. |

## The shape-library engine (the AI Jason wants to build)

| Page | Role |
|---|---|
| **[shape-library-engine.md](topics/shape-library-engine.md)** | Mine guaranteed-first-VCT → reduce to a typed **stencil** via context ablation → a fast candidate index with **L0 verifying every match** (never hallucinates). The certificate property (660/660 self-contained). |

## Can a net learn it? — the learnability trilogy

| Page | Question → answer |
|---|---|
| [vct-recognition-learnability.md](topics/vct-recognition-learnability.md) | **See** a VCT? Yes, AUROC 0.92+ — but easy + count-dominated (CNN & logreg beat attention); leave recognition to the oracle. |
| [seeker-steering-learnability.md](topics/seeker-steering-learnability.md) | **Steer** toward a VCT? Yes, learnable + generalizes (BC top-1 0.386); CNN beats attention again. |
| [phi-distance-field-learnability.md](topics/phi-distance-field-learnability.md) | **Regress** the proof-frontier Φ? Yes — the **first real L2 model**; Φ is NOT count-dominated (spatial), defense reads better than offense. |

## Where it feeds back into training

The seek-VCT objective drove the sound-world recipe and its results — those live
in the [AlphaZero hub](alphazero.md): the VCT-terminus result (#100/#101), the
VCT-defense aux head (#103), and the sound-world recipe itself.

## Full page index — every page in this hub

*Complete map (12 pages); the sections above surface the headline ones.*

| Page | Note |
|---|---|
| [vct-mining-research.md](topics/vct-mining-research.md) | the synthesis hub page |
| [mega-vct-solver.md](topics/mega-vct-solver.md) | canonical solver API |
| [gpu-vct-feasibility.md](topics/gpu-vct-feasibility.md) |  |
| [vct-backward-mining.md](topics/vct-backward-mining.md) |  |
| [vct-reachability-mining.md](topics/vct-reachability-mining.md) |  |
| [idx2-vct-frontier-map.md](topics/idx2-vct-frontier-map.md) |  |
| [vct-cascade-labeler.md](topics/vct-cascade-labeler.md) |  |
| [vct-cascade-run-2026-06-30.md](topics/vct-cascade-run-2026-06-30.md) | median-ply-19 / cap50 findings |
| [shape-library-engine.md](topics/shape-library-engine.md) | the AI Jason wants to build |
| [vct-recognition-learnability.md](topics/vct-recognition-learnability.md) | trilogy: see |
| [seeker-steering-learnability.md](topics/seeker-steering-learnability.md) | trilogy: steer |
| [phi-distance-field-learnability.md](topics/phi-distance-field-learnability.md) | trilogy: regress; first L2 model |
