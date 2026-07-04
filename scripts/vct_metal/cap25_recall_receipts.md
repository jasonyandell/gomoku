# cap50 → cap25 VCT-veto recall study (#114 attack-list item 2)

**Date:** 2026-07-03 · **Session:** cap25-recall (worktree `gomoku-cap25-recall`)
**Script:** `scripts/vct_metal/cap25_recall_study.py` (reusable; `--board-size / --ckpt / --batches / --out`)

## Question

The sound-world gen solves, every ply, a **merged** bulk batch (`_oracle_ply_solve`):
attacker-terminus boards + defense escape-children, at node budget **cap50**
(`--vct-terminus-budget 50`). Day-1's census found **42.5% of 13×13 veto boards
grind to the cap**, so lowering to **cap25** would cut most lanes' work — but it is
a **semantics** change: any blunder whose forced-win proof needs **26–50 nodes**
stops being vetoed (a recall *leak* = a played-through blunder).

**Recall** = of the boards **proven win at cap50** (`win=True`), what fraction are
**still proven at cap25**? Gate from the handoff: **recall ≥ ~98.5% → propose the flip.**

## Method

Monkeypatched `sp._vct_terminus_solver` (+ `sp._terminus_boards` to recover the
merged batch's terminus/defense split point) to record every real solver batch a
live gen dispatched. Exact sound-world semantics: `configure_vct_terminus(budget=50)`,
`configure_oracle_veto(max_cands=0)` (full breadth), **overlap OFF**, 48 games @ 32
concurrent, 100 sims, wave 32. Then re-solved every captured board at cap50 and
cap25 (same kernel, `max_nodes` param, lanes=1 base path) and counted.

Two hard kernel invariants asserted on **every** run (both held, 0 violations
everywhere): **monotonicity** (`proven@25 ⊆ proven@50` — a bigger budget can only
find more proofs) and **leak-capped** (every leak board must be `hit_cap` at 25 — a
clean no-win at 25 is a true budget-independent proof and could never be win@50).

## Recall results

| board | net | recall | proven@50 | proven@25 | **leaks** | miss % | boards |
|---|---|---:|---:|---:|---:|---:|---:|
| **9×9**  | champion `sound-world-107b`         | **98.64%** | 12 094 | 11 930 | 164 | 1.36% | 57 247 |
| **13×13**| `swap2/G-ladder-13` (128×10 full-game) | **99.39%** | 85 884 | 85 359 | 525 | 0.61% | 226 622 |
| **13×13**| `sound-world-13-scratch` (recipe net)  | **99.93%** | 40 961 | 40 931 |  30 | 0.07% | 135 437 |

**All proven-wins are DEFENSE escape-children (blunder vetoes); attacker-terminus
wins = 0 in every run** (902 / 1693 / 925 terminus boards, zero forced-VCT-for-mover
at a self-play root). So the entire recall question is about the **defense veto** —
which is exactly the soundness-critical path (a missed defense-child veto = a
played-through blunder).

## Timing (cap50 vs cap25, lanes=1 base, best-of-3, under the GPU lock)

| board | boards | solve@50 | solve@25 | **speedup** |
|---|---:|---:|---:|---:|
| 13×13 (G-ladder batches) | 226 622 | 29.84 s | 15.10 s | **1.98×** |
| 9×9 (champion batches)   |  57 247 |  6.80 s |  3.49 s | **1.95×** |

Halving the cap ≈ **halves the solve wall** at both sizes. At 13×13 the solve is
~100% of the gen wall (day-1), so this is ~1.98× **gen** — a larger lever than the
day-2 lanes=K kernel (1.34×), and directionally composable with it.

## Poison-check guardrail (budget 25)

`GOMOKU_POISON_BUDGET=25 gen_poison_check.py <9×9 champ> overlap 1000` →
**VIOLATIONS: 0/174**. Confirms the veto plumbing is self-consistent at cap25 (gen
at 25 records zero policy mass on cells proven-blunder at 25). This is a
plumbing guardrail, **not** the leak measurement — the leak rate is the recall
table above.

## Gate verdict — recall ≥ 98.5%: **MET** (decision is Jason's / the coordinator's)

- **13×13 (where the perf win lives):** 99.39% (full-game net) / **99.93%** (the
  actual sound-world recipe net) — **comfortably clears the gate**, with a ~1.98×
  solve win. The recipe-net's 0.07% miss (30 / 40 961) is negligible.
- **9×9:** 98.64% — clears 98.5% but **marginally** (1.36% miss). *And 9×9 does not
  need cap25*: width-is-free there, the solve is not the gen bottleneck. The clean
  move is a **board-size-conditional flip** (cap25 at 13×13, keep cap50 at 9×9),
  not a global default change.

### Caveats to weigh before flipping a production default

1. **Recall is distribution-dependent.** Measured on *current* nets; as play
   sharpens during training, the 26–50-node proof band could grow and recall drift
   down. The sound-world recipe net (99.93%) is the most production-faithful point,
   but re-measure / leak-monitor at the trained distribution before a permanent flip.
2. **Every leak is a played-through blunder** at that ply. The 9×9 K-cap ablation
   resurrected the fast-attack attractor (`wiki/topics/sound-world-recipe.md`), so
   soundness is not free — hence the marginal-9×9 caution above.
3. This study did **not** run a full training slice at cap25; it measures the leak
   rate statically. A Δelo confirmation (cap25 cell vs cap50 champion) would close
   the loop on whether 0.6% missed 13×13 vetoes actually degrades strength.

## Reproduce

```bash
# capture + recall (untimed, no GPU lock needed — correctness is contention-immune)
uv run python scripts/vct_metal/cap25_recall_study.py \
    --board-size 13 --ckpt ~/data/swap2/sweep_runs/G-ladder-13-board13/checkpoints/worker_weights.pt \
    --batches 20 --out /tmp/r13
# timing (RUN UNDER /tmp/gomoku-gpu.lock)
uv run python scripts/vct_metal/cap25_recall_study.py --time --board-size 13 --out /tmp/r13
# poison guardrail at cap25
GOMOKU_POISON_BUDGET=25 uv run python scripts/gen_poison_check.py <9x9 ckpt> overlap 1000
```
