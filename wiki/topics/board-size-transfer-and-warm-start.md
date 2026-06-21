# Board-size transfer & warm-start (9×9 → 15×15) — the proven curriculum

**Status (2026-06-20):** PROVEN, TESTED, READY. This page is the consolidated how-to that
was missing — the mechanism existed (`scripts/warmstart_15x15.py`, commit `221b314`, tested
in `tests/test_warmstart.py`) and is how the 15×15 nets were actually born, but it lived only
in the script docstring + scattered campaign logs. Don't lose it again.

## What it is

`scripts/warmstart_15x15.py` does a **cross-board partial load**: build a FRESH target-board
net, copy every tensor whose name AND shape match from a source (smaller-board) checkpoint,
leave the board-bound FCs at fresh init. Because the net uses **`global_pool`** (board-size-
agnostic conv trunk), almost everything transfers:

- **9×9 → 15×15, size=large: 98.9% of params transfer** (153/156 tensors, conv tower +
  global-pool blocks + value trunk **byte-identical**). Only **3 board-bound FCs re-init**:
  `policy_fc.weight`, `policy_fc.bias` (81→225 outputs), `value_fc1.weight` (spatial flatten).
- The script **asserts arch-match** and fails LOUD on divergence (different `n_filters` etc.) —
  it never silently half-transfers.
- The swap2 **choice head transfers too** (it's `Linear(value_hidden, N_CHOICES)`, board-size-
  independent); `load_checkpoint` also tolerates missing choice keys (fresh-init fallback).

## Why it matters — warm-start is the remedy for the 15×15 cold-start collapse

**Cold-starting a fresh net at 15×15 walks straight into fast-attack collapse** (plies crash,
shallow rush-fest) before defense develops. This is documented and we re-confirmed it live:

- 2026-06-13 `G15-seed` cold run → fast-attack collapse (plies 68→11), swapped to a
  9×9-warm-started seed → **plies ~85 from epoch 0 (defended play), collapse skipped.**
  Durable lesson then: "cold-start fast-attack collapse is real at 15×15; warm-start is the
  remedy." (A later WDL from-scratch run *recovered* from the collapse on its own, so warm-
  start isn't strictly mandatory — but it SKIPS the wasted collapse-and-recover cycle.)
- 2026-06-20 era-2 (`G15-swap2-e2`, fresh 15×15 swap2) reproduced it exactly: **plies 70→12
  (collapse) → recovered to ~25** over ~3 hours. That whole V-shape is the cold-start tax a
  warm-start avoids.

**The deeper trap:** era-1 warm-started the *pre-swap2 defensive champion* → imported the
**lose-slowly basin** ([[swap2-opening-protocol]] §9). So *what* you warm-start from matters:
warm-start an **aggressive, not-doomed** net and you import good priors; warm-start a
defensive one and you import the basin.

## The curriculum recipe (era-2 / Path A, the 2026-06-20 overnight)

Three wins from one move — train cheap-and-fast where white isn't doomed, then carry it up:

1. **Bootstrap on 9×9** (`G9-swap2-e2`): fresh swap2, aggression (`value-discount 0.95`),
   v2a OFF. 9×9 is the project's NATIVE board — native state-ops + MCTS exist, **~28 s/epoch
   vs 15×15's ~91 s (3.2×)**, games short, white trainable. Run to ~500 epochs (~3.7 h).
2. **Transfer**: `warmstart_15x15.py` → a 15×15 seed (98.9% trunk). Skips the cold-start
   collapse AND imports the *aggressive* 9×9 priors (not era-1's defensive ones).
3. **Continue on 15×15** (`G15-swap2-e3`): swap2, aggression 0.95, v2a OFF, `--resume <seed>`.

Orchestrated by `/Users/jason/data/swap2/babysit/overnight_autochain.sh` (detached, crash-
guarded, `touch STOP_overnight` to stop). 9×9 cell `G9-swap2-e2` (commit `cd93233`); 15×15
phase-2 cell `G15-swap2-e3` (commit `a16737e`); both = the e2 recipe with v2a removed.

### Live progress — phase 1 (9×9, run `lywhy1ba`, 2026-06-20)

The speed thesis is paying off: **~40 epochs reached before the first human check-in** vs
**3 h to hit e100** on the prior 15×15 cadence. Phase-1 reads at **e47** (model card:
[[gomoku-9x9-swap2-era2]]):

- **white-not-doomed is showing** — white takes **~45% of decisive self-play games**
  (e17 50.0% · e31 48.6% · e39 50.0% · e47 47.7%). Black keeps a slight first-mover edge
  (expected); that is **not** the warning.
- **plies-collapse in progress** — `plies_mean` 36 → 16.7 (p50 18→15). The normal
  offense-before-defense dip; healthy *because* white is still winning ~45% through it. We
  expect the V-shape: plies recover as white learns defense (same dynamic as the 15×15
  cold-start, just on a net that isn't doomed).
- **the warning to watch** (Jason's): white% trending *down toward 0* = relapse into the
  lose-slowly basin. Not seen. Distinguish from fast-attack collapse (plies falling **with**
  concave buffer-fill AND white% collapsing) — that's the bad version.

## The multi-rung LADDER (9→11→13→15) — the 2026-06-20 overnight, era-2 revised

9×9 is too cramped: by ~e100 the swap2 net's defense **saturates the board into
draws** (e102 last-3 epochs draw-dominant: draws 56/75/56 vs white ~14, black
~19–31). A draw-or-black regime teaches white bad habits and the win/loss gradient
thins. Jumping straight to 15×15 fixes the draws but is **2.7× slower** — and we
need epochs. So: climb a **board-size ladder**, warm-starting up a rung each time
the current board saturates, milking cheap native epochs at every step.

**Native at EVERY rung is mandatory — the make-or-break.** Measured (M5 Max, MCTS
sims/s): native-11 **68.3k** > native-15 **60.7k** (faster, as cell-count predicts;
native-9 64.8k). But **pure-Python is ~40k sims/s *regardless of board size***
(Python overhead dominates), so a fallback-only 11/13 would be ~1.5× SLOWER than
native-15 — the ladder on the fallback is worse than the destination it's trying to
cheapen. We therefore compiled native exts for 11 and 13 (commit adds
`_state_ops_native11/13.c` + `_mcts_native11/13.c` shims, `setup.py` blocks,
dispatch arms, `NATIVE_BOARD_SIZES=(9,11,13,15)`). The shared `.c` is fully
`#define BOARD_SIZE`-parametrized, so each size is a 3-line shim — adding a rung is a
seconds-long recompile, not real work. **Never run a rung on the pure-Python
fallback** (`GOMOKU_DISABLE_NATIVE_*`) — it violates the Δelo/hour budget.

**Graduation rule (Jason's): step up when `max(draw, white, black) == draw`** — i.e.
draws are the strict plurality of self-play outcomes; white's defense has saturated
this board, so move up to reclaim room. Denoised: draws strict-max for 3 consecutive
epochs, past a `MIN_EPOCH` guard (so we don't graduate during the post-warmstart
recovery-V, which is offense-heavy/short-games, not draws), with a per-rung epoch
CAP as anti-hang backstop. The check is a **pure read of the rung's wandb history**
(`babysit/ladder_grad.py` looks up the `wandb_run_id` embedded in `latest.pt`) — the
trainer is untouched. Rung 15 is **terminal**: it basically never draws (board too
big to fill), so it trains until the STOP sentinel.

**Cells** `G-ladder-11/13/15` (run_sweep.py) = the `G9-swap2-e2` recipe verbatim,
only the run-dir differs (board size is the env var). **Orchestrator**
`babysit/ladder_autochain.sh` (detached; `touch babysit/STOP_ladder` to stop;
20-min slices, graduation checked between them). Seeds: rung-N warm-starts the rung-
(N−1) champion (9→11 verified on our e102 champ: 98.9%, only the 3 board-bound FCs
re-init). The open empirical question — does laddered learning beat a direct 9→15
warm-start? — is a derby lane (Δelo/Δt vs the existing direct-15 reference); the
ladder's a priori case is purely epoch-efficiency, not correctness.

## How to run the transfer (entrypoint)

```bash
GOMOKU_DEVICE=cpu python scripts/warmstart_15x15.py \
  --from <9x9_checkpoint.pt> --out <15x15_seed.pt> --size large --board-size 15
# optional: --target-value-head wdl  (scalar 9x9 -> WDL 15x15; warm tower, fresh WDL head)
# then continue training:
GOMOKU_BOARD_SIZE=15 python -m gomoku.train --resume <15x15_seed.pt> ...
```

Verify: `GOMOKU_BOARD_SIZE=15 pytest tests/test_warmstart.py -q` (transfer accounting,
byte-identical trunk, arch-mismatch-fails-loud, scalar→WDL).

## Gotchas

- **`GOMOKU_BOARD_SIZE` must be set BEFORE importing `gomoku`** — board size is resolved once,
  process-level (`gomoku/board_config.py`), and written back to the env so workers inherit it.
  The warmstart script handles this internally; your own scripts must set it first.
- **Architecture must match** between source and target (same `n_filters`/`n_blocks`/
  `global_pool`/`stem_padding`/`activation`) — else the trunk won't shape-match. Script asserts.
- **The seed is buffer-less + optimizer-less** (`epoch=0`, no wandb id) → the first run builds
  a fresh buffer and a clean timeline. That's intended.
- **Board size is NOT a Cell field** in `run_sweep.py` — the `...board9`/`...board15` run-dir
  name is just an artifact-collision guard. You select board size with the env var at launch.
- Native ext is **compiled per board size (only 9 and 15)** by `uv pip install -e ".[dev]"`;
  both `.so` sets exist in a properly-installed worktree. No recompile needed to switch.
