# Board-size transfer, warm-start & the auto-graduating ladder — the proven curriculum

**Status (updated 2026-07-02):** PROVEN, TESTED, READY. This is THE reference for two things:
(a) the **cross-board warm-start / graduation mechanics** (partial-load a smaller-board champion
into a fresh bigger-board net) and (b) the **auto-graduating 9→11→13→15 ladder** that chains
warm-starts to milk cheap native epochs at each rung. The mechanism existed
(`scripts/warmstart_15x15.py`, commit `221b314`, tested in `tests/test_warmstart.py`) and is how
the 15×15 nets were born, but it lived only in the script docstring + scattered campaign logs.
Don't lose it again.

**Related:** [[training-run-lineage]] (which run warm-started from which),
[[15x15-training-campaign]] (the destination board's campaign),
[[sound-world-recipe]] (the 9×9 recipe carried up to 13×13 in #113),
[[bruce-lee-model]] (the large "Bruce" net — the 15×15 warm-start target),
[[rapfi-idx2-distillation-mine]] (the "Bruce Lee one-position" idx-2 champion — a same-size
warm-start / aux-head splice target, distinct from cross-board).

**On this page:** [what it is](#what-it-is) · [the board-size guard](#the-board-size-guard-and-head-splices)
· [why warm-start](#why-it-matters--warm-start-is-the-remedy-for-the-15x15-cold-start-collapse)
· [the curriculum recipe](#the-curriculum-recipe-era-2--path-a-the-2026-06-20-overnight)
· [the auto-graduating LADDER](#the-auto-graduating-ladder-911315) · [how to run](#how-to-run-the-transfer-entrypoint)
· [gotchas](#gotchas).

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

### The board-size guard, and the head splices

`gomoku/model.py load_checkpoint(...)` (≈`model.py:663`) enforces a **board-size contract**: the
checkpoint's embedded `board_size` (older pre-15×15 checkpoints predate the field → default 9)
**must equal the active process board size** or it raises `ValueError` — refusing to build a net
whose heads disagree with every other shape in the process (≈`model.py:711`). The signature is
`load_checkpoint(path, device, *, expect_board_size=BOARD_SIZE, force_aux_vct=False)`.
**Pass `expect_board_size=None` to SKIP the check** — offline inspection only; the warmstart
script deliberately bypasses `load_checkpoint` entirely (it `torch.load`s the raw payload) so it
can read a 9×9 source inside a 15×15 process.

Two same-size warm-start splices ride the same partial-load path (distinct from cross-board, but
worth knowing so you don't confuse them):
- **swap2 choice head** — a pre-swap2 champion has no `choice_*` weights; `load_checkpoint`
  builds the head (config default on) and falls its fresh init back in rather than disabling it.
- **aux VCT-defense head** — `force_aux_vct=True` builds the model WITH the `vct_*` head even when
  the checkpoint predates it, splicing fresh-init `vct_*` params onto an older same-size champion
  (e.g. layering the head onto Bruce/15×15 on a `--resume`). Off (default) ⇒ byte-identical load.
  See [[vct-defense-aux-head-result]].

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

## The auto-graduating LADDER (9→11→13→15)

*(2026-06-20 overnight, era-2 revised. This is the same mechanism run TWO ways — the swap2 ladder
below and the fair-opening ladder #74 further down — differing only in the graduation gate.)*


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

### The TWO graduation gates

The ladder is **auto-graduating**: an out-of-git watcher polls the live rung's wandb history
(never touches the trainer), and when the gate fires it warm-starts the champion up one rung and
relaunches. Two different gates have been used, one per campaign:

1. **Swap2 ladder (era-2) — DRAWMAX.** Graduate when `max(draw, white, black) == draw` (draws are
   the strict plurality of self-play outcomes → white's defense has saturated this board). Denoised:
   draws strict-max for **3 consecutive epochs**, past a **`MIN_EPOCH` guard** (don't graduate
   during the offense-heavy post-warmstart recovery-V, which is short-games, not draws), with a
   **per-rung epoch `CAP` backstop** as the anti-hang safety net.
2. **Fair-opening ladder (era-3, #74) — PLIES-P90 PLATEAU.** Graduate when `plies_p90` **plateaus
   at its peak for 5 epochs** (promote *before* the net learns to retreat / games shorten). Used by
   the fixed-fair-opening curriculum (swap2 OFF, canned Rapfi openers re-centered per board), where
   there's no swap2 negotiation to draw-saturate.

Both gates are implemented in `babysit/ladder_grad.py`, a **pure read of the rung's wandb
history**: it looks up the `wandb_run_id` embedded in the rung's `latest.pt` and evaluates the
criterion — the trainer is completely untouched. **Rung 15 is terminal** in both: 15×15 is too big
to fill, so it basically never draws / never plateaus short — it trains until a STOP sentinel.

### The HONEST caveat — bigger rungs graduated on the CAP, not drawmax

The full swap2 ladder climbed 9→11→13→15 unattended in one night (2026-06-20/21). How each rung
actually graduated (see [[training-run-lineage]] and `TRAINING_WIKI.md` ≈L4518):

| rung | run | epochs | graduated | how |
|---|---|---|---|---|
| 9×9  | `lywhy1ba` | →e102 | 2026-06-20 ~22:50 | **drawmax** (draws 56/75/56 vs white ~14) |
| 11×11 | `8jsd7qzw` | e0→e401 | 02:50 | **CAP** (stable black-edge equilibrium, never drawmaxed) |
| 13×13 | `2dvcxh0b` | e0→e424 | 07:40 | **CAP** (same equilibrium; black edge stronger, draws rarer ~10%) |
| 15×15 | (live)     | e0→   | terminal | runs until `STOP_ladder` |

**Durable lesson: only the smallest board (9×9) cleanly draw-saturates.** With v2a (choice head)
OFF the swap2 negotiation doesn't balance colors, so on 11 and 13 black keeps a genuine first-move
edge (white ~25–35% of decisive games; plies long/healthy ~50–70 = defending, NOT the 0% basin)
and draws never overtake black. So **the CAP backstop — not the drawmax rule — was the real
graduation mechanism for the bigger rungs.** That's fine (it advanced each rung cleanly, and white
"fought and learned" at every rung, Jason's bar), but don't overclaim the drawmax gate: it only
ever fired on 9×9.

### The out-of-git orchestrator scripts (`~/data/swap2/babysit/`)

The ladder's driver lives OUT OF GIT at `/Users/jason/data/swap2/babysit/` (detached, crash-guarded
shell loops + the pure-read grad watcher). Roster:

| script | role |
|---|---|
| `ladder_autochain.sh` | era-2 swap2-ladder orchestrator: detached loop, 20-min train slices, checks graduation between slices, warm-starts + relaunches the next rung. STOP: `touch babysit/STOP_ladder`. |
| `ladder_grad.py` | the graduation watcher — pure read of the live rung's wandb history via the `wandb_run_id` in `latest.pt`; implements both gates (drawmax / plies-p90-plateau). |
| `fairladder.sh` | era-3 fixed-fair-opening ladder orchestrator (#74): 15-min slices, warm-starts between rungs, plies-p90 gate. STOP: `touch babysit/STOP_fairladder`. Rung-9 run `eilfnz1e`. |
| `ladder_status.sh` | human-readable status dump of the running ladder (current rung / epoch / gate progress). |
| `ladder_eval15.sh` / `ladder_rapfi15.sh` | terminal-rung 15×15 evaluation cadence — periodic Rapfi-NNUE eval while the terminal rung trains (records white-vs-Rapfi off the era-1 0% floor to `babysit/eval_results.jsonl`). |

(These are evidence, not code — reproduce the mechanism from this page rather than depending on the
out-of-git files; the cells `G-ladder-11/13/15` and `G{9,11,13,15}-fixed-openings` in `run_sweep.py`
ARE in git and are the `G9-swap2-e2` / fixed-opening recipes verbatim, board size selected by env var.)

### Note (2026-07-02) — #113 re-implemented this warm-start ad-hoc

The 13×13 sound-world graduation (#113, `TRAINING_WIKI.md` 2026-07-02, wandb `8rp0gjpm`) **re-built
the cross-board warm-start by hand** — "fresh 13×13 net + shape-matching tower copy → strict-loadable
via the production board-size guard" — because this page had **no index row** and the mechanism
wasn't discoverable. That's exactly the "don't lose it again" failure this page exists to prevent:
`scripts/warmstart_15x15.py` already does the partial-load with transfer-accounting assertions
(it takes `--board-size 13`), and would have saved the reimplementation. **If you're carrying a
champion up a board size, reach for `warmstart_15x15.py` first.** (Substantive #113 finding, for
context: the warm 9×9 tower transferred offense but 13×13 **white-defense collapsed** — black forces
VCT wins by ply 9–13 so white's sharp-defense examples never enter the buffer; they pivoted to a
from-scratch 13×13 control. See [[sound-world-recipe]] and the #113 entries in `TRAINING_WIKI.md`.)

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
