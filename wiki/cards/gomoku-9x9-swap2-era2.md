---
# HuggingFace-style model card frontmatter (doubles as the HF README on push).
license: mit
library_name: pytorch
tags:
  - alphazero
  - gomoku
  - mcts
  - self-play
  - swap2
  - reinforcement-learning
pipeline_tag: reinforcement-learning
model-index:
  - name: gomoku-9x9-swap2-era2
    results: []   # no external (Rapfi) anchor at 9x9 yet — self-play strength only
---

# gomoku-9×9-swap2 · era-2 (Path-A curriculum, phase 1)

**Status (2026-06-20): IN TRAINING.** A research checkpoint, not a finished player.
This is the *bootstrap* of the era-2 curriculum: train an aggressive, **not-doomed**
white cheaply on the native 9×9 board, then warm-start it up to 15×15
([[board-size-transfer-and-warm-start]]). Numbers below are a live snapshot at **e47**.

## TL;DR

The first swap2 net where **white is not doomed.** Era-1 imported the *lose-slowly
basin* by warm-starting the pre-swap2 **defensive** champion → white played to delay
the loss (0% vs Rapfi). This net starts **fresh on 9×9** and white takes **~45% of
decisive self-play games** from e17 on. That's the whole reason for the curriculum:
bootstrap good (aggressive, balanced) priors before paying 15×15's cost.

## What it is

- **Game:** 9×9 free-style gomoku (five-in-a-row, no overline rules), **swap2** opening
  (opener places 2 black + 1 white; responder stays-white / swaps / places-2-and-defers).
- **Architecture:** AlphaZero residual policy/value net, `size=large`, `global_pool=True`
  (board-size-agnostic conv trunk — the property that makes 9×9→15×15 transfer work).
- **Search:** PUCT MCTS, wave-batched eval, native state-ops + native MCTS (9×9 compiled).
- **Choice head (v2a):** **OFF** (`--choice-head-weight 0.0`). Deliberately killed — it
  trains a head used only in the opening at ~20% throughput cost for no measured gain
  yet. To be revisited *after* white-not-doomed is locked in.

## Training recipe

| Knob | Value | Why |
|---|---|---|
| board | 9×9 (`GOMOKU_BOARD_SIZE=9`) | native ext exists; ~28 s/epoch vs 15×15's ~91 s (**3.2×**); white trainable |
| lineage | **fresh / cold-start** | NOT descended from the era-1 defensive champion → no basin import |
| value-discount | **0.95** (vs prior 0.98) | aggression shaping — faster wins worth more (win@40: 0.13 vs 0.30) |
| swap2 | on | the balanced opening the whole era pivots to |
| sgd-steps/epoch | 64, `--pack-buffer` | the e2 cell recipe |
| cell / run-dir | `G9-swap2-e2` / `G9-swap2-e2-board9` | `scripts/run_sweep.py` (commit `cd93233`) |
| wandb | `gomoku/lywhy1ba` | live run |

## Live snapshot — e47 (run `lywhy1ba`, ~30 min off the e19 launch)

| metric | value | read |
|---|---|---|
| white % of decisive self-play games | **47.7%** (42–48% band since e17) | **white not doomed** — the headline |
| black edge | slight, persistent | expected first-mover advantage; **not** the white<black warning |
| `selfplay/plies_mean` | **16.7**, falling from ~36 | **plies-collapse in progress** — offense found before defense; healthy *because* white still wins ~45% through it |
| plies p10/p50/p90 | 11 / 15 / 24 | games shortening as the rush is discovered |
| draws | 0 | free-style 9×9 rarely draws |

**Trajectory:** white% by epoch — e17 50.0 · e25 43.2 · e31 48.6 · e39 50.0 · e47 47.7.
plies_mean — e17 35.9 → e31 27.0 → e39 22.6 → e47 16.7 (the collapse leg).

## Watch / warning signs (for whoever inherits this run)

- 🟢 **white% < black%** — *the* signal Jason watches. A slight, steady black edge is the
  natural first-mover advantage and is fine. The **alarm** is white% trending *down toward
  0* — that's the net sliding back into the lose-slowly basin (doomed white). **Not seen.**
- 🟢 **plies-collapse is expected** — a `plies_mean` dip as offense outruns defense is a
  normal early-game phase *if the net isn't doomed*; white learns defense and plies recover
  (the V-shape). In progress now.
- 🔴 **fast-attack collapse** — `plies_mean` falling **with a concave buffer-fill** AND
  white% collapsing together = the bad version (rush-fest, no defense developing). Watch
  `selfplay/plies_mean` per [[conventions]] ML-judgment rule.

## Intended use & limitations

- **Intended:** research checkpoint; the phase-1 seed for the 9×9→15×15 warm-start. Target
  ~500 epochs, then `scripts/warmstart_15x15.py` → `G15-swap2-e3` (15×15 phase 2).
- **Limitations:** mid-training; **9×9 only** (board-bound FCs `policy_fc`, `value_fc1` are
  size-specific — the conv trunk transfers, the heads re-init on warm-start); strength is
  **self-play-relative only** — no external (Rapfi/engine-panel) anchor at 9×9 yet, so
  absolute strength here is uncalibrated. Don't quote self-play win% as absolute Elo.

## Lineage & provenance

```
pre-swap2 defensive champion   ──(era-1 warm-start: imported the basin ✗)──>  era-1 15×15 swap2 (white 0% vs Rapfi)
fresh init  ──(era-2 / Path-A, THIS net)──>  9×9 swap2, aggressive, white ~45%  ──(warmstart_15x15 →)──>  15×15 phase-2 (G15-swap2-e3)
```

- Orchestration: `/Users/jason/data/swap2/babysit/overnight_autochain.sh` (detached,
  crash-guarded; `touch STOP_overnight` to stop).
- Intermediate capture policy: **HuggingFace push**, not `cp` of checkpoints
  (`jasonyandell/gomoku-9x9`). This card is HF-README-ready (frontmatter above).

## See also

- [[board-size-transfer-and-warm-start]] — the 9×9→15×15 curriculum this seeds
- [[swap2-opening-protocol]] — the opening, the basin diagnosis (§9), the era-2 pivot
- [[conventions]] — ML-judgment rules (fixed baselines, plies_mean watch, small-n noise)
