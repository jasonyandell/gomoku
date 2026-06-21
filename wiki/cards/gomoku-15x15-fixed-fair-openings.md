---
# HuggingFace-style model card frontmatter (doubles as the HF README on push).
license: gpl-3.0
library_name: pytorch
tags:
  - alphazero
  - gomoku
  - mcts
  - self-play
  - gomocup
  - reinforcement-learning
pipeline_tag: reinforcement-learning
model-index:
  - name: gomoku-15x15-fixed-fair-openings
    results: []   # in training; fairness experiment, not a strength claim yet
---

# gomoku-15×15 · fixed-fair-openings (era-3, the fairness experiment)

**Status (2026-06-21): IN TRAINING.** A research checkpoint and a controlled experiment, not a
finished player. Run `nbctsiua`; cell `G15-fixed-openings`; loop
`babysit/g15fixed_loop.sh`. Plan: [[swap2-opening-protocol]] §11.

## TL;DR — the experiment

We diagnosed that swap2 self-play was an **unfair game**: our *opener* can't compose balanced
openings (it samples a policy head with no fairness gradient), so a competent responder always
swaps to black and white is stuck a doormat (~25–35% of decisive games, 0% vs Rapfi). White was
playing a **-EV seat** — a rigged game no amount of training wins ([[swap2-opening-protocol]] §10,
#73). This net **sidesteps the broken negotiation**: every game starts from a **known-fair board**
and the net just *plays*. The question it answers: **on a fair board, does white engage and trend
toward ~50/50?** If yes, fairness was the lock. If white still collapses from fair starts, the
weakness is real skill, not the seat.

## What it is

- **Game:** 15×15 free-style gomoku (five-in-a-row, no overline rules).
- **Opening:** every self-play game starts from one of **Rapfi's 9 balanced (known-fair) swap2
  openings** (2 black + 1 white, white to move), placed **directly** — **no negotiation, no opener
  policy, no choice head.** The net engages only *post-opening*. Construction is byte-identical to
  `swap2.OpeningState.to_normal()` for the 2B+1W→white-to-move (SWAP) outcome. D4 augmentation fans
  the 9 openings into ~72 mirror/rotation variants. (`--fixed-openings`, commit `3c6e9d7`, tested.)
- **Architecture:** AlphaZero residual policy/value net, `size=large`, `global_pool=True`.
- **Search:** PUCT + Gumbel root + Sequential Halving, native C MCTS (15×15), wave-batched eval.

## Training recipe

| Knob | Value | Why |
|---|---|---|
| board | 15×15 (`GOMOKU_BOARD_SIZE=15`) | the openings are balance-searched *for 15×15* (balance is board-specific) |
| lineage | **fresh / from-scratch** | era-3; NOT warm-started — a clean test of fair-from-birth play |
| opening | fixed book (9 fair openings), `fixed_openings=True`, swap2 OFF | sidestep the unfair opener |
| value-discount | 0.95 | aggression shaping (faster wins worth more) |
| recipe | the e2 cell otherwise (gumbel-m16, 64 sgd/epoch, pack-buffer, EMA, 8 workers) | |
| wandb | `gomoku/nbctsiua` | live |

## What to watch (the experiment's read-out)

- 🟢 **white-share of decisive self-play games → ~50%** — THE signal. On fair boards white should
  stop being the doormat. Every prior era pinned white ~25–35%.
- 🟢 plies healthy (real games), draws moderate, no fast-attack collapse.
- 🔴 **white-share stays ~25–35% from fair starts** → the seat wasn't the (only) problem; white has
  a genuine skill gap → pivot to a white-side defense teacher (#44).

## Intended use & limitations

- **Intended:** a controlled fairness experiment + the seed for an "expand" phase (Phase 2: bring a
  *trained* fair-opening generator / negotiation back — choice head v2a→v2b, or a balance-search
  generator over our own value head; then broaden the opening distribution).
- **Limitations:** **narrow by design** — it only ever sees 9 openings (×8 symmetries), so it is
  **out-of-distribution everywhere else**: expect ~0% on a standard empty board or vs Rapfi's
  openings (the same OOD that made the era-2 ladder net read 25% in-distribution but 0% non-swap2).
  It also **cannot play swap2** (there's no negotiation head). Strength here is **relative** (vs
  `anchor_e455` H2H) — don't quote self-play balance as absolute Elo. Mid-training.

## Lineage

```
era-2 ladder (9→11→13→15 swap2, best net epoch0235 = 25% vs Rapfi, PRESERVED)   — trained to PLAY swap2 on a rigged board (white capped ~0–29%)
fresh init  ──(era-3, THIS net)──>  15×15, fixed FAIR openings, no negotiation   — isolates: does FAIR play unlock white?
```

The era-2 ladder net is kept as a reference (best swap2 player so far); era-3 changes the *question*
(remove the rig), not the lineage.

## See also

- [[swap2-opening-protocol]] §10 (the fairness diagnosis — the opener is the broken half) and §11
  (this experiment's plan)
- [[gomoku-9x9-swap2-era2]] — the era-2 9×9 bootstrap card
- [[board-size-transfer-and-warm-start]] — the era-2 ladder this branches from
