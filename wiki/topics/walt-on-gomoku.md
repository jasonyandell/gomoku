# walt on gomoku — a deliberate misapplication, at best effort

**DEAD-END (lesson kept)** *(2026-10-07)* — as a *player*, walt ties MCTS with the same net
against a real engine. Its big wins came only against opponents shaped like its own model.

## Verdict (2026-10-07)

texas-42 walt (level-k best response over sampled hidden deals, played from trick 1) was moved
onto perfect-information 9×9 freestyle, with the GPU mega-VCT oracle as its exact terminal.
- **Against Rapfi-NNUE** (50 ms and 1 s, about 20 games per arm), walt ties MCTS using the same
  champion net.
- **Against weaker opponents it looks great:**
  - 10-0 and 9-1 vs its own field model;
  - 4-0-1 head-to-head vs MCTS1600 at half the wall time;
  - converts more wins than MCTS vs lookahead4.

**Why.** The fast VCT oracle **detects opponent mistakes**, and walt's best response over the
field tapes **amplifies that signal**. walt steers into positions where the modeled opponent lets
a forced win appear.
- The share of leaves the oracle resolves falls from **53% vs the field** to **12% vs Rapfi@50**
  and **4% vs Rapfi@1000**.
- The finisher fires on 38% of moves vs the field, 1.4% vs Rapfi@50, and on 0 of 53 moves vs
  Rapfi@1000.
- Rapfi never lets a VCT appear. walt is then just net@T1 playouts plus a value-head cutoff,
  which carries no more information than MCTS with the same net.
- This is walt's own documented failure mode (plunge cross-game reports): **the gain vanishes
  once the opponent isn't the modeled field.**

**The deeper why: there is almost no hidden information to reason about.** Two candidates for
the hidden information:
- **Opponent tapes** (the random seed of a stochastic opponent model). These are a manufactured
  stand-in for a deal.
- **Jason's reframe (the right one): logical uncertainty**, i.e. *where the VCT is*. The truth is
  fixed, and you see it as you approach, the way cards are revealed by play.

Measured on 16K field positions:
- A clean no-win at 50 nodes **never** flips to a win at 2,000 nodes (it's a proof).
- Overall, only 0.7% of positions flip.
- Of the 14% that hit the 50-node cap, 4.9% flip, and most stay unknown even at 40×.

The hidden card is rare, rarely a win, and mostly unrevealable at any affordable budget.

## What we built

All code is in `scripts/walt/`.
- **Tickertape.** The field is policy (net@T or uniform-near) plus the VCT finisher, sampled via
  `splitmix(zobrist(board) ^ tape)`.
  - The same tape at the same position draws the same move in every branch, so tapes partition
    into buckets instead of multiplying (texas-42 Def 3.5).
  - The finisher makes a VCT leaf an exact statement about the modeled field.
- **Level 1.**
  - walt's nodes take the max over the top-M prior candidates, choosing one action that covers
    every tape.
  - Field nodes sum over the buckets of tapes.
  - Payoff is W=2, D=1, L=0 per tape.
- **Native-batch lockstep rollouts** (`walt_vec.py`).
  - All (leaf, tape) playouts advance one ply per net call plus one VCT call.
  - Parity-checked byte-identical against `GameState` (planes, five-check, rollouts).
- **Rollout cap.** After `cap` plies the value head scores what's left. Needed because cost is
  plies × per-call latency.

## Measurements

| Finding | Number |
|---|---|
| Oracle batch curve, cap50, real 9×9 boards | 16 → 50 ms · 1K → 227 ms · 4K → 202 ms · **16K → 264 ms (62K boards/s, the knee)** · 32K → 470 ms |
| Before batching (K=32) | 70 calls/move at ~53 boards; VCT = 90% of decision wall |
| More tapes, K 32→4096 | 128× the samples for 2.7× the time; **decisions identical on all 6 test positions** (bias-limited, not sample-limited) |
| H=4 cost | 11.5 s/move with a 69-ply lockstep tail → **5.4 s** at cap 12 (16 calls × full 16K batches) |
| Samples / depth / width at equal batch (vs lookahead4) | 6-0-4 / 7-0-3 / 6-1-3: no lever moved strength |
| Rapfi, white column | **0 wins as white for every arm** (the chronic white-defense gap; walt doesn't touch it) |

Bets were stated before results (session 2026-10-07):
- **B1** (leaves go to rollout): confirmed.
- **B2** (oracle-latency mismatch): confirmed, then fixed by batching.
- **B4** (little hidden mass): confirmed.
- **B3′** (walt doesn't beat MCTS): **refuted against net-shaped opponents, confirmed against
  Rapfi.**

## What would still be worth trying (not run)

- **walt as an exploit finder.** Point it at a frozen net as the field to map *where* that net
  leaves VCTs open. It's a diagnostic, not a player.
- **Adaptive sampling** (texas-42's racing, paired sign-test elimination). It's moot here because
  the estimator is bias-limited, not variance-limited.

## Evidence

- `~/code/gomoku/sweep_logs/walt/` (gitignored evidence: phase1, phase2_bench, phase2b, phase3 JSONL; per-move walt diagnostics).
- [TRAINING_WIKI.md](../../TRAINING_WIKI.md) 2026-10-07.
- Branch `feat/walt-gomoku`.
- walt sources:
  - `~/code/texas-42/wiki/walt.md`
  - `~/code/plunge` branch `claude/walt-game-implementations-wrsu6q` (`lab/WALT-CARD.md`)
  - **Not** mk5-main's walt, a different algorithm with the same name.
