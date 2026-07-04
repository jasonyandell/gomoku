# AlphaZero — the training arc

The core learning artifact: what AlphaZero taught us on this game. Read
**best-performance-first**, then the headline facts, then the curated
what-worked / what-didn't. Chronological evidence is in
[TRAINING_WIKI.md](../TRAINING_WIKI.md); the compact route map of the run
sequence is [training-run-lineage.md](topics/training-run-lineage.md), and the
**[training timeline](training-timeline.md)** is the curated ~50-milestone index
into the notebook (era-grouped, with run ids).

> **← Hubs:** [index](index.md) · sibling hubs: [Experiments](experiments.md) ·
> [Derby](derby.md) · [M5-as-Mainframe](m5-mainframe.md) · [Reference](reference.md)

## Start → Now → Learned

- **Started:** 9×9 self-play, Z-series → the WL-series (wave-of-lockstep, scale
  emulation, diagnostics). First champions crowned by the autolab 2026-06-19.
- **Now:** the era moved 9×9 → **15×15**, then the **sound-world** recipe
  (oracle-veto + terminus). 9×9 sound-world validated; 13×13 graduation a
  structural negative (#113).
- **Learned:** fast-attack collapse is the enemy; gate on H2H-vs-frozen-champion;
  white-side defense is the binding wound; the net + oracle beats the net alone.

## Best performance so far

| Model | Board | Size | Result | Wound |
|---|---|---|---|---|
| **"Bruce Lee"** ([bruce-lee-model.md](topics/bruce-lee-model.md)) | 15×15 | 128×10 (~3.05M) | The single-opener champion — master idx-2; ~50 Δelo plateau; beats the ex-"champion" 96×8 **40-0** | **white 0/12 vs Rapfi** — the whole shortfall |
| **Sound-world 9×9** ([sound-world-recipe.md](topics/sound-world-recipe.md)) | 9×9 | — | **0-0-40 H2H vs the old champ**; finisher-hybrid **95% vs heuristic** | residual white-vs-la4 softness (5/20 @ e1368) |

*Sound-world caveat:* the headline **0-0-40** is an H2H vs the *old champion*
specifically (evidence: [training-timeline §Era 6](training-timeline.md) /
`TRAINING_WIKI.md` 2026-07-02) — not a claim of universal soundness; white-side
vs lookahead-4 was still slightly soft (5/20) when the 9×9 chapter closed.

*Best-net note:* the "capacity reversal" was a yardstick artifact —
`128×10` is the strongest trained net, `96×8` the weakest (the 2026-06-15
reckoning, since fixed). See
[alphazero-lessons-15x15-gomoku.md](topics/alphazero-lessons-15x15-gomoku.md)
§8–§9.

## Headline facts (the deep pages)

| Topic | Page |
|---|---|
| **The learning artifact** — what AZ taught us at 15×15 | [alphazero-lessons-15x15-gomoku.md](topics/alphazero-lessons-15x15-gomoku.md) |
| The run lineage / "how did we get here?" | [training-run-lineage.md](topics/training-run-lineage.md) |
| The 15×15 campaign (the live push) | [15x15-training-campaign.md](topics/15x15-training-campaign.md) |
| The net + representation (trunk, two heads, line planes, "Fable" rationale) | [net-architecture-and-representation.md](topics/net-architecture-and-representation.md) |
| Cross-board warm-start + the 9→11→13→15 ladder | [board-size-transfer-and-warm-start.md](topics/board-size-transfer-and-warm-start.md) |
| The sound-world recipe (current frontier) | [sound-world-recipe.md](topics/sound-world-recipe.md) |
| Every knob & switch to launch/tune a run | [training-run-reference.md](topics/training-run-reference.md) |
| Run designs (preserved records) | [WL1](topics/wave-of-lockstep-design.md) · [WL2+WL5 index](topics/wl-era.md) |

## What worked

- **Native MCTS + wave-batched eval** — moved the self-play bottleneck off Python
  tree churn onto the evaluator boundary. [mcts-perf-ceiling.md](topics/mcts-perf-ceiling.md)
- **Warm-start across board sizes** — `warmstart_15x15.py` transfers 98.9% of
  params (global-pool trunk), reinits the 3 board-bound FCs.
  [board-size-transfer-and-warm-start.md](topics/board-size-transfer-and-warm-start.md)
- **The sound-world veto on 9×9** — oracle in the *environment* (masks
  proven-losing moves), not the loss; causal ablation confirms the veto is the
  mechanism. [sound-world-recipe.md](topics/sound-world-recipe.md)
- **Soft-target > one-hot** for teacher distillation.
  [eval-teacher-sensei.md](topics/eval-teacher-sensei.md)

## What didn't work (banked negatives)

- **One-hot Rapfi distillation HARMS** — flattens the policy head, regresses the
  net even gently (#77/#86, matched control). [rapfi-idx2-distillation-mine.md](topics/rapfi-idx2-distillation-mine.md)
- **VCT-terminus self-play** — throughput win, robustness **loss**: never learns
  to defend, attack-only specialization (#100/#101).
  [vct-terminus-selfplay-result.md](topics/vct-terminus-selfplay-result.md)
- **VCT-defense aux head** — a working *sensor* with no *actuator*: learns the
  representation, policy never acts on it (#103).
  [vct-defense-aux-head-result.md](topics/vct-defense-aux-head-result.md)
- **13×13 sound-world graduation** — structural negative; walls stop saving white
  at 169 cells (#113). [sound-world-recipe.md](topics/sound-world-recipe.md)

## The open wound & the fixes in flight

The binding constraint is **white-side defense**
([white-side-defense-plan.md](topics/white-side-defense-plan.md)). Two
principled fixes confirmed at the data level (strength-vs-champion still open):
**swap2** ([swap2-opening-protocol.md](topics/swap2-opening-protocol.md)) and
**fixed-fair-openings** ([card](cards/gomoku-15x15-fixed-fair-openings.md)).

**Live "what next" (updated 2026-07-04, after the 13×13 negative):**
- **Rails-v0 TRIED** (2026-07-03, #116, wandb `vraf0b6e`, 15×15 idx-2) — partial:
  dropping the terminus + attacker-preserve DID record white positions (the cure
  worked), but on the black-tilted idx-2 opener white re-collapsed and the
  forced-win tails **poisoned value** (vl→0.03, death-tell tripped); closed at
  e5524. Removing the terminus is necessary but NOT sufficient on a side-favored
  opening. [sound-world-recipe.md](topics/sound-world-recipe.md) § rails-v0.
- **Next levers, in order:** (1) **tail subsampling** (`--tail-subsample`, #118 —
  staged, byte-identical off, Jason-gated); (2) a **fairer opening** (swap2 / a
  less black-favored opener than idx-2).
- **The pivotal unknown: is 13×13 a forced black win?** 15×15 is proven, 9×9 is
  drawish between sound players (a fast black win within cap50), **13×13 is
  UNKNOWN** — probe it with the mega-VCT oracle. Full analysis:
  [sound-world-recipe.md](topics/sound-world-recipe.md) § new directions.

## Full page index — every page in this hub

*Complete map (27 pages); the sections above surface the headline ones.*

| Page | Note |
|---|---|
| [alphazero-lessons-15x15-gomoku.md](topics/alphazero-lessons-15x15-gomoku.md) | the learning artifact; settled-verdicts-first (restructured 2026-07-04) |
| [15x15-training-campaign.md](topics/15x15-training-campaign.md) | the 15×15 era (feasibility merged in 2026-07-04) |
| [training-run-lineage.md](topics/training-run-lineage.md) |  |
| [training-run-reference.md](topics/training-run-reference.md) | also on workflow-train |
| [launch-sequence-runbook.md](topics/launch-sequence-runbook.md) | launch runbook; also workflow-train |
| [net-architecture-and-representation.md](topics/net-architecture-and-representation.md) |  |
| [board-size-transfer-and-warm-start.md](topics/board-size-transfer-and-warm-start.md) |  |
| [sound-world-recipe.md](topics/sound-world-recipe.md) | current frontier |
| [loss-floor-bouncing.md](topics/loss-floor-bouncing.md) | dynamics |
| [az-at-scale-vs-laptop.md](topics/az-at-scale-vs-laptop.md) | dynamics |
| [bruce-lee-model.md](topics/bruce-lee-model.md) |  |
| [white-side-defense-plan.md](topics/white-side-defense-plan.md) | the binding wound |
| [swap2-opening-protocol.md](topics/swap2-opening-protocol.md) |  |
| [eval-teacher-sensei.md](topics/eval-teacher-sensei.md) | teacher distillation; cross reference |
| [rapfi-idx2-distillation-mine.md](topics/rapfi-idx2-distillation-mine.md) |  |
| [wave-of-lockstep-design.md](topics/wave-of-lockstep-design.md) | run design WL1 |
| [wl-era.md](topics/wl-era.md) | WL2/WL5 index (designs archived 2026-07-04) |
| [vct-terminus-selfplay-result.md](topics/vct-terminus-selfplay-result.md) | training result; bridges seek-vct |
| [vct-defense-aux-head-result.md](topics/vct-defense-aux-head-result.md) | training result; bridges seek-vct |
| [cross-game-value-sidecar.md](topics/cross-game-value-sidecar.md) | derby lever; meta-lesson |
| [curated-buffer-and-curriculum-design.md](topics/curated-buffer-and-curriculum-design.md) |  |
| [fpu-reduction-eval-lever.md](topics/fpu-reduction-eval-lever.md) | search/eval lever |
| [swa-averaging.md](topics/swa-averaging.md) | post-training tool |
| [gomoku-9x9-swap2-era2.md](cards/gomoku-9x9-swap2-era2.md) | model card |
| [gomoku-15x15-fixed-fair-openings.md](cards/gomoku-15x15-fixed-fair-openings.md) | model card |
