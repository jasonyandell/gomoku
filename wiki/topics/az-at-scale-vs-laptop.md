# AlphaZero at Scale vs On a Laptop

A framing observation Jason articulated (2026-05-20) while watching
`buffer/age_mean` oscillate and `selfplay/plies_mean` swing between
exploration and consolidation arcs. The wrinkles we keep encountering
aren't bugs in our training; they're laptop-scale artifacts that
disappear when you have thousands of machines.

## Three structural differences from Google-scale AlphaZero

### 1. Concentration per weight version

Google's AlphaZero runs at thousands of self-play workers per training
machine. Each weight version is sampled by hundreds-to-thousands of
parallel games before training advances. Buffer composition smooths out
across many machine's contributions.

Our 8-worker setup concentrates each weight version into 32-64 games
total. When the model briefly favors short attack lines, the buffer's
recent-version slice is *dominated* by those short games. When the model
explores something new, the new behavior takes 100% of the next version's
buffer slice instead of being averaged with many parallel decisions.

Effect: we see "exploration arcs" — sharp swings in pl/vl over 50-100
epochs as one model version's whole self-play story dominates the
buffer's recent slice. Google's setup smooths these into gradual drifts.

### 2. Game length and per-game signal density

Go games run 200-250 moves. 9×9 freestyle gomoku games at strong play
often resolve in 12-50 moves; aggressive-attack regimes can drop below
5 moves. The same 8 self-play augmentations applied to a 200-move game
gives 1600 training positions; applied to a 5-move game gives 40.

A decisive 5-move game has:
- 5 positions of training signal
- Each position is an opening or near-opening
- No mid-game strategic content
- No endgame technique

Self-play buffers full of 5-move games miss the entire strategic-middle-
game distribution that longer games provide. Google's setup doesn't have
this failure mode at all because Go games are always long.

Effect: when our model collapses to fast attacks, the buffer's information
*content* per ingested position drops. The trainer is computing gradients
against the same 4 opening positions over and over.

### 3. Restart and recovery dynamics

Our laptop setup has noticeable restart artifacts (the `age_mean` bumps to
100-130 at e3000-3700 visible in the chart). When we kill and resume from
`latest.pt`, the buffer's per-slot weight-version tags are frozen at save
time; the model continues training while those slots age 1 epoch per cycle
until eventually evicted. This produces a multi-hour transient of "stale
buffer + new model" before the system re-equilibrates.

Google's setup never restarts a single run — it runs continuously across
thousands of machines for weeks. Restart artifacts are non-existent.

## Why we get away with this anyway

The current run's empirical claim, in Jason's words: *"steady progress is
what makes it learn, and this one has been learning despite the chaos."*

The mechanism: AZ training is robust to a large amount of buffer
composition noise as long as the signal stays directionally consistent.
Each cycle's gradient is computed against ~12k positions; even if 90% of
them are biased toward one regime, the policy improvement direction
averages out across many cycles. The model's `model_elo` trajectory
across the run confirms this — three exploration arcs, three
consolidations, each consolidation stronger than the last (1665 → 1718 →
... still in progress).

The wrinkles aren't training failures; they're *transients in a noisy
process that's still converging*. The framework's robustness budget is
generous; we keep using it up but don't blow through it.

## How this should shape our next-run decisions

When you read the wiki notebook entries and see swings in any of:
- pl / vl creeping up across 50-100 epochs
- plies dropping into short-attack regimes
- model_elo bouncing 200-300 Elo per check-in
- buffer/age_mean drifting up or down

The *default interpretation* is "this is a laptop-scale-AZ transient,
the model is doing something interesting." The *failure interpretation*
("training is broken") requires extra evidence — typically NaN losses,
processes dying, OR sustained degradation across multiple consolidation
periods.

This framing argues against twitchy interventions. Each time we restart
the run, we pay 100+ epochs of "buffer re-equilibration" cost; if the
intervention isn't worth 100 epochs, don't do it.

This framing also argues *for* big-buffer + lockstep + many-games-per-
version setups for the next run (filed in
[TRAINING_WIKI.md](../../TRAINING_WIKI.md) "Next-run config sketches"):
those choices specifically mitigate items 1-3 above. They're laptop-scale
approximations of the smoothing Google gets from sheer parallel volume.

## References

- `selfplay/plies_mean` and `selfplay/plies_p90` on the wandb dashboard
  for the current `sppjo3z5` run — see how plies oscillate between
  5-move attack regimes and 50+ move defensive regimes.
- `buffer/age_mean` on the same — the climb-then-fall pattern as
  defense was learned and ingest rate caught up with buffer size.
- TRAINING_WIKI.md "Lockstep vs continuous" section — concrete
  orchestration choice that reduces the per-version concentration
  problem (item 1).
- TRAINING_WIKI.md "Next-run config sketches" — the bigger-buffer
  trade-off table for raising age_mean toward 200-250.
- `feedback_self_play_eta.md` memory — the related ETA calibration
  that cycle time scales super-linearly with plies, which is itself a
  consequence of game-length variance (item 2).
