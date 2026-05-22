# WL2 — Scale-Emulation Design

Status note (2026-05-21): implemented and launched as WL2 (`9wng4yu9`).
The four-lever stack smoothed the early trajectory and raised the la4 peak,
but did not solve retention. Treat this page as the preserved WL2 design
record; read [training-run-lineage.md](training-run-lineage.md) and the WL2
run-end section in [../../TRAINING_WIKI.md](../../TRAINING_WIKI.md) for the
result.

Design recorded 2026-05-20 by Jason and assistant after `WL1` (wandb
`l8mbntcm`) showed strength regression past its e360-499 peak: model
oscillated between elo 620 and 1281 over 1100+ epochs without sustaining
any prior peak. The hypothesis we ruled in: **the 8-worker laptop setup
has structurally less in-flight version diversity than AlphaZero-at-scale
has by default, and that diversity is what stabilizes large-batch SGD on
self-play data.**

Cross-refs:
- [wave-of-lockstep-design.md](wave-of-lockstep-design.md) — the per-version
  uniformity hypothesis WL1 tested.
- [az-at-scale-vs-laptop.md](az-at-scale-vs-laptop.md) — the framing this
  design extends with concrete laptop-side emulations.
- [../../TRAINING_WIKI.md](../../TRAINING_WIKI.md) "WL1 live run log" —
  the run end summary that motivated this design.

## What WL1 told us

WL1 validated the per-version uniformity fix operationally (clean tiles,
barrier worked, race fix held). It also reached strength milestones
~5-8× faster than `az-recipe-160k` per epoch.

But by e500 the run had broken into a different failure mode than Z:
where Z's arcs were 800-1000 epoch wide with eventual recovery, WL1's
strength dynamics became *high-frequency chaotic oscillation* — elo
bouncing 620-1281 across single-eval intervals, la4 regressing from a
sustained 52% peak to ~5% over 1100 epochs, heuristic bouncing 0-50%.

Reframing: WL1 broke the per-version-bias feedback loop and replaced it
with a per-version *reactivity* loop. The model isn't more diverse;
it's more *reactive*. Each version's clean tile pulls the gradient
harder, and the next self-play wave then runs against an over-pulled
brain, generating data that pulls the next gradient hard in some new
direction. No averaging across in-flight versions, no stabilization.

## Why AZ at scale doesn't have this

At AlphaZero scale (~5000 TPUs for self-play, batch 4096):

1. **Implicit version diversity in the buffer.** At ~125k concurrent
   games and async publish/load, the buffer contains positions from
   *many* slightly different snapshots at any moment, not from one
   per-tile snapshot. The "brain that generated this data" is itself a
   distribution.
2. **Large-batch gradient noise reduction.** Batch 4096 against a
   500k+ position buffer averages over a broad, stable distribution.
   Per-step gradient variance — the dominant cause of training
   instability — is much smaller.
3. **Async training/generation lag.** The model used for self-play is
   inevitably a few hundred SGD steps behind the model being trained,
   because publishing weights to thousands of workers takes time. This
   accidental EMA-like effect decouples self-play data from gradient
   updates.
4. **No explicit past-checkpoint mix needed.** AlphaGo had explicit
   past-checkpoint pairing; AlphaZero dropped it because items 1-3
   supplied the diversity for free.

Our 8-worker laptop setup has *none* of those four properties. WL1
specifically removed the partial version diversity Z had accidentally
(via async restart timing) by enforcing one-version-per-tile.

## Hypothesis

A combined four-lever stack, each emulating one AZ-scale property,
together stabilizes training enough to retain strength rather than
oscillate. We expect WL2 to:

- Stop la4 from regressing past prior peaks (the hardest signal WL1 lost)
- Sustain a strength level across consecutive evals (no more 1281 → 620
  → 871 swings)
- Show eval-to-eval variance roughly halving (rough order-of-magnitude
  prediction, not a precise commitment)

Failure modes that would refute this:
- WL2 oscillates as much as WL1 → diversity is not the limiting factor;
  next suspects are capacity (move to WL3 = medium model) or fundamental
  policy/value head capacity for 9x9 with the chosen MCTS budget.
- WL2 is *more* stable but caps at lower strength → the diversity stack
  costs us learning rate. Either tune it down or accept the tradeoff.

## Architecture: the four levers

### Lever 1: EMA self-play weights (biggest single intervention)

The trainer maintains both raw weights `θ` (the brain that learns) and
EMA weights `θ_ema` (the brain that plays):

```
θ_ema = τ * θ_ema + (1 - τ) * θ    # τ ≈ 0.99 or 0.999, applied each
                                   # SGD step or each wave boundary
```

Workers load `θ_ema` for self-play. Trainer updates `θ` via SGD and
updates `θ_ema` via the EMA rule. On checkpoint save, persist both.

**Effect:** the self-play opponent changes much more slowly than the
training brain. Within a single wave, the brain has done one SGD step,
but the self-play data was generated against a brain that's effectively
~100 SGD steps behind. This is exactly the async lag that AZ has
naturally from publish-to-workers latency.

This is the canonical "stability for self-supervised feedback loops"
trick — BYOL, MoCo, many modern AZ forks (e.g. some KataGo variants).

**Knob:** `τ`. Higher = slower-moving self-play brain. Start at 0.99
(half-life ~70 SGD steps).

**Implementation:** ~30 lines in `gomoku/train.py`. EMA tensor lives
beside `model.state_dict()`, persisted to `worker_weights.pt` instead
of the raw weights.

### Lever 2: Past-checkpoint opponent mix

Each wave, with probability `p_mix`, a worker plays its games against
a past checkpoint instead of the current model. Mix distribution:

- 40% probability: random checkpoint from the *last 100 model versions*
- 10% probability: random checkpoint from *anywhere in run history*
- 50% probability: vs current EMA self-play brain

The "past checkpoint" worker plays *both sides* against the past
checkpoint (full reproduction of that snapshot), then attributes the
game to the current model's tile so the trainer ingests it normally.

**Effect:** direct opponent diversity. The current model sees positions
it can't reach from pure self-play because the past-checkpoint plays a
different style. La4 regression specifically should be addressed
because the model has to remember how to beat old versions of itself.

**Knob:** `p_mix` fractions above (40/10/50). Cheaper variant: just
40% recent / 60% current; ablate "ancient" axis later.

**Implementation:** `gomoku/selfplay_worker.py` already has an
`opp_picker` parameter for non-self-play opponents; need to wire a
"checkpoint mix" picker that randomly loads from the checkpoint dir.
~80 lines.

### Lever 3: Worker poll jitter (free, almost-no-code)

Currently every worker checks for new weights every `weights_poll_sec`
(2s by default). All 8 workers see the new model essentially at the
same time. Replace with: each worker has its own poll interval drawn
from `Uniform(2s, 8s)` plus a random per-worker offset, so they pick
up new weights with natural skew.

**Effect:** in any given wave, some workers are 0-3 waves behind on
weights. That replicates AZ's natural async-publish skew. Costs
almost nothing.

**Note:** in wave mode this interacts with the barrier. We need to
make sure jitter only affects *which version a worker starts its tile
with*, not the barrier semantics. Workers still write to the correct
versioned outbox dir.

**Knob:** jitter window (default 2-8s).

**Implementation:** ~5 lines in `selfplay_worker.py`. Each worker
samples its poll interval once at startup.

### Lever 4: Gradient accumulation 4×

Accumulate gradients across 4 minibatches before stepping the
optimizer. Effective batch 2048 (4× 512). Halve the effective LR if
needed (or use linear-scaling: keep LR same, expect somewhat hotter
training).

**Effect:** each gradient step averages over 4× more positions,
cutting per-step gradient noise by ~2× (`√4`). This is the simplest
emulation of AZ's batch 4096 against a buffer that's not 500k+ but
at least sees a broader recent slice per step.

**Knob:** accumulation factor (default 4).

**Implementation:** ~5 lines in `gomoku/train.py`. Change the
per-step `optimizer.step()` to accumulate.

## Property invariants we want to preserve from WL1

1. Wave barrier semantics unchanged (8 workers × 8 games per worker)
2. Per-version tile uniformity (no mid-game weight reloads)
3. Race-cleanup recovery (commit `0d2c106`) stays in place
4. Sampling stays uniform over buffer positions (do *not* combine with
   version-stratified sampling in this run — keep WL2 focused on
   in-flight diversity, not buffer sampling structure)

## Held-back levers (for follow-up runs)

Deliberately *not* changing in WL2 so we can isolate the
in-flight-diversity hypothesis. If WL2 doesn't move the needle, the
next candidates in order:

1. **Bigger model** (small 324k → medium ~1M params) — WL3. Capacity
   theory. Run after WL2 either way; bigger model is the eventual
   right answer regardless.
2. **Version-stratified buffer sampling** — sample equally from K random
   version-buckets, ensuring each batch contains positions from
   diverse training versions. Targets gradient signal quality
   directly.
3. **Stochastic worker checkpoint snapshots** — workers actively run
   *different recent checkpoints* (not just past-mix games), so
   generation itself is multi-model.
4. **Symmetry-breaking via Dirichlet bump** (α 0.13 → 0.25, ε 0.25 → 0.4)
   — more root exploration; cheap; orthogonal to diversity story.
5. **Random opening plies** (Jason previously rejected for gomoku;
   revisit only if everything else fails).

## Why we expect this to work

WL1 isolated and validated the per-version-uniformity property. What it
exposed is a *second* property that's load-bearing for stability at any
scale and that scale provides for free: in-flight version diversity in
the data-generating distribution. The four levers each provide one
piece of that diversity:

- EMA → temporal smoothing of the generator
- Past-checkpoint mix → explicit opponent diversity
- Worker poll jitter → spatial (per-worker) version skew
- Gradient accumulation → reduced gradient noise so the trainer doesn't
  amplify whatever diversity remains

Stacked, this should approximate the AZ regime closely enough on a
laptop that the chaotic oscillation we saw in WL1 dampens to something
that *retains* progress.

## Cost estimate

- Implementation: ~120 lines of code total across train.py +
  selfplay_worker.py + a new "checkpoint mix" picker in match.py
- Per-cycle cost: EMA update (negligible), past-checkpoint loads
  (~50ms per loaded snapshot, amortized once per wave for the affected
  workers), gradient accumulation (4× the SGD time but SGD time is
  ~0.5s/cycle currently → 2s instead of 0.5s, still way under
  generation time)
- Throughput hit: expect 5-15% wall-clock slowdown vs WL1 from the
  combined items above
- Wall-clock for 5000 epochs: estimate ~1.5-2× WL1's wall time, so
  ~2-3 hours total at WL1's pace, maybe 4-5h if plies regrow

## Sanity test before full launch

Same playbook as WL1: 50-100 epoch smoke run with 4 workers, verify:
- EMA tensor saves/loads correctly across trainer restart
- Past-checkpoint picker loads valid weights and generates legal games
- Worker poll jitter doesn't corrupt wave-barrier semantics
- Gradient accumulation gives reasonable losses (no NaN, no obvious
  step-size pathology)

Then full WL2 launch.

## Open questions for the implementation session

- EMA cadence: per-SGD-step or per-wave? Per-step matches BYOL et al.;
  per-wave is cheaper. Default to per-step, can ablate later.
- Should EMA weights be the *only* thing workers see, or should we
  occasionally let workers use raw weights for one tile to make sure
  the system can detect when raw and EMA diverge pathologically?
  (Default: workers always see EMA; trainer monitors `‖θ - θ_ema‖`
  and logs it.)
- Past-checkpoint window: last 100 versions is a guess. Could narrow
  to last 50 if storage / load time becomes an issue.
- Gradient accumulation interaction with `sgd_per_position=0.0025`:
  need to decide whether the rate is "raw SGD calls" or "effective
  optimizer steps." Probably the latter; document explicitly.

## Cross-ref summary

This page is the design for the next run after `WL1`. It assumes:
- Wave-of-lockstep barrier from
  [wave-of-lockstep-design.md](wave-of-lockstep-design.md) stays
- Native MCTS from `gomoku/_mcts_native.c` (commit `0ab3d9d`) stays
- Worker race-drop fix from commit `0d2c106` stays
- Cell name in `scripts/run_sweep.py` should be `WL2`

The launch sequence we built for WL1 (title card → human ACK → spin-up
verification → wiki run-log section → wandb workspace → self-pacing
monitor) is reusable. Worth canonicalizing as a topic page or skill
after WL2 lands.
