# Research Board — the Δelo Derby

A race between 8 fresh-start self-play training recipes ("ideas") to a fixed
**140-epoch budget**, run in **10-epoch chunks**, scored by the model's
**anchored elo** (vs `random`, `heuristic`, `lookahead:depth=2`, and the slow
`lookahead:depth=4` anchor). This is a research board that produces real models:
each idea gets a production-style title card, but the question it answers is
*which training recipe climbs fastest*.

## Rules

- **Race to 140 epochs.** 140 is the milestone because that's roughly where a
  fresh model historically first beats the heuristic baseline. **Beat-heuristic
  (model_elo ≥ 800)** is the early checkpoint; the real prize is the **strongest
  model by epoch 140**.
- **10-epoch chunks.** Each scheduling step advances one idea by a 10-epoch
  increment (`gomoku.train --resume <idea>/latest.pt --epochs 10`).
- **Δelo-rate priority queue.** The not-yet-capped idea whose *most-recent chunk*
  gained the most anchored elo gets the next chunk. **Everyone reaches 140
  eventually; leaders get there first.**
- **Fresh self-play, shared init.** All ideas start from an identical fresh init
  (`--size small --seed 0`). No warm-start, no shared parent.
- **One lever each.** Every idea changes exactly ONE flag vs **C0-baseline** —
  clean attribution. C0-baseline is the control.
- **Anchored-elo scoring.** Score = the last `eval/model_elo` in
  `<idea_dir>/checkpoints/eval_results.jsonl`. Δelo for the queue = the change in
  that score across the idea's most recent chunk.

**Shared knobs** (held constant across all ideas):
`--games-per-epoch 64`, `--training-steps 400`, `--batch-size 256`,
`--replay-buffer-size 100000`, `--lr 1e-3`, `--temperature-moves 8`,
`--c-puct 1.25`, `--size small`, `--seed 0`. C0's `--n-simulations 200` is the
generation-strength control point.

**Out of scope (future board).** Curator / curriculum ideas
(recency-weighted, lru, gomocup-seed) are deferred — they require the
`train_replay` flywheel engine (curated in-RAM sampling over an archived buffer),
not the fresh self-play engine this derby races on. They get their own board once
a headroom parent exists.

---

## Title cards

### C0-baseline
**Lever:** control — sims 200, 64 games, 400 steps, buf 100k, temp 8, lr 1e-3.

**Hypothesis:** The reference climb. Every other card is read as a delta against
C0's anchored-elo trajectory. No claim of its own; it defines "did the lever
help or hurt the *rate*."

**Expected Δelo signature:** A monotone climb that crosses elo ≥ 800
(beat-heuristic) somewhere near epoch ~140 — by construction the milestone is
calibrated to roughly this recipe. Sets the per-chunk Δelo baseline the queue
sorts against.

**Config delta vs C0:** none (`--n-simulations 200`).

---

### sims-400
**Lever:** `n_simulations 400` (vs 200) — stronger policy targets, slower gen.

**Hypothesis:** Deeper MCTS per move yields sharper, lower-noise policy/value
targets, which could steepen the early climb — *if* target quality, not game
volume, is the binding constraint on the fresh-start ascent. The cost is ~2×
slower generation per game, so within a fixed 10-epoch (not fixed-wall) chunk it
trains on the same number of games but better-labeled ones. The risk: at low
model strength the extra sims mostly refine an already-cheap-to-estimate policy,
buying little while the wall-clock per chunk balloons.

**Expected Δelo signature:** *Confirm* = a higher Δelo per chunk than C0,
especially in the mid-climb (epochs 40–100) where target sharpness should matter
most; reaches 140 with a higher final elo. *Refute* = Δelo tracking C0 within
noise (target quality wasn't the bottleneck) while each chunk costs ~2× the wall.

**Config delta vs C0:** `--n-simulations 400`.

---

### sims-100
**Lever:** `n_simulations 100` (vs 200) — weaker targets, ~2× faster gen / more
games per wall.

**Hypothesis:** The LF1 lesson, inverted. LF1 showed that *fast generation
floods the trainer* — cheap gen pushed new-positions/epoch so high the trainer
fell behind and re-ground stale buffer. Here gen is cheap by design: does
cheaper/faster generation *win the climb* (more, fresher self-play per wall
overcomes weaker per-move targets), or does it just produce *noisier targets*
that slow the ascent? This is the early-climb test of the
volume-vs-quality tradeoff that the converged-model flywheel work couldn't isolate.

**Expected Δelo signature:** *Confirm (volume wins)* = Δelo per chunk meets or
beats C0 at a fraction of the wall — cheap gen is the efficiency frontier.
*Refute (noise dominates)* = a visibly shallower climb than C0, late or never
crossing elo 800, the weak targets capping reachable strength.

**Config delta vs C0:** `--n-simulations 100`.

---

### sgd-800
**Lever:** `training_steps 800` (vs 400) — more fit per epoch.

**Hypothesis:** delta-e run-2 found that **extra SGD bought nothing on a
*converged* model** — `lru,sgd=300` netted the identical chess-score to
`lru,sgd=100`, just played sharper, no net strength. But that was a net at its
optimum re-grinding a fixed curated slice. **Does more fit-per-epoch help on the
*climb***, where the net is far from convergence and each fresh batch carries
real un-learned signal? If the binding constraint early is "we under-fit the
data we generate," doubling SGD steps should steepen Δelo. The risk is the same
over-grinding seen at convergence reappears once the buffer is dominated by
stale self-play.

**Expected Δelo signature:** *Confirm* = steeper early Δelo than C0 (epochs
0–60), tapering as the net approaches the data's information limit. *Refute* =
the run-2 result generalizes — Δelo ≈ C0 within noise despite 2× the SGD,
extra grinding wasted even on the climb.

**Config delta vs C0:** `--n-simulations 200 --training-steps 800`.

---

### buf-30k
**Lever:** `replay_buffer_size 30k` (vs 100k) — faster turnover, fits recent
self-play harder.

**Hypothesis:** A smaller buffer turns over faster, so each epoch's SGD sees a
higher fraction of *recent* (stronger-policy) self-play and less stale
early-model garbage. On a fast climb where the policy is improving every few
epochs, weighting toward recent games could steepen Δelo — a fixed-buffer echo
of the recency-weighted curator finding (recency >> lru). The risk: 30k is small
enough to over-fit a narrow recent distribution and lose the diversity that
keeps targets honest, inducing instability.

**Expected Δelo signature:** *Confirm* = Δelo at or above C0 with the gap opening
mid-climb as recency compounds. *Refute* = higher per-chunk variance and/or a
shallower climb, recent-overfit eating the freshness gain.

**Config delta vs C0:** `--n-simulations 200 --replay-buffer-size 30000`.

---

### open-div4
**Lever:** `random_opening_moves 4` (the WL3 lever) — opening diversity,
better-balanced climb.

**Hypothesis:** WL3's diversity lever. Forcing 4 random opening moves spreads
self-play across a wider opening distribution, preventing the fresh net from
collapsing onto one or two dominant lines and over-fitting them. A
better-balanced game distribution should produce a steadier, less-degenerate
climb — and is the same mechanism that fixed delta-e run-2's near-50% decisive
rate (paired random openings made games decisive instead of replaying one line).
The risk: opening randomness adds early-game noise that slows the first chunks
before the diversity pays off.

**Expected Δelo signature:** *Confirm* = a smoother, more monotone climb than C0
with fewer regressions per chunk, and equal-or-better final elo. *Refute* =
slower early Δelo (noise tax) that the diversity never recoups by 140.

**Config delta vs C0:** `--n-simulations 200 --random-opening-moves 4`.

---

### ema-099
**Lever:** `ema_tau 0.99` (the WL2 lever) — EMA self-play weights, smoother
targets.

**Hypothesis:** WL2's stability lever. Generating self-play from an
exponential-moving-average of the weights (τ=0.99) instead of the live net gives
a slower-moving, lower-variance target-generation policy — the actor lags the
learner, so targets stop chasing every SGD wobble. This should reduce
self-play-target variance and smooth the climb, potentially raising Δelo by
keeping the net from training against its own noise. The risk on a *fast* climb:
the EMA lag could hold generation behind the learner's actual strength, slowing
how fast better targets become available.

**Expected Δelo signature:** *Confirm* = lower chunk-to-chunk Δelo variance than
C0 and a steady, regression-light climb. *Refute* = a visibly lagged climb —
Δelo tracking below C0 because the EMA actor keeps generating from a staler,
weaker policy than the learner has already reached.

**Config delta vs C0:** `--n-simulations 200 --ema-tau 0.99`.

---

### temp-16
**Lever:** `temperature_moves 16` (vs 8) — more opening exploration.

**Hypothesis:** Doubling the temperature-1 (sampling) window from 8 to 16 plies
keeps self-play exploratory deeper into the game before switching to greedy
selection. More exploration → broader state coverage and richer policy targets
in the early/mid game, which could steepen the climb the same way opening
diversity does — but via in-game sampling rather than forced random openings.
The risk: 16 plies of sampling injects weaker, higher-entropy moves into the
training data, diluting target quality and slowing convergence.

**Expected Δelo signature:** *Confirm* = Δelo at or above C0 with better coverage
showing as a steadier mid-climb and equal-or-higher final elo. *Refute* = a
shallower climb than C0 — the extra sampled-move noise outweighs the coverage
gain.

**Config delta vs C0:** `--n-simulations 200 --temperature-moves 16`.

---

<!-- STANDINGS:AUTO — delo_derby.py rewrites everything below this line -->

## Standings

_Last updated: 2026-05-24T06:04:20Z — 0 chunks run._

**Champion so far:** _(no rated ideas yet)_

| Rank | Idea | Epochs | Elo | Δelo (last) | Δelo/epoch | Beat heuristic? | Status |
|-----:|------|:------:|----:|------------:|-----------:|:---------------:|--------|
| 1 | C0-baseline | 0/140 | — | — | — |  | queued |
| 2 | sims-400 | 0/140 | — | — | — |  | queued |
| 3 | sims-100 | 0/140 | — | — | — |  | queued |
| 4 | sgd-800 | 0/140 | — | — | — |  | queued |
| 5 | buf-30k | 0/140 | — | — | — |  | queued |
| 6 | open-div4 | 0/140 | — | — | — |  | queued |
| 7 | ema-099 | 0/140 | — | — | — |  | queued |
| 8 | temp-16 | 0/140 | — | — | — |  | queued |

