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
- **Current-elo priority queue.** The not-yet-capped idea with the highest
  *current* anchored elo gets the next chunk — "feed the leader." **Everyone
  reaches 140 eventually; leaders get there first.** (Revised 2026-05-24 mid-run
  from a last-chunk-Δelo rule, which pathologically fed the *worst* idea: a
  floor-stuck idea at Δ0 outranked strong climbers whose latest chunk was a
  downward oscillation. Ranking on elo *level* feeds the genuinely-strongest.)
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

## v1 FINAL — verdict (called 2026-05-24 at 5/8 capped; ranked by ELO, NOT wall-clock)

> **Wall-clock is busted for this run** and must NOT be used to rank: the derby ran
> single-process (`gomoku.train`, one stream, GPU ~30%), so every wall-time / Δelo/hr
> here is single-stream and unrepresentative of production (wave-mode, 8 workers,
> saturated). The under-counting trap ([[project-perf-bench-lesson]]). Rank by elo.

| rank | idea | lever | peak | final | beat-heur @ep |
|---:|---|---|---:|---:|:--:|
| 1 | **open-div4** | random_opening_moves 4 (WL3) | **1385** | 1385 | 90 |
| 2 | **temp-16** | temperature_moves 16 | **1340** | 1240 | 90 |
| 3 | sgd-800 | training_steps 800 | 1284 | 1081 | 70 |
| 4 | sims-400 | n_simulations 400 | 1265 | 1094 | 50 |
| 5 | buf-30k | replay_buffer 30k | 908 | 751 | 110 |
| — | C0-baseline | control | 567 (climbing, called @ep60) | — | — |
| — | ema-099 | ema_tau 0.99 | 405 (floor, ep50) | — | — |
| — | sims-100 | n_simulations 100 | 389 (NEVER grokked, ep110) | — | — |

**Findings:**
1. **Exploration/diversity levers win the ceiling.** Random openings (1385) and high temperature (1340) are the top 2 — *above* the compute levers (more sims 1265, more SGD 1284). Diversifying self-play raises reachable strength.
2. **Compute levers grok FASTER but peak LOWER.** Beat-heuristic timing tracks per-epoch compute: sims-400 @ep50 < sgd-800 @ep70 < open-div4/temp-16 @ep90 < buf-30k @ep110. More sims/SGD = earliest crossing; exploration = highest ceiling.
3. **Overtraining is real and lever-dependent.** sims-400/sgd-800 peaked ~ep90 then regressed ~180 elo by ep140; `open-div4` ended *at* its peak (openings sustain the climb, no overtrain); temp-16 mild (1340→1240).
4. **`sims-100` (100 sims) never groks** — floor 389 at ep110. Weak MCTS targets cap the climb (and it's the only *trainer-bound* recipe: gen<train).
5. **Generation-bound, not trainer-bound.** Train is a fixed ~10.5s/epoch floor; MCTS generation is 2–5× that and scales with sims. The trainer is cheap; Δelo/hr leverage is all on generation speed.
6. **Method fixes (mid-run):** priority must rank by *current elo*, not last-chunk-Δelo (the latter fed the *worst* idea); peak checkpoints were lost to `keep-last-n=3`.

## v2 — what's next (queued)

Re-run the **top 3** (`open-div4`, `temp-16`, `sgd-800`) **HEAD-TO-HEAD**, using the **production multiprocess recipe** (`run_sweep` wave-mode, 8 `selfplay_worker`s — saturates the GPU, so wall-clock is REAL). Eval = round-robin *direct matches* among the 3 (they're all >1280, so they'd saturate the anchor ladder — head-to-head via `delta_e_harness --head-to-head` is the correct eval for strong models). **Wall-native budget** (hours, not epochs; chunk = wall-slice; allocate+stop by Δelo/hr) measured on the saturated machine so Δelo/hr is finally honest. Carry the fixes: current-elo priority, peak-checkpoint snapshotting. (Verified 2026-05-24: the production recipe IS multiprocess; the single-stream drift was only in the v1 derby harness.)

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

_Last updated: 2026-05-24T15:31:05Z — 92 chunks run._

**Champion so far:** `open-div4` at 1385 elo (140/140 epochs).

| Rank | Idea | Epochs | Elo | Peak | Wall (min) | Δelo/hr | Beat heuristic? | Status |
|-----:|------|:------:|----:|-----:|-----------:|--------:|:---------------:|--------|
| 1 | open-div4 | 140/140 | 1385 | 1385 | 73.5 | 813 | ✓ | capped |
| 2 | temp-16 | 140/140 | 1240 | 1340 | 76.2 | 823 | ✓ | capped |
| 3 | sims-400 | 140/140 | 1094 | 1265 | 140.1 | 614 | ✓ | capped |
| 4 | sgd-800 | 140/140 | 1081 | 1284 | 105.0 | 874 | ✓ | capped |
| 5 | buf-30k | 140/140 | 751 | 908 | 77.9 | 488 | ✓ | capped |
| 6 | C0-baseline | 60/140 | 567 | 567 | 27.8 | 384 |  | queued |
| 7 | sims-100 | 110/140 | 389 | 389 | 33.2 | -8 |  | queued |
| 8 | ema-099 | 50/140 | 389 | 405 | 21.6 | 57 |  | queued |

_Δelo/hr = (peak elo − 389 floor) ÷ wall-hours-to-peak: real-strength gain per wall-clock hour, the north-star. Beat-heuristic ✓ = peak ≥ 800._
