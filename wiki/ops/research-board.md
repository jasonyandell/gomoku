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
- **Δelo/hour hill-climb priority** (Jason 2026-05-24: "never-run, then delta
  elo/hour — hill climb elo"). Order: **(1) never-run / entry-fee first** — an
  idea needs **2 elo points** to have a Δelo/hr slope, so round-0 then round-1 run
  for every idea (fewest points first); **(2) then highest Δelo/HOUR** over the
  most recent chunk — compute follows the *steepest recent climb*, not the highest
  absolute elo. Everyone caps; the steepest climbers get there first.
  - *History:* v1 ranked by last-chunk *raw Δelo* and pathologically fed the
    *worst* idea (a floored idea at Δ0 outranked a strong idea whose chunk dipped).
    v2 patched that with current-elo *level* (but that over-feeds an already-peaked
    champion and starves a faster challenger). The Δelo/**rate** rule is neither
    pathology: a floored idea sits at 0/hr and any genuine climber outranks it, and
    ranking the *rate* (not the level) is the literal hill-climb. The round-0/1
    entry fee avoids the floor-noise artifact (at the floor all ideas are ~equal, so
    a 1-point "rate" is meaningless).
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
   > ⚠ **CORRECTED 2026-05-24 (Jason): this v1 reading is stale + was measured single-process.** It was taken on the SINGLE-PROCESS v1 derby (GPU ~30%, busted wall-clock), so the gen/train ratio itself is suspect. More importantly, after the perf wins (fp16-eval, V=512, native MCTS) **the regime FLIPPED: generation now OUTPACES the trainer — it FLOODS it.** Per [perf-bench-vs-real-training-cost.md](../topics/perf-bench-vs-real-training-cost.md): "maximizing generation throughput floods the trainer"; the LF1 runaway (per-epoch 20s→7min) is the generator producing positions faster than the trainer can use them, with `sgd_per_position` blowing up trying to consume the flood. **"Generation-bound" is NOT a standing truth — it's recipe-dependent (high-sims wave leans slower-gen; optimized fp16/high-V floods).** The v3 cards below that invoke "generation-bound" should be read through this correction. The fix is a fixed per-epoch SGD cap decoupled from inflow → the `derby-gumbel-fast5s` lane.
6. **Method fixes (mid-run):** priority must rank by *current elo*, not last-chunk-Δelo (the latter fed the *worst* idea); peak checkpoints were lost to `keep-last-n=3`.

## v2 — what's next (queued)

Re-run the **top 3** (`open-div4`, `temp-16`, `sgd-800`) **HEAD-TO-HEAD**, using the **production multiprocess recipe** (`run_sweep` wave-mode, 8 `selfplay_worker`s — saturates the GPU, so wall-clock is REAL). Eval = round-robin *direct matches* among the 3 (they're all >1280, so they'd saturate the anchor ladder — head-to-head via `delta_e_harness --head-to-head` is the correct eval for strong models). **Wall-native budget** (hours, not epochs; chunk = wall-slice; allocate+stop by Δelo/hr) measured on the saturated machine so Δelo/hr is finally honest. Carry the fixes: current-elo priority, peak-checkpoint snapshotting. (Verified 2026-05-24: the production recipe IS multiprocess; the single-stream drift was only in the v1 derby harness.)

---

## v3 — UNIFIED prior-art race (LAUNCHED 2026-05-24, `scripts/derby_v3_board.json`)

A **unified board**: Jason called it — rather than run v2 (the top-3 head-to-head) to
cap and *then* a separate v3, we **ported the v2 carryover recipes into v3** and race
everything at once. v2 was stopped at round-0 (all at the 389 floor → zero data lost),
which freed the box for the native `.so` rebuild. The roster (9 ideas, one lever each
vs the `c0` control, fresh `--size small --seed 0`, scored by anchored elo then
head-to-head at the top): the **v1/v2 carryovers** (open-div4, temp-16, sgd-800) +
the **4 prior-art levers** (playoutcap, forced, swa, gumbel) + a **sims100 control**
for gumbel. Each picked to attack a **specific v1 finding** — v1 told us we're
**generation-bound** and **exploration/diversity beats raw compute for the ceiling**,
so the new levers are biased toward *better targets per unit of generation*. All ran
through the lab's two-queue fan-out (5 worktree code lanes, opt-in flags, production
byte-identical when off, merged `--no-ff` serially with one native rebuild).

> Wall-fairness resolved: **gumbel + forced-playouts both run in the native C engine**
> (`_mcts_native.c`, rebuilt). Gumbel's first cut came back python-only (~5× slow =
> DOA per Jason); the **native C port** (per-game Sequential Halving inside the wave)
> made it **0.86–1.26× native PUCT** — wall-fair, raced wall-matched like the rest.
> (Gumbel + sims100 run at sims=100: gumbel's value-prop is good targets at *cheap*
> sims; sims100 is the plain-MCTS control that isolates whether gumbel rescues them.)
> **aux-head** (opponent-reply, Class-C model-arch) is built + verified but parked on
> its **own axis/board** — not in this search/recipe race.

### v3-gumbel  (HIGHEST leverage)
**Lever:** `--gumbel-root` (+ `--gumbel-m 16`, `--gumbel-c-visit 50`, `--gumbel-c-scale 1`) — Gumbel-top-k root sampling + Sequential Halving, completed-Q policy target. **Source:** Gumbel AlphaZero/MuZero, Danihelka et al. (DeepMind, 2022).

**Hypothesis:** Directly attacks v1's #1 finding (generation-bound) and its sharpest failure (`sims-100` "never grokked" under vanilla MCTS). Gumbel *provably* improves the policy even at tiny sim budgets (n=2..16) — so it should let self-play run far fewer sims per move yet still emit strong, low-variance targets, buying generation speed without the target-quality collapse that floored sims-100. The risk: the completed-Q target is a different target shape than visit-count policy; it could interact badly with our short-game distribution or need m/c-tuning to beat plain PUCT at our sim counts.

**Expected Δelo signature:** *Confirm* = at LOW sims (e.g. 100), Gumbel's Δelo/hr clears C0's and clears vanilla-MCTS-at-100 by a wide margin — strong targets cheap = the generation-bound win. *Refute* = Δelo ≈ vanilla PUCT at matched sims (the completed-Q target bought nothing at 9×9 scale) or instability from the target-shape change.

**Config delta vs C0:** `--n-simulations 100 --gumbel-root` (the point is cheap-sims-that-still-train; an A/B at 200 is a secondary cell).

### v3-playoutcap
**Lever:** `--playout-cap-frac 0.25 --playout-cap-fast-sims 50` — most moves run a small budget and are NOT recorded; ~25% run the full budget and ARE the training targets. **Source:** KataGo, Wu (2019); inherited by KataGomo (the engine we surveyed @ Gomocup 2254).

**Hypothesis:** Concentrate expensive search where it actually trains the net. Generation-bound says wall-clock is dominated by sims/move; spending the full budget on only ¼ of moves (and a cheap budget elsewhere just to keep the game progressing) should multiply games/wall at near-constant target quality — a different route to the same "more, fresher self-play per wall" that cheap-sims chases, but without weakening the *recorded* targets. The risk: the cheap moves still shape the game trajectory, so low-quality intermediate play could bias the distribution the recorded positions come from.

**Expected Δelo signature:** *Confirm* = Δelo/hr above C0 — same-or-better climb at materially less wall, because the recorded targets stay full-strength while the game advances cheaply. *Refute* = a shallower climb (cheap intermediate moves degraded the trajectory the targets are drawn from) or no wall win (the fast moves weren't cheap enough to matter).

**Config delta vs C0:** `--playout-cap-frac 0.25 --playout-cap-fast-sims 50` (full budget stays C0's `--n-simulations 200`).

### v3-forcedplayout
**Lever:** `--forced-playout-k 2.0` — force ≥ `ceil(sqrt(k·P(a)·N))` visits to each root child, then prune the forced visits back out of the policy target. **Source:** KataGo, Wu (2019).

**Hypothesis:** v1 found exploration beats compute for the ceiling, but "more sims" is the expensive way to explore. Forced playouts buy *root exploration* of promising-by-prior moves that PUCT would starve — without raising the sim budget — and target-pruning keeps the forced visits from polluting the trained policy. So: the ceiling benefit of exploration at the cost profile of a normal sim budget. The risk: at our small sim counts the forced minimums could *crowd out* the search's own signal, and the pruning rule could over/under-correct the target.

**Expected Δelo signature:** *Confirm* = a higher ceiling than C0 (echoing open-div4/temp-16's exploration win) at C0's wall cost — exploration without the sim tax. *Refute* = Δelo ≈ C0 (forcing didn't add useful exploration at 9×9) or instability from target-pruning artifacts.

**Config delta vs C0:** `--n-simulations 200 --forced-playout-k 2.0`.

### v3-swa
**Lever:** `--swa-window K` — publish self-play generator weights as the flat average of the last K checkpoints, instead of EMA/live. **Source:** Stochastic Weight Averaging / Leela Chess Zero weight-averaging practice.

**Hypothesis:** v1's `ema-099` was a *floor* — the exponential moving average LAGGED the learner on a fast climb, generating from a staler/weaker policy. SWA is the targeted fix: a flat tail-average smooths target generation (the stability EMA was reaching for) *without* the unbounded lag of exponential decay, since old weights fall out of the window entirely. So it should recover EMA's stability benefit on the climb without the lag penalty that sank ema-099. The risk: on a *fast* climb even a short flat window still mixes in too-old weights and lags; or the smoothing simply isn't worth anything when fresh-start gradients are large and directionally consistent.

**Expected Δelo signature:** *Confirm* = lower chunk-to-chunk Δelo variance than C0 AND a climb that stays at-or-above C0 (beating ema-099's floor) — stability without lag. *Refute* = lags like ema-099 (window too wide / climb too fast) or no variance reduction (the live policy was already fine).

**Config delta vs C0:** `--n-simulations 200 --swa-window 5` (tune K; contrast directly with v1's `--ema-tau 0.99`).

### v3-auxhead (Class-C, design-only — see `auxiliary-targets-design.md`)
**Lever:** an opponent-reply auxiliary policy head (recommended), opt-in via an aux-loss weight; predicts the opponent's next-ply policy for extra gradient per position. **Source:** KataGo auxiliary targets, adapted to 9×9.

**Hypothesis:** Attacks the laptop-scale thin-signal problem ([[az-at-scale-vs-laptop]]): short gomoku games yield few near-opening positions, so each scarce position should teach the net more. An opponent-reply head squeezes a second supervised signal from data we already generate. *This is a model-architecture change (Class C)* — design first, user sign-off before any model.py edit. Card finalized from the design doc.

**Expected Δelo signature:** *Confirm* = steeper Δelo/hr than C0 at equal generation (more signal per position = faster learning from scarce data), aux head dropped at inference so self-play/eval cost is unchanged. *Refute* = aux loss distracts the shared tower (policy/value regress) or the extra signal is redundant with the value target on short games.

**Config delta vs C0:** TBD from the design doc (e.g. `--aux-opponent-reply-weight 0.15`, default 0.0 = off).

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
