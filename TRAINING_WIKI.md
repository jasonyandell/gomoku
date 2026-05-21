# Gomoku AlphaZero — Training Wiki

Live log of training runs, performance characteristics, experiments, and outcomes.
Append-only style — don't rewrite history, add a new entry at the bottom.

This file is the chronological training notebook inside the broader wiki. Start
from `wiki/index.md` for navigation and from `AGENTS.md` for the maintenance
schema. Use this file for experiment evidence, dated corrections, and changes to
the working theory; use topic pages under `wiki/topics/` when a reusable
synthesis deserves a stable home outside the long run log.

## Setup

- Board: **9×9** free-style gomoku (was 13×13 from 2026-05-17 23:46 through
  2026-05-18 16:53, when we rolled back to 9×9 for the AZ-mini recipe run).
  Original 9×9 small run is at epoch 136 in `checkpoints/`. 5-in-a-row, no
  opening restrictions.
- Hardware: M5 Max Mac, MPS backend
- Model sizes: tiny / small / medium / large (see `gomoku/model.py`)
- Default training: `--size small --games-per-epoch 64 --n-simulations 100 --training-steps 400 --batch-size 256`
- Replay buffer: 50k examples, ring buffer
- Augmentation: 8-fold D4 symmetry on each example (1 game ≈ plies*8 examples)
- **Network input: 17 planes** (`N_INPUT_PLANES = 2 * HISTORY_PLY + 1`,
  HISTORY_PLY=8). 8 history planes per side + 1 constant. Added 2026-05-18.
- wandb project: `gomoku`. Run IDs of interest:
  - `o9npssu1` — original 9×9 small, collapsed to defensive draws (epoch 136)
  - `k2wrbb8e` / `qx69005o` — 13×13 medium runs, all collapsed
  - `kze1lcti` — **current** AZ-mini 9×9 with history planes

## Architecture cheatsheet

```
train.py main loop (per epoch):
  1. generate_games(64) via batched MCTS    -- gen_s, dominates wall-clock
  2. add examples to ReplayBuffer
  3. 400 SGD steps from buffer              -- train_s, tiny on MPS
  4. every 5: eval vs random/heuristic/lookahead
  5. checkpoint to checkpoints/epochNNNN.pt
```

`run_batched_mcts` cross-batches leaf evaluations across all *active* games at
each simulation, so the model forward sees `B ≈ active_games` per sim. This is
the only place GPU saturation happens during gen.

## Performance baseline (epoch 55–104, prior run `o9npssu1`)

| Epoch | gen_s | train_s | plies | pl   | vl   | wr vs random |
|------:|------:|--------:|------:|-----:|-----:|:-------------|
| 55    | 25.3  | 3.1     | 20.6  | 1.12 | 0.27 | 100%         |
| 80    | 60.8  | 3.1     | 38.2  | 1.19 | 0.23 | 100%         |
| 100   | 75.7  | 3.1     | 50.7  | 0.74 | 0.08 | 100%         |
| 104   | 96.8  | 3.2     | 62.1  | 0.64 | 0.04 | —            |

Resume after kill+restart (epoch 101 of this session): gen=31.2s train=6.6s,
plies=39.3 — note plies bounce around per epoch due to MCTS noise.

### Key observation

`gen_s` scales roughly linearly with `plies` (games get longer as the model
improves). `train_s` is essentially constant (~3s) for a `small` model with
400 steps at batch 256 on MPS. **Gen dominates wall-clock by 25-30×.**

This is the primary optimization target.

## Experiments backlog

1. **Bigger games-per-epoch** — `--games-per-epoch 128 / 256`. Increases the
   per-sim leaf batch from ~64 to 128/256, better MPS utilization. Replay
   buffer is unchanged.
2. **Parallel self-play workers** — fork N processes that continuously generate
   games and ship examples to the trainer via a shared queue/files. Trainer
   pulls every N seconds and trains continuously. Decouples gen ↔ train.
3. **Overlap gen+train in one process** — start next gen while previous SGD is
   running. Risk: MPS contention between forward (gen) and forward+backward
   (train). Easy to try as a thread.
4. **More sims per move** — `--n-simulations 200`. Better data quality but ~2×
   gen cost. Probably wait until gen is faster.
5. **fp16 MCTS evaluator** — `make_torch_evaluator(..., fp16=True)`. Already
   wired but unused in train loop; check accuracy impact.
6. **Vectorize/numbafy MCTS tree ops** — `_select_action`, `_backprop` are
   numpy-on-81-element-arrays per sim. Probably <10% of total but worth
   profiling if gen-fork doesn't pan out.
7. **Larger model later** — once we can saturate MPS, jump to `medium` and see
   if loss continues dropping.

## Run history

| When (local)       | branch / commit | command                                       | resumed from | notes |
|--------------------|-----------------|-----------------------------------------------|:------------|:------|
| 2026-05-17 21:03   | main `f159212` | `--epochs 1000 --games-per-epoch 64 --wandb` | scratch      | reached epoch 104, killed for restart |
| 2026-05-17 22:37   | main `f159212` | `--resume latest.pt --epochs 1000 --wandb`    | epoch 100    | baseline-continuation run; established `gen ~30-115s, train ~3-6s` profile while we experiment |

## SUMMARY (2026-05-18, evening — distributed self-play workers)

After launching AZ-mini single-process, asked "are we doing everything we can
locally?" The answer was: **no — we're CPU-bound on Python MCTS tree ops, and
multi-process bypasses the GIL**.

### Bench that drove the decision

| games/epoch | gen time | ms/game |
|---:|---:|---:|
| 64 | 10.7 s | 167 |
| 128 | 21.8 s | 170 |
| 256 | 42.8 s | 167 |
| 512 | 81.7 s | 160 |

Per-game time is **flat** across all batch sizes — so going wider in one
process gives **zero throughput gain**. The GPU is sitting idle while Python
does tree work. Multi-process is the unlock.

### Worker scan (no trainer, fresh model, 4 ms/game baseline at sims=400)

| processes | aggregate g/s | per-worker | efficiency |
|---:|---:|---:|---:|
| 2 | 4.83 | 2.41 | — |
| 3 | 6.93 | 2.31 | 0.96× |
| 4 | 8.43 | 2.11 | 0.88× |

### File-handoff architecture (modeled on zeb)

- **Workers** (`gomoku/selfplay_worker.py`) generate game batches and write
  them atomically: `<id>_<ts>_<short>.pt.tmp` → `os.replace` → `.pt`. Poll
  the weights file's mtime each cycle and hot-reload when newer.
- **Trainer** (`gomoku/train.py` with `--worker-input-dir DIR
  --worker-weights-path PATH`) scans the dir by mtime, loads each `.pt`,
  appends records to the replay buffer, deletes the file. After every
  checkpoint, atomic-renames a fresh state_dict to the weights path so all
  workers see the new weights.
- Trainer's `--worker-min-games N` sets the per-cycle ingest target so it
  blocks until enough fresh data arrives before SGD.

### Live throughput

| | epoch wall | gen | train |
|---|---:|---:|---:|
| single-process (sims=800) | ~85 s | 70-90 s | 2-3 s |
| trainer + 4 workers (sims=800) | **~15 s** | 5 s (just file ingest) | 6-8 s |

**~5–6× speedup**, and now we're SGD-bound rather than gen-bound. With 4
workers feeding it, the trainer can keep up with fresh data.

Implementation cost: ~250 lines of new code in `selfplay_worker.py` plus
~70 lines of file-ingest / weights-publish plumbing in `train.py`.

### What we haven't done yet (deferred)

- Multi-machine scale-out (vast.ai with CUDA). Local Mac at 4 workers is
  near saturation; for more throughput we'd need actual additional GPUs.
- Training-side parallelism / async SGD. Trainer is now the bottleneck
  but it's already at ~1 step / 100 ms with batch=512 on medium — squeezing
  more out would mean larger batches, batched MCTS-feedback, or distillation.
- **The recipe is still imperfect**: plies dropped to ~15 again in the
  distributed run too, so the underlying defense-blindness collapse is
  independent of throughput. Real recipe fixes (more sims, larger model,
  longer history, etc.) are the next research lever, but throughput now
  lets us iterate them faster.

## SUMMARY (2026-05-18, late afternoon — AZ-mini run launched)

### What's running now

`wandb id kze1lcti` — `9x9-az-mini-history-800sims` on `checkpoints_az_mini_9x9/`.

This is a serious attempt to reproduce the AlphaZero recipe in miniature. The
hypothesis: every previous collapse was caused not by one bug but by a
**confluence of under-doing four AZ choices at once**, none of which alone
explains the failure but together prevent MCTS from discovering defense.

Config:

| knob | value | rationale |
|---|---|---|
| board | 9×9 | cheap, easy to throw away |
| input | **17 planes** (8 history per side + 1 const) | AZ scheme; lets the net see threats forming |
| size | medium (~1.04M params at 9×9, 17 input planes) | AZ's smallest Go experiment was 20×256 ResNet ≈ 6-12M; we're 10× under but want to see if recipe alone helps |
| sims | **800 per move** | AZ's actual number; up from our previous 200 |
| games/epoch | 64 | gen ≈ 93 s/epoch at this setting |
| training-steps | 64 | **1:1 train:gen ratio** matching AZ |
| batch-size | 512 | bigger sample diversity per gradient step |
| replay-buffer | 500k positions | 10× our older run; still ~300× under AZ but the best we'll do |
| **α** | **0.13** | scaled from AZ's 0.03 × (361 ÷ 81), should actually spread noise across the action space instead of concentrating |
| ε (dirichlet) | 0.25 | AZ default |
| temperature_moves | 10 | half of typical game length on 9×9 |
| opponent | self | no curriculum, pure AZ |

### The key refactor: history planes

`GameState` now carries a `history` tuple of past canonical boards (most-recent
first, up to `HISTORY_PLY - 1` = 7 entries). On each `apply()`, the pre-flip
board is snapshotted onto the head of history.

`to_planes()` emits 17 planes:

```
plane 0   : my stones at t                  (= side-to-move's stones, current)
plane 1..7: my stones at t-1, t-2, ..., t-7
plane 8   : opp stones at t
plane 9..15: opp stones at t-1, ..., t-7
plane 16  : constant 1.0 (side-indicator slot)
```

The subtlety: "my stones at t-k" depends on parity. Each ply the canonical
perspective flips, so `history[k-1]` was stored from the perspective of
whichever side moved k plies ago. If k is even that side's color matches
"my" now (we read `past_board[0]` into the "my" plane); if k is odd we read
`past_board[1]` instead. Encoded inline in `to_planes()`.

Augmentation (D4 symmetries) just transforms the spatial dims, so the
existing `_sym_board` function works on the new shape with no changes.

Model first conv input channels picks up automatically: `ModelConfig`'s
`n_input_planes` default is now `N_INPUT_PLANES` (the constant from
`game.py`).

### Why we expect this to behave differently

The previous collapses all had the same shape: model develops a sharp
attacking policy → MCTS at 200 sims confirms the prior → defensive moves
get 1-2 visits → policy targets stay attack-shaped → value head trivially
predicts the outcome → no defense gradient. Self-play opponents are equally
clueless so threats never form. We hacked around this with mixed-random
opponents and pacifist blockers, but those only moved the trivial-value
collapse from z=-1 (vs heuristic) to z=0 (vs pacifist), neither giving
useful policy signal.

The AZ recipe addresses every failure point simultaneously:

1. **800 sims** gives MCTS the budget to actually expand defensive subtrees
   and find that "if I block here the simulated game continues; if I attack
   here I lose in 3" — defense gets visit-count signal not just from prior
2. **History planes** let the network see "opponent just played their 3rd
   stone in this row" — threat patterns are visible directly, not inferred
3. **α=0.13 (small)** means root Dirichlet noise actually picks a few
   non-prior actions per game to explore, instead of slightly perturbing
   everything in the prior's direction
4. **Pure self-play** with all of the above means each side genuinely
   stumbles into threats and onto defenses; the data buffer gets a
   diverse mix of attack-only, defense-only, and mixed positions

Nothing prevents this from collapsing in a *new* way, but it removes the
specific mechanism that caused the previous collapses.

## SUMMARY (2026-05-18, mid-afternoon)

Everything performance-related is in great shape. The training *dynamics* on
13×13 medium have been the hard problem all afternoon.

### What works (all checked in)

- `--wave-size N` (zeb wave-batched MCTS with virtual loss) — **1.51× gen speedup** at 32
- `--opponent <spec>` (self / random / heuristic / lookahead:depth=N / defensive)
- `--opponent-mix-random P` — wrap opponent so it plays random with prob P
- `--random-opening-moves N` — each game starts with N uniform random moves before MCTS
- `defensive_player` — pure-defense opponent (blocks, never attacks)
- Vectorized `_has_five_in_a_row` (2× faster)
- Combined `.cpu()` in evaluator (2× faster microbench)
- `--games-per-epoch 128` default
- Tightened eval (random,heuristic only, 4 games)

### What the model can do

`checkpoints_13x13_medium_az/az_peak_e50.pt` — wandb run `qx69005o` at epoch 50.

| opponent | model record over 200 games |
|---|---|
| random | crushes 100% |
| 90% random + 10% heuristic | 76% W / 24% L / 0 D |
| 50% random + 50% heuristic | **7% W / 93% L** |
| 100% heuristic | ~0% W |
| pure defensive (blocker) | 0% W / 100% L (defender stumbles into 5s) |
| pure self-play | 64% black wins, 36% white wins, 0 D |

It plays **pure offense, zero defense.** It never blocks, because in training it
never faced an opponent who built threats consistently. When the heuristic plays
even part-time, the model loses because it spends every move attacking elsewhere.

### Why it's stuck (the actual diagnosis)

A replay of one model-vs-heuristic eval (model = black, heuristic = white):

```
MODEL    : g8, h7, g6, i8, f5    (scattered attack)
HEURISTIC: f7, f9, f8, f6, f10   (vertical 5 on column f — wins ply 10)
```

By the heuristic's 4th move (open-4 at f6-f9), the game is lost — model needed to
block at move 4 (its 4th MCTS pick) and instead played `i8` (attack). The model's
policy has no concept of "block when opponent has open 3-in-a-row."

This collapse compounds itself: in self-play, both sides attack, nobody builds
threats consistently, so defense gradient never appears. With opponent-mix-random
0.9, the heuristic acts on 10% of moves — its threats get broken up by random
play before they mature, so the model still doesn't see "open three I should
block" positions often enough to learn from them. With mix-random 0.5, the
heuristic acts on half the moves and **completes threats faster than the model
can attack**, so model loses 93% and value head trivially collapses to z=-1.

We are stuck in a curriculum gap: easy opponents don't teach defense; hard
opponents don't let the model win.

### Promising next ideas (not implemented — over to you)

1. **Blocker that explicitly avoids forming its own 5**. Implemented as
   `pacifist_blocker` (`gomoku/baselines.py`). Among legal moves, it skips any
   placement whose resulting own-line through that cell would have ≥ 4 stones
   (so it can never build an open four or 5). Within the remaining moves it
   picks the strongest defensive block. Exposed as `--opponent pacifist`
   (with optional `--opponent pacifist:max_own_line=N`). **Now under
   evaluation as a training opponent — see the live notes below.**
2. **Population-style opponent pool**: mix model-vs-self with model-vs-past-checkpoint.
   Older checkpoints play differently → diverse threats; current model wins more
   often because the past one is weaker. Implementation: pick from a list of
   checkpoint paths, weighted by recency, each game.
3. **Defense head**: add an auxiliary "is opponent threatening to make 5?"
   binary head to the network. Forces the trunk to develop threat-recognition
   features even if the policy ignores them.
4. **Curriculum on mix-random**, manual: start with mix=0.9, ramp toward 0.5
   gradually (epoch 0-50: 0.9; epoch 50-150: 0.7; epoch 150+: 0.5). The
   ramp could be conditional on wr_random staying ≥ 80%.

### Checkpoint inventory

All older checkpoints stay on disk for reference but **none are loadable by
the current code** because we changed:

- BOARD_SIZE from 13 → 9 (some dirs) / from 9 → 9 with history planes (others)
- Input planes from 3 → 17

The model architecture in those checkpoints is therefore incompatible with the
current `gomoku.model`. To re-use any of them you'd need to (a) load the cfg
from the saved file (which still works) and rebuild the model with the OLD
config, and (b) pass the OLD `to_planes` shape. Keep them as artifacts.

| dir | purpose | last epoch |
|---|---|---|
| `checkpoints/` | original 9×9 small, 3-plane input — collapsed to draws | 136 |
| `checkpoints_13x13_medium/` | first 13×13 medium, 3-plane — collapsed to 5-move openings | 1249 |
| `checkpoints_13x13_medium_az/` | 13×13 AZ-aligned, 3-plane; `az_peak_e50.pt` was the best 13×13 we made | 146 |
| `checkpoints_13x13_medium_vsheur/` | 13×13 mostly-loss probe vs heuristic | 16 |
| `checkpoints_13x13_medium_vsmix/` | 13×13 vs 90%rand+10%heur | 18 |
| `checkpoints_13x13_mix50/` / `_mix70/` | 13×13 curriculum attempts | ~58 / ~56 |
| `checkpoints_13x13_selfplay_rop/` / `_rop12/` | 13×13 random-opening attempts | ~58 each |
| `checkpoints_13x13_vs_pacifist/` | 13×13 vs pacifist_blocker | 73 |
| `checkpoints_az_mini_9x9/` | **current** — 9×9 medium, 17 input planes, AZ-faithful | — |

## Experiments — results

### Exp 1: Microbenchmark — `generate_games` throughput vs `games_per_epoch`

Setup: latest model (epoch ~115) on MPS, 100 sims/ply, **while main training ran**
(so MPS-contended; absolute numbers are inflated).

| games | gen_s | mean_plies | examples | ms/game | examples/s |
|------:|------:|-----------:|---------:|--------:|-----------:|
| 64    | 42.9  | 47.1       | 24,136   | 671     | 562        |
| 128   | 80.5  | 48.9       | 50,112   | 629     | 622        |
| 256   | 124.3 | 42.5       | 86,984   | 485     | 700        |

**Verdict**: per-game cost drops 28% from 64→256 batch. Throughput +24% overall.
The fixed Python/MCTS overhead per simulation amortizes better at larger batches.
**Action**: bump `games_per_epoch` default to 128 (modest +10% throughput, no plies/sample bias).

### Exp 2: Async gen+train pipeline (`gomoku/train_async.py`)

Setup: separate gen-worker thread holds its own model copy on MPS, runs
`generate_games(64)` in a loop. Trainer (main thread) runs 400 SGD steps per
iteration on the shared replay buffer. Weights pushed every iter. **Result: a loss.**

| | sync (epoch 113-116) | async (iter 117-118) |
|---|---|---|
| time per 64g + 400steps | 33-39 s | **45-48 s** |
| gen / train interaction | serialized: gen ~30s, train ~6.5s | **MPS-contended**: train ballooned 7× |
| games / s during training | ~1.6 | 1.41 |

Root cause: **a single MPS device is single-stream within a process.** Kernels
from the gen forward-pass and the trainer forward+backward serialize on the
device, and the train loop's 6.5s → 45s is the worst end of the trade. The
"free" overlap we were hoping for doesn't exist on MPS — the GPU is the limiter,
not Python or CPU, and we get strict timesharing.

**Verdict**: reverted. Async pipelining helps when one workload is CPU/IO-bound
and the other is GPU-bound. Here both are GPU-bound on the same device, so
overlap costs more than it saves.

**Kept**: `gomoku/train_async.py` for reference and for the case we ever move
to multi-GPU (CUDA cluster, separate machines). Not the default.

### Exp 3: cProfile of `generate_games(64, sims=100)` (epoch 118 model)

Total 38.7 s wall. Hot paths (cumulative %):

| function | cum_s | % |
|---|---|---|
| `_select_one` (MCTS descent) | 18.4 | 47% |
| └ `_select_action` (PUCT inner loop, 2.13M calls) | 13.6 | 35% |
| `evaluate` (MPS forward) | 18.3 | 47% |
| └ `.cpu()` (MPS→host sync) | 8.4 | 22% |
| └ model forward (conv+bn+linear) | 7.2 | 19% |
| `_init_node` (terminal check + legal mask) | 2.8 | 7% |
| `_backprop` | 0.8 | 2% |

The two big buckets: Python tree-ops (~35%) and MPS forward+sync (~47%, with 22%
just waiting on `.cpu()`). Wins are gated by these two.

**Action ideas** (in rough leverage order):
1. Vectorize `_select_action` across all games at a sim — fold 64 calls into 1
   `np.argmax((G, 81))`. Hard refactor.
2. Skip `.cpu()` round-trip during pure MCTS — keep priors as MPS tensor,
   only `.cpu()` when feeding back to Python tree ops. Possibly small.
3. numba/cython `_select_action`. Quick win if it works.
4. fp16 forward: **already tested — 17% slower on MPS** (Exp 4 below). Skip.

### Exp 4: fp16 evaluator on MPS

| dtype | gen_s for 64 games, 100 sims |
|---|---|
| fp32 | 41.5 s |
| fp16 | 49.9 s (**slower**) |

MPS' fp16 small-batch kernels appear less optimized than fp32 here. Don't use.

### Eval budget tightening (user-requested 2026-05-17 22:58)

Baseline eval (3 fast + slow rotating) was costing ~70 s every 5 epochs, so an
eval-epoch was ~140 s vs ~75 s for a non-eval epoch. User asked for a 10 s
budget. New eval config (from epoch 122):

- `--eval-baselines random,heuristic` (drop lookahead2)
- `--eval-baseline-games 4` (down from 16)
- `--eval-baselines-slow ""` (disable lookahead4 entirely)
- `--eval-sims 50` (unchanged)

**Verified at epoch 125: eval = 5.2 s.** Fits comfortably under the 10 s budget.

Side effect: with only n=4 games per matchup the win-rate signal is noisy
(one game flips a 25% → 50% reading). The user is fine treating these as a
curiosity, so leaving as is. If we ever want a more reliable signal we can
either bump to n=8 (~10 s eval, edge of budget) or sample less often (every 10
or 20 epochs at n=8).

### Plies growth → gen-time growth (epoch 122–125 in `sync-gpe128-fasteval`)

| epoch | plies | gen_s | s/ply |
|------:|------:|------:|------:|
| 122   | 34.5  | 48.5  | 1.41 |
| 123   | 36.5  | 51.4  | 1.41 |
| 124   | 47.0  | 75.3  | 1.60 |
| 125   | 52.8  | 102.5 | 1.94 |

Gen time grows super-linearly with mean plies — the per-ply cost itself grows
~38% from 34→52 plies, presumably because the MCTS trees are deeper and tree
ops cost more per sim. This is the next ceiling. Big wins would come from
either:

- vectorizing tree descent across the game-axis (Exp 3 action #1), or
- cutting `--n-simulations` (Exp 5 below).

### Exp 5: `n_simulations` 50 vs 100 — wall time vs data quality

Microbench, 64 games on epoch-118 model, **contended** with active training:

| sims | gen_s | mean plies | examples (8× aug) | ex/s |
|-----:|------:|-----------:|------------------:|-----:|
| 50   | 14.5  | 27.2       | 13,926            | 962  |
| 100  | 42.5  | 42.3       | 21,658            | 510  |

`sims=50` ~2.9× faster but games are 35% shorter → only 1.9× more examples per
wall-second. The shorter games are concerning: low-sim play produces weaker
moves that lead to faster blunders/wins, so the policy targets are noisier and
games are less informative per move. **Won't change the default** until we can
A/B the actual learning curve, but flagging as the most likely next win once
gen time becomes the binding constraint again.

### Exp 7: vectorize `_has_five_in_a_row` (deployed)

Profile said `is_terminal` was 2.6 s of 38.7 s gen (6.7%) at 155k calls/epoch.
The Python double-loop was 14.7 us/call; the vectorized version (4 directional
shifted ANDs of bool views) is 7.6 us/call — 2× faster on 9×9 boards.

| variant | us/call |
|---|---|
| Python loops | 14.7 |
| Vectorized | 7.6 |

Correctness: 203/203 boards agree (random + crafted cases). Full pytest suite
passes. **Deployed to main and worktree.** Expected saving: ~1.1 s/epoch.

### Exp 6: combine `.cpu()` calls in evaluator (deployed)

**Microbench (in isolation, B=128, N=81):**
- old `(logits.cpu(), values.cpu())`: 0.33 ms/call
- new `torch.cat([logits.flatten, values]).cpu()`: 0.17 ms/call (**2× faster**)

At 7030 evaluator calls / epoch, saves ~1.1 s/epoch under microbench conditions.
Real-world saving depends on MPS contention (training-step kernels overlap).


`make_torch_evaluator` was doing two `.cpu().numpy()` round-trips per call
(`logits` and `values`). Each `.cpu()` on MPS forces a stream sync; the data
is tiny (~20 KB) so syncs dominate over bandwidth. Profile showed `.cpu()` was
22% of generate-games time across 14060 calls.

Change applied in worktree `gomoku/mcts.py`: pack `[logits.flatten(), values]`
into one 1-D tensor, call `.cpu().numpy()` once, then split. Expected to save
~half of the 8.4 s `.cpu()` cost = ~4 s per 38 s gen = ~10% on the current
training profile.

**Deployed.** Smoke-tested for correctness (right shapes, values match an
unbatched reference). End-to-end speedup will be visible on the new
`sync-gpe128-fasteval-opts-e128` run vs the previous segment.

### Exp 8: zeb-style wave-batched MCTS with virtual loss (deployed)

Ported from `~/code/mk5-main/forge/zeb/{mcts.py,batched_mcts.py}`. Each round
collects `wave_size` leaves per game via virtual loss (`parent.N[a] += 1` along
the selection path), batches all `G * wave_size` leaves into ONE evaluator call,
then backprops with `_backprop_value_only` (W only — N already incremented).

Per-move evaluator calls drop from `n_simulations` to `n_simulations / wave_size`.
On MPS this is huge because per-`.cpu()` sync dominates over bandwidth.

Correctness: with `wave_size=1`, output `(N, W)` matches the original
`run_batched_mcts` byte-for-byte (8/8 games checked under fixed seed +
random evaluator). For `wave_size > 1` the trees diverge slightly within a wave
due to stale W, but plies and outcomes look normal.

**Clean isolated bench (32 games, sims=100, epoch-133 model):**

| wave | gen_s | plies | examples | ex/s | speedup vs wave=1 |
|---:|---:|---:|---:|---:|---:|
| 1  | 56.8 | 81.0 | 20736 | 365 | — |
| 4  | 48.1 | 81.0 | 20736 | 431 | 1.18× |
| 8  | 44.7 | 81.0 | 20736 | 463 | 1.27× |
| 16 | 41.5 | 79.1 | 20256 | 488 | 1.37× |
| 32 | 37.6 | 81.0 | 20736 | 551 | **1.51×** |

**Deployed:** main `mcts.py`, `self_play.py`, `train.py`. New CLI flag
`--wave-size` (default 1 for back-compat). Live training restarted at
epoch 134 with `--wave-size 16` (sweet spot: 37% speedup, half the staleness
of wave=32).

### FLAG: model has been regressing since ~epoch 105

| epoch | wr vs heuristic (n=4) | mean plies | training pl |
|--:|--:|--:|--:|
| 105 | 50% | — | 0.76 |
| 115 | 12% | — | 0.77 |
| 125 | 25% | 52.8 | 0.51 |
| 130 | 12% | 69.8 | 0.34 |
| 133 (no eval) | — | 80.4 | 0.20 |

Plies hit the 81-ply cap at epoch 133. The model has collapsed to a
defensive standoff: both sides know enough to never lose, and policy_loss
keeps dropping because the network is just memorizing the draw equilibrium.

This is a *training-quality* regression that perf opts can't fix. Possible
remedies (need user input):

1. **Roll back** to epoch ~100–105 (heuristic≈50%, plies≈50) and restart
   with sharper exploration (lower dirichlet eps, more sims) so the model
   doesn't re-collapse.
2. **Mix in weaker opponents** during self-play: a fraction of games where
   one side picks randomly or from an older checkpoint, to keep the
   distribution of decisive games up.
3. **Lower `--c-puct`** so MCTS exploits rather than explores — might
   break the draw equilibrium.
4. **Cap `--max-plies` lower** (e.g., 60) — just speed mitigation; doesn't
   fix the standoff.

### Pragmatic optimization backlog (in rough leverage order)

1. **Vectorize MCTS descent across games** (was the biggest single win;
   now partially captured by Exp 8). _select_action takes 35% of gen time
   as ~2.1M tiny numpy calls. Folding them into one `(G, 81)` argmax per
   sim-step could cut another 5-8 s/epoch. Risk: nontrivial refactor of
   tree traversal and per-game state tracking.

2. **`n_simulations` from 100 → 50** (Exp 5). 1.9× example/s but shorter games.
   Try in a separate experiment branch with own checkpoint dir.

3. **Apply Exp 6 (combined .cpu) on next restart** — already coded, just needs
   a clean measurement, and merge to main.

4. **`_has_five_in_a_row` via numpy strides / scipy.signal.correlate2d**.
   Currently 2.6 s/epoch (~7%) in Python loops; should be a 3-5× speedup,
   so ~2 s/epoch saved.

5. **Larger `games_per_epoch` (256)** — microbenched 24% throughput vs B=64
   under contention; need clean A/B. Trade: ~2× wall time per epoch but more
   stale weights per game.

6. **fp16 evaluator** — already tested (Exp 4) and slower on MPS. Skip.

Wins 3–4 are small but additive (~15-20% total). Wins 1 and 2 are bigger but
riskier.

## Run history (continued)

| When (local)       | branch / commit | command                                       | resumed from | notes |
|--------------------|-----------------|-----------------------------------------------|:------------|:------|
| 2026-05-17 22:52   | worktree        | `train_async --games-per-batch 64 --steps-per-iter 400` | epoch 116    | 2 iter then killed: MPS thread contention, 45s/iter |
| 2026-05-17 22:55   | main `f159212` | `gomoku-train --games-per-epoch 128` | epoch 118    | epoch 119: 65.9 s gen / 6.8 s train; epoch 120 (with old eval): 143 s; killed at epoch 121 to tighten eval |
| 2026-05-17 22:59   | main `f159212` | `--gpe 128 --eval-baselines random,heuristic --eval-baseline-games 4 --eval-baselines-slow ""` | epoch 121 | eval verified at 5.2s; plies grew 35→58 so gen grew 48→123 s; killed at epoch 128 to deploy two opts |
| 2026-05-17 23:13   | main + 2 worktree opts | same as above plus Exp 6 (combined `.cpu()`) and Exp 7 (vectorized terminal check) | epoch 128 | epoch 134: gen 165.8 s; epoch 135 with eval: 184.3 s |
| 2026-05-17 23:34   | main + waves | `--wave-size 16 --games-per-epoch 128` | epoch 133 (9×9 small) | epoch 134/135/136 gen ≈ 166/172/179 s vs prior 220 s. **22 % wall-clock speedup** sustained. Killed at epoch 136 to switch boards. |
| 2026-05-17 23:46   | main 13×13 medium | `--size medium --games-per-epoch 128 --wave-size 32 --checkpoint-dir checkpoints_13x13_medium` | scratch | board widened to 13×13, ~1.08 M params. Wave=32 delivered 0.29 s/game. Model collapsed by epoch ~30: pl≈2.5, plies≈10, vl≈0, heuristic=0%. 3 transient breakouts (epochs 333, 526, 600) all reverted. Killed at epoch 1249 (~8.5 hr, zero net eval progress). |
| 2026-05-18 07:26   | same code | resume + `--dirichlet-eps 0.5 --temperature-moves 20 --lr 5e-4` | epoch 1249 (13×13 medium) | exploration knobs failed to break the attractor across epochs 1250–1758. `temperature-moves=20` was a no-op (games end at move 9). Killed at user request. |
| 2026-05-18 11:27   | main + vs-baseline | `--size medium --opponent heuristic --wave-size 32 --epochs 30 --checkpoint-dir checkpoints_13x13_medium_vsheur` | scratch | model lost 100% to heuristic, vl trivially 0 (always-losing collapse). killed at e16. |
| 2026-05-18 11:33   | main + opp-mix-random | same plus `--opponent-mix-random 0.9` | scratch | huge improvement vs probe 1: pl 5.0→2.0 in 10 epochs, vl=0.25, plies 43-50, wr_random=100%. Killed at e18 to align with AZ recipe. |
| 2026-05-18 11:45   | AZ-aligned | `--training-steps 100 --replay-buffer-size 500000 --batch-size 512 --n-simulations 200 --opponent heuristic --opponent-mix-random 0.9` | scratch | wandb id `qx69005o`. Peaked at e50 with pl=1.48. From e50→e146 the model **regressed**: pl 1.48→2.45, plies 39→20, model specialized on "kill random opponent fast." heuristic baseline 0% throughout. Killed; saved `az_peak_e50.pt`. |
| 2026-05-18 13:09   | + curriculum | resume `e50` with `--opponent-mix-random 0.5` | epoch 50 (AZ run) | 200-game outcome probe: **model W=7 L=93 D=0**. mix=0.5 too hard — heuristic-half wins via threats the model can't block. Killed. |
| 2026-05-18 13:16   | + curriculum | resume `e50` with `--opponent-mix-random 0.7` | epoch 50 | pl stable around 1.5, vl rising to 0.44 (healthy), plies 30-34. Looked viable but killed at e6 to pivot. |
| 2026-05-18 13:26   | + random openings | resume `e50` with `--opponent self --random-opening-moves 6` | epoch 50 | same failure: pl 1.59→2.14 in 8 epochs, plies 34→27. Random openings give 13 distinct first-MCTS squares (was 2) but don't fix the *core* collapse — the model wins fast against any opponent because nobody forces it to defend. |
| 2026-05-18 13:36   | + bigger random | resume `e50` with `--opponent self --random-opening-moves 12` | epoch 50 | same failure pattern. Killed at e8. |
| 2026-05-18 16:11   | pacifist_blocker training | resume `e50` with `--opponent pacifist` (a blocker that refuses to grow own line past 3) | epoch 50 (13×13) | plies pegged at 169 (~all draws); vl trivially collapsed to **0.04** as model learned to predict z=0. pl bounced 1.65→2.06→1.58. Validates the diagnostic — model still can't break a real defense — but proves pacifist alone is no useful training partner. Killed at e73. |
| 2026-05-18 16:53   | **AZ-mini** | fresh 9×9 medium, 17-plane history, 800 sims, α=0.13, pure self-play, 1:1 train:gen | scratch | wandb id `kze1lcti`. Single-process; 5 restart segments. First heuristic non-zero at e75 (vl≈0.50, plies≈13). Held heuristic=50% (n=4 noise-bound) through e170 (vl≈0.32). Migrated to dist worker setup. |
| 2026-05-18 18:34   | **AZ-mini-dist** | same recipe + 4 workers + dynamic K (`--sgd-per-game 2.0`) | scratch | wandb id `oo53qzvf`. Live. ~7 s/epoch steady-state, 5–6× kze1lcti throughput. First heuristic non-zero at **e225** (vl=0.366); genuine repeated crossings start **e315** (vl=0.298, heuristic=25%). Slow climb in plies (13→21 by e360). e365 heuristic=75% (vl=0.250). See "Two data-mix slopes" below. |

## Two data-mix slopes

Same code path (medium 9×9, 17-plane history, 800 sims, α=0.13, pure
self-play), two very different routes through `(vl, heuristic-wr)` space:

| run         | infra                                              | first heuristic ≥ 25% | vl at crossing | plies at crossing |
|-------------|----------------------------------------------------|----------------------:|---------------:|------------------:|
| `kze1lcti`  | single-process, 64 g/epoch, ~1:1 train:gen         | e75   | **0.501** | 13.2 |
| `oo53qzvf`  | 4 workers + K=2 SGD-per-game, 32 g/cycle           | e315  | **0.298** | 14.0 |

The dist run sat **below the prior run's "breakthrough vl"** for ~100+
epochs before heuristic moved. So `vl` is not predictive of strength
across runs — buffer character is. The likely differences:

- **Train:gen ratio.** kze1lcti = K=1 (64 SGD steps per 64 games via
  `--training-steps 64` static). oo53qzvf is K=2 SGD-per-game by design
  (64 steps for 32 games). The dist run actually trains *harder* per game
  seen, not softer — opposite of what I had originally written here. The
  real difference is staleness, not gradient density.

  **Correction (2026-05-18 23:00):** an earlier version of this entry
  claimed kze was K≈6 ("400 steps / 64 games"). That was wrong — kze's
  wandb-captured args show `--training-steps 64`. K=1 for kze, K=2 for
  fresh-dist.
- **Opponent freshness.** Single-process generates with up-to-the-second
  weights. Workers run lagged weights between mtime polls + last ~30s of
  generation, so the buffer's "opponent strength" stratifies more.
- **Game-mix entropy.** Workers add gen-time RNG diversity (4 independent
  seeds rolling forward) vs single-process's one RNG.

Validates Jason's intuition: *the slope of opponent quality across the
buffer's time-axis matters more than the snapshot loss*. The dist run is
genuinely learning slower in epochs but apparently building cleaner
generalization — its heuristic curve is climbing on real progress (plies
growing, not on a momentary value collapse).

## Head-to-head: the only signal that doesn't lie

After both runs were stopped, we ran `fresh-e552` vs `kze-e176` directly,
n=40 with 4 random opening plies for diversity:

| matchup                       | result        | winrate |
|-------------------------------|---------------|--------:|
| fresh-e552 vs kze-e176 (h2h)  | 39W-1L-0D     | **97.5%** |
| fresh-e552 vs heuristic (n=30)| 6W-15L-9D     | **35%** |
| kze-e176   vs heuristic (n=30)| 15W-15L-0D    | **50%** |

**Non-transitive.** `fresh` crushes `kze` but **loses worse to heuristic**.
fresh's 9 draws confirm it's grinding games (consistent with the plies=48
trend), while `kze` plays decisively (0 draws). Head-to-head between
siblings of the same lineage measures local optimization, not absolute
strength — never use it as the only metric.

The first-mover bias is real (free-style gomoku is forced black at perfect
play) but **not the dominant source of n=4 eval noise** at this skill
level: head-to-head black-side won 52% of decisive games.

## Sweep infrastructure (this session)

To stop guessing one knob at a time, built a multi-run-safe launcher and
split eval out of the training loop:

- `scripts/run_sweep.py` defines 6 cells across **K × buffer-size** axes.
  Each cell gets unique paths (`sweep_runs/<cell>/checkpoints/...`) so
  multiple can run in parallel without colliding.
- `gomoku.eval_worker` is a separate process. mtime-polls
  `worker_weights.pt`, runs vs random + heuristic + lookahead:depth=2 at
  n=20, logs to the same wandb run via the embedded run id. Trainer runs
  with `--no-eval`, so gen+train never blocks on baselines.
- `ReplayBuffer` slots now carry a `weight_version` int tag (8 bytes/slot,
  costs nothing). Auto-overwritten on ring reuse — no separate deletion
  needed. `shape_stats()` reports `buffer/age_mean`, `_p50`, `_p90`,
  `_frac_current`, so you can *see* old-brain vs current-brain mix evolve.
- Disk hygiene: `--orphan-sweep-age-sec` (default 5 min) wipes `.tmp` /
  unknown files in checkpoint+records dirs each save cycle. Prevents the
  1.3 TB disk-fill we hit earlier.

### Sweep matrix (cells)

|         | K=1 | K=2 | K=4 |
|---------|-----|-----|-----|
| buf=50k  | A   | B   | C   |
| buf=500k | D   | E   | F   |

Target: heuristic ≥ 50% sustained for 3 consecutive evals before epoch 100.

### Cell C result (2026-05-18 20:45, wandb sweep-C-K4-buf50k)

**Failed target.** K=4, buf=50k = highest-contrast first try (opposite end
of fresh-dist's K=2, buf=500k). Across 13 eval polls in 100 epochs,
heuristic = **0/20 every single time**. Final state:

| epoch | pl    | vl    | plies | buffer age (p50) | heuristic |
|------:|------:|------:|------:|-----------------:|----------:|
| 20    | 2.400 | 0.316 | 20.3  | 4                | 0%        |
| 40    | 2.028 | 0.239 | 15.7  | 7                | 0%        |
| 60    | 1.864 | 0.253 | 13.2  | 7                | 0%        |
| 80    | 1.804 | 0.231 | 12.3  | 7                | 0%        |
| 100   | 1.850 | 0.065 | 11.9  | 8                | 0%        |

Diagnosis: classic fast-attack-collapse, *accelerated* by the high K +
small buffer combo. vl=0.065 by e100 (took fresh-dist 500+ epochs to
reach 0.08); plies *shrunk* 20→12 across the run. More SGD-per-game
amplified self-bait specialization rather than learning generalization.
Suggests next cell should swing opposite: low K + big buffer (cell D).

### Cell D result (2026-05-18 21:15, wandb sweep-D-K1-buf500k)

**Failed target too.** K=1, buf=500k = opposite corner from C. Across 11
eval polls, heuristic = **0/20 every single time**, same as C.

| epoch | pl    | vl    | plies | buffer age (p50) | heuristic |
|------:|------:|------:|------:|-----------------:|----------:|
| 20    | 3.428 | 0.499 | 23.9  | 11               | 0%        |
| 41    | 3.215 | 0.504 | 16.6  | 24               | 0%        |
| 61    | 3.032 | 0.476 | 14.4  | 37               | 0%        |
| 83    | 2.912 | 0.439 | 10.5  | 53               | 0%        |
| 100   | 2.834 | 0.406 |  9.8  | 65               | 0%        |

Slower-burn version of C: same destination (plies 24→10), different
speed. Both K=1 and K=4 collapse identically along the K axis. Confirms
**neither K nor buffer-size is the upstream cause** of the failure.

## The slope insight (Jason, watching the buffer_size chart)

Looking at buffer_size across runs in wandb, the failing runs (D, fresh-
dist) produce visibly *concave* fill curves — slope decreasing over
epochs. A healthier "K2-shape" reference run shows a straight (constant-
derivative) line at roughly 2× the fill rate of D / fresh-dist by e51.

**Buffer-fill slope = games-per-cycle × average plies-per-game.** The
games-per-cycle term is fixed by config. The plies term is the variable.
When plies collapse from ~30 → ~12 (fast-attack mode), the slope drops
~2.5× and the buffer-fill curve bends downward.

**Concavity in buffer_size IS the rate of collapse**, visible in real
time. It's a leading indicator that beats waiting for heuristic eval to
confirm the model is stuck. A straight buffer_size line = stable game
length = no collapse. A bowing line = active collapse, every cycle.

The K × buffer-size axes don't touch plies-per-game. That's why we
spent six cells getting identical failures: we were tuning the wrong
plane.

Things that DO move the slope:
- **`--random-opening-moves N`** — forces a minimum-plies floor (model
  can't end before move N+something), keeps fresh-data injection
  rate constant.
- Higher `--dirichlet-eps`, longer `--temperature-moves` — exploration.
- `--opponent-mix-random P` — opponent diversity stretches games.
- Structural fixes (renju overline ban, opening books) — future work.

### Cell C ↔ Cell D head-to-head probe (after stops)

Quick aside: we ran fresh-e552 (oo53qzvf) vs kze-e176 (kze1lcti dist)
n=40 with 4 random opening plies. Result: fresh-e552 wins **39-1
(97.5%)**. But against the heuristic at n=30, fresh-e552 = 35% while
kze-e176 = 50%. Non-transitive in the strict gomoku-style sense: the
"newer, more refined" model crushes its sibling but loses worse to an
external attacker. Head-to-head between same-lineage checkpoints
measures local mutual specialization, not absolute strength. Vs a fixed
external opponent (heuristic, lookahead) is the only signal that
generalizes.

## Sweep cell E result (skipped — equivalent to fresh-dist)

Cell E = K=2, buf=500k. This is exactly what fresh-dist already ran for
552 epochs. Don't re-run; treat fresh-dist as cell E's evidence.

## kze-recipe-K1-open6 (launched then aborted — 2026-05-18 23:00)

Launched the slope-hypothesis variant (`--random-opening-moves 6` on
kze's recipe, single-process, K=1) but killed it before any milestones.
Pivoted to a cleaner narrative: first replicate kze straight, *then*
test the +openings variant once we know the baseline reproduces.

## 9x9-kze-recipe-K1-pure (launched then aborted — 2026-05-18 23:01)

Pure replication of kze1lcti's recipe (single-process, K=1 static via
`--training-steps 64`, 64 g/cycle, no openings). Killed before any
milestones in favor of the next-up question: did dist infra alone break
us? Reframing to **use the new infra at K=1** to isolate "dist or not"
as the variable.

## 9x9-dist-K1-gpc64 (2026-05-18 23:09, wandb shvsohef) — FAILED

Our new dist infra (4 workers + eval-as-gauge), K=1 static, **games/
cycle=64** (matching kze's batch). Per-cell unique dirs, JSONL eval
handoff working — first run where wandb dashboard actually shows
`eval/*` curves live.

| epoch | pl    | vl    | plies | age | heuristic |
|------:|------:|------:|------:|----:|----------:|
| 10    | 3.685 | 0.510 | 23.5  |  5  | 0%        |
| 20    | 3.389 | 0.500 | 15.1  | 12  | 0%        |
| 30    | 3.146 | 0.447 | 12.1  | 18  | 0%        |
| 40    | 3.034 | 0.422 | 12.9  | 25  | 0%        |
| 51    | 2.917 | 0.400 | 11.3  | 33  | 0%        |
| 61    | 2.845 | 0.377 | 10.6  | 39  | 0%        |
| 71    | 2.651 | 0.389 | 13.6  | 40  | 0%        |
| 91    | 2.360 | 0.356 | 12.6  | 40  | 0%        |
| 100   | 2.257 | 0.333 | 12.0  | 39  | 0%        |

Failed target — heuristic = 0% across all 16 eval polls. Collapsed
*faster* than cell D (gpc=32) at the same epoch (e51: plies 11.3 vs
14.2). So **matching kze's games/cycle on dist did not reproduce kze's
behavior** — opposite direction, actually. Surprising.

Hypothesis (untested): on dist, more games per publish = fewer weight
versions represented in the buffer per unit wall-clock → less opponent
diversity → faster collapse. Smaller cycles publish weights more often
→ workers see fresher weights more often → more strata.

## 9x9-dist-K1-gpc16 (2026-05-19 ~00:00, wandb d2svpgk3) — FAILED

Pushing the publish-frequency theory further: same dist+K=1+buf=500k
setup but games/cycle=16. Weights published 4× more often than gpc=64
for the same data throughput.

| epoch | pl    | vl    | plies | gpc=32 (cell D) plies | gpc=64 plies |
|------:|------:|------:|------:|----------------------:|-------------:|
| 12    | 3.903 | 0.369 | 24.4  | (e10:) ~23.5          | (e10:) 23.5  |
| 25    | 3.495 | 0.395 | 19.4  | 21.6                  | (e20:) 15.1  |
| 35    | 3.327 | 0.445 | 20.5  | 17.2                  | (e30:) 12.1  |
| 46    | 3.149 | 0.414 | 15.9  | 15.8                  | (e40:) 12.9  |
| 59    | 2.995 | 0.365 | 15.6  | 15.2                  | (e51:) 11.3  |
| 72    | 2.891 | 0.343 | 19.2  | —                     | (e71:) 13.6  |
| 82    | 2.787 | 0.322 | 12.5  | —                     | (e81:) 11.9  |
| 100   | 2.685 | 0.336 | 15.1  | (e100:) 9.8           | (e100:) 12.0 |

Heuristic = **0% across all 8 eval polls**. Buffer never filled (final
size 238k/500k = 48%) because gpc=16 with K=1 is genuinely data-starved
— only 16 games × ~16 plies = ~256 positions per cycle going in.

**Negative result confirmed: games/cycle in [16, 32, 64] doesn't
fundamentally change the collapse trajectory.** gpc=16 ran roughly
parallel to gpc=32 from e35 onward; gpc=64 was a bit worse early.
None ever beat heuristic. The slope question is not about how often we
publish; it's about something upstream.

## The single-process vs dist semantic difference

Jason asked the sharp question: in single-process, what model generated
what data? Answer (precise, because it's serial):

- Each cycle: trainer's current model generates exactly N games, all of
  one version → SGD → new version → next cycle uses the new version.
- The buffer at any time has a strictly-ordered FIFO of "layer N = N
  games tagged version N." Stratification is exact, no overlap.

In our dist setup, workers run **continuously**. Between weight publishes
they keep generating with whatever weights they had last polled. The
trainer ingests these games and tags them with the trainer's *current*
version, which can be 1-2 cycles ahead of the actual gen-time version.
So:

- Per "version" there can be a variable, drift-y count of games — some
  workers finished a batch right before a publish, others finished
  right after, and the trainer can't tell them apart by version.
- The buffer is NOT cleanly stratified — it's jumbled.

This is the only structural difference between kze1lcti's setup
(single-process, worked at e85) and the dist family (continuous-gen
workers, all failed at e100). The K-axis tests killed publish-frequency
as the cure. Continuous-gen is the last untested axis.

## 9x9-dist-sync-K1-gpc32 (2026-05-19 ~01:00, wandb nox388ow) — LIVE

Tests the continuous-gen hypothesis by adding `--gen-once-per-publish`
to `gomoku.selfplay_worker`. After writing a batch, the worker sleeps
on the weights mtime until it advances; only then does it generate
again. Trades worker idle-time for clean per-version data stratification
that matches single-process.

Setup: same as cell D otherwise (dist, K=1, gpc=32, buf=500k, 4 workers
× 8 g/batch). The only knob changed is sync behavior.

Prediction: if continuous-gen was the culprit, this should reproduce
kze1lcti's e85 heuristic crossing. If it ALSO fails like cell D, then
the issue isn't worker behavior at all — it's somewhere deeper (code
regression, FP-arithmetic ordering on dist vs single-MPS, etc.).

### Result: failed target, but with one flicker

| epoch | pl    | vl    | plies | age | heuristic |
|------:|------:|------:|------:|----:|----------:|
| 11    | 3.526 | 0.388 | 21.8  |  6  | 0%        |
| 21    | 3.160 | 0.515 | 20.9  | 12  | 0%        |
| 31    | 3.116 | 0.568 | 16.6  | 18  | 0%        |
| 42    | 2.989 | 0.536 | 11.5  | 25  | 0%        |
| 52    | 2.894 | 0.485 | 10.6  | 32  | 0%        |
| 63    | 2.829 | 0.451 | 10.6  | 40  | 0%        |
| 74    | 2.728 | 0.415 | 11.3  | 48  | 0%        |
| 85    | 2.675 | 0.399 | 10.2  | 56  | 0%        |
| 100   | 2.622 | 0.393 | 12.5  | 66  | 0%        |

Across 16 eval polls, heuristic = 0% **except for one**: **e65 hit 5W-
15L-0D = 25%**. That's the first non-zero heuristic eval in any dist run
(cells C, D, gpc64, gpc16 all flat 0%). Not a sustained crossing — back
to 0% at e75 — but a real flicker that previous dist runs never showed.

Side-by-side at e100, K=1 buf=500k variants:

| variant                      | plies | vl    | pl    | best heur |
|------------------------------|------:|------:|------:|----------:|
| sync (gen-once-per-publish)  | 12.5  | 0.393 | 2.622 | **25%**   |
| cell D (continuous, gpc32)   |  9.8  | 0.406 | 2.834 | 0%        |
| gpc=64 (continuous)          | 12.0  | 0.333 | 2.257 | 0%        |
| gpc=16 (continuous)          | 15.1  | 0.336 | 2.685 | 0%        |

The plies trajectory shows sync collapsed *faster* than cell D in the
mid-game (e52: 10.6 vs cell D's 15.2) but had marginal recovery later
(11-12 range vs cell D's 9.8). vl is highest of any dist run at e21
(0.515) and stayed elevated mid-run — model held more uncertainty.

Conclusion: continuous-gen is NOT the main culprit. The flicker at e65
hints sync is *slightly* less bad, but the same ceiling — fast-attack
mode in 100 epochs. Hypothesis dies, but the data semantics fix is
worth keeping for future runs (clean per-version stratification
costs nothing and lets us reason about the buffer precisely).

## The real reframe: 100 epochs was just too few

After chasing the "regression," a subagent investigation flagged a bug
in `game.py:apply()` (snapshot taken AFTER stone placement instead of
BEFORE, making history planes redundant with the current frame). Pulled
the actual kze1lcti training log to verify and discovered the bigger
story:

| epoch | kze1lcti plies | kze heuristic (n=4) |
|------:|---------------:|--------------------:|
|     5 |          28.3  |        0%           |
|    20 |          16.2  |        0%           |
|    50 |          16.9  |        0%           |
|    75 |          13.2  |        50% ← noise spike (2/4) |
|    80 |          —     |        0%           |
|   100 |          12.9  |        **0%**       |
|   150 |          11.4  |        50% (2/4)    |

**kze1lcti at e100 also had heuristic=0%.** And plies at e100 = 12.9 —
basically identical to our "failed" runs (cell D: 9.8, gpc=64: 12.0,
sync: 12.5, diag-A: 14.6). The "kze hit heuristic at e85" claim that
drove this whole investigation was n=4 eval noise (2/4 wins = "50%").

The genuine signal comes from later checkpoints. kze1lcti's saved e176
checkpoint scores **15W-15L-0D vs heuristic at n=30 = real 50%**.
fresh-dist's e552 checkpoint scores 35% at n=30. Both are competitive
but neither crushes the heuristic.

**Conclusion: there is no regression.** Our 100-epoch runs at n=20 give
truer 0% readings; kze's "success" was n=4 noise. The pipeline genuinely
needs ~150-300+ epochs (and probably 800-sims-medium) to reach
heuristic-competitive play.

### Fix shipped anyway: history snapshot taken BEFORE placement

While investigating, found that `GameState.apply()` snapshots the board
AFTER placing the stone (before the flip). That means `history[0]` from
the next state's perspective is effectively identical to the current
frame, just from the previous player's view — i.e. plane 1 (my t-1)
duplicates plane 0 (my t), and plane H+1 duplicates plane H.

Fix: move the snapshot to BEFORE `new_board[0,r,c] = True`. Each
history slot now encodes "the board the mover observed when deciding,"
giving the network a real past observation per slot. Verified with a
trace: at s3 after a0,b0,a1 sequence, B's t-2 plane now correctly
shows empty (B had nothing before playing b0), where the bug had it
showing {b0}.

Not the regression (same code was present during kze1lcti — which
also hit collapse, just slower to be detected with n=4 evals), but the
representation is genuinely now correct, and the model gets distinct
history signal instead of redundant copies.

## Next: long-run baseline (500 epochs)

Pick best-looking config (TBD: probably dist+K=1+gpc=32+sync, or
single-process kze recipe) and run 300-500 epochs. The honest target
is "approximately match kze1lcti's e176 head-to-head strength at the
end." Compare via direct match against `kze-e176` checkpoint at n=40.



This kills the publish-frequency theory. What's left to explain why
kze1lcti beat heuristic at e85 while the entire dist family hasn't:

- **Hardware vs gradient interaction**: single-process MPS may produce
  different floating-point ordering / accumulation than 4-worker dist.
  Implausible but cheap to verify by running kze recipe straight (now
  on the queue properly).
- **Worker mtime-lag**: workers always run slightly stale weights
  between mtime polls. Single-process always has up-to-the-second
  weights. Could be a small bias toward "stale opponent" games.
- **Some package or code regression**: maybe one of the post-kze
  changes (history planes? wave-batched MCTS?) interacts badly with
  the dynamics. Hardest to test.

Next move (probable, pending live read): once gpc=16 finishes, run
a real pure kze replication (K=1 static, single-process, no opening
moves) on the same modern code. If it crosses heuristic, the
differentiator is single-process. If it doesn't, code regression is
strongly implicated.

## SUMMARY (2026-05-19, morning — michaelnny/alpha_zero recipe import)

After surveying https://github.com/michaelnny/alpha_zero (which ships a working
9×9 + 13×13 gomoku training recipe in PyTorch), confirmed our structure is
basically identical — same 17-plane input (`2*num_stack+1`, num_stack=8), same
ResNet trunk, same file-broadcast multi-worker pipeline, same wave-batched
MCTS + virtual loss. The interesting deltas are in stem padding, replay size,
temperature schedule, PUCT formula, and how long they train (their successful
runs are ~140–200k SGD steps, i.e. roughly 5–10× longer than ours).

### What we ported (worktree `worktree-alpha-zero-recipe`)

1. **Stem conv padding 1 → 3** (`model.py`). Their explicit "agent fails to
   block on edge cases" fix for gomoku. With kernel=3 and padding=3 the stem
   output grows from 9×9 to 13×13, giving the network a "virtual padding
   zone." The residual tower preserves that shape; the policy/value FC layers
   pick up the new spatial size automatically via `spatial = BOARD_SIZE +
   2*stem_padding - 2`. Added a `stem_padding` field on `ModelConfig` (default
   3). `load_checkpoint` defaults missing `stem_padding` to 1 so older
   checkpoints still load.

   Param-count impact (medium): ~1.04M → ~1.06M; tiny: ~37k → ~75k.

2. **τ=0.1 after warm-up plies, not greedy** (`self_play.py`). Previous
   default was `tau = 1.0 if ply < temperature_moves else 0.0`. New:
   `tau = 1.0 if ply < temperature_moves else temperature_final` with
   `temperature_final` defaulting to 0.1. Stays sharp but not one-hot — the
   policy training target is now a real soft distribution (one-hot CE was
   degenerate), and a tiny amount of whole-game exploration keeps the model
   away from total-determinism collapses. New CLI flag `--temperature-final`.

3. **Replay buffer default 50k → 1.5M** (`train.py`). Matches their gomoku
   capacity. With our preallocated tensor buffer at 17 planes / 9×9 this is
   ~8.5 GB on the device — fine on the M5 Max's unified memory, but worth
   noting; downstream workers don't allocate this, only the trainer.

4. **AlphaGo Zero log-schedule PUCT** (`mcts.py`):

   ```
   pb_c = log((1 + N_parent + c_puct_base) / c_puct_base) + c_puct_init
   ```

   Replaces our constant `c_puct`. Defaults: `c_puct_init=1.25` (AGZ),
   `c_puct_base=19652` (AGZ). At our sim budgets (200–800/move) pb_c is
   nearly constant (~1.25–1.30), so this is mostly a recipe-faithfulness
   change, not a numerics shift. CLI flags: `--c-puct` (now interpreted as
   c_puct_init), `--c-puct-base`. Threaded through `MCTSGame`,
   `generate_games`, `generate_games_vs_baseline`, and `selfplay_worker`.

5. **Virtual-loss sign audit (correctness check, no change)** (`mcts.py`).
   Walked through `_select_one_vloss` → `_backprop_value_only`. Our impl
   does N-only virtual loss (no `W -= 1`), which is a soft variant of AGZ's
   strong vloss. The accounting is correct: N gets the visit credited up
   front, W gets the real value added when the eval comes back, total
   counts agree at end-of-wave. The wave=1 path matches the sequential
   path byte-for-byte under fixed seed (verified). Documented the soft
   choice with a comment.

### Things we deliberately did NOT port

- **α=0.03 Dirichlet.** That's their default but it's AGZ's Go-19 value.
  Our 0.13 (scaled to 9×9 branching) is correct; keep it.
- **One-of-5 random-transform augmentation at 50%.** Their sample-time
  augmentation is strictly weaker than our store-all-8 D4 expansion.
- **No gradient clipping.** They omit it; we keep ours.
- **Single-game eval vs prev-checkpoint with no gating, always-black.** Useless
  signal in gomoku where first-move advantage is huge; we keep alternating-
  colors + fixed external baselines.

### What's still on the deck (deferred)

6. **numpy-array MCTS storage (port `mcts_v2.py` idea).** Parent stores
   `child_N`, `child_W`, `child_P` as `float32[N_ACTIONS]` arrays instead of
   per-Node fields with a `children: dict`. Their docs claim large speedups;
   we'd expect 2–3× on top of our wave batching. Skipped this pass — non-
   trivial refactor and `_select_action` is already vectorized; the win is
   in cutting the Python dict/Node-object overhead in `_select_one`. Worth
   a separate dedicated experiment.

7. **Pro-games policy-match eval.** Their `eval_on_pro_games` computes top-k
   policy match + entropy + value MSE on a fixed dataset; zero-variance per
   checkpoint. Would fix our n=4 noise floor problem (the kze "heuristic e85
   crossing" turned out to be 2/4 = "50%" noise). Need to track down a free-
   style gomoku game database; deferred to after this run.

### Compute-horizon calibration (the big takeaway)

Their README admits 9×9 Go peaked at 140–200k training steps before
collapsing — and they emit one checkpoint per 5000 games. Our "failed"
runs at 100 epochs × 64 games = 6400 games are roughly **20–30× shorter
than their successful phase.** Many of our "collapses" may just be
undertrained, with fast-attack mode being a transient state the model
trains through if given enough cycles.

### Exp 9: Cross-game BFS-vectorized MCTS descent (deployed)

After surveying michaelnny's `mcts_v2.py` discovered our `Node` already owns
per-action arrays (N, W, P), so the agent's claimed "2-3× from porting v2
storage" was a misread — we were already at v2-equivalent layout. The actual
high-leverage refactor is **vectorizing PUCT across games at each
tree-descent level** within a wave (Exp 3 action #1 from the original
backlog).

Implementation in `mcts.py:_bfs_descend_one_per_game`:

- Within a wave, iterate wave-slots sequentially (slot 0 across all games,
  then slot 1, …) so within-game VL ordering is preserved.
- At each slot, BFS-descend one descent per game in lockstep. At every BFS
  level we stack the still-going descents' `(W, N, P, legal_mask)` arrays
  into `(P, N_ACTIONS)` and do ONE `np.argmax` instead of P individual
  calls. Soft virtual loss (N += 1) is applied as we descend, matching the
  sequential path.

Verified byte-for-byte against the sequential reference under fixed seed
(see `tests/test_mcts.py::test_wave_bfs_matches_sequential_byte_for_byte`,
covers wave∈{1,8,16}, sims∈{20,64}, 4 games × 3 seeds).

**Clean bench (medium model on MPS, 3-trial median):**

| n_games | sims | wave | seq    | bfs    | speedup | savings |
|--------:|-----:|-----:|-------:|-------:|--------:|--------:|
| 32      | 200  | 16   | 0.407s | 0.324s | **1.26×** | 20.4% |
| 32      | 200  | 32   | 0.343s | 0.301s | 1.14×   | 12.4% |
| 16      | 400  | 16   | 0.388s | 0.365s | 1.06×   | 5.8%  |
| 32      | 100  | 8    | 0.129s | 0.111s | 1.16×   | 13.8% |

The speedup scales with G (games per wave-call) and inversely with
wave_size (larger wave = fewer wave iterations to amortize over). In dist
mode with 4 workers × 8 g/batch (G=8 per wave), expected win ≈ 5–10%.
In single-process mode with G=32 and wave=16, expected win ≈ 20%.

**Honest take vs prior estimate.** I had pitched this as ~20% gen savings
across the board; the data shows that's right at the high-G end but
modest at our dist-worker config. The 160k-step run would gain maybe 30–60
minutes total, not the couple of hours I'd hoped. Still positive, still
worth keeping — and the code is structurally cleaner.

**The bigger lesson.** Reviewing michaelnny/alpha_zero we'd been pitched a
2-3× from "porting their `mcts_v2` storage layout." That estimate was wrong
*for our codebase*: our `Node` already owns per-action `N`/`W`/`P` arrays
and `_select_action` is already vectorized. We were structurally at the
"mcts_v2" layout the entire time — the agent comparison was implicitly
against their `mcts_v1`, not against what we actually have. Filed the
finding as a synthesis page so a future session doesn't promise the same
2-3× again: see [wiki/topics/mcts-perf-ceiling.md](wiki/topics/mcts-perf-ceiling.md).
The genuine "next 2×" needs structural work (batched `state.apply` on
tensor, C-extension `_init_node`, multi-device gen/train split), not a
numpy reshuffle.

### Next: long run on the new recipe

Plan to kick off a fresh run with all five recipe fixes plus the
cross-game vectorized descent: pure self-play, ~160k SGD steps,
K=1 dynamic, medium model with stem_padding=3, 800 sims, τ_final=0.1,
1.5M replay buffer, AGZ log-PUCT, on the dist setup with 4 workers.
Eval gauge as a side process so the trainer doesn't block on baselines.

## SUMMARY (2026-05-19, evening — az-recipe-160k LIVE, uncharted territory)

Run is live. wandb id `sppjo3z5`, name `9x9-sweep-az-recipe-160k`. This
section is a snapshot at e1285 (~50 min in); update as the run progresses.

### What changed from "medium + padding=3 + 800 sims" to "what's actually running"

Smoke run with the full AZ-recipe config (medium, stem_padding=3,
sims=800, wave=32, K=1, 4 workers + eval_worker on MPS) benched at
**~30 s/cycle** steady-state. Extrapolating, the 160k-SGD target =
5000 cycles = ~46 hours wall-clock. Root cause: stem_padding=3 grows the
post-stem feature map from 9×9 → 13×13, so every ResBlock does
(13/9)² ≈ 2.1× more conv work. Medium under this scheme is ~2× the
FLOPs/forward of the old 9×9-internal medium, and 4 workers contending
on one MPS device get no parallel speedup (single-stream).

To keep the run inside a multi-day budget on the M5 Max laptop, accepted
three cutbacks (cell Z in `scripts/run_sweep.py`):

| knob | recipe default | cell Z | impact |
|---|---|---|---|
| model size | medium (~1.06M) | **small (~324k)** | ~3× cheaper forward |
| stem_padding | 3 (AGZ edge-fix) | **1** (legacy 9×9 internal) | ~2× cheaper forward |
| n_simulations | 800 | **400** | ~2× fewer sims/move |

Other recipe knobs unchanged: K=1, 1.5M replay buffer, τ_final=0.1,
AGZ log-PUCT (c_puct_init=1.25, c_puct_base=19652), dirichlet α=0.13,
wave_size=32, batch_size=512, lr=5e-4. Added a `--stem-padding` CLI
override in `train.py` so the knob is now adjustable per-cell.

### Throughput — much faster than smoke suggested, but for a reason

Smoke (cold start, buffer filling) hit ~6.8 s/cycle. Steady-state in the
live run (after the buffer hit 1.5M cap around e1000) is **~1-3 s/cycle**:

| epoch | plies | cycle_s | notes |
|------:|------:|--------:|:------|
|     1 | 31.7  | 11.10   | cold start, untrained models = long games |
|    51 | 15.5  |  2.80   | already collapsed to fast-attack |
|   501 | 11.4  |  1.20   | steady state |
|  1001 | 14.2  |  0.90   | buffer at cap, gen ≈ pure ingest |
|  1285 | 11.2  |  0.6-3  | current |

**Jason's calibration:** the cycle time is *artificially low* while
games are short. Gen scales super-linearly with mean plies (wiki Exp:
e122→e125 saw plies 34.5→52.8 grow gen 48.5s→102.5s — 53% more plies
but 111% more wall-clock). If the model genuinely learns defense,
plies regrow toward 50–80 and cycle time blows out. So:

- **Lower-bound ETA** (plies stays ~12, no defense): remaining ≈ 3700 × 2 s = ~2 h
- **Upper-bound ETA** (plies → 50-80, real defense): remaining ≈ 3700 × 8-13 s = ~8-13 h

The interesting outcome (defense learned) is the one that costs
wall-clock. **The run getting slower mid-flight is a positive signal,
not a problem.** Saved as a calibration in user memory so we don't
re-extrapolate naively.

### Eval crossings — never seen in dist mode before

Through e730 the model was the standard "beats random, gets crushed by
heuristic and lookahead" failure pattern. Then it broke:

| epoch | random | heuristic | lookahead:depth=2 |
|------:|-------:|----------:|------------------:|
|   0   |   95%  |    0%     |        0%         |
| 730   |  100%  |    0%     |        0%         |
| **779** | 100% |  **10%**  |        0%         | ← first non-zero
| **831** | 100% |  **45%**  |      **25%**      | ← first real crossing
| 879   |  100%  |   25%     |        5%         |
| 1027  |  100%  |   35%     |       10%         |
| 1077  |  100%  |   30%     |       10%         |
| **1119**| 100% |  **68%**  |       12%         | ← climbing
| 1179  |  100%  |   50%     |        2%         |
| **1227**| 100% |  **70%**  |       10%         | ← latest

n=20 per eval, so noise floor ≈ 5%. 50-70% sustained over 400+ epochs
is **signal, not noise**. The wiki has never seen a dist run cross
heuristic this hard:

- `kze1lcti` (single-process, AZ-mini-history-800sims): heuristic 50%
  at n=30 only at the e176 checkpoint, with n=4 wandb noise spikes
  earlier that we treated as "crossings" but weren't.
- `oo53qzvf` (fresh-dist, K=2): first real crossing at e315, plateaued
  ~35% by e552.
- All other dist cells (C, D, gpc16, gpc64, sync_gpc32): **0% heuristic
  every eval, 100 epochs each.**

This run is at **e1227 / heuristic 70%** and the trajectory is still
rising. lookahead2 stays low (2-12%), so the model is well short of
"actually strong" — but it's clearly doing something the prior recipes
couldn't.

### The plies puzzle

Despite the heuristic crossings, **plies have never recovered** from
the e51 collapse: 31.7 → 15.5 → 11-14 ever since. The eval crossings
happen *without* plies growing back. Two reads:

1. **Offense-only (Jason's working hypothesis):** the model is winning
   by attacking *better*, not blocking better. Heuristic plays simple
   greedy threats; if model's attack is faster/more accurate, model
   wins in fewer plies. No real defense learned. Predicts: lookahead2
   stays low; heuristic eventually plateaus around 70-80%; plies
   stay floor-pinned.

2. **Hidden-defense (alternative):** the model has learned to make
   threats that force the heuristic into defensive moves, then wins
   the race. In this read, plies stay short because the model dictates
   tempo. Predicts: lookahead2 starts climbing as this matures; plies
   start growing once it faces another strong attacker.

`vl` has dropped to ~0.11 (from 0.69 at e1) — model is very confident
in its evaluation. Combined with stable low plies + heuristic 50-70%,
this looks more like (1) than (2), but (2) isn't ruled out.

**Diagnostics to resolve it:**
- If plies climb in the next few hundred epochs → (2)
- If lookahead2 stays floor-pinned at 10-15% even as heuristic climbs → (1)
- Head-to-head against `kze-e176` once this run finishes will tell us
  whether this lineage is actually stronger or just better at exploiting
  the heuristic's specific weaknesses (recall fresh-dist beat kze 97.5%
  but lost worse to heuristic — same lineage non-transitivity may apply)

### Run hyperparameters as actually deployed

```
size                  : small (324,570 params, stem_padding=1)
sims                  : 400
wave_size             : 32
games_per_cycle       : 32 (4 workers × 8 g/batch)
batch_size            : 512
lr                    : 5e-4 (AdamW)
sgd_per_game (K)      : 1.0
replay_buffer         : 1,500,000
dirichlet_alpha       : 0.13
dirichlet_eps         : 0.25
temperature_moves     : 10
temperature_final     : 0.1   ← new
c_puct (init)         : 1.25  ← AGZ log-PUCT
c_puct_base           : 19652 ← AGZ log-PUCT
wave bfs descent      : on    ← new
target                : 5000 cycles = 160,000 SGD steps
infra                 : 4 dist workers + eval_worker on CPU
checkpoint_dir        : sweep_runs/az-recipe-160k/checkpoints/
launch                : python scripts/run_sweep.py --cell Z
```

### Open questions to resolve as the run continues

1. Does heuristic 70% hold or revert? Check at e2000, e3000, e4000.
2. Does lookahead2 ever climb above 25%? If yes, real defense.
3. Do plies regrow toward 50+ as the model gets stronger? Same signal.
4. Where does heuristic plateau — 70%, 85%, 100%?
5. Does the wall-clock prediction hold? Lower-bound (~2h remaining)
   ⇒ offense-only; upper-bound (~8-13h remaining) ⇒ real defense.
6. Head-to-head vs kze-e176 once done. The wiki found `fresh-e552 vs
   kze-e176 = 97.5%-2.5%` but `fresh-e552 vs heuristic = 35%` while
   `kze-e176 vs heuristic = 50%`. Same non-transitivity might bite
   here — sibling-vs-sibling measures mutual specialization, not
   absolute strength.

### UPDATE (2026-05-19, ~e1565 / +280 epochs) — diagnostics resolved: real defense

In the ~280 epochs since the SUMMARY was written, every "open question"
diagnostic resolved in favor of the **hidden-defense (real defense
learned)** hypothesis, not the offense-only read:

| diagnostic from previous entry | evidence at e1565 | verdict |
|---|---|---|
| lookahead2 stays floor-pinned 10-15% → offense-only | **55% at e1507** (from 0% at e779, 12% at e1119, 25-45% bouncing through e1300, 40-55% sustained e1400+) | refuted — real defense |
| plies regrow toward 50+ → real defense | **selfplay/plies_p90 spiking to 60-80** at e1.5k+, mean creeping 11→16-18 | confirmed — real defense |
| heuristic plateaus or reverts | **80% at e1507**, still rising | rejected — climbing past noise |
| eval times grow as games get longer | `time/eval_vs_lookahead2_s` went **10s → 100s**, `time/eval_vs_heuristic_s` went **12s → 35s** | confirmed |

The "fight" became visible in the **selfplay/plies_p90** chart before
it showed up cleanly in the mean. That's the right indicator for a
bimodal "fast attack + occasional long defense game" distribution —
the p90 captures the long-game tail that the mean averages over. Jason
flagged this on the wandb dashboard with "it's starting to put up a
fight." Worth keeping in mind: when the model is in transition between
offense-only and real-play regimes, **p90 is more sensitive than mean**.

### What this looks like in absolute terms at e1565

```
training:   pl 1.65   vl 0.087   plies 14-18   cycle 1.5-7.8s
eval:       random 100%   heuristic 80%   lookahead2 55%
buffer:     1.5M filled, age_p50 ≈ 215 epochs back
selfplay:   plies_mean 14-18, plies_p90 spiking to 60-80
```

**No prior run in this wiki has crossed lookahead:depth=2 above 25%
sustained.** We're at 55% and rising past e1500. This recipe + this
small/padding=1/sims=400 cutback combo is the first to break the
fast-attack collapse mode that every other run got stuck in.

ETA implication (per `feedback_self_play_eta.md` memory): the
lower-bound estimate (~2h remaining, plies stays at 12) is already
invalid — cycle times have started climbing as predicted. Realistic
remaining: **4-6 hours** based on the current rate of plies growth
and the cycle-time-grows-superlinearly-with-plies relationship. The
wall-clock blow-out is the predicted cost of the desired outcome.

### What's *probably* happening to make this recipe work

We can't isolate which knob made the difference without ablations,
but the most-likely culprits ranked by prior probability:

1. **τ_final=0.1 instead of greedy.** All prior runs sampled τ=0
   after warm-up, making policy targets one-hot and degenerate as
   cross-entropy targets. Soft targets carry distribution information
   that may have been the missing signal for defense.
2. **AGZ log-PUCT instead of constant c_puct.** Effective exploration
   constant rises slowly with parent visits, letting MCTS expand
   defensive subtrees at deep nodes that constant c_puct under-explored.
3. **Replay buffer 1.5M instead of 50-500k.** Buffer-fill rate
   amortized across ~250 epochs of fill (e0-e500). When the buffer is
   full, samples come from a much wider time window of opponents —
   may be what diversifies the policy gradient enough to learn
   defense.
4. **dirichlet α=0.13 (9×9-scaled).** Carried over from kze1lcti
   recipe; was already correct, so probably not the difference.

The most informative ablation, if we wanted to know: re-run with
τ_final=0 (greedy after warm-up) keeping everything else. If it
collapses back to offense-only, τ_final is the magic knob.

### UPDATE (2026-05-19 ~e2179 / +600 epochs) — regime change consolidated, ETA blowout confirmed

The previous UPDATE flagged that plies were starting to grow but the
mean still lagged the p90. ~600 epochs later the mean has fully caught
up: **self-play games are now playing through to ~27-32 plies on average,
the same range as the untrained e1 baseline — but for the opposite reason.**
At e1, both sides were random and couldn't end games; at e2179, both sides
defend well so games go deep.

| epoch | plies_mean | cycle_s | what's happening |
|------:|-----------:|--------:|:------------------|
|     1 | 31.7 | 11s | untrained, nobody can win |
|    51 | 15.5 |  3s | collapsed to fast-attack |
|  1285 | 11-14 | 1-3s | floor of fast-attack mode |
|  1565 | 14-18 | 2-5s | starting to put up a fight (p90 spiking 60-80) |
|  **2179** | **27-32** | **11-21s** | real defense in self-play; mean = untrained-rate, for opposite reason |

Loss trajectory across the regime change:

```
e=1   pl=4.22 vl=0.69  plies=31.7   (cold start)
e=51  pl=~4.0 vl=0.55  plies=15.5   (collapse)
e=1285 pl=1.65 vl=0.16  plies=11-14
e=1565 pl=~1.5 vl=0.10  plies=14-18
e=2179 pl=0.76 vl=0.08  plies=27-32  ← model very confident + games long
```

### Elo eval shipped — first live model_elo readings

`gomoku/rating.py` + `gomoku.eval_worker` integration deployed at e1854.
Per-cycle MLE Elo from `(opponent_anchor, score, n_games)` triples,
anchored at random=0, heuristic=800 (default), lookahead2=1200 (default).

Live readings (n=20 noise, ~±50 Elo SE):

| epoch | heuristic | lookahead2 | model_elo |
|------:|----------:|-----------:|----------:|
| 1854 | 70% | 55% (2W-0L-18D) | 1121 |
| 2148 | — | — | 1085 |
| 2159 | 62% | 75% (10W-0L-10D) | **1183** |
| 2167 | 60% | 60% (5W-1L-14D) | 1097 |

Stable around **1100-1150**, between heuristic (800) and lookahead2 (1200).
That maps to "competent attacker with solid defense" — qualitatively
matches the eval evidence and the self-play game lengths.

Calibration script (`scripts/calibrate_anchor_elos.py`) is running in the
background to replace the seeded anchor Elos with measured ones from
round-robin matches between random / heuristic / lookahead{2,3,4,5}.
When it finishes, ANCHOR_ELOS gets updated and the live readings will
re-anchor accordingly. Currently the model_elo trajectory is informative
in shape regardless because anchors are consistent across cycles.

### ETA blowout — confirmed exactly as Jason calibrated

The previous UPDATE quoted the lower/upper bound:

> Lower-bound ETA (~2h remaining, no defense): already invalid
> Upper-bound ETA (~8-13h remaining, real defense)

Reality at 2:47h in (e2179): cycle time has grown ~6× (2s → 15s mean).
Remaining 2821 cycles × 15s ≈ **11.8h remaining**. Total projected
**~14.6h** vs the original 10h estimate. The interesting outcome (real
defense learned) was always the one that costs wall-clock. Saved
[feedback_self_play_eta.md][] memory now has two confirming data points.

### What the trainer is still doing right

- `vl` keeps decreasing (0.69 → 0.08): value head is sharper than ever.
  This is the OPPOSITE of the prior "collapsed to z=0 / z=-1" failure
  mode where vl trivially crashed.
- `pl` keeps decreasing (4.22 → 0.76): policy distribution is matching
  MCTS visit-counts well; the τ_final=0.1 soft-target hypothesis from
  the previous UPDATE is consistent with this (one-hot targets would
  hit a floor higher than 0.76).
- `buffer/age_p50 ≈ 88` (was 246 at the diagnostic-resolution checkpoint):
  positions in the buffer are ~88 epochs old at median. The ring is
  rolling faster because cycle time is up but training_steps/cycle is
  fixed at 32. That means each "weight version" lives in the buffer for
  ~88 epochs of training_steps worth of decay.

### Where this leaves the picture

The az-recipe-160k run has demonstrated, in roughly 2.8 wall-clock
hours and 70k self-play games, that the fast-attack collapse the wiki
spent 600+ lines documenting is not a fundamental property of our
codebase. With the right recipe (τ_final=0.1, log-PUCT, 1.5M buffer,
adequate buffer-fill before SGD heat builds up), 9×9 freestyle gomoku
self-play CAN cross into real defense. The recipe replicates.

Remaining ~12h will tell us:

1. Does heuristic plateau at 70-80% or push higher?
2. Does lookahead2 plateau at 60-75% or push higher?
3. Does model_elo climb past lookahead2's anchor (~1200)?
4. Head-to-head against `kze-e176` once done: is this lineage
   transitively stronger, or just locally specialized?
5. Cycle time growth: does it stabilize around 15-20s or keep climbing
   as plies grow further? (Wiki Exp data: 53% more plies → 111% more
   cycle, super-linear.)

[feedback_self_play_eta.md]: ~/.claude/projects/-Users-jason-code-gomoku/memory/feedback_self_play_eta.md

## Anchor Elo calibration + lookahead-depth-3 bug (2026-05-20, while live run continues)

### Calibrated baseline Elos (n=50 round-robin, anchor=random=0)

|         | Elo  | notes |
|---------|-----:|:------|
| random  |    0 | anchor |
| heuristic | **591** | reference for "strong greedy without lookahead" |
| lookahead:depth=2 | **604** | essentially equal to heuristic (50W-0L-50D vs heuristic) |
| lookahead:depth=3 | **249** | **broken — weaker than heuristic** (see bug below) |
| lookahead:depth=4 | **629** | slightly above d=2 / heuristic |
| lookahead:depth=5 | **711** | strongest of the fixed baselines |

The real spread is much tighter than the seeded `ANCHOR_ELOS` (which had
heuristic=800, lookahead2=1200, lookahead4=1500). Keeping seeded values
in code for now per Jason's call: lookahead2 and heuristic playing equal
strength is a *result style* coincidence, not a meaningful similarity,
and re-anchoring to the calibration would compress the model_elo
trajectory misleadingly.

### Bug: odd-depth lookahead has a horizon-effect static eval flaw

Investigated by a subagent. Empirical pattern from the calibration above
is decisive — depth=3 (249) is the weakest depth>1 baseline; even depth=5
(711) only beats d=4 (629) by a small margin despite searching deeper.

**Root cause:** `evaluate_position` in `gomoku/baselines.py:105-116`
credits "4 stones in a window with 0 opp stones" with weight `_OPP_W[4]
= 22500` without distinguishing **open-fours** (actually unblockable) from
**half-open fours** (trivially blocked). At odd search depths, the
searcher's last move sits just before the leaf, builds such a threat,
and the search ends before the opponent's forced block — the static
eval reports the threat as if it'll resolve into a 5-in-a-row. At even
depths the opponent always gets a final move and naturally refutes the
hallucinated threat before the eval is taken.

Concrete trace (verified): position after `e5,f5,f4,d6,e4,d4,e6,e3`
(B to move). Heuristic and d=2 both pick `e7` (defensive). d=3 picks
`g4` because its 3-ply search builds a fake "live 4" the static eval
scores at +22350. After fix: d=3 still slightly prefers g4 (1976) over
e7 (1171) in this position, suggesting **the eval also over-credits
open-3 patterns** (3 stones in a window with both ends open), not
just live-fours. The fix below only catches the live-four case.

### Partial fix shipped: depth=0 quiescence for immediate-win threats

`gomoku/baselines.py:_negamax` now applies a 1-ply quiescence at
`depth==0`: if the opponent (just-moved side) has an immediate winning
move (a 4-in-a-row completable to 5), play the forced block before
calling `evaluate_position`. ~25 lines. O(81) max additional work per
leaf, only fires when needed.

**Empirical effect (regression tests in `tests/test_lookahead_quiescence.py`):**

|                          | pre-fix | post-fix |
|--------------------------|--------:|---------:|
| d=3 vs heuristic n=30    | 0% (heur wins all 50/50 in calib) | **83% (d=3 wins 25/30)** |
| d=3 vs d=2 n=30          | 5% (calib showed 5/50 d=3 wins) | 0% (still loses all 30) |
| d=2 vs heuristic n=30    | 50% (all draws)               | 35% (slight shift, within noise) |
| d=4 vs heuristic n=30    | ~37%                           | 45% (modest improvement) |
| Quiescence leaf vs raw   | raw=-23615 (hallucinated threat) | quiesce=+316 (post-block reality) |

**What remains broken:** d=3 still loses to d=2 100% of the time. The
gap is likely from the same eval flaw extended to open-3 patterns (the
eval credits any 3-stones-in-a-5-window without checking whether the
ends are blocked). Fixing that requires either a more nuanced static
eval (open-vs-half-open detection) or deeper quiescence (2-ply rollout
for live-three threats). Filed as a known limitation, not blocking
current work since we only use depth=2 in live eval and were planning
to add depth=4 (both even, unaffected by the odd-depth flaw).

### Why this bug doesn't affect the model's training

`heuristic_player` uses `_score_all_moves` + `_find_immediate_wins`
short-circuit (`baselines.py:202-226`), not `evaluate_position`.
Model training uses the network's value head, not `evaluate_position`.
So the static-eval horizon flaw only manifests in negamax lookahead
opponents we play against in eval. Live eval uses lookahead:depth=2
(unaffected). Calibration evidence here shows lookahead:depth=4 is
also unaffected. The bug is real, the fix is partial, but neither bites
our current training pipeline.

## SUMMARY (2026-05-20, ~e3500 — perf detour, GPU reality, white_wins → 0)

While the run continued from e3400 to e3500+, three things happened that
are worth recording. Run id `sppjo3z5`, az-recipe-160k.

### Perf detour: 1-worker × 32 games + compile was a regression

A subagent profiled the live setup, claimed 2-3× speedup available via
"collapse 4 workers × 8 games to 1 worker × 32 games + torch.compile"
based on the model "MPS is single-stream per process so 4 workers don't
parallelize." Shipped as cell Z reconfig at commit 05a2425. Backed up
checkpoint dir (`sweep_runs/az-recipe-160k.bak-20260520-0921`) + HF
push of worker_weights at e3410 before deploying.

**Result: slower, not faster.**

| config | cycle_s steady | games/sec | model_elo band |
|---|---:|---:|---:|
| 4w × 8g × wave=32 (pre-detour) | ~33 | 0.97 | 1400-1665 |
| 1w × 32g × wave=64 + compile | 36→63 (growing) | 0.5-0.9 | 1280-1530 |
| 4w × 8g × wave=64 (rollback, keep wave) | ~22 | 1.45 | recovered |
| 8w × 8g × wave=64 (Jason's "8 workers" call) | ~17 | 1.88 | sustained |

Lessons banked in [wiki/topics/mcts-perf-ceiling.md](wiki/topics/mcts-perf-ceiling.md)
2026-05-20 update section: per-process bench under-counts cross-process
MPS parallelism; `torch.compile` doesn't pay off when workers reload
weights every cycle; at this model size more workers > bigger batches.

Also re-added `lookahead:depth=4` to default eval baselines (had been
dropped during the launcher cycle); restored model_elo MLE precision.

### GPU underutilization is structural

8 workers running, Activity Monitor shows GPU at ~30-40%. Each MPS
forward call on the small (324k param) model is ~2ms regardless of
batch size 1-256. The Python tree-walk between calls
(`_bfs_descend_one_per_game`, `state.apply`, `_init_node`) is larger
than the GPU call itself. **No worker count can fix this** — the
kernels are individually too small to saturate ~5120 GPU cores; more
parallel callers just queue more 2ms calls.

The "real next 2×" remains structural: batched `state.apply` on tensor
+ C-extension `_init_node` to move Python work off the critical path so
the GPU can be fed continuously. Filed as pending — not blocking the
current run.

### `selfplay/white_wins → 0`: first-mover-advantage signal

Around the time the buffer age dropped from ~250 to ~75 (= the buffer
finishing its post-restart rollover into the current weight regime),
self-play white wins fell to near zero. Black wins ~all games now.

This is the natural endpoint of freestyle 9×9 gomoku self-play —
with no opening restrictions and both sides being the same model,
black's first-move initiative becomes decisive once the model finds
strong enough attacks. The asymptotic state of perfect freestyle
self-play is **black always wins**.

Implications:
- The value head's signal degrades for white-side positions (almost
  always z=-1, trivially predictable). Some of the ongoing pl/vl uptick
  may reflect this — the model is exploring policies in a regime where
  the value target is too easy to fit.
- Per-cycle Elo from fixed-baseline eval is the more informative
  metric now; self-play winrate is saturated.
- If we want to break the asymmetry deliberately:
  1. `--random-opening-moves N` (already in the CLI; decouples first-
     mover identity from the model's preferred attack opening)
  2. Opponent-mix (white plays as a past checkpoint occasionally)
  3. We're committed to freestyle rules — renju-style opening
     restrictions are off the table

For now, accept the signal. Strength-by-eval is still climbing. If
training stability degrades further (pl > 0.6 sustained, or value
trivially collapses to ±1), revisit then.

### Other small wins shipped during this phase

- `--worker-min-positions` + `--sgd-per-position` ingest mode
  (commit 85eeccc). Wasn't deployed for the current run but the code
  exists. Solves the "buffer age varies with mean plies" problem the
  trainer otherwise has. See [feedback_self_play_eta.md][] memory.
- Eval-side parallelism (commit d913447): `play_match_parallel` via
  multiprocessing.Pool. ~2× eval-cycle speedup, lets us include
  lookahead:depth=4 in the live anchor set without blowing the
  eval-budget per cycle.
- Lookahead:depth=3 horizon-fix quiescence in `_negamax` depth=0
  (commit c491d80, partial — fixes the immediate-win-completable-4
  case, leaves open-3 patterns as remaining d=3-vs-d=2 weakness).

[feedback_self_play_eta.md]: ~/.claude/projects/-Users-jason-code-gomoku/memory/feedback_self_play_eta.md

## Lockstep vs continuous worker orchestration — trade-off analysis (2026-05-20)

Jason raised the question: would going back to `--gen-once-per-publish`
(lockstep) help, possibly in a "2 waves of 4" staggered design? Recording
the analysis here so we don't re-derive it.

### What "lockstep" means in this codebase

The `gomoku.selfplay_worker --gen-once-per-publish` flag (commit 519fd0c,
2026-05-18) makes a worker: produce one batch → wait for the weights file's
mtime to advance → produce the next batch. The result is that each cycle
of trainer-ingested games comes from exactly ONE weight version, instead
of a smear of recent versions that continuous-mode workers produce when
they poll mtime less often than the trainer publishes.

### What "model does better with lockstep" actually says

Wiki history claim came from the 2026-05-19 `nox388ow` run ("9x9-dist-sync-K1-gpc32")
where sync mode produced the first non-zero heuristic crossing at e65 in
a 100-epoch window where every dist config otherwise failed. The wiki's
own honest read of that run: "the same ceiling — fast-attack mode in 100
epochs. Hypothesis dies, but the data semantics fix is worth keeping for
future runs." So the empirical evidence is "slightly less bad / earlier
flicker," not a documented training-strength gain at convergence. Re-litigate
at our current model_elo 1400-1600 strength would be a controlled A/B, not
a slam-dunk improvement.

### The "2 waves of 4 in lockstep" idea has a hard dependency wall

Intuitive design: split 8 workers into Group A (w0-w3) and Group B (w4-w7).
A locks to publish K, produces games. While A produces, B locks to publish
K+1 and produces concurrently. Keeps both groups busy, no idle, clean
stratification.

**This doesn't work** because publish K+1 doesn't exist until the trainer
consumes K's games and trains. So B has to wait for A to finish, then
wait for the trainer cycle, *then* it can produce for K+1. The "staggered
overlap" collapses to serial: only one group is producing for a real
publish at any time. Effective throughput = 4 workers, not 8.

The only way to get true concurrent production with lockstep semantics is
to accept *some* smear — Group B starts producing with the current weights
while the trainer is still consuming Group A's batch. But that's literally
what continuous mode (with mtime polling) does today.

### Realistic lockstep deployment at our scale

If you want to try lockstep, the right experiment is `--gen-once-per-publish`
on **all 8 workers** + `worker_min_games=64` (or `worker_min_positions=20_000`
if pairing with the constant-age fix). At our current 30s/batch worker speed
vs ~10s trainer-cycle, workers are already the bottleneck — they don't idle
in continuous mode, so lockstep adds zero idle cost. The buffer becomes
genuinely per-version stratified. The cost is 2× SGD steps per cycle (K=1
× 64 games instead of 32) and ~½ the buffer age (more positions per cycle
= faster turnover).

### Trade-off summary

| dimension | continuous (today) | pure lockstep | "2 waves of 4" |
|---|---|---|---|
| throughput at our 30s/batch | bottlenecked by workers (max) | same (workers still bottleneck) | **lower** — collapses to 4-worker serial |
| buffer stratification | smeared, ~1-2 versions per cycle | clean, 1 version per cycle | clean per group, but throughput half |
| training cost per cycle | 32 SGD (K=1 × 32 games) | 64 SGD (K=1 × 64 games) | 32 SGD |
| buffer age | ~75 (steady state) | ~37 (halved) | ~75 (unchanged) |
| code complexity | none — default | add `--gen-once-per-publish` flag | non-trivial — needs new launcher logic |
| empirical evidence for training benefit | baseline | "slightly less bad" at one early-phase data point | none (not yet tested) |

### Why lockstep won't help GPU utilization

The 30-40% GPU utilization is structural — each MPS forward call on a
324k-param model is ~2ms regardless of how many workers are calling it.
Lockstep aligns *which weight version* workers use; it doesn't change
per-call latency. The "real next 2×" is still in batched `state.apply`
on tensor + C-extension `_init_node` per
[wiki/topics/mcts-perf-ceiling.md](wiki/topics/mcts-perf-ceiling.md).

### Conclusion

Lockstep is a *training-side* lever, not a perf lever. The case for trying
it is "we want cleaner per-version buffer composition for training
stability arguments and the empirical sync-mode flicker hint." The case
against is "it's an A/B without strong evidence, costs 2× SGD per cycle,
and the existing continuous mode is producing strong results (model_elo
1400-1600)."

Not deploying today. If we see training-stability problems (pl > 0.6
sustained, value collapse, oscillating eval), revisit as the natural
next intervention. Filed as a known available option.

## Next-run config sketches — collected for re-assessment (2026-05-20)

Holding pen for ideas to evaluate when the current az-recipe-160k run
finishes. Jason: "for the next run, I really want a full buffer and this
flat AND a bunch of games-per-model, rather than some-games-across-a-
spread-of-models."

### Observation: `buffer/age_mean` trajectory of the current run

From wandb chart, full run e1 → e4200:
- e1-1000: rapid climb to ~250 as buffer fills with "old" weight tags
- e1000-2500: long descent as ingest-per-cycle caught up with buffer size
  (defense learned → longer games → more positions per cycle → faster
  buffer turnover → age drops)
- e2500-3000: plateau ~50-60 (steady-state equilibrium with current
  ingest/buffer ratio)
- e3000-3700: bumps to 100-130 — those are the restart-induced transients
  (--resume from latest.pt loads an old-version-tagged buffer, age climbs
  until ingest of new positions evicts the loaded slots)
- e3700+: back to ~50-60 equilibrium

### The age-vs-buffer-vs-ingest math

```
median_age (cycles) ≈ buffer_size / (2 × positions_per_cycle)
positions_per_cycle = games_per_cycle × mean_plies × D4_aug_factor (= 8)
```

Pick any two, the third follows. The current az-recipe-160k run has:
- buffer_size = 1.5M
- games_per_cycle = 32, mean_plies ~50, D4 = 8
- positions_per_cycle = 12,800
- → median_age = 1.5M / (2 × 12800) = **58** ← matches observed equilibrium

To hit Jason's target of age ≈ 200-250 with a stable model playing
50-plies games:

| games/cycle | needed buffer | memory at 17×9×9 f32 |
|---:|---:|---:|
| 16 | 2.5M | ~14 GB |
| **32** | **5.1M** | **~28 GB** ← practical sweet spot |
| 64 | 10M | ~57 GB ← upper limit before CPU-buffered |

M5 Max with 64-128 GB unified memory can hold 5M comfortably. 10M is
the ceiling before we'd need CPU-resident buffer with on-demand GPU
transfer.

### Proposed cell `Zlock` (lockstep + bigger buffer)

Combines the lockstep analysis above with the bigger-buffer math:

```
size:                  small (or medium — see below)
stem_padding:          1 (or 3 — see below)
n_simulations:         400 (or 800 if we have wall-clock budget)
wave_size:             64 (kept from current best)
games_per_batch:       8
n_workers:             4 (lockstep)
games_per_epoch:       32 (= 4 × 8, all from one weight version)
worker_min_positions:  ~12_800 (the constant-age fix from commit 85eeccc)
sgd_per_position:      0.0025 (K=1 SGD per game at this ingest)
batch_size:            512
lr:                    5e-4
buffer_size:           5_000_000           ← 3× current
dirichlet_alpha:       0.13
dirichlet_eps:         0.25
temperature_moves:     10
temperature_final:     0.1
c_puct / c_puct_base:  1.25 / 19652 (AGZ defaults)
epochs:                ? (TBD per total budget)
EXTRA worker arg:      --gen-once-per-publish
```

Effect:
- Each cycle's 32 games all come from one weight version (lockstep)
- 32 × 50 × 8 = 12,800 positions per cycle
- 5M buffer rolls every 5M / 12.8k = 391 cycles
- median_age ≈ 195 — close to Jason's 200-250 target
- Pairs cleanly with the constant-age positions-based ingest fix

### Other decisions to re-assess when the current run finishes

These are knobs we've been deferring; the next-run config decision is a
good time to reconsider each:

1. **stem_padding 1 vs 3.** Current run uses 1 (legacy 9×9 internal), which
   was a wall-clock cutback. With more compute headroom we could try 3
   (michaelnny's edge-fix). Cost: ~2× compute per forward.
2. **model size small (324k) vs medium (1.06M).** Medium would give the
   model more capacity to encode the strategies we've seen it discover.
   Probably worth trying once we've banked the perf improvements.
3. **sims 400 vs 800.** 800 is AZ-faithful; 400 was a wall-clock cutback.
   Could go to 800 if compute allows.
4. **K (sgd-per-game / sgd-per-position).** Currently 1.0; could try 2.0
   for more "per-version training depth" since the buffer is bigger.
5. **Random opening moves** (`--random-opening-moves N`). Would break the
   `white_wins ≈ 0` asymmetry by decoupling first-mover identity from the
   model's preferred attack opening. Useful if we want to keep training a
   more robust player past the freestyle-attack ceiling.
6. **Past-checkpoint opponent mix.** A fraction of self-play games where
   white plays as an older checkpoint. Same goal as #5 — diversify the
   training signal beyond pure self-play attack-only converged state.
7. **Lookahead bug structural fix** (the open-3 pattern issue from the
   2026-05-20 calibration). Currently lookahead:depth=3/5 are weaker than
   their parity-paired even depths. Would let us add depth=5 as a stronger
   anchor in eval without horizon-effect bias. Investigation deferred per
   2026-05-20 partial-fix commit c491d80.
8. **Structural perf: batched `state.apply` on tensor + C-extension
   `_init_node`.** The "real next 2×" from
   [wiki/topics/mcts-perf-ceiling.md](wiki/topics/mcts-perf-ceiling.md).
   Would let bigger model + bigger sims fit in same wall-clock.

### How this collection should be used

When the current run finishes (currently projected ~e8560), don't just
relaunch with the same config — pause and re-assess. The current run
already informs every choice above with real data. Pick the next-run
config with a specific question in mind ("is medium model better at this
strength level?" or "does lockstep+bigger buffer break the self-play
attack ceiling?") rather than a generic "improve everything."

## Buffer-composition feedback hypothesis — Jason's prediction at ~e4252 (2026-05-20)

After watching three full exploration arcs (e3041, e3441, e4093) each
peak-then-consolidate at successively-stronger floors, Jason flagged a
deeper concern: each arc *changes the shape of the buffer history* in a
way that the recovery has to fight against.

### The mechanism

During an exploration arc, the model plays short aggressive-attack
games (plies drops 50 → 35). The buffer ingests positions from those
short games. The recent buffer slice is increasingly dominated by:

- short-game opening positions (only the first ~10-15 plies appear
  often, mid-/end-game positions disappear)
- one-shot attack motifs the current arc found
- z = +1 / -1 outcomes (decisive games), few z = 0 draws

When the arc's exploration ends and the model tries to consolidate
toward longer defensive play, **it's training against a buffer that
disproportionately rewards what it just did**. The positive feedback
isn't just policy-target → action, it's also action → buffer
composition → next-cycle policy targets.

So far each arc has recovered. But each arc puts more short-game
positions into the buffer cumulatively. Eventually a consolidation
might not escape the bias.

### The prediction

> "I think it's going to keep bouncing off of these low vl/pl numbers.
> It plateaus, learns, plateaus, learns. What really bothers me is that
> while it is learning, we're changing the shape of the history (small
> games fill up buffer) so when it gets off, we're nudging it even
> more off balance. It is recovering for now, but sooner or later it'll
> tank. Prediction for next half hour: cl/pl will climb again, then go
> down again, and this will be the cycle in this recipe on this
> computer ... until eventually it doesn't recover (normal)."
>
> — Jason, 2026-05-20 at e4252

### Why constant-age doesn't fully address this

The constant-age ingest fix shipped at commit 85eeccc keeps buffer
*turnover rate* constant regardless of plies. It does NOT change WHAT'S
in the buffer at any given moment. During exploration, new positions
added are still short-game positions — there are just more of them per
cycle.

The genuine mitigations are deferred-decision items #5 (random opening
moves) and #6 (past-checkpoint opponent mix) — both inject NON-current-
arc content into self-play, diversifying the buffer's composition away
from "whatever the current arc is doing." Larger buffer (item already
in the next-run sketch) helps too — slower turnover means older,
pre-arc positions linger longer.

### Testing the prediction

The current run will either:
- Bounce as predicted (pl climbs, plateaus, climbs again, eventually
  fails to recover): validates the buffer-composition-feedback theory,
  argues strongly for items #5-#6 next run.
- Continue tightening (pl approaches an asymptotic floor with smaller
  and smaller arcs): suggests the recovery dynamic is robust enough
  that the buffer-shape concern was over-stated at this scale.

Cron check-ins over the next several hours will track this.

Connecting to [topics/az-at-scale-vs-laptop.md](wiki/topics/az-at-scale-vs-laptop.md):
this is the laptop-scale failure mode the topic page predicts — short
games + per-version concentration combining to make exploration arcs
self-reinforcing in a way Google's scale would smooth over.

## Run end: az-recipe-160k stopped at e5000 (2026-05-20)

Run `9x9-sweep-az-recipe-160k` (wandb id `sppjo3z5`) stopped by
user-requested kill at exactly e5000 — early termination from the
natural e8560 end-point because the data was already conclusive.

### Final-state metrics

```
epoch        : 5000 / 8560     (stopped early per Jason)
games        : 160,000         (effectively 159,968 since e0-3559 were resumed from)
wall-clock   : ~22 hours total run time
final pl     : 0.293           ← just above run's all-time floor of 0.27 (e3984)
final vl     : 0.035           ← just above floor of 0.027
final plies  : 59.2            ← long-game regime
final age    : 48              ← steady-state equilibrium
```

### Final eval sweep (e4990, last full cycle before kill)

| baseline | result | anchor |
|---|---|---:|
| random | 100% (20W-0L-0D) | 0 |
| heuristic | 60% (10W-6L-4D) | 800 |
| lookahead:depth=2 | 80% | 1200 |
| lookahead:depth=4 | 40% (with some losses) | 1500 |
| **model_elo** | **1290** (post-arc dip cycle) | |

Best eval of the run was e3881 with model_elo=1718 (perfect sweep of
random + heuristic + lookahead2, only 1 loss to lookahead4). The model
visited that peak twice and consolidated near it three times.

### The full arc story (5 explore-then-consolidate cycles)

| arc | epoch range | peak pl | peak vl | consolidation result | post-arc model_elo |
|----:|---|---:|---:|---|---:|
| 1st | e3041-e3265 | 0.58 | 0.094 | back to floor 0.36, plies recovered | 1665 |
| 2nd | e3441-e3500 | 0.55 | 0.094 | back to floor 0.27, plies recovered | **1718** (peak) |
| 3rd | e4093-e4150 | 0.41 | 0.056 | back to floor 0.27, plies recovered | 1555 |
| 4th | e4252-e4468 | **0.59** | 0.094 | back to ~0.30, plies recovered | 1531 |
| 5th | e4664-e4924 | 0.47 (stalled) | 0.080 | back to **0.293**, plies recovered | 1290-1444 |

### What this run validates / refutes

**Validated:**
- AlphaZero recipe (τ_final=0.1, log-PUCT, 1.5M buffer, dirichlet α=0.13)
  reliably breaks fast-attack collapse and learns real 9×9 freestyle
  gomoku from scratch. First time in this project.
- Eval setup (n=20 vs 4 fixed-Elo anchors + multiprocess parallel) gives
  a usable single-number strength signal (`model_elo`) that scales with
  real model strength.
- The cross-game BFS-vectorized MCTS descent (Exp 9 / commit `64551c8`)
  is the right perf architecture for our scale.
- Jason's calibration that cycle time scales super-linearly with plies
  played out exactly as predicted, twice (during defense learning, then
  again across each arc).

**Refuted / weakened:**
- The subagent's "2-3× speedup from 1 worker × big batch + torch.compile"
  estimate. Production parallelism dominated the bench-projected numbers.
  Lesson banked in `project_perf_bench_lesson` + the
  `mcts-perf-ceiling` 2026-05-20 update.
- The hypothesis that exploration arcs would shrink asymptotically (4th
  and 5th arcs were as deep as the 1st — the model is bouncing in a
  band, not converging to a fixed strength).
- The "eventually it doesn't recover" half of Jason's prediction — the
  5th arc was the broadest weakness (heuristic + lookahead2 losses,
  model_elo dipped to 1215) but the model fully recovered to pl=0.29
  by e5000. Five for five on consolidations.

**Partially supported:**
- Buffer-composition feedback IS a real mechanism (the arcs DID happen,
  they DID broaden over time, heuristic-specific lineage drift WAS
  observed), but it didn't break training catastrophically at this run
  length. The system has more robustness budget than the worst-case
  hypothesis predicted.

### Strength signal: was the model actually strong by the end?

The last few eval cycles (e4965-e4990) show:
- random: 100% across all five cycles
- heuristic: 57-75% with 0-6 losses (variance high)
- lookahead2: 68-92% (varies between perfect-sweep cycles and high-draw cycles)
- lookahead4: 40-82% (high variance, sometimes the model wins big, sometimes loses)
- model_elo: 1290-1519 band (wider than the mid-run 1500-1700 peak band)

By the end, the model is **noisier but not weaker** than its mid-run
peak. The arcs widened the variance envelope. Individual checkpoints
near e3881 (model_elo=1718) are probably stronger than the e5000 final
state — recommend retaining `epoch3881.pt` if it's still in checkpoints,
otherwise use the model_elo wandb chart to identify the highest-elo
saved checkpoint.

### What this run argues for in the next-run config

Re-stating from the [Next-run config sketches](#next-run-config-sketches---collected-for-re-assessment-2026-05-20)
section above, with run-end data as evidence:

1. **Lockstep + 5M buffer**: directly addresses the buffer-composition
   feedback that caused the arcs. The 5th arc happening *broader* than
   the 4th suggests we'd see continued widening of the variance
   envelope if we kept this recipe running. Lockstep + bigger buffer
   are the laptop-scale approximations of Google's per-version
   smoothing.
2. **Random opening moves**: would break the lineage-drift toward
   heuristic-loss specialization. The recurring "heuristic takes losses
   while lookahead2 stays strong" pattern is direct evidence the model
   is specializing AWAY from naive opponents.
3. **Past-checkpoint opponent mix**: same purpose — inject non-current-
   arc content into self-play.
4. **Larger model (medium 1.06M vs current small 324k)**: the small
   model may simply be at its capacity. With current recipe at floor
   pl=0.27 it has nowhere lower to go without overfitting. Medium has
   more room for strategy encoding.
5. **Sims 800 vs 400**: would give MCTS time to look further during
   exploration arcs, possibly tightening the consolidation depth.

The most informative single experiment for the next run would probably
be **lockstep + 5M buffer + random opening moves** — the three changes
most directly aimed at the failure mechanism we observed. Add medium
model + sims=800 if compute budget allows.

### Checkpoints worth retaining

- `epoch3881.pt` (if still on disk): the model_elo=1718 peak — best
  single checkpoint of the run by eval. Likely overwritten by the
  --keep-last-n=3 policy by now; check `sweep_runs/az-recipe-160k/
  checkpoints/` for what survives.
- `latest.pt`: the e5000 final state (pl=0.293) — embeds the replay
  buffer for resume, useful evidence for the next-run buffer-warm-start
  experiment.
- `worker_weights.pt`: lean inference-only copy of the e5000 model.
- HF backup at `https://huggingface.co/jasonyandell/gomoku-9x9` has
  `az-recipe-160k-e3409-perf-checkpoint.pt` — pre-perf-detour snapshot.

Backup directory `sweep_runs/az-recipe-160k.bak-20260520-0921/` also
preserves the e3409 state. Can be deleted once we're confident the
next run isn't going to need a fallback.

## 2026-05-20 — WL1 implementation smoke

Implemented the wave-of-lockstep plumbing in worktree `codex/wl1-lockstep`.
This is an implementation receipt, not a strength claim.

Changes:
- Trainer now has `--wave-mode`, `--wave-workers`, and
  `--wave-games-per-worker`. In wave mode it scans
  `worker-input-dir/v{version}/worker{worker_id}/*.pt`, waits until every
  expected worker has at least `G` completed games for the current version,
  ingests the whole visible tile, then advances the model version.
- Worker now has `--wave-mode`. It reads the checkpoint `epoch` as
  `model_version`, writes one file per completed game under the versioned
  outbox path, generates the first `G = --games-per-batch` games for a
  version, then greedily fills one extra game at a time until newer weights
  are available. It only reloads at game boundaries.
- `scripts/run_sweep.py` has Cell `WL1`:
  `WL1-wave-lockstep-5M-buffer`, small model, 400 sims, stem padding 3,
  8 workers x 8 games, 5M replay buffer, AGZ PUCT/Dirichlet defaults,
  temperature drop at move 30, `temperature_final=0.1`, and
  `sgd_per_position=0.0025`.

Smoke test:
- Command shape: 50 epochs, 4 workers, wave mode, `G=8`, tiny model,
  CPU, `n_simulations=1`, `wave_size=1`, 1 SGD step per wave, replay
  buffer capacity 1.3M so all version tags would survive to `latest.pt`.
- Artifact paths:
  `sweep_logs/wl1-smoke-50x4-fullbuffer/trainer.log` and
  `sweep_runs/wl1-smoke-50x4-fullbuffer/checkpoints/latest.pt`.
- Barrier receipt: 50 wave lines parsed, versions `0..49` all present.
  Minimum visible tile size was 38 games, maximum was 54 games, total
  greedy extras across the smoke were 685. Every parsed wave had worker
  minimum >= 8, so the barrier did not slip.
- Replay-buffer tag receipt: final replay buffer size was 1,153,416
  positions; `weight_version` contained all versions `0..49` with 50
  unique tags. Per-version position counts ranged from 17,904 to 25,976.

Verification:
- `python -m py_compile gomoku/train.py gomoku/selfplay_worker.py scripts/run_sweep.py`
- `python -m gomoku.train --help`
- `python -m gomoku.selfplay_worker --help`
- `python scripts/run_sweep.py --list`
- `git diff --check`
- `pytest` -> 60 passed in 17.82s

## 2026-05-20 — WL1 matched-throughput read

After the implementation smoke, we ran a short apples-to-apples throughput
check against the previous `az-recipe-160k` generation config:
`small`, `stem_padding=1`, 400 sims, `wave_size=64`, MPS, 8 workers.
This used wave mode for only 3 epochs, so it is a throughput sanity check,
not a run-quality result.

Matched WL1 benchmark:
- Artifact path: `sweep_logs/wl1-apples-zconfig-3epoch/trainer.log`.
- Epochs: 3.
- Games ingested: 250.
- Generation time: 31.9s.
- Generation throughput: 7.84 games/s.
- Wall throughput: 6.36 games/s.
- Mean plies: 29.2.
- Approximate training-position throughput: 1,817 positions/s.
- Average visible tile: 72.7 games with greedy extras.

`az-recipe-160k` comparison:
- First 100 epochs: 12.73 games/s, but games were much shorter at 17.4
  plies; approximate training-position throughput was 1,773 positions/s.
- First 3 epochs: 4.29 games/s, 30.7 plies, approximate
  training-position throughput 1,051 positions/s.
- Last 50 epochs: 2.37 games/s, 60.2 plies, approximate
  training-position throughput 1,141 positions/s.

Interpretation:
- By positions/s, wave-lockstep under the old Z-style generation config
  appears comparable to, or slightly faster than, the previous continuous
  setup. The barrier itself did not show a meaningful throughput tax.
- This strengthens the WL1 hypothesis operationally: the next run can test
  cleaner per-version buffer tiles without paying an obvious coordination
  penalty.
- What remains unproven is the training-dynamics claim: whether cleaner
  per-version tiles reduce the exploration/consolidation arcs and improve
  fixed-baseline strength.
- If WL1 still arcs despite clean tiles, the next suspects should move away
  from raw per-version under-sampling and toward opening monoculture,
  capacity, search depth, or greedy-extra bias.

## 2026-05-20 — Lane C perf run/config integration

Worktree `codex/gomoku-perf-extension`, Lane C only: run/config/docs surface
for Activity Monitor-oriented performance work. This is an implementation
receipt, not a strength or throughput claim.

Changes:
- Added `scripts/perf_microbench.py`, a bounded self-play/MCTS bench that runs
  through the existing evaluator and `generate_games` path. Use it to compare
  wall-clock seconds, games/sec, and positions/sec before changing a live run.
- Set WL1's `save_buffer_every=100` in `scripts/run_sweep.py` so the 5M replay
  buffer does not rewrite `latest.pt` every 20 epochs. Intermediate
  `epochNNNN.pt` snapshots remain cheap weights+optimizer saves; `latest.pt`
  is the expensive replay-buffer resume checkpoint.
- Added `wiki/topics/activity-monitor-perf-runbook.md` and linked it from the
  wiki index plus the MCTS perf ceiling page. The page preserves the practical
  operating rule: Activity Monitor GPU percent is a diagnostic hint, not the
  score. The score is production-shaped wall-clock throughput.
- README now exposes the microbench command and the checkpoint-throttling knob.

Working theory preserved:
- GPU underutilization remains structural at this model size: tiny MPS forwards
  are separated by Python MCTS/state work.
- Do not chase GPU percent by collapsing workers or inflating a single worker's
  batch unless same-shape wall throughput improves.
- The structural next wins belong at the native hot-path boundary, MCTS
  allocation cleanup, and evaluator materialization cleanup, not in core
  game/MCTS rewrites from this lane.

## 2026-05-20 — WL1 buffer-size correction and native state_ops phase

Correction to the WL1 plan: 5M replay-buffer capacity is too large for the
current hardware budget and changes more than we need. The next phase should
test wave-lockstep/per-version uniformity at the known 1.5M replay-buffer scale
instead of also changing buffer capacity.

Implementation changes in worktree `codex/gomoku-perf-extension`:
- `scripts/run_sweep.py` now names WL1 as `WL1-wave-lockstep-1p5M-buffer` and
  sets `buffer_size=1_500_000`.
- The native hot-path phase adds an optional `gomoku._state_ops_native`
  extension behind `gomoku.state_ops`. If the extension is absent, the NumPy
  fallback remains the source of truth.
- `GameState` and MCTS node initialization already call through this boundary,
  so future compiled work can focus on state apply / legal mask / terminal
  status without touching training semantics.
- Packaging now builds a platform wheel containing the extension via
  Setuptools. `GOMOKU_DISABLE_NATIVE_STATE_OPS=1` forces the fallback path for
  A/B checks.
- Smoke microbench on CPU tiny/MCTS shape showed native median 0.116s vs
  fallback median 0.124s. Treat this as a plumbing proof and small hot-path win,
  not a production MPS claim.

## 2026-05-21 — Native MCTS engine phase

Question from Jason: how deeply native can we go, and can we spend essentially
no amortized time on Python?

Implementation in worktree `codex/gomoku-perf-extension`:
- Added optional `gomoku._mcts_native`, an arena-backed C MCTS engine. It owns
  bitboard state/history, node storage, child creation, terminal/legal status,
  PUCT selection, wave virtual loss, backup, and input-plane materialization.
- `make_torch_evaluator` now exposes `evaluate_planes`, so native MCTS calls
  Python/Torch only at leaf-batch wave boundaries.
- `generate_games` automatically uses native MCTS when available and when the
  evaluator exposes `evaluate_planes`. `GOMOKU_DISABLE_NATIVE_MCTS=1` forces the
  old Python MCTS path for A/B checks.
- The smaller `gomoku._state_ops_native` extension remains the fallback boundary
  for Python `GameState` helpers.

Verification:
- `python setup.py build_ext --inplace`
- `pytest -q` with native MCTS enabled: full suite passed.
- `GOMOKU_DISABLE_NATIVE_MCTS=1 pytest -q`: fallback suite passed, native-only
  tests skipped.

MPS microbench results, small model, stem padding 1, fresh random weights:

| command shape | fallback | native MCTS | read |
|---|---:|---:|---|
| `--games 4 --n-simulations 32 --wave-size 4 --max-plies 6` | 2,438 aug pos/s | 2,888 aug pos/s | 1.18x |
| `--games 8 --n-simulations 400 --wave-size 64 --max-plies 16` | 701 aug pos/s | 2,200 aug pos/s | 3.14x |
| `--games 8 --n-simulations 400 --wave-size 64 --max-plies 32` | 728 aug pos/s | 2,007 aug pos/s | 2.76x |

Interpretation:
- This validates the Activity Monitor diagnosis. The big production-shaped win
  appears once the native boundary covers the full search engine, not just
  small state helper calls.
- Python is no longer paid per node/leaf in the Torch self-play path. It is
  still paid per wave evaluator callback and per outer move/record step.
- Next throughput question is multi-worker production behavior. The
  single-process result is large enough that WL1 should be re-benchmarked with
  native MCTS active before launching a long run.

## 2026-05-21 — WL1 10-epoch native production read

Jason asked for 10 epochs each at the parameters under consideration for the
next run. This was a throughput experiment, not a strength claim: fresh random
small models, MPS workers, no eval sidecar, no W&B, 10 epochs each, same WL1
recipe unless noted:

- `size=small`, `stem_padding=1`, `n_simulations=400`, `wave_size=64`
- `buffer_size=1_500_000`, `sgd_per_position=0.0025`, `batch_size=512`
- AGZ PUCT defaults (`c_puct=1.25`, `c_puct_base=19652`), `dirichlet_alpha=0.13`,
  `dirichlet_eps=0.25`, `temperature_moves=30`, `temperature_final=0.1`
- Wave-lockstep barrier; positions/sec below are D4-augmented training
  positions/sec computed from trainer log rows (`new_games * plies_mean * 8`).

Artifact summary: `sweep_logs/perf10-summary.tsv`.

| variant | workers x G | native MCTS | games | mean plies | wall pos/s | gen pos/s | wall games/s | mean tile |
|---|---:|:---:|---:|---:|---:|---:|---:|---:|
| `perf10-wl1-native-8w8g` | 8 x 8 | yes | 1065 | 26.8 | **2,379** | **3,303** | **11.25** | 77.0 |
| `perf10-wl1-native-4w16g` | 4 x 16 | yes | 853 | 28.0 | 1,918 | 2,152 | 8.61 | 75.0 |
| `perf10-wl1-fallback-8w8g` | 8 x 8 | no (`GOMOKU_DISABLE_NATIVE_MCTS=1`) | 915 | 26.8 | 1,863 | 2,264 | 8.85 | 74.6 |

Read:

- The next-run perf shape should stay **8 workers x 8 games**. Native search
  did not make the lower-process `4 x 16` shape better; it was 24% slower by
  wall positions/sec and 35% slower by wall games/sec.
- Native MCTS buys a real production-shaped win, but smaller than the
  single-process microbench: **1.28x wall positions/sec** and **1.46x
  generation positions/sec** over the Python-MCTS fallback at the same 8x8 WL1
  shape.
- Multi-worker scheduling already hid a lot of the old Python MCTS cost; the
  native engine still matters because it lets the same WL1 recipe ingest about
  516 more augmented positions/sec wall-clock under this short run.
- The 10-epoch WL1 launch estimate from this read is roughly 11 games/sec while
  games are in the 25-30 ply early-training range. As usual, expect wall time to
  grow if the model learns defense and plies climb toward 50-80.
