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

## WL1 live run log (2026-05-20, wandb l8mbntcm)

Live tracking for the WL1 wave-of-lockstep run launched 2026-05-20 20:53.
This section is appended-to as the run progresses; older entries are
preserved with their epoch + wall-clock so the trajectory is auditable.

**Setup**
- wandb run: `l8mbntcm` (https://wandb.ai/jasonyandell-forge42/gomoku/runs/l8mbntcm)
- wandb workspace (overlay WL1 + Z by clicking both in the run picker):
  https://wandb.ai/jasonyandell-forge42/gomoku?nw=ul0vliphj6x — created by
  [scripts/wandb_workspace.py](scripts/wandb_workspace.py). Switch x-axis
  to `_runtime` per-panel to see wall-clock savings.
- cell: `WL1` = wave-lockstep, 1.5M buffer, native MCTS, 8w × 8g, AGZ recipe (small,
  stem_padding=1, sims=400, τ=1.0→0.1 at move 30, AGZ PUCT/Dirichlet)
- baseline: `Z` = az-recipe-160k (wandb `sppjo3z5`); same model/MCTS recipe,
  continuous (not wave) self-play, python MCTS
- worker bug at launch: see "WL1 first launch + worker race fix" entry below

**Hypothesis under test** (cross-ref:
[wiki/topics/wave-of-lockstep-design.md](wiki/topics/wave-of-lockstep-design.md),
"Buffer-composition feedback hypothesis" section above):
Jason's prediction was that the explore-consolidate arcs Z showed were
driven by *per-version concentration* in the buffer — each version's small
biased slice of self-play trained the next version, amplifying drift.
Wave-of-lockstep makes every model version contribute a uniformly-sized
tile, removing that feedback loop.

**Live milestones**

| epoch | wall   | pl   | vl    | plies | elo  | h%  | la2% | la4% | note |
|------:|:------:|-----:|------:|------:|-----:|----:|-----:|-----:|------|
| 1     | 00:07  | 4.31 | 0.67  | 34.1  | —    | —   | —    | —    | start |
| 113   | 04:00  | 2.61 | 0.61  | 14.8  | 389  | 0   | 0    | 0    | first eval shipped, fast-attack regime |
| 146   | ~05:30 | 2.33 | 0.58  | 13.4  | 776  | 30  | 25   | 0    | **first heuristic crossing** — bouncy, not sustained |
| 263   | ~10:00 | 1.91 | 0.40  | 12.9  | 937  | 70  | 20   | 0    | heuristic 70%, value head accelerating |
| 360   | ~13:30 | 1.80 | 0.13  | 11.2  | 1271 | 80  | 48   | 48   | **arc 1 peak**, la4 hit 48% (Z barely reached this even at e3881) |
| 499   | ~18:30 | 1.78 | 0.05  | 10.3  | 1281 | 55  | 70   | 52   | la2 sustained 70%, la4 still high |
| 524   | ~19:30 | 1.79 | 0.04  | 11.0  | 1110 | 65  | 50   | 18   | la4 collapse — consolidation begins |
| 592   | ~22:30 | 1.78 | 0.03  | 10.3  | 1041 | 40  | 55   | 20   | arc 1 trough, ~80 epochs after peak |

Will update as new evals land.

**Wall-clock vs Z calibration** (rough):

| signal | Z (az-recipe-160k) | WL1 (live) | ratio |
|---|---|---|---|
| first heuristic crossing | e1119 (~7h wall) | e146 (~5min wall) | **~8× faster epoch, ~80× faster wall** |
| vl ≈ 0.08 | e2179 (~2:47h wall, "full regime change") | e360 (~13min wall) | ~6× faster epoch |
| la4 ever > 30% sustained | only briefly near peak e3881 | already hit 52% at e360 | qualitatively further along |
| arc wavelength | ~800-1000 epochs | ~80-100 epochs | ~10× compression |

The wall-clock ratios combine *two* speedups: native MCTS (~1.7× per
trial-bench measurement) and the per-epoch convergence speedup from
wave-lockstep. Disentangling them requires running WL1 with
`GOMOKU_DISABLE_NATIVE_MCTS=1` for an apples-to-apples comparison; not in
scope today.

**Open questions for the run**
1. Does WL1's plies regrow? At e605 plies are still 10-11 — fast-attack
   regime persists. Z's plies regrew at e2179, signaling the defense
   regime. WL1's value loss already matches Z's e2179 value loss, but
   policy loss + plies haven't yet.
2. Does the next arc consolidation fully recover? Z had 5 arcs in 5000
   epochs; if WL1's arcs are at 100-epoch wavelength, we'd expect ~50
   total. Most useful signal: do *later* arcs broaden or remain tight?
3. Does the run plateau or keep climbing? Z plateaued around elo 1100-1500
   for its last 1000 epochs.

## WL1 first launch + worker race fix (2026-05-20)

WL1 was launched twice on 2026-05-20:

**Attempt 1 (wandb `wo9py6m4`, killed at e97):** trainer worked fine, but
worker w6 crashed at e96. Root cause: greedy-fill race in
`gomoku/selfplay_worker.py::_atomic_save_wave_game`. Worker mkdir'd
`_records/v{N}/workerw6/` for a greedy-extra game; trainer ingested v{N}
and `rm -rf`'d the dir; worker then crashed on `torch.save` inside the
deleted directory. The dead worker blocked the wave barrier (which
requires all 8 workers ≥ 8 games), so the other 7 workers greedy-filled
~3,500 games each into a v97 tile that would never be consumed. ~25k
games of junk piled up before manual kill.

The wave-of-lockstep design doc had flagged this as an open question
("Should we add a barrier timeout? Probably yes, with a loud W&B
warning."), which wasn't shipped.

**Fix (commit `0d2c106`):** `_atomic_save_wave_game` catches
`FileNotFoundError | OSError | RuntimeError`, logs a one-liner, returns
None. The save loop drops the game (1 game of MCTS work wasted) and the
outer wave loop picks up new weights on its next iteration. Worker
survives; barrier eventually fires.

**Attempt 2 (wandb `l8mbntcm`, live):** same launch, fix in place. Has
seen 3 race-drops in 600+ epochs, all recovered cleanly. Barrier firing
every cycle.

**Why no buffer ingest of the 5M-buffer attempt's data:** an earlier
`WL1-wave-lockstep-5M-buffer` attempt (commit pre-`bd28670`, wandb
`i2pek12v`) crashed on `MPSGraph does not support tensor dims larger
than INT_MAX` because the 5M buffer × 17 planes × 81 cells (= 6.9B
elements) exceeded the MPS dim-product limit. Cell renamed to
`WL1-wave-lockstep-1p5M-buffer` and buffer dropped to 1.5M (matches Z,
max-under-INT_MAX). Per Jason: 5M was arbitrary; 1.5M is fine.

### WL1 run end — stopped at e1600 (~1h 18min wall, 2026-05-20)

Stopped by user after the run made it clear the new failure mode
wasn't going to self-correct.

**Final state:**
- e1600, 170,576 games, 12 race-drops handled cleanly across the run
- pl=1.706, vl=0.012, plies=10.2 (never regrew)
- Recent 5 evals: elo 776, 813, 620, 652, 871 — bouncing in 200-elo
  range with no net climb
- Final epoch: 2.9s (cycle time stable through the run)

**Run shape, summarized:**

| phase | epochs | what happened |
|---|---|---|
| spin-up | e1-100 | fast-attack collapse, plies 14 → 11, baselines pinned 0% |
| arc 1 climb | e146-499 | rapid elo climb to 1281, la4 hit sustained 52%, heuristic 88% |
| arc 1 trough | e500-605 | first consolidation, elo 1041 — *still* Z-e1854 class |
| regression band | e600-1600 | strength bouncing 620-1140, **la4 regressed to ~5%**, no net climb |

**Validated hypothesis (partial):** WL1 reached its arc-1 peak ~5-8×
faster per epoch than Z and exposed strength tiers Z barely touched
(la4 52% sustained at e499). The per-version-uniformity intervention
is *necessary* and operationally works.

**Refuted hypothesis (in this form):** removing per-version bias is
*not sufficient* for stable training. WL1 replaced Z's
800-1000-epoch consolidation arcs with high-frequency oscillation
(20-100 epoch wavelength) that doesn't retain prior peaks. The model
became more reactive, not more diverse.

**Reframe:** the laptop setup is missing the *in-flight version
diversity* that AZ-at-scale has by default (async publish lag, ~125k
concurrent games, batch 4096). WL1 actually *removed* the partial
version diversity Z had accidentally — one-version-per-tile is the
most-feedback-prone config possible.

**Next-run design:** [wiki/topics/wl2-scale-emulation-design.md](wiki/topics/wl2-scale-emulation-design.md)
stacks four cheap laptop-side emulations of AZ-scale properties:
EMA self-play weights, past-checkpoint opponent mix, worker poll
jitter, gradient accumulation 4×. Hypothesis: combined, they dampen
the WL1 oscillation enough that the model retains progress instead
of bouncing.

**Run artifacts:**
- wandb: `l8mbntcm` (run preserved, not deleted)
- workspace: https://wandb.ai/jasonyandell-forge42/gomoku?nw=ul0vliphj6x
- checkpoints: `sweep_runs/WL1-wave-lockstep-1p5M-buffer/checkpoints/`
- logs: `sweep_logs/WL1-wave-lockstep-1p5M-buffer/`
- commits anchoring the run: `bd28670` (cell), `c296e49` (stem_padding),
  `28f90bf` (wave-mode trainer), `0d2c106` (worker race-drop fix),
  `0ab3d9d` (native MCTS)

## WL2 live run log (2026-05-20, wandb 9wng4yu9)

Live tracking for the WL2 scale-emulation run launched 2026-05-20 ~23:00.
Same append-oriented structure as WL1's log section above. Cross-ref the
design at [wiki/topics/wl2-scale-emulation-design.md](wiki/topics/wl2-scale-emulation-design.md).

**Setup**
- wandb run: `9wng4yu9` (https://wandb.ai/jasonyandell-forge42/gomoku/runs/9wng4yu9)
- wandb workspace: https://wandb.ai/jasonyandell-forge42/gomoku?nw=ul0vliphj6x —
  add `9wng4yu9` to the run picker alongside `l8mbntcm` (WL1) and `sppjo3z5` (Z)
  for three-way overlays.
- cell: `WL2` (see scripts/run_sweep.py). Identical to WL1 except all four
  scale-emulation levers ON: `--ema-tau 0.99`, `--grad-accum-steps 4`,
  `--opponent-mix-recent 0.4`, `--opponent-mix-history 0.1`,
  `--opponent-mix-recent-window 100`, `--weights-poll-min-sec 2.0`,
  `--weights-poll-max-sec 8.0`.
- baseline: `WL1` (wandb `l8mbntcm`) — same recipe minus all four levers.
  Comparison shows whether the levers fix WL1's strength oscillation.

**Smoke (30 epochs, pre-launch):**
- All 4 lever signals fired cleanly
- Mix distribution: 12/43/45% (history/recent/self) — close to designed 10/40/50
- Cycle ~5s (vs WL1's 3.4s) — ~50% slowdown from grad-accum + past-checkpoint loads
- Training progresses normally (pl 4.31 → 3.50 over 30 epochs)

**Live milestones**

| epoch | wall   | pl   | vl    | plies | elo | h%  | la2% | la4% | note |
|------:|:------:|-----:|------:|------:|----:|----:|-----:|-----:|------|
| 1     | 00:08  | 4.36 | 0.94  | 33.9  | —   | —   | —    | —    | start, EMA-on baseline |

Will update as new evals land.

**Signals specifically to watch:**

- `train/ema_l2_distance` — should stay bounded; pathological divergence
  (rapid climb without bound) would mean the EMA isn't keeping up with the
  raw model and they're decoupling badly.
- `train/optimizer_steps_this_cycle` — should be ~1/4 of `train/steps_this_cycle`.
- `selfplay/plies_p90` — Jason's leading defense indicator; WL1 never moved
  past 11. The bigger question for WL2: does plies retention improve.
- `eval/vs_lookahead4_winrate` — the metric WL1 regressed badly on (52% → 5%).
  WL2 hypothesis: should hold any peak it reaches.

**Predictions to falsify:**
- WL2 makes Z-e1854-class strength faster than WL1 (e324 was WL1's mark).
  If WL2 is SLOWER to that mark, the levers cost convergence.
- WL2 retains la4 across consolidations (WL1's failure). If la4 oscillates
  the same way, the levers don't fix this failure mode.
- WL2's eval-to-eval elo variance is roughly half WL1's. If equal or worse,
  EMA isn't doing what we hoped.

### WL2 run end — stopped at e1200 (~1h 11min wall, 2026-05-21)

Stopped by user after the run made it clear the four scale-emulation
levers raised the ceiling but didn't break the underlying retention
failure.

**Final state:**
- e1200, 137,722 games, 11 race-drops handled cleanly
- pl=1.889, vl=0.012, plies=10.5 (selfplay plies never regrew)
- Last 6 evals (e944-e1151): elo bouncing 788-1071, heuristic 0-50%,
  la4 18-52%
- Final epoch: 3.0s — stable cycle time throughout

**Run shape, summarized:**

| phase | epochs | what happened |
|---|---|---|
| spin-up | e1-200 | fast-attack collapse, plies 13→11, baselines pinned 0% |
| slow climb | e200-370 | first heuristic crossing at e370 (15%) — 224 epochs later than WL1 |
| smooth ascent | e370-900 | la4 climbing 0→20→48→62; heuristic 15→70; pl/vl falling. **No bouncing** in this window — main early WL2 success |
| peak | e900 | la4=62% (PEAK, higher than WL1's 52%), h=70, elo=1197 |
| regression | e944-1200 | la4 drifted to 18, heuristic to 0-30 bouncing, elo 800-1071 |

**Validated hypothesis (partial):**
- EMA stabilization works — early WL2 trajectory was visibly smoother
  than WL1's bouncy chaos (heuristic 0→15→5→8 vs WL1's 30→0→15→0)
- Past-checkpoint mix raised the ceiling — la4=62% peak vs WL1's 52%
- Eval-time-vs-heuristic climbed 6s→17s through the climbing phase,
  indicating real defensive ability against a different style even
  while selfplay plies stayed flat at 11 (filed as a leading indicator
  in [wiki/topics/launch-sequence-runbook.md](wiki/topics/launch-sequence-runbook.md))

**Refuted hypothesis (in this form):**
- WL2 did NOT fix retention. la4 regression was 62→18 (44pp), almost
  identical magnitude to WL1's 52→5 (47pp). The four levers stabilized
  the trajectory and raised the ceiling but couldn't keep the model
  from forgetting hard-won deep play.
- Same depth-without-breadth pattern at trough as WL1 (heuristic 0%
  while la2 still at 42% at e1101 — model forgets simple kills first)

**Reframe:** even with EMA + past-checkpoint mix + jitter + grad-accum,
all model versions in this setup share the same opening lineage. Every
past checkpoint we mix in learned from positions reachable from the
"canonical" opening. Worker diversity doesn't help if the worker
diversity is "different brains thinking about the same opening."

**Next-run design:** WL3 = WL2 + K=2 random opening plies. Training
examples are not recorded for the random plies, so the model just sees
more diverse mid-game starting positions. Hypothesis: breaking opening
monoculture forces the model to learn defense/offense from positions
it can't reach from canonical play, which should improve retention.

**Run artifacts:**
- wandb: `9wng4yu9` (run preserved)
- workspace: https://wandb.ai/jasonyandell-forge42/gomoku?nw=cz8thj3cbh5
  (3-way WL2/WL1/Z overlay)
- checkpoints: `sweep_runs/WL2-scale-emulation/checkpoints/`
- logs: `sweep_logs/WL2-scale-emulation/`
- commits anchoring: `b582d37` (train: EMA + grad accum),
  `ded7728` (selfplay_worker: past-mix + poll jitter),
  `02c5fc3` (sweep: WL2 cell)

## WL3 live run log (2026-05-21, wandb TBD)

Live tracking for the WL3 run. WL2 + K=2 random opening plies (training
examples not recorded for the random plies). Same append-oriented
structure as the WL1/WL2 sections above.

**Setup**
- wandb run: `0o75gws5` (https://wandb.ai/jasonyandell-forge42/gomoku/runs/0o75gws5)
- wandb workspace: regenerated post-launch to include WL3 — open the URL
  printed by `python scripts/wandb_workspace.py` and select all four runs
  (`0o75gws5`, `9wng4yu9`, `l8mbntcm`, `sppjo3z5`) in the picker for the
  full WL3 → WL2 → WL1 → Z lineage.
- cell: `WL3` (see scripts/run_sweep.py). WL2 + `--random-opening-moves 2`.
- baseline: `WL2` (wandb `9wng4yu9`) — same recipe minus the random openings.
  Comparison shows whether opening diversity fixes the la4 retention failure.

**Smoke (30 epochs, pre-launch):**
- All four WL2 lever signals fired with random openings active
- Plies bumped +20% vs WL2 smoke at same range (22-26 vs 16-20)
- No crashes, no NaN, cycle 4-5s (comparable to WL2)

**Live milestones**

| epoch | wall   | pl   | vl    | plies | elo | h%  | la2% | la4% | note |
|------:|:------:|-----:|------:|------:|----:|----:|-----:|-----:|------|

Will populate as evals land.

**Predictions to falsify** (per WL2 close-out):
- WL3 retains la4 across consolidations (WL1: 52→5, WL2: 62→18). If WL3
  arcs as badly, opening monoculture wasn't the load-bearing cause.
- WL3 trajectory is smoother eval-to-eval than WL2 was post-peak.
- `time/eval_vs_heuristic_s` climbs past 20s sustained (WL2 plateaued
  around 17s before regression hit).

## Queued follow-up experiments (post-WL3, 2026-05-21)

Captured during WL3 run while the eval-distribution test was running.
Theme: WL3 keeps K=2 random openings ON the entire run. What if K is a
curriculum knob — high early, decayed away? That tests "does the model
internalize opening diversity, or does it need the crutch forever?"

To run after WL3 finishes (or if WL3 plateaus early). Listed in
no particular priority order — pick by what the WL3 close-out points at.

### Q1 — K=2 with mid-run anneal away

Start WL3-style with K=2 random openings. After some epoch threshold
(e.g. e1000 or e2000), turn random openings OFF and continue from the
existing checkpoint. Hypothesis: model has by then learned to handle
diverse positions, so removing the random crutch lets it focus on
canonical play without losing the breadth it gained.

Failure mode predictions: model collapses back to fast-attack within
a few hundred epochs after K → 0 (the past-checkpoint mix is no longer
producing diverse openings because the mixed-in past brains were
trained at K=2 too; once K=0, all selfplay starts converging on the
same opener again). If this happens, conclusion is "random openings
are load-bearing forever, not a curriculum."

### Q2 — Black-only random first move (K_black=1, K_white=0)

Just black's first move is uniform-random. White responds with MCTS as
normal. Asymmetric setup. Hypothesis: forces the model to handle
diverse openings as white (defender) while keeping black's opening
discipline intact. Cheaper than K=2 (only one random ply per game).

Implementation: needs a new flag — `--random-opening-moves` currently
applies to both sides. Could be `--random-opening-black-moves` /
`--random-opening-white-moves` or just `--random-opening-side {both,black,white}`.

### Q3 — Phased curriculum: K=2 / K=1 / K=0

Sequential schedule:
- e1-500:    K=2 random plies
- e500-1000: K=1 random plies
- e1000+:    K=0 (canonical)

Tests whether a decay path lets the model build robustness early then
specialize for canonical play late. The Q1 "binary anneal" is the
simplest version; this is the smoother version.

Implementation: schedule could be cell-level (a list of
(epoch_threshold, K) pairs) or runtime-only (`--random-opening-moves`
read from a file each cycle). Cell-level is cleaner for reproducibility.

### Q4 — Sweep K across full range

Run K ∈ {1, 2, 3, 4, 6} as independent cells, 500-1000 epochs each.
Cheap exploration of where the K sweet spot is. Combine with the
matched-distribution eval (from the 2026-05-21 ad-hoc test) so we
measure real strength on the trained distribution, not eval-distribution
mismatched strength.

### Cross-cutting note

All four of these queue items assume the matched-distribution eval
question (Jason's 2026-05-21 observation) is resolved. If WL3's slow
heuristic-crossing turns out to be an eval-distribution artifact,
the queue priorities should re-orient around what we learn about
training quality vs measurement quality.

### WL3 run end — crashed at e825 (NaN in MCTS policy, 2026-05-21)

Killed by infrastructure failure, not training quality. **Pre-crash WL3
was the strongest run in the WL series**:

- Peak la4 = 68% at e714 (higher than WL2's 62%, WL1 never sustained la4)
- All three baselines climbing TOGETHER (h50/la2:25/la4:38 at e515) —
  balanced strength profile, fundamentally different from WL1/WL2's
  single-baseline spike pattern
- Plies actually regrew: 13 → 18 across the run (first time in the
  WL series; WL1/WL2 stayed pinned at 11)
- elo climbed to 1434 at e801 — approaching Z's e3881 peak (1718) at
  ~5× fewer epochs

**Crash mechanism (NaN cascade — full diagnosis at
`$CLAUDE_JOB_DIR/wl3_nan_diagnosis.md`):**

1. At ~e825, native MCTS emitted a NaN visit-policy via `g.policy()` on
   the first worker (w6). Then w5, w3, w1 over the next ~5 minutes, then
   eventually all 8.
2. Each worker crashed in `_sample_action` (`gomoku/self_play.py:47`)
   because the old `if s <= 0` guard didn't catch NaN — NaN comparisons
   return False, so the divide propagated NaN, and `rng.choice` rejected.
3. Trainer barrier-stalled forever waiting on dead workers to publish to
   v825.
4. Recovery attempt #1 (during sleep window — applied band-aid at
   `_sample_action`, resumed from epoch0825.pt) failed because the
   band-aid didn't sanitize pi BEFORE storing it in trajectories. The
   NaN-laden pi went into the training buffer, and within 9 epochs the
   trainer logged `pl=nan` and corrupted all surviving checkpoints.
5. `keep_last_n=3` had already pruned the healthy e825 — corrupt
   checkpoints (e832/833/834) were all that remained.

**Two fixes in main (`c5049be` + `0557671`):**

- `_sample_action`: NaN guard via `not np.isfinite(s) or s <= 0` —
  prevents worker death.
- `_generate_games_native`: sanitize pi before storing in trajectories
  (NaN → 0 + renormalize; all-NaN → uniform) — prevents buffer poison.

Both fixes are safety nets, not root cause. The underlying MCTS NaN
source is under parallel investigation.

**What the WL3 wandb run preserved:**

- wandb: `0o75gws5` — eval/loss timeline through e825 is intact; the
  post-e825 NaN epochs are also in there but visually anomalous.
- Workspace overlay: https://wandb.ai/jasonyandell-forge42/gomoku?nw=816incbozdx
- Local artifacts moved to `sweep_runs/WL3-random-openings.dead-e825/`
  and `sweep_logs/WL3-random-openings.dead-e825/` for forensics.

**Lesson learned (filed in skill):** the "unattended infrastructure fix"
policy worked for the worker-death fix but missed the trajectory-recording
poisoning. Need to think about poisoning paths, not just process-death
paths, when triaging infrastructure bugs. Skill updated 2026-05-21.

## WL3.1 live run log (2026-05-21, wandb TBD)

Restart of WL3 with the two NaN guards in place (`c5049be` + `0557671`).
Identical cell config to WL3 (we know that's what produced WL3's
trajectory). Fresh wandb run (rather than rewind-resume) for clean
charts.

**Setup**
- wandb run: `i34ihwj9` (https://wandb.ai/jasonyandell-forge42/gomoku/runs/i34ihwj9)
- cell: `WL3.1` in `scripts/run_sweep.py`. Same as WL3 + new code paths
  (NaN guards in self_play.py).
- baseline: `WL3` (wandb `0o75gws5`) — the run that died at e825. WL3.1
  should reproduce the trajectory or come close (random seed + EMA
  randomness make exact reproduction unlikely, but the shape should
  match).

**Question under test:**
- Does WL3.1 reproduce WL3's pre-crash trajectory? (la4 peak ~e700,
  all baselines climbing balanced)
- Does the NaN crash recur? If yes, the band-aid lets us continue
  running (workers fall back to argmax), but native-MCTS root cause
  needs landing.

**Live milestones**

| epoch | wall | pl | vl | plies | elo | h% | la2% | la4% | note |
|---|---|---|---|---|---|---|---|---|---|
| TBD | | | | | | | | | populating from launch |


### WL3.1 relaunched at e92 with native MCTS C fix (2026-05-21 ~08:30)

The first WL3.1 (wandb `i34ihwj9`) ran cleanly to e92 with the Python
NaN band-aids in place. During that run, the background investigator
agent nailed the root cause of the original NaN:

**Native MCTS `policy(tau)` bug:** `pow(N, 1/tau)` was computed in
`double`, then cast to `float32` BEFORE normalizing. At τ=0.1 with a
concentrated child N ≥ ~7100, the cast overflowed to `+Inf`, then
`Inf/Inf` → NaN. The overflow requires long concentrated games — which
WL3 only produced once plies regrew past ~18 around e825. Below that
threshold, no NaN. That explains why WL3 ran 825 epochs before all 8
workers stumbled on it within minutes once the threshold crossed.

**Fix (commit `7c3e405`, plus tests in commit `397c784`):** keep the
sharpened scores in `double[]`, normalize in double, cast only the
final [0,1] probabilities to float. Plus an Inf-sum fallback to
argmax-tie. Regression test verifies the fix produces finite policies
at τ=0.1 with N=8000.

Python NaN band-aids (`c5049be` + `0557671`) remain as belt-and-
suspenders. With both layers, the failure mode is fully closed.

**WL3.1 relaunched as wandb `44cxzc9d`** with the rebuilt C extension.
Same cell config. Old artifacts at
`sweep_runs/WL3.1-random-openings-nanfix.preCfix-e92/`.

### WL3.1 paused at e1536 — WL4 launched as curriculum-decay experiment (2026-05-21)

WL3.1 (wandb `44cxzc9d`) paused at e1536 after demonstrating clear success:
peak la4=95% (single-eval) at e1105, sustained la4 60-90% across e931-1413,
heuristic 100% sustained e1123-1157, elo holding 1400-1700, plies hit 27.
Strongest WL-series state by every measure. Per Jason's call: take the
training wheels off and see if it breaks through further OR collapses.

**Pause mechanics:**
- WL3.1 artifacts preserved: `sweep_runs/WL3.1-random-openings-nanfix.paused-e1536/`
- e1536 snapshotted aside: `$CLAUDE_JOB_DIR/wl3.1_e1536_latest.pt` (8.2G — model + EMA + buffer)
- Slim version: `$CLAUDE_JOB_DIR/wl3.1_e1536_slim.pt`
- Can resume WL3.1 with K=2 anytime if needed

## WL4 live run log (2026-05-21, wandb 44cxzc9d continued)

WL3.1 + `random_opening_moves=0`. Resumes from WL3.1 e1536.

**Setup**
- wandb run: `44cxzc9d` continues (wandb resume restored from checkpoint;
  WL4 epochs are e1537+. The chart is a single trajectory with K=2 then
  K=0 at step 1537 — clearer than two separate runs to overlay.)
- cell: `WL4` in `scripts/run_sweep.py`. Same as WL3.1 with one change:
  `random_opening_moves=0`.
- parent state: WL3.1 e1536 (la4 60-90% sustained, heuristic 100% peaks,
  elo 1400-1700, plies 20-27)
- model identity: same neural weights, same EMA, same buffer, same
  optimizer state — only the *future* games will have canonical openings

**Hypothesis under test (Jason 2026-05-21):**
WL3.1 has had 1500+ epochs of K=2 random-opening training and is at
"established" by every measure. If diversity is now baked into the
weights, removing random openings should unlock canonical-line depth
that random plies were rate-limiting. If diversity is permanent
training infrastructure, removing it should cause rapid regression
toward attack-only collapse.

**Live milestones**

| epoch | wall | pl | vl | plies | elo | h% | la2% | la4% | note |
|---|---|---|---|---|---|---|---|---|---|
| 1536 | 02:18 | 1.076 | 0.096 | 29.8 | — | — | — | — | resume point (WL3.1 final) |

**Predictions to falsify**
- "Diversity baked in" hypothesis: WL4 should continue or improve from e1537,
  with plies pushing further past 25 toward the defense regime
- "Diversity load-bearing" hypothesis: heuristic drops below 70% sustained
  within ~200 epochs of K=0; if it falls below 50% the run is reverting
- Specifically watch eval-to-eval variance: if it widens after K=0, the
  model is bouncing again (WL1 failure mode returning)

**Recovery if it collapses**: resume from `$CLAUDE_JOB_DIR/wl3.1_e1536_latest.pt`
with `--random-opening-moves 2` to get back to WL3.1 trajectory.

### WL4 loss-floor bounce read (2026-05-21, W&B snapshot through ~e3776)

Jason observed the post-switch loss curve: `random_opening_moves=2` was removed
at e1537, loss bumped, then descended to a much lower floor and began bouncing
there. This has happened before in this repo: the model improves against fixed
external baselines while loss "bounces off the bottom."

W&B pull for run `44cxzc9d` supports the healthy interpretation for now:

| window | loss/total | loss/policy | loss/value | selfplay/plies_mean | external read |
|---|---:|---:|---:|---:|---|
| e1450-e1536, pre-switch | 1.21-1.37 | 1.06-1.24 | 0.080-0.098 | 14-31 | elo 1455-1592 |
| e1537-e1900, K=0 bump | 1.35-1.57 | 1.22-1.45 | 0.070-0.091 | 13-23 | elo 1385-1918 |
| e1901-e2400, absorb | 0.77-1.55 | 0.66-1.43 | 0.070-0.090 | 14-50 | h mean 84%, la2 mean 93% |
| e2401-e3000, lower floor | 0.39-0.86 | 0.34-0.73 | 0.027-0.102 | 28-69 | e2996: elo 1760, h 88%, la2 100%, la4 90% |
| e3001-e3773, floor bounce | 0.56-0.97 | 0.48-0.83 | 0.048-0.117 | 24-59 | e3772: elo 1519, h 75%, la2 100%, la4 62.5% |

Interpretation: this is not the WL3 NaN bug shape. Loss is finite, value loss
stays small but not poisoned, plies have regrown into the long-game regime, and
fixed external baselines remain broadly strong/noisy rather than collapsing.
The likely mechanism is moving-target AlphaZero training: `policy_loss` is
cross-entropy against the MCTS visit distribution, and the MCTS target
distribution changes when the self-play/search system discovers new canonical
lines after K=0.

Filed maintained synthesis at
[wiki/topics/loss-floor-bouncing.md](wiki/topics/loss-floor-bouncing.md). Current
triage rule: low-floor bouncing is healthy if plies and fixed baselines hold;
suspect a bug only with NaN/Inf, worker death, replay-buffer poisoning,
short-game collapse, or sustained multi-window external regression.

### Source-backed next-run lessons from the loss-bounce literature pass

Web/literature pass did NOT find a named "loss bounces off the bottom"
AlphaZero phenomenon. It DID find strong support for the mechanism: AlphaZero
policy loss is against a moving MCTS/self-play teacher, and small-scale
self-play is sensitive, noisy, and vulnerable to coverage/forgetting problems.
Important correction: published AlphaZero used 5000 TPUs for self-play, not
GPUs; the "scale smooths the wrinkles" point still stands.

Next-run implications, filed in
[wiki/topics/loss-floor-bouncing.md](wiki/topics/loss-floor-bouncing.md):

1. Add a frozen validation archive and log policy CE/KL on fixed positions
   (old WL3/WL4, heuristic-loss, lookahead-loss, long-defense, high-KL,
   canonical-opening positions). This separates true regression from target
   distribution movement.
2. Split policy loss into `H(pi_mcts)` and `KL(pi_mcts || p_net)`, plus net
   entropy. Total loss alone hides whether the teacher changed shape or the net
   failed to track it.
3. Keep the smoothing stack: EMA self-play, past-checkpoint mix, grad
   accumulation, careful replay age, and fixed external baselines are our
   laptop-scale substitutes for AlphaZero's giant self-play/batch/replay scale.
4. Prefer structured start diversity over crude permanent opening randomness:
   if WL4 needs a new lever, try 10-25% archive-start games from interesting
   states while keeping 75-90% canonical K=0 self-play.
5. Consider a KataGo-style policy-target pruning/downweighting ablation so
   exploration visits do not automatically become policy labels.
6. Instrument role asymmetry before architecture changes: evals by color, value
   error by side-to-move/ply bucket, and fixed-probe loss by role.

Working next-run shape: WL4 recipe + better diagnostics first. If behavior
needs changing, the cleanest first lever is archive-start diversity, because it
targets the documented coverage problem without putting random openings back in
as permanent infrastructure.

### WL4 plateau-end — stopped at e4024 (~5h 39min wall, 2026-05-21)

Stopped cleanly per Jason's call: WL4 reached the "healthy lower-floor-
bouncing" steady state described in
[wiki/topics/loss-floor-bouncing.md](wiki/topics/loss-floor-bouncing.md).
Letting it run further would mostly cycle through more local
canonical-line discoveries without changing the floor.

**Final state:**
- e4024, 457,889 games, ~2500 epochs of K=0 training (from e1500 resume)
- pl=0.604, vl=0.090, plies=40.0 (last epoch line)
- last 8 evals (e3783-3874): h locked 60-75, la2 mostly 100% (one 50% dip),
  la4 48-82%, elo 1262-1620
- 0 errors, 0 NaN warnings, 0 worker deaths in 5h 39min

**Run shape, summarized:**

| phase | epochs | story |
|---|---|---|
| spin-up | e1501-1700 | K=0 transition: plies dropped 21→14, model adapts to canonical openings |
| stable plateau | e1700-2200 | locked WL3.1-strength: h75-100, la4 60-90, elo 1455-1682 |
| regime change | e2329-2401 | plies surged 20→47, pl dropped 1.23→0.67, **elo hit 1841 ATH** (Z lifetime peak was 1718) |
| deep defense | e2401-3200 | plies sustained 36-64, la4 perfect at e3148, la2 sustained 100%, elo 1290-1760 |
| lower-floor plateau | e3200-4024 | losses bouncing 0.5-0.7 (healthy per article), external baselines steady, model cycles through local discoveries |

**Validated:**
- Random opening diversity is **necessary but not permanent infrastructure**.
  WL3.1 (K=2) built the diverse representations; WL4 (K=0) confirmed they
  persist when K→0, plus unlocked canonical-line depth that K=2 was
  rate-limiting (plies 21→47 mean, elo 1465→1841 ATH).
- WL series ATH elo=1841 (WL4 e2401) — 123 elo above Z's lifetime peak.
- Defense regime is genuinely achievable on 9x9 small-model: la4=100%
  at e3148, la2 sustained 100%, plies past Z's e5000 endpoint.

**Failure mode that didn't happen:**
- The "diversity is permanent training infrastructure" hypothesis was
  refuted. K=0 didn't cause regression toward attack-only.
- The WL1 oscillation pattern didn't return. Loss bouncing in late WL4
  is structured (per loss-floor-bouncing.md) not chaotic.

**Run artifacts:**
- wandb: `44cxzc9d` (started as WL3.1, continued through WL4 e1501-4024;
  K=2→0 transition visible at step 1537)
- WL3.1 paused state: `sweep_runs/WL3.1-random-openings-nanfix.paused-e1536/`
- WL4 final state: `sweep_runs/WL4-no-random-openings.plateau-e4024/`
- latest.pt preserved in the WL4 paused dir (8.2G, model+EMA+buffer)
- commits anchoring: `c5049be` `0557671` (Python NaN guards),
  `7c3e405` (native MCTS C fix), `e8e0cef` (worker_weights resume bug),
  `a88749d` (WL4 cell)

**Next-run shape (deferred to Jason design conversation):**
Per `wiki/topics/loss-floor-bouncing.md` "Candidate Next-Run Shape" —
diagnostics first (fixed validation archive, H/KL split, per-color
metrics), then behavioral lever (archive-start diversity: 10-25% of
self-play games from curated trouble states). NOT a simple parameter
tweak — needs design + code work before launch.

---

## 2026-05-21 — WL5 launched (diagnostics + Go-Exploit archive-start)

**wandb:** `o6cbjfnr`
**cell:** `WL5-diagnostics-archive-start` (`scripts/run_sweep.py`)
**resume parent:** WL4 `latest.pt` at e4024 (stripped of `wandb_run_id` so WL5
gets its own clean timeline)
**design:** [wiki/topics/wl5-diagnostics-archive-start-design.md](wiki/topics/wl5-diagnostics-archive-start-design.md)

**What's different vs WL4:**
1. **Validation archive scoring** every `eval_every` cycles against a
   frozen 1400-position archive mined from WL4 (7 buckets, 200 each;
   see [wiki/topics/mining-validation-archives.md](wiki/topics/mining-validation-archives.md)).
   Logs `val/policy_ce`, `val/policy_kl`, `val/value_mse`, `val/policy_acc`
   plus per-bucket variants. Separates *target-distribution noise* from
   *learning gap* across cycles.
2. **H/KL decomposition** of policy loss per train step: logs
   `train/policy_target_entropy`, `train/policy_net_entropy`,
   `train/policy_kl`. The central interpretive question from
   [loss-floor-bouncing.md](wiki/topics/loss-floor-bouncing.md).
3. **Per-color and per-ply-bucket metrics**:
   `train/policy_ce/side_{0,1}`, `train/value_mse/side_{0,1}`, plus
   ply buckets `[0,10)`, `[10,25)`, `[25,60)`.
4. **Archive-start lever (15% frac)**: each game-start in workers rolls
   U(0,1); if < 0.15 AND archive loaded, initialize the native MCTS game
   state from a curated trouble position instead of empty board. Go-Exploit
   pattern (Trudeau & Bowling 2023). Recorded with `archive_started` per
   wave for transparency.

**Levers preserved from WL4:** EMA τ=0.99, past-mix 0.4/0.1, poll jitter
2-8s, grad-accum 4×, K=0 (no random opening plies).

**Bugs hit + fixed before stable launch:**
1. **High_kl bucket had ply=0 on every position** — replay buffer loader
   zero-fills missing tags on backward-compat path; the mine script
   inherited the zeros. Archive-start games then had move_count=0 but
   ~50 stones on board → MCTS's `move_count >= N_ACTIONS` terminal gate
   never fired → C extension's `select_action` returned action 0 on
   full board (no legal moves) → `state_apply` raised "illegal move 0
   on occupied square".
2. **C-level safety net added**: `init_node_fields` now marks any node
   with `legal_count == 0` as terminal (draw). Defends against any future
   bad-input class hitting this same invariant.
3. **C-level select_action default fixed**: best_action defaults to the
   first legal action instead of 0; defends against pathological NaN W
   propagation that could leave all scores as -inf/NaN.
4. **Evaluator sanitize**: `gomoku/mcts.py` evaluator now `nan_to_num`s
   priors and values before returning. Defense in depth against
   pathological forward passes on archived positions.

Fixes anchored in commit `dc8c38b`.

**Initial state (epoch 4001-4007):**
- pl 0.63-0.65, vl 0.10, plies 32-46 (matches WL4 defense regime)
- ~9s/cycle (faster than WL4's 16-24s — possibly EMA warmup, or
  resumption efficiency)
- `archive_started` ≥1 firing on multiple workers (w4 batch 33 had 2;
  w7 batch 35 had 3) — confirming 15% archive-start mix
- past-mix: `mix=self(current)`, `mix=recent(epochNNNN)`,
  `mix=history(epochNNNN)` all observed

**Mining the archive** (replicate this for any future archive):
```bash
PYTHONUNBUFFERED=1 GOMOKU_DEVICE=mps PYTORCH_ENABLE_MPS_FALLBACK=1 \
  python -u scripts/mine_validation_archive.py \
    --wl4-checkpoint sweep_runs/WL4-no-random-openings.plateau-e4024/checkpoints/latest.pt \
    --output archives/wl5_validation_v1.pt \
    --target-per-bucket 200 --mcts-sims 200 --batch-games 64 \
    --wave-size 8 --max-rounds 3
```
Wall: ~13 min. Buckets: hard_kl_{selfplay,heuristic,lookahead2,lookahead4}
(top-K positions by `KL(p_net || pi_mcts)` per opponent),
`long_defense` + `canonical_opening` from self-play, `high_kl` from the
WL4 buffer top-K by KL. The "lost games" buckets in the original design
were replaced with `hard_kl_*` — saturated models don't lose to weak
baselines, but they always have positions where the prior disagrees
with searched policy.

**What to watch:**
1. **`train/policy_kl` vs `train/policy_target_entropy`**: if KL floors
   while H bounces → target-distribution noise (article's prediction
   confirmed). If KL doesn't close → fittability gap (capacity bet
   for next run).
2. **`val/*` per-bucket trends**: long_defense val_ce dropping faster
   than canonical_opening would confirm archive-start is unlocking
   the deeper-state coverage gap.
3. **`selfplay/plies_mean` past 50**: would push past WL4's defense ceiling.
4. **External baseline trajectory vs WL4 ATH 1841 elo**: any sustained
   move past 1841 would validate the lever quantitatively.

**Failure modes to abort on:**
- Sustained `selfplay/plies_mean` drop below 25 — fast-attack collapse.
- `val/policy_ce` regression on canonical_opening — the diagnostic
  archive is supposed to be a stable reference; regression there means
  the model is forgetting basic play.
- Any NaN/Inf reappearance in metrics.

## 2026-05-21 — Post-native perf pass: eval-only Conv+BN fusion

Question from Jason: now that native MCTS exists, how much farther can this
specific M5 Max go, with no implementation path off the table?

Current machine read:
- Hardware: M5 Max MacBook Pro, 18 CPU cores (6 efficiency + 12 performance),
  40-core GPU, 48 GB unified memory, MPS backend.
- A live WL5 run was active during inspection: trainer + 8 self-play workers
  + eval worker. One-off generator benches in this state are contention-noisy.

Evidence:
- `sample` on live worker `w0` showed the main hot stack inside
  `native_search_batch`, but almost all sampled time flowed through
  `_mcts_native.c:call_evaluator` into PyTorch/MPS graph execution. Repeated
  frames landed in MPS BatchNorm / graph setup rather than C tree traversal.
- Direct small-model forward bench under the same live load:

| batch | unfused eval | Conv+BN fused eval |
|---:|---:|---:|
| 8 | 1.948 ms | 0.858 ms |
| 32 | 2.125 ms | 0.967 ms |
| 64 | 2.136 ms | 1.008 ms |
| 128 | 2.130 ms | 1.139 ms |
| 256 | 3.335 ms | 2.468 ms |
| 512 | 10.314 ms | 4.750 ms |

Output parity check: max absolute diff was ~1.8e-7 policy, ~1.9e-8 value
on a random CPU batch.

Change:
- Added `fuse_model_for_inference(model)` in `gomoku/model.py`. It mutates
  eval-only models by folding Conv2d + BatchNorm2d into Conv2d and replacing
  BN modules with `Identity`.
- Applied it only after checkpoint loads in eval-only surfaces:
  `selfplay_worker.py`, `eval_worker.py`, parallel eval workers in
  `eval.py`, `match.py`, `cli.py`, and `web/server.py`.
- `scripts/perf_microbench.py` now uses fused eval by default and has
  `--no-fuse-eval` for A/B checks.

Interpretation:
- The next cheap speed layer is not more Python MCTS rearrangement. Native
  search moved the bottleneck to tiny-model MPS graph overhead, especially BN.
- Fusion is a safe eval-only optimization because trainer models remain
  unfused and checkpoints stay normal.
- Generator-level `perf_microbench` during live WL5 contention was noisy, so
  the direct forward timings are the clean evidence. Re-benchmark production
  8w x 8g after a restart with fused workers before banking a new wall-clock
  multiplier.

Verification:
- `python -m py_compile gomoku/model.py gomoku/selfplay_worker.py
  gomoku/eval_worker.py gomoku/eval.py gomoku/match.py gomoku/cli.py
  web/server.py scripts/perf_microbench.py`
- `pytest tests/test_model.py tests/test_native_mcts.py tests/test_mcts.py -q`
  passed (`17 passed`).
- `pytest -q` passed.
- CPU smoke:
  `python scripts/perf_microbench.py --device cpu --size tiny --games 2
  --n-simulations 2 --wave-size 1 --max-plies 2 --repeats 1 --warmup 0`.

### Production verification — hot-restart of WL5 workers (2026-05-21 21:35)

The fusion commits landed (`4f21cdd`, `aff6969`) ~2h after WL5 launched
(wandb `o6cbjfnr`, started 19:05:28), so the running self-play workers
had imported the un-fused code into memory. To convert the microbench
speedup into a production measurement on the same in-flight run, we
hot-restarted the 8 self-play workers in place while the trainer (and
WL5 replay buffer + wandb timeline) kept going.

**Hot-restart procedure (reusable):**
1. Snapshot baseline cycle stats in trainer log (`grep '^epoch ' .../trainer.log`).
2. SIGTERM canary worker w0 (PID from `ps`); wait 5s for exit.
3. Respawn w0 with identical args extracted from `ps`, redirecting stdout to
   `sweep_logs/<run>/w0.log` (append). Wait ~30s; verify new PID alive
   and worker log shows `wave batch` lines on a fresh model version.
4. If canary healthy, SIGTERM remaining 7 in parallel; respawn each with
   the same args + matching `--seed`/`--worker-id`. Trainer stalls ~30s
   during one wave barrier, then resumes.
5. Wait ~25 epochs for steady state; compare against pre-restart baseline.

**Microbench A/B (live with training contention):**

| run | median wall | aug pos/s |
|---|---:|---:|
| `--no-fuse-eval` | 1.036s | 710 |
| fused (default) | 0.718s | 1047 |
| **ratio** | — | **1.47×** |

(`perf_microbench --checkpoint .../worker_weights.pt --games 8 --n-simulations 400
--wave-size 32 --max-plies 12 --repeats 5 --warmup 1`)

**Production cycle stats — pre/post-restart (n=26 / n=25 epochs each):**

|   | pre (5020-5045) | post (5055-5079) | ratio |
|---|---:|---:|---:|
| wall/epoch | 8.58s | 8.74s | 1.02× (flat) |
| gen/epoch | 5.22s | 4.50s | 0.86× |
| train/epoch | 2.89s | 3.58s | 1.24× |
| new games/epoch | 113 | 148 | 1.31× |
| SGD steps/epoch | 86 | 86 | 1.00× |
| plies mean | 41.9 | 32.8 | 0.78× |
| pl | 0.590 | 0.734 | 1.24× |
| **games/sec on gen** | **21.6** | **33.0** | **1.53×** |
| games/hour | 47k | 61k | 1.29× |
| SGD steps/hour | 36k | 35k | 1.00× |
| aug-pos/hour | 15.9M | 16.1M | 1.01× |

**Interpretation:**
- The **gen-side games/sec speedup is 1.53×**, slightly *above* the
  microbench's 1.47×. Microbench ran against live WL5 contention; the
  post-restart workers don't compete with a bench, so the production
  number is cleaner upside.
- Per-batch worker logs show the same: pre-fusion v5044 8-game wave
  batch in `w0.log` was 3.3s; post-fusion v5046 8-game wave was 2.4s
  (~1.38×, confounded by colder caches in the new process).
- **Epochs/hour is unchanged** because the trainer scales SGD steps
  with new positions (`--sgd-per-game 1.0`, `--sgd-per-position 0.0025`),
  so faster generation just feeds the trainer more games per cycle.
- **aug-pos/hour is also ~flat** — but this is confounded: the model
  hit an absorption rough patch right at the restart (`pl 0.59 → 0.73`,
  plies 41.9 → 32.8). Shorter games = less work per game, eating the
  perf win on a positions-per-second basis. Once absorption settles
  and plies climb back, the throughput win should compound.
- The clean signal is the **gen-side games/sec ratio** because it
  isolates worker performance from trainer scaling and model state.
  Production confirms the microbench prediction.

**Caveat on hot-restart timing:** restart was deliberately mid-WL5,
which means the post-restart epoch window overlaps the archive-start
absorption phase. A cleaner perf measurement would be: do this
post-restart again after WL5 reports out, with the model in a
stable-plies regime. The 1.53× games/sec on gen will hold; the
end-to-end aug-pos/hour ratio will be more interpretable.

### Aggressive engine scout — Core ML vs MPS overlap (2026-05-22)

Implemented a first scout for the "self-play on ANE/Core ML, trainer on
MPS, eval on CPU" idea:

- `gomoku/coreml_evaluator.py`: Core ML evaluator matching the existing
  `evaluate_planes` boundary, plus checkpoint/model export helpers.
- `scripts/aggressive_engine_scout.py`: JSON-emitting harness for eval
  latency and trainer-overlap pressure.
- Receipt: `sweep_logs/aggressive-engine-scout-2026-05-22.json`.

Run shape: fresh `small` model, `stem_padding=1`, fused eval, batches
8/32/64/128, Core ML FP16 + INT8, `CPU_ONLY` + `CPU_AND_NE`, trainer-like
MPS batch 256.

Key numbers:

| lane | raw eval b128 median | b128 pos/s |
|---|---:|---:|
| PyTorch MPS | 2.94 ms | 43.5k |
| Core ML FP16 CPU_AND_NE | 9.05 ms | 14.1k |
| Core ML FP16 CPU_ONLY | 8.47 ms | 15.1k |
| Core ML INT8 CPU_AND_NE | 9.04 ms | 14.2k |
| Core ML INT8 CPU_ONLY | 9.05 ms | 14.1k |

Overlap probe: trainer baseline median step 13.94 ms. A competing
PyTorch/MPS eval process slowed trainer steps to 2.65× baseline. Core ML
pressure lanes were much gentler: 1.13-1.32× baseline.

Interpretation:
- Core ML is not yet a raw eval-latency win over fused PyTorch/MPS for the
  tiny/small model.
- INT8 weight quantization works mechanically but did not help latency in
  this path.
- The full-send idea remains alive as an **engine isolation** bet, not a
  "Core ML is faster per call" bet. Next measurement should wire Core ML
  into production-shaped self-play and compare end-to-end overlap with the
  trainer.

### WL5 phase-1 close — un-fused-workers era, e4001 → e5051 (2026-05-21, ~2.5h wall)

Closing the pre-fusion era of WL5 as a discrete chapter. WL5 phase 1 ran
from launch (`o6cbjfnr` start at 19:05:28) through e5051, at which point
the 8 self-play workers were hot-restarted to pick up the Conv+BatchNorm
eval fusion that landed in commit `4f21cdd` ~2h after launch. The wandb
run, trainer process, replay buffer, and design hyperparameters are all
**continuous** across the boundary — what changed at e5052 is purely the
self-play inference path (1.53× games/sec on gen). That regime change is
big enough that pre- vs post-boundary numbers are not directly comparable
on per-epoch metrics, so phase 1 deserves its own framing.

**Phase 1 final state (e5051):**
- 1051 epochs of WL5 training (resumed from WL4 `e4024` checkpoint)
- 123,453 games generated; replay buffer at 1.5M (full ring)
- Last 10 evals (e4969-e5048): elo 1424-1738 (mean 1551), plies 30.4-46.9
  (healthy defense range)
- Phase-1 elo peak: **1784 at e4035** — landed only 34 epochs after
  resume, i.e. it was still effectively the WL4 strength carrying over.
  The lever's absorption shock arrived shortly after.
- 0 NaN, 0 worker deaths, 0 barrier stalls

**Phase 1 stats (n=1051 epoch lines):**

| metric | value | vs WL4 plateau-end |
|---|---:|---|
| wall/epoch (median) | 8.70s | similar |
| pl (mean) | 0.673 | **0.604 → 0.673, +11%** (absorption signature) |
| vl (mean) | 0.082 | similar (0.090 → 0.082) |
| plies (mean) | 38.6 | similar (40.0 → 38.6) |
| elo (mean) | 1498 | regression from WL4 ATH 1841 |
| epochs/hr | 414 | similar |

**Run shape, summarized:**

| phase | epochs | story |
|---|---|---|
| resume bleed-over | e4001-4035 | residual WL4 strength: elo hit 1784, then absorption shock began |
| absorption descent | e4035-e4500 | pl climbed 0.60 → 0.75, elo dropped to 1300s, plies stayed 30-50 |
| oscillating absorption | e4500-e5051 | pl bouncing 0.55-0.84, elo 1159-1738 on 88 evals (no clear trend), plies sustained 30-50 |

**What was validated (as a lever-introduction shape):**
- **Archive-start at 15% does not crash the run.** No NaN, no worker
  death, no fast-attack plies collapse over 1051 epochs of operation.
- **Diagnostic streams are populating cleanly.** Validation archive
  per-bucket CE/KL recorded every cycle; H(pi_mcts) and
  KL(pi_mcts || p_net) split logged; per-color and per-ply-bucket
  metrics logged. The data the loss-floor-bouncing article asked for is
  now collected at scale.
- **The absorption-phase shape predicted by
  [[feedback-absorption-phase]] held.** ~1000 epochs of external-baseline
  regression after the new lever flipped on, plies in healthy range
  throughout, pl bouncing above prior floor but without collapse. WL3.1
  → WL4 (K=2 → K=0) had the same shape over ~500 epochs.
- **The C MCTS robustness fixes (commit `dc8c38b`) held.** No
  "illegal move on occupied square" crash recurrence in 1051 epochs.

**What didn't happen (failure modes ruled out for this phase):**
- No fast-attack collapse (plies never sustained below 25).
- No worker death from archive-start position oddities.
- No NaN reappearance.
- No buffer poisoning shape (vl held steady, didn't trend down to trivial).
- No barrier stall in the wave-mode coordinator.

**What didn't (yet) happen that we were hoping for:**
- elo did not push past WL4 ATH 1841 (peak 1784 was residual WL4 strength,
  not a phase-1 advance).
- Per-bucket val/policy_ce trends to confirm the lever is unlocking
  long_defense have not been audited yet (data collected, analysis
  pending).

**Run artifacts:**
- wandb: `o6cbjfnr` (continuous, includes phase 2). For phase-1-only
  analysis, filter step ≤ 5051.
- Trainer log: `sweep_logs/WL5-diagnostics-archive-start/trainer.log`
  (phase 1 = epoch lines e4001 → e5051, ~1051 lines from start).
- Worker logs: `sweep_logs/WL5-diagnostics-archive-start/w[0-7].log`
  (phase 1 = entries before the restart timestamp ~21:35).
- Checkpoint history rolls; `keep-last-n=3` means e5051 specifically is
  no longer on disk. Phase 1 is fully captured in wandb.
- Resume source preserved at
  `sweep_runs/_resume_from/wl4_e4024_fresh_wandb.pt` (8.2 GB,
  model+EMA+buffer from WL4).
- Commits anchoring phase 1: `7bbe70a` (trainer instrumentation),
  `cc6aa4e` / `8879254` (archive-start lever + cell wiring), `dc8c38b`
  (mining + C robustness fixes), `d8cc58f` (hard_kl mining strategy +
  recipe wiki), `bf130d0` (INST trainer machinery from worktree),
  `aff6969` / `4f21cdd` (Conv+BN fusion — landed mid-run, kicks off
  phase 2 once workers restart).

**Phase 2 (what's running now, e5052+):**
- Same wandb run, same trainer, same buffer, same design.
- 8 self-play workers replaced in place with fused-inference processes
  (PIDs 98749, 99565-99571).
- Gen-side throughput up 1.53× (measured over n=25 post-restart epochs);
  trainer absorbs the surplus as more games per cycle, so
  epochs-per-hour is ~unchanged but games-per-hour is up ~1.29×.
- Reading phase 2 metrics: pre-fusion per-epoch absolute numbers
  (games/epoch, training-steps/epoch) are not directly comparable to
  phase 1; epoch counter advances at the same rate but each epoch
  embeds more work. For ratio comparisons across the boundary, use
  rates per wallclock hour or per game, not per epoch.
- Phase 2 stop condition unchanged: run to e9000, push only on collapse
  / NaN / new ATH / canonical-opening regression.

**Cross-refs:**
- [wiki/topics/wl5-diagnostics-archive-start-design.md](wiki/topics/wl5-diagnostics-archive-start-design.md) — WL5 design as launched.
- [wiki/topics/loss-floor-bouncing.md](wiki/topics/loss-floor-bouncing.md) — interpretive frame for phase 1's loss shape.
- Production verification of the fusion that demarcates phase 1 from
  phase 2 lives a few sections above ("Production verification — hot-restart
  of WL5 workers (2026-05-21 21:35)").

### WL5 phase-2 close — fused-workers era, run-end at cap e10200 (2026-05-22, ~15.8h wall)

Closing WL5 phase 2 as the second discrete chapter of the same wandb run
(`o6cbjfnr`). Phase 2 ran across two segments separated by a ~1h pause
for parallel perf benches:

- **Segment A:** e5052 → e5249 (~2.6h, post-fusion-restart 2026-05-21
  19:35 → voluntary stop 22:12 to free MPS).
- **Segment B:** e5201 → e10200 (~13.3h, resume 23:08 from the e5200
  buffer snapshot, hit cap at 2026-05-22 12:25).

Segment B re-started from the same wandb run id; the e5201-e5249 overlap
spans 49 epochs of weights that were rolled back to the e5200 buffer
snapshot and re-derived from there. The trade was deliberate: lose 49
epochs of weight drift to keep the 1.5M-position buffer warm.

**Phase 2 final state (e10200):**
- Cap-reached normal exit; wandb finalized cleanly, last checkpoint
  written to disk: `sweep_runs/WL5-diagnostics-archive-start/checkpoints/epoch10200.pt`
- 724,334 games generated across phase 2 segment B (total run to date
  1,325,249 games)
- Replay buffer full ring at 1.5M (cycled ~28× across the run lifetime —
  see [[project-buffer-undersized]] in session memory)
- 0 NaN, 0 worker deaths, 0 barrier stalls, 0 tracebacks across 5000
  epochs of segment B

**Phase 2 segment B stats (n=5000 epoch lines, e5201 → e10200):**

| metric | value | vs phase-1 mean | vs phase-2 reference (e5051-end) |
|---|---:|---|---|
| pl (mean) | 0.621 | 0.673 → 0.621, **-7.7%** | 0.69 → 0.62, **-10%** |
| pl (min) | 0.423 | — | new low |
| vl (mean) | 0.073 | 0.082 → 0.073 | 0.077 → 0.073 |
| vl (min) | 0.042 | — | new low |
| plies (mean) | 41.5 | 38.6 → 41.5 | 38 → 42, longer defenses |
| plies (max) | 59.9 | — | longest games of the run |
| epochs/hr | ~376 | 414 → 376 (more SGD work/cycle absorbs gen surplus) | — |

**Eval scoreboard (523 eval cycles in segment B):**

| metric | value |
|---|---:|
| elo min/median/max | 1168 / 1543 / **1738** |
| la4 min/median/max | 25% / 82% / 100% |
| la2 min/median/max | 38% / 90% / 100% |
| h   min/median/max | 25% / 75% / 100% |
| **Best elo: 1738 at e5477** | la4=100%, la2=100%, h=75% |

WL4 ATH **1841** was not broken. Best elo (1738) landed ~5 hours into
segment B, then drifted to mid-band for the remaining ~8 hours. The eval
oscillation was wide: la4 single-cycle range 25%-100% across just 20
games is mostly sample noise, but the underlying training metrics show
the model genuinely moved into a lower-loss / longer-defense regime than
phase 1 occupied.

**Run shape, summarized:**

| sub-phase | epochs | story |
|---|---|---|
| segment A (post-fusion) | e5052-e5249 | brief fusion-validation run before perf-bench pause |
| segment B resume warm-up | e5201-e5500 | recovered from buffer snapshot; elo briefly hit 1738 (this run's ATH) |
| segment B main body | e5500-e9500 | wide oscillation: pl swung 0.42-0.80, vl 0.04-0.11, plies 25-60, elo bouncing 1168-1699 across hundreds of evals |
| segment B run-out | e9500-e10200 | metrics settled near phase-2 centroid (pl 0.65, vl 0.075, plies 40), elo mid-band 1414-1699, no late breakout |

**What got validated in phase 2:**
- Conv+BN fusion is stable in production over 5000+ epochs (no
  weight-drift, no eval/MPS path divergence, no thermal weirdness)
- The lever-set (archive-start 15%, EMA 0.99, past-mix 0.4/0.1,
  grad-accum 4×) is well-tolerated at scale — no degradation modes that
  hadn't already appeared in phase 1
- Wave-mode coordinator survives a 5000-epoch cap-reach without barrier
  stalls or worker-mismatch events

**What didn't (yet) happen:**
- WL4 ATH 1841 was not broken. Best 1738 = WL4-strength-equivalent
  carry-over, not a phase-2 breakthrough.
- The pl-mean drop (0.69 → 0.62) and plies-mean rise (38 → 42) suggest
  the policy genuinely moved, but that motion didn't translate to a new
  eval-side ceiling. Read this through [[loss-floor-bouncing]]: the
  *learning gap* improved (lower training loss against the in-distribution
  buffer), but the *target-distribution* against the fixed external
  baselines didn't reflect it. Phase 3 (or post-WL5) should mine fresh
  hard buckets from phase-2's evaluation positions rather than carrying
  forward WL4's archive.

**Phase 2 limits exposed:**
- Buffer is undersized vs the generation rate (cycled ~28× by 1M games)
  — see [[project-buffer-undersized]]. Default to 3M positions for the
  next cell.
- Eval-cycle sample size (20 games per baseline) is too small for a
  signal-dominated read on the 1500-1700 elo band; ±10% bands routinely
  swing elo by 100+ points cycle-to-cycle. Future cells should consider
  larger n on the slow-eval pass.

**Run artifacts:**
- wandb: `o6cbjfnr` (continuous across both segments, step counter
  monotonic e4001 → e10200).
- Trainer log: `sweep_logs/WL5-diagnostics-archive-start/trainer.log`
  (phase 2 = epoch lines e5052 → e10200).
- Last checkpoint on disk: `epoch10200.pt` (5.3 MB slim);
  `latest.pt` (8.8 GB full buffer snapshot at e10200).
- Commits anchoring phase 2: the phase-1-list (especially `aff6969`,
  `4f21cdd` — Conv+BN fusion) plus all of 2026-05-22's frontier merges
  that ran adjacent in other worktrees but did not modify the WL5
  trainer/worker code path.

**Cross-refs:**
- WL5 phase-1 close above for the un-fused-era retrospective and the
  phase-boundary framing.
- [wiki/topics/external-engine-baselines.md](wiki/topics/external-engine-baselines.md)
  — the next eval-side step is fixed external anchors (Rapfi etc.) so
  the noisy 20-game lookahead winrates stop being the strength signal.
- [wiki/topics/m5-max-as-mainframe.md](wiki/topics/m5-max-as-mainframe.md)
  — post-WL5 work pivots to chip-characterization perf sweeps before
  the next training cell.

## 2026-05-23 — LF1 launched (lean-fp16-canary: the perf lab's +152% recipe as a real run)

After the 2026-05-23 perf-lab era (ANE asked-and-answered; cross-engine
coupling pinned), we cashed in the headline perf finding as a real training
run. **LF1 = WL5 recipe + the R-TRAIN-LEAN-fp16 deltas**: `wave_size 512`,
`sgd_per_position 0.001`, workers `--fp16-eval` (wired via the Cell
`extra_worker_args`). Fresh, **1000 epochs**, started **HOT** (chip
heat-soaked from the perf session — Jason: training runs hot, don't wait;
noted for cold/hot comparison). run_sweep cell `LF1`, wandb **`h9al2e0k`**
(100-ep shakeout `geft5xmy` first). Trainer + 8 fp16 workers + eval worker.

**Spin-up verified**: fp16 cast in worker logs, V=512 wave mode, sgd=0.001
(epoch-1 steps=25 matches), validation archive (1400 pos) loaded, 0 NaN.

**Key finding (caveats the "+152%")**: the perf lab's 0.0667 epochs/s
(≈15s/epoch) was a **cold-buffer transient**. In the real run, V=512 fills
the 1.5M buffer by ~e27, then `sgd_per_position × fast V=512 inflow` →
**~1300+ (growing) SGD steps/epoch → ~3 min/epoch**. So 1000 epochs is a
multi-day run, not ~4h. The +152% is GENERATION (aug/s) throughput, NOT
training speed. **But it learns fast per epoch** — elo 437→776 across the
buffer-full transition (~e28), pl 4.42→3.78, plies ~27, 0 NaN. The honest
"faster recipe" verdict is wall-clock-to-elo + val/policy_ce, pending the
full LF1 dynamics. See the gomoku-train skill "Tuning knobs → LEAN-fp16"
and experiment-ledger "LF1".

**Monitoring**: cron `d443ef9c` (every 30 min, session-scoped) — pushes only
on NaN/crash/plies-collapse/completion; cleans up orphaned workers at e1000
or on early exit. TQ verdict on adopting the recipe into the production
lineage stays Jason's call.

## 2026-05-27 — Cross-game value sidecar: the live-flooding perf trap

New lever (bead `derby-eft`, cell `derby-x-crossgame`): **cross-game value
aggregation** — blend single-game `z` with a cross-game discounted-MC aggregate to
de-noise value targets (the opening-convergence / "mushy value" problem). It took
**three principled fixes that each passed CPU tests and still failed the derby
runner's LIVE re-race**, because per-epoch ingest cost scales with self-play
**inflow** and the CPU tests didn't replicate flooding:

- `derby-eft` (impl) → recency-decay was O(store)/cycle.
- `derby-eda` → lazy O(1) decay `scale`, but `save()→_renormalize()` re-introduced an
  O(store) per-epoch fold; store grew unbounded (116k→234k, 14MB); epoch wall
  14s→128s.
- `derby-4bq` → capped store to opening plies (`ply<10`) + decoupled renormalize from
  save; bounded the store, but the per-position 8-symmetry pure-Python canonical
  keygen still runs on EVERY ingested position *before* the ply filter → flat ~10×
  tax under flooding. Runner reverted again (`c575b2f`), reopened `derby-eda`,
  hardened its skill to verify re-races at full load (epoch 50+, `7ec7637`).
- Round 4 (in flight): ply-gate the keygen + vectorize `canonical_key` (property-test
  byte-identical) + flood-scale regression test. CPU tests are necessary but NOT
  sufficient — the runner's full-load live re-race is the real gate; if round 4
  misses, the lever is shelved pending design attention.

Synthesis + the durable lesson: `wiki/topics/cross-game-value-sidecar.md`. Lesson:
training-loop ingest perf must be validated under live flooding, not a CPU sim —
sibling of the perf-bench lesson (`wiki/topics/perf-bench-vs-real-training-cost.md`).
(Captured by the bead-runner session, 2026-05-27.)

## 2026-06-12 — 9×9 era closed (Rapfi cert) + 15×15 port landed

One-shot autonomous pass (Jason: "I'm losing internet soon"). Full synthesis +
plan: `wiki/topics/15x15-era-feasibility-and-plan.md` (§6 results log). Compact
record here:

- **The 9×9 strength frontier is closed.** v8 champion
  (`sweep_runs/derby_v8/_peaks/mate-discount/peak.pt`, 64×4,
  vcf+global-pool+value-discount-0.98+gumbel) at sims=400 + the verified eval
  config (`--fpu-reduction-c 0.45 --reuse-tree --proven-prop`) vs **Rapfi
  (Gomocup freestyle 2625 Elo)** on 9×9 freestyle, 40 games/tier, seed 0:
  rapfi100 = 16W-0L-24D (70%), rapfi500 = 15W-1L-24D (68%), rapfi1000 pending.
  **1 loss in 80 games against a 2625-rated engine.** 9×9 freestyle is drawish
  and the rating is a 15×15 rating, so this is a yardstick not an Elo label —
  but it confirms there is no 9×9 headroom left to chase. Cert JSONL:
  `sweep_logs/rapfi_cert_v8champ_20260612.jsonl`.
- **Why this matters:** combined with derby v9 (bigger nets lose at 9×9) and
  the FPU sweep (champion sweeps the lookahead ladder), the case to graduate
  to 15×15 is now evidence-backed, not aspirational.
- **The perf ceiling was the small model, not the Mac.** New bench
  `scripts/bench_board_scaling.py`: at production wave=64, the champion arch
  runs 15×15 for free (0.98×); a 96×8/1.55M-param 15×15 net costs only 2.32×;
  128×10 is 4.62×. Dispatch-bound regime confirmed. ~week-scale wall-clock for
  a 1M-game 15×15 run (envelope, to be validated by a live smoke).
- **Codebase ported to parameterized board size** (merge `27718c7`):
  `gomoku/board_config.py`, native C parameterized at compile time
  (`_*_native15`), checkpoints embed board_size, `96x8` preset. Full pytest
  green at 9×9 (608 passed); 22-test 15×15 module green; 9×9 fixed-seed match
  byte-identical before/after. Free-style rules kept (swap2 → #22, renju →
  #23). Single-process 15×15 throughput sanity: 2.11 games/s (tiny net).
- **Next:** run `SMOKE15` (`GOMOKU_BOARD_SIZE=15 python scripts/run_sweep.py
  --cell SMOKE15 --foreground --max-wall-secs 90`) for the live aug/s go/no-go,
  then Phase 4 (first real 15×15 run, WDL head as first new contestant,
  bit-packed buffer prerequisite). Epic #21.

## 2026-06-12 (cont.) — Rapfi cert complete + 15×15 smoke GO

Continuation of the one-shot pass (cert finished, GPU freed, smoke run clean).

- **Rapfi cert, all 3 tiers** (`sweep_logs/rapfi_cert_v8champ_20260612.jsonl`):
  rapfi100 16W-0L-24D (70%), rapfi500 15W-1L-24D (68%), rapfi1000 12W-2L-26D
  (62%). **Total 43W-3L-74D over 120 games** vs Rapfi (Gomocup freestyle 2625)
  at 9×9. 3 losses in 120; the rest wins (43) or draws (74). Expected trend:
  more Rapfi time → wins decay to draws, never to losses. 9×9 era closed.
- **15×15 plumbing smoke (SMOKE15, GO).** `board_size = 15` confirmed; full
  loop (trainer + 2 workers + eval) ran end-to-end via run_sweep with the
  native `_*_native15` extensions, 122 epochs in the 90 s cap, clean resumable
  teardown. Plies ~40–78 = real 15×15 games. Throughput at tiny net / sims=30
  / 2 workers / wave=16: **~7,574 aug/s, ~16.3 games/s** wall-clock — the
  small-net 15×15 regime is as cheap as 9×9, confirming the dispatch-bound
  bench. NOT the 96×8 production cost (that's a Phase-4 measurement).
- **State:** epic #21 Phases 0–3 done + merged to main. Phase 4 unblocked —
  first real 15×15 run (carry v8 recipe; WDL head = first new contestant;
  bit-pack the buffer first per buffer-bit-packing.md). Follow-ups #22 (swap2),
  #23 (renju), #24 (web UI 15×15).

## 2026-06-15 — The 96×8 "champion" is REGRESSED below its own untrained seed

**Correction to the 15×15 campaign record.** The overnight session (dbe0609b)
re-ranked all 15×15 nets via direct head-to-head (match.py validated:
champion-vs-self = 50%) and found that the Rapfi-based rankings were wrong
end-to-end (§8–§9 of `wiki/topics/alphazero-lessons-15x15-gomoku.md`).
Today's follow-up uncovered a deeper problem: the 96×8 trained "champion"
(`g15_champion_96x8_e499.pt`) is **catastrophically regressed below its own
untrained net2net seed** (`g15_96x8_seed.pt`):

- `g15_96x8_seed.pt` vs `g15_champion_96x8_e499.pt` = **40-0** (seed wins).
- `g15_96x8_seed.pt` vs `g15_champion_e909.pt` (64×4) = **50-50** (tied).
- `g15_champion_96x8_e499.pt` vs `g15_champion_e909.pt` (64×4) = **0-40**.
- `g15_champion_96x8_e499.pt` vs `g15_128x10_bigbuf_eval502.pt` = **0-40**.

400 epochs of self-play training on the v8 recipe made the 96×8 40-0 **worse**
than its starting point. This was entirely invisible to internal metrics
throughout the run: plies 30–48, vl 0.17–0.25, internal-ladder win-rates
85–100% — all looked healthy. The Rapfi yardstick (broken per §8) also did not
flag it.

**The net2net grow itself was valid** (output deviation <1e-4); the regression
is in the TRAINING, not the grow step. Why the run regressed is open (recipe?
self-play distribution drift?). The `G15-96x8-redo` experiment (cell added
to `scripts/run_sweep.py`) re-trains from the good seed to test reproducibility.

**Current champion (2026-06-15):** `g15_128x10_bigbuf_eval502.pt` (128×10) is
the strongest preserved net, beating the 96×8 e499 40-0. Absolute strength
awaits a fixed yardstick (#28); clean capacity curve awaits same-epoch matches
(#29). See §10 of `wiki/topics/alphazero-lessons-15x15-gomoku.md` for the full
lesson.

**Operational note:** `g15_96x8_seed.pt` is a GOOD checkpoint (ties 64×4-e909).
Do NOT use `g15_champion_96x8_e499.pt` as a resume point or production net —
it is weaker than its own seed. The seed is the right warm-start for the redo.

## 2026-06-15 — The brain wrapper & the empty-history trap (history-conditioned net through an order-free protocol)

We exposed our own net as a first-class Gomocup engine — `gomoku/gomocup_brain.py`
(#31, commit `1834df0`; shell wrapper `scripts/run-gomoku-az`, registerable via
`external:cmd=run-gomoku-az --checkpoint X --sims N`) — the brain-side mirror of the
client-side `gomoku/external_engine.py`. Building it surfaced a **silent strength
loss baked into the protocol bridge itself**, the §10-style "looks healthy, plays
worse" failure in a new place.

**Mechanism.** Our input is history-conditioned: `gomoku/game.py` uses
`HISTORY_PLY = 8` recency planes per side. `to_planes()` reads the CURRENT board
from `board[0]`/`board[1]` and the OLDER recency frames from `state.history`. The
classic Gomocup `BOARD` command re-dumps the whole position every move and is
order-free (move order is unrecoverable from a single dump). A naive brain rebuilds
a fresh `GameState` each move with EMPTY `history`, so `to_planes()` emits a full
current board but ALL-ZERO recency planes — a self-contradictory, OOD input the net
never trained on.

**Measured** (same checkpoint `g15_128x10_bigbuf_e588_best.pt`, vs the heuristic,
sims=100, seed=0; 4 games unless noted):

- Native in-process picker (full history): **4-0 = 100%**.
- Wrapped, BOARD-replay every move (empty history): **1-3 = 25%**.
- Wrapped, `incremental=1` TURN-mode (real history accumulates): **5-1 (n=6) = 83%**.

The empty-history path cost ~75 points against the same opponent — purely an I/O
artifact, not a net or search change. Small-n, but the gap dwarfs the noise band and
reproduced as a path difference.

**Fix.** `external_engine.py` gained an `incremental=1` mode (default off). After a
first `BOARD` sync it feeds the opponent's single new move as `TURN x,y` (no
`RESTART`, which would wipe history), so a stateful brain accumulates real move
history via `GameState.apply()`. `_can_turn()` gates it to clean continuations (our
stones unchanged, opponent +1 stone) and falls back to a `BOARD` resync at
boundaries/desyncs/the opening. Default-off keeps classical external engines (no
history planes — Rapfi et al.) on the robust BOARD path. **Nets must be registered
with `incremental=1`.**

**Lesson.** When exposing a history-conditioned net through a stateless/order-free
protocol you must reconstruct move RECENCY, not just the static position; drive the
engine incrementally (`TURN`) so history accumulates, or it silently sandbags
itself. Generalizes the silent-self-play-regression theme: internal-looking-healthy
≠ actually-strong; gate on a real same-checkpoint head-to-head. Full lesson: §13 of
`wiki/topics/alphazero-lessons-15x15-gomoku.md`. Tooling also built this session:
`scripts/panel_tournament.py` (#32, commit `0fb7fc1`), the calibrated panel
cross-table runner for the #30 engine-panel-anchored derby.

## 2026-06-15 — First panel tournament: the calibration broke (the failure IS the finding)

Ran the first 9-player round-robin (3 of our nets + 6 opponents, n=6 each) via
`scripts/panel_tournament.py` toward the #30 "calibrated yardstick." Raw records:
`sweep_runs/panel_tournament_results.jsonl`. Reader: `scripts/panel_white_elo.py`
(its §1 per-color rates are ground truth; §2 BT-Elo is a flagged estimate). The
honest headline is a **partial failure** — we do NOT have a calibrated strength
number, and that is the finding. Tracked as #35.

**What broke #1 — the engines, not our nets.** Of 36 pairs, **only 19 played; 17
ERRORED.** 13 = `engine timed out after 30s` (mostly `embryo26 vs *` — Embryo is
GPU/Vulkan-contended); 4 = `engine process has exited` / `closed stdout (EOF)`
(`* vs zetor17` — Zetor crashes on back-to-back reuse). **Our brain-wrapped nets
produced ZERO errors** — every failure was an opponent engine dying. Crashes drop
*whole pairs* → **missing data, not fabricated losses.**

**What broke #2 — the anchor (the real lesson).** The affine fit internal-strength →
published Gomocup Elo came out with a **NEGATIVE slope (~−0.071)**: internal strength
*anti-correlated* with published rating. Smoking gun: **yixin18 (published ~2310,
top-tier) went 0–30** — lost every completed game, including **0–6 to the
heuristic**; **pela23 (published ~1499) went 24–6**. Under wine + single-thread +
10s/move the engines do NOT play at their multi-thread tournament ratings, so the
published Elos are **invalid anchors**. `panel_white_elo.py` §2 **correctly REFUSES**
to print a calibrated Elo (detects the degenerate slope → mean-0 relative fallback,
loudly flagged). Right fix: **measure effective strength under our exact harness**,
don't assume the published ladder.

**What IS trustworthy (completed games; small-n hints per the noise band):**
- *Net-vs-net:* champ beats az-96x8 **5–1 (83%)**; az-96x8 beats e588 **4–2 (67%)**;
  e588 beats champ **4–2 (67%)** — a close rock-paper-scissors loop = noise, not a
  stable ordering.
- *Net-vs-heuristic* (non-trivial floor): e588 **6–0 (100%)**, champ **5–1 (83%)**,
  az-96x8 **5–1 (83%)**.
- *White-side defense gap (#33, the concrete next target):* champion aggregates **94%
  black (attack) vs 50% white (defend), +44pp**, a **50% white-loss over 18 white
  games**. Opponent-specific: champion goes **0–3 white (100% loss)** vs **embryo26**
  AND **zetor17** (and vs net e588), but holds **3–0 white** vs the *weaker* yixin18
  and eulring16. **Caveat (not hidden):** the reader flags that **az-96x8 does NOT
  show the gap** (67% white vs 67% black, n=12) — the gap is large for the champion
  but not yet a universal law; e588 matches the champion's shape (100% black / 67%
  white, +33pp, n=9).

**Conclusion.** The #30 calibrated yardstick is **NOT yet achieved** — it needs
reliable engines + empirically measured effective strengths under our exact harness,
not assumed published Elos (#35). But the tooling worked flawlessly on our side:
brain wrapper (#31), runner (#32), reader (#33 / `panel_white_elo.py`) — zero errors,
and the reader refused to over-claim. The white-side defense gap (#33) is confirmed
and quantified: the concrete next target. Reliability fixes queued in #35: per-engine
timeout, process-per-pair (no engine reuse), measure-don't-assume anchors. Full
synthesis: §14 of `wiki/topics/alphazero-lessons-15x15-gomoku.md`; design-doc status
updated in `wiki/topics/engine-panel-derby-design.md`.

## 2026-06-15 — White-side defense is a TRAINING gap (search-invariant, not an eval flag)

Followed the #33 white-side defense gap (§14 panel: champion `g15_128x10_bigbuf_eval502.pt`
~94% as black / ~50% as white, **0–3 white = 100% loss** vs the *strong* attackers
embryo26 + zetor17) into Step A of the defense plan — the cheap, eval-only branch —
to test whether any eval lever fixes it before committing a training slice. **Verdict:
it is a genuine TRAINING gap.** Both cheap fixes are FALSIFIED against the one
reliable real attacker we can head-to-head cleanly, **zetor17** (white = champion
defending):

**FPU-reduction (the 9×9 wiki's claimed white-loss fix — c=0.45) changes NOTHING vs a
real attacker:**
- vs `lookahead:depth=4` (weak searcher): FPU=0.0 → white **88%**; FPU=0.45 → white
  **100%** (closed a small residual tail that was nearly closed already).
- vs **zetor17** (real strong attacker): FPU=0.0 → white **0–6 (100% loss)**;
  FPU=0.45 → white **0–6 (100% loss)**. *Identical.* The 9×9 FPU-as-defense-fix does
  **NOT transfer** to 15×15 real-engine defense — CORRECT the old claim (it looked
  plausible because it worked on the weak searcher).

**Search budget (H3, "search too shallow") — 4× more search changes nothing:**
- vs zetor17: sims=200 → white **0–4 (0%)**; sims=800 → white **0–4 (0%)**.

**The dissociation that names the cause:** at *every* FPU and sims setting the
champion is **perfect as black** vs zetor17 (4–0 / 6–0) and **helpless as white**
(0–4 / 0–6) — same net, same search, same opponent, only the color flips. A weakness
immune to both eval-side levers (search prior AND search depth) lives in the
**weights**, not the search: the policy/value cannot REPRESENT the saving defense; it
must be TAUGHT (relabel the saving move).

**Hypothesis ledger:** H3 (search too shallow) **RULED OUT** for real-engine defense.
H1 (teaching gap, #18 — a lost white game is labeled only `z=−1` for the whole
trajectory, never *which* move would have saved it) and H2 (value-target asymmetry —
white wins → 0 at convergence, so the value head gets little gradient to split
"drawable" from "lost" white positions) **STAND**.

**Fix routes to training (#36 / #18 recipe):** clone the champion cell, turn on the
value-only `--defense-teacher` (stamps proven-lost white positions `z=−1`, "defend
earlier") paired with `--vct-teacher`. If value-only under-moves the **draw/loss
boundary** (where "never lose as white" lives), escalate to **I2 — stamp the saving
move** (one-hot policy on the unique defensive refutation).

This is the project's "internal-looking-healthy ≠ actually-strong / be suspicious"
ethos AGAIN, applied to the *fix*: the cheap eval fix looked plausible (worked on the
weak searcher, as the 9×9 wiki promised) and was falsified on the real attacker. Had
we trusted the depth-4 read we'd have shipped an FPU flag and declared defense solved
while still losing 0–6 to every strong engine. Full synthesis: §15 of
`wiki/topics/alphazero-lessons-15x15-gomoku.md`; plan updated (Step A FALSIFIED → Step
B is the first real experiment, I0 demoted / I1 promoted) in
`wiki/topics/white-side-defense-plan.md`. Tracked #33; fix #36 / #18. No code edited,
no games run this session (numbers verified against the GPU runs on #33).

---

## 2026-06-16 — Sliding derby launched; the frozen-reference gate, validated against known truth

**What shipped tonight (autonomous, Jason asleep — "plow forward, don't gate on me"):**
the sliding derby's GPU producer (`scripts/sliding_derby_runner.sh`) is live on the
**#36 G15-defense** cell, warm-started from `g15_128x10_bigbuf_eval502.pt`, trained in
resumable 1-hour slices with the frozen-reference promotion gate
(`scripts/sliding_gate.py`, #39) run between slices. **Reliable evals only** — wine
engines NIXED per Jason (another wine crash; #35): the verdict is anchor-free
net-vs-frozen-peak H2H (pure torch), secondary white-loss is net-vs-peak. The gate is
calibration-immune (never reads absolute Elo).

**The gate was validated against truth I was certain of before trusting it** (5 H2H
dry-runs, sims=200, openings=4p, MPS):
- NULL `eval502 vs eval502` → 12W-12L = **50%**, CI[.314,.686] → **REVERT** ✅ (refuses a tie — the load-bearing anti-noise property).
- `eval502 vs eval146` → 46%, Δelo −29±136 → REVERT;  `eval146 vs eval502` → 54%, +29±136 → REVERT (mirror-consistent).
- `eval502 vs 128x10_seed` → 57%, +51±98 → REVERT;  `eval502 vs 96x8_seed` → 52%, +17±106 → REVERT.
- **POSITIVE CONTROL** `eval502 vs RANDOM-init net` (same arch, weights re-init) → **40W-0L = 100%**, CI[.912,1.0] → **PROMOTE** ✅.

**Two findings fall out of this:**
1. **The gate works in both directions** — it REVERTs ties/noise (5×, incl. an exact 50%
   self-match) and PROMOTEs a genuine gap (champion crushes random 40-0 with a clean
   CI). The PROMOTE path is not just unit-tested; it fired on a real eval.
2. **The 128x10 bigbuf lineage is a tight ~50-elo plateau.** eval502, eval146,
   128x10_seed (the 96x8-champion grown), and 96x8_seed all land within 50±7% H2H of each
   other at n≤48. Because the champion-vs-random control proves the H2H instrument has
   FULL dynamic range (0→100%), this clustering is **real, not a measurement artifact** —
   502 epochs of warm-started bigbuf training added little absolute strength over the
   grown seed. (Faint, n=24-noisy hint, NOT concluded: eval146 scored ≥50% vs eval502 in
   both orientations — i.e. the *later* checkpoint is not stronger, mildly consistent with
   the silent-regression thesis. ±136 elo — a hint, not evidence.)

**Implication for the derby design (load-bearing):** a frozen-reference ratchet with
underpowered `n` never ratchets. At n=24 the Δelo CI is ±136; resolving a ~70-elo
per-lap gap needs **n≈120** (set as the runner's `GATE_N`). And for the *defense* arc
specifically, lineage-sibling overall-H2H is compressed, so the targeted **white-loss
signal is more likely to move than the promote/revert verdict** — the defense fix is a
narrow capability the overall game may not surface. Watch the gate's secondary
`white_loss_rate`, not just PROMOTE/REVERT.

**RESOLVED (same night) — gen stalled on the UNCAPPED defense-teacher VCF solve:** the trainer warm-starts
cleanly (epoch 501+, buf=1.5M, vl≈0.156) but self-play produced ZERO games (`new=0`,
0 record files, workers ~100% CPU) for ~6 min under TWO configs: first
`--vct-teacher + --defense-teacher`, then (after I dropped VCT) `--defense-teacher`
ONLY. **So my first instinct — "VCT starved gen" — is NOT proven**; the defense-only
config stalls identically. A 15×15 game at 100 sims should take seconds, so 6 min to
zero is anomalous for either config. Candidate causes, unranked: (H1) slow first batch
because the strong eval502 warm-start plays long, well-defended games (Jason's own
"defense learned = longer games = slower cycles" lesson); (H2) the `--defense-teacher`
VCF solve (200k-node default) firing on the strong net's many tactical positions; (H3)
a native-ext / MCTS config issue. **Decisive next step: a NO-TEACHER control run**
(champion config, warm-start eval502) — if it gens fast, the teacher (H2) is the cost;
if it also stalls, H1/H3. A `/tmp/gen_watch.sh` poller watches the live defense-only run
for ≤14 min; if gen appears it was just slow (H1) and the experiment proceeds. The VCT
drop still stands on experiment-cleanliness grounds (champion had no offensive teacher
→ defense-only is the clean single variable), independent of the gen cause. Ethos
applied to my OWN diagnosis: be suspicious, don't harden a plausible cause into fact
before the control runs.

**RESOLUTION (controls ran the same night → H2 confirmed):** a no-teacher control genned
eval502 at **~2.6 s/game**; the SAME config + `--defense-teacher` at the 200k-node VCF
default stalled (0 games / 6 min); + `--vcf-max-nodes 2000 --vcf-max-depth 10` genned at
**~3.1 s/game** (rescued). So the uncapped per-move defense VCF solve is the gen-killer,
not VCT and not slow-warmstart. Cell capped (commit `054a5be`); derby relaunched and gen
CONFIRMED flowing (epoch 504: `new=8`, `plies=42.5`). The cap still proves the SHORT
forced losses "defend earlier" needs. **General lesson: 9×9 teacher default solver
budgets do NOT transfer to 15×15 — always cap the per-move solve on the gen hot path.**
Evidence: `/tmp/noteacher_ctrl.log`, `/tmp/capdef_ctrl.log`.

Gate validation evidence: `/tmp/gate_validation_verdicts.jsonl`. Board:
`sweep_runs/sliding_derby_board.json` (frozen peak = eval502). Verdict trail:
`sweep_runs/sliding_derby_verdicts.jsonl`. Runner log:
`sweep_logs/G15-defense-board15/sliding_runner.log`. Tracked #38/#39/#36.

---

## 2026-06-16 — Defense-teacher CRASHED the champion (-458), diagnosed + course-corrected (#42)

The first real #36 defense experiment **failed catastrophically and was self-healed by the
overnight loop**. G15-defense (eval502 warm-start + value-only `--defense-teacher`, hard
`z=-1`, capped VCF nodes2000/depth10) degraded MONOTONICALLY over 3 gate laps:

| lap | win_rate vs champ | white_loss | Δelo |
|---|---|---|---|
| 1 | 0.483 | 0.783 | −12 |
| 2 | 0.242 | 0.917 | −199 |
| 3 | 0.067 | 0.983 | −458 |

value_loss collapsed 0.16→0.06, policy_loss ballooned 1.25→3.4, plies stayed HIGH (~40-50,
so NOT fast-attack collapse). **Mechanism (researcher-diagnosed, cited):** the defense-teacher
stamps a hard `-1.0` value on EVERY proven-lost white-to-move position with **no bound on the
fraction relabeled** (`self_play._apply_defense_teacher`) — the deepened solve over-fires → the
value head saturates to "white always loses" (vl→0.06) → that **contradicts the untouched
attacking policy target** → the shared residual trunk corrupts → policy degrades. This
**reproduced the exact `#18` "defense-blindness" trap** (value_loss→0.04) the teacher was
meant to avoid: teaching "you already lost" (a value the net can't act on) instead of "here's
the saving move" poisons the model. The gate **correctly REVERTed all 3 laps** — the frozen
peak (eval502) was never promoted, so **nothing was damaged**; the cost was only wasted GPU,
caught at lap 3.

**Course-correction (#42, merged 6063c2b, relaunched):** opt-in `--defense-soft-value` (stamp
−0.5 not −1) + `--defense-max-fraction` (cap the fraction of a game's positions relabeled,
keeping the LATEST firing plies nearest the loss) + shallow solve (vcf 800/depth 7). Defaults
byte-identical. The corrected run is training from eval502 (vl healthy 0.162); the gate grades
whether the gentler teacher drives white_loss down WITHOUT the Δelo crash. If it still
under-performs, the real fix is **#43 — stamp the SAVING MOVE on the policy head** (the #18 I2
arm: teach the refutation, don't just crush value). Filed: #41 (escalation, resolved), #42
(merged), #43/#44 (queued). General lesson banked: **a value-only "you're lost" teacher with no
fire-rate bound saturates the value head and tears the policy apart — bound the fire-rate AND/OR
teach the move, not just the value.** Broken-config evidence: `sweep_runs/_archive/sliding_derby_verdicts_BROKEN*`.

---

## 2026-06-16 — #37 death-spiral CONTROL (champion-continuation, no teacher): eval502 stable

First cycle of the measured-outcome composite derby (`.claude/workflows/sliding-derby-composite.js`,
the known-answer self-test) doubled as a clean **#37 control**: eval502 continued with **NO teacher**
for 13 epochs (501→513, `G15-128x10-bigbuf`) stayed **STABLE** — vl 0.162→0.170, plies ~33-46 (mean 37),
gate vs eval502 = AMBIGUOUS (win_rate 0.45, CI [0.307,0.602] straddles, Δelo −35±106), no gen-starvation,
no NaN. So the champion does **NOT spontaneously degrade without a teacher** — directional support that
the `--defense-teacher` CAUSED the #42 collapse (the value-head saturation), not training-instability-in-
general. Caveat: SHORT window (13 epochs) — directional, not conclusive; a longer no-teacher control would
firm it up. Board: `sweep_runs/composite_derby_board.jsonl`. This was ALSO the measured-outcome derby's
first end-to-end self-test (it correctly scored this known-stable case `confirm`).

## 2026-06-16 — #45 white-defense instrument: positive control PASSES; champion is at the FLOOR vs a weak attacker (defense is attacker-strength-gated)

Built + merged the white-defense eval suite (#45, commit `eda147b`): a fixed 80-position
white-to-move-threat fixture (`fixtures/white_defense_15x15_v1.json`) + `white_loss_rate`
with Wilson CI + a `white_defense_tally` gate primitive (drop-in for
`sliding_gate.run_gate`'s `white_loss_fn`, no `decide_verdict` change). The
orchestrator-run **positive-control GATE** (the science verify-gate before merge, run on
the GPU while a workflow built the code on CPU — the cockpit split): champion `eval502`
`white_loss=0.0375` (3/80, CI [0.013, 0.105]) vs a random-init net `0.95` (76/80, CI
[0.878, 0.980]), attacker `lookahead:depth=2`, sims=200, n=80/net — **CIs strictly
non-overlapping → the metric DISCRIMINATES** defensive skill.

**Science:** the champion defends 77/80 vs depth-2 — **SOLID at the weak-attacker end**,
consistent with the 2026-06-15 Step A depth-4 result (88–100% white success) and with
zetor17 0–6 (100% loss). The white-defense weakness is **attacker-strength-gated** (solid
vs weak searchers, brittle vs strong engines), NOT a flat "brittle defender." Consequence:
#45 v1 (weak attacker, weak-baseline-mined threats) sits at the champion's **floor** → no
headroom to measure an I1/I2 (#43) gain. The *diagnostic* instrument needs a strong
attacker → **#49** (depth-3/4 or champion-as-attacker over strong-attacker-derived
threats). #45 v1 is the validated *primitive* (CI + gate seam + play-from-position); #49
makes it diagnostic. Gate artifacts: `/tmp/wd_champ.json`, `/tmp/wd_weak.json`. The 3
champion losses concentrate in the `_four` (strong-threat) provenance — where residual
difficulty lives. See [white-side-defense-plan.md](wiki/topics/white-side-defense-plan.md)
§1B.2 results block.

## 2026-06-18 — FIRST contact vs REAL Rapfi-NNUE: champion 21% @ 5s, and white-defense gap is the WHOLE story (0/12 white)

First-ever measurement of our champion against the **real native Rapfi-NNUE** engine —
the #28 "weightless / under-search" yardstick bug is fixed (NNUE config loads, searches to
budget; see wiki/topics/external-engine-baselines.md), and Rapfi is now a registered
`_NATIVE_ENGINES` panel participant (#40, no wine). Run via the #30 panel harness:
`scripts/panel_tournament.py --only az-champ-128x10,rapfi --n-games 24 --sims 400
--timeout-ms 5000 --seed 7` (GOMOKU_BOARD_SIZE=15, GOMOKU_DEVICE=mps, 4-stone random
openings, color-alternated). Out: `sweep_logs/panel_champ_vs_rapfi_5s_n24_20260618.jsonl`.
Champion = `sweep_runs/g15_128x10_bigbuf_eval502.pt` (the frozen sliding-derby reference).

**Result: 5W-19L-0D = 20.8%** (relative Elo gap ≈ -232). Color split is the finding:
- **black (attacking): 5-7-0 = 42%** — competitive with the #1 Gomocup engine.
- **white (defending): 0-12-0 = 0%** — swept.

ALL of the strength shortfall is the **white-side defense gap** (#33/#37/#18). This is the
strong-attacker diagnostic that #45 v1 lacked (#45 v1 sat at the champion's floor vs a weak
attacker, 77/80; real Rapfi as a strong attacker exposes the gap completely, 0/12). It is
the most direct evidence yet for the **#37 hypothesis** (white-side defense weakness drives
degeneration) and a far sharper probe than self-play. **Caveat (do not over-read the 21%):**
NOT compute-matched — Rapfi got 5s/move single-thread, our net ~0.3-0.5s @ sims=400; and
this is freestyle (first-player-favored) with only 4-stone random openings, n=24 (white
0/12 is conclusive for the gap; the overall rate has a wide CI). Absolute Gomocup-Elo
calibration still pending (effective single-thread Rapfi strength under our harness must be
measured, not assumed = published ~2625; #35/#30). Next probes: compute-matched rematch
(net sims → ~1-2s/move), a TC sweep, and using Rapfi-as-attacker to drive the #43/#44/#49
white-defense interventions.

## 2026-06-18 — TC-tier CALIBRATION vs real Rapfi-NNUE: a CLIFF then a white-defense PLATEAU (the deficit is 100% white-side)

Parallel calibration via the new `--jobs` path (#52, `eval_vs_rapfi.py --jobs 8`, 8 spawn-workers
saturating the 18-core M5; 200 games in 7.6 min). Champion `g15_128x10_bigbuf_eval502.pt`,
sims=400, vs native Rapfi-NNUE (run-rapfi wrapper → NNUE config), 5 thinking-time tiers, n=40/tier,
4-stone balanced openings, seed 7. Out: `sweep_logs/calib_champ_vs_rapfi_tctiers_20260618.jsonl`.

| Rapfi TC | win% | W-L-D | white W-L-D | black W-L-D |
|---|---|---|---|---|
| 10ms | 100.0% | 40-0-0 | 20-0-0 | 20-0-0 |
| 100ms | 27.5% | 11-29-0 | 0-20-0 | 11-9-0 |
| 250ms | 37.5% | 15-25-0 | 2-18-0 | 13-7-0 |
| 500ms | 27.5% | 11-29-0 | 2-18-0 | 9-11-0 |
| 1000ms | 27.5% | 11-29-0 | 3-17-0 | 8-12-0 |

**Two findings.** (1) **CLIFF, not ramp:** at 10ms Rapfi has no search time — it plays its raw NNUE
move and our MCTS champion sweeps 40-0; the instant Rapfi gets real search (100ms) it drops to ~27%
and then PLATEAUS (100/500/1000ms all ~27%; the 250ms 37.5% is n=40 noise). 10× more Rapfi time
(100ms→1s) barely moves the result → Rapfi is already strong enough to exploit us at 100ms. So
10ms is below Rapfi's useful threshold (a degenerate tier); the real opponent starts ~100ms.
(2) **The plateau is a WHITE-defense plateau:** as black the champion is competitive at every real
tier (40-65%); as white it is pinned at the floor (0-15%) regardless of Rapfi time. The shapes
differ — black declines as Rapfi deepens (55→65→45→40%: Rapfi defends our attack better with more
time), white is already at zero with nowhere to fall. **The champion's entire deficit vs the #1
engine is white-side defense**, now confirmed across 5 independent measurements (n=24 first-contact
+ these 4 real tiers, 160 games). Strongest mandate yet for #43 (stamp the saving move on the
policy). Caveat: not compute-matched (Rapfi 100ms-1s vs our net ~0.4s @ sims=400); freestyle,
4-stone openings; n=40/tier (white-side signal is conclusive, per-tier overall rate has CI ~±15%).
Next: a compute-matched rematch + finer tiers (25/50ms) to locate Rapfi's activation threshold.

## 2026-06-19 — Autolab went LIVE (the self-driving lab; #64, epic #53)

First night of the autolab driving itself. The lab is now a **launchd-supervised
self-driving loop**, not a hand-kicked library (the vision-gap audit had put us at
~31%: every part built + unit-tested but nothing kept it alive). `python -m
gomoku.lab.up up` seeds one lane, renders + loads four LaunchAgents, and lets
launchd be the supervisor.

**Live run config (this night):**
- Lane `9x9-champ-recipe`, cell **`derby-v9-small`** (the v8/v9-champion 96×6
  recipe, vcf-teacher, gumbel, 8 selfplay workers), base **scratch**, commit
  pinned `71feebd`. Per-slice cap **3600 s** (the vision's hard max; "no minimum"
  holds — the trainer self-caps on the next epoch boundary). Expected **~6 slices
  by morning**.
- **wandb OFFLINE** (`WANDB_MODE=offline`) for the unattended daemon — avoids the
  first-ever-slice interactive-auth blocking failure. So **no W&B run ID this run**;
  read progress from the **ledger** (`~/data/autolab/ledger.jsonl` result rows:
  `eval/model_elo`, epochs, wall_s) + run_sweep logs under
  `~/data/autolab/runs/9x9-champ-recipe/sweep_logs/derby-v9-small/`.
- HF: each slice pushes a per-slice revision to `jasonyandell/gomoku-9x9`
  (revision `9x9-champ-recipe-9x9-champ-recipe@N`, slimmed weights + provenance in
  `training_state.json`); the arena moves the `champion` tag on PROMOTE.

**What was proven before unattended launch (the never-recently-run path):** an
attended SMOKE slice with a **real HF push** in a throwaway `AUTOLAB_HOME` —
`run_sweep`→checkpoint→`--final-eval`(model_elo 388.6)→`hf.push_slice`→real Hub
revision→result row + flywheel followups→clean worktree teardown. Then deleted the
throwaway branch. So the trainer prod chain incl. HF delivery is validated; the
**1h prod slice and the first real arena gate are still unproven until morning**.

**Cockpit:** launchd `monitor` agent writes `~/data/autolab/monitor/latest.md` +
macOS notification every 600 s (always-on, survives session/reboot); a deterministic
`research` lane (every 1800 s) writes an honest "current thinking" note
(`~/data/autolab/research/latest.md`) that **ranks on PROXIES** (per-slice Δelo/hr
secant), NOT a real wall-clock-to-Δelo gate — that gate is the open core of #61.
On night-1 the researcher does **continuity over thrash**: it proposes ≤2 lanes
strictly *below* the seed priority and refuses to propose anything until the first
slice produces signal (cold-start refusal).

**Stop it:** `python -m gomoku.lab.up down` (cooperative stop-file + launchctl
bootout). **Watch what it's about to do, never barge a tenant:** the trainer
preflight pgreps `selfplay_worker|gomoku.train|run_sweep|eval_worker` and defers to
any *foreign* tenant (excludes the autolab's own home), so the derby/IDE training
still own the box if they want it. Operating contract:
`wiki/topics/autolab-supervisor-and-monitor.md`. Follow-ups: #61 (the real gate),
#63 (first live arena gate, verifiable in the morning), #65 (GOMOKU_BOARD_SIZE
passthrough — blocks 15×15 via autolab), #66 (P7 polish nits).

## 2026-06-19 — 15×15 era: first self-driving 15×15 run (#65, #67, first champion)

The autolab proved itself overnight (the 9×9-launch entry above), then **pivoted to
15×15 and ran the full loop again from scratch — seed → train → collapse →
self-recover → eval → HF push → gate → crown → re-pick — with zero hand-holding.**
This is the first 15×15 run driven end-to-end by the lab, not a hand-kicked slice.
Three things had to land first; then the science arrived.

**9×9 proof recap (don't re-log — see the entry above).** The 9×9 lane
`9x9-champ-recipe` ran 6 slices (elo @0 1531, @1 1519, @2 1567), the arena did one
definitional PROMOTE then two contested **AMBIGUOUS** n=12 gates (co-tenancy shrank
n 40→12), crowned `9x9-champ-recipe@0` as the first champion, and pushed a per-slice
HF revision each slice. **0 failures.** The 9×9 lane is now retired below the 15×15
lane.

**#67 — the arena artifact-ref contract bug (the only real defect of the night).**
The first *live* arena gate crashed with `FileNotFoundError`. Root cause: the
trainer's `_deliver`/`hf.push_slice` returns a **bare** `"repo_id@revision"`
artifact ref (no scheme), but `arena._resolve_model` only understood `hf://…`. A
producer/consumer **artifact-contract scheme mismatch** — the push side and the gate
side were each unit-tested *separately*, so it survived the whole suite and only
surfaced on the first real end-to-end gate. Fixed with a shared
`ArenaRole._parse_hf_ref` that accepts **both** `hf://owner/repo@rev` and bare
`owner/repo@rev` (and rejects local paths), used by `_resolve_model` +
`_default_set_champion`. Merged; arena daemon restarted; the failed eval was
re-opened via a **ledger correction** and then crowned the champion (the financial-
journal recovery path working as designed — nothing was lost). **Lesson banked: an
artifact contract between two daemons needs an end-to-end smoke of the trainer→arena
handoff; per-side unit tests can't catch a scheme mismatch across the seam.**

**#65 — the 15×15 pivot (merged).** Board size is a **process-start constant**
resolved by `gomoku/board_config.py` from `GOMOKU_BOARD_SIZE`, which **must be set
before any `import gomoku.*`**. Two-part enablement:
- (a) `trainer._run_slice` now threads `config["board_size"]` into the `run_sweep`
  subprocess env (closes #65) — so a 15×15 cell actually runs at 15×15.
- (b) `autolab up --board-size N` bakes `GOMOKU_BOARD_SIZE` into **both** the train
  **and arena** daemon plists. The arena daemon is **long-running and imports
  `gomoku` at startup**, so it can only gate a 15×15 candidate if the env is set
  *before the process starts* — a plist var, not a per-slice env.

The HF `champion` tag was **RESET for the 15×15 era** (a 9×9 net can't be loaded to
gate a 15×15 candidate — shape mismatch). The three 9×9 revisions are kept as
branches (evidence preserved). Seeded lane **`15x15-wdl`**, cell **`G15-wdl`** (v8
recipe + **WDL value head**, **FROM SCRATCH**, **no vcf-teacher / no defense
teacher**), priority above the retired 9×9 lane.

**The science — from-scratch G15-wdl SURVIVED the cold-start collapse (the key
finding; full dated correction in
[15x15-training-campaign.md](wiki/topics/15x15-training-campaign.md)).** With **no
warm-start and no teacher**, the run went *through* the documented cold-start
fast-attack collapse — plies **69.5** (epoch ~21) → **9.2** trough (epoch ~65) —
and then **self-recovered to a stable ~35–40 plies** (epochs ~260–670) on its own.
Health signatures during recovery: WDL value-loss held **~0.81–0.89 the whole way**
(never collapsed toward zero = the healthy-maturation signature, **not** the
terminal "confident-in-bad-fast-attack" death-tell); policy-loss fell monotonically
**5.4 → 1.6** (initial 5.42 ≈ ln(225), confirming a genuine 15×15 policy).
**Conclusion: with the v8 recipe + WDL head, the cold-start collapse is a
SURVIVABLE TRANSIENT, not a terminal trap** — the warm-start "remedy" may not be
strictly required for this recipe. **Critical caveat:** "survived the collapse /
recovered mid-game richness" is **NOT** "learned white-side defense." The decisive
open question is this net's **white W-L-D vs Rapfi** — does it reproduce the
warm-started champion's **0/12-white** hole (⇒ the deficit is representational/
recipe-deep) or defend better (⇒ warm-start was baking in the attacker bias)? That
probe is the next frontier read.

**First 15×15 champion.** `15x15-wdl@0` completed (internal eval **elo 1918** — the
**first 15×15 number, NOT comparable to the 9×9 elo scale**; from scratch). The
arena did a definitional **PROMOTE** → first 15×15 champion crowned (HF revision
`15x15-wdl-15x15-wdl@0` + the reset `champion` tag). The flywheel rolled to
continuation **`15x15-wdl@1`** (resumed from the lane's own `latest.pt`). The full
loop ran at 15×15 with zero hand-holding: seed → from-scratch train → collapse →
self-recover → 1918 → HF push → gate → crown → re-pick. Issues: #65 (board-size
passthrough, merged), #67 (arena artifact-ref fix, merged), epic #53. The
arena-yardstick gap (no absolute Rapfi readout wired into the arena yet) is captured
in [autolab-architecture.md](wiki/topics/autolab-architecture.md) § Arena-yardstick
gap.

**2026-06-19 — #43 I2 defense slice: lever SOUND, killed on buffer dilution (wandb
`zrjfwny2`).** Ran the policy-stamp defense teacher live against the warm-started
128×10 champion (cell `G15-defense-i2`, warm-start `g15_128x10_bigbuf_eval502.pt`,
1.5M bit-packed buffer, board 15). Resumed e585 → **killed at e1286**. **The lever is
HEALTHY** — `pl` plateaued ~1.19–1.22 (bounded; NOT the #36/#42 `pl→3.4` trunk
corruption), `vl` clean ~0.13–0.14 (NOT the value-only `vl→0.06` saturation), `plies`
stable ~37 (no fast-attack collapse): stamping the saving move on POLICY and leaving
value at the natural outcome works as #43 designed. **But un-readable** — fresh stamped
games are only ~0.16–0.3 %/hr of the 1.5M buffer even at 16 generators (~1,100 games/hr
sustained), so the warm-start attacker-biased mass drowns the defensive stamps and the
Rapfi white-column gate can't leave the 0/12 floor at this pace. **Binding constraint =
buffer FRESHNESS (stamp density), not gen rate.** Killed deliberately (clean SIGTERM;
checkpoint + wandb preserved). Levers built this session: **#69** (`run_sweep
--n-workers` generator-count knob — 16 gens ≈ the 18-core M5 ceiling, but 16-vs-12
sustained throughput is within noise; the 10-min A/B over-read it — the LEAN-fp16
short/cold-window trap again, cf. LF1) and **#60** (refute only budget-kept plies → 4×
fewer `vcf_refutations` solves, equivalence-proven). Next: #60 + a buffer-freshness
rethink before the next live race. Full synthesis:
`wiki/topics/white-side-defense-plan.md` § "LIVE RACE RAN, then KILLED (2026-06-19)".

## 2026-06-19 — White-defense reframe: recipe-deep + the gen bottleneck is the PER-PLY SOLVE; SPARSE-BITE lever live

Hobby-day session (Jason + Claude). Three results; full synthesis in
`wiki/topics/white-side-defense-plan.md` § "ROOT-CAUSE REFRAME + SPARSE-BITE LEVER".

1. **The white hole is RECIPE-DEEP.** The autolab's **from-scratch** `15x15-wdl@0`
   (0.44M, WDL head, no warm-start, no teacher) reproduces the warm 128×10 champion's
   white sweep vs native Rapfi-NNUE **to the game**: white **0-20** / black 11-9 @100ms
   (byte-identical to the champion's row), white 1-19 @1000ms. 7.5× smaller + from-scratch
   + different value head ⇒ warm-start, capacity, and the WDL head are all exonerated; the
   deficit lives in the self-play **data distribution**. (Corroborated by the 9×9 champion
   defending ≤5% white-loss under the same recipe — the hole scales with board size.)
   Artifact `sweep_logs/probe_wdl0_vs_rapfi_n40.jsonl`.
2. **The #43 "drowning" ROOT CAUSE = the per-ply VCF solve, not buffer size.** Clean 2×2
   smokes + a steady-state profiler: the policy teacher costs **~7.1 s/game (94% of wall)** —
   ~21 detection solves/game @ ~180 ms (incl. the StM-own-win guard's *second* solve) +
   refutation re-solves — slowing gen **~32× from scratch, ~78× on the mature net** (warm
   `wdl@0`: 1864 games/180s teacher-OFF → 24 teacher-ON). The 1.5M buffer was the symptom;
   the slow gen is the root. (8-worker smoke read ~59 s/game vs the profiler's clean
   7 s/game — cold-window contention inflation; trust the profiler.)
3. **SPARSE-BITE lever (live).** New flag `--defense-detect-frac F` (self_play
   `_DEFENSE_DETECT_FRAC`, default 1.0 = byte-identical) samples the EXACT solver to a
   fraction of four-threat plies — stamps stay exact, cost scales ~1:1. AZ distills a
   defensive lesson over epochs from a *present* signal; per-ply perfection isn't required.
   Profiler @ F=0.1: solves 21→2.2/game, teacher 7.08→1.68 s/game (~10×). 10% in a fresh
   150k buffer ≈ **1000× denser** than the #43 race that drowned in 1.5M. Defense tests pass
   (43), byte-identity preserved. **LIVE cell `G15-wdl-defense`** = `G15-wdl` (the from-scratch
   0/20 control) + ONE lever (policy teacher, `--defense-detect-frac 0.1`,
   `--defense-max-fraction 0.25`, caps 800/7) into a small fresh **150k** buffer,
   `--resume wdl@0`, 16 workers (board 15, mps, wandb offline). Gate: `eval_vs_rapfi.py
   --jobs 8`, watch white leave 0/20. Restartable.

Deferred (parked, with rationale in the synthesis page): the off-path relabel-worker (solve
is ~7 s/game *wherever* it runs → off-path only parallelizes, capping at the same ~2 g/s as
inline-16-workers; its real value is keeping the trainer fed via a skim — build it only if
sparse-bite's density is insufficient) and a **GPU-native conv-based threat-block teacher**
(dense, every-ply, no CPU/GPU bounce; shallow but that's what white lacks; the layered
endgame = cheap-dense-shallow + rare-deep-exact).

NOTE (machine): this session's code runs from the worktree `gomoku-white-defense-probe` via
an **editable-MAPPING repoint** of the shared mise venv's PEP-660 finder to the worktree
(the documented gotcha; PYTHONPATH can't shadow it). Restore to `~/code/gomoku/gomoku` at
session end. Changes are backward-compatible (new optional flag, default byte-identical), so
the repoint is benign for any other consumer.

## 2026-06-20 — CONCLUSION: white-defense is the first-player-win THEOREM, not a net flaw → pivot to swap2 (#22)

The day's white-defense investigation closed out decisively. The whole arc, in order:

1. **Recipe-deep** (probe): from-scratch `wdl@0` (0.44M, WDL head) reproduces the warm
   128×10 champion's white sweep vs Rapfi to the game (white 0-20 @100ms). Warm-start,
   capacity, value-head all exonerated.
2. **Sparse-VCF teacher null** (e1726, `--defense-detect-frac 0.1`): white 1-19/0-20 — no move.
3. **Dense conv block-teacher null** (e1240, `--defense-teacher-conv`, ~13.7 stamps/game,
   ~37% of plies): white 2-18/2-18 — a faint flicker inside the noise band, and black
   *softened* 55%→30% (traded attack for caution). Not a fix.
4. **Diagnostic** (`scripts/diag_white_failuremode.py`, 30 white-vs-Rapfi games on `wdl@0`):
   white blocks forced fours essentially perfectly (**Tier-1 error 5.6%**, the lone miss
   on an already-lost final ply); has its OWN immediate win on **1 ply across 30 games**
   (zero initiative); is forced into an unstoppable **double-four in 28/30 games**; losses
   run ~23 plies with defensive pressure rising toward the death. ⇒ "competent passive
   retreat to a forced loss," NOT tactical blunders. The stamp-teachers fix the one error
   class white doesn't make.
5. **Clincher** (`/tmp/rapfi_vs_rapfi.py`): **Rapfi(1000ms) vs Rapfi(1000ms)**, same
   4-stone openings, n=20 → **WHITE 1-9 (~10%), BLACK 9-1 (~90%).** Even the #1 Gomocup
   engine playing ITSELF gets crushed as the second player.

**Verdict:** 15×15 freestyle is a proven first-player win; from an empty/random opening
white is a (near-)lost role. No policy/value teacher can make a lost role win — there is no
error to correct. Every defense lever this project tried (FPU, search budget, value-only
teacher #42, sparse-VCF policy #43, dense conv) was fighting the theorem. **The fix is to
DELETE the doomed role: swap2** (Gomocup's balancing protocol — P1 places 3 stones, P2
chooses stay/swap/place-2-and-let-opponent-pick-color; the player is never *forced* onto the
lost side). This also makes the Rapfi yardstick honest (Rapfi is a swap2 engine) and is the
real Gomocup game. **Next build = #22 swap2**: the 3-stone opening placement, the 3-way
color-choice decision node, the net learning to negotiate the color, and a swap2 eval-vs-Rapfi.
The `--defense-detect-frac` and `--defense-teacher-conv` levers are sound, tested (60 + 40/40
VCF cross-check + 1500-fuzz), default-off byte-identical — kept as evidence, not a path forward.
Artifacts preserved in `sweep_logs/`: `probe_wdl0_vs_rapfi_n40.jsonl` (the wdl@0 control),
`probe_wdldefense_e1726_vs_rapfi.jsonl` (sparse-VCF), `probe_convguard_e1240_vs_rapfi.jsonl`
(conv), `diag_white_failuremode_wdl0.jsonl` (the diagnostic). The large intermediate training
checkpoints (sparse e1726, conv e1240) were discarded with the worktree on merge — a negative
result we are not resuming; the eval JSONLs above are the cited evidence. The cells
`G15-wdl-defense` / `G15-wdl-conv` in `run_sweep.py` reproduce the runs from the wdl@0 seed if
ever needed.

## 2026-06-20 — SWAP2 (#72) BUILT (full Path A) + LIVE warm-started run launched — the real white fix is to DELETE the doomed role (wandb `8nq1a7cm`)

**This is the LIVE arm.** Acting on the same-day conclusion above (15×15 freestyle white
weakness is the **first-player-win THEOREM**, not a net flaw — Rapfi-vs-Rapfi from 4-stone
openings = white 1-9; three teachers all flattened: value-only #42, sparse-VCF #43, dense
conv), the fix shipped: **swap2** (Gomocup's balancing protocol). On `feat/swap2-opening-protocol`
(6 commits `ba37b92..167e526`, **73 tests green, NOT merged**).

**The ML thesis (the mechanism, not just an honest yardstick).** An imbalanced game **cannot
bootstrap** because self-play data collapses: every game is a black win → the value head only
ever sees `white = lost` → the policy gets **no gradient on winnable white positions** (there
are none in the data). No teacher can fix a role the data never shows winning. **Swap2 rebalances
the GAME** so self-play generates **~50/50 data** → white positions become *winnable in the
training set* → the loop can finally learn to defend because there is now a signal to learn from.
That is the actual unlock; the honest-yardstick property (Rapfi is a swap2 engine, so the real
Gomocup game) is a bonus, not the point.

**What was BUILT — full Path A (the net learns to negotiate), 6 pieces:**
1. **`gomoku/swap2.py`** (`ba37b92`) — pure negotiation state machine (`OpeningState`). Opener
   places 2B+1W; responder STAY (take white + place a white stone) / SWAP (take black) / PLACE2
   (add 1B+1W, opener then picks color). **Key modeling:** color is fixed by placement ORDER, so
   every placement stays a spatial move over the existing board policy head — **no action-space
   growth** for placements; only the two negotiation *moments* are abstract (a width-3 choice
   space). Value attribution uses explicit OPENER/RESPONDER actor tags + `backup_sign()` (the
   opener acts **3× in a row**, so "flip perspective every ply" does not hold).
2. **`gomoku/external_engine.py`** (`2d56314`) — SWAP2BOARD protocol path
   (`swap2_open`/`swap2_respond`/`swap2_pick` + `Swap2Reply`), eval-only, additive; the existing
   move path is byte-identical.
3. **`gomoku/model.py`** (`9e24e45`) — width-3 choice head via `forward_with_choice()` (taps off
   the value head's penultimate layer); **warm-start-tolerant load** (the champion predates the
   head → core loads strict, the choice head starts fresh). This is what makes it full Path A.
4. **`gomoku/swap2_search.py`** (`5860326`) — v1 negotiator: `negotiate(oracle, rng)` drives the
   opening. PLACE nodes are **sampled** for diversity; CHOICE nodes are selected by a **one-ply
   VALUE comparison** (no trained head needed in v1) with honest minimax over the nested opener
   pick; choice records are emitted as **future choice-head targets**. ~30 net forwards/game
   (~0.2% overhead, no MCTS).
5. **self-play wiring** (`6b629dc`) — `--swap2` flag threaded into all four generation paths
   (`self_play` / `selfplay_worker` / `train` / `run_sweep`) at the `_random_opening_state` seam;
   **mutually exclusive with `--random-opening-moves`**; byte-identical when OFF.
6. **`gomoku/eval_swap2.py`** (`55f3e4d`) — the honest gate: **both sides negotiate** (our net
   via `swap2_search.agent_act`, the engine via SWAP2BOARD), roles alternated, normal play from
   `to_normal()`; result splits by our final color + role. **There is no forced-white side.**

**The live run (cell `G15-swap2`, `167e526`) — LAUNCHED, results pending.** Clone of champion
`G15-128x10-bigbuf` (128×10 large, scalar value, global_pool, value-discount 0.98, gumbel) +
**TWO deltas**: (a) buffer **1.5M → 150k FRESH** — the swap2 lesson lives in the new ~50/50
games and a small fresh buffer turns over fast (the 2026-06-19 small-fresh-buffer finding), and
(b) **`--swap2`**. `n_workers 4 → 8` (the negotiation has **no VCF solver**, so no
solver-starves-gen trap). **Warm-started** from a weights-only stripped champion
(`/Users/jason/data/swap2/g15_champ_warmstart_weightsonly.pt` — no embedded buffer/optimizer/
wandb → fresh everything; the tolerant loader adds the fresh choice head).
- Launched **2026-06-20 ~07:40**, wandb run **`8nq1a7cm`**, board 15, MPS, 1h self-capping slices
  via `--max-wall-secs 3600 --run-base /Users/jason/data/swap2`. Spin-up healthy: ~2 s/game swap2
  gen, ~3.9 games/s aggregate, 0 errors. Babysit ledger: `/Users/jason/data/swap2/babysit/ledger.md`.
- **GATE:** `gomoku/eval_swap2.py` vs native Rapfi-NNUE (`run-rapfi` wrapper, `GOMOKU_REPO=main`
  for the weights) — **overall win% under the real protocol, NO forced-white floor**, re-measured
  each ~1h. Baseline context: the champion under the OLD forced-opening measure scored ~21-27%
  overall / white 0/12 swept; under swap2 there is **no forced-white side** — so the gate reads the
  honest, balanced number for the first time.

**Status: results pending the hourly `eval_swap2`-vs-Rapfi gate.** Next session: read the babysit
ledger, then the gate output, and watch whether the ~50/50 data lets the loop bootstrap a defending
white (the theorem says it can't be taught into a *forced* white role; swap2 removes the force).
Full plan + theorem chronology: `wiki/topics/white-side-defense-plan.md`.

## 2026-06-20 — SWAP2 (#72) LIVE: THE CORE BET IS CONFIRMED AT THE DATA LEVEL — white wins 27% in swap2 self-play (vs ~0% empty-board); white positions are now WINNABLE in the training set (wandb `8nq1a7cm`)

**~2h into the live warm-started swap2 run (cell `G15-swap2`, wandb `8nq1a7cm`, board 15, MPS),
the central hypothesis behind #72 is confirmed where it matters most — in the self-play DATA.**
The build + launch are the entry directly above; this entry records the first measured results.

**1. THE CORE BET IS CONFIRMED — white is now winnable in the training data.** Measured color
balance of the **64 most recent swap2 self-play games** (pulled from the live run's `_records`
GameRecords): **white wins 27% (black 69%, draw 5%).** In the OLD empty-board self-play regime
white won **~0%** — the imbalance collapse that made white-defense unlearnable, the entire reason
#72 exists. Under swap2, white is genuinely WINNABLE in the training data (**27% ≫ 0%**), so the
value/policy heads finally get gradient on **winnable white positions**. This is the bootstrap an
imbalanced game *cannot* do, working — exactly the ML thesis ("swap2 rebalances the GAME so
self-play generates ~50/50 data → white positions become winnable in the training set"). **This is
the single most important result of the run so far.** The white-defense teachers (#42 value-only,
#43 sparse-VCF, dense conv) all failed because there was no error to correct in a *forced* lost
role; swap2 instead supplies the missing signal by making the role winnable.

**2. The negotiation mechanism works.** In net-vs-net swap2 H2H, the **RESPONDER wins ~80%** — it
exploits its stay/swap/place2 choice to take the better side (`opener_color_dist` shows the
responder almost always grabs black). Swap2's balancing comes through the responder's choice,
exactly as designed.

**3. Not yet perfectly balanced — the honest caveat + the identified NEXT LEVER.** Black still
wins 69% (not 50/50) because **v1 SAMPLES opening placements for diversity rather than TRAINING
them** — the opener never learns to place a FAIR opening, so the responder retains a swap-to-black
edge. Pushing toward 50/50 = **train the negotiation.** The machinery is half-built: the width-3
CHOICE HEAD exists (`model.forward_with_choice`) and the negotiator already emits `choice_records`
as targets, but **those targets are NOT yet wired into the trainer loss** — v1 negotiates by a
one-ply value lookup, with no trained choice head. **Next lever: wire `choice_records` into
training** (+ optionally record/train opening placements so the opener learns fair openings). This
is the identified next step toward 50/50, not a failure of the run.

**4. The Rapfi gate is noise-dominated near the floor; the progress gate is now H2H-vs-frozen-
champion.** Vs Rapfi-NNUE @200ms our net sits at single-digit-to-~30% win-rate; at **n=16–48 the
SAME fixed baseline reads 4%–25% on noise** (variance swamps the ~5–8pt signal). Per the wiki's own
2026-06-15 rule ("gate did-this-help on H2H vs the preserved champion, not Rapfi"), the progress
gate is now **net-vs-net swap2 H2H vs the frozen warm champion** (near p≈0.5, resolvable; built in
`gomoku/eval_swap2.py:eval_swap2_h2h`, jobs-parallel, exact-deterministic). **First reading: trained
e129 vs frozen warm champ = 51.6% (n=64) — PARITY within noise, EARLY (~2h warm-started). This is
NOT a strength claim.** Rapfi stays a coarse absolute anchor only.

**5. Run mechanics — healthy throughout.** Cell `G15-swap2` (champion recipe + 150k fresh buffer +
swap2 lever), warm-started from a weights-only stripped champion, 1h self-capping slices, **~3
slices so far**. Dynamics healthy: **value loss bounces ~0.16–0.26** (no value-poisoning collapse),
**plies rose ~30 → 42** (more contested games, the expected swap2 signature), **no fast-attack
collapse** (the `selfplay/plies_mean` death-tell is absent).

**Net read (~2h in):** the run does the one thing the three teachers could not — it makes white
**winnable in the data** (27% vs ~0%), supplying the gradient the imbalanced game starved. Strength
vs the frozen champion is at parity/early (51.6%, n=64), so this is a confirmed *mechanism*, not yet
a confirmed *strength gain*. The path to 50/50 balance is concrete: train the negotiation (choice
head into the loss). Branch `feat/swap2-opening-protocol` (latest commit `a29e645` adds the
net-vs-net swap2 H2H gate); evidence is the live run `8nq1a7cm`. Plan + theorem chronology:
`wiki/topics/white-side-defense-plan.md`.

## 2026-06-20 — SWAP2 (#72) STRENGTH SIGNAL FIRES: H2H 51.6%→64.1%, WHITE 12%→41% (e129→e181)

The H2H-vs-frozen-champion gate (the resolvable progress gate; net-vs-net swap2, both
negotiate, n=64 / sims=200 / seed7) now has TWO comparable points and the trend is up —
specifically on the white side, the metric the project chased for months.

| gate | epoch | overall (trained's W-L) | as WHITE | as black | as opener | as responder |
|---|---|---|---|---|---|---|
| slice 2 end | e129 | 51.6% (33-31) | 12% (3-22) | 77% (30-9) | 22% (7-25) | 81% (26-6) |
| slice 3 end | e181 | 64.1% (41-23) | **41% (12-17)** | 83% (29-6) | 50% (16-16) | 78% (25-7) |

**Read.** Overall 51.6→64.1 (n=64 each, CI ~±12%) is suggestive but partly noisy. The
robust, thesis-consistent signal is **white: 3/25 → 12/29 white wins (12%→41%, a 4×
increase)**, plus opener-role 22%→50%. White is becoming viable via *balanced data*, not
a teacher — exactly the swap2 thesis. This is the first positive STRENGTH signal (prior
entry had it at parity/51.6%); ~52 epochs of balanced data on top of the warm start moved
it.

**Epoch context (Jason, 2026-06-20).** General AZ wisdom is "thousands of epochs to
move," but this lab's lived experience is real movement in **~100 epochs** (laptop-scale,
small buffer, high SGD/position). So a white shift at e129→e181 is *on-schedule for this
setup, not anomalous* — credible, not suspicious. "Thousands" is the conservative outer
bound; ~100 is the empirical inner bound here.

**Discipline.** A single n=64 gate is a data point; the RESULT is the trend across
INDEPENDENT checkpoints. Next gate (slice 4, e~233) at **n=128** to tighten the CI — if it
holds ~60%+ with white ~40%, this is a genuine result. Caveat tracked: not yet 50/50
balance (black still 69% in self-play data) because v1 samples (doesn't train) opening
placements — the learned-choice-head lever (§6 of the synthesis page) is the path to push
further, now a "go further" lever rather than a rescue. Synthesis +
high-res trend table: `wiki/topics/swap2-opening-protocol.md` §5.3. Run `8nq1a7cm`.

## 2026-06-20 — SWAP2 (#72) gate-4 CONFIRMATION (n=128): white 12%→33% holds; overall ~57% (e181's 64% was noise)

Tighter n=128 H2H (trained e235 vs frozen champ, sims=200/seed7): **73-55 = 57.0%** —
white 20-40 (33%), black 53-15 (78%), opener 26-38 (41%), responder 47-17 (73%).

Trend across INDEPENDENT checkpoints: e129 51.6%/w12% (n64) → e181 64.1%/w41% (n64) →
e235 57.0%/w33% (n128). The e181 64.1% was upward n=64 noise; the n=128 anchor is ~57%
overall (CI ~[48,66], grazes 50% → suggestive not conclusive on the overall). The
**white side is the robust signal: 12% (3/25) → 33% (20/60)** survives the tighter n — a
real white-defense gain via balanced data, not a teacher. Verdict ~235 epochs: confirmed
but MODEST, not plateaued — keep training (this lab's ~100-epoch movement window). All
future gates n=128 (n=64 too noisy). High-res table: `wiki/topics/swap2-opening-protocol.md` §5.3.

## 2026-06-20 — SWAP2 (#72) gate-5 (e289, n=128): 66.8%, white-LOSS 88%→67%→51% — at the crowning bar

Trained e289 vs frozen champ, n=128 sims=200 seed7: **85-42-1 = 66.8%**. white 29-30-0
(LOSS 51%), black 56-12-1 (81%), opener 33-31 (52%), responder 52-11-1 (82%).

Reliable-anchor trend (n=128 + e129 baseline): overall 51.6%(e129) → 57.0%(e235) →
66.8%(e289); **white LOSS-rate 88% → 67% → 51%** — falling cleanly on the exact metric
#18/#72 targeted. (The e181 n=64 64.1%/59%-white-loss was an upward overshoot — excluded.)
At e289 the overall CI ~[58.6,75] clears 50% AND the ~58% relative-crown lower bound
(§6.6) → "stronger than the champion" is essentially AT the bar; formal crown wants an
n≥200 gate. Still CLIMBING (e235→e289: 57→66.8), not plateaued — keep training toward
e1000. White winning ~half its games vs the frozen champ's defense is a RELATIVE signal
(beats the OLD champ's white play, not a game-theoretic white win). High-res table:
`wiki/topics/swap2-opening-protocol.md` §5.3.

## 2026-06-20 — SWAP2 (#72) gate-6 (e345, n=128): 76.2%, white-LOSS 88%→67%→51%→42% — slope STEEPENED

Trained e345 vs frozen champ, n=128 sims=200 seed7 jobs=16 (runtime 755.99s): **97-30-1 =
76.2%**. white 21-16-1 (win 55.3% / LOSS 42.1%), black 76-14-0 (84.4%), opener 43-20-1
(67.2%), responder 54-10-0 (84.4%). opener_color_dist {black 38, white 90}.

Reliable-anchor trend (n=128): overall 57.0%(e235) → 66.8%(e289) → **76.2%(e345)**;
**white LOSS-rate 88%(e129) → 67%(e235) → 51%(e289) → 42%(e345)** — monotonic, and at e345
**below 50% for the first time** (white wins MORE than it loses vs the champ's defense).
The e289→e345 jump is **+9.4 overall pts — the LARGEST slice-over-slice gain so far**, so
the slope STEEPENED this slice; this is acceleration, not the onset of a plateau. Overall
CI ~[68.8, 83.6] sits well clear of 50% AND the ~58% relative-crown lower bound (§6.6).
Caveat: white sample is small (n=38 white games, CI ~±16% on white-LOSS) → treat the white
number as directional even though the overall n=128 number is tight. (e129 51.6% and e181
64.1% were n=64; e181 was upward noise — both excluded from the reliable anchor line.)

Verdict at ~345 epochs: **climbing, slope steepened, NOT plateaued** — keep training. The
relative-crown lower bound is comfortably cleared; the formal crown still wants an n≥200
gate, deferred to plateau or e1000. White winning the majority of its games vs the frozen
champ's defense is a RELATIVE signal (beats the OLD champ's white play, not a game-theoretic
white win). High-res table: `wiki/topics/swap2-opening-protocol.md` §5.3. Run `8nq1a7cm`.

## 2026-06-20 — SWAP2 (#72) gate-7 (e403, n=128): 67.6%, PULLBACK within noise — corrects the gate-6 "slope steepened" read

Trained e403 vs frozen champ, n=128 sims=200 seed7 jobs=16 (runtime 844.39s): **86-41-1 =
67.6%**. white 25-33-0 (win 43.1% / LOSS 56.9%), black 61-8-1 (87.1%), opener 32-32-0
(50.0%), responder 54-9-1 (84.4%). opener_color_dist {black 18, white 110}.

Reliable-anchor trend (n=128): overall 57.0%(e235) → 66.8%(e289) → 76.2%(e345) →
**67.6%(e403)**, mean **~70%**; white LOSS-rate 67%(e235) → 51%(e289) → 42%(e345) →
**57%(e403)** — no longer a clean monotonic fall, now BOUNCING 42-57% (~ parity).

**This corrects the gate-6 "slope steepened" read: that was partly upward noise.** e345's
76.2%/42% was *partly* an upward sample-fluctuation (same flavor as the earlier e181 64.1%
n=64 spike), not a genuine acceleration. The e403 dip is a **PULLBACK within noise, NOT a
regression** — the CIs overlap heavily: overall e403 [59.5%, 75.7%] vs e345 [68.8%, 83.6%];
white-LOSS e403 [44%, 70%] vs e345 [26%, 58%]. The true LEVEL is **~70% overall**, holding
across the n=128 anchors (66.8 → 76.2 → 67.6), comfortably above the ~58% relative-crown
lower bound (§6.6).

The honest white-side statement is now: **white LOSS-rate fell from 88% (early) to ~50%
(now) and is fluctuating around parity, NOT monotonically marching to 0.** The early read
(88→67→51→42) was real *as a fall from the catastrophic floor*, but at the ~parity level the
gate-to-gate motion is noise. Black (87%) and responder (84%) sides stay strong; all the
variance is concentrated on the hard white/opener side, which also carries the smaller
sample (white n=58 this gate vs n=38 last — the negotiation/seed interaction shifts the
color mix gate-to-gate).

Verdict at ~403 epochs: **NOT a plateau, NOT a regression, still well above the crown bar —
keep training.** Recalibrate expectations: progress on the white side is **noisy around
parity, not a smooth descent**; read the trend across independent checkpoints *and their
CIs*, not any single gate (a single high gate can be an upward fluctuation, as e345 partly
was). Formal crown gate stays at n≥200, deferred to plateau or e1000. High-res table:
`wiki/topics/swap2-opening-protocol.md` §5.3. Run `8nq1a7cm`.

## 2026-06-20 (night) — Era-2 board-size LADDER launched (9→11→13→15)

Pivoted era-2 from "9×9 → warm-start straight to 15×15" to a **board-size ladder**.
Trigger: 9×9 swap2 **saturated into draw-dominance at e102** (last-3 epochs draws
56/75/56 vs white ~14, black ~19–31; run `lywhy1ba`) — exactly Jason's graduation
rule `max(draw, white, black) == draw`. 9×9 is too cramped for white to convert
defense into a win, so it learns "draw"; a bigger board reclaims room. Native exts
compiled for 11/13 (native-11 **68.3k** > native-15 **60.7k** sims/s; pure-Python
~40k flat regardless of size → **never fall back**, it would break Δelo/hour). Each
rung warm-starts the previous champion (98.9% transfer, only the 3 board-bound FCs
re-init) and trains until draw-dominant, then steps up. Rung 15 terminal (board too
big to draw). Cells `G-ladder-11/13/15`; orchestrator `babysit/ladder_autochain.sh`;
graduation `babysit/ladder_grad.py` (reads `wandb_run_id` from `latest.pt`).
Synthesis: `wiki/topics/board-size-transfer-and-warm-start.md` § the multi-rung ladder.

Run IDs: 9×9 `lywhy1ba` (graduated e102) → 11×11 `8jsd7qzw` (live). 11×11 at e73:
white%dec 30–42% (balanced, black slight edge), plies up to ~38–40 (vs 9×9 ~20),
draws creeping to ~8–11% — **saturation onset beginning, same shape as 9×9**.

**Jason's predictions (logged before the fact, for posterity — check against actual
cutover epochs in the AM):** (1) happy if white can "fight and learn" at 11 or 13 at
all; (2) 11×11 will drawmax **~30–50% later than 9×9** (so "pretty soon"); (3) 13×13
will drawmax **much later**; (4) draws already rising at 11 → 13 cutover maybe sooner
than expected; (5) success = "still training in the morning."

**Cadence (unattended, NO gates — "just see what happens"):** monitor every 30 min on
rungs 11/13. On reaching **15×15**, switch to **1-hr cadence with a Rapfi eval each
lap** (try-vs-Rapfi → record → 1 h train → repeat). If something goes sideways,
consult the wiki for the fix and keep it TRAINING — do not gate or stop. STOP control:
`touch babysit/STOP_ladder`. Keep updating wiki/TRAINING_WIKI at each check-in.

### 2026-06-21 02:50 — rung 11 → 13 cutover (graduated on CAP, not drawmax)

Rung 11 (`8jsd7qzw`) ran e0→**e401** and graduated via the **CAP=400 backstop**, NOT
the drawmax rule. Why: from ~e218 it settled into a **stable black-edge equilibrium** —
black ~40% / draw ~34% / white ~25%, plies flat ~64 — for ~180 epochs. Draws flickered
to single-epoch drawmax (e166, e221, e380–381 hit 44–52%) but **never robustly overtook
black**, so the 3-consecutive-drawmax denoise correctly never fired. Interpretation:
with **v2a (choice head) OFF**, swap2's color-balancing isn't trained, so black keeps its
intrinsic first-move edge; white's defense maxes out at "draw-or-lose-narrowly" rather
than forcing draw-dominance. The CAP backstop is exactly the right mechanism for this
"strong rung that plateaus short of drawmax" case — it advanced cleanly.

11→13 warm-start at e401 (`ladder_seed_13.pt`); rung 13 = run `2dvcxh0b`. Early 13×13
(e73–77): **white%dec climbing 40→49%**, draws 0, plies ~20 (fresh-board recovery, many
decisive games) — a clean transfer, even more balanced than 11 started. "Fight and learn
at 13" (Jason's happy condition) achieved out of the gate.

**Prediction scoring so far:** Jason guessed "11 drawmaxes ~30–50% later than 9 (e133–153)."
Reality: **11 never cleanly drawmaxed** — it hit a stable equilibrium and graduated on the
CAP at e401. So the *drawmax framing* didn't hold for 11 (equilibrium instead); the deeper
instinct (9×9-style saturation transfers up the ladder) gave way to a black-edge fixed
point once the board had room. "13 drawmaxes much later" — TBD; 13 has even more room, so
expect equilibrium-or-CAP again rather than a clean drawmax.

### 2026-06-21 07:40 — FULL LADDER COMPLETE: reached 15×15 (9→11→13→15 overnight)

The whole curriculum climbed unattended in one night. Timeline + how each rung graduated:

| rung | run | epochs | graduated | how |
|---|---|---|---|---|
| 9×9  | `lywhy1ba` | →e102 | 2026-06-20 ~22:50 | **drawmax** (draws 56/75/56 vs white ~14) |
| 11×11 | `8jsd7qzw` | e0→e401 | 02:50 | **CAP** (stable black-edge equilibrium, never drawmaxed) |
| 13×13 | `2dvcxh0b` | e0→e424 | 07:40 | **CAP** (same equilibrium; black edge *stronger*, draws rarer ~10%) |
| 15×15 | (live) | e0→ | terminal | runs until `STOP_ladder` |

**Durable lesson: only the smallest board (9×9) cleanly draw-saturates.** With v2a OFF
the swap2 negotiation doesn't balance colors, so on 11 and 13 black keeps a genuine
first-move edge (white ~25–35% of decisive games, plies long/healthy ~50–70 — defending,
NOT the 0% basin) and draws never overtake black. The **CAP backstop is therefore the
real graduation mechanism for the bigger rungs**, not the drawmax rule — and that's fine,
it advanced each rung cleanly. White "fights and learns" at every rung (Jason's bar met).

15×15 starts fresh-headed (the 13→15 transfer re-inits `policy_fc`/`value_fc1`); first
epochs are empty/raced (workers warming on slow ~plies-134 games). **Now on the 1-hr
Rapfi cadence** (`babysit/ladder_rapfi15.sh`, gentle/concurrent): baseline read first,
then eval-every-hour while training, recording white-vs-Rapfi off the era-1 0% floor.
**Baseline (07:45, fresh 15×15 net `epoch0095`, untrained heads): 0.0%** vs Rapfi-NNUE
(0W-32L-0D @ 200ms/sims200/n32; black 0/21, white 0/11). The floor — the transfer re-inits
the 15×15 heads so the net can't play coherently yet; loses fast (59s). Now we watch it
climb off 0% as the heads adapt (the whole bet of the ladder). Hourly reads append to
`babysit/eval_results.jsonl`; era-1's *trained* e455 was 10.2%/white-0% for comparison.

## 2026-06-21 ~11:07 — era-3 FAIR-OPENING LADDER launched (9→11→13→15)

Jason's "go all in": build the fixed-fair-opening run as a LADDER for cheap epochs.
Openers = Rapfi's 9 shapes RE-CENTERED per board (not generated — his call; the
shapes are sub-9×9 so all 9 fit on 9/11/13, zero dropped). A FRESH net climbs
9→11→13→15, same canned fair openers at every rung, **auto-promoting on p90-plies-max**
(#74, `babysit/ladder_grad.py`: graduate when plies_p90 plateaus at peak for 5 epochs
— promote before the net learns to retreat). Minimal gating on 9/11/13 (bank cheap
epochs, find a fresh killer); real gates at 15, incl. a TODO Rapfi-from-canned-openers
eval (our net vs Rapfi from the fixed post-opening positions, as black AND white).

Cells `G{9,11,13,15}-fixed-openings` (swap2 OFF, fixed_openings=True; commits 3c6e9d7,
744849a; tests green at all sizes). Orchestrator `babysit/fairladder.sh` (15-min slices,
warm-start between rungs). Rung-9 run `eilfnz1e`. Single-15 predecessor run `nbctsiua`
(stopped; showed white ~46–51% in its first epochs — the early fairness signal). era-2
swap2-ladder best net preserved: `G-ladder-15-board15/checkpoints/epoch0235.pt` (25% vs Rapfi).

THE METRIC: white-share of decisive self-play → ~50% on fair boards (was ~25–35% rigged).

## 2026-06-22 — Single-opener "Bruce Lee" 15×15 overnight + the worker-count / gen-flood finding ⭐

Run `gogpmbhw` (`G15-fixed-openings-board15`). Context: by 2026-06-21 night the
ladder had pivoted to a **single fair opener (idx-2** `((3,2),(5,4),(4,5))`, B,W,B →
white-to-move) after discovering that *re-centering* Rapfi's shapes does NOT preserve
balance (re-centered openers tested 0–95% black at 13×13; idx-2 was the fairest at
~50%). Jason's call: "drop anything that isn't fair first … an exceptionally strong
player from ONE fair opening beats an imbalanced player from many" — `GOMOKU_DROP_OPENERS`
keeps only idx-2. Specialize, don't generalize: **"Bruce Lee" — fear the man who
practiced one kick 10,000 times.** Plan: plow forth even if white lags; hold all
verdicts until the 15-series has depth.

**Overnight (8 workers, e224→~616, hands-off).** Three fixed rulers, idx-2 board, both
seats, 16 games/seat, `temp_plies=6` opening variety:
- **vs Rapfi** (saturated ceiling): **0/16 all night**, unmoved — expected; Rapfi reads
  clean through us at this strength and will for a long while.
- **vs champ0235** (era-2 best, warm-started from prior winners): bounced ~even, peaked
  **69%**. Trading blows with the old champion on its home board — promising for the recipe.
  (The early "16–0 sweep" was a *determinism artifact* — one line repeated 8×; it vanished
  once `temp_plies` variety was added. **Lesson: net-vs-net MCTS is deterministic → flat
  series; always sample opening plies for a real H2H read.**)
- **vs self126** (FROZEN e126 self, the sensitive "am I improving?" probe): climbed
  37→62→56→**75**→… settling a touch above 50. Tipped positive — beating its own past self.
- **self-play balance**: sloshed 39–62% white, no trend, no collapse. `vl` halved over the
  night (0.082→0.043) — value head sharpening while the policy thrashed (Jason's
  spidey-sense/decision split: it *knows* who's winning before it can reliably *decide*).

Verdict held: **trading real blows, not winning the war** — stronger than past-self,
even-ish with the milestone champ, can't touch Rapfi. A healthy *developing* net. (Jason's
pure-vibes pre-call — "won't win yet but will start to win" — landed.)

**The lever — buffer balance (8→4→3 workers, 2026-06-22).** Jason noticed self-play was
out-generating training: `train/sample_reuse_ratio ≈ 0.67` (consumed/ingested). At reuse
<1 with random sampling (~Poisson(0.67)), **~51% of generated positions are evicted from
the 150k buffer never having had a single gradient step** — a good move can be played and
never learned from. The 8-worker config was *flooding* the trainer. Cut workers, resumed
from `latest.pt` (weights+buffer; preserved copies `babysit/snapshots/PRE4_*`):

| workers | reuse (per-cycle) | buffer age_p90 | s/epoch | epochs/min |
|--------:|------------------:|---------------:|--------:|-----------:|
| 8       | ~0.67 (½ unseen)  | ~0 (firehose)  | 65.2    | 0.9        |
| 4       | ~1.4              | rising         | 36.5    | 1.6        |
| 3       | **~3** (2.2–4.4)  | ~15–20         | **26.1**| **2.3**    |

**The finding (⭐ the gen-flood double-tax):** flooding the buffer cost us on TWO axes at
once — sample-efficiency *and* wall-clock. At 8 workers each epoch drowned in inflow
(~128 new games/cycle → buffer inserts, cross-game-store updates), so epochs were both
*less useful* and ~2.5× *slower*. Dropping to 3 fixed both: reuse → ~3 (firmly normal-AZ;
each position studied ~3×, age_p90 off zero), and **65→26 s/epoch**. This is the SAME
gen-flood pattern already documented at the 96×8 cell (`run_sweep.py` ~L352: "8 workers
FLOODED it, per-epoch ingest ran away 62→313s"); it resurfaced at 15×15 fixed-openings.
**Takeaway: `n_workers` is a first-class buffer-balance knob, not just a throughput knob —
target reuse ~1–4; reuse <1 is self-sabotage (slower AND lossy).** Note 8→3 dropped
ingestion *more* than linearly (new_games/cycle 128→16–24), so 3 workers overshot the ~1.8
projection to ~3 — fine, still healthy AZ. `n_workers=3` is an uncommitted live toggle in
the worktree; flip to 4 for reuse ~1.8 if ~3 feels deep. Whether deeper reuse settles the
balance slosh is still open (the run was time-boxed by AC power).

**Babysit infra built this session** (`/Users/jason/data/swap2/babysit/`):
`rapfi_opener_eval.py` (vs Rapfi, idx-2, both seats), `champ_h2h_eval.py` +
`champ_h2h_cadence.sh` (vs self126/champ0235, `temp_plies` variety), hourly
`snapshot_loop.sh` (preserved last-good ladder; `latest.pt` embeds the buffer so each is a
real resume point). Tank-restart safety net agreed but kept *manual* (auto-restart-at-4
would fuse recovery with the reuse experiment → unreadable; a hard collapse is itself the
night's best data, not something to erase).

## 2026-06-23 — 1M buffer overnight: a clean NEGATIVE result + the recency-curator fix ⭐

Run `gogpmbhw` resumed from e877 with `buffer_size` **150k → 1M** (#73 follow-up; bit-packed
so 1M ≈ 1.3GB). Hypothesis: a longer consolidation window would steady the white/black slosh.
**Result: it did NOT.** ~930 epochs overnight (e877→e1804, ~2 epochs/min, no crashes/tanks):
- **slosh did not narrow** (white% band stayed ~30–60 pts, arguably wider).
- **pl/vl flatlined and smoothed** — vl crept 0.057→0.050 then leveled; Jason's read: *"that's
  not learning."* Correct.
- `buffer/age_p90` climbed **linearly** (13→285→545→897) instead of plateauing at the ~FIFO
  window — the tell.

**Root-cause (corrected mid-investigation):** the buffer (`gomoku/replay_buffer.py`) is a **FIFO
ring** (eviction overwrites oldest). The pathology was NOT eviction — it was **`_recency_frac=0.0`
= UNIFORM sampling over a ring so large it spanned the whole run.** Uniform-over-all-history makes
the training distribution **stationary**, so the loss converges to a fixed point and stops chasing
the improving policy → flat pl/vl, persistent slosh. (Jason had remembered adding "reservoir" to
fight collapse; the real mechanism is uniform-sampling-over-a-huge-ring, and collapse was actually
the provable first-player/black edge, NOT a buffer issue — so that fix was for a misdiagnosis and
had been quietly taxing learning.)

**The fix (already coded, just OFF):** `configure_curator(recency_frac, recency_window)` /
CLI `--buffer-recency-frac` — the #17 recency curator. Draws a fraction of each batch from the
most-recent `--buffer-recency-window` (200k) positions, rest uniform. Crucially this is a
**validated lever**: `run_sweep.py` history shows it was a **+90-elo derby winner** (v8
"buffer-comp", `--buffer-recency-frac 0.5` in multiple proven cells). KataGo's design exactly:
big window for memory, recency-weighted sampling for freshness.

**Experiment now LIVE (2026-06-23):** rewound to `snapshots/SHUTDOWN_e877` (clean pre-stationary
baseline), kept the 1M ring, flipped **`--buffer-recency-frac` 0 → 0.5**. ONE variable changed vs
the overnight, so any effect is attributable to the curator. **Watch `loss/policy` + `loss/value`:
if they come back ALIVE (keep decreasing) the recency lever is working;** secondary: does the slosh
band finally narrow. `n_workers=3`, reuse ~3, ~2.8 epochs/min. Resume point preserved as
`snapshots/PRE4_e601` (8w-era) + `SHUTDOWN_e877` (pre-1M).

### 2026-06-23 (later) — recency-0.5 VERDICT: loss alive, strength flat (the plateau is buffer-knob-proof) ⭐

Ran ~e877→**e2300** under recency-0.5 (~1400 epochs). **What it did:** broke the stationary plateau —
`loss/policy` went from smooth-dead (uniform, std 0.034) to alive-and-oscillating (recency, std 0.052 at
matched epochs); the loss *moved* again. **What it did NOT do:** improve strength. On-demand 3-ruler eval
@ e2300 (n=16/seat, idx-2, opening variety):
- **vs self126 (frozen e126 self): 37.5%** (6–10) — *below even*, losing to its own past self
- **vs champ0235 (era-2 milestone): 46.9%** (7–8–1) — slightly below even
- **vs Rapfi (ceiling): 0/16** — unmoved

Squarely in the same plateau band every self-play variant has occupied (even-ish vs beatable rulers,
0 vs Rapfi); flat-to-slightly-down vs the e793 read (40.6 / 56.2 / 0). **Conclusion: keeping the loss
alive ≠ keeping strength climbing.** Recency = *perturbation/mutation* that churns in place without a
*selection* mechanism to cash it (see swap2 §13). We have now exhausted three data-pipeline levers —
**reuse** (n_workers), **window** (buffer_size 1M), **freshness** (recency_frac) — and strength has not
moved off the self-play ceiling. **The plateau is real and buffer-knob-proof; the only lever left that
points up is an external TEACHER** (#46 curriculum / #18 exact-solver / distillation). Bruce-1 continues
as the self-play-only *baseline-to-beat* for the teacher era.

### 2026-06-23 (later still) — BUILT the eval+teacher sensei (the lever up after the knobs)

After the recency-0.5 verdict closed the self-play-knob era (three data-pipeline levers
exhausted; the plateau is buffer-knob-proof; teachers are the only lever that points up), built
the keystone: an **eval+teacher sensei** on `feat/eval-teacher-sensei` (off main). One subsystem,
two faces — the eval is the teacher's measuring stick AND its selector.

**Eval face (closes #34):** a warm `RapfiPool` (persistent NNUE processes, no per-pass respawn —
the 10×+ win) behind an HTTP daemon (`gomoku-eval-daemon serve`) PLUS a flatfile-reducer cadence
loop (`… cadence`) that watches a checkpoint and appends a per-color-split JSONL series every N
epochs. **White is reported separately** (the hard #34 constraint): every row carries
`white_score`/`white_loss_rate`/`white_wld`, never folded into the aggregate. CPU-only → never
competes with the MPS trainer (same property that makes the babysit cadence safe). Rulers map to
the babysit set (self126/champ0235/rapfi) at the fixed idx-2 opening via a new additive
`play_match_pickers(start_state=…)` seam (default None = byte-identical). `EvaluatorCache` keeps
fixed-ruler nets warm across epochs; `run_panel` pins one (weights, epoch) snapshot so the series
is never mislabeled.

**Teacher face (advances #46/#18/#44):** Rapfi exposes only a MOVE, and #18/#44 say the policy
must carry the load and value-only teaching is structurally wrong — so the teacher is **policy-side
one-hot distillation** ("the master plays here"), value untouched. `python -m gomoku.teacher
generate` self-plays from the opening (stall-guarded), labels positions with the warm pool, writes
an npz with D4-augment-at-sample-time; `gomoku-train --teacher-data-path … --teacher-weight 0.3`
mixes a teacher CE into every SGD step (BatchNorm frozen on the teacher forward so it doesn't
pollute inference running-stats; weight 0 = byte-identical).

**Verification:** full new-surface test suite (5 files) green at board 9; real-Rapfi pool tests
green at board 15; the FULL existing suite shows no regressions from the `train_step` /
`play_match_pickers` edits (the only failures are pre-existing-on-main `test_defense_teacher_conv`
+ an optional `huggingface_hub` dep missing from the minimal worktree venv). Adversarial 5-reviewer
workflow verified the load-bearing correctness (seat-parity, color accounting, white-split math,
gradient-accum scaling, action-perm alignment, cadence reducer) and surfaced 14 edge findings — the
2 HIGH (BN running-stat pollution from the teacher's 2nd forward; `gather_states` infinite-hang)
and the substantive mediums (RapfiPool close-vs-respawn leak; torn in-place checkpoint read;
malformed-opening 500→400; serve pool-leak-on-startup-failure) all fixed. Live smoke on real Rapfi:
panel reads epoch + white-split correctly; teacher labelled 60 idx-2 positions at 18.6/s.

**Operational constraint discovered:** the live Bruce checkpoints are written by the SWAP2 branch,
whose `ModelConfig` has a `choice_head` field main lacks — so a **main-built daemon cannot load
swap2-trained weights** (panel degrades gracefully to per-ruler error rows). To eval live Bruce,
run the daemon from a checkout whose `model.py` matches the trainer (swap2, or main after both this
build and swap2 merge). Watch `worker_weights.pt` (atomically written) not `latest.pt` (in-place).

**NOT validated:** that distillation actually breaks the plateau — that needs hours of live training
(gate on not competing for the GPU). Bruce-1 (recency-0.5) is the strength-to-beat. Wiki:
`topics/eval-teacher-sensei.md`.

## 2026-06-24 — #77 Rapfi policy-distillation teacher on warm-started Bruce: CATASTROPHIC REGRESSION (teacher@0.3 + high-LR warmstart wrecks the policy)

**Setup.** Warm-started plateaued Bruce (g15 e2659, 128x10 15x15, G15-fixed-openings recipe: 1M packed buffer, recency-0.5, 9 fair openings, 3 workers, gumbel-root m16, value-discount 0.95, sgd-steps 64), turned ON policy-side Rapfi distillation: `--teacher-weight 0.3` over `teacher_bruce_e2659_fair9.npz` (4050 fair-opening Rapfi-labeled one-hot positions). Value head untouched (per #18/#44). wandb `bruce-sensei-77` (`5nzr45ns`). e2659->e3021 (~362 epochs), 0 crashes. Preserved `snapshots/g15_sensei_e3021.pt`. Seed: `teacher/bruce_e2659_warmstart.pt` (e2659 weights+optim+1M buffer, wandb id stripped).

**Verdict (#77 = NO, worse than null).** H2H vs frozen Bruce-1/e2659 (`run_h2h.py`): teacher net **0W-48L-0D at sims=160 AND 0W-48L-0D at sims=100 = 0/96**, both seats (black 0-8, white 0-40), both swap2 roles (opener 0-24, responder 0-24). The teacher@0.3 didn't break the plateau — it destroyed the policy that defined the plateau's floor.

**Mechanism = policy-side trunk corruption, NOT the teacher term.** loss/policy (self-play CE) 1.1->5.0 carries ~all of total (1.85->7.30); policy_net_entropy 1.26->4.57 (toward uniform, log81=4.39); policy_acc 0.685->0.30; policy_kl 0.78->2.97. The teacher CE term stayed small/benign (0.96->1.83, weighted 0.3). Self-reinforcing diffusion: flat policy -> diffuse MCTS targets (target_entropy 0.49->1.60) -> longer games (plies 22->70) -> softer targets. Net-entropy outran target-entropy => the head destabilized on its own. Value degraded only mildly (vl 0.063->0.154). Stable plateau-of-degradation (no NaN, sat in the bad basin 340 epochs — did not blow up, could not recover). NOT a fast-attack collapse (plies ROSE; the opposite tell).

**#44 CONFIRMED — via the policy channel, not value.** Its predicted signature landed: strong warm-started net + sudden teacher target at fixed lr=0.001, no head/trunk freeze => destructive trunk step, pl balloons (1.1->5.0) and stays, instead of holding ~1.25. Same trunk-corruption-via-high-LR failure mode as #44, but through the POLICY-side one-hot Rapfi CE (value untouched). #44's mitigations (1/2-1/4 LR, staged freeze) are the obvious next interventions — UNTESTED in this run.

**#46 unresolved, but the plateau looks FRAGILE.** One naive external-signal injection knocked Bruce well below his own plateau and he stably stayed there (no self-heal) => mild evidence the equilibrium is a delicate basin, not a hardened floor. Curriculum/external-gradient direction (#46) still live, but the injection must be GENTLE or it corrupts the trunk before any benefit accrues.

**Caveats.** weight=0.3 unswept (can't separate "distillation harmful" from "0.3 too hot"); NO matched teacher-OFF control (warm-start/buffer-refresh transient not isolated from teacher harm); LR fixed, no freeze (#44 mitigations untested); H2H used full swap2 negotiation while teacher data was fair-opening-labeled (graded off-distribution); `run_h2h.py` hardcodes CPU. The 0/96 SIGN is certain; magnitude/attribution is what's caveated.

**Process win.** Ran all night crash-free, but the Rapfi *cadence* eval (0% vs Rapfi throughout) was NON-discriminating — Bruce was already 0/16 vs Rapfi. The H2H-vs-frozen-parent + loss decomposition are what revealed the harm. Future overnight teacher runs must gate on H2H-vs-frozen-parent and auto-abort on regression (don't burn 362 epochs on a known-bad basin).

## 2026-06-24 — #86 gentle one-hot teacher retry ALSO regressed (the one-hot SIGNAL is the culprit; soft target untested)

**Setup (#86, follow-up to #77).** The #77 caveats said "the injection must be GENTLE." So two cells were run off the same warm-started Bruce (resume from `/Users/jason/data/swap2/teacher/bruce_e2659_warmstart.pt`, ~e2659), this time at **HALF learning rate** (`lr=5e-4`, the #44 mitigation), fixed-openings, board15, 30-min wall cap each, **with a matched OFF control** (the #77 caveat: no control then):
- **`bruce-sensei-86-gentle-on`** (wandb `liy2dflw`, started 07:33): `teacher_weight=0.1`, ONE-HOT teacher npz (`teacher_bruce_e2659_fair9.npz`).
- **`bruce-sensei-86-gentle-off`** (wandb `5briruqf`, 08:56 then resumed 09:27): `teacher_weight=0.0`, otherwise matched.

**Verdict (#86 = NO; gentleness insufficient).** The gentle ON cell **COLLAPSED** anyway: by ~38 epochs in (~e2697) policy_acc fell to **0.18**, policy_net_entropy rose to **3.75** (vs the OFF control's 1.2; log81=4.39), loss/policy 3.76, selfplay/plies inflating (plies_mean ~60). loss/teacher was present (~0.13) and benign — the damage was again the **policy head flattening toward uniform**, same channel as #77. The matched OFF cell was **ROCK-STABLE**: policy_acc 0.69, net_entropy 1.2, no teacher loss.

**Because the matched OFF control was stable, the ONE-HOT SIGNAL ITSELF is the culprit — not the LR, not the warm-start, not buffer-refresh transient.** Half-LR (5e-4) + weight 0.1 was NOT enough to prevent the collapse. This isolates what #77 could only caveat (it had no control): the harm rides the one-hot Rapfi target, and turning the LR/weight knobs down does not detoxify it. Confirms the #44 failure mode via the policy channel, and sharpens the #77 postmortem (same collapse there at weight 0.3 / full LR).

**NOTE — no Elo/H2H this round.** These were 30-min slices and **no H2H/Elo was ever logged**; the H2H-vs-frozen-Bruce gate never fired. The collapse is read purely from the train-side policy metrics (policy_acc / net_entropy / plies). That is sufficient here ONLY because the matched OFF control isolates the cause — absent the control these train-side reads would not, by themselves, prove teacher harm (cf. #77's caveats).

**The designed fix is coded but NEVER live-validated.** This branch's commit `8d12d95` implements **SOFT-target distillation** — distill Rapfi's per-move WINRATE as a soft policy target via a masked temperature-softmax, instead of a one-hot best move. It has **13 passing unit tests**, but was committed at 09:58, AFTER both runs had ended (~09:57). `soft_policy_weight=0` in every run that actually executed and **no soft npz was ever generated**, so the soft target is **untested** — the untried next step, still subject to the same gate: H2H-vs-frozen-parent, over hours, machine idle.

## 2026-06-25 — #86 soft-target distillation, finally MINED AT SCALE + warm-started ("Bruce Lee one-position", idx-2 only) — infra SUCCESS, science inconclusive; run banked

**This closes the open thread above** ("soft target untested, no soft npz ever generated"). Instead of distilling a handful of fair openings, we built a mining harness and generated Rapfi's SOFT-policy winrate map over the idx-2 neighbourhood at scale, pretrained on it, and warm-started AlphaZero from it — idx-2 ONLY (the over-specialization bet: master one position, not breadth). Full synthesis in **[wiki/topics/rapfi-idx2-distillation-mine.md](wiki/topics/rapfi-idx2-distillation-mine.md)**; reusable capabilities indexed in **[wiki/capabilities.md](wiki/capabilities.md)** (NEW synthesis layer this session).

**Mine.** New `gomoku/rapfimine/` harness (multiprocess flat-file BFS, D4-canonical dedup, crash-robust resume) banked **1,126,597 canonical idx-2 positions** (soft_policy + value, teacher v2 npz) at **~700 moves/s** on the M5 Max (~75% machine). Two fixes surfaced: the long-undiagnosed Rapfi **multiPV mate-crash** (pv-scaled analysis cap — a forced-mate emits 2302 lines past the old 2000 cap) and a **thread-per-line** reader bug (68→17 ms/analyze). Data lives durably OUTSIDE git at `/Users/jason/data/rapfimine/idx2_15x15/` (9.5 GB; absolute paths, no symlink).

**Pretrain (the soft target, finally exercised).** `rapfimine.pretrain` = supervised distillation of the masked temperature-softmax soft policy + `2·best_wr−1` value into a STANDARD checkpoint (`build_model`/`save_checkpoint`, no reinvention). Banked the epoch-3 seed `checkpoints/idx2_pretrain.pt` (policy_ce 2.04, vmse 0.097 — vs ~5.4 uniform, so it captured Rapfi's idx-2 policy). Finding: pretrain is **GPU-bound** (~437 s/epoch, batch 1024); a per-step `float(loss)` was forcing an MPS→CPU sync every step (host pinned ~7%) — fixed to sync once/epoch.

**Warm-start AZ.** `run_sweep --cell G15-idx2-warmstart --resume checkpoints/idx2_pretrain.pt` with `GOMOKU_DROP_OPENERS=0,1,3,4,5,6,7,8` (idx-2 only; D4 recovered by the trainer's augment). Byte-identical to Bruce's `G15-fixed-openings` cell except run-dir + run-name. wandb `idx2-warmstart-86`. Resumed at epoch 4 (pretrained weights confirmed loaded), trained to **epoch 250** (pl 4.18 peak → 1.27, vmse 0.10), banked `checkpoints/idx2_warmstart_final.pt`.

**Outcome (science = inconclusive, by choice).** At epoch 250 the warm-started net **does NOT yet beat strong Rapfi @idx-2** — still 0/48 vs `timeout=1000ms`, the same wall the seed hit. Beating it is a multi-day climb (Bruce's black-42%/white-0% bar took ~3700 epochs); **not pursued** — Jason banked the run for its infrastructure value. The net DID climb the low end: it crushes random/heuristic/lookahead-d2 at 100% and beats `rapfi@25ms`, losing at `50ms+`.

**Eval-gradient finding (reusable).** Max-strength Rapfi is a wall (0 for hours), so `fast_eval.py` measures progress vs a graded-Rapfi ladder — **~20 s/pass** (batched net MCTS across all games + parallel `RapfiPool.label_states`), vs minutes serial. Two lessons: (1) **think-time is the strength dial, NOT max_node** — even `max_node=2,000,000` loses 100% to the net while `timeout=1000ms` wins 0/48; (2) the live band is LOW timeouts (net's transition at **rapfi 25↔50 ms** @ ep250), which are also fast. `sims=32` gives the same transition as 160.

**Banked & stopped (this is NOT a regression entry — the loop is healthy).** Training stopped cleanly by choice at ep250; nets in `checkpoints/`, data in `~/data/rapfimine/`, tooling + wiki committed on `feat/gentle-rapfi-teacher`. Also fixed a `.gitignore` bug (an inline comment silently broke the `mined/` rule → 9.5 GB of artifacts were not being ignored). The durable win: the mining + fast-eval + warm-start tooling, now a first-class capability.

## 2026-06-25 — On-book DAgger for idx-2: loop BUILT, a critical bug FOUND+FIXED (history-less train/inference mismatch), no strength gain yet ⭐

**The idea (Jason).** From ep250, run **on-book DAgger** (Ross/Gordon/Bagnell 2011): roll out the current net, have Rapfi label the states the net actually visits, AGGREGATE (never forget), and re-fit by **pure supervised imitation** — explicitly NOT the AZ teacher-mix that caused the #77 collapse. Stay on-book (no improvising), idx-2 only, **rollout vs Rapfi both seats, "partitioned and covered."** Hard constraint: the aggregator must run **≤ 2× slower than its constituent parts** (self-play gen + Rapfi labeling) — "don't waste a day testing something too slow." White is NOT specially targeted (the white weakness was likely an empty-board first-player artifact; from a Rapfi-seeded fixed opening it should wash out via balanced-seat coverage).

**Built (`gomoku/rapfimine/dagger.py`, + `tests/test_dagger.py`, full suite green).** A reuse-not-reinvent loop: `rollout_once` (concurrent student-vs-Rapfi, both seats, batched MCTS on MPS + `RapfiPool` opponent on CPU — the `fast_eval` pattern) → canonical dedup → soft/hard label → `store.ShardWriter` (byte-identical to the mine schema) → warm-continue supervised train (the `pretrain` pipeline) → **net-vs-frozen-parent** gate (NOT vs Rapfi — that cadence was non-discriminating in #77). Plus a `loop` subcommand = the flywheel (round i+1 rolls from the BEST net so far, gates vs the FROZEN ep250, stateless reducer over on-disk result JSONs).

**Perf gate PASSED (the 2× contract).** Microbench on the M5 Max: (A) student MCTS sims=32 batched ≈ **220 moves/s** (MPS); (B) `RapfiPool` label ≈ **872/s** @50ms/24w (the #86 reader-thread fix killed the old GIL ceiling); (C) the combined student-vs-Rapfi rollout ≈ **215 plies/s ≈ A** — the disjoint MPS/CPU compute genuinely overlaps, **no meaningful coupling penalty**, soft-labelling distinct states post-hoc adds <1 s on an ~8 s rollout. Coverage is balanced (both seats ~50/50 of stored states) and `rollout_temp_until_ply` sampling keeps novelty high (~800 fresh canonical/roll, no decay) — the funnelling I feared didn't bite.

**THE WALL — every first round REGRESSED the net to 0/48 vs the parent.** Three round-0 variants, all → 0/48 (gate), all → **rapfi@25ms = 0.00** on the deployment-path gradient (the parent reads **1.00** there). Crucially the *training loss DECREASED* each time (policy_ce fell) while *play collapsed* — the tell of a **train/inference mismatch, not a strength/teacher problem**. The diagnosis chain:
1. **First suspect (wrong as sole cause): weak teacher.** The SOFT label uses `analyze` (node-bounded, max_node 5000). This very notebook already records: *node-bounded Rapfi — even 2M nodes — loses 100% to the net; think-TIME is the strength dial.* So a soft label is a **sub-student teacher** → distilling it should drag the net down. Real, but **not the whole story**: a round with STRONG time-bounded HARD one-hot labels (`--label-timeout-ms 300`, on-book DAgger's actual classifier target) ALSO went 0/48.
2. **Root cause (confirmed): the aggregate stored HISTORY-LESS planes.** The mine drops history (`drop_history`) — sound there because AZ self-play afterward repopulates the recency channels. **Pure-supervised DAgger has no such repair step.** Training the net on zero-recency planes and then *playing it with real recency planes* is an out-of-distribution input → garbage output → clean 0/48 regardless of teacher. Fix = **store the real history** (one line: `canonical_state(s)` not `canonical_state(drop_history(s))`; the D4 canonical transform already carries history under the same symmetry; the label stays board-only since Rapfi is history-blind). Round 0c (history fix + value-weight 0) holds at **rapfi@25ms = 1.00 = parent** — the catastrophic regression is GONE.
3. **Second issue found: HARD one-hot labels corrupt the VALUE target.** `PretrainData` sets value = `2·best_winrate−1`; a one-hot `{move:1.0}` makes best_winrate≡1 → value target ≡ +1 everywhere → the value head collapses to constant "+1" (value_mse→0 trivially). Mitigation used: `--value-weight 0` (freeze the parent's already-good value head). The SOFT map's one virtue was a genuine value; a clean design decouples **strong timed pick for POLICY + soft winrate for VALUE** (or just freeze value).

**State / verdict.** The DAgger machinery is correct, fast, and tested; the catastrophic-regression BUG is fixed; round 0c sits at ≈parent (gate 0.354 = black 0.458 / white 0.25, but only 6 gradient steps — a confirmation, not a real training run). **No strength GAIN demonstrated yet** — that needs a proper round now that the blocker is cleared. Recommended next round: history fix (done) + **HARD timed-pick policy labels** + **value frozen (or soft-sourced)** + bigger `--target-new` (≥30k) + enough steps + gentle LR, gated vs frozen ep250; then iterate the `loop`. Artifacts: `mined/dagger_*_r0.log`, `mined/dagger_*_r0_result.json`; throwaway round checkpoints in `checkpoints/dagger_r0.pt` (overwritten across variants — not banked). Branch `feat/gentle-rapfi-teacher`. This is a "stopped cleanly at a well-characterized wall" entry, not a success — the loop is ready; the science question (does idx-2 DAgger beat the self-play ceiling) is still open.

### 2026-06-25 (later) — Proper DAgger rounds RUN: a clean NEGATIVE. Imitation sharpens the net but does NOT move the Rapfi ceiling (search-depth wall) ⭐

Ran the real experiment with **every** fix in (real-history planes + Monte-Carlo value + strong 150ms timed-pick one-hot policy labels, dagger-only data, gentle lr=1e-4): one 20k-state round (`dagger_v2_r0`) then a 2-round `loop` (~38k cumulative aggregate, best-net rollout, frozen-ep250 gate). **Result — reproducible and clear:**
- **Gate vs frozen parent (net-vs-net):** v2 **0.479** (black 0.79 / white 0.17) · loop r0 **0.458** (b0.79/w0.125) · loop r1 **0.438** (b0.83/w0.04). The net gets **markedly stronger as black and weaker as white** with each round — the asymmetry *intensifies* (b0.79→0.83, w0.17→0.04) while the aggregate sits ≈parity (the seat gains/losses cancel). More rounds did NOT compound toward beating the parent.
- **The decisive check — think-time gradient vs Rapfi (the real ceiling):** parent and the most-trained dagger net are **byte-for-byte identical** — both beat rapfi@25ms (1.00) and lose at 50ms+ (0.00). **The Rapfi wall never moved**, across the single round AND ~38k of cumulative DAgger training.

**Interpretation (the lesson).** Policy-imitation of Rapfi **sharpens the net's prior** (it beats its own past self, especially as black) but **cannot cross the rapfi@50ms wall, because that wall is a SEARCH-depth gap, not a prior gap** — Rapfi@50ms+ wins by *searching deeper at move time*, and distilling its *move* (a one-hot at sims=32) doesn't give the student that search. This is exactly what the project's own calibration predicted (think-time, not node budget, is Rapfi's strength dial). Corollary, from the seat asymmetry: **idx-2 looks black-favored** — imitation + MC outcomes double down on black's winning attack (sharper) while the white signal is mostly "you lose" (Rapfi's best white move still loses), so white play degrades. If idx-2 is a first-player win, "crush Rapfi from idx-2" is achievable only *as black*; white may be structurally lost (consistent with Jason's empty-board-artifact hypothesis being incomplete — the asymmetry is the *position's*, not a random-start's).

**Verdict: DAgger is the wrong lever for THIS wall.** It's correct, fast, and bug-free, and it *does* shape the policy — but it can't out-search Rapfi. **Pivot recommendation → [idea-pile](topics/idea-pile.md) #1 (out-*search*-yourself-then-distill):** give the net a big test-time search budget at idx-2 (high sims / in-tree VCF), find where deep-net-search beats Rapfi, distill *those* policies — distill SEARCH, not the prior. And #2 (solve idx-2): if it's a black win, a threat-space tablebase is the perfect black teacher. Artifacts: `mined/dagger_loop.log`, `mined/dagger_r{0,1}_result.json`, `checkpoints/dagger_r{0,1}.pt` (not banked). DAgger code is committed/tested on `feat/gentle-rapfi-teacher` and ready to reuse (swap the Rapfi label for a deep-search label = idea #1).

### 2026-06-25 (later still) — idea #1's premise FAILS: the net is EVAL-capped, not search-capped. + perf numbers (card is compute-bound; GPU VCF-detect is ~free)

Before building idea #1 (out-search-then-distill), tested its premise directly: does the net's DEEP search find what its shallow search misses? **Net-MCTS at sims 32 / 200 / 800 / 1600 vs the Rapfi gradient is IDENTICAL — all read rapfi@25ms=1.00, 50ms+=0.00.** 50× more search moves the wall ZERO. **The net's EVALUATION (policy+value) is the ceiling, not its search depth** — so *vanilla* #1 ("distill the net's own deep search") is dead: searching harder finds the same thing, nothing better to distill. And 1600 sims is a LOT of search — if the gap were *tactical* (forced wins/losses), that much MCTS would usually stumble into them and cross a rung; it didn't, which leans toward a **positional evaluation** gap (Rapfi-NNUE just judges positions better) over a tactical one. (**Confirmed:** a net+root-VCF-overlay gradient is IDENTICAL to net-only — rapfi@50/100ms = 0.00 either way — and the overlay logged **0 forced-win hits** across ~180 net positions. The net is NOT leaving findable forced wins on the table; the gap is **positional**, not attacker-tactical. Caveat: this tested attacker-VCF only, not defense — the net could still be walking into Rapfi's threats, which a *defensive* VCF/eval would catch.)

**The surviving form of #1:** search with a BETTER EVALUATOR (VCF tactics and/or Rapfi-NNUE eval at the leaves), then distill THAT — not the net's own eval-capped search.

**Perf numbers (M5 Max, 15×15; benches in `scratchpad/bench_batch.py`, `bench_gpu_vcf.py`):**
- **The card is COMPUTE-bound for the net, not memory-bound.** Net-eval throughput is FLAT at **~11,500 boards/s from batch 64 → 65,536** (no OOM at 65k). Batching more boards buys ZERO throughput — the GPU saturates at batch≈64. The hard currency is **net-evals/sec, not board count**; wave-batching cuts wall-clock latency, not total compute.
- **GPU VCF threat-detection is ~free** (Jason's "put VCF on the GPU, measure the handoff" brainstorm, measured): directional length-5 line-convs run **12.7M boards/s resident** (10.0M with the CPU↔GPU handoff — real but negligible), vs CPU `has_four_threat` **1,057/s** and CPU `solve_vcf` full-tree **53/s**. So *detection* isn't the cost — the **sequential forcing-TREE recursion** is. The build that follows: a **batched-frontier GPU-VCF** (thousands of VCF searches in lockstep, GPU detection kernel per ply) for a plausible ~100× full-solve throughput → VCF tactical truth becomes a real-time teacher/guard-rail (idea-pile #9).

**Net strategic read:** three things tried/measured — imitation (DAgger), more search (deep MCTS), attacker-tactics (VCF overlay) — all fail to cross the Rapfi wall; the wall is a **positional evaluation** ceiling (the net plays positions worse than rapfi@50ms; it isn't missing forced wins). So the front-runner lever is now **#7 (distill Rapfi-NNUE's positional evaluation)** — give the net a better static eval, not more search/tactics. **#9 (GPU-batched VCF)** is repurposed toward **DEFENSE** (don't walk into Rapfi's threats) + as a real-time guard-rail (the perf is proven: 12.7M detect/s). **#2 (solve idx-2)** remains the ground-truth option. Nothing further built this session pending Jason's read on the eval-lever fork.

### 2026-06-25 (perf spike) — GPU batch-VCF CRACKED: full forced-win solves at ~2,500× CPU, 100% correct vs `solve_vcf` ⭐

Built and measured the batched-frontier GPU-VCF that the prior entry and idea-pile #9 only projected (~100×). **It blew past the projection — ~2,500×, not ~100× — at 100% correctness.** Prototype: `scripts/gpu_vcf_prototype.py` (`solve_vcf_batch(boards (B,2,15,15) bool) -> (won, hit_cap)`, MPS). Run: `GOMOKU_BOARD_SIZE=15 uv run python scripts/gpu_vcf_prototype.py`.

**THE INSIGHT (why it's exact AND batchable).** Plain VCF is a pure **OR / reachability** search, *not* a real AND/OR tree: the defender is ALWAYS forced to the *unique* completion square of the attacker's four, so every defender node has exactly one child. Hence "is there a forced win?" == "from the root, can the attacker reach (within max_depth) a node with an immediate five or a sound double-four?" — run as a **breadth-first frontier**: every node in the current frontier is an attacker-to-move board at the *same depth*, so the whole frontier advances **in lockstep** and ALL threat detection batches across the B searches at once (directional length-5 shift-products over `(F,15,15)` bool planes — the proven-free conv primitive, applied frontier-wide). One host sync per BFS level (the child-gather `nonzero`); no `.item()` in the inner loop. Four-detection is provably identical to `vcf._five_completions` (a four-move m+completion c ⟺ a 5-window of 3 own + {m,c} empty + 0 opp; each (dir, signed-offset) → a unique completion cell, so #firing pairs == CPU `len(comps)` and the single-four block is `m+δ·d`); the forcing test == `_has_immediate_five(defender)`.

**CORRECTNESS — 100% agreement vs CPU `solve_vcf` across 5,300 positions** (500 spec + 2,400 random midgame + 2,400 dense), including **121 deep mates out to mate-distance 15** — every verdict matched. CPU `hit_cap`=0; the single GPU frontier-truncation case still agreed.

**THROUGHPUT (M5 Max MPS, random midgame mix) — full solves/sec:** scales to **~130–146k solves/s at B≈16k–65k** vs CPU's **53/s** = **~2,500× (peak 2,749× @ B=65,536, depth 8; ~2,480× @ depth 16)**. Saturates near B≈16k; small B is kernel-launch-bound (~8–10k/s @256). depth barely matters (random wins are shallow). Throughput is position-mix-dependent: a batch of deep-tree forced-wins branches the frontier wider and costs more (capped `max_frontier=4M`, ~1.8 GB, flagged `hit_cap`).

**What it unlocks.** VCF tactical TRUTH is now real-time at training scale — the throughput blocker on idea-pile #9 (ground-truth forced-win/certain-death teacher + anti-guard-rail move ranking) is GONE. Combined with the prior entry's strategic read, the highest-value use is **DEFENSE**: batch-VCF every state the net reaches to detect "the opponent has a forced four-win here" (certain-death, value −1, and the saving move via `vcf_refutations`) — directly attacking the net's positional/defensive wall in real time. **Open items:** returns the *verdict*, not `winning_move`/`mate_distance` yet (block-index machinery already present — easy add); **no child-board dedup yet** (per-level hash dedup is the obvious next win on tactical batches); plain VCF only, **not VCT** (the continuous-threes solver). Budget accounting differs (CPU DFS calls vs BFS frontier nodes) so the only possible disagreements are cap-boundary cases — and across ~5,900 total positions **exactly ONE** appeared (a later 600-dense-board run): a dense board where the CPU **hit its 200k-node DFS cap and returned an *unproven* `False`** (`hit_cap=True`) while the GPU completed and **proved a genuine forced win**. The divergence is the GPU being strictly *more complete* than the cap-limited CPU, **not a false positive** — clean (non-capped) agreement stays **100%** (598/598 in that run, incl. all 21 deep mates). Committed on `feat/gentle-rapfi-teacher` (not merged/pushed — left for Jason).

## 2026-06-26 — VCT-GPU REBUILD: a working batched GPU VCT solver + the on-device megakernel path proven ⭐

The v0 GPU-VCT spike hit the "this is not GPU-shaped" wall (correct but CPU-bound: ~84% host, the AND/OR orchestration fights the GPU — see `wiki/topics/gpu-vct-feasibility.md` §1–§6). This is a **from-scratch rebuild** pursuing Jason's three sharpenings — **(A) compiled line-threat grammar + (B) intersection (bitmask) defense generation + (C) work-first continuation stealing** — toward a persistent-epoch Metal megakernel. Hermetic in `scripts/vct_metal/` (nothing else imports it). **Vehicle: MLX** `mx.fast.metal_kernel` — runtime-compiled MSL, no Xcode needed; bitwise/popcount + device atomics confirmed on the M5 Max. Discipline: every layer validated against the CPU oracle `gomoku.vcf` before building the next; tests on **real Rapfi positions** (`~/data/games_raphi/`, a loader built this session) under a 2-min `timeout` cap. Full §7 lives in the wiki page.

**What landed (all oracle-validated):**
- `detect_ref` (numpy spec) + `detect_metal` (**Metal kernel**) — OR-node detection (fives, four-structure, candidates, tempo). Matches `vcf` cell-for-cell over 900 boards; ~1M boards/s on-GPU. Subtlety: `four_structure` counts fours *created by the move* (m in the five-window) = `vcf._completions_through` exactly in the no-immediate-five regime the OR-node runs in.
- `threes_ref` + `threes_metal` (**Metal kernel**) — forcing-threes + **(B) bitmask defense**: reply-set = OR of threats' `{f}∪comps` masks, **fork = a disjoint mask pair**. The v0 70% (host tempo-guard + open-four assembly) reduced to set-algebra. Matches `vcf` over ~1.8k threes.
- `search_ref` — the AND/OR solver *composed from the primitives*; verdict matches `vcf.solve_vct` on clean cases. Slow (B=1 recursion, 2.2 s/board) — the B=1 shape is exactly what batching fixes.
- **`wavefront` — THE WORKING GPU VCT SOLVER.** Host orchestration + all-GPU per-node kernels (detect + threes + a swapped-detect tempo pass), single reverse-order AND/OR backup; detection amortized over the whole frontier per wave. On real Rapfi positions verdict matches `vcf` with **0 false-positive and 0 false-negative** clean disagreements; ~12 ms/board batched at B=50. Sound: four-soundness conservative (never a false win), three-tempo exact via batched materialisation.
- **`mega_vcf` — fully on-device VCF megakernel** (one thread/position, iterative DFS with make/unmake on a thread-local board, NO host orchestration per node). Validated sound vs `vcf.solve_vcf`: clean-agree 40/40, 0 FP/FN. Throughput amortizes with batch (71→**12 ms/board** at B=2048) but is **tail-bound** — B=512 and B=2048 take the same ~24.6 s wall: one deep position serializes the batch (no work-stealing; per-node detection O(N²)).
- **`mega_vct` — fully on-device VCT megakernel** (the (C) vision: OR-frames = fours + forcing threes, AND-frames = defender replies, all in-kernel). Compiles, runs, returns plausible verdicts — **VCT fully on the GPU** — but **impractically slow naive** (~11 s/board): per-node detection recomputed O(N³)-ish + no work-stealing → severely tail-bound.

**Throughput measured (perfsweep subagent, real positions, M5 Max).** Correctness gate re-run B=50 × 3 seeds (150 positions): **0 FP / 0 FN** every seed, `found_extra=0` (no wavefront WIN ever needed the vcf cap-lift recheck). GPU throughput (`max_depth=8`, `max_nodes=20000`): B=128 → 5.47 ms/board / 183 solves/s; B=512 → 0.53 / 1893; B=2048 → 0.28 / 3528 (peak RSS 2.0 GB). **Two caveats that flip the naive reading:** (1) **host-orchestration-bound at every scale** — only **24–40 %** of wall time is inside the MLX kernels, the other 60–76 % is the host Python per-node expand+backup loop (incl. host↔GPU transfer); (2) **`max_nodes` is a GLOBAL pool across the batch, not per-board**, so the falling ms/board is a *node-cap artifact* (each board gets 20000/B nodes; cap% rises 34→42→47 %), **not** GPU amortization — for real corpus labeling, **scale `max_nodes` with B**. CPU baseline (single-thread `vcf.solve_vct`): **0.64 solves/s aggregate**, sharply bimodal (median 30 ms ≈ 33 solves/s typical; ~14 % of positions hit a ~90 s tail). **Speedup 5.5× (typical position) → 286× (B=128 vs aggregate CPU)** — clears the 20–50× "pay for itself" bar *via the aggregate measure, because the GPU dodges the CPU's ~90 s-per-hard-position tail*, not via raw per-board speed. **The punchline: the wavefront being host-bound (60–76 % of wall outside the kernels) is the empirical case FOR the megakernel** — the (C) direction is where the measured time actually is, and the megakernel's own bottlenecks (incremental detection + work-stealing) attack the other end. Both ends measured, not guessed.

**Verdict.** VCT is **on the GPU**. The **`wavefront` solver is the usable deliverable** (correct on real positions; throughput host-orchestration-bound, clears the 20–50× bar by aggregate measure). The **megakernel path is proven** — the fully-on-device search machine is correct (VCF: sound 40/40) and structurally complete for VCT — and the naive megakernel's two bottlenecks **pin the (C) levers precisely**: (1) **incremental/bitboard detection** (patch the ≤4 lines a move touches instead of rescanning O(N²–N³)/node), and (2) **work-stealing** to kill the single-deep-position tail. (A) and (B) are *proven correct*; (C)'s orchestration is the remaining adversary, exactly as v0 predicted — but now with a working solver in hand and the levers measured, not guessed. All committed + pushed on `feat/gentle-rapfi-teacher`.

## 2026-06-26 — VCT-GPU OPTIMIZED: bitboard megakernel, ~195× per-board, ~900× CPU throughput ⭐

Goal: make GPU VCT *as fast as I can*. Lever (1) from the rebuild (incremental/bitboard detection) was the dominant cost — the naive megakernel rescanned the board O(N²)/node and the per-move soundness check was O(N²)/move → **O(N⁴)/node**. Rebuilt detection on **bitboards** (`bb_ref.py` golden Python-bigint ref + `bb.py` MSL `ulong[4]` helpers: `shr256`/`shl256`/`has_five`/`completion_mask`; validated bit-for-bit, 1200 random + 600 real boards, 0 mismatch). Five-completion sets and forcing-move generation become shift-AND set-algebra.

- `mega_vcf_bb.py` (VCF): forcing-move gen by hole-pair set-algebra; soundness + double-four via `completion_mask`. **13–14× faster** than `mega_vcf` (84→6.4 ms/bd @B=512; 21→**1.48** @B=2048; wall 42.5→3.0 s). Validated **0 FP / 0 FN vs `gomoku.vcf.solve_vcf` over 360 real positions**.
- `mega_vct_bb.py` (full AND/OR VCT): bitboard detection + four more levers — fours generated once by set-algebra; threes restricted to **Chebyshev-2 of OWN** (radius-2-per-side argument; ~halves candidates, **1.65×**); follow-up `f` restricted to `vcf._collinear_empties(m)` (**correctness fix** — without it, far pre-existing fours over-generate spurious threes, which I caught via a verdict shift between versions); monotone defender-five/tempo fast-paths (computed once/node, skipped when globally absent); `rmask` aliased into `fmask`. Per-board algorithmic speedup at **equal config** (B=24, mn=600): cell-scan `mega_vct` **38.4 s → 4.0 s = ~10×, 0 verdict disagreements** (bitboard detection ≈10–14× as in VCF, candidate-own ×1.65 on top). **Validated 0 FP / 0 FN vs `vcf.solve_vct` over 320 real positions** (258 clean agreements, 8 seeds). The bigger practical win is *batchability*: the cell-scan kernel's tail can't finish even B=64 @ mn=1500 in 2 min; the bitboard kernel batches to B≈16k (below).

**The throughput finding.** Wall is **flat ~16 s for B=128…2048** at mn=1500 — completely **tail-bound by the single deepest board**. For batch labeling that's a *feature*: throughput ∝ B at constant wall. Measured solves/s (mn=1500, fully on-device, RSS 0.3 GB): 8192→526, 16384→**891**, 32768→**1 020** (saturating ≈ GPU concurrency). CPU `vcf.solve_vct` = 0.64 solves/s → **~1 600× aggregate**, and (unlike the wavefront's 24–40 % GPU util) there is **no host bottleneck** — fully on-device. A late **shift-precompute** (reuse each direction's five `shr256` of own/empty across the hole loops of `gen_forcing`/`completion_mask` — verdict-preserving, `test_*_bb` still pass) lifted the saturated ceiling ~1.5× (583→891 @ B=16384). NB: `load_position_stack` samples a *live-growing* corpus (`~/data/games_raphi/` is being collected), so raw win/cap counts drift between invocations — verdicts are deterministic within a process and bb always matches the oracle on the loaded set.

**What is and isn't the lever (honest).** Work-stealing (lever 2): board-level is already done by the GPU scheduler; the real tail is *one* deep board (one thread), so subtree-spill would cut single-board latency but **not** labeling throughput (already maximized by batching) — so it is *not* the lever for this use case. **Negative result:** generating threes by single-line open-three patterns is **incomplete** (misses *four-four-at-`f`* threats — the follow-up makes fours in two directions, only one through `m`); localizing detection to `m`'s lines hits the same wall, so `gen_threes` stays whole-board. The throughput knobs for labeling are **B** (batch) and **`max_nodes`** (depth/quality vs wall). **Negative result #2 (word width, tried + reverted):** a 256-bit detector can be `ulong[4]` (64-bit) or `uint[8]` (32-bit); a micro-bench of *pure* shift+AND favoured `uint[8]` 1.7× (Apple GPU 64-bit ALU is throughput-reduced), so I rewrote the whole kernel to `uint[8]` and re-validated (0 disagreements) — but it ran **~20 % slower** (VCT 693 vs 854 solves/s @ B=16384; VCF 1.55 vs 1.40 ms/bd) because the real kernels are bookkeeping-heavy (`popcount`/`lowbit`/`setbit`/`cpy`/`and`/frames) and that all doubles in word count, outweighing the detection-only gain. `ulong[4]` kept. Lesson: micro-benchmark the hot path *in situ*. Committed + pushed on `feat/gentle-rapfi-teacher`.

## [2026-06-26] is-VCT recognition is learnable on unseen games — but attention loses to a CNN

**What.** First learnability probe on the VCT puzzle labels: can a net classify "side-to-move
has a forced VCT?" from the raw 15×15 board, generalizing to **unseen games**? Full synthesis:
`wiki/topics/vct-recognition-learnability.md`. Code: `scripts/threat_shapes/gen_isvct_dataset.py`,
`scripts/threat_shapes/train_isvct_attn.py`. Artifacts: `~/data/puzzle_miner/isvct_exp/`.

**Setup.** Labels reused from the forward puzzle miner (`~/data/puzzle_miner/`), NO re-solve —
POSITIVE = `win&~cap`; NEGATIVE = manifest ply **absent** from `puzzles.jsonl.gz` (proven
no-VCT); `cap` excluded. Split **by shard** (md5%10): 400 manifest shards → **367 train / 33
test, overlap 0** (+49-shard val from train for early-stop). Train 1,167,002 (17.3% pos), test
101,745 (14.2% pos), balanced training (60k). Negative boards CPU-replayed in the exact
side-to-move frame, **0 frame mismatches**. Light enough (CPU replay + MPS train, no GPU solve)
to run without competing with the live `collect_rapfi` producer.

**Held-out result (AUROC, the fair metric):** majority 0.500 · logreg-on-counts 0.946 · **CNN
(168k) 0.971** · **attention (339k) 0.924**. Attention val→test 0.933→0.924 (no leakage). Wall:
gen 20s, train+eval(×4) 356s on MPS.

**Read.** (1) Feasibility = **yes** — the win-condition is perceivable and generalizes across
shards. (2) Attention is the **laggard** — beaten by a CNN with *half* the params and by linear
logreg-on-counts. VCT structure is local + translation-equivariant ⇒ conv bias fits; the signal
is count-dominated. (3) Strategic: recognition was always the **exact oracle's** job (cheap),
so this *clarifies* rather than dents the plan — **attention's real audition is the seeker**
(steering toward VCT-reachable regions), not recognition. Caveats: small/untuned (60k, ≤12 ep,
attention still inching up); "no-VCT" = no VCT within the miner's 500-node budget; trivial
early-game negatives inflate natural-accuracy (use AUROC/balanced).

**Also (same session):** the megakernel now emits a **passive GPU root-move**
(`solve_vct_mega_bb(return_move=True)`) — resolves the `vct-backward-mining.md` §5 move-extraction
gap; 2.38M forward puzzles move-labeled (`solutions.jsonl.gz`), 400/400 independently verified.
All on `feat/gentle-rapfi-teacher` (not yet merged).

## [2026-06-26] Seeker steering is learnable on unseen games (seek-VCT thesis, Phase A) — CNN > attention again

**What.** The **steering** half of the seek-VCT thesis (the recognizer half named the seeker as
attention's real audition). One question: can a net behaviorally-clone the **quiet-phase (pre-onset)
moves of the side that reaches the first forced VCT**, and generalize to **unseen games**? Full
synthesis: `wiki/topics/seeker-steering-learnability.md`. Code:
`scripts/threat_shapes/gen_seeker_dataset.py`, `scripts/threat_shapes/train_seeker.py`. Artifacts:
`~/data/puzzle_miner/seeker_exp/`.

**Setup.** Reuse the miner verdicts, NO re-solve. `onset(game)` = first ply with `win&~cap`; the
mover there = the **seeker S** (kept whether S converts or misses — "you reached a winnable position"
is the target). STEERING EXAMPLE = every pre-onset ply `p < onset` with `p%2==onset%2` (S to move);
input = side-to-move-relative board (`board[0]`=S), target = the move S actually played. Boards
CPU-replayed in the exact miner frame (`all_boards`), every present puzzle key cross-checked → **0
frame mismatches over 400 shards**. Split **by shard** (md5%10, the recognizer's rule for
comparability): **367 train / 33 test, overlap 0** (+49-shard val for early-stop). **500,747**
examples from **38,927 onset games** (1,073 no-onset); 459,415 train (200k used) / 41,332 test; mean
208 legal cells/board. Per-cell policy with legal-move masking; light (CPU gen + MPS train, no GPU
solve) → ran `nice`d without competing with the live `collect_rapfi` fleet.

**Held-out result (top-k legal-move-match = is the seeker's *actual* move in the policy's top-k):**

| model | params | top-1 | top-3 | top-5 | CE |
|---|---|---|---|---|---|
| uniform (random legal) | — | 0.005 | 0.014 | 0.023 | 5.37 |
| adjacency-to-stones | — | 0.025 | 0.072 | 0.121 | 4.78 |
| **CNN** | **224k** | **0.386** | **0.597** | **0.696** | **2.26** |
| attention | 339k | 0.263 | 0.457 | 0.569 | 2.76 |

Wall: gen 15 s (CPU), train+eval 1,541 s on MPS (CNN early-stop ep8 ~13 s/ep; attention full 20 ep
~71 s/ep).

**Read.** (1) Feasibility = **yes** — the CNN matches the *exact* strong-engine steering move ~39%
(top-1) / ~70% (top-5) on unseen games, **~15×** the adjacency prior at top-1; CE confirms genuine
calibration over the move distribution. The steering signal is learnable and generalizes — the cheap
green light the seek-VCT plan needed. (2) **CNN > attention again** (top-1 0.386 vs 0.263, fewer
params) — next steering move is *local*, fits the conv prior. (3) **Two honest limits:** attention
was **still climbing at the epoch cap** (val 0.066→0.253 monotone, undertrained not capped), AND
next-move BC is local so it does **NOT** settle attention's *global-receptive-field* bet for
*sequential* seeking; and top-1 match is a **weak proxy** (≠ strong play; conflates seeking with
general engine strength). The architecture verdict for seeking is deferred to the decisive test.

**Next (gated with Jason — the GPU-spending real tests).** **Phase B:** replace the imitation target
with an oracle-*constructed* one — score each pre-onset candidate by **VCT-reachability gain** (does
a forced win appear within k plies after move + best reply?); principled but costs k-step batched
lookahead per candidate. **Phase C (decisive):** a **hybrid player** — oracle every ply for attack +
defense, exact solver finishes any VCT, net steers only in the tactically-quiet region — played vs a
**fixed baseline** (heuristic/lookahead, not sibling H2H). Phase C is where attention's global bet is
actually adjudicated. On `feat/gentle-rapfi-teacher` (not merged).

## [2026-06-26] The pre-onset band is a KNIFE-EDGE — seek-VCT thesis update + non-VCF gold

**What.** Two search-free/GPU-only ways to mine VCT-reachability from the 500k Rapfi-v-Rapfi games,
toward the seeker. Full synthesis: `wiki/topics/vct-reachability-mining.md`. Code:
`scripts/threat_shapes/vct_fan.py` (consolidated probe). All solving on the Metal **GPU** kernels
(`mega_vct_bb`, `mega_vcf_bb`) — zero contention with the CPU `collect_rapfi` fleet. **Both seats are
strong Rapfi**; every "losing move" is a counterfactual we inject.

**The method (off-path fan).** Ride each game; at known-non-VCT pre-onset nodes, fan every alternative
move the side did NOT play and solve VCT on each. **Framing (load-bearing, code-verified):** a VCT
belongs to the side-to-move, so after S plays alt `m` it's the opponent's turn → a fanned VCT is the
**opponent's** forced win = `m` is a forced-LOSING move for S. The fan is a **defense/blunder + VCT
miner, NEVER an offense detector** (S would need its own turn = the expensive ∀-reply search). Integrity:
**0.000%** of fanned nodes are themselves VCT, 0 parity violations.

**Finding 1 — the knife-edge (the headline, a thesis update).** Fraction of a side's alternatives that
lose by force, by who's-to-move × distance-to-onset: opponent-to-move **98.3% (d1) / 92.7 / 84.6**;
VCT-holder-to-move **89.4 (d2) / 52.7 / 45.7**. Even **6 plies before the VCT, ~half of moves lose**;
the *winner* at onset−2 loses 89% if it deviates. **Both players walk a tightrope; sharpness ramps
BEFORE the onset.** ⇒ the seek-VCT split's "pre-onset = the net's forgiving region" is **wrong** — that
band is not approximation-tolerant; the net's safe domain is further back than onset−6, and the
solver/lookahead must own the whole sharp ramp (the "oracle every ply" hybrid already does — now with
the *why*).

**Finding 2 — 96% of the wins are trivial; the gold is the winner's combinations.** VCF kernel on the
406,202 fanned VCT-wins (81.1% of all fanned): **VCF 96.1%** (four-driven, trivial — the extreme is
"you didn't block my five"), **non-VCF VCT 3.5%** (14,380; need a *three* = combinational molecules),
VCF-cap 0.3%. The non-VCF gold splits by parity — concentrated on the **WINNER's** wins (defender
perturbed): non-VCF rate 1.9/6.0/6.2% on the winner's rows vs 0.0/0.7/1.2% on the opponent's.
**Combinations belong to the side with the initiative.** Harvest plan: **perturb the *defender*** at
pre-onset opponent-to-move nodes → ~100k+ non-VCF VCT boards (a few free-GPU hours) = non-trivial
offense termini for the distance field + hard defense lessons (the white-defense wound).

**Also banked — the free distance-to-VCT field** (from the existing per-ply verdicts, no re-solve):
terminal-VCT 99%, multi-window (lose-then-refind) 11.6% of games, offense coverage **49%** (an
upper-bound, censored target — the realized game found *a* path, maybe not the shortest), cap holes
13.9%. Proposed target Φ=γ^(my-moves-to-VCT) with Φ=0 floor + a defense channel (a value function for
"force a win", global by construction). **Banked negatives:** both yield predictions were wrong (81%
VCT / 5% cap, NOT cap-dominated as guessed); the 81% looks rich but is 96% trivial; a brief
"VCT-where-one-already-existed → impossible!" alarm was a **labeling** confusion (all fanned nodes are
pre-onset non-VCT). Separately this session: marked the slow CPU solver `gomoku/vcf.py` OBSOLETE
(reference/history only; GPU kernel is the sound, 1600× one). On `feat/gentle-rapfi-teacher` (not merged).

### 2026-06-27 — Φ distance-to-VCT field: the proof-frontier is learnable; CNN beats attention a 3rd time

Trained the **first real L2 model** (overnight, MPS, zero GPU contention with the live collector).
Target = the free dual potential from the VCT-reachability mine: `phi_off=γ^(my-moves-to-my-next-VCT)`
+ `phi_def=γ^(opp-moves-to-their-next-VCT)`, γ=0.8, read off the puzzle miner's per-ply verdicts (no
re-solve), cap excluded. **The gradient of Φ IS Jason's question** "which moves move the proof frontier
toward my VCT vs theirs". 40k games → 1.167M train / 101,745 held-out test, shard-disjoint (overlap 0),
0 frame mismatches. Scripts `gen_phi_dataset.py` / `train_phi.py` (committed); metrics
`~/data/puzzle_miner/phi_exp/phi_metrics.json`.

**Result — learnable + generalizes:** held-out **CNN** offense ρ=0.719 / R²=0.761 / reach-AUROC=0.912,
defense ρ=0.761 / R²=0.690 / 0.917; well-calibrated (top decile pred 0.91 → true 0.93). Two sharp
secondaries: **(1) NOT count-dominated** — CNN nearly doubles a ridge-on-raw-board baseline (offense ρ
0.36→0.72), unlike is-VCT recognition where logreg-on-counts nearly matched ⇒ closeness-to-a-fork is
genuinely *spatial*, structure lives in **distance, not presence**. **(2) CNN beats attention a third
time** — now param-matched (376k vs 348k) on the **global** target (attention's claimed home turf) with
3×+ the gradient steps (CNN early-stops ep6, best val by ep1; attn plateaus ~0.72 at ep20) ⇒ the
global-receptive-field bet **does not cash out** at this scale; conv tower + GAP wins. Surprise: defense
reads *better* than offense (net sees incoming danger best — aimed at the white wound).

**Banked negative + caveats:** the "global target is attention's chance" hypothesis was wrong (lost,
param-matched, epoch-alibi removed). Honest reach: it's the *realized-play proxy* frontier (upper-bound,
censored), single-position regression (not whole-game seeking), small/untuned. ρ on a proxy = green light,
not strong play — that's the Phase C hybrid-play eval (gate w/ Jason). **Default L2 arch = the CNN.** Synthesis:
`wiki/topics/phi-distance-field-learnability.md`. On `feat/gentle-rapfi-teacher` (not merged).

### 2026-06-27 — Molecule corpus banked: 146,655 non-VCF combinational forced wins, move-labeled

While Jason was at coffee (free GPU, his invite). Ran the corpus-scale writer of the §3 probe:
`harvest_molecules.py --side defender` — fan opponent-to-move pre-onset non-VCT nodes of real
Rapfi games, on each fanned (winner-to-move) board keep **VCT but NOT VCF** = forced wins needing
a *three* = the combinational "molecules". Move-labeled via the kernel's passive `return_move`.
GPU-only, 0 collector contention; banked to `~/data/molecule_gold/gold.jsonl.gz` (+ README, schema
matches puzzles.jsonl.gz). Scripts committed (79d7ad2).

**First bank:** 20,000 defender-side nodes / 68 of 400 shards / **25 min** → 4.01M fanned →
**3.71M VCT (92.4%)** → **146,655 non-VCF gold (3.95% of VCT)**, **99.0% distinct boards**, **100%
move-labeled**, from 3,438 source games. Boards **sparse** (winner mean 6.2 stones, median 6 — the
clean "combination already forced" regime). Gold **grows with distance-to-onset** (dist-1/3/5 =
28.8k/51.7k/66.2k): deeper = more genuinely *needs a three* = purer molecule (matches §3's
distance trend). The 92.4% VCT rate (vs §3's both-sides 81%) is the **defender-side knife-edge** —
nearly every pre-onset alt the defender could play loses by force.

**Each gold board pays twice** (the §4 thesis, now realized as data): a non-trivial offense
terminus (a molecule candidate) AND a defense lesson (defender's natural-looking losing move →
white-side wound). Only 68/400 shards consumed at the node cap ⇒ resumable, ~60× headroom on the
full corpus. Next natural uses (Jason's call): D4-canonical dedup → distinct-shape count; feed L1
stencil minimization; a non-VCF-aware Φ/defense target. On `feat/gentle-rapfi-teacher` (not merged).

## 2026-06-27 — `mega_vct_bb`: support + complete outputs (+ a canonical solver wiki page)

Extended the on-device VCT megakernel `scripts/vct_metal/mega_vct_bb.py :: solve_vct_mega_bb` with two
optional outputs, mirroring the passive `return_move` style and keeping the default `(win, hit, move)`
path **byte-identical** (asserted `_build_src(False,False)==_src()`; cross-check vs cell-scan `mega_vct`
0 disagreements). Each flag compiles its **own** kernel variant (`_KERNEL_CACHE`), so the fast path pays
nothing.

- **`return_support=True`** → `(B,4)` uint64 `support`: the cells the found proof line touches — a
  stencil seed / relevance window for the shape-library engine. Built by **return-path accumulation**: a
  per-frame `fsupp[]` merged into its parent ONLY on a winning return (`ret==1`), so abandoned/refuted
  branches never pollute it. OR-win adds move(+four-block); AND-win adds every defender reply; the three
  inline OR wins add move + completion/threat cells. `support ⊆ root EMPTY cells` (played cells, not the
  pre-existing threat stones — the ablation pass works the full board). Over-inclusive vs minimal, by design.
- **`complete=True`** (slower) → `(B,4)` uint64 `winmask`: ALL winning FIRST MOVES. The root OR node stops
  short-circuiting and records every winning forcing candidate; non-root nodes untouched (their
  short-circuit IS the per-root-move verdict). Winmask = winning *forcing* first moves (fours +
  tempo-guard-passing threes) = exactly the VCT first moves; non-forcing free-wins in already-won
  positions are correctly excluded.

**Validation (the work).** Invariants (B=32, budget per Jason = `max_nodes=500`, runs in seconds, added as
`test_support_and_complete_invariants`): support-variant verdict/move identical; complete `win`==default
`win` on clean boards; default move ∈ winmask; support cells empty/contain-move/zero-on-loss — all PASS.
**Gold** (`~/.claude/jobs/d524d833/tmp/gold_complete.py`, 6 winning boards, all empties × all replies via
the kernel, vs vcf's exact forcing-move generation): **0 unsound winmask moves, 0 winning-forcing-moves
missing.** *Lesson banked:* my first completeness oracle FAILED (21 "misses") because I forgot vcf's
**tempo guard** (`_defender_has_four_or_five`, vcf line 942) — a three where the defender has a counter
four/five is NOT a forcing VCT move; `verify_one` (wins-vs-all-replies) still flags it in an overwhelmingly
won position, but it is correctly absent from winmask. Adding the guard to the oracle → clean PASS. **The
solver was right; the verifier was wrong** — exactly the kind of subtlety the move-verifier pattern exists
to catch.

New canonical wiki page **`wiki/topics/mega-vct-solver.md`** (the API/contract reference) + index doorway
row; `gpu-vct-feasibility.md` §9 and `vct-backward-mining.md` §5 updated with pointers. On
`feat/gpu-vct-support-complete` (not merged).

## 2026-06-27 — CPU vcf solver RETIRED (gated, not deleted) + fast/deep VCT test tiers

Two linked changes on `feat/cpu-solver-retire` (branches off the just-landed
return_support/complete solver work; NOT merged).

**(1) CPU solver retired as a runtime dependency.** `gomoku/vcf.py` is the slow CPU
AND/OR solver (~0.65 ms/node, ~90s tail on hard 15×15). It was a wonderful bootstrap
and stays fully intact as the kept oracle/reference, but Jason's standing intent —
session after session he'd stop me reaching for it — is now enforced in code: the four
public entry points (`solve_vcf`/`solve_vct` + `*_from_planes`) raise `CpuSolverRetired`
with a message pointing at `scripts.vct_metal.mega_vct_bb.solve_vct_mega_bb` (the GPU
solver). The ONLY sanctioned bypass is env `GOMOKU_ALLOW_CPU_SOLVER=1` (fixture-gen,
deep validation, the kept-oracle test suite). All internals untouched; no CPU feature
parity intended.

**Runtime reaches surfaced for Jason's triage (all OPT-IN, so default runs are
unaffected; isolated on a branch so nothing breaks until merge):**
- `gomoku/self_play.py` — the VCF/VCT **teacher** (`vcf_teacher`/`_apply_vct_teacher`,
  `configure_vcf_teacher`): `solve_vcf/vct_from_planes` + `solve_vcf`. Fires only when a
  run enables the teacher (e.g. the `gentle-rapfi-teacher` line) → will throw unless
  ported to the GPU solver or run with the override.
- `gomoku/eval.py` — the VCF-**overlay** player (`solve_vcf` before/at MCTS leaf).
- `gomoku/train.py` — the MCTS-leaf-VCF flag (derby-b3n) wiring into the above.
These three are the triage list: port to `solve_vct_mega_bb`, set the override per-run,
or retire the lever.

**(2) Fast/deep test tiers (kills the minute-plus walls).** The slow test walls were never
the GPU solver (4s flat at B=16k) — they were slow ORACLES in the loop (live `vcf` ~90s
tail; the cell-scan `mega_vct`). Restructured:
- `scripts/vct_metal/regen_vct_fixture.py` — one-shot, sets the override, solves a fixed
  seeded real-position stack with the CPU oracle at high budget, keeps clean (non-cap)
  boards, writes `(boards, win, hit, move, winmask)` truth to committed
  `scripts/vct_metal/fixtures/vct_golden.npz` (seed+budget recorded inside).
- `scripts/vct_metal/test_mega_vct_bb.py` — FAST tier: loads the npz (NO vcf, NO cell-scan
  at test time), diffs `solve_vct_mega_bb` at `max_nodes=500` (cap→skip; non-cap verdicts
  are budget-independent so they MUST match high-budget truth) + the support/complete
  self-oracle invariants. Runs in seconds.
- `scripts/vct_metal/validate_deep.py` — DEEP tier (sets override): live vcf at high budget,
  larger n, the all-empties winmask soundness+completeness gold (oracle includes vcf's
  tempo guard `_defender_has_four_or_five` — the subtlety from this session). Run on-demand
  / when the verdict changes, not in the gate.

**Validated:** fast tier PASS; gate tests 9/9 (`CpuSolverRetired` raised without override,
runs with it); kept-oracle + overlay tests (`test_vcf`/`test_vct`/`test_eval_vcf_overlay`)
green via the session-wide override in `tests/conftest.py` (at the DEFAULT board size 9 —
they encode 9×9 cells). The reusable lesson banked on [topics/mega-vct-solver.md]: commit a
golden fixture, never re-derive truth at test time; the GPU solver is fast, slow oracles are
the test-wall.

## 2026-06-28 — md-extraction CRACKED (#91): the §3 stencil-minimizer blocker is gone; load-bearing W measured

The single named blocking prerequisite for the shape-library L1 minimizer
([shape-library-engine.md](wiki/topics/shape-library-engine.md) §8) was **md-extraction**:
the minimizer must ablate stones on **mate-distance invariance** (a load-bearing stone is one
whose removal *shortens* the mate — §3 correction #2), but `solve_vct_mega_bb` returned only
`(win, hit_cap)` and capped *nodes*, not depth. Cracked tonight on `feat/md-extraction` (NOT
merged). Autonomous overnight run; Jason's charge was "try things that don't work and write
them down so we learn" — both the wins and the honest bounds are below.

**Approach (de-risked by a 7-agent design workflow first).** A background Workflow ran 5
read-only design analysts (kernel audit, md theory, approach ranking, validation, minimizer) →
an adversarial reviewer → a synthesizer, before any kernel surgery (the megakernel is the #1
silent-wrong-answer trap). The adversary earned its keep: it killed the plan's CPU md_min
cross-oracle as **mis-calibrated** — the kernel's `candidate_own` (own-only Chebyshev-2) is
*narrower* than CPU `vcf`'s any-stone candidate set, so `md_gpu > md_cpu` can occur with **no
bug** — and it would have re-summoned the retired CPU solver (against canon + the
`feedback-trust-validated-oracle` memory). Dropped; validate **GPU-self** instead.

**The kernel primitive (issue #91, the chosen "Approach C").** A new compiled variant gated by
a `depth_cap` flag adds **one input** (`max_depth`, per-board int32) and **zero outputs**: a
branch reaching frame `sp == max_depth` returns a clean `ret=0` (a definitive "no forced win
within `sp < max_depth` frames") **without** setting `hit_cap` and **before making any move**
(so `own==own_in`/`opp==opp_in` at break is preserved — carriers/w safe). Then
**`md_min(b) = min{ d : solve(b, max_depth=d).win }`**, read from the boolean verdict alone, so
it is **order-independent** (no move-ordering / OR short-circuit can move a True/False
threshold), monotone, minimax-correct. `solve_md_min(boards)` binary-searches it per-board —
every board marches its own bracket in **one** bulk call, so a whole corpus resolves in ~5 flat
tails (the call-cost law). The edit is **purely additive**: `_build_src(s,c,cr,w,depth_cap=False)`
is **byte-identical to git HEAD** for all 16 flag combos (verified) — no existing variant moves.

**md is in FRAME units, not attacker-plies.** A four = +1 frame, a forcing three = +2 (it
pushes an AND node), and an inline win (immediate-five / sound double-four / fork-three)
**collapses** (ret=1 at its frame, +0). This is the right *consistent* measure for shortening-
detection but is coarser than the CPU's `mate_distance`; never reconcile the two (banked as
invariant #9 + a §metric note).

**Validation — all green, GPU-only (no CPU).** A gate confirmed, in order: [1] byte-identical
default vs HEAD; [2] depth_cap composes additively; [3] `max_depth=MAXD-1` reproduces the
default `(win,hit,move)` exactly; [4] **depth monotonicity** `win(d)` never True→False (the
adversary's "single most important" correctness gate); [5] `solve_md_min` brackets correctly
**and equals an independent linear scan** on every uncapped board. Permanent FAST-tier tests +
a GPU-self golden fixture (`regen_vct_md_fixture.py`, no CPU) added.

**L1 md-invariant minimizer built + measured** (`scripts/threat_shapes/md_minimize.py`).
Cumulative lockstep ablation, directional single-cap tests (exploiting freestyle monotonicity):
OWN stone probed at cap `md0` (clean win → redundant DROP; nowin → load-bearing `B` KEEP); OPP
stone at cap `md0−1` (clean win → a shorter mate opened → load-bearing `W` KEEP; nowin → DROP);
hit_cap → KEEP (fail-safe). No windowing (sound; sidesteps the found-line-vs-shortest-line
windowing risk the adversary flagged). One bulk call per ablation step, all boards in lockstep.

**RESULTS — `molecule_gold` (16,345 non-VCF combinational VCTs, the first 16,384 by `dist`):**
- **md_min over the corpus in 19.6 s; ablation 98 s.** 99.8% resolved, **zero ceiling pressure**
  (max md0 = 9 ≪ MAXD=32). Reduction: orig 13.2 stones → **4.91** (B+W) ablated (63% ↓).
- **Load-bearing W is the long-VCT phenomenon — MEASURED.** W-rate by md0: **md0=1 → 0%**
  (correct: inline root wins are degenerate, defender-cap=0), md0=2 → 72%, **md0≥4 → 100%**.
  Exactly the wiki's claim ("load-bearing white is the *rule* in long VCTs — they're long
  *because* white denies the short wins"), now quantified.
- **The `w` channel (#90) is a ~10× over-approximation.** It flagged 88,637 defender stones;
  ablation distilled only **8,694 actually load-bearing** (9.8%). So the cheap `w` over-approx is
  a real ~10× over-count of the minimal load-bearing W — md-ablation is *necessary* to get it.

**Honest bounds / negatives (the "write it down" half):**
- **73% of `molecule_gold` is md0=1** — root-collapsed inline wins (a fork-three detected at
  frame 0 without descending). The R5 inline-collapse *dominates* this corpus, so it's a **poor
  substrate for the W phenomenon** (the real W story lives in the md0≥2 tail). This is why the
  denser/deeper real-game `enable_serial` corpus is the cleaner test (run in progress; contrast
  to be appended).
- **Corpus caveat:** `molecule_gold` was harvested by *perturbing the defender* (injecting a
  blunder), which likely inflates the W-rate — `enable_serial` (real-game, unperturbed) is the
  control.
- **Vocabulary did NOT fully saturate at 16k** (902 distinct ablated stencils, curve decelerating
  but still +35/800). Honest yellow flag for the "finite vocabulary" bet; needs more boards
  (and/or the §3-deferred D4 fold to dedup 8×) to call.
- Two accepted, *length-over-estimating / over-keeping* (never unsound — L0 re-verifies) md
  bounds carried forward: the `def_tempo` veto can inflate three-opening lines; the inline
  collapse can hide a ≤2–3-ply shortening at constant `sp` (future fix = emit `md = sp +
  leaf_offset`, additive).

**RESULTS — `enable_serial` contrast (the deep, real-game, UNPERTURBED control; 105 resolved of
the 512 deepest-`run` boards, `max_nodes=1500`, frame-cap `hi=16`).** The W story flips from
"rare" to "universal" exactly as the depth hypothesis predicts. md0 is **deep** (histogram spans
6–13 frames; vs molecule's 1–9 with 73% at md0=1) — these are real enabling *setups*, not root
collapses. **Load-bearing W is 100% at *every* md0 (6–13), mean ~10 W stones/stencil** (vs
molecule ~0.5) — on *unperturbed* real-game data, so it is **not** a harvest artifact. Three sharp
contrasts: **(1)** the `w` channel (#90) is **regime-dependent** — a ~10× over-approximation on
shallow molecules (9.8% load-bearing) but only **~1.3×** on deep shapes (1106/1471 = **75%
load-bearing**); the cheap `w` is a *good* approximation exactly where defensive structure is real.
**(2)** deep stencils barely reduce (**37%** vs 63%) and the ablated `B+W` object (mean **20.6**)
is **larger** than `support∪carriers` (14.0) — because for deep defensive shapes the baseline is
**incomplete** (no W), not over-inclusive; ablation *adds* the ~10 load-bearing W the carriers
heuristic cannot represent. **(3)** vocabulary does **NOT** saturate — **96% of deep stencils are
distinct** (101/105, near-zero repetition) vs molecule's 6% — the §7 "library too specific" risk is
**real for the deep regime** (shallow molecules form a finite vocabulary; deep enabling-shapes are
nearly all unique). Honest costs banked: **407/512 deep boards capped** at this budget (the deep
tail wants more nodes/depth than `max_nodes=1500`/`hi=16`), and no-window ablation took **590 s for
105 boards** — the dense-board (32.6 stones) perf wall; **windowing is the fix** (§8 NEXT). The
16,384-board `enable` run did not finish under no-window (killed) — that *is* the perf finding.

**Net:** the §3/§8 blocker is gone, the L1 minimizer exists and produces typed minimal stencils
`(B, W, support, md0)` today, and the load-bearing-W hypothesis is confirmed on BOTH a shallow
perturbed corpus (W-rate 0%→100% with md0) and a deep unperturbed one (100% W, mean ~10), with the
`w`-channel over-approximation quantified and shown regime-dependent. Next: the `enable_serial` contrast (deeper VCTs), then the v0
distance-field + fork player (§5) and L2 (§4). Files: `scripts/vct_metal/mega_vct_bb.py`
(`max_depth`/`solve_md_min`), `scripts/threat_shapes/md_minimize.py`,
`scripts/vct_metal/regen_vct_md_fixture.py`, FAST tests; docs
[mega-vct-solver.md](wiki/topics/mega-vct-solver.md) (`max_depth` + invariant #9) +
[shape-library-engine.md](wiki/topics/shape-library-engine.md) §3/§8.

### 2026-06-28 (second pass) — calibration of the claims above (accuracy/durability; Jason: "null result is also fine")

A same-day audit of the morning entry's conclusions. The **kernel/tool results stand**; several
**interpretive claims were overstated** and are corrected here (originals left intact above per the
append-only norm). New analysis is CPU-only on the banked stencil dumps (`scripts` in
`$JOB/tmp/analyze_vocab.py`); no GPU.

- **RETRACTED — "deep VCT shapes don't saturate / have no small vocabulary / the §7 'library too
  specific' risk is real."** This was an **exact-match-on-large-objects artifact.** The banked
  stencils are wildly **over-inclusive** (mean **41 cells** on `enable`: found-line `support`
  openings, never ablated, + ~10 over-counted `w`-derived W + B), and *every* diversity metric is
  size-dominated. Measured (matched n=105): exact-set distinct **enable 96% / molecule 10%**, but
  **IoU≥0.5 clusters enable 2 (2%) / molecule 3 (3%)**, IoU≥0.3 → **1 each**. So exact-match
  over-counts diversity (big objects rarely identical) and IoU under-counts (big dense blobs all
  overlap); the truth is **unresolvable from these stencils.** Honest status: **the vocabulary /
  saturation question is OPEN (a null result)** — it needs *minimal* stencils (ablate the support
  openings, fold D4, use minimal-W) and a size-controlled metric before any saturation claim is
  meaningful. The morning "matched-n ~10% vs ~97%" contrast is real but **not** evidence of a
  tactical-vocabulary difference — it is mostly the ~4× size difference.
- **TIGHTENED — the W-rate-by-md0 curve.** "0% at md0=1" is **definitional, not a finding** (md0=1
  ⇒ the `md0−1=0` cap always returns nowin ⇒ no W is *testable*). For **deep** mates, "≥1
  load-bearing W" is **near-definitional**: a long forced mate *is* the defender delaying, so
  removing a delayer shortens it — "100% of deep boards have a load-bearing W" is close to what
  "deep mate" means. The informative, non-trivial numbers are the mid-range rate (**md0=2 → 72%,
  md0=3 → 59%**) and the **counts/ratios** below — not the 0%→100% sweep.
- **TIGHTENED — "load-bearing W" is an operational, order-dependent definition:** a stone whose
  *single* removal (under cumulative-greedy ablation) opens a win at `md0−1`. It can miss
  *jointly* load-bearing sets and is order-sensitive; it is a reasonable proxy, **not** the unique
  minimal load-bearing set. So the `w`-channel ratios (**~10×** over-inclusive on shallow molecules,
  **~1.3×** on deep) compare `w` to *this* proxy; the regime-direction (w over-approximates far more
  on shallow than deep) is the durable part, the exact multiplier is proxy-dependent.
- **CLARIFIED — md_min validation scope.** Verified: byte-identical default vs HEAD (16/16),
  depth-monotonicity (no counterexample; monotone by construction), the bracket, and md_min == an
  independent **linear scan**. The linear scan uses the *same* kernel, so this is **internal
  consistency + the depth-cap mechanism**, NOT a cross-check of md_min's *absolute value* against an
  independent oracle (the morning entry's "validated" slightly oversells this). No well-calibrated
  external md oracle exists — the CPU searches a *different* fragment (`candidate_own` own-only vs
  any-stone) — except on the **VCF (four-only) subset**, where the fragments coincide and a gated
  CPU `mate_distance` cross-check *would* be sound (an open, durable follow-up). The **FRAME unit**
  (`md = F + 2T + 1`) is **source-traced, not independently measured.**
- **TIGHTENED — causality.** The data show md-depth *correlates* with load-bearing-defender count;
  "long *because* white denies the short wins" remains a **hypothesis** consistent with (not proven
  by) the measurement.
- **STANDS (solid):** the kernel primitive (byte-identical default, monotone depth-cap, order-
  independent md_min), the validated FAST tests + golden fixture, and the **minimizer as a working
  analytical tool**. The honest net is a **tool + a method**, with the headline tactical questions
  (vocabulary, minimal-W) still open — which is a fine place to be.

**Follow-ups updated (#92):** ablate the `support` openings (the dominant over-inclusion source) +
fold D4 + minimal-W, THEN re-ask vocabulary on minimal stencils with a size-controlled metric; and
the VCF-subset CPU md cross-check as a one-time absolute-value validation.

## 2026-06-28 (n=1225 streaming scale-out) — the vocabulary null holds at 10× scale; a resumable harness for unbounded n

Jason: "run a larger sample, n=1000+, append-only, trivially resumable, to prove that out and set us
up for unbounded n later." Built `scripts/threat_shapes/md_minimize_stream.py` (reducer-over-a-log:
the JSONL output is the only state; each board content-addressed `sha1(corpus|atk|dfd)`, written once
with status ok/capped/dead; resume = skip logged ids; capped ids recorded so they aren't retried) +
`analyze_vocab_stream.py` (O(n) exact/D4 distinct over all n, IoU on a capped sample). Commit ca26b58.

- **Input-order is the throughput lever (a real finding).** Deepest-first (the prior n=105 strategy)
  cherry-picks pathologically-unsolvable boards: **256 → 51 ok / 205 md0-capped in 1109 s (~20% yield)**.
  Shuffled (seeded, representative real-game sample): **256 → 237 ok / 19 capped in 287 s (93% yield,
  4× faster)**. The capped boards are unusable anyway (no md0 ⇒ nothing to minimize), so deepest-first
  spends ~80% of compute on dead ends. Full shuffled run: **1225 ok / 55 capped / 0 dead in 1278 s
  (~21 min)**, log `~/data/md_stencils/stream_enable_shuf.jsonl`.
- **The vocabulary null is now stable at n=1225 (10× the n=105 second-pass result).** mean stencil
  **22.6 cells** (B 6.3, W 4.7); **exact-distinct 98% / D4-distinct 96%**, but **IoU≥0.5 → 19%
  clusters / IoU≥0.3 → 0.25% (one blob)** — the "diversity" number swings ~400× with the threshold.
  No stable vocabulary count exists because the metric is dominated by **stencil size + threshold**,
  not tactical content. Confirms the second-pass retraction: the question is **unanswerable until the
  stencils are minimal** (openings ablated).
- **NEW sub-finding — D4 is NOT the dedup lever.** The second pass flagged D4-folding as a possible
  ~8× deduplicator; at n=1225 it moves exact-distinct only **98% → 96%** (≈2% of stencils are D4
  images of another). So the inflator is **over-inclusion** (un-ablated `support` openings → mean
  22.6 cells), not missing symmetry. This re-prioritises #92: **ablate the support openings first**;
  D4 is a rounding error by comparison.
- **W findings reproduce cleanly + stably at scale.** W-rate **0% at md0=1 (definitional)**, **100%
  at md0=2, 95% at md0=3, 100% at md0≥4**; `w`-channel **2.2×** the single-removal load-bearing set
  — squarely between molecule's ~10× (shallow) and the deepest-band's ~1.3×, i.e. the `w`
  over-approximation tightens monotonically with mate depth, now traced across three regimes.
- **Honest scope caveat.** Shuffled enable is **moderate depth** — 91% of the 1225 are md0≤4 (mean 22.6
  cells), vs the prior n=105 *deepest* band (mean 41 cells, IoU≥0.5 → 2 clusters). The deepest band
  shows an even starker size-driven collapse; both regimes point the same way (over-inclusion
  dominates), but the n=1225 sample under-represents the 41-cell tail. Unbounded-n on the deep tail
  needs the #92 budget/windowing work (the deepest band caps at ~80%).

Net: the **harness is the deliverable** (resumable, append-only, 96% yield, ready for unbounded n),
and it **proves the second-pass null durable at 10× scale** while sharpening the path forward (ablate
openings ≫ fold D4). Files: `scripts/threat_shapes/md_minimize_stream.py`,
`analyze_vocab_stream.py`; log `stream_enable_shuf.jsonl` (+ `stream_enable.jsonl`, 51 deep stencils).

## 2026-06-30 (issue #98) — VCT-terminus self-play: end games at the first cap50 VCT, not at five (9×9 generator + the 15×15 oracle ported to 9×9)

Jason: "9×9 AlphaZero, but this time instead of playing to win, play to VCT @ 50-node budget. VCT is the
terminus — with that in hand we can win any game, so further searching is noise." This is idea-pile #11
(⭐ "first VCT at median ply 19") built onto the generators, using the validated `mega_vct_bb` GPU oracle
as a batched per-ply terminal test across the wave of live games. **Code-only capability landed (default
OFF = byte-identical); the training-science comparison is the next phase.** Merged `feat/9-9-az`.

- **Step 0 — the oracle was 15×15-ONLY; ported to be N-general (the prerequisite nobody had hit).**
  `scripts/vct_metal/bb.py` computed `TOPMASK=(1<<(NN-192))-1` and masked only bitboard **word 3** — correct
  only when the board fills words 0–2 (N∈{14,15}). At N=9 (NN=81) it crashes at import (negative shift) AND
  would leak off-board "empty" cells in the high words (`empty=~(own|opp)` unmasked). `n_sweep`'s "N" is
  *pool size*, not board size — **the solver had never run at board-9.** Fix: replace the word-3-only
  `TOPMASK` with a full 4-word `BMASK[4]` (= `bb_ref.BOARDMASK` split) + a `mask4()` helper, applied at
  every masking site (`bb.py` ×3, `mega_vcf_bb.py`, `mega_vct_bb.py`). **Provably byte-identical at N=15**
  (words 0–2 masked with all-ones = no-op). Revalidated: `test_bb.py` (Metal port vs golden `bb_ref`)
  ALL PASS at N=9 (1800 positions, 0 errors) AND N=15; `test_mega_vct_bb.py` 16/16 vs the committed N=15
  golden (byte-identical confirmed); 4 hand-verified 9×9 VCTs correct with sound winning moves. The
  15×15 threat-shapes tool `certificate_falsification.py` still compiles unchanged (TOPMASK kept).
- **Step 1 — the generator hook.** New `--vct-terminus` / `--vct-terminus-budget 50` (`selfplay_worker.py`)
  → `configure_vct_terminus()` (`self_play.py`; process-global gate, so default-off never imports MLX). In
  the wave loop (native + Python paths), BEFORE search each ply: gather every active game's side-to-move-
  relative root planes → **ONE** bulk-synchronous `solve_vct_mega_bb(boards, max_nodes=50, return_move=True)`
  (the call-cost law — never solve-in-a-loop) → any game where the side to move has a forced VCT is
  terminated now: record the decisive position with the oracle's winning move as a **one-hot policy target**
  (= the seek-VCT objective), credit that side the win, drop it from the wave; survivors continue to normal
  MCTS. The exact terminal value flows through the SAME sign-flip + mate-distance discount as a real five
  terminal. Plane convention reuses `vcf.solve_vct_from_planes` (attacker=plane 0, defender=plane
  HISTORY_PLY). Gumbel paths guarded (NotImplementedError); ownership masked at a VCT terminus (not a real
  five board).
- **Coexistence works (the frankenstein's viability question).** native-C MCTS + PyTorch-MPS evaluator +
  MLX(Metal) oracle all run in ONE process — verified live during generation, not just at import.
- **E2E smoke (random net, 9×9, 64 games each, augment off):** plies_mean **36.0 → 19.9 (0.55×)**, median
  **35 → 20**; every terminus game decisive (a VCT is always a win), **64/64 ended at a VCT** with a one-hot
  oracle-move terminal (vs 1/64 accidental for the five-terminated baseline). Median terminus ply **20**
  matches the 15×15 rapfi-corpus finding (median first-VCT ply 19–20) — on a *different board size with a
  random net*. The "~half the plies" prediction lands almost exactly.
- **Tests:** `tests/test_vct_terminus.py` (5 deterministic unit tests, monkeypatched oracle) + a focused
  gen-path gate (native / gumbel-guard / teachers / swap2 / ownership / board15) green.
- **Next (the science, gate with Jason):** a real 9×9 training slice with `--vct-terminus` vs a
  five-terminated control at equal wall-clock — MTTE/Δelo, value-head calibration on held-out oracle
  verdicts, and whether "reach a VCT" self-play trains a stronger net faster (idea #11's #4 ms-ladder
  crossing). **Caveat to watch (idea #11):** terminating at VCT removes defender "play-it-out" learning
  past onset — the knife-edge result predicts it should SHARPEN signal, but measure.

## 2026-06-30 (issue #99) — VCT-finisher hybrid player: policy to the VCT, GPU oracle to the win (eval vs anything, no special harness)

Follow-up to #98. A VCT-terminus net stops at the first VCT and takes the oracle verdict — great for
training, but to EVALUATE it vs any opponent through the standard match harness it must win a REAL game
(actual five-in-a-row), not lean on a "VCT = win" special harness. Jason: "extend the player so it plays
by policy to VCT then just hammers the rest out rote... if we capture the logic in the player, we can eval
vs anything, and not be trapped with a special harness." Merged `feat/vct-finisher-hybrid`.

- **Built `vct_finish_picker(base, *, budget)`** in `gomoku/eval.py` — the GPU-oracle sibling of the
  existing CPU `vcf_overlay_picker` (same compose-a-picker shape). Each turn: batched
  `solve_vct_mega_bb(state.board[None], max_nodes=budget, return_move=True)`; if the side to move has a
  forced VCT, play the oracle's winning move, else delegate to `base`. `state.board` is already the
  (2,N,N) side-to-move-relative solver input (plane 0 attacker / plane 1 defender) — no conversion, no
  HISTORY_PLY. `budget=0` = OFF (byte-identical, never imports MLX). Threaded via a `vct_finish_nodes`
  param on `mcts_picker` (wrapped OUTERMOST — VCT ⊇ VCF). Wired: `match.py` `model:` spec
  (`model:checkpoint=X,vct_finish=50` — ad-hoc eval vs any baseline / external Rapfi) + `eval_worker.py
  --vct-finish-nodes` (SEQUENTIAL; the parallel path is guarded — MLX under multiprocessing fork is
  unvalidated / Metal-wedge risk). Solver import behind `_load_vct_solver()` so tests monkeypatch without MLX.
- **Why cap50 also CONVERTS, not just detects:** after each forced exchange the remaining position is a
  SUB-proof of the original ≤50-node VCT proof, so re-solving each turn keeps handing back the next
  forcing move until a real five. Whatever the opponent plays, the win stays forced.
- **E2E smoke (random net, 9×9, standard `play_match_pickers`):**
  - vs random: plain 40W-0L (0 finisher fires — correct); hybrid 39W-1L, **finisher fired 54× over 264
    picks** → converts VCTs to genuine wins.
  - vs heuristic: plain 0W-40L, hybrid 0W-40L, **0 fires over 180 picks** — a random net earns no VCT vs a
    defender, so the finisher never fires and NEVER HURTS (≡ plain, only ever plays proven wins).
  - **Crafted open-four, hybrid to move (terminal_before=False): 1W-0L, fired once → wins a REAL
    five-in-a-row** through the standard harness. The money shot: policy→VCT→hammered to five, no special harness.
- Tests: `tests/test_vct_finish.py` (5 unit tests, monkeypatched oracle) + eval-path gate
  (vcf-overlay / proven-prop / tree-reuse / panel / color-split) green. This hybrid is also the deployable
  web-UI player and realizes the wiki's long-anticipated "Phase C hybrid-play eval" (seeker-steering).

## 2026-06-30 (issue #100) — VCT-terminus science run: the 9×9 A/B (throughput win, robustness LOSS)

The science slice for #98/#99. **Full synthesis: [wiki/topics/vct-terminus-selfplay-result.md](wiki/topics/vct-terminus-selfplay-result.md).**
Matched 9×9 pair (`scripts/run_sweep.py`: `vctsci-terminus` vs `vctsci-control`), byte-identical except the
terminus; both cloned from `derby-v9-small` (fresh 64×4) minus `--gumbel-root` (terminus guard) and
`--vcf-teacher` (VCF ⊆ VCT). Grown by `--resume` to **e500** (100 was smoke — strength only broke out in
the extended epochs). wandb terminus `cc0fy0ao` / control `7cu4ho9w`. Jason: "start with 100 epochs… keep
going to 500, it's screaming fast." Merged `feat/vct-terminus-science`.

- **Training:** terminus self-play plies 36→**9.1** (control 34→11.8), wall **~2.1 s/epoch** (control
  ~4.7 s) — the terminus reaches EQUAL fixed-baseline strength at **~45% of the control's wall-clock**
  (idea #11's throughput claim: CONFIRMED). Internal EMA elo @e500 **1366 vs 1347** — a tie. Both
  fast-attack-narrow but pass the balanced-baseline test (h/la2/la4 climb together).
- **⚠ Eval gotcha (reusable):** `worker_weights.pt` = the **EMA** weights (what self-play + internal eval
  use); `load_checkpoint(epochNNNN.pt)` returns the **raw** state_dict, *far weaker* under `ema_tau=0.99`
  on short-game training (terminus **6%** raw vs **68%** EMA vs heuristic). Eval the EMA. First pass used
  raw and looked like a 6% net — chased it down, re-ran on EMA.
- **Fixed baselines (EMA, n=40):** terminus 66/81/49% (h/la2/la4), control 80/75/38%, champion 62\*/75\*/61\*
  (\*draw-saturated). **Finisher lift concentrated at the top: terminus +12.5% vs la4**, ~0 else; control ~0
  everywhere. Fixed baselines saturate for strong nets ⇒ coarse ruler, gate on H2H.
- **Head-to-head (the headline — both predictions REFUTED):** terminus wins **0 of 120 games vs the
  control** (25%: 0W-20L-20D in every config), and **never reaches a VCT** (finisher fires **0**/1060).
  Loses **0-40** to the champion. Champion 40-0 on the control (calibration). Jason predicted a control
  *crush* + a champion win; Claude hedged to a control-crush too. **Both lost to the same mechanism**: the
  terminus's only opponent was a non-defending copy of itself, so it never learned to defend or play a long
  game; a sound opponent denies every VCT and it collapses out-of-distribution. The control, playing to
  five, learned both sides ⇒ wins the sibling H2H at equal fixed-baseline strength. **Non-transitivity in
  the flesh.**
- **Rapfi coda (giggles):** `champion+finisher vs Rapfi@50ms` (native mix9svq NNUE, 9×9) = **20 straight
  draws** (0W-0L-20D) — even a shallow Rapfi denies every VCT; 9×9 is drawish at this level.
- **Verdict:** idea #11's throughput claim holds; its "strong player" claim is refuted — VCT-terminus
  self-play induces attack-only specialization (idea #11's own caveat as the dominant effect). The seek-VCT
  *objective* survives; the missing piece is DEFENSE (record past-terminus for the losing side / mix full
  games / curriculum). Next probe filed: **#101** (train the terminus long — p90 plies → 81, or wicked
  strong at short games?). Eval harness: `scripts/vctsci_finisher_eval.py` (`--list`/`--run`/`--collate`,
  one matchup/process).

## 2026-07-01 (issue #101) — train the VCT-terminus player LONG: does p90 plies reach 81? (no — a stable attractor at ≈14.5)

The natural next probe of #100. **Full synthesis: [wiki/topics/vct-terminus-selfplay-result.md](wiki/topics/vct-terminus-selfplay-result.md) § Long-run coda (#101).**
Question: train the VCT-terminus player (games END at the first cap50 VCT) continuously with `--internal-eval`
**off** — just train, no train-time evals — and watch `selfplay/plies_p90`. The retired 9×9→11×11 graduation
gate was **p90 = 81** (a full board). Two hypotheses: **(A)** p90 climbs → the net learns to *avoid* VCTs
(emergent VCT-avoidance at equilibrium); **(B, Jason's bet)** it never gets there and just gets "wicked strong
exploring short games."

**Result — Hypothesis B held.** A **fresh from-scratch** run (the #100 terminus buffer died with its worktree;
fresh also gives a clean single p90 timeline through the collapse), `vctsci-terminus` recipe verbatim (64×4,
`n_sim=100`, 4 workers, `--vct-terminus --vct-terminus-budget 50`, `ema_tau=0.99`, 64 SGD-steps/epoch). wandb
**`kgajrge4`** (`jasonyandell-forge42/gomoku`; run dir `~/data/vctsci-101-long/`, outside the repo). Reached
**~2,700 epochs (≈14× the #100 e500 slice) and was still riding its 12h / 1M-epoch wall at writeup**; the
verdict was locked by ~e1,200 and only hardened over the next ~1,500 flat epochs. Hand-off was to the #103
moonshot.

- **The p90 trajectory (verified from the run, 200-epoch block means):** cold **~28** → collapsed to a trough
  **11.9** by ~e85 → a **decelerating creep** back up: 11.9 → 12.7 → 13.2 → 13.4 → 13.6 → 14.0 → 14.4 → **14.7**,
  increments off the trough **+0.8, +0.5, +0.2, +0.2, +0.4, +0.2 …** flattening to **~14.5–14.6** and holding
  for the final ~1,000 epochs. **Never a hint of a march toward 81** (≈6× short). mean plies pinned **~9.6**.
  `loss/policy` fell monotonically **4.38 → ~2.17** then flat; `loss/value` **0.39 → ~0.022**, flat. Every dial
  converged — a **stable attractor**, just a *rising* one in the low teens.
- **Mechanism (co-evolution, capped by the self-play ceiling):** the 11.9→14.5 creep is the defender (its own
  EMA twin) learning to **postpone** the VCT a few plies, *never to prevent* it — self-play offers no opponent
  strong enough to *punish* weak defense. The net got sharper at the **same ~9-ply game** (pl/vl down), not at
  longer games. Same self-play ceiling #100 exposed head-to-head; Hypothesis A (VCT-avoidance) refuted.
- **Confound (cap50 recall):** the terminus ends at the first *cap50*-detected VCT; as play sharpens some real
  VCTs need **>50 nodes** and cap50 misses them, so **part** of the p90 creep is the detector losing recall on
  the shifting distribution, not genuine defense. `plies` is therefore an **unreliable defense proxy** — only
  `fires>0` vs a real opponent (the #100 finisher yardstick) settles "did it learn defense," and #100 already
  answered **no** (fires = 0 vs the control/champion).
- **Honesty caveat:** #101 ran with **no evals**, so "gets stronger at short games" is **inferred from falling
  pl/vl, not a measured strength number** — a plausibility argument, not a proof.
- **Verdict + the way out:** the self-play defensive ceiling is **structural**, not undertraining — 2,700 epochs
  of pure self-play buys a few plies of postponement and nothing more. The path past it is **opponent-independent
  defensive signal**: the supervised VCT aux-head (**#102**) / the from-scratch VCT-gate + aux-head gauntlet
  (**#103**, now in-progress), which regress the VCT structure directly rather than hoping a non-defending twin
  will teach defense. Cross-ref #100 (the head-to-head yardstick), idea-pile #11 (lineage).

## 2026-07-01 (issue #103) — the VCT-defense aux head: it learns the percept, the policy never acts on it

Executes the #102 aux-head design. **Full synthesis: [wiki/topics/vct-defense-aux-head-result.md](wiki/topics/vct-defense-aux-head-result.md).**
Built a per-cell **"VCT-blunder map"** defense aux head (supervised by the GPU mega VCT solver: for each legal
move, does it walk the side-to-move into a forced opponent VCT; defense label via **escape-search over every
legal move** — a move is "lost" iff all children lose). `--aux-vct-weight 0.1`, default 0 byte-identical.
Ran **two** experiments; **both failed to make the net *defend*, instructively.**

**Experiment A — 9×9 from-scratch moonshot (wandb `8mtowemb`, retired e1152).** From-scratch 9×9, VCT-terminus
gate (budget 50), the defense head (weight 0.1, full escape-search), + surviving Bruce levers (value-discount
0.98, global-pool, WL2 stack, 64 SGD-steps/epoch, 64×4, 1.5M buf, sims=100). NO gumbel (terminus-incompatible),
NO `--vcf-teacher` (CPU solver RETIRED → `CpuSolverRetired`; also inert under terminus since VCF⊆VCT). Run dir
`~/data/moonshot-103/`.
- **The head learns the representation:** `train/vct_loss` **0.60→0.03**, `mask_frac` ~0.9.
- **Self-play does NOT change:** `selfplay/plies_mean` flat **~9-10 for all 1152 epochs** — the #101 attractor.
- **Conclusion:** the supervised defense gradient forms the **percept** but self-play offers no opponent strong
  enough to make the **policy** act on it. The #101 ceiling holds **even with the representation present** —
  rules out "the net can't *see* the blunder." Motivated concentrating on a strong net at a hard fixed position.

**Experiment B — Bruce/idx-2 pivot (wandb `zrjfwny2` from e613, retired e862 ≈ 257 pivot epochs).** When A didn't
dig out, pivoted: warm-start from **Bruce** (`g15_128x10_bigbuf_e588_best.pt`, 128×10 15×15, ep 605) + **layer
the VCT-defense head on** via a new `load_checkpoint(force_aux_vct=True)` splice (`gomoku/model.py:635/673`,
`gomoku/train.py:1417-1422`; core loads strict, fresh `vct_*` params splice in like the swap2 choice head, off =
byte-identical) + **restrict self-play to the idx-2 opening** (`GOMOKU_DROP_OPENERS=0,1,3,4,5,6,7,8`, the
white-to-move "Bruce-Lee board") + VCT-terminus. Idea: concentrate the defense gradient on Bruce's measured
white-defense wound.
- **The head learns again:** `train/vct_loss` **0.52→0.026** (loaded clean, no arch mismatch — the splice worked).
- **The self-play POLICY drifts hard:** `loss/policy` **1.93→2.62** (rising) and `selfplay/plies_mean`
  **collapses 11.6→9.6** — the terminus attractor, reached *even from a champion*. The terminus + narrow-opening
  regime **specializes / erodes** the champion's general play.

**THE EVAL-SATURATION CATCH (the thing most likely to be documented wrong).** Ran the idx-2 verdict eval
(`gomoku.rapfimine.eval_idx2`, **n=48, sims=160, GOMOKU_BOARD_SIZE=15**) on **both** the pivot EMA checkpoint
**and frozen Bruce**, identical settings:
- Pivot EMA: **0/48** (black 0/24, white 0/24).
- **Frozen Bruce (identical settings): ALSO 0/48** (black 0/24, white 0/24).
So the eval at sims=160 is **SATURATED — Rapfi crushes both nets; it does NOT discriminate**, and therefore does
**NOT** show the pivot degraded Bruce. Do **NOT** write "the pivot degraded the champion black 42%→0" — frozen
Bruce also scores 0/48 here. The wiki's "Bruce black ~42% / white 0/12" is a **different eval config**
(stronger-net / higher think-time, white-defense-plan §1B.2, n=24 vs Rapfi 5s/move), not comparable. **The real
evidence of "it fell apart" is the self-play policy drift** (`loss/policy` up, `plies` collapsed), NOT the Rapfi
eval. A clean strength-delta (pivot vs frozen Bruce) would need a **higher-sim or direct-H2H** eval — **not run**
(experiment abandoned at the pivot).

**Verdict.** Across both experiments the recurring lesson: **the VCT-defense aux head reliably learns the
defensive REPRESENTATION, but nothing so far makes the POLICY act on it** — not from-scratch self-play (no
opponent to punish weak defense) and not a frozen-champion warm-start (terminus/narrow regime just specializes
the policy). **"Frankenstein + aux head" is not the recipe** (Jason). The head is a working **sensor** with **no
actuator yet**. What-to-try-next (open directions): target the POLICY directly — the escape-search as a
defensive *policy* target (mask/penalize blunder moves, cf. #43), the defense head at MCTS *inference* time to
prune blunders, or a curriculum/opponent that actually forces defense. Cross-ref #100/#101 (the structural
self-play defensive ceiling this head was meant to break), idea-pile #11.

## 2026-07-01 (issue #107) — the sound world: oracle veto + defender terminus + line planes. The attractor is gone.

**Thesis** (from #100/#101/#103's "sensor with no actuator"): every prior VCT injection edited TARGETS
off-policy; none changed the BEHAVIOR distribution the net distills. The fix: put the oracle in the
ENVIRONMENT — (1) `--oracle-veto`: per ply, the bulk escape-solve MASKS proven-blunder moves out of
both the played move and the recorded policy target (on-policy by construction); all-moves-lose ends
the game as a DEFENDER terminus (z=−1, mirror of #98); with `--vct-terminus` both ends of every game
are oracle-sound — the twin can never hand over a VCT, so the missing punisher exists from epoch 0.
(2) `--line-planes`: 8 in-forward line-potential input channels (per-cell × 4-dir × {me,opp} max
live-5-window count /4) — double threats become LOCAL reads (the claw wound). Both byte-identical-off.
Code merged `aa91a34`; cell `sound-world` (clone of `moonshot` minus aux-head levers); NO
--record-vct / --aux-vct-weight (the #103 sensor had no reader; the veto IS the actuator).

**Run:** wandb `zeed2xw5`, from-scratch 9×9 small (345,885 params, 25-ch stem), run base
`~/data/sound-world-107/`. Smoke (random net, 8 games): **mean 48 plies vs the ~9-10 attractor**,
two 81-ply draws, 6 defender termini, 24/24 recorded targets zero-mass on proven blunders.

**Result — the #101 attractor is structurally gone.** plies_mean lived in the mid-20s→high-50s for
the entire first ~1240 epochs (monitor watch, 6 cycles), never trending to 9-10. pl fell
**4.38→1.34** by e1256 (vs #101: 4.38→2.17 in 2,700 epochs), vl stable ~0.06. Buffer full (1.5M) by
~e600. ms/game warmed 2200→~830-1040 (sharper play compresses games → oracle cost falls with skill).
Ladder while it ran: elo 389→~1120, la:2 0%→75%, heuristic 0%→25%, strongly DRAWISH shape.

**Evals @ e1239** (arena `gomoku-arena`, sims=100, 40 games each, bare net): la:4 **3W-0L-37D**;
la:2 0W-0L-40D; heuristic 5W-3L-32D; **old derby champion (peak.pt): 0W-0L-40D** — the matchup where
the #100 terminus-only net went **0/120** is now UNLOSEABLE at 5h from scratch. Finisher-hybrid
(`vct_finish=50`, legacy match path — the batched arena silently drops `vct_finish`, filed **#109**):
vs heuristic **14W-0L-6D (85%)** vs bare 52.5% — the net REACHES winning positions and doesn't cash
them (trained where such positions never arise); the oracle converts. vs champ still 0-0-20: no cap50
VCT ever exists between two sound players. **Mounting evidence 9×9 freestyle is a practical draw**
(Jason's wall intuition) — the sound world converges toward the game's truth instead of blunder-wins.

**Accidental causal ablation (perf-scout `--oracle-veto-max-cands 24` A/B):** capping veto breadth at
9×9 → leaked blunders get played → **games collapse to ~11 plies — the attractor returns**. Cap the
veto, resurrect the disease: the veto is confirmed as THE anti-attractor mechanism, not a bystander.

**Perf (perf-scout, receipts in wiki/topics/mcts-perf-ceiling.md, merged `ec69e1b`):** gen wall =
oracle 75% / net evals 23% / **native MCTS tree 0.6% (exonerated — "ton of headroom" refuted)**.
Solver cost = calls × ~44ms tail-grind; **width is FREE** (43.8ms @151 boards ≈ 44.3ms @48) — so:
merged per-ply solve (bit-identical, sha256 receipt) 1.07×; `--oracle-overlap` (solve under the MPS
wave) **1.18×, now ON in the cell**; null-board precheck REFUTED at 9×9 (61% fewer boards, +17 calls
→ slower; default OFF, big-board re-measure). Future levers, leverage-ranked: cross-worker shared
solve (width-free ⇒ aggregate oracle ÷4), kernel tail pass (0.9ms/node/thread), cap50→25 recall
study, fp16 eval (1.7×/position, needs TQ canary).

**State:** resumed e1239→ on merged+overlap code (buffer intact, same wandb timeline), verified
healthy through e1275 — including surviving its worktree being janitor-reclaimed mid-run (#111; all
data paths live in ~/data, processes on open handles). Oracle proven board-size-parametric (salvaged
`9863cce`→`fa0dac2`, tests for 11/13). **Next:** let 9×9 run; graduation 11→13 is the attractive
frontier (walls stop dominating; needs the ÷4 solver lever for veto cost); product shape = net +
finisher hybrid (#109 unblocks arena-speed hybrid evals); verdict evals stay H2H + white-column,
never internal elo.

## 2026-07-01 (issue #107, dated correction) — the defender-terminus uniform-pi wound: white collapse at e1982, root-caused and fixed

**Symptom.** 30-min re-eval after the resume: e1982 vs old derby champ **0W-20L-20D — black 0/0/20,
white 0W-20L-0D** (ALL white games lost); heuristic 0W-12L-28D. At e1239 the same matchups were
all-draws / near-clean. Self-metrics (pl falling, plies healthy) saw NOTHING — the #100 lesson again.

**Ruled out, in order:** (1) the new #110 eval code — champ vs heuristic 40W-0L-0D, eval sane; (2) the
perf-scout merge / `--oracle-overlap` race — poison detector (re-solve every recorded position at full
breadth, assert recorded pi has zero mass on proven blunders) found **5/309 violations IDENTICALLY**
with overlap ON, overlap OFF, and on pre-merge code `fa0dac2` at the same seed. **Scout exonerated;
the wound shipped with the launch code.**

**Root cause (classified violators):** all 5 = side=1(white), last-example-of-game, z=−1,
**uniform-over-legal pi at a DEFENDER TERMINUS** — the launch design recorded doomed positions with
pi = uniform over ~70 legal cells ("no move is better; z does the teaching"). At scale that IS the
teaching: by e1982, black forces a proven all-moves-lose position by ply 9-15 in ~5/8 self-play games,
so white's most common late-training examples were pure policy noise exactly at the sharpest
positions → white's policy degraded toward uniform → 20/20 white losses. Delayed onset explained:
early training rarely reached defender termini; the sharper black got, the higher the poison dose.
(Launch-day smoke missed it: the spot-check sampled only early, threat-free plies.)

**Fix (merged with this entry):** `_oracle_veto_partition` records NO example for the doomed
position — the game still ends (z propagates via mate-distance discount; the trap-completing black
move with real MCTS pi becomes the final example). The poison-detector invariant is now STRICT: no
recorded example may carry blunder mass. Detector: `scratchpad/poison_check.py` pattern — re-solve
recorded positions, assert zero blunder mass; run it on any future gen-semantics change.

**Also observed:** black forcing proven wins by ply 9-15 under near-sound defense is evidence 9×9
freestyle is a fast BLACK WIN within cap50 horizons (not a draw) — the walls do not save white at 9×9.

**Decision:** restart from scratch on fixed code (run dir `sound-world-107b`), keeping the poisoned
run's artifacts intact as evidence. Cheap (~2.5h to e2000) and gives clean attribution vs a
buffer-wash resume.

## 2026-07-02 (issue #107, closing entry) — 107b validates the fix; the 9×9 sound-world chapter closes with a carry-forward recipe

**107b eval @ e1368** (fresh run on fixed code; same battery as run A's e1239, arena sims=100 n=40,
finisher via legacy path):
| matchup | 107b e1368 | run A e1239 (pre-collapse) |
|---|---|---|
| old derby champ, bare | **0W-0L-40D, color-symmetric (20D/20D)** | 0-0-40 |
| heuristic, bare | 0W-0L-40D | 5W-3L-32D |
| la:2, bare | 0-0-40 | 0-0-40 |
| la:4, bare | 0W-5L-35D (all 5 as WHITE) | 3W-0L-37D |
| heuristic, finisher(cap50) | **18W-0L-2D (95%)** | 14W-0L-6D (85%) |
| la:4, finisher | 0W-1L-19D | — |
The uniform-pi wound stayed closed (champ column clean + symmetric where run A later collapsed
20/20 as white); the white-vs-la:4 softness (5/20) is the open question the run didn't live long
enough to settle (stopped e1540; the e2000 bet — white holds champ + finisher ≥70% heuristic —
retires UNSETTLED, leaning held). la:4's brute 4-ply tactics still poke white occasionally.

**Chapter verdict (Jason, 2026-07-02): "9×9 has taught us what it can."** What it taught, in one
entry: (1) the actuator belongs in the ENVIRONMENT (oracle veto at gen), not the loss — one day of
the sound world beat weeks of target-side injection; (2) the veto is causally THE anti-attractor
mechanism (K-cap ablation resurrects the 9-ply attractor); (3) never hand the policy head a
"harmless" degenerate target — the uniform-pi shrug scaled into white's collapse precisely because
the experiment SUCCEEDED (dose ∝ black's skill); (4) self-metrics cannot see self-play diseases —
H2H + color columns only (third confirmation); (5) 9×9 freestyle within cap50 is a fast BLACK WIN
(proven all-moves-lose by ply 9-15 vs sound defense) — the walls cap trap complexity, they don't
save white; two sound players draw; (6) the product shape is net + oracle finisher (95% vs
heuristic where bare draws) — bare-net drawishness is division of labor, not weakness.

**Carry-forward recipe → 13×13** (wiki/topics/sound-world-recipe.md): cell `sound-world` = terminus
+ veto + `--oracle-overlap` + line-planes + no-example-at-defender-terminus, gate on
`gen_poison_check` (scripts/) before trusting any gen-semantics change. Perf prerequisite for 13×13:
the cross-worker shared oracle solve (÷4, width-is-free law) — filed. Runs preserved:
`~/data/sound-world-107` (poisoned, evidence), `~/data/sound-world-107b` (clean, resumable e1540,
wandb zeed2xw5→107b run).

## 2026-07-02 (issue #113, 13×13 sound-world graduation) — SLICE 1: offense transfers, white-defense collapses (undertraining, NOT poison); wandb 8rp0gjpm

**Setup:** carry the validated #107 sound-world recipe up to 13×13. WARM-STARTED from
9×9 107b e1540 — the conv tower (75 params, the threat-shape features) transferred; the
board-shaped FC heads (`policy_fc`, `value_fc1`) are flattened-board and were REINIT fresh
for 13×13 (seed built offline: fresh 13×13 net + shape-matching tower copy → strict-loadable
via the production board-size guard). Run dir `~/data/sound-world-13`, cell `sound-world`
unchanged (full-breadth veto + vct-terminus + oracle-overlap + line-planes), 40-min slices,
resume-latest cadence. Fresh wandb run 8rp0gjpm (lineage cleaned — no 9×9 run-id inherited).

**Slice 1 (40 min → e801):** `pl` 3.36→2.18, `vl` 0.047 (value transferred well from tower),
buffer full 1.5M. `selfplay/plies` slid 27–30 → stabilized ~14–17 (did NOT crash to 9). Warm
tower = strong attacker → cap50 VCT-terminus fires earlier as policy sharpens.

**Eval @ e801 (batched arena, EMA worker_weights.pt, n=40/matchup, sims=100, 13×13):**
| matchup | black (w-l-d) | white (w-l-d) |
|---|---|---|
| bare vs heuristic | 0-8-12 | **0-20-0** |
| bare vs lookahead:4 | 6-10-4 | **0-18-2** |
| finisher(50) vs heuristic | **12**-8-0 | **0-20-0** |
| finisher(50) vs lookahead:4 | 11-9-0 | 2-17-1 |
Offense is REAL: the cap50 finisher takes black from 0 wins (bare) to 11–12. WHITE is 0/20
in every config — total defensive collapse, even vs the weak heuristic. (Note: no 13×13 champ
opponent exists yet — a 9×9 net can't play 13×13; heuristic/lookahead are the board-agnostic
rulers.)

**Poison guardrail: 0/612 positions with blunder mass (clean).** So the white collapse is NOT
the #107 uniform-pi wound — the fix holds at 13×13. Poison-gen output corroborates the eval:
31/32 self-play games end decisive (outcome 1.0) by ~ply 13 → black forces a fast VCT, white
almost never holds in self-play.

**Interpretation (working, to falsify over next slices):** UNDERTRAINING of the reinit'd
defense heads, not a defect. The warm tower gives offense for free (transfers) but 13×13 white
defense is a policy/value-head skill that must be relearned from fresh init, and 735 epochs on
`pl`=2.18 is early (9×9 sound world defended only near `pl`~1.3 / ~1300 epochs). WATCH: does
white recover as `pl` converges? Secondary concern — plies FELL here (30→14) whereas the 9×9
sound world plies ROSE as defense developed; if the warm-started aggressive tower is actively
suppressing defense development, from-scratch may be the cleaner path. ESCALATION LINE: if by
~slice 4 (`pl` < 1.6) white is still <10% vs heuristic, flag Jason + recommend a from-scratch
13×13 control. Decision: CONTINUE (poison clean, offense transferring, too early to judge).

## 2026-07-02 (issue #113, cont.) — SLICE 2 confirms warm-start ATTACK-COLLAPSE; pivoting the loop to a from-scratch 13×13 control

**Slice 2 @ e1500** (wandb 8rp0gjpm): pl 2.18→2.01 (PLATEAUING, not marching to 1.3), vl 0.032,
plies flat ~14. Eval (EMA, n=40, 13×13):
| config | black (w-l-d) | white (w-l-d) |
|---|---|---|
| bare vs heuristic | 7-7-6 (↑ from slice-1 0-8-12) | **0-20-0** |
| finisher vs heuristic | **20-0-0** | **0-20-0** |
| bare vs lookahead:4 | 4-14-2 | 0-18-2 |
Poison CLEAN (0/414). Poison-gen: 32/32 self-play games decisive by ply 9–13, several at the
9-ply FLOOR (fastest possible five).

**Diagnosis (confirmed over 2 slices):** attack-collapse caused by the aggressive warm-start.
Black offense → perfect (20-0 finisher); white defense → perfect ZERO (0/20 every config, no
movement across 2 slices, pl plateaued ~2.0). MECHANISM: the warm 9×9 tower forces black VCT
wins by ply 9–13 in self-play, so white is ALWAYS already-lost when threats appear → the veto
masks all white's moves → defender terminus → **white sharp-defense examples never enter the
buffer**. So white cannot learn defense at any slice count; the slice-4/pl<1.6 gate is moot
because pl won't reach 1.6 (no gradient left — offense saturated, white starved). Contrast 9×9
sound world: plies ROSE (20s→50s), white could draw. Here plies FELL to the floor.

**Decision — PIVOT to a from-scratch 13×13 control** (deviating from the stated slice-4 gate
deliberately; the mechanism makes more warm-start slices ~zero-information). From-scratch is the
VALIDATED sound-world recipe and the documented #113 alternative; it is the control that
DISTINGUISHES the two hypotheses: (H1) warm-start broke it → from-scratch plies RISE like 9×9,
white learns; (H2) the 13×13 veto itself is broken/insufficient at 169 cells → from-scratch
ALSO collapses to the 9-ply floor. Either result is decisive. Warm-start run PRESERVED intact
(~/data/sound-world-13, HF jasonyandell/gomoku-13x13, wandb 8rp0gjpm) as evidence; Jason can
resume it if he disagrees with the pivot. From-scratch run dir: ~/data/sound-world-13-scratch,
fresh wandb. Escalate to Jason with the H1/H2 verdict once from-scratch has ~3–4 slices.

## 2026-07-02 (issue #113, cont.) — FROM-SCRATCH control slice 1: white ALSO 0/20 at matched epoch (H1/H2 undecided; leaning watch H2); wandb uublz536

**From-scratch slice 1 @ e814** (fresh 13×13 sound-world, no resume; wandb uublz536): pl 1.83
(converging FASTER than warm-start's 2.18 @ e801 — no transferred-bias fight), vl 0.053, plies
~16-18. Eval (EMA, n=40, 13×13):
| config | black (w-l-d) | white (w-l-d) |
|---|---|---|
| bare vs heuristic | 0-11-9 | **0-20-0** |
| finisher vs heuristic | 14-6-0 | **0-20-0** |
Poison CLEAN (0/411). Poison-gen plies cluster 9–13 with MANY at the 9-floor (like warm-start).

**Read:** at MATCHED slice-1 epoch, from-scratch ≈ warm-start on the white gate — both white 0/20.
So slice 1 does NOT separate H1 (warm-start was the seed problem) from H2 (13×13 sound-world can't
teach white defense within cap50 regardless of seed). The warm-start/from-scratch DIVERGENCE (if
any) must show in slices 2–4: does from-scratch white climb off 0 + plies RISE (H1), or stay stuck
+ plies floored (H2)? Hopeful-for-H1 signs: from-scratch pl lower/faster (1.83 vs 2.18), plies not
yet floored (~17 vs warm-start's 14→9). Worrying-for-H2 sign: from-scratch self-play ALSO forces
fast black wins (poison plies many 9s) — the same white-starvation mechanism could bite from scratch
too. NB vs heuristic (a WEAK non-forcing opponent), white 0/20 is NET weakness not game-unfairness
(a sound white should beat heuristic-as-black), so white defense IS learnable in principle — the
question is whether THIS recipe teaches it at 13×13. Continuing from-scratch; escalate to Jason with
the H1/H2 verdict at scratch-slice ~3–4. Now launching scratch-slice 2.

## 2026-07-02 (issue #113, cont.) — FROM-SCRATCH slice 2: white STILL 0/20 at BETTER pl than warm-start → H2 strengthening (structural, not seed)

**From-scratch slice 2 @ e1531** (wandb uublz536): pl 1.83→1.74 (better than warm-start's 2.0 @
matched e1500), vl 0.039, plies still flat ~14. Eval (EMA, n=40):
| config | black (w-l-d) | white (w-l-d) |
|---|---|---|
| bare vs heuristic | 4-13-3 (↑ from 0-11-9) | **0-20-0** |
| finisher vs heuristic | 15-5-0 | **0-20-0** |
Poison CLEAN (0/410).

**KEY COMPARISON (matched ~e1500–1530):** warm-start = white 0/20, plies ~14, pl 2.0; from-scratch
= white 0/20, plies ~14, pl 1.74. From-scratch has BETTER policy convergence but IDENTICAL white
result and IDENTICAL flat plies — the faster pl is buying only OFFENSE (black bare 0→4, finisher
15), zero defense. Two independent runs (warm + scratch) now both show white EXACTLY 0/20 across 2
slices each with flat plies → the white-defense failure is looking STRUCTURAL to the sound-world
recipe at 13×13, NOT a warm-start seed artifact. Shared mechanism: black forces fast VCT wins (plies
~14, poison plies cluster at 9) → white always already-lost in self-play → veto masks white's moves →
white sharp-defense examples never enter the buffer → white can't learn to defend at any pl.
Distinguishes from 9×9 where the veto made plies RISE (20s→50s) and white could draw.

**Plan:** running scratch-slice 3 — pl should cross under ~1.6 toward the 9×9 white-defense zone
(~1.3); a genuine last test of whether white emerges late. If white is STILL ~0 at pl<1.6 with flat
plies → H2 CONFIRMED, escalate to Jason with the recipe-change recommendation. This is shaping up to
be the night's headline learning: the 9×9 sound-world recipe does NOT transfer white defense to 13×13
under a fixed cap50 terminus — the bigger board lets black force wins before white ever learns to hold.

## 2026-07-02 (issue #113, morning) — Comprehensive 13×13 eval + perf isolation: sound-world nets are ATTACK-ONLY specialists (lose to everything that defends); the OLD full-game net is stronger; perf gain was NET SIZE, not the gen path

Ran on Jason's request after the overnight loop. **Jason's stated predictions (PVE bet, logged before results):** eval — (P1) both nets NOT always lose vs rapfi@50ms; (P2) both win-or-draw vs simple heuristics; (P3) warm-start wins H2H vs from-scratch. perf — (P4) much faster gen rate; (P5) similar train rate.

**EVAL MATRIX** (n=40 each, EMA worker_weights, sims=100, 13×13; our two nets FINISHER-armed vct_finish=50):
| # | matchup | result (A w-l-d) | black | white |
|---|---|---|---|---|
| 1 | from-scratch+fin vs **rapfi@50ms** | 0-40-0 (0%) | 0/20 | 0/20 |
| 2 | warm-start+fin vs **rapfi@50ms** | 0-40-0 (0%) | 0/20 | 0/20 |
| 3 | from-scratch+fin vs warm-start+fin [H2H] | 20-20 (50%) | 20-0 | 0-20 |
| 4 | from-scratch+fin vs **OLD 128×10** (bare) | 0-40-0 (0%) | 0/20 | 0/20 |
| 5 | warm-start+fin vs **OLD 128×10** (bare) | 0-40-0 (0%) | 0/20 | 0/20 |
| 6 | OLD 128×10 (bare) vs **rapfi@50ms** [anchor] | 3-37 (7.5%) | 3/20 | 0/20 |
OLD = G-ladder-13-board13 e424 (large 128×10, swap2/full-game recipe, NO cap50 terminus, self-play plies 50–64). Earlier smoke: OLD went 2-0 as WHITE vs heuristic (our terminus nets are 0/20 white).

**PREDICTIONS SCORED:**
- P1 (both not-always-lose vs rapfi) → **REFUTED.** Both lose 40/40 = 100% vs rapfi, even finisher-armed.
- P2 (win-or-draw vs heuristic) → **HALF.** Win as black (finisher 20-0/15-5), but LOSE as white 0/20 — not always win-or-draw.
- P3 (warm-start wins H2H) → **REFUTED.** Exactly 50/50, purely color-determined (whoever is BLACK wins by forcing a VCT; both nets are behaviorally identical attack specialists — no skill delta).
- P4 (much faster gen rate) → **REFUTED** (see perf below).
- P5 (similar train rate) → **CONFIRMED.**

**THE FINDING:** the sound-world/cap50-terminus recipe produces an ATTACK-ONLY SPECIALIST at 13×13. The finisher only fires when a forced VCT exists; a defending opponent (rapfi, or even our own OLD net) never hands one over → the finisher never fires → the bare attack-only net plays → 0%. Our nets only "win" when the opponent lets them force a fast VCT (weak heuristic-as-black, or their own twin when black). **The OLD "we-never-focused-on-it" 128×10 net BEATS both sound-world nets 40-0 AND scores 7.5% vs rapfi where ours score 0%** — because it trained on FULL games and learned to DEFEND. This is the overnight H2 white-defense wound taken to its conclusion: no defense ⇒ lose to everything that defends. Net+finisher is NOT a product at 13×13; it's a black-only party trick.

**PERF ISOLATION** (bench_gen_refill, SAME small 64×4 net, 13×13, sims=100 — net size held constant):
| config | games/min | aug_pos/s | oracle_s / wall_s |
|---|---|---|---|
| oracle ON, lockstep (concurrent=0) | 215.6 | 449 | 16.2 / 17.8 (91%) |
| oracle ON, streaming (concurrent=64→256) | 216.4 | 451 | 16.2 / 17.7 |
| oracle OFF, lockstep | 638 | 2497 | 0 |
| oracle OFF, streaming | 644 | 2520 | 0 |
- Streaming ≈ lockstep in a SINGLE process, oracle-on OR off. #112's 3.4× win was 8-proc FLEET → 1 wide proc, a DIFFERENT comparison; the refill loop itself adds ~0 single-process throughput here.
- The **VCT oracle veto is the gen bottleneck at 13×13**: 91% of wall, cuts throughput ~3× (640→216 games/min). That's the #114-kernel domain, not the gen loop.
- **The overnight "14× faster epoch (45s→3.1s)" was ~ENTIRELY NET SIZE** (large 128×10 train ~44s → small 64×4 train ~2.7s); train time is net-size-bound (P5 confirmed). The "2 results overnight" was bought by the SMALL NET + unattended looping, NOT a gen-path perf overhaul.

**Takeaways:** (1) if we want a STRONG 13×13 net, the full-game (defense-learning) recipe beats the terminus recipe — the OLD net is the better lineage; (2) the sound-world recipe needs a real modification to teach defense at 13×13 (raise terminus budget / white curriculum) before it's worth more GPU; (3) perf: the lever for faster sound-world gen at 13×13 is the VCT solver (oracle = 91% of gen), not the gen loop.

## 2026-07-03 (issue #116, RAILS-V0 LAUNCH) — the #113 white-starvation cure: DROP the terminus, PLAY ON, keep both sides sound (veto + attacker-preserve), 15×15 idx-2. Predictions logged BEFORE results.

**Thesis (direct follow-up to the #113 13×13 negative result).** #113 proved the cap-terminus recipe is a STRUCTURAL trap: black forces a fast VCT (self-play plies floored ~9–14), so white is already-lost when threats appear → the veto masks all white's moves → the defender terminus fires → **white's sharp-defense examples never enter the buffer** → white 0/20 everywhere, two independent runs. The cure (sound-world-recipe.md open-directions #1+#2): **remove the terminus and PLAY ON to a natural five** (z = actual result) so black's forced win no longer ejects white — white stays ON-POLICY everywhere and generates the sharp-defense examples the terminus never let it see. Keep the defender `--oracle-veto` (both sides sound) and add a NEW **`--attacker-preserve`** mask (idea #2): when the mover has a proven VCT, restrict the recorded+played policy to the winmask (all winning first moves) so the net learns to CLOSE on-policy — folding the oracle finisher INTO the net (9×9: bare net attacks but draws, finisher converts 95%).

**New code (merged to main, #116, byte-identical-off).** `--attacker-preserve` worker flag: per ply, one bulk complete-mode `solve_vct_mega_bb` over the mover-to-move roots → winmask; `_preserve_policy` masks pi to the winning first moves at the same single masking point as the veto, BEFORE pi is recorded or sampled. Gate on `win` alone (0-FP kernel ⇒ a capped partial winmask is sound, only ever over-restricts). Unit-tested + real-MLX smoke at board 15 (`scripts/rails_smoke_check.py`): plies ~45 (vs terminus-era ~14 floor), balanced 4W/4B outcomes from a random net, **veto poison 0/338, 0 preserve leaks** — the #107 guardrail passes at the rails live config.

**Run.** Cell `rails-v0` = FRESH 15×15 small (64×4) net + `--line-planes`, `--oracle-veto` + `--oracle-overlap` + `--attacker-preserve`, **NO `--vct-terminus`** (play-on), idx-2 opener ONLY (`GOMOKU_DROP_OPENERS=0,1,3,4,5,6,7,8`, the Bruce-Lee board, white-to-move, fairest measured 15×15 opening), 1M packed ring + `--buffer-recency-frac 0.5` (mandatory — a 1M ring without recency goes stationary, §13), n_workers=1, sims=100, value-discount 0.98, cap25 shared veto/preserve budget. Run base `/Users/jason/data`, wandb project `gomoku`, run id **`vraf0b6e`** (run name `9x9-sweep-rails-v0` — the "9x9" prefix is a run_sweep naming default; the board IS 15×15 via `GOMOKU_BOARD_SIZE=15`). Launched from worktree `/Users/jason/code/gomoku-attacker-preserve` (own venv + native ext; run DATA in ~/data so a janitor reclaim can't touch it, #111).

**PRE-STATED PREDICTIONS (logged before any result, Jason's PVE discipline):**
- **P1:** `selfplay/plies_mean` RISES vs the terminus-era runs (#113 floored ~9–14) — games play out; smoke already shows ~45. Watch it stay high, not collapse to the floor.
- **P2:** White's self-play column comes ALIVE — white takes a real share of decisive self-play games (NOT the 0/20 signature of #113). Morning arena H2H should show white's per-color column climbing off 0.
- **P3:** Bare-net conversion improves over the terminus-era nets (attacker-preserve teaches closing on-policy) — the bare net should start cashing winning positions instead of needing the oracle finisher bolted on.
- **Risk being watched:** the buffer tilting toward low-entropy forced-tail positions (attacker-preserve one-hot-ish targets on long forcing tails). Death-tells: `loss/value` → <0.08 (value poisoning), `plies` collapsing to the floor (attack-collapse returned), or `train/sample_reuse_ratio` <1 (gen flood) / ≫4 (gen starved) — adjust n_workers per the §12 buffer-balance knob.

Overnight monitoring: wandb `plies_mean`, `loss/policy`, `loss/value`, white-share, `sample_reuse_ratio`, s/epoch (Jason predicts 8–20s trainer-bound at the small net). Gate on H2H + per-color columns in the morning, never internal loss. Results to follow in append-only entries below.

## 2026-07-03 (issue #116, RAILS-V0 EARLY READ + FRAMING CORRECTION) — the "overnight" never happened: run is 8 MINUTES old at check-in; bets scored provisionally from self-play, heavy eval DEFERRED

**Framing correction (authoritative, from the W&B server timestamp + wandb debug log, not the wall-clock narrative).** A replacement driver session was briefed to do an "overnight rails-v0 morning close-out." That premise is **factually wrong.** W&B run `vraf0b6e` `created_at = 2026-07-03T06:56:44Z = 01:56 CDT`; the trainer/worker pids (40240/40275) both have process-start 01:56 AM; the wandb config shows `resume: None`. At check-in the machine clock read **02:04 CDT** — the run is **~8 minutes / ~235 epochs old**, and 01:56 is the **ORIGINAL launch, NOT a watchdog relaunch** (fresh wandb init, no resume). There was no overnight trajectory to reconstruct. The predecessor ("rails-15") logged the launch entry + the "epoch ~120" early signal and was stopped ~8 min later. Recording this so no future session mistakes this run for a matured overnight campaign.

**Run health @ e235 (alive, healthy, progressing ~3.3 s/epoch):**
| metric | value | read |
|---|---|---|
| `selfplay/plies_mean` | 49.5 (e105) → **32** (e230–234), p10 17 / p50 29 / p90 50 | above the #113 ~9–14 floor; the drop is random-net→sharpening, not collapse. Sparse (3 points). |
| `loss/policy` | 4.99 → **3.39** monotone | converging; still near-random (policy_acc 0.33) — 8 min in. |
| `loss/value` | 0.29 → dipped 0.08 (e120) → settled **~0.16** | NOT poisoned; never stuck <0.08. |
| `train/sample_reuse_ratio` | 7.4 → **8.5** | **HIGH (≫4 = gen-starved, §12).** n_workers=1 trainer outruns gen (`selfplay/new_games=0` most epochs). The one live watch item. |
| `time/epoch_s` | **~3.3** stable | trainer-bound at the small 64×4 net. |
| buffer | 0 → **537k** of the 1M ring; `frac_current 0`, `z_wins 0.503 / z_losses 0.494 / z_draws 0.004` | filling fast (streaming concurrent=256); balanced win/loss. |
| self-play decisive (cumulative) | black **1070** / white **620** / draws 1 → **white share 0.367** | recent batches 16W/48B, 13W/54B. |
| per-side | pl black 3.26 / white 3.60; value-mse black 0.155 / white 0.161 | BOTH colors training — no one-sided starvation. |

**PRE-STATED BETS SCORED (provisional, 8-min read — self-play data only; no GPU eval burned):**
- **Throughput bet → CONFIRMED (down).** Jason 8–10 s/ep, Fable hedged ≤20, launch ~3.5 → **actual ~3.3 s/ep**. Both predictions beaten downward; trainer-bound at the small net as expected.
- **P1 (plies RISE vs the terminus-era ~9–14 floor) → PROVISIONAL PASS, watch.** plies ~32 (p90 50) sits **well above** the #113 floor — games play out, white is not ejected. Caveat: only 3 plies points and a within-run decline from the ~49 early-random peak; if it slides toward the floor as pl drops, revisit (attack-collapse tell).
- **P2 (white column ALIVE, not #113's 0/20) → PROVISIONAL PASS (self-play).** White wins **37%** of decisive self-play games (620/1690) — decisively off the 0/20 signature; both colors show real value/policy gradient. This is the falsifiable P2 signal and it needs **no GPU**. Arena H2H confirmation deferred (below).
- **P3 (bare-net conversion improves) → NOT YET SCOREABLE.** pl 3.4 is a near-random net; conversion is meaningless at 8 min. Score after maturity.

**Eval DEFERRED, deliberately (ML judgment + tenant respect).** The briefed "arena H2H + white-column eval on the latest checkpoint" was NOT run, for three compounding reasons: (1) the checkpoint is **8 minutes / pl 3.4** — below hint-level signal; (2) at **15×15 the heuristic/lookahead ladder saturates at 0.0 win-rate from epoch 0** (`scripts/ladder_eval_15x15.py` docstring), so a "vs-heuristic white column" tests nothing here — the only honest 15×15 yardstick is **Rapfi (CPU)**, against which a pl-3.4 net loses 40/40 for zero signal; (3) the live trainer **owns the GPU** and short noisy evals compete for nothing. The real close-out eval (Rapfi at 15×15 + per-color split) should run once the net matures (trigger ~`loss/policy < 1.6`, cf. #113 where meaningful evals landed at pl ~1.7). Filed as **#117**.

**Live watch item / recommendation.** `sample_reuse_ratio ≈ 8.5` (≫4) says the single self-play worker can't feed the trainer — most epochs record `new_games=0`. Not yet harmful (buffer still filling, losses healthy), but as the 1M ring saturates this risks staleness. Per the §12 buffer-balance knob, if reuse stays ≫4 after the ring fills, add a second worker (or cut `--sgd-steps-per-epoch`). Death-tells all currently NEGATIVE: vl not <0.08, plies not floored, reuse not <1. Continuing to watch light-touch; one relaunch authorized if it dies.

## 2026-07-03 (issue #116/#117, ~e1150–1280, pl≈1.6) — falling-plies investigation + CPU per-color eval: NOT the #113 collapse, white OFF the zero-floor, but competence UNCONFIRMED (#117 stays open)

**Trigger.** Between the launch and ~e1096, `selfplay/plies_mean` drifted down 49→32→24→~20 (approaching, not at, the #113 ~14 floor), prompting the question: healthy sharpening or the #113 white-starvation signature re-emerging? Investigated the discriminator = **white's share of decisive self-play games**, then ran a brief CPU eval (device=cpu, local Rapfi build, zero GPU/Metal contention with the live trainer) to harden the verdict. Net at check: e1149 pl 1.71 (eval checkpoint, snapshotted to /tmp so `keep_last_n 3` couldn't delete it mid-eval); live run at write time e1281 pl 1.56 vl 0.155 reuse 7.0.

**Self-play white-share arc (pulled from wandb `vraf0b6e`, per-batch `selfplay/white_wins`/`black_wins`):**
- e97–171: **0.35–0.50** (healthy, ~43%)
- e230–330: falls 0.25 → 0.05
- e330–900: **long trough ~0.05–0.09** (white 0–5 wins of ~65 games/batch for ~600 epochs)
- e1084–1101: **RECOVERING — 0.12 / 0.18 / 0.11 / 0.15 / 0.18 / 0.22 / 0.08 / 0.265** (last batch 18W/50B)

**Read = healthy sharpening on a black-tilted opening, NOT #113 starvation.** Four structural tells: (1) **the #113 mechanism is structurally absent** — that run starved white via the defender *terminus* ejecting already-lost positions (no example recorded); rails-v0 has **no terminus** (play-on), so white positions are *always* recorded and the pathway cannot fire. (2) **White is still actively training** — per-side `train/policy_ce` black 0.78 vs white 2.64 (falling), value-mse black 0.097 ≈ white 0.107; starvation = a frozen/degenerate white head, not seen. (3) **Plies not floored + draws rising** — buffer `z_draws` climbed 0.4%→**22.9%**; games go the distance, opposite of fast-attack collapse. (4) `z_wins 0.387 ≈ z_losses 0.384`. If this were #113 the white share would pin at ~0 and stay; instead it bottomed and recovered. The falling plies is black converting its idx-2 initiative faster as the net sharpens (pl 4.99→1.56) — idx-2 is "fairest *measured*" but still black-tilted, exactly the case recipe open-directions #3/#4 anticipated.

**CPU per-color eval (n=20 games/tier unless noted, sims=100, 15×15 freestyle, eval ckpt e1149 pl 1.71):**
| opponent | agg W-L-D | our net as BLACK (W-L-D) | our net as WHITE (W-L-D) |
|---|---|---|---|
| **Rapfi @50ms** | 0-20-0 | 0-10-0 | **0-10-0** |
| **Rapfi @200ms** | 0-20-0 | 0-10-0 | **0-10-0** |
| **heuristic** (n=20) | 8-12-0 | 7-3-0 | **1-9-0** |
| **heuristic** (n=40) | 11-26-3 | 9-8-3 | **2-18-0** |

**Interpretation (partial-satisfaction of #117 — issue kept OPEN):**
- **Rapfi is too strong to discriminate at pl 1.71** — our net is 0/40 as *both* colors. This does NOT reproduce the #113 asymmetry (there black was competent while white was zero); here both colors are zero because a ~2625-Elo engine simply crushes a pl-1.71 net. Uninformative about white specifically — the exact "near-random-vs-Rapfi 0/40" caveat pre-stated at launch. So Rapfi neither confirms nor refutes.
- **The #113-comparable probe is vs the heuristic** (the same weak opponent that gave #113 white **0/20** across *two independent* 13×13 runs with zero movement). rails-v0 white scores **2/20 (10%)** vs heuristic — **off the zero-floor**, and consistent across the n=20 (1/10) and n=40 (2/20) samples. Small-n (Wilson 95% on 2/20 ≈ [2.8%, 30%] overlaps 0), so not *statistically* decisive alone — but it converges with the recovering self-play white-share and the absent-terminus argument.
- **BUT white competence is UNCONFIRMED.** White at 2/20 vs a *weak* heuristic is barely off the floor and still far below black (9-8-3 = 60% non-loss). The net is genuinely weak overall this early (black only 60% non-loss vs heuristic; 0/40 vs Rapfi). A black>white asymmetry persists — expected on a black-tilted opening, but it means "white can defend" is not yet demonstrated, only "white is not #113-dead."

**Verdict.** The falling plies is **sharpening, not the #113 starvation collapse** (high confidence: absent mechanism + recovering white-share + white non-zero vs heuristic + white head still training + draws rising). Whether rails-v0 produces a *genuinely competent* white is **not yet answerable** — the net is too weak for Rapfi to probe and only marginally beats the heuristic. **#117 stays OPEN** for the matured-net full eval (target: white per-color vs Rapfi with n≥40 once the net is strong enough that Rapfi discriminates — i.e., when black itself starts scoring >0 vs Rapfi). Recommend NOT intervening in the live run (no collapse signature). Watch-flip conditions (none present): white-share rolling back toward 0 AND plies to ~14 AND white `pl` freezing.

## 2026-07-03 (issue #116, ~e1880, pl≈1.6 — CORRECTION to the entry above) — the e1096 recovery was TRANSIENT; white-share RE-COLLAPSED + vl death-tell tripped. The flip condition is MET. Key learning: **same OUTCOME as #113 via a DIFFERENT pathway (hopeless-tail buffer tilt, NOT terminus-ejection).**

**Dated correction, not a rewrite of the above.** The immediately-preceding entry (~e1150–1280) called the falling plies "healthy sharpening, not #113," resting substantially on the e1084–1101 white-share *recovery* (0.05 trough → 0.11–0.265). **That recovery did NOT hold.** ~600 epochs later the discriminator reversed and two death-tells are now firing. This is exactly why we watch-and-recheck instead of declaring — a transient recovery is not a trend.

**The flip evidence @ ~e1880 (wandb `vraf0b6e`, live run e1881 pl 1.63 vl 0.052 reuse 7.0, ~2.9 s/ep):**
- **white-share RE-COLLAPSED to 0.015** over the window e1856–1880 (12 batches: white 13 / black 857, per-batch white 0–2 of ~70). Back near the #113-like near-zero, far below the 0.15–0.25 recovery band.
- **plies ~17–18** (16.3–19.5), closing on the ~14 floor (a 49→32→24→20→17 monotone slide from launch).
- **`loss/value` = 0.052, flat <0.08 for 15+ logged steps** → the **pre-stated value-poisoning death-tell is TRIPPED** (was 0.155 at e1280).
- **per-side `train/policy_ce`: black 0.45 (very sharp) vs white 2.83 and RISING** (2.64 @ e1280 → 2.83) — the black↔white gap is *widening*; white's head is getting relatively worse, not better. Value-mse black 0.051 ≈ white 0.053.
- **draw-rate → ~0** (buffer `z_draws` 0.23 → 0.19 and falling; recent batches essentially 0 draws) — games are decisive black wins again.

Cluster = white-share re-collapse + plies→floor + vl<0.08 + white-pl rising = the **pre-stated "buffer tilting toward low-entropy forced-tail positions" risk materializing.** The watch-flip condition (white-share → 0 alongside plies → floor) is **MET.**

**KEY LEARNING — same outcome as #113, DIFFERENT pathway (make this prominent).**
- **#113 pathway (terminus-ejection):** the defender *terminus* fired on already-lost white positions and recorded **no example** → white's sharp-defense positions never entered the buffer → white couldn't learn defense. The cure hypothesis (rails-v0) was: drop the terminus, PLAY ON, so white positions are always recorded.
- **rails-v0 pathway (hopeless-tail buffer tilt):** the terminus cure *worked on its own terms* — white positions ARE recorded now. But idx-2 is a **black-tilted opening**, and as the net sharpened black learned to force wins by ply ~17. So the white positions that get recorded are **losses from already-hopeless states**: white learns "everything is lost" (its policy target degrades → white pl rises), and the buffer fills with long low-entropy black-forced-win tails that drive value toward certainty (vl → 0.05, poisoning). **Net outcome is the same as #113 — white cannot learn to defend — reached by a completely different mechanism.** Recording *more* white examples doesn't help if they're all hopeless; the problem migrated from "white examples are missing" to "white examples are all losses."
- **Implication for the recipe line:** removing the terminus (rails idea #1) is *necessary but not sufficient*. On a side-favored opening, on-policy play-on still starves the losing side of *learnable* (contestable) positions. Two levers follow, in order.

**Lever 1 (pre-agreed first knob) — TAIL SUBSAMPLING.** Subsample the long low-entropy forced-win tails (the positions at/after the mover has a proven VCT — we already solve this for attacker-preserve) so the buffer stops being dominated by black-win-tail examples. Rebalances toward the contested early/mid game where white still has agency, and directly attacks BOTH death-tells (value-poisoning from certain-outcome tails, and white-starvation-of-learnable-positions). Being implemented behind a flag (byte-identical off, unit-tested, poison/smoke-extended), staged unmerged — **Jason decides whether to launch it.** NOT abandoning rails.

**Lever 2 (fallback) — FAIRER OPENING.** If subsampling doesn't move white-share, the root cause is the opening being too black-tilted: bolt rails onto a **fair opening** — swap2, or a less black-favored opener than idx-2 (recipe open-directions #3/#4: "rails must be bolted onto FAIR openings if the board is a black win"). A sound white can only learn to hold if the game it's defending is actually holdable.

**Disposition.** Run left ALIVE and untouched (deeper collapse is cheap, clean evidence at 2.9 s/ep; Jason makes the intervention call at wake). No relaunch, no flag change to the live run. Both levers staged/planned for his decision.
