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
