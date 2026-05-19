# Gomoku AlphaZero — Training Wiki

Live log of training runs, performance characteristics, experiments, and outcomes.
Append-only style — don't rewrite history, add a new entry at the bottom.

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

- **Train:gen ratio.** kze1lcti = 1:1 (400 SGD steps per 64 games ≈ 6
  steps/game). oo53qzvf is K=2 SGD-per-game by design = 2 steps/game.
  Lower SGD-per-game means weights move slower per data point → buffer
  rolls over with a flatter learning slope (gentler opponent gradient).
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

