# gomoku

AlphaZero for 9x9 free-style gomoku on Apple Silicon (M-series Mac via PyTorch MPS).

## Setup

```bash
uv sync --extra dev        # creates .venv from the pinned uv.lock
```

Run everything with `uv run <cmd>` — no `source .venv/bin/activate` needed, and it
resolves the right environment from your current directory (this matters when you
work in [git worktrees](wiki/topics/worktree-hygiene.md)).

## Train

```bash
uv run gomoku-train --epochs 100 --games-per-epoch 64 --n-simulations 100 --wandb
```

Checkpoints land in `checkpoints/`. Resume with `--resume checkpoints/latest.pt`.

For long worker runs, throttle full replay-buffer checkpoint writes:

```bash
gomoku-train ... --save-every 1 --save-buffer-every 100 --keep-last-n 3
```

`epochNNNN.pt` snapshots are weights+optimizer only; `latest.pt` embeds the
replay buffer for resume and is the expensive write. This matters most when
testing larger replay-buffer runs.

## Perf Microbench

Before changing a live run based on Activity Monitor, compare wall-clock
throughput with the bounded MCTS bench:

```bash
python scripts/perf_microbench.py --device mps --size small --stem-padding 1 \
  --games 8 --n-simulations 400 --wave-size 64 --max-plies 16 --repeats 3
```

Moderate GPU percent is expected for the current tiny-model MPS path. Score
config changes by seconds, games/sec, and positions/sec, not by the GPU graph
alone. See `wiki/topics/activity-monitor-perf-runbook.md`.

The gen hot path has optional native extensions:

- `gomoku._mcts_native` owns the self-play MCTS arena/search loop when the
  evaluator exposes a plane-batch callback, which `make_torch_evaluator` does.
  Use `GOMOKU_DISABLE_NATIVE_MCTS=1` to force the Python MCTS fallback.
- `gomoku._state_ops_native` backs `gomoku.state_ops` for the remaining Python
  state helpers. Use `GOMOKU_DISABLE_NATIVE_STATE_OPS=1` to force NumPy state
  ops for A/B checks.

## Play

```bash
gomoku-play --checkpoint checkpoints/latest.pt
```

## Layout

```
gomoku/
├── game.py        9x9 free-style rules, terminal detection, symmetries
├── model.py       small ResNet, policy(81) + value(scalar) heads
├── mcts.py        PUCT with batched leaf evaluation
├── self_play.py   parallel game generation
├── train.py       generate→train cycle, replay buffer, W&B
├── eval.py        vs random, vs prior checkpoint
└── cli.py         play the trained model from the terminal
```

Reference: `~/code/mk5-main/forge/zeb/` (Texas 42 AlphaZero pipeline).
