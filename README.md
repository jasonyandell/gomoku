# gomoku

AlphaZero for 9x9 free-style gomoku on Apple Silicon (M-series Mac via PyTorch MPS).

## Setup

```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

## Train

```bash
gomoku-train --epochs 100 --games-per-epoch 64 --n-simulations 100 --wandb
```

Checkpoints land in `checkpoints/`. Resume with `--resume checkpoints/latest.pt`.

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
