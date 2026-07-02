# Publish / play a model — the workflow

The pinned "let me see it / ship it" page. Two things live here: **playing** a
checkpoint locally, and **publishing** it to HuggingFace.

> **← [index](index.md)** · workflows: [Train a model](train-a-model.md) · [Eval a model](eval-a-model.md)

## Play it locally

```bash
uv run gomoku-play --checkpoint checkpoints/latest.pt   # CLI
uv run gomoku-web                                        # FastAPI UI around a checkpoint
```

Deep page: [playing-the-model.md](topics/playing-the-model.md) (web UI / live SPA).

## Publish to HuggingFace

Everything lives in one module — **`gomoku/hf.py`** — and the lab calls into it.
The model home is the public model repo **`jasonyandell/gomoku-9x9`**.

### Manual publish (the human command)

```bash
python -m gomoku.hf push --checkpoint checkpoints/latest.pt
#   optional: --repo jasonyandell/gomoku-9x9   --name model.pt
```

This **slims** the checkpoint first (`slim_checkpoint` — keeps `model_state_dict`
+ `model_config`, **drops optimizer state + replay buffer**, ~61 MB → ~4 MB),
then uploads 4 files in one commit (`model.pt`, `config.json`,
`training_state.json`, `README.md` card) to the **`main`** branch.

### Manual pull / warm-start

```python
from huggingface_hub import hf_hub_download
from gomoku.model import load_checkpoint
p = hf_hub_download("jasonyandell/gomoku-9x9", "model.pt")        # or revision="champion"
net = load_checkpoint(p)
```
Training warm-start takes `base: "hf://owner/repo@rev"` — the HF copy has no
buffer, so a resume pays a one-time buffer re-warm.

### How the autolab publishes (the flywheel)

Each ~1 h slice → eval → **`push_slice`** creates a per-slice HF **branch =
revision** (named `lane-rowid`) → the arena gates it vs the champion → on PROMOTE
the **`champion` tag** is moved to the winning revision (`arena.py` delete+recreate
tag). So `main` = last manual push; revisions = slice history; the `champion` tag
= the current best. Deep page: [autolab-architecture.md](topics/autolab-architecture.md).

### Auth & gotchas

- **Auth is ambient** huggingface_hub — `hf auth login` once, or set `HF_TOKEN`.
  There is **no HF keychain path** (unlike W&B, which uses the `wandb-api-key`
  keychain entry). Public pulls work token-free.
- **Never publish the raw training checkpoint** — always via `slim_checkpoint`
  (the raw file carries the buffer/optimizer, useless for inference).
- **`gomoku.hf push` overwrites `main`** — it does *not* create a revision or move
  the champion tag (those are autolab-only, via `push_slice`).
- **Crossing a board-size era resets the `champion` tag** (a 9×9 net can't gate a
  15×15 candidate — shape mismatch); old revisions remain as HF history.
