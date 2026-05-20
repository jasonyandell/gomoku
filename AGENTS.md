# Gomoku Agent Orientation

Start with the wiki.

`TRAINING_WIKI.md` is the living source of truth for this repo. It records the
training story, failed hypotheses, W&B run IDs, checkpoint meanings,
performance findings, and current interpretation of the learning dynamics.
Read it before making claims about why training is working or failing.

The wiki is intentionally append-only. Do not rewrite old conclusions to make
the story cleaner. If new evidence contradicts an older note, add a new dated
entry that explains the correction and points to the evidence.

## Shape Of The Repo

- `gomoku/` contains the Python training stack:
  - `game.py` is the canonical 9x9 gomoku state, terminal detection, history
    planes, and D4 augmentation.
  - `model.py` defines the residual policy/value network and checkpoint format.
  - `mcts.py` contains PUCT search, batched evaluation, and wave-batched search.
  - `self_play.py` generates training records.
  - `selfplay_worker.py` is the distributed worker side of file-based self-play.
  - `train.py` is the trainer, replay-buffer ingest, W&B logging, checkpointing,
    and optional distributed handoff.
  - `baselines.py`, `eval.py`, and `match.py` define fixed opponents and match
    probes used to judge progress.
- `TRAINING_WIKI.md` is the training lab notebook and should be updated when an
  experiment changes the working theory.
- `README.md` is the quick user-facing setup and command surface.
- `tests/` holds the Python smoke and correctness tests.
- `scripts/` holds utility scripts such as export helpers.
- `web/` and `app/` are playable/browser-facing surfaces around checkpoints.
- `checkpoints*/`, `sweep_logs/`, and `wandb/` are artifacts. Treat them as
  evidence; avoid cleaning or overwriting them unless the user explicitly asks.

## Working Rules

- Wiki first, then W&B/logs/checkpoints, then code. The training dynamics are
  subtle enough that code inspection alone is usually misleading.
- Prefer fixed external baselines such as heuristic/lookahead for strength
  claims. Head-to-head between sibling checkpoints can be non-transitive and can
  mostly measure mutual specialization.
- Treat short evals as noisy. A small n result is a hint, not proof. When a claim
  matters, verify with a larger match or cite the uncertainty.
- Watch game length. Falling `selfplay/plies_mean` and concave buffer-fill
  curves usually mean fast-attack collapse; stable or growing plies are a better
  sign than policy loss alone.
- Keep experiment notes evidence-backed: command/config, run ID, checkpoint path,
  key metrics, and what changed in the working theory.
- Preserve user work. This repo often contains large local artifacts and
  unfinished experimental state.

## Useful Commands

```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
pytest
gomoku-train --help
```

W&B project: `gomoku`.

When W&B is available locally, prefer pulling exact run histories over guessing
from memory or summary text.
