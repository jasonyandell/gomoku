# Gomoku Agent Orientation

Start with the wiki index: `wiki/index.md`.

This repo follows the LLM-wiki pattern: raw evidence stays stable, the wiki is
the maintained synthesis layer, and this file is the schema that tells future
agents how to work. Do not treat the wiki as a prettier transcript. Treat it as
a compounding project artifact that should make each session smarter than the
last.

After the index, read `TRAINING_WIKI.md`. It is the main chronological training
notebook and records the training story, failed hypotheses, W&B run IDs,
checkpoint meanings, performance findings, and current interpretation of the
learning dynamics. Read it before making claims about why training is working or
failing.

The training notebook is intentionally append-oriented. Do not rewrite old
conclusions to make the story cleaner. If new evidence contradicts an older
note, add a new dated entry that explains the correction and points to the
evidence.

## Wiki Architecture

- `wiki/index.md` is the content-oriented entry point. Keep it current when new
  durable pages appear or when the current synthesis materially changes.
- `wiki/log.md` is the chronological maintenance log. Append entries when wiki
  structure or maintained synthesis changes.
- `wiki/sources/` holds source records for external references and other
  evidence that should be stable.
- `wiki/topics/` holds maintained synthesis pages that are too reusable to leave
  buried in chat or a long run log.
- `TRAINING_WIKI.md` remains the training lab notebook. Use it for experiment
  history, dated corrections, run evidence, and working-theory changes.

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

- Wiki index first, then the training notebook, then W&B/logs/checkpoints, then
  code. The training dynamics are subtle enough that code inspection alone is
  usually misleading.
- Git workflow is canonical and load-bearing here:
  [wiki/topics/branch-and-worktree-workflow.md](wiki/topics/branch-and-worktree-workflow.md).
  **Every unit of work happens in its own worktree off `main`, lands via `git
  merge --no-ff`, and is torn down afterward — you do not edit the shared
  `main` checkout.** The overnight derby, the user's IDE, and other agent
  sessions routinely share that checkout; working there entangles diffs and
  blocks clean merges. Never rebase, fast-forward, or squash. Run the
  session-start janitor (`python scripts/reclaim_worktrees.py --apply`) to
  reclaim what crashed sessions leak — cleanup is a janitor, not a remembered
  procedure ([wiki/topics/worktree-hygiene.md](wiki/topics/worktree-hygiene.md)).
- File reusable answers back into the wiki. If a question produces a useful
  synthesis, add a topic page or update the index/log so the next session does
  not rediscover it from scratch.
- Keep evidence and synthesis separate. Do not overwrite raw artifacts or clean
  away local run evidence unless the user explicitly asks.
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
