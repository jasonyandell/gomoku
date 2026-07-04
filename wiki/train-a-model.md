# Train a model — the workflow

The pinned "just get me going" page. Skill: **`gomoku-train`** (start / resume /
stop / tune). Deep hub: [AlphaZero](alphazero.md).

> **← [index](index.md)** · workflows: [Eval a model](eval-a-model.md) · [Publish a model](publish-a-model.md)

## Fastest path

```bash
uv sync --extra dev                       # per-worktree env (uv.lock-pinned)
uv run gomoku-train --help                # latest.pt embeds the buffer for resume
```

## The pages you'll want

| Step | Page |
|---|---|
| Launch / smoke / monitor / stop a run | [launch-sequence-runbook.md](topics/launch-sequence-runbook.md) |
| **Every knob & switch** (run_sweep/train/worker, defaults, when-to-change, `file:line`) | [training-run-reference.md](topics/training-run-reference.md) |
| How did we get here / what to resume | [training-run-lineage.md](topics/training-run-lineage.md) |
| Interpret the dynamics while it runs | [loss-floor-bouncing.md](topics/loss-floor-bouncing.md) · [az-at-scale-vs-laptop.md](topics/az-at-scale-vs-laptop.md) |
| Warm-start across board sizes | [board-size-transfer-and-warm-start.md](topics/board-size-transfer-and-warm-start.md) |
| The current recipe | [sound-world-recipe.md](topics/sound-world-recipe.md) |

Watch **`selfplay/plies_mean`** — falling + concave buffer-fill = fast-attack
collapse. Gate strength on **H2H vs a frozen champion**, not siblings.
