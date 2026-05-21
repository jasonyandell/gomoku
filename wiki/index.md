# Gomoku Wiki Index

This wiki is the maintained synthesis layer for the Gomoku training project.
It should compound what we learn from experiments instead of forcing each new
session to rediscover the same story from W&B runs, checkpoints, logs, and chat
history.

## Start Here

| Page | Role |
|---|---|
| [AGENTS.md](../AGENTS.md) | Schema for agents: wiki rules, repo map, and working conventions. |
| [TRAINING_WIKI.md](../TRAINING_WIKI.md) | Primary training notebook: run history, hypotheses, results, and corrections. |
| [log.md](log.md) | Chronological wiki maintenance log. |
| [topics/wiki-operating-model.md](topics/wiki-operating-model.md) | Gomoku-specific adaptation of the LLM wiki pattern. |
| [topics/mcts-perf-ceiling.md](topics/mcts-perf-ceiling.md) | Where MCTS gen-time wins are and aren't. Don't re-port "v2 storage" — we're already there. |
| [topics/activity-monitor-perf-runbook.md](topics/activity-monitor-perf-runbook.md) | Practical knobs and interpretation rules for Mac Activity Monitor perf experiments. |
| [topics/az-at-scale-vs-laptop.md](topics/az-at-scale-vs-laptop.md) | Why the wrinkles in our run (exploration arcs, plies swings, age oscillations) are laptop-scale artifacts, not training bugs. |
| [topics/wave-of-lockstep-design.md](topics/wave-of-lockstep-design.md) | Next run's design: per-version uniformity via wave-of-lockstep + greedy fill. Implementation plan and held-back levers. |
| [sources/karpathy-llm-wiki.md](sources/karpathy-llm-wiki.md) | Source record for the LLM wiki charter that inspired this structure. |

## Current Synthesis

The central project question is not just "can the code run faster?" It is "what
training loop reliably teaches 9x9 AlphaZero-style Gomoku to defend, not merely
to imitate fast attacks from its own search?"

Current evidence lives in [TRAINING_WIKI.md](../TRAINING_WIKI.md). The high-level
read is:

- Local throughput work succeeded; distributed self-play made iteration much
  faster.
- The current perf worktree includes optional native MCTS (`gomoku._mcts_native`)
  for Torch self-play. Production-shaped single-process MPS benches moved from
  ~700 to ~2,000-2,200 augmented positions/sec; the 10-epoch WL1 multi-worker
  read now shows ~2,379 wall augmented positions/sec at 8 workers x 8 games; see
  [topics/mcts-perf-ceiling.md](topics/mcts-perf-ceiling.md).
- The main training failure mode is fast-attack collapse: policy targets sharpen
  around attacks, self-play opponents fail to punish missing defense, and fixed
  heuristic/lookahead opponents expose the gap.
- Short evals are noisy. Strength claims need fixed baselines, enough games, and
  clear checkpoint/run IDs.
- The next useful additions to the wiki should preserve evidence: command,
  config, W&B run ID, checkpoint path, metrics, and the working-theory change.

## Layers

- **Evidence sources**: W&B histories, local logs, checkpoint files, match
  outputs, scripts, raw command output, and external source records under
  [sources/](sources/).
- **Maintained synthesis**: this index, topic pages under [topics/](topics/),
  and the training notebook.
- **Schema**: [AGENTS.md](../AGENTS.md), which tells future sessions how to
  maintain and use the wiki.

## Maintenance Rules

- Read this index first, then drill into the pages it names.
- Keep source records and artifacts immutable unless the user explicitly asks
  for cleanup.
- Keep the training notebook append-oriented. When a conclusion changes, add a
  dated correction with evidence instead of polishing the old entry.
- File useful answers back into the wiki when they would save a future session
  from recomputing the same synthesis.
- Update [log.md](log.md) whenever the wiki structure, index, or synthesis pages
  change in a meaningful way.
