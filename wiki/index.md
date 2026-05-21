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
| [topics/loss-floor-bouncing.md](topics/loss-floor-bouncing.md) | Why low-floor loss bounces can be healthy in small-scale AZ, and when to suspect a real bug. |
| [topics/wave-of-lockstep-design.md](topics/wave-of-lockstep-design.md) | WL1 design: per-version uniformity via wave-of-lockstep + greedy fill. Implementation plan and held-back levers. |
| [topics/wl2-scale-emulation-design.md](topics/wl2-scale-emulation-design.md) | WL2 design: emulate AZ-at-scale in-flight diversity via EMA self-play + past-checkpoint mix + worker poll jitter + grad accumulation. Motivated by WL1's high-frequency oscillation failure mode. |
| [topics/wl5-diagnostics-archive-start-design.md](topics/wl5-diagnostics-archive-start-design.md) | WL5 design: 3 diagnostic streams (fixed validation archive, H/KL decomposition, per-color/ply metrics) + Go-Exploit-style archive-start lever (15% of self-play games from curated WL4 trouble positions). Resume from WL4 e4024. Targets the article's central interpretive distinction: target-distribution noise vs learning gap. |
| [topics/mining-validation-archives.md](topics/mining-validation-archives.md) | Operational recipe for `scripts/mine_validation_archive.py` — buckets, knobs, throughput, anti-patterns. Reuse this every time we need a fresh validation archive. |
| [topics/launch-sequence-runbook.md](topics/launch-sequence-runbook.md) | Reusable playbook for kicking off a training run. Pre-launch checks (incl. MPS INT_MAX + worker race gotchas), title card → ACK, smoke, real launch, wiki + workspace updates, /loop monitoring cadence, fan-out implementation pattern. |
| [topics/playing-the-model.md](topics/playing-the-model.md) | How to actually play a trained checkpoint: local web UI (strongest), live SPA (convenient), which checkpoint to pick, knobs that matter, common annoyances. |
| [sources/karpathy-llm-wiki.md](sources/karpathy-llm-wiki.md) | Source record for the LLM wiki charter that inspired this structure. |
| [../scripts/wandb_workspace.py](../scripts/wandb_workspace.py) | Creates the 6-section wandb workspace tuned for WL1-vs-Z overlays. Live view: https://wandb.ai/jasonyandell-forge42/gomoku?nw=ul0vliphj6x |

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
