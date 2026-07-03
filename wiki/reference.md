# Reference — the look-things-up shelf

Not a narrative arc — a catalog. The Training-wiki broken into topics, **all our
evals** (including forgotten ones), the **tools**, and cross-cutting conventions.

> **← Hubs:** [index](index.md) · sibling hubs: [AlphaZero](alphazero.md) ·
> [Experiments](experiments.md) · [Derby](derby.md) · [M5-as-Mainframe](m5-mainframe.md)

## Capabilities & conventions

| Page | Role |
|---|---|
| [capabilities.md](capabilities.md) | The one-screen capability map (mine · pretrain · train · evaluate · search · operate). |
| [conventions.md](topics/conventions.md) | Deny-list autonomy, merge-commits-never-rebase, memories-also-go-to-wiki, Opus-minutes. |
| [wiki-operating-model.md](topics/wiki-operating-model.md) | This wiki's adaptation of the LLM-wiki pattern. |
| [AGENTS.md](../AGENTS.md) | Schema for agents (native twin of CLAUDE.md). |

## Training-dynamics reference

- [loss-floor-bouncing.md](topics/loss-floor-bouncing.md) — when low-floor loss bounces are healthy vs a bug.
- [az-at-scale-vs-laptop.md](topics/az-at-scale-vs-laptop.md) — laptop-scale artifacts vs bug evidence.
- [training-run-reference.md](topics/training-run-reference.md) — every knob & switch (also linked from the Train workflow).

## Evals (including the forgotten ones)

| Page | Role |
|---|---|
| [eval-suite.md](topics/eval-suite.md) | How to EVALUATE a checkpoint (command-first; the white-defense gate; the EMA-not-raw gotcha). |
| [batched-eval-arena.md](topics/batched-eval-arena.md) | `gomoku-arena` — 40-game eval ≈4s (wave-MCTS + warm Rapfi pool). |
| [reliable-eval-set.md](topics/reliable-eval-set.md) | What's a RELIABLE eval? (wine engines shelved; net-vs-net + heuristic/lookahead + native Rapfi anchor). |
| [external-engine-baselines.md](topics/external-engine-baselines.md) | Rated OSS Gomocup engine candidates + the native Rapfi-NNUE anchor. |
| [gomocup-engines-catalog.md](topics/gomocup-engines-catalog.md) | Which external engines are open-source / runnable. |
| [reliable-eval-set.md](topics/reliable-eval-set.md) · [probe-100pct.md](topics/probe-100pct.md) | The distance-to-100% sweep driver. |
| [mining-validation-archives.md](topics/mining-validation-archives.md) | Mine/use validation archives. |

## Tools

| Tool | Page |
|---|---|
| **Rapfi pool** (HuggingFace-pinned NNUE, warm CPU process pool) | [rapfi-pool.md](topics/rapfi-pool.md) |
| **Arena** (batched fast eval) | [batched-eval-arena.md](topics/batched-eval-arena.md) |
| Rapfi **sensei/teacher** (always-on eval + distillation) | [eval-teacher-sensei.md](topics/eval-teacher-sensei.md) |
| W&B workspace generator | [scripts/wandb_workspace.py](../scripts/wandb_workspace.py) |
| Containerize a training run (backlog) | [containerize-training-runs.md](topics/containerize-training-runs.md) |

## Sources (external records)

[karpathy-llm-wiki.md](sources/karpathy-llm-wiki.md) ·
[sid-bidasaria-stop-babysitting-agents-2026-05-20.md](sources/sid-bidasaria-stop-babysitting-agents-2026-05-20.md) ·
[gomocup-az-techniques-2026-05-27.md](sources/gomocup-az-techniques-2026-05-27.md) ·
[gomocup-external-engines-2026-05-22.md](sources/gomocup-external-engines-2026-05-22.md)

## Full page index — every page in this hub

*Complete map (23 pages); the sections above surface the headline ones.*

| Page | Note |
|---|---|
| [capabilities.md](capabilities.md) | capability map |
| [conventions.md](topics/conventions.md) |  |
| [wiki-operating-model.md](topics/wiki-operating-model.md) |  |
| [training-run-reference.md](topics/training-run-reference.md) | every knob & switch |
| [loss-floor-bouncing.md](topics/loss-floor-bouncing.md) | training-dynamics ref |
| [az-at-scale-vs-laptop.md](topics/az-at-scale-vs-laptop.md) | training-dynamics ref |
| [branch-and-worktree-workflow.md](topics/branch-and-worktree-workflow.md) | canonical lifecycle |
| [worktree-hygiene.md](topics/worktree-hygiene.md) | cleanup half + uv gotcha |
| [eval-suite.md](topics/eval-suite.md) | also workflow-eval |
| [batched-eval-arena.md](topics/batched-eval-arena.md) | arena tool |
| [reliable-eval-set.md](topics/reliable-eval-set.md) |  |
| [external-engine-baselines.md](topics/external-engine-baselines.md) |  |
| [gomocup-engines-catalog.md](topics/gomocup-engines-catalog.md) |  |
| [probe-100pct.md](topics/probe-100pct.md) | eval probe driver |
| [mining-validation-archives.md](topics/mining-validation-archives.md) |  |
| [rapfi-pool.md](topics/rapfi-pool.md) | tool: Rapfi HF pool |
| [eval-teacher-sensei.md](topics/eval-teacher-sensei.md) | tool: sensei/teacher |
| [playing-the-model.md](topics/playing-the-model.md) | also workflow-publish |
| [containerize-training-runs.md](topics/containerize-training-runs.md) | backlog |
| [karpathy-llm-wiki.md](sources/karpathy-llm-wiki.md) |  |
| [sid-bidasaria-stop-babysitting-agents-2026-05-20.md](sources/sid-bidasaria-stop-babysitting-agents-2026-05-20.md) |  |
| [gomocup-az-techniques-2026-05-27.md](sources/gomocup-az-techniques-2026-05-27.md) |  |
| [gomocup-external-engines-2026-05-22.md](sources/gomocup-external-engines-2026-05-22.md) |  |
