---
name: experiment-ledger
description: Create concise, reproducible ML experiment receipts with metrics, artifacts, confidence, and promote/reject/block decisions.
---

# Experiment Ledger

Use this skill when writing worker receipts, open notes, or entries for `wiki/ops/experiment-ledger.md`.

## Receipt Schema

```yaml
lane:
hypothesis:
code_ref:
dataset_ref:
baseline_command:
candidate_command:
hardware:
seed:
baseline_metric:
candidate_metric:
delta:
confidence:
artifacts:
commands_run:
decision: promote | reject | blocked | needs_repeat
next_action:
```

## Rules

- Include exact commands, not paraphrases.
- Include commit hash or branch when code changed.
- Include paths to JSON/TSV/log artifacts.
- Include negative results; rejected ideas are valuable.
- If no benchmark ran, classify as `blocked` and explain the smallest unblock.
- If n is small or eval is noisy, set `decision: needs_repeat` unless the result is only a smoke/verifier.

## Good Decision Language

- `promote`: measured win, guardrails pass, acceptable complexity.
- `reject`: measured regression or complexity not justified.
- `blocked`: missing artifact, broken harness, unavailable hardware, unresolved correctness issue.
- `needs_repeat`: promising/noisy result needs larger n or production-shaped replication.
