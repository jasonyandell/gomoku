---
name: frontier-orchestration
description: BFS/hot-lane orchestration for project frontier work. Use when coordinating multiple agents, worktrees, wiki receipts, or performance lanes without losing control of scope.
---

# Frontier Orchestration

Use this skill when a project uses `.frontier/lanes.json`, `wiki/ops/*`, and the `/frontier-*` pi commands.

## Loop

1. Read project instructions first (`AGENTS.md`) and the wiki index.
2. Read `.frontier/config.json`, `.frontier/lanes.json`, and `wiki/ops/frontier.md`.
3. Pick unblocked lanes in BFS order.
4. DFS only when a lane is hot: failing benchmark, active implementation, reproducible regression, or blocker that prevents progress.
5. Keep writes single-lane and receipt-backed.
6. Curate receipts into the wiki after integration.

## Lane Done Shape

A lane is not done until it has:

- source/code mapping,
- hypothesis or blocker,
- baseline command,
- candidate command or reason no candidate exists,
- hardware/env details,
- metric delta or verifier result,
- artifact path,
- confidence/noise caveat,
- decision: `promote`, `reject`, `blocked`, or `needs_repeat`,
- next action,
- wiki/open-note update.

## Concurrency Rules

- One manager owns lane selection.
- Workers must not pick a different lane.
- Use git worktrees for concurrent writers.
- Curator is the only shared-board writer for `wiki/ops/status.md`, `wiki/ops/frontier.md`, `wiki/ops/baselines.md`, `wiki/ops/experiment-ledger.md`, and `wiki/ops/test-ledger.md`.
- Advisory workers can inspect broadly; implementation workers change one bounded surface.

## ML Perf Bias

For performance work, paired measurement beats intuition. If a worker cannot produce a paired baseline/candidate result, it should return a blocker or a benchmark design, not a speedup claim.
