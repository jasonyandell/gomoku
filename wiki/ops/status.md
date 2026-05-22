# ML Perf Frontier Status

This page is the current control-room summary for bounded Gomoku performance work. Raw evidence stays in logs, W&B, sweep artifacts, worktrees, and open notes; this page is the curated mixdown.

## Current Focus

Build an evidence-backed M5 Max performance map for Gomoku self-play/training while preserving training-quality guardrails. The next work should produce receipts, not intuition: command, commit, hardware/env, metrics, artifacts, confidence, and decision.

Use the frontier lab commands from pi:

```text
/frontier-start --dry-run
/frontier-start --max=5
/frontier-status
/frontier-curate
```

## Operating Rule

BFS by default: current-main baseline receipts and contour mapping before deep implementation. DFS only when a lane is hot because it has an active implementation worktree, a benchmark regression, a verifier failure, or a blocker that prevents progress.

## Current Read

- Native MCTS and eval Conv+BN fusion are already landed; new perf work should not rediscover those wins.
- Perf10 evidence in `/Users/jason/code/gomoku-perf-extension/sweep_logs/perf10-summary.tsv` says production shape matters: native 8 workers × 8 games beat native 4×16 and fallback 8×8, but the production win is smaller than the single-process microbench.
- Engine-isolation evidence has split into two questions: Core ML pressure hurts MPS training less than PyTorch/MPS eval pressure, but `CPU_AND_NE` alone is not ANE proof. The 934b worktree adds a rail-proof lab; advance it with `powermetrics`, not labels.
- Activity Monitor GPU percent remains a supporting observation only. Score by wall-clock, games/sec, positions/sec, trainer slowdown, plies/game shape, and fixed baseline/archive quality gates.
- Replay-buffer bit-packing is a post-WL5 candidate, but the cheap buffer-width ablation should decide whether the refactor is worth doing.

## Next Action

Run or dispatch the first five lanes from `.frontier/lanes.json`: baseline receipts, production contour sweep, ANE residency rail proof, quality promotion gates, and control-room curation. Keep production engine-overlap blocked until the residency lane produces an ANE-metered or explicit CPU-only isolation candidate.
