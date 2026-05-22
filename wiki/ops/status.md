# ML Perf Frontier Status

This page is the current control-room summary for bounded Gomoku performance work. Raw evidence stays in logs, W&B, sweep artifacts, and open notes; this page is the curated mixdown.

## Current Focus

Maximize trustworthy training/self-play throughput on Apple Silicon without degrading strength, game-shape signals, or reproducibility.

Use the frontier lab commands from pi:

```text
/frontier-start --dry-run
/frontier-start --max=5
/frontier-status
/frontier-curate
```

## Operating Rule

BFS by default: map and baseline before deep implementation. DFS only when a lane is hot because a benchmark regression, verifier failure, run-blocking bug, or active implementation makes deeper focus worth it.

## Current Read

- Existing wiki evidence says Activity Monitor GPU percent is not the objective; score changes by wall-clock, games/sec, positions/sec, and training-quality guardrails.
- Native MCTS and eval Conv+BN fusion are already landed; new perf claims should not rediscover those wins.
- Engine-isolation evidence suggests Core ML raw latency is slower than fused PyTorch/MPS, but it may reduce trainer contention. The next useful test is production-overlap shaped.

## Next Action

Start with the `baseline-harness` and `selfplay-throughput` lanes so every future optimization has paired baseline/candidate receipts.
