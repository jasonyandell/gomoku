# ML Perf Frontier Status

This page is the current control-room summary for bounded Gomoku performance work. Raw evidence stays in logs, W&B, sweep artifacts, worktrees, and open notes; this page is the curated mixdown.

## Current Focus

Frontier run `20260522T061713Z` completed and integrated the post-native outer-loop Python profiling lane. The result is a **reject/no-op for another post-search Python native pass**: action sampling, trajectory staging, D4 augmentation, record creation, and file handoff together accounted for only ~4-5% of wall time in the bounded production-shaped profile.

The next useful perf work is evaluator/engine isolation or deeper `native_search_batch` / evaluator-boundary investigation. Production engine-overlap remains blocked until ANE rail proof can be repeated with `powermetrics`; absent that, no unblocked hot perf lane should be promoted from this receipt.

Use the frontier lab commands from pi:

```text
/frontier-start --dry-run
/frontier-status
/frontier-curate
# If sudo/powermetrics is unblocked: /frontier-start --max=1 --lane=ane-residency-rail-proof
```

## Operating Rule

BFS by default: current-main baseline receipts and production contour are now done. DFS only when a lane is hot because it has an active implementation worktree, a benchmark regression, a verifier failure, or a blocker that prevents progress.

## Current Read

- Baseline receipts are complete under `sweep_logs/frontier-baselines/20260522T054845Z`: current-main native MCTS beat fallback by ~3.7x on the standard MPS microbench under live WL5 contention.
- Production contour is complete under `sweep_logs/production-contour-20260522/`: promote native small `8w x 8g`, `sims=400`, `wave=64` as the throughput default; reject fallback, `4w16g`, and `wave32` for throughput. `sims=200` and `tiny` are speed candidates only and need quality gates before behavior-changing use.
- Quality promotion gates are codified in `wiki/ops/experiment-ledger.md`.
- Core ML / ANE residency harness is integrated, but the lane is blocked for fresh proof because `sudo -n true` failed and `powermetrics required` could not run. Existing detached 934b artifacts remain useful candidates, not production proof.
- Outer-loop profiling (`20260522T061713Z`) found the worker wall dominated by `native_search_batch` / evaluator time: wave-mode wall 1.064s, evaluator 0.896s (84.3%), native search excluding evaluator 0.117s (11.0%), and measured post-search Python 0.050s (4.7%). Decision: reject post-search Python optimization pass.
- The frontier extension stale-UI-context failure was recovered manually and patched in commit `7e26e7c` so future completed background runs should not be marked failed solely because the command UI context expired.

## Active / Blocked Lanes

| Lane | Status | Evidence / blocker |
| --- | --- | --- |
| Outer self-play loop profiling | completed / rejected | Run `20260522T061713Z`; no 10-20% post-search Python owner found. |
| ANE residency rail proof | blocked | Needs cached/passwordless sudo for same-window `powermetrics` plus Vision positive control / CPU_ONLY negative. |
| Production engine-overlap | blocked | Wait for ANE-metered or explicit CPU-only isolation candidate; do not launch from `CPU_AND_NE` labels alone. |
| Replay-buffer width cheap test | active in another session | BAB1-buf-ablation-1p5M live as of 2026-05-22; trainer PID 27579, e10215/10700 at last check. See [perf-log.md](perf-log.md). |
| Canonical 5-axis M5 Max contour sweep | blocked / warm | Scaffolded this session: `scripts/canonical_sweep.py`, `scripts/plot_canonical_sweep.py`, [perf-lab-session-runbook](../topics/perf-lab-session-runbook.md). Launches when the box clears BAB1+BAB2. |

## Next Action

Do not launch a D4/action-sampling/file-handoff native pass. Promote evaluator/engine overlap only after the ANE rail-proof blocker is removed, or open a narrowly-scoped `native_search_batch` / evaluator-boundary profiling lane if the manager wants another CPU-side perf fanout. Otherwise keep the replay-buffer cheap test warm until WL5 reports out.
