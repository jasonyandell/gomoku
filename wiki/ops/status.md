# ML Perf Frontier Status

This page is the current control-room summary for bounded Gomoku performance work. Raw evidence stays in logs, W&B, sweep artifacts, worktrees, and open notes; this page is the curated mixdown.

## Current Focus

The first perf frontier fanout (`20260522T054739Z`) finished all workers and was manually integrated after the manager hit a stale-UI-context bug. The current actionable lane is now post-native outer-loop Python profiling. GPU engine-overlap work is blocked until ANE rail proof can be repeated with `powermetrics`.

Use the frontier lab commands from pi:

```text
/frontier-start --dry-run
/frontier-start --max=1 --lane=outer-loop-python-profile
/frontier-status
/frontier-curate
```

## Operating Rule

BFS by default: current-main baseline receipts and production contour are now done. DFS only when a lane is hot because it has an active implementation worktree, a benchmark regression, a verifier failure, or a blocker that prevents progress.

## Current Read

- Baseline receipts are complete under `sweep_logs/frontier-baselines/20260522T054845Z`: current-main native MCTS beat fallback by ~3.7x on the standard MPS microbench under live WL5 contention.
- Production contour is complete under `sweep_logs/production-contour-20260522/`: promote native small `8w x 8g`, `sims=400`, `wave=64` as the throughput default; reject fallback, `4w16g`, and `wave32` for throughput. `sims=200` and `tiny` are speed candidates only and need quality gates before behavior-changing use.
- Quality promotion gates are codified in `wiki/ops/experiment-ledger.md`.
- Core ML / ANE residency harness is integrated, but the lane is blocked for fresh proof because `sudo -n true` failed and `powermetrics required` could not run. Existing detached 934b artifacts remain useful candidates, not production proof.
- The frontier extension stale-UI-context failure was recovered manually and patched in commit `7e26e7c` so future completed background runs should not be marked failed solely because the command UI context expired.

## Active / Blocked Lanes

| Lane | Status | Evidence / blocker |
| --- | --- | --- |
| Outer self-play loop profiling | hot / open | Next actionable CPU lane. Profile post-native worker-loop Python: sampling, trajectory staging, D4, record creation, and file handoff. |
| ANE residency rail proof | blocked | Needs cached/passwordless sudo for same-window `powermetrics` plus Vision positive control / CPU_ONLY negative. |
| Production engine-overlap | blocked | Wait for ANE-metered or explicit CPU-only isolation candidate; do not launch from `CPU_AND_NE` labels alone. |
| Replay-buffer width cheap test | warm / seeded | Post-WL5 ablation: 1.5M vs 750k buffer before bit-packing. |

## Next Action

Start or dispatch `outer-loop-python-profile` as a single CPU lane. Keep GPU lanes blocked until the ANE rail-proof blocker is removed.
