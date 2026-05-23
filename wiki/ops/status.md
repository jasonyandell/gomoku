# ML Perf Frontier Status

This page is the current control-room summary for bounded Gomoku performance work. Raw evidence stays in logs, W&B, sweep artifacts, worktrees, and open notes; this page is the curated mixdown.

## Current Focus

**Canonical 5-axis sweep complete (`canonical-sweep-mainframe` lane, 2026-05-23).** 23/23 cells passed under idle-box conditions, sweep_logs/canonical-sweep-20260523T015614Z. Headline finding: **wave_size is materially under-tuned in production.** Going from the WL5-era default (small W=8 G=8 sims=400 wave=64 = 3,188 aug pos/s) to wave=128 gives **+27%** (4,048 aug/s) with no behavior change; wave=256 gives +38% (4,409). The workers axis is near-flat past W=8; sims scales perfectly inversely (pure quality knob); per-step model scaling is 2.3× (tiny→small→medium). Full receipt in [experiment-ledger.md](experiment-ledger.md) "2026-05-23 — canonical 5-axis M5 Max contour sweep".

**Promoted throughput default:** small / W=8 / G=8 / sims=400 / **wave=128**. The next behavior-changing run that adopts this needs a canary cycle reporting `val/policy_ce` + `val/policy_kl` against `archives/wl5_validation_v1.pt` per the Training-Quality Promotion Gate.

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
| Canonical 5-axis M5 Max contour sweep | completed / promote | 23/23 cells in `sweep_logs/canonical-sweep-20260523T015614Z`; wave=128 promoted as new throughput default (+27% over V=64); receipt in [experiment-ledger.md](experiment-ledger.md); contour/axes/model_compare PNGs alongside summary. |

## Next Action

Do not launch a D4/action-sampling/file-handoff native pass. Promote evaluator/engine overlap only after the ANE rail-proof blocker is removed, or open a narrowly-scoped `native_search_batch` / evaluator-boundary profiling lane if the manager wants another CPU-side perf fanout. Otherwise keep the replay-buffer cheap test warm until WL5 reports out.
