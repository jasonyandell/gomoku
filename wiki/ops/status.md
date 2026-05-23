# ML Perf Frontier Status

This page is the current control-room summary for bounded Gomoku performance work. Raw evidence stays in logs, W&B, sweep artifacts, worktrees, and open notes; this page is the curated mixdown.

## Current Focus

**Autonomous perf lab running.** Three promotes in two lanes:

| ref | best cell | aug/s | speedup vs WL5 V=64 |
|---|---|---|---|
| R-S400 | small W8 G8 S400 **V=512** | 4,765 | **+49.5%** |
| R-S200 | small W8 G8 S200 **V=512** | 9,156 | **+52.5%** |
| R-S100 | small W8 G8 S100 **V=512** | 15,082 | **+35.2%** |

**Latest:** L03 sims-x-wave landed double promotes at R-S200 and R-S100 — V=512 wins at every quality point measured so far. The eval-batch-shape benefit dominates regardless of n-sims. Reviewer PENDING on L03; L01 (R-S400) already APPROVE.

**Up next:** Tier 2 L02 (W × V=512) is the next unblocked compound; tests whether the wave win still holds at W=4/12/16. Tier 1 R-TRAIN-* cells stay blocked on L12 (live-training driver). Bg L07 tiny contour at V=512 becomes valuable after L02.

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
| Canonical 5-axis M5 Max contour sweep | completed / promote | 23/23 cells; wave=128 promoted as new R-S400 default (+27% over V=64); receipt in [experiment-ledger.md](experiment-ledger.md). |
| L01-wave-extrapolation | completed / promote | 4/4 cells; **wave=512** promoted as new R-S400 default (+17.7% over V=128; +49.5% cumulative). Plateau confirmed at V=512 (V=768/1024 flat). Reviewer: APPROVE. |
| L03-sims-x-wave | completed / promote (2x) | 2/2 cells; **V=512** promoted at R-S200 (+52.5%) AND R-S100 (+35.2%). Wave-size win uniform across sims axis. Reviewer: APPROVE. |
| L02-W-x-wave-compound | completed / reject | 3/3 cells; W=8 V=512 stays optimal (4,765). W=4=4,367; W=12=4,501; W=16=4,504. **Wave saturation INVERTED the W-axis peak** (was W=16 at V=64; now W=8 at V=512). Knob wins interact non-monotonically at chip envelope. Reviewer: APPROVE. |
| L04-G-x-wave | completed / reject | 3/3 cells; G=8 V=512 stays optimal. G=4=4,608 (-3.3%); G=16=4,541 (-4.7%); G=32=4,514 (-5.3%). G mildly non-monotone at V=512 (was flat at V=64). **Compound finding with L02**: at V=512 BOTH W and G axes peak at canonical defaults — wider perimeter exploration won't beat the center. Reviewer: APPROVE. |

## Next Action

Do not launch a D4/action-sampling/file-handoff native pass. Promote evaluator/engine overlap only after the ANE rail-proof blocker is removed, or open a narrowly-scoped `native_search_batch` / evaluator-boundary profiling lane if the manager wants another CPU-side perf fanout. Otherwise keep the replay-buffer cheap test warm until WL5 reports out.
