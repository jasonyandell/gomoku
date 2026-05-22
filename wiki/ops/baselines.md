# ML Perf Baselines

Use this page to pin benchmark commands and reference numbers that future workers can compare against. Do not overwrite old baseline rows; add a new dated row when hardware, code, config, or run shape changes.

## Command Surfaces

### CPU syntax smoke

```bash
python scripts/perf_microbench.py --device cpu --size tiny --games 2 --n-simulations 2 --wave-size 1 --max-plies 2 --repeats 1 --warmup 0
```

Purpose: fast correctness/syntax smoke. Not a performance claim.

### MPS production-shaped microbench

```bash
python scripts/perf_microbench.py --device mps --size small --stem-padding 1 --games 8 --n-simulations 400 --wave-size 64 --max-plies 16 --repeats 3
```

Purpose: bounded self-play/MCTS throughput comparison. Score by seconds, games/sec, and positions/sec.

## Baseline Rows

| Date | Commit | Hardware | Command | Metric | Result | Artifact |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-05-22 | existing wiki | M5 Max / MPS | see `wiki/topics/activity-monitor-perf-runbook.md` | native MCTS small MPS max_plies=16 | 2,200 aug pos/s reference | wiki topic |
| 2026-05-22 | `4f21cdd` worktree | M5 Max / MPS | WL1-shaped 10-epoch production sweep, small model, 400 sims, wave 64 | native 8w8g | wall 2,379 aug pos/s; gen 3,303 aug pos/s; wall 11.25 games/s | `/Users/jason/code/gomoku-perf-extension/sweep_logs/perf10-summary.tsv` |
| 2026-05-22 | `4f21cdd` worktree | M5 Max / MPS | same as above | native 4w16g | wall 1,918 aug pos/s; gen 2,152 aug pos/s; wall 8.61 games/s | `/Users/jason/code/gomoku-perf-extension/sweep_logs/perf10-summary.tsv` |
| 2026-05-22 | `4f21cdd` worktree | M5 Max / MPS | same as above with Python MCTS fallback | fallback 8w8g | wall 1,863 aug pos/s; gen 2,264 aug pos/s; wall 8.85 games/s | `/Users/jason/code/gomoku-perf-extension/sweep_logs/perf10-summary.tsv` |
| 2026-05-22 | `a418f67` main/frontier worktree | M5 Max / CPU | CPU syntax smoke, default native | native_mcts=true; native_state_ops=true; fused_eval=true | median 0.008s; 266.12 games/s; 4,258 aug pos/s; plies_mean 2.0 | `sweep_logs/frontier-baselines/20260522T054845Z/cpu-smoke-native.txt` |
| 2026-05-22 | `a418f67` main/frontier worktree | M5 Max / CPU | CPU syntax smoke with `GOMOKU_DISABLE_NATIVE_MCTS=1` | native_mcts=false; native_state_ops=true; fused_eval=true | median 0.009s; 224.51 games/s; 3,592 aug pos/s; plies_mean 2.0 | `sweep_logs/frontier-baselines/20260522T054845Z/cpu-smoke-fallback.txt` |
| 2026-05-22 | `a418f67` main/frontier worktree | M5 Max / MPS, live WL5 contention | MPS production-shaped microbench, default native | native_mcts=true; native_state_ops=true; fused_eval=true | median 0.626s; 12.79 games/s; 1,637 aug pos/s; plies_mean 16.0 | `sweep_logs/frontier-baselines/20260522T054845Z/mps-microbench-native.txt` |
| 2026-05-22 | `a418f67` main/frontier worktree | M5 Max / MPS, live WL5 contention | MPS production-shaped microbench with `GOMOKU_DISABLE_NATIVE_MCTS=1` | native_mcts=false; native_state_ops=true; fused_eval=true | median 2.309s; 3.46 games/s; 443 aug pos/s; plies_mean 16.0 | `sweep_logs/frontier-baselines/20260522T054845Z/mps-microbench-fallback.txt` |

## Notes

- Same-shape comparisons beat isolated intuition.
- Record env flags such as `GOMOKU_DISABLE_NATIVE_MCTS=1`, `GOMOKU_DISABLE_NATIVE_STATE_OPS=1`, and `PYTORCH_ENABLE_MPS_FALLBACK=1`.
- If a benchmark contends with a live training run, say so.
- Current frontier artifact convention: raw command output under `sweep_logs/frontier-baselines/<timestamp>/`, plus a small `summary.tsv` or `summary.json` for ledger ingestion.
