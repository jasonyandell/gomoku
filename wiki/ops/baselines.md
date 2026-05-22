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

## Notes

- Same-shape comparisons beat isolated intuition.
- Record env flags such as `GOMOKU_DISABLE_NATIVE_MCTS=1`, `GOMOKU_DISABLE_NATIVE_STATE_OPS=1`, and `PYTORCH_ENABLE_MPS_FALLBACK=1`.
- If a benchmark contends with a live training run, say so.
