# ML Perf Test Ledger

Record validation commands and outcomes that support frontier decisions.

| Date | Lane | Command | Result | Artifact / note |
| --- | --- | --- | --- | --- |
| 2026-05-22 | setup | `pi --mode rpc --no-extensions -e .pi/extensions/frontier-lab/index.ts --no-session` command discovery | pending local load test | project-local frontier lab setup |
| 2026-05-22 | production-contour-sweep | WL1-shaped 10-epoch sweep, native MCTS, 8 workers x 8 games | wall 2,379 aug pos/s; gen 3,303 aug pos/s; wall 11.25 games/s | `/Users/jason/code/gomoku-perf-extension/sweep_logs/perf10-summary.tsv` |
| 2026-05-22 | production-contour-sweep | WL1-shaped 10-epoch sweep, native MCTS, 4 workers x 16 games | wall 1,918 aug pos/s; gen 2,152 aug pos/s; wall 8.61 games/s | `/Users/jason/code/gomoku-perf-extension/sweep_logs/perf10-summary.tsv` |
| 2026-05-22 | production-contour-sweep | WL1-shaped 10-epoch sweep, Python MCTS fallback, 8 workers x 8 games | wall 1,863 aug pos/s; gen 2,264 aug pos/s; wall 8.85 games/s | `/Users/jason/code/gomoku-perf-extension/sweep_logs/perf10-summary.tsv` |
| 2026-05-22 | baseline-receipts | `python scripts/perf_microbench.py --device cpu --size tiny --games 2 --n-simulations 2 --wave-size 1 --max-plies 2 --repeats 1 --warmup 0` | passed; native_mcts=true; native_state_ops=true; fused_eval=true; median 0.008s; 266.12 games/s; 4,258 aug pos/s; plies_mean 2.0 | `sweep_logs/frontier-baselines/20260522T054845Z/cpu-smoke-native.txt` |
| 2026-05-22 | baseline-receipts | `GOMOKU_DISABLE_NATIVE_MCTS=1 python scripts/perf_microbench.py --device cpu --size tiny --games 2 --n-simulations 2 --wave-size 1 --max-plies 2 --repeats 1 --warmup 0` | passed; native_mcts=false; native_state_ops=true; fused_eval=true; median 0.009s; 224.51 games/s; 3,592 aug pos/s; plies_mean 2.0 | `sweep_logs/frontier-baselines/20260522T054845Z/cpu-smoke-fallback.txt` |
| 2026-05-22 | baseline-receipts | `python scripts/perf_microbench.py --device mps --size small --stem-padding 1 --games 8 --n-simulations 400 --wave-size 64 --max-plies 16 --repeats 3` | passed under live WL5 contention; native_mcts=true; native_state_ops=true; fused_eval=true; median 0.626s; 12.79 games/s; 1,637 aug pos/s; plies_mean 16.0 | `sweep_logs/frontier-baselines/20260522T054845Z/mps-microbench-native.txt` |
| 2026-05-22 | baseline-receipts | `GOMOKU_DISABLE_NATIVE_MCTS=1 python scripts/perf_microbench.py --device mps --size small --stem-padding 1 --games 8 --n-simulations 400 --wave-size 64 --max-plies 16 --repeats 3` | passed under live WL5 contention; native_mcts=false; native_state_ops=true; fused_eval=true; median 2.309s; 3.46 games/s; 443 aug pos/s; plies_mean 16.0 | `sweep_logs/frontier-baselines/20260522T054845Z/mps-microbench-fallback.txt` |
| 2026-05-22 | baseline-receipts | `pytest -q` | passed | `sweep_logs/frontier-baselines/20260522T054845Z/pytest-q.txt` |

## Standard Gates

- Code correctness: `pytest -q`
- CPU smoke: `python scripts/perf_microbench.py --device cpu --size tiny --games 2 --n-simulations 2 --wave-size 1 --max-plies 2 --repeats 1 --warmup 0`
- MPS perf comparison: use same-shape baseline/candidate commands from `wiki/ops/baselines.md`.
