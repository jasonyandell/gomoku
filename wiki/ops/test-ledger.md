# ML Perf Test Ledger

Record validation commands and outcomes that support frontier decisions.

| Date | Lane | Command | Result | Artifact / note |
| --- | --- | --- | --- | --- |
| 2026-05-22 | setup | `pi --mode rpc --no-extensions -e .pi/extensions/frontier-lab/index.ts --no-session` command discovery | pending local load test | project-local frontier lab setup |
| 2026-05-22 | production-contour-sweep | WL1-shaped 10-epoch sweep, native MCTS, 8 workers x 8 games | wall 2,379 aug pos/s; gen 3,303 aug pos/s; wall 11.25 games/s | `/Users/jason/code/gomoku-perf-extension/sweep_logs/perf10-summary.tsv` |
| 2026-05-22 | production-contour-sweep | WL1-shaped 10-epoch sweep, native MCTS, 4 workers x 16 games | wall 1,918 aug pos/s; gen 2,152 aug pos/s; wall 8.61 games/s | `/Users/jason/code/gomoku-perf-extension/sweep_logs/perf10-summary.tsv` |
| 2026-05-22 | production-contour-sweep | WL1-shaped 10-epoch sweep, Python MCTS fallback, 8 workers x 8 games | wall 1,863 aug pos/s; gen 2,264 aug pos/s; wall 8.85 games/s | `/Users/jason/code/gomoku-perf-extension/sweep_logs/perf10-summary.tsv` |

## Standard Gates

- Code correctness: `pytest -q`
- CPU smoke: `python scripts/perf_microbench.py --device cpu --size tiny --games 2 --n-simulations 2 --wave-size 1 --max-plies 2 --repeats 1 --warmup 0`
- MPS perf comparison: use same-shape baseline/candidate commands from `wiki/ops/baselines.md`.
