# ML Perf Test Ledger

Record validation commands and outcomes that support frontier decisions.

| Date | Lane | Command | Result | Artifact / note |
| --- | --- | --- | --- | --- |
| 2026-05-22 | setup | `pi --mode rpc --no-extensions -e .pi/extensions/frontier-lab/index.ts --no-session` command discovery | pending local load test | project-local frontier lab setup |

## Standard Gates

- Code correctness: `pytest -q`
- CPU smoke: `python scripts/perf_microbench.py --device cpu --size tiny --games 2 --n-simulations 2 --wave-size 1 --max-plies 2 --repeats 1 --warmup 0`
- MPS perf comparison: use same-shape baseline/candidate commands from `wiki/ops/baselines.md`.
