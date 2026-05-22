---
name: ml-performance-lab
description: Evidence-first ML performance workflow for Gomoku/Apple Silicon work. Use when optimizing training, self-play, MCTS, inference engines, or benchmark throughput.
---

# ML Performance Lab

Use this skill for Gomoku performance work.

## Required Reading

1. `AGENTS.md`
2. `wiki/index.md`
3. `TRAINING_WIKI.md` when making training-quality claims
4. `wiki/topics/activity-monitor-perf-runbook.md`
5. `wiki/topics/mcts-perf-ceiling.md`
6. `wiki/topics/ane-int8-inference.md` for Core ML / engine split work
7. `wiki/ops/baselines.md`

## Measurement Rules

- Score by wall-clock, games/sec, positions/sec, eval time, and strength/quality guardrails — not by Activity Monitor GPU percent alone.
- Use same-shape baseline/candidate commands.
- Record hardware, backend (`mps`, `cpu`, Core ML mode), env flags, model size, stem padding, sims, wave size, workers, repeats, and live-run contention.
- Short evals are noisy. If strength matters, use fixed external baselines or archive metrics and cite uncertainty.
- Do not claim a perf win unless the receipt includes paired evidence or an explicit reason pairing is impossible.

## Standard Commands

CPU syntax smoke:

```bash
python scripts/perf_microbench.py --device cpu --size tiny --games 2 --n-simulations 2 --wave-size 1 --max-plies 2 --repeats 1 --warmup 0
```

MPS production-shaped microbench:

```bash
python scripts/perf_microbench.py --device mps --size small --stem-padding 1 --games 8 --n-simulations 400 --wave-size 64 --max-plies 16 --repeats 3
```

Correctness gate:

```bash
pytest -q
```

## Common Pitfalls

- Re-porting MCTS storage ideas already present in `gomoku/mcts.py`.
- Believing single-process speedups without a production-shaped worker-count check.
- Using `torch.compile` without considering worker weight reload cadence.
- Treating Core ML raw latency as the only engine-isolation metric; trainer contention may matter more.
- Promoting faster self-play that damages plies, archive metrics, or fixed-baseline strength.
