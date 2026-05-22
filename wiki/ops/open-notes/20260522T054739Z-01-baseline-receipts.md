# 20260522T054739Z-01-baseline-receipts

## Receipt

```yaml
lane: baseline-receipts
hypothesis: Current-main native MCTS remains materially faster than the Python MCTS fallback on the standard production-shaped MPS microbench, and the raw-output artifact convention is sufficient for future citation.
code_ref: a418f677b831488a71333a3e60d3a0ca7108dbfc on frontier/20260522T054739Z/01-baseline-receipts; same commit as /Users/jason/code/gomoku main at measurement time
dataset_ref: fresh self-play microbench only; no training dataset; seed=0
baseline_command: GOMOKU_DISABLE_NATIVE_MCTS=1 python scripts/perf_microbench.py --device mps --size small --stem-padding 1 --games 8 --n-simulations 400 --wave-size 64 --max-plies 16 --repeats 3
candidate_command: python scripts/perf_microbench.py --device mps --size small --stem-padding 1 --games 8 --n-simulations 400 --wave-size 64 --max-plies 16 --repeats 3
hardware: MacBook Pro Mac17,6; Apple M5 Max; 18 cores (6 Super, 12 Performance); 48 GB; MPS; live WL5 trainer + 8 self-play workers + eval worker active
seed: 0
baseline_metric: fallback median 2.309s; 3.46 games/s; 443 aug pos/s; plies_mean 16.0; native_mcts=false; native_state_ops=true; fused_eval=true
candidate_metric: native median 0.626s; 12.79 games/s; 1,637 aug pos/s; plies_mean 16.0; native_mcts=true; native_state_ops=true; fused_eval=true
delta: native vs fallback = 3.69x lower median seconds and 3.70x higher games/s and aug_pos/s under paired live contention
confidence: medium; paired same-shape repeats on current main, but absolute MPS numbers are contended by live WL5 and should be repeated on an idle machine for stable reference rows
artifacts: sweep_logs/frontier-baselines/20260522T054845Z/{metadata.txt,commands.txt,summary.tsv,summary.json,cpu-smoke-native.txt,cpu-smoke-fallback.txt,mps-microbench-native.txt,mps-microbench-fallback.txt,pytest-q.txt}
commands_run:
  - python scripts/perf_microbench.py --device cpu --size tiny --games 2 --n-simulations 2 --wave-size 1 --max-plies 2 --repeats 1 --warmup 0
  - GOMOKU_DISABLE_NATIVE_MCTS=1 python scripts/perf_microbench.py --device cpu --size tiny --games 2 --n-simulations 2 --wave-size 1 --max-plies 2 --repeats 1 --warmup 0
  - python scripts/perf_microbench.py --device mps --size small --stem-padding 1 --games 8 --n-simulations 400 --wave-size 64 --max-plies 16 --repeats 3
  - GOMOKU_DISABLE_NATIVE_MCTS=1 python scripts/perf_microbench.py --device mps --size small --stem-padding 1 --games 8 --n-simulations 400 --wave-size 64 --max-plies 16 --repeats 3
  - pytest -q
decision: promote
next_action: Curator can treat this as the current-main contended baseline receipt and artifact convention; repeat the MPS pair when WL5 is idle if absolute comparison to the older 2,200 aug pos/s reference matters.
```

## Result

- CPU syntax smoke passed in both default-native and `GOMOKU_DISABLE_NATIVE_MCTS=1` modes.
- MPS production-shaped microbench passed in both default-native and fallback modes.
- Paired MPS result under live WL5 contention: native median 0.626s / 1,637 aug pos/s versus fallback median 2.309s / 443 aug pos/s, a 3.70x throughput ratio with equal plies_mean=16.0.
- `pytest -q` passed.

## Files touched

- `wiki/ops/baselines.md`
- `wiki/ops/test-ledger.md`
- `wiki/ops/experiment-ledger.md`
- `wiki/ops/open-notes/20260522T054739Z-01-baseline-receipts.md`
- External manager receipt: `/Users/jason/code/gomoku/.frontier/runs/20260522T054739Z/workers/01-baseline-receipts/receipt.md`

Ignored raw artifacts written under `sweep_logs/frontier-baselines/20260522T054845Z/`.

## Blockers / caveats

No harness blocker. Absolute MPS timings are contended because the main worktree had the WL5 trainer, 8 self-play workers, and eval worker active. The native/fallback comparison is still same-shape and paired under that contention.

## Board-update recommendation

Keep the baseline/test/experiment ledger rows added by this lane. If the curator wants an idle-machine reference baseline, rerun the same MPS native/fallback pair after WL5 or other long self-play jobs are stopped.
