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

### Canonical 5-axis sweep

```bash
python scripts/canonical_sweep.py --out-dir sweep_logs/canonical-sweep-$(date -u +%Y%m%dT%H%M%SZ)
# Defaults: --secs-per-cell 300 --max-plies 16 --device mps. 23 cells, ~2-3 h wall.
# Resumable: --status anytime, --retry-failed for transients, --max-wall-secs for top-ups.
# Plot when complete:
python scripts/plot_canonical_sweep.py --sweep-dir sweep_logs/canonical-sweep-latest
```

Purpose: produce a per-cell aug-positions/sec contour over workers × games-per-worker × n-simulations × wave-size × model-size, with each cell measured as a bounded multi-worker production-shape self-play batch. The canonical sweep replaces single-process intuition with a calibrated map for the specific M5 Max SKU. See [perf-lab-session-runbook](../topics/perf-lab-session-runbook.md) for the contract; [m5-max-as-mainframe](../topics/m5-max-as-mainframe.md) for the philosophy.

### Outer-loop worker profile

```bash
python -m gomoku.selfplay_worker \
  --weights-path sweep_logs/outer-loop-profile-20260522T061713Z/checkpoints/worker_weights.pt \
  --output-dir sweep_logs/outer-loop-profile-20260522T061713Z/records-wave \
  --worker-id profile --device mps --games-per-batch 8 \
  --n-simulations 400 --wave-size 64 --max-plies 16 \
  --wave-mode --seed 0 --max-batches 1 \
  --profile-output sweep_logs/outer-loop-profile-20260522T061713Z/profile-mps-wave-mode-8g-s400-p16.json
```

Purpose: separate evaluator/native-search time from post-search Python in a bounded worker-shaped self-play batch. Use as a verifier only; it is not a training-quality or strength benchmark.

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
| 2026-05-22 | `5e20aaa` worker / `411ed75` integrated | M5 Max class / MPS | outer-loop worker profile, wave-mode 8 games, 400 sims, wave 64, max 16 plies, seed 0 | wall-share owners | wall 1.064s; evaluator 0.896s / 84.3%; native search excl evaluator 0.117s / 11.0%; post-search Python 0.050s / 4.7%; file handoff 3.2%; D4 0.82%; action sampling 0.30% | `wiki/ops/open-notes/20260522T061713Z-01-outer-loop-python-profile.md` |
| 2026-05-23 | `2ca5ab2` main | M5 Max / MPS / idle | canonical sweep, small / W=8 / G=8 / sims=400 / wave=**64** / max-plies 16, 300s wall, fresh random fused-eligible weights | aug pos/s (infrastructure-bound, plies cap 16) | 3,188 aug pos/s; 7,499 games; plies_mean 15.96 | `sweep_logs/canonical-sweep-20260523T015614Z/{summary.tsv,axes.png,contour.png}` |
| 2026-05-23 | `2ca5ab2` main | M5 Max / MPS / idle | canonical sweep, same cell with wave=**128** (proposed throughput default) | aug pos/s (infrastructure-bound) | 4,048 aug pos/s; 9,531 games; plies_mean 15.96; **+27% vs V=64** | same dir, cell_small_W08_G08_S400_V128 |
| 2026-05-23 | `2ca5ab2` main | M5 Max / MPS / idle | canonical sweep, same cell with wave=**256** | aug pos/s (infrastructure-bound) | 4,409 aug pos/s; 10,379 games; plies_mean 15.96; **+38% vs V=64** | same dir, cell_small_W08_G08_S400_V256 |
| 2026-05-23 | `2ca5ab2` main | M5 Max / MPS / idle | canonical sweep, model axis at default W8 G8 S400 V64 | aug pos/s | tiny=7,326; small=3,188; medium=1,393 (≈2.3× per step) | same dir, cell_{tiny,small,medium}_W08_G08_S400_V064 |
| 2026-05-23 | `2ca5ab2` main | M5 Max / MPS / idle | canonical sweep, workers axis at small G8 S400 V64 | aug pos/s, per-worker efficiency | W1=1,111 / W2=1,497 / W4=2,583 / W8=3,188 / W12=3,243 / W16=3,411; per-worker eff falls W1=1,111→W16=213 (contention dominates by W=2) | same dir |
| 2026-05-23 | `2ca5ab2` main | M5 Max / MPS / idle | canonical sweep, max-throughput corner cell | aug pos/s | tiny_W16_G16_S100_V32 = 19,346 (infrastructure ceiling; not a quality-comparable cell) | same dir, cell_tiny_W16_G16_S100_V032 |

## Engine-Isolation / Residency References

| Date | Worktree | Hardware | Command shape | Metric | Result | Artifact |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-05-22 | main/perf scout | M5 Max / MPS + Core ML | `python scripts/aggressive_engine_scout.py --size small --stem-padding 1 --batches 8,32,64,128 ...` | raw b128 eval + MPS trainer pressure | PyTorch/MPS raw b128 2.94ms / 43.5k pos/s; Core ML CPU_ONLY/CPU_AND_NE ~8.5-9.1ms / ~14-15k pos/s; PyTorch/MPS pressure slowed trainer 2.65x vs Core ML 1.13-1.32x | `sweep_logs/aggressive-engine-scout-2026-05-22.json` |
| 2026-05-22 | 934b detached dirty | M5 Max / Core ML + powermetrics | `python scripts/coreml_ane_residency_scout.py --model-kinds gomoku --compute-units CPU_AND_NE --compute-precision FLOAT16 --batch-size 32 --workers 4 --duration-s 15 ...` plus same-window powermetrics wrapper | Gomoku FP16 fixed fused b32 ANE rail | 122,039 positions/s; 1,830,688 total positions; 4 ready workers; ANE mean 4,061 mW, max 6,605 mW, 16/24 active samples | `/Users/jason/.codex/worktrees/934b/gomoku/sweep_logs/coreml_ane_residency/v3_gomoku_fixed_fused_fp16_b32_ne.{json,power.json}` |
| 2026-05-22 | 934b detached dirty | M5 Max / Core ML + powermetrics | same, batch size 128 | Gomoku FP16 fixed fused b128 ANE rail | 99,526 positions/s; 1,493,376 total positions; 4 ready workers; ANE mean 3,683 mW, max 5,728 mW, 16/23 active samples | `/Users/jason/.codex/worktrees/934b/gomoku/sweep_logs/coreml_ane_residency/v3_gomoku_fixed_fused_fp16_b128_ne.{json,power.json}` |
| 2026-05-22 | 934b detached dirty | M5 Max / Core ML + powermetrics | same, batch size 1 | Gomoku FP16 fixed fused b1 ANE rail negative | 33,043 positions/s; ANE 0 mW, 0/23 active samples | `/Users/jason/.codex/worktrees/934b/gomoku/sweep_logs/coreml_ane_residency/v3_gomoku_fixed_fused_fp16_b1_ne.{json,power.json}` |

## Notes

- Same-shape comparisons beat isolated intuition.
- Record env flags such as `GOMOKU_DISABLE_NATIVE_MCTS=1`, `GOMOKU_DISABLE_NATIVE_STATE_OPS=1`, and `PYTORCH_ENABLE_MPS_FALLBACK=1`.
- If a benchmark contends with a live training run, say so.
- Current frontier artifact convention: raw command output under `sweep_logs/frontier-baselines/<timestamp>/`, plus a small `summary.tsv` or `summary.json` for ledger ingestion.
- Outer-loop profile raw JSON/log paths are recorded in the worker receipt, but the durable curated citation is the open note. If exact JSON is needed, rerun the bounded worker profile command and keep the `sweep_logs/outer-loop-profile-*` directory outside the removed worker worktree.
