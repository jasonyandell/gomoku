# 20260522T054739Z-02 production-contour-sweep

## Receipt

```yaml
lane: production-contour-sweep
hypothesis: On this M5 Max, the production-shaped WL1/WL5 self-play contour still favors native MCTS with more narrow workers (8 workers x 8 games, wave 64) over fewer wider workers or Python MCTS fallback; lower sims or tiny models are throughput levers but need training-quality gates before production use.
code_ref: a418f677b831488a71333a3e60d3a0ca7108dbfc on branch frontier/20260522T054739Z/02-production-contour-sweep; native extensions built locally with `python setup.py build_ext --inplace`
dataset_ref: fresh random-weight self-play only; 2-3 short wave-lockstep trainer-shaped epochs per cell; measurements ran while the main /Users/jason/code/gomoku WL5 trainer + 8 workers + eval worker were active on the same M5 Max
baseline_command: `python sweep_logs/production-contour-20260522/run_production_contour.py` cell `fallback_8w8g_s400_wave64_small`; exact trainer/worker commands in `sweep_logs/production-contour-20260522/fallback_8w8g_s400_wave64_small/commands.json`
candidate_command: `python sweep_logs/production-contour-20260522/run_production_contour.py` cells `native_8w8g_s400_wave64_small`, `native_4w16g_s400_wave64_small`, `native_4w8g_s400_wave64_small`, `native_8w4g_s400_wave64_small`, `native_8w8g_s200_wave64_small`, `native_8w8g_s400_wave32_small`, `native_8w8g_s400_wave64_tiny`; exact commands in each cell's `commands.json`
hardware: M5 Max MacBook Pro, 18 CPU cores, 40-core GPU, 48 GB unified memory, MPS; live WL5 contention present
seed: worker seeds 2026052200..2026052207 per cell; fresh random model init
baseline_metric: fallback 8w8g small sims400 wave64 = 2 epochs, 250 games, plies_mean 33.5, wall 2.58 games/s, gen 7.37 games/s, wall 690 aug_pos/s, gen 1975 aug_pos/s
candidate_metric: native 8w8g small sims400 wave64 = 3 epochs, 354 games, plies_mean 31.3, wall 4.96 games/s, gen 10.83 games/s, wall 1244 aug_pos/s, gen 2712 aug_pos/s; native 4w16g same = wall 4.40 games/s, wall 1099 aug_pos/s; native 8w4g same = wall 4.86 games/s, wall 1234 aug_pos/s; native wave32 same = wall 4.63 games/s, wall 1085 aug_pos/s; native sims200 = wall 7.09 games/s, wall 1774 aug_pos/s; native tiny = wall 8.23 games/s, wall 2126 aug_pos/s
delta: native 8w8g vs fallback 8w8g = +92.6% wall games/s and +80.1% wall aug_pos/s (+46.8% gen games/s, +37.3% gen aug_pos/s); native 8w8g vs native 4w16g = +12.8% wall games/s and +13.2% wall aug_pos/s; native 8w8g vs wave32 = +7.2% wall games/s and +14.6% wall aug_pos/s; 8w4g beats 4w8g by +12.0% wall games/s and +14.8% wall aug_pos/s at the 32-game tile scale
confidence: medium for the worker-count/games-per-worker contour because this repeats perf10's ordering under exact command capture, but absolute rates are low/noisy due to live WL5 contention and only 2-3 epochs per cell; low for sims/model-size promotion because strength/quality gates were not run
artifacts: `sweep_logs/production-contour-20260522/summary.tsv`, `sweep_logs/production-contour-20260522/summary.json`, per-cell `trainer.log`, `w*.log`, `commands.json`, `result.json`, `manifest.json`, and `mps_microbench_live_contention.log`
commands_run:
  - `python setup.py build_ext --inplace`
  - `python scripts/perf_microbench.py --device cpu --size tiny --games 2 --n-simulations 2 --wave-size 1 --max-plies 2 --repeats 1 --warmup 0`
  - `pytest -q tests/test_native_mcts.py tests/test_mcts.py tests/test_model.py`
  - `python sweep_logs/production-contour-20260522/run_production_contour.py`
  - `python scripts/perf_microbench.py --device mps --size small --stem-padding 1 --games 8 --n-simulations 400 --wave-size 64 --max-plies 16 --repeats 3 | tee sweep_logs/production-contour-20260522/mps_microbench_live_contention.log`
  - `pytest -q`
decision: promote
next_action: Keep native small 8w8g wave64 as the production default over 4w16g/fallback/wave32; do not promote sims200 or tiny without a quality/strength gate. Repeat only the top 2-3 cells after WL5 is idle if absolute wall-clock rates are needed.
```

## Result table

| cell | wall games/s | gen games/s | wall aug pos/s | gen aug pos/s | plies_mean | extras | read |
|---|---:|---:|---:|---:|---:|---:|---|
| fallback_8w8g_s400_wave64_small | 2.58 | 7.37 | 690 | 1975 | 33.5 | 12 | baseline; Python MCTS fallback loses badly under the same 8x8 shape |
| native_8w8g_s400_wave64_small | 4.96 | 10.83 | 1244 | 2712 | 31.3 | 27 | best same-quality small/s400 production cell |
| native_4w16g_s400_wave64_small | 4.40 | 8.09 | 1099 | 2020 | 31.2 | 20 | fewer wider workers still lose |
| native_4w8g_s400_wave64_small | 4.34 | 7.10 | 1075 | 1760 | 31.0 | 5 | 32-game tile, fewer workers |
| native_8w4g_s400_wave64_small | 4.86 | 10.31 | 1234 | 2620 | 31.8 | 25 | 32-game tile, more workers; confirms worker count matters |
| native_8w8g_s200_wave64_small | 7.09 | 26.44 | 1774 | 6612 | 31.3 | 22 | throughput win from lower sims; quality unproven |
| native_8w8g_s400_wave32_small | 4.63 | 8.25 | 1085 | 1932 | 29.3 | 22 | wave64 beats wave32 in this contour |
| native_8w8g_s400_wave64_tiny | 8.23 | 18.83 | 2126 | 4863 | 32.3 | 35 | model-size speed lever; training quality/capacity unproven |

## Files touched

- Added this open note: `wiki/ops/open-notes/20260522T054739Z-02-production-contour-sweep.md`.
- Wrote ignored artifacts under `sweep_logs/production-contour-20260522/` and `sweep_runs/production-contour-20260522/`.
- Built ignored native extension binaries under `gomoku/*.so` and `build/` for this worktree.
- External manager receipt to be written at `/Users/jason/code/gomoku/.frontier/runs/20260522T054739Z/workers/02-production-contour-sweep/receipt.md`.

## Board-update recommendation

Curator should append the receipt above to `wiki/ops/experiment-ledger.md` and add the validation commands to `wiki/ops/test-ledger.md`. Suggested short board read: "production-contour-sweep promoted native small 8w8g wave64 as the current M5 Max default; 4w16g, 4w8g, and wave32 rejected for throughput; sims200/tiny need quality gates before use."
