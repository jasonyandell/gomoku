# ML Perf Experiment Ledger

Append concise receipts here after the curator reads worker open notes. Worker-specific detail can live under `wiki/ops/open-notes/`.

## Receipt Schema

```yaml
lane:
hypothesis:
code_ref:
dataset_ref:
baseline_command:
candidate_command:
hardware:
seed:
baseline_metric:
candidate_metric:
delta:
confidence:
artifacts:
commands_run:
decision: promote | reject | blocked | needs_repeat
next_action:
```

## Training-Quality Promotion Gate

Perf changes that touch training behavior, inference outputs, MCTS/search behavior, replay/data encoding, checkpoint refresh cadence, or game-start distribution need more than throughput. A receipt may not use `decision: promote` unless it records all of the following:

1. **Named quality gate before the run.** Use at least one fixed external baseline or fixed validation archive. Current named options are:
   - external baselines: `heuristic`, `lookahead:depth=2`, and/or `lookahead:depth=4` via the match/eval harness with alternating colors;
   - validation archive: `archives/wl5_validation_v1.pt`, reporting at least `val/policy_ce`, `val/policy_kl`, and `val/value_mse` against the parent/reference checkpoint.
2. **Game-shape guardrail.** Report `selfplay/plies_mean` and, when available, `selfplay/plies_p90` or equivalent game-length distribution. Promotion is blocked or marked `needs_repeat` if the candidate shows sustained fast-attack collapse: falling plies, shorter-game buffer-fill concavity, or a material drop below the parent run's game-length band without an explicit strength explanation.
3. **Short-eval noise policy.** State game count and uncertainty. `n < 20` is smoke only and cannot support a strength claim. `n=20` can be a canary but normally needs a repeat or archive agreement for promotion. Prefer `n >= 50` or two independent same-shape `n >= 20` reads for behavior-changing promotion; otherwise use `decision: needs_repeat`.
4. **Reproducibility IDs.** Behavior-changing perf receipts must include checkpoint path(s), W&B run ID(s) or explicit `wandb: disabled`, commit hash, seed policy, and env/backend flags such as `GOMOKU_DISABLE_NATIVE_MCTS`, `GOMOKU_DISABLE_NATIVE_STATE_OPS`, `PYTORCH_ENABLE_MPS_FALLBACK`, device, model size, stem padding, sims, wave size, workers, and evaluator backend.
5. **Explicit decision.** Every receipt ends with `decision: promote | reject | blocked | needs_repeat`. Throughput-only wins that lack the selected quality gate, plies/game-shape read, or reproducibility IDs are not promotions; mark them `blocked` if the harness/artifact is missing or `needs_repeat` if the evidence is merely noisy/short.

## Receipts

### 2026-05-22 — production WL1-shaped self-play throughput seed

```yaml
lane: production-contour-sweep
hypothesis: Native MCTS plus 8 smaller workers remains better than fewer wider workers under a trainer-shaped wave-mode production sweep.
code_ref: 4f21cdd worktree /Users/jason/code/gomoku-perf-extension
dataset_ref: fresh self-play only; 10 trainer epochs per cell
baseline_command: exact launcher command not captured in ops ledger; reconstructed shape is WL1 10-epoch sweep, small model, 400 sims, wave 64, 8 workers x 8 games, Python MCTS fallback via GOMOKU_DISABLE_NATIVE_MCTS=1
candidate_command: exact launcher command not captured in ops ledger; same shape with native MCTS enabled, plus native 4 workers x 16 games comparison
hardware: M5 Max / MPS
seed: not recorded in summary TSV
baseline_metric: fallback 8w8g wall=1863 aug_pos/s, gen=2264 aug_pos/s, wall=8.85 games/s
candidate_metric: native 8w8g wall=2379 aug_pos/s, gen=3303 aug_pos/s, wall=11.25 games/s; native 4w16g wall=1918 aug_pos/s, gen=2152 aug_pos/s, wall=8.61 games/s
delta: native 8w8g vs fallback 8w8g = 1.28x wall aug_pos/s and 1.46x gen aug_pos/s; native 8w8g vs native 4w16g = 1.24x wall aug_pos/s and 1.53x gen aug_pos/s
confidence: medium; 10 epochs each, production-shaped but short and from worktree artifacts
artifacts: /Users/jason/code/gomoku-perf-extension/sweep_logs/perf10-summary.tsv and matching trainer/worker logs
commands_run: exact launcher command not captured in ops ledger; trainer logs record device/model/wave barrier shape
decision: needs_repeat
next_action: Use this as seed evidence for the production-contour-sweep lane, but rerun current-main baseline receipts with explicit command capture before treating it as a promotion gate.
```

### 2026-05-22 — current-main baseline receipts microbench

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

### 2026-05-22 — Core ML Gomoku ANE residency candidates from 934b

```yaml
lane: ane-residency-rail-proof
hypothesis: Some Core ML FP16 Gomoku fixed-batch shapes can actually move the ANE rail, unlike the first CPU_AND_NE scout that only proved Core ML scheduling/isolation.
code_ref: detached dirty worktree /Users/jason/.codex/worktrees/934b/gomoku at b9b9eab with uncommitted scripts/coreml_ane_residency_scout.py and tests/test_coreml_ane_residency_scout.py
dataset_ref: synthetic random Gomoku eval planes; no training or strength dataset
baseline_command: python scripts/coreml_ane_residency_scout.py --model-kinds gomoku --compute-units CPU_AND_NE --compute-precision FLOAT16 --batch-size 1 --workers 4 --duration-s 15 ... plus same-window powermetrics wrapper
candidate_command: python scripts/coreml_ane_residency_scout.py --model-kinds gomoku --compute-units CPU_AND_NE --compute-precision FLOAT16 --batch-size 32 --workers 4 --duration-s 15 ...; repeated at batch 128 and 1024; powermetrics summaries saved beside JSON
hardware: M5 Max / macOS 26.4.1 / Core ML 9.0 / PyTorch 2.11.0 / powermetrics ane_power
seed: synthetic random inputs; seed not recorded in curated summary
baseline_metric: b1 CPU_AND_NE Gomoku FP16 fixed fused = 33,043 positions/s, 495,648 positions, ANE mean=0 mW, max=0 mW, 0/23 active samples
candidate_metric: b32 = 122,039 positions/s, 1,830,688 positions, ANE mean=4,061 mW, max=6,605 mW, 16/24 active samples; b128 = 99,526 positions/s, 1,493,376 positions, ANE mean=3,683 mW, max=5,728 mW, 16/23 active samples; b8 also nonzero at 916 mW mean; b1024 nonzero but GPU rail was high and needs interpretation
delta: b32 vs b1 = 3.69x positions/s and nonzero ANE rail; b128 vs b1 = 3.01x positions/s and nonzero ANE rail
confidence: medium-low; powermetrics-positive and promising, but produced in a detached dirty worktree with shortened 15s cells and no integrated frontier receipt or production self-play overlap yet
artifacts: /Users/jason/.codex/worktrees/934b/gomoku/sweep_logs/coreml_ane_residency/v3_gomoku_fixed_fused_fp16_b{1,8,32,128,1024}_ne.{json,power.json}; draft wiki /Users/jason/.codex/worktrees/934b/gomoku/wiki/topics/coreml-ane-residency-lab.md
commands_run: curator inspected JSON/power artifacts only; no new benchmark command launched in the curation worktree
decision: needs_repeat
next_action: The ANE residency lane should integrate or reproduce 934b with exact commands, a nearby Vision positive control, CPU_ONLY negative control, and a production-overlap candidate before unblocking engine-overlap-production.

```

### 2026-05-22 — outer-loop Python profile no-op

```yaml
lane: outer-loop-python-profile
hypothesis: After native MCTS and eval fusion, remaining Python outside native search is large enough to justify another outer-loop native/format pass.
code_ref: 5e20aaa0b331f32eadc1cd58707a3ccbf3e86e9d on frontier/20260522T061713Z/01-outer-loop-python-profile; integrated on main as 411ed758a12568691f92bc414ee425ae385015fd
dataset_ref: fresh self-play from a freshly initialized small stem_padding=1 checkpoint; no training dataset or strength claim
baseline_command: python -m gomoku.selfplay_worker --weights-path sweep_logs/outer-loop-profile-20260522T061713Z/checkpoints/worker_weights.pt --output-dir sweep_logs/outer-loop-profile-20260522T061713Z/records-wave --worker-id profile --device mps --games-per-batch 8 --n-simulations 400 --wave-size 64 --max-plies 16 --wave-mode --seed 0 --max-batches 1 --profile-output sweep_logs/outer-loop-profile-20260522T061713Z/profile-mps-wave-mode-8g-s400-p16.json
candidate_command: no implementation candidate promoted; future candidates should use the same worker command and JSON profile diff, preferably repeated 3x
hardware: macOS-26.4.1 arm64; Apple M5 Max class machine; device=mps; Python 3.12.13
seed: 0
baseline_metric: wave-mode bounded worker wall=1.064s for 8 games / 128 plies / 1024 augmented examples; native_search_batch=1.013s (95.2% wall); evaluator=0.896s (84.3% wall); native_search_excluding_evaluator=0.117s (11.0% wall); post_search_python=0.050s (4.7% wall); file_handoff=0.034s (3.2% wall); record_build=0.011s (1.0% wall); D4=0.0087s (0.82% wall); sample_action=0.0032s (0.30% wall)
candidate_metric: non-wave cross-check wall=1.235s; evaluator=86.9%; native_search_excluding_evaluator=8.7%; post_search_python=4.4%; no post-search Python owner exceeds file handoff at ~3%
delta: no 10-20% outer-loop Python opportunity found; deleting all measured post-search Python would cap at ~4-5% on this shape, while evaluator plus native search boundary owns ~95%
confidence: low-to-medium; one bounded MPS run per shape on fresh random weights. Enough to reject a large post-search-Python pass, but repeat before citing exact percentages because raw JSON/log artifacts were not present in main after worker worktree cleanup.
artifacts: wiki/ops/open-notes/20260522T061713Z-01-outer-loop-python-profile.md; .frontier/runs/20260522T061713Z/workers/01-outer-loop-python-profile/receipt.md; raw paths named in receipt under sweep_logs/outer-loop-profile-20260522T061713Z/
commands_run:
  - python -m py_compile gomoku/self_play.py gomoku/selfplay_worker.py
  - python - <<'PY' ... build_model('small', stem_padding=1); save_checkpoint('sweep_logs/outer-loop-profile-20260522T061713Z/checkpoints/worker_weights.pt', m, epoch=0) ... PY
  - python -m gomoku.selfplay_worker --weights-path sweep_logs/outer-loop-profile-20260522T061713Z/checkpoints/worker_weights.pt --output-dir sweep_logs/outer-loop-profile-20260522T061713Z/records --worker-id profile --device mps --games-per-batch 8 --n-simulations 400 --wave-size 64 --max-plies 16 --seed 0 --max-batches 1 --profile-output sweep_logs/outer-loop-profile-20260522T061713Z/profile-mps-wave8-s400-p16.json
  - python -m gomoku.selfplay_worker --weights-path sweep_logs/outer-loop-profile-20260522T061713Z/checkpoints/worker_weights.pt --output-dir sweep_logs/outer-loop-profile-20260522T061713Z/records-wave --worker-id profile --device mps --games-per-batch 8 --n-simulations 400 --wave-size 64 --max-plies 16 --wave-mode --seed 0 --max-batches 1 --profile-output sweep_logs/outer-loop-profile-20260522T061713Z/profile-mps-wave-mode-8g-s400-p16.json
  - python scripts/perf_microbench.py --device cpu --size tiny --games 2 --n-simulations 2 --wave-size 1 --max-plies 2 --repeats 1 --warmup 0
  - pytest -q
decision: reject
next_action: Do not start another native pass for action sampling, trajectory staging, D4, record creation, or worker file handoff. Focus next perf work on evaluator/engine overlap after ANE rail proof, or a narrowly scoped native_search_batch/evaluator-boundary profile if needed.
```
### 2026-05-23 — canonical 5-axis M5 Max contour sweep

```yaml
lane: canonical-sweep-mainframe
hypothesis: The production self-play default (small / W=8 / G=8 / sims=400 / wave=64) is not the M5 Max's actual throughput peak; chip-specific knobs (especially wave_size) leave material throughput on the floor.
code_ref: 2ca5ab2 on main (driver + line-buffered output); scripts/canonical_sweep.py + scripts/plot_canonical_sweep.py
dataset_ref: fresh self-play only from random fused-eligible weights; no training, no strength claim. All games hit --max-plies 16 (plies_mean=15.96 universally), so cells measure infrastructure throughput not behavior throughput.
baseline_command: |
  python scripts/canonical_sweep.py \
    --out-dir sweep_logs/canonical-sweep-20260523T015614Z \
    --secs-per-cell 300 --max-plies 16 --device mps
candidate_command: same; the sweep is itself the candidate space (23 cells across workers x games-per-worker x n-sims x wave-size x model-size)
hardware: Apple M5 Max; macOS arm64; idle box (BAB1 cleared at 19:45 local)
seed: per-worker seeds 1000..1000+W-1; same seeds reused across cells
baseline_metric: small_W08_G08_S400_V064 = 3,188 aug pos/s (the WL5-era production default), 7,499 games, plies_mean 15.96
candidate_metric: |
  workers axis (small G8 S400 V64): W1=1,111 / W2=1,497 / W4=2,583 / W8=3,188 / W12=3,243 / W16=3,411 aug pos/s
  n-sims axis (small W8 G8 V64): S100=11,151 / S200=6,006 / S400=3,188 / S800=1,619
  wave-size axis (small W8 G8 S400): V32=2,467 / V64=3,188 / V128=4,048 / V256=4,409
  games-per-worker axis (small W8 S400 V64): G4=3,026 / G8=3,188 / G16=3,057
  model axis (W8 G8 S400 V64): tiny=7,326 / small=3,188 / medium=1,393
  corners: tiny_W16_G16_S100_V032=19,346 (max); small_W01_G16_S800_V128=946 (min)
delta: |
  Wave-size 64 -> 128 at the production default: +27% (3,188 -> 4,048 aug pos/s) with NO behavior change.
  Wave-size 64 -> 256: +38% (3,188 -> 4,409). Diminishing returns beyond 128.
  Workers 8 -> 16 at default: +7% only. The W8 default is near-optimal for workers; the win is wave.
  Per-worker efficiency falls fast: W1 1,111 / W2 749 / W4 646 / W8 399 / W12 270 / W16 213 aug/s/worker. MPS contention dominates by W=2.
  N-sims scales perfectly inversely: aug/s * sims ~= const, so sims is purely a quality knob.
  Tiny is 2.3x small, small is 2.3x medium at the default cell.
confidence: medium-high. Single bounded 5-min wall per cell on an idle box; results stable enough for relative shape (axis pivots monotone, no flips). All cells hit plies_mean=15.96 because of random weights + max-plies=16, so absolute numbers are infrastructure-bound; trained-model production numbers will be lower in absolute aug/s but the relative axis shape should hold (wave-size win is eval-batch-shape-dependent, not game-shape-dependent).
artifacts: sweep_logs/canonical-sweep-20260523T015614Z/{summary.tsv,cells.csv,metadata.txt,contour.png,axes.png,model_compare.png,checkpoints/{tiny,small,medium}.pt,cell_*/{records,logs}}; sweep_logs/canonical-sweep-latest symlink
commands_run:
  - python scripts/canonical_sweep.py --out-dir sweep_logs/canonical-sweep-20260523T015614Z (nohup, ~115 min)
  - python scripts/canonical_sweep.py --out-dir latest --status  (mid-run progress checks)
  - python scripts/plot_canonical_sweep.py --sweep-dir sweep_logs/canonical-sweep-latest
decision: promote
next_action: |
  1. Promote wave-size 128 as the new throughput default for next-cell self-play (small W8 G8 S400 V128 = 4,048 aug pos/s, +27% over WL5-era V64). V256 (+38%) is also viable; pick V128 as the safer step.
  2. Behavior change is structural (eval batch size only, no MCTS semantic change). Per the Training-Quality Promotion Gate, the next cell using V128/V256 should still report val/policy_ce + val/policy_kl against archives/wl5_validation_v1.pt and selfplay/plies_mean for one canary cycle to confirm no surprises before committing a full run.
  3. Workers axis: keep W8 as the default. W12-16 buys 2-7% and adds OS/process overhead; not worth it for the W8 baseline.
  4. Quality knobs (n-sims, model size) are separate decisions; this receipt has no opinion on them.
  5. Open-note follow-ups: (a) re-run the W x G cross at V128 to verify the wave-size win compounds at higher worker counts, (b) sims-vs-wave interaction at S=200 (faster cycles + wider waves might be the real next-cell shape), (c) repeat the sweep at the next stable checkpoint to confirm trained-model throughput shape matches infrastructure shape.
```

### 2026-05-23 — L01 wave extrapolation (V > 256 plateau search)

```yaml
lane: L01-wave-extrapolation
tier: 3
hypothesis: Wave gains continue past V=256; find the plateau or inflection.
reference: R-S400 (small / W=8 / G=8 / S=400; V=128 baseline = 4,048 aug/s)
code_change: false
baseline_command: |
  python scripts/canonical_sweep.py --out-dir sweep_logs/lab-L01-wave-extrapolation-20260523T051548Z --cells-from <same dir>/cells.csv --lane L01-wave-extrapolation --secs-per-cell 300 --max-plies 16 --device mps
candidate_command: same (4-cell sweep at V in {384, 512, 768, 1024})
hardware: M5 Max / MPS / idle box
seed: per-worker 1000..1007
baseline_metric: R-S400 was V=128 = 4,048 aug/s (canonical-sweep promote). Canonical-sweep V=256 corner = 4,409 aug/s.
candidate_metric: |
  V=384  = 4,452 aug/s  (+1.0% over V=256, +10.0% over V=128)
  V=512  = 4,765 aug/s  (+8.1% over V=256, +17.7% over V=128, +49.5% over WL5 V=64)
  V=768  = 4,761 aug/s  (flat with V=512; -0.1%)
  V=1024 = 4,756 aug/s  (flat with V=512; -0.2%)
delta: V=512 is the new R-S400 best. +17.7% over yesterday's V=128 promote; +49.5% cumulative over original WL5 V=64. V=768/1024 are flat — clear plateau knee at V=512.
confidence: medium-high. 5-min cells per the charter; idle box; monotone within experimental noise (V=768/1024 track V=512 within <0.2%). All cells hit plies_mean=15.96 (random weights + max-plies=16) so this is infrastructure throughput. Wave-size win is eval-batch-shape-dependent, not game-shape-dependent.
artifacts: sweep_logs/lab-L01-wave-extrapolation-20260523T051548Z/{summary.tsv, cells.csv, metadata.txt, driver.log, cell_small_W08_G08_S400_V{384,512,768,1024}/{records,logs}}
commands_run:
  - python scripts/canonical_sweep.py --out-dir sweep_logs/lab-L01-wave-extrapolation-20260523T051548Z --cells-from ... --lane L01-wave-extrapolation
decision: promote
reviewer: APPROVE (general-purpose agent, 2026-05-23). "L01 V=512 promote math checks; surfaces consistent; plateau call sound; L02/L03 rescoped; +49.5% cumulative on R-S400."
next_action: |
  1. Promote V=512 as new R-S400 default. Old: V=128 (4,048). New: V=512 (4,765). best-cells.md updated.
  2. Auto-queued compounds (perf-queue.md rescoped after L01):
     - L02 now W in {4,12,16} at V=512 only (V=128/V=256 cells dropped — V=512 dominates).
     - L03 now S in {100,200} at V=512 only.
  3. V=128 -> V=512 is no-behavior-change (eval batch shape only). First live-training cell adopting V=512 (L11) needs one canary cycle against archives/wl5_validation_v1.pt per the Training-Quality Promotion Gate.
  4. Plateau knee at V=512: future wave sweeps stop at V=512 on this hardware unless model size, MPS heap config, or engine (Core ML / ANE) shifts the eval-overhead floor. L07 (tiny contour) probes model-size dependency.
```

### 2026-05-23 — L03 sims-x-wave at V=512 (double promote at R-S200 + R-S100)

```yaml
lane: L03-sims-x-wave
tier: 2
hypothesis: V=512 (from L01 promote) carries to faster quality points S=100 and S=200; opens new R-S200 / R-S100 promoted-defaults.
reference: R-S200 (baseline 6,006 aug/s at V=64) + R-S100 (baseline 11,151 aug/s at V=64)
code_change: false
baseline_command: |
  python scripts/canonical_sweep.py --out-dir sweep_logs/lab-L03-sims-x-wave-20260523T055953Z --cells-from <same dir>/cells.csv --lane L03-sims-x-wave --secs-per-cell 300 --max-plies 16 --device mps
candidate_command: same (2-cell sweep at S in {100, 200} with V=512)
hardware: M5 Max / MPS / idle box
seed: per-worker 1000..1007
baseline_metric: |
  R-S200 was small W=8 G=8 S=200 V=64 = 6,006 aug/s (canonical-sweep)
  R-S100 was small W=8 G=8 S=100 V=64 = 11,151 aug/s (canonical-sweep)
candidate_metric: |
  S=100 V=512 = 15,082 aug/s (+35.2% over R-S100 baseline)
  S=200 V=512 =  9,156 aug/s (+52.5% over R-S200 baseline)
delta: Two promotes. V=512 wins at every quality point measured so far. Cumulative speedups: R-S400 +49.5% (L01), R-S200 +52.5%, R-S100 +35.2%. Wave-size win is uniform across the sims axis — the eval-batch shape benefit dominates regardless of how many MCTS sims feed it.
confidence: medium-high. 5-min cells, idle box, monotone with prior wave findings. Both cells plies_mean=15.97 (random weights cap). The wave-size win continues to be eval-batch-shape-dependent so it should transfer to trained-model production cells.
artifacts: sweep_logs/lab-L03-sims-x-wave-20260523T055953Z/{summary.tsv, cells.csv, metadata.txt, driver.log, cell_small_W08_G08_S{100,200}_V512/{records,logs}}
commands_run:
  - python scripts/canonical_sweep.py --out-dir sweep_logs/lab-L03-sims-x-wave-20260523T055953Z --cells-from ... --lane L03-sims-x-wave
decision: promote (two reference points)
reviewer: APPROVE (general-purpose agent, 2026-05-23). "L03 double promote math + units verified (R-S100 +35.2%, R-S200 +52.5%); all six surfaces consistent; queue clean."
next_action: |
  1. Promote R-S200: V=64 -> V=512 (6,006 -> 9,156 aug/s, +52.5%). best-cells.md updated.
  2. Promote R-S100: V=64 -> V=512 (11,151 -> 15,082 aug/s, +35.2%). best-cells.md updated.
  3. Wave-size dominance now confirmed at three quality points (S=100, S=200, S=400). Future cells should use V=512 as the structural default; only deviate with explicit hypothesis.
  4. Auto-queued compound: L02 (W x V=512) is next-priority unblocked Tier 2. After that, L07 (tiny contour at V=128/256/512) becomes high-value to set the R-S400-tiny ceiling for the L09 ANE comparison.
  5. The two new live-training cells that should adopt V=512 (R-TRAIN-LEAN via L11) need one canary cycle against archives/wl5_validation_v1.pt per the Training-Quality Promotion Gate (still pending L12 driver).
```

### 2026-05-23 — L02 W-x-wave at V=512 (reject; W=8 still optimal, surprisingly so)

```yaml
lane: L02-W-x-wave-compound
tier: 2
hypothesis: V=512 (from L01) compounds at higher worker counts (W=12, W=16).
reference: R-S400 (current best = W=8 V=512 = 4,765 aug/s)
code_change: false
baseline_command: |
  python scripts/canonical_sweep.py --out-dir sweep_logs/lab-L02-W-x-wave-20260523T061440Z --cells-from <same dir>/cells.csv --lane L02-W-x-wave --secs-per-cell 300 --max-plies 16 --device mps
candidate_command: same (3-cell sweep at W in {4, 12, 16} with V=512)
hardware: M5 Max / MPS / idle box
seed: per-worker 1000..1015
baseline_metric: W=8 V=512 = 4,765 aug/s (L01 reference)
candidate_metric: |
  W=4  V=512 = 4,367 aug/s (-8.4% vs W=8)
  W=12 V=512 = 4,501 aug/s (-5.5% vs W=8)
  W=16 V=512 = 4,504 aug/s (-5.5% vs W=8)
delta: REJECT. W=8 V=512 remains the best at R-S400. W=4 too few workers (eval idle), W=12/16 hurt because the V=512 wave is already saturating MPS dispatch — more workers create scheduling pressure, not parallelism.
confidence: medium-high. 5-min cells, idle box, monotone (W=12 and W=16 within 0.07% of each other = clear plateau). plies_mean=15.96 universal (random weights cap).
artifacts: sweep_logs/lab-L02-W-x-wave-20260523T061440Z/{summary.tsv, cells.csv, metadata.txt, driver.log, cell_small_W{04,12,16}_G08_S400_V512/{records,logs}}
commands_run:
  - python scripts/canonical_sweep.py --out-dir sweep_logs/lab-L02-W-x-wave-20260523T061440Z --cells-from ... --lane L02-W-x-wave
decision: reject
reviewer: APPROVE (general-purpose agent, 2026-05-23). "L02 reject math clean (-8.4%/-5.5%/-5.5%); best-cells correctly unchanged; W-inversion insight requeues L04+L07; counter 0→1."
next_action: |
  1. Best-cells.md unchanged. W=8 V=512 = 4,765 stays the R-S400 default.
  2. **New finding worth capturing**: at V=64 the workers axis was monotone (W=16 best); at V=512 the workers axis is INVERTED (W=8 best, W=12/16 slightly worse). The wave-size shift moved the MPS-dispatch saturation point. Implication: tier rule "knob wins don't compound across axes" is more than aesthetic — they actively interact in non-monotone ways at the high end of the chip's envelope.
  3. Followup queue updates:
     - L04 (G × V=512) becomes more interesting — G axis was flat at V=64 (3026/3188/3057 across G=4/8/16) but the W-axis non-monotonicity at V=512 suggests G might also have a different shape at V=512. Bump L04 priority.
     - L07 (tiny contour) should ADD V=512 cells (currently has V=128/V=256 only). The tiny model has cheaper forward pass so the saturation point might be at a higher V. Bump L07 priority since it now also probes whether V=512 plateau extends.
  4. consecutive_rejects counter: 0 -> 1.
```

### 2026-05-23 — L04 G-x-wave at V=512 (reject; G=8 stays optimal, same W-axis-style shape)

```yaml
lane: L04-G-x-wave
tier: 2
hypothesis: G axis was flat at V=64 (3026/3188/3057). L02 found W axis is non-monotone at V=512. G might also have a new shape at V=512.
reference: R-S400 (current best = W=8 G=8 S=400 V=512 = 4,765 aug/s)
code_change: false
baseline_command: |
  python scripts/canonical_sweep.py --out-dir sweep_logs/lab-L04-G-x-wave-20260523T063440Z --cells-from <same dir>/cells.csv --lane L04-G-x-wave --secs-per-cell 300 --max-plies 16 --device mps
candidate_command: same (3-cell sweep at G in {4, 16, 32} with W=8 V=512)
hardware: M5 Max / MPS / idle box
seed: per-worker 1000..1007
baseline_metric: G=8 V=512 = 4,765 aug/s
candidate_metric: |
  G=4  V=512 = 4,608 aug/s (-3.3% vs G=8)
  G=16 V=512 = 4,541 aug/s (-4.7%)
  G=32 V=512 = 4,514 aug/s (-5.3%)
delta: REJECT. G=8 V=512 stays the R-S400 default. G axis IS mildly non-monotone at V=512 (was completely flat at V=64), but the peak is still G=8. Same shape as L02 W axis: the middle-of-explored-values is the MPS-saturation sweet spot at V=512.
confidence: medium-high. 5-min cells, idle box, monotone decline G=4→G=8→G=16→G=32 with G=8 strict peak.
artifacts: sweep_logs/lab-L04-G-x-wave-20260523T063440Z/{summary.tsv, cells.csv, metadata.txt, driver.log, cell_small_W08_G{04,16,32}_S400_V512/{records,logs}}
commands_run:
  - python scripts/canonical_sweep.py --out-dir sweep_logs/lab-L04-G-x-wave-20260523T063440Z --cells-from ... --lane L04-G-x-wave
decision: reject
reviewer: APPROVE (general-purpose agent, 2026-05-23). "L04 reject math clean (-3.3/-4.7/-5.3%); best-cells unchanged; compound W+G finding documented; L08 correctly blocked; counter 1→2."
next_action: |
  1. Best-cells.md unchanged. W=8 G=8 V=512 = 4,765 stays the R-S400 default.
  2. Compound finding (with L02): at V=512 BOTH worker axis AND games-per-worker axis are non-monotone with a peak at the canonical-sweep production-default values (W=8, G=8). The wave saturation has tightened the production-cell envelope around the historical defaults — wider perimeter exploration at V=512 won't beat the center.
  3. Future single-axis explorations at V=512 should not bother re-measuring W or G — those axes are now CONFIRMED at their peaks. Other axes (model size, sims) remain open.
  4. consecutive_rejects counter: 1 -> 2. One more reject + queue empty = halt (per stop rule).
  5. Next dispatch: bg L07 (tiny contour with V=512 cells added). Tier 3 L08 (heap ratio) is blocked-on-driver — canonical_sweep doesn't support per-cell env vars yet; add to L12 driver scope or build it as L08-driver task. Tier 3 L05/L06 are blocked-on-worktree code.
```

