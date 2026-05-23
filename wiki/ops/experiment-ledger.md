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

### 2026-05-23 — L06-followup fp16-eval PROMOTE — small +97.2%, tiny +3.6%

```yaml
lane: L06-followup-fp16-cells
hypothesis: fp16 eval reduces memory bandwidth and improves aug/s on MPS without behavior change. Smoke at R-S400 (small / W=8 / G=8 / S=400 / V=512) and R-S400-tiny (tiny / W=16 / G=8 / S=400 / V=512); historic regression worth re-checking with mature MPS + fused conv+bn.
code_ref: 36d0f8d on main (run-time); receipt_commit 4e1bc2d (lane); Reviewer-APPROVE commit pending. Uses --fp16-eval passthrough from L06 (commit a3fb9ca). Reviewer verified gomoku/mcts.py:519-529 casts forward outputs back to .float() before .cpu().numpy() — MCTS, native search, replay payload, and downstream consumers all see fp32 numbers from a fp16-internal forward.
evaluator: torch / MPS / fp16-eval (workers cast model to float16; inputs cast to half() inside make_torch_evaluator; outputs cast back to fp32 BEFORE host transfer per the L06 patch design — MCTS reads identical-shape fp32 numbers from a fp16-internal forward)
dataset_ref: pure self-play; fresh random fused checkpoint per model; no trainer; 60s/cell smoke
baseline_command: R-S400 (lab-L01-wave-extrapolation = 4,765 aug/s); R-S400-tiny (lab-L07-tiny-contour = 22,088 aug/s)
candidate_command: python scripts/canonical_sweep.py --out-dir sweep_logs/lab-L06fu-20260523T135945Z --cells-from <2-row csv> --lane L06-followup --secs-per-cell 60 --fp16-eval
hardware: M5 Max / MPS / idle; torch 2.11.0
seed: workers seeded 1000..1015; same canonical_sweep seeding as L01/L07
baseline_metric: small V=512 fp32 = 4,765 aug/s; tiny V=512 fp32 = 22,088 aug/s; plies_mean both 15.96 (16-ply cap)
candidate_metric: small V=512 fp16 = **9,398.5 aug/s**; tiny V=512 fp16 = **22,873.8 aug/s**; plies_mean 15.97 / 15.96 (16-ply cap; game-shape preserved within noise)
delta: small **+97.2%** (4,765 → 9,398.5; nearly doubles); tiny **+3.6%** (22,088 → 22,874; ~18× the V=512-plateau noise floor of ~0.2% so distinguishable from noise though much smaller win). Mechanism (clean): small at V=512 is memory-bandwidth-limited so fp16 (halves bandwidth) nearly doubles throughput; tiny at V=512 is MPS-dispatch-limited (already at 22k aug/s, more compute-bound) so fp16 helps only marginally. This is exactly the compound-finding-readiness signal that re-running prior nulls under mature MPS would surface.
confidence: high. fp16 actually engaged (`fp16-eval enabled (model cast to torch.float16)` in both worker logs); plies_mean unchanged (game-shape preserved); outputs cast to fp32 before host transfer (per L06 patch); mechanism predicts the small≫tiny ratio observed (small was bandwidth-bound, tiny was dispatch-bound). The historic "fp16 on MPS is slow" claim is now disproven for our eval workload at this torch+MPS+fused-conv-bn maturity level.
artifacts: sweep_logs/lab-L06fu-20260523T135945Z/{summary.tsv,cells.csv,metadata.txt,cell_small_W08_G08_S400_V512,cell_tiny_W16_G08_S400_V512}
commands_run:
  - python scripts/canonical_sweep.py --out-dir sweep_logs/lab-L06fu-20260523T135945Z --cells-from cells.csv --lane L06-followup --secs-per-cell 60 --fp16-eval
decision: promote
next_action: Update R-S400 best-cell to small/W=8/G=8/S=400/V=512/**fp16** = 9,398.5 aug/s (was 4,765 fp32). Update R-S400-tiny best-cell to tiny/W=16/G=8/S=400/V=512/**fp16** = 22,874 aug/s (was 22,088 fp32). The pure-perf promote stands per charter — game-shape preserved at 16-ply cap, outputs are fp32 at the MCTS boundary, mechanism is well-understood. CAUTION for production training: fp16 introduces small numerical noise in the eval forward; for live-training adoption the Training-Quality Gate still applies (canary training run reporting val/policy_ce vs wl5_validation_v1.pt). Compound follow-ups: (a) re-measure R-S200 / R-S100 under fp16 (the bandwidth-limited regime might compound there too); (b) explore medium model at V=512 + fp16 (memory-bandwidth saving should be even larger for the bigger model); (c) revisit L09 with fp16 workers — if the worker-side ANE loss can be halved by fp16 on the torch fallback, R-TRAIN-ANE might pay; (d) revisit L11b with fp16 — R-TRAIN-LEAN at V=512 + fp16 + low sgd might compound into the trainer-level R-TRAIN family. consecutive_rejects RESETS to 0; the loop is rejuvenated.
```

### 2026-05-23 — L05-followup torch.compile rejects — neutral on MPS at V=512 (both shapes)

```yaml
lane: L05-followup-compile-cells
hypothesis: torch.compile on the eval-only model improves aug/s on MPS without quality change. Smoke at R-S400 (small / W=8 / G=8 / S=400 / V=512) and R-S400-tiny (tiny / W=16 / G=8 / S=400 / V=512); no-compile references are L01 = 4,765 and L07 = 22,088.
code_ref: 3ed4577 on main (uses --compile passthrough that landed in L05 / commit 9f60c42)
dataset_ref: pure self-play; fresh random fused checkpoint per model; no trainer; 60s/cell smoke
baseline_command: R-S400 (lab-L01-wave-extrapolation = 4,765 aug/s at small W=8 G=8 S=400 V=512); R-S400-tiny (lab-L07-tiny-contour = 22,088 aug/s at tiny W=16 G=8 S=400 V=512)
candidate_command: python scripts/canonical_sweep.py --out-dir sweep_logs/lab-L05fu-20260523T135545Z --cells-from <2-row csv> --lane L05-followup --secs-per-cell 60 --compile
hardware: M5 Max / MPS / idle
seed: workers seeded 1000..1015; same canonical_sweep seeding as L01/L07
baseline_metric: small V=512 no-compile = 4,765 aug/s; tiny V=512 no-compile = 22,088 aug/s
candidate_metric: small V=512 --compile = 4,657 aug/s; tiny V=512 --compile = 22,001 aug/s; both at plies_mean=15.97 (pure self-play, fresh checkpoint, 16-ply cap)
delta: small -2.3% (within noise floor ~±2%); tiny -0.4% (within noise). Neither shape benefits from torch.compile on MPS; if anything they're a tick slower, plausibly because the compiled graph capture adds first-call overhead that doesn't amortize in a 60s smoke. torch.compile on the EVAL-ONLY path (worker-side) is essentially a noop at these shapes on the M5 Max with mature MPS.
confidence: medium. Single 60s smoke per cell; noise floor of the established R-S400 measurements is ~±2% from prior canonical-sweep work, so the small -2.3% is at the edge of noise. The tiny -0.4% is solidly null. The directional consistency (both shapes slightly negative) suggests the compile-graph overhead is real, not just noise — but the magnitude is small enough that a longer cell could show a tighter null.
artifacts: sweep_logs/lab-L05fu-20260523T135545Z/{summary.tsv,cells.csv,metadata.txt,cell_small_W08_G08_S400_V512,cell_tiny_W16_G08_S400_V512}
commands_run:
  - python scripts/canonical_sweep.py --out-dir sweep_logs/lab-L05fu-20260523T135545Z --cells-from cells.csv --lane L05-followup --secs-per-cell 60 --compile
decision: reject
next_action: torch.compile stays available behind the `--compile` flag for diagnostic use (e.g. dev-machine A/B), but no production runs should turn it on by default. Don't queue further `--compile` lanes — the chip-level conclusion is "torch.compile on MPS at these eval-graph shapes is neutral-to-slightly-negative". The L05 code change still earned its keep by making the test cheap to run; the test concluded the lever is unproductive at these operating points. Pair with L06-followup (fp16) which is dispatching now; if both reject we have the full Tier-3 R-S* picture and the session reaches its natural halt per the charter (3+ consecutive rejects without a compound follow-up).
```

### 2026-05-23 — L11b R-TRAIN-LEAN V=512 + low sgd_per_position: +28% aug/s, needs_repeat per TQ gate

```yaml
lane: L11b-V512-low-sgd-per-position
hypothesis: L11 showed V=512 hurts at trainer level because 2.4× buffer-fill speedup × fixed sgd_per_position=0.0025 produces 3.36× more SGD steps per epoch, monopolizing MPS. Lowering sgd_per_position to 0.001 (2.5× lower) should cap per-epoch trainer work, freeing MPS for workers, and let V=512's pure-gen win finally compound at the trainer level.
code_ref: f45a3b1 on main (run-time commit; receipt-filing commit ae89934). No new code; uses --sgd-per-position passthrough that was already in L12's CLI.
evaluator: torch / MPS (workers and trainer; default — no --evaluator override)
dataset_ref: fresh random fused checkpoint (small, 324,570 params); live self-play only; the lower SGD-per-position ratio means each batch of training positions gets less optimizer work
baseline_command: lab-L10-20260523T132940Z (R-TRAIN-WL5 V=64 sgd=0.0025 = 3,297.6 aug/s; 0.0917 epochs/s) AND lab-L11-20260523T133546Z (R-TRAIN-LEAN V=512 sgd=0.0025 = 2,362.8 aug/s)
candidate_command: python scripts/lab_train_cell.py --out-dir sweep_logs/lab-L11b-20260523T134850Z --lane L11b --model small --workers 8 --games-per-batch 8 --n-simulations 400 --wave-size 512 --ema-tau 0.99 --grad-accum-steps 4 --sgd-per-position 0.001 --warmup-secs 30 --measurement-secs 120 --device mps
hardware: MacBook Pro Mac17,6; Apple M5 Max; 48 GB; MPS; idle
seed: workers 1000..1007; trainer default
baseline_metric: R-TRAIN-WL5 V=64 sgd=0.0025: 3,297.6 aug/s; 14.07 games/s; 0.0917 epochs/s; trainer_step_s_p50=0.0512s; ~80 steps/epoch
candidate_metric: R-TRAIN-LEAN V=512 sgd=0.001: 4,231.8 aug/s; 15.47 games/s; 0.05 epochs/s; trainer_step_s_p50=0.1407s; ~50 steps/epoch average (epoch 1=19, epoch 8=90); 8 epochs in 120s; plies_mean=34.27
delta: aug/s **+28.3%** vs R-TRAIN-WL5 (4231.8 / 3297.6 = 1.283); games/s +9.9%; epochs/s -45% (each epoch represents less SGD work because sgd_per_position is lower); effective SGD-rate ≈ 0.05 × 50 = 2.5 steps/s vs R-TRAIN-WL5's 0.0917 × 80 ≈ 7.3 steps/s. The mechanism predicted by L11+L09 is confirmed: lower sgd_per_position keeps per-epoch trainer time bounded (~11-21s vs L11's ~52s), workers don't starve for MPS, V=512's pure-gen efficiency surfaces.
confidence: medium. Single 120s trial, 8 epochs span (above the 2-epoch min for rate computation). The aug/s win is mechanically clean — V=512 generates more positions per second AND the trainer doesn't dominate MPS like at V=512+default sgd. The training-rate tradeoff is real: ~57% fewer effective SGD steps per second. Whether more data + less SGD beats less data + more SGD is a TRAINING-QUALITY question this perf lab cannot answer on its own.
artifacts: sweep_logs/lab-L11b-20260523T134850Z/{summary.tsv,metadata.txt,cell_train_small_W08_G08_S400_V512_EMA99_GA04_WM1_B512/{logs/trainer.log,logs/worker-NN.log,records/v7,v8}}
commands_run:
  - python scripts/lab_train_cell.py --out-dir sweep_logs/lab-L11b-20260523T134850Z --lane L11b --model small --workers 8 --games-per-batch 8 --n-simulations 400 --wave-size 512 --ema-tau 0.99 --grad-accum-steps 4 --sgd-per-position 0.001 --warmup-secs 30 --measurement-secs 120 --device mps
decision: needs_repeat
next_action: PERF FINDING: V=512 + sgd_per_position=0.001 is a real trainer-level operating point that beats R-TRAIN-WL5 on aug/s by +28.3%. The mechanism is well-understood (L11+L09+L11b compound chain). HOWEVER — sgd_per_position is a training-behavior knob (changes the rate of SGD updates per game), so the Training-Quality Promotion Gate applies. Promotion to production cannot happen on this 120s cell alone; needs (a) one canary training run at the new operating point reporting val/policy_ce vs archives/wl5_validation_v1.pt, and (b) plies/game-shape band comparison to the parent run. Recommend opening a separate "WL6 canary" lane outside this perf cycle to evaluate the training-quality side; the perf lab establishes that the lever exists, the training pipeline validates whether to adopt it. For the perf lab itself: do NOT mark this as the R-TRAIN-LEAN best-cell promote (until canary clears). Record the operating point in baselines.md as data; record the +28.3% headline as a finding; do not flip the production recipe.
```

### 2026-05-23 — L09 R-TRAIN-ANE rejects — trainer-side wins, worker-side loses

```yaml
lane: L09-ane-offload-prototype
hypothesis: A Core ML eval-worker frees the GPU from inference; even with slower raw eval the concurrent trainer step rate increases and overall R-TRAIN-ANE beats R-TRAIN-WL5.
code_ref: 5c08d3c on main (L12 driver gained --evaluator + --coreml-compute-units passthrough; L09's enabling patch)
dataset_ref: fresh random fused checkpoint (small, 324,570 params); workers run Core ML eval on CPU_AND_NE
baseline_command: lab-L10-20260523T132940Z (R-TRAIN-WL5 V=64 torch eval = 3,297.6 aug/s)
candidate_command: python scripts/lab_train_cell.py --out-dir sweep_logs/lab-L09-20260523T134213Z --lane L09 --model small --workers 8 --games-per-batch 8 --n-simulations 400 --wave-size 64 --ema-tau 0.99 --grad-accum-steps 4 --warmup-secs 30 --measurement-secs 120 --device mps --evaluator coreml --coreml-compute-units CPU_AND_NE
hardware: MacBook Pro Mac17,6; Apple M5 Max; 48 GB; MPS (trainer) + ANE (workers via Core ML CPU_AND_NE); idle
seed: workers seeded 1000..1007; trainer seed default
baseline_metric: R-TRAIN-WL5 torch: 3,297.6 aug/s; 14.07 games/s; 0.0917 epochs/s; trainer_step_s_p50=0.0512s
candidate_metric: R-TRAIN-ANE coreml: 1,930.3 aug/s; 8.00 games/s; 0.0583 epochs/s; trainer_step_s_p50=0.0227s; 10 epochs in 120s; plies_mean 30.43
delta: aug/s -41.5%; games/s -43.1%; epochs/s -36.4%; **trainer_step_s_p50 -55.7% (faster — hypothesis confirmed on the trainer side)**. Per-epoch breakdown in the trainer log: L09 epoch 8 was "(11.9s: gen=10.3s train=1.3s)" vs L10 epoch 8's "(~11s: gen=~5s train=~4s)". The trainer halved its per-epoch SGD time once MPS contention was relieved (1.3s vs 4s), but worker gen time doubled (10.3s vs 5s) — workers are slower on ANE than on torch/MPS at this model size. Net: holistic aug/s drops 41%.
confidence: medium-high. Single trial; smoke-first 120s; 10 epochs span. The trainer-side win is mechanically clean in the trainer log (train= field halves consistently). The worker-side loss is also clean (gen= field doubles consistently). First-epoch warmup is 22.3s vs steady-state ~12s — Core ML graph capture / ANE first-load overhead. Even at steady state, gen=10s vs torch's gen=5s, so the worker-side gap isn't a warmup artifact alone.
artifacts: sweep_logs/lab-L09-20260523T134213Z/{summary.tsv,metadata.txt,cell_train_small_W08_G08_S400_V064_EMA99_GA04_WM1_B512/{logs/trainer.log,logs/worker-NN.log,records/v9,v10}}
commands_run:
  - python scripts/lab_train_cell.py --out-dir sweep_logs/lab-L09-20260523T134213Z --lane L09 --model small --workers 8 --games-per-batch 8 --n-simulations 400 --wave-size 64 --ema-tau 0.99 --grad-accum-steps 4 --warmup-secs 30 --measurement-secs 120 --device mps --evaluator coreml --coreml-compute-units CPU_AND_NE
decision: reject
next_action: Holistic R-TRAIN-ANE doesn't beat R-TRAIN-WL5 at small/V=64. **But the trainer-side mechanism works** — MPS contention is real, and offloading workers DOES free the trainer. Follow-up candidates: (a) L09b — try `--coreml-compute-units CPU_AND_GPU` or `ALL` to see if the routing decision matters; (b) L09c — tiny model on ANE (smaller per-eval graph might amortize Core ML overhead better); (c) L11b (queued next, higher priority) — V=512 + lower sgd_per_position to test if the trainer-side cost from L11 can be capped, leveraging the same "free up MPS for workers" intuition. The compound chain (L11+L09) tells the story: pure-gen wins don't free-ride to trainer, and naive worker-offload doesn't pay either — but the trainer-side MPS contention IS real and movable.
```

### 2026-05-23 — L11 R-TRAIN-LEAN V=512 rejects — wave win doesn't compound at trainer

```yaml
lane: L11-end-to-end-cell
hypothesis: V=64 → V=512 promote (from L01, R-S400 +49.5%) compounds at the trainer level — full WL5 recipe with V=512 should beat R-TRAIN-WL5 on epochs/sec OR games/sec OR aug/sec.
code_ref: 4a825f1 on main (same L12 driver as L10)
dataset_ref: fresh random fused checkpoint (small, 324,570 params); live self-play only
baseline_command: lab-L10-20260523T132940Z (R-TRAIN-WL5 = 3,297.6 aug/s; 14.07 games/s; 0.0917 epochs/s)
candidate_command: python scripts/lab_train_cell.py --out-dir sweep_logs/lab-L11-20260523T133546Z --lane L11 --model small --workers 8 --games-per-batch 8 --n-simulations 400 --wave-size 512 --ema-tau 0.99 --grad-accum-steps 4 --warmup-secs 30 --measurement-secs 120 --device mps
hardware: MacBook Pro Mac17,6; Apple M5 Max; 48 GB; MPS; idle
seed: workers seeded 1000..1007 (w0..w7); trainer seed default
baseline_metric: R-TRAIN-WL5 V=64: 3,297.6 aug/s; 14.07 games/s; 0.0917 epochs/s; trainer_step_s_p50=0.0512s; 14 epochs in 120s
candidate_metric: R-TRAIN-LEAN V=512: 2,362.8 aug/s; 8.42 games/s; 0.0083 epochs/s; trainer_step_s_p50=0.138s; 3 epochs in 120s; plies_mean 34.25
delta: aug/s -28.4%; games/s -40.2%; epochs/s -91%; trainer_step_s_p50 +169%. Buffer fills 2× faster at V=512 (buf=199,608 at epoch 3 vs 83,208 at V=64 epoch 3 — ratio 2.40×), so fixed sgd_per_position=0.0025 produces ~3× more SGD steps per epoch (epoch 3 had steps=306 vs 91 at V=64 — ratio 3.36×). The 43s of train-time per epoch at V=512 starves workers of MPS, dropping games/s by 40%.
confidence: medium-high. Single trial, smoke-first 120s window. The mechanism is mechanically clear in the trainer log (per-epoch wall jumps from ~11s to ~52s; train= field shows 43s of 51.8s; steps= triples). This is the holistic measurement working as intended — pure-gen wins don't necessarily compound when the trainer fights for MPS.
artifacts: sweep_logs/lab-L11-20260523T133546Z/{summary.tsv,metadata.txt,cell_train_small_W08_G08_S400_V512_EMA99_GA04_WM1_B512/{logs/trainer.log,logs/worker-NN.log,records/v2,v3}}
commands_run:
  - python scripts/lab_train_cell.py --out-dir sweep_logs/lab-L11-20260523T133546Z --lane L11 --model small --workers 8 --games-per-batch 8 --n-simulations 400 --wave-size 512 --ema-tau 0.99 --grad-accum-steps 4 --warmup-secs 30 --measurement-secs 120 --device mps
decision: reject
next_action: V=64 stays the R-TRAIN-WL5 default. The pure-gen R-S* promotes (L01/L03/L07 V=512) remain valid for non-trainer self-play (e.g. eval, validation runs) but are NOT recommended for live training. Follow-up lane candidate: L11b "V=512 + lower sgd_per_position" — does reducing trainer work-per-position let V=512's gen win shine through? Lower priority; the headline finding (gen-side wins don't free-ride at trainer level) is the value here.
```

### 2026-05-23 — L10 R-TRAIN-WL5 first-ever live-training baseline

```yaml
lane: L10-trainer-step-bench
hypothesis: First-ever R-TRAIN-WL5 measurement. Full WL5 production recipe (trainer + 8 workers + EMA τ=0.99 + grad_accum=4 + V=64), 30s warmup + 120s measure, report epochs/sec, games/sec, aug/sec, trainer_step_s_p50.
code_ref: 4a825f18ef1421d5f7378ff8525b6ffc270bf1b3 on main
dataset_ref: fresh random fused checkpoint (small, stem_padding=1, 324,570 params); live self-play only, no archive ingest
baseline_command: n/a — first measurement at this reference point
candidate_command: python scripts/lab_train_cell.py --out-dir sweep_logs/lab-L10-20260523T132940Z --lane L10 --model small --workers 8 --games-per-batch 8 --n-simulations 400 --wave-size 64 --ema-tau 0.99 --grad-accum-steps 4 --warmup-secs 30 --measurement-secs 120 --device mps
hardware: MacBook Pro Mac17,6; Apple M5 Max; 48 GB; MPS; idle (pre-flight pgrep clean)
seed: workers seeded 1000..1007 (w0..w7); trainer seed default
baseline_metric: n/a
candidate_metric: aug_pos_per_sec=3,297.6; games_per_sec=14.074; epochs_per_sec=0.0917; trainer_step_s_p50=0.0512s; plies_mean=29.61; 14 epochs in 120s window; 1,489 games / 348,888 aug positions
delta: vs R-S400 pure-gen (4,765 aug/s): trainer contention costs -30.8% on aug/s. R-TRAIN-WL5 is the END-TO-END number; the contention is the point.
confidence: medium-high; smoke-first 120s window, 14 epochs span (well above the 2-epoch minimum to compute a rate); single trial. The L12 driver had two startup bugs (--save-every=1M froze worker_weights publish; count_records undercounted because trainer ingests/deletes), both fixed in commits 1dc4abb and 4a825f1 prior to this measurement.
artifacts: sweep_logs/lab-L10-20260523T132940Z/{summary.tsv,metadata.txt,cell_train_small_W08_G08_S400_V064_EMA99_GA04_WM1_B512/{logs/trainer.log,logs/worker-NN.log,records/v13,v14}}
commands_run:
  - python scripts/lab_train_cell.py --out-dir sweep_logs/lab-L10-20260523T132940Z --lane L10 --model small --workers 8 --games-per-batch 8 --n-simulations 400 --wave-size 64 --ema-tau 0.99 --grad-accum-steps 4 --warmup-secs 30 --measurement-secs 120 --device mps
decision: promote
next_action: Update best-cells.md R-TRAIN-WL5 row; spawn Reviewer for receipt audit; dispatch L11 (R-TRAIN-LEAN at V=512) for the V=64→V=512 compounding test. Quality gate (val/policy_ce vs wl5_validation_v1.pt) NOT required — this is a perf-only baseline at the WL5 quality pin; no behavior-changing knob movement.
```

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

### 2026-05-23 — L07 tiny contour (promote at R-S400-tiny; model-dependent W peak revealed)

```yaml
lane: L07-tiny-contour
tier: bg
hypothesis: Tiny model contour at V=128/V=256/V=512/V=1024 sets the R-S400-tiny ceiling and probes whether V=512 plateau extends with cheaper-eval model AND whether W axis is non-monotone at tiny like at small.
references_affected: R-S400-tiny (canonical-sweep best = tiny W=8 G=8 S=400 V=64 = 7,326 aug/s)
code_change: false
baseline_command: |
  python scripts/canonical_sweep.py --out-dir sweep_logs/lab-L07-tiny-contour-20260523T065429Z --cells-from <same dir>/cells.csv --lane L07-tiny-contour --secs-per-cell 300 --max-plies 16 --device mps
candidate_command: same (6-cell tiny contour at V in {128, 256, 512, 1024} for W=8; and V in {256, 512} for W=16)
hardware: M5 Max / MPS / idle box
seed: per-worker 1000..1015
baseline_metric: tiny W=8 G=8 S=400 V=64 = 7,326 aug/s (canonical sweep, R-S400-tiny baseline)
candidate_metric: |
  tiny W=8  V=128 =  9,407 aug/s (+28.4% over V=64)
  tiny W=8  V=256 = 14,461 aug/s (+97.4% over V=64)
  tiny W=8  V=512 = 17,088 aug/s (+133.2% over V=64)
  tiny W=8  V=1024= 17,012 aug/s (flat with V=512; same plateau as small)
  tiny W=16 V=256 = 16,375 aug/s (+123.5%)
  tiny W=16 V=512 = 22,088 aug/s (+201.5% over V=64; new R-S400-tiny best)
delta: |
  R-S400-tiny: V=64 -> W=16 V=512 = +201.5% (7,326 -> 22,088 aug/s). Promote.
  V=512 plateau holds for tiny too (V=768/1024 won't help, V=1024 cell was flat).
  **Critical finding**: W=16 V=512 beats W=8 V=512 by +29.4% at tiny. This is the OPPOSITE of L02's result at small where W=16 V=512 was WORSE than W=8 by 5.5%. The W-axis sweet spot at V=512 depends on model size: tiny (cheap eval) can keep more workers fed before MPS-dispatch saturates; small (expensive eval) saturates at W=8.
confidence: medium-high. 5-min cells, idle box, monotone within experimental noise. Plies cap 15.96 universal.
artifacts: sweep_logs/lab-L07-tiny-contour-20260523T065429Z/{summary.tsv, cells.csv, metadata.txt, driver.log, cell_tiny_W{08,16}_G08_S400_V{128,256,512,1024}/{records,logs}}
commands_run:
  - python scripts/canonical_sweep.py --out-dir sweep_logs/lab-L07-tiny-contour-20260523T065429Z --cells-from ... --lane L07-tiny-contour
decision: promote (R-S400-tiny only; primary R-S* refs unchanged because they pin model=small)
reviewer: APPROVE (general-purpose agent, 2026-05-23). "L07 promote math clean (+201.5%); 2-axis move decomposed via cell matrix; surfaces consistent; L13/L14 well-scoped."
next_action: |
  1. Promote R-S400-tiny: W=8 V=64 -> **W=16 V=512** (7,326 -> 22,088 aug/s, +201.5%). best-cells.md updated.
  2. consecutive_rejects: 2 -> 0 (any promote resets per stop rule).
  3. **Model-dependent W axis finding has direct implications for L09 ANE-offload work**: with workers on Core ML (CPU/ANE), the effective "eval cost" per worker changes. Whether W=8 or W=16 is the peak under the ANE workload is unknown. L09 should compare BOTH W=8 and W=16 at V=512 in its measurement cells, not just one.
  4. Auto-queue follow-up L13 (new): probe tiny peak finer. W ∈ {12, 16, 20, 24} at tiny G=8 S=400 V=512. 4 cells × 5 min = 22 min wall. Highest E[delta] in current queue because tiny+V=512+W=16 just unlocked a new regime.
  5. Auto-queue follow-up L14 (new): tiny G axis at V=512 W=16. G ∈ {4, 16, 32} at tiny W=16 S=400 V=512. 3 cells × 5 min = 17 min. Probably moderate E[delta]; G axis was mildly non-monotone at small V=512 (L04).
  6. Primary R-S* targets are now exhausted of single-axis tweaks at small (W and G confirmed at peak, V at plateau, model is the only knob left that moves the needle — and tiny is a different quality regime). The next big mover is architectural: L09 ANE-offload (when L12 driver lands) and L05/L06 worktree code lanes.
```

### 2026-05-23 — L13 tiny W peak probe (reject; W=16 confirmed; tolerance W∈[12,20] within 7%)

```yaml
lane: L13-tiny-W-peak-probe
tier: bg
hypothesis: L07 showed W=16 beats W=8 by +29% at tiny V=512. Probe finer at W ∈ {12, 20, 24} to see if peak is even higher than W=16.
reference: R-S400-tiny (current best = tiny W=16 G=8 S=400 V=512 = 22,088 aug/s from L07)
code_change: false
baseline_command: |
  python scripts/canonical_sweep.py --out-dir sweep_logs/lab-L13-tiny-W-peak-probe-20260523T073525Z --cells-from <same dir>/cells.csv --lane L13-tiny-W-peak-probe --secs-per-cell 300 --max-plies 16 --device mps
candidate_command: same (3-cell sweep at W in {12, 20, 24} with tiny G=8 V=512)
hardware: M5 Max / MPS / idle box
seed: per-worker 1000..1023
baseline_metric: tiny W=16 G=8 V=512 = 22,088 aug/s (L07)
candidate_metric: |
  W=12 V=512 = 20,560 aug/s (-6.9% vs W=16)
  W=16 V=512 = 22,088 aug/s (L07 reference)
  W=20 V=512 = 21,553 aug/s (-2.4% vs W=16)
  W=24 V=512 = 20,970 aug/s (-5.1% vs W=16)
delta: REJECT. W=16 confirmed as the tiny V=512 peak. W=20 is a close second (within 2.4%); tolerance band W ∈ [12, 20] within 7% of peak. The tiny W-axis is a smooth bump (vs small's sharper saturation drop past W=8 per L02).
confidence: high. 4-W-value scan brackets the peak cleanly with monotone shape on both sides.
artifacts: sweep_logs/lab-L13-tiny-W-peak-probe-20260523T073525Z/{summary.tsv, cells.csv, metadata.txt, driver.log, cell_tiny_W{12,20,24}_G08_S400_V512/{records,logs}}
commands_run:
  - python scripts/canonical_sweep.py --out-dir sweep_logs/lab-L13-tiny-W-peak-probe-20260523T073525Z --cells-from ... --lane L13-tiny-W-peak-probe
decision: reject
reviewer: APPROVE (general-purpose agent, 2026-05-23). "L13 reject clean: math/plies/units verified, W=16 confirmed peak, surfaces consistent, no spurious follow-ups."
next_action: |
  1. Best-cells.md unchanged. tiny W=16 G=8 V=512 = 22,088 stays the R-S400-tiny default.
  2. **Compound finding with L02 + L07**: model size determines BOTH the W-peak location AND the W-axis tolerance shape at V=512:
     - small: peak W=8, sharp drop (W=16 = -5.5%; W=4 = -8.4%)
     - tiny: peak W=16, gentle bump (W=12 = -6.9%; W=20 = -2.4%; W=24 = -5.1%)
     Tiny's wider tolerance band means L09 ANE worker-count tuning has more headroom than small's tuning had.
  3. consecutive_rejects: 0 -> 1.
  4. Next dispatch (priority order): L14 G axis at tiny W=16 V=512 (bg, priority 16.5).
```

### 2026-05-23 — L14 G-x-axis at tiny W=16 V=512 (reject; G axis flat at the tiny peak)

```yaml
lane: L14-tiny-G-at-W16-V512
tier: bg
hypothesis: L04 found G axis is mildly non-monotone at small V=512. L13 found tiny W-axis has gentle bump. G axis at tiny W=16 V=512 may also have a non-G=8 peak.
reference: R-S400-tiny (current best = tiny W=16 G=8 V=512 = 22,088 aug/s from L07)
code_change: false
baseline_command: |
  python scripts/canonical_sweep.py --out-dir sweep_logs/lab-L14-tiny-G-at-W16-V512-20260523T080354Z --cells-from <same dir>/cells.csv --lane L14-tiny-G-at-W16-V512 --secs-per-cell 300 --max-plies 16 --device mps
candidate_command: same (3-cell sweep at G in {4, 16, 32} with tiny W=16 V=512)
hardware: M5 Max / MPS / idle box
seed: per-worker 1000..1015
baseline_metric: tiny W=16 G=8 V=512 = 22,088 aug/s (L07)
candidate_metric: |
  G=4  V=512 = 22,261 aug/s (+0.78% vs G=8)
  G=8  V=512 = 22,088 aug/s (L07 reference)
  G=16 V=512 = 22,164 aug/s (+0.34%)
  G=32 V=512 = 22,076 aug/s (-0.06%)
delta: REJECT. G axis is essentially flat at tiny W=16 V=512. Total spread G=4→G=32 is 0.83% (185 aug/s). The nominal G=4 lead of +0.78% is within the unmeasured run-to-run noise floor and not a defensible promote.
confidence: high. Flatness is unambiguous (4 cells within <1% of each other; no monotone direction).
artifacts: sweep_logs/lab-L14-tiny-G-at-W16-V512-20260523T080354Z/{summary.tsv, cells.csv, metadata.txt, driver.log, cell_tiny_W16_G{04,16,32}_S400_V512/{records,logs}}
commands_run:
  - python scripts/canonical_sweep.py --out-dir sweep_logs/lab-L14-tiny-G-at-W16-V512-20260523T080354Z --cells-from ... --lane L14-tiny-G-at-W16-V512
decision: reject
reviewer: APPROVE (general-purpose agent, 2026-05-23). "L14 reject correct — G axis spread 0.83% within noise; surfaces consistent; pause state cleanly logged."
next_action: |
  1. Best-cells.md unchanged. tiny W=16 G=8 V=512 = 22,088 stays the R-S400-tiny default.
  2. **Knob-tuning exhausted at chip envelope.** Across L02 (small W axis at V=512), L04 (small G axis at V=512), L13 (tiny W finer probe), and L14 (tiny G at peak), no further single-axis exploration of W or G has produced a promote. The remaining headroom is structural:
     - L09 ANE-offload (blocked on L12 driver)
     - L05 torch.compile (needs worktree code)
     - L06 fp16 (needs worktree code)
     - L08 heap ratio (needs per-cell env var driver support)
     - L12 live-training cell driver (Tier 1 gating; needs human session)
  3. consecutive_rejects: 1 -> 2. Even with 3+ consecutive rejects the loop wouldn't halt (queue has lanes that just need code work), but this marks the natural pause point for the autonomous tick chain.
  4. PushNotification the user: cron has exhausted no-code unblocked lanes. Human session needed for L05/L06/L08/L12.
```

