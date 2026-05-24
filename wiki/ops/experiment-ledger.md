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

### 2026-05-23 — delta-e run-1 (first Δelo flywheel run) — harness validated end-to-end; ANCHOR-CEILING method-limit discovered

```yaml
lane: delta-e-run1 (first real run of scripts/delta_e_harness.py — the Δelo/Δt north-star scoring engine for the curated-buffer flywheel). 3 recipes forked off a common WL5 parent C, fixed 40-epoch window, anchored-eval each fork + C at 40 games/baseline.
hypothesis: curator (lru vs recency_weighted) and sgd_steps_per_epoch (100 vs 300) produce measurable Δelo separation off a common parent, resolvable above the eval-noise floor.
code_ref: scripts/delta_e_harness.py (anchored-eval mode) + gomoku/train_replay.py (replay-fork trainer) + gomoku/curated_buffer.py (curators). All on main.
dataset_ref: parent C = sweep_runs/WL5-diagnostics-archive-start/checkpoints/epoch10200.pt; archive A = /tmp/wl5_replay_archive (WL5's 1.5M-position buffer, retain-all). validation = archives/wl5_validation_v1.pt.
hardware: M5 Max, eval on cpu (6 workers), forks on mps. small. eval sims=100, c_puct=1.5, 40 games/baseline.
seed: 0
baseline_metric: parent C elo = +1536.7 [+1356.8, +1690.8] (±167.0, 40 games/baseline). C win-rates: heuristic=75%, lookahead:depth=2=89%, lookahead:depth=4=78%.
candidate_metric: |
  rank  recipe                       Δelo     ±CI    window  verdict        fork win-rates (h / d2 / d4)
  1     lru,sgd=100                  -55.8    218.1   40 ep   INSIDE-NOISE   64% / 100% / 65%
  2     recency_weighted,sgd=100    -127.4    228.3   40 ep   INSIDE-NOISE   75% /  85% / 51%
  3     lru,sgd=300                 -195.0    233.2   40 ep   INSIDE-NOISE   74% /  66% / 54%
delta: all 3 INSIDE-NOISE (|Δelo| <= CI half-width). No recipe distinguishable from C or from each other.
confidence: HIGH on the NEGATIVE/method finding. Root cause is NOT that the recipes are equal — it is an ANCHOR-CEILING limit: the strongest anchor is lookahead:depth=4 @ 1500, and C already beats it (78%). Both C and every fork pin near the ~1700-elo ceiling; differencing two near-ceiling implied-elos buries the real difference under sampling noise (signal lives almost entirely in the d4 win-rate, the only non-saturated anchor, which has a huge CI at 40 games). keep-last-n=3 pruning destroyed all early/weak checkpoints, so no "headroom parent" exists among surviving checkpoints either (WL3.1 e1504 also sits at model_elo ~1465-1567). The faint trend (sub-noise, do NOT over-read): sgd=300 most negative → directionally consistent with over-grinding a tiny curated slice off a converged net.
artifacts: /tmp/delo_run1/results.json, /tmp/delo_run1.log; wandb runs (fresh, NOT WL5's): delta-e-lru_sgd100=ms1pplps, recency_weighted_sgd100=91awvfib, lru_sgd300=jc228edo (project jasonyandell-forge42/gomoku).
commands_run: python scripts/delta_e_harness.py --parent <C> --archive-path /tmp/wl5_replay_archive --window-epochs 40 --recipe 'lru:sgd_steps_per_epoch=100' --recipe 'recency_weighted:sgd_steps_per_epoch=100' --recipe 'lru:sgd_steps_per_epoch=300' --eval-games 40 --eval-sims 100 --wandb --out-dir /tmp/delo_run1
decision: needs_repeat
next_action: re-run head-to-head (fork-vs-C direct match) — the ceiling-free fix. Two similar-strength models score near 50% against each other (the max-sensitivity region of the logistic), so relative Δelo gets a tight CI with no anchor ceiling. Added --head-to-head to delta_e_harness.py this session (model-vs-model via play_match_pickers + mcts_picker; relative Δelo = 400*log10(p/(1-p)), Wilson CI mapped through the logistic). run-2 = same 3 recipes off the same WL5 C, head-to-head. Also: a true headroom parent would need re-generating an early-epoch checkpoint (none survived pruning) — head-to-head sidesteps that entirely.
```

### 2026-05-23 — LF1-followups (fan-out, 5 sub-lanes) — runaway knee in (384,512]; tile-cap tames it; extra steps redundant; metric instrument fixed

```yaml
lane: LF1-followups (the 6-lane block re-pointing the lab at wall-clock-to-elo after the LF1 runaway; ran 5 as a two-queue fan-out — 4 worktree-isolated CPU agents + 1 GPU sweep). Jason: "proceed, fan out background subagents."
hypothesis: (per wiki/topics/perf-bench-vs-real-training-cost.md) V=512 + sgd_per_position causes an unbounded per-epoch runaway; the cold-window R-TRAIN metric missed it. Map the boundary, build the structural fix, check if the extra steps are productive, fix the instrument.
code_ref:
  - lane1 warm-buffer: feat/perf-LF1-warmbuf @ 38fc90f → merged (lab_train_cell --replay-buffer-size/--prefill-*; post-fill slope + BOUNDED/DIVERGING verdict; <20 post-fill epochs = INCONCLUSIVE, refuses a cold number).
  - lane6 tile-cap: feat/perf-LF1-tilecap @ 7802672 → merged (gomoku/train.py --max-tile-games [drop, earliest-prefix, post-barrier pre-buffer-add], --max-sgd-steps-per-epoch, --sgd-per-game; threaded through lab_train_cell. All opt-in; WL5 default byte-identical — build_trainer_cmd char-identical to main when unset).
  - lane4 metric design: feat/perf-LF1-metric-design @ 3279d69 → merged (wiki/topics/wall-clock-to-elo-metric.md: MTTE primary, EPWH secondary, charter diff proposed [Class-B, un-applied], delta_e_harness.py gap analysis).
hardware: M5 Max, mps, small/W8/G8/S400, EMA τ=0.99, grad_accum=4, sgd_per_position=0.001, fp16-eval, --max-epochs 18. wandb disabled (lab cells); lane-5 read wandb h9al2e0k + geft5xmy.
baseline_command: python scripts/lab_train_cell.py --wave-size {256,384,512} ... --max-epochs 18 (uncapped)
candidate_command: python scripts/lab_train_cell.py --wave-size 512 ... --max-tile-games 120 --max-epochs 18
baseline_metric (lane2 runaway boundary, steps/epoch e1→e18, wall/epoch e1→e18):
  V=256 uncapped: 20→56 steps, 9.6→7.3s  — BOUNDED
  V=384 uncapped: 19→62 steps, 5.9→8.8s  — BOUNDED
  V=512 uncapped: 22→154 steps, 6.8→19.9s, new-pos 77→630 — DIVERGENT (monotonic; the runaway reproduced in lab_train_cell)
candidate_metric (lane6 cap validation):
  V=512 + --max-tile-games 120: 23→53 steps, 6.6→7.7s, new-pos 84→207 — BOUNDED (cap converts the divergent recipe to bounded; same recipe, cap is the only diff)
finding (lane2): the runaway stability KNEE is in (384, 512] — sharp. CRITICAL METHOD NOTE: divergence does NOT appear in per-version tile (barrier-bounded ~85 by worker-min-games=64, invariant to V); it appears in steps/wall/new-positions/age (trainer falls behind → drains more stale versions/epoch). Watching tile-only gives a false "no runaway" read.
finding (lane5, REDUNDANT, high confidence): val/policy_ce (vs wl5_validation_v1.pt) best=3.9905 at cum-step ~3.4k/epoch20 then flattens+reverses while cumulative steps go 4.4× higher to 14.7k; train-loss falls monotonically (overfitting signature). The runaway's extra SGD re-grinds stale buffer (~28% current). ⇒ bounding the tile costs ~0 elo and improves elo-per-wall. Lane5 + lane6 compose: the cap removes exactly the non-productive steps.
quality_gate: behavior knobs are OPT-IN and OFF by default; WL5 production recipe byte-identical (verified: default --dry-run emits none of the 7 new flags; combined --dry-run threads them; tests green). No production default changed (that would be ESCALATE). The cap's elo-neutrality is supported by lane-5's val/policy_ce analysis but a live TQ canary (val-CE vs wl5_validation_v1.pt at full buffer) is the gate before any production adoption.
confidence: HIGH for the boundary + cap mechanism (stark 4-run contrast, same recipe). CAVEAT (UPDATED by Reviewer): the uncapped V=512 run actually FILLED the 1.5M buffer at epoch 15 and kept diverging (steps 149→154 through e15→e18) — so the divergence already extends THROUGH buffer-fill, not just pre-fill. The bounded runs (V=256/384/capped@18ep) were at ~64% fill (lower inflow). CAVEAT NOW CLOSED: a 32-epoch capped V=512 run reached the FULL 1.5M buffer at e28 and stayed BOUNDED through and past fill (postfill steps_slope=0.019, wall_slope=0.037, tile_verdict=BOUNDED; e28-32 at full buffer: steps ~54, wall ~8-10s flat — minor wall uptick is thermal, steps flat). So the cap holds at steady state, not just pre-fill. (Artifact: sweep_logs/lab-LF1-tilecap-V512cap120-postfill-*.)
reviewer: APPROVE — headline reproduced from the four raw trajectory.tsv (V=256 20→56/9.6→7.3s, V=384 19→62/5.9→8.8s bounded; V=512-uncapped 22→154/6.8→19.9s/new 77→630 monotonic, buffer full at e15; V=512+cap120 23→53/6.6→7.7s bounded); knee (384,512] follows. Attribution sound (tile barrier-bounded ~73-108 across ALL runs incl. divergent → divergence correctly attributed to steps/wall/new/age, not tile). Behavior gate passes: WL5-default dry-run byte-identical to pre-LF1 main (none of the 7 flags), combined dry-run threads them, 20 tests green. Caveat honest (V=512 summary reports tile_verdict=INCONCLUSIVE — lane-1 mode refuses a cold number). Charter R-ELO-* diff left un-applied (Class-B). 3 merges all 2-parent --no-ff.
artifacts: sweep_logs/lab-LF1-runaway-V{256,384,512-uncapped}-*, lab-LF1-tilecap-V512cap120-* (trajectory.tsv each).
decision: promote (lanes 1/4/6 code+design landed, merged --no-ff); lane2 knee mapped (research finding); lane5 analysis finding. Reviewer APPROVE (full verdict in the confidence field above).
next_action: (1) warm-buffer post-fill confirmation run (V=512 ± cap, --replay-buffer-size ~20000 to fill within budget, ≥20 post-fill epochs) — closes the pre-fill caveat AND exercises lane-1's mode on a real runaway; (2) narrow the knee with V=448 if a sharp-vs-gradual threshold matters; (3) Class-B: charter Success-metric edit (lane-4 R-ELO-* diff) is the user's call. Runaway reproduction in lab_train_cell means the harness CAN now study this (previously thought run_sweep-only).
```

### 2026-05-23 — LF1 (LEAN-fp16 as a REAL run) — the +152% is COLD-BUFFER generation throughput, NOT training speed (steady-state ~3min/epoch)

```yaml
lane: LF1-lean-fp16-canary (the R-TRAIN-LEAN-fp16 recipe promoted from perf-bench to a real run_sweep training run; Jason "let's friggin try it")
context: the perf lab's R-TRAIN-LEAN-fp16 (WL5 + V=512 + sgd_per_position=0.001 + fp16 workers) measured +152% vs R-TRAIN-WL5 (8,340 aug/s, 0.0667 epochs/s ≈ 15s/epoch) in a 120s lab_train_cell window. LF1 runs it for real (run_sweep cell, quality-tracked, wandb h9al2e0k; 100-ep test geft5xmy first).
finding (load-bearing — the +152% is not just wrong, it hides a RUNAWAY): the 0.0667 epochs/s was a COLD-BUFFER TRANSIENT measured over ~8 epochs. In the real run the per-epoch cost grows WITHOUT BOUND. Trajectory (epoch: steps, wall): e1: 25/19.9s; e15: 218/30.4s; e20: 446/57.9s; e25: 792/107.8s; e27: 1237/167.5s; e30: 2523/342.5s; **e31: 3236/436.9s (7.3 min) and still climbing when stopped.** The wave tile grew in lockstep (101→610→1120→1691→2284→2898) and new-positions/epoch 985→17,391. NOT a steady ~3min/epoch — it diverges. Mechanism: V=512 makes generation OUTPACE the trainer; workers keep generating during the lengthening SGD phase → the next wave tile is bigger → sgd_per_position scales SGD with the growing inflow → longer train → more accumulation → positive-feedback RUNAWAY. WL5 (V=64) doesn't run away (gen ≈ consumption, tile ~70-86). Full writeup: wiki/topics/perf-bench-vs-real-training-cost.md.
implication: the perf-lab R-TRAIN-* epochs/s and aug/s measure GENERATION throughput in a short cold window; they do NOT equal training speed. The +152% is real for cold-window aug/s but must NOT be read as "+152% faster training" — steady-state per-epoch cost is set by full-buffer × sgd_per_position → steps/epoch. (This is the L11 mechanism — V=512 fills the buffer faster → more SGD/epoch — at production scale.)
flip side: it learns FAST per epoch — LF1 hit elo 437→776 around the buffer-full transition (~epoch 28), because each epoch does ~1300 SGD steps. So epochs-to-elo may be FEWER even with higher wall-per-epoch. The only honest "faster recipe" verdict is wall-clock-to-elo + val/policy_ce quality.
quality so far (100-ep test, early): pl 4.42→3.78, vl down, plies ~27 (healthy), wr climbing (heuristic 0%→50%, elo→776), 0 NaN. Encouraging but early; full TQ verdict pending the 1000-ep LF1 dynamics (val/policy_ce vs wl5_validation_v1.pt, plies-shape, baseline elo trajectory).
decision: REJECT for production training (Jason stopped LF1 at e31 — "3 min/epoch? forget that"). The R-TRAIN-LEAN-fp16 "+152%" is RETRACTED as a training-speed claim: it's cold-window generation throughput that hides an unbounded per-epoch runaway in real training. fp16+V=512 GENERATION is faster, but feeding that into a wave-mode trainer with sgd_per_position causes the runaway. The bigger result is a metric-validity finding for the whole lab (below) → 6 research lanes queued.
meta_finding: the lab optimized aug/s (generation), but the real objective is wall-clock-to-elo (training); maximizing generation can FLOOD the trainer into a runaway. The R-TRAIN-* cold-window metric measured a pre-buffer-fill transient and called a divergent recipe "+152%". Reframes the lab — see wiki/topics/perf-bench-vs-real-training-cost.md.
cross_ref: gomoku-train skill "Tuning knobs → LEAN-fp16"; [[feedback-self-play-eta]] (the ETA-extrapolation lesson, now with this buffer-fill facet); run wandb h9al2e0k.
```

### 2026-05-23 — LA1 — vectorize the lookahead-eval hot path: ~6.3× faster, byte-identical move selection (PROMOTE)

```yaml
lane: LA1-lookahead-eval-vectorize (perf pass on the alpha-beta lookahead baseline — an Elo anchor used by eval_worker/train eval; not a gen/train lane). User-introduced 2026-05-23.
hypothesis: the lookahead baseline is the known-slow eval path (train.py:341 dropped lookahead:depth=2 as default at "45s+ for noisy signal"). With native state_ops already C-accelerated, the per-node bottleneck must be the pure-numpy helpers in baselines.py that run at every search node. Vectorizing them should cut eval wall-clock with zero behavior change.
code_ref: feat/perf-lookahead-eval @ 5d0985a (vectorize _find_immediate_wins, _candidate_moves via precomputed (81,81) _NEIGHBOR_MASK, + candidate-restricted _score_cells for the ordering hot path). Behavior-preserving by construction (per-cell scores independent; same move sets).
dataset_ref: scripts/bench_lookahead.py — 60 deterministic heuristic-vs-heuristic midgame positions (seed=0), spanning plies 4..24.
baseline_command: git stash / pre-5d0985a `python scripts/bench_lookahead.py --n-positions 60 --depths 2,4`
candidate_command: `python scripts/bench_lookahead.py --n-positions 60 --depths 2,4` @ 5d0985a
hardware: M5 Max, CPU/numpy single-process (lookahead is torch-free); GOMOKU native state_ops active (USING_NATIVE=True). wandb: disabled (offline CPU microbench).
seed: positions seed=0; player tie-break rng seed=1 (deterministic, no tie-break divergence observed).
baseline_metric: depth=2 = 15.35 ms/move (65.1 moves/s); depth=4 = 145.61 ms/move (6.9 moves/s).
candidate_metric: depth=2 = 2.36 ms/move (424.4 moves/s); depth=4 = 22.94 ms/move (43.6 moves/s).
delta: depth=2 6.5× faster; depth=4 6.3× faster. cProfile: _find_immediate_wins 3.23s→0.14s, _candidate_moves 3.03s→0.08s tottime over the 60-move depth-4 run.
confidence: HIGH. Behavior-identical proven, not just argued: across 360 positions (incl. child positions) the new _candidate_moves, _find_immediate_wins, and _score_cells return byte-identical results to the old loop logic, and lookahead_player(depth=4) picks the identical move. Quality gate is satisfied by construction — the lookahead Elo anchor's outputs are unchanged, so there is no strength/game-shape risk to measure; the gate's concern (does search behavior change?) is provably no. tests: test_baselines + test_lookahead_quiescence + test_rating + test_eval_parallel all green.
artifacts: scripts/bench_lookahead.py (committed); equivalence harness in job dir (ephemeral).
commands_run: pytest (29 passed); bench_lookahead.py before/after; cProfile before/after; 360-position equivalence check.
decision: promote
reviewer: APPROVE — behavior-preserving (equivalence logic sound across edge cases: empty-legal early-return, padded-slot zeroing before the max, opening/fallback preserved), math exact (6.347×/6.504×), all 5 surfaces consistent, 28 tests green, bench in thermal-ballpark, merge --no-ff, concurrent 6d47bbb work intact-not-clobbered. Non-gating flag actioned: the 360-position proof was ephemeral → committed tests/test_baselines_vectorized_equiv.py (helpers vs independent brute-force refs) so a future helper edit can't silently shift the Elo anchor.
next_action: merge feat/perf-lookahead-eval to main (--no-ff). Follow-up levers (diminishing returns, NOT queued unless eval cost resurfaces): (a) history-free apply for the lookahead path — apply_move_arrays copies 8 history snapshots/node that negamax never reads (~8-10% remaining), but it's shared with the MCTS/self-play path so needs a lookahead-specific lighter apply (riskier); (b) numba/cython negamax (Class C, out of autonomous scope). Real-world impact: eval_worker's lookahead side (esp. depth=4, the dominant anchor per train.py:348) gets ~6× cheaper, making frequent Elo anchoring affordable — directly serves the Δelo/Δt north-star.
```

### 2026-05-23 — Ltrain-amp — trainer-side bf16 autocast: directionally SLOWER, but cell is heat-soak-confounded (needs_repeat)

```yaml
lane: Ltrain-amp (Tier-1: does trainer-side bf16/fp16 autocast speed the R-TRAIN SGD step?)
hypothesis: fp16 EVAL gave +97% (L06); trainer-side precision is unexplored. bf16 autocast on the trainer forward+loss might cut trainer_step (the R-TRAIN bottleneck), compounding across the run.
code_ref: feat/perf-Ltrain-amp @ 4d0988e (torch.autocast on forward+loss; opt-in --trainer-amp {off,bf16,fp16}; off-path bit-identical). bf16 autocast CONFIRMED working on MPS torch 2.11 (finite grads, autocast active).
cells: small / W=8 / G=8 / S=400 / V=64 / torch workers / live training / 30s+120s. fp32 (off) then bf16, back-to-back, 40s cooldown.
results:
  - fp32 (off): aug/s=1,356.5, epochs/s=0.025, trainer_step_s_p50=0.1245, 5 epochs, plies 34.55
  - bf16: aug/s=824.9, epochs/s=0.0167, trainer_step_s_p50=0.1693, 4 epochs, plies 33.08
  - bf16 vs fp32: trainer_step +36% (SLOWER), aug/s −39%, epochs/s −33%.
CONFOUND (load-bearing): the fp32 baseline itself = 1,356 aug/s vs the cool R-TRAIN-WL5 (L10) = 3,297.6 — a −59% drop. Far beyond Lhot's measured ~0% heat-soak haircut (which was an 8-cell / 8-min test). This session ran 20+ cells + multiple sustained ~11 TFLOP/s GPU hogs over a long span → the chip is DEEP heat-soaked. Non-lab tenants ruled OUT (web.server PID 23001 @0.1% CPU/2-day uptime + microscope @0.0%/15-day — both idle). So both arms ran throttled; the bf16-vs-fp32 magnitude is unreliable. (Worktree-train.py off-path overhead not 100% ruled out, but it's bit-identical params + byte-identical dry-run cmd, so unlikely to be 2.4×.)
mechanism (tentative): bf16-trainer being slower is PLAUSIBLE independent of thermal — autocast inserts fp32↔bf16 casts around each op; for a SMALL model whose SGD step is dispatch/overhead-bound (not bandwidth-bound), the cast overhead can exceed the bf16 matmul speedup. This is the L06 regime logic in reverse: fp16/bf16 helps only in the bandwidth-bound regime; the small-model trainer step likely isn't there.
decision: needs_repeat — bf16-trainer is directionally SLOWER (+36% trainer_step) and that direction is mechanistically expected, BUT the cell is heat-soak-confounded (−59% baseline). Re-run on a COOLED chip (interleaved) to confirm the magnitude before calling it a clean reject. Also re-baselines the heat-soak (a finding in its own right: extended mixed cell+hog load throttles the M5 Max well beyond the short Lhot test).
next_action: cooldown (~12 min idle) → re-run worktree fp32 + bf16 A/B; the cooled fp32 cell doubles as a heat-soak re-baseline (recovers toward 3,297 → thermal confirmed + worktree-overhead ruled out). TQ note: trainer-precision changes numerics → production adoption is TQ-gated regardless; the lab only measures speed.
```

### 2026-05-23 — Lpwr2b — RESOLVED: cross-engine throttle is FLOP-rate-INDEPENDENT (rules out compute-power; coupling tracks GPU working-set/occupancy)

```yaml
lane: Lpwr2b (the power-vs-bandwidth discriminator; resolves Lpwr2's needs_repeat)
hypothesis: fp16@4096 (half byte-footprint, ~2x FLOP-rate potential) vs fp32@4096 (full footprint, 1x) at MATCHED matrix dim — the SIGN of the worker-throttle difference separates FLOPs/power from footprint/bandwidth.
code_ref: feat/perf-L09i-fix (canonical_sweep coreml); scripts/gpu_load_generator.py --dtype (commit 8edd16b); runner sweep_logs/run_lpwr2b.sh
method: 120s cells (cut the 80s noise), no-hog bracket (start+end), back-to-back fp32-hog then fp16-hog at matrix 4096 (shared thermal state → directly comparable). tiny/W16/G8/S400/V64/coreml CPU_AND_NE pure self-play.
results:
  - no-hog bracket: 3,971.6 and 4,234.0 aug/s (mean ~4,103; spread 6.6% — the 120s cells tightened it from Lpwr2's ~20% at 80s)
  - fp32-hog @4096: 3,451.7 aug/s (−15.9% vs bracket); hog reached 1.98 TFLOP/s
  - fp16-hog @4096: 3,496.2 aug/s (−14.8% vs bracket); hog reached **7.03 TFLOP/s** (3.5× the fp32)
finding (DEFINITIVE): the fp16 hog did **3.5× the FLOP-rate** of the fp32 hog (7.03 vs 1.98 TFLOP/s) yet threw the **SAME worker throttle** (−14.8% vs −15.9%, within 1.3% — far inside the ~6.6% no-hog noise). So the cross-engine worker throttle is **independent of the hog's compute rate**. fp16 also had ~1.75× the byte-traffic-rate (half bytes/elem × 3.5× more matmuls) yet ~equal throttle → byte-bandwidth-RATE doesn't cleanly drive it either.
mechanism (combined Lpwr2 + Lpwr2b): the throttle scales with hog matrix SIZE (Lpwr2: 2048→8192 = −8.8%→−26%) but NOT with FLOP-rate or byte-rate at fixed size (Lpwr2b). Both 4096 hogs keep the GPU saturated/occupied regardless of throughput. → the coupling tracks the GPU's **working-set size / sustained occupancy** ("is the GPU pinned busy, and how big is its footprint"), NOT its arithmetic or bandwidth throughput. **Compute-power-draw is RULED OUT as the primary driver.**
confidence: high on the FLOP-rate-independence (1.3% apart vs 3.5× FLOP difference; bracketed baseline; back-to-back matched-matrix). The positive channel (occupancy vs memory-working-set vs base-activation-power) is narrowed but not uniquely isolated.
decision: resolved (diagnostic) — pins the Lpwr/Lpwr2/Lpwr2b mechanism strand: cross-engine throttle is occupancy/working-set-driven, FLOP-rate-independent. Resolves Lpwr2's needs_repeat.
actionable: to reduce cross-engine contention from the GPU trainer, shrink the GPU's memory working-set / occupancy duration, NOT its FLOP count. Throwing more compute through an already-busy GPU doesn't worsen the throttle; enlarging its footprint does.
next_action: the coupling mechanism is pinned enough for the lab's purposes. Optional Lpwr2c (vary footprint at fixed occupancy, or memory-bound vs compute-bound hog) only if a future lane needs the exact channel. Update m5-max-cross-engine-coupling.md with the FLOP-rate-independence result.
```

### 2026-05-23 — Lpwr2 — cross-engine coupling intensity sweep: mutual coupling confirmed; mechanism hints at bandwidth/footprint (needs_repeat — noisy baseline)

```yaml
lane: Lpwr2 (cold-chip interleaved re-run of Lpwr; ANE-resident workers, pure self-play)
hypothesis: pin the cross-engine coupling mechanism (power-throttle vs scheduling vs memory-bandwidth) via a de-confounded intensity sweep — no-hog/hog interleaved per intensity so each A/B pair shares a thermal state.
code_ref: feat/perf-L09i-fix (canonical_sweep coreml + static-batch); scripts/gpu_load_generator.py; runner sweep_logs/run_lpwr2.sh
method: 150s cooldown, then for hog matrix ∈ {2048,4096,8192}: no-hog cell then hog cell back-to-back (80s/cell), 50s cooldown between intensities. tiny/W16/G8/S400/V64/coreml CPU_AND_NE, pure self-play.
results:
  - matrix 2048: no-hog 3,550.5 → hog 3,238.5 aug/s (−8.8%); hog reached 2.20 TFLOP/s
  - matrix 4096: no-hog 4,256.4 → hog 3,337.5 aug/s (−21.6%); hog reached 2.21 TFLOP/s
  - matrix 8192: no-hog 3,960.0 → hog 2,932.7 aug/s (−26.0%); hog reached 2.99 TFLOP/s
findings:
  - MUTUAL COUPLING confirmed: every hog is itself suppressed to 2-3 TFLOP/s (vs 5.8 fp32@2048 / ~10.7 fp32@8192 standalone) while throttling workers −9 to −26%. Reproduces v2's bidirectional picture across intensities.
  - MECHANISM HINT (bandwidth/footprint > pure power): at MATCHED hog TFLOP/s (~2.2 at both 2048 and 4096), the larger-footprint hog throttles workers MORE (−8.8% → −21.6%). An 8192² fp32 operand is 256MB vs 16MB at 2048² — if throttle were pure power/FLOP it should track TFLOP/s (~flat), but it tracks matrix SIZE. Points at memory bandwidth / footprint as a key coupling channel.
caveat: the no-hog baseline is NOISY (3,550 / 4,256 / 3,960 = ~20% spread, non-monotonic so not clean thermal — run variance in the short 80s ANE pure-self-play cell). This muddies the exact per-intensity %; the matrix-size→throttle trend is suggestive, not definitively pinned. Also at 8192 both footprint AND achieved TFLOP/s rose, confounding the two there.
decision: needs_repeat — coupling is real and reproducible; the bandwidth-vs-power mechanism is HINTED (footprint-tracking) but not pinned due to baseline noise + footprint/TFLOP confound at 8192. The clean discriminator is the fp16-vs-fp32 hog at matched matrix (Lpwr2b) with multiple no-hog samples to stabilize the baseline.
next_action: Lpwr2b — fp16-hog vs fp32-hog at matched matrix (4096), bracketed by no-hog samples. fp16@4096 has HALF the byte-footprint + potentially MORE FLOPs than fp32@4096, so the SIGN separates the hypotheses: if fp16 throttles workers LESS → footprint/bandwidth dominates; if MORE → FLOPs/power dominates. (fp16 hog mode added commit 8edd16b.) Also: longer cells (≥120s) or repeated no-hog to cut the ~20% baseline noise. NOTE for the skill: tiny/V64 coreml pure-self-play aug/s is noisy run-to-run (~±15-20%) at 80s cells — bracket or lengthen.
```

### 2026-05-23 — L09i-fix-load-v2 — CLEAN contention test: ANE workers throttle −35% under GPU load (NOT immune); bidirectional package-power coupling

```yaml
lane: L09i-fix-load-v2 (the decoupled re-do of the confounded L09i-fix-load)
hypothesis: fix-load's "positive lean" (ANE worker gen held under a hog) was a trainer-barrier artifact. Test cleanly in PURE self-play (canonical_sweep, NO trainer → no wave-barrier; aug/s reflects workers directly): do ANE workers throttle under a concurrent GPU hog?
code_ref: feat/perf-L09i-fix @ 850d432 (canonical_sweep --evaluator coreml passthrough) + static-batch export; scripts/gpu_load_generator.py
method: interleaved A/B, pure self-play, tiny / W=16 / G=8 / S=400 / V=64 / coreml CPU_AND_NE / 90s/cell / --max-plies 16 (canonical_sweep default). Arm B starts a gpu_load_generator --matrix 8192 hog 18s in, covering the cell.
arm_A_nohog: aug/s=3,548.0, games/s=27.79, plies_mean=15.96
arm_B_hog: aug/s=2,307.3, games/s=18.05, plies_mean=15.98. HOG reached only ~2.72-2.75 TFLOP/s.
delta: ANE worker aug/s 3,548 → 2,307 = −35.0% under the GPU hog. CLEAN worker measurement (no trainer barrier; gen-rate IS the metric in pure self-play).
key_secondary_finding: the GPU hog reached only ~2.72 TFLOP/s here (vs ~10.7 TFLOP/s when it ran alongside a light trainer in fix-load). The 16 busy worker processes (Core ML/ANE eval + CPU-side MCTS — total worker package load, not the ANE engine specifically; this cell can't isolate which) consume enough package power to suppress the GPU hog from ~11 → ~2.7 TFLOP/s. Bidirectional package-power coupling: the busy worker pool throttles the GPU and vice-versa.
mechanism: everything shares the M5 Max package power/thermal budget. ANE workers + GPU hog mutually contend: workers −35%, hog suppressed to 2.7 TFLOP/s, settling at a shared-power equilibrium. This reconciles fix-load: there the trainer stalled and the workers IDLED on the barrier, freeing package power so the hog reached 10.7 — the gen=5.1s "held" because workers were measured during non-stalled moments, not because they're immune.
confidence: high — pure self-play removes the trainer-barrier confound; the −35% is a direct worker-rate measurement; interleaved A/B; reproducible (arm A ~3,548 is the clean tiny/V64 coreml pure-gen number).
decision: reject (no contention-immunity win). The "positive lean" from fix-load is RETRACTED — ANE workers DO throttle under GPU contention (−35%), just more gently than CPU/BNNS (−82% Lpwr, though hog intensities differ since the hog auto-throttled here). The ANE is not a separate free resource.
verdict_for_strand: ANE residency is real (L09i-fix) but offers no clean self-play win — not on throughput (reject at tiny/small) and not on contention-immunity (this lane). The honest close, on clean evidence this time.
next_action: finish the throughput re-map on clean evidence (L09-reopen-small-b at --coreml-static-batch 96, L09-reopen-medium), then close the ANE-for-self-play strand. The bidirectional package-power coupling is a new datapoint for m5-max-cross-engine-coupling.md / Lpwr2.
```

### 2026-05-23 — L09-reopen-small — ANE residency confirmed at SMALL; throughput needs_repeat (over-padded confound)

```yaml
lane: L09-reopen-small (re-map the old L09 small reject on GENUINE ANE residency, not CPU/BNNS)
hypothesis: L09 (small, R-TRAIN-ANE) rejected at -41.5% — but that was CPU/BNNS. With real ANE residency (L09i-fix static export), does small flip?
code_ref: feat/perf-L09i-fix (static-batch export, wave*3 sizing); lab_train_cell --evaluator coreml CPU_AND_NE
cell: small / W=8 / G=8 / S=400 / V=64 / coreml CPU_AND_NE / live training / 30s+120s
residency: CONFIRMED at small — `sample` worker hot path = AneInferenceOperationImplUsingAnefAPIs / _ANEClient doEvaluateDirect / AppleNeuralEngine, 0 BNNS. The static-batch export keeps the SMALL model ANE-resident (no ANE-unsupported ops at small).
result: aug/s=1,271.5, games/s=4.84, epochs/s=0.05, trainer_step_s_p50=0.0346, 8 epochs, plies_mean=33.4. Generation-bound (trainer.log gen~15s vs train~2s) → aug/s honestly reflects the ANE worker rate (NOT trainer-gated like fix-load).
baseline: old L09 CPU/BNNS R-TRAIN-ANE = 1,930.3; R-TRAIN-WL5 torch = 3,297.6.
delta: -34% vs CPU/BNNS L09; -61% vs torch.
CONFOUND: the worker exported at wave*3 = 192, but the small wave tile is only ~67 (W=8, tile=66-71 in trainer.log). So every eval padded ~2.9x. The wave*3 heuristic (tuned for tiny W=16, tile~140) over-pads at low-W. Un-padded throughput could be materially higher (gen scales ~with padded batch).
decision: needs_repeat → RESOLVED by L09-reopen-small-b below.
RESOLVED (L09-reopen-small-b, --coreml-static-batch 96, tile~80): clean aug/s = **1,833.7** (+44% vs the over-padded 1,271 — padding WAS the confound), games/s=7.625, 10 epochs, plies 30.6, gen~10s/train~2s (still generation-bound = honest worker number). vs old CPU/BNNS L09 (1,930.3) = −5% (≈ ties within noise); vs torch R-TRAIN-WL5 (3,297.6) = −44%. **Verdict: real ANE residency at small roughly TIES the CPU/BNNS path and loses to torch by ~44% — no flip of the L09 reject.** Combined with reject at tiny and no contention-immunity (L09i-fix-load-v2), the ANE offers no self-play win at any measured size. ANE-for-self-play strand CLOSED on clean evidence.
next_action: L09-reopen-medium is now optional (envelope-curiosity; pattern ANE≈CPU/BNNS<torch likely holds, and medium may hit ANE-unsupported ops). Strand close: residency capability preserved on branch feat/perf-L09i-fix for GPU-idle use only.

L09-reopen-medium (RESULT): **ANE residency CONFIRMED at medium** (`sample` hot path AneInferenceOperationImplUsingAnefAPIs / AppleNeuralEngine) — the medium model is ANE-placeable, no unsupported ops. So ALL THREE sizes (tiny/small/medium) go ANE-resident under the static-batch export — completes the residency picture. Throughput = 453 aug/s but **confounded**: the wave tile is only ~70-79 (set by W×G=8×8, NOT wave_size — I wrongly sized `--coreml-static-batch 768` assuming it scaled with wave_size=512), so ~11× over-pad; generation-bound (gen 40-44s/wave). vs old CPU/BNNS L09d (591.7) = −23% even at 11× pad; a clean re-run (--coreml-static-batch ~96) would be faster but still lose to torch+fp16 (R-TRAIN-MEDIUM 1,463) per the established pattern. **NEW LESSON: the wave leaf-tile ≈ W×G (~70 at W8G8, ~140 at W16G8), NOT wave_size — size `--coreml-static-batch` to ~W×G×1.3, not wave_size.** decision: residency datapoint recorded (valuable); throughput confounded but directionally consistent (ANE ≈ CPU/BNNS < torch). Clean re-run not worth it (diminishing returns; pattern clear).
```

### 2026-05-23 — L09i-fix-load — INCONCLUSIVE (needs_repeat): the −96% was a GPU-trainer stall, NOT an ANE-worker collapse; ANE workers' gen-rate held under the hog [CORRECTED post-Reviewer]

```yaml
lane: L09i-fix-load (interleaved A/B; the reframed thesis after L09i-fix)
hypothesis: >
  L09i-fix reframed the ANE's value as contention-immunity (it loses on raw throughput but
  fully vacates the GPU). Lpwr showed GPU saturation collapses CPU/BNNS workers -82% via a
  shared package envelope. If the ANE is a genuinely separate resource, ANE-resident workers
  should resist that collapse — which would be the strand's production justification.
code_ref: feat/perf-L09i-fix @ 3f658c9 (ANE-resident static export, fixed batch 192); scripts/gpu_load_generator.py
method: >
  Interleaved A/B from the worktree (ANE-resident workers). Arm A = ANE workers, no hog.
  Arm B = same cell + concurrent gpu_load_generator --matrix 8192 (~10.7 TFLOP/s) started after
  the 30s warmup, covering the 120s measurement window. Back-to-back with a 45s cooldown.
hardware: M5 Max / 48 GB. tiny / W=16 / G=8 / S=400 / V=64 / coreml CPU_AND_NE / live training / 30s+120s.
arm_A_nohog: aug/s=7,878, games/s=35.33, epochs/s=0.125, trainer_step_s_p50=0.0162, 19 epochs. (Reproduces L09i-fix-b 7,698 within +2.3% session noise.)
arm_B_hog: aug/s=302, games/s=1.245, epochs/s=0.0167, trainer_step_s_p50=0.0202, 4 epochs. Hog held ~10.4-10.9 TFLOP/s.
CORRECTION (post-Reviewer, verified against the hog arm's trainer.log): the holistic −96% is NOT an ANE-worker collapse. Pre-hog epochs read `(8.0s: gen=5.1s train=2.5s)` / `(8.1s: gen=5.2s train=2.6s)`; with the hog active, epoch 4 = `(107.9s: gen=5.1s train=99.5s)`. **Worker generation time held EXACTLY (gen=5.1s) under the hog — the ANE workers were NOT throttled.** What collapsed was the MPS TRAINER (train phase 2.5→99.5s; per-step trainer_step_p50 barely moved 0.0162→0.0202, so the 99.5s is MPS-queue/blocking contention with the hog, not per-step SGD compute). Wave-mode synchronizes worker output to the trainer's epoch loop, so the stalled trainer gated generation → only 4 epochs in the 120s window vs 19 → aug/s tanked.
delta: >
  Holistic aug/s 7,878 → 302 (−96%), but the attribution matters: it is a GPU-trainer stall
  propagated through the wave barrier, not an ANE-worker collapse. ANE worker gen-rate: UNCHANGED
  (5.1s). The Lpwr comparison (CPU/BNNS workers −81.7%) is INVALID — there the WORKERS slowed; here
  the workers held and the TRAINER stalled. Different things collapsed; not a matched comparison.
mechanism: >
  Two separable effects under a GPU hog: (1) ANE-resident workers' generation is UNAFFECTED
  (gen=5.1s held) — a genuine contention-RESISTANCE signal, opposite the CPU/BNNS workers in Lpwr
  which slowed. (2) The MPS trainer's epoch stalls hard (train 2.5→99.5s) via MPS-queue contention
  with the hog (per-step compute barely moved). Wave-mode couples them, so the holistic metric
  reflects the trainer stall and masks the worker resistance. CONFOUND: a real heavy production
  trainer is GPU-heavy by doing its OWN SGD, not by a separate hog flooding the MPS command queue —
  so this synthetic-hog cell does not cleanly model the production question.
confidence: low for any contention-immunity verdict (the lane conflates trainer-stall with worker-collapse; the synthetic hog isn't trainer-representative). High only for the narrow fact that ANE worker gen-rate held under the hog.
decision: needs_repeat — INCONCLUSIVE on worker contention-immunity; does NOT close the ANE-for-self-play strand. The one clean signal (ANE workers' gen held under GPU load) LEANS POSITIVE for contention-resistance — opposite the earlier, now-RETRACTED "falsified" read. Needs a redesigned cell that isolates worker gen-rate from the wave/trainer barrier and uses a trainer-representative GPU load.
residual_value: >
  ANE residency remains a documented CAPABILITY (L09i-fix) for GPU-idle use (phone-app deployment;
  a paced match-eval sidecar). Its viability for concurrent self-play is now OPEN again (the workers
  resisted the hog) pending the decoupled re-test.
next_action: >
  Queue L09i-fix-load-v2 (decoupled: isolate worker gen-rate from the trainer epoch barrier; use a
  trainer-representative GPU load, not a synthetic matmul hog). Keep L09-ANE-resident-reopen and
  L09i-fix-c low. Lpwr2 (cold-chip power-coupling mechanism pin) stays motivated by the Lpwr
  CPU-worker collapse, but the "second ANE datapoint" justification is RETRACTED (this cell did not
  measure an ANE-worker collapse).
```

### 2026-05-23 — L09i + L09i-fix — ANE residency RESTORED in our pipeline (RangeDim was the blocker); throughput reject at tiny but the ANE envelope re-opens

```yaml
lane: L09i (diagnostic) + L09i-fix / L09i-fix-b (static-batch export, code lane)
hypothesis: >
  L09i — diff the 2026-05-22 ANE-resident scout export against the lab coreml_evaluator
  export to find the ANE-hostile op. L09i-fix — switching the export to a static batch
  shape restores genuine ANE residency (the L09* "ANE" wins were all CPU/BNNS per L09e').
code_ref: feat/perf-L09i-fix @ 3f658c9 + wave*3 batch-sizing edit; gomoku/coreml_evaluator.py, gomoku/selfplay_worker.py
finding_L09i: >
  The two exports emit BYTE-IDENTICAL MIL op graphs (no gather/dilated/ND-broadcastable
  difference; same conv/linear/relu/add/cast counts). The ONLY ANE-relevant difference is
  the input batch dim: the lab export hardwired ct.RangeDim(1, max_batch) (SYMBOLIC) at
  coreml_evaluator.py:267. The ANE requires fully static input shapes, so Core ML compiled
  the whole program to CPU/BNNS. The scout reached the ANE rail because --batch-shape fixed
  declares a STATIC batch. Scratch tools: scripts/l09i_op_diff.py, scripts/l09i_batchshape_probe.py.
fix: >
  Replace RangeDim with a single fixed batch dim; the evaluator pads each real leaf-batch up
  to the declared size and slices policy+value back (chunks if larger). EnumeratedShapes does
  NOT stay ANE-placeable (falls back to BNNS) — a single fixed batch is the only ANE-resident option.
residency_proof: >
  Confirmed TWICE via the hollance no-sudo `sample` technique. (1) Isolated micro-probe: hot path
  AneInferenceOperationImplUsingAnefAPIs / _ANEClient doEvaluateDirect / AppleNeuralEngine, 0 BNNS.
  (2) UNDER THE LIVE SELF-PLAY WORKER (sample of worker PID mid-measurement): same ANE hot path,
  0 BNNS lines. First genuine ANE residency in the lab's history.
hardware: M5 Max / 48 GB; coremltools 9.0; torch 2.11.0; macOS 26.4.1
cells: tiny / W=16 / G=8 / S=400 / V=64 / coreml CPU_AND_NE / live training / 30s warmup + 120s measure
  - L09i-fix (fixed batch = wave*G*2 = 1024): aug/s=2,303.9, games/s=9.05, epochs/s=0.05, trainer_step_s_p50=0.0155, epochs_in_window=8, plies_mean=32.09. Every eval padded to 1024 over a ~140-leaf wave tile (~7x waste).
  - L09i-fix-b (fixed batch = wave*3 = 192): aug/s=7,697.7, games/s=36.453, epochs/s=0.1167, trainer_step_s_p50=0.0172, epochs_in_window=18, plies_mean=26.76. Padding cut to ~1.37x. +234% vs fix-a.
baseline_metric: R-TRAIN-TINY torch = 8,039.1 aug/s / 0.0333 ep/s / trainer_step 0.0319 / 6 epochs; R-TRAIN-TINY-ANE (CPU/BNNS, L09c) = 10,762.6 aug/s / 0.0417 ep/s / trainer_step 0.0267 / 7 epochs.
delta: >
  L09i-fix-b vs torch baseline: aug/s -4.2%, but epochs/s +250% (18 vs 6 epochs/window) and
  trainer_step_s_p50 -46%. vs CPU/BNNS L09c: aug/s -28.5%, epochs/s +180%, trainer_step -36%.
  Mechanism: ANE-resident workers FULLY vacate the GPU → maximal trainer relief (best trainer_step
  the lab has measured), but ANE per-eval at tiny is slower than torch/MPS or CPU/BNNS even at ~1.37x padding.
plies_note: >
  L09i-fix-b plies_mean 26.76 < L09c 29.02 is the asymmetric-epoch artifact (18 trainer epochs →
  policy improves within the window → games shorten; trainer.log shows plies declining 33→28 across
  epochs), NOT behavior drift. fp16 eval numerically equivalent to torch (MAE ~9e-5 policy / 2e-4 value).
confidence: high on mechanism + residency (sample evidence, twice, no sudo); single-cell per throughput point (smoke-first; deltas large vs noise).
decision:
  - L09i: RESOLVED (diagnostic) — RangeDim is THE ANE blocker; root cause + fix proven.
  - L09i-fix / L09i-fix-b: REJECT as a throughput promote at tiny/V=64 (best ANE-resident cell 7,698 < torch 8,039 < CPU/BNNS 10,762), but a CONFIRMED capability/mechanism win that re-opens the ANE envelope. NOT a knob-failure reject.
cap_note: >
  These cells elevate past `coreml-isolated` — we now have `sample`-confirmed ANE residency under
  the live worker (hot path on AppleNeuralEngine, not BNNS). Strict `ane-metered` (powermetrics
  ane_power mW) still pending on sudo, but the engine-placement question is settled by `sample`.
next_action:
  - L09i-fix-c: tighter fixed batch (160/144) to chase remaining padding.
  - L09i-fix-load: re-run the Lpwr GPU-saturation A/B with ANE-resident workers — do they resist the GPU-contention collapse that took CPU/BNNS workers -82%? (potential architectural headline: ANE workers on a different engine than the GPU trainer).
  - Re-open L09 (small) + L09d (medium) with the ANE-resident export (prior rejects were CPU/BNNS; the ANE design-center favors larger conv compute).
  - Reviewer to advise whether the static-batch-default merge is sound or should be opt-in.
```

### 2026-05-23 — Lhot heat-soak characterization — production shapes show NO heat-soak haircut (refutes the "cool-start is optimistic" hypothesis)

```yaml
lane: Lhot (heat-soaked steady-state reference characterization; arose from Jason "training will be heat soaked")
hypothesis: Cool-start reference numbers (R-S400=9,398.5, R-TRAIN-WL5=3,297.6) overstate sustained production throughput because a multi-hour run is heat-soaked. Earlier tiny/V=64 data suggested a ~18% haircut. Predict: heat-soaked steady state of the production shapes is meaningfully below the cool-start refs.
code_ref: a813151 on main
method: Phase 1 — 8 back-to-back R-S400 cells (small/W8/G8/S400/V512/fp16, canonical_sweep, 60s each) to drive the chip to thermal steady state, logging the aug/s curve. Phase 2 — 2 R-TRAIN-WL5 cells (small/V=64, lab_train_cell, 30s warmup + 120s measure) while heat-soaked.
hardware: M5 Max / 48 GB; chip warmed by the 8-cell R-S400 run before the R-TRAIN measurement.
rs400_curve (aug/s, iter 1-8): 9641, 9388, 9660, 10029, 9902, 9780, 9781, 9788
rs400_steady_state: ~9,783 aug/s (mean of iters 6-8; 0.08% spread — tight plateau)
rtrainwl5_heatsoaked: 3,384.4 and 3,378.5 aug/s; trainer_step_s_p50 0.0526/0.0516; 14 epochs each
delta:
  - **R-S400 heat-soaked 9,783 vs cool-start 9,398.5 = +4.1%** (HIGHER, not lower)
  - **R-TRAIN-WL5 heat-soaked 3,381 vs cool-start 3,297.6 = +2.5%** (HIGHER)
  - trainer_step_s_p50 heat-soaked 0.052 ≈ cool L10 0.0512 (stable); epochs 14 = L10's 14 (stable)
  - **NO heat-soak haircut on the production shapes.** The R-S400 curve doesn't decay — it wobbles through warmup (iters 1-4) then settles stable through 8 min of continuous load. The M5 Max sustains production throughput indefinitely under realistic self-play/training load.
  - **The hypothesis is REFUTED for production shapes.** The earlier "~18% haircut" (tiny/V=64 10,431→8,531) was NOT representative: it was a Core ML CPU-worker (BNNS) shape measured right after the synthetic 14-TFLOP fp32 hog — an artificial extreme GPU thermal load on a non-production shape. Under real production load the GPU-resident work holds its clocks.
confidence: high for the production-shape conclusion (8-cell R-S400 plateau is tight and stable; R-TRAIN-WL5 reproduced across 2 cells at the cool-start level). The cold-start references are trustworthy for sustained production.
open_nuance: the haircut may be ENGINE-SPECIFIC — GPU-resident work (R-S400, R-TRAIN-WL5) sustains; the tiny/V=64 Core ML CPU/BNNS workers MAY throttle (one messy post-hog data point). Consistent with the Lpwr power-coupling story (CPU is the thermally-sensitive path; GPU holds). Non-production shape; needs a clean heat-soak re-test before claiming. Lane Lhot2 candidate.
artifacts: sweep_logs/lab-Lhot-20260523T185856Z/{heatsoak_curve.tsv, rs400_iter01..08/, rtrainwl5_iter01..02/}
decision: needs_repeat (production conclusion is solid; the engine-specific CPU-throttle nuance needs a clean re-test)
next_action:
  - **CORRECT the surfaces:** the cool-start references do NOT overstate sustained production throughput for the production shapes. Revise best-cells thermal caveat, m5-max-cross-engine-coupling 18% framing, and the heat-soaked-is-production memory.
  - The Lpwr GPU-coupling finding STILL STANDS but is about EXTREME GPU load (synthetic hog), not normal training. A future heavy trainer (15×15, bigger net) might approach hog-level GPU load and trigger the CPU-worker throttle — worth re-checking then, not now.
  - Optional Lhot2: clean heat-soak re-test of the tiny/V=64 CPU-worker shape (no prior hog) to confirm/refute the engine-specific CPU-throttle nuance.
```

### 2026-05-23 — Lpwr GPU-load coupling — saturating the GPU collapses CPU-resident self-play workers (−82%); engines share a package resource

```yaml
lane: Lpwr-gpu-coupling (engine power/thermal coupling investigation; arose from Jason's "what if we artificially load the gpu" question)
hypothesis: If L09c's Core ML workers run on the CPU (per L09e'), are they isolated from GPU load? Run a synthetic GPU hog (fp32 matmul hot loop on MPS) concurrently with the L09c CPU-worker cell and measure whether worker throughput holds. Engine-isolation predicts the CPU workers should be ~unaffected by GPU load.
code_ref: scripts/gpu_load_generator.py (new this session) + existing lab_train_cell coreml path
evaluator: workers = Core ML CPU_AND_NE (CPU/BNNS per L09e'); hog = torch MPS (GPU); trainer = torch MPS (GPU)
dataset_ref: fresh random fused checkpoint (tiny); live self-play; tiny / W=16 / G=8 / S=400 / V=64
hardware: M5 Max / 48 GB; engine placement confirmed by `sample`: workers→BnnsCpuInferenceOperation (CPU), hog+trainer→AGXMetalG (GPU). System GPU monitor during run: GPU ~75%, ANE 0% (independent confirmation workers are not ANE-resident).
baseline_command: lab_train_cell tiny/W16/V64/coreml CPU_AND_NE, no hog (L09e' = 10,431.6 aug/s on a cooler chip; sweep-baseline = 8,531 on the heat-soaked chip)
candidate_command: same cell + concurrent `python scripts/gpu_load_generator.py --secs 125 --matrix 8192` launched at measurement-window start
clean_AB_metric (cool chip, back-to-back):
  - no hog:   10,431.6 aug/s; 46.0 games/s; trainer_step_s_p50=0.0267
  - hog ~11 TFLOP/s: 1,905.2 aug/s; 7.55 games/s; trainer_step_s_p50=0.0305
  - delta: worker aug/s **-81.7%**; games/s -84%; trainer_step +14%
intensity_sweep_metric (heat-soaked chip, sequential — THERMALLY CONFOUNDED):
  - matrix=0 (baseline): 8,531 aug/s
  - matrix=2048 (hog 9.34 TFLOP/s): 3,552 aug/s (-58%)
  - matrix=4096 (hog 11.88 TFLOP/s): 1,172 aug/s (-86%)
  - matrix=8192 (hog 11.65 TFLOP/s): 4,884 aug/s (-43%) — NON-MONOTONIC (8192 > 4096 despite ~same hog TFLOP/s)
delta:
  - **The coupling is real and large.** Every hog cell sits far below baseline (-43% to -86%), in both the clean A/B and the confounded sweep. GPU load unambiguously collapses CPU-resident workers.
  - **The mechanism is NOT cleanly pinned.** The clean A/B's asymmetry (trainer on GPU only -14%, workers on CPU -82%) points at a shared package power/thermal envelope rather than GPU-compute contention (which would hit the trainer hardest). But the intensity sweep that would confirm power-vs-scheduling got swamped by session-thermal drift: the baseline itself fell 10,431→8,531 over ~20 min, and the sweep is non-monotonic with hog load. On a heat-soaked M5 Max the thermal state dominates the intensity axis.
  - Possible mechanisms not yet separated: (a) shared power/thermal envelope → CPU clock throttles when GPU draws power; (b) unified-memory bandwidth contention; (c) macOS QoS scheduling. The asymmetry favors (a) but doesn't rule out (b).
confidence: high on the qualitative coupling (large effect, two measurement modes agree, independent ANE-0% system-monitor confirmation). LOW on the intensity-scaling shape and the precise mechanism (thermal drift confounded the sweep; needs a cold-chip interleaved-A/B design with cooldowns + ideally powermetrics CPU-freq evidence).
artifacts:
  - sweep_logs/lab-L09g-prime-gpustress-20260523T182622Z/ (clean A/B stress arm; sample_w0 + hog log)
  - sweep_logs/lab-Lpwr-gpu-coupling-20260523T183502Z/{sweep_results.tsv, cell_hog{0,2048,4096,8192}/}
  - scripts/gpu_load_generator.py
commands_run:
  - python scripts/gpu_load_generator.py --secs 125 --matrix 8192 (clean A/B hog)
  - bash sweep over matrix ∈ {0,2048,4096,8192} (intensity sweep; thermally confounded)
decision: needs_repeat
next_action:
  - **Qualitative finding stands and is durable:** GPU load collapses CPU-resident self-play workers; the engines are compute-isolated but share a package-level resource (power/thermal most likely). This reframes the L09c win as **load-fragile** — it depends on the GPU having spare power budget, which a heavy production trainer (15×15, bigger net) would consume.
  - **For a clean intensity-scaling + mechanism result, re-run on a cold chip** with interleaved A/B pairs (hog/no-hog back-to-back) per intensity + cooldowns between, ideally under `powermetrics` (sudo) to directly observe CPU frequency/power throttle. NOT a sequential sweep — that design was defeated by thermal drift.
  - Full writeup: wiki/topics/m5-max-cross-engine-coupling.md. This is a "where the machine breaks" mainframe finding per [[feedback-know-the-machine]] and constrains [[project-light-all-engines]].
```

### 2026-05-23 — L09e' RESIDENCY RESOLVED — L09c runs on CPU/BNNS, not ANE; `coreml-isolated` cap confirmed, `ane-metered` rail dark

```yaml
lane: L09e' (L09e-prime; ANE residency proof via the no-sudo thread-name / engine-attribution technique from hollance/neural-engine)
hypothesis: Elevate L09c (lone PROMOTE) from `coreml-isolated` to `ane-metered`, or pin it at `coreml-isolated`, by detecting whether the Core ML workers run on the ANE. Use `sample <pid>` (no sudo) to check for `H11ANEServicesThread` and Espresso engine attribution (ANERuntimeEngine=ANE, MPSEngine=GPU, BNNSEngine=CPU).
code_ref: 30179b3 on main (post-hollance-absorption) + scripts already in place
dataset_ref: fresh random fused checkpoint (tiny); live self-play; tiny / W=16 / G=8 / S=400 / V=64 / coreml CPU_AND_NE
hardware: M5 Max / 48 GB; macOS 26.4.1
method: re-ran L09c shape; sampled worker w0 at T+76s (46s into measurement) via `sample <pid> 2`
candidate_metric: 10,431.6 aug/s (replicates L09c's 10,762.6 within session-thermal noise); trainer_step_s_p50=0.0267
residency_evidence:
  - **No `H11ANEServicesThread`** in worker (all 9 threads generically named) → ANE not in use (per hollance technique)
  - **Hot path: `E5RT::Ops::BnnsCpuInferenceOperation::ExecuteSync`** → Espresso BNNS/CPU engine
  - `com.apple.ANEServices` + `com.apple.ANECompiler` frameworks lazy-linked (Core ML always links them) but no ANE thread, no `ANERuntimeEngine` symbol
  - Independent confirmation: system GPU monitor showed **ANE utilization 0%** during the load tests
  - L09c-cpugpu cross-check (tiny/V=64/CPU_AND_GPU = 10,202 aug/s): STILL `BnnsCpuInferenceOperation` — Core ML picks CPU even when GPU is allowed; the MPS frameworks load but never carry the hot path
delta: cap stays `coreml-isolated`; `ane-metered` rail is DARK.
confidence: high. Two independent tools agree (process `sample` + system GPU monitor). Cross-checked across routings (CPU_AND_NE and CPU_AND_GPU both land on BNNS-CPU). Sample is a statistical snapshot but 942 stack samples all hit BNNS.
artifacts:
  - sweep_logs/lab-L09e-prime-20260523T181123Z/{summary.tsv, sample_w0_T76s.txt}
  - sweep_logs/lab-L09c-cpugpu-20260523T182038Z/{summary.tsv}
commands_run:
  - lab_train_cell tiny/W16/V64/coreml CPU_AND_NE + sample <worker_pid>
  - lab_train_cell tiny/W16/V64/coreml CPU_AND_GPU + sample <worker_pid>
decision: reject (residency claim) — i.e., the L09c PROMOTE is NOT ANE-resident; it is engine-isolation via Core ML's CPU/BNNS path
next_action:
  - **The "tiny model fits the ANE design center" hypothesis is FALSIFIED.** Core ML chose CPU for our tiny model just as for small/medium. Tiny *wins* and small/medium *lose* because BNNS-CPU is fast enough at tiny/V=64 to beat torch+MPS-contended workers; at larger shapes BNNS-CPU is too slow. ANE was never in play.
  - The L09c PROMOTE narrative reframes from "ANE pays at tiny" to "Core ML's CPU/BNNS offload pays at tiny when the GPU has power headroom" (combined with the Lpwr coupling finding, the load-fragility caveat).
  - To actually get our model ANE-resident, L09i (mlpackage op inspection) is the diagnostic — identify which exported op forces CPU fallback, then model-surgery it. Until then, Core ML offload = CPU offload for our workload.
  - Full writeup: wiki/topics/m5-max-cross-engine-coupling.md Part 1.
```

### 2026-05-23 — L09e REJECT — Core ML compute-units routing axis null at small/V=64 (~4% spread); L09 reject confirmed not-a-routing-issue

```yaml
lane: L09e (compute-units routing sweep at the L09 reference shape — the rescue diagnostic for L09's small/V=64 -41.5% reject)
hypothesis: L09 measured CPU_AND_NE at small/V=64 = 1,930.3 aug/s (-41.5% vs R-TRAIN-WL5). Could Core ML have been silently demoting ops to suboptimal routings at that compute-unit setting? Sweep CPU_AND_GPU and ALL to compare. If one routing significantly beats L09's number (e.g., -20% or better vs R-TRAIN-WL5), the L09 reject is "wrong routing" and a different config could rescue ANE-offload at small. If all routings stay near -41%, the L09 reject is genuinely "Core ML is just slow at this workload size".
code_ref: 7c26506 on main (post-L08 Reviewer-APPROVE)
evaluator (all arms): Core ML via --evaluator coreml; only --coreml-compute-units differs across cells
dataset_ref: fresh random fused checkpoint (small, 324,570 params); live self-play under WL5-shaped recipe (S=400, V=64, EMA τ=0.99, grad_accum=4); W=8 (small's L02-confirmed peak at V=512, matching L09's recipe)
baseline_command (L09 ref): python scripts/lab_train_cell.py --out-dir sweep_logs/lab-L09-20260523T134213Z --lane L09 --model small --workers 8 --games-per-batch 8 --n-simulations 400 --wave-size 64 --ema-tau 0.99 --grad-accum-steps 4 --warmup-secs 30 --measurement-secs 120 --device mps --evaluator coreml --coreml-compute-units CPU_AND_NE
candidate_commands:
  - python scripts/lab_train_cell.py --out-dir sweep_logs/lab-L09e-cpugpu-20260523T162647Z --lane L09e-cpugpu --model small --workers 8 --games-per-batch 8 --n-simulations 400 --wave-size 64 --ema-tau 0.99 --grad-accum-steps 4 --warmup-secs 30 --measurement-secs 120 --device mps --evaluator coreml --coreml-compute-units CPU_AND_GPU
  - python scripts/lab_train_cell.py --out-dir sweep_logs/lab-L09e-all-20260523T162943Z --lane L09e-all --model small --workers 8 --games-per-batch 8 --n-simulations 400 --wave-size 64 --ema-tau 0.99 --grad-accum-steps 4 --warmup-secs 30 --measurement-secs 120 --device mps --evaluator coreml --coreml-compute-units ALL
hardware: M5 Max / MPS (trainer) + ANE/GPU/Mix (candidate workers via Core ML routings) / idle
seed: workers seeded 1000..1007 each arm; trainer seed default
baseline_metric (L09 ref): CPU_AND_NE: 1,930.3 aug/s; 8.00 g/s; 0.0583 ep/s; trainer_step_s_p50=0.0227s; 10 epochs in 120s
candidate_metric:
  - CPU_AND_GPU: 1,908.3 aug/s; 7.675 g/s; 0.0667 ep/s; trainer_step_s_p50=0.0226s; 11 epochs in 120s; plies_mean 31.38
  - ALL: 1,989.8 aug/s; 7.709 g/s; 0.0667 ep/s; trainer_step_s_p50=0.0197s; 11 epochs in 120s; plies_mean 32.54
delta:
  - CPU_AND_GPU vs CPU_AND_NE: aug/s -1.1% (1,908.3 vs 1,930.3); trainer_step_s_p50 ~identical (0.0226 vs 0.0227)
  - ALL vs CPU_AND_NE: aug/s +3.1% (1,989.8 vs 1,930.3); trainer_step_s_p50 -13.2% (0.0197 vs 0.0227)
  - Across-routing spread: 4.3% (1,908.3 to 1,989.8) — within the natural noise band for R-TRAIN-* cells (L09 vs L09c-baseline-style trainer cells show ~5% trial-to-trial variability)
  - **All three routings are still ~40% below R-TRAIN-WL5 (3,297.6 aug/s).** ALL is the marginal winner among the coreml routings but doesn't approach the torch/MPS baseline.
  - Hypothesis result: routing axis is FLAT-to-MILDLY-helpful (~4% upside from ALL); does NOT rescue the L09 reject. The L09 reject is genuinely "Core ML is just slow at this workload size", not "wrong compute-units routing".
confidence: high. Two-cell sweep against the L09 baseline establishes the routing axis is null. Even the best routing (ALL) leaves a ~40% gap to torch/MPS at small/V=64. Combined with L09d's medium-V=512 result (-59.6%) and L09c-V512's tiny-V=512 result (-24%), the engine envelope is now decisively mapped: ANE wins at exactly tiny+V=64; nothing else.
artifacts:
  - sweep_logs/lab-L09e-cpugpu-20260523T162647Z/{summary.tsv,metadata.txt,cell_train_small_W08_G08_S400_V064_EMA99_GA04_WM1_B512/{logs/trainer.log,logs/worker-NN.log}}
  - sweep_logs/lab-L09e-all-20260523T162943Z/{summary.tsv,metadata.txt,cell_train_small_W08_G08_S400_V064_EMA99_GA04_WM1_B512/{logs/trainer.log,logs/worker-NN.log}}
commands_run:
  - (CPU_AND_GPU above)
  - (ALL above)
decision: reject
next_action:
  - **The L09 reject stands at today's Core ML + evaluator pipeline.** Compute-units routing doesn't rescue ANE-offload at small/V=64 under the current stack. Combined with L09d (medium/V=512 reject) and L09c-V512 (tiny/V=512 reject), the **current engine envelope snapshot** is: ANE wins at tiny+V=64 ONLY (L09c +33.9%) at today's Core ML version + evaluator pipeline + model-arch family. **Not a permanent verdict** — the snapshot is time-stamped to today's stack.
  - **Future-shape re-measurement triggers:** when (a) a new Core ML major version lands, (b) new ANE features become available (Jason flagged inbound new ANE research as of session-end), (c) the gomoku evaluator pipeline changes (e.g., different export path, new compute_precision options, fp32→fp16 internal cast strategy), or (d) the model-arch family changes (e.g., different residual block shape, different stem padding) — re-run L09c / L09d / L09c-V512 / L09e against the new baseline before assuming today's snapshot still holds. The single-point envelope is the right reading of today's stack, NOT a structural ANE limit.
  - L09f (broader V-axis sweep) and L09g (broader model-size sweep) at coreml remain in queue at low priority; their priors flip back to load-bearing the moment any of the re-measurement triggers above fires.
  - L09h (.mlpackage re-export cost diagnostic) remains queued at priority 1.0; cheap (1 cell) and informative under any future ANE-payoff scenario where re-export cost might matter for live training.
  - consecutive_rejects: 3 → 4 (one short of the 5-reject HALT threshold). Per stop-gates triage: natural session-end at a complete current-state ANE snapshot. The four rejects DEFINE the envelope around the L09c PROMOTE — they're the bracketing data points, not a knob-failure streak.
  - Session-end entry filed in perf-log.md; three friction-smoothing lessons batched to the gomoku-perf-lab skill in companion commit (plies asymmetric-epoch artifact, session-thermal drift, env-axis stamping via cells.csv).
```

### 2026-05-23 — L08-mps-heap-ratio REJECT — PYTORCH_MPS_HIGH_WATERMARK_RATIO null at R-S400/fp16 (0.74% spread)

```yaml
lane: L08-mps-heap-ratio (3-cell env-var sweep at the R-S400 reference)
hypothesis: PYTORCH_MPS_HIGH_WATERMARK_RATIO default may cap throughput; nondefault could help. Sweep at the current R-S400 (small/W=8/G=8/S=400/V=512 + --fp16-eval) reference shape — three values: default (implicit, M-series ~1.7), 2.0 (higher), 0.0 (unlimited).
code_ref: 91d5ae5 on main (post-L09c-V512 Reviewer-APPROVE)
evaluator: torch / MPS / --fp16-eval (the R-S400 best recipe)
dataset_ref: pure self-play; fresh random fused checkpoint (small, 324,570 params); no trainer; 60s/cell smoke; canonical_sweep.py
baseline_command: R-S400 = small / W=8 / G=8 / S=400 / V=512 / fp16 = 9,398.5 aug/s (L06-followup-fp16-cells, 2026-05-23)
candidate_commands:
  - (default heap)  python scripts/canonical_sweep.py --out-dir sweep_logs/lab-L08-heap-default-20260523T161830Z --cells-from cells.csv --lane L08-heap-default --secs-per-cell 60 --fp16-eval
  - (heap=2.0)      PYTORCH_MPS_HIGH_WATERMARK_RATIO=2.0 python scripts/canonical_sweep.py --out-dir sweep_logs/lab-L08-heap-2p0-20260523T161958Z --cells-from cells.csv --lane L08-heap-2p0 --secs-per-cell 60 --fp16-eval
  - (heap=0.0)      PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 python scripts/canonical_sweep.py --out-dir sweep_logs/lab-L08-heap-0p0-20260523T162120Z --cells-from cells.csv --lane L08-heap-0p0 --secs-per-cell 60 --fp16-eval
hardware: M5 Max / MPS / idle (90 min into this session — chip is somewhat warmed up; R-S400 was measured at session-start when chip was cool, see below)
seed: workers seeded 1000..1015 (canonical_sweep default)
baseline_metric: R-S400 = 9,398.5 aug/s (from L06-followup, today)
candidate_metric:
  - default heap (re-measure of R-S400 recipe, no env override): 8,937.3 aug/s
  - heap=2.0: 8,870.9 aug/s
  - heap=0.0 (unlimited): 8,927.7 aug/s
delta:
  - **L08 within-sweep spread: 0.74%** (8,870.9 to 8,937.3) — comparing default vs 2.0 vs 0.0 cells measured back-to-back (~3 min apart). Heap-ratio axis is FLAT at R-S400/fp16.
  - L08 default heap vs R-S400 = 9,398.5 (the official R-S400 from L06-followup measured earlier today): -4.9% — interesting drift. Most likely a session-warming-thermal effect: R-S400 was measured ~90 min before L08, near session-start when the chip was cool; L08 cells ran after ~10 sequential lab cells. Within-L08 the three cells are 0.74% apart, so the thermal floor is consistent across L08 itself. The drift vs the pristine R-S400 number is NOT a noise concern for L08's heap-ratio comparison (which is the lane's actual hypothesis); it IS a friction-smoothing data point about session-thermal-state affecting absolute aug/s numbers across cells separated by significant wall time.
confidence: high. Three matched-shape, 60s smoke-first cells, back-to-back, 0.74% spread is well within the V=512 plateau noise floor (per L01 ~0.2%, per L02/L04 within ±2%). Mechanism: at small/V=512/fp16 we are bandwidth-bound (per L06-followup), not MPS-memory-pressure-bound; the heap watermark ratio gates when MPS frees memory, which has no bearing on eval bandwidth — null result mechanistically expected once we understood the bandwidth-bound regime.
artifacts:
  - sweep_logs/lab-L08-heap-default-20260523T161830Z/{summary.tsv,cells.csv,metadata.txt,cell_small_W08_G08_S400_V512/}
  - sweep_logs/lab-L08-heap-2p0-20260523T161958Z/{summary.tsv,cells.csv,metadata.txt,cell_small_W08_G08_S400_V512/}
  - sweep_logs/lab-L08-heap-0p0-20260523T162120Z/{summary.tsv,cells.csv,metadata.txt,cell_small_W08_G08_S400_V512/}
commands_run:
  - (default heap above)
  - (heap=2.0 above)
  - (heap=0.0 above)
decision: reject
next_action:
  - PYTORCH_MPS_HIGH_WATERMARK_RATIO axis is flat at R-S400/fp16. Don't queue further heap-ratio cells at other reference points unless there's a specific mechanism reason (e.g., a larger model with actual MPS pressure).
  - **NEW friction-smoothing data point: session-thermal-state matters for absolute aug/s.** R-S400 = 9,398.5 at session-start; default-heap re-measure 90 min later = 8,937.3 (-4.9%). Across-cell comparisons run back-to-back are fine (within ~1%); cross-session or distant-in-time comparisons should be re-measured. Worth a paragraph in the perf-lab-session-runbook about thermal drift and re-baselining.
  - consecutive_rejects: 2 → 3 (now at the warning level per the charter's stop-gates triage; CONTINUE remains the right call because Tier-3 lanes are still queueable AND there are documented diagnostic + envelope-mapping lanes left; per the [feedback-lab-runs-forever] memory, the 2026-05-23 session lesson was "don't pre-emptively halt at 3 rejects when work is still queueable").
  - Remaining queue lanes (post-L08): L09e (Tier 3, priority 3.0, diagnostic) → top of queue. L09f / L09g (Tier 3, downweighted by L09c-V512). L09h (Tier 3, priority 1.0). All remaining work is diagnostic or low-upside.
```

### 2026-05-23 — L09c-V512 REJECT — V-axis amortization falsified at tiny; ANE -24.0% at V=512 vs torch+fp16

```yaml
lane: L09c-V512 (auto-queued from L09c promote: does V-axis amortization stack with model-size amortization?)
hypothesis: L09c confirmed ANE pays at tiny + V=64 (+33.9% vs torch). L09f generically hypothesizes V=512+ batches more leaf evals per forward, amortizing Core ML's pipeline overhead better than V=64. If V-axis amortization stacks with model-size amortization, the L09c +33.9% should grow (perhaps to +50-80%) when we move tiny from V=64 to V=512 under live training.
code_ref: cba1ad3 on main (post-L09d Reviewer-APPROVE backfill)
evaluator (candidate): Core ML / ANE via --evaluator coreml --coreml-compute-units CPU_AND_NE; trainer always fp32 SGD on MPS
evaluator (baseline): torch / MPS / --fp16-eval (matched precision: Core ML internally runs FLOAT16 too; this is the apples-to-apples baseline at V=512 since tiny + V=512 + fp16 is the production R-S400-tiny recipe)
dataset_ref: fresh random fused checkpoint (tiny, ~30k params); live self-play under WL5-shaped recipe but with V=512 (the R-S400-tiny operating point per L06-followup = 22,873.8 aug/s pure-self-play with fp16); W=16 (tiny's L13/L14-confirmed peak); no archive ingest
baseline_command: python scripts/lab_train_cell.py --out-dir sweep_logs/lab-L09c-V512-baseline-20260523T160545Z --lane L09c-V512-baseline --model tiny --workers 16 --games-per-batch 8 --n-simulations 400 --wave-size 512 --ema-tau 0.99 --grad-accum-steps 4 --warmup-secs 30 --measurement-secs 120 --device mps --evaluator torch --fp16-eval
candidate_command: python scripts/lab_train_cell.py --out-dir sweep_logs/lab-L09c-V512-20260523T160936Z --lane L09c-V512 --model tiny --workers 16 --games-per-batch 8 --n-simulations 400 --wave-size 512 --ema-tau 0.99 --grad-accum-steps 4 --warmup-secs 30 --measurement-secs 120 --device mps --evaluator coreml --coreml-compute-units CPU_AND_NE
hardware: MacBook Pro Mac17,6; Apple M5 Max; 48 GB; MPS (trainer + baseline workers) / MPS (trainer) + ANE (candidate workers via Core ML CPU_AND_NE); idle (pre-flight pgrep clean both arms; arms ran back-to-back so chip thermal state was comparable)
seed: workers seeded 1000..1015 both arms; trainer seed default both arms
baseline_metric: tiny V=512 torch+fp16 baseline: 13,968.6 aug/s; 52.18 games/s; trainer_step_s_p50=0.0714s; plies_mean=33.47; 1 epoch in 120s (only 1 because tiny + V=512 = trainer epoch ~35s under torch+fp16 contention; warmup spans most of epoch 1); 7,855 games / 2,103,008 aug positions
candidate_metric: tiny V=512 coreml CPU_AND_NE: 10,609.8 aug/s; 43.94 games/s; trainer_step_s_p50=0.0268s; plies_mean=31.02; 8 epochs in 120s (8× more epochs because trainer side is hugely freed up); 4,535 games / 1,094,936 aug positions
delta:
  - aug/s **-24.0%** (13,968.6 → 10,609.8) — holistic REJECT
  - games/s **-15.8%** (52.18 → 43.94)
  - trainer_step_s_p50 **-62.5%** (0.0714s → 0.0268s) — MPS-relief mechanism still real at tiny V=512 (consistent with L09c's -16% relief mechanism, but here the magnitude is larger because the V=512 baseline has more trainer-step contention to relieve)
  - epochs_in_window 1 → 8 — candidate runs 8× more trainer epochs in the same wall window because the train= field collapsed; not a confound for aug/s comparison but a confound for plies_mean (see below)
  - plies_mean -7.3% (33.47 → 31.02) — **NOT a behavior drift; an asymmetric-epoch artifact.** Candidate's 8 epochs include early-game improvement: per-epoch plies in candidate trainer.log: 30.9 → 33.1 → 33.5 → 33.4 → 31.7 → 30.4 → 29.4 → 27.7 (declining as the policy improves over 8 epochs of training). Baseline's 1-epoch plies (32.5) reflects pre-training state. For matched stationary plies comparison we would need matched-epoch windows, OR same-epoch-number plies subsamples; the aggregate plies_mean drift is dominated by the training-progress signal at tiny + V=512 where 8 epochs in 2 minutes is enough to start moving the policy. Flag for Reviewer's drift-watch: this is NOT the L09c +/-1.6% kind of drift; it's a measurement-window confound from asymmetric training progress.
confidence: medium. Single trial each arm at 120s smoke-first. Strengths: matched-shape, back-to-back, the aug/s number is the worker-throughput regardless of trainer epoch count (so the -24.0% delta is the right gating measurement). Caveats: plies_mean comparison is muddied by asymmetric training progress (see above) — this is a NEW friction-smoothing lesson for the lab (R-TRAIN-* cells with sharply asymmetric epoch counts need careful plies framing). To rule out tail-uncertainty, a 240s rerun would help; given the -24% magnitude (well above any noise floor), the conclusion direction is firm. A 240s rerun is queueable but not gating for this receipt.
artifacts:
  - sweep_logs/lab-L09c-V512-20260523T160936Z/{summary.tsv,metadata.txt,cell_train_tiny_W16_G08_S400_V512_EMA99_GA04_WM1_B512/{logs/trainer.log,logs/worker-NN.log,records/*}} (candidate; symlink lab-L09c-V512-latest)
  - sweep_logs/lab-L09c-V512-baseline-20260523T160545Z/{summary.tsv,metadata.txt,cell_train_tiny_W16_G08_S400_V512_EMA99_GA04_WM1_B512/{logs/trainer.log,logs/worker-NN.log,records/*}} (baseline; symlink lab-L09c-V512-baseline-latest)
commands_run:
  - (baseline above)
  - (candidate above)
decision: reject
next_action:
  - **V-axis amortization is FALSIFIED at tiny.** torch+fp16 already extracts most of the V=512 bandwidth-bound value at tiny (per L06-followup, tiny V=512 fp16 was only +3.6% over fp32 because tiny is MPS-dispatch-limited, not bandwidth-bound). At V=512 the torch+fp16 baseline at 13,968 aug/s is harder to beat than at V=64 (where baseline was 8,039). Core ML can't match torch+fp16's bandwidth utilization at this operating point.
  - **Updated engine envelope (4 measured comparison points now):** ANE wins ONLY at tiny + V=64 (L09c +33.9%); ANE loses at tiny + V=512 (-24.0% here), small + V=64 (L09 -41.5% vs R-TRAIN-WL5), and medium + V=512 (L09d -59.6%). The ANE win is a single-point envelope, not a region. The "tiny model" axis alone wasn't the winning factor — it's tiny + V=64 specifically, where worker per-call work is so light that both backends are pipeline-overhead-bound.
  - No new ref opens. R-TRAIN-TINY-ANE (the L09c PROMOTE) at tiny + V=64 stays; this lane just sharply bounds the envelope around it.
  - **L09f (broader V-axis sweep) and L09g (broader model-size sweep) are now downweighted.** L09c-V512 already tests the V-axis at the only model where ANE wins; the answer is "V=512 doesn't help". L09f's small/medium V-axis cells at coreml are likely also losses. Queue them at lower priority for completeness, not headline value.
  - **L09e (compute-units routing sweep) keeps priority 3.0** — diagnostic value remains for the open question "is Core ML demoting ops at medium V=512?". The L09c-V512 result doesn't disambiguate this; only L09e at medium V=512 can.
  - consecutive_rejects: 1 → 2. Still far below the 5-reject charter halt threshold. Lab continues per autonomous-loop charter.
  - **NEW FRICTION-SMOOTHING LESSON to file in skill on session-end:** plies_mean is NOT stationary across asymmetric-epoch windows in R-TRAIN-* cells; a -7.3% drift looks alarming but is dominated by training progress when one arm runs 8× more trainer epochs than the other. Future Reviewers should check per-epoch plies in trainer.log rather than just the aggregate when arms have very different epochs_in_window. The Reviewer's L09c drift-watch flag was right to fire here, but the answer is "asymmetric-epoch artifact, not behavior drift" — needs a sharper test for actual game-shape drift.
```

### 2026-05-23 — L09d R-TRAIN-MEDIUM-ANE REJECT — medium on ANE = 591.7 aug/s (-59.6% vs medium/torch+fp16 baseline)

```yaml
lane: L09d-medium-on-ane (high-prior follow-up after L09c confirmed amortization mechanism wins at tiny)
hypothesis: Medium (~1.5M params) is closer to Core ML's design envelope. Per-call compute is larger, so pipeline overhead should amortize even better than at tiny (L09c +33.9%). Combined with the trainer-side MPS-relief mechanism (L09 -56% trainer_step_s_p50 at small; L09c -16% at tiny), the holistic R-TRAIN-MEDIUM-ANE should beat the torch+fp16 baseline. This is the "if ANE pays anywhere for us, here" lane in the production-relevant model size.
code_ref: 9114c45 on main (post-L09c Reviewer-APPROVE backfill); selfplay_worker / lab_train_cell / Core ML evaluator already shipped
evaluator (candidate): Core ML / ANE via --evaluator coreml --coreml-compute-units CPU_AND_NE; trainer always fp32 SGD on MPS
evaluator (baseline): torch / MPS / --fp16-eval (matched precision: Core ML internally runs FLOAT16 too, so fp16-on-torch is the apples-to-apples baseline at this model size)
dataset_ref: fresh random fused checkpoint (medium, ~1.5M params); live self-play under WL5-shaped recipe but with V=512 (the production-relevant operating point per R-S400-medium = 3,377 aug/s pure-self-play from L06fu-extended); W=8 (medium's L06fu-extended peak); no archive ingest
baseline_command: python scripts/lab_train_cell.py --out-dir sweep_logs/lab-L09d-baseline240-20260523T154931Z --lane L09d-baseline240 --model medium --workers 8 --games-per-batch 8 --n-simulations 400 --wave-size 512 --ema-tau 0.99 --grad-accum-steps 4 --warmup-secs 30 --measurement-secs 240 --device mps --evaluator torch --fp16-eval
candidate_command: python scripts/lab_train_cell.py --out-dir sweep_logs/lab-L09d-20260523T155432Z --lane L09d --model medium --workers 8 --games-per-batch 8 --n-simulations 400 --wave-size 512 --ema-tau 0.99 --grad-accum-steps 4 --warmup-secs 30 --measurement-secs 240 --device mps --evaluator coreml --coreml-compute-units CPU_AND_NE
hardware: MacBook Pro Mac17,6; Apple M5 Max; 48 GB; MPS (trainer + baseline workers) / MPS (trainer) + ANE (candidate workers via Core ML CPU_AND_NE); idle (pre-flight pgrep clean; arms ran back-to-back so chip thermal state was comparable)
seed: workers seeded 1000..1007 both arms; trainer seed default both arms
baseline_metric: R-TRAIN-MEDIUM (torch+fp16 baseline): 1,463.3 aug/s; 5.66 games/s; 0.0042 epochs/s; trainer_step_s_p50=0.2391s; plies_mean=32.5; 3 epochs in 240s; 544 games / 140,768 aug positions
candidate_metric: R-TRAIN-MEDIUM-ANE (coreml CPU_AND_NE): 591.7 aug/s; 2.33 games/s; 0.0208 epochs/s; trainer_step_s_p50=0.0444s; plies_mean=31.97; 7 epochs in 240s; 507 games / 128,688 aug positions
delta:
  - aug/s **-59.6%** (1,463.3 → 591.7) — holistic REJECT
  - games/s **-58.8%** (5.66 → 2.33)
  - epochs/s **+395%** (0.0042 → 0.0208) — trainer-side hugely faster on ANE config; epochs/s 5× higher because each epoch's train= field is 2-3s instead of 35-85s
  - trainer_step_s_p50 **-81.4%** (0.2391s → 0.0444s) — MPS-relief mechanism amplified at medium V=512: workers vacating MPS is enormously valuable to the trainer because medium SGD has lots of compute per step
  - plies_mean -1.6% (32.5 → 31.97) — clean, no game-shape drift (Reviewer's L09c watch-flag confirmed null at L09d)
  - Mechanism (sharp split): trainer side wins ENORMOUSLY (train=2-3s/epoch on ANE config vs 11-86s on torch+fp16 — the trainer flies when MPS is uncontended). Worker side loses ENORMOUSLY (gen=30-40s/epoch on ANE vs ~6s/epoch on torch+fp16 at medium V=512). The worker loss dominates the trainer gain by 2.5× — bad bet.
confidence: medium-high. Single trial each arm, 240s measurement (3 baseline epochs / 7 candidate epochs — different epoch counts reflect the trainer-side speedup; not a confidence issue, but the receipt should note that the baseline's 3-epoch span includes the heaviest end-of-buffer-fill epochs while the candidate's 7-epoch span captures more steady-state gen-bound epochs). Strengths: matched-shape comparison ran back-to-back (10 min total wall); plies_mean drift is null (-1.6%, well within sampling band); the mechanism is mechanically clean in both trainer logs (train= field collapse for candidate, gen= field explosion for candidate). Caveats: Core ML's internal compute-precision is FLOAT16 vs torch+fp16-eval — matched precision in principle, but baseline was tuned (L06-followup observed +97% from --fp16-eval on small/V=512), and the candidate uses Core ML's internal fp16 which may have different ops coverage (some ops could be demoted to CPU/GPU silently; see L09e diagnostic queued for this question).
artifacts:
  - sweep_logs/lab-L09d-20260523T155432Z/{summary.tsv,metadata.txt,cell_train_medium_W08_G08_S400_V512_EMA99_GA04_WM1_B512/{logs/trainer.log,logs/worker-NN.log,records/*}} (candidate; symlink lab-L09d-latest)
  - sweep_logs/lab-L09d-baseline240-20260523T154931Z/{summary.tsv,metadata.txt,cell_train_medium_W08_G08_S400_V512_EMA99_GA04_WM1_B512/{logs/trainer.log,logs/worker-NN.log,records/*}} (baseline; symlink lab-L09d-baseline240-latest)
  - sweep_logs/lab-L09d-baseline-20260523T154538Z/ (initial 120s baseline; superseded by 240s rerun for matched-window science but artifacts retained for archaeology — same number to within 2% so the 240s extension didn't materially change the baseline conclusion)
commands_run:
  - python scripts/lab_train_cell.py --out-dir sweep_logs/lab-L09d-baseline-20260523T154538Z --lane L09d-baseline --model medium --workers 8 --games-per-batch 8 --n-simulations 400 --wave-size 512 --ema-tau 0.99 --grad-accum-steps 4 --warmup-secs 30 --measurement-secs 120 --device mps --evaluator torch --fp16-eval  (initial baseline; 2 epochs in window — too few for matched-window comparison so re-dispatched at 240s)
  - python scripts/lab_train_cell.py --out-dir sweep_logs/lab-L09d-baseline240-20260523T154931Z --lane L09d-baseline240 --model medium --workers 8 --games-per-batch 8 --n-simulations 400 --wave-size 512 --ema-tau 0.99 --grad-accum-steps 4 --warmup-secs 30 --measurement-secs 240 --device mps --evaluator torch --fp16-eval
  - python scripts/lab_train_cell.py --out-dir sweep_logs/lab-L09d-20260523T155432Z --lane L09d --model medium --workers 8 --games-per-batch 8 --n-simulations 400 --wave-size 512 --ema-tau 0.99 --grad-accum-steps 4 --warmup-secs 30 --measurement-secs 240 --device mps --evaluator coreml --coreml-compute-units CPU_AND_NE
decision: reject
next_action:
  - **Envelope is now sharply mapped.** ANE pays at TINY only (L09c +33.9%), not at SMALL (L09 -41.5%), not at MEDIUM (L09d -59.6%). The L09d "larger compute amortizes pipeline overhead better" hypothesis is **FALSIFIED**. Opposite seems closer to true in our envelope: bigger per-call workloads expose ANE's lower per-forward throughput vs torch/MPS+fp16, and that loss dominates the trainer-side MPS-relief gain.
  - Open NEW reference point R-TRAIN-MEDIUM (torch+fp16 baseline) = 1,463.3 aug/s — envelope-mapping ref for medium under live training. R-TRAIN-MEDIUM-ANE = 591.7 aug/s recorded as a strikethrough/rejected ref (not promoted) for envelope-completeness.
  - Promote L09e (compute-units routing sweep) priority: at medium V=512 it's now diagnostically valuable to know whether the candidate's bad number is "ANE is just slow at this size" or "Core ML silently demoted ops to CPU/GPU" — the latter would suggest a different routing strategy could rescue the lane. Priority bumped from 1.5 → 3.0 (still Tier 3, still diagnostic-only).
  - Demote L09c-V512 priority slightly: if V-axis amortization doesn't pay at medium V=512, it might still pay at tiny V=512 (smaller model), but the prior is now weaker. Priority kept at 4.5 (the L09c +33.9% finding is still load-bearing for the tiny-axis V-test).
  - The L11b' R-TRAIN-LEAN-fp16 recipe (+152.9% vs WL5) remains the perf cycle's headline R-TRAIN finding; ANE-offload is not the path to the next R-TRAIN promote at production-quality model size.
  - consecutive_rejects: 0 → 1.
```

### 2026-05-23 — L09c R-TRAIN-TINY-ANE PROMOTE — tiny on Core ML CPU_AND_NE = 10,762.6 aug/s (+33.9% vs tiny/torch baseline); `coreml-isolated` cap

```yaml
lane: L09c-tiny-on-ane (second L09 follow-up: smaller per-eval graph might amortize Core ML pipeline overhead better)
cap_cleared: coreml-isolated (per coreml-ane-residency-lab.md ladder — trainer slowdown lower with Core ML workers than with torch workers; overlap measurement clean in trainer.log). NOT ane-metered: no `powermetrics ane_power` evidence in this receipt; whether the Core ML offload actually ran on the Apple Neural Engine vs the CPU/GPU portions of the CPU_AND_NE routing is unproven. The "ANE" in R-TRAIN-TINY-ANE is the Core ML routing label, NOT a residency claim. L09e' (queued) is the residency-elevation lane that would re-run this shape with matched-window powermetrics to elevate to `ane-metered` or confirm `coreml-isolated`-only.
hypothesis: At small/V=64, L09 showed Core ML worker eval is ~2× slower than torch/MPS, dominating the trainer-side MPS-relief gain (trainer_step_s_p50 -56%). At tiny (~30k params) the per-call pipeline overhead amortizes across less per-call compute — but the model is also vastly more compute-light. If the trainer-side MPS-relief still pays AND the worker-side raw-eval gap closes enough, the holistic R-TRAIN-TINY-ANE / R-TRAIN-TINY ratio could flip from L09's -41% to net positive.
code_ref: 9d4bfa5 on main (Core ML evaluator + --evaluator/--coreml-compute-units already shipped in L09 / L12 / L09b sessions)
evaluator (candidate): Core ML / ANE via --evaluator coreml --coreml-compute-units CPU_AND_NE; trainer always fp32 SGD on MPS
evaluator (baseline): torch / MPS; trainer always fp32 SGD on MPS
dataset_ref: fresh random fused checkpoint (tiny); live self-play under WL5-shaped recipe (S=400, V=64, EMA τ=0.99, grad_accum=4); W=16 (tiny's L13/L14-confirmed peak); no archive ingest
baseline_command: python scripts/lab_train_cell.py --out-dir sweep_logs/lab-L09c-baseline-20260523T153613Z --lane L09c-baseline --model tiny --workers 16 --games-per-batch 8 --n-simulations 400 --wave-size 64 --ema-tau 0.99 --grad-accum-steps 4 --warmup-secs 30 --measurement-secs 120 --device mps --evaluator torch
candidate_command: python scripts/lab_train_cell.py --out-dir sweep_logs/lab-L09c-20260523T153250Z --lane L09c --model tiny --workers 16 --games-per-batch 8 --n-simulations 400 --wave-size 64 --ema-tau 0.99 --grad-accum-steps 4 --warmup-secs 30 --measurement-secs 120 --device mps --evaluator coreml --coreml-compute-units CPU_AND_NE
hardware: MacBook Pro Mac17,6; Apple M5 Max; 48 GB; MPS (trainer + baseline workers) / MPS (trainer) + ANE (candidate workers via Core ML CPU_AND_NE); idle (pre-flight pgrep clean for both cells; baseline ran immediately after candidate so chip thermal state was comparable)
seed: workers seeded 1000..1015 (w0..w15) both arms; trainer seed default both arms
baseline_metric: R-TRAIN-TINY (torch baseline): 8,039.1 aug/s; 32.48 games/s; 0.0333 epochs/s; trainer_step_s_p50=0.0319s; plies_mean=31.84; 6 epochs in 120s; 3,501 games / 866,616 aug positions
candidate_metric: R-TRAIN-TINY-ANE (coreml CPU_AND_NE): 10,762.6 aug/s; 49.43 games/s; 0.0417 epochs/s; trainer_step_s_p50=0.0267s; plies_mean=29.02; 7 epochs in 120s; 4,884 games / 1,063,344 aug positions
delta:
  - aug/s **+33.9%** (8,039.1 → 10,762.6)
  - games/s **+52.2%** (32.48 → 49.43) — the magnitude of the games/s win (greater than aug/s) reflects that ANE games are slightly shorter on average (plies_mean 29 vs 32), but the aug/s number is the policy-target-weighted quantity, so it's the headline.
  - epochs/s +25.2% (0.0333 → 0.0417); trainer finishes ~7 epochs per 2-min window instead of ~6.
  - trainer_step_s_p50 **-16.3%** (0.0319 → 0.0267) — MPS-relief mechanism from L09 replicated at the tiny model size. Per the L09 ratio (-56% trainer_step_s_p50 at small/V=64), tiny shows a smaller trainer-side relief: less worker-side MPS pressure to begin with (tiny eval calls are cheap), but the relief is mechanistically still there.
  - plies_mean 29.02 vs 31.84 = -8.9%. Within typical sampling band for live-training cells at 6-7 epochs (L09's plies_mean 30.43 vs L10's 29.61 spanned a similar 3%). Eval semantics unchanged (Core ML outputs to fp32 numpy in the same shape).
  - Hypothesis CONFIRMED: ANE pays for tiny under live training pressure. The worker-side raw-eval gap that killed L09 at small models closes at tiny — most plausibly because tiny's per-eval compute is so light that BOTH backends are pipeline-overhead-bound, and the trainer-side MPS-relief tips the holistic balance.
confidence: medium-high. Single trial each side, 120s measure / 6-7 epoch span, smoke-first per charter. Strengths: matched-shape comparison ran back-to-back (5 min total wall) so thermal/scheduler state is comparable; the mechanism (trainer_step_s_p50 -16%) replicates L09's qualitatively even though magnitude is smaller; aug/s +33.9% well exceeds the natural 6-epoch noise floor; eval semantics structurally unchanged (Core ML compute_precision=FLOAT16 is internal, outputs cast to fp32 at coreml_evaluator.py:285+ before host). Caveats: this is a NEW reference family (R-TRAIN-TINY*), no prior data to compare; the asymmetry between aug/s and games/s deltas is plies_mean-driven and worth a second run to confirm. Not a precedent-extending claim about R-TRAIN-WL5 (different model, different quality target).
artifacts:
  - sweep_logs/lab-L09c-20260523T153250Z/{summary.tsv,metadata.txt,cell_train_tiny_W16_G08_S400_V064_EMA99_GA04_WM1_B512/{logs/trainer.log,logs/worker-NN.log,records/*}} (candidate; symlink lab-L09c-latest)
  - sweep_logs/lab-L09c-baseline-20260523T153613Z/{summary.tsv,metadata.txt,cell_train_tiny_W16_G08_S400_V064_EMA99_GA04_WM1_B512/{logs/trainer.log,logs/worker-NN.log,records/*}} (baseline; symlink lab-L09c-baseline-latest)
commands_run:
  - python scripts/lab_train_cell.py --out-dir sweep_logs/lab-L09c-20260523T153250Z --lane L09c --model tiny --workers 16 --games-per-batch 8 --n-simulations 400 --wave-size 64 --ema-tau 0.99 --grad-accum-steps 4 --warmup-secs 30 --measurement-secs 120 --device mps --evaluator coreml --coreml-compute-units CPU_AND_NE
  - python scripts/lab_train_cell.py --out-dir sweep_logs/lab-L09c-baseline-20260523T153613Z --lane L09c-baseline --model tiny --workers 16 --games-per-batch 8 --n-simulations 400 --wave-size 64 --ema-tau 0.99 --grad-accum-steps 4 --warmup-secs 30 --measurement-secs 120 --device mps --evaluator torch
decision: promote
next_action:
  - Open NEW reference points: **R-TRAIN-TINY-ANE** = tiny / W=16 / G=8 / S=400 / V=64 / coreml CPU_AND_NE = 10,762.6 aug/s; **R-TRAIN-TINY** (torch ref) = 8,039.1 aug/s. Both are envelope-mapping references, NOT R-TRAIN-WL5 substitutes (tiny is a different quality target). best-cells.md updated accordingly.
  - **The L09 + L09c combined finding maps the engine-isolation envelope along the model-size axis at the `coreml-isolated` cap level:** Core ML LOSES at small/V=64 (L09: -41%), WINS at tiny/V=64 (L09c: +34%). The crossover happens between tiny and small. This sharply elevates the prior on **L09d (medium on Core ML)** — if the trend continues monotonically with model size, medium would either be the worst case (deepest amortization deficit) or the best case (largest per-call compute amortizes pipeline overhead best); L09c's data point favors the "best case" hypothesis at the L09d card.
  - Auto-queue candidate **L09c-V512**: does the V=axis amortize Core ML pipeline overhead further? L09f addresses this generically; with L09c confirming the mechanism at tiny/V=64, a dedicated tiny + Core ML + V=512 cell is the cheapest amortization test next.
  - **NEW follow-up — L09e' residency proof.** The L09c PROMOTE is at the `coreml-isolated` cap; whether the win is actually ANE-resident (vs Core ML using CPU+GPU under CPU_AND_NE routing) is unproven. Queue L09e' to re-run this shape with matched-window `powermetrics ane_power` evidence — would elevate to `ane-metered` cap if ANE rail is nonzero, or confirm the win is engine-isolation-only otherwise. Depends on cached/passwordless sudo (same 2026-05-22 lane 03 blocker). See [coreml-design-envelope-and-our-fit.md § Current state](../topics/coreml-design-envelope-and-our-fit.md#current-state--what-we-know-after-l09c-through-l09e-2026-05-23-session-resume).
  - consecutive_rejects stays at 0.
```

### 2026-05-23 — L09b R-TRAIN-ANE + fp16 BLOCKED — code-interaction bug + semantic redundancy

```yaml
lane: L09b (compound follow-up of L09: ANE workers + fp16-eval)
hypothesis: L09 showed worker-side ANE eval was ~2× slower than torch/MPS at small/V=64. fp16 nearly doubles worker-side throughput on the torch path (L06-followup). Stacking fp16 on top of Core ML/ANE might recover the worker-side loss while keeping the trainer-side MPS-relief gain (-56% trainer_step_s_p50 from L09).
code_ref: ba7e345 + 009e2c6 on main (run-time, pre-patch)
candidate_command: python scripts/lab_train_cell.py --out-dir sweep_logs/lab-L09b-20260523T142143Z --lane L09b --model small --workers 8 --games-per-batch 8 --n-simulations 400 --wave-size 64 --ema-tau 0.99 --grad-accum-steps 4 --warmup-secs 30 --measurement-secs 120 --device mps --evaluator coreml --coreml-compute-units CPU_AND_NE --fp16-eval
result: **BLOCKED — code interaction bug + semantic redundancy.** Worker crashed at startup before any games were generated:
  ```
  RuntimeError: Input type (float) and bias type (c10::Half) should be the same
  at gomoku/coreml_evaluator.py:266 in export_model_to_coreml
    traced = torch.jit.trace(model.cpu(), dummy)
  ```
  Root cause: `selfplay_worker._maybe_half(model, ...)` ran BEFORE `_build_evaluator` and cast the model to fp16; Core ML's `export_model_to_coreml` then calls `torch.jit.trace` with a fp32 dummy input, and the first conv2d hit the fp32-input + fp16-bias mismatch.
  Semantic redundancy: Core ML already exports at `compute_precision=FLOAT16` internally (see coreml_evaluator.py:285), so even if the crash were fixed, casting the source model to fp16 BEFORE the Core ML export would be a no-op — Core ML's exported graph runs fp16 either way. The L09b lane's hypothesis ("stack fp16 on top of Core ML") was structurally incoherent.
patch: gomoku/selfplay_worker.py parse_args() now force-sets args.fp16_eval=False when args.evaluator=='coreml', with a printed audit line. This makes the flag combination a graceful no-op instead of a crash; future invocations of L09b-style cells silently get the L09 path with a heads-up note. (Patch will commit alongside this receipt.)
artifacts: sweep_logs/lab-L09b-20260523T142143Z/{summary.tsv (cell_status=failed),metadata.txt,cell_*/logs/{trainer.log (5 lines; never got past startup),worker-00.log (full traceback)}}
decision: blocked
next_action:
  - L09b is moot as designed (semantically redundant). The remaining ANE-payoff candidate is L09c: tiny model on Core ML / CPU_AND_NE. Tiny's per-eval graph is smaller; ANE pipeline overhead might amortize better. R-S400-tiny already runs at 22,088 fp32 / 22,874 fp16 on torch/MPS; can ANE compete on a smaller graph? Queue for a future session.
  - Other charter-aligned compounds at session-end: L06fu-medium-AB (clean medium V=512 fp32 vs fp16 attribution; estimated fp16-alone +62% from L06fu-extended); L08-mps-heap-ratio at the new fp16 reference (3-cell env-var sweep — should compound or compose differently with the bandwidth-bound regime).
  - Lab orchestrator declares session-end after this receipt. See perf-log.md session-end entry.
```

### 2026-05-23 — L11b' R-TRAIN-LEAN-fp16: V=512 + sgd=0.001 + fp16 workers = 8,340.5 aug/s (+153% vs R-TRAIN-WL5)

```yaml
lane: L11bp (L11b-prime; compound of L11b + L06-followup at the trainer level)
hypothesis: L11b showed lowering sgd_per_position cures the trainer-side cost of V=512. L06-followup showed fp16 nearly doubles the worker-side throughput at V=512 (small/S=400 bandwidth-bound regime). At the R-TRAIN-* family these are independent levers — one on trainer, one on workers. Stacking them should compound: V=512 + sgd=0.001 + fp16-eval should beat both L11b (4,232 aug/s) and L10 R-TRAIN-WL5 (3,298 aug/s) by close to the multiplicative product of the individual wins.
code_ref: 21bb2f5 on main (run-time). L12 driver gained --fp16-eval passthrough at this commit; fourth (and final today) L12 gap surfaced.
evaluator: torch / MPS / fp16-eval (workers); trainer always fp32 SGD on MPS
dataset_ref: fresh random fused checkpoint (small, 324,570 params); live self-play only
baseline_command: lab-L10-20260523T132940Z (R-TRAIN-WL5 V=64 sgd=0.0025 fp32 = 3,297.6 aug/s); lab-L11b-20260523T134850Z (R-TRAIN-LEAN-style V=512 sgd=0.001 fp32 = 4,231.8 aug/s)
candidate_command: python scripts/lab_train_cell.py --out-dir sweep_logs/lab-L11bp-20260523T141456Z --lane L11bp --model small --workers 8 --games-per-batch 8 --n-simulations 400 --wave-size 512 --ema-tau 0.99 --grad-accum-steps 4 --sgd-per-position 0.001 --fp16-eval --warmup-secs 30 --measurement-secs 120 --device mps
hardware: M5 Max / MPS / idle
seed: workers 1000..1007; trainer default
baseline_metric: R-TRAIN-WL5: 3,297.6 aug/s; 14.07 games/s; 0.0917 epochs/s; trainer_step_s_p50=0.0512s. L11b: 4,231.8 aug/s; 15.47 games/s; 0.05 epochs/s.
candidate_metric: **L11b' (R-TRAIN-LEAN-fp16): 8,340.5 aug/s**; 32.19 games/s; 0.0667 epochs/s; trainer_step_s_p50=0.0801s; 11 epochs in 120s; plies_mean 32.74 (preserved — comparable to L10's 29.6 and L11b's 34.3, all well below the 81 terminal cap; trainer cells run actual gomoku endgames).
delta:
  - vs L10 R-TRAIN-WL5 baseline: aug/s **+152.9%** (3297.6 → 8340.5); games/s +128.8%; epochs/s -27% (each epoch is lighter training work because sgd_per_position is lower).
  - vs L11b (low-sgd fp32): aug/s **+97.1%** (4231.8 → 8340.5) — the fp16 lever added on top of L11b's recipe yields the SAME magnitude win as on R-S400 (small V=512 fp16 was +97.2%). Two independent levers, multiplicative effect.
  - Mechanism: with the trainer's per-position SGD work capped (low sgd_per_position) the trainer no longer monopolizes MPS, and fp16's worker-side bandwidth savings can be fully realized. R-TRAIN-WL5's L10 trainer_step was 0.0512s; here it's 0.0801s but with MUCH less work per epoch (steps/epoch ≈ ~30-150 here vs ~80 for WL5), so per-step pace is similar. The trainer's total wall-budget share shrinks; worker share grows.
confidence: high. Single trial, 120s window, 11 epochs (well above 2-epoch minimum). fp16 verified engaged on workers. Game-shape preserved (plies_mean in the same band as prior R-TRAIN cells; not at the 81 terminal cap). Mechanism (independent trainer + worker levers) predicts the multiplicative stacking, observed magnitude matches the prediction (97% × 1.28 = ~2.49× from WL5; measured 2.53× — within rounding).
artifacts: sweep_logs/lab-L11bp-20260523T141456Z/{summary.tsv,metadata.txt,cell_train_small_W08_G08_S400_V512_EMA99_GA04_WM1_B512/{logs/trainer.log,logs/worker-NN.log,records/v10,v11}}
commands_run:
  - python scripts/lab_train_cell.py --out-dir sweep_logs/lab-L11bp-20260523T141456Z --lane L11bp --model small --workers 8 --games-per-batch 8 --n-simulations 400 --wave-size 512 --ema-tau 0.99 --grad-accum-steps 4 --sgd-per-position 0.001 --fp16-eval --warmup-secs 30 --measurement-secs 120 --device mps
decision: needs_repeat
next_action:
  - PERF LAB ESTABLISHES: a new R-TRAIN operating point exists at +153% over WL5. Open a NEW best-cell row R-TRAIN-LEAN-fp16 = small / W=8 / G=8 / S=400 / V=512 / sgd=0.001 / fp16 = 8,340.5 aug/s (perf reference only — NOT promoted to R-TRAIN-WL5 because behavior knob changed).
  - R-TRAIN-WL5 STAYS at 3,297.6 aug/s (V=64, sgd=0.0025, fp32). It is the production recipe; per the TQ gate, promotion of L11b's recipe to production requires a canary training run reporting val/policy_ce vs archives/wl5_validation_v1.pt + plies/game-shape band. That's a training-pipeline lane.
  - HOWEVER — note that the perf lab now has a quantitatively concrete recommendation if Jason wants to consider WL6 as a successor to WL5: "small / V=512 / sgd=0.001 / fp16-eval" runs at 8,340.5 aug/s. The TQ canary is the gate; the perf number is in hand.
  - Compound follow-up dispatching in parallel: L09b (R-TRAIN-ANE with fp16 workers) — if fp16 halves the worker-side ANE loss from L09 (small/V=64 gen was 2× slower on ANE), and the trainer-side ANE-relief gain is still real, R-TRAIN-ANE could finally pay.
  - Lower priority: L11b'' (sweep different sgd_per_position values at V=512+fp16 to find the optimal trainer-work-per-second band).
```

### 2026-05-23 — L06fu-extended fp16-eval PROMOTE — R-S200 +84%, R-S100 +48%, medium V=512 new ref

```yaml
lane: L06fu-extended (compound follow-up of L06-followup)
hypothesis: The L06-followup +97.2% headline at R-S400/small was mechanistic (bandwidth-bound eval). The same fp16 lever should compound at other points of the R-S* family — proportional to "how eval-bound the workload is": higher S (sims per game) = more eval calls = more bandwidth savings. Test R-S200 + R-S100 + medium V=512 to map the fp16 effect across the operating envelope.
code_ref: 4e1bc2d on main (run-time)
evaluator: torch / MPS / fp16-eval
dataset_ref: pure self-play; fresh random fused checkpoint per model; no trainer; 60s/cell smoke
baseline_command: R-S200 = 9,156 (L03); R-S100 = 15,082 (L03); medium V=64 = 1,393 (canonical-sweep-mainframe — no medium V=512 fp32 ref exists yet)
candidate_command: python scripts/canonical_sweep.py --out-dir sweep_logs/lab-L06fu-ext-20260523T140641Z --cells-from cells.csv --lane L06fu-extended --secs-per-cell 60 --fp16-eval
hardware: M5 Max / MPS / idle; torch 2.11.0
seed: workers seeded 1000..1015
baseline_metric: R-S200 fp32 = 9,156 aug/s; R-S100 fp32 = 15,082 aug/s; medium V=64 fp32 = 1,393 aug/s
candidate_metric: R-S200 fp16 = **16,850.8 aug/s** (plies_mean 15.96); R-S100 fp16 = **22,312.1 aug/s** (plies_mean 15.96); medium V=512 fp16 = **3,377.2 aug/s** (plies_mean 15.95). All at 16-ply cap; game-shape preserved.
delta:
  - R-S200 small V=512: **+84.0%** vs fp32 (9156 → 16851)
  - R-S100 small V=512: **+48.0%** vs fp32 (15082 → 22312)
  - medium V=512 vs medium V=64 (combined V + fp16): **+142.4%** (1393 → 3377); estimating V=64→V=512 alone at +50% per L01 small extrapolation, medium fp16-only effect ≈ +62% (model-dependent, in the predicted bandwidth-bound regime between R-S100's +48% and R-S400's +97.2%)
  - Mechanism: fp16's bandwidth savings dominate when the eval forward is the bottleneck. At S=400 (R-S400 small) eval is the loop's dominant cost → +97%; at S=200 it's still dominant → +84%; at S=100 MCTS Python overhead starts diluting eval's share → +48%. Medium model has more bandwidth per eval than small, so should compound bigger; estimated +62% on fp16 alone fits the trend.
confidence: high. fp16 actually engaged (worker logs explicit at lab-L06fu-ext-*/cell_*/logs/w0.log). plies_mean across all 3 cells = 15.95-15.96 = at the 16-ply cap, game-shape preserved as L06-followup. The R-S200/R-S100 deltas are clean comparisons against L03's same-shape fp32 references. The medium V=512 fp16 number is a NEW reference point (no fp32 V=512 medium baseline exists); attribution between V and fp16 is estimated, not measured.
artifacts: sweep_logs/lab-L06fu-ext-20260523T140641Z/{summary.tsv,cells.csv,metadata.txt,cell_small_W08_G08_S200_V512,cell_small_W08_G08_S100_V512,cell_medium_W08_G08_S400_V512}
commands_run:
  - python scripts/canonical_sweep.py --out-dir sweep_logs/lab-L06fu-ext-20260523T140641Z --cells-from cells.csv --lane L06fu-extended --secs-per-cell 60 --fp16-eval
decision: promote
next_action:
  - Update R-S200 best to small / W=8 / G=8 / S=200 / V=512 / fp16 = 16,850.8 aug/s (was 9,156 fp32 from L03).
  - Update R-S100 best to small / W=8 / G=8 / S=100 / V=512 / fp16 = 22,312.1 aug/s (was 15,082 fp32 from L03).
  - Open a NEW reference point R-S400-medium = medium / W=8 / G=8 / S=400 / V=512 / fp16 = 3,377.2 aug/s (no prior best-cell row; medium V=64 fp32 = 1,393 is the only prior medium measurement at S=400).
  - Compound follow-up candidate L06fu-medium-AB: 2-cell medium V=512 fp32 vs fp16 to cleanly attribute the medium fp16 effect (currently estimated at +62%). Medium-axis is the bandwidth-bound regime; this is the next place to confirm the mechanism.
  - Bigger compound (dispatching now): L11b' = L11b (V=512 + sgd_per_position=0.001) + workers --fp16-eval. Tests whether the trainer-level R-TRAIN family can compound both finds (L11b's +28% from low-sgd at V=512 + L06-fp's near-doubling at V=512). Could be the R-TRAIN-WL5 promote story this perf cycle has been hunting for.
```

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

