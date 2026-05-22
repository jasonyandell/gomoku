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

## Promotion Gate Addendum

Perf changes that touch training behavior, inference outputs, MCTS/search behavior, replay encoding, or checkpoint refresh cadence need more than throughput:

- fixed external baseline or validation-archive metric named before promotion,
- `plies_mean` / fast-attack-collapse check,
- short-eval noise caveat with game count or repeat policy,
- checkpoint/run IDs when strength claims are made,
- explicit `promote`, `reject`, `blocked`, or `needs_repeat` decision.

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
