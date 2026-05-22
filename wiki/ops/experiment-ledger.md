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
