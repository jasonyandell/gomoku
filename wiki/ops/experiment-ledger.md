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
