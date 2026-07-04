# Training-Quality Promotion Gate

**LIVE** (2026-07-04) — the standing rule a perf/behavior receipt must satisfy
before it may claim `decision: promote`. This is doctrine, not history: 10+ wiki
pages cite it. Extracted verbatim from
[experiment-ledger.md](experiment-ledger.md) on 2026-07-04, unchanged, so the
live rule stops being trapped inside frozen campaign evidence. Receipts still
land in the ledger; the *rule* lives here.

## Training-Quality Promotion Gate

Perf changes that touch training behavior, inference outputs, MCTS/search behavior, replay/data encoding, checkpoint refresh cadence, or game-start distribution need more than throughput. A receipt may not use `decision: promote` unless it records all of the following:

1. **Named quality gate before the run.** Use at least one fixed external baseline or fixed validation archive. Current named options are:
   - external baselines: `heuristic`, `lookahead:depth=2`, and/or `lookahead:depth=4` via the match/eval harness with alternating colors;
   - validation archive: `archives/wl5_validation_v1.pt`, reporting at least `val/policy_ce`, `val/policy_kl`, and `val/value_mse` against the parent/reference checkpoint.
2. **Game-shape guardrail.** Report `selfplay/plies_mean` and, when available, `selfplay/plies_p90` or equivalent game-length distribution. Promotion is blocked or marked `needs_repeat` if the candidate shows sustained fast-attack collapse: falling plies, shorter-game buffer-fill concavity, or a material drop below the parent run's game-length band without an explicit strength explanation.
3. **Short-eval noise policy.** State game count and uncertainty. `n < 20` is smoke only and cannot support a strength claim. `n=20` can be a canary but normally needs a repeat or archive agreement for promotion. Prefer `n >= 50` or two independent same-shape `n >= 20` reads for behavior-changing promotion; otherwise use `decision: needs_repeat`.
4. **Reproducibility IDs.** Behavior-changing perf receipts must include checkpoint path(s), W&B run ID(s) or explicit `wandb: disabled`, commit hash, seed policy, and env/backend flags such as `GOMOKU_DISABLE_NATIVE_MCTS`, `GOMOKU_DISABLE_NATIVE_STATE_OPS`, `PYTORCH_ENABLE_MPS_FALLBACK`, device, model size, stem padding, sims, wave size, workers, and evaluator backend.
5. **Explicit decision.** Every receipt ends with `decision: promote | reject | blocked | needs_repeat`. Throughput-only wins that lack the selected quality gate, plies/game-shape read, or reproducibility IDs are not promotions; mark them `blocked` if the harness/artifact is missing or `needs_repeat` if the evidence is merely noisy/short.

## Where receipts go

The gate is the rule; the evidence it gates lands in
[experiment-ledger.md](experiment-ledger.md) (new perf receipts append there
under dated era headers). The complementary Standard Gates (`uv run pytest -q`,
CPU smoke) live in [test-ledger.md](test-ledger.md). Promotion updates
[best-cells.md](best-cells.md). Provenance of the gate itself: codified
2026-05-22 (`open-notes/20260522T054739Z-04-quality-promotion-gates.md`).
