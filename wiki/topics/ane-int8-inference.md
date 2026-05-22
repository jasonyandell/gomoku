# ANE INT8 Inference (Post-WL5 Task)

Scoping doc for porting self-play + eval inference to Apple Neural Engine
(ANE) at INT8 precision. Captured during WL5 monitoring (2026-05-21).

## Why this is worth doing

- ANE is optimized for INT8 — typically 2-3× higher TOPS at INT8 vs FP16.
- KataGo precedent: production deployments run INT8 with calibration and
  report essentially no elo loss for board games.
- Our `small` model is 316k params; conversion cost is small.
- Self-play and eval are the *training-era* inference budget; faster
  inference there compresses how long each WL run takes (~16h → ~10h
  estimated for WL5-class runs).
- Pairs naturally with the WL5 validation archive (1400 curated positions)
  — that archive is exactly the calibration set we need.

## What it does NOT speed up

- Trainer (FP32 stays — needs gradients).
- Lookahead4 opponent in mining (CPU-bound alpha-beta).
- Tree ops (native C MCTS); only the leaf-eval batch.

## Realistic ceiling

- Self-play cycle ~50-60% faster end-to-end if it lands cleanly.
- Memory: 4× smaller weights (doesn't matter; already tiny).
- Power: lower watts per inference (matters for sustained 8-worker runs).

## Stack

- **ANE access requires Core ML.** PyTorch's own INT8 path (`torch.ao`)
  targets CPU, not ANE. `coremltools` is the only realistic path.
- We already have `scripts/export_onnx.py` for the live SPA — Core ML
  conversion goes through the same ONNX intermediate or directly from
  PyTorch.

## Implementation plan

1. **Conversion script** `scripts/export_coreml_int8.py`:
   - Inputs: `--checkpoint PATH`, `--calibration-archive PATH`
     (default `archives/wl5_validation_v1.pt`), `--output PATH`.
   - Load checkpoint → trace or script the model.
   - `coremltools.convert(..., compute_precision=ct.precision.FLOAT16)`
     first (sanity check FP16 path works); then re-convert with
     `compute_precision=ct.precision.INT8` and `--quantize-weights`
     passing the calibration tensors.
   - Output: `models/<run-name>.mlpackage`.

2. **Core ML evaluator** `gomoku/coreml_evaluator.py`:
   - Match `Evaluator` API: `evaluate(states) → (priors, values)` and
     `evaluate_planes(planes_arr) → (priors, values)`.
   - Construct via `make_coreml_evaluator(mlpackage_path)`.
   - Internally: `coremltools.models.MLModel(...)`; predict on numpy
     arrays; reshape outputs.
   - Honor `compute_units=ct.ComputeUnit.CPU_AND_NE` so ANE is used
     when supported.

3. **Worker wiring** `gomoku/selfplay_worker.py`:
   - New flag `--evaluator-backend pytorch|coreml-int8`
     (default `pytorch`).
   - When `coreml-int8` is selected: instead of `make_torch_evaluator`,
     call `make_coreml_evaluator(args.coreml_model_path)`.
   - Cell wiring follows: add cell fields, dispatch via `worker_cmd`.

4. **Calibration + validation** `scripts/validate_int8_elo.py`:
   - Load FP32 model and INT8 model side by side.
   - Play `N=200` games of FP32_model vs INT8_model via `gomoku.match`.
   - Report elo delta + per-color split. Abort criteria: |elo delta| > 30
     points → calibration regression, do not deploy.
   - Also score both models against the WL5 validation archive (val/policy_ce
     and val/value_mse should be within 5%).

5. **Wiki update**: capture elo delta + throughput numbers in
   [activity-monitor-perf-runbook.md](activity-monitor-perf-runbook.md) and
   in [mcts-perf-ceiling.md](mcts-perf-ceiling.md) (ANE INT8 path joins
   the ceiling table).

## Calibration data

The WL5 validation archive (`archives/wl5_validation_v1.pt`) is the
calibration set:

- 1400 positions across 7 provenance buckets
- Spans canonical-opening, mid-game, long-defense, and high-disagreement
  positions
- Matches the distribution the deployed model will actually see
- Same source-of-truth as the trainer's diagnostic scoring, so
  calibration drift is observable in the existing wandb metrics

## Risk: subtle elo regression

The value head is the sensitive part — small value errors cascade
across MCTS subtrees. Calibration must cover the value head's typical
input distribution. If we see > 30 point elo delta in validation:

- Try INT8 weights + FP16 activations (hybrid).
- Try FP16 throughout (smaller win but lower risk).
- Try selective quantization (skip value head, quantize policy + resnet
  trunk only).

## Cost estimate

- Conversion + evaluator integration: 1 day
- Calibration + elo validation script: 0.5 day
- Production rollout (cell wiring, smoke, real run comparison): 0.5 day
- **Total: ~2 days** if things go cleanly; 3-4 days with calibration
  iteration.

## When to do this

Not during WL5. Mid-run backend changes invalidate comparisons.

The natural window: after WL5 reports out and we're in design mode for
WL6. Quantize the WL5 final checkpoint, validate, and use INT8 for
WL6's self-play if it passes the elo gate.

## Open questions

- Do we want INT8 weights only or INT8 activations too? Activations
  give more speedup but more accuracy risk.
- Should the trainer also load a Core ML INT8 model to score the
  validation archive on the same backend the workers use? (Avoids
  backend-dependent diagnostic drift.)
- Is there value in keeping both backends switchable per worker for
  ablation (e.g., 4 workers on PyTorch+MPS, 4 on Core ML+ANE, compare
  game quality)?

## Cross-refs

- [mcts-perf-ceiling.md](mcts-perf-ceiling.md) — current backend
  throughput ceiling table; INT8 path is the next addition.
- [activity-monitor-perf-runbook.md](activity-monitor-perf-runbook.md)
  — practical perf experiment knobs.
- [wl5-diagnostics-archive-start-design.md](wl5-diagnostics-archive-start-design.md)
  — where the calibration archive came from.
- `scripts/export_onnx.py` — existing PyTorch → ONNX path for the live
  SPA; calibration data path will look similar.
- KataGo INT8 docs: https://github.com/lightvector/KataGo/blob/master/docs/CompilingKataGo.md
  (search "TensorRT" / "FP16" sections — same calibration principles
  apply for Core ML).
