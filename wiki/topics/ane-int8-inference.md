# ANE INT8 Inference (Post-WL5 Task)

Scoping doc for moving eval-only inference off the MPS training path,
starting with self-play on Apple Neural Engine (ANE) at INT8 precision
and extending to a three-engine pipeline. Captured during WL5 monitoring
(2026-05-21).

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
- Bigger architectural win: self-play on ANE, trainer on MPS GPU, and
  eval sidecar on CPU/Accelerate should reduce accelerator contention.
  The value is engine isolation more than weight-size reduction.

## What it does NOT speed up

- Trainer (FP32 stays — needs gradients).
- Lookahead4 opponent in mining (CPU-bound alpha-beta).
- Tree ops (native C MCTS); only the leaf-eval batch.

## Realistic ceiling

- Self-play cycle ~50-60% faster end-to-end if it lands cleanly.
- Memory: 4× smaller weights (doesn't matter; already tiny).
- Power: lower watts per inference (matters for sustained 8-worker runs).
- Pipeline ceiling: ANE self-play + MPS training + CPU/BNNS eval can
  overlap work that currently competes for the same MPS path. This is
  the "unified-memory Mac" bet worth testing.

## Stack

- **ANE access requires Core ML.** PyTorch's own INT8 path (`torch.ao`)
  targets CPU, not ANE. `coremltools` is the only realistic path.
- We already have `scripts/export_onnx.py` for the live SPA — Core ML
  conversion goes through the same ONNX intermediate or directly from
  PyTorch.
- Use `CPU_AND_NE`, not `ALL`, for self-play if the goal is to keep
  the GPU free for PyTorch training. `ALL` may let Core ML borrow the
  GPU and reintroduce the contention this page is trying to remove.
- CPU eval should use the public CPU inference stack: Core ML CPU-only
  or Accelerate/BNNS. Treat "AMX" as an implementation detail Apple may
  use under that stack, not as a stable direct API target.

## Three-engine pipeline target

Jason's proposed shape:

| workload | preferred engine | reason |
|---|---|---|
| Self-play leaf evaluation | ANE via Core ML INT8/FP16 | Removes the dominant eval-only worker load from MPS. |
| Trainer forward/backward | MPS GPU via PyTorch | Keeps gradients on the best-supported training backend. |
| Eval sidecar / match probes | CPU via Core ML CPU-only or BNNS | Keeps scheduled evaluations from stealing GPU or ANE budget. |
| Native MCTS tree work | CPU C extension | Already moved most per-node churn out of Python. |

The important correction: unified memory does **not** guarantee that one
PyTorch tensor object moves zero-copy through PyTorch MPS, Core ML, and
BNNS. Those runtimes have different tensor wrappers, layout expectations,
and synchronization rules. The real win to chase is keeping the three
heavy lanes from contending for the same accelerator while accepting some
boundary copies if they are small relative to leaf-eval throughput.

This pipeline is especially plausible here because the model is tiny and
the current bottleneck is dispatch/graph overhead, not parameter memory.
It is also easy to stage: a Core ML evaluator only needs to satisfy the
existing `Evaluator.evaluate_planes` boundary.

## Implementation plan

0. **Boundary scouting microbench** before any launch wiring:
   - Convert one fixed checkpoint to Core ML FP16 and INT8.
   - Measure batch latencies at 8/32/64/128/256 planes for:
     PyTorch+MPS, Core ML `CPU_AND_NE`, Core ML `CPU_ONLY`.
   - Measure model conversion, compile, load, and first-predict time.
   - Run the same test while a trainer step loop is active on MPS to
     prove whether ANE self-play actually overlaps with GPU training.
   - Record Activity Monitor/`powermetrics` observations, but score by
     wall seconds and positions/sec, not percent utilization.

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
   - Also support `CPU_ONLY` for the eval sidecar lane. This is the
     cleanest public approximation of the "AMX/CPU" idea.

3. **Worker wiring** `gomoku/selfplay_worker.py`:
   - New flag `--evaluator-backend pytorch|coreml-int8`
     (default `pytorch`).
   - When `coreml-int8` is selected: instead of `make_torch_evaluator`,
     call `make_coreml_evaluator(args.coreml_model_path)`.
   - Cell wiring follows: add cell fields, dispatch via `worker_cmd`.
   - Treat checkpoint freshness as a first-class metric. If Core ML
     conversion/compile is slow, self-play may need to run one checkpoint
     behind the trainer or refresh less often than the PyTorch path.

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

6. **Three-engine smoke**:
   - Trainer loops on MPS with normal PyTorch.
   - One self-play worker uses Core ML `CPU_AND_NE`.
   - Eval sidecar uses Core ML `CPU_ONLY` or BNNS.
   - Compare cycle time, self-play positions/sec, eval wall time, and
     training step time against the current all-PyTorch/MPS baseline.

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

- Boundary scouting microbench: 0.5 day
- Conversion + evaluator integration: 1 day
- Calibration + elo validation script: 0.5 day
- Production rollout (cell wiring, smoke, real run comparison): 0.5 day
- Three-engine smoke / overlap proof: 0.5 day
- **Total: ~3 days** if things go cleanly; 4-5 days with calibration
  or checkpoint-refresh iteration.

## When to do this

Not during WL5. Mid-run backend changes invalidate comparisons.

The natural window: after WL5 reports out and we're in design mode for
WL6. Quantize the WL5 final checkpoint, validate, and use INT8 for
WL6's self-play if it passes the elo gate.

## Open questions

- Does Core ML conversion/compile fit inside the checkpoint cadence, or
  do ANE workers need to run one checkpoint behind?
- Do we want INT8 weights only or INT8 activations too? Activations
  give more speedup but more accuracy risk.
- Should the trainer score the validation archive with PyTorch/MPS for
  consistency, while the eval sidecar uses CPU-only Core ML for isolation?
- Is there value in keeping both backends switchable per worker for
  ablation (e.g., 4 workers on PyTorch+MPS, 4 on Core ML+ANE, compare
  game quality)?
- Can the CPU eval lane keep up without stealing too many performance
  cores from native MCTS workers?

## Cross-refs

- [mcts-perf-ceiling.md](mcts-perf-ceiling.md) — current backend
  throughput ceiling table; INT8 path is the next addition.
- [activity-monitor-perf-runbook.md](activity-monitor-perf-runbook.md)
  — practical perf experiment knobs.
- [wl5-diagnostics-archive-start-design.md](wl5-diagnostics-archive-start-design.md)
  — where the calibration archive came from.
- `scripts/export_onnx.py` — existing PyTorch → ONNX path for the live
  SPA; calibration data path will look similar.
- Apple Core ML / ANE principle: optimize model layout to reduce
  inter-engine transfers; engine partitioning is valuable because it
  opens CPU/GPU for other work while ANE handles inference.
- KataGo INT8 docs: https://github.com/lightvector/KataGo/blob/master/docs/CompilingKataGo.md
  (search "TensorRT" / "FP16" sections — same calibration principles
  apply for Core ML).
