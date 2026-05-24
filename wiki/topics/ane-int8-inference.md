# ANE INT8 Inference (Post-WL5 Task) — HISTORICAL SCOPING DOC

> **Status note (2026-05-23):** This is the original (WL5-era) scoping doc for the ANE work. Parts have shipped, parts have been superseded, parts are still relevant. **For the current state of ANE research in this project, read [coreml-design-envelope-and-our-fit.md](coreml-design-envelope-and-our-fit.md) — that page is now the canonical entry point** with the measured envelope, the live research lanes, and the inbound-research landing zone. This page is preserved as the historical record of how the ANE strand started.
>
> What's shipped from this doc's plan: `gomoku/coreml_evaluator.py`, `scripts/aggressive_engine_scout.py`, `selfplay_worker --evaluator coreml --coreml-compute-units …` flags, `scripts/coreml_ane_residency_scout.py` (companion). What's superseded: the "When to do this — not during WL5" gating (we are past WL5). What hasn't shipped: INT8 quantization specifically (FP16 path is what landed and what we measured against in L09c/L09d/L09e/etc.).

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

## First aggressive scout (2026-05-22)

Implemented `gomoku/coreml_evaluator.py` plus
`scripts/aggressive_engine_scout.py` and ran one bounded scout on the
small model (`stem_padding=1`, fused eval, batches 8/32/64/128). Output
receipt:
`sweep_logs/aggressive-engine-scout-2026-05-22.json` (ignored artifact).

Command shape:

```bash
python scripts/aggressive_engine_scout.py \
  --size small --stem-padding 1 \
  --batches 8,32,64,128 --warmup 2 --repeats 5 \
  --coreml-precisions fp16,int8 \
  --coreml-compute-units CPU_ONLY,CPU_AND_NE \
  --train-batch-size 256 --train-warmup 6 \
  --train-prewarm-steps 8 --train-steps 16 \
  --overlap-batch 64 --pressure-warmup 2 --pressure-sleep-ms 0 \
  --overlap-lanes torch_mps,coreml_fp16_CPU_AND_NE,coreml_fp16_CPU_ONLY,coreml_int8_CPU_AND_NE,coreml_int8_CPU_ONLY \
  --output sweep_logs/aggressive-engine-scout-2026-05-22.json
```

Raw eval latency at batch 128:

| lane | median | positions/sec |
|---|---:|---:|
| PyTorch MPS | 2.94 ms | 43.5k |
| Core ML FP16 CPU_AND_NE | 9.05 ms | 14.1k |
| Core ML FP16 CPU_ONLY | 8.47 ms | 15.1k |
| Core ML INT8 CPU_AND_NE | 9.04 ms | 14.2k |
| Core ML INT8 CPU_ONLY | 9.05 ms | 14.1k |
| PyTorch CPU | 56.78 ms | 2.3k |

Overlap probe (MPS trainer batch 256, median step baseline 13.94 ms):

| pressure lane | train slowdown | pressure eval positions/sec |
|---|---:|---:|
| PyTorch MPS eval process | 2.65x | 8.7k |
| Core ML FP16 CPU_AND_NE | 1.32x | 2.4k |
| Core ML FP16 CPU_ONLY | 1.14x | 2.2k |
| Core ML INT8 CPU_AND_NE | 1.19x | 3.1k |
| Core ML INT8 CPU_ONLY | 1.13x | 2.3k |

Interpretation:

- Core ML is **not** a raw per-call speed win yet. Fused PyTorch/MPS is
  still much faster for eval batches on this tiny model.
- The three-engine idea survives because the contention story is very
  different: a competing PyTorch/MPS eval process made trainer steps
  ~2.65x slower, while Core ML pressure lanes were ~1.13-1.32x.
- INT8 weight quantization worked mechanically, but did not improve raw
  latency in this first scout. It may still matter if a different Core ML
  conversion path actually lands on ANE; this run should not be read as
  proof that ANE is exhausted.
- Same-process threaded MPS eval pressure crashed Metal during smoke
  (`commit an already committed command buffer`), so the scout now uses
  spawned pressure processes, matching the production worker shape.

Next move: make the Core ML lane production-shaped enough to measure
generation throughput, not just naked eval calls. If CPU/ANE eval is slower
per leaf but preserves trainer time, it may still win end-to-end when the
trainer and self-play overlap.

## ANE rail proof and correction (2026-05-21/22)

Follow-up metering corrected an important interpretation risk:
`CPU_AND_NE` is a request to Core ML, not proof that the Apple Neural
Engine actually ran the model.

- Apple Vision person segmentation is the current known-good positive
  control: `/tmp/vision-ane-powermetrics-1779421070.txt` parses to 25
  ANE samples, mean 4474 mW, max 4488 mW, all active.
- Detached 934b work produced a dedicated rail-proof lab page and scout
  harness; Lane 03 integrated it as
  `scripts/coreml_ane_residency_scout.py` plus
  `wiki/topics/coreml-ane-residency-lab.md`.
- The 2026-05-22 Lane 03 attempt to run a fresh powermetrics-required
  `conv,resnet,gomoku` scout was blocked because `sudo -n true` failed
  (`cached/passwordless sudo is unavailable`). Exports succeeded, but no
  same-window raw rail logs were produced, so no claim can exceed
  `coreml-scheduled`.

Working correction:

- No Gomoku/Core ML path should be called ANE-backed without nonzero ANE
  rail evidence from the same pressure window.
- Use the dedicated residency lab page for caps and receipts:
  [coreml-ane-residency-lab.md](coreml-ane-residency-lab.md).
- The next measurement is not more `CPU_AND_NE` label checking; it is a
  fresh Vision positive control plus a powermetrics-required
  conv/resnet/gomoku shape scout when sudo is available.

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

## When to do this (HISTORICAL — see status note at top)

Original framing (2026-05-21): Not during WL5. Mid-run backend changes
invalidate comparisons. The natural window was after WL5 reports out
and we're in design mode for WL6 — quantize the WL5 final checkpoint,
validate, and use INT8 for WL6's self-play if it passes the elo gate.

**Update 2026-05-23:** WL5 is closed. The Core ML evaluator and ANE
scout did ship between WL5 close and 2026-05-23 (see "What shipped"
section below). INT8 specifically did not ship — we measured against
the FP16 Core ML path (which is Core ML's default `compute_precision`)
and the L09 family of receipts at FP16 is what currently maps the
envelope. INT8 remains a future-research lane if the L09e' residency
proof confirms ANE residency and the FP16-vs-INT8 throughput delta is
worth the quantization-calibration cost.

## What shipped from this doc's plan (2026-05-23 reality check)

The original implementation plan below has been partly executed. Status
of each plan item:

| Plan item | Status | Where it lives |
|---|---|---|
| Boundary scouting microbench | **SHIPPED** | `scripts/aggressive_engine_scout.py`; receipt at `sweep_logs/aggressive-engine-scout-2026-05-22.json` |
| `scripts/export_coreml_int8.py` (INT8 conversion script) | **NOT SHIPPED** | FP16 path landed via `gomoku/coreml_evaluator.py` instead; INT8 specifically not pursued |
| `gomoku/coreml_evaluator.py` (Core ML evaluator) | **SHIPPED** | `gomoku/coreml_evaluator.py:285+` (FP16 path; INT8 not implemented) |
| `selfplay_worker` `--evaluator-backend` flag | **SHIPPED** (named `--evaluator coreml`) | `gomoku/selfplay_worker.py`; flag landed 2026-05-23 alongside L09 dispatch |
| `scripts/validate_int8_elo.py` (calibration + elo validation) | **NOT SHIPPED** | INT8-specific; depends on the un-shipped INT8 conversion path |
| Wiki update with elo delta + throughput numbers | **SHIPPED (different shape)** | Receipts in [experiment-ledger.md](../ops/experiment-ledger.md) L09 / L09c / L09c-V512 / L09d / L09e; FP16 numbers not INT8 |
| Three-engine smoke | **PARTIALLY SHIPPED** | Engine-isolation measured by perf-lab L09 family; residency proof (the third leg) still blocked on sudo for powermetrics — see [coreml-ane-residency-lab.md](coreml-ane-residency-lab.md) |
| ANE residency rail scout (`coreml_ane_residency_scout.py`) | **SHIPPED** (separate from this plan) | `scripts/coreml_ane_residency_scout.py`; companion page [coreml-ane-residency-lab.md](coreml-ane-residency-lab.md); 2026-05-22 lane 03 blocked on sudo, no rail evidence yet |

The INT8 path remains a future-research lane: it would only become load-bearing if (a) L09e' confirms L09c's win is actually ANE-resident, (b) the inbound new ANE research suggests INT8 would shift the envelope materially, or (c) deployment-time inference cost becomes a constraint.

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
