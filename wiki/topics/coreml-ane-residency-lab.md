# Core ML / ANE Residency Lab

Created 2026-05-22 as the control-plane notebook for the autonomous
Core ML / ANE residency lab. This page owns the claim discipline and
sweep shape only. Raw receipts stay in `/tmp`, `sweep_logs/`, or the
script output paths; durable interpretation can be appended here.

**Companion pages:**
- [coreml-design-envelope-and-our-fit.md](coreml-design-envelope-and-our-fit.md) — the design-context page: what Core ML is built for (the iOS/macOS app ML stack: Vision, Siri, AR, FaceID), where our research-compute workload sits relative to that envelope, where ANE could still pay (deployment; concurrent compute stream; medium-and-larger models), and concrete L09c-L09h research lanes for mapping the envelope's edges. Read this first if you're new to the ANE story in this project.
- [ane-moonshots-and-oss-frontier.md](ane-moonshots-and-oss-frontier.md) — the OSS-frontier route map: what llama.cpp/Ollama/MLX/ANEMLL/private-API projects imply for this lab, and the next moonshot lanes for making ANE useful to Gomoku rather than merely requested by `CPU_AND_NE`.
- [m5-max-fp16-and-throughput-regimes.md](m5-max-fp16-and-throughput-regimes.md) — chip-level findings; Finding 2 (bandwidth-bound vs dispatch-bound regimes) is the MPS analog of what L09g would measure for Core ML.

## Goal

Find the Core ML model shapes, compute-unit settings, batch geometry,
and worker pressure patterns that actually make the Apple Neural Engine
resident for Gomoku-like inference on Jason's M5 Max.

The practical target is still the three-engine training loop:

| Lane | Desired engine | Why |
|---|---|---|
| Self-play leaf eval | Core ML on ANE | Keep eval-only worker load off PyTorch/MPS. |
| Trainer forward/backward | PyTorch on MPS GPU | Preserve the supported gradient path. |
| Eval sidecar / probes | CPU-only Core ML or BNNS | Keep scheduled probes from stealing GPU or ANE budget. |

This lab is not trying to prove that Core ML is faster in a naked
microbench. It is trying to find the production-shaped point where
engine isolation beats same-engine contention.

2026-05-23 moonshot correction: the lab's larger goal is not merely to
ask Core ML for `CPU_AND_NE`. It is to max out this M5 Max as a
whole-machine Gomoku proving ground. The OSS frontier says the ANE is
most plausible as a static dense sidecar/prefill-like engine, with CPU
and GPU handling dynamic control flow and training. See
[ane-moonshots-and-oss-frontier.md](ane-moonshots-and-oss-frontier.md).

## Definition Of Cap

The cap is the maximum claim a receipt is allowed to support.

| Cap | Allowed claim | Required evidence |
|---|---|---|
| `coreml-scheduled` | The model ran through Core ML with the requested compute units. | JSON receipt with model plan, compute units, worker count, batch size, and successful predictions. |
| `coreml-isolated` | The Core ML lane causes less MPS trainer slowdown than a competing PyTorch/MPS eval lane. | Overlap receipt with trainer baseline, pressure lane throughput, and trainer slowdown ratio. |
| `ane-metered` | The workload moved the ANE rail on this Mac. | Same-window `powermetrics` with `ane_power` samples above idle while the Core ML pressure phase is running. |
| `ane-resident-candidate` | A model shape is worth production self-play wiring. | `ane-metered` plus stable throughput across workers, no worker errors, and GPU not carrying the primary eval load. |
| `production-ready` | The lane can enter self-play experiments. | Production-shaped self-play throughput, trainer-overlap measurement, and checkpoint/accuracy validation. |

Do not call a path "ANE-backed" from `CPU_AND_NE` alone. `CPU_AND_NE`
is a request to Core ML, not proof that the model landed on the ANE.

## Evidence Standards

- Every cell needs a JSON receipt and, when the claim exceeds
  `coreml-scheduled`, a raw `powermetrics` receipt from the same window.
- Include host facts: machine, macOS, Python, PyTorch, coremltools, and
  MPS availability.
- Include model facts: kind, input shape, depth/blocks/filters/hidden,
  output width, parameter count, precision, Core ML package path when
  retained, conversion time, and max batch range.
- Include pressure facts: compute units, worker count, batch size,
  warmup, duration, max iterations, ready workers, exit codes,
  positions/sec, total positions, load time, first-predict time, and
  worker errors.
- Include rail facts: sampler command, sample count, min/mean/max ANE
  mW, nonzero sample count, CPU/GPU rails when available, and whether a
  known-good ANE workload was sampled nearby as a meter sanity check.
- Score success by rail-backed residency and production overlap, not by
  Activity Monitor percentages or requested compute-unit labels.

## Current Known Facts

- `wiki/topics/ane-int8-inference.md` records the first aggressive
  Core ML scout: raw Core ML eval was slower than fused PyTorch/MPS on
  the tiny Gomoku model, but Core ML pressure hurt MPS trainer steps far
  less than a competing PyTorch/MPS eval process.
- The same page corrected the central evidence risk: the Gomoku Core ML
  FP16 scout with `CPU_AND_NE` processed 6,603,776 random positions at
  roughly 88k positions/sec across four workers, but showed no active
  ANE rail rows in `powermetrics`; a follow-up all-sampler check showed
  `ANE Power: 0 mW`.
- Apple Vision person segmentation is the local positive control:
  `/tmp/vision-ane-powermetrics-1779421070.txt` shows 25 ANE samples
  between 4461 and 4488 mW. Use this to prove the meter can see ANE
  load before interpreting Gomoku receipts.
- `/tmp/gomoku-ane-live-powermetrics.log` is a long mixed/live log, not a
  scoped same-window receipt. Re-parsing it on 2026-05-22 found nonzero
  ANE samples later in the file, so do **not** use it as a quantified
  Gomoku-negative control without matching it to the exact pressure
  interval.
- `/tmp/gomoku-ane-powermetrics-1779420622.txt` is a weak/partial
  negative: the `ane_power` sampler emitted no parseable `ANE Power`
  rows. Treat it as "no ANE evidence," not as a quantified 0 mW run.
- `/tmp/coreml-ane-residency-smoke.json` proves the new scout harness
  can export and run a tiny MLP through `CPU_ONLY` and `CPU_AND_NE`, but
  it used `--powermetrics never`, `--duration-s 0.1`, and `--max-iters
  1`; cap it at `coreml-scheduled`.
- `scripts/coreml_ane_residency_scout.py` is now integrated in this
  worktree. It supports toy `mlp`, `conv`, `resnet`, and `gomoku` model
  kinds, `CPU_ONLY` / `CPU_AND_NE`, FLOAT16/FLOAT32 conversion,
  multiprocessing pressure, optional retained models, JSON receipts, and
  optional/cached-sudo `powermetrics`.
- `wiki/topics/m5-max-as-mainframe.md` is the philosophy page: this lab
  is part of learning the exact M5 Max contour, with rail evidence above
  generic assumptions.
- A fresh 2026-05-23 web pass found the same pattern across OSS: mainline
  LLM runners use Metal/MLX/GPU for general decode, while ANE-specific
  projects succeed by shaping static dense graph islands, prefill-sized
  matmuls, Core ML packages, IOSurface buffers, and/or private APIs.
  Durable source record:
  [../sources/ane-oss-frontier-2026-05-23.md](../sources/ane-oss-frontier-2026-05-23.md).

## Sweep Matrix

Keep cells short enough for autonomous iteration. Start with 10-30s
pressure windows, then promote only promising cells to longer repeats.

| Axis | First values | Promotion values | Notes |
|---|---|---|---|
| model kind | `mlp`, `conv`, `resnet`, `gomoku` | best 1-2 shapes | Toy shapes answer residency; Gomoku answers deployment fit. |
| compute units | `CPU_ONLY`, `CPU_AND_NE` | `CPU_AND_NE` only after positive rail | `CPU_ONLY` is the negative scheduling control. |
| precision | `FLOAT16` | `FLOAT32` if needed | Script does not yet do INT8; INT8 belongs after residency is understood. |
| batch size | 1, 8, 32, 128, 512 | local best +/- neighbors | ANE may require enough work per predict to light up. |
| workers | 1, 2, 4 | 8 if stable | Watch load/compile overhead and rail saturation. |
| conv filters | 16, 32, 64 | best +/- one step | Larger shapes may be easier for ANE placement than tiny Gomoku. |
| residual blocks | 1, 2, 4 | best +/- one step | Residency can be shape/op dependent. |
| image shape | 16x16, 32x32, 64x64 | best shape with Gomoku-like channels | Toy conv/resnet can search supported ANE layouts. |
| powermetrics | `required` for claim cells | same | `auto` is fine for smoke, not for `ane-metered`. |

Suggested bounded command shape:

```bash
python scripts/coreml_ane_residency_scout.py \
  --model-kinds conv,resnet,gomoku \
  --compute-units CPU_ONLY,CPU_AND_NE \
  --compute-precision FLOAT16 \
  --batch-size 128 \
  --workers 4 \
  --duration-s 20 \
  --warmup 2 \
  --powermetrics required \
  --output /tmp/coreml-ane-residency-YYYYMMDD-HHMMSS-label.json \
  --raw-dir /tmp/coreml-ane-residency-YYYYMMDD-HHMMSS-label-raw
```

## Receipt Naming Convention

Use UTC-ish sortable local timestamps and keep raw rail files adjacent to
the JSON summary.

| Artifact | Pattern |
|---|---|
| JSON summary | `/tmp/coreml-ane-residency-YYYYMMDD-HHMMSS-<label>.json` |
| raw powermetrics dir | `/tmp/coreml-ane-residency-YYYYMMDD-HHMMSS-<label>-raw/` |
| phase rail log | `/tmp/coreml-ane-residency-YYYYMMDD-HHMMSS-<label>-raw/<model>-<units>.powermetrics.txt` |
| retained models | `sweep_logs/coreml_ane_models/<label>/` |
| promoted durable copy | `sweep_logs/coreml-ane-residency-YYYYMMDD-HHMMSS-<label>.json` |

Labels should encode the hypothesis, not the result:
`conv32-b128-w4`, `resnet64-b512-w4`, `gomoku-b128-w4`,
`vision-positive-control`.

## Append Log

### 2026-05-22 Control Plane Created

The lab begins from 2026-05-21 negative Gomoku rail results and a
positive Vision rail control. Current interpretation: the existing
Gomoku Core ML export is useful evidence for Core ML isolation, but not
yet evidence for ANE residency. The next useful work is shape scouting
with rail-required receipts.

### 2026-05-22 Lane 03 integration + sudo blocker

Integrated the detached 934b harness into the frontier lane worktree:
`./scripts/coreml_ane_residency_scout.py`,
`./tests/test_coreml_ane_residency_scout.py`, and this page. Validation:
`python -m py_compile scripts/coreml_ane_residency_scout.py` and
`pytest -q tests/test_coreml_ane_residency_scout.py` passed.

Artifacts from this lane:

| Artifact | Cap | Read |
|---|---|---|
| `sweep_logs/coreml-ane-residency-20260522-lane03/dry-run.json` | smoke only | Parsed `conv,resnet,gomoku` plans without Core ML/powermetrics execution. |
| `sweep_logs/coreml-ane-residency-20260522-lane03/coreml-scheduled-smoke.json` | `coreml-scheduled` | `conv,resnet,gomoku` exported and ran under `CPU_ONLY` and `CPU_AND_NE` with 1 worker, batch 16, max 5 predicts; no worker errors; powermetrics skipped by design. Tiny-run throughput was ~39-51k positions/sec, but this is not a residency claim. |
| `sweep_logs/coreml-ane-residency-20260522-lane03/blocked-powermetrics-required.json` | blocked | Exports succeeded, but every `CPU_ONLY` / `CPU_AND_NE` phase reports `sudo -n true failed; cached/passwordless sudo is unavailable`; no same-window raw rail logs were produced. |

Nearby meter evidence: re-parsing
`/tmp/vision-ane-powermetrics-1779421070.txt` with the scout parser gives
25 ANE samples, mean 4474.36 mW, max 4488 mW, all active. That proves the
meter can see ANE load on this Mac, but it is not a new same-window
positive control for this lane because cached/passwordless sudo was not
available at run time.

Decision: the harness can make `coreml-scheduled` receipts, but the lane
is **blocked** from `ane-metered` conclusions until a session has cached
or passwordless sudo for `powermetrics` and runs the conv/resnet/gomoku
scout plus a fresh Vision positive control in adjacent windows. No
Gomoku/Core ML path should be called ANE-backed from this lane.

### 2026-05-23 OSS frontier moonshot pass

Fresh web research changed the lab posture from "try the current Core ML
worker harder" to "shape the workload like the successful ANE projects."
Mainstream runners still live on Metal/MLX/GPU because generic LLM decode
needs dynamic kernels and memory-bandwidth-heavy KV-cache traffic. ANE
successes cluster around static dense graph islands, prefill-sized
matmuls, app-deployment Core ML models, profiler-guided fallback removal,
IOSurface buffer discipline, long-lived compiled packages, and private
API research runtimes.

Durable artifacts:

| Artifact | Role |
|---|---|
| [../sources/ane-oss-frontier-2026-05-23.md](../sources/ane-oss-frontier-2026-05-23.md) | Source record for the web pass. |
| [ane-moonshots-and-oss-frontier.md](ane-moonshots-and-oss-frontier.md) | Lab route map and lane cards. |
| `scripts/ane_vision_furnace.swift` | Local Vision positive-control helper for rail-meter sanity checks; not Gomoku evidence. |

Decision: keep the evidence cap discipline, but widen the research target.
The next useful lane is not another blind `CPU_AND_NE` run. It is a
profiler-backed, rail-backed attempt to make Gomoku leaf eval look like a
static dense ANE sidecar while CPU/native MCTS and MPS/MLX training keep
the dynamic parts.

## Next Actions

1. Re-run with cached/passwordless sudo available (`sudo -n true` must
   pass). Start with a fresh Vision positive control and a fresh
   `conv,resnet,gomoku` scout in adjacent windows.
2. Read [ane-moonshots-and-oss-frontier.md](ane-moonshots-and-oss-frontier.md)
   before designing the next ANE lane. Prefer profiler-guided shape changes
   over repeating the current exported graph.
3. Use `--powermetrics required` for any claim cell; do not promote any
   claim above `coreml-scheduled` if sudo/powermetrics is unavailable.
4. Summarize each cell by cap, ANE mean/max mW, positions/sec, ready
   workers, and errors. Append only the interpretation here; leave raw
   receipts where they are.
5. If any toy conv/resnet shape reaches `ane-metered`, narrow around its
   shape before trying to make Gomoku match it.
6. If Gomoku reaches `ane-metered`, run a production-shaped overlap probe
   before wiring self-play. The win condition is lower trainer slowdown
   plus acceptable self-play throughput, not raw Core ML latency.
