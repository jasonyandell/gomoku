# Core ML / ANE Residency Lab — evidence-discipline control plane

**Scope of this page:** the **Cap ladder** for ANE-residency claims, the **receipt schema** and **powermetrics protocol** that elevate a receipt past `coreml-scheduled`, and the shape-scouting matrix for residency exploration. This is the evidence-discipline page only.

**Where to read for other ANE topics:**
- [coreml-design-envelope-and-our-fit.md](coreml-design-envelope-and-our-fit.md) — the **canonical entry point** for ANE research: strategic framing (what Core ML is built for, where our workload fits), the **current envelope state** (with L09 through L09e measurements folded in), the **research lanes** (status + findings + reactivation triggers), and the **inbound-research landing zone**. **Read that page first** if you're new to the ANE story; come back here for cap discipline and residency-proof workflow.
- [m5-max-fp16-and-throughput-regimes.md](m5-max-fp16-and-throughput-regimes.md) — chip-level MPS findings; Finding 2 is the MPS analog of what L09g would measure for Core ML.
- [ane-int8-inference.md](ane-int8-inference.md) — **historical** (WL5-era) scoping doc; partly executed (Core ML evaluator shipped 2026-05-23).

Created 2026-05-22 as the control-plane notebook for the autonomous Core ML / ANE residency lab. This page owns the claim-cap discipline and the residency-proof workflow. Raw receipts stay in `/tmp`, `sweep_logs/`, or the script output paths; durable interpretation can be appended here. Three-engine pipeline framing has moved to [coreml-design-envelope-and-our-fit.md § Our workload through that lens](coreml-design-envelope-and-our-fit.md#our-workload-through-that-lens) — this page is residency-evidence only.

## Goal

Determine which Core ML model shapes, compute-unit settings, batch geometries, and worker pressure patterns *actually* make the Apple Neural Engine resident (per `powermetrics ane_power` evidence) for gomoku-like inference on Jason's M5 Max — distinct from claims that merely *request* ANE via `CPU_AND_NE` routing.

The residency question is independent of, and complementary to, the engine-isolation question that the [perf-lab](perf-lab-charter.md) measures via the L09* lane family. Engine-isolation (which the perf-lab proves via trainer-step deltas) tells us *whether Core ML offload is worth it holistically*; residency (which this page's cap ladder gates) tells us *whether the Core ML offload is actually running on the ANE silicon* vs CPU/GPU under the CPU_AND_NE routing. Both matter. The perf-lab can produce a `coreml-isolated` win without resolving the residency question; this page's discipline is what would elevate such a win to `ane-metered`.

## Cap status of the perf-lab L09* receipts (2026-05-23)

Every Core ML measurement that has landed in [experiment-ledger.md](../ops/experiment-ledger.md) so far sits at `coreml-scheduled` or `coreml-isolated`. None has cleared `ane-metered`. The table below pins the cap each L09* receipt has actually established:

| Receipt | Source | Engine arm | Cap cleared | Evidence | What it'd take to elevate |
|---|---|---|---|---|---|
| L09 (R-TRAIN-ANE reject) | 2026-05-23 | small/V=64/CPU_AND_NE | `coreml-isolated` | trainer_step_s_p50 -55.7% vs torch baseline (overlap measurement clean in trainer.log) | Re-run with `powermetrics ane_power` in matched window |
| L09c (R-TRAIN-TINY-ANE PROMOTE) | 2026-05-23 | tiny/V=64/CPU_AND_NE | `coreml-isolated` | trainer_step_s_p50 -16.3% vs torch baseline | Re-run with `powermetrics ane_power`; see L09e' lane in design-envelope page |
| L09d (R-TRAIN-MEDIUM-ANE reject) | 2026-05-23 | medium/V=512/CPU_AND_NE | `coreml-isolated` | trainer_step_s_p50 -81.4% vs torch+fp16 baseline | Re-run with `powermetrics ane_power` |
| L09c-V512 (reject) | 2026-05-23 | tiny/V=512/CPU_AND_NE | `coreml-isolated` | trainer_step_s_p50 -62.5% vs torch+fp16 baseline | Re-run with `powermetrics ane_power` |
| L09e CPU_AND_GPU | 2026-05-23 | small/V=64/CPU_AND_GPU | `coreml-isolated` | trainer_step_s_p50 similar to L09 | Re-run with `powermetrics ane_power` (would also resolve whether CPU_AND_GPU ever actually uses ANE — known unknown) |
| L09e ALL | 2026-05-23 | small/V=64/ALL | `coreml-isolated` | trainer_step_s_p50 similar to L09 | Re-run with `powermetrics ane_power` |
| L09b (blocked) | 2026-05-23 | small/V=64/CPU_AND_NE + fp16-eval | n/a (failed at startup) | Pipeline-order bug in selfplay_worker._maybe_half; fixed and made graceful no-op | Lane semantically redundant; Core ML already uses FLOAT16 internally |

**Implications:**

- The L09c PROMOTE narrative ("ANE pays at tiny+V=64") is **`coreml-isolated`-cap correct** but **`ane-metered`-cap unproven**. Equivalently safe wording: "Core ML at CPU_AND_NE routing pays at tiny+V=64; whether the gain comes from ANE residency or from CPU dispatch under CPU_AND_NE routing is not yet resolved." (Note post-hollance: under CPU_AND_NE the alternative to ANE is **CPU**, not GPU — `CPU_AND_NE` excludes the GPU entirely. The GPU-via-MPS path is only reachable via `CPU_AND_GPU` or `ALL`.)
- The L09 (small) and L09d (medium) rejects are **`coreml-isolated`-cap correct rejects** — the data shows the Core ML offload doesn't pay holistically at those shapes, regardless of which silicon Core ML actually ran on. Residency proof wouldn't change the rejection.
- **The next ANE-evidence-elevating lane is L09e'** (residency proof for L09c). **UNBLOCKED 2026-05-23 (post-hollance-absorption)** via the no-sudo `H11ANEServicesThread` technique from [hollance/neural-engine](https://github.com/hollance/neural-engine/blob/master/docs/is-model-using-ane.md). See [coreml-design-envelope-and-our-fit.md § L09e'](coreml-design-envelope-and-our-fit.md#l09e--ane-residency-proof-via-thread-name-unblocked-post-hollance-absorption). Original powermetrics version retained as a fallback if thread-name evidence is inconclusive.
- Future Core ML or ANE research dropping (e.g., new ANE features, new residency-instrumentation APIs) should default to the `ane-metered`-or-better cap when re-running these shapes.

**Updated no-sudo proof-of-residency techniques (folded from hollance/neural-engine 2026-05-23):**

| Technique | Sudo needed? | What it proves | What it doesn't |
|---|---|---|---|
| `ps -M <pid>` showing `H11ANEServicesThread` | NO | Core ML uses ANE for at least part of the model | Doesn't tell you what fraction is on ANE vs CPU |
| `powermetrics ane_power` > idle | YES (sudo) | ANE rail drew power during the measurement window | Same caveat — fraction unknown without per-engine timing |
| `lldb` + breakpoint on `-[_ANEModel program]` | NO | ANE is being invoked at least once per inference | Same caveat |
| `lldb` + `image list Espresso` + Espresso symbols | NO | Which Core ML engines (ANE / MPS / BNNS) were loaded | Doesn't tell you which engine ran which layer |
| Instruments Time Profiler call tree | NO | Per-call attribution to engine via stack symbols | Requires Instruments setup; heavier than `ps -M` |

Cheapest for our lab's needs: **`ps -M <pid>`** during the measurement window of any running `lab_train_cell` cell with `--evaluator coreml`. L09e' uses this as the primary technique.

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

## Next Actions

1. Re-run with cached/passwordless sudo available (`sudo -n true` must
   pass). Start with a fresh Vision positive control and a fresh
   `conv,resnet,gomoku` scout in adjacent windows.
2. Use `--powermetrics required` for any claim cell; do not promote any
   claim above `coreml-scheduled` if sudo/powermetrics is unavailable.
3. Summarize each cell by cap, ANE mean/max mW, positions/sec, ready
   workers, and errors. Append only the interpretation here; leave raw
   receipts where they are.
4. If any toy conv/resnet shape reaches `ane-metered`, narrow around its
   shape before trying to make Gomoku match it.
5. If Gomoku reaches `ane-metered`, run a production-shaped overlap probe
   before wiring self-play. The win condition is lower trainer slowdown
   plus acceptable self-play throughput, not raw Core ML latency.
