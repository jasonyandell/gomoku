# Core ML / ANE Residency Lab — evidence-discipline control plane

**Scope of this page:** the **Cap ladder** for ANE-residency claims, the **receipt schema** and **powermetrics protocol** that elevate a receipt past `coreml-scheduled`, and the shape-scouting matrix for residency exploration. This is the evidence-discipline page only.

**Where to read for other ANE topics:**
- [coreml-design-envelope-and-our-fit.md](coreml-design-envelope-and-our-fit.md) — the **canonical entry point** for ANE research: strategic framing (what Core ML is built for, where our workload fits), the **current envelope state** (with L09 through L09e measurements folded in), the **research lanes** (status + findings + reactivation triggers), and the **inbound-research landing zone**. **Read that page first** if you're new to the ANE story; come back here for cap discipline and residency-proof workflow.
- [m5-max-fp16-and-throughput-regimes.md](m5-max-fp16-and-throughput-regimes.md) — chip-level MPS findings; Finding 2 is the MPS analog of what L09g would measure for Core ML.
- [ane-int8-inference.md](ane-int8-inference.md) — **historical** (WL5-era) scoping doc; partly executed (Core ML evaluator shipped 2026-05-23).

Created 2026-05-22 as the control-plane notebook for the autonomous Core ML / ANE residency lab. This page owns the claim-cap discipline and the residency-proof workflow. Raw receipts stay in `/tmp`, `sweep_logs/`, or the script output paths; durable interpretation can be appended here. Three-engine pipeline framing has moved to [coreml-design-envelope-and-our-fit.md § Our workload through that lens](coreml-design-envelope-and-our-fit.md#our-workload-through-that-lens) — this page is residency-evidence only.

## Goal

Determine which Core ML model shapes, compute-unit settings, batch geometries, and worker pressure patterns *actually* make the Apple Neural Engine resident for gomoku-like inference on Jason's M5 Max — distinct from claims that merely *request* ANE via `CPU_AND_NE` routing.

**Resolved 2026-05-23 (L09i + L09i-fix):** the dominant gate is the **input-shape declaration**, not the compute-unit hint. A symbolic `ct.RangeDim` (or `ct.EnumeratedShapes`) batch dimension silently demotes the whole program to CPU/BNNS regardless of `CPU_AND_NE`; **a single fixed static batch is the only ANE-resident export**. With that fix, gomoku inference reaches `sample`-confirmed ANE residency. The remaining open question is no longer "can we get ANE residency" (yes) but "does ANE residency *pay*" — and the early answer is: not on raw throughput, but plausibly on **contention-immunity** (workers vacate the GPU entirely, freeing a heavy MPS trainer). See the cap-status section below and [coreml-design-envelope-and-our-fit.md](coreml-design-envelope-and-our-fit.md).

The residency question is independent of, and complementary to, the engine-isolation question that the [research lab](research-lab-charter.md) measures via the L09* lane family. Engine-isolation (which the lab proves via trainer-step deltas) tells us *whether Core ML offload is worth it holistically*; residency (which this page's cap ladder gates) tells us *whether the Core ML offload is actually running on the ANE silicon* vs CPU/GPU under the CPU_AND_NE routing. Both matter. The lab can produce a `coreml-isolated` win without resolving the residency question; this page's discipline is what would elevate such a win to `ane-metered`.

## Cap status of the perf-lab L09* receipts (2026-05-23)

**RESIDENCY RESOLVED (2026-05-23, L09i + L09i-fix):** The whole pre-L09i L09* family was running on **CPU/BNNS, never the ANE** — the lab export (`gomoku/coreml_evaluator.py:267`, `export_model_to_coreml`) declared a symbolic `ct.RangeDim` batch dimension, which silently demotes the entire Core ML program off the ANE (the ANE needs fully static input shapes). The lab export and a known-ANE-resident scout export (`scripts/coreml_ane_residency_scout.py --batch-shape fixed`) emit **byte-identical MIL op graphs**; RangeDim-vs-fixed-batch is the only difference. So none of L09 / L09c / L09d / L09c-V512 / L09e ever cleared `ane-metered` — and now we know they couldn't have, because they weren't on the ANE at all. Their `coreml-isolated` engine-isolation wins are still valid (workers vacated MPS), but that was *CPU/BNNS* isolation. **L09i-fix** (RangeDim → single fixed static batch) is the first lab receipt to reach **`sample`-confirmed ANE residency** (a status above `coreml-isolated`; strict `ane-metered` via powermetrics still pending sudo, but `sample` settles the engine-placement question that the cap ladder cares about).

| Receipt | Source | Engine arm | Cap cleared | Evidence | What it'd take to elevate |
|---|---|---|---|---|---|
| **L09i-fix (fixed-batch, tile-sized)** | **2026-05-23** | **tiny/V=64/CPU_AND_NE, fixed static batch** | **`sample`-confirmed ANE residency (above `coreml-isolated`)** | **`sample` shows `AneInferenceOperationImplUsingAnefAPIs` / `_ANEClient doEvaluateDirect` / `AppleNeuralEngine`, zero BNNS lines — confirmed twice (isolated micro-probe + live self-play worker). 7,697.7 aug/s, trainer_step 0.0172s, 18 epochs/window** | Strict `ane-metered` via same-window `powermetrics ane_power` (pending cached sudo) |
| **L09i-fix (fixed-batch 1024)** | **2026-05-23** | **tiny/V=64/CPU_AND_NE, fixed static batch=1024** | **`sample`-confirmed ANE residency** | **Same `sample` evidence; 2,303.9 aug/s (~7× pad tax over the ~140-leaf wave tile)** | Strict `ane-metered` via `powermetrics ane_power` |
| L09i (diagnostic) | 2026-05-23 | export-graph inspection | n/a (root-cause diagnostic) | Found symbolic `ct.RangeDim` batch dim → CPU/BNNS demotion; byte-identical MIL graph vs ANE-resident scout export | n/a — resolved the strand |
| L09 (R-TRAIN-ANE reject) | 2026-05-23 | small/V=64/CPU_AND_NE | `coreml-isolated` — **was CPU/BNNS, not ANE (per L09i)** | trainer_step_s_p50 -55.7% vs torch baseline (overlap measurement clean in trainer.log) | Re-run on the L09i-fix fixed-batch export to reach the ANE; old powermetrics re-run is moot (it was never on ANE) |
| L09c (R-TRAIN-TINY-ANE PROMOTE) | 2026-05-23 | tiny/V=64/CPU_AND_NE | `coreml-isolated` — **was CPU/BNNS, not ANE (per L09i)** | trainer_step_s_p50 -16.3% vs torch baseline; 10,762.6 aug/s (now understood as the CPU/BNNS number) | Already superseded by L09i-fix on the residency axis; throughput compare is CPU/BNNS-vs-ANE |
| L09d (R-TRAIN-MEDIUM-ANE reject) | 2026-05-23 | medium/V=512/CPU_AND_NE | `coreml-isolated` — **was CPU/BNNS, not ANE (per L09i)** | trainer_step_s_p50 -81.4% vs torch+fp16 baseline | Re-run on the fixed-batch ANE export (L09-ANE-resident-reopen lane) |
| L09c-V512 (reject) | 2026-05-23 | tiny/V=512/CPU_AND_NE | `coreml-isolated` — **was CPU/BNNS, not ANE (per L09i)** | trainer_step_s_p50 -62.5% vs torch+fp16 baseline | Re-run on the fixed-batch ANE export |
| L09e CPU_AND_GPU | 2026-05-23 | small/V=64/CPU_AND_GPU | `coreml-isolated` — **was CPU/BNNS, not ANE (per L09i)** | trainer_step_s_p50 similar to L09 | Routing sweep is moot for residency — RangeDim demoted all arms to CPU/BNNS regardless of routing hint |
| L09e ALL | 2026-05-23 | small/V=64/ALL | `coreml-isolated` — **was CPU/BNNS, not ANE (per L09i)** | trainer_step_s_p50 similar to L09 | Same — re-run on fixed-batch export to actually exercise routing |
| L09b (blocked) | 2026-05-23 | small/V=64/CPU_AND_NE + fp16-eval | n/a (failed at startup) | Pipeline-order bug in selfplay_worker._maybe_half; fixed and made graceful no-op | Lane semantically redundant; Core ML already uses FLOAT16 internally |

**Implications:**

- **The `ct.RangeDim` → CPU/BNNS demotion invalidates the "ANE" label on every pre-L09i receipt.** The L09c PROMOTE's +33.9% and L09's/L09d's rejects were all CPU/BNNS. Their `coreml-isolated` engine-isolation deltas remain correct (the workers were off MPS), but it was CPU isolation, not ANE isolation. `ct.EnumeratedShapes` (a few discrete sizes) also falls back to BNNS — **only a single fixed static batch stays ANE-resident.**
- **L09i-fix is the first `sample`-confirmed ANE-resident receipt in the lab.** `sample` is a no-sudo engine-placement proof: it shows the ANE hot-path symbols with zero BNNS lines. It sits **above `coreml-isolated`** on the ladder because it answers the engine question the cap discipline was built around. It is **not yet** strict `ane-metered` — that still requires same-window `powermetrics ane_power`, which is blocked on cached sudo — but `sample` settles "which engine ran the model," which `powermetrics` and `sample` agree on when both are available.
- The L09 (small) and L09d (medium) rejects were `coreml-isolated`-cap correct *rejects of the CPU/BNNS path*. Whether the ANE-resident fixed-batch export rejects at those shapes is now an open re-measurement (L09-ANE-resident-reopen).
- **Throughput note:** L09i-fix's genuine ANE residency is **still a reject on raw throughput** — 7,697.7 aug/s tile-sized is −4.2% vs torch baseline and −28.5% vs the CPU/BNNS L09c. The ANE's plausible value is **contention-immunity under a heavy GPU trainer** (best trainer_step 0.0172s, 18 epochs/window), not eval speed. See [coreml-design-envelope-and-our-fit.md § Current state](coreml-design-envelope-and-our-fit.md#current-state--residency-resolved-the-prior-envelope-was-all-cpubnns-2026-05-23-l09i--l09i-fix).
- The prior "next elevating lane is L09e' (thread-name)" plan is **superseded** — L09i-fix used the stronger `sample` technique (full hot-path symbol attribution, not just thread presence) and confirmed residency directly. Strict `ane-metered` via powermetrics remains the only higher rung, pending sudo.

**Updated no-sudo proof-of-residency techniques (folded from hollance/neural-engine 2026-05-23):**

| Technique | Sudo needed? | What it proves | What it doesn't |
|---|---|---|---|
| **`sample <pid>` stack-sampling** | **NO** | **Which engine ran the model: ANE hot-path symbols (`AneInferenceOperationImplUsingAnefAPIs` / `_ANEClient doEvaluateDirect` / `AppleNeuralEngine`) vs `BNNS` lines for CPU. Zero BNNS + ANE symbols = ANE-resident.** | **Sampled attribution, not a per-layer split or rail power; a near-idle ANE could be under-sampled (mitigate with a hot probe).** |
| `ps -M <pid>` showing `H11ANEServicesThread` | NO | Core ML uses ANE for at least part of the model | Doesn't tell you what fraction is on ANE vs CPU |
| `powermetrics ane_power` > idle | YES (sudo) | ANE rail drew power during the measurement window | Same caveat — fraction unknown without per-engine timing |
| `lldb` + breakpoint on `-[_ANEModel program]` | NO | ANE is being invoked at least once per inference | Same caveat |
| `lldb` + `image list Espresso` + Espresso symbols | NO | Which Core ML engines (ANE / MPS / BNNS) were loaded | Doesn't tell you which engine ran which layer |
| Instruments Time Profiler call tree | NO | Per-call attribution to engine via stack symbols | Requires Instruments setup; heavier than `ps -M` |

Cheapest decisive technique for our lab's needs: **`sample <pid>`** during the measurement window. **L09i-fix used `sample` (not `ps -M`) and confirmed ANE residency twice** — an isolated micro-probe and under the live self-play worker — by showing the ANE hot-path symbols with zero BNNS lines. `sample` is stronger than the `H11ANEServicesThread` thread-presence check (the L09e' plan): a thread can exist without the model running on ANE, but the ANE inference symbols in the live stack are direct engine-placement evidence. `sample` is the technique that promotes a receipt to **`sample`-confirmed ANE residency** (above `coreml-isolated`); only strict `ane-metered` via `powermetrics ane_power` sits higher, and that remains sudo-blocked.

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
