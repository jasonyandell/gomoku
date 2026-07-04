# M5 as Mainframe — pushing this Mac

The perf-and-hardware philosophy: treat one M5 Max (48 GB unified) as a knowable
mainframe and get everything out of it. This hub collects what we explored and
learned about *this machine*.

> **← Hubs:** [index](index.md) · sibling hubs: [AlphaZero](alphazero.md) ·
> [Experiments](experiments.md) · [Derby](derby.md) · [Autolab](autolab.md) ·
> [Reference](reference.md)

## The guiding lens

[m5-max-as-mainframe.md](topics/m5-max-as-mainframe.md) — the philosophy and
sequence for the perf era on Jason's M5 Max.

## What we explored & learned

| Topic | Finding |
|---|---|
| **[m5-max-fp16-and-throughput-regimes.md](topics/m5-max-fp16-and-throughput-regimes.md)** ⭐ | **Flagship perf findings**: fp16-on-MPS reversal (+97% eval), bandwidth-vs-dispatch regimes, independent levers compose multiplicatively (2.529× vs 2.530× predicted). |
| [perf-bench-vs-real-training-cost.md](topics/perf-bench-vs-real-training-cost.md) | The fp16 epilogue: the +152% bench measured cold-buffer generation; the real run runs away to ~3–7 min/epoch — a bench that stops before buffer-fill is non-predictive. |
| [mcts-perf-ceiling.md](topics/mcts-perf-ceiling.md) | Where MCTS gen-time wins are and are not. Don't re-port "v2 storage" — we're already there. **2026-07 perf blitz: the gen-loop thread is CLOSED** — in the oracle-dominated regime the VCT solver is ~90%+ of 13×13 gen wall, so the levers all landed on the *solver* (cap25 budget flip #114 + `lanes=K` kernel #114 + one-worker refill #112), not the MCTS loop. |
| [batched-eval-arena.md](topics/batched-eval-arena.md) | Current core eval infra (`gomoku/arena.py`, 2026-07-01): bulk matches at self-play speed by batching net leaf-evals across the whole field. ~6–120× faster eval; batched VCT finisher; default eval path with byte-identical legacy escape hatches. |
| [wall-clock-to-elo-metric.md](topics/wall-clock-to-elo-metric.md) | _Design-only spec_: the R-ELO-\* metric family (MTTE + EPWH) — wall-clock-to-elo as the runaway-proof objective the throughput proxies must be checked against. Not yet implemented. |
| 15x15-era-feasibility-and-plan.md *(removed 2026-07-04; recover: `git show ca76350:wiki/_archive/topics/15x15-era-feasibility-and-plan.md`)* | Measured board/net scaling on MPS (96×8 @15×15 costs only 2.32× at wave=64); week-scale feasibility envelope. |
| **[coreml-design-envelope-and-our-fit.md](topics/coreml-design-envelope-and-our-fit.md)** | **Canonical ANE/Core ML entry**: why Core ML misfits our tiny high-rate workload; the RangeDim bug that faked prior "ANE" results; the value is contention-immunity, not throughput. |
| [coreml-ane-residency-lab.md](topics/coreml-ane-residency-lab.md) | Rail-proof lab for ANE residency claims (powermetrics-gated). |
| [ane-int8-inference.md](topics/ane-int8-inference.md) | _Historical_ — the WL5-era ANE int8 scout; **superseded by coreml-design-envelope**. |
| [buffer-bit-packing.md](topics/buffer-bit-packing.md) | Replay-buffer compression: bit-packed planes + FP16 policy. |
| [m5-max-cross-engine-coupling.md](topics/m5-max-cross-engine-coupling.md) | Measured co-tenancy envelope (GPU/CPU/ANE interference). |
| [activity-monitor-perf-runbook.md](topics/activity-monitor-perf-runbook.md) | Practical knobs + interpretation rules for Activity Monitor perf experiments. |

## Hard-won gotcha

Killing MLX/Metal work **mid-compile** (e.g. a `timeout` SIGKILL) wedges the
Metal compiler service system-wide and can crash WindowServer — **bound GPU runs
by WORK, never an external timeout kill.** (See also memory: *Metal compiler
wedge*.)

## Full page index — every page in this hub

*Complete map (12 pages); the sections above surface the headline ones.*

| Page | Note |
|---|---|
| [m5-max-as-mainframe.md](topics/m5-max-as-mainframe.md) | guiding lens |
| [mcts-perf-ceiling.md](topics/mcts-perf-ceiling.md) |  |
| [m5-max-fp16-and-throughput-regimes.md](topics/m5-max-fp16-and-throughput-regimes.md) | flagship perf findings |
| [perf-bench-vs-real-training-cost.md](topics/perf-bench-vs-real-training-cost.md) | the fp16 epilogue / runaway trap |
| [batched-eval-arena.md](topics/batched-eval-arena.md) | current core eval infra (2026-07-01) |
| [wall-clock-to-elo-metric.md](topics/wall-clock-to-elo-metric.md) | design-only spec (R-ELO-\* / MTTE + EPWH) |
| [m5-max-cross-engine-coupling.md](topics/m5-max-cross-engine-coupling.md) |  |
| [coreml-design-envelope-and-our-fit.md](topics/coreml-design-envelope-and-our-fit.md) | canonical ANE entry |
| [coreml-ane-residency-lab.md](topics/coreml-ane-residency-lab.md) |  |
| [ane-int8-inference.md](topics/ane-int8-inference.md) | absorbed by coreml-design-envelope; keep historical with banner _([superseded])_ |
| [buffer-bit-packing.md](topics/buffer-bit-packing.md) |  |
| [activity-monitor-perf-runbook.md](topics/activity-monitor-perf-runbook.md) |  |
