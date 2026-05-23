# Best Cells per Reference Point

Single source of truth for the current best cell at each quality
reference point. Updated on every Reviewer-approved promote per the
[perf-lab-charter](../topics/perf-lab-charter.md). Promotion requires
a no-behavior-change knob movement (sims and model size pin the
quality point).

## R-S* — generator throughput (aug-pos/sec)

| ref | quality pin | current best cell | aug/s | promoted | from receipt |
|---|---|---|---|---|---|
| **R-S400** | small / S=400 | small / W=8 / G=8 / S=400 / **V=512 / fp16-eval** | **9,398.5** | 2026-05-23 | L06-followup-fp16-cells |
| **R-S200** | small / S=200 | small / W=8 / G=8 / S=200 / **V=512 / fp16-eval** | **16,850.8** | 2026-05-23 | L06fu-extended |
| **R-S100** | small / S=100 | small / W=8 / G=8 / S=100 / **V=512 / fp16-eval** | **22,312.1** | 2026-05-23 | L06fu-extended |
| R-S400-tiny | tiny / S=400 | tiny / **W=16** / G=8 / S=400 / **V=512 / fp16-eval** | **22,873.8** | 2026-05-23 | L06-followup-fp16-cells |
| **R-S400-medium** (new ref) | medium / S=400 | medium / W=8 / G=8 / S=400 / **V=512 / fp16-eval** | **3,377.2** | 2026-05-23 | L06fu-extended |
| R-S100-tiny | tiny / S=100 | tiny / W=16 / G=16 / S=100 / V=32 | 19,346 | 2026-05-23 | canonical-sweep-mainframe |

## R-TRAIN-* — trainer + concurrent generator (epochs/sec)

The holistic metric. Live-training cells; cell-budget-stitched (warmup
cell + measure cell) per the charter's cell-time ceiling.

| ref | quality pin | current best cell | epochs/sec | games/sec | promoted | from receipt |
|---|---|---|---|---|---|---|
| **R-TRAIN-WL5** | full WL5 production recipe | small / W=8 / G=8 / S=400 / V=64 / EMA τ=0.99 / grad_accum=4 | **0.0917** | **14.07** | 2026-05-23 | L10-trainer-step-bench |
| ~~R-TRAIN-LEAN~~ (rejected as a target — V=512 trainer cell measured 2,363 aug/s, worse than R-TRAIN-WL5's 3,298) | WL5 with V=512 | small / W=8 / G=8 / S=400 / V=512 / same EMA + grad-accum | 0.0083 | 8.42 | 2026-05-23 (REJECT) | L11-end-to-end-cell |
| ~~R-TRAIN-ANE~~ (rejected as a target — naive ANE-offload measured 1,930 aug/s vs R-TRAIN-WL5's 3,298) | WL5 recipe but workers use --evaluator coreml --coreml-compute-units CPU_AND_NE | small / W=8 / G=8 / S=400 / V=64 / coreml workers | 0.0583 | 8.00 | 2026-05-23 (REJECT, with partial-hypothesis confirmation) | L09-ane-offload-prototype |

R-S400 is the primary metric — it's the WL5-era production shape and
the headline number in [status.md](status.md).

## Promotion log

Newest first. Append on every promote; never overwrite an old row.

- **2026-05-23** — `R-S200` **promote**: V=512 fp32 → **V=512 fp16-eval** (9,156 → 16,850.8 aug/s, **+84.0%**). Mechanism: same as R-S400, eval is bandwidth-bound at S=200/V=512. plies_mean unchanged (15.96 → 15.96). Source: L06fu-extended. Reviewer: pending.
- **2026-05-23** — `R-S100` **promote**: V=512 fp32 → **V=512 fp16-eval** (15,082 → 22,312.1 aug/s, **+48.0%**). Smaller win than R-S200 because at S=100 the MCTS Python overhead dilutes eval's share of wall time. plies_mean unchanged (15.96 → 15.96). Source: L06fu-extended. Reviewer: pending.
- **2026-05-23** — `R-S400-medium` **baselined (new ref)**: medium / W=8 / G=8 / S=400 / V=512 / fp16-eval = **3,377.2 aug/s** (plies_mean 15.95). Combined V=64→V=512 + fp32→fp16 = +142.4% vs medium V=64 fp32=1,393. No prior medium V=512 fp32 baseline exists; estimated fp16-only effect is +62% (between R-S100's +48% and R-S400's +97%, fitting the bandwidth-bound mechanism). Source: L06fu-extended. Reviewer: pending.
- **2026-05-23** — `R-S400` **promote**: V=512 fp32 → **V=512 fp16-eval** (4,765 → 9,398.5 aug/s, **+97.2%** — nearly doubles). No behavior change at the MCTS boundary (outputs cast back to fp32 before host); plies_mean unchanged (15.96 → 15.97). Mechanism: small model at V=512 is memory-bandwidth-limited; fp16 halves bandwidth. The historic "fp16 on MPS is slow" claim is now disproven for our eval workload at torch 2.11.0 + fused conv+bn. Source: L06-followup-fp16-cells. Reviewer: APPROVE (precedent: fp16 with fp32 output-cast counts as no-behavior-change for the perf lab; verified at gomoku/mcts.py:519-529).
- **2026-05-23** — `R-S400-tiny` **promote**: V=512 fp32 → **V=512 fp16-eval** (22,088 → 22,873.8 aug/s, **+3.6%** — much smaller than small's near-doubling because tiny at V=512 is MPS-dispatch-limited, not memory-bandwidth-limited). Same no-behavior-change reasoning; plies_mean 15.96 → 15.96 (unchanged). Source: L06-followup-fp16-cells. Reviewer: APPROVE (precedent: fp16 with fp32 output-cast counts as no-behavior-change for the perf lab; verified at gomoku/mcts.py:519-529).
- **2026-05-23** — `R-TRAIN-ANE` **rejected** (with partial-hypothesis confirmation): small / W=8 / G=8 / S=400 / V=64 / EMA τ=0.99 / grad_accum=4 / workers on Core ML CPU_AND_NE — measured **1,930.3 aug/s, 8.00 games/s, 0.0583 epochs/s, trainer_step_s_p50=0.0227s** — **-41.5%** vs R-TRAIN-WL5 V=64 torch. Trainer-side hypothesis CONFIRMED: trainer_step_s_p50 dropped 56% (0.051s → 0.023s) once workers vacated MPS. Worker-side hypothesis REJECTED: Core ML eval at small/V=64 is ~2× slower than torch/MPS (epoch 8 gen=10.3s vs L10's gen=~5s), and that loss dominates the trainer-side gain. Mechanism is mechanically clean in both trainer logs. Follow-up candidates: L09b (different `--coreml-compute-units` routing) and L09c (tiny model on ANE — smaller per-eval graph might amortize Core ML overhead better). Source: L09-ane-offload-prototype. Reviewer: APPROVE.
- **2026-05-23** — `R-TRAIN-LEAN` V=512 **rejected**: small / W=8 / G=8 / S=400 / V=512 / EMA τ=0.99 / grad_accum=4 measured **2,362.8 aug/s, 8.42 games/s, 0.0083 epochs/s, trainer_step_s_p50=0.138s** — **-28.4%** vs R-TRAIN-WL5 V=64. Mechanism: V=512 fills the buffer 2.4× faster (buf=199,608 vs 83,208 at epoch 3) → fixed sgd_per_position=0.0025 produces 3.36× more SGD steps per epoch (306 vs 91) → trainer's 43s of per-epoch SGD starves workers of MPS → games/s drops 40%. Pure-gen R-S* promotes (V=512) do NOT compound at the live-training level. V=64 stays the R-TRAIN-WL5 default. Source: L11-end-to-end-cell. Reviewer: APPROVE.
- **2026-05-23** — `R-TRAIN-WL5` **baselined**: small / W=8 / G=8 / S=400 / V=64 / EMA τ=0.99 / grad_accum=4 — **3,297.6 aug/s, 14.07 games/s, 0.0917 epochs/s, trainer_step_s_p50=0.051s**. First-ever end-to-end live-training measurement. Trainer contention costs ~30.8% on aug/s vs R-S400 pure-gen (4,765 aug/s). Source: L10-trainer-step-bench. Reviewer: APPROVE.
- **2026-05-23** — `R-S400-tiny`: W=8 V=64 → **W=16 V=512** (7,326 →
  22,088 aug/s, **+201.5%**). No behavior change; tiny model, eval
  batch shape + worker count both shifted. **Model-dependent W peak**:
  at tiny W=16 beats W=8 at V=512 (opposite of small at V=512 per L02).
  Source: L07-tiny-contour. Reviewer: APPROVE.
- **2026-05-23** — `R-S100`: V=64 → **V=512** (11,151 → 15,082 aug/s,
  **+35.2%**). No behavior change; eval batch shape only. Source:
  L03-sims-x-wave. Reviewer: APPROVE.
- **2026-05-23** — `R-S200`: V=64 → **V=512** (6,006 → 9,156 aug/s,
  **+52.5%**). No behavior change; eval batch shape only. Source:
  L03-sims-x-wave. Reviewer: APPROVE.
- **2026-05-23** — `R-S400`: V=128 → **V=512** (4,048 → 4,765 aug/s,
  **+17.7%**; **+49.5%** cumulative since WL5 V=64). No behavior
  change; eval batch size only. Plateau knee at V=512 (V=768/1024
  flat). Source: L01-wave-extrapolation. Reviewer: APPROVE.
- **2026-05-23** — `R-S400`: V=64 → V=128 (3,188 → 4,048 aug/s, +27%).
  No behavior change; eval batch size only. Source:
  canonical-sweep-mainframe.
- **2026-05-23** — `R-S200` baselined at V=64. No prior data.
- **2026-05-23** — `R-S100` baselined at V=64. No prior data.
- **2026-05-23** — `R-S400-tiny` baselined at V=64. No prior data.
- **2026-05-23** — `R-S100-tiny` baselined at the max-throughput
  corner cell from canonical-sweep-mainframe. No prior data.

## Quality gate reminder

A best-cell promotion at the reference points above is **not**
automatically a green light to change the live training cell. Per
the Training-Quality Promotion Gate in
[experiment-ledger.md](experiment-ledger.md), the trainer needs at
least one canary cycle reporting `val/policy_ce` + `val/policy_kl`
against `archives/wl5_validation_v1.pt` + plies/game-shape against
the parent run's band before adopting the new knob. The lab promotes
the cell; the trainer adopts the cell only after the canary.
