# Best Cells per Reference Point

Single source of truth for the current best cell at each quality
reference point. Updated on every Reviewer-approved promote per the
[perf-lab-charter](../topics/perf-lab-charter.md). Promotion requires
a no-behavior-change knob movement (sims and model size pin the
quality point).

## R-S* — generator throughput (aug-pos/sec)

| ref | quality pin | current best cell | aug/s | promoted | from receipt |
|---|---|---|---|---|---|
| **R-S400** | small / S=400 | small / W=8 / G=8 / S=400 / **V=512** | 4,765 | 2026-05-23 | L01-wave-extrapolation |
| **R-S200** | small / S=200 | small / W=8 / G=8 / S=200 / **V=512** | 9,156 | 2026-05-23 | L03-sims-x-wave |
| **R-S100** | small / S=100 | small / W=8 / G=8 / S=100 / **V=512** | 15,082 | 2026-05-23 | L03-sims-x-wave |
| R-S400-tiny | tiny / S=400 | tiny / W=8 / G=8 / S=400 / V=64 | 7,326 | 2026-05-23 | canonical-sweep-mainframe |
| R-S100-tiny | tiny / S=100 | tiny / W=16 / G=16 / S=100 / V=32 | 19,346 | 2026-05-23 | canonical-sweep-mainframe |

## R-TRAIN-* — trainer + concurrent generator (epochs/sec)

The holistic metric. Live-training cells; cell-budget-stitched (warmup
cell + measure cell) per the charter's cell-time ceiling.

| ref | quality pin | current best cell | epochs/sec | games/sec | promoted | from receipt |
|---|---|---|---|---|---|---|
| **R-TRAIN-WL5** | full WL5 production recipe | small / W=8 / G=8 / S=400 / V=64 / EMA τ=0.99 / grad_accum=4 | TBD | TBD | pending | L10 first measurement |
| **R-TRAIN-LEAN** | WL5 with V=128 | small / W=8 / G=8 / S=400 / V=128 / same EMA + grad-accum | TBD | TBD | pending | L11 first measurement |
| **R-TRAIN-ANE** | WL5 with workers on Core ML | WL5 recipe but workers use --evaluator coreml | TBD | TBD | pending | L09 first measurement |

R-S400 is the primary metric — it's the WL5-era production shape and
the headline number in [status.md](status.md).

## Promotion log

Newest first. Append on every promote; never overwrite an old row.

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
