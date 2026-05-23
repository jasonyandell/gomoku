# Best Cells per Reference Point

Single source of truth for the current best cell at each quality
reference point. Updated on every promote per the
[perf-lab-charter](../topics/perf-lab-charter.md). Promotion requires
a no-behavior-change knob movement (sims and model size pin the
quality point).

## Reference points

| ref | quality pin | current best cell | aug/s | promoted | from receipt |
|---|---|---|---|---|---|
| **R-S400** | small / S=400 | small / W=8 / G=8 / S=400 / **V=128** | 4,048 | 2026-05-23 | canonical-sweep-mainframe |
| **R-S200** | small / S=200 | small / W=8 / G=8 / S=200 / V=64 | 6,006 | 2026-05-23 | canonical-sweep-mainframe |
| **R-S100** | small / S=100 | small / W=8 / G=8 / S=100 / V=64 | 11,151 | 2026-05-23 | canonical-sweep-mainframe |
| R-S400-tiny | tiny / S=400 | tiny / W=8 / G=8 / S=400 / V=64 | 7,326 | 2026-05-23 | canonical-sweep-mainframe |
| R-S100-tiny | tiny / S=100 | tiny / W=16 / G=16 / S=100 / V=32 | 19,346 | 2026-05-23 | canonical-sweep-mainframe |

R-S400 is the primary metric — it's the WL5-era production shape and
the headline number in [status.md](status.md).

## Promotion log

Newest first. Append on every promote; never overwrite an old row.

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
