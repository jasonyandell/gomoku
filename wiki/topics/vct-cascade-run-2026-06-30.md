# VCT cascade — at-scale run record (2026-06-30, full rapfi corpus) (#97)

Live, append-only record of the **actual full-corpus labeling run** — what each
ladder rung does *at scale* on the real 56.1M-position rapfi corpus: where width
settles, sustained boards/s, wall time, and the verdict split (how many resolve
vs. how many cap and fall through to the next rung). This is the deepening curve
the throughput-knee sweep ([[vct-cascade-labeler]]) could only sample.

- **Corpus:** `~/data/raphi_vct/positions/` — 56,121,658 unique D4-canonical positions.
- **Run root:** `~/data/raphi_vct/` (`results/cap<N>/`, `survivors/cap<N>/`, `perf/`).
- **Launched:** 2026-06-30 06:58 (after the 06:53 reboot cleared the Metal wedge).
- **Execution:** `watchdog.sh` → `cascade.py`, all proof-outputs ON
  (move+support+carriers+w), `complete=OFF`, work-bounded (no `timeout` kills).
- **Ladder:** `50, 100, 250, 500, 1000, 2000, 4000, 10000, 20000, 50000, 100000`.

## Per-rung results at scale (the deepening curve)
Each rung runs only on the prior rung's `cap` survivors. `cap` count = input to the
next rung. Throughput = cascade-only perf rows (last night's sweep filtered out by ts).

| rung | input boards | steady width | peak b/s | median b/s | GPU wall | win | no_win | cap (→next) | win% | cap% |
|-----:|-------------:|-------------:|---------:|-----------:|---------:|----:|-------:|------------:|-----:|-----:|
| cap50 | 56,121,658 | 524,288 | 43,089 | 42,496 | 22.5 min | 26,837,059 | 21,605,369 | 7,130,052 | 48.3 | 12.8 |

*(rows appended as each rung completes)*

## Width-ramp curve at scale (cap50 — where throughput saturates)
The cascade auto-doubles batch width from 2,048 until throughput stops climbing,
then holds. cap50 saturated at **W=524,288** (going wider bought <5%):

| width | boards/s |
|------:|---------:|
| 2,048 | 3,306 |
| 4,096 | 7,142 |
| 8,192 | 13,061 |
| 16,384 | 21,370 |
| 32,768 | 27,566 |
| 65,536 | 35,411 |
| 131,072 | 38,953 |
| 262,144 | 41,403 |
| **524,288** | **43,089** ← knee |

**Matches the standalone sweep's cap50 knee (43,397 b/s).** The cascade reproduces
the measured throughput law in production: width is king until GPU saturation, then
flat. Note it settled at 524k, not the sweep's nominal 2M plateau — the extra width
gains nothing at cap50, so the ramp correctly stopped early.

## Notes / anomalies
- cap50 resolved **87.2%** of the whole corpus definitively (48.3% win + 38.9%
  no_win) at just 50 nodes — confirming most rapfi positions are tactically
  shallow. The interesting tail is the **12.8% (7.13M)** that cap and fall through.

**Cross-links:** [vct-cascade-labeler.md](vct-cascade-labeler.md) (architecture +
knee sweep) · [mega-vct-solver.md](mega-vct-solver.md) ·
[gpu-vct-feasibility.md](gpu-vct-feasibility.md).
