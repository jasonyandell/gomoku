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
| cap100 | 7,200,627 | 262,144 | 9,257 | 9,176 | 13.2 min | 108,070 | 450,445 | 6,642,112 | 1.5 | 92.2 |

*(rows appended as each rung completes)*

> **cap50→cap100 survivor count grew 7,130,052 → 7,200,627**: cap50 was still
> flushing its last shards when first sampled; 7.20M is the true cap50 survivor set.

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
  shallow. The interesting tail is the **12.8% (7.20M)** that cap and fall through.
- **THE KEY AT-SCALE FINDING — survivor-rung throughput collapses far below the
  single-shot knee.** The standalone sweep measured the knee on the *natural*
  rapfi mix; each cascade rung after cap50 runs only on the *hard survivors* (boards
  that already capped at the prior budget), which run the **full** node budget with
  no early-out. So measured b/s per rung << the sweep knee:

  | rung | survivor-rung b/s (this run) | single-shot knee (natural mix) | ratio |
  |-----:|-----------------------------:|-------------------------------:|------:|
  | cap50 | 43,089 | 43,397 | 0.99× (cap50 IS the natural mix) |
  | cap100 | 9,257 | 23,822 | **0.39×** |
  | cap250 | ~3,800 | 10,107 | **~0.38×** |

  Plan deep-rung wall-clock off the *survivor* rate (~0.4× the knee), not the knee.
- **The deeper you go, the less budget buys.** cap100 (2× the nodes) converted only
  **7.8%** of cap50's survivors to a definitive verdict (1.5% win + 6.3% no_win);
  the other 92.2% still cap. The tail is hard, not slow — the deep-win gold is rare
  and lives only at the high-budget rungs, exactly as the cascade was built to find.

**Cross-links:** [vct-cascade-labeler.md](vct-cascade-labeler.md) (architecture +
knee sweep) · [mega-vct-solver.md](mega-vct-solver.md) ·
[gpu-vct-feasibility.md](gpu-vct-feasibility.md).
