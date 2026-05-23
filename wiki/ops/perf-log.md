# ML Perf Lab Log

Narrative timeline for the M5 Max perf era. Sibling to
[experiment-ledger.md](experiment-ledger.md) (formal receipts) and
[status.md](status.md) (control-room summary). This page is for the
day-by-day story: what was tried, what surprised us, what comes next.

Append-only. Lead each entry with `## [YYYY-MM-DD] <lane> | <one-line headline>`
so future sessions can scan with grep.

Cross-refs:
- Philosophy: [../topics/m5-max-as-mainframe.md](../topics/m5-max-as-mainframe.md)
- Procedure: [../topics/perf-lab-session-runbook.md](../topics/perf-lab-session-runbook.md)
- Receipts: [experiment-ledger.md](experiment-ledger.md)
- Board: [frontier.md](frontier.md)

---

## [2026-05-22] lab | post-WL5 perf era opened

WL5 phase-2 closed at e10200 yesterday. Box is idle for the first time in
weeks. Jason called it: "let's get serious about this M5 Max as a
mainframe and squeezing every drop from it. let's set up a lab, run
experiments, keep a log, the whole thing."

State at lab-open:
- Frontier-lab infrastructure exists in `wiki/ops/` and `.frontier/lanes.json`.
- Headline experiment from [m5-max-as-mainframe.md](../topics/m5-max-as-mainframe.md)
  step 4 (the canonical 5-axis sweep producing the contour chart) has **not**
  been executed. The prior `production-contour-20260522` lane only swept
  workers x games-per-worker at default sims/wave/model.
- ANE residency rail-proof lane is blocked on cached/passwordless sudo for
  `powermetrics`. Real evidence exists in the 934b detached worktree but
  was never reproduced in main.
- Buffer-width cheap-test is seeded but warm; deferred per the m5-max
  sequencing in favor of the canonical sweep deliverable.

Next action: run the canonical sweep. Receipts under
`sweep_logs/canonical-sweep-<TS>/`. Open the
[canonical-sweep-mainframe](frontier.md) lane on completion.

## [2026-05-22] lab | canonical sweep paused — BAB1 active on the box

Discovered mid-setup that a `BAB1-buf-ablation-1p5M` run was already
alive when this session opened — another agent context launched it for
the `packed-buffer-cheap-test` lane:

```
trainer PID 27579  (gomoku.train, --resume archives/wl5_e10200_seed.pt,
                    --epochs 500, --replay-buffer-size 1500000,
                    --wave-workers 8 --wave-games-per-worker 8,
                    --validation-archive-path archives/wl5_validation_v1.pt)
8 workers       (PIDs 27596-27603)
eval_worker     (PID 27604, CPU baselines)
wandb-core      (PID 27644)
zsh monitor     (PID 28262, watches for `epoch 10700/10700`)
```

At pause, BAB1 was at e10215/10700 (~10 s/epoch). Expected completion
~80 min. Per [[project-buffer-curation]] memory, this is part of a
buffer-curation research arc, not just a 1.5M-vs-750k throughput
ablation. After BAB1 there may be a paired BAB2 (750k).

What was done in this session before the pause:
- Added [topics/perf-lab-session-runbook.md](../topics/perf-lab-session-runbook.md).
- Wrote `scripts/canonical_sweep.py` (23-cell 5-axis design).
- Wrote `scripts/plot_canonical_sweep.py` (axes + model + contour plots).
- Smoked the driver; first iteration crashed because spawned workers
  collided with BAB1 for MPS memory and exited zombies, then
  `killpg(zombie_pgid)` returned EPERM. Patched to use
  `Popen(start_new_session=True)` + `p.terminate()` + zombie-tolerant
  cleanup, plus a `cell_status=failed` column so contended cells don't
  pollute the summary.

What this session is **not** doing: running the canonical sweep
concurrent with BAB1. The WL5-era 2026-05-22 baseline receipt already
documented how much MPS trainer contention skews bench numbers; running
the sweep against a live trainer would produce numbers we couldn't
defend as "the M5 Max's behavior" without contention.

Next session pickup:
1. Confirm BAB1 (and BAB2 if present) finished — `pgrep -fl
   'gomoku.train|selfplay_worker|eval_worker'` must be empty.
2. Re-smoke 2 tiny cells:
   `python scripts/canonical_sweep.py
   --out-dir sweep_logs/canonical-sweep-smoke-$(date -u +%Y%m%dT%H%M%SZ)
   --secs-per-cell 60 --only tiny_W01,tiny_W08`.
3. Kick the full sweep in background:
   `python scripts/canonical_sweep.py --out-dir
   sweep_logs/canonical-sweep-$(date -u +%Y%m%dT%H%M%SZ)` — ~2 to 3 h.
4. Check progress anytime:
   `python scripts/canonical_sweep.py --out-dir latest --status`.
5. After completion: `python scripts/plot_canonical_sweep.py
   --sweep-dir sweep_logs/canonical-sweep-latest`.
6. File receipt in [experiment-ledger.md](experiment-ledger.md),
   add baseline rows to [baselines.md](baselines.md), promote the
   winning cell in [status.md](status.md), close the
   `canonical-sweep-mainframe` lane in `.frontier/lanes.json`.

## [2026-05-22] lab | canonical sweep driver is first-class resumable

Per Jason: "make resumability first class since this will require many
hours of processing and I'll be using it from time to time." Refactored
`scripts/canonical_sweep.py` against an 8-point contract now documented
under [Resumability contract](../topics/perf-lab-session-runbook.md):

1. **Stable cell IDs** — derived purely from params (e.g.
   `small_W08_G08_S400_V064`); no list-position prefix that would
   shift when the cell list grows.
2. **Atomic source of truth** — append-with-fsync per row;
   write-temp-then-rename for full rewrites.
3. **Per-cell `cell_status`** — `ok` / `failed`; `--retry-failed`
   drops failed rows and wipes their cell_dir before re-running.
4. **PID lock file** at `<out>/.sweep.lock`; aborts on live PID,
   reclaims dead PIDs.
5. **`--status` mode** — done / failed / pending + ETA from median
   wall_secs of completed cells. Exits without spawning GPU work.
6. **`--max-wall-secs N`** budget for short top-up sessions.
7. **SIGINT/SIGTERM handler** — kills workers, drops the
   interrupted cell's row, releases the lock.
8. **`sweep_logs/canonical-sweep-latest`** symlink, refreshed on
   every session; `--out-dir latest` follows it.

All eight surfaces smoked without GPU (box still has BAB1 alive):
stable IDs verified, --status read fake-seeded rows correctly,
--retry-failed dropped + wiped, lock blocked a live PID and reclaimed
a dead one, `latest` symlink resolution worked. The plan banner now
shows e.g. `[plan] 23 cells total | 7 ok | 1 failed-skipped |
15 to run this session | ETA ~76.4 min`.

This means a full sweep can be done in fits and starts: kick it off,
walk away, check `--status` later, top-up with `--max-wall-secs 1800`
between meetings, retry whatever failed once the box is calmer. The
contract applies to future drivers too (ANE rail proof, packed-buffer
ablation) — see the runbook.

## [2026-05-22] lab | smoke caught two real bugs before the full sweep

Once BAB1 cleared the box and a real-worker smoke became possible,
two things broke that the dry-run resumability smoke couldn't have
seen:

- **Pre-fused checkpoint vs un-fused load path.** `stage_checkpoint`
  was calling `fuse_model_for_inference` before `save_checkpoint`.
  The worker's `_load_model` builds a fresh un-fused `GomokuNet` and
  calls `load_state_dict`, which then rejects the fused state_dict
  (extra `tower.*.conv*.bias`, missing `tower.*.bn*.running_mean`,
  etc.). Workers crashed on first load. Fix: stage un-fused; workers
  fuse internally after load (`selfplay_worker.py:198`,
  `:632`, `:749`).
- **8x throughput double-count.** `selfplay_worker` writes
  `n_examples = len(record.examples)`, but the examples list is
  already D4-augmented (8 entries per raw ply). My driver was
  computing `aug_pos_per_sec = total_n_examples * 8 / wall_secs`
  — off by 8x. Tiny W8 G8 S400 V64 first reported 56,735 aug/s
  (impossible for tiny on M5 Max — small ref is 2,379). Fix:
  track `total_aug_examples` and `total_raw_plies` separately;
  aug throughput is `total_aug_examples / wall_secs`;
  `plies_mean` is `total_raw_plies / total_games`.

Schema in `summary.tsv` is now `total_aug_examples` +
`total_raw_plies` (replacing the ambiguous `total_plies`). Old rows
from broken smokes were wiped — no production data lost since the
sweep had never run.

Real-worker post-fix numbers (45s/cell, fresh random weights so all
games hit `--max-plies 16`):

| cell | aug pos/s | games/s | plies_mean | games |
|---|---|---|---|---|
| tiny_W08_G08_S400_V064 | 7,135 | 55.9 | 16.0 | 2,529 |
| tiny_W01_G04_S100_V032 | 2,485 | 19.5 | 16.0 | 879   |

Calibrates against the existing baseline row for native small 8w8g
sims=400 wave=64 (~2,379 wall aug pos/s): tiny is ~3x faster than
small on the same shape, which matches a tiny-vs-small forward-pass
ratio. Sanity-passes.

## [2026-05-22] ops | BAB1 stopped early at e10247

Independently of this lab work, the `BAB1-buf-ablation-1p5M` run
stopped at e10247/10700 — neither at its 10700 cap nor on a crash
(no traceback, no NaN, trainer log just stopped advancing at
19:45 local). All workers, the trainer, the eval worker, and the
wandb sidecar were gone by the time I checked again. Likely the
other session driving BAB1 deliberately paused/killed it; the
buffer-curation arc that BAB1 belongs to is a separate workstream
([[project-buffer-curation]] memory).

This perf-log notes the state only so future readers don't assume
BAB1 ran to its written cap. The `packed-buffer-cheap-test` lane
(or its successor) is the canonical place for BAB1 interpretation.

## [2026-05-22] lab | canonical sweep launched

Box is idle, smoke is green, driver is first-class resumable, and the
user said go. Kicked the 23-cell canonical sweep in background.

- Sweep dir: `sweep_logs/canonical-sweep-20260523T015614Z/`
- Symlink:   `sweep_logs/canonical-sweep-latest`
- Driver:    `python scripts/canonical_sweep.py --out-dir
              sweep_logs/canonical-sweep-20260523T015614Z` (nohup)
- Defaults:  `--secs-per-cell 300 --max-plies 16 --device mps`
- Expected:  ~2-3 h wall (23 cells × ~5 min + per-cell setup)
- Driver log: `<sweep dir>/driver.log` (line-buffered from this commit
  forward; the in-flight run will only flush when its buffer fills, so
  use `--status` for live progress instead of tailing the log).

Recipes the user can run at any time during or after the sweep:

```bash
# Progress + ETA:
python scripts/canonical_sweep.py --out-dir latest --status

# Re-run any cells that failed (e.g. transient MPS contention):
python scripts/canonical_sweep.py --out-dir latest --retry-failed

# After the sweep finishes:
python scripts/plot_canonical_sweep.py --sweep-dir sweep_logs/canonical-sweep-latest
```

Next-session pickup once the sweep completes (or stalls):
1. `python scripts/canonical_sweep.py --out-dir latest --status` to
   confirm 23 ok / 0 pending; if not, `--retry-failed` and let it
   finish.
2. `python scripts/plot_canonical_sweep.py --sweep-dir
   sweep_logs/canonical-sweep-latest` → `contour.png`, `axes.png`,
   `model_compare.png`.
3. File a receipt in [experiment-ledger.md](experiment-ledger.md)
   under the `canonical-sweep-mainframe` lane; add per-cell-class
   rows to [baselines.md](baselines.md); promote the winning cell
   in [status.md](status.md); close the lane in
   `.frontier/lanes.json`; append a "[YYYY-MM-DD] lab | canonical
   sweep complete" entry here.

## [2026-05-23] lab | canonical sweep complete — wave_size is under-tuned

23/23 cells ok in `sweep_logs/canonical-sweep-20260523T015614Z` (also
at `canonical-sweep-latest`), median 300.5s/cell, 0 failed. All games
hit `--max-plies 16` (random weights → no learned defense → universal
cap), so cell numbers are **infrastructure throughput**, not behavior
throughput. Trained-model production cycles will be slower in
absolute aug/s; the relative axis shape should hold.

### Axis-by-axis results

| Axis (other params at default) | Cells | Headline |
|---|---|---|
| **workers** (small G=8 S=400 V=64) | W1=1,111 → W2=1,497 → W4=2,583 → **W8=3,188** → W12=3,243 → W16=3,411 aug/s | Diminishing returns; W=8 is near-optimal. Per-worker eff falls from 1,111 at W=1 to 213 at W=16; MPS contention dominates by W=2. |
| **n-simulations** (small W=8 G=8 V=64) | S100=11,151 / S200=6,006 / **S400=3,188** / S800=1,619 | Perfectly inverse: aug/s × sims ≈ const. Pure quality knob. |
| **wave-size** (small W=8 G=8 S=400) | V32=2,467 / **V64=3,188** / V128=4,048 / V256=4,409 aug/s | **+27% at V128, +38% at V256 over the WL5 default V64.** No behavior change, just bigger eval batches. |
| **games-per-worker** (small W=8 S=400 V=64) | G4=3,026 / **G8=3,188** / G16=3,057 | Flat. G=8 default is fine. |
| **model** (W=8 G=8 S=400 V=64) | tiny=7,326 / **small=3,188** / medium=1,393 | ≈2.3× per step. Forward pass dominates. |
| **max corner** | tiny W16 G16 S100 V32 | 19,346 aug/s — infrastructure ceiling, not quality-comparable. |
| **min corner** | small W1 G16 S800 V128 | 946 aug/s — single fat worker. |

### Promoted default

**Old throughput default:** small / W=8 / G=8 / sims=400 / **wave=64** → 3,188 aug pos/s
**New throughput default:** small / W=8 / G=8 / sims=400 / **wave=128** → 4,048 aug pos/s (+27%)

Wave=256 is also viable (+38%). Chose V=128 as the safer step: V=256
is past the inflection point and may interact poorly with MPS heap
sizing under sustained training pressure (none of these cells used
the trainer; production cells should canary the V=128 candidate
first per the Training-Quality Promotion Gate).

The wave-size win is the single most actionable result from this
sweep. It is exactly the kind of chip-specific calibration the
[m5-max-as-mainframe](../topics/m5-max-as-mainframe.md) page predicted
we'd find by sweeping the production shape on this exact SKU
instead of transplanting CUDA recipes.

### Caveats and follow-ups

- **All cells hit plies_mean=15.96.** Random weights + max-plies=16
  meant every game terminated at the cap, so absolute throughput
  numbers reflect infrastructure (eval batching, worker spawn, file
  handoff) more than realistic game shape. Wave-size win is
  eval-batch-shape-dependent (not game-shape-dependent), so it
  should transfer; worker-axis numbers may shift somewhat with real
  plies.
- **W × G cross was not run.** The workers axis fixed G=8 and the
  games-per-worker axis fixed W=8. Re-running the cross at V=128
  would verify whether the wave win compounds at higher worker
  counts.
- **Sims-vs-wave interaction** is unexplored. S=200 with V=128 or
  V=256 might be the real next-cell shape if quality holds at lower
  sims.
- **Trained-checkpoint re-sweep.** Repeat once a stable post-WL5
  trained checkpoint exists to confirm trained-model throughput
  shape matches infrastructure shape.

### Surfaces updated

- Receipt: [experiment-ledger.md](experiment-ledger.md) "2026-05-23
  — canonical 5-axis M5 Max contour sweep".
- Baseline rows: [baselines.md](baselines.md) (7 new rows; wave-size,
  workers axis, model axis, max corner).
- Status: [status.md](status.md) Current Focus + lane row.
- Frontier: [frontier.md](frontier.md) + `.frontier/lanes.json`
  (lane completed/done).

### Suggested next lanes for the lab

1. **W × G cross at V=128** (small focus): ~12 cells × 300s = 1 h.
   Confirms the wave win compounds.
2. **Sims-vs-wave interaction**: small W=8 G=8 over
   S ∈ {100, 200, 400} × V ∈ {64, 128, 256} = 9 cells; ~45 min.
3. **Trained-checkpoint re-sweep** once post-WL5 training stabilizes
   on a strong checkpoint. Same 23 cells, swap the staged random
   weights for the trained ones.
4. **ANE rail-proof unblocker** — still gated on passwordless sudo
   for `powermetrics`. Independent of this sweep.
5. **Engine-overlap experiment** — unblocks once ANE rail is real.
   The wave=128 throughput default is the right MPS-side baseline
   for that experiment.
