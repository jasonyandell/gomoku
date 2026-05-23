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

## [2026-05-23] L11 | R-TRAIN-LEAN V=512 REJECT — gen wins don't compound at trainer

L11 tested whether V=64→V=512's pure-gen +49.5% promote (R-S400, L01) carries through to the holistic R-TRAIN-* family. Same WL5 recipe as L10 but `--wave-size 512`. Same 30s warmup + 120s measure.

**Result: reject.** Every metric got worse:

| Metric | L10 V=64 | L11 V=512 | Δ |
|---|---|---|---|
| aug/s | 3,297.6 | 2,362.8 | **-28.4%** |
| games/s | 14.07 | 8.42 | -40.2% |
| epochs/s | 0.0917 | 0.0083 | -91% |
| trainer_step_s_p50 | 0.051s | 0.138s | +2.7× |
| epochs in window | 14 | 3 | — |
| plies_mean | 29.6 | 34.3 | — |

**Mechanism (clean in the trainer log).** At V=512 workers fill the buffer 2.4× faster (buf=199,608 at epoch 3 vs 83,208 at V=64 epoch 3). The trainer's `--sgd-per-position 0.0025` is a fixed ratio, so 2.4× positions = 3.36× SGD steps (epoch 3 ran 306 steps vs 91 at V=64). The per-epoch tail in the trainer log went from `(11s: gen=7s train=3s)` at V=64 to `(52s: gen=6s train=43s)` at V=512. While the trainer monopolizes MPS for 43s of SGD, workers get less GPU time — games/s collapses, aug/s collapses with it.

This is the holistic R-TRAIN-* family working as intended. The L11 yaml's own caveat called it: *"If it doesn't, R-S* metrics need humility — gen throughput isn't the whole story."* Confirmed.

**Lab implications:**
- V=64 stays the R-TRAIN-WL5 default. WL5 production recipe is correct as-is.
- The R-S* V=512 promotes remain valid for *non-trainer* self-play (eval probes, validation rolls, dataset mining). They do NOT free-ride to live training.
- Follow-up candidate L11b: would lowering `--sgd-per-position` at V=512 (to match V=64's SGD work per second) let the gen win shine? Lower priority than L09 — the headline finding here (gen wins don't free-ride) is already the load-bearing insight.

Next: dispatch L09 (R-TRAIN-ANE via Core ML eval on workers) — the architectural ANE-offload lever. First needs a small L12 driver patch to pass `--evaluator coreml --coreml-compute-units CPU_AND_NE` through to workers.

> Pure-gen wins were the easy half. The trainer is fighting for the same chip, and at V=512 it wins the fight — which is exactly the wrong fight to win.
> Now we know: any future "what if we crank V higher" idea has to be paired with a sgd-per-position cut, or it costs us at the level that matters.

---

## [2026-05-23] L10 | R-TRAIN-WL5 baselined at 3,297 aug/s; trainer contention ≈ 30%

First-ever R-TRAIN-WL5 measurement. End-to-end production recipe (small / W=8 / G=8 / sims=400 / V=64 / EMA τ=0.99 / grad_accum=4) under the live trainer + 8 self-play workers competing for MPS. 120s measurement window, 14 epochs:

- **aug_pos_per_sec: 3,297.6** (vs R-S400 pure-gen 4,765 → trainer contention costs ~30.8% on generator throughput)
- **games_per_sec: 14.07** (vs ~17.7 implied by R-S400's 4,765 / 269 aug-per-game)
- **epochs_per_sec: 0.0917** (~10.9s wall per epoch in steady state — 50 SGD steps at trainer_step_s_p50=0.051s = 2.56s training plus ~5-6s of barrier-wait for fresh self-play)
- **trainer_step_s_p50: 0.0512s** (per SGD step; the trainer is GPU-bound here, not blocked on data)

The autonomous lab restart hit two L12 driver bugs in flight, both surfaced and patched:

1. **`--save-every=1000000` froze worker_weights.pt**. `gomoku/train.py:1220` publishes the worker-facing weights file inside the save-every block; with save-every set high "to disable mid-run checkpoint IO", workers stayed on v0 forever and the trainer hung waiting for v1+ games. Fix: `--save-every=1 --keep-last-n=1` (small per-epoch ~4MB writes, auto-pruned; the 1.4GB latest.pt still gated by save-buffer-every=1M). Commit `1dc4abb`.
2. **`count_records()` at SIGTERM undercounted by ~30×**. The trainer ingests + deletes worker `game*.pt` files as it goes, so the end-of-window file count was ~80 games / 16k aug-positions where the trainer log's cumulative `games=` counter showed ~1,500 games / ~350k positions. Fix: parse the trainer's epoch line directly (cumulative `games=N`, `buf=N`, per-epoch wall `(Xs:`) and prefer those over file counts. Commit `4a825f1`.

Both bugs were invisible to L12's `--dry-run` and synthetic-log smoke tests — only the real workload exposed them. Receipt under L10-trainer-step-bench in the ledger.

Next: dispatch L11 (R-TRAIN-LEAN at V=512) to test whether V=64→V=512's +49.5% gen win compounds at the trainer level. Then L09 (R-TRAIN-ANE via Core ML eval on the workers).

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

## [2026-05-23] lab | L01 wave extrapolation — V=512 is the plateau knee, +49.5% cumulative on R-S400

First lab-dispatched lane under the charter. 4 cells × 5 min, all ok.

| cell | aug/s | vs V=128 | vs WL5 V=64 |
|---|---|---|---|
| V=384  | 4,452 | +10.0% | +39.6% |
| **V=512**  | **4,765** | **+17.7%** | **+49.5%** |
| V=768  | 4,761 | +17.6% | +49.4% |
| V=1024 | 4,756 | +17.5% | +49.2% |

V=512 is the plateau knee. V=768/1024 are flat — eval overhead caps
further wave gains on this exact hardware. The lab will stop sweeping
V > 512 unless something else (model size, MPS heap config, ANE
engine) shifts the eval-overhead floor (L07 tiny contour will check
the model-size dependency).

**Promotion: small / W=8 / G=8 / sims=400 / V=128 → V=512** at R-S400.
Pending Reviewer signoff per
[perf-lab-reviewer-role](../topics/perf-lab-reviewer-role.md).
+17.7% over yesterday's V=128 promote. +49.5% cumulative since the
WL5 production V=64. No behavior change; eval batch shape only.

**Auto-queued compounds** (per the charter's Tier-1-after-promote
discipline): L02 (W × V=512: W ∈ {4,12,16}), L03 (S × V=512: S ∈
{100,200}). Both rescoped to drop V=128/V=256 cells that L01 now
dominates. L02's E[delta] dropped from 800 to 400 aug/s and P from
0.6 to 0.5 since the workers axis was already shown to be near-flat
past W=8 in the canonical sweep — most likely outcome is "V=512 holds
across W". L03 stayed high-priority because S × wave was the most
under-explored cross in the canonical sweep.

Process notes:
- L01 was originally Tier-1 in the day-1 queue; the charter v2 tier
  refactor demoted it to Tier-3 (single-axis speculation past a known
  win). The run completed before the refactor landed; archived as
  Tier-3 in retrospect.
- Reviewer Gate is on for all subsequent receipts. L01's receipt has
  `reviewer: PENDING`; a Reviewer spawn will audit and the verdict
  appended before the next commit closing the receipt.

## [2026-05-23] lab | L03 sims-x-wave — V=512 carries to every quality point (R-S200 +52.5%, R-S100 +35.2%)

Cron tick caught L03 just-completed. 2/2 ok, ~5 min each.

| cell | aug/s | vs WL5 V=64 |
|---|---|---|
| small W=8 G=8 S=100 **V=512** | **15,082** | **+35.2%** over 11,151 |
| small W=8 G=8 S=200 **V=512** |  **9,156** | **+52.5%** over 6,006  |

**Double promote** — V=512 now wins at every R-S* reference point measured. The wave-size lever is uniform across the sims axis: same speedup mechanism (eval-batch shape) applies regardless of how many MCTS sims feed the batch.

Three reference points are now at V=512:

| ref | best aug/s | speedup vs WL5 V=64 |
|---|---|---|
| R-S400 | 4,765  | +49.5% |
| R-S200 | 9,156  | +52.5% |
| R-S100 | 15,082 | +35.2% |

R-S200 has the biggest gain because S=200 V=64 was particularly under-saturated on eval (wave was too narrow vs the per-call kernel cost). At S=100 the gain shrinks because games-per-batch × wave already saturated MPS at smaller batch sizes; at S=400 the gain is between because each sim contributes more wall-time but fewer eval calls.

The cron's Speedup Report line is now load-bearing. Reviewer: APPROVE — "L03 double promote math + units verified; all six surfaces consistent; queue clean."

## [2026-05-23] lab | L02 W-x-wave reject — W-axis INVERTS at V=512

3/3 cells ok. No promote — and the absence is itself a finding.

| cell | aug/s | vs W=8 V=512 ref (4,765) |
|---|---|---|
| W=4  V=512 | 4,367 | -8.4% |
| **W=8 V=512** | **4,765** | reference (L01) |
| W=12 V=512 | 4,501 | -5.5% |
| W=16 V=512 | 4,504 | -5.5% |

At V=64 the canonical sweep had W=16 as the peak (3,411 vs W=8 3,188 = +7%). At V=512 the peak is W=8, and W=12/W=16 are slightly worse. The wave-saturation pressure shifted the MPS-dispatch sweet spot.

**Implication**: knob wins don't just fail to compound — they actively interact in non-monotone ways at the chip's high end. The tier system's "no leapfrogging" rule is more than aesthetic; it's about not assuming linear combinatorics. Future cells should always re-measure the W axis when V changes substantially, not extrapolate.

Auto-queue updates (in `wiki/ops/perf-queue.md`):
- L04 G-x-wave bumped from priority 1.4 → 9.0 (G might also be non-monotone at V=512; was flat at V=64).
- L07 tiny-contour bumped from priority 12 → 36.4 (added V=512 + V=1024 cells; tiny model may extend the wave plateau further because forward pass is cheaper).

consecutive_rejects: 0 → 1.
Reviewer: APPROVE — "L02 reject math clean (-8.4%/-5.5%/-5.5%); best-cells correctly unchanged; W-inversion insight requeues L04+L07; counter 0→1."

## [2026-05-23] lab | L04 G-x-wave reject — G=8 stays optimal (compound finding with L02)

3/3 cells ok.

| cell | aug/s | vs G=8 V=512 ref (4,765) |
|---|---|---|
| G=4  V=512 | 4,608 | -3.3% |
| **G=8 V=512** | **4,765** | reference |
| G=16 V=512 | 4,541 | -4.7% |
| G=32 V=512 | 4,514 | -5.3% |

G axis IS mildly non-monotone at V=512 (was completely flat at V=64: 3026/3188/3057). But the peak is still G=8 — same shape as L02's W-axis result.

**Compound finding with L02 — sharper than either alone**: at V=512 BOTH the workers axis AND the games-per-worker axis peak at the canonical-sweep production defaults (W=8, G=8). Wave-saturation has tightened the production-cell envelope around the historical defaults. Wider perimeter exploration at V=512 won't beat the center.

**Practical implication**: future single-axis explorations at V=512 should not bother re-measuring W or G — those axes are CONFIRMED at their peaks. Open axes for further exploration: model size (L07 tiny), n-sims at V=512 (L03 done), architectural (L09 ANE), engine-isolation (L05/L06 worktrees).

Followup:
- L08 (MPS heap ratio) marked blocked-on-driver. canonical_sweep doesn't support per-cell env vars; cells.csv schema needs extension. Add to L12 scope or carve out an L08-driver task.
- Next dispatch: L07 tiny contour (bg priority 36.4 after the L02 bump). The strict tier rule says Tier-3 before bg, but L05/L06/L08 are all blocked-on-code-work, so L07 is the only unblocked lane.

consecutive_rejects: 1 → 2. One more reject would still NOT halt the loop (the stop rule requires `consecutive_rejects ≥ 3 AND queue empty AND no compound follow-ups`; queue is not empty).

Reviewer: APPROVE — "L04 reject math clean (-3.3/-4.7/-5.3%); best-cells unchanged; compound W+G finding documented; L08 correctly blocked; counter 1→2."

## [2026-05-23] lab | L07 tiny contour — R-S400-tiny promote +201.5%; W peak is model-dependent at V=512

6/6 cells ok. The lab's biggest single-lane jump so far.

| cell | aug/s | vs tiny V=64=7,326 |
|---|---|---|
| tiny W=8  V=128 |  9,407 | +28.4% |
| tiny W=8  V=256 | 14,461 | +97.4% |
| tiny W=8  V=512 | 17,088 | +133.2% |
| tiny W=8  V=1024| 17,012 | flat with V=512 (same plateau as small) |
| tiny W=16 V=256 | 16,375 | +123.5% |
| **tiny W=16 V=512** | **22,088** | **+201.5%** ← new R-S400-tiny best |

V=512 plateau holds for tiny (V=1024 flat). But the **model-dependent W peak** is the headline:

| model | best W at V=512 | second-best W |
|---|---|---|
| small (L01/L02)  | **W=8** = 4,765 | W=16 = 4,504 (-5.5%) |
| tiny  (L07)      | **W=16** = 22,088 | W=8 = 17,088 (-22.7%) |

At small, eval cost per worker is high enough that 8 workers saturate MPS dispatch. At tiny, eval cost is ~3× cheaper so MPS can stay fed with 16 workers — the saturation point shifted right.

**Direct implication for L09 ANE-offload**: with workers on Core ML (CPU/ANE), the effective per-worker eval cost changes again. Whether W=8, W=16, or W=24+ is the peak under the ANE workload is unknown a priori — L09's measurement cells should test BOTH W=8 and W=16 at V=512, not just one. Added this note to the L09 queue entry.

**Auto-queued follow-ups** (both bg, both new):
- L13 (priority 58.8 — highest in current queue): probe tiny peak finer at W ∈ {12, 16, 20, 24}. If W=20 or W=24 beats W=16, even bigger gain available.
- L14 (priority 16.5): G axis at tiny W=16 V=512.

consecutive_rejects: 2 → 0 (any promote resets per stop rule).

Reviewer: APPROVE — "L07 promote math clean (+201.5%); 2-axis move decomposed via cell matrix; surfaces consistent; L13/L14 well-scoped."

## [2026-05-23] lab | L13 tiny W-peak probe reject — W=16 confirmed; tolerance band W∈[12,20] within 7%

3/3 cells ok. Fine-grained peak confirmation.

| cell | aug/s | vs W=16 V=512 ref (22,088) |
|---|---|---|
| W=12 V=512 | 20,560 | -6.9% |
| **W=16 V=512** | **22,088** | reference (L07) |
| W=20 V=512 | 21,553 | -2.4% |
| W=24 V=512 | 20,970 | -5.1% |

W=16 is confirmed the tiny V=512 peak, with W=20 a very close second (within 2.4%). The whole W ∈ [12, 20] band is within 7% of peak — tiny's W-axis at V=512 is a smooth bump, not a sharp saturation drop.

**Compound finding (L02 + L07 + L13)** — model size determines BOTH the W-peak location AND the tolerance shape at V=512:
- **small**: peak W=8, sharp drop (W=16=-5.5%, W=4=-8.4%, narrow tolerance)
- **tiny**: peak W=16, gentle bump (W=12=-6.9%, W=20=-2.4%, W=24=-5.1%, wide tolerance)

Direct implication: L09 ANE-offload worker tuning has more wiggle room with tiny than the small data suggested. The optimal under Core ML/ANE is probably also in the W ∈ [12, 20] band rather than a single sharp peak.

consecutive_rejects: 0 → 1.
Next dispatch: L14 (G axis at tiny W=16 V=512).

Reviewer: APPROVE — "L13 reject clean: math/plies/units verified, W=16 confirmed peak, surfaces consistent, no spurious follow-ups."

## [2026-05-23] lab | L14 G axis flat — knob-tuning exhausted at chip envelope

3/3 cells ok. G axis at tiny W=16 V=512 is essentially flat:

| cell | aug/s | vs G=8 ref (22,088) |
|---|---|---|
| G=4  V=512 | 22,261 | +0.78% |
| G=8  V=512 | 22,088 | reference |
| G=16 V=512 | 22,164 | +0.34% |
| G=32 V=512 | 22,076 | -0.06% |

Total spread 0.83% — within unmeasured run-to-run noise. G=4 nominal lead of +0.78% is not a defensible promote.

**The headline finding across L02 + L04 + L13 + L14 is now decisive: at V=512 (the new structural default), single-axis knob exploration of W and G has been exhausted for both small and tiny models.** No further knob tweaks within the {W ∈ [4, 24]} × {G ∈ [4, 32]} envelope produce a promote. The wave-size lever was the regime-changing knob; everything else is fine-tuning noise relative to it.

**Cumulative lab state**:

| reference | best cell | best aug/s | cumulative speedup |
|---|---|---|---|
| R-S400 | small W=8 G=8 V=512 | 4,765 | +49.5% |
| R-S200 | small W=8 G=8 S=200 V=512 | 9,156 | +52.5% |
| R-S100 | small W=8 G=8 S=100 V=512 | 15,082 | +35.2% |
| R-S400-tiny | tiny W=16 G=8 V=512 | 22,088 | +201.5% |

**Remaining headroom is structural, not knob**:
- L09 ANE-offload (blocked on L12)
- L05 torch.compile (worktree code)
- L06 fp16 (worktree code)
- L08 heap ratio (per-cell env var driver work)
- L12 live-training cell driver (Tier 1 gating)
- L10 R-TRAIN-WL5 baseline (blocked on L12)
- L11 R-TRAIN-LEAN end-to-end (blocked on L12)

All require human-session code work. Cron is at a natural pause point. PushNotification sent.

consecutive_rejects: 1 → 2.

Reviewer: APPROVE — "L14 reject correct — G axis spread 0.83% within noise; surfaces consistent; pause state cleanly logged."

## [2026-05-23] lab | charter v2 — tier system + R-TRAIN family + Reviewer Gate

After L01 launched but before it landed, Jason gave four new
directives that reshape the lab:

1. **Live training cells allowed** — ≤ 5 min/cell, multi-cell stitch
   for warmup + measure. Opens the R-TRAIN-* metric family
   (epochs/sec under live trainer). This is the holistic metric
   that matters for elo gain, not just isolated self-play.
2. **Reviewer role** — codified in
   [perf-lab-reviewer-role](../topics/perf-lab-reviewer-role.md).
   Spawned per lane + every ~5 lanes for discipline audit. APPROVE
   / REVISE / BLOCK. No promote without signoff.
3. **/loop 10m check-in** — periodic auto-tick: read queue, file
   receipts, dispatch next-priority lane.
4. **ANE / engine isolation > batch sizes** — explicit tier rule.
   Architectural lanes (L09 ANE-offload, L10 trainer bench, L11
   end-to-end) can't be leapfrogged by knob lanes on raw priority
   alone.

Charter v2 committed in `7491401`. Queue reranked.
