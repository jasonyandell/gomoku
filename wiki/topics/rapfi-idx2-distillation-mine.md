# Rapfi idx-2 distillation mine — the "Bruce Lee one-position" experiment

**Status: IN PROGRESS (2026-06-25).** Mine DONE — **1,126,597 canonical idx-2
positions on disk** (≥1M goal met, crash-robust). Pretrain running as a 12-epoch
warm-start seed; AlphaZero warm-start + Rapfi-probe loop wired into the autonomous
driver (`mined/drive_pipeline.sh`). This page is the synthesis; raw run detail
goes to `TRAINING_WIKI.md`.

## Hypothesis (Jason's framing)

> "I fear not the man who has practiced 10,000 kicks once, but the man who has
> practiced one kick 10,000 times."

Master **one** 15×15 opening — **idx-2** — and *nothing else* (only its D4
symmetries). The 15×15 champion era plateaued at ~50 Δelo and the binding wound
is the **white-defense gap** (`eval502` vs native Rapfi-NNUE @idx-2: black 42%,
**white 0/12**; see [white-side-defense-plan.md](white-side-defense-plan.md) and
the index reckoning). Instead of another defense knob on a generalist, this asks:
if we **deliberately over-specialize** — pretrain a net on Rapfi's own play of
the idx-2 tree, then run standard AlphaZero self-play from idx-2 only — can the
net stand against Rapfi *in that one position*?

Over-specialization is the point, not a risk: at eval/inference the only position
that ever occurs is idx-2, so every parameter spent elsewhere is waste. Coverage
is not a virtue here; **depth on one position is**.

## The tool — `gomoku/rapfimine/`

A reusable, crash-robust, max-throughput harness to mine `(position, Rapfi soft
policy+value)` pairs by BFS over a fixed opening. Built because the in-process
`RapfiPool` could not feed many engines (see Throughput below).

- **`canonical.py`** — D4-canonical state hashing. Stores ONE representative per
  symmetry class; the trainer's existing sample-time D4 augment recovers the 8×.
  `transform_state` carries board AND all `HISTORY_PLY` frames under one symmetry
  so every plane stays in a consistent frame. `canonical_key` = blake2b-16(canon
  board + move_count).
- **`worker.py`** — one OS process = one Rapfi engine in a tight analyze loop,
  with its OWN GIL and its OWN shard stream (planes never cross IPC). Respawns its
  engine on death; idle-exits if the coordinator vanishes.
- **`store.py`** — append-only sharded npz (teacher **v2** format: `planes`,
  `soft_policy`, `moves`, + a `keys` array for resume), atomic temp+`os.replace`.
  Frontier checkpoint = the pending-work half of the durable log (immutable
  snapshot, newest wins).
- **`coordinator.py`** — BFS frontier (deque, breadth) + global canonical dedup
  (gate at enqueue) + dispatch/collect + live monitor (count, inst/mean moves/s,
  Rapfi CPU% vs cores). Resume rebuilds the seen-set from shard keys and restores
  the frontier checkpoint (filtered against seen); in-flight boards are tracked by
  id and re-queued. A SIGKILL loses at most the branches found since the last 60 s
  checkpoint — **never a completed example**.
- **`pretrain.py`** — supervised distillation of the mined set into a standard
  checkpoint (`build_model`/`save_checkpoint` + the trainer's `policy_loss`/
  `value_loss`; no reinvention). Policy target = masked temperature-softmax of
  Rapfi's winrate map; value target = `2·best_winrate−1` (auxiliary — policy
  carries the load, #18/#44).

CLI:
```bash
GOMOKU_BOARD_SIZE=15 uv run python -m gomoku.rapfimine run \
    --out mined/idx2_15x15 --total 1200000 --workers 24 --max-node 5000
GOMOKU_BOARD_SIZE=15 uv run python -m gomoku.rapfimine status --out mined/idx2_15x15
GOMOKU_BOARD_SIZE=15 uv run python -m gomoku.rapfimine.pretrain \
    --shards mined/idx2_15x15 --out checkpoints/idx2_pretrain.pt --size large
```

## Two correctness/perf fixes this surfaced (both committed, #86)

1. **Rapfi multiPV crash (correctness).** `_read_analysis` capped chatter at a
   shared `_MAX_CHATTER_LINES=2000`. A forced-mate position (`EVAL -M4`) makes
   Rapfi deepen to DEPTH 26, re-emitting all PV blocks each round → **2302 lines,
   terminator and all** — past the cap, so it raised "no bestmove terminator" and
   crashed the mine at ~depth-5 tactical boards (the long-undiagnosed "200k crash"
   from the 2026-06-24 handoffs). Fix: a **pv-scaled** cap for the analysis loop;
   the quiet belt stays for the single-reply protocols.
2. **One reader thread per engine, not per line (throughput).** `_read_line`
   spawned a fresh `threading.Thread` for EVERY stdout line to enforce its
   timeout. At ~2000 lines/analyze that was ~50 ms of pure thread churn **serial
   with** (and starving) the engine. Fix: one persistent daemon reader thread per
   engine draining stdout into a queue; `_read_line` is now a cheap `queue.get`.
   Per-analyze on a deep board dropped **68 ms → 17 ms**.

## Throughput (the goal metric: moves/sec) — measured on the M5 Max (18 cores)

| Config | examples/s | machine | note |
|---|---|---|---|
| in-process `RapfiPool(60)` | ~75–150 | ~9% | GIL-serial stdio feed starves engines |
| harness, 24 workers, max_node 20000, pre-reader-fix | ~170 | ~28% | thread-per-line overhead |
| + reader-thread fix | ~480 | ~75% | engines genuinely busy |
| + **max_node 5000** (BFS) | **~700** | ~75% | 6× cheaper/board, SAME top-1 move, ~21 scored |

`max_node` sweep (per-board, with top-1 agreement vs the 20000 reference):
20000 = 118 ms; **5000 = 18 ms, top-1 SAME, ~21 moves scored**; 2000 = 10 ms but
only 8 scored and top-1 disagrees midgame (too shallow). **5000 is the
quality-preserving sweet spot.** 1.2M positions ≈ ~28 min.

## A third fix — pretrain per-epoch sync (perf, #86)

The pretrain loop called `float(pl.detach())` **every step** to accumulate the
running loss. On MPS each `float()` forces a device→host sync that flushes the
command buffer and serializes the GPU — the host sat at **~7% CPU** blocked on
syncs while a 432 s/epoch run looked "GPU-bound" but wasn't fully pipelined. Fix:
accumulate `pl_acc/vl_acc` as on-device scalars and sync **once per epoch**.
(Mirrors the mine's lesson: measure, find the host-side stall, remove it.)

## Pipeline — autonomous driver (`mined/drive_pipeline.sh`)

Sequential by design (each stage wants the machine): the driver waits for the
mine to finish, then pretrains, gates, warm-starts AZ, and probes. Re-runnable
from scratch (mine-wait is a no-op once ≥1M is on disk) = crash-robust.

1. **Mine** idx-2 to ≥1M canonical positions (`rapfimine run`, max_node 5000). ✅
2. **Pretrain** a `large` (128×10, "Bruce" size) net on the soft policy+value —
   **12 epochs** (a warm-START seed only needs Rapfi's idx-2 move preferences;
   AZ self-play continues training). Gate: H2H vs Rapfi @idx-2 on the seed.
3. **Warm-start AlphaZero**: the `G15-idx2-warmstart` cell (byte-identical to
   `G15-fixed-openings`, own run-dir + wandb run) launched via
   `run_sweep --cell G15-idx2-warmstart --resume checkpoints/idx2_pretrain.pt`
   with `GOMOKU_DROP_OPENERS=0,1,3,4,5,6,7,8` so self-play sees **only idx-2**
   (`book[2] == ((3,2),(5,4),(4,5))`; D4 recovered by the trainer's augment).
4. **Probe**: every 30 min the driver runs `rapfimine.eval_idx2` on the live
   `latest.pt` vs native Rapfi-NNUE @idx-2 (n=48), logging one strength-vs-time
   curve to `mined/az_vs_rapfi.log` — both colors separately (white is the real
   bar; see [white-side-defense-plan.md](white-side-defense-plan.md)).

## Results so far (2026-06-25)

- **Dataset:** 1,126,597 canonical idx-2 positions (577 shards, 8.6 GB f16),
  ~28 min to mine at ~700/s on the M5 Max.
- **Pretrain is GPU-bound, not host-bound.** Epoch = ~437 s at batch 1024 / 1100
  steps (`large` net). The per-step-sync fix did NOT change wall-clock (436 s vs
  432 s) — the host was already just waiting on the GPU; the net's forward+backward
  IS the cost. The only real lever is fewer total steps, so the warm-start uses an
  early checkpoint, not all 12 epochs.
- **Pretrain CE curve (the seed learning Rapfi's idx-2 policy):**
  epoch 1 ce=2.89 / vmse=0.173 → ep2 2.17 / 0.110 → **ep3 2.04 / 0.097** (flattening;
  banked the epoch-3 checkpoint as the warm-start seed). vs ~5.4 uniform.
- **Seed H2H vs Rapfi @idx-2 (n=48, MCTS-160):** **0/48, both colors 0%.** The raw
  distilled seed does NOT stand a chance against full Rapfi-NNUE — expected, and the
  baseline to beat. This is the experiment's whole question: can AZ self-play *from
  this seed* close the gap (esp. the white-defense wound)?
- **Warm-start AlphaZero: RUNNING** (`G15-idx2-warmstart`, resumed from the seed at
  epoch 4 → pretrained weights confirmed loaded; 3 self-play workers; idx-2 only via
  `GOMOKU_DROP_OPENERS`). First trained epoch (18): games=8 buf=1584 pl=3.21 vl=0.39
  plies=27.8 — genuinely training on its own idx-2 games.
- **Warm-started self-play H2H vs Rapfi @idx-2, by color over training** (verdict
  curve, `mined/az_vs_rapfi.log`, accruing every 30 min):
  - epoch ~70 (~35 min of self-play): **0/48, BLACK 0% / WHITE 0%** — still the
    seed's baseline; ~70 epochs from a weak distilled seed is nowhere near enough
    to challenge mature Rapfi-NNUE.
  - **Learning trajectory (leading indicator):** self-play policy loss rises to a
    peak (~4.2 @ ep36, re-fitting from the Rapfi-softmax seed to MCTS-visit targets)
    then falls steadily — 2.67 @ep57 → 1.89 @ep103 → **1.53 @ep125**; value_mse
    0.39→0.16. The net IS improving; H2H gains lag net strength. Rate ≈ **27 s/epoch
    (~133 epochs/hr)** on the M5 Max with the 3-worker cell. For scale, Bruce's
    black-42% bar came after ~3,700 epochs — so a meaningful idx-2 verdict is a
    **multi-hour-to-multi-day climb**, measured unattended by the 60-min probe loop.
  - **Reference bar:** Bruce (the generalist, ~3700 epochs / 33 h) reached only
    black ~42% / **white 0/12** vs Rapfi @idx-2 — the white-defense wall is the
    known-hard part. The over-specialization bet is whether idx-2-only self-play
    from a Rapfi-distilled seed can beat that, esp. crack white. This is a
    multi-hour-to-multi-day climb; the probe loop measures it unattended.

## Eval GRADIENT — measuring progress below the max-Rapfi wall

Max-strength Rapfi-NNUE is a wall: a climbing net reads **0/48 for hours** before
denting it, so it can't show progress. `gomoku/rapfimine/eval_gradient.py` plays
the net @idx-2 (white split) against **native Rapfi graded purely by per-move
think-time** so improvement is visible as it clears rungs:

> rapfi@25ms < rapfi@50ms < rapfi@100ms < rapfi@250ms < rapfi@1000ms

One clean strength dial (Rapfi's own engine, just less time), nothing invented.
The classical baselines were dropped: at epoch ~145 the net already **crushes
random / heuristic / lookahead-d2 at 100% (both colors)** (saturated), and
lookahead-d4 is resource-heavy (negamax on 15×15). The 40× think-time spread
localizes where the net sits and tracks its climb toward the 1000 ms bar (≈ the
max-strength Rapfi the plain idx-2 gate reads 0 against).

Driven by `mined/gradient_loop.sh` (detached, every 45 min on the newest
`epoch*.pt`) → one `GRADIENT` line per pass in `mined/az_gradient.log`. In-session
a Monitor tails that log so new rungs surface without polling.

**Gotcha — snapshot before eval.** The trainer keeps only `--keep-last-n 3`, so a
checkpoint rotates out every ~7 min while a gradient eval takes ~5–8 min — picking
`ls -t … | head -1` and evaluating it directly races and dies `FileNotFoundError`.
The loop first `cp`s the checkpoint to a stable epoch-preserving name
(`mined/_snap_epochNNNN.pt`), evals that, then removes it.

## Crash recovery / resume (durability)

The run is crash-robust by checkpoint, not by a supervisor — any fresh session
reading this is the recovery loop. If the AZ trainer dies, relaunch resuming the
run-dir's `latest.pt` (weights **+** the 1.4 GB buffer) when it exists, else the
newest `epochNNNN.pt` (weights only; workers refill the buffer). NOT the pretrain
seed — that discards self-play progress. **Note:** `latest.pt` is only written
every `save_buffer_every=100` epochs, so for the first ~100 epochs only
`epochNNNN.pt` exists (this also drives the probe target, below).

```bash
cd <this worktree>; export GOMOKU_BOARD_SIZE=15
pkill -f 'sweep_runs/G15-idx2-warmstart-board15/' ; sleep 2   # clear orphan workers
CK=sweep_runs/G15-idx2-warmstart-board15/checkpoints
RESUME=$CK/latest.pt; [ -f "$RESUME" ] || RESUME=$(ls -t $CK/epoch*.pt | head -1)
GOMOKU_DROP_OPENERS=0,1,3,4,5,6,7,8 \
  uv run python scripts/run_sweep.py --cell G15-idx2-warmstart --resume "$RESUME"
```

**Probe loop** (`mined/probe_loop.sh`, detached): every 30 min H2H's the **newest
`epochNNNN.pt`** vs Rapfi @idx-2 (n=48) into `mined/az_vs_rapfi.log`. It targets
`epoch*.pt`, NOT `latest.pt` (which is absent for the first ~100 epochs — the
original driver bug that left the probe a no-op).
The whole pipeline is also re-runnable cold via `bash mined/drive_pipeline.sh`
(the mine-wait is a no-op once ≥1M is on disk; it re-pretrains → gates → AZ →
probes). Mine shards (`mined/idx2_15x15/`) and the seed
(`checkpoints/idx2_pretrain.pt`) are the durable artifacts.
