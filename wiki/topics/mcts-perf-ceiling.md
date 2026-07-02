# MCTS Perf Ceiling

Synthesis page on where gen-time wins are and aren't in `gomoku/mcts.py`. The
goal is to stop re-discovering, every few sessions, that "porting v2 storage
from some upstream AZ codebase" is a no-op for us.

## TL;DR

Our `Node` already owns per-action arrays (`N`, `W`, `P`) of size `N_ACTIONS`,
`_select_action` is a single vectorized `np.argmax` over 81 elements, and no
dict iteration happens in the PUCT hot path. **Structurally we are already at
the "AGZ mcts_v2" layout** that other gomoku/AZ READMEs advertise as a big win
over their "mcts_v1." That refactor is therefore a no-op for us.

Real production constraint: the gen path was bounded by `state.apply` (board
copy + history snapshot, Python), `_init_node` (terminal check + legal mask,
Python), MCTS tree orchestration, and the MPS forward + sync. None of those is
a "vectorize-with-numpy" win — closing the gap requires moving the boundary,
not rearranging Python arrays.

2026-05-21 update: the first deep-native pass moved `Node`, child creation,
PUCT selection, virtual loss, backup, bitboard state/history, and leaf
input-plane materialization into optional `gomoku._mcts_native`. Python still
owns the PyTorch callback at wave boundaries and the outer self-play loop, but
the per-node/per-leaf search churn is no longer Python-owned when the Torch
evaluator is used.

## What is already in the tree

- **Per-action numpy arrays on each `Node`.** `N`, `W`, `P` are `(81,)`
  arrays. Selection is one `np.argmax` over masked-legal scores.
- **Wave-batched MCTS with soft virtual loss.** See `run_batched_mcts_waves`
  and `_select_one_vloss`. Soft vloss (`N += 1`, no `W` change) verified
  byte-for-byte against sequential descent under fixed seed.
- **Cross-game BFS-vectorized descent.** `_bfs_descend_one_per_game` stacks
  current-level nodes from all games into `(P, 81)` arrays and does one
  batched PUCT-`argmax` per BFS level. Within-game wave-slot ordering is
  preserved, so output is identical to the pre-refactor sequential wave
  under any RNG.
- **AlphaGo Zero log-schedule PUCT.** `pb_c = log((1 + N + c_puct_base) /
  c_puct_base) + c_puct_init`, with AGZ defaults 19652 / 1.25.
- **Combined `.cpu()` evaluator transfer** to halve MPS syncs.
- **Vectorized terminal check** (`_has_five_in_a_row` directional ANDs).

## Things that do NOT work as a quick win

- **Porting michaelnny/alpha_zero `mcts_v2.py` storage.** We are already at
  that layout. The fact that other codebases advertise a 2-3× speedup from
  "porting v2" is comparing to their previous per-Node-field design, which
  we never had.
- **fp16 evaluator on MPS.** 17% *slower* in our bench (see
  [TRAINING_WIKI.md](../../TRAINING_WIKI.md) Exp 4).
- **Async gen+train in one process on one MPS device.** MPS is single-stream
  per process, so forward (gen) and forward+backward (train) serialize and
  the train step ballooned 7× (Exp 2). Async only helps with multiple GPUs
  or genuinely CPU-bound paths.

## What the cross-game BFS descent actually buys (Exp 9)

Measured median of 3 trials, medium model on MPS:

| n_games | sims | wave | seq    | bfs    | speedup | savings |
|--------:|-----:|-----:|-------:|-------:|--------:|--------:|
| 32      | 200  | 16   | 0.407s | 0.324s | **1.26×** | 20.4% |
| 32      | 200  | 32   | 0.343s | 0.301s | 1.14×   | 12.4% |
| 16      | 400  | 16   | 0.388s | 0.365s | 1.06×   | 5.8%  |
| 32      | 100  | 8    | 0.129s | 0.111s | 1.16×   | 13.8% |

Pattern: win scales with **G** (games-per-wave-call) and shrinks as
**wave_size** grows (fewer BFS calls per wave to amortize Python dispatch
overhead).

In our dist worker config (4 workers × 8 games/batch ⇒ G=8 per wave call) the
realized win is in the 5–10% range. In single-process at G=32 with wave=16
it's ~20%. Verified byte-for-byte against the sequential reference, see
`tests/test_mcts.py::test_wave_bfs_matches_sequential_byte_for_byte`.

## Where the next 2× lived (and what remains)

Before the native engine pass, the rough leverage order was:

1. **Batched `state.apply` on GPU.** Today every newly-visited child does a
   board-copy + history-snapshot in Python. If state can be encoded as a
   tensor and `apply` becomes a batched op, the per-child Python overhead
   disappears. Big refactor — affects `game.py`, `mcts.py`, `replay_buffer.py`.
2. **C/Cython `_init_node`.** Terminal check + legal-mask construction per
   new child. Probably 5–10% wall-clock standalone.
3. **Multi-device gen-vs-train split.** Async pipelining only helps when the
   two workloads run on different devices; this is the right answer on a
   CUDA cluster, not on one MPS device.
4. **Strong virtual loss (W -= 1 with explicit revert).** Marginal — would
   make waves more diverse but our wave=32 already matches wave=1 in plies.

The 2026-05-21 native engine fused #1/#2 with the MCTS tree itself instead of
making a GPU tensor-state rewrite. Same-shape MPS microbench results:

| config | Python MCTS + native state ops | native MCTS | speedup |
|---|---:|---:|---:|
| 4 games, 32 sims, wave 4, max 6 plies | 2,438 aug pos/s | 2,888 aug pos/s | 1.18× |
| 8 games, 400 sims, wave 64, max 16 plies | 701 aug pos/s | 2,200 aug pos/s | **3.14×** |
| 8 games, 400 sims, wave 64, max 32 plies | 728 aug pos/s | 2,007 aug pos/s | **2.76×** |

Interpretation: the win gets much larger in the production-shaped wave-64
config because the old path paid Python descent/child/state overhead across
many leaves per evaluator call. The native path amortizes Python over the whole
wave callback.

Remaining likely wins:

1. Fuse the eval-only network graph before workers build their evaluator.
   A live WL5 worker sample on the M5 Max showed nearly all sampled
   `native_search_batch` time falling through `call_evaluator`, with repeated
   MPS BatchNorm / graph-execution frames. Direct small-model MPS forward
   timing under live WL5 load dropped from roughly 1.9-2.1ms to
   0.86-1.14ms for batch sizes 8-128 after Conv+BatchNorm fusion, with
   output parity at float noise. **Landed and validated in production**
   2026-05-21 via WL5 worker hot-restart: microbench A/B showed
   **1.47×** aug-pos/s under live contention; production wave-mode
   measurement showed **1.53× games/sec on gen** over 25+25 epochs.
   See [TRAINING_WIKI.md](../../TRAINING_WIKI.md) "Production
   verification" subsection under the 2026-05-21 fusion entry for
   the reusable hot-restart procedure.
2. Split eval-only inference across Apple Silicon engines. The next large
   hardware-specific bet is not just "make MPS faster"; it is self-play
   leaf eval on ANE/Core ML, training on MPS GPU, and eval sidecar work on
   CPU/BNNS so those lanes stop contending for one backend. See
   [ane-int8-inference.md](ane-int8-inference.md).
3. Move more of the outer self-play loop native: action sampling, trajectory
   staging, and D4 augmentation. This should be smaller than the search-engine
   jump but removes Python at the move/record boundary.
4. Profile the post-search worker loop before another native pass; the search
   engine is no longer the only plausible Python owner.
5. Consider a heavier evaluator/model only after native search reduces CPU gaps
   enough that the MPS forward becomes the pacing item.

The actual multi-worker WL1 check is now done. Ten epochs at the next-run
shape (`small`, stem padding 1, 400 sims, wave 64, 1.5M replay,
wave-lockstep, no eval sidecar) produced:

| config | wall aug pos/s | gen aug pos/s | wall games/s |
|---|---:|---:|---:|
| native MCTS, 8 workers x 8 games | **2,379** | **3,303** | **11.25** |
| native MCTS, 4 workers x 16 games | 1,918 | 2,152 | 8.61 |
| Python MCTS fallback, 8 workers x 8 games | 1,863 | 2,264 | 8.85 |

Read: native is still a production win (1.28x wall positions/sec, 1.46x
generation positions/sec over fallback), but multi-worker scheduling already
hid enough Python work that the single-process 2.8-3.1x speedup should not be
used as the launch ETA. The 8-worker shape remains better than same-tile 4
workers x 16 games even after native search.

## How to reason about MCTS perf claims in the future

When some upstream AZ repo claims a big speedup from a refactor:

1. **Grep our own `gomoku/mcts.py` first** for the named technique. We've
   already vectorized PUCT selection, cached terminal info, vectorized
   five-in-a-row, batched `.cpu()` transfers, and BFS-vectorized descent.
2. **Look at our profile, not theirs.** From wiki Exp 3: `_select_action`
   was 35% of gen but already vectorized, so the win there is at most the
   Python dispatch overhead, ~5-10%. `evaluate` (MPS forward + .cpu()) was
   47%, attacked by combined-`.cpu()` and wave batching.
3. **Estimate the realized win on our dominant config** before committing
   to the refactor. The dist mode with G=8 per wave is the right baseline
   for "would this help during a real training run?" — not a microbench at
   G=32.

## References

- [activity-monitor-perf-runbook.md](activity-monitor-perf-runbook.md) —
  practical run knobs and microbench command for interpreting Mac GPU/CPU
  readings without chasing GPU percent.
- [TRAINING_WIKI.md](../../TRAINING_WIKI.md) Exp 9 — the cross-game BFS
  descent: implementation, bench, and the "agent's 2-3× claim was wrong"
  finding.
- [TRAINING_WIKI.md](../../TRAINING_WIKI.md) Exp 3, 6, 7, 8 — earlier perf
  work (profiling, combined `.cpu()`, vectorized terminal, wave-batched
  MCTS).
- `gomoku/mcts.py` — current implementation, `_bfs_descend_one_per_game`
  for the BFS path.
- `tests/test_mcts.py` — byte-for-byte correctness check on the BFS path.

## 2026-05-20 update — the 1-worker collapse and the GPU-saturation reality

Mid-run perf intervention attempt. A subagent measurement of the live
checkpoint concluded:
- workers are CPU-bound (~80% on Python tree-traversal between MPS calls)
- per-MPS-call latency is ~2ms regardless of batch 1-256 (kernel dispatch
  dominates compute; the small 324k-param model can't fill the device)
- claim: 4 workers on one MPS device serialize on the per-process stream,
  so collapsing to 1 worker × big batches would unlock 2-3× throughput

We deployed the change as `cell Z` reconfig + torch.compile: 1 worker × 32
games × wave=64 + compile. **Result was a regression, not a win.**

| config | cycle_s | games/sec | notes |
|---|---:|---:|---|
| 4w × 8g × wave=32 (pre-detour) | ~33 | 0.97 | baseline |
| 1w × 32g × wave=64 + compile | 36-63 (growing) | 0.5-0.9 | **slower** |
| 4w × 8g × wave=64 (rollback + kept wave win) | ~22 | 1.45 | **+50% over baseline** |
| 8w × 8g × wave=64 (more parallel) | ~17 | 1.88 | +94% over baseline; **GPU only 30-40%** |

Lessons:

1. **The "MPS serializes processes" claim is overstated.** macOS + MPS
   queue kernels across processes well enough that 4 small workers achieve
   significant device-time overlap. The single-worker config doesn't
   inherit that — it leaves the GPU idle during its long Python phases.
2. **`torch.compile` doesn't pay off when workers reload weights every
   cycle.** The compile cache is per-process, partly invalidated on weight
   reload, and the 1-cycle reload cadence pays compile-warmup repeatedly.
   Combined cycle time GREW (36→63s) over a few cycles rather than
   amortizing.
3. **At our model size, more workers > bigger batches.** Each MPS call is
   2ms whether the batch is 32 or 1024 samples. The win comes from having
   more processes queue calls in parallel, not from saturating any single
   call. Practical ceiling ~8-12 workers before process-contention and
   file-IO overheads eat into the gain.
4. **GPU underutilization is structural, not a tuning problem.** Activity
   Monitor showing 30% even at 8 workers means each call is ~2ms but the
   gaps between calls (Python tree-walk in `_bfs_descend_one_per_game` and
   `state.apply`/`_init_node`) are larger. The model is genuinely too small
   to saturate ~5120 GPU cores. No worker count fixes this. **The actual
   "next 2×" is batched `state.apply` on tensor + C-extension
   `_init_node`** — moves the Python-side work off the critical path so
   the GPU can be fed continuously.

### How to avoid this kind of misread

When estimating production speedup from an isolated bench:

- A single-process bench under-counts the OS-scheduled cross-process
  parallelism that multi-worker setups achieve. Always re-measure with the
  same number of competing processes the production setup uses, or
  estimate from production logs (which already reflect real wall-clock).
- `torch.compile` benefits assume long-lived models. If your workers
  reload weights more often than the compile-amortization window, compile
  is net-negative.
- "GPU isn't saturated" doesn't always mean "throw more parallelism at
  it." If the kernels are individually too small, more parallel callers
  just queue more 2ms calls — same total GPU time, more CPU spent
  dispatching.

Saved as memory `project_perf_bench_lesson` for cross-session recall.

## 2026-07-01 — The oracle-dominated regime (sound-world #107) and its levers

The #107 gen recipe (`--vct-terminus --oracle-veto`, cap50) moved the gen
bottleneck OFF the net/search entirely: at the live config (8 games/batch,
sims=100, wave=32, GPU-quiet, weights=v1239), one batch profiled as **oracle
75% / evaluator 23% / native tree work 0.6% / python ~1%** (wall 5.11 s =
terminus solve 1.45 + defense escape-solve 2.40 + search 1.19 [evaluator 1.16
of it] + 0.06). Games desync, so the per-ply solver batch is ~150 boards
(~8 games × ~legal-cells children + terminus boards).

**Correction to the call-cost-law intuition at this operating point:** the
"one call = one tail, flat in B" law holds for big `max_nodes` on hard boards;
at **cap50 with ~10–1000 board batches the call is WIDTH/WORK-bound, not
tail-bound** (measured: empty boards ~0.3–1 ms at any B — dispatch+sync is
negligible; realistic boards scale ~2–3× when B goes 8→800). Consequences:
merging calls saves only the dispatch (~1 ms), and skipping whole waves early
saves ~nothing — the trivial call is its own cheap precondition. The real
lever is **reducing solver board-work**.

Levers landed 2026-07-01 (branch `worktree-agent-abc3efd96c269e71e`,
`gomoku/self_play.py`; A/B receipts in the session log / final report):

1. **Merged per-ply solve** (`_oracle_ply_solve`, default-on): terminus +
   defense children in ONE dispatch. Bit-identical (per-thread node budget =>
   batch-composition-independent verdicts; `return_move` selects the same
   kernel). Worth ~1.07× alone.
2. **Null-board precheck** (`--no-oracle-precheck` to disable; default ON,
   byte-identical BY PROOF): solve the "I pass" board per position in phase 1;
   a CLEAN no-win (win=False AND hit_cap=False) proves every escape child is
   no-win (freestyle monotonicity + solver 0-FP), so children are never built
   or solved. A **capped** null is NOT skippable (the child's extra defender
   stone can prune the attacker's tree into the node budget). Measured on live
   gen positions: 67.9% of plies clean → **63.9% of children solver-work
   skipped**; the veto-stretched endgame (plies 40+) is ~all clean, early-mid
   plies 10–29 are the hot band (~52–55%).
3. **Oracle/search overlap** (`--oracle-overlap`, default OFF, flag-gated):
   the merged solve runs in a background thread while the MPS wave searches;
   partitions apply post-search. MLX releases the GIL, but **MLX and MPS
   contend for the same GPU** (evaluator ~2× slower while a solve runs), so
   the overlap gain is partial, not min(solve, search). Deterministic per
   seed; not guaranteed byte-identical to serial order (terminated games are
   searched-and-discarded on their firing ply → evaluator batch shapes shift —
   the wave-size-change numeric class).
4. **Staged veto breadth** (`--oracle-veto-max-cands K`, default 0 = full):
   K-nearest stage 1 + full-breadth escalation only for all-tested-blunder
   positions (defender terminus stays exactly sound). **WARNING (measured):**
   at 9×9/K=24 games collapse from ~25 to ~11 plies — missed vetoes are played
   blunders, gutting exactly the anti-attractor effect #107 wants. This is a
   BIG-BOARD lever (N² children growth), to be tuned there with a leak-rate
   measurement, not a 9×9 speed knob.

Side notes: `--fp16-eval` measured **1.7× faster per position** here
(0.075 vs 0.127 ms/pos) but changes eval numerics (different games) — still
needs the TQ canary per the L06/L11b' entries above. The MCTS engine itself
(hypothesis "the impl has headroom") is 0.6% of gen wall at sims=100 — no
meaningful headroom left there; after the oracle levers the next wall is the
evaluator's ~2 ms/call MPS dispatch floor (see the sections above).

### 2026-07-01 CORRECTION (same session, round-2 A/B): the cost model is
### CALL-COUNT x TAIL-GRIND; the null-board precheck is REFUTED at 9x9

Round-2 measurement (same protocol, GPU-quiet) overturns the "width/work-
bound" reading above and refutes lever #2 as a 9×9 speedup:

- **Per-call solver cost is ~CONSTANT at this operating point:** merged
  single-call = 81 calls / 12,218 boards / 3.545 s = **43.8 ms/call**;
  precheck two-phase = 98 calls / 4,727 boards / 4.345 s = **44.3 ms/call**.
  Cutting boards 2.6× changed nothing per call — the call pays the hardest
  board's full cap50 grind (~0.9 ms/node on ONE GPU thread), width rides
  free. The round-1 "width scaling" probe read was a difficulty-sampling
  artifact of random boards (more random boards ⇒ harder max).
- **Precheck verdict: byte-identical but SLOWER** (oracle 3.55 → 4.35 s,
  wall 4.80 → 5.57 s/batch): the 61% board reduction saved nothing and the
  phase-2 split added ~17 calls/batch. Default flipped to OFF; the flag
  remains a big-board experiment (children build cost and widths differ
  there). The monotonicity-skip PROOF and its receipts remain valid.
- **What actually shipped as the win:** merged solve (1.06–1.07×, byte-
  identical, default-on) + `--oracle-overlap` (**1.18× end-to-end** at live
  config, identical games at the test seeds, deterministic; MLX/MPS GPU
  contention — evaluator 1.16→2.3 s while a solve runs — caps the overlap
  gain well below min(solve, search)).
- **The measured next levers, in leverage order:** (1) **cross-worker solver
  batching** — width is free, so ONE shared solve for all 4 workers costs the
  same ~44 ms as each worker's own call ⇒ aggregate oracle GPU time ÷4
  (architectural: a solver service or worker consolidation); (2) **kernel
  tail** — ~0.9 ms/node/thread is the grind; a multi-thread-per-board or
  memory-layout pass on `mega_vct_bb` attacks the floor directly; (3) lower
  cap (cap50→cap25) = semantics change, needs a recall measurement first
  (cascade data says cap50 ≈ 98.8% of VCTs; cap25 recall unknown).

## 2026-07-01 — Lever (1) LANDED as continuous-refill consolidation (#112); the law gains a divergence bump; 13×13 census says the next wall is the kernel (#114)

**What shipped (merged to main, `Closes #112`):** the cross-worker shared solve
landed as the *worker-consolidation* option, plus a lever the filing didn't
anticipate: **continuous game refill** (`--concurrent-games W`, `--stream`).
`_generate_games_native` now tracks per-game plies (legacy lockstep is
byte-identical: values coincide) and seeds a replacement game the moment one
completes, so every merged per-ply solve and every MPS search wave runs at
full width W instead of paying the lockstep batch's thinning tail (baseline:
81 solve calls to finish an 8-game batch of mean ~26 plies — 10.1 calls/game;
at W=256 it's ~0.5 calls/game). `--stream` makes production continuous:
chunks of finished games flush to the trainer as they complete and
`worker_weights.pt` hot-reloads between rounds (in-flight games finish under
the new net). The `sound-world` cell is rewired: ONE streaming worker at
W=256 replaces the 4×8 fleet.

**Measured (bench_gen_refill.py, 107b champion weights, live semantics):**
one process = 4,080 aug-pos/s at W=256 / 5,579 at W=512 vs the whole 4-worker
fleet's ~1,000–1,300 → **~3.4–4.6×**, with the solve fully hidden under
search (join stall 0.3 s) and evaluator cost down 186→14–20 µs/pos from batch
fattening alone. `gen_poison_check`: 0/174 legacy, 0/1790 refill.

**Law amendment — width is free only up to a divergence bump:** per-call cost
at cap50, GPU-quiet: 44 ms @ ~150 boards → 108 ms @ 862 → 116 ms @ 1,708 →
148 ms @ 7,276. Intra-simdgroup divergence loads the call ~2.5× as hard lanes
appear in every simdgroup, then SATURATES — so beyond ~1k boards width rides
~free again, which is exactly what refill exploits.

**13×13 (the #113 prerequisite) — measured, and the story inverts:** the
solve is 100% of the binding constraint (oracle 87–89 s of a 92–94 s wall at
W=64; search idles 45 s at the join; fp16-eval frees the evaluator 70.6→44.2 s
— the bandwidth-bound prediction holds at bigger boards — and buys ~nothing).
Refill still gives 2.8× over lockstep per process (539 vs 192 aug-pos/s), but
the resolve census on 22 captured real veto batches kills the "tail" mental
model at 13×13: **48.0% of boards solve @budget 10, 9.5% @11–50, 42.5% CAP at
50** — half the batch grinds to cap, every simdgroup is saturated. Measured
consequences (verdict-equality asserted): budget ladder 10→50 = **0.83×
(loss)**; oracle-sort resolve-class clustering (upper bound of any hardness
sort) = 0.98×; tg 64/128 = 1.00×; **null-board precheck refuted AGAIN at
13×13 (0.59×)** by a new mechanism — the null boards are themselves cap-bound
mid-game (opp-has-VCT at 13×13 IS the bottomless search), so phase 1 costs
more than the skip saves (at 9×9 it was refuted because width was free).
Receipts + attack list in issue #114: (1) multi-thread-per-board kernel
rewrite (the per-node cost every capped lane pays), (2) cap50→cap25 recall
study (the 9.5% census bucket bounds the 13×13 miss rate), (3) veto-breadth
staging with a leak-rate measurement (last resort, semantics-gated).
