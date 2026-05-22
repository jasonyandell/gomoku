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
