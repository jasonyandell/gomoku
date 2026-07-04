# Activity Monitor Perf Runbook

> ⚠️ **SUPERSEDED-BY [mcts-perf-ceiling.md](mcts-perf-ceiling.md) (2026-07) — this
> is a small satellite page.** The durable *how to read the Mac* guidance below
> is still LIVE; the specific perf boundaries and reference tables are **pre-oracle
> (2026-05-20/21)** and HISTORICAL. Since the VCT oracle-veto era the **oracle veto
> is ~91% of 13×13 self-play gen wall-clock** and is the top perf lever — see the
> "2026-07 oracle-dominated regime" sections of
> [mcts-perf-ceiling.md](mcts-perf-ceiling.md) before treating any boundary here as
> the next win.

Practical run/config notes for the MCTS perf-extension worktree. This page is
about operating the experiment and reading the Mac correctly; it is **not** a
claim that GPU percent is the objective.

## The durable guardrail (LIVE)

**Score changes by wall-clock, `time/gen_s`, games/sec, and positions/sec — not
by making the GPU graph taller.** Activity Monitor showing 30-50% GPU during a
healthy run is still compatible with good throughput (tiny forwards separated by
Python MCTS/state work). Do **not** chase Activity Monitor GPU percent by
collapsing workers or inflating single-call batch size unless wall-clock
throughput improves under the same production shape.

## Microbench (LIVE)

Use the lightweight bench before changing a long W&B run:

```bash
python scripts/perf_microbench.py \
  --device mps \
  --size small \
  --stem-padding 1 \
  --games 8 \
  --n-simulations 400 \
  --wave-size 64 \
  --max-plies 16 \
  --repeats 3
```

For a quick syntax/smoke check on any machine:

```bash
python scripts/perf_microbench.py \
  --device cpu \
  --size tiny \
  --games 2 \
  --n-simulations 2 \
  --wave-size 1 \
  --max-plies 2 \
  --repeats 1 \
  --warmup 0
```

The script intentionally runs through `generate_games` and the existing MCTS
evaluator boundary. It is production-shaped enough to compare config ideas, but
short enough not to be a training claim.

To compare the native MCTS engine against the Python MCTS fallback, run the same
command twice and set `GOMOKU_DISABLE_NATIVE_MCTS=1` for the fallback sample.
To isolate only the state-ops extension, set `GOMOKU_DISABLE_NATIVE_STATE_OPS=1`.

## Activity Monitor checklist (LIVE)

- Watch Python process CPU and GPU together. High CPU with moderate GPU was the
  expected shape when Python tree/state work spaced out small MPS forwards; if
  native MCTS is active and CPU is still dominant, the next suspect is the outer
  self-play/record/worker loop rather than per-node search.
- Compare same-shape benches: same model size, stem padding, games, sims, wave
  size, and `max_plies`.
- Prefer median seconds and positions/sec over a single run. Early plies and
  random seeds are noisy.
- Re-measure production-like worker count before believing a single-process
  bench. The 1-worker x big-batch detour regressed despite plausible isolated
  reasoning.
- Treat `torch.compile` as suspect for workers that reload weights every cycle;
  compile warmup can dominate the intended forward-speed win.

---

## HISTORICAL — pre-oracle reference numbers (2026-05-20/21)

**These boundaries and tables predate the VCT oracle-veto era and are kept as
era-1 evidence.** In the current regime the dominant lever is the oracle veto,
not the native hot-path boundary — see the 2026-07 sections of
[mcts-perf-ceiling.md](mcts-perf-ceiling.md).

The 2026-05-20 evidence in [mcts-perf-ceiling.md](mcts-perf-ceiling.md) said the
MPS utilization ceiling was structural: tiny forwards were separated by Python
MCTS/state work. The 2026-05-21 native MCTS pass confirmed the diagnosis:
removing most per-node Python from search moved the production-shaped
single-process bench from ~700 to ~2,200 augmented positions/sec.

Reference result, MPS small model, stem padding 1:

| config | fallback | native MCTS | read |
|---|---:|---:|---|
| 8 games, 400 sims, wave 64, max 16 plies | 701 aug pos/s | 2,200 aug pos/s | 3.14× |
| 8 games, 400 sims, wave 64, max 32 plies | 728 aug pos/s | 2,007 aug pos/s | 2.76× |

Production-shaped 10-epoch WL1 read, MPS small model, stem padding 1, 400 sims,
wave 64, 1.5M buffer, wave-lockstep, no eval sidecar:

| config | wall aug pos/s | gen aug pos/s | wall games/s | read |
|---|---:|---:|---:|---|
| native MCTS, 8 workers x 8 games | **2,379** | **3,303** | **11.25** | launch shape |
| native MCTS, 4 workers x 16 games | 1,918 | 2,152 | 8.61 | fewer workers lost |
| Python MCTS fallback, 8 workers x 8 games | 1,863 | 2,264 | 8.85 | native is 1.28x wall pos/s |

Artifact: `sweep_logs/perf10-summary.tsv`. The multi-worker speedup is smaller
than the single-process microbench because 8 worker processes already hide some
Python gaps, but the native engine still improves the real WL1 shape. Treat these
as bounded generation throughput numbers, not strength claims.

### Long-run knobs (WL1-era)

- `scripts/run_sweep.py --cell WL1` was the lockstep + 1.5M buffer cell: 8
  workers x 8 games, wave size 64, small model, stem padding 1, positions-scaled
  SGD. The point was to isolate per-version uniformity without also increasing
  replay-buffer size. (For the current knob surface see
  [training-run-reference.md](training-run-reference.md).)
- WL1 set `save_buffer_every=100`. The full `latest.pt` embeds the replay buffer,
  so long runs should not rewrite it every epoch. Intermediate `epochNNNN.pt`
  snapshots remain cheap weights+optimizer checkpoints.
- Keep `keep_last_n=3` unless preserving a specific checkpoint for evidence.
- Pair `--no-eval` trainer runs with the separate eval worker so expensive
  lookahead baselines do not interrupt generation/training throughput.
- Use `--worker-min-positions` plus `--sgd-per-position` when the question is
  buffer age/turnover stability. Use wave mode when the question is clean
  per-version tiles.

**Guardrail (pre-oracle framing):** the next structural wins were said to belong
at the native hot-path boundary (`state.apply`, `_init_node`, evaluator
materialization). In the current oracle-veto era the dominant lever is instead
the VCT oracle veto — see the 2026-07 sections of
[mcts-perf-ceiling.md](mcts-perf-ceiling.md).
