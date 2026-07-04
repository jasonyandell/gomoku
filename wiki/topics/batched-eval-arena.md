# Batched eval arena — bulk matches at self-play speed (`gomoku/arena.py`, #105)

> **Status: LIVE (2026-07-01).** Current core eval infrastructure — the default
> eval path for the derby gate, `eval_worker.py`, and in-trainer eval, each with a
> byte-identical legacy escape hatch.

**One-line:** bulk evals were slow because `play_match_pickers` plays games one
at a time and every model move runs MCTS over a single-game list — batch-size-1
MPS forwards, ~10^5 sequential dispatches per 40-game eval. The arena plays ALL
games concurrently and batches net leaf-evals across the whole field via
`run_batched_mcts_waves` (the self-play regime), generalizing the pattern
`gomoku/rapfimine/fast_eval.py` proved (~minutes → ~20s).

## Use it

```bash
uv run gomoku-arena "model:checkpoint=checkpoints/latest.pt,sims=100" vs rapfi@50ms --n-games 100
uv run gomoku-arena "model:checkpoint=a.pt" vs "model:checkpoint=b.pt" \
    --n-games 120 --random-opening-moves 4 --device mps --json
```

Library: `gomoku.arena.play_match_specs(spec_a, spec_b, n_games=..., ...)` →
`MatchResult` (same dataclass, same color-split fields, same draw=½-win
scoring as `play_match_pickers`). Specs = the `gomoku.match` grammar plus
`rapfi[:timeout_ms=N,size=K,cmd=...]` and the `rapfi@50ms` sugar;
`model:...` also takes `wave=K`. `external:cmd=...` opponents get a warm
`RapfiPool` (it's engine-agnostic Gomocup, not Rapfi-specific).

## How it works

- One process, one resident model per checkpoint (load → fuse → warmup
  forward, so MPS graph-compile isn't paid on game 1 move 1).
- Lockstep rounds partition active games by agent-to-move: net-side games go
  into ONE `run_batched_mcts_waves` call (batch ≈ games × wave_size, soft
  virtual loss — identical machinery to self-play); engine-side games fan
  across the warm pool's `label_states`; CPU pickers loop (GIL-bound anyway).
- When exactly one side is a net, the two sides run in overlapped threads
  (MPS + CPU genuinely overlap, TRAINING_WIKI 2026-06 reader-thread era);
  net-vs-net runs the two batched calls back to back (no device contention).
- Games desync in ply count; finished games just drop out of the partition.

## Measured (M5 Max, MPS, az_mini_9x9 ckpt, sims=100)

| matchup | legacy sequential | arena |
|---|---|---|
| 40 games vs `lookahead:depth=2` | ~28s (extrapolated 8-game run) | **1.9s** |
| 40 games model-vs-model (+4-ply openings) | ~34–52s (extrapolated) | **4.0s** |

Short weak-net games; the gap **widens** with longer games and bigger fields
(startup amortizes, batches stay wide). ~0.05–0.10 s/game after ~2s startup.

## Semantics / caveats

- Match semantics mirror `play_match_pickers` (color alternation, paired
  random openings, `start_state`, draw=½), but per-game RNG is independent
  (`seed+idx+1`, the `play_match_parallel` convention) — unbiased, **not
  byte-identical** to the sequential path.
- Net decision rule = legacy eval default: fresh tree per move, no root
  noise, temperature 0. Wave batching (default 16) is the same
  slightly-stale-W search self-play uses; `wave=1` recovers exact
  `run_batched_mcts`. Eval numbers therefore shift slightly vs the batch-1
  path — **switch harnesses between derby rounds, not mid-field.**
- Not yet supported: `reuse_tree`, `eval_vcf_nodes`, `proven_prop` (the
  mcts_picker levers). Fall back to the legacy path for those, or extend
  `NetAgent`. *(`vct_finish` **is** now supported — #109 added a batched arena
  finisher; see § The batched VCT finisher below.)*
- The arena uses the device you give it — a live MPS trainer is a tenant;
  pass `--device cpu` or wait (standard GPU-tenancy rule).
- Naming: `gomoku.lab.arena` (`gomoku-lab-arena`) is the autolab *promotion
  daemon* — unrelated to this match engine (`gomoku-arena`).

## Wired-through callers (#106, 2026-07-01)

All three standing callers now default to the arena, each with an escape
hatch back to the byte-identical legacy path:

- **Derby gate** — `scripts/delta_e_harness.py::head_to_head_eval` takes
  `use_arena=True` (default). Openings are regenerated with the SAME
  per-`(seed, pair_idx)` derivation as the legacy path
  (`_h2h_opening_states`), slot layout matches `_h2h_tasks` (game 2k =
  fork-as-black on opening k), and `n_workers` is ignored under the arena.
  `scripts/round_robin.py` inherits it; both CLIs take `--no-arena`.
- **`eval_worker.py`** — arena by default; `--no-arena` restores legacy.
- **`train.py` in-trainer eval** — arena by default; `--no-eval-arena`
  restores legacy.
- **Lever auto-fallback**: any eval lever (`--eval-vcf-nodes`,
  `--fpu-reduction-c`, `--reuse-tree`, `--proven-prop`,
  `--proven-vcf-leaf-nodes`, `--vct-finish-nodes`) silently isn't portable to
  the wave-batched search yet, so setting one auto-disables the arena (with a
  printed note) — the levers keep their byte-identical legacy semantics.

**Derby-gate measurement** (M5 Max, az_mini 9x9 pair, 120 games, sims=200,
paired 4-ply openings): legacy pool (8 CPU workers, device=cpu) = **59s wall
/ 630s CPU**; arena (device=mps) = **9.6s wall / 6.4s CPU** — ~6× wall, ~100×
less CPU, GPU barely warm. Short weak-net games; real champions play longer
games where the gap widens. Same story at 40 games: 13s/118s-CPU → 6s/3s-CPU.

Reminder: the switch changes eval *numbers* slightly (statistically
equivalent, not byte-identical) — verdicts recorded pre/post switch are not
directly comparable; the derby runner should note the changeover on the
research board.

## The batched VCT finisher (#109, 2026-07-01)

The sound-world product is **net + cap50 VCT finisher** (the bare net attacks but
draws on conversion at 9×9; the finisher cashes the forced win). The arena couldn't
run it — a `vct_finish` kwarg was silently ignored, so a "finisher-armed" spec in
the arena played as the **bare net**, and bare-net-vs-heuristic scored only
**5W-3L-32D** (draws it should have won). #109 gave the arena a real batched
finisher: **one bulk `solve_vct_mega_bb` per round over all to-move boards; a
proven-VCT board plays the oracle move, the rest fall through to the batched MCTS
wave.** Same machinery, wave-wide.

**Measured (107b champion, hybrid = net + cap50 finisher):** hybrid vs heuristic
**15W-0L-5D (87.5%) in 4.4 s** through the arena — matches the legacy-path receipt
(**14W-0L-6D**) that took *minutes*. (Both W-L-D are real: 15-0-5 is the arena,
14-0-6 the legacy path; the arena is not byte-identical, per § Semantics.)

**Loud-unknown-kwarg guard (the bug's root cause).** Model specs now FAIL LOUD on
any kwarg the arena doesn't implement (`gomoku/arena.py` — `unknown =
set(spec.kwargs) - known`), so a silently-dropped lever can never again masquerade
as a real result. This is why the § Semantics "not yet supported" list above is now
load-bearing: an unsupported lever *errors*, it doesn't no-op.

**Post-run final eval on MPS.** `run_sweep`'s post-run final eval moved to MPS (the
training run is torn down → the GPU is free; the *live* in-trainer eval sidecar
stays CPU, still a tenant). The standard 4-baseline battery (80 games incl.
`lookahead:4`) went **37 s CPU → 14.3 s MPS wall, identical results**. Commit
`a9e6fbf` / merge `c4c0e98`.

## CPU-side parallelism + native lookahead (#110, 2026-07-01)

#106 fixed the net side (batched MPS); the CPU side still capped at 100% of
one core. Three composing fixes (all on the `#110` branch):

1. **`PooledPickerAgent`** — expensive pickers (`lookahead`) fan each round's
   picks across a persistent spawn-Pool. Workers build the picker torch-free
   from `(kind, kwargs)` (no model load, no Metal); each pick ships
   `(state, seed)` with the seed drawn from the slot's rng in the parent, so
   same-seed runs stay deterministic. Cheap pickers (heuristic ~0.03 ms/move)
   keep the plain loop. Default workers `min(12, cpu-4)`;
   `GOMOKU_PICKER_WORKERS` env / `--picker-workers` CLI; `1` disables.
2. **`play_matches_batched_multi`** — A vs SEVERAL opponents in one field:
   one net `pick_batch` per round across all matchups, every opponent's CPU
   picks concurrent, cheap baselines finish early. Per-opponent seeding
   matches the old sequential calls (tested), so matchup semantics are
   unchanged. `eval_worker` and the in-trainer eval now play all their
   baselines in one field; `play_matches_batched` is the single-opponent
   front door over it.
3. **Native negamax** (`gomoku/_lookahead_native.c`) — the whole lookahead
   player in C, **move-identical** to Python (stable ordering + exact
   integer-valued double weights; `tests/test_lookahead_native.py` asserts
   equal tied-best lists, and whole matches reproduce W-L-D exactly).
   `GOMOKU_DISABLE_NATIVE_LOOKAHEAD=1` for A/B; per-board-size shims like
   the other native extensions.

**Measured (M5 Max):** lookahead4 38.6 → 1.5 ms/move (~26×); the 48-game
heuristic-vs-lookahead4 arena match 35.6 s → 0.30 s (~120× composed,
99% → 650%+ CPU). Note the pooled path consumes slot rngs differently than
the serial path (one seed-draw per pick vs tie-break-only), so pooled vs
serial results differ game-by-game while both stay seed-deterministic and
statistically equivalent — switch the derby gate between rounds, same rule
as the #106 arena swap.

## Cross-refs

- [mega-vct-solver.md](mega-vct-solver.md) — the `solve_vct_mega_bb` / `mega_vct_bb`
  GPU solver the batched VCT finisher (#109) calls once per round over all
  to-move boards; its contract + lane/budget numbers.
- [mcts-perf-ceiling.md](mcts-perf-ceiling.md) — the gen-side counterpart. The
  arena reuses the same `run_batched_mcts_waves` wave-batched search documented
  there; this page is the eval-side application of that machinery.
- [eval-suite.md](eval-suite.md) — the eval harness / baseline battery the arena
  now backs (derby gate, `eval_worker`, in-trainer eval).
- [rapfi-pool.md](rapfi-pool.md) — the warm `RapfiPool` the arena fans
  engine-side games across for external/Gomocup opponents.
- [m5-mainframe.md](../m5-mainframe.md) — parent perf hub.
