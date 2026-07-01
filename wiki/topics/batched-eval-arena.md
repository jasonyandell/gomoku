# Batched eval arena — bulk matches at self-play speed (`gomoku/arena.py`, #105)

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
- Not yet supported: `reuse_tree`, `eval_vcf_nodes`, `proven_prop`,
  `vct_finish` (the mcts_picker levers). Fall back to the legacy path for
  those, or extend `NetAgent`.
- The arena uses the device you give it — a live MPS trainer is a tenant;
  pass `--device cpu` or wait (standard GPU-tenancy rule).
- Naming: `gomoku.lab.arena` (`gomoku-lab-arena`) is the autolab *promotion
  daemon* — unrelated to this match engine (`gomoku-arena`).

## Not yet rewired (follow-ups)

In-trainer eval (`train.py` eval block), `eval_worker.py`,
`scripts/delta_e_harness.py::head_to_head_eval` / `round_robin.py` (the derby
gate — biggest Δelo/Δt win: 120 games/pair × sims=200 currently on 6 CPU
process workers) still use the sequential/spawn paths. See #105 follow-ups.
