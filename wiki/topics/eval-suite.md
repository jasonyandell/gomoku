# The eval suite — how to measure a checkpoint's strength

**THE how-to for evaluating a checkpoint.** Every reliable tool, copy-paste
commands, and the gotchas that have burned us. Doctrine (what counts as
reliable, and why wine engines are shelved) lives in
[reliable-eval-set.md](reliable-eval-set.md); the batched engine these commands
drive is [batched-eval-arena.md](batched-eval-arena.md); the Rapfi anchor's
warm-pool wrapper is [rapfi-pool.md](rapfi-pool.md).

## READ THIS FIRST — the gotchas that silently lie

1. **Eval the EMA `worker_weights.pt`, NOT the raw epoch checkpoint** (#100,
   #8G). The trainer publishes `checkpoints*/worker_weights.pt` = the EMA that
   actually plays. `epoch*.pt` / `latest.pt` carry the *raw* training weights,
   transiently weak mid-training. Same epoch has read **6% raw vs 68% EMA** (#100)
   and **35% raw vs 83% EMA** (15×15, #8G) — a 48-point artifact. Point every
   command below at `worker_weights.pt`.
2. **`GOMOKU_BOARD_SIZE` must match the checkpoint.** Board size is an env var
   (`gomoku/game.py:24`, default 9). A 13×13 net eval'd at the default 9×9 shape
   is garbage. Export it before any eval of a non-9×9 net:
   `export GOMOKU_BOARD_SIZE=13`.
3. **Gate on H2H + the per-color columns, NOT internal elo.** Fixed baselines
   SATURATE (#100: the champion reads 62% vs heuristic yet 40-0s the control
   the heuristic-elo calls its equal). Anchored elo caps ~1700. The robust
   verdict is head-to-head vs a fixed reference plus the black/white split.
4. **The white column is the defense gate.** Our attack-only specialists win
   as black (they force a VCT) and go **0/20 as white** vs anything that
   defends — invisible to every self-metric until you read the split (#100,
   #107, #113). A net that isn't strong in the `--json` `white` column has not
   learned to defend, no matter what the aggregate win-rate says.
5. **Use the batched arena, not the sequential path.** `gomoku-arena` runs a
   40-game eval in ~4s on MPS (all games concurrent, net leaf-evals batched
   across the field) vs minutes for the legacy `gomoku.match` loop. The arena
   is the default front door for everything below.
6. **The MPS trainer/derby is a tenant.** A live GPU run means `--device cpu`
   or wait — standard tenancy rule. Contended evals are noisy (and, for
   net-vs-baseline, biased); prefer uncontended.

## The tools, with copy-paste commands

All via `gomoku-arena` (`gomoku/arena.py`, entry point in `pyproject.toml:47`).
Spec grammar: `model:checkpoint=PATH[,sims=N,c_puct=F,wave=K,vct_finish=N]`,
`heuristic`, `lookahead:depth=N`, `rapfi@50ms` (sugar for
`rapfi:timeout_ms=50`), `rapfi:timeout_ms=N,size=K,cmd=PATH`. `--json` splits
the result by color. Unknown `model:` kwargs FAIL LOUD (#109 — a `vct_finish=50`
spec once silently played the bare net; the arena now rejects any kwarg outside
`{checkpoint, sims, c_puct, wave, vct_finish}`).

### (1) Fast net vs pure-python baselines (the floor)

```bash
CKPT=checkpoints/worker_weights.pt      # EMA, not epoch*.pt (gotcha #1)
uv run gomoku-arena "model:checkpoint=$CKPT,sims=100" vs heuristic \
    --n-games 40 --device mps --json
uv run gomoku-arena "model:checkpoint=$CKPT,sims=100" vs lookahead:depth=4 \
    --n-games 40 --device mps --json
```

`heuristic` is the absolute floor; `lookahead:depth=4` is the top of the
pure-python ladder (native negamax, ~1.5 ms/move, move-identical to Python,
#110). Both are torch-free, never crash. **Read the `white` column** — winning
only as black is the attack-only tell (gotcha #4).

### (2) Net vs Rapfi@50ms (the honest external anchor)

```bash
uv run gomoku-arena "model:checkpoint=$CKPT,sims=100" vs rapfi@50ms \
    --n-games 40 --device mps --json
```

Native arm64 Rapfi-NNUE (mix9svq weights, single-thread) — the first honest
strength reference past the ~1700 ladder ceiling (#40/#28). The arena spins a
warm `RapfiPool` and fans engine moves across it (`--engine-pool-size`, default
8). Rapfi must be a real NNUE build — the stock classical config is weightless
and ignores search time (#28). If you drive Rapfi outside the arena (e.g.
`scripts/eval_vs_rapfi.py --jobs J`, parallel spawn-pool, #52), pass the
`run-rapfi` WRAPPER, not the bare binary, so the NNUE loads. See
[reliable-eval-set.md](reliable-eval-set.md) and
[external-engine-baselines.md](external-engine-baselines.md).

### (3) Net vs net head-to-head (THE yardstick)

```bash
uv run gomoku-arena "model:checkpoint=A/worker_weights.pt,sims=100" \
                 vs "model:checkpoint=B/worker_weights.pt,sims=100" \
    --n-games 120 --random-opening-moves 4 --device mps --json
```

Pure torch, never crashes. Non-transitive across recipes, so pin ONE fixed
reference (a frozen champion) and read every candidate against it. Paired
4-ply openings (`--random-opening-moves 4`) neutralize first-mover advantage;
each color-swap pair shares one opening. This is the measure to gate on
(gotcha #3). Two nets contend for MPS, so net-vs-net runs the two batched calls
back-to-back (still fast, just no CPU/MPS overlap).

### (4) Bare net vs finisher-armed (VCT finisher A/B, #109)

```bash
# bare
uv run gomoku-arena "model:checkpoint=$CKPT,sims=100" vs heuristic \
    --n-games 40 --device mps --json
# finisher-armed (cap50 GPU-VCT oracle hammers a forced VCT to a real five)
uv run gomoku-arena "model:checkpoint=$CKPT,sims=100,vct_finish=50" vs heuristic \
    --n-games 40 --device mps --json
```

`vct_finish=50` arms the batched GPU-VCT finisher: each round ONE bulk
`solve_vct_mega_bb` over every to-move board; games with a forced VCT within
the node cap play the oracle's winning move, the rest fall through to MCTS. `0`
(default) = OFF, never imports MLX, byte-identical to the bare agent. The
finisher only *converts an existing forced win* — it does nothing against an
opponent that never hands one over, so it lifts the black/attack column, not
the white/defense column. `vct_finish` is arena-native; other mcts_picker
levers (`reuse_tree`, `eval_vcf_nodes`, `proven_prop`) are NOT — for those, use
the legacy `python -m gomoku.match` sequential path.

### (5) The gen-semantics poison check (a guardrail, not a strength eval)

```bash
uv run python scripts/gen_poison_check.py checkpoints/worker_weights.pt
# optional: uv run python scripts/gen_poison_check.py CKPT overlap <seed> <concurrent>
```

Run after ANY change to generation semantics (#107). It generates games at the
LIVE config (MPS, sims=100, wave=32, VCT terminus + oracle veto on), then
re-solves every recorded position at full breadth and asserts the veto
invariant: **recorded `pi` must have ZERO mass on proven-blunder cells**
(`VIOLATIONS: 0/N`). This caught the sound-world white-collapse wound (doomed
defender-terminus positions recorded with uniform pi). Not a strength measure —
a soundness gate on the training data.

## Worked example (2026-07-02, #113 morning) — read the columns

40-game arena evals, EMA `worker_weights.pt`, sims=100, **13×13** (so
`GOMOKU_BOARD_SIZE=13`), our two sound-world nets finisher-armed (`vct_finish=50`):

| matchup | A w-l-d | black | white |
|---|---|---|---|
| from-scratch+fin vs **rapfi@50ms** | 0-40-0 | 0/20 | 0/20 |
| warm-start+fin vs **rapfi@50ms** | 0-40-0 | 0/20 | 0/20 |
| from-scratch+fin vs warm-start+fin [H2H] | 20-20 | 20-0 | 0-20 |
| **OLD 128×10** (bare) vs **rapfi@50ms** | 3-37 | 3/20 | 0/20 |

The lesson is entirely in the columns: our nets go **0-40 vs rapfi**, and the
H2H is **exactly 50/50, purely color-determined** — whoever is black forces a
VCT, both nets are behaviorally identical attack specialists (no skill delta;
the aggregate 50% hides that). The OLD full-game 128×10 net — which we "never
focused on defense" — **beats both sound-world nets 40-0 and scores 7.5% vs
rapfi where ours score 0%**, because it trained on full games and learned to
defend. Net+finisher was NOT a product at 13×13; it's a black-only party trick.
Full entry: TRAINING_WIKI 2026-07-02 (#113 morning). Recipe context:
[sound-world-recipe.md](sound-world-recipe.md).

## Quick reference

| I want to… | command |
|---|---|
| net vs floor | `gomoku-arena "model:checkpoint=W,sims=100" vs heuristic --n-games 40 --device mps --json` |
| net vs lookahead | `… vs lookahead:depth=4 …` |
| net vs external anchor | `… vs rapfi@50ms …` |
| net vs net (the gate) | `gomoku-arena "model:checkpoint=A,sims=100" vs "model:checkpoint=B,sims=100" --n-games 120 --random-opening-moves 4 --device mps --json` |
| finisher A/B | add `,vct_finish=50` to the model spec |
| gen soundness | `python scripts/gen_poison_check.py W` |

`W` = `checkpoints*/worker_weights.pt` always. `export GOMOKU_BOARD_SIZE=N` for
non-9×9 nets. Read the `white` column.
