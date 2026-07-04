# FPU reduction at eval (`--fpu-reduction-c`)

**Status: LIVE — landed (derby-3w0, 2026-05-28). Eval-only knob, default OFF**
(`fpu_reduction_c = 0.0` = byte-identical to legacy). **Scope caveat: the gain is
9×9-champion-specific.** A clean frozen-checkpoint A/B at 15×15 found **no gain**
(FPU-off 50/83% vs FPU-on 33/83% vs Rapfi = within noise) — FPU reduction does
**NOT** transfer to 15×15. See
[the 15×15 campaign](15x15-training-campaign.md) (2026-06-13 log).

## What

KataGo-style **First-Play Urgency** reduction in MCTS selection. Replaces the
legacy `Q = 0` default for unvisited children with

```
Q_fpu = parent_V - c * sqrt(sum_visited_priors)
```

where

- `parent_V = sum(node.W) / sum(node.N)` (node's running value across visited
  children — implemented in `gomoku/mcts.py:_select_action`),
- `sum_visited_priors = sum(node.P[a] for a in node.children if node.N[a] > 0)`,
- `c = --fpu-reduction-c` (CLI flag; 0.0 = OFF). KataGo c≈0.45 subtree / 0.20
  root; LCZero default 0.33.

When `c == 0.0` OR the parent has no visited children yet (`total_visits == 0`),
the implementation falls through to the legacy `Q = 0` branch. This makes the
default fully byte-identical to the pre-lever path.

## Why

The legacy `Q = 0` default makes unvisited siblings indistinguishable from
weakly-explored ones in **drawish positions** (`parent_V ≈ 0` — the
lookahead4-black binding regime where every line evaluates to a 0-EV draw).
UCT then scatters ~30-50% of the 100-sim budget on dead-equal siblings instead
of deepening the principal variation, so 5-6-ply forcing wins go undiscovered.

FPU reduction makes unvisited siblings inherit **pessimism** from the parent's
visited prior mass — they look slightly losing by default — so PUCT prefers to
deepen the already-explored PV. Search-side counterpart to WDL's value-side
fix; the two stack (eval-VCF for exact tactical solving, eval-sims for raw
ceiling, FPU for budget allocation).

## Where it's wired

EVAL-ONLY by design. Threading:

- `gomoku/mcts.py`: `_select_action`, `_select_one`, `MCTSGame.__init__`,
  `run_batched_mcts` — all accept the optional `fpu_reduction_c`. Default 0.0.
- `gomoku/eval.py`: `mcts_picker` and `play_match_parallel` thread it down.
- `gomoku/eval_worker.py`: `--fpu-reduction-c` argparser flag (default 0.0).
- `gomoku/train.py`: `--fpu-reduction-c` argparser flag for the in-trainer
  eval (default 0.0). Self-play / generation / training MCTS are NOT affected.
- `gomoku/selfplay_worker.py`, `gomoku/self_play.py`: UNTOUCHED. Gen MCTS still
  takes `Q = 0` for unvisited children. The C native MCTS (gen hot path) is
  not touched.

## How to use

Standalone eval cycle:

```bash
gomoku-eval-worker --checkpoint-path ... --fpu-reduction-c 0.45
```

In-trainer eval (`run_sweep` / direct `gomoku-train`):

```bash
gomoku-train --eval-in-trainer --eval-sims 100 --fpu-reduction-c 0.45
```

Sweep ideas (defer to a Δelo Derby cell or `probe_100pct.py` extension):

- 4-cell sweep at fixed sims=100, eval_vcf=0: c ∈ {0.0, 0.20, 0.45, 0.65}.
- Stacking grid (probe_100pct.py-style): sims × eval_vcf_nodes × fpu_c.

## Tests

`tests/test_fpu_reduction.py`:

- OFF byte-identical: identical visit counts on a fixed-seed run between the
  default (no kwarg) path and explicit `fpu_reduction_c=0.0`.
- ON math-exact: `parent_V=0.1, sum_visited_priors=0.6, c=0.45` produces
  `fpu_q = 0.1 - 0.45 * sqrt(0.6) ≈ -0.2486` (sign-correct).
- ON degenerate (no visited children): falls back to OFF — same action chosen.
- PV-preference smoke: in a drawish fixture (uniform priors, values=0), ON
  with c=0.45 concentrates more visits on the top PV root child than OFF.
- Flag plumbing: `mcts_picker(fpu_reduction_c=0.45)` is threaded through to
  the MCTSGame instance.
- Self-play guard: `selfplay_worker.py` exposes no `--fpu-reduction-c` and
  does not reference `fpu_reduction_c` anywhere.

## See also

- `derby-ehw` — eval-VCF overlay (tactical exact-solver counterpart).
- `derby-5xs` — `probe_100pct.py` (sims sweep harness; natural place to add
  an FPU c-axis next).
- KataGo (Wu 2019); LCZero defaults.
