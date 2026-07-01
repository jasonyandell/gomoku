# The VCT-defense aux head — a working sensor with no actuator (#103)

> **RESULT (2026-07-01): the supervised VCT-defense aux head reliably learns the defensive
> REPRESENTATION, but nothing so far makes the POLICY act on it.** Two experiments — a
> from-scratch 9×9 moonshot (wandb `8mtowemb`) and a Bruce/idx-2 warm-start pivot (wandb
> `zrjfwny2`) — both had the head learn cleanly (`train/vct_loss` → ~0.03) while self-play
> behavior did **not** change: the from-scratch run stayed pinned to the #101 attractor
> (plies flat ~9-10 for 1152 epochs), and the warm-start pivot's self-play *policy drifted*
> (`loss/policy` 1.93→2.62, plies collapsed 11.6→9.6 — the terminus/narrow-opening regime
> specializing the champion). **"Frankenstein + aux head" is not the recipe** (Jason). The
> head is a working **sensor** with **no actuator**: it forms the percept, but neither
> from-scratch self-play (no opponent to punish weak defense) nor a frozen-champion
> warm-start makes the policy defend on it.

This is the #103 executes-#102 arc. It is the direct sequel to the VCT-terminus self-play
result ([vct-terminus-selfplay-result.md](vct-terminus-selfplay-result.md), #100/#101):
that page's long-run coda argued the self-play defensive ceiling is **structural**, and the
way past it is an **opponent-independent** supervised defensive gradient — the VCT-defense
aux head. #103 built that head and ran it two ways. Both failed to make the net *defend* —
but instructively, and the failure sharpens the diagnosis rather than refuting it.

## The head (#102 design, built + committed on main)

A per-cell **"VCT-blunder map"** defense aux head, supervised by the GPU mega VCT solver
(`solve_vct_mega_bb`, [mega-vct-solver.md](mega-vct-solver.md)): for each legal move, does
playing it walk the **side-to-move** into a forced **opponent** VCT? The defense label is
computed by **escape-search over every legal move** (Tier 2): for each of L legal moves,
solve whether the opponent then has a VCT; a move is labeled "lost" iff *all* of its
children lose. `--aux-vct-weight 0.1`; **default weight 0 is byte-identical** (verified: only
`vct_*` keys change), the standard ownership-head-pattern no-op invariant.

**Warm-start head layering — the `force_aux_vct` splice** (the code that made Experiment B
possible): `load_checkpoint(..., force_aux_vct=True)` (`gomoku/model.py:635`, `:673`) forces
`aux_vct=True` in the saved config so the model is *built with* the aux head even when the
checkpoint predates it; the core loads strict and the freshly-initialized `vct_*` params
splice in via the same path as the swap2 choice head. Off (default) ⇒ byte-identical. Wired
into the trainer at `gomoku/train.py:1417-1422` (`force_aux_vct=vct_on` on `--resume`).

## Experiment A — the 9×9 from-scratch moonshot (wandb `8mtowemb`, retired e1152)

**Recipe.** From-scratch 9×9, VCT-terminus gate (`--vct-terminus --vct-terminus-budget 50`,
games end at the first cap50 VCT), the VCT-defense head (`--aux-vct-weight 0.1`, full
escape-search over every legal move), + the surviving Bruce levers (value-discount 0.98,
`global_pool`, WL2 stack — ema_tau 0.99 / grad-accum 4 / opp-mix 0.4·0.1, 64 SGD-steps/epoch,
64×4, 1.5M buffer, sims=100). **NO gumbel** (the terminus code raises `NotImplementedError` on
gumbel+terminus). **NO `--vcf-teacher`** — its CPU solver is **RETIRED** (`CpuSolverRetired` at
runtime; the GPU mega solver is the oracle now) and it is *inert* under the terminus anyway
(VCF ⊆ VCT, so the terminus preempts it). Run dir `~/data/moonshot-103/`.

**Result — the head learns; self-play does not move.**
- **The head learns the representation:** `train/vct_loss` **0.60 → 0.03**, `mask_frac` ~0.9.
  The supervised defensive percept forms cleanly.
- **Self-play behavior does NOT change:** `selfplay/plies_mean` flat at **~9-10 for all 1152
  epochs** — the exact **#101 attractor**. The from-scratch VCT-terminus player, *with* the
  defensive representation, still converges to the same ~9-ply rush-to-VCT game.

**Conclusion.** The supervised defense gradient forms the **percept** but self-play offers no
opponent strong enough to make the **policy** act on it. The #101 ceiling holds **even with the
representation present** — which is the load-bearing negative: it rules out "the net simply
can't *see* the blunder" as the cause of the self-play defensive ceiling. It can see it; it
just has no reason (no punisher) to avoid it. This motivated concentrating the gradient on a
**strong** net at a **hard fixed position** — Experiment B.

## Experiment B — the Bruce/idx-2 pivot (wandb `zrjfwny2` from e613, retired e862 ≈ 257 pivot epochs)

**Recipe.** When A didn't "dig out", pivot: **warm-start from Bruce** (the 128×10 15×15
champion, `g15_128x10_bigbuf_e588_best.pt`, ep 605) + **layer the VCT-defense head on** (the
`force_aux_vct` splice above) + **restrict self-play to the idx-2 opening**
(`GOMOKU_DROP_OPENERS=0,1,3,4,5,6,7,8` — the single white-to-move "Bruce-Lee board") +
VCT-terminus. Idea: concentrate the supervised defense gradient on Bruce's *measured*
white-defense wound (the idx-2 position where the champion reads black ~42% / white 0/12 vs
Rapfi — [white-side-defense-plan.md](white-side-defense-plan.md)).

**Result — the head learns again; the policy drifts hard.**
- **The head learns:** `train/vct_loss` **0.52 → 0.026** (loaded clean, no arch mismatch —
  the `force_aux_vct` splice worked).
- **The self-play POLICY drifts:** `loss/policy` **1.93 → 2.62** (rising, not falling) and
  `selfplay/plies_mean` **collapses 11.6 → 9.6** — the terminus attractor, reached *even from a
  128×10 champion*. The terminus + narrow-single-opening regime **specializes / erodes** the
  champion's general play, pulling it toward the same short rush-to-VCT game.

## The critical eval nuance — the idx-2 gate is SATURATED (do not misread this)

We ran the idx-2 verdict eval (`gomoku.rapfimine.eval_idx2`, **n=48, sims=160,
`GOMOKU_BOARD_SIZE=15`**) on **both** the pivot EMA checkpoint **and** frozen Bruce, at
identical settings:

| net | idx-2 vs Rapfi (n=48) | black | white |
|---|---|---|---|
| pivot EMA (`zrjfwny2`) | **0/48** | 0/24 | 0/24 |
| **frozen Bruce** (identical settings) | **0/48** | 0/24 | 0/24 |

**The eval at sims=160 is SATURATED — Rapfi crushes BOTH nets.** It does **not** discriminate,
and therefore it does **NOT** show that the pivot degraded Bruce. **Do not write "the pivot
degraded the champion black 42%→0"** — that is false; frozen Bruce *also* scores 0/48 here. The
wiki's "Bruce black ~42% / white 0/12" number comes from a **different eval config** (a
stronger-net / higher-think-time setup, [white-side-defense-plan.md](white-side-defense-plan.md)
§1B.2, n=24 vs Rapfi 5s/move); it is not comparable to this sims=160 idx-2 gate.

**The real evidence that "it fell apart" is the self-play POLICY DRIFT** (`loss/policy` up,
`plies` collapsed) — an internal training-dynamics signal — **not** the Rapfi eval. A clean
strength-delta between the pivot and frozen Bruce would need a **higher-sim** eval (one that
gets off this saturation floor) or a **direct H2H** between the two nets. **Neither was run —
the experiment was abandoned at the pivot stage.** So we can honestly say the self-play policy
eroded; we **cannot** put a number on how much strength (if any) the pivot cost the champion.

## Synthesis — the recurring lesson across both experiments

**The VCT-defense aux head reliably learns the defensive REPRESENTATION, but nothing so far
makes the POLICY act on it.** Two independent regimes, same outcome:
- **from-scratch self-play (A):** the head forms the percept; the policy ignores it because
  self-play has no opponent strong enough to *punish* weak defense (the #101 structural
  ceiling, now shown to survive even when the net can *see* the blunder);
- **frozen-champion warm-start (B):** the head forms the percept; the terminus + narrow-opening
  regime just *specializes* the champion's policy (drift), rather than teaching it to defend.

**"Frankenstein + aux head" is not the recipe** (Jason's words). The head is a working
**sensor** with **no actuator yet** — it maps the board to a blunder-map, but that map never
reaches the move-selection.

## What to try next (open directions, evidence-tied)

The through-line is: **target the POLICY directly, not just an aux head.**
- **The escape-search as a defensive POLICY target** — mask/penalize the blunder moves the head
  already identifies, stamping them onto the *policy* head (cf. the #43 saving-move-on-policy
  lever in [white-side-defense-plan.md](white-side-defense-plan.md) §I2, which was healthy but
  drowned on data density — the aux-head escape-search is a denser, board-wide version of the
  same signal).
- **Use the defense head at MCTS *inference* time** — prune/deprioritize blunder moves in the
  tree, so the head becomes an actuator on move selection without waiting for a self-play
  gradient to internalize it.
- **A curriculum / opponent that actually forces defense** — the missing punisher; without one,
  no aux head changes the self-play policy (the #100/#101 finding, restated).

The sensor works. The open problem is wiring it to an actuator.

## Provenance / reproduce

- **Head + splice + tests:** merged to main during #103 (`gomoku/model.py` `force_aux_vct`
  path :635/:673, `gomoku/train.py:1417-1422`); the aux head + tier-configurable defensive
  labeler + no-op invariant test.
- **Experiment A:** `moonshot` cell (from-scratch 9×9, terminus budget 50, `--aux-vct-weight
  0.1` full escape-search, Bruce levers minus gumbel/vcf-teacher). wandb **`8mtowemb`**
  (`jasonyandell-forge42/gomoku`), retired e1152. Run dir `~/data/moonshot-103/`.
- **Experiment B:** `moonshot-bruce-idx2` cell (resume `g15_128x10_bigbuf_e588_best.pt` +
  `force_aux_vct` + `GOMOKU_DROP_OPENERS=0,1,3,4,5,6,7,8` + terminus). wandb **`zrjfwny2`**
  (continues Bruce's run from ~e613), retired e862.
- **Eval:** `GOMOKU_BOARD_SIZE=15 uv run python -m gomoku.rapfimine.eval_idx2 --checkpoint
  <ckpt> --n-games 48 --sims 160` (EMA `worker_weights.pt`; the #100 rule). Reads 0/48 on both
  the pivot and frozen Bruce — **saturated, non-discriminating** at these sims.

## Cross-refs

- [vct-terminus-selfplay-result.md](vct-terminus-selfplay-result.md) — #100/#101, the terminus
  result + the structural-self-play-ceiling coda this head was meant to break.
- [white-side-defense-plan.md](white-side-defense-plan.md) — the white-defense wound (#43
  saving-move-on-policy, the idx-2 black ~42% / white 0/12 measurement from a *different*
  eval config), the natural home for the "target the policy" next step.
- [idea-pile.md](idea-pile.md) #11 — the seek-VCT / terminus lineage.
- [mega-vct-solver.md](mega-vct-solver.md) — the GPU oracle that supervises the head.
- `TRAINING_WIKI.md` 2026-07-01 (#103) — the chronological run record (both runs + the
  eval-saturation catch).
