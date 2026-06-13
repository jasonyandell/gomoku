# The 15×15 Era: Feasibility And Plan

**Date:** 2026-06-12. **Status:** proposal + measured feasibility evidence; no
port work started. **One-line thesis:** the 9×9 perf ceiling is a
*dispatch-bound small-model* ceiling, not a hardware ceiling — a 15×15 board
with a substantially bigger net costs only 2.3–4.7× at the production wave
size, so the M5 Max can train a 15×15 player in week-scale wall-clock, and
most of the perf levers that rejected at 9×9 were predicted to pay in exactly
this heavier regime.

Companion pages: [mcts-perf-ceiling.md](mcts-perf-ceiling.md),
[m5-max-fp16-and-throughput-regimes.md](m5-max-fp16-and-throughput-regimes.md),
[m5-max-cross-engine-coupling.md](m5-max-cross-engine-coupling.md),
[external-engine-baselines.md](external-engine-baselines.md),
[../sources/gomocup-az-techniques-2026-05-27.md](../sources/gomocup-az-techniques-2026-05-27.md),
[../ops/research-board.md](../ops/research-board.md).

## 1. Why now: the 9×9 strength frontier is closing

Evidence as of 2026-06-12 (see [../ops/research-board.md](../ops/research-board.md)
and [training-run-lineage.md](training-run-lineage.md)):

- **The ladder is swept.** The v8 champion
  (`sweep_runs/derby_v8/_peaks/mate-discount/peak.pt`, 64ch×4blk,
  vcf + global-pool + value-discount-0.98 + gumbel-m16, elo ~1811 H2H at
  epoch 2848) with the eval-only flag `--fpu-reduction-c 0.45` achieves
  100% black-wins / 0% white-loss against heuristic, lookahead2, lookahead4,
  and lookahead8; lookahead6 sits at 86–90% / 0% (distance ≤ 0.14). The
  depth-4 loss-tail (#18's original motivation) was an aggregation artifact,
  not a per-checkpoint property.
- **Net capacity is not binding at 9×9.** Derby v9's verdict
  (2026-05-27): v8-champion +151, v9-small +16, v9-medium −28, v9-large −139
  (mean-centered H2H). Bigger nets *lose* at 9×9.
- **9×9 free-style is intrinsically drawish and near-solved in practice.**
  More compute poured into 9×9 buys little Δelo — the constraint is the
  board, not aug/s.

Meanwhile the stated goal (memory + [m5-max-as-mainframe.md](m5-max-as-mainframe.md))
has always been: 9×9 is the proving ground; the destination is 15×15 and
Gomocup-class opposition. The blocker was the assumption that the Mac maxed
out at 9×9. Section 2 tests that assumption directly.

## 2. The measurement: board/net scaling on MPS

**Cell:** `python scripts/bench_board_scaling.py` (committed with this page).
torch 2.12.0, MPS, fp16 eval, eval-mode unfused BN (slightly conservative vs
the fused production path), 20-wave warmup, 6s timed loops,
`torch.mps.synchronize()` bracketing. Run 2026-06-12 on the idle M5 Max
(no derby/training tenant). Two runs, numbers stable within ~1%.

| Net @ board | Params | Wave=64 (training regime) | Wave=512 (pure-gen regime) |
|---|---|---|---|
| 64×4 @ 9×9 (champ arch) | 317k | 94.2k evals/s (0.68 ms/wave) | 426k evals/s (1.20 ms/wave) |
| 64×4 @ 15×15 | 415k | 95.7k evals/s — **0.98× (free)** | 191k evals/s (2.24×) |
| 96×8 @ 15×15 | 1.45M | 40.6k evals/s (**2.32×**) | 51.1k evals/s (8.34×) |
| 128×10 @ 15×15 | 3.08M | 20.4k evals/s (**4.62×**) | 22.5k evals/s (18.9×) |

Readings:

1. **The dispatch-bound hypothesis is confirmed in the flesh.** At wave=64,
   moving the same net from 9×9 to 15×15 (2.8× the spatial compute) costs
   *nothing* — 0.67 ms/wave either way. The GPU has been idling through the
   9×9 era; per-kernel launch overhead, not arithmetic, sets the floor.
   This is the same mechanism behind
   [m5-max-fp16-and-throughput-regimes.md](m5-max-fp16-and-throughput-regimes.md)
   Finding 1 and the ~1.9–2.1 ms MPS forward latency in
   [mcts-perf-ceiling.md](mcts-perf-ceiling.md).
2. **A serious 15×15 net is affordable at the training wave size.** 96×8
   (4.6× params on 2.8× board area) costs 2.32× vs today's production eval.
   Even 128×10 is only 4.62×.
3. **Wave=512 ratios are much worse (8.3×/18.9×)** — the big-batch pure-gen
   regime leaves dispatch-bound territory and starts paying real FLOP/
   bandwidth cost. Implication: at 15×15 the optimal wave size question must
   be re-swept; the 9×9 answer (bigger V is near-free) does not carry over.
4. **Receptive-field note:** the 64×4 tower (stem + 8 convs, all 3×3) has a
   19×19 receptive field — it *covers* 15×15 with little margin, which is
   what makes the warm-start in §4 Phase 3 plausible. Global pooling (already
   a derby winner) provides whole-board context regardless.

## 3. Feasibility envelope (back-of-envelope, to be validated by smoke)

Production 9×9 reference: R-TRAIN-WL5 live training = 3,297.6 aug/s =
14.07 games/s ([../ops/best-cells.md](../ops/best-cells.md)).

For a 96×8 net at 15×15, the multiplicative guesses:

| Factor | Multiplier | Basis |
|---|---|---|
| NN eval cost | ~2.3× | measured (table above, wave=64) |
| Sims per move | ~2× | 225 actions vs 81; guess pending sims-sweep |
| Plies per game | ~1.7× | 15×15 games run longer; guess |
| **Total per game** | **~8×** | |

→ ~1.5–2 games/s live, ~150k games/day, so a WL5-scale 1M-game run is
**roughly a week of wall-clock**, not months. A 128×10 net roughly doubles
that. These are planning numbers, not commitments.

**Why this bench must not be over-trusted** (the repo's own scars):

- [perf-bench-vs-real-training-cost.md](perf-bench-vs-real-training-cost.md):
  a +152.9% bench (R-TRAIN-LEAN) turned into an unbounded runaway in the real
  loop. Cold/eval-only benches miss equilibrium dynamics.
- Ingest-flooding lesson (memory + TRAINING_WIKI): CPU-scale tests passed 3×,
  failed live flood 3×. **The go/no-go gate is a live `run_sweep` smoke
  slice at 15×15, not this table.**
- Contention tax: live training pays ~30.8% vs pure-gen at 9×9
  ([../ops/best-cells.md](../ops/best-cells.md)), and the Lpwr2b finding
  ([m5-max-cross-engine-coupling.md](m5-max-cross-engine-coupling.md)) says
  throttle scales with the GPU trainer's *memory working-set* — a bigger
  15×15 trainer will throttle workers harder. Unmeasured at 15×15.
- Native-MCTS tree cost at 225 actions (selection, expansion, D4 symmetry
  ops, hash tables) is not in this bench at all.

**Memory pressure makes bit-packing near-mandatory.** A 15×15 position is
~2.8× a 9×9 one (~15 KB/pos unpacked at current encoding); a 3M-position
buffer (the [[project-buffer-undersized]] recommendation) would be ~45 GB
unpacked. The bit-packing plan
([buffer-bit-packing.md](buffer-bit-packing.md), ~17× → ~1 KB/pos at 15×15)
turns that into ~3 GB. What was an optional 9×9 lever is a prerequisite here.

**Perf levers that rejected at 9×9 and were predicted to flip in this regime**
(re-open after the port, in this order):

1. **fp16 trainer + worker everywhere** — Finding 1 territory; never measured
   for training forward+backward.
2. **ANE/Core ML workers under a heavy GPU trainer** — the un-run
   `L09i-fix-load` lane ([coreml-design-envelope-and-our-fit.md](coreml-design-envelope-and-our-fit.md));
   ANE's contention-immunity only pays when the GPU trainer is actually
   heavy, which a 96×8/128×10 15×15 trainer finally is.
3. **Bit-packed buffer** — prerequisite, see above.
4. **Batched `state.apply` on GPU** (the deferred "next 2×",
   [mcts-perf-ceiling.md](mcts-perf-ceiling.md)) — bigger boards raise the
   payoff of moving state ops off the CPU.

## 4. The plan

Phases are ordered; each has a gate. GPU-required items go through the lab's
GPU queue ([../ops/gpu-queue.md](../ops/gpu-queue.md)); code-only items are
normal worktree work. Effort in Opus-minutes per the conventions page.

### Phase 0 — Certify the 9×9 champion externally (GPU, ~30 Opus-min)

Run the v8 champion + `--fpu-reduction-c 0.45` vs Rapfi
(`scripts/eval_vs_rapfi.py`, wrapper `gomoku/external_engine.py`, binary
built + smoke-tested) at 2–3 time controls, ≥40 games each, on 9×9. The only
existing datapoint is a 4-game smoke from a *WL5 seed* (2W-0L-2D at 100 ms) —
not evidence. **Deliverable:** an external anchor for the lineage and the
formal close-out of the 9×9 strength era. **Gate:** none; pure measurement.

### Phase 1 — Decide the rules variant (HUMAN-GATED, Jason's call)

Free-style 15×15 is solved (black wins, Allis/Wu), so pure free-style is the
wrong target. Options:

| Option | Pro | Con |
|---|---|---|
| **Renju** | The balanced classical game; Gomocup table exists | Forbidden-move rules (3-3, 4-4, overline for black) are real game-logic work in `game.py` + native ext |
| **Free-style + swap2 opening** | Game logic unchanged (only opening protocol); Gomocup freestyle is played this way | Swap2 changes the training distribution (must train both colors from swapped openings); protocol work in match/eval |
| **Standard gomoku (exactly-5)** | Small rule delta (overlines don't win) | Still black-favored without an opening rule |

The survey pages lean renju ([m5-max-as-mainframe.md](m5-max-as-mainframe.md)
line ~156); swap2-freestyle is the cheaper first step and matches Rapfi's
strongest table (2625 elo is its *freestyle* rating). **Recommendation:
free-style + swap2 first** (cheapest path to playing Rapfi on its rated
table), renju as a later variant. **Gate:** Jason decides; shapes Phases 2–5.

### Phase 2 — Port the codebase to parameterized board size (code-only, ~120 Opus-min)

- `gomoku/game.py:22` `BOARD_SIZE`/`N_ACTIONS` → constructor/config
  parameter threaded through state, D4 augmentation (already
  size-agnostic `np.rot90` code), win detection (direction-agnostic).
- `gomoku/model.py`: policy head 81→N output, stem/spatial sizing, config
  presets for 15×15 nets (64×4 warm-start, 96×8, 128×10).
- **Verify the native extensions** (`state_ops`, native MCTS) — board size
  assumptions unverified; recompile/parameterize; keep the
  `GOMOKU_DISABLE_NATIVE_*` A/B escapes working at both sizes.
- Checkpoint format: embed board size; refuse mixed-size resume.
- Tests: full suite parameterized to run at 9 and 15; 9×9 byte-identical
  behavior is the regression gate (existing checkpoints must still load and
  reproduce eval results).
- VCF/VCT solver and threat semantics at 15×15 (overline handling per the
  Phase 1 decision).

**Gate:** `pytest` green at both sizes + 9×9 champion evals reproduce.

### Phase 3 — Smoke slice + sweep the new regime (GPU, ~60 Opus-min)

- 60–90s `run_sweep --max-wall-secs` smoke cell at 15×15 with the 64×4
  warm-start net: does the loop run, and what is *live* aug/s and games/s?
  This is the go/no-go on §3's envelope.
- Mini-sweeps (smoke-first doctrine): wave size at 15×15 (the 9×9 V=512
  answer does not carry over, §2 reading 3), sims-per-move, worker count.
- **Warm-start check:** initialize the 15×15 64×4 net from the 9×9
  champion's conv tower (fully convolutional except the policy fc; global
  pooling transfers). Measure whether early self-play shows transferred
  tactics vs a fresh init (cheap A/B: plies_mean + first eval vs heuristic).

**Gate:** live games/s within ~2× of the §3 envelope → proceed. Worse →
re-open the perf levers (§3 list) *before* committing to long runs.

### Phase 4 — First real 15×15 training run (GPU, week-scale wall-clock)

- Recipe: carry the v8 winners forward as the seed (vcf + global-pool +
  value-discount + gumbel; FPU 0.45 as eval config).
- **Add the WDL value head as the first new contestant** — the keystone
  untried lever per
  [../sources/gomocup-az-techniques-2026-05-27.md](../sources/gomocup-az-techniques-2026-05-27.md);
  draw/contempt structure matters more once swap2/renju balance the game.
- Buffer: 3M+ positions ⇒ bit-packing lands first (Phase 3.5, code-only,
  has an existing cheap-test gate in
  [buffer-bit-packing.md](buffer-bit-packing.md)).
- Net-growth curriculum: start 64×4 (free at 15×15), grow to 96×8 when
  self-play plateaus (net2net-style widen/deepen or fresh-train from the
  archive — decide then; 9×9's "bigger nets lose" verdict does NOT carry
  over because the 15×15 game is not near-solved).
- Baseline ladder for the era (replaces lookahead-N as the primary):
  SlowRenju (1857) → AlphaGomoku/MK (2256) → Rapfi capped-time → Rapfi full
  (2625). Lookahead-N stays as a cheap smoke probe.

**Gate:** Δelo/Δt vs the external ladder, measured per
[wall-clock-to-elo-metric.md](wall-clock-to-elo-metric.md).

### Phase 5 — Derby at 15×15 + perf-lab reopen (ongoing)

- Re-open the rejected-at-9×9 perf lanes in the heavy regime (§3 list,
  ANE `L09i-fix-load` first — it was explicitly parked waiting for a heavy
  trainer).
- Derby vNext on the 15×15 board: same operating model
  ([../topics/research-loop.md](research-loop.md), derby skills), contestants
  from the AZ-techniques survey backlog (LCB root, variance-PUCT, moves-left
  head, in-search VCF).

## 5. Open questions

1. **Rules variant** — Phase 1, human-gated (the recommendation is
   swap2-freestyle first).
2. **Sims budget at 225 actions** — the ~2× in §3 is a guess; needs the
   Phase 3 sweep. Gumbel root sampling (already a winner) may keep it lower
   than feared.
3. **Net-growth mechanics** — warm-start widen/deepen vs fresh-train at the
   plateau; decide on Phase 4 evidence.
4. **Trainer working-set vs worker throttle at 15×15** — Lpwr2b says shrink
   the trainer's memory footprint, not its FLOPs; the 15×15 trainer is the
   first real test of that lever's value.
5. **Swap2 in the training loop** — if Phase 1 picks swap2: train with the
   opening protocol in self-play, or train free openings and bolt swap2 on
   at match time? (Rapfi-side precedent suggests the latter is weak.)

## 6. Decision record

- 2026-06-12 — Page created with the scaling bench evidence
  (`scripts/bench_board_scaling.py`); plan proposed, no port work started.
  GPU idle (derby v8 concluded, v9-medium errored + parked).
