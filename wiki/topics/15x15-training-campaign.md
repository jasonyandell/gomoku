# The 15×15 era — feasibility, port, and training campaign

**Status: HISTORICAL (2026-06-12 → 06-19).** This is the merged record of the
15×15 era: the feasibility case + board-size port (formerly a separate page, now
[archived verbatim](../_archive/topics/15x15-era-feasibility-and-plan.md)) and the
autonomous training campaign that ran on top of it. The era is **superseded by the
sound-world line** ([sound-world-recipe.md](sound-world-recipe.md)); its durable
findings are hoisted below. Epic: GitHub #21.

> **Where the era's open question landed.** The campaign's decisive open read —
> *"does the from-scratch net learn white-side defense, or reproduce the 0%-white
> hole?"* — was **answered downstream**: the sound-world recipe carried up to 13×13
> and produced a **structural white-defense negative** (white 0/20; black forces a
> VCT by ply ~9–13, so white's sharp-defense examples never enter the buffer). The
> white deficit is **recipe-deep, not warm-start-baked**. See
> [sound-world-recipe.md](sound-world-recipe.md) § 13×13 graduation (#113).

## Durable lessons (settled — the era's keepers)

1. **The 9×9 ceiling was dispatch-bound, not hardware-bound.** At the training wave
   size (wave=64), moving the *same* net from 9×9 → 15×15 (2.8× the spatial compute)
   costs **~nothing** (0.67 ms/wave either way) — per-kernel launch overhead sets
   the floor, and the GPU idled through the 9×9 era. A serious 15×15 net is
   affordable: 96×8 costs 2.32×, 128×10 costs 4.62× vs 9×9 production eval.
   The measured table (`scripts/bench_board_scaling.py`, 2026-06-12, torch 2.12.0
   MPS fp16, idle M5 Max, two runs stable within ~1%):

   | Net @ board | Params | Wave=64 (training regime) | Wave=512 (pure-gen regime) |
   |---|---|---|---|
   | 64×4 @ 9×9 (champ arch) | 317k | 94.2k evals/s (0.68 ms/wave) | 426k evals/s (1.20 ms/wave) |
   | 64×4 @ 15×15 | 415k | 95.7k evals/s — **0.98× (free)** | 191k evals/s (2.24×) |
   | 96×8 @ 15×15 | 1.45M | 40.6k evals/s (**2.32×**) | 51.1k evals/s (8.34×) |
   | 128×10 @ 15×15 | 3.08M | 20.4k evals/s (**4.62×**) | 22.5k evals/s (18.9×) |

   Wave=512 ratios are much worse (8.3×/18.9×): big-batch gen leaves
   dispatch-bound territory and pays real FLOP cost — the optimal wave size
   must be re-swept per board size; the 9×9 answer doesn't carry over.
2. **The 9×9 strength era is formally CLOSED.** The v8 champion + `--fpu-reduction-c
   0.45` went **43W-3L-74D over 120 games vs Rapfi** (Gomocup freestyle 2625) across
   100/500/1000 ms tiers — 3 losses total, never blown out. 9×9 freestyle is
   intrinsically drawish/near-solved; no headroom left to chase there. (Caveat: 2625
   is a 15×15/20×20 rating — a yardstick, not a 9×9 Elo label.)
3. **Cold-start fast-attack collapse is a SURVIVABLE transient for the v8+WDL
   recipe** — not a terminal trap. Warm-start SKIPS the wasted collapse-and-recover
   V, but is **not strictly required** with a WDL value head (full evidence in the
   dated correction below). The white/per-color H-L-D vs Rapfi is the decisive
   diagnostic, not survival-of-the-collapse.
4. **The operating doctrine (below) is reusable:** the two-lane GPU-serial model,
   the 5-minute loop manual, and the HALT/ESCALATE conditions carried forward into
   the research lab and later campaigns.
5. **FPU reduction does NOT transfer to 15×15** (9×9-champion-scoped; clean A/B gave
   no gain). Recorded negative — see [fpu-reduction-eval-lever.md](fpu-reduction-eval-lever.md).

*Companion (still LIVE):* [board-size-transfer-and-warm-start.md](board-size-transfer-and-warm-start.md)
(the proven cross-board warm-start mechanics + auto-graduating ladder),
[research-lab-charter.md](research-lab-charter.md) (the two-queue lab this ran inside).

## North star (as it stood)

Maximize 15×15 playing strength against the **external engine ladder** (SlowRenju
1857 → AlphaGomoku/MK 2256 → Rapfi capped-time → Rapfi full 2625), measured as
**Δelo/Δt** — slope vs a stable anchor, not a throughput proxy.

**Mandate (Jason, 2026-06-13):** "Take this all the way to the best player in the
world (or at least the best we can make)… Orchestrate background agents to preserve
context and start a loop every 5m." Autonomous, runs across context resets — **the
loop is the continuity mechanism.**

## The two lanes (hard constraint: GPU is serial)

- **GPU lane — serial, exactly one tenant.** One continuous, crash-resumable 15×15
  training run. Agents CANNOT hold the MPS lock → the GPU lane is never
  parallelized. Contestants are *swapped*, not stacked (the derby model). Never
  start a second GPU run while one is live; never barge in on a non-lab GPU tenant.
- **Everything-else lane — parallel via agents/workflows.** Code, design, analysis,
  eval-harness work. Each unit gets its own worktree + Reviewer gate + independent
  merge to `main`. Validated recipe improvements become eligible to *swap into* the
  GPU lane.

## Roadmap (as recorded; later superseded by the sound-world line)

1. **P4 — free-style 15×15 seed run (was IN PROGRESS).** v8 recipe (vcf +
   global-pool + value-discount-0.98 + gumbel), 64×4 net, warm-started from the 9×9
   champion tower. Free-style 15×15 is a first-player win (Allis/Wu) — a tactical
   foundation whose tower transfers to swap2/renju later.
2. **Capacity + search levers (derby).** Grow net (64×4 → 96×8 → 128×10 — the 9×9
   "bigger nets lose" verdict does NOT hold at 15×15), race the AZ-techniques
   backlog (WDL head #26, LCB root, variance-PUCT, moves-left, in-search VCF/VCT).
   Bit-packed buffer (#25) unlocks 3M+ positions.
3. **swap2 opening protocol (#22).** Balanced play → Rapfi's rated freestyle table.
4. **renju (#23) and/or larger board.** The Gomocup-competitive endgame.

## Feasibility & board-size port (merged — full detail archived)

The [archived feasibility page](../_archive/topics/15x15-era-feasibility-and-plan.md)
holds the measured scaling bench (`scripts/bench_board_scaling.py`), the phased plan,
and the one-shot execution of Phases 0–3 (2026-06-12). Headlines that survive:

- **Phase 0 — 9×9 champion certified externally:** 43W-3L-74D vs Rapfi (durable
  lesson 2 above). Formal close of the 9×9 strength era.
- **Phase 1 — rules variant DECIDED:** kept **free-style, free openings** so the port
  stayed pure parameterization. swap2 → #22, renju → #23 (deferred).
- **Phase 2 — board size is now process-level config** (`gomoku/board_config.py`:
  `--board-size` > `GOMOKU_BOARD_SIZE` env > default 9). Native C exts parameterized
  at compile time (`_*_native11/13/15`); checkpoints embed `board_size`; mixed-size
  resume refused. Full pytest green at 9 (608) + 15 (22); 9×9 byte-identical
  regression held.
- **Phase 3 — 15×15 plumbing smoke: GO.** `SMOKE15` ran the full loop end-to-end,
  122 epochs in 90 s, native exts in use, genuine 15×15 game lengths, resumable.
- **The perf levers predicted to flip at 15×15** (rejected at 9×9): fp16
  trainer+worker, ANE/CoreML workers under a heavy trainer, bit-packed buffer
  (near-mandatory at 15×15 memory scale), batched `state.apply` on GPU.

## The 5-minute loop — what to do each tick

Cadence ~5 min (270s wakeups to stay cache-warm). Order:

1. **Assert health.** Trainer PID alive? `pgrep -fl run_sweep|gomoku.train`. Tail
   the active `trainer.log`. If dead → **restart from the embedded-buffer
   `latest.pt`** (resumable slice; no cold refill). Non-lab GPU tenant → wait.
2. **Read progress (record ≠ report).** Latest epoch, `plies_mean`, policy/value
   loss, per-epoch wall (climbing? = the LF1 runaway trap), latest `eval` row. Append
   a one-line observation to the campaign log (via `scripts/lab_log.py` or the wiki).
3. **Keep it moving.** A queued improvement that landed on `main` AND passed gates
   AND (for recipe changes) shows positive smoke → swap it in at the next clean
   checkpoint. Plateaued contestant (no Δelo over a fair window) → swap by judgment.
   Unblocked code lane with no agent → dispatch one.
4. **Escalate only on HALT conditions.** Otherwise CONTINUE — the lab runs forever.

### HALT / ESCALATE conditions

- NaN loss, or `plies_mean` collapsing **with low value-loss** = fast-attack collapse
  (NOT the healthy absorption dip).
- Per-epoch cost climbing unboundedly (wave-tile → SGD → game-length positive
  feedback; the [perf-bench-vs-real-training-cost.md](perf-bench-vs-real-training-cost.md)
  runaway). Fix = fixed SGD cap decoupled from inflow.
- Repeated trainer crash (≥3) that `latest.pt` resume does not fix.
- Disk pressure from checkpoints/buffers, or a non-lab GPU tenant that won't clear.

## Guardrails (do NOT violate)

- One GPU run. Worktree per code unit; `merge --no-ff`; **never rebase**; push.
- Validate ingest/perf at **flood scale**, not CPU-sim (the cross-game-value trap,
  #7). Short evals are noisy — strength claims need the external ladder, enough
  games, explicit checkpoint/run IDs.
- Don't clean `checkpoints*/`, `sweep_*`, `wandb/` — evidence.
- Preserve Jason's work. File reusable findings back to the wiki + memory.

## Campaign log (compressed; newest last)

- **2026-06-13 — opened.** GPU idle, port + smoke done (P4 unblocked). Recon agents
  dispatched (seed-cell author + 15×15 training-path readiness audit).
- **2026-06-13 — recon cleared (GO-WITH-FIXES).** `G15-seed` cell landed (v8 recipe
  @ board 15). Smoke caught `--vcf-teacher` costing 3–9 s/game on the wide-open board
  → DROPPED from the cold seed (→ ~1.8 s/game). Launched cold (wandb `d6z4o5dw`).
- **2026-06-13 — cold run slid into fast-attack collapse** (plies 68→11, VL rising =
  expected cold chaos). Warm-start loader (`scripts/warmstart_15x15.py`) built +
  merged: **94.6% param transfer** from the 9×9 champion tower. **SWAPPED to
  warm-started** (wandb `qvr95npw`, `--resume g15_warmstart_seed.pt`) → plies ~85
  from epoch 0 (defended play), collapse skipped. *Durable lesson then:* cold-start
  collapse is real; warm-start is the remedy (later refined — see the dated
  correction).
- **2026-06-13 — warm run healthy + climbing.** By ~e93 plies stabilized ~30 (VL
  0.41→0.31, pl 1.7→1.63), anchored **elo ≈ 1253** (heuristic 60% / la2 75% / la4
  35%). External-ladder harness (`scripts/ladder_eval_15x15.py`) + `G15-vcf`
  contestant merged. **First Rapfi yardstick** (vs 2625, n=6/tier): warm net
  competitive + climbing — e180 50%/25% → e300 67%/67% → e503 67%/83% (200/1000 ms).
- **2026-06-13 — FPU does NOT transfer to 15×15** (frozen-ckpt A/B: off 50/83%, on
  33/83% = within noise). Recorded negative, deprioritized.
- **2026-06-13 — loss-bounce read correctly as BENIGN maturation** (pl rose to a
  higher OSCILLATING band ~1.4-1.5, plies stable ~30-40, vl steady, Rapfi strength
  rose through it). Structural signals override the raw loss number.
- **2026-06-13 — both capacity/strength levers merged** (Reviewer APPROVE,
  byte-identical-OFF): bit-packed replay buffer (#25, ~3M-buffer capability) + WDL
  value head (#26, `G15-wdl` contestant). **PHASE SHIFT:** base `G15-seed`
  PLATEAUING ~60-70% vs Rapfi (e300→e600 bounces 50-83%, mean ~67%, no upward
  trend). Contestants ready; NEXT was plateau-break (packed 3M buffer, net2net widen
  64×4→96×8) via GPU-serial derby.

## DATED CORRECTION (2026-06-19) — cold-start collapse is a SURVIVABLE TRANSIENT for the v8+WDL recipe, not a terminal trap

> Annotates — does **not** delete — the 2026-06-13 conclusion "cold-start fast-attack
> collapse is real at 15×15; warm-start is the remedy." That stands **as observed on
> its run** (the cold `G15-seed`, v8 recipe **no WDL head**: plies 68→11, VL rising,
> no recovery in-window before the swap). What the autolab's first 15×15 run shows is
> that a *different* recipe **survives the same collapse without intervention** — so
> "warm-start is required" was over-general; it was the fastest fix tried, not the
> only path through.

**Evidence (autolab `15x15-wdl` lane, cell `G15-wdl`, board 15, from scratch, no
warm-start, no vcf/defense teacher).** The from-scratch run went **through** the
cold-start fast-attack collapse and then **self-recovered**, with no teacher and no
warm-start:

| phase | epoch | plies | read |
|---|---|---|---|
| pre-collapse | ~21 | **69.5** | defended, random-ish opening play |
| collapse trough | ~65 | **9.2** | the fast-attack collapse (exactly as documented) |
| self-recovery | ~260–670 | **~35–40** (stable) | mid-game richness regrown — *without any teacher* |

**Why this is healthy recovery, not the death-tell** (cf.
[loss-floor-bouncing.md](loss-floor-bouncing.md) + the HALT conditions above):
- **WDL value-loss held ~0.81–0.89 the whole way** — it **never collapsed toward
  zero**. A value head saturating to ~0 is the terminal "confident-in-a-bad-fast-
  attack" tell (the white-defense death-spiral, vl→0.04–0.06); a value-loss that
  *stays up* through the plies trough is healthy maturation on hard positions.
- **Policy-loss fell monotonically 5.4 → 1.6** (initial **5.42 ≈ ln(225)** confirms
  a genuine 15×15 policy over 225 cells, not a degenerate head).

**Conclusion (revised, recipe-scoped):** with the **v8 recipe + WDL value head**, the
cold-start collapse is a **survivable transient**, not a terminal trap. Warm-start
**may not be strictly required for this recipe** — the WDL head appears to carry the
run through the trough that crashed the no-WDL cold seed. (Open: WDL specifically, or
just a longer patience window than the 06-13 cold run got before the swap?)

**CRITICAL CAVEAT — survival ≠ defense.** "Survived the collapse / recovered mid-game
richness" is **NOT** "learned white-side defense." The decisive open read was this
net's **white W-L-D vs Rapfi** (see
[white-side-defense-plan.md](white-side-defense-plan.md) §1B.2):
- reproduces the warm-started champion's **0/12-white** hole ⇒ the white deficit is
  **representational / recipe-deep** (the recipe, not the warm-start);
- defends **better** ⇒ **warm-start was baking in the attacker bias**.

**That probe was the era's next frontier read — and it was answered downstream**
(see the forward-link at top): the sound-world 13×13 graduation reproduced the
0%-white hole from *both* warm-start and from-scratch, confirming the white deficit
is **recipe-deep**. First 15×15 champion from this run: `15x15-wdl@0`, internal eval
**elo 1918** (first 15×15 number, **not** comparable to the 9×9 scale).
