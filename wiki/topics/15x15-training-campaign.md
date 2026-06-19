# 15×15 Training Campaign — Operating Plan

**Started:** 2026-06-13. **Mandate (Jason):** "Take this all the way to the
best player in the world (or at least the best we can make). You're clear to do
anything you think will help. Orchestrate background agents to preserve context
and start a loop every 5m to check on things and keep them moving." Autonomous,
runs across context resets and Jason's 5h session limit — **the loop is the
continuity mechanism.**

This is the campaign-level companion to
[15x15-era-feasibility-and-plan.md](15x15-era-feasibility-and-plan.md) (the
why + the port) and [research-lab-charter.md](research-lab-charter.md) (the
two-queue lab it runs inside). It is the **loop's per-tick manual** and the
**north-star record**. Epic: GitHub #21.

## North star

Maximize 15×15 playing strength against the **external engine ladder**
(SlowRenju 1857 → AlphaGomoku/MK 2256 → Rapfi capped-time → Rapfi full 2625),
measured as **Δelo/Δt** (the project metric — slope vs a stable anchor, not a
throughput proxy). The 9×9 era is closed (champion 43W-3L-74D vs Rapfi 2625);
15×15 is where "strong player" has room to exist and where the unused perf
levers pay.

## The two lanes (hard constraint: GPU is serial)

**GPU lane — serial, exactly one tenant.** One continuous, crash-resumable
15×15 training run. Agents CANNOT hold the MPS lock, so the GPU lane is never
parallelized — it is a single supervised background process. Contestants are
*swapped*, not stacked (the derby model). Never start a second GPU run while
one is live; never barge in on a non-lab GPU tenant.

**Everything-else lane — parallel via agents/workflows.** All code, design,
analysis, and eval-harness work. Each unit gets its own worktree, a
Reviewer gate, and merges to `main` independently. Validated training-recipe
improvements then become eligible to *swap into* the GPU lane.

## Roadmap (revise as evidence lands)

1. **P4 — free-style 15×15 seed run (IN PROGRESS).** Carry the v8 recipe
   forward (vcf + global-pool + value-discount-0.98 + gumbel), 64×4 net,
   warm-started from the 9×9 champion tower if available. Validates the whole
   pipeline at 15×15 and builds tactical strength. Free-style 15×15 is a
   first-player win (Allis/Wu) — the net will learn forcing attacks; that's a
   fine tactical foundation whose tower transfers to swap2/renju later.
2. **Capacity + search levers (derby).** Grow net (64×4 → 96×8 → 128×10 — at
   15×15 the 9×9 "bigger nets lose" verdict does NOT hold, the game isn't
   near-solved) and race the AZ-techniques backlog: **WDL value head** (the
   keystone untried lever, #26), LCB root selection, variance-PUCT,
   moves-left head, in-search VCF/VCT. Bit-packed buffer (#25) unlocks 3M+
   positions (prerequisite at 15×15 memory scale).
3. **swap2 opening protocol (#22).** Balanced play → benchmark on Rapfi's
   rated freestyle table on its own terms.
4. **renju (#23) and/or larger board (19×19/20×20).** The Gomocup-competitive
   endgame.

## The 5-minute loop — what to do each tick

Cadence ~5 min (use 270s wakeups to stay cache-warm). Order:

1. **Assert health.** Is the trainer PID alive? `pgrep -fl run_sweep|gomoku.train`.
   Tail the active `trainer.log`. If dead/crashed → **restart from the
   embedded-buffer `latest.pt`** (the run is a resumable slice; no cold
   refill). If a *non-lab* GPU tenant appeared → wait, don't barge in.
2. **Read progress (record ≠ report).** Latest epoch, `plies_mean`, policy/
   value loss, per-epoch wall (is it climbing? = the LF1 runaway trap), latest
   `eval` row. Append a one-line observation to the campaign log section below
   via the normal wiki/commit flow or `scripts/lab_log.py`.
3. **Keep it moving.** If a queued improvement has landed on `main` AND passed
   its gates AND (for recipe changes) shows positive smoke signal → swap it
   into the GPU lane at the next clean checkpoint (stop → adjust cell/resume →
   relaunch). If the current contestant has plateaued (no Δelo over a fair
   window vs the ladder) → swap by judgment. If the everything-else queue has
   an unblocked code lane and no agent on it → dispatch one.
4. **Escalate only on HALT conditions** (below). Otherwise CONTINUE — the lab
   is designed to run forever; do not pre-emptively halt.

### HALT / ESCALATE conditions (from the charter)

- NaN loss, or `plies_mean` collapsing **with low value-loss** = fast-attack
  collapse (not the healthy absorption dip — see
  [feedback-absorption-phase] memory). 
- Per-epoch cost climbing unboundedly (wave-tile → SGD → game-length positive
  feedback; the `perf-bench-vs-real-training-cost.md` runaway). Fix = fixed
  SGD cap decoupled from inflow.
- Repeated trainer crash (≥3) that `latest.pt` resume does not fix.
- Disk pressure from checkpoints/buffers, or a non-lab GPU tenant that won't
  clear.

## Guardrails (do NOT violate)

- One GPU run. Worktree per code unit; `merge --no-ff`; **never rebase**; push.
- Validate ingest/perf at **flood scale**, not CPU-sim (the cross-game-value
  trap, #7). Short evals are noisy — strength claims need the external ladder,
  enough games, explicit checkpoint/run IDs.
- Don't clean `checkpoints*/`, `sweep_*`, `wandb/` — evidence.
- Preserve Jason's work. File reusable findings back to the wiki + memory.

## Campaign log (newest last)

- 2026-06-13 — Campaign opened. GPU idle, port + smoke done (P4 unblocked).
  Recon agents dispatched (seed-cell author + 15×15 training-path readiness
  audit) to gate the multi-day GPU commitment before launch.
- 2026-06-13 — Recon cleared (GO-WITH-FIXES; the one blocking bug, cross-game
  key truncation at 15×15, is inert because G15-seed doesn't use that lever).
  `G15-seed` cell landed (v8 recipe at board 15). Smoke caught `--vcf-teacher`
  costing 3–9 s/game on the wide-open 15×15 board (starves gen) → DROPPED it
  from the cold-start seed (→ ~1.8 s/game). Launched cold (wandb `d6z4o5dw`).
- 2026-06-13 — Cold run slid into the **fast-attack collapse** (plies 68→11,
  VL rising = not terminal, expected cold chaos). Warm-start loader
  (`scripts/warmstart_15x15.py`) built + merged: 94.6% param transfer from the
  9×9 champion tower. **SWAPPED the run to warm-started** (wandb `qvr95npw`,
  `--resume sweep_runs/g15_warmstart_seed.pt`). Result: plies ~85 from epoch 0
  (defended play), collapse skipped. **Durable lesson: cold-start fast-attack
  collapse is real at 15×15; warm-start is the remedy.**
- 2026-06-13 — Warm run healthy + climbing: by epoch ~93 plies stabilized ~30
  (efficient, NOT collapsing — VL declining 0.41→0.31, pl 1.7→1.63), anchored
  **elo ≈ 1253** (heuristic 60% / lookahead2 75% / lookahead4 35%). The
  anchored ladder still gives signal at this strength; it will saturate as the
  net crushes la4/la6 — that's when the external Rapfi ladder takes over.
  Improvement Workflow `wx2qh95qd` in flight (external-ladder eval + tuned
  vcf-teacher `G15-vcf` cell), Reviewer-gated.
- 2026-06-13 — External-ladder eval harness (`scripts/ladder_eval_15x15.py`,
  CPU-default) + `G15-vcf` contestant merged. **First Rapfi yardstick** (vs
  Gomocup-2625, 15×15 freestyle, n=6/tier): the warm net is competitive and
  climbing — e180 50%/25% → e300 67%/67% → e503 67%/83% (200ms/1000ms).
- 2026-06-13 — **FPU eval-lever does NOT transfer to 15×15** (clean frozen-ckpt
  A/B: FPU-off 50/83%, FPU-on 33/83% — within noise / no gain). Unlike the
  mature 9×9 champion. Recorded negative; deprioritized.
- 2026-06-13 — Loss-bounce interpreted correctly: pl rose to a higher BAND
  (~1.0→~1.4-1.5) but OSCILLATING not climbing, plies stable ~30-40, vl steady,
  and Rapfi strength rose through it → BENIGN maturation (training on harder
  positions), not collapse. The structural signals (plies/vl/external strength)
  override the raw loss number.
- 2026-06-13 — **Both capacity/strength levers merged** (Reviewer APPROVE,
  byte-identical-OFF verified): bit-packed replay buffer (#25, enables a ~3M
  15×15 buffer in a few GB; flood-scale validation still pending before the
  live run adopts it) and the WDL value head (#26, `G15-wdl` contestant; the
  head impl was already on main, this added the cell + warmstart head-swap).
  **PHASE SHIFT:** base `G15-seed` appears to be PLATEAUING ~60-70% vs Rapfi
  (e300→e600 win-rate bounces 50-83%, mean ~67%, no clear upward trend; rapid
  early gains over). Contestants now ready (`G15-vcf`, `G15-wdl`) + bigger-buffer
  capability landed. NEXT: plateau-break — flood-smoke the packed 3M buffer,
  and/or net2net widen 64×4→96×8 (capacity is the likely 15×15 ceiling;
  net2net warm-starts to avoid cold-start collapse), then a GPU-serial derby
  (swapped contestants, never 2 concurrent runs). Base keeps training as the
  control meanwhile (healthy, not regressing).

## DATED CORRECTION (2026-06-19) — cold-start collapse is a SURVIVABLE TRANSIENT for the v8+WDL recipe, not a terminal trap (warm-start may not be strictly required)

> Annotates — does **not** delete — the 2026-06-13 conclusion "**cold-start
> fast-attack collapse is real at 15×15; warm-start is the remedy**" (campaign log,
> 2026-06-13 warm-start entry). That conclusion stands **as observed on its run**
> (the cold `G15-seed` cell with the v8 recipe but **no WDL head**: plies 68→11, VL
> rising, no recovery seen in the window before we swapped to warm-start). What the
> autolab's first 15×15 run shows is that a *different* recipe **survives the same
> collapse without intervention** — so "warm-start is required" was over-general; it
> was the fastest fix we tried, not the only path through.

**Evidence (autolab `15x15-wdl` lane, cell `G15-wdl`, board 15, from scratch, no
warm-start, no vcf/defense teacher; first self-driving 15×15 run — see
`TRAINING_WIKI.md` 2026-06-19 "15×15 era").** The from-scratch run went **through**
the documented cold-start fast-attack collapse and then **self-recovered**, with no
teacher and no warm-start:

| phase | epoch | plies | read |
|---|---|---|---|
| pre-collapse | ~21 | **69.5** | defended, random-ish opening play |
| collapse trough | ~65 | **9.2** | the fast-attack collapse (exactly as documented) |
| self-recovery | ~260–670 | **~35–40** (stable) | mid-game richness regrown — *without any teacher* |

**Why this is healthy recovery, not the death-tell** (the signatures that distinguish
survivable absorption from terminal fast-attack collapse — cf.
[loss-floor-bouncing.md](loss-floor-bouncing.md), the HALT conditions above):
- **WDL value-loss held ~0.81–0.89 the whole way** — it **never collapsed toward
  zero**. A value head saturating to ~0 is the terminal "confident-in-a-bad-fast-
  attack" tell (the #18/#42 white-defense death-spiral, vl→0.04–0.06); a value-loss
  that *stays up* through the plies trough is healthy maturation on hard positions.
- **Policy-loss fell monotonically 5.4 → 1.6** (initial **5.42 ≈ ln(225)** confirms a
  genuine 15×15 policy over 225 cells, not a degenerate head).

**Conclusion (revised, recipe-scoped):** with the **v8 recipe + WDL value head**, the
cold-start collapse is a **survivable transient**, not a terminal trap. The
warm-start "remedy" of 2026-06-13 **may not be strictly required for this recipe** —
the WDL head appears to carry the run through the trough that crashed the
no-WDL cold seed. (Open: is it the WDL head specifically, or simply a longer
patience window than the 2026-06-13 cold run got before we swapped it?)

**CRITICAL CAVEAT — survival ≠ defense.** "Survived the collapse / recovered
mid-game richness" is **NOT** the same as "learned white-side defense." The decisive
open question is this from-scratch net's **white W-L-D vs Rapfi** (the strong-attacker
diagnostic; see [white-side-defense-plan.md](white-side-defense-plan.md) §1B.2 and
the 2026-06-18 Rapfi entries in `TRAINING_WIKI.md`):
- if it reproduces the warm-started champion's **0/12-white** hole ⇒ the white deficit
  is **representational / recipe-deep** (warm-start never fixed it; it's the recipe);
- if it defends **better** ⇒ **warm-start was baking in the attacker bias** (the 9×9
  champion tower transferred its first-mover-win habits, and from-scratch avoids them).

**That probe is the next frontier read** — it decides whether the white-defense work
(#43/#37) is fighting the recipe or fighting the warm-start. First 15×15 champion from
this run: `15x15-wdl@0`, internal eval **elo 1918** (first 15×15 number, **not**
comparable to the 9×9 scale).
