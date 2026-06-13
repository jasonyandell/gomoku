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
  audit) to gate the multi-day GPU commitment before launch. Seed run + the
  parallel improvement workflow + the loop come up once recon clears.
