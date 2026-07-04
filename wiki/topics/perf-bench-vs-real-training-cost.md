# The perf bench measured generation; real training has an unbounded per-epoch runaway
> **Status: LIVE lesson** *(2026-07-04)* — May 2026 evidence; the runaway trap.

*Finding from the gomoku AlphaZero perf lab on 2026-05-23. Hardware: MacBook Pro Mac17,6, Apple M5 Max, 48 GB, macOS 26.4.1, PyTorch 2.11.0. Workload: ResNet-style policy/value network (~325k params), MCTS-driven self-play with a wave-mode trainer in this repo's `gomoku` module. Run LF1, wandb run `h9al2e0k`.*

This page documents a case where a perf-lab benchmark and a real training run disagreed by an order of magnitude — not by noise, but because the bench measured a regime the real run only sits in for its first ~15 epochs. The headline number ("+152% throughput") was real and reproducible. It was also nearly useless as a training-speed claim, because the recipe that produced it makes a real run get *slower every epoch, without bound,* once the replay buffer fills.

We're publishing this because it's a trap that's easy to fall into and hard to see coming. If a search engine sent you here looking for *"AlphaZero replay buffer SGD steps per epoch blow up,"* *"self-play generation outpaces trainer,"* or *"why is my training getting slower every epoch"* — the short answer is: you probably have a positive feedback loop between generation rate, buffer inflow, and a per-position SGD schedule, and your benchmark stopped before the buffer filled so it never saw it.

## TL;DR

1. **The lab maximized the wrong objective.** It tuned for peak *generation* throughput (augmented-positions/sec, epochs/sec) in 120-second benchmark windows. The real objective is *fastest training* — wall-clock-to-elo. These two diverge, and the recipe that wins the first can lose the second.

2. **The promoted recipe (R-TRAIN-LEAN-fp16: wave_size 512 + `sgd_per_position=0.001` + fp16 workers) caused an unbounded runaway in per-epoch cost in a real run.** Per-epoch wall time climbed monotonically from ~20 s to over 7 minutes and was still climbing when we stopped it at epoch 31. Steps/epoch went from 25 to 3,236 over the same span.

3. **The mechanism is a positive feedback loop.** A large wave_size makes generation outpace the trainer; workers keep generating during the (lengthening) SGD phase, so each version's ingested "tile" of games grows; `sgd_per_position` scales SGD steps with the tile, so the train phase lengthens; which lets even more games accumulate during it. Tile → steps/epoch → wall-time all grow in lockstep, without a fixed point.

4. **The bench missed it entirely because it stopped before the buffer filled.** The 120 s cell ran ~8 epochs, all in the cold-buffer, small-tile, pre-runaway regime (~15 s/epoch). The runaway only emerges after the 1.5M-position buffer fills (~epoch 14–27) and then accelerates. A perf cell that ends before buffer-fill measures a transient that real training never sits in.

## Background and vocabulary

The relevant pieces of the pipeline:

- **wave-mode (lockstep) trainer.** Self-play workers generate games against a single frozen model version `v`. The trainer waits for a barrier, ingests *all* the games produced against `v` (the "tile"), runs SGD on them, publishes `v+1`, and the workers reload and generate against `v+1`. See [wave-of-lockstep-design.md](wave-of-lockstep-design.md) for the design rationale (per-version buffer uniformity).
- **tile.** The set of self-play games ingested for one model version. In WL5 (`wave_size=64`) the tile is bounded at roughly 70–86 games. The size of the tile per version is the load-bearing variable on this page.
- **wave_size (V).** The MCTS eval-batch size — how many leaf evaluations a worker batches into a single forward pass. Higher V = each worker holds far more concurrent MCTS sims in flight, so generation throughput is much higher. WL5 uses V=64; the LEAN-fp16 recipe uses V=512.
- **`sgd_per_position`.** The trainer's SGD schedule: it does roughly `sgd_per_position × (positions ingested this epoch)` optimizer steps per epoch. WL5 uses 0.0025; the LEAN-fp16 recipe uses 0.001. Note that it scales SGD work with *inflow*, not with a fixed budget.
- **replay buffer.** 1.5M positions, inherited from the WL5 lineage. Once full it evicts oldest-first; the relevant thing here is *when it fills*, because the runaway dynamics change at that point.
- **aug-positions/sec (aug/s), epochs/sec.** The perf lab's throughput proxies. Each position is augmented to 8 D4 symmetries before training. These measure how fast positions are *produced and consumed in a short window* — they are generation/ingest-rate proxies, not training-speed metrics.
- **R-TRAIN-\*.** The research lab's "live training" reference points: trainer + N workers all sharing MPS, scored in aug/s and epochs/s. R-TRAIN-WL5 is the production baseline (3,297.6 aug/s); R-TRAIN-LEAN-fp16 is the candidate this page is about. Defined in [research-lab-charter.md § Success metric](research-lab-charter.md).

The two relevant drivers:
- [`scripts/lab_train_cell.py`](../../scripts/lab_train_cell.py) — the perf-bench harness. Short (≤3 min total) windows, throughput-scored.
- `run_sweep` cells — real, quality-tracked training runs, hundreds to thousands of epochs, wandb-logged. LF1 is one of these.

## What the bench said

The perf lab spent a cycle maximizing generation throughput. Its headline R-TRAIN result was **R-TRAIN-LEAN-fp16** — the WL5 recipe plus `wave_size=512`, `sgd_per_position=0.001`, and fp16-eval workers — which measured:

> **8,340.5 aug/s, 0.0667 epochs/s ≈ 15 s/epoch, +152.9% vs R-TRAIN-WL5** (lane L11b', 120 s `lab_train_cell` window, Reviewer APPROVE).

That number is real and reproducible. The two levers it stacks (a fp16 worker eval forward, and a lowered `sgd_per_position` to free MPS for the workers) are independently verified, and they compose multiplicatively almost exactly as predicted — that result has its own page, [m5-max-fp16-and-throughput-regimes.md](m5-max-fp16-and-throughput-regimes.md) (Finding 3). The perf lab did its job: it found a real lever and measured it honestly *for what it measured*.

What it measured was **generation throughput in a 120-second window**. It was correctly filed `needs_repeat` under the [Training-Quality Promotion Gate](../ops/experiment-ledger.md#training-quality-promotion-gate) — the lab does not certify training-behavior knobs for production, it surfaces them. The handoff to a real run was the next step. That's the run this page is about.

## What the real run did

We promoted the R-TRAIN-LEAN-fp16 recipe to a real 1000-epoch `run_sweep` training run — cell **LF1**, wandb `h9al2e0k` (a 100-epoch test, `geft5xmy`, ran first and already showed the shape). It does **not** behave like the bench.

The real run exhibits an **unbounded runaway in per-epoch cost.** The trajectory:

| epoch | SGD steps/epoch | wall time | new positions/epoch | tile (games ingested) |
|---|---|---|---|---|
| e1  | 25   | 19.9 s  | ~985   | 101  |
| e10 | ~135 | 18.4 s  | —      | 610  |
| e15 | 218  | 30.4 s  | —      | 1120 |
| e20 | 446  | 57.9 s  | —      | 1691 |
| e25 | 792  | 107.8 s | —      | 2284 |
| e27 | 1237 | 167.5 s | —      | —    |
| e29 | 1990 | 267.6 s | —      | —    |
| e30 | 2523 | 342.5 s (5.7 min) | —     | 2898 |
| e31 | 3236 | 436.9 s (7.3 min) | 17,391 | —   |

By epoch 31 the run was at **7.3 minutes/epoch and still climbing** — that's the point at which we stopped it. The wave tile grew in lockstep with the SGD-step count: 101 → 610 → 1120 → 1691 → 2284 → 2898 games per version. New positions ingested per epoch went from ~985 to ~17,391.

Read against the bench's "15 s/epoch": the real run's steady-state per-epoch cost is roughly **10×–30× the bench number and rising**. A "1000-epoch run" at this recipe is a multi-day job, not the ~4 hours a naive extrapolation from 15 s/epoch would suggest.

## The mechanism

The runaway is a positive feedback loop with no fixed point. Step by step:

1. **Wave-mode ingest.** Workers generate a tile of self-play games against model version `v`. The trainer ingests the tile, runs SGD, publishes `v+1`. Tile size is set by how many games accumulate before the trainer fires its barrier.

2. **Large wave_size raises generation throughput.** `wave_size=512` (vs WL5's 64) lets each worker hold far more concurrent MCTS sims, so the workers produce games much faster than at V=64.

3. **Generation outpaces the trainer.** At V=512, the workers produce games faster than the trainer can consume them. Critically, **the workers keep generating during the trainer's SGD phase** — they don't block. So by the time the trainer finishes one epoch's SGD, a *larger* pile of games has accumulated, and the next tile is bigger.

4. **`sgd_per_position=0.001` couples SGD work to inflow.** A bigger tile means more new positions, which (via the per-position schedule) means more SGD steps, which means a *longer* train phase, which gives the still-running workers *more time* to pile up an even bigger next tile.

5. **The loop has positive gain.** Bigger tile → more positions → more SGD steps → longer train phase → more games accumulate during it → bigger next tile. Each term feeds the next with gain > 1, so tile, steps/epoch, and wall-time all grow without bound. There is no equilibrium tile size; the system diverges until something external (eviction churn, thermal, or a human) intervenes.

6. **WL5 (wave_size=64) does not run away.** At V=64, generation rate ≈ trainer consumption rate, so the tile reaches a small steady state (~70–86 games) and stays there. The loop's gain is below 1; it has a fixed point. **V=512 broke that balance.** wave_size is the knob that moves the loop across the stability boundary.

This is the **L11 research-lab finding at unbounded scale.** L11 already observed that V=512 fills the buffer ~2.4× faster, producing ~3× more SGD steps/epoch and causing the trainer to monopolize MPS — and it *rejected* V=512-at-default-sgd for exactly this reason (see [perf-log.md](../ops/perf-log.md) L11, and [research-lab-charter.md § R-TRAIN](research-lab-charter.md)). What L11 saw as a bounded ~3× penalty in a short window is, in a real run past buffer-fill, an *unbounded* runaway. The lowered `sgd_per_position` in LEAN-fp16 was meant to *cure* the L11 MPS-monopolization at the bench scale (and it did, in the window) — but it doesn't remove the feedback loop, it just delays the point at which the loop's growth becomes visible.

## Why the perf bench completely missed it

The R-TRAIN-LEAN-fp16 cell ran 120 seconds ≈ 8 epochs. Every one of those epochs was in the **cold-buffer, small-tile, pre-runaway regime** (~15 s/epoch). The buffer was nowhere near its 1.5M capacity; the tile was still small; the feedback loop's growth hadn't compounded enough to dominate.

In the real run, the runaway only becomes visible after the buffer approaches capacity (~epoch 14–27 for LF1) and then *accelerates* past it. The bench window ended a dozen epochs before the regime change.

**The lesson is general: a perf cell that stops before the buffer fills — and before several post-fill epochs — measures a regime that real training never sits in.** The cold-buffer transient is a real measurement of a real thing (cold-buffer generation throughput), but it is *non-predictive* of steady-state training cost. The charter's existing calibration note ("R-TRAIN cells need a window that spans ≥3 of the trainer's actual epochs") is necessary but not sufficient: 3 epochs in the cold regime still misses a regime change that happens at epoch 27. The window has to span the regime change, not just a few epochs.

This is the same family of trap as [m5-max-fp16-and-throughput-regimes.md](m5-max-fp16-and-throughput-regimes.md) (don't extrapolate wall-clock from a short cycle-time sample) and [mcts-perf-ceiling.md](mcts-perf-ceiling.md) (know what's already been optimized) — measured proxies that look like the thing you care about but aren't.

## The other reason a short sample under-predicts: plies grow as the model learns to defend

The buffer-fill runaway above is the *second* reason a short window misreads real
training wall-clock. The first is more fundamental and applies even at WL5 settings:
**self-play cycle time grows ~quadratically with mean plies, and plies grow as the
model learns to defend — which is the actual training goal.** A "fast" cycle is the
symptom of a *collapsed* model that races to win in ~12 plies; it is not a speed win.

**Mechanism.** Each ply costs `n_simulations` MCTS sims to pick a move, so plies-per-game
scales per-game cost linearly; on top of that, MCTS trees get deeper at higher plies, so
per-sim cost rises too — gen time ends up super-linear (~quadratic) in plies. Evidence:
`sync-gpe128-fasteval` epoch 122→125 showed plies 34.5→52.8 (**+53%**) but gen 48.5 s→102.5 s
(**+111%**) — roughly the square. Counter-example trap: in the `az-recipe-160k` run, the
e1 smoke showed plies≈32 (two untrained models can't end a game), by e51 plies had
*collapsed* to ~15 (fast-attack mode), cycle time fell 11 s→2 s, and a naive extrapolation
projected ~2.3 h for 3900 remaining epochs at 2 s/cycle. That projection was implicitly
betting *against* the model learning defense: the moment it does, plies climb back toward
real-game levels (~50–80 on 9×9) and the per-cycle cost blows out.

**How to quote a mid-flight self-play ETA — give a range, never one number:**
- **Lower bound:** current cycle time × remaining epochs (assumes no defense learned / plies stay collapsed).
- **Upper bound:** assume plies grow to real-game levels and gen scales super-linearly — rule of thumb **3–5×** the lower-bound cycle time.
- State the assumption explicitly. A run getting *slower* mid-flight is a **positive** signal (defense is being learned), not a problem; a tight single-number ETA is implicitly a bet that training fails. See also `selfplay/plies_mean` falling + concave buffer-fill = fast-attack collapse ([loss-floor-bouncing.md](loss-floor-bouncing.md)).

**Net:** there are two independent reasons a short or mid-flight rate under-predicts real
training wall-clock — **(1)** plies grow as defense is learned (this section), and **(2)** the
replay buffer fills and SGD-steps/epoch climbs (the rest of this page). Always read the real
`train=Xs` phase from `trainer.log` and check the plies trend before quoting a training ETA.

## The meta-point: the lab optimized the wrong objective

The perf lab's stated mission for this cycle was *"fastest generator (aug/s)."* The real objective of the whole project is *"fastest training — minimum wall-clock-to-elo."* This finding is the clearest case yet that **these two objectives diverge, and maximizing the first can actively harm the second.**

Maximizing generation throughput floods the trainer. In a balanced system (WL5) the flood is bounded. In an unbalanced one (LEAN-fp16) the flood is a runaway: the generator wins its benchmark by producing positions faster than the trainer can use them, and the trainer's attempt to use all of them (via `sgd_per_position`) is what blows up the per-epoch cost. The aug/s metric *rewards* the imbalance that wrecks the run.

A subtlety that keeps the verdict honest: LF1's faster epochs do more learning *per epoch*. It hit elo ~437→776 around the buffer-full transition (~epoch 28), because each epoch ran ~1300 SGD steps. So epochs-to-elo may be *fewer* even though wall-per-epoch is much higher — which is exactly why neither the bench's aug/s nor a naive epoch count is the right metric. **Only wall-clock-to-elo (with a quality check on val/policy_ce) can adjudicate whether this recipe is actually faster.** That measurement is pending.

## Research lanes — for the lab to explore

Concrete directions, in rough priority order:

1. **Fix the R-TRAIN metric: warm the buffer to capacity before measuring.** Adopt an Lhot-style pre-warm so the cell starts at full-buffer steady state, then measure steps/epoch, train-time, *and the tile-growth trend* over ≥20 post-fill epochs. The deliverable is not a single throughput number but a *slope*: is the tile bounded or diverging? A cell that doesn't run past buffer-fill is non-predictive and should not produce an R-TRAIN number at all. This is the single highest-value lane because it fixes the instrument that produced the misleading +152%.

2. **Map the runaway stability boundary.** Sweep wave_size (V) against the trainer-consumption rate and find the V at which the tile stays bounded. We have two points: V=64 is stable (tile ~70–86), V=512 is divergent. Where is the knee? Is it a sharp threshold (gain crosses 1) or a gradual one? This is the parameter that decides whether a recipe runs away at all.

3. **`sgd_per_position` vs `sgd_per_game` vs a hard step cap — at full buffer.** `sgd_per_position` is the term that scales SGD work with inflow, so it *amplifies* the runaway. Compare against (a) `sgd_per_game` (scales with tile count, not positions — may have lower gain), and (b) a fixed per-epoch step cap that decouples SGD work from inflow entirely and is structurally incapable of running away. Measure all three at full buffer, where the difference actually shows up.

4. **Add wall-clock-to-elo as a first-class metric family.** The throughput proxies (aug/s, epochs/s) provably diverge from the real objective. The lab should add a real-training-cost metric — minutes-to-target-elo, or elo-per-wall-hour — even though it forces cells to run much longer than the current smoke-first 60–90 s doctrine. Some questions can only be answered by a long run; this is one of them. Pair with a quality gate (val/policy_ce vs `wl5_validation_v1.pt`).

5. **Is the high step count even productive learning, or redundant SGD on stale data?** LF1's elo was noisy (~339–751, not cleanly climbing) while steps/epoch exploded. The extra SGD steps may be largely wasted — re-grinding stale buffer positions rather than learning new structure. Check val/policy_ce against *cumulative* SGD steps: if the loss-vs-steps curve flattens while steps/epoch keep climbing, the runaway is burning wall-clock on redundant updates. If it doesn't, the steps are buying something. This determines whether the runaway is merely slow or actively pathological.

6. **Architectural fix: bound the tile or backpressure the workers.** Should the trainer cap the number of games it ingests per version (dropping or deferring the excess), so the tile is structurally bounded regardless of generation rate? Or should it backpressure the workers — pause generation during a long train phase so games don't pile up? Either change would convert the open loop into a closed one. Prototype both and measure whether bounding the tile costs anything in elo-per-wall-hour (it may not, if lane 5 shows the excess steps are redundant).

## Caveats and scope

- **Single recipe, single run.** The runaway trajectory is from one run (LF1, `h9al2e0k`) plus its 100-epoch precursor (`geft5xmy`). The shape is mechanistically clear and reproducible from the mechanism, but the exact epoch numbers (buffer fills ~e27, 7.3 min/epoch by e31) are one run's data. Thermal state at run-start was HOT, which may shift absolute wall-times.
- **The +152% bench number is not wrong.** It is a correct measurement of cold-window generation throughput. The error this page documents is *interpreting* it as a training-speed claim, which the lab explicitly flagged against in the LF1 receipt. The bench and the run measured different things; both measured their thing honestly.
- **Buffer capacity is 1.5M here.** The buffer-fill epoch (~27) scales with capacity and generation rate. A 3M buffer (the next-cell default per [project-buffer-undersized]) would delay the regime change further — and a bench window that fit inside the old 1.5M fill would be even more badly fooled by a larger buffer.
- **`gomoku.train` stays fp32.** The fp16 in this recipe is worker-eval-only with an fp32 cast-back at the MCTS boundary; it is not implicated in the runaway. The runaway is purely about generation-rate × tile-growth × `sgd_per_position`, independent of precision.
- **TQ verdict still pending.** LF1 is filed `needs_repeat` under the Training-Quality Promotion Gate. The recipe is NOT promoted to production. R-TRAIN-WL5 remains the production training recipe. Whether LEAN-fp16 (or a bounded variant of it) is ever faster in wall-clock-to-elo is the open question lanes 4–6 exist to answer.

## Cross-refs and primary sources

- [experiment-ledger.md](../ops/experiment-ledger.md) — the **LF1 receipt** (2026-05-23, "LEAN-fp16 as a REAL run") with the full runaway trajectory and quality notes, and the **L11b' R-TRAIN-LEAN-fp16 receipt** (the +152.9% bench finding it was promoted from).
- [research-lab-charter.md § Success metric](research-lab-charter.md) — the R-TRAIN-\* metric definitions. **Note for the next charter pass:** the R-TRAIN-\* metric definition needs the warm-buffer fix from lane 1 — as defined, a cold-window R-TRAIN cell is non-predictive of real training cost, which is exactly the gap this page documents.
- [m5-max-fp16-and-throughput-regimes.md](m5-max-fp16-and-throughput-regimes.md) — the sibling public page; its Finding 3 is the multiplicative-composition result that *produced* the +152% number, measured honestly as a generation-throughput compound. This page is its cautionary epilogue: the throughput compound was real, the training-speed extrapolation was not.
- [perf-log.md](../ops/perf-log.md) — the L11 finding (V=512 fills the buffer faster → more SGD/epoch → trainer monopolizes MPS). The LF1 runaway is L11's mechanism at unbounded, post-buffer-fill scale.
- [wave-of-lockstep-design.md](wave-of-lockstep-design.md) — the wave-mode / tile design that the runaway exploits.
- gomoku-train skill, "Tuning knobs → The LEAN-fp16 'faster' recipe" — the operator-facing version of this warning, with the explicit "do NOT extrapolate training wall-clock from perf-lab epochs/s" guidance and the LF1 cell wiring.

If you found this page via search and it helped, the open-source repo is at [github.com/jasonyandell/gomoku](https://github.com/jasonyandell/gomoku). The general shape of the trap — a benchmark that measures a pre-equilibrium transient of a system that has a positive feedback loop past some fill threshold — is not specific to gomoku or to Apple silicon. PRs welcome with the same dynamics in other AlphaZero / replay-buffer pipelines.
