# ML Perf Lab Log

Narrative timeline for the M5 Max perf era. Sibling to
[experiment-ledger.md](experiment-ledger.md) (formal receipts) and
[status.md](status.md) (control-room summary). This page is for the
day-by-day story: what was tried, what surprised us, what comes next.

Append-only. Lead each entry with `## [YYYY-MM-DD] <lane> | <one-line headline>`
so future sessions can scan with grep.

Cross-refs:
- Philosophy: [../topics/m5-max-as-mainframe.md](../topics/m5-max-as-mainframe.md)
- Procedure: [../topics/perf-lab-session-runbook.md](../topics/perf-lab-session-runbook.md)
- Receipts: [experiment-ledger.md](experiment-ledger.md)
- Board: [frontier.md](frontier.md)

---

## [2026-05-23] delta-e run-1 | First Δelo flywheel run: harness works end-to-end, but the anchor ladder is too short to measure strong models

Jason: "yeah light it up, fan it out" / "wandb please." Ran the first real run of the Δelo north-star scoring engine (`scripts/delta_e_harness.py`): fork 3 recipes off a common WL5 parent C, train each a fixed 40-epoch window on the replay trainer, then anchored-eval each fork + C at 40 games/baseline and rank by Δelo.

**It ran clean, top to bottom** — fork → replay-train → `latest.pt` → anchored-eval → Wilson-CI → implied-elo → Δelo → INSIDE-NOISE verdict → ranked table → `results.json`, 3 forks in one shot, each logging to its own fresh wandb run. The flywheel is real. (Also fixed a wandb bug first: `train_replay --resume` was inheriting the checkpoint's `wandb_run_id`, so forks were *resuming* WL5's run instead of starting fresh — now an explicitly `--run-name`'d replay fork starts a new run.)

**The result: all 3 INSIDE-NOISE.**

| rank | recipe | Δelo | ±CI | verdict |
|---|---|---|---|---|
| 1 | lru, sgd=100 | −55.8 | ±218 | INSIDE-NOISE |
| 2 | recency_weighted, sgd=100 | −127.4 | ±228 | INSIDE-NOISE |
| 3 | lru, sgd=300 | −195.0 | ±233 | INSIDE-NOISE |

Parent C elo = **+1536.7** [+1357, +1691]. Fresh wandb: ms1pplps, 91awvfib, jc228edo.

**The finding is a method-limit, not "the recipes are equal."** The strongest anchor we have is `lookahead:depth=4 @ 1500`, and C already beats it (78%). So both C and every fork pin near the **~1700-elo ceiling** of the ladder — the signal lives almost entirely in the d4 win-rate (the only non-saturated anchor), and at 40 games that win-rate has a huge CI. Differencing two near-ceiling implied-elos buries any real recipe difference under sampling noise. And there's no escape via a "headroom parent": `keep-last-n=3` pruning destroyed every early/weak checkpoint (WL3.1 at e1504 also sits at model_elo ~1465–1567 — the ladder saturates around there regardless). The faint sub-noise trend (do NOT over-read): sgd=300 is the *most* negative, directionally consistent with over-grinding a tiny curated slice off an already-converged net.

**The fix — head-to-head.** Don't measure both models against a too-weak fixed anchor; play the **fork directly against C**. Two similar-strength models score near 50% against each other, which is the *maximally sensitive* region of the logistic, so the relative Δelo gets a tight CI with no ceiling (this is exactly how AZ-style gating works). Added `--head-to-head` to `delta_e_harness.py` this session (model-vs-model via the existing `play_match_pickers` + `mcts_picker`; relative Δelo = 400·log₁₀(p/(1−p)), Wilson CI mapped through the logistic; self-test + micro-smoke green — self-play scores 50% → Δelo +0). run-2 re-runs the same 3 recipes off the same WL5 C, head-to-head. decision: needs_repeat (recipes un-resolved by the anchored method, not rejected).

---

## [2026-05-23] LF1-followups | Fan-out (4 CPU agents + GPU): runaway knee mapped to (384,512], tile-cap validated end-to-end, metric instrument fixed

Jason: "proceed, fan out background subagents." Ran the LF1-followups block as a **two-queue fan-out** — 4 worktree-isolated CPU agents in parallel + the GPU runaway-boundary sweep driven serially by the orchestrator. (The orchestration pattern is now codified in the gomoku-perf-lab skill, "Fan-out orchestration mode.") All five sub-lanes landed coherently into one story.

**The headline science — runaway stability boundary (lane 2), four lab_train_cell runs, `--max-epochs 18`, LEAN recipe (V, sgd_per_position=0.001, fp16):**

| run | steps e1→e18 | wall/epoch e1→e18 | new-pos/epoch | verdict |
|---|---|---|---|---|
| V=256 uncapped | 20→56 | 9.6→7.3s | 75→208 | **bounded** |
| V=384 uncapped | 19→62 | 5.9→8.8s | 73→231 | **bounded** |
| V=512 uncapped | 22→**154** | 6.8→**19.9s** | 77→**630** | **DIVERGENT** (monotonic) |
| V=512 + `--max-tile-games 120` | 23→53 | 6.6→7.7s | 84→207 | **bounded — cap tamed it** |

**The runaway knee is in (384, 512]** — a sharp threshold. V≤384 keeps up; V=512 falls behind and diverges.

**Method lesson that nearly bit me (filed to the skill friction log):** the divergence does NOT show in the per-version `tile` — that's barrier-bounded at ~85 by lab_train_cell's `worker-min-games`=64 (8×8) and is *invariant to V*. I first concluded "lab_train_cell can't reproduce the runaway" from the flat tile column. Wrong. The runaway lives in **steps/epoch, wall/epoch, new-positions/epoch, and `age`** — at V=512 the trainer falls behind and drains *more stale versions per epoch* (age 2→3), so per-epoch SGD work climbs without bound while the per-version tile stays flat. I also realized I'd imported LF1's *run_sweep* divergence as the V=512 point without ever running uncapped V=512 in lab_train_cell — ran that missing control, and it diverged exactly as predicted (steps 22→154). Same friction-log discipline (attribute before concluding) applied to my own GPU data.

**Lane 6 (architectural fix) — VALIDATED end-to-end, not just unit-tested.** `--max-tile-games 120` converts the divergent V=512 (steps 154, wall 19.9s) into a bounded run (steps 53, wall 7.7s) — same recipe, the cap is the only difference. The structural lever closes the open loop. (Also shipped: `--max-sgd-steps-per-epoch`, `--sgd-per-game`; all opt-in, WL5 defaults byte-identical.)

**Lane 5 (productivity analysis) — the extra steps are REDUNDANT.** From LF1 wandb (`h9al2e0k`): `val/policy_ce` hits its best (3.9905) at cum-step ~3.4k / epoch 20 — *before* the runaway — then flattens and reverses while cumulative steps go 4.4× higher. Train-loss keeps falling, val-loss doesn't → the giant late tiles re-grind stale buffer (~28% current). **So bounding the tile costs ~0 elo and improves elo-per-wall** — lane 5 and lane 6 compose: the cap removes exactly the steps that weren't buying quality.

**Lane 1 (metric fix) + Lane 4 (metric design) — the instrument.** Lane 1 shipped an opt-in warm-buffer / shrunk-buffer measurement mode in lab_train_cell (`--replay-buffer-size`, `--prefill-*`) that reports the post-fill steps/wall/tile *slope* + a BOUNDED/DIVERGING verdict (and refuses to emit a number with <20 post-fill epochs — the cold-window cell that produced the misleading +152% is structurally disallowed). Lane 4 designed the higher-order objective: `wall-clock-to-elo-metric.md` (MTTE primary, EPWH/Δelo·hr⁻¹ secondary), reconciled with the existing `delta_e_harness.py` (5 gaps identified, chiefly: fix the window in wall-clock not epochs, or a runaway re-imports the bug) + a proposed (un-applied, Class-B) charter diff adding an R-ELO-* family.

**Integration:** 3 code/design branches merged `--no-ff` to main, integrated tree verified (WL5 default byte-identical, 7 new flags coexist, tests green). Both code agents hit a stale-base worktree + editable-install path-leak (filed to skill; fix = "merge local main at startup" now in the fan-out guardrails). decision: lanes 1/4/6 promote (code/design landed); lane 2 = knee mapped (research finding); lane 5 = analysis finding. **Reviewer: APPROVE** (headline reproduced from raw TSVs, attribution sound, WL5 byte-identical, 20 tests green, charter Class-B un-applied). Caveat — and a Reviewer correction that strengthens it: the bounded runs (V=256/384/capped) sit at ~64% fill, BUT the divergent uncapped V=512 run actually reached the full 1.5M buffer at epoch 15 and kept diverging (steps 149→154 through e15→18) — so divergence extends THROUGH buffer-fill, not just pre-fill. Only the capped-stays-bounded-post-fill piece remained open — now CONFIRMED: a 32-epoch capped V=512 run hit the full 1.5M buffer at e28 and stayed BOUNDED through and past fill (postfill steps_slope=0.019, wall_slope=0.037, tile_verdict=BOUNDED; e28-32 at full buffer steps ~54 / wall ~8-10s flat). The cap holds at steady state, not just pre-fill. Caveat closed; LF1-followups fully resolved.

## [2026-05-23] LA1 | Lookahead-eval ~6.3× faster, byte-identical — the eval anchor was paying a pure-Python tax at every node

Jason redirected the lab: "perf pass on lookahead eval." Target = `lookahead_player`, the alpha-beta baseline that anchors model Elo in `eval_worker` and the training-loop eval. It's the known-slow eval path — `train.py:341` dropped `lookahead:depth=2` as a default precisely because it cost "45s+ for noisy signal," and depth=4 (the dominant anchor) is worse. This isn't a gen (R-S*) or train (R-TRAIN-*) lane; it's a new eval-path family, R-EVAL-*.

**Smoke-first measurement (`scripts/bench_lookahead.py`, 60 deterministic midgame positions):** depth=2 = 15.35 ms/move, depth=4 = 145.6 ms/move (6.9 moves/s). A 20-game pass plays ~240 lookahead moves → ~35s at depth=4 on the lookahead side alone. Matches the "45s+" lore.

**The surprise was where the time went.** Native state_ops (`_state_ops_native.so`) is built and active — so `apply`/`is_terminal`/`legal_mask` are already C-fast (~0.1s total in the profile). The bottleneck was three *pure-numpy* helpers in `baselines.py` running at every search node, each doing a Python loop wrapped around tiny numpy ops: `_find_immediate_wins` (52% — a per-legal-cell loop scanning every empty square at every leaf), `_candidate_moves` (29% — a 24-offset neighbor-dilation loop with fancy-index bounds masking per offset), and the full-81-cell `_score_all_moves` used only for candidate ordering. cProfile made it unambiguous: 27,789 leaf calls and 37,338 internal-node calls dominating a 10.6s run.

**Fix = three vectorizations, all behavior-preserving:** `_find_immediate_wins` → one dense per-cell max-after-placement over `_DENSE_WIN_BY_CELL` (mask the padded slots before the max); `_candidate_moves` → one gather over a precomputed (81,81) `_NEIGHBOR_MASK`; new `_score_cells` restricts the dense gather to candidate rows for the ordering hot path. Result: **depth=2 6.5×, depth=4 6.3×** (22.94 ms/move, 43.6 moves/s). cProfile after: the two loop helpers dropped from 6.26s to 0.22s combined tottime.

**The discipline that mattered: prove byte-identical, don't argue it.** The lookahead is an Elo *anchor* — if its move selection shifts, every model's measured Elo shifts with it, silently. So I reimplemented the old loop logic alongside and asserted equality across 360 positions (candidate sets, immediate-win sets, per-cell scores, and the final depth-4 move) — all byte-identical. That makes the Training-Quality gate trivially satisfied: there's no strength/game-shape risk to measure because the outputs are provably unchanged; only the cost moved. test_baselines/test_lookahead_quiescence/test_rating/test_eval_parallel all green.

**Why it matters for the north-star:** cheaper lookahead → frequent Elo anchoring (esp. depth=4) becomes affordable inside the eval loop, which feeds Δelo/Δt directly. Remaining levers are diminishing-returns: history-free `apply` for the lookahead path (negamax never reads the 8 history snapshots `apply_move_arrays` copies per node, ~8-10% — but it's shared with the MCTS path, so it needs a lookahead-specific lighter apply), or a numba/cython negamax (Class C, out of autonomous scope). Not queued unless eval cost resurfaces. **decision: promote — Reviewer APPROVE** (integrity: concurrent 6d47bbb work intact-not-clobbered; math exact; equivalence logic sound across edge cases; surfaces consistent; tests green). The Reviewer flagged that the byte-identical proof was an ephemeral job-dir harness — so I committed `tests/test_baselines_vectorized_equiv.py` pinning the three helpers against independent brute-force references, so a future edit can't silently shift the Elo anchor.

## [2026-05-23] Lpwr2b | RESOLVED: the cross-engine throttle is FLOP-rate-independent — it's occupancy/working-set, not compute-power

The discriminator. Same matrix (4096), back-to-back fp32-hog then fp16-hog, bracketed by no-hog. fp16@4096 has half the byte-footprint and can push far more FLOPs; the *sign* of its throttle-vs-fp32 was meant to separate the hypotheses.

It came out flat. **fp32-hog: 1.98 TFLOP/s, workers −15.9%. fp16-hog: 7.03 TFLOP/s (3.5×), workers −14.8%.** A 3.5× increase in the hog's compute rate produced ~zero additional worker throttle (1.3% apart, inside the 6.6% no-hog noise). So the cross-engine throttle does **not** scale with the hog's FLOP-rate — and since fp16 also had ~1.75× the byte-traffic-rate, not cleanly with bandwidth-rate either.

Put together with Lpwr2 (throttle grows with matrix *size*: −8.8%→−26% across 2048→8192), the picture is: the coupling tracks the GPU's **working-set size / sustained occupancy** — "is the GPU pinned busy, and how big is its footprint" — not how many FLOPs or bytes/sec it pushes. **Compute-power-draw is ruled out as the driver.** Both 4096 hogs saturate the GPU's occupancy equally regardless of throughput, so they throttle the ANE workers equally.

Actionable "know the machine" takeaway: to cut the cross-engine contention a heavy GPU trainer inflicts on CPU/ANE self-play workers, shrink the GPU's **memory working-set / occupancy**, not its FLOP count. More compute through an already-busy GPU is ~free of extra contention; a bigger footprint is not. (The 120s cells also tightened the no-hog baseline to 6.6% from Lpwr2's 20% — lesson applied.)

decision: resolved (diagnostic) — pins the Lpwr/Lpwr2/Lpwr2b mechanism strand. Both this session's big strands (ANE-for-self-play; cross-engine coupling) are now resolved on clean evidence.

## [2026-05-23] Lpwr2 | Cross-engine coupling sweep: mutual throttling confirmed; mechanism hints at bandwidth/footprint (not pure power)

The corrected Lpwr re-run: cold-chip, interleaved (no-hog/hog back-to-back per intensity), pure self-play, ANE-resident workers. Swept hog matrix {2048, 4096, 8192}.

Worker throttle under the hog: −8.8% (m2048), −21.6% (m4096), −26.0% (m8192). And every hog was itself suppressed to 2.2–3.0 TFLOP/s (vs 5.8 / ~10.7 standalone) — the mutual coupling from v2, now across intensities.

The mechanism tell: between m2048 and m4096 the hog reached the **same** ~2.2 TFLOP/s, yet the bigger-footprint hog throttled workers ~2.5× more (−8.8% → −21.6%). If the coupling were pure power/FLOP it should track TFLOP/s (flat); instead it tracks matrix **size** (memory footprint: 16MB → 64MB → 256MB per operand). That points at **memory bandwidth / footprint** as a key coupling channel, not just the power rail.

Honest caveat: the no-hog baseline was noisy (3,550 / 4,256 / 3,960 aug/s, ~20% spread, non-monotonic — run variance in the short 80s ANE cell, not clean thermal). So the matrix-size→throttle trend is suggestive, not pinned; and at m8192 footprint and achieved-TFLOP both rose (confounded). The clean discriminator is fp16-vs-fp32 hog at matched matrix (Lpwr2b, running): fp16@4096 = half the footprint + more FLOPs, so the *sign* of its throttle-vs-fp32 separates bandwidth from power.

decision: needs_repeat (coupling real + reproducible; bandwidth-vs-power hinted, not pinned). Skill note: tiny/V64 coreml pure-self-play aug/s is ~±15-20% noisy at 80s cells — bracket or lengthen. next: Lpwr2b discriminator.

## [2026-05-23] L09i-fix-load-v2 | Clean contention test: ANE workers throttle −35% under GPU load (not immune); the ANE throttles the GPU back

fix-load left a confound: in wave-mode lab_train_cell, aug/s is trainer-gated, so the −96% there was a trainer stall, and the ANE workers' held gen-time looked like a "positive lean." v2 removed the confound — **pure self-play** (canonical_sweep, no trainer, no wave barrier), so aug/s is a direct worker-rate. Interleaved A/B, tiny, with/without a GPU hog.

**ANE workers DO throttle: 3,548 → 2,307 aug/s, −35% under the hog.** So the contention-immunity hope is not supported — the ANE is not a free side-channel; it shares the package power budget. The degradation is gentler than CPU/BNNS (Lpwr −82%), but the hog intensities aren't matched, so that's not a clean ranking.

The striking part is the **second direction of the coupling**: the GPU hog reached only **~2.72 TFLOP/s** here, versus ~10.7 TFLOP/s when it ran next to a light trainer in fix-load. The 16 busy worker processes (Core ML/ANE eval + CPU-side MCTS — total package load, not the ANE engine specifically; this cell can't isolate which) draw enough package power to throttle a GPU matmul hog down by ~4×. So the coupling is **bidirectional** — ANE↔GPU each brown the other out, settling at a shared-power equilibrium (workers −35%, hog −75%).

This also reconciles fix-load: there the trainer stalled, the workers idled on the barrier, package power was free, and the hog reached 10.7. The gen=5.1s "held" because the workers were sampled during their brief non-stalled bursts — not because they're immune.

**Strand verdict (clean evidence now):** ANE residency is real (L09i-fix) but offers no self-play win — not throughput (reject at tiny; small re-mapping in progress), not contention-immunity (this lane). The honest close.

decision: reject (no contention-immunity win); "positive lean" retracted. The bidirectional package-power coupling (ANE throttles GPU to 2.7 TFLOP/s) is a new datapoint for the coupling page + Lpwr2. next: finish the clean throughput re-map (reopen-small-b @ batch 96, reopen-medium), then close the strand.

## [2026-05-23] L09i-fix-load | INCONCLUSIVE — the −96% was a GPU-trainer stall, not an ANE-worker collapse (Reviewer caught my misread)

L09i-fix left one hope standing: the ANE-resident path loses on raw throughput but fully vacates the GPU, so maybe its value is *contention-immunity* — surviving a heavy GPU trainer where the CPU/BNNS path (Lpwr: −82%) does not. This lane tried to test that: interleaved A/B, ANE-resident workers, with and without a concurrent ~10.7 TFLOP/s GPU hog. The holistic number cratered — **7,878 → 302 aug/s, −96%** — and I filed it as "contention-immunity falsified, ANE is the most load-fragile path." **That was wrong**, and the Reviewer caught it by reading the trainer.log I hadn't fully attributed.

**What actually happened.** The hog arm's trainer.log: pre-hog epochs `(8.0s: gen=5.1s train=2.5s)`, `(8.1s: gen=5.2s train=2.6s)`; then with the hog active, epoch 4 `(107.9s: gen=5.1s train=99.5s)`. **Worker generation held exactly — gen=5.1s — under the hog. The ANE workers were not throttled at all.** What collapsed was the *MPS trainer*: its epoch train phase went 2.5s → 99.5s. (The per-step `trainer_step_p50` barely moved, 0.0162→0.0202 — so the 99.5s is MPS-command-queue/blocking contention with the hog, not per-step SGD compute.) Because wave-mode synchronizes worker output to the trainer's epoch loop, the stalled trainer gated generation: only 4 epochs completed in the 120s window vs 19 without the hog, so aug/s tanked. The −96% is a trainer stall propagated through the barrier, **not** an ANE-worker collapse.

That inverts the read. The one clean signal here actually *leans positive* for the ANE: its workers kept generating at full speed while the GPU was saturated — the opposite of the CPU/BNNS workers in Lpwr, which genuinely slowed. So the "ANE −96% vs CPU −82%" comparison is invalid (different things collapsed), and **the strand is not closed** — it's reopened, with a positive lean on the worker-resistance question.

The lane is also confounded for the production question: a real heavy trainer is GPU-heavy by doing its *own* SGD, not by a separate hog flooding the MPS command queue. So this synthetic-hog cell doesn't cleanly model "heavy production trainer + ANE workers" anyway.

decision: needs_repeat (inconclusive). Retracted: the "falsified / strand-closed / second-collapse-datapoint" claims. Next: L09i-fix-load-v2 — decouple worker gen-rate from the wave barrier, and use a trainer-representative GPU load. Lesson for me: when a holistic metric collapses, attribute it to a phase (gen vs train) from the log *before* writing the conclusion — the wave barrier makes aug/s a trainer-gated number.

## [2026-05-23] L09i + L09i-fix | The ANE was one symbolic batch dim away the whole time — residency restored, envelope re-opened

The load-bearing diagnostic of the ANE strand, and it paid off. L09e' had resolved that our Core ML self-play workers run on CPU/BNNS, not the ANE — but the 2026-05-22 scout export *had* hit the ANE rail on the same model. So which export property loses residency? L09i diffed the two `.mlpackage` op graphs.

**Finding: they're byte-identical.** No `gather`, no dilated conv, no ND-broadcastable op — the MIL op histograms match exactly. The only ANE-relevant difference is the **input batch dimension**: `coreml_evaluator.export_model_to_coreml` hardwired `ct.RangeDim(1, max_batch)` (a *symbolic* batch), while the scout's `--batch-shape fixed` declares a *static* one. The ANE requires fully static input shapes; with a symbolic dim, Core ML silently compiles the entire program to CPU/BNNS. That single `RangeDim` is why every L09* "ANE" lane (small −41.5%, tiny +33.9%, medium −59.6%) was actually CPU/BNNS. The "ANE" envelope we mapped over five lanes was never the ANE.

**L09i-fix — static batch restores genuine residency.** Swapped `RangeDim` for a single fixed batch dim (the evaluator pads each leaf-batch up to it and slices outputs back; chunks larger ones). `EnumeratedShapes` was the obvious "few sizes, pad to nearest" escape — but it falls back to BNNS too, so a single fixed batch is the only ANE-placeable option. Residency confirmed twice with hollance's no-sudo `sample` technique: in an isolated micro-probe, and **under the live self-play worker** — hot path `AneInferenceOperationImplUsingAnefAPIs` → `_ANEClient doEvaluateDirect` → `AppleNeuralEngine`, zero BNNS lines. First time the lab's workers have actually run on the Neural Engine.

**But out-of-the-box it cratered.** The worker exported at `wave×G×2 = 1024` and the static model pads *every* eval to that one size — over a ~140-leaf wave tile, a ~7× compute tax. L09i-fix: **2,304 aug/s (−78.6%)**, despite the best trainer-step the lab has ever seen (0.0155s — workers fully off the GPU). Sizing the fixed batch to the tile (`wave×3 = 192`) recovered 3.3×:

| cell | fixed batch | aug/s | games/s | epochs/win | trainer_step |
|---|---|---|---|---|---|
| L09i-fix | 1024 (~7× pad) | 2,304 | 9.05 | 8 | 0.0155 |
| **L09i-fix-b** | 192 (~1.37× pad) | **7,698** | 36.45 | **18** | 0.0172 |
| R-TRAIN-TINY torch | — | 8,039 | 32.48 | 6 | 0.0319 |
| L09c (CPU/BNNS) | — | 10,762.6 | 49.43 | 7 | 0.0267 |

**Verdict: throughput reject** at tiny/V=64 (7,698 < torch 8,039 < CPU/BNNS 10,762) — but a confirmed capability win that re-opens the whole ANE strand. The standout is the trainer axis: **18 epochs in the window vs 6 (torch) / 7 (CPU/BNNS)** because ANE workers contend for the GPU *not at all*. That's the property that should matter when the trainer is heavy or the GPU is contended — exactly the regime where Lpwr showed CPU/BNNS workers collapse −82% under GPU load. The ANE sits on a different engine; whether that buys contention-immunity is the next headline lane (L09i-fix-load).

decision: L09i RESOLVED; L09i-fix/-b REJECT-on-throughput, mechanism win. next: tighter batch (fix-c); ANE-under-GPU-load (fix-load); re-open L09/L09d (small/medium) with real ANE residency.

## [2026-05-23] Lhot | Heat-soak measured: production shapes have NO haircut (hypothesis refuted)

Jason: "heat soaked numbers are not bad to know, training will be heat soaked." Correct instinct — so we measured it instead of assuming. Lane Lhot: 8 back-to-back R-S400 cells (small/V=512/fp16, 60s each) to drive the chip to thermal steady state, then 2 R-TRAIN-WL5 cells while heat-soaked.

**Result — the M5 Max sustains production throughput; cold-start refs are trustworthy.**

R-S400 curve (aug/s, iters 1-8): 9641 → 9388 → 9660 → 10029 → 9902 → 9780 → 9781 → 9788. It **wobbles through warmup then settles stable at ~9,783** — no thermal decay over 8 minutes of continuous load. Steady state +4% above the cool-start reference (9,398.5). R-TRAIN-WL5 heat-soaked: 3,384 / 3,379 aug/s, trainer_step 0.052, 14 epochs — +2.5% above cool (3,297.6), and trainer_step matches L10's cool 0.0512.

**Both production shapes are at-or-above their cool-start references after heat-soak.** There is no haircut. The hypothesis (cool-start overstates sustained production) is refuted for the production shapes.

**Self-correction:** I had committed (a813151) a "~18% haircut, cool-start is optimistic" claim, derived from the tiny/V=64 baseline falling 10,431→8,531. That was wrong — that number was a Core ML *CPU/BNNS-worker* shape measured right after the synthetic 14-TFLOP hog (artificial extreme GPU thermal load, non-production shape). Corrected across best-cells, the coupling page, and the memory. The lab working as intended: hypothesis tested, refuted by measurement, surfaces corrected.

**Surviving nuance:** the haircut may be engine-specific — GPU-resident work sustains its clocks; the CPU/BNNS path *may* throttle under sustained heat/power (fits the Lpwr power-coupling story). One messy data point on a non-production shape; Lhot2 would re-test cleanly. The Lpwr GPU-coupling collapse (−82%) remains real but needs EXTREME GPU load — normal training doesn't reach it.

decision: needs_repeat (production conclusion solid; CPU-throttle nuance needs clean re-test). New heat-soaked datapoints: R-S400 ≈ 9,783, R-TRAIN-WL5 ≈ 3,381.

## [2026-05-23] L09e' + Lpwr | Residency resolved (CPU/BNNS, not ANE) + GPU load collapses CPU workers

Post-session-end addendum, driven by Jason flagging the [hollance/neural-engine](https://github.com/hollance/neural-engine) repo as inbound ANE research. Two findings, both "where the machine breaks" material.

**L09e' — residency resolved.** hollance documents a no-sudo way to check ANE residency: `sample <pid>` and look for `H11ANEServicesThread` / Espresso engine attribution (ANERuntimeEngine=ANE, MPSEngine=GPU, BNNSEngine=CPU). Re-ran the L09c shape (replicated cleanly at 10,431.6 aug/s) and sampled the workers: **no ANE thread; hot path is `E5RT::Ops::BnnsCpuInferenceOperation::ExecuteSync` — the CPU/BNNS engine.** Independent confirmation: Jason's system GPU monitor showed **ANE utilization 0%** during the run. Cross-checked under CPU_AND_GPU (L09c-cpugpu = 10,202 aug/s) — STILL BNNS-CPU; Core ML picks CPU for our tiny model even when GPU is allowed. **Verdict: the L09c PROMOTE is `coreml-isolated` via CPU/BNNS, NOT ANE residency.** The "tiny model fits the ANE design center" hypothesis is falsified — Core ML chose CPU for tiny just like small/medium; tiny wins only because BNNS-CPU is fast enough at tiny/V=64 to beat torch+MPS-contended workers. ANE was never in play.

**Lpwr — the engines share a package resource.** Jason: "what if we artificially load the gpu... we have it screaming but the gpu has plenty of headroom." Built `scripts/gpu_load_generator.py` (fp32 matmul hot loop, ~11-14 TFLOP/s on MPS) and ran it concurrently with the L09c CPU-worker cell. Clean back-to-back A/B on a cool chip:

| arm | worker aug/s | trainer_step | engine placement (sampled) |
|---|---|---|---|
| no hog | 10,431.6 | 0.0267 | workers→CPU/BNNS, trainer→GPU |
| GPU hog ~11 TFLOP/s | **1,905.2** | 0.0305 | + hog→GPU |
| delta | **−81.7%** | +14% | — |

**The CPU workers collapsed 82% when the GPU was saturated** — even though they're on the CPU. The asymmetry (trainer on GPU −14%, workers on CPU −82%) points at a shared **power/thermal envelope**: a GPU pinned at ~11 TFLOP/s eats the package power budget, the CPU throttles, BNNS convolutions slow. **This makes the L09c win load-fragile** — it depends on GPU power headroom (true for our light tiny-model trainer; false for a heavy production trainer at 15×15).

**The intensity sweep got thermally confounded** (and that's a finding too). Sweeping hog matrix dim {0,2048,4096,8192} sequentially gave a non-monotonic result (8192 beat 4096 at ~same TFLOP/s) because the baseline itself fell 10,431→8,531 over ~20 min of heat-soak — thermal state dominated the intensity axis. **The trustworthy measurement is the tight cool-chip back-to-back A/B, not the spread-out sweep.** Mechanism (power vs scheduling vs memory-bandwidth) remains unpinned; needs a cold-chip interleaved-A/B design with cooldowns + powermetrics. Logged as a worked example of friction-lesson #2 (session-thermal drift) defeating an experiment.

**Net for the lab:** "make the Mac sing" has a power-budget ceiling — you cannot run CPU + GPU + ANE at full tilt simultaneously; pushing one steals headroom from the others. Engine offload helps by *balancing* load under that ceiling, not by summing peak rates. Full writeup: [m5-max-cross-engine-coupling.md](../topics/m5-max-cross-engine-coupling.md). decision: needs_repeat (clean mechanism pin pending cold-chip re-run). New memory: [[project-light-all-engines]].

## [2026-05-23] L09e + session-end | Routing axis null at small/V=64; ANE envelope snapshotted (not a verdict)

L09e was the final session lane — the routing-rescue diagnostic for L09's small/V=64 -41.5% reject. Two cells under live training:

| compute-units | aug/s | vs L09 CPU_AND_NE (1,930.3) | trainer_step_s_p50 |
|---|---|---|---|
| CPU_AND_GPU | 1,908.3 | -1.1% | 0.0226s |
| ALL (auto) | 1,989.8 | +3.1% | 0.0197s |
| L09 ref (CPU_AND_NE) | 1,930.3 | — | 0.0227s |

Across-routing spread: 4.3% — within natural noise. **All three routings still ~40% below R-TRAIN-WL5 (3,297.6 aug/s).** ALL is the marginal winner but doesn't approach the torch/MPS baseline. The trainer_step_s_p50 is clustered around 0.02s across all three routings (vs R-TRAIN-WL5's 0.05s) — the MPS-relief mechanism is real in all three routings, consistent with L09's original finding. The worker-side raw eval throughput on Core ML is just slower than torch/MPS at this workload size, and routing hint doesn't change that.

**Engine envelope, decisively mapped now (5 measured comparison points across the ANE family):**

| model / shape | engine config | aug/s | vs matched baseline |
|---|---|---|---|
| **tiny / V=64** | **ANE** (CPU_AND_NE) | **10,762.6** | **+33.9%** ← the lone win (L09c) |
| tiny / V=512 | ANE | 10,609.8 | -24.0% (L09c-V512) |
| small / V=64 | ANE (CPU_AND_NE) | 1,930.3 | -41.5% vs R-TRAIN-WL5 (L09) |
| small / V=64 | ANE (CPU_AND_GPU) | 1,908.3 | -42.1% (L09e) |
| small / V=64 | ANE (ALL) | 1,989.8 | -39.7% (L09e) |
| medium / V=512 | ANE | 591.7 | -59.6% (L09d) |

**At today's Core ML version + today's evaluator pipeline + today's model arch family, ANE pays at exactly one point in our envelope: tiny + V=64.** Across measured axes (V, model-size, routing), ANE is currently null-to-negative outside that single point. The L09c PROMOTE remains the lone win; the ANE chapter is **paused at a complete current-state snapshot, not concluded** — future ANE research (new ANE features in macOS / Xcode, Core ML major-version updates, evaluator-pipeline changes, model-arch family changes) will warrant a re-measurement of this envelope. Jason flagged inbound ANE research; when it lands, re-run L09c/L09d/L09c-V512/L09e against the new baseline before assuming today's snapshot holds.

**Session-end summary (12 lanes since session-restart, headline-ordered):**

- **L09c PROMOTE** (the headline): R-TRAIN-TINY-ANE = 10,762.6 aug/s (+33.9% vs matched torch baseline) — Core ML/ANE pays at the tiny model size under live training, where both backends are pipeline-overhead-bound and trainer-side MPS-relief tips the balance.
- L09d REJECT: medium/V=512 ANE -59.6%; "larger compute amortizes" falsified.
- L09c-V512 REJECT: tiny/V=512 ANE -24.0%; V-axis amortization falsified.
- L08 REJECT: heap-ratio axis null at R-S400/fp16; bandwidth-bound regime confirmed.
- L09e REJECT: compute-units routing null at small/V=64; L09 reject is final.

**Reference points opened in this session:**
- R-TRAIN-TINY-ANE = 10,762.6 aug/s (engine-mapping ref)
- R-TRAIN-TINY = 8,039.1 aug/s (engine-mapping ref, torch baseline arm)
- R-TRAIN-MEDIUM = 1,463.3 aug/s (engine-mapping ref, torch+fp16 baseline arm)
- R-TRAIN-MEDIUM-ANE = 591.7 aug/s (rejected ref, engine-mapping data point)

**consecutive_rejects: 4** (L09d → L09c-V512 → L08 → L09e). One short of the 5-reject HALT threshold. **Session-end declared by orchestrator** because: (1) the four rejects are all envelope-mapping with clean mechanism (not knob-failure noise); (2) the current ANE envelope is fully snapshotted (L09c PROMOTE + 4 rejects = complete current-state map); (3) remaining queueable lanes (L09f, L09g, L09h) are all downweighted or low-upside diagnostic with no compound mechanism left to test against today's Core ML; (4) the L11b' R-TRAIN-LEAN-fp16 +152.9% perf-reference from the prior session-portion remains the perf cycle's headline R-TRAIN finding, unaffected. **Future-shape note:** this is the right time to pause the ANE axis because the inbound new ANE research will likely reset the priors; re-running L09c/L09d/L09c-V512/L09e against the new baseline is the natural next-session move when it drops.

**Three NEW friction-smoothing lessons surfaced this session-portion, filed to the gomoku-perf-lab skill:**

1. **plies_mean is NOT stationary across asymmetric-epoch R-TRAIN cells.** When two arms have very different `epochs_in_window` counts (because the candidate's trainer epoch is much shorter), aggregate plies_mean drift is dominated by within-window training progress, not engine-induced game-shape drift. Future Reviewers' drift-watch should check per-epoch plies values in trainer.log, not just the aggregate. Surfaced in L09c-V512 (-7.3% plies_mean drift; per-epoch trainer.log showed monotonic-from-epoch-3 decline from 33.5 → 27.7 as the candidate's 8 trainer epochs improved the policy).

2. **Session-thermal drift can produce ~5% absolute aug/s drift over 90-min sessions.** Within-lane back-to-back A/B remains reliable (chip stable across minutes). Cross-time comparisons (> ~30 min apart in same session OR across sessions) should re-measure under matched thermal state. Surfaced in L08 (default-heap re-measure -4.9% vs R-S400 measured 90 min earlier).

3. **Env-axis lanes: use the L08-driver cells.csv `env` column, not shell-prefix env.** Shell-prefix env propagates via Popen inheritance but doesn't stamp the env_overrides field in metadata.txt, so the on-disk artifacts don't discriminate which env value each cell ran under. Reviewer flagged this as a soft artifact-capture gap on L08; future env-axis lanes should use the cells.csv `env` column for per-cell stamping.

These three lessons go into the skill's Friction-smoothing log at session-end (companion commit to ~/.claude/skills/gomoku-perf-lab/SKILL.md).

Reviewer APPROVE on L09e (math reconciles, routing axis null with ALL marginal-winner at +3.1% but still ~40% below R-TRAIN-WL5; session-end is the correct triage per the envelope-mapping-vs-knob-failure distinction; all 5 surfaces touched under the 5-min cap). Session-end commit landed.

## [2026-05-23] L08 | Heap-ratio axis null at R-S400/fp16 — bandwidth-bound regime confirmed; thermal drift surfaced

Pivoted to L08-mps-heap-ratio post-L09c-V512 reject. With the ANE-axis nearly exhausted (single-point envelope at tiny+V=64), MPS-side knob tuning becomes the next-best perf lever. L08 tests whether PYTORCH_MPS_HIGH_WATERMARK_RATIO (default ~1.7 on M-series) is capping throughput at the R-S400 reference (small/V=512/fp16).

**Result: REJECT — flat axis at R-S400/fp16.** Three cells back-to-back:

| heap ratio | aug/s | vs default |
|---|---|---|
| default (implicit ~1.7) | 8,937.3 | — |
| 2.0 (higher) | 8,870.9 | -0.7% |
| 0.0 (unlimited) | 8,927.7 | -0.1% |

Within-sweep spread: 0.74% — well below the V=512 plateau noise floor (~0.2-2%). Mechanism: at small/V=512/fp16 the workload is **bandwidth-bound** (per L06-followup's +97% fp16 finding), not MPS-memory-pressure-bound. The heap watermark ratio governs when MPS frees memory, which doesn't change eval bandwidth. Null result was mechanistically predictable once the bandwidth-bound regime was understood.

**Side-effect data point: session-thermal drift surfaced.** The L08 default-heap re-measure (8,937.3) is -4.9% vs R-S400 (9,398.5) measured by L06-followup ~90 min ago at session-start. Within-L08 the three cells are 0.74% apart (back-to-back, chip in same thermal state). The 4.9% drift between session-start and ~10 sequential cells later is most plausibly thermal: M5 Max's sustained-load throughput drops as the chip warms. Implications:
- **Within-lane back-to-back A/B comparisons remain reliable** (chip thermal state stable across minutes).
- **Cross-lane comparisons against numbers measured far apart in time may have a thermal-drift confound on the order of 5%.** R-S400 measured at session-start ≠ R-S400 measured 90 min in.
- This is a friction-smoothing data point worth filing in the session-runbook: when comparing against a reference measured in a different session OR > ~30 min ago in the same session, consider a re-measure under matched thermal state.

**The queue state, post-L08:**
- consecutive_rejects: 2 → 3 (now at the warning level per the charter's stop-gates triage matrix).
- Per [feedback-lab-runs-forever] and the 2026-05-23 lesson on stop-gates: at 3 rejects, the call is still CONTINUE as long as Tier-3 lanes are queueable AND there's a compound mechanism left to test.
- Remaining queueable lanes: L09e (Tier 3, priority 3.0, diagnostic-only — answers "is Core ML demoting ops?" for L09d/L09 reject mechanism); L09f (Tier 3, priority 2.5, downweighted by L09c-V512); L09g (Tier 3, priority 2.0, downweighted); L09h (Tier 3, priority 1.0).
- **All remaining work is diagnostic or low-upside.** No queueable lane has a headline-moving expected delta.
- Triage call: file L08 receipt + Reviewer spawn (this commit); then dispatch L09e as the next-best diagnostic to close the ANE-envelope-mapping chapter cleanly. If L09e also rejects (likely), session-end is a natural pause point.

**Headline of the session (post-L08):**
- L09c PROMOTE: R-TRAIN-TINY-ANE = 10,762.6 aug/s (+33.9% vs torch baseline) — the lone ANE win, single-point envelope at tiny+V=64.
- L09d REJECT: R-TRAIN-MEDIUM-ANE -59.6% — "larger compute amortizes" falsified.
- L09c-V512 REJECT: tiny+V=512 ANE -24.0% — "V-axis amortizes" falsified.
- L08 REJECT: heap-ratio null at R-S400/fp16 — bandwidth-bound regime confirmed, MPS-side knob axes exhausted.
- The L09c PROMOTE is the headline. The 3 rejects together SHARPLY map the engine envelope; that's not noise, it's evidence.

Reviewer pending. Receipt commit pending.

## [2026-05-23] L09c-V512 | V-axis amortization falsified at tiny; ANE win is a single-point envelope

Auto-queued from L09c PROMOTE. The L09f generic hypothesis says V=512+ batches more leaf evals per Core ML forward, so the pipeline overhead amortizes better. L09c-V512 tests this at tiny (the only model size where ANE wins, per L09c +33.9% at V=64).

**Result: REJECT at -24.0% aug/s.** torch+fp16 already extracts most of the V=512 bandwidth-bound value at tiny (per L06-followup, tiny + V=512 + fp16 was only +3.6% over fp32 because tiny is MPS-dispatch-limited, not bandwidth-bound). At V=512 the torch+fp16 baseline (13,968.6 aug/s under live training) is harder to beat than at V=64 (where baseline was 8,039.1). Core ML can't match torch+fp16's bandwidth utilization at this operating point.

| | candidate (ANE) | baseline (torch+fp16) | delta |
|---|---|---|---|
| aug/s | 10,609.8 | **13,968.6** | **-24.0%** |
| games/s | 43.94 | 52.18 | -15.8% |
| trainer_step_s_p50 | 0.0268s | 0.0714s | -62.5% (MPS-relief still real) |
| plies_mean | 31.02 | 33.47 | -7.3% (asymmetric-epoch artifact — see below) |
| epochs in window | 8 | 1 | — |

**The plies_mean drift is NOT behavior drift.** The candidate's 8 trainer epochs include training-progress; per-epoch plies in candidate trainer.log peaks at epoch 3 then descends: 30.9 → 33.1 → 33.5 → 33.4 → 31.7 → 30.4 → 29.4 → 27.7 (net 30.9 → 27.7 over the window). The aggregate plies_mean is dominated by the last few epochs where the policy has started to play better (faster wins). The baseline's single epoch (plies=32.5) reflects pre-training state. This is a measurement-window confound from asymmetric training progress, not engine-induced game-shape drift. **New friction-smoothing lesson for the lab:** plies_mean is NOT stationary across asymmetric-epoch R-TRAIN cells; future Reviewers' drift-watch should check per-epoch trainer.log plies values when arms have very different epochs_in_window.

**The engine envelope, now with 4 measured points and a single-point ANE win:**

| model / shape | engine | aug/s | vs matched baseline |
|---|---|---|---|
| **tiny / V=64** | **ANE** | **10,762.6** | **+33.9%** (L09c — the lone win) |
| tiny / V=64 | torch | 8,039.1 | — |
| tiny / V=512 | ANE | 10,609.8 | **-24.0%** (L09c-V512 — V-axis falsified) |
| tiny / V=512 | torch+fp16 | 13,968.6 | — |
| small / V=64 | ANE | 1,930.3 | -41.5% (L09 vs R-TRAIN-WL5) |
| small / V=64 | torch | 3,297.6 | — |
| medium / V=512 | ANE | 591.7 | -59.6% (L09d) |
| medium / V=512 | torch+fp16 | 1,463.3 | — |

**ANE pays at exactly one point in our envelope: tiny + V=64.** Not "tiny in general" (V=512 loses) and not "low-V in general" (small+V=64 loses too). The single-point win is the L09c +33.9% — a narrow regime where worker per-call compute is so light that both backends are pipeline-overhead-bound, and the trainer-side MPS-relief tips the balance.

**Implications for the queue:**
- L09f (broader V-axis sweep at small/medium) is now downweighted — V-axis amortization is falsified at tiny, unlikely to suddenly pay at larger models. Queue for completeness, not priority.
- L09g (broader model-size sweep at V=512) is also downweighted — V=512 ANE is losing at both tiny and medium; queue for completeness.
- **L09e (compute-units routing sweep) keeps priority 3.0** — it's now the ONLY remaining diagnostic that could rescue any ANE result. If a non-CPU_AND_NE routing reveals that Core ML was silently demoting ops at small/medium, the L09 / L09d rejects might be misattributed. L09e at small/V=64 and medium/V=512 is the load-bearing test.
- **L11b' R-TRAIN-LEAN-fp16 (+152.9% vs R-TRAIN-WL5) remains the perf cycle's headline R-TRAIN finding** — ANE-offload is unambiguously not the path to the next R-TRAIN promote at any model size where it matters.
- **L08-mps-heap-ratio (Tier 3, priority 2.6) moves up in relative priority** — with ANE-axis nearly exhausted, MPS-side knob tuning becomes the next-best lever (3 cells, ~5 min wall).

**consecutive_rejects: 1 → 2.** Still far below the 5-reject charter halt threshold. The two rejects (L09d, L09c-V512) are envelope-mapping rejects with clean mechanism — not a knob-failure streak. Lab continues per autonomous-loop charter.

Reviewer pending. Receipt commit pending.

## [2026-05-23] L09d | ANE doesn't pay at medium — envelope sharply mapped

The high-prior follow-up to L09c. Hypothesis: at medium (~1.5M params) the per-call compute is large enough that Core ML's pipeline overhead amortizes even better than at tiny (where L09c showed +33.9%); combined with trainer-side MPS-relief, R-TRAIN-MEDIUM-ANE should beat the torch+fp16 baseline. **Hypothesis falsified.**

**Calibration note (carried lesson from L11).** The 120s default measurement window only caught 2 epochs at medium V=512 (trainer epoch is ~50-90s as buffer fills). Per friction-smoothing log "R-TRAIN cells need a window that spans ≥ 3 of the trainer's actual epochs", re-dispatched the baseline at 240s, then dispatched the candidate at matched 240s. The 240s baseline rerun (1,463 aug/s) was within 2% of the 120s read (1,489 aug/s), so the conclusion didn't shift — but the 3-epoch matched window IS the right basis for the receipt's official delta.

**The two cells (matched 240s windows):**

| | L09d-candidate (ANE) | L09d-baseline (torch+fp16) | delta |
|---|---|---|---|
| aug/s | 591.7 | **1,463.3** | **-59.6%** |
| games/s | 2.33 | 5.66 | -58.8% |
| epochs/s | 0.0208 | 0.0042 | +395% |
| trainer_step_s_p50 | 0.0444s | 0.2391s | **-81.4%** |
| plies_mean | 31.97 | 32.5 | -1.6% (Reviewer's L09c drift-watch confirmed null) |
| epochs in window | 7 | 3 | — |

**The mechanism, sharp split.** Trainer side wins ENORMOUSLY: per-epoch train= field collapsed from 11-86s (torch+fp16 baseline, growing with buffer fill) to a flat 2-3s (ANE candidate). The trainer simply flies when MPS is uncontended at medium-model SGD work. Workers side LOSES enormously: per-epoch gen= field expanded from ~6s (torch+fp16) to 30-40s (ANE). The worker loss is 5-7× the trainer gain, so the holistic balance is firmly negative at medium V=512.

**Engine envelope, now sharply mapped:**

| model / shape | engine | aug/s | vs matched torch (or vs WL5 for L09) |
|---|---|---|---|
| tiny / V=64 (L09c) | ANE | 10,762.6 | **+33.9%** |
| tiny / V=64 (L09c-baseline) | torch | 8,039.1 | (baseline) |
| small / V=64 (L09) | ANE | 1,930.3 | **-41.5% vs R-TRAIN-WL5 3,297.6** |
| small / V=64 (R-TRAIN-WL5) | torch | 3,297.6 | (baseline) |
| medium / V=512 (L09d) | ANE | 591.7 | **-59.6%** |
| medium / V=512 (L09d-baseline) | torch+fp16 | 1,463.3 | (baseline) |

**ANE pays at TINY only.** The "larger compute amortizes pipeline overhead better" hypothesis (L09d's bet) is the OPPOSITE of what we see in our envelope. Two readings of why:

1. **Core ML's eval throughput per forward at medium V=512 may be much lower than torch/MPS+fp16.** This is the simplest read — Core ML's CPU_AND_NE routing might be silently demoting some ops to CPU/GPU at this model size, or the ANE itself just doesn't deliver the throughput torch+MPS can at this size. **L09e (compute-units routing sweep) is the diagnostic** to distinguish "ANE is just slow at medium" from "Core ML silently demoted ops".
2. **Bandwidth-bound regime favors MPS+fp16 sharply at medium V=512.** L06-followup showed fp16-eval nearly doubles small/V=512 on MPS (+97.2%). Medium V=512 is even more bandwidth-bound (L06fu-extended estimated +62% fp16-alone at medium). Core ML's internal FLOAT16 doesn't capture the same bandwidth savings as torch+--fp16-eval at this operating point, OR ANE's memory hierarchy just doesn't deliver MPS-level bandwidth.

**Decision: REJECT.** No promote — R-TRAIN-MEDIUM-ANE = 591.7 logged as a rejected envelope-mapping data point. R-TRAIN-MEDIUM = 1,463.3 (torch+fp16 arm) opens as a new envelope-mapping ref.

**Next-step implications:**
- L09e priority bumped 1.5 → 3.0 — diagnostic value just spiked. If ANE-residency is non-uniform across compute-units routings, the L09d reject might be "Core ML demoted ops to CPU/GPU at medium" rather than "ANE is just slow at medium" — different conclusion.
- L09c-V512 (tiny + ANE + V=512) priority kept at 4.5 — the L09c +33.9% finding is still load-bearing; V-axis amortization could still pay at tiny.
- L09f / L09g (the broader V-axis and model-size envelope sweeps) move down — the headline finding ("ANE pays at tiny only") makes the broader sweep less load-bearing than diagnostics on the medium-loss mechanism.
- L11b' R-TRAIN-LEAN-fp16 (+152.9% vs R-TRAIN-WL5) remains the perf cycle's headline R-TRAIN finding; ANE-offload is **not** the path to the next R-TRAIN promote at production-quality model size.

**consecutive_rejects: 0 → 1.** (L09c was promote; L09d is reject. The "5-reject-streak" charter threshold remains far away; the lab continues.)

Reviewer pending. Receipt commit pending.

## [2026-05-23] L09c | ANE pays at tiny — engine envelope maps along the model-size axis

Resumed the lab from the session-end RESUME STATE. Box was idle; consecutive_rejects=0; top-of-queue named lane was **L09c — tiny model on Core ML / CPU_AND_NE under live training**, the second of L09's two follow-up candidates (the first, L09b, blocked on the fp16 × coreml interaction). Hypothesis: at tiny (~30k params) the per-call ANE pipeline overhead amortizes well enough that even with slower-than-MPS raw eval, the trainer-side MPS-relief from L09 tips the holistic R-TRAIN-* aug/s win positive — opposite outcome to L09 (small model, ANE -41%).

**Lane-card under-spec note.** L09c's card listed `n_cells: 1` (just the ANE candidate). But the hypothesis explicitly compares to "workers on MPS torch fighting the trainer" at the tiny model size — and we have **no prior tiny under live training measurement**. R-TRAIN-WL5 is small, R-S400-tiny is pure self-play (no trainer pressure). So the candidate-only number would have been a homeless data point. Dispatched the matched torch baseline immediately after the candidate (~5 min total wall, comparable thermal/scheduler state). The friction-smoothing lesson here: when a lane card asks for n_cells:1, sanity-check the comparison anchor first — if the anchor doesn't exist as a prior measurement, the card is implicitly under-spec'd. Filed as L09c with both arms.

**The two cells:**

| | L09c-candidate (ANE) | L09c-baseline (torch) | delta |
|---|---|---|---|
| aug/s | **10,762.6** | 8,039.1 | **+33.9%** |
| games/s | 49.43 | 32.48 | +52.2% |
| epochs/s | 0.0417 | 0.0333 | +25.2% |
| trainer_step_s_p50 | 0.0267s | 0.0319s | -16.3% (trainer faster on ANE config) |
| plies_mean | 29.02 | 31.84 | -8.9% (within sampling band) |
| epochs in window | 7 | 6 | — |

**The mechanism, in one paragraph.** L09 confirmed that offloading workers from MPS to ANE relieves trainer-side contention (trainer_step_s_p50 -56% at small) but at small/V=64 the worker-side raw-eval gap is too large (Core ML eval ~2× slower than torch/MPS) — net -41%. At tiny, the per-eval compute is so light that **both backends are pipeline-overhead-bound**: torch/MPS has its own dispatch overhead floor, and Core ML has its pipeline overhead, and the two get within striking distance. The trainer-side MPS-relief (replicated here at -16% trainer_step_s_p50, smaller magnitude because trainer wasn't as contended to begin with on tiny) then tips the balance positive. Engine envelope along the model-size axis: **ANE pays at tiny, doesn't pay at small**. The crossover sits between tiny and small in our shape envelope.

**Implications for L09d (medium on ANE).** L09d sits in the high-prior slot now. Its hypothesis ("medium amortizes pipeline overhead even better than tiny because more compute per call") leans on the same amortization logic L09c confirmed at the small-end. If the trend is monotonic with model size (more per-call compute = better ANE amortization = larger win or smaller loss), medium is the most interesting test point — and the most production-relevant. L09c just established that the amortization argument has empirical legs in our envelope, not just generic Core ML marketing claims. Queue bumped accordingly.

**Decision: PROMOTE** as a new envelope-mapping reference family — R-TRAIN-TINY-ANE (10,762.6) and R-TRAIN-TINY (8,039.1, torch arm). NOT a R-TRAIN-WL5 substitute (different model size, different quality target); this is engine-fit research, not a production recipe.

**consecutive_rejects stays at 0.** Reviewer pending.

## [2026-05-23] session-end | the mac is singing

Autonomous-loop session opened with Jason's directive to "make this mac SING" and orchestrate the perf cycle without manual intervention. 12 lanes later, the headline numbers:

| Reference | WL5-era / fp32 default | This session's best | Δ vs WL5 baseline |
|---|---|---|---|
| **R-S400** | 4,765 (fp32, V=512 from L01) | **9,398.5** (fp16, V=512) | **+194.8% vs WL5 V=64=3,188** |
| **R-S200** | 9,156 (fp32) | **16,850.8** (fp16) | **+180.5% vs WL5 V=64=6,006** |
| **R-S100** | 15,082 (fp32) | **22,312.1** (fp16) | **+100.0% vs WL5 V=64=11,151** |
| **R-S400-tiny** | 22,088 (fp32) | **22,873.8** (fp16) | **+212.2% vs tiny V=64=7,326** |
| **R-S400-medium** (new ref) | n/a | **3,377.2** (fp16, V=512) | +142% vs medium V=64=1,393 |
| **R-TRAIN-WL5** | n/a (TBD) | **3,297.6** (V=64, fp32 — first-ever baseline) | +0% (reference) |
| **R-TRAIN-LEAN-fp16** (new perf ref; TQ-gated for production) | n/a | **8,340.5** (V=512, sgd=0.001, fp16) | **+152.9% vs R-TRAIN-WL5** 🔥🔥 |

**The compound mechanism, in one paragraph.** Live-training throughput on the M5 Max is determined by how trainer and workers share MPS. At WL5 defaults (V=64, sgd=0.0025, fp32) the trainer runs ~7 SGD steps/sec on a small model, workers run ~14 games/sec, and the chip is moderately contended. At V=512 alone the trainer's per-position SGD work multiplies (2.4× buffer fill speedup × 0.0025 ratio = 3.36× more steps/epoch), the trainer monopolizes MPS for 43s of every ~52s epoch, and the workers starve — net loss. At V=512 + low-sgd (0.001) the trainer's per-epoch work is capped, MPS is shared, workers regain their game/s, and the gen-side V=512 win compounds at trainer level (+28% vs WL5). And fp16 on the eval-only worker model halves the bandwidth requirement for the model's forward pass, which is the dominant cost when sims is high (bandwidth-bound regime). Stack the two independent levers — fp16 on workers + low-sgd on trainer — and the trainer-level recipe doubles (+152.9%). The mechanism predicted multiplicative composition (1.28 × 1.97 = 2.52); the measurement confirmed (2.53). The chip has more capacity than WL5 was using; the lab now knows how to extract it.

**The 12-lane play-by-play:**

1. **CPU queue × 4 in parallel** (early): L12 lab_train_cell driver shipped (gating; ~720 LOC + smoke test); L05 --compile passthrough; L06 --fp16-eval flag; L08-driver env column on cells.csv. All merged via merge-commit; L05/L06 needed manual conflict resolution against each other in canonical_sweep.py.
2. **L10** (Tier 1, baseline): First R-TRAIN-WL5 measurement at **3,297.6 aug/s**. Surfaced two L12 driver bugs en route: `--save-every=1000000` froze worker_weights publishing (workers stayed on v0 forever); `count_records()` at SIGTERM undercounted by ~30× (trainer ingests + deletes records as it goes). Both patched (1dc4abb + 4a825f1) before the canonical measurement landed. Reviewer APPROVE.
3. **L11** (Tier 1, reject): V=512 default-sgd at trainer level = **2,362.8 aug/s** (-28% vs WL5). Mechanism: trainer monopolizes MPS for 43s/epoch when V=512 fills the buffer 2.4× faster and sgd_per_position stays at 0.0025. Reviewer APPROVE.
4. **L09** (Tier 1, reject with partial-confirm): Workers on Core ML / CPU_AND_NE at small/V=64 = **1,930.3 aug/s** (-41% vs WL5). Trainer side wins (trainer_step_s_p50 -56%); worker side loses (Core ML eval ~2× slower than torch/MPS at this model size). Confirmed MPS contention is real and bidirectionally costly. Reviewer APPROVE.
5. **L11b** (Tier 1, needs_repeat per TQ gate): V=512 + sgd_per_position=0.001 = **4,231.8 aug/s** (+28% vs WL5). The trainer-side cap works; pure-gen V=512 win finally compounds at trainer level. sgd_per_position is behavior-affecting → TQ gate. Reviewer APPROVE (precedent set: behavior-borderline knobs → needs_repeat).
6. **L05-followup** (Tier 3, reject): torch.compile at small + tiny V=512 within noise (-2.3% small, -0.4% tiny). Compile-graph overhead doesn't amortize at 60s smokes. Three rejects in a row triggered the stop signal — but L06-followup was dispatching in parallel as the next compound. Reviewer APPROVE.
7. **L06-followup** (Tier 3, the headline): fp16-eval at small + tiny V=512 = **9,398.5 aug/s small (+97.2%)**, **22,873.8 tiny (+3.6%)**. The historic "fp16 on MPS is slow" claim disproven for our eval workload at torch 2.11.0 + fused conv+bn. Mechanism predicts the asymmetry (small bandwidth-bound, tiny dispatch-bound). consecutive_rejects RESET. Reviewer APPROVE (precedent set: fp16-with-fp32-output-cast = no-behavior-change for the perf lab; verified at mcts.py:519-529).
8. **L06fu-extended** (Tier 3, promote × 3): R-S200 fp16 = **16,850.8** (+84%); R-S100 fp16 = **22,312.1** (+48%); medium V=512 fp16 = **3,377.2** (new ref). Sims-scaling mechanism confirmed monotonic (higher S = more eval-bound = bigger fp16 win). Reviewer APPROVE.
9. **L11b'** (Tier 1, the compound headline, needs_repeat per TQ gate): V=512 + sgd=0.001 + fp16 workers at trainer level = **8,340.5 aug/s** (**+152.9% vs R-TRAIN-WL5**). Mechanism independence predicted multiplicative composition; measured 2.53× vs predicted 2.52× — empirically exact. Reviewer APPROVE (precedent-extending).
10. **L09b** (Tier 1, blocked): Code-interaction bug — `_maybe_half` cast model to fp16 before `torch.jit.trace` inside Core ML export, which expects fp32 dummy. Patched (selfplay_worker parse_args force-sets fp16_eval=False when evaluator=coreml). Also semantically redundant — Core ML already runs FLOAT16 internally. Lane was incoherent as designed.
11. + 12. **Code lanes:** Beyond the four CPU-queue lanes from item 1, the dispatch process surfaced 4 more L12 driver patches (--save-every fix, count_records fix, --evaluator passthrough for L09, --fp16-eval passthrough for L11b'). All shipped in flight.

**Lab discipline:** every receipt-affecting lane drew a Reviewer audit (8 Reviewer spawns, all APPROVE, with two precedent-setting precedents recorded: (i) fp16-with-output-cast counts as no-behavior-change for the perf lab; (ii) behavior-affecting knobs like sgd_per_position get `needs_repeat` and an explicit "PERF LAB ESTABLISHES" / "Production adoption needs canary" separation). The Training-Quality Promotion Gate kept the lab honest: L11b and L11b' both saw their throughput numbers recorded but neither flipped a production default. R-TRAIN-WL5 stays the WL5 production recipe; the new perf reference R-TRAIN-LEAN-fp16 opens a clear handoff point for whoever drives a WL6 canary outside the perf lab.

**Charter staleness flagged 5 times in a row by Reviewers** — `wiki/topics/perf-lab-charter.md:50` R-TRAIN-LEAN row still reads "V=128 (today's promoted gen default)" while L01 promoted V=512 as the gen default, AND L11 rejected V=512 at the trainer level (so the row's framing is doubly out of date), AND L11b' has now established V=512 + sgd=0.001 + fp16 as the new perf reference. Class B (charter modification → user). Surfaced for next charter pass.

**Resuming the lab:** the queue has clean follow-up candidates — L09c (tiny on ANE), L06fu-medium-AB (clean medium fp16 attribution), L08-mps-heap-ratio at the new fp16 reference, L11b'' (sgd_per_position sweep at V=512+fp16). All Tier-3-ish. The Tier-1 architectural lever space (V × sgd × fp16 × Core ML) is well-mapped at small/V=64-512; the next architectural play likely involves model-size scaling (medium more aggressively, or even tiny on ANE) or a deeper trainer-side change (bf16 SGD? trainer-side compile?). Reasonable to call this perf cycle done at the chip-envelope it explored.

> The PyTorch forums told us fp16 on MPS was slow without saying what
> they meant by slow; Apple Core ML docs say small models benefit from
> ANE without saying which small. We measured both: R-S400 fp16 doubled
> (+97%), small/V=64 on ANE lost ~2× on workers, R-TRAIN doubled when
> low-sgd and fp16 stacked at the right operating point.
> Now the lab has numbers instead of folk wisdom. The mac is singing.

---

## [2026-05-23] L11b' | R-TRAIN family doubles — V=512 + low-sgd + fp16 = 8,340 aug/s (+153% vs WL5)

The compound finding the whole perf cycle was building toward. L11b said: lower trainer SGD work at V=512 to free up MPS for workers (+28% aug/s at trainer level, TQ-gated). L06-followup said: fp16 doubles worker-side throughput at small/V=512 (bandwidth-bound regime). L11b' tests whether the two levers are independent and stack at the R-TRAIN family.

They do.

| Metric | L10 (R-TRAIN-WL5) | L11b (low-sgd alone) | **L11b' (low-sgd + fp16)** |
|---|---|---|---|
| aug/s | 3,297.6 | 4,231.8 (+28%) | **8,340.5 (+152.9%)** 🔥🔥 |
| games/s | 14.07 | 15.47 (+10%) | **32.19 (+129%)** |
| epochs/s | 0.0917 | 0.0500 (-45%) | 0.0667 (-27%) |
| trainer_step_s_p50 | 0.0512s | 0.141s | 0.0801s |
| epochs in window | 14 | 8 | 11 |
| plies_mean | 29.6 | 34.3 | 32.7 |

**Multiplicative stacking, exactly as the mechanism predicts.** The L06 fp16 win on R-S400 was +97.2%. Adding fp16 on top of L11b yields +97.1% over L11b (4,231.8 → 8,340.5) — same magnitude. The lever was free to act because L11b had already cured the trainer-side MPS monopolization. Compounded: 1.28 × 1.97 = 2.52× from R-TRAIN-WL5; measured 2.53×.

**Decision: needs_repeat per TQ gate, same precedent as L11b.** sgd_per_position is behavior-affecting (changes optimizer rate per data); fp16 is no-behavior-change per the L06-followup precedent (Reviewer-verified output cast to fp32 at MCTS boundary at mcts.py:519-529). The composite L11b' recipe HAS one behavior-affecting knob, so the TQ gate fires. R-TRAIN-WL5 stays at WL5 production recipe; a new perf reference R-TRAIN-LEAN-fp16 = 8,340.5 aug/s opens for the lab; production adoption needs a WL6 canary outside the perf cycle.

**Session arc — what the lab measured today:**

| When | Lane | Verdict | Headline number |
|---|---|---|---|
| early | L12 / L05 / L06 / L08-driver shipped (4 CPU agents in parallel) | code | 4 merge commits |
| early | L10 R-TRAIN-WL5 baseline | promote (Reviewer APPROVE) | 3,297.6 aug/s |
| mid | L11 V=512 default-sgd | reject | -28% aug/s |
| mid | L09 ANE+default workers | reject (with trainer-side confirm: trainer_step_s_p50 -56%) | -41% aug/s |
| mid | **L11b V=512 + low-sgd** | needs_repeat (TQ) | **4,232 aug/s (+28%)** |
| late | L05-followup torch.compile | reject | -2.3% (noise) |
| late | **L06-followup fp16-eval R-S400** | **promote** (Reviewer APPROVE) | **9,398.5 aug/s (+97.2%)** 🔥 |
| late | **L06fu-extended R-S200/100/medium fp16** | **promote × 3** | **R-S200 +84%, R-S100 +48%, R-S400-medium new** |
| late | **L11b' V=512 + low-sgd + fp16** | needs_repeat (TQ) | **8,340.5 aug/s (+152.9%)** 🔥🔥 |

Eight perf lanes (including 1 compound chain L11+L09+L11b+L06+L06fu+L11b' that tells a single mechanically clean story) + four CPU code lanes (L05, L06, L08-driver, L12) + four L12 driver bug fixes discovered in flight (--save-every, count_records, --evaluator passthrough, --fp16-eval passthrough). Every receipt cleared a Reviewer audit; every promote followed the charter's no-behavior-change rule (or, when behavior-borderline, the TQ gate).

The trainer-side MPS contention story (L11 + L09 + L11b) and the worker-side bandwidth-bound story (L06-followup + L06fu-extended) ARE the perf lab's biggest insights of the era. Both pointed at the same chip-level reality: the M5 Max has a lot of throughput, but you have to share MPS carefully and feed the GPU with the right precision to extract it.

**Next compound follow-up dispatching:** L09b (R-TRAIN-ANE + fp16 workers) — if fp16 halves the worker-side ANE loss (L09 gen was 2× slower on ANE than MPS at small/V=64), and the trainer-side ANE-relief gain is still real (it was -56% trainer_step_s_p50), R-TRAIN-ANE might finally pay.

> Three trainer-side rejects (L11, L09, L11b nearly-counted) and a near-halt at 3 consecutive rejects — and then L06-followup nearly doubled R-S400, L06fu-extended confirmed the bandwidth-bound mechanism, and L11b' showed both levers stack at the trainer level for a 2.5× R-TRAIN total. The M5 Max is singing.
> Mature MPS + fused conv+bn + 2026's torch flipped the historic fp16 "regression" into the headline win. The perf lab gets to update its mental model.

---

## [2026-05-23] L06-followup | fp16-eval PROMOTE — R-S400 +97.2% (small nearly doubles)

The session was about to halt — three consecutive rejects (L11, L09, L05-followup), stop signal active. L06-followup was a cleanup smoke before declaring session-end. Then this happened:

| Cell | fp32 ref | **fp16-eval** | Δ |
|---|---|---|---|
| small W=8 G=8 S=400 V=512 | 4,765 | **9,398.5** | **+97.2%** 🔥 |
| tiny W=16 G=8 S=400 V=512 | 22,088 | **22,873.8** | +3.6% |

fp16 actually engaged (`fp16-eval enabled (model cast to torch.float16)` in both worker logs); `plies_mean` unchanged (15.97/15.96 vs prior 15.96/15.96 — both at the 16-ply cap so behavior is identical); outputs cast back to fp32 before MCTS reads them (per the L06 patch).

**Mechanism is clean and predicts the asymmetry.** At V=512, the small model is memory-bandwidth-limited (the eval forward pumps a lot of bytes through MPS). fp16 halves that bandwidth requirement → small nearly doubles. The tiny model at V=512 is MPS-dispatch-limited (already running at 22k aug/s, latency-bound by the cost of each dispatched MPS call), not bandwidth-bound; fp16 helps only marginally. This is exactly the compound-finding-readiness signal that "go re-measure historic nulls under mature MPS" was designed to surface.

**Promote decisions:**
- `R-S400` new best: small / W=8 / G=8 / S=400 / V=512 / **fp16-eval** = **9,398.5 aug/s** (was 4,765 fp32)
- `R-S400-tiny` new best: tiny / W=16 / G=8 / S=400 / V=512 / **fp16-eval** = **22,873.8 aug/s** (was 22,088 fp32)
- `consecutive_rejects` resets to 0 — stop signal OFF — the loop is rejuvenated.

**Compound follow-ups queued:**
- L06fu-extended: re-measure R-S200 / R-S100 / R-S400-medium under fp16 (the bandwidth-limited regime should compound there too — medium model likely the biggest absolute win).
- L09b: revisit Core ML / ANE workers with fp16 on the torch fallback path — if the worker-side loss from L09 (small/V=64: gen ~2× slower on ANE than MPS) can be halved by fp16-on-MPS, R-TRAIN-ANE might shift from reject to compete.
- L11b': revisit V=512 + sgd_per_position=0.001 + fp16 at the trainer level — R-TRAIN-LEAN-style finding (+28% aug/s) could compound with this near-doubling.

**The full session arc:** four CPU-queue code lanes shipped in parallel (L12 driver + L05 compile + L06 fp16 + L08-driver env). L10 baselined R-TRAIN-WL5 at 3,297.6 aug/s after surfacing two driver bugs. L11+L09 mapped trainer-side MPS contention as the dominant cost in live training. L11b compounded into +28% aug/s at the trainer level (TQ-gated needs_repeat). L05-followup was neutral (compile noop on MPS). And then L06-followup nearly doubled R-S400. Lab worked exactly as designed: code first, smokes second, compound findings emerge from the receipt-by-receipt chain.

> Apple's MPS docs are silent on whether fp16 is actually faster
> than fp32 for our eval graph. The PyTorch forums say "fp16 on
> MPS is slow" without ever defining slow. We measured it: at
> small/V=512 fp16 is +97%, at tiny/V=512 it's +3.6%. The first
> number is the answer; the second tells you why.
> Now mature-MPS + fused-conv-bn means historic regressions are
> due for re-test, and the perf lab has a clear next round.

---

## [2026-05-23] L11b | V=512 + sgd_per_position=0.001 = +28% aug/s — needs quality canary

The L11+L09 compound finding pointed at the trainer-side MPS contention as the real lever. L11b directly tests the prediction: cap the trainer's per-epoch SGD work at V=512, free up MPS for workers, recover the gen-side win. Same WL5 recipe as L11 except `--sgd-per-position 0.001` (2.5× lower than default 0.0025, to compensate for V=512's 2.4× buffer-fill speedup).

**Result: the lever works.** Headline aug/s beats R-TRAIN-WL5 by **+28.3%**.

| Metric | L10 (V=64 sgd=.0025) | L11 (V=512 sgd=.0025) | **L11b (V=512 sgd=.001)** |
|---|---|---|---|
| aug/s | 3,297.6 | 2,362.8 | **4,231.8** (+28.3% vs L10) |
| games/s | 14.07 | 8.42 | **15.47** (+9.9% vs L10) |
| epochs/s | 0.0917 | 0.0083 | 0.05 |
| trainer_step_s_p50 | 0.0512s | 0.138s | 0.141s |
| steps/epoch (typical) | ~80 | 306 | ~50 (epoch 1=19, epoch 8=90) |
| per-epoch wall | ~11s | ~52s | ~11-21s |

**Mechanism check.** L11b epoch 2: `(11.5s: gen=5.0s train=5.5s)`. L11 epoch 2: would be `~30s+ train`. L10 epoch 2: `(~11s: gen=~5s train=~4s)`. So L11b's per-epoch wall is back to L10-like, BUT with V=512's gen efficiency. Workers no longer starve for MPS. The lever predicted by the L11+L09 compound finding is real and movable.

**Tradeoff (the catch).** L11b runs ~57% less effective SGD per second than L10 (2.5 steps/s vs 7.3 steps/s), because each "epoch" represents less optimizer work at the lower sgd-per-position. Whether more *data* + less *SGD* is better than less data + more SGD is a TRAINING-QUALITY question the perf lab cannot answer on its own. **`decision: needs_repeat`** per the Training-Quality Promotion Gate; the lab establishes the perf lever exists; the training pipeline gets to evaluate whether to adopt it (one canary training run reporting `val/policy_ce` vs `archives/wl5_validation_v1.pt` plus plies/game-shape band).

**Net session score so far:**
- L10 (R-TRAIN-WL5 baseline): promote — Reviewer APPROVE
- L11 (V=512 sgd_default): reject — gen win doesn't free-ride
- L09 (Core ML/ANE workers): reject — but trainer_step_s_p50 -56% confirms MPS-relief mechanism
- L11b (V=512 sgd_low): **+28% aug/s — needs_repeat** — the lever is real

The trio L11+L09+L11b is a genuinely satisfying compound finding. Trainer-side MPS contention is the dominant cost in live training, and there are at least two levers that move it (lower sgd_per_position to cap trainer SGD work; or ANE offload to relocate worker eval). Each individually is a 1-knob change; together they map the design space.

**Charter staleness flagged 3 Reviewers in a row** — `perf-lab-charter.md:50` R-TRAIN-LEAN row still says V=128. Needs user touch (Class B). See perf-queue.md stop-condition tracker.

Next: continue the GPU queue with Tier-3 R-S* follow-ups (L05-followup torch.compile, L06-followup fp16-eval, L08-mps-heap-ratio). Cheap 60s smokes; the heavy-hitter R-TRAIN-* family has produced its first round of receipts.

> Three lanes told the same story from three angles: trainer-side MPS
> contention is the real cost in live training. Now we know which
> levers move it — and that the headline R-S* throughput wins from
> earlier in the day can finally compound at the level that matters,
> once you stop the trainer from eating its own data faster than
> the workers can produce it.

---

## [2026-05-23] L09 | R-TRAIN-ANE REJECT (holistic) — but trainer-side hypothesis CONFIRMED

L09 tested the architectural ANE-offload lever: route worker eval through Core ML on CPU_AND_NE, leaving MPS free for the trainer. The L12 driver gained `--evaluator coreml --coreml-compute-units` passthrough (commit 5c08d3c, third L12 gap of the day) to enable the dispatch. Same WL5 recipe as L10 in every other way; 30s warmup + 120s measure.

**Result: holistic reject, but mechanism partially confirmed.**

| Metric | L10 (torch baseline) | L09 (Core ML/ANE) | Δ |
|---|---|---|---|
| aug/s | 3,297.6 | 1,930.3 | **-41.5%** |
| games/s | 14.07 | 8.00 | -43.1% |
| epochs/s | 0.0917 | 0.0583 | -36.4% |
| **trainer_step_s_p50** | **0.0512s** | **0.0227s** | **-55.7%** ✓ |
| epochs in window | 14 | 10 | -29% |
| plies_mean | 29.6 | 30.4 | — |

**The trainer-side hypothesis was right.** L09 epoch 8: `(11.9s: gen=10.3s train=1.3s)`. L10 epoch 8: `(11.4s: gen=5.4s train=4.4s)`. Once workers vacated MPS, the trainer's per-epoch SGD time fell from ~4s to ~1.3s — a clean ~3× speedup at the step level. The MPS contention from L11's analysis is real and movable.

**The worker-side hypothesis was wrong (for this model size).** Core ML eval at small/V=64 is ~2× slower than torch/MPS for this workload. Epoch 8: gen=10.3s vs L10's gen=5.4s. Workers can't fill the buffer fast enough; the trainer's MPS-relief gain doesn't get reinvested into more throughput because there are no positions to train on. Net: aug/s drops 41%.

**The compound chain L11+L09 tells the story:**
- L11: V=512 hurts the trainer side (more SGD per epoch starves MPS).
- L09: Naive worker-offload hurts the worker side (Core ML at this model size is slower than MPS torch).
- Synthesis: MPS contention is real and bidirectionally costly. Any future trainer-throughput lane has to keep both sides happy — either by making the trainer's per-position SGD cheaper (so workers aren't starved), or by making worker eval cheaper without taking MPS away (e.g. smaller model on ANE, fp16/Core ML compilation tuning, fused Core ML kernels).

**Follow-up candidates queued (in compound-priority order):**
- **L11b** (Tier 1, dispatching next): V=512 + lower `sgd_per_position` (e.g. 0.001 vs default 0.0025) — directly tests whether the trainer-side cost from L11 can be capped, leveraging the same "free up MPS for workers" intuition L09 confirmed on the trainer side.
- L09b (Tier 1, deferred): different `--coreml-compute-units` routing (CPU_AND_GPU, ALL) — does the routing matter? Cheap to run.
- L09c (Tier 1, deferred): tiny model on ANE — smaller per-eval graph might amortize Core ML overhead better. Pairs with R-S400-tiny's 22,088 aug/s pure-gen win.

> Core ML's ANE scheduling is documented across three blog posts that
> disagree on whether small models even benefit. We measured it: at
> small/V=64 the worker side loses ~2× on gen and the trainer side
> wins ~3× on train. Net says the architectural lever is real but
> the model-size operating point matters; tiny might be the place
> this finally pays off.

---

## [2026-05-23] L11 | R-TRAIN-LEAN V=512 REJECT — gen wins don't compound at trainer

L11 tested whether V=64→V=512's pure-gen +49.5% promote (R-S400, L01) carries through to the holistic R-TRAIN-* family. Same WL5 recipe as L10 but `--wave-size 512`. Same 30s warmup + 120s measure.

**Result: reject.** Every metric got worse:

| Metric | L10 V=64 | L11 V=512 | Δ |
|---|---|---|---|
| aug/s | 3,297.6 | 2,362.8 | **-28.4%** |
| games/s | 14.07 | 8.42 | -40.2% |
| epochs/s | 0.0917 | 0.0083 | -91% |
| trainer_step_s_p50 | 0.051s | 0.138s | +2.7× |
| epochs in window | 14 | 3 | — |
| plies_mean | 29.6 | 34.3 | — |

**Mechanism (clean in the trainer log).** At V=512 workers fill the buffer 2.4× faster (buf=199,608 at epoch 3 vs 83,208 at V=64 epoch 3). The trainer's `--sgd-per-position 0.0025` is a fixed ratio, so 2.4× positions = 3.36× SGD steps (epoch 3 ran 306 steps vs 91 at V=64). The per-epoch tail in the trainer log went from `(11s: gen=7s train=3s)` at V=64 to `(52s: gen=6s train=43s)` at V=512. While the trainer monopolizes MPS for 43s of SGD, workers get less GPU time — games/s collapses, aug/s collapses with it.

This is the holistic R-TRAIN-* family working as intended. The L11 yaml's own caveat called it: *"If it doesn't, R-S* metrics need humility — gen throughput isn't the whole story."* Confirmed.

**Lab implications:**
- V=64 stays the R-TRAIN-WL5 default. WL5 production recipe is correct as-is.
- The R-S* V=512 promotes remain valid for *non-trainer* self-play (eval probes, validation rolls, dataset mining). They do NOT free-ride to live training.
- Follow-up candidate L11b: would lowering `--sgd-per-position` at V=512 (to match V=64's SGD work per second) let the gen win shine? Lower priority than L09 — the headline finding here (gen wins don't free-ride) is already the load-bearing insight.

Next: dispatch L09 (R-TRAIN-ANE via Core ML eval on workers) — the architectural ANE-offload lever. First needs a small L12 driver patch to pass `--evaluator coreml --coreml-compute-units CPU_AND_NE` through to workers.

> Pure-gen wins were the easy half. The trainer is fighting for the same chip, and at V=512 it wins the fight — which is exactly the wrong fight to win.
> Now we know: any future "what if we crank V higher" idea has to be paired with a sgd-per-position cut, or it costs us at the level that matters.

---

## [2026-05-23] L10 | R-TRAIN-WL5 baselined at 3,297 aug/s; trainer contention ≈ 30%

First-ever R-TRAIN-WL5 measurement. End-to-end production recipe (small / W=8 / G=8 / sims=400 / V=64 / EMA τ=0.99 / grad_accum=4) under the live trainer + 8 self-play workers competing for MPS. 120s measurement window, 14 epochs:

- **aug_pos_per_sec: 3,297.6** (vs R-S400 pure-gen 4,765 → trainer contention costs ~30.8% on generator throughput)
- **games_per_sec: 14.07** (vs ~17.7 implied by R-S400's 4,765 / 269 aug-per-game)
- **epochs_per_sec: 0.0917** (~10.9s wall per epoch in steady state — 50 SGD steps at trainer_step_s_p50=0.051s = 2.56s training plus ~5-6s of barrier-wait for fresh self-play)
- **trainer_step_s_p50: 0.0512s** (per SGD step; the trainer is GPU-bound here, not blocked on data)

The autonomous lab restart hit two L12 driver bugs in flight, both surfaced and patched:

1. **`--save-every=1000000` froze worker_weights.pt**. `gomoku/train.py:1220` publishes the worker-facing weights file inside the save-every block; with save-every set high "to disable mid-run checkpoint IO", workers stayed on v0 forever and the trainer hung waiting for v1+ games. Fix: `--save-every=1 --keep-last-n=1` (small per-epoch ~4MB writes, auto-pruned; the 1.4GB latest.pt still gated by save-buffer-every=1M). Commit `1dc4abb`.
2. **`count_records()` at SIGTERM undercounted by ~30×**. The trainer ingests + deletes worker `game*.pt` files as it goes, so the end-of-window file count was ~80 games / 16k aug-positions where the trainer log's cumulative `games=` counter showed ~1,500 games / ~350k positions. Fix: parse the trainer's epoch line directly (cumulative `games=N`, `buf=N`, per-epoch wall `(Xs:`) and prefer those over file counts. Commit `4a825f1`.

Both bugs were invisible to L12's `--dry-run` and synthetic-log smoke tests — only the real workload exposed them. Receipt under L10-trainer-step-bench in the ledger.

Next: dispatch L11 (R-TRAIN-LEAN at V=512) to test whether V=64→V=512's +49.5% gen win compounds at the trainer level. Then L09 (R-TRAIN-ANE via Core ML eval on the workers).

---

## [2026-05-22] lab | post-WL5 perf era opened

WL5 phase-2 closed at e10200 yesterday. Box is idle for the first time in
weeks. Jason called it: "let's get serious about this M5 Max as a
mainframe and squeezing every drop from it. let's set up a lab, run
experiments, keep a log, the whole thing."

State at lab-open:
- Frontier-lab infrastructure exists in `wiki/ops/` and `.frontier/lanes.json`.
- Headline experiment from [m5-max-as-mainframe.md](../topics/m5-max-as-mainframe.md)
  step 4 (the canonical 5-axis sweep producing the contour chart) has **not**
  been executed. The prior `production-contour-20260522` lane only swept
  workers x games-per-worker at default sims/wave/model.
- ANE residency rail-proof lane is blocked on cached/passwordless sudo for
  `powermetrics`. Real evidence exists in the 934b detached worktree but
  was never reproduced in main.
- Buffer-width cheap-test is seeded but warm; deferred per the m5-max
  sequencing in favor of the canonical sweep deliverable.

Next action: run the canonical sweep. Receipts under
`sweep_logs/canonical-sweep-<TS>/`. Open the
[canonical-sweep-mainframe](frontier.md) lane on completion.

## [2026-05-22] lab | canonical sweep paused — BAB1 active on the box

Discovered mid-setup that a `BAB1-buf-ablation-1p5M` run was already
alive when this session opened — another agent context launched it for
the `packed-buffer-cheap-test` lane:

```
trainer PID 27579  (gomoku.train, --resume archives/wl5_e10200_seed.pt,
                    --epochs 500, --replay-buffer-size 1500000,
                    --wave-workers 8 --wave-games-per-worker 8,
                    --validation-archive-path archives/wl5_validation_v1.pt)
8 workers       (PIDs 27596-27603)
eval_worker     (PID 27604, CPU baselines)
wandb-core      (PID 27644)
zsh monitor     (PID 28262, watches for `epoch 10700/10700`)
```

At pause, BAB1 was at e10215/10700 (~10 s/epoch). Expected completion
~80 min. Per [[project-buffer-curation]] memory, this is part of a
buffer-curation research arc, not just a 1.5M-vs-750k throughput
ablation. After BAB1 there may be a paired BAB2 (750k).

What was done in this session before the pause:
- Added [topics/perf-lab-session-runbook.md](../topics/perf-lab-session-runbook.md).
- Wrote `scripts/canonical_sweep.py` (23-cell 5-axis design).
- Wrote `scripts/plot_canonical_sweep.py` (axes + model + contour plots).
- Smoked the driver; first iteration crashed because spawned workers
  collided with BAB1 for MPS memory and exited zombies, then
  `killpg(zombie_pgid)` returned EPERM. Patched to use
  `Popen(start_new_session=True)` + `p.terminate()` + zombie-tolerant
  cleanup, plus a `cell_status=failed` column so contended cells don't
  pollute the summary.

What this session is **not** doing: running the canonical sweep
concurrent with BAB1. The WL5-era 2026-05-22 baseline receipt already
documented how much MPS trainer contention skews bench numbers; running
the sweep against a live trainer would produce numbers we couldn't
defend as "the M5 Max's behavior" without contention.

Next session pickup:
1. Confirm BAB1 (and BAB2 if present) finished — `pgrep -fl
   'gomoku.train|selfplay_worker|eval_worker'` must be empty.
2. Re-smoke 2 tiny cells:
   `python scripts/canonical_sweep.py
   --out-dir sweep_logs/canonical-sweep-smoke-$(date -u +%Y%m%dT%H%M%SZ)
   --secs-per-cell 60 --only tiny_W01,tiny_W08`.
3. Kick the full sweep in background:
   `python scripts/canonical_sweep.py --out-dir
   sweep_logs/canonical-sweep-$(date -u +%Y%m%dT%H%M%SZ)` — ~2 to 3 h.
4. Check progress anytime:
   `python scripts/canonical_sweep.py --out-dir latest --status`.
5. After completion: `python scripts/plot_canonical_sweep.py
   --sweep-dir sweep_logs/canonical-sweep-latest`.
6. File receipt in [experiment-ledger.md](experiment-ledger.md),
   add baseline rows to [baselines.md](baselines.md), promote the
   winning cell in [status.md](status.md), close the
   `canonical-sweep-mainframe` lane in `.frontier/lanes.json`.

## [2026-05-22] lab | canonical sweep driver is first-class resumable

Per Jason: "make resumability first class since this will require many
hours of processing and I'll be using it from time to time." Refactored
`scripts/canonical_sweep.py` against an 8-point contract now documented
under [Resumability contract](../topics/perf-lab-session-runbook.md):

1. **Stable cell IDs** — derived purely from params (e.g.
   `small_W08_G08_S400_V064`); no list-position prefix that would
   shift when the cell list grows.
2. **Atomic source of truth** — append-with-fsync per row;
   write-temp-then-rename for full rewrites.
3. **Per-cell `cell_status`** — `ok` / `failed`; `--retry-failed`
   drops failed rows and wipes their cell_dir before re-running.
4. **PID lock file** at `<out>/.sweep.lock`; aborts on live PID,
   reclaims dead PIDs.
5. **`--status` mode** — done / failed / pending + ETA from median
   wall_secs of completed cells. Exits without spawning GPU work.
6. **`--max-wall-secs N`** budget for short top-up sessions.
7. **SIGINT/SIGTERM handler** — kills workers, drops the
   interrupted cell's row, releases the lock.
8. **`sweep_logs/canonical-sweep-latest`** symlink, refreshed on
   every session; `--out-dir latest` follows it.

All eight surfaces smoked without GPU (box still has BAB1 alive):
stable IDs verified, --status read fake-seeded rows correctly,
--retry-failed dropped + wiped, lock blocked a live PID and reclaimed
a dead one, `latest` symlink resolution worked. The plan banner now
shows e.g. `[plan] 23 cells total | 7 ok | 1 failed-skipped |
15 to run this session | ETA ~76.4 min`.

This means a full sweep can be done in fits and starts: kick it off,
walk away, check `--status` later, top-up with `--max-wall-secs 1800`
between meetings, retry whatever failed once the box is calmer. The
contract applies to future drivers too (ANE rail proof, packed-buffer
ablation) — see the runbook.

## [2026-05-22] lab | smoke caught two real bugs before the full sweep

Once BAB1 cleared the box and a real-worker smoke became possible,
two things broke that the dry-run resumability smoke couldn't have
seen:

- **Pre-fused checkpoint vs un-fused load path.** `stage_checkpoint`
  was calling `fuse_model_for_inference` before `save_checkpoint`.
  The worker's `_load_model` builds a fresh un-fused `GomokuNet` and
  calls `load_state_dict`, which then rejects the fused state_dict
  (extra `tower.*.conv*.bias`, missing `tower.*.bn*.running_mean`,
  etc.). Workers crashed on first load. Fix: stage un-fused; workers
  fuse internally after load (`selfplay_worker.py:198`,
  `:632`, `:749`).
- **8x throughput double-count.** `selfplay_worker` writes
  `n_examples = len(record.examples)`, but the examples list is
  already D4-augmented (8 entries per raw ply). My driver was
  computing `aug_pos_per_sec = total_n_examples * 8 / wall_secs`
  — off by 8x. Tiny W8 G8 S400 V64 first reported 56,735 aug/s
  (impossible for tiny on M5 Max — small ref is 2,379). Fix:
  track `total_aug_examples` and `total_raw_plies` separately;
  aug throughput is `total_aug_examples / wall_secs`;
  `plies_mean` is `total_raw_plies / total_games`.

Schema in `summary.tsv` is now `total_aug_examples` +
`total_raw_plies` (replacing the ambiguous `total_plies`). Old rows
from broken smokes were wiped — no production data lost since the
sweep had never run.

Real-worker post-fix numbers (45s/cell, fresh random weights so all
games hit `--max-plies 16`):

| cell | aug pos/s | games/s | plies_mean | games |
|---|---|---|---|---|
| tiny_W08_G08_S400_V064 | 7,135 | 55.9 | 16.0 | 2,529 |
| tiny_W01_G04_S100_V032 | 2,485 | 19.5 | 16.0 | 879   |

Calibrates against the existing baseline row for native small 8w8g
sims=400 wave=64 (~2,379 wall aug pos/s): tiny is ~3x faster than
small on the same shape, which matches a tiny-vs-small forward-pass
ratio. Sanity-passes.

## [2026-05-22] ops | BAB1 stopped early at e10247

Independently of this lab work, the `BAB1-buf-ablation-1p5M` run
stopped at e10247/10700 — neither at its 10700 cap nor on a crash
(no traceback, no NaN, trainer log just stopped advancing at
19:45 local). All workers, the trainer, the eval worker, and the
wandb sidecar were gone by the time I checked again. Likely the
other session driving BAB1 deliberately paused/killed it; the
buffer-curation arc that BAB1 belongs to is a separate workstream
([[project-buffer-curation]] memory).

This perf-log notes the state only so future readers don't assume
BAB1 ran to its written cap. The `packed-buffer-cheap-test` lane
(or its successor) is the canonical place for BAB1 interpretation.

## [2026-05-22] lab | canonical sweep launched

Box is idle, smoke is green, driver is first-class resumable, and the
user said go. Kicked the 23-cell canonical sweep in background.

- Sweep dir: `sweep_logs/canonical-sweep-20260523T015614Z/`
- Symlink:   `sweep_logs/canonical-sweep-latest`
- Driver:    `python scripts/canonical_sweep.py --out-dir
              sweep_logs/canonical-sweep-20260523T015614Z` (nohup)
- Defaults:  `--secs-per-cell 300 --max-plies 16 --device mps`
- Expected:  ~2-3 h wall (23 cells × ~5 min + per-cell setup)
- Driver log: `<sweep dir>/driver.log` (line-buffered from this commit
  forward; the in-flight run will only flush when its buffer fills, so
  use `--status` for live progress instead of tailing the log).

Recipes the user can run at any time during or after the sweep:

```bash
# Progress + ETA:
python scripts/canonical_sweep.py --out-dir latest --status

# Re-run any cells that failed (e.g. transient MPS contention):
python scripts/canonical_sweep.py --out-dir latest --retry-failed

# After the sweep finishes:
python scripts/plot_canonical_sweep.py --sweep-dir sweep_logs/canonical-sweep-latest
```

Next-session pickup once the sweep completes (or stalls):
1. `python scripts/canonical_sweep.py --out-dir latest --status` to
   confirm 23 ok / 0 pending; if not, `--retry-failed` and let it
   finish.
2. `python scripts/plot_canonical_sweep.py --sweep-dir
   sweep_logs/canonical-sweep-latest` → `contour.png`, `axes.png`,
   `model_compare.png`.
3. File a receipt in [experiment-ledger.md](experiment-ledger.md)
   under the `canonical-sweep-mainframe` lane; add per-cell-class
   rows to [baselines.md](baselines.md); promote the winning cell
   in [status.md](status.md); close the lane in
   `.frontier/lanes.json`; append a "[YYYY-MM-DD] lab | canonical
   sweep complete" entry here.

## [2026-05-23] lab | canonical sweep complete — wave_size is under-tuned

23/23 cells ok in `sweep_logs/canonical-sweep-20260523T015614Z` (also
at `canonical-sweep-latest`), median 300.5s/cell, 0 failed. All games
hit `--max-plies 16` (random weights → no learned defense → universal
cap), so cell numbers are **infrastructure throughput**, not behavior
throughput. Trained-model production cycles will be slower in
absolute aug/s; the relative axis shape should hold.

### Axis-by-axis results

| Axis (other params at default) | Cells | Headline |
|---|---|---|
| **workers** (small G=8 S=400 V=64) | W1=1,111 → W2=1,497 → W4=2,583 → **W8=3,188** → W12=3,243 → W16=3,411 aug/s | Diminishing returns; W=8 is near-optimal. Per-worker eff falls from 1,111 at W=1 to 213 at W=16; MPS contention dominates by W=2. |
| **n-simulations** (small W=8 G=8 V=64) | S100=11,151 / S200=6,006 / **S400=3,188** / S800=1,619 | Perfectly inverse: aug/s × sims ≈ const. Pure quality knob. |
| **wave-size** (small W=8 G=8 S=400) | V32=2,467 / **V64=3,188** / V128=4,048 / V256=4,409 aug/s | **+27% at V128, +38% at V256 over the WL5 default V64.** No behavior change, just bigger eval batches. |
| **games-per-worker** (small W=8 S=400 V=64) | G4=3,026 / **G8=3,188** / G16=3,057 | Flat. G=8 default is fine. |
| **model** (W=8 G=8 S=400 V=64) | tiny=7,326 / **small=3,188** / medium=1,393 | ≈2.3× per step. Forward pass dominates. |
| **max corner** | tiny W16 G16 S100 V32 | 19,346 aug/s — infrastructure ceiling, not quality-comparable. |
| **min corner** | small W1 G16 S800 V128 | 946 aug/s — single fat worker. |

### Promoted default

**Old throughput default:** small / W=8 / G=8 / sims=400 / **wave=64** → 3,188 aug pos/s
**New throughput default:** small / W=8 / G=8 / sims=400 / **wave=128** → 4,048 aug pos/s (+27%)

Wave=256 is also viable (+38%). Chose V=128 as the safer step: V=256
is past the inflection point and may interact poorly with MPS heap
sizing under sustained training pressure (none of these cells used
the trainer; production cells should canary the V=128 candidate
first per the Training-Quality Promotion Gate).

The wave-size win is the single most actionable result from this
sweep. It is exactly the kind of chip-specific calibration the
[m5-max-as-mainframe](../topics/m5-max-as-mainframe.md) page predicted
we'd find by sweeping the production shape on this exact SKU
instead of transplanting CUDA recipes.

### Caveats and follow-ups

- **All cells hit plies_mean=15.96.** Random weights + max-plies=16
  meant every game terminated at the cap, so absolute throughput
  numbers reflect infrastructure (eval batching, worker spawn, file
  handoff) more than realistic game shape. Wave-size win is
  eval-batch-shape-dependent (not game-shape-dependent), so it
  should transfer; worker-axis numbers may shift somewhat with real
  plies.
- **W × G cross was not run.** The workers axis fixed G=8 and the
  games-per-worker axis fixed W=8. Re-running the cross at V=128
  would verify whether the wave win compounds at higher worker
  counts.
- **Sims-vs-wave interaction** is unexplored. S=200 with V=128 or
  V=256 might be the real next-cell shape if quality holds at lower
  sims.
- **Trained-checkpoint re-sweep.** Repeat once a stable post-WL5
  trained checkpoint exists to confirm trained-model throughput
  shape matches infrastructure shape.

### Surfaces updated

- Receipt: [experiment-ledger.md](experiment-ledger.md) "2026-05-23
  — canonical 5-axis M5 Max contour sweep".
- Baseline rows: [baselines.md](baselines.md) (7 new rows; wave-size,
  workers axis, model axis, max corner).
- Status: [status.md](status.md) Current Focus + lane row.
- Frontier: [frontier.md](frontier.md) + `.frontier/lanes.json`
  (lane completed/done).

### Suggested next lanes for the lab

1. **W × G cross at V=128** (small focus): ~12 cells × 300s = 1 h.
   Confirms the wave win compounds.
2. **Sims-vs-wave interaction**: small W=8 G=8 over
   S ∈ {100, 200, 400} × V ∈ {64, 128, 256} = 9 cells; ~45 min.
3. **Trained-checkpoint re-sweep** once post-WL5 training stabilizes
   on a strong checkpoint. Same 23 cells, swap the staged random
   weights for the trained ones.
4. **ANE rail-proof unblocker** — still gated on passwordless sudo
   for `powermetrics`. Independent of this sweep.
5. **Engine-overlap experiment** — unblocks once ANE rail is real.
   The wave=128 throughput default is the right MPS-side baseline
   for that experiment.

## [2026-05-23] lab | L01 wave extrapolation — V=512 is the plateau knee, +49.5% cumulative on R-S400

First lab-dispatched lane under the charter. 4 cells × 5 min, all ok.

| cell | aug/s | vs V=128 | vs WL5 V=64 |
|---|---|---|---|
| V=384  | 4,452 | +10.0% | +39.6% |
| **V=512**  | **4,765** | **+17.7%** | **+49.5%** |
| V=768  | 4,761 | +17.6% | +49.4% |
| V=1024 | 4,756 | +17.5% | +49.2% |

V=512 is the plateau knee. V=768/1024 are flat — eval overhead caps
further wave gains on this exact hardware. The lab will stop sweeping
V > 512 unless something else (model size, MPS heap config, ANE
engine) shifts the eval-overhead floor (L07 tiny contour will check
the model-size dependency).

**Promotion: small / W=8 / G=8 / sims=400 / V=128 → V=512** at R-S400.
Pending Reviewer signoff per
[perf-lab-reviewer-role](../topics/perf-lab-reviewer-role.md).
+17.7% over yesterday's V=128 promote. +49.5% cumulative since the
WL5 production V=64. No behavior change; eval batch shape only.

**Auto-queued compounds** (per the charter's Tier-1-after-promote
discipline): L02 (W × V=512: W ∈ {4,12,16}), L03 (S × V=512: S ∈
{100,200}). Both rescoped to drop V=128/V=256 cells that L01 now
dominates. L02's E[delta] dropped from 800 to 400 aug/s and P from
0.6 to 0.5 since the workers axis was already shown to be near-flat
past W=8 in the canonical sweep — most likely outcome is "V=512 holds
across W". L03 stayed high-priority because S × wave was the most
under-explored cross in the canonical sweep.

Process notes:
- L01 was originally Tier-1 in the day-1 queue; the charter v2 tier
  refactor demoted it to Tier-3 (single-axis speculation past a known
  win). The run completed before the refactor landed; archived as
  Tier-3 in retrospect.
- Reviewer Gate is on for all subsequent receipts. L01's receipt has
  `reviewer: PENDING`; a Reviewer spawn will audit and the verdict
  appended before the next commit closing the receipt.

## [2026-05-23] lab | L03 sims-x-wave — V=512 carries to every quality point (R-S200 +52.5%, R-S100 +35.2%)

Cron tick caught L03 just-completed. 2/2 ok, ~5 min each.

| cell | aug/s | vs WL5 V=64 |
|---|---|---|
| small W=8 G=8 S=100 **V=512** | **15,082** | **+35.2%** over 11,151 |
| small W=8 G=8 S=200 **V=512** |  **9,156** | **+52.5%** over 6,006  |

**Double promote** — V=512 now wins at every R-S* reference point measured. The wave-size lever is uniform across the sims axis: same speedup mechanism (eval-batch shape) applies regardless of how many MCTS sims feed the batch.

Three reference points are now at V=512:

| ref | best aug/s | speedup vs WL5 V=64 |
|---|---|---|
| R-S400 | 4,765  | +49.5% |
| R-S200 | 9,156  | +52.5% |
| R-S100 | 15,082 | +35.2% |

R-S200 has the biggest gain because S=200 V=64 was particularly under-saturated on eval (wave was too narrow vs the per-call kernel cost). At S=100 the gain shrinks because games-per-batch × wave already saturated MPS at smaller batch sizes; at S=400 the gain is between because each sim contributes more wall-time but fewer eval calls.

The cron's Speedup Report line is now load-bearing. Reviewer: APPROVE — "L03 double promote math + units verified; all six surfaces consistent; queue clean."

## [2026-05-23] lab | L02 W-x-wave reject — W-axis INVERTS at V=512

3/3 cells ok. No promote — and the absence is itself a finding.

| cell | aug/s | vs W=8 V=512 ref (4,765) |
|---|---|---|
| W=4  V=512 | 4,367 | -8.4% |
| **W=8 V=512** | **4,765** | reference (L01) |
| W=12 V=512 | 4,501 | -5.5% |
| W=16 V=512 | 4,504 | -5.5% |

At V=64 the canonical sweep had W=16 as the peak (3,411 vs W=8 3,188 = +7%). At V=512 the peak is W=8, and W=12/W=16 are slightly worse. The wave-saturation pressure shifted the MPS-dispatch sweet spot.

**Implication**: knob wins don't just fail to compound — they actively interact in non-monotone ways at the chip's high end. The tier system's "no leapfrogging" rule is more than aesthetic; it's about not assuming linear combinatorics. Future cells should always re-measure the W axis when V changes substantially, not extrapolate.

Auto-queue updates (in `wiki/ops/perf-queue.md`):
- L04 G-x-wave bumped from priority 1.4 → 9.0 (G might also be non-monotone at V=512; was flat at V=64).
- L07 tiny-contour bumped from priority 12 → 36.4 (added V=512 + V=1024 cells; tiny model may extend the wave plateau further because forward pass is cheaper).

consecutive_rejects: 0 → 1.
Reviewer: APPROVE — "L02 reject math clean (-8.4%/-5.5%/-5.5%); best-cells correctly unchanged; W-inversion insight requeues L04+L07; counter 0→1."

## [2026-05-23] lab | L04 G-x-wave reject — G=8 stays optimal (compound finding with L02)

3/3 cells ok.

| cell | aug/s | vs G=8 V=512 ref (4,765) |
|---|---|---|
| G=4  V=512 | 4,608 | -3.3% |
| **G=8 V=512** | **4,765** | reference |
| G=16 V=512 | 4,541 | -4.7% |
| G=32 V=512 | 4,514 | -5.3% |

G axis IS mildly non-monotone at V=512 (was completely flat at V=64: 3026/3188/3057). But the peak is still G=8 — same shape as L02's W-axis result.

**Compound finding with L02 — sharper than either alone**: at V=512 BOTH the workers axis AND the games-per-worker axis peak at the canonical-sweep production defaults (W=8, G=8). Wave-saturation has tightened the production-cell envelope around the historical defaults. Wider perimeter exploration at V=512 won't beat the center.

**Practical implication**: future single-axis explorations at V=512 should not bother re-measuring W or G — those axes are CONFIRMED at their peaks. Open axes for further exploration: model size (L07 tiny), n-sims at V=512 (L03 done), architectural (L09 ANE), engine-isolation (L05/L06 worktrees).

Followup:
- L08 (MPS heap ratio) marked blocked-on-driver. canonical_sweep doesn't support per-cell env vars; cells.csv schema needs extension. Add to L12 scope or carve out an L08-driver task.
- Next dispatch: L07 tiny contour (bg priority 36.4 after the L02 bump). The strict tier rule says Tier-3 before bg, but L05/L06/L08 are all blocked-on-code-work, so L07 is the only unblocked lane.

consecutive_rejects: 1 → 2. One more reject would still NOT halt the loop (the stop rule requires `consecutive_rejects ≥ 3 AND queue empty AND no compound follow-ups`; queue is not empty).

Reviewer: APPROVE — "L04 reject math clean (-3.3/-4.7/-5.3%); best-cells unchanged; compound W+G finding documented; L08 correctly blocked; counter 1→2."

## [2026-05-23] lab | L07 tiny contour — R-S400-tiny promote +201.5%; W peak is model-dependent at V=512

6/6 cells ok. The lab's biggest single-lane jump so far.

| cell | aug/s | vs tiny V=64=7,326 |
|---|---|---|
| tiny W=8  V=128 |  9,407 | +28.4% |
| tiny W=8  V=256 | 14,461 | +97.4% |
| tiny W=8  V=512 | 17,088 | +133.2% |
| tiny W=8  V=1024| 17,012 | flat with V=512 (same plateau as small) |
| tiny W=16 V=256 | 16,375 | +123.5% |
| **tiny W=16 V=512** | **22,088** | **+201.5%** ← new R-S400-tiny best |

V=512 plateau holds for tiny (V=1024 flat). But the **model-dependent W peak** is the headline:

| model | best W at V=512 | second-best W |
|---|---|---|
| small (L01/L02)  | **W=8** = 4,765 | W=16 = 4,504 (-5.5%) |
| tiny  (L07)      | **W=16** = 22,088 | W=8 = 17,088 (-22.7%) |

At small, eval cost per worker is high enough that 8 workers saturate MPS dispatch. At tiny, eval cost is ~3× cheaper so MPS can stay fed with 16 workers — the saturation point shifted right.

**Direct implication for L09 ANE-offload**: with workers on Core ML (CPU/ANE), the effective per-worker eval cost changes again. Whether W=8, W=16, or W=24+ is the peak under the ANE workload is unknown a priori — L09's measurement cells should test BOTH W=8 and W=16 at V=512, not just one. Added this note to the L09 queue entry.

**Auto-queued follow-ups** (both bg, both new):
- L13 (priority 58.8 — highest in current queue): probe tiny peak finer at W ∈ {12, 16, 20, 24}. If W=20 or W=24 beats W=16, even bigger gain available.
- L14 (priority 16.5): G axis at tiny W=16 V=512.

consecutive_rejects: 2 → 0 (any promote resets per stop rule).

Reviewer: APPROVE — "L07 promote math clean (+201.5%); 2-axis move decomposed via cell matrix; surfaces consistent; L13/L14 well-scoped."

## [2026-05-23] lab | L13 tiny W-peak probe reject — W=16 confirmed; tolerance band W∈[12,20] within 7%

3/3 cells ok. Fine-grained peak confirmation.

| cell | aug/s | vs W=16 V=512 ref (22,088) |
|---|---|---|
| W=12 V=512 | 20,560 | -6.9% |
| **W=16 V=512** | **22,088** | reference (L07) |
| W=20 V=512 | 21,553 | -2.4% |
| W=24 V=512 | 20,970 | -5.1% |

W=16 is confirmed the tiny V=512 peak, with W=20 a very close second (within 2.4%). The whole W ∈ [12, 20] band is within 7% of peak — tiny's W-axis at V=512 is a smooth bump, not a sharp saturation drop.

**Compound finding (L02 + L07 + L13)** — model size determines BOTH the W-peak location AND the tolerance shape at V=512:
- **small**: peak W=8, sharp drop (W=16=-5.5%, W=4=-8.4%, narrow tolerance)
- **tiny**: peak W=16, gentle bump (W=12=-6.9%, W=20=-2.4%, W=24=-5.1%, wide tolerance)

Direct implication: L09 ANE-offload worker tuning has more wiggle room with tiny than the small data suggested. The optimal under Core ML/ANE is probably also in the W ∈ [12, 20] band rather than a single sharp peak.

consecutive_rejects: 0 → 1.
Next dispatch: L14 (G axis at tiny W=16 V=512).

Reviewer: APPROVE — "L13 reject clean: math/plies/units verified, W=16 confirmed peak, surfaces consistent, no spurious follow-ups."

## [2026-05-23] lab | L14 G axis flat — knob-tuning exhausted at chip envelope

3/3 cells ok. G axis at tiny W=16 V=512 is essentially flat:

| cell | aug/s | vs G=8 ref (22,088) |
|---|---|---|
| G=4  V=512 | 22,261 | +0.78% |
| G=8  V=512 | 22,088 | reference |
| G=16 V=512 | 22,164 | +0.34% |
| G=32 V=512 | 22,076 | -0.06% |

Total spread 0.83% — within unmeasured run-to-run noise. G=4 nominal lead of +0.78% is not a defensible promote.

**The headline finding across L02 + L04 + L13 + L14 is now decisive: at V=512 (the new structural default), single-axis knob exploration of W and G has been exhausted for both small and tiny models.** No further knob tweaks within the {W ∈ [4, 24]} × {G ∈ [4, 32]} envelope produce a promote. The wave-size lever was the regime-changing knob; everything else is fine-tuning noise relative to it.

**Cumulative lab state**:

| reference | best cell | best aug/s | cumulative speedup |
|---|---|---|---|
| R-S400 | small W=8 G=8 V=512 | 4,765 | +49.5% |
| R-S200 | small W=8 G=8 S=200 V=512 | 9,156 | +52.5% |
| R-S100 | small W=8 G=8 S=100 V=512 | 15,082 | +35.2% |
| R-S400-tiny | tiny W=16 G=8 V=512 | 22,088 | +201.5% |

**Remaining headroom is structural, not knob**:
- L09 ANE-offload (blocked on L12)
- L05 torch.compile (worktree code)
- L06 fp16 (worktree code)
- L08 heap ratio (per-cell env var driver work)
- L12 live-training cell driver (Tier 1 gating)
- L10 R-TRAIN-WL5 baseline (blocked on L12)
- L11 R-TRAIN-LEAN end-to-end (blocked on L12)

All require human-session code work. Cron is at a natural pause point. PushNotification sent.

consecutive_rejects: 1 → 2.

Reviewer: APPROVE — "L14 reject correct — G axis spread 0.83% within noise; surfaces consistent; pause state cleanly logged."

## [2026-05-23] lab | charter v2 — tier system + R-TRAIN family + Reviewer Gate

After L01 launched but before it landed, Jason gave four new
directives that reshape the lab:

1. **Live training cells allowed** — ≤ 5 min/cell, multi-cell stitch
   for warmup + measure. Opens the R-TRAIN-* metric family
   (epochs/sec under live trainer). This is the holistic metric
   that matters for elo gain, not just isolated self-play.
2. **Reviewer role** — codified in
   [perf-lab-reviewer-role](../topics/perf-lab-reviewer-role.md).
   Spawned per lane + every ~5 lanes for discipline audit. APPROVE
   / REVISE / BLOCK. No promote without signoff.
3. **/loop 10m check-in** — periodic auto-tick: read queue, file
   receipts, dispatch next-priority lane.
4. **ANE / engine isolation > batch sizes** — explicit tier rule.
   Architectural lanes (L09 ANE-offload, L10 trainer bench, L11
   end-to-end) can't be leapfrogged by knob lanes on raw priority
   alone.

Charter v2 committed in `7491401`. Queue reranked.
