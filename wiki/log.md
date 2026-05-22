# Gomoku Wiki Log

Chronological record of wiki maintenance. Keep entries append-only and use a
consistent heading so future sessions can scan recent changes with simple tools.

## [2026-05-22] ops | frontier-lab ML perf control room seeded

- Added project-local pi frontier-lab setup under `.pi/` plus machine-readable
  `.frontier/config.json` and `.frontier/lanes.json`.
- Seeded `wiki/ops/` control-room pages for status, frontier, baselines,
  experiment receipts, test ledger, and open notes.
- Indexed the frontier-lab ops pages so future performance fanout starts from
  maintained baseline/receipt surfaces instead of raw chat state.

## [2026-05-22] ops | perf frontier lanes rewritten from current worktree evidence

- Rewrote `.frontier/lanes.json` around the current perf frontier: baseline receipts, M5 Max production contour, Core ML / ANE rail proof, quality promotion gates, control-room curation, outer-loop Python profiling, blocked production engine-overlap, and replay-buffer width cheap test.
- Updated [ops/status.md](ops/status.md) and [ops/frontier.md](ops/frontier.md) to reflect the current worktree inventory: perf10 artifacts in `/Users/jason/code/gomoku-perf-extension` and uncommitted Core ML / ANE residency work in `/Users/jason/.codex/worktrees/934b/gomoku`.
- Filed the perf10 production-shaped sweep into [ops/baselines.md](ops/baselines.md), [ops/experiment-ledger.md](ops/experiment-ledger.md), and [ops/test-ledger.md](ops/test-ledger.md) as seed evidence for the production-contour lane; the ledger marks it `needs_repeat` because exact launcher commands were not captured in ops.
- Tightened the promotion gate language: behavior-touching perf changes need fixed baseline/archive quality checks, plies/game-shape checks, noise caveats, checkpoint/run IDs for strength claims, and an explicit decision.

## [2026-05-21] run | WL5 phase-1 closed at e5051 (un-fused-workers era)

- Closed the pre-fusion era of WL5 as a discrete chapter. WL5 phase 1
  ran from launch (`o6cbjfnr`, 19:05:28) through e5051, then continued
  as phase 2 from e5052 once 8 workers were hot-restarted with
  Conv+BN-fused inference. Same wandb run, same trainer, same buffer,
  same design — but gen-side throughput at e5052+ is **1.53× higher**,
  which makes per-epoch absolute numbers (games/epoch, steps/epoch) not
  directly comparable across the boundary. That regime shift justifies
  framing phase 1 as its own closed-out chapter.
- Phase 1 = 1051 epochs, ~2.5h wall, 123,453 games, ~414 epochs/hr,
  zero NaN/crashes/worker deaths. Validated the archive-start lever
  doesn't destabilize the pipeline; validated the diagnostic streams
  populate cleanly; phase shape matched the [[feedback-absorption-phase]]
  prediction (200-1000 epochs of absorption shock, plies stayed healthy).
- Phase 1 elo peak of 1784 at e4035 was residual WL4 strength (34
  epochs after resume); the rest of phase 1 was absorption with elo
  oscillating 1159-1738 (mean 1498), pl mean 0.673 (up from WL4
  plateau-end's 0.604), plies mean 38.6 (vs WL4's 40.0).
- Phase 2 monitoring continues: stop when run hits e9000, or on
  collapse / NaN / new ATH > 1841 / canonical-opening regression.
  Full close-out entry in [TRAINING_WIKI.md](../TRAINING_WIKI.md).

## [2026-05-22] perf | aggressive Apple Silicon engine scout

- Added `gomoku/coreml_evaluator.py` with lazy Core ML loading/export helpers
  and `scripts/aggressive_engine_scout.py`, a bounded JSON-emitting harness
  for PyTorch MPS vs Core ML CPU_ONLY/CPU_AND_NE latency plus MPS trainer
  overlap pressure.
- Ran the scout once on the small fused model. Receipt:
  `sweep_logs/aggressive-engine-scout-2026-05-22.json`.
- First verdict: raw Core ML eval is slower than fused PyTorch/MPS at batch
  128 (Core ML ~8.5-9.1 ms vs PyTorch/MPS ~2.9 ms), and INT8 weight
  quantization did not help raw latency in this conversion path.
- The engine-isolation thesis still has teeth: PyTorch/MPS eval pressure
  slowed MPS trainer steps by ~2.65x, while Core ML pressure lanes slowed
  trainer steps by ~1.13-1.32x. Next scout should be production-shaped
  self-play throughput with trainer overlap, not just naked eval latency.
- Filed the receipt and interpretation in
  [topics/ane-int8-inference.md](topics/ane-int8-inference.md).

## [2026-05-21] perf | Conv+BN fusion validated in production via WL5 worker hot-restart

- Microbench (`perf_microbench --no-fuse-eval` vs default): **1.47×**
  throughput speedup (710 → 1047 aug pos/s, median of 5 trials each,
  both contending with live WL5 for MPS).
- Hot-restarted 8 self-play workers in-place while WL5 trainer kept
  running. Canary w0 first (verified healthy on a fresh model version),
  then the remaining 7 in parallel. Total wave-mode stall: ~30s across
  two restart blips (epochs 5046 and 5051).
- Production gen-side measurement (n=26/n=25 epochs): **1.53× games/sec**
  on gen (21.6 → 33.0), slightly above microbench because workers no
  longer compete with the bench. Per-batch wave times in worker logs
  confirmed: pre-fusion 8-game wave 3.3s, post-fusion 2.4s.
- Per-batch worker log evidence (`w0.log`): pre-fusion v5044 batch 8011
  = 3.3s, post-fusion v5046 batch 20 = 2.4s (same 8-game shape).
- Caveat: post-restart window overlaps the WL5 archive-start
  absorption rough patch (pl jumped 0.59→0.73, plies 41.9→32.8). That
  ate the games/sec gain on a positions/sec basis (aug-pos/hour ~flat).
  Re-measure after WL5 reports out for a stable-plies cycle-time
  ratio.
- Full numbers + reusable hot-restart procedure landed in
  [TRAINING_WIKI.md](../TRAINING_WIKI.md) under the 2026-05-21
  fusion entry "Production verification" subsection.

## [2026-05-21] philosophy | M5 Max as mainframe, 9×9 as perf proving ground

- Added [topics/m5-max-as-mainframe.md](topics/m5-max-as-mainframe.md)
  capturing the guiding philosophy for the post-WL5 perf era. Treats
  the M5 Max as a single, knowable mainframe — invest in chip-specific
  tuning (parameter sweeps, unified-memory pipelining, MPS-fallback
  elimination, custom Metal kernels) rather than generic ML recipes.
- 9×9 gomoku is explicitly the perf proving ground, not the endpoint.
  Deliverable is a calibrated chart of the M5 Max's gomoku-AZ behavior
  (the contour plot), used to confidently pick knobs for 15×15.
- Compounded chip-specific levers (ANE INT8 × pipelined ANE+GPU+AMX ×
  custom Metal kernels) plausibly buy 10-25× throughput, which is
  what makes a month-long 15×15 + Gomocup submission realistic on
  one machine.
- Sequenced after WL5 → buffer cheap-test → ANE INT8 → canonical
  sweep → 15×15 renju port → Gomocup protocol → calibrated ELO.
- Indexed in [index.md](index.md). Short-form lives in memory as
  `feedback-know-the-machine`.

## [2026-05-21] plan | bit-packed replay buffer as post-WL5 task

- Added [topics/buffer-bit-packing.md](topics/buffer-bit-packing.md)
  scoping a refactor that shrinks per-position storage 17× by bit-packing
  the binary stones (currently float32) and using FP16 for pi.
- Current buffer: 1.5M positions = 6,250 games at 8.2 GB on MPS. Packed
  encoding at the same RAM footprint: 100k games (16× wider). At a 10 GB
  CPU footprint: 1M games (Jason's target).
- Motivated by WL5's archive-start absorption phase showing a real
  loss bump — wider buffer smooths target-distribution shifts and
  reduces need for EMA + past-mix kludges.
- Cheap-test first: train 1.5M vs 750k buffer ablation for 500 epochs;
  only do the refactor if halving meaningfully worsens stability.
- ~3 days work. Do AFTER WL5 reports out, ideally after
  [ANE INT8 inference](topics/ane-int8-inference.md) (the two don't
  conflict but ANE pays off in cycle time immediately).
- Indexed in [index.md](index.md).

## [2026-05-21] plan | ANE INT8 inference as post-WL5 task

- Added [topics/ane-int8-inference.md](topics/ane-int8-inference.md)
  scoping the port of self-play + eval inference to Apple Neural Engine
  at INT8 precision via Core ML.
- Captured during WL5 monitoring. Estimated ~50-60% faster self-play
  cycle if it lands; ~2 days of work. Calibration data is the WL5
  validation archive (1400 positions, already mined). KataGo INT8
  precedent says board games tolerate INT8 with proper calibration.
- Gate: validate INT8 model elo within 30 points of FP32 over 200+
  games before deploying to workers. Trainer stays FP32.
- DO this AFTER WL5 reports out — mid-run backend changes invalidate
  comparisons.
- Indexed in [index.md](index.md).

## [2026-05-21] run | WL5 launched (diagnostics + Go-Exploit archive-start)

- WL5 cell running as wandb run `o6cbjfnr`, resumed from WL4 e4024 with a
  fresh wandb timeline (stripped wandb_run_id from a copy of the
  checkpoint). 5000-epoch target. ~11s/cycle, ~320 epochs/h.
- Added the launch entry to [`../TRAINING_WIKI.md`](../TRAINING_WIKI.md)
  including the 3-bug triage that gated launch (high_kl positions had
  ply=0 from buffer backward-compat, causing C MCTS to play action 0 on
  a full board). Fixes in commit `dc8c38b`: derive-ply at mine time,
  C-level "no legal action = terminal" safety net, C-level select_action
  default to first legal action, Python evaluator nan_to_num.
- Workspace regenerated with WL5 + section 7 (validation archive +
  H/KL decomposition + per-color/per-ply): https://wandb.ai/jasonyandell-forge42/gomoku?nw=sm5st7cmye2

## [2026-05-21] docs | mining-validation-archives recipe

- Added [topics/mining-validation-archives.md](topics/mining-validation-archives.md):
  command, bucket cost drivers, throughput numbers (40-90 min wall for 6
  buckets × 200 positions on MPS), and the anti-patterns we learned the
  hard way during WL5 setup (don't `torch.load` the full 8 GB checkpoint
  N times in parallel; don't run without `-u`; don't run on CPU; don't
  co-run with training).
- Indexed in [index.md](index.md) Start-Here table.

## [2026-05-21] docs | how-to-play page

- Added [topics/playing-the-model.md](topics/playing-the-model.md) covering the
  local web UI surface (strong play), the live SPA, checkpoint selection (incl.
  the "latest.pt is huge, prefer epochNNNN.pt" trap), play/replay-tab knobs,
  and common annoyances (MPS contention, stale `epoch0136.pt` smoke checkpoint
  in `./checkpoints/`).
- Indexed in [index.md](index.md) Start-Here table.

## [2026-05-19] setup | LLM wiki operating model

- Added [index.md](index.md) as the wiki entry point and content catalog.
- Added [topics/wiki-operating-model.md](topics/wiki-operating-model.md) to adapt
  the Karpathy LLM wiki pattern to this repo.
- Added [sources/karpathy-llm-wiki.md](sources/karpathy-llm-wiki.md) as the
  source record for the organizing charter.
- Updated [../AGENTS.md](../AGENTS.md) and [../TRAINING_WIKI.md](../TRAINING_WIKI.md)
  so future sessions treat the wiki as a compounding synthesis layer, not just a
  large experiment transcript.

## [2026-05-19] synthesis | MCTS perf ceiling topic page

- Added [topics/mcts-perf-ceiling.md](topics/mcts-perf-ceiling.md) capturing
  the finding that our `gomoku/mcts.py` is already at the AGZ "mcts_v2"
  storage layout that other AZ codebases advertise as a big upgrade. Ports
  of that design are a no-op for us.
- The cross-game BFS-vectorized descent (Exp 9 in
  [../TRAINING_WIKI.md](../TRAINING_WIKI.md)) is real but small: 1.26× at
  G=32 / wave=16, ~1.05–1.10× at our dist G=8 / wave=32 config. The next
  2× requires batched `state.apply` on tensor, C-extension `_init_node`,
  or multi-device gen-vs-train split — not numpy reshuffling.
- Updated [index.md](index.md) Start Here table to surface the new topic.

## [2026-05-19] notebook | az-recipe-160k launched, first sustained heuristic crossing

- Live run `9x9-sweep-az-recipe-160k` (wandb `sppjo3z5`) is at e1285 with
  heuristic 50-70% sustained across e1119/1179/1227 — never seen in dist
  mode before. Prior best dist crossing was fresh-dist plateauing at ~35%
  by e552; all other dist cells stayed at 0% heuristic for the entire 100
  epochs they ran.
- Launched config differs from the documented recipe: small model +
  stem_padding=1 + sims=400 (instead of medium / 3 / 800) to keep wall-
  clock inside one day on the M5 Max. The recipe defaults benched at
  ~30 s/cycle = ~46h for 5000 cycles; trimmed config is ~2 s/cycle when
  games are short.
- Captured Jason's calibration that cycle-time scales super-linearly
  with mean plies, so the current ~2 s/cycle is a *floor* — real defense
  learning would blow out the ETA. Filed as a feedback memory so future
  sessions don't quote naive ETAs.
- Added live SUMMARY to [../TRAINING_WIKI.md](../TRAINING_WIKI.md) with
  the eval table, plies puzzle, and open questions to resolve as the run
  continues (does heuristic hold, does lookahead2 climb, do plies
  regrow, head-to-head vs kze-e176).

## [2026-05-19] notebook | az-recipe-160k diagnostics resolved: real defense

- ~280 epochs after the SUMMARY was written, every "open question"
  diagnostic resolved in favor of real defense being learned, not
  offense-only. lookahead:depth=2 climbed from 12% (e1119) to 55%
  (e1507). selfplay/plies_p90 spikes to 60-80 at e1.5k+, eval times
  doubled-to-tripled as games got longer.
- Jason flagged the **selfplay/plies_p90** chart as the leading
  indicator before it showed up in the mean. Filed as a tactical note
  in the wiki: when the model is in transition between offense-only
  and real-play, p90 is more sensitive than mean because the
  distribution is bimodal (short attack wins + long defense games).
- This recipe + cutback combo is the first in the wiki to break the
  fast-attack collapse: no prior dist run crossed lookahead2 above 25%.
- Speculation on what made the difference: most likely τ_final=0.1
  (soft policy targets instead of degenerate one-hot), then AGZ
  log-PUCT, then 1.5M replay. A τ_final=0 ablation would resolve it.

## [2026-05-19] notebook | az-recipe-160k e2179 checkpoint — full regime change

- At e2179 (43% complete, 2:47h wall-clock), the self-play plies have
  fully regrown to 27-32 mean — the same range as the e1 untrained
  baseline, but for the opposite reason: defense, not random play.
- Loss/policy down to 0.76 from 4.22 at e1. Loss/value at 0.08 — model
  is very confident. No sign of value-head collapse to z=0/-1 (the
  classic failure mode from earlier runs).
- Elo eval shipped at e1854 (commit fa656b9); model_elo bouncing
  1085-1183 across e1854/2148/2159/2167. Stable around 1100-1150,
  between heuristic (anchor 800) and lookahead2 (anchor 1200).
- ETA blowout exactly as Jason's calibration predicted: cycle time
  grew 2s → 15s as plies regrew. Total projected ~14.6h vs original
  10h estimate. The interesting outcome (real defense) was always the
  one that costs wall-clock.
- Anchor Elo calibration script running in background to replace the
  seeded ANCHOR_ELOS with measured values from a round-robin between
  random + heuristic + lookahead{2,3,4,5}. Will update rating.py when
  calibration finishes.

## [2026-05-20] notebook | lookahead-depth-3 bug diagnosed + partial fix

- Calibration finished. The Elo spread among baselines is much tighter
  than seeded: heuristic=591, depth=2=604 (≡heuristic, all-draws), depth=4=629,
  depth=5=711. Anchored at random=0.
- Per Jason's call, **NOT re-anchoring** ANCHOR_ELOS in code: heuristic and
  depth=2 being equal-Elo is a style coincidence, not a meaningful collapse.
- depth=3 came in at **Elo=249** (weaker than heuristic). Subagent investigated;
  it's a horizon-effect bug in the static `evaluate_position` — credits "live
  4" patterns without distinguishing open-fours (unblockable) from half-open
  fours (trivially blocked). At odd depths the searcher builds a hallucinated
  threat the opponent never gets to refute before the leaf.
- Shipped **partial fix** in `gomoku/baselines.py:_negamax` adding depth=0
  1-ply quiescence for immediate-win threats. Effect: depth=3 vs heuristic
  goes from 0% (all losses) to 83% (d=3 wins majority). depth=3 vs depth=2
  unchanged (still 0%) — remaining bug likely in open-3 pattern credit, not
  just live-fours. Regression test in `tests/test_lookahead_quiescence.py`
  pins the corrected leaf value.
- The bug doesn't affect model training (uses network value head, not
  `evaluate_position`) or the live eval pipeline (we use depth=2, which is
  even and unaffected). Filed the remaining open-3 issue as a known
  limitation, not blocking the current run.

## [2026-05-20] notebook | perf detour, GPU reality, white_wins → 0 (e3500+)

- Tried a subagent-proposed perf change: 1 worker × 32 games × wave=64 +
  torch.compile, expected 2-3× speedup. **Regression** — cycle time grew
  from 33s (4-worker baseline) to 36→63s on the 1-worker path. Rolled
  back to 4 workers, kept wave=64 (the real win from the bench), dropped
  torch.compile. Net: ~22s/cycle, +50% vs the original baseline.
- Jason flagged the 1-worker setup as "kinda crazy" and predicted GPU
  underutilization — right on both counts. Bumped to 8 workers per his
  call: ~17s/cycle, ~94% over baseline. GPU still at 30-40% though —
  the structural ceiling is small kernels (2ms regardless of batch) on
  a tiny 324k-param model, not parallelism.
- Updated [topics/mcts-perf-ceiling.md](topics/mcts-perf-ceiling.md) with a
  2026-05-20 section: lessons on per-process bench vs production parallelism,
  compile-vs-reload-cadence, why more workers > bigger batches at this
  model size. Memory `project_perf_bench_lesson` mirrors this for cross-
  session recall.
- Observed `selfplay/white_wins → 0` once buffer/age_p50 rolled from 250
  to ~75. This is the first-mover-advantage signal in freestyle 9×9:
  asymptotic state of perfect self-play has black always winning. Value
  head signal degrades for white-side positions (always z=-1, trivially
  fittable), which probably explains some of the ongoing pl/vl uptick.
  Recorded in TRAINING_WIKI.md as a strength-signal observation, not a
  bug. `--random-opening-moves` would break the asymmetry if desired.
- Also pushed: `--worker-min-positions` + `--sgd-per-position` ingest mode
  (commit 85eeccc, not yet deployed live), eval-side `play_match_parallel`
  via mp.Pool (commit d913447, lets lookahead:depth=4 fit in eval budget).

## [2026-05-20] notebook | lockstep vs continuous orchestration analysis

- Jason raised the lockstep question — would `--gen-once-per-publish` mode
  help, possibly in a "2 waves of 4" staggered design?
- Filed full trade-off analysis in
  [../TRAINING_WIKI.md](../TRAINING_WIKI.md) "Lockstep vs continuous"
  section. Key findings: (1) at our 30s-batch / 10s-cycle ratio, workers
  are already the bottleneck so lockstep adds zero idle cost; (2) the
  "2 waves of 4" staggered overlap collapses to serial because publish
  K+1 has a hard dependency on K being consumed first; (3) the "model
  does better with lockstep" intuition cites one heuristic flicker at
  e65 in `nox388ow` — slightly-less-bad, not a documented training win;
  (4) lockstep is a training-side lever, not a perf lever (GPU
  utilization is bound by per-call kernel size, not worker alignment).
- Decision: not deploying today. Realistic deployment recipe documented
  as the natural next intervention if training-stability problems
  emerge (pl > 0.6 sustained, value collapse).

## [2026-05-20] notebook | next-run config sketches collected

- Jason flagged buffer/age_mean as an under-utilized knob: "for the next
  run, I really want a full buffer and this flat AND a bunch of games-
  per-model, rather than some-games-across-a-spread-of-models."
- Captured the math (`median_age ≈ buffer_size / (2 × positions/cycle)`)
  + the chart interpretation (buffer age climbed to 250 e1-1000, fell to
  steady-state 50-60 as games lengthened, restart-induced transients
  visible at e3000-3700).
- Drafted cell Zlock as a candidate next-run config: 4 workers in
  lockstep (--gen-once-per-publish), 5M buffer, positions-based ingest.
  Gives age ~195 — close to Jason's 200-250 target.
- Also listed 8 decisions to re-assess when the current run finishes
  (stem_padding 1 vs 3, model size, sims 400 vs 800, K, random-opening
  moves, past-checkpoint opponent mix, lookahead bug structural fix,
  the structural perf "real next 2×").
- Decision: NOT pre-registering the cell. Hold the collection in
  TRAINING_WIKI.md until the current run finishes; re-assess with the
  current run's full data in hand. Next-run config should be picked
  with a specific question in mind, not "improve everything."

## [2026-05-20] synthesis | az-at-scale-vs-laptop topic page

- Added [topics/az-at-scale-vs-laptop.md](topics/az-at-scale-vs-laptop.md)
  capturing Jason's framing observation that "steady progress is what
  makes [AZ] learn, and this one has been learning despite the chaos."
- The page documents three structural reasons our laptop setup wrinkles
  (exploration arcs, plies swings, age oscillations) don't exist at
  Google scale: (1) per-version concentration because 8 workers can't
  smooth across thousands of parallel games, (2) short freestyle gomoku
  games give 10-40× less signal per game than Go's 200-250 move games,
  (3) restart artifacts that thousand-machine continuous runs don't have.
- Key framing argument: when reading our wandb metrics, the *default
  interpretation* of swings should be "laptop-scale transient, model is
  doing something interesting," not "training is broken." Failure
  diagnosis requires extra evidence (NaN, dying processes, OR sustained
  multi-arc degradation).
- This argues *against* twitchy interventions (each restart costs 100+
  epochs of buffer re-equilibration) and *for* big-buffer + lockstep +
  many-games-per-version next-run config (the Next-run sketches in
  TRAINING_WIKI.md address exactly these scale-effect items).
- Updated [index.md](index.md) Start Here to surface the new topic.

## [2026-05-20] notebook | Jason's buffer-composition-feedback prediction

- After observing three arcs and the constant-age math, Jason articulated
  a deeper failure mode than the surface-metric swings: each exploration
  arc *changes the shape of the buffer's history* by ingesting short-game
  positions, so the consolidation phase is fighting against the very
  data it's training on. Eventually one consolidation will fail.
- Prediction (e4252): pl/vl will climb again over the next half hour, then
  drop again — same plateau-learn-plateau-learn cycle until eventually
  it doesn't recover.
- Filed in [../TRAINING_WIKI.md](../TRAINING_WIKI.md) "Buffer-composition
  feedback hypothesis" section. Notes that the constant-age fix shipped
  earlier (85eeccc) keeps turnover stable but does NOT change buffer
  composition during exploration — the genuine mitigations are random
  opening moves and past-checkpoint opponent mix (deferred decisions #5
  and #6 in the Next-run sketches).
- Cron check-ins continue tracking. The prediction is falsifiable in two
  ways: (a) bounces as predicted, eventually fails to recover → validates
  the theory and argues strongly for items #5-#6; (b) tightens
  asymptotically with smaller arcs → theory over-stated at this scale.

## [2026-05-20] notebook | az-recipe-160k run ended at e5000

- Run stopped by user-requested kill at exactly e5000 (early from natural
  e8560 endpoint — data was already conclusive). Final state: pl=0.293,
  vl=0.035, plies=59.2, model_elo bouncing 1290-1519 in the last 5 evals.
- 5 explore-then-consolidate arcs across e3041-e4924. Peak model_elo
  1718 at e3881 (perfect sweep of random + heuristic + lookahead2).
- Jason's "buffer-composition feedback causes arcs" hypothesis partially
  validated: arcs DID happen, DID broaden over time (5th arc the
  broadest weakness, heuristic-specific lineage drift visible), but the
  "eventually doesn't recover" branch did NOT materialize — every arc
  recovered, even the broadest one.
- Filed full run-end SUMMARY in
  [../TRAINING_WIKI.md](../TRAINING_WIKI.md) with arc table, validated/
  refuted/partially-supported hypotheses, and the case for the next run's
  design choices. Most informative single next-run experiment: lockstep
  + 5M buffer + random opening moves (the three changes most directly
  aimed at the failure mechanisms we observed).
- Deleted the 15-min check-in cron (job 43ad02e9). Next session that
  spawns a check-in cron should start fresh.

## [2026-05-20] design | wave-of-lockstep design page added

- Jason and I talked through the next-run design. Locked-in choices:
  8 workers × 8 games per worker per wave, greedy-fill barrier with
  finish-on-old-model semantics, K=1 SGD step per wave, 5M buffer,
  natural openings (no randomization), temperature unchanged from
  `az-recipe-160k`.
- Filed full design at
  [topics/wave-of-lockstep-design.md](topics/wave-of-lockstep-design.md):
  hypothesis, architecture diagram, property invariants, implementation
  plan (trainer barrier, worker greedy-fill state machine, new Cell
  WL1, W&B metrics), throughput expectations, held-back levers.
- Indexed from [index.md](index.md) under "Start Here".
- Next session picks this up to implement. Sanity test on 50 epochs / 4
  workers before launching full WL1 run.

## [2026-05-20] implementation | WL1 wave-lockstep landed and smoked

- Implemented the trainer wave barrier (`--wave-mode --wave-workers
  --wave-games-per-worker`) and worker greedy-fill state machine
  (`--wave-mode`) in worktree `codex/wl1-lockstep`.
- Added `WL1` to [../scripts/run_sweep.py](../scripts/run_sweep.py):
  small model, 400 sims, stem padding 3, 8 workers x 8 games, 5M buffer,
  AGZ PUCT/Dirichlet defaults, temperature drop at move 30, and
  `sgd_per_position=0.0025`.
- Smoke-tested 50 epochs / 4 workers with `G=8` and a 1.3M replay buffer.
  Parsed 50 wave tiles for versions `0..49`; each wave met worker minimum
  >= 8, tile sizes ranged 38-54 games, and final replay-buffer
  `weight_version` tags contained all versions `0..49`.
- Filed the detailed receipt in
  [../TRAINING_WIKI.md](../TRAINING_WIKI.md) under "2026-05-20 — WL1
  implementation smoke".

## [2026-05-20] benchmark | WL1 matched-throughput read

- Ran a short apples-to-apples throughput check with the previous
  `az-recipe-160k` generation config: small model, stem padding 1, 400
  sims, wave size 64, MPS, 8 workers.
- The 3-epoch wave-mode check ingested 250 games in 31.9s of generation
  time: 7.84 games/s, 1,817 approximate training positions/s, average
  visible tile 72.7 games with greedy extras.
- Compared against `az-recipe-160k`, this is comparable by positions/s
  to the early continuous run (1,773 positions/s over first 100 epochs)
  and stronger than the first-3-epoch and late-run slices. The barrier
  itself did not show a meaningful throughput tax.
- Filed the interpretation in
  [../TRAINING_WIKI.md](../TRAINING_WIKI.md) under "2026-05-20 — WL1
  matched-throughput read".

## [2026-05-20] runbook | Activity Monitor perf integration lane

- Added [topics/activity-monitor-perf-runbook.md](topics/activity-monitor-perf-runbook.md)
  as the maintained run/config surface for Mac Activity Monitor-oriented perf
  checks. It routes future sessions toward wall-clock/games/sec/positions/sec
  and away from GPU-percent chasing.
- Added `scripts/perf_microbench.py` as a bounded production-shaped MCTS
  generation bench. It exercises the existing evaluator + `generate_games`
  path without touching core game or MCTS code.
- Updated [../README.md](../README.md) with the bench command and the
  `--save-buffer-every` / `--keep-last-n` checkpoint-throttling recipe for
  long 5M-buffer runs.
- Changed WL1 in `scripts/run_sweep.py` to write full replay-buffer
  `latest.pt` checkpoints every 100 epochs instead of every 20. Intermediate
  epoch snapshots remain cheap weights+optimizer files.

## [2026-05-20] config | WL1 buffer downshift

- Updated `scripts/run_sweep.py` WL1 from a 5M replay buffer to 1.5M after the
  hardware/readiness decision that 5M is too much to justify right now.
- Preserved WL1's main experimental axis: wave-lockstep / per-version uniformity.
  The next run now avoids changing buffer size at the same time.
- Kept `save_buffer_every=100` as a low-risk disk-pressure guardrail. It is less
  critical at 1.5M than 5M, but still prevents full replay-buffer rewrites from
  becoming a hidden Activity Monitor problem.

## [2026-05-21] implementation | native MCTS engine landed

- Added optional `gomoku._mcts_native` in worktree `codex/gomoku-perf-extension`.
  It moves the self-play MCTS arena, bitboard state/history, PUCT selection,
  child creation, virtual loss, backup, and leaf plane materialization into C.
- Wired `generate_games` to use native MCTS automatically when the evaluator
  exposes `evaluate_planes`; `GOMOKU_DISABLE_NATIVE_MCTS=1` keeps the old Python
  MCTS path available for A/B checks.
- Updated [topics/mcts-perf-ceiling.md](topics/mcts-perf-ceiling.md),
  [topics/activity-monitor-perf-runbook.md](topics/activity-monitor-perf-runbook.md),
  [../TRAINING_WIKI.md](../TRAINING_WIKI.md), and [../README.md](../README.md)
  with the new boundary and benchmark receipts.
- Reference MPS microbench: `8 games / 400 sims / wave 64 / max_plies 16`
  improved from 701 to 2,200 augmented positions/sec; `max_plies 32` improved
  from 728 to 2,007 augmented positions/sec.

## [2026-05-21] benchmark | WL1 10-epoch native production read

- Ran three fresh 10-epoch wave-lockstep throughput trials under the next-run
  WL1 recipe (`small`, stem padding 1, 400 sims, wave 64, 1.5M buffer).
- Results saved in `sweep_logs/perf10-summary.tsv`.
- Best launch shape remains 8 workers x 8 games with native MCTS: 2,379 wall
  augmented positions/sec and 3,303 generation augmented positions/sec.
- Same 64-game tile with 4 workers x 16 games was slower (1,918 wall pos/s).
- Python-MCTS fallback at 8 workers x 8 games was 1,863 wall pos/s, so native
  is a 1.28x production-shaped wall-throughput win even though the single-process
  microbench showed a larger 2.8-3.1x jump.

## [2026-05-20] notebook | WL1 live run — buffer-mix hypothesis showing positive early signal

- WL1 launched 20:53 as wandb `l8mbntcm` (commit `0d2c106`). First-attempt
  worker race + 5M-buffer MPS crash both diagnosed and fixed pre-launch;
  see [../TRAINING_WIKI.md](../TRAINING_WIKI.md) "WL1 first launch + worker
  race fix" for the receipts.
- At e605 (~35 min wall), the run is well ahead of Z's e605 by every
  fixed-baseline measure: first heuristic crossing happened at e146 (Z:
  e1119), elo hit 1271 at e360 (Z: ~e1854 reaches similar), la4 sustained
  52% at e499 (Z barely reached this even at e3881 peak).
- Arc behavior is compressed: WL1's first explore-consolidate arc had a
  ~80-100 epoch wavelength vs Z's 800-1000 epoch arcs. First consolidation
  dip in progress at e605, recovering to elo 1041 (the trough is still
  Z-e1854-class strength).
- Wall-clock advantage combines native MCTS perf (~1.7×) and per-epoch
  convergence speedup. To disentangle requires WL1 with
  `GOMOKU_DISABLE_NATIVE_MCTS=1`; not in scope today.
- Started a live training log section in
  [../TRAINING_WIKI.md](../TRAINING_WIKI.md) "WL1 live run log" — milestone
  table updates as new evals land.
- Open question still ahead: do plies regrow (defense regime) on a
  similarly accelerated schedule? At e605 plies still 10-11, value head
  already at Z-e2179 levels.

## [2026-05-20] tooling | wandb workspace layout script

- Added [../scripts/wandb_workspace.py](../scripts/wandb_workspace.py) — one-shot
  creator for a 6-section wandb workspace tuned for WL1-vs-Z overlays
  (strength, learning, game shape, buffer, wave dynamics, wall economy).
  Requires `wandb-workspaces` (pip install).
- Live workspace URL (the view this script created on 2026-05-20 20:55):
  https://wandb.ai/jasonyandell-forge42/gomoku?nw=ul0vliphj6x
  Open it, click both `l8mbntcm` (WL1) and `sppjo3z5` (Z) in the run
  picker. Switch the x-axis to `_runtime` (per-panel) to see wall-clock
  savings instead of per-epoch convergence.
- The workspaces API does not update views in place — re-running the
  script creates a new URL. Treat the bookmarked URL as canonical until
  it's superseded; delete stale views via the UI if they accumulate.
- Cross-ref: training notebook "WL1 live run log" section in
  [../TRAINING_WIKI.md](../TRAINING_WIKI.md) names the metrics each
  section is meant to surface.

## [2026-05-20] notebook | WL1 stopped at e1600; WL2 scale-emulation design landed

- WL1 ran 1h 18min wall, peaked elo 1281 at e360-499, then dropped into
  a regression band (elo 620-1140, **la4 regressed from 52% to ~5%**).
  Stopped by user when it was clear the new failure mode wasn't
  self-correcting. Final state + arc breakdown in
  [../TRAINING_WIKI.md](../TRAINING_WIKI.md) "WL1 run end — stopped at
  e1600" entry.
- Reframe: per-version uniformity is *necessary but not sufficient*. WL1
  replaced Z's slow consolidation arcs with high-frequency oscillation
  because removing per-version bias also removed the *in-flight version
  diversity* that AZ-at-scale has by default (async publish lag, ~125k
  concurrent games, batch 4096).
- Filed next-run design at
  [topics/wl2-scale-emulation-design.md](topics/wl2-scale-emulation-design.md).
  Four levers, each emulating one AZ-scale property:
  EMA self-play weights (biggest single intervention), past-checkpoint
  opponent mix, worker poll jitter, gradient accumulation 4×.
  Implementation cost ~120 LoC, throughput hit ~5-15%.
- Indexed from [index.md](index.md) Start Here.

## [2026-05-20] notebook | WL2 launched — four scale-emulation levers stacked

- Wave 2 of the wave-lockstep series. Cell `WL2` adds all four levers from
  [topics/wl2-scale-emulation-design.md](topics/wl2-scale-emulation-design.md)
  on top of the WL1 recipe: EMA self-play (tau=0.99), past-checkpoint
  opponent mix (recent=0.4 / history=0.1 / window=100), worker poll jitter
  (Uniform 2-8s), gradient accumulation 4x.
- Two background implementation agents in parallel landed cleanly:
  `b582d37` (train.py: EMA + grad accum) and `ded7728` (selfplay_worker.py:
  past-mix + poll jitter). Cell wiring + new Cell fields at `02c5fc3`.
  All 88 tests pass.
- Pre-launch 30-epoch smoke validated all four lever signals; mix distribution
  reached 12/43/45% (history/recent/self) against designed 10/40/50.
  Cycle ~5s vs WL1's 3.4s — ~50% slowdown from grad-accum + past-ckpt loads.
- Live run: wandb `9wng4yu9`, launched 2026-05-20 ~23:00, 5000 epochs.
  Tracked in [../TRAINING_WIKI.md](../TRAINING_WIKI.md) "WL2 live run log"
  section. Add `9wng4yu9` to the wandb workspace run picker for three-way
  overlays with WL1 (`l8mbntcm`) and Z (`sppjo3z5`).

## [2026-05-20] tooling | launch sequence runbook + skill update + 3-way workspace

- Added [topics/launch-sequence-runbook.md](topics/launch-sequence-runbook.md)
  capturing the WL1+WL2 launch pattern as a reusable playbook: pre-launch
  state check + the two known gotchas (MPS INT_MAX, worker race fix), title
  card protocol, smoke test pattern, real launch + spin-up verify, wiki +
  workspace updates, /loop monitoring cadence, and the fan-out-implementation
  pattern that landed WL2 cleanly via two parallel agents in ~10min wall.
- Updated `scripts/wandb_workspace.py` to include WL2 (`9wng4yu9`) alongside
  WL1 (`l8mbntcm`) and Z (`sppjo3z5`). Fresh 3-way overlay view:
  https://wandb.ai/jasonyandell-forge42/gomoku?nw=cz8thj3cbh5
- Updated the `gomoku-train` user skill at
  `/Users/jason/.claude/skills/gomoku-train/SKILL.md` to surface the production
  cell-based launch path, the cell map (A-F / Z / Zc / WL1 / WL2 with wandb
  ids), the two gotchas as Don'ts, and a pointer to the runbook for every
  "start a run" type request. Single-process `gomoku.train` path stays
  documented for ad-hoc smoke work.

## [2026-05-20] notebook | eval-time-vs-heuristic as a hidden plies-regrowth indicator (WL2)

- Jason flagged at WL2 e420 that `time/eval_vs_heuristic_s` climbed from ~6s
  at e1 to ~17s by e420 — roughly 3x more wall-clock per eval. Since the
  eval plays 16 fixed games vs heuristic per cycle, the per-move cost is
  constant; the wall-clock growth maps directly to more plies per game.
- This is the **plies-regrowth signal hiding outside of `selfplay/plies_mean`**:
  in WL2 the self-play tile still shows plies ~11-12 (model beats its own
  EMA-smoothed brain fast via attacks), but vs heuristic — a different
  style — the model has learned to fight back to 30+ plies. Eval-time
  surfaces real defensive capability that selfplay metrics don't expose.
- WL2's eval-time climb roughly coincides with the first heuristic
  crossing at e370 (15%), suggesting the time-climb leads the win-rate
  signal: defensive ability shows up in game length before it shows up
  in actual wins.
- Watch for: does WL2's eval-time *stay* climbing as the model approaches
  sustained crossing, or does it plateau then drop? Z plateaued; WL1's
  late-run collapse coincided with eval times dropping.
- Filing in the runbook "Leading indicators" section as a hidden but
  high-value signal.

## [2026-05-21] notebook | WL2 ended at e1200, WL3 launched with K=2 random openings

- WL2 (wandb `9wng4yu9`) stopped at e1200 / 1h 11min wall. Final state:
  pl=1.89, vl=0.012, plies=10.5 (selfplay), elo bouncing 788-1071 in
  the last 6 evals, **la4 regressed from peak 62% at e900 to 18% at
  e1101**. The four scale-emulation levers raised the ceiling (peak la4
  62% > WL1's 52%) and smoothed early trajectory (heuristic 0→15→5→8
  vs WL1's 30→0→15→0) — but the late-run failure mode matched WL1's
  (~44pp la4 drop vs WL1's ~47pp). Full close-out in
  [../TRAINING_WIKI.md](../TRAINING_WIKI.md) "WL2 run end" section.
- Reframe: even with EMA + past-checkpoint mix + jitter + grad-accum,
  all model versions share the same opening lineage. Worker diversity
  doesn't help if the "diversity" is "different brains thinking about
  the same opening."
- WL3 = WL2 + K=2 uniform-random opening plies. Per train.py:165-170,
  training examples are NOT recorded for the random plies, so the
  model sees more diverse mid-game starting positions without learning
  broken-move signal. K=2 is conservative; 30-epoch smoke showed plies
  bumped +20% (22-26 vs WL2 smoke's 16-20), confirming random openings
  produce slightly longer games as expected.
- Live run: wandb `0o75gws5`, launched 2026-05-21 ~00:10. Anchored by
  commit `91f7408`. Tracked in
  [../TRAINING_WIKI.md](../TRAINING_WIKI.md) "WL3 live run log" section.
- Workspace regenerated via `scripts/wandb_workspace.py` to include
  WL3. New URL printed by the script (workspaces API doesn't update in
  place — each run produces a new view).

## [2026-05-21] notebook | WL3 e515 sustained crossing + eval-distribution test result

- WL3 (wandb `0o75gws5`) reached its **first sustained heuristic crossing
  at e487-e515** with h=50% held across two consecutive evals and all
  three baselines climbing together (h50/la2:25/la4:38, elo 1031 at e515).
  Slower to first crossing than WL2 (e487 vs WL2's e370) but the
  *strength profile* is fundamentally different — WL2's first 200
  epochs after crossing showed single-baseline spikes (heuristic 88%
  while la4 collapsed to 5%); WL3's first 30 epochs after crossing show
  balanced wins across all three baselines.
- Tentative plies regrowth signal: `selfplay/plies_mean` bumped
  13.2 → 15.0 over the last 250 epochs. WL2 stayed pinned at 11.x for
  its entire run. Too small to call decisive, but it's in the right
  direction for the first time in the WL series.
- Retention test still in progress. WL2 lost la4 from peak 62% (e900)
  to 18% (e1101) over 200 epochs. WL3 is in the equivalent window now.
- **Eval-distribution test (Jason's hypothesis about K-mismatch hiding
  signal):** ad-hoc CPU match on WL3 epoch0361 vs heuristic at K=0/2/4
  random openings. Result: **K=0 and K=2 win rates within noise (0.350
  vs 0.338 over 40 games)** — the matched-distribution eval does NOT
  reveal hidden strength. K=4 was much worse (0.200), showing the
  model is fragile far OOD. Conclusion: WL3's slow first-crossing is
  a real training-side phenomenon, not an eval-distribution artifact.
  Decision: skip the "plumb random-opening eval everywhere" lever;
  lean into the queued opening-curriculum experiments (Q1-Q4 in
  [../TRAINING_WIKI.md](../TRAINING_WIKI.md)) instead.
- **Important caveat surfaced by this test**: trainer's e361 eval
  reported h=5% (≤1 win on 16 games); 40-game test reported h=35%.
  Possibly sample-size variance, possibly native-MCTS-vs-python-MCTS
  path difference. Filed in the runbook + user skill so future
  sessions don't over-interpret single-eval bouncing.
- Updated `gomoku-train` user skill at
  `/Users/jason/.claude/skills/gomoku-train/SKILL.md` with WL3 cell
  entry, eval interpretation gotchas, and an "ad-hoc match" recipe
  for `$CLAUDE_JOB_DIR`-style forensic tests.

## [2026-05-21] notebook | WL3 crashed at e825 (NaN), WL3.1 launched with NaN guards

- WL3 (wandb `0o75gws5`) crashed at e825 from native MCTS emitting NaN
  visit-policies. All 8 workers died in sequence over ~15 min. Trainer
  barrier-stalled forever. Full diagnosis at `$CLAUDE_JOB_DIR/wl3_nan_diagnosis.md`.
  Run-end summary in [../TRAINING_WIKI.md](../TRAINING_WIKI.md) "WL3 run end".
- **Pre-crash WL3 was the strongest run in the WL series**: peak la4=68%
  at e714 (>WL2's 62%), all three baselines climbing balanced together,
  plies regrew 13→18. Crash was infrastructure failure, not training-
  quality failure.
- Two NaN fixes landed (`c5049be` + `0557671`): `_sample_action` guard
  for the play path, plus pi sanitization at the trajectory-recording
  path. The first fix alone was insufficient — NaN pi was being stored
  into the buffer before `_sample_action` ran, poisoning the trainer's
  cross-entropy targets. Found this during recovery attempt #1.
- WL3.1 (wandb `i34ihwj9`) launched as fresh restart with both guards
  in place. Identical cell config to WL3 (proven trajectory). Skipped
  smoke since the only changes are NaN-fallback paths covered by pytest.
- Old WL3 artifacts preserved as
  `sweep_runs/WL3-random-openings.dead-e825/` and parallel sweep_logs/
  for forensics.
- Workspace refreshed to include WL3.1:
  https://wandb.ai/jasonyandell-forge42/gomoku?nw=q5fg9ei2ash
- Native MCTS NaN root cause is under parallel investigation (background
  agent in worktree). The band-aids keep the run going while the C-level
  fix lands.
- **Skill update**: added "Unattended-run policy" section to
  `~/.claude/skills/gomoku-train/SKILL.md` defining what kinds of
  infrastructure fixes future sessions can apply autonomously during
  monitoring (single-file <50 line process-death prevention + hot
  resume), what requires human (training hyperparameter changes,
  re-architecture). Also need to think about poisoning paths, not
  just process-death paths — the WL3 recovery missed this and
  burned the buffer for 9 epochs.

## [2026-05-21] notebook | WL3.1 relaunched with native MCTS C fix — root cause closed

- Background investigator agent landed the root cause of the WL3 NaN
  crash (commit `7c3e405`): `NativeMCTSGame.policy(tau)` was casting
  `pow(N, 1/tau)` to float32 before normalizing. At τ=0.1 with
  concentrated visits N≥~7100, the cast overflowed FLT_MAX (~3.4e38) →
  `Inf/Inf` → NaN. Long concentrated games (which only appeared at
  e825+ in WL3 as plies regrew past ~18) reliably triggered it.
- Fix: do the τ-normalization in `double[]`, cast only final
  probabilities to float32. Plus +Inf-sum argmax-tie fallback.
  Regression test in `tests/test_native_mcts.py::test_native_policy_
  finite_at_low_temperature_with_concentrated_visits` — fails on
  pre-fix, passes on post-fix. 89 tests passing.
- WL3.1 first try (wandb `i34ihwj9`) ran 92 epochs with only the
  Python band-aids; relaunched as `44cxzc9d` after rebuilding the C
  extension. Same cell config. Old artifacts preserved at
  `sweep_runs/WL3.1-random-openings-nanfix.preCfix-e92/`.
- The C fix + the two Python band-aids (`c5049be` + `0557671`) close
  the failure mode at both levels: C-side overflow prevented, and
  Python-side NaN-in-pi still falls back gracefully if anything similar
  ever emerges.
- Workspace refreshed: https://wandb.ai/jasonyandell-forge42/gomoku?nw=tfzwgv1hwbp

## [2026-05-21] notebook | WL3.1 paused at e1536, WL4 (no random openings) launched

- WL3.1 (wandb `44cxzc9d`) paused after reaching "established" trigger
  Jason proposed: eval/vs_heuristic 100% sustained, la4 60-95%
  sustained across many evals, plies 20-27 (defense regime forming),
  elo 1400-1700. Strongest WL-series state by every measure.
- e1536 snapshotted aside: `$CLAUDE_JOB_DIR/wl3.1_e1536_latest.pt`
  (8.2G, includes model + EMA + buffer). WL3.1 artifacts preserved at
  `sweep_runs/WL3.1-random-openings-nanfix.paused-e1536/`.
- WL4 cell (`a88749d`): WL3.1 config with `random_opening_moves=0`.
  Resumes from the snapshot. Wandb run id continues (`44cxzc9d`) — the
  chart is a single trajectory with the K=2→0 transition at step 1537,
  cleaner than two separate runs to overlay.
- Hypothesis: WL3.1 baked in opening-diverse representations; removing
  random plies should either (a) unlock canonical-depth compounding
  that the random plies were rate-limiting, OR (b) trigger rapid
  regression — testing whether diversity is permanent training
  infrastructure at this model size.
- Either outcome informative. Recovery path: restart from the snapshot
  with K=2 if it collapses badly.

## [2026-05-21] runbook | handoff-friction section added to launch runbook

- Filed everything that bit us today as a "Handoff friction" section
  in [topics/launch-sequence-runbook.md](topics/launch-sequence-runbook.md).
  Eight gotchas covered:
  1. `latest.pt` vs `epochNNNN.pt` (the buffer-resume distinction —
     WL4's "resume at e1500 not e1536" surprise)
  2. `keep_last_n=3` brutally short — snapshot aside immediately
  3. `--resume` always continues the old wandb run id (feature for
     WL4 curriculum continuation, but surprising)
  4. Cell rename pattern for branching experiments from a paused run
  5. Workspaces API doesn't update in place (each script run = new URL)
  6. macOS `pgrep` quirks (`\b` doesn't work; transient self-PIDs)
  7. Old `/loop` chains keep firing once after you change cells
  8. `$CLAUDE_JOB_DIR/` is ephemeral — don't park anything important
     there for a future session
- Also noted: WL3.1 e1500 buffer-checkpoint is preserved at
  `sweep_runs/WL3.1-random-openings-nanfix.paused-e1536/checkpoints/latest.pt`
  (8.8G). That's the canonical resume point if WL3.1 needs to come
  back online. The `$CLAUDE_JOB_DIR` copy is redundant and will be
  cleaned up automatically.

## [2026-05-21] synthesis | loss-floor bouncing interpretation

- Added [topics/loss-floor-bouncing.md](topics/loss-floor-bouncing.md) after
  Jason asked whether WL4's low-floor loss bounce is a documented AlphaZero
  phenomenon or a bug.
- W&B pull for live run `44cxzc9d` shows the K=2→K=0 transition at e1537:
  loss/total bumped from ~1.2-1.4 to ~1.5, then fell to a new floor near
  0.4-0.8 while plies regrew and fixed external baselines stayed broadly
  strong/noisy. That shape is not the WL3 NaN bug signature.
- Filed the working interpretation: policy loss is cross-entropy against a
  moving MCTS visit distribution, so small-scale AZ can show "bump, absorb,
  lower floor" cycles when self-play discovers new lines. Treat the bounce as
  healthy unless paired with NaN/Inf, worker death, replay-buffer poisoning,
  short-game collapse, or sustained multi-window external regression.
- Updated [index.md](index.md) and appended the WL4 evidence note to
  [../TRAINING_WIKI.md](../TRAINING_WIKI.md).

## [2026-05-21] synthesis | source-backed next-run lessons

- Extended [topics/loss-floor-bouncing.md](topics/loss-floor-bouncing.md) with
  the web/literature takeaways from AlphaZero, AlphaGo Zero, KataGo,
  Go-Exploit, small-game hyperparameter work, MCTS-policy-imitation work, and
  the Tablut AlphaZero reproduction.
- Main next-run lesson: do not optimize for a pretty loss curve; instrument the
  moving teacher. Add a frozen validation archive and split policy loss into
  target entropy plus KL so we can distinguish true regression from MCTS target
  distribution movement.
- Behavioral recommendation if WL4 needs another lever: prefer 10-25%
  archive-start self-play from curated trouble/long-defense/high-KL states over
  reintroducing permanent random openings. Keep most games canonical K=0.
- Appended the action-oriented summary to
  [../TRAINING_WIKI.md](../TRAINING_WIKI.md) under the WL4 loss-floor section.

## [2026-05-21] notebook | WL4 plateau-end at e4024 — best WL series outcome to date

- WL4 (wandb `44cxzc9d`, K=0 from e1537) stopped cleanly at e4024 after
  ~5h 39min wall. Reached the "healthy lower-floor-bouncing" plateau
  described in [topics/loss-floor-bouncing.md](topics/loss-floor-bouncing.md).
- **Best WL-series outcome by every measure**: elo ATH=1841 at e2401
  (123 above Z's lifetime peak), la4=100% at e3148, la2 sustained 100%,
  plies past Z's e5000 endpoint (peak 63.8 at e2960), 0 NaN/crashes in
  5h 39min.
- Validated: random opening diversity is necessary but not permanent
  training infrastructure. K=2 (WL3.1) built diverse representations;
  K=0 (WL4) confirmed they persist AND unlocked canonical-line depth
  that K=2 was rate-limiting.
- Refuted: "diversity is permanent training infrastructure" hypothesis.
  WL4 with K=0 didn't regress toward attack-only.
- Full run-end summary in
  [../TRAINING_WIKI.md](../TRAINING_WIKI.md) "WL4 plateau-end" entry.
- Run artifacts preserved at
  `sweep_runs/WL4-no-random-openings.plateau-e4024/` (incl. latest.pt
  with full buffer for any future resume).
- Next-run shape (WL5) NOT auto-launched. Per the article's
  "Candidate Next-Run Shape" section, the next experiment is
  diagnostics-first (fixed validation archive, H/KL split, per-color
  metrics) then archive-start diversity (10-25% from curated trouble
  states). Needs design conversation + code work — not a one-shot
  parameter tweak.

## [2026-05-21] design | WL5 design landed — diagnostics + Go-Exploit archive-start

- Filed [topics/wl5-diagnostics-archive-start-design.md](topics/wl5-diagnostics-archive-start-design.md)
  capturing the next-run shape from
  [topics/loss-floor-bouncing.md](topics/loss-floor-bouncing.md)
  "Candidate Next-Run Shape" section.
- Design choices (post-Jason ACK 2026-05-21):
  - **Single run** with both diagnostics + archive-start lever (not
    two sequential runs).
  - **Static archive** mined from WL4 artifacts (heuristic-loss /
    lookahead-loss / high-KL / long-defense / canonical-opening
    positions, target ~1000-2000 total).
  - **Resume from WL4 e4024** (continues the WL series; new wandb run id
    for clean charts).
- Three diagnostic streams:
  1. Fixed validation archive, scored every eval cycle, per-provenance
     breakdown (val/policy_ce/heuristic_loss etc.)
  2. Policy CE decomposition into H(pi_mcts) + KL(pi_mcts || p_net)
  3. Per-color and per-ply-bucket loss metrics
- One behavioral lever: archive-start (15% of self-play games begin
  from a random archive position; 85% canonical empty board).
- Implementation surface: ~640 LoC + tests. Worth parallelizing
  across 2-3 agents per the wave-of-lockstep / WL2 launch pattern.
- Indexed from [index.md](index.md) Start Here.
- Next session: implementation (archive-mining script, trainer
  instrumentation, buffer side+ply tagging, worker archive-start,
  WL5 cell wiring, smoke).

## [2026-05-21] perf | eval-only Conv+BatchNorm fusion

- Sampled a live WL5 worker on the M5 Max and found the post-native hot stack
  now flows mostly through `_mcts_native.c:call_evaluator` into PyTorch/MPS
  BatchNorm / graph execution rather than C tree traversal.
- Added `fuse_model_for_inference(model)` and routed eval-only checkpoint
  loads through it for self-play workers, eval workers, match/CLI/web play,
  and the perf microbench. Trainer models remain unfused.
- Direct small-model MPS forward timing under live WL5 load roughly halved for
  batch sizes 8-128; full generator benches were contention-noisy because WL5
  was actively running.
- Appended the evidence and verification receipt to
  [../TRAINING_WIKI.md](../TRAINING_WIKI.md) and updated
  [topics/mcts-perf-ceiling.md](topics/mcts-perf-ceiling.md).

## [2026-05-22] design | three-engine Apple Silicon perf split

- Expanded [topics/ane-int8-inference.md](topics/ane-int8-inference.md) from
  a narrow ANE INT8 conversion task into a three-engine pipeline design:
  self-play leaf eval on ANE/Core ML, trainer forward/backward on MPS GPU,
  eval sidecar/match probes on CPU via Core ML CPU-only or Accelerate/BNNS,
  and native MCTS tree work in the C extension.
- Captured the key correction to the "unified memory" intuition: the likely
  win is accelerator isolation, not assuming one PyTorch tensor object can
  pass zero-copy through PyTorch MPS, Core ML, and BNNS without runtime/layout
  boundaries.
- Added a boundary-scouting microbench before launch wiring: fixed-checkpoint
  FP16/INT8 conversion, batch latency at 8/32/64/128/256, conversion/compile/
  load timing, and an overlap test while MPS trainer work is active.
- Updated [index.md](index.md) so future perf sessions route to the engine
  partitioning idea directly, and corrected the buffer-bit-packing index
  summary to match the 48 GB M5 Max math.

## [2026-05-22] organization | task routes + WL-series lineage map

- Full wiki pass found the main organization issue was routing, not raw
  content quality: [index.md](index.md) had become a flat catalog mixing
  current state, run designs, perf pages, operations, and wiki schema.
- Reshaped [index.md](index.md) into task-oriented start routes plus grouped
  catalog sections: core, training dynamics, run designs, performance/hardware,
  and operations/use.
- Added [topics/training-run-lineage.md](topics/training-run-lineage.md) as a
  compact maintained synthesis map for Z -> WL1 -> WL5. It keeps run IDs,
  conclusions, and next links in one place without flattening
  [../TRAINING_WIKI.md](../TRAINING_WIKI.md).
- Added status notes to the WL1, WL2, and WL5 design pages so future sessions
  read them as preserved design records, then jump to the lineage map and
  notebook for results.
- Updated [topics/wiki-operating-model.md](topics/wiki-operating-model.md) to
  name route maps and lineage maps as first-class synthesis when the index
  starts carrying too much navigation load.
