# Gomoku Wiki Log

Chronological record of wiki maintenance. Keep entries append-only and use a
consistent heading so future sessions can scan recent changes with simple tools.

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
