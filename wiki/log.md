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
