# Handoff — 2026-07-03 — Perf blitz: day-1 (#112/#109) full-fidelity context + reconciliation with days 2-3

**Companion:** `handoff-2026-07-02-13x13-sound-world-negative-rails-and-wiki-refresh.md` covers days 2-3
(#113 negative result + CLOSED, the rails idea, the Fable-rationale recovery, the wiki refresh). This handoff
carries what only the day-1 session context held — the #112/#109 design internals, rejections, and measured
laws — plus a reconciliation where the day-2 record corrected day-1 claims. Where the two conflict, the
07-02 handoff's *measurements* win; this one explains the mechanism behind both.

## Goal & current status
Jason's 2026-07-01 directive: **5 short days, make the box sing** — training, gen, VCT solve, standard eval;
knobs may be consolidated; "don't worry about science." Status by end of day 2 (all merged to main, pushed):
- **#112 CLOSED** — continuous-refill + streaming generator; ONE worker replaces the 4×8 fleet
  (~3.4-4.6× vs the fleet at 9×9). `sound-world` cell rewired (n_workers=1, `--stream --concurrent-games 256`).
- **#109 CLOSED** — batched-arena VCT finisher (hybrid 20-game eval: minutes → 4.4s), loud unknown-kwarg
  guard on model specs, post-run final eval on MPS (4-baseline battery 37s → 14.3s).
- **#114 OPEN/in-progress** — kernel lane. Day-2 landed the **lanes=K multi-thread-per-board kernel**
  (1.34× solve, 1.29× gen at 13×13, merge `31a8a69`, invariant #11 added). Worktree
  `gomoku-mega-vct-bb` (branch `feat/mega-vct-bb` @ `536c832`, fully merged) is still on disk for further
  kernel work.
- **#113 CLOSED (days 2-3)** with a negative result: the cap50-terminus recipe is attack-only at 13×13 —
  the perf target for training runs has shifted to the full-game lineage / rails direction (see 07-02 handoff).
Today is day 3 of 5. Remaining perf backlog is in "Next action" and #114's attack list.

## Decisions made + rationale (day-1 fidelity; do not re-propose the rejects)
- **#112 landed as worker-consolidation + continuous refill, NOT a solver-service IPC.** The issue offered
  both; consolidation is config + a generator change, matches Jason's stateless/delegate-to-hardened-tool
  priors, and the refill insight beat the filed ÷4: per-call oracle cost is ~flat in width, so the win is
  keeping the batch full — the old lockstep shape paid **81 solve calls to finish 8 games of mean ~26 plies**
  (10.1 calls/game); at width 256 it's ~0.5 calls/game. Refill = per-game plies (`ply_of`) + seed-on-completion.
  Legacy path (`concurrent_games=0`) is byte-identical (per-game ply values coincide with the global round).
- **Streaming mode (`--stream`) accepts mid-game weight hot-reload** (in-flight games finish under the new
  net — standard continuous self-play). Deliberate semantics change under Jason's "don't preserve knobs"
  mandate. Chunks of `--games-per-batch` finished games flush via `flush_records`; per-game state is freed
  (memory flat over unbounded runs).
- **fp16 worker eval: REJECTED as a 9×9 gen lever** (measured wash, ~901→937 aug/s). Mechanism: at the new
  fat-batch shapes the evaluator is MPS **sync/dispatch-bound** (27-33µs/pos vs 5.6µs pure-forward floor at
  batch 4096), not bandwidth-bound — the wiki's 1.7×/pos was a different regime. **At 13×13 fp16 IS real**
  (evaluator 70.6→44.2s, 1.6×) **but buys no wall** while the solver binds. Don't re-propose fp16 for 9×9 gen;
  reconsider at 13×15 only after the solver stops binding.
- **CoreML/ANE evaluator: PARKED, not tried** — wiki envelope says 2-6× slower worker eval for the small net;
  its value is contention-immunity only. With the solve now hidden under search at 9×9, contention-immunity
  buys little there; at 13×13 the binding constraint is the solver, not eval contention.
- **13×13 cheap solver tricks: ALL REJECTED BY MEASUREMENT** (verdict-equality asserted, receipts in #114):
  budget ladder 10→50 = 0.83× (survivors too dense — 42.5% capped); **oracle-sorted hardness clustering =
  0.98× — this is the UPPER BOUND of ANY hardness-sort scheme, so the whole sorting family is dead**;
  tg=64/128 = 1.00×; **null-board precheck = 0.59×, refuted at 13×13 by a NEW mechanism** (the null boards
  are themselves cap-bound mid-game — the opp-has-VCT question IS the bottomless search), distinct from the
  9×9 refutation (there width was free so skipping bought nothing). #113's "re-measure precheck at 13" is
  answered: no.
- **Trainer anti-runaway: nothing needed to change** — the `sound-world` cell already runs fixed
  `--sgd-steps-per-epoch 64`, which overrides `sgd_per_position` in the trainer's priority chain, so the
  4-5× gen inflow cannot re-open the LF1 feedback loop.
- **Rewired the `sound-world` cell in place** rather than adding a variant cell (Jason: consolidate knobs).
- **DAY-2 CORRECTION, ACCEPTED (from the 07-02 handoff — do not relitigate):** #112's "3.4-4.6×" is the
  *fleet → one wide process* comparison; **streaming ≈ lockstep at EQUAL width in a single process**
  (216 vs 216 games/min oracle-on at 13×13). Both are true: refill/consolidation's win is width amortization
  vs the old 4×8 shape + tail erasure; the streaming/lockstep delta at the same width is small when games are
  long and the solver binds. Also: at 13×13 the oracle is ~91% of gen wall → the lever is the kernel (#114),
  which is exactly where day 2 went.
- **Process slip to not repeat:** I once ran `git merge` from *inside* the feature worktree (merged the branch
  into itself, pushed the branch, then removed the worktree out from under the shell). Merge from the main
  checkout: `cd ~/code/gomoku && git merge --no-ff <branch> && git push`.

## Constraints & invariants discovered (day 1, measured)
- **Width-divergence amendment to the call-cost law** (now in `wiki/topics/mcts-perf-ceiling.md`): per-call
  cap50 cost GPU-quiet = 44ms @ ~150 boards → 108ms @ 862 → 116ms @ 1,708 → 148ms @ 7,276. Intra-simdgroup
  divergence loads ~2.5× then SATURATES — past ~1k boards width rides ~free again (what refill exploits).
- **13×13 resolve census** (22 real captured veto batches, fresh small net): **48.0% solve @budget 10,
  9.5% @11-50, 42.5% CAP at 50**. No "tail" — half the batch grinds to cap; every simdgroup is saturated.
  The 9.5% bucket bounds what a cap25 flip would miss *at these conditions*.
- **Gen evaluator overhead law (9×9):** in-loop eval cost fell 186µs → 14-20µs/pos purely from refill batch
  fattening; the remaining ~3× over the 5.6µs pure floor is MPS sync/dispatch + MLX-solve contention
  (solves make concurrent MPS calls ~2-3× slower; measured 38ms vs ~13ms quiet for same-size batches).
- **`terminus_fired=0` is EXPECTED in sound-world gen** — the full-breadth veto prevents attacker-VCT
  positions from ever arising (every move that would concede a VCT is masked; all-masked ⇒ defender terminus).
  Not a bug. Games end via defender terminus (~80%) or real fives.
- **Bench numbers understate steady state**: `bench_gen_refill` runs games=2×concurrent, so ~half the
  measured wall is fill/drain ramp; production streaming sits at steady state.
- **`scripts.` imports need the repo root on sys.path** — house pattern `sys.path.insert(0, os.getcwd())`;
  console entry points (e.g. `gomoku-arena`) don't get cwd, so `gomoku/eval.py::_load_vct_solver` now has a
  `__file__`-based repo-root fallback.
- **`gen_poison_check.py` argv extended:** `<ckpt> [overlap] [seed] [concurrent]` — argv[4] > 0 exercises the
  refill path with 4× that many games. Day-1 receipts: 0/174 legacy, 0/1790 refill.
- **The arena's model-spec grammar now FAILS LOUD on unknown kwargs** (whitelist: checkpoint, sims, c_puct,
  wave, vct_finish) — the #109 silent-bare-net class of bug can't recur.

## Open questions / parked threads
- **[BLOCKING for the perf thread] What's left in #114?** lanes=K landed (1.34× solve / 1.29× gen @13×13);
  the issue's remaining items: **cap50→cap25 recall study** (semantics-gated; needs gen_poison_check + a
  leak-rate measurement; the census's 9.5% bucket is the first bound) and **veto-breadth staging at 13 with a
  leak-rate measurement** (last resort). Whether more lanes=K headroom exists (K sweep, memory layout) is
  unassessed in my context — check the day-2 wiki receipts (`536c832`) before planning.
- **[non-blocking] Trainer quick wins, still unmeasured:** per-step Python L2 loop (`train.py:432-434`, on top
  of AdamW's default weight_decay=0.01 — a double-regularization curiosity), ~10-20 `float(tensor)` host
  syncs/step (`train.py:449-486`), `buffer.shape_stats()` full-buffer scan EVERY epoch (`train.py:2722`),
  GB-scale buffer embed in `latest.pt` every `--save-buffer-every`. Payoff unknown — **measure a real
  13×13/15×15 epoch's gen/ingest/train split first** (nobody has profiled the "70s epoch" directly).
- **[non-blocking] First live streaming run:** the rewired cell has never driven a full training run. Watch
  trainer ingest at 4-5× inflow (chunk-file churn ~75 files/min at width 256 / flush 64) and
  `selfplay/plies_mean`. Note #113's negative result means a 13×13 sound-world run is NOT the next run anyway.
- **[non-blocking] 15×15 never directly profiled this blitz** — all 13×13 receipts; 15×15 adds board area
  (19×19 internal spatial) and the 128×10/96×8 net question (fp16 should matter more there).
- **[flavor] The `--fp16-eval` TQ canary** requirement stands if anyone flips it on for real gen (changes
  game distribution; poison invariant unaffected).

## Artifacts
- **main @ `8335937`**, pushed; everything green. Day-1 merges: `aab6e92` (#112), `c4c0e98` (#109),
  `e994dae` (day-1 wiki). Day-2+: `31a8a69` (#114 lanes=K), `0246d5c` (#113 negative), wiki passes
  `ee981e8`/`ff63304`/`33a1b1c`/`b77193d`/`8335937`.
- **Worktree kept:** `/Users/jason/code/gomoku-mega-vct-bb` (branch `feat/mega-vct-bb`, fully merged,
  session-logged via `worktree_session.py`). NOT mine / don't touch: `gomoku-autolab-sim`,
  `gomoku-sound-world-run` (detached), `~/.codex/worktrees/*`.
- **Key code (day 1):** `gomoku/self_play.py` (`concurrent_games`/`flush_records`/`flush_games`/
  `refresh_evaluator` on `generate_games`; `_seed_game`/`_build_record` closures; `_ply_for` helper),
  `gomoku/selfplay_worker.py` (`--concurrent-games`, `--stream` branch in `main()`),
  `gomoku/arena.py` (`NetAgent(vct_finish=)`, `vct_finish_fired` counter, spec whitelist),
  `scripts/run_sweep.py` (`sound-world` cell; `eval_cmd(device=)`; `_run_final_eval` → mps),
  `scripts/bench_gen_refill.py` (+`--precheck`), `scripts/gen_poison_check.py` (argv[4]).
- **Tests:** `tests/test_concurrent_refill.py` (10: refill, streaming flush, hot-swap, lockstep identity),
  `tests/test_arena_vct_finish.py` (5).
- **Bench data (session scratchpad, ephemeral — dies with job cleanup):**
  `~/.claude/jobs/*/scratchpad/{fresh13.pt, veto13_batches.npz, capture13.py, bench_variants13.py}` —
  the capture+variant harness is 15 min to recreate if gone; the *numbers* are preserved in #114 comments.
- **Receipts:** issues #112 (comment with full shape table), #114 (day-1 duds + census; day-2 lanes=K),
  `wiki/topics/mcts-perf-ceiling.md` (2026-07-01 refill entry + divergence amendment), `wiki/log.md`
  (2026-07-01 perf-blitz entry), `~/data/sound-world-107b/.../worker_weights.pt` (the 9×9 champion used
  for all day-1 benches).

## Next action
Read #114's current state (issue comments + `git show 536c832` day-2 wiki receipts), then run the
**cap50→cap25 recall study**: re-solve a labeled pool of cap50-proven blunder children at cap25 and measure
the miss rate (the day-1 census harness in the scratchpad — or 15 min to rebuild from `capture13.py` — gives
the batches; label with cap50 verdicts, re-solve at 25, compare). If recall ≥ ~98.5%, propose the cap25 flip
gated on `gen_poison_check` + a plies_mean watch; if not, write the refutation to #114 and move to the
trainer quick-wins measurement (profile one real 13×13 epoch first).

## Vibe snippets (paste verbatim)

Jason's kickoff (the whole directive — this is the register and the mandate):

> alright buddy. make it fast. see issues #112 and #113. read the latest wiki entry to see the latest recipe. please dig deep and do your level best to make the training and the standard eval as fast as you can figure out how to make it. kernels? yes please. more efficient mcts on the GPU? heck yes. VCT is pretty fast I think, but could it be faster? maybe! generator speedup? yes please. 9x9 does an epoch ever 4-6s. 15x15 takes ~70s per epoch. and we need a lot of epochs! the post-run evals take minutes, could take seconds. the mid-run evals? I turn them off. too expensive! I only have you for 5 short days. don't worry about science, don't worry about derbies, dont even worry about preserving all the zillion knobs and switches if you can combine stuff to speed things up. this box is not getting pushed to its limits, and that aint right!

How the working style answers it (from my day-1 report, the shape Jason engages with — pre-stated bets, settled honestly):

> **My bet, stated before measuring:** consolidated continuous-refill gen at 32 games ≥ 3× end-to-end gen throughput vs today's 4×8, with oracle share falling below 40%. […] **Bet settlement:** I predicted ≥3× gen end-to-end with oracle share falling below 40% — **won on both**: 3.4–4.6×, and at 256-wide the oracle hides entirely under search.

(Register notes the snippets show: "buddy" = peer-to-peer; negative results are wins if written down — the
day-1 report celebrated the four measured refutations as loudly as the speedups; woohoos welcome on wins.)

## Least confident survived
1. **The single-user-message problem:** this session had ONE user message (the kickoff) — every calibration
   after that was autonomous, steered by memory files and the wiki, not live feedback. A fresh instance
   should know Jason never actually reacted to the day-1 report in this context; the 07-02 handoff implies
   later sessions DID interact (e.g., the perf reframe came from his questions). If he pushes back on the
   "3.4-4.6×" framing, the reframe in Decisions is the already-agreed resolution — don't defend day-1's number.
2. **The divergence-bump mechanism is my inference, not ground truth.** The 44→108ms width bump is measured;
   the intra-simdgroup-divergence *explanation* is consistent with the oracle-sort null result and the wiki's
   lock-step-mates note, but no Metal-level profiling confirmed it. The day-2 lanes=K result (1.34×) partially
   validates it. Flagged so nobody treats the mechanism as proven.
3. **Precise cell-rewire values were judgment calls**, not optima: width 256 (not 512, which measured faster)
   chose headroom for trainer/MPS contention in a LIVE run; flush 64 balanced file churn vs freshness. Neither
   was validated in a live training run. Cheap to revisit.
4. **The 13×13 census numbers are fresh-random-net conditions** (long games, dense threats). A trained net's
   board distribution will shift the 48/9.5/42.5 split — the cap25 recall study should re-capture from a
   trained 13×13 net (the old 128×10 baselines at `~/data/swap2/...` per the 07-02 handoff) if possible.
5. **Emotional weight flattened:** the day-1 session was a genuinely joyful romp — five straight
   measure→build→measure loops with every guardrail green on first try. The schema records the outcomes;
   what it can't record is that the *pace* (bench in minutes, decide, move) is itself the method Jason hired
   for. Written at ~320k tokens but with day-1 content fully in-window — compression risk here is moderate,
   not severe.
