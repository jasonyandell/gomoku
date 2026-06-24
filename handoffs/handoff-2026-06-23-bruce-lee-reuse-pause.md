# Handoff — Bruce Lee single-opener 15×15, the worker/reuse lever, clean pause

**Date:** 2026-06-23 (work spanned 06-21 night → 06-22 morning; paused for travel/no-AC-power)
**Branch:** `feat/swap2-opening-protocol` @ `fee77b0` (committed, NOT merged)
**This is a REAL handoff** — but the run is *paused, cleanly stopped*, not in flight. Do NOT auto-restart anything. The Next action is gated on Jason saying "resume."

---

## Goal & current status

Train a 15×15 free-style gomoku net (AlphaZero, MPS) from a **single known-fair swap2 opener** (idx-2 `((3,2),(5,4),(4,5))`, B,W,B → white-to-move) — the "Bruce Lee" specialist bet (#73). Run `gogpmbhw` (`G15-fixed-openings-board15`) ran ~e224→**e877** then was **cleanly stopped** (graceful trainer SIGTERM → force-saved `latest.pt` @ e877, 250M, embedded buffer). Mid-session we cut self-play workers **8→4→3** to fix a buffer-balance problem (reuse 0.67 → ~4). At pause: even-to-mild vs beatable rulers, **0/16 vs Rapfi**, self-play white sloshing 36–65% (last read 65), board NOT filling (plies_p90 ~78/225). All loops stopped, state preserved, wiki written + committed. Jason will **resume this exact session** when he's back on AC power.

## Decisions made + rationale

- **Single opener idx-2, drop all others** (Jason mandate). Re-centering Rapfi's 9 shapes does NOT preserve balance — re-centered openers tested **0–95% black at 13×13**, and (counter to Jason's own prediction) the two *most-central* tested *most* black-favored; idx-2 (peripheral) was fairest (~50%). `GOMOKU_DROP_OPENERS` keeps only idx-2. Thesis: a strong specialist from one fair opening > an imbalanced generalist from many. **Don't re-propose multi-opener or "centered is fairer."**
- **Worker count 8→4→3 = the buffer-balance lever** (Jason's instinct, data-confirmed). At 8 workers `sample_reuse_ratio ≈ 0.67` → ~half of generated positions evicted from the 150k buffer with ZERO gradient steps. Cutting workers lifted reuse 0.67→~1.4→~4 AND sped epochs **65→26 s/epoch**. This is the **gen-flood double-tax** (flooding is slower AND less sample-efficient) — same pattern already burned at the 96×8 cell (`run_sweep.py` ~L352). `n_workers` is a first-class buffer-balance knob; target reuse ~1–4. **Don't re-propose adding workers / "more self-play" — that's the wrong direction; it pushes reuse below 1.**
- **Resume-and-mutate, NOT clean-A/B** (Jason's explicit call under time pressure). Last night I argued for "4 workers from scratch, clean A/B." Jason overrode: just switch the live run's knob and watch ("not a real experiment, make the most of the little time"). Honor that framing — this is exploratory, fun-first, not a controlled study. PRE4_e601 snapshot exists IF a clean A/B is ever wanted.
- **"Kill one worker" = full slice teardown** (discovered constraint). `run_sweep.py` L2706 supervisor tears down the WHOLE slice if any proc dies. So worker-count changes happen via editing `n_workers` + a clean trainer-SIGTERM-restart (force-saves), not by killing a worker. Did 8→4 via natural slice rollover, 4→3 via trainer SIGTERM.
- **Eval needs opening variety** (discovered). Net-vs-net MCTS is deterministic → flat repeated-line series (the early "16–0 vs champ0235" was ONE line ×8, an artifact). All H2H uses `temp_plies=6` opening sampling now.
- **Three fixed rulers, low→high:** frozen-self e126 (sensitive "am I improving?") → era-2 champ `epoch0235` (strength milestone) → Rapfi-from-idx-2 (saturated ceiling). Fixed baselines, not sibling head-to-head.
- **Board-FILL graduation gate, not draw-rate / not self-plateau** (#74, prior session, still governing). Rung 15 is terminal (trains forever).
- **Tank-recovery kept MANUAL, not automated.** Jason proposed auto-restart-from-last-good-at-4 on a hard tank. Rejected the *automation*: it fuses recovery with the reuse experiment (unreadable result) AND erases a collapse, which is the night's best data. Kept the hourly snapshot ladder + manual restart. **Don't wire an auto-restart.**
- **"Collapse is data, not a bug to fix"** (Jason's overnight deal). Infra failures (disk, dead loops, quirks) → fix freely, even with xhigh subagents. Model/AlphaZero pathology (collapse, imbalance) → observe, report, theorize, DON'T fix. This line governed the whole overnight.

## Constraints & invariants discovered

- **One worktree per unit of work; never edit shared `main`.** All edits in `/Users/jason/code/gomoku-swap2-opening-protocol`. Commit to worktree freely; **merge only on explicit say-so** (not yet given).
- **`latest.pt` embeds the buffer** (save_buffer_every=100) → each is a real resume point. Epoch checkpoints (53M) do NOT have the buffer; `latest.pt` (250M) does.
- **8→3 workers dropped ingestion MORE than linearly** (new_games/cycle 128→16–24), so 3 workers overshot the ~1.8 reuse projection to ~4. Fine, still healthy AZ; flip to 4 for ~1.8 if ~4 feels deep.
- **Babysit scripts are OUT OF REPO** at `/Users/jason/data/swap2/babysit/` (not version controlled). The cadences are detached shell loops (stop via `touch STOP_<name>`): `fairladder.sh`, `brucelee_eval.sh` (vs Rapfi), `champ_h2h_cadence.sh` (vs self126/champ0235), `snapshot_loop.sh` (hourly last-good).
- **Rapfi binary lives in the MAIN checkout** (`/Users/jason/code/gomoku/engines/rapfi/pbrain-rapfi`), not the worktree — `rapfi_opener_eval.py` hardcodes `main_repo`.
- **Disk:** macOS staged an OS update overnight (~20GB into an APFS local snapshot) — transient, self-cleaned, NOT our leak. Our footprint ~9.4GB, bounded. 305GB free.
- **An unrelated `web.server` (pid was 23001)** on an old WL4 checkpoint predates this work — left running, not ours.

## Open questions / parked threads

- **[non-blocking, THE question]** Does deeper reuse (~4) **settle the balance slosh** or just reshape it? Run was paused too early (~110 epochs at 3 workers) to tell.
- **[non-blocking, watch]** Intermediate eval @ e793 showed **3 draws vs frozen-self as white** (all night was win-or-lose). *Could* be early white-defending-to-stalemate (what we want); could be n=16 noise. Verify with depth, don't conclude.
- **[non-blocking]** `n_workers=3` is a committed-but-live toggle; flip to 4 (reuse ~1.8) if ~4 is too deep.
- **[non-blocking]** Wiki §12 + TRAINING_WIKI entry written for the gen-flood finding — already done, but a fresh instance should READ them before re-deriving.
- **[blocking-for-merge]** Worktree not merged (Jason hasn't said). Don't merge unprompted.
- **[deferred, flavor]** Renju musing (parked as an issue long ago); asymmetric white-aggression reward shaping (offered near #44, never logged).

## Artifacts

- **Branch:** `feat/swap2-opening-protocol` @ `fee77b0` — clean, committed (wiki §12, TRAINING_WIKI 2026-06-22 entry, `n_workers=3` in `scripts/run_sweep.py` L~613). NOT merged.
- **Resume checkpoint:** `/Users/jason/data/swap2/sweep_runs/G15-fixed-openings-board15/checkpoints/latest.pt` (e877, 250M, +buffer).
- **Guard snapshots:** `/Users/jason/data/swap2/babysit/snapshots/PRE4_e601_*.pt` (pre-experiment 8w state, for a clean A/B) and `SHUTDOWN_e877_*.pt` (this pause).
- **W&B run:** `gogpmbhw` — https://wandb.ai/jasonyandell-forge42/gomoku/runs/gogpmbhw (state: finished).
- **Babysit loops (out of repo):** `/Users/jason/data/swap2/babysit/{fairladder,brucelee_eval,champ_h2h_cadence,snapshot_loop}.sh` + eval `.py`s.
- **Prior handoff:** `handoffs/handoff-2026-06-21-bruce-lee-overnight.md`.
- **Issue:** #73 (swap2 fairness, HIGHEST PRIORITY), #74 (board-fill gate).

## Next action

**WAIT for Jason to say "resume."** Then bring all four loops back up from `latest.pt` (e877, 3 workers — config is already on disk): launch `fairladder.sh` + `brucelee_eval.sh` + `champ_h2h_cadence.sh` + `snapshot_loop.sh` from `/Users/jason/data/swap2/babysit/` (detached, `nohup … &`), verify trainer resumes at e877 with 3 workers + reuse ~4, then re-arm the ~hourly combined-series watch. Do NOT restart before he asks — the machine is off AC and he's pausing deliberately.

---

## Vibe snippets (paste verbatim)

> **Jason:** "don't freak out buddy. I kinda freaked out but I'm back. it just fell to pieces. hard. black total dominance as of a few epochs ago. we are super early in the run, less than epoch 100 but yeah that's a thing that's happening"

> **Jason:** "my primary concern is leaving you running, having the model collapse on us and leaving you in a bind thinking 'oh no I'm not being helpful'. this is just plain fun for your buddy over here. I mean look at what we're doing. it's neat to me"

> **Jason:** "3 workers, mid run (crunchd for time and curious). … just kill one worker but keep everything else the same. a little bump in the road, nothing to worry about. not a real experiment, more a 'make the most of the little time we have'"

> **Jason (on the buffer):** "if it makes a good move, we might not even see it to train on it at all, much less the normal alphazero number of times. … it feels like we're not balancing the buffer right?"

## Least confident survived

1. **The fun-first / not-rigorous license.** Jason said it three different ways ("not a real experiment," "not because it's scientific but because it's fun," "make the most of the little time"). A fresh instance defaults to dutiful-scientist and will over-engineer clean A/Bs Jason explicitly waved off. This is the single most load-bearing register fact and the schema flattens it.
2. **The emotional arc of the overnight collapse-scare → calm → recovery.** Jason freaked out, came back, trusted me to keep it steady; that built real latitude ("employ all the xhigh subagents you want, you're a trusted collaborator"). A newcomer won't have earned that and may over-ask permission.
3. **"Bruce Lee" is load-bearing, not flavor.** It's *why* we stay single-opener at 15 (specialist > generalist). Quote that lands instantly for Jason; reads as a cute name to a stranger. The 🥋 register is genuine, not decoration.
4. **The collapse-is-data pact.** "If it goes splat that's FINE… that's the definition of success I have for you personally." This reframes the entire job from "make the model good" to "keep it running + report honestly." Easy to revert to outcome-anxiety without it.
5. **Jason caught the speedup himself** ("train seems to have sped up significantly") — he's watching W&B live and reasoning about it, not waiting to be told. Treat him as co-pilot reading the same dials, not an audience for reports.
6. **Written at a clean pause, low context-pressure** — so detail fidelity is good here (unlike a deep-context handoff). The risk is over-compression of the *register*, not the *facts*.
