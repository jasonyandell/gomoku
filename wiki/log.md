# Gomoku Wiki Log

Chronological record of wiki maintenance. Keep entries append-only and use a
consistent heading so future sessions can scan recent changes with simple tools.

Older eras: [2026-05 archive](_archive/log-2026-05.md) (rotated out 2026-07-04).

## [2026-07-04] The great curation: story layer + curation playbook + settled-verdicts-first sweep

Whole-wiki curation pass (worktree `feat/wiki-curation`, 14-agent inventory +
10 execution work-packages). What changed structurally:

- **New layer: [story.md](story.md)** — the narrative arc (prologue → 9 chapters
  → epilogue), above the timeline, below the index. Update at era boundaries.
- **New playbook: [curation.md](curation.md)** — THE curation instructions
  (layer map, status-banner vocabulary, settled-verdict-first, tell-once,
  rotation thresholds, lint). "Curate this into the wiki" = follow that page.
- **Settled-verdicts-first restructures** of the big chronicles:
  alphazero-lessons (1256→741 lines, verbatim yardstick saga → archive),
  white-side-defense-plan (695→83, full chronicle → archive), swap2 (durable
  syntheses hoisted, superseded §6.x → archive), coreml-design-envelope
  (DORMANT, lane catalog → archive), idea-pile (graduated ideas → verdict+link).
  The white-defense theorem is now told ONCE (lessons §15) + two pointers.
- **Status banners everywhere**: every touched topic page now opens with
  LIVE / HISTORICAL / SUPERSEDED-BY / DORMANT / DESIGN-NEVER-BUILT /
  DEAD-END(lesson kept), dated. Missing stopped-derby banners added (charter,
  session-runbook, sliding-v2, engine-panel, autolab pages).
- **Rotations**: log.md May era → [_archive/log-2026-05.md](_archive/log-2026-05.md);
  research-board v1–v6 verdicts → _archive; gpu-queue completed ledger →
  _archive; perf-log ARCHIVED-IN-PLACE banner; open-notes linked from
  experiment-ledger (orphan fix).
- **Merges/archives**: 15x15-era-feasibility → merged into
  15x15-training-campaign; wl2/wl5 designs → _archive with new
  [wl-era.md](topics/wl-era.md) index; sliding-derby v1 → reuse-ledger only.
- **Safety fix**: worktree-hygiene no longer documents the retired janitor in
  present tense (re-summon hazard closed; full design archived).
- **Link/integration fixes**: batched-eval-arena wired in (was zero-linked),
  m5/capabilities/experiments hub rows, broken links fixed, lineage extended to
  the sound-world era, playing-the-model → `uv run gomoku-web`.

Nothing was deleted: every cut moved verbatim to `_archive/` with pointers both
ways. Live wiki (ex-archive) ~1.99 MB; archive ~480 KB.

## [2026-07-03] Perf-blitz findings integrated into the restructured wiki + cap25 DECISION landed

Folded the 2026-07-01→03 perf blitz (#112 / #109 / #114 / #115) into the
post-restructure synthesis pages — the findings had lived only as appended log
entries + one `mcts-perf-ceiling` section written BEFORE the hub-of-hubs
restructure. Synthesis-only pass: no evidence pages touched (issues / receipts /
`TRAINING_WIKI.md` stay canonical); dated corrections, never rewrites.

- **cap25 flip DECIDED + LANDED — correction to the "human-gated" fan-out-wrap entry
  below.** Jason approved: *"cap25 is large savings and minimal cost. I'll happily
  pay 2% gap on the high end in order to get the speedup."* Shipped as a flat
  `--vct-terminus-budget` 50→25 on the `sound-world` cell (commit `8e2d9e1`, merge
  `09c067b`, `Closes #114`); eval-time finisher stays cap50. Recall of cap50-proven
  vetoes at cap25 = 99.93% (13×13 sound-world net) / 99.39% (13×13 full-game) /
  98.64% (9×9 champ); ~1.98× solve; poison@25 0/174.
- **Pages brought current:**
  [sound-world-recipe.md](topics/sound-world-recipe.md) (terminus-budget lever →
  cap25 + new § Oracle budget; the 13×13-perf "open edge" marked RESOLVED with the
  streaming≈lockstep correction; finisher cross-link);
  [mega-vct-solver.md](topics/mega-vct-solver.md) (§5.3 `lanes=K` verdict COMPLETE —
  recommend K=16 at 13×13, synthetic-K16 regression, shared-stack rejected, `--synth`
  self-check, default-OFF; new §5.6 cap25 flip);
  [mcts-perf-ceiling.md](topics/mcts-perf-ceiling.md) (new 2026-07-03 section closing
  the "measured next levers" list — all three resolved — + the day-2
  streaming≈lockstep correction to #112's 3.4–4.6×);
  [batched-eval-arena.md](topics/batched-eval-arena.md) (new § batched VCT finisher
  #109 — hybrid 15-0-5 in 4.4 s vs the legacy "minutes" 14-0-6; loud-unknown-kwarg
  guard; MPS final eval 37→14.3 s; `vct_finish` dropped from the "not supported"
  list); [training-run-reference.md](topics/training-run-reference.md)
  (`--shape-stats-every` row; the `--vct-terminus-budget` flag row + `sound-world`
  cell → cap25; #115 trainer-step perf note); hub touches on
  [index.md](index.md) + [m5-mainframe.md](m5-mainframe.md).
- **Receipt-verified corrections applied** (a subagent cross-checked the
  #112/#109/#114/#115 receipts + commits before integration): lanes=K recommended
  value is **K=16** at 13×13 (not the 1.34× K8 headline — K16=1.36×≥K8 on real
  batches); the flip is a **flat cap25 + manual per-run override**, NOT
  board-size-conditional logic; lanes=K stays **default-OFF** (only cap25 is live in
  the cell); dropped the unreceipted "`_foreach_norm` benched-and-lost" claim (the
  merged #115 code documents only `_foreach_pow` vs the old Python loop).

## [2026-07-03] reference.md touch-up: three cosmetic nits

Fixed a malformed Evals row (dropped a duplicate `reliable-eval-set.md` link
prefixed onto the `probe-100pct.md` row), surfaced
[branch-and-worktree-workflow.md](topics/branch-and-worktree-workflow.md) and
[worktree-hygiene.md](topics/worktree-hygiene.md) into the "Capabilities &
conventions" table (previously only reachable via the full page index), and
corrected the backwards "memories-also-go-to-wiki" blurb to
"memory-narrow-project-knowledge-to-wiki" to match the actual doctrine in
[conventions.md](topics/conventions.md) § What belongs in memory vs the wiki.

## [2026-07-02] Curation pass: fact-correctness audit + fix (28 pages)

After the hub-of-hubs restructure, ran a **content-correctness curation pass** (the
higher-level pass Jason deferred until the facts were organized). Deterministic
link/orphan check first: **live pages are clean** (the 267 "broken" links are all
inside the frozen `_archive/` snapshot using root-relative paths; the 5 orphans are
dated `ops/open-notes/` receipts). Then **four read-only auditors** (one per hub:
Experiments · Derby · M5-mainframe · Reference/Ops) hunted contradictions, stale-as-live
mechanisms, and cross-page number disagreements against the canonical current-state
facts. Fixed **~33 real defects across 28 pages** via four in-place editor agents (diffs
reviewed before commit). Highlights:

- **Retired-mechanism cleanup (highest severity).** `reclaim_worktrees.py` (retired
  2026-07-01) was still written in as a *session-start* action in
  [research-lab-charter.md](topics/research-lab-charter.md) with the exact
  "safe while others live" claim that failed — replaced with the manual ps-check
  procedure; also de-listed it in autolab-architecture, event-log, cockpit-vs-autopilot,
  workflow-orchestration. The CPU `gomoku/vcf` solver (retired 2026-06-27 as a
  cross-check) marked retired wherever it read as a live oracle (vct-backward-mining,
  vct-reachability-mining).
- **Era-shift staleness.** Perf pages predating (a) the torch-2.11 **fp16 reversal** and
  (b) the **oracle-veto era** (veto ≈ 91% of 13×13 gen wall) got regime/era notes
  (mcts-perf-ceiling, activity-monitor-perf-runbook); the refuted **ANE-as-inference**
  lever marked superseded (m5-max-as-mainframe, m5-max-cross-engine-coupling).
- **Stale numbers reconciled.** VCT enabling-shape count 63k → **200,242** (run-len 15→17,
  at `~/data/vct_shapes/`) across shape-library-engine / vct-backward-mining /
  vct-mining-research (and "move-labeled" → *unlabeled*); Bruce net **~3.3M → ~3.05M**
  base; native ext board sizes "9 and 15" → **9,11,13,15**; engine-catalog Elo pool
  disambiguated vs the internal anchors.
- **Derby/autolab framing.** The stopped derby was written in present tense in several
  pages → historical banners; v1 sliding-derby banner-superseded by v2; broken cross-refs
  fixed (§10 → `sliding_gate.py`; `perf-lab-charter.md` → `research-lab-charter.md`; v2's
  predecessor filename). The **two competing autolab architectures** (launchd-daemon vs
  Claude-workflow composite) got a neutral "two explored approaches, neither supersedes"
  note — deliberately NOT crowning a canonical winner (Jason's design call).
- **Hub-index completeness.** experiments.md (6→8 pages) and reference.md (19→23) full-page
  indexes filled in; sound-world-recipe "Known open edges" cross-pointed to the #113 negative.

Method: fan-out audit → triage → fan-out fix → diff review → single commit. Facts only;
no synthesis/re-ranking (that's a later pass).

## [2026-07-02] Major restructure: hub-of-hubs index + revived Ops hub + curated timeline

Rebuilt the landing page from a 26,680-token wall (it overflowed a single 25k
fetch) into a **~950-token hub-of-hubs** that fits one fetch with headroom. The
old index linked ~90 pages inline, with 400–900-word essay-cells; **14 of the 83
topic pages were orphans** (on disk, never linked). New shape:

- **Pinned top = the [Ops hub](ops.md)** (replaces the "common workflows" header):
  Train/Eval/Publish workflow pages + the live operating surfaces, each verified
  live-vs-archived (2026-07-02). Banner-flagged the two dead ops pages
  ([status.md](ops/status.md) self-superseded, [frontier.md](ops/frontier.md)
  retired pi-mechanism).
- **5 knowledge hubs** — [AlphaZero](alphazero.md), [Experiments](experiments.md)
  (hub-of-hubs; [Seek-VCT](seek-vct.md) nested), [Derby](derby.md),
  [M5-as-Mainframe](m5-mainframe.md), [Reference](reference.md) — each with a
  start→now→learned skeleton and a **complete "every page in this hub" index**, so
  **all 99 pages are reachable from exactly one hub (zero orphans)**.
- **New [training-timeline.md](training-timeline.md)** — the append-only
  TRAINING_WIKI (5,799 lines) broken into a ~50-milestone era-grouped index with
  run ids (curated via a full-notebook extraction pass).
- Provenance + reducer state preserved in `_archive/` (old index +
  `manifest.json`, the per-page hub/verdict map that drives future curation passes).
- New workflow pages: [train-a-model](train-a-model.md), [eval-a-model](eval-a-model.md),
  [publish-a-model](publish-a-model.md) (the real HuggingFace mechanism from `gomoku/hf.py`).

Built with three ultracode passes (inventory → coverage+manifest → extraction+cutover).
Fixed a pre-existing broken link in this log (`alphazero-lessons` → `-15x15-gomoku`).

## [2026-07-01] New page: the VCT-defense aux head — a working sensor with no actuator (#103)

Created [topics/vct-defense-aux-head-result.md](topics/vct-defense-aux-head-result.md) — the outcome of #103,
executing the #102 supervised VCT-defense aux-head design. Ran TWO experiments; **both learn the defensive
REPRESENTATION but fail to make the POLICY defend.** **A** (from-scratch 9×9, wandb `8mtowemb`, e1152):
`train/vct_loss` 0.60→0.03, mask_frac ~0.9, but `plies_mean` flat ~9-10 for 1152 epochs = the #101 attractor
*unchanged even with the representation present*. **B** (Bruce/idx-2 pivot, wandb `zrjfwny2`, e862): warm-start
the 128×10 champion + layer the head via the new `force_aux_vct` splice + restrict self-play to the idx-2 wound;
the head learns (0.52→0.026) but the self-play policy drifts (`loss/policy` 1.93→2.62, plies 11.6→9.6). Kept the
honesty bar high on the **eval-saturation nuance**: the idx-2 gate (n=48, sims=160) reads **0/48 on the pivot AND
0/48 on frozen Bruce** — saturated, so it does NOT show the pivot degraded Bruce; the real "fell apart" evidence
is the policy drift, and a clean strength-delta was never measured (abandoned). Verdict: **a working sensor with
no actuator** ("Frankenstein + aux head is not the recipe" — Jason); next = target the policy directly. Added an
index ⭐ row (after the #100/#101 terminus row), a TRAINING_WIKI 2026-07-01 (#103) entry, and an idea-pile #11
aux-head-sequel marker. Documentation only; the head/splice code was already merged to main (6ba92b5).

## [2026-06-26] New page: mining VCT-reachability from the Rapfi corpus (off-path fan + knife-edge)

Created [topics/vct-reachability-mining.md](topics/vct-reachability-mining.md) — cheap ways to mine
seek-VCT steering signal from the 500k Rapfi-v-Rapfi games, with a **thesis update** as the headline.
The off-path fan (ride a game; at pre-onset non-VCT nodes fan the moves a side did NOT play; solve
VCT) revealed: (1) **the pre-onset band is a knife-edge** — ~80% of alternative moves lose by force
(~99% one ply before onset; ~half even 6 plies out), so the band we assumed was the net's *forgiving*
steering region is **not** approximation-tolerant — sharpness ramps *before* the VCT, the net/solver
boundary is fuzzy + earlier than assumed (the most valuable finding). (2) **Framing, code-verified:**
a VCT belongs to the side-to-move, so fanning S's alternatives finds the **opponent's** forced wins
= S's losing moves → the fan is a **defense/blunder + VCT-board miner, never an offense detector**;
integrity check 0.000% of fanned nodes are VCT. (3) **Triviality split** (VCF kernel on the VCT-wins):
of the 81% fanned VCT-wins, **96.1% are trivial VCF** (four-blocks), only **3.5% non-VCF VCT** (need a
*three* = combinational molecules) — and the gold concentrates on the **winner's** wins (defender
perturbed; combinations belong to the side with initiative). Harvest plan = perturb the *defender*
for 100k+ non-VCF VCT boards = offense termini + defense lessons (the white-defense wound). Also banks
the **free distance-to-VCT field** (terminal-VCT 99%, multi-window 11.6%, offense coverage 49%, an
upper-bound/censored target) and the proposed Φ=γ^dist potential. Banked negatives: both yield
predictions wrong (81% VCT / 5% cap, not cap-dominated); 81% looks rich but is mostly trivial; a
"VCT-where-one-existed" alarm was a labeling confusion (all fanned nodes are pre-onset). Reusable
script `scripts/threat_shapes/vct_fan.py`; reused the GPU VCF+VCT kernels (no CPU solver). Added an
index row + cross-linked the seeker row. On `feat/gentle-rapfi-teacher`; not merged.

## [2026-06-26] New page: seeker steering learnability (seek-VCT thesis, Phase A)

Created [topics/seeker-steering-learnability.md](topics/seeker-steering-learnability.md) — the
**steering** half of the seek-VCT thesis (the recognizer page named the seeker as attention's real
audition). One question: can a net imitate the **quiet-phase (pre-onset) moves of the side that
reaches the first forced VCT**, and generalize to **unseen games**? **Yes** — held-out,
shard-disjoint **top-1 0.386 / top-5 0.696** (matching the *exact* strong-engine move) vs
adjacency-to-stones 0.025/0.121 vs random-legal 0.005/0.023 ⇒ the steering signal is learnable and
real. As with recognition, a **CNN (224k) beats attention (339k)** at *next-move* imitation (top-1
0.386 vs 0.263) — **but** attention was still climbing at the epoch cap (undertrained, not capped),
and next-move BC is *local*, so this does **NOT** settle attention's global-receptive-field bet for
*sequential* seeking (that's Phase C). Honest framing recorded: top-1 match is a **weak proxy** (≠
strong play; conflates seeking with general engine strength). onset/labels reused from the miner
(`onset = first win&~cap` ply), no re-solve; 500,747 examples / 38,927 onset games; split by shard
(367/33, overlap 0); **0 frame mismatches**. Code: `scripts/threat_shapes/{gen_seeker_dataset,train_seeker}.py`;
artifacts `~/data/puzzle_miner/seeker_exp/`. Added an index row + linked the recognizer row to it.
Next (gated): **Phase B** oracle-labeled VCT-reachability target, **Phase C** hybrid-play eval
(oracle every ply + net steering) vs a fixed baseline. On `feat/gentle-rapfi-teacher`; not merged.

## [2026-06-26] New page: is-VCT recognition learnability + move-extraction gap RESOLVED

Created [topics/vct-recognition-learnability.md](topics/vct-recognition-learnability.md) — the
first learnability probe on the VCT labels. A net **can** classify "side-to-move has a forced
VCT" on **held-out, shard-disjoint** games (AUROC 0.92+, real generalization), but for these
*local, translation-equivariant* shapes a **CNN (0.971, 168k params) beats a transformer
(0.924, 339k)** and even **logreg-on-counts (0.946) beats attention** ⇒ recognition is easy +
count-dominated, leave it to the exact oracle; **attention's bet is the SEEKER, not the
recognizer**. Records Jason's **seek-VCT thesis** (learn the approximation-tolerant steering,
solve the approximation-intolerant forcing finish — anti-correlated tractability). Methodology
note for posterity: labels reused from the miner (**absence = proven no-VCT**, Jason's
correction), no re-solve; split **by shard** (367 train / 33 test, overlap 0) so test games are
unseen — the load-bearing guard against position-level leakage.

Also marked [topics/vct-backward-mining.md](topics/vct-backward-mining.md) **§5 RESOLVED**: the
verdict-only "catalyst-move extraction" gap is closed by **option 2** — a passive **GPU
root-move output** on the megakernel (`solve_vct_mega_bb(return_move=True)`, no extra nodes);
2.38M forward puzzles move-labeled (`solutions.jsonl.gz`), 400/400 moves independently verified.
Updated both index rows (vct-backward "OPEN"→"RESOLVED"; added a recognition-learnability
doorway). Code on `feat/gentle-rapfi-teacher`; not yet merged.

## [2026-06-26] New page: the Shape-Library Engine (the gomoku-AI plan) + gpu-vct row corrected

Created [topics/shape-library-engine.md](topics/shape-library-engine.md) — the plan outline
for **the gomoku AI Jason wants to build** (2026-06-26 brainstorm): mine the first-VCT
*enabling shape* → reduce to its **minimal full-board prime implicant** (VCT-win is monotone
in freestyle ⇒ 3 cell roles, "blank" collapses into defender-forbidden, validation = one
solver call, extraction = batched ablation on L0) → a library = **the monotone DNF of "you
have a VCT"** (bitmask-matched, D4-deduped) → a **two-player pursuit / df-pn** player that
seeks an un-blockable **fork** of shapes and denies the opponent theirs, **leaf-verified by
L0 so it never hallucinates a win**; **L2 = the AlphaZero layer** regressing the
shape-reachability potential into the fog on *verifiable* targets. Captures Jason's binding
working principles (go-all-the-way / no safe half-steps / negative-result-welcome;
full-board for soundness not locality; forks day-1; telemetry rides along, not a gate). Sits
on the **already-built** L0 ([gpu-vct-feasibility.md](topics/gpu-vct-feasibility.md) §8) and
the **63k banked enabling shapes** ([vct-backward-mining.md](topics/vct-backward-mining.md));
added the forward pointer from that page (the shapes are L1's raw material) and an index
doorway row. Also **corrected the stale index row** for gpu-vct-feasibility — it still read
"correct but CPU-bound v0," now reflects that **§8 overturned it** (on-device bitboard
megakernel, ~1600× CPU, 0 FP/FN).

## [2026-06-25] New page: the Idea Pile (autolab seed bank) + idx-2 DAgger entry

Created [topics/idea-pile.md](topics/idea-pile.md) — a parking lot for wild-but-grounded
research directions (bigger swings than a derby cell), each with mechanism / rough cost /
how-we'd-know, framed as seeds the incoming autolab can each give a shot. Seeded with 8 from
a 2026-06-25 brainstorm (out-search-and-distill, solve-idx-2, blind-spot judo, the think-time
ladder as a self-paced metric, move→value oracle, museum/disagreement dojo, + two moonshots).
Added an index doorway row. Companion evidence: `TRAINING_WIKI` 2026-06-25 (the DAgger loop
built + the history-mismatch bug found/fixed).

## [2026-06-19] autolab WENT LIVE → 15×15 era → first 15×15 champion (#64/#65/#67, epic #53)

The night the self-driving lab proved itself, then pivoted to 15×15 — synthesized
across `TRAINING_WIKI.md` (evidence, two new dated entries) and three synthesis pages.
**Arc:** autolab launched (#64) → ran **6 real 9×9 slices** unattended (crowned the
first champion `9x9-champ-recipe@0`, 0 failures) → **pivoted to 15×15** (#65: board
size is a process-start `GOMOKU_BOARD_SIZE` constant; trainer threads it into the
`run_sweep` env + `autolab up --board-size` bakes it into both the train **and**
arena plists; HF `champion` tag reset for the new era) → ran the whole loop again from
scratch (lane `15x15-wdl`, cell `G15-wdl` = v8 + WDL head, no warm-start, no teacher)
→ **crowned the first 15×15 champion `15x15-wdl@0` at internal elo 1918** (not
comparable to the 9×9 scale). Updates: (1) `TRAINING_WIKI.md` — new 2026-06-19 entry
"15×15 era: first self-driving 15×15 run" (the #67 arena artifact-ref bug + ledger-
correction recovery, the #65 pivot, the science, the first champion). (2)
[topics/15x15-training-campaign.md](topics/15x15-training-campaign.md) — **DATED
CORRECTION**: the from-scratch run went **through** the cold-start fast-attack collapse
(plies 69.5→9.2) and **self-recovered** to ~35-40 plies with **no teacher / no
warm-start** (WDL value-loss held ~0.81-0.89 = healthy maturation, not the death-tell)
→ cold-start collapse is a **survivable transient** for v8+WDL, the warm-start
"remedy" may not be strictly required; preserves the original 2026-06-13 conclusion,
annotates it. Decisive open probe: this net's **white W-L-D vs Rapfi** (recipe-deep
deficit vs warm-start attacker-bias). (3)
[topics/autolab-architecture.md](topics/autolab-architecture.md) — 15×15-capability
section, P3/P4 phases marked **LIVE**, the **#67 lesson** (artifact-contract scheme
mismatch survived per-side unit tests; needs an end-to-end trainer→arena smoke), and a
new **Arena-yardstick gap** section (the arena gates only *relatively*; wire
`ExternalAnchor.play()` from `eval_vs_rapfi`, pin a *measured* rapfi-100ms point not a
published Gomocup Elo, add a non-gating absolute W-L-D readout, surface Δwhite-elo/Δt).
(4) `index.md` — autolab doorway + Page Catalog row marked LIVE/15×15. Issues:
#64/#65/#67/#68, epic #53.

## [2026-06-19] autolab P5–P7 operating contract — new page: supervisor + monitor + research-lite (unattended overnight)

New canonical page [topics/autolab-supervisor-and-monitor.md](topics/autolab-supervisor-and-monitor.md):
the contract that turns the tested autolab library into a running overnight lab.
Synthesizes four P-reports into one buildable spec — (a) `~/data/autolab/` home
layout with the monitor↔research path-ownership rule (research owns `research/`,
monitor owns `monitor/`); (b) the process tree + literal launchd plist XML for
**four** jobs — `train`/`arena` (`KeepAlive{SuccessfulExit:false}`, run the
daemons directly; the flock singleton + ledger re-pick is the whole recovery
story, no parent respawn) and `monitor`/`research` (`StartInterval` 600/1800) —
with the full env table (`HOME` for HF token, `PATH` for homebrew git,
`WANDB_MODE=offline`); (c) the one-row seed — **base=scratch, cell=derby-v9-small
(fresh 9×9 v8-champion recipe), max_wall_secs=3600**, p10 seed band, ~6–7 slices
by morning, with the `GOMOKU_BOARD_SIZE`-only-9×9 rationale; (d) the monitor
digest spec (latest.md template + notify-on-change + empty-state degradation);
(e) the research-lite deterministic tick + the `priority < P_seed` starvation
guard; (f) the `autolab up`/`down` runbook incl. the attended `--no-hf`→real-push
PROD-slice proof to run BEFORE unattended launch; + a ranked risk list. Index
gets a doorway row under the autolab/ops route. Did NOT touch
`autolab-architecture.md` (the build step de-stales it).

## [2026-06-19] autolab P4 arena built (#59) — architecture page: arena section + phases

`gomoku/lab/arena.py` `ArenaRole` shipped: gates a candidate vs the HF `champion`
tag via `sliding_gate.run_gate(dry_run=True)` (PROMOTE/REVERT/AMBIGUOUS), appends
`eval`+`verdict` rows, moves the `champion` tag on PROMOTE, shrinks `n_games` when a
trainer slice is live (co-tenancy). First candidate (no champion) auto-promotes.
8 mocked tests (gate/HF/eval_fn injected — GPU-free). `gomoku-lab-arena` entry point.
Updated the Arena section (built; dry_run + HF-tag champion; Rapfi panel a logged
follow-up) + phases table (P4 DONE; P5 research next). Live gate proof deferred
(needs real models + a free box). P1–P4 of epic #53 now shipped.

## [2026-06-19] autolab-architecture.md — de-stale to the SHIPPED P2/P3 design (no-claim flock, ~/data home, run-base)

Reconciled the design page with what's built (P1 #54, P2 #56, P3 #57 all merged).
Dropped the `claim`/lease path from the shared-loop pseudocode + code-shape items
6–7 (singleton is now an OS flock that auto-frees on death — `FD_CLOEXEC` so
subprocesses can't pin it — and recovery is plain re-pick; nothing to reclaim). New
**`~/data/autolab/` home** section (ledger + `runs/<lane>/` + `worktrees/<row>` +
`daemon-<role>.lock`) with the **`~/data` buffer convention** (big artifacts local;
HF gets slimmed weights only) and the **data↔code decoupling** via
`run_sweep --run-base`/`GOMOKU_RUN_DIR` (default REPO_ROOT) + per-commit ephemeral
code worktree. Updated Locked decisions + the phases table (P1–P3 DONE). Built this
session: `gomoku/lab/{daemon,status,trainer}.py`, `hf.push_slice`, `run_sweep`
run-base, entry points `gomoku-lab-{train,status}`.

## [2026-06-18] NEW topic/autolab-architecture.md — formalize the autolab (epic #53, P1 spine #54 built)

New canonical design page for Jason's self-driving-lab spec: one external, out-of-git,
append-only ledger (`gomoku/lab/ledger.py`, financial-journal corrections, reducer +
priority-pick — built + 21 tests green this session) read by four same-shape loops
(trainer 1h-singleton-slices→HF · mac-native arena · ideate-and-wait research · GitHub
worker). Captures the ~80%-there finding (every load-bearing mechanism already exists; the
spine was the gap), the 8-point code-shape contract, the **measured** M5 co-tenancy envelope
(one heavy SGD trainer max; arena concurrent under a guard), the per-loop cockpit overlay, the
locked decisions (ledger in `~/code`; buffer local + HF slimmed; per-slice HF revision +
champion tag; 1h cap, MVP at 1-epoch), and the P1–P6 plan. **Supersedes the framing of #2**
(three-tier queue) and folds in #19. Added an index doorway row + Page Catalog entry.

## [2026-06-18] white-side-defense-plan — #43 (I2) LEVER BUILT: stamp the saving move on the policy head

Recorded that the #43 defense-teacher I2 arm is code-complete + merged (`Closes #43`). New
`vcf.vcf_refutations` primitive (the defender moves that break the opponent's forced VCF, since
the recorded side moves first — sound, re-solve-confirmed); `self_play._apply_defense_teacher_policy`
stamps a soft saving-move policy target and leaves value untouched (pure policy lever, no value
crush on truly-lost positions); `--defense-teacher-policy` worker flag; full test file incl. a
300-position soundness fuzz. Added a "#43 (I2) LEVER IS BUILT" subsection to §1B.2 with the why
(vs the failed value-only #36/#42) and the gate (re-run the Rapfi TC-tier white-column calibration
after a training slice). Remaining = the live GPU race (`needs-live-validation`).

## [2026-06-18] white-side-defense-plan + reliable-eval-set — SYNTHESIZE the TC-tier calibration + the parallel --jobs eval (#52)

Promoted the champion-vs-Rapfi TC-tier calibration (TRAINING_WIKI evidence) into synthesis.
`white-side-defense-plan.md` §1B.2: added the tier table + the two readings — **cliff** (10ms
below Rapfi's search threshold, swept 40-0) then a **white-defense plateau** (~27% flat from
100ms-1s; black competitive 40-65%, white pinned 0-15%). The deficit vs the #1 engine is 100%
white-side, now 5× confirmed → strongest mandate for #43 (Rapfi = the before/after gate).
`reliable-eval-set.md`: documented the parallel `--jobs` eval path (#52, spawn-pool, pass the
run-rapfi WRAPPER so NNUE loads; 200 games in 7.6 min).

## [2026-06-18] index + white-side-defense-plan + external-engine-baselines — SYNTHESIZE first real-Rapfi result; de-stale the "broken yardstick" reckoning

The first champion-vs-real-Rapfi result (eval502 20.8% @5s, n=24; **black 42% / white
0/12**) was recorded only in `TRAINING_WIKI.md` (evidence) — promoted it into the synthesis
layer. (1) `white-side-defense-plan.md`: new §1B.2 (2026-06-18) "STRONG-ATTACKER
MEASUREMENT ARRIVED" — real Rapfi is the harder attacker #45/#49 was reaching for; 0/12 is
the opposite of the #45-v1 floor, the cleanest #37 evidence, and validates #43 as the
target. (2) `index.md`: added a ✅ 2026-06-18 UPDATE to the 2026-06-15 RECKONING banner (the
"broken Rapfi" was the weightless build; native NNUE Rapfi is fixed + online, #40) and
de-staled the reliable-eval / panel-derby / white-defense table rows. (3)
`external-engine-baselines.md`: first-contact result line under the anchor-online section.
Cross-refs wired: white-defense ↔ external-engine-baselines ↔ #37/#43/#49 ↔ TRAINING_WIKI.

## [2026-06-18] topics | external-engine-baselines + reliable-eval-set + gomocup-engines-catalog — native Rapfi-NNUE anchor ONLINE (#40, resolves #28 under-search)

Brought the first hard EXTERNAL eval online natively (no wine, per Jason's nix
directive). Native arm64 **Rapfi-NNUE** is now a default reliable anchor in
`panel_tournament.py::_NATIVE_ENGINES`. Key finding: the #28 "Rapfi ignores its
time budget / illusory TC tiers" wound was the **weightless classical build**
(no `--config`); with the committed `engines/rapfi/config.toml` + mix9svq NNUE
weights it searches to its full budget (verified: depth 32 / 2.0M nodes / 4105ms
of a 4970ms budget / forced mate), single-threaded (Gomocup-legal). Added
`scripts/run-rapfi` wrapper (hard-errors if binary/config missing), extended
`build_rapfi.sh` to fetch the `Networks` submodule weights (sha256-verified
byte-identical; weights gitignored, config committed), and `tests/test_rapfi_native.py`
(skips when artifact absent; pins handshake + unmirrored coords via a forced-block
tactic). Still open before trusting an ABSOLUTE number: balanced openings (#22) +
measuring effective single-thread strength under our harness (#35).

## [2026-06-16] topics | workflow-orchestration § Resilience — workflows degrade, don't crash (#50)

Added a Resilience section to
[topics/workflow-orchestration.md](topics/workflow-orchestration.md): an
API-overload window (529s) crashed `implement-backlog` dereferencing `triage.picks`
on a `null` (a dead subagent). The synthesis: a workflow's resilience lives in its
deterministic JS, not its agents — bounded re-spawn (`agentTry`) for idempotent
chokepoints, graceful degradation everywhere else, and **never retry a
side-effectful agent** (the composite's train-launch — double-launch risk). Gauge:
`scripts/check_workflow_resilience.mjs` (verified red on the un-hardened code, green
after). All four `.claude/workflows/*.js` hardened.

## [2026-06-16] wiki | Memory-vs-wiki reckoning — pruned agent memory to machine+user only, promoted 37 memories into the wiki

Jason: "memories compete with the wiki." Pruned the agent's persistent memory
to **machine-only + working-with-Jason-only** (8 of 45 kept); promoted the
remaining 37 project/process/roadmap memories into the wiki, the source of
truth.

New/updated synthesis:
- **CREATED** [topics/fleet-management.md](topics/fleet-management.md).
- Appended sections to [topics/research-lab-charter.md](topics/research-lab-charter.md)
  (clean-milestone-not-stop, run-cap fast-filter, the three-tier redesign #2,
  the training-slice resume-mechanism),
  [topics/alphazero-lessons-15x15-gomoku.md](topics/alphazero-lessons-15x15-gomoku.md) (threat-semantics +
  founding decisions),
  [topics/perf-bench-vs-real-training-cost.md](topics/perf-bench-vs-real-training-cost.md)
  (plies-ETA), [topics/buffer-bit-packing.md](topics/buffer-bit-packing.md) (3M
  turnover), [topics/white-side-defense-plan.md](topics/white-side-defense-plan.md)
  (#42-failed / I2-fired update),
  [topics/launch-sequence-runbook.md](topics/launch-sequence-runbook.md)
  (overnight workhorse + narrator), [ops/gpu-queue.md](ops/gpu-queue.md)
  (gpu_daemon historical + fixed the dangling gpu-daemon.md link).
- Rewrote [topics/conventions.md](topics/conventions.md) § "Memories also go to
  the wiki" → "What belongs in memory vs the wiki": memory = machine + user
  only; project/process/roadmap knowledge is **wiki-only, not mirrored**. Same
  rule tightened in [../CLAUDE.md](../CLAUDE.md) and [../AGENTS.md](../AGENTS.md).
- Cleaned dangling "Mirrored in memory" footers for the deleted slugs.

## [2026-06-12] topics | Added 15x15-era-feasibility-and-plan: the perf ceiling was the small model, not the Mac

Filed [topics/15x15-era-feasibility-and-plan.md](_archive/topics/15x15-era-feasibility-and-plan.md) *(2026-07-04: merged into [15x15-training-campaign.md](topics/15x15-training-campaign.md); link repointed to archive)*
plus the evidence cell `scripts/bench_board_scaling.py`. Measured on the idle
M5 Max (torch 2.12.0, fp16, MPS): at the production wave=64 the champion arch
runs 15×15 for **free** (0.98×), a 96×8/1.45M-param 15×15 net costs only
**2.32×**, 128×10 costs 4.62× — direct confirmation of the dispatch-bound
regime from [topics/mcts-perf-ceiling.md](topics/mcts-perf-ceiling.md).
Envelope: a WL5-scale 1M-game 15×15 run ≈ a week of wall-clock (to be
validated by a live smoke slice per
[topics/perf-bench-vs-real-training-cost.md](topics/perf-bench-vs-real-training-cost.md)).
Plan: Phase 0 certify the v8 champion vs Rapfi at 9×9 → Phase 1 rules-variant
decision (human-gated; swap2-freestyle recommended first) → Phase 2 port →
Phase 3 smoke + sweeps + warm-start check → Phase 4 first 15×15 run (WDL head
as first new contestant, bit-packed buffer prerequisite) → Phase 5 derby +
perf-lab reopen (ANE `L09i-fix-load` first). Indexed as a new Start Here row
and under Performance And Hardware.

## [2026-06-01] topics | Added workflow-orchestration: Claude Code Workflows mapped onto the lab

Filed [topics/workflow-orchestration.md](topics/workflow-orchestration.md) and the
lab's first real workflow at `.claude/workflows/reviewer-gated-fanout.js`. The
synthesis: the Claude Code *Workflow* feature is **deterministic agent-chaining**
(distinct from the `/loop` looper) and maps exactly onto the *everything-else*
lane of the two-queue scheduler — the half that is prose today (fan-out, Reviewer
gate, worktree lifecycle). It does **not** fit the GPU lane (agents can't hold the
MPS lock; `delo_derby`/`run_sweep`/watchdog stay as-is) or cross-session crons.
Page carries the fit/misfit table and the cockpit framing: a `pipeline` whose
second stage is the Reviewer makes the verify-gate non-skippable structure rather
than skippable discipline. Indexed under Performance And Hardware (next to
conventions). Next candidate noted: an issue-runner workflow for the bead-runner
loop. Pairs with [topics/cockpit-vs-autopilot.md](topics/cockpit-vs-autopilot.md)
(still unwritten — referenced from MEMORY.md but absent from the repo).

## [2026-06-13] topics | Added alphazero-lessons-15x15-gomoku: the learning artifact

Filed [topics/alphazero-lessons-15x15-gomoku.md](topics/alphazero-lessons-15x15-gomoku.md)
— the distilled UNDERSTANDING from the 15×15 campaign, written as the primary
artifact (Jason: "the learning is the artifact"; Gomocup is not a goal). Seven
sections: representation transfer (warm-start skips the cold fast-attack
collapse), net×search multiplicative (capacity pays at deep TC), read structure
not loss (plies/vl), eval discipline (small-n noise, both-tiers, FPU negative),
systems shape learning (dispatch-bound regime, gen-flood runaway, smoke-first),
the methodology meta-lesson (warm-start→smoke→external-gate→net2net-grow→preserve),
and honest bounds (short-TC caveat). Indexed as the top "understand what we
learned" doorway. See [[feedback-learning-is-the-artifact]] memory.

## 2026-06-20 — swap2 (#72) synthesis page added
Created `topics/swap2-opening-protocol.md` — the durable synthesis of the swap2 arc
(why → thesis → architecture → what we learned → what to try next → operational
gotchas). Repointed the index doorway from the code files to the page. Headline
finding: the core bet is confirmed at the data level (white wins 27% in swap2
self-play vs ~0% empty-board); strength-vs-champion still at parity (H2H 51.6%, early).
Evidence chronology stays in TRAINING_WIKI.md (2026-06-20); this page is the synthesis.

## 2026-06-23 — sensei page synced: Rapfi HF packaging + daemon fail-fast
Updated `topics/eval-teacher-sensei.md` to match where the build landed this session:
(1) Rapfi now resolves friction-free via `rapfi_artifacts()` — local build → a
**public**, commit-SHA-pinned + sha256-verified HF snapshot (`jasonyandell/rapfi-arm64`,
a GPL mirror of dhbloo/rapfi @ 6e0a132; this project is MIT, see `THIRD_PARTY.md`),
cached machine-globally so any worktree/fresh box resolves identically (chosen over
Docker: workflow scored HF-resolver 8 vs Docker 3). (2) The daemon/teacher are
**fail-fast** — a configured rapfi ruler (the default) means run-with-Rapfi or
`SystemExit(2)` with an actionable message; no silent baseline-only fallback
(repeatable-by-default; opt out by configuring non-rapfi rulers). (3) Refreshed two
now-resolved operational notes (swap2 schema merged to main; `save_checkpoint` atomic
#76). `available()`=no-network/test-gating vs `obtainable()`=may-fetch distinction
documented.

## 2026-06-25 — capabilities synthesis layer added + spine fixed (#86)
A new session read `TRAINING_WIKI.md` and missed the entire 2026-06-25 idx-2
distillation run — root cause: the work landed in a topic page + an index row, but
the **chronological spine had no dated entry** (last was 2026-06-24), and the index's
synthesis was a *page catalog* (by doc-type), not a *capability map*. Two additive
fixes: (1) **fixed the spine** — appended the dated 2026-06-25 `TRAINING_WIKI.md`
entry (mine 1.13M @ ~700/s, soft-target pretrain, warm-start AZ to ep250, fast
eval-gradient; infra success / science inconclusive / banked), which also closes the
open "soft target untested" thread. (2) Added **`capabilities.md`** — a one-screen
"what can this repo DO" synthesis (mine · pretrain/warm-start · train · evaluate ·
search · operate), each capability → how-to + deep doc, NEW rows tagged. Linked from
the index Start-Here callout + table + read-order, plus a Maintenance Rule: **every
significant run closes the spine AND updates capabilities**, not just a topic page
(the discoverability bug this fixes). Purely additive — no existing topic deprecated.

**Follow-up same day:** refreshed the index's stale `## Current Synthesis` (it still
framed the project as 9×9 defend-vs-imitate / WL-series / Core ML). Now reads the
current 15×15 front — central question = make WHITE winnable / stand vs Rapfi-NNUE;
binding wound = white-defense gap (Bruce: black ~42% / white 0/12 @idx-2); the two
principled white fixes (swap2, fixed-fair-openings); the one-hot-harms / soft-target
distillation lesson (#77/#86); think-time-not-node strength dial; matured tooling →
`capabilities.md`. Kept the era-independent durable lessons (fast-attack collapse,
noisy evals, native-MCTS bottleneck, preserve-evidence). Scoped refresh, not a full
re-curation (status-tags / pruning still deferred).

## 2026-06-26 — external reference page: standard gomoku strategy ("rule of priorities")
Filed a new `topics/gomoku-standard-strategy.md` distilling a GomokuTV YouTube video
(*"How to play Gomoku? — The rule of priorities"*, <https://youtu.be/1boqoa2rQfU>) that
Jason flagged as a clear presentation of **standard, known** gomoku theory. Ingest was
**transcript-only** — Gemini's `analyze_youtube` was down (Google IneligibleTier auth
error on the local CLI); the spoken theory is captured, the on-screen shape diagrams are
not (geometry cross-checked against `allis-threat-theory.md`, not read off the video).
Content: the 5-depth **priority ladder** (overline/five → four/VCF → three/fukumi/VCT →
two/yobi/VC2 → sh-win/positional/"ear-reddening") + a fukumi/yobi/cut/sh-win glossary
mapping the community's terms to Allis's formalism and to `gomoku/vcf.py`. The page is
explicitly framed as the **standard-theory foil for idea #10 (molecule ⊋ line)**: the
ladder is fully line-organized through depth 4, and depth 5 ("positional / tempo /
open-area moves with no direct connection") is exactly where standard theory *gestures
at* non-line/field structure but **runs out of names** — the contrast we want. Added an
index doorway row under the Allis row; purely additive.

## 2026-06-27 — Φ distance-to-VCT field learnability (the trilogy's 3rd leg)

New topic `phi-distance-field-learnability.md` + index doorway row (after the seeker row): the first
trained L2 model. Regresses the dual proof-frontier potential Φ (offense+defense distance-to-VCT) off
the free miner verdicts; held-out shard-disjoint CNN offense ρ=0.72 / reach-AUROC=0.91, defense
ρ=0.76 / 0.92 ⇒ the field whose gradient = "which moves move the proof frontier toward mine vs theirs"
is learnable + generalizes. NOT count-dominated (CNN ≫ ridge), and CNN beats attention a third time —
now param-matched on the global target with 3× the epochs, so the global-receptive-field bet does not
cash out at this scale. Updated `vct-reachability-mining.md` §1 (Φ design → trained) and appended a
`TRAINING_WIKI.md` entry. Scripts `gen_phi_dataset.py` / `train_phi.py` committed earlier (dc14555).

## 2026-06-27 — Molecule corpus harvested (non-VCF combinational forced wins)

New script `harvest_molecules.py` (the §4 corpus writer) + first bank to `~/data/molecule_gold/`:
146,655 move-labeled non-VCF VCT boards (combinational forced wins = the molecule candidates),
99% distinct, sparse, gold-grows-with-distance. Updated `vct-reachability-mining.md` §4 (RAN note)
+ §6 artifacts (canonical corpus row), the index row, and appended a `TRAINING_WIKI.md` entry.
68/400 shards at the node cap ⇒ resumable with ~60× headroom.

## 2026-06-27 — `mega_vct_bb` gets a canonical solver page + two optional outputs

New topic **[mega-vct-solver.md](topics/mega-vct-solver.md)** — the API/contract reference for the
production on-device VCT solver `solve_vct_mega_bb` (previously only documented inside
`gpu-vct-feasibility.md`, a feasibility-narrative page). It is the doorway every threat-shape/mining
consumer should read: plane convention (side-to-move `board[0]`=attacker, NO swap), the call-cost law
(bulk-synchronous only), the full output table, and the regression invariants. Added an index doorway
row (after the gpu-vct-feasibility row).

Also shipped two **optional** outputs on the solver, each behind its own compiled kernel variant so the
default `(win, hit, move)` path is **byte-identical** (verdict/throughput provably untouched):
`return_support=True` → a 4×uint64 `support` mask (proof relevance window / stencil seed, accumulated on
the winning return path only ⇒ no pollution from refuted branches; ⊆ root empties = played cells) and
`complete=True` → a 4×uint64 `winmask` of ALL winning **first moves** (root stops short-circuiting;
non-root nodes unchanged). Validated 2026-06-27 (invariants + gold pass: winmask 0 unsound / 0
winning-forcing-moves-missing; the tempo guard was a *verifier* subtlety, the solver was right). Updated
`gpu-vct-feasibility.md` §9 + `vct-backward-mining.md` §5 with pointers; added a `TRAINING_WIKI.md` entry.
On `feat/gpu-vct-support-complete` (not merged).

## [2026-06-27] CPU vcf solver retired (gated, not deleted) + fast/deep VCT test tiers

Extended [topics/mega-vct-solver.md](topics/mega-vct-solver.md) with a **"CPU solver retired"** section.
`gomoku/vcf.py` (`solve_vcf`/`solve_vct` + `*_from_planes`) is now a **runtime-gated** oracle: every
public entry point raises `CpuSolverRetired` unless `GOMOKU_ALLOW_CPU_SOLVER=1`. Kept intact as a
bootstrap/reference (all internals untouched); the GPU `solve_vct_mega_bb` is the runtime future. Runtime
reaches (MCTS leaf-VCF, eval overlay, self-play teachers, web/play) are left to **throw** so Jason can
triage them to the GPU solver place-by-place — no silent CPU parity wanted.

Banked the **reusable fixture-based fast-test pattern**: commit a small golden npz of CPU-oracle truth on
clean/non-capped boards (`regen_vct_fixture.py` → `fixtures/vct_golden.npz`), then diff the kernel against
it at a tight budget with NO vcf at test time (a non-capped verdict is budget-independent). Three tiers —
FAST (`test_mega_vct_bb.py`, no vcf, <15 s), GATE (`tests/` via `conftest.py` override, the sanctioned
oracle path), DEEP (`validate_deep.py`, live vcf + winmask soundness/completeness gold, on-demand). The
deep completeness oracle must include vcf's tempo guard `_defender_has_four_or_five` (the solver was right;
the verifier was the subtle part). Added a one-line pointer in `topics/conventions.md`; `TRAINING_WIKI.md`
entry. On `feat/cpu-solver-retire` (not merged).

## [2026-06-27] Solver `carriers` output (#88) + the stencil CERTIFICATE property

Two linked additions, both merged/landing to `main`.

**(1) `return_carriers` (#88, merged `ad4d52e`).** `return_support` was returning the required-**openings**
(empty cells the forcing line plays into), not the **stones** that form the threat — a spec gap surfaced
while mining VCT stencils (the `.BBBB.` ask gave back the two `.`s, not the four `B`s). Added a complementary
`carriers` output: the load-bearing OWN stones (root-own ∩ collinear-within-4 of any support cell), the `B`
channel to support's `./p`. Purely additive (every existing kernel variant byte-identical; golden
`.BBBB.`/`BB.BB` → carriers == the four `B`). Documented in
[topics/mega-vct-solver.md](topics/mega-vct-solver.md) (carriers section + invariants 5–6) and the index row.

**(2) The certificate property — measured.** With `support ∪ carriers` now a complete replayable stencil,
tested whether a stencil that wins **in isolation** is a context-free win. Result (harness
`scripts/threat_shapes/certificate_falsification.py`, pool 4096, `max_nodes=500`): **660/660** mined attacker
VCTs win from their carrier stones alone on an empty board (extraction is faithful); **0/2913** tempo-safe
placements (random non-attacking defender stones) refute the win; the sole breaker is defender counter-tempo
(control: defender-with-VCT refutes ~7%). This is Allis's dependency-based / threat-space search soundness
([topics/allis-threat-theory.md](topics/allis-threat-theory.md) §3, principle 2) made operational + GPU-
falsifiable, and it promotes L1 (self-contained subset) from "candidate index" toward a **certificate
engine**. Written into [topics/shape-library-engine.md](topics/shape-library-engine.md) §3 (*Empirical
certificate property*), §7, §8; pointer in [topics/allis-threat-theory.md](topics/allis-threat-theory.md)
§3; index row. Honest bounds recorded: empirical not formal QED; the filter used ("no defender VCT") is
stronger than the true *immediate-tempo* condition; self-contained/offensive subset only (W-dependent
defensive shapes need the v2 `W` channel). On `feat/stencil-certificate`.

## 2026-06-28 — md-extraction (#91): the §3 minimizer blocker cracked; load-bearing W measured

Autonomous overnight run on `feat/md-extraction` (pushed, not merged). Goal: crack the single
named blocking prerequisite for the shape-library L1 minimizer — **md-extraction** (ablate on
mate-distance invariance, but the GPU solver returned only `(win, hit_cap)` and capped nodes,
not depth). De-risked first by a 7-agent design Workflow (5 analysts → adversary → synthesizer)
before any kernel surgery; the adversary killed the planned CPU md cross-oracle as mis-calibrated
(kernel `candidate_own` narrower than CPU's any-stone set ⇒ `md_gpu>md_cpu` with no bug) + against
the retired-solver canon.

Shipped: a `depth_cap` kernel variant (`solve_vct_mega_bb(max_depth=)`, one new per-board input,
zero new outputs, byte-identical default — invariant #9) + `solve_md_min` (order-independent
binary search). GPU-self validated (byte-identical-vs-HEAD, depth-monotonicity, md bracket,
md_min == an independent linear scan); FAST tests + a GPU-self golden fixture; full suite green.
The md-invariant minimizer (`md_minimize.py`) on `molecule_gold` (16,345): orig 13.2 → 4.91 (B+W)
stones; **load-bearing W is the long-VCT phenomenon, measured** (0% at md0=1 → 100% at md0≥4); the
cheap `w` channel (#90) is a **~10× over-approximation** of the minimal load-bearing W. Honest
bounds banked (md0=1 collapse dominance, defender-perturbation, vocabulary not yet saturated).
Pages: [topics/mega-vct-solver.md](topics/mega-vct-solver.md) (`max_depth` + invariant #9),
[topics/shape-library-engine.md](topics/shape-library-engine.md) §1/§8, [capabilities.md](capabilities.md)
(Search & solve), [TRAINING_WIKI.md](../TRAINING_WIKI.md) 2026-06-28.

## [2026-06-28] New page: idx-2 forward VCT frontier + danger map ("solve the Bruce-Lee board for black")

Created [topics/idx2-vct-frontier-map.md](topics/idx2-vct-frontier-map.md) for a one-session
exploratory experiment (Jason: "I don't expect to accomplish this, but run it, capture the data,
see what we learn"). Forward-expand idx-2 [white to move] as an AND/OR frontier where **Rapfi-top-8
generates the moves for both sides** and the **mega GPU VCT solver is the only oracle** (black VCT =
win-terminus, white VCT = black-fumble loss-terminus; no minimax/backup). Deliberate approximation,
NOT a sound solve (the AND-node top-K gap). Three scripts under `scripts/idx2_vct/`: `frontier.py`
(append-only, resumable, D4-content-addressed reducer-over-a-log, bulk-synchronous), `probe_capped.py`,
`analyze_opening.py`. Findings: **run-a = 9.6M nodes / depth-11 / 90-min wall (TIME-bound, not space)**;
**throughput dead-flat ~1,750 nodes/s** (bulk-sync + the 250-node cap clips the solver tail) ⇒ "8 nodes
trivial, millions not a wall"; branching decelerates 3.0→1.66× (attacker ~5.1 Rapfi moves vs defender
~2.8; terminus frac →39%); dedup only ~2–3%. **The 250-node cap hides almost nothing cheap** (<1% of
capped flip to a win at 16× budget ⇒ a genuinely-hard third regime; the 2.4M win harvest is a near-floor).
The **danger map** (depths 0–7, 149,627 nodes): per-move both-sides danger densities + nearest forced
win/loss + honest cap/gap accounting + Rapfi-prior-vs-oracle-danger — idx-2 reads black-favourable
(WT 0.28 vs BT 0.08) but ~half-unknown; Rapfi's mid-ranking is NOT danger-calibrated; low-danger+high-
uncertainty ≠ safe. Also benchmarked the Rapfi cost knob (`max_node` binds; small-ms timeout truncates
multiPV — the wrong tool). UI noted, not built. Added the index doorway row (VCT cluster) +
[capabilities.md](capabilities.md) "Search & solve" row. Data (out-of-git): `~/data/idx2_solve/run-a/`.

## [2026-06-28] mega-VCT streaming / work-stealing dispatch (#93)
Attacks the call-cost-law tail: the base kernel is one thread per board, so in the long tail the wall is
the *single hardest board* grinding to `max_nodes` while easy-board lanes have retired. **Option A —
`solve_vct_mega_bb(..., work_steal=True, resident=N)`:** a persistent dispatch of `resident` lanes that
each pull the next board index from a shared **atomic cursor** (MLX `atomic_outputs=True` →
`device atomic<uint>*`; `init_value=0` zeroes the cursor across threadgroups, no barrier dance). The
per-board AND/OR search is **byte-identical** to the base kernel — only the gid source + the (now-atomic,
uint32-widened since Metal has no `atomic<uchar>`) output store change; the Python wrapper narrows the
dtypes back. **Option B — `solve_vct_streaming(boards, budgets=...)`:** iterative deepening over the pool —
solve all at the smallest budget via A, recirculate the still-capped subset at deeper budgets, latch the
first clean verdict (clean verdicts are budget-independent). A keeps lanes full within a round; B shrinks
what the deep rounds run on. **Validated** byte-identical to base across seeds × budgets × `resident`
{<B, ≈B, ≫B} and sub-threadgroup `B` (regression invariant #10; `test_work_steal_*`/`test_streaming_*`).
**Honest scope:** the GPU already backfills at threadgroup-dispatch granularity (why the law is flat), so
for a single ~16k batch the gain is modest; A's real win is streaming pools **larger than one dispatch**
(deep frontier labeling — millions of boards, one tail at the end). Feasibility spike + full rationale in
[topics/mega-vct-solver.md](topics/mega-vct-solver.md) § Streaming / work-stealing. Spawned by Jason's
"introduce new boards as they complete" brainstorm.

## [2026-06-28] Benchmarked the work-stealing/streaming solver — mapped its niche (#94)
Apples-to-apples on 84k real idx-2 `run-a` boards (replayed from move-sequences, **no Rapfi**; 24%
capped@250), solver-only (`scripts/vct_metal/bench_throughput.py`). Findings — and a design bug caught by
measuring: (1) **work_steal NEVER beats base** on a single in-memory pool (0.93–0.97× at resident=16384;
0.34× at 4096) — the GPU already backfills one dispatch, so the cursor is overhead; its only justification
is pools too large/incremental to gather up front. (2) **Iterative deepening on the BASE kernel is 1.60×**
vs a single budget-4000 dispatch (identical verdicts) — the deep budget only touches the shrinking survivor
tail. **The niche.** (3) **The shipped `solve_vct_streaming` was built on work_steal → 0.87× (a LOSS)**;
work_steal's big-round-0 handicap ate the deepening win. **Fix:** `solve_vct_streaming` now deepens on the
base kernel by default (`work_steal=True` opt-in for the streamed regime). (4) Coarse ladder beats fine
(3 rungs 1.60× vs 5 rungs 1.10× — each round re-solves survivors from scratch). (5) The frontier's old
wave-chunking@16384 was 0.66× of one big dispatch. Bonus apples-to-apples: solver alone ≈ 6,900 boards/s
@ budget 250 = ~3.9× the prior ~1,750 nodes/s *whole-pipeline* frontier rate ⇒ the frontier was
**Rapfi-bound, not solver-bound** (confirming work_steal could never have sped it up). We predicted, tried,
measured, wrote it down. Full runtime-properties table in
[topics/mega-vct-solver.md](topics/mega-vct-solver.md) § Streaming / work-stealing → Measured runtime properties.

## [2026-06-29] mega-VCT throughput characterization sweep (#95) — overnight
Append-only/resumable sweep over 4 board-populations (quiet frontier / capped-only / random / deep,
all Rapfi-free) × strategies × budgets + per-pool resolution profiles + an N-sweep + a MAXD 32→64 study
(`scripts/vct_metal/{sweep_throughput,analyze_sweep,n_sweep,maxd_study}.py`). Findings: (1) **hardness is
bimodal** — 71–83% resolve by budget 10; budget beyond ~250 buys almost nothing; the **capped tail is
near-bottomless** (6.5% resolved at 20000 = 80× budget, 1.3% wins). (2) **work_steal and chunked lose on
every pool** (0.6–0.8×) — confirms #93/#94 across board shapes. (3) **#94's deepening 1.60× is
N-CONDITIONAL** (dated correction): deepen-vs-base@4000 = 0.69×/0.79×/1.19×/1.63× at N=10k/20k/40k/84k,
crossover ~30–35k; the win needs a hard-survivor batch dense enough to saturate the GPU (~0.9M nodes/s
all-hard vs ~0.4M mixed). (4) **MAXD 32→64 changes nothing** — caps are node-bound not frame-bound,
`gained_by_64=0` on random, keep MAXD=32 (added env-gated `GOMOKU_VCT_MAXD`, default 32 = validated path).
Synthesis: optimal throughput = **screen-cheap (budget ~10–100) then batch the hard survivors densely**;
use the append-only logs to route known-hard boards straight into the dense batch (oracle, avoids the
~1.8× re-solve tax). Neither `max_nodes` nor `MAXD` can fill the caps — a genuinely-hard regime.
**Process note:** a self-referential `pgrep -f <name>` waiter bug stalled the MAXD=64 chain overnight (the
waiter matched its own command line); the study jobs themselves were fine — rely on task-completion
notifications, not pgrep waiters. Full tables in
[topics/mega-vct-solver.md](topics/mega-vct-solver.md) § Throughput characterization sweep.

## [2026-06-29] where work_steal DOES win — the forced-wave tail + the width knob (#96)
Ran the matchup #93/#94 never did: *forced waves* (supply runs dry because you chopped the work into
separate dispatches), where the question is whether refilling the queue across wave boundaries beats
relaunch-per-wave. `bench_refill_vs_wait.py`, mixed pool N=16384, cap=2000, parity 12324/6555. **Two
axes:** (1) **at matched width, refill beats relaunch-per-wave** — 1.20× at width 4096 (W=4), 1.29× at
width 1024 (W=16); finer slicing = more inter-wave tails erased = bigger win. So the tail is real and the
cursor erases it — Jason's original "feed the tail" intuition was right *for the forced-wave regime*.
(2) **but width dominates** — `resident` is a width throttle (0.12/0.22/0.40/0.66/0.96× oneshot at
R=1k/2k/4k/8k/16k); going narrow costs 0.34–0.40× *however* you handle the tail, far more than the
1.2–1.3× refill recovers. **Synthesis:** width is king (gather a big batch, run wide = 3–8× any narrow
strategy); work_steal/refill is the right tool ONLY when forced narrow (streaming / memory-bound / unmergeable
waves), and even then only if the next boards are already in hand — a *gated* frontier can't be refilled.
This resolves the #93/#94 "no-op" — it was a no-op *at full width*; the value lives at forced-narrow width.
Full tables in [topics/mega-vct-solver.md](topics/mega-vct-solver.md) § Where work_steal does win.

## [2026-06-30] New page: VCT-terminus self-play A/B result (#100) — throughput win, robustness loss

Created [topics/vct-terminus-selfplay-result.md](topics/vct-terminus-selfplay-result.md) — the science
slice for #98/#99. Matched 9×9 A/B (`vctsci-terminus` vs `vctsci-control`, grown to e500): the terminus
reaches **equal fixed-baseline strength at ~45% of the control's wall-clock** but **loses head-to-head
75–25 (0 wins/120)** and 0-40 to the champion — ending every game at the first VCT means it never learns
to defend, so a sound opponent denies every VCT (finisher fires = 0) and it collapses. Idea #11's caveat
is the dominant effect. Added an index row (VCT cluster) + updated idea-pile #11 (BUILT → **TESTED**) +
TRAINING_WIKI 2026-06-30. Two reusable lessons banked: **eval the EMA `worker_weights.pt`, not the epoch
checkpoint's raw state_dict** (terminus 6% raw vs 68% EMA), and **fixed baselines saturate for strong nets
⇒ gate on H2H** (champion reads 62% vs heuristic yet 40-0s the control). Filed #101 (train the terminus
long — p90 plies → 81?).

## [2026-07-01] Long-run coda (#101): VCT-terminus p90 is a stable attractor at ≈14.5, not the 81 gate

Extended [topics/vct-terminus-selfplay-result.md](topics/vct-terminus-selfplay-result.md) with a
"## Long-run coda (#101)" section — the natural next probe of #100. Trained the VCT-terminus player from
scratch for **~2,700 epochs with no evals** (wandb `kgajrge4`, `jasonyandell-forge42/gomoku`, run dir
`~/data/vctsci-101-long/`) and watched `selfplay/plies_p90`: it does **not** climb toward 81 (the retired
9×9→11×11 gate = a full board), so **Hypothesis B held** (Jason's bet) — a **rising-then-flattening fixed
point at p90 ≈14.5 / mean ≈9.6**, the net sharpening inside the fast forced-win regime (pl 4.38→2.17, vl
0.39→0.022, all flat). Verified block-mean trajectory off the run: cold ~28 → trough 11.9 (~e85) → decelerating
creep 11.9→12.7→13.2→13.4→13.6→14.0→14.4→14.7, flat for the final ~1,000 epochs. Mechanism: the defender (its
own EMA twin) learns to **postpone** the VCT, never to **prevent** it — the self-play defensive ceiling is
**structural**, not undertraining. Banked two honesty caveats: `plies` is an unreliable defense proxy (cap50
loses recall as play sharpens ⇒ part of the creep is the detector, not defense — only `fires>0` vs a real
opponent settles it, and #100 said no), and "stronger at short games" is inferred from falling pl/vl, not a
measured strength number (evals were off). The way past the ceiling = opponent-independent defensive signal =
the supervised VCT aux-head (#102/#103). Updated idea-pile #11 lineage (added the #101 coda), the index row,
and TRAINING_WIKI 2026-07-01. Closes #101.

## [2026-07-01] Perf blitz day 1 (#112 landed, #109 landed, #114 filed): one streaming worker replaces the gen fleet; standard eval minutes → seconds

Jason's 5-day directive: max out the box — training, gen, VCT, standard eval; knobs may be
consolidated. **Gen (#112, merged):** converted `_generate_games_native` to per-game plies and added
**continuous refill** (`--concurrent-games`) + **streaming production mode** (`--stream`: chunk
flushes + between-round weight hot-reload) — ONE process at width 256–512 = **4,080–5,579 aug-pos/s
vs the 4-worker fleet's ~1,000–1,300 (~3.4–4.6×)**, oracle fully hidden under search, poison check
0-violation both paths, `sound-world` cell rewired to the single streaming worker. Law amendment:
width-is-free has an intra-simdgroup **divergence bump** (44→108 ms/call at 150→862 boards) that
saturates — past ~1k boards width is ~free again. **Eval (#109, merged):** the batched arena now
implements the **VCT finisher** (one bulk mega-solve per round; hybrid vs heuristic 15W-0L-5D in
4.4 s — matches the legacy receipt that took minutes), model specs FAIL LOUD on unimplemented
kwargs, and the post-run final eval runs on **MPS** (run is torn down, GPU free): standard
4-baseline battery 37 s CPU → **14.3 s** wall, identical results. **13×13 (#113 prerequisite,
#114 filed):** the solve is 100% of the wall (fp16-eval frees the evaluator and buys nothing);
resolve census on real veto batches = 48% @10 / 9.5% @50 / **42.5% capped** — no tail, half the
batch grinds; ladder 0.83×, oracle-sort 0.98×, tg-variants 1.00×, **precheck refuted again
(0.59×, new mechanism: null boards are themselves cap-bound)**. Next: multi-thread-per-board
kernel pass + cap25 recall study (worktree `gomoku-mega-vct-bb` kept, receipts in #114). Full
receipts: [topics/mcts-perf-ceiling.md](topics/mcts-perf-ceiling.md) 2026-07-01 refill entry.

## [2026-07-02] Perf blitz day 2 (#114): lanes=K multi-thread-per-board kernel lands — 1.34× solve / 1.29× gen at 13×13, bit-identical

Day-1's census left one lever; built it. K simd lanes cooperate per board in
`mega_vct_bb` (lockstep replicated-state DFS; the two per-node candidate scans
lane-partitioned + simd-reduced; MIN commit == lowbit order ⇒ verdict
**bit-identical**, invariant #11). Receipts on 132 REAL 13×13 merged-veto
batches (360,925 boards, cap50, `bench_lanes13.py` — committed, unlike day-1's
scratchpad): K=2/4/8/16 = 1.09/1.23/1.34/1.36×, verdicts identical every
batch. End-to-end gen 48@32: wall 67.7→52.6 s (1.29×), aug-pos/s 297.5→383.2,
game stream identical. Honest scoresheet: predicted 2–4×, got 1.34× (K×
thread inflation at real widths; narrow-batch ceiling 1.72× @B=150 15×15).
`GOMOKU_VCT_LANES=8` env knob (default off) wires it into gen. FAST tier +
full pytest green. Left in #114: cap25 recall study, veto-breadth staging.
Details: [topics/mega-vct-solver.md](topics/mega-vct-solver.md)
§ Multi-thread-per-board; [topics/mcts-perf-ceiling.md](topics/mcts-perf-ceiling.md) 2026-07-02.

## 2026-07-02 — Fan-out wiki refresh + fact-organization pass (pre-curation)

A two-stage subagent fan-out to get the accumulated notes into real, discoverable
pages *before* a later high-level curation pass. Stage 1 (9 pages): created
[topics/training-run-reference.md](topics/training-run-reference.md) (every knob/switch
+ quick-start), [topics/eval-suite.md](topics/eval-suite.md) (command-first how-to-eval),
[topics/rapfi-pool.md](topics/rapfi-pool.md), [topics/vct-mining-research.md](topics/vct-mining-research.md)
(seek-VCT synthesis hub), [topics/bruce-lee-model.md](topics/bruce-lee-model.md), and
[topics/net-architecture-and-representation.md](topics/net-architecture-and-representation.md)
(recovers the 2026-07-01 Fable representation rationale from session logs — line planes /
global pool / the hybrid-Rapfi north star — with an honest recovered-vs-lost box, since
Fable's reasoning channel is redacted); curated/extended
[topics/board-size-transfer-and-warm-start.md](topics/board-size-transfer-and-warm-start.md)
(+ the auto-graduating 9→11→13→15 ladder), [topics/mega-vct-solver.md](topics/mega-vct-solver.md)
(+ performance: lanes=K #114, the oracle-veto = 91%-of-gen-wall finding), and
[topics/sound-world-recipe.md](topics/sound-world-recipe.md) (+ the 13×13 graduation NEGATIVE
result #113 + the new role-invariant "rails" ideas). Stage 2 (curation): integrated all 9
into the index task-view table (temp block removed), re-pointed the parallel-write link
stand-ins to their real siblings, disambiguated the "Bruce" checkpoint provenance (two runs:
live `gogpmbhw` e2659 w/ choice-head vs the `zrjfwny2` 128×10-bigbuf eval ladder where
eval502=e500 and e588_best=e605), and reconciled the small-net param count (**345,885 @ 9×9
vs 395,605 @ 13×13** — the +49,720 is the board-area flatten in `policy_fc`+`value_fc1`; the
raw 396,774 state_dict also counts BatchNorm buffers). A **high-level curation pass is still
pending** (this stage was facts-in-place-and-organized, not synthesis).

## [2026-07-03] Perf-blitz fan-out wrap: kernel lane COMPLETE, cap25 gate MET (~2x, human-gated flip), trainer fat trimmed (#114/#115)

Three parallel verified-first agent lanes closed out the day-1 leverage list (GPU benches serialized
via an atomic mkdir lock). (1) **lanes=K kernel: COMPLETE** — K-sweep saturated (K8=1.34x/K16=1.36x
real-gen; synthetic 1.60x@K8, K16 regresses past the B×K≈25k thread-inflation ceiling); byte-identity
green K∈{2..32}; shared-stack rewrite explicitly not pursued; `bench_lanes13 --synth` self-check
shipped. Prefer GOMOKU_VCT_LANES=16 at 13×13 if enabling. (2) **cap50→cap25 recall study: gate MET** —
recall of cap50-proven vetoes at cap25 = 98.64% (9×9 champ) / 99.39% (13×13 full-game) / **99.93%
(13×13 sound-world net)**; solve ~**1.98×** both sizes, composable with lanes=K (composed 13×13 stack
≈2.7× on the ~90%-of-wall component). All proven wins are defense escape-children (terminus fires 0);
monotonicity/leak-capped invariants + poison@25 clean. **Flip is HUMAN-GATED** (#114, label swapped):
recommended board-size-conditional (cap25 @13×13+, keep cap50 @9×9 — marginal recall AND not the
bottleneck there); caveats = distribution-dependence, leaks-are-played-blunders (K-cap precedent),
Δelo slice pending. Tooling: `scripts/vct_metal/cap25_recall_study.py`, `GOMOKU_POISON_BUDGET`.
(3) **Trainer quick wins (#115 closed)**: fused L2 (-9.7%/step, gradient bitwise-identical;
_foreach_norm SLOWER on MPS, _foreach_pow won), ~15 host syncs/step → one packed transfer
(byte-matched logs), `--shape-stats-every` (default 10; ~230 ms/epoch amortized at 1.5M rows), plus
the honest profile: at fixed sgd-steps=64 the epoch is GEN-dominated (train 1.4–1.9 s vs gen 10–34 s
at 13×13) — the trainer was never the wall. Same-seed 3-epoch trajectory identical before/after;
1129 tests green. Receipts: #114/#115 comments.

## [2026-07-03] LeGomoku proposed — latent-space world-model experiment enters the Experiments hub

New thread [topics/legomoku.md](topics/legomoku.md) (brainstorm only, no code): can a JEPA-style
latent world model make a *better search* for a rule-following game? Weak version (learned simulator)
pre-killed (MuZero precedent — rules are free); strong version = learn the threat-space abstraction
Allis hand-built, search over macro-moves (forcing exchanges) at fixed node budget. Honest prior from
Jason's Texas-42 world-model splat: these models wall on the non-geometrical ("well who friggin
knows"); gomoku's analog is search-fog, not hidden state — "probably goes splat," pre-stated bets on
record. Unfair advantages logged: VCT oracle as latent-geometry ground truth, 234k rails-v0 games on
disk, the momentum-swing eval. First spike scoped: k-step latent unroll value-fidelity on contested
positions. Second stated goal is the learning itself — a written-down splat is a win.
