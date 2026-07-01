# Gomoku Wiki Log

Chronological record of wiki maintenance. Keep entries append-only and use a
consistent heading so future sessions can scan recent changes with simple tools.

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
  [topics/alphazero-lessons.md](topics/alphazero-lessons.md) (threat-semantics +
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

Filed [topics/15x15-era-feasibility-and-plan.md](topics/15x15-era-feasibility-and-plan.md)
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

## [2026-05-25] sources | Added Sid Bidasaria "Stop babysitting your agents" talk transcript

Filed [sources/sid-bidasaria-stop-babysitting-agents-2026-05-20.md](sources/sid-bidasaria-stop-babysitting-agents-2026-05-20.md)
— the *correct* Sid talk (`wI0ptqCSL0I`, Claude channel, 2026-05-20), not the
SDK/GitHub-Action talk (`dRsjO-88nBs`). The video has no YouTube caption track,
so the gemini MCP couldn't transcribe it; audio was pulled with `yt-dlp` and run
through **Whisper locally on the M5 Max** (raw output at
`~/.claude/jobs/60c8a964/out/`). Light ASR cleanup: Whisper heard "Claude" as
"Cloud" throughout (corrected), misspelled the name, and looped on a few phrases
during silent demo stretches (collapsed). Talk arc: verification (teach Claude to
check its own work) → multi-Claude (run several once they're reliable) →
background loops (`/loop`, routines) to take the keyboard out of the hot path.
Indexed under the Source Records table.

## [2026-05-24] wiki | Lab identity rename: perf lab → research lab; perf-queue → gpu-queue

Four files renamed via `git mv`; no content added or deleted elsewhere:

- `wiki/topics/perf-lab-charter.md` → `wiki/topics/research-lab-charter.md`
- `wiki/topics/perf-lab-session-runbook.md` → `wiki/topics/research-lab-session-runbook.md`
- `wiki/topics/perf-lab-reviewer-role.md` → `wiki/topics/research-lab-reviewer-role.md`
- `wiki/ops/perf-queue.md` → `wiki/ops/gpu-queue.md`

Content reframed in the renamed files:

- **research-lab-charter.md**: retitled "Research Lab Charter — Make the Mac Sing." Mission expanded to two research areas: perf research (original scope) and training-recipe research (new). Two-queue scheduler renamed from "GPU queue / CPU queue" (hardware) to "GPU-required (serial) / everything-else (parallel)" (hardware requirement). Added "Training runs as GPU-required items" subsection: a training slice is a `run_sweep --max-wall-secs --final-eval` dispatch; the lab reads `eval/model_elo` from `<cell>/checkpoints/eval_results.jsonl`; eval stays inside the bundle. All existing perf machinery (R-S*/R-TRAIN-*, tiers, smoke-first, Reviewer gate, 12-row stop-gate triage) preserved as the perf research area's rules.
- **gpu-queue.md**: retitled "GPU-required queue — the serial lane for anything needing MPS." Notes training slices now sit alongside perf cells. RESUME STATE block and all queue content preserved verbatim except title/framing.
- **research-lab-reviewer-role.md**: updated title, embedded audit prompts, and cross-refs.
- **research-lab-session-runbook.md**: updated title and section headers; added training slice to "When to use this page."

All navigable cross-refs in wiki/ updated to new names. Skill referred to as `gomoku-research-lab` (renamed separately). No touches to scripts/, gomoku/, or ~/.claude/skills/.

## [2026-05-24] topics | Backlog idea filed — containerize the training run

- Added [topics/containerize-training-runs.md](topics/containerize-training-runs.md) — "for soon" backlog capture of Jason's idea: containerize a training run, run one container at a time, refine the `gomoku-train` skill for lower startup friction/time (proper caching). Captured during the research-lab ↔ training integration design discussion.
- Recorded the one real open question rather than filing a plan that hits a wall: **Docker on macOS has no Metal/MPS passthrough** (Linux-VM containers can't reach the Apple GPU → CPU fallback). So the idea targets either the off-Mac/at-scale path ([[az-at-scale-vs-laptop]]) or a non-Docker reproducible run unit on the Mac (lockfile + warm venv + `run` verb + weight cache). Decide that fork first.
- Linked from the index Page Catalog (Operations And Use). Not started; no memory entry yet (design still in flux).

## [2026-05-23] topics | Core ML design-envelope page published + L09c-L09h research lanes queued

- Added [topics/coreml-design-envelope-and-our-fit.md](topics/coreml-design-envelope-and-our-fit.md) — characterizes Core ML / ANE's design center (the iOS/macOS app ML stack: Vision, Siri, AR, FaceID), maps our research-compute gomoku workload against that envelope (20-100× above design call rate, 3-30× below design model size — worst corner), and proposes six concrete research lanes (L09c tiny on ANE, L09d medium on ANE, L09e routing-units sweep, L09f larger-V amortization, L09g model-size sweep at V=512, L09h .mlpackage re-export cost). Frame: "M5 Max as mainframe is learning where it breaks; even if we don't directly leverage it in the end, we'll know."
- Cross-linked from [topics/m5-max-as-mainframe.md](topics/m5-max-as-mainframe.md) (parent philosophy), [topics/coreml-ane-residency-lab.md](topics/coreml-ane-residency-lab.md) (sister control-plane page; now points at the design-context page as recommended first-read), and the gpu-queue Background section (six new lane yamls).
- Memory `project-coreml-reality` updated with the 2026-05-23 framing: design envelope vs our workload, the MPS-relief mechanism is real but the production lever was elsewhere (L11b' sgd_per_position cap), where Core ML is the right tool for us (deployment, possible match-eval sidecar).

## [2026-05-23] topics | M5 Max fp16 + throughput regimes findings page published

- Added [topics/m5-max-fp16-and-throughput-regimes.md](topics/m5-max-fp16-and-throughput-regimes.md) — public-facing writeup of three surprising chip findings from the 2026-05-23 perf cycle: (1) fp16 on MPS is no longer slow at torch 2.11.0 + fused conv+bn (small/V=512 +97.2%); (2) same chip has bandwidth-bound and dispatch-bound regimes depending on model size (small bandwidth-bound, tiny dispatch-bound; same V=512); (3) independent perf levers compose multiplicatively (predicted 2.530, measured 2.529 — to four decimals).
- Goal: searchable from "PyTorch MPS fp16 slow", "Apple silicon fp16 benchmark", "M5 Max throughput", etc. Open-source the corrections to the folk wisdom we ran into in the forum-thread archaeology.
- Cross-linked from [topics/m5-max-as-mainframe.md](topics/m5-max-as-mainframe.md) and [topics/research-lab-charter.md](topics/research-lab-charter.md). All numbers backed by yaml receipts in [ops/experiment-ledger.md](ops/experiment-ledger.md) with Reviewer-APPROVE audits.

## [2026-05-22] ops | frontier run 20260522T061713Z curated

- Integrated `outer-loop-python-profile` receipt from run `20260522T061713Z` after worker merge (`5e20aaa`, integrated as `411ed75`). Marked the lane completed/rejected in `.frontier/lanes.json` and the ops board.
- Curated the profile result into [ops/status.md](ops/status.md), [ops/frontier.md](ops/frontier.md), [ops/baselines.md](ops/baselines.md), [ops/experiment-ledger.md](ops/experiment-ledger.md), and [ops/test-ledger.md](ops/test-ledger.md): bounded wave-mode worker profile wall 1.064s, evaluator 84.3%, native search excluding evaluator 11.0%, measured post-search Python 4.7%.
- Preserved worker detail in [ops/open-notes/20260522T061713Z-01-outer-loop-python-profile.md](ops/open-notes/20260522T061713Z-01-outer-loop-python-profile.md). Added the artifact caveat: raw JSON/log paths named by the worker were not present in main after worktree cleanup, so rerun the bounded profile command if exact JSON is needed.
- Promoted no unblocked post-search Python lane; next perf attention should be evaluator/engine overlap after ANE rail proof, or a narrowly scoped native-search/evaluator-boundary profile if the manager wants another CPU pass.

## [2026-05-22] ops | frontier run 20260522T054739Z manually recovered

- Frontier workers all exited successfully, but the manager failed during integration with a stale UI context (`Extension ctx is stale after session replacement or reload`). Manually merged all five worker branches, resolved curation conflicts, removed the run worktrees/branches, and patched `.pi/extensions/frontier-lab/index.ts` so stale background UI handles no longer mark completed runs failed.
- Marked completed lanes in `.frontier/lanes.json`: baseline receipts, production contour, quality gates, and curation. Marked ANE residency and production engine-overlap blocked until `powermetrics` can run with cached/passwordless sudo.
- Updated [ops/status.md](ops/status.md) and [ops/frontier.md](ops/frontier.md): the next actionable lane is now `outer-loop-python-profile`.

## [2026-05-22] perf | Core ML / ANE residency lab integrated

- Integrated the detached 934b Core ML / ANE residency harness into the lane
  worktree: `scripts/coreml_ane_residency_scout.py`,
  `tests/test_coreml_ane_residency_scout.py`, and
  [topics/coreml-ane-residency-lab.md](topics/coreml-ane-residency-lab.md).
- Ran the harness tests and short Core ML scheduled smoke; the smoke can only
  claim `coreml-scheduled` because powermetrics was skipped.
- Attempted the required `conv,resnet,gomoku` powermetrics scout, but it was
  blocked by unavailable cached/passwordless sudo (`sudo -n true` failed).
  Exports succeeded, but no same-window rail logs were produced.
- Updated [topics/ane-int8-inference.md](topics/ane-int8-inference.md),
  [topics/m5-max-as-mainframe.md](topics/m5-max-as-mainframe.md), and the
  wiki index with the corrected rule: trust nonzero ANE rail evidence, not the
  `CPU_AND_NE` label.

## [2026-05-22] ops | control-room curation for frontier run 20260522T054739Z

- Synced ops status/frontier pages with active frontier worktrees and receipt state: five lanes claimed under `.frontier/worktrees/20260522T054739Z-*`; before this lane wrote its own receipt, no sibling worker receipt files existed yet when checked.
- Curated perf10 production-shaped evidence from `/Users/jason/code/gomoku-perf-extension/sweep_logs/perf10-summary.tsv` as the active production-contour seed, with exact-command capture still marked as the repeat blocker.
- Curated detached 934b Core ML / ANE residency evidence into ops baselines, test ledger, and experiment ledger: Gomoku FP16 fixed fused Core ML b32/b128 cells show nonzero ANE rail by powermetrics, but remain `needs_repeat` because the harness is uncommitted/detached and not yet production-overlap tested.
- Wrote the lane open note at `wiki/ops/open-notes/20260522T054739Z-05-control-room-curation.md` and manager receipt under `.frontier/runs/20260522T054739Z/workers/05-control-room-curation/receipt.md`.


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

## [2026-05-22] research | external rated engine baselines

- Added [topics/external-engine-baselines.md](topics/external-engine-baselines.md)
  to preserve the decision path for OSS/source-available external Gomoku
  engines that can become fixed eval anchors.
- Added [sources/gomocup-external-engines-2026-05-22.md](sources/gomocup-external-engines-2026-05-22.md)
  as the source snapshot for Gomocup ratings, downloads, protocol docs, and
  candidate repos.
- Current recommendation: start with Rapfi because it combines the strongest
  external signal, explicit GPL-3.0 source, Piskvork protocol support, and
  vectorized build paths including ARM64 NEON. Treat Gomocup Elo as provenance,
  then calibrate local 9x9 strength by time control inside our harness.
- Indexed the route under "Add or interpret external engine baselines" so
  future sessions do not need to rediscover the Gomocup ecosystem from scratch.

## [2026-05-22] run | WL5 phase-2 reached cap e10200, run end

- WL5 hit its `--epochs 5000` cap on the overnight-resume segment B,
  ending the wandb run `o6cbjfnr` cleanly with `epoch10200.pt` written
  to disk and wandb finalized. All 10 processes terminated, GPU/MPS free.
- Added the **WL5 phase-2 close** retrospective to
  [TRAINING_WIKI.md](../TRAINING_WIKI.md), mirroring the phase-1 close
  format: phase-2 final state, segment-B stats table, eval scoreboard,
  run shape, what got validated, what limits got exposed, run artifacts.
- Headline numbers (segment B, n=5000 epochs, ~13.3h wall):
  pl mean 0.621 (-10% vs phase-2 reference), vl mean 0.073, plies mean
  41.5, best elo 1738 at e5477 (la4=100%, la2=100%, h=75%) — WL4 ATH
  1841 was not broken. 0 NaN, 0 worker deaths, 0 tracebacks.
- Noted two new limits-of-this-cell findings for the next cell:
  buffer undersized vs generation rate (cycled ~28× by 1M games), and
  20-game-per-baseline eval-cycle sample size is too small to read the
  1500-1700 elo band cleanly.

## [2026-05-22] runbook | formalize WL5-era launch → monitor → close recipe

- The WL5 overnight cycle (launch from cell, cron-based monitor, run-end
  cleanup, phase-N close-out, commit + push) ran smoothly enough that
  Jason called it "the recipe." Folded it into the durable docs so the
  next run doesn't reinvent it.
- **`wiki/topics/launch-sequence-runbook.md`** (this file's main
  procedure) updated:
  - Phase 5 split into 5a (active `/loop` cadence) and 5b
    (overnight `CronCreate` at `7,22,37,52 * * * *` with a
    self-contained check prompt template + cell-name-filtered proc
    counts + tight push-trigger list).
  - Phase 6 rewritten around the three "end" flavors
    (cap-reached / user-stopped / crash) and the discovery that
    cap-reach leaves 8 workers + 1 eval polling; added the **phase-N
    close-out template** that WL5 phase-1 and phase-2 closes use, and
    a commit + push checklist with the `app/**` deploy-trigger
    pre-check.
  - Handoff friction section extended with WL5-overnight findings:
    concurrent worktree procs inflating `pgrep`, macOS `awk` lacking
    3-arg `match()`, zsh `==`/`===` errors, and the buffer-snapshot
    lag tradeoff (`save_buffer_every` means resume rolls back up to
    100 epochs of weights — usually the right call).
- **gomoku-train skill (`~/.claude/skills/gomoku-train/SKILL.md`)**
  updated:
  - Cell map extended with WL4 (`44cxzc9d`, plateau ATH 1841) and
    WL5 (`o6cbjfnr`, two-phase 6199-epoch run, peak 1738).
  - Production launch summary updated to reference both monitor
    cadences and the run-end's mandatory worker cleanup.
  - "Resume from latest" common-asks row corrected — `run_sweep.py`
    does support `--resume`, used successfully in WL5 phase-2.
  - New "Friction workarounds learned during WL5 overnight" subsection
    consolidating the macOS/zsh/proc-filter/cap-reach/buffer-lag/
    cron-expiry/multi-session-commits/deploy-trigger gotchas in one
    place.

## [2026-05-27] research | gomocup-AZ technique survey source page

- New source page [sources/gomocup-az-techniques-2026-05-27.md](sources/gomocup-az-techniques-2026-05-27.md)
  filed during the post-v7 gomocup-AZ implementation arc: a frozen survey of
  AlphaGomoku/KataGo training and search techniques considered for the lab
  (WDL value head, LCB / variance-PUCT, moves-left, in-search-VCF, SE blocks,
  ConvNext, threat-block defense, etc.) with red-team verdicts for each.
- The synthesis-and-running-verdict layer lives in
  [ops/research-board.md](ops/research-board.md) — the "Open candidates"
  section + the v8 RR3/RR4/RR5 H2H verdicts confirmed WDL as the lone net-
  positive lever (+35/+56) but never beating the value-discount champion.
- Catalog row added under Core → Sources in [index.md](index.md).

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
