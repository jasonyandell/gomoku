# Gomoku Wiki Log

Chronological record of wiki maintenance. Keep entries append-only and use a
consistent heading so future sessions can scan recent changes with simple tools.

Older eras: May + June 2026 rotated out 2026-07-04, preserved in git history —
recover with `git show ca76350:wiki/_archive/log-2026-05.md` (or `log-2026-06.md`).

## [2026-07-04] Autolab primary design UNIFIED — complete, TV + lessons designed in

Jason's directive: *refine/simplify/unify; no parts of the design left for
future us — implementation will follow design.* Same-day pass over
[topics/autolab-primary-design.md](topics/autolab-primary-design.md):

- **Scheduler collapsed to four rules** (quantum · bounded-window share ·
  champion-as-idle-task · admission). Deleted, not deferred: deficit counters,
  rung-as-priority, Δelo-ranked pools, aging-as-mechanism (FIFO by age IS the
  order), SMOKE-as-queue-item (now an admission check). **The scheduler never
  reads a performance number** — the derby lives in keep/park decisions; the
  ladder is budget vocabulary.
- **NEW §5 The TV**: dashboard = a third window on the one fold (worker/pick ·
  researcher/packet · human/TV); panels NOW / ON DECK / THE DESK / SCOREBOARD /
  OVERNIGHT / VITALS; overnight diff = `seq > watermark`, free from
  append-only; the auditor role is a TV panel.
- **NEW §6 Lessons compound**: `lesson` rows (executable one-liners, ledger) +
  wiki prose (cited at commit+path) + the **admission lessons-wall** (scope-tag
  intersection with `refuted` lessons rejects, `challenges:<id>` re-litigates
  loudly) + lessons always in the packet.
- **NEW §7 Worker**: designed now (merge SHA = citable trainable commit),
  built last.
- Finish line → an 8-step **build order**; scouts/reviewer-as-role/
  auditor-as-role/separate-lesson-subsystem **rejected with reasons**.
- Blast radius: [review page](topics/autolab-design-adversarial-review.md)
  same-day addendum (A1/A6/A7/A8 dispositions superseded by unification);
  [researcher-contract](topics/autolab-researcher-contract.md) deferred-list
  resolved; [hub](autolab.md) re-blurbed.

## [2026-07-04] Autolab PRIMARY DESIGN banked + adversarial review (design session)

The fresh-eyes design session (Jason + Claude, working from the 2026-07-04
vision corrections) produced and banked the autolab's target design:

- **NEW [topics/autolab-primary-design.md](topics/autolab-primary-design.md)**
  (LIVE, wins over pages it disagrees with): the one-breath version, the
  OS-scheduler design (champion-as-idle-task, verdict-denominated fairness,
  v0 minimal cut, Δelo-honesty), the two ledger walls (facts-not-commands;
  champion-tag-as-projection), the researcher packet promoted to
  cage-readiness, the invocation shape (OS-permissions wall), the five blessed
  decisions of 2026-07-04, and the phase-1 finish line.
- **NEW [topics/autolab-design-adversarial-review.md](topics/autolab-design-adversarial-review.md)**
  (HISTORICAL): 12 attacks (A1–A12), 5 BROKE / 4 BENT / 3 HELD; accepted fixes
  folded into the design page.
- **Blast radius:** [autolab.md](autolab.md) hub re-pointed (primary design
  first; architecture reframed "the what as BUILT");
  [architecture](topics/autolab-architecture.md) got dated corrections retiring
  researcher-set priority; [researcher-contract](topics/autolab-researcher-contract.md)
  got the dossier-ranking reversal (cage-readiness, not presentation) + the Y4
  `max_wall_secs` doc/code fix; [dr-tabletop](topics/autolab-dr-tabletop.md)
  gained **two new REDs** (rows 7–8: append-side torn tail — where the naive
  prepend-`\n` fix is provably wrong, truncate-under-flock is correct; the
  multi-row-transaction class) + the 2026-07-04 review addendum (Y2/Y5–Y8),
  fully banking the session's `/tmp/autolab-vision-and-gaps.md` capture.

## [2026-07-04] Curated the autolab topic *bodies* (the fold-in only added banners)

The [fold-in](#2026-07-04-the-autolab-is-now-a-headline-hub-6th-hub--featautolab-sim-fold-in)
gave the six autolab topic pages status banners but left their **bodies** in
old-wiki style. This pass applied the [five hard rules](curation.md) to the bodies
(in `feat/autolab-sim`):
- **[architecture](topics/autolab-architecture.md)** (CANONICAL) — hoisted the
  settled P1–P7 outcome above the old forward-looking `**Status:**` block; **fixed a
  direct contradiction** where the research lane said it *"waits hours"* while
  [doctrine §4](topics/autolab-doctrine.md) titles itself *"'waits' is deleted"*;
  flipped the thesis from build-plan to past tense.
- **[supervisor](topics/autolab-supervisor-and-monitor.md)** — cut the
  build-handoff task list + file-inventory table + full `uv run pytest` command
  (implementation receipts for shipped PR #64, not operating contract) to a git
  tombstone; 36 KB → 31.7 KB.
- **[doctrine](topics/autolab-doctrine.md)** — fixed broken section numbering
  (§5b landed after §6 → renumbered §7); compressed the ~40-line uv dev-diary and
  the dated build-state Status to the doctrine-relevant core.
- **[arena](topics/autolab-arena-eval-lane.md)** — banner now disambiguates the
  **live H2H gate** (P4 #59, ran + crowned champions) from the **unbuilt
  panel/gamut** this page specifies (the old banner conflated them).
- **[researcher-contract](topics/autolab-researcher-contract.md)** — hoisted the
  shipped safety-core verdict under the banner instead of burying it two-thirds
  down.
- **[dr-tabletop](topics/autolab-dr-tabletop.md)** — left as-is (already
  verdict-first). Lint clean (0 errors); tombstones recover via `git show 7c82b7d`.

## [2026-07-04] The Autolab is now a headline hub (6th hub) — feat/autolab-sim fold-in

Merged `main` (the great wiki curation, 360 commits) into `feat/autolab-sim`,
then folded the branch's autolab work into the new hub-of-hubs as **new content**
rather than old-format table rows (the merge dropped the pre-curation index/log
entries; the topic pages themselves never left the tree). **The autolab is now a
headline hub, peer of the Derby:** new [autolab.md](autolab.md) gathers all seven
pages —
[doctrine](topics/autolab-doctrine.md) (the why),
[architecture](topics/autolab-architecture.md) (CANONICAL, the ledger spine),
[supervisor-and-monitor](topics/autolab-supervisor-and-monitor.md) (operating
appendix),
[researcher-contract](topics/autolab-researcher-contract.md) (#61 smart lane),
[arena-eval-lane](topics/autolab-arena-eval-lane.md) (measurement leg),
[dr-tabletop](topics/autolab-dr-tabletop.md) (survive-weeks DR),
[cockpit-vs-autopilot](topics/cockpit-vs-autopilot.md) (operating lens) —
under the **trainer · arena · researcher · worker** triad framing. Wired into
[index.md](index.md) (5→6 hubs) and the sibling-hub nav on all five other hubs;
[derby.md](derby.md) now points to the autolab as "the autopilot that ran the
charter unattended." Added the four missing dated status banners
(doctrine=LIVE; researcher-contract/arena/dr=DORMANT). Nothing lost — the fold-in
is purely discoverability + banners; the 2098 lines of autolab topic content came
through the merge intact.

## [2026-07-04] New topic: wiki-curation-lessons — the portable meta

Closing synthesis of the whole curation effort, written for export to Jason's
other projects: [wiki-curation-lessons.md](topics/wiki-curation-lessons.md).
Nine lessons with local evidence — chronicle-itis lives at the leaves,
staleness flows upward, query-is-a-write-operation, mechanism-over-vigilance
(the day's four vigilance failures + their tool fixes), git-is-the-archive,
test-with-cold-agents-and-count-fetches, the load-bearing separations,
self-hosted schema + the minimal-instruction bar, curation-is-a-pass /
maintenance-is-a-loop. Linked from reference.md (conventions table) and
curation.md. Includes a bootstrap recipe for starting a new project's wiki
from these parts.

## [2026-07-04] Chronicle-itis check adjudicated: all 13 >25 KB topics KEEP-AS-IS

The lint's 13 size notices were formally reviewed (Opus, conservative rules).
**Every page passed** — the length is legitimate in each case (reference
dictionaries, API contracts/perf atlases, design specs, verdict-first synthesis
whose length is evidence density), and the earlier 2026-07-04 passes already
compressed the actual chronicle bulk (visible as in-page tombstones). Zero
inbound-anchor conflicts exist on any of the 13. Four borderline micro-trims
(~1 KB total) were recorded and deliberately declined. Future lint runs showing
these same 13 notices can treat them as adjudicated as of this date; re-review
only pages that GROW past their reviewed size.

## [2026-07-04] `_archive/` audited and DELETED — git history is the archive now

`wiki/_archive/` (528 KB, 17 files) was a safe side-by-side for the curation
pass, never long-term storage. An Opus audit (4 adversarial sub-auditors,
spot-checks incl. random mid-file entries) confirmed **every file
SAFE-TO-DELETE — zero lost knowledge**; three low-severity archive-only details
were hoisted to live pages first (v4 derby verdict → perf-log's v3→v5 gap;
claw-rediscovery v0 numbers + mod-5 probe nuance → idea-pile #10; raw 15×15
scaling-bench table → 15x15-training-campaign). Doctrine updated in
[curation.md](curation.md): rule 5 is now **"Cut, never lose — git is the
archive"** (no parallel archive tree; live pages carry dated tombstones naming
the recovery commit), and rotation is a three-step ritual (rotate with
reconcile → commit → drop staging). All ~70 inbound `_archive` references
across ~23 pages converted to tombstones. **Recovery commit for everything
deleted: `ca76350`** (`git show ca76350:wiki/_archive/<path>`).

## [2026-07-04] Extracted the TQ Promotion Gate out of the experiment-ledger into its own live page

Per the standing curation verdict (live doctrine trapped inside frozen
evidence): the **Training-Quality Promotion Gate** — the rule a behavior-changing
perf receipt must satisfy to `promote`, cited by 10+ pages — is now its own page
[ops/promotion-gate.md](ops/promotion-gate.md), banner **LIVE**, gate text moved
verbatim (~1.4 KB, five criteria unchanged) with a dated provenance line. The
[experiment-ledger.md](ops/experiment-ledger.md) gate section is now a 3-line
pointer; its head was restructured to lead with a LIVE-intake preamble ("new
receipts append at the bottom under dated era headers", per curation.md routing
table) then a `## May 2026 campaign … ARCHIVED-IN-PLACE, frozen evidence` marker
above the 44 receipts. "ARCHIVED-IN-PLACE" now sits in the first 10 lines
(exempts the file from the lint's >60 KB rotation-smell warning) while staying
honest — worded "campaign section archived-in-place; intake live", because the
file *is* still the live receipt intake. Retargeted 8 gate-specific links across
6 pages (ops.md ×2, ops/frontier.md ×2, ops/status.md, and the three
`#training-quality-promotion-gate` anchor links in perf-bench-vs-real-training-cost,
m5-max-fp16-and-throughput-regimes, wall-clock-to-elo-metric) from the ledger to
the new page; left ~30 evidence/receipt links to the ledger untouched. Lint:
**0 errors, 0 warnings** (rotation warning cleared).

## [2026-07-04] First Query-loop filing: white-VCT rail redux → idea-pile addendum

First real use of the new [curation.md](curation.md) § Query rule. Jason
proposed a 15×15 Bruce-Lee-opener run with a white "lookahead-2 VCT" rail; two
independent Opus agents (one wiki-guided, one flat-footed) answered from the
wiki and converged on the same verdict with the same run IDs — the proposal ≡
rails-v0 (`vraf0b6e`, #116), already run, failed by value-poisoning on
black-tilted idx-2. The synthesis (the lookahead-2-vs-oracle-solve ambiguity,
the idx-2 holdability probe as cheap falsifier, the untried inference-time
actuator seed) is filed as a dated addendum in
[idea-pile.md](topics/idea-pile.md) rather than left in chat. Side-note for
the operating model: the flat-footed agent self-routed via CLAUDE.md → wiki
and matched the guided one's conclusions — the knowledge base, not the prompt
scaffolding, carried the answer.

## [2026-07-04] Hardened tooling: wiki_rotate.py + wiki_lint.py; June rotated; 9 banners canonicalized

The two curation mechanisms are now scripts, not vigilance
([curation.md](curation.md) updated to point at both):

- **`scripts/wiki_rotate.py`** — journal rotation with the reconcile check
  built in (refuses to write unless entry counts + bytes across live+archive
  equal pre-rotation totals; atomic writes; post-write re-verify; dry-run).
  8 pytest cases. First real use: **June rotated out of log.md** (27 entries,
  46,777 bytes → `_archive/log-2026-06.md` *(since removed; recover via `ca76350`)*,
  reconciled 40 = 13 + 27; live log 70 KB → 23 KB).
- **`scripts/wiki_lint.py`** — the mechanized lint: banner presence+date on
  every topics/ page, broken relative links (live pages), topic orphans,
  rotation smells (>60 KB, ARCHIVED/FROZEN-exempt), chronicle-itis triggers
  (>25 KB, notice-tier), index-vs-log staleness. 18 pytest cases; `--json`.
- First real lint run found **9 topic pages with non-canonical status prose**
  (BANKED, CHAPTER CLOSED, bare RESULT, etc.) — all now lead with a canonical
  marker, prose preserved: board-size-transfer → LIVE; legomoku → LIVE (seed);
  engine-panel-derby → DEAD-END (lesson kept); rapfi-idx2-mine, shape-library,
  sliding-derby-v2 → DORMANT; both vct result pages + wave-of-lockstep →
  HISTORICAL. Post-fix: **0 errors, 1 warning** (experiment-ledger >60 KB —
  known, quarterly rotation planned), 13 chronicle-itis notices (trigger tier).
- CLAUDE.md + AGENTS.md (twin edit) now point "remember/curate anything" at
  [curation.md](curation.md).

## [2026-07-04] Curation playbook: routing table + the Query write path

Two gap-closes in [curation.md](curation.md), from re-reading the
[Karpathy source](sources/karpathy-llm-wiki.md) against our adaptation:
(1) the ingest classify step is now a **routing table** — every input class
(run, perf receipt, eval, idea, external material, decision, machine fact) has
an explicit evidence home + synthesis home; perf receipts route to
[experiment-ledger](ops/experiment-ledger.md) under new era headers now that
perf-log is archived-in-place. (2) A new **Query section**: answering a
question that took 2+ pages or raw evidence is itself curation input — file
the answer back (verdict update, hub row, or the missing topic). This was the
source's subtle self-reinforcing loop we hadn't adopted structurally: the wiki
compounds from *use*, not only deliberate ingestion. Recurring >1-fetch
questions added to the lint list as structural-gap findings. Source page's
Local Mapping updated to record the adoption.

## [2026-07-04] Rotation hygiene: reconcile counts + bytes before/after

Hardened the [curation.md](curation.md) rotation rule with the "how" learned the
hard way: rotate an append-only journal by splitting on the date-prefix with a
script, then reconcile — entries and byte totals across (live + archive) must
equal the pre-rotation totals. A freehand log.md rotation earlier today silently
dropped 21 entries and had to be redone from git; a two-line count/byte check
catches it. Same class of hazard as the retired-janitor re-summon: the fix is a
verification step baked into the rule, not vigilance.

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
- **Rotations**: log.md May era → `_archive/log-2026-05.md` *(since removed; recover via `ca76350`)*;
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
