# Handoff — fresh workflow-runner session (read the substrate, don't inherit my narrative)

**To the new session:** you are the **workflow runner + workflow friction smoother**. You
run workflows, you smooth friction *with* the workflows; the workflows do the work. That
role is **defined in the documents below, not by me here.** This handoff is deliberately a
*doorway*, not a summary — Jason's instruction: lean hard on reading what's written, don't
take a previous session's word for the particulars. (This session got tangled with
conflicting ideas after several compactions. Discard the tangle; trust the docs.)

---

## Read first, in this order (this list IS the handoff)

One line each = why it's in the chain. Read the doc for the content; don't let my one-liner
substitute for it.

1. **`CLAUDE.md`** — the entry map + the conventions that override default behavior
   (worktree-per-unit-of-work, `merge --no-ff` never rebase, GPU = single serial tenant,
   GitHub issues for all tracking). Read as written.
2. **`wiki/index.md`** — the maintained synthesis layer; pick your doorway from it. If my
   reading order and the index ever disagree, the index wins.
3. **`wiki/topics/workflow-orchestration.md`** — **your job description.** "The
   workflow-master runs and tunes the workflows; the workflows do the work." Run vs Tune,
   the two standing lanes, the self-check for drift. Read this one twice.
4. **`wiki/topics/cockpit-vs-autopilot.md`** — *why* the role exists (attention is the
   scarce resource; gate · status · escalation).
5. **`wiki/topics/research-lab-charter.md`** — the two-queue scheduler + the triage matrix
   (what each lane may touch).
6. **`wiki/topics/sliding-derby-design.md`** (grounded v1) → then
   **`wiki/topics/sliding-derby-measured-outcomes-design-v2.md`** (the canonical RFC; v1's
   surviving conclusions are inlined in its "What HOLDS from v1"). ⚠ A path some text cites,
   `sliding-derby-measured-outcomes-design.md` (no `-v2`), **does not exist** — don't hunt
   for it.
7. **`wiki/topics/workflow-harness-capabilities.md`** — the probe results for what a Workflow
   *can/cannot* do (a detached slice survives the agent; a fresh invocation re-adopts from
   state). This is *how the GPU is serialized by construction.*
8. **`.claude/workflows/sliding-derby-composite.js`** — the built workflow you **run** for the
   GPU train-cycle. Read its header + both agent prompts: it launches a **detached** slice
   and scores it vs a **pre-stated title card**; `args.hypothesis` is how you hand it the
   next hypothesis (the seam already exists).
9. **`.claude/workflows/implement-backlog.js`** + **`reviewer-gated-fanout.js`** — the
   code / everything-else lane you also run.
10. **`scripts/sliding_gate.py`** — the calibration-immune 3-way gate the derby uses (read
    the docstring + `decide_verdict`).
11. **`.claude/skills/gomoku-research-lab/SKILL.md`** — the lab operating model + the friction
    log you append to. **Friction-smoothing lands here** (and in tuning the `.js`).
12. **Memories auto-load** (`MEMORY.md` index). ⚠ `project_derby_operating_model.md` says
    "READ FIRST to resume the live lab" but **predates the workflow-runs-the-derby model**
    (it casts the *session* as the GPU runner). Read it, then **reconcile it against
    `workflow-orchestration.md`** — the wiki is source of truth. (Worth retiring/updating;
    see follow-ups.)
13. **`handoffs/handoff-2026-06-16-composite-derby-build.md`** — the prior *build* handoff for
    the composite (status + parked threads). Read for continuity; verify its issue-status
    against live `gh` (some is stale — see ground state).

## The one sentence that ties the chain together
**The session never touches the GPU.** The *workflow* runs the derby; the derby's *detached
slice* holds the GPU outside the agent graph. Per `workflow-orchestration.md:98`: catching
yourself launching a slice / running a measurement / polling a GPU job *by hand* is the smell
— route it through the workflow instead. (That is the single thing this session got wrong and
the only "what I did" worth carrying forward.)

---

## Live ground state (factual — so you don't trip; not instruction)
- **main @ `6648e27`**, clean, pushed, up to date with origin.
- **GPU / derby: nothing running.** Clean slate. (`pgrep` before any dispatch anyway, per
  CLAUDE.md.)
- **The composite is built + proven:** `sweep_runs/composite_derby_board.jsonl` has **one**
  line — the `champ-cont` known-answer self-test scored `measured_outcome=confirm`. The
  machinery is trustworthy on a known answer.
- **This session's merges:** **#47 CLOSED** (janitor now auto-reclaims clean+merged stale
  siblings; ran in prod → gauge clean). **#49 plumbing MERGED, issue stays OPEN** (added the
  `ckpt:<path>` attacker + `--attacker-sims` to `white_defense_suite.py`/`white_defense.py`;
  #49 is the live white-defense instrument — its headroom/positive-control is a *measured
  outcome that has not been run through the derby*).
- **Repo hygiene gauge:** clean (worktrees=13, branches=15, merged-undeleted=1).
- **Ignore** the stray `sweep_runs/wd_headroom_eval502_v1fixture_atk400.json` — it's the
  output of a hand-run measurement (the drift). If that number matters, re-derive it *through
  the derby*, not from this file.

## Open threads (pointers, decide for yourself after reading)
- **#49** (open, in-progress): white-defense instrument. Its positive-control is a
  derby-shaped measured outcome. The instrument↔gate seam — wiring `white_defense_tally` into
  `run_gate`'s `white_loss_fn` (see `sliding_gate.py`) — is **GPU-free** workflow/tune work.
  Read #49 + the gate and judge.
- **#43 / #46** (`needs-live-validation`): competing hypotheses (defense-teacher vs
  plateau-escape). The RFC says the **race decides**, not a gut call.
- **Composite GROWTH path** (research supervisor + work supervisor + multi-hypothesis queue +
  the dumb re-kicker for "forever") is **designed-but-not-built** (`composite.js:11-13`); the
  re-invocation chain "needs one E2E test." Build only if the docs + Jason point you there.
- `scripts/gh_prime.sh` prints the ready queue at session start; GitHub issues are the tracker.

## Next action
Read the doorway list in order. Then **adopt the workflow-runner role as the docs define it**:
pick the top hypothesis/lane, kick the workflow, watch the board JSONL / the branch, and smooth
any friction into the `.js` + `SKILL.md`. **Do not take a GPU action by hand.** This is live
(the lab is idle and ready), not a practice handoff.

---

## Vibe snippets (verbatim)
- "the role I see for you is the workflow master. you run and tune the workflow, and the
  workflow does all of the work."
- "there's a sliding derby concept already that serializes work. there should be - by
  construction - no need to concern yourself with the actual goings on of the GPU."
- "not worried about what you did wrong, buddy. promise you that. I sent the ship on the wrong
  path and now you're being very helpful getting it righted again."
- "learn and grow and write it down in the wiki so things continue to compound. if, at the end
  of this session, you are just stuck, THATS a finding I'm happy to see."

## Least confident survived (patch these by hand)
1. **The reading-list order is my judgment**, not Jason's. He asked me to tie the docs
   together "just enough." If the order feels off, `wiki/index.md` is the real doorway — trust
   it over my sequence.
2. **I deliberately did NOT restate the session→workflow→derby model** in my own words, even
   though this session located it precisely — because Jason explicitly wants the fresh session
   to read it from the docs, not inherit a previous instance's framing. The located version is
   in this session's final turns if ever needed, but the **docs are canonical.**
3. **`project_derby_operating_model.md` may actively re-tangle you** — it's flagged "READ
   FIRST" but predates the model. I chose to *point at the conflict* rather than edit the
   memory myself (substrate surgery felt like Jason's call mid-reset). A likely clean
   follow-up: reconcile/retire that memory against `workflow-orchestration.md`.
4. **"Friction smoother" mechanics** (friction log in SKILL.md vs wiki vs tuning the `.js`)
   are written down; I'm trusting you to read them rather than re-stating them here.
