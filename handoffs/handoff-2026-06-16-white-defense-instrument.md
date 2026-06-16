# Handoff — #45 white-defense instrument built+gated; #49→#43 race teed up (2026-06-16)

## Goal & current status
This session ran the **workflow-master** operating model (run/tune workflows, stay lean, let
the workflows do the work — the recovery from the prior session's ~400k context-wall). It
**built, validated, and merged the #45 white-defense eval suite** (the instrument the defense
research needs), and the instrument's first measurement **sharpened the defense hypothesis**.
main @ `da58a67`, all pushed, gauge clean, GPU idle. No work in flight. Clean stop.

## Decisions made + rationale
- **Workflow-master role (Jason's mandate this session):** "you run and tune the workflow, and
  the workflow does all of the work." I author/kick workflows + hold only the decision thread;
  context-heavy work is fanned out. Codified in `wiki/topics/workflow-orchestration.md` §
  orchestrator discipline + the new `cockpit-vs-autopilot.md`.
- **The cockpit split for "needs-measurement" issues:** a workflow builds the CODE (CPU,
  unit-tested, worktree-isolated, reviewer re-runs the checks) → merge-ready branch; the
  **orchestrator runs the GPU verify-gate before merge** (the science gate is the operator's
  call, like the serial merge). Keeps build workflows GPU-free. This is now a proven, reusable
  pattern (`.claude/workflows/build-issue-45-white-defense.js`, registered as a skill).
- **#45 was NOT pure-code** — an `implement-backlog` run correctly skipped all 15 ready issues
  (backlog drained of pure-code work; rest is `needs-live-validation`). #45's DoD needs a
  measurement (the positive control), so I drove it as build-plus-gate, not a fire-and-forget drain.
- **The weak-arm of the positive control = a random-init net** (not the bigbuf seed — it's
  warm-started and clusters with eval502, so not cleanly weak). Random-init mirrors the
  champion-vs-random-40-0 validation: unambiguous, clean control.
- **REJECTED my own first framing** of the finding ("champion solid → contradicts the brittle-
  defender hypothesis"): `white-side-defense-plan.md` Step A (2026-06-15) ALREADY had the
  depth-4(~90%)-vs-zetor17(0%) dissociation. The honest finding is a *quantification* + an
  instrument-design implication, not a contradiction. (Read the wiki before claiming novelty.)
- **Did NOT launch #49** (the strong-attacker instrument) this session: it's a new GPU-heavy
  fixture-mining initiative, and launching one at deepening context unsupervised is the exact
  failure mode Jason is recovering from. Teed up instead.

## Constraints & invariants (held this session)
- **GPU = single serial tenant.** Verified idle before every GPU touch; the positive-control
  agent verified `pgrep` itself. Don't compete with the derby/other tenants.
- **Gate stays calibration-immune** — the #45 `white_defense_tally` is fixture-vs-fixed-attacker
  + Wilson CI, no absolute Elo (the panel calibration is broken, #35).
- **Worktree-import friction (recurring):** a worktree's new module isn't in the `-e` install
  (which points at MAIN); run worktree code with `PYTHONPATH=<worktree>`, and use ABSOLUTE main
  paths for gitignored artifacts (`sweep_runs/`). A "run this command" receipt from a
  build-in-worktree workflow must account for where code+artifacts live PRE-merge.
- Worktree-per-unit, `merge --no-ff`, never rebase, push once merged. Prove-then-commit workflows.

## Open questions / parked threads
- **(blocking the #43 race) #49** — the #45 v1 instrument is at the champion's FLOOR vs a weak
  attacker (3.75% loss) → no headroom to measure a #43 gain. #49 = a strong-attacker variant
  (depth-3/4 or **champion-as-attacker**) over strong-attacker-derived threats. *This is the next
  step before any defense race.* The mining (champion-as-attacker games where white loses) needs MPS.
- **(then) #43** — stamp the saving move on the POLICY head (value-only #42 FAILED — value-head
  saturation; closed). Race it on #49's instrument once built.
- **(non-blocking, human-gated) #48** — wire the janitor into the SessionStart hook (it's in NO
  hook today; doctrine says session-start). Repo-wide, flagged for Jason's eyeball.
- **(non-blocking) #47** — janitor's manual-sibling worktree coverage gap (12 stale siblings sit
  on disk; gauge undercounts them).
- **(non-blocking) wiki rot** — `fleet-management.md` is a dangling memory→wiki pointer (pattern:
  memories forward-reference pages never written; cockpit-vs-autopilot was the same, now fixed).

## Artifacts
- Code (merged): `gomoku/white_defense.py`, `scripts/white_defense_suite.py`,
  `fixtures/white_defense_15x15_v1.json`, `tests/test_white_defense.py` (#45, commit `eda147b`).
- Workflows: `.claude/workflows/build-issue-45-white-defense.js` (the build-then-gate pattern,
  now a skill). Prior: `sliding-derby-composite.js`, `implement-backlog.js`.
- Wiki/notes: `wiki/topics/cockpit-vs-autopilot.md` (NEW), `wiki/topics/workflow-orchestration.md`
  (§ workflow-master), `wiki/topics/white-side-defense-plan.md` (§1B.2 results block), TRAINING_WIKI
  2026-06-16 #45 entry, `gomoku-research-lab/SKILL.md` friction entry.
- Gate artifacts: `/tmp/wd_champ.json`, `/tmp/wd_weak.json`, `/tmp/wd_randinit_128x10.pt`.
- Issues: #45 CLOSED. Open: #49 (next instrument), #43 (the fix), #47/#48 (janitor), #46 (plateau).
- Champion: `sweep_runs/g15_128x10_bigbuf_eval502.pt` (model_config: 128 filters × 10 blocks, 17 planes).

## Next action
**Build #49 — the strong-attacker white-defense instrument** — via the same cockpit split:
a workflow mines strong-attacker-derived threat positions (champion-as-attacker / depth-4) where
white actually loses, adds a `--attacker ckpt:<path>` option to `white_defense_suite.py`, and the
orchestrator re-runs the positive control (champion should now post a measurably-above-floor
white_loss with CI hi well above 0, so a ≥0.10 #43 gain is resolvable). THEN race #43 vs the
plateau-escape #46 on it. Do NOT start #49 at deep context — it's a fresh-session initiative.

## Vibe snippets (paste verbatim)
- "the role I see for you is the workflow master. you run and tune the workflow, and the workflow
  does all of the work. does that make sense?"
- "if, at the end of this session, you are just stuck, THATS a finding I'm happy to see... one
  perfect solution right now would be great, sure, but writing it down, learning and growing?
  that's what I really want to see"
- "I just made a mistake by asking the previous session at about ~400k context length to start a
  major new initiative and am trying to recover."

## Least confident survived
1. **The "attacker-strength-gated" framing rests on cross-comparing different harnesses** (#45's
   depth-2 fixture @ 3.75% vs the older zetor17 0-6 vs depth-4 ~90%). They're consistent, but no
   single run sweeps attacker strength on the SAME fixture — #49 is partly to nail that cleanly.
   Hold the framing as well-supported-but-not-single-run-proven.
2. **Whether to launch #49 now vs. hand off** was a judgment call. Jason said "go all the way" +
   "anything else you think of," which a fresher instance might read as license to build #49 now.
   I weighted his recovery-from-overreach lesson + my context depth heavier. A fresh session should
   just build it (cleanly) — the next-action is unambiguous.
3. **Register calibration:** Jason calls me "buddy", trusts deeply, explicitly decouples "good work"
   from the result ("stuck is a finding"). Build boldly, report honestly, write the learning down —
   that IS the deliverable. This handoff was written at moderate-deep context (flagged per the skill).
