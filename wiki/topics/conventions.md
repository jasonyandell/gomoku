# Lab Conventions

Durable cross-cutting conventions for the gomoku project. Cross-referenced
from the memory system so any session (current Claude, fresh Claude, other
agent, human reader) operates on the same rules without re-deriving them.

**The wiki is the source of truth.** Memories are personalized summaries
that point back here.

## Autonomy boundaries are a deny-list

Default-allow. Only ask for confirmation on items in the Manual-confirm
column. The "Autonomous" column is illustrative; the "Manual confirm"
column is exhaustive.

### Risk classes

| Class | Examples | Policy |
|---|---|---|
| **A** — local, reversible | files under `scripts/`, `tests/`, `wiki/`; worktrees + merge-commits on `feat/*`; per-cell artifact dirs under `sweep_logs/`; anything that lives only on disk | **Just do it. No size limit.** |
| **B** — hard to reverse / affects shared state | git push, wandb writes, archive mutations, `pyproject`/CI/deps, settings.json | **Confirm with the human.** |
| **C** — architectural / multi-day | custom Metal kernels, native C extensions, model architecture changes | **Discuss before starting.** |

### Anti-patterns

- Treating "substantial new code surface" as if it were Class B/C. Code
  size is not a permission gate; reversibility is.
- Confusing **timing/context** with **permission**. A cron tick is the
  wrong *context* for a 100-LOC task, but the charter still *permits*
  the code. Right move: surface the task and wait for a live session,
  or spawn an Agent to do the code in parallel. Wrong move: say "needs
  human go" when really you mean "wrong place to write code right now."
- Requiring explicit inclusion in the "Autonomous" column to act.

Mirrored in memory: [[feedback-autonomy-denylist]].

### Autonomous loops: triage stops, don't blanket-halt

For autonomous loops (the perf lab is the canonical example), every stop
condition is one of three actions, not a blanket halt:

- **CONTINUE** — loop self-handles via documented protocol.
- **ESCALATE** — one-line PushNotification, pause loop.
- **HALT** — clean session-end + log entry.

Default to CONTINUE. The cost of one wasted cell is small; the cost of
pre-emptive HALT can be a missed +97% headline (the 2026-05-23 perf
session would have halted at 3 consecutive rejects, immediately before
the L06-followup fp16-eval lane that nearly doubled R-S400).

See [research-lab-charter.md § Stop gates and escalation protocol](research-lab-charter.md#stop-gates-and-escalation-protocol)
for the canonical 12-row triage matrix. Other autonomous-loop scopes
should write their own matrices in the same shape.

Mirrored in memory: [[feedback-lab-runs-forever]].

## Branch integration: merge-commit, never rebase

Every feature/perf/experiment branch lands on `main` via
`git merge --no-ff <branch> -m "..."` to produce an explicit merge
commit. Never `git rebase`. Never rely on fast-forward. Never squash.

After merge: `git worktree remove <path>` then `git branch -d <name>`.
A losing experiment: same cleanup, no rebase to "preserve" it.

**Why:** Jason's words at lab kickoff — "embrace [worktrees], and merge
commit them, it's safe every time and rebase bores me." Safety driven
(merge commits preserve the experiment as a discrete event in `git log
--graph`) and aesthetic (rebasing is tedious).

Mirrored in memory: [[feedback-merge-commits]].

## Memories also go to the wiki

When saving a memory under
`~/.claude/projects/-Users-jason-code-gomoku/memory/`, **also write or
update a corresponding wiki section** in the project repo. The memory
points back to the wiki; the wiki is the canonical source.

**Why:** memories are agent-local and ephemeral — they don't survive
host migrations, aren't visible to other agents, aren't reviewable by
humans, aren't version-controlled. The wiki is project-durable.

**How:**
- Save the memory normally (slug, description, frontmatter, body).
- Add or update a section in `wiki/topics/conventions.md` (cross-cutting
  rules) or a more specific topic page (lab-specific rules → charter
  or session-runbook).
- The memory's `description` field should include a one-line pointer
  to the wiki section.
- The wiki section should include a "Mirrored in memory: [[slug]]"
  footer so a wiki reader knows there's a personalized index.

**Don't** save a memory without a wiki mirror unless it's strictly
personal-to-current-Claude (rare; usually you want the rule durable).

Mirrored in memory: [[feedback-memories-to-wiki]].

## Estimate in Opus-minutes, not human-days

A "1-2 day refactor" in human-developer-week terms is usually 20-60 min
of Opus time. Don't pad estimates by human assumptions about typing
speed, context-switching, meeting overhead, or fear of unfamiliarity.

This matters because **bad estimates change priorities**. If task A
looks like "20 minutes" and task B looks like "2 days," B gets
deferred. If they're actually 20 min and 60 min, the priority order
flips entirely.

**Rule of thumb:** if past-Claude estimated "1-2 days" for a code task
in this repo, the right re-estimate is "1-2 hours." If past-Claude said
"1-2 hours," it's usually "20-60 min."

**Why:** Jason 2026-05-23 — "coding '1-2 days' of Opus developer time
estimate is likely 20 minutes. not 1-2 days. but if you look at tasks
as 'this takes days and that takes minutes' then one of them sure looks
faster! so basically coding time is approved, all the way. code away."

Mirrored in memory: [[feedback-lab-scheduler]] (this rule lives there
alongside the broader scheduler philosophy).

## Cross-refs

- [research-lab-charter](research-lab-charter.md) — the lab-specific charter
  that builds on these conventions.
- [research-lab-session-runbook](research-lab-session-runbook.md) — per-session
  mechanics.
- [research-lab-reviewer-role](research-lab-reviewer-role.md) — review process
  for lane completions.
