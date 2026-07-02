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
| **B** — hard to reverse / affects shared state | **force-push** or pushing shared/long-lived branches, wandb writes, archive mutations, `pyproject`/CI/deps, settings.json | **Confirm with the human.** |
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

## Branch integration: merge-commit, never rebase

The full lifecycle (worktree off `main` → `feat/<slug>` → merge `--no-ff` →
teardown, and why you never edit the shared `main` checkout) is canonical in
[branch-and-worktree-workflow.md](branch-and-worktree-workflow.md). This
section is the integration rule it builds on.

Every feature/perf/experiment branch lands on `main` via
`git merge --no-ff <branch> -m "..."` to produce an explicit merge
commit. Never `git rebase`. Never rely on fast-forward. Never squash.

After merge: **`git push`** (push `main` — *encouraged* once merged; it's a
clean fast-forward of your own work, so it's Class A, not a confirm-gated push),
then `git worktree remove <path>` and `git branch -d <name>`.
A losing experiment: same cleanup, no rebase to "preserve" it. (Only force-push
and pushes to shared/long-lived branches stay Class B — confirm first.)

The session-start janitor (`reclaim_worktrees.py`) that used to backstop
this is **retired** (2026-07-01: it removed a worktree a live training run
was executing from — see [worktree-hygiene.md](worktree-hygiene.md)).
Cleanup is manual: before `git worktree remove`, check
`ps aux | grep <path>` for live processes.

**Why:** Jason's words at lab kickoff — "embrace [worktrees], and merge
commit them, it's safe every time and rebase bores me." Safety driven
(merge commits preserve the experiment as a discrete event in `git log
--graph`) and aesthetic (rebasing is tedious).

## Fan out to preserve context

The orchestrator session's **context window is the scarcest resource** in a long
run — once it fills, the through-line of reasoning degrades and compaction can
drop detail. Treat it as a budget: delegate work that would spend context
without adding to the decision thread. Fan out to a subagent — **background
(`run_in_background: true`) when it can run async** — when:

- a search/investigation must read many files or long logs/transcripts to
  extract a small conclusion (grep sweeps, log trawls, "where is X / how does Y
  work");
- two or more **independent** tasks can run at once (dispatch them in one
  message so they run concurrently);
- a task will emit lots of intermediate tool output you won't need verbatim;
- editing work can be isolated (pair with `isolation: worktree`,
  [branch-and-worktree-workflow](branch-and-worktree-workflow.md)).

The subagent does the noisy work in **its** context and returns only the
distilled result; the orchestrator keeps the conclusion, **not** the file
dumps. Background agents notify on completion — don't poll. Reserve the main
thread for reasoning, decisions, and integration that must stay coherent.

**Anti-patterns:** reading ten files in the main thread to answer one question;
running independent subtasks serially; pulling a whole log/file into context
when a subagent could hand back the one line that matters.

This is the general form of the lab's two-queue fan-out
([research-lab-charter](research-lab-charter.md)); the `gomoku-research-lab`
skill § Fan-out orchestration is its lab-specific instantiation.

## Two native agent-instruction files

`AGENTS.md` (read by Codex / other agents) and `CLAUDE.md` (auto-loaded by
Claude Code — which reads `CLAUDE.md`, **not** `AGENTS.md`) are kept as **two
standalone, fully-duplicated native translations** of the same guidance, each in
its own voice. We do **not** use a `CLAUDE.md → @AGENTS.md` import: `@`-imports
are read unreliably by both models, so each file must stand alone.

**The cost is drift, and it is real** — we caught `AGENTS.md` silently missing
`worktree_session.py` and `--gauge` because they were added only to `CLAUDE.md`.
So the rule: **when you change a shared rule, edit BOTH files in the same
change.** Each file carries a "native twin — keep in sync" note at its top.
Claude-Code-only bits (skills, the `~/.claude` memory system) stay prominent in
`CLAUDE.md` and appear as a brief "Claude Code specifics" note in `AGENTS.md`.

## What belongs in memory vs the wiki

**The wiki is the single source of truth for everything about the project.**
The agent's persistent memory (`~/.claude/projects/-Users-jason-code-gomoku/memory/`,
auto-loaded each session) is deliberately NARROW. It holds ONLY:

- **(a) local-machine facts** — this Mac's hardware, file paths on this machine,
  keychain entries, locally-installed tools / MCP setup, the machine's
  power/thermal/contention behavior; and
- **(b) working-with-Jason facts** — who he is, his background, preferences,
  relationship/communication register, autonomy/permission boundaries, and how he
  wants to be worked with.

Everything else lives **in the wiki, and ONLY in the wiki** — the project itself,
ML/training judgment and dynamics, how to operate the lab (procedures,
conventions, scheduler, worktrees, janitor, fan-out, overnight ops), and the
roadmap (epics, derby, 15×15, fleet, white-defense). Do **not** mirror project or
process knowledge into memory. A memory that duplicates a wiki topic page is the
anti-pattern this rule exists to kill — *"memories compete with the wiki"*
(Jason, 2026-06-16).

**Why the split:** memory is agent-local, auto-loaded, and unreviewable; the wiki
is project-durable, versioned, and shared across agents (Codex too) and humans.
Auto-loading a project fact into every session both spends the context budget and
lets a stale memory silently contradict the maintained wiki. Keep memory to the
two things the wiki genuinely can't hold — *this machine* and *this person* — and
let the wiki carry the project.

**On save:** before writing a memory, ask "is this about THIS machine or THIS
person?" If no, it's a wiki edit, not a memory. A genuine machine/user memory
needs **no** wiki mirror (the wiki isn't about Jason or his Mac); a durable
project/process lesson is a wiki edit with **no** memory at all. (This sharpens
the older "every memory also gets a wiki section" rule, which over-mirrored
project content into memory.)

## Estimate in Opus-minutes, not human-days

A "1-2 day refactor" in human-developer-week terms is usually 20-60 min
of Opus time. Don't pad estimates by human assumptions about typing
speed, context-switching, meeting overhead, or fear of unfamiliarity.

This matters because **bad estimates change priorities**. If task A
looks like "20 minutes" and task B looks like "2 days," B gets
deferred. If they're actually 20 min and 60 min, the priority order
flips entirely. (This rule sits alongside the broader scheduler
philosophy in [research-lab-charter](research-lab-charter.md).)

**Rule of thumb:** if past-Claude estimated "1-2 days" for a code task
in this repo, the right re-estimate is "1-2 hours." If past-Claude said
"1-2 hours," it's usually "20-60 min."

**Why:** Jason 2026-05-23 — "coding '1-2 days' of Opus developer time
estimate is likely 20 minutes. not 1-2 days. but if you look at tasks
as 'this takes days and that takes minutes' then one of them sure looks
faster! so basically coding time is approved, all the way. code away."

## Cross-refs

- [research-lab-charter](research-lab-charter.md) — the lab-specific charter
  that builds on these conventions.
- [research-lab-session-runbook](research-lab-session-runbook.md) — per-session
  mechanics.
- [research-lab-reviewer-role](research-lab-reviewer-role.md) — review process
  for lane completions.
