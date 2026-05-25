# Branch & worktree workflow (canonical)

**This is the canonical workflow for every change to this repo.** All other
mentions — [conventions.md](conventions.md) § Branch integration,
[research-lab-charter.md](research-lab-charter.md) § Operating loop, the
`gomoku-research-lab` / `gomoku-train` skills, [AGENTS.md](../../AGENTS.md),
[worktree-hygiene.md](worktree-hygiene.md) — defer to this page. Established
2026-05-25 after a session accumulated edits in the shared `main` checkout
while another session was writing to it, and couldn't merge cleanly. The
workflow below makes that failure structurally impossible.

## The rule, in one line

**Every unit of work happens in its own git worktree off `main`, lands via
`git merge --no-ff`, and the worktree + branch are removed afterward. You do
not edit files in the shared `main` checkout.**

`main` is a shared *integration point*, not a workbench. At any moment the
overnight derby, the user's IDE, and one or more agent sessions may all have
the `main` checkout open. If you scribble there you will (a) entangle your
diff with theirs and (b) be unable to switch branches without hijacking
their HEAD. Your own worktree is private — concurrent writers to `main`
cannot touch it, and your `--no-ff` merge stays clean.

## The lifecycle

```bash
# 1. Branch off current main into a private worktree (siblings of the repo).
git worktree add ~/code/gomoku-<slug> -b feat/<slug>      # feat/perf-<lane> for lab lanes
cd ~/code/gomoku-<slug>

# 2. (Agent / stale-base case) pin to current local main, then work.
git merge --no-ff main        # NEVER rebase; isolation:worktree can branch from a stale commit
#    ...edit, run, commit in small coherent steps...
git add -p && git commit -m "<what changed>"

# 3. Integrate from the main checkout — fast-forward is forbidden, so this is
#    always an explicit merge commit. Run it from the primary checkout on main.
cd ~/code/gomoku
git merge --no-ff feat/<slug> -m "<lane/area>: <one-line summary>"

# 4. Tear down. Merged → -d (refuses if somehow unmerged: investigate, don't -D blindly).
git worktree remove ~/code/gomoku-<slug>
git branch -d feat/<slug>
#    Rejected experiment: same teardown, force the branch:
#    git worktree remove ~/code/gomoku-<slug> && git branch -D feat/<slug>
```

**Never:** `git rebase`, fast-forward merges, `git squash`, or editing files
directly in the shared `main` checkout for anything beyond a trivial,
immediately-committed fix. The merge commit is the point — it preserves each
experiment as a discrete event in `git log --graph` ([conventions.md](conventions.md)).

## The one exception: code-free measurement

A lane that only *runs* things (a sweep / cell with **no file edits**) may run
from `main` — see [research-lab-charter.md](research-lab-charter.md) operating
loop, the `run_cells(main, …)` branch. Even then: leave no uncommitted edits
behind. The moment a lane needs a file change, it gets a worktree.

## Naming & locations

| Thing | Pattern | Who makes it |
|---|---|---|
| Manual / lane worktree | `~/code/gomoku-<slug>` (sibling of repo) | you, `git worktree add` |
| Lane branch | `feat/<slug>` (`feat/perf-<lane>` for perf lanes) | you |
| Agent worktree | `.claude/worktrees/agent-<hex>` (locked to session PID) | Claude Code `isolation: worktree` |
| Agent scratch branch | `worktree-agent-<hex>` (never meant to persist) | the harness |
| External-tool worktree | `~/.codex/worktrees/*` | the `codex` CLI — leave it alone |

## Cleanup is a janitor, not a procedure

Step 4 above is the happy path. But it runs at session *end*, so it fails
exactly when a session crashes / is killed (the overnight regime) — leaking a
worktree + dead-PID lock + branch every time. The backstop is the
liveness-aware janitor, run at session **start**:

```bash
python scripts/reclaim_worktrees.py            # dry-run preview
python scripts/reclaim_worktrees.py --apply     # reclaim dead-PID worktrees + merged/empty branches
python scripts/reclaim_worktrees.py --gauge      # one-line repo-hygiene metric for the narrator
```

It is safe to run while other sessions / the derby are live (it skips
live-PID worktrees, external worktrees, and unmerged `feat/*`). Full mechanics
and forensics: [worktree-hygiene.md](worktree-hygiene.md). The general rule
this instantiates — *for every artifact-creator, a janitor + a gauge, not a
remembered procedure* — is in [conventions.md](conventions.md) and memory
[[feedback-janitor-not-procedure]].

## Variations (all are the lifecycle above, specialized)

- **Lab multi-agent fan-out:** each code/doc agent gets `isolation: worktree`,
  merges `main` first, commits to `feat/perf-<lane>`, and does **not** merge —
  the orchestrator merges each branch `--no-ff` serially. See the
  `gomoku-research-lab` skill § Fan-out orchestration.
- **Training slice:** a GPU-required worktree lane running
  `run_sweep --max-wall-secs --final-eval`; integrates the same way.
- **Single manual change (incl. repo-meta / docs work like this page):** one
  worktree, one branch, merge `--no-ff`, tear down. There is no "it's just
  docs, I'll edit main directly" exemption — that is precisely how the
  shared-checkout collision happens.
