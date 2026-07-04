# Worktree & branch hygiene

> ✅ **LIVE — cleanup is MANUAL (the auto-janitor is RETIRED, 2026-07-01).**
> Before removing any worktree, `ps aux | grep <path>` for live tenants. The
> `reclaim_worktrees.py` / `session_janitor.sh` auto-janitor was removed after it
> reclaimed a worktree a live training run was executing from; its full retired
> design is preserved verbatim in
> [_archive/topics/worktree-janitor-retired.md](../_archive/topics/worktree-janitor-retired.md).

The overall branch/worktree workflow lives in
[branch-and-worktree-workflow.md](branch-and-worktree-workflow.md); this page is
its **cleanup half**.

## Manual cleanup (the current rule)

Leaked worktrees/branches from crashed sessions are cleaned **by hand**.

- **Before any `git worktree remove`: `ps aux | grep <path>`.** A clean,
  fully-merged tree can still be somebody's cwd/venv or the working directory of
  a live training run. "Clean + merged" says **nothing** about live processes —
  this is exactly what bit the retired janitor (below).
- Remove a merged, idle worktree with `git worktree remove <path>` then
  `git branch -d <name>` (`-d` refuses an unmerged branch as a safety net;
  investigate before `-D`).
- Leave alone: the main checkout, **live-PID** worktrees, **external-tool**
  worktrees (`~/.codex/...`), and any manual sibling worktree still in use.
- Recovery: merged tips live in `main`; the reflog holds deleted branch tips for
  the gc window; agent worktrees re-create on demand. Nothing here is
  unrecoverable — but a live run losing its code/venv mid-flight is not worth the
  tidy.

## Why worktrees leak (the mess generator)

Claude Code's `isolation: worktree` agents (the lab's fan-out mode, see
[research-lab-charter.md](research-lab-charter.md) and the
`gomoku-research-lab` skill) create a worktree under
`.claude/worktrees/agent-<hex>` and **lock it to the spawning session's PID**.
The harness removes that worktree only on **graceful session exit**.

Our regime is the opposite of graceful: overnight autonomous derbies that get
killed, crash, or OOM. Every ungraceful exit leaks one
`.claude/worktrees/agent-<hex>` dir, a stale lock pointing at a now-dead PID, and
a `feat/*` or `worktree-agent-<hex>` branch. None of these ever caused a *bad
moment* — a session just ended. The cost was only legible in aggregate, days
later ("this repo is messy"). That is the signature of **slow entropy**. (The
2026-05-25 audit found the repo had silently grown to **26 worktrees / 57
branches**, 21 of them orphaned by three dead sessions.)

## Per-worktree uv envs (kills the editable-install gotcha by construction)

**The trap (cost ~1h on 2026-06-24, #87).** The repo ships a PEP 660 editable
install whose finder maps `gomoku` to ONE physical dir. With a single shared
`.venv`, a worktree never had its own code installed: `source /…/gomoku/.venv/bin/
activate` from a worktree silently ran the **main** checkout. The finder is
*appended* to `sys.meta_path`, so the symptom is invisible — `import gomoku`
resolves to whatever `gomoku/` is first on `sys.path`, which for `python
scripts/x.py` (sys.path[0] = `scripts/`) or `pytest` (entry in `.venv/bin`) is
main, not your worktree. You get a real object off the wrong source tree — e.g. a
method a branch *added* is simply absent. A `sitecustomize` shim that patched
`sys.path` at startup was prototyped and **rejected**: papering over the wrong
layer with startup magic.

**The fix = isolation by construction.** Every worktree gets its OWN uv-managed
`.venv` with `gomoku` editable-installed → *itself*. On APFS uv clones from its
cache, so this is ~1–4 s and ~0 incremental disk (measured). `scripts/
worktree_session.py add` (and therefore `gh_worktree.py`) runs `uv sync --extra
dev` at creation; `--no-venv` skips it. `uv.lock` is committed for reproducible
envs.

**Access via `uv run`, never activate.** `uv run <cmd>` finds the project root
from the cwd and uses that worktree's `.venv` — so it is *impossible* to silently
hit main. `uv run pytest`, `uv run python scripts/x.py`, `uv run gomoku-train …`.
There is no activation state to get wrong; the wall is real, not remembered.

**Still on the shared venv (Phase 2, a quiet-window migration):** the live fleet —
the launchd autolab agents, `gomoku/lab/up.py` (hardcoded `VENV_PY`), and the
GPU-runner scripts (`sliding_derby_runner.sh`, `train_workhorse.sh`) — is pinned
to main's `.venv` and untouched until nothing is training.

## Branch / worktree taxonomy (so future audits know who made what)

| Pattern | Created by | Lifecycle owner |
|---|---|---|
| `.claude/worktrees/agent-<hex>` | Claude Code `isolation: worktree` | harness on graceful exit; else removed **by hand** (was the janitor's job) |
| `feat/*` branches | lab fan-out / manual lanes | merge `--no-ff` then `git branch -d` **by hand** |
| `worktree-agent-<hex>` branches | harness auto-name for a worktree | removed by hand (empty scratch: safe to drop; with commits: check reflog hash first) |
| `~/code/gomoku-perf-*` siblings | manual `git worktree add` | remove by hand when the lane closes |
| `~/.codex/worktrees/*` | the `codex` CLI (external tool) | codex owns it — leave it alone |

## The janitor, retired (2026-07-01)

For ~5 weeks (2026-05-25 → 2026-07-01) a **liveness-aware auto-janitor**,
`scripts/reclaim_worktrees.py`, ran at session start (wired via
`scripts/session_janitor.sh` into the `SessionStart` + `PreCompact` hooks, #48).
It reclaimed dead-PID agent worktrees + fully-merged branches and emitted a
`--gauge` hygiene metric for the cron narrator.

It was **retired 2026-07-01** (Jason): it reclaimed
`~/code/gomoku-sound-world-run`, a "clean + fully-merged" sibling worktree that a
**live training run** (trainer + 4 self-play workers) was executing from. Its
liveness check covered Claude-session PID *locks*, not arbitrary processes
running out of a tree, so "clean + merged" passed while the run was one lazy
import away from crashing. Verdict: not as safe as believed, value minimal.
Cleanup is manual again (see the top of this page).

The full retired design — the reclaim/gauge/wiring internals, the 15s gauge
budget, and the 2026-05-25 forensic snapshot — is preserved verbatim in
[_archive/topics/worktree-janitor-retired.md](../_archive/topics/worktree-janitor-retired.md).

## The standing rule (generalizes beyond worktrees — with the janitor caveat)

**For every class of artifact the lab creates — worktrees, branches,
checkpoints, buffers, sweep_logs, crons — a gauge that *counts* it makes slow
entropy visible the day it grows, not the week someone notices.** When a
friction-log fix is a *procedure* ("remember to run X after Y"), that is a smell:
procedures decay and fail silently exactly when a session crashes.

**But the janitor half of that rule earned a scar.** An auto-*reclaimer* acts on
liveness it may not fully observe — the retired worktree janitor deleted a live
run's tree because "clean + merged" is not "nobody is using it." So: **gauges
(read-only counters) compound safely; auto-reclaimers that delete on an
inferred-idle signal do not** unless the idle signal is airtight (and a PID-lock
check is not, for processes running out of the tree). Prefer a gauge that flags +
a human/scoped-manual reclaim over an unattended deleter. See the friction-log
meta-rule in the `gomoku-research-lab` skill.
