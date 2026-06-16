# Worktree & branch hygiene — the janitor, not a procedure

**Status:** canonical (the janitor's deep-dive). The overall branch/worktree
workflow lives in
[branch-and-worktree-workflow.md](branch-and-worktree-workflow.md); this page
is its cleanup half.
Established 2026-05-25 after the repo had silently grown to **26 worktrees /
57 branches**.

## The mess generator

Claude Code's `isolation: worktree` agents (the lab's fan-out mode, see
[research-lab-charter.md](research-lab-charter.md) and the
`gomoku-research-lab` skill) create a worktree under
`.claude/worktrees/agent-<hex>` and **lock it to the spawning session's
PID**. The harness removes that worktree only on **graceful session exit**.

Our regime is the opposite of graceful: overnight autonomous derbies that
get killed, crash, or OOM. Every ungraceful exit leaks:

- one `.claude/worktrees/agent-<hex>` dir,
- a stale lock pointing at a now-dead PID,
- and a `feat/*` or `worktree-agent-<hex>` branch.

None of these ever caused a *bad moment* — a session just ended. The cost
was only legible in aggregate, days later ("this repo is messy"). That is
the signature of **slow entropy**, the class of problem our narrated
friction-log sensor is structurally blind to.

### Forensic snapshot (2026-05-25)

Of 26 agent worktrees, lock-PID liveness explained everything:

| Lock PID | State | Worktrees | Disposition |
|---|---|---|---|
| 67879 | dead | 13 | reclaim |
| 49138 | dead | 5 | reclaim |
| 56258 | dead | 3 | reclaim |
| 11684 | **alive** (live derby session) | 4 | keep |
| 37912 | **alive** (bg-spare) | 1 | keep |

21 of 26 were orphaned by three dead sessions. The cleanup took the repo to
**15 worktrees / 17 branches** with zero live work disturbed.

## The fix: `scripts/reclaim_worktrees.py`

A **liveness-aware janitor**, not a remembered procedure. It runs at session
**start** (the robust trigger — session-end runs die with the session):

```
python scripts/reclaim_worktrees.py            # dry-run preview (default, safe)
python scripts/reclaim_worktrees.py --apply     # reclaim
python scripts/reclaim_worktrees.py --gauge      # one-line hygiene metric for the narrator
python scripts/reclaim_worktrees.py --apply --include-scratch  # also drop superseded worktree-agent-* branches
```

What it reclaims (and only this):
- agent worktrees under `.claude/worktrees/` whose **lock PID is dead**;
- local branches **fully merged into main** (`git branch -d`, which refuses
  unmerged as a hard safety net);
- `worktree-agent-*` branches with **zero original (non-merge) commits**
  (pure scratch — nothing to lose).

What it never touches:
- the main checkout, **live-PID** worktrees, **external-tool** worktrees
  (`~/.codex/...`), manual sibling worktrees;
- any `feat/*` branch with unmerged work, or `worktree-agent-*` branches
  that carry real commits (those need `--include-scratch`, and it prints
  the reflog-recoverable hash first).

It is safe to run while the derby or other sessions are live — the PID
check protects them. Everything it removes is recoverable (merged tips live
in main; reflog holds deleted tips for the gc window; worktrees re-create).

## The gauge

`--gauge` emits one line for the cron narrator:

```
repo-hygiene: worktrees=15 (orphaned=0) branches=17 (merged-undeleted=0) — clean
```

`orphaned>0` or `merged-undeleted>3` flips it to `⚠ run reclaim_worktrees
--apply`. This is what makes slow entropy visible **the day it grows**
instead of the week someone notices.

### Gauge has a built-in 15s budget (derby-o3s)

The gauge runs from a cron narrator, so it must be fast **and**
unblockable. Every git subprocess in `reclaim_worktrees.py` is invoked
with `stdin=subprocess.DEVNULL`, `GIT_TERMINAL_PROMPT=0` /
`GIT_ASKPASS=/bin/true`, `--no-pager`, and a per-call `timeout` (8s). The
gauge path wraps that with a **15-second hard wall budget**: if any
subprocess times out, the gauge still emits the metric line with a
`[partial: <which call> timed out at Ns]` suffix and a `⚠` flag so the
narrator never silently wedges. Bullet-proof I/O closes the four blocking
surfaces — inherited stdin, credential prompt, implicit pager, stale lock
— that wedged the gauge ~2min on 2026-05-28 (derby-o3s). The metric
definition itself is unchanged; only the I/O is hardened.

## The standing rule (generalizes beyond worktrees)

**For every class of artifact the lab creates — worktrees, branches,
checkpoints, buffers, sweep_logs, crons — there should be a janitor that
reclaims it and a gauge that counts it.** When you add a creator, add its
janitor + gauge in the same change.

And when a friction-log fix is a *procedure* ("remember to run X after Y"),
treat that as a smell and upgrade it to janitor + gauge. Procedures decay
and fail silently exactly when a session crashes; janitors + gauges
compound. See the friction-log meta-rule in the `gomoku-research-lab` skill.

## Branch / worktree taxonomy (so future audits know who made what)

| Pattern | Created by | Lifecycle owner |
|---|---|---|
| `.claude/worktrees/agent-<hex>` | Claude Code `isolation: worktree` | harness on graceful exit; else the janitor |
| `feat/*` branches | lab fan-out / manual lanes | merge `--no-ff` then `git branch -d` (janitor sweeps merged) |
| `worktree-agent-<hex>` branches | harness auto-name for a worktree | janitor (empty) / `--include-scratch` (with commits) |
| `~/code/gomoku-perf-*` siblings | manual `git worktree add` | remove by hand when the lane closes |
| `~/.codex/worktrees/*` | the `codex` CLI (external tool) | codex owns it — the janitor leaves it alone |
