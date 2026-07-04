# Worktree janitor (retired) — `reclaim_worktrees.py` + `session_janitor.sh`

**Status: RETIRED / ARCHIVE — full-fidelity extract, 2026-07-04.** This is the
complete design of the liveness-aware worktree janitor, moved verbatim out of
[worktree-hygiene.md](../../topics/worktree-hygiene.md) during the 2026-07-04
wiki curation. The janitor was **retired 2026-07-01** (Jason) after it reclaimed
`~/code/gomoku-sound-world-run` — a "clean + fully-merged" sibling worktree that
a LIVE training run (trainer + 4 self-play workers) was executing from. The
procedural sections were relocated here (rather than left on the live hygiene
page) because reading a deleted tool's `--apply` commands in present-tense voice
is itself a hazard — it invites re-summoning a tool that no longer exists and was
retired as unsafe. Cleanup is manual again; see the live page for the current
rules. **No facts were changed — only relocated.**

---

## Why it was retired (2026-07-01)

`reclaim_worktrees.py` and `session_janitor.sh` were removed on the #110 branch
after the janitor reclaimed `~/code/gomoku-sound-world-run` — a "clean +
fully-merged" sibling worktree that a LIVE training run (trainer + 4 self-play
workers) was executing from. Its liveness check covered Claude-session PID
locks, not *arbitrary processes running out of the tree*; "clean + merged" says
nothing about live processes, and the venv/code deletion left the run one lazy
import away from crashing. Jason's verdict: not as safe as believed, value
minimal. The incident run survived — the path was recreated as a detached
worktree at main + `uv sync` before anything crashed.

## The mess it was built to sweep

Established 2026-05-25 after the repo had silently grown to **26 worktrees /
57 branches**.

Claude Code's `isolation: worktree` agents (the lab's fan-out mode) create a
worktree under `.claude/worktrees/agent-<hex>` and **lock it to the spawning
session's PID**. The harness removes that worktree only on **graceful session
exit**. Our regime is the opposite of graceful: overnight autonomous derbies
that get killed, crash, or OOM. Every ungraceful exit leaks:

- one `.claude/worktrees/agent-<hex>` dir,
- a stale lock pointing at a now-dead PID,
- and a `feat/*` or `worktree-agent-<hex>` branch.

None of these ever caused a *bad moment* — a session just ended. The cost was
only legible in aggregate, days later ("this repo is messy"). That is the
signature of **slow entropy**, the class of problem our narrated friction-log
sensor is structurally blind to.

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

A **liveness-aware janitor**, not a remembered procedure. It ran at session
**start** (the robust trigger — session-end runs die with the session). It was
**wired** into the `SessionStart` + `PreCompact` hooks via the
`scripts/session_janitor.sh` wrapper (issue #48 — before that the doctrine was
documented but unwired, so the auto-janitor never fired and merged branches +
stale siblings piled up across compactions). See *Wiring* below.

The raw CLI:

```
python scripts/reclaim_worktrees.py            # dry-run preview (default, safe)
python scripts/reclaim_worktrees.py --apply     # reclaim
python scripts/reclaim_worktrees.py --gauge      # one-line hygiene metric for the narrator
python scripts/reclaim_worktrees.py --apply --include-scratch  # also drop superseded worktree-agent-* branches
```

What it reclaimed (and only this):
- agent worktrees under `.claude/worktrees/` whose **lock PID is dead**;
- local branches **fully merged into main** (`git branch -d`, which refuses
  unmerged as a hard safety net);
- `worktree-agent-*` branches with **zero original (non-merge) commits**
  (pure scratch — nothing to lose).

What it never touched:
- the main checkout, **live-PID** worktrees, **external-tool** worktrees
  (`~/.codex/...`), manual sibling worktrees;
- any `feat/*` branch with unmerged work, or `worktree-agent-*` branches
  that carry real commits (those need `--include-scratch`, and it prints
  the reflog-recoverable hash first).

It was believed safe to run while the derby or other sessions are live — the PID
check protected them. Everything it removed was recoverable (merged tips live
in main; reflog holds deleted tips for the gc window; worktrees re-create).
**The 2026-07-01 incident showed this belief was wrong for processes running out
of a tree that were not Claude-session PID locks.**

## The gauge

`--gauge` emitted one line for the cron narrator:

```
repo-hygiene: worktrees=15 (orphaned=0) branches=17 (merged-undeleted=0) — clean
```

`orphaned>0` or `merged-undeleted>3` flipped it to `⚠ run reclaim_worktrees
--apply`. This is what made slow entropy visible **the day it grows** instead of
the week someone notices.

### Gauge had a built-in 15s budget (derby-o3s)

The gauge ran from a cron narrator, so it had to be fast **and** unblockable.
Every git subprocess in `reclaim_worktrees.py` was invoked with
`stdin=subprocess.DEVNULL`, `GIT_TERMINAL_PROMPT=0` / `GIT_ASKPASS=/bin/true`,
`--no-pager`, and a per-call `timeout` (8s). The gauge path wrapped that with a
**15-second hard wall budget**: if any subprocess timed out, the gauge still
emitted the metric line with a `[partial: <which call> timed out at Ns]` suffix
and a `⚠` flag so the narrator never silently wedged. Bullet-proof I/O closed
the four blocking surfaces — inherited stdin, credential prompt, implicit pager,
stale lock — that wedged the gauge ~2min on 2026-05-28 (derby-o3s). The metric
definition itself was unchanged; only the I/O was hardened.

## Wiring (issue #48): `scripts/session_janitor.sh` in the hooks

The doctrine "*the janitor runs at session start*" only held if something
**called** it. That something was `scripts/session_janitor.sh`, a small wrapper
mirroring `gh_prime.sh`: it `cd`s to `$CLAUDE_PROJECT_DIR` (falling back to its
own repo root), picks `.venv/bin/python` if present else `python3`, runs
`reclaim_worktrees.py --apply` then prints the `--gauge` line — and was
**robust + non-fatal by design** (no `set -e`; every line `|| true`; always
`exit 0`) so a janitor hiccup could never block session start. A failing
interpreter still emitted a `repo-hygiene: gauge unavailable …` sentinel.

It was wired as a **second command** alongside `gh_prime.sh` in **both** the
`SessionStart` and `PreCompact` hook arrays of `.claude/settings.json`.
`PreCompact` mattered specifically: compaction-resume re-fires `SessionStart`
with `source=compact`, and that is the case that bit the workflow-master
session (4 merged-undeleted branches + stale siblings accumulated across a
compaction because nothing re-ran the janitor).

**Caveat — `.claude/settings.json` is gitignored** (`.claude/*` allows only
`skills/` + `workflows/`). It was machine-local and did not ride a merge. The
*tracked, mergeable* deliverable was the wrapper + its test
(`tests/test_session_janitor.py`); the settings edit (add the
`session_janitor.sh` command to both arrays) had to be applied to the live
main-repo file by hand once. The test asserted the wiring **iff** that file is
present (it `skip`s in fresh checkouts/worktrees where it is absent).
