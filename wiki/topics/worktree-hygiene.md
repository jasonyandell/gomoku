# Worktree & branch hygiene

**Status: the janitor is RETIRED (2026-07-01, Jason).** `reclaim_worktrees.py`
and `session_janitor.sh` were removed on the #110 branch after the janitor
reclaimed `~/code/gomoku-sound-world-run` — a "clean + fully-merged" sibling
worktree that a LIVE training run (trainer + 4 self-play workers) was
executing from. Its liveness check covered Claude-session PID locks, not
*arbitrary processes running out of the tree*; "clean + merged" says nothing
about live processes, and the venv/code deletion left the run one lazy import
away from crashing. Jason's verdict: not as safe as believed, value minimal.
**Cleanup is manual again: before `git worktree remove`, check
`ps aux | grep <path>` for live tenants.** The incident run survived — the
path was recreated as a detached worktree at main + `uv sync` before anything
crashed. The sections below are kept as history of the retired design.

The overall branch/worktree workflow lives in
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
**start** (the robust trigger — session-end runs die with the session). It is
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

## Wiring (issue #48): `scripts/session_janitor.sh` in the hooks

The doctrine "*the janitor runs at session start*" only holds if something
**calls** it. That something is `scripts/session_janitor.sh`, a small wrapper
mirroring `gh_prime.sh`: it `cd`s to `$CLAUDE_PROJECT_DIR` (falling back to its
own repo root), picks `.venv/bin/python` if present else `python3`, runs
`reclaim_worktrees.py --apply` then prints the `--gauge` line — and is
**robust + non-fatal by design** (no `set -e`; every line `|| true`; always
`exit 0`) so a janitor hiccup can never block session start. A failing
interpreter still emits a `repo-hygiene: gauge unavailable …` sentinel.

It is wired as a **second command** alongside `gh_prime.sh` in **both** the
`SessionStart` and `PreCompact` hook arrays of `.claude/settings.json`.
`PreCompact` matters specifically: compaction-resume re-fires `SessionStart`
with `source=compact`, and that is the case that bit the workflow-master
session (4 merged-undeleted branches + stale siblings accumulated across a
compaction because nothing re-ran the janitor).

**Caveat — `.claude/settings.json` is gitignored** (`.claude/*` allows only
`skills/` + `workflows/`). It is machine-local and **does not ride a merge**.
The *tracked, mergeable* deliverable is the wrapper + its test
(`tests/test_session_janitor.py`); the settings edit (add the
`session_janitor.sh` command to both arrays) must be applied to the live
main-repo file by hand once. The test asserts the wiring **iff** that file is
present (it `skip`s in fresh checkouts/worktrees where it is absent).

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
